"""Attention across the temporal axis of an interferogram sequence.

The sequence a sample carries is ``[t-k*11d, ..., t-11d, t]`` — the current
interferogram and its immediate predecessors, oldest first, with the label
belonging to the last one (``dataprep/dataset.py:42``). So attention over this
axis is by construction attention over *past* interferograms: there is no future
frame in the data to leak from.

Attention runs at the U-Net bottleneck, independently at every spatial location.
At the project's 200x100 patch that grid is 12x6, i.e. 72 positions over T <= 11
timesteps, which is why the whole block costs ~1M parameters against the
ConvLSTM cell's 11.8M.

Two readout shapes are provided:

- ``CausalTemporalAttention(layers=1)`` — the present frame is the only query
  and attends over the whole sequence. "What in my history explains what I am
  looking at now." Costs T rather than T^2, and yields exactly one weight per
  (head, timestep, pixel), which is what the skip fusion below consumes.
- ``layers > 1`` — causal self-attention blocks transform the full sequence
  first, then the same present-query readout ends it. The lower-triangular mask
  matters *here*: with a single readout layer it would be a no-op, since the
  only query is already the last position.

Positional encoding is sinusoidal over the offset ``T-1-t`` — "steps before the
present" — and carries no parameters. That is deliberate: chains are built at a
fixed 11-day spacing (``meta.py:88``), so index and elapsed time are the same
thing here, and encoding the offset rather than the absolute index keeps T out
of the architecture. A model trained at k_prevs=5 still runs at k_prevs=10, the
same contract ConvLSTMUNet holds and the ``--fallback_replicate`` eval path
relies on.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Width the temporal tokens are projected to before attending. Like the
#: ConvLSTM's hidden width, deliberately narrower than the 1024-channel
#: bottleneck: the sequence is short and the grid is small, so the useful
#: capacity is in the encoder, not here.
DEFAULT_TATTN_DIM = 256

#: Attention heads. Also the number of channel groups the skip fusion splits
#: each skip into, so every skip width (64/128/256/512) must divide by it.
DEFAULT_TATTN_HEADS = 8

#: Hidden expansion of the per-block feed-forward layer.
FFN_RATIO = 2


def temporal_position_encoding(t: int, dim: int, *, device=None,
                               dtype=torch.float32) -> torch.Tensor:
    """Sinusoidal encoding of "steps before the present", shape ``(T, dim)``.

    Row ``t`` encodes the offset ``T-1-t``, so the current interferogram is
    always at offset 0 and the oldest at ``T-1``, whatever T happens to be.
    """
    if dim % 2 != 0:
        raise ValueError(f"temporal_position_encoding needs an even dim, got {dim}.")
    # t = 0 is the oldest frame, so its offset is the largest.
    offsets = torch.arange(t - 1, -1, -1, device=device, dtype=torch.float32)
    freqs = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    angles = offsets.unsqueeze(1) * freqs.unsqueeze(0)  # (T, dim/2)
    pe = torch.zeros(t, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe.to(dtype)


def _softmax_fp32(scores: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Softmax in float32 regardless of autocast, cast back afterwards.

    Every run trains under --amp. A half-precision softmax over a short axis is
    exactly the kind of thing that degrades quietly rather than failing, and the
    tensor is small enough that the upcast costs nothing.
    """
    return torch.softmax(scores.float(), dim=-1).to(out_dtype)


def _feed_forward(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim * FFN_RATIO),
        nn.GELU(),
        nn.Linear(dim * FFN_RATIO, dim),
    )


class _CausalSelfAttentionBlock(nn.Module):
    """Pre-LN causal self-attention over T. ``(B, P, T, d) -> (B, P, T, d)``.

    Every position attends over itself and everything older, never anything
    newer. Only reachable with ``layers > 1``.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = _feed_forward(self.dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, p, t, _ = x.shape
        return x.view(b, p, t, self.heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p, t, _ = x.shape
        normed = self.norm1(x)
        q = self._heads(self.q_proj(normed))
        k = self._heads(self.k_proj(normed))
        v = self._heads(self.v_proj(normed))

        scores = torch.einsum("bpqhd,bpkhd->bphqk", q, k) * self.scale
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = _softmax_fp32(scores, v.dtype)

        attended = torch.einsum("bphqk,bpkhd->bpqhd", weights, v).reshape(b, p, t, self.dim)
        x = x + self.out_proj(attended)
        return x + self.ffn(self.norm2(x))


class _TemporalReadoutBlock(nn.Module):
    """Pre-LN cross-attention: the present token queries the whole sequence.

    ``(B, P, T, d) -> ((B, P, d), weights (B, P, heads, T))``. The weights are
    returned because the skip fusion reuses them; they sum to 1 over T.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = _feed_forward(self.dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, p, t, _ = x.shape
        normed = self.norm1(x)
        present = normed[:, :, -1]  # index T-1 is the current interferogram

        q = self.q_proj(present).view(b, p, self.heads, self.head_dim)
        k = self.k_proj(normed).view(b, p, t, self.heads, self.head_dim)
        v = self.v_proj(normed).view(b, p, t, self.heads, self.head_dim)

        scores = torch.einsum("bphd,bpkhd->bphk", q, k) * self.scale
        weights = _softmax_fp32(scores, v.dtype)  # (B, P, heads, T)

        attended = torch.einsum("bphk,bpkhd->bphd", weights, v).reshape(b, p, self.dim)
        out = x[:, :, -1] + self.out_proj(attended)
        out = out + self.ffn(self.norm2(out))
        return out, weights


class CausalTemporalAttention(nn.Module):
    """Attention over time at each spatial location of a feature sequence.

    ``forward(seq)`` takes ``(B, T, C, H, W)`` oldest-first and returns

    - ``fused``   ``(B, C, H, W)`` — the present, rewritten by its history
    - ``weights`` ``(B, heads, T, H, W)`` — how much each timestep contributed,
      summing to 1 over T. Feed these to :func:`fuse_over_time` to collapse the
      skip connections with the same reasoning.
    """

    def __init__(self, in_channels: int, dim: int = DEFAULT_TATTN_DIM,
                 heads: int = DEFAULT_TATTN_HEADS, layers: int = 1):
        super().__init__()
        dim, heads, layers = int(dim), int(heads), int(layers)
        if dim % heads != 0:
            raise ValueError(
                f"tattn_dim must divide by tattn_heads; got dim={dim}, heads={heads}."
            )
        if dim % 2 != 0:
            raise ValueError(f"tattn_dim must be even for the positional encoding; got {dim}.")
        if layers < 1:
            raise ValueError(f"tattn_layers must be at least 1; got {layers}.")

        self.in_channels = int(in_channels)
        self.dim = dim
        self.heads = heads
        self.layers = layers

        self.in_proj = nn.Conv2d(self.in_channels, dim, kernel_size=1)
        # Normalising the projected tokens *before* the positional encoding is
        # added keeps content and position on comparable scales whatever the
        # encoder's output magnitude happens to be. Without it the balance is
        # set by an accident of activation scale: at the untrained bottleneck
        # the content term is ~28x smaller than the encoding, and attention
        # degenerates to a fixed function of position that ignores the frames.
        self.in_norm = nn.LayerNorm(dim)
        # All but the last layer transform the whole sequence; the last one is
        # the present-query readout that collapses it.
        self.self_blocks = nn.ModuleList(
            _CausalSelfAttentionBlock(dim, heads) for _ in range(layers - 1)
        )
        self.readout = _TemporalReadoutBlock(dim, heads)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Conv2d(dim, self.in_channels, kernel_size=1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = seq.shape
        if c != self.in_channels:
            raise ValueError(
                f"CausalTemporalAttention was built for {self.in_channels} channels "
                f"per timestep but received {c} (input {tuple(seq.shape)})."
            )

        x = self.in_proj(seq.reshape(b * t, c, h, w))          # (B*T, d, H, W)
        # (B*T, d, H, W) -> (B, P, T, d): one token sequence per spatial location.
        x = self.in_norm(x.reshape(b, t, self.dim, h * w).permute(0, 3, 1, 2))
        x = x + temporal_position_encoding(t, self.dim, device=x.device, dtype=x.dtype)

        for block in self.self_blocks:
            x = block(x)
        fused, weights = self.readout(x)                       # (B, P, d), (B, P, heads, T)

        fused = self.norm(fused).permute(0, 2, 1).reshape(b, self.dim, h, w)
        fused = self.out_proj(fused)                           # (B, C, H, W)
        weights = weights.permute(0, 2, 3, 1).reshape(b, self.heads, t, h, w)
        return fused, weights


def fuse_over_time(skip: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Collapse a ``(B, T, C, H, W)`` skip to ``(B, C, H, W)`` using bottleneck attention.

    The U-TAE trick (Garnot & Landrieu, ICCV 2021): the attention computed once
    at the bottleneck is upsampled to each skip's resolution and used to take a
    weighted mean over time, per head-group of channels. It replaces "use the
    latest timestep and discard the rest", which is what leaves ConvLSTMUNet's
    decoder with single-frame detail at every resolution.

    Because the weights sum to 1 over T and interpolation is linear, they still
    sum to 1 after upsampling — the result is a convex combination of the
    per-timestep skips, so its scale matches the single-frame skip it replaces
    and the two fusion modes stay comparable.
    """
    b, t, c, h, w = skip.shape
    heads = weights.shape[1]
    if c % heads != 0:
        raise ValueError(
            f"skip width {c} does not divide into {heads} attention heads; "
            f"choose tattn_heads so it divides 64, 128, 256 and 512."
        )
    if weights.shape[2] != t:
        raise ValueError(
            f"attention weights cover {weights.shape[2]} timesteps but the skip has {t}."
        )

    # The cast back to the skip's dtype is load-bearing under --amp. Softmax and
    # upsample_bilinear2d are both on autocast's fp32 promotion list, and
    # fp32 * fp16 promotes rather than narrowing, so without this the upsampled
    # map, every product and the output all silently run at fp32 — doubling the
    # cost of the one part of this model that is actually memory-hungry.
    upsampled = F.interpolate(
        weights.reshape(b, heads * t, *weights.shape[-2:]),
        size=(h, w), mode="bilinear", align_corners=False,
    ).reshape(b, heads, t, h, w).to(skip.dtype)

    grouped = skip.reshape(b, t, heads, c // heads, h, w)
    # Accumulate rather than building a (B, T, C, H, W) product: broadcasting the
    # (B, heads, 1, H, W) weight against the grouped skip keeps autograd holding
    # only the small tensor, where a repeat_interleave to full width would cost
    # a copy of the skip per timestep.
    out = torch.zeros_like(grouped[:, 0])
    for step in range(t):
        out = out + upsampled[:, :, step].unsqueeze(2) * grouped[:, step]
    return out.reshape(b, c, h, w)
