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

**Sequences with holes.** The identity "index == elapsed time" holds only for a
gap-free chain, and the archive is not gap-free: the North frame is missing 12
acquisition slots and South 33, so requiring an unbroken chain is what collapses
the usable set from 273 interferograms to 20 at a 40-slot lookback. Both
entry points therefore take two optional arguments:

- ``offsets``  ``(T,)`` or ``(B, T)`` — each frame's real age in 11-day slot
  units, so a frame two slots older than its neighbour is encoded as such
  instead of as the next index along;
- ``valid``    ``(B, T)`` — which slots carry a real frame, for batches padded
  to a common length.

Both default to None, which reproduces the gap-free behaviour exactly — the
encoding carries no parameters, so every existing checkpoint keeps its numbers
and can be run on a gappy sequence without retraining.

**Why the queries and keys are built from a contrast.** Measured at the
bottleneck, the attention tokens are dominated by a component shared across the
whole sequence: at initialisation the frame-to-frame differences are 0.09% of
the token magnitude and consecutive frames have cosine similarity 1.0000. The
softmax is therefore uniform before a single gradient step, the gradient
reaching q/k is ~10x weaker than the one reaching the value path, and the
projections decay to zero long before the encoder learns to separate the frames
(it eventually does — the ratio reaches 0.50 by the end of training, far too
late). All fifteen checkpoints trained before this was found average their
history instead of selecting from it; ``docs/ATTENTION_COLLAPSE.md`` has the
measurements.

Two changes fix it, and neither is sufficient alone:

- ``contrast`` — form queries and keys from ``token - mean_over_time(token)``,
  renormalised, so attention selects on *how frames differ* rather than on what
  they share. The value path still sees the whole token.
- ``qk_norm`` — unit-norm queries and keys and scale the logits by one learned
  temperature, so the softmax's sharpness stops depending on the magnitude of
  the projections. Without it the same block collapses to uniform at lr 1e-6
  and saturates to one-hot at lr 1e-4.

Together they take effective frames used from 10.97/11 to 4.43/11 at init, and
hold it there through training where the unfixed block degenerates. Both are on
by default for new models and off for any checkpoint that predates them — the
extra parameters exist only when enabled, so old weights load untouched.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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

#: Starting temperature for QK-normalised logits, and the ceiling it may reach.
#: With unit-norm queries and keys a raw score lives in [-1, 1], so this
#: multiplier is what decides whether the softmax can be selective at all. 10
#: puts the initial logit range at about +-10 across T; the clamp is CLIP's, and
#: stops a runaway temperature driving the softmax one-hot.
DEFAULT_LOGIT_SCALE = 10.0
MAX_LOGIT_SCALE = 100.0


def temporal_mean(x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over the time axis of ``(B, P, T, d)``, ignoring padded frames.

    Averaging padded slots in would fold the replicated filler into the very
    baseline the contrast is measured against — worst for exactly the long,
    gappy sequences the mask exists to serve.
    """
    if valid is None:
        return x.mean(dim=2, keepdim=True)
    b, _, t, _ = x.shape
    m = valid.view(b, 1, t, 1).to(x.dtype)
    return (x * m).sum(dim=2, keepdim=True) / m.sum(dim=2, keepdim=True).clamp_min(1.0)


def temporal_position_encoding(t: int, dim: int, *, offsets=None, device=None,
                               dtype=torch.float32) -> torch.Tensor:
    """Sinusoidal encoding of "steps before the present".

    Without ``offsets`` the sequence is assumed gap-free: row ``t`` encodes the
    offset ``T-1-t``, so the current interferogram is always at offset 0 and the
    oldest at ``T-1``, whatever T happens to be. Shape ``(T, dim)``.

    ``offsets`` states each frame's age explicitly, in **11-day slot units**
    (offset 0 is the present, 1 is eleven days earlier). It is what makes a
    sequence with holes mean anything: on a gap-free chain the list index and
    the elapsed time are the same number, but as soon as an acquisition is
    missing, "three back in the list" may be three or five slots old, and
    encoding the index would tell the model something false. Accepts ``(T,)``
    -> ``(T, dim)``, or ``(B, T)`` -> ``(B, T, dim)`` when samples in a batch
    have different holes.

    Slot units rather than days on purpose: a dense chain then produces exactly
    the encoding this function returned before ``offsets`` existed, so trained
    checkpoints are unaffected.
    """
    if dim % 2 != 0:
        raise ValueError(f"temporal_position_encoding needs an even dim, got {dim}.")
    if offsets is None:
        # t = 0 is the oldest frame, so its offset is the largest.
        offsets = torch.arange(t - 1, -1, -1, device=device, dtype=torch.float32)
    else:
        offsets = torch.as_tensor(offsets, dtype=torch.float32,
                                  device=device if device is not None else None)
        if offsets.dim() not in (1, 2):
            raise ValueError(
                f"offsets must be (T,) or (B, T); got shape {tuple(offsets.shape)}."
            )
        if offsets.shape[-1] != t:
            raise ValueError(
                f"offsets cover {offsets.shape[-1]} timesteps but the sequence has {t}."
            )
        device = offsets.device
    freqs = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    angles = offsets.unsqueeze(-1) * freqs  # (..., T, dim/2)
    pe = torch.zeros(*offsets.shape, dim, device=device, dtype=torch.float32)
    pe[..., 0::2] = torch.sin(angles)
    pe[..., 1::2] = torch.cos(angles)
    return pe.to(dtype)


def check_valid_mask(valid, b: int, t: int) -> torch.Tensor:
    """Normalise a padding mask to a ``(B, T)`` bool tensor, or raise.

    The present frame — index ``T-1`` — must be present in every sample. It is
    the readout's only query, so a sample padded there would softmax over an
    all ``-inf`` row and return NaN for the whole batch. Padding belongs at the
    *old* end of the sequence, which is also the only end where a real chain
    runs out of history.
    """
    valid = torch.as_tensor(valid)
    if valid.dim() != 2 or valid.shape != (b, t):
        raise ValueError(
            f"valid must be a (B, T) mask matching the sequence; expected "
            f"{(b, t)}, got {tuple(valid.shape)}."
        )
    valid = valid.bool()
    if not bool(valid[:, -1].all()):
        raise ValueError(
            "the present frame (index T-1) is masked out for at least one sample. "
            "It is the attention query, so the sample has nothing to predict from; "
            "pad at the oldest end of the sequence, never at the newest."
        )
    return valid


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

    def __init__(self, dim: int, heads: int, contrast: bool = True,
                 qk_norm: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.contrast = bool(contrast)
        self.qk_norm = bool(qk_norm)

        self.norm1 = nn.LayerNorm(self.dim)
        if self.contrast:
            self.contrast_norm = nn.LayerNorm(self.dim)
        if self.qk_norm:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(DEFAULT_LOGIT_SCALE), dtype=torch.float32)
            )
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = _feed_forward(self.dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, p, t, _ = x.shape
        return x.view(b, p, t, self.heads, self.head_dim)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, p, t, _ = x.shape
        normed = self.norm1(x)
        # Same split as the readout: select on the differences, carry the content.
        selector = (self.contrast_norm(normed - temporal_mean(normed, valid))
                    if self.contrast else normed)
        q = self._heads(self.q_proj(selector))
        k = self._heads(self.k_proj(selector))
        v = self._heads(self.v_proj(normed))

        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            scores = (torch.einsum("bpqhd,bpkhd->bphqk", q, k)
                      * self.logit_scale.exp().clamp(max=MAX_LOGIT_SCALE))
        else:
            scores = torch.einsum("bpqhd,bpkhd->bphqk", q, k) * self.scale
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        allowed = causal.view(1, 1, 1, t, t)
        if valid is not None:
            # Mask the KEY axis: no position may read a padded frame.
            allowed = allowed & valid.view(b, 1, 1, 1, t)
            # A padded *query* early in the sequence can end up with no legal
            # key at all (everything at or before it is padding), and softmax
            # over an all -inf row returns NaN, which the residual would then
            # spread to every position including the present. Letting every
            # query keep its own diagonal costs nothing — the readout masks
            # these rows out again — and keeps the row finite.
            allowed = allowed | torch.eye(t, device=x.device, dtype=torch.bool).view(1, 1, 1, t, t)
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = _softmax_fp32(scores, v.dtype)

        attended = torch.einsum("bphqk,bpkhd->bpqhd", weights, v).reshape(b, p, t, self.dim)
        x = x + self.out_proj(attended)
        return x + self.ffn(self.norm2(x))


class _TemporalReadoutBlock(nn.Module):
    """Pre-LN cross-attention: the present token queries the whole sequence.

    ``(B, P, T, d) -> ((B, P, d), weights (B, P, heads, T))``. The weights are
    returned because the skip fusion reuses them; they sum to 1 over T.

    ``contrast`` and ``qk_norm`` are the two halves of the fix for the collapse
    documented in ``docs/ATTENTION_COLLAPSE.md``; see :class:`CausalTemporalAttention`
    for what each does and why neither works alone.
    """

    def __init__(self, dim: int, heads: int, contrast: bool = True,
                 qk_norm: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.contrast = bool(contrast)
        self.qk_norm = bool(qk_norm)

        self.norm1 = nn.LayerNorm(self.dim)
        # Registered only when enabled, so a checkpoint trained without them
        # loads with no unexpected keys and keeps its exact behaviour.
        if self.contrast:
            self.contrast_norm = nn.LayerNorm(self.dim)
        if self.qk_norm:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(DEFAULT_LOGIT_SCALE), dtype=torch.float32)
            )
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = _feed_forward(self.dim)

    def forward(self, x: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, p, t, _ = x.shape
        normed = self.norm1(x)

        # Queries and keys are formed from what makes the frames DIFFER; the
        # value path keeps the whole token, because the answer still has to
        # carry the content, not just the deviation.
        if self.contrast:
            selector = self.contrast_norm(normed - temporal_mean(normed, valid))
        else:
            selector = normed
        present = selector[:, :, -1]  # index T-1 is the current interferogram

        q = self.q_proj(present).view(b, p, self.heads, self.head_dim)
        k = self.k_proj(selector).view(b, p, t, self.heads, self.head_dim)
        v = self.v_proj(normed).view(b, p, t, self.heads, self.head_dim)

        if self.qk_norm:
            # Unit-norm q and k, so the logit range is set by one learned number
            # instead of drifting with q/k magnitude. That decoupling is what
            # keeps the softmax selective at any T and under any learning rate:
            # without it the same block goes uniform at lr 1e-6 and one-hot at
            # lr 1e-4, purely through the scale of the projections.
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            scores = (torch.einsum("bphd,bpkhd->bphk", q, k)
                      * self.logit_scale.exp().clamp(max=MAX_LOGIT_SCALE))
        else:
            scores = torch.einsum("bphd,bpkhd->bphk", q, k) * self.scale
        if valid is not None:
            # Padded frames get exactly zero weight, which is also what makes
            # fuse_over_time need no change at all: the skip fusion is a convex
            # combination over these same weights, so a padded timestep drops
            # out of it by arithmetic rather than by a second mask.
            scores = scores.masked_fill(~valid.view(b, 1, 1, t), float("-inf"))
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
                 heads: int = DEFAULT_TATTN_HEADS, layers: int = 1,
                 contrast: bool = True, qk_norm: bool = True):
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
        self.contrast = bool(contrast)
        self.qk_norm = bool(qk_norm)

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
            _CausalSelfAttentionBlock(dim, heads, contrast=self.contrast,
                                      qk_norm=self.qk_norm)
            for _ in range(layers - 1)
        )
        self.readout = _TemporalReadoutBlock(dim, heads, contrast=self.contrast,
                                             qk_norm=self.qk_norm)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Conv2d(dim, self.in_channels, kernel_size=1)

    def forward(self, seq: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = seq.shape
        if c != self.in_channels:
            raise ValueError(
                f"CausalTemporalAttention was built for {self.in_channels} channels "
                f"per timestep but received {c} (input {tuple(seq.shape)})."
            )
        if valid is not None:
            valid = check_valid_mask(valid, b, t).to(seq.device)

        x = self.in_proj(seq.reshape(b * t, c, h, w))          # (B*T, d, H, W)
        # (B*T, d, H, W) -> (B, P, T, d): one token sequence per spatial location.
        x = self.in_norm(x.reshape(b, t, self.dim, h * w).permute(0, 3, 1, 2))
        pe = temporal_position_encoding(t, self.dim, offsets=offsets,
                                        device=x.device, dtype=x.dtype)
        # (T, d) broadcasts over (B, P, T, d) as it stands; a per-sample (B, T, d)
        # needs the spatial axis inserted.
        x = x + (pe if pe.dim() == 2 else pe.unsqueeze(1))

        for block in self.self_blocks:
            x = block(x, valid)
        fused, weights = self.readout(x, valid)                # (B, P, d), (B, P, heads, T)

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
