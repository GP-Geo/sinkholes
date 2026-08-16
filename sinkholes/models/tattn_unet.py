"""U-Net with temporal attention over an interferogram sequence.

Where ``ConvLSTMUNet`` compresses the sequence into one recurrent hidden state,
this model lets the *current* interferogram look back at its predecessors
directly and decide, per pixel, which ones matter. Two structural changes follow
from that:

    1. a *shared* U-Net encoder runs independently on every timestep (as in
       ConvLSTMUNet),
    2. multi-head attention over the T axis, at each bottleneck location, with
       the present frame as the query and its history as keys/values,
    3. the attended present becomes the decoder input,
    4. **the skip connections are fused over time by the same attention
       weights** — instead of taking the latest timestep and discarding the
       rest, which is what leaves the ConvLSTM decoder with single-frame detail
       at every resolution,
    5. the standard U-Net decoder produces the segmentation logits.

``tattn_recurrence="convlstm"`` puts a ConvLSTM back in front of the attention
and keeps *all* its hidden states rather than only the last, so the two
mechanisms compose: recurrence integrates the sequence, attention looks things
up in it.

Chronological order is oldest -> newest, matching the dataset. Because every
frame in a sample is the current interferogram or one of its predecessors,
attention here is past-only by construction; see ``temporal_attention`` for
where the causal mask does and does not bite.

Output is raw logits ``[B, n_classes, H, W]`` — no sigmoid — exactly like
``UNet``, so the same losses and evaluation apply.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .convlstm_unet import DEFAULT_CONVLSTM_HIDDEN_CHANNELS, ConvLSTMCell
from .parts import DoubleConv, Down, OutConv, Up
from .temporal import (
    at_latest_timestep,
    check_channels_per_timestep,
    to_sequence,
    unfold_time,
)
from .temporal_attention import (
    DEFAULT_TATTN_DIM,
    DEFAULT_TATTN_HEADS,
    CausalTemporalAttention,
    fuse_over_time,
)

#: Key under which a TemporalAttentionUNet checkpoint carries the config needed
#: to rebuild the model. Stored alongside 'mask_values' and popped the same way.
CONFIG_KEY = "tattn_unet_config"

#: What sits between the encoder and the attention.
RECURRENCE_CHOICES = ("none", "convlstm")

#: How many skips can be fused over time — one per decoder stage.
MAX_FUSED_SKIPS = 4

#: Encoder skip widths, which the head count has to divide for attention fusion.
SKIP_WIDTHS = (64, 128, 256, 512)


class TemporalAttentionUNet(nn.Module):
    """U-Net whose bottleneck attends over time, past-only, with fused skips.

    As with ``ConvLSTMUNet``, the number of timesteps T is deliberately *not* an
    architectural parameter: the encoder is shared, the attention is computed
    over whatever T arrives, and the positional encoding is a function of the
    offset from the present rather than a learned table. A model trained at T=6
    runs at T=11. Consequently ``n_channels`` means channels **per timestep**.

    Accepted input shapes are exactly ``ConvLSTMUNet``'s:

    - ``[B, T, H, W]``     when ``n_channels_per_timestep == 1``
    - ``[B, 2T, H, W]``    when ``n_channels_per_timestep == 2`` — BLOCK layout
      ``[img_t0..img_t{T-1}, V_t0..V_t{T-1}]``, as the dataset concatenates it
    - ``[B, T, C, H, W]``  the canonical explicit form
    """

    #: Checkpoint key this model's ``config_dict()`` is stored under. Exposed on
    #: the class (rather than left to the caller) so the trainer can write the
    #: blob for any architecture that has one, instead of growing an isinstance
    #: chain that has to be updated in step with every new model.
    CONFIG_KEY = CONFIG_KEY

    def __init__(
        self,
        n_channels_per_timestep: int = 1,
        n_classes: int = 1,
        bilinear: bool = False,
        tattn_dim: Optional[int] = None,
        tattn_heads: int = DEFAULT_TATTN_HEADS,
        tattn_layers: int = 1,
        tattn_recurrence: str = "none",
        tattn_fuse_skips: int = MAX_FUSED_SKIPS,
        convlstm_hidden_channels: Optional[int] = None,
        convlstm_kernel_size: int = 3,
    ):
        super().__init__()
        self.n_channels_per_timestep = check_channels_per_timestep(n_channels_per_timestep)
        self.n_channels = self.n_channels_per_timestep
        self.n_classes = int(n_classes)
        self.bilinear = bool(bilinear)

        if tattn_recurrence not in RECURRENCE_CHOICES:
            raise ValueError(
                f"tattn_recurrence must be one of {RECURRENCE_CHOICES}; "
                f"got {tattn_recurrence!r}."
            )
        if not 0 <= int(tattn_fuse_skips) <= MAX_FUSED_SKIPS:
            raise ValueError(
                f"tattn_fuse_skips must be between 0 and {MAX_FUSED_SKIPS}; "
                f"got {tattn_fuse_skips!r}."
            )
        self.tattn_recurrence = tattn_recurrence
        self.tattn_fuse_skips = int(tattn_fuse_skips)

        # 0 is the CLI's spelling of "unset", as with --convlstm_hidden.
        self.tattn_dim = DEFAULT_TATTN_DIM if tattn_dim in (None, 0) else int(tattn_dim)
        self.tattn_heads = int(tattn_heads)
        self.tattn_layers = int(tattn_layers)

        if self.tattn_fuse_skips:
            bad = [c for c in SKIP_WIDTHS if c % self.tattn_heads]
            if bad:
                raise ValueError(
                    f"fusing skips splits each one into tattn_heads channel groups, "
                    f"but {self.tattn_heads} heads does not divide {bad}. Use a power "
                    f"of two up to 64, or tattn_fuse_skips=0."
                )

        # Shared encoder — identical widths to the plain U-Net.
        self.inc = DoubleConv(self.n_channels_per_timestep, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.bottleneck_channels = 1024 // factor
        self.down4 = Down(512, self.bottleneck_channels)

        # The recurrent stage is optional and named 'recurrence', NOT 'convlstm':
        # the checkpoint factory detects ConvLSTMUNet with a 'convlstm.' key
        # prefix (models/factory.py), so a cell under that name here would make a
        # hybrid checkpoint load as the wrong architecture.
        if self.tattn_recurrence == "convlstm":
            hidden = (
                DEFAULT_CONVLSTM_HIDDEN_CHANNELS
                if convlstm_hidden_channels in (None, 0)
                else int(convlstm_hidden_channels)
            )
            self.convlstm_hidden_channels = hidden
            self.convlstm_kernel_size = int(convlstm_kernel_size)
            self.recurrence: Optional[ConvLSTMCell] = ConvLSTMCell(
                input_channels=self.bottleneck_channels,
                hidden_channels=hidden,
                kernel_size=self.convlstm_kernel_size,
            )
            attended_channels = hidden
        else:
            # Recorded even when unused so config_dict() round-trips a stable shape.
            self.convlstm_hidden_channels = (
                DEFAULT_CONVLSTM_HIDDEN_CHANNELS
                if convlstm_hidden_channels in (None, 0)
                else int(convlstm_hidden_channels)
            )
            self.convlstm_kernel_size = int(convlstm_kernel_size)
            self.recurrence = None
            attended_channels = self.bottleneck_channels

        self.temporal_attn = CausalTemporalAttention(
            in_channels=attended_channels,
            dim=self.tattn_dim,
            heads=self.tattn_heads,
            layers=self.tattn_layers,
        )
        # A narrower recurrent state still has to enter up1 at the bottleneck width.
        self.recurrence_proj: nn.Module = (
            nn.Identity()
            if attended_channels == self.bottleneck_channels
            else nn.Conv2d(attended_channels, self.bottleneck_channels, kernel_size=1)
        )

        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    # -- input handling ----------------------------------------------------------------

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise any accepted input to [B, T, C, H, W], oldest -> newest."""
        return to_sequence(
            x, self.n_channels_per_timestep, model_name="TemporalAttentionUNet"
        )

    # -- forward -----------------------------------------------------------------------

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Logits plus the temporal attention weights ``(B, heads, T, Hb, Wb)``.

        The weights answer "which past interferogram did this pixel rely on",
        which is the diagnostic that separates a model actually using its
        history from one that has collapsed onto the latest frame.
        """
        seq = self._to_sequence(x)
        b, t, c, h, w = seq.shape

        # Fold time into the batch dim so the SHARED encoder runs once over all
        # timesteps; BatchNorm then sees B*T samples instead of B, which matters
        # at batch size 1.
        folded = seq.reshape(b * t, c, h, w)

        s1 = self.inc(folded)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        bottleneck = self.down4(s4)

        # (B*T, Cb, Hb, Wb) -> (B, T, Cb, Hb, Wb); index 0 is the oldest timestep.
        bottleneck_seq = unfold_time(bottleneck, b, t)

        if self.recurrence is not None:
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            hidden: List[torch.Tensor] = []
            for step in range(t):  # unroll chronologically, oldest -> newest
                state = self.recurrence(bottleneck_seq[:, step], state)
                hidden.append(state[0])
            # Every hidden state is kept, not just the last: the attention is
            # what decides which of them the present frame needs.
            attn_input = torch.stack(hidden, dim=1)
        else:
            attn_input = bottleneck_seq

        decoded, weights = self.temporal_attn(attn_input)
        decoded = self.recurrence_proj(decoded)

        # Coarsest first: level 1 is s4, whose 25x12 grid is only a 2x upsample of
        # the 12x6 attention field, and level 4 is s1 at 200x100, a 16x upsample
        # where the weights are near-constant across each block. Fusing fewer
        # levels is therefore the conservative setting, and 0 reproduces
        # ConvLSTMUNet's "latest timestep only" contract exactly.
        skips = []
        for level, feat in enumerate((s4, s3, s2, s1), start=1):
            if level <= self.tattn_fuse_skips:
                skips.append(fuse_over_time(unfold_time(feat, b, t), weights))
            else:
                skips.append(at_latest_timestep(feat, b, t))
        skip4, skip3, skip2, skip1 = skips

        decoded = self.up1(decoded, skip4)
        decoded = self.up2(decoded, skip3)
        decoded = self.up3(decoded, skip2)
        decoded = self.up4(decoded, skip1)
        return self.outc(decoded), weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_attention(x)
        return logits

    # -- checkpoint interface ----------------------------------------------------------

    def config_dict(self) -> Dict[str, Any]:
        """Everything needed to rebuild this model, for checkpoint metadata."""
        return {
            "n_channels_per_timestep": self.n_channels_per_timestep,
            "n_classes": self.n_classes,
            "bilinear": self.bilinear,
            "tattn_dim": self.tattn_dim,
            "tattn_heads": self.tattn_heads,
            "tattn_layers": self.tattn_layers,
            "tattn_recurrence": self.tattn_recurrence,
            "tattn_fuse_skips": self.tattn_fuse_skips,
            "convlstm_hidden_channels": self.convlstm_hidden_channels,
            "convlstm_kernel_size": self.convlstm_kernel_size,
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pop_model_config(state_dict: Any) -> Optional[Dict[str, Any]]:
    """Remove and return the TemporalAttentionUNet config from a checkpoint.

    A no-op returning None on any other checkpoint. Must run before
    ``load_state_dict``, exactly like the ``mask_values`` pop.
    """
    if isinstance(state_dict, dict):
        cfg = state_dict.pop(CONFIG_KEY, None)
        if isinstance(cfg, dict):
            return cfg
    return None


def build_tattn_unet(state_dict: Any = None, **fallback: Any) -> TemporalAttentionUNet:
    """Build a TemporalAttentionUNet, preferring the config stored in `state_dict`.

    Checkpoints written by training carry ``CONFIG_KEY``, so inference rebuilds
    the exact architecture without the user re-specifying heads, width or
    fusion mode; ``fallback`` covers checkpoints saved without it. The config
    key is popped, so the returned model can load the state dict directly.
    """
    cfg = pop_model_config(state_dict)
    merged: Dict[str, Any] = dict(fallback)
    if cfg:
        merged.update(cfg)
    return TemporalAttentionUNet(**merged)
