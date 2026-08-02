"""U-Net with a ConvLSTM at the bottleneck, for temporal interferogram sequences.

The plain ``UNet`` receives a temporal stack as ordinary input channels, so the
first convolution mixes every timestep at once and their ordering carries no
meaning. This model treats the stack as a sequence instead:

    1. a *shared* U-Net encoder runs independently on every timestep,
    2. the bottleneck feature maps are collected across time (kept 2D),
    3. a ConvLSTM consumes that sequence chronologically,
    4. its final hidden state becomes the decoder input,
    5. the standard U-Net decoder produces the segmentation logits,
    6. skip connections come from the *latest* timestep only.

Chronological order is oldest -> newest, matching the dataset: chains are
returned oldest-first and the stack is ``prevs + [current]``, so index 0 is the
oldest interferogram and index T-1 the current one.

Output is raw logits ``[B, n_classes, H, W]`` — no sigmoid — exactly like
``UNet``, so the same losses and evaluation apply.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .parts import DoubleConv, Down, OutConv, Up

#: Key under which a ConvLSTMUNet checkpoint carries the config needed to
#: rebuild the model. Stored alongside 'mask_values' and popped the same way.
CONFIG_KEY = "convlstm_unet_config"

#: Channels a single interferogram contributes: 1 = phase only, 2 = phase +
#: validity (--treat_nodata_regions). Anything else would make the flat
#: [B, C*T, H, W] form ambiguous.
SUPPORTED_CHANNELS_PER_TIMESTEP = (1, 2)

#: Hidden width of the ConvLSTM when nothing is specified. Deliberately *not*
#: the bottleneck width: at a 200x100 patch the bottleneck grid is only ~13x7,
#: and a cell matched to 1024 channels there is 75.5M parameters — 71% of the
#: whole network — against 11.8M (27%) at 256. The decoder still receives the
#: bottleneck width; ``convlstm_proj`` widens the hidden state back up.
DEFAULT_CONVLSTM_HIDDEN_CHANNELS = 256

#: Index of the forget gate in the packed gate axis. This cell emits the gates
#: in the order i, f, o, g — NOT torch.nn.LSTM's i, f, g, o — so any recipe
#: copied from a standard LSTM would land on the wrong slice, silently, with
#: no shape error to catch it.
GATE_ORDER = ("i", "f", "o", "g")
FORGET_GATE_INDEX = GATE_ORDER.index("f")


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell. Gates are 2D convolutions, so spatial dims persist.

    Shapes: x_t (B, C_in, H, W); h, c (B, C_hidden, H, W).
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"ConvLSTMCell needs an odd kernel_size so that padding=kernel_size//2 "
                f"preserves H and W; got kernel_size={kernel_size}."
            )
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        # One convolution produces all four gates at once: i, f, o, g.
        self.conv = nn.Conv2d(
            self.input_channels + self.hidden_channels,
            4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Gate-aware initialisation, replacing Conv2d's generic default.

        The single convolution is really eight blocks: four gates, each with an
        input-to-hidden half (columns ``[:C_in]``) and a recurrent
        hidden-to-hidden half (columns ``[C_in:]``). They are initialised
        differently on purpose:

        - **input-to-hidden: Xavier.** Fan-in and fan-out are computed per gate,
          so each gate is scaled for the one hidden block it actually feeds
          rather than for the packed 4H output.
        - **hidden-to-hidden: orthogonal.** The recurrent path is applied
          repeatedly, so an orthogonal map keeps the state's scale steady across
          the unroll instead of compounding it (Saxe et al., 2014).
        - **biases: zero, except the forget gate, which starts at 1.**
          A zero forget bias means ``f = sigmoid(0) = 0.5``, halving the cell
          state every timestep — at T=3 the oldest interferogram would reach the
          decoder at ~25% strength. Starting at 1 gives ``f ~ 0.73``, so the
          sequence is remembered by default and the model has to learn to
          forget (Jozefowicz et al., 2015).

        Each block is built in its own contiguous tensor and copied in: the
        strided view ``weight[rows, :C_in]`` cannot be passed to
        ``orthogonal_``, which reshapes its argument.
        """
        h, c_in, k = self.hidden_channels, self.input_channels, self.kernel_size
        with torch.no_grad():
            for gate in range(len(GATE_ORDER)):
                rows = slice(gate * h, (gate + 1) * h)

                input_half = torch.empty(h, c_in, k, k)
                nn.init.xavier_uniform_(input_half)
                self.conv.weight[rows, :c_in].copy_(input_half)

                recurrent_half = torch.empty(h, h, k, k)
                nn.init.orthogonal_(recurrent_half)
                self.conv.weight[rows, c_in:].copy_(recurrent_half)

            if self.conv.bias is not None:
                self.conv.bias.zero_()
                forget = slice(FORGET_GATE_INDEX * h, (FORGET_GATE_INDEX + 1) * h)
                self.conv.bias[forget].fill_(1.0)

    def init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        return (
            x.new_zeros(b, self.hidden_channels, h, w),
            x.new_zeros(b, self.hidden_channels, h, w),
        )

    def forward(self, x_t, state=None):
        if state is None:
            state = self.init_state(x_t)
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x_t, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMUNet(nn.Module):
    """U-Net with a ConvLSTM bottleneck over a temporal interferogram sequence.

    The number of timesteps T is deliberately *not* an architectural parameter:
    the encoder is shared and the ConvLSTM is unrolled dynamically, so a model
    trained at T=3 runs at any T. Consequently ``n_channels`` on this class
    means channels **per timestep**, not the flat batch channel count.

    Accepted input shapes (nothing is guessed):

    - ``[B, T, H, W]``     when ``n_channels_per_timestep == 1``
    - ``[B, 2T, H, W]``    when ``n_channels_per_timestep == 2`` — BLOCK layout
      ``[img_t0..img_t{T-1}, V_t0..V_t{T-1}]``, as the dataset concatenates it
    - ``[B, T, C, H, W]``  the canonical explicit form

    Anything else raises ``ValueError``.
    """

    def __init__(
        self,
        n_channels_per_timestep: int = 1,
        n_classes: int = 1,
        bilinear: bool = False,
        convlstm_hidden_channels: Optional[int] = None,
        convlstm_kernel_size: int = 3,
    ):
        super().__init__()
        if n_channels_per_timestep not in SUPPORTED_CHANNELS_PER_TIMESTEP:
            raise ValueError(
                f"n_channels_per_timestep must be one of {SUPPORTED_CHANNELS_PER_TIMESTEP} "
                f"(1 = phase only, 2 = phase + validity); got {n_channels_per_timestep!r}."
            )

        self.n_channels_per_timestep = int(n_channels_per_timestep)
        self.n_channels = self.n_channels_per_timestep
        self.n_classes = int(n_classes)
        self.bilinear = bool(bilinear)

        # Shared encoder — identical widths to the plain U-Net.
        self.inc = DoubleConv(self.n_channels_per_timestep, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.bottleneck_channels = 1024 // factor
        self.down4 = Down(512, self.bottleneck_channels)

        # None (or 0, the CLI's spelling of "unset") selects the default width.
        # It is a flat number rather than the bottleneck width, so --bilinear
        # no longer changes the size of the recurrent state: how far the decoder
        # upsamples has no bearing on how much memory the sequence needs.
        hidden = (
            DEFAULT_CONVLSTM_HIDDEN_CHANNELS
            if convlstm_hidden_channels in (None, 0)
            else int(convlstm_hidden_channels)
        )
        self.convlstm_hidden_channels = hidden
        self.convlstm_kernel_size = int(convlstm_kernel_size)
        self.convlstm = ConvLSTMCell(
            input_channels=self.bottleneck_channels,
            hidden_channels=hidden,
            kernel_size=self.convlstm_kernel_size,
        )
        # A narrower hidden state still has to enter up1 at the bottleneck width.
        self.convlstm_proj: nn.Module = (
            nn.Identity()
            if hidden == self.bottleneck_channels
            else nn.Conv2d(hidden, self.bottleneck_channels, kernel_size=1)
        )

        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    # -- input handling ----------------------------------------------------------------

    def _shape_error(self, x: torch.Tensor, detail: str) -> ValueError:
        c = self.n_channels_per_timestep
        hint = ""
        if x.dim() == 4 and x.shape[1] % c != 0:
            hint = (
                " If this is a non-temporal batch then the temporal dimension is "
                "missing — pass --add_temporal (and --k_prevs N) when training."
            )
        elif x.dim() == 3:
            hint = (
                " A 3D tensor has no batch or temporal dimension. Note that batch "
                "size 1 is common here, so never squeeze() the batch dim away."
            )
        return ValueError(
            f"ConvLSTMUNet expects [B, T, H, W] (n_channels_per_timestep=1), "
            f"[B, 2T, H, W] (n_channels_per_timestep=2, block layout "
            f"[imgs..., validity...]) or the explicit [B, T, C, H, W]. "
            f"Got shape {tuple(x.shape)} with n_channels_per_timestep={c}. {detail}{hint}"
        )

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise any accepted input to [B, T, C, H, W], oldest -> newest."""
        c = self.n_channels_per_timestep

        if x.dim() == 5:
            if x.shape[2] != c:
                raise self._shape_error(
                    x, f"5D input has {x.shape[2]} channels per timestep, expected {c}."
                )
            return x
        if x.dim() != 4:
            raise self._shape_error(x, f"Expected a 4D or 5D tensor, got {x.dim()}D.")

        b, c_flat, h, w = x.shape
        if c_flat % c != 0:
            raise self._shape_error(x, f"{c_flat} channels is not divisible by {c}.")
        t = c_flat // c

        if c == 1:
            return x.unsqueeze(2)

        # Two channels per timestep arrive BLOCK-laid-out ([imgs..., validity...]),
        # so un-flattening must go (B, C, T, H, W) then permute — a bare
        # reshape(B, T, C, H, W) would pair each image with the wrong validity map.
        return x.reshape(b, c, t, h, w).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _at_latest_timestep(feat: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """(B*T, C, H, W) -> (B, C, H, W) for the newest timestep (index T-1)."""
        return feat.reshape(b, t, *feat.shape[1:])[:, t - 1]

    # -- forward -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        bottleneck_seq = bottleneck.reshape(b, t, *bottleneck.shape[1:])

        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for step in range(t):  # unroll chronologically, oldest -> newest
            state = self.convlstm(bottleneck_seq[:, step], state)
        h_final, _ = state  # type: ignore[misc]

        decoded = self.convlstm_proj(h_final)

        # Skip connections come from the latest timestep only.
        skip4 = self._at_latest_timestep(s4, b, t)
        skip3 = self._at_latest_timestep(s3, b, t)
        skip2 = self._at_latest_timestep(s2, b, t)
        skip1 = self._at_latest_timestep(s1, b, t)

        decoded = self.up1(decoded, skip4)
        decoded = self.up2(decoded, skip3)
        decoded = self.up3(decoded, skip2)
        decoded = self.up4(decoded, skip1)
        return self.outc(decoded)

    # -- checkpoint interface ----------------------------------------------------------

    def config_dict(self) -> Dict[str, Any]:
        """Everything needed to rebuild this model, for checkpoint metadata."""
        return {
            "n_channels_per_timestep": self.n_channels_per_timestep,
            "n_classes": self.n_classes,
            "bilinear": self.bilinear,
            "convlstm_hidden_channels": self.convlstm_hidden_channels,
            "convlstm_kernel_size": self.convlstm_kernel_size,
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pop_model_config(state_dict: Any) -> Optional[Dict[str, Any]]:
    """Remove and return the ConvLSTMUNet config from a checkpoint, if present.

    A no-op returning None on a plain U-Net checkpoint. Must run before
    ``load_state_dict``, exactly like the ``mask_values`` pop.
    """
    if isinstance(state_dict, dict):
        cfg = state_dict.pop(CONFIG_KEY, None)
        if isinstance(cfg, dict):
            return cfg
    return None


def build_convlstm_unet(state_dict: Any = None, **fallback: Any) -> ConvLSTMUNet:
    """Build a ConvLSTMUNet, preferring the config stored in `state_dict`.

    Checkpoints written by training carry ``CONFIG_KEY``, so inference rebuilds
    the exact architecture without the user re-specifying hidden size or
    kernel; ``fallback`` covers checkpoints saved without it. The config key is
    popped, so the returned model can load the state dict directly.
    """
    cfg = pop_model_config(state_dict)
    merged: Dict[str, Any] = dict(fallback)
    if cfg:
        merged.update(cfg)
    return ConvLSTMUNet(**merged)
