"""U-Net with a ConvLSTM at the bottleneck, for temporal interferogram sequences.

EDIT 2026-07-28: new file. CHANGELOG.md #6

The stock `UNet` (unet.py) receives a temporal stack as ordinary input channels, so the
first convolution mixes every timestep at once and their ordering carries no meaning.
This model treats the stack as a sequence instead:

    1. a *shared* U-Net encoder runs independently on every timestep,
    2. the bottleneck feature maps are collected across time (kept 2D, never flattened),
    3. a ConvLSTM consumes that sequence,
    4. its final hidden state becomes the decoder input,
    5. the standard U-Net decoder produces the segmentation logits,
    6. skip connections come from the *latest* timestep only (v1 simplification).

Chronological order is `oldest -> newest`, matching the dataset: `find_11day_sequences`
returns `prevs` oldest-first (get_intf_info.py:313-321) and `SubsiDataset` builds
`tids = list(prevs) + [id]` (sinkholes_data_loading.py:176), so index 0 is the oldest
interferogram and index T-1 is the current one.

Output is raw logits `[B, n_classes, H, W]` — no sigmoid, no thresholding — exactly like
`UNet`, so the existing BCEWithLogits + Dice loss and `evaluate()` work unchanged.
"""

from __future__ import annotations

# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from unet_parts import DoubleConv, Down, Up, OutConv

# Key under which a ConvLSTMUNet checkpoint carries the config needed to rebuild the
# model. Stored alongside the existing 'mask_values' key, and popped the same way.
CONFIG_KEY = 'convlstm_unet_config'

# Channels a single interferogram contributes. 1 = phase only; 2 = phase + validity
# (--treat_nodata_regions). Nothing else is meaningful in this pipeline, and accepting
# arbitrary values would make the flat [B, C*T, H, W] form ambiguous.
SUPPORTED_CHANNELS_PER_TIMESTEP = (1, 2)


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell. Gates are 2D convolutions, so spatial dims are preserved.

    Shapes
    ------
    x_t     : (B, input_channels,  H, W)
    h, c    : (B, hidden_channels, H, W)
    returns : (h_next, c_next), both (B, hidden_channels, H, W)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f'ConvLSTMCell needs an odd kernel_size so that padding=kernel_size//2 '
                f'preserves H and W; got kernel_size={kernel_size}.'
            )
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)

        # One convolution produces all four gates at once: i, f, o, g.
        # padding = kernel_size // 2 is what keeps the bottleneck H, W unchanged.
        self.conv = nn.Conv2d(
            self.input_channels + self.hidden_channels,
            4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=bias,
        )

    def init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero hidden/cell state on the same device and dtype as `x`."""
        b, _, h, w = x.shape
        return (
            x.new_zeros(b, self.hidden_channels, h, w),
            x.new_zeros(b, self.hidden_channels, h, w),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.init_state(x_t)
        h_prev, c_prev = state

        combined = torch.cat([x_t, h_prev], dim=1)   # (B, C_in + C_hidden, H, W)
        gates = self.conv(combined)                  # (B, 4 * C_hidden,    H, W)
        i, f, o, g = torch.chunk(gates, 4, dim=1)    # each (B, C_hidden, H, W)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # candidate cell state

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMUNet(nn.Module):
    """U-Net with a ConvLSTM bottleneck over a temporal sequence of interferograms.

    The number of timesteps T is deliberately *not* an architectural parameter — the
    encoder weights are shared across time and the ConvLSTM is unrolled dynamically, so
    a model trained at T=3 can be run at any T (memory permitting).

    Accepted input shapes (nothing is inferred or guessed):

    ==========================  ===========================================  ==========
    shape                       condition                                    meaning
    ==========================  ===========================================  ==========
    ``[B, T, H, W]``            ``n_channels_per_timestep == 1``              T single-channel timesteps
    ``[B, 2T, H, W]``           ``n_channels_per_timestep == 2``              block layout ``[imgs..., validity...]``
    ``[B, T, C, H, W]``         ``C == n_channels_per_timestep``              canonical explicit form
    ==========================  ===========================================  ==========

    Anything else raises ``ValueError``.
    """

    def __init__(
        self,
        n_channels_per_timestep: int = 1,
        n_classes: int = 1,
        bilinear: bool = False,
        convlstm_hidden_channels: Optional[int] = None,
        convlstm_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if n_channels_per_timestep not in SUPPORTED_CHANNELS_PER_TIMESTEP:
            raise ValueError(
                f'n_channels_per_timestep must be one of '
                f'{SUPPORTED_CHANNELS_PER_TIMESTEP} (1 = phase only, '
                f'2 = phase + validity from --treat_nodata_regions); '
                f'got {n_channels_per_timestep!r}.'
            )

        # --- number of TIMESTEPS vs number of CHANNELS PER TIMESTEP -------------------
        # `n_channels` is the attribute the rest of the pipeline reads (the training
        # banner, the inference scripts). For this model it is the channel count of ONE
        # interferogram, NOT the flat channel count of the batch: the flat count is
        # T * n_channels_per_timestep and T varies per run.
        self.n_channels_per_timestep = int(n_channels_per_timestep)
        self.n_channels = self.n_channels_per_timestep
        self.n_classes = int(n_classes)
        self.bilinear = bool(bilinear)

        # --- shared encoder: identical widths to unet.py:153-158 ---------------------
        self.inc = DoubleConv(self.n_channels_per_timestep, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.bottleneck_channels = 1024 // factor
        self.down4 = Down(512, self.bottleneck_channels)

        # --- ConvLSTM over the bottleneck sequence ------------------------------------
        # `None` (or 0, which is how the CLI spells "unset") means: match the U-Net
        # bottleneck width, so the decoder can consume the hidden state directly.
        hidden = (
            self.bottleneck_channels
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
        # A narrower hidden state still has to enter `up1` at the bottleneck width.
        self.convlstm_proj: nn.Module = (
            nn.Identity()
            if hidden == self.bottleneck_channels
            else nn.Conv2d(hidden, self.bottleneck_channels, kernel_size=1)
        )

        # --- decoder: identical to unet.py:159-163 -----------------------------------
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    # ---------------------------------------------------------------------------------
    # input handling
    # ---------------------------------------------------------------------------------
    def _shape_error(self, x: torch.Tensor, detail: str) -> ValueError:
        c = self.n_channels_per_timestep
        missing = ''
        if x.dim() == 4 and x.shape[1] % c != 0:
            missing = (
                ' If this is a non-temporal batch then the temporal dimension is '
                'missing — pass --add_temporal (and --k_prevs N) when training.'
            )
        elif x.dim() == 3:
            missing = (
                ' A 3D tensor has no batch or temporal dimension. Note that batch '
                'size 1 is common here, so never squeeze() the batch dim away.'
            )
        return ValueError(
            f'ConvLSTMUNet expects [B, T, H, W] (n_channels_per_timestep=1), '
            f'[B, 2T, H, W] (n_channels_per_timestep=2, block layout '
            f'[imgs..., validity...]) or the explicit [B, T, C, H, W]. '
            f'Got shape {tuple(x.shape)} with n_channels_per_timestep={c}. '
            f'{detail}{missing}'
        )

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise any accepted input to `[B, T, C, H, W]`, chronological oldest→newest."""
        c = self.n_channels_per_timestep

        if x.dim() == 5:
            if x.shape[2] != c:
                raise self._shape_error(
                    x, f'5D input has {x.shape[2]} channels per timestep, expected {c}.'
                )
            return x

        if x.dim() != 4:
            raise self._shape_error(x, f'Expected a 4D or 5D tensor, got {x.dim()}D.')

        b, c_flat, h, w = x.shape
        if c_flat % c != 0:
            raise self._shape_error(
                x, f'{c_flat} channels is not divisible by {c}.'
            )
        t = c_flat // c

        if c == 1:
            # [B, T, H, W] -> [B, T, 1, H, W]. Explicit dim; never a bare unsqueeze/squeeze.
            return x.unsqueeze(2)

        # Two channels per timestep. The dataset builds this with
        #   np.concatenate([image_data, V], axis=0)   # sinkholes_data_loading.py:272
        # on a (T, N, H, W) array, so the layout is BLOCK — [img_t0..img_t{T-1},
        # V_t0..V_t{T-1}] — not interleaved. Un-flattening therefore has to go
        # (B, C, T, H, W) then permute; a bare reshape(B, T, C, H, W) would pair the
        # wrong image with the wrong validity map.
        return x.reshape(b, c, t, h, w).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _at_latest_timestep(feat: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """(B*T, C, H, W) -> (B, C, H, W) for the newest timestep (index T-1)."""
        return feat.reshape(b, t, *feat.shape[1:])[:, t - 1]

    # ---------------------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits `[B, n_classes, H, W]` (no sigmoid, no threshold).

        Stage shapes for B=2, T=3, C=1, 200x100, bilinear=False:

            input                [2, 3, 200, 100]
            sequence             [2, 3, 1, 200, 100]
            folded (B*T)         [6, 1, 200, 100]
            inc   -> s1          [6, 64, 200, 100]
            down1 -> s2          [6, 128, 100, 50]
            down2 -> s3          [6, 256, 50, 25]
            down3 -> s4          [6, 512, 25, 12]
            down4 -> bottleneck  [6, 1024, 12, 6]
            bottleneck sequence  [2, 3, 1024, 12, 6]
            ConvLSTM final h     [2, 1024, 12, 6]
            decoder + outc       [2, 1, 200, 100]
        """
        seq = self._to_sequence(x)
        b, t, c, h, w = seq.shape

        # Fold time into the batch dim so the SHARED encoder runs once over all
        # timesteps. This also gives BatchNorm B*T samples instead of B, which matters
        # at the project's default --batch_size 1.
        folded = seq.reshape(b * t, c, h, w)

        s1 = self.inc(folded)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        bottleneck = self.down4(s4)

        # (B*T, Cb, Hb, Wb) -> (B, T, Cb, Hb, Wb). Index 0 is the OLDEST timestep,
        # index T-1 the newest, because `folded` was laid out (b0t0, b0t1, ..., b1t0...).
        bottleneck_seq = bottleneck.reshape(b, t, *bottleneck.shape[1:])

        # Unroll the ConvLSTM in chronological order, oldest -> newest.
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for step in range(t):
            state = self.convlstm(bottleneck_seq[:, step], state)
        h_final, _c_final = state  # type: ignore[misc]  # t >= 1 guarantees state is set

        decoded = self.convlstm_proj(h_final)

        # v1: skip connections come from the latest timestep only.
        # TODO(temporal-skips): aggregate skips recurrently across time as well, once the
        # bottleneck-only version has a baseline to beat.
        skip4 = self._at_latest_timestep(s4, b, t)
        skip3 = self._at_latest_timestep(s3, b, t)
        skip2 = self._at_latest_timestep(s2, b, t)
        skip1 = self._at_latest_timestep(s1, b, t)

        decoded = self.up1(decoded, skip4)
        decoded = self.up2(decoded, skip3)
        decoded = self.up3(decoded, skip2)
        decoded = self.up4(decoded, skip1)
        return self.outc(decoded)

    # ---------------------------------------------------------------------------------
    # pipeline interface parity with unet.UNet
    # ---------------------------------------------------------------------------------
    def config_dict(self) -> Dict[str, Any]:
        """Everything needed to rebuild this model, for checkpoint metadata."""
        return {
            'n_channels_per_timestep': self.n_channels_per_timestep,
            'n_classes': self.n_classes,
            'bilinear': self.bilinear,
            'convlstm_hidden_channels': self.convlstm_hidden_channels,
            'convlstm_kernel_size': self.convlstm_kernel_size,
        }

    def use_checkpointing(self) -> None:
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pop_model_config(state_dict: Any) -> Optional[Dict[str, Any]]:
    """Remove and return the ConvLSTMUNet config from a checkpoint, if present.

    Safe to call on a stock UNet checkpoint — it returns None and changes nothing. Must
    be called before `load_state_dict`, exactly like the existing `mask_values` pop.
    """
    if isinstance(state_dict, dict):
        cfg = state_dict.pop(CONFIG_KEY, None)
        if isinstance(cfg, dict):
            return cfg
    return None


def build_convlstm_unet(state_dict: Any = None, **fallback: Any) -> ConvLSTMUNet:
    """Build a ConvLSTMUNet, preferring the config stored in `state_dict`.

    Checkpoints written by train_sinkholes_unet.py carry `CONFIG_KEY`, so inference
    reconstructs the exact architecture without the user re-specifying hidden size,
    kernel or bilinear. `fallback` (from CLI flags) covers checkpoints saved without it.
    The config key is popped, so the returned model can load the state dict directly.
    """
    cfg = pop_model_config(state_dict)
    merged: Dict[str, Any] = dict(fallback)
    if cfg:
        merged.update(cfg)
    return ConvLSTMUNet(**merged)


if __name__ == '__main__':
    net = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1)
    print(f'ConvLSTMUNet params: {net.get_num_params():,}')
    print(f'  bottleneck channels : {net.bottleneck_channels}')
    print(f'  ConvLSTM hidden     : {net.convlstm_hidden_channels}')
    print(f'  ConvLSTM params     : '
          f'{sum(p.numel() for p in net.convlstm.parameters()):,}')
    out = net(torch.randn(2, 3, 200, 100))
    print(f'  [2, 3, 200, 100] -> {tuple(out.shape)}')
