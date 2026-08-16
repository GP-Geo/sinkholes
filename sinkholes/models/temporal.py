"""Input-layout helpers shared by the models that consume a sequence.

The dataloader always yields a 4D ``[B, C*T, H, W]`` batch — ``__getitem__``
returns ``(T, H, W)`` or ``(2T, H, W)`` per sample and there is no custom
collate (``dataprep/dataset.py:388``) — so every temporal model has to
un-flatten the same way and step around the same block-layout trap. Keeping
that in one place means the two models cannot drift apart on what a batch
means.

Nothing here holds parameters, so nothing here is load-bearing for checkpoints.
"""

from __future__ import annotations

import torch

#: Channels a single interferogram contributes: 1 = phase only, 2 = phase +
#: validity (--treat_nodata_regions). Anything else would make the flat
#: [B, C*T, H, W] form ambiguous.
SUPPORTED_CHANNELS_PER_TIMESTEP = (1, 2)


def check_channels_per_timestep(value: int) -> int:
    """Validate a per-timestep channel count, returning it as an int."""
    if value not in SUPPORTED_CHANNELS_PER_TIMESTEP:
        raise ValueError(
            f"n_channels_per_timestep must be one of {SUPPORTED_CHANNELS_PER_TIMESTEP} "
            f"(1 = phase only, 2 = phase + validity); got {value!r}."
        )
    return int(value)


def sequence_shape_error(x: torch.Tensor, channels_per_timestep: int, model_name: str,
                         detail: str) -> ValueError:
    """The shared 'this is not a temporal batch' message, with the usual causes."""
    c = channels_per_timestep
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
        f"{model_name} expects [B, T, H, W] (n_channels_per_timestep=1), "
        f"[B, 2T, H, W] (n_channels_per_timestep=2, block layout "
        f"[imgs..., validity...]) or the explicit [B, T, C, H, W]. "
        f"Got shape {tuple(x.shape)} with n_channels_per_timestep={c}. {detail}{hint}"
    )


def to_sequence(x: torch.Tensor, channels_per_timestep: int, *,
                model_name: str) -> torch.Tensor:
    """Normalise any accepted input to ``[B, T, C, H, W]``, oldest -> newest."""
    c = channels_per_timestep

    if x.dim() == 5:
        if x.shape[2] != c:
            raise sequence_shape_error(
                x, c, model_name,
                f"5D input has {x.shape[2]} channels per timestep, expected {c}.",
            )
        return x
    if x.dim() != 4:
        raise sequence_shape_error(
            x, c, model_name, f"Expected a 4D or 5D tensor, got {x.dim()}D."
        )

    b, c_flat, h, w = x.shape
    if c_flat % c != 0:
        raise sequence_shape_error(
            x, c, model_name, f"{c_flat} channels is not divisible by {c}."
        )
    t = c_flat // c

    if c == 1:
        return x.unsqueeze(2)

    # Two channels per timestep arrive BLOCK-laid-out ([imgs..., validity...]),
    # so un-flattening must go (B, C, T, H, W) then permute — a bare
    # reshape(B, T, C, H, W) would pair each image with the wrong validity map.
    return x.reshape(b, c, t, h, w).permute(0, 2, 1, 3, 4)


def at_latest_timestep(feat: torch.Tensor, b: int, t: int) -> torch.Tensor:
    """``(B*T, C, H, W)`` -> ``(B, C, H, W)`` for the newest timestep (index T-1)."""
    return feat.reshape(b, t, *feat.shape[1:])[:, t - 1]


def unfold_time(feat: torch.Tensor, b: int, t: int) -> torch.Tensor:
    """``(B*T, C, H, W)`` -> ``(B, T, C, H, W)``, the whole sequence."""
    return feat.reshape(b, t, *feat.shape[1:])
