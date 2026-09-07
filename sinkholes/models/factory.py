"""Checkpoint -> model. The one place any command turns saved weights into a network.

A checkpoint already describes itself:

  - a ConvLSTM U-Net has ``convlstm.*`` weights (and a ``convlstm_unet_config`` blob),
  - a temporal-attention U-Net has ``temporal_attn.*`` weights (and a
    ``tattn_unet_config`` blob); its optional recurrent stage is named
    ``recurrence.*`` precisely so it cannot be mistaken for the ConvLSTM above,
  - ``UNet(add_attn=True)`` has ``attn.*`` weights,
  - ``AttentionUNet`` has a deeper ``DoubleConv`` (``...double_conv.5.*``),
  - the input channel count is ``inc.double_conv.0.weight.shape[1]``.

So the architecture is *detected*, not declared. CLI flags remain as an
explicit override, and a flag that disagrees with the weights is a clear error
instead of a cryptic ``load_state_dict`` key mismatch.

Adding an architecture is one ``register()`` entry below — most specific
first, the plain U-Net last as the catch-all. No command needs editing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch.nn as nn

from .attention_unet import AttentionUNet
from .convlstm_unet import CONFIG_KEY as CONVLSTM_CONFIG_KEY
from .convlstm_unet import build_convlstm_unet
from .tattn_unet import CONFIG_KEY as TATTN_CONFIG_KEY
from .tattn_unet import build_tattn_unet
from .unet import UNet

logger = logging.getLogger(__name__)

#: Checkpoint key recording what the network was fed and what it predicted:
#: ``{"context": [H, W], "predict": [H, W]}``. Architecture-independent on
#: purpose -- every consumer (eval-scenes, test-patches, predict, the attention
#: probe) reads geometry from this one key instead of a per-command flag, so a
#: large-context checkpoint cannot be silently evaluated at 200x100.
IO_GEOMETRY_KEY = "io_geometry"

#: Keys a checkpoint carries that are not model parameters and must be removed
#: before ``load_state_dict``.
NON_PARAMETER_KEYS: Tuple[str, ...] = ("mask_values", CONVLSTM_CONFIG_KEY, TATTN_CONFIG_KEY,
                                       IO_GEOMETRY_KEY, "data_contract")

#: First encoder convolution, shared by every architecture here. Its shape[1]
#: is the input channel count the checkpoint was trained with.
FIRST_CONV_WEIGHT = "inc.double_conv.0.weight"

#: Present only in AttentionUNet — its DoubleConv has a Dropout2d, so index 5
#: holds parameters where the plain U-Net's Sequential stops at 4.
ATTENTION_UNET_MARKER = "down1.maxpool_conv.1.double_conv.5.weight"

#: The ConvLSTM's packed gate convolution. Its output axis is 4 * hidden, which
#: makes the hidden width recoverable from the weights alone.
CONVLSTM_GATE_WEIGHT = "convlstm.conv.weight"

#: TemporalAttentionUNet's attention stage. The prefix identifies the
#: architecture; the input projection's shape gives its width and tells us
#: whether a recurrent stage feeds it.
TATTN_PREFIX = "temporal_attn."
TATTN_IN_PROJ_WEIGHT = "temporal_attn.in_proj.weight"

#: Architectures whose ``n_channels`` counts channels **per timestep** rather
#: than the flat batch channel count, so ``infer_input_channels`` on their
#: weights must not be compared against the loader's T*C. Callers that warn on a
#: channel mismatch have to skip these.
PER_TIMESTEP_ARCHITECTURES: Tuple[str, ...] = ("convlstm_unet", "tattn_unet")


@dataclass(frozen=True)
class Architecture:
    """One model family: how to recognise its checkpoints and how to rebuild it."""

    name: str
    detect: Callable[[Mapping[str, Any]], bool]
    build: Callable[..., nn.Module]
    cli_flag: Optional[str] = None
    description: str = ""


@dataclass
class LoadedModel:
    """Result of :func:`build_from_checkpoint`."""

    model: nn.Module
    architecture: str
    n_channels: Optional[int]
    mask_values: Any = field(default_factory=lambda: [0, 1])
    #: (H, W) the checkpoint was fed, and (H, W) it predicts. Both None for a
    #: checkpoint written before large context, which means 200x100 -> 200x100.
    context_size: Optional[Tuple[int, int]] = None
    predict_size: Optional[Tuple[int, int]] = None
    data_contract: Dict[str, str] = field(default_factory=lambda: {
        "preprocessing_version": "legacy-row-v1", "aoi_selection_version": "legacy-v1"})

    def input_size_for(self, predict_size) -> Tuple[int, int]:
        """Validate the target grid and return the checkpoint's input footprint."""
        from ..dataprep.context import context_margin

        target = tuple(predict_size)
        if self.predict_size is not None and tuple(self.predict_size) != target:
            raise ValueError(f"checkpoint predicts {self.predict_size}, but target grid is {target}")
        context = tuple(self.context_size or target)
        context_margin(target, context)
        return context


_REGISTRY: list[Architecture] = []


def register(arch: Architecture) -> Architecture:
    """Add an architecture to the lookup. Registration order is match order."""
    _REGISTRY.append(arch)
    return arch


def architectures() -> Tuple[Architecture, ...]:
    return tuple(_REGISTRY)


# -- introspection helpers -------------------------------------------------------------

def infer_input_channels(state_dict: Mapping[str, Any]) -> Optional[int]:
    """Input channel count the checkpoint was trained with, or None if unreadable.

    For every architecture except the ConvLSTM this is the flat count (T * C);
    for the ConvLSTM it is the count for one timestep — T is not stored in the
    weights at all.
    """
    weight = state_dict.get(FIRST_CONV_WEIGHT)
    if weight is None:
        weight = next(
            (v for v in state_dict.values() if hasattr(v, "ndim") and v.ndim == 4),
            None,
        )
    return None if weight is None else int(weight.shape[1])


def infer_convlstm_hidden(state_dict: Mapping[str, Any]) -> Optional[int]:
    """Hidden width read straight off the gate convolution, or None.

    The gate conv emits four gates stacked on the output axis, so its shape is
    ``(4 * hidden, C_in + hidden, k, k)`` and ``hidden = shape[0] // 4``. This
    is what keeps checkpoints written before ``CONFIG_KEY`` existed loadable
    across a change to the default hidden size: the weights say what they need,
    so no default has to guess right.
    """
    weight = state_dict.get(CONVLSTM_GATE_WEIGHT)
    if weight is None or getattr(weight, "ndim", 0) != 4:
        return None
    return int(weight.shape[0]) // 4


def infer_tattn_dim(state_dict: Mapping[str, Any]) -> Optional[int]:
    """Attention width read off the input projection, or None.

    ``temporal_attn.in_proj`` is a 1x1 conv ``(dim, C_in, 1, 1)``, so ``dim`` is
    ``shape[0]``. Same purpose as :func:`infer_convlstm_hidden`: a checkpoint
    written before ``CONFIG_KEY`` still rebuilds at its own width rather than at
    whatever the current default happens to be. The head count is *not*
    recoverable this way — heads only change how the width is viewed, never a
    parameter shape — so it falls back to the constructor default and a
    checkpoint carrying the config blob is the only fully self-describing one.
    """
    weight = state_dict.get(TATTN_IN_PROJ_WEIGHT)
    if weight is None or getattr(weight, "ndim", 0) != 4:
        return None
    return int(weight.shape[0])


def strip_non_parameters(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Pop non-parameter keys **in place**; return what was removed.

    Mutates the dict because callers hand the same object to ``load_state_dict``
    immediately afterwards.
    """
    return {k: state_dict.pop(k) for k in NON_PARAMETER_KEYS if k in state_dict}


def io_geometry_of(extras: Dict[str, Any], predict_default=(200, 100)):
    """(context, predict) from a checkpoint's popped non-parameter keys.

    A checkpoint without the key predates large context and is 200x100 in,
    200x100 out -- which is exactly what ``predict_size = None`` means, so old
    checkpoints need no migration.
    """
    geo = extras.get(IO_GEOMETRY_KEY)
    if not geo:
        return None, None
    ctx = tuple(geo.get("context") or predict_default)
    pred = tuple(geo.get("predict") or predict_default)
    return ctx, pred


# -- builders --------------------------------------------------------------------------

def _build_convlstm(state_dict, *, n_channels_per_timestep=None, treat_nodata_regions=False,
                    n_classes=1, bilinear=False, **_):
    # Hidden size, kernel and channels-per-timestep come from the config the
    # trainer embedded in the checkpoint; these are only fallbacks for old files.
    # The hidden fallback is read from the weights rather than left to the
    # constructor default, so a checkpoint predating CONFIG_KEY still rebuilds
    # at its own width instead of at whatever the current default happens to be.
    if n_channels_per_timestep is None:
        n_channels_per_timestep = 2 if treat_nodata_regions else 1
    return build_convlstm_unet(
        state_dict,
        n_channels_per_timestep=n_channels_per_timestep,
        n_classes=n_classes,
        bilinear=bilinear,
        convlstm_hidden_channels=infer_convlstm_hidden(state_dict),
    )


def _build_tattn(state_dict, *, n_channels_per_timestep=None, treat_nodata_regions=False,
                 n_classes=1, bilinear=False, **_):
    # As with the ConvLSTM: heads, layers, recurrence and fusion depth come from
    # the config the trainer embedded; the width is recovered from the weights so
    # an older file still rebuilds correctly.
    if n_channels_per_timestep is None:
        n_channels_per_timestep = 2 if treat_nodata_regions else 1
    return build_tattn_unet(
        state_dict,
        n_channels_per_timestep=n_channels_per_timestep,
        n_classes=n_classes,
        bilinear=bilinear,
        tattn_dim=infer_tattn_dim(state_dict),
    )


def _build_unet(state_dict, *, n_channels=None, n_classes=1, bilinear=False,
                add_attn=False, **_):
    n = n_channels if n_channels is not None else infer_input_channels(state_dict)
    if n is None:
        raise ValueError(
            "cannot determine input channels: checkpoint has no readable first conv. "
            "Pass n_channels= explicitly."
        )
    return UNet(n_channels=n, n_classes=n_classes, bilinear=bilinear, add_attn=add_attn)


def _build_unet_add_attn(state_dict, **kw):
    kw["add_attn"] = True
    return _build_unet(state_dict, **kw)


def _build_attention_unet(state_dict, *, n_channels=None, n_classes=1, bilinear=False, **_):
    n = n_channels if n_channels is not None else infer_input_channels(state_dict)
    if n is None:
        raise ValueError("cannot determine input channels for AttentionUNet. Pass n_channels=.")
    return AttentionUNet(n_channels=n, n_classes=n_classes, bilinear=bilinear)


# -- registry — most specific first, plain U-Net last as the catch-all -----------------

# Ahead of convlstm_unet deliberately. The hybrid (tattn_recurrence="convlstm")
# contains a ConvLSTM cell, so if the ConvLSTM entry matched first a hybrid
# checkpoint would rebuild as the wrong architecture. The cell is also named
# 'recurrence.' rather than 'convlstm.' so neither guard alone is load-bearing.
register(Architecture(
    name="tattn_unet",
    detect=lambda sd: (
        TATTN_CONFIG_KEY in sd or any(k.startswith(TATTN_PREFIX) for k in sd)
    ),
    build=_build_tattn,
    cli_flag="--tattn_unet",
    description="U-Net with causal attention over the bottleneck sequence.",
))

register(Architecture(
    name="convlstm_unet",
    detect=lambda sd: (
        CONVLSTM_CONFIG_KEY in sd or any(k.startswith("convlstm.") for k in sd)
    ),
    build=_build_convlstm,
    cli_flag="--convlstm_unet",
    description="U-Net with a ConvLSTM over the bottleneck sequence.",
))

register(Architecture(
    name="unet_add_attn",
    detect=lambda sd: any(k.startswith("attn.") for k in sd),
    build=_build_unet_add_attn,
    cli_flag="--add_attn",
    description="Plain U-Net with channel self-attention at the bottleneck.",
))

register(Architecture(
    name="attention_unet",
    detect=lambda sd: ATTENTION_UNET_MARKER in sd,
    build=_build_attention_unet,
    cli_flag="--attn_unet",
    description="AttentionUNet with attention gates on every skip connection.",
))

register(Architecture(
    name="unet",
    detect=lambda sd: True,
    build=_build_unet,
    cli_flag=None,
    description="Plain U-Net. Frames stacked as input channels.",
))


# -- public API ------------------------------------------------------------------------

def detect_architecture(state_dict: Mapping[str, Any]) -> str:
    """Name of the architecture these weights belong to."""
    for arch in _REGISTRY:
        if arch.detect(state_dict):
            return arch.name
    raise ValueError("no architecture matched (the U-Net catch-all should be unreachable)")


def get(name: str) -> Architecture:
    for arch in _REGISTRY:
        if arch.name == name:
            return arch
    known = ", ".join(a.name for a in _REGISTRY)
    raise KeyError(f"unknown architecture {name!r}. Known: {known}")


def architecture_from_flags(**flags: bool) -> Optional[str]:
    """Map boolean CLI flags (convlstm_unet / attn_unet / add_attn) to a name.

    Returns None when no flag is set, meaning "detect from the checkpoint".
    Raises when two architecture flags are combined.
    """
    selected = [name for name, on in flags.items() if on]
    if not selected:
        return None
    if len(selected) > 1:
        raise SystemExit(
            f"these options select different architectures and cannot be combined: "
            f"{', '.join(sorted(selected))}. Pass at most one."
        )
    return {
        "convlstm_unet": "convlstm_unet",
        "tattn_unet": "tattn_unet",
        "add_attn": "unet_add_attn",
        "attn_unet": "attention_unet",
    }[selected[0]]


def build_from_checkpoint(
    state_dict: Dict[str, Any],
    *,
    arch: Optional[str] = None,
    strict_flag_check: bool = True,
    **hints: Any,
) -> LoadedModel:
    """Build the network a checkpoint belongs to, ready for ``load_state_dict``.

    ``state_dict`` is **mutated**: non-parameter keys (mask_values, the
    ConvLSTM config) are popped so the returned model can load it directly.

    :param arch: force an architecture (from a CLI flag); None = detect.
    :param strict_flag_check: when ``arch`` contradicts the weights, raise;
        set False to warn and trust the flag.
    :param hints: forwarded to the builder — n_channels, n_classes, bilinear,
        treat_nodata_regions, n_channels_per_timestep.
    """
    detected = detect_architecture(state_dict)

    if arch is not None and arch != detected:
        message = (
            f"checkpoint looks like {detected!r} but {arch!r} was requested "
            f"(flag {get(arch).cli_flag}). "
        )
        if strict_flag_check:
            raise SystemExit(
                message + "Loading it would fail with a key mismatch. Drop the flag to "
                "use the detected architecture, or pass the right one."
            )
        logger.warning("%s— trusting the flag as asked.", message)
        chosen = arch
    else:
        chosen = arch or detected

    n_channels = infer_input_channels(state_dict)
    spec = get(chosen)
    # build() may pop the ConvLSTM config, so capture mask_values around it.
    model = spec.build(state_dict, **hints)
    removed = strip_non_parameters(state_dict)

    logger.info(
        "architecture %s (%s from weights), %s input channels",
        chosen, "detected" if arch is None else "forced", n_channels,
    )
    # Apply the stored geometry to the model itself, so every caller of
    # forward() crops correctly without knowing this key exists.
    ctx, pred = io_geometry_of(removed)
    if pred is not None:
        model.predict_size = None if ctx == pred else tuple(pred)
        logger.info("io_geometry: context %s -> predict %s", tuple(ctx), tuple(pred))

    return LoadedModel(
        model=model,
        architecture=chosen,
        n_channels=n_channels,
        mask_values=removed.get("mask_values", [0, 1]),
        context_size=ctx,
        predict_size=pred,
        data_contract=removed.get("data_contract", {
            "preprocessing_version": "legacy-row-v1", "aoi_selection_version": "legacy-v1"}),
    )


def describe() -> str:
    """Human-readable table of registered architectures."""
    rows = ["  name             flag              description",
            "  ---------------- ----------------- -----------"]
    for a in _REGISTRY:
        rows.append(f"  {a.name:<16} {a.cli_flag or '(default)':<17} {a.description}")
    return "\n".join(rows)
