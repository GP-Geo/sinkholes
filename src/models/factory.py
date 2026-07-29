"""Checkpoint → model. The one place any script turns saved weights into a network.

EDIT 2026-07-29: new file. CHANGELOG.md #12

Why this exists
---------------
Model construction was copy-pasted into five scripts (`test.py`, `test_full_intf.py`,
`inspect_run.py`, `predict_new_intf.py`, `predict_new_intf_vA.py`), each an
`if flag: A() else: B()` chain. Adding the ConvLSTM U-Net (`CHANGELOG.md #6`) only
updated `test.py`, so the other four silently could not load it — stages (3b) and (4)
were unavailable for the best-scoring model in the repo.

The fix is not "add the flag in four more places". It is to notice that **a checkpoint
already describes itself**:

  - a ConvLSTM U-Net has ``convlstm.*`` weights (and a ``convlstm_unet_config`` blob),
  - ``UNet(add_attn=True)`` has ``attn.*`` weights,
  - ``AttentionUNet`` has a deeper ``DoubleConv`` (``...double_conv.5.*``),
  - the input channel count is ``inc.double_conv.0.weight.shape[1]``.

So the architecture is *detected*, not declared. CLI flags stay supported as an explicit
override, and now a flag that disagrees with the weights is a clear error instead of a
cryptic ``load_state_dict`` key mismatch.

Adding an architecture
----------------------
Append one ``Architecture`` entry below. Every script picks it up with no further edits::

    register(Architecture(
        name='my_net',
        detect=lambda sd: any(k.startswith('my_block.') for k in sd),
        build=_build_my_net,
        cli_flag='--my_net',
    ))

Order matters: entries are tried in registration order, most specific first, with the
plain U-Net as the catch-all. Keep new entries above it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch.nn as nn

from unet import UNet
from attn_unet import AttentionUNet
from convlstm_unet import (
    CONFIG_KEY as CONVLSTM_CONFIG_KEY,
    build_convlstm_unet,
)

logger = logging.getLogger(__name__)

#: Keys a checkpoint carries that are not model parameters and must be removed
#: before ``load_state_dict``.
NON_PARAMETER_KEYS: Tuple[str, ...] = ('mask_values', CONVLSTM_CONFIG_KEY)

#: First encoder convolution, shared by every architecture here. Its ``shape[1]`` is the
#: input channel count the checkpoint was trained with.
FIRST_CONV_WEIGHT = 'inc.double_conv.0.weight'

#: Present only in ``AttentionUNet`` — its ``DoubleConv`` has two extra layers, so index
#: 5 exists where the plain U-Net stops at 4.
ATTENTION_UNET_MARKER = 'down1.maxpool_conv.1.double_conv.5.weight'


@dataclass(frozen=True)
class Architecture:
    """One model family: how to recognise its checkpoints and how to rebuild it."""

    name: str
    detect: Callable[[Mapping[str, Any]], bool]
    build: Callable[..., nn.Module]
    cli_flag: Optional[str] = None
    description: str = ''


@dataclass
class LoadedModel:
    """Result of :func:`build_from_checkpoint`."""

    model: nn.Module
    architecture: str
    n_channels: Optional[int]
    mask_values: Any = field(default_factory=lambda: [0, 1])


_REGISTRY: list[Architecture] = []


def register(arch: Architecture) -> Architecture:
    """Add an architecture to the lookup. Registration order is match order."""
    _REGISTRY.append(arch)
    return arch


def architectures() -> Tuple[Architecture, ...]:
    """All registered architectures, in match order."""
    return tuple(_REGISTRY)


# --------------------------------------------------------------------------------------
# introspection helpers
# --------------------------------------------------------------------------------------

def infer_input_channels(state_dict: Mapping[str, Any]) -> Optional[int]:
    """Input channel count the checkpoint was trained with, or ``None`` if unreadable.

    For every architecture except the ConvLSTM this is the *flat* count (``T * C``).
    For the ConvLSTM it is the count for **one timestep** — the model is unrolled over T
    dynamically, so T is not stored in the weights at all.
    """
    weight = state_dict.get(FIRST_CONV_WEIGHT)
    if weight is None:
        weight = next(
            (v for v in state_dict.values() if hasattr(v, 'ndim') and v.ndim == 4),
            None,
        )
    if weight is None:
        return None
    return int(weight.shape[1])


def strip_non_parameters(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Pop non-parameter keys **in place**; return what was removed.

    Mutates the dict because callers hand the same object to ``load_state_dict``
    immediately afterwards — the behaviour ``test.py`` already relied on.
    """
    return {k: state_dict.pop(k) for k in NON_PARAMETER_KEYS if k in state_dict}


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------

def _build_convlstm(state_dict, *, n_channels_per_timestep=None, treat_nodata_regions=False,
                    n_classes=1, bilinear=False, **_):
    # Hidden size, kernel and channels-per-timestep come from the config the trainer
    # embedded in the checkpoint; these values are only a fallback for older files.
    if n_channels_per_timestep is None:
        n_channels_per_timestep = 2 if treat_nodata_regions else 1
    return build_convlstm_unet(
        state_dict,
        n_channels_per_timestep=n_channels_per_timestep,
        n_classes=n_classes,
        bilinear=bilinear,
    )


def _build_unet(state_dict, *, n_channels=None, n_classes=1, bilinear=False,
                add_attn=False, **_):
    n = n_channels if n_channels is not None else infer_input_channels(state_dict)
    if n is None:
        raise ValueError(
            'cannot determine input channels: checkpoint has no readable first conv. '
            'Pass n_channels= explicitly.'
        )
    return UNet(n_channels=n, n_classes=n_classes, bilinear=bilinear, add_attn=add_attn)


def _build_unet_add_attn(state_dict, **kw):
    kw['add_attn'] = True
    return _build_unet(state_dict, **kw)


def _build_attention_unet(state_dict, *, n_channels=None, n_classes=1, bilinear=False, **_):
    n = n_channels if n_channels is not None else infer_input_channels(state_dict)
    if n is None:
        raise ValueError(
            'cannot determine input channels for AttentionUNet. Pass n_channels=.'
        )
    return AttentionUNet(n_channels=n, n_classes=n_classes, bilinear=bilinear)


# --------------------------------------------------------------------------------------
# registry — most specific first, plain U-Net last as the catch-all
# --------------------------------------------------------------------------------------

register(Architecture(
    name='convlstm_unet',
    detect=lambda sd: (
        CONVLSTM_CONFIG_KEY in sd or any(k.startswith('convlstm.') for k in sd)
    ),
    build=_build_convlstm,
    cli_flag='--convlstm_unet',
    description='U-Net with a ConvLSTM over the bottleneck sequence (CHANGELOG #6).',
))

register(Architecture(
    name='unet_add_attn',
    detect=lambda sd: any(k.startswith('attn.') for k in sd),
    build=_build_unet_add_attn,
    cli_flag='--add_attn',
    description='Plain U-Net with self-attention at the bottleneck.',
))

register(Architecture(
    name='attention_unet',
    detect=lambda sd: ATTENTION_UNET_MARKER in sd,
    build=_build_attention_unet,
    cli_flag='--attn_unet',
    description='AttentionUNet with attention gates on every skip connection.',
))

register(Architecture(
    name='unet',
    detect=lambda sd: True,
    build=_build_unet,
    cli_flag=None,
    description='Plain U-Net. Frames stacked as input channels.',
))


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------

def detect_architecture(state_dict: Mapping[str, Any]) -> str:
    """Name of the architecture these weights belong to."""
    for arch in _REGISTRY:
        if arch.detect(state_dict):
            return arch.name
    raise ValueError('no architecture matched (the U-Net catch-all should be unreachable)')


def get(name: str) -> Architecture:
    for arch in _REGISTRY:
        if arch.name == name:
            return arch
    known = ', '.join(a.name for a in _REGISTRY)
    raise KeyError(f'unknown architecture {name!r}. Known: {known}')


def architecture_from_flags(**flags: bool) -> Optional[str]:
    """Map legacy boolean CLI flags to an architecture name.

    Accepts the flag names as they appear in argparse (``convlstm_unet=True``,
    ``attn_unet=True``, ``add_attn=True``). Returns ``None`` when no flag is set, which
    means "detect from the checkpoint". Raises if two architecture flags are combined.
    """
    selected = [name for name, on in flags.items() if on]
    if not selected:
        return None
    if len(selected) > 1:
        # Deliberately not prefixed with "--": the same selector is spelled differently
        # across scripts (test.py uses --unet_attn where the others use --attn_unet).
        raise SystemExit(
            f'these options select different architectures and cannot be combined: '
            f'{", ".join(sorted(selected))}. Pass at most one.'
        )
    return {
        'convlstm_unet': 'convlstm_unet',
        'add_attn': 'unet_add_attn',
        'attn_unet': 'attention_unet',
    }[selected[0]]


def build_from_checkpoint(
    state_dict: Dict[str, Any],
    *,
    arch: Optional[str] = None,
    strict_flag_check: bool = True,
    **hints: Any,
) -> LoadedModel:
    """Build the network a checkpoint belongs to, ready for ``load_state_dict``.

    ``state_dict`` is **mutated**: non-parameter keys (``mask_values``, the ConvLSTM
    config) are popped so the returned model can load it directly.

    :param arch: force an architecture (from a CLI flag). ``None`` = detect from weights.
    :param strict_flag_check: when ``arch`` is given and contradicts the weights, raise.
        Set ``False`` to warn and trust ``arch`` instead.
    :param hints: forwarded to the builder — ``n_channels``, ``n_classes``, ``bilinear``,
        ``treat_nodata_regions``, ``n_channels_per_timestep``.

    Detection reads the weights, so it is correct even when a caller passes the wrong
    ``--k_prevs``: the channel count comes from the checkpoint, not the flag.
    """
    detected = detect_architecture(state_dict)

    if arch is not None and arch != detected:
        message = (
            f'checkpoint looks like {detected!r} but {arch!r} was requested '
            f'(flag {get(arch).cli_flag}). '
        )
        if strict_flag_check:
            raise SystemExit(
                message + 'Loading it would fail with a key mismatch. Drop the flag to '
                'use the detected architecture, or pass the right one.'
            )
        logger.warning('%s— trusting the flag as asked.', message)
        chosen = arch
    else:
        chosen = arch or detected

    n_channels = infer_input_channels(state_dict)
    spec = get(chosen)
    # build() may pop the ConvLSTM config, so capture mask_values around it.
    model = spec.build(state_dict, **hints)
    removed = strip_non_parameters(state_dict)

    logger.info(
        'architecture %s (%s from weights), %s input channels',
        chosen, 'detected' if arch is None else 'forced', n_channels,
    )
    return LoadedModel(
        model=model,
        architecture=chosen,
        n_channels=n_channels,
        mask_values=removed.get('mask_values', [0, 1]),
    )


def describe() -> str:
    """Human-readable table of registered architectures (for ``--list_architectures``)."""
    rows = ['  name             flag              description',
            '  ---------------- ----------------- -----------']
    for a in _REGISTRY:
        rows.append(f'  {a.name:<16} {a.cli_flag or "(default)":<17} {a.description}')
    return '\n'.join(rows)


if __name__ == '__main__':
    print('Registered architectures (match order):')
    print(describe())
