"""Tests for the checkpoint -> model factory.

EDIT 2026-07-29: new file. CHANGELOG.md #12

The contract these lock down: a checkpoint saved by any supported architecture must be
detected and rebuilt from its weights alone, with no CLI flag. That is what makes stages
(3b) and (4) work for ConvLSTM without a per-script if/else chain, so a regression here
silently re-breaks them.

Pure tensor tests — models are built small where shape does not matter, no data files.
"""

import pytest
import torch

from attn_unet import AttentionUNet
from convlstm_unet import CONFIG_KEY, ConvLSTMUNet
from unet import UNet

import factory
from factory import (
    NON_PARAMETER_KEYS,
    Architecture,
    architecture_from_flags,
    build_from_checkpoint,
    detect_architecture,
    infer_input_channels,
    strip_non_parameters,
)

PATCH_H, PATCH_W = 200, 100


def checkpoint_for(model, mask_values=(0, 1), extra=None):
    """A state_dict shaped like the trainer writes it: weights + non-parameter keys."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd['mask_values'] = list(mask_values)
    if extra:
        sd.update(extra)
    return sd


CONVLSTM_CONFIG = {
    'n_channels_per_timestep': 1,
    'n_classes': 1,
    'bilinear': False,
    'convlstm_hidden_channels': 1024,
    'convlstm_kernel_size': 3,
}


def make_cases():
    return {
        'unet': (UNet(n_channels=3, n_classes=1, bilinear=False), None),
        'unet_add_attn': (
            UNet(n_channels=3, n_classes=1, bilinear=False, add_attn=True), None,
        ),
        'attention_unet': (
            AttentionUNet(n_channels=3, n_classes=1, bilinear=False), None,
        ),
        'convlstm_unet': (
            ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1),
            {CONFIG_KEY: dict(CONVLSTM_CONFIG)},
        ),
    }


@pytest.mark.parametrize('expected', sorted(make_cases()))
def test_detects_architecture_from_weights_alone(expected):
    model, extra = make_cases()[expected]
    sd = checkpoint_for(model, extra=extra)
    assert detect_architecture(sd) == expected


@pytest.mark.parametrize('expected', sorted(make_cases()))
def test_rebuilt_model_loads_its_own_checkpoint(expected):
    """The end-to-end contract: detect -> build -> load_state_dict, no flags."""
    model, extra = make_cases()[expected]
    sd = checkpoint_for(model, extra=extra)

    loaded = build_from_checkpoint(sd)

    assert loaded.architecture == expected
    assert loaded.mask_values == [0, 1]
    # No leftover non-parameter keys, so this must not raise.
    loaded.model.load_state_dict(sd)


@pytest.mark.parametrize('expected', sorted(make_cases()))
def test_rebuilt_model_reproduces_the_original_output(expected):
    """Detection is not enough — the rebuilt weights must compute the same thing."""
    model, extra = make_cases()[expected]
    model.eval()
    sd = checkpoint_for(model, extra=extra)

    loaded = build_from_checkpoint(sd)
    loaded.model.load_state_dict(sd)
    loaded.model.eval()

    x = torch.randn(2, 3, PATCH_H, PATCH_W)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded.model(x), atol=1e-6)


def test_convlstm_config_drives_reconstruction():
    """Hidden size comes from the checkpoint, not from a caller-supplied default."""
    model = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1,
                         convlstm_hidden_channels=64, convlstm_kernel_size=3)
    sd = checkpoint_for(model, extra={CONFIG_KEY: {
        **CONVLSTM_CONFIG, 'convlstm_hidden_channels': 64,
    }})

    # Pass a deliberately wrong fallback; the stored config must win.
    loaded = build_from_checkpoint(sd, n_channels_per_timestep=1)

    assert loaded.model.convlstm_hidden_channels == 64
    loaded.model.load_state_dict(sd)


def test_infer_input_channels_reads_the_first_conv():
    for n in (1, 3, 6):
        sd = UNet(n_channels=n, n_classes=1, bilinear=False).state_dict()
        assert infer_input_channels(sd) == n


def test_unet_channels_inferred_when_no_hint_given():
    """A caller that passes the wrong --k_prevs should not silently build a wrong net."""
    sd = checkpoint_for(UNet(n_channels=6, n_classes=1, bilinear=False))
    loaded = build_from_checkpoint(sd)
    assert loaded.n_channels == 6
    loaded.model.load_state_dict(sd)


def test_explicit_n_channels_hint_is_respected():
    sd = checkpoint_for(UNet(n_channels=3, n_classes=1, bilinear=False))
    loaded = build_from_checkpoint(sd, n_channels=3)
    loaded.model.load_state_dict(sd)


def test_strip_non_parameters_removes_exactly_the_known_keys():
    sd = checkpoint_for(UNet(n_channels=3, n_classes=1, bilinear=False),
                        extra={CONFIG_KEY: dict(CONVLSTM_CONFIG)})
    n_before = len(sd)

    removed = strip_non_parameters(sd)

    assert set(removed) == set(NON_PARAMETER_KEYS)
    assert len(sd) == n_before - len(NON_PARAMETER_KEYS)
    assert not any(k in sd for k in NON_PARAMETER_KEYS)


def test_wrong_flag_is_rejected_with_a_readable_error():
    """A flag contradicting the weights must fail loudly, not as a key mismatch."""
    sd = checkpoint_for(ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1),
                        extra={CONFIG_KEY: dict(CONVLSTM_CONFIG)})

    with pytest.raises(SystemExit, match='convlstm_unet'):
        build_from_checkpoint(sd, arch='attention_unet')


def test_wrong_flag_can_be_forced_through():
    sd = checkpoint_for(UNet(n_channels=3, n_classes=1, bilinear=False))
    loaded = build_from_checkpoint(sd, arch='attention_unet', strict_flag_check=False)
    assert loaded.architecture == 'attention_unet'


def test_flags_map_to_architectures():
    assert architecture_from_flags() is None
    assert architecture_from_flags(convlstm_unet=False, attn_unet=False) is None
    assert architecture_from_flags(convlstm_unet=True) == 'convlstm_unet'
    assert architecture_from_flags(attn_unet=True) == 'attention_unet'
    assert architecture_from_flags(add_attn=True) == 'unet_add_attn'


def test_combining_architecture_flags_is_rejected():
    with pytest.raises(SystemExit, match='cannot be combined'):
        architecture_from_flags(convlstm_unet=True, attn_unet=True)


def test_plain_unet_is_the_catch_all():
    """An unrecognised checkpoint falls through to UNet rather than raising."""
    sd = {'inc.double_conv.0.weight': torch.zeros(64, 4, 3, 3)}
    assert detect_architecture(sd) == 'unet'


def test_registering_a_new_architecture_needs_no_call_site_change():
    """The extension point: one register() call and every script picks it up."""
    marker = 'my_experimental_block.weight'
    sentinel = object()

    arch = Architecture(
        name='experimental',
        detect=lambda sd: marker in sd,
        build=lambda sd, **kw: sentinel,
        cli_flag='--experimental',
    )
    factory._REGISTRY.insert(0, arch)
    try:
        sd = {marker: torch.zeros(1), 'inc.double_conv.0.weight': torch.zeros(64, 3, 3, 3)}
        assert detect_architecture(sd) == 'experimental'
        assert build_from_checkpoint(sd).model is sentinel
    finally:
        factory._REGISTRY.remove(arch)

    # and detection reverts once it is unregistered
    assert detect_architecture({'inc.double_conv.0.weight': torch.zeros(64, 3, 3, 3)}) == 'unet'
