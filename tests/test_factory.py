"""Checkpoint -> model factory: detect from weights alone, rebuild, load.

This is what lets every inference command load every architecture with no
flag; a regression here silently breaks full-scene evaluation and deployment
for whichever architecture stops being detected.
"""

from pathlib import Path

import pytest
import torch

from sinkholes.models import factory
from sinkholes.models.attention_unet import AttentionUNet
from sinkholes.models.convlstm_unet import CONFIG_KEY, ConvLSTMUNet
from sinkholes.models.factory import (
    NON_PARAMETER_KEYS,
    Architecture,
    architecture_from_flags,
    build_from_checkpoint,
    detect_architecture,
    infer_input_channels,
    strip_non_parameters,
)
from sinkholes.models.unet import UNet

PATCH_H, PATCH_W = 200, 100
REPO_ROOT = Path(__file__).resolve().parent.parent


def checkpoint_for(model, mask_values=(0, 1), extra=None):
    """A state_dict shaped like the trainer writes it: weights + non-parameter keys."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["mask_values"] = list(mask_values)
    if extra:
        sd.update(extra)
    return sd


#: A config for a ConvLSTM built with explicit, non-default settings. Kept
#: independent of the current defaults so these tests keep exercising the
#: "checkpoint overrides the default" path rather than agreeing with it by
#: accident.
CONVLSTM_CONFIG = {
    "n_channels_per_timestep": 1,
    "n_classes": 1,
    "bilinear": False,
    "convlstm_hidden_channels": 128,
    "convlstm_kernel_size": 3,
}


def make_cases():
    # The ConvLSTM config comes from the model itself: duplicating it here would
    # go stale the moment a default moves, and the mismatch would surface as an
    # unrelated-looking load failure.
    convlstm = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1)
    return {
        "unet": (UNet(n_channels=3, n_classes=1, bilinear=False), None),
        "unet_add_attn": (UNet(n_channels=3, n_classes=1, bilinear=False, add_attn=True), None),
        "attention_unet": (AttentionUNet(n_channels=3, n_classes=1, bilinear=False), None),
        "convlstm_unet": (convlstm, {CONFIG_KEY: convlstm.config_dict()}),
    }


@pytest.mark.parametrize("expected", sorted(make_cases()))
def test_detects_architecture_from_weights_alone(expected):
    model, extra = make_cases()[expected]
    assert detect_architecture(checkpoint_for(model, extra=extra)) == expected


@pytest.mark.parametrize("expected", sorted(make_cases()))
def test_rebuilt_model_loads_its_own_checkpoint(expected):
    model, extra = make_cases()[expected]
    sd = checkpoint_for(model, extra=extra)

    loaded = build_from_checkpoint(sd)

    assert loaded.architecture == expected
    assert loaded.mask_values == [0, 1]
    loaded.model.load_state_dict(sd)  # no leftover non-parameter keys


@pytest.mark.parametrize("expected", sorted(make_cases()))
def test_rebuilt_model_reproduces_the_original_output(expected):
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
        **CONVLSTM_CONFIG, "convlstm_hidden_channels": 64,
    }})

    loaded = build_from_checkpoint(sd, n_channels_per_timestep=1)

    assert loaded.model.convlstm_hidden_channels == 64
    loaded.model.load_state_dict(sd)


def test_legacy_convlstm_checkpoint_without_config_is_rebuilt_from_its_weights():
    """Checkpoints predating CONFIG_KEY have no recorded width. The gate conv's
    output axis is 4*hidden, so the weights still say what they need — without
    that, moving the default would break every one of these files."""
    width = 128
    assert width != ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1).convlstm_hidden_channels
    model = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1,
                         convlstm_hidden_channels=width)
    sd = checkpoint_for(model)  # deliberately no CONFIG_KEY
    assert CONFIG_KEY not in sd

    loaded = build_from_checkpoint(sd, n_channels_per_timestep=1)

    assert loaded.architecture == "convlstm_unet", "detected via the convlstm.* keys"
    assert loaded.model.convlstm_hidden_channels == width
    loaded.model.load_state_dict(sd)


def test_infer_convlstm_hidden_reads_the_gate_conv():
    for width in (64, 128, 256):
        sd = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1,
                          convlstm_hidden_channels=width).state_dict()
        assert factory.infer_convlstm_hidden(sd) == width


def test_infer_convlstm_hidden_is_none_on_a_non_convlstm_checkpoint():
    sd = UNet(n_channels=3, n_classes=1, bilinear=False).state_dict()
    assert factory.infer_convlstm_hidden(sd) is None


def test_recorded_config_still_beats_the_inferred_width():
    """Inference is only a fallback: an explicit config must win."""
    model = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1,
                         convlstm_hidden_channels=64)
    sd = checkpoint_for(model, extra={CONFIG_KEY: model.config_dict()})

    loaded = build_from_checkpoint(sd, n_channels_per_timestep=1)

    assert loaded.model.convlstm_hidden_channels == 64
    loaded.model.load_state_dict(sd)


def test_infer_input_channels_reads_the_first_conv():
    for n in (1, 3, 6):
        sd = UNet(n_channels=n, n_classes=1, bilinear=False).state_dict()
        assert infer_input_channels(sd) == n


def test_unet_channels_inferred_when_no_hint_given():
    sd = checkpoint_for(UNet(n_channels=6, n_classes=1, bilinear=False))
    loaded = build_from_checkpoint(sd)
    assert loaded.n_channels == 6
    loaded.model.load_state_dict(sd)


def test_strip_non_parameters_removes_exactly_the_known_keys():
    # Seeded from NON_PARAMETER_KEYS itself rather than a hand-listed pair, so a
    # new architecture's config blob is covered the moment it is declared. A key
    # left behind here fails load_state_dict at the end of a training run.
    extra = {key: {"placeholder": True}
             for key in NON_PARAMETER_KEYS if key != "mask_values"}
    extra[CONFIG_KEY] = dict(CONVLSTM_CONFIG)
    sd = checkpoint_for(UNet(n_channels=3, n_classes=1, bilinear=False), extra=extra)
    n_before = len(sd)
    removed = strip_non_parameters(sd)
    assert set(removed) == set(NON_PARAMETER_KEYS)
    assert len(sd) == n_before - len(NON_PARAMETER_KEYS)


def test_wrong_flag_is_rejected_with_a_readable_error():
    sd = checkpoint_for(ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1),
                        extra={CONFIG_KEY: dict(CONVLSTM_CONFIG)})
    with pytest.raises(SystemExit, match="convlstm_unet"):
        build_from_checkpoint(sd, arch="attention_unet")


def test_wrong_flag_can_be_forced_through():
    sd = checkpoint_for(UNet(n_channels=3, n_classes=1, bilinear=False))
    loaded = build_from_checkpoint(sd, arch="attention_unet", strict_flag_check=False)
    assert loaded.architecture == "attention_unet"


def test_flags_map_to_architectures():
    assert architecture_from_flags() is None
    assert architecture_from_flags(convlstm_unet=False, attn_unet=False) is None
    assert architecture_from_flags(convlstm_unet=True) == "convlstm_unet"
    assert architecture_from_flags(attn_unet=True) == "attention_unet"
    assert architecture_from_flags(add_attn=True) == "unet_add_attn"


def test_combining_architecture_flags_is_rejected():
    with pytest.raises(SystemExit, match="cannot be combined"):
        architecture_from_flags(convlstm_unet=True, attn_unet=True)


def test_plain_unet_is_the_catch_all():
    sd = {"inc.double_conv.0.weight": torch.zeros(64, 4, 3, 3)}
    assert detect_architecture(sd) == "unet"


def test_registering_a_new_architecture_needs_no_call_site_change():
    """The extension point: one register() call and every command picks it up."""
    marker = "my_experimental_block.weight"
    sentinel = object()
    arch = Architecture(
        name="experimental",
        detect=lambda sd: marker in sd,
        build=lambda sd, **kw: sentinel,
        cli_flag="--experimental",
    )
    factory._REGISTRY.insert(0, arch)
    try:
        sd = {marker: torch.zeros(1), "inc.double_conv.0.weight": torch.zeros(64, 3, 3, 3)}
        assert detect_architecture(sd) == "experimental"
        assert build_from_checkpoint(sd).model is sentinel
    finally:
        factory._REGISTRY.remove(arch)
    assert detect_architecture({"inc.double_conv.0.weight": torch.zeros(64, 3, 3, 3)}) == "unet"


# -- the real checkpoints in models/ ----------------------------------------------------

REAL_CHECKPOINTS = {
    "run_v2_best.pt": ("unet", 1),
    "unet_temporal_v1_best.pt": ("unet", 3),
    "convlstm_v1_best.pt": ("convlstm_unet", 1),  # per-timestep channels
}


@pytest.mark.parametrize("name,expected", sorted(REAL_CHECKPOINTS.items()))
def test_real_checkpoint_detects_and_loads(name, expected):
    path = REPO_ROOT / "models" / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    arch, n_ch = expected
    sd = torch.load(path, map_location="cpu")
    loaded = build_from_checkpoint(sd)
    assert loaded.architecture == arch
    assert loaded.n_channels == n_ch
    loaded.model.load_state_dict(sd)
