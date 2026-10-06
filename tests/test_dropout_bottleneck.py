"""Dropout on the ConvLSTM's recurrent summary: the fallback contract.

This option exists to be *rolled back by not passing it*. That is only true if
``dropout_bottleneck=0.0`` reproduces the previous network exactly and leaves
every checkpoint loadable, so the tests here pin four things:

  1. the state dict is byte-identical with the option on or off -- ``Dropout2d``
     is a sibling module, never a ``Sequential`` entry, so no key shifts and
     ``factory.py``'s architecture detection cannot be disturbed;
  2. at 0.0 the module is ``nn.Identity`` and the forward pass is unchanged;
  3. inference is unaffected at any rate, because every command calls ``eval()``;
  4. a checkpoint written before the option existed rebuilds at 0.0, and a
     resume cannot cross a change to it.

Small models on 32x32 throughout: nothing here is an assertion about spatial
correctness, and the real geometry costs 43M parameters per test.
"""

import argparse

import pytest
import torch
import torch.nn as nn

from sinkholes.models.convlstm_unet import (
    CONFIG_KEY,
    ConvLSTMUNet,
    build_convlstm_unet,
)
from sinkholes.models.factory import build_from_checkpoint, detect_architecture
from sinkholes.training import resume as R
from sinkholes.training.train import add_arguments, build_model

H = W = 32
T = 3
HIDDEN = 8


def make_model(dropout=0.0, **kwargs):
    torch.manual_seed(0)
    return ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1,
                        convlstm_hidden_channels=HIDDEN,
                        dropout_bottleneck=dropout, **kwargs)


def make_args(**overrides) -> argparse.Namespace:
    """Real CLI defaults, so these move with the parser rather than a copy."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def batch(seed=0):
    return torch.randn(2, T, 1, H, W, generator=torch.Generator().manual_seed(seed))


# -- 1. the checkpoint contract ----------------------------------------------------------

def test_state_dict_keys_are_identical_with_and_without_dropout():
    """The whole reason the dropout is a sibling module and not a Sequential
    entry. AttentionUNet is told apart from every other architecture by the
    index shift ITS Dropout2d causes (factory.py ATTENTION_UNET_MARKER), so a
    dropout inserted into a Sequential here would both break every existing
    ConvLSTM checkpoint and make it detect as the wrong architecture."""
    off, on = make_model(0.0).state_dict(), make_model(0.4).state_dict()
    assert list(off) == list(on)
    assert all(off[k].shape == on[k].shape for k in off)


def test_dropout_does_not_disturb_architecture_detection():
    for rate in (0.0, 0.25):
        sd = make_model(rate).state_dict()
        assert detect_architecture(sd) == "convlstm_unet"


def test_a_dropout_trained_checkpoint_loads_into_the_plain_model():
    """The fallback in its bluntest form: weights trained with dropout must load
    into a model built without it, since dropout adds no parameters."""
    trained = make_model(0.3)
    plain = make_model(0.0)
    plain.load_state_dict(trained.state_dict())  # raises on any mismatch


# -- 2. 0.0 is the previous network exactly ----------------------------------------------

def test_default_is_off():
    assert ConvLSTMUNet(n_channels_per_timestep=1).dropout_bottleneck == 0.0
    assert make_args().dropout_bottleneck == 0.0


def test_zero_builds_identity_and_leaves_the_forward_pass_unchanged():
    model = make_model(0.0)
    assert isinstance(model.bottleneck_drop, nn.Identity)
    model.train()
    x = batch()
    # Identity is a literal no-op, so even in train mode the pass is deterministic.
    assert torch.equal(model(x), model(x))


def test_above_zero_builds_dropout2d_at_the_requested_rate():
    model = make_model(0.15)
    assert isinstance(model.bottleneck_drop, nn.Dropout2d)
    assert model.bottleneck_drop.p == pytest.approx(0.15)


@pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
def test_rejects_a_rate_outside_zero_to_one(rate):
    with pytest.raises(ValueError, match="dropout_bottleneck"):
        make_model(rate)


# -- 3. training vs inference ------------------------------------------------------------

def test_inference_is_unaffected_at_any_rate():
    """Every inference path calls eval() (scenes, patch_test, predict,
    reconstruct, inspect_run, attention_probe), so a dropout checkpoint must
    predict exactly what the same weights predict without dropout."""
    plain, dropped = make_model(0.0), make_model(0.5)
    dropped.load_state_dict(plain.state_dict())
    plain.eval(), dropped.eval()
    x = batch(seed=1)
    with torch.no_grad():
        assert torch.equal(plain(x), dropped(x))


def test_dropout_is_active_in_training_mode():
    model = make_model(0.5)
    model.train()
    x = batch(seed=2)
    torch.manual_seed(0)
    first = model(x)
    second = model(x)
    assert not torch.equal(first, second), "two train-mode passes should differ"


def test_it_drops_whole_channels_of_the_hidden_state_before_the_projection():
    """Placement and granularity in one assertion: the tensor entering
    convlstm_proj must still be `hidden` channels wide (so the dropout ran on
    the recurrent state, not on the 1024-wide projection of it) and whole
    channels of it must be zero (Dropout2d, not element-wise Dropout -- on a
    12x6 map single zeroed elements are just filled in from next door)."""
    model = make_model(0.5)
    model.train()
    seen = {}

    def hook(_module, inputs, _output):
        seen["x"] = inputs[0].detach().clone()

    handle = model.convlstm_proj.register_forward_hook(hook)
    torch.manual_seed(0)
    model(batch(seed=3))
    handle.remove()

    entering = seen["x"]
    assert entering.shape[1] == HIDDEN, "dropout must run before convlstm_proj widens it"
    per_channel = entering.abs().amax(dim=(2, 3))          # (B, hidden)
    assert (per_channel == 0).any(), "no channel was dropped at p=0.5"
    assert (per_channel > 0).any(), "every channel was dropped"


# -- 4. config, rebuild, resume ----------------------------------------------------------

def test_config_dict_records_the_rate_and_rebuilds_it():
    model = make_model(0.2)
    sd = model.state_dict()
    sd[CONFIG_KEY] = model.config_dict()
    assert sd[CONFIG_KEY]["dropout_bottleneck"] == pytest.approx(0.2)

    rebuilt = build_convlstm_unet(sd, dropout_bottleneck=0.9)  # the config must win
    assert rebuilt.dropout_bottleneck == pytest.approx(0.2)
    assert CONFIG_KEY not in sd, "build_convlstm_unet pops it, ready for load_state_dict"
    rebuilt.load_state_dict(sd)


def test_a_checkpoint_predating_the_option_rebuilds_at_zero():
    """Old ConvLSTM checkpoints carry a config blob without this key. They were
    trained with no dropout, so that is what they must come back as."""
    model = make_model(0.3)
    legacy = {k: v for k, v in model.config_dict().items() if k != "dropout_bottleneck"}
    sd = model.state_dict()
    sd[CONFIG_KEY] = legacy

    loaded = build_from_checkpoint(sd, n_channels_per_timestep=1)

    assert loaded.model.dropout_bottleneck == 0.0
    assert isinstance(loaded.model.bottleneck_drop, nn.Identity)
    loaded.model.load_state_dict(sd)


def test_run_config_records_it_only_for_the_convlstm():
    convlstm = make_args(convlstm_unet=True, add_temporal=True, dropout_bottleneck=0.15)
    assert R.run_config(convlstm)["dropout_bottleneck"] == pytest.approx(0.15)
    assert R.run_config(make_args())["dropout_bottleneck"] is None


def test_resume_is_refused_across_a_change_to_the_rate():
    assert "dropout_bottleneck" in R.STRICT_CONFIG_KEYS
    base = dict(convlstm_unet=True, add_temporal=True)
    saved = R.run_config(make_args(**base, dropout_bottleneck=0.0))
    now = R.run_config(make_args(**base, dropout_bottleneck=0.2))
    with pytest.raises(SystemExit, match="dropout_bottleneck"):
        R.check_config_compatible(saved, now, "resume.pt")


def test_a_config_written_before_the_key_existed_still_resumes():
    """check_config_compatible skips STRICT keys absent from the saved config,
    which is what keeps every run trained before 2026-09-08 resumable."""
    args = make_args(convlstm_unet=True, add_temporal=True)
    current = R.run_config(args)
    saved = {k: v for k, v in current.items() if k != "dropout_bottleneck"}
    assert R.check_config_compatible(saved, current, "resume.pt") == []


# -- 5. the CLI wiring -------------------------------------------------------------------

def test_build_model_passes_the_flag_through():
    args = make_args(convlstm_unet=True, add_temporal=True, k_prevs=2,
                     convlstm_hidden=HIDDEN, dropout_bottleneck=0.25)
    model, _ = build_model(args, torch.device("cpu"))
    assert model.dropout_bottleneck == pytest.approx(0.25)


@pytest.mark.parametrize("flags", [{}, {"attn_unet": True}, {"add_attn": True},
                                   {"tattn_unet": True, "add_temporal": True}])
def test_the_flag_is_refused_where_there_is_no_such_bottleneck(flags):
    """Silently ignoring it would put a run in the registry whose recorded
    flags do not describe the network that was trained."""
    args = make_args(dropout_bottleneck=0.2, **flags)
    with pytest.raises(SystemExit, match="dropout_bottleneck"):
        build_model(args, torch.device("cpu"))
