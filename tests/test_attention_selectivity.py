"""The attention must actually select frames, and keep selecting after training.

Every tattn checkpoint trained before 2026-08-19 produces exactly uniform
weights (docs/ATTENTION_COLLAPSE.md). The cause is measurable at init and is
what these tests pin:

- the tokens attention sees are dominated by a component shared across the whole
  sequence, so the frame-to-frame signal is ~0.1% of what reaches q/k,
- with unnormalised q/k the softmax's sharpness rides on projection magnitude,
  so the same block goes uniform at one learning rate and one-hot at another.

The existing ``test_attention_is_content_based`` did not catch any of this: it
runs on a randomly initialised model and asks only whether the weights *move*.
They do move — by 1e-5, around a uniform mean. The assertions here are about
*how much* selectivity there is, which is the property that was actually lost.
"""

import pytest
import torch

from sinkholes.models.tattn_unet import (
    CONFIG_KEY,
    TemporalAttentionUNet,
    build_tattn_unet,
    infer_attention_variant,
)
from sinkholes.models.temporal_attention import (
    DEFAULT_LOGIT_SCALE,
    _CausalSelfAttentionBlock,
    causal_temporal_mean,
    temporal_mean,
)

T = 11


def make(**kw):
    torch.manual_seed(42)
    m = TemporalAttentionUNet(n_channels_per_timestep=1, n_classes=1,
                              tattn_fuse_skips=0, **kw)
    m.eval()
    return m


def effective_frames(model, x, **kw):
    """exp(entropy) of the weights, per (sample, head, pixel), then averaged.

    Equals T exactly when attention is uniform. Averaging exp(H) rather than
    exponentiating the mean H is deliberate: the latter would let a mixture of
    collapsed and spread-out pixels read as moderate selectivity.
    """
    with torch.no_grad():
        _, w = model.forward_with_attention(x, **kw)
    ent = -(w.clamp_min(1e-12).log() * w).sum(dim=2)
    return float(ent.exp().mean())


# -- the defect ---------------------------------------------------------------------------

def test_unfixed_attention_is_uniform_at_initialisation():
    """The bug, pinned. If this stops holding, the fix's baseline has moved."""
    m = make(tattn_contrast=False, tattn_qk_norm=False)
    torch.manual_seed(0)
    eff = effective_frames(m, torch.randn(2, T, 64, 32))
    assert eff > 0.98 * T, (
        f"the unfixed block used {eff:.2f}/{T} frames; it is supposed to be "
        f"degenerate here, so the collapse baseline has changed"
    )


def test_fixed_attention_is_selective_at_initialisation():
    m = make()
    torch.manual_seed(0)
    eff = effective_frames(m, torch.randn(2, T, 64, 32))
    assert eff < 0.75 * T, f"attention used {eff:.2f}/{T} frames — barely better than uniform"


@pytest.mark.parametrize("contrast,qk_norm", [(True, False), (False, True)])
def test_neither_half_is_sufficient_alone(contrast, qk_norm):
    """Measured on real data: each alone leaves >90% of uniform, both give ~40%."""
    torch.manual_seed(0)
    x = torch.randn(2, T, 64, 32)
    half = effective_frames(make(tattn_contrast=contrast, tattn_qk_norm=qk_norm), x)
    both = effective_frames(make(), x)
    assert both < half, (
        f"contrast={contrast}, qk_norm={qk_norm} alone gave {half:.2f}/{T} but both "
        f"gave {both:.2f}/{T}; the two are supposed to compose"
    )


# -- why it works -------------------------------------------------------------------------

def test_contrast_removes_the_shared_component():
    """What q/k see must be the frame-to-frame difference, not the shared bulk."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, T, 16) + 50.0 * torch.randn(2, 5, 1, 16)
    centred = x - temporal_mean(x)
    assert centred.abs().mean() < 0.1 * x.abs().mean()
    assert torch.allclose(centred.mean(dim=2), torch.zeros(2, 5, 16), atol=1e-4)


def test_temporal_mean_ignores_padded_frames():
    x = torch.randn(2, 3, 4, 8)
    valid = torch.tensor([[False, True, True, True]] * 2)
    got = temporal_mean(x, valid)
    assert torch.allclose(got, x[:, :, 1:].mean(dim=2, keepdim=True), atol=1e-6)


def test_qk_norm_makes_selectivity_independent_of_projection_scale():
    """The property that stops the collapse being a function of learning rate.

    Blowing q/k up 50x must not change the weights when they are unit-normed;
    without QK-norm the same change saturates the softmax.
    """
    torch.manual_seed(0)
    x = torch.randn(2, T, 64, 32)

    def after_scaling(**kw):
        m = make(**kw)
        with torch.no_grad():
            _, before = m.forward_with_attention(x)
            for p in (m.temporal_attn.readout.q_proj, m.temporal_attn.readout.k_proj):
                p.weight.mul_(50.0)
                p.bias.mul_(50.0)
            _, after = m.forward_with_attention(x)
        return float((before - after).abs().max())

    assert after_scaling() < 1e-4, "QK-norm did not decouple the logits from q/k magnitude"
    assert after_scaling(tattn_contrast=False, tattn_qk_norm=False) > 1e-2


def test_padded_frames_still_get_zero_weight_with_the_fix():
    m = make()
    valid = torch.tensor([[False, False, True, True]])
    with torch.no_grad():
        _, w = m.forward_with_attention(torch.randn(1, 4, 64, 32), valid=valid)
    assert torch.all(w[:, :, :2] == 0.0)
    total = w.sum(dim=2)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_the_value_path_still_sees_the_whole_token():
    """Contrast is for selection only; the answer must still carry content.

    A constant shift of the whole sequence changes nothing about which frame is
    interesting, but it does change what the frames say — so the logits must move.
    """
    m = make()
    x = torch.randn(1, T, 64, 32)
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x + 0.5), atol=1e-4)


def test_logit_scale_starts_at_the_documented_temperature():
    m = make()
    assert float(m.temporal_attn.readout.logit_scale.detach().exp()) == pytest.approx(
        DEFAULT_LOGIT_SCALE, rel=1e-5)


def test_attention_still_responds_to_content():
    torch.manual_seed(0)
    m = make()
    with torch.no_grad():
        _, a = m.forward_with_attention(torch.randn(1, 4, 64, 32))
        _, b = m.forward_with_attention(torch.randn(1, 4, 64, 32))
    # Not merely "different" — different by far more than the 1e-5 the broken
    # block managed, which is what let the old test pass on a dead mechanism.
    assert (a - b).abs().mean() > 1e-3


# -- causality is not sacrificed to the contrast ------------------------------------------

@pytest.mark.parametrize("contrast", [True, False])
def test_contrast_does_not_leak_future_frames_into_self_attention(contrast):
    """Centring on the whole-sequence mean would make every position depend on
    the ones after it, and the lower-triangular mask cannot undo that — the leak
    is already inside the vectors it multiplies.

    Asserted as EXACT equality on purpose. The whole-sequence mean leaked 2.4e-7
    here, which slipped under the 1e-6 tolerance of the causality test in
    test_tattn_unet.py: it passed by magnitude, not by correctness.
    """
    torch.manual_seed(0)
    block = _CausalSelfAttentionBlock(dim=8, heads=2, contrast=contrast,
                                      qk_norm=True).eval()
    x = torch.randn(1, 4, 5, 8)
    perturbed = x.clone()
    perturbed[:, :, 3:] += 10.0
    with torch.no_grad():
        before, after = block(x), block(perturbed)
    assert torch.equal(before[:, :, :3], after[:, :, :3]), (
        "an earlier position moved when a later one changed — attention is "
        "reading forwards in time"
    )
    assert not torch.allclose(before[:, :, 3:], after[:, :, 3:], atol=1e-6)


def test_causal_mean_is_a_prefix_mean():
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 4, 3)
    got = causal_temporal_mean(x)
    for t in range(4):
        assert torch.allclose(got[0, 0, t], x[0, 0, :t + 1].mean(dim=0), atol=1e-6)


def test_causal_mean_skips_padded_frames():
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 4, 3)
    valid = torch.tensor([[False, True, True, True]])
    got = causal_temporal_mean(x, valid)
    assert torch.allclose(got[0, 0, 3], x[0, 0, 1:4].mean(dim=0), atol=1e-6)


def test_the_readout_may_use_the_whole_sequence_mean():
    """No leak there: its only query is the present, so "all frames" and "all
    frames at or before the query" are the same set."""
    x = torch.randn(2, 3, 5, 8)
    assert torch.allclose(temporal_mean(x)[:, :, 0], x.mean(dim=2), atol=1e-6)


# -- long sequences -----------------------------------------------------------------------

def test_selectivity_survives_a_long_sequence():
    """A 41-frame lookback must not force the softmax back toward uniform."""
    m = make()
    torch.manual_seed(0)
    offsets = torch.arange(40, -1, -1, dtype=torch.float32)
    eff = effective_frames(m, torch.randn(1, 41, 64, 32), offsets=offsets)
    assert eff < 0.5 * 41, f"used {eff:.1f}/41 frames — attention diluted by length"


# -- checkpoint compatibility -------------------------------------------------------------

def test_variant_is_readable_from_the_weights():
    for contrast in (True, False):
        for qk in (True, False):
            sd = make(tattn_contrast=contrast, tattn_qk_norm=qk).state_dict()
            assert infer_attention_variant(sd) == (contrast, qk)


def test_infer_returns_none_without_attention_weights():
    assert infer_attention_variant({"inc.double_conv.0.weight": torch.zeros(1)}) is None
    assert infer_attention_variant(None) is None


def test_a_pre_fix_checkpoint_rebuilds_as_itself():
    """The compatibility guarantee: an old config blob has neither key, and
    guessing today's default would build a model the file cannot fill."""
    legacy = make(tattn_contrast=False, tattn_qk_norm=False)
    sd = legacy.state_dict()
    cfg = legacy.config_dict()
    del cfg["tattn_contrast"], cfg["tattn_qk_norm"]   # exactly what an old run wrote
    sd[CONFIG_KEY] = cfg

    rebuilt = build_tattn_unet(sd)
    assert (rebuilt.tattn_contrast, rebuilt.tattn_qk_norm) == (False, False)
    rebuilt.load_state_dict(sd)   # strict: raises on any missing or unexpected key


def test_a_fixed_checkpoint_round_trips():
    m = make()
    sd = m.state_dict()
    sd[CONFIG_KEY] = m.config_dict()
    rebuilt = build_tattn_unet(sd, tattn_contrast=False, tattn_qk_norm=False)
    assert (rebuilt.tattn_contrast, rebuilt.tattn_qk_norm) == (True, True)
    rebuilt.load_state_dict(sd)


def test_config_records_the_variant():
    cfg = make().config_dict()
    assert cfg["tattn_contrast"] is True and cfg["tattn_qk_norm"] is True


def test_gradients_reach_the_new_parameters():
    m = TemporalAttentionUNet(n_channels_per_timestep=1, n_classes=1, tattn_fuse_skips=0)
    m.train()
    m(torch.randn(2, 3, 64, 32)).mean().backward()
    params = dict(m.named_parameters())
    for name in ("temporal_attn.readout.contrast_norm.weight",
                 "temporal_attn.readout.logit_scale",
                 "temporal_attn.readout.k_proj.weight"):
        assert name in params, f"{name}: no such parameter"
        g = params[name].grad
        assert g is not None and torch.count_nonzero(g) > 0, f"{name}: dead gradient"


# -- CLI / resume -------------------------------------------------------------------------

def parse(argv):
    import argparse

    from sinkholes.training.train import add_arguments
    p = argparse.ArgumentParser()
    add_arguments(p)
    return p.parse_args(argv)


def test_flags_default_to_the_fix_being_on():
    a = parse([])
    assert a.tattn_contrast is True and a.tattn_qk_norm is True


def test_legacy_behaviour_is_reachable_from_the_cli():
    a = parse(["--no-tattn_contrast", "--no-tattn_qk_norm"])
    assert a.tattn_contrast is False and a.tattn_qk_norm is False


def test_build_model_honours_the_flags():
    from sinkholes.training.train import build_model

    a = parse(["--tattn_unet", "--add_temporal", "--k_prevs", "5", "--no-tattn_qk_norm"])
    m, _ = build_model(a, torch.device("cpu"))
    assert m.tattn_contrast is True and m.tattn_qk_norm is False


def test_resume_fingerprint_covers_the_variant():
    from sinkholes.training.resume import run_config

    cfg = run_config(parse(["--tattn_unet", "--add_temporal", "--k_prevs", "5"]))
    assert cfg["tattn_contrast"] is True and cfg["tattn_qk_norm"] is True
    other = run_config(parse(["--convlstm_unet", "--add_temporal", "--k_prevs", "5"]))
    assert other["tattn_contrast"] is None and other["tattn_qk_norm"] is None
