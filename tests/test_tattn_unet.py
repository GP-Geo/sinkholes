"""Temporal-attention U-Net: shape contract, causality, and skip fusion.

Pure tensor tests. The project's real patch size (200 x 100) is used wherever
the assertion is about spatial correctness, since neither dimension is a power
of two and the decoder has to land back on them exactly.

Two properties get more attention than the rest because breaking either is
silent — no shape error, just a worse model:

- **the present is index T-1**, so the query and the residual must be taken
  from the end of the sequence, not the start;
- **T is not architectural**, so a model trained at k_prevs=5 has to run at
  k_prevs=10 (and under ``--fallback_replicate``).
"""

import argparse

import pytest
import torch
import torch.nn.functional as F

from sinkholes.models.factory import detect_architecture
from sinkholes.models.tattn_unet import (
    CONFIG_KEY,
    MAX_FUSED_SKIPS,
    TemporalAttentionUNet,
    build_tattn_unet,
    pop_model_config,
)
from sinkholes.models.temporal_attention import (
    DEFAULT_TATTN_DIM,
    CausalTemporalAttention,
    _CausalSelfAttentionBlock,
    _TemporalReadoutBlock,
    fuse_over_time,
    temporal_position_encoding,
)

PATCH_H, PATCH_W = 200, 100


def make_model(**kwargs):
    model = TemporalAttentionUNet(**kwargs)
    model.eval()
    return model


# -- forward shapes ---------------------------------------------------------------------

@pytest.mark.parametrize("recurrence", ["none", "convlstm"])
@pytest.mark.parametrize("fuse_skips", [0, MAX_FUSED_SKIPS])
def test_forward_pass_shape(recurrence, fuse_skips):
    model = make_model(n_channels_per_timestep=1, n_classes=1,
                       tattn_recurrence=recurrence, tattn_fuse_skips=fuse_skips)
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 1, PATCH_H, PATCH_W))
    assert logits.shape == (2, 1, PATCH_H, PATCH_W)


def test_flat_and_explicit_inputs_agree():
    """[B, T, H, W] and [B, T, 1, H, W] are the same batch."""
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    flat = torch.randn(2, 4, 32, 16)
    with torch.no_grad():
        assert torch.allclose(model(flat), model(flat.unsqueeze(2)), atol=1e-6)


def test_batch_size_one_keeps_every_dimension():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        logits = model(torch.randn(1, 3, PATCH_H, PATCH_W))
    assert logits.shape == (1, 1, PATCH_H, PATCH_W)


@pytest.mark.parametrize("h,w", [(PATCH_H, PATCH_W), (101, 57)])
def test_decoder_returns_exact_input_spatial_size(h, w):
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        logits = model(torch.randn(1, 2, h, w))
    assert logits.shape[-2:] == (h, w)


def test_nodata_block_layout_is_regrouped_correctly():
    """[B, 2T, H, W] is BLOCK-laid-out: [img_t0..img_t{T-1}, V_t0..V_t{T-1}].

    Guards the shared view+permute against a naive reshape(B, T, 2, H, W),
    which would pair each image with the wrong validity map.
    """
    model = make_model(n_channels_per_timestep=2, n_classes=1)
    b, t, h, w = 2, 3, 32, 16
    imgs = torch.randn(b, t, h, w)
    valid = torch.randint(0, 2, (b, t, h, w)).float()

    flat = torch.cat([imgs, valid], dim=1)
    seq = model._to_sequence(flat)

    assert seq.shape == (b, t, 2, h, w)
    for step in range(t):
        assert torch.equal(seq[:, step, 0], imgs[:, step]), f"image at t={step} misplaced"
        assert torch.equal(seq[:, step, 1], valid[:, step]), f"validity at t={step} misplaced"

    with torch.no_grad():
        assert model(flat).shape == (b, 1, h, w)


# -- T is not architectural -------------------------------------------------------------

@pytest.mark.parametrize("t", [1, 2, 6, 11])
def test_one_model_runs_at_any_sequence_length(t):
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        assert model(torch.randn(1, t, 64, 32)).shape == (1, 1, 64, 32)


def test_a_k5_checkpoint_runs_at_k10():
    """The guarantee --fallback_replicate and cross-k evaluation rest on."""
    trained = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        trained(torch.randn(1, 6, 64, 32))  # k_prevs=5

    state = trained.state_dict()
    rebuilt = TemporalAttentionUNet(n_channels_per_timestep=1, n_classes=1)
    rebuilt.load_state_dict(state)  # no missing/unexpected keys => T is not stored
    rebuilt.eval()
    with torch.no_grad():
        assert rebuilt(torch.randn(1, 11, 64, 32)).shape == (1, 1, 64, 32)


# -- positional encoding ----------------------------------------------------------------

def test_position_encoding_indexes_offset_from_the_present():
    """Row t encodes T-1-t, so the same offset lands on the same vector at any T."""
    short = temporal_position_encoding(6, 16)
    long = temporal_position_encoding(11, 16)
    for t in range(6):
        assert torch.allclose(short[t], long[t + 5], atol=1e-6), (
            f"offset {5 - t} differs between T=6 and T=11; the encoding is indexed "
            f"by absolute position, which would make T architectural"
        )


def test_the_present_always_encodes_the_same_position():
    reference = temporal_position_encoding(2, 16)[-1]
    for t in (1, 3, 6, 11):
        assert torch.allclose(temporal_position_encoding(t, 16)[-1], reference, atol=1e-6)


def test_position_encoding_is_not_a_parameter():
    """A learned table would pass every forward test and silently break k5 -> k10."""
    model = TemporalAttentionUNet()
    keys = [k for k in model.state_dict() if "pos" in k.lower() or "embed" in k.lower()]
    assert keys == [], f"positional encoding leaked into the state dict: {keys}"


def test_position_encoding_rejects_odd_width():
    with pytest.raises(ValueError, match="even dim"):
        temporal_position_encoding(4, 15)


# -- causality --------------------------------------------------------------------------

def test_causal_self_attention_ignores_later_timesteps():
    """Position t must not see t+1. This is what the lower-triangular mask buys.

    Note it is reachable only with tattn_layers > 1: with a single readout layer
    the sole query is already the last position, so the mask is a no-op there.
    """
    torch.manual_seed(0)
    block = _CausalSelfAttentionBlock(dim=8, heads=2).eval()
    x = torch.randn(1, 4, 5, 8)  # (B, P, T, d)
    with torch.no_grad():
        base = block(x)
        perturbed = x.clone()
        perturbed[:, :, 3:] += 10.0
        after = block(perturbed)

    assert torch.allclose(base[:, :, :3], after[:, :, :3], atol=1e-6), (
        "an earlier position changed when a later one did — attention is leaking "
        "backwards in time"
    )
    assert not torch.allclose(base[:, :, 3:], after[:, :, 3:], atol=1e-6)


def test_readout_residual_is_the_present_frame():
    """The residual and query come from index T-1, not index 0.

    Zeroing the output projection and the feed-forward leaves the residual
    alone, so the block must return exactly the last token. An off-by-one here
    reverses the model's meaning with no shape error to catch it.
    """
    block = _TemporalReadoutBlock(dim=8, heads=2).eval()
    with torch.no_grad():
        for param in [*block.out_proj.parameters(), *block.ffn.parameters()]:
            param.zero_()

    x = torch.randn(2, 3, 5, 8)  # (B, P, T, d)
    with torch.no_grad():
        out, weights = block(x)

    assert torch.allclose(out, x[:, :, -1], atol=1e-6)
    assert weights.shape == (2, 3, 2, 5)


# -- attention weights ------------------------------------------------------------------

def test_attention_weights_sum_to_one_over_time():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        _, weights = model.forward_with_attention(torch.randn(2, 5, 64, 32))
    total = weights.sum(dim=2)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_single_timestep_puts_all_weight_on_it():
    """T=1 degenerates to a plain U-Net readout of the only frame."""
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        _, weights = model.forward_with_attention(torch.randn(1, 1, 64, 32))
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


def test_attention_is_content_based():
    """Weights must respond to what is in the frames, not be a fixed average.

    This is what separates the model from learned temporal pooling; if it fails,
    the 1M attention parameters are decoration.
    """
    torch.manual_seed(0)
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        _, a1 = model.forward_with_attention(torch.randn(1, 4, 64, 32))
        _, a2 = model.forward_with_attention(torch.randn(1, 4, 64, 32))
    assert not torch.allclose(a1, a2, atol=1e-4)


# -- temporal order is load-bearing -----------------------------------------------------

def test_reversing_the_sequence_changes_the_output():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    x = torch.randn(1, 3, 64, 32)
    with torch.no_grad():
        forward_order = model(x)
        reversed_order = model(torch.flip(x, dims=[1]))
    assert not torch.allclose(forward_order, reversed_order, atol=1e-4)


def test_the_newest_frame_moves_the_output_most():
    model = make_model(n_channels_per_timestep=1, n_classes=1, tattn_fuse_skips=0)
    x = torch.zeros(1, 3, 64, 32)
    with torch.no_grad():
        base = model(x)
        oldest = x.clone()
        oldest[:, 0] = 1.0
        d_oldest = (model(oldest) - base).abs().mean()
        newest = x.clone()
        newest[:, -1] = 1.0
        d_newest = (model(newest) - base).abs().mean()
    assert d_newest > d_oldest


# -- skip fusion ------------------------------------------------------------------------

def fusion_inputs(b=2, t=4, c=8, h=6, w=5, heads=2, hb=3, wb=2):
    skip = torch.randn(b, t, c, h, w)
    weights = torch.softmax(torch.randn(b, heads, t, hb, wb), dim=2)
    return skip, weights


def test_fusion_matches_an_einsum_reference():
    """The accumulation loop exists to avoid a full (B,T,C,H,W) product, not to
    change the arithmetic."""
    skip, weights = fusion_inputs()
    b, t, c, h, w = skip.shape
    heads, hb, wb = weights.shape[1], weights.shape[3], weights.shape[4]

    got = fuse_over_time(skip, weights)
    upsampled = F.interpolate(
        weights.reshape(b, heads * t, hb, wb), size=(h, w),
        mode="bilinear", align_corners=False,
    ).reshape(b, heads, t, h, w)
    want = torch.einsum(
        "bgthw,btgchw->bgchw", upsampled, skip.reshape(b, t, heads, c // heads, h, w)
    ).reshape(b, c, h, w)
    assert torch.allclose(got, want, atol=1e-5)


def test_fusion_is_a_convex_combination_after_upsampling():
    """Interpolation is linear, so weights still sum to 1 at full resolution —
    which is what keeps a fused skip the same scale as a single-frame one."""
    _, weights = fusion_inputs()
    b, heads, t, hb, wb = weights.shape
    upsampled = F.interpolate(
        weights.reshape(b, heads * t, hb, wb), size=(40, 24),
        mode="bilinear", align_corners=False,
    ).reshape(b, heads, t, 40, 24)
    total = upsampled.sum(dim=2)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_fusion_with_all_weight_on_the_present_is_the_latest_skip():
    """Fusion strictly generalises 'latest timestep only', so tattn_fuse_skips=0
    and a collapsed attention distribution have to agree."""
    skip, weights = fusion_inputs()
    weights = torch.zeros_like(weights)
    weights[:, :, -1] = 1.0
    assert torch.allclose(fuse_over_time(skip, weights), skip[:, -1], atol=1e-5)


def test_fusion_rejects_a_head_count_that_does_not_divide_the_width():
    skip, weights = fusion_inputs(c=9, heads=2)
    with pytest.raises(ValueError, match="does not divide"):
        fuse_over_time(skip, weights)


def test_fuse_skips_changes_the_output():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64, 32)
    outputs = []
    for level in (0, MAX_FUSED_SKIPS):
        torch.manual_seed(0)
        model = make_model(n_channels_per_timestep=1, tattn_fuse_skips=level)
        with torch.no_grad():
            outputs.append(model(x))
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-5)


# -- gradients --------------------------------------------------------------------------

@pytest.mark.parametrize("recurrence", ["none", "convlstm"])
def test_backward_reaches_every_stage(recurrence):
    model = TemporalAttentionUNet(n_channels_per_timestep=1, n_classes=1,
                                  tattn_recurrence=recurrence)
    model.train()
    model(torch.randn(2, 3, 64, 32)).mean().backward()

    watched = [
        "inc.double_conv.0.weight",
        "down4.maxpool_conv.1.double_conv.0.weight",
        "temporal_attn.in_proj.weight",
        # The key path specifically: the value path alone can carry the signal
        # while attention has collapsed to a fixed average, and only a live
        # gradient on k proves the weights themselves are being learned.
        "temporal_attn.readout.k_proj.weight",
        "temporal_attn.out_proj.weight",
        "up4.conv.double_conv.0.weight",
        "outc.conv.weight",
    ]
    if recurrence == "convlstm":
        watched.append("recurrence.conv.weight")

    params = dict(model.named_parameters())
    for name in watched:
        assert name in params, f"{name}: no such parameter"
        grad = params[name].grad
        assert grad is not None, f"{name}: no gradient"
        assert torch.count_nonzero(grad) > 0, f"{name}: gradient is all zeros"


# -- strict interface -------------------------------------------------------------------

def test_indivisible_channel_count_raises_with_shapes_named():
    model = make_model(n_channels_per_timestep=2, n_classes=1)
    with pytest.raises(ValueError) as excinfo:
        model(torch.randn(1, 3, PATCH_H, PATCH_W))
    msg = str(excinfo.value)
    assert "(1, 3, 200, 100)" in msg
    assert "not divisible" in msg
    assert "--add_temporal" in msg
    assert "TemporalAttentionUNet" in msg


def test_three_dimensional_input_raises():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with pytest.raises(ValueError, match="squeeze"):
        model(torch.randn(3, PATCH_H, PATCH_W))


@pytest.mark.parametrize("bad", [0, 3, -1])
def test_unsupported_channels_per_timestep_rejected(bad):
    with pytest.raises(ValueError, match="n_channels_per_timestep"):
        TemporalAttentionUNet(n_channels_per_timestep=bad)


@pytest.mark.parametrize("bad", [-1, MAX_FUSED_SKIPS + 1])
def test_out_of_range_fuse_skips_rejected(bad):
    with pytest.raises(ValueError, match="tattn_fuse_skips"):
        TemporalAttentionUNet(tattn_fuse_skips=bad)


def test_head_count_must_divide_every_skip_width():
    with pytest.raises(ValueError, match="does not divide"):
        TemporalAttentionUNet(tattn_heads=6)


def test_head_count_is_unconstrained_when_skips_are_not_fused():
    model = make_model(tattn_heads=6, tattn_dim=252, tattn_fuse_skips=0)
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 32)).shape == (1, 1, 64, 32)


def test_attention_width_must_divide_by_heads():
    with pytest.raises(ValueError, match="tattn_dim must divide"):
        CausalTemporalAttention(in_channels=16, dim=100, heads=8)


def test_unknown_recurrence_rejected():
    with pytest.raises(ValueError, match="tattn_recurrence"):
        TemporalAttentionUNet(tattn_recurrence="gru")


# -- checkpoint round-trip --------------------------------------------------------------

def test_checkpoint_config_round_trip():
    """A checkpoint carrying CONFIG_KEY rebuilds the exact architecture and loads."""
    model = TemporalAttentionUNet(n_channels_per_timestep=2, n_classes=1, tattn_dim=128,
                                  tattn_heads=4, tattn_layers=2,
                                  tattn_recurrence="convlstm", tattn_fuse_skips=2)
    state = model.state_dict()
    state[CONFIG_KEY] = model.config_dict()

    # Deliberately wrong fallbacks: the stored config must win.
    rebuilt = build_tattn_unet(state, n_channels_per_timestep=1, tattn_dim=DEFAULT_TATTN_DIM,
                               tattn_heads=8, tattn_recurrence="none")
    assert rebuilt.config_dict() == model.config_dict()
    assert CONFIG_KEY not in state, "config key must be popped before load_state_dict"
    rebuilt.load_state_dict(state)


def test_config_records_the_resolved_width_not_the_sentinel():
    assert TemporalAttentionUNet(tattn_dim=0).config_dict()["tattn_dim"] == DEFAULT_TATTN_DIM


def test_pop_model_config_is_a_no_op_on_other_checkpoints():
    assert pop_model_config({"inc.double_conv.0.weight": torch.zeros(1)}) is None


@pytest.mark.parametrize("recurrence", ["none", "convlstm"])
def test_detected_as_tattn_not_convlstm(recurrence):
    """The hybrid contains a ConvLSTM cell; registry order and the 'recurrence.'
    submodule name both exist to stop it loading as the wrong architecture."""
    state = TemporalAttentionUNet(tattn_recurrence=recurrence).state_dict()
    assert detect_architecture(state) == "tattn_unet"


# -- CLI --------------------------------------------------------------------------------

def parse_train_args(argv):
    from sinkholes.training.train import add_arguments

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(argv)


def test_flags_reach_the_parser():
    args = parse_train_args([
        "--tattn_unet", "--tattn_dim", "128", "--tattn_heads", "4",
        "--tattn_layers", "2", "--tattn_recurrence", "convlstm",
        "--tattn_fuse_skips", "2",
    ])
    assert args.tattn_unet and args.tattn_dim == 128 and args.tattn_heads == 4
    assert args.tattn_layers == 2 and args.tattn_recurrence == "convlstm"
    assert args.tattn_fuse_skips == 2


def test_defaults_select_a_pure_attention_model_with_fused_skips():
    args = parse_train_args([])
    assert not args.tattn_unet
    assert args.tattn_recurrence == "none"
    assert args.tattn_fuse_skips == MAX_FUSED_SKIPS


def test_build_model_selects_the_architecture():
    from sinkholes.training.train import build_model

    args = parse_train_args(["--tattn_unet", "--add_temporal", "--k_prevs", "5"])
    model, num_c = build_model(args, torch.device("cpu"))
    assert isinstance(model, TemporalAttentionUNet)
    assert model.n_channels_per_timestep == 1 and num_c == 6


def test_build_model_requires_a_temporal_stack():
    from sinkholes.training.train import build_model

    args = parse_train_args(["--tattn_unet"])
    with pytest.raises(SystemExit, match="--add_temporal"):
        build_model(args, torch.device("cpu"))


def test_build_model_rejects_two_architecture_flags():
    from sinkholes.training.train import build_model

    args = parse_train_args(["--tattn_unet", "--convlstm_unet", "--add_temporal"])
    with pytest.raises(SystemExit, match="cannot be combined"):
        build_model(args, torch.device("cpu"))


def test_resume_fingerprint_names_the_architecture():
    from sinkholes.training.resume import architecture_name, run_config

    args = parse_train_args(["--tattn_unet", "--add_temporal", "--k_prevs", "5",
                             "--tattn_fuse_skips", "2"])
    assert architecture_name(args) == "tattn_unet"
    config = run_config(args)
    assert config["architecture"] == "tattn_unet"
    assert config["tattn_fuse_skips"] == 2
    assert config["tattn_dim"] == DEFAULT_TATTN_DIM


class TinyTemporalSet(list):
    """A handful of {image, mask} samples shaped like the temporal loader's."""

    mask_values = [0, 1]

    @classmethod
    def make(cls, n=4, t=3, size=32, seed=0):
        gen = torch.Generator().manual_seed(seed)
        return cls(
            {"image": torch.rand(t, size, size, generator=gen),
             "mask": (torch.rand(size, size, generator=gen) > 0.7).long()}
            for _ in range(n)
        )


@pytest.mark.slow
def test_a_real_epoch_runs_end_to_end(tmp_path):
    """Through the actual training loop, not a hand-rolled one.

    Covers the three integration points that no unit test reaches: the batch
    shape assert accepts (B, T, H, W), checkpoint_state() writes the config
    blob, and the end-of-run best-weight restore strips it again before
    load_state_dict — which used to be hard-coded to the ConvLSTM's key.
    """
    from sinkholes.models.factory import build_from_checkpoint
    from sinkholes.training.train import build_model, train_model

    args = parse_train_args([
        "--tattn_unet", "--add_temporal", "--k_prevs", "2",
        "--epochs", "2", "--batch_size", "2", "--learning-rate", "1e-3",
        "--patch_size", "32", "32", "--sample_every", "0", "--save_best_only",
        "--job_name", "tiny", "--no-reporter",
    ])
    model, _ = build_model(args, torch.device("cpu"))
    train_model(args, model, torch.device("cpu"),
                TinyTemporalSet.make(), TinyTemporalSet.make(seed=1), None,
                str(tmp_path))

    best = tmp_path / "checkpoints" / "best.pt"
    assert best.exists(), "no best.pt written"
    state = torch.load(best, map_location="cpu", weights_only=False)
    assert CONFIG_KEY in state, "the config blob was not written into the checkpoint"

    # The full inference path: detect, rebuild, load — with no flags at all.
    loaded = build_from_checkpoint(state, n_classes=1)
    assert loaded.architecture == "tattn_unet"
    loaded.model.load_state_dict(state)


def test_resume_fingerprint_leaves_other_architectures_alone():
    """The new keys must be None elsewhere, so an existing ConvLSTM run keeps
    the fingerprint it had before they existed."""
    from sinkholes.training.resume import run_config

    args = parse_train_args(["--convlstm_unet", "--add_temporal", "--k_prevs", "5"])
    config = run_config(args)
    assert config["architecture"] == "convlstm_unet"
    assert all(config[k] is None for k in
               ("tattn_dim", "tattn_heads", "tattn_layers",
                "tattn_recurrence", "tattn_fuse_skips"))
