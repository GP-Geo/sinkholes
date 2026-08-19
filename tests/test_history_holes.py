"""Sequences with holes: real offsets, padding masks, and hole-tolerant chains.

Two mechanisms, tested against one property each.

**Offsets.** The positional encoding used to index list positions, which is
only the same as elapsed time on a gap-free chain. With holes it is not, and
getting it wrong is silent — no shape error, just a model told the wrong ages.

**Masking.** A batch padded to a common length must produce *exactly* the
answer the unpadded sequence would. That is the property worth pinning, because
a leak here also looks like nothing at all: the model simply averages in frames
that do not exist.

Both default to off, and ``test_defaults_reproduce_the_gap_free_behaviour``
holds the line that says so — every trained checkpoint depends on it.
"""

import numpy as np
import pytest
import torch

from sinkholes.meta import (
    frame_groups_of,
    parse_history_schedule,
    select_history,
)
from sinkholes.models.tattn_unet import TemporalAttentionUNet
from sinkholes.models.temporal_attention import (
    CausalTemporalAttention,
    _CausalSelfAttentionBlock,
    check_valid_mask,
    fuse_over_time,
    temporal_position_encoding,
)


def make_model(**kwargs):
    torch.manual_seed(0)
    model = TemporalAttentionUNet(n_channels_per_timestep=1, n_classes=1, **kwargs)
    model.eval()
    return model


# -- positional encoding over real offsets -----------------------------------------------

def test_defaults_reproduce_the_gap_free_behaviour():
    """The contract every existing checkpoint rests on: no offsets => no change."""
    for t in (1, 3, 6, 11, 41):
        implicit = temporal_position_encoding(t, 16)
        explicit = temporal_position_encoding(t, 16, offsets=torch.arange(t - 1, -1, -1))
        assert torch.allclose(implicit, explicit, atol=1e-6), f"T={t}"


def test_offsets_encode_age_not_list_position():
    """A gappy chain must encode the ages it really has.

    [-4, -2, 0] is three frames whose middle one is *two* slots old. Encoded by
    position it would look one slot old — indistinguishable from the dense
    [-2, -1, 0], which is a different physical history.
    """
    gappy = temporal_position_encoding(3, 16, offsets=torch.tensor([4.0, 2.0, 0.0]))
    dense = temporal_position_encoding(3, 16)  # offsets 2, 1, 0
    assert not torch.allclose(gappy[1], dense[1], atol=1e-6)

    # The same age must land on the same vector whatever sequence it came from.
    reference = temporal_position_encoding(5, 16)  # offsets 4..0
    assert torch.allclose(gappy[0], reference[0], atol=1e-6), "offset 4 moved"
    assert torch.allclose(gappy[1], reference[2], atol=1e-6), "offset 2 moved"


def test_per_sample_offsets_give_a_batched_encoding():
    pe = temporal_position_encoding(3, 8, offsets=torch.tensor([[4.0, 2.0, 0.0],
                                                               [3.0, 1.0, 0.0]]))
    assert pe.shape == (2, 3, 8)
    assert torch.allclose(pe[0], temporal_position_encoding(3, 8, offsets=[4.0, 2.0, 0.0]))


def test_offsets_must_match_the_sequence_length():
    with pytest.raises(ValueError, match="cover 2 timesteps"):
        temporal_position_encoding(3, 8, offsets=torch.tensor([1.0, 0.0]))


# -- the padding mask ---------------------------------------------------------------------

def test_padding_is_exactly_equivalent_to_a_shorter_sequence():
    """The property the whole mask exists for.

    A 3-frame history padded out to 6 must give the same logits as the bare
    3-frame one. If it does not, the padding is leaking into the answer.
    """
    model = make_model(tattn_fuse_skips=0)
    torch.manual_seed(1)
    real = torch.randn(2, 3, 64, 32)
    junk = torch.randn(2, 3, 64, 32)
    padded = torch.cat([junk, real], dim=1)                 # 3 pads, then the real history
    offsets = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    valid = torch.tensor([[False, False, False, True, True, True]] * 2)

    with torch.no_grad():
        want = model(real, offsets=torch.tensor([2.0, 1.0, 0.0]))
        got = model(padded, offsets=offsets, valid=valid)
    assert torch.allclose(want, got, atol=1e-5)


def test_padded_frames_receive_exactly_zero_attention():
    model = make_model()
    valid = torch.tensor([[False, True, True, True]])
    with torch.no_grad():
        _, weights = model.forward_with_attention(torch.randn(1, 4, 64, 32), valid=valid)
    assert torch.all(weights[:, :, 0] == 0.0), "a masked timestep drew attention"
    total = weights.sum(dim=2)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_masking_survives_skip_fusion():
    """fuse_over_time needs no mask of its own — it inherits the zeros.

    Worth pinning explicitly: the fusion is a *separate* consumer of the same
    weights, and it would be entirely natural to forget it.
    """
    model = make_model(tattn_fuse_skips=4)
    torch.manual_seed(2)
    real = torch.randn(1, 3, 64, 32)
    padded = torch.cat([torch.randn(1, 2, 64, 32), real], dim=1)
    with torch.no_grad():
        want = model(real, offsets=torch.tensor([2.0, 1.0, 0.0]))
        got = model(padded,
                    offsets=torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
                    valid=torch.tensor([[False, False, True, True, True]]))
    assert torch.allclose(want, got, atol=1e-5)


def test_recurrence_skips_padded_steps():
    """The hybrid's ConvLSTM carries state forward, so a mask alone cannot save it."""
    model = make_model(tattn_recurrence="convlstm", tattn_fuse_skips=0)
    torch.manual_seed(3)
    real = torch.randn(1, 3, 64, 32)
    padded = torch.cat([torch.randn(1, 2, 64, 32), real], dim=1)
    with torch.no_grad():
        want = model(real, offsets=torch.tensor([2.0, 1.0, 0.0]))
        got = model(padded,
                    offsets=torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
                    valid=torch.tensor([[False, False, True, True, True]]))
    assert torch.allclose(want, got, atol=1e-5)


def test_all_padding_but_the_present_does_not_produce_nan():
    """The degenerate case: one real frame in a long padded grid.

    With layers > 1 the causal self-attention gives every padded query an
    all -inf row unless the diagonal is kept, and one NaN there reaches the
    present through the residual.
    """
    model = make_model(tattn_layers=3)
    valid = torch.tensor([[False, False, False, True]])
    with torch.no_grad():
        logits = model(torch.randn(1, 4, 64, 32), valid=valid)
    assert torch.isfinite(logits).all()


def test_masking_out_the_present_is_refused():
    model = make_model()
    with pytest.raises(ValueError, match="present frame"):
        model(torch.randn(1, 3, 64, 32), valid=torch.tensor([[True, True, False]]))


def test_valid_mask_shape_is_checked():
    with pytest.raises(ValueError, match=r"\(2, 3\)"):
        check_valid_mask(torch.ones(3, dtype=torch.bool), 2, 3)


def test_self_attention_block_ignores_padded_keys():
    torch.manual_seed(0)
    block = _CausalSelfAttentionBlock(dim=8, heads=2).eval()
    x = torch.randn(1, 4, 5, 8)
    valid = torch.tensor([[False, True, True, True, True]])
    perturbed = x.clone()
    perturbed[:, :, 0] += 50.0
    with torch.no_grad():
        assert torch.allclose(block(x, valid)[:, :, 1:],
                              block(perturbed, valid)[:, :, 1:], atol=1e-5)


def test_a_batch_may_mix_different_hole_patterns():
    """Samples in one batch have different histories; masks are per sample."""
    model = make_model()
    valid = torch.tensor([[False, False, True, True],
                          [False, True, True, True]])
    with torch.no_grad():
        _, weights = model.forward_with_attention(torch.randn(2, 4, 64, 32), valid=valid)
    assert torch.all(weights[0, :, :2] == 0.0)
    assert torch.all(weights[1, :, 0] == 0.0)
    assert torch.any(weights[1, :, 1] > 0.0)


def test_fuse_over_time_drops_masked_steps_by_arithmetic():
    """Zero weight times a replicated pad is zero — no second mask needed."""
    skip = torch.randn(1, 4, 8, 6, 5)
    weights = torch.zeros(1, 2, 4, 3, 2)
    weights[:, :, 2:] = 0.5
    fused = fuse_over_time(skip, weights)
    assert torch.allclose(fused, 0.5 * (skip[:, 2] + skip[:, 3]), atol=1e-5)


# -- long sequences -----------------------------------------------------------------------

def test_a_k10_checkpoint_runs_at_a_40_slot_lookback():
    """Stage 0's premise: no retraining, no weight change, just a longer input."""
    trained = make_model()
    with torch.no_grad():
        trained(torch.randn(1, 11, 64, 32))
    state = trained.state_dict()

    rebuilt = TemporalAttentionUNet(n_channels_per_timestep=1, n_classes=1)
    rebuilt.load_state_dict(state)   # no missing keys => nothing is T-shaped
    rebuilt.eval()
    offsets = torch.arange(40, -1, -1, dtype=torch.float32)
    with torch.no_grad():
        logits, weights = rebuilt.forward_with_attention(
            torch.randn(1, 41, 64, 32), offsets=offsets,
        )
    assert logits.shape == (1, 1, 64, 32)
    assert weights.shape[2] == 41


def test_offsets_do_not_enter_the_state_dict():
    """A learned age embedding would pass every test above and break k5 -> k40."""
    keys = [k for k in TemporalAttentionUNet().state_dict()
            if "offset" in k.lower() or "pos" in k.lower()]
    assert keys == []


# -- hole-tolerant chain selection ---------------------------------------------------------

def test_schedule_dense_is_every_slot():
    assert parse_history_schedule("dense", 5) == [1, 2, 3, 4, 5]


def test_schedule_thins_the_old_end():
    got = parse_history_schedule("1:6,2:12,4:40", 40)
    assert got[:6] == [1, 2, 3, 4, 5, 6]
    assert got[-1] <= 40 and len(got) == 16
    assert got == sorted(got), "offsets must come out ascending"


def test_schedule_is_clamped_to_the_lookback():
    assert max(parse_history_schedule("1:6,2:40", 8)) <= 8


def test_bad_schedule_names_the_segment():
    with pytest.raises(ValueError, match="bad history schedule segment"):
        parse_history_schedule("1:6,nonsense", 10)


def synthetic_meta(dates, frame="South"):
    """{id: meta} for a run of 11-day interferograms starting at each date."""
    from datetime import datetime, timedelta

    out = {}
    for d in dates:
        start = datetime.strptime(d, "%Y%m%d")
        end = start + timedelta(days=11)
        out[f"{d}_{end.strftime('%Y%m%d')}"] = {"frame": frame, "nonz_num": 5}
    return out


def test_select_history_keeps_going_across_a_hole():
    """The whole point: a missing acquisition costs one frame, not the sample."""
    from datetime import datetime, timedelta

    base = datetime.strptime("20200101", "%Y%m%d")
    dates = [(base + timedelta(days=11 * i)).strftime("%Y%m%d") for i in range(6)]
    del dates[3]                                     # punch a hole 2 slots back
    meta = synthetic_meta(dates)
    current = sorted(meta)[-1]

    ids, offsets = select_history(current, meta, lookback=5, groups=frame_groups_of(meta))
    assert offsets[-1] == 0 and ids[-1] == current, "current interferogram must come last"
    assert 2 not in offsets, "the hole must not be filled in"
    assert offsets == sorted(offsets, reverse=True), "oldest (largest offset) first"
    assert offsets == [5, 4, 3, 1, 0]
    assert len(ids) == len(offsets) and len(set(ids)) == len(ids)


def test_select_history_stays_on_one_frame():
    meta = synthetic_meta(["20200101", "20200112"], frame="South")
    meta.update(synthetic_meta(["20200123"], frame="North"))
    north = [k for k, v in meta.items() if v["frame"] == "North"][0]
    ids, offsets = select_history(north, meta, lookback=3, groups=frame_groups_of(meta))
    assert offsets == [0], "South predecessors must not enter a North chain"


def test_select_history_on_the_real_dictionary_beats_the_strict_rule():
    """The measurement that motivates the whole design, on the committed asset."""
    from sinkholes.meta import find_11day_sequences, load_coord_dict

    meta = load_coord_dict()
    groups = frame_groups_of(meta)
    _, strict = find_11day_sequences(meta, k_prev=40)

    tolerant, depths = [], []
    for intf_id, info in meta.items():
        try:
            nonz = int(info.get("nonz_num", 0))
        except (TypeError, ValueError):
            continue
        if nonz <= 0:
            continue
        _, offsets = select_history(intf_id, meta, lookback=40, groups=groups)
        tolerant.append(intf_id)
        depths.append(len(offsets) - 1)

    assert len(strict) < 30, "the strict rule should be nearly empty at a 40-slot lookback"
    assert len(tolerant) > 250, "hole tolerance should keep essentially every interferogram"
    assert np.mean(depths) > 25, "and most of the history should still be there"
