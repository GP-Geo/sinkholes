"""The attention probe's mapping and statistics, without touching the data.

Two pieces carry the weight and neither is obvious from reading it:

- :func:`replicated_plan` lays a gappy history onto a fixed offset grid, so
  "mean weight at offset 22" means the same thing for interferograms with
  different holes. Get it wrong and every per-offset number is quietly
  misaligned.
- :class:`WeightStats` accumulates sums rather than holding the weights, so its
  denominators have to track which offsets each sample actually carried.
"""

import numpy as np
import pytest
import torch

from sinkholes.inference.attention_probe import WeightStats, replicated_plan


# -- laying a gappy history onto a fixed grid ---------------------------------------------

def test_a_complete_history_maps_straight_through():
    avail = [("d", 3), ("c", 2), ("b", 1), ("a", 0)]
    ids, offsets, valid = replicated_plan(avail, [3, 2, 1, 0])
    assert ids == ["d", "c", "b", "a"]
    assert offsets == [3, 2, 1, 0]
    assert valid == [True] * 4


def test_a_hole_is_filled_and_marked_invalid():
    """The filler exists only to keep BatchNorm fed; attention masks it out."""
    ids, offsets, valid = replicated_plan([("d", 3), ("b", 1), ("a", 0)], [3, 2, 1, 0])
    assert offsets == [3, 2, 1, 0]
    assert valid == [True, False, True, True]
    assert ids[1] in ("d", "b"), "a hole must be filled from a real neighbour"
    assert ids == ["d", ids[1], "b", "a"]


def test_the_present_is_never_padded():
    """Offset 0 is the attention query; a padded one would softmax to NaN."""
    _, offsets, valid = replicated_plan([("a", 0), ("z", 9)], list(range(9, -1, -1)))
    assert offsets[-1] == 0 and valid[-1] is True


def test_far_offsets_beyond_the_history_are_filled_from_the_oldest():
    """A 2019 scene has no 2018 predecessors; the grid still has those slots."""
    ids, offsets, valid = replicated_plan([("old", 2), ("new", 0)], [6, 4, 2, 0])
    assert valid == [False, False, True, True]
    assert ids[:2] == ["old", "old"], "the nearest real frame is the oldest one"


def test_every_slot_gets_a_real_id():
    ids, _, _ = replicated_plan([("a", 0)], list(range(40, -1, -1)))
    assert len(ids) == 41 and set(ids) == {"a"}


def test_the_grid_is_identical_whatever_the_holes():
    """What makes per-offset means comparable across interferograms."""
    grid = [5, 4, 3, 2, 1, 0]
    a = replicated_plan([("p", 5), ("q", 3), ("r", 0)], grid)
    b = replicated_plan([("s", 4), ("t", 1), ("u", 0)], grid)
    assert a[1] == b[1] == grid


# -- statistics ---------------------------------------------------------------------------

def uniform_weights(b, heads, t, hb=3, wb=2):
    return torch.full((b, heads, t, hb, wb), 1.0 / t)


def test_uniform_weights_report_exactly_T_effective_frames():
    """The number that exposed the collapse. It must be exact, not approximate."""
    st = WeightStats(offsets=[3, 2, 1, 0], heads=2)
    st.update(uniform_weights(4, 2, 4), torch.ones(4, 4, dtype=torch.bool), horizon=2)
    assert st.summary()["effective_frames"] == pytest.approx(4.0, rel=1e-6)


def test_one_hot_weights_report_one_effective_frame():
    w = torch.zeros(2, 2, 4, 3, 2)
    w[:, :, -1] = 1.0
    st = WeightStats(offsets=[3, 2, 1, 0], heads=2)
    st.update(w, torch.ones(2, 4, dtype=torch.bool), horizon=2)
    s = st.summary()
    assert s["effective_frames"] == pytest.approx(1.0, rel=1e-4)
    assert s["collapse_fraction"] == pytest.approx(1.0)


def test_availability_tracks_which_offsets_a_sample_carried():
    """The denominator: an offset only two of four samples had must not be
    averaged as though all four did."""
    valid = torch.tensor([[False, True, True, True],
                          [False, True, True, True],
                          [True, True, True, True],
                          [True, True, True, True]])
    w = torch.zeros(4, 2, 4, 3, 2)
    for i in range(4):
        n = int(valid[i].sum())
        w[i, :, -n:] = 1.0 / n
    st = WeightStats(offsets=[3, 2, 1, 0], heads=2)
    st.update(w, valid, horizon=3)
    rows = {r["offset"]: r for r in st.rows()}
    assert rows[3]["availability"] == pytest.approx(0.5)
    assert rows[0]["availability"] == pytest.approx(1.0)
    # Offset 3 was carried by two samples, each giving it 1/4 — not diluted by
    # the two samples that never had it.
    assert rows[3]["mean_weight"] == pytest.approx(0.25, rel=1e-5)


def test_mass_beyond_the_horizon_is_measured_only_past_it():
    w = torch.zeros(1, 1, 4, 1, 1)
    w[:, :, 0] = 1.0                      # everything on offset 3
    st = WeightStats(offsets=[3, 2, 1, 0], heads=1)
    st.update(w, torch.ones(1, 4, dtype=torch.bool), horizon=2)
    assert st.summary()["mass_beyond_offset_2"] == pytest.approx(1.0)


def test_argmax_share_is_normalised_by_availability():
    valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
    w = torch.zeros(2, 1, 4, 1, 1)
    w[0, :, 1] = 1.0                      # sample 0 peaks at offset 2
    w[1, :, 1] = 1.0                      # sample 1 too
    st = WeightStats(offsets=[3, 2, 1, 0], heads=1)
    st.update(w, valid, horizon=2)
    rows = {r["offset"]: r for r in st.rows()}
    assert rows[2]["argmax_share"] == pytest.approx(1.0)
    assert rows[3]["argmax_share"] == pytest.approx(0.0)


def test_rows_come_out_ascending_by_offset():
    st = WeightStats(offsets=[5, 3, 1, 0], heads=1)
    st.update(uniform_weights(1, 1, 4), torch.ones(1, 4, dtype=torch.bool), horizon=2)
    assert [r["offset"] for r in st.rows()] == [0, 1, 3, 5]
    assert [r["days"] for r in st.rows()] == [0, 11, 33, 55]


def test_effective_frames_averages_exp_entropy_not_the_other_way_round():
    """A half-collapsed, half-uniform batch must not read as uniformly moderate.

    exp(mean entropy) would give sqrt(1*4) = 2.0 here; the honest statistic is
    the mean of exp, (1 + 4) / 2 = 2.5.
    """
    w = torch.zeros(2, 1, 4, 1, 1)
    w[0, :, -1] = 1.0                     # collapsed
    w[1] = 0.25                           # uniform
    st = WeightStats(offsets=[3, 2, 1, 0], heads=1)
    st.update(w, torch.ones(2, 4, dtype=torch.bool), horizon=2)
    assert st.summary()["effective_frames"] == pytest.approx(2.5, rel=1e-3)
