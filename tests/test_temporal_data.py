"""Tests for the temporal data contract: target definition, ordering, validity channels.

EDIT 2026-07-28: new file. CHANGELOG.md #7, #8
"""

import numpy as np
import pytest

from get_intf_info import find_11day_sequences
from sinkholes_data_loading import SubsiDataset, temporal_target_mask


# ---------------------------------------------------------------------------------------
# 11. The training target is the LATEST timestep's mask, not the union
# ---------------------------------------------------------------------------------------
def make_masks_with_disjoint_positives():
    """Three (N, H, W) masks whose positives do not overlap, so union != latest."""
    n, h, w = 2, 4, 4
    oldest = np.zeros((n, h, w), dtype=np.float32)
    middle = np.zeros((n, h, w), dtype=np.float32)
    newest = np.zeros((n, h, w), dtype=np.float32)
    oldest[:, 0, 0] = 1.0
    middle[:, 1, 1] = 1.0
    newest[:, 2, 2] = 1.0
    return [oldest, middle, newest]


def test_target_is_the_latest_timestep_only():
    masks = make_masks_with_disjoint_positives()

    target = temporal_target_mask(masks)

    assert np.array_equal(target, masks[-1]), \
        'the default target must be the newest interferogram mask'
    assert target[:, 0, 0].sum() == 0, 'the oldest timestep must not contribute positives'
    assert target[:, 1, 1].sum() == 0, 'the middle timestep must not contribute positives'
    assert target[:, 2, 2].all(), 'the newest timestep positives must survive'
    assert target.sum() == 2, 'exactly the newest mask, nothing more'


def test_union_flag_restores_the_legacy_target():
    masks = make_masks_with_disjoint_positives()

    union = temporal_target_mask(masks, union=True)

    assert union.sum() == 6, 'union keeps every timestep positive (3 sites x N=2)'
    assert union[:, 0, 0].all() and union[:, 1, 1].all() and union[:, 2, 2].all()


def test_target_is_binary_float32_regardless_of_input_values():
    masks = [np.zeros((1, 2, 2), dtype=np.float32),
             np.full((1, 2, 2), 7.0, dtype=np.float32)]

    target = temporal_target_mask(masks)

    assert target.dtype == np.float32
    assert set(np.unique(target)) <= {0.0, 1.0}
    assert target.all(), 'any positive value must map to 1.0'


def test_single_timestep_target_is_that_timestep():
    only = np.array([[[1.0, 0.0]]], dtype=np.float32)
    assert np.array_equal(temporal_target_mask([only]), only)


# ---------------------------------------------------------------------------------------
# 12. Chain order is chronological, oldest -> newest
# ---------------------------------------------------------------------------------------
def synthetic_meta():
    """Four consecutive 11-day interferograms in the same frame."""
    keys = ['20190113_20190124', '20190124_20190204',
            '20190204_20190215', '20190215_20190226']
    return {k: {'frame': 'North', 'nonz_num': 10} for k in keys}


def test_prevs_are_returned_oldest_first():
    meta = synthetic_meta()
    current = '20190215_20190226'

    chains, valid = find_11day_sequences(meta, k_prev=3, restrict_to=[current])

    assert current in valid
    prevs = chains[current]['prevs']
    assert prevs == ['20190113_20190124', '20190124_20190204', '20190204_20190215']
    assert prevs == sorted(prevs), 'prevs must be ascending by date (oldest first)'


def test_tids_sequence_is_chronological_with_current_last():
    """`tids = list(prevs) + [id]` — the ordering SubsiDataset builds at :176."""
    meta = synthetic_meta()
    current = '20190215_20190226'
    chains, _ = find_11day_sequences(meta, k_prev=2, restrict_to=[current])

    tids = list(chains[current]['prevs']) + [current]

    assert tids == sorted(tids), 'the temporal stack must be ordered oldest -> newest'
    assert tids[-1] == current, 'the current interferogram must be the LAST timestep'
    assert len(tids) == 3, 'T = k_prevs + 1'


def test_chain_is_dropped_when_a_predecessor_is_missing():
    meta = synthetic_meta()
    del meta['20190124_20190204']

    chains, valid = find_11day_sequences(meta, k_prev=3,
                                         restrict_to=['20190215_20190226'])

    assert valid == [] and chains == {}


# ---------------------------------------------------------------------------------------
# Validity channels must survive preprocessing as strict {0, 1}
# ---------------------------------------------------------------------------------------
def test_preprocess_leaves_validity_channels_untouched():
    """With --treat_nodata_regions the stack is [img_t0, img_t1, V_t0, V_t1].

    Phase channels get their exact zeros mapped to 0.5; the validity channels must not,
    or `segmentation_loss`'s V_any is never 0 and the no-data masking does nothing.
    """
    t, h, w = 2, 4, 4
    imgs = np.full((t, h, w), 0.3, dtype=np.float32)
    imgs[:, 0, 0] = 0.0                                   # a no-data pixel
    valid = np.ones((t, h, w), dtype=np.float32)
    valid[:, 0, 0] = 0.0                                  # matching validity flag
    stack = np.concatenate([imgs, valid], axis=0)         # (2T, H, W), block layout

    out = SubsiDataset.preprocess([0, 1], stack, 0, n_value_channels=t)

    assert np.allclose(out[t:], valid), 'validity channels must pass through unchanged'
    assert set(np.unique(out[t:])) <= {0.0, 1.0}, 'validity must stay strictly {0, 1}'
    assert out[0, 0, 0] == pytest.approx(0.5), 'phase zeros are still mapped to 0.5'


def test_preprocess_without_the_offset_keeps_the_old_behaviour():
    """n_value_channels=None must normalise every channel, as before this change."""
    stack = np.zeros((2, 3, 3), dtype=np.float32)

    out = SubsiDataset.preprocess([0, 1], stack, 0)

    assert np.allclose(out, 0.5), 'legacy path: every exact zero becomes 0.5'


def test_preprocess_still_rescales_raw_phase():
    """Values outside [0, 1] are treated as radians and mapped to [0, 1]."""
    stack = np.array([[[-np.pi, 0.0, np.pi]]], dtype=np.float32)

    out = SubsiDataset.preprocess([0, 1], stack, 0)

    assert out[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[0, 0, 1] == pytest.approx(0.5, abs=1e-6)
    assert out[0, 0, 2] == pytest.approx(1.0, abs=1e-6)
