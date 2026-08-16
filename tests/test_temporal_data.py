"""Temporal data contract: sample selection, target definition, chain ordering,
validity channels."""

import json

import numpy as np
import pytest
import torch

from sinkholes.dataprep.dataset import (
    RingNegatives,
    SubsiDataset,
    temporal_target_mask,
    validation_negatives,
)
from sinkholes.dataprep.patchify import patch_file_name
from sinkholes.meta import find_11day_sequences


# -- the training target is the LATEST timestep's mask, not the union -------------------

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
    assert np.array_equal(target, masks[-1])
    assert target[:, 0, 0].sum() == 0
    assert target[:, 1, 1].sum() == 0
    assert target[:, 2, 2].all()
    assert target.sum() == 2


def test_union_flag_restores_the_legacy_target():
    masks = make_masks_with_disjoint_positives()
    union = temporal_target_mask(masks, union=True)
    assert union.sum() == 6
    assert union[:, 0, 0].all() and union[:, 1, 1].all() and union[:, 2, 2].all()


def test_target_is_binary_float32_regardless_of_input_values():
    masks = [np.zeros((1, 2, 2), dtype=np.float32),
             np.full((1, 2, 2), 7.0, dtype=np.float32)]
    target = temporal_target_mask(masks)
    assert target.dtype == np.float32
    assert set(np.unique(target)) <= {0.0, 1.0}
    assert target.all()


def test_single_timestep_target_is_that_timestep():
    only = np.array([[[1.0, 0.0]]], dtype=np.float32)
    assert np.array_equal(temporal_target_mask([only]), only)


# -- sample selection: the same patches a single-frame run sees -------------------------

CHAIN = ["20190113_20190124", "20190124_20190204", "20190204_20190215"]
CURRENT = CHAIN[-1]
CHAINS = {CURRENT: {"prevs": CHAIN[:-1], "frame": "North"}}
PATCH = (4, 4)
STRIDE = 2


def patch_code(t: int, i: int, j: int) -> float:
    """The constant value filling patch (i, j) of timestep t.

    Encodes both identity and chronological order, and stays inside (0, 1)
    with no exact zeros so normalisation is a pass-through.
    """
    return round(0.1 * (t + 1) + 0.01 * i + 0.001 * j, 3)


def write_patch_tree(root, positives, ny=3, nx=3):
    """A synthetic patch tree: full grids, nonz files and nonz_indices.json.

    ``positives`` maps interferogram id -> grid coordinates whose mask is
    non-empty. Returns (image_dir, mask_dir).
    """
    H, W = PATCH
    img_dir, msk_dir = root / "data", root / "mask"
    img_dir.mkdir()
    msk_dir.mkdir()

    nonz_indices = {}
    for t, tid in enumerate(CHAIN):
        grid = np.zeros((ny, nx, H, W), dtype=np.float32)
        mask = np.zeros((ny, nx, H, W), dtype=np.float32)
        for i in range(ny):
            for j in range(nx):
                grid[i, j] = patch_code(t, i, j)
        coords = sorted(positives[tid])
        for (i, j) in coords:
            mask[i, j, 0, 0] = 1.0

        np.save(img_dir / patch_file_name("data", tid, H, W, STRIDE), grid)
        np.save(msk_dir / patch_file_name("mask", tid, H, W, STRIDE), mask)
        np.save(img_dir / patch_file_name("data", tid, H, W, STRIDE, nonz=True),
                np.stack([grid[i, j] for (i, j) in coords]))
        np.save(msk_dir / patch_file_name("mask", tid, H, W, STRIDE, nonz=True),
                np.stack([mask[i, j] for (i, j) in coords]))
        nonz_indices[tid] = [[i, j] for (i, j) in coords]

    (img_dir / "nonz_indices.json").write_text(json.dumps(nonz_indices))
    return img_dir, msk_dir


#: (0, 0) and (2, 2) are positive only in predecessors; (1, 1) throughout;
#: (0, 2) only in the current frame. Union has 4 coordinates, current has 2.
DISJOINT_POSITIVES = {
    CHAIN[0]: [(0, 0), (1, 1)],
    CHAIN[1]: [(1, 1), (2, 2)],
    CHAIN[2]: [(1, 1), (0, 2)],
}


def build_pair(tmp_path, positives=None, **temporal_kwargs):
    """(single-frame, temporal) datasets over the same synthetic tree."""
    img_dir, msk_dir = write_patch_tree(tmp_path, positives or DISJOINT_POSITIVES)
    common = dict(patch_size=PATCH, stride=STRIDE)
    single = SubsiDataset(img_dir, msk_dir, [CURRENT], **common)
    temporal = SubsiDataset(img_dir, msk_dir, [CURRENT], temporal=True,
                            seq_dict=CHAINS, **common, **temporal_kwargs)
    return single, temporal


def test_temporal_samples_are_exactly_the_single_frame_patches(tmp_path):
    """Same count, same patches, same targets — only the input depth differs."""
    single, temporal = build_pair(tmp_path)

    assert len(temporal) == len(single) == 2
    for n in range(len(single)):
        s, t = single[n], temporal[n]
        assert tuple(s["image"].shape) == (1, *PATCH)
        assert tuple(t["image"].shape) == (len(CHAIN), *PATCH)
        assert torch.equal(t["image"][-1], s["image"][0]), "same patch, current frame"
        assert torch.equal(t["mask"], s["mask"]), "same target"


def test_patches_empty_in_the_current_frame_are_not_sampled(tmp_path):
    """A patch positive only in a predecessor must not enter the set."""
    _, temporal = build_pair(tmp_path)

    sampled = {round(float(temporal[n]["image"][-1, 0, 0]), 3) for n in range(len(temporal))}
    last = len(CHAIN) - 1
    assert sampled == {patch_code(last, 1, 1), patch_code(last, 0, 2)}
    assert patch_code(last, 0, 0) not in sampled, "positive only in the oldest frame"
    assert patch_code(last, 2, 2) not in sampled, "positive only in the middle frame"


def test_every_temporal_target_is_non_empty(tmp_path):
    """No all-zero targets: per-patch Dice scores those 1.0 for predicting nothing."""
    _, temporal = build_pair(tmp_path)

    for n in range(len(temporal)):
        assert temporal[n]["mask"].sum() > 0


def test_temporal_input_still_carries_the_whole_chain_oldest_first(tmp_path):
    """Restricting the coordinates must not shorten or reorder the input stack."""
    _, temporal = build_pair(tmp_path)

    stack = temporal[0]["image"]
    assert tuple(stack.shape) == (len(CHAIN), *PATCH), "T = k_prevs + 1 frames of context"
    # Sample 0 is (0, 2) — nonz order is the sorted coordinate list.
    values = [round(float(stack[t, 0, 0]), 3) for t in range(len(CHAIN))]
    assert values == [patch_code(t, 0, 2) for t in range(len(CHAIN))]


def test_union_target_mode_samples_the_union_to_match_its_target(tmp_path):
    """The legacy target stays self-consistent: union target, union coordinates."""
    single, temporal = build_pair(tmp_path, union_temporal_mask=True)

    assert len(single) == 2
    assert len(temporal) == 4, "(0, 0), (1, 1), (2, 2) and (0, 2)"
    for n in range(len(temporal)):
        assert temporal[n]["mask"].sum() > 0, "a union target is never empty either"


# -- ring negatives: the single-frame control draws the same ones -----------------------

#: A 9x9 grid, so a 1..3 annulus has somewhere to land. (4, 4) is positive in
#: every frame; (1, 1) only in the oldest, which makes it a patch that WAS
#: positive and must therefore never be offered as a negative.
RING_GRID = 9
RING_POSITIVES = {
    CHAIN[0]: [(1, 1), (4, 4)],
    CHAIN[1]: [(4, 4)],
    CHAIN[2]: [(4, 4)],
}


def build_ring_pair(tmp_path, **ring_kwargs):
    """(single-frame, temporal) datasets over one tree, both with ring negatives."""
    img_dir, msk_dir = write_patch_tree(tmp_path, RING_POSITIVES, ny=RING_GRID, nx=RING_GRID)
    ring = RingNegatives(seed=0, **ring_kwargs)
    common = dict(patch_size=PATCH, stride=STRIDE, ring_negatives=ring)
    single = SubsiDataset(img_dir, msk_dir, [CURRENT], seq_dict=CHAINS, **common)
    temporal = SubsiDataset(img_dir, msk_dir, [CURRENT], temporal=True,
                            seq_dict=CHAINS, **common)
    return single, temporal


def sampled_coords(ds, frame_index=0):
    """Decode each sample's (i, j) back out of its constant patch value."""
    out = []
    for n in range(len(ds)):
        img = ds[n]["image"]
        v = float(img[frame_index, 0, 0]) if img.shape[0] > 1 else float(img[0, 0, 0])
        rest = round(v - 0.1 * len(CHAIN), 3)          # strip the timestep term
        out.append((int(round(rest / 0.01)), int(round((rest % 0.01) / 0.001))))
    return out


def test_single_frame_ring_negatives_match_the_temporal_ones(tmp_path):
    """The control is only a control if both paths get the SAME negatives."""
    single, temporal = build_ring_pair(tmp_path)

    assert len(single) == len(temporal) > 1, "positives plus at least one negative"
    # Temporal's current frame is the last of the stack; single's is its only one.
    assert sampled_coords(temporal, frame_index=len(CHAIN) - 1) == sampled_coords(single)


def test_single_frame_ring_negatives_are_empty_patches(tmp_path):
    single, _ = build_ring_pair(tmp_path)

    masks = [single[n]["mask"].sum().item() for n in range(len(single))]
    assert masks[0] > 0, "the positive comes first"
    assert any(m == 0 for m in masks), "ring negatives are all-zero targets"


def test_a_patch_positive_in_any_frame_is_never_a_negative(tmp_path):
    """(1, 1) is positive only in the oldest frame — still not background."""
    single, _ = build_ring_pair(tmp_path)

    assert (1, 1) not in sampled_coords(single), \
        "the exclusion grid must be the union over the chain, not the current frame"


def test_single_frame_without_a_chain_still_excludes_its_own_positives(tmp_path):
    """No seq_dict: degrades to the current frame, but never mislabels a positive."""
    img_dir, msk_dir = write_patch_tree(tmp_path, RING_POSITIVES, ny=RING_GRID, nx=RING_GRID)
    ds = SubsiDataset(img_dir, msk_dir, [CURRENT], patch_size=PATCH, stride=STRIDE,
                      ring_negatives=RingNegatives(seed=0))

    coords = sampled_coords(ds)
    assert (4, 4) in coords, "the current frame's positive is still sampled"
    assert coords.count((4, 4)) == 1, "and not duplicated as a negative"


def test_training_ring_negatives_never_reach_validation(tmp_path):
    """--add_ring_negatives is a TRAINING intervention, whatever the split."""
    img_dir, msk_dir = write_patch_tree(tmp_path, RING_POSITIVES, ny=RING_GRID, nx=RING_GRID)
    val = SubsiDataset(img_dir, msk_dir, [CURRENT], mode="val", patch_size=PATCH,
                       stride=STRIDE, seq_dict=CHAINS, ring_negatives=RingNegatives(seed=0))

    assert len(val) == 1, "only (4, 4), the current frame's single positive"
    assert val[0]["mask"].sum() > 0


# -- validation negatives: the fixed measurement set ------------------------------------

def build_val(root, *, temporal, seed=42, ring_negatives=None):
    """A validation dataset with the fixed 1:1 negatives, single-frame or temporal."""
    root.mkdir(parents=True, exist_ok=True)
    img_dir, msk_dir = write_patch_tree(root, RING_POSITIVES, ny=RING_GRID, nx=RING_GRID)
    return SubsiDataset(
        img_dir, msk_dir, [CURRENT], mode="val", patch_size=PATCH, stride=STRIDE,
        seq_dict=CHAINS, temporal=temporal, ring_negatives=ring_negatives,
        val_negatives=validation_negatives(seed),
    )


def test_validation_negatives_are_added_to_the_val_split(tmp_path):
    val = build_val(tmp_path, temporal=True)

    assert len(val) == 2, "the one positive plus one negative at the fixed 1:1 ratio"
    assert val.n_negative == 1
    assert val[0]["mask"].sum() > 0
    assert val[1]["mask"].sum() == 0


def test_validation_negatives_do_not_reach_the_train_split(tmp_path):
    """The measurement set is the val split's alone — it must not become training data."""
    img_dir, msk_dir = write_patch_tree(tmp_path, RING_POSITIVES, ny=RING_GRID, nx=RING_GRID)
    train = SubsiDataset(img_dir, msk_dir, [CURRENT], mode="train", patch_size=PATCH,
                         stride=STRIDE, seq_dict=CHAINS, temporal=True,
                         val_negatives=validation_negatives(42))

    assert len(train) == 1 and train.n_negative == 0


def test_a_validation_negative_is_empty_at_every_timestep(tmp_path):
    """The same temporal validity a training negative must satisfy.

    (1, 1) is positive in the oldest frame only, so it is empty in the current
    one — it would pass a current-frame-only test and must still be excluded.
    """
    val = build_val(tmp_path, temporal=True)
    coords = sampled_coords(val, frame_index=len(CHAIN) - 1)

    assert (1, 1) not in coords, "empty now, but positive in the chain — not background"
    assert [val[n]["mask"].sum().item() for n in range(len(val))] == [1, 0]


def test_validation_negatives_are_the_same_across_architectures(tmp_path):
    """A single-frame run and a temporal one must be SCORED on identical patches."""
    single = build_val(tmp_path / "s", temporal=False)
    temporal = build_val(tmp_path / "t", temporal=True)

    assert len(single) == len(temporal)
    assert sampled_coords(single) == sampled_coords(temporal, frame_index=len(CHAIN) - 1)


@pytest.mark.parametrize("ring", [None, RingNegatives(inner=1, outer=10, per_pos=3.0, seed=7)])
def test_validation_negatives_ignore_the_training_negative_config(tmp_path, ring):
    """Sweeping --neg_ring_outer / --neg_per_pos must not move the ruler."""
    val = build_val(tmp_path / str(ring is None), temporal=True, ring_negatives=ring)

    assert sampled_coords(val, frame_index=len(CHAIN) - 1) == \
        sampled_coords(build_val(tmp_path / "ref", temporal=True), len(CHAIN) - 1)


def test_validation_negatives_are_deterministic_given_the_seed(tmp_path):
    """Drawn from default_rng(seed), so the global RNG cannot perturb them."""
    np.random.seed(1)
    first = sampled_coords(build_val(tmp_path / "a", temporal=True), len(CHAIN) - 1)
    np.random.seed(999)
    second = sampled_coords(build_val(tmp_path / "b", temporal=True), len(CHAIN) - 1)

    assert first == second


def test_validation_negatives_need_a_seed():
    """A run without --seed would draw a different val set every time."""
    with pytest.raises(ValueError, match="seed"):
        validation_negatives(None)


def test_the_validation_negative_configuration_is_fixed():
    """Pinned constants, not defaults someone can pass past — see train.py."""
    ring = validation_negatives(42)
    assert (ring.inner, ring.outer, ring.per_pos) == (1, 3, 1.0)
    assert ring.seed == 42


# -- chain order is chronological, oldest -> newest -------------------------------------

def synthetic_meta():
    """Four consecutive 11-day interferograms in the same frame."""
    keys = ["20190113_20190124", "20190124_20190204",
            "20190204_20190215", "20190215_20190226"]
    return {k: {"frame": "North", "nonz_num": 10} for k in keys}


def test_prevs_are_returned_oldest_first():
    meta = synthetic_meta()
    current = "20190215_20190226"
    chains, valid = find_11day_sequences(meta, k_prev=3, restrict_to=[current])
    assert current in valid
    prevs = chains[current]["prevs"]
    assert prevs == ["20190113_20190124", "20190124_20190204", "20190204_20190215"]
    assert prevs == sorted(prevs), "prevs must be ascending by date (oldest first)"


def test_tids_sequence_is_chronological_with_current_last():
    """The stack the dataset builds is prevs + [current] — current is the LAST timestep."""
    meta = synthetic_meta()
    current = "20190215_20190226"
    chains, _ = find_11day_sequences(meta, k_prev=2, restrict_to=[current])
    tids = list(chains[current]["prevs"]) + [current]
    assert tids == sorted(tids)
    assert tids[-1] == current
    assert len(tids) == 3, "T = k_prevs + 1"


def test_chain_is_dropped_when_a_predecessor_is_missing():
    meta = synthetic_meta()
    del meta["20190124_20190204"]
    chains, valid = find_11day_sequences(meta, k_prev=3, restrict_to=["20190215_20190226"])
    assert valid == [] and chains == {}


def test_chain_requires_positive_patches_by_default():
    meta = synthetic_meta()
    meta["20190215_20190226"]["nonz_num"] = "none"
    _, valid = find_11day_sequences(meta, k_prev=1, restrict_to=["20190215_20190226"])
    assert valid == []
    _, valid = find_11day_sequences(meta, k_prev=1, restrict_to=["20190215_20190226"],
                                    require_current_nonz_gt0=False)
    assert valid == ["20190215_20190226"]


# -- validity channels must survive preprocessing as strict {0, 1} ----------------------

def test_preprocess_leaves_validity_channels_untouched():
    """With validity channels the stack is [img_t0, img_t1, V_t0, V_t1].

    Phase channels get exact zeros mapped to 0.5; the validity channels must
    not, or the masked loss's V_any is never 0 and no-data masking does nothing.
    """
    t, h, w = 2, 4, 4
    imgs = np.full((t, h, w), 0.3, dtype=np.float32)
    imgs[:, 0, 0] = 0.0
    valid = np.ones((t, h, w), dtype=np.float32)
    valid[:, 0, 0] = 0.0
    stack = np.concatenate([imgs, valid], axis=0)

    out = SubsiDataset.preprocess([0, 1], stack, 0, n_value_channels=t)

    assert np.allclose(out[t:], valid), "validity channels must pass through unchanged"
    assert set(np.unique(out[t:])) <= {0.0, 1.0}
    assert out[0, 0, 0] == pytest.approx(0.5), "phase zeros still map to 0.5"


def test_preprocess_without_the_offset_normalises_every_channel():
    stack = np.zeros((2, 3, 3), dtype=np.float32)
    out = SubsiDataset.preprocess([0, 1], stack, 0)
    assert np.allclose(out, 0.5)


def test_preprocess_still_rescales_raw_phase():
    """Values outside [0, 1] are treated as radians and mapped to [0, 1]."""
    stack = np.array([[[-np.pi, 0.0, np.pi]]], dtype=np.float32)
    out = SubsiDataset.preprocess([0, 1], stack, 0)
    assert out[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[0, 0, 1] == pytest.approx(0.5, abs=1e-6)
    assert out[0, 0, 2] == pytest.approx(1.0, abs=1e-6)


def test_preprocess_maps_mask_values_to_class_indices():
    mask = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    out = SubsiDataset.preprocess([0, 1], mask, 1)
    assert out.dtype == np.int64
    assert np.array_equal(out, [[0, 1], [1, 0]])
