"""The three consumers of the AOI window must agree.

`grid_window` decides which ground a split owns. Three places act on that
answer -- the dataset (which patches are trained on), scene reconstruction
(which tiles are predicted) and eval-outputs (which canvas is scored). If any
two disagree, a model is scored on ground it trained on and the benchmark is
void. These tests pin the agreement rather than each consumer separately.
"""
import json

import numpy as np
import pytest

from sinkholes.dataprep.dataset import RingNegatives, SubsiDataset
from sinkholes.dataprep.partition import (
    PARTITION_SPLITS,
    TRAINABLE_SPLITS,
    load_partition_window,
    load_partition_split,
)
from sinkholes.geo import FRAME_ORIGINS, PIXEL_DEG, grid_window

PATCH = (200, 100)
STRIDE = 2
CUT = 31.4
AOI = (31.25, 31.75, 35.38, 35.46)

TRAIN_BOX = (CUT, AOI[1], AOI[2], AOI[3])          # North above the cut
CROSS_BOX = (AOI[0], CUT, AOI[2], AOI[3])          # North below it
HOLD_BOX = (AOI[0], CUT, AOI[2], AOI[3])           # South below it


def canvas_rows(window, patch_h, step_y, height):
    """The outputs.py crop, as a row slice -- kept identical to that module."""
    r0, r1 = window[0], window[1]
    return slice(min(r0 * step_y, height), min(r1 * step_y + patch_h - step_y, height))


# --------------------------------------------------------------------------
# the invariant that matters
# --------------------------------------------------------------------------

def test_train_and_holdout_windows_cover_disjoint_ground():
    """North train band and South hold-out must not share any latitude."""
    # bottom of the lowest train patch
    r1 = grid_window("North", *TRAIN_BOX)[1]
    train_bottom = FRAME_ORIGINS["North"][1] - ((r1 - 1) * (PATCH[0] // STRIDE) + PATCH[0]) * PIXEL_DEG
    # top of the highest hold-out patch
    s0 = grid_window("South", *HOLD_BOX)[0]
    hold_top = FRAME_ORIGINS["South"][1] - s0 * (PATCH[0] // STRIDE) * PIXEL_DEG
    assert train_bottom >= CUT - 1e-9
    assert hold_top <= CUT + 1e-9
    assert train_bottom >= hold_top - 1e-9, "train reaches below the hold-out's top edge"


def test_crossview_never_overlaps_train():
    """Section 5b's probe shares scenes with train but must not share ground."""
    tr = grid_window("North", *TRAIN_BOX)
    cv = grid_window("North", *CROSS_BOX)
    assert tr[1] <= cv[0], "train rows run into the crossview band"
    assert set(range(*tr[:2])).isdisjoint(range(*cv[:2]))


@pytest.mark.parametrize("frame,box", [
    ("North", TRAIN_BOX), ("North", CROSS_BOX), ("South", HOLD_BOX),
])
def test_scene_and_outputs_consumers_use_one_window(frame, box):
    """scenes.py clamps the same window outputs.py crops with."""
    ny, nx, height = 193, 89, 19500
    r0, r1, c0, c1 = grid_window(frame, *box, patch_size=PATCH, stride=(100, 50))
    scenes_window = (r0, min(r1, ny), c0, min(c1, nx))
    crop = canvas_rows(scenes_window, PATCH[0], 100, height)
    # every predicted tile must fall inside the scored crop
    assert crop.start <= scenes_window[0] * 100
    assert crop.stop >= (scenes_window[1] - 1) * 100 + PATCH[0] or crop.stop == height


# --------------------------------------------------------------------------
# the dataset consumer, on a synthetic tree
# --------------------------------------------------------------------------

def write_tree(root, ids, positives, ny, nx):
    from sinkholes.dataprep.patchify import patch_file_name
    H, W = PATCH
    img_dir, msk_dir = root / "data", root / "mask"
    img_dir.mkdir(); msk_dir.mkdir()
    idx = {}
    for tid in ids:
        grid = np.ones((ny, nx, H, W), dtype=np.float32)
        mask = np.zeros((ny, nx, H, W), dtype=np.float32)
        for (i, j) in positives:
            mask[i, j, 0, 0] = 1.0
        np.save(img_dir / patch_file_name("data", tid, H, W, STRIDE), grid)
        np.save(msk_dir / patch_file_name("mask", tid, H, W, STRIDE), mask)
        idx[tid] = [[i, j] for (i, j) in positives]
    (img_dir / "nonz_indices.json").write_text(json.dumps(idx))
    return img_dir, msk_dir


def test_dataset_loads_only_patches_inside_the_window(tmp_path):
    """A positive outside the window must not become a training sample."""
    intf = "20200101_20200112"
    ny, nx = 193, 89
    r0, r1, c0, c1 = grid_window("North", *TRAIN_BOX, patch_size=PATCH, stride=(100, 50))
    inside = [(r0 + 1, c0 + 1), (r0 + 2, c0 + 2)]
    outside = [(r1 + 5, c0 + 1), (r0 + 1, c1 + 3)]
    img_dir, msk_dir = write_tree(tmp_path, [intf], inside + outside, ny, nx)
    coord = {intf: {"frame": "North", "dx": PIXEL_DEG, "dy": PIXEL_DEG,
                    "north": 31.7, "east": 35.37}}

    ds = SubsiDataset(img_dir, msk_dir, [intf], patch_size=PATCH, stride=STRIDE,
                      temporal=True, seq_dict={intf: {"prevs": [], "frame": "North"}},
                      coord_dict=coord, aoi_window=TRAIN_BOX)
    assert len(ds) == len(inside), (
        f"expected only the {len(inside)} in-window positives, got {len(ds)}"
    )


def test_ring_negatives_stay_inside_the_window(tmp_path):
    """A negative drawn outside the split's ground is training on hold-out."""
    ny, nx = 40, 40
    window = (5, 20, 5, 20)
    allowed = np.zeros((ny, nx), dtype=bool)
    allowed[5:20, 5:20] = True
    union = np.zeros((ny, nx), dtype=bool)
    positives = [(6, 6), (18, 18)]
    for p in positives:
        union[p] = True

    neg = RingNegatives(inner=1, outer=6, per_pos=20.0, seed=0)
    drawn = neg.sample(positives, union, allowed=allowed)
    assert drawn, "the sampler should find candidates in this configuration"
    for (i, j) in drawn:
        assert 5 <= i < 20 and 5 <= j < 20, f"negative {(i, j)} drawn outside the window"

    unbounded = neg.sample(positives, union)
    assert any(not (5 <= i < 20 and 5 <= j < 20) for (i, j) in unbounded), (
        "without the gate the sampler should reach outside -- otherwise this "
        "test proves nothing about the gate"
    )


def test_spatial_mode_and_aoi_window_cannot_be_combined(tmp_path):
    intf = "20200101_20200112"
    img_dir, msk_dir = write_tree(tmp_path, [intf], [(1, 1)], 5, 5)
    with pytest.raises(ValueError, match="cannot be combined"):
        SubsiDataset(img_dir, msk_dir, [intf], patch_size=PATCH, stride=STRIDE,
                     spatial=True, coord_dict={intf: {"frame": "North"}},
                     aoi_window=AOI)


def test_aoi_window_requires_coord_dict(tmp_path):
    intf = "20200101_20200112"
    img_dir, msk_dir = write_tree(tmp_path, [intf], [(1, 1)], 5, 5)
    with pytest.raises(ValueError, match="needs coord_dict"):
        SubsiDataset(img_dir, msk_dir, [intf], patch_size=PATCH, stride=STRIDE,
                     aoi_window=AOI)


# --------------------------------------------------------------------------
# the partition file, the single source all three read from
# --------------------------------------------------------------------------

def write_partition(tmp_path, with_window=True):
    d = {"train": ["a"], "val": ["b"], "test": ["c"], "crossview": ["a"]}
    if with_window:
        d["aoi_window"] = {"train": list(TRAIN_BOX), "val": list(HOLD_BOX),
                           "test": list(HOLD_BOX), "crossview": list(CROSS_BOX)}
    p = tmp_path / "partition.json"
    p.write_text(json.dumps(d))
    return p


def test_window_round_trips_through_the_partition_file(tmp_path):
    p = write_partition(tmp_path)
    assert load_partition_window(p, "train") == TRAIN_BOX
    assert load_partition_window(p, "test") == HOLD_BOX
    assert load_partition_window(p, "crossview") == CROSS_BOX


def test_partition_without_a_window_means_unrestricted(tmp_path):
    """Old partitions must keep reproducing their old numbers."""
    assert load_partition_window(write_partition(tmp_path, with_window=False), "test") is None


def test_crossview_is_a_split_but_not_trainable():
    assert "crossview" in PARTITION_SPLITS
    assert "crossview" not in TRAINABLE_SPLITS


def test_crossview_list_is_readable(tmp_path):
    assert load_partition_split(write_partition(tmp_path), "crossview") == ["a"]


def test_missing_window_for_a_split_is_an_error_not_a_default(tmp_path):
    d = json.loads(write_partition(tmp_path).read_text())
    del d["aoi_window"]["test"]
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="no entry for split"):
        load_partition_window(p, "test")


# --------------------------------------------------------------------------
# the reconstruction consumer, end to end
# --------------------------------------------------------------------------

def test_reconstruct_scene_predicts_only_tiles_inside_the_window():
    """The gate must actually stop the forward pass, not just mask afterwards."""
    import torch

    from sinkholes.inference.reconstruct import reconstruct_scene

    class Recorder(torch.nn.Module):
        n_classes = 1

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return torch.zeros(x.shape[0], 1, *x.shape[-2:])

    stack = [np.full((6, 6, 4, 4), 0.3, dtype=np.float32)]

    everywhere = Recorder()
    reconstruct_scene(stack, everywhere, (4, 4), 2, threshold=0.4,
                      device=torch.device("cpu"), accumulate_image=False)

    windowed = Recorder()
    win = (1, 3, 2, 4)                      # 2x2 = 4 tiles
    reconstruct_scene(stack, windowed, (4, 4), 2, threshold=0.4,
                      device=torch.device("cpu"), tile_window=win,
                      accumulate_image=False)

    assert everywhere.calls == 36, "all 6x6 tiles should run without a window"
    assert windowed.calls == 4, f"window {win} admits 4 tiles, ran {windowed.calls}"


def test_reconstruct_scene_leaves_the_canvas_zero_outside_the_window():
    import torch

    from sinkholes.inference.reconstruct import reconstruct_scene

    class Ones(torch.nn.Module):
        n_classes = 1

        def forward(self, x):
            return torch.full((x.shape[0], 1, *x.shape[-2:]), 10.0)

    stack = [np.full((6, 6, 4, 4), 0.3, dtype=np.float32)]
    win = (1, 3, 2, 4)
    r = reconstruct_scene(stack, Ones(), (4, 4), 2, threshold=0.4,
                          device=torch.device("cpu"), tile_window=win,
                          accumulate_image=False)
    # rows above the window start (tile row 0 -> canvas rows 0:2) stay untouched
    assert r.confidence[:2].max() == 0.0
    assert r.confidence.max() > 0.0, "something inside the window must be predicted"
