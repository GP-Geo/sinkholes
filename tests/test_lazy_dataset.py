"""The dataset indexes at construction and reads pixels per sample.

Two properties are pinned here. First, ``__init__`` must stay metadata-only:
its cost may scale with the number of samples but never with the bytes on
disk — that is what made ConvLSTM runs hit the 64 GB job limit. Second, the
samples themselves must be exactly what the eager implementation produced, so
every configuration is compared against a reference that loads the whole grids
the way it used to (see ``eager_reference``).
"""

import json
import pickle
import random
import tracemalloc

import numpy as np
import pytest
import torch

from sinkholes.dataprep import dataset as ds_mod
from sinkholes.dataprep.dataset import (
    RAW_VALIDITY_TOL,
    RingNegatives,
    SubsiDataset,
    clear_grid_cache,
    load_grid,
)
from sinkholes.dataprep.patchify import patch_file_name

#: Patches are 12x10 so a row of zeros is long enough to count as a no-data
#: streak (``has_consecutive_zeros`` wants a run of 10).
H, W = 12, 10
NY, NX = 6, 3

#: Patches carrying a no-data streak: the candidates for null sampling.
STREAKS = [(0, 0), (2, 1), (3, 2), (5, 1)]
STRIDE = 2
TIDS = ["20190113_20190124", "20190124_20190204", "20190204_20190215"]
SEQ = {TIDS[2]: {"prevs": TIDS[:2]}}

#: Positive patches per interferogram, deliberately not identical across time:
#: the temporal union must pick up all of them, oldest interferogram first.
POSITIVES = {
    TIDS[0]: [(0, 0), (1, 1)],
    TIDS[1]: [(1, 1), (4, 2)],
    TIDS[2]: [(2, 0), (4, 2)],
}

#: thresh_line = floor((north - thresh_lat) / ((H // 2) * dy)) = 3, mid-grid.
COORD = {tid: {"north": 31.79, "dy": 0.02, "frame": "North"} for tid in TIDS}
THRESH_LAT = 31.4
THRESH_LINE = 3


@pytest.fixture
def tree(tmp_path):
    """A miniature patch tree: full grids, nonz subsets, nonz_indices.json."""
    return write_tree(tmp_path, patch_size=(H, W))


def write_tree(root, patch_size=(H, W), ny=NY, nx=NX):
    """Write one patch tree and return (image_dir, mask_dir).

    Values are deterministic per (tid, i, j, pixel) so a mis-indexed patch
    cannot accidentally match, and every grid carries no-data zeros plus one
    NaN so the validity channels have something to find.
    """
    h, w = patch_size
    img_dir = root / "data_patches"
    msk_dir = root / "mask_patches"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    nonz_indices = {}
    for t, tid in enumerate(TIDS):
        rng = np.random.default_rng(100 + t)
        images = rng.random((ny, nx, h, w), dtype=np.float32) * 0.8 + 0.1
        for (i, j) in STREAKS:
            images[i, j, 0, :] = 0.0        # a no-data streak
        images[1, 1, 3, 2] = np.nan
        masks = np.zeros((ny, nx, h, w), dtype=np.uint8)
        for (i, j) in POSITIVES[tid]:
            masks[i, j, 2:5, 1:3] = 1

        np.save(img_dir / patch_file_name("data", tid, h, w, STRIDE), images)
        np.save(msk_dir / patch_file_name("mask", tid, h, w, STRIDE), masks)

        coords = [[i, j] for i in range(ny) for j in range(nx) if masks[i, j].any()]
        nonz_indices[tid] = coords
        np.save(img_dir / patch_file_name("data", tid, h, w, STRIDE, nonz=True),
                np.stack([images[i, j] for i, j in coords]))
        np.save(msk_dir / patch_file_name("mask", tid, h, w, STRIDE, nonz=True),
                np.stack([masks[i, j] for i, j in coords]))

    with open(img_dir / "nonz_indices.json", "w") as fh:
        json.dump(nonz_indices, fh)
    return img_dir, msk_dir


def make(tree, **kwargs):
    img_dir, msk_dir = tree
    ids = kwargs.pop("ids", [TIDS[2]])
    clear_grid_cache()
    return SubsiDataset(img_dir, msk_dir, ids, patch_size=(H, W), stride=STRIDE, **kwargs)


# -- the reference: how the eager implementation built its samples ----------------------

def eager_reference(tree, tids, *, union=False, nodata=False, offsets=None, rows=None):
    """(image, target) per sample, built by loading whole grids.

    Mirrors the pre-refactor loader: grids clipped to their common extent, the
    positives of every timestep unioned oldest-first, target from the newest
    frame, validity channels appended in BLOCK layout.
    """
    img_dir, msk_dir = tree
    load = lambda d, kind, tid: np.load(  # noqa: E731
        d / patch_file_name(kind, tid, H, W, STRIDE)).astype(np.float32)
    img_pa = [load(img_dir, "data", tid) for tid in tids]
    msk_pa = [load(msk_dir, "mask", tid) for tid in tids]
    if rows is not None:
        img_pa = [p[rows[0]:rows[1]] for p in img_pa]
        msk_pa = [p[rows[0]:rows[1]] for p in msk_pa]
    offsets = offsets or {tid: 0 for tid in tids}

    ny = min(p.shape[0] for p in img_pa)
    nx = min(p.shape[1] for p in img_pa)
    with open(img_dir / "nonz_indices.json") as fh:
        nonz = json.load(fh)

    rc, seen = [], set()
    for tid in tids:
        for ij in nonz.get(tid, []):
            i, j = int(ij[0]) - offsets[tid], int(ij[1])
            if 0 <= i < ny and 0 <= j < nx and (i, j) not in seen:
                rc.append((i, j))
                seen.add((i, j))

    samples = []
    for (i, j) in rc:
        image = np.stack([p[i, j] for p in img_pa], axis=0).astype(np.float32)
        target = np.stack([m[i, j] for m in msk_pa], axis=0)
        target = ((target > 0).any(axis=0) if union else target[-1] > 0).astype(np.float32)
        if nodata:
            valid = (np.abs(image) > RAW_VALIDITY_TOL).astype(np.float32)
            if np.isnan(image).any():
                valid = valid * ~np.isnan(image)
                image = np.nan_to_num(image, nan=0.0)
            image = np.concatenate([image, valid], axis=0).astype(np.float32)
        samples.append((image, target))
    return rc, samples


def assert_matches_reference(ds, expected):
    assert len(ds) == len(expected)
    for idx, (image, target) in enumerate(expected):
        got_img, got_msk = ds._lazy_sample(idx)
        assert np.array_equal(got_img, image, equal_nan=True), f"image of sample {idx}"
        assert np.array_equal(got_msk, target), f"mask of sample {idx}"


# -- 1. construction does not scale with the data ---------------------------------------

def test_init_reads_no_pixels(tree, monkeypatch):
    """Every array opened during construction is a memory map."""
    clear_grid_cache()
    real_load = np.load
    calls = []

    def spy(path, *args, **kwargs):
        calls.append(kwargs.get("mmap_mode"))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy)
    make(tree, temporal=True, seq_dict=SEQ)

    assert calls, "the dataset must open its grids to index them"
    assert set(calls) == {"r"}, f"a full read slipped in: {calls}"


def test_dataset_holds_no_pixel_arrays(tree):
    """Nothing reachable from the instance is a patch array."""
    ds = make(tree, temporal=True, seq_dict=SEQ)
    for name, value in vars(ds).items():
        assert not isinstance(value, np.ndarray), f"{name} holds an array"
        if isinstance(value, list):
            assert not any(isinstance(v, np.ndarray) for v in value), f"{name} holds arrays"


def test_init_memory_is_independent_of_patch_size(tmp_path):
    """Same sample count, 64x the pixels on disk: construction cost must not follow."""
    small = write_tree(tmp_path / "small", patch_size=(H, W))
    big = write_tree(tmp_path / "big", patch_size=(H * 8, W * 8))

    def peak_bytes(dirs, patch_size):
        img_dir, msk_dir = dirs
        clear_grid_cache()
        tracemalloc.start()
        SubsiDataset(img_dir, msk_dir, [TIDS[2]], patch_size=patch_size, stride=STRIDE,
                     temporal=True, seq_dict=SEQ)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    peak_small = peak_bytes(small, (H, W))
    peak_big = peak_bytes(big, (H * 8, W * 8))
    on_disk = sum(p.stat().st_size for p in (big[0].iterdir()))

    assert peak_big < on_disk / 4, (
        f"construction allocated {peak_big} bytes for {on_disk} bytes of images"
    )
    assert peak_big < 4 * peak_small, (
        f"construction scaled with patch size: {peak_small} -> {peak_big} bytes"
    )


def test_index_is_paths_and_coordinates(tree):
    """The index names the sequence and the patch location, nothing more."""
    ds = make(tree, temporal=True, seq_dict=SEQ)
    spec = ds.sample_spec(0)
    assert spec["target"] == TIDS[2]
    assert spec["prevs"] == TIDS[:2]
    assert (spec["row"], spec["col"]) == ds.samples[0][1:]

    src = ds.groups[0]
    assert src.tids == tuple(TIDS)
    assert all(isinstance(p, str) for p in src.image_paths + src.mask_paths)


def test_pickled_split_carries_the_index_not_the_pixels(tree):
    """Saved test splits used to be gigabytes; now they are the index."""
    ds = make(tree, temporal=True, seq_dict=SEQ)
    blob = pickle.dumps(ds)
    assert len(blob) < 8_000, f"pickle is {len(blob)} bytes — arrays crept back in"

    restored = pickle.loads(blob)
    assert len(restored) == len(ds)
    a, b = restored[0], ds[0]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["mask"], b["mask"])


# -- 2. shapes -------------------------------------------------------------------------

def test_single_frame_sample_shapes(tree):
    ds = make(tree)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (1, H, W)
    assert tuple(sample["mask"].shape) == (H, W)
    assert sample["mask"].dtype == torch.int64
    assert sample["image"].dtype == torch.float32


def test_temporal_sample_shapes(tree):
    """k_prevs=2 -> T=3 channels of (H, W), mask of the newest frame only."""
    ds = make(tree, temporal=True, seq_dict=SEQ)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (3, H, W)
    assert tuple(sample["mask"].shape) == (H, W)
    assert ds.n_value_channels is None


def test_validity_channels_double_the_stack(tree):
    ds = make(tree, temporal=True, seq_dict=SEQ, treat_nodata_regions=True)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (6, H, W)
    assert ds.n_value_channels == 3
    assert set(np.unique(sample["image"][3:].numpy()).tolist()) <= {0.0, 1.0}


def test_batching_through_a_dataloader(tree):
    from torch.utils.data import DataLoader

    ds = make(tree, temporal=True, seq_dict=SEQ)
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
    assert tuple(batch["image"].shape) == (2, 3, H, W)
    assert tuple(batch["mask"].shape) == (2, H, W)


# -- 3. temporal ordering ---------------------------------------------------------------

def test_channels_are_chronological_with_the_current_frame_last(tree):
    img_dir, _ = tree
    ds = make(tree, temporal=True, seq_dict=SEQ)
    row, col = ds.samples[0][1:]
    for t, tid in enumerate(TIDS):
        grid = np.load(img_dir / patch_file_name("data", tid, H, W, STRIDE))
        expected = grid[row, col].astype(np.float32)
        got = ds._lazy_sample(0)[0][t]
        assert np.array_equal(got, expected, equal_nan=True), f"channel {t} is not {tid}"


def test_sample_order_is_the_union_oldest_interferogram_first(tree):
    """Sample order drives the validation grid's spatial heuristic — pin it."""
    ds = make(tree, temporal=True, seq_dict=SEQ)
    expected_rc, _ = eager_reference(tree, TIDS)
    assert [tuple(s[1:]) for s in ds.samples] == expected_rc


def test_target_is_the_newest_frame_and_union_flag_widens_it(tree):
    ds = make(tree, temporal=True, seq_dict=SEQ)
    ds_union = make(tree, temporal=True, seq_dict=SEQ, union_temporal_mask=True)
    latest = sum(int(ds[i]["mask"].sum()) for i in range(len(ds)))
    union = sum(int(ds_union[i]["mask"].sum()) for i in range(len(ds_union)))
    assert 0 < latest < union, "the union must cover strictly more than the latest frame"


# -- 4. samples are what the eager implementation produced ------------------------------

def test_temporal_samples_match_the_eager_reference(tree):
    _, expected = eager_reference(tree, TIDS)
    assert_matches_reference(make(tree, temporal=True, seq_dict=SEQ), expected)


def test_union_target_matches_the_eager_reference(tree):
    _, expected = eager_reference(tree, TIDS, union=True)
    assert_matches_reference(
        make(tree, temporal=True, seq_dict=SEQ, union_temporal_mask=True), expected)


def test_validity_channels_match_the_eager_reference(tree):
    _, expected = eager_reference(tree, TIDS, nodata=True)
    assert_matches_reference(
        make(tree, temporal=True, seq_dict=SEQ, treat_nodata_regions=True), expected)


def test_masks_survive_preprocessing_unchanged(tree):
    """The returned mask is the raw patch as class indices — no remapping."""
    _, expected = eager_reference(tree, TIDS)
    ds = make(tree, temporal=True, seq_dict=SEQ)
    assert ds.mask_values == [0, 1]
    for idx, (_, target) in enumerate(expected):
        assert torch.equal(ds[idx]["mask"], torch.as_tensor(target).long())


def test_nonz_files_serve_the_single_frame_path(tree):
    """Default single-frame training reads the pre-extracted positive patches."""
    img_dir, msk_dir = tree
    ds = make(tree)
    imgs = np.load(img_dir / patch_file_name("data", TIDS[2], H, W, STRIDE, nonz=True))
    msks = np.load(msk_dir / patch_file_name("mask", TIDS[2], H, W, STRIDE, nonz=True))
    assert len(ds) == len(imgs)
    for idx in range(len(ds)):
        img, msk = ds._lazy_sample(idx)
        assert np.array_equal(img, imgs[idx].astype(np.float32), equal_nan=True)
        assert np.array_equal(msk, msks[idx].astype(np.float32))


def test_full_grid_path_indexes_every_patch_row_major(tree):
    img_dir, _ = tree
    ds = make(tree, nonz_only=False)
    grid = np.load(img_dir / patch_file_name("data", TIDS[2], H, W, STRIDE))
    assert len(ds) == NY * NX
    assert [tuple(s[1:]) for s in ds.samples] == [(i, j) for i in range(NY) for j in range(NX)]
    assert np.array_equal(ds._lazy_sample(NX + 1)[0], grid[1, 1].astype(np.float32),
                          equal_nan=True)


# -- preserved features -----------------------------------------------------------------

def test_ring_negatives_add_empty_patches_around_the_positives(tree):
    plain = make(tree, temporal=True, seq_dict=SEQ)
    ringed = make(tree, temporal=True, seq_dict=SEQ,
                  ring_negatives=RingNegatives(1, 3, 1.0, seed=0))
    assert len(ringed) > len(plain)
    assert [tuple(s[1:]) for s in ringed.samples[:len(plain)]] == \
        [tuple(s[1:]) for s in plain.samples], "positives keep their order"
    for idx in range(len(plain), len(ringed)):
        assert ringed[idx]["mask"].sum() == 0, "a ring negative must be empty"


def test_ring_negatives_are_absent_from_val_splits(tree):
    val = make(tree, temporal=True, seq_dict=SEQ, mode="val",
               ring_negatives=RingNegatives(1, 3, 1.0, seed=0))
    assert len(val) == len(make(tree, temporal=True, seq_dict=SEQ, mode="val"))


@pytest.mark.parametrize("mode,rows,offset", [
    ("train", (0, THRESH_LINE), 0),
    ("val", (THRESH_LINE, NY), THRESH_LINE),
])
def test_spatial_temporal_split_matches_the_eager_reference(tree, mode, rows, offset):
    _, expected = eager_reference(tree, TIDS, rows=rows,
                                  offsets={tid: offset for tid in TIDS})
    ds = make(tree, mode=mode, temporal=True, seq_dict=SEQ, spatial=True,
              coord_dict=COORD, thresh_lat=THRESH_LAT)
    assert_matches_reference(ds, expected)


def test_spatial_split_is_disjoint_and_covers_the_grid(tree):
    kwargs = dict(spatial=True, coord_dict=COORD, thresh_lat=THRESH_LAT, nonz_only=False)
    north = make(tree, mode="train", **kwargs)
    south = make(tree, mode="val", **kwargs)
    rows_n = {s[1] for s in north.samples}
    rows_s = {s[1] for s in south.samples}
    assert rows_n and rows_s and not (rows_n & rows_s)
    assert max(rows_n) < THRESH_LINE <= min(rows_s)
    assert len(north) + len(south) == NY * NX


def test_spatial_single_frame_split_keeps_only_positive_patches(tree):
    ds = make(tree, mode="val", spatial=True, coord_dict=COORD, thresh_lat=THRESH_LAT)
    assert len(ds) > 0
    for idx in range(len(ds)):
        assert ds[idx]["mask"].sum() > 0


def test_null_patch_sampling_is_deterministic_under_a_seed(tree):
    def build():
        random.seed(7)
        return make(tree, nonz_only=True, add_nulls_to_train=True)

    first, second = build(), build()
    assert [tuple(s[1:]) for s in first.samples] == [tuple(s[1:]) for s in second.samples]
    # Positives are always kept; the no-data streak at (0, 0) is a candidate.
    positives = {tuple(c) for c in POSITIVES[TIDS[2]]}
    assert positives <= {tuple(s[1:]) for s in first.samples}
    assert len(first) > len(positives)


def test_non_binary_masks_are_rejected(tmp_path, tree):
    img_dir, msk_dir = tree
    path = msk_dir / patch_file_name("mask", TIDS[2], H, W, STRIDE, nonz=True)
    masks = np.load(path)
    masks[0, 0, 0] = 3
    np.save(path, masks)
    clear_grid_cache()
    with pytest.raises(RuntimeError, match="binary"):
        make(tree)


# -- the mmap cache ---------------------------------------------------------------------

def test_grids_are_memory_mapped_and_cached(tree):
    img_dir, _ = tree
    clear_grid_cache()
    path = str(img_dir / patch_file_name("data", TIDS[2], H, W, STRIDE))
    grid = load_grid(path)
    assert isinstance(grid, np.memmap)
    assert load_grid(path) is grid, "repeated reads must reuse the open map"
    clear_grid_cache()
    assert load_grid(path) is not grid


def test_cache_is_bounded(tree):
    assert ds_mod.load_grid.cache_info().maxsize == ds_mod.GRID_CACHE_SIZE
    clear_grid_cache()
    ds = make(tree, temporal=True, seq_dict=SEQ, treat_nodata_regions=True,
              union_temporal_mask=True)
    for idx in range(len(ds)):
        ds[idx]
    info = ds_mod.load_grid.cache_info()
    assert info.currsize <= ds_mod.GRID_CACHE_SIZE
    assert info.hits > info.misses, "consecutive samples should hit the cache"
