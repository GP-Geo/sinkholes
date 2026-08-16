"""Patchify geometry, normalisation, georeferencing, and dataset assembly
against synthetic grids (plus the real local subset when it is present)."""

from pathlib import Path

import numpy as np
import pytest

from sinkholes.dataprep.dataset import RingNegatives, SubsiDataset, load_test_dataset
from sinkholes.dataprep.partition import split_nonoverlap_patches
from sinkholes.dataprep.patchify import patch_dir_name, patch_file_name, patch_strides, patchify
from sinkholes.geo import FRAME_ORIGINS, aligned_origin, crop_to_start_xy
from sinkholes.normalise import (
    SCENE_RANGE_TOL,
    normalise_phase,
    validity_from_normalised,
)
from sinkholes.polygons import mask_array_to_polygons, pixel_polygons_to_lonlat

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PATCHES = REPO_ROOT / "data" / "patches" / "train_ready"


# -- patchify ---------------------------------------------------------------------------

def test_patchify_grid_geometry_matches_the_naming_convention():
    """(ny, nx) follow from rows/cols, window and stride; nx=4500 crop gives 89 cols."""
    rows, cols = 700, 5000
    arr = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    H, W = 200, 100
    Sy, Sx = patch_strides((H, W), 2)
    assert (Sy, Sx) == (100, 50)

    grid = patchify(arr, (H, W), (Sy, Sx))
    ny = (rows - H) // Sy + 1
    nx = (4500 - W) // Sx + 1
    assert grid.shape == (ny, nx, H, W)
    assert nx == 89
    # Patch (i, j) covers rows [i*Sy, i*Sy+H), cols [j*Sx, j*Sx+W) of the crop.
    assert np.array_equal(grid[2, 3], arr[2 * Sy : 2 * Sy + H, 3 * Sx : 3 * Sx + W])


def test_patchify_collects_positive_patches_with_grid_coordinates():
    arr = np.zeros((400, 200), dtype=np.float32)
    mask = np.zeros_like(arr, dtype=np.uint8)
    mask[250:260, 60:70] = 1  # inside patches around grid row 1-2, col 0-1

    data, masks, d_nonz, m_nonz, idx = patchify(
        arr, (200, 100), (100, 50), mask_array=mask, nx=200
    )
    assert data.shape[:2] == masks.shape[:2]
    assert len(idx) == len(d_nonz) == len(m_nonz) > 0
    for (i, j), m in zip(idx, m_nonz):
        assert m.any()
        assert np.array_equal(masks[i, j], m)
    # Row-major emission order.
    assert idx == sorted(idx)


def test_patchify_empty_mask_yields_empty_nonz_arrays():
    arr = np.zeros((400, 200), dtype=np.float32)
    mask = np.zeros_like(arr, dtype=np.uint8)
    _, _, d_nonz, m_nonz, idx = patchify(arr, (200, 100), (100, 50), mask_array=mask, nx=200)
    assert d_nonz.shape == (0, 200, 100)
    assert m_nonz.shape == (0, 200, 100)
    assert idx == []


def test_patch_naming_roundtrip():
    assert patch_dir_name("data", 200, 100, 2) == "data_patches_H200_W100_strpp2_11days_Aligned"
    assert (
        patch_file_name("mask", "20190205_20190216", 200, 100, 2, nonz=True)
        == "mask_patches_nonz_20190205_20190216_H200_W100_strpp2.npy"
    )


# -- normalisation ----------------------------------------------------------------------

def test_normalise_passthrough_maps_exact_zero_to_half():
    x = np.array([0.0, 0.25, 0.9997], dtype=np.float32)
    out = normalise_phase(x, range_tol=SCENE_RANGE_TOL)
    assert out[0] == 0.5
    assert out[1] == np.float32(0.25)
    assert out[2] == np.float32(0.9997)


def test_normalise_rescales_radians():
    x = np.array([-np.pi, 0.0, np.pi], dtype=np.float32)
    out = normalise_phase(x, range_tol=SCENE_RANGE_TOL)
    assert out == pytest.approx([0.0, 0.5, 1.0], abs=1e-6)


def test_validity_marks_only_the_nodata_code():
    x = np.array([0.5, 0.5 + 1e-6, 0.0, 1.0], dtype=np.float32)
    v = validity_from_normalised(x)
    assert v.tolist() == [0.0, 1.0, 1.0, 1.0]


# -- geo --------------------------------------------------------------------------------

def test_aligned_origins_are_the_documented_constants():
    """These anchor every patch grid and every exported polygon."""
    assert FRAME_ORIGINS["North"] == (35.37, 31.79)
    assert FRAME_ORIGINS["South"] == (35.32, 31.44)
    assert aligned_origin("North") == (35.37, 31.79)
    with pytest.raises(ValueError):
        aligned_origin("East")


def test_crop_to_start_xy_shifts_by_whole_pixels():
    dx = dy = 2.777e-05
    arr = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    x0, y0 = 35.0, 32.0
    cropped, _, new_x0, new_y0, (row_off, col_off) = crop_to_start_xy(
        arr, None, x0, y0, x0 + 3 * dx, y0 - 5 * dy, dx, dy
    )
    assert (row_off, col_off) == (5, 3)
    assert cropped.shape == (15, 27)
    assert np.array_equal(cropped, arr[5:, 3:])
    assert new_x0 == pytest.approx(x0 + 3 * dx)
    assert new_y0 == pytest.approx(y0 - 5 * dy)


def test_crop_to_start_xy_rejects_subpixel_targets():
    dx = dy = 2.777e-05
    arr = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="pixel grid"):
        crop_to_start_xy(arr, None, 35.0, 32.0, 35.0 + 0.4 * dx, 32.0, dx, dy, tol=1e-9)


# -- polygons ---------------------------------------------------------------------------

def test_polygon_georeferencing_uses_origin_and_pixel_size():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:4, 3:5] = 1.0
    gdf = mask_array_to_polygons(mask)
    assert len(gdf) == 1

    x_start, y0, dx, dy = 35.37, 31.79, 0.01, 0.01
    lonlat = pixel_polygons_to_lonlat(gdf, x_start, y0, dx, dy)
    minx, miny, maxx, maxy = lonlat.total_bounds
    assert minx == pytest.approx(x_start + 3 * dx)
    assert maxx == pytest.approx(x_start + 5 * dx)
    assert maxy == pytest.approx(y0 - 2 * dy)
    assert miny == pytest.approx(y0 - 4 * dy)


def test_polygon_count_matches_connected_components():
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[1:3, 1:3] = 1
    mask[10:12, 10:12] = 1
    mask[17, 5] = 1
    assert len(mask_array_to_polygons(mask)) == 3


# -- ring negatives ---------------------------------------------------------------------

def test_ring_negatives_sample_only_empty_cells_in_the_annulus():
    union_grid = np.zeros((9, 9), dtype=bool)
    union_grid[4, 4] = True  # the positive patch itself
    union_grid[4, 5] = True  # a neighbouring positive: never a negative
    positives = [(4, 4)]

    ring = RingNegatives(inner=1, outer=3, per_pos=50.0, seed=0)
    negs = ring.sample(positives, union_grid)

    assert negs, "there are empty cells in the annulus"
    for (i, j) in negs:
        assert max(abs(i - 4), abs(j - 4)) > 1, "inner ring must be excluded"
        assert max(abs(i - 4), abs(j - 4)) <= 3, "outside the outer ring"
        assert not union_grid[i, j], "negatives must be empty at every timestep"


def test_ring_negatives_count_scales_with_positives():
    union_grid = np.zeros((20, 20), dtype=bool)
    union_grid[10, 10] = True
    ring = RingNegatives(inner=1, outer=3, per_pos=1.0, seed=0)
    assert len(ring.sample([(10, 10)], union_grid)) == 1


# -- non-overlap split ------------------------------------------------------------------

def test_nonoverlap_split_keeps_train_away_from_test():
    import random

    random.seed(0)
    ny, nx, H, W = 6, 6, 4, 4
    images = np.random.rand(ny, nx, H, W).astype(np.float32)
    masks = np.zeros((ny, nx, H, W), dtype=np.float32)
    masks[:, :, 0, 0] = 1  # every patch positive

    tr_img, tr_msk, te_img, te_msk = split_nonoverlap_patches(images, masks, test_fraction=0.2)
    assert len(te_img) == int(0.2 * ny * nx)
    assert len(tr_img) > 0
    # Reconstruct grid coords by matching arrays and check the Chebyshev gap.
    coords = {(i, j): images[i, j].tobytes() for i in range(ny) for j in range(nx)}
    inv = {v: k for k, v in coords.items()}
    test_coords = [inv[t.tobytes()] for t in te_img]
    for t in tr_img:
        ti, tj = inv[t.tobytes()]
        for (i, j) in test_coords:
            assert max(abs(ti - i), abs(tj - j)) >= 2


# -- the real local subset (skipped when the download is absent) ------------------------

needs_local_data = pytest.mark.skipif(
    not LOCAL_PATCHES.exists(), reason="local patch subset not present"
)


@needs_local_data
def test_dataset_loads_real_nonz_patches():
    img_dir = LOCAL_PATCHES / "data_patches_H200_W100_strpp2_11days_Aligned"
    msk_dir = LOCAL_PATCHES / "mask_patches_H200_W100_strpp2_11days_Aligned"
    ds = SubsiDataset(img_dir, msk_dir, ["20190204_20190215"], patch_size=(200, 100), stride=2)
    assert len(ds) == 87  # positive patches of this interferogram
    assert ds.mask_values == [0, 1]
    sample = ds[0]
    assert tuple(sample["image"].shape) == (1, 200, 100)
    assert tuple(sample["mask"].shape) == (200, 100)
    assert sample["image"].min() > 0, "exact zeros must have been remapped to 0.5"
    assert sample["mask"].dtype.__str__() == "torch.int64"


@needs_local_data
def test_dataset_temporal_stack_is_chronological_with_current_last():
    """The loaded (T, H, W) stack must put the current interferogram at index T-1."""
    from sinkholes.meta import find_11day_sequences, load_coord_dict

    img_dir = LOCAL_PATCHES / "data_patches_H200_W100_strpp2_11days_Aligned"
    msk_dir = LOCAL_PATCHES / "mask_patches_H200_W100_strpp2_11days_Aligned"
    coord = load_coord_dict(str(REPO_ROOT / "data" / "metadata" / "intf_coord_local.json"))
    current = "20190216_20190227"
    chains, valid = find_11day_sequences(coord, k_prev=2, restrict_to=[current])
    assert current in valid

    ds = SubsiDataset(
        img_dir, msk_dir, [current], patch_size=(200, 100), stride=2,
        temporal=True, seq_dict=chains,
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (3, 200, 100)

    # Rebuild the same patch from the raw grids to verify channel order.
    import json

    import numpy as np
    with open(img_dir / "nonz_indices.json") as fh:
        nonz = json.load(fh)
    tids = chains[current]["prevs"] + [current]
    grids = [
        np.load(img_dir / f"data_patches_{tid}_H200_W100_strpp2.npy", mmap_mode="r")
        for tid in tids
    ]
    ny = min(g.shape[0] for g in grids)
    nx = min(g.shape[1] for g in grids)
    # Coordinates come from the current interferogram alone — predecessors
    # supply context, not patches of their own.
    rc = [(int(i), int(j)) for i, j in nonz[current]
          if 0 <= int(i) < ny and 0 <= int(j) < nx]
    assert len(ds) == len(rc)

    from sinkholes.normalise import PATCH_RANGE_TOL, normalise_phase

    i0, j0 = rc[0]
    for t, g in enumerate(grids):
        raw = np.array(g[i0, j0], dtype=np.float32)
        expected = normalise_phase(raw, range_tol=PATCH_RANGE_TOL)
        assert np.allclose(sample["image"][t].numpy(), expected), (
            f"channel {t} must be {tids[t]} (oldest -> newest, current last)"
        )


@needs_local_data
def test_legacy_pickled_test_set_still_loads():
    pkl = REPO_ROOT / "test_data" / "test_dataset_run_v2_2026-07-27_14h44.pkl"
    if not pkl.exists():
        pytest.skip("legacy test set not present")
    ds = load_test_dataset(pkl)
    assert isinstance(ds, SubsiDataset)
    assert len(ds) > 0
    sample = ds[0]
    assert tuple(sample["image"].shape) == (1, 200, 100)
    assert tuple(sample["mask"].shape) == (200, 100)
