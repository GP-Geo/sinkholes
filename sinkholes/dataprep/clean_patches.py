"""Optional patch QC: drop mask polygons that are mostly no-data or hug edges.

Run as ``sinkholes clean-patches``. For every mask patch, polygons whose
interior is mostly raw-zero (no-data) pixels, or that are small and touch the
patch boundary (annotation slivers from the patch cut), are removed; the
cleaned grids and nonz subsets land in ``cleaned/`` subdirectories and are
selected at training time with ``--use_cleaned_patches``.
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from ..meta import INTF_ID_RE
from .patchify import patch_dir_name


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--patches_path", type=str, required=True,
                   help="root holding the {data,mask}_patches_* trees")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--strides_per_patch", type=int, default=2)
    p.add_argument("--days_diff", type=int, default=11)
    p.add_argument("--min_area", type=float, default=20,
                   help="edge-touching polygons below this area are dropped")
    p.add_argument("--nodata_frac", type=float, default=0.7,
                   help="polygons with more than this fraction of raw-zero pixels are dropped")


def clean_mask_patch(mask_patch, data_patch, patch_boundary, min_area, nodata_frac):
    """The cleaned mask, and whether anything was removed."""
    import rasterio.features
    from affine import Affine
    from rasterio.features import rasterize
    from shapely.geometry import mapping, shape

    H, W = mask_patch.shape
    polygons = [shape(g) for g, v in
                rasterio.features.shapes(mask_patch, mask_patch > 0) if v > 0]
    kept = []
    removed = False
    for poly in polygons:
        xs = [pt[0] for pt in poly.exterior.coords]
        ys = [pt[1] for pt in poly.exterior.coords]
        inside = rasterize([(poly, 1)], out_shape=mask_patch.shape, fill=0, dtype=np.uint8) == 1
        zero_frac = float((data_patch[inside] == 0).sum()) / poly.area

        thin_at_x_edge = (max(xs) - min(xs) < 5) and (max(xs) == W or min(xs) == 1)
        thin_at_y_edge = (max(ys) - min(ys) < 5) and (max(ys) == H or min(ys) == 1)
        small_on_boundary = poly.area < min_area and poly.touches(patch_boundary)

        if zero_frac > nodata_frac or small_on_boundary or thin_at_x_edge or thin_at_y_edge:
            removed = True
        else:
            kept.append(poly)

    if not kept:
        return np.zeros_like(mask_patch, dtype=np.uint8), removed
    cleaned = rasterize(
        [(mapping(p), 1) for p in kept],
        out_shape=(H, W), transform=Affine.identity(), fill=0, all_touched=True, dtype=np.uint8,
    )
    return cleaned, removed


def main(args) -> None:
    from shapely.geometry import LineString

    logging.basicConfig(level=logging.INFO)
    H, W = args.patch_size
    root = Path(args.patches_path)
    mask_dir = root / patch_dir_name("mask", H, W, args.strides_per_patch, args.days_diff)
    data_dir = root / patch_dir_name("data", H, W, args.strides_per_patch, args.days_diff)
    cleaned_mask_dir = mask_dir / "cleaned"
    cleaned_data_dir = data_dir / "cleaned"
    for d in (cleaned_mask_dir, cleaned_data_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    patch_boundary = LineString([(0, 0), (W, 0), (W, H), (0, H), (0, 0)])
    mask_files = sorted(f for f in mask_dir.iterdir()
                        if f.is_file() and "nonz" not in f.name and f.suffix == ".npy")

    for mask_file in mask_files:
        intf_id = INTF_ID_RE.search(mask_file.name).group(0)
        data_file = data_dir / mask_file.name.replace("mask_", "data_", 1)
        mask_grid = np.load(mask_file).astype(np.float32)
        data_grid = np.load(data_file).astype(np.float32)

        orig_shape = mask_grid.shape
        if mask_grid.ndim == 4:
            mask_grid = mask_grid.reshape(-1, H, W)
            data_grid = data_grid.reshape(-1, H, W)

        cleaned, nonz_masks, nonz_data = [], [], []
        n_changed = 0
        for i in range(mask_grid.shape[0]):
            if not np.any(mask_grid[i]):
                cleaned.append(mask_grid[i].astype(np.uint8))
                continue
            cleaned_patch, removed = clean_mask_patch(
                mask_grid[i], data_grid[i], patch_boundary, args.min_area, args.nodata_frac
            )
            cleaned.append(cleaned_patch)
            n_changed += int(removed)
            if cleaned_patch.any():
                nonz_masks.append(cleaned_patch)
                nonz_data.append(data_grid[i])

        cleaned = np.array(cleaned)
        if len(orig_shape) == 4:
            cleaned = cleaned.reshape(orig_shape)
        stem = mask_file.stem
        np.save(cleaned_mask_dir / f"{stem}_cleaned.npy", cleaned)
        np.save(cleaned_mask_dir / f"{stem.replace('mask_patches_', 'mask_patches_nonz_')}_cleaned.npy",
                np.array(nonz_masks))
        np.save(cleaned_data_dir / f"{stem.replace('mask_patches_', 'data_patches_nonz_')}_cleaned.npy",
                np.array(nonz_data))
        logging.info(f"{intf_id}: {n_changed} patches changed, "
                     f"{len(nonz_masks)} positive patches remain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
