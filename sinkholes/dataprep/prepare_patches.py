"""Patch generation: raw .unw scenes + GT polygons -> aligned patch grids.

Run as ``sinkholes prepare-patches``. For each interferogram of the requested
duration it rasterises the ground-truth polygons matched by exact start/end
date, aligns the scene to its frame's fixed origin, cuts overlapping patches,
and writes four arrays per interferogram: the full ``(ny, nx, H, W)`` data and
mask grids plus the positive-only ``(N, H, W)`` nonz subsets — training reads
the nonz files, full-scene evaluation the grids, so both are required. Grid
coordinates of the positive patches go to ``nonz_indices.json``.
"""

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from ..geo import FRAME_ORIGINS, X_CROP_COLS, X_CROP_OFFSET, crop_to_start_xy
from ..meta import intf_id_from_filename, intf_meta, parse_intf_id
from .context import assert_centre_matches, context_size
from .patchify import patch_dir_name, patch_file_name, patch_strides, patchify


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input_dir", type=str, required=True, help="directory of the .unw scenes")
    p.add_argument("--output_dir", type=str, required=True, help="root for the patch trees")
    p.add_argument("--gt_polygon_file_path", type=str, required=True,
                   help="ground-truth subsidence polygons (shapefile)")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--context_margin", nargs=2, type=int, default=[0, 0], metavar=("MY", "MX"),
                   help="grow every DATA patch by this many pixels on each side, keeping the "
                        "grid and the 200x100 target unchanged. '50 50' writes a ctx50x50 tree "
                        "whose cells are 300x200 and whose centre is bit-identical to the plain "
                        "cell. Masks are never margined, so --write_masks can be turned off and "
                        "the plain mask tree reused")
    p.add_argument("--write_masks", action=argparse.BooleanOptionalAction, default=True,
                   help="write the mask tree. Off for a context run whose plain mask tree "
                        "already exists — the targets are identical, and rewriting them is "
                        "142 GB of I/O for bit-identical files")
    p.add_argument("--strides_per_patch", type=int, default=2,
                   help="2 means a half-window step (50%% overlap)")
    p.add_argument("--days_diff", type=int, default=11,
                   help="only process interferograms of exactly this duration")
    p.add_argument("--by_list", type=str, default=None, help="comma list of ids to process")
    p.add_argument("--year_range", nargs=2, type=int, default=None, metavar=("Y0", "Y1"))
    p.add_argument("--offset_x", type=int, default=X_CROP_OFFSET)
    p.add_argument("--nx", type=int, default=X_CROP_COLS,
                   help="columns kept after the aligned crop")
    p.add_argument("--intf_dict_path", type=str, default=None)
    p.add_argument("--index_mode", choices=["merge", "replace"], default="merge",
                   help="merge updates processed scenes and preserves other index entries; "
                        "replace rebuilds the index after a successful unfiltered full pass")


def read_nonz_index(path):
    if not path.exists():
        return {}
    with path.open() as fh:
        index = json.load(fh)
    if not isinstance(index, dict):
        raise ValueError(f"{path} must contain a scene-to-coordinates dictionary")
    return index


def write_nonz_index_atomic(path, index):
    """Publish a complete index on the same filesystem; never expose partial JSON.

    Regeneration is a single-writer operation; this is crash protection, not a
    concurrent dataset writer/reader transaction across the separate arrays.
    """
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as fh:
            tmp = Path(fh.name)
            json.dump(index, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main(args) -> None:
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask

    logging.basicConfig(level=logging.INFO)
    index_mode = getattr(args, "index_mode", "merge")
    if index_mode == "replace" and (args.by_list is not None or args.year_range is not None):
        raise ValueError("--index_mode replace requires an unfiltered full pass; use merge for subsets")
    gdf = gpd.read_file(args.gt_polygon_file_path)
    patch_h, patch_w = args.patch_size

    margin = tuple(args.context_margin)
    data_out = Path(args.output_dir) / patch_dir_name(
        "data", patch_h, patch_w, args.strides_per_patch, args.days_diff, margin)
    mask_out = Path(args.output_dir) / patch_dir_name(
        "mask", patch_h, patch_w, args.strides_per_patch, args.days_diff)
    if margin != (0, 0):
        ch, cw = context_size((patch_h, patch_w), margin)
        logging.info(f"context tree: cells are {ch}x{cw}, target stays {patch_h}x{patch_w}, "
                     f"grid and nonz_indices unchanged -> {data_out.name}")
    for d in ((data_out,) if not args.write_masks else (data_out, mask_out)):
        if d.exists():
            logging.info(f"{d} already exists — existing grids will be overwritten by id")
        d.mkdir(parents=True, exist_ok=True)

    wanted = set(args.by_list.split(",")) if args.by_list else None
    out_json = data_out / "nonz_indices.json"
    # Validate existing metadata before overwriting any arrays.
    previous_index = read_nonz_index(out_json)
    nonz_by_intf = previous_index if index_mode == "merge" else {}
    processed = 0

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".unw"):
            continue
        intf_id = intf_id_from_filename(filename)
        if wanted is not None and intf_id not in wanted:
            continue
        start, _, duration = parse_intf_id(intf_id)
        if duration != args.days_diff:
            continue
        if args.year_range is not None and not (args.year_range[0] <= start.year <= args.year_range[1]):
            continue

        meta = intf_meta(intf_id, args.intf_dict_path)
        data = np.fromfile(os.path.join(args.input_dir, filename), dtype=np.float32)
        data = data.reshape(meta.nlines, meta.ncells)
        if meta.byte_order == "MSBFirst":
            data = data.byteswap().view(data.dtype.newbyteorder("<"))

        # GT polygons are matched by exact acquisition dates; no match means an
        # unlabelled scene and an all-zero mask.
        start_date = f"{intf_id[:4]}-{intf_id[4:6]}-{intf_id[6:8]}"
        end_date = f"{intf_id[9:13]}-{intf_id[13:15]}-{intf_id[15:17]}"
        subset = gdf[(gdf["start_date"] == start_date) & (gdf["end_date"] == end_date)]
        if subset.empty:
            logging.warning(f"no GT polygons for {intf_id} ({start_date}..{end_date}); empty mask")
            mask = np.zeros((meta.nlines, meta.ncells), dtype=np.uint8)
        else:
            mask = geometry_mask(
                subset.geometry,
                transform=rasterio.transform.from_origin(meta.east, meta.north, meta.dx, meta.dy),
                invert=True,
                out_shape=(meta.nlines, meta.ncells),
            ).astype(np.uint8)

        x_star, y_star = FRAME_ORIGINS[meta.frame]
        data, mask, *_ = crop_to_start_xy(data, mask, meta.east, meta.north, x_star, y_star,
                                          meta.dx, meta.dy)

        grids = patchify(
            data, (patch_h, patch_w),
            patch_strides((patch_h, patch_w), args.strides_per_patch),
            mask_array=mask, offset=args.offset_x, nx=args.nx, margin=margin,
        )
        data_patches, mask_patches, data_nonz, mask_nonz, nonz_indices = grids

        if margin != (0, 0):
            # Prove the invariant on the grid we just cut, not on a unit-test
            # fixture: the centre of every context cell must be the plain cell.
            plain = patchify(
                data, (patch_h, patch_w),
                patch_strides((patch_h, patch_w), args.strides_per_patch),
                mask_array=None, offset=args.offset_x, nx=args.nx,
            )
            ny_, nx_ = data_patches.shape[:2]
            checked = [(0, 0), (ny_ // 2, nx_ // 2), (ny_ - 1, nx_ - 1)]
            for (ci, cj) in checked:
                assert_centre_matches(data_patches[ci, cj], plain[ci, cj],
                                      (patch_h, patch_w), where=f"{intf_id} cell ({ci},{cj})")
            logging.info(f"{intf_id}: centre-alignment verified on cells {checked}")
        logging.info(f"{intf_id}: grid {data_patches.shape[:2]}, {len(nonz_indices)} positive patches")

        outputs = [("data", False, data_patches), ("data", True, data_nonz)]
        if args.write_masks:
            outputs += [("mask", False, mask_patches), ("mask", True, mask_nonz)]
        for kind, nonz, array in outputs:
            out_dir = data_out if kind == "data" else mask_out
            np.save(out_dir / patch_file_name(kind, intf_id, patch_h, patch_w,
                                              args.strides_per_patch, nonz=nonz), array)
        nonz_by_intf[intf_id] = [[int(i), int(j)] for i, j in nonz_indices]
        processed += 1
        if index_mode == "merge":
            write_nonz_index_atomic(out_json, nonz_by_intf)

    if index_mode == "replace":
        if not processed:
            raise ValueError("no scenes processed; refusing to replace the index with an empty one")
        write_nonz_index_atomic(out_json, nonz_by_intf)
    logging.info(f"{index_mode}: {processed} scenes processed; index at {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
