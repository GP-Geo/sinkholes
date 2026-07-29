"""Prediction on new raw interferograms (no ground truth).

Run as ``sinkholes predict``. Reads ``.unw`` scenes, aligns the temporal chain
onto a common grid, crops the model's column window, reconstructs the
LiDAR-gated prediction through the shared engine and exports the detected
polygons as a lon/lat shapefile per interferogram.

The exported longitudes are computed from the common origin plus
``--x_pxls_offset`` — it must match how the scene was cropped before tiling
(default 3000 columns), or every polygon shifts silently.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from ..geo import PREDICT_X_OFFSET, crop_to_start_xy
from ..meta import find_11day_sequences, intf_meta, load_coord_dict
from ..normalise import SCENE_RANGE_TOL, normalise_phase
from ..polygons import mask_array_to_polygons, pixel_polygons_to_lonlat
from .reconstruct import canvas_shape, rasterise_lidar_gates, reconstruct_scene


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--intfs_dir", type=str, default="./", help="directory of the raw .unw scenes")
    p.add_argument("--intfs_list", type=str, required=True,
                   help="comma list of interferogram ids (YYYYMMDD_YYYYMMDD)")
    p.add_argument("--model", "-m", type=str, required=True, help="checkpoint path")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--strdpp", type=int, default=2, help="strides per patch")
    p.add_argument("--rth", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--k_prevs", type=int, default=0)
    p.add_argument("--x_pxls_offset", type=int, default=PREDICT_X_OFFSET,
                   help="columns cropped off the west edge before tiling; polygon "
                        "longitudes are referenced to origin + this offset")
    p.add_argument("--blend_type", type=str, default=None, choices=["hann"])
    p.add_argument("--window_gamma", type=float, default=1.0)
    p.add_argument("--add_lidar_mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--treat_nodata_regions", action="store_true")
    p.add_argument("--attn_unet", action="store_true")
    p.add_argument("--add_attn", action="store_true")
    p.add_argument("--convlstm_unet", action="store_true")
    p.add_argument("--output_polygs_dir", type=str, default="out_polygs")
    p.add_argument("--intf_dict_path", type=str, default=None)
    p.add_argument("--add_gt_polygs", action="store_true",
                   help="also load ground-truth polygons for a comparison figure")
    p.add_argument("--gt_polygons_file_path", type=str, default="sub_20231001.shp")
    p.add_argument("--unified_mask", action="store_true",
                   help="ground-truth overlay unions the polygons of the whole chain")
    p.add_argument("--plot_polygs", action="store_true",
                   help="save a scene + polygons overview PNG next to the shapefile")


def read_unw_scene(intfs_dir: str, intf_id: str, meta) -> np.ndarray:
    """The raw scene raster of one interferogram, byte-swapped if needed."""
    filename = next(
        (f for f in os.listdir(intfs_dir)
         if f.endswith(".unw") and intf_id[:8] in f and intf_id[9:] in f),
        None,
    )
    if filename is None:
        raise FileNotFoundError(f"no .unw scene for {intf_id} under {intfs_dir}")
    data = np.fromfile(os.path.join(intfs_dir, filename), dtype=np.float32)
    data = data.reshape(meta.nlines, meta.ncells)
    if meta.byte_order == "MSBFirst":
        data = data.byteswap().view(data.dtype.newbyteorder("<"))
    return data


def tile_view(scene: np.ndarray, patch_size, stride) -> np.ndarray:
    """A zero-copy (ny, nx, H, W) sliding-window view over a full scene."""
    patch_h, patch_w = patch_size
    step_y, step_x = patch_h // stride, patch_w // stride
    windows = np.lib.stride_tricks.sliding_window_view(scene, (patch_h, patch_w))
    return windows[::step_y, ::step_x]


def build_unified_gt_polygons(gdf, intf_ids):
    """Ground-truth polygons of the listed interferograms, unioned into one geometry."""
    import geopandas as gpd
    import pandas as pd

    def dashed(d):
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    parts = []
    for intf_id in intf_ids:
        sel = gdf[(gdf["start_date"] == dashed(intf_id[:8])) & (gdf["end_date"] == dashed(intf_id[9:]))]
        if not sel.empty:
            parts.append(sel)
    if not parts:
        logging.warning(f"no ground truth for {intf_ids}")
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs), None
    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=gdf.crs)
    unified = merged.union_all()
    return gpd.GeoDataFrame(geometry=[unified], crs=gdf.crs), unified


def save_polygon_figure(out_png, scene, extent, polygons, gt_polygons=None):
    """Scene with predicted (red) and ground-truth (blue) polygon outlines."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 10), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.imshow(scene, extent=extent, cmap="gray")

    def draw(gdf, color):
        if gdf is None:
            return
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for poly in polys:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=0.8)

    draw(polygons, "red")
    draw(gt_polygons, "blue")
    fig.tight_layout()
    fig.savefig(out_png)


def main(args) -> None:
    import geopandas as gpd
    import torch

    from ..device import get_device
    from ..models.factory import architecture_from_flags, build_from_checkpoint

    logging.basicConfig(level=logging.INFO)
    patch_h, patch_w = args.patch_size
    intfs_list = args.intfs_list.split(",")
    coord_dict = load_coord_dict(args.intf_dict_path)

    prev_dict = {}
    if args.k_prevs > 0:
        prev_dict, valid = find_11day_sequences(coord_dict, k_prev=args.k_prevs,
                                                restrict_to=intfs_list)
        missing = sorted(set(intfs_list) - set(valid))
        if missing:
            raise SystemExit(
                f"no full {args.k_prevs}-previous chains for {missing}; these scenes "
                f"cannot be predicted with a temporal model"
            )

    device = get_device()
    state_dict = torch.load(args.model, map_location=device)
    loaded = build_from_checkpoint(
        state_dict,
        arch=architecture_from_flags(
            attn_unet=args.attn_unet, add_attn=args.add_attn, convlstm_unet=args.convlstm_unet
        ),
        n_classes=1,
        bilinear=False,
        treat_nodata_regions=args.treat_nodata_regions,
    )
    net = loaded.model
    net.to(device=device)
    net.load_state_dict(state_dict)
    net.eval()
    logging.info(f"architecture {loaded.architecture} ({loaded.n_channels} in-channels) on {device}")
    num_c = args.k_prevs + 1
    if loaded.architecture != "convlstm_unet" and loaded.n_channels not in (None, num_c):
        logging.warning(f"checkpoint expects {loaded.n_channels} input channels but "
                        f"--k_prevs {args.k_prevs} produces {num_c}; "
                        f"set --k_prevs {loaded.n_channels - 1}.")

    out_dir = Path(args.output_polygs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for intf in intfs_list:
        meta = intf_meta(intf, args.intf_dict_path)

        # Chronological chain, current frame last.
        chain_ids = (prev_dict[intf]["prevs"] if args.k_prevs > 0 else []) + [intf]
        metas = [intf_meta(tid, args.intf_dict_path) for tid in chain_ids]
        scenes = [read_unw_scene(args.intfs_dir, tid, m) for tid, m in zip(chain_ids, metas)]

        # Align every date onto the common grid (the intersection origin).
        common_x0 = max(m.east for m in metas)
        common_y0 = min(m.north for m in metas)
        aligned = [
            crop_to_start_xy(scene, None, m.east, m.north, common_x0, common_y0, m.dx, m.dy)[0]
            for scene, m in zip(scenes, metas)
        ]

        # Model column window, matched shapes, per-scene normalisation.
        cropped = [s[:, args.x_pxls_offset :] for s in aligned]
        h = min(s.shape[0] for s in cropped)
        w = min(s.shape[1] for s in cropped)
        cropped = [
            normalise_phase(s[:h, :w].astype(np.float32, copy=False), range_tol=SCENE_RANGE_TOL)
            for s in cropped
        ]
        stack = [tile_view(s, (patch_h, patch_w), args.strdpp) for s in cropped]
        ny, nx = stack[0].shape[:2]

        x_start = common_x0 + args.x_pxls_offset * meta.dx

        lidar_gates = None
        if args.add_lidar_mask:
            sources = [m.lidar_mask for m in metas]
            lidar_gates = rasterise_lidar_gates(
                sources, (x_start, common_y0, meta.dx, meta.dy),
                canvas_shape(ny, nx, (patch_h, patch_w), args.strdpp),
            )

        logging.info(f"{intf}: grid {ny}x{nx}, T={len(stack)}")
        result = reconstruct_scene(
            stack, net, (patch_h, patch_w), args.strdpp, args.rth,
            device=device, lidar_gates=lidar_gates,
            treat_nodata_regions=args.treat_nodata_regions,
            blend=args.blend_type, window_gamma=args.window_gamma,
            average="coverage", accumulate_image=False, log_progress=True,
        )

        polygons = pixel_polygons_to_lonlat(
            mask_array_to_polygons(result.thresholded), x_start, common_y0, meta.dx, meta.dy
        )
        shp_path = out_dir / f"{intf}_predicted_polygons.shp"
        polygons.to_file(shp_path)
        logging.info(f"{intf}: {len(polygons)} polygons -> {shp_path}")

        gt_polygons = None
        if args.add_gt_polygs:
            gdf = gpd.read_file(args.gt_polygons_file_path)
            gt_ids = chain_ids if (args.unified_mask and args.k_prevs > 0) else [intf]
            gt_polygons, _ = build_unified_gt_polygons(gdf, gt_ids)

        if args.plot_polygs:
            scene = cropped[-1]
            extent = [x_start, x_start + meta.dx * scene.shape[1],
                      common_y0 - meta.dy * scene.shape[0], common_y0]
            png = out_dir / f"{intf}_predicted_polygons.png"
            save_polygon_figure(png, scene, extent, polygons, gt_polygons)
            logging.info(f"{intf}: overview figure -> {png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
