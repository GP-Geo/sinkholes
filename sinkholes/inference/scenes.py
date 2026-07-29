"""Full-interferogram evaluation: aligned patch grids -> reconstructed scene,
LiDAR-gated prediction, polygons and saved arrays.

Run as ``sinkholes eval-scenes``. For each interferogram it loads the full
patch grids (current + k previous), reconstructs the scene through the shared
engine, thresholds, polygonises and georeferences against the aligned frame
origin. Outputs land under ``<output_dir>/<model>/<job>_<ts>/``:
``<intf>_image.npy`` (C, H, W; channel 0 = current, then previous frames
newest-first), ``_pred.npy`` (confidence), ``_pred_th.npy``, ``_gt.npy``, and
``polygs/<intf>_predicted_polygs.shp`` (EPSG:4326).
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ..geo import aligned_origin
from ..meta import INTF_ID_RE, find_11day_sequences, intf_meta, load_coord_dict
from ..normalise import SCENE_RANGE_TOL, normalise_phase
from ..paths import DEFAULT_PREDICTIONS_DIR
from ..polygons import mask_array_to_polygons, pixel_polygons_to_lonlat
from .reconstruct import canvas_shape, rasterise_lidar_gates, reconstruct_scene


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", type=str, required=True, help="checkpoint path (.pt/.pth)")
    p.add_argument("--input_patch_dir", type=str, required=True,
                   help="directory containing the {data,mask}_patches_* trees")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--data_stride", type=int, default=2)
    p.add_argument("--days_diff", type=int, default=11)
    p.add_argument("--recon_th", type=float, default=0.25,
                   help="threshold on the reconstructed confidence map")

    p.add_argument("--intf_source", type=str, default="intf_list",
                   choices=["intf_list", "test_dataset", "preset", "all"])
    p.add_argument("--intf_list", type=str, default=None, help="comma list of interferogram ids")
    p.add_argument("--test_dataset", type=str, default=None,
                   help="pickled test split whose interferograms to evaluate")
    p.add_argument("--valset_from_partition", type=str, default=None,
                   help="partition JSON whose 'val' list to evaluate")
    p.add_argument("--year_range", nargs=2, type=int, default=[2019, 2023],
                   metavar=("Y0", "Y1"), help="year filter for --intf_source all")

    p.add_argument("--k_prevs", type=int, default=0,
                   help="temporal context; must match how the model was trained")
    p.add_argument("--treat_nodata_regions", action="store_true",
                   help="the checkpoint was trained with validity channels")
    p.add_argument("--replicate_input", action="store_true",
                   help="feed the current interferogram in every temporal slot")
    p.add_argument("--fallback_replicate", action="store_true",
                   help="replicate the current frame when predecessors are missing "
                        "instead of skipping the interferogram")
    p.add_argument("--unioned_mask", action="store_true",
                   help="ground truth = union of masks over the temporal stack")

    p.add_argument("--add_lidar_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="predict only tiles inside the LiDAR coverage of every timestep")
    p.add_argument("--blend_type", type=str, default=None, choices=["hann"],
                   help="Hann-window blending of overlapping predictions")
    p.add_argument("--window_gamma", type=float, default=1.0)

    p.add_argument("--attn_unet", action="store_true")
    p.add_argument("--add_attn", action="store_true")
    p.add_argument("--convlstm_unet", action="store_true")

    p.add_argument("--job_name", type=str, default="job")
    p.add_argument("--output_dir", type=str, default=DEFAULT_PREDICTIONS_DIR)
    p.add_argument("--intf_dict_path", type=str, default=None)
    p.add_argument("--save_confidence", action="store_true")
    p.add_argument("--merge_polygs", action="store_true",
                   help="also write one combined shapefile across all interferograms")


def resolve_intf_list(args, data_dir) -> list:
    if args.intf_source == "intf_list":
        if not args.intf_list:
            sys.exit("--intf_source intf_list needs --intf_list")
        return args.intf_list.split(",")
    if args.intf_source == "test_dataset":
        if not args.test_dataset:
            sys.exit("--intf_source test_dataset needs --test_dataset")
        from ..dataprep.dataset import load_test_dataset

        return load_test_dataset(args.test_dataset).ids
    if args.intf_source == "preset":
        if not args.valset_from_partition:
            sys.exit("--intf_source preset needs --valset_from_partition")
        with open(args.valset_from_partition) as fh:
            return json.load(fh)["val"]
    # all
    y0, y1 = args.year_range
    ids = set()
    for name in os.listdir(data_dir):
        if not name.endswith(".npy") or "nonz" in name:
            continue
        m = INTF_ID_RE.search(name)
        if m and y0 <= int(m.group(0)[:4]) <= y1:
            ids.add(m.group(0))
    logging.info(f"year range {y0}-{y1}: found {len(ids)} interferograms")
    return sorted(ids)


def load_model(args, device):
    import torch

    from ..models.factory import architecture_from_flags, build_from_checkpoint

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
    logging.info(f"architecture {loaded.architecture} ({loaded.n_channels} in-channels)")
    num_c = (args.k_prevs + 1) * (2 if args.treat_nodata_regions else 1)
    if loaded.architecture != "convlstm_unet" and loaded.n_channels not in (None, num_c):
        logging.warning(
            f"checkpoint expects {loaded.n_channels} input channels but --k_prevs "
            f"{args.k_prevs} produces {num_c}; set --k_prevs "
            f"{loaded.n_channels // (2 if args.treat_nodata_regions else 1) - 1} to match."
        )
    net = loaded.model
    net.to(device=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def main(args) -> None:
    import geopandas as gpd
    import pandas as pd

    from ..device import get_device

    logging.basicConfig(level=logging.INFO)
    now = datetime.now().strftime("%m_%d_%Hh%M")
    job_name = f"{args.job_name}_{now}"
    model_name = Path(args.model).stem
    output_path = os.path.join(args.output_dir, model_name, job_name)
    polyg_dir = os.path.join(output_path, "polygs")
    os.makedirs(polyg_dir, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(os.path.join(output_path, f"{job_name}.log")))
    logging.info(f"eval-scenes: model={args.model} args={vars(args)}")

    patch_h, patch_w = args.patch_size
    from ..dataprep.patchify import patch_dir_name, patch_file_name

    data_dir = os.path.join(args.input_patch_dir,
                            patch_dir_name("data", patch_h, patch_w, args.data_stride, args.days_diff))
    mask_dir = os.path.join(args.input_patch_dir,
                            patch_dir_name("mask", patch_h, patch_w, args.data_stride, args.days_diff))

    intf_list = resolve_intf_list(args, data_dir)

    device = get_device()
    logging.info(f"loading model on {device}")
    net = load_model(args, device)

    coord_dict = load_coord_dict(args.intf_dict_path)

    prev_dict = None
    if args.k_prevs > 0 and not args.replicate_input:
        prev_dict, with_chains = find_11day_sequences(
            coord_dict, k_prev=args.k_prevs, restrict_to=intf_list,
            require_current_nonz_gt0=False,
        )
        if not args.fallback_replicate:
            dropped = sorted(set(intf_list) - set(with_chains))
            if dropped:
                logging.warning(f"skipping {len(dropped)} interferograms without full "
                                f"{args.k_prevs}-previous chains: {dropped} "
                                f"(--fallback_replicate keeps them)")
            intf_list = with_chains

    def grid_path(kind, tid):
        d = data_dir if kind == "data" else mask_dir
        return os.path.join(d, patch_file_name(kind, tid, patch_h, patch_w, args.data_stride))

    for intf in intf_list:
        meta = intf_meta(intf, args.intf_dict_path)
        x0a, y0a = aligned_origin(meta.frame)
        cur = np.load(grid_path("data", intf)).astype(np.float32)  # (ny, nx, H, W)

        # -- assemble the temporal stack -------------------------------------------------
        # prev_ids are held newest-first here (matching how the historical
        # output stack is saved); the chronological order the network needs is
        # produced by one reversal at the end. Predecessors whose grid file is
        # missing are padded with the current frame in the oldest slots.
        if args.k_prevs > 0:
            if args.replicate_input or (args.fallback_replicate and intf not in (prev_dict or {})):
                prev_ids = []
                arrays_newest_first = [cur] * (args.k_prevs + 1)
            else:
                prev_ids = prev_dict[intf]["prevs"][: args.k_prevs][::-1]  # newest first
                available = [pid for pid in prev_ids if os.path.exists(grid_path("data", pid))]
                prevs = [np.load(grid_path("data", pid)).astype(np.float32) for pid in available]
                while len(prevs) < args.k_prevs:
                    prevs.append(cur)
                arrays_newest_first = [cur] + prevs
        else:
            prev_ids = []
            arrays_newest_first = [cur]

        # Crop every timestep to the common grid, then normalise per scene.
        ny = min(p.shape[0] for p in arrays_newest_first)
        nx = min(p.shape[1] for p in arrays_newest_first)
        arrays_newest_first = [
            normalise_phase(p[:ny, :nx, :patch_h, :patch_w].astype(np.float32, copy=False),
                            range_tol=SCENE_RANGE_TOL)
            for p in arrays_newest_first
        ]
        stack = arrays_newest_first[::-1]  # chronological, current frame last

        # -- ground truth ----------------------------------------------------------------
        mask_cur = np.load(grid_path("mask", intf)).astype(np.float32)[:ny, :nx, :patch_h, :patch_w]
        if args.unioned_mask and args.k_prevs > 0 and not args.replicate_input:
            prev_masks = [
                np.load(grid_path("mask", pid)).astype(np.float32)[:ny, :nx, :patch_h, :patch_w]
                for pid in prev_ids if os.path.exists(grid_path("mask", pid))
            ]
            gt_grid = (np.stack([mask_cur] + prev_masks, axis=0) > 0).any(axis=0).astype(np.float32)
        else:
            gt_grid = mask_cur

        # -- LiDAR gates (AND across the current frame and every predecessor) ------------
        lidar_gates = None
        if args.add_lidar_mask:
            sources = [meta.lidar_mask] + [
                intf_meta(pid, args.intf_dict_path).lidar_mask for pid in prev_ids
            ]
            if len(sources) < len(stack):  # replicated slots gate on the current source
                sources += [meta.lidar_mask] * (len(stack) - len(sources))
            lidar_gates = rasterise_lidar_gates(
                sources, (x0a, y0a, meta.dx, meta.dy),
                canvas_shape(ny, nx, (patch_h, patch_w), args.data_stride),
            )

        logging.info(f"{intf}: grid {ny}x{nx}, T={len(stack)}")
        result = reconstruct_scene(
            stack, net, (patch_h, patch_w), args.data_stride, args.recon_th,
            device=device, gt_grid=gt_grid, lidar_gates=lidar_gates,
            treat_nodata_regions=args.treat_nodata_regions,
            blend=args.blend_type, window_gamma=args.window_gamma,
            average="uniform", log_progress=True,
        )

        polygons = pixel_polygons_to_lonlat(
            mask_array_to_polygons(result.thresholded), x0a, y0a, meta.dx, meta.dy
        )
        shp_path = os.path.join(polyg_dir, f"{intf}_predicted_polygs.shp")
        polygons.to_file(shp_path)
        logging.info(f"{intf}: {len(polygons)} polygons -> {shp_path}")

        prefix = os.path.join(output_path, intf)
        # Saved channel order is current first, then previous frames
        # newest-first — the historical format the outputs command reads.
        image_to_save = result.image[::-1].astype(np.float32)
        if args.save_confidence:
            np.save(prefix + "_pred", result.confidence)
        if args.intf_source != "all":
            np.save(prefix + "_image", image_to_save)
            np.save(prefix + "_pred_th", result.thresholded)
            np.save(prefix + "_pred", result.confidence)
            if result.gt is not None:
                np.save(prefix + "_gt", result.gt)

    if args.merge_polygs:
        shp_files = sorted(glob.glob(os.path.join(polyg_dir, "*.shp")))
        logging.info(f"merging {len(shp_files)} per-intf shapefiles")
        gdfs = []
        for shp in shp_files:
            gdf = gpd.read_file(shp)
            if gdf.empty:
                continue
            key = os.path.basename(shp)[:17]
            frame = coord_dict.get(key, {}).get("frame", "")
            gdf["intf_key"] = key
            gdf["start_date"] = key[:8]
            gdf["end_date"] = key[9:]
            gdf["track"] = "asc" if frame == "North" else ("desc" if frame == "South" else None)
            gdfs.append(gdf)
        if gdfs:
            combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            combined = combined[combined.geometry.notna() & ~combined.geometry.is_empty]
            y0, y1 = args.year_range
            out = os.path.join(output_path, f"{model_name}_{y0}_{y1}_combined.shp")
            combined.to_file(out)
            logging.info(f"combined shapefile -> {out} ({len(combined)} polygons)")
        else:
            logging.warning("no polygons to merge")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
