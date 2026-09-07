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

from ..geo import aligned_origin, grid_window
from ..meta import (
    INTF_ID_RE,
    LIDAR_FALLBACK_SOURCE,
    NO_LIDAR_MASK,
    find_11day_sequences,
    intf_meta,
    lidar_source_for,
    load_coord_dict,
)
from ..dataprep.context import context_margin
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
    p.add_argument("--min_positives", type=int, default=150,
                   help="skip interferograms with this many positive patches or fewer "
                        "(strictly MORE than this is kept). Applies to every "
                        "--intf_source, including an explicit --intf_list, and the "
                        "dropped ids are logged. 0 disables it. NOTE the count is "
                        "'nonz_num' from the coordinate dictionary, which is WHOLE-SCENE: "
                        "it is not restricted to --aoi_window, so a scene with most of "
                        "its positives outside the window can still pass")

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

    p.add_argument("--context_margin", nargs=2, type=int, default=None, metavar=("MY", "MX"),
                   help="optional check of the input margin; by default infer the patch "
                        "tree from checkpoint geometry")
    p.add_argument("--add_lidar_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="predict only tiles inside the LiDAR coverage of every timestep")
    p.add_argument("--aoi_window", nargs=4, type=float, default=None,
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                   help="predict only tiles wholly inside this box. Normally left unset and "
                        "taken from the partition file's aoi_window for --split, which is what "
                        "keeps this command, the dataset and eval-outputs on the same ground; "
                        "pass it explicitly only to score a scene outside any partition")
    p.add_argument("--aoi_from_partition", type=str, default=None,
                   help="partition JSON to read the AOI window from")
    p.add_argument("--aoi_split", type=str, default="test",
                   help="which split's window to read from --aoi_from_partition")
    p.add_argument("--positives_only", action="store_true",
                   help="predict only tiles whose ground truth is non-empty — the "
                        "benchmark paper's protocol ('delineate subsidence where it is "
                        "known to be'). ANDs with the LiDAR gate, honours --unioned_mask, "
                        "and produces numbers that are NOT comparable with full-scene "
                        "ones and must never be used to select a model")
    p.add_argument("--recon_average", type=str, default="uniform",
                   choices=["uniform", "coverage", "vote"],
                   help="how overlapping tile predictions are combined. 'uniform' divides "
                        "the summed probability by stride^2 (every full-scene number before "
                        "2026-08-19). 'coverage' divides by the per-pixel tile count. "
                        "'vote' is the PAPER'S CONFIDENCE FACTOR: each tile is binarised at "
                        "--vote_threshold and the value is the fraction of overlapping tiles "
                        "voting positive, so --recon_th becomes the paper's RTh. Use it with "
                        "--data_stride 4 for the paper's 16-tiles-per-pixel geometry")
    p.add_argument("--vote_threshold", type=float, default=0.5,
                   help="probability at which a tile casts a positive vote; --recon_average "
                        "vote only. 0.5 is the paper's 'positive (1) label'")
    p.add_argument("--blend_type", type=str, default=None, choices=["hann"],
                   help="Hann-window blending of overlapping predictions")
    p.add_argument("--window_gamma", type=float, default=1.0)

    p.add_argument("--attn_unet", action="store_true")
    p.add_argument("--add_attn", action="store_true")
    p.add_argument("--convlstm_unet", action="store_true")
    p.add_argument("--tattn_unet", action="store_true")

    p.add_argument("--job_name", type=str, default="job")
    p.add_argument("--output_dir", type=str, default=DEFAULT_PREDICTIONS_DIR)
    p.add_argument("--intf_dict_path", type=str, default=None)
    p.add_argument("--save_confidence", action="store_true")
    p.add_argument("--merge_polygs", action="store_true",
                   help="also write one combined shapefile across all interferograms")


def filter_by_min_positives(ids, args) -> list:
    """Drop interferograms with too few positive patches to be worth scoring.

    Thin scenes make the object-level mean per-scene precision/recall noisy:
    one missed sinkhole on a scene holding four of them moves that scene's
    recall by 0.25, and every scene carries equal weight in the mean. The
    threshold is applied to every ``--intf_source`` -- an explicit
    ``--intf_list`` included -- so one number describes the whole run, and the
    dropped ids are logged rather than silently vanishing.
    """
    if args.min_positives <= 0:
        return list(ids)
    coord = load_coord_dict(args.intf_dict_path)
    kept, dropped = [], []
    for intf_id in ids:
        n = coord.get(intf_id, {}).get("nonz_num", "none")
        keep = isinstance(n, int) and n > args.min_positives
        (kept if keep else dropped).append((intf_id, n))
    if dropped:
        logging.info(
            f"--min_positives {args.min_positives}: dropped {len(dropped)} of {len(ids)} "
            f"interferograms with too few positive patches: "
            + ", ".join(f"{i}({n})" for i, n in sorted(dropped))
        )
    if not kept:
        sys.exit(
            f"--min_positives {args.min_positives} removed every interferogram in the "
            f"list ({len(ids)} of {len(ids)}). Lower it, or pass --min_positives 0."
        )
    logging.info(f"{len(kept)} interferograms with more than {args.min_positives} "
                 f"positive patches")
    return [i for i, _ in kept]


def resolve_intf_list(args, data_dir) -> list:
    """The interferograms to score, from --intf_source, minus the thin ones."""
    return filter_by_min_positives(_intf_list_from_source(args, data_dir), args)


def _intf_list_from_source(args, data_dir) -> list:
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

    from ..models.factory import (
        PER_TIMESTEP_ARCHITECTURES,
        architecture_from_flags,
        build_from_checkpoint,
    )

    state_dict = torch.load(args.model, map_location=device)
    loaded = build_from_checkpoint(
        state_dict,
        arch=architecture_from_flags(
            attn_unet=args.attn_unet, add_attn=args.add_attn,
            convlstm_unet=args.convlstm_unet, tattn_unet=args.tattn_unet,
        ),
        n_classes=1,
        bilinear=False,
        treat_nodata_regions=args.treat_nodata_regions,
    )
    logging.info(f"architecture {loaded.architecture} ({loaded.n_channels} in-channels)")
    # Resolve geometry before discovering the input tree. An explicit margin
    # remains a compatibility check; omission now uses the saved input footprint.
    input_size = loaded.input_size_for(tuple(args.patch_size))
    margin = context_margin(tuple(args.patch_size), input_size)
    requested = getattr(args, "context_margin", None)
    if requested is not None and tuple(requested) != margin:
        raise SystemExit(f"geometry mismatch: checkpoint margin {margin}, "
                         f"--context_margin {tuple(requested)}")
    args.context_margin = margin
    args._context_size = input_size
    logging.info(f"input geometry {input_size}, target grid {tuple(args.patch_size)}")
    num_c = (args.k_prevs + 1) * (2 if args.treat_nodata_regions else 1)
    # Skipped for the sequence models: their n_channels counts one timestep, so
    # comparing it against the flat T*C the loader produces always disagrees.
    if (loaded.architecture not in PER_TIMESTEP_ARCHITECTURES
            and loaded.n_channels not in (None, num_c)):
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


def resolve_aoi_window(args):
    """The lat/lon box to restrict prediction to, or None.

    Prefers an explicit ``--aoi_window``; otherwise reads the partition file's
    window for ``--aoi_split``. Both being set is an error rather than a
    precedence rule — silently ignoring one of two conflicting windows is how a
    model ends up scored on ground it trained on.
    """
    from ..dataprep.partition import load_partition_window

    explicit = tuple(args.aoi_window) if getattr(args, "aoi_window", None) else None
    from_file = None
    if getattr(args, "aoi_from_partition", None):
        from_file = load_partition_window(args.aoi_from_partition, args.aoi_split)
    if explicit is not None and from_file is not None and explicit != from_file:
        raise SystemExit(
            f"--aoi_window {explicit} conflicts with {args.aoi_from_partition} "
            f"[{args.aoi_split}] = {from_file}. Pass one, not both."
        )
    return explicit if explicit is not None else from_file


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

    device = get_device()
    logging.info(f"loading model on {device}")
    net = load_model(args, device)

    cli_margin = tuple(getattr(args, "context_margin", (0, 0)) or (0, 0))
    data_dir = os.path.join(args.input_patch_dir,
                            patch_dir_name("data", patch_h, patch_w, args.data_stride,
                                           args.days_diff, cli_margin))
    mask_dir = os.path.join(args.input_patch_dir,
                            patch_dir_name("mask", patch_h, patch_w, args.data_stride, args.days_diff))

    intf_list = resolve_intf_list(args, data_dir)
    if args.add_lidar_mask:
        # Which scenes are gated on a survey that is not theirs. Reported once
        # here rather than per tile, because it changes what the object-level
        # numbers mean for those scenes: the gate is the LAST survey, not one
        # contemporaneous with the interferogram.
        unmapped = [i for i in intf_list if lidar_source_for(i) == NO_LIDAR_MASK]
        if unmapped:
            logging.warning(
                f"{len(unmapped)} of {len(intf_list)} interferograms have no LiDAR "
                f"mapping of their own (assets/lidar_intf_mask.txt ends at 20240605) "
                f"and are gated on {LIDAR_FALLBACK_SOURCE}: {', '.join(sorted(unmapped))}"
            )
    aoi = resolve_aoi_window(args)
    if aoi is not None:
        logging.info(f"AOI window lat {aoi[0]}-{aoi[1]}, lon {aoi[2]}-{aoi[3]}: "
                     f"tiles outside it are not predicted and eval-outputs must crop to it")

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
        # A large-context tree stores bigger cells on the SAME grid, so (ny, nx)
        # and the stamping below are untouched -- only how much of each cell is
        # kept changes. reconstruct_scene stamps a prediction of patch_size at
        # the cell's own location regardless, because the model crops its logits.
        ctx_h, ctx_w = getattr(args, "_context_size", None) or (patch_h, patch_w)
        if any(p.shape[-2:] != (ctx_h, ctx_w) for p in arrays_newest_first):
            raise ValueError(f"patch file input geometry must match checkpoint {(ctx_h, ctx_w)}")
        arrays_newest_first = [
            normalise_phase(p[:ny, :nx, :ctx_h, :ctx_w].astype(np.float32, copy=False),
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

        # -- positives-only gate ---------------------------------------------------------
        # Derived from gt_grid itself rather than from nonz_indices.json: it is
        # the same set by construction (both are "this tile's mask has a
        # positive pixel", on the same grid), it needs no extra file, and it
        # follows --unioned_mask for free — which matters, because a ground
        # truth unioned over the chain can be positive in a tile the current
        # frame's nonz list does not hold, and that object would then be
        # scored as a miss.
        positive_tiles = None
        if args.positives_only:
            positive_tiles = {(int(i), int(j))
                              for i, j in zip(*np.where((gt_grid > 0).any(axis=(-2, -1))))}
            logging.info(f"{intf}: positives-only — {len(positive_tiles)} of {ny * nx} tiles "
                         f"({100 * len(positive_tiles) / (ny * nx):.1f}%) hold ground truth; "
                         f"these numbers are not comparable with full-scene ones")
            if not positive_tiles:
                logging.warning(f"{intf}: no positive tiles, nothing to predict — skipped")
                continue

        # The AOI in this scene's grid coordinates. Derived per frame from the
        # aligned origin, never from the scene's raw north -- see geo.grid_window.
        tile_window = None
        if aoi is not None:
            r0, r1, c0, c1 = grid_window(
                meta.frame, *aoi,
                patch_size=(patch_h, patch_w),
                stride=(patch_h // args.data_stride, patch_w // args.data_stride),
            )
            tile_window = (r0, min(r1, ny), c0, min(c1, nx))
            if tile_window[0] >= tile_window[1] or tile_window[2] >= tile_window[3]:
                logging.warning(f"{intf}: no tile lies inside the AOI window — skipped")
                continue
            logging.info(f"{intf}: AOI tiles rows {tile_window[0]}:{tile_window[1]}, "
                         f"cols {tile_window[2]}:{tile_window[3]} of {ny}x{nx}")

        logging.info(f"{intf}: grid {ny}x{nx}, T={len(stack)}")
        result = reconstruct_scene(
            stack, net, (patch_h, patch_w), args.data_stride, args.recon_th,
            device=device, gt_grid=gt_grid, lidar_gates=lidar_gates,
            positive_tiles=positive_tiles, tile_window=tile_window,
            treat_nodata_regions=args.treat_nodata_regions,
            blend=args.blend_type, window_gamma=args.window_gamma,
            average=args.recon_average, vote_threshold=args.vote_threshold,
            log_progress=True,
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
        if args.save_confidence or args.intf_source != "all":
            np.save(prefix + "_pred", result.confidence)
        if args.intf_source != "all":
            np.save(prefix + "_image", image_to_save)
            np.save(prefix + "_pred_th", result.thresholded)
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
