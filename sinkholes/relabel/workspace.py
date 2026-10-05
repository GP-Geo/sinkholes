"""``sinkholes relabel init`` -- build the relabelling workspace.

Additive by construction: every output lands under ``--workspace``, and a file
that already exists is skipped (``gt_test_original.gpkg`` above all -- it is
written once, made read-only, and its SHA-256 recorded in provenance.json).
"""

import csv
import glob
import json
import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np

from ..meta import find_11day_sequences, load_coord_dict, parse_ers_header
from ..paths import REPO_ROOT
from .common import (
    GPKG_OPTIONS,
    write_gt,
    EDIT_STATUSES,
    LAYOUT,
    METRIC_CRS,
    OFFICIAL_PARTITION,
    ORIGINAL_GT_SHP,
    SOURCE_FIELDS,
    content_fingerprint,
    THRESHOLD_FREE_PARTITIONS,
    USER_FLAGGED,
    date_strings,
    find_scene_file,
    load_partition,
    now_iso,
    read_original_gt,
    read_raw_window,
    scene_geometry,
    scene_window_on_raw,
    sha256_of,
    write_json,
    ws_path,
)

#: Patch trees whose grid files a scene and its k=5 chain must have on disk.
CHECK_TREES = (
    ("data_patches_H200_W100_strpp2_11days_Aligned", 2),
    ("data_patches_H200_W100_strpp4_11days_Aligned", 4),
    ("data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned", 4),
    ("mask_patches_H200_W100_strpp4_11days_Aligned", 4),
)

#: Evaluations used for the review-priority diagnostics: the testeval twin of
#: the official partition (its 'val' list IS the official test list), scored
#: with the paper's vote protocol.
OFFICIAL_EVAL_PARTITIONS = ("partition_temporal_k5_testeval_clean_th350x200.json",)
#: Candidates are not in any generation-4 list; the generation-3 evaluations
#: (k10 testeval list, which holds most of them) are the only scores they have.
CANDIDATE_EVAL_PARTITIONS = ("partition_temporal_k10_testeval_clean.json",
                             "partition_temporal_k5_testeval_clean.json")


def add_arguments(p):
    p.add_argument("--workspace", required=True)
    p.add_argument("--data_dir", required=True,
                   help="deadsea_sinkholes_data root (raw scenes, sub_20260701.shp, patches/)")
    p.add_argument("--predictions_root", default=str(REPO_ROOT / "outputs" / "predictions"),
                   help="read-only: existing evaluations, for the priority diagnostics")
    p.add_argument("--raster_margin_deg", type=float, default=0.005,
                   help="context kept around the AOI in the labelling rasters (~550 m)")
    p.add_argument("--skip_rasters", action="store_true")


# --------------------------------------------------------------------------- scope


def resolve_scope(coord):
    """(official, candidates, k5 chains) -- verified against the code's own rules."""
    official = sorted(load_partition(OFFICIAL_PARTITION)["test"])
    pool = set()
    for rel in THRESHOLD_FREE_PARTITIONS:
        pool |= set(load_partition(rel)["test"])
    candidates = sorted(pool - set(official))
    chains, _ = find_11day_sequences(coord, k_prev=5, require_current_nonz_gt0=False)
    return official, candidates, chains


def scene_rows(data_dir, coord, official, candidates, chains, gt):
    from ..dataprep.partition import load_partition_window

    aoi = load_partition_window(str(REPO_ROOT / OFFICIAL_PARTITION), "test")
    k10 = set(load_partition("assets/partition_temporal_k10_clean_th350x200.json")["test"])
    rows = []
    for intf in official + candidates:
        m = coord[intf]
        geo = scene_geometry(intf)
        raw = find_scene_file(data_dir, intf)
        header_ok = False
        if raw and os.path.exists(raw + ".ers"):
            h = parse_ers_header(raw + ".ers")
            header_ok = all(
                (h[k] == m[k]) if k == "byte_order" else abs(float(h[k]) - float(m[k])) < 1e-9
                for k in h)
        prevs = chains.get(intf, {}).get("prevs")
        missing = []
        for pid in (prevs or []) + [intf]:
            for tree, s in CHECK_TREES:
                kind = tree.split("_")[0]
                f = Path(data_dir) / "patches" / tree / f"{kind}_patches_{pid}_H200_W100_strpp{s}.npy"
                if not f.exists():
                    missing.append(f"{tree}:{pid}")
        sub = gt[gt["intf_id"] == intf]
        lat_min, lat_max, lon_min, lon_max = aoi
        c = sub.geometry.representative_point()
        in_aoi = int(((c.y >= lat_min) & (c.y <= lat_max) & (c.x >= lon_min) & (c.x <= lon_max)).sum())
        nonz = m.get("nonz_num")
        th = 350 if float(m["north"]) > 31.5 else 200
        sd, ed = date_strings(intf)
        sx, sy = geo.subpixel_offset
        rows.append({
            "intf_id": intf,
            "role": "official_test" if intf in official else "candidate_excluded_by_threshold",
            "start_date": sd, "end_date": ed, "frame": m["frame"],
            "flight_dir": (sub["flight_dir"].mode().iat[0] if len(sub) else ""),
            "in_k10_test": intf in k10,
            "gt_count": int(len(sub)), "gt_in_aoi": in_aoi,
            "nonz_num": nonz, "nonz_threshold": th,
            "user_flagged": intf in USER_FLAGGED,
            "raw_scene": os.path.basename(raw) if raw else "",
            "raw_header_matches_coord_dict": header_ok,
            "k5_prevs": ";".join(prevs) if prevs else "",
            "k5_chain_complete": bool(prevs) and not missing,
            "missing_grid_files": ";".join(missing),
            "row_off": geo.row_off, "col_off": geo.col_off,
            "canvas_x0": repr(geo.canvas_x0), "canvas_y0": repr(geo.canvas_y0),
            "nominal_minus_actual_px_x": round(sx, 4), "nominal_minus_actual_px_y": round(sy, 4),
            "raster": f"{LAYOUT['rasters']}/{intf}_int_aoi.tif",
        })
    return rows, aoi


# --------------------------------------------------------------------------- layers


def build_original_layers(gt, rows, aoi, margin):
    """(original_gt, scenes) GeoDataFrames."""
    import geopandas as gpd
    from shapely.geometry import box

    keep = ["orig_uid", "orig_fid", "intf_id", *SOURCE_FIELDS, "geometry"]
    orig = gt[keep].copy()
    orig["area_m2"] = orig.geometry.to_crs(METRIC_CRS).area.round(1)
    geoms = []
    for r in rows:
        geo = scene_geometry(r["intf_id"])
        r0, r1, c0, c1 = scene_window_on_raw(geo, aoi, margin)
        x0, y0 = geo.east + c0 * geo.dx, geo.north - r0 * geo.dy
        geoms.append(box(x0, geo.north - r1 * geo.dy, geo.east + c1 * geo.dx, y0))
    scenes = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    return orig, scenes


def build_working_layer(orig):
    work = orig.drop(columns=["area_m2"]).copy()
    work.insert(0, "feat_uid", work["orig_uid"])
    work["edit_status"] = EDIT_STATUSES[0]
    for col in ("edit_reason", "edit_notes", "edited_by", "edit_timestamp"):
        work[col] = None
    for col in ("edit_reason", "edit_notes", "edited_by", "edit_timestamp"):
        work[col] = work[col].astype("object")
    return work


def build_aux_layers(rows, aoi, data_dir):
    """AOI box, LiDAR gate footprint, and the area the model can predict at all.

    'predictable_area' is the union of stride-4 tiles that lie wholly inside the
    AOI grid window and wholly inside the LiDAR2022 gate (every 2025-26 scene and
    its whole k=5 chain gates on LiDAR2022 -- meta.LIDAR_FALLBACK_SOURCE). A GT
    polygon outside it can never be detected: it scores as a miss for every
    model regardless of labelling quality. Computed per frame on the largest
    canvas of that frame, using the nominal origin as eval-scenes does.
    """
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union

    from ..geo import FRAME_ORIGINS, grid_window
    from ..inference.reconstruct import canvas_shape, rasterise_lidar_gates
    from ..meta import LIDAR_FALLBACK_SOURCE

    lat_min, lat_max, lon_min, lon_max = aoi
    aoi_gdf = gpd.GeoDataFrame({"name": ["test AOI window"]},
                               geometry=[box(lon_min, lat_min, lon_max, lat_max)], crs="EPSG:4326")
    lidar = gpd.read_file(str(REPO_ROOT / "assets" / "lidar_mask_polygs.shp"))
    lidar = lidar[lidar["source"].astype(str).str.strip().str.lower()
                  == LIDAR_FALLBACK_SOURCE.lower()].copy()
    # The committed shapefile has no .prj; its coordinates are lon/lat degrees
    # (rasterise_lidar_gates burns them with a degree transform), so say so.
    lidar = lidar.set_crs("EPSG:4326") if lidar.crs is None else lidar.to_crs("EPSG:4326")

    pa_rows, pa_geoms = [], []
    ph, pw, stride = 200, 100, 4
    sy, sx = ph // stride, pw // stride
    for frame in ("North", "South"):
        frs = [r for r in rows if r["frame"] == frame]
        if not frs:
            continue
        ny = max((scene_geometry(r["intf_id"]).nlines - r["row_off"] - ph) // sy + 1 for r in frs)
        nx = (4500 - pw) // sx + 1
        H, W = canvas_shape(ny, nx, (ph, pw), stride)
        x0, y0 = FRAME_ORIGINS[frame]
        dx = dy = 2.777e-05
        gate = rasterise_lidar_gates([LIDAR_FALLBACK_SOURCE], (x0, y0, dx, dy), (H, W))[0]
        r0, r1, c0, c1 = grid_window(frame, *aoi, patch_size=(ph, pw), stride=(sy, sx))
        r1, c1 = min(r1, ny), min(c1, nx)
        tiles = []
        for i in range(r0, r1):
            for j in range(c0, c1):
                if gate[i * sy:i * sy + ph, j * sx:j * sx + pw].all():
                    tiles.append(box(x0 + j * sx * dx, y0 - (i * sy + ph) * dy,
                                     x0 + (j * sx + pw) * dx, y0 - i * sy * dy))
        pa_rows.append({"frame": frame, "n_tiles": len(tiles),
                        "note": "stride-4 tiles inside AOI and LiDAR2022; GT outside is never detectable"})
        pa_geoms.append(unary_union(tiles) if tiles else None)
    pa = gpd.GeoDataFrame(pa_rows, geometry=pa_geoms, crs="EPSG:4326")
    return aoi_gdf, lidar, pa


def write_rasters(workspace, data_dir, rows, aoi, margin):
    """One GeoTIFF per scene: the raw tgeo_int pixels of the canvas cut to the AOI.

    An integer window of the raw raster written with the raw transform shifted
    by whole pixels -- no resampling, no reprojection -- and read back to prove
    it. Overviews use nearest: averaging wrapped phase is meaningless. A VRT
    over the full raw scene sits beside it for anything outside the window.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    out_dir = ws_path(workspace, "rasters")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for r in rows:
        intf = r["intf_id"]
        tif = Path(workspace) / r["raster"]
        raw = os.path.join(data_dir, r["raw_scene"])
        geo = scene_geometry(intf)
        win = scene_window_on_raw(geo, aoi, margin)
        vrt = out_dir / f"{intf}_int_full.vrt"
        if not vrt.exists():
            rel = os.path.relpath(raw, out_dir)
            subprocess.run(["gdal_translate", "-q", "-of", "VRT", raw + ".ers", str(vrt)], check=True)
            text = vrt.read_text()
            # gdal_translate writes an absolute source path; make it relative so
            # the VRT resolves from /home/labs/... on WEXAC and /Volumes/... alike.
            for absname in (raw + ".ers", os.path.abspath(raw + ".ers")):
                text = text.replace(f'relativeToVRT="0">{absname}<',
                                    f'relativeToVRT="1">{rel}.ers<')
            vrt.write_text(text)
        if tif.exists():
            report.append({"intf_id": intf, "raster": str(tif), "status": "exists, skipped"})
            continue
        data = read_raw_window(raw, geo, win, coord_byte_order(intf))
        r0, r1, c0, c1 = win
        with rasterio.open(raw + ".ers") as src:
            transform = src.window_transform(Window(c0, r0, c1 - c0, r1 - r0))
            assert np.array_equal(src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)), data), \
                f"{intf}: GDAL and the pipeline's np.fromfile read disagree"
        tmp = tif.with_suffix(".tmp.tif")
        profile = dict(driver="GTiff", width=c1 - c0, height=r1 - r0, count=1, dtype="float32",
                       crs="EPSG:4326", transform=transform, nodata=0.0, tiled=True,
                       blockxsize=512, blockysize=512, compress="deflate", predictor=3,
                       BIGTIFF="IF_SAFER")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(intf_id=intf, source=r["raw_scene"], window_rows=f"{r0}:{r1}",
                            window_cols=f"{c0}:{c1}",
                            note="raw tgeo_int pixels (wrapped phase, rad) as read by "
                                 "prepare-patches; integer window, no resampling")
            dst.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")
        with rasterio.open(tmp) as chk:
            assert np.array_equal(chk.read(1), data), f"{intf}: GeoTIFF round trip changed pixels"
        os.replace(tmp, tif)
        report.append({"intf_id": intf, "raster": str(tif), "window": list(win),
                       "status": "written, round-trip identical"})
        logging.info(f"{intf}: {tif.name} {data.shape}")
    return report


def coord_byte_order(intf):
    from ..meta import intf_meta

    return intf_meta(intf).byte_order


# --------------------------------------------------------------------------- priority


def priority_rows(predictions_root, official, candidates, gt_counts):
    """Per-scene diagnostics from the saved evaluations (nothing is re-run)."""
    from .common import read_eval_args

    def collect(partition_names, scenes):
        per = {s: [] for s in scenes}
        for d in sorted(glob.glob(os.path.join(predictions_root, "*", "*", "scenes_*"))):
            try:
                a = read_eval_args(d)
            except (FileNotFoundError, ValueError, SyntaxError):
                continue
            if Path(str(a.get("valset_from_partition", ""))).name not in partition_names:
                continue
            if a.get("recon_average") != "vote" or a.get("positives_only"):
                continue
            js = sorted(glob.glob(os.path.join(d, "olm_results_*.json")))
            if not js:
                continue
            with open(js[-1]) as fh:
                J = json.load(fh)
            model = Path(d).parts[-3]
            # rank inside THIS evaluation's own scene list
            rec_all = sorted((v["0.25"]["recall"], i) for i, v in J["per_intf"].items()
                             if "0.25" in v and v["0.25"]["recall"] == v["0.25"]["recall"])
            bottom5 = {i for _, i in rec_all[:5]}
            for s in scenes:
                v = J["per_intf"].get(s, {}).get("0.25")
                if not v:
                    continue
                shp = os.path.join(d, "polygs", f"{s}_predicted_polygs.shp")
                n_pred = None
                if os.path.exists(shp):
                    import pyogrio

                    n_pred = pyogrio.read_info(shp)["features"]
                per[s].append((model, v["recall"], v["precision"], n_pred, s in bottom5))
        return per

    out = []
    for scenes, names, role in ((official, OFFICIAL_EVAL_PARTITIONS, "official_test"),
                                (candidates, CANDIDATE_EVAL_PARTITIONS,
                                 "candidate_excluded_by_threshold")):
        per = collect(names, scenes)
        for s in scenes:
            v = per[s]
            rec = np.array([x[1] for x in v], float)
            prec = np.array([x[2] for x in v], float)
            npred = [x[3] for x in v if x[3] is not None]
            nb = sum(x[4] for x in v)
            out.append({
                "intf_id": s, "role": role, "user_flagged": s in USER_FLAGGED,
                "gt_count": gt_counts.get(s, 0),
                "n_models": len(v),
                "pred_polygons_mean": round(float(np.mean(npred)), 1) if npred else None,
                "recall_mean": round(float(rec.mean()), 3) if len(v) else None,
                "recall_min": round(float(rec.min()), 3) if len(v) else None,
                "recall_max": round(float(rec.max()), 3) if len(v) else None,
                "precision_mean": round(float(prec.mean()), 3) if len(v) else None,
                "precision_min": round(float(prec.min()), 3) if len(v) else None,
                "models_with_scene_in_bottom5_recall": nb,
                "consistent_outlier": bool(len(v)) and nb >= 0.75 * len(v),
                "metric_source": "RTh 0.25, ith0.7/b5, vote-protocol evals on " + "|".join(names),
            })
    # tiers: 1 flagged or consistent outlier; 2 recall below the official mean; 3 rest
    off = [r for r in out if r["role"] == "official_test" and r["recall_mean"] is not None]
    mean_rec = float(np.mean([r["recall_mean"] for r in off])) if off else 0
    for r in out:
        if r["user_flagged"] or r["consistent_outlier"]:
            r["priority"] = 1
        elif r["recall_mean"] is not None and r["recall_mean"] < mean_rec:
            r["priority"] = 2
        else:
            r["priority"] = 3
        if r["role"] != "official_test":
            r["priority"] = f"{r['priority']} (candidate)"
    out.sort(key=lambda r: (r["role"] != "official_test", str(r["priority"]),
                            r["recall_mean"] if r["recall_mean"] is not None else 9))
    return out


# --------------------------------------------------------------------------- main


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_read_only(path):
    mode = os.stat(path).st_mode
    os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def git_state():
    try:
        rev = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": rev, "dirty_worktree": dirty}
    except Exception as e:  # noqa: BLE001 -- provenance is best effort
        return {"error": str(e)}


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ws = Path(args.workspace)
    data_dir = args.data_dir
    for key in ("history", "changes", "exports", "review", "eval", "qgis"):
        ws_path(ws, key).mkdir(parents=True, exist_ok=True)
    ws_path(ws, "original_gpkg").parent.mkdir(parents=True, exist_ok=True)

    coord = load_coord_dict()
    official, candidates, chains = resolve_scope(coord)
    logging.info(f"official test scenes: {len(official)}; threshold candidates: {len(candidates)}")
    gt = read_original_gt(data_dir, official + candidates)
    rows, aoi = scene_rows(data_dir, coord, official, candidates, chains, gt)
    bad = [r["intf_id"] for r in rows if r["role"] == "official_test"
           and not (r["raw_header_matches_coord_dict"] and r["k5_chain_complete"])]
    if bad:
        raise SystemExit(f"official scenes failing verification: {bad}")

    orig_path = ws_path(ws, "original_gpkg")
    orig, scenes = build_original_layers(gt, rows, aoi, args.raster_margin_deg)
    if orig_path.exists():
        logging.info(f"{orig_path} exists -- never rewritten")
    else:
        # write_gt: real DATE fields (QGIS shows a DATETIME default in UTC,
        # which moved a new polygon's date to the previous day in testing)
        write_gt(orig, orig_path, layer="original_gt")
        scenes.to_file(orig_path, layer="scenes", driver="GPKG",
            dataset_options=GPKG_OPTIONS)
        make_read_only(orig_path)
        logging.info(f"{orig_path}: {len(orig)} polygons, {len(scenes)} scenes (read-only)")

    work_path = ws_path(ws, "working_gpkg")
    if work_path.exists():
        logging.info(f"{work_path} exists -- left alone")
    else:
        write_gt(build_working_layer(orig), work_path, layer="working_gt")
        logging.info(f"{work_path}: working_gt initialised from the original")

    aux_path = ws_path(ws, "aux_gpkg")
    if not aux_path.exists():
        aoi_gdf, lidar, pa = build_aux_layers(rows, aoi, data_dir)
        aoi_gdf.to_file(aux_path, layer="aoi_window", driver="GPKG",
            dataset_options=GPKG_OPTIONS)
        lidar.to_file(aux_path, layer="lidar2022_gate", driver="GPKG",
            dataset_options=GPKG_OPTIONS)
        pa.to_file(aux_path, layer="predictable_area", driver="GPKG",
            dataset_options=GPKG_OPTIONS)
        scenes.to_file(aux_path, layer="scene_rasters_extent", driver="GPKG",
            dataset_options=GPKG_OPTIONS)

    write_csv(ws_path(ws, "scenes_csv"), rows)

    manifest = ws_path(ws, "manifest")
    if not manifest.exists():
        write_csv(manifest, [{
            "intf_id": r["intf_id"], "role": r["role"], "status": "not_reviewed",
            "original_count": r["gt_count"], "corrected_count": "", "added": "", "deleted": "",
            "modified": "", "reviewed_by": "", "review_date": "", "external_reviewer": "",
            "external_review_date": "", "notes": "",
        } for r in rows])

    prio = priority_rows(args.predictions_root, official, candidates,
                         {r["intf_id"]: r["gt_count"] for r in rows})
    write_csv(ws_path(ws, "priority_csv"), prio)

    raster_report = [] if args.skip_rasters else write_rasters(
        ws, data_dir, rows, aoi, args.raster_margin_deg)

    src = Path(data_dir) / ORIGINAL_GT_SHP
    prov_path = ws_path(ws, "provenance")
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov.setdefault("created", now_iso())
    prov.update({
        "updated": now_iso(),
        "original_gt": {ext: {"path": str(src.with_suffix(ext)), "sha256": sha256_of(src.with_suffix(ext))}
                        for ext in (".shp", ".shx", ".dbf", ".prj")},
        "original_gpkg_sha256_at_init": sha256_of(orig_path),
        "original_content_sha256": prov.get("original_content_sha256") or content_fingerprint(
            __import__("geopandas").read_file(orig_path, layer="original_gt")),
        "official_partition": OFFICIAL_PARTITION,
        "official_partition_sha256": sha256_of(REPO_ROOT / OFFICIAL_PARTITION),
        "official_test": official,
        "candidates_excluded_by_threshold": candidates,
        "aoi_window": list(aoi),
        "scene_variant": "tgeo_int (data root), as scripts/data/link_scenes.py --variant int",
        "crs": "EPSG:4326",
        "repo": git_state(),
        "rasters": raster_report or prov.get("rasters", []),
    })
    write_json(prov_path, prov)

    goto = REPO_ROOT / "scripts" / "relabel" / "qgis_goto_scene.py"
    if goto.exists():
        shutil.copy(goto, ws_path(ws, "qgis") / "goto_scene.py")
    readme_src = REPO_ROOT / "docs" / "RELABELING.md"
    if readme_src.exists() and not (ws / "README.md").exists():
        shutil.copy(readme_src, ws / "README.md")
    logging.info(f"workspace ready: {ws}")
