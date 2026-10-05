"""``sinkholes relabel export`` -- Output A, the corrected ground truth.

Only live polygons (``edit_status != deleted``) are exported, in the source
shapefile's schema, so the file drops into ``prepare-patches
--gt_polygon_file_path`` and into ``relabel eval`` unchanged. The dates are
re-derived from ``intf_id`` (the single source of truth for scene membership)
and the export is read back and matched the way prepare-patches matches, per
scene, before it is published.

Outputs (``exports/``):

``gt_test_corrected_<tag>.gpkg``       layer ``gt`` -- the evaluation GT (scope scenes only)
``shp/sub_test_corrected_<tag>.shp``   the same as a shapefile
``full/sub_20260701_testcorrected_<tag>.shp``  (``--full_merge``) every original
                                       polygon outside the workspace scenes + the
                                       corrected ones: a drop-in for regenerating patches
``history/``                           timestamped copies of every export
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import (
    write_gt,
    ORIGINAL_GT_SHP,
    SOURCE_FIELDS,
    date_strings,
    now_iso,
    sha256_of,
    write_json,
    ws_path,
)


def add_arguments(p):
    p.add_argument("--workspace", required=True)
    p.add_argument("--tag", default="v2", help="version tag in the output names")
    p.add_argument("--allow_errors", action="store_true",
                   help="export even if validation reports errors (not for evaluation use)")
    p.add_argument("--full_merge", action="store_true",
                   help="also write the full-archive shapefile with the scope scenes replaced")
    p.add_argument("--data_dir", default=None, help="needed for --full_merge")


def build_active(workspace):
    import geopandas as gpd

    from . import diff as d

    changes, summary, detected = d.run(workspace, take_snapshot=True)
    _, work, scenes = d.load_layers(workspace)
    live = work[work["edit_status"].fillna("") != "deleted"].copy()
    change_of = {c["feat_uid"]: c["change_id"] for c in changes if c["feat_uid"] and c["change_type"] != "DEL"}
    fd = dict(zip(scenes["intf_id"], scenes["flight_dir"]))
    out = pd.DataFrame(index=live.index)
    for col in SOURCE_FIELDS:
        out[col] = live[col] if col in live.columns else None
    sd_ed = live["intf_id"].map(date_strings)
    # datetime64 -> a Date field in the shapefile, like the source (python date
    # objects would be written as TEXT -- still matchable, but not the schema).
    out["start_date"] = pd.to_datetime(sd_ed.str[0])
    out["end_date"] = pd.to_datetime(sd_ed.str[1])
    new = live["orig_uid"].isna()
    out.loc[new, "platform"] = out.loc[new, "platform"].fillna("TSX")
    out.loc[new, "flight_dir"] = live.loc[new, "intf_id"].map(fd)
    out["id"] = pd.to_numeric(out["id"], errors="coerce").fillna(0).astype("int64")
    out["decor"] = pd.to_numeric(out["decor"], errors="coerce")
    edited = live["feat_uid"].map(lambda u: detected.get(u, "added")).isin(["added", "modified"])
    out.loc[edited, "reporter"] = live.loc[edited, "edited_by"].fillna(out.loc[edited, "reporter"])
    ts = pd.to_datetime(live["edit_timestamp"], errors="coerce").dt.normalize()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.normalize()
    out.loc[edited, "timestamp"] = ts[edited].fillna(pd.Timestamp(datetime.now().date()))
    out["intf_id"] = live["intf_id"]
    out["gt_uid"] = live["feat_uid"]
    out["orig_uid"] = live["orig_uid"]
    out["edit_stat"] = live["feat_uid"].map(lambda u: detected.get(u, "added"))
    out["change_id"] = live["feat_uid"].map(change_of)
    gdf = gpd.GeoDataFrame(out, geometry=live.geometry, crs="EPSG:4326")
    return gdf.reset_index(drop=True), summary, changes


def check_like_prepare_patches(path, expected):
    """Re-read ``path`` and match each scene by exact dates, as prepare-patches does."""
    import geopandas as gpd

    g = gpd.read_file(path)
    bad = {}
    for intf, n in expected.items():
        sd, ed = date_strings(intf)
        got = int(((g["start_date"] == sd) & (g["end_date"] == ed)).sum())
        if got != n:
            bad[intf] = (n, got)
    if bad:
        raise SystemExit(f"{path}: date matching does not reproduce the export: {bad}")


def main(args):
    import geopandas as gpd

    from . import validate as v

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ws = Path(args.workspace)
    issues = v.run(ws)
    errors = [i for i in issues if i["level"] == "ERROR"]
    if errors and not args.allow_errors:
        for i in errors[:20]:
            logging.error(f"{i['check']}: {i['intf_id']} {i['feat_uid']} {i['message']}")
        raise SystemExit(f"{len(errors)} validation error(s): fix them, or --allow_errors "
                         "(the result is then not fit for evaluation)")

    gdf, summary, changes = build_active(ws)
    expected = gdf.groupby("intf_id").size().to_dict()
    out_dir = ws_path(ws, "exports")
    hist = out_dir / "history"
    for d in (out_dir, out_dir / "shp", hist):
        d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpkg = out_dir / f"gt_test_corrected_{args.tag}.gpkg"
    tmp = out_dir / f".tmp_{stamp}.gpkg"
    write_gt(gdf, tmp, layer="gt")
    check_like_prepare_patches(tmp, expected)
    os.replace(tmp, gpkg)

    shp = out_dir / "shp" / f"sub_test_corrected_{args.tag}.shp"
    write_gt(gdf.drop(columns=["change_id"]), shp)
    check_like_prepare_patches(shp, expected)

    full = None
    if args.full_merge:
        if not args.data_dir:
            raise SystemExit("--full_merge needs --data_dir")
        src = gpd.read_file(Path(args.data_dir) / ORIGINAL_GT_SHP)
        sd = src["start_date"].dt.strftime("%Y%m%d")
        ed = src["end_date"].dt.strftime("%Y%m%d")
        keep = src[~(sd + "_" + ed).isin(set(gpd.read_file(ws_path(ws, "original_gpkg"),
                                                          layer="scenes")["intf_id"]))].copy()
        merged = gpd.GeoDataFrame(pd.concat([keep, gdf[list(SOURCE_FIELDS) + ["geometry"]]],
                                            ignore_index=True), crs="EPSG:4326")
        (out_dir / "full").mkdir(exist_ok=True)
        full = out_dir / "full" / f"sub_20260701_testcorrected_{args.tag}.shp"
        write_gt(merged, full)
        check_like_prepare_patches(full, expected)

    shutil.copy2(gpkg, hist / f"gt_test_corrected_{args.tag}_{stamp}.gpkg")
    info = {
        "exported": now_iso(), "tag": args.tag, "polygons": int(len(gdf)),
        "per_scene": expected, "changes": len(changes),
        "validation": {lvl: sum(i["level"] == lvl for i in issues) for lvl in ("ERROR", "WARNING", "INFO")},
        "allow_errors": bool(args.allow_errors),
        "gpkg": str(gpkg), "gpkg_sha256": sha256_of(gpkg),
        "shp": str(shp), "full_merge": str(full) if full else None,
        "working_gpkg_sha256": sha256_of(ws_path(ws, "working_gpkg")),
    }
    write_json(out_dir / f"gt_test_corrected_{args.tag}.json", info)
    write_json(hist / f"gt_test_corrected_{args.tag}_{stamp}.json", info)
    logging.info(json.dumps({k: info[k] for k in ("polygons", "changes", "validation")}))
    logging.info(f"-> {gpkg}\n-> {shp}" + (f"\n-> {full}" if full else ""))
