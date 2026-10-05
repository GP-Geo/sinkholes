"""``sinkholes relabel validate`` -- checks the working GT before it is exported.

ERROR blocks ``relabel export``; WARNING is reported and allowed; INFO is
context. Pre-existing problems of untouched original polygons are reported as
WARNING ("inherited"): fixing them is a relabelling decision, not a precondition.

The date checks matter more than they look: prepare-patches attaches a polygon
to an interferogram ONLY by exact start_date/end_date equality, so a polygon
with a typo in a date silently vanishes from the training/evaluation mask.
"""

import csv
import logging
from pathlib import Path

import numpy as np

from .common import EDIT_REASONS, EDIT_STATUSES, METRIC_CRS, date_strings, scene_geometry, ws_path

#: Below this many raster pixels a polygon is effectively invisible to the pipeline.
MIN_PIXELS = 4
#: Above this area (m^2) a single sinkhole/subsidence polygon is suspicious.
MAX_AREA_M2 = 500_000


def add_arguments(p):
    p.add_argument("--workspace", required=True)
    p.add_argument("--intf", default=None, help="validate one interferogram only")


def pixel_count(geom, geo) -> int:
    """Raw-grid pixels whose centre is inside ``geom`` -- what prepare-patches burns."""
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds

    t = geo.raw_transform()
    minx, miny, maxx, maxy = geom.bounds
    w = from_bounds(minx, miny, maxx, maxy, transform=t)
    c0, r0 = int(np.floor(w.col_off)) - 1, int(np.floor(w.row_off)) - 1
    c1, r1 = int(np.ceil(w.col_off + w.width)) + 1, int(np.ceil(w.row_off + w.height)) + 1
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, geo.ncells), min(r1, geo.nlines)
    if c0 >= c1 or r0 >= r1:
        return 0
    from rasterio.windows import transform as wtransform

    wt = wtransform(Window(c0, r0, c1 - c0, r1 - r0), t)
    m = geometry_mask([geom], transform=wt, invert=True, out_shape=(r1 - r0, c1 - c0))
    return int(m.sum())


def run(workspace, only=None):
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.validation import explain_validity

    from . import diff as d

    orig, work, scenes = d.load_layers(workspace)
    raw_work = gpd.read_file(ws_path(workspace, "working_gpkg"), layer="working_gt")
    aux = ws_path(workspace, "aux_gpkg")
    aoi = gpd.read_file(aux, layer="aoi_window").geometry.iloc[0]
    pa = gpd.read_file(aux, layer="predictable_area")
    pa_by_frame = dict(zip(pa["frame"], pa.geometry))
    issues = []

    def add(level, check, row=None, intf=None, msg=""):
        issues.append({
            "level": level, "check": check,
            "intf_id": intf if intf is not None else (getattr(row, "intf_id", "") if row is not None else ""),
            "feat_uid": getattr(row, "feat_uid", "") if row is not None else "",
            "orig_uid": getattr(row, "orig_uid", "") if row is not None else "",
            "message": msg,
        })

    if str(raw_work.crs).upper() not in ("EPSG:4326",) and raw_work.crs.to_epsg() != 4326:
        add("ERROR", "crs", msg=f"working_gt CRS is {raw_work.crs}, must be EPSG:4326")

    changes, detected = d.detect_changes(orig, work)
    touched = {c["feat_uid"] for c in changes if c["feat_uid"]}
    scene_ids = set(scenes["intf_id"])
    frame = dict(zip(scenes["intf_id"], scenes["frame"]))
    geos = {}

    # identity
    if raw_work["feat_uid"].isna().any():
        add("WARNING", "feat_uid_missing",
            msg=f"{int(raw_work['feat_uid'].isna().sum())} polygon(s) without feat_uid -- drawn "
                "outside the QGIS form defaults; identity falls back to a geometry hash")
    dup = work["feat_uid"][work["feat_uid"].duplicated(keep=False)]
    for uid in sorted(set(dup)):
        add("ERROR", "feat_uid_duplicate", intf="",
            msg=f"feat_uid {uid} on {int((work['feat_uid'] == uid).sum())} polygons (copy/paste "
                "keeps it): give the copies a new uuid() or clear orig_uid on the copy")

    for row in work.itertuples(index=False):
        intf = row.intf_id
        if only and intf != only:
            continue
        status = row.edit_status if isinstance(row.edit_status, str) else None
        live = status != "deleted"
        new_or_edited = row.feat_uid in touched
        lvl = "ERROR" if new_or_edited else "WARNING"
        if not isinstance(intf, str) or intf not in scene_ids:
            add("ERROR", "intf_id", row, msg=f"intf_id {intf!r} is not a workspace scene")
            continue
        if status is not None and status not in EDIT_STATUSES:
            add("ERROR", "edit_status", row, msg=f"edit_status {status!r} not in {EDIT_STATUSES}")
        reason = getattr(row, "edit_reason", None)
        if isinstance(reason, str) and reason and reason not in EDIT_REASONS:
            add("WARNING", "edit_reason", row, msg=f"edit_reason {reason!r} not in the vocabulary")
        g = row.geometry
        if g is None or g.is_empty:
            add("ERROR" if live else "WARNING", "empty_geometry", row, msg="null or empty geometry")
            continue
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            add("ERROR", "geometry_type", row, msg=f"{g.geom_type}, expected Polygon")
            continue
        if not g.is_valid:
            add(lvl if live else "INFO", "invalid_geometry", row,
                msg=("" if new_or_edited else "inherited from the original: ") + explain_validity(g))
        # dates: exact equality is what prepare-patches matches on
        sd, ed = date_strings(intf)
        for col, want in (("start_date", sd), ("end_date", ed)):
            v = getattr(row, col, None)
            if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ("NaT", ""):
                add("INFO", f"{col}_missing", row, msg=f"{col} empty -- export fills {want} from intf_id")
            elif str(v)[:10] != want:
                add("ERROR", f"{col}_mismatch", row, msg=f"{col}={str(v)[:10]} but intf_id says {want}")
        if not live:
            continue
        geo = geos.setdefault(intf, scene_geometry(intf))
        ext = box(geo.east, geo.north - geo.nlines * geo.dy, geo.east + geo.ncells * geo.dx, geo.north)
        if not g.intersects(ext):
            add(lvl, "outside_raster", row,
                msg=("" if new_or_edited else "inherited from the original: ")
                + "polygon is outside the interferogram raster (wrong scene or date?)")
            continue
        if not ext.contains(g):
            add("WARNING", "partly_outside_raster", row, msg="polygon crosses the raster edge")
        if not g.intersects(aoi):
            add("INFO", "outside_aoi", row, msg="outside the test AOI: never scored")
        elif frame[intf] in pa_by_frame and pa_by_frame[frame[intf]] is not None \
                and not g.intersects(pa_by_frame[frame[intf]]):
            add("INFO", "outside_predictable_area", row,
                msg="inside the AOI but outside the LiDAR-gated tiles: always a miss")
        npx = pixel_count(g, geo)
        if npx < MIN_PIXELS:
            add("WARNING" if new_or_edited else "INFO", "tiny", row,
                msg=f"burns {npx} pixel(s) -- invisible or nearly so in the mask")
        area = gpd.GeoSeries([g], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0].area
        if area > MAX_AREA_M2:
            add("WARNING", "large", row, msg=f"area {area:,.0f} m^2")

    # duplicates / overlaps within a scene (live polygons)
    live = work[work["edit_status"].fillna("") != "deleted"]
    for intf, grp in live.groupby("intf_id"):
        if only and intf != only:
            continue
        g = grp.reset_index(drop=True)
        sidx = g.sindex
        for i, gi in enumerate(g.geometry):
            if gi is None or gi.is_empty:
                continue
            for j in sidx.query(gi, predicate="intersects"):
                if j <= i:
                    continue
                gj = g.geometry.iloc[j]
                v = d.iou(gi, gj)
                fi, fj = g["feat_uid"].iloc[i], g["feat_uid"].iloc[j]
                edited = fi in touched or fj in touched
                if v > 0.999:
                    add("ERROR" if edited else "WARNING", "duplicate", intf=intf,
                        msg=f"{fi} and {fj} are the same polygon")
                elif v > 0.5:
                    add("WARNING", "near_duplicate", intf=intf,
                        msg=f"{fi} and {fj} overlap with IoU {v:.2f}")
                elif gi.intersection(gj).area > 0 and edited:
                    add("INFO", "overlap", intf=intf,
                        msg=f"{fi} and {fj} overlap (IoU {v:.2f}); rasterisation unions them")

    # change bookkeeping
    for c in changes:
        if only and c["intf_id"] != only:
            continue
        if c["needs_review"]:
            add("WARNING", "ambiguous_change", intf=c["intf_id"],
                msg=f"{c['change_type']} {c['orig_uid'] or c['feat_uid']}: {c['review_note']} "
                    f"{('related: ' + c['related']) if c['related'] else ''}")
        if c["change_type"] != "DEL" or c["match_method"] == "soft_delete":
            if not c.get("edit_reason"):
                add("WARNING", "reason_missing", intf=c["intf_id"],
                    msg=f"{c['change_type']} {c['feat_uid'] or c['orig_uid']}: no edit_reason")
        if c["match_method"] in ("removed", "unmatched") and c["change_type"] == "DEL":
            add("INFO", "hard_delete", intf=c["intf_id"],
                msg=f"{c['orig_uid']} was deleted outright -- no reason can be recorded; "
                    "prefer edit_status=deleted")
        if not c["declared_matches"]:
            add("INFO", "declared_status", intf=c["intf_id"],
                msg=f"{c['feat_uid'] or c['orig_uid']}: declared {c['declared_status']!r}, "
                    f"detected {c['change_type']}")
    return issues


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    issues = run(args.workspace, args.intf)
    out = ws_path(args.workspace, "changes") / "validation_report.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["level", "check", "intf_id", "feat_uid", "orig_uid", "message"])
        w.writeheader()
        w.writerows(issues)
    from collections import Counter

    by = Counter((i["level"], i["check"]) for i in issues)
    for (lvl, chk), n in sorted(by.items()):
        logging.info(f"{lvl:8s} {chk:28s} {n}")
    n_err = sum(i["level"] == "ERROR" for i in issues)
    logging.info(f"{n_err} error(s) -> {out}")
    raise SystemExit(1 if n_err else 0)
