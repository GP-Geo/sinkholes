"""``sinkholes relabel diff`` -- what changed between the original and working GT.

The edit fields a labeller fills in (``edit_status``, ``edit_reason``) are
*declarations*; the change list is *detected* from geometry, and a declaration
that disagrees with the geometry is reported, never trusted over it.

Matching, per interferogram, in three passes:

1. **By lineage.** A working polygon carrying ``orig_uid`` is the continuation
   of that original. Identical geometry -> unchanged, otherwise modified. One
   original carried by several working polygons is a split: the piece that
   overlaps the original most is the modification, the others are additions
   related to it.
2. **By geometry, for orphans.** QGIS loses ``orig_uid`` when a polygon is
   deleted and redrawn, or pasted from elsewhere. Originals no longer carried by
   any polygon are matched to new polygons by intersection-over-union: a pair
   that is each other's unique overlap with IoU >= ``--match_iou`` is a
   (redrawn) modification. Anything else that overlaps -- one new polygon over
   two originals, two over one, or a weak IoU -- is NOT matched: it is reported
   as additions + deletions with ``needs_review`` set and the candidates named.
3. **The rest.** Unmatched originals are deletions, unmatched new polygons are
   additions. A polygon with ``edit_status = deleted`` is a deletion whatever
   its geometry (the soft delete keeps the reason in the audit trail).

Change ids (``<intf>_ADD_001``) are kept in ``changes/change_registry.csv`` keyed
on (type, polygon identity), so re-running the diff after more editing never
renumbers an existing change; a change that disappears is marked withdrawn and
its number is not reused.
"""

import csv
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import GPKG_OPTIONS, EDIT_STATUSES, METRIC_CRS, now_iso, ws_path

TYPES = ("ADD", "DEL", "MOD")
#: Hausdorff distance (degrees) below which two geometries are the same polygon
#: (~1 mm): absorbs float round trips through QGIS, never a real vertex move.
SAME_TOL_DEG = 1e-8


def add_arguments(p):
    p.add_argument("--workspace", required=True)
    p.add_argument("--match_iou", type=float, default=0.5,
                   help="IoU at which an orphan original and a new polygon are one "
                        "redrawn object (unique mutual overlap required as well)")
    p.add_argument("--no_snapshot", action="store_true",
                   help="do not copy the working GeoPackage into history/ first")


# --------------------------------------------------------------------------- geometry helpers


def same_geometry(a, b) -> bool:
    if a is None or b is None:
        return False
    if a.equals_exact(b, 0.0):
        return True
    try:
        return a.hausdorff_distance(b) <= SAME_TOL_DEG
    except Exception:  # noqa: BLE001 -- invalid geometry: compare normalised WKB instead
        return a.normalize().wkb == b.normalize().wkb


def iou(a, b) -> float:
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
    except Exception:  # noqa: BLE001 -- invalid input; fall back to buffered repair
        a, b = a.buffer(0), b.buffer(0)
        inter, union = a.intersection(b).area, a.union(b).area
    return float(inter / union) if union > 0 else 0.0


def metric(geom):
    """Shapely geometry in UTM 36N (areas and distances in metres)."""
    import geopandas as gpd

    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]


# --------------------------------------------------------------------------- core


def detect_changes(orig, work, match_iou=0.5):
    """List of change dicts + per-feature detected status (feat_uid -> status).

    ``orig``: original_gt GeoDataFrame (orig_uid, intf_id, geometry, ...).
    ``work``: working_gt GeoDataFrame (feat_uid, orig_uid, intf_id, edit_*, geometry).
    """
    changes = []
    detected = {}
    orig_by_uid = {r.orig_uid: r for r in orig.itertuples(index=False)}

    def rec(kind, intf, o=None, w=None, **extra):
        old = o.geometry if o is not None else None
        new = w.geometry if (w is not None and kind != "DEL") else None
        c = {
            "change_type": kind, "intf_id": intf,
            "orig_uid": getattr(o, "orig_uid", None) if o is not None else getattr(w, "orig_uid", None),
            "feat_uid": getattr(w, "feat_uid", None) if w is not None else None,
            "old_geom": old, "new_geom": new,
            "edit_reason": getattr(w, "edit_reason", None) if w is not None else None,
            "edit_notes": getattr(w, "edit_notes", None) if w is not None else None,
            "edited_by": getattr(w, "edited_by", None) if w is not None else None,
            "edit_timestamp": getattr(w, "edit_timestamp", None) if w is not None else None,
            "declared_status": getattr(w, "edit_status", None) if w is not None else "(feature removed)",
            "match_method": extra.pop("match_method", ""),
            "needs_review": bool(extra.pop("needs_review", False)),
            "review_note": extra.pop("review_note", ""),
            "related": extra.pop("related", ""),
        }
        if old is not None and new is not None:
            c["iou"] = round(iou(old, new), 4)
        changes.append(c)
        return c

    intfs = sorted(set(orig["intf_id"]) | set(work["intf_id"].dropna()))
    for intf in intfs:
        O = orig[orig["intf_id"] == intf]
        W = work[work["intf_id"] == intf]
        soft_del = W[W["edit_status"].fillna("") == "deleted"]
        active = W[W["edit_status"].fillna("") != "deleted"]

        # Lineage carried into another scene: report, then treat as a move.
        moved = work[(work["intf_id"] != intf) & work["orig_uid"].isin(set(O["orig_uid"]))
                     & (work["edit_status"].fillna("") != "deleted")]
        moved_uids = set(moved["orig_uid"])

        linked = set()
        # Soft deletions of originals.
        for w in soft_del.itertuples(index=False):
            if isinstance(w.orig_uid, str) and w.orig_uid in orig_by_uid:
                if (active["orig_uid"] == w.orig_uid).any():
                    continue  # another live copy carries it: not a deletion
                o = orig_by_uid[w.orig_uid]
                rec("DEL", intf, o=o, w=w, match_method="soft_delete")
                linked.add(w.orig_uid)
            detected[w.feat_uid] = "deleted"

        # Pass 1 -- lineage.
        carried = active[active["orig_uid"].isin(set(O["orig_uid"]))]
        for uid, grp in carried.groupby("orig_uid"):
            o = orig_by_uid[uid]
            linked.add(uid)
            rows = list(grp.itertuples(index=False))
            if len(rows) == 1:
                w = rows[0]
                if same_geometry(o.geometry, w.geometry):
                    detected[w.feat_uid] = "unchanged"
                    continue
                v = iou(o.geometry, w.geometry)
                rec("MOD", intf, o=o, w=w, match_method="lineage",
                    needs_review=v == 0.0,
                    review_note="no overlap with the original it claims to be" if v == 0.0 else "")
                detected[w.feat_uid] = "modified"
                continue
            # split: the piece overlapping the original most continues it
            overlaps = [o.geometry.intersection(w.geometry).area for w in rows]
            keep = int(np.argmax(overlaps))
            uids = [w.feat_uid for w in rows]
            for k, w in enumerate(rows):
                if k == keep:
                    if same_geometry(o.geometry, w.geometry):
                        detected[w.feat_uid] = "unchanged"
                    else:
                        rec("MOD", intf, o=o, w=w, match_method="lineage_split",
                            related="split into " + ",".join(u for u in uids if u != w.feat_uid))
                        detected[w.feat_uid] = "modified"
                else:
                    rec("ADD", intf, w=w, match_method="split_piece",
                        related=f"split from {uid}",
                        review_note="shares orig_uid with another polygon (copy or split)")
                    detected[w.feat_uid] = "added"

        # Pass 2 -- geometry matching of orphans.
        orphans_o = O[~O["orig_uid"].isin(linked | moved_uids)]
        # Everything live in this scene that does not continue one of ITS
        # originals: drawn here, pasted, or moved in from another scene.
        new_w = active[~active["feat_uid"].isin(set(carried["feat_uid"]))]
        new_w = new_w[~new_w["feat_uid"].isin(detected.keys())]
        pairs = {}
        if len(orphans_o) and len(new_w):
            oi = list(orphans_o.itertuples(index=False))
            wi = list(new_w.itertuples(index=False))
            M = np.zeros((len(oi), len(wi)))
            for a, o in enumerate(oi):
                for b, w in enumerate(wi):
                    if o.geometry.intersects(w.geometry):
                        M[a, b] = iou(o.geometry, w.geometry)
            for a in range(len(oi)):
                for b in range(len(wi)):
                    if M[a, b] <= 0:
                        continue
                    unique = (M[a] > 0).sum() == 1 and (M[:, b] > 0).sum() == 1
                    if unique and M[a, b] >= match_iou:
                        pairs[a] = b
            matched_o = set(pairs)
            matched_w = set(pairs.values())
            for a, b in pairs.items():
                o, w = oi[a], wi[b]
                rec("MOD", intf, o=o, w=w, match_method=f"geometry_iou={M[a, b]:.2f}",
                    review_note="orig_uid was lost (redrawn/pasted); matched by unique overlap")
                detected[w.feat_uid] = "modified"
            for a, o in enumerate(oi):
                if a in matched_o:
                    continue
                cand = [wi[b].feat_uid for b in range(len(wi)) if M[a, b] > 0]
                rec("DEL", intf, o=o, match_method="unmatched",
                    needs_review=bool(cand),
                    review_note="overlaps new polygon(s) but not as a unique match -- "
                                "check whether this is a merge/split/redraw" if cand else "",
                    related=",".join(cand))
            for b, w in enumerate(wi):
                if b in matched_w:
                    continue
                cand = [oi[a].orig_uid for a in range(len(oi)) if M[a, b] > 0]
                rec("ADD", intf, w=w, match_method="unmatched",
                    needs_review=bool(cand),
                    review_note="overlaps deleted original(s) but not as a unique match"
                    if cand else "",
                    related=",".join(cand))
                detected[w.feat_uid] = "added"
        else:
            for o in orphans_o.itertuples(index=False):
                rec("DEL", intf, o=o, match_method="removed")
            for w in new_w.itertuples(index=False):
                rec("ADD", intf, w=w, match_method="new")
                detected[w.feat_uid] = "added"

        # Originals reassigned to another scene count as deleted here.
        for w in moved.itertuples(index=False):
            o = orig_by_uid[w.orig_uid]
            rec("DEL", intf, o=o, w=w, match_method="moved_to_other_scene",
                needs_review=True, review_note=f"intf_id changed to {w.intf_id}")

    for c in changes:
        decl = c["declared_status"]
        want = {"ADD": "added", "DEL": "deleted", "MOD": "modified"}[c["change_type"]]
        c["declared_matches"] = decl in (want, "(feature removed)")
    return changes, detected


# --------------------------------------------------------------------------- registry


def load_registry(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def change_key(c):
    """A change is identified by its type and the polygon it is about: the
    original for a deletion or modification, the new polygon for an addition."""
    ident = c["feat_uid"] if c["change_type"] == "ADD" else c["orig_uid"]
    return f"{c['change_type']}|{c['intf_id']}|{ident}"


def assign_ids(changes, registry_path):
    """Give every change its stable id; update and rewrite the registry."""
    reg = load_registry(registry_path)
    by_key = {r["change_key"]: r for r in reg}
    counters = {}
    for r in reg:
        n = int(r["change_id"].rsplit("_", 1)[1])
        counters[r["intf_id"]] = max(counters.get(r["intf_id"], 0), n)
    ts = now_iso()
    seen = set()

    def order(c):
        g = c["new_geom"] if c["new_geom"] is not None else c["old_geom"]
        pt = g.representative_point()
        return (c["intf_id"], TYPES.index(c["change_type"]), -pt.y, pt.x)

    for c in sorted(changes, key=order):
        k = change_key(c)
        seen.add(k)
        if k in by_key:
            r = by_key[k]
            r["last_seen"], r["state"] = ts, "active"
        else:
            n = counters.get(c["intf_id"], 0) + 1
            counters[c["intf_id"]] = n
            r = {"change_key": k, "change_id": f"{c['intf_id']}_{c['change_type']}_{n:03d}",
                 "intf_id": c["intf_id"], "change_type": c["change_type"],
                 "first_seen": ts, "last_seen": ts, "state": "active"}
            reg.append(r)
            by_key[k] = r
        c["change_id"] = r["change_id"]
    for r in reg:
        if r["change_key"] not in seen and r["state"] == "active":
            r["state"] = "withdrawn"
            r["last_seen"] = ts
    Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(registry_path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["change_id", "change_key", "intf_id", "change_type",
                                           "state", "first_seen", "last_seen"])
        w.writeheader()
        w.writerows(sorted(reg, key=lambda r: r["change_id"]))
    os.replace(tmp, registry_path)
    return changes


# --------------------------------------------------------------------------- outputs


def enrich(changes, scenes):
    """Areas, shift, centroid, dates -- the columns the audit layers carry."""
    frame = dict(zip(scenes["intf_id"], scenes["frame"]))
    for c in changes:
        old, new = c["old_geom"], c["new_geom"]
        mo = metric(old) if old is not None else None
        mn = metric(new) if new is not None else None
        c["area_before_m2"] = round(mo.area, 1) if mo is not None else None
        c["area_after_m2"] = round(mn.area, 1) if mn is not None else None
        c["centroid_shift_m"] = (round(mo.centroid.distance(mn.centroid), 1)
                                 if mo is not None and mn is not None else None)
        ref = new if new is not None else old
        pt = ref.representative_point()
        c["lon"], c["lat"] = round(pt.x, 6), round(pt.y, 6)
        c["start_date"] = f"{c['intf_id'][:4]}-{c['intf_id'][4:6]}-{c['intf_id'][6:8]}"
        c["end_date"] = f"{c['intf_id'][9:13]}-{c['intf_id'][13:15]}-{c['intf_id'][15:17]}"
        c["frame"] = frame.get(c["intf_id"], "")
    return changes


ATTRS = ["change_id", "intf_id", "change_type", "start_date", "end_date", "frame", "orig_uid",
         "feat_uid", "edit_reason", "edit_notes", "edited_by", "edit_timestamp",
         "declared_status", "declared_matches", "match_method", "needs_review", "review_note",
         "related", "iou", "area_before_m2", "area_after_m2", "centroid_shift_m", "lon", "lat"]


def write_change_layers(changes, out_gpkg):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    tmp = Path(str(out_gpkg) + ".tmp.gpkg")
    if tmp.exists():
        tmp.unlink()

    def frame_of(rows, geom_key):
        df = pd.DataFrame([{k: c.get(k) for k in ATTRS} for c in rows], columns=ATTRS)
        geoms = [c[geom_key] for c in rows]
        return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")

    layers = {
        "added": ([c for c in changes if c["change_type"] == "ADD"], "new_geom"),
        "deleted": ([c for c in changes if c["change_type"] == "DEL"], "old_geom"),
        "modified_original": ([c for c in changes if c["change_type"] == "MOD"], "old_geom"),
        "modified_corrected": ([c for c in changes if c["change_type"] == "MOD"], "new_geom"),
    }
    for name, (rows, key) in layers.items():
        frame_of(rows, key).to_file(tmp, layer=name, driver="GPKG",
            dataset_options=GPKG_OPTIONS)
    markers = gpd.GeoDataFrame(
        pd.DataFrame([{k: c.get(k) for k in ATTRS} for c in changes], columns=ATTRS),
        geometry=[Point(c["lon"], c["lat"]) for c in changes], crs="EPSG:4326")
    markers.to_file(tmp, layer="change_markers", driver="GPKG",
            dataset_options=GPKG_OPTIONS)
    os.replace(tmp, out_gpkg)


def summarise(changes, orig, work, detected, scenes):
    rows = []
    for s in scenes.itertuples(index=False):
        intf = s.intf_id
        cs = [c for c in changes if c["intf_id"] == intf]
        W = work[(work["intf_id"] == intf) & (work["edit_status"].fillna("") != "deleted")]
        n = {t: sum(c["change_type"] == t for c in cs) for t in TYPES}
        rows.append({
            "intf_id": intf, "role": s.role, "frame": s.frame,
            "start_date": s.start_date, "end_date": s.end_date,
            "original_count": int((orig["intf_id"] == intf).sum()),
            "corrected_count": int(len(W)),
            "unchanged": int(sum(detected.get(u) == "unchanged" for u in W["feat_uid"])),
            "added": n["ADD"], "deleted": n["DEL"], "modified": n["MOD"],
            "needs_review": sum(c["needs_review"] for c in cs),
            "declared_status_mismatches": sum(not c["declared_matches"] for c in cs),
        })
    return rows


def update_manifest(path, summary):
    if not Path(path).exists():
        return
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys()) if rows else []
    by = {r["intf_id"]: r for r in summary}
    for r in rows:
        s = by.get(r["intf_id"])
        if s:
            r["corrected_count"] = s["corrected_count"]
            r["added"], r["deleted"], r["modified"] = s["added"], s["deleted"], s["modified"]
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_layers(workspace):
    import geopandas as gpd

    from .common import check_original_unchanged

    orig = gpd.read_file(ws_path(workspace, "original_gpkg"), layer="original_gt")
    check_original_unchanged(workspace, orig)
    scenes = gpd.read_file(ws_path(workspace, "original_gpkg"), layer="scenes")
    work = gpd.read_file(ws_path(workspace, "working_gpkg"), layer="working_gt")
    for col in ("feat_uid", "orig_uid", "intf_id", "edit_status"):
        if col not in work.columns:
            work[col] = None
    work["edit_status"] = work["edit_status"].where(work["edit_status"].notna(), None)
    # A polygon drawn without the form's default (feat_uid = uuid()) has no
    # identity; derive one from its geometry so its change id is still stable
    # while it is not reshaped. validate reports these.
    missing = work["feat_uid"].isna() | (work["feat_uid"].astype(str).str.strip() == "")
    if missing.any():
        import hashlib

        work.loc[missing, "feat_uid"] = [
            "geom:" + hashlib.sha1(g.wkb).hexdigest()[:12] if g is not None else f"row:{i}"
            for i, g in zip(work.index[missing], work.geometry[missing])]
    return orig, work, scenes


def snapshot(workspace):
    src = ws_path(workspace, "working_gpkg")
    dst = ws_path(workspace, "history") / f"gt_test_working_{datetime.now():%Y%m%d_%H%M%S}.gpkg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def run(workspace, match_iou=0.5, take_snapshot=True):
    """Detect, number and write all changes; returns (changes, summary)."""
    import csv as _csv

    if take_snapshot:
        logging.info(f"snapshot -> {snapshot(workspace)}")
    orig, work, scenes = load_layers(workspace)
    unknown = sorted(set(work["edit_status"].dropna()) - set(EDIT_STATUSES))
    if unknown:
        logging.warning(f"unknown edit_status values (treated as not deleted): {unknown}")
    changes, detected = detect_changes(orig, work, match_iou)
    assign_ids(changes, ws_path(workspace, "registry"))
    enrich(changes, scenes)
    out = ws_path(workspace, "changes")
    out.mkdir(parents=True, exist_ok=True)
    write_change_layers(changes, out / "gt_changes_v2.gpkg")
    with open(out / "change_list.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=ATTRS)
        w.writeheader()
        w.writerows([{k: c.get(k) for k in ATTRS} for c in sorted(changes, key=lambda c: c["change_id"])])
    summary = summarise(changes, orig, work, detected, scenes)
    with open(out / "change_summary.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    update_manifest(ws_path(workspace, "manifest"), summary)
    return changes, summary, detected


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    changes, summary, _ = run(args.workspace, args.match_iou, not args.no_snapshot)
    tot = {t: sum(c["change_type"] == t for c in changes) for t in TYPES}
    logging.info(f"{len(changes)} changes: {tot}; "
                 f"{sum(c['needs_review'] for c in changes)} need review")
    for s in summary:
        if s["added"] or s["deleted"] or s["modified"]:
            logging.info(f"  {s['intf_id']}: {s['original_count']} -> {s['corrected_count']} "
                         f"(+{s['added']} -{s['deleted']} ~{s['modified']})")
    logging.info(f"-> {ws_path(args.workspace, 'changes')}")
