"""``sinkholes relabel eval`` / ``compare`` -- old vs corrected labels, same predictions.

Inference never reads the ground truth (``reconstruct_scene`` gates on LiDAR,
the AOI and -- only under ``--positives_only`` -- the GT; the GT canvas is
accumulated beside the prediction, not into it). So a saved ``_pred.npy`` is
the model's answer whatever the labels say, and relabelling only needs the GT
side re-derived. For each saved evaluation directory this:

1. reads the eval-scenes arguments from its log and refuses directories whose
   predictions DO depend on the GT (``positives_only``) or whose GT is not the
   current scene's alone (``unioned_mask``);
2. rebuilds every scene's GT canvas from the ORIGINAL shapefile and requires it
   to equal the saved ``_gt.npy`` bit for bit -- the proof that the canvas,
   crop, alignment and rasterisation are reproduced exactly, so that the
   corrected canvas differs from the original in the labels and nothing else;
3. rebuilds the canvas from the corrected GT (cached per canvas shape);
4. scores the unchanged ``_pred.npy`` twice with the unchanged ``eval-outputs``
   code -- once against the saved GT, once against the corrected one (via
   ``--gt_map``) -- with the directory's own stride, AOI and protocol;
5. optionally (``--objects``) classifies every predicted and GT object under
   both labelings at one operating point, which is what answers "which false
   positives were missing GT".

Nothing is written into the source directory. Results go to
``<workspace>/eval/<label>/<model>/<job>/``.
"""

import argparse
import glob
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from ..paths import REPO_ROOT
from .common import (
    GPKG_OPTIONS,
    ORIGINAL_GT_SHP,
    USER_FLAGGED,
    aligned_mask,
    gt_canvas,
    list_eval_intfs,
    now_iso,
    read_active_gt,
    read_eval_args,
    scene_geometry,
    write_json,
    ws_path,
)


def add_arguments(p):
    sub = p.add_subparsers(dest="eval_cmd", required=True)
    e = sub.add_parser("run", help="re-score saved evaluations against corrected labels")
    e.add_argument("--workspace", required=True)
    e.add_argument("--data_dir", required=True, help="holds sub_20260701.shp")
    e.add_argument("--eval_dirs", nargs="*", default=None,
                   help="saved eval-scenes directories (read-only). Default: every "
                        "generation-4 k=5 vote-protocol directory under --predictions_root")
    e.add_argument("--predictions_root", default=str(REPO_ROOT / "outputs" / "predictions"))
    e.add_argument("--gt", default=None,
                   help="corrected GT (default: exports/gt_test_corrected_<label>.gpkg, layer gt)")
    e.add_argument("--label", default="v2", help="name of this corrected-label version")
    e.add_argument("--objects", action="store_true",
                   help="also write per-object old/new classification at --object_th")
    e.add_argument("--object_th", type=float, default=0.25)
    e.add_argument("--verify_only", action="store_true",
                   help="only prove each saved _gt.npy is reproduced from the original GT "
                        "(no corrected GT needed, nothing scored); writes eval/verify_*.json")
    c = sub.add_parser("compare", help="tables: old vs corrected, per scene and per model")
    c.add_argument("--workspace", required=True)
    c.add_argument("--label", default="v2")
    c.add_argument("--th", default="0.25", help="operating point to tabulate (string key)")


# --------------------------------------------------------------------------- discovery


def discover(predictions_root):
    """Generation-4 vote-protocol evaluations (k=5 list, and the k=10 subset list)."""
    out = []
    for d in sorted(glob.glob(os.path.join(predictions_root, "*", "*", "scenes_*"))):
        try:
            a = read_eval_args(d)
        except (FileNotFoundError, ValueError, SyntaxError):
            continue
        if (Path(str(a.get("valset_from_partition", ""))).name in (
                "partition_temporal_k5_testeval_clean_th350x200.json",
                "partition_temporal_k10_testeval_clean_th350x200.json")
                and a.get("recon_average") == "vote" and not a.get("positives_only")):
            out.append(d)
    return out


# --------------------------------------------------------------------------- canvases


def _geoms_by_scene(gdf):
    return {k: list(v.geometry) for k, v in gdf.groupby("intf_id")}


def build_gt_canvas(intf, geoms, shape, patch, stride):
    geo = scene_geometry(intf)
    return gt_canvas(aligned_mask(geoms, geo), shape, patch, stride)


def canvas_path(cache_root, label, stride, patch, shape, intf):
    d = Path(cache_root) / label / f"s{stride}_H{patch[0]}_W{patch[1]}" / f"{shape[0]}x{shape[1]}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{intf}_gt.npy"


def fingerprint(geoms):
    h = hashlib.sha256()
    for g in sorted((g.wkb for g in geoms if g is not None)):
        h.update(g)
    return h.hexdigest()


def outputs_args(src_dir, a, out_json, gt_map=None):
    """Namespace for inference.outputs.main reproducing the directory's scoring."""
    from ..inference import outputs

    p = argparse.ArgumentParser()
    outputs.add_arguments(p)
    argv = ["--path", src_dir, "--out_json", str(out_json),
            "--patch_size", *map(str, a["patch_size"]), "--data_stride", str(a["data_stride"])]
    if a.get("recon_average") == "vote":
        argv.append("--rth")
    if a.get("aoi_from_partition"):
        part = a["aoi_from_partition"]
        if not os.path.isabs(part):
            part = str(REPO_ROOT / part)
        argv += ["--aoi_from_partition", part, "--aoi_split", a.get("aoi_split", "test")]
    elif a.get("aoi_window"):
        argv += ["--aoi_window", *map(str, a["aoi_window"])]
    if a.get("intf_dict_path"):
        argv += ["--intf_dict_path", a["intf_dict_path"]]
    if gt_map:
        argv += ["--gt_map", str(gt_map)]
    return p.parse_args(argv)


def run_one(src_dir, ws, label, orig_by, corr_by, orig_fp, corr_fp, objects, object_th,
            flagged_by=None):
    from ..inference import outputs

    a = read_eval_args(src_dir)
    if a.get("positives_only"):
        raise SystemExit(f"{src_dir}: positives-only predictions depend on the GT; not reusable")
    if a.get("unioned_mask"):
        raise SystemExit(f"{src_dir}: --unioned_mask GT is not a single scene's labels")
    patch, stride = tuple(a["patch_size"]), int(a["data_stride"])
    rel = Path(src_dir).resolve().relative_to(Path(src_dir).resolve().parents[2])
    out_dir = ws_path(ws, "eval") / label / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ws_path(ws, "eval") / "gt_canvases"
    intfs = list_eval_intfs(src_dir)
    verify, gt_map = {}, {}
    for intf in intfs:
        conf = np.load(os.path.join(src_dir, f"{intf}_pred.npy"), mmap_mode="r")
        shape = conf.shape
        saved = np.load(os.path.join(src_dir, f"{intf}_gt.npy"), mmap_mode="r")
        rebuilt = build_gt_canvas(intf, orig_by.get(intf, []), shape, patch, stride)
        same = bool(np.array_equal(rebuilt, saved))
        verify[intf] = same
        if not same:
            raise SystemExit(
                f"{src_dir}/{intf}_gt.npy is NOT reproduced from {ORIGINAL_GT_SHP} "
                f"(max |diff| {float(np.abs(rebuilt - saved).max())}). This evaluation was built "
                f"from different labels or geometry; a before/after comparison on it would not "
                f"isolate the relabelling. Leave it out (--eval_dirs).")
        if intf not in corr_fp:
            raise SystemExit(f"{intf} is not in the corrected GT's scope")
        cp = canvas_path(cache, f"{label}_{corr_fp[intf][:12]}", stride, patch, shape, intf)
        if not cp.exists():
            if corr_fp[intf] == orig_fp.get(intf):
                arr = rebuilt                       # unchanged scene: identical by construction
            else:
                arr = build_gt_canvas(intf, corr_by.get(intf, []), shape, patch, stride)
            tmp = cp.with_name(cp.name + ".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, cp)
        gt_map[intf] = str(cp)
        logging.info(f"  {intf}: saved GT reproduced; corrected canvas "
                     f"{'(unchanged)' if corr_fp[intf] == orig_fp.get(intf) else '(relabelled)'}")
    write_json(out_dir / "gt_map.json", gt_map)

    old_json, new_json = out_dir / "olm_original.json", out_dir / f"olm_corrected_{label}.json"
    outputs.main(outputs_args(src_dir, a, old_json))
    outputs.main(outputs_args(src_dir, a, new_json, gt_map=out_dir / "gt_map.json"))

    # Is today's scoring code reproducing the archived numbers for the old GT?
    archived = sorted(glob.glob(os.path.join(src_dir, "olm_results_*.json")))
    repro = None
    if archived:
        A = json.load(open(archived[-1]))
        B = json.load(open(old_json))
        diffs = [abs(A["per_intf"][i][t][k] - B["per_intf"][i][t][k])
                 for i in B["per_intf"] if i in A["per_intf"]
                 for t in B["per_intf"][i] if t in A["per_intf"][i]
                 for k in ("recall", "precision")]
        repro = {"archived": archived[-1], "max_abs_diff": max(diffs) if diffs else None}
    meta = {"created": now_iso(), "source_dir": str(Path(src_dir).resolve()),
            "model": rel.parts[0], "job": rel.parts[-1], "eval_scenes_args": a,
            "saved_gt_reproduced": verify, "rescore_vs_archived": repro,
            "corrected_gt_fingerprints": {i: corr_fp[i] for i in intfs},
            "relabelled_scenes": sorted(i for i in intfs if corr_fp[i] != orig_fp.get(i))}
    write_json(out_dir / "meta.json", meta)
    if objects:
        object_analysis(src_dir, a, gt_map, out_dir, object_th, flagged_by or {})
    return out_dir


# --------------------------------------------------------------------------- per-object


def aoi_slices(meta_frame, aoi, patch, stride, shape):
    """The canvas crop inference/outputs.py applies (same formula, same order)."""
    from ..geo import grid_window

    ph, pw = patch
    r0, r1, c0, c1 = grid_window(meta_frame, *aoi, patch_size=(ph, pw), stride=(ph // stride, pw // stride))
    H, W = shape
    sy, sx = ph // stride, pw // stride
    return (slice(min(r0 * sy, H), min(r1 * sy + ph - sy, H)),
            slice(min(c0 * sx, W), min(c1 * sx + pw - sx, W)))


def classify_objects(gt, pred, th=0.7, buffer=5):
    """Per-object version of training.evaluate.object_level_evaluate (one scene).

    Returns (gt_objects, pred_objects, recall, precision) with the same
    definitions: a GT object is detected when the buffered prediction union
    covers more than ``th`` of it; a predicted object is a false positive when
    less than ``th`` of it lies in the buffered GT union.
    """
    import rasterio.features
    from affine import Affine
    from shapely.geometry import shape
    from shapely.ops import unary_union

    t = Affine.identity()
    gtp = [shape(g) for g, v in rasterio.features.shapes(gt.astype(np.float32), transform=t) if v > 0]
    prp = [shape(g) for g, v in rasterio.features.shapes(pred.astype(np.float32), transform=t) if v > 0]
    bpu = unary_union([p.buffer(buffer) for p in prp])
    bgu = unary_union([g.buffer(buffer) for g in gtp])
    gt_rows, pr_rows = [], []
    det_area = 0.0
    for g in gtp:
        f = g.intersection(bpu).area / g.area
        gt_rows.append((g, f > th, f))
        det_area += g.area if f > th else 0.0
    fp_area = 0.0
    for p in prp:
        f = p.intersection(bgu).area / p.area
        pr_rows.append((p, f >= th, f))
        fp_area += p.area if f < th else 0.0
    tot = sum(g.area for g in gtp)
    eps = 1e-7
    rec = det_area / (tot + eps) if tot else float("nan")
    prec = min(det_area / (det_area + fp_area + eps), 1.0) if tot else float("nan")
    return gt_rows, pr_rows, rec, prec


def object_analysis(src_dir, a, gt_map, out_dir, th_conf, flagged_by=None):
    """Every predicted / GT object, old vs corrected labels, as GIS layers + CSV."""
    import geopandas as gpd
    from shapely.affinity import affine_transform

    from ..dataprep.partition import load_partition_window
    from ..meta import intf_meta

    part = a.get("aoi_from_partition")
    part = part if not part or os.path.isabs(part) else str(REPO_ROOT / part)
    aoi = load_partition_window(part, a.get("aoi_split", "test")) if part else tuple(a["aoi_window"])
    patch, stride = tuple(a["patch_size"]), int(a["data_stride"])
    pred_rows, gt_rows_all, summary = [], [], []
    for intf, newp in sorted(gt_map.items()):
        m = intf_meta(intf)
        geo = scene_geometry(intf)
        conf = np.load(os.path.join(src_dir, f"{intf}_pred.npy"), mmap_mode="r")
        rs, cs = aoi_slices(m.frame, aoi, patch, stride, conf.shape)
        pred = np.where(np.asarray(conf[rs, cs]) > th_conf, 1, 0)
        old = np.asarray(np.load(os.path.join(src_dir, f"{intf}_gt.npy"), mmap_mode="r")[rs, cs])
        new = np.asarray(np.load(newp, mmap_mode="r")[rs, cs])
        go, po, ro, pro = classify_objects(old, pred)
        gn, pn, rn, prn = classify_objects(new, pred)
        # pixel (col,row) of the crop -> lon/lat of the ACTUAL canvas origin
        x0 = geo.canvas_x0 + cs.start * geo.dx
        y0 = geo.canvas_y0 - rs.start * geo.dy
        T = [geo.dx, 0, 0, -geo.dy, x0, y0]
        for (p, tp_old, fo), (_, tp_new, fn_) in zip(po, pn):
            pred_rows.append({"intf_id": intf, "status_old": "TP" if tp_old else "FP",
                              "status_new": "TP" if tp_new else "FP",
                              "transition": f"{'TP' if tp_old else 'FP'}->{'TP' if tp_new else 'FP'}",
                              "gt_overlap_old": round(fo, 3), "gt_overlap_new": round(fn_, 3),
                              "area_px": p.area, "geometry": affine_transform(p, T)})
        # Quality flags of the corrected labels: a GT object is flagged when it
        # overlaps a flagged polygon. Flagged objects stay in every number above;
        # this only splits recall so their share of the misses is visible.
        from shapely.ops import unary_union

        fl = unary_union(list((flagged_by or {}).get(intf, []))) if (flagged_by or {}).get(intf) else None
        flag_area = {True: [0.0, 0.0], False: [0.0, 0.0]}      # [detected, total] px
        for tag, rows in (("original", go), ("corrected", gn)):
            for g, det, f in rows:
                geo_g = affine_transform(g, T)
                is_fl = bool(fl is not None and tag == "corrected" and geo_g.intersects(fl))
                if tag == "corrected":
                    flag_area[is_fl][0] += g.area if det else 0.0
                    flag_area[is_fl][1] += g.area
                gt_rows_all.append({"intf_id": intf, "labels": tag, "detected": bool(det),
                                    "covered_fraction": round(f, 3), "area_px": g.area,
                                    "qc_flagged": is_fl, "geometry": geo_g})
        summary.append({"intf_id": intf, "recall_old": ro, "precision_old": pro,
                        "recall_new": rn, "precision_new": prn,
                        "FP_to_TP": sum(r["transition"] == "FP->TP" for r in pred_rows if r["intf_id"] == intf),
                        "TP_to_FP": sum(r["transition"] == "TP->FP" for r in pred_rows if r["intf_id"] == intf),
                        "gt_missed_old": sum(not d for _, d, _ in go),
                        "gt_missed_new": sum(not d for _, d, _ in gn),
                        "recall_new_flagged": (flag_area[True][0] / flag_area[True][1]
                                               if flag_area[True][1] else None),
                        "recall_new_unflagged": (flag_area[False][0] / flag_area[False][1]
                                                 if flag_area[False][1] else None),
                        "flagged_gt_area_share": (flag_area[True][1] / (flag_area[True][1] + flag_area[False][1])
                                                  if (flag_area[True][1] + flag_area[False][1]) else None)})
    gpkg = Path(out_dir) / f"objects_th{th_conf}.gpkg"
    if pred_rows:
        gpd.GeoDataFrame(pred_rows, crs="EPSG:4326").to_file(gpkg, layer="pred_objects", driver="GPKG",
            dataset_options=GPKG_OPTIONS)
    if gt_rows_all:
        gpd.GeoDataFrame(gt_rows_all, crs="EPSG:4326").to_file(gpkg, layer="gt_objects", driver="GPKG",
            dataset_options=GPKG_OPTIONS)
    import pandas as pd

    pd.DataFrame(summary).to_csv(Path(out_dir) / f"objects_th{th_conf}_summary.csv", index=False)


# --------------------------------------------------------------------------- compare


def f1(r, p):
    return 2 * r * p / (r + p) if r is not None and p is not None and (r + p) > 0 else None


def compare(ws, label, th):
    import pandas as pd

    root = ws_path(ws, "eval") / label
    per_scene, agg = [], []
    for meta_f in sorted(root.glob("*/**/meta.json")):
        d = meta_f.parent
        meta = json.load(open(meta_f))
        old = json.load(open(d / "olm_original.json"))
        new = json.load(open(d / f"olm_corrected_{label}.json"))
        model = meta["model"]
        scene_set = Path(str(meta["eval_scenes_args"].get("valset_from_partition", ""))).name
        for key in old.get("summary_by_tolerance", {"primary": None}):
            so = old["summary_by_tolerance"][key][th] if key != "primary" else old["summary"][th]
            sn = new["summary_by_tolerance"][key][th] if key != "primary" else new["summary"][th]
            row = {"model": model, "job": meta["job"], "scene_set": scene_set,
                   "tolerance": key, "th": th}
            for tag, s in (("old", so), ("new", sn)):
                row[f"w_recall_{tag}"] = s["weighted_recall"]
                row[f"w_precision_{tag}"] = s["weighted_precision"]
                row[f"w_f1_{tag}"] = f1(s["weighted_recall"], s["weighted_precision"])
                row[f"mean_recall_{tag}"] = s["mean_recall"]
                row[f"mean_precision_{tag}"] = s["mean_precision"]
            for k in ("w_recall", "w_precision", "w_f1"):
                if row[f"{k}_old"] is not None and row[f"{k}_new"] is not None:
                    row[f"d_{k}"] = row[f"{k}_new"] - row[f"{k}_old"]
            agg.append(row)
        po, pn = old["per_intf"], new["per_intf"]
        for intf in po:
            o, n = po[intf].get(th), pn[intf].get(th)
            if not o or not n:
                continue
            per_scene.append({"model": model, "intf_id": intf,
                              "relabelled": intf in meta["relabelled_scenes"],
                              "user_flagged": intf in USER_FLAGGED,
                              "recall_old": o["recall"], "recall_new": n["recall"],
                              "precision_old": o["precision"], "precision_new": n["precision"],
                              "gt_area_old": o["gt_area"], "gt_area_new": n["gt_area"],
                              "d_recall": n["recall"] - o["recall"],
                              "d_precision": n["precision"] - o["precision"]})
    if not agg:
        raise SystemExit(f"nothing under {root}; run 'relabel eval run' first")
    A, S = pd.DataFrame(agg), pd.DataFrame(per_scene)
    out = root / "comparison"
    out.mkdir(exist_ok=True)
    A.to_csv(out / f"aggregate_th{th}.csv", index=False)
    S.to_csv(out / f"per_scene_th{th}.csv", index=False)
    # scene view: averaged over models
    sv = (S.groupby(["intf_id", "relabelled", "user_flagged"])
          [["recall_old", "recall_new", "precision_old", "precision_new", "d_recall", "d_precision"]]
          .mean().reset_index().sort_values("d_recall", ascending=False))
    sv.to_csv(out / f"per_scene_mean_over_models_th{th}.csv", index=False)
    # Model ranking old vs new (primary tolerance, weighted F1), only ever
    # WITHIN one scene list: a k=10 model is scored on 17 scenes, a k=5 one on 20.
    prim = A[A["tolerance"] == A["tolerance"].iloc[0]].copy()
    prim["rank_old"] = prim.groupby("scene_set")["w_f1_old"].rank(ascending=False)
    prim["rank_new"] = prim.groupby("scene_set")["w_f1_new"].rank(ascending=False)
    prim.sort_values(["scene_set", "rank_new"]).to_csv(out / f"model_ranking_th{th}.csv", index=False)
    lines = [f"# Old vs corrected labels ({label}), operating point {th}", ""]
    for sset, grp in prim.groupby("scene_set"):
        try:
            from scipy.stats import kendalltau

            tau = kendalltau(grp["rank_old"], grp["rank_new"]).statistic if len(grp) > 1 else None
        except Exception:  # noqa: BLE001
            tau = None
        lines += [f"## {sset}: {len(grp)} models; ranking Kendall tau old vs new (weighted F1): "
                  f"{'n/a' if tau is None or tau != tau else round(float(tau), 3)}", "",
                  "| model | w_recall old->new | w_precision old->new | w_F1 old->new | rank old->new |",
                  "|---|---|---|---|---|"]
        for r in grp.sort_values("rank_new").itertuples():
            lines.append(f"| {r.model} | {r.w_recall_old:.4f} -> {r.w_recall_new:.4f} | "
                         f"{r.w_precision_old:.4f} -> {r.w_precision_new:.4f} | "
                         f"{r.w_f1_old:.4f} -> {r.w_f1_new:.4f} | {r.rank_old:g} -> {r.rank_new:g} |")
        lines.append("")
    lines += ["", "## Scenes (mean over models), largest recall change first", "",
              "| scene | relabelled | flagged | recall old->new | precision old->new |", "|---|---|---|---|---|"]
    for r in sv.itertuples():
        lines.append(f"| {r.intf_id} | {r.relabelled} | {r.user_flagged} | {r.recall_old:.3f} -> "
                     f"{r.recall_new:.3f} | {r.precision_old:.3f} -> {r.precision_new:.3f} |")
    (out / f"comparison_th{th}.md").write_text("\n".join(lines) + "\n")
    logging.info("\n".join(lines))
    logging.info(f"-> {out}")


def verify_all(ws, args):
    """Bit-identity of every saved _gt.npy with a rebuild from the original GT."""
    from datetime import datetime

    orig = read_active_gt(os.path.join(args.data_dir, ORIGINAL_GT_SHP))
    orig_by = _geoms_by_scene(orig)
    dirs = args.eval_dirs or discover(args.predictions_root)
    report = {}
    for d in dirs:
        a = read_eval_args(d)
        patch, stride = tuple(a["patch_size"]), int(a["data_stride"])
        res = {}
        for intf in list_eval_intfs(d):
            saved = np.load(os.path.join(d, f"{intf}_gt.npy"), mmap_mode="r")
            rebuilt = build_gt_canvas(intf, orig_by.get(intf, []), saved.shape, patch, stride)
            res[intf] = bool(np.array_equal(rebuilt, saved))
        report[str(Path(d).resolve())] = {"stride": stride, "scenes": len(res),
                                          "all_identical": all(res.values()),
                                          "not_identical": sorted(k for k, v in res.items() if not v)}
        logging.info(f"{'OK ' if all(res.values()) else 'BAD'} {len(res)} scenes  {d}")
    out = ws_path(ws, "eval") / f"verify_saved_gt_{datetime.now():%Y%m%d_%H%M}.json"
    write_json(out, {"created": now_iso(), "source_gt": ORIGINAL_GT_SHP, "dirs": report})
    logging.info(f"-> {out}")


# --------------------------------------------------------------------------- main


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ws = Path(args.workspace)
    if args.eval_cmd == "compare":
        compare(ws, args.label, args.th)
        return
    if args.verify_only:
        verify_all(ws, args)
        return
    gt_path = args.gt or str(ws_path(ws, "exports") / f"gt_test_corrected_{args.label}.gpkg")
    if not Path(gt_path).exists():
        raise SystemExit(f"{gt_path} not found -- run 'sinkholes relabel export' first")
    corr = read_active_gt(gt_path, layer="gt" if gt_path.endswith(".gpkg") else None)
    orig = read_active_gt(os.path.join(args.data_dir, ORIGINAL_GT_SHP))
    scope = set(corr["intf_id"]) | set(
        __import__("geopandas").read_file(ws_path(ws, "original_gpkg"), layer="scenes")["intf_id"])
    orig = orig[orig["intf_id"].isin(scope)]
    orig_by, corr_by = _geoms_by_scene(orig), _geoms_by_scene(corr)
    flagged_by = (_geoms_by_scene(corr[corr["qc_flag"].notna() & (corr["qc_flag"].astype(str) != "")])
                  if "qc_flag" in corr.columns else {})
    orig_fp = {i: fingerprint(orig_by.get(i, [])) for i in scope}
    corr_fp = {i: fingerprint(corr_by.get(i, [])) for i in scope}
    dirs = args.eval_dirs or discover(args.predictions_root)
    logging.info(f"{len(dirs)} evaluation directories; "
                 f"{sum(orig_fp[i] != corr_fp[i] for i in scope)} relabelled scene(s)")
    for d in dirs:
        logging.info(f"== {d}")
        out = run_one(d, ws, args.label, orig_by, corr_by, orig_fp, corr_fp,
                      args.objects, args.object_th, flagged_by)
        logging.info(f"-> {out}")
