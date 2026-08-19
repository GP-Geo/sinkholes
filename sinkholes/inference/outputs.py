"""Metrics and figures over saved full-scene outputs.

Run as ``sinkholes eval-outputs`` on a directory written by
``sinkholes eval-scenes`` (the ``<intf>_image/_pred/_gt.npy`` files). The
confidence map is re-thresholded at several operating points and object-level
precision/recall is computed at each; South-frame scenes are cropped to their
northern half (the populated part of that frame). Results go to a JSON next to
the inputs; ``--save_figures`` adds a per-interferogram overview PNG.
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np

from ..geo import aligned_origin, grid_window
from ..meta import INTF_ID_RE, intf_meta

#: Operating points the confidence map is re-thresholded at. The upper end
#: matters: models trained with --nonz_only (the default) never see a
#: background-only patch, so they over-predict at scene scale and the useful
#: operating point sits well above the 0.25 that eval-scenes thresholds at.
THRESHOLDS = (0.125, 0.25, 0.5, 0.7, 0.9)

#: The benchmark paper's Reconstruction Thresholds. Meaningful only against a
#: vote-mode confidence map (eval-scenes --recon_average vote), where the value
#: is the FRACTION OF OVERLAPPING TILES voting positive: at the paper's
#: quarter-patch stride these are "at least 2, 4, 6, 8 of 16 tiles agreed".
#: The paper reports 0.125/0.25/0.5 in its Table 2 and uses 0.375 for the
#: polygon-level analysis, so all four are swept here.
#: An RTh is NOT comparable with a THRESHOLDS entry -- one counts tiles, the
#: other cuts a mean probability.
RTH_THRESHOLDS = (0.125, 0.25, 0.375, 0.5)

#: (intersection threshold, buffer px) pairs the paper reports reconstruction
#: metrics under: its default Intersection Tolerance and its softer one.
#: The FIRST entry is the primary and fills the top-level per_intf/summary keys.
DEFAULT_TOLERANCE = (0.7, 5)
SOFT_TOLERANCE = (0.5, 10)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--path", type=str, required=True,
                   help="an eval-scenes output directory (holds <intf>_*.npy)")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS),
                   metavar="T", help="confidence operating points to score at")
    p.add_argument("--th", type=float, default=0.7, help="object-level overlap threshold")
    p.add_argument("--buffer", type=int, default=5, help="object-matching buffer, pixels")
    p.add_argument("--extra_tolerance", type=str, nargs="*", default=None, metavar="ITH:BUFFER",
                   help="additional Intersection Tolerance settings to score under, e.g. "
                        "'0.5:10' for the paper's softer parameters. --th/--buffer stay the "
                        "PRIMARY setting and keep the top-level 'per_intf'/'summary' keys; "
                        "every setting including the primary is also reported under "
                        "'by_tolerance'. Pass no value for the historical single-tolerance "
                        "output")
    p.add_argument("--rth", action="store_true",
                   help="score the paper's Reconstruction Thresholds (0.125/0.25/0.375/0.5) "
                        "and its two Intersection Tolerances, overriding --thresholds and "
                        "--extra_tolerance. Only meaningful on a map reconstructed with "
                        "eval-scenes --recon_average vote; on a probability map the numbers "
                        "are silently meaningless, which is why this is opt-in")
    p.add_argument("--skip_ol_metrics", action="store_true")
    p.add_argument("--save_figures", action="store_true",
                   help="write a per-interferogram overview PNG next to the metrics")
    p.add_argument("--out_json", type=str, default=None,
                   help="metrics output (default: <path>/olm_results_<ts>.json)")
    p.add_argument("--intf_dict_path", type=str, default=None)
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"),
                   help="patch geometry the canvas was reconstructed with")
    p.add_argument("--data_stride", type=int, default=2,
                   help="strides per patch used by eval-scenes; must match, or the AOI "
                        "crop lands on the wrong pixels")
    p.add_argument("--aoi_window", nargs=4, type=float, default=None,
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                   help="crop the canvas to this box before scoring. Must match the window "
                        "eval-scenes predicted with, or the metrics cover different ground "
                        "than the predictions")
    p.add_argument("--aoi_from_partition", type=str, default=None,
                   help="partition JSON to read the AOI window from (preferred over "
                        "--aoi_window: one source for all three consumers)")
    p.add_argument("--aoi_split", type=str, default="test",
                   help="which split's window to read from --aoi_from_partition")
    p.add_argument("--legacy_south_half", action="store_true",
                   help="reproduce the pre-AOI rule that scored only the northern half of a "
                        "South-frame canvas. Superseded by --aoi_window; kept so archived "
                        "evaluations can be re-scored exactly as they were")


def scene_figure(out_png, image, gt, preds_by_th, confidence, extent):
    """Current frame (+ prevs), GT, thresholded predictions and confidence."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scipy.ndimage import gaussian_filter

    frames = image if image.ndim == 3 else image[None]
    n_frames = frames.shape[0]
    ncols = n_frames + 2 + len(preds_by_th)
    fig = Figure(figsize=(3.2 * ncols, 8), dpi=110)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols)

    col = 0
    titles = ["Current"] + [f"Prev (t-{k})" for k in range(1, n_frames)]
    for k in range(n_frames):
        axes[col].imshow(frames[k], extent=extent, cmap="jet")
        axes[col].set_title(titles[k])
        col += 1
    axes[col].imshow(gt, extent=extent, vmin=0, vmax=1, cmap="binary")
    axes[col].set_title("True mask")
    col += 1
    for th, pred in preds_by_th.items():
        axes[col].imshow(pred, extent=extent, vmin=0, vmax=1, cmap="binary")
        axes[col].set_title(f"Pred (th={th})")
        col += 1
    axes[col].imshow(gaussian_filter(confidence.astype(np.float32), sigma=2),
                     extent=extent, vmin=0, vmax=1, cmap="viridis")
    axes[col].set_title("Confidence")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png)


def _resolve_aoi(args):
    """The AOI box for this scoring run, or None.

    Mirrors ``inference/scenes.py``'s resolver exactly -- the two must agree or
    the metrics cover different ground than the predictions.
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
    if (explicit or from_file) and args.legacy_south_half:
        raise SystemExit(
            "--legacy_south_half is the rule the AOI window replaced; passing both is "
            "contradictory. Use the window for new evaluations, the flag for archived ones."
        )
    return explicit if explicit is not None else from_file


def _resolve_tolerances(args):
    """[(ith, buffer), ...] to score under; the first is the primary.

    The primary keeps the historical top-level JSON keys, so a reader written
    against a single-tolerance file still finds what it expects.
    """
    if args.rth:
        return [DEFAULT_TOLERANCE, SOFT_TOLERANCE]
    tolerances = [(args.th, args.buffer)]
    for spec in args.extra_tolerance or []:
        try:
            ith, buf = spec.split(":")
            pair = (float(ith), int(buf))
        except ValueError:
            raise SystemExit(f"--extra_tolerance takes ITH:BUFFER, got {spec!r}")
        if pair not in tolerances:
            tolerances.append(pair)
    return tolerances


def _tolerance_key(ith, buffer) -> str:
    return f"ith{ith:g}_b{buffer:g}"


def main(args) -> None:
    from .reconstruct import canvas_shape  # noqa: F401  (kept import graph light)
    from ..training.evaluate import object_level_evaluate

    logging.basicConfig(level=logging.INFO)
    path = args.path

    tolerances = _resolve_tolerances(args)
    if args.rth:
        args.thresholds = list(RTH_THRESHOLDS)
        logging.info("--rth: scoring the paper's Reconstruction Thresholds "
                     f"{args.thresholds} under Intersection Tolerances {tolerances}. "
                     "These are tile-vote fractions -- valid ONLY if eval-scenes ran with "
                     "--recon_average vote.")
    logging.info(f"object-level tolerances (primary first): {tolerances}")
    patch_h, patch_w = args.patch_size
    aoi = _resolve_aoi(args)
    if aoi is not None:
        logging.info(f"scoring inside AOI lat {aoi[0]}-{aoi[1]}, lon {aoi[2]}-{aoi[3]}")
    elif args.legacy_south_half:
        logging.info("legacy South-half crop in force (pre-AOI behaviour)")
    intf_ids = sorted({
        m.group(0)
        for f in os.listdir(path)
        if f.endswith(".npy") and (m := INTF_ID_RE.search(f))
    })
    if not intf_ids:
        raise SystemExit(f"no <intf>_*.npy files under {path}")
    logging.info(f"{len(intf_ids)} interferograms under {path}")

    per_intf = {}
    per_intf_by_tol = {}
    for intf in intf_ids:
        meta = intf_meta(intf, args.intf_dict_path)
        x0a, y0a = aligned_origin(meta.frame)

        confidence = np.load(os.path.join(path, f"{intf}_pred.npy"), allow_pickle=True)
        gt = np.load(os.path.join(path, f"{intf}_gt.npy"), allow_pickle=True)

        # _image.npy is the (T, H, W) input stack -- ~3.7 GB per scene and by
        # far the largest thing eval-scenes writes (78% of an eval directory).
        # The METRICS NEVER READ IT: object_level_evaluate touches `image` only
        # to build per-object features, and this caller passes features=(), so
        # image=None produces byte-identical numbers. It is therefore loaded
        # only for --save_figures.
        #
        # That is what makes an archived eval directory prunable: drop the
        # input stacks and rescore.sh still re-thresholds in minutes, where
        # rebuilding them costs a 3 h run_eval.sh. See docs/OUTPUTS.md.
        image = None
        if args.save_figures:
            image_path = os.path.join(path, f"{intf}_image.npy")
            if not os.path.exists(image_path):
                raise SystemExit(
                    f"--save_figures needs {image_path}, which is not there.\n"
                    "Pruned eval directories keep their metrics and confidence "
                    "maps but not the input stacks. Either drop --save_figures "
                    "(metrics do not need it) or rebuild with run_eval.sh."
                )
            image = np.load(image_path, mmap_mode="r")

        preds = {th: np.where(confidence > th, 1, 0) for th in args.thresholds}

        # Crop the canvas to the ground this split owns, before scoring.
        #
        # This replaces a hard-coded rule that scored only the northern half of
        # a South-frame canvas -- a crude stand-in for "where the LiDAR is"
        # that had no counterpart in training and cut at an arbitrary line. The
        # AOI window does the same job explicitly, on both frames, from the same
        # numbers dataprep/dataset.py and inference/scenes.py use. The extent
        # comes from the confidence map, which carries the scene's (H, W)
        # exactly as gt and every frame of the input stack do.
        row_slice = None
        col_slice = None
        if aoi is not None:
            r0, r1, c0, c1 = grid_window(
                meta.frame, *aoi,
                patch_size=(patch_h, patch_w),
                stride=(patch_h // args.data_stride, patch_w // args.data_stride),
            )
            H_, W_ = confidence.shape
            # Grid cells -> canvas pixels: cell (r, c) starts at (r*step_y, c*step_x).
            step_y, step_x = patch_h // args.data_stride, patch_w // args.data_stride
            row_slice = slice(min(r0 * step_y, H_), min(r1 * step_y + patch_h - step_y, H_))
            col_slice = slice(min(c0 * step_x, W_), min(c1 * step_x + patch_w - step_x, W_))
        elif args.legacy_south_half and meta.frame == "South":
            row_slice = slice(0, confidence.shape[0] // 2)

        if row_slice is not None or col_slice is not None:
            rs = row_slice or slice(None)
            cs = col_slice or slice(None)
            confidence = confidence[rs, cs]
            gt = gt[rs, cs]
            preds = {th: p[rs, cs] for th, p in preds.items()}
            if image is not None:
                image = image[:, rs, cs] if image.ndim == 3 else image[rs, cs]
            logging.info(f"{intf}: scored on rows {rs}, cols {cs} of the canvas")

        results = {}
        tol_results = {_tolerance_key(*t): {} for t in tolerances}
        if not args.skip_ol_metrics:
            for th, pred in preds.items():
                for ith, buf in tolerances:
                    # round_ndigits=None: at scene scale the historical round(_, 2)
                    # quantises each scene BEFORE the area weighting below, which
                    # is coarser than the deltas this project compares.
                    r, p, gt_area, _ = object_level_evaluate(
                        gt[None], pred[None], None,
                        features=(), th=ith, buffer=buf, round_ndigits=None,
                    )
                    entry = {"recall": r, "precision": p, "gt_area": float(gt_area)}
                    tol_results[_tolerance_key(ith, buf)][str(th)] = entry
                    if (ith, buf) == tolerances[0]:
                        # The primary tolerance keeps the historical schema.
                        results[str(th)] = {"recall": r, "precision": p,
                                            "gt_area": float(gt_area)}
                        logging.info(f"{intf} @ th {th} [ith{ith:g}/b{buf:g}]: "
                                     f"recall={r:.4f} precision={p:.4f}")
        per_intf[intf] = results
        per_intf_by_tol[intf] = tol_results

        if args.save_figures:
            H, W = confidence.shape
            extent = [x0a, x0a + meta.dx * W, y0a - meta.dy * H, y0a]
            png = os.path.join(path, f"{intf}_overview.png")
            scene_figure(png, image, gt, preds, confidence, extent)
            logging.info(f"{intf}: figure -> {png}")

    def _aggregate(rows, th):
        """Unweighted and GT-area-weighted means over scenes at one threshold."""
        vals = [v[str(th)] for v in rows.values() if str(th) in v]
        if not vals:
            return {"mean_recall": None, "mean_precision": None,
                    "weighted_recall": None, "weighted_precision": None,
                    "gt_area": 0.0, "n_scenes": 0}
        rs = np.array([v["recall"] for v in vals], dtype=float)
        ps = np.array([v["precision"] for v in vals], dtype=float)
        ws = np.array([v.get("gt_area", 0.0) for v in vals], dtype=float)
        tot = float(ws.sum())
        return {
            # Unweighted: the historical aggregation, kept so the numbers in
            # docs/PREDICTIONS.md remain reproducible from a re-score.
            "mean_recall": float(rs.mean()),
            "mean_precision": float(ps.mean()),
            # GT-area weighted: the paper's aggregation. A scene holding three
            # polygons stops counting as much as one holding three hundred.
            "weighted_recall": float((rs * ws).sum() / tot) if tot else None,
            "weighted_precision": float((ps * ws).sum() / tot) if tot else None,
            "gt_area": tot,
            "n_scenes": int(len(vals)),
        }

    summary = {}
    summary_by_tol = {}
    if not args.skip_ol_metrics:
        for th in args.thresholds:
            summary[str(th)] = _aggregate(per_intf, th)
            a = summary[str(th)]
            logging.info(
                f"{'RTh' if args.rth else 'th'} {th}: "
                f"recall={a['mean_recall']:.4f} precision={a['mean_precision']:.4f} "
                f"| area-weighted recall={a['weighted_recall']:.4f} "
                f"precision={a['weighted_precision']:.4f}  (n={a['n_scenes']})"
            )
        for ith, buf in tolerances:
            key = _tolerance_key(ith, buf)
            rows = {i: v[key] for i, v in per_intf_by_tol.items() if key in v}
            summary_by_tol[key] = {str(th): _aggregate(rows, th)
                                   for th in args.thresholds}
        if len(tolerances) > 1:
            logging.info("--- by Intersection Tolerance (area-weighted) ---")
            for ith, buf in tolerances:
                key = _tolerance_key(ith, buf)
                for th in args.thresholds:
                    a = summary_by_tol[key][str(th)]
                    logging.info(f"  ith={ith:g} b={buf:g} th={th}: "
                                 f"recall={a['weighted_recall']:.4f} "
                                 f"precision={a['weighted_precision']:.4f}")

    out_json = args.out_json or os.path.join(
        path, f"olm_results_{datetime.now().strftime('%m%d%H%M')}.json"
    )
    with open(out_json, "w") as fh:
        json.dump({"per_intf": per_intf, "summary": summary,
                   "ol_th": tolerances[0][0], "buffer": tolerances[0][1],
                   "thresholds": list(args.thresholds),
                   # 'vote' means the thresholds above are the paper's RTh
                   # (tile-vote fractions); 'probability' means they cut a mean.
                   "threshold_kind": "vote" if args.rth else "probability",
                   "tolerances": [{"ol_th": t[0], "buffer": t[1],
                                   "key": _tolerance_key(*t)} for t in tolerances],
                   "per_intf_by_tolerance": per_intf_by_tol,
                   "summary_by_tolerance": summary_by_tol}, fh, indent=2)
    logging.info(f"metrics -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
