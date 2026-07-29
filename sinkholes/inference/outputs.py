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

from ..geo import aligned_origin
from ..meta import INTF_ID_RE, intf_meta

#: Operating points the confidence map is re-thresholded at.
THRESHOLDS = (0.125, 0.25, 0.5)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--path", type=str, required=True,
                   help="an eval-scenes output directory (holds <intf>_*.npy)")
    p.add_argument("--th", type=float, default=0.7, help="object-level overlap threshold")
    p.add_argument("--buffer", type=int, default=5, help="object-matching buffer, pixels")
    p.add_argument("--skip_ol_metrics", action="store_true")
    p.add_argument("--save_figures", action="store_true",
                   help="write a per-interferogram overview PNG next to the metrics")
    p.add_argument("--out_json", type=str, default=None,
                   help="metrics output (default: <path>/olm_results_<ts>.json)")
    p.add_argument("--intf_dict_path", type=str, default=None)


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


def main(args) -> None:
    from .reconstruct import canvas_shape  # noqa: F401  (kept import graph light)
    from ..training.evaluate import object_level_evaluate

    logging.basicConfig(level=logging.INFO)
    path = args.path
    intf_ids = sorted({
        m.group(0)
        for f in os.listdir(path)
        if f.endswith(".npy") and (m := INTF_ID_RE.search(f))
    })
    if not intf_ids:
        raise SystemExit(f"no <intf>_*.npy files under {path}")
    logging.info(f"{len(intf_ids)} interferograms under {path}")

    per_intf = {}
    for intf in intf_ids:
        meta = intf_meta(intf, args.intf_dict_path)
        x0a, y0a = aligned_origin(meta.frame)

        image = np.load(os.path.join(path, f"{intf}_image.npy"), allow_pickle=True)
        confidence = np.load(os.path.join(path, f"{intf}_pred.npy"), allow_pickle=True)
        gt = np.load(os.path.join(path, f"{intf}_gt.npy"), allow_pickle=True)
        preds = {th: np.where(confidence > th, 1, 0) for th in THRESHOLDS}

        # The southern half of the South frame is open water/desert with no
        # LiDAR coverage; metrics run on the populated northern half only.
        if meta.frame == "South":
            half = (image.shape[1] if image.ndim == 3 else image.shape[0]) // 2
            image = image[:, :half] if image.ndim == 3 else image[:half]
            confidence = confidence[:half]
            gt = gt[:half]
            preds = {th: p[:half] for th, p in preds.items()}

        results = {}
        if not args.skip_ol_metrics:
            for th, pred in preds.items():
                r, p, _, _ = object_level_evaluate(
                    gt[None], pred[None], image[None] if image.ndim == 2 else image[0][None],
                    features=(), th=args.th, buffer=args.buffer,
                )
                results[str(th)] = {"recall": r, "precision": p}
                logging.info(f"{intf} @ recon_th {th}: recall={r} precision={p}")
        per_intf[intf] = results

        if args.save_figures:
            H, W = confidence.shape
            extent = [x0a, x0a + meta.dx * W, y0a - meta.dy * H, y0a]
            png = os.path.join(path, f"{intf}_overview.png")
            scene_figure(png, image, gt, preds, confidence, extent)
            logging.info(f"{intf}: figure -> {png}")

    summary = {}
    if not args.skip_ol_metrics:
        for th in THRESHOLDS:
            rs = [v[str(th)]["recall"] for v in per_intf.values() if str(th) in v]
            ps = [v[str(th)]["precision"] for v in per_intf.values() if str(th) in v]
            summary[str(th)] = {
                "mean_recall": float(np.mean(rs)) if rs else None,
                "mean_precision": float(np.mean(ps)) if ps else None,
            }
            logging.info(f"mean @ recon_th {th}: recall={summary[str(th)]['mean_recall']} "
                         f"precision={summary[str(th)]['mean_precision']}")

    out_json = args.out_json or os.path.join(
        path, f"olm_results_{datetime.now().strftime('%m%d%H%M')}.json"
    )
    with open(out_json, "w") as fh:
        json.dump({"per_intf": per_intf, "summary": summary,
                   "ol_th": args.th, "buffer": args.buffer}, fh, indent=2)
    logging.info(f"metrics -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
