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

#: Operating points the confidence map is re-thresholded at. The upper end
#: matters: models trained with --nonz_only (the default) never see a
#: background-only patch, so they over-predict at scene scale and the useful
#: operating point sits well above the 0.25 that eval-scenes thresholds at.
THRESHOLDS = (0.125, 0.25, 0.5, 0.7, 0.9)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--path", type=str, required=True,
                   help="an eval-scenes output directory (holds <intf>_*.npy)")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS),
                   metavar="T", help="confidence operating points to score at")
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

        # The southern half of the South frame is open water/desert with no
        # LiDAR coverage; metrics run on the populated northern half only.
        # The crop extent comes from the confidence map, which carries the
        # scene's (H, W) exactly as gt and every frame of the input stack do --
        # previously this was read off `image`, which is no longer loaded.
        if meta.frame == "South":
            half = confidence.shape[0] // 2
            confidence = confidence[:half]
            gt = gt[:half]
            preds = {th: p[:half] for th, p in preds.items()}
            if image is not None:
                image = image[:, :half] if image.ndim == 3 else image[:half]

        results = {}
        if not args.skip_ol_metrics:
            for th, pred in preds.items():
                r, p, _, _ = object_level_evaluate(
                    gt[None], pred[None], None,
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
        for th in args.thresholds:
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
