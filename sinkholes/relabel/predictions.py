"""``sinkholes relabel predictions`` -- a model's saved confidence maps as QGIS rasters.

For the second, non-blind review pass: each saved ``<intf>_pred.npy`` of one
evaluation directory is cut to exactly the pixel window of the scene's
labelling raster (``rasters/<intf>_int_aoi.tif``) and written as a GeoTIFF with
that raster's transform, so the two overlay pixel for pixel. Nothing is
re-run and the evaluation directory is only read.

The canvas eval-scenes saves starts at the raw pixel (row_off, col_off) the
frame crop lands on, so canvas pixel (r, c) is raw pixel (row_off + r,
col_off + c); the labelling raster records its raw window in its tags. The
crop is therefore an integer slice -- and it is *verified*, not assumed: the
saved ``_gt.npy`` cut the same way must be positive exactly where the original
polygons rasterise on the labelling raster's own grid. Pixels of the window the
canvas does not cover (a few trailing rows) are nodata.

Output: ``predictions/<model>/<intf>_conf.tif`` + ``predictions/<model>/model.json``.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from .common import (
    list_eval_intfs,
    now_iso,
    read_eval_args,
    read_original_gt,
    scene_geometry,
    write_json,
    ws_path,
)

#: The reference model for the second pass (docs/RESULTS.md section 3: best
#: generation-4 temporal k5 model, and it covers all 20 official scenes).
DEFAULT_EVAL_DIR = ("outputs/predictions/th350_tattn_ctx50_neg10_2026-09-10_01h16_lsf_816217/"
                    "best/scenes_temporal_th350_rth_09_14_12h21")
NODATA = -1.0


def add_arguments(p):
    from ..paths import REPO_ROOT

    p.add_argument("--workspace", required=True)
    p.add_argument("--data_dir", required=True, help="holds sub_20260701.shp (for the check)")
    p.add_argument("--eval_dir", default=str(REPO_ROOT / DEFAULT_EVAL_DIR),
                   help="saved eval-scenes directory (read-only)")
    p.add_argument("--name", default=None,
                   help="layer/folder name (default: the model directory name, shortened)")


def window_of(tif):
    import rasterio

    with rasterio.open(tif) as src:
        t = src.tags()
        r0, r1 = map(int, t["window_rows"].split(":"))
        c0, c1 = map(int, t["window_cols"].split(":"))
        return (r0, r1, c0, c1), src.profile.copy(), src.transform


def crop_canvas(canvas, geo, window, fill=NODATA):
    """The raw-window part of an eval-scenes canvas; ``fill`` where it has none."""
    r0, r1, c0, c1 = window
    out = np.full((r1 - r0, c1 - c0), fill, dtype=np.float32)
    cr0, cr1 = r0 - geo.row_off, r1 - geo.row_off      # canvas rows of the window
    cc0, cc1 = c0 - geo.col_off, c1 - geo.col_off
    H, W = canvas.shape
    sr0, sr1 = max(cr0, 0), min(cr1, H)
    sc0, sc1 = max(cc0, 0), min(cc1, W)
    if sr0 < sr1 and sc0 < sc1:
        out[sr0 - cr0:sr1 - cr0, sc0 - cc0:sc1 - cc0] = canvas[sr0:sr1, sc0:sc1]
    return out


def verify_alignment(intf, src_dir, geo, window, transform, polygons, patch, stride):
    """Saved GT, cut like the prediction, vs the polygons burnt on the window's grid.

    Inside the patch grid every canvas pixel is covered by at least one tile, so
    the saved GT is > 0 exactly where the original mask is 1. A one-pixel shift
    in the crop, origin or transform breaks the equality.
    """
    from rasterio.features import geometry_mask

    gt = np.load(os.path.join(src_dir, f"{intf}_gt.npy"), mmap_mode="r")
    cut = crop_canvas(np.asarray(gt), geo, window, fill=np.nan)
    r0, r1, c0, c1 = window
    burnt = (geometry_mask(polygons, transform=transform, invert=True, out_shape=(r1 - r0, c1 - c0))
             if polygons else np.zeros((r1 - r0, c1 - c0), bool))
    # restrict to the patch grid's extent (rows/cols tiles cover) on the canvas
    ph, pw = patch
    H, W = gt.shape
    grid_h, grid_w = H - ph // stride, W - pw // stride      # last stride-step is uncovered
    grid_w = min(grid_w, 4500)                               # the column crop
    rr = np.arange(r0, r1) - geo.row_off
    cc = np.arange(c0, c1) - geo.col_off
    inside = ((rr >= 0) & (rr < grid_h))[:, None] & ((cc >= 0) & (cc < grid_w))[None, :]
    ok = np.array_equal((cut > 0)[inside], burnt[inside])
    n_pos = int(burnt[inside].sum())
    return ok, n_pos


def main(args):
    import rasterio

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ws = Path(args.workspace)
    src = os.path.abspath(args.eval_dir)
    a = read_eval_args(src)
    model = Path(src).parts[-3]
    name = args.name or model.split("_2026")[0]
    out = ws / "predictions" / name
    out.mkdir(parents=True, exist_ok=True)
    patch, stride = tuple(a["patch_size"]), int(a["data_stride"])
    kind = "vote fraction (RTh)" if a.get("recon_average") == "vote" else "mean probability"

    import csv

    with open(ws_path(ws, "scenes_csv")) as fh:
        scenes = {r["intf_id"]: r for r in csv.DictReader(fh)}
    intfs = [i for i in list_eval_intfs(src) if i in scenes]
    missing = sorted(i for i, r in scenes.items() if r["role"] == "official_test" and i not in intfs)
    gt = read_original_gt(args.data_dir, intfs)
    report = {}
    for intf in intfs:
        tif = ws / scenes[intf]["raster"]
        window, profile, transform = window_of(tif)
        geo = scene_geometry(intf)
        polys = list(gt[gt["intf_id"] == intf].geometry)
        ok, n_pos = verify_alignment(intf, src, geo, window, transform, polys, patch, stride)
        if not ok:
            raise SystemExit(f"{intf}: saved GT and the labelling raster disagree after the crop "
                             f"-- the prediction would not overlay correctly; nothing written")
        conf = np.load(os.path.join(src, f"{intf}_pred.npy"), mmap_mode="r")
        cut = crop_canvas(np.asarray(conf), geo, window)
        dst = out / f"{intf}_conf.tif"
        tmp = dst.with_suffix(".tmp.tif")
        profile.update(dtype="float32", nodata=NODATA, count=1, compress="deflate", predictor=3,
                       tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(tmp, "w", **profile) as d:
            d.write(cut, 1)
            d.update_tags(intf_id=intf, model=model, source=os.path.join(src, f"{intf}_pred.npy"),
                          value=kind, window_rows=f"{window[0]}:{window[1]}",
                          window_cols=f"{window[2]}:{window[3]}")
            from rasterio.enums import Resampling

            d.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
        with rasterio.open(tmp) as chk, rasterio.open(tif) as ref:
            assert chk.transform == ref.transform and chk.shape == ref.shape and chk.crs == ref.crs
            assert np.array_equal(chk.read(1), cut)
        os.replace(tmp, dst)
        cov = float((cut != NODATA).mean())
        report[intf] = {"gt_alignment_verified": ok, "gt_pixels_checked": n_pos,
                        "window": list(window), "canvas_coverage_of_window": round(cov, 4),
                        "max_confidence": float(cut.max())}
        logging.info(f"{intf}: aligned (GT check on {n_pos} px), -> {dst.name}")
    write_json(out / "model.json", {
        "created": now_iso(), "model": model, "name": name, "eval_dir": src,
        "confidence": kind, "eval_scenes_args": a, "scenes": report,
        "official_scenes_without_prediction": missing,
        "note": "derived, read-only copy; the evaluation directory was not modified"})
    if missing:
        logging.warning(f"no prediction for official scene(s): {missing}")
    logging.info(f"-> {out}  (rebuild the QGIS project to add the layers)")
