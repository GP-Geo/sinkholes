#!/usr/bin/env python3
"""Derive a shoreline AOI by bounding the positive patches, and price it.

Backs the measured tables in docs/PLAN_CLEAN_BENCHMARK.md §10. Reads only
``nonz_indices.json`` and ``assets/intf_coord.json`` -- no rasters, no GPU.

The question it answers: an AOI trades positives away for scored background.
Since scene-level precision is dominated by background that has no polygons on
it (§1), the metric that matters is *positive density* -- positives per scored
grid cell -- not the raw positive count.

A patch counts as inside the AOI only when its whole footprint is, which is the
same rule the latitude cut of §4 uses.

Usage:
    python scripts/data/measure_aoi.py [--patches_dir DIR] [--years 2019 2022]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sinkholes.geo import FRAME_ORIGINS, PIXEL_DEG  # noqa: E402

PATCH_H, PATCH_W = 200, 100
STRIDE_H, STRIDE_W = PATCH_H // 2, PATCH_W // 2      # stride-2 tree
N_COL = 89

#: Grid height varies per scene (191-200 rows): alignment crops every scene to a
#: common origin, not a common height. These medians are only for normalising the
#: "cells kept" percentages; every window is defined in latitude, never in rows.
MEDIAN_ROWS = {"North": 192, "South": 195}

CANDIDATES = [
    ("none", -90.0, 90.0, -180.0, 180.0),
    ("31.23-31.76 / 35.375-35.47", 31.23, 31.76, 35.375, 35.47),
    ("31.25-31.75 / 35.38-35.46", 31.25, 31.75, 35.38, 35.46),
    ("31.25-31.75 / 35.38-35.45", 31.25, 31.75, 35.38, 35.45),
    ("31.30-31.75 / 35.38-35.45", 31.30, 31.75, 35.38, 35.45),
]


def masks(frame, lat_lo, lat_hi, lon_lo, lon_hi):
    """(row_inside, col_inside) for patches wholly within the AOI."""
    lon0, lat0 = FRAME_ORIGINS[frame]
    r = np.arange(MEDIAN_ROWS[frame])
    c = np.arange(N_COL)
    top = lat0 - r * STRIDE_H * PIXEL_DEG
    bottom = lat0 - (r * STRIDE_H + PATCH_H) * PIXEL_DEG
    left = lon0 + c * STRIDE_W * PIXEL_DEG
    right = lon0 + (c * STRIDE_W + PATCH_W) * PIXEL_DEG
    return (bottom >= lat_lo) & (top <= lat_hi), (left >= lon_lo) & (right <= lon_hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches_dir", default=os.environ.get(
        "PATCHES_DIR",
        "/Volumes/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches"))
    ap.add_argument("--intf_dict", default="assets/intf_coord.json")
    ap.add_argument("--years", nargs=2, type=int, default=[2019, 2022])
    ap.add_argument("--cut_lat", type=float, default=31.4,
                    help="the §4 geo cut; train = North above it, hold-out = South below")
    ap.add_argument("--by_year", action="store_true",
                    help="per-year table: does the AOI hold outside the years it was fitted on?")
    ap.add_argument("--aoi", nargs=4, type=float, default=[31.25, 31.75, 35.38, 35.46],
                    metavar=("LAT_LO", "LAT_HI", "LON_LO", "LON_HI"),
                    help="the AOI used by --by_year")
    args = ap.parse_args()

    nonz_path = os.path.join(
        args.patches_dir, "data_patches_H200_W100_strpp2_11days_Aligned", "nonz_indices.json")
    with open(nonz_path) as fh:
        nonz = json.load(fh)
    with open(args.intf_dict) as fh:
        coord = json.load(fh)

    y0, y1 = args.years
    per_frame, n_intf = {}, {}
    for intf, meta in coord.items():
        if not (y0 <= int(intf[:4]) <= y1) or not nonz.get(intf):
            continue
        per_frame.setdefault(meta["frame"], []).append(np.asarray(nonz[intf], dtype=int))
        n_intf[meta["frame"]] = n_intf.get(meta["frame"], 0) + 1

    print(f"Positive patches, {y0}-{y1}, stride-2 grid\n")
    rc = {f: np.concatenate(v, axis=0) for f, v in per_frame.items()}
    for frame in ("North", "South"):
        lon0, lat0 = FRAME_ORIGINS[frame]
        rows, cols = rc[frame][:, 0], rc[frame][:, 1]
        print(f"  {frame}: {n_intf[frame]} intfs, {len(rows):,} positives  "
              f"lat {lat0 - (rows.max() * STRIDE_H + PATCH_H) * PIXEL_DEG:.4f}"
              f"-{lat0 - rows.min() * STRIDE_H * PIXEL_DEG:.4f}  "
              f"lon {lon0 + cols.min() * STRIDE_W * PIXEL_DEG:.4f}"
              f"-{lon0 + (cols.max() * STRIDE_W + PATCH_W) * PIXEL_DEG:.4f}")

    print(f"\nGeo split under each AOI (train = North above {args.cut_lat}, "
          f"hold-out = South below)\n")
    hdr = f"{'AOI':>28} | {'train pos':>10} {'N cells':>8} | {'holdout':>9} {'S cells':>8} | {'density N/S':>13}"
    print(hdr + "\n" + "-" * len(hdr))
    for name, la, lb, lc, ld in CANDIDATES:
        out = {}
        for frame, above in (("North", True), ("South", False)):
            rm, cm = masks(frame, la, lb, lc, ld)
            lat0 = FRAME_ORIGINS[frame][1]
            r = np.arange(MEDIAN_ROWS[frame])
            edge = (lat0 - (r * STRIDE_H + PATCH_H) * PIXEL_DEG >= args.cut_lat) if above \
                else (lat0 - r * STRIDE_H * PIXEL_DEG <= args.cut_lat)
            band = rm & edge
            rows = np.clip(rc[frame][:, 0], 0, MEDIAN_ROWS[frame] - 1)
            out[frame] = (int((band[rows] & cm[rc[frame][:, 1]]).sum()),
                          int(band.sum() * cm.sum()),
                          n_intf[frame])
        (tp, tc, tn), (hp, hc, hn) = out["North"], out["South"]
        print(f"{name:>28} | {tp:10,} {tc:8,} | {hp:9,} {hc:8,} | "
              f"{tp / (tc * tn):.4f}/{hp / (hc * hn):.4f}")

    if args.by_year:
        by_year(coord, nonz, args)


def by_year(coord, nonz, args):
    """Positives inside the AOI, year by year. The AOI was fitted on 2019-2022;
    this is what says whether it transfers to the years that were excluded."""
    la, lb, lc, ld = args.aoi
    print(f"\n\nPer-year: positives inside AOI lat {la}-{lb} / lon {lc}-{ld}\n")
    hdr = (f"{'year':>6} | {'intfs':>5} {'labelled':>8} | {'positives':>10} {'inside':>10} "
           f"{'kept':>6} | {'lat span':>17} {'lon span':>17}")
    print(hdr + "\n" + "-" * len(hdr))

    masks_by_frame = {f: masks(f, la, lb, lc, ld) for f in FRAME_ORIGINS}
    years = sorted({i[:4] for i in coord})
    for y in years:
        tot = ins = n_lab = n_intf = 0
        lats, lons = [], []
        for intf, meta in coord.items():
            if intf[:4] != y:
                continue
            n_intf += 1
            idx = nonz.get(intf)
            if not idx:
                continue
            n_lab += 1
            frame = meta["frame"]
            rm, cm = masks_by_frame[frame]
            lon0, lat0 = FRAME_ORIGINS[frame]
            rc_a = np.asarray(idx, dtype=int)
            rows = np.clip(rc_a[:, 0], 0, MEDIAN_ROWS[frame] - 1)
            tot += len(rc_a)
            ins += int((rm[rows] & cm[rc_a[:, 1]]).sum())
            lats += [lat0 - (rc_a[:, 0].max() * STRIDE_H + PATCH_H) * PIXEL_DEG,
                     lat0 - rc_a[:, 0].min() * STRIDE_H * PIXEL_DEG]
            lons += [lon0 + rc_a[:, 1].min() * STRIDE_W * PIXEL_DEG,
                     lon0 + (rc_a[:, 1].max() * STRIDE_W + PATCH_W) * PIXEL_DEG]
        span = (f"{min(lats):.4f}-{max(lats):.4f}", f"{min(lons):.4f}-{max(lons):.4f}") \
            if lats else ("-", "-")
        pct = f"{100 * ins / tot:.1f}%" if tot else "-"
        print(f"{y:>6} | {n_intf:5d} {n_lab:8d} | {tot:10,} {ins:10,} {pct:>6} | "
              f"{span[0]:>17} {span[1]:>17}")


if __name__ == "__main__":
    main()
