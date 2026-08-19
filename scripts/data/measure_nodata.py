#!/usr/bin/env python3
"""Per-cell nodata (zero-phase) fraction, by year and frame, inside vs outside an AOI.

Companion to measure_aoi.py. That script bounds the *labelled* ground; this one
measures the *data*, which is what the AOI is supposed to be protecting against.

Nodata is ``phase == 0`` -- the same signal ``clean_patches`` uses to drop a
polygon (``zero_frac > 0.7``) and ``--treat_nodata_regions`` turns into a
validity channel. Each patch is subsampled on a coarse lattice; the fraction is
a per-cell estimate, not an exact count.

    python scripts/data/measure_nodata.py --years 2019 2026
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sinkholes.geo import FRAME_ORIGINS, PIXEL_DEG  # noqa: E402

PATCH_H, PATCH_W = 200, 100
STRIDE_H, STRIDE_W = PATCH_H // 2, PATCH_W // 2
STEP = 25                       # subsample lattice inside each patch (8 x 4 samples)
#: Sample every Nth grid cell in each direction. Reading every cell touches
#: nearly every page of a 1.37 GB scene file, so the scan is bound by ~600 GB of
#: SMB traffic; at 3 it is ~9x less for the same broad spatial picture.
GRID_STEP = 3
ID_RE = re.compile(r"(\d{8}_\d{8})")


def aoi_masks(frame, n_rows, lat_lo, lat_hi, lon_lo, lon_hi):
    lon0, lat0 = FRAME_ORIGINS[frame]
    r, c = np.arange(n_rows), np.arange(89)
    top, bottom = lat0 - r * STRIDE_H * PIXEL_DEG, lat0 - (r * STRIDE_H + PATCH_H) * PIXEL_DEG
    left, right = lon0 + c * STRIDE_W * PIXEL_DEG, lon0 + (c * STRIDE_W + PATCH_W) * PIXEL_DEG
    return (bottom >= lat_lo) & (top <= lat_hi), (left >= lon_lo) & (right <= lon_hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches_dir", default=os.environ.get(
        "PATCHES_DIR",
        "/Volumes/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches"))
    ap.add_argument("--intf_dict", default="assets/intf_coord.json")
    ap.add_argument("--years", nargs=2, type=int, default=[2019, 2026])
    ap.add_argument("--aoi", nargs=4, type=float, default=[31.25, 31.75, 35.38, 35.46],
                    metavar=("LAT_LO", "LAT_HI", "LON_LO", "LON_HI"))
    args = ap.parse_args()
    la, lb, lc, ld = args.aoi

    tree = os.path.join(args.patches_dir, "data_patches_H200_W100_strpp2_11days_Aligned")
    with open(args.intf_dict) as fh:
        coord = json.load(fh)

    files = sorted(f for f in os.listdir(tree)
                   if f.endswith(".npy") and "nonz" not in f and "indices" not in f)
    y0, y1 = args.years

    # (year, frame) -> [zero_in, n_in, zero_out, n_out, n_scenes, grid_rows]
    acc = {}
    t0, done = time.time(), 0
    for n, fn in enumerate(files, 1):
        m = ID_RE.search(fn)
        if not m or m.group(1) not in coord:
            continue
        intf = m.group(1)
        year = int(intf[:4])
        if not (y0 <= year <= y1):
            continue
        frame = coord[intf]["frame"]
        try:
            arr = np.load(os.path.join(tree, fn), mmap_mode="r")
            if arr.ndim != 4:
                continue
            sub = np.asarray(arr[::GRID_STEP, ::GRID_STEP, ::STEP, ::STEP])
        except Exception as e:                                   # noqa: BLE001
            print(f"  skip {intf}: {e}", file=sys.stderr)
            continue

        zero = (sub == 0)                                        # (R', C', s, s)
        rm, cm = aoi_masks(frame, arr.shape[0], la, lb, lc, ld)
        rm, cm = rm[::GRID_STEP], cm[::GRID_STEP]                # match the sampled cells
        inside = rm[:, None] & cm[None, :]                       # (R', C')
        per_cell = zero.mean(axis=(2, 3))                        # (R', C')

        k = (year, frame)
        a = acc.setdefault(k, [0.0, 0, 0.0, 0, 0])
        a[0] += float(per_cell[inside].sum()); a[1] += int(inside.sum())
        a[2] += float(per_cell[~inside].sum()); a[3] += int((~inside).sum())
        a[4] += 1
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"  ... {done} scenes in {el/60:.1f} min "
                  f"({el/done:.1f}s each, ~{(len(files)-n)*el/done/60:.0f} min left)",
                  file=sys.stderr, flush=True)

    print(f"\nNodata (phase == 0) fraction, AOI lat {la}-{lb} / lon {lc}-{ld}")
    print("'inside' is the AOI, 'outside' is the rest of the scored canvas.\n")
    hdr = (f"{'year':>6} {'frame':>6} {'scenes':>7} | {'inside':>8} {'outside':>8} "
           f"{'whole':>8} | {'ratio out/in':>12}")
    print(hdr + "\n" + "-" * len(hdr))
    for (year, frame) in sorted(acc):
        zi, ni, zo, no, ns = acc[(year, frame)]
        fi, fo = zi / max(ni, 1), zo / max(no, 1)
        fw = (zi + zo) / max(ni + no, 1)
        print(f"{year:>6} {frame:>6} {ns:7d} | {fi:8.4f} {fo:8.4f} {fw:8.4f} | "
              f"{fo / fi if fi else float('nan'):12.2f}")


if __name__ == "__main__":
    main()
