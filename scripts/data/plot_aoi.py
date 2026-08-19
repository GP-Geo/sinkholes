#!/usr/bin/env python3
"""Draw the AOI box on real interferograms, one per frame, for visual inspection.

Renders the aligned canvas of a randomly chosen scene from each frame at reduced
resolution, overlays the AOI rectangle from docs/PLAN_CLEAN_BENCHMARK.md §10, the
31.4 deg geo cut, and the scene's positive patches, with lon/lat axes.

    python scripts/data/plot_aoi.py --seed 0 --out_dir outputs/aoi_preview
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sinkholes.geo import FRAME_ORIGINS, PIXEL_DEG  # noqa: E402
from sinkholes.paths import asset as asset_path  # noqa: E402

PATCH_H, PATCH_W = 200, 100
STRIDE_H, STRIDE_W = PATCH_H // 2, PATCH_W // 2
DS = 10                                    # pixel downsample inside each patch


def build_canvas(arr):
    """Tile the non-overlapping top-left STRIDE_H x STRIDE_W of each patch."""
    ny, nx = arr.shape[:2]
    tile = arr[:, :, :STRIDE_H:DS, :STRIDE_W:DS]        # (ny, nx, h, w)
    h, w = tile.shape[2], tile.shape[3]
    return tile.transpose(0, 2, 1, 3).reshape(ny * h, nx * w), h, w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches_dir", default=os.environ.get(
        "PATCHES_DIR",
        "/Volumes/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches"))
    ap.add_argument("--intf_dict", default="assets/intf_coord.json")
    ap.add_argument("--aoi", nargs=4, type=float, default=[31.25, 31.75, 35.38, 35.46],
                    metavar=("LAT_LO", "LAT_HI", "LON_LO", "LON_HI"))
    ap.add_argument("--cut_lat", type=float, default=31.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_scenes", type=int, default=1, help="scenes to draw per frame")
    ap.add_argument("--no_lidar", action="store_true", help="skip the LiDAR gate overlay")
    ap.add_argument("--out_dir", default="outputs/aoi_preview")
    args = ap.parse_args()
    la, lb, lc, ld = args.aoi
    os.makedirs(args.out_dir, exist_ok=True)

    tree = os.path.join(args.patches_dir, "data_patches_H200_W100_strpp2_11days_Aligned")
    coord = json.load(open(args.intf_dict))
    nonz = json.load(open(os.path.join(tree, "nonz_indices.json")))

    rng = random.Random(args.seed)
    picks = []
    for frame in ("North", "South"):
        cands = sorted(i for i, m in coord.items()
                       if m["frame"] == frame and nonz.get(i)
                       and os.path.exists(os.path.join(
                           tree, f"data_patches_{i}_H200_W100_strpp2.npy")))
        picks += [(frame, i) for i in rng.sample(cands, min(args.n_scenes, len(cands)))]

    lidar = None
    if not args.no_lidar:
        import geopandas as gpd
        lidar = gpd.read_file(asset_path("lidar_mask_polygs.shp")).set_crs("EPSG:4326")

    for frame, intf in picks:
        lon0, lat0 = FRAME_ORIGINS[frame]
        arr = np.load(os.path.join(tree, f"data_patches_{intf}_H200_W100_strpp2.npy"),
                      mmap_mode="r")
        canvas, h, w = build_canvas(np.asarray(arr))
        ny, nx = arr.shape[:2]

        # geographic extent of the tiled canvas
        lat_top, lat_bot = lat0, lat0 - ny * STRIDE_H * PIXEL_DEG
        lon_left, lon_right = lon0, lon0 + nx * STRIDE_W * PIXEL_DEG
        extent = [lon_left, lon_right, lat_bot, lat_top]

        disp = np.ma.masked_where(canvas == 0, canvas)
        v = np.percentile(disp.compressed(), [2, 98]) if disp.count() else (-np.pi, np.pi)

        fig, ax = plt.subplots(figsize=(7, 15))
        ax.imshow(disp, extent=extent, origin="upper", cmap="jet",
                  vmin=v[0], vmax=v[1], aspect="auto", interpolation="nearest")

        # positives of this scene
        rc = np.asarray(nonz[intf], dtype=int)
        plat = lat0 - (rc[:, 0] * STRIDE_H + PATCH_H / 2) * PIXEL_DEG
        plon = lon0 + (rc[:, 1] * STRIDE_W + PATCH_W / 2) * PIXEL_DEG
        ax.scatter(plon, plat, s=3, c="black", marker="s", linewidths=0,
                   label=f"positive patches ({len(rc)})")

        # AOI, clipped to this canvas
        ax.add_patch(Rectangle((lc, la), ld - lc, lb - la, fill=False,
                               edgecolor="white", linewidth=2.5, zorder=5))
        ax.add_patch(Rectangle((lc, la), ld - lc, lb - la, fill=False,
                               edgecolor="black", linewidth=1.0, linestyle="--", zorder=6,
                               label="AOI"))
        ax.axhline(args.cut_lat, color="magenta", lw=1.5, ls=":",
                   label=f"geo cut {args.cut_lat}")

        # LiDAR gate: what evaluation actually scores today
        if lidar is not None:
            first = True
            for geom in lidar.geometry:
                for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
                    xs, ys = poly.exterior.xy
                    ax.plot(xs, ys, color="lime", lw=0.8, zorder=4,
                            label="LiDAR gate" if first else None)
                    first = False

        inside = ((plat >= la) & (plat <= lb) & (plon >= lc) & (plon <= ld)).sum()
        ax.set_xlim(lon_left, lon_right)
        ax.set_ylim(lat_bot, lat_top)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"{frame} frame — {intf}\n"
                     f"canvas lat {lat_bot:.3f}–{lat_top:.3f}, lon {lon_left:.3f}–{lon_right:.3f}\n"
                     f"AOI holds {inside}/{len(rc)} positives ({100*inside/len(rc):.0f}%)",
                     fontsize=10)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        out = os.path.join(args.out_dir, f"aoi_{frame.lower()}_{intf}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"{frame:>6}  {intf}  grid {arr.shape[:2]}  ->  {out}")


if __name__ == "__main__":
    main()
