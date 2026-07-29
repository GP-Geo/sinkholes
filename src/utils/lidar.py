#!/usr/bin/env python3
# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import math
import geopandas as gpd
import matplotlib.pyplot as plt

# ==== EDIT THESE ====
SHP_PATH   = _bootstrap.asset("lidar_mask_polygs.shp")
GROUPBY    = "source"      # column to split panels by
MAX_COLS   = 4             # max panels per row
SAVE_PATH  = None          # e.g. "masks.png" or None to just show
# ====================

gdf = gpd.read_file(SHP_PATH)
groups = list(gdf.groupby(GROUPBY, dropna=False)) if GROUPBY in gdf.columns else [("all", gdf)]
if not groups: raise SystemExit("No features found.")

n = len(groups)
ncols = min(MAX_COLS, n)
nrows = math.ceil(n / ncols)

# Common extent for all panels
minx, miny, maxx, maxy = gdf.total_bounds
dx, dy = (maxx - minx) * 0.05, (maxy - miny) * 0.05

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
for idx, (name, sub) in enumerate(groups):
    r, c = divmod(idx, ncols)
    ax = axes[r][c]
    sub.plot(ax=ax, facecolor="#ff9999", edgecolor="black", linewidth=0.6, alpha=0.5)
    ax.set_title(str(name))
    ax.set_xlim(minx - dx, maxx + dx); ax.set_ylim(miny - dy, maxy + dy)
    ax.set_aspect("equal")

# Hide any unused axes
for k in range(n, nrows*ncols):
    r, c = divmod(k, ncols)
    axes[r][c].axis("off")

fig.tight_layout()
if SAVE_PATH: fig.savefig(SAVE_PATH, dpi=300)
else: plt.show()
