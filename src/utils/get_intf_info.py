# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import argparse
import json
def get_intf_coords(intf_name):
    intf_dict_file = open(_bootstrap.asset('intf_coord.json'), 'r')
    intf_coords = json.load(intf_dict_file)
    x0 = intf_coords[intf_name]['east']
    y0 = intf_coords[intf_name]['north']
    dx = intf_coords[intf_name]['dx']
    dy = intf_coords[intf_name]['dy']
    nlines = intf_coords[intf_name]['nlines']
    ncells = intf_coords[intf_name]['ncells']
    lidar_mask = intf_coords[intf_name]['lidar_mask']
    num_nonz_p = intf_coords[intf_name]['nonz_num']
    bo = intf_coords[intf_name]['byte_order']
    frame = intf_coords[intf_name]['frame']

    x4000 = x0 + 4000*dx
    x8500 = x4000 + 4500*dx

    return (x0, y0, dx, dy,ncells, nlines, x4000, x8500,lidar_mask,num_nonz_p,bo,frame)
def get_intf_lidar_mask(intf_name):
    with open(_bootstrap.asset('lidar_intf_mask.txt'), 'r') as f:
        mask = 'no_mask'
        for line in f:
            if intf_name[:8] == line[8:16] and intf_name[9:17] == line[24:32]:
                mask = line[40:49]
            elif intf_name[:8] == line[8:16]:
                mask = line[40:49]


    return mask

import numpy as np
def crop_to_start_xy(intf: np.ndarray,
                     mask: np.ndarray,
                     x0: float, y0: float,          # source top-left (easting, northing)
                     x_star: float, y_star: float,  # desired top-left to crop to
                     dx: float = 2.777e-05,
                     dy: float = 2.777e-05,
                     tol: float = 1e-2):
    """
    Crop intf and mask so that the output starts at (x_star, y_star).
    Optionally crop to a specific (width, height). If width/height are None,
    returns everything from that start to the image edge.

    Coordinates follow rasterio.from_origin(x0, y0, dx, dy):
      - x increases to the RIGHT
      - y increases DOWNWARD in array indices; georeferenced 'northing' decreases by dy per row.

    Args:
      intf, mask: 2D arrays with identical shape (H, W)
      x0, y0:   source top-left easting/northing
      x_star, y_star: desired new top-left easting/northing
      dx, dy:   pixel size in degrees (or meters), positive
      width, height: optional size of the crop in pixels
      tol:      tolerance for sub-pixel rounding (in coordinate units)

    Returns:
      intf_c, mask_c, new_x0, new_y0, (row_off, col_off)
    """
    if intf.shape != mask.shape:
        raise ValueError(f"intf and mask must have same shape, got {intf.shape} vs {mask.shape}")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    H, W = intf.shape

    # Compute integer offsets that align x_star, y_star to pixel grid
    col_f = (x_star - x0) / dx
    row_f = (y0 - y_star) / dy  # note y decreases as row increases

    col_off = int(round(col_f))
    row_off = int(round(row_f))

    # Check sub-pixel mismatch
    if abs(col_f - col_off) * dx > tol or abs(row_f - row_off) * dy > tol:
        raise ValueError(
            f"(x*,y*) not on pixel grid within tolerance: "
            f"col_off ~ {col_f} (dx={dx}), row_off ~ {row_f} (dy={dy}). "
            f"Consider snapping x*/y* to grid."
        )

    # Bounds check for start
    if not (0 <= col_off < W) or not (0 <= row_off < H):
        raise ValueError(f"Requested start is outside the image: row_off={row_off}, col_off={col_off}, shape={intf.shape}")

    # Determine crop size

    width = W - col_off

    height = H - row_off

    # Bounds check for end
    if col_off + width > W or row_off + height > H:
        raise ValueError(
            f"Requested crop (row_off={row_off}, col_off={col_off}, height={height}, width={width}) "
            f"exceeds image bounds {intf.shape}"
        )

    # Do the crop
    intf_c = intf[row_off:row_off + height, col_off:col_off + width]
    mask_c = mask[row_off:row_off + height, col_off:col_off + width]

    # New georeferenced top-left equals the requested (x*, y*)
    new_x0 = x0 + col_off * dx  # should equal x_star within tol
    new_y0 = y0 - row_off * dy  # should equal y_star within tol

    return intf_c, mask_c, new_x0, new_y0, (row_off, col_off)
import math
from typing import Dict, Any, Iterable, Tuple

def _extent_from_meta(m: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return (left, right, top, bottom) from one meta dict."""
    x0 = float(m["east"])
    y0 = float(m["north"])
    dx = float(m["dx"])
    dy = float(m["dy"])
    W  = int(m["ncells"])
    H  = int(m["nlines"])
    left   = x0
    right  = x0 + dx * W
    top    = y0
    bottom = y0 - dy * H
    return left, right, top, bottom

def _check_consistent_spacing(metas: Iterable[Dict[str, Any]], tol=1e-12) -> Tuple[float, float]:
    """Verify dx, dy consistent across metas; return (dx,dy)."""
    dxs = {round(float(m["dx"]), 15) for m in metas}
    dys = {round(float(m["dy"]), 15) for m in metas}
    if len(dxs) != 1 or len(dys) != 1:
        raise ValueError(f"Inconsistent spacing: dxs={dxs}, dys={dys}. Resample first.")
    dx = float(list(dxs)[0]); dy = float(list(dys)[0])
    if dx <= 0 or dy <= 0: raise ValueError("Non-positive dx/dy")
    return dx, dy

def _intersection_bbox(extents: Iterable[Tuple[float,float,float,float]]):
    """Intersect a list of (left,right,top,bottom)."""
    extents = list(extents)
    if not extents:
        raise ValueError("No extents to intersect")
    left   = max(e[0] for e in extents)
    right  = min(e[1] for e in extents)
    top    = min(e[2] for e in extents)   # 'top' is the largest latitude; intersection uses min of tops
    bottom = max(e[3] for e in extents)
    if right <= left or top <= bottom:
        raise ValueError("No overlapping area across frames in this region.")
    return left, right, top, bottom

def _is_on_grid(x_star: float, x0: float, step: float, atol_pix=1e-9) -> bool:
    """Check (x_star - x0)/step is (near) integer."""
    col = (x_star - x0) / step
    return abs(col - round(col)) <= atol_pix

from typing import Optional, Dict, Any, Iterable, Tuple

def build_common_grid_for_region(
    meta_dict: Dict[str, Dict[str, Any]],
    frame_label: str,                      # "North" or "South"
    keys: Optional[Iterable[str]] = None,  # optional subset of keys
    tol_pix: float = 0.01
) -> Tuple[float, float, int, int, float, float]:
    """
    From meta JSON, compute a common crop grid for a region:
      returns (x_star, y_star, width, height, dx, dy)

    - Uses the INTERSECTION of all frames' extents in that region.
    - Ensures (x_star, y_star) lies on the pixel grid of *every* frame.
    """
    # pick region frames
    if keys is None:
        keys = [k for k, m in meta_dict.items() if m.get("frame") == frame_label]
    else:
        keys = [k for k in keys if meta_dict[k].get("frame") == frame_label]
    if not keys:
        raise ValueError(f"No frames with frame='{frame_label}'")

    metas = [meta_dict[k] for k in keys]

    # check spacing consistency
    dx, dy = _check_consistent_spacing(metas)

    # intersect extents
    extents = [_extent_from_meta(m) for m in metas]
    left_i, right_i, top_i, bottom_i = _intersection_bbox(extents)

    # choose target top-left as intersection's left & top
    x_star = left_i
    y_star = top_i

    # verify x_star/y_star align to each frame's grid
    for m in metas:
        x0 = float(m["east"])
        y0 = float(m["north"])
        if not _is_on_grid(x_star, x0, dx, atol_pix=tol_pix) or not _is_on_grid(y_star, y0, dy, atol_pix=tol_pix):
            raise ValueError(
                "Common origin not aligned to all frames' pixel grids.\n"
                f"  offending frame: x0={x0}, y0={y0}\n"
                f"  candidate x*={x_star}, y*={y_star}, dx={dx}, dy={dy}\n"
                "If this persists, you have mixed grids; resample or round x0/y0 to grid."
            )

    # compute pixel width/height from intersection
    width  = int(math.floor((right_i  - x_star) / dx + 1e-9))
    height = int(math.floor((y_star   - bottom_i) / dy + 1e-9))
    if width <= 0 or height <= 0:
        raise ValueError("Degenerate common grid (zero width/height).")

    # snap right/bottom back to grid exactly (informative; not strictly needed for cropping)
    right_snapped  = x_star + width  * dx
    bottom_snapped = y_star - height * dy
    # Optional: you can log differences (right_i - right_snapped), etc.

    return x_star, y_star, width, height, dx, dy
import json

intf_dict_file = open(_bootstrap.asset('intf_coord.json'), 'r')
intf_coords = json.load(intf_dict_file)
new_grid = build_common_grid_for_region(intf_coords,'North')
# EDIT 2026-07-27: was `print(new_grid)`. This runs at *import*, so every
# DataLoader worker re-printed the North grid tuple — 15+ lines of noise before
# training output, and the reason a clean CLI table was impossible. `new_grid`
# is not read anywhere else in the repo (grep confirms), so only the print is
# removed; the assignment stays in case something imports it. Restore the line
# to get the old behaviour back. CHANGELOG.md #3
# print(new_grid)
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

def find_11day_sequences(
    meta: Dict[str, Dict[str, Any]],
    k_prev: int = 2,
    step_days: int = 11,
    restrict_to: Optional[List[str]] = None,
    require_current_nonz_gt0: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Build per-interferogram chains:
      returns (chains_map, valid_currents)

      chains_map = {
         current_key: {
            "prevs": [prev_K, ..., prev_1],   # oldest -> most recent
            "frame": "North" or "South"
         },
         ...
      }

      valid_currents = [current_key, ...]     # only those with full K-previous chains

    Rules:
      - Only consider interferograms with duration == step_days (default 11).
      - 'frame' is taken from meta[k]['frame'] and must be 'North' or 'South'.
      - Previous frames are required exactly i*step_days earlier (i=1..K),
        and must belong to the SAME frame as current.
      - If require_current_nonz_gt0=True, current must have nonz_num > 0.
      - If restrict_to is provided, only those current keys are attempted.
    """

    def parse_key(k: str):
        s, e = k.split("_")
        sd = datetime.strptime(s, "%Y%m%d").date()
        ed = datetime.strptime(e, "%Y%m%d").date()
        return sd, ed, (ed - sd).days

    def make_key(sd, ed) -> str:
        return sd.strftime("%Y%m%d") + "_" + ed.strftime("%Y%m%d")

    # Precompute durations
    daydiff: Dict[str, int] = {}
    for k in meta.keys():
        sd, ed, dd = parse_key(k)
        daydiff[k] = dd

    all_keys = set(meta.keys())
    curr_keys = list(all_keys if restrict_to is None else (all_keys & set(restrict_to)))

    # Partition keys by frame
    frame_groups = {"North": set(), "South": set()}
    for k, info in meta.items():
        frm = info.get("frame")
        if frm in frame_groups:
            frame_groups[frm].add(k)

    chains_map: Dict[str, Dict[str, Any]] = {}
    valid_intfs: List[str] = []

    for cur_key in curr_keys:
        info = meta.get(cur_key)
        if not info:
            continue

        frm = info.get("frame")
        if frm not in frame_groups:
            continue  # skip 'Boundary'/missing/invalid

        sd, ed, dd = parse_key(cur_key)
        if dd != step_days:
            continue

        if require_current_nonz_gt0:
            nonz_raw = info.get("nonz_num", 0)
            try:
                nonz = int(nonz_raw)
            except Exception:
                nonz = 0
            if nonz <= 0:
                continue

        # Build previous list within SAME frame group
        group = frame_groups[frm]
        prevs: List[str] = []
        ok = True
        # build in oldest->newest order
        for i in range(k_prev, 0, -1):
            psd = sd - timedelta(days=i * step_days)
            ped = ed - timedelta(days=i * step_days)
            pk = make_key(psd, ped)
            if (pk not in group) or (daydiff.get(pk) != step_days):
                ok = False
                break
            prevs.append(pk)

        if not ok:
            continue

        chains_map[cur_key] = {"prevs": prevs, "frame": frm}
        valid_intfs.append(cur_key)

    return chains_map, valid_intfs

if __name__ == '__main__':

    with open (_bootstrap.asset('intf_coord.json'), 'r') as f:
        info = json.load(f)
    intf_list =  ['20210120_20210131', '20210222_20210305', '20210119_20210130', '20210210_20210221']
    p_dict, upt_list = find_11day_sequences(info,restrict_to=intf_list)
    print(p_dict)