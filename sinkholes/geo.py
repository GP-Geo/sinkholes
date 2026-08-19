"""Frame geometry: aligned origins, grid cropping, common-grid intersection.

The coast is imaged by two Sentinel-1 frames whose rasters have slightly
different origins per date. Aligning every scene of a frame to one fixed
origin is what makes patch (i, j) of one date cover the same ground as
patch (i, j) of another — all temporal stacking and all polygon
georeferencing depend on it.
"""

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

#: Pixel size in degrees (both axes) for every interferogram in the dataset.
PIXEL_DEG = 2.777e-05

#: Aligned top-left origin (lon, lat) per frame. Every patch grid and every
#: exported polygon is referenced to these; changing them silently shifts all
#: outputs, so they live in exactly one place.
FRAME_ORIGINS: Dict[str, Tuple[float, float]] = {
    "North": (35.37, 31.79),
    "South": (35.32, 31.44),
}

#: Columns kept when patchifying an aligned scene: [X_CROP_OFFSET, X_CROP_OFFSET + X_CROP_COLS).
X_CROP_OFFSET = 0
X_CROP_COLS = 4500

#: Column crop used when predicting on raw (unaligned-to-patch-grid) scenes.
#: Polygon longitudes are computed from origin + this offset, so it must match
#: how the scene was cropped before tiling.
PREDICT_X_OFFSET = 3000


def aligned_origin(frame: str) -> Tuple[float, float]:
    """The fixed (lon, lat) origin scenes of `frame` are cropped to."""
    try:
        return FRAME_ORIGINS[frame]
    except KeyError:
        raise ValueError(
            f"unknown frame {frame!r}; expected one of {sorted(FRAME_ORIGINS)}"
        ) from None


#: Default patch geometry of every tree in this project: (H, W) and the
#: stride-2 step that tiles them.
DEFAULT_PATCH = (200, 100)


def grid_window(
    frame: str,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    patch_size: Tuple[int, int] = DEFAULT_PATCH,
    stride: Optional[Tuple[int, int]] = None,
    tol: float = 1e-9,
) -> Tuple[int, int, int, int]:
    """Half-open grid ranges ``(row0, row1, col0, col1)`` whose patches lie
    **entirely** inside the lat/lon box.

    This is the one place the benchmark's spatial restrictions are turned into
    grid indices. Three consumers must agree on it — ``dataprep/dataset.py``
    (which patches are loaded), ``inference/scenes.py`` (which tiles are
    predicted) and ``inference/outputs.py`` (which canvas is scored). If they
    disagree, a model is scored on ground it trained on.

    Rows and columns are derived from :data:`FRAME_ORIGINS`, **never** from a
    scene's raw ``north``: every grid on disk was cropped to the frame's
    aligned origin, and the raw value differs from it by up to ~700 rows.

    A patch is inside only when its whole footprint is, which is what keeps the
    31.4 deg cut's two sides disjoint — a straddling patch belongs to neither.

    The upper bounds are *not* clamped to any particular grid: scene grids vary
    from 191 to 200 rows because alignment fixes the origin, not the height.
    Callers clamp with ``min(row1, arr.shape[0])``.

    Args:
      frame: 'North' or 'South'.
      lat_min, lat_max, lon_min, lon_max: the box, in degrees. Defaults are
        wide open, so passing none of them returns the whole grid.
      patch_size: (H, W) of one patch.
      stride: (row_step, col_step) in pixels; defaults to half a patch.
      tol: slack for float comparisons, in grid-step units.

    Returns:
      (row0, row1, col0, col1), half-open. Empty when row0 >= row1 or
      col0 >= col1 — which is a real answer, not an error: it means no patch
      of this frame fits the box.
    """
    lon0, lat0 = aligned_origin(frame)
    ph, pw = patch_size
    sh, sw = stride if stride is not None else (ph // 2, pw // 2)
    if sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, got {(sh, sw)}")
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError(
            f"empty box: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]"
        )

    d_row = sh * PIXEL_DEG          # latitude advanced per grid row (southward)
    d_col = sw * PIXEL_DEG          # longitude advanced per grid column (eastward)

    # patch i spans lat [lat0 - (i*sh + ph)*P, lat0 - i*sh*P]
    #   top    <= lat_max  ->  i >= (lat0 - lat_max) / d_row
    #   bottom >= lat_min  ->  i <= (lat0 - lat_min - ph*P) / d_row
    row0 = math.ceil((lat0 - lat_max) / d_row - tol)
    row1 = math.floor((lat0 - lat_min - ph * PIXEL_DEG) / d_row + tol) + 1

    # patch j spans lon [lon0 + j*sw*P, lon0 + (j*sw + pw)*P]
    col0 = math.ceil((lon_min - lon0) / d_col - tol)
    col1 = math.floor((lon_max - lon0 - pw * PIXEL_DEG) / d_col + tol) + 1

    row0, col0 = max(0, row0), max(0, col0)
    return row0, max(row0, row1), col0, max(col0, col1)


def window_contains(
    window: Tuple[int, int, int, int], row: int, col: int
) -> bool:
    """True when grid cell (row, col) is inside a :func:`grid_window` result."""
    row0, row1, col0, col1 = window
    return row0 <= row < row1 and col0 <= col < col1


def crop_to_start_xy(
    intf: np.ndarray,
    mask: Optional[np.ndarray],
    x0: float,
    y0: float,
    x_star: float,
    y_star: float,
    dx: float = PIXEL_DEG,
    dy: float = PIXEL_DEG,
    tol: float = 1e-2,
):
    """Crop `intf` (and `mask`) so the output's top-left sits at (x_star, y_star).

    Follows rasterio's from_origin convention: x grows to the right, latitude
    decreases by dy per row. Raises when the requested origin is off the pixel
    grid (beyond `tol` in coordinate units) or outside the image.

    Returns (intf_c, mask_c, new_x0, new_y0, (row_off, col_off)); mask_c is
    None when no mask was given.
    """
    if mask is not None and intf.shape != mask.shape:
        raise ValueError(f"intf and mask must have same shape, got {intf.shape} vs {mask.shape}")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    H, W = intf.shape
    col_f = (x_star - x0) / dx
    row_f = (y0 - y_star) / dy
    col_off = int(round(col_f))
    row_off = int(round(row_f))

    if abs(col_f - col_off) * dx > tol or abs(row_f - row_off) * dy > tol:
        raise ValueError(
            f"(x*, y*) not on the pixel grid within tolerance: "
            f"col ~ {col_f} (dx={dx}), row ~ {row_f} (dy={dy})"
        )
    if not (0 <= col_off < W) or not (0 <= row_off < H):
        raise ValueError(
            f"requested start is outside the image: row_off={row_off}, "
            f"col_off={col_off}, shape={intf.shape}"
        )

    intf_c = intf[row_off:, col_off:]
    mask_c = mask[row_off:, col_off:] if mask is not None else None
    return intf_c, mask_c, x0 + col_off * dx, y0 - row_off * dy, (row_off, col_off)


def _extent(m: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """(left, right, top, bottom) of one interferogram's metadata entry."""
    x0, y0 = float(m["east"]), float(m["north"])
    dx, dy = float(m["dx"]), float(m["dy"])
    return x0, x0 + dx * int(m["ncells"]), y0, y0 - dy * int(m["nlines"])


def build_common_grid_for_region(
    meta: Dict[str, Dict[str, Any]],
    frame: str,
    keys: Optional[Iterable[str]] = None,
    tol_pix: float = 0.01,
) -> Tuple[float, float, int, int, float, float]:
    """Intersection grid of all scenes of one frame: (x*, y*, width, height, dx, dy).

    The chosen (x*, y*) is checked to lie on every scene's pixel grid, so
    cropping each scene to it is an integer shift, never a resample.
    """
    if keys is None:
        keys = [k for k, m in meta.items() if m.get("frame") == frame]
    else:
        keys = [k for k in keys if meta[k].get("frame") == frame]
    if not keys:
        raise ValueError(f"no scenes with frame={frame!r}")

    metas = [meta[k] for k in keys]
    dxs = {round(float(m["dx"]), 15) for m in metas}
    dys = {round(float(m["dy"]), 15) for m in metas}
    if len(dxs) != 1 or len(dys) != 1:
        raise ValueError(f"inconsistent pixel spacing: dx={dxs}, dy={dys}; resample first")
    dx, dy = dxs.pop(), dys.pop()

    extents = [_extent(m) for m in metas]
    left = max(e[0] for e in extents)
    right = min(e[1] for e in extents)
    top = min(e[2] for e in extents)
    bottom = max(e[3] for e in extents)
    if right <= left or top <= bottom:
        raise ValueError(f"no overlapping area across {frame} scenes")

    for m in metas:
        col = (left - float(m["east"])) / dx
        row = (float(m["north"]) - top) / dy
        if abs(col - round(col)) > tol_pix or abs(row - round(row)) > tol_pix:
            raise ValueError(
                f"common origin ({left}, {top}) is not on the pixel grid of the scene "
                f"at ({m['east']}, {m['north']}); the frames mix grids"
            )

    width = int(math.floor((right - left) / dx + 1e-9))
    height = int(math.floor((top - bottom) / dy + 1e-9))
    if width <= 0 or height <= 0:
        raise ValueError("degenerate common grid")
    return left, top, width, height, dx, dy
