"""The shared full-scene reconstruction engine.

Tiles a temporal stack of patch grids through the network and stitches the
per-patch probabilities back into one scene-sized confidence map, with LiDAR
gating, optional Hann blending and thresholding. Both the aligned-patch
evaluation path and the raw-scene prediction path run through here.

Contracts that are easy to break and expensive to break:

- The stack is **chronological, oldest -> newest, current frame last** — the
  order training feeds the network. Callers that assemble stacks any other way
  must reorder before calling.
- LiDAR gating is an **AND across timesteps**: a tile is predicted only when
  every pixel of the tile lies inside the LiDAR mask of the current frame and
  of every predecessor.
- Tiles that are gated out contribute nothing; under ``average='uniform'``
  their area is averaged as zero probability (the historical behaviour the
  full-scene metrics are calibrated against), under ``average='coverage'``
  the sum is divided by the actual number of contributing tiles.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..device import memory_format_for
from ..normalise import validity_from_normalised
from ..paths import asset


def canvas_shape(ny: int, nx: int, patch_size: Tuple[int, int], stride: int) -> Tuple[int, int]:
    """Output canvas for an (ny, nx) patch grid.

    One trailing stride-step of rows/columns beyond the last patch stays zero
    (for stride >= 2); it produces no polygons but keeps the historical array
    shapes, which downstream outputs and saved arrays match.
    """
    patch_h, patch_w = patch_size
    out_h = ny * (patch_h // stride) + patch_h * (1 - 1 // stride)
    out_w = nx * (patch_w // stride) + patch_w * (1 - 1 // stride)
    return out_h, out_w


def hann_window(h: int, w: int, gamma: float = 1.0) -> np.ndarray:
    """Separable Hann window, peak-normalised, optionally sharpened by gamma."""
    wy = np.hanning(h) if h > 1 else np.ones(1, dtype=np.float32)
    wx = np.hanning(w) if w > 1 else np.ones(1, dtype=np.float32)
    win = (wy[:, None] * wx[None, :]).astype(np.float32)
    if gamma != 1.0:
        win = np.power(win, gamma, dtype=np.float32)
    return win / (win.max() + 1e-8)


def rasterise_lidar_gates(
    sources: Sequence,
    origin: Tuple[float, float, float, float],
    shape: Tuple[int, int],
    lidar_shp_path: Optional[str] = None,
) -> np.ndarray:
    """Rasterise one LiDAR coverage mask per timestep onto the output canvas.

    ``sources`` are 'source' ids of lidar_mask_polygs.shp (case-insensitive);
    a missing/None/'none' source falls back to ALL polygons, as does an id not
    present in the shapefile. ``origin`` is (x0, y0, dx, dy) of canvas pixel
    (0, 0) — pass the *aligned* origin (plus any column-offset shift), or the
    gates land on the wrong ground.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    x0, y0, dx, dy = origin
    lidar_gdf = gpd.read_file(lidar_shp_path or asset("lidar_mask_polygs.shp"))
    transform = rasterio.transform.from_origin(x0, y0, dx, dy)

    gates = np.zeros((len(sources), *shape), dtype=np.uint8)
    for c, src in enumerate(sources):
        if src is None or str(src).strip().lower() in ("", "none", "null"):
            polygons = lidar_gdf
        else:
            col = lidar_gdf["source"].astype(str).str.strip().str.lower()
            polygons = lidar_gdf[col == str(src).strip().lower()]
            if polygons.empty:
                print(f"[warn] LiDAR source {src!r} not in shapefile -> using ALL polygons")
                polygons = lidar_gdf
        gates[c] = rasterize(
            [(g, 1) for g in polygons["geometry"].tolist()],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
    return gates


@dataclass
class SceneReconstruction:
    """confidence/thresholded are (out_h, out_w); image is the accumulated
    input per timestep (T, out_h, out_w), chronological — or None when not
    requested; gt likewise (out_h, out_w) or None."""

    confidence: np.ndarray
    thresholded: np.ndarray
    image: Optional[np.ndarray]
    gt: Optional[np.ndarray]


def reconstruct_scene(
    stack: List[np.ndarray],
    net,
    patch_size: Tuple[int, int],
    stride: int,
    threshold: float,
    *,
    device,
    gt_grid: Optional[np.ndarray] = None,
    lidar_gates: Optional[np.ndarray] = None,
    treat_nodata_regions: bool = False,
    blend: Optional[str] = None,
    window_gamma: float = 1.0,
    average: str = "uniform",
    accumulate_image: bool = True,
    log_progress: bool = False,
) -> SceneReconstruction:
    """Predict every tile of a scene and stitch the results.

    ``stack``: per-timestep (ny, nx, H, W) patch grids, already normalised,
    **chronological oldest -> newest** (T=1 for single-frame models).
    ``lidar_gates``: (T, out_h, out_w) uint8 from :func:`rasterise_lidar_gates`,
    or None to predict everywhere. ``blend='hann'`` weights overlapping
    predictions by a Hann window instead of ``average`` ('uniform' divides the
    plain sum by stride^2; 'coverage' divides by the per-pixel tile count).
    """
    if average not in ("uniform", "coverage"):
        raise ValueError(f"average must be 'uniform' or 'coverage', got {average!r}")
    patch_h, patch_w = patch_size
    T = len(stack)
    ny, nx = stack[0].shape[:2]
    out_h, out_w = canvas_shape(ny, nx, patch_size, stride)
    step_y, step_x = patch_h // stride, patch_w // stride

    image = np.zeros((T, out_h, out_w), dtype=np.float32) if accumulate_image else None
    gt = np.zeros((out_h, out_w), dtype=np.float32) if gt_grid is not None else None

    use_hann = blend == "hann"
    if use_hann:
        window = hann_window(patch_h, patch_w, gamma=window_gamma)
        pred_num = np.zeros((out_h, out_w), dtype=np.float32)
        pred_den = np.zeros((out_h, out_w), dtype=np.float32)
    else:
        pred_sum = np.zeros((out_h, out_w), dtype=np.float32)
        counts = np.zeros((out_h, out_w), dtype=np.float32) if average == "coverage" else None

    memory_format = memory_format_for(device)
    net.eval()
    for i in range(ny):
        if log_progress and i % 20 == 0:
            print(f"  row {i}/{ny}")
        for j in range(nx):
            y0, y1 = i * step_y, i * step_y + patch_h
            x0, x1 = j * step_x, j * step_x + patch_w

            if image is not None:
                for t in range(T):
                    image[t, y0:y1, x0:x1] += stack[t][i, j] / (stride**2)
            if gt is not None:
                gt[y0:y1, x0:x1] += gt_grid[i, j] / (stride**2)

            # LiDAR gate: inside the mask of EVERY timestep, over the whole tile.
            if lidar_gates is not None and not lidar_gates[:, y0:y1, x0:x1].all():
                continue

            x_np = np.stack([np.asarray(stack[t][i, j], dtype=np.float32) for t in range(T)], axis=0)
            if treat_nodata_regions:
                if np.isnan(x_np).any():
                    x_np = np.nan_to_num(x_np, nan=0.0)
                # Validity from the normalised no-data code; block layout
                # [imgs chronological..., validity chronological...].
                x_np = np.concatenate([x_np, validity_from_normalised(x_np)], axis=0)

            batch = torch.from_numpy(x_np[None]).to(device=device, memory_format=memory_format)
            with torch.no_grad():
                prob = torch.sigmoid(net(batch)).squeeze().cpu().numpy().astype(np.float32)

            if use_hann:
                pred_num[y0:y1, x0:x1] += prob * window
                pred_den[y0:y1, x0:x1] += window
            else:
                pred_sum[y0:y1, x0:x1] += prob / (stride**2) if average == "uniform" else prob
                if counts is not None:
                    counts[y0:y1, x0:x1] += 1.0

    if use_hann:
        confidence = (pred_num / (pred_den + 1e-8)).astype(np.float32)
    elif average == "coverage":
        counts[counts == 0] = 1.0
        confidence = (pred_sum / counts).astype(np.float32)
    else:
        confidence = pred_sum

    thresholded = (confidence > threshold).astype(np.float32)
    return SceneReconstruction(confidence=confidence, thresholded=thresholded, image=image, gt=gt)
