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
- ``average='vote'`` is the benchmark paper's Confidence Factor and is a
  different quantity from the other two: each tile is BINARISED before it is
  accumulated, so a pixel's value is the *fraction of overlapping tiles that
  labelled it positive*, not a mean probability. At a quarter-patch stride
  (``stride=4``) an interior pixel is covered by up to 16 tiles and the value
  lands on {0, 1/16, ..., 1}. Threshold it with RTh, not with a probability.
"""

from dataclasses import dataclass
from typing import AbstractSet, List, Optional, Sequence, Tuple

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
    positive_tiles: Optional[AbstractSet[Tuple[int, int]]] = None,
    tile_window: Optional[Tuple[int, int, int, int]] = None,
    treat_nodata_regions: bool = False,
    blend: Optional[str] = None,
    window_gamma: float = 1.0,
    average: str = "uniform",
    vote_threshold: float = 0.5,
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

    ``average='vote'`` reproduces the benchmark paper's **Confidence Factor**:
    every tile's probability map is binarised at ``vote_threshold`` (the
    paper's "positive (1) label", 0.5) and the accumulated votes are divided by
    the number of tiles that actually covered each pixel. The result is the
    fraction of overlapping patches voting positive, in {0, 1/n, ..., 1}, and
    the paper's Reconstruction Threshold (RTh) is a threshold on *that* --
    "RTh = 0.25" means "at least 4 of 16 tiles agreed", which is NOT the same
    statement as "the mean probability exceeded 0.25". Do not compare an RTh
    number against a ``recon_th`` number.

    Gated-out tiles (LiDAR, AOI, positives-only) never reach the accumulator,
    so they lower the denominator rather than voting zero -- which is what the
    paper's "up to sixteen overlapping patches" means at scene edges and mask
    boundaries.

    ``positive_tiles``: grid coordinates to restrict prediction to — the
    positives-only protocol, ANDed with the LiDAR gate. Pass the tiles whose
    ground truth is non-empty and the scene is scored the way the benchmark
    paper scores it: "delineate subsidence where it is known to be", rather
    than "find it anywhere on the map". Recall is unaffected by the gate (a
    tile holding a ground-truth pixel is positive by definition, so every such
    pixel keeps all of its overlapping tiles); predictions spilling *outside*
    the positive tiles lose part of their overlap and are attenuated under
    ``average='uniform'``, which is inherent to the protocol.

    ``tile_window``: half-open ``(row0, row1, col0, col1)`` from
    :func:`sinkholes.geo.grid_window` — the split's AOI. Tiles outside it are
    not predicted, which is what makes the scored canvas match the ground the
    model was trained on. It must be the same window ``dataprep/dataset.py``
    used to build the split and the same one ``inference/outputs.py`` crops to;
    a mismatch scores a model on ground it trained on.
    """
    if average not in ("uniform", "coverage", "vote"):
        raise ValueError(f"average must be 'uniform', 'coverage' or 'vote', got {average!r}")
    if average == "vote" and blend == "hann":
        # Hann weights a tile's contribution by position; a vote is by
        # definition unweighted (a tile either labelled the pixel or it did
        # not). Silently letting one win would produce a map that is neither.
        raise ValueError("average='vote' and blend='hann' are mutually exclusive: "
                         "the Confidence Factor counts unweighted tile votes")
    patch_h, patch_w = patch_size
    T = len(stack)
    from ..dataprep.context import centre_slices

    if not stack:
        raise ValueError("reconstruction needs at least one input frame")
    input_shape = stack[0].shape
    if len(input_shape) != 4 or any(p.shape != input_shape for p in stack):
        raise ValueError("all input frames must share (ny, nx, context_h, context_w)")
    centre_rows, centre_cols = centre_slices(input_shape[-2:], patch_size)
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
        counts = (np.zeros((out_h, out_w), dtype=np.float32)
                  if average in ("coverage", "vote") else None)

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
                    image[t, y0:y1, x0:x1] += stack[t][i, j, centre_rows, centre_cols] / (stride**2)
            if gt is not None:
                gt[y0:y1, x0:x1] += gt_grid[i, j] / (stride**2)

            # LiDAR gate: inside the mask of EVERY timestep, over the whole tile.
            if lidar_gates is not None and not lidar_gates[:, y0:y1, x0:x1].all():
                continue
            # AOI gate. Like the positives-only gate below, it sits after the
            # image/gt accumulation so those canvases stay whole and stay
            # comparable with a full evaluation of the same scene; outputs.py
            # crops them to the same window before scoring.
            if tile_window is not None and not (
                tile_window[0] <= i < tile_window[1]
                and tile_window[2] <= j < tile_window[3]
            ):
                continue
            # Positives-only gate. Placed after the image/gt accumulation above
            # so those canvases stay whole and remain comparable with a full
            # evaluation of the same scene.
            if positive_tiles is not None and (i, j) not in positive_tiles:
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
            if prob.shape != (patch_h, patch_w):
                raise ValueError(f"model output {prob.shape} does not match target grid {patch_size}")

            if use_hann:
                pred_num[y0:y1, x0:x1] += prob * window
                pred_den[y0:y1, x0:x1] += window
            else:
                if average == "vote":
                    # Binarise FIRST, then accumulate: one tile casts one vote
                    # per pixel regardless of how confident it was. This is the
                    # whole difference between the Confidence Factor and a mean.
                    pred_sum[y0:y1, x0:x1] += (prob > vote_threshold).astype(np.float32)
                else:
                    pred_sum[y0:y1, x0:x1] += prob / (stride**2) if average == "uniform" else prob
                if counts is not None:
                    counts[y0:y1, x0:x1] += 1.0

    if use_hann:
        confidence = (pred_num / (pred_den + 1e-8)).astype(np.float32)
    elif average in ("coverage", "vote"):
        # counts == 0 means no tile was predicted there (gated out entirely).
        # Dividing by 1 leaves the accumulated 0 as 0, which is the right
        # answer for both modes: no tile voted, so the confidence is zero.
        counts[counts == 0] = 1.0
        confidence = (pred_sum / counts).astype(np.float32)
    else:
        confidence = pred_sum

    thresholded = (confidence > threshold).astype(np.float32)
    return SceneReconstruction(confidence=confidence, thresholded=thresholded, image=image, gt=gt)
