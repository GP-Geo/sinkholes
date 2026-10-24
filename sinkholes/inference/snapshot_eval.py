"""Score several checkpoints on whole scenes from ONE pass over the data.

Run as ``sinkholes eval-snapshots``. This is ``eval-scenes`` + ``eval-outputs``
under the RTh protocol (stride-4 grid, unweighted tile votes at 0.5, LiDAR
gate, partition AOI, ITh/buffer object metric), rebuilt for model selection,
where the same scenes are scored for many checkpoints:

* **one scene load serves every checkpoint.** The stack, the LiDAR gates, the
  ground truth and its polygons are built once per scene; each checkpoint only
  adds its tile predictions and its metrics.
* **only the AOI tiles are read.** Grids are memory-mapped and the AOI's rows
  and columns copied out. The normalisation BRANCH is decided over the whole
  common grid exactly as ``eval-scenes`` decides it -- from a cached per-cell
  min/max computed once per grid file -- and the arithmetic is elementwise, so
  the values are bit-identical while the I/O and memory are a fraction.
* **tiles are batched** (``--tile_batch``). ``--tile_batch 1`` is the legacy
  one-tile-per-forward path exactly; larger batches are what make the GPU earn
  its slot. The effective batch is recorded with the results.
* **missing history is an error.** ``eval-scenes`` used to pad an absent
  predecessor with the current frame and report a plausible, wrong number.
* **one confidence map, every threshold.** Votes and counts are kept as uint8
  (lossless: the confidence is votes/counts), scored at every RTh, and saved
  compressed only with ``--save_maps``.
* **per-scene results are written as they finish**, so a job killed at its wall
  limit continues where it stopped when resubmitted with the same arguments.

The object metric is :func:`sinkholes.training.evaluate.object_level_evaluate`
mirrored operation for operation (tests pin bit-equality), with the GT polygons
and their buffered union computed once per scene instead of once per call, plus
diagnostics the ranking never reads: false-positive and background objects,
fragmentation, pixel counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ENGINE_VERSION = "snapshot-eval-v1"

#: The protocol run_eval.sh's PROTOCOL=rth DATA_STRIDE=4 path uses.
DEFAULT_RTH = (0.125, 0.25, 0.375, 0.5)

log = logging.getLogger("sinkholes.snapshot_eval")


# -- protocol --------------------------------------------------------------------------------

@dataclass
class Protocol:
    partition: str
    k_prevs: int
    split: str = "val"
    aoi_split: str = "val"
    patch_size: Tuple[int, int] = (200, 100)
    data_stride: int = 4
    days_diff: int = 11
    vote_threshold: float = 0.5
    rth: Tuple[float, ...] = DEFAULT_RTH
    ith: float = 0.7
    buffer: float = 5
    min_positives: int = 150
    add_lidar_mask: bool = True
    intf_dict_path: Optional[str] = None

    def key(self) -> dict:
        d = asdict(self)
        d["engine"] = ENGINE_VERSION
        return d


class MissingHistory(SystemExit):
    """A predecessor grid the chain needs is not in the patch tree."""


# -- scene list ------------------------------------------------------------------------------

def resolve_scenes(protocol: Protocol, coord: dict) -> Tuple[List[str], dict, dict]:
    """(scored scenes, chains, dropped) under the eval-scenes rules, sorted."""
    from ..meta import find_11day_sequences

    with open(protocol.partition) as fh:
        listed = list(json.load(fh)[protocol.split])
    dropped: Dict[str, list] = {"min_positives": [], "no_chain": []}
    kept = []
    for i in listed:
        n = coord.get(i, {}).get("nonz_num", "none")
        # filter_by_min_positives: strictly more than the gate is kept.
        if protocol.min_positives <= 0 or (isinstance(n, int) and n > protocol.min_positives):
            kept.append(i)
        else:
            dropped["min_positives"].append((i, n))
    chains: dict = {}
    if protocol.k_prevs > 0:
        chains, with_chains = find_11day_sequences(
            coord, k_prev=protocol.k_prevs, restrict_to=kept, require_current_nonz_gt0=False)
        dropped["no_chain"] = sorted(set(kept) - set(with_chains))
        kept = [i for i in kept if i in set(with_chains)]
    return sorted(kept), chains, dropped


def history_of(intf: str, chains: dict, k_prevs: int) -> Tuple[List[str], List[int]]:
    """Predecessors NEWEST FIRST and the matching ages, as eval-scenes builds them."""
    from ..meta import chain_offsets

    if k_prevs <= 0:
        return [], [0]
    chain = chains[intf]
    age = dict(zip(chain["prevs"], chain_offsets(chain)[:-1]))
    prev_ids = chain["prevs"][:k_prevs][::-1]
    return prev_ids, [0] + [age[p] for p in prev_ids]


def required_ids(protocol: Protocol, coord: dict) -> Tuple[List[str], List[str]]:
    """(scored scenes, every id whose data grid the evaluation opens)."""
    scenes, chains, _ = resolve_scenes(protocol, coord)
    need = set(scenes)
    for s in scenes:
        need.update(history_of(s, chains, protocol.k_prevs)[0])
    return scenes, sorted(need)


# -- the normalisation range cache -----------------------------------------------------------

def _stat_key(path) -> dict:
    st = os.stat(path)
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def cell_minmax(path, cache_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell nanmin/nanmax of a (ny, nx, H, W) grid, cached beside the study.

    min/max are exact and associative, so the min over any [:ny, :nx] block of
    cells equals ``np.nanmin`` over that block -- which is what eval-scenes
    measures before deciding the normalisation branch. Computed once per grid
    file by one streaming pass over a memory map; the cache is keyed on the
    file's size and mtime and is recomputed if either changes.
    """
    path = Path(path)
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / path.parent.name / (path.stem + ".minmax.npz")
        if cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as z:
                    meta = json.loads(str(z["meta"]))
                    if meta == _stat_key(path):
                        return z["cell_min"], z["cell_max"]
            except Exception:  # an unreadable cache is just recomputed
                pass
    grid = np.load(path, mmap_mode="r")
    ny, nx = grid.shape[:2]
    cmin = np.empty((ny, nx), np.float32)
    cmax = np.empty((ny, nx), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells -> NaN, as nanmin does
        for r in range(ny):
            block = np.asarray(grid[r]).astype(np.float32, copy=False).reshape(nx, -1)
            cmin[r] = np.nanmin(block, axis=1)
            cmax[r] = np.nanmax(block, axis=1)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp.npz")
        np.savez(tmp, cell_min=cmin, cell_max=cmax, meta=json.dumps(_stat_key(path)))
        os.replace(tmp, cache)
    return cmin, cmax


def block_range(cmin, cmax, ny, nx) -> Tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmin(cmin[:ny, :nx])), float(np.nanmax(cmax[:ny, :nx]))


# -- one scene -------------------------------------------------------------------------------

@dataclass
class Scene:
    intf: str
    stack: List[np.ndarray]            # chronological; (n_tiles, ch, cw): predicted tiles only
    offsets: List[int]                 # chronological ages, current last (0)
    window: Tuple[int, int, int, int]  # tile window (r0, r1, c0, c1), clipped to the grid
    tile_ok: np.ndarray                # (nr, nc) bool: inside every timestep's LiDAR
    tiles: np.ndarray                  # (n_tiles, 2) window-relative (a, b), row-major
    crop: Tuple[int, int, int, int]    # canvas crop (Y0, Y1, X0, X1) that is scored
    gt: np.ndarray                     # (Y1-Y0, X1-X0) float32, exactly eval-scenes' canvas
    canvas: Tuple[int, int]
    grid: Tuple[int, int]
    branches: List[str] = field(default_factory=list)
    load_s: float = 0.0


def _crop_of(window_unclipped, canvas, patch_size, stride):
    """eval-outputs' AOI crop, in canvas pixels."""
    r0, r1, c0, c1 = window_unclipped
    H_, W_ = canvas
    ph, pw = patch_size
    sy, sx = ph // stride, pw // stride
    return (min(r0 * sy, H_), min(r1 * sy + ph - sy, H_),
            min(c0 * sx, W_), min(c1 * sx + pw - sx, W_))


def load_scene(intf: str, protocol: Protocol, *, data_dir: str, mask_dir: str,
               context_size: Tuple[int, int], chains: dict, aoi, cache_dir=None) -> Scene:
    """Everything one scene contributes, identical to eval-scenes over the AOI."""
    from ..dataprep.patchify import patch_file_name
    from ..geo import aligned_origin
    from ..meta import intf_meta
    from ..normalise import SCENE_RANGE_TOL, is_radian_range, normalise_phase_given_range
    from . import snapshot_eval as _self  # monkeypatch-friendly lookups
    from .reconstruct import canvas_shape

    t0 = time.time()
    ph, pw = protocol.patch_size
    stride = protocol.data_stride
    ch, cw = context_size

    def grid_path(kind, tid):
        d = data_dir if kind == "data" else mask_dir
        return os.path.join(d, patch_file_name(kind, tid, ph, pw, stride))

    prev_ids, ages_newest_first = history_of(intf, chains, protocol.k_prevs)
    ids_newest_first = [intf] + prev_ids
    absent = [t for t in ids_newest_first if not os.path.exists(grid_path("data", t))]
    if absent:
        raise MissingHistory(
            f"{intf}: {len(absent)} grid(s) missing from {data_dir}: {absent}. The "
            f"evaluation refuses to substitute frames; build the tree first.")
    maps = [np.load(grid_path("data", t), mmap_mode="r") for t in ids_newest_first]
    ny = min(m.shape[0] for m in maps)
    nx = min(m.shape[1] for m in maps)
    if any(tuple(m.shape[-2:]) != (ch, cw) for m in maps):
        raise ValueError(f"{intf}: patch file input geometry must match checkpoint {(ch, cw)}")
    canvas = canvas_shape(ny, nx, (ph, pw), stride)

    meta = intf_meta(intf, protocol.intf_dict_path)
    if aoi is not None:
        r0, r1, c0, c1 = _self.grid_window(meta.frame, *aoi, patch_size=(ph, pw),
                                           stride=(ph // stride, pw // stride))
    else:
        r0, r1, c0, c1 = 0, ny, 0, nx
    window = (r0, min(r1, ny), c0, min(c1, nx))
    if window[0] >= window[1] or window[2] >= window[3]:
        raise SystemExit(f"{intf}: no tile lies inside the AOI window")
    crop = _crop_of((r0, r1, c0, c1), canvas, (ph, pw), stride) if aoi is not None \
        else (0, canvas[0], 0, canvas[1])

    # -- LiDAR: inside the mask of the current frame AND every predecessor, over
    # the whole tile. Each distinct source is rasterised once (AND is idempotent)
    # and a tile is tested in O(1) against a summed-area table of the uncovered
    # pixels -- the same boolean eval-scenes computes tile by tile.
    sy, sx = ph // stride, pw // stride
    nr, nc = window[1] - window[0], window[3] - window[2]
    tile_ok = np.ones((nr, nc), bool)
    if protocol.add_lidar_mask:
        x0a, y0a = aligned_origin(meta.frame)
        sources = [meta.lidar_mask] + [intf_meta(p, protocol.intf_dict_path).lidar_mask
                                       for p in prev_ids]
        distinct = list(dict.fromkeys(sources))
        gates = _self.rasterise_lidar_gates(distinct, (x0a, y0a, meta.dx, meta.dy), canvas)
        outside = ~gates.astype(bool).all(axis=0)
        del gates
        sat = np.zeros((outside.shape[0] + 1, outside.shape[1] + 1), np.int32)
        sat[1:, 1:] = outside.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        del outside
        ys = (window[0] + np.arange(nr))[:, None] * sy
        xs = (window[2] + np.arange(nc))[None, :] * sx
        ye = np.minimum(ys + ph, sat.shape[0] - 1)
        xe = np.minimum(xs + pw, sat.shape[1] - 1)
        n_out = sat[ye, xe] - sat[ys, xe] - sat[ye, xs] + sat[ys, xs]
        tile_ok = n_out == 0
        del sat


    # -- ground truth over the crop: every tile that touches it, as eval-scenes sums them
    Y0, Y1, X0, X1 = crop
    i_lo = max(0, -(-(Y0 - ph + 1) // sy)) if Y0 > 0 else 0
    j_lo = max(0, -(-(X0 - pw + 1) // sx)) if X0 > 0 else 0
    i_hi = min(ny, -(-Y1 // sy))
    j_hi = min(nx, -(-X1 // sx))
    mask = np.load(grid_path("mask", intf), mmap_mode="r")
    mgrid = np.array(mask[i_lo:i_hi, j_lo:j_hi, :ph, :pw]).astype(np.float32)
    gt = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    for i in range(i_lo, i_hi):
        for j in range(j_lo, j_hi):
            y0, x0 = i * sy, j * sx
            ys, ye = max(y0, Y0), min(y0 + ph, Y1)
            xs, xe = max(x0, X0), min(x0 + pw, X1)
            if ys >= ye or xs >= xe:
                continue
            tile = mgrid[i - i_lo, j - j_lo] / (stride ** 2)
            gt[ys - Y0:ye - Y0, xs - X0:xe - X0] += tile[ys - y0:ye - y0, xs - x0:xe - x0]
    del mgrid

    # -- the stack: only the tiles that will be predicted, normalised with the
    # whole common grid's branch (elementwise arithmetic, so bit-identical)
    tiles = np.argwhere(tile_ok)                       # row-major, like eval-scenes' loop
    branches = []
    stack_newest_first = []
    for t, m in zip(ids_newest_first, maps):
        cmin, cmax = cell_minmax(grid_path("data", t), cache_dir)
        mn, mx = block_range(cmin, cmax, ny, nx)
        out = np.empty((len(tiles), ch, cw), np.float32)
        k = 0
        for a in range(nr):
            cols = np.nonzero(tile_ok[a])[0]
            if not len(cols):
                continue
            row = np.asarray(m[window[0] + a, window[2]:window[3], :ch, :cw])
            out[k:k + len(cols)] = normalise_phase_given_range(
                row[cols].astype(np.float32, copy=False), mn, mx, range_tol=SCENE_RANGE_TOL)
            k += len(cols)
        stack_newest_first.append(out)
        branches.append("radians" if is_radian_range(mn, mx, range_tol=SCENE_RANGE_TOL)
                        else "unit")
    stack = stack_newest_first[::-1]
    offsets = ages_newest_first[::-1]

    return Scene(intf=intf, stack=stack, offsets=offsets, window=window, tile_ok=tile_ok,
                 tiles=tiles, crop=crop, gt=gt, canvas=canvas, grid=(ny, nx),
                 branches=branches, load_s=time.time() - t0)


# -- prediction ------------------------------------------------------------------------------

def predict_votes(scene: Scene, net, protocol: Protocol, device, tile_batch: int):
    """(votes, counts) uint8 over the scored crop, plus the batch actually used."""
    import torch

    from ..device import memory_format_for
    from ..models.temporal import run_model

    ph, pw = protocol.patch_size
    stride = protocol.data_stride
    sy, sx = ph // stride, pw // stride
    Y0, Y1, X0, X1 = scene.crop
    votes = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    counts = np.zeros_like(votes)
    r0, _, c0, _ = scene.window
    tiles = scene.tiles
    T = len(scene.stack)
    offsets_t = (torch.as_tensor(list(scene.offsets), dtype=torch.float32, device=device)
                 if protocol.k_prevs > 0 else None)
    fmt = memory_format_for(device)
    net.eval()
    batch = max(1, int(tile_batch))
    k = 0
    while k < len(tiles):
        chunk = tiles[k:k + batch]
        x = np.stack([scene.stack[t][k:k + len(chunk)] for t in range(T)], axis=1)
        try:
            with torch.no_grad():
                inp = torch.from_numpy(x).to(device=device, memory_format=fmt)
                prob = torch.sigmoid(run_model(net, inp, offsets_t, None))
                v = (prob[:, 0] > protocol.vote_threshold).cpu().numpy()
        except RuntimeError as exc:  # CUDA OOM: halve and retry, and say so in the record
            if "out of memory" not in str(exc).lower() or batch == 1:
                raise
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            log.warning(f"{scene.intf}: out of memory at tile batch {len(chunk)} -- "
                        f"continuing at {batch}")
            continue
        if v.shape[-2:] != (ph, pw):
            raise ValueError(f"model output {v.shape[-2:]} does not match target grid {(ph, pw)}")
        for (a, b), vote in zip(chunk, v):
            y0 = (r0 + a) * sy - Y0
            x0 = (c0 + b) * sx - X0
            votes[y0:y0 + ph, x0:x0 + pw] += vote.astype(np.uint8)
            counts[y0:y0 + ph, x0:x0 + pw] += 1
        k += len(chunk)
    return votes, counts, batch


def confidence_from_votes(votes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """eval-scenes' vote-mode confidence, bit for bit: votes / max(counts, 1)."""
    pred_sum = votes.astype(np.float32)
    den = counts.astype(np.float32)
    den[den == 0] = 1.0
    return (pred_sum / den).astype(np.float32)


# -- object metrics --------------------------------------------------------------------------

def _polygons(mask: np.ndarray):
    import rasterio.features
    from affine import Affine
    from shapely.geometry import shape

    return [shape(g) for g, v in rasterio.features.shapes(mask, transform=Affine.identity())
            if v > 0]


@dataclass
class GTGeometry:
    polygons: list
    total_area: float
    buffered_union: object


def gt_geometry(gt: np.ndarray, buffer: float) -> GTGeometry:
    from shapely.ops import unary_union

    polys = _polygons(gt.astype(np.float32))
    return GTGeometry(polys, sum(p.area for p in polys),
                      unary_union([p.buffer(buffer) for p in polys]))


def object_scores(gt_geo: GTGeometry, pred: np.ndarray, th: float, buffer: float,
                  epsilon: float = 1e-7) -> dict:
    """object_level_evaluate(gt[None], pred[None], round_ndigits=None), plus diagnostics.

    The recall/precision/gt_area are computed by the same operations in the
    same order as the official function (tests pin equality); the remaining
    fields describe the error without entering any ranking.
    """
    from shapely import STRtree
    from shapely.ops import unary_union

    pred = pred.astype(np.float32)
    pred_polygons = _polygons(pred)
    buffered_pred_union = unary_union([pp.buffer(buffer) for pp in pred_polygons])

    intersection_area = 0.0
    detected = []
    for pgt in gt_geo.polygons:
        covered_fraction = pgt.intersection(buffered_pred_union).area / pgt.area
        hit = covered_fraction > th
        detected.append(hit)
        if hit:
            intersection_area += pgt.area

    fp_area = 0
    n_fp = n_bg = 0
    bg_area = 0.0
    for pp in pred_polygons:
        inter = pp.intersection(gt_geo.buffered_union).area
        if inter / pp.area < th:
            fp_area += pp.area
            n_fp += 1
        if inter == 0:
            n_bg += 1
            bg_area += pp.area

    total_gt_area = gt_geo.total_area
    recall_i = intersection_area / (total_gt_area + epsilon)
    precision_i = intersection_area / (intersection_area + fp_area + epsilon)
    areas = [total_gt_area]
    batch_gt_area = float(np.sum(areas))
    if batch_gt_area == 0.0:
        recall = precision = float("nan")
    else:
        recall = np.sum(np.array([recall_i]) * np.array(areas)) / batch_gt_area
        precision = np.sum(np.array([min(precision_i, 1.0)]) * np.array(areas)) / batch_gt_area

    # Fragmentation: predicted pieces touching each DETECTED ground-truth object.
    fragments = []
    if pred_polygons and any(detected):
        tree = STRtree(pred_polygons)
        for pgt, hit in zip(gt_geo.polygons, detected):
            if hit:
                fragments.append(len(tree.query(pgt, predicate="intersects")))
    return {
        "recall": float(recall), "precision": float(precision), "gt_area": batch_gt_area,
        "n_gt": len(gt_geo.polygons), "n_detected": int(sum(detected)),
        "n_pred": len(pred_polygons), "pred_area": float(sum(p.area for p in pred_polygons)),
        "n_fp": n_fp, "fp_area": float(fp_area),
        "n_background": n_bg, "background_area": float(bg_area),
        "fragments_per_detected": float(np.mean(fragments)) if fragments else None,
    }


def pixel_counts(gt: np.ndarray, pred: np.ndarray) -> dict:
    g, p = gt > 0, pred > 0
    return {"tp_px": int((g & p).sum()), "fp_px": int((~g & p).sum()),
            "fn_px": int((g & ~p).sum())}


def score_maps(gt_geo: GTGeometry, gt: np.ndarray, confidence: np.ndarray,
               protocol: Protocol) -> dict:
    out = {}
    for th in protocol.rth:
        pred = np.where(confidence > th, 1, 0)
        entry = object_scores(gt_geo, pred, protocol.ith, protocol.buffer)
        entry.update(pixel_counts(gt, pred))
        out[str(th)] = entry
    return out


def f1(p, r):
    if p is None or r is None or not (p + r):
        return None
    return 2 * p * r / (p + r)


def aggregate(per_scene: Dict[str, dict], thresholds) -> dict:
    """eval-outputs' _aggregate per threshold, plus the F1s this project quotes."""
    summary = {}
    for th in thresholds:
        vals = [v[str(th)] for v in per_scene.values() if str(th) in v]
        rs = np.array([v["recall"] for v in vals], dtype=float)
        ps = np.array([v["precision"] for v in vals], dtype=float)
        ws = np.array([v.get("gt_area", 0.0) for v in vals], dtype=float)
        ok = ~(np.isnan(rs) | np.isnan(ps))
        n_undefined = int((~ok).sum())
        rs, ps, ws = rs[ok], ps[ok], ws[ok]
        if not len(rs):
            summary[str(th)] = {"mean_recall": None, "mean_precision": None, "f1": None,
                                "n_scenes": 0, "n_undefined": n_undefined}
            continue
        tot = float(ws.sum())
        mr, mp = float(rs.mean()), float(ps.mean())
        wr = float((rs * ws).sum() / tot) if tot else None
        wp = float((ps * ws).sum() / tot) if tot else None
        good = [v for v, k in zip(vals, ok) if k]
        summary[str(th)] = {
            "mean_recall": mr, "mean_precision": mp,
            # The project's "object F1": harmonic mean of the per-scene MEAN P and R.
            "f1": f1(mp, mr),
            "weighted_recall": wr, "weighted_precision": wp, "weighted_f1": f1(wp, wr),
            "mean_scene_f1": float(np.mean([f1(v["precision"], v["recall"]) or 0.0
                                            for v in good])),
            "gt_area": tot, "n_scenes": int(len(rs)), "n_undefined": n_undefined,
            "n_fp": int(sum(v.get("n_fp", 0) for v in good)),
            "fp_area": float(sum(v.get("fp_area", 0.0) for v in good)),
            "n_background": int(sum(v.get("n_background", 0) for v in good)),
            "background_area": float(sum(v.get("background_area", 0.0) for v in good)),
        }
    return summary


# -- checkpoints -----------------------------------------------------------------------------

def sha256_of(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_net(path, device, patch_size, arch: Optional[str] = None):
    """The model and its input footprint, the way eval-scenes loads a checkpoint."""
    import torch

    from ..models import factory

    state = torch.load(path, map_location=device)
    loaded = factory.build_from_checkpoint(state, arch=arch, n_classes=1, bilinear=False,
                                           treat_nodata_regions=False)
    context = loaded.input_size_for(tuple(patch_size))
    net = loaded.model
    net.to(device=device)
    net.load_state_dict(state)
    net.eval()
    return net, tuple(context), loaded.architecture


# -- re-scoring saved maps -------------------------------------------------------------------

class RescoreMismatch(SystemExit):
    """A saved map no longer reproduces the scores it was saved with."""


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def rescore_saved(src_dir, out_dir, labels: Sequence[str], rth: Sequence[float],
                  gt_dir) -> List[str]:
    """Score a finished evaluation's saved vote maps at the thresholds ``rth``.

    A map holds a scene's raw votes and counts, so another threshold needs no
    model and no data grid -- only the confidence map and the cached GT. Every
    threshold the source scored must come out identical, field for field, for
    every scene and label before anything is written, so a map or GT file that
    changed since cannot pass. Writes ``<label>.json`` (the source's header and
    checkpoint, the new per-scene results and summary, and a ``rescored``
    record) and a copy of the maps into ``out_dir``. Returns the labels written.
    """
    import shutil
    from dataclasses import replace

    src_dir, out_dir, gt_dir = Path(src_dir), Path(out_dir), Path(gt_dir)
    rth = tuple(float(t) for t in rth)
    results, gts = {}, {}
    for label in labels:
        src_file = src_dir / f"{label}.json"
        src = json.loads(src_file.read_text())
        fields = {k: v for k, v in src["protocol"].items() if k != "engine"}
        old = Protocol(**fields)
        if not set(old.rth) <= set(rth):
            raise SystemExit(f"{src_file}: the thresholds {list(rth)} must include the scored "
                             f"{list(old.rth)}, which the maps are checked against")
        protocol = replace(old, rth=rth)
        per_scene = {}
        for intf in src["scenes"]:
            if intf not in gts:
                with np.load(gt_dir / f"{intf}_s{old.data_stride}.npz") as z:
                    gt, crop, canvas = z["gt"], z["crop"], z["canvas"]
                gts[intf] = (gt, gt_geometry(gt, old.buffer), crop, canvas)
            gt, gt_geo, crop, canvas = gts[intf]
            with np.load(src_dir / "maps" / label / f"{intf}.npz") as z:
                if not (np.array_equal(z["crop"], crop) and np.array_equal(z["canvas"], canvas)):
                    raise RescoreMismatch(f"{label} {intf}: the map and the cached GT disagree "
                                          f"on the scored crop")
                confidence = confidence_from_votes(z["votes"], z["counts"])
            result = score_maps(gt_geo, gt, confidence, protocol)
            for t in old.rth:
                ref, got = src["per_scene"][intf][str(t)], result[str(t)]
                bad = [k for k in ref if not _same(got.get(k), ref[k])]
                if bad:
                    raise RescoreMismatch(
                        f"{label} {intf} RTh {t}: the saved map gives {bad[0]} "
                        f"{got.get(bad[0])!r}, the source recorded {ref[bad[0]]!r} -- nothing written")
            per_scene[intf] = result
            log.info(f"{label} {intf}: re-scored at {len(rth)} thresholds")
        summary = aggregate(per_scene, rth)
        best = max(rth, key=lambda t: summary[str(t)]["f1"] or -1)
        results[label] = {
            **src, "protocol": protocol.key(), "per_scene": per_scene, "summary": summary,
            "best_rth": best, "best_f1": summary[str(best)]["f1"],
            "finished": datetime.now().isoformat(timespec="seconds"),
            "rescored": {"from": str(src_file), "from_sha256": sha256_of(src_file),
                         "maps": str(src_dir / "maps" / label), "gt": str(gt_dir),
                         "verified_rth": list(old.rth), "engine": ENGINE_VERSION}}

    for label, rec in results.items():
        maps = out_dir / "maps" / label
        maps.mkdir(parents=True, exist_ok=True)
        for intf in rec["scenes"]:
            shutil.copy2(src_dir / "maps" / label / f"{intf}.npz", maps / f"{intf}.npz")
        _atomic_json(out_dir / f"{label}.json", rec)
    return list(results)


# -- the command -----------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH",
                   help="a checkpoint to score, repeatable; LABEL names its results file. "
                        "Every checkpoint must share one input geometry")
    p.add_argument("--partition", required=True, help="partition JSON holding the scene list")
    p.add_argument("--split", default="val", help="which list of --partition to score")
    p.add_argument("--aoi_split", default="val", help="which split's aoi_window to use")
    p.add_argument("--no_aoi", action="store_true", help="score the whole canvas")
    p.add_argument("--patches_dir", required=True)
    p.add_argument("--k_prevs", type=int, required=True)
    p.add_argument("--arch", default=None, help="architecture hint, e.g. tattn_unet")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"),
                   help="the target grid (what is scored); the input size comes from the "
                        "checkpoint")
    p.add_argument("--data_stride", type=int, default=4)
    p.add_argument("--rth", type=float, nargs="+", default=list(DEFAULT_RTH))
    p.add_argument("--ith", type=float, default=0.7)
    p.add_argument("--buffer", type=float, default=5)
    p.add_argument("--min_positives", type=int, default=150)
    p.add_argument("--no-add_lidar_mask", dest="add_lidar_mask", action="store_false")
    p.add_argument("--intf_dict_path", default=None)
    p.add_argument("--intfs", default=None, help="comma list: only these scenes of the split")
    p.add_argument("--tile_batch", type=int, default=32,
                   help="tiles per forward pass; 1 = eval-scenes' exact path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None,
                   help="where per-grid min/max caches live (shared by every job of a study)")
    p.add_argument("--save_maps", action="store_true",
                   help="keep each checkpoint's votes/counts per scene (uint8, compressed)")
    p.add_argument("--legacy_olm", default=None, metavar="LABEL=olm_results.json",
                   help="compare LABEL's per-scene recall/precision with an eval-outputs "
                        "result and report the largest difference (engine check)")


def _parse_checkpoints(items) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--checkpoint takes LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        if not label or any(c in label for c in "/\\ "):
            raise SystemExit(f"bad checkpoint label {label!r}")
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint {path} does not exist")
        out.append((label, path))
    if len({l for l, _ in out}) != len(out):
        raise SystemExit("checkpoint labels must be unique")
    return out


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def _peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)


def main(args) -> None:
    import torch

    from ..dataprep.context import context_margin
    from ..dataprep.partition import load_partition_window
    from ..dataprep.patchify import patch_dir_name
    from ..device import get_device
    from ..meta import load_coord_dict

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = Protocol(partition=args.partition, k_prevs=args.k_prevs, split=args.split,
                        aoi_split=args.aoi_split, data_stride=args.data_stride,
                        patch_size=tuple(args.patch_size),
                        rth=tuple(args.rth), ith=args.ith, buffer=args.buffer,
                        min_positives=args.min_positives, add_lidar_mask=args.add_lidar_mask,
                        intf_dict_path=args.intf_dict_path)
    checkpoints = _parse_checkpoints(args.checkpoint)
    device = get_device()
    log.info(f"{ENGINE_VERSION}: {len(checkpoints)} checkpoint(s) on {device}, "
             f"tile batch {args.tile_batch}")

    nets, contexts, archs, ck_meta = {}, set(), {}, {}
    for label, path in checkpoints:
        net, context, arch = load_net(path, device, protocol.patch_size, args.arch)
        nets[label] = net
        contexts.add(context)
        archs[label] = arch
        ck_meta[label] = {"path": os.path.abspath(path), "sha256": sha256_of(path),
                          "architecture": arch, "input_size": list(context)}
    if len(contexts) != 1:
        raise SystemExit(f"checkpoints disagree on input geometry: {sorted(contexts)}")
    context = contexts.pop()
    margin = context_margin(tuple(protocol.patch_size), context)
    ph, pw = protocol.patch_size
    data_dir = os.path.join(args.patches_dir, patch_dir_name("data", ph, pw, protocol.data_stride,
                                                             protocol.days_diff, margin))
    mask_dir = os.path.join(args.patches_dir, patch_dir_name("mask", ph, pw, protocol.data_stride,
                                                             protocol.days_diff))
    coord = load_coord_dict(args.intf_dict_path)
    scenes, chains, dropped = resolve_scenes(protocol, coord)
    if args.intfs:
        wanted = args.intfs.split(",")
        unknown = sorted(set(wanted) - set(scenes))
        if unknown:
            raise SystemExit(f"--intfs not among the scored scenes: {unknown}")
        scenes = [s for s in scenes if s in set(wanted)]
    if dropped["no_chain"]:
        log.warning(f"{len(dropped['no_chain'])} scene(s) have no {protocol.k_prevs}-chain and "
                    f"are not scored: {dropped['no_chain']}")
    aoi = None if args.no_aoi else load_partition_window(protocol.partition, protocol.aoi_split)

    # Every grid the run will open, checked before the first byte is read.
    need = set()
    for s in scenes:
        need.update([s] + history_of(s, chains, protocol.k_prevs)[0])
    from ..dataprep.patchify import patch_file_name
    absent = sorted(t for t in need if not os.path.exists(
        os.path.join(data_dir, patch_file_name("data", t, ph, pw, protocol.data_stride))))
    if absent:
        raise MissingHistory(f"{len(absent)} of {len(need)} grids missing from {data_dir}: "
                             f"{absent[:10]}{' ...' if len(absent) > 10 else ''}")

    header = {"engine": ENGINE_VERSION, "protocol": protocol.key(), "scenes": scenes,
              "dropped": dropped, "data_dir": data_dir, "mask_dir": mask_dir,
              "context_size": list(context), "tile_batch_requested": int(args.tile_batch),
              "device": str(device), "torch": torch.__version__,
              "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
    partial = out_dir / "partial"
    timings = {}
    for n, intf in enumerate(scenes, 1):
        done = {label: partial / label / f"{intf}.json" for label, _ in checkpoints}
        if all(p.exists() for p in done.values()):
            log.info(f"[{n}/{len(scenes)}] {intf}: already scored -- skipped")
            continue
        scene = load_scene(intf, protocol, data_dir=data_dir, mask_dir=mask_dir,
                           context_size=context, chains=chains, aoi=aoi,
                           cache_dir=args.cache_dir)
        gt_geo = gt_geometry(scene.gt, protocol.buffer)
        log.info(f"[{n}/{len(scenes)}] {intf}: loaded {len(scene.stack)} frames, "
                 f"{int(scene.tile_ok.sum())} of {scene.tile_ok.size} AOI tiles in LiDAR, "
                 f"{len(gt_geo.polygons)} GT polygons, {scene.load_s:.0f}s, "
                 f"peak RSS {_peak_rss_gb():.1f} GB")
        for label, _ in checkpoints:
            if done[label].exists():
                continue
            t0 = time.time()
            votes, counts, used = predict_votes(scene, nets[label], protocol, device,
                                                args.tile_batch)
            t_pred = time.time() - t0
            confidence = confidence_from_votes(votes, counts)
            result = score_maps(gt_geo, scene.gt, confidence, protocol)
            if args.save_maps:
                maps = out_dir / "maps" / label
                maps.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(maps / f"{intf}.npz", votes=votes, counts=counts,
                                    crop=np.array(scene.crop), canvas=np.array(scene.canvas))
            _atomic_json(done[label], {"intf": intf, "per_rth": result,
                                       "tile_batch": used, "predict_s": round(t_pred, 1),
                                       "score_s": round(time.time() - t0 - t_pred, 1),
                                       "branches": scene.branches,
                                       "n_tiles": int(scene.tile_ok.sum())})
            best = max(protocol.rth, key=lambda t: f1(result[str(t)]["precision"],
                                                      result[str(t)]["recall"]) or 0)
            log.info(f"    {label}: {t_pred:.0f}s predict; best RTh {best}: "
                     f"R {result[str(best)]['recall']:.3f} P {result[str(best)]['precision']:.3f}")
        timings[intf] = scene.load_s
        if args.cache_dir:
            gt_cache = Path(args.cache_dir) / "gt" / f"{intf}_s{protocol.data_stride}.npz"
            if not gt_cache.exists():
                gt_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = gt_cache.with_name(f".{gt_cache.stem}.{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, gt=scene.gt, crop=np.array(scene.crop),
                                    canvas=np.array(scene.canvas))
                os.replace(tmp, gt_cache)
        del scene, gt_geo

    for label, _ in checkpoints:
        per_scene, extra = {}, {}
        for intf in scenes:
            rec = json.loads((partial / label / f"{intf}.json").read_text())
            per_scene[intf] = rec["per_rth"]
            extra[intf] = {k: rec[k] for k in ("tile_batch", "predict_s", "score_s",
                                               "branches", "n_tiles")}
        summary = aggregate(per_scene, protocol.rth)
        best = max(protocol.rth, key=lambda t: summary[str(t)]["f1"] or -1)
        _atomic_json(out_dir / f"{label}.json", {
            **header, "label": label, "checkpoint": ck_meta[label],
            "per_scene": per_scene, "summary": summary, "scene_runs": extra,
            "best_rth": best, "best_f1": summary[str(best)]["f1"],
            "peak_rss_gb": round(_peak_rss_gb(), 1),
            "finished": datetime.now().isoformat(timespec="seconds")})
        log.info(f"{label}: F1 by RTh " + ", ".join(
            f"{t}: {summary[str(t)]['f1']:.4f}" for t in protocol.rth
            if summary[str(t)]["f1"] is not None))

    if args.legacy_olm:
        label, olm = args.legacy_olm.split("=", 1)
        legacy = json.loads(Path(olm).read_text())["per_intf"]
        mine = json.loads((out_dir / f"{label}.json").read_text())["per_scene"]
        diffs = [abs(mine[i][str(t)][k] - legacy[i][str(t)][k])
                 for i in mine if i in legacy for t in protocol.rth
                 for k in ("recall", "precision")
                 if legacy[i].get(str(t)) and not np.isnan(legacy[i][str(t)][k])]
        log.info(f"legacy check against {olm}: {len(diffs)} values compared, "
                 f"max |difference| {max(diffs) if diffs else float('nan'):.3g}")
        _atomic_json(out_dir / f"{label}.legacy_check.json",
                     {"olm": olm, "n_compared": len(diffs),
                      "max_abs_diff": max(diffs) if diffs else None})


# Imported names that tests (and eval-scenes' own tests) monkeypatch.
from ..geo import grid_window  # noqa: E402
from .reconstruct import rasterise_lidar_gates  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
