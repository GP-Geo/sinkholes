"""Where does a trained temporal-attention model actually look?

This is the cheap test that has to pass before a long-history architecture is
worth building. ``TemporalAttentionUNet`` stores no T anywhere — shared encoder,
attention over whatever arrives, parameter-free positional encoding — so a
checkpoint trained at k_prevs=10 can be *run* on a 40-slot history today, with
no retraining and no weight changes. The only thing standing between it and a
gappy 40-slot sequence was that the positional encoding counted list positions
instead of elapsed time, and that padded batches had no mask; both now take
optional arguments (``models/temporal_attention.py``).

So: feed the model far more history than it was trained on and read
``forward_with_attention``'s weights. Two outcomes, and they point opposite ways.

- **Weight reaches back.** Frames beyond the training horizon draw real
  attention mass, so there is signal out there the model already knows how to
  use, and a hole-tolerant loader is worth building.
- **Weight collapses onto the newest frames.** The far history is decoration.
  Stop here, having spent an afternoon rather than a GPU-week.

The comparison that makes either reading trustworthy is against the *same
patches* at the trained depth, which is why every run scores a control
condition too: attention could look diffuse at T=41 simply because the input is
unfamiliar, and only the control separates "spread out over real history" from
"spread out because confused".

Padding is by frame replication, never zeros. Padded timesteps still pass
through the shared encoder, and ``DoubleConv`` normalises over the folded
``B*T`` batch, so BatchNorm would mix an all-zero frame into the statistics
before the attention mask could hide it. ``scenes.py --fallback_replicate``
already established the convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from bisect import bisect_left
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..dataprep.partition import load_partition_split, load_partition_window
from ..dataprep.patchify import patch_file_name, patch_strides, resolve_patch_dirs
from ..device import get_device
from ..geo import FRAME_ORIGINS, grid_window
from ..meta import DENSE_SCHEDULE, frame_groups_of, load_coord_dict, select_history
from ..models.factory import build_from_checkpoint
from ..normalise import PATCH_RANGE_TOL, normalise_channels

logger = logging.getLogger(__name__)

#: Offsets at or below this count as "the model is looking at the present".
#: Used only for the collapse statistic, which asks how often the argmax over
#: time is the current frame or the one immediately before it.
COLLAPSE_OFFSET = 1


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="a tattn_unet checkpoint (best.pt)")
    p.add_argument("--partition", required=True, help="partition JSON naming the split")
    p.add_argument("--split", default="val", help="which list to probe (default: val)")
    p.add_argument("--patches_dir", required=True, help="root holding the data_patches_* tree")
    p.add_argument("--intf_dict", default=None, help="coordinate dictionary (default: the asset)")
    p.add_argument("--patch_size", type=int, nargs=2, default=[200, 100])
    p.add_argument("--strides_per_patch", type=int, default=2)
    p.add_argument("--use_cleaned_patches", action="store_true")

    p.add_argument("--lookback", type=int, default=40,
                   help="probe depth, in 11-day slots (default: 40, about 14 months)")
    p.add_argument("--schedule", default=DENSE_SCHEDULE,
                   help="'dense' or a 'step:until' list such as '1:6,2:12,4:40'. Dense is "
                        "the right choice for a probe: the question is where the weight "
                        "lands, and thinning would decide part of that answer in advance.")
    p.add_argument("--control_lookback", type=int, default=10,
                   help="depth of the control condition; set it to the k_prevs the "
                        "checkpoint was trained at (default: 10)")

    p.add_argument("--score_depths", type=int, nargs="*", default=[],
                   help="also score patch Dice at each of these lookbacks, on the same "
                        "patches. Worth running whenever the weights come out uniform: a "
                        "model that averages over time should suppress temporally "
                        "inconsistent noise better the more frames it averages, and that "
                        "prediction is testable without retraining anything.")
    p.add_argument("--max_per_intf", type=int, default=16,
                   help="patches sampled per interferogram; the grids are 1.4 GB each, so "
                        "this bounds the read, not the compute (default: 16)")
    p.add_argument("--max_intfs", type=int, default=0, help="cap the split (0 = all)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="outputs/attention_probe")
    p.add_argument("--device", default=None, help="override the auto-selected device")
    p.add_argument("--require_selectivity", type=float, default=None, metavar="FRAC",
                   help="exit non-zero unless the control condition uses fewer than "
                        "FRAC x T effective frames. 1.0 means 'exactly uniform', so "
                        "0.9 is a usable collapse alarm. This is the check that would "
                        "have caught docs/ATTENTION_COLLAPSE.md, and it can only be "
                        "made against a trained checkpoint -- a fresh model passes any "
                        "content-sensitivity test and still collapses in training.")


# -- history assembly ---------------------------------------------------------------------

def replicated_plan(
    available: Sequence[Tuple[str, int]],
    candidates: Sequence[int],
) -> Tuple[List[str], List[int], List[bool]]:
    """Lay the available frames onto the full candidate offset grid.

    Every sample then has the same length and the same offsets, and differs only
    in ``valid`` — which is what makes "mean weight at offset 22" a well-defined
    quantity across interferograms with different holes.

    A slot with no acquisition is filled by the nearest available frame (ties to
    the older one) and marked invalid. The fill exists purely to keep BatchNorm
    fed with in-distribution data; attention gives it exactly zero weight.

    ``available`` is (id, offset) oldest first; ``candidates`` is the offset grid,
    also oldest first and ending at 0. Returns (ids, offsets, valid).
    """
    have: Dict[int, str] = {offset: tid for tid, offset in available}
    present = sorted(have)  # ascending offsets, i.e. newest first

    ids: List[str] = []
    valid: List[bool] = []
    for offset in candidates:
        if offset in have:
            ids.append(have[offset])
            valid.append(True)
            continue
        pos = bisect_left(present, offset)
        neighbours = [present[i] for i in (pos - 1, pos) if 0 <= i < len(present)]
        nearest = min(neighbours, key=lambda o: (abs(o - offset), -o))
        ids.append(have[nearest])
        valid.append(False)
    return ids, list(candidates), valid


class GridCache:
    """Memmapped patch grids, opened once and shared across interferograms.

    A 40-slot history over a 23-interferogram split touches each grid file from
    many currents; over a network mount, re-opening them is most of the runtime.
    Memmap also means only the patches actually indexed are read — 80 KB each,
    contiguous — rather than the 1.4 GB file.
    """

    def __init__(self, image_dir: str, patch_size, strides_per_patch: int, cleaned: bool,
                 kind: str = "data"):
        self.image_dir = image_dir
        self.patch_h, self.patch_w = patch_size
        self.strides_per_patch = strides_per_patch
        self.cleaned = cleaned
        self.kind = kind
        self._open: Dict[str, np.ndarray] = {}

    def get(self, intf_id: str) -> Optional[np.ndarray]:
        if intf_id not in self._open:
            name = patch_file_name(self.kind, intf_id, self.patch_h, self.patch_w,
                                   self.strides_per_patch, cleaned=self.cleaned)
            path = os.path.join(self.image_dir, name)
            self._open[intf_id] = (np.load(path, mmap_mode="r")
                                   if os.path.exists(path) else None)
        return self._open[intf_id]


# -- weight accumulation ------------------------------------------------------------------

class WeightStats:
    """Running attention statistics for one condition, over one offset grid.

    Everything is accumulated as sums so the probe never holds more than one
    batch of weights: the raw tensor is (B, heads, T, Hb, Wb), which at T=41 and
    a 13x7 bottleneck is already 380k floats per sample.
    """

    def __init__(self, offsets: Sequence[int], heads: int):
        self.offsets = list(offsets)
        self.heads = heads
        t = len(self.offsets)
        self.weight_sum = np.zeros((heads, t), dtype=np.float64)
        self.valid_sum = np.zeros(t, dtype=np.float64)   # weighted by pixels*heads
        self.argmax_counts = np.zeros(t, dtype=np.float64)
        self.n_positions = 0.0                            # samples * heads * pixels
        self.effective_frames = 0.0
        self.collapsed = 0.0
        self.beyond_horizon = 0.0
        self.horizon = 0

    def update(self, weights: torch.Tensor, valid: torch.Tensor, horizon: int) -> None:
        """``weights`` (B, heads, T, Hb, Wb) summing to 1 over T; ``valid`` (B, T)."""
        self.horizon = horizon
        w = weights.detach().float().cpu().numpy()
        v = valid.detach().cpu().numpy().astype(bool)
        b, heads, t, hb, wb = w.shape

        # Mean over the bottleneck grid, kept per (sample, head, offset).
        per_position = w.reshape(b, heads, t, hb * wb)
        self.weight_sum += per_position.sum(axis=(0, 3))
        self.valid_sum += (v.astype(np.float64).sum(axis=0) * heads * hb * wb)
        self.n_positions += b * heads * hb * wb

        # Entropy per (sample, head, pixel) -> effective number of frames used.
        # Averaging exp(H) over positions is the honest version: exp of the mean
        # entropy would hide a mixture of collapsed and spread-out pixels.
        safe = np.clip(per_position, 1e-12, None)
        entropy = -(per_position * np.log(safe)).sum(axis=2)
        self.effective_frames += float(np.exp(entropy).sum())

        # Which single frame each (sample, head, pixel) leans on hardest. The
        # mean weight alone cannot answer this: a set of pixels each peaked on a
        # *different* offset averages out to something indistinguishable from a
        # model that ignores time altogether, and those are opposite findings.
        argmax = per_position.argmax(axis=2)                       # (B, heads, P)
        offsets = np.asarray(self.offsets)
        self.argmax_counts += np.bincount(argmax.reshape(-1), minlength=t).astype(np.float64)
        self.collapsed += float((offsets[argmax] <= COLLAPSE_OFFSET).sum())

        beyond = offsets > horizon
        if beyond.any():
            self.beyond_horizon += float(per_position[:, :, beyond, :].sum())

    def rows(self) -> List[Dict[str, float]]:
        """Per-offset summary, newest first."""
        out = []
        for i, offset in enumerate(self.offsets):
            seen = self.valid_sum[i]
            mean = float(self.weight_sum[:, i].sum() / seen) if seen else float("nan")
            out.append({
                "offset": offset,
                "days": offset * 11,
                "mean_weight": mean,
                # Of the positions where this offset existed, how often it won.
                "argmax_share": (float(self.argmax_counts[i] / seen) if seen else float("nan")),
                "availability": float(seen / self.n_positions) if self.n_positions else 0.0,
                **{f"head{h}": (float(self.weight_sum[h, i] / (seen / self.heads))
                                if seen else float("nan"))
                   for h in range(self.heads)},
            })
        return sorted(out, key=lambda r: r["offset"])

    def summary(self) -> Dict[str, float]:
        n = self.n_positions or 1.0
        # Frames actually present, averaged over samples: the uniform baseline.
        mean_valid = self.valid_sum.sum() / n
        return {
            "frames_in_grid": len(self.offsets),
            "mean_frames_available": mean_valid,
            "effective_frames": self.effective_frames / n,
            "effective_frames_if_uniform": mean_valid,
            "mass_beyond_offset_%d" % self.horizon: self.beyond_horizon / n,
            "collapse_fraction": self.collapsed / n,
        }


# -- the probe ----------------------------------------------------------------------------

def probe_condition(
    model,
    device,
    condition: str,
    lookback: int,
    schedule: str,
    intf_ids: Sequence[str],
    coord: Dict,
    groups: Dict[str, set],
    cache: GridCache,
    nonz: Dict[str, list],
    window_box,
    patch_size,
    strides_per_patch: int,
    max_per_intf: int,
    batch_size: int,
    horizon: int,
    rng: np.random.Generator,
) -> Optional[WeightStats]:
    """Run one (lookback, schedule) condition over the split and accumulate weights."""
    patch_h, patch_w = patch_size
    stats: Optional[WeightStats] = None
    n_patches = 0

    for intf_id in intf_ids:
        ids_avail, offs_avail = select_history(intf_id, coord, lookback=lookback,
                                               schedule=schedule, groups=groups)
        candidates = sorted(set(offs_avail) | set(_candidate_grid(lookback, schedule)),
                            reverse=True)
        ids, offsets, valid = replicated_plan(list(zip(ids_avail, offs_avail)), candidates)

        grids = [cache.get(tid) for tid in ids]
        if any(g is None for g in grids):
            logger.warning("%s: missing a grid file in its history — skipped", intf_id)
            continue
        ny = min(g.shape[0] for g in grids)
        nx = min(g.shape[1] for g in grids)

        coords = _sample_coords(intf_id, nonz, coord, ny, nx, window_box,
                               patch_size, strides_per_patch, max_per_intf, rng)
        if not coords:
            continue

        if stats is None:
            stats = WeightStats(offsets, heads=model.tattn_heads)
        elif stats.offsets != offsets:
            raise RuntimeError(
                f"{intf_id} produced offset grid {offsets} against {stats.offsets}; "
                f"the candidate grid must be identical across interferograms for the "
                f"per-offset means to be comparable"
            )

        offsets_t = torch.tensor(offsets, dtype=torch.float32, device=device)
        for start in range(0, len(coords), batch_size):
            chunk = coords[start:start + batch_size]
            stack = np.stack([
                normalise_channels(
                    np.stack([np.asarray(g[i, j, :patch_h, :patch_w], dtype=np.float32)
                              for g in grids], axis=0),
                    range_tol=PATCH_RANGE_TOL,
                )
                for (i, j) in chunk
            ], axis=0)                                        # (B, T, H, W)

            images = torch.from_numpy(stack).to(device=device, dtype=torch.float32)
            valid_t = torch.tensor(valid, dtype=torch.bool, device=device)
            valid_t = valid_t.unsqueeze(0).expand(images.shape[0], -1)

            with torch.no_grad():
                _, weights = model.forward_with_attention(
                    images, offsets_t, valid_t,
                )
            stats.update(weights, valid_t, horizon)
            n_patches += len(chunk)

        logger.info("  %s: %d/%d frames available, %d patches",
                    intf_id, sum(valid), len(valid), len(coords))

    if stats is not None:
        logger.info("%s: %d patches over %d interferograms", condition, n_patches, len(intf_ids))
    return stats


def score_depths(
    model, device, depths: Sequence[int], intf_ids, coord, groups, cache, mask_cache,
    nonz, window_box, patch_size, strides_per_patch, max_per_intf, batch_size, rng_seed,
) -> List[Dict[str, float]]:
    """Patch Dice at each lookback, over identical patches.

    The companion measurement to the weight table, and the one that matters if
    the weights come out uniform: a uniform average over T frames suppresses
    temporally inconsistent noise as roughly 1/sqrt(T), so deeper history should
    score better *even with the attention dead*. If Dice is flat in T, the extra
    frames carry nothing and neither hole tolerance nor a longer lookback is
    worth building.

    Scored at a fixed 0.5 threshold on the pooled patches, which is enough to
    rank depths against each other; it is not comparable to the run's own
    val/dice, which is computed over a different set with its own threshold.
    """
    patch_h, patch_w = patch_size
    out = []
    for depth in depths:
        inter = pred_sum = gt_sum = 0.0
        n_patches = frames_total = 0
        rng = np.random.default_rng(rng_seed)     # same patches at every depth
        for intf_id in intf_ids:
            ids, offs = select_history(intf_id, coord, lookback=depth,
                                       schedule=DENSE_SCHEDULE, groups=groups)
            grids = [cache.get(tid) for tid in ids]
            gt_grid = mask_cache.get(intf_id)
            if any(g is None for g in grids) or gt_grid is None:
                continue
            ny = min([g.shape[0] for g in grids] + [gt_grid.shape[0]])
            nx = min([g.shape[1] for g in grids] + [gt_grid.shape[1]])
            coords = _sample_coords(intf_id, nonz, coord, ny, nx, window_box,
                                    patch_size, strides_per_patch, max_per_intf, rng)
            if not coords:
                continue
            frames_total += len(ids) * len(coords)

            offsets_t = torch.tensor(offs, dtype=torch.float32, device=device)
            for start in range(0, len(coords), batch_size):
                chunk = coords[start:start + batch_size]
                stack = np.stack([
                    normalise_channels(
                        np.stack([np.asarray(g[i, j, :patch_h, :patch_w], dtype=np.float32)
                                  for g in grids], axis=0),
                        range_tol=PATCH_RANGE_TOL,
                    )
                    for (i, j) in chunk
                ], axis=0)
                truth = np.stack([np.asarray(gt_grid[i, j, :patch_h, :patch_w],
                                             dtype=np.float32) for (i, j) in chunk]) > 0

                images = torch.from_numpy(stack).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(images, offsets_t)
                pred = (torch.sigmoid(logits).squeeze(1).cpu().numpy() > 0.5)

                inter += float((pred & truth).sum())
                pred_sum += float(pred.sum())
                gt_sum += float(truth.sum())
                n_patches += len(chunk)

        dice = 2 * inter / (pred_sum + gt_sum) if (pred_sum + gt_sum) else float("nan")
        row = {
            "lookback": depth,
            "mean_frames": frames_total / n_patches if n_patches else float("nan"),
            "dice": dice,
            "precision": inter / pred_sum if pred_sum else float("nan"),
            "recall": inter / gt_sum if gt_sum else float("nan"),
            "patches": n_patches,
        }
        logger.info("  lookback %2d: %5.1f frames, dice %.4f (P %.4f / R %.4f) over %d patches",
                    depth, row["mean_frames"], dice, row["precision"], row["recall"], n_patches)
        out.append(row)
    return out


def _candidate_grid(lookback: int, schedule: str) -> List[int]:
    from ..meta import parse_history_schedule

    return parse_history_schedule(schedule, lookback) + [0]


def _sample_coords(intf_id, nonz, coord, ny, nx, window_box, patch_size,
                   strides_per_patch, max_per_intf, rng) -> List[Tuple[int, int]]:
    """Positive patch coordinates of one interferogram, inside the split's window."""
    window = None
    if window_box is not None:
        meta = coord.get(intf_id)
        if meta is None or meta.get("frame") not in FRAME_ORIGINS:
            raise KeyError(f"{intf_id} has no usable frame; its AOI window cannot be resolved")
        r0, r1, c0, c1 = grid_window(
            meta["frame"], *window_box,
            patch_size=tuple(patch_size),
            stride=patch_strides(tuple(patch_size), strides_per_patch),
        )
        window = (r0, min(r1, ny), c0, min(c1, nx))

    out = []
    for ij in nonz.get(intf_id, []):
        i, j = int(ij[0]), int(ij[1])
        if not (0 <= i < ny and 0 <= j < nx):
            continue
        if window is not None and not (window[0] <= i < window[1] and window[2] <= j < window[3]):
            continue
        out.append((i, j))
    if max_per_intf and len(out) > max_per_intf:
        pick = rng.choice(len(out), size=max_per_intf, replace=False)
        out = [out[k] for k in sorted(pick.tolist())]
    return out


# -- reporting ----------------------------------------------------------------------------

def report(stats: Dict[str, WeightStats], out_dir: str, probe_name: str, control_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for name, st in stats.items():
        rows = st.rows()
        path = os.path.join(out_dir, f"weights_{name}.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("wrote %s", path)

    summary = {name: st.summary() for name, st in stats.items()}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    probe, control = stats[probe_name], stats[control_name]
    logger.info("")
    logger.info("=" * 78)
    logger.info("ATTENTION OVER OFFSET — %s  (offset 0 = current, 1 = 11 days back)",
                probe_name)
    logger.info("=" * 78)
    logger.info("%7s %6s %10s %10s %9s %8s  %s",
                "offset", "days", "weight", "x uniform", "argmax%", "avail%", "")
    uniform = 1.0 / max(probe.summary()["mean_frames_available"], 1e-9)
    for row in sorted(probe.rows(), key=lambda r: r["offset"]):
        # An offset no sample ever carried has nothing to report — with a short
        # split most far offsets are like this, and it is not an error.
        if not np.isfinite(row["mean_weight"]):
            logger.info("%7d %6d %10s %10s %9s %8.0f", row["offset"], row["days"],
                        "-", "-", "-", 0.0)
            continue
        ratio = row["mean_weight"] / uniform
        logger.info("%7d %6d %10.5f %10.2f %9.1f %8.0f  %s",
                    row["offset"], row["days"], row["mean_weight"], ratio,
                    100 * row["argmax_share"], 100 * row["availability"],
                    "#" * int(round(min(ratio, 6.0) * 8)))

    logger.info("")
    logger.info("%-32s %14s %14s", "", control_name, probe_name)
    logger.info("%-32s %14s %14s", "-" * 32, "-" * 14, "-" * 14)
    for key in ("frames_in_grid", "mean_frames_available", "effective_frames",
                "effective_frames_if_uniform", "collapse_fraction"):
        logger.info("%-32s %14.3f %14.3f", key, control.summary()[key], probe.summary()[key])
    for key in (k for k in probe.summary() if k.startswith("mass_beyond")):
        logger.info("%-32s %14s %14.3f", key, "-", probe.summary()[key])


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(args.device) if args.device else get_device()
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    loaded = build_from_checkpoint(state, n_classes=1)
    if loaded.input_size_for(tuple(args.patch_size)) != tuple(args.patch_size):
        raise SystemExit("attention-probe does not yet support context patches; "
                         "use eval-scenes/test-patches for context checkpoints")
    if loaded.architecture != "tattn_unet":
        raise SystemExit(
            f"{args.model} is a {loaded.architecture!r} checkpoint. This probe reads "
            f"temporal attention weights, which only tattn_unet has."
        )
    model = loaded.model
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info("model on %s: %s", device, model.config_dict())

    coord = load_coord_dict(args.intf_dict)
    groups = frame_groups_of(coord)
    intf_ids = [i for i in load_partition_split(args.partition, args.split) if i in coord]
    if args.max_intfs:
        intf_ids = intf_ids[: args.max_intfs]
    window_box = load_partition_window(args.partition, args.split)
    logger.info("%s split: %d interferograms, AOI window %s",
                args.split, len(intf_ids), window_box)

    image_dir, mask_dir = resolve_patch_dirs(args.patches_dir, tuple(args.patch_size),
                                             args.strides_per_patch,
                                             cleaned=args.use_cleaned_patches)
    with open(os.path.join(image_dir, "nonz_indices.json")) as fh:
        nonz = json.load(fh)
    cache = GridCache(image_dir, tuple(args.patch_size), args.strides_per_patch,
                      args.use_cleaned_patches)
    mask_cache = GridCache(mask_dir, tuple(args.patch_size), args.strides_per_patch,
                           args.use_cleaned_patches, kind="mask")

    conditions = [
        (f"control_k{args.control_lookback}", args.control_lookback, DENSE_SCHEDULE),
        (f"probe_k{args.lookback}", args.lookback, args.schedule),
    ]
    stats: Dict[str, WeightStats] = {}
    for name, lookback, schedule in conditions:
        logger.info("")
        logger.info("-- %s (lookback %d, schedule %s) --", name, lookback, schedule)
        # The seed is reset per condition so both score the SAME patches; the
        # whole comparison rests on that.
        result = probe_condition(
            model, device, name, lookback, schedule, intf_ids, coord, groups, cache, nonz,
            window_box, tuple(args.patch_size), args.strides_per_patch,
            args.max_per_intf, args.batch_size, horizon=args.control_lookback,
            rng=np.random.default_rng(args.seed),
        )
        if result is None:
            raise SystemExit(f"{name}: no patches were scored — check the split and the AOI window")
        stats[name] = result

    report(stats, args.out_dir, conditions[1][0], conditions[0][0])

    if args.score_depths:
        logger.info("")
        logger.info("-- Dice vs history depth, same patches at every depth --")
        rows = score_depths(
            model, device, args.score_depths, intf_ids, coord, groups, cache, mask_cache,
            nonz, window_box, tuple(args.patch_size), args.strides_per_patch,
            args.max_per_intf, args.batch_size, args.seed,
        )
        path = os.path.join(args.out_dir, "dice_vs_depth.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("wrote %s", path)

    if args.require_selectivity is not None:
        control = stats[conditions[0][0]]
        used = control.summary()["effective_frames"]
        total = control.summary()["mean_frames_available"]
        limit = args.require_selectivity * total
        logger.info("")
        if used >= limit:
            raise SystemExit(
                f"COLLAPSED: the attention used {used:.3f} of {total:.3f} available "
                f"frames ({100 * used / total:.1f}% of uniform), at or above the "
                f"{100 * args.require_selectivity:.0f}% limit. This checkpoint is "
                f"averaging its history, not selecting from it -- see "
                f"docs/ATTENTION_COLLAPSE.md. Retrain with --tattn_contrast "
                f"--tattn_qk_norm (both are on by default)."
            )
        logger.info("selectivity OK: %.3f of %.3f frames used (%.1f%% of uniform, "
                    "limit %.0f%%)", used, total, 100 * used / total,
                    100 * args.require_selectivity)
