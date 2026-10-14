"""The fixed validation visual panel, and precision at matched recall.

The panel answers a different question from the rankings: does a model draw
cleaner boundaries, fragment objects less, show tile seams, miss things a
human would not? It is kept apart from the numbers on purpose:

* windows are chosen from the ground truth and the LiDAR coverage ALONE -- GT
  objects stratified by size, plus pure-background windows -- with a fixed
  seed, so no model's output decides what is shown;
* models appear under shuffled letters; the key is written to a separate file
  (``panel_key.json``) to be opened only after the panel has been rated;
* no score appears on it, and a rating sheet (boundaries, fragmentation,
  seams, misses) is written beside it.

Precision at matched recall is the numeric companion: a model that "looks
cleaner" may simply predict less. From the saved vote maps each checkpoint is
scored at every vote level (not only the four RTh), its mean precision/recall
curve over the validation scenes built, and its precision read off at a
common recall. That table names the trials -- it is a number, not a picture --
and lives in its own file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .controller import E30, E60, open_ledger
from .spec import Spec, localize

WINDOW = 240


def _maps_dir(t: dict, epoch: int) -> Path:
    for b in t["evals"].values():
        if epoch in b["epochs"]:
            return Path(localize(b["out_dir"])) / "maps" / f"e{epoch:03d}"
    raise SystemExit(f"{t['key']} has no evaluation of epoch {epoch}")


def _load(spec: Spec, t: dict, epoch: int, intf: str):
    from ..inference.snapshot_eval import confidence_from_votes

    z = np.load(_maps_dir(t, epoch) / f"{intf}.npz")
    g = np.load(Path(localize(spec.root)) / "cache" / "gt" /
                f"{intf}_s{spec.evaluation['data_stride']}.npz")
    if tuple(z["crop"]) != tuple(g["crop"]):
        raise SystemExit(f"{intf}: map crop {tuple(z['crop'])} != GT crop {tuple(g['crop'])}")
    return confidence_from_votes(z["votes"], z["counts"]), z["counts"], g["gt"]


def _threshold_of(ledger, t: dict, rth: Optional[float]) -> float:
    if rth is not None:
        return rth
    lock = ledger.data.get("locked", {}).get(t["geometry"])
    if lock:
        return lock["rth"]
    for b in (E60, E30):
        if b in t["objectives"]:
            return t["objectives"][b]["rth"]
    return 0.25


def select_windows(spec: Spec, scenes: List[str], counts_of, n_windows: int, seed: int):
    """Model-independent windows: GT objects by size tertile, plus background."""
    from ..inference.snapshot_eval import _polygons

    rng = np.random.default_rng(seed)
    objects = []
    for intf in scenes:
        g = np.load(Path(localize(spec.root)) / "cache" / "gt" /
                    f"{intf}_s{spec.evaluation['data_stride']}.npz")["gt"]
        for p in _polygons((g > 0).astype(np.float32)):
            c = p.centroid
            objects.append((intf, int(c.y), int(c.x), p.area))
    n_bg = max(1, n_windows // 4)
    chosen = []
    if objects:
        areas = np.array([o[3] for o in objects])
        cuts = np.quantile(areas, [1 / 3, 2 / 3])
        strata = [[o for o in objects if o[3] <= cuts[0]],
                  [o for o in objects if cuts[0] < o[3] <= cuts[1]],
                  [o for o in objects if o[3] > cuts[1]]]
        k = 0
        while len(chosen) < n_windows - n_bg and any(strata):
            s = strata[k % 3]
            k += 1
            if s:
                o = s.pop(int(rng.integers(len(s))))
                chosen.append({"intf": o[0], "y": o[1], "x": o[2], "kind": ["small", "medium", "large"][(k - 1) % 3]})
    tries = 0
    while sum(c["kind"] == "background" for c in chosen) < n_bg and tries < 500:
        tries += 1
        intf = scenes[int(rng.integers(len(scenes)))]
        counts = counts_of(intf)
        g = np.load(Path(localize(spec.root)) / "cache" / "gt" /
                    f"{intf}_s{spec.evaluation['data_stride']}.npz")["gt"]
        y, x = int(rng.integers(counts.shape[0])), int(rng.integers(counts.shape[1]))
        sl = (slice(max(0, y - WINDOW // 2), y + WINDOW // 2), slice(max(0, x - WINDOW // 2), x + WINDOW // 2))
        if (counts[sl] > 0).mean() > 0.9 and not (g[sl] > 0).any():
            chosen.append({"intf": intf, "y": y, "x": x, "kind": "background"})
    return chosen


def panel(spec: Spec, items: List[str], rth: Optional[float] = None, n_windows: int = 6,
          seed: int = 20261013, out: Optional[str] = None) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ledger = open_ledger(spec)
    models = []
    for item in items:
        key, epoch = item.rsplit("@", 1)
        t = ledger.trial(key)
        models.append({"trial": key, "epoch": int(epoch), "t": t,
                       "rth": _threshold_of(ledger, t, rth)})
    scenes = sorted(set.intersection(*[
        {p.stem for p in _maps_dir(m["t"], m["epoch"]).glob("*.npz")} for m in models]))
    if not scenes:
        raise SystemExit("the selected checkpoints share no scene with saved maps")
    first = models[0]
    windows = select_windows(spec, scenes, lambda i: _load(spec, first["t"], first["epoch"], i)[1],
                             n_windows, seed)
    rng = np.random.default_rng(seed + 1)
    letters = [chr(ord("A") + i) for i in range(len(models))]
    order = list(rng.permutation(len(models)))
    root = Path(out or localize(f"{spec.root}/reports/panel_{seed}"))
    root.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(3.0 * (len(models) + 1), 3.0 * len(windows)), dpi=110)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(windows), len(models) + 1, squeeze=False)
    for r, w in enumerate(windows):
        y0, x0 = max(0, w["y"] - WINDOW // 2), max(0, w["x"] - WINDOW // 2)
        sl = (slice(y0, y0 + WINDOW), slice(x0, x0 + WINDOW))
        gt = None
        for c, idx in enumerate([None] + order):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if idx is None:
                _, _, g = _load(spec, first["t"], first["epoch"], w["intf"])
                gt = g[sl] > 0
                ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"window {r + 1} ({w['kind']})", fontsize=8)
                continue
            m = models[idx]
            conf, _, _ = _load(spec, m["t"], m["epoch"], w["intf"])
            ax.imshow(conf[sl], cmap="gray", vmin=0, vmax=1)
            if gt.any():
                ax.contour(gt, levels=[0.5], colors="#00e5ff", linewidths=0.9)
            pred = conf[sl] > m["rth"]
            if pred.any():
                ax.contour(pred, levels=[0.5], colors="#ffd400", linewidths=0.9)
            if r == 0:
                ax.set_title(f"model {letters[order.index(idx)]}", fontsize=10)
    fig.suptitle("validation panel -- grey: tile-vote confidence; cyan: ground truth; "
                 "yellow: prediction at the model's threshold", fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "panel.png")

    key = {letters[order.index(i)]: {"trial": m["trial"], "epoch": m["epoch"], "rth": m["rth"]}
           for i, m in enumerate(models)}
    (root / "panel_key.json").write_text(json.dumps(
        {"OPEN ONLY AFTER RATING": True, "key": key, "windows": windows, "seed": seed}, indent=1))
    with (root / "rating_sheet.csv").open("w") as fh:
        fh.write("window,model,boundaries_1to5,fragmentation_1to5,seams_1to5,"
                 "missed_detections_1to5,notes\n")
        for r in range(len(windows)):
            for L in letters:
                fh.write(f"{r + 1},{L},,,,,\n")
    mr = matched_recall(spec, models, scenes, root)
    return (f"panel -> {root / 'panel.png'} (key sealed in panel_key.json; rating_sheet.csv)\n"
            f"precision at matched recall -> {mr}")


def pr_curve(spec: Spec, t: dict, epoch: int, scenes: List[str], thresholds) -> Dict[str, dict]:
    """Mean P and R over ``scenes`` at each threshold, from the saved maps."""
    from ..inference.snapshot_eval import aggregate, gt_geometry, object_scores

    per = {}
    ev = spec.evaluation
    for intf in scenes:
        conf, _, gt = _load(spec, t, epoch, intf)
        geo = gt_geometry(gt, ev["buffer"])
        per[intf] = {str(th): object_scores(geo, np.where(conf > th, 1, 0), ev["ith"], ev["buffer"])
                     for th in thresholds}
    return aggregate(per, thresholds)


def precision_at_recall(curve: Dict[str, dict], target: float) -> Optional[dict]:
    """Interpolated precision at mean recall ``target``.

    The best mean precision among operating points whose mean recall is at
    least ``target``, where the candidates are the measured thresholds plus
    the point linearly interpolated at exactly ``target`` between the two
    thresholds that bracket it (the standard interpolated-precision rule: a
    model is credited with the cleanest operating point that still finds as
    much). None when no threshold reaches the target recall.
    """
    pts = sorted((v["mean_recall"], v["mean_precision"], float(t)) for t, v in curve.items()
                 if v.get("mean_recall") is not None and v.get("mean_precision") is not None)
    cands = [(p, t, t) for r, p, t in pts if r >= target]
    for (r0, p0, t0), (r1, p1, t1) in zip(pts, pts[1:]):
        if r0 < target <= r1 and r1 > r0:
            w = (target - r0) / (r1 - r0)
            cands.append((p0 + w * (p1 - p0), t0, t1))
    if not cands:
        return None
    p, ta, tb = max(cands)
    return {"precision": p, "between_rth": sorted({ta, tb})}


def matched_recall(spec: Spec, models: List[dict], scenes: List[str], root: Path,
                   target: Optional[float] = None) -> Path:
    ev = spec.evaluation
    levels = sorted({round((k - 0.5) / 16, 6) for k in range(1, 17)} | set(ev["rth"]))
    curves = {}
    for m in models:
        curves[f"{m['trial']}@{m['epoch']}"] = pr_curve(spec, m["t"], m["epoch"], scenes, levels)
    if target is None:  # the first model's recall at its own threshold
        ref = curves[f"{models[0]['trial']}@{models[0]['epoch']}"]
        target = ref[str(models[0]["rth"])]["mean_recall"]
    table = {name: precision_at_recall(c, target) for name, c in curves.items()}
    out = root / "matched_recall.json"
    out.write_text(json.dumps({"target_recall": target, "scenes": scenes, "levels": levels,
                               "precision_at_target": table, "curves": curves}, indent=1))
    return out
