"""Tables, learning curves and the improvement/deterioration/recovery read-out.

Everything here is read from the ledger, the evaluation JSONs and each run's
results.csv; nothing is recomputed from weights. Numbers and figures only --
the visual panel (panel.py) is a separate, anonymised artefact on purpose.

Learning curves plot every reconstruction threshold as its own line, the
best-across-thresholds curve on top, and the LR the epochs actually trained at
(``lr_used``) underneath, so a change of the best threshold, an LR-driven
recovery and a recovery at a fixed threshold can be told apart. The training
fit probe (``fit/*``) is drawn against validation patch Dice.

The pattern detector is descriptive. For each curve it reports the first
sustained peak, the trough after it and the best value after the trough, and
flags the sequence only when the dip and the recovery both exceed ``delta``
(default 0.006, the project's F1 noise floor), stating which thresholds show it
and what the LR was at each point.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .controller import E30, E60, open_ledger
from .spec import Spec, localize

NOISE_FLOOR = 0.006


def _rows_scores(ledger) -> List[dict]:
    rows = []
    for t in ledger.trials.values():
        for e, summary in t["scores"].items():
            for rth, s in summary.items():
                rows.append({"trial": t["key"], "origin": t["origin"], "geometry": t["geometry"],
                             "seed": t["seed"], "epoch": int(e), "rth": float(rth),
                             **{k: s.get(k) for k in ("f1", "mean_precision", "mean_recall",
                                                      "weighted_f1", "mean_scene_f1", "n_scenes",
                                                      "n_fp", "fp_area", "n_background",
                                                      "background_area")}})
    return sorted(rows, key=lambda r: (r["trial"], r["epoch"], r["rth"]))


def _rows_trials(ledger) -> List[dict]:
    rows = []
    for t in ledger.trials.values():
        r = {"trial": t["key"], "origin": t["origin"], "phase": t["phase"], "seed": t["seed"],
             "lr": t["params"]["lr"], "weight_decay": t["params"]["weight_decay"],
             "pos_w": t["params"]["pos_w"], "run_dir": t["run_dir"]}
        for b in (E30, E60):
            o = t["objectives"].get(b)
            r[f"{b}_objective"] = o["value"] if o else None
            r[f"{b}_rth"] = o["rth"] if o else None
            for rth, v in (o["by_rth"].items() if o else []):
                r[f"{b}_F@{rth}"] = v
        bs = t.get("best_scene")
        r["best_scene_epoch"] = bs["epoch"] if bs else None
        rows.append(r)
    return rows


def _per_scene_rows(ledger) -> List[dict]:
    rows = []
    for t in ledger.trials.values():
        for b in t["evals"].values():
            if not b.get("collected"):
                continue
            for e in b["epochs"]:
                f = Path(localize(b["out_dir"])) / f"e{e:03d}.json"
                if not f.exists():
                    continue
                data = json.loads(f.read_text())
                for intf, per in data["per_scene"].items():
                    for rth, s in per.items():
                        rows.append({"trial": t["key"], "epoch": e, "intf": intf, "rth": float(rth),
                                     **{k: s.get(k) for k in (
                                         "recall", "precision", "gt_area", "n_gt", "n_detected",
                                         "n_pred", "n_fp", "fp_area", "n_background",
                                         "background_area", "fragments_per_detected",
                                         "tp_px", "fp_px", "fn_px")}})
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def results_columns(run_dir: str, cols) -> Dict[str, List]:
    """{column: [(epoch, value), ...]} from a run's results.csv."""
    from ..training.reporter import read_results_csv

    p = Path(localize(run_dir)) / "results.csv"
    if not p.exists():
        return {}
    data = read_results_csv(p)
    out = {}
    for c in cols:
        if c in data:
            out[c] = [(e, v) for e, v in zip(data["epoch"], data[c]) if isinstance(v, (int, float))]
    return out


def curve(t: dict, rth: Optional[str]) -> List[tuple]:
    """[(epoch, F1)]; rth=None gives the best-across-thresholds curve."""
    pts = []
    for e in sorted(t["scores"], key=int):
        s = t["scores"][e]
        if rth is None:
            vals = [v.get("f1") for v in s.values() if v.get("f1") is not None]
            if vals:
                pts.append((int(e), max(vals)))
        elif s.get(rth, {}).get("f1") is not None:
            pts.append((int(e), s[rth]["f1"]))
    return pts


def pattern(points: List[tuple], delta: float = NOISE_FLOOR) -> Optional[dict]:
    """Peak -> trough -> (recovery), when the dip exceeds ``delta``; else None.

    The dip starts at the first epoch more than ``delta`` below the running
    maximum. The peak is the maximum before it, the trough the minimum after
    the peak, and the curve has recovered when its best value after the trough
    is more than ``delta`` above the trough.
    """
    if len(points) < 3:
        return None
    epochs = [p[0] for p in points]
    vals = [p[1] for p in points]
    start = next((j for j in range(1, len(vals)) if vals[j] < max(vals[:j]) - delta), None)
    if start is None:
        return None
    i = max(range(start), key=lambda q: (vals[q], -q))
    j = min(range(i + 1, len(vals)), key=lambda q: (vals[q], q))
    out = {"peak": (epochs[i], vals[i]), "trough": (epochs[j], vals[j]), "recovered": False,
           "after": None}
    if j < len(vals) - 1:
        k = max(range(j + 1, len(vals)), key=lambda q: (vals[q], -q))
        out["after"] = (epochs[k], vals[k])
        out["recovered"] = vals[k] - vals[j] > delta
        out["recovers_above_peak"] = vals[k] > vals[i]
    return out


def pattern_readout(spec: Spec, t: dict) -> dict:
    lr = dict(results_columns(t["run_dir"], ["lr_used"]).get("lr_used", []))
    out = {"best_across": pattern(curve(t, None))}
    for rth in map(str, spec.evaluation["rth"]):
        out[f"rth_{rth}"] = pattern(curve(t, rth))
    fixed = [k for k, v in out.items() if k.startswith("rth_") and v and v["recovered"]]
    for v in out.values():
        if v:
            v["lr_at"] = {name: lr.get(v[name][0]) for name in ("peak", "trough", "after")
                          if v.get(name)}
    out["summary"] = (
        "no dip beyond the noise floor" if not any(out.values()) else
        f"dip-and-recovery at fixed threshold(s) {fixed}" if fixed else
        "dip in the best-across curve without a fixed-threshold recovery: threshold switching"
        if out["best_across"] and out["best_across"]["recovered"] else
        "dip without recovery")
    return out


def _figure(w, h):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w, h), dpi=110)
    FigureCanvasAgg(fig)
    return fig


def plot_trial(spec: Spec, t: dict, out_png: Path) -> Optional[Path]:
    if len(t["scores"]) < 2:
        return None
    fig = _figure(15, 4.6)
    ax_f1, ax_lr, ax_fit = fig.subplots(1, 3)
    for rth in map(str, spec.evaluation["rth"]):
        pts = curve(t, rth)
        if pts:
            ax_f1.plot(*zip(*pts), marker="o", ms=3, lw=1.2, label=f"RTh {rth}")
    best = curve(t, None)
    if best:
        ax_f1.plot(*zip(*best), color="black", lw=2.4, label="best across RTh", zorder=5)
    ax_f1.set(title=f"{t['key']} -- val object F1 (ITh 0.7, b5)", xlabel="epoch", ylabel="F1")
    ax_f1.grid(alpha=.3)
    ax_f1.legend(fontsize=8)
    cols = results_columns(t["run_dir"], ["lr_used", "lr", "val/dice", "fit/dice", "train/loss",
                                         "fit/loss", "val/loss"])
    lr = cols.get("lr_used") or cols.get("lr")
    if lr:
        ax_lr.step(*zip(*lr), where="post", color="#555")
        ax_lr.set_yscale("log")
    ax_lr.set(title="LR the epoch trained at", xlabel="epoch")
    ax_lr.grid(alpha=.3)
    for c, style in (("val/dice", "-"), ("fit/dice", "--")):
        if cols.get(c):
            ax_fit.plot(*zip(*cols[c]), style, label=c)
    if cols.get("fit/loss") and cols.get("val/loss"):
        ax2 = ax_fit.twinx()
        ax2.plot(*zip(*cols["fit/loss"]), ":", color="#888", label="fit/loss")
        ax2.plot(*zip(*cols["val/loss"]), ":", color="#c55", label="val/loss")
        ax2.set_ylabel("loss")
    ax_fit.set(title="patch-level fit (training health only)", xlabel="epoch")
    if ax_fit.get_legend_handles_labels()[0]:
        ax_fit.legend(fontsize=8)
    ax_fit.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png)
    return out_png


def plot_seed_overlay(spec: Spec, trials: List[dict], out_png: Path) -> Optional[Path]:
    if not trials:
        return None
    rths = list(map(str, spec.evaluation["rth"]))
    fig = _figure(4 * (len(rths) + 1), 4.2)
    axes = fig.subplots(1, len(rths) + 1)
    for t in trials:
        for ax, rth in zip(axes, rths + [None]):
            pts = curve(t, rth)
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=2.5, lw=1.3, label=f"seed {t['seed']}")
    for ax, rth in zip(axes, rths + [None]):
        ax.set(title=f"RTh {rth}" if rth else "best across RTh", xlabel="epoch")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("val object F1")
    if axes[-1].get_legend_handles_labels()[0]:
        axes[-1].legend(fontsize=8)
    fig.suptitle(f"{trials[0]['geometry']}: seeds overlaid")
    fig.tight_layout()
    fig.savefig(out_png)
    return out_png


def report(spec: Spec, out: Optional[str] = None) -> str:
    ledger = open_ledger(spec)
    root = Path(out or localize(f"{spec.root}/reports/{datetime.now():%Y%m%d_%H%M}"))
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "trials.csv", _rows_trials(ledger))
    _write_csv(root / "scores_by_epoch_rth.csv", _rows_scores(ledger))
    _write_csv(root / "scores_per_scene.csv", _per_scene_rows(ledger))
    patterns = {}
    for t in ledger.trials.values():
        safe = t["key"].replace("/", "_")
        plot_trial(spec, t, root / f"curves_{safe}.png")
        if len(t["scores"]) >= 3:
            patterns[t["key"]] = pattern_readout(spec, t)
    for g in spec.geometries:
        for origin in ("long", "seed_repeat"):
            group = [t for t in ledger.trials.values()
                     if t["geometry"] == g and t["origin"] == origin]
            if origin == "seed_repeat" and group:
                group = [ledger.trial(group[0]["repeat_of"])] + group
            plot_seed_overlay(spec, group, root / f"seeds_{g}_{origin}.png")
    (root / "patterns.json").write_text(json.dumps(patterns, indent=1))
    (root / "decisions.json").write_text(json.dumps(
        {"decisions": ledger.data.get("decisions", []), "locked": ledger.data.get("locked", {})},
        indent=1))
    lines = [f"report -> {root}"]
    for k, v in patterns.items():
        lines.append(f"  {k}: {v['summary']}")
    return "\n".join(lines)
