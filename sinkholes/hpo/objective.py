"""The selection objective, computed one way everywhere.

For a trial scored at epochs E (20 and 30 for the initial budget, 50 and 60
for the promoted one) and reconstruction thresholds RTh:

    F(t)      = mean over e in E of  F1(e, t)
    objective = max over t of F(t)

where F1(e, t) is the project's object F1 -- the harmonic mean of the
per-scene MEAN recall and MEAN precision over the validation scenes (ITh 0.7,
buffer 5) at epoch e, threshold t. ONE threshold serves every scene and both
checkpoints; the per-scene best threshold is never taken. A threshold at which
an epoch's F1 is undefined (no scene scored, or P = R = 0) counts as 0.

Patch Dice never enters. The budget is part of the result's identity: an
objective over epochs 20/30 and one over 50/60 are different quantities and
are stored under different keys (``e30``, ``e60``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .spec import localize


def epoch_scores(result_json) -> Dict[str, dict]:
    """{rth: summary} from one eval-snapshots result file."""
    data = json.loads(Path(localize(str(result_json))).read_text())
    return data["summary"]


def objective(scores: Dict[str, Dict[str, dict]], epochs: Sequence[int],
              thresholds: Iterable[float]) -> Optional[dict]:
    """The objective over ``epochs``, or None while any of them is unscored.

    ``scores`` is {str(epoch): {str(rth): summary-with-"f1"}}.
    """
    epochs = [int(e) for e in epochs]
    if any(str(e) not in scores for e in epochs):
        return None
    by_rth = {}
    for t in thresholds:
        vals = []
        for e in epochs:
            f = scores[str(e)].get(str(t), {}).get("f1")
            vals.append(float(f) if f is not None else 0.0)
        by_rth[str(t)] = sum(vals) / len(vals)
    best_t = max(by_rth, key=lambda t: (by_rth[t], -float(t)))
    detail = {e: {k: scores[str(e)][best_t].get(k) for k in
                  ("f1", "mean_precision", "mean_recall", "weighted_f1", "n_scenes",
                   "n_background", "background_area")}
              for e in epochs}
    return {"value": by_rth[best_t], "rth": float(best_t), "by_rth": by_rth,
            "epochs": epochs, "at_best_rth": detail}


def rank(trials: Sequence[dict], budget_key: str) -> list:
    """Trials with an objective under ``budget_key``, best first.

    Ties (to 1e-12) break on the mean F1 over ALL thresholds, then on the trial
    key, so the order is total and reproducible. (Within one trial, a tie
    between thresholds goes to the lower RTh.)
    """
    scored = [t for t in trials if (t.get("objectives") or {}).get(budget_key)]

    def key(t):
        o = t["objectives"][budget_key]
        mean_all = sum(o["by_rth"].values()) / len(o["by_rth"])
        return (-round(o["value"], 12), -round(mean_all, 12), t["key"])

    return sorted(scored, key=key)
