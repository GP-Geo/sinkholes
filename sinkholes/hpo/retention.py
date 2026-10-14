"""The explicit retention policy. Dry run unless ``--apply``; every deletion logged.

Never deleted, by any rule: the ledger, the Optuna journal, every results.csv,
run_manifest.json, run_status.json, log, evaluation JSON (metrics are complete
and permanent), ``best.pt`` (best by patch Dice), ``last.pt`` (the final
weights) and ``best_scene.pt``.

Per trial, once an evaluation batch has been COLLECTED into the ledger:

* the snapshot that scored best on the scenes (max over thresholds of the
  val object F1, earliest epoch on a tie) is copied to
  ``checkpoints/best_scene.pt`` -- the best-by-scene-score checkpoint -- and
  recorded in the ledger;
* initial HPO trials (shared/tpe): the other evaluated snapshots are deleted;
  promoted trials, seed repeats and long runs keep every scheduled snapshot
  (they are the finalists, and long runs keep snapshots only at their
  scheduled evaluation epochs to begin with);
* confidence maps (votes/counts) are kept for the best-scene epoch and for
  every finalist epoch; the rest are deleted;
* ``interrupted.pt`` goes once the run is complete; ``resume_prev.pt`` too.

``resume.pt`` of an initial trial is the only way to promote it, so it is kept
until the promotion decision for its geometry exists, and deleted after that
only with ``--drop_unpromoted_resume``. Unevaluated snapshots are never
touched.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from . import jobs as J
from .controller import open_ledger, run_status
from .ledger import now
from .spec import Spec, localize

FINALIST_ORIGINS = ("seed_repeat", "long", "long_control")


def best_scene_epoch(trial: dict) -> Optional[dict]:
    best = None
    for e, summary in trial["scores"].items():
        for rth, s in summary.items():
            f = s.get("f1")
            if f is None:
                continue
            cand = (f, -int(e))
            if best is None or cand > best[0]:
                best = (cand, {"epoch": int(e), "f1": f, "rth": float(rth)})
    return best[1] if best else None


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def plan(spec: Spec, ledger, drop_unpromoted_resume: bool = False):
    """[(action, path, why, trial key)] -- action is 'delete' or 'copy:<target>'."""
    out = []
    decided = {d["geometry"] for d in ledger.data.get("decisions", []) if d["what"] == "promote"}
    promoted = {k for d in ledger.data.get("decisions", []) if d["what"] == "promote"
                for k in d["trials"]}
    for t in ledger.trials.values():
        run = Path(localize(t["run_dir"]))
        ck = run / "checkpoints"
        collected = sorted({e for b in t["evals"].values() if b.get("collected") for e in b["epochs"]})
        finalist = t["origin"] in FINALIST_ORIGINS or t["key"] in promoted
        best = best_scene_epoch(t)
        if best:
            src = Path(localize(J.snapshot_file(t, best["epoch"])))
            tgt = ck / "best_scene.pt"
            if src.exists() and t.get("best_scene", {}).get("epoch") != best["epoch"]:
                out.append((f"copy:{tgt}", src, f"best by scene score: epoch {best['epoch']} "
                                                 f"F1 {best['f1']:.4f} @ RTh {best['rth']}", t["key"]))
        for e in collected:
            snap = Path(localize(J.snapshot_file(t, e)))
            if not snap.exists() or finalist or (best and e == best["epoch"]):
                continue
            if t["origin"] in ("shared", "tpe") and t["geometry"] not in decided \
                    and e == max(t["snapshot_epochs"]):
                continue  # keep the last initial snapshot until promotion is decided
            out.append(("delete", snap, f"epoch {e} evaluated and not best-by-scene", t["key"]))
        for b in t["evals"].values():
            if not b.get("collected"):
                continue
            maps = Path(localize(b["out_dir"])) / "maps"
            for e in b["epochs"]:
                d = maps / f"e{e:03d}"
                if d.exists() and not finalist and not (best and e == best["epoch"]):
                    out.append(("delete", d, f"confidence maps of epoch {e}: not selected", t["key"]))
        complete = run_status(t).get("state") == "complete"
        for name in ("interrupted.pt", "resume_prev.pt"):
            if complete and (ck / name).exists():
                out.append(("delete", ck / name, "run complete", t["key"]))
        if (drop_unpromoted_resume and t["origin"] in ("shared", "tpe")
                and t["geometry"] in decided and t["key"] not in promoted
                and (ck / "resume.pt").exists()):
            out.append(("delete", ck / "resume.pt", "not promoted; promotion decided", t["key"]))
    return out


def retain(spec: Spec, apply: bool = False, drop_unpromoted_resume: bool = False) -> List[str]:
    ledger = open_ledger(spec)
    lines, freed = [], 0
    ctx = ledger.locked() if apply else _null()
    with ctx:
        if apply:
            open_ledger(spec)
        for action, path, why, key in plan(spec, ledger, drop_unpromoted_resume):
            size = sum(_size(p) for p in path.rglob("*")) if path.is_dir() else _size(path)
            if action.startswith("copy:"):
                lines.append(f"{'COPY' if apply else 'would copy'}  {path} -> {action[5:]}  ({why})")
                if apply:
                    tmp = Path(action[5:]).with_name(".best_scene.pt.tmp")
                    shutil.copy2(path, tmp)
                    tmp.replace(action[5:])
                    t = ledger.trial(key)
                    t["best_scene"] = best_scene_epoch(t) | {"copied": now()}
                continue
            freed += size
            lines.append(f"{'DELETE' if apply else 'would delete'}  {path}  "
                         f"({size / 2**30:.2f} GB; {why})")
            if apply:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                ledger.event("retain", path=str(path), why=why, bytes=size)
        if apply:
            ledger.save()
    lines.append(f"{'freed' if apply else 'would free'} {freed / 2**30:.1f} GB"
                 + ("" if apply else " -- dry run, nothing changed (add --apply)"))
    return lines


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
