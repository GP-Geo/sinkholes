"""The study controller: what to run next, and the record of what ran.

    sinkholes hpo init     --spec S            create the ledger, the Optuna studies, the six
                                                shared trials per geometry (nothing submitted)
    sinkholes hpo status   --spec S            the table: every trial, its stage and scores
    sinkholes hpo approve  --spec S --phase P  allow submissions of phase P (recorded)
    sinkholes hpo step     --spec S            DRY RUN: collect results, list next actions
    sinkholes hpo step     --spec S --submit   ... and submit the approved ones
    sinkholes hpo collect  --spec S            only read finished evaluations into the ledger
    sinkholes hpo adopt    --spec S            take over the finished trials the spec's
                                                'adopt' section names (see :func:`adopt`)

Phases, in the order the study reaches them:

    prep         utility jobs: the min/max cache, the ctx50 GPU profile
    baseline     shared config 0 (the current configuration), both geometries
    shared       shared configs 1-5, both geometries
    tpe          two TPE-proposed trials per geometry, asked once its six are scored
    promote      the best two per geometry continued 30 -> 60 epochs (full-state resume)
    seeds        the best promoted config per geometry at seeds 7 and 123, 60 epochs
    long         300-epoch cosine runs of each geometry's selected config, seed 42
    long_seeds   ... at seeds 7 and 123
    long_control the optional constant-LR diagnostic run

Nothing is ever submitted for a phase that has not been approved, and ``step``
without ``--submit`` changes nothing at all.

Duplicate submissions are prevented three ways: the ledger lock (one
controller at a time), deterministic job names per action checked against both
the ledger and ``bjobs`` before every bsub, and a refusal to resubmit a
training chunk that made no progress (or an evaluation that wrote nothing)
unless ``--retry KEY`` says so.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..training.provenance import source_fingerprint
from ..training.schedules import parse_epoch_spec
from . import jobs as J
from . import lsf
from . import objective as O
from .ledger import Ledger, LedgerError, now, write_protect_marker
from .spec import Spec, load_spec, localize, to_cluster

PHASES = ("prep", "baseline", "shared", "tpe", "promote", "seeds",
          "long", "long_seeds", "long_control")

#: Initial trials' Optuna budget key and the promoted one.
E30, E60 = "e30", "e60"


# -- trial records ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e")


def make_slug(params: dict, seed: int) -> str:
    return f"lr{_fmt(params['lr'])}_wd{_fmt(params['weight_decay'])}_pw{params['pos_w']:g}_seed{seed}"


def new_trial(spec: Spec, *, geometry: str, short: str, origin: str, phase: str, params: dict,
              seed: int, target_epochs: int, snapshot_epochs, eval_batches, budgets: dict,
              schedule: dict, optuna=None, extra=None) -> dict:
    slug = make_slug(params, seed)
    t = {
        "key": f"{geometry}/{short}", "geometry": geometry, "origin": origin, "phase": phase,
        "params": {"lr": float(params["lr"]), "weight_decay": float(params["weight_decay"]),
                   "pos_w": params["pos_w"]},
        "seed": int(seed), "slug": f"{spec.name}_{geometry}_{short}",
        "run_dir": f"{spec.root}/runs/{geometry}/{short}_{slug}",
        # The full-state recovery point every continuation (chunk or promotion)
        # resumes from, and its rotated previous copy.
        "resume_path": f"{spec.root}/runs/{geometry}/{short}_{slug}/checkpoints/resume.pt",
        "resume_prev_path": f"{spec.root}/runs/{geometry}/{short}_{slug}/checkpoints/resume_prev.pt",
        "schedule": dict(schedule), "target_epochs": int(target_epochs),
        "snapshot_epochs": sorted(set(int(e) for e in snapshot_epochs)),
        "eval_batches": [list(map(int, b)) for b in eval_batches] if eval_batches != "dynamic"
        else "dynamic",
        "budgets": {k: list(map(int, v)) for k, v in budgets.items()},
        "optuna": optuna, "jobs": [], "evals": {}, "scores": {}, "objectives": {},
        "told": {}, "created": now(),
    }
    if extra:
        t.update(extra)
    return t


def hpo_schedule(spec: Spec) -> dict:
    s = spec.raw["hpo_schedule"]
    return {"lr_schedule": s["lr_schedule"], "lr_patience": s["lr_patience"],
            "lr_factor": s["lr_factor"], "min_lr": s["min_lr"]}


# -- init ------------------------------------------------------------------------------------

def init(spec: Spec, spec_path: str, repo: Optional[str] = None) -> Ledger:
    from . import optuna_io as Q

    ledger = Ledger(spec.root)
    if ledger.exists():
        raise LedgerError(f"{ledger.path} already exists -- this study was initialised; "
                          f"use `status`/`step`, or choose a new study name and root")
    if Path(localize(Q.storage_path(spec))).exists():
        raise LedgerError(
            f"{Q.storage_path(spec)} exists but no ledger does: an earlier `init` stopped "
            f"half-way and its Optuna trials would be duplicated. Remove {spec.root} (nothing "
            f"has been submitted from it) and run init again.")
    root = Path(localize(spec.root))
    for d in ("runs", "evals", "jobs", "cache", "reports", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    write_protect_marker(root, spec.name)
    shutil.copy(localize(spec_path), root / "spec.json")
    # The partitions the jobs read are frozen into the study (see jobs.study_partition).
    (root / "inputs").mkdir(exist_ok=True)
    for which in ("partition", "test_partition"):
        shutil.copy(localize(spec.partition if which == "partition" else spec.test_partition),
                    localize(J.study_partition(spec, which)))
    if repo:
        spec.repo_override = to_cluster(repo)
    with ledger.locked():
        ledger.data = {
            "study": spec.name, "spec_sha256": spec.sha256, "spec_path": to_cluster(spec_path),
            "repo": to_cluster(repo or spec.repo),
            "created": now(), "source_at_init": source_fingerprint(),
            "approved_phases": [], "approvals": [], "trials": {}, "decisions": [],
            "utility_jobs": [], "locked": {},
        }
        ev = spec.evaluation
        initial_epochs = parse_epoch_spec(ev["initial_epochs"])
        for g in spec.geometries:
            study = Q.open_study(spec, g, E30, create=True)
            Q.open_study(spec, g, E60, create=True)
            numbers = Q.ask_enqueued(study, spec, spec.shared)
            for i, (params, number) in enumerate(zip(spec.shared, numbers)):
                t = new_trial(
                    spec, geometry=g, short=f"s{i}", origin="shared",
                    phase="baseline" if i == spec.raw.get("baseline_index", 0) else "shared",
                    params=params, seed=spec.budget["initial_seed"],
                    target_epochs=spec.budget["initial_epochs"],
                    snapshot_epochs=initial_epochs, eval_batches=[initial_epochs],
                    budgets={E30: ev["objective_epochs"]}, schedule=hpo_schedule(spec),
                    optuna={"study": Q.study_name(spec, g, E30), "number": number},
                    extra={"design_index": i})
                ledger.trials[t["key"]] = t
        ledger.save()
        ledger.event("init", spec_sha256=spec.sha256, trials=sorted(ledger.trials))
    return ledger


# -- refresh / collect ------------------------------------------------------------------------

def run_status(trial: dict) -> dict:
    path = Path(localize(trial["run_dir"])) / "run_status.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"state": "not_started", "completed_epoch": 0}


def eval_dir(spec: Spec, trial: dict, epochs) -> str:
    name = "e" + "-".join(f"{e:03d}" for e in epochs)
    return f"{spec.root}/evals/{trial['geometry']}/{trial['key'].split('/')[1]}/{name}"


def collect(spec: Spec, ledger: Ledger, tell: bool = True) -> List[str]:
    """Read finished evaluation batches into the ledger; compute objectives.

    Idempotent. With ``tell`` the objectives are also recorded in Optuna.
    Returns a list of human-readable notes.
    """
    notes = []
    for t in ledger.trials.values():
        for name, batch in t["evals"].items():
            if batch.get("collected"):
                continue
            out = Path(localize(batch["out_dir"]))
            files = {e: out / f"e{e:03d}.json" for e in batch["epochs"]}
            if not all(f.exists() for f in files.values()):
                continue
            for e, f in files.items():
                data = json.loads(f.read_text())
                if data["checkpoint"]["path"] != to_cluster(J.snapshot_file(t, e)) and \
                        localize(data["checkpoint"]["path"]) != localize(J.snapshot_file(t, e)):
                    raise LedgerError(f"{f} scored {data['checkpoint']['path']}, not "
                                      f"{t['key']} epoch {e}")
                t["scores"][str(e)] = data["summary"]
                t.setdefault("checkpoint_sha256", {})[str(e)] = data["checkpoint"]["sha256"]
            batch["collected"] = now()
            notes.append(f"{t['key']}: collected epochs {batch['epochs']}")
            ledger.event("collect", trial=t["key"], epochs=batch["epochs"])
        for budget, epochs in t["budgets"].items():
            if budget in t["objectives"]:
                continue
            obj = O.objective(t["scores"], epochs, spec.evaluation["rth"])
            if obj is not None:
                t["objectives"][budget] = obj
                notes.append(f"{t['key']}: {budget} objective {obj['value']:.4f} at RTh {obj['rth']}")
                ledger.event("objective", trial=t["key"], budget=budget, objective=obj)
    if tell:
        notes += tell_optuna(spec, ledger)
    return notes


def tell_optuna(spec: Spec, ledger: Ledger) -> List[str]:
    from . import optuna_io as Q

    notes, studies = [], {}

    def study(g, budget):
        if (g, budget) not in studies:
            studies[g, budget] = Q.open_study(spec, g, budget)
        return studies[g, budget]

    for t in ledger.trials.values():
        attrs = lambda b: {"trial_key": t["key"], "origin": t["origin"], "seed": t["seed"],
                           "rth": t["objectives"][b]["rth"],
                           "by_rth": t["objectives"][b]["by_rth"], "run_dir": t["run_dir"]}
        if t.get("optuna") and E30 in t["objectives"] and not t["told"].get(E30):
            Q.tell(study(t["geometry"], E30), t["optuna"]["number"], t["objectives"][E30]["value"],
                   attrs(E30))
            t["told"][E30] = now()
            notes.append(f"{t['key']}: told Optuna {E30}")
        if E60 in t["objectives"] and not t["told"].get(E60) and t["origin"] in (
                "shared", "tpe", "seed_repeat"):
            number = Q.add_finished(study(t["geometry"], E60), spec, t["params"],
                                    t["objectives"][E60]["value"], attrs(E60))
            t["told"][E60] = now()
            t.setdefault("optuna_e60", number)
            notes.append(f"{t['key']}: recorded {E60} in Optuna")
    return notes


# -- transitions -----------------------------------------------------------------------------

def _initial(ledger, g):
    return [t for t in ledger.trials.values()
            if t["geometry"] == g and t["origin"] in ("shared", "tpe")]


def _decided(ledger, what, g) -> Optional[dict]:
    for d in ledger.data["decisions"]:
        if d["what"] == what and d["geometry"] == g:
            return d
    return None


def transitions(spec: Spec, ledger: Ledger, apply: bool) -> List[str]:
    """Advance each geometry as far as its scores and the approvals allow."""
    from . import optuna_io as Q

    notes = []
    ev, b = spec.evaluation, spec.budget
    for g in spec.geometries:
        shared = [t for t in ledger.trials.values() if t["geometry"] == g and t["origin"] == "shared"]
        tpe = [t for t in ledger.trials.values() if t["geometry"] == g and t["origin"] == "tpe"]

        # -- TPE proposals, once all six shared configs are scored and told
        if not tpe and shared and all(t["told"].get(E30) for t in shared):
            if not ledger.approved("tpe"):
                notes.append(f"{g}: six shared trials scored -- TPE trials wait for `approve --phase tpe`")
            elif apply:
                study = Q.open_study(spec, g, E30)
                for i, prop in enumerate(Q.ask_tpe(study, spec, int(spec.tpe["n_trials"]))):
                    t = new_trial(spec, geometry=g, short=f"t{i}", origin="tpe", phase="tpe",
                                  params=prop["params"], seed=b["initial_seed"],
                                  target_epochs=b["initial_epochs"],
                                  snapshot_epochs=parse_epoch_spec(ev["initial_epochs"]),
                                  eval_batches=[parse_epoch_spec(ev["initial_epochs"])],
                                  budgets={E30: ev["objective_epochs"]}, schedule=hpo_schedule(spec),
                                  optuna={"study": Q.study_name(spec, g, E30), "number": prop["number"]})
                    ledger.trials[t["key"]] = t
                    notes.append(f"{g}: TPE proposed {t['key']} {t['params']}")
                ledger.data["decisions"].append({"time": now(), "what": "tpe", "geometry": g,
                                                 "trials": [f"{g}/t{i}" for i in range(int(spec.tpe['n_trials']))]})
                ledger.event("tpe", geometry=g)
            else:
                notes.append(f"{g}: would ask Optuna for {spec.tpe['n_trials']} TPE trials")

        # -- promotion of the best two initial trials to 60 epochs
        initial = _initial(ledger, g)
        expected = len(spec.shared) + int(spec.tpe["n_trials"])
        if (len(initial) == expected and all(E30 in t["objectives"] for t in initial)
                and not _decided(ledger, "promote", g)):
            ranked = O.rank(initial, E30)
            top = ranked[: int(b["promote_top"])]
            if not ledger.approved("promote"):
                notes.append(f"{g}: initial trials scored; best {[t['key'] for t in top]} "
                             f"wait for `approve --phase promote`")
            elif apply:
                for t in top:
                    promote(spec, t)
                ledger.data["decisions"].append({
                    "time": now(), "what": "promote", "geometry": g,
                    "trials": [t["key"] for t in top],
                    "basis": {t["key"]: t["objectives"][E30] for t in ranked}})
                ledger.event("promote", geometry=g, trials=[t["key"] for t in top])
                notes.append(f"{g}: promoted {[t['key'] for t in top]} to {b['promoted_epochs']} epochs")
            else:
                notes.append(f"{g}: would promote {[t['key'] for t in top]}")

        # -- seed repeats of the best promoted configuration
        dec = _decided(ledger, "promote", g)
        if dec and not _decided(ledger, "seeds", g):
            promoted = [ledger.trial(k) for k in dec["trials"]]
            if all(E60 in t["objectives"] for t in promoted):
                best = O.rank(promoted, E60)[0]
                if not ledger.approved("seeds"):
                    notes.append(f"{g}: {best['key']} is the best promoted config -- seed repeats "
                                 f"wait for `approve --phase seeds`")
                elif apply:
                    for seed in b["repeat_seeds"]:
                        epochs = parse_epoch_spec(ev["seed_epochs"])
                        t = new_trial(spec, geometry=g, short=f"{best['key'].split('/')[1]}-seed{seed}",
                                      origin="seed_repeat", phase="seeds", params=best["params"],
                                      seed=seed, target_epochs=b["promoted_epochs"],
                                      snapshot_epochs=epochs,
                                      eval_batches=[epochs[:len(epochs) // 2], epochs[len(epochs) // 2:]],
                                      budgets={E30: ev["objective_epochs"],
                                               E60: ev["promoted_objective_epochs"]},
                                      schedule=hpo_schedule(spec), extra={"repeat_of": best["key"]})
                        ledger.trials[t["key"]] = t
                    ledger.data["decisions"].append({
                        "time": now(), "what": "seeds", "geometry": g, "config_of": best["key"],
                        "basis": {t["key"]: t["objectives"][E60] for t in promoted}})
                    ledger.event("seeds", geometry=g, config_of=best["key"])
                    notes.append(f"{g}: seed repeats of {best['key']} created")
                else:
                    notes.append(f"{g}: would create seed repeats of {best['key']}")

        # -- the locked configuration and deployment threshold
        sdec = _decided(ledger, "seeds", g)
        if sdec and g not in ledger.data["locked"]:
            order = list(b["repeat_seeds"])
            group = [ledger.trial(sdec["config_of"])] + sorted(
                (t for t in ledger.trials.values()
                 if t["geometry"] == g and t.get("repeat_of") == sdec["config_of"]),
                key=lambda t: order.index(t["seed"]) if t["seed"] in order else len(order))
            if all(E60 in t["objectives"] for t in group):
                lock = lock_threshold(spec, group)
                if apply:
                    ledger.data["locked"][g] = lock
                    ledger.data["decisions"].append({"time": now(), "what": "lock", "geometry": g,
                                                     **lock})
                    ledger.event("lock", geometry=g, lock=lock)
                notes.append(f"{g}: {'locked' if apply else 'would lock'} RTh {lock['rth']} "
                             f"for {lock['config_of']} (val F1 {lock['value']:.4f} over "
                             f"{len(group)} seeds)")
    return notes


def promote(spec: Spec, trial: dict) -> None:
    ev, b = spec.evaluation, spec.budget
    extra = parse_epoch_spec(ev["promoted_epochs"])
    trial["target_epochs"] = int(b["promoted_epochs"])
    trial["snapshot_epochs"] = sorted(set(trial["snapshot_epochs"]) | set(extra))
    trial["eval_batches"].append(extra)
    trial["budgets"][E60] = list(ev["promoted_objective_epochs"])
    trial["phase"] = "promote"
    trial["promoted"] = now()


def lock_threshold(spec: Spec, group: List[dict]) -> dict:
    """One RTh for the configuration: best mean val F1 over its seeds at epochs 50/60."""
    rths = [str(t) for t in spec.evaluation["rth"]]
    by_rth = {r: sum(t["objectives"][E60]["by_rth"][r] for t in group) / len(group) for r in rths}
    best = max(by_rth, key=lambda r: (by_rth[r], -float(r)))
    return {"config_of": group[0]["key"], "params": group[0]["params"],
            "rth": float(best), "value": by_rth[best], "by_rth": by_rth,
            "seeds": [t["seed"] for t in group], "trials": [t["key"] for t in group],
            "epochs": spec.evaluation["promoted_objective_epochs"],
            "rule": "argmax over RTh of the mean over seeds of the epoch-50/60 mean val object F1; "
                    "lock before any test-set scoring"}


def create_long(spec: Spec, ledger: Ledger, g: str, seeds, phase: str,
                constant: bool = False) -> List[str]:
    """300-epoch runs of a geometry's locked configuration."""
    lock = ledger.data["locked"].get(g)
    if not lock:
        raise LedgerError(f"{g}: no locked configuration yet -- finish the HPO phases first")
    L = spec.long
    epochs = parse_epoch_spec(L["eval_epochs"])
    lr = float(lock["params"]["lr"])
    sched = ({"lr_schedule": "constant", "min_lr": lr} if constant else
             {"lr_schedule": "cosine", "lr_horizon": int(L["epochs"]),
              "min_lr": lr * float(L["lr_floor_ratio"])})
    made = []
    for seed in seeds:
        short = f"long{'-const' if constant else ''}-seed{seed}"
        key = f"{g}/{short}"
        if key in ledger.trials:
            continue
        t = new_trial(spec, geometry=g, short=short, origin="long_control" if constant else "long",
                      phase=phase, params=lock["params"], seed=seed, target_epochs=L["epochs"],
                      snapshot_epochs=epochs, eval_batches="dynamic", budgets={},
                      schedule=sched, extra={"config_of": lock["config_of"]})
        ledger.trials[key] = t
        made.append(key)
    ledger.event("long", geometry=g, trials=made)
    return made


# -- actions ---------------------------------------------------------------------------------

def _active(job: dict, lsf_state: Dict[str, dict]) -> bool:
    info = lsf_state.get(str(job.get("job_id")))
    return bool(info and info["stat"] in lsf.ACTIVE_STATES)


#: A long run's snapshots are evaluated in batches of at least this many while
#: it trains (one scene load serves the batch), and whatever is left once the
#: training chunk has stopped.
DYNAMIC_MIN_BATCH = 3


def _pending_eval_batches(t: dict, training_active: bool) -> List[List[int]]:
    if t["eval_batches"] != "dynamic":
        return [b for b in t["eval_batches"] if "e" + "-".join(f"{e:03d}" for e in b) not in t["evals"]]
    done = {e for b in t["evals"].values() for e in b["epochs"]}
    have = [e for e in t["snapshot_epochs"]
            if e not in done and Path(localize(J.snapshot_file(t, e))).exists()]
    if len(have) >= DYNAMIC_MIN_BATCH or (have and not training_active):
        return [have]
    return []


def plan(spec: Spec, ledger: Ledger, lsf_state: Dict[str, dict], retry=()) -> List[dict]:
    """The next action per trial: a training chunk, an evaluation, or a blocked note."""
    actions = []
    unknown = "_unavailable" in lsf_state
    for t in ledger.trials.values():
        status = run_status(t)
        t["last_status"] = {k: status.get(k) for k in ("state", "completed_epoch", "updated")}
        done = int(status.get("completed_epoch") or 0)
        train_jobs = [j for j in t["jobs"] if j["kind"] == "train"]
        # Only jobs that reached LSF can have run; a failed bsub is not a stall.
        launched = [j for j in train_jobs if j.get("job_id")]
        running = [j for j in launched if _active(j, lsf_state)]

        if unknown and launched and done < t["target_epochs"]:
            # From the Mac there is no bjobs: a submitted job may be pending or
            # running, so nothing can be decided about it here.
            actions.append({"kind": "blocked", "trial": t["key"], "phase": t["phase"],
                            "why": f"{done}/{t['target_epochs']} epochs, run status "
                                   f"'{status.get('state')}'; LSF state unknown here -- run "
                                   f"`step` on a WEXAC login node"})
        elif done < t["target_epochs"] and not running:
            last = launched[-1] if launched else None
            stalled = (last is not None and done <= int(last.get("start_epoch", 0))
                       and status.get("state") != "paused")
            if status.get("state") == "failed" and t["key"] not in retry:
                actions.append({"kind": "blocked", "trial": t["key"], "phase": t["phase"],
                                "why": f"training failed: {status.get('error', '?')} -- inspect, "
                                       f"then --retry {t['key']}"})
            elif stalled and t["key"] not in retry:
                actions.append({"kind": "blocked", "trial": t["key"], "phase": t["phase"],
                                "why": f"job {last.get('job_id')} ended without completing an "
                                       f"epoch past {done} -- inspect its log, then --retry {t['key']}"})
            else:
                actions.append({"kind": "train", "trial": t["key"], "phase": t["phase"],
                                "chunk": len(train_jobs) + 1, "from_epoch": done})

        for batch in _pending_eval_batches(t, training_active=bool(running)):
            missing = [e for e in batch if not Path(localize(J.snapshot_file(t, e))).exists()]
            if missing:
                continue  # not trained that far yet
            actions.append({"kind": "eval", "trial": t["key"], "phase": t["phase"],
                            "epochs": batch})
        for name, b in t["evals"].items():
            if b.get("collected"):
                continue
            job = b.get("job") or {}
            if job and not _active(job, lsf_state) and not lsf_state.get("_unavailable"):
                out = Path(localize(b["out_dir"]))
                if not all((out / f"e{e:03d}.json").exists() for e in b["epochs"]):
                    if t["key"] in retry:
                        actions.append({"kind": "eval", "trial": t["key"], "phase": t["phase"],
                                        "epochs": b["epochs"], "retry_of": name})
                    else:
                        actions.append({"kind": "blocked", "trial": t["key"], "phase": t["phase"],
                                        "why": f"evaluation {name} (job {job.get('job_id')}) ended "
                                               f"without results -- it resumes from its finished "
                                               f"scenes: --retry {t['key']}"})
    return actions


def _job_name(spec: Spec, t: dict, kind: str, suffix: str) -> str:
    return f"{spec.name}-{t['key'].replace('/', '-')}-{kind}-{suffix}"


def execute(spec: Spec, ledger: Ledger, actions: List[dict], submit: bool,
            only=(), coverage_ok=None) -> List[str]:
    """Submit (or describe) the approved actions. Returns printable lines."""
    lines = []
    repo = ledger.data["repo"]
    logs = f"{spec.root}/logs"
    for a in actions:
        if only and a["trial"] not in only:
            continue
        t = ledger.trial(a["trial"])
        if a["kind"] == "blocked":
            lines.append(f"BLOCKED  {a['trial']}: {a['why']}")
            continue
        approved = ledger.approved(a["phase"])
        if a["kind"] == "train":
            res = lsf.Resources.from_spec(spec.resources("train", t["geometry"]))
            budget = float(spec.resources("train", t["geometry"])["time_budget_hours"])
            name = _job_name(spec, t, "train", f"c{a['chunk']}")
            script_path = f"{spec.root}/jobs/{name}.sh"
            text = J.train_script(spec, repo, t, a["chunk"], budget)
            what = f"train {a['trial']} chunk {a['chunk']} (epochs {a['from_epoch']}->{t['target_epochs']})"
        else:
            res = lsf.Resources.from_spec(spec.resources("eval", t["geometry"]))
            out_dir = eval_dir(spec, t, a["epochs"])
            suffix = "e" + "-".join(f"{e:03d}" for e in a["epochs"])
            n_prior = sum(1 for j in t["jobs"] if j["kind"] == "eval" and j.get("batch") == suffix)
            name = _job_name(spec, t, "eval", suffix + (f"-r{n_prior}" if n_prior else ""))
            script_path = f"{spec.root}/jobs/{name}.sh"
            text = J.eval_script(spec, repo, t, a["epochs"], out_dir)
            what = f"evaluate {a['trial']} epochs {a['epochs']}"
            if coverage_ok is not None:
                problem = coverage_ok(t["geometry"])
                if problem:
                    lines.append(f"BLOCKED  {what}: {problem}")
                    continue
        argv = lsf.bsub_argv(name, res, logs)
        cmd = lsf.render_command(argv, script_path)
        if not approved:
            lines.append(f"NEEDS APPROVAL ({a['phase']})  {what}\n    {cmd}")
            continue
        if not submit:
            lines.append(f"WOULD SUBMIT ({a['phase']})  {what}\n    {cmd}")
            continue
        # -- duplicate guards: the ledger, then LSF itself
        if any(j.get("job_id") for j in ledger.jobs_named(name)):
            lines.append(f"SKIP  {name}: already submitted (ledger)")
            continue
        live = lsf.query(names=[name])
        if any(v["stat"] in lsf.ACTIVE_STATES for v in live.values()):
            lines.append(f"SKIP  {name}: LSF already has it {live}")
            continue
        if a["kind"] == "train":
            Path(localize(t["run_dir"])).mkdir(parents=True, exist_ok=True)
        J.write_script(localize(script_path), text)
        Path(localize(logs)).mkdir(parents=True, exist_ok=True)
        job = {"kind": a["kind"], "name": name, "script": script_path, "argv": argv,
               "submitted": now(), "state": "submitting"}
        if a["kind"] == "train":
            job.update(chunk=a["chunk"], start_epoch=a["from_epoch"], target_epochs=t["target_epochs"])
        else:
            job.update(batch=suffix, epochs=a["epochs"], out_dir=out_dir)
        t["jobs"].append(job)
        ledger.save()                      # recorded BEFORE bsub: a crash cannot lose it
        try:
            job["job_id"] = lsf.submit(argv, localize(script_path))
            job["state"] = "submitted"
        except lsf.SubmitError as exc:
            job["state"] = "submit_failed"
            job["error"] = str(exc)
            ledger.save()
            lines.append(f"FAILED  {what}: {exc}")
            continue
        if a["kind"] == "eval":
            t["evals"][suffix] = {"epochs": a["epochs"], "out_dir": out_dir, "job": job,
                                  "submitted": job["submitted"]}
        ledger.save()
        ledger.event("submit", trial=t["key"], name=name, job_id=job["job_id"], job_kind=a["kind"])
        lines.append(f"SUBMITTED  {what}: LSF job {job['job_id']} ({name})")
    return lines


# -- adoption --------------------------------------------------------------------------------

#: What an adopting study's spec may change relative to its source: names,
#: places, prose, the adoption itself -- and evaluation.rth, which may only grow
#: (the source's thresholds are what the re-scored maps are checked against).
ADOPT_MAY_DIFFER = ("study", "root", "description", "adopt")

#: Trial fields that must agree for a finished run to stand for the new trial.
ADOPT_SAME = ("geometry", "origin", "phase", "params", "seed", "target_epochs",
              "snapshot_epochs", "eval_batches", "budgets", "schedule", "design_index")


def _comparable(raw: dict) -> dict:
    out = {k: v for k, v in raw.items() if k not in ADOPT_MAY_DIFFER}
    out["evaluation"] = {k: v for k, v in raw["evaluation"].items() if k != "rth"}
    return out


def adoption_problems(spec: Spec, src: Spec) -> List[str]:
    """Why ``src``'s runs could not stand for this study's (empty when they can)."""
    a, b = _comparable(src.raw), _comparable(spec.raw)
    probs = [f"spec section {k!r} differs" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if not set(src.evaluation["rth"]) <= set(spec.evaluation["rth"]):
        probs.append(f"evaluation.rth {spec.evaluation['rth']} drops some of the source's "
                     f"{src.evaluation['rth']}")
    for which in ("partition", "test_partition"):
        mine, theirs = (Path(localize(J.study_partition(s, which))) for s in (spec, src))
        if not mine.exists() or mine.read_bytes() != theirs.read_bytes():
            probs.append(f"the frozen {which} {mine} is not byte-identical to {theirs}")
    return probs


def _adopt_trial(spec: Spec, src: Spec, t: dict, s: dict) -> List[str]:
    from ..inference.snapshot_eval import rescore_saved, sha256_of

    key = t["key"]
    diff = [f for f in ADOPT_SAME if t.get(f) != s.get(f)]
    if diff:
        raise LedgerError(f"{key}: {diff} differ from {src.name}'s {key}")
    if t["jobs"] or t["evals"] or t["scores"] or t["told"]:
        raise LedgerError(f"{key} already has jobs or scores here -- it cannot be adopted")
    st = run_status(s)
    if st.get("state") != "complete" or int(st.get("completed_epoch") or 0) != s["target_epochs"]:
        raise LedgerError(f"{src.name} {key} is not complete ({st.get('state')}, "
                          f"{st.get('completed_epoch')}/{s['target_epochs']})")
    pending = [n for n, b in s["evals"].items() if not b.get("collected")]
    if pending or len(s["evals"]) != len(s["eval_batches"]):
        raise LedgerError(f"{src.name} {key}: evaluations {pending or 'missing'} not collected")
    for e, sha in s["checkpoint_sha256"].items():
        if sha256_of(localize(J.snapshot_file(s, int(e)))) != sha:
            raise LedgerError(f"{src.name} {key}: epoch {e}'s snapshot is not the one it scored")

    evals = {}
    for name, b in s["evals"].items():
        out_dir = eval_dir(spec, t, b["epochs"])
        rescore_saved(localize(b["out_dir"]), localize(out_dir), [f"e{e:03d}" for e in b["epochs"]],
                      spec.evaluation["rth"], localize(f"{J.cache_dir(src)}/gt"))
        evals[name] = {"epochs": b["epochs"], "out_dir": out_dir, "rescored_from": b["out_dir"]}
    # The run is shared, not copied: a promotion resumes it in place, under the
    # name and partition path its resume fingerprint was written with.
    for f in ("run_dir", "resume_path", "resume_prev_path", "slug"):
        t[f] = s[f]
    t["partition_file"] = J.study_partition(src)
    t["evals"] = evals
    t["adopted_from"] = {"study": src.name, "root": src.root, "spec_sha256": src.sha256,
                         "time": now(), "jobs": s["jobs"],
                         "checkpoint_sha256": s["checkpoint_sha256"],
                         "objectives": s["objectives"]}
    return [f"{key}: adopted {src.name}'s run {s['run_dir']}; evaluations {sorted(evals)} "
            f"re-scored at RTh {spec.evaluation['rth']}"]


def _copy_missing(src_dir: Path, dst_dir: Path) -> int:
    n = 0
    for f in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        to = dst_dir / f.relative_to(src_dir)
        if not to.exists():
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, to)
            n += 1
    return n


def adopt(spec: Spec) -> List[str]:
    """Take over the finished trials the spec's ``adopt`` section names.

    For a protocol change that leaves training untouched (so far: a wider RTh
    list). Each named trial of the source study must be complete, collected,
    and configured exactly as its namesake here, and the two specs may differ
    only as :data:`ADOPT_MAY_DIFFER` allows. The run directory is shared, not
    copied; its evaluations are re-scored from their saved vote maps at this
    study's thresholds -- no GPU, and every threshold the source scored must
    reproduce exactly. The source's caches are copied, and the source study is
    marked superseded: its controller then refuses every change, so nothing
    there can touch the runs this study now owns.

    Idempotent: adopted trials are skipped. Collection and the Optuna tell
    happen at the next ``step --submit``, as for any finished evaluation.
    """
    a = spec.raw.get("adopt")
    if not a:
        raise LedgerError(f"{spec.path} has no 'adopt' section")
    src = load_spec(a["from_spec"])
    probs = adoption_problems(spec, src)
    if probs:
        raise LedgerError(f"cannot adopt from {src.name}: " + "; ".join(probs))
    notes = []
    ledger = open_ledger(spec)
    with ledger.locked():
        open_ledger(spec)
        src_ledger = Ledger(src.root)
        with src_ledger.locked(allow_superseded=True):
            open_ledger(src)
            sup = src_ledger.data.get("superseded_by")
            if sup and sup["study"] != spec.name:
                raise LedgerError(f"{src.name} was already superseded by {sup['study']}")
            for key in a["trials"]:
                t = ledger.trial(key)
                if t.get("adopted_from"):
                    notes.append(f"{key}: already adopted")
                    continue
                notes += _adopt_trial(spec, src, t, src_ledger.trial(key))
                ledger.save()
                ledger.event("adopt", trial=key, from_study=src.name, run_dir=t["run_dir"],
                             evals=t["evals"])
            n = _copy_missing(Path(localize(J.cache_dir(src))), Path(localize(J.cache_dir(spec))))
            notes.append(f"copied {n} cache file(s) from {src.name}")
            if not sup:
                src_ledger.data["superseded_by"] = {
                    "study": spec.name, "root": spec.root, "time": now(),
                    "adopted": list(a["trials"]), "note": a.get("note", "")}
                src_ledger.save()
                src_ledger.event("superseded", by=spec.name, adopted=list(a["trials"]))
                notes.append(f"{src.name} is now closed (superseded by {spec.name})")
    return notes


# -- status ----------------------------------------------------------------------------------

def status_rows(spec: Spec, ledger: Ledger) -> List[dict]:
    rows = []
    for t in ledger.trials.values():
        st = run_status(t)
        o30 = t["objectives"].get(E30)
        o60 = t["objectives"].get(E60)
        rows.append({
            "trial": t["key"], "phase": t["phase"], "seed": t["seed"],
            "lr": _fmt(t["params"]["lr"]), "wd": _fmt(t["params"]["weight_decay"]),
            "pos_w": t["params"]["pos_w"],
            "epochs": f"{st.get('completed_epoch', 0)}/{t['target_epochs']}",
            "state": st.get("state", "not_started"),
            "scored": ",".join(sorted(t["scores"], key=int)) or "-",
            "e30": f"{o30['value']:.4f}@{o30['rth']:g}" if o30 else "-",
            "e60": f"{o60['value']:.4f}@{o60['rth']:g}" if o60 else "-",
            "jobs": len(t["jobs"]),
        })
    return rows


def format_table(rows: List[dict]) -> str:
    if not rows:
        return "(no trials)"
    cols = list(rows[0])
    width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    head = "  ".join(c.ljust(width[c]) for c in cols)
    sep = "  ".join("-" * width[c] for c in cols)
    body = ["  ".join(str(r[c]).ljust(width[c]) for c in cols) for r in rows]
    return "\n".join([head, sep, *body])


# -- entry points used by the CLI --------------------------------------------------------------

def open_ledger(spec: Spec) -> Ledger:
    ledger = Ledger(spec.root).load()
    spec.repo_override = ledger.data.get("repo")
    if ledger.data.get("spec_sha256") != spec.sha256:
        raise LedgerError(
            f"the spec at {spec.path} is not the one this study was initialised with "
            f"(sha256 {spec.sha256[:12]} vs {ledger.data.get('spec_sha256', '?')[:12]}). A study's "
            f"specification is fixed; start a new study to change it.")
    return ledger


def lsf_snapshot(ledger: Ledger) -> Dict[str, dict]:
    ids = [j.get("job_id") for t in ledger.trials.values() for j in t["jobs"] if j.get("job_id")]
    ids += [j.get("job_id") for j in ledger.data.get("utility_jobs", []) if j.get("job_id")]
    if not lsf.available():
        return {"_unavailable": {"stat": "?", "name": "bjobs not on PATH"}}
    return lsf.query(job_ids=ids)


def reconcile_submitting(ledger: Ledger) -> List[str]:
    """Jobs recorded as 'submitting' (the controller died around bsub): ask LSF.

    The ledger entry is written BEFORE bsub, so a crash can leave an entry
    without a job id whose job may nevertheless be queued. LSF is asked by the
    entry's unique job name; a match is adopted, no match means the job never
    reached LSF. Either way the action can then be decided without a duplicate.
    """
    notes = []
    entries = [j for t in ledger.trials.values() for j in t["jobs"]] + \
        ledger.data.get("utility_jobs", [])
    for j in entries:
        if j.get("state") == "submitting" and not j.get("job_id"):
            found = lsf.query(names=[j["name"]])
            if found:
                j["job_id"] = sorted(found)[-1]
                j["state"] = "submitted (adopted from LSF after a controller crash)"
            else:
                j["state"] = "lost (never reached LSF)"
            notes.append(f"reconciled {j['name']}: {j['state']}")
            ledger.event("reconcile", name=j["name"], state=j["state"], job_id=j.get("job_id"))
    return notes


def step(spec: Spec, *, submit: bool, only=(), retry=(), break_lock=False,
         coverage_check=True) -> List[str]:
    lines: List[str] = []
    ledger = open_ledger(spec)
    if submit:
        if not lsf.available():
            raise LedgerError("bsub is not on PATH -- run `step --submit` on a WEXAC login node")
        with ledger.locked(break_lock=break_lock):
            open_ledger(spec)  # re-read under the lock
            lines += reconcile_submitting(ledger)
            lines += collect(spec, ledger, tell=True)
            lines += transitions(spec, ledger, apply=True)
            ledger.save()
            state = lsf_snapshot(ledger)
            acts = plan(spec, ledger, state, retry=set(retry))
            lines += execute(spec, ledger, acts, submit=True, only=set(only),
                             coverage_ok=_coverage_gate(spec) if coverage_check else None)
            ledger.save()
    else:
        sup = ledger.data.get("superseded_by")
        if sup:
            lines.append(f"SUPERSEDED by {sup['study']} on {sup['time']}: this study is closed "
                         f"and `--submit` is refused; the lines below are hypothetical")
        lines += [f"(dry run) {n}" for n in collect(spec, ledger, tell=False)]
        lines += [f"(dry run) {n}" for n in transitions(spec, ledger, apply=False)]
        state = lsf_snapshot(ledger)
        acts = plan(spec, ledger, state, retry=set(retry))
        lines += execute(spec, ledger, acts, submit=False, only=set(only),
                         coverage_ok=_coverage_gate(spec) if coverage_check else None)
    return lines


def _coverage_gate(spec: Spec):
    """Per geometry: None if every grid the evaluation opens exists, else why not."""
    cache = {}

    def check(g):
        if g not in cache:
            from .coverage import missing_eval_ids

            miss = missing_eval_ids(spec, g, split="val")
            cache[g] = (None if not miss else
                        f"{len(miss)} stride-4 grid(s) missing for the {g} validation chains "
                        f"(e.g. {miss[:3]}) -- see `sinkholes hpo coverage`")
        return cache[g]

    return check


def approve(spec: Spec, phases: List[str], revoke: bool = False, note: str = "") -> List[str]:
    bad = [p for p in phases if p not in PHASES]
    if bad:
        raise LedgerError(f"unknown phase(s) {bad}; phases are {PHASES}")
    ledger = open_ledger(spec)
    with ledger.locked():
        open_ledger(spec)
        cur = ledger.data.setdefault("approved_phases", [])
        for p in phases:
            if revoke and p in cur:
                cur.remove(p)
            elif not revoke and p not in cur:
                cur.append(p)
        ledger.data.setdefault("approvals", []).append(
            {"time": now(), "phases": phases, "revoke": revoke, "note": note,
             "by": os.environ.get("USER")})
        ledger.save()
        ledger.event("approve", phases=phases, revoke=revoke, note=note)
    return [f"approved phases: {ledger.data['approved_phases']}"]
