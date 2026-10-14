"""Optuna, used as a persistent record and as the TPE proposer -- nothing more.

Optuna never runs anything. The controller asks it for parameters, submits the
jobs itself, and tells it the objective once the scene evaluations are in.
Storage is a journal file in the study root (append-only, the NFS-safe Optuna
backend), so the studies survive every controller process.

Two studies per geometry keep budgets apart:

* ``<study>_<geometry>_e30`` -- every initial trial (6 shared + 2 TPE), valued at
  the epoch-20/30 objective. The TPE sampler reads only this study.
* ``<study>_<geometry>_e60`` -- promoted continuations and seed repeats, valued
  at the epoch-50/60 objective, added as finished trials with their origin and
  seed as attributes. A 60-epoch number can never leak into the 30-epoch study.

Optuna is imported lazily: GPU jobs never need it, only the controller.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .spec import Spec, localize


def _optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit("the HPO controller needs optuna (pip install optuna==5.0.0 in the "
                         "sinkholes env); GPU jobs do not") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import warnings

    # multivariate TPE is flagged experimental; the choice is deliberate and recorded
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna


def storage_path(spec: Spec) -> str:
    return f"{spec.root}/optuna.journal"


def storage(spec: Spec):
    optuna = _optuna()
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(localize(storage_path(spec))))


def study_name(spec: Spec, geometry: str, budget: str) -> str:
    return f"{spec.name}_{geometry}_{budget}"


def distributions(spec: Spec) -> Dict[str, Any]:
    optuna = _optuna()
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    s = spec.space
    return {
        "lr": FloatDistribution(s["lr"]["low"], s["lr"]["high"], log=bool(s["lr"]["log"])),
        "weight_decay": FloatDistribution(s["weight_decay"]["low"], s["weight_decay"]["high"],
                                          log=bool(s["weight_decay"]["log"])),
        "pos_w": CategoricalDistribution(list(s["pos_w"]["choices"])),
    }


def sampler(spec: Spec):
    optuna = _optuna()
    t = spec.tpe
    return optuna.samplers.TPESampler(
        n_startup_trials=int(t["n_startup_trials"]), seed=int(t["seed"]),
        multivariate=bool(t["multivariate"]), constant_liar=bool(t["constant_liar"]))


def open_study(spec: Spec, geometry: str, budget: str, create: bool = False):
    optuna = _optuna()
    kw = dict(study_name=study_name(spec, geometry, budget), storage=storage(spec))
    if create:
        return optuna.create_study(direction="maximize", sampler=sampler(spec),
                                   load_if_exists=True, **kw)
    return optuna.load_study(sampler=sampler(spec), **kw)


def ask_enqueued(study, spec: Spec, params_list: List[dict]) -> List[int]:
    """Enqueue fixed configurations and ask them, in order; return trial numbers."""
    numbers = []
    dist = distributions(spec)
    for params in params_list:
        study.enqueue_trial(dict(params))
        trial = study.ask(dist)
        if {k: trial.params[k] for k in params} != dict(params):
            raise SystemExit(f"Optuna returned {trial.params} for the enqueued {params}")
        numbers.append(trial.number)
    return numbers


def ask_tpe(study, spec: Spec, n: int) -> List[dict]:
    """``n`` TPE proposals: [{"number": .., "params": {..}}]."""
    dist = distributions(spec)
    out = []
    for _ in range(n):
        trial = study.ask(dist)
        out.append({"number": trial.number, "params": dict(trial.params)})
    return out


def tell(study, number: int, value: float, attrs: Dict[str, Any]) -> None:
    """Record a finished trial's objective (and its details as attributes)."""
    optuna = _optuna()
    frozen = study.trials[number]
    if frozen.state.is_finished():
        if frozen.value != value:
            raise SystemExit(f"Optuna trial {number} already holds {frozen.value}, not {value}")
        return
    try:  # attributes are a convenience; the ledger holds the full record
        trial_id = study._storage.get_trial_id_from_study_id_trial_number(study._study_id, number)
        handle = optuna.trial.Trial(study, trial_id)
        for k, v in attrs.items():
            handle.set_user_attr(k, v)
    except Exception:
        pass
    study.tell(number, value)


def add_finished(study, spec: Spec, params: dict, value: float, attrs: Dict[str, Any]) -> int:
    optuna = _optuna()
    trial = optuna.trial.create_trial(params=dict(params), distributions=distributions(spec),
                                      value=value, user_attrs=dict(attrs))
    study.add_trial(trial)
    return len(study.trials) - 1
