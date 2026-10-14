"""The study specification: everything fixed before the first job runs.

A study is one JSON file (``assets/hpo/*.json``) naming the data, the two
geometries, the fixed training configuration, the search space, the shared
designs, the budget, the evaluation protocol and the resources. It is hashed
into the ledger at ``init`` and every later controller call checks the hash,
so a study cannot be quietly re-specified halfway through.

Paths: the spec holds cluster paths (``/home/labs/rudich/...``). On the Mac the
same files are under ``/Volumes/rudich/...``; :func:`localize` maps between the
two so ``status``/``report``/``coverage`` run from either side.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..training.schedules import parse_epoch_spec

#: (cluster prefix, Mac mount prefix) -- the same storage seen from both sides.
PATH_ALIASES = (("/home/labs/rudich/", "/Volumes/rudich/"),)

GEOMETRIES = ("plain", "ctx50")
SEARCH_PARAMS = ("lr", "weight_decay", "pos_w")


def localize(path: str) -> str:
    """The same file as seen from this machine (cluster <-> Mac mount)."""
    p = str(path)
    if os.path.exists(p):
        return p
    for cluster, mac in PATH_ALIASES:
        for a, b in ((cluster, mac), (mac, cluster)):
            if p.startswith(a) and os.path.exists(b + p[len(a):]):
                return b + p[len(a):]
    return p


def to_cluster(path: str) -> str:
    """The cluster spelling of a path (what job scripts must contain)."""
    p = str(path)
    for cluster, mac in PATH_ALIASES:
        if p.startswith(mac):
            return cluster + p[len(mac):]
    return p


@dataclass
class Spec:
    raw: Dict[str, Any]
    path: str
    #: The code tree a study's jobs run from, pinned in the ledger at init. Kept
    #: out of ``raw`` so the spec's hash does not depend on where it was opened.
    repo_override: Optional[str] = None

    # -- identity
    @property
    def name(self) -> str:
        return self.raw["study"]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()

    # -- locations (cluster spelling)
    @property
    def root(self) -> str:
        return self.raw["root"]

    @property
    def repo(self) -> str:
        """Ledger-pinned repo, else the spec's, else this checkout (cluster spelling)."""
        if self.repo_override:
            return self.repo_override
        if self.raw.get("repo"):
            return self.raw["repo"]
        from ..training.provenance import REPO_ROOT

        return to_cluster(str(REPO_ROOT))

    def repo_path(self, rel: str) -> str:
        return rel if rel.startswith("/") else f"{self.repo}/{rel}"

    @property
    def partition(self) -> str:
        return self.repo_path(self.raw["partition"])

    @property
    def test_partition(self) -> str:
        return self.repo_path(self.raw["test_partition"])

    @property
    def k_prevs(self) -> int:
        return int(self.raw["k_prevs"])

    # -- sections
    def geometry(self, name: str) -> Dict[str, Any]:
        return self.raw["geometries"][name]

    @property
    def geometries(self) -> List[str]:
        return list(self.raw["geometries"])

    @property
    def fixed(self) -> Dict[str, Any]:
        return self.raw["fixed"]

    @property
    def space(self) -> Dict[str, Any]:
        return self.raw["search_space"]

    @property
    def shared(self) -> List[Dict[str, Any]]:
        return self.raw["shared_configs"]

    @property
    def budget(self) -> Dict[str, Any]:
        return self.raw["budget"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def tpe(self) -> Dict[str, Any]:
        return self.raw["tpe"]

    @property
    def long(self) -> Dict[str, Any]:
        return self.raw["long"]

    def resources(self, kind: str, geometry: str = None) -> Dict[str, Any]:
        r = self.raw["resources"][kind]
        return r[geometry] if geometry and geometry in r else r


def load_spec(path) -> Spec:
    raw = json.loads(Path(localize(str(path))).read_text())
    spec = Spec(raw=raw, path=str(path))
    validate(spec)
    return spec


class SpecError(SystemExit):
    pass


def validate(spec: Spec) -> None:
    """Refuse a spec that would not run the protocol it claims to."""
    r = spec.raw
    need = ("study", "root", "repo", "partition", "test_partition", "k_prevs", "geometries",
            "fixed", "search_space", "shared_configs", "budget", "evaluation", "tpe",
            "resources", "long")
    missing = [k for k in need if k not in r]
    if missing:
        raise SpecError(f"study spec {spec.path} lacks {missing}")
    if set(spec.geometries) - set(GEOMETRIES):
        raise SpecError(f"unknown geometries {sorted(set(spec.geometries) - set(GEOMETRIES))}")
    space = spec.space
    if set(space) != set(SEARCH_PARAMS):
        raise SpecError(f"search space must be exactly {SEARCH_PARAMS}, got {sorted(space)}")
    for cfg in spec.shared:
        if set(cfg) != set(SEARCH_PARAMS):
            raise SpecError(f"shared config {cfg} must set exactly {SEARCH_PARAMS}")
        for k in ("lr", "weight_decay"):
            lo, hi = space[k]["low"], space[k]["high"]
            if not lo <= cfg[k] <= hi:
                raise SpecError(f"shared config {cfg}: {k} outside [{lo}, {hi}]")
        if cfg["pos_w"] not in space["pos_w"]["choices"]:
            raise SpecError(f"shared config {cfg}: pos_w not in {space['pos_w']['choices']}")
    base = spec.raw.get("baseline_index", 0)
    if not 0 <= base < len(spec.shared):
        raise SpecError("baseline_index must point into shared_configs")
    ev = spec.evaluation
    for key in ("initial_epochs", "promoted_epochs"):
        parse_epoch_spec(ev[key])
    b = spec.budget
    if max(parse_epoch_spec(ev["initial_epochs"])) > b["initial_epochs"]:
        raise SpecError("an initial evaluation epoch exceeds the initial budget")
    if not set(ev["objective_epochs"]) <= set(parse_epoch_spec(ev["initial_epochs"])):
        raise SpecError("objective_epochs must be evaluated epochs")
    if not set(ev["promoted_objective_epochs"]) <= set(parse_epoch_spec(ev["promoted_epochs"])):
        raise SpecError("promoted_objective_epochs must be evaluated epochs")
    forbidden = {"lr", "weight_decay", "pos_w", "seed", "epochs", "context_margin"} & set(spec.fixed)
    if forbidden:
        raise SpecError(f"'fixed' may not set per-trial values {sorted(forbidden)}")
    for g in spec.geometries:
        if spec.geometry(g).get("batch_size") != spec.geometry(spec.geometries[0]).get("batch_size") \
                or spec.geometry(g).get("accum_steps") != spec.geometry(spec.geometries[0]).get("accum_steps"):
            raise SpecError("micro-batch and accumulation must match across geometries")
