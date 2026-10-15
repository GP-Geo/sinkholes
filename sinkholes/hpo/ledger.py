"""The study ledger: one JSON file that is the controller's memory.

Everything a controller call needs to know and nothing it can re-derive goes
here: the trials, their parameters and seeds, the Optuna trial numbers, every
job ever submitted (name, LSF id, argv, script), the evaluation batches and
their scores per epoch and threshold, the objectives per budget, the
decisions (promotions, seed repeats, the locked threshold) with their basis,
and which phases the user has approved for submission.

Writes are atomic (temp file + rename) and the previous version is kept as
``ledger.json.prev``; ``events.jsonl`` is an append-only log of every change.
A lock file stops two controllers acting at once -- the duplicate-submission
guard that matters most, since bsub itself will happily run a job twice.

All paths in the ledger are cluster paths; use :func:`spec.localize` to open
them from the Mac.
"""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .spec import localize

LEDGER = "ledger.json"
EVENTS = "events.jsonl"
LOCK = ".controller.lock"
#: The marker that tells any tidy-up tool this tree is live study state.
PROTECT = "PROTECTED_STUDY_STATE"
STALE_LOCK_S = 6 * 3600


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class LedgerError(SystemExit):
    pass


class Ledger:
    def __init__(self, root: str):
        self.root = Path(localize(root))
        self.path = self.root / LEDGER
        self.data: Dict[str, Any] = {}

    # -- persistence ---------------------------------------------------------------------------
    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> "Ledger":
        prev = self.path.with_name(LEDGER + ".prev")
        if not self.path.exists() and prev.exists():
            # killed between the two renames of save(): the previous version is whole
            self.data = json.loads(prev.read_text())
            return self
        if not self.path.exists():
            raise LedgerError(f"no study ledger at {self.path} -- run `sinkholes hpo init` first")
        self.data = json.loads(self.path.read_text())
        return self

    def save(self) -> None:
        """New version written in full first; then the old one becomes .prev."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{LEDGER}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True, default=_json_default))
        if self.path.exists():
            os.replace(self.path, self.path.with_name(LEDGER + ".prev"))
        os.replace(tmp, self.path)

    def event(self, kind: str, **fields) -> None:
        rec = {"time": now(), "kind": kind, **fields}
        with open(self.root / EVENTS, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=_json_default) + "\n")

    # -- accessors -----------------------------------------------------------------------------
    @property
    def trials(self) -> Dict[str, Dict[str, Any]]:
        return self.data.setdefault("trials", {})

    def trial(self, key: str) -> Dict[str, Any]:
        if key not in self.trials:
            raise LedgerError(f"no trial {key!r} in the ledger")
        return self.trials[key]

    def approved(self, phase: str) -> bool:
        return phase in self.data.get("approved_phases", [])

    def jobs_named(self, name: str):
        for t in self.trials.values():
            for j in t.get("jobs", []):
                if j["name"] == name:
                    yield j
        for j in self.data.get("utility_jobs", []):
            if j["name"] == name:
                yield j

    # -- the lock ------------------------------------------------------------------------------
    @contextmanager
    def locked(self, break_lock: bool = False) -> Iterator["Ledger"]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / LOCK
        if break_lock and lock.exists():
            lock.unlink()
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                info = json.loads(lock.read_text())
            except Exception:
                info = {}
            age = time.time() - lock.stat().st_mtime
            stale = " (older than 6 h -- probably stale)" if age > STALE_LOCK_S else ""
            raise LedgerError(
                f"another controller holds {lock}: {info}{stale}. If no controller is "
                f"running, rerun with --break_lock.")
        with os.fdopen(fd, "w") as fh:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "since": now()}, fh)
        try:
            if self.path.exists():
                self.load()
            yield self
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_protect_marker(root: Path, study: str) -> None:
    """A README at the study root naming it as live state, for humans and tools."""
    (Path(root) / PROTECT).write_text(
        f"Live state of HPO study {study!r}: ledger, Optuna journal, runs (with the\n"
        f"resume checkpoints promotions continue from), evaluations and caches.\n\n"
        f"Do NOT delete, move or prune anything under this directory by hand or with\n"
        f"an outputs/ clean-up script. Retention is\n"
        f"applied explicitly by `sinkholes hpo retain --apply`, which knows which\n"
        f"checkpoints are still needed.\n")
