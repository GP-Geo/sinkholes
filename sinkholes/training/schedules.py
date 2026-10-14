"""Learning-rate schedules whose shape is fixed when a run starts.

``--lr_schedule cosine`` without ``--lr_horizon`` is the historical
``CosineAnnealingLR(T_max=--epochs)``: its horizon is whatever ``--epochs`` was
on the first execution, and PyTorch's recursive form turns back UP past
``T_max`` (it is periodic, a warm restart). Neither is acceptable for a long run
trained in several wall-limited chunks:

* the horizon has to be set once, at epoch 1, and survive every resume -- a
  chunk that passes ``--epochs 300`` must not be able to re-shape a schedule
  started with a different number;
* the decay must end at the floor and stay there, never restart.

:class:`CosineToFloor` is the closed form ``floor + (base - floor) *
(1 + cos(pi * min(t, H) / H)) / 2``. The horizon, base LR and floor live in its
own ``state_dict`` and :meth:`load_state_dict` REFUSES a different horizon, so
the schedule cannot drift by a resume that changed a flag. Closed form also
means no accumulated recursion error over hundreds of steps.

:class:`ConstantSchedule` holds the LR fixed -- the diagnostic control for
telling an LR-driven recovery from one that happens at a fixed LR.

Epoch convention, matching every scheduler in train.py: ``step()`` runs after
an epoch's validation, so epoch ``e`` (1-based) trains at ``t = e - 1``; the
floor is reached at ``t = H``, i.e. it is what epoch ``H + 1`` would use.
"""

from __future__ import annotations

import math
from typing import Iterable, List

__all__ = ["CosineToFloor", "ConstantSchedule", "parse_epoch_spec", "cosine_lr"]


def cosine_lr(t: int, base: float, floor: float, horizon: int) -> float:
    """LR after ``t`` completed epochs of a ``horizon``-epoch cosine decay."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    t = min(max(int(t), 0), int(horizon))
    return floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * t / horizon))


class _FixedShapeSchedule:
    """Shared plumbing: owns the optimizer's LR, keeps its state picklable."""

    kind = "base"

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0
        self._apply()

    def lr_at(self, t: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def _apply(self) -> None:
        lr = self.lr_at(self.last_epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self, *_ignored) -> None:
        """Advance one epoch. Ignores a metric so it is a drop-in for plateau's call."""
        self.last_epoch += 1
        self._apply()

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class CosineToFloor(_FixedShapeSchedule):
    """Closed-form cosine decay from ``base_lr`` to ``floor`` over ``horizon`` epochs."""

    kind = "cosine_floor"

    def __init__(self, optimizer, *, base_lr: float, floor: float, horizon: int):
        if horizon <= 0:
            raise ValueError(f"--lr_horizon must be positive, got {horizon}")
        if not 0.0 <= floor <= base_lr:
            raise ValueError(f"the LR floor {floor:g} must lie in [0, base LR {base_lr:g}]")
        self.base_lr = float(base_lr)
        self.floor = float(floor)
        self.horizon = int(horizon)
        super().__init__(optimizer)

    def lr_at(self, t: int) -> float:
        return cosine_lr(t, self.base_lr, self.floor, self.horizon)

    def state_dict(self) -> dict:
        return {"kind": self.kind, "base_lr": self.base_lr, "floor": self.floor,
                "horizon": self.horizon, "last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != self.kind:
            raise ValueError(f"checkpoint scheduler is {state.get('kind')!r}, not {self.kind!r}: "
                             f"the schedule cannot change across a resume")
        mismatched = [(k, state.get(k), getattr(self, k)) for k in ("base_lr", "floor", "horizon")
                      if state.get(k) != getattr(self, k)]
        if mismatched:
            detail = ", ".join(f"{k} {was!r} -> {now!r}" for k, was, now in mismatched)
            raise ValueError(f"the cosine schedule was fixed at the start of the run and a "
                             f"resume may not change it ({detail})")
        self.last_epoch = int(state["last_epoch"])
        self._apply()


class ConstantSchedule(_FixedShapeSchedule):
    """The LR never moves: the control arm for an LR-driven recovery."""

    kind = "constant"

    def __init__(self, optimizer, *, base_lr: float):
        self.base_lr = float(base_lr)
        super().__init__(optimizer)

    def lr_at(self, t: int) -> float:
        return self.base_lr

    def state_dict(self) -> dict:
        return {"kind": self.kind, "base_lr": self.base_lr, "last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != self.kind or state.get("base_lr") != self.base_lr:
            raise ValueError(f"checkpoint scheduler {state!r} is not a constant LR of "
                             f"{self.base_lr:g}: the schedule cannot change across a resume")
        self.last_epoch = int(state["last_epoch"])
        self._apply()


def parse_epoch_spec(spec) -> List[int]:
    """``"1,10,20:60:10,80:300:20"`` -> sorted unique positive epochs.

    Items are comma separated; ``a:b:s`` is the inclusive range a, a+s, ... <= b
    (``a:b`` steps by 1). An iterable of ints passes through, deduplicated.
    Empty or None gives [].
    """
    if spec is None:
        return []
    if not isinstance(spec, str):
        values: Iterable = spec
    else:
        values = []
        for item in (s.strip() for s in spec.split(",")):
            if not item:
                continue
            if ":" in item:
                parts = [int(p) for p in item.split(":")]
                if len(parts) == 2:
                    parts.append(1)
                if len(parts) != 3 or parts[2] <= 0 or parts[0] > parts[1]:
                    raise ValueError(f"bad epoch range {item!r}: use start:stop[:step]")
                values.extend(range(parts[0], parts[1] + 1, parts[2]))
            else:
                values.append(int(item))
    out = sorted({int(v) for v in values})
    if any(v <= 0 for v in out):
        raise ValueError(f"epochs are 1-based and positive, got {out}")
    return out
