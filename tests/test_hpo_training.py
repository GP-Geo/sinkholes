"""Training-side guarantees the HPO and long-run protocol rests on.

* a run trained in wall-limited chunks is the SAME run as an uninterrupted one
  (weights bit-identical, same results rows), whatever the chunking;
* the cosine horizon is fixed at epoch 1 and survives every resume;
* evaluation (validation, the training-fit probe) cannot move the training RNG;
* snapshots land at exactly the scheduled epochs and load like best.pt;
* the recovery checkpoint rotates, and a resume falls back to the previous copy;
* --strict_resume locks the optimisation settings for the life of the run.

Everything trains a tiny U-Net on synthetic patches on the CPU for seconds.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from sinkholes.models.unet import UNet
from sinkholes.training import resume as R
from sinkholes.training import train as T
from sinkholes.training.schedules import (
    ConstantSchedule,
    CosineToFloor,
    cosine_lr,
    parse_epoch_spec,
)

PATCH = 32


def make_args(**overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    T.add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TinySet(list):
    """{image, mask} samples; every third one is an all-negative patch."""

    mask_values = [0, 1]

    @classmethod
    def make(cls, n=6, seed=0):
        gen = torch.Generator().manual_seed(seed)
        items = []
        for k in range(n):
            image = torch.rand(1, PATCH, PATCH, generator=gen)
            mask = (torch.rand(PATCH, PATCH, generator=gen) > 0.7).long()
            if k % 3 == 2:
                mask = torch.zeros_like(mask)
            items.append({"image": image, "mask": mask})
        return cls(items)


def tiny_model():
    torch.manual_seed(0)
    return UNet(n_channels=1, n_classes=1, bilinear=False)


BASE = dict(batch_size=2, lr=1e-3, patience=0, sample_every=0, job_name="tiny",
            save_best_only=True, optimizer="adamw", weight_decay=1e-2, seed=3)


def train(outpath, epochs, checkpoint=None, model=None, **overrides):
    args = make_args(epochs=epochs, **{**BASE, **overrides})
    model = model or tiny_model()
    torch.manual_seed(args.seed)  # what main() does before building anything
    T.train_model(args, model, torch.device("cpu"), TinySet.make(), TinySet.make(seed=1),
                  None, str(outpath), checkpoint=checkpoint)
    return model, args


def resume_from(run_dir):
    state, _ = R.load_resume_with_fallback(Path(run_dir) / "checkpoints" / R.RESUME_NAME)
    return state


def csv_rows(run_dir):
    with open(Path(run_dir) / "results.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same_weights(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


DETERMINISTIC_COLS = ("train/loss", "val/dice", "val/loss", "lr", "lr_used", "grad/norm",
                      "fit/dice", "fit/loss", "opt/steps")


# -- schedules ------------------------------------------------------------------------------

def test_cosine_to_floor_hits_base_and_floor_and_never_restarts():
    assert cosine_lr(0, 3e-4, 3e-6, 300) == pytest.approx(3e-4)
    assert cosine_lr(150, 3e-4, 3e-6, 300) == pytest.approx((3e-4 + 3e-6) / 2)
    assert cosine_lr(300, 3e-4, 3e-6, 300) == pytest.approx(3e-6)
    # past the horizon it stays at the floor -- CosineAnnealingLR would turn back up
    assert cosine_lr(450, 3e-4, 3e-6, 300) == pytest.approx(3e-6)
    lrs = [cosine_lr(t, 1.0, 0.01, 50) for t in range(51)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_cosine_schedule_state_refuses_a_changed_horizon():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = CosineToFloor(opt, base_lr=1.0, floor=0.01, horizon=300)
    for _ in range(7):
        sched.step()
    state = sched.state_dict()
    assert state["horizon"] == 300 and state["last_epoch"] == 7

    opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    again = CosineToFloor(opt2, base_lr=1.0, floor=0.01, horizon=300)
    again.load_state_dict(state)
    assert opt2.param_groups[0]["lr"] == pytest.approx(cosine_lr(7, 1.0, 0.01, 300))

    other = CosineToFloor(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0),
                          base_lr=1.0, floor=0.01, horizon=200)
    with pytest.raises(ValueError, match="horizon"):
        other.load_state_dict(state)


def test_constant_schedule_never_moves():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.5)
    sched = ConstantSchedule(opt, base_lr=0.5)
    for _ in range(5):
        sched.step(0.123)  # a metric is accepted and ignored, like plateau's call
    assert opt.param_groups[0]["lr"] == 0.5


@pytest.mark.parametrize("spec,expected", [
    ("10,20,30", [10, 20, 30]),
    ("1,10:60:10,80:300:20", [1, 10, 20, 30, 40, 50, 60] + list(range(80, 301, 20))),
    ("5:7", [5, 6, 7]),
    ("", []),
    (None, []),
    ([30, 10, 10], [10, 30]),
])
def test_epoch_specs(spec, expected):
    assert parse_epoch_spec(spec) == expected


@pytest.mark.parametrize("bad", ["0,10", "10:5", "1:10:0", "a"])
def test_bad_epoch_specs_are_refused(bad):
    with pytest.raises(ValueError):
        parse_epoch_spec(bad)


@pytest.mark.parametrize("overrides,match", [
    (dict(lr_schedule="plateau", lr_horizon=10), "cosine only"),
    (dict(lr_schedule="cosine", lr_horizon=5, epochs=6), "exceeds"),
    (dict(lr_schedule="cosine", lr_horizon=5, epochs=5, min_lr=1.0, lr=1e-3), "must lie"),
    (dict(snapshot_epochs="0,5"), None),
])
def test_schedule_flags_that_would_not_mean_what_they_say_are_refused(overrides, match):
    args = make_args(**{**BASE, "epochs": 5, **overrides})
    with pytest.raises((SystemExit, ValueError), match=match):
        T.validate_schedule_args(args)


# -- chunked training is the same run ------------------------------------------------------

LONG = dict(lr_schedule="cosine", lr_horizon=6, min_lr=1e-5, eval_rng_isolation=True,
            fit_eval_samples=3, keep_prev_resume=True, snapshot_epochs="1,3,5",
            strict_resume=True)


def test_a_run_trained_in_chunks_is_bit_identical_to_an_uninterrupted_one(tmp_path):
    straight, _ = train(tmp_path / "straight", 5, **LONG)

    chunked_dir = tmp_path / "chunked"
    train(chunked_dir, 2, **LONG)                      # chunk 1: epochs 1-2
    train(chunked_dir, 4, checkpoint=resume_from(chunked_dir), **LONG)   # chunk 2: 3-4
    chunked, _ = train(chunked_dir, 5, checkpoint=resume_from(chunked_dir), **LONG)

    # train_model leaves the best weights loaded; compare the final epoch instead.
    a = torch.load(tmp_path / "straight" / "checkpoints" / "last.pt")
    b = torch.load(chunked_dir / "checkpoints" / "last.pt")
    assert all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))

    ra, rb = csv_rows(tmp_path / "straight"), csv_rows(chunked_dir)
    assert [r["epoch"] for r in rb] == ["1/2", "2/2", "3/4", "4/4", "5/5"]
    for x, y in zip(ra, rb):
        for col in DETERMINISTIC_COLS:
            assert x[col] == y[col], (col, x["epoch"], x[col], y[col])


def test_the_cosine_horizon_survives_resumes_and_lr_used_is_the_epochs_own(tmp_path):
    run = tmp_path / "r"
    train(run, 2, **LONG)
    train(run, 5, checkpoint=resume_from(run), **LONG)
    rows = csv_rows(run)
    for e, row in enumerate(rows, start=1):
        assert float(row["lr_used"]) == pytest.approx(cosine_lr(e - 1, 1e-3, 1e-5, 6))
        # the historical column is read after scheduler.step(): the NEXT epoch's
        assert float(row["lr"]) == pytest.approx(cosine_lr(e, 1e-3, 1e-5, 6))


def test_a_resume_cannot_change_the_horizon(tmp_path):
    run = tmp_path / "r"
    train(run, 2, **LONG)
    with pytest.raises(SystemExit, match="lr_horizon"):
        train(run, 4, checkpoint=resume_from(run), **{**LONG, "lr_horizon": 8})


# -- evaluation does not move the training RNG ---------------------------------------------

def test_with_isolation_the_fit_probe_cannot_change_training(tmp_path):
    base = dict(LONG, snapshot_epochs="", keep_prev_resume=False)
    train(tmp_path / "a", 3, **{**base, "fit_eval_samples": 0})
    train(tmp_path / "b", 3, **{**base, "fit_eval_samples": 3})
    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt")
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt")
    assert all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))


def test_isolated_rng_restores_every_stream():
    import random

    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    expected = (torch.rand(3), np.random.rand(3), random.random())
    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    with R.isolated_rng():
        torch.rand(100); np.random.rand(100); random.random()
        torch.utils.data.DataLoader(list(range(4)), shuffle=False).__iter__()
    got = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(expected[0], got[0])
    assert np.array_equal(expected[1], got[1]) and expected[2] == got[2]


def test_fit_subset_is_positive_fixed_and_unaugmented():
    class Flippable(TinySet):
        augment_flips = "hv"

    data = Flippable(TinySet.make(n=9))
    sub1, idx1 = T.fit_eval_subset(data, 4, seed=3)
    sub2, idx2 = T.fit_eval_subset(data, 4, seed=3)
    assert idx1 == idx2 and len(idx1) == 4
    assert all(k % 3 != 2 for k in idx1)            # the negatives are never drawn
    assert data.augment_flips == "hv"               # the training set is untouched
    assert sub1.dataset.augment_flips == "none"


# -- snapshots, rotation, fallback ---------------------------------------------------------

def test_snapshots_are_written_at_exactly_the_scheduled_epochs_and_load_like_best(tmp_path):
    from sinkholes.models.factory import build_from_checkpoint

    train(tmp_path, 5, **LONG)
    snaps = sorted(p.name for p in (tmp_path / "checkpoints" / "snapshots").iterdir())
    assert snaps == ["epoch_001.pt", "epoch_003.pt", "epoch_005.pt"]
    state = torch.load(tmp_path / "checkpoints" / "snapshots" / "epoch_005.pt")
    # exactly what eval-scenes does: the factory strips the extras in place
    loaded = build_from_checkpoint(state, n_classes=1, bilinear=False)
    loaded.model.load_state_dict(state)
    # the snapshot of the final epoch IS last.pt
    last = torch.load(tmp_path / "checkpoints" / "last.pt")
    assert all(torch.equal(state[k], last[k]) for k in state if torch.is_tensor(state[k]))
    # and no per-epoch .pth is written under --save_best_only
    assert not list((tmp_path / "checkpoints").glob("*.pth"))


def test_the_recovery_checkpoint_rotates_and_resume_falls_back_to_the_previous(tmp_path):
    train(tmp_path, 3, **LONG)
    ck = tmp_path / "checkpoints"
    assert R.load_resume_checkpoint(ck / "resume.pt")["epoch"] == 3
    assert R.load_resume_checkpoint(ck / "resume_prev.pt")["epoch"] == 2

    # a corrupt latest file: the previous epoch is loaded instead
    (ck / "resume.pt").write_bytes(b"truncated")
    state, used = R.load_resume_with_fallback(ck / "resume.pt")
    assert state["epoch"] == 2 and used.name == "resume_prev.pt"

    # killed between the two renames: only the previous copy exists
    (ck / "resume.pt").unlink()
    location = R.resolve_run_location(make_args(resume=str(tmp_path), job_name="tiny"))
    assert location.resume_path == ck / "resume_prev.pt"

    # resuming from it repeats epoch 3 and finishes the run
    train(tmp_path, 4, checkpoint=resume_from(tmp_path), **LONG)
    assert [r["epoch"] for r in csv_rows(tmp_path)][-2:] == ["3/4", "4/4"]


# -- time budget -----------------------------------------------------------------------------

def test_a_time_budget_pauses_cleanly_and_the_continuation_finishes(tmp_path):
    train(tmp_path, 4, time_budget_hours=1e-6, **LONG)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["state"] == "paused" and status["completed_epoch"] == 1
    assert status["target_epochs"] == 4

    train(tmp_path, 4, checkpoint=resume_from(tmp_path), **LONG)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["state"] == "complete" and status["completed_epoch"] == 4
    assert status["snapshots"] == ["epoch_001.pt", "epoch_003.pt"]


# -- strict resume ---------------------------------------------------------------------------

@pytest.mark.parametrize("change", [dict(lr=2e-3), dict(pos_w=4.0), dict(min_lr=1e-6),
                                    dict(beta1=0.8), dict(eval_rng_isolation=False)])
def test_strict_resume_refuses_a_changed_optimisation_setting(tmp_path, change):
    train(tmp_path, 1, **LONG)
    # strict_resume is recorded in the checkpoint: even a continuation that
    # forgets the flag stays locked.
    with pytest.raises(SystemExit):
        train(tmp_path, 2, checkpoint=resume_from(tmp_path),
              **{**LONG, "strict_resume": False, **change})


def test_without_strict_resume_a_changed_lr_is_only_reported(tmp_path):
    loose = dict(LONG, strict_resume=False, lr_schedule="plateau", lr_horizon=0)
    train(tmp_path, 1, **loose)
    train(tmp_path, 2, checkpoint=resume_from(tmp_path), **{**loose, "pos_w": 4.0})


def test_snapshot_schedule_and_budget_may_change_between_chunks(tmp_path):
    """A promotion adds evaluation epochs; that must not count as a new experiment."""
    train(tmp_path, 2, **LONG)
    train(tmp_path, 4, checkpoint=resume_from(tmp_path),
          **{**LONG, "snapshot_epochs": "4", "time_budget_hours": 5.0})


# -- records -------------------------------------------------------------------------------

def test_every_execution_appends_a_provenance_record(tmp_path):
    train(tmp_path, 1, **LONG)
    train(tmp_path, 2, checkpoint=resume_from(tmp_path), **LONG)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    first, second = manifest["executions"]
    assert first["resumed_from_epoch"] is None and second["resumed_from_epoch"] == 1
    assert len(first["source"]["package_sha256"]) == 64
    assert first["optimizer"]["betas"] == [0.9, 0.999]
    assert first["schedule"]["horizon"] == 6
    assert first["numerics"]["eval_rng"] == R.EVAL_RNG_ISOLATED
    assert first["config"]["lr_horizon"] == 6 and first["config"]["strict_resume"] is True


def test_new_columns_are_recorded(tmp_path):
    train(tmp_path, 1, **LONG)
    row = csv_rows(tmp_path)[0]
    for col in T.EXTRA_COLUMNS + T.FIT_COLUMNS:
        assert col in row
    assert row["opt/skipped"] == "0" and int(row["opt/steps"]) == 3
    assert 0.0 <= float(row["fit/dice"]) <= 1.0


def test_skipped_amp_updates_are_counted(tmp_path, monkeypatch):
    """A scale that backs off across step+update is exactly a skipped step."""

    class FakeScaler:
        def __init__(self, enabled=True, **_):
            self.scale_value, self.n = 1024.0, 0

        def scale(self, loss):
            return loss

        def unscale_(self, opt):
            pass

        def step(self, opt):
            self.n += 1
            if self.n % 2:            # every odd step "found an inf" and is skipped
                self.skip = True
            else:
                self.skip = False
                opt.step()

        def update(self):
            self.scale_value *= 0.5 if self.skip else 1.0

        def get_scale(self):
            return self.scale_value

        def state_dict(self):
            return {"scale": self.scale_value}

        def load_state_dict(self, s):
            self.scale_value = s["scale"]

    monkeypatch.setattr(torch.amp, "GradScaler", FakeScaler)
    train(tmp_path, 1, **{**LONG, "amp": True})
    row = csv_rows(tmp_path)[0]
    assert row["opt/steps"] == "3" and row["opt/skipped"] == "2"
    assert float(row["amp/scale_min"]) == 256.0
