"""Preemption-safe resumption: run directories, full-state checkpoints, restart.

WEXAC preempts jobs and reruns the same LSF job id. Everything here protects
one property: a requeued execution must land in the same directory, load the
state of the last fully completed epoch, and continue — never silently restart
from epoch 1 in a fresh timestamped directory, and never mix two experiments.

The end-to-end tests train a tiny U-Net on synthetic patches for a couple of
seconds; that is the only way to exercise the optimizer/scheduler/scaler/CSV
interaction the way training does.
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from sinkholes.models.unet import UNet
from sinkholes.training import resume as R
from sinkholes.training.reporter import BestTracker, ResultsCSV
from sinkholes.training.train import add_arguments, train_model

PATCH_H, PATCH_W = 32, 32


def make_args(**overrides) -> argparse.Namespace:
    """Real CLI defaults, so these tests move with the parser rather than a copy."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TinySet(list):
    """A handful of {image, mask} samples — enough for a real epoch loop."""

    mask_values = [0, 1]

    @classmethod
    def make(cls, n=4, seed=0):
        gen = torch.Generator().manual_seed(seed)
        items = []
        for _ in range(n):
            image = torch.rand(1, PATCH_H, PATCH_W, generator=gen)
            mask = (torch.rand(PATCH_H, PATCH_W, generator=gen) > 0.7).long()
            items.append({"image": image, "mask": mask})
        return cls(items)


def tiny_model():
    torch.manual_seed(0)
    return UNet(n_channels=1, n_classes=1, bilinear=False)


def run_training(outpath, epochs, checkpoint=None, **overrides):
    """One `train_model` execution against the tiny dataset."""
    args = make_args(epochs=epochs, batch_size=2, lr=1e-3, patience=0, sample_every=0,
                     job_name="tiny", save_best_only=True, **overrides)
    train_model(args, tiny_model(), torch.device("cpu"),
                TinySet.make(), TinySet.make(seed=1), None, str(outpath),
                checkpoint=checkpoint)
    return args


def read_csv_epochs(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [row["epoch"] for row in csv.DictReader(f)]


# -- run directory resolution --------------------------------------------------------------

def test_resume_auto_names_the_directory_after_the_job_id_and_the_start_time(tmp_path):
    args = make_args(job_name="convlstm_geo_k10", output_dir=str(tmp_path), resume="auto")
    location = R.resolve_run_location(args, env={"LSB_JOBID": "123456"})
    name = Path(location.outpath).name
    # Timestamp for navigating outputs/, job id for finding it again.
    assert name.startswith("convlstm_geo_k10_20") and name.endswith("_lsf_123456")
    # The job name itself must stay untouched: it is half of the search key.
    assert location.job_name == "convlstm_geo_k10"


def test_repeated_execution_of_the_same_job_picks_the_same_directory(tmp_path):
    """The requeued execution finds the first one's directory, timestamp and all."""
    env = {"LSB_JOBID": "987"}
    args = make_args(job_name="j", output_dir=str(tmp_path), resume="auto")

    first = R.resolve_run_location(args, env=env)
    Path(first.outpath).mkdir(parents=True)  # as the first execution would

    second = R.resolve_run_location(args, env=env)
    assert second.outpath == first.outpath
    assert "_lsf_987" in second.outpath


def test_a_requeue_reuses_the_directory_even_when_the_clock_has_moved_on(tmp_path):
    """A name minted an hour ago must still be found, not replaced by a new one."""
    env = {"LSB_JOBID": "555"}
    stale = tmp_path / "j_2026-01-01_03h07_lsf_555"
    (stale / "checkpoints").mkdir(parents=True)
    (stale / "checkpoints" / R.RESUME_NAME).write_bytes(b"")

    location = R.resolve_run_location(
        make_args(job_name="j", output_dir=str(tmp_path), resume="auto"), env=env)
    assert location.outpath == str(stale)
    assert location.resume_path == stale / "checkpoints" / R.RESUME_NAME


def test_directories_from_the_timestamp_free_naming_are_still_found(tmp_path):
    env = {"LSB_JOBID": "606"}
    old = tmp_path / "j_lsf_606"
    old.mkdir(parents=True)
    location = R.resolve_run_location(
        make_args(job_name="j", output_dir=str(tmp_path), resume="auto"), env=env)
    assert location.outpath == str(old)


def test_another_jobs_directory_is_never_adopted(tmp_path):
    (tmp_path / "j_2026-01-01_03h07_lsf_111").mkdir(parents=True)   # a different job id
    (tmp_path / "other_2026-01-01_03h07_lsf_222").mkdir(parents=True)  # a different job name
    location = R.resolve_run_location(
        make_args(job_name="j", output_dir=str(tmp_path), resume="auto"),
        env={"LSB_JOBID": "222"})
    assert Path(location.outpath).name.endswith("_lsf_222")
    assert not Path(location.outpath).exists()  # a new directory, not either of those


def test_array_jobs_do_not_share_one_directory(tmp_path):
    base = {"LSB_JOBID": "42"}
    args = make_args(job_name="j", output_dir=str(tmp_path), resume="auto")
    one = R.resolve_run_location(args, env={**base, "LSB_JOBINDEX": "1"})
    Path(one.outpath).mkdir(parents=True)
    two = R.resolve_run_location(args, env={**base, "LSB_JOBINDEX": "2"})
    assert one.outpath != two.outpath


def test_missing_job_id_fails_loudly_rather_than_risking_a_collision():
    with pytest.raises(SystemExit) as excinfo:
        R.resolve_run_location(make_args(job_name="j", resume="auto"), env={})
    assert "LSB_JOBID" in str(excinfo.value)


def test_no_resume_keeps_the_timestamped_run_directory():
    location = R.resolve_run_location(make_args(job_name="j", output_dir="outputs"))
    assert location.resume_path is None
    assert location.mode == "fresh"
    assert location.job_name.startswith("j_20")  # j_<YYYY-MM-DD_HHhMM>
    assert location.outpath == os.path.join("outputs", location.job_name)


def test_auto_finds_an_existing_checkpoint_and_starts_fresh_without_one(tmp_path):
    env = {"LSB_JOBID": "5"}
    args = make_args(job_name="j", output_dir=str(tmp_path), resume="auto")

    fresh = R.resolve_run_location(args, env=env)
    assert fresh.resume_path is None
    assert "first execution" in " ".join(fresh.notes)

    ckpt = Path(fresh.outpath) / "checkpoints" / R.RESUME_NAME
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    again = R.resolve_run_location(args, env=env)
    assert again.resume_path == ckpt
    assert again.outpath == fresh.outpath


def test_a_run_directory_without_a_checkpoint_is_reused_not_duplicated(tmp_path):
    """Preempted during dataset construction: same directory, epoch 1 again."""
    env = {"LSB_JOBID": "31"}
    args = make_args(job_name="j", output_dir=str(tmp_path), resume="auto")
    first = R.resolve_run_location(args, env=env)
    Path(first.outpath).mkdir(parents=True)

    again = R.resolve_run_location(args, env=env)
    assert again.outpath == first.outpath and again.resume_path is None
    assert "no checkpoint yet" in " ".join(again.notes)
    assert len(list(tmp_path.iterdir())) == 1


def test_explicit_run_directory_and_checkpoint_file(tmp_path):
    run_dir = tmp_path / "some_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / R.RESUME_NAME
    ckpt.write_bytes(b"")

    by_dir = R.resolve_run_location(make_args(job_name="j", resume=str(run_dir)))
    assert (by_dir.outpath, by_dir.resume_path) == (str(run_dir), ckpt)

    by_file = R.resolve_run_location(make_args(job_name="j", resume=str(ckpt)))
    # The run directory is recovered through the checkpoints/ convention.
    assert (by_file.outpath, by_file.resume_path) == (str(run_dir), ckpt)

    with pytest.raises(SystemExit):
        R.resolve_run_location(make_args(job_name="j", resume=str(tmp_path / "nope.pt")))


# -- atomic checkpoint I/O ------------------------------------------------------------------

def test_save_atomic_round_trips_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "checkpoints" / R.RESUME_NAME
    R.save_atomic({"format": R.RESUME_FORMAT, "epoch": 3,
                   "tensor": torch.arange(4.0)}, target)
    assert target.exists()
    assert [p.name for p in target.parent.iterdir()] == [R.RESUME_NAME]

    loaded = R.load_resume_checkpoint(target)
    assert R.is_full_resume(loaded) and loaded["epoch"] == 3
    assert torch.equal(loaded["tensor"], torch.arange(4.0))


def test_a_failed_write_leaves_the_previous_checkpoint_intact(tmp_path):
    target = tmp_path / R.RESUME_NAME
    R.save_atomic({"format": R.RESUME_FORMAT, "epoch": 1}, target)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        R.save_atomic({"bad": Unpicklable()}, target)
    assert R.load_resume_checkpoint(target)["epoch"] == 1
    assert list(tmp_path.iterdir()) == [target]  # the temp file was cleaned up


def test_corrupt_checkpoint_gives_a_readable_error(tmp_path):
    bad = tmp_path / R.RESUME_NAME
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit) as excinfo:
        R.load_resume_checkpoint(bad)
    assert "could not read" in str(excinfo.value)


# -- configuration compatibility --------------------------------------------------------------

def test_incompatible_configurations_fail_with_a_useful_error():
    saved = R.run_config(make_args(convlstm_unet=True, add_temporal=True, k_prevs=10,
                                   convlstm_hidden=256, batch_size=128))
    now = R.run_config(make_args(convlstm_unet=True, add_temporal=True, k_prevs=5,
                                 convlstm_hidden=256, batch_size=128))
    with pytest.raises(SystemExit) as excinfo:
        R.check_config_compatible(saved, now, "checkpoints/resume.pt")
    message = str(excinfo.value)
    assert "k_prevs" in message and "10" in message and "5" in message

    hidden = R.run_config(make_args(convlstm_unet=True, add_temporal=True, k_prevs=10,
                                    convlstm_hidden=1024, batch_size=128))
    with pytest.raises(SystemExit) as excinfo:
        R.check_config_compatible(saved, hidden, "resume.pt")
    assert "convlstm_hidden" in str(excinfo.value)


@pytest.mark.parametrize("field,value", [
    ("batch_size", 64),
    ("partition_file", "assets/other.json"),
    ("attn_unet", True),
    ("seed", 7),
])
def test_each_identity_setting_is_guarded(field, value):
    base = dict(convlstm_unet=True, add_temporal=True, k_prevs=10, batch_size=128,
                partition_mode="preset_by_intf", partition_file="assets/a.json", seed=42)
    saved = R.run_config(make_args(**base))
    changed = dict(base)
    changed[field] = value
    if field == "attn_unet":
        changed["convlstm_unet"] = False
    with pytest.raises(SystemExit):
        R.check_config_compatible(saved, R.run_config(make_args(**changed)), "resume.pt")


def test_more_epochs_is_allowed_and_merely_reported():
    saved = R.run_config(make_args(epochs=60, lr=1e-6))
    advisory = R.check_config_compatible(saved, R.run_config(make_args(epochs=80, lr=1e-6)),
                                         "resume.pt")
    assert any(line.startswith("epochs:") for line in advisory)


def test_validation_negatives_are_part_of_the_dataset_fingerprint():
    """Resuming across them would splice two differently-scored halves together."""
    saved = R.run_config(make_args(add_val_negatives=True))
    with pytest.raises(SystemExit):
        R.check_config_compatible(saved, R.run_config(make_args()), "resume.pt")


def test_a_run_without_validation_negatives_keeps_its_old_fingerprint():
    """The term is appended only when set, so pre-existing checkpoints still resume."""
    assert "valneg" not in R.run_config(make_args())["dataset"]


def test_a_checkpoint_without_a_recorded_setting_is_not_a_mismatch():
    now = R.run_config(make_args())
    assert R.check_config_compatible({}, now, "resume.pt") == []


# -- RNG and tracker state ---------------------------------------------------------------------

def test_rng_state_round_trips_across_python_numpy_and_torch():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    state = R.rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.random(), np.random.rand(), torch.rand(1)  # advance all three
    R.restore_rng_state(state)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_tracker_state_round_trips():
    tracker = BestTracker("val/dice", mode="max", patience=3)
    tracker.update(1, {"val/dice": 0.4})
    tracker.update(2, {"val/dice": 0.2})
    state = tracker.state_dict()

    restored = BestTracker("val/dice", mode="max", patience=3).load_state_dict(state)
    assert (restored.best, restored.best_epoch) == (0.4, 1)
    assert restored.update(3, {"val/dice": 0.3}) is False  # 0.3 is not a new best
    restored.update(4, {"val/dice": 0.1})
    assert restored.should_stop  # three epochs without improvement, counted across the resume


# -- results.csv ---------------------------------------------------------------------------------

def test_results_csv_keeps_completed_epochs_and_drops_the_repeated_one(tmp_path):
    columns = ["epoch", "val/dice"]
    path = tmp_path / "results.csv"
    original = ResultsCSV(path, columns)
    for epoch in (1, 2, 3):
        original.append({"epoch": f"{epoch}/10", "val/dice": epoch / 10})

    # Epoch 3 was logged but never checkpointed: it is about to be repeated.
    resumed = ResultsCSV(path, columns, keep_through=2)
    assert resumed.kept == 2
    assert resumed.backup is not None and resumed.backup.exists()
    resumed.append({"epoch": "3/10", "val/dice": 0.9})
    assert read_csv_epochs(path) == ["1/10", "2/10", "3/10"]


def test_results_csv_header_is_written_once(tmp_path):
    columns = ["epoch", "val/dice"]
    path = tmp_path / "results.csv"
    ResultsCSV(path, columns).append({"epoch": "1/2", "val/dice": 0.5})
    ResultsCSV(path, columns, keep_through=1).append({"epoch": "2/2", "val/dice": 0.6})
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == columns
    assert [r[0] for r in rows[1:]] == ["1/2", "2/2"]


# -- end to end ------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_run_resumed_after_epoch_two_starts_at_epoch_three(tmp_path):
    run_dir = tmp_path / "run"
    run_training(run_dir, epochs=2)

    checkpoint = R.load_resume_checkpoint(run_dir / "checkpoints" / R.RESUME_NAME)
    assert checkpoint["epoch"] == 2
    assert R.is_full_resume(checkpoint)
    assert read_csv_epochs(run_dir / "results.csv") == ["1/2", "2/2"]

    # Same directory, a higher total target: --epochs is a target, not an increment.
    run_training(run_dir, epochs=4, checkpoint=checkpoint)

    # The two completed rows are preserved verbatim; the new ones are appended.
    assert read_csv_epochs(run_dir / "results.csv") == ["1/2", "2/2", "3/4", "4/4"]
    final = R.load_resume_checkpoint(run_dir / "checkpoints" / R.RESUME_NAME)
    assert final["epoch"] == 4
    assert final["global_step"] > checkpoint["global_step"]


@pytest.mark.slow
def test_resume_restores_optimizer_scheduler_scaler_and_tracker(tmp_path):
    run_dir = tmp_path / "run"
    run_training(run_dir, epochs=2, lr_schedule="cosine")
    checkpoint = R.load_resume_checkpoint(run_dir / "checkpoints" / R.RESUME_NAME)

    # RMSprop keeps a square_avg buffer per parameter: a fresh optimizer has none.
    assert checkpoint["optimizer"]["state"], "optimizer momentum was not saved"
    assert "square_avg" in next(iter(checkpoint["optimizer"]["state"].values()))
    assert checkpoint["scheduler"]["last_epoch"] == 2
    # The scaler is recorded either way; its state is empty unless AMP is on,
    # and a disabled scaler must still accept it on the way back in.
    assert "scaler" in checkpoint
    torch.amp.GradScaler(enabled=False).load_state_dict(checkpoint["scaler"])
    assert checkpoint["tracker"]["best_epoch"] in (1, 2)
    assert checkpoint["rng"]["python"] is not None
    assert checkpoint["config"]["batch_size"] == 2
    assert R.learning_rate_of(checkpoint) is not None

    run_training(run_dir, epochs=3, checkpoint=checkpoint, lr_schedule="cosine")
    after = R.load_resume_checkpoint(run_dir / "checkpoints" / R.RESUME_NAME)
    # The cosine schedule kept counting from 2 instead of restarting.
    assert after["scheduler"]["last_epoch"] == 3
    assert after["best_score"] >= checkpoint["best_score"]


@pytest.mark.slow
def test_best_and_last_stay_loadable_by_the_existing_consumers(tmp_path):
    """best.pt / last.pt must keep the model-only format inference relies on."""
    from sinkholes.models.factory import build_from_checkpoint

    run_dir = tmp_path / "run"
    run_training(run_dir, epochs=1)

    for name in ("best.pt", "last.pt"):
        state = torch.load(run_dir / "checkpoints" / name, map_location="cpu",
                           weights_only=False)
        assert isinstance(state, dict) and "mask_values" in state
        assert "optimizer" not in state and "format" not in state
        loaded = build_from_checkpoint(state, n_classes=1)
        assert loaded.architecture == "unet"
        loaded.model.load_state_dict(state)


@pytest.mark.slow
def test_a_legacy_model_only_checkpoint_loads_weights_without_claiming_a_resume(tmp_path):
    run_dir = tmp_path / "run"
    run_training(run_dir, epochs=1)
    legacy = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu",
                        weights_only=False)
    assert not R.is_full_resume(legacy)

    model = tiny_model()
    before = model.outc.conv.weight.clone()
    R.load_legacy_weights(legacy, model)
    assert not torch.equal(before, model.outc.conv.weight)

    # Fed to train_model it must start a normal run at epoch 1, not pretend to resume.
    second = tmp_path / "legacy_run"
    run_training(second, epochs=1, checkpoint=legacy)
    assert read_csv_epochs(second / "results.csv") == ["1/1"]


@pytest.mark.slow
def test_a_legacy_checkpoint_never_writes_into_the_run_it_came_from(tmp_path, monkeypatch):
    """Weights-only is a new experiment, so it gets a new (timestamped) directory."""
    import sinkholes.training.train as train_module

    outputs = tmp_path / "outputs"
    source = outputs / "original"
    run_training(source, epochs=1)
    before = (source / "results.csv").read_text()

    monkeypatch.setattr(train_module, "build_datasets",
                        lambda args, rep: (TinySet.make(), TinySet.make(seed=1), None, "stub"))
    monkeypatch.setattr(train_module, "build_model",
                        lambda args, device: (tiny_model().to(device), 1))
    train_module.main(make_args(
        job_name="legacy", output_dir=str(outputs), epochs=1, batch_size=2, lr=1e-3,
        sample_every=0, save_best_only=True,
        resume=str(source / "checkpoints" / "last.pt")))

    assert (source / "results.csv").read_text() == before, "the source run was overwritten"
    fresh = [p for p in outputs.iterdir() if p.name.startswith("legacy_")]
    assert len(fresh) == 1 and read_csv_epochs(fresh[0] / "results.csv") == ["1/1"]


@pytest.mark.slow
def test_an_already_complete_run_is_not_retrained(tmp_path):
    run_dir = tmp_path / "run"
    run_training(run_dir, epochs=2)
    checkpoint = R.load_resume_checkpoint(run_dir / "checkpoints" / R.RESUME_NAME)

    run_training(run_dir, epochs=2, checkpoint=checkpoint)
    # The loop had nothing to do, so the history is untouched.
    assert read_csv_epochs(run_dir / "results.csv") == ["1/2", "2/2"]


# -- preemption handling ------------------------------------------------------------------------------

@pytest.mark.parametrize("signame", ["SIGINT", "SIGTERM"])
def test_the_guard_turns_preemption_signals_into_a_keyboard_interrupt(signame):
    import signal

    sig = getattr(signal, signame)
    before = signal.getsignal(sig)
    with R.PreemptionGuard() as guard:
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), sig)
        assert guard.triggered and guard.signal_name == signame
    # Handlers are restored on exit, so nothing leaks into the rest of the suite.
    assert signal.getsignal(sig) is before


def test_a_signal_during_a_checkpoint_write_is_deferred_until_it_finishes(tmp_path):
    import signal

    target = tmp_path / R.RESUME_NAME
    with R.PreemptionGuard() as guard:
        with pytest.raises(KeyboardInterrupt):
            with guard.critical("resume.pt"):
                os.kill(os.getpid(), signal.SIGTERM)
                # Still running: the write must be allowed to complete.
                R.save_atomic({"format": R.RESUME_FORMAT, "epoch": 7}, target)
    assert R.load_resume_checkpoint(target)["epoch"] == 7


@pytest.mark.slow
def test_main_with_resume_auto_checks_the_checkpoint_before_building_datasets(
        tmp_path, monkeypatch):
    """The expensive part of a run must not happen when the answer is already known."""
    import sinkholes.training.train as train_module

    outputs = tmp_path / "outputs"
    run_training(outputs / "tiny_2026-01-01_09h00_lsf_777", epochs=2)
    monkeypatch.setenv("LSB_JOBID", "777")

    def must_not_run(*a, **kw):
        raise AssertionError("datasets were built for a run with nothing left to do")

    monkeypatch.setattr(train_module, "build_datasets", must_not_run)
    common = dict(job_name="tiny", output_dir=str(outputs), resume="auto",
                  batch_size=2, lr=1e-3, sample_every=0, save_best_only=True)

    # Two epochs requested, two epochs done: report and exit successfully.
    train_module.main(make_args(epochs=2, **common))

    # A different batch size is a different experiment, and must not load silently.
    with pytest.raises(SystemExit) as excinfo:
        train_module.main(make_args(epochs=4, **{**common, "batch_size": 8}))
    assert "batch_size" in str(excinfo.value)


@pytest.mark.slow
def test_an_interrupted_epoch_is_repeated_not_recorded(tmp_path):
    """A preemption mid-epoch must leave resume.pt describing the previous one."""
    import sinkholes.training.train as train_module

    run_dir = tmp_path / "run"
    run_training(run_dir, epochs=1)
    checkpoint = R.load_resume_checkpoint(run_dir / "checkpoints" / R.RESUME_NAME)
    assert checkpoint["epoch"] == 1

    real_evaluate = train_module.evaluate
    calls = {"n": 0}

    def evaluate_then_preempt(*a, **kw):
        calls["n"] += 1
        raise KeyboardInterrupt("SIGTERM")

    train_module.evaluate = evaluate_then_preempt
    try:
        run_training(run_dir, epochs=3, checkpoint=checkpoint)
    finally:
        train_module.evaluate = real_evaluate

    assert calls["n"] == 1
    still = R.load_resume_checkpoint(run_dir / "checkpoints" / R.RESUME_NAME)
    assert still["epoch"] == 1, "a partial epoch overwrote a valid checkpoint"
    assert read_csv_epochs(run_dir / "results.csv") == ["1/1"]
    assert (run_dir / "checkpoints" / "interrupted.pt").exists()
