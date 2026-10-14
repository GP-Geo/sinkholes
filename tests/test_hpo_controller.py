"""The study controller, end to end, against a fake LSF.

``bsub``/``bjobs`` are small scripts in the test's temp dir; "running" a job
means writing what the real job would write (snapshots + run_status.json for
training, eval-snapshots result files for evaluation). The whole protocol is
driven through every phase -- approvals, duplicate guards, stalls and
retries, Optuna's TPE asks and tells, promotion by objective, seed repeats,
the locked threshold, retention, the report and the panel.
"""

import argparse
import json
import os
import shlex
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("optuna")

from sinkholes.hpo import controller as C
from sinkholes.hpo import jobs as J
from sinkholes.hpo import objective as O
from sinkholes.hpo.ledger import Ledger, LedgerError
from sinkholes.hpo.spec import load_spec

REPO = Path(__file__).resolve().parents[1]
RTH = ["0.125", "0.25", "0.375", "0.5"]

FAKE_BSUB = r'''#!{py}
import os, sys, json, re
state = os.environ["FAKE_LSF_STATE"]
d = json.load(open(state)) if os.path.exists(state) else {{"next": 1000, "jobs": {{}}}}
argv = sys.argv[1:]
name = argv[argv.index("-J") + 1]
jid = str(d["next"]); d["next"] += 1
d["jobs"][jid] = {{"name": name, "stat": "PEND", "argv": argv, "script": sys.stdin.read()}}
json.dump(d, open(state, "w"))
print(f"Job <{{jid}}> is submitted to queue <q>.")
'''

FAKE_BJOBS = r'''#!{py}
import os, sys, json
state = os.environ["FAKE_LSF_STATE"]
d = json.load(open(state)) if os.path.exists(state) else {{"jobs": {{}}}}
argv = sys.argv[1:]
want_name = argv[argv.index("-J") + 1] if "-J" in argv else None
ids = [a for a in argv if a.isdigit()]
for jid, j in d["jobs"].items():
    if (want_name and j["name"] == want_name) or jid in ids:
        print(f"{{jid}}|{{j['stat']}}|{{j['name']}}")
'''


@pytest.fixture
def lsf(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("bsub", FAKE_BSUB), ("bjobs", FAKE_BJOBS)):
        p = bindir / name
        p.write_text(body.format(py=sys.executable))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    state = tmp_path / "lsf.json"
    monkeypatch.setenv("FAKE_LSF_STATE", str(state))
    monkeypatch.setenv("BSUB", str(bindir / "bsub"))
    monkeypatch.setenv("BJOBS", str(bindir / "bjobs"))

    class LSF:
        def jobs(self):
            return json.loads(state.read_text())["jobs"] if state.exists() else {}

        def set(self, jid, stat_):
            d = json.loads(state.read_text())
            d["jobs"][jid]["stat"] = stat_
            state.write_text(json.dumps(d))

        def submit_external(self, name):
            d = json.loads(state.read_text()) if state.exists() else {"next": 1000, "jobs": {}}
            jid = str(d["next"]); d["next"] += 1
            d["jobs"][jid] = {"name": name, "stat": "RUN", "argv": [], "script": ""}
            state.write_text(json.dumps(d))
    return LSF()


@pytest.fixture
def spec(tmp_path):
    raw = json.loads((REPO / "assets/hpo/k10t2026_plain_vs_ctx50.json").read_text())
    raw["root"] = str(tmp_path / "study")
    raw["repo"] = str(REPO)
    raw["patches_dir"] = str(tmp_path / "patches")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    return load_spec(path)


def ledger_of(spec):
    return C.open_ledger(spec)


# -- the synthetic "cluster" ----------------------------------------------------------------

def quality(params, epoch, rth):
    """A smooth, known response surface, so the controller's choices are checkable."""
    lr, wd, pw = params["lr"], params["weight_decay"], params["pos_w"]
    base = 0.80 - 60 * (np.log10(lr) + 3.5) ** 2 * 1e-3 - 0.01 * abs(np.log10(wd) + 2) - 0.002 * abs(pw - 8)
    return float(base + 0.0005 * epoch - 0.03 * abs(float(rth) - 0.25))


def run_jobs(spec, lsf_):
    """Every PEND job runs to completion, writing what the real job writes."""
    led = ledger_of(spec)
    by_name = {j["name"]: (t, j) for t in led.trials.values() for j in t["jobs"]}
    for jid, j in lsf_.jobs().items():
        if j["stat"] != "PEND" or j["name"] not in by_name:
            if j["stat"] == "PEND":
                lsf_.set(jid, "DONE")
            continue
        t, job = by_name[j["name"]]
        if job["kind"] == "train":
            run = Path(t["run_dir"])
            snaps = run / "checkpoints" / "snapshots"
            snaps.mkdir(parents=True, exist_ok=True)
            for e in t["snapshot_epochs"]:
                if e <= job["target_epochs"] and not (snaps / f"epoch_{e:03d}.pt").exists():
                    torch.save({"w": torch.tensor([float(e)])}, snaps / f"epoch_{e:03d}.pt")
            for f in ("best.pt", "last.pt", "resume.pt", "resume_prev.pt"):
                torch.save({"w": 1}, run / "checkpoints" / f)
            (run / "run_status.json").write_text(json.dumps(
                {"state": "complete", "completed_epoch": job["target_epochs"],
                 "target_epochs": job["target_epochs"]}))
        else:
            out = Path(job["out_dir"])
            (out / "maps").mkdir(parents=True, exist_ok=True)
            for e in job["epochs"]:
                summary = {}
                for r in RTH:
                    f1 = quality(t["params"], e, r)
                    summary[r] = {"f1": f1, "mean_precision": f1 + 0.02, "mean_recall": f1 - 0.02,
                                  "weighted_f1": f1, "n_scenes": 11, "n_background": 3,
                                  "background_area": 50.0}
                (out / f"e{e:03d}.json").write_text(json.dumps({
                    "summary": summary, "per_scene": {},
                    "checkpoint": {"path": J.snapshot_file(t, e), "sha256": "x" * 64}}))
                (out / "maps" / f"e{e:03d}").mkdir(exist_ok=True)
                (out / "maps" / f"e{e:03d}" / "s.npz").write_bytes(b"m")
        lsf_.set(jid, "DONE")


def step(spec, **kw):
    return C.step(spec, coverage_check=False, **kw)


# -- init ----------------------------------------------------------------------------------

def test_init_creates_the_shared_trials_and_submits_nothing(spec, lsf):
    import optuna

    C.init(spec, spec.path)
    led = ledger_of(spec)
    assert sorted(led.trials) == sorted(f"{g}/s{i}" for g in ("plain", "ctx50") for i in range(6))
    assert led.trial("plain/s0")["params"] == {"lr": 3e-4, "weight_decay": 1e-2, "pos_w": 8}
    assert led.trial("plain/s0")["phase"] == "baseline" and led.trial("plain/s1")["phase"] == "shared"
    assert led.trial("plain/s3")["params"] == led.trial("ctx50/s3")["params"]   # shared means shared
    assert lsf.jobs() == {}
    from sinkholes.hpo import optuna_io as Q
    st = Q.open_study(spec, "plain", "e30")
    assert [t.state for t in st.trials] == [optuna.trial.TrialState.RUNNING] * 6
    assert [t.params for t in st.trials] == spec.shared
    assert (Path(spec.root) / "PROTECTED_STUDY_STATE").exists()
    with pytest.raises(LedgerError, match="already exists"):
        C.init(spec, spec.path)


def test_a_half_finished_init_is_not_repeated(spec, lsf):
    Path(spec.root).mkdir(parents=True)
    (Path(spec.root) / "optuna.journal").write_text("")
    with pytest.raises(LedgerError, match="half-way"):
        C.init(spec, spec.path)


def test_a_changed_spec_is_refused(spec, tmp_path, lsf):
    C.init(spec, spec.path)
    raw = dict(spec.raw, budget=dict(spec.budget, initial_epochs=31))
    (tmp_path / "spec2.json").write_text(json.dumps(raw))
    with pytest.raises(SystemExit, match="not the one"):
        C.open_ledger(load_spec(tmp_path / "spec2.json"))


# -- the commands it writes ----------------------------------------------------------------

def _parse(add_arguments, argv):
    p = argparse.ArgumentParser()
    add_arguments(p)
    return p.parse_args(argv)


@pytest.mark.parametrize("geometry", ["plain", "ctx50"])
def test_the_training_command_is_valid_and_is_the_baseline(spec, lsf, geometry):
    from sinkholes.training.train import add_arguments, validate_schedule_args

    C.init(spec, spec.path)
    t = ledger_of(spec).trial(f"{geometry}/s0")
    argv = J.train_argv(spec, t, time_budget_hours=22.5)
    assert argv[:4] == ["python", "-m", "sinkholes", "train"]
    a = _parse(add_arguments, argv[4:])
    validate_schedule_args(a)
    assert (a.lr, a.weight_decay, a.pos_w, a.seed, a.epochs) == (3e-4, 1e-2, 8.0, 42, 30)
    assert (a.optimizer, a.beta1, a.beta2, a.seg_loss, a.augment_flips) == ("adamw", 0.9, 0.999, "dice", "hv")
    assert (a.batch_size, a.accum_steps, a.amp, a.patience) == (32, 4, True, 0)
    assert (a.lr_schedule, a.lr_patience, a.lr_factor, a.min_lr) == ("plateau", 8, 0.5, 1e-8)
    assert (a.add_ring_negatives, a.neg_ring_inner, a.neg_ring_outer, a.neg_per_pos) == (True, 1, 10, 1.0)
    assert (a.tattn_unet, a.tattn_dim, a.tattn_heads, a.tattn_layers, a.tattn_recurrence,
            a.tattn_fuse_skips, a.tattn_contrast, a.tattn_qk_norm) == (True, 0, 8, 1, "none", 0, True, True)
    assert (a.k_prevs, a.add_temporal, a.stride, a.patch_size) == (10, True, 2, [200, 100])
    assert a.context_margin == ([50, 50] if geometry == "ctx50" else [0, 0])
    # the study's frozen copy, not the checkout's: a moved code tree must not break a resume
    assert a.partition_file == f"{spec.root}/inputs/partition_temporal_k10_test2026_fill_th350x200.json"
    assert Path(a.partition_file).read_bytes() == (
        REPO / "assets/partition_temporal_k10_test2026_fill_th350x200.json").read_bytes()
    assert (a.snapshot_epochs, a.keep_prev_resume, a.eval_rng_isolation, a.strict_resume) == \
        ("10,20,30", True, True, True)
    assert a.resume == t["run_dir"] and a.save_best_only and a.fit_eval_samples == 2048


def test_the_evaluation_command_is_valid(spec, lsf):
    from sinkholes.inference.snapshot_eval import add_arguments

    C.init(spec, spec.path)
    t = ledger_of(spec).trial("ctx50/s0")
    argv = J.eval_argv(spec, t, [10, 20, 30], "/out")
    a = _parse(add_arguments, argv[4:])
    assert [c.split("=")[0] for c in a.checkpoint] == ["e010", "e020", "e030"]
    assert a.rth == [0.125, 0.25, 0.375, 0.5] and (a.ith, a.buffer) == (0.7, 5)
    assert (a.split, a.aoi_split, a.data_stride, a.min_positives, a.k_prevs) == ("val", "val", 4, 150, 10)
    assert a.tile_batch == 16 and a.save_maps


# -- approvals, submission, duplicates -----------------------------------------------------

def test_dry_run_changes_nothing_and_unapproved_phases_are_not_submitted(spec, lsf):
    C.init(spec, spec.path)
    before = (Path(spec.root) / "ledger.json").read_text()
    lines = step(spec, submit=False)
    assert sum("NEEDS APPROVAL (baseline)" in l for l in lines) == 2
    assert sum("NEEDS APPROVAL (shared)" in l for l in lines) == 10
    assert (Path(spec.root) / "ledger.json").read_text() == before
    lines = step(spec, submit=True)
    assert lsf.jobs() == {} and not any("SUBMITTED" in l for l in lines)


def test_the_baseline_pair_and_nothing_else(spec, lsf):
    C.init(spec, spec.path)
    C.approve(spec, ["baseline"])
    lines = step(spec, submit=True)
    submitted = sorted(j["name"] for j in lsf.jobs().values())
    assert submitted == ["k10t2026_hpo_v1-ctx50-s0-train-c1", "k10t2026_hpo_v1-plain-s0-train-c1"]
    led = ledger_of(spec)
    job = led.trial("plain/s0")["jobs"][0]
    assert job["job_id"] and job["state"] == "submitted" and Path(job["script"]).exists()
    script = " ".join(Path(job["script"]).read_text().replace("\\\n", " ").replace("'", "").split())
    assert "--resume " + led.trial("plain/s0")["run_dir"] in script
    assert "-gpu" in job["argv"] and "rusage[mem=160GB]" in job["argv"]
    ctx = ledger_of(spec).trial("ctx50/s0")["jobs"][0]
    assert "rusage[mem=340GB]" in ctx["argv"] and "num=1:j_exclusive=yes:gmem=40G" in ctx["argv"]

    # still pending: a second step submits nothing
    step(spec, submit=True)
    assert len(lsf.jobs()) == 2


def test_a_job_that_ends_without_progress_needs_retry(spec, lsf):
    C.init(spec, spec.path)
    C.approve(spec, ["baseline"])
    step(spec, submit=True, only=["plain/s0"])
    jid = next(iter(lsf.jobs()))
    lsf.set(jid, "EXIT")                      # died before finishing an epoch
    lines = step(spec, submit=True, only=["plain/s0"])
    assert any("BLOCKED" in l and "--retry plain/s0" in l for l in lines)
    assert len(lsf.jobs()) == 1
    step(spec, submit=True, only=["plain/s0"], retry=["plain/s0"])
    names = sorted(j["name"] for j in lsf.jobs().values())
    assert names == ["k10t2026_hpo_v1-plain-s0-train-c1", "k10t2026_hpo_v1-plain-s0-train-c2"]


def test_a_paused_chunk_is_continued_automatically(spec, lsf):
    C.init(spec, spec.path)
    C.approve(spec, ["baseline"])
    step(spec, submit=True, only=["plain/s0"])
    jid = next(iter(lsf.jobs()))
    t = ledger_of(spec).trial("plain/s0")
    Path(t["run_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(t["run_dir"]) / "run_status.json").write_text(
        json.dumps({"state": "paused", "completed_epoch": 17, "target_epochs": 30}))
    lsf.set(jid, "DONE")
    step(spec, submit=True, only=["plain/s0"])
    job = ledger_of(spec).trial("plain/s0")["jobs"][-1]
    assert job["chunk"] == 2 and job["start_epoch"] == 17


def test_a_job_already_in_lsf_is_never_submitted_twice(spec, lsf):
    C.init(spec, spec.path)
    C.approve(spec, ["baseline"])
    lsf.submit_external("k10t2026_hpo_v1-plain-s0-train-c1")   # e.g. submitted by hand
    lines = step(spec, submit=True, only=["plain/s0"])
    assert any("SKIP" in l for l in lines) and len(lsf.jobs()) == 1


def test_one_controller_at_a_time(spec, lsf):
    C.init(spec, spec.path)
    (Path(spec.root) / ".controller.lock").write_text("{}")
    C.approve.__globals__  # noqa -- approve also takes the lock
    with pytest.raises(LedgerError, match="another controller"):
        C.approve(spec, ["baseline"])


# -- the whole study -----------------------------------------------------------------------

def drive(spec, lsf_, rounds=30):
    for _ in range(rounds):
        step(spec, submit=True)
        if not any(j["stat"] == "PEND" for j in lsf_.jobs().values()):
            step(spec, submit=True)
            if not any(j["stat"] == "PEND" for j in lsf_.jobs().values()):
                return
        run_jobs(spec, lsf_)


def test_the_full_protocol(spec, lsf):
    import optuna

    from sinkholes.hpo import optuna_io as Q

    C.init(spec, spec.path)
    C.approve(spec, ["baseline", "shared"])
    drive(spec, lsf)
    led = ledger_of(spec)
    # six scored per geometry; TPE has not been asked without approval
    assert all(C.E30 in led.trial(f"{g}/s{i}")["objectives"] for g in ("plain", "ctx50") for i in range(6))
    assert not any(t["origin"] == "tpe" for t in led.trials.values())
    o = led.trial("plain/s0")["objectives"]["e30"]
    expected = max(np.mean([quality(led.trial("plain/s0")["params"], e, r) for e in (20, 30)]) for r in RTH)
    assert o["value"] == pytest.approx(expected) and o["rth"] == 0.25

    C.approve(spec, ["tpe"])
    drive(spec, lsf)
    led = ledger_of(spec)
    tpe = [t for t in led.trials.values() if t["origin"] == "tpe"]
    assert len(tpe) == 4 and all(C.E30 in t["objectives"] for t in tpe)
    for g in ("plain", "ctx50"):
        st = Q.open_study(spec, g, "e30")
        assert [t.state for t in st.trials] == [optuna.trial.TrialState.COMPLETE] * 8
    assert not C._decided(led, "promote", "plain")     # promote not approved yet

    C.approve(spec, ["promote", "seeds"])
    drive(spec, lsf)
    led = ledger_of(spec)
    for g in ("plain", "ctx50"):
        dec = C._decided(led, "promote", g)
        ranked = O.rank(C._initial(led, g), "e30")
        assert dec["trials"] == [t["key"] for t in ranked[:2]]
        for k in dec["trials"]:
            t = led.trial(k)
            assert t["target_epochs"] == 60 and C.E60 in t["objectives"]
            chunk2 = [j for j in t["jobs"] if j["kind"] == "train"][-1]
            assert chunk2["start_epoch"] == 30 and chunk2["target_epochs"] == 60
            text = " ".join(Path(chunk2["script"]).read_text().replace("\\\n", " ").split())
            assert "--epochs 60" in text and "--snapshot_epochs 10,20,30,40,50,60" in text
        seeds = [t for t in led.trials.values() if t["geometry"] == g and t["origin"] == "seed_repeat"]
        assert sorted(t["seed"] for t in seeds) == [7, 123]
        best = O.rank([led.trial(k) for k in dec["trials"]], "e60")[0]
        assert all(t["repeat_of"] == best["key"] and t["params"] == best["params"] for t in seeds)
        lock = led.data["locked"][g]
        assert lock["config_of"] == best["key"] and lock["rth"] == 0.25 and lock["seeds"] == [42, 7, 123]
        e60 = Q.open_study(spec, g, "e60")
        assert len(e60.trials) == 4          # 2 promoted + 2 seed repeats, never in e30
        assert len(Q.open_study(spec, g, "e30").trials) == 8

    # nothing left to do, and nothing was ever submitted twice
    names = [j["name"] for j in lsf.jobs().values()]
    assert len(names) == len(set(names))
    assert not any("WOULD" in l or "SUBMITTED" in l for l in step(spec, submit=False))

    # retention: dry run lists, apply deletes only evaluated non-best snapshots
    from sinkholes.hpo.retention import plan, retain

    actions = plan(spec, led)
    deleted = [str(p) for a, p, _, _ in actions if a == "delete"]
    promoted = set(C._decided(led, "promote", "plain")["trials"])
    for p in deleted:
        assert not any(led.trial(k)["run_dir"] in p for k in promoted) or "maps" not in p
        assert "resume.pt" not in p and "best.pt" not in p and "last.pt" not in p
    retain(spec, apply=True)
    led = ledger_of(spec)
    s5 = led.trial("plain/s5")
    if "plain/s5" not in promoted:
        snaps = sorted(p.name for p in (Path(s5["run_dir"]) / "checkpoints" / "snapshots").iterdir())
        assert snaps == [f"epoch_{s5['best_scene']['epoch']:03d}.pt"]
        assert (Path(s5["run_dir"]) / "checkpoints" / "best_scene.pt").exists()
        assert (Path(s5["run_dir"]) / "checkpoints" / "resume.pt").exists()

    # long runs are only created from a locked config, and wait for approval
    made = C.create_long(spec, led, "plain", [42], "long")
    led.save()
    assert made == ["plain/long-seed42"]
    lt = ledger_of(spec).trial("plain/long-seed42")
    assert lt["schedule"] == {"lr_schedule": "cosine", "lr_horizon": 300,
                              "min_lr": pytest.approx(lt["params"]["lr"] * 0.01)}
    lines = step(spec, submit=False, only=["plain/long-seed42"])
    assert any("NEEDS APPROVAL (long)" in l for l in lines)
    from sinkholes.training.train import add_arguments, validate_schedule_args
    a = _parse(add_arguments, J.train_argv(spec, lt, time_budget_hours=22.5)[4:])
    validate_schedule_args(a)
    assert (a.lr_schedule, a.lr_horizon, a.epochs) == ("cosine", 300, 300)

    from sinkholes.hpo.report import report
    out = report(spec)
    assert "report ->" in out


# -- objective and pattern -----------------------------------------------------------------

def test_the_objective_uses_one_threshold_for_both_epochs():
    scores = {"20": {"0.125": {"f1": 0.80}, "0.25": {"f1": 0.70}},
              "30": {"0.125": {"f1": 0.60}, "0.25": {"f1": 0.75}}}
    o = O.objective(scores, [20, 30], [0.125, 0.25])
    # per-epoch best would give (0.80 + 0.75)/2 = 0.775; one threshold gives 0.725
    assert o["value"] == pytest.approx(0.725) and o["rth"] == 0.25
    assert O.objective(scores, [20, 30, 40], [0.125]) is None


def test_undefined_f1_counts_as_zero():
    scores = {"20": {"0.5": {"f1": None}}, "30": {"0.5": {"f1": 0.6}}}
    assert O.objective(scores, [20, 30], [0.5])["value"] == pytest.approx(0.3)


@pytest.mark.parametrize("vals,expect", [
    ([.70, .72, .74, .75], None),                                   # monotone
    ([.70, .76, .74, .70, .73, .78], ("peak", 2, "trough", 4, True)),
    ([.70, .76, .70, .69], ("peak", 2, "trough", 4, False)),
    ([.70, .76, .758, .77], None),                                   # dip inside the noise floor
])
def test_the_dip_and_recovery_detector(vals, expect):
    from sinkholes.hpo.report import pattern

    pts = list(zip(range(1, len(vals) + 1), vals))
    p = pattern(pts)
    if expect is None:
        assert p is None
    else:
        assert p["peak"][0] == expect[1] and p["trough"][0] == expect[3]
        assert p["recovered"] is expect[4]


def test_a_submission_lost_in_a_crash_is_adopted_not_repeated(spec, lsf):
    """The ledger entry is written before bsub; the controller dies before the id."""
    C.init(spec, spec.path)
    C.approve(spec, ["baseline"])
    led = ledger_of(spec)
    t = led.trial("plain/s0")
    name = "k10t2026_hpo_v1-plain-s0-train-c1"
    t["jobs"].append({"kind": "train", "name": name, "state": "submitting", "chunk": 1,
                      "start_epoch": 0, "target_epochs": 30})
    led.save()
    lsf.submit_external(name)                  # ... but bsub had gone through
    step(spec, submit=True, only=["plain/s0"])
    assert len(lsf.jobs()) == 1                # adopted, not resubmitted
    job = ledger_of(spec).trial("plain/s0")["jobs"][0]
    assert job["job_id"] and "adopted" in job["state"]


def test_a_failed_bsub_is_not_a_stall(spec, lsf, monkeypatch):
    """A rejected submission never ran, so the next step simply submits again."""
    C.init(spec, spec.path)
    C.approve(spec, ["baseline"])
    from sinkholes.hpo import lsf as L

    real, calls = L.submit, []

    def flaky(argv, script):
        calls.append(argv)
        if len(calls) == 1:
            raise L.SubmitError("queue closed")
        return real(argv, script)
    monkeypatch.setattr(L, "submit", flaky)
    lines = step(spec, submit=True, only=["plain/s0"])
    assert any("FAILED" in l for l in lines) and lsf.jobs() == {}
    lines = step(spec, submit=True, only=["plain/s0"])        # no --retry needed
    assert any("SUBMITTED" in l for l in lines)
    assert [j["name"] for j in lsf.jobs().values()] == ["k10t2026_hpo_v1-plain-s0-train-c2"]
