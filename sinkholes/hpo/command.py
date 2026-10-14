"""``sinkholes hpo <step>`` -- the study's command line. See controller.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SPEC = "assets/hpo/k10t2026_plain_vs_ctx50.json"


def add_arguments(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="step", required=True, metavar="step")

    def cmd(name, help_text):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--spec", default=DEFAULT_SPEC, help="the study specification JSON")
        return s

    s = cmd("init", "create the ledger, Optuna studies and the shared trials (submits nothing)")
    s.add_argument("--repo", default=None,
                   help="code tree the jobs run from (default: this checkout, cluster path)")

    s = cmd("status", "every trial: stage, epochs, scores, objectives")
    s.add_argument("--json", action="store_true")

    s = cmd("approve", "allow submissions for one or more phases (recorded in the ledger)")
    s.add_argument("--phase", action="append", required=True)
    s.add_argument("--revoke", action="store_true")
    s.add_argument("--note", default="")

    s = cmd("step", "collect results and list (or --submit) the next approved actions")
    s.add_argument("--submit", action="store_true", help="actually submit; default is a dry run")
    s.add_argument("--only", action="append", default=[], metavar="TRIAL",
                   help="restrict to these trial keys, e.g. plain/s0 (repeatable)")
    s.add_argument("--retry", action="append", default=[], metavar="TRIAL",
                   help="resubmit a trial whose last job made no progress")
    s.add_argument("--break_lock", action="store_true")
    s.add_argument("--no_coverage_check", action="store_true")

    cmd("collect", "read finished evaluations into the ledger and Optuna (submits nothing)")

    s = cmd("coverage", "check every grid both geometries need; print generation commands")
    s.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    s.add_argument("--shallow", action="store_true", help="existence only, skip header checks")
    s.add_argument("--json", default=None, help="also write the report here")

    s = cmd("cache-minmax", "precompute the per-grid min/max the evaluator needs (CPU job body)")
    s.add_argument("--geometry", action="append", default=None)
    s.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])

    s = cmd("prep", "list or submit the utility jobs (min/max caches, the GPU profile)")
    s.add_argument("--submit", action="store_true")
    s.add_argument("--what", nargs="+", default=["minmax", "profile"],
                   choices=["minmax", "profile"])

    s = cmd("profile", "GPU memory and step time per geometry and micro-batch (job body)")
    s.add_argument("--microbatch", type=int, nargs="+", default=[32, 24, 16])
    s.add_argument("--steps", type=int, default=12)
    s.add_argument("--out", default=None)

    s = cmd("long", "create the 300-epoch runs of a geometry's locked config (submits nothing)")
    s.add_argument("--geometry", action="append", required=True)
    s.add_argument("--seeds", type=int, nargs="+", required=True)
    s.add_argument("--constant_lr", action="store_true", help="the constant-LR control arm")

    s = cmd("retain", "apply the retention policy (dry run unless --apply)")
    s.add_argument("--apply", action="store_true")
    s.add_argument("--drop_unpromoted_resume", action="store_true",
                   help="also delete resume checkpoints of trials that can no longer be promoted")

    s = cmd("report", "tables, learning curves and the threshold record")
    s.add_argument("--out", default=None)

    s = cmd("panel", "the fixed, anonymised validation visual panel")
    s.add_argument("--trials", nargs="+", required=True,
                   metavar="KEY@EPOCH", help="e.g. plain/s0@30 ctx50/s0@30")
    s.add_argument("--rth", type=float, default=None,
                   help="threshold to draw (default: each trial's locked or best RTh)")
    s.add_argument("--n_windows", type=int, default=6)
    s.add_argument("--seed", type=int, default=20261013)
    s.add_argument("--out", default=None)

    cmd("resources", "resource and storage estimates for every phase")

    s = cmd("preview", "render a shared trial's job script and bsub line from the spec alone "
                       "(no ledger, no Optuna, nothing written unless --write)")
    s.add_argument("--design", type=int, default=0, help="shared config index (0 = baseline)")
    s.add_argument("--geometry", action="append", default=None)
    s.add_argument("--repo", default=None)
    s.add_argument("--write", default=None, metavar="DIR", help="also write the scripts here")


def main(args) -> None:
    from . import controller as C
    from .spec import load_spec, localize

    spec = load_spec(args.spec)
    step = args.step

    if step == "init":
        ledger = C.init(spec, args.spec, repo=args.repo)
        print(f"initialised {spec.name} at {ledger.root}: {len(ledger.trials)} trials, nothing "
              f"submitted. Next: `sinkholes hpo coverage`, `approve --phase ...`, `step`.")
        print(C.format_table(C.status_rows(spec, ledger)))
        return

    if step == "status":
        ledger = C.open_ledger(spec)
        rows = C.status_rows(spec, ledger)
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            print(f"study {spec.name}  root {ledger.root}  repo {ledger.data['repo']}")
            print(f"approved phases: {ledger.data.get('approved_phases') or 'none'}")
            print(C.format_table(rows))
            for d in ledger.data.get("decisions", []):
                print(f"decision {d['time']}: {d['what']} {d['geometry']} "
                      f"{d.get('trials') or d.get('config_of') or ''}")
            for g, lock in ledger.data.get("locked", {}).items():
                print(f"LOCKED {g}: {lock['config_of']} RTh {lock['rth']} (val F1 {lock['value']:.4f})")
        return

    if step == "approve":
        for line in C.approve(spec, args.phase, revoke=args.revoke, note=args.note):
            print(line)
        return

    if step == "step":
        lines = C.step(spec, submit=args.submit, only=args.only, retry=args.retry,
                       break_lock=args.break_lock, coverage_check=not args.no_coverage_check)
        print("\n".join(lines) if lines else "nothing to do")
        if not args.submit:
            print("\nDRY RUN -- nothing was submitted or recorded. Add --submit on a WEXAC login node.")
        return

    if step == "collect":
        ledger = C.open_ledger(spec)
        with ledger.locked():
            C.open_ledger(spec)
            notes = C.collect(spec, ledger, tell=True)
            ledger.save()
        print("\n".join(notes) if notes else "nothing new")
        return

    if step == "coverage":
        from .coverage import format_report, minmax_status, report

        rep = report(spec, splits=tuple(args.splits), deep=not args.shallow)
        print(format_report(rep))
        for g in spec.geometries:
            try:
                m = minmax_status(spec, g, "val")
                print(f"min/max cache {g} val: {m['cached']}/{m['needed']} grids")
            except Exception as exc:  # the cache is optional
                print(f"min/max cache {g}: {exc}")
        if args.json:
            Path(args.json).write_text(json.dumps(rep, indent=1))
        if not all(r["ok"] for r in rep["requirements"]):
            sys.exit(2)
        return

    if step == "cache-minmax":
        from ..dataprep.patchify import patch_file_name
        from ..inference.snapshot_eval import cell_minmax
        from .coverage import _dirs, eval_ids

        H, W = spec.fixed["patch_size"]
        stride = int(spec.evaluation["data_stride"])
        cache = localize(f"{spec.root}/cache")
        for g in args.geometry or spec.geometries:
            data_dir, _ = _dirs(spec, g, stride)
            ids = sorted({i for s in args.splits for i in eval_ids(spec, s)[1]})
            for n, i in enumerate(ids, 1):
                f = Path(data_dir) / patch_file_name("data", i, H, W, stride)
                if not f.exists():
                    print(f"[{g} {n}/{len(ids)}] {i}: grid absent -- skipped (build it first)")
                    continue
                cell_minmax(f, cache)
                print(f"[{g} {n}/{len(ids)}] {i}: cached", flush=True)
        return

    if step == "prep":
        from .prep import prep

        print("\n".join(prep(spec, args.spec, submit=args.submit, what=args.what)))
        return

    if step == "profile":
        from .profile import profile

        profile(spec, args.microbatch, args.steps, args.out)
        return

    if step == "long":
        ledger = C.open_ledger(spec)
        with ledger.locked():
            C.open_ledger(spec)
            made = []
            for g in args.geometry:
                phase = ("long_control" if args.constant_lr else
                         "long" if set(args.seeds) <= set(spec.long["first_seeds"]) else "long_seeds")
                made += C.create_long(spec, ledger, g, args.seeds, phase, constant=args.constant_lr)
            ledger.save()
        print(f"created {made or 'nothing (already present)'}; submit with `approve` + `step --submit`")
        return

    if step == "retain":
        from .retention import retain

        print("\n".join(retain(spec, apply=args.apply,
                                 drop_unpromoted_resume=args.drop_unpromoted_resume)))
        return

    if step == "report":
        from .report import report as make_report

        print(make_report(spec, args.out))
        return

    if step == "panel":
        from .panel import panel

        print(panel(spec, args.trials, rth=args.rth, n_windows=args.n_windows, seed=args.seed,
                    out=args.out))
        return

    if step == "resources":
        from .resources import estimate, format_estimate

        print(format_estimate(estimate(spec)))
        return

    if step == "preview":
        from . import jobs as J
        from . import lsf
        from ..training.schedules import parse_epoch_spec

        if args.repo:
            spec.repo_override = args.repo
        ev = spec.evaluation
        epochs = parse_epoch_spec(ev["initial_epochs"])
        for g in args.geometry or spec.geometries:
            t = C.new_trial(spec, geometry=g, short=f"s{args.design}", origin="shared",
                            phase="baseline" if args.design == spec.raw.get("baseline_index", 0)
                            else "shared", params=spec.shared[args.design],
                            seed=spec.budget["initial_seed"],
                            target_epochs=spec.budget["initial_epochs"], snapshot_epochs=epochs,
                            eval_batches=[epochs], budgets={C.E30: ev["objective_epochs"]},
                            schedule=C.hpo_schedule(spec))
            tres = spec.resources("train", g)
            train_name = C._job_name(spec, t, "train", "c1")
            train_sh = J.train_script(spec, spec.repo, t, 1, float(tres["time_budget_hours"]))
            eval_name = C._job_name(spec, t, "eval", "e" + "-".join(f"{e:03d}" for e in epochs))
            eval_sh = J.eval_script(spec, spec.repo, t, epochs, C.eval_dir(spec, t, epochs))
            for name, text, kind in ((train_name, train_sh, "train"), (eval_name, eval_sh, "eval")):
                res = lsf.Resources.from_spec(spec.resources(kind, g))
                cmd_line = lsf.render_command(lsf.bsub_argv(name, res, f"{spec.root}/logs"),
                                              f"{spec.root}/jobs/{name}.sh")
                print(f"===== {g} {kind}: {name}\n# bsub (issued by `hpo step --submit`):\n"
                      f"{cmd_line}\n# script:\n{text}")
                if args.write:
                    J.write_script(str(Path(args.write) / f"{name}.sh"), text)
        return

    raise SystemExit(f"unknown step {step!r}")
