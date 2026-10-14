"""Utility jobs before (and beside) the trials: min/max caches and the GPU profile.

Both are phase ``prep``: listed by default, submitted only with ``--submit``
once ``approve --phase prep`` is recorded, and never twice.

* ``minmax`` (CPU): one streaming pass over every stride-4 grid the validation
  and test evaluations open, caching per-cell min/max so no evaluation job has
  to read a whole grid just to decide the normalisation branch. Resumable.
* ``profile`` (GPU): peak VRAM and step time of the real model, both
  geometries, at each candidate micro-batch -- the number that decides whether
  32 x 4 holds for ctx50 (an earlier ctx50 k10 run measured 33.5 GiB reserved
  at 32 x 4, so it is expected to).
"""

from __future__ import annotations

from typing import List

from . import jobs as J
from . import lsf
from .controller import open_ledger
from .ledger import now
from .spec import Spec, localize, to_cluster


def prep(spec: Spec, spec_path: str, submit: bool, what=("minmax", "profile")) -> List[str]:
    ledger = open_ledger(spec)
    repo = ledger.data["repo"]
    logs = f"{spec.root}/logs"
    planned = []
    if "minmax" in what:
        for g in spec.geometries:
            planned.append((f"{spec.name}-prep-minmax-{g}", lsf.Resources.from_spec(spec.resources("minmax")),
                            J.util_script(spec, repo, spec_path, "cache-minmax",
                                          ["--geometry", g, "--splits", "val", "test"])))
    if "profile" in what:
        planned.append((f"{spec.name}-prep-profile", lsf.Resources.from_spec(spec.resources("profile")),
                        J.util_script(spec, repo, spec_path, "profile",
                                      ["--out", f"{spec.root}/reports/profile.json"])))
    lines = []
    if submit and not lsf.available():
        raise SystemExit("bsub is not on PATH -- run `prep --submit` on a WEXAC login node")
    with (ledger.locked() if submit else _null()):
        for name, res, text in planned:
            script = f"{spec.root}/jobs/{name}.sh"
            argv = lsf.bsub_argv(name, res, logs)
            cmd = lsf.render_command(argv, script)
            if not ledger.approved("prep"):
                lines.append(f"NEEDS APPROVAL (prep)  {name}\n    {cmd}")
                continue
            if not submit:
                lines.append(f"WOULD SUBMIT (prep)  {name}\n    {cmd}")
                continue
            if any(j.get("job_id") for j in ledger.jobs_named(name)):
                lines.append(f"SKIP  {name}: already submitted (ledger)")
                continue
            if any(v["stat"] in lsf.ACTIVE_STATES for v in lsf.query(names=[name]).values()):
                lines.append(f"SKIP  {name}: already in LSF")
                continue
            J.write_script(localize(script), text)
            job = {"kind": "prep", "name": name, "script": script, "argv": argv,
                   "submitted": now(), "state": "submitting"}
            ledger.data.setdefault("utility_jobs", []).append(job)
            ledger.save()
            job["job_id"] = lsf.submit(argv, localize(script))
            job["state"] = "submitted"
            ledger.save()
            ledger.event("submit", name=name, job_id=job["job_id"], job_kind="prep")
            lines.append(f"SUBMITTED  {name}: LSF job {job['job_id']}")
    return lines


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
