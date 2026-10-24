"""The thin layer over IBM LSF: submit, query, and kill a duplicate submission.

The controller never relies on LSF to requeue, rerun or chain anything: every
continuation is a new job it submits itself, and the truth about a job's
outcome is read from the files the job writes (``run_status.json``, eval
results), with ``bjobs`` consulted only to know what is still pending or
running. ``bjobs`` forgets finished jobs after a while, so its silence is never
read as success.

``BSUB``/``BJOBS``/``BKILL`` environment variables override the executables,
which is how the tests drive this against fakes.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

JOB_ID_RE = re.compile(r"Job <(\d+)> is submitted")

#: LSF states that mean "still going"; anything else is over (or unknown).
ACTIVE_STATES = {"PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "WAIT", "PROV"}


@dataclass
class Resources:
    queue: str
    mem_gb: int
    wall: str
    gmem_gb: Optional[int] = None

    @classmethod
    def from_spec(cls, d: dict) -> "Resources":
        return cls(queue=d["queue"], mem_gb=int(d["mem_gb"]), wall=str(d["wall"]),
                   gmem_gb=int(d["gmem_gb"]) if d.get("gmem_gb") else None)


def bsub_argv(name: str, res: Resources, log_dir: str) -> List[str]:
    argv = [os.environ.get("BSUB", "bsub"), "-J", name, "-q", res.queue,
            "-R", f"rusage[mem={res.mem_gb}GB]", "-W", res.wall,
            "-o", f"{log_dir}/{name}_%J.out", "-e", f"{log_dir}/{name}_%J.err"]
    if res.gmem_gb:
        argv += ["-gpu", f"num=1:j_exclusive=yes:gmem={res.gmem_gb}G"]
    return argv


def render_command(argv: List[str], script: str) -> str:
    return " ".join(shlex.quote(a) for a in argv) + f" < {shlex.quote(script)}"


class SubmitError(RuntimeError):
    pass


class KillError(RuntimeError):
    pass


def submit(argv: List[str], script: str) -> str:
    """Run bsub with the script on stdin; return the LSF job id."""
    with open(script) as fh:
        out = subprocess.run(argv, stdin=fh, capture_output=True, text=True, timeout=120)
    text = (out.stdout or "") + (out.stderr or "")
    m = JOB_ID_RE.search(text)
    if out.returncode != 0 or not m:
        raise SubmitError(f"bsub failed (exit {out.returncode}): {text.strip()[:500]}")
    return m.group(1)


def kill(job_ids: Iterable[str]) -> None:
    """bkill these jobs."""
    argv = [os.environ.get("BKILL", "bkill"), *[str(j) for j in job_ids]]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise KillError(f"bkill could not run: {exc}")
    if out.returncode != 0:
        text = (out.stdout or "") + (out.stderr or "")
        raise KillError(f"bkill failed (exit {out.returncode}): {text.strip()[:500]}")


def query(job_ids: Iterable[str] = (), names: Iterable[str] = ()) -> Dict[str, Dict[str, str]]:
    """{job_id: {"stat": .., "name": ..}} for the jobs LSF still knows about.

    By id and/or by exact job name (``-J``). Unknown ids are simply absent.
    Returns {} when bjobs is unavailable (e.g. on the Mac) -- callers must then
    fall back to the files the jobs write.
    """
    exe = os.environ.get("BJOBS", "bjobs")
    found: Dict[str, Dict[str, str]] = {}
    ids = [str(j) for j in job_ids if j]
    calls = []
    if ids:
        calls.append([exe, "-a", "-noheader", "-o", "jobid stat job_name delimiter='|'", *ids])
    for n in names:
        calls.append([exe, "-a", "-noheader", "-o", "jobid stat job_name delimiter='|'", "-J", n])
    for argv in calls:
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return {}
        for line in (out.stdout or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 3 and parts[0].isdigit():
                found[parts[0]] = {"stat": parts[1], "name": parts[2]}
    return found


def available() -> bool:
    exe = os.environ.get("BSUB", "bsub")
    from shutil import which

    return which(exe) is not None or os.path.isfile(exe)
