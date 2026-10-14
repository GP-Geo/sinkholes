"""What a run was trained with: source, data, partition, numerics, hardware.

Every execution of a run (the first one and each resumed chunk) appends one
record to ``<run>/run_manifest.json``. The record is deliberately cheap to
produce -- hashing a few hundred GB of patch grids is not -- so the data trees
are identified by path, file count and a digest over (name, size, mtime) of the
files this run reads, while the small files that decide the experiment (the
partition, the coordinate dictionary, the source tree) are hashed in full.

Nothing here touches an RNG or the model; it only reads files and metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

MANIFEST_NAME = "run_manifest.json"

#: The package root; the source fingerprint covers every .py beneath it.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def sha256_file(path, chunk: int = 1 << 20) -> Optional[str]:
    """Full SHA-256 of a (small) file, or None when it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def source_fingerprint(root: Path = PACKAGE_ROOT) -> Dict[str, Any]:
    """SHA-256 over every package .py (path + content), plus git state if available.

    The content digest is what identifies the code: it works in a checkout whose
    git metadata is unreachable (a worktree read from another host) and it
    changes with an uncommitted edit, which a commit hash alone would hide.
    """
    h = hashlib.sha256()
    n = 0
    for path in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
        n += 1
    info: Dict[str, Any] = {"package_sha256": h.hexdigest(), "n_files": n,
                            "package_root": str(root)}
    info.update(_git_state(Path(root).parent))
    return info


def _git_state(repo: Path) -> Dict[str, Any]:
    def git(*args) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                 text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"git_commit": None}
    status = git("status", "--porcelain", "--untracked-files=no") or ""
    return {"git_commit": commit, "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(status.strip())}


def tree_digest(files: Iterable) -> Dict[str, Any]:
    """Identify a set of large files by (name, size, mtime) without reading them."""
    h = hashlib.sha256()
    n = total = 0
    missing = []
    for f in sorted({str(p) for p in files}):
        try:
            st = os.stat(f)
        except OSError:
            missing.append(f)
            continue
        h.update(f"{os.path.basename(f)}\0{st.st_size}\0{int(st.st_mtime)}\0".encode())
        n += 1
        total += st.st_size
    return {"n_files": n, "bytes": total, "stat_sha256": h.hexdigest(), "missing": missing}


def environment() -> Dict[str, Any]:
    """Interpreter, library and accelerator versions."""
    import numpy as np
    import torch

    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "lsf_job_name": os.environ.get("LSB_JOBNAME"),
        "lsf_queue": os.environ.get("LSB_QUEUE"),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
    return env


def append_execution_record(run_dir, record: Dict[str, Any]) -> Path:
    """Append one execution's record to ``run_manifest.json`` (atomic rewrite)."""
    path = Path(run_dir) / MANIFEST_NAME
    data: Dict[str, Any] = {"executions": []}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            # A damaged manifest is kept aside, never silently dropped.
            path.rename(path.with_suffix(f".corrupt-{datetime.now():%Y%m%d%H%M%S}.json"))
            data = {"executions": []}
    data.setdefault("executions", []).append(record)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path


def numerical_policies(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """The numerical rules a reader needs to interpret this run's numbers."""
    return {
        "amp": bool(args.amp),
        "amp_dtype": "float16" if args.amp else None,
        "gradient_clipping": config.get("gradient_clipping"),
        "clip_max_norm": 1.0,
        "region_loss_dtype": config.get("region_loss_dtype"),
        "eval_rng": config.get("eval_rng") or "shared-with-training",
        "accum_loss_scaling": "loss/accum, partial final group stepped",
        "preprocessing_version": config.get("preprocessing_version"),
        "aoi_selection_version": config.get("aoi_selection_version"),
    }


def optimizer_record(args) -> Dict[str, Any]:
    name = getattr(args, "optimizer", "rmsprop")
    rec: Dict[str, Any] = {"name": name, "lr": float(args.lr),
                           "weight_decay": float(args.weight_decay)}
    if name == "rmsprop":
        rec.update(momentum=float(args.momentum), alpha=0.99, eps=1e-8,
                   weight_decay_mode="L2 in gradient")
    else:
        rec.update(betas=[float(args.beta1), float(args.beta2)], eps=1e-8,
                   weight_decay_mode="decoupled" if name == "adamw" else "L2 in gradient")
    return rec


def schedule_record(args, scheduler) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"lr_schedule": args.lr_schedule, "min_lr": float(args.min_lr)}
    if args.lr_schedule == "plateau":
        rec.update(monitor="val/dice", mode="max", factor=float(args.lr_factor),
                   patience=int(args.lr_patience))
    elif args.lr_schedule == "cosine":
        rec.update(horizon=int(getattr(args, "lr_horizon", 0) or args.epochs),
                   form=("closed-form, clamped at floor" if getattr(args, "lr_horizon", 0)
                         else "torch CosineAnnealingLR(T_max=epochs)"))
    try:
        rec["state"] = {k: v for k, v in scheduler.state_dict().items()
                        if isinstance(v, (int, float, str, bool, list, type(None)))}
    except Exception:  # a record must never take a run down
        pass
    return rec


def run_record(args, *, config, scheduler, files_read=(), extra=None) -> Dict[str, Any]:
    """The full record for one execution."""
    partition = getattr(args, "partition_file", None)
    dict_path = getattr(args, "intf_dict_path", None)
    if not dict_path:
        from ..paths import asset

        dict_path = asset("intf_coord.json")
    rec: Dict[str, Any] = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "job_name": args.job_name,
        "source": source_fingerprint(),
        "partition": {"path": partition, "sha256": sha256_file(partition) if partition else None},
        "intf_dict": {"path": str(dict_path), "sha256": sha256_file(dict_path)},
        "data": {"patches_dir": args.patches_dir, "files": tree_digest(files_read)},
        "optimizer": optimizer_record(args),
        "schedule": schedule_record(args, scheduler),
        "numerics": numerical_policies(args, config),
        "seed": args.seed,
        "config": dict(config),
        "environment": environment(),
    }
    if extra:
        rec.update(extra)
    return rec
