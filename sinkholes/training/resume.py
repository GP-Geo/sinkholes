"""Preemption-safe run directories and full training-state checkpoints.

WEXAC (IBM LSF) preempts a job by signalling it and later *reruns the same LSF
job ID*. Nothing about the rerun tells the program it is a rerun, so recovery
has to be derived from two things the scheduler does guarantee: the job keeps
its ``LSB_JOBID``, and the working directory survives. Hence:

  * ``--resume auto`` puts the run in
    ``outputs/<job_name>_<timestamp>_lsf_<LSB_JOBID>`` and a requeued execution
    *finds* that directory by the job id rather than recomputing the name — so
    the timestamp stays in the name for navigation without making the location
    unreproducible;
  * every completed epoch writes ``checkpoints/resume.pt`` — model, optimizer,
    scheduler, AMP scaler, early-stopping tracker and every RNG stream — with
    a temp file plus :func:`os.replace`, so a kill mid-write can never leave a
    truncated checkpoint behind;
  * a requeued execution finds that file and continues at ``epoch + 1``.

The recovery point is deliberately the last *fully completed* epoch. Mid-epoch
state (a half-consumed shuffled loader, a partially accumulated loss) has no
faithful representation here, so an interrupted epoch is simply repeated.

``best.pt`` / ``last.pt`` keep their existing model-only format so every
inference and evaluation command loads them unchanged; ``resume.pt`` is a new,
separate file.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

#: Marks a file written by :func:`save_resume_checkpoint`. Anything without it
#: is treated as a legacy model-only checkpoint.
RESUME_FORMAT = "sinkholes-resume"
RESUME_VERSION = 1

#: Name of the full-state checkpoint inside a run's ``checkpoints/`` directory.
RESUME_NAME = "resume.pt"

#: Settings that change what is being trained or on what data. Resuming across
#: a change to any of these would silently produce a run that is neither the
#: old experiment nor a clean new one, so it is refused.
STRICT_CONFIG_KEYS: Sequence[str] = (
    "preprocessing_version",
    "aoi_selection_version",
    "gradient_clipping",
    "momentum",
    "weight_decay",
    "augment_flips",
    "architecture",
    "n_classes",
    "add_temporal",
    "k_prevs",
    "treat_nodata_regions",
    "union_temporal_mask",
    "convlstm_hidden",
    "convlstm_kernel",
    "tattn_dim",
    "tattn_heads",
    "tattn_layers",
    "tattn_recurrence",
    "tattn_fuse_skips",
    "tattn_contrast",
    "tattn_qk_norm",
    "bilinear",
    "patch_size",
    "context_margin",
    "stride",
    "partition",
    "dataset",
    "batch_size",
    "accum_steps",
    "seed",
)

#: Settings a resume may legitimately change (extending --epochs is the whole
#: point). Differences are reported, not refused.
ADVISORY_CONFIG_KEYS: Sequence[str] = (
    "epochs",
    "lr",
    "lr_schedule",
    "lr_factor",
    "lr_patience",
    "min_lr",
    "patience",
    "pos_w",
    "amp",
)

logger = logging.getLogger(__name__)


# -- run configuration fingerprint -------------------------------------------------------

def architecture_name(args) -> str:
    """The registered architecture name the CLI flags select.

    Delegates to the factory so this fingerprint, ``build_model`` and the
    inference commands can never disagree about what a set of flags means.
    """
    from ..models.factory import architecture_from_flags

    return architecture_from_flags(
        convlstm_unet=getattr(args, "convlstm_unet", False),
        tattn_unet=getattr(args, "tattn_unet", False),
        attn_unet=getattr(args, "attn_unet", False),
        add_attn=getattr(args, "add_attn", False),
    ) or "unet"


def _partition_signature(args) -> str:
    """Everything that decides which interferograms land in which split."""
    parts = [
        f"mode={args.partition_mode}",
        f"file={args.partition_file or '-'}",
        f"val={args.val}",
        f"test={args.test}",
        f"preset21={bool(args.preset_test_val_21)}",
        f"nonoverlap={bool(args.nonoverlap_tr_tst)}",
        f"thresh_lat={args.thresh_lat}",
        f"exclude={args.test_data_to_exclude or '-'}",
        f"explicit={args.train_intfs or '-'}|{args.val_intfs or '-'}|{args.test_intfs or '-'}",
    ]
    return " ".join(parts)


def _dataset_signature(args) -> str:
    """Everything that decides which patches exist and what they contain."""
    parts = [
        f"patches_dir={args.patches_dir}",
        f"11d={bool(args.train_on_11d_diff)}",
        f"cleaned={bool(args.use_cleaned_patches)}",
        f"nonz_only={bool(args.nonz_only)}",
        f"nulls={bool(args.add_nulls_to_train)}",
        f"nonz_th={bool(args.train_with_nonz_th)}:{tuple(args.nonz_th)}",
        f"ring={bool(args.add_ring_negatives)}:"
        f"{args.neg_ring_inner}-{args.neg_ring_outer}x{args.neg_per_pos}",
    ]
    # Appended only when set, so a run without validation negatives keeps the
    # exact signature it had before this option existed and every checkpoint
    # written earlier still resumes. `dataset` is a STRICT key: adding a term
    # unconditionally would refuse them all.
    if getattr(args, "add_val_negatives", False):
        from ..dataprep.dataset import (
            VAL_NEGATIVE_INNER,
            VAL_NEGATIVE_OUTER,
            VAL_NEGATIVE_PER_POS,
        )

        parts.append(f"valneg={VAL_NEGATIVE_INNER}-{VAL_NEGATIVE_OUTER}"
                     f"x{VAL_NEGATIVE_PER_POS}")
    return " ".join(parts)


def run_config(args) -> Dict[str, Any]:
    """The settings a resume is validated against, as plain JSON-ish values."""
    from ..models.convlstm_unet import DEFAULT_CONVLSTM_HIDDEN_CHANNELS
    from ..models.temporal_attention import DEFAULT_TATTN_DIM

    convlstm = bool(getattr(args, "convlstm_unet", False))
    tattn = bool(getattr(args, "tattn_unet", False))
    # The hybrid uses a ConvLSTM too, so its hidden width is part of the
    # fingerprint whenever either flag selects a recurrent stage.
    recurrent = convlstm or (tattn and getattr(args, "tattn_recurrence", "none") == "convlstm")
    hidden = (args.convlstm_hidden or DEFAULT_CONVLSTM_HIDDEN_CHANNELS) if recurrent else None
    temporal = bool(args.add_temporal)
    H, W = args.patch_size
    return {
        "preprocessing_version": getattr(args, "preprocessing_version", "frame-v2"),
        "aoi_selection_version": getattr(args, "aoi_selection_version", "coordinates-v2"),
        "gradient_clipping": "unscaled-v2",
        "momentum": float(getattr(args, "momentum", 0.999)),
        "weight_decay": float(getattr(args, "weight_decay", 1e-8)),
        "augment_flips": str(getattr(args, "augment_flips", "none")),
        "architecture": architecture_name(args),
        "n_classes": int(args.classes),
        "add_temporal": temporal,
        # k_prevs only shapes the input when a temporal stack is actually built.
        "k_prevs": int(args.k_prevs) if temporal else None,
        "treat_nodata_regions": bool(args.treat_nodata_regions),
        "union_temporal_mask": bool(args.union_temporal_mask),
        "convlstm_hidden": hidden,
        "convlstm_kernel": int(args.convlstm_kernel) if recurrent else None,
        # None on every non-attention run, so an existing ConvLSTM or U-Net
        # resume keeps the fingerprint it had before these keys existed.
        "tattn_dim": (args.tattn_dim or DEFAULT_TATTN_DIM) if tattn else None,
        "tattn_heads": int(args.tattn_heads) if tattn else None,
        "tattn_layers": int(args.tattn_layers) if tattn else None,
        "tattn_recurrence": args.tattn_recurrence if tattn else None,
        "tattn_fuse_skips": int(args.tattn_fuse_skips) if tattn else None,
        # Both change what the attention block IS, so resuming across a flip
        # would splice two different architectures into one run.
        "tattn_contrast": bool(args.tattn_contrast) if tattn else None,
        "tattn_qk_norm": bool(args.tattn_qk_norm) if tattn else None,
        "bilinear": bool(args.bilinear),
        "patch_size": f"{H}x{W}",
        # Emitted only when it is set, so every checkpoint written before large
        # context keeps its existing fingerprint and stays resumable.
        "context_margin": (f"{args.context_margin[0]}x{args.context_margin[1]}"
                           if tuple(getattr(args, "context_margin", (0, 0)) or (0, 0)) != (0, 0)
                           else None),
        # Emitted only when accumulating, so every checkpoint written before
        # this keeps its fingerprint. batch_size alone no longer describes the
        # optimiser: batch_size * accum_steps is the effective batch.
        "accum_steps": (int(args.accum_steps)
                        if int(getattr(args, "accum_steps", 1) or 1) != 1 else None),
        "stride": int(args.stride),
        "partition": _partition_signature(args),
        "dataset": _dataset_signature(args),
        "batch_size": int(args.batch_size),
        "seed": args.seed,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "lr_schedule": args.lr_schedule,
        "lr_factor": float(args.lr_factor),
        "lr_patience": int(args.lr_patience),
        "min_lr": float(args.min_lr),
        "patience": int(args.patience),
        "pos_w": float(args.pos_w),
        "amp": bool(args.amp),
    }


class IncompatibleResume(SystemExit):
    """Raised (as a clean CLI exit) when a checkpoint does not match the flags."""


def check_config_compatible(saved: Mapping[str, Any], current: Mapping[str, Any],
                            checkpoint_path) -> list[str]:
    """Refuse an incompatible resume; return advisory differences as messages.

    Unknown keys absent from old checkpoints are skipped. The patch policies
    and AMP clipping have known historical defaults and are guarded explicitly.
    """
    # These omissions have KNOWN historical semantics, unlike unknown optional
    # keys below. Never continue an old run with a different population or
    # preprocessing policy just because it predates the version fields.
    saved = dict(saved)
    if "dataset" in saved:
        saved.setdefault("preprocessing_version", "legacy-row-v1")
        saved.setdefault("aoi_selection_version", "legacy-v1")
    if saved.get("amp") and saved.get("gradient_clipping") != "unscaled-v2":
        raise IncompatibleResume(
            f"cannot resume {checkpoint_path}: legacy AMP clipped scaled gradients. "
            "Continue that run only with its original source tree; the fixed branch "
            "must start a new experiment (model-only weights may be used explicitly)."
        )
    missing = object()
    breaking = [
        (key, saved.get(key, missing), current.get(key))
        for key in STRICT_CONFIG_KEYS
        if saved.get(key, missing) is not missing and saved.get(key) != current.get(key)
    ]
    if breaking:
        lines = [
            f"cannot resume {checkpoint_path}: the checkpoint was written with "
            f"different settings.",
            "",
            f"  {'setting':<22} {'checkpoint':<28} {'requested now':<28}",
            f"  {'-' * 22} {'-' * 28} {'-' * 28}",
        ]
        lines += [f"  {k:<22} {str(was):<28} {str(now):<28}" for k, was, now in breaking]
        lines += [
            "",
            "Resuming across these would mix two different experiments. Either pass the "
            "original settings, or start a fresh run (drop --resume, or use a different "
            "--job_name / LSF job).",
        ]
        raise IncompatibleResume("\n".join(lines))

    return [
        f"{key}: checkpoint {saved.get(key)!r} -> now {current.get(key)!r}"
        for key in ADVISORY_CONFIG_KEYS
        if saved.get(key, missing) is not missing and saved.get(key) != current.get(key)
    ]


# -- run directory resolution -------------------------------------------------------------

@dataclass
class RunLocation:
    """Where this execution writes, and what (if anything) it resumes from."""

    outpath: str
    job_name: str
    #: full-state or legacy checkpoint to load, or None for a fresh start
    resume_path: Optional[Path] = None
    #: "fresh" | "auto" | "explicit"
    mode: str = "fresh"
    #: True when --resume named a checkpoint file rather than a directory
    from_file: bool = False
    #: lines to print prominently once the reporter exists
    notes: list = field(default_factory=list)


def lsf_job_id(env: Optional[Mapping[str, str]] = None) -> str:
    """The LSF job id this execution belongs to, or a clean error.

    A requeued execution keeps ``LSB_JOBID``, which is exactly what makes the
    run directory reproducible. Array elements share the job id and differ by
    ``LSB_JOBINDEX``, so a non-zero index is appended — without it two array
    elements would fight over one directory.
    """
    env = os.environ if env is None else env
    job_id = str(env.get("LSB_JOBID", "")).strip()
    if not job_id:
        raise SystemExit(
            "--resume auto needs the LSF job id, but LSB_JOBID is not set in the "
            "environment.\nWithout it the run directory cannot be tied to this job, and "
            "two jobs with the same --job_name would write into the same place.\n"
            "Run this under bsub, or name the directory explicitly with "
            "--resume <run-directory>."
        )
    index = str(env.get("LSB_JOBINDEX", "")).strip()
    if index and index not in ("0", ""):
        job_id = f"{job_id}_{index}"
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in job_id)


def auto_run_dir(output_dir: str, job_name: str,
                 env: Optional[Mapping[str, str]] = None,
                 log=None) -> tuple[str, bool]:
    """The run directory for this LSF job: ``(path, existed_already)``.

    ``<output_dir>/<job_name>_<timestamp>_lsf_<LSB_JOBID>``. The timestamp is
    the *first* execution's, so ``outputs/`` still sorts and reads by date —
    which is the whole reason it is in the name. A requeued execution therefore
    cannot recompute the name, and does not try to: it *finds* the directory by
    the job id, which is unique to this job and unchanged by a requeue. Only
    when nothing matches is a new name minted, with the current timestamp.

    Directories written by the earlier, timestamp-free naming
    (``<job_name>_lsf_<id>``) match the same search, so runs started under it
    keep resuming into their own directory.
    """
    job_id = lsf_job_id(env)
    root = Path(output_dir)
    suffix = f"_lsf_{job_id}"
    matches = sorted(
        (p for p in (root.iterdir() if root.is_dir() else ())
         if p.is_dir() and p.name.endswith(suffix) and p.name.startswith(f"{job_name}_")),
        key=lambda p: p.stat().st_mtime,
    )
    if matches:
        # One job id can only belong to one run; if a stray directory ever
        # matches too, the one holding a checkpoint is the real one.
        with_state = [p for p in matches if _checkpoint_in(p).exists()]
        chosen = (with_state or matches)[-1]
        if len(matches) > 1 and log is not None:
            log.warning(f"{len(matches)} directories match LSF job {job_id} "
                        f"({', '.join(p.name for p in matches)}) — using {chosen.name}.")
        return str(chosen), True

    stamped = f"{job_name}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}{suffix}"
    return str(root / stamped), False


def _checkpoint_in(run_dir: Path) -> Path:
    return Path(run_dir) / "checkpoints" / RESUME_NAME


def fresh_run_location(args) -> RunLocation:
    """The historical default: a new ``<job_name>_<timestamp>`` directory."""
    stamped = f"{args.job_name}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    return RunLocation(outpath=os.path.join(args.output_dir, stamped),
                       job_name=stamped, mode="fresh")


def resolve_run_location(args, env: Optional[Mapping[str, str]] = None) -> RunLocation:
    """Decide the output directory and the checkpoint to resume from.

    Three shapes, all driven by ``--resume``:

    * absent  — unchanged legacy behaviour: ``<job_name>_<timestamp>``, fresh;
    * ``auto`` — this LSF job's directory (see :func:`auto_run_dir`), resuming
      if it holds a checkpoint and starting fresh in it if it does not;
    * a path  — a run directory (same rule as ``auto``, without needing LSF) or
      a checkpoint file, which must exist.

    A *new* timestamp is only ever minted for a directory that does not exist
    yet; reusing a name that a requeued execution cannot find again is precisely
    the bug that makes preempted jobs restart from scratch.
    """
    resume = getattr(args, "resume", None)
    job_name = args.job_name

    if resume is None:
        return fresh_run_location(args)

    if str(resume).strip().lower() == "auto":
        run_dir, existed = auto_run_dir(args.output_dir, job_name, env)
        ckpt = _checkpoint_in(Path(run_dir))
        location = RunLocation(outpath=run_dir, job_name=job_name, mode="auto",
                               resume_path=ckpt if ckpt.exists() else None)
        if ckpt.exists():
            location.notes.append(f"--resume auto: LSF job {lsf_job_id(env)} already has "
                                  f"{run_dir} — resuming from {ckpt}")
        elif existed:
            location.notes.append(
                f"--resume auto: {run_dir} belongs to LSF job {lsf_job_id(env)} but holds "
                f"no checkpoint yet (the previous execution did not finish an epoch) — "
                f"starting again from epoch 1 in it."
            )
        else:
            location.notes.append(
                f"--resume auto: first execution of LSF job {lsf_job_id(env)} — creating "
                f"{run_dir}. A requeued execution finds this directory by its job id and "
                f"continues from the last completed epoch."
            )
        return location

    path = Path(str(resume)).expanduser()
    if path.is_dir():
        ckpt = _checkpoint_in(path)
        location = RunLocation(outpath=str(path), job_name=job_name, mode="explicit",
                               resume_path=ckpt if ckpt.exists() else None)
        location.notes.append(
            f"--resume {path}: found {ckpt}" if ckpt.exists()
            else f"--resume {path}: no {RESUME_NAME} there yet — starting a fresh run in "
                 f"that directory."
        )
        return location

    if not path.exists():
        raise SystemExit(
            f"--resume {path}: no such file or directory.\nPass a run directory, a "
            f"checkpoint file, or 'auto' to use outputs/<job_name>_lsf_<LSB_JOBID>."
        )

    # A checkpoint file: the run directory is the one that contains it, looking
    # through the conventional checkpoints/ subdirectory.
    run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    return RunLocation(outpath=str(run_dir), job_name=job_name, mode="explicit",
                       resume_path=path, from_file=True, notes=[f"--resume {path}"])


# -- atomic checkpoint I/O ------------------------------------------------------------------

def save_atomic(obj: Any, path) -> Path:
    """``torch.save`` to a temp file in the same directory, then ``os.replace``.

    ``os.replace`` is atomic within a filesystem, so a reader (this program,
    requeued) sees either the previous checkpoint or the complete new one — a
    kill part-way through the write cannot produce a truncated file. The
    fsync matters for the same reason across a node crash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return path


def rng_state() -> Dict[str, Any]:
    """Every RNG stream training draws from, so a resume continues the sequence."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Mapping[str, Any]], log=None) -> None:
    """Reinstate saved RNG streams; a stream that cannot be restored only warns."""
    log = log or logger
    if not state:
        return
    with contextlib.suppress(Exception):
        random.setstate(state["python"])
    with contextlib.suppress(Exception):
        np.random.set_state(state["numpy"])
    with contextlib.suppress(Exception):
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available():
        try:
            if len(cuda) != torch.cuda.device_count():
                raise RuntimeError(
                    f"checkpoint holds {len(cuda)} CUDA RNG states but this node has "
                    f"{torch.cuda.device_count()} device(s)"
                )
            torch.cuda.set_rng_state_all([_as_byte_tensor(t) for t in cuda])
        except Exception as exc:
            log.warning(f"Could not restore the CUDA RNG state ({exc}) — CUDA-side "
                        f"randomness (dropout, cuDNN) restarts from the seed instead.")


def _as_byte_tensor(value):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.cpu().to(torch.uint8)


def build_resume_state(
    *,
    epoch: int,
    global_step: int,
    model,
    optimizer,
    scheduler,
    grad_scaler,
    tracker,
    config: Mapping[str, Any],
    elapsed: float = 0.0,
    early_stopped: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything needed to continue training as if the interruption never happened."""
    state: Dict[str, Any] = {
        "format": RESUME_FORMAT,
        "version": RESUME_VERSION,
        # The number of epochs *fully finished*: validated, logged, checkpointed.
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": grad_scaler.state_dict(),
        "tracker": tracker.state_dict(),
        "best_score": float(tracker.best),
        "best_epoch": int(tracker.best_epoch),
        "rng": rng_state(),
        "config": dict(config),
        "elapsed": float(elapsed),
        "early_stopped": bool(early_stopped),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
    }
    if extra:
        state.update(extra)
    return state


def is_full_resume(state: Any) -> bool:
    return isinstance(state, Mapping) and state.get("format") == RESUME_FORMAT


def load_resume_checkpoint(path, map_location="cpu") -> Dict[str, Any]:
    """Load a checkpoint file, with a readable error instead of a traceback.

    ``weights_only=False`` is required and safe here: the file holds RNG states
    and an optimizer state — not plain tensors — and it is one this run wrote
    into its own output directory.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise SystemExit(
            f"could not read the resume checkpoint {path}: {exc}\n"
            f"If it is corrupt, delete it — the run then restarts from epoch 1 in the "
            f"same directory (best.pt / last.pt are left untouched)."
        ) from exc


def resume_banner(path, state: Mapping[str, Any], target_epochs: int,
                  learning_rate: Optional[float] = None) -> list[str]:
    """The prominent 'this is a resumed run' block, as lines."""
    epoch = int(state.get("epoch", 0))
    if learning_rate is None:
        learning_rate = learning_rate_of(state)
    best = state.get("best_score", float("-inf"))
    best_text = (f"{best:.4f} @ epoch {state.get('best_epoch', 0)}"
                 if best not in (float("-inf"), float("inf")) else "none yet")
    return [
        "=" * 72,
        f"RESUMING an interrupted run — {path}",
        "=" * 72,
        f"  Completed epoch    {epoch}",
        f"  Next epoch         {epoch + 1} of {target_epochs}",
        f"  Global step        {int(state.get('global_step', 0))}",
        f"  Best val/dice      {best_text}",
        f"  Learning rate      {learning_rate if learning_rate is not None else 'unknown'}",
        f"  Checkpoint written {state.get('saved_at', 'unknown')}",
        "=" * 72,
    ]


def learning_rate_of(state: Mapping[str, Any]) -> Optional[float]:
    """The LR the optimizer will resume with, read without building anything."""
    groups = (state.get("optimizer") or {}).get("param_groups") or []
    return float(groups[0]["lr"]) if groups and "lr" in groups[0] else None


def load_legacy_weights(state: Any, model, log=None) -> None:
    """Load a model-only checkpoint (``best.pt`` / ``last.pt``) into ``model``.

    Explicitly *not* a resume: there is no optimizer, scheduler, epoch counter
    or RNG state in such a file, and pretending otherwise would restart momentum
    and the LR schedule from scratch while claiming continuity.
    """
    log = log or logger
    if not isinstance(state, dict):
        raise SystemExit("the checkpoint is not a state dict; cannot load weights from it")

    from ..models.factory import strip_non_parameters

    weights = dict(state)
    strip_non_parameters(weights)
    try:
        model.load_state_dict(weights)
    except Exception as exc:
        raise SystemExit(
            f"the checkpoint's weights do not fit the requested model: {exc}\n"
            f"Check --convlstm_unet / --attn_unet / --add_attn, --k_prevs, "
            f"--convlstm_hidden and --treat_nodata_regions."
        ) from exc
    log.warning(
        "Loaded MODEL WEIGHTS ONLY from a legacy checkpoint. This is not a resume: "
        "optimizer momentum, the LR schedule, the AMP scaler, the epoch counter, the "
        "early-stopping history and all RNG states could not be restored. Training "
        "starts at epoch 1 with a fresh optimizer."
    )


# -- preemption handling --------------------------------------------------------------------

class PreemptionGuard:
    """Turn SIGINT/SIGTERM into a ``KeyboardInterrupt`` at a safe moment.

    LSF signals the job before it takes the node away. Raising immediately is
    right almost everywhere — mid-batch work is discarded and the epoch is
    repeated — but not while a checkpoint is being written: an exception there
    would leave the run with no record of the epoch that just finished. So a
    signal arriving inside :meth:`critical` is remembered and re-raised the
    moment the write is complete.
    """

    def __init__(self, log=None):
        self.log = log or logger
        self.signum: Optional[int] = None
        self.count = 0
        self._depth = 0
        self._previous: Dict[int, Any] = {}

    # -- context manager: install/restore handlers
    def __enter__(self) -> "PreemptionGuard":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):  # not the main thread / unsupported
                pass
        return self

    def __exit__(self, *exc_info) -> bool:
        for sig, handler in self._previous.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(sig, handler)
        self._previous.clear()
        return False

    @property
    def triggered(self) -> bool:
        return self.signum is not None

    @property
    def signal_name(self) -> str:
        if self.signum is None:
            return ""
        try:
            return signal.Signals(self.signum).name
        except ValueError:  # pragma: no cover
            return str(self.signum)

    def _handle(self, signum, frame) -> None:
        first = self.signum is None
        self.signum = signum
        self.count += 1
        name = self.signal_name
        if self._depth:
            self.log.warning(f"{name} received while a checkpoint is being written — "
                             f"finishing the write first, then stopping.")
            return
        if first:
            self.log.warning(f"{name} received (WEXAC preemption?) — stopping now; the "
                             f"run continues from the last completed epoch when requeued.")
        raise KeyboardInterrupt(name)

    @contextlib.contextmanager
    def critical(self, what: str = "checkpoint"):
        """Defer preemption for the duration of a write, then honour it."""
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0 and self.signum is not None:
                self.log.warning(f"{what} written — honouring the deferred "
                                 f"{self.signal_name} now.")
                raise KeyboardInterrupt(self.signal_name)

    def check(self) -> None:
        """Raise if a signal arrived while it was being deferred."""
        if self.signum is not None and self._depth == 0:
            raise KeyboardInterrupt(self.signal_name)


def stamp() -> str:
    """Compact wall-clock stamp for backup file names."""
    return time.strftime("%Y%m%d-%H%M%S")
