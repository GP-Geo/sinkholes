"""Training driver: dataset assembly, partitioning, the epoch loop, checkpoints.

Run as ``sinkholes train ...``. Outputs land under ``<output_dir>/<job>_<ts>/``:
``checkpoints/best.pt`` / ``last.pt`` (+ one .pth per epoch unless
``--save_best_only``), ``results.csv``, ``curves.png``, per-epoch validation
sample grids, the run log, and the pickled held-out test split that the
evaluation commands consume.

With ``--resume auto`` the directory is
``<output_dir>/<job>_<ts>_lsf_<LSB_JOBID>`` and each completed epoch also
writes a full training state to ``checkpoints/resume.pt``. A WEXAC job
preempted and requeued under the same LSF job id finds that directory again by
its job id — timestamp and all — and continues at the next epoch instead of
starting over. See :mod:`sinkholes.training.resume`.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..dataprep.dataset import (
    VAL_NEGATIVE_INNER,
    VAL_NEGATIVE_OUTER,
    VAL_NEGATIVE_PER_POS,
    RingNegatives,
    SubsiDataset,
    save_test_dataset,
    validation_negatives,
)
from ..dataprep.partition import (
    discover_intf_ids,
    filter_by_nonz_count,
    load_partition_window,
    load_preset_partition,
    split_nonoverlap_patches,
    split_preset_21,
    split_random_by_intf,
)
from ..dataprep.patchify import patch_file_name, resolve_patch_dirs
from ..device import get_device, memory_format_for
from ..meta import find_11day_sequences, load_coord_dict
from ..models.attention_unet import AttentionUNet
from ..models.convlstm_unet import DEFAULT_CONVLSTM_HIDDEN_CHANNELS, ConvLSTMUNet
from ..models.factory import PER_TIMESTEP_ARCHITECTURES, architecture_from_flags
from ..models.factory import get as architecture
from ..models.factory import strip_non_parameters
from ..models.tattn_unet import (
    DEFAULT_TATTN_DIM,
    DEFAULT_TATTN_HEADS,
    MAX_FUSED_SKIPS,
    RECURRENCE_CHOICES,
    TemporalAttentionUNet,
)
from ..models.unet import UNet
from .evaluate import evaluate
from .losses import segmentation_loss
from .reporter import (
    BestTracker,
    EpochTable,
    ResultsCSV,
    banner,
    human_time,
    plot_curves,
    quiet_root_console,
    save_prediction_grid,
    setup_logger,
)
from .resume import (
    RESUME_NAME,
    PreemptionGuard,
    build_resume_state,
    check_config_compatible,
    fresh_run_location,
    is_full_resume,
    learning_rate_of,
    load_legacy_weights,
    load_resume_checkpoint,
    resolve_run_location,
    restore_rng_state,
    resume_banner,
    run_config,
    save_atomic,
)

REPORTER_NAME = "sinkholes.train"


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", "-e", type=int, default=5)
    p.add_argument("--batch_size", "-b", type=int, default=1)
    p.add_argument("--learning-rate", "-l", dest="lr", type=float, default=1e-5)
    p.add_argument("--validation", "-v", dest="val", type=float, default=10.0,
                   help="validation share of the data, percent")
    p.add_argument("--test", type=float, default=10.0, help="test share, percent")
    p.add_argument("--amp", action="store_true", help="mixed precision")
    p.add_argument("--bilinear", action="store_true", help="bilinear upsampling")
    p.add_argument("--classes", "-c", type=int, default=1)
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--stride", type=int, default=2, help="strides per patch window")
    p.add_argument("--pos_w", type=float, default=1, help="BCE positive-class weight")
    p.add_argument("--seed", type=int, default=None,
                   help="seed python/numpy/torch, incl. the interferogram shuffle")

    p.add_argument("--patches_dir", type=str,
                   default="/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/",
                   help="directory containing the {data,mask}_patches_* trees")
    p.add_argument("--intf_dict_path", type=str, default=None,
                   help="interferogram coordinate dictionary (default: the committed asset)")
    p.add_argument("--train_on_11d_diff", action=argparse.BooleanOptionalAction, default=True,
                   help="use the 11-day patch tree (default) or the _all tree")
    p.add_argument("--use_cleaned_patches", action="store_true")
    p.add_argument("--job_name", type=str, default="run")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                   metavar="auto|DIR|FILE",
                   help="continue an interrupted run. 'auto' uses this LSF job's "
                        "directory, outputs/<job_name>_<timestamp>_lsf_$LSB_JOBID, which "
                        "a WEXAC job requeued after preemption finds again by its job id, "
                        "and picks up checkpoints/resume.pt if it is there. A run "
                        "directory or a "
                        "checkpoint file may be given instead. Without --resume the run "
                        "goes to a fresh timestamped directory, as before. --epochs is "
                        "always the total target, never 'more epochs'")

    p.add_argument("--nonz_only", action=argparse.BooleanOptionalAction, default=True,
                   help="train only on patches containing positive pixels")
    p.add_argument("--add_nulls_to_train", action="store_true",
                   help="also sample hard empty patches (no-data streaks)")
    p.add_argument("--train_with_nonz_th", action="store_true",
                   help="drop interferograms below --nonz_th positive patches")
    p.add_argument("--nonz_th", nargs=2, type=int, default=[350, 150], metavar=("NORTH", "SOUTH"))

    p.add_argument("--partition_mode", type=str, default="random_by_patch",
                   choices=["random_by_patch", "random_by_intf", "spatial", "preset_by_intf"])
    p.add_argument("--partition_file", type=str, default=None,
                   help="preset partition JSON; required by --partition_mode preset_by_intf. "
                        "There is deliberately no default: assets/ holds three generations of "
                        "partition (see assets/PARTITIONS.md) and picking one implicitly is how "
                        "a clean-data run silently trains on the 2019-2026 noisy lists.")
    p.add_argument("--preset_test_val_21", action="store_true",
                   help="fixed 2021 temporal hold-out for val+test")
    p.add_argument("--nonoverlap_tr_tst", action="store_true",
                   help="random_by_patch variant that keeps test patches spatially "
                        "disjoint from training patches")
    p.add_argument("--test_data_to_exclude", type=str, default=None,
                   help="pickled test set of a previous run; its interferograms become "
                        "this run's test set")
    p.add_argument("--thresh_lat", type=float, default=31.4,
                   help="spatial mode: latitude separating train (north) from val/test")
    p.add_argument("--train_intfs", type=str, default=None,
                   help="comma list: use exactly these interferograms for training "
                        "(with --val_intfs/--test_intfs; overrides the partition mode)")
    p.add_argument("--val_intfs", type=str, default=None)
    p.add_argument("--test_intfs", type=str, default=None)

    p.add_argument("--add_temporal", action="store_true",
                   help="stack the k previous 11-day interferograms as input")
    p.add_argument("--k_prevs", type=int, default=2)
    p.add_argument("--union_temporal_mask", action="store_true",
                   help="legacy temporal target: union over the stack instead of the "
                        "latest timestep")
    p.add_argument("--treat_nodata_regions", action="store_true",
                   help="append per-time validity channels and use the masked loss")
    p.add_argument("--add_ring_negatives", action="store_true")
    p.add_argument("--neg_ring_inner", type=int, default=1)
    p.add_argument("--neg_ring_outer", type=int, default=3)
    p.add_argument("--neg_per_pos", type=float, default=1.0)
    p.add_argument("--add_val_negatives", action="store_true",
                   help="also put negative patches in the VALIDATION set, so val/dice "
                        f"can see the false positives negatives exist to suppress. "
                        f"Fixed configuration — ring {VAL_NEGATIVE_INNER}.."
                        f"{VAL_NEGATIVE_OUTER}, {VAL_NEGATIVE_PER_POS:g}:1, drawn once "
                        f"from the validation interferograms with --seed and unchanged "
                        f"for every epoch. Deliberately NOT tunable: it is the ruler, "
                        f"not the experiment, so any two runs on one partition and seed "
                        f"are scored on identical samples. Independent of "
                        f"--add_ring_negatives — either may be used without the other.")

    p.add_argument("--attn_unet", action="store_true", help="AttentionUNet")
    p.add_argument("--add_attn", action="store_true", help="UNet + bottleneck attention")
    p.add_argument("--convlstm_unet", action="store_true",
                   help="ConvLSTM U-Net (requires --add_temporal)")
    p.add_argument("--convlstm_hidden", "--convlstm_hidden_channels",
                   dest="convlstm_hidden", type=int, default=0,
                   help=f"ConvLSTM hidden channels "
                        f"(0 = the default, {DEFAULT_CONVLSTM_HIDDEN_CHANNELS}). The value "
                        f"used is written into the checkpoint, so it is restored on load "
                        f"whatever the default becomes later.")
    p.add_argument("--convlstm_kernel", type=int, default=3)
    p.add_argument("--tattn_unet", action="store_true",
                   help="temporal-attention U-Net (requires --add_temporal)")
    p.add_argument("--tattn_dim", type=int, default=0,
                   help=f"width the temporal tokens are projected to "
                        f"(0 = the default, {DEFAULT_TATTN_DIM}). Written into the "
                        f"checkpoint, so it is restored on load whatever the default "
                        f"becomes later.")
    p.add_argument("--tattn_heads", type=int, default=DEFAULT_TATTN_HEADS,
                   help="attention heads, and the number of channel groups the fused "
                        "skips are split into — must divide 64, 128, 256 and 512")
    p.add_argument("--tattn_layers", type=int, default=1,
                   help="1 = a single present-queries-past readout. Above 1, the extra "
                        "layers are causal self-attention over the sequence first")
    p.add_argument("--tattn_recurrence", type=str, default="none",
                   choices=list(RECURRENCE_CHOICES),
                   help="'convlstm' runs a ConvLSTM first and attends over all of its "
                        "hidden states instead of only the last")
    p.add_argument("--tattn_fuse_skips", type=int, default=MAX_FUSED_SKIPS,
                   metavar=f"0..{MAX_FUSED_SKIPS}",
                   help=f"how many skip connections are fused over time by the "
                        f"attention weights, coarsest first. 0 reproduces the "
                        f"ConvLSTM's 'latest timestep only' skips; {MAX_FUSED_SKIPS} "
                        f"fuses every level, including the 200x100 one where the "
                        f"12x6 attention field is upsampled 16x")

    p.add_argument("--reporter", action=argparse.BooleanOptionalAction, default=True,
                   help="per-epoch table + results.csv + curves.png")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop after N epochs without val/dice improvement (0 = off)")
    p.add_argument("--lr_schedule", type=str, default="plateau", choices=["plateau", "cosine"],
                   help="plateau: cut the LR when val/dice stops improving. cosine: decay "
                        "from --learning-rate to --min_lr over --epochs, ignoring val "
                        "(deterministic, so runs stay comparable when val is small and noisy)")
    p.add_argument("--lr_factor", type=float, default=0.5,
                   help="plateau: multiply the LR by this on each plateau")
    p.add_argument("--lr_patience", type=int, default=5,
                   help="plateau: epochs without val/dice improvement before cutting the LR. "
                        "Kept short deliberately — on the k10split run the first cut is what "
                        "broke a five-epoch plateau; a longer patience delays it past the "
                        "point where it helps")
    p.add_argument("--min_lr", type=float, default=1e-8,
                   help="floor for both schedules; stops the LR decaying into dead epochs")
    p.add_argument("--save_best_only", action="store_true",
                   help="write only best.pt / last.pt, not one checkpoint per epoch")
    p.add_argument("--save_val", action="store_true",
                   help="save validation arrays under validation/")
    p.add_argument("--sample_every", type=int, default=1,
                   help="save a validation sample grid every N epochs (0 = off)")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--sample_min_sep", type=int, default=4,
                   help="minimum distance, in validation samples, between the patches "
                        "shown in the sample grid (windows overlap by half a patch, so "
                        "neighbouring indices show the same ground)")


def build_model(args, device):
    """The network the flags select; conflicting selections are an error."""
    num_c = (args.k_prevs + 1) if args.add_temporal else 1
    if args.treat_nodata_regions:
        num_c *= 2

    # One flags -> name mapping, shared with inference and the resume
    # fingerprint (models/factory.py), so a new architecture cannot be wired
    # into one of the three and silently forgotten in the others. It also
    # rejects two architecture flags at once on everyone's behalf.
    arch = architecture_from_flags(
        convlstm_unet=args.convlstm_unet,
        tattn_unet=args.tattn_unet,
        attn_unet=args.attn_unet,
        add_attn=args.add_attn,
    ) or "unet"

    if arch in PER_TIMESTEP_ARCHITECTURES and not args.add_temporal:
        raise SystemExit(
            f"{architecture(arch).cli_flag} needs a temporal sequence but "
            f"--add_temporal is not set. Pass --add_temporal --k_prevs N "
            f"(e.g. --k_prevs 2)."
        )

    per_timestep = 2 if args.treat_nodata_regions else 1
    if arch == "convlstm_unet":
        model = ConvLSTMUNet(
            n_channels_per_timestep=per_timestep,
            n_classes=args.classes,
            bilinear=args.bilinear,
            convlstm_hidden_channels=args.convlstm_hidden,
            convlstm_kernel_size=args.convlstm_kernel,
        )
    elif arch == "tattn_unet":
        model = TemporalAttentionUNet(
            n_channels_per_timestep=per_timestep,
            n_classes=args.classes,
            bilinear=args.bilinear,
            tattn_dim=args.tattn_dim,
            tattn_heads=args.tattn_heads,
            tattn_layers=args.tattn_layers,
            tattn_recurrence=args.tattn_recurrence,
            tattn_fuse_skips=args.tattn_fuse_skips,
            # Only read when --tattn_recurrence convlstm.
            convlstm_hidden_channels=args.convlstm_hidden,
            convlstm_kernel_size=args.convlstm_kernel,
        )
    elif arch == "attention_unet":
        model = AttentionUNet(n_channels=num_c, n_classes=args.classes, bilinear=args.bilinear)
    else:
        model = UNet(n_channels=num_c, n_classes=args.classes, bilinear=args.bilinear,
                     add_attn=args.add_attn)

    model = model.to(memory_format=memory_format_for(device))
    model.to(device=device)
    return model, num_c


def _log_val_composition(val_set, val_ring) -> None:
    """Record what val/dice is measured over — positives, negatives, ratio.

    Worth a line of its own: with negatives in the set, val/dice from this run
    is not on the same scale as one from a positives-only run, and the log is
    where that gets established after the fact.
    """
    n_neg = int(getattr(val_set, "n_negative", 0))
    n_pos = len(val_set) - n_neg
    if val_ring is None:
        logging.info(f"validation set: {n_pos} positive patches (no validation negatives)")
        return
    # The requested ratio is capped by how many empty patches the annulus holds,
    # so the achieved one is the number that has to be reported.
    achieved = n_neg / n_pos if n_pos else float("nan")
    logging.info(
        f"validation set: {n_pos} positive + {n_neg} negative patches "
        f"({achieved:.2f}:1 achieved against {VAL_NEGATIVE_PER_POS:g}:1 requested; "
        f"the request is capped by annulus availability)"
    )


def build_datasets(args, rep):
    """Interferogram discovery, filtering, partitioning -> (train, val, test) sets."""
    H, W = args.patch_size
    days = 11 if args.train_on_11d_diff else None
    image_dir, mask_dir = resolve_patch_dirs(args.patches_dir, (H, W), args.stride,
                                             days_diff=days, cleaned=args.use_cleaned_patches)
    logging.info(f"patch directories: {image_dir} | {mask_dir}")

    spatial = args.partition_mode == "spatial"
    coord_dict = load_coord_dict(args.intf_dict_path)

    discovery_nonz = args.nonz_only and not spatial and not args.add_temporal
    intf_list = discover_intf_ids(image_dir, nonz=discovery_nonz)
    intf_list = filter_by_nonz_count(
        intf_list, coord_dict, tuple(args.nonz_th) if args.train_with_nonz_th else None
    )
    logging.info(f"{len(intf_list)} interferograms after the nonz filter")

    seq_dict = None
    # The chain is also built for a SINGLE-FRAME run that asks for ring
    # negatives. A negative must be empty at every timestep, so the exclusion
    # grid is the union over the chain -- and building it the same way here is
    # what lets a single-frame control draw exactly the negatives its temporal
    # twin drew. Restricting to interferograms with full chains costs coverage,
    # but an unmatched control is not a control.
    if args.add_temporal or args.add_ring_negatives or args.add_val_negatives:
        seq_dict, intf_list = find_11day_sequences(coord_dict, k_prev=args.k_prevs,
                                                   restrict_to=intf_list)
        logging.info(f"{len(intf_list)} interferograms have full {args.k_prevs}-previous chains")
        if not args.add_temporal:
            logging.info(
                f"single-frame run with negatives: negatives are excluded "
                f"against the union over each {args.k_prevs}-previous chain, "
                f"matching a temporal run on this partition"
            )
    if not intf_list:
        raise SystemExit("no usable interferograms left after filtering")

    dataset_kwargs = dict(
        patch_size=(H, W),
        stride=args.stride,
        nonz_only=args.nonz_only,
        temporal=args.add_temporal,
        seq_dict=seq_dict,
        treat_nodata_regions=args.treat_nodata_regions,
        union_temporal_mask=args.union_temporal_mask,
        add_nulls_to_train=args.add_nulls_to_train,
        use_cleaned_patches=args.use_cleaned_patches,
    )
    # --add_nulls_to_train only reaches the non-temporal branch of the dataset
    # builder: temporal data goes through _load_temporal, which never calls
    # _add_null_patches. Accepting the flag silently would leave the run looking
    # like it sampled background when it did not.
    if args.add_nulls_to_train and args.add_temporal:
        logging.warning(
            "--add_nulls_to_train has no effect with --add_temporal: null-patch "
            "sampling lives in the single-frame path only. Use "
            "--add_ring_negatives for background patches in a temporal run."
        )

    # --add_ring_negatives reaches the temporal path and the single-frame path,
    # but NOT a spatial (--partition_mode spatial) single-frame run, whose
    # loader filters the grid without ever consulting the coordinate list.
    # Accepting the flag silently there would leave a run whose name, log and
    # config all claim negatives it never sampled -- exactly the failure the
    # --add_nulls_to_train warning above guards against.
    if args.add_ring_negatives and spatial and not args.add_temporal:
        raise SystemExit(
            "--add_ring_negatives is not supported for a single-frame run under "
            "--partition_mode spatial: that loader slices the grid by latitude "
            "and never builds the coordinate list an annulus needs. Use a preset "
            "partition (--partition_mode preset_by_intf), or add --add_temporal."
        )

    ring = RingNegatives(args.neg_ring_inner, args.neg_ring_outer,
                         args.neg_per_pos, args.seed) if args.add_ring_negatives else None

    # --add_val_negatives changes what val/dice MEANS, so the ways it could be
    # accepted while quietly not applying are all rejected here instead.
    val_ring = None
    if args.add_val_negatives:
        if args.seed is None:
            raise SystemExit(
                "--add_val_negatives needs --seed. The validation negatives are drawn "
                "once, and the whole point of them is that two runs on one partition "
                "draw the SAME ones — without a seed they would differ per run and the "
                "val/dice numbers could not be compared."
            )
        # The remaining partition modes either build no val dataset of their own
        # (random_by_patch splits one train-mode dataset after the fact) or hand
        # the val split to a loader with no coordinate list to draw an annulus
        # from (single-frame spatial). Both would leave a run whose name and
        # config claim validation negatives it never sampled.
        if args.partition_mode == "random_by_patch" or args.nonoverlap_tr_tst:
            raise SystemExit(
                "--add_val_negatives is not supported under --partition_mode "
                "random_by_patch: that mode splits one dataset into train/val/test "
                "after it is built, so there is no validation set to sample negatives "
                "for. Use --partition_mode preset_by_intf or random_by_intf."
            )
        if spatial and not args.add_temporal:
            raise SystemExit(
                "--add_val_negatives is not supported for a single-frame run under "
                "--partition_mode spatial: that loader slices the grid by latitude and "
                "never builds the coordinate list an annulus needs. Use a preset "
                "partition (--partition_mode preset_by_intf), or add --add_temporal."
            )
        val_ring = validation_negatives(args.seed)
        logging.info(
            f"validation negatives: ring {VAL_NEGATIVE_INNER}..{VAL_NEGATIVE_OUTER}, "
            f"{VAL_NEGATIVE_PER_POS:g} per positive, seed {args.seed} — fixed "
            f"configuration, drawn once from the validation interferograms"
        )

    explicit = [args.train_intfs, args.val_intfs, args.test_intfs]
    if any(explicit):
        if not all(explicit):
            raise SystemExit("--train_intfs/--val_intfs/--test_intfs must be given together")
        train_list, val_list, test_list = (s.split(",") for s in explicit)
        mode = "explicit lists"
    elif args.partition_mode == "random_by_intf":
        exclude = None
        if args.test_data_to_exclude:
            from ..dataprep.dataset import load_test_dataset

            exclude = load_test_dataset(args.test_data_to_exclude).ids
            logging.info(f"pinning the test split to {len(exclude)} interferograms "
                         f"from {args.test_data_to_exclude}")
        if args.preset_test_val_21:
            train_list, val_list, test_list = split_preset_21(intf_list)
        else:
            train_list, val_list, test_list = split_random_by_intf(
                intf_list, args.val / 100, args.test / 100, exclude_test=exclude
            )
        mode = "random_by_intf"
    elif args.partition_mode == "preset_by_intf":
        if not args.partition_file:
            raise SystemExit(
                "--partition_mode preset_by_intf needs an explicit --partition_file. "
                "assets/ holds three generations of partition; see assets/PARTITIONS.md "
                "and pass the one this run is meant to use."
            )
        train_list, val_list = load_preset_partition(args.partition_file)
        test_list = []
        mode = "preset_by_intf"
    else:
        train_list = val_list = test_list = None  # patch-level modes use all intfs
        mode = args.partition_mode

    if train_list is not None:
        overlap = (set(train_list) & set(val_list)) | (set(train_list) & set(test_list)) \
            | (set(val_list) & set(test_list))
        if overlap:
            raise SystemExit(f"interferograms appear in more than one split: {sorted(overlap)}")

        # `crossview` (plan section 5b) deliberately shares interferograms with
        # train -- it is the geo memorisation probe, scored on the band below
        # the 31.4 deg cut. Training never reads it: load_preset_partition takes
        # only train/val, so it cannot reach a dataset from here. Said out loud
        # because it is the one split whose overlap with train is intended, and
        # a future change that starts loading it would silently void the geo
        # result.
        if args.partition_file:
            try:
                with open(args.partition_file) as _fh:
                    _cv = json.load(_fh).get("crossview")
            except (OSError, ValueError):
                _cv = None
            if _cv:
                logging.info(f"partition carries a crossview list of {len(_cv)} interferograms; "
                             f"it is evaluation-only and is NOT loaded for training")
        logging.info(f"train intfs ({len(train_list)}): {sorted(train_list)}")
        logging.info(f"val intfs ({len(val_list)}): {sorted(val_list)}")
        logging.info(f"test intfs ({len(test_list)}): {sorted(test_list)}")

        # The AOI window travels per split inside the partition file, so train
        # and val can be restricted to different ground -- which is exactly what
        # the geo axis does. Reading it here, from the same file that gave the
        # interferogram lists, is what keeps training on the same ground the
        # evaluation commands score.
        windows = {}
        if args.partition_file:
            for split in ("train", "val"):
                w = load_partition_window(args.partition_file, split)
                if w is not None:
                    windows[split] = w
                    logging.info(f"{split} AOI window: lat {w[0]}-{w[1]}, lon {w[2]}-{w[3]}")
            if not windows:
                logging.info(f"{args.partition_file} carries no aoi_window; "
                             f"training on the whole canvas")

        train_set = SubsiDataset(image_dir, mask_dir, train_list, mode="train",
                                 ring_negatives=ring,
                                 aoi_window=windows.get("train"),
                                 coord_dict=coord_dict, **dataset_kwargs)
        val_set = SubsiDataset(image_dir, mask_dir, val_list, mode="val",
                               val_negatives=val_ring,
                               aoi_window=windows.get("val"),
                               coord_dict=coord_dict, **dataset_kwargs)
        # The test split stays positives-only: it is scored by
        # scripts/eval/run_eval.sh at scene scale, where the background is
        # already all there and sampling a slice of it would only bias the
        # number.
        test_set = (SubsiDataset(image_dir, mask_dir, test_list, mode="test",
                                 aoi_window=(load_partition_window(args.partition_file, "test")
                                             if args.partition_file else None),
                                 coord_dict=coord_dict, **dataset_kwargs)
                    if test_list else None)
        _log_val_composition(val_set, val_ring)
        return train_set, val_set, test_set, mode

    if spatial:
        spatial_kwargs = dict(dataset_kwargs, spatial=True, thresh_lat=args.thresh_lat,
                              coord_dict=coord_dict)
        train_set = SubsiDataset(image_dir, mask_dir, intf_list, mode="train",
                                 ring_negatives=ring, **spatial_kwargs)
        southern = SubsiDataset(image_dir, mask_dir, intf_list, mode="val",
                                val_negatives=val_ring, **spatial_kwargs)
        _log_val_composition(southern, val_ring)
        n_val = len(southern) // 2
        val_set, test_set = random_split(southern, [n_val, len(southern) - n_val])
        return train_set, val_set, test_set, "spatial"

    # random_by_patch
    if args.nonoverlap_tr_tst:
        train_imgs, train_msks, test_imgs, test_msks = [], [], [], []
        for intf in intf_list:
            grid_i = np.load(os.path.join(image_dir, patch_file_name("data", intf, H, W, args.stride)))
            grid_m = np.load(os.path.join(mask_dir, patch_file_name("mask", intf, H, W, args.stride)))
            tr_i, tr_m, te_i, te_m = split_nonoverlap_patches(grid_i, grid_m, args.test / 100)
            train_imgs.append(tr_i)
            train_msks.append(tr_m)
            test_imgs.append(te_i)
            test_msks.append(te_m)
        train_set = SubsiDataset.from_arrays(np.concatenate(train_imgs),
                                             np.concatenate(train_msks), intf_list, mode="train")
        test_set = SubsiDataset.from_arrays(np.concatenate(test_imgs),
                                            np.concatenate(test_msks), intf_list, mode="test")
        return train_set, test_set, test_set, "random_by_patch (non-overlapping)"

    dataset = SubsiDataset(image_dir, mask_dir, intf_list, mode="train",
                           ring_negatives=ring, **dataset_kwargs)
    n_total = len(dataset)
    n_val = int(n_total * args.val / 100)
    n_test = int(n_total * args.test / 100)
    n_train = n_total - n_val - n_test
    train_set, temp = random_split(dataset, [n_train, n_total - n_train],
                                   generator=torch.Generator().manual_seed(0))
    val_set, test_set = random_split(temp, [n_val, n_test],
                                     generator=torch.Generator().manual_seed(0))
    return train_set, val_set, test_set, "random_by_patch"


def mask_values_of(dataset):
    """mask_values of a SubsiDataset, seen through any random_split wrappers."""
    while not hasattr(dataset, "mask_values") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return getattr(dataset, "mask_values", [0, 1])


def train_model(args, model, device, train_set, val_set, test_set, outpath,
                checkpoint=None, guard=None):
    """The epoch loop. ``checkpoint`` resumes it; ``guard`` defers preemption.

    With a full-state ``checkpoint`` the loop restarts at ``epoch + 1`` and every
    optimizer/scheduler/scaler/tracker/RNG stream picks up where it stopped;
    ``--epochs`` is the total target, so a resumed run stops at the same place
    an uninterrupted one would have.
    """
    dir_checkpoint = Path(outpath) / "checkpoints"
    dir_validation = Path(outpath) / "validation"

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1,
                            num_workers=1, pin_memory=True)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8,
                              momentum=0.999, foreach=True)
    # 'plateau' reacts to val/dice, so a noisy validation curve can trigger a cut
    # that has nothing to do with real progress; 'cosine' follows a fixed path and
    # keeps runs comparable. Both stop at --min_lr rather than decaying to nothing.
    if args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", factor=args.lr_factor, patience=args.lr_patience,
            min_lr=args.min_lr)
    grad_scaler = torch.amp.GradScaler(enabled=args.amp)
    criterion = (nn.CrossEntropyLoss() if model.n_classes > 1
                 else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_w], device=device)))

    def loss_of(logits, images, true_masks):
        return segmentation_loss(logits, images, true_masks, n_classes=model.n_classes,
                                 treat_nodata_regions=args.treat_nodata_regions,
                                 criterion=criterion)

    n_train, n_val = len(train_set), len(val_set)
    rep = setup_logger(REPORTER_NAME) if args.reporter else None
    table = results_csv = None
    tracker = BestTracker("val/dice", mode="max", patience=args.patience)
    guard = guard if guard is not None else PreemptionGuard(rep or logging.getLogger())

    # -- resume: put every stateful object back where the last completed epoch left it
    resuming = is_full_resume(checkpoint)
    completed_epoch = 0
    global_step = 0
    elapsed_before = 0.0
    if resuming:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # An empty scaler state means the checkpoint ran without AMP; feeding
        # that to an enabled scaler raises, so only restore a real one.
        if checkpoint.get("scaler"):
            grad_scaler.load_state_dict(checkpoint["scaler"])
        elif args.amp:
            (rep or logging.getLogger()).warning(
                "The checkpoint carries no AMP scaler state (it was written without "
                "--amp): the scaler starts from its default scale.")
        tracker.load_state_dict(checkpoint.get("tracker", {}))
        completed_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step", 0))
        elapsed_before = float(checkpoint.get("elapsed", 0.0))
        # After the datasets are built, so the construction-time draws match the
        # original run before the saved streams take over.
        restore_rng_state(checkpoint.get("rng"), rep or logging.getLogger())
    elif checkpoint is not None:
        load_legacy_weights(checkpoint, model, rep or logging.getLogger())

    start_epoch = completed_epoch + 1

    if rep is not None:
        if device.type == "cpu":
            rep.warning("No accelerator found — training on CPU (expect ~10x slower epochs).")
        if n_val == 0:
            rep.warning("Validation split is empty — no Dice, no best-checkpoint tracking.")
        arch = ("ConvLSTMUNet" if args.convlstm_unet
                else "TemporalAttentionUNet" if args.tattn_unet
                else "AttentionUNet" if args.attn_unet
                else "UNet+attn" if args.add_attn else "UNet")
        if args.convlstm_unet:
            channels = (
                f"{model.n_channels_per_timestep} ch/timestep x T={args.k_prevs + 1} "
                f"/ {model.n_classes} out (ConvLSTM hidden="
                f"{model.convlstm_hidden_channels}, k={model.convlstm_kernel_size})"
            )
        elif args.tattn_unet:
            recurrence = (f", ConvLSTM hidden={model.convlstm_hidden_channels}"
                          if model.tattn_recurrence == "convlstm" else "")
            channels = (
                f"{model.n_channels_per_timestep} ch/timestep x T={args.k_prevs + 1} "
                f"/ {model.n_classes} out (attn dim={model.tattn_dim}, "
                f"heads={model.tattn_heads}, layers={model.tattn_layers}, "
                f"fused skips={model.tattn_fuse_skips}/{MAX_FUSED_SKIPS}{recurrence})"
            )
        else:
            channels = (
                f"{model.n_channels} in / {model.n_classes} out "
                f"(k_prevs={args.k_prevs}, temporal={args.add_temporal})"
            )
        n_val_neg = int(getattr(val_set, "n_negative", 0))
        banner(
            rep, f"Sinkholes {arch} — {args.job_name}",
            device=device.type,
            epochs=(f"{args.epochs} total (resuming at {start_epoch})"
                    if resuming else args.epochs),
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patch_size=f"{args.patch_size[0]}x{args.patch_size[1]} (stride {args.stride})",
            channels=channels,
            target=(("union over stack (legacy)" if args.union_temporal_mask
                     else "latest timestep") if args.add_temporal else "single intf"),
            partition=args.partition_mode,
            train_val_test=f"{n_train} / {n_val} / "
                           f"{len(test_set) if test_set is not None else 'n/a'} patches",
            # Stated in the header because it changes what val/dice means: with
            # negatives in the set, this run's curve is not on the same scale as
            # a positives-only run's.
            val_scored_on=(f"{n_val - n_val_neg} positive + {n_val_neg} negative "
                           f"(fixed {VAL_NEGATIVE_PER_POS:g}:1 set, ring "
                           f"{VAL_NEGATIVE_INNER}..{VAL_NEGATIVE_OUTER}, seed {args.seed})"
                           if args.add_val_negatives else "positives only"),
            pos_weight=args.pos_w,
            seed=args.seed,
            output_dir=outpath,
        )
        columns = ["epoch", "time", "train/loss", "val/loss", "val/dice",
                   "val/IoU", "val/F1", "val/P", "val/R", "lr"]
        table = EpochTable(columns, rep, header_every=25)
        # Resuming keeps epochs 1..completed and appends from there; a fresh run
        # starts the file over exactly as before.
        results_csv = ResultsCSV(Path(outpath) / "results.csv", columns,
                                 keep_through=completed_epoch if resuming else None)
        rep.info(f"Logging results to {outpath}/results.csv")
        if resuming:
            if results_csv.backup is not None:
                rep.warning(f"results.csv had rows past epoch {completed_epoch} (an epoch "
                            f"that never checkpointed and will be repeated) — kept "
                            f"{results_csv.kept} rows, original copied to "
                            f"{results_csv.backup.name}")
            rep.info(f"Kept {results_csv.kept} completed epoch rows in results.csv; "
                     f"appending from epoch {start_epoch}")
            rep.info(f"Continuing training: epochs {start_epoch}..{args.epochs}")
        else:
            rep.info(f"Starting training for {args.epochs} epochs...")

    run_start = time.time()
    best_path = dir_checkpoint / "best.pt"
    last_path = dir_checkpoint / "last.pt"
    resume_path = dir_checkpoint / RESUME_NAME
    config = run_config(args)
    interrupted = False
    partial_epoch = None
    early_stopped = False
    last_epoch = completed_epoch

    def checkpoint_state():
        state = model.state_dict()
        state["mask_values"] = mask_values_of(train_set)
        # Any architecture that describes itself gets its blob written, keyed by
        # the class rather than by an isinstance chain that has to be updated in
        # step with every new model.
        config_key = getattr(model, "CONFIG_KEY", None)
        if config_key is not None:
            state[config_key] = model.config_dict()
        return state

    def save_resume(epoch: int) -> None:
        """Atomically record a *completed* epoch as the recovery point."""
        save_atomic(
            build_resume_state(
                epoch=epoch, global_step=global_step, model=model, optimizer=optimizer,
                scheduler=scheduler, grad_scaler=grad_scaler, tracker=tracker,
                config=config, elapsed=elapsed_before + (time.time() - run_start),
                early_stopped=early_stopped,
                extra={"mask_values": mask_values_of(train_set),
                       "job_name": args.job_name, "run_dir": str(outpath)},
            ),
            resume_path,
        )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            partial_epoch = epoch
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            with tqdm(total=n_train, desc=f"{epoch}/{args.epochs}", unit="img", leave=False) as pbar:
                for batch in train_loader:
                    images, true_masks = batch["image"], batch["mask"]
                    if hasattr(model, "n_channels_per_timestep"):
                        # dim 1 is time (x channels per timestep), so only
                        # divisibility can be checked — T is not architectural.
                        assert images.dim() == 4 and images.shape[1] % model.n_channels_per_timestep == 0, (
                            f"batch shape {tuple(images.shape)} does not divide into "
                            f"{model.n_channels_per_timestep}-channel timesteps; check "
                            f"--k_prevs and --treat_nodata_regions"
                        )
                    else:
                        assert images.shape[1] == model.n_channels, (
                            f"network has {model.n_channels} input channels but the "
                            f"loader produced {images.shape[1]}"
                        )

                    images = images.to(device=device, dtype=torch.float32,
                                       memory_format=memory_format_for(device))
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.autocast(device.type if device.type != "mps" else "cpu",
                                        enabled=args.amp):
                        logits = model(images)
                    loss = loss_of(logits, images, true_masks)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    loss_val = loss.item()
                    if not math.isfinite(loss_val):
                        if rep is not None and n_batches == 0:
                            rep.warning(f"Non-finite loss ({loss_val}) at step {global_step} — "
                                        f"check the learning rate.")
                    else:
                        epoch_loss += loss_val
                        n_batches += 1
                    logging.info(f"step {global_step} epoch {epoch} train loss {loss_val:.8f}")
                    pbar.set_postfix(**{"loss (batch)": loss_val})

            dir_validation.mkdir(parents=True, exist_ok=True)
            dir_checkpoint.mkdir(parents=True, exist_ok=True)

            logging.info(f"Validation Round: {epoch}")
            val_metrics = {} if args.reporter else None
            val_loss_fn = loss_of if args.reporter else None
            want_samples = (args.reporter and args.sample_every
                            and (epoch % args.sample_every == 0 or epoch == 1))
            # Channel k_prevs is the current interferogram in a temporal stack
            # (oldest -> newest); later channels are validity maps.
            cur_channel = args.k_prevs if args.add_temporal else 0
            val_samples = ({"n": args.n_samples, "channel": cur_channel,
                            "min_sep": args.sample_min_sep} if want_samples else None)

            val_score = evaluate(
                model, val_loader, device, args.amp,
                mode="val", epoch=epoch, out_path=str(dir_validation),
                save_val=args.save_val, metrics_out=val_metrics,
                loss_fn=val_loss_fn, samples_out=val_samples,
            )
            # ReduceLROnPlateau steps on the metric; CosineAnnealingLR on the epoch.
            if args.lr_schedule == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_score)

            if val_samples and "image" in val_samples:
                try:
                    png = save_prediction_grid(
                        val_samples, dir_validation / "preds" / f"epoch_{epoch:03d}.png",
                        epoch=epoch)
                    if png is not None and epoch == 1 and rep is not None:
                        rep.info(f"  validation samples → {png.parent}")
                except Exception as exc:  # a figure must never take down a run
                    (rep or logging).warning(f"Could not save the sample grid: {exc}")

            logging.info(f"Epoch {epoch} Validation Dice score: {float(val_score)}")

            row = {
                "epoch": f"{epoch}/{args.epochs}",
                "time": human_time(elapsed_before + time.time() - run_start),
                "train/loss": epoch_loss / max(n_batches, 1),
                "val/loss": round(val_metrics["loss"], 6) if val_metrics and "loss" in val_metrics else float("nan"),
                "val/dice": float(val_score),
                "val/IoU": round(val_metrics["iou"], 4) if val_metrics else float("nan"),
                "val/F1": round(val_metrics["f1"], 4) if val_metrics else float("nan"),
                "val/P": round(val_metrics["precision"], 4) if val_metrics else float("nan"),
                "val/R": round(val_metrics["recall"], 4) if val_metrics else float("nan"),
                "lr": optimizer.param_groups[0]["lr"],
            }

            # One deferred-signal window for the whole per-epoch record: weights,
            # the results row, the best checkpoint and finally resume.pt. A
            # preemption landing in the middle would otherwise leave resume.pt
            # describing epoch N-1 while best.pt already held epoch N's weights.
            with guard.critical(f"epoch {epoch} checkpoints"):
                state = checkpoint_state()
                if not args.save_best_only:
                    save_atomic(state, dir_checkpoint / f"{args.job_name}checkpoint_epoch{epoch}.pth")
                    logging.info(f"Checkpoint {epoch} saved!")
                save_atomic(state, last_path)

                if rep is not None:
                    table.row(row)
                    results_csv.append(row)

                improved = tracker.update(epoch, row)
                if improved:
                    save_atomic(state, best_path)
                early_stopped = tracker.should_stop
                # Written last: its epoch number is what declares everything above
                # complete and consistent for this epoch.
                save_resume(epoch)
                last_epoch = completed_epoch = epoch
                partial_epoch = None

            if improved:
                msg = (f"  new best val/dice={tracker.best:.4f} @ epoch {epoch}"
                       f" — saved {best_path.name}")
                rep.info(msg) if rep is not None else logging.info(msg)

            if early_stopped:
                msg = (f"Early stopping: no improvement in {tracker.patience} epochs "
                       f"(best val/dice={tracker.best:.4f} @ epoch {tracker.best_epoch})")
                rep.warning(msg) if rep is not None else logging.warning(msg)
                break

    except KeyboardInterrupt:
        interrupted = True
        log = rep or logging.getLogger()
        if partial_epoch is not None:
            log.warning(
                f"Interrupted DURING epoch {partial_epoch} — that epoch is incomplete and "
                f"is NOT recorded. Recovery point is the end of epoch {completed_epoch}; "
                f"epoch {partial_epoch} will be repeated in full when the job is requeued."
            )
        else:
            log.warning(f"Interrupted after completing epoch {completed_epoch} — "
                        f"the checkpoint for it is already written.")
        # Weights as they stand mid-epoch, for inspection only. Deliberately a
        # separate file: resume.pt must keep describing the last complete epoch.
        try:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            save_atomic(checkpoint_state(), dir_checkpoint / "interrupted.pt")
        except Exception as exc:  # never turn a preemption into a crash
            log.warning(f"Could not save interrupted.pt: {exc}")
        if completed_epoch:
            log.warning(f"Resume with:  --resume {resume_path}   (or --resume auto "
                        f"under the same LSF job)")

    # Leave `model` holding the best weights, not the last epoch's.
    if best_path.exists() and tracker.best_epoch:
        best_state = torch.load(best_path, map_location=device)
        # Every non-parameter key the factory knows about, not a hand-listed
        # pair: leaving one behind fails load_state_dict at the very end of a
        # multi-hour run, which is the worst possible moment to find out.
        strip_non_parameters(best_state)
        model.load_state_dict(best_state)
        msg = (f"Restored best weights from epoch {tracker.best_epoch} "
               f"(val/dice={tracker.best:.4f}) into the model")
        rep.info(msg) if rep is not None else logging.info(msg)

    if rep is not None:
        this_run = time.time() - run_start
        elapsed = elapsed_before + this_run
        epochs_here = max(completed_epoch - (start_epoch - 1), 0)
        rep.info("=" * 72)
        status = ("interrupted" if interrupted
                  else "complete" if completed_epoch >= args.epochs or early_stopped
                  else "stopped")
        rep.info(f"Training {status} — {completed_epoch} epochs done in "
                 f"{human_time(elapsed)}"
                 f" ({human_time(this_run / max(epochs_here, 1))}/epoch"
                 f"{f', {epochs_here} this execution' if resuming else ''})")
        if tracker.best_epoch:
            rep.info(f"Best val/dice {tracker.best:.4f} @ epoch {tracker.best_epoch}")
        # LSF reports host memory but never gmem, so the -gpu gmem= request in
        # scripts/submit_all.sh has no feedback loop and drifts upward "to be
        # safe" — which is how seven jobs ended up PEND behind a 48G request on
        # 2026-08-10. Printing the real peak is what turns that ladder from a
        # guess into a measurement:  grep "peak VRAM" logs/*.out
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            reserved = torch.cuda.max_memory_reserved(device) / 2**30
            rep.info(f"peak VRAM {peak:.1f} GiB allocated, {reserved:.1f} GiB reserved "
                     f"— size the gmem request off the RESERVED figure")
        rep.info(f"Weights   {dir_checkpoint}")
        if best_path.exists():
            rep.info(f"Best      {best_path}  (epoch {tracker.best_epoch})")
        if last_path.exists():
            rep.info(f"Last      {last_path}  (epoch {last_epoch})")
        if resume_path.exists():
            rep.info(f"Resume    {resume_path}  (epoch {completed_epoch})")
        rep.info(f"Results   {outpath}/results.csv")
        try:
            curves = plot_curves(Path(outpath) / "results.csv", Path(outpath) / "curves.png",
                                 best_epoch=tracker.best_epoch or None, title=args.job_name)
            if curves is not None:
                rep.info(f"Curves    {curves}")
        except Exception as exc:
            rep.warning(f"Could not write curves.png: {exc}")
        preds_dir = dir_validation / "preds"
        if preds_dir.exists():
            rep.info(f"Samples   {preds_dir}  ({len(list(preds_dir.glob('*.png')))} epochs)")
        rep.info("=" * 72)


def main(args) -> None:
    logging.basicConfig(level=logging.INFO)
    # Without --resume this is the old timestamped directory; with it, a fixed
    # one a requeued LSF execution lands in again.
    location = resolve_run_location(args)

    # Read the checkpoint before claiming a directory: a model-only file named
    # explicitly is not a continuation of the run it came from, and must not
    # write over that run's results.csv or best.pt.
    checkpoint = None
    legacy_source = None
    if location.resume_path is not None:
        checkpoint = load_resume_checkpoint(location.resume_path)
        if not is_full_resume(checkpoint) and location.from_file:
            legacy_source = location.resume_path
            location = fresh_run_location(args)
            location.notes.append(
                f"{legacy_source} holds model weights only, so this is a new run, not a "
                f"continuation: writing to {location.outpath} and leaving the original "
                f"run directory untouched."
            )

    args.job_name = location.job_name
    outpath = location.outpath
    os.makedirs(outpath, exist_ok=True)

    log_file = os.path.join(outpath, f"{args.job_name}.log")
    logging.getLogger().addHandler(logging.FileHandler(log_file))
    logging.info("train job started")

    if args.reporter:
        # Console goes to the reporter; the run's .log file keeps everything.
        quiet_root_console(logging.WARNING)
        setup_logger(REPORTER_NAME, log_file=Path(outpath) / "logs" / "reporter.log")
    rep = logging.getLogger(REPORTER_NAME) if args.reporter else None
    log = rep or logging.getLogger()
    for note in location.notes:
        log.info(note)

    # -- vet the checkpoint before anything expensive happens
    if checkpoint is not None:
        if is_full_resume(checkpoint):
            advisory = check_config_compatible(checkpoint.get("config", {}),
                                               run_config(args), location.resume_path)
            for line in resume_banner(location.resume_path, checkpoint, args.epochs):
                log.info(line)
            for diff in advisory:
                log.warning(f"setting changed since the checkpoint — {diff}")
            if any(d.startswith("lr:") for d in advisory):
                log.warning("--learning-rate differs from the checkpoint, but the restored "
                            f"optimizer and scheduler state wins: training continues at "
                            f"lr={learning_rate_of(checkpoint)}.")
            done = int(checkpoint.get("epoch", 0))
            if checkpoint.get("early_stopped"):
                log.info(f"This run already stopped early after epoch {done} "
                         f"(best val/dice={checkpoint.get('best_score', float('nan')):.4f} "
                         f"@ epoch {checkpoint.get('best_epoch')}). Nothing to do.")
                return
            if done >= args.epochs:
                log.info(f"All {args.epochs} requested epochs are already complete "
                         f"({done} done) in {outpath}. Nothing to do — raise --epochs to "
                         f"train further.")
                return
        else:
            log.warning(f"{legacy_source or location.resume_path} is a model-only "
                        f"checkpoint, not a full training state.")

    if args.seed is not None:
        # Seeded before the datasets are built, so a resumed execution reproduces
        # the same partition and ring-negative draws; the saved RNG streams are
        # restored afterwards, inside train_model.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    elif checkpoint is not None:
        log.warning("Resuming without --seed: dataset partitioning that draws on the "
                    "global RNG may differ from the original execution. Preset partitions "
                    "(--partition_mode preset_by_intf) are unaffected.")

    device = get_device()
    logging.info(f"Using device {device}")

    guard = PreemptionGuard(log)
    with guard:
        try:
            train_set, val_set, test_set, partition_desc = build_datasets(args, rep)
            logging.info(f"partition: {partition_desc}; train/val/test = "
                         f"{len(train_set)}/{len(val_set)}/"
                         f"{len(test_set) if test_set is not None else 0} samples")

            if test_set is not None:
                pkl = os.path.join(outpath, f"test_dataset_{args.job_name}.pkl")
                # Re-pickling gigabytes on every requeue buys nothing: the split is
                # rebuilt identically, and a rewrite is one more thing preemption
                # could truncate.
                if checkpoint is not None and os.path.exists(pkl):
                    logging.info(f"test split already pickled at {pkl} — kept")
                else:
                    size_gb = save_test_dataset(test_set, pkl)
                    logging.info(f"test split pickled to {pkl} ({size_gb:.2f} GB)")

            model, num_c = build_model(args, device)
            ch_desc = (f"{model.n_channels} input channels per timestep "
                       f"(sequence length T={args.k_prevs + 1}, not fixed by the architecture)"
                       if args.convlstm_unet else f"{model.n_channels} input channels")
            logging.info(f"Network:\n\t{ch_desc}\n\t{model.n_classes} output channels\n"
                         f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling")
        except KeyboardInterrupt:
            # Nothing has been trained yet, so there is nothing to save. The run
            # directory is deterministic under --resume, so the requeued execution
            # comes back here and tries again.
            log.warning(f"Interrupted while preparing the run ({guard.signal_name or 'SIGINT'}) "
                        f"— no epoch had started, so there is nothing to checkpoint. The requeued job "
                        f"reuses {outpath} and starts again.")
            return

        train_model(args, model, device, train_set, val_set, test_set, outpath,
                    checkpoint=checkpoint, guard=guard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
