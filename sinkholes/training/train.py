"""Training driver: dataset assembly, partitioning, the epoch loop, checkpoints.

Run as ``sinkholes train ...``. Outputs land under ``<output_dir>/<job>_<ts>/``:
``checkpoints/best.pt`` / ``last.pt`` (+ one .pth per epoch unless
``--save_best_only``), ``results.csv``, ``curves.png``, per-epoch validation
sample grids, the run log, and the pickled held-out test split that the
evaluation commands consume.
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..dataprep.dataset import RingNegatives, SubsiDataset, save_test_dataset
from ..dataprep.partition import (
    discover_intf_ids,
    filter_by_nonz_count,
    load_preset_partition,
    split_nonoverlap_patches,
    split_preset_21,
    split_random_by_intf,
)
from ..dataprep.patchify import patch_dir_name, patch_file_name
from ..device import get_device, memory_format_for
from ..meta import find_11day_sequences, load_coord_dict
from ..models.attention_unet import AttentionUNet
from ..models.convlstm_unet import CONFIG_KEY as CONVLSTM_CONFIG_KEY
from ..models.convlstm_unet import ConvLSTMUNet
from ..models.unet import UNet
from ..paths import asset
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
                   help="preset partition JSON (default: the committed asset)")
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

    p.add_argument("--attn_unet", action="store_true", help="AttentionUNet")
    p.add_argument("--add_attn", action="store_true", help="UNet + bottleneck attention")
    p.add_argument("--convlstm_unet", action="store_true",
                   help="ConvLSTM U-Net (requires --add_temporal)")
    p.add_argument("--convlstm_hidden", type=int, default=0,
                   help="ConvLSTM hidden channels (0 = match the bottleneck, 1024)")
    p.add_argument("--convlstm_kernel", type=int, default=3)

    p.add_argument("--reporter", action=argparse.BooleanOptionalAction, default=True,
                   help="per-epoch table + results.csv + curves.png")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop after N epochs without val/dice improvement (0 = off)")
    p.add_argument("--save_best_only", action="store_true",
                   help="write only best.pt / last.pt, not one checkpoint per epoch")
    p.add_argument("--save_val", action="store_true",
                   help="save validation arrays under validation/")
    p.add_argument("--sample_every", type=int, default=1,
                   help="save a validation sample grid every N epochs (0 = off)")
    p.add_argument("--n_samples", type=int, default=4)


def build_model(args, device):
    """The network the flags select; conflicting selections are an error."""
    num_c = (args.k_prevs + 1) if args.add_temporal else 1
    if args.treat_nodata_regions:
        num_c *= 2

    if args.convlstm_unet:
        conflicts = [f for f, on in (("--attn_unet", args.attn_unet),
                                     ("--add_attn", args.add_attn)) if on]
        if conflicts:
            raise SystemExit(
                f"--convlstm_unet cannot be combined with {', '.join(conflicts)}: "
                f"they select different architectures. Pass exactly one."
            )
        if not args.add_temporal:
            raise SystemExit(
                "--convlstm_unet needs a temporal sequence but --add_temporal is not "
                "set. Pass --add_temporal --k_prevs N (e.g. --k_prevs 2)."
            )
        model = ConvLSTMUNet(
            n_channels_per_timestep=2 if args.treat_nodata_regions else 1,
            n_classes=args.classes,
            bilinear=args.bilinear,
            convlstm_hidden_channels=args.convlstm_hidden,
            convlstm_kernel_size=args.convlstm_kernel,
        )
    elif args.attn_unet:
        if args.add_attn:
            raise SystemExit("--attn_unet and --add_attn select different architectures")
        model = AttentionUNet(n_channels=num_c, n_classes=args.classes, bilinear=args.bilinear)
    else:
        model = UNet(n_channels=num_c, n_classes=args.classes, bilinear=args.bilinear,
                     add_attn=args.add_attn)

    model = model.to(memory_format=memory_format_for(device))
    model.to(device=device)
    return model, num_c


def build_datasets(args, rep):
    """Interferogram discovery, filtering, partitioning -> (train, val, test) sets."""
    H, W = args.patch_size
    days = 11 if args.train_on_11d_diff else None
    image_dir = os.path.join(args.patches_dir, patch_dir_name("data", H, W, args.stride, days))
    mask_dir = os.path.join(args.patches_dir, patch_dir_name("mask", H, W, args.stride, days))
    if args.use_cleaned_patches:
        image_dir = os.path.join(image_dir, "cleaned")
        mask_dir = os.path.join(mask_dir, "cleaned")
    logging.info(f"patch directories: {image_dir} | {mask_dir}")
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        raise SystemExit(f"patch directories not found: {image_dir} — prepare patches first")

    spatial = args.partition_mode == "spatial"
    coord_dict = load_coord_dict(args.intf_dict_path)

    discovery_nonz = args.nonz_only and not spatial and not args.add_temporal
    intf_list = discover_intf_ids(image_dir, nonz=discovery_nonz)
    intf_list = filter_by_nonz_count(
        intf_list, coord_dict, tuple(args.nonz_th) if args.train_with_nonz_th else None
    )
    logging.info(f"{len(intf_list)} interferograms after the nonz filter")

    seq_dict = None
    if args.add_temporal:
        seq_dict, intf_list = find_11day_sequences(coord_dict, k_prev=args.k_prevs,
                                                   restrict_to=intf_list)
        logging.info(f"{len(intf_list)} interferograms have full {args.k_prevs}-previous chains")
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
    ring = RingNegatives(args.neg_ring_inner, args.neg_ring_outer,
                         args.neg_per_pos, args.seed) if args.add_ring_negatives else None

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
        train_list, val_list = load_preset_partition(
            args.partition_file or asset("partition_20_05_13h45.json")
        )
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
        logging.info(f"train intfs ({len(train_list)}): {sorted(train_list)}")
        logging.info(f"val intfs ({len(val_list)}): {sorted(val_list)}")
        logging.info(f"test intfs ({len(test_list)}): {sorted(test_list)}")

        train_set = SubsiDataset(image_dir, mask_dir, train_list, mode="train",
                                 ring_negatives=ring, **dataset_kwargs)
        val_set = SubsiDataset(image_dir, mask_dir, val_list, mode="val", **dataset_kwargs)
        test_set = (SubsiDataset(image_dir, mask_dir, test_list, mode="test", **dataset_kwargs)
                    if test_list else None)
        return train_set, val_set, test_set, mode

    if spatial:
        spatial_kwargs = dict(dataset_kwargs, spatial=True, thresh_lat=args.thresh_lat,
                              coord_dict=coord_dict)
        train_set = SubsiDataset(image_dir, mask_dir, intf_list, mode="train",
                                 ring_negatives=ring, **spatial_kwargs)
        southern = SubsiDataset(image_dir, mask_dir, intf_list, mode="val", **spatial_kwargs)
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


def train_model(args, model, device, train_set, val_set, test_set, outpath):
    dir_checkpoint = Path(outpath) / "checkpoints"
    dir_validation = Path(outpath) / "validation"

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1,
                            num_workers=1, pin_memory=True)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8,
                              momentum=0.999, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
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

    if rep is not None:
        if device.type == "cpu":
            rep.warning("No accelerator found — training on CPU (expect ~10x slower epochs).")
        if n_val == 0:
            rep.warning("Validation split is empty — no Dice, no best-checkpoint tracking.")
        arch = ("ConvLSTMUNet" if args.convlstm_unet
                else "AttentionUNet" if args.attn_unet
                else "UNet+attn" if args.add_attn else "UNet")
        channels = (
            f"{model.n_channels_per_timestep} ch/timestep x T={args.k_prevs + 1} "
            f"/ {model.n_classes} out (ConvLSTM hidden={model.convlstm_hidden_channels}, "
            f"k={model.convlstm_kernel_size})"
            if args.convlstm_unet else
            f"{model.n_channels} in / {model.n_classes} out "
            f"(k_prevs={args.k_prevs}, temporal={args.add_temporal})"
        )
        banner(
            rep, f"Sinkholes {arch} — {args.job_name}",
            device=device.type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patch_size=f"{args.patch_size[0]}x{args.patch_size[1]} (stride {args.stride})",
            channels=channels,
            target=(("union over stack (legacy)" if args.union_temporal_mask
                     else "latest timestep") if args.add_temporal else "single intf"),
            partition=args.partition_mode,
            train_val_test=f"{n_train} / {n_val} / "
                           f"{len(test_set) if test_set is not None else 'n/a'} patches",
            pos_weight=args.pos_w,
            seed=args.seed,
            output_dir=outpath,
        )
        columns = ["epoch", "time", "train/loss", "val/loss", "val/dice",
                   "val/IoU", "val/F1", "val/P", "val/R", "lr"]
        table = EpochTable(columns, rep, header_every=25)
        results_csv = ResultsCSV(Path(outpath) / "results.csv", columns)
        rep.info(f"Logging results to {outpath}/results.csv")
        rep.info(f"Starting training for {args.epochs} epochs...")

    run_start = time.time()
    best_path = dir_checkpoint / "best.pt"
    last_path = dir_checkpoint / "last.pt"
    interrupted = False
    last_epoch = 0
    global_step = 0

    def checkpoint_state():
        state = model.state_dict()
        state["mask_values"] = mask_values_of(train_set)
        if isinstance(model, ConvLSTMUNet):
            state[CONVLSTM_CONFIG_KEY] = model.config_dict()
        return state

    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            with tqdm(total=n_train, desc=f"{epoch}/{args.epochs}", unit="img", leave=False) as pbar:
                for batch in train_loader:
                    images, true_masks = batch["image"], batch["mask"]
                    if isinstance(model, ConvLSTMUNet):
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
            val_samples = {"n": args.n_samples, "channel": cur_channel} if want_samples else None

            val_score = evaluate(
                model, val_loader, device, args.amp,
                mode="val", epoch=epoch, out_path=str(dir_validation),
                save_val=args.save_val, metrics_out=val_metrics,
                loss_fn=val_loss_fn, samples_out=val_samples,
            )
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

            state = checkpoint_state()
            if not args.save_best_only:
                torch.save(state, dir_checkpoint / f"{args.job_name}checkpoint_epoch{epoch}.pth")
                logging.info(f"Checkpoint {epoch} saved!")
            torch.save(state, last_path)

            row = {
                "epoch": f"{epoch}/{args.epochs}",
                "time": human_time(time.time() - run_start),
                "train/loss": epoch_loss / max(n_batches, 1),
                "val/loss": round(val_metrics["loss"], 6) if val_metrics and "loss" in val_metrics else float("nan"),
                "val/dice": float(val_score),
                "val/IoU": round(val_metrics["iou"], 4) if val_metrics else float("nan"),
                "val/F1": round(val_metrics["f1"], 4) if val_metrics else float("nan"),
                "val/P": round(val_metrics["precision"], 4) if val_metrics else float("nan"),
                "val/R": round(val_metrics["recall"], 4) if val_metrics else float("nan"),
                "lr": optimizer.param_groups[0]["lr"],
            }
            if rep is not None:
                table.row(row)
                results_csv.append(row)

            if tracker.update(epoch, row):
                torch.save(state, best_path)
                msg = (f"  new best val/dice={tracker.best:.4f} @ epoch {epoch}"
                       f" — saved {best_path.name}")
                rep.info(msg) if rep is not None else logging.info(msg)

            if tracker.should_stop:
                msg = (f"Early stopping: no improvement in {tracker.patience} epochs "
                       f"(best val/dice={tracker.best:.4f} @ epoch {tracker.best_epoch})")
                rep.warning(msg) if rep is not None else logging.warning(msg)
                break

    except KeyboardInterrupt:
        interrupted = True
        if rep is not None:
            rep.warning(f"Interrupted at epoch {last_epoch} — saving current weights.")
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_state(), dir_checkpoint / "interrupted.pt")

    # Leave `model` holding the best weights, not the last epoch's.
    if best_path.exists() and tracker.best_epoch:
        best_state = torch.load(best_path, map_location=device)
        best_state.pop("mask_values", None)
        best_state.pop(CONVLSTM_CONFIG_KEY, None)
        model.load_state_dict(best_state)
        msg = (f"Restored best weights from epoch {tracker.best_epoch} "
               f"(val/dice={tracker.best:.4f}) into the model")
        rep.info(msg) if rep is not None else logging.info(msg)

    if rep is not None:
        elapsed = time.time() - run_start
        rep.info("=" * 72)
        status = "interrupted" if interrupted else "complete"
        rep.info(f"Training {status} — {last_epoch} epochs in {human_time(elapsed)}"
                 f" ({human_time(elapsed / max(last_epoch, 1))}/epoch)")
        if tracker.best_epoch:
            rep.info(f"Best val/dice {tracker.best:.4f} @ epoch {tracker.best_epoch}")
        rep.info(f"Weights   {dir_checkpoint}")
        if best_path.exists():
            rep.info(f"Best      {best_path}  (epoch {tracker.best_epoch})")
        if last_path.exists():
            rep.info(f"Last      {last_path}  (epoch {last_epoch})")
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
    now = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    args.job_name = f"{args.job_name}_{now}"
    outpath = os.path.join(args.output_dir, args.job_name)
    os.makedirs(outpath, exist_ok=True)

    log_file = os.path.join(outpath, f"{args.job_name}.log")
    logging.getLogger().addHandler(logging.FileHandler(log_file))
    logging.info("train job started")

    if args.reporter:
        # Console goes to the reporter; the run's .log file keeps everything.
        quiet_root_console(logging.WARNING)
        setup_logger(REPORTER_NAME, log_file=Path(outpath) / "logs" / "reporter.log")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = get_device()
    logging.info(f"Using device {device}")

    rep = logging.getLogger(REPORTER_NAME) if args.reporter else None
    train_set, val_set, test_set, partition_desc = build_datasets(args, rep)
    logging.info(f"partition: {partition_desc}; train/val/test = "
                 f"{len(train_set)}/{len(val_set)}/{len(test_set) if test_set is not None else 0} samples")

    if test_set is not None:
        pkl = os.path.join(outpath, f"test_dataset_{args.job_name}.pkl")
        size_gb = save_test_dataset(test_set, pkl)
        logging.info(f"test split pickled to {pkl} ({size_gb:.2f} GB)")

    model, num_c = build_model(args, device)
    ch_desc = (f"{model.n_channels} input channels per timestep "
               f"(sequence length T={args.k_prevs + 1}, not fixed by the architecture)"
               if args.convlstm_unet else f"{model.n_channels} input channels")
    logging.info(f"Network:\n\t{ch_desc}\n\t{model.n_classes} output channels\n"
                 f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling")

    train_model(args, model, device, train_set, val_set, test_set, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
