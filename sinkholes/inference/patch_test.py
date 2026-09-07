"""Patch-level test: a test split + a checkpoint -> Dice, pixel and
object-level precision/recall.

Run as ``sinkholes test-patches``. Reports the mean of per-patch Dice (macro —
the harsher number; ``inspect-run`` reports the pooled micro variant).

The split comes from one of two sources:

- ``--test_data_path``: a pickled split written by training. Only
  ``--partition_mode random_by_intf`` runs have one — under
  ``preset_by_intf`` training builds no test set and pickles nothing.
- ``--partition_file`` + ``--split``: the split is built here, in place, from
  a partition JSON and the patch trees on disk. This is how a
  ``preset_by_intf`` checkpoint (the whole ``geo_*`` / ``temporal_*`` family)
  is scored without retraining it.

Either way the set is **positives-only** — patches that contain subsidence and
nothing else — which is the protocol the paper this work is benchmarked
against uses. It is a baseline-comparison number, never a model-selection one:
positives-only precision cannot see the false positives a model scatters over
the 97.5% of the map that holds no subsidence, and ranks ring-negative runs
backwards for exactly that reason (docs/RESULTS.md).
"""

import argparse
import json
import logging


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--test_data_path", type=str, default=None,
                   help="pickled test split written by training (random_by_intf runs only)")
    p.add_argument("--model", type=str, required=True, help="checkpoint path")
    p.add_argument("--th", type=float, default=0.7,
                   help="object-level overlap threshold (fraction of a GT object "
                        "that must be covered)")
    p.add_argument("--b", type=int, default=5,
                   help="buffer in pixels applied to predicted objects when matching")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--k_prevs", type=int, default=0,
                   help="temporal context: the channel check for a pickled split, and "
                        "the chain length built for --partition_file --add_temporal")
    p.add_argument("--attn_unet", action="store_true")
    p.add_argument("--add_attn", action="store_true")
    p.add_argument("--convlstm_unet", action="store_true")
    p.add_argument("--tattn_unet", action="store_true")
    p.add_argument("--treat_nodata_regions", action="store_true")
    p.add_argument("--preprocessing_version", choices=["frame-v2", "legacy-row-v1"],
                   default=None, help="default: saved checkpoint policy, legacy-row-v1 for old weights")
    p.add_argument("--aoi_selection_version", choices=["coordinates-v2", "legacy-v1"],
                   default=None, help="default: saved checkpoint policy, legacy-v1 for old weights")

    # -- building the split in place from a partition JSON ---------------------------
    # These mirror the train.py flag names exactly so they can be pasted out of
    # a run's log header, which is the only record of how a run was configured.
    g = p.add_argument_group(
        "split built from a partition JSON",
        "instead of --test_data_path; the flags mirror train.py so a run's "
        "settings can be copied from its log")
    g.add_argument("--partition_file", type=str, default=None,
                   help="partition JSON to evaluate")
    g.add_argument("--split", type=str, default="val", choices=["val", "test", "train"],
                   help="which of its lists to score (default: val, the held-out data "
                        "for a preset_by_intf run)")
    g.add_argument("--patches_dir", type=str, default=None,
                   help="root holding the {data,mask}_patches_* trees")
    g.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    g.add_argument("--stride", type=int, default=2, help="strides per patch window")
    g.add_argument("--train_on_11d_diff", action=argparse.BooleanOptionalAction, default=True,
                   help="the 11-day patch tree (the default) rather than the all-durations one")
    g.add_argument("--use_cleaned_patches", action="store_true")
    g.add_argument("--add_temporal", action="store_true",
                   help="build the k-previous stacks; required by the sequence architectures")
    g.add_argument("--union_temporal_mask", action="store_true",
                   help="target is the union over the stack, as in training")
    g.add_argument("--intf_dict_path", type=str, default=None,
                   help="coordinate dictionary used to find the 11-day chains")
    g.add_argument("--nonz_only", action=argparse.BooleanOptionalAction, default=True,
                   help="positives-only, as training's val/test sets are. Turning it off "
                        "loads the full grids and is not the paper's protocol")
    g.add_argument("--save_test_dataset", type=str, default=None,
                   help="pickle the built split here so a rescore can use --test_data_path")
    g.add_argument("--metrics_out", type=str, default=None,
                   help="write the reported numbers to this JSON")


def build_split_from_partition(args):
    """The positives-only split of a partition JSON, built the way training builds it.

    Mirrors ``train.build_datasets``: same directory resolution, same chain
    filter, same dataset arguments. ``mode="test"`` is what keeps the set
    positives-only — it is the mode that takes neither ring nor validation
    negatives (``dataprep/dataset.py``).
    """
    from ..dataprep.dataset import SubsiDataset
    from ..dataprep.partition import load_partition_split, load_partition_window
    from ..dataprep.context import context_margin
    from ..dataprep.patchify import resolve_patch_dirs
    from ..meta import find_11day_sequences, load_coord_dict

    # The split list is read first: it is the cheaper check of the two, so a
    # typo in --split is reported as itself rather than behind a directory
    # probe that also failed.
    intf_list = load_partition_split(args.partition_file, args.split)
    logging.info(f"{args.split} split of {args.partition_file}: "
                 f"{len(intf_list)} interferograms")

    H, W = args.patch_size
    ctx = tuple(getattr(args, "_context_size", None) or (H, W))
    image_dir, mask_dir = resolve_patch_dirs(
        args.patches_dir, (H, W), args.stride,
        days_diff=11 if args.train_on_11d_diff else None,
        cleaned=args.use_cleaned_patches,
        context_margin=context_margin((H, W), ctx),
    )
    logging.info(f"patch directories: {image_dir} | {mask_dir}")

    seq_dict = None
    coord_dict = load_coord_dict(args.intf_dict_path)
    if args.add_temporal:
        seq_dict, with_chains = find_11day_sequences(coord_dict, k_prev=args.k_prevs,
                                                     restrict_to=intf_list)
        dropped = sorted(set(intf_list) - set(with_chains))
        if dropped:
            # Training would have raised on these, so a partition that loses
            # interferograms here is not the one the checkpoint was trained on:
            # the denominator of every metric below has changed.
            logging.warning(
                f"{len(dropped)} interferograms have no full {args.k_prevs}-previous "
                f"chain and are dropped: {dropped}. Training on this partition would "
                f"have failed, so check --k_prevs and --partition_file against the run log."
            )
        intf_list = with_chains
    if not intf_list:
        raise SystemExit(f"no interferograms left in the {args.split} split")

    return SubsiDataset(
        image_dir, mask_dir, intf_list, mode="test",
        patch_size=(H, W), stride=args.stride,
        context_size=ctx,
        coord_dict=coord_dict,
        # Before this hotfix, test-patches did not pass an AOI for ANY model.
        # Keep that separate historical command behavior for legacy weights.
        aoi_window=(load_partition_window(args.partition_file, args.split)
                    if (getattr(args, "aoi_selection_version", None) or "coordinates-v2") == "coordinates-v2"
                    else None),
        preprocessing_version=getattr(args, "preprocessing_version", None) or "frame-v2",
        aoi_selection_version=getattr(args, "aoi_selection_version", None) or "coordinates-v2",
        nonz_only=args.nonz_only,
        temporal=args.add_temporal, seq_dict=seq_dict,
        treat_nodata_regions=args.treat_nodata_regions,
        union_temporal_mask=args.union_temporal_mask,
        use_cleaned_patches=args.use_cleaned_patches,
    )


def resolve_test_data(args):
    """The split to score, from whichever of the two sources was given."""
    from ..dataprep.dataset import load_test_dataset, save_test_dataset

    if bool(args.test_data_path) == bool(args.partition_file):
        raise SystemExit("give exactly one of --test_data_path (a pickled split) or "
                         "--partition_file (build the split from a partition JSON)")

    if args.test_data_path:
        test_data = load_test_dataset(args.test_data_path)
        logging.info(f"{len(test_data)} test patches from {args.test_data_path}")
        return test_data, args.test_data_path

    if not args.patches_dir:
        raise SystemExit("--partition_file needs --patches_dir (the root holding the "
                         "data_patches_* / mask_patches_* trees)")
    test_data = build_split_from_partition(args)
    source = f"{args.partition_file} [{args.split}]"
    logging.info(f"{len(test_data)} patches from {source}")
    if args.save_test_dataset:
        size_gb = save_test_dataset(test_data, args.save_test_dataset)
        logging.info(f"split pickled to {args.save_test_dataset} ({size_gb:.2f} GB)")
    return test_data, source


def main(args) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    from ..device import get_device
    from ..models.factory import architecture_from_flags, build_from_checkpoint
    from ..training.evaluate import evaluate

    logging.basicConfig(level=logging.INFO)
    device = get_device()
    state_dict = torch.load(args.model, map_location=device)
    loaded = build_from_checkpoint(
        state_dict,
        arch=architecture_from_flags(
            convlstm_unet=args.convlstm_unet, tattn_unet=args.tattn_unet,
            attn_unet=args.attn_unet, add_attn=args.add_attn,
        ),
        n_channels=args.k_prevs + 1,
        n_classes=1,
        bilinear=False,
        treat_nodata_regions=args.treat_nodata_regions,
    )
    net = loaded.model
    net.to(device=device)
    net.load_state_dict(state_dict)
    net.eval()
    logging.info(f"architecture {loaded.architecture} ({loaded.n_channels} in-channels) on {device}")

    args._context_size = loaded.input_size_for(tuple(args.patch_size))
    for key in ("preprocessing_version", "aoi_selection_version"):
        requested = getattr(args, key, None)
        saved = loaded.data_contract[key]
        if requested is None:
            setattr(args, key, saved)
        elif requested != saved:
            logging.warning(f"explicit {key} override: {saved} -> {requested}; "
                            "this can change historical patch metrics")
    logging.info(f"patch policies: {args.preprocessing_version}, {args.aoi_selection_version}")
    test_data, source = resolve_test_data(args)
    if getattr(test_data, "n_negative", 0):
        logging.warning(f"{test_data.n_negative} of {len(test_data)} patches are empty — "
                        f"this is not a positives-only set")
    else:
        logging.info("positives-only protocol: every patch contains subsidence. These "
                     "numbers compare against the paper, NOT against full-scene ones, "
                     "and must never be used to select a model (docs/RESULTS.md)")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if len(test_data):
        sample = test_data[0]
        if tuple(sample["image"].shape[-2:]) != args._context_size:
            raise SystemExit("test dataset input geometry does not match checkpoint context")
        if tuple(sample["mask"].shape[-2:]) != tuple(args.patch_size):
            raise SystemExit("test dataset target geometry does not match checkpoint")
        base_data = test_data
        while isinstance(base_data, Subset):
            base_data = base_data.dataset
        actual = getattr(base_data, "preprocessing_version", "legacy-row-v1")
        if actual != args.preprocessing_version:
            raise SystemExit(f"pickled dataset uses {actual}, requested {args.preprocessing_version}; "
                             "build a split from --partition_file with the desired policy")

    metrics: dict = {}
    dice = evaluate(net, test_loader, device, amp=False, mode="test",
                    th=args.th, buffer=args.b, metrics_out=metrics)
    logging.info(f"test dice score: {dice}")

    if args.metrics_out:
        results = dict(metrics.get("test", {}))
        results.update(
            source=source, model=args.model, patches=len(test_data),
            preprocessing_version=args.preprocessing_version,
            aoi_selection_version=args.aoi_selection_version,
            input_size=list(args._context_size), target_size=list(args.patch_size),
            positives_only=bool(getattr(test_data, "n_negative", 0) == 0),
            th=args.th, buffer=args.b,
        )
        if args.partition_file:
            results.update(partition_file=args.partition_file, split=args.split,
                           interferograms=len(test_data.ids))
        with open(args.metrics_out, "w") as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"metrics written to {args.metrics_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
