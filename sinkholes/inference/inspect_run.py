"""Inspect a finished training run.

Run as ``sinkholes inspect-run outputs/<run_dir> [epoch]``. Prints the learning
curve (from results.csv, or parsed from the run log for older runs), the best
epoch, and a decision-threshold sweep over the run's held-out test set.

Two Dice numbers exist in this project and both are correct: ``test-patches``
reports the mean of per-patch Dice (macro, the harsher number), this sweep
reports Dice over all test pixels pooled (micro).
"""

import argparse
import csv
import glob
import logging
import os
import re


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dir")
    p.add_argument("epoch", nargs="?", type=int, default=None,
                   help="checkpoint epoch to evaluate (default: best)")
    p.add_argument("--attn_unet", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)


def learning_curve(run_dir):
    """Per-epoch val Dice. Prefers results.csv; falls back to the run log."""
    csv_path = os.path.join(run_dir, "results.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows and "val/dice" in rows[0]:
            return {int(r["epoch"].split("/")[0]): float(r["val/dice"])
                    for r in rows if r.get("val/dice")}, "results.csv"

    logs = glob.glob(os.path.join(run_dir, "*.log"))
    if not logs:
        raise SystemExit(f"no results.csv and no *.log in {run_dir}")
    scores = {}
    with open(logs[0]) as fh:
        for line in fh:
            m = re.search(r"Epoch (\d+) Validation Dice score: ([\d.]+)", line)
            if m:
                scores[int(m.group(1))] = float(m.group(2))
    return scores, os.path.basename(logs[0])


def main(args) -> None:
    import pickle

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from ..dataprep.dataset import load_test_dataset
    from ..device import get_device
    from ..models.factory import architecture_from_flags, build_from_checkpoint

    logging.basicConfig(level=logging.WARNING)
    scores, source = learning_curve(args.run_dir)
    if not scores:
        raise SystemExit(f"no validation scores found in {args.run_dir}")
    best = max(scores, key=scores.get)
    last = max(scores)
    print(f"=== learning curve ({len(scores)} epochs, from {source}) ===")
    print(f"  first   epoch {min(scores):3d}   {scores[min(scores)]:.4f}")
    print(f"  last    epoch {last:3d}   {scores[last]:.4f}")
    print(f"  BEST    epoch {best:3d}   {scores[best]:.4f}")
    print(f"  last-vs-best gap       {scores[last] - scores[best]:+.4f}")

    epoch = args.epoch or best
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    hits = glob.glob(os.path.join(ckpt_dir, f"*checkpoint_epoch{epoch}.pth"))
    if not hits:
        named = {best: "best.pt", last: "last.pt"}.get(epoch)
        candidate = os.path.join(ckpt_dir, named) if named else None
        if candidate and os.path.exists(candidate):
            hits = [candidate]
            print(f"  (no per-epoch checkpoint; using {named} for epoch {epoch})")
    if not hits:
        available = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, "*.pt*")))
        raise SystemExit(
            f"no checkpoint for epoch {epoch} in {ckpt_dir}. "
            f"Available: {', '.join(available) or '(none)'}. "
            f"Use the best ({best}) or last ({last}) epoch."
        )
    ckpt = hits[0]
    pkls = glob.glob(os.path.join(args.run_dir, "test_dataset_*.pkl"))
    if not pkls:
        raise SystemExit(f"no test_dataset_*.pkl in {args.run_dir}")

    print(f"\n=== epoch {epoch} on the held-out test set ===")
    print(f"  checkpoint {os.path.basename(ckpt)}")
    test_data = load_test_dataset(pkls[0])
    print(f"  test set   {len(test_data)} patches")

    device = get_device()
    state = torch.load(ckpt, map_location=device)
    loaded = build_from_checkpoint(
        state, arch=architecture_from_flags(attn_unet=args.attn_unet),
        n_classes=1, bilinear=False,
    )
    net = loaded.model
    net.load_state_dict(state)
    net.to(device).eval()
    print(f"  device {device} | n_channels {loaded.n_channels}")

    ths = np.arange(0.05, 0.96, 0.05)
    tp = np.zeros_like(ths)
    fp = np.zeros_like(ths)
    fn = np.zeros_like(ths)
    n_pos = n_tot = 0

    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["image"].to(device, dtype=torch.float32)
            y = batch["mask"].to(device).float()
            if y.ndim == 4:
                y = y.squeeze(1)
            prob = torch.sigmoid(net(x)).squeeze(1)
            pos = y > 0.5
            n_pos += int(pos.sum().item())
            n_tot += int(y.numel())
            for k, t in enumerate(ths):
                pred = prob > float(t)
                tp[k] += float((pred & pos).sum().item())
                fp[k] += float((pred & ~pos).sum().item())
                fn[k] += float((~pred & pos).sum().item())
            if (i + 1) % 4 == 0:
                print(f"    ... {(i + 1) * args.batch_size} patches", flush=True)

    print(f"\n  positive pixels {n_pos:,} / {n_tot:,} "
          f"({100 * n_pos / n_tot:.2f}% — the imbalance --pos_w counteracts)")
    print(f"\n  {'thresh':>7} {'precision':>10} {'recall':>8} {'Dice':>8}")
    best_t, best_d = 0.0, 0.0
    for k, t in enumerate(ths):
        prec = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        rec = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        dice = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if dice > best_d:
            best_t, best_d = t, dice
        mark = "  <- default 0.5" if abs(t - 0.5) < 1e-6 else ""
        print(f"  {t:7.2f} {prec:10.4f} {rec:8.4f} {dice:8.4f}{mark}")
    print(f"\n  best pooled Dice {best_d:.4f} at threshold {best_t:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
