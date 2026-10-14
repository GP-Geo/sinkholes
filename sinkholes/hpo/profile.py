"""Peak VRAM and step time of the study's model, per geometry and micro-batch.

Synthetic inputs of the real shape (B, T=k+1, H, W), the real network built by
``train.build_model`` from the spec's fixed flags, AdamW, AMP with the same
unscale-then-clip, BCE(pos_w) + Dice, accumulation as in the spec. What this
cannot measure is HOST memory, which is the dataset's (see the resource
estimate); the first real training job's LSF summary is that measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from .spec import Spec, localize


def _train_args(spec: Spec, geometry: str, micro: int):
    from ..training.train import add_arguments

    p = argparse.ArgumentParser()
    add_arguments(p)
    f, g = spec.fixed, spec.geometry(geometry)
    args = p.parse_args([])
    args.add_temporal, args.k_prevs, args.tattn_unet = True, spec.k_prevs, True
    for k in ("tattn_dim", "tattn_heads", "tattn_layers", "tattn_recurrence", "tattn_fuse_skips",
              "tattn_contrast", "tattn_qk_norm", "convlstm_hidden", "amp", "seg_loss"):
        setattr(args, k, f[k])
    args.patch_size = list(f["patch_size"])
    args.context_margin = list(g["context_margin"])
    args.batch_size, args.accum_steps = micro, int(g["accum_steps"])
    args.pos_w = 8.0
    return args


def profile_one(spec: Spec, geometry: str, micro: int, steps: int, device) -> dict:
    import torch
    import torch.nn as nn

    from ..dataprep.context import context_size
    from ..models.temporal import run_model
    from ..training.losses import segmentation_loss
    from ..training.train import build_model

    args = _train_args(spec, geometry, micro)
    torch.manual_seed(0)
    model, _ = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-2)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_w], device=device))
    H, W = context_size(tuple(args.patch_size), tuple(args.context_margin)) \
        if tuple(args.context_margin) != (0, 0) else tuple(args.patch_size)
    T = spec.k_prevs + 1
    x = torch.rand(micro, T, H, W, device=device)
    y = (torch.rand(micro, *args.patch_size, device=device) > 0.9).long()
    offsets = torch.arange(T - 1, -1, -1, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    times = []
    model.train()
    for step in range(steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum_steps):
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=args.amp):
                logits = run_model(model, x, offsets)
            loss = segmentation_loss(logits, x, y, n_classes=1, treat_nodata_regions=False,
                                     criterion=crit, region_loss=args.seg_loss)
            scaler.scale(loss / args.accum_steps).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step >= 2:          # warm-up excluded
            times.append(time.time() - t0)
    out = {"geometry": geometry, "micro_batch": micro, "accum": args.accum_steps,
           "input": [T, H, W], "s_per_optimizer_step": sum(times) / max(len(times), 1),
           "samples_per_s": micro * args.accum_steps * len(times) / max(sum(times), 1e-9)}
    if device.type == "cuda":
        out["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        out["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2 ** 30
        out["gpu"] = torch.cuda.get_device_name(device)
    del model, opt, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def profile(spec: Spec, microbatches: List[int], steps: int, out_path=None) -> List[dict]:
    import torch

    from ..device import get_device

    device = get_device()
    results = []
    for g in spec.geometries:
        for mb in microbatches:
            if device.type != "cuda" and mb > 2:
                continue  # off-GPU this is a smoke test only
            try:
                r = profile_one(spec, g, mb, steps, device)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                r = {"geometry": g, "micro_batch": mb, "oom": True}
            results.append(r)
            print(json.dumps(r), flush=True)
        if device.type != "cuda":
            results.append(profile_one(spec, g, 2, 3, device) | {"note": "not CUDA: smoke test"})
            print(json.dumps(results[-1]), flush=True)
    if out_path:
        Path(localize(out_path)).parent.mkdir(parents=True, exist_ok=True)
        Path(localize(out_path)).write_text(json.dumps(results, indent=1))
    return results
