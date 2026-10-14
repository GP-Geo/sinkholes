"""Does every file the study will open exist, whole, for both geometries?

Training (stride-2 trees) reads every train and val scene plus each one's
filled k-chain; scene evaluation (stride-4 trees) reads every scored scene plus
its chain. A context tree is a SUBSET by design (6.5 TB in full), so a missing
predecessor is the expected failure -- and the one the old eval-scenes turned
into a silently wrong number. This module derives the id sets with the
project's own chain code, checks the files, reads every ``.npy`` header (a
truncated write shows as a size mismatch), and prints the commands that fill
the gaps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..dataprep.patchify import patch_dir_name, patch_file_name
from .spec import Spec, localize


def _protocol(spec: Spec, split: str):
    from ..inference.snapshot_eval import Protocol

    ev = spec.evaluation
    partition = spec.partition if split == "val" else spec.test_partition
    # A _testeval partition carries the parent's TEST list under "val".
    return Protocol(partition=localize(partition), k_prevs=spec.k_prevs, split="val",
                    aoi_split="val", patch_size=tuple(spec.fixed["patch_size"]),
                    data_stride=int(ev["data_stride"]), rth=tuple(ev["rth"]),
                    min_positives=int(ev["min_positives"]))


def _coord():
    from ..meta import load_coord_dict

    return load_coord_dict(None)


def train_ids(spec: Spec) -> List[str]:
    """Every id the training dataset opens (train + val scenes and their chains)."""
    from ..meta import find_11day_sequences

    part = json.loads(Path(localize(spec.partition)).read_text())
    scenes = list(part["train"]) + list(part["val"])
    chains, valid = find_11day_sequences(_coord(), k_prev=spec.k_prevs, restrict_to=scenes)
    lost = sorted(set(scenes) - set(valid))
    if lost:
        raise SystemExit(f"partition scenes without a {spec.k_prevs}-chain: {lost}")
    need = set(scenes)
    for s in scenes:
        need.update(chains[s]["prevs"])
    return sorted(need)


def eval_ids(spec: Spec, split: str):
    from ..inference.snapshot_eval import required_ids

    return required_ids(_protocol(spec, split), _coord())


def _dirs(spec: Spec, geometry: str, stride: int):
    root = localize(spec.raw["patches_dir"])
    H, W = spec.fixed["patch_size"]
    margin = tuple(spec.geometry(geometry)["context_margin"])
    return (os.path.join(root, patch_dir_name("data", H, W, stride, 11, margin)),
            os.path.join(root, patch_dir_name("mask", H, W, stride, 11)))


def check_npy(path: str, cell: tuple) -> Optional[str]:
    """None if the header is sane and the file is as long as it says, else why not."""
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0(fh) if version == (1, 0)
                                     else np.lib.format.read_array_header_2_0(fh))
            offset = fh.tell()
    except Exception as exc:
        return f"unreadable header ({exc})"
    if len(shape) != 4 or tuple(shape[-2:]) != tuple(cell):
        return f"shape {shape}, expected (ny, nx, {cell[0]}, {cell[1]})"
    expected = offset + int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        return f"{actual} bytes, header implies {expected} (truncated write?)"
    return None


def requirement(spec: Spec, geometry: str, purpose: str, deep: bool = True) -> Dict:
    """One requirement: which files, which are missing or damaged."""
    H, W = spec.fixed["patch_size"]
    margin = tuple(spec.geometry(geometry)["context_margin"])
    cell = (H + 2 * margin[0], W + 2 * margin[1])
    if purpose == "train":
        stride, ids, scored = 2, train_ids(spec), None
    else:
        stride = int(spec.evaluation["data_stride"])
        scored, ids = eval_ids(spec, "val" if purpose == "eval_val" else "test")
    data_dir, mask_dir = _dirs(spec, geometry, stride)
    out = {"geometry": geometry, "purpose": purpose, "stride": stride, "data_dir": data_dir,
           "mask_dir": mask_dir, "n_ids": len(ids), "missing_data": [], "bad_data": {},
           "missing_mask": [], "bad_mask": {}, "nonz_index": None}
    for i in ids:
        f = os.path.join(data_dir, patch_file_name("data", i, H, W, stride))
        if not os.path.exists(f):
            out["missing_data"].append(i)
        elif deep and (why := check_npy(f, cell)):
            out["bad_data"][i] = why
    # Training reads masks for every chain member (negatives are excluded on
    # the union over time); evaluation only the scored scenes' masks.
    for i in (ids if purpose == "train" else scored):
        f = os.path.join(mask_dir, patch_file_name("mask", i, H, W, stride))
        if not os.path.exists(f):
            out["missing_mask"].append(i)
        elif deep and (why := check_npy(f, (H, W))):
            out["bad_mask"][i] = why
    if purpose == "train":
        idx = os.path.join(data_dir, "nonz_indices.json")
        if not os.path.exists(idx):
            out["nonz_index"] = "missing"
        else:
            keys = set(json.loads(Path(idx).read_text()))
            part = json.loads(Path(localize(spec.partition)).read_text())
            absent = sorted((set(part["train"]) | set(part["val"])) - keys)
            out["nonz_index"] = "ok" if not absent else f"lacks {absent}"
    else:
        out["scored"] = scored
    out["ok"] = not (out["missing_data"] or out["bad_data"] or out["missing_mask"]
                     or out["bad_mask"] or (out["nonz_index"] not in (None, "ok")))
    return out


def missing_eval_ids(spec: Spec, geometry: str, split: str = "val") -> List[str]:
    """The gate the controller applies before an evaluation job: absent data grids."""
    req = requirement(spec, geometry, "eval_val" if split == "val" else "eval_test", deep=False)
    return req["missing_data"] + [f"mask:{i}" for i in req["missing_mask"]]


def estimate_gb(spec: Spec, ids: List[str], geometry: str, stride: int) -> float:
    """Size of grids still to build, from the plain tree's real sizes x the pixel ratio."""
    H, W = spec.fixed["patch_size"]
    margin = tuple(spec.geometry(geometry)["context_margin"])
    ratio = (H + 2 * margin[0]) * (W + 2 * margin[1]) / (H * W)
    plain_dir, _ = _dirs(spec, "plain", stride)
    total = 0
    for i in ids:
        f = os.path.join(plain_dir, patch_file_name("data", i, H, W, stride))
        if os.path.exists(f):
            total += os.path.getsize(f)
    return total * ratio * 1.01 / 2 ** 30


def generation_command(spec: Spec, geometry: str, purpose: str) -> Optional[str]:
    if geometry == "ctx50" and purpose in ("eval_val", "eval_test"):
        part = spec.raw["partition"] if purpose == "eval_val" else spec.raw["test_partition"]
        return (f"PARTITION={part} K_PREVS={spec.k_prevs} MIN_POS={spec.evaluation['min_positives']} "
                f"bsub < scripts/data/make_context_stride4_patches.sh")
    if purpose == "train":
        return ("stride-2 trees are cut by scripts/data/make_context_patches.sh (ctx50) / "
                "scripts/data/regenerate_patches.sh (plain); none was needed when this tool was written")
    return "the plain stride-4 tree is built by scripts/data/make_stride4_patches.sh"


def minmax_status(spec: Spec, geometry: str, split: str = "val") -> Dict:
    req = requirement(spec, geometry, "eval_val" if split == "val" else "eval_test", deep=False)
    H, W = spec.fixed["patch_size"]
    cache = Path(localize(spec.root)) / "cache" / Path(req["data_dir"]).name
    have = [i for i in eval_ids(spec, split)[1]
            if (cache / (Path(patch_file_name("data", i, H, W, req["stride"])).stem
                         + ".minmax.npz")).exists()]
    return {"cached": len(have), "needed": req["n_ids"]}


def report(spec: Spec, splits=("val", "test"), deep: bool = True) -> Dict:
    rows = []
    for g in spec.geometries:
        for purpose in ["train"] + [f"eval_{s}" for s in splits]:
            req = requirement(spec, g, purpose, deep=deep)
            req["generate"] = None if req["ok"] else generation_command(spec, g, purpose)
            if req["missing_data"]:
                req["to_build_gb"] = round(estimate_gb(spec, req["missing_data"], g, req["stride"]), 1)
            rows.append(req)
    return {"study": spec.name, "requirements": rows}


def format_report(rep: Dict) -> str:
    lines = [f"coverage for study {rep['study']}", ""]
    for r in rep["requirements"]:
        state = "OK" if r["ok"] else "INCOMPLETE"
        lines.append(f"[{state:10s}] {r['geometry']:5s} {r['purpose']:9s} stride {r['stride']}: "
                     f"{r['n_ids']} ids  ({Path(r['data_dir']).name})")
        if r["missing_data"]:
            lines.append(f"     missing data grids: {len(r['missing_data'])} "
                         f"(~{r.get('to_build_gb', '?')} GB to build): {r['missing_data']}")
        for kind in ("bad_data", "bad_mask"):
            for i, why in r[kind].items():
                lines.append(f"     DAMAGED {kind[4:]} grid {i}: {why}")
        if r["missing_mask"]:
            lines.append(f"     missing mask grids: {r['missing_mask']}")
        if r.get("nonz_index") not in (None, "ok"):
            lines.append(f"     nonz_indices.json: {r['nonz_index']}")
        if r.get("generate"):
            lines.append(f"     generate: {r['generate']}")
    return "\n".join(lines)
