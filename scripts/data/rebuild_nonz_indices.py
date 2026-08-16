"""Rebuild nonz_indices.json from the mask grids already on disk.

``prepare-patches`` accumulates the positive-patch grid coordinates in memory
and writes ``nonz_indices.json`` only after its last interferogram, so a run
that dies partway leaves regenerated arrays beside a stale index — and
``dataset.py`` reads that index at training time. This recovers it without
redoing patch generation: the coordinates are recomputed from the mask grids,
and each interferogram is cross-checked against the length of its nonz array.

    python scripts/data/rebuild_nonz_indices.py \
        --patches_root /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches \
        --days_diff 11 --strides_per_patch 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinkholes.dataprep.patchify import patch_dir_name, patch_file_name  # noqa: E402
from sinkholes.meta import INTF_ID_RE  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patches_root", required=True)
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--strides_per_patch", type=int, default=2)
    p.add_argument("--days_diff", type=int, default=11)
    p.add_argument("--out_path", default=None,
                   help="default: nonz_indices.json inside the data tree")
    p.add_argument("--dry_run", action="store_true", help="compute and check, write nothing")
    args = p.parse_args()

    H, W = args.patch_size
    ddir = os.path.join(args.patches_root,
                        patch_dir_name("data", H, W, args.strides_per_patch, args.days_diff))
    mdir = os.path.join(args.patches_root,
                        patch_dir_name("mask", H, W, args.strides_per_patch, args.days_diff))
    for d in (ddir, mdir):
        if not os.path.isdir(d):
            raise SystemExit(f"not a directory: {d}")

    ids = sorted({INTF_ID_RE.search(f).group(0) for f in os.listdir(mdir)
                  if f.endswith(".npy") and "_nonz_" not in f and "_cleaned" not in f})
    print(f"{len(ids)} interferograms in {mdir}")

    out, mismatched = {}, []
    t0 = time.time()
    for n, intf_id in enumerate(ids, 1):
        grid = np.load(os.path.join(mdir, patch_file_name(
            "mask", intf_id, H, W, args.strides_per_patch)), mmap_mode="r")
        if grid.ndim != 4:
            raise SystemExit(f"{intf_id}: expected a (ny, nx, H, W) grid, got {grid.shape}")
        # Row-major, the order patchify appends in — so index k here is row k
        # of the nonz array.
        hits = np.argwhere(np.asarray(grid).any(axis=(2, 3)))
        out[intf_id] = [[int(i), int(j)] for i, j in hits]

        n_nonz = np.load(os.path.join(ddir, patch_file_name(
            "data", intf_id, H, W, args.strides_per_patch, nonz=True)), mmap_mode="r").shape[0]
        if n_nonz != len(out[intf_id]):
            mismatched.append((intf_id, len(out[intf_id]), n_nonz))

        if n % 25 == 0 or n == len(ids):
            el = time.time() - t0
            print(f"  {n}/{len(ids)}  {intf_id}: {len(out[intf_id]):5d} positive "
                  f"({el:.0f}s elapsed, {el / n * (len(ids) - n):.0f}s left)", flush=True)

    total = sum(len(v) for v in out.values())
    print(f"\n{len(out)} interferograms, {total} positive patches")
    if mismatched:
        print(f"!! {len(mismatched)} disagree with their nonz array length:")
        for row in mismatched[:10]:
            print(f"     {row[0]}: grid says {row[1]}, nonz array has {row[2]}")
        raise SystemExit("refusing to write a mismatched index")
    print("every count matches the nonz arrays")

    if args.dry_run:
        print("dry run: nothing written")
        return
    out_path = args.out_path or os.path.join(ddir, "nonz_indices.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
