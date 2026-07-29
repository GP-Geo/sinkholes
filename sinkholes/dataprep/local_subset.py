"""Make a locally-downloaded patch subset training-ready.

Run as ``sinkholes prepare-local-subset``. A partial download typically holds
the full ``(ny, nx, H, W)`` grids but not the positive-only ``nonz`` subsets
that training reads, and the committed dictionary's ``nonz_num`` counts come
from a different patchify run. This closes both gaps without touching the
downloaded files:

1. derives the ``*_nonz_*`` files by re-indexing the grids with
   ``nonz_indices.json``,
2. symlinks the full grids alongside them (temporal training and full-scene
   evaluation read those),
3. writes a corrected dictionary with ``nonz_num`` recounted for the local
   interferograms and ``'none'`` for everything else — which automatically
   restricts training to exactly what is on disk.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from ..paths import asset

INTF_RE = re.compile(r"data_patches_(\d{8}_\d{8})_H(\d+)_W(\d+)_strpp(\d+)\.npy$")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image_dir", type=str, default="data/patches/images",
                   help="downloaded full data patch grids")
    p.add_argument("--mask_dir", type=str, default="data/patches/masks",
                   help="downloaded full mask patch grids")
    p.add_argument("--nonz_indices", type=str, default="data/metadata/nonz_indices.json",
                   help="positive patch grid coordinates produced by patchify")
    p.add_argument("--intf_dict", type=str, default=None,
                   help="reference dictionary to copy and correct (default: the asset)")
    p.add_argument("--out_dir", type=str, default="data/patches/train_ready",
                   help="value to pass to training as --patches_dir")
    p.add_argument("--out_intf_dict", type=str, default="data/metadata/intf_coord_local.json",
                   help="corrected dictionary to pass as --intf_dict_path")
    p.add_argument("--days_diff", type=int, default=11)
    p.add_argument("--verify_full", action="store_true",
                   help="recount positives from the mask grids instead of trusting "
                        "nonz_indices.json (reads every mask; slow)")
    p.add_argument("--force", action="store_true", help="rewrite existing nonz files")


def discover(image_dir, mask_dir):
    """[(intf, H, W, strpp)] for grids present as BOTH image and mask."""
    found = []
    for fname in sorted(os.listdir(image_dir)):
        m = INTF_RE.match(fname)
        if not m:
            continue
        intf, H, W, strpp = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if not (Path(mask_dir) / f"mask_patches_{intf}_H{H}_W{W}_strpp{strpp}.npy").exists():
            print(f"  ! {intf}: image grid present but mask grid missing, skipping")
            continue
        found.append((intf, H, W, strpp))
    return found


def positive_indices(mask_grid, nonz_json, intf, verify_full):
    if verify_full:
        grid = (np.asarray(mask_grid) > 0).any(axis=(-2, -1))
        return [tuple(c) for c in np.argwhere(grid).tolist()], "recounted"
    idx = [(int(i), int(j)) for i, j in nonz_json.get(intf, [])]
    ny, nx = mask_grid.shape[:2]
    return [(i, j) for i, j in idx if 0 <= i < ny and 0 <= j < nx], "nonz_indices.json"


def main(args) -> None:
    image_dir, mask_dir = Path(args.image_dir), Path(args.mask_dir)
    for d in (image_dir, mask_dir):
        if not d.is_dir():
            sys.exit(f"not a directory: {d}")

    nonz_json = json.load(open(args.nonz_indices)) if not args.verify_full else {}
    if not args.verify_full and not nonz_json:
        sys.exit(f"{args.nonz_indices} is empty; rerun with --verify_full")

    grids = discover(image_dir, mask_dir)
    if not grids:
        sys.exit(f"no data_patches_<intf>_H*_W*_strpp*.npy grids under {image_dir}")
    print(f"found {len(grids)} interferograms with both image and mask grids")

    geoms = {(H, W, s) for _, H, W, s in grids}
    if len(geoms) > 1:
        sys.exit(f"mixed patch geometries {geoms}; run once per geometry with --out_dir")
    H, W, strpp = geoms.pop()
    suffix = f"H{H}_W{W}_strpp{strpp}_{args.days_diff}days_Aligned"
    d_out = Path(args.out_dir) / f"data_patches_{suffix}"
    m_out = Path(args.out_dir) / f"mask_patches_{suffix}"
    d_out.mkdir(parents=True, exist_ok=True)
    m_out.mkdir(parents=True, exist_ok=True)
    print(f"patch geometry H={H} W={W} strpp={strpp}")

    counts, total_patches, total_bytes = {}, 0, 0
    for intf, *_ in grids:
        ext = f"{intf}_H{H}_W{W}_strpp{strpp}.npy"
        img_src = image_dir / f"data_patches_{ext}"
        msk_src = mask_dir / f"mask_patches_{ext}"
        d_nonz = d_out / f"data_patches_nonz_{ext}"
        m_nonz = m_out / f"mask_patches_nonz_{ext}"

        for src, dst in ((img_src, d_out / f"data_patches_{ext}"),
                         (msk_src, m_out / f"mask_patches_{ext}")):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

        if d_nonz.exists() and m_nonz.exists() and not args.force:
            n = np.load(d_nonz, mmap_mode="r").shape[0]
            counts[intf] = n
            total_patches += n
            print(f"  = {intf}: {n:4d} positive patches (exists, --force to rebuild)")
            continue

        img = np.load(img_src, mmap_mode="r")
        msk = np.load(msk_src, mmap_mode="r")
        if img.shape != msk.shape or img.ndim != 4:
            print(f"  ! {intf}: unexpected grid shapes {img.shape} / {msk.shape}, skipping")
            continue

        idx, source = positive_indices(msk, nonz_json, intf, args.verify_full)
        if not idx:
            print(f"  ! {intf}: no positive patches, skipping")
            continue

        d = np.stack([np.asarray(img[i, j]) for i, j in idx]).astype(np.float32)
        m = np.stack([np.asarray(msk[i, j]) for i, j in idx]).astype(np.uint8)
        empty = int((~(m.reshape(len(m), -1) > 0).any(axis=1)).sum())
        if empty:
            print(f"  ! {intf}: {empty} listed patches have an empty mask — "
                  f"{args.nonz_indices} disagrees with the grids, rerun --verify_full")

        np.save(d_nonz, d)
        np.save(m_nonz, m)
        counts[intf] = len(idx)
        total_patches += len(idx)
        total_bytes += d.nbytes + m.nbytes
        print(f"  + {intf}: {len(idx):4d} positive patches from {source}  "
              f"grid={img.shape[:2]}  {(d.nbytes + m.nbytes) / 1e6:6.1f} MB")

    if not counts:
        sys.exit("nothing was written")

    # Temporal training reads this from inside the image directory.
    shutil.copyfile(args.nonz_indices, d_out / "nonz_indices.json")
    print(f"\ncopied {args.nonz_indices} -> {d_out / 'nonz_indices.json'}")

    meta = json.load(open(args.intf_dict or asset("intf_coord.json")))
    changed = 0
    for intf, info in meta.items():
        want = counts.get(intf, "none")
        if info.get("nonz_num") != want:
            changed += 1
        info["nonz_num"] = want
    Path(args.out_intf_dict).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_intf_dict, "w") as fh:
        json.dump(meta, fh, indent=4)

    print(f"wrote {args.out_intf_dict}: {len(counts)} interferograms counted, "
          f"{len(meta) - len(counts)} set to 'none' ({changed} values changed)")
    print(f"\ntotal: {total_patches} positive patches, {total_bytes / 1e6:.0f} MB written")
    print(f"\ntrain with:\n  --patches_dir {str(args.out_dir).rstrip('/')}/ \\\n"
          f"  --intf_dict_path {args.out_intf_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
