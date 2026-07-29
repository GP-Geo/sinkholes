"""
Make a locally-downloaded patch subset compatible with train_sinkholes_unet.py.

The download under data/patches/ holds the FULL per-interferogram patch grids
(ny, nx, H, W) but not the `*_nonz_*` positive-only subsets that the training
code requires, and data/metadata/intf_coord.json carries `nonz_num` values from
a different patchify run. This script closes both gaps without touching the
downloaded files:

  1. derives data_patches_nonz_<intf>_*.npy / mask_patches_nonz_<intf>_*.npy
     by re-indexing the existing grids with nonz_indices.json,
  2. symlinks the full grids alongside them (needed by --add_temporal and
     --partition_mode spatial),
  3. writes intf_coord_local.json with nonz_num recounted for the local
     interferograms and 'none' for every interferogram not downloaded, so
     training skips them automatically.

Everything lands in --out_dir; the download is opened read-only.

    python prepare_local_subset.py
    python prepare_local_subset.py --verify_full     # recount from the masks
"""

# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

INTF_RE = re.compile(r'data_patches_(\d{8}_\d{8})_H(\d+)_W(\d+)_strpp(\d+)\.npy$')


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image_dir', type=str, default='data/patches/images',
                   help='downloaded full data patch grids')
    p.add_argument('--mask_dir', type=str, default='data/patches/masks',
                   help='downloaded full mask patch grids')
    p.add_argument('--nonz_indices', type=str, default='data/metadata/nonz_indices.json',
                   help='positive patch grid coordinates produced by patchify()')
    p.add_argument('--intf_dict', type=str, default=_bootstrap.asset('intf_coord.json'),
                   help='reference interferogram metadata to copy and correct')
    p.add_argument('--out_dir', type=str, default='data/patches/train_ready',
                   help='parent to pass to train_sinkholes_unet.py --patches_dir')
    p.add_argument('--out_intf_dict', type=str, default='data/metadata/intf_coord_local.json',
                   help='corrected metadata to pass as --intf_dict_path')
    p.add_argument('--days_diff', type=int, default=11,
                   help='must match the _<N>days suffix the training code builds')
    p.add_argument('--verify_full', action='store_true',
                   help='recount positives from the mask grids instead of trusting '
                        'nonz_indices.json (reads every mask; slow)')
    p.add_argument('--force', action='store_true',
                   help='rewrite nonz files that already exist')
    return p.parse_args()


def discover(image_dir, mask_dir):
    """Return [(intf, H, W, strpp)] for grids present as BOTH image and mask."""
    found = []
    for fname in sorted(os.listdir(image_dir)):
        m = INTF_RE.match(fname)
        if not m:
            continue
        intf, H, W, strpp = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        mask_name = f'mask_patches_{intf}_H{H}_W{W}_strpp{strpp}.npy'
        if not (Path(mask_dir) / mask_name).exists():
            print(f'  ! {intf}: image grid present but mask grid missing, skipping')
            continue
        found.append((intf, H, W, strpp))
    return found


def positive_indices(mask_grid, nonz_json, intf, verify_full):
    """Grid coords of patches whose mask has >=1 positive pixel."""
    if verify_full:
        grid = (np.asarray(mask_grid) > 0).any(axis=(-2, -1))
        return [tuple(c) for c in np.argwhere(grid).tolist()], 'recounted'
    idx = [(int(i), int(j)) for i, j in nonz_json.get(intf, [])]
    ny, nx = mask_grid.shape[:2]
    return [(i, j) for i, j in idx if 0 <= i < ny and 0 <= j < nx], 'nonz_indices.json'


def main():
    args = get_args()
    image_dir, mask_dir = Path(args.image_dir), Path(args.mask_dir)
    for d in (image_dir, mask_dir):
        if not d.is_dir():
            sys.exit(f'not a directory: {d}')

    nonz_json = json.load(open(args.nonz_indices)) if not args.verify_full else {}
    if not args.verify_full and not nonz_json:
        sys.exit(f'{args.nonz_indices} is empty; rerun with --verify_full')

    grids = discover(image_dir, mask_dir)
    if not grids:
        sys.exit(f'no data_patches_<intf>_H*_W*_strpp*.npy grids under {image_dir}')
    print(f'found {len(grids)} interferograms with both image and mask grids')

    # All grids must share patch geometry: the output directory name encodes it.
    geoms = {(H, W, s) for _, H, W, s in grids}
    if len(geoms) > 1:
        sys.exit(f'mixed patch geometries {geoms}; run once per geometry with --out_dir')
    H, W, strpp = geoms.pop()
    suffix = f'H{H}_W{W}_strpp{strpp}_{args.days_diff}days_Aligned'
    d_out = Path(args.out_dir) / f'data_patches_{suffix}'
    m_out = Path(args.out_dir) / f'mask_patches_{suffix}'
    d_out.mkdir(parents=True, exist_ok=True)
    m_out.mkdir(parents=True, exist_ok=True)
    print(f'patch geometry H={H} W={W} strpp={strpp}')
    print(f'output image dir: {d_out}')
    print(f'output mask  dir: {m_out}\n')

    counts, total_patches, total_bytes = {}, 0, 0
    for intf, *_ in grids:
        ext = f'{intf}_H{H}_W{W}_strpp{strpp}.npy'
        img_src = image_dir / f'data_patches_{ext}'
        msk_src = mask_dir / f'mask_patches_{ext}'
        d_nonz, m_nonz = d_out / f'data_patches_nonz_{ext}', m_out / f'mask_patches_nonz_{ext}'

        # Full grids stay available for --add_temporal / --partition_mode spatial.
        for src, dst in ((img_src, d_out / f'data_patches_{ext}'),
                         (msk_src, m_out / f'mask_patches_{ext}')):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

        if d_nonz.exists() and m_nonz.exists() and not args.force:
            n = np.load(d_nonz, mmap_mode='r').shape[0]
            counts[intf] = n
            total_patches += n
            print(f'  = {intf}: {n:4d} positive patches (exists, --force to rebuild)')
            continue

        img = np.load(img_src, mmap_mode='r')
        msk = np.load(msk_src, mmap_mode='r')
        if img.shape != msk.shape:
            print(f'  ! {intf}: image grid {img.shape} != mask grid {msk.shape}, skipping')
            continue
        if img.ndim != 4:
            print(f'  ! {intf}: expected a 4-D (ny,nx,H,W) grid, got {img.ndim}-D, skipping')
            continue

        idx, source = positive_indices(msk, nonz_json, intf, args.verify_full)
        if not idx:
            print(f'  ! {intf}: no positive patches, skipping')
            continue

        d = np.stack([np.asarray(img[i, j]) for i, j in idx]).astype(np.float32)
        m = np.stack([np.asarray(msk[i, j]) for i, j in idx]).astype(np.uint8)

        # A patch listed as positive must actually contain a positive pixel.
        empty = int((~(m.reshape(len(m), -1) > 0).any(axis=1)).sum())
        if empty:
            print(f'  ! {intf}: {empty} of {len(m)} listed patches have an empty mask '
                  f'-- {args.nonz_indices} disagrees with the grids, rerun --verify_full')

        np.save(d_nonz, d)
        np.save(m_nonz, m)
        counts[intf] = len(idx)
        total_patches += len(idx)
        total_bytes += d.nbytes + m.nbytes
        print(f'  + {intf}: {len(idx):4d} positive patches from {source}  '
              f'grid={img.shape[:2]}  {(d.nbytes + m.nbytes) / 1e6:6.1f} MB')

    if not counts:
        sys.exit('nothing was written')

    # --add_temporal reads this from the image directory, not from data/metadata/.
    shutil.copyfile(args.nonz_indices, d_out / 'nonz_indices.json')
    print(f'\ncopied {args.nonz_indices} -> {d_out / "nonz_indices.json"}')

    # Correct nonz_num so training filters to exactly what was downloaded.
    meta = json.load(open(args.intf_dict))
    changed = 0
    for intf, info in meta.items():
        want = counts.get(intf, 'none')
        if info.get('nonz_num') != want:
            changed += 1
        info['nonz_num'] = want
    Path(args.out_intf_dict).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_intf_dict, 'w') as f:
        json.dump(meta, f, indent=4)

    print(f'wrote {args.out_intf_dict}: {len(counts)} interferograms with a real '
          f"nonz_num, {len(meta) - len(counts)} set to 'none' ({changed} values changed)")
    print(f'\ntotal: {total_patches} positive patches, {total_bytes / 1e6:.0f} MB written')
    print(f'\ntrain with:\n'
          f'  --patches_dir {args.out_dir.rstrip("/")}/ \\\n'
          f'  --intf_dict_path {args.out_intf_dict}')


if __name__ == '__main__':
    main()
