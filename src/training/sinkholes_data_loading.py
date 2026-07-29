# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import logging
import os
import sys
import random

import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import math
import albumentations as A
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Black for 0, White for 1
BW = ListedColormap(['black', 'white'])

def plot_temporal_patch_with_mask(img: np.ndarray, mask: np.ndarray = None,
                                  titles=None, cmap='jet'):
    """
    img : (T, H, W)
    mask: (H, W) or (T, H, W). If 3D, show union across T.
    """
    assert img.ndim == 3, f"Expected img shape (T,H,W), got {img.shape}"
    T, H, W = img.shape

    if titles is None:
        titles = ['Current'] + [f'Prev (t-{k})' for k in range(1, T)]

    ncols = T + (1 if mask is not None else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    # Common scaling for all temporal slices
    vmin = float(np.nanmin(img))
    vmax = float(np.nanmax(img))

    # Plot temporal slices
    for t in range(T):
        ax = axes[t]
        ax.imshow(img[t], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(titles[t], fontsize=10)
        ax.axis('off')

    # Final panel: mask (black=0, white=1)
    if mask is not None:
        ax = axes[-1]
        m = (mask > 0).astype(np.uint8) if mask.ndim == 2 else (mask > 0).any(axis=0).astype(np.uint8)
        ax.imshow(m, cmap=BW, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title('Mask', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob('mask_patches_nonz_' + idx + mask_suffix))[0]
    mask = np.load(mask_file)
    if mask[0].ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def has_consecutive_zeros(arr, min_consecutive=10, max_consecutive=1000):
    # Check rows
    row_check = np.apply_along_axis(
        lambda x: np.any(np.convolve(x == 0, np.ones(min_consecutive, dtype=int), 'valid') == min_consecutive),
        axis=1,
        arr=arr
    )

    # Check columns
    col_check = np.apply_along_axis(
        lambda x: np.any(np.convolve(x == 0, np.ones(min_consecutive, dtype=int), 'valid') == min_consecutive),
        axis=0,
        arr=arr
    )

    return np.any(row_check) or np.any(col_check)


# EDIT 2026-07-28: the temporal target is now the LATEST timestep's mask, not the union
# across the stack. CHANGELOG.md #7
def temporal_target_mask(masks_per_t, union: bool = False):
    """Ground-truth mask for a temporal sample.

    `masks_per_t`: list of (N, H, W) arrays, chronological **oldest -> newest**
    (`tids = list(prevs) + [id]`, see :176).

    Default: the newest interferogram's mask, which is what the model is asked to
    predict — the decoder's skip connections come from the newest timestep and the
    ConvLSTM hidden state summarises the history leading up to it.

    `union=True` restores the previous behaviour (positive wherever *any* timestep was
    positive) for comparison against runs made before 2026-07-28; it is reachable from
    the CLI as `--union_temporal_mask`.
    """
    if union:
        return (np.stack(masks_per_t, axis=0) > 0).any(axis=0).astype(np.float32)
    return (masks_per_t[-1] > 0).astype(np.float32)


class SubsiDataset(Dataset):
    test_dataset_for_nonoverlap_split = []
    test_mask_for_nonoverlap_split = []

    def __init__(self, args, image_dir, mask_dir, intrfrgrm_list=None, scale: float = 1.0, mask_suffix: str = '',
                 dset='train', augment=False, seq_dict=None):
        super(SubsiDataset, self).__init__()

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        assert os.path.exists(self.image_dir) and os.path.exists(
            self.mask_dir), 'The data you are requesting does not exist'
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        if args.partition_mode == 'preset_by_intf':
            assert intrfrgrm_list is not None, "Partition mode 'preset by intf' requires an input intrfrgrm list"
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.dataset = dset
        if args.add_temporal:
            self.temporal = True
            with open(image_dir + '/' + 'nonz_indices.json', "r") as f:
                nonz_patches_dict = json.load(f)
                self.seq_dict = seq_dict
        else:
            self.temporal = False
        # EDIT 2026-07-28: index at which the validity channels begin, i.e. T when
        # --treat_nodata_regions appends a per-time validity map. None means the sample
        # carries no validity channels. Set where the concatenation actually happens
        # (:296) so it can never disagree with the data. CHANGELOG.md #8
        self.n_value_channels = None
        if args.nonz_only and args.partition_mode != 'spatial' and not args.add_nulls_to_train and not args.nonoverlap_tr_tst and not self.temporal:
            pref, mask_pref = 'data_patches_nonz_', 'mask_patches_nonz_'
        else:
            pref, mask_pref = 'data_patches_', 'mask_patches_'
        if args.use_cleaned_patches:
            suff, mask_suff = '_cleaned', '_cleaned'
        else:
            suff, mask_suff = '', ''
        start_intf_name = len(pref)
        if args.partition_mode == 'spatial':
            with open(args.intf_dict_path, 'r') as json_file:
                coord_dict = json.load(json_file)

        if intrfrgrm_list is None:
            self.ids = [file.split('.')[0][start_intf_name:start_intf_name + 17] for file in listdir(image_dir) if
                        ('nonz' in file and args.nonz_only and args.partition_mode != 'spatial') or (
                                    'nonz' not in file and (not args.nonz_only or args.partition_mode == 'spatial'))]
        else:
            self.ids = intrfrgrm_list
        if (args.test == 0 and dset == 'test'):
            self.index_map = []
            return

        if not self.ids:
            raise RuntimeError(f'empty set when trying to takr data from {image_dir}, recheck!!')

        self.image_data, self.mask_data, self.index_map = [], [], []
        test_data, test_mask_data = [], []
        for i, id in enumerate(self.ids):
            def img_path(tid):
                return join(self.image_dir,
                            pref + tid + f"_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}{suff}.npy")

            def msk_path(tid):
                return join(self.mask_dir, pref.replace("data_",
                                                        "mask_") + tid + f"_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}{suff}.npy")

            if self.temporal:

                tids = list(self.seq_dict[id]["prevs"]) + [id]  # prevs…present (T = k_prev+1)

            if args.partition_mode != 'spatial':

                image_data = np.load(join(self.image_dir, pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                    args.patch_size[1]) + '_strpp{}'.format(args.stride) + suff + '.npy'))
                mask_data = np.load(join(self.mask_dir,
                                         mask_pref + id + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                             args.patch_size[1]) + '_strpp{}'.format(args.stride) + mask_suff + '.npy'))
                if self.temporal:

                    # load per-time images (current already in memory as image_data; same for mask_data if you have it)
                    img_pa = [image_data if tid == id else np.load(img_path(tid)).astype(np.float32) for tid in
                              tids]  # each (X,Y,H,W)
                    msk_pa = [mask_data if tid == id else np.load(msk_path(tid)).astype(np.float32) for tid in
                              tids]  # each (X,Y,H,W)

                    # # keep only patch coords valid for **all** times
                    # rc_valid = [(x, y) for (x, y) in rc if
                    #             all(0 <= x < p.shape[0] and 0 <= y < p.shape[1] for p in img_pa)]
                    # ---- UNION of non-zero indices across ALL tids (current + prevs), clipped to common grid ----
                    ny_common = min(p.shape[0] for p in img_pa)
                    nx_common = min(p.shape[1] for p in img_pa)
                    H, W = args.patch_size

                    img_pa = [p[:ny_common, :nx_common, :H, :W] for p in img_pa]
                    msk_pa = [p[:ny_common, :nx_common, :H, :W] for p in msk_pa]

                    rc_union, seen = [], set()
                    for tid_u in tids:
                        for xy in nonz_patches_dict.get(tid_u, []):
                            x, y = int(xy[0]), int(xy[1])
                            if 0 <= x < ny_common and 0 <= y < nx_common and (x, y) not in seen:
                                rc_union.append((x, y))
                                seen.add((x, y))

                    # Start from unioned positives
                    rc_valid = rc_union

                    # ---------- Ring negatives (extra all-zero patches near positives) ----------
                    from scipy.ndimage import binary_dilation


                    if getattr(args, 'add_ring_negatives', True) and dset == 'train':
                        # 1) patch-level union mask grid across time (True where any pixel > 0 in any t)
                        #    msk_pa: list of (ny,nx,H,W)
                        union_grid = np.zeros((ny_common, nx_common), dtype=bool)
                        for t in range(len(msk_pa)):
                            union_grid |= (msk_pa[t] > 0).any(axis=(-2, -1))

                        # 2) binary grid of positive patches (from rc_valid)
                        pos_grid = np.zeros_like(union_grid, dtype=bool)
                        for (x, y) in rc_valid:
                            pos_grid[x, y] = True

                        # 3) build an annulus around positives: (inner, outer) radii in PATCH units
                        inner = int(getattr(args, 'neg_ring_inner', 1))  # exclude immediate neighbors
                        outer = int(getattr(args, 'neg_ring_outer', 3))  # how far to search
                        k_inner = np.ones((2 * inner + 1, 2 * inner + 1), dtype=bool) if inner > 0 else np.ones((1, 1),
                                                                                                                dtype=bool)
                        k_outer = np.ones((2 * outer + 1, 2 * outer + 1), dtype=bool)

                        ring_outer = binary_dilation(pos_grid, structure=k_outer)
                        ring_inner = binary_dilation(pos_grid, structure=k_inner) if inner > 0 else pos_grid
                        ring = ring_outer & (~ring_inner)

                        # 4) keep only ring cells that are truly ZERO in the union mask over time
                        cand_grid = ring & (~union_grid)

                        # 5) turn candidates into indices and sample
                        cand_idx = list(zip(*np.where(cand_grid)))
                        rng = np.random.default_rng(getattr(args, 'seed', None))
                        neg_per_pos = float(getattr(args, 'neg_per_pos', 1.0))  # e.g., 1 negative per positive
                        k = min(len(cand_idx), int(neg_per_pos * len(rc_valid)))
                        rc_neg = rng.choice(cand_idx, size=k, replace=False).tolist()

                        # 6) final coords = positives + sampled ring negatives
                        rc_use = rc_valid + rc_neg
                    else:
                        rc_use = rc_valid
                    # ---------------------------------------------------------------------------

                    # Build (T, N, H, W) and unioned mask (N, H, W) as before
                    patches_per_t = [np.stack([p[x, y] for (x, y) in rc_use], axis=0) for p in img_pa]  # each (N,H,W)
                    image_data = np.stack(patches_per_t, axis=0).astype(np.float32)  # (T,N,H,W)

                    masks_per_t = [np.stack([p[x, y] for (x, y) in rc_use], axis=0) for p in msk_pa]  # each (N,H,W)
                    # EDIT 2026-07-28: was `(stack > 0).any(axis=0)` — the union over the
                    # temporal stack. Now the newest timestep only. CHANGELOG.md #7
                    mask_data = temporal_target_mask(
                        masks_per_t, union=getattr(args, 'union_temporal_mask', False))  # (N,H,W)

                    # (optional) your no-data treatment stays unchanged
                    if args.treat_nodata_regions:
                        tol = 1e-9
                        V = (np.abs(image_data) > tol).astype(np.float32)
                        if np.isnan(image_data).any():
                            V = V & (~np.isnan(image_data))
                            image_data = np.nan_to_num(image_data, nan=0.0)
                        image_data = np.concatenate([image_data, V], axis=0).astype(np.float32)  # -> (2T,N,H,W)
                        # BLOCK layout: channels [0,T) are the images (chronological,
                        # oldest->newest) and [T,2T) the matching validity maps. Anything
                        # un-flattening this to (T,2,H,W) must account for that.
                        # EDIT 2026-07-28: record T so preprocess() can leave the validity
                        # channels strictly {0,1}. CHANGELOG.md #8
                        self.n_value_channels = len(tids)

                elif args.nonoverlap_tr_tst:
                    if dset == 'train':
                        nz_patches, nz_masks, nz_indices = [], [], []
                        for ii in range(image_data.shape[0]):
                            for jj in range(image_data.shape[1]):
                                if np.any(mask_data[ii, jj] > 0):
                                    nz_patches.append(image_data[ii, jj])
                                    nz_masks.append(mask_data[ii, jj])
                                    nz_indices.append([ii, jj])

                        test_num = int(args.test / 100 * len(nz_indices))
                        test_patch_nums = random.sample(range(len(nz_indices)), test_num)
                        test_patch_nums = sorted(test_patch_nums)
                        test_patches = [nz_patches[p] for p in test_patch_nums]
                        test_mask_patches = [nz_masks[p] for p in test_patch_nums]
                        test_indices = [nz_indices[p] for p in test_patch_nums]

                        train_patches, tr_mask_patches = [], []
                        for n, ind in enumerate(nz_indices):

                            if not (any((abs(item[0] - ind[0]) < 2 and abs(item[1] - ind[1]) < 2) for item in
                                        test_indices)):  ##only for stride 2!!
                                train_patches.append(nz_patches[n])
                                tr_mask_patches.append(nz_masks[n])

                        image_data, mask_data = np.array(train_patches), np.array(tr_mask_patches)
                        SubsiDataset.test_dataset_for_nonoverlap_split.append(np.array(test_patches))
                        SubsiDataset.test_mask_for_nonoverlap_split.append(np.array(test_mask_patches))
                    elif dset == 'test':
                        image_data, mask_data = SubsiDataset.test_dataset_for_nonoverlap_split[i], \
                        SubsiDataset.test_mask_for_nonoverlap_split[i]

                elif not args.nonz_only or args.add_nulls_to_train:
                    image_data = image_data.reshape(-1, image_data.shape[2], image_data.shape[3])
                    mask_data = mask_data.reshape(-1, mask_data.shape[2], mask_data.shape[3])

                    if args.nonz_only:
                        image_patches = []
                        mask_patches = []

                        for j in range(image_data.shape[0]):
                            image_patch = image_data[j]
                            mask_patch = mask_data[j]
                            add_null_patch = False
                            if random.choice([True, False]) and not np.any(mask_patch > 0):
                                count = np.sum(image_patch == 0) < 1000
                                cons_zeros = has_consecutive_zeros(image_patch)
                                if count and cons_zeros:
                                    add_null_patch = True

                            if np.any(mask_patch > 0) or (add_null_patch):
                                image_patches.append(image_patch)
                                mask_patches.append(mask_patch)

                        print('number of patches with nulls: {}'.format(len(image_patches)))
                        print('from {} patches'.format(image_data.shape[0]))
                        image_data = np.array(image_patches)
                        mask_data = np.array(mask_patches)


            else:

                # shared stride & threshold (based on current id)
                dy = coord_dict[id]["dy"]  # constant > 0
                stride = math.floor(args.patch_size[0] / 2) * dy
                thresh_line = math.floor((coord_dict[id]["north"] - args.thresh_lat) / stride)

                # clamp thresh_line to this id's ny just for safety
                ref_img = np.load(join(self.image_dir,
                                       f"{pref}{id}_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}.npy"),
                                  mmap_mode='r')
                ny_id = ref_img.shape[0]
                thresh_line = max(0, min(thresh_line, ny_id))

                def _slice_first_dim(arr, tl, train_split: bool):
                    """Clamp slice by this array's ny so tids with smaller ny don't break."""
                    ny = arr.shape[0]
                    if train_split:
                        hi = min(tl, ny)  # [:tl]  -> [:min(tl, ny)]
                        return arr[:hi]
                    else:
                        lo = min(tl, ny)  # [tl:]  -> [min(tl, ny):]
                        return arr[lo:]

                # ---------------- your logic with tiny changes ----------------
                if dset == 'train':
                    image_data = np.load(join(self.image_dir,
                                              f"{pref}{id}_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}.npy"))[
                                 :thresh_line]
                    mask_data = np.load(join(self.mask_dir,
                                             f"{mask_pref}{id}_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}.npy"))[
                                :thresh_line]
                    if self.temporal:
                        img_pa = [image_data if tid == id else
                                  _slice_first_dim(np.load(img_path(tid)), thresh_line, True).astype(np.float32)
                                  for tid in tids]
                        msk_pa = [mask_data if tid == id else
                                  _slice_first_dim(np.load(msk_path(tid)), thresh_line, True).astype(np.float32)
                                  for tid in tids]
                else:
                    image_data = np.load(join(self.image_dir,
                                              f"{pref}{id}_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}.npy"))[
                                 thresh_line:]
                    mask_data = np.load(join(self.mask_dir,
                                             f"{mask_pref}{id}_H{args.patch_size[0]}_W{args.patch_size[1]}_strpp{args.stride}.npy"))[
                                thresh_line:]
                    if self.temporal:
                        img_pa = [image_data if tid == id else
                                  _slice_first_dim(np.load(img_path(tid)), thresh_line, False).astype(np.float32)
                                  for tid in tids]
                        msk_pa = [mask_data if tid == id else
                                  _slice_first_dim(np.load(msk_path(tid)), thresh_line, False).astype(np.float32)
                                  for tid in tids]

                # keep your reshape; temporal branch will reshape/stack later anyway
                image_data = image_data.reshape(-1, image_data.shape[2], image_data.shape[3])
                mask_data = mask_data.reshape(-1, mask_data.shape[2], mask_data.shape[3])
                if self.temporal:

                    # # keep only patch coords valid for **all** times
                    # rc_valid = [(x, y) for (x, y) in rc if
                    #             all(0 <= x < p.shape[0] and 0 <= y < p.shape[1] for p in img_pa)]
                    # ---- UNION of non-zero indices across ALL tids (current + prevs), clipped to common grid ----
                    ny_common = min(p.shape[0] for p in img_pa)
                    nx_common = min(p.shape[1] for p in img_pa)
                    H, W = args.patch_size

                    img_pa = [p[:ny_common, :nx_common, :H, :W] for p in img_pa]
                    msk_pa = [p[:ny_common, :nx_common, :H, :W] for p in msk_pa]

                    if dset == 'train':
                        row_offset = {tid: 0 for tid in tids}
                    else:
                        # we need each tid's original ny to clamp min(thresh_line, ny_tid)
                        def _ny_of_tid(tid):
                            return np.load(img_path(tid), mmap_mode='r').shape[0]  # (ny, nx, H, W)

                        row_offset = {tid: min(thresh_line, _ny_of_tid(tid)) for tid in tids}

                    # ... after you computed ny_common, nx_common and before rc_union loop ...

                    rc_union, seen = [], set()
                    for tid_u in tids:
                        off = row_offset[tid_u]
                        for xy in nonz_patches_dict.get(tid_u, []):
                            x_glob, y = int(xy[0]), int(xy[1])
                            x_loc = x_glob - off  # <-- convert global row to local (post-slice) row
                            if 0 <= x_loc < ny_common and 0 <= y < nx_common and (x_loc, y) not in seen:
                                rc_union.append((x_loc, y))
                                seen.add((x_loc, y))

                    # Start from unioned positives
                    rc_valid = rc_union
                    if not rc_valid:
                        continue

                    # ---------- Ring negatives (extra all-zero patches near positives) ----------
                    from scipy.ndimage import binary_dilation

                    if getattr(args, 'add_ring_negatives', True) and dset == 'train':
                        # 1) patch-level union mask grid across time (True where any pixel > 0 in any t)
                        #    msk_pa: list of (ny,nx,H,W)
                        union_grid = np.zeros((ny_common, nx_common), dtype=bool)
                        for t in range(len(msk_pa)):
                            union_grid |= (msk_pa[t] > 0).any(axis=(-2, -1))

                        # 2) binary grid of positive patches (from rc_valid)
                        pos_grid = np.zeros_like(union_grid, dtype=bool)
                        for (x, y) in rc_valid:
                            pos_grid[x, y] = True

                        # 3) build an annulus around positives: (inner, outer) radii in PATCH units
                        inner = int(getattr(args, 'neg_ring_inner', 1))  # exclude immediate neighbors
                        outer = int(getattr(args, 'neg_ring_outer', 3))  # how far to search
                        k_inner = np.ones((2 * inner + 1, 2 * inner + 1), dtype=bool) if inner > 0 else np.ones((1, 1),
                                                                                                                dtype=bool)
                        k_outer = np.ones((2 * outer + 1, 2 * outer + 1), dtype=bool)

                        ring_outer = binary_dilation(pos_grid, structure=k_outer)
                        ring_inner = binary_dilation(pos_grid, structure=k_inner) if inner > 0 else pos_grid
                        ring = ring_outer & (~ring_inner)

                        # 4) keep only ring cells that are truly ZERO in the union mask over time
                        cand_grid = ring & (~union_grid)

                        # 5) turn candidates into indices and sample
                        cand_idx = list(zip(*np.where(cand_grid)))
                        rng = np.random.default_rng(getattr(args, 'seed', None))
                        neg_per_pos = float(getattr(args, 'neg_per_pos', 1.0))  # e.g., 1 negative per positive
                        k = min(len(cand_idx), int(neg_per_pos * len(rc_valid)))
                        rc_neg = rng.choice(cand_idx, size=k, replace=False).tolist()

                        # 6) final coords = positives + sampled ring negatives
                        rc_use = rc_valid + rc_neg
                    else:
                        rc_use = rc_valid
                    # ---------------------------------------------------------------------------

                    # Build (T, N, H, W) and unioned mask (N, H, W) as before
                    patches_per_t = [np.stack([p[x, y] for (x, y) in rc_use], axis=0) for p in img_pa]  # each (N,H,W)
                    image_data = np.stack(patches_per_t, axis=0).astype(np.float32)  # (T,N,H,W)

                    masks_per_t = [np.stack([p[x, y] for (x, y) in rc_use], axis=0) for p in msk_pa]  # each (N,H,W)
                    # EDIT 2026-07-28: was `(stack > 0).any(axis=0)` — the union over the
                    # temporal stack. Now the newest timestep only. CHANGELOG.md #7
                    mask_data = temporal_target_mask(
                        masks_per_t, union=getattr(args, 'union_temporal_mask', False))  # (N,H,W)

                elif args.nonz_only:
                    mask_nz, image_nz = [], []
                    for p in range(image_data.shape[0]):
                        if np.any(mask_data[p] > 0):
                            mask_nz.append(mask_data[p])
                            image_nz.append(image_data[p])
                    image_data = np.array(image_nz)
                    mask_data = np.array(mask_nz)
                    logging.info('Sptial partition: Created {} patches for {} set'.format(image_data.shape[0], dset))

            if not self.temporal:
                image_data = np.expand_dims(image_data, axis=0)
            mask_data = np.expand_dims(mask_data, axis=0)
            len_examples = image_data.shape[1]

            self.image_data.append(image_data)
            self.mask_data.append(mask_data)
            self.index_map.extend([[i, j] for j in range(len_examples)])
            logging.info('loaded patch for intf {}'.format(id))

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir,
                               mask_suffix='_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                                   args.patch_size[1]) + '_strpp{}'.format(args.stride) + mask_suff + '.npy'),
                       self.ids),
                total=len(self.ids)
            ))
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

        # self.do_augmentations = (args.augment and dset == 'train')
        # if self.do_augmentations:
        #     self.augment = A.Compose([
        #         A.Rotate(limit=30, p=0.4),
        #         A.ElasticTransform(p=0.3),
        #         A.RandomBrightnessContrast(p=0.3),
        #         A.GaussNoise(var_limit=(0.1, 0.2), p=0.3),
        #         A.HorizontalFlip(p=0.3),
        #         A.GridDistortion(p=0.2),
        #     ], additional_targets={'mask': 'mask'})

        logging.info(f'Created dataset: from {len(self.ids)} interferograms {len(self.index_map)} patches!!!')

    def __len__(self):
        return len(self.index_map)

    @staticmethod
    def preprocess(mask_values, img, is_mask, n_value_channels=None):
        """Normalise a patch.

        `n_value_channels`: index at which validity channels begin, or None. Channels at
        or after it are binary validity maps and are left EXACTLY as they are.

        EDIT 2026-07-28: previously every channel went through the normalisation below,
        including the validity maps appended by --treat_nodata_regions. Their min/max are
        0/1, so they took the `else` branch and had their zeros rewritten to 0.5 — which
        meant `segmentation_loss` (train_sinkholes_unet.py:124) computed `V_any` values of
        0.5 rather than 0, and the no-data masking never actually masked anything.
        CHANGELOG.md #8
        """

        if is_mask:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        out = img.copy()
        tol = 1e-1
        # EDIT 2026-07-28: stop at the validity block; those channels must stay {0,1}.
        # CHANGELOG.md #8
        n_normalise = out.shape[0] if n_value_channels is None else int(n_value_channels)
        for c in range(n_normalise):
            mn = float(np.nanmin(out[c]));
            mx = float(np.nanmax(out[c]))
            if (mn < -tol) or (mx > 1.0 + tol):
                out[c] = (out[c] + np.pi) / (2 * np.pi)
            else:
                zmask = (out[c] == 0.0)  # only exact zeros
                if zmask.any():
                    out[c][zmask] = 0.5
        return out



    def __getitem__(self, sample):
        intf_idx, patch_idx = self.index_map[sample]
        if getattr(self, "temporal", False):
            # image: [T, N, H, W] -> pick patch -> [T, H, W]
            img = self.image_data[intf_idx][:, patch_idx].astype(np.float32)
            # mask:  [1, N, H, W] -> pick patch & drop singleton -> [H, W]
            msk = self.mask_data[intf_idx][0, patch_idx].astype(np.float32)
            # plot_temporal_patch_with_mask(img, msk, cmap='jet')
            # plot_temporal_patch_with_mask(
            #     img,
            #     msk
            # )
        else:
            # image/mask stored as [1, N, H, W] -> pick channel 0 & patch -> [H, W]
            img = self.image_data[intf_idx][0, patch_idx].astype(np.float32)
            msk = self.mask_data[intf_idx][0, patch_idx].astype(np.float32)

        # Preprocess
        # EDIT 2026-07-28: pass the validity-block offset so those channels are left
        # strictly {0,1}. getattr() keeps datasets pickled before this change loadable.
        # CHANGELOG.md #8
        img = self.preprocess(self.mask_values, img, 0,
                              n_value_channels=getattr(self, 'n_value_channels', None))  # keeps shape; (T,H,W) or (H,W)
        msk = self.preprocess(self.mask_values, msk, 1)  # -> (H,W) labels
        # if self.dataset != 'train':
        #     T = img.shape[0];
        #     fig, ax = plt.subplots(1, T + 1, figsize=(2 * (T + 1), 2))
        #     for i in range(T): ax[i].imshow(img[i], cmap='jet'); ax[i].axis('off')
        #     ax[-1].imshow(msk);
        #     ax[-1].axis('off');
        #     plt.tight_layout();
        #     plt.show()

        # # Augment (Albumentations expects channels-last)
        # if self.do_augmentations:
        #     if getattr(self, "temporal", False):
        #         aug = self.augment(image=img.transpose(1, 2, 0), mask=msk)  # (H,W,T)
        #         img = aug["image"].transpose(2, 0, 1)  # back to (T,H,W)
        #         msk = aug["mask"]
        #     else:
        #         aug = self.augment(image=img, mask=msk)  # both (H,W)
        #         img = aug["image"]
        #         msk = aug["mask"]

        # Sanity: spatial sizes must match
        assert img.shape[-2:] == msk.shape[-2:], \
            f"Spatial size mismatch: image {img.shape} vs mask {msk.shape}"

        # To tensors
        if getattr(self, "temporal", False):
            # (T,H,W)
            img_t = torch.as_tensor(img.copy()).float().contiguous()
        else:
            # (1,H,W)
            img_t = torch.as_tensor(img.copy()).unsqueeze(0).float().contiguous()

        msk_t = torch.as_tensor(msk.copy()).long().contiguous()

        return {"image": img_t, "mask": msk_t}


if __name__ == '__main__':
    intrfrgrm_list = None  # ['20230802T153919_20230813T153921']
    dataset = SubsiDataset('./data_patches', './mask_patches')
    train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True)
    print('train_loaderlength: ', len(train_loader))
    for batch, (input, targ) in enumerate(train_loader):
        print(batch)

