# Running this project locally — a hands-on walkthrough

A tutorial for exploring the sinkhole pipeline on a Mac with a **partial, already-patched**
download, without the raw interferograms or the ground-truth shapefile.

Companion to [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) (what the code does and where the bugs
are) and [`PIPELINE_MANUAL.md`](./PIPELINE_MANUAL.md) (the full command reference for WEXAC).
This document is the *local* path: what you can run today, what you cannot, and the exact
commands, in order.

Every command below was executed on the machine this was written for
(macOS, Python 3.12.3 via pyenv, torch 2.10.0, no CUDA). Timings and outputs are real.

```bash
export PY=/Users/guypi/.pyenv/versions/3.12.3/bin/python
cd /Users/guypi/Projects/sinkholes     # paths below are written relative to the repo root
```

The committed data assets (`intf_coord.json`, `lidar_mask_polygs.shp`) now live in `assets/` and
are opened by absolute path, so `get_intf_info.py`, `check_patches.py` and `test_full_intf.py`
work from any working directory. The relative paths in this walkthrough still assume you are at
the repo root.

---

## 1. Where your data sits in the pipeline

The full pipeline has five stages. Your download starts you at stage 2.

```
  (0) .ers headers ────────────► intf_coord.json                    ✅ in the repo
                                                                       (716 interferograms)

  (1) .unw rasters  ──┐
      sub_*.shp GT  ──┴────────► data_patches_<intf>_*.npy          ✅ DOWNLOADED (30 intfs)
      prepare_intrfrgrm         mask_patches_<intf>_*.npy           ✅ DOWNLOADED (30 intfs)
      _pathches.py              data_patches_nonz_<intf>_*.npy      ❌ NOT downloaded
                                mask_patches_nonz_<intf>_*.npy      ❌ NOT downloaded
                                nonz_indices.json                   ✅ DOWNLOADED

  (1b) check_patches.py ───────► nonz_num inside intf_coord.json    ⚠️  STALE (see §4)

  (2) train_sinkholes_unet.py ─► checkpoints/*.pth                  ✅ RUNS (§6)

  (3) test.py / test_full_intf.py ─► metrics, .npy, polygons        ✅ RUNS (§7)

  (4) predict_new_intf_vA.py ──► shapefiles from raw .unw           ❌ needs .unw
```

**You have the products of stage 1, minus the `nonz` subsets.** Stages 0 and 1 are done and do
not need to be repeated. Stage 1b needs correcting. Stages 2 and 3 run locally today.

### What the patch values already are

The `.unw` products in this dataset are **already scaled to `[0, 1]`** — verified across a full
grid:

```
min 0.0000   max 0.9997   NaNs 0   exact zeros 17.2%
```

`prepare_intrfrgrm_pathches.py:168` reads them with `np.fromfile` and applies **no scaling**
before patchify, so what is on your disk is what came out of the InSAR processor. This matters
because `SubsiDataset.preprocess()` (`sinkholes_data_loading.py:541-549`) branches on the range:

```python
if (mn < -tol) or (mx > 1.0 + tol):
    out[c] = (out[c] + np.pi) / (2 * np.pi)   # radians -> [0,1]     ← never taken here
else:
    out[c][out[c] == 0.0] = 0.5               # exact zeros = no-data ← always taken here
```

So for your data the **only** preprocessing at load time is `0 → 0.5`. The `0` value is the
no-data code (17.2 % of pixels), and `0.5` is the "neutral phase" the network is shown instead.
`test_full_intf.py:413-432` reproduces the same conditional logic, so training and by-intf
evaluation agree.

> ⚠️ `predict_new_intf_vA.py:254` does **not**: it applies `(d + np.pi) / (2*np.pi)`
> unconditionally. On `[0,1]` input that squashes everything into `[0.500, 0.659]` and never
> inserts the `0.5` no-data code. Any polygons that script produced on this dataset were computed
> from a badly compressed input range. Treat its output as unreliable until that line is fixed.

---

## 2. What you have — inventory

```bash
$PY - <<'PY'
import numpy as np, json, os, re
ids = sorted(re.match(r'data_patches_(\d{8}_\d{8})_', f).group(1)
             for f in os.listdir('data/patches/images'))
print(f'{len(ids)} interferograms: {ids[0]} .. {ids[-1]}')
meta = json.load(open('intf_coord.json'))
frames = {}
for i in ids: frames[meta[i]['frame']] = frames.get(meta[i]['frame'], 0) + 1
print('frames:', frames)
a = np.load(f'data/patches/images/data_patches_{ids[0]}_H200_W100_strpp2.npy', mmap_mode='r')
m = np.load(f'data/patches/masks/mask_patches_{ids[0]}_H200_W100_strpp2.npy',  mmap_mode='r')
print(f'image grid {a.shape} {a.dtype} | mask grid {m.shape} {m.dtype}')
PY
```

| | |
|---|---|
| interferograms | **30**, `20190113_20190124` … `20190730_20190810`, a **contiguous** 11-day series |
| frames | 15 North + 15 South |
| image grids | `(ny, 89, 200, 100)` `float32`, `ny ∈ {191…197}`, ~1.37 GB each, **38 GB** total |
| mask grids | same shape, `uint8`, values `{0, 1}`, **9.6 GB** total |
| `nonz` subsets | **absent** — this is the one real gap |
| `nonz_indices.json` | present, and **exactly** matches the downloaded grids (§4) |

Because the series is contiguous, even **temporal** training works: with `--k_prevs 2`,
**24 of 30** interferograms have both required predecessors on disk.

```bash
$PY - <<'PY' 2>/dev/null
import json, os, re, sys; sys.path.insert(0, '.')
from get_intf_info import find_11day_sequences
meta  = json.load(open('intf_coord.json'))
local = sorted(re.match(r'data_patches_(\d{8}_\d{8})_', f).group(1)
               for f in os.listdir('data/patches/images'))
for k in (0, 1, 2):
    chains, valid = find_11day_sequences(meta, k_prev=k, restrict_to=local)
    on_disk = [v for v in valid if all(p in local for p in chains[v]['prevs'])]
    print(f'k_prevs={k}: {len(valid)}/30 pass the chain filter, {len(on_disk)} have all prevs on disk')
PY
```

### What you cannot run

| Blocked | Needs |
|---|---|
| `prepare_intrfrgrm_pathches.py` | `*.unw` **and** `sub_20231001.shp` — it calls `gpd.read_file()` first thing, so it fails even for a subset |
| `prepare_intf_coord_dict.py` | `*.ers` headers — moot, `intf_coord.json` is committed |
| `clean_patches.py` | the full patch tree; optional stage |
| `predict_new_intf*.py` | raw `*.unw` — it reads rasters, not patches |
| `lidar.py` | per-year LiDAR shapefiles — moot, the merged `lidar_mask_polygs.shp` is committed |

---

## 3. Make the data training-compatible

### The gap

`SubsiDataset` computes its label vocabulary through `unique_mask_values`
(`sinkholes_data_loading.py:81-90`):

```python
mask_file = list(mask_dir.glob('mask_patches_nonz_' + idx + mask_suffix))[0]
```

The `nonz_` prefix is **hardcoded** — it does not follow `--nonz_only`, `--partition_mode` or
`--add_temporal`. With your download the glob returns `[]` and you get `IndexError` at line 82.
Symlinking a `nonz`-named alias to a full grid does not help either: lines 84-90 accept only 2-D
or 3-D arrays, and a full grid is 4-D → `ValueError: ... found 4`.

So **every** mode needs real `nonz` files. The good news: they are pure re-indexing of data you
already have. `patchify` built them by selecting the grid cells listed in `nonz_indices.json`
(`prepare_intrfrgrm_pathches.py:53-57`) — no raw data required.

### The fix

`prepare_local_subset.py` derives them, and fixes the metadata at the same time:

```bash
$PY src/data_prep/prepare_local_subset.py
```

It opens the download read-only and writes a new tree:

```
data/patches/train_ready/
├── data_patches_H200_W100_strpp2_11days_Aligned/
│   ├── data_patches_nonz_<intf>_H200_W100_strpp2.npy   (N,200,100) float32   ← derived
│   ├── data_patches_<intf>_H200_W100_strpp2.npy        symlink to the download
│   └── nonz_indices.json                               copy (--add_temporal reads it here)
└── mask_patches_H200_W100_strpp2_11days_Aligned/
    ├── mask_patches_nonz_<intf>_H200_W100_strpp2.npy   (N,200,100) uint8     ← derived
    └── mask_patches_<intf>_H200_W100_strpp2.npy        symlink to the download
```

Result: **9825 positive patches, 982 MB**, 187–528 per interferogram. Zero disagreements between
`nonz_indices.json` and the mask grids.

The directory names are not cosmetic. `train_sinkholes_unet.py:106-116` *builds* the leaf name by
string concatenation and exposes no flag for it:

```
<--patches_dir> + "data_patches_H200_W100_strpp2" + ("_11days"|"_all") + "_Aligned"
```

`_11days` comes from `--train_on_11d_diff` (default `'True'`); `_Aligned` is forced by an
`or True` at line 112. `test_full_intf.py:338-347` builds the identical name from
`--input_patch_dir` + `--days_diff`. That is why one tree serves both.

Symlinks keep the full grids available for `--add_temporal` and `--partition_mode spatial`
without duplicating 38 GB.

---

## 4. Fix the metadata (`nonz_num`)

`intf_coord.json`'s `nonz_num` was written by a **different** patchify run than your download:

| interferogram | actual positives | `nonz_indices.json` | `intf_coord.json` |
|---|---|---|---|
| 20190113_20190124 | 221 | 221 ✅ | 240 ❌ |
| 20190204_20190215 | 87 | 87 ✅ | 96 ❌ |
| 20190205_20190216 | 297 | 297 ✅ | 262 ❌ |
| 20190216_20190227 | 499 | 499 ✅ | 472 ❌ |

`nonz_indices.json` is the trustworthy one — set equality with the grids, zero symmetric
difference. The dictionary is wrong, which corrupts `--train_with_nonz_th` and any per-intf
statistics you compute from it.

You cannot run `check_patches.py` to fix it: it rewrites `intf_coord.json` **in the working
directory** (clobbering the committed copy), and it would only see the 30 intfs you have.
`prepare_local_subset.py` writes a separate file instead:

```
data/metadata/intf_coord_local.json
```

with `nonz_num` recounted for your 30 interferograms and `'none'` for the other 686. Since
training drops any intf whose `nonz_num == 'none'` (`train_sinkholes_unet.py:139-142`), that one
file **automatically restricts every run to exactly what you downloaded**. Pass it as
`--intf_dict_path` and the committed `intf_coord.json` stays untouched.

To re-derive the counts from the masks themselves rather than trusting the JSON (slow — reads all
9.6 GB):

```bash
$PY src/data_prep/prepare_local_subset.py --verify_full --force
```

---

## 5. Inspect one sample end to end

Before training, look at what a sample actually is.

### 5.1 One image, one mask

```bash
$PY - <<'PY'
import numpy as np, json
iid = '20190204_20190215'
img = np.load(f'data/patches/images/data_patches_{iid}_H200_W100_strpp2.npy', mmap_mode='r')
msk = np.load(f'data/patches/masks/mask_patches_{iid}_H200_W100_strpp2.npy',  mmap_mode='r')
print('image grid', img.shape, img.dtype, f'{img.nbytes/1e9:.2f} GB')
print('mask  grid', msk.shape, msk.dtype)

i, j = json.load(open('data/metadata/nonz_indices.json'))[iid][0]
p, m = np.asarray(img[i, j]), np.asarray(msk[i, j])
print(f'\npatch ({i},{j})')
print('  image  min/max/mean %.4f / %.4f / %.4f' % (p.min(), p.max(), p.mean()))
print('  NaNs', int(np.isnan(p).sum()), '| exact zeros (no-data)', int((p == 0).sum()))
print('  mask   uniques', np.unique(m), '| positive px', int((m > 0).sum()))
PY
```

`mmap_mode='r'` reads ~80 KB, not 1.37 GB. Expect `(192, 89, 200, 100) float32`, patch `(4,46)`,
`0.0000 / 0.9946 / 0.6279`, 0 NaNs, 455 zeros, mask uniques `[0 1]`, 439 positive pixels.

### 5.2 How a grid index maps to geography

Verified against all four interferograms tested, both frames:

```
stride  Sy = H // strides_per_patch = 100      Sx = W // strides_per_patch = 50

patch (i, j) covers   rows [ i*100 , i*100+200 )   cols [ j*50 , j*50+100 )
                      in the ALIGNED, x-cropped array

lon = x_star + (j*50 + col) * dx        lat = y_star - (i*100 + row) * dy
x_star, y_star = (35.37, 31.79) North  |  (35.32, 31.44) South       dx = dy = 2.777e-05
```

Grid width is always `(4500-100)//50 + 1 = 89`, confirming the `nx=4500` column crop. Grid height
follows from the frame-alignment crop — this reproduces `ny` exactly:

```bash
$PY - <<'PY'
import json, numpy as np
c = json.load(open('intf_coord.json'))
for iid in ['20190113_20190124','20190204_20190215','20190205_20190216','20190216_20190227']:
    m = c[iid]; y_star = 31.79 if m['frame'] == 'North' else 31.44
    cropped = round((m['north'] - y_star) / m['dy'])
    ny_pred = (m['nlines'] - cropped - 200) // 100 + 1
    ny_act  = np.load(f'data/patches/images/data_patches_{iid}_H200_W100_strpp2.npy',
                      mmap_mode='r').shape[0]
    print(f"{iid} {m['frame']:5s} rows_cropped={cropped:4d} "
          f"ny_pred={ny_pred} ny_actual={ny_act} {'OK' if ny_pred == ny_act else 'MISMATCH'}")
PY
```

**Two index conventions you will keep meeting:**

- `nonz_indices.json[id]` → `[[i, j], …]`, *grid* coordinates, row-major, in the order `patchify`
  emitted them.
- `SubsiDataset.index_map[k]` → `[intf_idx, flat]`. With `--nonz_only True` (the `nonz` files)
  `flat` is the position in `nonz_indices[id]`. With `--nonz_only False` the reshape at
  `sinkholes_data_loading.py:307` is row-major, so `flat = i*89 + j`.

### 5.3 Through the repo Dataset, one batch, one forward pass

```bash
$PY - <<'PY'
import matplotlib; matplotlib.use('Agg')   # BEFORE repo imports — see §9
from types import SimpleNamespace
from torch.utils.data import DataLoader
from sinkholes_data_loading import SubsiDataset
from unet import UNet
import torch

args = SimpleNamespace(
    patch_size=[200, 100], stride=2, nonz_only=True,
    partition_mode='random_by_intf', add_temporal=False, add_nulls_to_train=False,
    nonoverlap_tr_tst=False, use_cleaned_patches=False, treat_nodata_regions=False,
    add_ring_negatives=False, test=10.0, thresh_lat=31.4, seed=0,
    intf_dict_path='data/metadata/intf_coord_local.json')

root = 'data/patches/train_ready'
ds = SubsiDataset(args,
                  f'{root}/data_patches_H200_W100_strpp2_11days_Aligned',
                  f'{root}/mask_patches_H200_W100_strpp2_11days_Aligned',
                  intrfrgrm_list=['20190204_20190215'], dset='train')
print('len', len(ds), '| mask_values', ds.mask_values, '| index_map[:3]', ds.index_map[:3])

s = ds[0]
print("ds[0] image", tuple(s['image'].shape), s['image'].dtype,
      f"[{s['image'].min():.4f}, {s['image'].max():.4f}]  <- min>0: zeros became 0.5")
print("ds[0] mask ", tuple(s['mask'].shape), s['mask'].dtype,
      torch.unique(s['mask']).tolist())

batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))
print('batch image', tuple(batch['image'].shape), '| mask', tuple(batch['mask'].shape))

net = UNet(n_channels=1, n_classes=1, bilinear=False).eval()
with torch.no_grad():
    logits = net(batch['image'])
print('forward', tuple(batch['image'].shape), '->', tuple(logits.shape),
      f'logits [{logits.min():.3f}, {logits.max():.3f}]')
PY
```

This exercises the real code path: `unique_mask_values`, `preprocess`, `__getitem__`, collation,
and `UNet.forward`.

---

## 6. Train

### 6.1 Smoke test — 1 epoch, 3 interferograms, ~33 s

```bash
LOCAL_ENVIRONMENT=1 $PY src/training/train_sinkholes_unet.py \
  --epochs 1 \
  --partition_mode random_by_intf \
  --patches_dir data/patches/train_ready/ \
  --intf_dict_path data/metadata/intf_coord_local.json \
  --patch_size 200 100 --stride 2 \
  --nonz_only True --k_prevs 0 \
  --batch_size 8 --learning-rate 1e-5 \
  --job_name local_smoke
```

`LOCAL_ENVIRONMENT=1` discards whatever split the code computed and forces
`train=['20190204_20190215']` (87 patches), `val=['20190216_20190227']` (499),
`test=['20190205_20190216']` (297) — see §9. All three are in your download, which is why this
works out of the box.

Measured: 8.5 img/s training, 36 batch/s validation, **33 s** wall clock. Writes

```
outputs/local_smoke_<ts>/
├── checkpoints/local_smoke_<ts>checkpoint_epoch1.pth   (124 MB — one per epoch)
├── checkpoints/best.pt, last.pt                        (124 MB each)
├── results.csv                                         (one row per epoch)
├── curves.png                                          (~110 KB — CHANGELOG.md #5)
├── validation/preds/epoch_001.png                      (~250 KB per epoch — #5)
├── test_dataset_local_smoke_<ts>.pkl                   (30 MB — feeds §7)
├── logs/reporter.log
└── local_smoke_<ts>_<ts>.log
```

`curves.png` and `validation/preds/` are written by the reporter and use the Agg canvas directly,
so unlike `--plot` they are safe on macOS (§9). Add `--sample_every 0` to skip the grids, or
`--save_best_only` to drop the per-epoch 124 MB checkpoints. To rebuild `curves.png` for a run that
finished earlier: `$PY src/training/train_reporter.py outputs/<run_dir>`.

A validation Dice around `0.016` after one epoch at `lr=1e-5` is expected — the network has
barely moved off initialization. You are checking plumbing, not accuracy.

**Note the trailing slash on `--patches_dir`.** It is concatenated, not `os.path.join`ed.

### 6.2 A real local run — all 30 interferograms

Drop `LOCAL_ENVIRONMENT` so your own split is honoured:

```bash
$PY src/training/train_sinkholes_unet.py \
  --epochs 20 \
  --partition_mode random_by_intf \
  --patches_dir data/patches/train_ready/ \
  --intf_dict_path data/metadata/intf_coord_local.json \
  --patch_size 200 100 --stride 2 \
  --nonz_only True --k_prevs 0 \
  --pos_w 8 \
  --validation 10 --test 10 \
  --batch_size 8 --learning-rate 1e-5 \
  --job_name local_full
```

With `--k_prevs 0` all 30 interferograms survive the chain filter and split 24 train / 3 val /
3 test — **8207 / 680 / 938 patches**. Measured on MPS at batch 8: **112 img/s**, so ≈ **73 s per
training epoch** plus validation. On CPU the same epoch is ≈ 10 min. Start with `--epochs 1` to
confirm the split, then scale up. Watch the log line `train val and test sets have N, M, K
samples`.

Caveats specific to this configuration:

- `--k_prevs` defaults to `2` and `train_sinkholes_unet.py:157` runs the 11-day chain filter
  **unconditionally** (an `or True`), even without `--add_temporal`. That silently drops your
  first 6 interferograms. Pass **`--k_prevs 0`** for non-temporal runs to keep all 30.
- `random.shuffle` in `random_by_intf` is **not seeded** — `--seed` only reaches ring-negative
  sampling. Two runs give different splits. Record `train/val/test intfs` from the log.
- With only 30 interferograms, `--validation 10 --test 10` leaves 3 val + 3 test. `n_val == 0`
  makes the script `sys.exit(0)`.

### 6.3 Temporal training

```bash
$PY src/training/train_sinkholes_unet.py \
  --epochs 5 \
  --partition_mode random_by_intf \
  --patches_dir data/patches/train_ready/ \
  --intf_dict_path data/metadata/intf_coord_local.json \
  --add_temporal --k_prevs 2 \
  --nonz_only True \
  --batch_size 1 --learning-rate 1e-5 \
  --job_name local_temporal
```

Works on the 24 interferograms with complete chains. Input becomes `(B, 3, 200, 100)`; the target
is the **latest timestep** (it was the union of the 3 masks until `CHANGELOG.md #7` — pass
`--union_temporal_mask` for the old behaviour). `--treat_nodata_regions` doubles the channels to 6
and switches to the masked loss — note that branch **hardcodes `pos_w=8.0`** at line 388,
ignoring `--pos_w`.

This stacks the frames as input **channels**, which measured worst of the four architectures
(val dice 0.342). Prefer the ConvLSTM below.

### 6.3b ConvLSTM training (best-scoring architecture)

```bash
$PY src/training/train_sinkholes_unet.py \
  --epochs 5 \
  --partition_mode random_by_intf \
  --patches_dir data/patches/train_ready/ \
  --intf_dict_path data/metadata/intf_coord_local.json \
  --convlstm_unet --add_temporal --k_prevs 2 \
  --convlstm_hidden 256 \
  --nonz_only True \
  --batch_size 4 --learning-rate 1e-5 \
  --job_name local_convlstm
```

Same input shape, but the encoder runs per timestep and a ConvLSTM consumes the bottleneck
sequence, so temporal ordering is preserved (`CHANGELOG.md #6`). `--convlstm_unet` requires
`--add_temporal` and cannot be combined with `--attn_unet` / `--add_attn`.

Locally, drop `--convlstm_hidden` to 256 (from the 1024 default) — it dominates the parameter
count, and the full-size model costs 2.5–4× the plain U-Net per epoch. Hidden size and kernel are
stored **inside** the checkpoint, so you never re-specify them at test time.

### 6.4 Apple Silicon (MPS)

**Enabled** as of `CHANGELOG.md` #1 — `get_device()` picks MPS automatically, and
`memory_format_for()` drops `channels_last` there (MPS cannot autograd through it). You should see
`INFO - Using device mps` in the log. Nothing to pass on the command line.

Measured throughput, forward + backward + step:

| device | bs=1 | bs=4 | bs=8 |
|---|---|---|---|
| CPU | 9.5 img/s | 12.1 img/s | 13.2 img/s |
| MPS | 69.6 img/s | 113.5 img/s | **125.4 img/s** |

**Use `--batch_size 8`.** At batch 1 the speedup is 7.3×, but Python and dataloader overhead
dominate and you will not see it end to end; at batch 8 it is 9.5×.

#### What `channels_last` is

A 4-D tensor `(N, C, H, W)` has a logical shape and, separately, a physical layout in memory. The
default `contiguous_format` (NCHW) stores one channel plane at a time — all of channel 0, then all
of channel 1. `channels_last` (NHWC) stores all channels of a given pixel adjacently instead.
Shape, indexing and results are identical; only the strides change.

NHWC is the layout cuDNN's tensor-core convolution kernels want, so on modern NVIDIA GPUs it is a
significant convnet speedup — which is why this repo applied it to the model and every input batch.
It is not free everywhere: on CPU it made no measurable difference here, and on MPS the backward
pass raises `RuntimeError: view size is not compatible with input tensor's size and stride`.
Inference with NHWC inputs *does* work on MPS; only autograd breaks.

---

## 7. Test and inference

> **Architecture is read off the checkpoint** (`models/factory.py`, `CHANGELOG.md #12`). Every
> script in this section builds the matching network by itself, so U-Net, Attention U-Net and
> ConvLSTM checkpoints all work without an architecture flag. `--convlstm_unet` / `--attn_unet` /
> `--add_attn` remain as explicit overrides; one that contradicts the weights is an error, not a
> key-mismatch traceback.

### 7.1 Patch-level metrics

```bash
RUN=outputs/local_smoke_<ts>
$PY src/inference/test.py \
  --test_data_path $RUN/test_dataset_local_smoke_<ts>.pkl \
  --model $RUN/checkpoints/local_smoke_<ts>checkpoint_epoch1.pth \
  --k_prevs 0 --th 0.7 --b 5
```

Prints mean Dice, pixel precision/recall, object-level precision/recall. Verified working
(~54 s for 297 patches).

- `--k_prevs` must match training: `n_channels = k_prevs + 1`. A mismatch is now reported against
  the checkpoint's real channel count instead of failing as a key mismatch.
- `--treat_nodata_regions` **exists** as of `CHANGELOG.md #6`, so checkpoints trained with the
  doubled channel count do load. (This bullet previously said the opposite.)
- For a ConvLSTM checkpoint, add nothing — it is detected. Example:
  `$PY src/inference/test.py --test_data_path <pkl> --model <best.pt> --k_prevs 2`
- Do **not** pass `--plot` (§9). `test.py:88` hardcodes `is_local=True`, so `--plot` will always
  try to open a GUI window.
- `test.py:88` also passes `net_aux=net` unconditionally, so the auxiliary path always runs
  against the same network and `--aux_model` is a no-op. Roughly doubles runtime.

### 7.2 Full-interferogram reconstruction and polygons

This is the by-intf evaluation path, and the one that produces georeferenced shapefiles. It reads
the **full grids** — which the symlinks in `train_ready/` provide.

It expects the checkpoint under `./models/` (hardcoded at `test_full_intf.py:333`):

```bash
mkdir -p models
cp outputs/local_smoke_<ts>/checkpoints/local_smoke_<ts>checkpoint_epoch1.pth \
   models/local_smoke_epoch1.pth

$PY src/inference/test_full_intf.py \
  --model local_smoke_epoch1.pth \
  --input_patch_dir data/patches/train_ready/ \
  --intf_source intf_list --intf_list "20190205_20190216" \
  --patch_size 200 100 --data_stride 2 \
  --k_prevs 0 --recon_th 0.25 \
  --add_lidar_mask True \
  --job_name local_fullintf \
  --save_confidence
```

One interferogram is 192 × 89 = **17 088 patch predictions**, ≈ 9 min on CPU. Verified working
end to end — outputs land in `outputs/predictions/<model>/<job>_<ts>/` (**1.4 GB** for one
interferogram; `--output_dir` moves the tree, `CHANGELOG.md #14`):

| file | shape | notes |
|---|---|---|
| `<intf>_image.npy` | `(1, 19400, 4550)` `float32` | reconstructed input, `[0, 1]` |
| `<intf>_pred.npy` | `(19400, 4550)` `float32` | confidence map, `--save_confidence` |
| `<intf>_pred_th.npy` | `(19400, 4550)` `float32` | thresholded at `--recon_th` |
| `<intf>_gt.npy` | `(19400, 4550)` `float32` | rasterized ground truth |
| `polygs/<intf>_predicted_polygs.shp` | — | **EPSG:4326**, lon/lat |

The 1-epoch checkpoint produced 484 257 polygons — pure noise above `--recon_th 0.25`, which is
what an untrained network at a low operating threshold looks like. Again: plumbing, not accuracy.

> Note the canvas is `ny*step_y + patch_H = 19400` rows × `nx*step_x + patch_W = 4550` cols, but
> the last patch only reaches row 19300 / col 4500. The trailing 100 rows and 50 columns are
> never written by any patch and stay zero (verified). Harmless — they produce no polygons — but
> do not mistake the array shape for the covered extent.

Other `--intf_source` values: `test_dataset` (+`--test_dataset <file>` looked up under
`./test_data/`), `preset`, `all` (+`--year_range`).

---

## 8. Debugging — following one sample through

Suggested `.vscode/launch.json` entry:

```json
{
  "name": "train (local smoke)",
  "type": "debugpy", "request": "launch",
  "program": "${workspaceFolder}/train_sinkholes_unet.py",
  "cwd": "${workspaceFolder}",
  "python": "/Users/guypi/.pyenv/versions/3.12.3/bin/python",
  "env": { "LOCAL_ENVIRONMENT": "1" },
  "args": ["--epochs", "1", "--partition_mode", "random_by_intf",
           "--patches_dir", "data/patches/train_ready/",
           "--intf_dict_path", "data/metadata/intf_coord_local.json",
           "--nonz_only", "True", "--k_prevs", "0",
           "--batch_size", "1", "--job_name", "dbg"],
  "justMyCode": false
}
```

`justMyCode: false` lets you step into torch. Breakpoints, in execution order:

| # | Stage | File : line | Inspect |
|---|---|---|---|
| 1 | Path construction | `train_sinkholes_unet.py:106-116` | `image_dir`, `mask_dir` — the `_11days_Aligned` name the `or True` at `:112` forces. Wrong `--patches_dir` dies here. |
| 2 | File discovery | `train_sinkholes_unet.py:128-135` | `pref`, `intf_list` before the `nonz_num` filter at `:139-142`; compare `len()` after. |
| 3 | Chain filter | `train_sinkholes_unet.py:157-160` | `prev_dict`, `updated_intf_list` — runs even without `--add_temporal`. This is where `--k_prevs 2` silently costs you 6 intfs. |
| 4 | **Local override** | `train_sinkholes_unet.py:251` | `is_running_locally`, then `train_list`/`val_list`/`test_list` **after** `:254`. Confirm your CLI intfs were discarded. |
| 5 | Prefix + id parsing | `sinkholes_data_loading.py:137-153` | `pref`, `self.ids`, `start_intf_name` — the `[start:start+17]` slice parses `YYYYMMDD_YYYYMMDD`. |
| 6 | **`.npy` load** | `sinkholes_data_loading.py:180-184` | `image_data.shape` — `(N,200,100)` = `nonz` path, `(ny,89,200,100)` = full grid. |
| 7 | Reshape | `sinkholes_data_loading.py:306-308` | Only with `--nonz_only False`. The row-major flatten that defines `flat = i*89 + j`. |
| 8 | Index map | `sinkholes_data_loading.py:489-497` | `image_data.shape` after `expand_dims`, `len_examples`, `index_map[:5]`. |
| 9 | **`mask_values`** | `sinkholes_data_loading.py:82` | `mask_dir`, `idx`, the glob result. Where the missing-`nonz` crash fires (§3). Runs in a `multiprocessing.Pool` (`:499`) — breakpoint the caller at `:507` instead. |
| 10 | Item fetch | `sinkholes_data_loading.py:555-569` | `sample`, `intf_idx`, `patch_idx`, `img.shape`. |
| 11 | **Normalization** | `sinkholes_data_loading.py:541-549` | `mn`, `mx`, which branch. Always the `else` here: `zmask.sum()`, `out[c].min()` before/after. |
| 12 | Label mapping | `sinkholes_data_loading.py:529-537` | `mask_values`, `np.unique(mask)` — expect `[0 1]`. |
| 13 | Augmentation | `sinkholes_data_loading.py:583-592` | **Entirely commented out**; `self.do_augmentations` is never assigned, so `--augment` is inert. Verify the attribute is absent. |
| 14 | Tensor conversion | `sinkholes_data_loading.py:599-606` | `img_t.shape` — `(1,H,W)` non-temporal, `(T,H,W)` temporal; `msk_t.dtype` must be `int64`. |
| 15 | Batching | `train_sinkholes_unet.py:358-362` | `images.shape`, and the `assert images.shape[1] == model.n_channels` — the usual `--k_prevs` trap. |
| 16 | **Model forward** | `unet.py:173-186` | `x1..x5` down the encoder; `x5` is `(B,1024,12,6)` for 200×100. Break at `:178` to check `self.add_attn`. |
| 17 | Loss | `train_sinkholes_unet.py:405-407` | Unmasked branch: `criterion` output, then the `dice_loss` addition. |
| 18 | Masked loss | `train_sinkholes_unet.py:387-403` | Only with `--treat_nodata_regions`. `V_any.mean()`, `loss_bce`, `loss_dice`, `bce_fp`. `pos_w` is hardcoded `8.0` at `:388`. |
| 19 | Validation | `evaluate.py:703-731` | `mask_pred` before/after the `sigmoid > 0.5` at `:710`; `dice_score` at `:728`. |

**Debugging DataLoader internals:** set `num_workers=0` (`train_sinkholes_unet.py:329`) or your
breakpoints inside `__getitem__` will fire in a subprocess.

---

## 9. Local landmines

**`LOCAL_ENVIRONMENT` is not a debug flag.** It is tested for truthiness only, so `0` and `false`
also enable it. Two unrelated effects:

- `train_sinkholes_unet.py:251-254` — `random_by_intf` only: replaces your train/val/test lists.
  Overrides `--preset_test_val_21` too.
- `train_sinkholes_unet.py:293-294` — `spatial` only: replaces `intf_list` with one interferogram.
- `test_full_intf.py:282-285` — **inverted**: when *unset* it forces `args.plot = False`, so
  setting it is what *enables* `--plot`.
- `random_by_patch` and `preset_by_intf` are unaffected.

**Never pass `--plot` on macOS.** `evaluate.py:64` and `test_full_intf.py:280` assign
`plt.rcParams['backend'] = 'Qt5Agg'` at **import** time. Because it is an rcParams write, the
`MPLBACKEND` environment variable cannot override it, and PyQt5 is not installed. Figure creation
is gated by `plot and is_local` (`evaluate.py:412`), so with `plot=False` you are safe. In your
own scripts call `matplotlib.use('Agg')` **before** importing anything from this repo.

**Importing `get_intf_info` does work.** Lines 216-219 run at import time — they load the 194 KB
`intf_coord.json` from the CWD, build the North common grid, and `print` it. You will see
`(35.283142415, 31.794724185, 15660, 19437, 2.777e-05, 2.777e-05)` appear mid-training: that is
DataLoader workers re-importing the module. Harmless, but it is why every script must run from
the repo root.

**`torch.cuda.amp.GradScaler` warns.** `train_sinkholes_unet.py:349` uses the deprecated spelling;
torch 2.10 emits a `FutureWarning` and continues.

**The OOM fallback is broken.** `train_sinkholes_unet.py:582-591` calls `train_model(model=...)`
without the required positional `args`, so the recovery path itself raises `TypeError`.

**Checkpoints are 124 MB each, one per epoch.** A 20-epoch run writes 2.5 GB to `outputs/`.
There is no best-checkpoint tracking; you pick an epoch by hand.

---

## 10. Quick reference

```bash
export PY=/Users/guypi/.pyenv/versions/3.12.3/bin/python
cd /Users/guypi/Projects/sinkholes

# one-time: derive nonz patches + corrected metadata
$PY src/data_prep/prepare_local_subset.py

# 33-second smoke test
LOCAL_ENVIRONMENT=1 $PY src/training/train_sinkholes_unet.py --epochs 1 \
  --partition_mode random_by_intf --patches_dir data/patches/train_ready/ \
  --intf_dict_path data/metadata/intf_coord_local.json \
  --nonz_only True --k_prevs 0 --batch_size 8 --job_name local_smoke

# patch-level metrics
$PY src/inference/test.py --test_data_path outputs/<run>/test_dataset_<run>.pkl \
  --model outputs/<run>/checkpoints/<run>checkpoint_epoch1.pth --k_prevs 0

# full interferogram -> polygons
$PY src/inference/test_full_intf.py --model <name>.pth --input_patch_dir data/patches/train_ready/ \
  --intf_source intf_list --intf_list "20190205_20190216" --k_prevs 0 --job_name fi
```

Always: repo root, trailing slash on `--patches_dir`, `--intf_dict_path` pointing at the local
dictionary, `--k_prevs` matching between train and test, and no `--plot`.
