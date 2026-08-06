# Pipeline reference

One document for the whole pipeline: the conventions the science depends on, then each
stage with its command. Companion to the top-level [`README.md`](../README.md)
(orientation + quickstart) and [`TRAINING_RUNS.md`](./TRAINING_RUNS.md) (what the runs
so far actually scored). WEXAC paths below refer to the data host at
`/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/`.

```
(0) prepare-metadata      .ers headers      → intf_coord.json
(1) prepare-patches       .unw + GT .shp    → {data,mask}_patches_* grids + nonz files
(1b) count-positives      patch dir         → nonz_num inside the dictionary
(1c) clean-patches        [optional QC]     → cleaned/ patch variants
(2) train                 patches           → outputs/<job>/checkpoints/best.pt + test pickle
(3a) test-patches         pickle + ckpt     → patch-level metrics
(3b) eval-scenes          grids + ckpt      → scene arrays + polygon shapefiles
     eval-outputs         saved arrays      → object-level metrics + figures
(4) predict               raw .unw + ckpt   → polygon shapefiles for new scenes
```

---

## Conventions (load-bearing — read once)

**Interferogram ids** are `YYYYMMDD_YYYYMMDD` (start_end). Raw scene filenames embed the
dates at fixed positions; `sinkholes.meta.intf_id_from_filename` extracts them.

**Frames and aligned origins.** The coast is covered by two Sentinel-1 frames whose
rasters start at slightly different origins per date. Every scene is cropped to its
frame's fixed origin before patching — **North `(35.37, 31.79)`, South `(35.32, 31.44)`**
(`sinkholes.geo.FRAME_ORIGINS`) — so patch `(i, j)` of one date covers the same ground
as patch `(i, j)` of any other. Exported polygon coordinates are computed from these
origins; changing them, or the column offset below, silently shifts every shapefile.

**Column crops.** Patch grids keep columns `[0, 4500)` of the aligned scene
(`geo.X_CROP_COLS`). The raw-scene predict path instead crops `--x_pxls_offset` columns
(default 3000, `geo.PREDICT_X_OFFSET`) off a common grid and references polygon
longitudes to `origin + offset · dx`.

**Patch geometry.** Default 200×100 (H×W) with `strides_per_patch 2`, i.e. steps of
(100, 50) — 50% overlap. Directory and file names encode all of it:
`data_patches_H200_W100_strpp2_11days_Aligned/data_patches[_nonz]_<intf>_H200_W100_strpp2.npy`.
Per interferogram there are **four arrays**: the full `(ny, nx, H, W)` data/mask grids
(full-scene evaluation and temporal training read these) and the positive-only
`(N, H, W)` `nonz` subsets (single-frame training reads these) — both are required.
`nonz_indices.json` in the data directory maps each id to its positive patches' grid
coordinates.

**Normalisation and no-data.** One conditional rule everywhere
(`sinkholes.normalise`): data already in `[0, 1]` passes through with exact `0`
remapped to `0.5`; anything else is wrapped-phase radians and maps through
`(φ + π) / 2π`. After normalisation, `0.5` **is** the no-data code — validity masks are
recovered as `|x − 0.5| > tol`. Training decides the branch per patch, full-scene
inference per scene; the archive mixes `[0, 1]`-scaled and radian scenes, and the
conditional rule handles both.

**Temporal chains.** `--add_temporal --k_prevs k` stacks each interferogram with its
`k` predecessors — **exactly `i × 11` days earlier, same frame** (`meta.find_11day_sequences`).
Ids missing any predecessor are dropped from training; at evaluation time
`--fallback_replicate` keeps them and missing predecessor *files* are padded with the
current frame. The stack is always **chronological, oldest → newest, current frame
last** — in the dataset, inside the ConvLSTM, and through scene reconstruction. The
training target is the **newest timestep's mask** (`--union_temporal_mask` restores the
legacy union-over-stack target).

**Validity channels.** `--treat_nodata_regions` appends one validity map per timestep in
**block layout** `[img_t0..img_tT-1, V_t0..V_tT-1]`, doubling the channel count, and
switches training to the masked loss. The validity channels pass through normalisation
untouched (they must stay strictly {0, 1}).

**LiDAR gating.** Full-scene prediction only runs on tiles that lie entirely inside the
LiDAR coverage polygons (`assets/lidar_mask_polygs.shp`, selected per interferogram via
its `lidar_mask` source id) — **of the current frame AND every predecessor**.

**Checkpoints describe themselves.** Every evaluation command builds its network through
`sinkholes.models.build_from_checkpoint`, which detects the architecture from the
weights (ConvLSTM: `convlstm.*` keys + an embedded config blob carrying hidden
size/kernel; `--add_attn`: `attn.*` keys; AttentionUNet: its deeper `DoubleConv`
indices; else plain UNet) and reads the input channel count off the first convolution.
ConvLSTM channel counts are **per timestep**, not per stack. The architecture flags
remain as explicit overrides; one that contradicts the weights is an error. Adding a
fifth architecture is one `register()` entry in `sinkholes/models/factory.py` — no
command changes (`sinkholes architectures` prints the registry).

---

## (0) Coordinate dictionary

```bash
sinkholes prepare-metadata --intf_dir <dir with .ers headers> --out_path intf_coord.json
```

Parses origin, pixel size, raster dimensions and byte order per interferogram, attaches
the LiDAR source id (from `assets/lidar_intf_mask.txt`) and the frame. Run once per data
drop; the committed copy is `assets/intf_coord.json` (716 interferograms, 2019–2023).
`nonz_num` is filled later by `count-positives`. The per-year LiDAR shapefiles merge
into the committed coverage mask with `sinkholes merge-lidar`.

## (1) Patches

```bash
sinkholes prepare-patches \
  --input_dir <dir with .unw> --output_dir <patches root> \
  --gt_polygon_file_path sub_20231001.shp \
  --patch_size 200 100 --strides_per_patch 2 --days_diff 11
```

Rasterises the GT polygons matched by exact start/end date (no match → warning + empty
mask), aligns each scene to its frame origin, cuts the four arrays per interferogram and
writes `nonz_indices.json`. `--by_list id1,id2` / `--year_range Y0 Y1` restrict the set.

### (1b) Count positives — required before training

```bash
sinkholes count-positives --input_patch_dir <patches root>/data_patches_H200_W100_strpp2_11days_Aligned \
  --intf_dict intf_coord.json --out_path intf_coord.json
```

Training **skips** every interferogram whose `nonz_num` is `'none'`, so the counts in
the dictionary you pass as `--intf_dict_path` must come from the same patchify run as
the patches you train on. Input and output paths are explicit — nothing is overwritten
implicitly.

### (1c) Clean patches — optional

```bash
sinkholes clean-patches --patches_path <patches root> --patch_size 200 100 --strides_per_patch 2
```

Drops mask polygons that are mostly raw-zero (no-data) pixels or that are slivers
hugging patch edges → `cleaned/` subdirectories, used with `--use_cleaned_patches`.

### Partial local downloads

```bash
sinkholes prepare-local-subset          # defaults match data/patches/{images,masks}
```

Derives the `nonz` files from the downloaded full grids (they are pure re-indexing via
`nonz_indices.json`), symlinks the grids alongside, and writes
`data/metadata/intf_coord_local.json` with counts corrected for the download and
`'none'` elsewhere — which restricts any training run to exactly what is on disk.
`--verify_full` recounts from the masks instead of trusting the JSON.

## (2) Train

```bash
sinkholes train --epochs 30 --partition_mode random_by_intf \
  --patches_dir <patches root>/ --patch_size 200 100 --stride 2 \
  --pos_w 8 --learning-rate 1e-5 --batch_size 8 --job_name my_run --seed 0
```

**Architectures** — pass at most one flag:

| flag | model | time axis |
|---|---|---|
| *(none)* | `UNet` | frames stacked as input channels |
| `--add_attn` | `UNet` + channel self-attention at the bottleneck | as above |
| `--attn_unet` | `AttentionUNet` (gates on every skip) | as above |
| `--convlstm_unet` | `ConvLSTMUNet` | shared encoder per timestep, ConvLSTM over the bottleneck sequence |

`--convlstm_unet` requires `--add_temporal`; `--convlstm_hidden` (0 = match the 1024
bottleneck) dominates its parameter count and `--convlstm_kernel` must be odd. Hidden
size and kernel are stored **inside the checkpoint**, so they are never re-specified at
evaluation time. On the runs to date the ConvLSTM at batch 16 is the best model — see
`TRAINING_RUNS.md`, including the caveat that each run drew a different split.

**Partition modes** (`--partition_mode`):

| mode | split unit | notes |
|---|---|---|
| `random_by_patch` | individual patches | overlapping patches leak across splits — optimistic |
| `random_by_intf` | whole interferograms | recommended; `--seed` makes the shuffle reproducible |
| `spatial` | latitude line (`--thresh_lat`) within each interferogram | train north of it, val/test south |
| `preset_by_intf` | `--partition_file partition_*.json` | fixed split from `make-partition` |

Add-ons: `--preset_test_val_21` (fixed 2021 hold-out for val+test);
`--test_data_to_exclude <old test pickle>` (pin the test split to a previous run's
interferograms); `--nonoverlap_tr_tst` (patch-level split with a spatial gap);
`--train_intfs/--val_intfs/--test_intfs` (explicit comma lists, e.g. for smoke tests);
`--add_ring_negatives --neg_per_pos 1.0 --neg_ring_inner 1 --neg_ring_outer 3` (empty
patches sampled in an annulus around positives — candidates must be empty at *every*
timestep); `--nonz_only/--no-nonz_only`; `--add_nulls_to_train`;
`--train_with_nonz_th --nonz_th N S` (per-region positive-count threshold).

**Loss.** Default: `BCEWithLogits(pos_weight=--pos_w) + soft Dice`. With
`--treat_nodata_regions`: masked BCE (pos_weight fixed at 8.0, deliberately independent
of `--pos_w` — the masked objective was tuned with it) + masked Dice + a false-positive
suppression term inside no-data areas. Validation reports the *same* objective, so the
`train/loss` vs `val/loss` gap is a straight overfitting read.

**Optimisation:** RMSprop, `ReduceLROnPlateau` on val Dice, gradient clipping at 1.0,
`--amp` for mixed precision. Device selection is automatic (CUDA → MPS → CPU).

**Outputs** under `outputs/<job>_<ts>/`: `checkpoints/best.pt` + `last.pt` (+ one
`.pth` per epoch unless `--save_best_only`; `interrupted.pt` on Ctrl-C), `results.csv`
(one row per epoch), `curves.png`, `validation/preds/epoch_*.png` (input/GT/probability
grids of the same patches every `--sample_every` epochs — a flipbook of the model
improving; the `--n_samples` patches span the range of ground-truth area in the
validation set, sparsest to densest, and are kept `--sample_min_sep` samples apart so
the grid never shows one sinkhole through several overlapping windows),
`logs/reporter.log`, the run log, and `test_dataset_<job>.pkl` — the
held-out split that feeds `test-patches`. `--patience N` early-stops; `--reporter/
--no-reporter` toggles the console table. Note `val/F1` pools every pixel (micro) while
`val/dice` averages per batch (macro); they differ by design.

## (3a) Patch-level test

```bash
sinkholes test-patches --test_data_path outputs/<run>/test_dataset_<run>.pkl \
  --model outputs/<run>/checkpoints/best.pt --k_prevs 2 --th 0.7 --b 5
```

Mean per-patch Dice, pixel precision/recall, and object-level precision/recall
(predicted objects buffered by `--b` pixels; a GT object counts as detected when the
covered fraction exceeds `--th`). Test pickles from before the rewrite still load.
`--k_prevs` must match training — a mismatch is reported against the checkpoint's real
channel count.

## (3b) Full-scene evaluation

```bash
sinkholes eval-scenes --model models/<ckpt>.pt \
  --input_patch_dir <patches root>/ \
  --intf_source test_dataset --test_dataset outputs/<run>/test_dataset_<run>.pkl \
  --patch_size 200 100 --data_stride 2 --recon_th 0.25 --k_prevs 2 \
  --blend_type hann --job_name eval_my_run --save_confidence --merge_polygs
```

Reconstructs each scene from its aligned patch grid: overlapping tile predictions are
averaged (or Hann-blended with `--blend_type hann --window_gamma g`), LiDAR-gated,
thresholded at `--recon_th` and polygonised. `--intf_source` is one of `intf_list`
(+`--intf_list id1,id2`), `test_dataset`, `preset` (+`--valset_from_partition`), or
`all` (+`--year_range`). `--unioned_mask` unions the GT over the temporal stack;
`--replicate_input` feeds the current frame in every temporal slot;
`--no-add_lidar_mask` disables gating.

Outputs under `<output_dir>/<model>/<job>_<ts>/` (default root `outputs/predictions`):
`polygs/<intf>_predicted_polygs.shp` (EPSG:4326), `<intf>_image.npy` (`(C, H, W)`,
channel 0 = current then previous frames newest-first), `_pred.npy` (confidence),
`_pred_th.npy`, `_gt.npy`. Budget ≈ 2 GB per interferogram; prefer `test_dataset` /
`intf_list` over `all` on a laptop. With `--merge_polygs` a combined shapefile with
`intf_key/start_date/end_date/track` columns is added.

The canvas is one stride-step taller/wider than the covered extent (a zero trailing
band) — historical shape, harmless, kept so saved arrays stay comparable across runs.

```bash
sinkholes eval-outputs --path outputs/predictions/<model>/<job>_<ts>/ --save_figures
```

Re-thresholds the saved confidence maps at 0.125/0.25/0.5, computes object-level
precision/recall at each (South-frame scenes are cropped to their populated northern
half), writes a metrics JSON and, with `--save_figures`, a per-interferogram overview
PNG. All figures render via Agg — no GUI, safe on headless nodes.

### Inspect a finished run

```bash
sinkholes inspect-run outputs/<run_dir> [epoch]     # curve + threshold sweep
sinkholes curves outputs/<run_dir>                  # regenerate curves.png only
```

The sweep reports Dice over pooled test pixels (micro) — quote `test-patches`' macro
number when comparing runs. The default 0.5 threshold is rarely optimal; on the runs to
date the best pooled Dice sat at 0.70 (single-frame U-Net) to 0.85 (temporal U-Net).

## (4) Predict on new scenes

```bash
sinkholes predict --intfs_dir <dir with .unw> \
  --intfs_list "20220613_20220624,20220624_20220705" \
  --model models/<ckpt>.pt --patch_size 200 100 --strdpp 2 \
  --rth 0.25 --k_prevs 2 --x_pxls_offset 3000 \
  --output_polygs_dir out_polygs --plot_polygs
```

Reads the raw scenes of each interferogram's chain, aligns them onto their common grid,
crops `--x_pxls_offset` columns, reconstructs the LiDAR-gated prediction and writes
`<intf>_predicted_polygons.shp` per scene (+ an overview PNG with `--plot_polygs`;
`--add_gt_polygs` overlays ground truth where it exists). **`--x_pxls_offset` must match
the crop** — a wrong value shifts every polygon. Overlap averaging here divides by the
actual per-pixel tile count, so scene borders carry full-strength probabilities.

---

## Regression oracle

The end-to-end check that the science is intact — same model, same scene, same count:

```bash
sinkholes eval-scenes --model models/run_v2_best.pt \
  --input_patch_dir data/patches/train_ready/ \
  --intf_source intf_list --intf_list 20190205_20190216 \
  --patch_size 200 100 --data_stride 2 --k_prevs 0 --recon_th 0.25 --job_name oracle
```

must report **464 polygons** for `20190205_20190216` (LiDAR gating on, no blending).
One run takes ~3 min on MPS and writes ~1.3 GB — delete the output directory after.

## Gotchas checklist

- [ ] `count-positives` ran after patch generation, on the same patchify run you train on.
- [ ] Both the full grids and the `nonz` files exist in the patch directories.
- [ ] `--k_prevs` at evaluation matches training (the channel warning names the fix).
- [ ] Temporal evaluation of an interferogram needs its predecessors' grids on disk —
      `--fallback_replicate` pads missing ones with the current frame.
- [ ] `--x_pxls_offset` (predict) matches how the scenes were cropped.
- [ ] Partition mode is leakage-free for the claim you're making: `random_by_patch`
      leaks overlapping patches; `random_by_intf` still shares locations across dates;
      the strongest hold-outs are `spatial` and `--preset_test_val_21`.
- [ ] Pass `--seed` when a split must be reproducible, and record the split lists the
      run logs print.
