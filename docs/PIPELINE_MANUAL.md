# Pipeline Manual — quick reference

How to run each stage and call the main functions. Companion to
[`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) (audit, known bugs) and
[`TRAINING_RUNS.md`](./TRAINING_RUNS.md) (what the runs so far actually scored).
Commands are for the **WEXAC** environment where the data lives; defaults already point at
`/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/`.

> **Conventions**
> - Scripts live under `src/<stage>/` and are invoked by path:
>   `python src/training/train_sinkholes_unet.py …` (`CHANGELOG.md #10`).
> - Interferogram id format: `YYYYMMDD_YYYYMMDD` (start_end), e.g. `20190204_20190215`.
> - Patch size default `200 100` (H W); stride `2` means a 50 % overlap step.
> - Boolean flags come in **two styles** here: some are real switches (`--attn_unet`), some take a
>   string (`--nonz_only True`). Match what each script expects (shown below).
> - Committed data assets (`intf_coord.json`, `lidar_mask_polygs.shp`, the partition JSONs) live
>   in `assets/` and now resolve by absolute path, so scripts run from **any** working directory.
>   Paths you pass yourself (`--patch_dir`, `--out_dir`, …) are still relative to your CWD.
> - `LOCAL_ENVIRONMENT` does **two unrelated things** — see the box below. Leave it unset for
>   real runs.

### `LOCAL_ENVIRONMENT` — what it actually changes

It is read in exactly two places, and it is **not** a debugger, a device selector, or a path
switch. It is only ever tested for truthiness (`os.environ.get('LOCAL_ENVIRONMENT', False)`),
so **any** non-empty value — `1`, `0`, `false` — turns it *on*. To disable it, unset it.

| Script | Line | Effect |
|---|---|---|
| `train_sinkholes_unet.py` | `:521` | sets `is_running_locally` |
| ↳ | `:251-254` | **`--partition_mode random_by_intf` only:** silently replaces your train/val/test lists with `train=['20190204_20190215']`, `test=['20190205_20190216']`, `val=['20190216_20190227']`. Runs *after* `--preset_test_val_21`, so it wins over that too. |
| ↳ | `:293-294` | **`--partition_mode spatial` only:** replaces `intf_list` with `['20191129_20191210']`. |
| ↳ | `:435` | passed to `evaluate(..., is_local=...)`, where it gates plotting together with `plot` (`evaluate.py:412`). |
| `test_full_intf.py` | `:282-285` | **inverted meaning:** when *unset*, forces `args.plot = False`. Setting it therefore *enables* `--plot`. |

`random_by_patch` and `preset_by_intf` are **not** affected — the variable does nothing there.

⚠ On macOS, enabling plotting anywhere will crash: `evaluate.py:64` and `test_full_intf.py:280`
set `plt.rcParams['backend'] = 'Qt5Agg'` at import time (which `MPLBACKEND` cannot override), and
PyQt5 is typically not installed. Never combine `LOCAL_ENVIRONMENT` with `--plot` locally.

---

## End‑to‑end flow

```
(0)  src/data_prep/prepare_intf_coord_dict.py     → assets/intf_coord.json
(1)  src/data_prep/prepare_intrfrgrm_pathches.py  → data_/mask_patches_*  + nonz_indices.json
(1b) src/data_prep/check_patches.py               → fills nonz_num in assets/intf_coord.json
(1c) src/data_prep/clean_patches.py   [optional]  → cleaned/ patch variants
(2)  src/training/train_sinkholes_unet.py         → outputs/<job>/checkpoints/*.pth
                                                    + test_dataset_<job>.pkl
       └─ architecture: default UNet │ --add_attn │ --attn_unet │ --convlstm_unet
(3a) src/inference/test.py                        → patch‑level metrics
(3b) src/inference/test_full_intf.py              → outputs/predictions/... (*.npy + polygs/*.shp)
     src/inference/evaluate_full_intf_output.py   → metrics JSON + figures
(4)  src/inference/predict_new_intf_vA.py         → out_polygs/*.shp   (new, GT‑free data)

Every stage that loads a checkpoint detects its architecture from the weights
(src/models/factory.py), so all of them work with UNet / AttentionUNet / ConvLSTM alike.
```

---

## (0) Build the coordinate dictionary

Parses `.ers` headers → `intf_coord.json` (origin, dx/dy, size, byte order, frame, LiDAR id).

```bash
python src/data_prep/prepare_intf_coord_dict.py \
    --intf_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/ \
    --out_dir  ./
```
Run once (already provided in the repo). `nonz_num` is filled later by `check_patches.py`.

---

## (1) Generate patches

Rasterizes GT polygons, aligns frames, cuts `data`/`mask` patches + the `nonz` (positive) subset,
and writes `nonz_indices.json` (positive patch grid coords, used by temporal training).

```bash
python src/data_prep/prepare_intrfrgrm_pathches.py \
    --input_dir  /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/ \
    --output_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/ \
    --gt_polygon_file_path /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/sub_20231001.shp \
    --patch_size 200 100 \
    --strides_per_patch 2 \
    --days_diff 11 \
    --align_frames
```
Useful extras: `--by_list "20190204_20190215,20190215_20190226"` (subset), `--year_range 2019 2020`,
`--plot_data True` (visual QC). Output dirs are named
`data_patches_H200_W100_strpp2_11days_Aligned/` and `mask_patches_..._Aligned/`.

⚠ `--align_frames` is declared `action='store_true', default=True` (line 95), so alignment is
**always on and cannot be disabled** — the `_Aligned` suffix is unconditional. `train_sinkholes_unet.py:112`
and `test_full_intf.py:342` both hardcode that suffix, so this is consistent, just not optional.

Each interferogram writes **four** arrays (line 250): the full grids
`data_patches_<intf>_*.npy` / `mask_patches_<intf>_*.npy` with shape `(ny, nx, H, W)`, and the
positive-only subsets `data_patches_nonz_<intf>_*.npy` / `mask_patches_nonz_<intf>_*.npy` with
shape `(N, H, W)`. **Both are required** — training reads the `nonz` files, and
`SubsiDataset` needs them even when `--nonz_only False` (`sinkholes_data_loading.py:82` globs the
`nonz` prefix unconditionally).

**Reusable function** (`prepare_intrfrgrm_pathches.py:19`):
```python
patchify(input_array, window_size, stride, mask_array=None, nonz_pathces=True, offset=0, nx=4500)
```
- `window_size` — `(H, W)` tuple.
- `stride` — a **2-tuple** `(Sy, Sx)`, *not* an int. The caller passes
  `(H // strides_per_patch, W // strides_per_patch)`, i.e. `(100, 50)` for the defaults.
- `nonz_pathces` *(sic)* — 5th **positional** parameter. Passing `offset` positionally in its
  place is a silent bug.
- `offset`, `nx` — the input is cropped to columns `[offset : offset+nx]` before patching.

Returns `(data_patches, mask_patches, data_nonz, mask_nonz, nonz_indices)` **only when
`mask_array is not None`**; with `mask_array=None` it returns `data_patches` alone.
`nonz_indices` is a list of `[i, j]` *grid* coordinates in row-major order.

### (1b) Count positives — required before training
```bash
python src/data_prep/check_patches.py \
    --input_patch_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/data_patches_H200_W100_strpp2_11days_Aligned/
```
Counts `data_patches_nonz_*.npy` files (it reads `shape[0]` of each) and writes `nonz_num` per
intf. Training **skips** intfs whose `nonz_num == 'none'`.

⚠ Three things the script does not advertise:
- It requires a **trailing slash** on `--input_patch_dir` (it concatenates, line 24).
- It only sees interferograms that have a `nonz` file; every other intf in the dictionary is set
  to `'none'` and thereby excluded from training.
- It reads and **overwrites `intf_coord.json` in the current working directory**, ignoring
  `--input_patch_dir`. Back the file up, or point training at a copy via `--intf_dict_path`.

### (1c) Clean patches — optional
```bash
python src/data_prep/clean_patches.py \
    --patches_path /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/ \
    --patch_size 200 100 --strides_per_patch 2
```
Drops mask polygons that are mostly no‑data or hug patch edges → `cleaned/` subdirs. Use later with
`--use_cleaned_patches`.

---

## (2) Train

```bash
python src/training/train_sinkholes_unet.py \
    --epochs 30 \
    --partition_mode random_by_intf \
    --patches_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/ \
    --patch_size 200 100 --stride 2 \
    --nonz_only True \
    --pos_w 8 \
    --learning-rate 1e-5 --batch_size 1 \
    --job_name my_run
```

**Partition modes** (`--partition_mode`):
| mode | split unit | notes |
|---|---|---|
| `random_by_patch` | individual patches | ⚠ leaks overlapping patches across splits — optimistic. |
| `random_by_intf` | whole interferograms | recommended for by‑intf evaluation. |
| `spatial` | latitude line within each intf (`--thresh_lat`) | geographic hold‑out. |
| `preset_by_intf` | `--partition_file partition_*.json` | fixed reproducible split. |

**Common add‑ons:**
- Temporal stacking: `--add_temporal --k_prevs 2` (input channels become `k_prevs+1`).
- No‑data handling: `--treat_nodata_regions` (adds a validity channel → channels ×2).
- Architecture: `--attn_unet` (Attention U‑Net) **or** `--add_attn` (bottleneck attention in `UNet`)
  **or** `--convlstm_unet` (ConvLSTM U‑Net — see below). Pass at most one.
- Ring negatives: `--add_ring_negatives --neg_per_pos 1.0 --neg_ring_inner 1 --neg_ring_outer 3`.
- 2021 temporal hold‑out preset: `--preset_test_val_21`.
- Reproducibility: `--seed 0` (note: currently only seeds ring‑neg sampling — see summary §7.6).
- Exclude a previous run's test set: `--test_data_to_exclude outputs/<old>/test_dataset_<old>.pkl`.

Run `python src/training/train_sinkholes_unet.py -h` for the full ~40 flags.

### Architectures — which one to use

| Flag | Model | How the time axis is handled |
|---|---|---|
| *(none)* | `UNet` | Frames stacked as **input channels** (`k_prevs+1`). |
| `--add_attn` | `UNet` + bottleneck self‑attention | as above |
| `--attn_unet` | `AttentionUNet` | as above |
| `--convlstm_unet` | `ConvLSTMUNet` | Encoder runs **per timestep**, a ConvLSTM consumes the bottleneck **sequence** (`CHANGELOG.md #6`). |

Measured on this data (`TRAINING_RUNS.md`), the ConvLSTM at `--batch_size 16` is the best model so
far (val dice 0.591), channel‑stacking into a plain U‑Net is the worst (0.342), and the
single‑frame baseline sits between them (0.565). **Do not read those gaps as final** — each run
drew a different `random_by_intf` split.

### Adding an architecture (`src/models/factory.py`)

Inference no longer branches on flags. Every script that loads a checkpoint
(`test.py`, `test_full_intf.py`, `inspect_run.py`, both `predict_new_intf*.py`) calls
`factory.build_from_checkpoint()`, which **detects the architecture from the weights**:

| Architecture | Detected by |
|---|---|
| `convlstm_unet` | `convlstm.*` keys / a `convlstm_unet_config` blob |
| `unet_add_attn` | `attn.*` keys |
| `attention_unet` | `down1.maxpool_conv.1.double_conv.5.*` (its deeper `DoubleConv`) |
| `unet` | catch‑all |

Input channels come from `inc.double_conv.0.weight.shape[1]`, so a wrong `--k_prevs` is caught
with a message instead of a key mismatch. Flags (`--convlstm_unet`, `--attn_unet`, `--add_attn`)
still work as an explicit override, and a flag that contradicts the weights is an error.

To add a fifth architecture, append one entry to the registry in `src/models/factory.py` —
**no inference script changes**:

```python
register(Architecture(
    name='my_net',
    detect=lambda sd: any(k.startswith('my_block.') for k in sd),
    build=_build_my_net,
    cli_flag='--my_net',
))
```

Entries are tried in registration order, so put specific ones above the `unet` catch‑all. Run
`python src/models/factory.py` to print the registry. The contract is covered by
`tests/test_factory.py` (detect → rebuild → `load_state_dict` → identical forward output).

### ConvLSTM U‑Net (`--convlstm_unet`)

**Requires `--add_temporal`.** Mutually exclusive with `--attn_unet` / `--add_attn`.

```bash
python src/training/train_sinkholes_unet.py \
    --convlstm_unet --add_temporal --k_prevs 2 \
    --convlstm_hidden 1024 --convlstm_kernel 3 \
    --epochs 30 \
    --partition_mode random_by_intf \
    --patches_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/ \
    --patch_size 200 100 --stride 2 \
    --nonz_only True --pos_w 8 \
    --learning-rate 1e-5 --batch_size 16 \
    --job_name convlstm_v1
```

| Flag | Default | Notes |
|---|---|---|
| `--convlstm_hidden` | `0` → match the U‑Net bottleneck (**1024**) | Dominates parameter count. Lower it first if you hit memory limits. |
| `--convlstm_kernel` | `3` | Gate convolution kernel; must be odd. |
| `--k_prevs` | — | Sequence length is `k_prevs+1` (so `2` → T=3). Must match at test time. |
| `--union_temporal_mask` | off | Legacy. Target is the **latest timestep** by default (`#7`). |

**Config travels inside the checkpoint.** Training writes a `convlstm_unet_config` key
(hidden size, kernel, bilinear, channels per timestep) into the `state_dict`, and
`build_convlstm_unet()` pops it and rebuilds the exact architecture. You do **not** re‑specify
`--convlstm_hidden` / `--convlstm_kernel` at test time — only the `--convlstm_unet` flag itself,
so the right branch is taken.

**Practical notes from the runs so far:**
- Roughly **2.5–4×** the per‑epoch cost of the plain U‑Net (3m24s–4m28s/epoch vs ~1m38s at these
  batch sizes, on MPS).
- `--batch_size 32` for 100 epochs **overfit** and scored *worse* than 30 epochs at batch 16.
  Prefer the smaller batch and use `--patience` rather than more epochs.
- A ConvLSTM checkpoint is **not** loadable by the `UNet` branch — see the compatibility note in
  step (3b).

**Outputs** (under `./outputs/<job_name>_<timestamp>/`):
- `checkpoints/<job>checkpoint_epoch<N>.pth` — one per epoch (state_dict + `mask_values`).
- `checkpoints/best.pt` / `last.pt` — best `val/dice` and most recent epoch.
- `checkpoints/interrupted.pt` — written on Ctrl+C (reporter only).
- `results.csv` — one row per epoch, same columns as the console table (reporter only).
- `curves.png` — learning-curve figure (reporter only, `CHANGELOG.md #5`).
- `validation/preds/epoch_<NNN>.png` — input/GT/prediction grids (reporter only, `#5`).
- `test_dataset_<job>.pkl` — the held‑out test set (feeds steps 3a/3b).
- `validation/` — optional saved val arrays (`--save_val True`).
- `logs/reporter.log` — DEBUG-level reporter log (reporter only).
- `<job>.log`.

### CLI reporter (`CHANGELOG.md #3`, extended by `#5`)

On by default. Replaces the per-step log flood with a resolved-config banner, a fixed-width
per-epoch table, and a closing summary:

```
        epoch        time  train/loss    val/loss    val/dice     val/IoU      val/F1       val/P       val/R          lr
          1/3       1m30s       1.266       1.301      0.4709      0.3081      0.4711      0.4797      0.7363       1e-05
  new best val/dice=0.4709 @ epoch 1 — saved best.pt
```

- `val/loss` is the **same objective** as `train/loss` (both call `segmentation_loss()`), so the
  gap between the two columns is a straight overfitting read.
- `val/F1` is *not* `val/dice`: F1 pools every pixel in the split (micro), `val/dice` averages per
  batch (macro). They differ by design — see the note under *Inspect a finished run*.
- `--reporter False` restores the previous console output exactly (no banner/table/`results.csv`/
  `curves.png`/sample grids).
- `--patience N` early-stops after N epochs without a `val/dice` improvement (`0` = off).
- The run's `<job>.log` is **unchanged** — only the root logger's *console* handler is quietened,
  so existing `grep "Validation Dice"` workflows still work.

### Run figures (`CHANGELOG.md #5`)

`curves.png` is written at the end of every reporter run (including after a Ctrl+C): train/val loss
with the learning rate as a dashed step on a log twin axis, every `val/*` metric, best epoch marked.
Regenerate it for any finished run — including runs trained before this existed — without
retraining:

```bash
python src/training/train_reporter.py outputs/<run_dir>       # → outputs/<run_dir>/curves.png
```

`validation/preds/epoch_<NNN>.png` shows input / ground truth / predicted probability with the 0.5
contour, for a few validation patches:

- `--sample_every N` — write a grid every N epochs (default `1`, `0` disables).
- `--n_samples K` — patches per grid (default `4`).

Patches with positive ground truth are preferred, and the selection is deterministic across epochs
(`val_loader` is not shuffled), so the grids form a flipbook of the *same* patches improving.
Plotting uses the Agg canvas directly, so it is safe on macOS and on compute nodes without PyQt5 —
unlike `--plot`, which is still a GUI path (see the warning at the top of this manual).

### Inspect a finished run

```bash
python src/inference/inspect_run.py outputs/<run_dir> [epoch]
```
Prints the learning curve (from `results.csv`, or parsed from the log for older runs), the best
epoch, and a decision-threshold sweep over the held-out test set. Note it reports Dice over pooled
pixels, whereas `test.py` reports the mean of per-patch Dice — the latter is the harsher number.

Works for every architecture (`CHANGELOG.md #12`). When the run was trained with
`--save_best_only` (the default) there are no per-epoch checkpoint files, so it falls back to
`best.pt` / `last.pt` for those epochs; ask for another epoch and it lists what is available.

The sweep is worth running — the default 0.5 threshold is rarely optimal. On the runs to date the
best pooled Dice sat at 0.75 (ConvLSTM), 0.45 (single-frame U-Net) and 0.85 (temporal U-Net).

**Core functions:** `train_model(args, model, device, ...)`; losses
`masked_bce_with_logits(logits, y, V, pos_w)`, `masked_dice_loss_binary(logits, y, V)`,
`masked_ce_plus_softdice_multiclass(...)`; `dice_loss(pred, target)` from `dice_score.py`.

**Model construction:**
```python
from unet import UNet
from attn_unet import AttentionUNet
num_c = (k_prevs + 1) * (2 if treat_nodata_regions else 1)
model = UNet(n_channels=num_c, n_classes=1, bilinear=False, add_attn=False)
# or AttentionUNet(n_channels=num_c, n_classes=1)

# ConvLSTM U-Net: channels are per *timestep*, not the flattened stack.
from convlstm_unet import ConvLSTMUNet, build_convlstm_unet
model = ConvLSTMUNet(
    n_channels_per_timestep=2 if treat_nodata_regions else 1,
    n_classes=1, bilinear=False,
    convlstm_hidden_channels=1024, convlstm_kernel_size=3,
)
# from a checkpoint, let the stored config drive it:
model = build_convlstm_unet(state_dict, n_channels_per_timestep=1, n_classes=1, bilinear=False)
```

---

## (3a) Patch‑level test

Loads the pickled test set + a checkpoint, reports Dice, pixel P/R, and object‑level (OL) P/R.

```bash
python src/inference/test.py \
    --test_data_path outputs/my_run_<ts>/test_dataset_my_run.pkl \
    --model outputs/my_run_<ts>/checkpoints/my_runcheckpoint_epoch28.pth \
    --th 0.7 --b 5 \
    --k_prevs 2          # must match training
# add --unet_attn if you trained AttentionUNet, --add_attn for UNet bottleneck attention, --plot to visualize
```
- `--th` = OL overlap threshold (fraction of a GT object that must be covered).
- `--b`  = buffer in pixels applied to predicted objects when matching.

**ConvLSTM checkpoints** — add `--convlstm_unet` (and `--treat_nodata_regions` if it was trained
with validity channels). Hidden size and kernel come from the checkpoint; do not re-specify them:

```bash
python src/inference/test.py \
    --test_data_path outputs/convlstm_v1_<ts>/test_dataset_convlstm_v1_<ts>.pkl \
    --model outputs/convlstm_v1_<ts>/checkpoints/best.pt \
    --convlstm_unet --k_prevs 2
```

`--convlstm_unet` cannot be combined with `--unet_attn` / `--add_attn`; the script exits if you try.

**Core function:** `evaluate(net, dataloader, device, amp, is_local, out_path, epoch, mode='test',
th=..., buffer=..., plot=...)` in `evaluate.py`; object metrics via
`object_level_evaluate(gt, pred, image, th=, buffer=)`.

---

## (3b) Full‑interferogram test (by‑intf evaluation)

Runs the model on **aligned patch tiles**, reconstructs the full scene, LiDAR‑gates, thresholds,
polygonizes, and saves arrays + shapefiles.

Works with **every** architecture — the checkpoint is inspected and the matching model built
automatically (`CHANGELOG.md #12`). No architecture flag is needed for ConvLSTM here.

```bash
python src/inference/test_full_intf.py \
    --model my_runcheckpoint_epoch28.pth \        # looked up under ./models/
    --input_patch_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/ \
    --intf_source test_dataset \
    --test_dataset test_dataset_my_run.pkl \       # under ./test_data/
    --patch_size 200 100 --data_stride 2 \
    --recon_th 0.25 \
    --k_prevs 2 \
    --add_lidar_mask True \
    --blend_type hann \                            # optional overlap blending
    --job_name eval_my_run \
    --save_confidence --merge_polygs
```
- `--intf_source`: `intf_list` (+`--intf_list "id1,id2"`), `test_dataset`, `preset`
  (+`--valset_from_partition file.json`), or `all` (+`--year_range 2019 2023`).
- `--recon_th`: reconstruction threshold on the confidence map (default 0.25).
- `--blend_type hann` + `--window_gamma`: Hann‑window overlap blending.
- `--merge_polygs`: also emit one combined shapefile across all intfs (with date/track columns).
- `--attn_unet` if the checkpoint is an Attention U‑Net.
- `--output_dir`: root for the results tree (default `outputs/predictions`, `CHANGELOG.md #14`).

**Outputs** (`<output_dir>/<model>/<job>_<timestamp>/`): `polygs/<intf>_predicted_polygs.shp`,
`<intf>_image.npy`, `<intf>_pred.npy`, `<intf>_gt.npy` (saved when `intf_source != all`).

**Core function:** `reconstruct_intf_prediction(...)` (tiling + per‑patch predict + LiDAR gate +
blend + threshold).

### Running (3b) locally

Only `--input_patch_dir` and `--output_dir` are flags; the model and the pickled test set are
looked up under **`./models/`** and **`./test_data/`** relative to the working directory
(`test_full_intf.py:353` and `:342`), so run from the repo root and symlink into place:

```bash
mkdir -p models test_data
ln -sf ../outputs/<run>/checkpoints/best.pt models/<run>_best.pt
ln -sf ../outputs/<run>/test_dataset_<run>.pkl test_data/test_dataset_<run>.pkl
```

```bash
LOCAL_ENVIRONMENT=1 python src/inference/test_full_intf.py \
    --model <run>_best.pt \
    --input_patch_dir data/patches/train_ready/ \
    --intf_source test_dataset --test_dataset test_dataset_<run>.pkl \
    --patch_size 200 100 --data_stride 2 \
    --recon_th 0.25 --k_prevs 2 --add_lidar_mask True \
    --blend_type hann \
    --job_name eval_local --save_confidence --merge_polygs
```

Verified end to end on MPS (temporal U‑Net, 2 test intfs → 7 178 polygons).

- **`--k_prevs` must match the checkpoint** or the forward pass fails: read the in‑channel count
  off the training log (`Network: N input channels`). A U‑Net with 3 in‑channels needs
  `--k_prevs 2`, a single‑frame U‑Net `--k_prevs 0`. For ConvLSTM the channel count is
  per‑timestep, so it cannot be inferred — the log's `sequence length T` is `k_prevs + 1`.
- **Disk**: ≈ 2 GB per interferogram (`<intf>_image.npy` alone is ~1 GB per channel). Prefer
  `--intf_source test_dataset` / `--intf_list` over `all` on a laptop.
- **Predecessor masks** (`:487`) are loaded without the existence check the data patches get, so a
  test intf whose 11‑day predecessor was never downloaded raises `FileNotFoundError`; use
  `--fallback_replicate`.
- Never pass `--plot` without PyQt5 — `:303` forces the `Qt5Agg` backend (`PROJECT_SUMMARY.md`,
  *Considered and rejected*). Without `LOCAL_ENVIRONMENT` set, `--plot` is ignored anyway.

### Evaluate the saved full‑intf outputs (metrics + figures)
```bash
python src/inference/evaluate_full_intf_output.py \
    --path outputs/predictions/<model>/<job>_<ts>/ \
    --aligned_patches
# --skip_ol_metrics to skip object‑level metrics
```
Writes `olm_results_with22_23.json`. ⚠ It currently **forces plotting on** (`plot=True` hardcoded)
and re‑thresholds the confidence map at 0.125/0.25/0.5 internally — see `PROJECT_SUMMARY.md §6.3`.

---

## (4) Predict on new interferograms (no GT)

Runs a trained model on raw `.unw` scenes and exports predicted sinkhole polygons (lon/lat) as
shapefiles for GIS. **Prefer `predict_new_intf_vA.py`** — it fixes a LiDAR‑mask indexing bug in the
v1 script (`PROJECT_SUMMARY.md §6.5`).

Works with every architecture, ConvLSTM included — same auto-detection as (3b)
(`CHANGELOG.md #12`).

```bash
python src/inference/predict_new_intf_vA.py \
    --intfs_dir  /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/ \
    --intfs_list "20220613_20220624,20220624_20220705" \
    --model_dir ./models/ --model_file my_runcheckpoint_epoch28.pth \
    --patch_size 200 100 --strdpp 2 \
    --rth 0.25 \
    --k_prevs 2 \
    --x_pxls_offset 3000 \
    --output_polygs_dir ./out_polygs/ \
    --add_gt_polygs --gt_polygons_file_path sub_20231001.shp \
    --plot_polygs
```
- `--intfs_list` is **required in practice** (its default is a placeholder sentence).
- `--x_pxls_offset` must match how the model grid was cropped (default 3000) — a wrong value
  silently shifts all exported polygons.
- ⚠ Give `--output_polygs_dir` a **trailing slash** (`./out_polygs/`) — the script concatenates
  without a separator. Output filenames use the (misspelled) suffix `_predicted_polyogns.shp`.

**Core functions:** `predict_from_full_images(all_data, intf_coords, net, patch_size, strdpp, rth, ...)`,
`build_unified_gt_polygons(gdf, ids)`, plus the shared georeferencing helpers below.

---

## Reusable functions cheat‑sheet

| Function (module) | Signature → returns |
|---|---|
| `get_intf_coords(intf)` (`get_intf_info`) | → 12‑tuple `(x0,y0,dx,dy,ncells,nlines,x4000,x8500,lidar_mask,nonz,bo,frame)`. ⚠ predict scripts define a **10‑tuple** local version. |
| `find_11day_sequences(meta, k_prev=2, step_days=11, restrict_to=None)` (`get_intf_info`) | → `(chains_map, valid_intfs)` — per‑intf same‑frame temporal chains. |
| `crop_to_start_xy(intf, mask, x0, y0, x*, y*, dx, dy)` (`get_intf_info`) | → cropped arrays aligned to a common origin. |
| `build_common_grid_for_region(meta, 'North'|'South')` (`get_intf_info`) | → `(x*, y*, W, H, dx, dy)` intersection grid. |
| `patchify(arr, (H,W), (Sy,Sx), mask_array=None, nonz_pathces=True, offset=0, nx=4500)` (`prepare_intrfrgrm_pathches`) | → 5‑tuple `(data_patches, mask_patches, data_nonz, mask_nonz, nonz_indices)` when `mask_array` is given, else `data_patches` alone. `stride` is a **tuple**; `nonz_pathces` is the 5th positional arg. |
| `SubsiDataset(args, image_dir, mask_dir, intrfrgrm_list, dset, seq_dict)` (`sinkholes_data_loading`) | PyTorch `Dataset`; `__getitem__ → {"image","mask"}`. |
| `mask_array_to_polygons(mask)` (`polygs`) | → GeoDataFrame of pixel‑space polygons. |
| `plg_indx2longlat(gdf, intf_coords, x_start)` (`polygs`) | → GeoDataFrame in EPSG:4326 (lon/lat). |
| `build_convlstm_unet(state_dict=None, **fallback)` (`convlstm_unet`) | → `ConvLSTMUNet` rebuilt from the config stored in the checkpoint; `fallback` covers checkpoints saved without it. Pops the config key so the state dict loads directly. |
| `pop_model_config(state_dict)` (`convlstm_unet`) | → the stored config dict (or `None`); no‑op on stock U‑Net checkpoints. |
| `dice_coeff / dice_loss` (`dice_score`) | soft Dice metric / loss. |
| `evaluate(net, loader, device, amp, is_local, out_path, epoch, mode, ...)` (`evaluate`) | val/test loop → mean Dice (+ prints P/R, OL P/R in `mode='test'`). |
| `object_level_evaluate(gt, pred, image, th=0.7, buffer=5)` (`evaluate`) | → `(OL_recall, OL_precision, gt_area, feature_lists)`. |
| `remove_no_data_predictions(gdf, intf, th=0.7)` (`remove_no_Data_predictions`) | drops polygons that are mostly no‑data (0.5) pixels. |

---

## Gotchas checklist (before you trust a result)

- [ ] Ran `check_patches.py` after generating patches (else intfs are filtered out), and the
      `nonz_num` values in the dictionary you pass as `--intf_dict_path` were produced by the
      **same** patchify run as the patches you are training on.
- [ ] `data_patches_nonz_*` / `mask_patches_nonz_*` exist in the patch dirs — required even with
      `--nonz_only False`.
- [ ] `--k_prevs` at test/predict **matches** training.
- [ ] Architecture flags at load time are **optional now** — the checkpoint is detected. If you do
      pass one and it contradicts the weights, the script stops with an explicit message.
- [ ] `--convlstm_unet` runs were trained with `--add_temporal`.
- [ ] Relative paths you pass (`--patch_dir`, `--out_dir`) match your working directory.
      The `assets/` files resolve on their own.
- [ ] `--x_pxls_offset` / frame origins consistent with how patches were made.
- [ ] Partition mode is leakage‑free for the claim you're making (see `PROJECT_SUMMARY.md §7.1`).
- [ ] `LOCAL_ENVIRONMENT` is **unset** for real runs.
- [ ] Trailing slash on `--output_polygs_dir`.
