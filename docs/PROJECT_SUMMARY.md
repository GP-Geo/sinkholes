# Dead Sea Sinkhole Detection — Project Summary & Audit

> Semantic segmentation of subsidence (sinkhole precursor) areas in Sentinel‑1 SAR
> interferograms of the Dead Sea coast, using a U‑Net trained on interferogram patches
> with manually mapped subsidence polygons as ground truth.
>
> *This document is a code audit and orientation guide written from a fresh read of the
> repository. It changes no code. For the step‑by‑step command reference, see
> [`PIPELINE_MANUAL.md`](./PIPELINE_MANUAL.md).*

---

## 1. What the project does (in one paragraph)

Sentinel‑1 InSAR **interferograms** (wrapped phase rasters, `.unw`) over the Dead Sea are
cut into fixed 200×100 pixel patches. Manually mapped **subsidence polygons** (GSI
shapefile, `sub_20231001.shp`) are rasterized into binary masks and cut into matching
patches — these are the ground truth. A **U‑Net** (optionally an Attention U‑Net) is trained
to segment subsidence pixels. Because subsidence is rare, training uses only "non‑zero"
patches (those containing ≥1 subsidence pixel) plus sampled "ring" negatives, a
positive‑weighted BCE + Dice loss, and an optional validity mask for no‑data regions.
Evaluation happens at two levels: **patch‑level** (Dice, pixel P/R, object‑level P/R) and
**full‑interferogram** (patches are reconstructed into a whole‑scene prediction, thresholded,
converted to polygons, and restricted to the LiDAR‑covered valid region). A separate
inference path runs a trained model on **new interferograms without ground truth** and exports
predicted sinkhole polygons as shapefiles for GIS use.

**Scale of the data:** `intf_coord.json` catalogs **716 interferograms** (367 North frame,
349 South frame), 2019–2023. Pixel size ≈ 2.777e‑05° (~3 m). Most heavy data
(`.unw`, `.npy` patches, checkpoints) lives on **WEXAC** at
`/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/` and is *not* in the repo
(see `.gitignore`).

---

## 2. Pipeline at a glance

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ WEXAC DATA:  *.unw  (full interferograms)   sub_20231001.shp  (GT polygons)   │
 │              *.ers  (headers)               patches/…            (derived)     │
 └─────────────────────────────────────────────────────────────────────────────┘
        │
  (0)   ├─ prepare_intf_coord_dict.py ──► intf_coord.json   (geo metadata per intf)
        │
  (1)   ├─ prepare_intrfrgrm_pathches.py ─► data_/mask_patches_*  +  nonz_indices.json
        │
  (1b)  ├─ check_patches.py ───────────► fills nonz_num back into intf_coord.json
  (1c)  ├─ clean_patches.py (optional) ─► edge/no‑data‑cleaned patch variants
        │
  (2)   ├─ train_sinkholes_unet.py ────► outputs/<job>/checkpoints/*.pth
        │      architecture:              + test_dataset_<job>.pkl + logs
        │      UNet │ --add_attn │ --attn_unet │ --convlstm_unet
        │
        ├──(3a) test.py ──────────────► patch‑level metrics (Dice, P/R, OL P/R)
        │
        ├──(3b) test_full_intf.py ────► outputs/predictions/<model>/<job>/*_image/_pred/_gt.npy
        │           │                    + polygs/*.shp  (+ combined shp)
        │           └─ evaluate_full_intf_output.py ─► metrics JSON + paper figures
        │
        └──(4)  predict_new_intf_vA.py ► out_polygs/*.shp   (inference on new data)

  Stages 3a/3b/4 detect the architecture from the checkpoint (models/factory.py),
  so all of them work with any of the four architectures above.
```

---

## 3. File‑by‑file map

### Models
| File | Role | Notes |
|---|---|---|
| `unet_parts.py` | `DoubleConv`, `Down`, `Up`, `OutConv` building blocks | Standard Pytorch‑UNet parts. |
| `unet.py` | `UNet` (n_channels→n_classes) + 3 self‑attention modules | See **§6.1** — attention wiring is buggy/dead; `HistoryGate`/`input_mix` are commented out. |
| `attn_unet.py` | `AttentionUNet` with attention gates on skip connections + dropout | Selected with `--attn_unet`. Self‑contained (redefines `DoubleConv` etc. with dropout). |
| `convlstm_unet.py` | `ConvLSTMUNet` — shared encoder run per timestep, ConvLSTM over the bottleneck sequence | Selected with `--convlstm_unet` (requires `--add_temporal`). Best score to date; see `TRAINING_RUNS.md`. `CHANGELOG.md #6`. |
| `factory.py` | Checkpoint → model. Detects architecture from the weights and rebuilds it | The single construction path for all five inference scripts. Add new architectures here — no call‑site edits. `CHANGELOG.md #12`. |

### Data
| File | Role | Notes |
|---|---|---|
| `prepare_intf_coord_dict.py` | Parse `.ers` headers → `intf_coord.json` (origin, dx/dy, size, byte order, frame, LiDAR mask id) | Run once. |
| `prepare_intrfrgrm_pathches.py` | Rasterize GT polygons, align frames, cut `data`/`mask` patches (+ `nonz` subset) | The core patch generator. `patchify()` is the reusable function. |
| `check_patches.py` | Count non‑zero patches per intf, write `nonz_num` into `intf_coord.json` | Must run **after** patchifying — training filters on `nonz_num`. |
| `clean_patches.py` | Drop mask polygons that are mostly no‑data or hug patch edges → `cleaned/` variants | Optional; used with `--use_cleaned_patches`. |
| `sinkholes_data_loading.py` | `SubsiDataset` (PyTorch `Dataset`) + preprocessing | The heart of data handling; see **§5**. |
| `create_intf_partition.py` | Build a preset train/val split file (`partition_*.json`) | **Has a runtime bug — see §6.2.** |
| `lidar.py` | Merge per‑year LiDAR polygons into `lidar_mask_polygs.shp` | One‑off; needs the per‑year shapefiles (not in repo). |
| `untie_mask_polygs.py` | (empty helper — placeholder) | — |

### Training / metrics
| File | Role | Notes |
|---|---|---|
| `train_sinkholes_unet.py` | Training driver, 4 partition modes, masked/unmasked losses | ~30 CLI flags; see **§4** and **§6**. |
| `dice_score.py` | `dice_coeff`, `dice_loss` | Standard soft Dice. |
| `evaluate.py` | `evaluate()` val/test loop + `object_level_evaluate()` + phase‑feature functions | Large (770 lines); much is plotting + candidate object‑features (FFT ratio, radial symmetry, phase gradient). |

### Inference / full‑scene evaluation
| File | Role | Notes |
|---|---|---|
| `test.py` | Patch‑level test from a saved test‑set pickle | Thin wrapper around `evaluate(..., mode='test')`. |
| `test_full_intf.py` | Reconstruct full‑scene predictions from **aligned patches**, LiDAR‑gate, polygonize | Main by‑intf evaluation path. |
| `evaluate_full_intf_output.py` | Post‑hoc metrics + paper figures from saved `.npy` outputs | Lots of dead plotting code; **`plot=True` hardcoded** (§6.3). |
| `predict_new_intf.py` | Inference on raw `.unw` (v1) | **Has a LiDAR‑index bug + path bug — prefer vA.** |
| `predict_new_intf_vA.py` | Inference on raw `.unw` (v2, refactored) | Cleaner reconstruction; **fixes** the LiDAR bug. Recommended. |
| `inspect_run.py` | Learning curve + decision‑threshold sweep for a finished run | Falls back to `best.pt`/`last.pt` when a run has no per‑epoch checkpoints. |

All five of the above build their network through `models/factory.py`, which **detects the
architecture from the checkpoint weights** — so every one of them handles U‑Net, Attention U‑Net
and ConvLSTM alike, with no architecture flag required (`CHANGELOG.md #12`). Before that fix, all
but `test.py` were hardcoded to U‑Net/AttentionUNet and could not load a ConvLSTM checkpoint.
| `get_intf_info.py` | `get_intf_coords`, `find_11day_sequences`, `crop_to_start_xy`, `build_common_grid_for_region` | Shared utilities. **Runs code on import — see §6.4.** |
| `polygs.py` | `mask_array_to_polygons`, `plg_indx2longlat` (pixel→lon/lat) | Georeferencing of predicted polygons. |

### Data artifacts committed to the repo
| File | Role |
|---|---|
| `intf_coord.json` (194 KB) | Per‑interferogram geo/metadata dictionary (716 entries). |
| `partition_20_05_*.json` | Saved train/val splits. |
| `lidar_mask_polygs.{shp,shx,dbf,cpg}` (37 MB) | LiDAR coverage polygons = valid‑region mask for prediction. |
| `lidar_intf_mask.txt` | Maps each intf → its LiDAR mask id. |

---

## 4. Key concepts you'll meet in the code

- **Frame (North / South):** the coast is imaged by two Sentinel‑1 frames with slightly
  different origins. Alignment crops each frame to a fixed origin so patches from different
  dates line up. North → origin `(35.37, 31.79)`, South → `(35.32, 31.44)` in
  `prepare_intrfrgrm_pathches.py` / `test_full_intf.py`. *(Note: `evaluate_full_intf_output.py`
  uses slightly different constants `35.3 / 35.25` — see §6.5.)*
- **11‑day difference:** the pipeline works with 11‑day interferograms (Sentinel‑1 repeat).
  `find_11day_sequences()` builds, per interferogram, its chain of *k* previous 11‑day
  interferograms **of the same frame**, used for temporal stacking.
- **Non‑zero (`nonz`) patches:** patches whose GT mask contains ≥1 subsidence pixel. Training
  defaults to these only (`--nonz_only True`) to fight extreme class imbalance.
- **Ring negatives:** all‑zero patches sampled in an annulus *around* positive patches, added
  to training so the model sees hard negatives near true subsidence (`--add_ring_negatives`).
- **Temporal context (`--add_temporal`, `--k_prevs`):** stacks the *k* previous interferograms
  as extra input channels; GT becomes the **union** of masks across the stack. Model input
  channels = `k_prevs + 1` (×2 if `--treat_nodata_regions` appends a validity channel per time).
- **No‑data / validity (`--treat_nodata_regions`):** InSAR scenes have masked/zero pixels. The
  code builds a validity mask `V` and uses masked BCE + masked Dice, plus a penalty that
  suppresses false positives in no‑data areas.
- **Reconstruction:** to score a whole interferogram, overlapping patch predictions
  (`--data_stride`) are stitched back into a full map, either simple‑averaged or **Hann‑window
  blended** (`--blend_type hann`), then thresholded at `--recon_th` (default 0.25).

---

## 5. `SubsiDataset` — how a training sample is built

`sinkholes_data_loading.py` is the most complex file; understanding it is key.

1. **Which patch files load** depends on the interaction of `nonz_only`, `partition_mode`,
   `add_temporal`, `add_nulls_to_train`, `nonoverlap_tr_tst` — the prefix is either
   `data_patches_nonz_*` or `data_patches_*`.
2. **Temporal path** (`add_temporal`): loads current + *k* previous patch grids `(ny, nx, H, W)`,
   clips them to a common grid, takes the **union of non‑zero patch coordinates** across time
   (`nonz_indices.json`), optionally adds **ring negatives**, then stacks into `(T, N, H, W)`
   with a unioned mask `(N, H, W)`.
3. **Spatial path** (`partition_mode='spatial'`): splits each interferogram by a latitude line
   (`--thresh_lat`) into train (north of line) vs val/test (south) — a within‑scene geographic
   hold‑out.
4. **`preprocess()`**: for images, per‑channel — if values look like radians (outside `[0,1]`)
   it maps `(φ + π)/(2π) → [0,1]`; otherwise it replaces exact zeros with `0.5` (treated as
   no‑data). For masks, maps values to class indices via `mask_values`.
5. **`__getitem__`** returns `{"image": (1 or T, H, W) tensor, "mask": (H, W) long tensor}`.

> ⚠️ **Scientific caveat (see §7):** the `0.5` substitution conflates true zero‑phase (φ=0) with
> no‑data, and the train‑time normalization is a *per‑patch heuristic* while the predict scripts
> normalize *unconditionally* — a subtle train/inference mismatch.

---

## 6. Bugs & landmines found in the audit

These are concrete issues surfaced while reading. **Nothing was changed** — listing them so you
can decide. Ordered by how likely they are to bite.

### 6.1 U‑Net attention wiring is dead / self‑overwriting (`unet.py`)
```python
if self.add_attn:
    self.attn = SelfAttention2D(1024)
    self.attn = MultiHeadSelfAttention2D(1024,8)
    self.attn = ChannelSelfAttention(in_channels=1024)   # only this one survives
```
The first two assignments are immediately overwritten, so `--add_attn` **always** uses
`ChannelSelfAttention` (which pools away all spatial information at the bottleneck). The other
two modules and `HistoryGate`/`input_mix` are dead code. If you meant to compare attention
variants, this needs a selector.

### 6.2 `create_intf_partition.py` crashes on run
```python
parser.add_argument('--eleven_days_20', ...)      # defines _20
...
args.eleven_days_21 = str2bool(args.eleven_days_21)  # reads _21  → AttributeError
```
Also the path line `... + '_11days/' if args.eleven_days_20 else '/'` has operator‑precedence
that collapses the whole path to `'/'` in the false branch. This script cannot run as‑is.

### 6.3 `evaluate_full_intf_output.py` forces plotting on
`plot = True` is hardcoded (~line 600), overriding the parsed `--plot`. Two of three plot
functions and `plot_prediction_hist_all` are **dead code**; `plot_full_intfs_preds_vprevs_wlidarmask`
references an undefined `x3000` and would `NameError` if called. No `if __name__ == '__main__'`
guard — importing the module runs the whole evaluation.

### 6.4 `get_intf_info.py` executes work on import
```python
intf_coords = json.load(open('intf_coord.json'))
new_grid = build_common_grid_for_region(intf_coords,'North')
print(new_grid)          # runs at import time, prints, needs intf_coord.json in CWD
```
Because most scripts do `from get_intf_info import *`, this runs on **every** invocation and
hard‑requires `intf_coord.json` in the working directory. Also `get_intf_coords()` re‑opens and
re‑parses the 194 KB JSON on **every call**.

### 6.5 Two `get_intf_coords`, two different tuple layouts
`get_intf_info.get_intf_coords` returns a **12‑tuple** (lidar at index 8, frame at 11), but
`predict_new_intf.py` / `predict_new_intf_vA.py` define their **own** 10‑tuple version (lidar at
index 6). This mismatch is the root of:
- **`predict_new_intf.py` LiDAR bug:** uses `intf_coords[8]` (which is `byte_order`, not the LiDAR
  id) as the fallback mask source. Only masked because the caller passes the source list
  explicitly. **`predict_new_intf_vA.py` fixes this to `[6]`** → prefer vA.

Frame‑origin constants also differ between `test_full_intf.py` (`35.37/35.32`) and
`evaluate_full_intf_output.py` (`35.3/35.25`). Worth reconciling so figures and metrics use one
grid.

### 6.6 Training OOM fallback is broken
```python
except torch.cuda.OutOfMemoryError:
    ...
    train_model(model=model, ...)   # missing required first positional arg `args` → TypeError
```
The recovery path itself crashes. (Also `train_model` reads module globals `outpath`,
`is_running_locally`, `dir_checkpoint`, `dir_validation` that only exist when run as `__main__`.)

### 6.7 "`or True`" flags that are silently always‑on (`train_sinkholes_unet.py`)
```python
if args.add_temporal or True:   # the `or True` makes --add_temporal a no‑op here
```
Several such conditions mean the "Aligned" patch dirs and temporal‑list filtering are **always**
applied regardless of the flag. Confusing when reading the CLI.

### 6.8 Local‑debug shortcuts override user input
`LOCAL_ENVIRONMENT` is read in two files and does **two unrelated things**. It is only tested for
truthiness, so *any* non‑empty value — including `0` and `false` — enables it; unset it to disable.

- `train_sinkholes_unet.py:521` sets `is_running_locally`, then:
  - `:251-254` (`--partition_mode random_by_intf` only) silently replaces your train/val/test
    lists with `train=['20190204_20190215']`, `test=['20190205_20190216']`,
    `val=['20190216_20190227']` — overriding `--preset_test_val_21` as well;
  - `:293-294` (`--partition_mode spatial` only) replaces `intf_list` with `['20191129_20191210']`;
  - `:434` passes it to `evaluate(..., is_local=...)`, where it gates plotting together with
    `plot` (`evaluate.py:412`).
  - `random_by_patch` and `preset_by_intf` are **unaffected**.
- `test_full_intf.py:282-285` inverts the meaning: when *unset*, it forces `args.plot = False`.
  Setting the variable is what *enables* `--plot` there.

⚠ Enabling plotting on macOS crashes: `evaluate.py:64` and `test_full_intf.py:280` assign
`plt.rcParams['backend'] = 'Qt5Agg'` at import time — an rcParams write that `MPLBACKEND` cannot
override — and PyQt5 is usually absent. Never pair `LOCAL_ENVIRONMENT` with `--plot` locally.

Full walkthrough of running this locally: [`LOCAL_WALKTHROUGH.md`](./LOCAL_WALKTHROUGH.md).

### 6.9 Output path/typo issues in `predict_new_intf*.py`
Output is `args.output_polygs_dir + intf + "_predicted_polyogns.shp"` — no separator (default
`"./out_polygs"` has no trailing `/`, so files land as `./out_polygsYYYYMMDD…`) and "polyogns"
is misspelled.

### 6.10 `evaluate.object_level_evaluate` precision defined twice
```python
ol_intersection_precision = round(pred_intersection_area / (total_pred_area+eps),2)
ol_intersection_precision = round(intersection_area / (intersection_area + fp_area+eps),2)  # this wins
```
Harmless (second wins) but the first line is dead and misleading.

### 6.11 Repo hygiene
- The 37 MB `lidar_mask_polygs.shp` is committed to git — it bloats history. Consider Git‑LFS or
  keeping it on WEXAC.
- No `requirements.txt` / `environment.yml`. Dependencies (`torch, torchvision, numpy, pandas,
  matplotlib, rasterio, geopandas, shapely, scikit-image, scipy, albumentations, affine, tqdm`)
  are implicit.
- Duplicated imports and commented‑out augmentation blocks throughout.

---

## 7. Scientific / methodological suggestions

These are the higher‑leverage items for the *research*, roughly in priority order. They are
suggestions for discussion with your advisor, not prescriptions.

1. **Make the evaluation split leakage‑free and report it as the headline.**
   With stride‑2 patches (50 % overlap) and the same sinkhole recurring across many dates,
   `--partition_mode random_by_patch` leaks train into test and will give optimistically high
   metrics. `random_by_intf` removes patch overlap but **not** temporal correlation (adjacent
   11‑day interferograms of the same location). The most defensible headline numbers come from a
   **geographically disjoint** hold‑out (`spatial`) and/or a **temporal** hold‑out (train
   2019–2020, test 2021+, which `--preset_test_val_21` approximates). Recommend reporting the
   temporal/spatial split as primary and `random_by_intf` as secondary.

2. **Respect the wrapped‑phase discontinuity.**
   Interferometric phase is *wrapped* to `(−π, π]`; mapping it linearly to `[0,1]` puts a hard
   seam between physically adjacent values (just below +π and just above −π). Encoding phase as
   two channels **(sin φ, cos φ)** removes the discontinuity and is a natural InSAR‑aware input.
   Adding **coherence** as an input channel (where available) typically helps a lot, since
   subsidence signal quality tracks coherence.

3. **Separate "no‑data" from "zero phase" cleanly.**
   The `0.5` substitution overloads φ=0 and no‑data. The `--treat_nodata_regions` validity‑mask
   path is the principled fix — consider making it the default and dropping the `0.5` trick when a
   validity channel is present, so the two code paths don't optimize different objectives.

4. **Unify and expose the loss.**
   In the masked branch `pos_w` is hardcoded to `8.0` (ignoring `--pos_w`), and constants
   `lam=0.2`, `r_tol=2` are inline. The masked vs unmasked branches implement *different*
   objectives, which makes ablations hard to interpret. Recommend one configurable loss.
   Given the strong imbalance and that detection recall usually matters most for sinkhole
   early‑warning, a **Tversky / focal‑Tversky loss** (tunable FP↔FN) is worth trying against BCE+Dice.

5. **Report threshold‑free metrics.**
   `recon_th=0.25` is a fixed, low operating point (high recall, low precision). Report a
   **precision–recall curve and average precision** (pixel‑level) and an **object‑detection
   curve** (objects matched by IoU across confidence thresholds), then *calibrate* the operating
   threshold on validation and justify it. Break results down **by frame (N/S)** and **by year**,
   since their statistics differ.

6. **Fix reproducibility.**
   `random.shuffle` in `random_by_intf` and `random.seed()` (no arg) in the partition script make
   splits non‑deterministic. Add a single `--seed` that seeds Python/NumPy/PyTorch, log it, and
   save the resolved config + split alongside each checkpoint. Essential for a thesis.

7. **Automate model selection.**
   Checkpoints are saved every epoch and you pick one by hand (e.g. `checkpoint_epoch28.pth`).
   Track the **best‑validation** checkpoint automatically (and optionally early‑stop), and
   report which epoch/threshold produced the reported numbers.

8. **Consider a stronger temporal representation.** *(partly done — see below)*
   Channel‑stacking previous interferograms discards temporal ordering. Independent 11‑day
   interferograms are also not a coherent deformation time series. If the science allows, feeding
   a **cumulative / time‑series InSAR product (SBAS or PS displacement)** as input — rather than,
   or in addition to, single interferograms — is the standard approach for subsidence mapping and
   is likely a large accuracy gain.

   The "lighter middle ground" suggested here — a ConvLSTM over the stack — **was implemented**
   (`convlstm_unet.py`, `CHANGELOG.md #6`) and is now the best-scoring model: val dice 0.591 vs
   0.565 for the single-frame baseline, while channel-stacking into a plain U‑Net scored only
   0.342. That confirms the diagnosis in this item — ordering matters — but the splits differ per
   run, so see `TRAINING_RUNS.md` before treating the ranking as settled. The
   time‑series‑product idea above is still open and remains the larger expected gain.

9. **Quantify label uncertainty.**
   GT masks come from manual polygons matched to interferograms by exact `start_date/end_date`;
   any mismatch yields an empty mask. Consider auditing coverage (how many intfs have GT), using a
   tolerance buffer at boundaries, and reporting sensitivity of metrics to the buffer/overlap
   thresholds (`--b`, `--th`).

---

## 8. Engineering / design suggestions (secondary)

- **Add `requirements.txt`** (pin `torch`, `rasterio`, `geopandas`, `shapely`, `scikit-image`,
  `scipy`, `albumentations`). Geospatial stacks break easily across versions.
- **Collapse `predict_new_intf.py` into `predict_new_intf_vA.py`** (vA fixes the LiDAR bug and has
  cleaner reconstruction) and delete the buggy one, or clearly mark v1 deprecated.
- **Fix the crash bugs** in §6.2 and §6.6 even if those paths are rarely hit.
- **Move `get_intf_info.py`'s import‑time code under `if __name__ == '__main__':`** and cache the
  parsed `intf_coord.json` (load once, pass around) — removes a hidden CWD dependency and repeated
  194 KB parses.
- **Config object over 30 flags.** A YAML/dataclass config that is saved per run would tame the
  `nonz_only × partition_mode × add_temporal × treat_nodata` interactions and kill the `or True`
  short‑circuits.
- **Add a couple of unit tests** for the georeferencing math — `crop_to_start_xy`,
  `plg_indx2longlat`, and a `patchify → reconstruct` round‑trip. A wrong `x_pxls_offset` silently
  shifts every exported polygon, which is the highest‑risk class of bug here and the hardest to
  notice visually.
- **Document the WEXAC data layout** (paths, which products, how to fetch) in the README so future
  experiments can pull the right subset — you flagged this as important.

---

## 9. Quick orientation for a new reader

If you have 20 minutes, read in this order:
1. `README.md` → this document → `PIPELINE_MANUAL.md`.
2. `prepare_intrfrgrm_pathches.py` — how raw `.unw` + GT polygons become patches (`patchify`).
3. `sinkholes_data_loading.py::SubsiDataset` — how a sample is assembled (§5).
4. `train_sinkholes_unet.py::train_model` — the four partition modes and the two loss branches.
5. `evaluate.py::evaluate` + `object_level_evaluate` — how metrics are computed.
6. `test_full_intf.py::reconstruct_intf_prediction` + `polygs.py` — patches → full scene →
   georeferenced polygons.

---

*Audit performed on a clean read of branch `guy_branch`. No source files were modified.*
