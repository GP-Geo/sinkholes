# Pipeline reference

One document for the whole pipeline: the conventions the science depends on, then each
stage with its command. Companion to the top-level [`README.md`](../README.md)
(orientation + quickstart), [`RESULTS.md`](./RESULTS.md) (what the runs so far actually
scored) and [`MODEL_RUNS.md`](./MODEL_RUNS.md) (the per-run registry). WEXAC paths below
refer to the data host at
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
legacy union-over-stack target). Samples are the **positive patches of the interferogram
being predicted** — the same set a single-frame run sees, so the two differ only in input
depth and their metrics are directly comparable. Predecessors supply context, never
patches of their own; a location missing from any grid in the chain is dropped, since a
stack needs it at every timestep.

**Context windows.** `--context_margin MY MX` (default `0 0`) reads a *large-context*
patch tree and feeds the network `patch_size + 2 x margin` while still supervising,
scoring and reconstructing the centre `patch_size`. `50 50` feeds **300x200** and
supervises the centre **200x100**; the tree is
`data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned`. Everything else is deliberately
unchanged — the grid, the targets, sample selection, ring negatives, the AOI window and
the evaluation protocol — so a `ctx50` run and its plain twin differ in what surrounds
each target and in nothing else. The geometry is written into the checkpoint as
`io_geometry = {"context": [300, 200], "predict": [200, 100]}`, so `eval-scenes`,
`test-patches` and `predict` derive the margin from the weights; passing a
`--context_margin` that contradicts the checkpoint is an error, not a silent override.
`attention-probe` **refuses** a context checkpoint rather than probing it on plain
patches. Costs ~3x the activations (so pair it with `--accum_steps`) and ~3x the host
memory, because the dataset still materialises every sample's pixels.

**Validity channels.** `--treat_nodata_regions` appends one validity map per timestep in
**block layout** `[img_t0..img_tT-1, V_t0..V_tT-1]`, doubling the channel count, and
switches training to the masked loss. The validity channels pass through normalisation
untouched (they must stay strictly {0, 1}).

**LiDAR gating.** Full-scene prediction only runs on tiles that lie entirely inside the
LiDAR coverage polygons (`assets/lidar_mask_polygs.shp`, selected per interferogram via
its `lidar_mask` source id) — **of the current frame AND every predecessor**.

**Checkpoints describe themselves.** Every evaluation command builds its network through
`sinkholes.models.build_from_checkpoint`, which detects the architecture from the
weights (temporal attention: `temporal_attn.*` keys + a config blob; ConvLSTM:
`convlstm.*` keys + a blob carrying hidden size/kernel; `--add_attn`: `attn.*` keys;
AttentionUNet: its deeper `DoubleConv` indices; else plain UNet) and reads the input
channel count off the first convolution. Sequence-model channel counts are **per
timestep**, not per stack. The architecture flags remain as explicit overrides; one
that contradicts the weights is an error. Adding an architecture is one `register()`
entry in `sinkholes/models/factory.py` — no command changes (`sinkholes architectures`
prints the registry). **Registration order is match order**: the temporal-attention
entry sits ahead of the ConvLSTM one because its hybrid variant contains a ConvLSTM
cell, which would otherwise match the wrong detector first.

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
| `--tattn_unet` | `TemporalAttentionUNet` | shared encoder per timestep, causal attention over the bottleneck sequence |

Both sequence models require `--add_temporal` and count channels **per timestep**.

`--convlstm_hidden` (0 = match the 1024 bottleneck) dominates the ConvLSTM's parameter
count and `--convlstm_kernel` must be odd. `--dropout_bottleneck` applies `Dropout2d` to
the final hidden state — after the frames are merged, so it cannot manufacture
frame-to-frame "change" the way a dropout inside the folded `B*T` encoder would. All
three are stored **inside the checkpoint**, so they are never re-specified at evaluation
time; a checkpoint written before 2026-09-08 carries no `dropout_bottleneck` key and the
constructor default reads it back as `0.0`, the network those weights were trained as.

`--tattn_unet` makes the *current* interferogram query its predecessors at each of the
72 bottleneck locations, rather than compressing them into one recurrent state. It is
past-only by construction: a sample is the current frame plus its immediate
predecessors, so there is no future frame in the tensor to leak from. Its knobs:

- `--tattn_dim` (0 = 256) and `--tattn_heads` — the head count also sets how many
  channel groups a fused skip is split into, so it must divide 64/128/256/512.
- `--tattn_layers` — 1 is a single present-queries-past readout. Above 1, the extra
  layers are causal self-attention over the whole sequence, and the lower-triangular
  mask starts to matter (at one layer it is a no-op: the only query is already the
  last position).
- `--tattn_recurrence convlstm` — the hybrid. A ConvLSTM runs first and attention reads
  over **all** its hidden states instead of only the last.
- `--tattn_fuse_skips 0..4` — how many skips are collapsed over time by the attention
  weights, coarsest first, instead of taken from the latest frame. 0 reproduces the
  ConvLSTM's skip contract exactly; 4 also fuses the 200×100 level, where the 12×6
  attention field is upsampled 16×.

**T is not architectural** for either model: the encoder is shared, the ConvLSTM is
unrolled dynamically, and the attention's positional encoding is a parameter-free
function of the offset from the present. A model trained at `--k_prevs 5` runs at
`--k_prevs 10`, which is what `--fallback_replicate` relies on at evaluation time.

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
timestep); `--nonz_only/--no-nonz_only`; `--add_nulls_to_train`.

> ⚠️ **`--train_with_nonz_th --nonz_th N S` does nothing under `preset_by_intf`.** It
> filters the *discovered* interferogram list, but a preset run takes its train/val lists
> straight out of the partition JSON and never reads that list. Label quality on a preset
> partition is a property of the **partition file**, so it is enforced when the file is
> generated: `make-benchmark-partitions --nonz_th 350 200`
> ([`assets/PARTITIONS.md`](../assets/PARTITIONS.md), generation 4).

**Negatives reach the train split only. Validation is positives-only.** That is a
deliberate, permanent property of this pipeline as of 2026-08-20, and it has a known
cost: `val/dice` cannot see the false positives ring negatives exist to suppress — the
2026-08-10 batch moved object F1 by +0.096 while dice sat inside the noise floor. Read
`val/F1` instead when you want a negative-aware number out of `results.csv` (it is
pooled over raw pixel counts, `evaluate.py:300-306`), and settle precision claims at
scene scale with `scripts/eval/run_eval.sh`.

**`--add_val_negatives` — DEPRECATED, do not use.** It put a fixed 1:1 negative set
(ring 1..3, drawn once from the validation interferograms with `--seed`) into the
**validation** split. It was used by the `valneg` reruns and all 14 `clean22` runs, and
it was a mistake: an empty prediction on an empty mask scores dice 1.0
(`losses.py:21`), so at 1:1 the mean becomes roughly `(1 + dice_on_positives)/2` — a
number that ranks nothing and cannot be read against a positives-only run. The pixel-
pooled `val/F1` was already negative-aware, so it bought nothing real.

The flag still parses, and the trainer warns when it is passed. It exists for one
reason: it is part of the strict `dataset` resume fingerprint
(`sinkholes/training/resume.py`), so removing it would make every run trained with it
unresumable — the five `attnfix` runs of 2026-08-19 included. Nothing under `scripts/`
sets it. Delete it, `validation_negatives()`, the `VAL_NEGATIVE_*` constants and
`SubsiDataset(val_negatives=…)` once those runs have landed.

**Loss.** Default: `BCEWithLogits(pos_weight=--pos_w) + soft Dice`. `--seg_loss jaccard`
swaps the region term for soft IoU — the same quantity reparametrised (`J = D/(2−D)`) but
a steeper penalty on the same error, so it trains differently. Binary only; `--classes >1`
keeps Dice. **`train/loss` is not comparable across the two** (Jaccard reads higher at
every identical prediction), so read `train/dice` and `train/jaccard`, which are recorded
every epoch whichever is the objective — the one that is not the objective is computed
under `no_grad` and cannot reach the optimiser.

With `--treat_nodata_regions`: masked BCE (pos_weight fixed at 8.0, deliberately
independent of `--pos_w` — the masked objective was tuned with it) + the masked region
term + a false-positive suppression term inside no-data areas. Validation reports the
*same* objective, so the `train/loss` vs `val/loss` gap is a straight overfitting read.

> ⚠️ **Every `--amp` run before 2026-09-08 optimised BCE alone.** The region term reduced
> in fp16 and overflowed to `inf` above a mean predicted probability of 2.56% at batch
> 128, returning loss exactly 1.0 with zero gradient. Fixed by widening every region term
> to fp32 before it reduces; recorded as `region_loss_dtype=fp32-v2` in `run_config`.
> [`CORRECTNESS_FIXES.md`](./CORRECTNESS_FIXES.md) has the measurement and the list of
> affected runs.

**Optimisation.** `--optimizer rmsprop|adam|adamw`, `ReduceLROnPlateau` on val Dice
(`--lr_patience`, `--lr_factor`, `--min_lr`), gradient clipping at 1.0 **after** AMP
unscaling, `--amp` for mixed precision, `--weight_decay`, and `--accum_steps` (the
effective batch is `batch_size × accum_steps`). Device selection is automatic
(CUDA → MPS → CPU).

- `rmsprop` is the historical default and the only setting the pre-2026-09-08 numbers are
  comparable across. Its momentum buffer is **not** bias-corrected, so the effective step
  is `lr/(1−momentum)` — at the historical `--momentum 0.999` that is a 1000× step
  amplification, survivable only while the AMP clipping defect was crushing every
  gradient. With correct clipping it diverges inside one epoch
  (`outputs/failed_2026-09-07_ampfix_divergence/`).
- `adam`/`adamw` bias-correct, so the step is ~`lr` whatever `--beta1` is. **An `lr`
  carried over from an RMSprop run is not the same step size** — retune it. `adamw`
  decouples weight decay, which is the only form in which `--weight_decay` means what the
  literature means by it. Every batch from 2026-09-09 on runs
  `adamw lr 3e-4 wd 0.01`, `--momentum` is ignored there and a non-default one is refused
  rather than silently dropped.
- `--augment_flips none|h|v|hv` — random flips on the **train** split only; image and mask
  flip together.
- `--dropout_bottleneck` (ConvLSTM only) drops whole channels of the final hidden state,
  the one tensor the whole sequence has been compressed into. `0.0` builds `nn.Identity`
  and is the network exactly as it was before the option existed.

**Outputs** under `outputs/<job>_<ts>/`: `checkpoints/best.pt` + `last.pt` (+ one
`.pth` per epoch unless `--save_best_only`; `interrupted.pt` on Ctrl-C), `results.csv`
(one row per epoch), `curves.png`, `validation/preds/epoch_*.png` (input/GT/probability
grids of the same patches every `--sample_every` epochs — a flipbook of the model
improving; the `--n_samples` patches span the range of ground-truth area in the
validation set, sparsest to densest, and are kept `--sample_min_sep` samples apart so
the grid never shows one sinkhole through several overlapping windows),
`logs/reporter.log`, the run log, and `test_dataset_<job>.pkl` — the
held-out split that feeds `test-patches`. `--patience N` early-stops; `--reporter/
--no-reporter` toggles the console table.

`results.csv` columns, and what each is for:

| column | what it is |
|---|---|
| `train/loss`, `val/loss` | the same objective on both splits — the gap is the overfitting read |
| `train/bce`, `train/dice`, `train/jaccard` | the objective **split into its terms**, which says *which* term a plateau is stuck on. Both region terms are always recorded, so a `dice` arm and a `jaccard` arm are readable against each other |
| `val/dice`, `val/jaccard` | per-sample means (macro). A training-health signal — see the warning above about ranking |
| `val/IoU`, `val/F1`, `val/P`, `val/R` | pooled over raw pixel counts (micro), so negative-aware |
| `grad/norm`, `grad/clip%`, `amp/scale` | what the gradient clip actually did that epoch, and whether AMP was rescaling. A clip% that sits near 100 means the clip, not the LR, is setting the step |

Note `val/F1` pools every pixel (micro) while `val/dice` averages per batch (macro); they
differ by design. `val/jaccard` is the per-sample twin of `val/dice`, and is **not** the
same as `val/IoU`: pooled IoU is exactly `F1/(2−F1)` and carries nothing `val/F1` does
not, whereas the per-sample mean can rank two models differently (`J` is convex in `D`,
so `mean(J) ≠ J(mean(D))`).

**Resuming a preempted job (`--resume auto`).** WEXAC preempts a job with SIGINT/SIGTERM
and later reruns it under the *same* LSF job id. With `--resume auto` the run directory
is `outputs/<job_name>_<timestamp>_lsf_$LSB_JOBID`, where the timestamp is the *first*
execution's: a requeued execution does not recompute that name, it finds the directory
by the job id (unique, and unchanged by a requeue), so `outputs/` still reads by date
while the location stays reproducible. Every completed epoch atomically writes
`checkpoints/resume.pt`: model, optimizer, LR scheduler, AMP scaler, best/early-stopping
state, global step and the Python/NumPy/Torch/CUDA RNG streams. The rerun restores all
of it, continues at the next epoch (with `--epochs` as the *total* target), and appends
to the existing `results.csv`. Settings that define the experiment (architecture,
`k_prevs`, ConvLSTM hidden size, partition, batch size, seed, patch geometry) are
checked against the checkpoint and an incompatible resume is refused. `--resume <dir>`
and `--resume <file.pt>` do the same for an explicit location; an explicitly given
legacy `last.pt` loads *weights only*, with a warning that it is not a resume. Without
`--resume` nothing changes: a fresh `outputs/<job>_<ts>/`. The recovery point is the
last fully completed epoch, so an epoch interrupted part-way is repeated, not resumed
mid-stream. `best.pt` and `last.pt` keep their model-only format.

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

Runs trained with `--partition_mode preset_by_intf` pickle no test split (the whole
`geo_*` / `temporal_*` family), so point the command at the partition instead and it
builds the split in place:

```bash
sinkholes test-patches --partition_file assets/partition_geo_k10.json --split test \
  --patches_dir <patches root> --model outputs/<run>/checkpoints/best.pt \
  --add_temporal --k_prevs 10 --convlstm_unet --patch_size 200 100 --stride 2
```

`--split` is `val`, `test` or `train`; the dataset flags mirror `train.py`'s names so they
can be pasted out of a run's log header. Either way the set is positives-only — see
[POSITIVES_ONLY_EVAL.md](POSITIVES_ONLY_EVAL.md) for what that number may and may not be
compared against.

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
`--no-add_lidar_mask` disables gating. `--positives_only` restricts prediction to tiles
whose ground truth is non-empty — the benchmark paper's protocol, off by default, and not
comparable with a full-scene number ([POSITIVES_ONLY_EVAL.md](POSITIVES_ONLY_EVAL.md)).

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

Re-thresholds the saved confidence maps, computes object-level precision/recall at each
operating point, writes a metrics JSON and, with `--save_figures`, a per-interferogram
overview PNG. All figures render via Agg — no GUI, safe on headless nodes.

Two threshold families, and **they are not comparable with each other**:

| | thresholds | meaning | use on |
|---|---|---|---|
| default | 0.125 / 0.25 / 0.5 / 0.7 / 0.9 | a cut on the **mean probability** | a map from `eval-scenes --recon_average uniform` |
| `--rth` | 0.125 / 0.25 / 0.375 / 0.5 | the **fraction of overlapping tiles** voting positive | a map from `eval-scenes --recon_average vote` |

`--rth` is the benchmark paper's Reconstruction Threshold protocol and is what every
generation-3 evaluation from `eval6` on reports. It is opt-in because on a probability
map the numbers are silently meaningless rather than wrong-looking.

Object matching runs under one or more **Intersection Tolerances** — `--th` (covered
fraction for a GT object to count as detected) with `--buffer` (pixels the prediction is
dilated by). `--rth` sweeps the paper's two, `0.7/5` (primary, and what fills the
top-level `per_intf`/`summary` keys) and the softer `0.5/10`; `--extra_tolerance ITH:BUFFER`
adds them by hand. Everything, primary included, is also reported under
`summary_by_tolerance` keyed `ith0.7_b5` / `ith0.5_b10`.

Crop the canvas with `--aoi_from_partition <partition.json>` (preferred) or
`--aoi_window`; it **must** match the window `eval-scenes` predicted with or the metrics
cover different ground than the predictions. `--legacy_south_half` reproduces the pre-AOI
rule that scored only the northern half of a South-frame canvas, kept so archived
evaluations re-score exactly as they were.

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
- [ ] `--context_margin` is *not* passed at evaluation: it is read off the checkpoint's
      `io_geometry`. Passing one that contradicts the weights is an error.
- [ ] An `lr` copied from an RMSprop preset is the wrong step size under `adam`/`adamw` —
      RMSprop's uncorrected momentum makes its effective step `lr/(1−momentum)`.
- [ ] `--rth` scoring needs a map reconstructed with `eval-scenes --recon_average vote`;
      on a probability map the numbers are meaningless, not merely different.
- [ ] `--train_with_nonz_th` is a no-op under `preset_by_intf`. Filter at partition
      generation time instead.
