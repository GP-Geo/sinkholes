# Plan — train on a large window, predict on the middle

*Written 2026-08-20. **Not started** — this is a design note for a future experiment, not a
record of one. Nothing here changes a finished run or a published number.*

The idea: feed the network a large window (e.g. 600×300) and supervise/predict only the middle
200×100, so the prediction carries spatial context and never sees zero-padding at its own border.
The overlap-tile ("valid region") strategy.

**Settled before starting:** the receptive field is 188–220 px, so context beyond ~110 px per side
is invisible to the centre · 400×300 is the efficient size, 600×400 the thorough one, 600×300 the
unbalanced one · no new patch tree · the dataset has to go lazy first · the geo partitions have no
buffer between splits and a context margin crosses the cut.

---

## 1. Measure first — the receptive field caps the useful margin

Backpropagating one output pixel to the input (untrained nets, eval mode, so it is the
architecture's ceiling and not a training artefact):

| model | receptive field | margin it can use |
|---|---|---|
| `UNet` / `--add_attn` / `--attn_unet` | **188 px** | ±94 |
| `TemporalAttentionUNet`, any T | **220 px** | ±110 |
| `ConvLSTMUNet`, current frame | **220 px** | ±110 |
| `ConvLSTMUNet`, frame of age *a* | 220 + 32·*a* | ±(110 + 16·*a*) |

The ConvLSTM grows because its 3×3 gate convolution sits at the 1/16 bottleneck and spreads one
cell — 16 px — per unroll step. Attention does not: it runs per location, so tattn's field is the
same at T=11 as at T=1.

All three architectures already accept 600×300 unchanged — fully convolutional, `Up` pads back onto
each skip, output shape equals input shape. Nothing in the models blocks this.

**Sizing.** Margins must be multiples of the grid step (100 rows / 50 cols) to be cut out of the
existing `strpp2` grids:

| context | margins | covers | cost |
|---|---|---|---|
| **400×300** | 100 / 100 | U-Net fully; tattn & ConvLSTM-current at 91% | **6×** |
| 600×300 | 200 / 100 | as above, plus ConvLSTM history to age ~5 in rows only | 9× |
| **600×400** | 200 / 150 | tattn & ConvLSTM-current fully, both axes | 12× |

600×300 is over-provisioned on rows (200 px of margin against a 94–110 px reach — the outer
~100 rows per side are literally unreachable) and under-provisioned on columns. Pick 400×300 for
the cheap arm, 600×400 if paying 9× anyway.

**How much is there to win.** Trained `geo_k10_single` U-Net, synthetic wrapped phase, centre
200×100 of a large-window pass against a standalone 200×100 pass: mean |ΔP| = 0.0002, confined to
the outer ~5 px of the patch (0.0016 there, ~0 beyond), max 0.15, 0.01% of pixels crossing 0.5.
Consistent with the effective receptive field — 99% of the gradient mass sits within ±5 px of the
output pixel. Synthetic input and a patch-trained model, so treat it as an order of magnitude, not
a verdict; but the 188/220 px theoretical ceiling is real regardless. **If the goal is for the
model to genuinely use more spatial context, the receptive field has to grow too** — a fifth
down/up level takes it to ~408 px, a dilated bottleneck is cheaper. Enlarging the input alone is
necessary, not sufficient.

## 2. Two experiments, one flag apart

"Train on 600×300, supervise the middle" and "train on 600×300, supervise all of it" are the same
plumbing and buy opposite things:

- **crop the loss** — removes padding contamination, adds context; the margin contributes no
  gradient;
- **do not crop** — 9× the supervised pixels, mostly background, which attacks the scene-scale
  false-positive rate (`MODEL_RUNS.md`: P = 0.12 at confidence 0.125) the way
  `--add_ring_negatives` was reaching for.

Build `--loss_region center|full` so both arms are one flag apart. `center` is the default.

## 3. Design decisions

**No new patch tree.** A 600×300 / `strpp6` tree is 9× the current one (~1 TB → ~9 TB). Cut the
context out of the existing `strpp2` grids instead: at 50% overlap, cell (i, j) covers rows
[100i, 100i+200), so a 600×300 window centred on (i, j) is the 3×3 block (i±2, j±2) and a 400×300
window the 2×3 block (i±1, j±2). Disk, `nonz_indices.json`, `count-positives` and every partition
stay exactly as they are.

**The sample set does not change.** Samples stay the positive cells of the current interferogram,
targets stay 200×100 at cell (i, j). Partitions, AOI windows, the positives-only protocol,
`canvas_shape`, blending and every metric keep their meaning.

**The dataset must go lazy — and it pays for itself.** `_load_temporal` materialises every sample's
pixels today: 65,030 train samples × T=11 × 200×100 × 4 B ≈ **57 GB** resident, which is what
`rusage[mem=128GB]` is for. At 600×300 that is ~515 GB. Dead end. Instead hold **one de-tiled scene
band per interferogram** (from the grid's even cells) plus the coordinate list, and cut the window
in `__getitem__`. A full band is ~358 MB, AOI-cropped ~150 MB; a geo_k10 train split (~150 distinct
interferograms across all chains) is ~22 GB — **less than it uses today**, at any context size,
with no per-sample I/O.

**The crop belongs to the model.** Add a plain `self.predict_size` attribute (not a parameter) to
the four model classes and end each `forward` with a centre crop. State-dict keys are untouched, so
`factory.py`'s detectors and every existing checkpoint keep working. Store the geometry under one
architecture-independent checkpoint key, `io_geometry = {"context": [600,300], "predict":
[200,100]}`, add it to `NON_PARAMETER_KEYS`, and have `build_from_checkpoint` set the attribute —
then `eval-scenes`, `test-patches`, `predict` and the attention probe are correct with no new flag,
which is the contract the rest of the pipeline already holds.

Alignment is exact: pooling floors and `Up`'s asymmetric padding both anchor top-left, so output
pixel (r, c) corresponds to input pixel (r, c) even at sizes not divisible by 16.

## 4. Change list

**New `sinkholes/dataprep/context.py`** — the one place window geometry lives, so training and
inference cannot disagree (same doctrine as `resolve_patch_dirs`):
`context_margin(patch_size, context_size, grid_stride)`, raising unless margins are even and
multiples of the grid step; `detile(grid)` with a round-trip property as its test;
`context_window(source, i, j, ..., pad_value)` padding at grid edges. **Pad with raw 0 before
normalisation and 0.5 after** — both land on the 0.5 no-data code, and `--treat_nodata_regions`
then marks the padding invalid for free.

| file | change |
|---|---|
| `dataprep/dataset.py` | `context_size=`; `_load_temporal` (`:341`) / `_load_single_ring` (`:492`) store bands + coords instead of stacking pixels (`:411`); `__getitem__` (`:619`) cuts the window; relax the shape assert (`:632`); pickle paths + coords, not arrays |
| `geo.py` | `context_grid_window(...)` — centre cells whose whole **context** footprint is inside the AOI box |
| `training/train.py` | `--context_size H W` (default `--patch_size`) at `:111`; thread through `build_datasets` (`:375`); set `predict_size`; write `io_geometry`; banner (`:753`); `--accum_steps`; raise `num_workers` (`:653`) |
| `training/losses.py` | crop `V_any` to `logits.shape[-2:]` (`:88`) — one line |
| `training/evaluate.py` | crop `image` for the sample grid and object features (`:275`); logits arrive cropped |
| `inference/reconstruct.py` | only the `x_np` assembly (`:238`) changes — stamping, canvas, gates, blending and `average='uniform'` stay valid because the output tile is still 200×100 at the same place |
| `inference/{scenes,patch_test,predict,outputs}.py` | geometry from the checkpoint's `io_geometry`, `--context_size` as an override that errors on contradiction |
| `training/resume.py` | `context_size` in `STRICT_CONFIG_KEYS` (`:72`) and `run_config` (`:194`), emitted only when it differs from `patch_size`, so existing checkpoints keep their fingerprint |

**Tests.** New: `detile` round-trip; `context_window` at all four grid edges; centre-of-context
equals the target cell; `context_grid_window` admits no cell whose context crosses 31.4°. Update:
`test_data_layer`, `test_aoi_window_consumers`, `test_grid_window`, `test_convlstm_unet`,
`test_tattn_unet`, `test_factory`, `test_reconstruction`.

**Docs.** A "Context windows" subsection under `PIPELINE.md`'s Conventions plus its (2)/(3b)/(4)
commands; `PRESETS.md`; the margin/gap rule in `assets/PARTITIONS.md`.

## 5. The trap — geo partitions have no buffer

`partition_geo_k10_pre2023.json` puts train at lat 31.4–31.75 and val/test at 31.25–31.4:
**adjacent, no gap**. A 200-row margin is 0.00555° ≈ 617 m, so a training sample centred just north
of the cut takes input imagery from inside the hold-out band. Labels never leak — the target is the
centre only — but this is the same class of problem `_load_spatial` already refuses when it drops
patches straddling the line.

Fix: build the dataset's AOI window with `context_grid_window`, admitting a cell only when its whole
context footprint is inside the split's box. Costs 2 grid rows/cols at each edge of a ~126-row band,
about 3% of the positives. `dataset.py`, `scenes.py` and `outputs.py` must switch together —
`geo.grid_window`'s docstring says what happens if they do not.

## 6. Cost

Baseline: `geo_k10` epochs run 4m45s at BATCH=128 on a 64 GB GPU
(`outputs/2026-08-19/*/results.csv`). FLOPs scale with pixels.

| context | FLOPs | batch that fits ~64 GB | epoch | 60 epochs |
|---|---|---|---|---|
| 200×100 | 1× | 128 | 4m45s | ~5 h |
| 400×300 | 6× | ~20 | ~28 min | ~28 h |
| 600×300 | 9× | ~14 | ~43 min | ~43 h |

Three consequences to decide before submitting anything:

- **Batch size drops ~9×**, and `batch_size` is a STRICT resume key with every preset at 128. Add
  gradient accumulation to hold the effective batch at 128, or the run confounds context with
  optimiser — the confound `PRESETS.md` preset 11 exists to settle.
- **`-W 14:00` no longer fits a run.** `--resume auto` handles it, but expect several requeues.
- **Full-scene eval is 9× too** (~3 h → ~27 h). Valid-only prediction removes the reason for 4×
  overlap averaging, so predicting only even cells with `--recon_average coverage` brings it to
  ~2.25× instead of 9×.

If the 6–9× is prohibitive, the phase-2 optimisation is to crop *early*: run the encoder on the full
window but crop skips and bottleneck to the centre footprint before the temporal stage and decoder.
Levels 1, 1/2, 1/4, 1/8 crop exactly (200, 100, 50, 25 rows); the 1/16 bottleneck lands on 12.5, so
crop it generously and re-crop after `up1`. Roughly halves the total.

## 7. Order

0. **Half a day, no training.** Every checkpoint already runs at 400×300. Take
   `geo_k10_convlstm_ring3_valpos`, feed it real context windows over one test split, compare
   centre-crop predictions against today's 200×100 ones. If object-level F1 barely moves on real
   interferograms the way ΔP barely moved on synthetic, that is known before spending 28 GPU-hours.
   Needs only `context.py` + the `reconstruct.py` change — the first two items of the real work.
1. `context.py` + `geo.context_grid_window` + tests.
2. Lazy `SubsiDataset` — worth doing on its own merits (57 GB → 22 GB).
3. `predict_size` + `io_geometry` + `--context_size` through train/eval/predict.
4. One arm at 400×300 against its 200×100 twin: same partition, same seed, effective batch 128.
