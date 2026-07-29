# Change log

Every edit to source files in this repo, with the reason. This is a shared repo: code changes
need a stated justification and a way for the next person to reverse them.

**How to use this file**

- One numbered entry per logical change, newest last.
- Each entry: date, files touched, what changed, **why**, and the blast radius on WEXAC.
- Every edited line carries an inline `# EDIT <date>: ... CHANGELOG.md #<n>` comment pointing back
  here, so the reason is visible where the code is read.
- New files are listed too, but they are additive and carry no risk to existing runs.

Read alongside [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) (audit, known bugs) and
[`LOCAL_WALKTHROUGH.md`](./LOCAL_WALKTHROUGH.md) (running it locally).

---

## Entries

### #0 — Pre-existing working-tree changes (before this log started)

*Author: not recorded. Documented retroactively 2026-07-27 so the diff is accounted for.*

| File | Change |
|---|---|
| `unet.py` | Imports rewritten: `from unet_parts import *` → explicit `DoubleConv, Down, Up, OutConv`; added explicit `torch.nn` / `torch.nn.functional` imports; dropped an unused `matplotlib.pyplot` import. Behaviour unchanged. |
| `evaluate.py` | Added the missing `import matplotlib.pyplot as plt`. The module used `plt` at line 64 and in its plotting helpers without importing it, so any `--plot` path raised `NameError`. |

**Why:** both are import hygiene fixes. Neither alters numerics.

---

### #1 — Apple Silicon (MPS) support

*2026-07-27*

**Files**

| File | Change |
|---|---|
| `device_utils.py` | **New.** `get_device()` and `memory_format_for()`. Also documents what `channels_last` is. |
| `train_sinkholes_unet.py` | `:541` device selection → `get_device()`; `:556` model layout and `:364` batch layout → `memory_format_for(device)`; import added. |
| `evaluate.py` | `:699` batch layout → `memory_format_for(device)`; import added. |
| `test.py` | `:65` device selection → `get_device()`; import added. |
| `test_full_intf.py` | `:330` device selection → `get_device()`; `:173` batch layout → `memory_format_for(device)`; import added. |

**What changed**

1. Device selection was hardcoded in four places as
   `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, so Apple Silicon machines
   always ran on CPU. `get_device()` inserts MPS between CUDA and CPU.
2. `channels_last` was applied unconditionally to the model and to every input batch. MPS cannot
   autograd through that layout — the backward pass raises
   `RuntimeError: view size is not compatible with input tensor's size and stride`.
   `memory_format_for(device)` returns `channels_last` on CUDA and CPU, and the default layout on
   MPS.

**Why**

Local development on a Mac was an order of magnitude slower than it needed to be. Measured on this
UNet at 200×100 (forward + backward + optimizer step):

| device | memory format | bs=1 | bs=4 | bs=8 |
|---|---|---|---|---|
| CPU | `channels_last` | 9.5 img/s | — | — |
| CPU | default | 9.5 img/s | 12.1 img/s | 13.2 img/s |
| MPS | `channels_last` | ❌ RuntimeError in backward | ❌ | ❌ |
| MPS | default | **69.6 img/s** | **113.5 img/s** | **125.4 img/s** |

`channels_last` made no measurable difference on CPU here (105 vs 109 ms/step at bs=1), so nothing
is lost by dropping it on MPS. The speedup is 7.3× at batch 1 and **9.5× at batch 8** — the gain
grows with batch size because at batch 1 Python and dataloader overhead dominate. Use
`--batch_size 8` on MPS to actually see it.

**Verified end to end.** A full run over the 30 local interferograms (24 train / 3 val / 3 test,
8207 / 680 / 938 patches) at `--batch_size 8 --k_prevs 0`: `INFO - Using device mps`, **112 img/s**
sustained in the real training loop, dataset construction + 1 epoch + validation in **98 s**,
validation Dice 0.282, checkpoint written. The same epoch on CPU is ≈ 10 min.

The codebase had already anticipated MPS: `train_sinkholes_unet.py:376` and `evaluate.py:693`
both contain `torch.autocast(device.type if device.type != 'mps' else 'cpu', ...)`. That special
case was unreachable because no code path ever produced an MPS device.

**Blast radius on WEXAC: none.** On a CUDA host `torch.cuda.is_available()` is true, so
`get_device()` returns `cuda` and `memory_format_for()` returns `channels_last` — byte-identical
to the previous behaviour. The new branches are only reachable on a machine with MPS and no CUDA.

**To revert:** restore the four `torch.device('cuda' if ... else 'cpu')` lines and the four
`memory_format=torch.channels_last` arguments, drop the imports, delete `device_utils.py`.

**Not changed, deliberately:** `predict_new_intf_vA.py:423` / `:356` still hardcode CUDA-or-CPU and
`channels_last`. That script cannot run locally anyway (it needs raw `.unw` rasters) and it has an
unresolved normalization bug (`PROJECT_SUMMARY.md` §6.5, `LOCAL_WALKTHROUGH.md` §1), so it was left
alone rather than half-migrated.

---

### #2 — Local subset preparation tooling

*2026-07-27*

**Files**

| File | Change |
|---|---|
| `prepare_local_subset.py` | **New.** Derives `*_nonz_*` patch files and a corrected metadata dictionary from an already-downloaded patch subset. |
| `.gitignore` | Added `/outputs/`, `/pred_outputs2/`, `/test_data/`, `/out_polygs/`, `/data/`, `.DS_Store`, `__pycache__/`. |

**Why**

The downloaded patch subset contains the full `(ny, nx, H, W)` grids but not the positive-only
`(N, H, W)` `nonz` subsets. `SubsiDataset` requires the latter in **every** mode, because
`sinkholes_data_loading.py:82` globs the `mask_patches_nonz_` prefix unconditionally, regardless of
`--nonz_only`. Without them the dataset raises `IndexError` before training starts.

The `nonz` files are pure re-indexing of data already on disk — `patchify` produced them by
selecting the grid cells listed in `nonz_indices.json` — so no raw `.unw` or ground-truth shapefile
is needed to reconstruct them.

Separately, `nonz_num` in `intf_coord.json` came from a different patchify run and disagrees with
the downloaded patches (e.g. 96 vs the actual 87 for `20190204_20190215`). Rather than run
`check_patches.py`, which rewrites `intf_coord.json` **in the working directory**, the script emits
`data/metadata/intf_coord_local.json` with counts corrected for the local interferograms and
`'none'` for every interferogram not downloaded. Since training skips `nonz_num == 'none'`
(`train_sinkholes_unet.py:139-142`), that single file restricts any run to exactly what is on disk.

**Blast radius: none.** No existing source file is modified. The script reads the download
read-only and writes only under `--out_dir` / `--out_intf_dict`.

**`.gitignore`:** a single local run produces ~1.5 GB of checkpoints, pickles, reconstructions and
shapefiles. Without these entries `git add .` stages them.

---

### #3 — CLI training reporter, validation metrics, run inspection

*2026-07-27*

**Files**

| File | Change |
|---|---|
| `train_reporter.py` | **New.** `setup_logger` / `banner` / `EpochTable` / `ResultsCSV` / `BestTracker` / `quiet_root_console` / `human_time`. Stdlib only apart from an optional `tqdm` import. |
| `inspect_run.py` | **New.** Learning curve + best epoch + test-set threshold sweep for a finished run. |
| `train_sinkholes_unet.py` | Reporter wired into `train_model`; `--reporter` and `--patience` flags; `tqdm` total fixed; `best.pt`; `KeyboardInterrupt` handling; closing summary. |
| `evaluate.py` | `evaluate(..., metrics_out=None)` — optional dict receiving pixel precision/recall. |
| `get_intf_info.py` | `:219` `print(new_grid)` commented out. |

**What changed**

1. **Console output.** Training printed one `logging.info` block *per step* — 964 per epoch,
   19 280 for a 20-epoch run — which is unreadable and made the 988 KB run log. The reporter
   replaces the console with a fixed-width per-epoch table (`epoch / time / train/loss / val/dice
   / val/P / val/R / lr`), a resolved-config banner, and a closing summary.

   The root logger's **FileHandler is untouched**; only its *console* handler is raised to
   WARNING (`quiet_root_console`). `outputs/<job>/<job>_<ts>.log` therefore keeps exactly the
   content it had before, so `grep "Validation Dice"` and every existing log-scraping habit still
   work. The reporter additionally writes `outputs/<job>/logs/reporter.log` at DEBUG.

2. **`results.csv`** — one row per epoch, same columns as the table, reopened and closed per row
   so a Ctrl+C'd run keeps its history.

3. **`best.pt`** — `BestTracker` on `val/dice`. The script previously saved one checkpoint per
   epoch (124 MB each) with **no** record of which was best; picking the last epoch was the
   obvious-but-wrong default. Measured on the 20-epoch run of 2026-07-27: best was epoch 17
   (0.5830), last was epoch 20 (0.5621) — taking the last cost 0.021 Dice.

4. **`--patience N`** — early stop after N epochs without improvement. Default `0` = off, so
   behaviour is unchanged unless asked for.

5. **Validation precision/recall.** `evaluate()` only computed P/R in `mode='test'`. It now
   accepts an optional `metrics_out` dict and fills in pixel TP/FP/FN plus precision/recall.
   Counts are accumulated globally rather than averaged per batch, which avoids the
   division-by-zero `calc_precision_recall` hits on an all-negative batch. The **return value is
   unchanged**, so `test.py` and `test_full_intf.py` are unaffected.

6. **`tqdm` total fixed.** `tqdm(total=n_train)` was wrong under `--partition_mode random_by_intf`,
   where `n_train` holds an *interferogram* count: the bar read `0/24` while counting to 7705, so
   the percentage was meaningless. Now `len(train_set)`.

7. **`KeyboardInterrupt`** saves `checkpoints/interrupted.pt` and prints the summary instead of
   dumping a traceback.

8. **Warnings** on non-finite loss (once per epoch), an empty validation split, and CPU fallback
   when no accelerator is present.

9. **`get_intf_info.py:219`.** `print(new_grid)` ran at *module import*, so every DataLoader
   worker re-printed the North grid tuple — 15+ lines of noise before any training output.
   `new_grid` is read nowhere else in the repo (`grep -rn new_grid *.py` returns only its own
   assignment and the print), so only the print is commented out.

**Why**

The previous output made it impossible to see a run's trajectory without post-processing the log,
and impossible to know which checkpoint to keep. Points 3, 5 and 6 are correctness issues rather
than cosmetics.

**Blast radius on WEXAC**

- `--reporter False` restores the previous console output **exactly** — no banner, no table, no
  `results.csv`, no `best.pt`, root console handler left at INFO, and `metrics_out=None` so
  `evaluate` takes its original path.
- With the reporter on (the default), the per-run `.log` file is byte-compatible; only the
  terminal differs. New files (`results.csv`, `best.pt`, `logs/reporter.log`) are additive.
- `evaluate()`'s signature gained a trailing keyword argument with a `None` default; all three
  existing callers are positional-compatible and unchanged in behaviour.
- `best.pt` adds one 124 MB write per improving epoch.

**To revert:** pass `--reporter False`, or drop the `train_reporter` import and the blocks marked
`CHANGELOG.md #3` in `train_sinkholes_unet.py`, remove `metrics_out` from `evaluate.py`, restore
`print(new_grid)`, and delete `train_reporter.py` / `inspect_run.py`.

---

### #4 — Best-weight guarantees and early stopping, decoupled from the reporter

*2026-07-27*

**Files**

| File | Change |
|---|---|
| `train_sinkholes_unet.py` | `BestTracker` built unconditionally; `last.pt`; `--save_best_only`; best weights restored into `model` at end of run. |

**What changed**

1. **Best tracking and `--patience` no longer require `--reporter`.** #3 built the tracker inside
   the `if rep is not None:` block, so `--reporter False` silently disabled both `best.pt` and
   early stopping. A run needs its best weights regardless of how the console is formatted. The
   tracker is now created unconditionally and messages fall back to `logging` when the reporter is
   off.

2. **`checkpoints/last.pt`** — always mirrors the most recent epoch, so `best.pt` / `last.pt` are
   the two files that matter (the Ultralytics convention). Previously the "last" checkpoint was
   only findable by sorting the per-epoch filenames.

3. **`--save_best_only`** — skips the per-epoch 124 MB dump and writes only `best.pt` + `last.pt`.
   A 20-epoch run drops from 2.5 GB to 248 MB. Default **off**, so existing behaviour is unchanged
   unless asked for.

4. **Best weights restored into `model` before `train_model` returns.** Previously the in-memory
   model held the *final* epoch's weights. On the 20-epoch run of 2026-07-27 that was epoch 20
   (0.5621) rather than epoch 17 (0.5830) — 0.021 Dice worse. `mask_values` is stripped before
   `load_state_dict` since it is not a parameter.

**Why**

"The weights this run produced" should mean the best ones, not the last ones. #3 made the best
epoch *visible*; this makes it the **default artifact**. The per-epoch files remain available for
anyone who wants to pick a specific epoch by hand.

**Blast radius on WEXAC**

- `--save_best_only` is opt-in; without it every per-epoch checkpoint is still written exactly as
  before, plus `last.pt` (one extra 124 MB file per run) and `best.pt`.
- Restoring best weights only affects the in-memory `model` after `train_model` returns. Nothing
  in `__main__` uses it afterwards, so no saved artifact changes.
- `--patience` still defaults to `0` (off), so no run stops early unless asked.
- With `--reporter False` the console output is unchanged apart from two new lines: the
  `new best ...` notice and, if `--patience` is set, the early-stop warning.

**To revert:** drop `--save_best_only` / `last.pt` / the restore block, and move the `BestTracker`
construction back inside the `if rep is not None:` block.

---

### #5 — More run outputs: validation loss, IoU/F1, learning-curve figure, sample grids

*2026-07-27*

**Files**

| File | Change |
|---|---|
| `train_reporter.py` | New section 5: `read_results_csv`, `plot_curves`, `save_prediction_grid`, `_figure`. `__main__` takes an optional run directory. Self-test columns renamed to `val/*`. |
| `evaluate.py` | `evaluate()` gained `loss_fn` and `samples_out`; keeps the pre-threshold logits/probabilities; `metrics_out` gained `loss`, `iou`, `f1`. |
| `train_sinkholes_unet.py` | Loss body extracted to `segmentation_loss()`; three new `results.csv` columns; `curves.png` and per-epoch sample grids; `--sample_every`, `--n_samples`. |

**What changed**

1. **`val/loss` in `results.csv`.** Only `train/loss` was recorded, so a run gave no direct view of
   the train-vs-val gap — the primary overfitting signal. To make the two numbers comparable the
   loss body was lifted **verbatim** out of the training loop into `segmentation_loss(logits,
   images, true_masks, model, args, criterion)`, and validation now calls that same function
   through `evaluate(..., loss_fn=...)`. A copy-pasted second implementation would have drifted
   the first time either branch was tuned; this cannot.

   Verified bit-identical to the old inline loss — **value and gradient** — across
   `n_classes ∈ {1,3}` × `treat_nodata_regions ∈ {False,True}` × `pos_w ∈ {1,8}`. The hardcoded
   `pos_w=8.0` on the masked path (which deliberately differs from `--pos_w`) and the `r_tol=2` /
   `lam=0.2` constants were carried over untouched.

2. **`val/IoU` and `val/F1`**, computed from the TP/FP/FN counts #3 already accumulated — no extra
   passes. Note `val/F1` is **not** `val/dice`: F1 pools every pixel in the split (micro) while
   the returned Dice averages per batch (macro), so a patch with 3 positive pixels weighs the same
   as one with 3000. They will differ, and both are correct — the same macro/micro distinction
   `inspect_run.py` documents.

   Column order is now `epoch, time, train/loss, val/loss, val/dice, val/IoU, val/F1, val/P,
   val/R, lr`. Readers key off column *names* (`inspect_run.py:44`), so appending is safe;
   verified against both an old and a new `results.csv`.

3. **`curves.png`** in the run directory: train/val loss with the learning rate as a dashed step on
   a log twin axis (which is where the `ReduceLROnPlateau` drops line up against the loss they were
   meant to fix), plus every `val/*` metric, best epoch marked. Written at the end of the run —
   including after a Ctrl+C — and regenerable for any finished run without retraining:

   ```bash
   python train_reporter.py outputs/<run_dir>
   ```

4. **`validation/preds/epoch_NNN.png`** — input / ground truth / predicted probability with the
   0.5 contour drawn on, for a few validation patches, every `--sample_every` epochs (default 1,
   `0` disables; `--n_samples` sets the count, default 4). Patches with positive ground truth are
   preferred, since an all-background sample shows nothing, and the pick is deterministic across
   epochs because `val_loader` is not shuffled — so the grids form a flipbook of the *same*
   patches improving. The displayed channel is `T-1`, the current interferogram
   (`sinkholes_data_loading.py:176` stacks prevs…present).

   `evaluate()` previously discarded the sigmoid output at
   `mask_pred = (F.sigmoid(mask_pred) > 0.5).float()`; the probability map is now kept, which is
   what makes a *nearly* right boundary distinguishable from an absent one.

**Why**

A finished run reported one loss curve and a Dice number. The three additions answer the three
questions that actually come up afterwards: is it overfitting (`val/loss`), how good is the overlap
under a threshold-free measure (IoU/F1), and *what is it actually predicting* (the grids). The last
one is the reason this was worth doing — on the 2-epoch check run the grids immediately showed the
model firing on the bright decorrelated band rather than on sinkholes, which no scalar metric says.

**Blast radius on WEXAC**

- `evaluate()`'s signature is **append-only** and its return value is unchanged; `test.py`,
  `test_full_intf.py` and `evaluate_full_intf_output.py` call it exactly as before and were
  verified to produce an identical Dice. Both new kwargs default to `None`, and when they are
  `None` no extra work is done.
- Training numerics are unchanged (see 1). `--reporter False` produces no `results.csv`,
  no `curves.png` and no grids, exactly as before.
- Cost per epoch: the validation loss adds one `no_grad` loss evaluation per batch; the sample
  grids stop collecting after `--n_samples` patches. Measured on the local 87/499-patch run,
  epoch time was unchanged at ~18 s.
- Disk: `curves.png` ≈ 110 KB per run, each sample grid ≈ 250 KB — a 30-epoch run adds ~8 MB,
  against 124 MB for a single checkpoint.
- Plotting goes through `FigureCanvasAgg` directly rather than pyplot, so the `Qt5Agg` backend
  `evaluate.py:6` forces at import (see "Considered and rejected" below) is never consulted and
  nothing tries to open a GUI window. Both plot calls are additionally wrapped in `try/except` —
  a figure must never take down a run that has already trained.

**To revert:** drop the `loss_fn`/`samples_out` parameters and the `metrics_out` additions in
`evaluate.py`, restore the three columns, and inline `segmentation_loss()` back into the loop.
`train_reporter.py` section 5 is additive and can simply go unused.

---

### #6 — ConvLSTM U-Net: a temporal segmentation architecture

*2026-07-28*

**Files**

| File | Change |
|---|---|
| `convlstm_unet.py` | **New.** `ConvLSTMCell`, `ConvLSTMUNet`, and the `build_convlstm_unet` / `pop_model_config` checkpoint helpers. |
| `tests/conftest.py`, `tests/test_convlstm_unet.py` | **New.** Repo root on `sys.path`; 30 tensor-level tests. |
| `train_sinkholes_unet.py` | `--convlstm_unet`, `--convlstm_hidden`, `--convlstm_kernel`; explicit three-way architecture selection; model-aware input assert; ConvLSTM config saved into the checkpoint; banner/log wording. |
| `test.py` | `--convlstm_unet`, `--treat_nodata_regions`; checkpoint loaded before the model is built so the architecture can be rebuilt from the config it carries. |

**What changed**

`UNet` receives a temporal stack as ordinary input channels, so the first convolution mixes every
timestep at once and their ordering means nothing. `ConvLSTMUNet` treats the stack as a sequence:
a **shared** encoder runs on every timestep (time folded into the batch dim, so BatchNorm sees
`B*T` samples rather than `B`), the bottleneck feature maps are collected across time as 2D maps,
a ConvLSTM consumes them chronologically, and its final hidden state drives the standard decoder.
Skip connections come from the **latest** timestep only — a deliberate v1 simplification, marked
with a `TODO` in the source.

Shapes at `B=2, T=3, 200x100`: input `[2,3,200,100]` → sequence `[2,3,1,200,100]` → folded
`[6,1,200,100]` → bottleneck `[6,1024,12,6]` → sequence `[2,3,1024,12,6]` → final hidden
`[2,1024,12,6]` → logits `[2,1,200,100]`. Output is raw logits, exactly like `UNet`, so the
existing BCEWithLogits + Dice loss and `evaluate()` work untouched.

`T` is **not** an architectural parameter — the encoder is shared and the ConvLSTM is unrolled
dynamically, so a model trained at `T=3` runs at any `T`. Consequently `model.n_channels` on this
class means *channels per timestep*, not the flat channel count; the training loop's
`images.shape[1] == model.n_channels` assert is therefore replaced by a divisibility check on the
ConvLSTM branch only. The input contract is strict: `[B,T,H,W]`, `[B,2T,H,W]` (block layout) or the
explicit `[B,T,C,H,W]`, and nothing else — every other shape raises a `ValueError` naming the
received shape and whether the temporal dimension looks missing.

**Why:** temporal structure is the point of `--add_temporal`, and channel-stacking discards it.

**Blast radius on WEXAC:** none unless `--convlstm_unet` is passed. `UNet` and `AttentionUNet`
construct exactly as before, their checkpoints are byte-identical (the config key is written only
for the new class), and every existing command line still runs. Conflicting flags now **fail**
rather than silently picking one: `--convlstm_unet` with `--attn_unet`/`--add_attn`, or without
`--add_temporal`, exits 2 with an explanatory message.

**Cost:** at the default `hidden = bottleneck = 1024` the ConvLSTM gate convolution is
`(1024+1024) → 4096` at 3×3 = **75.5 M** parameters, on top of the 31 M U-Net — **106.5 M** total.
`--convlstm_hidden 256` (≈12 M) or `--convlstm_kernel 1` are the cheap knobs; a narrower hidden
state is projected back to the bottleneck width with a 1×1 conv.

**To revert:** delete `convlstm_unet.py` and `tests/test_convlstm_unet.py`, and drop the
`--convlstm_*` flags plus the `isinstance(model, ConvLSTMUNet)` branches. Nothing else depends on it.

---

### #7 — Temporal target is the latest timestep, not the union

*2026-07-28*

**Files:** `sinkholes_data_loading.py` (new `temporal_target_mask()`, both temporal branches),
`train_sinkholes_unet.py` (`--union_temporal_mask`), `tests/test_temporal_data.py` (**new**).

**What changed:** the temporal ground truth was
`mask_data = (np.stack(masks_per_t, 0) > 0).any(0)` — positive wherever *any* interferogram in the
stack was positive. It is now the **newest** interferogram's mask, extracted into
`temporal_target_mask(masks_per_t, union=False)` so the rule is unit-testable without touching disk.

**Why:** the model is asked to predict the state at the newest timestep — the decoder's skips are
taken there and the ConvLSTM hidden state summarises the history leading up to it. A unioned target
asks it to also mark subsidence that had already healed or moved, which is a different task.

**Blast radius on WEXAC: every `--add_temporal` run changes its ground truth.** Positives shrink to
the newest interferogram's polygons, so Dice/precision/recall are not comparable with runs made
before 2026-07-28. `--union_temporal_mask` restores the old target exactly. Ring-negative selection
still uses the union grid on purpose — a negative patch should be empty at *every* timestep —
and says so in a comment.

**To revert:** pass `--union_temporal_mask`, or change the `union` default in
`temporal_target_mask()`.

---

### #8 — Validity channels stay {0, 1} through preprocessing

*2026-07-28*

**Files:** `sinkholes_data_loading.py` (`preprocess`, `__getitem__`, `self.n_value_channels`).

**What changed:** `SubsiDataset.preprocess` looped over **all** channels and rewrote exact zeros to
`0.5`. With `--treat_nodata_regions --add_temporal` the stack is `[img_t0…img_t{T-1},
V_t0…V_t{T-1}]`, and the validity maps have min 0 / max 1, so they took that branch and arrived at
the model as `{0.5, 1.0}`. `segmentation_loss` (`train_sinkholes_unet.py:124`) derives
`V_any = V.max(...)`, which was therefore **never zero** — the no-data masking never masked
anything, and the false-positive suppression term ran at half weight.

`preprocess` now takes `n_value_channels`: channels at or after that index are left exactly as they
are. The dataset records it where the concatenation happens, so the two cannot disagree.
`__getitem__` reads it through `getattr`, so datasets pickled before this change still load.

Verified on the local subset: validity channels are now `{0., 1.}` with 1398 no-data pixels in the
worst of 40 sampled patches, and the phase channel still reads `0.5` at those pixels — which is
what `test_full_intf.py:171` assumes when it recomputes validity as `|x - 0.5| > tol` at inference.

**Blast radius on WEXAC:** `--add_temporal --treat_nodata_regions` runs only — the dataset never
appends validity channels on the non-temporal path. Those runs now genuinely mask no-data regions
in the loss, so their numbers will move. Everything else is untouched.

**To revert:** pass `n_value_channels=None` at the `preprocess` call site in `__getitem__`.

---

### #9 — `test_full_intf.py` fed temporal channels in reverse

*2026-07-28*

**Files:** `test_full_intf.py` (`reconstruct_intf_prediction`, one line).

**What changed:** training stacks timesteps **oldest → newest** — `find_11day_sequences` returns
`prevs` oldest-first (`get_intf_info.py:313-321`) and the dataset builds
`tids = list(prevs) + [id]` (`sinkholes_data_loading.py:176`). But `test_full_intf.py:396`
assembles `pa = [cur] + prevs[::-1]`, i.e. **newest → oldest**. Every `--k_prevs > 0` evaluation
through this script has therefore been running with its temporal channels reversed relative to
training. The channel axis is now reversed at the point the tensor is fed to the net:

```python
x_np = data_stack[::-1, i, j].copy()   # oldest -> newest
```

`data_stack` itself is deliberately **not** reordered — index 0 must stay the current
interferogram for `reconstructed_intf_all[0]`, the LiDAR AND-gating and the plots. The reversal
happens before the validity concat, so the no-data block layout stays
`[imgs chronological…, validity chronological…]`.

**Why:** the ConvLSTM makes ordering load-bearing, so this had to be correct; and it was simply
wrong for `UNet` too.

**Blast radius on WEXAC: every `test_full_intf.py` run with `--k_prevs > 0` changes its output**,
for all architectures. Results should improve, but numbers reported before 2026-07-28 will not
reproduce. No-op at `--k_prevs 0` or `--replicate_input`. `predict_new_intf_vA.py` was checked and
is already chronological (`all_data = prev_data + [current]`, `:470-475`) — it needed no change.

**To revert:** restore `x_np = data_stack[:, i, j]`.

---

### #10 — Repository reorganised into `src/`, `docs/`, `assets/`

*2026-07-29*

**Why:** the repo root held ~28 `.py` files, 5 `.md` files and the data assets in one flat list,
which made the editor side pane unusable for navigation.

**Files** — moved, not edited (except as noted):

| From (root) | To |
|---|---|
| `unet.py`, `unet_parts.py`, `attn_unet.py`, `convlstm_unet.py` | `src/models/` |
| `prepare_*.py`, `create_intf_partition.py`, `clean_patches.py`, `check_patches.py`, `untie_mask_polygs.py` | `src/data_prep/` |
| `train_sinkholes_unet.py`, `train_reporter.py`, `evaluate.py`, `dice_score.py`, `sinkholes_data_loading.py` | `src/training/` |
| `test.py`, `test_full_intf.py`, `predict_new_intf*.py`, `evaluate_full_intf_output.py`, `inspect_run.py`, `remove_no_Data_predictions.py` | `src/inference/` |
| `device_utils.py`, `get_intf_info.py`, `polygs.py`, `lidar.py` | `src/utils/` |
| `intf_coord.json`, `lidar_mask_polygs.*`, `lidar_intf_mask.txt`, `partition_20_05_*.json` | `assets/` |
| `CHANGELOG.md`, `LOCAL_WALKTHROUGH.md`, `PIPELINE_MANUAL.md`, `PROJECT_SUMMARY.md` | `docs/` |

`README.md` stays at the root. Moves used `git mv` for tracked files, so history is preserved.

**Two code changes were needed to keep the flat imports working:**

1. **New `src/_bootstrap.py`.** Every module still imports its neighbours by bare name
   (`from unet import UNet`). Python only puts the running script's own directory on `sys.path`,
   so 20 entry-point modules got a 5-line header importing `_bootstrap`, which puts every
   `src/*` folder on the path. No import statement anywhere was rewritten.

2. **Asset paths resolved absolutely.** 23 call sites opened `intf_coord.json`,
   `lidar_mask_polygs.shp`, `lidar_intf_mask.txt` and `partition_20_05_13h45.json` by bare
   relative name, which only worked when launched from the repo root. They now call
   `_bootstrap.asset(...)`. **This is a behaviour improvement, not just a move:** scripts run
   correctly from any working directory now. `check_patches.py` writes `intf_coord.json` back to
   `assets/` rather than the CWD.

`tests/conftest.py` now delegates to `_bootstrap` instead of adding the repo root.
Doc command invocations were updated to the new paths (`python src/training/train_sinkholes_unet.py`).
The stale "run from the repo root or assets won't resolve" warnings in `PIPELINE_MANUAL.md`
and `LOCAL_WALKTHROUGH.md` were corrected.

**Blast radius on WEXAC: this changes how every script is invoked.** Any job script, cron entry or
shell alias calling `python train_sinkholes_unet.py` must become
`python src/training/train_sinkholes_unet.py`. Numerics are untouched — no model, loss, data
loading or metric code was modified, and the full test suite (39 tests) passes unchanged.

**To revert:** `git mv` the files back to the root, delete `src/_bootstrap.py`, strip the
`# --- path bootstrap ---` headers, and restore the bare relative asset literals.

---

### #11 — `PIPELINE_MANUAL.md` covers the ConvLSTM U-Net and the new paths

*2026-07-29*

**Files:** `docs/PIPELINE_MANUAL.md` (documentation only — no source changed).

**Why:** the manual predated `#6` and had **zero** mention of `--convlstm_unet`, so the
best-scoring architecture in the repo was undocumented. Its invocations also still assumed the
pre-`#10` flat layout.

**Added:** an architecture-selection table; a ConvLSTM training section (`--convlstm_hidden`,
`--convlstm_kernel`, the `--add_temporal` requirement, the batch-32 overfitting result); the
`--convlstm_unet` form of `test.py`; how `convlstm_unet_config` travels inside the checkpoint so
hidden size/kernel need not be re-specified at test time; `build_convlstm_unet` /
`pop_model_config` in the cheat-sheet; ConvLSTM construction in the model-construction snippet;
and two checklist items.

**Recorded a real gap, not a workaround.** `test_full_intf.py` (`:342-344`), `inspect_run.py`
(`:72-73`) and both `predict_new_intf*.py` scripts construct only `UNet` / `AttentionUNet` and
have no `--convlstm_unet` flag, so a ConvLSTM checkpoint fails at `load_state_dict`. Verified by
loading `outputs/convlstm_v1_2026-07-28_15h34/checkpoints/best.pt` into a `UNet` (raises on key
mismatch) and by confirming `build_convlstm_unet()` loads it cleanly. **Stages (3b) and (4) are
therefore unavailable for ConvLSTM runs**; patch-level `test.py` is the only evaluation route.
This is flagged in three places in the manual rather than papered over — fixing it means routing
those scripts through `build_convlstm_unet()`, which is a code change and out of scope for a
documentation pass.

**Blast radius:** none — documentation only.

---

### #12 — One checkpoint→model factory; stages (3b) and (4) work with every architecture

*2026-07-29*

**Why:** `#11` documented that `test_full_intf.py`, `inspect_run.py` and both
`predict_new_intf*.py` scripts could not load a ConvLSTM checkpoint — the best-scoring model in
the repo had no route to full-interferogram evaluation or deployment. Each script carried its own
copy of an `if flag: A() else: B()` construction chain, so `#6` updated `test.py` and silently
left the other four behind. Adding the flag in four more places would guarantee the same drift on
the next architecture.

**Files**

| File | Change |
|---|---|
| `src/models/factory.py` | **New.** Registry of architectures + `build_from_checkpoint()`. |
| `src/inference/test_full_intf.py` | Hardcoded `UNet`/`AttentionUNet` pair → factory. Added `--add_attn`, `--convlstm_unet`. |
| `src/inference/predict_new_intf.py` | Hardcoded `UNet` → factory. Added the four architecture/nodata flags. |
| `src/inference/predict_new_intf_vA.py` | Same, plus a missing `matplotlib` import (see below). |
| `src/inference/inspect_run.py` | Local `build_net()` → factory; checkpoint lookup falls back to `best.pt`/`last.pt`. |
| `src/inference/test.py` | Its `build_net()` and flag-exclusion check → factory. Behaviour unchanged. |
| `tests/test_factory.py` | **New.** 23 tests. |

**The design.** A checkpoint already describes itself, so the architecture is *detected*, not
declared: ConvLSTM has `convlstm.*` weights and a `convlstm_unet_config` blob, `add_attn` has
`attn.*`, `AttentionUNet` has a deeper `DoubleConv` (`...double_conv.5.*`), and the input channel
count is `inc.double_conv.0.weight.shape[1]`. Adding an architecture is now one `register()` entry
in one file with no call-site edits.

**Backward compatible.** `--attn_unet` / `--add_attn` / `--convlstm_unet` still work as an explicit
override, so existing WEXAC job scripts are unaffected. Two improvements come free: a flag that
contradicts the weights now fails with a readable message instead of a `load_state_dict` key
mismatch, and a wrong `--k_prevs` is reported against the checkpoint's real channel count.

**Two pre-existing bugs surfaced and fixed:**

1. `predict_new_intf_vA.py` used `plt` in 7 places but never imported matplotlib — it relied on
   `from unet import *` re-exporting it, which entry **#0** removed from `unet.py` as "unused". The
   script therefore died at import with `NameError` **for every architecture**; stage (4) via the
   preferred v1A script was broken outright. Fixed with an explicit import.
2. `inspect_run.py` only looked for `*checkpoint_epoch<N>.pth`, which `--save_best_only` runs (the
   default since `#5`) never write — so its threshold sweep failed on all four runs to date. It now
   falls back to `best.pt`/`last.pt` and lists what is available otherwise.

**Verified:** all four real checkpoints in `outputs/` detect and load correctly; `test.py` on the
ConvLSTM run gives dice **0.5460** both with `--convlstm_unet` and with no flag at all (identical,
so detection matches the old explicit path); `inspect_run.py` threshold sweeps now run on all
four. Full suite 62 passed.

**Blast radius on WEXAC:** none for existing invocations — flags behave as before and numerics are
untouched. New capability only. The one visible change is an extra `INFO` line naming the detected
architecture.

**To revert:** restore the per-script construction blocks and delete `src/models/factory.py`; the
two bug fixes above are independent and should be kept.

---

### #13 — Remaining docs brought in line with the ConvLSTM + factory; broken local commands fixed

*2026-07-29*

**Why:** an audit after `#12` found `PIPELINE_MANUAL.md` was updated but the other docs were not.

**Fixed — genuinely wrong, not just incomplete:**

- **`LOCAL_WALKTHROUGH.md` had 12 broken commands.** The `#10` path rewrite only matched
  `python <script>.py`; this file invokes scripts as `$PY <script>.py`, so every command in it
  still pointed at the pre-`#10` repo root and would fail. All 12 corrected.
- **`PROJECT_SUMMARY.md §8` listed "a ConvLSTM over the stack" as a *future suggestion*** when it
  has been implemented since `#6` and is the best-scoring model. Rewritten to record the outcome
  (0.591 vs 0.565 single-frame vs 0.342 channel-stacked), keeping the still-open
  time-series-product recommendation.
- **`LOCAL_WALKTHROUGH.md §6.3` claimed the temporal target is the union of masks** — changed to
  latest-timestep by `#7`. Corrected, with `--union_temporal_mask` noted for the old behaviour.
- **`LOCAL_WALKTHROUGH.md §7.1` claimed `test.py` has no `--treat_nodata_regions` flag** — it has
  had one since `#6`. Corrected.

**Added:** `convlstm_unet.py` and `factory.py` to the file tables in `README.md` and
`PROJECT_SUMMARY.md`; the ConvLSTM to the README architecture list; architecture-detection notes to
the inference sections of all three; a ConvLSTM training recipe at `LOCAL_WALKTHROUGH.md §6.3b`
(with `--convlstm_hidden 256` for local runs).

**Code:** removed imports left dead by `#12` — `build_convlstm_unet`/`pop_model_config` in
`test.py`, `UNet`/`AttentionUNet` in `test_full_intf.py` and `inspect_run.py`. No behaviour change;
all five scripts re-verified, ConvLSTM test dice still 0.5460 and the sweep still 0.6686 @ 0.75.

**`.vscode/settings.json`:** added `python.analysis.extraPaths` for the `src/*` folders. After
`#10`, Pylance flagged every first-party import as unresolved because it did not know the search
path that `_bootstrap.py` sets at runtime — which defeated the point of reorganising for
navigation. Also enabled pytest discovery.

**Blast radius:** none — documentation, dead imports and editor config only.

---

### #14 — Full-intf predictions land under `outputs/`, not a second top-level tree

*2026-07-29*

**Files**

| File | Change |
|---|---|
| `test_full_intf.py` | `output_path` was `f'pred_outputs2/{model_name}/{job_name}/'`; now `os.path.join(args.output_dir, model_name, job_name, '')`. New `--output_dir`, default `outputs/predictions`. |
| `evaluate_full_intf_output.py` | `--path` default followed suit (and no longer names a stale 2009 job dir). |
| `polygs.py` | `directory_path` in the `__main__` block followed suit. |
| `PIPELINE_MANUAL.md`, `PROJECT_SUMMARY.md`, `LOCAL_WALKTHROUGH.md` | Paths updated; a "Running (3b) locally" section added to `PIPELINE_MANUAL.md §3b`. |

**Why:** `pred_outputs2/` was a second results tree at the working directory, unrelated by name or
location to `outputs/`, where training already writes checkpoints, logs and the pickled test sets.
Predictions belong beside the runs that produced them, and one root makes both trees easy to find
and to clean up. The `2` suffix also had no meaning left — there is no `pred_outputs/`.

**Blast radius:** WEXAC jobs that hardcode a `pred_outputs2/...` path when reading results back
(job scripts, notebooks) will not find new runs — pass `--output_dir pred_outputs2` to keep the old
layout. Existing `pred_outputs2/` directories are untouched; nothing reads the default in
`evaluate_full_intf_output.py`, which is always given `--path` explicitly. `.gitignore` covers the
new location via `/outputs/` and keeps `/pred_outputs2/` for older results.

**Verified:** stage (3b) re-run locally end to end (temporal U-Net, `--k_prevs 2`, 2 test intfs,
Hann blending) → `outputs/predictions/<model>/<job>_<ts>/` with per-intf `.npy` arrays, per-intf
shapefiles and a combined shapefile of 7 178 polygons.

---

## Considered and rejected

Changes that looked reasonable but were **not** made, so nobody re-litigates them from scratch.

### `--k_prevs` default of 2

`train_sinkholes_unet.py:157` applies the 11-day chain filter unconditionally (`if args.add_temporal or True:`),
so a non-temporal run still requires each interferogram to have `k_prevs` predecessors — silently
dropping 6 of the 30 local interferograms at the default of 2, even though `--k_prevs` only affects
the channel count when `--add_temporal` is set.

Changing the default to `0` would fix that, but it would also break any WEXAC run that passes
`--add_temporal` while relying on the default (input channels would drop from 3 to 1, making saved
checkpoints unloadable). **Pass `--k_prevs 0` explicitly for non-temporal runs instead** — that is
what the commands in `LOCAL_WALKTHROUGH.md` do.

The better fix is to make line 157 honour `args.add_temporal`, i.e. remove the `or True`. That is a
real behaviour change for existing runs and needs a decision from whoever owns the WEXAC
experiments, so it is deferred rather than done quietly.

### `unique_mask_values` hardcoded `nonz_` prefix

`sinkholes_data_loading.py:82` should follow `--nonz_only` rather than always globbing
`mask_patches_nonz_`. Worked around by generating the files (#2) instead of patching the loader,
to keep the local setup on exactly the same code path as WEXAC. Worth fixing properly.

### `test.py` unconditional `net_aux=net`

`test.py:88` passes the primary network as its own auxiliary, so `--aux_model` is a no-op and every
test run does double the forward work. Left alone pending a decision on whether the aux-model
feature is still wanted.

### Qt5Agg forced at import

`evaluate.py:64` and `test_full_intf.py:280` assign `plt.rcParams['backend'] = 'Qt5Agg'` at module
import, which `MPLBACKEND` cannot override and which crashes on machines without PyQt5. Left alone;
the workaround is simply never passing `--plot` locally.
