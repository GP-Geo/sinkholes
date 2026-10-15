# HPO and long training: plain vs ctx50 temporal attention (k10, generation 5-t2026)

Study `k10t2026_hpo_v1`, specified in [`assets/hpo/k10t2026_plain_vs_ctx50.json`](../assets/hpo/k10t2026_plain_vs_ctx50.json)
and run by `sinkholes hpo`. This page is the protocol; the code is the source of truth for
every definition quoted here.

**Nothing is submitted without an explicit approval per phase, and `step` is a dry run unless
given `--submit`.**

## 1. The question and what is held fixed

Plain (200×100 in/out) against ctx50 (300×200 in, the centre 200×100 supervised and
reconstructed) temporal attention, both on the generation 5-t2026 k10 partition
`assets/partition_temporal_k10_test2026_fill_th350x200.json`: generation 5's filled chains,
scenes and 350N/200S threshold re-split by date blocks with full separation -- **train 131
scenes (incl. 2024 and Jan-Oct 2025; 43,944 AOI positives), val 11 (2022-2023), test 10
(Feb-Apr 2026)**; k10 = ten previous frames + the current one, real frame ages for chains that
skip a missing acquisition (assets/PARTITIONS.md, "Generation 5-t2026"). Context is back in scope for this study: its earlier predictions looked
visually better despite lower object scores, which is what the visual panel (§6) is for.

Identical for both geometries: scenes, filled history, frame ages, labels, ring negatives,
augmentation, optimiser, schedule, micro-batch × accumulation, evaluation protocol.

| fixed | value |
|---|---|
| architecture | tattn U-Net, dim 256, 8 heads, 1 layer, recurrence none, fused skips 0, contrast + QK-norm |
| optimiser | AdamW, betas 0.9/0.999 (decoupled weight decay) |
| loss | BCE(pos_weight) + Dice; region loss reduced in FP32 under AMP |
| negatives | ring 1:1, inner 1, outer 10 |
| augmentation | horizontal + vertical flips (train only) |
| numerics | AMP with unscale-then-clip at norm 1 (`unscaled-v2`) |
| batch | micro 32 × accumulation 4 = 128, **both geometries** |
| HPO schedule | plateau on val/dice, patience 8, factor 0.5, min LR 1e-8; no early stopping, no pruning |

**Micro-batch.** ctx50 k10 tattn at 32×4 has already been measured on this architecture and
optimiser: 33.5 GiB VRAM reserved (LSF 254183). 32×4 therefore holds for both geometries on a
≥40 GB GPU; the `prep` profile job re-measures it on the current torch before the study.

## 2. The search

| parameter | range |
|---|---|
| learning rate | log-uniform 1e-4 – 8e-4 |
| weight decay | log-uniform 1e-4 – 3e-2 |
| pos_weight | {2, 4, 8, 12} |

Six shared configurations, run on **both** geometries (12 trials), seed 42, 30 epochs:

| | lr | weight decay | pos_w | |
|---|---|---|---|---|
| s0 | 3.0e-4 | 1.0e-2 | 8 | the current configuration (baseline) |
| s1 | 1.1e-4 | 1.5e-2 | 4 | |
| s2 | 1.9e-4 | 1.3e-3 | 12 | |
| s3 | 2.6e-4 | 1.4e-4 | 2 | |
| s4 | 4.9e-4 | 7.0e-4 | 12 | |
| s5 | 7.1e-4 | 6.9e-3 | 4 | |

s1–s5 are a maximin Latin hypercube over (log lr, log wd) anchored on the baseline (numpy seed
20261013, 200k candidates, minimum distance 0.415 in the unit square), rounded to two
significant figures; pos_w 2/4/4/12/12 is assigned so it is not confounded with lr. **This is
a proposal**: it can be changed before the `shared` phase is approved (a changed spec is a new
study, so change it before `init`).

Then, per geometry: **two TPE trials** (Optuna `TPESampler`, `n_startup_trials=6` so the six
shared trials are the startup set, multivariate, constant liar, seed 20261013) → **the best two
of eight continued to 60 epochs** by full-state resume → **the best promoted configuration
repeated at seeds 7 and 123** for 60 epochs.

## 3. The objective (never patch Dice)

Whole **validation** scenes (the partition's 11 `val` scenes, 2022-2023; AOI and LiDAR gate as always),
stride-4 grid, unweighted tile votes at 0.5, every RTh ∈ {0.125, 0.25, 0.375, 0.5}, strict
object metric ITh 0.7 / buffer 5 px. Object F1 is this project's: the harmonic mean of the
per-scene **mean** recall and **mean** precision (`F1 = 2·P̄·R̄/(P̄+R̄)`; 0.8114 for the fill
k10 model on its 24-scene test list reproduces this way).

```
F(t)      = mean over e in {20, 30} of F1(e, t)        (budget e30)
objective = max over t of F(t)
```

One threshold for all scenes and both checkpoints — never a per-scene or per-epoch best. A
threshold with an undefined F1 counts 0. The top two by this `e30` objective are promoted; once
continued, they are scored by the same rule over epochs {50, 60} (budget `e60`), and the better
of the two is the configuration repeated at seeds 7 and 123. The two budgets are different quantities and are stored apart: Optuna study
`<study>_<geometry>_e30` holds the initial trials (and is what TPE reads); `<study>_<geometry>_e60`
holds promoted continuations and seed repeats. Every threshold's score, P/R, area-weighted
variants, per-scene results and false-positive / background-object counts are recorded.

**Deployment threshold.** After the seed repeats, per geometry: the RTh maximising the mean
over seeds {42, 7, 123} of the epoch-50/60 mean val F1 is written to the ledger as `locked`,
before any test scoring. Test evaluation uses that RTh.

## 4. Scene evaluation: `sinkholes eval-snapshots`

`eval-scenes` + `eval-outputs` rebuilt for model selection (`sinkholes/inference/snapshot_eval.py`):

* **one scene load serves every checkpoint** of a job (a trial's epochs 10/20/30 together);
* **only gated-in AOI tiles are read**, from memory-mapped grids; the normalisation branch is
  decided over the whole common grid exactly as `eval-scenes` decides it, from a cached per-cell
  min/max (`hpo cache-minmax`), so values are bit-identical while I/O and RAM shrink;
* **tiles are batched** (`--tile_batch`; 1 is the legacy one-tile path exactly);
* **missing history is an error** — and `eval-scenes` now refuses too (`--allow_missing_history`
  restores the old padding explicitly);
* **one confidence map, every threshold**: votes/counts kept as uint8 (lossless), scored at all
  RTh, saved compressed; GT once per scene in the study cache, no input stacks, no shapefiles;
* **per-scene results written as they finish**: a killed job resumes.

Equivalence: `tests/test_snapshot_eval.py` runs both pipelines on synthetic scenes (partial
history, both normalisation branches, an AOI overhanging the grid, per-survey LiDAR gates,
plain and context geometry, tile batch 1 and 5) and requires identical per-scene recall,
precision, GT area **and** confidence maps. The object metric is pinned to
`object_level_evaluate` operation for operation. A real-data check (one validation scene with a
gappy chain, the fill k10 `best.pt`) is recorded in §11.

## 5. Training changes (all opt-in flags; existing runs are unchanged)

| flag | does |
|---|---|
| `--snapshot_epochs SPEC` | model-only weights at exactly these epochs (`checkpoints/snapshots/epoch_NNN.pt`) |
| `--keep_prev_resume` | rolling full-state `resume.pt` each epoch plus the previous one (`resume_prev.pt`); resume falls back to it |
| `--time_budget_hours H` | stop cleanly after the last epoch that fits; `run_status.json` says `paused`; the controller submits the continuation (LSF requeue is never assumed) |
| `--lr_schedule cosine --lr_horizon N` | closed-form cosine to `--min_lr`, horizon fixed at epoch 1 and refused if a resume changes it; never restarts |
| `--lr_schedule constant` | the diagnostic control |
| `--eval_rng_isolation` | validation and the fit probe run inside a saved/restored RNG context: evaluation cannot move training |
| `--fit_eval_samples N` | each epoch, a fixed seeded subset of positive TRAINING samples, unaugmented, eval-mode → `fit/*` columns |
| `--strict_resume` | also locks lr, schedule, min_lr, plateau settings, pos_w, AMP for the life of the run |

`results.csv` gains `lr_used` (the LR the epoch trained at — the old `lr` column is read after
`scheduler.step()` and is the next epoch's), `grad/norm_mean`, `grad/norm_max`, `opt/steps`,
`opt/skipped` (AMP-skipped updates), `amp/scale_min`, `epoch_s`. Every execution appends a
record to `run_manifest.json`: package SHA-256 and git state, partition and coordinate
dictionary SHA-256, a stat digest of every grid read, optimizer betas/eps/decay mode, schedule
state, numerical policies, versions, GPU, LSF ids. `run_status.json` is rewritten at each state
change (`running`/`paused`/`complete`/`interrupted`/`failed`).

Tested (`tests/test_hpo_training.py`): a run trained in three chunks is **bit-identical** to an
uninterrupted one; the cosine horizon survives resumes; the fit probe cannot change training;
snapshots land at exactly the scheduled epochs and load like `best.pt`; the rotation and
fallback; the time-budget pause and continuation; strict resume; skipped-update counting.

## 6. Long training and the visual panel

After HPO, per geometry, the locked configuration is trained fresh for **300 epochs**, seed 42
first (`long`), then 7 and 123 (`long_seeds`): cosine from the selected LR to a floor of 1 % of
it over a 300-epoch horizon fixed at epoch 1, no early stopping, no pruning, nothing changed
mid-run (`--strict_resume`). Scene evaluation at epochs 1, 10, 20, …, 60, then every 20 to 300.
The constant-LR control (`long_control`) is optional. `hpo report` plots each threshold's curve,
the best-across-thresholds curve, the LR the epochs trained at, and the fit probe against
validation, and reads out any dip-and-recovery: whether it holds at a fixed threshold or only
through the best threshold changing, and what the LR was at peak, trough and recovery.

`hpo panel --trials KEY@EPOCH ...` writes the visual panel: windows chosen from the ground
truth and LiDAR coverage only (GT objects stratified by size + background windows, fixed seed),
models under shuffled letters with the key sealed in `panel_key.json`, no scores on the figure,
and a rating sheet (boundaries, fragmentation, seams, missed detections). Beside it,
`matched_recall.json`: each model scored at every vote level from its saved maps, and its
interpolated precision at a common recall — the number that separates "cleaner" from
"predicts less". Visual findings and numerical rankings stay in separate files.

## 7. Storage, retention, protection

The study root is **`/home/labs/rudich/pinkas/sinkholes/studies/k10t2026_hpo_v1`** (moved from
`pinkas/sinkholes_studies/` on 2026-10-15; job scripts and run manifests written before that keep
the old path). `studies/` is gitignored and outside `outputs/`, so `scripts/tidy_outputs.sh`
and any other `outputs/` clean-up cannot reach it; a `PROTECTED_STUDY_STATE` file says so in
the root. Per run: `best.pt` (patch Dice), `last.pt`
(final), `resume.pt` + `resume_prev.pt`, snapshots only at scheduled epochs, and
`best_scene.pt` once evaluated. No per-epoch `.pth`, no input stacks, no shapefiles.

`hpo retain` (dry run; `--apply` to act, every deletion logged): copies the best-by-scene
snapshot to `best_scene.pt`; deletes evaluated, non-best snapshots of initial trials (finalists
keep theirs); keeps confidence maps only for best-scene and finalist epochs; never touches
metrics, the ledger, Optuna, `best.pt`, `last.pt`, or `resume.pt` before promotion is decided
(`--drop_unpromoted_resume` after). Estimated: ~25 GB of checkpoints for the 16 initial trials,
~14 GB after retention, ~3.6 GB per 300-epoch run, ~3 GB of HPO confidence maps.

## 8. Data coverage (checked 2026-10-13 with `sinkholes hpo coverage`)

| geometry | use | trees | state |
|---|---|---|---|
| plain | training | stride 2, 341 ids | complete, headers intact |
| plain | val / test scenes | stride 4, 80 / 32 ids | complete |
| ctx50 | training | stride 2, 341 ids | complete |
| ctx50 | **val scenes (2022-23 chains)** | stride 4, 80 ids | **all 80 missing (~1.24 TB)** |
| ctx50 | test scenes (2026 chains) | stride 4, 32 ids | complete |

```bash
# resumable, checks df, verifies itself; ~1.24 TB may need two submissions of the same line
PARTITION=assets/partition_temporal_k10_test2026_fill_th350x200.json K_PREVS=10 MIN_POS=150 \
  bsub < scripts/data/make_context_stride4_patches.sh
```

The controller refuses to submit a ctx50 evaluation until the tree is complete; ctx50
*training* is not blocked (its stride-2 tree is complete).

## 9. Resources (`sinkholes hpo resources`)

| | plain | ctx50 | source |
|---|---|---|---|
| min/epoch, 32×4 | **14.1** (A40) | **42.5** (A40), **26.9** (L40S) | measured on the baseline pair, LSF 790942 / 790941 (87,834 samples); the earlier ~19.2 extrapolation for ctx50 was wrong |
| dataset build | **1 h 43 m** | **2 h 30 m – 6 h 42 m** | measured (hgn40; hgn48 / lgn06), paid again on every LSF requeue |
| host memory, training | **114 GB** peak | ~295 GB (est.) | plain measured (LSF 790942); ctx50 206 GB measured at 54k samples |
| VRAM | 11.5 GiB | 33.5 GiB | measured |
| eval host memory | ~45 GB | ~125 GB | estimate (gated AOI tiles only; legacy ctx50 k10 measured 224 GB) |

Requested: training plain 160 GB / 24 GB GPU, ctx50 340 GB / 40 GB GPU, both 24 h wall with a
22.5 h budget (continuations are automatic chunks); evaluation plain 80 GB, ctx50 192 GB.
Totals: the pre-launch estimate (HPO through seed repeats ≈ 340 GPU-h, six 300-epoch runs ≈ 640
GPU-h more) assumed ctx50 at ~1.4× plain per epoch; on A40 it is 3.0×, so the ctx50 half is
roughly double that estimate. To be redone once ctx50/s0 completes.

## 10. Commands

```bash
# on a WEXAC login node, once
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes
pip install optuna==5.0.0                    # the controller only; GPU jobs do not import it
cd /home/labs/rudich/pinkas/sinkholes        # every study job runs from this checkout
python -m pytest -q tests/test_hpo_training.py tests/test_snapshot_eval.py tests/test_hpo_controller.py
python -m sinkholes hpo coverage
python -m sinkholes hpo preview              # the baseline pair's scripts and bsub lines, no state
bsub < scripts/hpo/engine_check.sh           # the engine check (§11), before trusting any score

python -m sinkholes hpo init                 # ledger + Optuna studies + 12 shared trials; submits nothing
python -m sinkholes hpo approve --phase prep --note "minmax caches + GPU profile"
python -m sinkholes hpo prep                 # dry run
python -m sinkholes hpo prep --submit
python -m sinkholes hpo approve --phase baseline --note "first matched pair"
python -m sinkholes hpo step                 # dry run: exactly two training jobs
python -m sinkholes hpo step --submit
python -m sinkholes hpo status               # any time, also from the Mac
python -m sinkholes hpo step --submit        # periodically: evaluations, collection, continuations
```

Later phases are the same two lines: `approve --phase shared|tpe|promote|seeds`, then
`step --submit`. Long runs: `hpo long --geometry plain --geometry ctx50 --seeds 42`,
`approve --phase long`, `step --submit`.

## 11. Verification record

* Test suite: 628 passed, 6 skipped (552 before this work; 76 new across
  `test_hpo_training.py`, `test_snapshot_eval.py`, `test_hpo_controller.py`, `test_hpo_panel.py`).
* `test_hpo_controller.py::test_the_full_protocol` drives a whole study against a fake LSF:
  approvals, duplicate guards, stalls and retries, TPE asks and tells, promotion, seed repeats,
  the locked threshold, retention, long-run creation, the report. Every generated command line
  is parsed by the real `train` / `eval-snapshots` parsers.
* Real-data engine check: **passed** (LSF 797689, 2026-10-14, L40S, 32 min; output
  `/home/labs/rudich/pinkas/sinkholes/studies/engine_check_20261014_0942/report.json`). Legacy
  `eval-scenes` + `eval-outputs` vs `eval-snapshots`, the fill k10 `best.pt`, whole AOI, t2026 val
  scenes 20230116_20230127 (offsets 13..4) and 20230322_20230402 (offsets 12..3).
  - tile batch 1: confidence **bit-identical** on both scenes (51.0 M and 16.0 M pixels scored),
    every recall and precision difference exactly 0 at all four RTh.
  - tile batch 32 (the study's plain setting): 75 and 33 pixels differ (batch-size floating point
    on CUDA); recall identical, largest precision difference 4.2e-5.
  - wall time per checkpoint for the two scenes: legacy 911 s, new 514 s (tb 32) / 427 s (tb 1).

## 12. Open risks

1. **ctx50 host memory (~295 GB est.).** If 340 GB cannot be scheduled on `long-gpu`, the
   remedy is an exact de-duplicated patch store (a predecessor frame shared by several targets
   held once; same samples, same order): measured 4.4× on positives, ~1.9× overall because ring
   negatives rarely coincide, i.e. ~190 GB. Not built yet.
2. **Optuna on WEXAC** must be 5.0.0 to share the journal with any other machine.
3. **Evaluation timings** are estimates until the first evaluation job logs its seconds per
   scene and checkpoint; `tile_batch` can be lowered if a GPU runs short of memory (the effective
   batch is recorded with the results).
4. **Batched inference on CUDA** may differ from the one-tile path in the last bits; a vote
   flips only if a tile probability lands within that of 0.5. The same engine scores every trial,
   so the comparison is unaffected. Measured by the engine check (§11): ≤ 75 flipped pixels per
   scene, precision ≤ 4.2e-5 off, recall unchanged.
5. **Preemption on `long-gpu`.** LSF 790941 (ctx50/s0) was preempted twice in its first 8 h and
   requeued by LSF under the same job id; each restart resumes from the last completed epoch but
   rebuilds the dataset first (up to 6.7 h on lgn06). A ctx50 trial can lose more time to
   rebuilding than to training; the de-duplicated store in (1) would also shorten the build.
