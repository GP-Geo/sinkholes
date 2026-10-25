# HPO and long training: plain vs ctx50 temporal attention (k10, generation 5-t2026)

Study `k10t2026_hpo_v2`, specified in [`assets/hpo/k10t2026_plain_vs_ctx50_v2.json`](../assets/hpo/k10t2026_plain_vs_ctx50_v2.json)
and run by `sinkholes hpo`. It is `k10t2026_hpo_v1`
([`assets/hpo/k10t2026_plain_vs_ctx50.json`](../assets/hpo/k10t2026_plain_vs_ctx50.json)) with
the RTh list extended above 0.5 in steps of 1/16 up to 0.9375, because both v1 baseline trials
peaked at v1's 0.5 ceiling; v1's finished baseline pair is adopted, not retrained, and v1 is closed (§13). This
page is the protocol; the code is the source of truth for every definition quoted here.

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

**Micro-batch.** 32×4 holds for both geometries: the baseline pair trained at it (VRAM, §9:
ctx50 33.5 GiB reserved, first measured in LSF 254183; plain 11.5 GiB). The optional `prep`
profile job was not run in v1.

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
significant figures; pos_w 2/4/4/12/12 is assigned so it is not confounded with lr. **The
design is fixed**: `hpo adopt` requires v2's shared configurations to equal v1's, so a different
design is a new study that retrains the baseline pair.

Then, per geometry: **two TPE trials** (Optuna `TPESampler`, `n_startup_trials=6` so the six
shared trials are the startup set, multivariate, constant liar, seed 20261013) → **the best two
of eight continued to 60 epochs** by full-state resume → **the best promoted configuration
repeated at seeds 7 and 123** for 60 epochs.

## 3. The objective (never patch Dice)

Whole **validation** scenes (the partition's 11 `val` scenes, 2022-2023; AOI and LiDAR gate as always),
stride-4 grid, unweighted tile votes at 0.5, every RTh ∈ {0.125, 0.25, 0.375, 0.5} and every
1/16 step above, {0.5625, 0.625, …, 0.9375} (v1 stopped at 0.5; §13), strict
object metric ITh 0.7 / buffer 5 px. Object F1 is this project's: the harmonic mean of the
per-scene **mean** recall and **mean** precision (`F1 = 2·P̄·R̄/(P̄+R̄)`; 0.8114 for the fill
k10 model on its 24-scene test list reproduces this way). A scene without ground truth has no
recall or precision and is left out of the means: 3 of the 11 val scenes (20231029_20231109,
20231222_20240102, 20231223_20240103), so the objective averages 8. Those three *are*
labelled in the current GT (86–207 polygons), but their stride-4 masks were cut in June 2026
from the previous GT file, which had not labelled them yet
([`DATASET_11DAY.md`](DATASET_11DAY.md#known-stale)). Every trial is scored on the same 8, so
comparisons inside the study hold; the study keeps it that way (decided 2026-10-25) and the
masks are rebuilt after it. The objects predicted on those three (110–262 per scene for the
baseline pair) never reach the objective.

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
  RTh, saved compressed; GT once per scene in the study cache, no input stacks, no shapefiles.
  Any other threshold can be scored later from these files on CPU (`rescore_saved`, §13);
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

The study root is **`/home/labs/rudich/pinkas/sinkholes/studies/k10t2026_hpo_v2`**. The closed
v1 root beside it, `studies/k10t2026_hpo_v1` (moved from `pinkas/sinkholes_studies/` on
2026-10-15; job scripts and run manifests written before that keep the old path), still holds
the two adopted baseline runs, which v2's ledger points at: keep it. `studies/` is gitignored and outside `outputs/`, so `scripts/tidy_outputs.sh`
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

## 8. Data coverage (`sinkholes hpo coverage`)

| geometry | use | trees | state |
|---|---|---|---|
| plain | training | stride 2, 341 ids | complete, headers intact |
| plain | val / test scenes | stride 4, 80 / 32 ids | complete |
| ctx50 | training | stride 2, 341 ids | complete |
| ctx50 | val scenes (2022-23 chains) | stride 4, 80 ids | complete (built 2026-10-13, LSF 790948, 27 min) |
| ctx50 | test scenes (2026 chains) | stride 4, 32 ids | complete |

The ctx50 val tree was the one gap; it is rebuilt, should it ever be needed, with

```bash
# resumable, checks df, verifies itself
PARTITION=assets/partition_temporal_k10_test2026_fill_th350x200.json K_PREVS=10 MIN_POS=150 \
  bsub < scripts/data/make_context_stride4_patches.sh
```

The controller refuses to submit a ctx50 evaluation until the tree is complete.

## 9. Resources (`sinkholes hpo resources`)

| | plain | ctx50 | source |
|---|---|---|---|
| min/epoch, 32×4 | **14.1** (A40) | **42.5** (A40), **26.9** (L40S) | measured on the baseline pair, LSF 790942 / 790941 (87,834 samples); the earlier ~19.2 extrapolation for ctx50 was wrong |
| dataset build | **1 h 43 m** | **2 h 30 m – 6 h 42 m** | measured (hgn40; hgn48 / lgn06), paid again on every LSF requeue |
| host memory, training | **114 GB** peak | **297.8 GB** peak | measured (LSF 790942 / 790941) |
| VRAM | 11.5 GiB | 33.5 GiB | measured |
| evaluation, 3 checkpoints × 11 val scenes | **1 h 09 m**, 23.9 GB peak | **3 h 04 m**, 68.7 GB peak | measured (LSF 870786 / 870780) |

Requested: training plain 160 GB / 24 GB GPU, ctx50 340 GB / 40 GB GPU, both 24 h wall with a
22.5 h budget (continuations are automatic chunks); evaluation plain 80 GB, ctx50 192 GB. The
evaluation requests are about 3× what was used, but `resources` is part of the adopted protocol
and stays v1's.

Per trial, from those numbers: a plain 30-epoch trial is one chunk, ≈ 8.8 h + 1.2 h evaluation. A
ctx50 30-epoch trial does not fit one 22.5 h budget on an A40 (30 × 42.5 min + the build), so it
runs as two chunks and builds twice, ≈ 26–35 h; on an L40S ≈ 16–20 h; evaluation ≈ 3.1 h.
Totals (estimates, A40 to L40S): the 10 shared trials ≈ 145–240 GPU-h; HPO through the seed
repeats ≈ 400–550 GPU-h; the six 300-epoch runs ≈ 700–1,050 GPU-h more (pre-launch estimates:
340 and 640, which assumed ctx50 at ~1.4× plain per epoch instead of the measured 3.0× on A40).

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

That block is how v1 started (without `prep`, which is optional). v2 starts from v1's finished
baseline pair instead (§13):

```bash
python -m sinkholes hpo init                 # v2 is the default --spec now
python -m sinkholes hpo adopt                # CPU, ~40 min: re-score v1's pair, close v1
python -m sinkholes hpo step --submit        # collects the pair and tells Optuna; submits nothing
python -m sinkholes hpo status
```

Later phases are the same two lines: `approve --phase shared|tpe|promote|seeds`, then
`step --submit`. A `DUPLICATE` line means one bsub queued a job twice and the controller
killed the copy (§12, risk 5); if it says bkill failed, run the `bkill` it prints. Long runs: `hpo long --geometry plain --geometry ctx50 --seeds 42`,
`approve --phase long`, `step --submit`. v1 is still readable with
`--spec assets/hpo/k10t2026_plain_vs_ctx50.json` (`status`, dry-run `step`); everything that
would change it is refused.

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
* v1 → v2 (§13), 2026-10-18: test suite 645 passed, 6 skipped. Six tests are new:
  `test_snapshot_eval.py` checks that re-scored maps equal a fresh evaluation at the wider
  list, and that a changed map is refused with nothing written; `test_hpo_controller.py`
  covers adoption end to end against the fake LSF (shared run, closed source, collect and tell,
  ten shared trials under v2's names, a promotion resuming with v1's fingerprint), refusal of a
  changed protocol, refusal of an unfinished run, and refusal of a changed snapshot.
* Real data, 2026-10-18: `rescore_saved` on local copies of v1's baseline evaluations (both
  geometries, epochs 10/20/30, 11 scenes each). Every per-scene field reproduced at v1's four
  thresholds, and the objectives at v2's eight are §13's (plain 0.7568 @ 0.875, ctx50 0.7401 @
  0.9375). It took ~12 min on one Mac core; most of that is polygonising at the low thresholds.
* v2 started 2026-10-24: `init` at 99a1812 (clean tree), `adopt` on the WEXAC login node in ~40
  min (v1's four thresholds reproduced), then collected and told to Optuna: plain/s0 **0.7568** @
  0.875, ctx50/s0 **0.7401** @ 0.9375, as §13 predicted. The new 1/16 steps did not move either
  best (plain 0.7420 @ 0.5625, 0.7560 @ 0.6875).
* `shared` phase submitted 2026-10-24: ten training jobs (LSF 341833–341844). The guard below
  caught one LSF duplicate on its first use (plain/s1: 341839 killed, 341840 kept). All ten
  runs recorded a clean tree at da449d9.
* Duplicate-submission guard, 2026-10-24: `test_hpo_controller.py` drives a bsub that queues a
  job twice (the twin is killed, the recorded job kept), a crash with two jobs queued, and a
  dry-run `step` that must leave `events.jsonl` untouched; full suite 625 passed, 6 skipped.

## 12. Open risks

1. **ctx50 host memory (297.8 GB measured, 340 GB requested).** The baseline was scheduled on
   `long-gpu` at that request. The remedy, if it stops being scheduled, is an exact de-duplicated patch store (a predecessor frame shared by several targets
   held once; same samples, same order): measured 4.4× on positives, ~1.9× overall because ring
   negatives rarely coincide, i.e. ~190 GB. Not built yet.
2. **Optuna on WEXAC** must be 5.0.0 to share the journal with any other machine (the Mac has
   4.9.0: run every `hpo` command that touches Optuna on WEXAC).
3. **Batched inference on CUDA** may differ from the one-tile path in the last bits; a vote
   flips only if a tile probability lands within that of 0.5. The same engine scores every trial,
   so the comparison is unaffected. Measured by the engine check (§11): ≤ 75 flipped pixels per
   scene, precision ≤ 4.2e-5 off, recall unchanged.
4. **Preemption on `long-gpu`.** LSF 790941 (ctx50/s0) was preempted twice in its first 8 h and
   requeued by LSF under the same job id; each restart resumes from the last completed epoch but
   rebuilds the dataset first (up to 6.7 h on lgn06). A ctx50 trial can lose more time to
   rebuilding than to training; the de-duplicated store in (1) would also shorten the build.
5. **Duplicate LSF submissions.** Both v1 baseline evaluations ran twice: ctx50 as 870777 and
   870780 (submitted in the same second), plain as 870783 and 870786. The ledger holds only the
   second id of each pair and the controller calls `bsub` once per action; the cause is not
   established (one candidate: `bsub` retrying internally while LSF was slow). Two identical
   evaluations wrote the same files, so the scores are unaffected, at a cost of ~4 GPU-h. Two
   copies of a *training* chunk would both write the run's `resume.pt`. Since 2026-10-24 the
   controller asks LSF for the job name right after each bsub (and after adopting a job lost
   in a crash) and kills every other live job under it, keeping the recorded id
   (`controller.kill_twins`; ledger `killed_twins`, event `duplicate`, a `DUPLICATE` line).

## 13. v1 → v2: the threshold ceiling (2026-10-17/18)

**What v1 showed.** v1 scored the paper's RTh grid, {0.125, 0.25, 0.375, 0.5}. Both baseline
trials (s0, pos_w 8) scored best at 0.5, the highest value allowed, and the objective rose at
every step: plain 0.632 → 0.688 → 0.714 → 0.733, ctx50 0.674 → 0.685 → 0.693 → 0.704.
Precision was the weak side (~0.59 against recall 0.93–0.98): the models over-predict, and a
stricter vote removes false objects.

**The check.** The saved maps keep each pixel's raw votes and counts, so the pair was
re-scored on CPU at higher thresholds with the study's own scoring code; at the four v1
thresholds every per-scene field reproduced exactly. A pixel's confidence is votes ÷ tiles,
and ~87 % of pixels are covered by 16 tiles at stride 4, so only steps of 1/16 matter (0.55
scores like 0.5, 0.65 like 0.625). The top level, 0.9375, means all 16 tiles agree.

| RTh | votes needed (of 16) | plain F | P / R | ctx50 F | P / R |
|---|---|---|---|---|---|
| 0.5 (v1's top) | ≥ 9 | 0.7329 | 0.587 / 0.976 | 0.7040 | 0.568 / 0.927 |
| 0.6 | ≥ 10 | 0.7423 | 0.600 / 0.974 | 0.7070 | 0.573 / 0.924 |
| 0.625 | ≥ 11 | 0.7486 | 0.611 / 0.967 | 0.7139 | 0.582 / 0.923 |
| 0.7 | ≥ 12 | 0.7560 | 0.622 / 0.963 | 0.7178 | 0.588 / 0.922 |
| 0.75 | ≥ 13 | 0.7516 | 0.633 / 0.926 | 0.7244 | 0.598 / 0.920 |
| 0.8125 | ≥ 14 | 0.7542 | 0.646 / 0.905 | 0.7288 | 0.604 / 0.919 |
| 0.875 | ≥ 15 | **0.7568** | 0.658 / 0.891 | 0.7329 | 0.612 / 0.915 |
| 0.9375 | 16 | 0.7542 | 0.674 / 0.856 | **0.7401** | 0.625 / 0.909 |

F is the objective (mean F1 over epochs 20 and 30); P and R are the per-scene means averaged
over the same two epochs; "votes needed" holds where 16 tiles cover the pixel. Plain levels
off from 0.7 up (0.752–0.757); ctx50 is still rising when every tile must agree, even though
its precision there is only 0.625. Plain still leads, by 0.017 at each geometry's best
instead of 0.029 at 0.5.

**Why it matters for the search.** pos_weight is searched ({2, 4, 8, 12}), and a larger
pos_weight makes a model say "sinkhole" more readily, which a stricter vote compensates. With
the grid capped at 0.5, the configurations that need a stricter vote (pos_w 8 and 12: s0, s2,
s4) would be under-scored relative to pos_w 2 and 4, and TPE and promotion would learn "low
pos_weight is better" partly from the cap. This is reasoning, not a measurement: both finished
trials use pos_w 8.

**Why a new study.** A study's spec is hashed into its ledger and cannot change, and the s0
objectives had already been told to Optuna, which refuses a second value for a finished trial.
So v2 is v1 with `evaluation.rth` = {0.125, 0.25, 0.375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
0.8125, 0.875, 0.9375}: v1's four (so v1's numbers stay comparable and the re-scored maps can be
checked against them), then every 1/16 step to the unanimous level. Because the vote test is
strict (`confidence > RTh`), k/16 asks for k+1 of 16 votes: 0.5 needs ≥ 9 and the seven steps
above it need ≥ 10 … 16. Nothing else changed: data, partitions,
training, search space, shared configurations, budgets and resources are v1's, and `hpo adopt`
refuses otherwise. A longer threshold list makes the max over thresholds very slightly more
optimistic, by the same amount for every trial. Decided by the user on 2026-10-18 (eighths and
0.9375); completed to every 1/16 step on 2026-10-24, before `init`, because the eighths left the
≥ 10, ≥ 12 and ≥ 14 levels unscored. In the table above those rows were scored at 0.6, 0.7 and
0.8125; v2 scores them at 0.5625, 0.6875 and 0.8125, which give the same pixels where 16 tiles
cover and differ only where fewer do, so `adopt` reports the exact values.

**`sinkholes hpo adopt`** (`controller.adopt`, driven by the spec's `adopt` section). In order:

1. It refuses unless the two specs differ only in `study`, `root`, `description`, `adopt` and
   a grown `evaluation.rth`, and the two studies' frozen partitions are byte-identical.
2. For each named trial it refuses unless the v1 trial is complete and collected, its
   configuration matches v2's namesake field for field (`ADOPT_SAME`), the v2 trial has no jobs
   or scores yet, and every scored snapshot still has the SHA-256 it was scored with.
3. It re-scores each evaluation batch from its saved maps at v2's thresholds
   (`snapshot_eval.rescore_saved`). Every v1 threshold must reproduce field for field for every
   scene and epoch, or nothing is written. The results and copies of the maps go to v2's
   `evals/`; each JSON carries a `rescored` record (the source file and its SHA-256, the
   verified thresholds).
4. It shares the run directory rather than copying it (2.7 GB stays in v1's `runs/`). A
   promotion resumes it in place, with the run name and the partition path its strict resume
   fingerprint holds (the trial's `slug` and `partition_file` stay v1's).
5. It copies v1's min/max and GT caches (86 MB) into v2's `cache/`.
6. It marks v1 `superseded_by` v2. From then on, every v1 command that takes the ledger lock
   (`approve`, `step --submit`, `collect`, `long`, `prep --submit`, `retain --apply`) is
   refused, so nothing in v1 can touch the runs v2 now owns. `status` and dry-run `step` still
   work and say the study is closed.

It is idempotent (an adopted trial is skipped) and submits nothing. The next `step --submit`
in v2 collects the adopted evaluations and tells Optuna, like any finished evaluation; the
other ten initial trials then wait for `approve --phase shared` as before.
