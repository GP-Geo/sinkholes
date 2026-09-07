# Experiments — what each batch was for, and what it settled

*Written 2026-09-03, from the ~1700 lines of commentary that used to live inside
`scripts/submit_all.sh`. That file is now a launcher; this is the notebook.*

A batch is listed here once it has run. **A batch that appears here is not in the
launcher** — `submit_all.sh` carries only work that has never run, so
`submit_all.sh <kind> --submit` can never silently repeat finished GPU-hours.

To recover the exact job definitions of any retired batch:

```bash
git show d708fba:scripts/submit_all.sh     # the full 2032-line version
```

For the numbers see [PREDICTIONS.md](PREDICTIONS.md) (object level, the ones
that count), [RESULTS.md](RESULTS.md) (the readable summary) and
[MODEL_RUNS.md](MODEL_RUNS.md) (the per-run registry).

---

## The standing rule

**Patch dice cannot rank these models.** Ring negatives go to the *train* split
only and validation is positives-only, so suppressing scene-scale false
positives has almost no upside on the val curve. That prediction held exactly:
all nine arms of 2026-08-10 landed in 0.639–0.660, at or under the ±0.006 noise
floor, with the geo arms slightly *down* — while object-level F1 on the same
models moved **+0.096**. Every verdict below comes from `run_eval.sh`, not from
`results.csv`.

---

## Training batches

### `train` — negative sampling (2026-08-10, 9 arms)

The batch that fixed over-prediction. Ring negatives — training patches drawn
from a ring *around* each sinkhole rather than only on it — lifted precision
**20 % on geo and 60 % on temporal** with no recall loss on geo. The first
change in the project that clearly moved the metric that matters, and the one
that taught the lesson above: dice said it did nothing.

Archived under `outputs/archive_2019_2026_noisy_data/`.

### `tattn` / `control` (2026-08-11, 4 arms)

Two geo attention arms and two single-frame controls. All four finished and were
scored. **The two attention evaluation directories were deleted in the
2026-08-20 prune**, so `geo_k5_tattn_ring3` (obj F1 0.683) and
`geo_k5_tattn_hybrid_ring3` can no longer be re-scored or checked. This is why
the 2026-09-01 attention rows matter: they are the live replacement for a
comparison that had become unverifiable.

### `clean22` — the clean benchmark (2026-08-19, 14 arms)

First batch on generation-3 partitions: AOI-windowed, latitude cut at 31.4°,
`_testeval` splits. ~50 GPU-hours. **Do not resubmit.** Its checkpoints were
deleted 2026-08-20, which is what made six of the `eval5` jobs permanently
unrunnable.

This batch is also where the attention block was found to have **collapsed to
uniform** — it trained, converged and produced respectable dice while doing
temporal *averaging*, not selection. `contrast` + `qk_norm` is the fix.

### `valpos` / `attnfix` (2026-08-19)

The same correction applied twice: validation moved to positives-only, because a
`valneg` curve scores an empty prediction on an empty mask as dice 1.0 and is
therefore inflated by roughly half a negative-cleanliness term. **Validation has
been positives-only for every job since 2026-08-20**; the `valneg` kind is
rejected by name so the mistake stays visible.

Nine runs survive under `outputs/2026-08-19/` with `best.pt`. Four of them are
the `eval6ref` anchors, one of the live batches in the launcher.

### `attnpos` (2026-08-20, 8 arms) — now `outputs/2026-08-20/*_valpos`

Four `attnfix` arms re-run on positives-only validation, plus four new temporal
arms. **Never a full architecture sweep** — it is an attention-correction batch,
which is why it ships no single-frame arm and only one ConvLSTM. That gap is
what `eval6ref` exists to fill.

### `pre23` (2026-08-20, 11 arms) — now `outputs/2026-08-20/*_pre2023_*`

The same generator, cut, AOI, seed and val share as `clean22`, with the archive
stopped at 2022-12-31 — so a `pre23` run read against its twin measures the
newer training years and nothing else.

**Verdict: the 2023–2026 training years buy nothing measurable.** `pre23` is
ahead in 4 of 6 matched pairs and significantly ahead in two, despite an archive
ending three and a half years before the test scenes.

### `long200` — the best temporal model taken to completion (2026-09-03, 1 arm)

`outputs/long200_t5_convlstm_ring3_200e_2026-09-03_16h22_lsf_643128`, LSF 643128.
Ran all 200 epochs in 11h44m (3m31s/epoch), peak VRAM 23.4 GiB allocated /
25.8 GiB reserved.

Its premise was that `pre23_temporal_k5_convlstm_ring3` — the best temporal
model on record at object F1 0.780 — had never converged, because its 2026-08-20
log stops mid-epoch at 91/100 with no completion line. **That premise was
wrong**: the original's `best.pt` is from epoch 51, and the 40 epochs that
followed it improved nothing. Nothing was lost to the interruption.

**Verdict on the dice curve: 200 epochs bought nothing.** Best val/dice 0.6622
@ epoch 200, against the original's 0.6626 @ epoch 51 — a difference of −0.0004,
inside the ±0.006 noise floor.

`LR_PATIENCE=20` did work mechanically. The original hit the 1.95e-08 `--min_lr`
floor at epoch 75 and spent its last 16 epochs unable to move; long200 cut the
LR five times and ended at 3.13e-07, live the whole way. It landed in the same
place regardless, so **the floored LR was not what limited the original** — a
clean negative result about the schedule.

**Settled 2026-09-07 by `evallong200`** (LSF 580743): object F1 **0.7529**
against `temporal_k5_pre2023_convlstm_ring3`'s **0.7797**, on the convention
RESULTS.md ⑨ quotes (`ith0.7_b5`, threshold 0.25, mean of per-scene). That is
**−0.0268**, more than four times the ±0.006 noise floor, in the wrong
direction. long200's rising dice curve — best on the final epoch, a new best at
198 — did *not* correspond to a better detector, which is RESULTS.md ② holding
once more: dice ranks these models backwards.

At the looser tolerance (`ith0.5_b10`) long200 *wins* at low vote thresholds
(+0.018 at 0.125, +0.011 at 0.25) and loses at 0.5. It finds more and localises
worse. Both figures belong to the pre-`ab23b07` tree, where AMP clipping
renormalised nearly every step, so they rank two models against each other and
say nothing about either in absolute terms.

---

## Evaluation batches

| Kind | What | Status |
|---|---|---|
| `eval` … `eval4` | generation-2 backlog, 2026-08-09 → 08-12 | done |
| `eval5` | clean benchmark, 8 of 14 runs | done; the other 6 permanently unrunnable |
| `posonly` | positives-only, the paper's protocol | done |
| `eval6` | **all 19 runs of 2026-08-20**, RTh protocol | done 2026-09-01, LSF 984720–984749 |
| `probe` | selectivity of all 12 attention arms | done 2026-09-02 |
| `eval7` | the 11 `pre23` arms on their own era | done 2026-09-02, LSF 261436–261464 |
| `eval6ref` | 4 positives-only anchors | **LIVE** |
| `evallong200` | long200 on the `eval6` protocol | done 2026-09-07, LSF 580743 — **0.7529 vs 0.780** |

### `eval6` (2026-09-01)

Nineteen object-level scores on generation-3 ground. Audited comparable
2026-09-02: identical scene sets within each axis, a common 23-scene geo
denominator, and all 19 JSON summaries reproducing their logs to 4 decimals.

Settled: **recurrence beats the single-frame U-Net** (+0.038 geo, +0.047
temporal, both significant) and it is a *recall* gain. **Attention never beats
recurrence** — not on either axis, not at either depth; at k10 it loses
significantly on both.

### `probe` (2026-09-02)

All 12 attention checkpoints, on real data. **8 of 12 select** — 7 of 8 pure
arms, so `contrast` + `qk_norm` works. **3 of 4 hybrids fail the collapse
alarm**, two at exactly 1.000, indistinguishable from the pre-fix control: when
a ConvLSTM already integrates over time, nothing pushes the attention to select.

But selectivity correlates with object F1 at **r = −0.10**, and only 1 of 7
attention arms beats its matched ConvLSTM. *Is the attention real?* is now
answered yes — and it was not the question that mattered.

The probe **runs on a laptop in ~5 min per checkpoint**, I/O-bound, no cluster
needed, despite `run_probe.sh` being written as a `bsub` job.

### `eval7` (2026-09-02)

The 11 `pre23` arms on `partition_*_testeval_pre2023.json`, whose scenes are
100 % LiDAR-mapped where the generation-3 temporal split is 0 %. The gating
worked — zero fallback warnings — but the batch **does not answer the
false-positive question**: the temporal splits share no scenes and
`min_positives=150` cut 35 scenes to 8 whose GT area is a third of `eval6`'s.
See [PREDICTIONS.md §3c](PREDICTIONS.md). The decisive test — same scenes,
`--no-add_lidar_mask` — has not been run.

---

## Retired, and why

**`long500`, `ctx50`, `evallong200`** (2026-09-07). `evallong200` completed and
is recorded above. `long500` and the first `ctx50` pair were submitted (LSF
580750/580752/580753) and **killed the same morning**, unrun, when the
correctness hotfix `ab23b07` landed: it corrects AMP gradient clipping, and
`check_config_compatible` refuses to resume an `--amp` checkpoint written
without `gradient_clipping=unscaled-v2`. Those runs could not have been
continued on this tree, and their premise changed anyway — `long500` existed to
spend more epochs on long200's rising curve, which `evallong200` has since shown
buys nothing. Replaced by `ampfix`.

**`blend`** (Hann-window stitching, 2 jobs). Both checkpoints —
`outputs/2026-08-11/geo_k5_tattn_ring3` and
`outputs/2026-08-10/geo_k5_convlstm_ring3` — were deleted, so the jobs fail
preflight. The one data point that exists is negative: `geo_k5_single_ring3`
scores **0.640 blended against 0.662 flat**. Reviving this needs a retrain
first.

**`rescore`** (`geo_k10_convlstm_posw8`, 3 of 5 thresholds missing). Its
evaluation directory holds metrics, figures and shapefiles but **no
`_pred.npy`** — it was pruned before 2026-09-03, and re-scoring reads exactly
that. Only a full `run_eval.sh` can rebuild it.

**`valneg`** — retired with negative validation itself (2026-08-20). Rejected by
name rather than left to select zero jobs.

**`archive_2026-08-18_runs.sh`** — a one-off that already ran, kept for a
fortnight as a record. It expected 14 run dirs that no longer exist and died
silently with status 1. Deleted 2026-09-03; this entry is the record.

**Correlation maps** (2026-08-19, was `docs/PLAN_CORRELATION.md`) — the
coherence rasters at `deadsea_sinkholes_data/{004,013}/` were tested as a way to
explain false positives and rejected. **Do not re-audit the data**: 436/437
interferograms covered, byte-complete, aligning to the phase grid within
0.02 px. The audit was clean; the signal is absent.

Two hypotheses, both null, on 20,581 predicted polygons from
`geo_k5_convlstm_ring3` labelled TP/FP by the project's own object rule:

| Hypothesis | AUC (0.5 = nothing) |
|---|---|
| errors sit in low-coherence ground now | 0.534 |
| errors follow decorrelated history | 0.512 |
| **polygon area** | **0.828** |

Coherence does not even mark sinkholes: 0.578 inside GT polygons against 0.588
outside.

> ### ✗ Settled negative, 2026-09-03 — do not implement an area filter
>
> This note used to read "the one result worth acting on, and it is still
> unimplemented", and asked for a re-measurement on the generation-3 benchmark
> before hard-coding a cutoff. That re-measurement has been done, and it kills
> the recommendation.
>
> `temporal_k5_pre2023_convlstm_ring3`, all 20 `eval6` temporal scenes, every
> combination of RTh 0.125–0.5 with an area floor of 0–800 px, at both
> tolerances — scored by a reimplementation that reproduces the published
> `olm_results` to four decimals at floor 0.
>
> **Not one of the 24 filtered configurations beats simply raising RTh.** Best
> F1 over the whole grid is 0.7797 at RTh 0.25 and floor **0**, which is the
> published number. Against the plain-threshold curve interpolated to the same
> recall every filtered point is *worse*; the best is −0.0023 precision
> (RTh 0.25 + 25 px), and the soft tolerance agrees (best −0.0040, best F1
> again at floor 0).
>
> **The area signal replicates; the conversion to metric does not.** FP median
> is 35–67 px against TP median ~300–330 px, so area really does separate
> true from false. But the object-level metric is **area-weighted**
> (`training/evaluate.py:98-105` — a false positive contributes its *area*, not
> a count), and at RTh 0.25 a 50 px floor removes **47.5 % of false positives by
> count but only 4.1 % of false-positive area**. The original "half the false
> positives for 2.5 % of recall" was a *count* statistic being read as a
> precision claim. That is the whole error.
>
> **RTh already is an area filter, and a better one.** Across 0.125 → 0.5 the FP
> count falls 6,881 → 1,961 *and* the median FP area rises 35 → 67 px — the
> threshold prunes small blobs while also using confidence, which size alone
> throws away.
>
> **The recall cost is not small sinkholes.** Only 0.2 % of GT area sits in
> objects under 100 px (GT median 639 px), yet a 100 px floor still costs 4
> points of recall: large GT objects are covered by several disconnected
> prediction blobs, and deleting the small ones drops coverage below the 0.7
> detection threshold. You lose big sinkholes, not small ones.
>
> **One operational caveat.** At RTh 0.125 a 50 px floor cuts 6,881 predicted
> polygons to ~2,900 at no F1 cost. Where a human reviews shapefiles that is
> half the objects to look at for free — a reason to offer a floor on the
> *export* path, never as a scoring default.
>
> Measured on temporal / RTh / one model, where the original was geo / prob /
> generation-2. The count-versus-area mechanism is a property of the metric, not
> the model, so geo should behave the same; ~20 min of saved-`_pred.npy` work to
> confirm if it ever matters. `sinkholes/polygons.py` still has no
> `min_area_px` argument, and on this evidence it does not need one.

If anyone reopens the coherence question, the untested version is: evaluate
twice, once normally and once with `eval-scenes --replicate_input`, then split
scenes by predecessor coherence. That distinguishes "bad history does no harm"
from "the ConvLSTM already discounts it by reading the phase" — the one thing
the null results above cannot separate. No new code, two eval runs.

*Measurement scripts and CSVs were deleted 2026-08-19 by request; regenerating
them is ~40 min of polygon/raster work.*

---

## Housekeeping of 2026-09-03

- `outputs/` **2.2 TB → 483 GB.** `_image.npy` (1.58 TB) is the input stack and
  no metric reads it; `_pred_th.npy` (0.25 TB) was written and read by nothing.
  `_pred.npy` and `_gt.npy` are kept so `rescore.sh` still works. Proven
  metric-neutral on one directory first: 374 numeric fields, 0 differences.
- `resume.pt` and `last.pt` dropped from 28 settled runs (16.7 GB); every
  `best.pt` kept, including all 12 attention checkpoints.
- The 19 runs of 2026-08-20 renamed to the convention and filed under
  `outputs/2026-08-20/`, with their 19 evaluation directories renamed to match.
  `pre23_<part>` became `<part>_pre2023` (it marks a different *training set*,
  so it belongs where the partition is named); `_fixed` dropped, since every arm
  now carries it. See [OUTPUTS.md](OUTPUTS.md) for the mapping.
