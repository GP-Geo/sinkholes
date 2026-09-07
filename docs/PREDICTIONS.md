# Predictions — the short version

*Last updated 2026-09-03. Covers **two** output trees: `outputs/predictions/`
(full-scene) and `outputs/predictions_positives/` (positives-only, the benchmark
paper's protocol). For the model-level story see [RESULTS.md](RESULTS.md); for
the positives-only protocol see [POSITIVES_ONLY_EVAL.md](POSITIVES_ONLY_EVAL.md);
for attention selectivity see [MODEL_RUNS.md](MODEL_RUNS.md).*

**This is where the numbers that matter live** — object-level precision and
recall over whole interferograms, not patches.

Counted on disk 2026-09-03: **56 evaluation job directories**, each holding
exactly one `olm_results*.json`.

- **53 under `predictions/`** — 16 generation-2 full-scene evaluations on two
  scene lists (§2), 2 Hann-window reruns of models already in that table, the
  5 clean-benchmark jobs of §3a, the **19 `eval6` jobs of §3b** (2026-09-01) and
  the **11 `eval7` jobs of §3c** (2026-09-02).
- **3 under `predictions_positives/`** (§3), patch *and* scene level, all on the
  held-out `test` split.
- The **35 generation-3 evaluations** (§3a + §3b + §3c) are the object-level
  numbers measured on ground the model never trained on. §2's are not.

Separately, `outputs/attention_probe/` holds **18 selectivity probes** — the 12
attention arms of the 2026-08-20 batches (run 2026-09-02) and the 6 references
of 2026-08-19. Those are not evaluations and carry no F1; see
[MODEL_RUNS.md](MODEL_RUNS.md), *Does the attention actually select?*

> ### ⚠️ Every attention evaluation in this file has lost its directory
>
> §2 counts 10 geo + 11 temporal; §3 lists five positives-only models. **Seven
> of those job directories no longer exist, and all seven are attention arms:**
>
> | Model | Missing from |
> |---|---|
> | `geo_k5_tattn_ring3` | §2 geo **and** §3 |
> | `geo_k5_tattn_hybrid_ring3` | §2 geo **and** §3 |
> | `temporal_k5_tattn_ring3` | §2 temporal |
> | `temporal_k5_tattn_hybrid_ring3` | §2 temporal |
> | `temporal_k5_tattn_base` | §2 temporal |
>
> Their rows below are now the **only** record of those numbers. None can be
> re-scored, extended to a new threshold, or reproduced: no `_pred.npy` remains,
> and no `tattn` run directory or checkpoint from 2026-08-10 or 2026-08-11
> survives under `outputs/` or `outputs/archive_2019_2026_noisy_data/` either.
> The oldest attention weights that still exist are the 2026-08-19 `valneg` runs.
> **Finding ① below — "attention does not beat recurrence on geo" — therefore
> rests on two numbers that can no longer be checked.**

---

## The one-paragraph summary

The four `geo_k5` runs that `RESULTS.md` listed as "trained, not yet scored" are
**now scored** — attention does *not* beat recurrence on geo, and temporal
context is worth **+0.04 to +0.05 object F1** there, far more than dice implied.
Separately, the positives-only tree is now populated, and it behaves exactly as
predicted: **recall is bit-identical to full-scene, precision is not.** Every
scene scores 0.86–0.99 precision under it, including the scene that scores
**0.09** full-scene. Consequently the worst geo model in the project is the
**best** one under positives-only. Use that tree to compare against the paper.
Never to choose a model.

---

## 1. Layout

```
predictions/<model>/best/scenes_<list>/         ← full scene
predictions_positives/<model>/best/posonly_<list>/   ← positive tiles only
    olm_results_<ts>.json    ← THE RESULT: per-scene and mean P/R at 5 thresholds
    <intf>_pred.npy          confidence map (needed to re-score)
    <intf>_gt.npy            ground truth
    <intf>_pred_th.npy       thresholded prediction
    <intf>_overview.png      per-scene figure
    polygs/, *.shp           polygonised detections

predictions_positives/<model>/patches_test.json  ← patch level, one level up
```

`<model>` is the training-run directory name, so a result maps to its run by
name alone. `<list>` is `geo` (18 scenes) or `temporal` (20 scenes).

Reading a result:

```bash
python3 -c "import json;d=json.load(open('<dir>/olm_results_<ts>.json'));print(d['summary'])"
```

`summary` gives mean recall and precision at confidence 0.125 / 0.25 / 0.5 /
0.7 / 0.9; `per_intf` gives the same per interferogram. Both trees carry
`ol_th` (0.7) and `buffer` (5) — every number below uses those.

> Throughout this file, **F1 is computed from the mean precision and mean
> recall** across scenes, so it summarises the table rather than being measured
> separately. "Best F1" is the best of the five thresholds.

---

## 2. Full-scene results — `predictions/`

Every evaluation falls on exactly one of two scene lists, and **within a list
all of them are directly comparable**:

| List | Scenes | Evaluations |
|---|---|---|
| geo | 18 | 10 |
| temporal | 20 | 11 |

The geo list carries both `geo_k5` and `geo_k10` models, so k5 and k10 *can* be
compared there — which patch dice cannot do.

### Geo — 18 scenes

| Model | Best F1 | at conf. | R / P there |
|---|---|---|---|
| `geo_k5_convlstm_ring3_3x` | **0.724** | 0.9 | 0.773 / 0.681 |
| `geo_k5_convlstm_ring3` | 0.712 | 0.7 | 0.784 / 0.652 |
| `geo_k5_tattn_hybrid_ring3` | 0.708 | 0.7 | 0.820 / 0.623 |
| `geo_k5_tattn_ring3` | 0.683 | 0.7 | 0.803 / 0.594 |
| `geo_k5_convlstm_base` | 0.667 | 0.9 | 0.657 / 0.677 |
| `geo_k10_convlstm_ring3` | 0.666 | 0.7 | 0.785 / 0.578 |
| `geo_k5_single_ring3` | 0.662 | 0.9 | 0.669 / 0.656 |
| `geo_k5_convlstm_posw8` | 0.647 | 0.9 | 0.696 / 0.605 |
| `geo_k5_single_base` | 0.629 | 0.9 | 0.663 / 0.598 |
| `geo_k10_convlstm_posw8` | 0.456 † | 0.5 | 0.893 / 0.306 |

† scored at only 3 of the 5 thresholds and cut off above its own operating
point — a floor, not a peak. `submit_all.sh rescore` fixes it cheaply.

### Temporal — 20 scenes

| Model | Best F1 | at conf. | R / P there |
|---|---|---|---|
| `temporal_k5_convlstm_ring10` | **0.698** | 0.25 | 0.673 / 0.726 |
| `temporal_k5_tattn_ring3` | 0.642 | 0.25 | 0.663 / 0.621 |
| `temporal_k5_tattn_hybrid_ring3` | 0.639 | 0.25 | 0.667 / 0.613 |
| `temporal_k5_convlstm_ring3` | 0.626 | 0.25 | 0.664 / 0.591 |
| `temporal_k5_convlstm_base` | 0.612 | 0.5 | 0.609 / 0.616 |
| `temporal_k10_convlstm_posw4` | 0.601 | 0.5 | 0.577 / 0.628 |
| `temporal_k5_tattn_base` | 0.598 | 0.5 | 0.589 / 0.608 |
| `temporal_k10_convlstm_posw8` | 0.581 | 0.5 | 0.585 / 0.577 |
| `temporal_k5_single_b64` | 0.581 | 0.5 | 0.676 / 0.509 |
| `temporal_k5_convlstm_run2` | 0.576 | 0.5 | 0.550 / 0.605 |
| `temporal_k10_stack` | 0.562 | 0.5 | 0.545 / 0.581 |

### What the four new geo evaluations settled

They were `RESULTS.md`'s top priority; they have now run.

**① Attention does not beat recurrence on geo.** The hybrid ties ConvLSTM
(0.708 vs 0.712, inside noise); pure attention is **0.03 behind**. `RESULTS.md`
④ made attention the default off a temporal-only pairing — that no longer holds
across partitions, and geo is the partition we care about.

| geo, ring3 1:1 | Best F1 |
|---|---|
| `geo_k5_convlstm_ring3` | 0.712 |
| `geo_k5_tattn_hybrid_ring3` | 0.708 |
| `geo_k5_tattn_ring3` | 0.683 |

**② Temporal context is worth much more at object level than at patch level.**
Same partition, same negatives, single-frame against ConvLSTM:

| Pair | single-frame | ConvLSTM | gap |
|---|---|---|---|
| geo, ring3 1:1 | 0.662 | 0.712 | **+0.050** |
| geo, baseline | 0.629 | 0.667 | **+0.038** |

Dice put this gap at **+0.012**. Object F1 puts it at 4× that. This does not
overturn `RESULTS.md` ③ — it says the dice measurement understates the effect,
which is the same lesson as finding ② there.

**③ Ring negatives hold on a single-frame model too**: 0.629 → 0.662, +0.033.
The gain is architecture-independent.

### The Hann blending window — one data point, and it is negative

`predictions/geo_k5_single_ring3/best/scenes_geo_hann` re-runs that model with a
Hann tile-blending window instead of the uniform one. It is **worse**:

| `geo_k5_single_ring3` | Best F1 | at conf. | R / P there |
|---|---|---|---|
| uniform (`scenes_geo`) | **0.662** | 0.9 | 0.669 / 0.656 |
| Hann (`scenes_geo_hann`) | 0.640 | 0.9 | 0.698 / 0.592 |

Hann trades precision (−0.064) for recall (+0.029) and loses 0.022 F1 on net —
the wrong direction for geo, where precision is the binding constraint. The
matching runs for `geo_k5_convlstm_ring3` and `geo_k5_tattn_ring3` were killed
before finishing and need resubmitting; **one model is not enough to settle
this**, but the sign is worth knowing before spending the GPU hours.

---

## 3. Positives-only results — `predictions_positives/`

The paper's protocol: score only where subsidence is known to be. At scene
level `--positives_only` predicts only tiles whose ground truth is non-empty —
**about 1.2% of tiles** (e.g. 211 of 17,533 on `20230812_20230823`).

**These numbers exist to compare against the paper and for nothing else.**
Every one of them is on `test`, the split no model has seen.

### Scene level

| Model | List | Best F1 | at conf. | R / P there | full-scene best F1 |
|---|---|---|---|---|---|
| `geo_k10_convlstm_posw8` | geo | **0.911** | 0.5 | 0.893 / 0.929 | 0.456 † |
| `geo_k5_tattn_ring3` | geo | **0.911** | 0.5 | 0.899 / 0.924 | 0.683 |
| `geo_k5_convlstm_ring3_3x` | geo | 0.904 | 0.5 | 0.870 / 0.941 | 0.724 |
| `geo_k5_tattn_hybrid_ring3` | geo | 0.904 | 0.5 | 0.887 / 0.921 | 0.708 |
| `temporal_k5_convlstm_ring10` | temporal | 0.815 | 0.125 | 0.722 / 0.936 | 0.698 |

### Patch level (`th 0.7`, buffer 5, `test` split)

| Model | dice | pixel P / R | object P / R | patches |
|---|---|---|---|---|
| `geo_k5_tattn_ring3` | 0.635 | 0.75 / 0.74 | 0.90 / 0.86 | 4,445 |
| `geo_k10_convlstm_posw8` | 0.632 | 0.72 / 0.74 | 0.90 / 0.85 | 4,445 |
| `geo_k5_tattn_hybrid_ring3` | 0.631 | 0.75 / 0.74 | 0.90 / 0.85 | 4,445 |
| `geo_k5_convlstm_ring3_3x` | 0.625 | 0.78 / 0.70 | 0.88 / 0.81 | 4,445 |
| `temporal_k5_convlstm_ring10` | 0.490 | 0.91 / 0.44 | 0.69 / 0.56 | 8,519 |

This dice is on **test**, not the val dice in `results.csv` — do not put the two
in one table.

### ① Recall is identical; only precision moves

Across all five paired models and every shared threshold, positives-only mean
recall matches full-scene mean recall to **|ΔR| ≤ 0.0006** — zero for four of
the five. This is the documented behaviour, now confirmed on real outputs: a
tile holding a ground-truth pixel is positive by definition, so restricting to
positive tiles removes no recall. **Any recall difference you see between the
trees is a configuration mismatch, not a protocol effect.**

### ② Precision inflates by up to 10×, worst exactly where it matters

Per-scene precision @0.5, `geo_k5_convlstm_ring3_3x` — full-scene → positives-only:

| Scene | full | positives-only |
|---|---|---|
| `20241127_20241208` | **0.09** | **0.92** |
| `20241219_20241230` | 0.28 | 0.96 |
| `20250602_20250613` | 0.29 | 0.95 |
| `20250818_20250829` | 0.30 | 0.97 |
| … | … | … |
| `20210304_20210315` | 0.95 | 0.98 |

Under positives-only **every scene lands between 0.86 and 0.99**, and the
spread across scenes — the thing worth investigating — is gone. The worst scene
in the project (`RESULTS.md` trap ②) becomes an ordinary one. The protocol
cannot see over-prediction because over-prediction happens on the 98.8% of
tiles it never looks at.

### ③ It ranks models backwards — the concrete case

`geo_k10_convlstm_posw8` is **last on geo full-scene (0.456)** and **first
under positives-only (0.911)**, on the same 18 scenes. Meanwhile the best
full-scene geo model (`geo_k5_convlstm_ring3_3x`, 0.724) is *third* here. Ring
negatives cost a little on-target precision and buy a lot of off-target
precision; positives-only prices only the first half. **This is the same
failure as ranking by dice, for the same reason.**

### ④ The best threshold moves too

Full-scene peaks at 0.7–0.9 (geo) and 0.25 (temporal). Positives-only peaks at
**0.5 (geo)** and **0.125 (temporal)** — with precision near-saturated, the
recall side of F1 decides, so lower thresholds win. Quote the protocol *and*
the threshold, always.

---

## 3a. The clean benchmark — generation-3 partitions, RTh protocol (5 evaluations)

*Added 2026-09-01. These are the **first object-level scores measured on ground
the model has never seen** — and they are not comparable to §2 or §3 on any
axis. Read this section against itself only.*

Five of the eight queued `eval5` jobs completed on 2026-08-19 (job name
`scenes_{geo,temporal}_clean_rth_08_19_14h01`). Three axes are new at once:

| | This section | §2 / §3 |
|---|---|---|
| partitions | generation 3, `*_clean.json`, hard 31.4° cut | generation 2, frame split — **train/hold-out overlap** |
| reconstruction | `DATA_STRIDE=4`, 16 tiles per interior pixel | stride 2, 4 tiles |
| threshold | **RTh** — fraction of overlapping tiles voting positive, each tile binarised at 0.5 | `recon_th` — a cut on a mean probability |
| scene lists | geo 35 (32 scored), temporal 22 | geo 18, temporal 20 |

**An RTh number and a `recon_th` number count different things** — RTh 0.25
means "at least 4 of 16 tiles agreed". Per the paper's protocol, **recall is the
primary metric**: full-range reconstruction scores a great deal of unannotated
ground, so precision partly measures the digitisation rather than the model.

Every number below is at tolerance **ITh 0.7 / buffer 5** (the strict one),
averaged over the scenes where the metric is defined. F1 is computed from the
mean recall and mean precision, so it summarises the row rather than being
measured separately.

**Geo** — `geo_k10`'s 35-scene test list; **32 scored**, 3 excluded. The three
(`20230801_20230812`, `20231222_20240102`, `20240102_20240113`) sit in an AOI
window holding no ground truth, so `gt_area` is 0 and object recall/precision
are 0/0. Each file carries a `repair` block recording the recomputation, and
`per_intf` still reports them as NaN.

| Model | RTh | recall | precision | F1 |
|---|---|---|---|---|
| `geo_k5_convlstm_ring3_valneg` | 0.125 | **0.863** | 0.322 | 0.469 |
| | 0.25 | 0.829 | 0.360 | 0.502 |
| | 0.375 | 0.810 | 0.379 | 0.516 |
| | **0.5** | 0.797 | **0.400** | **0.532** |
| `geo_k5_convlstm_ring3_3x_valneg` | 0.125 | 0.829 | 0.325 | 0.467 |
| | 0.25 | 0.796 | 0.347 | 0.483 |
| | 0.375 | 0.785 | 0.361 | 0.494 |
| | **0.5** | 0.767 | 0.395 | 0.522 |
| `geo_k5_single_ring3_valneg` | 0.125 | 0.845 | 0.237 | 0.370 |
| | 0.25 | 0.820 | 0.273 | 0.410 |
| | 0.375 | 0.807 | 0.295 | 0.432 |
| | **0.5** | 0.795 | 0.320 | 0.456 |

**Temporal** — `temporal_k10`'s 22-scene test list, all 22 scored:

| Model | RTh | recall | precision | F1 |
|---|---|---|---|---|
| `temporal_k5_convlstm_ring3_valneg` | **0.125** | **0.755** | 0.645 | **0.695** |
| | 0.25 | 0.678 | 0.701 | 0.689 |
| | 0.375 | 0.637 | 0.729 | 0.680 |
| | 0.5 | 0.601 | 0.754 | 0.669 |
| `temporal_k5_single_ring3_valneg` | **0.125** | 0.686 | 0.639 | 0.662 |
| | 0.25 | 0.617 | 0.704 | 0.658 |
| | 0.375 | 0.566 | 0.732 | 0.638 |
| | 0.5 | 0.516 | 0.751 | 0.612 |

### ① Recurrence beats the single-frame U-Net on both axes — in opposite currencies

The single-frame control is behind on every threshold of both scene lists, and
the *shape* of the deficit differs by axis:

| Axis, at RTh 0.5 | single-frame R / P | ConvLSTM R / P | what recurrence bought |
|---|---|---|---|
| geo (32 scenes) | 0.795 / 0.320 | 0.797 / **0.400** | **+0.080 precision at identical recall** |
| temporal (22 scenes) | 0.516 / 0.751 | **0.601** / 0.754 | **+0.085 recall at identical precision** |

On geo the precision margin holds across all four thresholds (+0.080 to +0.087);
on temporal the recall margin holds across all four (+0.061 to +0.085). Neither
model gives anything back on the other metric.

**The patch curve says the opposite on geo.** On the positives-only reruns of
2026-08-20 the single-frame arm is *top* of its geo batch on dice and `val/F1`
(`MODEL_RUNS.md`, "The 2026-08-20 batches"). Object level reverses it. This is
the same mechanism as `RESULTS.md` finding ②, in its sharpest form yet: the
quantity that separates these two models on geo is **precision on background**,
and a positives-only patch curve contains no background to be wrong about.

### ② Geo precision is much lower here than §2 reports, and the old number was inflated

§2 quotes `geo_k5_convlstm_ring3` at precision 0.652 / F1 0.712. The same
architecture, same negatives, on generation-3 ground reads **0.400 / 0.532** at
RTh 0.5. Both the partition and the threshold definition changed, so this is not
a clean subtraction — but the direction is the one `MODEL_RUNS.md` predicts: the
generation-2 geo split put the 31.25–31.44° band in train *and* in the hold-out,
so **every geo object-level score in §2 was measured partly on training ground.**
This section replaces that table for geo; it does not extend it.

### ③ 3:1 negatives do not survive the move to clean ground

`geo_k5_convlstm_ring3_3x` was "the best model in the project" at obj F1 0.724
on generation 2, with 1:1 second. Here it is **behind 1:1 at every threshold** —
0.522 against 0.532 at RTh 0.5, and lower on recall throughout (0.767 vs 0.797).
The gap is small, but the ordering that the old recommendation rested on is
gone, and the old recommendation was itself an object-level-only finding.

### ④ The softer tolerance moves everything, and it moves temporal most

Every evaluation also carries **ITh 0.5 / buffer 10** in `summary_by_tolerance`,
plus a GT-area-weighted aggregate alongside the unweighted mean. At RTh 0.25:

| Model | strict R / P | soft R / P | area-weighted soft R / P |
|---|---|---|---|
| `geo_k5_convlstm_ring3_valneg` | 0.829 / 0.360 | 0.883 / 0.418 | 0.835 / 0.562 |
| `geo_k5_single_ring3_valneg` | 0.820 / 0.273 | 0.864 / 0.298 | 0.814 / 0.388 |
| `temporal_k5_convlstm_ring3_valneg` | 0.678 / 0.701 | **0.889 / 0.756** | **0.869 / 0.834** |
| `temporal_k5_single_ring3_valneg` | 0.617 / 0.704 | 0.827 / 0.760 | 0.797 / 0.830 |

Temporal recall gains **+0.21** from the tolerance alone. The area-weighted
column is uniformly kinder to precision than the plain mean on both axes, which
says the low-precision scenes are the small-GT ones — consistent with §4's trap
② at the scene level. **Quote the tolerance and the aggregation with every
number here**; three of them differ by more than any architecture in this file.

### ⑤ What these five evaluations cannot be extended to

All five scored a **`valneg`-selected checkpoint**, and every checkpoint in the
2026-08-18 batch was deleted on 2026-08-20. Stored `_pred.npy` means
`scripts/eval/rescore.sh` can still add thresholds or tolerances to these five;
**nothing can add a scene, and no other clean22 model can ever be scored.** The
three `eval5` jobs that never ran — `eval_clean_g5_tattn`,
`eval_clean_g5_tattn_hybrid`, `eval_clean_g10_tattn_hybrid` — are therefore
permanently unrunnable as submitted. ~~No attention model of any kind has an object-level score on
generation-3 partitions.~~ **Superseded 2026-09-01: §3b scores twelve of
them.** What remains true is that no *clean22* attention checkpoint survives to
be scored.

---

## 3b. `eval6` — all nineteen 2026-08-20 runs on generation-3 ground (2026-09-01)

Nineteen object-level evaluations, LSF 984720–984749, same RTh protocol as §3a
(stride 4, Confidence Factor, RTh 0.125/0.25/0.375/0.5, ITh 0.7/b5 and 0.5/b10).
Results at
`<model>/best/scenes_{geo,temporal}_clean_rth_09_01_*/olm_results_0901*.json`.

**Comparability was audited 2026-09-02 and holds.** All 9 geo runs share one
identical 26-interferogram set and all 10 temporal runs one identical 20-scene
set, despite differing directory timestamps (restart artefacts). The three
undefined geo scenes are GT-empty and the *same three* for every model, so all
geo means are over a common 23 scenes — no model is flattered by dropping scenes
it failed. All 19 JSON summaries reproduce their job logs to 4 decimals. Both
families use positives-only validation, so checkpoint selection is not a
confound between them.

### Geo — 23 scored scenes, best F1 of four thresholds (ITh 0.7/b5)

| Model | F1 | P | R | @th |
|---|---|---|---|---|
| `geo_k5_pre2023_convlstm_ring3` | **0.648** | 0.568 | 0.754 | 0.5 |
| `geo_k10_pre2023_convlstm_ring3` | 0.647 | 0.570 | 0.748 | 0.5 |
| `geo_k10_pre2023_tattn_hybrid_ring3` | 0.641 | 0.565 | 0.742 | 0.5 |
| `geo_k5_pre2023_tattn_ring3` | 0.638 | 0.540 | 0.781 | 0.5 |
| `geo_k10_tattn_hybrid` | 0.630 | 0.522 | 0.795 | 0.5 |
| `geo_k10_tattn` | 0.626 | 0.525 | 0.773 | 0.5 |
| `geo_k5_pre2023_single_ring3` | 0.612 | 0.510 | 0.766 | 0.5 |
| `geo_k5_tattn` | 0.595 | 0.469 | 0.812 | 0.5 |
| `geo_k10_pre2023_tattn_ring3` | 0.584 | 0.448 | 0.838 | 0.5 |

### Temporal — 20 scenes

| Model | F1 | P | R | @th |
|---|---|---|---|---|
| `temporal_k5_pre2023_convlstm_ring3` | **0.780** | 0.793 | 0.767 | 0.25 |
| `temporal_k10_tattn_hybrid` | 0.772 | 0.773 | 0.772 | 0.125 |
| `temporal_k10_pre2023_convlstm_ring3` | 0.772 | 0.801 | 0.744 | 0.125 |
| `temporal_k5_pre2023_tattn_ring3` | 0.768 | 0.723 | 0.818 | 0.125 |
| `clean_temporal_k10_convlstm_ring3` | 0.761 | 0.769 | 0.754 | 0.125 |
| `temporal_k5_tattn` | 0.759 | 0.740 | 0.779 | 0.125 |
| `temporal_k10_tattn` | 0.751 | 0.773 | 0.731 | 0.125 |
| `temporal_k10_pre2023_tattn_ring3` | 0.748 | 0.729 | 0.767 | 0.125 |
| `temporal_k5_tattn_hybrid` | 0.747 | 0.730 | 0.765 | 0.125 |
| `temporal_k5_pre2023_single_ring3` | 0.733 | 0.781 | 0.691 | 0.25 |

### ① Recurrence beats the single-frame U-Net, and it is a recall gain

Paired per-scene bootstrap, 10k resamples, 95% CI. With 20–23 scenes anything
below ~0.02 does not separate from noise.

| Comparison | ΔF1 | 95% CI | |
|---|---|---|---|
| geo — ConvLSTM k5 vs single | +0.038 | +0.021 … +0.054 | **significant** |
| geo — attention k5 vs single | +0.027 | +0.014 … +0.041 | **significant** |
| geo — ConvLSTM k10 vs attention k10 | +0.066 | +0.018 … +0.111 | **significant** |
| temporal — ConvLSTM k5 vs single | +0.047 | +0.019 … +0.073 | **significant** |
| temporal — ConvLSTM k10 vs attention k10 | +0.030 | +0.003 … +0.060 | **significant** |
| geo — ConvLSTM k5 vs attention k5 | +0.011 | −0.001 … +0.024 | not resolved |
| geo — ConvLSTM k10 vs hybrid k10 | +0.007 | −0.006 … +0.020 | not resolved |
| temporal — attention k5 vs single | +0.031 | −0.009 … +0.064 | not resolved |
| temporal — ConvLSTM k5 vs attention k5 | +0.016 | −0.002 … +0.036 | not resolved |
| temporal — ConvLSTM k10 vs hybrid k10 | −0.011 | −0.030 … +0.008 | not resolved |

On both axes the single-frame U-Net is the most conservative model in the field —
competitive on precision, last on recall. Everything that beats it does so by
finding more sinkholes, not by being cleaner. This confirms §3a ① on nineteen
runs instead of five.

### ② Attention never beats recurrence — at k10 it loses significantly on both axes

In no comparison, on either axis, at either depth, does attention beat ConvLSTM.
The hybrid ties it. §3a's finding ① held on two numbers that no longer exist
(see the banner at the top of this file); it now rests on nineteen live ones.
The selectivity probe (`MODEL_RUNS.md`, *Does the attention actually select?*)
settles why: 8 of 12 arms genuinely select, but measured selectivity correlates
with object F1 at **r = −0.10**, and only 1 of 7 attention arms beats its
matched ConvLSTM.

### ③ The 2023–2026 training years buy nothing measurable

Six matched pairs where the only difference is the training archive — same
architecture, same depth, same positives-only validation, same scored ground.
`pre23` is ahead in **4 of 6** and significantly ahead in two (geo k5 tattn
+0.053 [+0.013, +0.087]; geo k10 hybrid +0.049 [+0.020, +0.079]), despite an
archive that stops three and a half years before the test scenes. No pair
favours `clean` significantly.

### ④ ⚠️ 17 of 19 peak at the edge of the sweep — every geo F1 here is a lower bound

All 9 geo runs rise monotonically across 0.125 → 0.5 and peak at **0.5, the top
of the sweep**: the true optimum is above it and unmeasured. On temporal, 8 of
10 peak at **0.125, the bottom**, and fall from there; only
`temporal_k5_pre2023_convlstm` and `temporal_k5_pre2023_single` peak at an interior
threshold. Because the geo curves are all still climbing, **the geo ranking
itself could reorder under a wider sweep.** Re-scoring is cheap — the `_pred.npy`
are on disk and `scripts/eval/rescore.sh` can add thresholds.

---

## 3c. `eval7` — the eleven `pre23` arms on their own era (2026-09-02)

Requested to remove the stale-LiDAR false-positive load: `assets/lidar_intf_mask.txt`
ends 20240605, so on the generation-3 **temporal** test split **0 of 22**
interferograms have a mapping of their own — every one falls back to LiDAR2022.
Geo is 28 of 35. The `pre2023` test splits are **22 of 22** and **35 of 35**
mapped, on LiDAR2019/2020/2021. `GEN=p23` was added to `run_eval.sh` on
2026-09-02 for this (it previously accepted only 2 and 3). LSF 261436–261464,
results at `<model>/best/scenes_*_pre23_rth_09_02_14h50/`.

The AOI windows are **identical** between the clean and pre2023 testeval files on
both axes, so an `eval7` row differs from its `eval6` row by the scene list and
nothing else.

| Model | F1 | P | R | @th | n |
|---|---|---|---|---|---|
| `geo_k5_pre2023_convlstm_ring3` | 0.564 | 0.472 | 0.701 | 0.5 | 14 |
| `geo_k10_pre2023_convlstm_ring3` | 0.552 | 0.465 | 0.680 | 0.5 | 14 |
| `geo_k10_pre2023_tattn_hybrid_ring3` | 0.546 | 0.420 | 0.778 | 0.125 | 14 |
| `geo_k5_pre2023_tattn_ring3` | 0.540 | 0.439 | 0.701 | 0.5 | 14 |
| `geo_k10_pre2023_tattn_ring3` | 0.518 | 0.384 | 0.793 | 0.5 | 14 |
| `geo_k5_pre2023_single_ring3` | 0.504 | 0.401 | 0.679 | 0.5 | 14 |
| `temporal_k10_pre2023_convlstm_ring3` | 0.702 | 0.578 | 0.893 | 0.5 | 8 |
| `temporal_k5_pre2023_single_ring3` | 0.651 | 0.495 | 0.949 | 0.5 | 8 |
| `temporal_k10_pre2023_tattn_ring3` | 0.648 | 0.506 | 0.903 | 0.5 | 8 |
| `temporal_k5_pre2023_convlstm_ring3` | 0.630 | 0.482 | 0.910 | 0.5 | 8 |
| `temporal_k5_pre2023_tattn_ring3` | 0.628 | 0.478 | 0.915 | 0.5 | 8 |

**The LiDAR gating worked** — zero fallback warnings across all eleven jobs. And
the pipeline verifies: the 9 geo scenes present in both `eval6` and `eval7`
reproduce across 216 comparisons at a **median Δ of exactly 0** (mean 9.4e-05,
one outlier at 0.0136 where a borderline polygon flips at the lowest threshold),
with identical `gt_area`.

### ⚠️ These numbers do NOT answer the false-positive question — and they cannot

Precision fell everywhere: −0.06 to −0.14 on geo, **−0.22 to −0.31 on
temporal**, with recall rising. That is the opposite of what removing a
false-positive load should do, and the reason is that the comparison is not
measuring the mask:

- **Temporal has zero scene overlap** with `eval6`, and `min_positives=150`
  dropped **27 of 35** scenes. The 8 survivors have median GT area **54k against
  eval6's 167k** — a third the target size. The same absolute false-positive
  load against a third the true area gives exactly a large precision drop with
  recall held up. **At n=8 the temporal rows are too thin to lean on regardless.**
  Lowering `min_positives` would recover the sample: 50 → 15 scenes, 25 → 21.
- **On geo**, decomposing the −0.084 on `k5_convlstm`: the 9 shared scenes are
  the hard core (mean per-scene F1 0.475), while the 14 `eval6`-only scenes —
  which *include* the fallback-LiDAR 2023–2025 ones — average **0.731**. The
  scenes running on the stale mask are the **easier** ones.

The direction actually suggests the opposite of the original hypothesis: the
mask *gates* predictions, so a 2022 mask on a 2025 scene suppresses anything in
ground the 2022 LiDAR never saw — flattering `eval6` precision and costing it
recall, which is the observed pattern (eval6 temporal P 0.78–0.80 / R 0.74–0.77
against eval7 P 0.48–0.58 / R 0.89–0.92). **Suggestive, not settled** — the
splits are disjoint, so mask effect and scene composition cannot be separated
here.

**The decisive test is cheap and has not been run:** re-score the *same*
2025–2026 scenes with `--no-add_lidar_mask` and compare like for like. One
model, 20 scenes, no retraining.

### How to read `eval7`

Comparable **across its own eleven rows and nothing else.** The `eval6`→`eval7`
gap on one model is not an error bar; it is the LiDAR change plus four years of
drift plus a different scene mix, confounded.

---

## 4. Traps — read before quoting a number

**① Never mix the trees.** A positives-only precision and a full-scene
precision are not the same quantity. Both code paths say so in their logs;
`run_eval_positives.sh` says so in its header.

**② Geo eval-scenes reads a `_testeval` partition.** The positives-only geo jobs
pass `assets/partition_geo_k10_testeval.json`, which carries the test list under
`"val"` — that is how the test split reaches `eval-scenes`. `test-patches` reads
the parent partition with `--split test` instead. Two different files, same 18
scenes; do not "fix" one to look like the other.

**③ `geo_k10_convlstm_posw8` is cut off at 0.5** in the full-scene tree — its
0.456 is a floor, not a peak.

**④ `*_image.npy` was pruned from 7 full-scene directories on 2026-08-11** and
this is deliberate — no metric reads it. Re-scoring (`scripts/eval/rescore.sh`)
and every stored number still work; only `--save_figures` does not, and it now
fails with a clear message. Verified before pruning: re-scoring a pruned
directory reproduces its stored numbers exactly, on both North and South frame
paths.

**⑤ A model's name no longer records its pos_w.** `geo_k10_convlstm_posw8` and
`geo_k5_convlstm_posw8` are pre-`pos_w`-4 runs and are not valid references for
anything in the 2026-08-10 batch onward. Everything without a `posw*` suffix is
`pos_w` 4.

---

## 5. What to do next

1. **Give the geo baseline a positives-only score.** Four of the five entries
   are ring-negative or old models; without `geo_k5_convlstm_base` there is no
   positives-only ring-negatives comparison, which is the one claim the paper
   comparison will be challenged on.
2. **Add `geo_k5_convlstm_ring3`** — the second-best full-scene geo model has no
   positives-only number, so the tree's top of table is currently decided by a
   model nobody would deploy.
3. **Rescore `geo_k10_convlstm_posw8`** at 0.7 and 0.9 in the full-scene tree, so
   the geo table has no floor-only row.
4. **Score `geo_k5_convlstm_ring10`** — still the missing cell in the negatives
   grid, and far-field negatives were the biggest temporal win.
5. **Revisit `RESULTS.md` ④** with §2 finding ① above: attention as the default
   architecture rested on temporal alone and geo disagrees.

### Housekeeping

94 GB of superseded positives-only job directories were deleted 2026-08-17.
What remains prunable:

| What | Size |
|---|---|
| `predictions/`, total | ~900 G |
| — `*_image.npy` in 14 unpruned dirs (the `eval2`/`eval3` batches) | **499 G** |
| `predictions_positives/`, total | ~310 G |
| — `*_image.npy`, none pruned yet | **288 G** |

The input stacks are 55–85 GB per evaluation and no metric reads them. Prune
both trees the way 2026-08-11 pruned the first seven.
