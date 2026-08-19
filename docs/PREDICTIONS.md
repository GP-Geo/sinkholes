# Predictions — the short version

*Last updated 2026-08-17. Covers **two** output trees: `outputs/predictions/`
(full-scene) and `outputs/predictions_positives/` (positives-only, the benchmark
paper's protocol). For the model-level story see [RESULTS.md](RESULTS.md); for
the positives-only protocol see [POSITIVES_ONLY_EVAL.md](POSITIVES_ONLY_EVAL.md).*

**This is where the numbers that matter live** — object-level precision and
recall over whole interferograms, not patches.

- **21 full-scene evaluations**, on two scene lists.
- **5 positives-only evaluations**, patch *and* scene level, all on the held-out
  `test` split.

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
