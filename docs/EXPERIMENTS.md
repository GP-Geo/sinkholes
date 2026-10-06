# Experiments — what each batch was for, and what it settled

*Written 2026-09-03, from the ~1700 lines of commentary that used to live inside
`scripts/submit_all.sh`. That file is now a launcher; this is the notebook.*

A batch is listed here once it has run. The launcher's own rule is that
**a batch that appears here is removed from it** — `submit_all.sh` carries only work that
has never run, so `submit_all.sh <kind> --submit` can never silently repeat finished
GPU-hours.

> ⚠️ **That rule has drifted, as of 2026-09-17.** `lrscan`, `dropscan`, `lossscan`,
> `adamscan`, `combo`, `th350`, `evalth350`, `posw8`, `evalposw8`, `k10`, `k10plain`,
> `evalk10` and `evalk10plain` have all run and are all still in the launcher. Until they are
> pruned, `--submit` on one of those kinds **would** repeat finished GPU-hours — the exact
> failure the shield exists to prevent. Dry-run first (the default) and check this file
> before submitting anything. `eval6ref`, `reg` and `momscan` are unrun; `k5plain` and
> `k10plain45` were added on 2026-09-17. `longreg`, `longgrid` and `evallonggrid` **were** retired on 2026-09-17; their
> blocks were never committed, so `git show` cannot recover them — they are archived in
> [`reference/submit_all_retired_2026-09-17.sh`](reference/submit_all_retired_2026-09-17.sh).

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

## September 2026 — the correctness fix, and everything it forced

The 2026-09-07 hotfix (`ab23b07`, [CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md)) made AMP
gradient clipping unscale first. That one change invalidated the optimiser the whole
project had been using, and the eight batches below are the consequences, in order.

### `ampfix` (2026-09-07, 5 arms) — the batch that diverged

The first batch on the corrected tree, re-baselining the temporal ConvLSTM. **Three of
five arms diverged to NaN at epoch 2** and are kept as evidence under
`outputs/failed_2026-09-07_ampfix_divergence/` (LSF 612826 / 612827 / 612828): epoch 1
scored val/dice 0.469, epoch 2 val/loss `nan` and dice `1e-08`. The two survivors are in
`outputs/2026-09-07/`.

**The cause is the fix working.** RMSprop's momentum buffer is not bias-corrected, so at
the historical `--momentum 0.999` the effective step is `lr/(1−m)` — a **1000×**
amplification. That was survivable only while the AMP clipping defect was crushing every
gradient to a fixed tiny norm. With correct clipping the real gradient arrives, gets
multiplied by 1000, and the activations blow up inside one epoch. Nothing about the model
changed; the optimiser had been running on a bug.

### `lrscan` (2026-09-07, 6 × 8-epoch screens) — it was the momentum, not the LR

`outputs/screens/2026-09-07/`. Four learning rates at `m=0.999`, plus `lr 1e-5` at
`m=0.9` and `m=0.0`.

| arm | best val/dice (8 ep) |
|---|---|
| `lr1e5_m999` | **0.0000** — diverged at epoch 6, reproducing `ampfix` |
| `lr1e6_m999` | 0.6195 |
| `lr1e7_m999` | 0.6262 |
| `lr1e8_m999` | 0.5792 |
| `lr1e5_m090` | **0.6661** |
| `lr1e5_m000` | 0.6210 |

**Verdict: lowering the LR does not recover it.** Three decades of LR at `m=0.999` all
land at 0.58–0.63, while the *same* LR that diverged is the best arm in the batch once
the momentum comes down to 0.9. The step-size amplification is the defect, and it is not
reachable through `lr`.

### `adamscan` (2026-09-08, 5 × 20-epoch screens) — the replacement optimiser

`outputs/screens/2026-09-08/`. AdamW bias-corrects, so the step is ~`lr` whatever
`--beta1` is, and `adamw` decouples the weight decay.

| arm | best val/dice (20 ep) |
|---|---|
| `adamscan_lr3e4` | **0.6762** |
| `adamscan_lr1e4` | 0.6719 |
| `adamscan_lr1e4_wd1e2` | 0.6717 |
| `adamscan_lr3e5` | 0.6530 |
| `adamscan_lr1e5` | 0.6346 |

**`adamw lr 3e-4 wd 1e-2` became the standard**, and every batch from 2026-09-09 on runs
it. Note the spread across the top three is 0.0045 — inside the noise floor — so this
picked a *region*, not a point. An `lr` carried over from an RMSprop preset is not the
same step size; that is why the whole scan was needed rather than a translation.

### `dropscan` / `lossscan` (2026-09-08, 7 × 20-epoch screens) — two small yeses

| `dropscan` (ConvLSTM bottleneck Dropout2d) | best val/dice |
|---|---|
| `p00` | 0.6585 |
| `p01` | 0.6690 |
| `p02` | **0.6704** |
| `p03` | 0.6648 |

| `lossscan` (`--seg_loss`) | best val/dice |
|---|---|
| `dice_s7` | 0.6572 |
| `jac_s7` | 0.6633 |
| `jac_s42` | **0.6726** |

Dropout 0.1–0.2 is worth ~0.01 dice over none; Jaccard is ahead of Dice at both seeds.
Both are **screens**, both margins are ~1–2× the noise floor, and neither has an
object-level score. `DROPOUT_BOTTLENECK=0.2` was carried into the long batches; `SEG_LOSS`
stayed at `dice`.

### `combo` (2026-09-08, 4 × 20-epoch screens) — and an accidental noise measurement

| arm | best val/dice |
|---|---|
| `combo_ctx_p00` | **0.7017** |
| `combo_ctx_p00_dup` | 0.6908 |
| `combo_ctx_p02` | 0.6832 |
| `combo_p00` | 0.6605 |

Context is worth **+0.041 dice** over the plain geometry at otherwise identical settings,
which is large for this project. Dropout on top of context is not — it costs 0.018.

> ### ⚠️ The duplicate is the most important number in the batch
>
> `combo_ctx_p00` was submitted twice by accident (LSF 275654 / 275798). **Two runs of an
> identical configuration differ by 0.0109 val/dice** — nearly twice the ±0.006 noise
> floor every screen in `screens/` is being read against, and both peaked at epoch 7 of
> 20. It was measured incidentally and has never been measured on purpose. **Half the
> margins in this section are inside it.** Nothing has been re-read in light of it.

### `longreg` / `longsingle` / `longgrid` (2026-09-09/10) — context at full length

> 🗑️ **Deleted 2026-09-17.** All eight run directories and the six `evallonggrid`
> prediction directories (~294 GB) were removed after scoring: generation 3 still contains
> the badly digitised scenes the generation-4 threshold removes, so these runs are not
> kept for decisions. This section is the record; nothing here can be re-scored.

Sixty-epoch runs on `partition_temporal_k5_clean`, `adamw 3e-4 / wd 1e-2`, batch 64 × 2
accum = 128 effective, seed 42.

| run | best val/dice | @ epoch | note |
|---|---|---|---|
| `longreg_ring10` | 0.6988 | 16 | ConvLSTM ctx50, outer ring 10, hv, dropout 0.2 — **stopped at 19/60**, preempted and never requeued |
| `longreg_aughv` | 0.6972 | 16 | reached 52/60 |
| `longreg_noaug` | 0.6870 | 7 | **stopped at 19/60**, same preemption |
| `longsingle_ctx50` | 0.6900 | 23 | single frame, context |
| `longsingle_plain` | 0.6658 | 21 | single frame, no context |
| `longgrid_tattn_ctx50` | **0.7025** | 19 | 30 epochs |
| `longgrid_convlstm_plain` | 0.6767 | 23 | |
| `longgrid_tattn_plain` | 0.6734 | 25 | |

**Context replicates at full length and on a second architecture**: +0.024 dice on the
single-frame pair, +0.029 on the tattn pair, against `combo`'s +0.041 at 20 epochs. That
is three independent measurements of the same sign, which is more than any of them is
worth alone. Flips (`hv`) are worth ~+0.010 — one noise floor.

**Object-level scores (`evallonggrid`, 2026-09-17, LSF 316677–316682).** The six grid
cells on the generation-3 clean test list (20 scenes), RTh protocol, best of four
thresholds — the same list and convention as `eval6`:

| architecture | plain F1 `ith0.7_b5` (P / R) | ctx50 F1 `ith0.7_b5` (P / R) | Δ context | `ith0.5_b10` plain → ctx50 |
|---|---|---|---|---|
| single | 0.7167 (0.751 / 0.685) | 0.7089 (0.832 / 0.618) | −0.008 | 0.8400 → 0.8377 |
| convlstm | 0.7437 (0.778 / 0.712) | 0.7196 (0.829 / 0.636) | −0.024 | 0.8743 → 0.8475 |
| tattn | **0.7498** (0.793 / 0.711) | 0.7212 (0.847 / 0.628) | **−0.029** | 0.8759 → 0.8618 |

> **At the object level context loses in all three pairs, and dice ranked every pair
> backwards.** It buys +0.05 to +0.08 precision and costs −0.07 to −0.08 recall on every
> architecture, and the recall loss wins. That is the same trade the k10 single-frame pair
> shows on generation 4 ([`k10`](#k10--k10plain-2026-09-1617-4-arms--depth-helps-attention-context-costs-recall)),
> so the inversion is not a k10 or single-frame effect. The "+0.024 / +0.029" context gains
> above are patch dice only.

> ⚠️ **Superseded as a basis for decisions, 2026-09-17.** These runs train and score on the
> generation-3 partition, which still contains the badly digitised scenes the generation-4
> 350N/200S threshold removes ([`assets/PARTITIONS.md`](../assets/PARTITIONS.md)). Their
> absolute F1s sit below every generation-4 batch and are not comparable with them. What
> they still show is the *within-pair* direction of context, measured on one shared list.

### `th350` (2026-09-10, 4 arms) — the first generation-4 batch, and the best F1 on record

Four architectures on `partition_temporal_k5_clean_th350x200` — generation 4, the
label-quality threshold ([`assets/PARTITIONS.md`](../assets/PARTITIONS.md)) — at ctx50,
ring negatives out to 10, `hv` flips, `adamw 3e-4 / wd 1e-2`, pos_w 4, 60 epochs.
Scored by `evalth350` on 20 test scenes, RTh protocol, best of four thresholds.

| arm | obj F1 `ith0.7_b5` | P / R | obj F1 `ith0.5_b10` | val/dice |
|---|---|---|---|---|
| `th350_tattn_ctx50_neg10` | **0.7905** | 0.869 / 0.725 | **0.8888** | 0.6970 |
| `th350_hybrid_ctx50_neg10` | 0.7782 | 0.866 / 0.706 | 0.8737 | 0.7065 |
| `th350_single_ctx50_neg10` | 0.7535 | 0.859 / 0.671 | 0.8800 | 0.6859 |
| `th350_convlstm_ctx50_neg10` | 0.7489 | 0.879 / 0.652 | 0.8729 | 0.7035 |

All four peak at **RTh 0.125**, the bottom of the sweep — so, exactly as in `eval6`,
every F1 here is a **lower bound**.

> ### ⚠️ These numbers are NOT comparable with `eval6`, `eval7` or anything earlier
>
> The generation-4 scene list is **not** the generation-3 one: 17 of 20 overlap, 3 are
> added (no 10-previous chain, so k10 could never hold them) and 3 dropped (346/343/339
> positives, just under the cut). Both lists happen to score 20 scenes; they are not the
> same 20, and a mean over different scenes is not a comparison.
>
> **What the swap is worth was measured**: re-averaging the three finished generation-3
> ctx50 runs over only the above-threshold scenes moves object F1 by **+0.013 to +0.024**
> at every RTh, for all three models. That is a level shift from an easier list, not model
> quality — and it is the *only* term in the September-vs-August gap that has been
> isolated. These arms also change the optimiser, the objective (both correctness fixes),
> the geometry and the ring radius. Do not attribute the rest.
>
> Within this table the comparison is clean: same partition, same protocol, same 20 scenes.

**Two results the earlier batches did not give.** Attention is **+0.042 F1 over
recurrence** and +0.037 over the single-frame floor — the first time on this project that
attention has beaten a ConvLSTM on ground where both were trained identically, and it
reverses `eval6`'s finding ② on a different partition. And the gain is **recall**
(0.725 vs 0.652) at equal precision, which is the axis recurrence was supposed to own.
`val/dice` ranks this table backwards again: the ConvLSTM is second on dice and last on
F1.

**Arm 2 was designed to measure the threshold itself** — `th350_convlstm_ctx50_neg10` is
`longreg_ring10`'s configuration exactly, differing only in the partition. Dropout 0.2 is
on that arm only, because `--dropout_bottleneck` is refused on any other architecture;
the four are unmatched on that axis and it is worth saying so when reading them together.

### `posw8` (2026-09-14, 4 arms) — buying recall with the class weight

The same four architectures, the same partition, protocol and 20 scenes, with `pos_w 8`
and 20–25 epochs instead of 60.

| arm | obj F1 `ith0.7_b5` | P / R | obj F1 `ith0.5_b10` | Δ F1 vs its `th350` twin |
|---|---|---|---|---|
| `posw8_tattn_ctx50_25e` | 0.7699 | 0.850 / 0.704 | **0.8898** | −0.021 |
| `posw8_convlstm_ctx50_25e` | 0.7699 | 0.862 / 0.696 | 0.8870 | **+0.021** |
| `posw8_single_ctx50_20e` | 0.7676 | 0.850 / 0.700 | 0.8875 | +0.014 |
| `posw8_hybrid_ctx50_25e` | 0.7571 | 0.868 / 0.671 | 0.8803 | −0.021 |

**Verdict: `pos_w` 8 compresses the batch rather than lifting it.** The four arms land
within 0.013 F1 of each other, against a 0.042 spread at `pos_w` 4 — the weight buys the
weaker arms the recall they were missing (ConvLSTM +0.044 recall, single +0.029) and
costs the leader precision. The architecture ranking that `th350` established **does not
survive the weight change**, which means it is a property of the operating point, not of
the models. Under the soft tolerance the ordering is different again.

The recall lever this batch spent GPU-hours on was also available for free: re-reading
the saved confidence maps at RTh 0.125 instead of 0.25 buys tattn +0.018 recall for
−0.005 precision, and `eval-outputs` had already written all four thresholds.

### `k10` / `k10plain` (2026-09-16/17, 4 arms) — depth helps attention, context costs recall

`partition_temporal_k10_clean_th350x200` — 74 / 10 / 17 interferograms and 27,191 train
positives, against k5's 100 / 11 / 20 and 35,558 — at T=11, `pos_w` 8, 30 epochs,
`adamw 3e-4 / wd 1e-2`, ring 1–10, `hv`. tattn at batch 32 × 4, single at 64 × 2, both
128 effective. A 2×2 over {single, tattn} × {plain, ctx50}. Scored by `evalk10` on the 17
k10 test scenes (LSF 303197–303199) — **a different denominator from the k5 tables above.**

| arm | obj F1 `ith0.7_b5` | P / R | obj F1 `ith0.5_b10` | val/dice @ epoch |
|---|---|---|---|---|
| `posw8_tattn_k10_ctx50_30e` | 0.7748 | 0.858 / 0.706 | 0.8870 | 0.6986 @ 11 |
| `posw8_single_k10_plain_30e` | 0.7685 | 0.789 / 0.749 | 0.8798 | 0.6613 @ 21 |
| `posw8_single_k10_ctx50_30e` | 0.7427 | 0.871 / 0.647 | 0.8735 | 0.7060 @ 15 |
| `posw8_tattn_k10_plain_30e` | **0.7907** | 0.819 / 0.764 | **0.9056** | 0.6786 @ **29** of 30 |

`tattn_k10_plain` peaked at its second-to-last epoch and may be under-trained relative to
the other three (best epochs 11–21): a win for it at object level is robust to that, a
loss is not.

**Depth, on identical scenes.** The 17 k10 test scenes are a strict subset of the k5
generation-4 list, so the k5 `posw8` arms can be re-averaged over exactly these 17 from
their saved per-scene results — no new job:

| arm | k5 `posw8`, same 17 scenes | k10 | Δ |
|---|---|---|---|
| tattn ctx50 | 0.7681 (0.858 / 0.696) | 0.7748 (0.858 / 0.706) | +0.007 |
| single ctx50 | 0.7609 (0.850 / 0.689) | 0.7427 (0.871 / 0.647) | −0.018 |
| **tattn − single** | +0.007 | **+0.032** | |

k10 trains on 24% fewer positives. The single-frame arm, which cannot use the extra
history, loses 0.018 to that cut; the attention arm, under the same cut, gains 0.007. So
the temporal edge — attention over the single frame at matched data and context — widens
from +0.007 at k5 to +0.032 at k10. Each per-arm Δ is inside the 0.011 duplicate spread
(`combo`), so the widening edge is the reading, and it is one partition and one seed. For
scale, `th350_tattn_ctx50_neg10` (k5, `pos_w` 4) scores 0.7863 on the same 17 scenes and
is still the best model on them.

> **Context costs recall, and dice gets it backwards.** The plain single-frame arm is
> 0.045 val/dice behind the ctx50 arm and 0.026 object F1 ahead: context trades 0.102
> recall for 0.082 precision. The `longgrid` object scores show the same trade on all
> three architectures at k5 on generation 3, so it is neither a depth nor a single-frame
> effect — the earlier k5 "context helps" evidence was patch dice only. With history the
> loss is mostly bought back: ctx50 attention sits +0.006 over the plain single frame.
> **The plain attention cell completes the 2×2 and is the best k10 model** (`evalk10plain`,
> LSF 330515): **0.7907**, 0.016 over ctx50 attention, again on recall (0.764 vs 0.706).
> Context has now lost in five of five pairs. History helps at both geometries — attention
> over the single frame is +0.022 plain and +0.032 ctx50 — and context costs attention less
> (−0.016) than it costs the single frame (−0.026). On the same 17 scenes plain k10
> attention also edges the overall leader `th350_tattn_ctx50_neg10` (0.7863), by less than
> the noise. It never cut its learning rate in 30 epochs and peaked at 29: `k10plain45`
> retrains it for 45 epochs at `LR_PATIENCE=5`.

**Attention at k10** (probed 2026-09-17 on the same 480 val patches for both tattn cells;
[ATTENTION.md](ATTENTION.md#k10-history-carries-the-recall)).
Both select — 2.6 and 2.5 of 11 frames, 23% of uniform against 53% untrained, temperature
~10. The ctx50 model depends on its history far more than the k5 generation-4 runs did:
masking it costs −0.044 patch Dice (−0.004 for `posw8_tattn_ctx50_25e`), and the loss is
recall, 0.834 → 0.666 — the history is what recovers the recall context costs, which is
the object-level +0.059 recall over the ctx50 single frame. *Which* frames it picks barely
matters with context (forced uniform +0.002); at plain it matters more than the history
itself (−0.022 vs −0.013). Both models put heavy weight on the oldest frame in the window,
110 days back — a hint that deeper history may pay, which only a run trained deeper can
test.

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
| `eval6ref` | 4 positives-only anchors | **LIVE — unrun** |
| `evallong200` | long200 on the `eval6` protocol | done 2026-09-07, LSF 580743 — **0.7529 vs 0.780** |
| `evalampfix` / `evalctx` | the corrected-tree arms, and context at `DATA_STRIDE=2` | done 2026-09-07 |
| `evalth350` | the 4 generation-4 arms, 20 scenes, RTh | done 2026-09-14, LSF 816217–816220 — **tattn 0.7905** |
| `evalposw8` | the same 4 at `pos_w` 8 | done 2026-09-15, LSF 182833–182851 |
| `evalk10` | the 3 k10 arms, 17 scenes | done 2026-09-17, LSF 303197–303199 — **tattn ctx50 0.7748** |
| `evalk10plain` | the 4th k10 cell, `posw8_tattn_k10_plain_30e` | done 2026-09-17, LSF 330515 — **0.7907, the best k10 model** |
| `evallonggrid` | object scores for the 6 `longgrid`/`longsingle` arms | done 2026-09-17, LSF 316677–316682 — plain beats ctx50 in all 3 pairs; runs and predictions **deleted** the same day |

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
