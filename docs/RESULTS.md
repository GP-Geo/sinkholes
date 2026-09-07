# Results — the short version

*Last updated 2026-09-03. For the exhaustive per-run registry see
[MODEL_RUNS.md](MODEL_RUNS.md); for the evaluation outputs, both full-scene and
positives-only, see [PREDICTIONS.md](PREDICTIONS.md). This file is the readable
summary.*

> **Run directories were renamed on 2026-08-17** to
> `<partition>_<arch>[_<variant>]`, and the names in this file follow the new
> scheme. Anything without a `posw*` suffix is `pos_w` 4. See
> `outputs/README.md`.

We detect sinkhole subsidence in 11-day InSAR interferograms over the Dead Sea.
A model sees the current interferogram plus *k* previous ones and predicts which
pixels are subsiding. Counted on disk 2026-09-03: **67 run directories carry a
`results.csv`** (35 live, 32 under `outputs/archive_2019_2026_noisy_data/`) and
**56 evaluation job directories exist**, of which **35 sit on the current
generation-3 partitions**. All 19 runs of 2026-08-20 are now scored (finding ⑨)
and all 12 of their attention arms probed (finding ⑩).

---

## The one-paragraph summary

The big open problem — models flagging far too much ground — **is fixed.**
Adding "ring negatives" (training patches taken from a ring *around* each
sinkhole, not just on it) **lifted precision by 20% on geo and 60% on temporal**,
with no loss of recall on geo. This is the first change in the project that
clearly improved the metric that matters. It also came with an uncomfortable
lesson: **patch dice said this batch did nothing.** The single best model at
scene level was *last* on dice. Stop ranking models by dice.

---

## 1. The two metrics, and why only one counts

| | What it measures | Where it comes from |
|---|---|---|
| **`val/dice`** | pixel overlap on validation **patches** | `results.csv`, every epoch |
| **object P/R** | whether real sinkholes are **found**, on whole scenes | `predictions/*/olm_results*.json` |

**Object-level is the one that counts, and dice is now actively misleading.**
Not just uninformative — wrong. Details in finding ② below.

Three rules that are easy to get wrong:

- **±0.006 dice is noise.** Most "improvements" in this project are smaller.
- **Only compare within a partition.** Geo and temporal share no scenes.
- **Always quote the confidence threshold.** A model's precision doubles
  between 0.25 and 0.9. A number without a threshold is meaningless.

> Throughout this file, **F1 is computed from the mean precision and mean recall**
> across scenes, so it is a summary of the table, not a separately measured
> quantity.

---

## 2. What we know

### ① Ring negatives fix over-prediction — the headline result

Models used to be trained only on patches that contain subsidence (~2.5% of the
map), then applied to all of it. They found nearly everything and flagged far
too much. Training on a ring of nearby negative patches fixes this:

**Geo partition** (18 scenes, at confidence 0.5):

| Model | recall | precision | F1 |
|---|---|---|---|
| `geo_k5` baseline | 0.834 | 0.449 | 0.584 |
| `geo_k5` + ring negatives 1:1 | 0.862 | **0.545** | **0.668** |
| `geo_k5` + ring negatives 3:1 | 0.870 | 0.537 | 0.664 |

**Temporal partition** (20 scenes, at confidence 0.25):

| Model | recall | precision | F1 |
|---|---|---|---|
| `temporal_k5` baseline | 0.743 | 0.451 | 0.561 |
| `temporal_k5` + ring3 (near) | 0.664 | 0.591 | 0.626 |
| `temporal_k5` + ring10 (far) | 0.673 | **0.726** | **0.698** |

Precision improves by **+0.10 on geo and +0.28 on temporal**. On geo, recall
does not drop at all. This is the largest real gain the project has produced.

### ② Patch dice got this exactly backwards

Every ring-negative run was **flat or down on dice**, and the worst one on dice
is the best one on scenes:

| Model | dice (rank) | best object F1 (rank) |
|---|---|---|
| `geo_k5` baseline | 0.6558 (1st) | 0.667 (3rd) |
| `geo_k5` ring3 1:1 | 0.6541 (2nd) | 0.712 (2nd) |
| `geo_k5` ring3 **3:1** | 0.6386 (**last**) | **0.724 (1st)** |

The reason is known: ring negatives are added to the **train** split only, so
validation stays positives-only and the false positives being suppressed are
mostly outside the val set. Dice cannot see the thing that changed.

**Consequence: never promote or kill a model on dice again.** Dice is now only
a training-health signal (is it learning at all, is it diverging).

**We tried fixing the measurement, and it did not work.** `--add_val_negatives`
put a fixed 1:1 negative set into validation — same ring 1..3 the training
negatives use, drawn once from the validation interferograms with the run seed —
so that a false positive on background would cost dice. It was used by the
`valneg` reruns and by all 14 `clean22` runs.

**Removed 2026-08-20.** It made dice *look* different without making it
discriminate: an empty prediction on an empty mask scores dice **1.0**
(`losses.py:21`), so at 1:1 the mean is roughly `(1 + dice_on_positives)/2` —
which is exactly why `clean22` reads 0.72–0.80 against everything else's
0.63–0.66. What it bought was a second incomparable dice scale on top of the
first. Meanwhile `val/F1` / `val/P` / `val/R` are pooled over raw pixel counts
(`evaluate.py:300-306`) and were **already** negative-aware the whole time.

So the rule above stands, unchanged and now permanent: **dice is a
training-health signal only** — and if you want a negative-aware number from
`results.csv`, read `val/F1`. Promote/kill still belongs to `run_eval.sh`.
Validation is positives-only for every run from 2026-08-20 on; nothing under
`scripts/` can turn negatives on. The flag survives, deprecated, only so the
`attnfix` runs of 2026-08-19 stay resumable.

### ③ Temporal context helps much less than we thought: +0.012, not +0.046

The old "+0.046 dice from temporal context" was mostly **optimizer settings**,
not temporal context. The old single-frame baselines ran at batch 64 / lr 1e-5;
everything else ran batch 128 / lr 1e-6. With settings matched, the gap
collapses — and it now replicates at the smaller value on both geo groups:

| Partition | single-frame | ConvLSTM | gap |
|---|---|---|---|
| `geo_k5` (new, 2026-08-11) | 0.6434 | 0.6558 | **+0.0124** |
| `geo_k10` (2026-08-09) | 0.6513 | 0.6646 | **+0.0133** |
| `geo_k5` (old, confounded) | 0.6084 | 0.6539 | +0.0455 ✗ |

Temporal context is still a real gain, but it is about **2× the noise floor,
not 8×**. This is a downgrade of what used to be finding ①.

### ④ ~~Attention matches or slightly beats recurrence~~ — SUPERSEDED by ⑨

> **Overturned 2026-09-01.** This finding rested on a single temporal pairing
> whose evaluation directories have since been deleted. Nineteen runs on
> generation-3 ground now say the opposite: attention **never** beats ConvLSTM
> on either axis, and loses significantly at k10 on both. Attention is *not* the
> default architecture. The parameter-count argument below still stands; the
> accuracy claim does not. Kept for the record.

On temporal, under ring negatives, at confidence 0.25:

| Model | F1 | params |
|---|---|---|
| attention (`tattn` fuse0) | **0.642** | 32.1 M |
| attention + ConvLSTM hybrid | 0.639 | — |
| ConvLSTM | 0.626 | 43.1 M |

On dice these were a dead tie (0.6459 vs 0.6460). At object level attention is
nominally ahead. Given it is the cheaper model, **attention is now the sensible
default architecture** — though this margin is small and rests on one pairing.

### ⑤ `pos_w` 4 is the operating point

Unchanged and still well-established. Do not revisit.

### ⑥ Capacity is saturated

hidden 256 → 512 changed nothing; 1024 was *worse*. `lr 5e-6 + cosine` is
harmful. Both settled; do not revisit.

### ⑦ The first honest geo object-level numbers — and they demote the geo table below

*Added 2026-09-01 from the five `eval5` evaluations of 2026-08-19. Full tables:
[PREDICTIONS.md](PREDICTIONS.md) §3a.*

Everything in §3's geo table was measured on generation-2 partitions, whose geo
split divides by **frame** — so the 31.25–31.44° band sits in train *and* in the
hold-out. Generation 3 replaces that with a hard 31.4° latitude cut. Scored on
that clean ground, under the paper's RTh protocol at stride 4:

| geo, 32 scenes, RTh 0.5 | recall | precision | F1 |
|---|---|---|---|
| `geo_k5_convlstm_ring3` | 0.797 | **0.400** | **0.532** |
| `geo_k5_convlstm_ring3_3x` | 0.767 | 0.395 | 0.522 |
| `geo_k5_single_ring3` | 0.795 | 0.320 | 0.456 |

The same architecture reads **0.652 precision / 0.712 F1** in §3. The partition
and the threshold definition both changed, so this is not a clean subtraction —
but the direction is the one the leakage predicts. **§3's geo table is
superseded, not extended.** Two consequences follow immediately:

- **"3:1 negatives is the best model in the project" is withdrawn.** On clean
  ground 3:1 is *behind* 1:1 at every threshold, on recall and F1 alike.
- **Geo precision is the open problem, not geo recall.** All three models find
  ~0.8 of the objects and disagree only about how much else they flag.

### ⑧ Recurrence beats the single-frame U-Net on both axes — in opposite currencies

The same five evaluations carry the single-frame control on both axes, and this
is the cleanest measurement of temporal context the project has:

| Axis, RTh 0.5 | single-frame R / P | ConvLSTM R / P | what recurrence bought |
|---|---|---|---|
| geo (32 scenes) | 0.795 / 0.320 | 0.797 / **0.400** | **+0.080 precision, identical recall** |
| temporal (22 scenes) | 0.516 / 0.751 | **0.601** / 0.754 | **+0.085 recall, identical precision** |

Both margins hold across all four RTh thresholds and neither model gives
anything back on the other metric. **This is a larger and better-established
effect than finding ③ measured** — ③ read temporal context off patch dice at
+0.012, and patch dice is measuring the wrong thing again.

**Because on geo the patch curve says the opposite.** In the 2026-08-20
positives-only reruns the single-frame arm is *top* of its geo batch on both
`val/dice` and `val/F1` (`MODEL_RUNS.md`, "The 2026-08-20 batches"). Object
level reverses it. Finding ② in its sharpest form yet: what separates these two
models on geo is **precision on background**, and a positives-only patch curve
holds no background to be wrong about.

### ⑨ All 19 runs of 2026-08-20 are scored — recurrence wins, attention never does

Scored 2026-09-01 (`eval6`, 19 jobs). Two batches — `attnpos` (8 runs) and
`pre23` (11 runs) — on the generation-3 RTh protocol, 23 geo scenes and 20
temporal, audited comparable. Full tables in
[PREDICTIONS.md §3b](PREDICTIONS.md).

| | geo | temporal |
|---|---|---|
| best | `geo_k5_pre2023_convlstm_ring3` **0.648** | `temporal_k5_pre2023_convlstm_ring3` **0.780** |
| single-frame baseline | `geo_k5_pre2023_single_ring3` 0.612 | `temporal_k5_pre2023_single_ring3` 0.733 |
| worst | `geo_k10_pre2023_tattn_ring3` 0.584 | `temporal_k5_pre2023_single_ring3` 0.733 |

**Recurrence beats the single-frame U-Net on both axes**, +0.038 geo and +0.047
temporal, both significant on a paired per-scene bootstrap. It is a *recall*
gain: the single-frame model is the most conservative in the field, competitive
on precision and last on recall.

**Attention never beats recurrence.** Not on either axis, not at either depth.
At k10 it loses significantly on both (+0.066 geo, +0.030 temporal in ConvLSTM's
favour). The hybrid ties. This **supersedes finding ④**, which made attention
the default off a single temporal pairing whose evaluation directories have
since been deleted.

The dice curve remains useless here, exactly as finding ② says: geo k10 spans
0.0149 dice across the three architectures and changes leader on `val/F1`,
against a ±0.006 noise floor.

### ⑩ The attention selects on most arms — and it does not help

Probed 2026-09-02, all 12 attention checkpoints, on real data. **8 of 12
select**; 7 of 8 *pure*-attention arms do, one better than any 2026-08-19
reference. So `contrast` + `qk_norm` works, and that is a real positive result.

**3 of 4 hybrids fail the collapse alarm**, two at exactly 1.000 —
indistinguishable from the pre-fix control. That is the predicted failure mode:
when a ConvLSTM already integrates over time, nothing pushes the attention to
select.

**But selectivity buys nothing.** Measured selectivity correlates with object F1
at **r = −0.10**, and only **1 of 7** attention arms beats its matched ConvLSTM.
The second-most-selective arm is the worst geo model on record (0.595); the
fully dead hybrid scores 0.641, ahead of five arms that genuinely select. The
question "is the attention real?" is now answered yes — and it turns out not to
be the question that mattered.

Two practical notes: the probe **runs on a laptop in ~5 min per checkpoint**, no
cluster needed, despite `run_probe.sh` being written as a `bsub` job; and
`logit_scale` temperature predicts measured selectivity only weakly (r = −0.61),
reliably *only* at exactly 1.000. Rank on the probe, never the temperature.

### ⑪ The 2023–2026 training years buy nothing measurable

Six matched `pre23` vs `clean` pairs — same architecture, same depth, same
positives-only validation, same scored ground, differing only in whether the
training archive stops at 2022-12-31. **`pre23` is ahead in 4 of 6** and
significantly ahead in two, despite an archive ending three and a half years
before the test scenes. No pair favours `clean` significantly.

That is the question the `pre23` batch was built to answer, and the answer is
negative. Before scoping more GPU-hours on the larger archive it is worth
understanding why — the LiDAR mask ending 20240605 (finding ⑫) is one candidate.

### ⑫ The stale LiDAR mask is a real gap — but it is not inflating false positives

`assets/lidar_intf_mask.txt` ends 20240605. On the generation-3 **temporal** test
split **0 of 22** interferograms have a mapping of their own; every one falls
back to LiDAR2022. Geo is 28 of 35.

`eval7` (2026-09-02) re-scored the eleven `pre23` arms on the `pre2023` splits,
which are 100 % mapped. The LiDAR gating worked — zero fallback warnings — but
precision *fell* (−0.22 to −0.31 temporal), the opposite of the hypothesis, and
the comparison turns out not to measure the mask: the temporal splits share no
scenes and `min_positives=150` cut 35 scenes to 8 whose GT area is a third of
`eval6`'s. On geo, the scenes running on the stale mask are the **easier** ones
(mean F1 0.731 against 0.475 for the shared hard core).

If anything the mask *flatters* precision by suppressing predictions in ground
the 2022 LiDAR never saw — which would also cost recall, the pattern actually
observed. **Not settled.** The decisive test is one model on the same 20 scenes
with `--no-add_lidar_mask`, and it has not been run.

## 3. The best models we have

**Geo** — all six scored on the same 18 scenes, so these are directly
comparable, including across k5 and k10. **⚠️ Superseded for geo by finding ⑦:**
this table was measured on generation-2 partitions whose split put the
31.25–31.44° band in train *and* in the hold-out. Kept as the historical record;
do not quote a geo number from it.

| Model | dice | Best F1 | at conf. | R / P there |
|---|---|---|---|---|
| `geo_k5_convlstm_ring3_3x` | 0.6386 | **0.724** | 0.9 | 0.773 / 0.681 |
| `geo_k5_convlstm_ring3` | 0.6541 | **0.712** | 0.7 | 0.784 / 0.652 |
| `geo_k5_convlstm_base` | 0.6558 | 0.667 | 0.9 | 0.657 / 0.677 |
| `geo_k10_convlstm_ring3` | 0.6596 | 0.666 | 0.7 | 0.785 / 0.578 |
| `geo_k5_convlstm_posw8` | 0.6539 | 0.647 | 0.9 | 0.696 / 0.605 |
| `geo_k10_convlstm_posw8` | 0.6583 | 0.456 | 0.5 | 0.893 / 0.306 |

**Temporal** — all scored on the same 20 scenes:

| Model | dice | Best F1 | at conf. | R / P there |
|---|---|---|---|---|
| `temporal_k5_convlstm_ring10` | 0.6432 | **0.698** | 0.25 | 0.673 / 0.726 |
| `temporal_k5_tattn_ring3` | 0.6459 | 0.642 | 0.25 | 0.663 / 0.621 |
| `temporal_k5_tattn_hybrid_ring3` | 0.6468 | 0.639 | 0.25 | 0.667 / 0.613 |
| `temporal_k5_convlstm_ring3` | 0.6460 | 0.626 | 0.25 | 0.664 / 0.591 |
| `temporal_k5_convlstm_base` | 0.6391 | 0.612 | 0.5 | 0.609 / 0.616 |
| `temporal_k5_single_b64` | 0.5966 | 0.581 | 0.5 | 0.676 / 0.509 |

**The best threshold moved.** Older notes quoted everything at 0.25. With ring
negatives, geo models now peak at **0.7–0.9** and temporal at **0.25**. Quoting
a geo model at 0.25 now understates it by ~0.15 F1.

### The four 2026-08-11 `geo_k5` runs — all scored, two unreproducible

Finished 2026-08-11, all four scored under the generation-2 protocol (§2 of
[PREDICTIONS.md](PREDICTIONS.md)). Three stopped on early-stopping patience and
one ran the full 60 epochs — none was killed, so none is undertrained.

| Run | dice | obj F1 | directory |
|---|---|---|---|
| `geo_k5_tattn_hybrid_ring3` | 0.6487 | — | **deleted** |
| `geo_k5_single_base` | 0.6434 | 0.629 | present |
| `geo_k5_tattn_ring3` | 0.6433 | 0.683 | **deleted** |
| `geo_k5_single_ring3` | 0.6341 | 0.662 | present |

The two attention arms lost their evaluation directories in the 2026-08-20
prune, so their numbers cannot be re-scored, extended or checked — see the
warning box at the top of [PREDICTIONS.md](PREDICTIONS.md). This is why the
`eval6` attention rows (finding ⑨) matter: they are the live replacement for a
comparison that had become unverifiable.

---

## 4. Traps — read before quoting a number

**① The k10split model is invalid.** It once looked like the best model in the
project (dice 0.6694, object F1 0.743). It was trained with a sample-selection
bug *and* validated on its own era. Deleted 2026-08-11; post-mortem in
`outputs/2026-08-03/README.md`. **If you find an old number above 0.66, check
which run it came from.**

**② One scene destroys geo precision.** `20241127_20241208` scores precision
**0.06–0.11** for every geo model, against a ~0.5 average — it finds the
sinkholes (recall 0.96–0.98) but flags roughly ten times too much.
Dropping that single scene lifts geo mean precision @0.5 from 0.545 → 0.571;
dropping the worst three lifts it to 0.609. Worth looking at directly before
concluding geo precision is bad everywhere. It is not; it is bad *there*.

**③ Six scenes fail for every model.** `20250329_20250409`,
`20250512_20250523`, `20250830_20250910`, `20251002_20251013`,
`20251024_20251104`, `20250819_20250830` sit at ~0.45–0.60 recall regardless of
architecture or negatives. **Re-confirmed across all four temporal models.**
That points at the data, not the model.

**④ The two partitions fail in opposite directions.** Geo over-predicts
(high recall, low precision). Temporal under-predicts on future scenes.
"Low recall" is a temporal-partition symptom only.

**⑤ Recall collapses at high confidence — on temporal only.** Temporal models
lose half their recall by 0.9. Geo models with 3:1 negatives *hold* recall
(0.773 @0.9), which is exactly why they win there.

---

## 5. What to do next

> **Written 2026-08-12 and not revised since.** It predates findings ⑦–⑨, the
> generation-3 partitions, the positives-only protocol and the 19 runs of
> 2026-08-20. Items 1–4 below refer to models and partitions that in several
> cases no longer exist. Read it as the state of play in August, not as the
> current queue.

**The geo partition is the focus.** Training on the north and predicting the
south asks whether the model generalises to ground it has never seen, which is
what deployment requires.

### Do these first

1. **Submit the four `eval6ref` anchor jobs** (`submit_all.sh eval6ref`). All
   four have `best.pt` in `outputs/2026-08-19/` and fill the missing
   same-protocol controls — the single-frame geo baseline and geo k5/k10 +
   temporal k5 ConvLSTM — whose absence forces every `attnpos` row to be read
   against a `pre23` comparator across a training-archive boundary. Four
   evaluations, no retraining. The block's own note on `eval6ref_g5_single`
   flags an unresolved contradiction: it tops its batch on patch dice while its
   twin is 0.076 obj F1 behind, and "one of those two readings is wrong."
2. **Run the LiDAR control** — one model on the same 20 generation-3 temporal
   scenes with `--no-add_lidar_mask`. Finding ⑫ shows `eval7` could not isolate
   the mask; this can, and it is one job.
3. **Widen the geo threshold sweep above 0.5.** All nine geo `eval6` runs peak at
   the top of the sweep, so every geo F1 is a lower bound and the ranking could
   reorder. The `_pred.npy` are on disk — `scripts/eval/rescore.sh` is enough.
4. **Score `geo_k5` ring10 `geo_k5_convlstm_ring10`.** It is the one missing cell in the
   negatives grid. Far-field negatives were the *biggest* temporal win
   (F1 0.698) and nobody has tried them on geo. This is the highest
   expected-value single job available.
5. **Settle 1:1 vs 3:1 negatives on geo.** 3:1 has the best peak F1 (0.724) and
   holds recall at high confidence, but costs 12.5 h against 7.6 h. The earlier
   "3:1 is a settled negative" verdict came from dice and **is now withdrawn.**
6. **Clear the object-level backlog:** `geo_k10_convlstm_posw4`
   (highest dice in the project) and `geo_k10_single` have never
   been scored on scenes.
7. **Lower `min_positives` for any further `pre2023` scoring.** At 150 it cut
   the temporal split from 35 scenes to 8 (finding ⑫); 50 recovers 15, 25
   recovers 21. The threshold is tuned for the denser 2025–26 era.

### Then

- **Look at `20241127_20241208`** and the six failing temporal scenes as *data*,
  not as models. Six scenes at ~0.5 recall and one at 0.1 precision are worth
  more than another architecture arm.
- **Decide what the hybrid is for.** Three of four have collapsed to uniform
  attention (finding ⑩) and the one that did not is the only attention arm
  anywhere that edges its ConvLSTM. Either it needs a term that forces
  selection, or it should be retired in favour of the plain ConvLSTM it ties.
- **Extend `lidar_intf_mask.txt` past 20240605.** Every generation-3 temporal
  scene currently runs on the LiDAR2022 fallback.
- **Housekeeping:** the four 2026-08-11 runs are still loose at the top level of
  `outputs/` and still carry `last.pt` + `resume.pt` (~2 GB). Move them into
  `outputs/2026-08-11/` and prune, per `MODEL_RUNS.md`.

### Do not bother with

Bigger hidden sizes, cosine LR, `fuse4` skips. All settled negatives on
evidence that dice *could* legitimately see.

**Added 2026-09-03: the polygon area filter.** Swept on the generation-3 AOI
benchmark — 24 combinations of RTh and an area floor, on
`temporal_k5_pre2023_convlstm_ring3` — and **not one beats simply raising RTh**;
best F1 is at area floor 0, the published number. The object-level metric is
area-weighted, so a 50 px floor removes 47.5 % of false positives by *count* and
4.1 % of false-positive *area*. The old "half the false positives for 2.5 % of
recall" was a count statistic read as a precision claim. Full decomposition in
[EXPERIMENTS.md](EXPERIMENTS.md), under *Retired, and why* → correlation maps.

**No longer on this list:** 3:1 negatives — see item 5 above. Also removed:
"score the four 2026-08-11 runs" (all four are scored; two lost their
directories) and "probe the attention" (all 12 arms probed 2026-09-02).
