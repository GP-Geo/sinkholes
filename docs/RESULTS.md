# Results — the short version

*Last updated 2026-08-12. For the exhaustive per-run registry see
[MODEL_RUNS.md](MODEL_RUNS.md); for the evaluation outputs, both full-scene and
positives-only, see [PREDICTIONS.md](PREDICTIONS.md). This file is the readable
summary.*

> **Run directories were renamed on 2026-08-17** to
> `<partition>_<arch>[_<variant>]`, and the names in this file follow the new
> scheme. Anything without a `posw*` suffix is `pos_w` 4. See
> `outputs/README.md`.

We detect sinkhole subsidence in 11-day InSAR interferograms over the Dead Sea.
A model sees the current interferogram plus *k* previous ones and predicts which
pixels are subsiding. **38 models have been trained since 2026-08-03**, and
**17 of them have been scored on whole scenes.**

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

**The measurement, not the metric, was the problem.** `--add_val_negatives`
puts a fixed 1:1 negative set into validation — same ring 1..3 the training
negatives use, drawn once from the validation interferograms with the run seed,
identical across architectures and training-negative ratios — so a false
positive on background now costs dice. The five geo_k5 arms are being retrained
under it (`bash scripts/submit_all.sh valneg --submit`; settings in
`scripts/train/PRESETS.md`). Whether that closes the dice/object-F1 disagreement
above is the question those runs answer; until they land, the rule stands.

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

### ④ Attention matches or slightly beats recurrence, at 26% fewer parameters

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

---

## 3. The best models we have

**Geo** — all six scored on the same 18 scenes, so these are directly
comparable, including across k5 and k10:

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

### Trained yesterday, not yet scored on scenes

Four `geo_k5` runs finished 2026-08-11. Three stopped on early-stopping
patience and one ran the full 60 epochs — **none was killed**, so none is
undertrained. All kept `best.pt`, and **none has an object-level score yet** —
so by finding ②, none of these dice numbers should be read as a ranking:

| Run | dice | epochs |
|---|---|---|
| `tattn_geo_k5` hybrid ring3 `geo_k5_tattn_hybrid_ring3` | 0.6487 | best @22, stopped @42 |
| `unet_single_geo_k5` baseline `geo_k5_single_base` | 0.6434 | best @54, ran all 60 |
| `tattn_geo_k5` fuse0 ring3 `geo_k5_tattn_ring3` | 0.6433 | best @16, stopped @36 |
| `unet_single_geo_k5` ring3 `geo_k5_single_ring3` | 0.6341 | best @20, stopped @40 |

These are what finding ③ is built on, and they add the attention arm to geo.
**Scoring them is the top priority.**

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

**The geo partition is the focus.** Training on the north and predicting the
south asks whether the model generalises to ground it has never seen, which is
what deployment requires.

### Do these first

1. **Score the four 2026-08-11 `geo_k5` runs on scenes.** They have `best.pt`
   and no object-level number. Until they are scored we cannot say whether
   attention beats recurrence on geo, and finding ④ rests only on temporal.
2. **Score `geo_k5` ring10 `geo_k5_convlstm_ring10`.** It is the one missing cell in the
   negatives grid. Far-field negatives were the *biggest* temporal win
   (F1 0.698) and nobody has tried them on geo. This is the highest
   expected-value single job available.
3. **Settle 1:1 vs 3:1 negatives on geo.** 3:1 has the best peak F1 (0.724) and
   holds recall at high confidence, but costs 12.5 h against 7.6 h. The earlier
   "3:1 is a settled negative" verdict came from dice and **is now withdrawn.**
4. **Clear the object-level backlog:** `geo_k10_convlstm_posw4`
   (highest dice in the project) and `geo_k10_single` have never
   been scored on scenes.

### Then

- **Look at `20241127_20241208`** and the six failing temporal scenes as *data*,
  not as models. Six scenes at ~0.5 recall and one at 0.1 precision are worth
  more than another architecture arm.
- **Housekeeping:** the four 2026-08-11 runs are still loose at the top level of
  `outputs/` and still carry `last.pt` + `resume.pt` (~2 GB). Move them into
  `outputs/2026-08-11/` and prune, per `MODEL_RUNS.md`.

### Do not bother with

Bigger hidden sizes, cosine LR, `fuse4` skips. All settled negatives on
evidence that dice *could* legitimately see.

**No longer on this list:** 3:1 negatives — see item 3 above.
