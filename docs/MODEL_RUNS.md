# Trained model registry

Every training run under `outputs/`, its result, and what was kept on disk.

> **Looking for the short version?** [RESULTS.md](RESULTS.md) is the readable
> summary: what we know, what to ignore, and what to do next. Each dated folder
> under `outputs/` also carries a `README.md` for that batch. This file is the
> exhaustive record.

Written 2026-08-09, when the run directories were pruned; extended 2026-08-11
with the 13-run 2026-08-10 batch; extended 2026-08-12 with the four-run
2026-08-11 batch and **the object-level scores for all 17 evaluated models**;
extended 2026-08-19 with the 14-run clean benchmark (`clean22`), the first batch
on generation-3 partitions; extended 2026-09-01 with the 19 runs of the
2026-08-20 batches (`attnpos` and `pre23`); extended 2026-09-03 with **the
object-level scores for all 19** (`eval6`, run 2026-09-01), **the selectivity
probe for all 12 attention arms** (run 2026-09-02), and **the 11 `pre23` arms
re-scored on their own era** (`eval7`, run 2026-09-02). See
[PREDICTIONS.md](PREDICTIONS.md) for the numbers.
The point of this file is that **the metrics outlive the weights**:
`results.csv`, the training log and `curves.png` are kept for every run listed
here, so a deleted checkpoint costs a retrain (~2.5 h GPU) but never costs the
evidence.

## How to read the numbers

`val/dice` is patch-level pixel Dice at a fixed 0.5 threshold, averaged over
batches (`sinkholes/training/evaluate.py:252`). It is **not** the object-level
metric that `eval-outputs` reports, and it is **not** `val/F1` either: dice is
the **macro** per-image mean, F1 the **micro** pooled-pixel score
(`evaluate.py:190`). They diverge whenever the mix of easy and hard patches
changes — which is exactly what happened on 2026-08-18.

> ### ⚠️ Do not rank models by `val/dice`
>
> The 2026-08-11 evaluations settled this. Dice does not merely fail to
> discriminate between the leading models — on the negatives batch it ranks them
> **backwards**. `geo_k5_convlstm_ring3_3x` is **last on dice**
> in group G5 (0.6386, −0.0172 from anchor) and **first on scenes**
> (object F1 0.724). The mechanism is known and structural: ring negatives reach
> the **train** split only (`train.py:327/337/362`), so validation stays
> positives-only and the false positives being suppressed lie mostly outside the
> val set.
>
> Dice is now a **training-health signal only** — is the run learning, is it
> diverging, when did it peak. Every promote/kill decision belongs to the
> object-level column.
>
> **Attempted repair 2026-08-18, reverted 2026-08-20.** The 14-run clean batch
> passes `--add_val_negatives`, so its validation is half empty background
> patches and dice can see a false positive. That fixed the blindness above and
> opened a worse trap: it makes `val/dice` **incomparable across the boundary**,
> because half the new score is a negative-patch-cleanliness term the old number
> does not contain at all. **Never read a pre-2026-08-18 dice against a
> 2026-08-18/19 one.** See
> [The 2026-08-18 batch](#the-2026-08-18-batch--the-clean-benchmark-14-runs).
>
> The flag was removed on 2026-08-20 and validation is positives-only again, so
> that boundary is closed and no future run reopens it. If you want a
> negative-aware number out of `results.csv`, read **`val/F1`** — it is pooled
> over raw pixel counts (`evaluate.py:300-306`), so it always could see a false
> positive, on every batch, including the old ones.

### Object-level scores

The `obj F1` columns below are the **best F1 across the five scored confidence
thresholds**, with that threshold given, computed from the mean per-scene
precision and recall in `predictions/<run>/best/*/olm_results*.json`. All 6 geo
evaluations ran on one identical 18-scene list and all 11 temporal evaluations
on one identical 20-scene list, so **within each group every object-level number
here is directly comparable — including across `geo_k5` and `geo_k10`**, which
dice cannot do. `—` means the model has never been scored on scenes.

**Runs are only comparable inside a partition group.** The partitions define
different validation sets:

| Group | Partition | Split axis | Val intfs | Val samples |
|---|---|---|---|---|
| **T** | `temporal_k5` / `temporal_k10` | train 2019–22 → val 2023–24 → test 2025–26 | 30 (identical lists) | 9368 |
| **G5** | `geo_k5` | train 100% north → val/test 100% south | 25 | 5846 |
| **G10** | `geo_k10` | train 100% north → val/test 100% south | 17 | 4060 |
| **K10S** | `20_05_16h53_k10_compatible` | superseded | — | — |

`temporal_k5` and `temporal_k10` share an identical val list (Jaccard 1.00), so
all group-T runs are mutually comparable. Geo and temporal val sets are
**disjoint** (Jaccard 0.00) — a geo dice and a temporal dice cannot be ranked
against each other. `geo_k5` and `geo_k10` overlap only 0.62.

### Noise floor

Two same-config repeats and one seed change measure it:

- `convlstm_temporal_k5_h256` run A vs run B, identical config: **Δ 0.0044**
  (they diverge from epoch 1 — AMP/cuDNN nondeterminism, not seed).
- `convlstm_temporal_k10_h256` seed 42 vs seed 7: **Δ 0.0059**.

**Group T noise floor ≈ ±0.006 dice.** Group G10's val set is under half the
size (17 intfs / 4060 samples), so its floor is expected to be *wider*, and is
currently unmeasured — `scripts/train/12_..._seed7_...sh` exists to measure it.

## Group T — temporal partition (comparable, 10 runs)

`obj F1` is best-across-thresholds on the shared 20-scene list; `P`/`R` here are
the patch-level values at the best dice epoch.

| Run | dice | obj F1 | P | R | Verdict | Kept |
|---|---|---|---|---|---|---|
| `temporal_k10_convlstm_posw4` | 0.6437 | 0.601 @0.5 | 0.671 | 0.803 | top dice scorer; `pos_w` 8→4 bought precision | `best.pt` |
| `temporal_k10_convlstm_posw8` | 0.6427 | 0.581 @0.5 | 0.659 | 0.815 | group-T reference point | `best.pt` |
| `temporal_k5_convlstm_run2` | 0.6420 | 0.576 @0.5 | 0.674 | 0.793 | **k5 champion** — ties k10 on 34% more data | `best.pt` |
| `temporal_k10_convlstm_h512` | 0.6401 | — | 0.656 | 0.818 | capacity is saturated; no gain over h256 | metrics only |
| `temporal_k5_convlstm_run1` | 0.6376 | — | 0.667 | 0.791 | first of the duplicate pair; noise probe | metrics only |
| `temporal_k10_convlstm_seed7` | 0.6368 | — | 0.622 | 0.872 | seed probe; defines the ±0.006 floor | metrics only |
| `temporal_k10_convlstm_h1024` | 0.6296 | — | 0.618 | 0.864 | **worse**; early-stopped @36, undertrained | metrics only |
| `temporal_k10_stack` | 0.6284 | 0.562 @0.5 | 0.662 | 0.804 | channel-stacking control — loses to ConvLSTM by 0.0143 | `best.pt` |
| `temporal_k10_convlstm_cosine` | 0.6179 | — | 0.614 | 0.843 | **worst ConvLSTM**; early-stop @34, 5× the jitter | metrics only |
| `baseline_single_temporal_k5partition_b64_lr1e5_60e` | 0.5966 | 0.581 @0.5 | 0.544 | 0.897 | single-frame floor (lr/batch confounded) | `best.pt` |

**The top six are one statistical tie on dice** (0.6368–0.6437, span 0.0069 ≈
the noise floor), and the object-level column does not rank them in the same
order — the single-frame floor (obj F1 0.581) beats three ConvLSTMs above it.
Four effects survive:

1. **Temporal context: +0.012, not +0.046.** ⚠️ *Revised 2026-08-12.* The old
   +0.046 was mostly **lr/batch**: the single-frame baselines ran b64/lr 1e-5
   against b128/lr 1e-6 everywhere else. With settings matched it collapses, and
   the smaller value now replicates on both geo groups —
   `geo_k5` 0.6434 → 0.6558 (**+0.0124**) and
   `geo_k10` 0.6513 → 0.6646 (**+0.0133**), i.e. ~2× the noise floor, not 8×.
   The confounded geo_k5 pairing (0.6084 → 0.6539, +0.0455) should not be quoted.
2. **Recurrence beats channel-stacking: +0.0143** (~2.4× noise) on dice. The
   cleanest architecture result — lr, batch, partition and k all held fixed.
   Object level agrees in direction (0.576 vs 0.562) but by less.
3. **Capacity is saturated.** h256 → h512 flat; h1024 *worse* and undertrained.
4. **lr 5e-6 + cosine is harmful.** Worst ConvLSTM, and 5× the epoch-to-epoch
   jitter of any other run (std 0.011 vs ~0.002) — past the stability edge.

**k5 vs k10 is a dead tie** (0.6420 vs 0.6427, Δ 0.0007) but k10 costs a
quarter of the data: 166 vs 214 usable interferograms, 28,905 vs 38,705
training samples. k5 is the better default.

## Group G5 — geo_k5 partition (2 runs)

| Run | dice | obj F1 | P | R | Verdict | Kept |
|---|---|---|---|---|---|---|
| `geo_k5_convlstm_posw8` | 0.6539 | 0.647 @0.9 | 0.746 | 0.722 | most balanced patch P/R of any run | `best.pt` |
| `geo_k5_single_b64` | 0.6084 | — | 0.616 | 0.809 | single-frame floor; **lr/batch confounded — superseded** by `geo_k5_single_base` | `best.pt` |

## Group G10 — geo_k10 partition (1 run)

| Run | dice | obj F1 | P | R | Verdict | Kept |
|---|---|---|---|---|---|---|
| `geo_k10_convlstm_posw8` | 0.6583 | 0.456 @0.5 | 0.707 | 0.800 | best epoch 28/60, early-stop @48 | `best.pt` |

**This run has no sibling on its val set** for dice purposes. Its 0.6583 was
long the highest number in `outputs/`, and the object-level column shows why
that meant nothing: **obj F1 0.456 is the worst score of any evaluated geo
model** — it was scored at only 3 of the 5 thresholds and is a severe
over-predictor (precision 0.179 @0.25). Reference runs 11–14 under
`scripts/train/` exist to give group G10 a dice sibling.

## The 2026-08-10 batch — negatives and attention (13 runs)

`outputs/2026-08-10/`. Two experiments submitted together, all 13 pinned at
`pos_w` 4 so the arms can be read across partitions without `pos_w` shifting
underneath them. **All 13 completed** — every one stopped on early-stopping
patience (20 epochs without a `val/dice` improvement), none was killed by
walltime, so no run here is undertrained in the sense that `h1024` was.

They are kept as their own section rather than folded into groups T/G5/G10
above because the older runs sit at `pos_w` 2 and 8; only the runs below are
mutually comparable on that axis.

### Read this before ranking anything here

**`val/dice` cannot score this batch, and did not.** Ring negatives reach the
TRAIN split only (`train.py:327/337/362`); validation stays positives-only, so
the false positives being suppressed are mostly outside the val set.
`scripts/train/PRESETS.md:105` predicted "flat or slightly down", and that is
exactly what arrived: **all 13 land in 0.6386–0.6596**, a 0.021 span across two
experiments and three partitions, with every within-group delta at or under the
±0.006 noise floor. The attention batch was flagged the same way
(`PRESETS.md:138`).

**✅ Settled 2026-08-12 — 10 of these 13 are now scored on scenes**
(`submit_all.sh eval2` + `eval3`, completed 2026-08-11). Both questions have
answers, and the dice column turned out to be not merely uninformative but
**anti-correlated** with the outcome:

- **Do ring negatives fix scene-scale over-prediction? Yes, decisively.**
  Geo precision @0.5 rose 0.449 → 0.545 with recall *up* (0.834 → 0.862); best
  geo obj F1 rose 0.667 → 0.724. Temporal precision @0.25 rose 0.451 → 0.726
  using far-field (`ring10`) negatives. This is the largest real improvement the
  project has produced.
- **Does attention match recurrence at object level? Yes, and slightly ahead.**
  Under negatives on temporal, `tattn` fuse0 scores obj F1 0.642 against the
  ConvLSTM's 0.626, from a model 26% smaller.

Three runs of this batch remain unscored: `geo_k5 ring10 _96437`,
`temporal neg3x_ring3 _96433` and `tattn fuse2 _96447`. **`geo_k5 ring10` is the
highest-value job available** — far-field negatives were the biggest temporal
win and have never been tried on geo.

### Group T additions — temporal_k5 (val 9368)

Sorted by **obj F1** — the column that decides. Note how little it agrees with
the dice ordering it replaces.

| Run | obj F1 | dice | Δ dice | Verdict | Kept |
|---|---|---|---|---|---|
| `temporal_k5_convlstm_ring10` | **0.698** @0.25 | 0.6432 | +0.0041 | ⭐ **group-T winner.** Far-field negatives: precision 0.451 → 0.726. Was 6th on dice | `best.pt` |
| `temporal_k5_tattn_ring3` | 0.642 @0.25 | 0.6459 | −0.0001 | **attention beats recurrence** by +0.016 at 26% fewer params | `best.pt` |
| `temporal_k5_tattn_hybrid_ring3` | 0.639 @0.25 | 0.6468 | +0.0009 | hybrid; no gain over plain attention, and costlier | `best.pt` |
| `temporal_k5_convlstm_ring3` | 0.626 @0.25 | 0.6460 | +0.0069 | **the negatives reference**; twin of the `tattn` run above | `best.pt` |
| `temporal_k5_convlstm_base` | 0.612 @0.5 | 0.6391 | anchor | the `pos_w` 4 baseline both experiments are read against | `best.pt` |
| `temporal_k5_tattn_base` | 0.598 @0.5 | 0.6460 | +0.0069 | head-to-head, no negatives | `best.pt` |
| `temporal_k5_convlstm_ring3_3x` | — | 0.6449 | +0.0058 | 3:1 negatives; **unscored** — see the 3:1 note below | `best.pt` |
| `temporal_k5_tattn_fuse2_ring3` | — | 0.6399 | −0.0060 | temporally-fused skips, **wrong direction**; unscored | `best.pt` |

Δ dice is against `convlstm_temporal_k5_h256_posw4_60e` for the negatives arms,
and against the corresponding `fuse0` run for the attention variants.

**Far-field beat near-field on temporal, by a lot.** `ring10` scores 0.698
against `ring3`'s 0.626 — a +0.072 gap, larger than any architecture effect in
this project. On dice the two were 0.6432 and 0.6460, i.e. ranked the wrong way
round inside the noise floor.

**Attention ties recurrence.** 0.6459 vs 0.6460 under negatives, from a model
26% smaller (32.1 M vs 43.1 M params). `PRESETS.md:132` called landing inside
the noise floor a result in itself, and it did. The apparent +0.0069 in the
no-negatives pairing is not a win: the ConvLSTM anchor peaked at **epoch 11**
and stopped at 31, its best arriving before the LR schedule had decayed once —
it is the weakest anchor in the batch.

**`fuse4` should not be submitted.** `PRESETS.md:136` makes it conditional on
`fuse2` beating `fuse0`; `fuse2` lost 0.0060, the largest single delta in
group T and the only one pointing anywhere.

### Group G5 additions — geo_k5 (val 5846)

| Run | obj F1 | dice | Δ dice | Verdict | Kept |
|---|---|---|---|---|---|
| `geo_k5_convlstm_ring3_3x` | **0.724** @0.9 | 0.6386 | −0.0172 | ⭐ **best geo model in the project — and last on dice.** Holds recall 0.773 @0.9 where 1:1 drops to 0.615 | `best.pt` |
| `geo_k5_convlstm_ring3` | 0.712 @0.7 | 0.6541 | −0.0017 | near-field negatives; precision 0.449 → 0.545 @0.5 at no recall cost | `best.pt` |
| `geo_k5_convlstm_base` | 0.667 @0.9 | 0.6558 | anchor | the `pos_w` 4 geo baseline; **top of the group on dice, third on scenes** | `best.pt` |
| `geo_k5_convlstm_ring10` | — | 0.6461 | −0.0097 | far-field; **unscored, and the single highest-value job available** | `best.pt` |

### Group G10 addition — geo_k10 (val 4060)

| Run | obj F1 | dice | Δ dice | Verdict | Kept |
|---|---|---|---|---|---|
| `geo_k10_convlstm_ring3` | 0.666 @0.7 | 0.6596 | −0.0050 | ring negatives lift obj F1 0.456 → 0.666 over `geo_k10_convlstm_posw8` | `best.pt` |

G10's dice noise floor is still unmeasured and expected to be *wider* than
±0.006 (its val set is under half the size), so −0.0050 is not a decline — it is
inside a bar nobody has drawn yet.

**At object level G10 needs no such caveat**, because it was scored on the same
18-scene list as G5: ring negatives took `geo_k10` from obj F1 0.456 to 0.666,
and precision @0.25 from 0.179 to 0.312. That is the clearest single
demonstration of the over-prediction fix in the project. It still trails the
best `geo_k5` arms (0.712–0.724), which is consistent with the standing
recommendation to default to `geo_k5`.

### The one cost signal that is not noise — ⚠️ reversed 2026-08-12

The 3:1 arms cost 9.4 h (temporal) and 12.5 h (geo) against 4.8 h and 7.6 h for
1:1, and bought −0.017 dice on geo and nothing on temporal. The conclusion drawn
from that — *"1:1 near-field is the setting to carry forward"* — **was wrong,
and eval2 did contradict it.**

On scenes, `geo_k5` 3:1 is the **best model in the project** (obj F1 0.724 @0.9
against 1:1's 0.712 @0.7), and the gap widens with confidence: at 0.9 it holds
recall 0.773 where 1:1 collapses to 0.615. The −0.017 dice that made it look
like the batch's clearest failure was measuring the wrong thing.

**Current position on 3:1:** genuinely open, not settled either way. It wins on
peak F1 and high-confidence recall; it costs 1.6× the GPU time; and the margin
(+0.012) rests on a single pairing with no error bar at object level. The
temporal 3:1 arm `temporal_k5_convlstm_ring3_3x` is still unscored, which would give it a second data
point cheaply.

### Measured resources

Every run in this batch logged its own peak through
`torch.cuda.max_memory_allocated()`, so the VRAM ladder in `submit_all.sh` can
stop being a guess:

| Config | VRAM allocated | VRAM reserved | Host mem requested |
|---|---|---|---|
| k5 ConvLSTM (T=6), ±negatives | 23.4 GiB | **25.8 GiB** | 88 G |
| k5 attention (T=6) | 23.2 GiB | **25.6 GiB** | 88 G |
| k5 attention + ConvLSTM hybrid | 23.7 GiB | **26.2 GiB** | 88 G |
| k10 ConvLSTM (T=11) + negatives | 41.6 GiB | **44.4 GiB** | 144 G |

Size `gmem` off the RESERVED column. The eval jobs are set at 36 G on that
basis; the previous 48 G standard excluded 40 G A100s outright and left seven
jobs pending on 2026-08-10 for no reason.

## The 2026-08-11 batch — geo attention + de-confounded single-frame (4 runs)

Still at the **top level of `outputs/`**, not yet moved into a dated folder, and
still carrying `last.pt` + `resume.pt` (~2.1 GB across the four). All four are
`geo_k5` at `pos_w` 4, batch 128, lr 1e-6 — matched to the `geo_k5_convlstm_base` anchor, so
they are directly comparable to the group-G5 additions above on dice.

**All four finished cleanly.** Three stopped on early-stopping patience (20
epochs without improvement) and one ran the full 60; none was killed by
walltime, so none is undertrained. All four kept `best.pt`.

**None has an object-level score.** Per the warning at the top of this file, the
dice column below must not be used to rank them.

| Run | dice | obj F1 | Epochs | Purpose | Kept |
|---|---|---|---|---|---|
| `geo_k5_tattn_hybrid_ring3` | 0.6487 | — | best @22, stop @42 | hybrid attention on geo | `best.pt` |
| `geo_k5_single_base` | 0.6434 | — | best @54, ran 60 | ⭐ **de-confounded single-frame control** | `best.pt` |
| `geo_k5_tattn_ring3` | 0.6433 | — | best @16, stop @36 | plain attention on geo | `best.pt` |
| `geo_k5_single_ring3` | 0.6341 | — | best @20, stop @40 | single-frame **with** ring negatives | `best.pt` |

### What this batch already settled

`geo_k5_single_base` is the run that **corrects the
"+0.046 temporal context" claim** (finding 1 in group T). It is the single-frame
control at matched optimizer settings, which the 2026-08-05 baselines never
were:

| Pairing | single-frame | ConvLSTM | gap |
|---|---|---|---|
| `geo_k5`, matched b128/lr 1e-6 | 0.6434 | 0.6558 | **+0.0124** |
| `geo_k10`, matched (2026-08-09) | 0.6513 | 0.6646 | **+0.0133** |
| `geo_k5`, old b64/lr 1e-5 baseline | 0.6084 | 0.6539 | +0.0455 ✗ |

Two independent geo groups agree on ~+0.012–0.013, about **2× the noise floor
rather than 8×**. The old figure was mostly optimizer settings.

Note also that ring negatives *hurt* the single-frame model on dice
(0.6434 → 0.6341) — expected, and exactly the signal dice cannot interpret.

### What it cannot answer until it is scored

Whether attention beats recurrence **on geo**. That result currently rests
entirely on the temporal pairing (0.642 vs 0.626). The `geo_k5` attention arms
above are its replication, and until they are evaluated the architecture
recommendation has one leg. **Scoring these four is the top priority.**

## The 2026-08-18 batch — the clean benchmark (14 runs)

> **Renamed 2026-08-20.** Every run in this batch validated against negatives,
> so each surviving directory is now `<partition>_<arch>_<variant>_valneg` — the
> `clean_` prefix and the `_<date>_lsf_<id>` suffix are gone. The tables below
> keep the original `clean_*` names, because only 7 of these 14 survive as
> directories and half-rewriting the record would be worse than not rewriting
> it. `outputs/README.md` carries the old→new mapping.

`outputs/2026-08-18/*_valneg/`. The first runs on the **generation-3**
partitions (`assets/partition_*_clean.json`): all years 2019–2026, every split
restricted to the shoreline AOI lat 31.25–31.75 / lon 35.38–35.46, and the geo
axis cut at 31.4° so train (North, 31.40–31.75) and hold-out (South,
31.25–31.40) are disjoint ground for the first time. Submitted as
`submit_all.sh clean22`; all 14 finished by 2026-08-19 06:14.

Shared config across all fourteen — they differ in partition, architecture and
negative ratio, in nothing else: `pos_w` 4, hidden / attn dim 256, batch 128,
**lr 1e-5** (10× every previous batch), **100 epochs**, **patience 40**, seed 42,
ring 1–3 negatives in training at 1:1 (except the `_3x` arm), and
`--add_val_negatives`.

### ⚠️ `val/dice` from this batch cannot be compared to any earlier number — or any later one

Every previous batch validated on **positives only**, and so does every batch
after 2026-08-20, when `--add_val_negatives` was removed. This batch and the
five `attnfix` runs of 2026-08-19 are the only ones that ever used it, which
makes their dice column an island: readable against each other, against nothing
else. All 14 runs here pass `--add_val_negatives`, so validation is **50% empty
negative patches**:

```
validation set: 4093 positive + 4093 negative patches   (geo_k5; the log header of every run)
```

That changes what the number means, mechanically. `val/dice` is the **macro**
per-image mean (`evaluate.py:190` says so outright: "the pooled F1 (micro) is
*not* the returned Dice"), and in `losses.py:21`
`sets_sum = torch.where(sets_sum == 0, inter, sets_sum)` makes an empty patch
predicted empty score **exactly 1.0**. So on a 50/50 val set:

```
val/dice  =  0.5 · (dice on positives)  +  0.5 · (fraction of negatives with ZERO false-positive pixels)
```

Half the score is now a false-positive-cleanliness term. `val/F1`, `val/P` and
`val/R` are pooled pixel counts and get no such free half — which is why dice
rose and they did not.

**This is the fix the 2026-08-10 batch needed**, not a regression: that batch
landed all 13 arms inside ±0.006 precisely because positives-only validation
could not see the false positives ring negatives were suppressing. Dice can now
see them. It still must not be compared across the boundary.

### What actually changed, generation 2 → generation 3

Nine runs exist under the same name in both batches. Same architecture, same
negatives, same partition *axis* — only the partition generation, the validation
composition and the schedule differ.

| Run | dice | F1 | P | R |
|---|---|---|---|---|
| `geo_k10_convlstm_ring3` | 0.660→0.722 **+0.063** | 0.750→0.624 **−0.127** | −0.100 | −0.154 |
| `geo_k5_convlstm_ring3` | 0.654→0.726 **+0.072** | 0.745→0.634 **−0.111** | −0.083 | −0.143 |
| `geo_k5_convlstm_ring3_3x` | 0.639→0.728 **+0.090** | 0.732→0.627 **−0.105** | −0.051 | −0.155 |
| `geo_k5_single_ring3` | 0.634→0.691 **+0.057** | 0.728→0.635 **−0.093** | −0.044 | −0.157 |
| `geo_k5_tattn_hybrid_ring3` | 0.649→0.735 **+0.086** | 0.741→0.639 **−0.102** | −0.035 | −0.174 |
| `geo_k5_tattn_ring3` | 0.643→0.725 **+0.081** | 0.732→0.639 **−0.093** | −0.084 | −0.103 |
| `temporal_k5_convlstm_ring3` | 0.646→0.792 **+0.146** | 0.729→0.731 **+0.002** | +0.007 | −0.005 |
| `temporal_k5_tattn_hybrid_ring3` | 0.647→0.794 **+0.148** | 0.735→0.743 **+0.007** | +0.025 | −0.017 |
| `temporal_k5_tattn_ring3` | 0.646→0.794 **+0.148** | 0.739→0.735 **−0.004** | +0.029 | −0.043 |

**Recall is the metric to read here.** Adding empty negatives cannot change it —
they contain no positive ground truth, so they contribute no TP and no FN. It is
confounded only by the partition change itself, and it splits cleanly by axis:

- **temporal −0.005 to −0.043.** Flat, and F1 flat with it. Nothing real moved.
- **geo −0.10 to −0.17.** Every pair, large, same direction.

### 🔴 The generation-2 geo numbers were inflated by train/hold-out overlap

That geo recall drop is not a regression, it is the **leakage coming out**.
`assets/PARTITIONS.md` records why generation 2 was superseded:

> The geo split divides by **frame**, so the 31.25–31.44° band (~21 km) imaged
> by both frames sits in train *and* in the hold-out.

Generation 3 replaces the frame split with a hard 31.4° latitude cut, so the
geo hold-out is now ground the model has never seen. That costs **~0.10–0.15
recall**, and the temporal axis — which never had a frame-based split — is the
control that confirms it: its honest numbers barely moved.

**Every geo object-level score in `docs/PREDICTIONS.md` was measured partly on
training ground.** The `eval5` batch replaces that table rather than extending
it. `geo_k5 ring3_3x`'s obj F1 0.724, currently "the best model in the project",
is among the numbers that need re-earning on clean ground.

### The batch, ranked (comparable *within* this batch only)

At the best-dice epoch. Epoch-to-epoch noise over the last 10 epochs is
**±0.004 dice / ±0.003 F1** (median across the 14), so treat gaps under ~0.01 as
ties. No run-to-run seed repeat exists on generation 3 yet, so the true floor is
wider than that.

**Geo** — the target axis:

| Run | dice | F1 | P | R | best/last ep | obj F1 |
|---|---|---|---|---|---|---|
| `clean_geo_k10_tattn_hybrid_ring3` | **0.7378** | 0.6437 | 0.6558 | 0.6320 | 25 / 65 | — |
| `clean_geo_k5_tattn_hybrid_ring3` | **0.7350** | 0.6395 | 0.6607 | 0.6196 | 50 / 90 | — |
| `clean_geo_k5_convlstm_ring3_3x` | 0.7282 | 0.6273 | 0.6651 | 0.5935 | 73 / 100 | — |
| `clean_geo_k10_tattn_ring3` | 0.7271 | 0.6426 | 0.5976 | 0.6950 | 46 / 86 | — |
| `clean_geo_k5_convlstm_ring3` | 0.7262 | 0.6344 | 0.6264 | 0.6426 | 57 / 97 | — |
| `clean_geo_k5_tattn_ring3` | 0.7246 | 0.6395 | 0.6231 | 0.6569 | 19 / 59 | — |
| `clean_geo_k10_convlstm_ring3` | 0.7222 | 0.6237 | 0.6409 | 0.6073 | **9** / 49 | — |
| `clean_geo_k5_single_ring3` | 0.6906 | 0.6348 | 0.6177 | 0.6529 | 33 / 73 | — |

**Temporal:**

| Run | dice | F1 | P | R | best/last ep | obj F1 |
|---|---|---|---|---|---|---|
| `clean_temporal_k10_convlstm_ring3` | 0.8038 | 0.7390 | 0.7464 | 0.7317 | 59 / 99 | — |
| `clean_temporal_k10_tattn_ring3` | 0.7978 | 0.7330 | 0.7298 | 0.7362 | 74 / 100 | — |
| `clean_temporal_k5_tattn_hybrid_ring3` | 0.7943 | 0.7429 | 0.6964 | 0.7960 | 68 / 100 | — |
| `clean_temporal_k5_tattn_ring3` | 0.7942 | 0.7349 | 0.7194 | 0.7510 | 67 / 100 | — |
| `clean_temporal_k5_convlstm_ring3` | 0.7923 | 0.7314 | 0.6783 | 0.7935 | 95 / 100 | — |
| `clean_temporal_k5_single_ring3` | 0.7777 | 0.7274 | 0.7190 | 0.7359 | 79 / 100 | — |

Geo and temporal dice are **not** rankable against each other: different splits,
different AOI (geo scores a 31.25–31.40 South band, temporal the full
31.25–31.75 box on both frames), different scene lists.

### The one thing dice and F1 disagree about, and it is informative

`clean_geo_k5_single_ring3` is **last on dice by 0.035** but **mid-pack on F1,
P and R** — 0.6348 / 0.6177 / 0.6529 against the ConvLSTM anchor's 0.6344 /
0.6264 / 0.6426. Same pixel-level quality; much worse dice.

Both facts are real and they measure different things. Dice-on-negatives is
near-binary — one stray pixel drops a patch from 1.0 to 0.0 — while pooled
precision weighs by area. So the single-frame model leaks a *few* pixels into
*more* negative patches, and the temporal models' advantage is concentrated in
**completely clearing background patches**, not in delineating sinkholes better.

That is exactly what ring negatives were built to buy, and exactly what
positives-only validation could not see. Whether it survives at scene scale is
`eval_clean_g5_single_control`, the highest-value job in `eval5`.

### Convergence — half this batch did not finish learning

`PATIENCE=40` at `LR=1e-5`, against the previous batches' 20 at 1e-6.

- **Six ran out of epochs** rather than early-stopping: all five temporal k5/k10
  arms and `geo_k5_convlstm_ring3_3x` hit 100/100. Their reported peak is a
  floor, not a plateau — `temporal_k5_convlstm_ring3` peaked at epoch **95 of
  100**.
- **`clean_geo_k10_convlstm_ring3` peaked at epoch 9 of 49** and then degraded
  for 40 epochs. The fastest overfit in the project; it is the weakest geo run
  in the batch and is deliberately excluded from `eval5`.

### Object-level status — scored 2026-09-01 (`eval6`)

**The `obj F1` cells above are stale `—` placeholders for the clean22 table
only.** All nineteen runs of the 2026-08-20 batches were scored 2026-09-01 under
the RTh protocol described below; the numbers are in
[PREDICTIONS.md §4](PREDICTIONS.md). Per the standing rule at the top of this
file, the dice column still must not be used to promote or kill any of these —
and `eval6` is now the object-level evidence that supersedes it.

`submit_all.sh eval5` scores **8 of the 14**, under the **benchmark paper's RTh
protocol** (implemented 2026-08-19, `PLAN_CLEAN_BENCHMARK.md` §7(b) items 1 and
4). Geo on `geo_k10`'s 35-scene test list, temporal on `temporal_k10`'s 22-scene
list — verified 2026-08-19 that the k10 test lists are strict subsets of the k5
ones on both axes, so k5 and k10 models share ground. The six skipped runs and
the reason for each are listed in the `eval5` block of `scripts/submit_all.sh`;
all six keep `best.pt`.

The protocol, in one line each:

- **stride 4** — quarter-patch step, 16 tiles per interior pixel.
- **Confidence Factor** — each tile binarised at 0.5, the pixel value is the
  *fraction of overlapping tiles voting positive*, on {0, 1/16, …, 1}.
- **RTh 0.125 / 0.25 / 0.375 / 0.5** — "at least 2 / 4 / 6 / 8 of 16 agreed".
- **Both Intersection Tolerances** — ITh 0.7 / b 5 and the softer ITh 0.5 / b 10.
- **GT-area-weighted aggregate** reported alongside the unweighted scene mean.
- **Recall is primary**, per the paper: full-range reconstruction scores a lot
  of unannotated ground, so precision partly measures the digitisation.

⚠️ **An RTh is not a `recon_th`.** One counts tiles, the other cuts a mean
probability. No number produced by `eval5` belongs in the same column as
anything in `docs/PREDICTIONS.md`.

### Retention

~~All 14 keep `best.pt`, `last.pt`, `resume.pt` and `interrupted.pt` as of
2026-08-19.~~ The standing rule — **keep `best.pt` for every run until it has an
object-level score** — covers all of them, including the six `eval5` skips. The
`last`/`resume`/`interrupted` checkpoints are the normal post-batch pruning
target once the runs that ran out of epochs are decided (six of them could
legitimately be *extended* from `resume.pt` rather than retrained, which is an
argument for holding `resume.pt` longer than usual here).

**Superseded 2026-08-20: the seven runs of the 2026-08-18 batch hold no
checkpoints at all.** Every one of them validated against negatives, which is
the defect that the whole positives-only protocol exists to remove, so the
weights were deleted rather than pruned — `best.pt` included, 5.5 GB. This
overrides the standing rule above for exactly two runs,
`clean_geo_k10_convlstm_ring3` and `clean_temporal_k10_convlstm_ring3`, which
went without an object-level score. The reasoning: an object score of a
checkpoint chosen on an inflated curve measures the wrong checkpoint, so
scoring them first would not have answered anything the positives-only rerun
does not answer better. **Cost if that is wrong: ~2.5 h GPU each**, and
`temporal_k10_convlstm_ring3` has no positives-only twin yet — the `attnpos`
batch is where it was meant to come from. The other five keep their
`predictions/` object scores from before the deletion.

## The 2026-08-20 batches — positives-only reruns and the pre-2023 archive (19 runs)

*Recorded 2026-09-01 from `results.csv` and `logs/reporter.log`. **No run in
either batch has an object-level score**, so by the rule at the top of this file
none of the tables below is a ranking. They are the training record.*

Two batches were submitted on 2026-08-20 and all 19 runs finished on GPU:

| Batch | Kind | Partitions | Runs | Where |
|---|---|---|---|---|
| positives-only attention reruns | `attnpos` | `*_clean.json` (2019–2026) | 8 | top level of `outputs/` |
| the 2019–2022 archive | `pre23` | `*_pre2023.json` (2019–2022) | 11 | top level of `outputs/` |

Both validate on **positives only** — the protocol adopted 2026-08-20 — so they
are readable against the four `valpos` runs already in `outputs/2026-08-19/` and
**not** against any `valneg` number. Every one of the 19 kept `best.pt`, plus
`last.pt` and `resume.pt`; seven also hold an `interrupted.pt` from a requeue.

### The comparability rule this batch adds: read `valN`, not just the partition

Within one partition the **single-frame and temporal arms do not share a
validation set**, and the gap is large on geo. `_load_temporal`
(`dataprep/dataset.py:341`) clips every grid to the chain's common extent
(`ny = min(...)` over the T frames) and drops coordinates missing from an
earlier grid; `_load_single_ring` (`:428`) reads the current frame's grid only
and keeps them. The temporal val set is therefore a **subset** of the
single-frame one:

| Group | single-frame `valN` | temporal `valN` | ratio |
|---|---:|---:|---:|
| `geo_k5` clean | 6,526 | 4,093 | 1.59x |
| `geo_k5` pre23 | 4,270 | 2,576 | 1.66x |
| `temporal_k5` pre23 | 6,417 | 6,072 | 1.06x |

On temporal this is a 6% difference and can be ignored. **On geo the
single-frame arm is scored on 60% more patches than the model it is the control
for**, so a single-vs-temporal dice gap under ~0.02 on geo says nothing. `valN`
is in every table below for that reason.

### Batch 1 — `attnpos`, positives-only, generation-3 partitions

The five `attnfix` arms of 2026-08-19 rerun without validation negatives, plus
four temporal arms that had no positives-only twin. Ring-3 1:1 negatives,
`pos_w` 4, batch 128, lr 1e-5, 100 epochs, patience 40, seed 42 throughout. The
four `valpos` runs already under `outputs/2026-08-19/` are folded in — same
protocol, same partitions, same hyper-parameters.

**Geo** — the target axis. Only rows sharing a `valN` are strictly comparable:

| Run | dice | val/F1 | val/P | val/R | best/last ep | valN | obj F1 |
|---|---|---|---|---|---|---:|---|
| `geo_k10_tattn_hybrid_ring3_valpos` | **0.6025** | 0.6688 | 0.6406 | 0.6997 | 37 / 77 | 2,649 | — |
| `2026-08-19/geo_k10_convlstm_ring3_valpos` | 0.5921 | **0.6744** | 0.6485 | 0.7026 | 43 / 83 | 2,649 | — |
| `geo_k10_tattn_ring3_valpos` | 0.5876 | 0.6625 | 0.6421 | 0.6842 | 41 / 81 | 2,649 | — |
| `2026-08-19/geo_k5_single_ring3_valpos` | 0.5869 | **0.6748** | 0.6533 | 0.6978 | 59 / 93 (!) | 6,526 | — |
| `geo_k5_tattn_ring3_valpos` | 0.5861 | 0.6544 | 0.5911 | 0.7329 | 53 / 93 | 4,093 | — |
| `2026-08-19/geo_k5_convlstm_ring3_valpos` | 0.5847 | 0.6628 | 0.6191 | 0.7133 | 48 / 88 | 4,093 | — |

**Temporal:**

| Run | dice | val/F1 | val/P | val/R | best/last ep | valN | obj F1 |
|---|---|---|---|---|---|---:|---|
| `2026-08-19/temporal_k5_convlstm_ring3_valpos` | **0.6526** | **0.7453** | 0.6861 | 0.8158 | 66 / 100 | 5,917 | — |
| `temporal_k10_convlstm_ring3_valpos` | 0.6510 | 0.7451 | 0.6936 | 0.8048 | 75 / 100 | 4,735 | — |
| `temporal_k5_tattn_ring3_valpos` | 0.6493 | 0.7420 | 0.6862 | 0.8077 | 82 / 100 | 5,917 | — |
| `temporal_k5_tattn_hybrid_ring3_valpos` | 0.6484 | 0.7385 | 0.6844 | 0.8018 | 83 / 100 | 5,917 | — |
| `temporal_k10_tattn_hybrid_ring3_valpos` | 0.6465 | 0.7424 | 0.7005 | 0.7896 | 77 / 100 | 4,735 | — |
| `temporal_k10_tattn_ring3_valpos` | 0.6420 | 0.7441 | 0.7271 | 0.7619 | 77 / 100 | 4,735 | — |

(!) `geo_k5_single_ring3_valpos` has no completion line — killed at epoch 93 of
100, most likely `TERM_RUNLIMIT` (`outputs/README.md`). Its best epoch is 59 and
its last seven epochs ran at lr 1.95e-08 with `val/dice` flat in the fourth
decimal, so the missing epochs cannot move the number; nothing that parses
`Best val/dice` will see it.

**The dice deflation predicted for this batch happened, at the predicted size.**
`submit_all.sh` expected 0.63–0.66 where the `valneg` twins read 0.72–0.80.
Measured, per matched pair:

| Run | `valneg` dice | `valpos` dice | delta |
|---|---|---|---|
| `geo_k10_tattn_fixed_ring3` | 0.7331 | 0.5876 | −0.1455 |
| `geo_k10_tattn_hybrid_fixed_ring3` | 0.7442 | 0.6025 | −0.1417 |
| `geo_k5_tattn_fixed_ring3` | 0.7233 | 0.5861 | −0.1372 |
| `temporal_k5_tattn_fixed_ring3` | 0.7957 | 0.6493 | −0.1464 |

`val/F1` — pooled over raw pixel counts and negative-aware on both protocols —
moves **+0.0032, +0.0061, +0.0119, +0.0176** across the same four pairs: an
order of magnitude smaller than the dice move, and in the *opposite* direction.
**That is the check that the deflation is the padding coming off and not a
regression.** (Each figure is read at its own run's best-dice epoch, and the
protocol changes which epoch that is, so these deltas carry a selection
difference as well as a protocol one — they bound the effect, they do not
isolate it.)

**Nothing in this batch separates on the patch curve.** Geo k10 spans 0.0149
dice across three architectures and **changes leader** on `val/F1`: ConvLSTM
first at 0.6744, the dice leader (hybrid) second at 0.6688, attention last on
both. Temporal spans 0.0106 dice and 0.0068
`val/F1` across six runs. Against a ±0.006 dice noise floor — and with **no
seed repeat on generation 3**, so the true floor is wider — attention, the
hybrid and ConvLSTM are a three-way tie on both axes.

**The attention fix is not isolated here, by design.** `attnpos` carries no
`prefix` control; the fixed/prefix pair was run once, under `valneg`
(0.7331 vs 0.7271 — a 0.006 gap, at the noise floor). No positives-only
comparison of working against dead attention exists, and the `valneg`
checkpoints that would license one were deleted with the 2026-08-18 batch.

### Does the attention actually select? Measured 2026-09-02 — 8 of 12 do

*Probed 2026-09-02. All twelve attention checkpoints of both 2026-08-20 batches,
each against the partition family it **trained** on, `--control_lookback` set to
that run's own `k_prevs`, `--lookback 40` to match the 2026-08-19 references.
The headline number is the **control** ratio `effective_frames /
effective_frames_if_uniform` at the run's own depth — that is the quantity the
six reference points quote, not the probe-depth ratio. 1.000 is exactly uniform.
0.9 is `run_probe.sh`'s own collapse alarm.*

**The probe does not need the cluster.** It runs on a laptop against the patch
tree over the mount, roughly 5 min per checkpoint, I/O-bound rather than
compute-bound; the whole sweep of twelve took about an hour. `run_probe.sh` is
written as a `bsub` job, which made this look like queued work for a year.

| Checkpoint | control | k40 | temp | obj F1 | reading |
|---|---|---|---|---|---|
| `clean_temporal_k10_tattn_hybrid` | **0.708** | 0.711 | 1.093 | 0.772 | selecting |
| `geo_k5_tattn` | 0.735 | 0.719 | 1.431 | 0.595 | selecting |
| `temporal_k10_pre2023_tattn` | 0.741 | 0.707 | 1.860 | 0.748 | selecting |
| `geo_k10_tattn` | 0.767 | 0.743 | 1.208 | 0.626 | selecting |
| `temporal_k10_tattn` | 0.792 | 0.798 | 1.562 | 0.751 | selecting |
| `temporal_k5_pre2023_tattn` | 0.822 | 0.822 | 1.151 | 0.768 | selecting |
| `geo_k5_pre2023_tattn` | 0.872 | 0.858 | 1.140 | 0.638 | selecting |
| `geo_k10_pre2023_tattn` | 0.880 | 0.880 | 1.122 | 0.584 | marginal |
| `clean_geo_k10_tattn_hybrid` | 0.927 | 0.936 | 1.000 | 0.630 | **fails alarm** |
| `temporal_k5_tattn` | 0.948 | 0.944 | 1.192 | 0.759 | **fails alarm** |
| `geo_k10_pre2023_tattn_hybrid` | **1.000** | 1.000 | 1.000 | 0.641 | **dead** |
| `clean_temporal_k5_tattn_hybrid` | **1.000** | 1.000 | 1.000 | 0.747 | **dead** |

The 2026-08-19 `valneg` reference points, same metric: `geo_k5_tattn_fixed`
0.757, `geo_k10_tattn_fixed` 0.788, `geo_k10_tattn_hybrid_fixed` 0.893,
`temporal_k5_tattn_fixed` 0.928, and the two dead controls
(`geo_k10_tattn_prefix`, clean22 `geo_k10_tattn`) at 1.000.

**① The fix took, on the pure-attention arms.** Seven of eight pure arms select,
and `geo_k5_tattn` at 0.735 is more selective than any 2026-08-19
reference. `contrast` + `qk_norm` works. Only `temporal_k5_tattn`
(0.948) fails among them. This could not have been established from the weights
or from an architecture flag — it needed the probe.

**② The hybrids are where it collapses — three of four fail.**
`geo_k10_pre2023_tattn_hybrid` and `clean_temporal_k5_tattn_hybrid` sit at
**exactly 1.000**, indistinguishable from the `prefix` control and from the
clean22 collapse this fix exists for; `clean_geo_k10_tattn_hybrid` is 0.927.
All three carry `logit_scale` temperature 1.000. This is the predicted failure
mode confirmed: when a ConvLSTM already integrates over time, nothing pushes the
attention to select, and the softmax relaxes flat. The exception is real though —
`clean_temporal_k10_tattn_hybrid` at 0.708 is the **most** selective arm of all
twelve, so this is a strong tendency, not a law.

**③ Selectivity does not buy accuracy. This is the finding that matters.**
Across the twelve, measured selectivity correlates with object F1 at
**r = −0.10** — essentially zero — and the within-axis correlations flip sign
(+0.49 geo, −0.38 temporal), which is what noise looks like. Against the matched
ConvLSTM in the same cell, **only 1 of 7 attention arms comes out ahead**
(`clean_temporal_k10_hybrid`, +0.011, not significant). The two ends make the
point unaided: the most selective arm is the one that edges its ConvLSTM, the
second most selective (`clean_geo_k5`, 0.735) is the **worst geo model on
record** at 0.595, and the fully dead `geo_k10_pre2023_tattn_hybrid` scores 0.641 —
better than five arms that genuinely select.

> ### ⚠️ Temperature is a weak predictor of selectivity — the 2026-09-01 warning stands, now quantified
>
> Over all twelve, `logit_scale` temperature correlates with measured
> selectivity at **r = −0.61**. It is reliable at exactly one place: all three
> arms at temperature 1.000 fail the alarm. Everywhere else it misleads —
> `clean_temporal_k10_hybrid` at temperature 1.093 is the most selective arm
> measured, while `temporal_k5_tattn` at a *hotter* 1.192 is nearly
> uniform. Reading collapse off a checkpoint would have mislabelled both.
> **Rank on the probe, never on the temperature.**

### Batch 2 — `pre23`, the 2019–2022 archive

Eleven arms on `assets/partition_{geo,temporal}_k{5,10}_pre2023.json` —
section 5's 2019–2022 design rebuilt with the current generator: same 31.4°
cut, same AOI, same seed, same val share, `--years 2019 2022`
(`PLAN_CLEAN_BENCHMARK.md` section 0a). Same hyper-parameters as batch 1.

**Geo:**

| Run | dice | val/F1 | val/P | val/R | best/last ep | valN | obj F1 |
|---|---|---|---|---|---|---:|---|
| `geo_k5_pre2023_single_ring3` | **0.5990** | **0.6837** | 0.6230 | 0.7575 | 56 / 96 | 4,270 | — |
| `geo_k10_pre2023_tattn_hybrid_ring3` | 0.5904 | 0.6641 | 0.6334 | 0.6978 | 62 / 100 | 1,462 | — |
| `geo_k10_pre2023_convlstm_ring3` | 0.5886 | 0.6593 | 0.6259 | 0.6966 | 45 / 73 (!) | 1,462 | — |
| `geo_k10_pre2023_tattn_ring3` | 0.5871 | 0.6596 | 0.5800 | 0.7647 | 29 / 69 | 1,462 | — |
| `geo_k5_pre2023_tattn_ring3` | 0.5861 | 0.6342 | 0.5617 | 0.7281 | 62 / 100 | 2,576 | — |
| `geo_k5_pre2023_convlstm_ring3` | 0.5853 | 0.6401 | 0.5861 | 0.7052 | 57 / 97 | 2,576 | — |

**Temporal:**

| Run | dice | val/F1 | val/P | val/R | best/last ep | valN | obj F1 |
|---|---|---|---|---|---|---:|---|
| `temporal_k5_pre2023_convlstm_ring3` | **0.6626** | 0.7356 | 0.6404 | 0.8641 | 51 / 91 | 6,072 | — |
| `temporal_k5_pre2023_tattn_ring3` | 0.6625 | **0.7372** | 0.6519 | 0.8482 | 44 / 84 | 6,072 | — |
| `temporal_k10_pre2023_convlstm_ring3` | 0.6448 | 0.7180 | 0.6735 | 0.7689 | 54 / 94 | 2,632 | — |
| `temporal_k5_pre2023_single_ring3` | 0.6421 | 0.7259 | 0.6564 | 0.8118 | 47 / 87 | 6,417 | — |
| `temporal_k10_pre2023_tattn_ring3` | 0.6381 | 0.7055 | 0.6162 | 0.8250 | 52 / 92 | 2,632 | — |

(!) `geo_k10_pre2023_convlstm_ring3` took SIGINT (WEXAC preemption) at epoch 73 of
100 and was never requeued. Its best epoch is 45 and the 28 epochs after it ran
at lr <= 1.95e-08 without improving, so it is converged in substance; it is the
one run in the batch with no `Training complete` line.

**`pre23` and `clean22` dice cannot be read against each other.** The archives
differ, so the validation sets differ — `geo_k10` is 1,462 patches here against
2,649 on the clean partitions, `temporal_k10` 2,632 against 4,735. The twin
comparison the batch was built for is a **precision** claim about background in
the newer years (section 0a: "how much of the false-positive rate is the newer
years"), and precision on background is exactly what a positives-only patch
curve cannot see. **That question stays open until both batches are scored on
scenes.**

### The single-frame U-Net, on the patch curve

Three of the 19 runs are single-frame controls, and on geo the control is at the
top of its batch on both dice and `val/F1`:

| Group | single-frame | best temporal arm, same group | delta dice | delta val/F1 |
|---|---|---|---|---|
| `geo_k5` clean | 0.5869 / 0.6748 | tattn 0.5861 / 0.6544 | +0.0008 | +0.0204 |
| `geo_k5` pre23 | 0.5990 / 0.6837 | tattn 0.5861 / 0.6342 | +0.0129 | +0.0495 |
| `temporal_k5` pre23 | 0.6421 / 0.7259 | convlstm 0.6626 / 0.7356 | −0.0205 | −0.0097 |

Read with the `valN` rule above: the two geo rows compare 4,270–6,526 patches
against 2,576–4,093, so the geo margins are **not** a like-for-like measurement
and cannot be quoted as "one frame beats recurrence". The temporal row *is*
close to like-for-like (6,417 vs 6,072, a 6% difference) and reproduces
finding ③ of `RESULTS.md` at its stated size: **+0.021 dice for temporal
context, about 3x the noise floor.**

The object-level answer to the same question already exists on generation-3
ground for a different pair of checkpoints, and it points the other way on geo —
see `PREDICTIONS.md`, "The clean benchmark".

### Bookkeeping

**`0.7378` is a `valneg` dice, not an object F1.** `scripts/submit_all.sh:938`
and `:1400` describe it as "obj F1 0.7378 geo_k10 hybrid", and the `pre23`
attention arms are justified partly on that reading. The number is
`clean_geo_k10_tattn_hybrid_ring3`'s best `val/dice` under validation negatives
(table above in this file); its `obj F1` column is `—` and that checkpoint no
longer exists. Nothing downstream was computed from it, so no result changes.

## K10S — retired partition, INVALID RESULTS (`outputs/2026-08-03/`)

| Run | dice | Verdict | Kept |
|---|---|---|---|
| `k10split_convlstm_INVALID` | 0.6694 | ⚠️ **invalid — union-bug training**; weights + 88 G eval deleted 2026-08-11 | metrics only |
| `k10split_single` | 0.6098 | single-frame on the retired partition; superseded | metrics only |

**The 0.6694 is not a real number and neither is the 0.743 object F1 that went
with it.** Two independent inflations, both pushing the same way:

1. **The union bug.** Until 2026-08-06 ~12:41, `dataset.py` took positive patch
   coordinates as the union over the whole input stack rather than from the
   frame being predicted, so a patch positive *last month* entered training with
   an all-zero target — and per-patch Dice scores 1.0 for predicting nothing on
   those. This run's val set was **1.91×** its single-frame sibling (10,403 vs
   5,437), i.e. roughly half of it was all-zero targets. After the fix the two
   paths match to within 0.1% on every partition.
2. **The val split was not held out** — 2019-08→2021-09, interleaved with its
   own 2019-05→2021-09 training range.

`k10split_convlstm_INVALID` is the **only trained model on the wrong side of the
fix**; every run from 2026-08-06 14:34 onward is clean, and the other three
pre-fix runs are single-frame, which never touches that code path.

Full post-mortem, including the two cheap checks that would have caught it:
`outputs/2026-08-03/README.md`. Metrics are kept deliberately — the sample
counts are the evidence for the diagnosis.

`outputs/05082026/` holds one aborted baseline (185 K, no checkpoints).

## Retention policy

Applied to every run directory:

- **`results.csv`, `*.log`, `curves.png`, `validation/`** — kept for every run,
  always. ~13 MB per run; this is what makes the tables above reproducible.
- **`resume.pt`** — deleted everywhere. Full training state (model + optimizer
  + scheduler + RNG) written each epoch for `--resume auto`
  (`sinkholes/training/resume.py:13`). Every run here has finished, by
  completion or by patience, so it is dead weight. *Cost: no run listed here
  can be extended past its final epoch without starting over.*
- **`last.pt`** — deleted everywhere. Final-epoch weights, worse than `best.pt`
  by the definition of model selection; identical to `best.pt` where the best
  epoch was the last one.
- **`interrupted.pt`** — deleted everywhere. Written when a run caught a signal
  before its epoch boundary; four of the 2026-08-10 runs had one from a requeue,
  and all four went on to finish normally, which makes it strictly worse than
  the `best.pt` beside it.
- **`best.pt`** — kept only for runs that are a deployment candidate, a group
  reference point, or a control that a future comparison needs. Dead ends and
  noise probes keep their metrics and lose their weights.

**Exception, 2026-08-11: all 13 runs of the 2026-08-10 batch keep `best.pt`.**
The usual rule would drop several as dead ends on dice — `fuse2`, both 3:1 arms,
both `ring10` arms — but dice is precisely the metric that cannot see what this
batch changed, so "dead end" was not knowable for any of them at the time.
Pruning early costs a retrain *per model* at 4.8–12.5 h.

**This exception was vindicated and now stands indefinitely.** Every one of the
runs the dice rule would have deleted turned out to matter: `geo_k5` 3:1 is the
best model in the project, and `temporal ring10` is the group-T winner. Had they
been pruned on dice on 2026-08-11, both findings would have cost ~10 h of GPU
each to rediscover. **Keep `best.pt` for every run until it has an object-level
score** — the three still unscored (`geo_k5 ring10 _96437`,
`temporal neg3x_ring3 _96433`, `tattn fuse2 _96447`) are the ones that must not
be touched. The same rule applies to the four runs of the 2026-08-11 batch.

Applied 2026-08-11: the batch went from 11 GB to 2.1 GB (30 files: 13 `last.pt`,
13 `resume.pt`, 4 `interrupted.pt`), and moved from the top level of `outputs/`
into `outputs/2026-08-10/` to match the dated-archive convention.

## Evaluation outputs — `predictions/`

The same principle, one level up: **the metrics outlive the inputs.**

- **`olm_results*.json`, `*_pred.npy`, `*_gt.npy`, figures, shapefiles** — kept.
  `_pred.npy` is what makes re-scoring at new thresholds a minutes-long job.
- **`*_image.npy`** — deleted from every completed evaluation on 2026-08-11,
  reclaiming **368 GB** (136 files; it is ~78% of an eval directory). It is the
  reconstructed *input* stack, derivable from the source patches, and no metric
  reads it — `object_level_evaluate` touches the image only for per-object
  features and `eval-outputs` passes `features=()`.

  `outputs.py` was changed to load it only under `--save_figures`, taking the
  South-frame crop extent from `confidence.shape` instead. Verified against real
  data before pruning: a pruned directory re-scores to byte-identical numbers on
  both the North and South frame paths. *Cost: `--save_figures` on a pruned
  evaluation now fails with an explicit message; new figures need a full
  `run_eval.sh`.*

The six `eval2` evaluations submitted 2026-08-11 12:22 were excluded from that
sweep because they were still running. **They and the four `eval3` evaluations
have since finished and still hold their `*_image.npy`** — they are the next
pruning target, worth roughly 3.7 GB per scene.

**17 evaluations now exist**, on two scene lists: 6 geo runs on one 18-scene
list and 11 temporal runs on one 20-scene list. Every object-level number in
this file comes from those, and within a list they are mutually comparable.

## Deletions applied 2026-08-11

| What | Freed | Why |
|---|---|---|
| `predictions/convlstm_10prev_k10split/` | 88 GB | evaluation of the invalid union-bug model |
| `*_image.npy`, completed evals | 368 GB | never read by any metric |
| 2026-08-10 `last`/`resume`/`interrupted.pt` | 8.9 GB | run finished; superseded by `best.pt` |
| `k10split_convlstm_INVALID/best.pt` | 164 MB | invalid training; metrics kept as the record |
| `geo_k10_convlstm_seed7/best.pt` | 164 MB | error-bar probe; metrics carry the finding |

`outputs/` went from 749 GB to ~354 GB. **No run lost its metrics.** The five
metrics-only runs of 2026-08-06 (seed7, cosine-lr5e6, h512, h1024, run1) were
left untouched at 7–20 MB each: they are what establish the ±0.006 noise floor
and the "capacity is saturated" / "cosine is harmful" conclusions, so deleting
them would cost ~2.5 h GPU each to rediscover and would leave every future
"X beats Y" claim without an error bar.

