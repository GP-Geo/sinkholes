# Trained model registry

Every training run under `outputs/`, its result, and what was kept on disk.

> **Looking for the short version?** [RESULTS.md](RESULTS.md) is the readable
> summary: what we know, what to ignore, and what to do next. Each dated folder
> under `outputs/` also carries a `README.md` for that batch. This file is the
> exhaustive record.

Written 2026-08-09, when the run directories were pruned; extended 2026-08-11
with the 13-run 2026-08-10 batch; extended 2026-08-12 with the four-run
2026-08-11 batch and **the object-level scores for all 17 evaluated models**.
The point of this file is that **the metrics outlive the weights**:
`results.csv`, the training log and `curves.png` are kept for every run listed
here, so a deleted checkpoint costs a retrain (~2.5 h GPU) but never costs the
evidence.

## How to read the numbers

`val/dice` is patch-level pixel Dice at a fixed 0.5 threshold, averaged over
batches (`sinkholes/training/evaluate.py:252`). It is **not** the object-level
metric that `eval-outputs` reports.

> ### ⚠️ Do not rank models by `val/dice`
>
> The 2026-08-11 evaluations settled this. Dice does not merely fail to
> discriminate between the leading models — on the negatives batch it ranks them
> **backwards**. `convlstm_geo_k5_..._neg3x_ring3_..._96439` is **last on dice**
> in group G5 (0.6386, −0.0172 from anchor) and **first on scenes**
> (object F1 0.724). The mechanism is known and structural: ring negatives reach
> the **train** split only (`train.py:327/337/362`), so validation stays
> positives-only and the false positives being suppressed lie mostly outside the
> val set.
>
> Dice is now a **training-health signal only** — is the run learning, is it
> diverging, when did it peak. Every promote/kill decision belongs to the
> object-level column.

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
| `convlstm_temporal_k10_h256_..._posw4_60e_..._209866` | 0.6437 | 0.601 @0.5 | 0.671 | 0.803 | top dice scorer; `pos_w` 8→4 bought precision | `best.pt` |
| `convlstm_temporal_k10_h256_b128_lr1e6_60e_lsf_202343` | 0.6427 | 0.581 @0.5 | 0.659 | 0.815 | group-T reference point | `best.pt` |
| `convlstm_temporal_k5_h256_..._15h58_lsf_208207` | 0.6420 | 0.576 @0.5 | 0.674 | 0.793 | **k5 champion** — ties k10 on 34% more data | `best.pt` |
| `convlstm_temporal_k10_h512_..._209863` | 0.6401 | — | 0.656 | 0.818 | capacity is saturated; no gain over h256 | metrics only |
| `convlstm_temporal_k5_h256_..._13h51` | 0.6376 | — | 0.667 | 0.791 | first of the duplicate pair; noise probe | metrics only |
| `convlstm_temporal_k10_h256_..._seed7_..._209867` | 0.6368 | — | 0.622 | 0.872 | seed probe; defines the ±0.006 floor | metrics only |
| `convlstm_temporal_k10_h1024_..._202344` | 0.6296 | — | 0.618 | 0.864 | **worse**; early-stopped @36, undertrained | metrics only |
| `unet_stack_temporal_k10_..._209864` | 0.6284 | 0.562 @0.5 | 0.662 | 0.804 | channel-stacking control — loses to ConvLSTM by 0.0143 | `best.pt` |
| `convlstm_temporal_k10_h256_..._lr5e6_cosine_..._209865` | 0.6179 | — | 0.614 | 0.843 | **worst ConvLSTM**; early-stop @34, 5× the jitter | metrics only |
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
| `convlstm_geo_k5_h256_..._207929` | 0.6539 | 0.647 @0.9 | 0.746 | 0.722 | most balanced patch P/R of any run | `best.pt` |
| `baseline_single_geo_k5partition_b64_lr1e5_60e` | 0.6084 | — | 0.616 | 0.809 | single-frame floor; **lr/batch confounded — superseded** by `unet_single_geo_k5_posw4_60e_..._501433` | `best.pt` |

## Group G10 — geo_k10 partition (1 run)

| Run | dice | obj F1 | P | R | Verdict | Kept |
|---|---|---|---|---|---|---|
| `convlstm_geo_k10_h256_..._206375` | 0.6583 | 0.456 @0.5 | 0.707 | 0.800 | best epoch 28/60, early-stop @48 | `best.pt` |

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
| `convlstm_temporal_k5_..._neg1x_ring10_..._96432` | **0.698** @0.25 | 0.6432 | +0.0041 | ⭐ **group-T winner.** Far-field negatives: precision 0.451 → 0.726. Was 6th on dice | `best.pt` |
| `tattn_..._fuse0_..._neg1x_ring3_..._96445` | 0.642 @0.25 | 0.6459 | −0.0001 | **attention beats recurrence** by +0.016 at 26% fewer params | `best.pt` |
| `tattn_..._fuse0_recur-convlstm_..._neg1x_ring3_..._96449` | 0.639 @0.25 | 0.6468 | +0.0009 | hybrid; no gain over plain attention, and costlier | `best.pt` |
| `convlstm_temporal_k5_..._neg1x_ring3_..._96429` | 0.626 @0.25 | 0.6460 | +0.0069 | **the negatives reference**; twin of the `tattn` run above | `best.pt` |
| `convlstm_temporal_k5_h256_posw4_60e_..._82458` | 0.612 @0.5 | 0.6391 | anchor | the `pos_w` 4 baseline both experiments are read against | `best.pt` |
| `tattn_temporal_k5_d256_fuse0_posw4_60e_..._96442` | 0.598 @0.5 | 0.6460 | +0.0069 | head-to-head, no negatives | `best.pt` |
| `convlstm_temporal_k5_..._neg3x_ring3_..._96433` | — | 0.6449 | +0.0058 | 3:1 negatives; **unscored** — see the 3:1 note below | `best.pt` |
| `tattn_..._fuse2_..._neg1x_ring3_..._96447` | — | 0.6399 | −0.0060 | temporally-fused skips, **wrong direction**; unscored | `best.pt` |

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
| `convlstm_geo_k5_..._neg3x_ring3_..._96439` | **0.724** @0.9 | 0.6386 | −0.0172 | ⭐ **best geo model in the project — and last on dice.** Holds recall 0.773 @0.9 where 1:1 drops to 0.615 | `best.pt` |
| `convlstm_geo_k5_..._neg1x_ring3_..._96436` | 0.712 @0.7 | 0.6541 | −0.0017 | near-field negatives; precision 0.449 → 0.545 @0.5 at no recall cost | `best.pt` |
| `convlstm_geo_k5_h256_posw4_60e_..._82459` | 0.667 @0.9 | 0.6558 | anchor | the `pos_w` 4 geo baseline; **top of the group on dice, third on scenes** | `best.pt` |
| `convlstm_geo_k5_..._neg1x_ring10_..._96437` | — | 0.6461 | −0.0097 | far-field; **unscored, and the single highest-value job available** | `best.pt` |

### Group G10 addition — geo_k10 (val 4060)

| Run | obj F1 | dice | Δ dice | Verdict | Kept |
|---|---|---|---|---|---|
| `convlstm_geo_k10_..._neg1x_ring3_..._96441` | 0.666 @0.7 | 0.6596 | −0.0050 | ring negatives lift obj F1 0.456 → 0.666 over `_206375` | `best.pt` |

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
temporal 3:1 arm `_96433` is still unscored, which would give it a second data
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
`geo_k5` at `pos_w` 4, batch 128, lr 1e-6 — matched to the `_82459` anchor, so
they are directly comparable to the group-G5 additions above on dice.

**All four finished cleanly.** Three stopped on early-stopping patience (20
epochs without improvement) and one ran the full 60; none was killed by
walltime, so none is undertrained. All four kept `best.pt`.

**None has an object-level score.** Per the warning at the top of this file, the
dice column below must not be used to rank them.

| Run | dice | obj F1 | Epochs | Purpose | Kept |
|---|---|---|---|---|---|
| `tattn_geo_k5_..._recur-convlstm_..._neg1x_ring3_..._493315` | 0.6487 | — | best @22, stop @42 | hybrid attention on geo | `best.pt` |
| `unet_single_geo_k5_posw4_60e_..._501433` | 0.6434 | — | best @54, ran 60 | ⭐ **de-confounded single-frame control** | `best.pt` |
| `tattn_geo_k5_d256_fuse0_..._neg1x_ring3_..._493314` | 0.6433 | — | best @16, stop @36 | plain attention on geo | `best.pt` |
| `unet_single_geo_k5_posw4_neg1x_ring3_..._501434` | 0.6341 | — | best @20, stop @40 | single-frame **with** ring negatives | `best.pt` |

### What this batch already settled

`unet_single_geo_k5_posw4_60e_..._501433` is the run that **corrects the
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

## K10S — retired partition, INVALID RESULTS (`outputs/2026-08-03/`)

| Run | dice | Verdict | Kept |
|---|---|---|---|
| `k10split_convlstm-h256` | 0.6694 | ⚠️ **invalid — union-bug training**; weights + 88 G eval deleted 2026-08-11 | metrics only |
| `k10split_unet-single` | 0.6098 | single-frame on the retired partition; superseded | metrics only |

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

`k10split_convlstm-h256` is the **only trained model on the wrong side of the
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
| `k10split_convlstm-h256/best.pt` | 164 MB | invalid training; metrics kept as the record |
| `geo-k10_..._seed7_lsf594073/best.pt` | 164 MB | error-bar probe; metrics carry the finding |

`outputs/` went from 749 GB to ~354 GB. **No run lost its metrics.** The five
metrics-only runs of 2026-08-06 (seed7, cosine-lr5e6, h512, h1024, run1) were
left untouched at 7–20 MB each: they are what establish the ±0.006 noise floor
and the "capacity is saturated" / "cosine is harmful" conclusions, so deleting
them would cost ~2.5 h GPU each to rediscover and would leave every future
"X beats Y" claim without an error bar.

