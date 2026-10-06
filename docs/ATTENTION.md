# Temporal attention — how it works, how we check it, what we know

*Formerly `docs/ATTENTION_COLLAPSE.md`, renamed 2026-09-17. The collapse it was named after
is now the [history section](#history-the-august-collapse).*

**Status, 2026-09-17.** The attention **selects**. Every run probed since the September
retraining under AdamW uses 22–34% of uniform, sharper than it started, and none has
collapsed. How much the
prediction **depends on the history** varies ten-fold between runs, and depth moves it: at
k10 with context, hiding the history costs 0.044 patch Dice, almost all of it recall. At the
object level the best k10 model is **plain** attention (F1 0.7907): context has lost to plain
in all five pairs scored. Still open: whether the history effect holds on whole scenes, with
negatives.

- [What the attention does](#what-the-attention-does)
- [How we check it](#how-we-check-it)
- [What we know (September 2026)](#what-we-know-september-2026)
- [Open questions](#open-questions)
- [History: the August collapse](#history-the-august-collapse)
- [Reference](#reference)

## What the attention does

`tattn_unet` (`models/tattn_unet.py`) is a U-Net whose input is a **stack** of
interferograms: the current one and the k before it, oldest first, 11 days apart. Every
frame goes through the same encoder. At the bottleneck — a 12×6 grid for a 200×100 patch,
18×12 at ctx50 — each cell of the **current** frame looks back at the same cell in every
frame of its history and gives each frame a weight. The weights add up to 1, and the cell's
output is the history combined with those weights (`_TemporalReadoutBlock`, 8 heads, each
with its own weights). The decoder upsamples that; with `FUSE_SKIPS=0`, as in every
September run, its skip connections come from the current frame alone.

That can go two ways:

- **Averaging** — every frame gets 1/T. The ~1M attention parameters then do nothing a
  plain mean would not. Every checkpoint trained before 2026-08-19 did exactly this
  ([History](#history-the-august-collapse)).
- **Selecting** — each cell puts most of its weight on a few frames, and which frames
  depends on what it sees.

`--tattn_contrast` and `--tattn_qk_norm`, on by default since 2026-08-19, are what made
selecting possible: queries and keys are built from how each frame *differs* from the
temporal mean, and their scale is set by one learned temperature (initialised at 10,
clamped at 100) instead of by the size of the projections ([The fix](#the-fix)).

## How we check it

Three questions, in order, on real validation patches from the run's own partition, at the
depth it was trained at.

### 1. Is it averaging or selecting?

Read the weights and count how many frames each cell **effectively uses**: `exp(entropy)` of
its weight distribution, averaged over every (patch, head, cell) — not the entropy of the
mean distribution, which would hide a mix of peaked and flat cells. T of T is a pure
average; 1 is a single frame. It is quoted as a share of T, "of uniform".

On its own that number means little. Two references make it readable:

- **The untrained reference** — the same architecture with fresh weights, on the same
  patches. The gap is what training taught. In the August RMSprop runs training made
  models *less* selective than they started, even with the fix; in the September AdamW
  runs it makes them more.
- **Between-sample spread** — how much the weight profile changes from patch to patch.
  Near zero is a fixed recipe, applied whatever the input.

`--require_selectivity 0.9` turns the share into an exit code. It is a **collapse alarm,
not a pass mark**: a model at 89% of uniform passes it and is still averaging in practice.

### 2. Which frames does it pick?

The mean weight at each offset — 0 is the current frame, 1 is 11 days back — as a multiple
of uniform, overall and per head, alongside the **argmax share**: how often each offset is
the single heaviest frame. Both are needed. Cells that each peak on a *different* frame
average out to a profile that looks flat.

### 3. Does the choice reach the prediction?

The weights show what the model does, not whether it matters. So the trained model is
broken on purpose, two ways, and scored against itself on the same patches:

- **Forced uniform** — `logit_scale` set to −60, so every frame weighs exactly 1/T: the
  collapsed model, built deliberately. The Dice it loses is what *which frames* is worth.
- **Present only** — the `valid` mask hides every past frame, so the attention sits
  entirely on the current one and no history enters. The Dice it loses is what *the
  history* is worth.

Both are scored as pooled pixel Dice at 0.5, with precision, recall, and the share of pixels
whose thresholded decision flips. Which of precision or recall moves says what the history
was doing.

### The two controls that make a number mean something

A single effective-frames figure is close to uninterpretable on its own — 8.672
of 11 is neither uniform nor selective until you know what the same architecture
does untouched. Both controls are built by writing a state dict to a scratch
path and pointing the probe at it, so neither needs a GPU or a training run.

- **The untrained reference.** Build `TemporalAttentionUNet` with the run's own
  `config_dict()`, save `state_dict()` plus the `tattn_unet_config` blob, probe
  it on the same split. This is the only way to tell selectivity that was
  *learned* from selectivity that initialisation supplied — and for the fixed
  block in August the answer was that training gave back more than it earned.
- **The temperature override.** Load a trained checkpoint, set
  `temporal_attn.readout.logit_scale` to `log(10)`, change nothing else, re-probe.
  Because argmax over time does not depend on temperature, the argmax shares come
  back bit-identical and any change in effective frames is *purely* sharpness.
  That separates "the queries and keys learned nothing" from "they learned
  something the temperature is flattening", which are opposite findings with
  opposite remedies.

### What these numbers cannot show

- **Positives only.** Validation holds only patches that contain a sinkhole, so none of
  this sees false alarms on empty ground — which may be where history matters most.
- **Pooled Dice is not object F1, and neither is `val/dice`.** Pooled Dice is dominated by
  large objects; `results.csv`'s `val/dice` is a per-patch mean that weights small ones.
  The generation-3 k5 pair tied on the first and differed by 0.029 on the second. Object
  F1 from `run_eval.sh` decides.
- **No error bars.** One training run per configuration and one patch sample. The only
  measured noise is 0.011 val/dice between two identical runs ([EXPERIMENTS.md](EXPERIMENTS.md), `combo`).
- **Patches differ between runs** unless probed together. Within each pair below the cells
  are identical, and checked; across pairs they are not.

### Running it

Everything here runs locally on MPS against the mounted patch tree — no cluster job. It is
I/O-bound: 2–10 minutes per checkpoint over SMB.

**Stock probe, plain 200×100 checkpoints only.** `sinkholes attention-probe` exits on a
context checkpoint.

```bash
sinkholes attention-probe --model outputs/<run>/checkpoints/best.pt \
  --partition assets/<the partition the run trained on>.json --split val \
  --patches_dir "$DATA/patches" --control_lookback <k> --require_selectivity 0.9
```

To run a checkpoint on a longer history than it was trained at, and score Dice at several
depths on the same patches:

```bash
sinkholes attention-probe \
  --model outputs/<run>/checkpoints/best.pt \
  --partition assets/partition_geo_k10_clean.json --split val \
  --patches_dir "$DATA/patches" \
  --lookback 40 --control_lookback 10 \
  --score_depths 1 3 5 10 20 40 \
  --max_per_intf 12 --out_dir outputs/attention_probe/<run>
```

Writes `weights_<condition>.csv` (per-offset mean weight, argmax share,
availability), `summary.json` and `dice_vs_depth.csv`. The 2026-08-20 results
are on disk under `outputs/attention_probe/`, one directory per run plus
`diagnostics/` for the two controls above.

**Either geometry, with the untrained reference and both ablations.** A wrapper around the
probe's own `select_history`, `replicated_plan`, `WeightStats` and `_sample_coords`. With
`--margin 50` it reads full 300×200 windows from
`data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned`; with `--margin 0`, the plain tree.
The two trees share one grid and one `nonz_indices.json`, so the same seed samples the same
cells, and the JSON records the coordinates so that can be checked. Pass several `--models`
of one geometry to probe them on one patch set.

```bash
PYTHONPATH=. python outputs/attention_probe/k10_tattn_plain_vs_ctx50_2026-09-17/geom_probe.py \
  --margin 50 --models outputs/<run>/checkpoints/best.pt \
  --partition assets/partition_temporal_k10_clean_th350x200.json --split val \
  --patches_dir "$DATA/patches" --k 10 --max_per_intf 48 \
  --out outputs/attention_probe/<name>/<run>.json
```

The wrapper lives in a gitignored directory and is not part of the package
([Open questions](#open-questions), 5). `outputs/` is gitignored as a whole, so the tables in
this file are the durable record and the CSVs and JSONs are working copies.

## What we know (September 2026)

The six attention-only runs probed since the correctness fix — AdamW 3e-4, weight decay
1e-2, `FUSE_SKIPS=0`, `RECURRENCE=none`:

| run | k | input | partition | pos_w | frames used (of uniform) | temp. | no history ΔDice | forced uniform ΔDice | object F1 `ith0.7_b5` |
|---|---|---|---|---|---|---|---|---|---|
| `posw8_tattn_k10_ctx50_30e` | 10 | ctx50 | gen 4 | 8 | 2.48 / 11 (22.5%) | 10.7 | **−0.044** | +0.002 | 0.7748 (17 scenes) |
| `posw8_tattn_k10_plain_30e` | 10 | plain | gen 4 | 8 | 2.62 / 11 (23.8%) | 10.0 | −0.013 | −0.022 | **0.7907** (17 scenes) |
| `th350_tattn_ctx50_neg10` | 5 | ctx50 | gen 4 | 4 | 1.88 / 6 (31.3%) | 9.7 | −0.002 | −0.010 | 0.7905 (20 scenes) |
| `posw8_tattn_ctx50_25e` | 5 | ctx50 | gen 4 | 8 | 1.82 / 6 (30.3%) | 10.2 | −0.004 | −0.008 | 0.7699 (20 scenes) |
| `longgrid_tattn_ctx50` (deleted) | 5 | ctx50 | gen 3 | 4 | 1.91 / 6 (31.9%) | 9.6 | −0.036 | −0.012 | 0.7212 (gen-3 list) |
| `longgrid_tattn_plain` (deleted) | 5 | plain | gen 3 | 4 | 2.01 / 6 (33.5%) | 7.8 | −0.048 | −0.035 | 0.7498 (gen-3 list) |

The untrained reference is 53% of uniform at both depths. Temperatures are at `best.pt`.
Object F1s are best of four RTh and compare only within one scene list. ΔDice is pooled
pixel Dice on each run's own val patches, so it compares only within a pair.

**In short**

- **Selection is settled.** All six fixed September runs probed select, at 22–34% of
  uniform, each sharper than its untrained reference, with the temperature held at
  7.8–10.7. The erosion measured in August has not recurred under AdamW.
- **How much the history is used is not a property of the architecture.** It ranges ten-fold
  across these runs (0.002–0.048 Dice), and on generation 4 depth is the one lever
  measured to move it.
- **Context changes what the history is for.** With context, the selection matters less
  in both pairs, and at k10 the history is what recovers the recall context costs. At the
  object level context has lost to plain in every pair scored so far — five of five,
  including attention at k10, where plain is the best model (0.7907 vs 0.7748; see
  [EXPERIMENTS.md](EXPERIMENTS.md), `longgrid` and `k10`).

### k10: history carries the recall

`posw8_tattn_k10_plain_30e` vs `posw8_tattn_k10_ctx50_30e`, identical except for the
50 px margin.

`partition_temporal_k10_clean_th350x200`, pos_w 8, ring 1–10, batch 32 × 4, 30 epochs;
480 patches over 10 val interferograms, 11-frame histories.

| | plain | ctx50 |
|---|---|---|
| effective frames | 2.618 / 11 (23.8%) | 2.479 / 11 (22.5%) |
| temperature | 9.97 | 10.69 |
| as trained, Dice (P / R) | 0.7878 (0.747 / 0.833) | 0.7829 (0.738 / 0.834) |
| forced uniform, ΔDice (P / R) | −0.022 (0.680 / 0.876) | +0.002 (0.747 / 0.826) |
| present only, ΔDice (P / R) | −0.013 (0.737 / 0.817) | **−0.044** (0.830 / 0.666) |
| object F1 `ith0.7_b5` (P / R), 17 test scenes | **0.7907** (0.819 / 0.764) | 0.7748 (0.858 / 0.706) |

The untrained reference is 5.848 / 11 (53.2%) for both.

1. **Selective at 11 frames, and sharper relative to T than at 6.** Both use about 23% of
   uniform, against ~32% at k5. Training sharpened them from 53%, and the temperature
   held at ~10.
2. **In the ctx50 model the history carries the recall.** Masking it drops recall
   0.834 → 0.666 and lifts precision 0.738 → 0.830 — close to where the ctx50
   single-frame model sits at object level (0.871 / 0.647). The object scores agree:
   ctx50 attention gets +0.059 recall over its single-frame twin for −0.013 precision.
   At k10 the history is what buys back the recall context costs.
3. **With context, *which* frames matters little.** Flattening the ctx50 model's
   selection changes nothing (+0.002); at plain it costs more than removing the history
   altogether (−0.022 vs −0.013). That matches k5, where context also shrank what
   selection adds (−0.035 → −0.012).
4. **Depth changes how much the history is used.** On generation 4 at pos_w 8, masking
   the history cost the k5 ctx50 run −0.004 and costs the k10 ctx50 run −0.044 — ten
   times more with the same partition family, weight and ring. The val splits differ
   (k10's 10 interferograms are a subset of k5's 11). This is the patch-level side of
   the object-level temporal edge widening from +0.007 at k5 to +0.032 at k10
   ([EXPERIMENTS.md](EXPERIMENTS.md), `k10`).
5. **At the object level, plain attention is the better k10 model.** 0.7907 against
   0.7748, and the difference is recall (0.764 vs 0.706) bought with precision (0.819 vs
   0.858). History helps at both geometries — attention over the single frame is +0.022
   plain and +0.032 ctx50 — and context costs F1 at both, less with attention (−0.016)
   than without (−0.026). The ctx50 model leans on its history for recall because
   context took recall away in the first place.

**Where the weight lands differs by geometry.** Plain down-weights the current frame
(0.57x), has a bump at 22–33 days (1.26x), and puts its heaviest weight on the oldest
frame, 110 days back (1.70x, the argmax at 16.6% of positions). ctx50 is U-shaped:
1.37–1.60x on the newest three frames, a trough of 0.34–0.40x at 55–66 days, and 1.26x
again at 110 days. Its heads split cleanly: head 4 sits on the current frame (5.07x),
heads 1 and 7 on 0–22 days, head 0 at 88 days, heads 2 and 3 at 99–110 days. **Both lean
on the edge of their window**, which is what a model that wants more history would do.
A k=10 checkpoint cannot show whether it does; that needs a run trained deeper.

What this does not show: positive patches only, pooled pixel Dice, no bootstrap. The
plain checkpoint's `best.pt` is from epoch 29 of 30 and may be under-trained against its
ctx50 twin (epoch 11): it never cut its learning rate in 30 epochs. It won at object level
anyway; `k10plain45` retrains it for 45 epochs at `LR_PATIENCE=5`.

Results and wrapper: `outputs/attention_probe/k10_tattn_plain_vs_ctx50_2026-09-17/`.

### k5, generation 4: selective, and barely used

> **Scope.** "Selective" has held in every run probed since. "Barely used" is true of these
> two runs only: hiding the history costs 0.013–0.048 Dice on the k10 pair
> ([k10](#k10-history-carries-the-recall)) and on the generation-3 k5 pair
> ([k5, generation 3](#k5-generation-3-plain-vs-ctx50)).

**Measured 2026-09-16** on two attention-only runs, both
`RECURRENCE=none FUSE_SKIPS=0`, k=5, ctx50, AdamW at lr 3e-4 with weight decay
0.01, on `partition_temporal_k5_clean_th350x200`:

- `th350_tattn_ctx50_neg10` (pos_w 4; `best.pt` from epoch 15 of 60)
- `posw8_tattn_ctx50_25e` (pos_w 8; `best.pt` from epoch 18 of 25)

Both `best.pt` checkpoints were probed on the same 528 positive patches: 48 per
interferogram over the 11-interferogram th350 val split, all with a gap-free
6-frame history.

The trained models score Dice 0.79 on these patches, against 0.076 untrained, so the
context windows are fed correctly.

#### No collapse, and no erosion either

| | `th350_tattn` | `posw8_tattn` | same architecture, untrained |
|---|---|---|---|
| effective frames | **1.876 / 6** | **1.821 / 6** | 3.215 / 6 |
| of uniform | **31.3%** | **30.3%** | 53.6% |
| `logit_scale.exp()` at `best.pt` | 9.678 | 10.217 | 10.000 |
| `logit_scale.exp()` at `last.pt` | 9.052 (epoch 60) | 10.076 (epoch 25) | — |
| between-sample std of the weight profile | 0.040 | 0.043 | 0.0005 |

This is the reverse of
[Training erodes the selectivity it is given](#training-erodes-the-selectivity-it-is-given).
There, training moved the fixed block from 49.7% to 78.8% of uniform. Here it
moved from 53.6% to about 30%. The temperature did not decay either: th350 still
had 9.05 after 60 epochs, where the August runs were down to 1.0–1.3. The weights
also depend on the input — they vary between patches about 80x more than the
untrained model's do.

**Why the temperature held here is not isolated.** These runs differ from the
August ones in optimiser (AdamW at lr 3e-4 against RMSprop at 1e-6), loss
(`SEG_LOSS=dice`), context size, ring negatives and partition. The optimiser is
the obvious candidate, but no paired run has tested it.

Where the weight lands (multiple of uniform; argmax share in brackets):

| offset | days | `th350_tattn` | `posw8_tattn` |
|---|---|---|---|
| 0 (current) | 0 | 0.88x (13%) | 0.46x (7%) |
| 1 | 11 | 1.50x (26%) | 0.98x (16%) |
| 2 | 22 | 1.55x (27%) | 1.36x (23%) |
| 3 | 33 | 0.93x (15%) | 1.15x (19%) |
| 4 | 44 | 0.66x (11%) | 1.12x (19%) |
| 5 | 55 | 0.48x (8%) | 0.94x (15%) |

Both down-weight the current frame, as the August runs did (it already reaches
the decoder through the skips), and both peak 11–22 days back. The heads split
by age. In `posw8_tattn`, heads 1 and 4 put 2.1–2.7x uniform on offsets 1–2 and
almost nothing on offset 5, while heads 0, 5 and 7 peak at offsets 4–5. In
`th350_tattn`, head 4 holds the recent frames and head 2 the oldest.

#### The selection barely reaches the prediction

Two ablations on the same trained weights and the same patches, pixel Dice at
0.5:

- **forced uniform:** `logit_scale` set to −60, so the softmax is exactly 1/T —
  the collapsed model, built on purpose.
- **present only:** `valid` masks every past frame, so attention is one-hot on
  offset 0 and no history enters the value path.

| condition | `th350_tattn` Dice (P / R) | Δ | `posw8_tattn` Dice (P / R) | Δ |
|---|---|---|---|---|
| as trained | 0.7925 (0.751 / 0.839) | — | 0.7919 (0.768 / 0.818) | — |
| forced uniform | 0.7829 (0.730 / 0.844) | −0.0095 | 0.7844 (0.736 / 0.840) | −0.0075 |
| present only | 0.7904 (0.762 / 0.822) | −0.0020 | 0.7884 (0.770 / 0.807) | −0.0036 |

Either ablation flips the thresholded decision on only 0.7–0.8% of pixels.
Three readings:

1. **The attention is selecting, not averaging.** Turning it into an average
   costs Dice, mostly as precision.
2. **An average is worse than no history at all.** Forced uniform costs 2–5x
   more Dice than present only. What the learned weighting mainly does is keep
   the history from hurting.
3. **The net value of the history is small on these patches**: 0.002–0.004
   Dice, a little recall for a little precision. At th350, the tattn arm beats
   the single-frame arm on object F1 (0.7905 vs 0.7535, [EXPERIMENTS.md](EXPERIMENTS.md),
   `th350`). This measurement cannot attribute that gain to temporal selection.

**What this does not show.** The th350 val split has no negatives, so these are
positive patches only. Rejecting false alarms on empty ground is where history
might matter most, and this test cannot see it. It is pooled pixel Dice, not
object F1, over 11 interferograms, with no bootstrap: a Δ of 0.002–0.004 may be
noise. The two hybrids (`th350_hybrid_ctx50_neg10`, `posw8_hybrid_ctx50_25e`)
also carry attention and were not probed.

Results, with per-offset and per-head rows for every condition:
`outputs/attention_probe/ctx50_tattn_2026-09-16/`. They were produced by the ctx50-only
predecessor of the wrapper under [Running it](#running-it); the numbers are the same
measurement.

### k5, generation 3: plain vs ctx50

`longgrid_tattn_plain` vs `longgrid_tattn_ctx50`, identical except for the 50 px margin.
**Both deleted 2026-09-17**, with the rest of the generation-3 `long*` runs.

`partition_temporal_k5_clean`, pos_w 4, ring 1–3, batch 64 × 2; 1,008 patches over 21
val interferograms.

| | plain | ctx50 |
|---|---|---|
| effective frames | 2.008 / 6 (33.5%) | 1.913 / 6 (31.9%) |
| temperature | 7.80 | 9.59 |
| as trained, Dice (P / R) | 0.7726 (0.749 / 0.798) | 0.7717 (0.747 / 0.798) |
| forced uniform, ΔDice | −0.035 | −0.012 |
| present only, ΔDice (P / R) | **−0.048** (0.628 / 0.857) | **−0.036** (0.649 / 0.850) |
| object F1 `ith0.7_b5` (P / R), 20 test scenes | **0.7498** (0.793 / 0.711) | 0.7212 (0.847 / 0.628) |

The untrained reference is 3.215 / 6 (53.6%) for both.

1. **Here the history carries real weight.** Masking it costs 0.036–0.048 Dice, roughly
   ten times the generation-4 k5 runs. The two families differ in partition
   (generation 3 vs 4) and ring radius (3 vs 10), and which of those matters is not
   isolated.
2. **What the history buys is precision.** Without it precision falls 0.749 → 0.628 and
   recall rises. That is the high-recall, low-precision shape the single-frame models
   show at object level.
3. **Context stands in for part of it.** The ctx50 model loses less without its history
   (−0.036 vs −0.048) and far less when its selection is flattened (−0.012 vs −0.035).
4. **Context does not win.** Pooled Dice ties (0.7726 vs 0.7717). The logged val/dice
   gap of +0.029 for ctx50 is a per-patch mean, a different metric, and at object level
   plain leads by 0.029 F1 because context costs 0.083 recall.

Both put most weight on the oldest frames — plain peaks at 55 days (1.60x uniform) and
ctx50 at 44 days (1.40x), with the current frame at 0.35x and 0.54x. That differs from
the generation-4 k5 runs, which peaked 11–22 days back. Generation 3 still contains
the badly digitised scenes the 350N/200S threshold removes, so read this pair for its
within-pair differences, not its absolute scores.

Results and wrapper: `outputs/attention_probe/longgrid_tattn_plain_vs_ctx50_2026-09-16/`.
The checkpoints are gone, so this pair cannot be re-probed; the table above and that
directory are the record.

## Open questions

1. **How good is plain attention when trained to convergence, and does the context loss
   hold for the k5 leaders?** `posw8_tattn_k10_plain_30e` is the best k10 model (0.7907)
   while still improving at epoch 29 with its learning rate never cut:
   `submit_all.sh k10plain45` retrains it for 45 epochs at `LR_PATIENCE=5`.
   `submit_all.sh k5plain` trains plain twins of `th350_tattn_ctx50_neg10` and
   `posw8_tattn_ctx50_25e` — at `LR_PATIENCE=5`, so each pair differs in the schedule as
   well as the context.
2. **Does the history carry the gain on whole scenes?** Score the present-only ablation at
   scene level, with negatives, against the as-trained `posw8_tattn_k10_ctx50_30e` — the run
   where it should show if it shows anywhere. Patch-level probes cannot see false alarms on
   empty ground.
3. **Would more history help?** Both k10 models put heavy weight on the oldest frame in their
   window, 110 days back, but only a model trained deeper can say. The model side is ready:
   `select_history()` and the padding mask make a gappy 40-slot lookback expressible. The
   data layer is not. It materialises `(T, N, H, W)` per interferogram
   (`dataprep/dataset.py:411`) and reloads each grid once per referencing current, so host
   RAM grows linearly with T — `posw8_tattn_k10_ctx50_30e` already peaked at 206 GB
   (LSF 254183). A dense long lookback needs the deduplicated frame store first: load each
   distinct grid once and have samples point at rows. Memory then stops growing with depth,
   because sinkholes sit still and the per-frame coordinate union saturates.
4. **Why does history use vary ten-fold?** 0.002–0.004 Dice on the generation-4 k5 runs,
   0.036–0.048 on the generation-3 k5 pair, 0.013–0.044 at k10. Partition, ring radius and
   depth all differ between those groups; only depth has been moved on its own.
5. **Fold the wrapper into `sinkholes attention-probe`**, so context checkpoints get the
   ablations and `--require_selectivity`, and the tool stops living in a gitignored folder.
6. **The hybrids are unread.** In August 3 of 4 hybrids failed the collapse alarm
   ([EXPERIMENTS.md](EXPERIMENTS.md), `probe`): when a ConvLSTM already integrates over
   time, nothing pushes the attention to select. `th350_hybrid_ctx50_neg10` and
   `posw8_hybrid_ctx50_25e` have not been probed.

## History: the August collapse

*The record of 2026-08-19/20: how the collapse was found, what caused it, the fix, and what
the fix did on the first runs trained with it. Those runs used RMSprop at lr 1e-6–1e-5. The
selectivity erosion and temperature decay measured here have not recurred under AdamW —
see [What we know](#what-we-know-september-2026).*

### What happened

Every trained `tattn_unet` checkpoint in `outputs/` — **15 of 15** — produced
attention weights that were *exactly* `1/T` at every timestep, head and bottleneck
pixel. The temporal attention was not selecting frames; it was computing an
unweighted mean, and the ~1M attention parameters were decoration.

The cause is **not** a training pathology that develops over a run. It is present
before the first gradient step: at initialisation the tokens attention sees are
99.91% a component shared across the whole sequence, so there is nothing to select
on. Two changes fix it, and neither is sufficient alone.

> **Naming, 2026-08-20.** Every run named in this document validated against
> negatives and its directory now ends `_valneg` (`clean_geo_k10_tattn_ring3` →
> `attention_probe/geo_k10_tattn_ring3_valneg`, and so on). The names in the
> prose below are the ones the runs had when the work was done;
> `outputs/README.md` carries the mapping.

This does not invalidate the results: `clean_geo_k10_tattn_ring3` scores 0.727
object F1 and still beats its ConvLSTM twin. But the mechanism those results were
attributed to was not the mechanism running, and any experiment premised on
attention *choosing* frames — long or hole-tolerant histories above all — was
untestable until now.

The fix has since been trained and measured. It works, and it is half-finished:
the retrained runs select rather than average, but training walks most of that
selectivity back, and one of them collapses anyway. See
[What the fix did in a real run](#what-the-fix-did-in-a-real-run).

### The measurement

`sinkholes attention-probe` on `clean_geo_k10_tattn_ring3`, 272 patches over the
23-interferogram `partition_geo_k10_clean` val split:

| condition | frames | effective frames used | uniform would be |
|---|---|---|---|
| control, trained depth | 11.0 | **11.000** | 11.0 |
| probe, 40-slot lookback | 32.7 | **32.669** | 32.669 |

"Effective frames" is `exp(entropy)` of the weight distribution, averaged per
(sample, head, pixel) — not of the mean distribution, which would hide a mixture of
collapsed and spread-out pixels. It equals the frame count to five significant
figures: uniform, not merely broad.

Across all 15 checkpoints, measured on real patches: **14 at ≥99.9% of uniform**,
8 with weight spread *exactly* 0.0. The only partial exception is
`temporal_k5_tattn_fuse2_ring3` (94.3%) — the run `docs/MODEL_RUNS.md:206` already
records as "wrong direction, unscored".

Three independent confirmations that this is the checkpoints and not the probe:

1. The same uniformity appears on the untouched legacy path (`offsets=None`,
   `valid=None`, T=11 — exactly the training configuration), and the model scores
   dice 0.64 on those patches, so they are fed correctly.
2. Replacing the input with uniform random noise moves the weights by `4.3e-5` —
   the same size as their own variation. The attention cannot see its input.
3. `q_proj`/`k_proj` weight RMS is 0.0000–0.0041 against an initialisation of
   ~0.036, and `in_norm.weight` is 0.022–0.053 against an initialisation of 1.0.

### The cause

Trace the share of variation that is *temporal* through the encoder, and it holds
up fine — 87% of the bottleneck's variation is across-time at init. The signal is
there. What kills attention is its **scale relative to what it sits on top of**:

| | shared component | per-frame difference | ratio |
|---|---|---|---|
| at initialisation | 0.9852 | 0.000925 | **0.00094** |
| after training | 0.1514 | 0.076046 | 0.50226 |

At init the frame-to-frame differences are **0.09%** of the token magnitude, and
consecutive frames have cosine similarity **1.0000**. Attention is asked to choose
between eleven vectors that are, to five decimal places, the same vector. The
softmax is uniform before training starts (10.97/11 effective frames), the gradient
reaching `q_proj`/`k_proj` is ~10x weaker than the one reaching the value path, and
`<grad, param>` is consistently positive — descent shrinks them.

RMSprop then finishes the job. It normalises by gradient RMS, so each step moves
roughly `lr` regardless of gradient size: 60 epochs x ~430 steps x `lr=1e-6` is
~0.026 of travel, and the projections only need to cover 0.036 to reach zero.

The encoder *does* eventually learn temporal contrast — the ratio reaches 0.50 by
the end of training. It learns it far too late. By then q/k are dead, and a uniform
softmax is a plateau with no gradient back out. The attention loses a race it was
never in a position to win.

Note this also makes the failure mode a function of learning rate, which is the
tell that the logits are unscaled. Same block, same data, 400 steps:

| lr | effective frames at step 400 |
|---|---|
| 1e-6 (production) | 7.62 → heading to uniform |
| 1e-5 | 1.13 → saturated one-hot |

Both are degenerate; which one you land on is decided by projection magnitude.

### The fix

Two changes to `_TemporalReadoutBlock` (and to `_CausalSelfAttentionBlock`, which
`tattn_layers > 1` reaches):

1. **`contrast`** — form queries and keys from `token - mean_over_time(token)`,
   renormalised, so attention selects on *how frames differ* rather than on what
   they share. The value path still sees the whole token, because the answer must
   still carry content, not just deviation. The mean skips padded frames, or the
   replicated filler would pollute the very baseline the contrast measures against.
2. **`qk_norm`** — unit-norm queries and keys and scale the logits by one learned
   temperature (init 10, clamped at 100). Selectivity then stops depending on
   projection magnitude, which is what makes the block behave the same at any
   learning rate and at any T.

At initialisation, on real patches:

| variant | effective frames | weight spread |
|---|---|---|
| current code | 10.97 / 11 | 1.6e-2 |
| contrast only | 10.27 / 11 | 9.9e-2 |
| qk_norm only | 9.74 / 11 | 1.3e-1 |
| **both** | **4.43 / 11** | **4.8e-1** |

And through training, where the unfixed block degenerates in both directions:

| lr | baseline eff@400 | fixed eff@400 | baseline failure mode |
|---|---|---|---|
| 1e-6 (production) | 7.62 | **2.85** | drifting to uniform |
| 1e-5 | 1.13 | **3.58** | saturated one-hot |
| 1e-4 | 1.00 | **4.64** | saturated one-hot |

The baseline degenerates at **every** learning rate — it merely picks a different
degenerate state depending on how fast the projections grow. The fixed block stays
in a healthy 2.85–4.64 range across a 100x span of learning rate, which is exactly
what decoupling the logit scale from projection magnitude is supposed to buy.

At the production learning rate the fixed block *sharpens* to ~3 of 11 frames while
reaching the same loss as the averaging baseline (0.6167 vs 0.6169). It is
selecting, and selection costs nothing.

### What the fix did in a real run

**Measured 2026-08-20** on the five `attnfix` runs of 2026-08-19, the first
trained with it. Probed on their own val split at the depth they were trained
at, none of them is the old collapse — but only two of the four fixed arms are
convincingly selective, and one fails the alarm outright.

| run | effective frames | of uniform | `--require_selectivity 0.9` |
|---|---|---|---|
| `geo_k5_tattn_ring3` | 4.540 / 6 | 75.7% | pass |
| `geo_k10_tattn_ring3` | 8.672 / 11 | 78.8% | pass |
| `geo_k10_tattn_hybrid_ring3` | 9.826 / 11 | 89.3% | pass, by 0.7 points |
| `temporal_k5_tattn_ring3` | 5.566 / 6 | **92.8%** | **COLLAPSED** |
| `clean_geo_k10_tattn_prefix_ring3` (control) | 11.000 / 11 | 100.0% | COLLAPSED, as designed |

The paired control is what makes the table readable. `prefix` is the same
commit, the same data, the same hyper-parameters and `CONTRAST=no QK_NORM=no`,
and it lands on exactly 1/T to five decimals like the fifteen checkpoints before
it. **Every departure from uniform above is attributable to the fix and to
nothing else in the batch.** Per-offset weights and summaries are under
`outputs/attention_probe/<run>/`.

#### Training erodes the selectivity it is given

The number to read that table against is not uniform. It is what the *untrained*
fixed model already does — same architecture, same patches, zero gradient steps:

| on the 272 `geo_k10_clean` val patches | effective frames | of uniform |
|---|---|---|
| fixed architecture, **untrained** | 5.462 / 11 | 49.7% |
| fixed architecture, after 74 epochs | 8.672 / 11 | 78.8% |
| pre-fix architecture, **untrained** | 10.974 / 11 | 99.8% |
| pre-fix architecture, after 86 epochs | 11.000 / 11 | 100.0% |

Training moved the fixed block **58% of the way back toward uniform**. The
selectivity these checkpoints have is mostly what initialisation handed them.
The pressure documented under [The cause](#the-cause) is still there and still
gaining ground — contrast and qk_norm buy a far better starting point and a much
slower slide, not a different dynamic.

#### The learned temperature is what decays now

`qk_norm` took the logit scale off q/k magnitude and put it on one parameter, so
that parameter is where the decay now shows up:

| run | `logit_scale.exp()` at `best.pt` | at `last.pt` | sharpest still reachable |
|---|---|---|---|
| initialisation (every run) | 10.000 | — | ~1.0 / T |
| `geo_k5_fixed` | 1.287 | 1.311 | 2.81 / 6 |
| `temporal_k5_fixed` | 1.170 | 1.188 | 3.17 / 6 |
| `geo_k10_fixed` | 1.280 | 1.164 | 5.41 / 11 |
| `geo_k10_hybrid_fixed` | 0.999 | 1.000 | 7.44 / 11 |

"Sharpest still reachable" is the entropy of the most peaked softmax that
temperature permits at all — one frame at cosine +1 and every other at −1. At
1.280 a *perfectly* aligned set of queries and keys could not get below 5.41 of
11 frames; at 0.999 it could not get below 7.44. The hybrid clears the 90% alarm
by 0.7 points because its temperature no longer allows anything sharper, and
that arm should be read as still averaging.

`weight_decay` is 1e-8 (`training/train.py:658`), far too small to account for
the travel from `log 10` to ~`log 1.2`, so this is gradient-driven. Note also
that `geo_k10`'s temperature was still falling when `best.pt` was written — 1.280
at epoch 34, 1.164 by epoch 74 — so the kept checkpoint is not a settled state.

#### The queries and keys did learn; the temperature hides it

Reset `logit_scale` to 10 on a trained checkpoint and change nothing else:

| `geo_k10_tattn_ring3` | effective frames | weight range | argmax share at offset 2 |
|---|---|---|---|
| as trained (temperature 1.280) | 8.672 / 11 | 0.69–1.31 x | 25.4276% |
| same weights, temperature 10 | **4.194 / 11** | 0.39–1.84 x | 25.4276% |

The argmax shares are bit-identical at every offset — argmax does not depend on
temperature — so the q/k directions are untouched and only the sharpness moved.
Those directions encode a real, structured temporal preference, and at the
initialisation temperature this checkpoint sits at 4.19/11, inside the 2.85–4.64
band the bench predicted. **The mechanism works. The temperature throttles it.**

The same test separates `temporal_k5`, the arm that fails the alarm, from the
rest: at temperature 10 it reaches only 4.457/6 (74.3%), over a weight range of
0.73–1.20x against `geo_k10`'s 0.39–1.84x. Its queries and keys really are close
to uninformative, so a temperature floor alone would not rescue that arm.

#### Where the weight landed in August

Every fixed run **down-weights the current frame** and prefers older ones. With
`FUSE_SKIPS=0` the decoder already receives the present frame through the skips,
so that is a sensible division of labour rather than a defect.

| run | weight on the current frame | shape over offset |
|---|---|---|
| `geo_k5_fixed` | 0.79 x | rises monotonically to 1.32x at the oldest frame |
| `geo_k10_fixed` | 0.84 x | hump at offsets 2–4 (22–44 days), peak 1.31x |
| `geo_k10_hybrid_fixed` | 0.70 x | rises monotonically to 1.48x at the oldest frame |
| `temporal_k5_fixed` | 0.88 x | nearly flat; 1.07x at its highest |

#### Long histories are no better supported than before

Run at a 40-slot lookback the fixed runs hold their selectivity ratio — 79.2% at
32.7 frames against 78.8% at 11 — which is the T-independence `qk_norm` was
supposed to buy, and which the pre-fix control cannot manage at any length.

They do not, however, *reach back*. The exactly-uniform control puts 0.641 of its
mass beyond the trained horizon on this patch set, so 0.641 is what no depth
preference at all looks like here. `geo_k10_fixed` puts 0.629 there and the
hybrid 0.664 — a couple of points either side of nothing. Ranking depths still
needs a model trained at depth, and §4 below is unchanged.

### What it means for long histories

The probe's depth sweep on the *averaging* model showed accuracy falling away with
depth — dice 0.649 at 6 frames, 0.596 at 33, with precision collapsing 0.619 →
0.474 while recall climbed. Averaging a 14-month window does not suppress noise, it
smears a signal that grows and migrates.

That measurement says nothing about a model that selects, which is the whole point
of fixing this first. The long-history machinery is already in place — real offsets
instead of list positions, and a padding mask, so a gappy 40-slot lookback keeps
all 273 interferograms where the strict chain rule keeps 20 (`meta.select_history`).
What was missing was a mechanism able to use it.

**Measured 2026-08-20:** that mechanism now exists and holds its selectivity at
40 slots as designed, but it shows no preference for the far history — it spreads
the same mild weighting wider rather than reaching back. Nothing there is settled
by a model trained at k=10 and merely *run* at 40; see
[Long histories are no better supported than before](#long-histories-are-no-better-supported-than-before).

#### Trained fresh at a 41-slot depth

Both variants trained from scratch on full 41/41 hole-tolerant histories (10 train
and 6 val interferograms chosen for having a complete 40-slot history, 76/48
patches, 500 steps, lr 1e-5) — so neither is being run outside the regime it was
trained in, which is the confound that made the probe's depth sweep suggestive
rather than conclusive:

| variant | val dice | precision | recall | effective frames |
|---|---|---|---|---|
| fixed | 0.5524 | 0.584 | 0.524 | **9.08 / 41** |
| baseline | 0.5249 | 0.650 | 0.440 | **1.05 / 41** |

The mechanism claim is clean: at 41 frames the unfixed block degenerates to a
single frame (1.05/41 — at this learning rate it saturates rather than averages),
while the fixed one spreads over ~9 of 41 with structured weighting — offsets 2-4
and a peak near offset 20, with the far tail suppressed.

**The accuracy claim is not clean and should not be quoted.** 500 steps on 76
patches leaves both models far from converged (the fixed one was still at dice
0.07 at step 250), so the 0.5524 vs 0.5249 gap ranks nothing. What this run
establishes is that a long, gappy history is expressible end to end and that the
fixed attention stays selective at that length. Ranking depths needs a cluster run.

### What was done next, in August

**1. ~~Retrain the tattn family with the fix.~~ DONE** — the five `attnfix`
runs of 2026-08-19, including the paired `prefix` control. To train the old
architecture deliberately, set `CONTRAST=no QK_NORM=no`.

**2. ~~Check the result actually selects.~~ DONE 2026-08-20**, and it is the
reason for the section above. The command, now under [Running it](#running-it):

```bash
sinkholes attention-probe --model outputs/<run>/checkpoints/best.pt \
  --partition assets/partition_geo_k10_clean.json --split val \
  --patches_dir "$DATA/patches" --control_lookback 10 --require_selectivity 0.9
```

`--require_selectivity` exits non-zero if the attention is at or above 90% of
uniform. Still worth wiring into the eval scripts: it caught
`temporal_k5_tattn_ring3`, and it can only be made against a
*trained* checkpoint — a fresh model passes every content-sensitivity test and
still dies. Note that 90% is a **collapse alarm, not a pass mark**: the hybrid
clears it by 0.7 points while averaging, so read the number, not the exit code.

**3. Put a floor under the temperature, before spending the next batch.**
*Not needed under AdamW — see the 2026-09-16 note below.* At the time this
was the open item, and it blocks the `attnpos` batch
(`scripts/submit_all.sh:968`) — five re-runs, ~30 GPU-hours, which as configured
will reproduce exactly the decay measured above. The evidence says the queries
and keys are learning something real and one scalar is flattening it, so the
cheap experiment is to stop `logit_scale` from travelling: clamp it below (it is
already clamped above at `MAX_LOGIT_SCALE`), freeze it at its initial 10, or put
it in its own optimiser group at a much lower learning rate. `geo_k10_fixed`
read at temperature 10 sits at 4.19/11, so the upside is roughly the difference
between 79% and 38% of uniform. Any of the three is a few lines in
`models/temporal_attention.py` plus a `STRICT_CONFIG_KEYS` entry, and it needs
one paired run to settle — not five.

Whatever the temperature does, `temporal_k5_tattn_ring3` needs its
own answer: it is the one arm whose q/k are genuinely uninformative, and a
floor would not rescue it.

*2026-09-16:* the ctx50 AdamW runs did **not** show this decay: the temperature
was 9.68 and 10.22 at `best.pt`, and still 9.05 at th350's epoch-60 `last.pt`.
Under that recipe a floor looks unnecessary. Whether RMSprop runs still need
one is untested, since the optimiser was not changed in isolation.

### Why no test caught this

`tests/test_tattn_unet.py::test_attention_is_content_based` exists for exactly this
and passes, because it runs on a **randomly initialised** model and asks only
whether the weights *move*. They do — by 1e-5, around a uniform mean. The property
was true at init and false after training, and "different" is not the same claim as
"selective".

`tests/test_attention_selectivity.py` replaces that with assertions on *how much*
selectivity there is, including one that pins the broken baseline so the fix cannot
silently stop being a fix.

**That is still not enough, and the 2026-08-20 measurement is why.** Those
assertions also run at initialisation —
`test_fixed_attention_is_selective_at_initialisation` requires the fixed block to
use fewer than `0.75 * T` frames, and an untrained model clears it comfortably at
49.7% of uniform. Every trained checkpoint in this batch then finished *above*
that same threshold (75.7–92.8%), with the one at 92.8% failing the collapse
alarm outright. The suite proves the architecture can select on the day it is
built; only `--require_selectivity` against a real checkpoint proves it still
does at the end of a run. Neither claim substitutes for the other.

## Reference

### Compatibility

Both options add parameters (`contrast_norm`, `logit_scale`), so a checkpoint's
weights say unambiguously which variant it is. `infer_attention_variant()` reads
that off the state dict *before* the config blob is applied, so every run trained
before this date rebuilds as the model it actually is rather than as today's
default. `sinkholes eval-scenes`, `test-patches` and `predict` need no flags.

- New models: both on.
- Pre-fix checkpoints: both off, detected from the weights, loaded strictly.
- `--no-tattn_contrast --no-tattn_qk_norm` reproduces the old architecture exactly.
- Both are in `STRICT_CONFIG_KEYS`, so a resume cannot splice the two together.

### The code

- `models/temporal_attention.py` — `temporal_mean()`; `contrast`/`qk_norm` on both
  attention blocks; `temporal_position_encoding(..., offsets=)` for real frame ages;
  `valid=` padding masks.
- `models/tattn_unet.py` — `tattn_contrast`/`tattn_qk_norm` through the config and
  checkpoint round-trip; `infer_attention_variant()`; `forward(x, offsets, valid)`;
  the hybrid's ConvLSTM holds state across a masked step.
- `meta.py` — `select_history()` (holes allowed), `parse_history_schedule()`.
- `inference/attention_probe.py` — the diagnostic, as `sinkholes attention-probe`.
- `tests/test_attention_selectivity.py`, `tests/test_history_holes.py`.

Padding must replicate a real frame, never zeros: padded timesteps still pass
through the shared encoder, and `DoubleConv`'s BatchNorm mixes them into the batch
statistics before `valid` can hide them.
