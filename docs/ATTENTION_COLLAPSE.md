# The temporal attention collapsed to a uniform average — cause and fix

**Status:** diagnosed and fixed 2026-08-19. The fix is on by default for new runs
(`--tattn_contrast`, `--tattn_qk_norm`); every existing checkpoint still loads and
behaves exactly as before. Reproduce the measurements with `sinkholes attention-probe`.

## Summary

Every trained `tattn_unet` checkpoint in `outputs/` — **15 of 15** — produced
attention weights that were *exactly* `1/T` at every timestep, head and bottleneck
pixel. The temporal attention was not selecting frames; it was computing an
unweighted mean, and the ~1M attention parameters were decoration.

The cause is **not** a training pathology that develops over a run. It is present
before the first gradient step: at initialisation the tokens attention sees are
99.91% a component shared across the whole sequence, so there is nothing to select
on. Two changes fix it, and neither is sufficient alone.

This does not invalidate the results: `clean_geo_k10_tattn_ring3` scores 0.727
object F1 and still beats its ConvLSTM twin. But the mechanism those results were
attributed to was not the mechanism running, and any experiment premised on
attention *choosing* frames — long or hole-tolerant histories above all — was
untestable until now.

## The measurement

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

## The cause

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

## The fix

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

## Compatibility

Both options add parameters (`contrast_norm`, `logit_scale`), so a checkpoint's
weights say unambiguously which variant it is. `infer_attention_variant()` reads
that off the state dict *before* the config blob is applied, so every run trained
before this date rebuilds as the model it actually is rather than as today's
default. `sinkholes eval-scenes`, `test-patches` and `predict` need no flags.

- New models: both on.
- Pre-fix checkpoints: both off, detected from the weights, loaded strictly.
- `--no-tattn_contrast --no-tattn_qk_norm` reproduces the old architecture exactly.
- Both are in `STRICT_CONFIG_KEYS`, so a resume cannot splice the two together.

## What it means for long histories

The probe's depth sweep on the *averaging* model showed accuracy falling away with
depth — dice 0.649 at 6 frames, 0.596 at 33, with precision collapsing 0.619 →
0.474 while recall climbed. Averaging a 14-month window does not suppress noise, it
smears a signal that grows and migrates.

That measurement says nothing about a model that selects, which is the whole point
of fixing this first. The long-history machinery is already in place — real offsets
instead of list positions, and a padding mask, so a gappy 40-slot lookback keeps
all 273 interferograms where the strict chain rule keeps 20 (`meta.select_history`).
What was missing was a mechanism able to use it.

### Trained fresh at a 41-slot depth

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

## What to run next

**1. Retrain the tattn family with the fix.** This is a drop-in change — same
partitions, same data layer, same flags, and `CONTRAST`/`QK_NORM` default to `yes`:

```bash
PARTITION=assets/partition_geo_k10_clean.json K_PREVS=10 RING_NEGS=yes \
  bsub -J tattn_geo_k10_fixed_ring3 < scripts/train/train_tattn.sh
```

The comparison that matters is against the pre-fix twin at the same depth, since
that isolates "what did working selection buy" from every other difference. To
train the old architecture deliberately, set `CONTRAST=no QK_NORM=no`.

**2. Check the result actually selects**, rather than assuming it:

```bash
sinkholes attention-probe --model outputs/<run>/checkpoints/best.pt \
  --partition assets/partition_geo_k10_clean.json --split val \
  --patches_dir "$DATA/patches" --control_lookback 10 --require_selectivity 0.9
```

`--require_selectivity` exits non-zero if the attention is at or above 90% of
uniform. Worth wiring into the eval scripts: this is the check that would have
caught the original collapse, and it can only be made against a *trained*
checkpoint — a fresh model passes every content-sensitivity test and still dies.

**3. Only then, the long-history work.** `select_history()` and the mask make a
40-slot gappy lookback expressible, and the model now has a mechanism that can use
it, but the *data layer* still materialises `(T, N, H, W)` per interferogram
(`dataprep/dataset.py:411`) and reloads each grid once per referencing current.
That is ~31 GB of host RAM at T=6 and linear in T, so a dense long lookback needs
the deduplicated frame store first: resolve every sample's coordinates, load each
distinct grid once, and have samples point at rows. Memory then stops growing with
depth, because sinkholes sit still and the per-frame coordinate union saturates.

## Running it

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
availability), `summary.json` and `dice_vs_depth.csv`.

## Why no test caught this

`tests/test_tattn_unet.py::test_attention_is_content_based` exists for exactly this
and passes, because it runs on a **randomly initialised** model and asks only
whether the weights *move*. They do — by 1e-5, around a uniform mean. The property
was true at init and false after training, and "different" is not the same claim as
"selective".

`tests/test_attention_selectivity.py` replaces that with assertions on *how much*
selectivity there is, including one that pins the broken baseline so the fix cannot
silently stop being a fix.

## The code

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
