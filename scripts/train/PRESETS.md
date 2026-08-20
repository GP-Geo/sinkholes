# Training presets

Exact settings for every run trained so far, plus the four geo reference runs
that were planned but never submitted. Each row is the **delta from the
template defaults** — copy it into the CONFIG block and submit.

Results and verdicts live in `docs/MODEL_RUNS.md`; this file is only "how do I
reproduce it".

## Template defaults

Both templates ship with these, so an empty delta means "submit as-is":

    PARTITION=assets/partition_geo_k10.json   K_PREVS=10   HIDDEN=256
    POS_W=8   SEED=42   LR=1e-6   SCHEDULE=plateau
    EPOCHS=60   BATCH=128   PATIENCE=20
    RING_NEGS=no   NEG_RING_INNER=1   NEG_RING_OUTER=3   NEG_PER_POS=1.0
    VAL_NEGS=no
    RESUME=auto

`VAL_NEGS` is **deprecated as of 2026-08-20 and must stay `no`**. Validation is
positives-only for every run from here on; see
[Negative validation, and why it is gone](#negative-validation-and-why-it-is-gone).
The presets below that still list `VAL_NEGS=yes` are the historical record of
what those runs were trained with — they are no longer reproducible as written
and are not meant to be resubmitted.

Plus, fixed in both templates and not normally changed: `--patch_size 200 100`,
`--stride 2`, `--partition_mode preset_by_intf`, `--amp`, `--save_best_only`,
`--resume auto`.

## `train_convlstm.sh`

| Job name (`#BSUB -J`) | CONFIG delta | Run |
|---|---|---|
| `convlstm_geo_k10_h256_b128_lr1e6_60e` | *(defaults)* | `2026-08-06/geo_k10_convlstm_posw8` |
| `convlstm_geo_k5_h256_b128_lr1e6_60e` | `PARTITION=…geo_k5.json` `K_PREVS=5` | `2026-08-06/geo_k5_convlstm_posw8` |
| `convlstm_temporal_k10_h256_b128_lr1e6_60e` | `PARTITION=…temporal_k10.json` | `2026-08-06/temporal_k10_convlstm_posw8` |
| `convlstm_temporal_k5_h256_b128_lr1e6_60e` | `PARTITION=…temporal_k5.json` `K_PREVS=5` | `2026-08-06/temporal_k5_convlstm_run2` (and `_run1`) |
| `convlstm_temporal_k10_h512_b128_lr1e6_60e` | `PARTITION=…temporal_k10.json` `HIDDEN=512` | `2026-08-06/temporal_k10_convlstm_h512` |
| `convlstm_temporal_k10_h1024_b128_lr1e6_60e` | `PARTITION=…temporal_k10.json` `HIDDEN=1024` | `2026-08-06/temporal_k10_convlstm_h1024` |
| `convlstm_temporal_k10_h256_b128_lr1e6_posw4_60e` | `PARTITION=…temporal_k10.json` `POS_W=4` | `2026-08-06/temporal_k10_convlstm_posw4` |
| `convlstm_temporal_k10_h256_b128_lr1e6_seed7_60e` | `PARTITION=…temporal_k10.json` `SEED=7` | `2026-08-06/temporal_k10_convlstm_seed7` |
| `convlstm_temporal_k10_h256_b128_lr5e6_cosine_60e` | `PARTITION=…temporal_k10.json` `LR=5e-6` `SCHEDULE=cosine` | `2026-08-06/temporal_k10_convlstm_cosine` |

## `train_control.sh`

| Job name (`#BSUB -J`) | CONFIG delta | Run |
|---|---|---|
| `unet_stack_temporal_k10_b128_lr1e6_60e` | `ARCH=stack` `PARTITION=…temporal_k10.json` | `2026-08-06/temporal_k10_stack` |
| `geo_k5_single_b64` | `ARCH=single` `PARTITION=…geo_k5.json` `BATCH=64` `LR=1e-5` | `2026-08-05/geo_k5_single_b64` |
| `baseline_single_temporal_k5partition_b64_lr1e5_60e` | `ARCH=single` `PARTITION=…temporal_k5.json` `BATCH=64` `LR=1e-5` | `2026-08-05/temporal_k5_single_b64` |

The two 2026-08-05 baselines ran at `b64`/`lr1e-5` while every ConvLSTM ran at
`b128`/`lr1e-6`. That confounds architecture with optimizer — the +0.046
"temporal context" gain is measured across that difference. Preset 11 below
exists to settle it.

## Planned: geo reference runs (never submitted)

Group G10 has exactly one occupant, so `geo_k10_convlstm_posw8`'s
0.6583 has nothing to be measured against — it is the highest number in
`outputs/` and also the only member of its comparison group. These four give it
a group. Submit 12 first: without an error bar nothing else here is
interpretable.

| Order | Template | Job name | CONFIG delta | Why |
|---|---|---|---|---|
| **1st** | convlstm | `convlstm_geo_k10_h256_b128_lr1e6_seed7_60e` | `SEED=7` | **The error bar.** Byte-for-byte the 206375 run but a different seed. On the temporal partition the same experiment spread 0.0059; geo_k10's val set is under half the size (17 intfs / 4060 samples vs 30 / 9368) so expect *wider*. If the spread exceeds ~0.009, the 0.0044 gap to `geo-k5_convlstm-h256` is noise. |
| 2nd | control | `baseline_single_geo_k10_b128_lr1e6_60e` | `ARCH=single` | The temporal-gain anchor, at matched `b128`/`lr1e-6` — de-confounds the optimizer. **Caveat:** lr 1e-6 was tuned for the ConvLSTM and may undertrain a single-frame U-Net; check val/dice near epoch 10 against the 2026-08-05 baseline curve and rerun at 1e-5 if it lags. |
| 3rd | control | `unet_stack_geo_k10_b128_lr1e6_60e` | `ARCH=stack` | Recurrence vs channel-stacking under a north→south domain shift. Worth +0.0143 on the temporal split; unknown whether it survives the shift. |
| 4th | convlstm | `convlstm_geo_k10_h256_b128_lr1e6_posw4_60e` | `POS_W=4` | Operating point. `pos_w` 8→4 topped the temporal group and moved precision 0.659→0.671. Geo models already sit at higher precision (0.707), so it may behave differently. |

## The negative-sampling batch (submitted 2026-08-10)

Every run above trained on positive patches only — `--nonz_only` is the code
default — which is ~2.5% of the patch grid (38.3 K of ~1.56 M for geo_k10,
38.7 K of ~1.81 M for temporal_k5). The model is fit on the fraction of the map
that contains subsidence and then applied to all of it, which is what the
object-level evals exposed: `geo_k10` scores **P=0.12** at confidence 0.125.
`sinkholes/inference/outputs.py:22-25` predicted exactly this.

`RING_NEGS=yes` adds all-zero patches from an annulus around the positives
(`dataset.py:69-97`). Candidates must be empty at *every* timestep, so a patch
that was positive last month is never used as a negative.

Nine runs, all at `POS_W=4` — the value the object-level evals picked out
(`pos_w` 8→4 was worth +0.023 F1 on 18 of 20 test scenes). Every negatives run
differs from its baseline in the negative sampling and nothing else.

| Job name | CONFIG delta | Role |
|---|---|---|
| `convlstm_temporal_k5_h256_posw4_60e` | `PARTITION=…temporal_k5.json` `K_PREVS=5` `POS_W=4` | **baseline** |
| `convlstm_temporal_k5_h256_posw4_neg1x_ring3_60e` | + `RING_NEGS=yes` | 1:1, near field |
| `convlstm_temporal_k5_h256_posw4_neg1x_ring10_60e` | + `RING_NEGS=yes` `NEG_RING_OUTER=10` | 1:1, far field |
| `convlstm_temporal_k5_h256_posw4_neg3x_ring3_60e` | + `RING_NEGS=yes` `NEG_PER_POS=3.0` | 3:1, near field |
| `convlstm_geo_k5_h256_posw4_60e` | `PARTITION=…geo_k5.json` `K_PREVS=5` `POS_W=4` | **baseline** |
| `convlstm_geo_k5_h256_posw4_neg1x_ring3_60e` | + `RING_NEGS=yes` | 1:1, near field |
| `convlstm_geo_k5_h256_posw4_neg1x_ring10_60e` | + `RING_NEGS=yes` `NEG_RING_OUTER=10` | 1:1, far field |
| `convlstm_geo_k5_h256_posw4_neg3x_ring3_60e` | + `RING_NEGS=yes` `NEG_PER_POS=3.0` | 3:1, near field |
| `convlstm_geo_k10_h256_posw4_neg1x_ring3_60e` | `PARTITION=…geo_k10.json` `K_PREVS=10` `POS_W=4` `RING_NEGS=yes` | 1:1, near field |

**Submit the two baselines first.** Pinning the batch at `pos_w` 4 invalidated
the previously-trained references — they sit at `pos_w` 2 (`temporal_k5_convlstm_posw2`,
0.6496) and `pos_w` 8 (`geo_k5_convlstm_posw8`, 0.6539). Until the two
baselines exist, nothing in those groups is interpretable. They are also the
cheapest jobs in the batch.

`geo_k10` needs no new baseline: `geo_k10_convlstm_posw4` was
trained at `pos_w` 4 on 2026-08-09 and holds the highest val dice in the
project (0.6646).

**`val/dice` cannot score these runs.** Ring negatives reach the train split
only; val stayed positives-only, so the false positives being suppressed are
mostly outside the validation set. Expect the curve flat or slightly down while
object-level precision improves. Judge them with `scripts/eval/run_eval.sh`.

That is a fact of life again, and the answer is `run_eval.sh`, not a padded
validation set — see the next section.

## Negative validation, and why it is gone

**Removed 2026-08-20.** `VAL_NEGS=yes` / `--add_val_negatives` is deprecated:
nothing in `scripts/submit_all.sh` sets it, no new run should, and the trainer
warns if it is passed. The flag still exists **only** so runs already trained
with it stay resumable — it is part of the strict `dataset` resume fingerprint
(`sinkholes/training/resume.py`), and the five `attnfix` runs of 2026-08-19
carry `valneg=1-3x1.0` in theirs. It goes for good once those land.

Why it was dropped: `dice_coeff` maps an empty prediction on an empty mask to
`(0+eps)/(0+eps) = 1.0`, so at 1:1 about half the validation samples score ~1.0
and the mean is roughly `(1 + dice_on_positives)/2`. That is the whole reason
the `clean22` batch reads 0.72–0.80 where every earlier batch reads 0.63–0.66.
The curve moved; the ranking did not. And `val/F1`, `val/P` and `val/R` are
pooled from raw pixel counts (`sinkholes/training/evaluate.py:300-306`) rather
than averaged per sample, so they were **already** negative-aware — the padding
bought nothing they did not already give.

What replaces it: keep background patches in **training** (`RING_NEGS=yes`),
read `val/F1` rather than `val/dice` for training health, and settle every
precision claim at object level with `scripts/eval/run_eval.sh`.

What it costs, honestly: `best.pt` is still selected on `val/dice`, so
checkpoint selection is once again blind to false positives on background and
will lean recall-heavy. That is a real regression against what this flag was
introduced to fix, and the object-level eval is what has to absorb it.

### The reruns it produced (historical)

The batch above was measured on 5,846 purely positive validation patches, so
`val/dice` was structurally blind to the false positives ring negatives exist to
suppress. All nine arms landed in 0.639–0.660 — inside the ±0.006 noise floor —
while object-level F1 on the same models moved **+0.096**. Every verdict has had
to come from `run_eval.sh` at ~3 h per scene list.

`VAL_NEGS=yes` (`--add_val_negatives`) puts negatives in the **validation** set
too, so a false positive on background costs dice. The configuration is **fixed
in the code**, not a knob: ring 1..3, 1:1, drawn once from the validation
interferograms with `SEED`, identical for every epoch. Partition and seed alone
decide the samples, so every run below is scored on the same ~11,700-patch set
whatever its architecture or training-negative ratio. `val/dice` keeps driving
the plateau schedule, early stopping and `best.pt` — only the set underneath it
grew.

These are **reruns, not rescores**. `val/dice` selects `best.pt`; a model whose
checkpoints were chosen on a positives-only curve is not the model this set
would have picked, so re-scoring the old weights would answer a different
question.

Everything except the validation set is copied from the run each one replaces —
verified against each run's own banner, not from memory. **These are no longer
submittable**: the `valneg` batch is retired and `bash scripts/submit_all.sh
valneg` now exits with an error. The table is kept as the record of what the
runs in `outputs/` were trained with.

| Order | Template | Job name | CONFIG delta | Replaces (old dice) |
|---|---|---|---|---|
| **1st** | convlstm | `convlstm_geo_k5_h256_posw4_neg1x_ring3_valneg1x_60e` | `PARTITION=…geo_k5.json` `K_PREVS=5` `POS_W=4` `RING_NEGS=yes` `VAL_NEGS=yes` | `…neg1x_ring3_60e` (0.6541 @30) |
| 2nd | convlstm | `convlstm_geo_k5_h256_posw4_neg3x_ring3_valneg1x_60e` | + `NEG_PER_POS=3.0` | `…neg3x_ring3_60e` (0.6386 @30) |
| 3rd | convlstm | `convlstm_geo_k5_h256_posw4_neg1x_ring10_valneg1x_60e` | + `NEG_RING_OUTER=10` | `…neg1x_ring10_60e` (0.6461 @24) |
| 4th | tattn | `tattn_geo_k5_d256_fuse0_posw4_neg1x_ring3_valneg1x_60e` | `PARTITION=…geo_k5.json` `K_PREVS=5` `POS_W=4` `FUSE_SKIPS=0` `RING_NEGS=yes` `VAL_NEGS=yes` | `tattn_geo_k5_…_neg1x_ring3_60e` (0.6433 @16) |
| 5th | control | `unet_single_geo_k5_posw4_neg1x_ring3_valneg1x_60e` | `ARCH=single` `PARTITION=…geo_k5.json` `K_PREVS=5` `POS_W=4` `RING_NEGS=yes` `VAL_NEGS=yes` | `unet_single_geo_k5_…_neg1x_ring3_60e` (0.6341 @20) |

**Do not read a rerun against the number beside it.** A positives-only val set
and a 1:1 one are different scales; the old five span 0.0200 and are precisely
the figures that cannot rank anything. Compare the reruns with each other.

Walltime rises because validation runs at batch size 1, so doubling the val set
costs more than its sample count suggests — decomposed from the two single-frame
runs (identical but for training samples), validation was ~80 s of a 131 s
epoch. Expect roughly +52 s/epoch on a ConvLSTM arm and +80 s on the
single-frame one; `submit_all.sh` carries the arithmetic and the per-job
requests. The only rung that changed is the attention arm's, 20:00 → 12:00: the
20:00 rested on an estimate of 14m43s/epoch that the run itself disproved at
5m46s.

## Planned: temporal attention (`train_tattn.sh`, never submitted)

`TemporalAttentionUNet` replaces the ConvLSTM bottleneck with multi-head
attention over the T axis — the current interferogram queries its own history at
each of the 72 bottleneck locations — and can reuse those weights to fuse the
skip connections over time instead of taking them from the latest frame alone.
It is **26% smaller** than the ConvLSTM (32.1 M vs 43.1 M params; the attention
block is 1.05 M against the cell's 11.8 M), which is a feature given that
`HIDDEN` 256→512→1024 showed capacity is already saturated.

Template defaults: `PARTITION=…temporal_k5.json K_PREVS=5 POS_W=4 DIM=0
HEADS=8 LAYERS=1 RECURRENCE=none FUSE_SKIPS=0`, everything else as
`train_convlstm.sh`. The reference these are read against is
`convlstm_temporal_k5_h256_posw4_60e` — same partition, same `pos_w`, same
optimizer.

Submit with `bash scripts/submit_all.sh tattn --submit`. They sit under their
own kind so they do not ride along with `submit_all.sh train`.

| Order | Job name | CONFIG delta | Why |
|---|---|---|---|
| **1st** | `tattn_temporal_k5_d256_fuse0_posw4_60e` | *(defaults)* | **The head-to-head.** `FUSE_SKIPS=0` is byte-for-byte the ConvLSTM's skip contract, so the only difference from `convlstm_temporal_k5_h256_posw4_60e` is attention vs recurrence at the bottleneck. Landing inside the noise floor is still a result: a 26%-smaller model matching it means the recurrence was compressing, not looking up. |
| 2nd | `tattn_temporal_k5_d256_fuse0_posw4_neg1x_ring3_60e` | + `RING_NEGS=yes` | **The pairing that matters.** Same head-to-head, but under the negative sampling that actually moves scene-scale precision — against `convlstm_temporal_k5_h256_posw4_neg1x_ring3_60e`. Architecture and negatives are separable only if both this and run 1 exist. |
| 3rd | `tattn_temporal_k5_d256_fuse2_posw4_neg1x_ring3_60e` | `FUSE_SKIPS=2` + `RING_NEGS=yes` | Isolates the temporally-fused skips as a clean +1 delta on run 2. `s4`/`s3` only, where the 12×6 attention field is upsampled 2× and 4× rather than 16×. |
| 4th | `tattn_temporal_k5_d256_fuse0_recur-convlstm_posw4_neg1x_ring3_60e` | `RECURRENCE=convlstm` + `RING_NEGS=yes` | The hybrid: the ConvLSTM runs first and attention reads over **all** its hidden states rather than only the last. Answers whether the two mechanisms compound or are redundant. |
| later | `tattn_temporal_k5_d256_fuse4_…` | `FUSE_SKIPS=4` | Only if 3 beats 2. Fusing `s1` at 200×100 stretches the attention field 16×, and since an 11-day interferogram measures *rate*, averaging the fine skip can import stale signatures from frames where a now-quiescent sinkhole was still moving. Needs `gmem=48G`. |

**These runs cannot be scored on `val/dice` alone.** The top six models already
span 0.0069 against a ±0.006 noise floor. The mechanism being tested — artefacts
are temporally inconsistent, subsidence is persistent, so a weighted average
over the chain suppresses one and keeps the other — is a *precision* argument,
and precision at scene scale is what `scripts/eval/run_eval.sh` measures. Budget
the eval, and three seeds if you want to claim a dice win.

**The cheapest decisive control is not in the table**: replace the content-based
Q·Kᵀ with a bare learned per-offset logit vector (T parameters), softmax over
time, weighted-average the bottleneck. If that matches run 1, the 1.05 M
attention parameters are decoration; if it matches the *ConvLSTM*, 11.8 M
parameters of recurrence have been buying a learned average.
`tests/test_tattn_unet.py::test_attention_is_content_based` is the unit-level
version of the same question.

## Notes

- **`K_PREVS` must match the partition.** `partition_*_k5.json` lists only
  interferograms with 5-previous chains, `_k10` only those with 10 — mixing
  them trains on a smaller set than intended. Both templates assert this.
- **Capacity is saturated at `HIDDEN=256`.** h512 was flat, h1024 was worse and
  early-stopped undertrained. Raising it also inflates the checkpoint (165 MB →
  407 MB) and the resume state.
- **`SCHEDULE=cosine` at `LR=5e-6` was the worst ConvLSTM run** and showed 5×
  the epoch-to-epoch jitter of any other. Treat that combination as ruled out.
- **`POS_W`'s code default is 1** (`sinkholes/training/train.py:95`); every run
  so far used 8, which is what drives the recall skew across the whole model
  set (R 0.79–0.90 vs P 0.54–0.67). At object level `pos_w` 8→4 was worth
  **+0.023 F1 on 18 of 20 test scenes** — the best-established result here.
- **The VRAM ladder dropped to 36G / 48G on 2026-08-10.** It was 48/64, which
  had never been measured — LSF reports host memory but no gmem — and on
  2026-08-10 seven jobs sat `PEND` behind it. Over-requesting VRAM excludes
  whole classes of card: a 48G request cannot land on a 40G A100 at all, so the
  job waits for a bigger GPU instead of running on an idle one. The new numbers
  are an *estimate* (k5 ~24 GiB, k10 ~40 GiB peak; see the arithmetic in
  `scripts/submit_all.sh`), and every run now prints its own peak so the ladder
  can become measured: `grep "peak VRAM" logs/*.out`. **k10 keeps the higher
  rung** — at 36G it would be running below its estimated peak.
- **`--add_nulls_to_train` does nothing on a temporal run.** Null-patch sampling
  lives in the single-frame branch of the dataset builder; temporal data goes
  through `_load_temporal`, which never calls it. `train.py` now warns instead
  of accepting it silently. `RING_NEGS=yes` is the only working route to
  background patches in a ConvLSTM run.
- **`VAL_NEGS` is not a knob and must not become one.** Its radii and ratio are
  constants in `sinkholes/dataprep/dataset.py`; the CLI exposes only on/off. Two
  runs that sweep it are not comparable, which defeats the point of having it.
  It needs `SEED` — `train.py` refuses the combination without one rather than
  drawing a val set that differs per run.
- **`VAL_NEGS=yes` changes what `val/dice` means**, so it is part of the resume
  fingerprint: a checkpoint written with it will not resume without it. Runs
  that leave it off keep the exact fingerprint they had before the option
  existed, so every earlier checkpoint still resumes.
