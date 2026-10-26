# Project summary and results

## The project in brief

**The question.** Sinkhole-induced land subsidence along the Dead Sea shows up in Sentinel-1
interferograms as small, local fringe patterns. The Geological Survey of Israel (GSI) maps it
by hand, one 11-day interferogram at a time. Can a network do that mapping automatically, on
whole scenes?

**The starting point.** The published benchmark, Dekel, Nof, Sarafian & Rudich, *Remote
Sensing* 18:211 ([PDF](reference/remotesensing-18-00211-v2.pdf)): a single-frame U-Net on
200×100 patches of the wrapped phase, with object-level scores above 0.8, measured on the
tiles that contain subsidence. The inherited code (Gali Dekel, 2024 – mid-2026) also had first
versions of a LiDAR coverage gate, ring negatives and temporal input.

**The data.** 437 11-day interferograms, 2019–2026, on two frames (North, South). 273 are
labelled by GSI (30,729 polygons, `sub_20260701.shp`); a LiDAR coverage mask gates where a
prediction can be scored ([`DATASET_11DAY.md`](DATASET_11DAY.md)).

**What the rotation did (late July – October 2026).** Each step is a numbered finding below.

| | step | what changed | finding |
|---|---|---|---|
| 1 | Score the real task | Whole scenes, every tile, object-level F1. The paper's positives-only scoring inflates precision up to 10× and ranked the worst full-scene model first; patch Dice ranked models backwards. | ⑩ ② |
| 2 | Stop the over-prediction | Ring negatives (empty patches around each sinkhole) measured at scene level and made standard at ring 1–10: precision +0.10 (geographic split) and +0.28 (temporal). The largest single gain. | ① |
| 3 | Use the history | The current interferogram plus the *k* before it: a ConvLSTM, then temporal attention, repaired from a collapse to uniform weights. History helps, more at k10 than k5. | ④ ⑨ |
| 4 | Make training correct | Two mixed-precision bugs (fixed 2026-09-07/08) meant earlier runs at batch ≥ 32 optimised BCE alone; the old RMSprop setting then diverged, so every run since uses AdamW 3e-4, weight decay 1e-2. | ③ |
| 5 | Clean the benchmark | AOI windows; the temporal split (train on earlier years, test on later ones, which is how the model would be used) replaced the geographic one on 2026-09-03; a label-quality threshold; history chains that skip a missing acquisition (+0.021). | ⑦ ⑥ |
| 6 | Check the labels | All 20 official test scenes relabelled (366 changes, 46 doubtful polygons flagged), with every change rendered for a GSI review (2026-10-19): the old labels missed objects the models found. | ⑧ |
| 7 | Tune it | A new split with full chain separation (test = 2026) and an Optuna study: plain vs ctx50 temporal attention, then 300-epoch runs. First phase: plain reaches 0.770, ctx50 0.757; at equal precision plain also recalls more. Running. | ⑫ |

Underneath: the inherited scripts became one package with one CLI (`python -m sinkholes`),
625 tests and an LSF workflow for WEXAC.

**Where it stands (2026-10-25).** The best model is plain temporal attention over the current
interferogram and the ten before it, on filled history chains
(`outputs/2026-10-06/temporal_k10_th350_fill_tattn_posw8_30e`): object F1 **0.8114** on the 24
generation-5 test scenes (2025–26), from **precision 0.834 and recall 0.790**, and **0.8660**
on the ten 2026 scenes. On the 20 relabelled test scenes, same scenes for all three, object F1
(precision / recall):

| | single-frame U-Net | temporal attention, k10 | + filled history chains |
|---|---|---|---|
| original labels | 0.778 (0.790 / 0.766) | 0.797 (0.832 / 0.766) | **0.815** (0.851 / 0.781) |
| corrected labels | 0.814 (0.805 / 0.823) | 0.829 (0.844 / 0.815) | **0.846** (0.861 / 0.830) |
| corrected, doubtful polygons not counted | 0.852 (0.866 / 0.839) | 0.856 (0.838 / 0.875) | **0.876** (0.855 / 0.897) |

Read across the first row: temporal attention's gain over the single frame is **precision**
(same recall, 0.766); filling the history chains adds **both**. Read down: correcting the labels
is mostly a **recall** gain (0.781 → 0.830 → 0.897), because the old labels missed objects the
models found. **On these test scenes recall is the limiting side**, and every model does best
at the loosest threshold scored (RTh 0.125), so a lower threshold may recover more.

Spatial context (ctx50: a 300×200 input around each 200×100 tile) raised patch Dice and lost
all five object-level comparisons (⑤). Tuned in the HPO study it still trails, and at equal
precision it recalls less than plain (⑫).

**What is next.** The HPO study's result: a tuned configuration per geometry, scored once on
the 2026 test block against the 0.8660 bar. Then the LiDAR control, deeper history (it needs a
leaner data layer first) and the six stale stride-4 masks ([Open](#open)).

---

## How to read a number

- **Object-level F1 on whole scenes decides everything.** Patch `val/dice` ranked models
  backwards in four separate batches (finding ②); it is a training-health signal only.
- **Read precision and recall, not only F1.** Recall (the share of mapped subsidence that is
  found) matters most to this project: a missed subsidence costs more than a false alarm. So
  tables give F1 with its precision / recall. Recall moves with the threshold, so compare two
  models **at equal precision** (or across the whole threshold sweep), never each at its own
  best-F1 threshold: the one picked at a looser point will always show more recall.
- **Validation and test want different thresholds.** On the HPO's validation scenes (2022–23)
  precision is the weak side and the strictest threshold wins; on the 2025–26 test scenes
  recall is the weak side and the loosest wins. A threshold chosen on one is not optimal on
  the other.
- **F1 is computed from the per-scene mean precision and mean recall** (`2PR/(P+R)`), the
  convention every table here uses. "Area-weighted" F1, where quoted, weights scenes by
  ground-truth area as the benchmark paper does.
- **The protocol** is the paper's: tiles at stride 4 (16 per pixel), each binarised at
  0.5, the pixel's confidence is the fraction voting positive, swept at Reconstruction
  Thresholds 0.125 / 0.25 / 0.375 / 0.5. Numbers are the best of the four, at the strict
  Intersection Tolerance `ith0.7_b5` unless marked; the soft `ith0.5_b10` compresses every
  table into a band ~0.02 wide, so always quote the tolerance.
- **The four thresholds can stop too low.** On the t2026 val scenes the k10 baseline pair
  scored best at the top of the grid, 0.5, and kept improving above it: plain 0.7329 → 0.7568
  at 0.875, ctx50 0.7040 → 0.7401 at 0.9375, where all 16 tiles agree
  ([`HPO.md`](HPO.md) §13). The HPO study now scores every 1/16 step from 0.5 to 0.9375. Elsewhere on this page a
  number is the best of the four, which may understate a model that over-predicts.
- **Compare within one scene list.** A partition generation changes the scenes *and* the
  scored ground ([`assets/PARTITIONS.md`](../assets/PARTITIONS.md)). Re-average a model's
  saved per-scene results over the other list instead.
- **The noise.** Two runs of an identical configuration differed by **0.011 val/dice**
  (`combo_ctx_p00`, submitted twice by accident) — nearly twice the ±0.006 the August notes
  used. Per-scene object F1 has a standard deviation of ~0.07, so a mean over 10 scenes
  moves ~0.02–0.03 on scene sampling alone. The HPO study's seed repeats are the first
  deliberate measurement ([`HPO.md`](HPO.md)).
- **Two correctness fixes split the record on 2026-09-07/08.** Every `--amp` run before
  them clipped the *scaled* gradient (so clipping fired on nearly every step), and at batch
  ≥ 32 the region loss reduced in fp16 and overflowed, so the run optimised BCE alone.
  Both are recorded in the checkpoint (`gradient_clipping=unscaled-v2`,
  `region_loss_dtype=fp32-v2`) and a resume across them is refused. Rankings inside an
  older batch survive (both arms had the same objective); absolute numbers and claims
  about the Dice term do not.

---

## The best models

The five focus models, kept in `outputs/`; everything else is under `outputs/archive/`.
Strict tolerance, best RTh (all peak at 0.125), original labels.

| Run (`outputs/<date>/…`) | Role | 24-scene gen-5 list | 17-scene gen-4 k10 list |
|---|---|---|---|
| `2026-10-06/temporal_k10_th350_fill_tattn_posw8_30e` | **best model** | **0.8114** (area-weighted 0.8155) | — |
| `2026-10-06/temporal_k5_th350_fill_tattn_posw8_40e` | best k5 | 0.7934 | — |
| `2026-09-16/temporal_k10_th350_tattn_posw8_30e` | reference model; the leader's no-fill twin | 0.7906 | 0.7907 |
| `2026-09-17/temporal_k5_th350_tattn_posw8_40e` | the k5 leader's twin | 0.7876 | — |
| `2026-09-16/temporal_k10_th350_single_posw8_30e` | single-frame baseline | — | 0.7685 |

All four temporal models are plain (200×100 in, no context margin) temporal attention,
`pos_w` 8, AdamW 3e-4 / wd 1e-2, ring negatives 1–10, flips `hv`. The leader's exact
configuration is the building-block block at the top of `scripts/submit_all.sh`.

**The best model, in brief.** `temporal_k10_th350_fill_tattn_posw8_30e` (trained 2026-10-06,
LSF 261016) is a U-Net with temporal attention. One shared encoder reads the current 11-day
interferogram and the ten before it (200×100 tiles of wrapped phase); at the bottleneck each
cell of the current frame attends over the same cell in all eleven frames (8 heads, contrast
and QK-norm, each frame's real age as a position code), and the decoder draws the current
frame's subsidence mask. Trained on generation 5 (label-quality threshold 350 North / 200
South; history chains that skip missing acquisitions) for 30 epochs: AdamW 3e-4, weight decay
1e-2, BCE (pos_weight 8) + Dice, ring negatives 1–10 at 1:1, horizontal and vertical flips.
The evaluated checkpoint is `best.pt`, **epoch 25 of 30**, picked by patch validation Dice,
which is a weak selector (②); the HPO study picks checkpoints on whole scenes instead.

**On the corrected labels** (the 20 relabelled official test scenes,
[`RELABELING.md`](RELABELING.md) §6c), area-weighted / mean F1:

| labels | fill k10 (leader) | no-fill twin | single-frame k10 |
|---|---|---|---|
| original | **0.8081** / **0.8146** | 0.7917 / 0.7974 | 0.7671 / 0.7777 |
| corrected | **0.8444** / **0.8455** | 0.8292 / 0.8291 | 0.8122 / 0.8140 |
| corrected, flagged polygons not counted | **0.8863** / **0.8757** | 0.8660 / 0.8557 | 0.8583 / 0.8523 |

**On the 2026 scenes** (generation 5-t2026's test block: 10 scenes, Feb–Apr 2026), object F1
(precision / recall), best of the four RTh (0.125 unless marked), strict tolerance. The single
U-Net was evaluated on the 8 relabelled ones only, so the three-model comparison uses those 8:

| 8 relabelled 2026 scenes | single-frame U-Net | temporal attention, k10 | + filled history chains |
|---|---|---|---|
| original labels | 0.849 (0.892 / 0.810) @0.25 | 0.845 (0.864 / 0.827) | **0.860** (0.873 / 0.848) |
| corrected labels | 0.861 (0.843 / **0.880**) | 0.862 (0.878 / 0.846) | **0.874** (0.881 / 0.866) |
| corrected, doubtful not counted | 0.868 (0.843 / **0.895**) | 0.870 (0.878 / 0.862) | **0.883** (0.882 / 0.883) |

All 10 scenes, original labels: the leader **0.8660** (0.861 / 0.871), its no-fill twin 0.8499
(0.859 / 0.841). That is the bar a model trained on the t2026 split has to clear
([`HPO.md`](HPO.md)); it was taken at the test scenes' own best threshold.

On 2026 the three models are close: the leader is ahead on F1 in every row, but the single
U-Net has the highest **recall** on corrected labels (0.880 against 0.866), at lower precision.
With eight scenes, gaps under ~0.03 are within scene noise.

**Where the single U-Net passes the attention models.** On the 20 scenes, never past the
leader (any tolerance, threshold or labeling), and past the no-fill twin only in a tie (strict,
doubtful not counted, RTh 0.375: 0.845 each). On the eight 2026 scenes it passes the twin at most
thresholds and the leader at the strict ones (RTh 0.375 and 0.5, both tolerances), where
every model's precision converges (~0.93) and recall decides; never at either model's best
threshold. The recall edge comes from the three 2026 scenes **without a complete 10-frame
history** (20260304, 20260326, 20260406; corrected labels, strict, RTh 0.125):

| | full history (5 scenes) | gappy history (3 scenes) |
|---|---|---|
| single U-Net | 0.877 (0.877 / 0.876) | 0.834 (0.787 / **0.886**) |
| temporal attention, k10 | 0.870 (0.864 / 0.875) | **0.847** (0.902 / 0.798) |
| + filled chains (leader) | **0.892** (0.889 / **0.894**) | 0.844 (0.868 / 0.820) |

Those three are also the only **South-frame** 2026 scenes, so frame and gaps are confounded
here. Over all 20 relabelled scenes they separate. Recall (precision), strict, RTh 0.125:

| scenes | labels | single U-Net | leader (fill) |
|---|---|---|---|
| North, full history (11) | original | 0.70 (0.81) | **0.76** (0.85) |
| | corrected | 0.77 (0.83) | **0.81** (0.86) |
| | corrected, doubtful not counted | 0.88 (0.82) | **0.92** (0.85) |
| South, full history (6) | original | **0.83** (0.76) | 0.81 (0.84) |
| | corrected | **0.89** (0.77) | 0.87 (0.85) |
| | corrected, doubtful not counted | **0.91** (0.77) | 0.89 (0.85) |
| South, gappy history (3) | original | **0.86** (0.79) | 0.80 (0.87) |
| | corrected (= doubtful not counted: no flags there) | **0.89** (0.79) | 0.82 (0.87) |

The pattern holds under all three label versions. On the North frame the leader wins on
recall, precision and F1. On the South frame the single U-Net predicts more: about +0.02
recall at about −0.08 precision with a full history, and history gaps widen its recall edge to
+0.06–0.07 because the temporal models lose recall there and the single frame does not. The
leader still has the higher F1 on the South frame with full history (0.86–0.87 against
0.82–0.83). Six and three scenes: a consistent pattern, not a measured effect.

How the no-fill twin scores those three scenes at all: it was trained on gap-free chains only,
and its own generation-4 evaluation leaves them out. Its numbers here come from the
generation-5 evaluation (2026-10-06), which hands every model the filled chains: still ten
previous frames, but reaching 154–198 days back instead of 110 (20260304's frame ages, in
11-day slots: 0, 1, 2, 3, 4, 5, 8, 11, 12, 13, 18). Each frame's real age goes into the
attention as a sinusoidal position code; it has no learned parameters, so the twin accepts
ages it never trained on (up to 18 slots against 10) and runs, outside its training
distribution. The leader was trained on exactly such chains.

---

## What we know

### ① Ring negatives fixed over-prediction

Training only on patches that contain subsidence (~2.5% of the map) and predicting on all
of it flagged far too much. Adding empty patches from a ring around each sinkhole lifted
object precision by **+0.10 on geo and +0.28 on temporal** in August, with no recall cost
on geo; the far ring (1–10 patches) beat the near one (1–3). The largest single gain the
project has produced, and every model since trains with ring 1–10 at 1:1.

### ② Patch dice ranks models backwards

Ring negatives were flat or down on `val/dice` while lifting object F1 by ~0.1; the worst
model on dice was the best on scenes. It recurred with recurrence vs. single frame on geo,
with the generation-4 architecture sweep, with context (finding ⑤) and with `long200`,
whose rising dice curve (best epoch 200 of 200) came with **−0.027** object F1. Validation
is positives-only by design, so dice cannot see the background false positives the
negatives suppress. Putting negatives into validation (`--add_val_negatives`) was tried and
retired on 2026-08-20 (deleted from the code 2026-10-24): an empty prediction on an empty mask scores dice 1.0, so it only
compressed the scale. For a negative-aware number from `results.csv` read the pixel-pooled
`val/F1`; settle anything at scene level.

### ③ The optimiser: AdamW 3e-4, weight decay 1e-2

Once clipping was fixed, the historical RMSprop at momentum 0.999 diverged inside one
epoch: its momentum is not bias-corrected, so the effective step is `lr/(1−m)` = 1000×
`lr`. No learning rate recovers it (`lrscan`: three decades of LR at m=0.999 all land at
0.58–0.63 dice; the same LR at m=0.9 was the best arm). `adamscan` picked
**AdamW lr 3e-4 / wd 1e-2** — a region, not a point (the top three were within 0.0045) —
and every run since 2026-09-09 uses it. The training templates still *default* to the
historical RMSprop so old runs reproduce; never submit with the defaults
([`PIPELINE.md`](PIPELINE.md#lsf-templates-and-the-current-recipe)).

### ④ Temporal history helps, more with depth; the architecture ranking depends on the operating point

- Generation 3: recurrence (ConvLSTM) beat the single-frame U-Net on both axes (+0.038 geo,
  +0.047 temporal) and attention never beat recurrence.
- Generation 4, `pos_w` 4, k5: temporal attention beat recurrence by **+0.042** (0.7905 vs
  0.7489), all of it recall. At `pos_w` 8 the four architectures collapsed into a 0.013
  band and reordered — so an architecture claim needs its operating point quoted with it.
- Depth, on identical scenes: the attention edge over the single frame widens from +0.007
  at k5 to **+0.032 at k10** (ctx50), and is +0.022 for the plain k10 pair. k10 trains on
  24% fewer positives; the single frame loses to that cut, attention gains anyway.
- Corrected labels shrink the no-fill model's advantage over the single frame (+0.025 →
  +0.017 → +0.008 area-weighted as the labels get cleaner, `RELABELING.md` §6c): much of
  it sat on the hard, flagged polygons. The filled-chain leader keeps more of its lead
  (+0.041 → +0.032 → +0.028).

### ⑤ Spatial context (ctx50) raises dice and costs object recall — plain is the default

`--context_margin 50 50` (300×200 in, centre 200×100 supervised) was worth +0.024 to +0.041
val/dice in three independent pairs. At object level it **lost all five scored pairs**
(−0.008 to −0.029, both generations, both depths): it buys 0.04–0.08 precision and costs
0.06–0.10 recall. History shrinks the loss (−0.016 for attention at k10 against −0.026 for
the single frame), and costs ~3× the host memory. The HPO study re-asks the question at
tuned settings on the t2026 split; so far ctx50 recalls less than plain at equal precision
too (⑫).

### ⑥ Filled history chains are worth +0.021 at k10

Generation 5 lets a chain skip a missing acquisition (up to 3k slots back) and tells the
attention each frame's real age, so scenes the gap-free rule dropped come back: +47 train
scenes at k10, +30 at k5. Against twins differing only in the partition: **+0.021 at k10**
(0.8114 vs 0.7906, well past the noise) and +0.006 at k5 (inside it). The two causes — more
supervised dates and uneven ages — arrive together and are not separated. The leader also
leads under every labeling (16/20 scenes against the twin with flags not counted), in both
recall and precision.

### ⑦ The label-quality threshold is a scene-list shift, not a better model

Generation 4 drops a scene from every split when its positive-patch count is below 350
(North) / 200 (South); predecessors are exempt. Re-averaging finished generation-3 runs
over only the surviving scenes moved object F1 by +0.013 to +0.024 — easier scenes, not
model quality. 350 cuts through the low-precision cluster rather than below it (the real
gap is 359–496).

### ⑧ Relabelling: the old labels missed objects the models found

All 20 official temporal test scenes were relabelled (366 changes, 46 polygons flagged as
doubtful; [`RELABELING.md`](RELABELING.md)). The gain is **recall**: missed objects roughly
halve, precision barely moves. Flagged polygons are what the models miss (recall ~0.13 on
them against ~0.84 elsewhere). Fill leads under every labeling, by a steady +0.015–0.020
area-weighted F1. The QGIS prediction layer during labelling was the no-fill twin, so any
bias from it favours the twin, not the leader.

### ⑨ The attention selects — and selectivity is not what makes it good

`tattn_unet` lets each bottleneck cell of the current frame weight the same cell in each
predecessor. Before 2026-08-19 every checkpoint had collapsed to a uniform average;
`--tattn_contrast` (queries and keys from each frame's deviation from the temporal mean)
and `--tattn_qk_norm` (unit-norm q/k with one learned temperature) fixed it and are on by
default. Every run probed since the AdamW retraining selects (22–34% of uniform). But
across the August arms selectivity correlated with object F1 at **r = −0.10**: whether
the attention is real turned out not to be the question that mattered. At k10 the history
carries recall — masking it costs the ctx50 model 0.044 patch dice, recall 0.834 → 0.666 —
and both k10 models weight the oldest frame in the window (110 days back) heavily, a hint
that deeper history may pay.

`sinkholes attention-probe` measures this on a laptop in minutes (plain checkpoints only;
it refuses a context one). The wrapper used for the September ctx50 probes was deleted
with `outputs/attention_probe/` on 2026-10-15.

### ⑩ The paper's positives-only protocol cannot rank models

Scoring only tiles whose ground truth is non-empty (~1.2% of tiles; `--positives_only`)
leaves recall identical and inflates precision up to 10×, worst on exactly the scenes that
over-predict; it ranked the worst full-scene geo model first. Best positives-only F1 in
August (generation 2): 0.911 geo, 0.815 temporal. Use it only to compare against the
paper, never to choose a model.

### ⑪ Settled — do not re-run

| | evidence |
|---|---|
| bigger ConvLSTM (hidden 512 / 1024) | no gain / worse |
| more epochs on a plateau schedule | `long200`: dice +0.000, object F1 −0.027 |
| the 2023–2026 training years (generation 3) | `pre23` ahead of `clean` in 4 of 6 matched pairs |
| a polygon area filter | 24 RTh × floor combinations; none beats raising RTh (area-weighted metric) |
| coherence maps to explain false positives | AUC 0.53 / 0.51; polygon area alone 0.83 |
| Hann-window stitching | one point, negative (0.640 vs 0.662) |
| 3:1 vs 1:1 negatives | 3:1 behind 1:1 on clean ground at every threshold |
| RMSprop at momentum 0.999 | diverges once clipping is correct (③) |
| bottleneck dropout 0.1–0.2, Jaccard region loss | +~0.01 dice in 20-epoch screens, inside 2× the noise, never scored on scenes |

### ⑫ The HPO study so far: plain leads, and recalls more at equal precision

First phase, 2026-10-26: the baseline (s0) and five spread-out settings per geometry, 30
epochs, seed 42; ctx50 s1 and s2 are still training. Validation objective = mean object F1
over epochs 20 and 30 on the 8 scored val scenes (2022–23), with its precision / recall, at
each trial's best threshold:

| | lr | wd | pos_w | plain | P / R | ctx50 | P / R |
|---|---|---|---|---|---|---|---|
| s0 (baseline) | 3.0e-4 | 1.0e-2 | 8 | 0.757 | 0.66 / 0.89 | 0.740 | 0.63 / 0.91 |
| s1 | 1.1e-4 | 1.5e-2 | 4 | **0.770** | 0.71 / 0.85 | training | |
| s2 | 1.9e-4 | 1.3e-3 | 12 | 0.760 | 0.67 / 0.88 | training | |
| s3 | 2.6e-4 | 1.4e-4 | 2 | 0.759 | 0.69 / 0.85 | 0.727 | 0.61 / 0.90 |
| s4 | 4.9e-4 | 7.0e-4 | 12 | 0.756 | 0.65 / 0.90 | 0.701 | 0.56 / 0.94 |
| s5 | 7.1e-4 | 6.9e-3 | 4 | 0.759 | 0.68 / 0.87 | **0.757** | 0.65 / 0.91 |

- **Plain is flat** across the settings (0.756–0.770); **ctx50 is sensitive** (0.701–0.757),
  and its best only reaches plain's baseline.
- **pos_w trades precision for recall**, as designed: 2–4 gives the highest precision, 12 the
  highest recall (and ctx50's worst F1).
- **Precision is the weak side on these scenes**, and most trials peak at the strictest
  thresholds (0.875–0.9375).
- **Recall, compared fairly.** At its best-F1 threshold ctx50 shows more recall (s0: 0.909 vs
  0.891), but it is picked at a looser precision. Recall at equal precision, the best over all
  thresholds:

  | | P ≥ 0.55 | P ≥ 0.60 | P ≥ 0.625 | highest P reached |
  |---|---|---|---|---|
  | plain s0 | 0.983 | 0.973 | 0.953 | 0.674 |
  | plain s1 (best plain) | 0.970 | 0.938 | 0.925 | 0.708 |
  | ctx50 s0 | 0.936 | 0.920 | – | 0.625 |
  | ctx50 s5 (best ctx50) | 0.953 | 0.934 | 0.915 | 0.647 |

  Plain recalls as much or more at every precision ctx50 reaches, and reaches precisions
  ctx50 does not. Between the two best settings the gap is small (0.004–0.017); between the
  baselines it is 0.04–0.05. The study's visual panel is where "context looks better" gets
  its own test.
- **But ctx50's recall barely depends on the threshold.** At the *same* strict threshold
  (RTh ≥ 0.8125) ctx50 recalls more than plain. From RTh 0.5 to 0.9375 every ctx50 trial loses
  only 0.018–0.028 recall, every plain trial 0.095–0.120; precision rises by +0.05–0.06 against
  +0.09–0.12. ctx50's overlapping tiles agree with each other, on true objects and on false
  alarms alike, so a stricter vote removes little of either. Plain's tiles disagree more, so
  the threshold trims both. (Likely cause, not yet measured: a ctx50 tile sees 50 px beyond its
  edges, so tiles that overlap an object judge it with the same surroundings.) Two
  consequences:
  - an RTh is each model's own agreement level, not a common scale: at equal precision (0.62)
    ctx50 sits at RTh 0.9375 and plain at 0.6875, where plain recalls 0.963 against 0.909;
  - ctx50 is **robust to a mis-set threshold**. Validation and test want different thresholds
    here, so at a threshold locked on validation and applied to test, ctx50 may lose less
    recall than plain even though its precision–recall curve is lower. The study's final test
    numbers must be read across the whole sweep for this reason.

Two cautions. The 2022–23 validation scenes were never relabelled, and ⑧ showed that old
labels miss objects the models find, so part of the low validation precision may be unmapped
subsidence. And with 8 scenes and one seed, gaps of 0.01–0.02 are not yet established; the
seed repeats measure that.

### Known data limits

- **Six stride-4 masks are stale.** They were cut in June 2026 from the previous GT file,
  which had not labelled these six scenes yet; the current GT holds 54–207 polygons in each
  ([`DATASET_11DAY.md`](DATASET_11DAY.md#known-stale)). Every RTh evaluation reads stride 4, so
  it scores them as scenes without ground truth. No evaluation on this page scores them; the
  HPO study's validation holds three of them and therefore averages the other 8.
- **`assets/lidar_intf_mask.txt` ends 2024-06-05**, so every temporal test scene falls
  back to the 2022 LiDAR coverage for gating. The decisive control — one model, the same
  scenes, `--no-add_lidar_mask` — has never been run.
- Six 2025 scenes sat at ~0.45–0.60 recall for every model on the original labels; all are
  in the relabelled set, where flags mark what the models miss.

---

## Open

1. **The HPO study** ([`HPO.md`](HPO.md)): plain vs ctx50 temporal attention at k10 on the
   t2026 split, tuned and seed-repeated, with 300-epoch runs. Study v2 (`k10t2026_hpo_v2`,
   thresholds in 1/16 steps up to 0.9375) adopted v1's baseline pair: val objective plain
   **0.7568** @ RTh 0.875, ctx50 0.7401 @ 0.9375 (best val/dice was the other way round,
   0.6189 vs 0.6356). The ten `shared` trials were submitted 2026-10-24; 8 are scored (⑫),
   ctx50 s1 and s2 are still training. TPE, promotion to 60 epochs, seed repeats, a locked
   threshold and the 300-epoch runs follow. The objective averages 8 of the 11 val scenes
   (the stale masks below). Its plain baseline is the leader's configuration on the t2026
   split, so scoring its chosen checkpoint on the 10 test scenes answers whether training up
   to 2025 helps. **Read that test score across the whole threshold sweep, with precision and
   recall:** the study locks its threshold on validation, where the strictest wins, while the
   0.8660 bar was taken at the test scenes' own best threshold (the loosest), so the bar is
   optimistic and a single locked-threshold number would understate recall.
2. **The LiDAR control** and **extending `lidar_intf_mask.txt` past 2024-06-05** (above).
3. **Deeper history.** The data layer materialises every sample's full stack, so host
   memory grows with T (ctx50 k10 peaked at 298 GB in the HPO baseline). A deduplicated frame store —
   each grid loaded once, samples pointing at rows — comes first.
4. **A context-checkpoint attention probe**, folded into `sinkholes attention-probe`.
5. **The hybrid** (ConvLSTM + attention) won at neither operating point and 3 of 4 August
   hybrids collapsed to uniform attention: give it a term that forces selection, or drop it.
6. **Rebuild the six stale stride-4 masks** once the HPO study is finished (not before: its
   scores must stay comparable).
7. **A recall-first operating point.** Choose the threshold by recall at a precision floor
   (or by F2) instead of F1; the study records every threshold's precision and recall, so this
   needs no new runs. On the test scenes, also score thresholds below 0.125, where recall is
   still rising.

---

## History

This page condenses the lab notebooks that used to sit beside it (`EXPERIMENTS.md`,
`MODEL_RUNS.md`, `PREDICTIONS.md`, `ATTENTION.md`, `CORRECTNESS_FIXES.md`,
`POSITIVES_ONLY_EVAL.md` and the executed `PLAN_*` documents). They are kept verbatim at the
tag `notebook-2026-10-15`: `git show notebook-2026-10-15:docs/EXPERIMENTS.md`.
