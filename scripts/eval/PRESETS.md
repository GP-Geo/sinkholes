# Evaluation presets

Configurations for `run_eval.sh`. Each row is the CONFIG delta; everything not
listed keeps the template default.

## Template defaults

    RUN=outputs/2026-08-06/geo_k10_convlstm_posw8   CKPT=best.pt
    GROUP=geo_k10   SPLIT=test   ARCH=convlstm   K_PREVS=10
    RECON_TH=0.25   OL_TH=0.7    BUFFER=5
    BLEND=none      GAMMA=1.0

`ARCH` and `K_PREVS` must match how the checkpoint was **trained** —
`scripts/train/PRESETS.md` records that for every run. The template refuses a
`K_PREVS` that contradicts the group, and warns if the run's own group (read
from its directory name) differs from the group you are evaluating on.

## Run so far

| Job name (`#BSUB -J`) | CONFIG delta | Notes |
|---|---|---|
| `eval_geo_k10_test` | *(defaults)* | The 18 held-out geo-k10 test interferograms. Submitted 2026-08-09. |
| `eval_k10split_val` | `RUN=outputs/2026-08-03/k10split_convlstm_INVALID` `GROUP=20_05_16h53_k10_compatible` `SPLIT=val` | Legacy: the retired partition, which has no test split and no `_testeval` variant. Output already exists (88 GB) under `outputs/predictions/convlstm_10prev_k10split/`. |

## Worth running next

Patch-level `val/dice` has stopped discriminating between the leading models —
the top six temporal runs sit inside one noise floor (±0.006, see
`docs/MODEL_RUNS.md`). Object-level precision/recall on the **test** splits is
the metric that can still separate them. One eval per candidate:

**Always score on the `_k10` list, even for k5 models.** Unlike the val splits,
the test splits are *not* interchangeable: `temporal_k5`/`temporal_k10` test
overlap only 0.83, and geo/temporal test overlap 0.15–0.22 where their val sets
were perfectly disjoint. But `temporal_k10` test (20) is a strict subset of
`temporal_k5` test (24), `geo_k10` test (18) of `geo_k5` test (26), and every
scene in both has 5- *and* 10-previous chains. So the k10 list is common ground
— held out for k5 and k10 models alike, and the only way these numbers rank
against each other. Set `GROUP` to the k10 partition and `K_PREVS` to whatever
the model was trained with; the template notes the mismatch instead of blocking.

### Geo axis — 18 scenes (`geo_k10` test)

| Template delta | Why |
|---|---|
| *(already run)* | `geo_k10_convlstm_posw8`, the template defaults. |
| `RUN=…/geo_k5_convlstm_posw8` `K_PREVS=5` | **Highest value.** The geo leader on the *same* 18 scenes, so it settles "is 206375 actually the better geo model" at object level — the question the whole best-model premise rested on. |

### Temporal axis — 20 scenes (`temporal_k10` test), a one-factor design

`temporal_k10_convlstm_posw8` (k=10, `pos_w` 8, ConvLSTM h256) is the
**hub**: every other row differs from it in exactly one factor, so each contrast
is clean. Run the hub — without it, k5-vs-k10 and pos_w-8-vs-4 are confounded
with each other and neither is interpretable.

| Template delta | Factor vs hub |
|---|---|
| `RUN=…/temporal_k10_convlstm_posw8` `GROUP=temporal_k10` | **the hub** |
| `RUN=…/temporal_k5_convlstm_run2` `GROUP=temporal_k10` `K_PREVS=5` | k: 10 → 5 |
| `RUN=…/temporal_k10_convlstm_posw4` `GROUP=temporal_k10` | `pos_w`: 8 → 4 |
| `RUN=…/temporal_k10_stack` `GROUP=temporal_k10` `ARCH=stack` | architecture: ConvLSTM → stacked channels |
| `RUN=outputs/2026-08-05/temporal_k5_single_b64` `GROUP=temporal_k10` `ARCH=single` | architecture → single-frame. **Not one-factor:** also b64/lr1e-5, so it bounds the floor rather than isolating the effect. |

All of these are wired up in `scripts/submit_all.sh` — `bash scripts/submit_all.sh eval`
lists them, `--submit` sends them.

Evaluating on `SPLIT=val` reproduces the number the model was *selected* on and
is optimistically biased; use it only as a sanity check against `results.csv`.

### Stitch blending — `BLEND`

`BLEND=hann` weights each tile's contribution by a Hann window instead of
averaging all overlapping tiles equally, so a pixel is scored mostly by the
tiles that saw it *centrally* rather than at their own border. It targets the
spurious blobs that appear along tile seams. **Every number in
`docs/RESULTS.md` and `docs/MODEL_RUNS.md` is `BLEND=none`** — the flag existed
in `reconstruct.py` from the start but no eval ever set it until the `blend`
batch in `submit_all.sh` (the geo_k5 architecture triple at 1:1 ring-3
negatives, each paired with an existing unblended eval on the same 18 scenes).

Two things to know before comparing:

- **Do not compare at a single threshold.** Unblended stitching divides by
  `stride**2` regardless of how many tiles contributed, so pixels near scene
  borders and LiDAR-mask edges are attenuated; Hann divides by the true
  accumulated weight and removes that attenuation. A fixed-threshold delta mixes
  seam cleanup with de-attenuation. Compare best-F1 across the full sweep.
- **`rescore.sh` cannot produce these numbers.** Blending happens during
  reconstruction, before `_pred.npy` is written — this needs a full stage-1
  re-run, not the cheap re-threshold path.

`GAMMA>1` sharpens the window further; leave it at 1.0 until a plain Hann is
shown to move something, or blending and its strength get confounded.

## The clean benchmark — `GEN=3` + `DATA_STRIDE=4` (kind: eval5)

`submit_all.sh eval5 --submit` — 8 of the 14 clean22 runs, the queued work as of
2026-08-19. Two knobs are new; both default to the old behaviour so nothing
archived is affected.

**`GEN`** picks the partition generation (`assets/PARTITIONS.md`):

| `GEN` | resolves to | AOI | comparable to |
|---|---|---|---|
| `2` (default) | `partition_<group>[_testeval].json` | none — whole canvas | `docs/PREDICTIONS.md` |
| `3` | `partition_<group>[_testeval]_clean.json` | per-split `aoi_window` | nothing before it |

Under `GEN=3` the template passes `--aoi_from_partition "$PARTITION" --aoi_split val`
to **both** stages. Both, not one: `eval-scenes` decides which tiles are
*predicted* and `eval-outputs` crops the canvas it *scores*, and if they
disagree the metrics cover different ground than the predictions.

`--aoi_split` is **`val`**, never `test`, and never the tools' own default of
`test`. `--intf_source preset` reads only the `"val"` key, and the `_testeval`
files hold the parent's test list there and carry no `"test"` window at all —
so `val` is the window belonging to the scenes actually being scored, in both
`SPLIT` modes. Passing `test` against a `_testeval` file exits with a message
rather than mis-scoring.

**`DATA_STRIDE`** picks the reconstruction geometry:

| | stride 2 (default) | stride 4 |
|---|---|---|
| Step | half patch | quarter patch |
| Tiles per interior pixel | 4 | **16** — the paper's |
| Whole-canvas grid | 193 × 89 = 17,177 | 386 × 177 = 68,322 |
| Bytes per scene per timestep | 1.37 GB | **5.47 GB** |
| Patch tree | `*_strpp2_11days_Aligned` | `*_strpp4_11days_Aligned` |

**`PROTOCOL`** picks what the confidence map *means* — implemented 2026-08-19,
`docs/PLAN_CLEAN_BENCHMARK.md` §7(b) items 1 and 4:

| | `prob` (default) | `rth` |
|---|---|---|
| Per-pixel value | mean predicted probability | **fraction of tiles voting 1** |
| Tile handling | probabilities averaged | each tile **binarised at 0.5** first |
| Value range | continuous [0, 1] | {0, 1/16, …, 1} at stride 4 |
| Swept at | 0.125/0.25/0.5/0.7/0.9 | **RTh 0.125/0.25/0.375/0.5** |
| Tolerances | `--th`/`--buffer` only | **ITh 0.7/b 5 *and* ITh 0.5/b 10** |
| Aggregation | unweighted scene mean | unweighted **and** GT-area weighted |

`PROTOCOL=rth` sets `--recon_average vote --vote_threshold 0.5` on `eval-scenes`
and `--rth` on `eval-outputs`. Both stages move together on purpose: scoring a
vote map with probability thresholds (or the reverse) yields numbers that look
plausible and are not.

### ⚠️ An RTh is not a `recon_th`

One counts tiles, the other cuts a mean. **RTh 0.25 means "at least 4 of the 16
overlapping tiles voted positive"**, which is a different statement from "the
mean probability exceeded 0.25". Never put an RTh number in the same column as
anything in `docs/PREDICTIONS.md`.

`PROTOCOL=rth` **requires `DATA_STRIDE=4`** and the template refuses otherwise:
the paper's thresholds are defined as 2/4/8 of *sixteen*, and stride 2 gives
four tiles per pixel, so the quantisation becomes 1/4 and the numbers stop
meaning what the paper says. It also refuses `BLEND=hann` — a vote is unweighted
by definition.

**Recall is the primary metric under RTh**, per the paper: full-range
reconstruction scores a great deal of unannotated ground, so precision partly
measures the digitisation rather than the model.

### What the RTh JSON holds

`olm_results_*.json` stays backward compatible — `per_intf` and `summary` keep
the historical schema at the **primary** tolerance — and adds:

- `threshold_kind`: `"vote"` under RTh, `"probability"` otherwise. Check this
  before reading any threshold key.
- `summary[th].weighted_recall` / `weighted_precision`: the GT-area-weighted
  aggregate, next to the unweighted `mean_*`. A scene with three polygons stops
  counting as much as one with three hundred.
- `per_intf_by_tolerance` / `summary_by_tolerance`, keyed `ith0.7_b5`,
  `ith0.5_b10`.
- `gt_area` per scene per threshold, which is what the weighting uses.

Scene-level scoring now passes `round_ndigits=None` into `object_level_evaluate`,
so per-scene figures are no longer quantised to 0.01 before being averaged. The
patch-level path keeps the historical rounding.

### Still unimplemented, and what it costs

Two §7(b) items remain, both performance rather than protocol — they are what
the sizing below works around:

- **Batched tile inference** (§7(b) item 2): one tile per forward pass,
  `reconstruct.py`. Stride 4 is ~4× the tiles.
- **Banded streaming** (§7(b) item 3): `scenes.py` loads each timestep's whole
  grid as float32, which is what drives the host-memory rungs below.

Neither changes a number; both change how long and how much memory it takes.

### Sizing a stride-4 evaluation

The stride-2 rung (`80 GB / 8:00`) does not transfer. Host memory is the binding
constraint, not the GPU: the input stack alone is `T × 5.47 GB`, so 33 GB at k5
and 60 GB at k10, against ~8 GB when the 80 GB rung was measured at a ~57 GB
peak. The rungs in `submit_all.sh` are 112 GB (k5), 144 GB (k10), 40 GB
(single-frame — one timestep, so 5.5 GB not 33).

The AOI is what makes stride 4 affordable at all. Tiles actually predicted per
scene, from `sinkholes.geo.grid_window`:

| Split | stride 2 | stride 4 | share of canvas |
|---|---:|---:|---|
| geo test/val — South band 31.25–31.40 | 2,340 | 9,360 | 13.7% |
| geo train — North 31.40–31.75 | 6,820 | 27,528 | 40.3% |
| geo crossview — North 31.25–31.40 | 2,860 | 11,544 | 16.9% |
| temporal — North, full box | 9,790 | 39,516 | 57.8% |
| temporal — South, full box | 3,015 | 11,970 | 17.5% |

A geo scene at stride 4 costs 9,360 tiles — only 1.4× a *whole* stride-2 North
scene. A temporal North scene costs 39,516, which is why the temporal rung
carries 16:00 against geo's 12:00.

**Read the real peaks back and correct these rungs** before the next batch:
`grep -A3 "Max Memory" logs/*.out`.

## Gotchas

- **`eval-scenes` only reads a partition's `"val"` list**
  (`sinkholes/inference/scenes.py`). `SPLIT=test` therefore resolves to
  `assets/partition_<group>_testeval[_clean].json`, whose `"val"` is exactly the
  parent's `"test"` — verified identical for all four groups, both generations.
  Do not hand it a plain partition file and expect the test split. The same rule
  is why `--aoi_split` is `val` under `GEN=3`.
- **Generations are not comparable, on any metric.** `GEN=2` and `GEN=3` differ
  in scene list *and* in scored ground (generation 3 crops to the AOI). The
  generation-2 geo split also put the 31.25–31.44° band in train and hold-out
  both, so its geo scores were measured partly on training ground —
  `assets/PARTITIONS.md`. Generation-3 numbers replace `docs/PREDICTIONS.md`
  rather than extending it.
- **`CKPT=last.pt` will fail preflight on most runs.** `last.pt` was pruned
  repo-wide on 2026-08-09; only `best.pt` survives, and only for 9 runs. The
  template checks before requesting GPU time.
- **`_image.npy` dominates the output size.** A single 18-interferogram eval
  wrote 44 GB of input stacks against 12 GB of predictions/GT. Those are
  reconstructible from the patch tree; the metrics JSON and shapefiles are what
  you actually keep.
- **Output location changed.** The template derives
  `outputs/predictions/<run-dir-name>`, so a re-run of the geo-k10 eval lands
  in `…/geo_k10_convlstm_posw8`, not the hand-named
  `…/geo_k10_convlstm_posw8` that the 2026-08-09 job is writing to. Expect
  two trees if you re-run it; delete whichever you do not want.
