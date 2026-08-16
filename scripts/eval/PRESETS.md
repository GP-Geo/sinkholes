# Evaluation presets

Configurations for `run_eval.sh`. Each row is the CONFIG delta; everything not
listed keeps the template default.

## Template defaults

    RUN=outputs/2026-08-06/geo-k10_convlstm-h256_lsf206375   CKPT=best.pt
    GROUP=geo_k10   SPLIT=test   ARCH=convlstm   K_PREVS=10
    RECON_TH=0.25   OL_TH=0.7    BUFFER=5

`ARCH` and `K_PREVS` must match how the checkpoint was **trained** —
`scripts/train/PRESETS.md` records that for every run. The template refuses a
`K_PREVS` that contradicts the group, and warns if the run's own group (read
from its directory name) differs from the group you are evaluating on.

## Run so far

| Job name (`#BSUB -J`) | CONFIG delta | Notes |
|---|---|---|
| `eval_geo_k10_test` | *(defaults)* | The 18 held-out geo-k10 test interferograms. Submitted 2026-08-09. |
| `eval_k10split_val` | `RUN=outputs/2026-08-03/k10split_convlstm-h256` `GROUP=20_05_16h53_k10_compatible` `SPLIT=val` | Legacy: the retired partition, which has no test split and no `_testeval` variant. Output already exists (88 GB) under `outputs/predictions/convlstm_10prev_k10split/`. |

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
| *(already run)* | `geo-k10_convlstm-h256_lsf206375`, the template defaults. |
| `RUN=…/geo-k5_convlstm-h256_lsf207929` `K_PREVS=5` | **Highest value.** The geo leader on the *same* 18 scenes, so it settles "is 206375 actually the better geo model" at object level — the question the whole best-model premise rested on. |

### Temporal axis — 20 scenes (`temporal_k10` test), a one-factor design

`temporal-k10_convlstm-h256_lsf202343` (k=10, `pos_w` 8, ConvLSTM h256) is the
**hub**: every other row differs from it in exactly one factor, so each contrast
is clean. Run the hub — without it, k5-vs-k10 and pos_w-8-vs-4 are confounded
with each other and neither is interpretable.

| Template delta | Factor vs hub |
|---|---|
| `RUN=…/temporal-k10_convlstm-h256_lsf202343` `GROUP=temporal_k10` | **the hub** |
| `RUN=…/temporal-k5_convlstm-h256_run2_lsf208207` `GROUP=temporal_k10` `K_PREVS=5` | k: 10 → 5 |
| `RUN=…/temporal-k10_convlstm-h256_posw4_lsf209866` `GROUP=temporal_k10` | `pos_w`: 8 → 4 |
| `RUN=…/temporal-k10_unet-stack_lsf209864` `GROUP=temporal_k10` `ARCH=stack` | architecture: ConvLSTM → stacked channels |
| `RUN=outputs/2026-08-05/temporal-k5_unet-single_baseline` `GROUP=temporal_k10` `ARCH=single` | architecture → single-frame. **Not one-factor:** also b64/lr1e-5, so it bounds the floor rather than isolating the effect. |

All of these are wired up in `scripts/submit_all.sh` — `bash scripts/submit_all.sh eval`
lists them, `--submit` sends them.

Evaluating on `SPLIT=val` reproduces the number the model was *selected* on and
is optimistically biased; use it only as a sanity check against `results.csv`.

## Gotchas

- **`eval-scenes` only reads a partition's `"val"` list**
  (`sinkholes/inference/scenes.py`). `SPLIT=test` therefore resolves to
  `assets/partition_<group>_testeval.json`, whose `"val"` is exactly the
  parent's `"test"` — verified identical for all four groups. Do not hand it a
  plain partition file and expect the test split.
- **`CKPT=last.pt` will fail preflight on most runs.** `last.pt` was pruned
  repo-wide on 2026-08-09; only `best.pt` survives, and only for 9 runs. The
  template checks before requesting GPU time.
- **`_image.npy` dominates the output size.** A single 18-interferogram eval
  wrote 44 GB of input stacks against 12 GB of predictions/GT. Those are
  reconstructible from the patch tree; the metrics JSON and shapefiles are what
  you actually keep.
- **Output location changed.** The template derives
  `outputs/predictions/<run-dir-name>`, so a re-run of the geo-k10 eval lands
  in `…/geo-k10_convlstm-h256_lsf206375`, not the hand-named
  `…/convlstm_geo_k10_lsf206375` that the 2026-08-09 job is writing to. Expect
  two trees if you re-run it; delete whichever you do not want.
