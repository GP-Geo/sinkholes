# `outputs/` — what is kept, what is prunable, and what pruning costs

*Written 2026-09-03, last checked 2026-09-17. `sinkholes/inference/outputs.py:226` points
here.*

Four trees, one naming convention, and one rule about what may be deleted.

| Tree | Holds |
|---|---|
| `outputs/<date>/` | **training runs**, one per model, grouped by the day it trained |
| `outputs/screens/<date>/` | **screens** — 8/20-epoch probes, not models (since 2026-09-09) |
| `outputs/predictions/` | **full-scene evaluations**, one directory per evaluated model |
| `outputs/predictions_positives/` | **positives-only evaluations**, the paper's protocol |

The naming convention itself lives in [`outputs/README.md`](../outputs/README.md).

---

## What an evaluation directory holds

`eval-scenes` writes four `.npy` per interferogram plus figures and polygons.
They are not equally valuable:

| File | Size share | Read by | Keep? |
|---|---|---|---|
| `<intf>_pred.npy` | 12 % | `outputs.py:214`, `rescore.sh:51` | **yes** — every re-score reads it |
| `<intf>_gt.npy` | 12 % | `outputs.py:215` | **yes** |
| `<intf>_image.npy` | 68 % | **nothing, unless `--save_figures`** | no |
| `<intf>_pred_th.npy` | 12 % | **nothing at all** | no |
| `olm_results_*.json` | ~0 % | you | **yes — this is the result** |
| `*_overview.png`, shapefiles | ~0 % | you | yes |

`_image.npy` is the `(T, H, W)` **input** stack — the interferograms fed to the
model, not anything it produced. `object_level_evaluate` touches `image` only to
build per-object features, and the caller passes `features=()`, so `image=None`
produces byte-identical numbers (`outputs.py:217-226`). It is loaded solely for
`--save_figures`.

`_pred_th.npy` is `_pred.npy` thresholded at `recon_th`. It is written by
`scenes.py:442` and **read by nothing** — not by `eval-outputs`, not by
`rescore.sh`. It is fully derivable from `_pred.npy`.

## The prune rule

> **Delete `_image.npy` and `_pred_th.npy` freely. Never delete `_pred.npy` or
> `_gt.npy` from an evaluation whose numbers you still quote.**

Keeping the first pair costs 80 % of the tree and buys back only
`--save_figures` on an old directory. Keeping the second pair is what makes
`scripts/eval/rescore.sh` able to add a threshold or a tolerance in minutes
where a rebuild costs a ~3 h `run_eval.sh`.

Two guards worth reusing if you script this:

- Refuse to delete inside any directory with no `olm_results*.json` — that is an
  evaluation still in flight.
- Existing `*_overview.png` are already on disk and unaffected. Only *new*
  figures need `_image.npy`.

`eval-outputs` fails loudly rather than silently mis-scoring if you ask for
figures from a pruned directory:

```
--save_figures needs <path>/<intf>_image.npy, which is not there.
Pruned eval directories keep their metrics and confidence maps but not the
input stacks. Either drop --save_figures (metrics do not need it) or rebuild
with run_eval.sh.
```

### Applied 2026-09-03

`outputs/` went **2.2 TB → 483 GB**: 1.58 TB of `_image.npy`, 0.25 TB of
`_pred_th.npy`, and 16.7 GB of `resume.pt`/`last.pt` from 28 settled runs.
Every `best.pt` (52) and every metrics JSON (56) survived unchanged. Verified
before the sweep by pruning one directory and re-scoring it: **374 numeric
fields, 0 differences.**

`resume.pt` is only a requeue key — `--resume auto` globs
`outputs/*_lsf_$LSB_JOBID` — so it is dead weight once a batch is settled and
its runs have been filed under a dated folder.

### Applied 2026-09-09

`outputs/` went **885 GiB → 594 GiB**: **290.5 GiB** across **280 files** —
140 `_image.npy` and 140 `_pred_th.npy` — from the **seven** evaluation
directories written since the 2026-09-03 sweep, plus 821 MB of duplicate
weights (below). Everything else in `predictions/` had already been pruned by
that sweep and was untouched.

Both guards above were applied and both held: no directory was missing its
`olm_results*.json` (0 skipped), and every `_pred.npy` / `_gt.npy` survived —
verified after the sweep by re-counting, with `pred` and `gt` equal in all 42
directories and every metrics JSON and `*_overview.png` present.

The seven were the two ctx50 arms (two evaluation directories each: the
stride-4 `rth` run and the stride-2 `prob_s2` run), `ampfix_t5_single_ring3_200e`,
`long200_t5_convlstm_ring3_200e`, and `temporal_k5_pre2023_convlstm_ring3`.
They were expensive to hold because a ctx50 `_image.npy` is a **300×200** input
stack rather than 200×100 — ~46 GiB per directory against ~13 GiB for a plain
one.

Filed the same day: 31 runs off the top level, 22 of them into the new
`outputs/screens/` tree. `outputs/README.md` has the mapping, the
`combo_ctx_p00` duplicate, and why the `ampfix_`/`long200_` prefixes stayed on
the four renamed evaluation directories.

### Not applied since: the eleven September evaluation directories

As of 2026-09-17 the generation-4 evaluation directories are **unpruned** — **205
`_image.npy` and 205 `_pred_th.npy`** across eleven directories (four `th350`, four
`posw8` k5, three `posw8_*_k10`). Ten of the eleven are `ctx50`, where an `_image.npy` is
a **300×200** input stack rather than 200×100 — ~46 GiB per directory against ~13 GiB for
a plain one. The exception is `posw8_single_k10_plain_30e`.

> **The first guard fires here, and it matters.**
> `posw8_tattn_k10_ctx50_30e_…_lsf_254183` holds **11 of 17** scenes and **no
> `olm_results_*.json`** — it is an evaluation still in flight (`evalk10_tattn_ctx50`).
> Do not prune inside it, and do not read a score out of it.

Everything else in `predictions/` was already pruned by the 2026-09-03 and 2026-09-09
sweeps.

---

## The 2026-09-03 rename

The 19 runs of 2026-08-20 were filed under `outputs/2026-08-20/` and renamed to
the convention, with their evaluation directories renamed to match. Two rules
beyond the usual suffix strip:

- **`pre23_<partition>` → `<partition>_pre2023`.** The tag marks a different
  *training set* (the archive truncated at 2022-12-31), which is the one thing
  about those runs that changes what their numbers mean — so it belongs where
  the partition is named, not as a batch prefix. `geo_k5` and `geo_k5_pre2023`
  now read as the two partitions they are.
- **`_fixed` dropped.** It marked `contrast`+`qk_norm` when only some arms had
  it. Every arm of both 2026-08-20 batches carries it, so by the "defaults are
  not in the name" rule it is no longer a variant. The dated folder separates
  these from the pre-fix runs that still need the distinction.

| Was | Is |
|---|---|
| `clean_geo_k10_tattn_fixed_ring3_valpos_2026-08-20_13h26_lsf_943905` | `2026-08-20/geo_k10_tattn_ring3_valpos` |
| `clean_geo_k10_tattn_hybrid_fixed_ring3_valpos_…_943908` | `2026-08-20/geo_k10_tattn_hybrid_ring3_valpos` |
| `clean_geo_k5_tattn_fixed_ring3_valpos_…_943906` | `2026-08-20/geo_k5_tattn_ring3_valpos` |
| `clean_temporal_k10_convlstm_ring3_valpos_…_943913` | `2026-08-20/temporal_k10_convlstm_ring3_valpos` |
| `clean_temporal_k10_tattn_fixed_ring3_valpos_…_943910` | `2026-08-20/temporal_k10_tattn_ring3_valpos` |
| `clean_temporal_k10_tattn_hybrid_fixed_ring3_valpos_…_943912` | `2026-08-20/temporal_k10_tattn_hybrid_ring3_valpos` |
| `clean_temporal_k5_tattn_fixed_ring3_valpos_…_943909` | `2026-08-20/temporal_k5_tattn_ring3_valpos` |
| `clean_temporal_k5_tattn_hybrid_fixed_ring3_valpos_…_943911` | `2026-08-20/temporal_k5_tattn_hybrid_ring3_valpos` |
| `pre23_geo_k5_single_ring3_…_945268` | `2026-08-20/geo_k5_pre2023_single_ring3` |
| `pre23_geo_k5_convlstm_ring3_…_945266` | `2026-08-20/geo_k5_pre2023_convlstm_ring3` |
| `pre23_geo_k5_tattn_ring3_…_945272` | `2026-08-20/geo_k5_pre2023_tattn_ring3` |
| `pre23_geo_k10_convlstm_ring3_…_945267` | `2026-08-20/geo_k10_pre2023_convlstm_ring3` |
| `pre23_geo_k10_tattn_ring3_…_945273` | `2026-08-20/geo_k10_pre2023_tattn_ring3` |
| `pre23_geo_k10_tattn_hybrid_ring3_…_945275` | `2026-08-20/geo_k10_pre2023_tattn_hybrid_ring3` |
| `pre23_temporal_k5_single_ring3_…_945271` | `2026-08-20/temporal_k5_pre2023_single_ring3` |
| `pre23_temporal_k5_convlstm_ring3_…_945269` | `2026-08-20/temporal_k5_pre2023_convlstm_ring3` |
| `pre23_temporal_k5_tattn_ring3_…_945276` | `2026-08-20/temporal_k5_pre2023_tattn_ring3` |
| `pre23_temporal_k10_convlstm_ring3_…_945270` | `2026-08-20/temporal_k10_pre2023_convlstm_ring3` |
| `pre23_temporal_k10_tattn_ring3_…_945277` | `2026-08-20/temporal_k10_pre2023_tattn_ring3` |

The evaluation directories under `outputs/predictions/` carry the same 19 new
names. The other 21 already followed the convention and were left alone.

`scripts/tidy_outputs.sh` performs this — dry-run by default, `--apply` to move.
It reads `_valpos`/`_valneg` out of each run's own `reporter.log` rather than
guessing, and refuses to move a run with no completion line.

> ⚠️ **Close Finder first, and check no evaluation is queued.** `mv` on a
> directory under `/Volumes/rudich` fails while Finder holds it open, and an
> LSF job carries its `RUN=` path from submit time — a PEND job whose run
> directory moves underneath it dies on its preflight.

## A known limitation

`outputs/predictions/` is a **flat namespace**: it is keyed by run name with no
date component, so two runs with the same name trained on different days cannot
both have an evaluation directory. The 2026-09-03 mapping does not trigger this
(checked against `outputs/2026-08-19/`), but `eval6ref` will add `_valpos` names
to that tree — check for a clash before adding a batch that reuses names.
