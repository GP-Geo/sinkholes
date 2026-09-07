# `outputs/` — what is kept, what is prunable, and what pruning costs

*Written 2026-09-03. `sinkholes/inference/outputs.py:226` points here.*

Three trees, one naming convention, and one rule about what may be deleted.

| Tree | Holds |
|---|---|
| `outputs/<date>/` | **training runs**, one per model, grouped by the day it trained |
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
