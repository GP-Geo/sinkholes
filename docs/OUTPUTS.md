# What is on disk, and what may be deleted

*Rewritten 2026-10-15. `sinkholes/inference/outputs.py` points here. The record of every
past prune and rename is in this file at tag `notebook-2026-10-15`.*

All of it is gitignored and lives on the share (`/home/labs/rudich/pinkas/sinkholes` on
WEXAC, `/Volumes/rudich/pinkas/sinkholes` on a Mac).

| Tree | Holds |
|---|---|
| `outputs/<date>/<run>/` | the five **focus models**' training runs ([`RESULTS.md`](RESULTS.md)) |
| `outputs/predictions/<run>/` | their **full-scene evaluations**, one directory per job |
| `outputs/predictions/_gt/` | the de-duplicated ground-truth canvases every evaluation links into |
| `outputs/archive/` | every other run and evaluation, mirroring the tree above (`archive/<date>/`, `archive/predictions/`, `archive/screens/`, …) |
| `studies/<study>/` | an HPO study's ledger, runs, evaluations and caches — never touch by hand ([`HPO.md`](HPO.md)) |
| `logs/` | LSF stdout/stderr of `scripts/` jobs |
| `data/scenes_11day`, `data/scenes_all` | symlink farms onto the raw scenes (`scripts/data/link_scenes.py`) |

The run-naming convention and the history of every rename are in
[`outputs/README.md`](../outputs/README.md), which travels with the tree.

## Server-side symlinks: move on WEXAC, not from the Mac

Every `<intf>_gt.npy` in an evaluation directory is a **relative** symlink into
`outputs/predictions/_gt/`, and `data/scenes_*` hold ~1,900 links onto the raw `.unw` / `.ers`
files. The Mac's SMB client cannot see them — it shows the scene folders as empty and the
`_gt.npy` links as plain files — and a symlink made from the Mac is not one the server
understands. So **any move or deletion of an evaluation directory, or of `data/scenes_*`,
runs on WEXAC**, and a move one level deeper re-points the relative links. Count dangling
links before and after (`find -L <dir> -type l | wc -l`).

On the Mac a directory rename also fails with `Permission denied` while Finder, Preview
(an open `curves.png`) or Console (an open `reporter.log`) holds anything inside it;
`lsof +D <dir>` names the app. Deleting files inside a directory is never refused.

## What an evaluation directory holds

| File | Size share | Read by | Keep? |
|---|---|---|---|
| `<intf>_pred.npy` | ~12% | `eval-outputs`, `scripts/eval/rescore.sh`, `sinkholes relabel eval` | **yes** — every re-score reads it |
| `<intf>_gt.npy` | link | `eval-outputs` | **yes** |
| `<intf>_image.npy` | ~75% | only `eval-outputs --save_figures` | no |
| `<intf>_pred_th.npy` | ~12% | nothing | no |
| `olm_results_*.json` | ~0% | you | **yes — this is the result** |
| `*_overview.png`, shapefiles | ~0% | you | yes |

> **Delete `_image.npy` and `_pred_th.npy` freely. Never delete `_pred.npy` or `_gt.npy`
> from an evaluation whose numbers you still quote** — they are what lets `rescore.sh` or
> the relabel re-scoring add a threshold, a tolerance or a corrected label set in minutes,
> where a rebuild is a multi-hour `run_eval.sh`.

Two guards for any script that prunes: refuse a directory with no `olm_results*.json` (an
evaluation still in flight), and leave existing `*_overview.png` alone (only *new* figures
need `_image.npy`; `eval-outputs --save_figures` on a pruned directory exits with a message
rather than mis-scoring).

In a training run, `resume.pt` is only a requeue key. Once a run is finished and filed it is
dead weight — except under `studies/`, where promotions continue from it.

## Filing finished runs

`scripts/tidy_outputs.sh` moves finished runs from the top level of `outputs/` into
`outputs/<date>/` (or `outputs/screens/<date>/` for the screen batches) under their
conventional name. Dry run by default, `--apply` to move. It reads `_valpos`/`_valneg` from
each run's own `reporter.log` and refuses a run with no completion line. Check no evaluation
is queued first: an LSF job carries its `RUN=` path from submit time, so a pending job whose
run directory moves dies on its preflight.

`outputs/predictions/` is a **flat namespace** keyed by run name, with no date: two runs of
the same name cannot both have an evaluation directory. Check for a clash before reusing a
name.
