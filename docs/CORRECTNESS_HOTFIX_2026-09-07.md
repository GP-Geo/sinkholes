# Correctness hotfix — 2026-09-07

## Isolation and review boundary

Base: `25c4dc516be1dc1d650c4c0185c78235631febc1`.
Fix branch: `fix/correctness-hotfix-20260907`.
Local fix worktree: `/private/tmp/sinkholes-correctness-hotfix-20260907`.
Active WEXAC checkout: `/home/labs/rudich/pinkas/sinkholes`, mounted locally at
`/Volumes/rudich/pinkas/sinkholes`, branch `guy_branch_2`.

All fixes and tests were made in the separate local worktree. No cluster jobs
were submitted, queried through SSH, changed, killed, resumed or restarted.
No active shell scripts, Conda environments, dependencies, CUDA settings,
checkpoints, outputs, partitions or real patch data were modified. Existing
local logs were read; the live LSF queue was not available on the mount.

Already-imported Python code normally remains resident, but delayed imports,
spawned workers, pending jobs and especially requeues can read source again.
The templates explicitly `cd /home/labs/rudich/pinkas/sinkholes` and launch
`python -m sinkholes train --resume ...`. Therefore editing the active checkout
would not be safe merely because a job started once. It was left untouched.
Creating the worktree adds Git branch/worktree metadata, not execution-source
changes. Neither the active branch nor its HEAD was switched or merged.

The active checkout has pre-existing, uncommitted gradient diagnostics in
`training/train.py` (`clip_and_record`, additional CSV columns). These are NOT
part of the base commit or this fix branch, and were left intact. When eventually
integrating after jobs are protected/completed, reconcile that diff deliberately:
unscale before its clipping call, and record the now-unscaled norm directly,
without dividing it by the AMP scale again. Do not overwrite the active file
with this worktree's version while jobs can restart from it.

## Fixes and scientific-output classifications

| Files | Fix | Classification / scientific impact |
|---|---|---|
| `inference/reconstruct.py` | Feed the full context to the model; accumulate only its central target footprint into the diagnostic image. Validate consistent input shapes and target-sized logits. | Pure bug fix: target grid, placement, overlap, LiDAR and AOI gates are unchanged. Previously crashing context evaluations now complete. |
| `inference/scenes.py`, `inference/predict.py`, `models/factory.py` | Share checkpoint input/target validation. Infer the context patch tree before discovery. Raw prediction creates context tiles with the same cropped-scene, zero-padding convention as preparation. Reject missing/mismatched context. | Context prediction can change: it previously silently discarded context in `predict`. Ordinary model geometry is unchanged. |
| `inference/patch_test.py`, `inference/attention_probe.py` | Load model geometry/policies before building a patch split; validate pickled input geometry. The attention probe now rejects context rather than silently using plain patches. | Context patch metrics can change. Probe guard is compatibility-only; adding context support to the probe is deferred. |
| `normalise.py`, `dataprep/dataset.py`, `training/train.py` | A 2-D patch is one frame, not H independent rows, under `frame-v2`. Explicit `legacy-row-v1` retains old behavior. Both constructor and `from_arrays` paths are covered. | Can change new single-frame training inputs and patch metrics. Temporal normalization and scene-level range heuristics are unchanged. |
| `training/train.py`, `training/resume.py` | Call `GradScaler.unscale_(optimizer)` once before clipping at every optimizer boundary, including the trailing accumulated group. Record `gradient_clipping=unscaled-v2`. | Can change future AMP optimization and resulting weights. No stored weights are changed. Existing `loss/accum` weighting, including partial groups, is retained. |
| `dataprep/prepare_patches.py` | Merge processed scene indices with existing entries; atomically publish after each completed scene. An explicit unfiltered `--index_mode replace` publishes a deterministic full replacement only on success. | Metadata integrity fix; future regeneration no longer drops untouched scenes. Relative to buggy regeneration, available samples can change. No real regeneration was performed. |
| `polygons.py` | Affine-transform complete Polygon/MultiPolygon geometries, including interior rings. | Scientific GIS output change: hole areas are no longer filled. CRS, coordinate transform, confidence maps and array-based scores are unchanged. |
| `inference/scenes.py` | Write each confidence array once when requested or required by the evaluation source. | Storage/output-only: identical canonical filename and array values. |
| `dataprep/dataset.py`, `inference/patch_test.py` | Coordinate-based AOI selection also covers ordinary single-frame positives-only validation/test and full/null-sampling paths. Reuse the existing single-frame coordinate loader. | Can change sample populations, validation metrics and future selected checkpoints. Historical partitions remain unchanged. |
| `models/factory.py`, `training/train.py`, `training/resume.py` | Store preprocessing/AOI policy in `data_contract` metadata; strip it before loading parameters; validate resume policy. | Compatibility-only protection against silently changing a run's input/sample semantics. |

## Compatibility controls

New training defaults:

- `--preprocessing_version frame-v2`
- `--aoi_selection_version coordinates-v2`
- gradient clipping after unscaling (not selectable back to the buggy AMP path).

Historical patch behavior can be selected explicitly with
`--preprocessing_version legacy-row-v1 --aoi_selection_version legacy-v1`.
An old pickled dataset without a preprocessing-version attribute retains row-wise
behavior; nested `Subset` objects read their underlying dataset's version.

`test-patches` defaults to the saved checkpoint's `data_contract`. Old weights
without it use legacy policies. Explicit overrides warn and are recorded in the
new metrics JSON. A pickled dataset with a conflicting preprocessing policy is
rejected; build a fresh split with the desired policy instead of silently
reinterpreting that pickle.

Historical `test-patches --partition_file` did not pass an AOI for ANY model.
Its legacy command mode retains that behavior. In contrast, historical training
already applied AOI to temporal and ring-negative paths; dataset `legacy-v1`
retains those existing filters and only retains the ordinary single-frame bypass.
The corrected patch command passes the partition's selected split window.

Resume configuration guards the two patch-policy fields, including their known
legacy defaults when old full checkpoints omit them. A legacy AMP full-state
checkpoint is refused on this branch, even with legacy patch settings: continuing
it with corrected clipping would splice different optimizers into one run.
Use the original source checkout for that continuation. Model-only weights can
still initialize an explicitly new experiment through the existing interface.

Scene inference keeps historical per-scene normalization. The new preprocessing
version describes PATCH loading; it does not force old single-frame weights to
use row-wise normalization in `predict` or `eval-scenes`, which never did so.

`eval-scenes --context_margin MY MX` is still accepted and checked. If omitted,
the margin now comes from the checkpoint. Keep the original target `--patch_size`;
300x200 context does not make the prediction grid 300x200. Older checkpoints with
no geometry still use the supplied target patch size as their input size.

## Canonical confidence artifact

Both old save calls used `np.save(prefix + "_pred", ...)`; NumPy appends `.npy`,
so they overwrote the SAME `<intf>_pred.npy`. There were not two historical names
requiring aliases. Consumers found in `inference/outputs.py`, `scripts/eval/rescore.sh`,
`scripts/submit_all.sh`, tests, `docs/OUTPUTS.md`, `docs/PREDICTIONS.md` and
`docs/PIPELINE.md` all use `_pred.npy`. Filenames and source-specific output
selection remain unchanged. No existing output files were removed.

## Historical results to re-check, without declaring them invalid

### Normalization and AOI

Affected normalization paths are `SubsiDataset(temporal=False)`: ordinary nonz
files, full grids, ring-negative single-frame samples, null sampling, spatial
splits, and `from_arrays`/non-overlap splits. This includes single-frame U-Net,
AttentionUNet and U-Net+attention whenever they use that loader. Temporal channel
stacks, including stacked U-Net and sequence models using `temporal=True`, did
not treat rows as channels. The affected range heuristic can differ for rows
whose values happen to fall within the normalized-value range while the full
patch contains wrapped phase. It does not imply every row changed.

Recorded single-frame controls such as `geo_k5_pre2023_single_ring3` and
`temporal_k5_pre2023_single_ring3`, and comparisons against their temporal arms,
are candidates for a corrected, explicitly versioned comparison. Their existing
weights and scene-evaluation preprocessing remain preserved. Single-frame
positives-only validation/test also bypassed AOI even when ring-negative TRAIN
samples used it; the old validation population and resulting model selection
may therefore need checking. No claim is made that recorded rankings are invalid.

### AMP clipping

Direct local evidence: September 7 logs/CSVs for jobs `580750` (long500), `580752`
(ctx50 60 epochs), and `580753` (ctx50 200 epochs) show an initial AMP scale of
65,536 and clipping on 100% of optimizer steps. The uncommitted diagnostics
explicitly measure clipping BEFORE unscaling. These are observations of the
old path, not estimates of the effect on final F1. These jobs were untouched.

Source history places the same ordering in `c2d093a` (2026-07-29 package rewrite)
and even `8ce9220` (2024-05-06 `train_sinkholes_unet.py`). Accumulation commit
`25c4dc5` added a second optimizer boundary with the same ordering.

Template evidence: commit `33c11fa` includes `--amp` in all three training
wrappers. `scripts/train/PRESETS.md` states AMP is shared by the presets;
`docs/EXPERIMENTS.md` records completed negative-sampling, attention/control,
clean22, valpos/attnfix, attnpos, pre23 and long200 batches. These documented
AMP-template runs likely used the old clipping behavior. Individual source hashes
and AMP state are not retained for every historical run, so this is not proof
about every checkpoint. Runs without AMP are not affected by scaled clipping.

Corrected AMP runs should form a new, explicitly identified comparison series;
do not silently continue a legacy AMP run on the fixed code. No retraining or
rescoring of real experiments was performed.

## Tests and validation

New regression files:

- `tests/test_context_scene_hotfix.py`: ordinary 200x100 and 300x200->200x100
  reconstruction; exact synthetic target alignment; uniform/coverage/vote/Hann
  overlaps; LiDAR/AOI invariance; raw tiler vs preparation; synthetic scene and
  predict CLI executions; canonical output count across sources/flags.
- `tests/test_preprocessing_aoi_hotfix.py`: equivalent frame normalization;
  actual coordinate selection for train/val/test; explicit legacy modes;
  pickled dataset and Subset compatibility; checkpoint metadata and resume guards;
  patch-test command geometry/policy resolution.
- `tests/test_amp_clipping_hotfix.py`: actual training optimizer boundaries with
  an enabled CPU GradScaler at scale 65,536, gradients below/above clipping
  threshold, accumulation and a trailing group. Checks gradients both before
  clipping and at optimizer.step, not just call order.
- `tests/test_patch_index_hotfix.py`: real preparation on temporary synthetic
  scenes, subset preservation, deterministic full replacement, malformed-index
  protection, atomic publish failure, and progress surviving a later scene error.
- `tests/test_polygon_holes_hotfix.py`: donut Polygon and MultiPolygon transform
  plus real shapefile export/reload, checking interiors, area, bounds and CRS.

Baseline in the local sandbox: 394 passed, 6 skipped, 13 failed because the sandbox
blocked PyTorch's local `torch_shm_manager`. The full suite was then run locally
with shared-memory permission, without changing dependencies or WEXAC settings.
Final results are recorded in the delivery message. Tests use temporary data and
small CPU training fixtures; no full WEXAC training experiment was run.

## Deliberately unchanged / deployment follow-up

No architecture relocation, launcher rewrite, partition consolidation, new data
storage format, dataset pruning, output cleanup or module renaming. Existing
large diagnostic outputs remain enabled; only their duplicate confidence save
was removed. Losses, optimizer hyperparameters, scheduler, checkpoint selection
metric, accumulation weighting, stride, frame origins and LiDAR rules are unchanged.

Before future experiments: reconcile the active diagnostic diff only when the
execution checkout is safe to change; use the corrected policy labels; retain a
separate legacy source for ongoing AMP resumes. No fixes are merged or deployed
by this task. The temporary worktree is a review workspace; the committed branch
lives in the shared repository and should be checked out separately for cluster
use after review, without switching the active jobs' checkout.

The context patch tree currently observed on WEXAC is stride 2. A context
checkpoint does not create a missing stride-4 context tree; a future stride-4
context evaluation needs matching prepared inputs. No new tree was generated.
The attention probe intentionally rejects context checkpoints until its own
context-aware sampling is implemented. Its standard path is unchanged.

Atomic index publication is not a multi-file dataset transaction and does not
make simultaneous regeneration/training safe. Regeneration remains a single-writer
operation against a dataset that is not being used by active jobs.
