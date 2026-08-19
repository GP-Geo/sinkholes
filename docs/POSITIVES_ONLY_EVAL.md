# Positives-only evaluation

**Status:** implemented 2026-08-16. Both levels ship; see [Running it](#running-it).

## Goal

The paper this work is benchmarked against evaluates **only on patches that contain
subsidence** — the task it poses is "delineate subsidence in areas known to have
subsidence", not "find subsidence anywhere on the map". To compare like for like we
need the same protocol: patch-level Dice + object-level precision/recall computed over
positive patches only.

This is **for baseline comparison only**. It is not a model-selection metric — see
[Why this is not the default](#why-this-is-not-the-default) at the end.

## The problem it solved

`sinkholes test-patches` already implemented exactly this protocol, but it could only read
a **pickled test split**, and the runs we care about have none. In `build_datasets`:

```python
elif args.partition_mode == "preset_by_intf":
    train_list, val_list = load_preset_partition(...)
    test_list = []          # <-- always empty
```

`test_list = []` leaves `test_set` as `None`, so the `save_test_dataset` call in `main`
never runs and no `.pkl` is written. Every run trained with
`--partition_mode preset_by_intf` — everything driven by a partition JSON, i.e. the whole
`geo_*` / `temporal_*` family — is affected:

```
test intfs (0): []
partition: preset_by_intf; train/val/test = 4713/1681/0 samples
```

Retraining to produce the pickles was never viable: there are many such runs and the
checkpoints already exist. So the split is now built **in place** from the partition JSON
instead. Note the partition files do carry a real `"test"` list
(`assets/partition_geo_k10.json` is 91/17/18) — training simply drops it, and evaluation is
where it is finally read.

## Running it

### Patch level

```bash
sinkholes test-patches \
  --model outputs/2026-08-06/geo_k10_convlstm_posw8/checkpoints/best.pt \
  --partition_file assets/partition_geo_k10.json \
  --split test \
  --patches_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches \
  --add_temporal --k_prevs 10 --convlstm_unet \
  --patch_size 200 100 --stride 2 \
  --th 0.7 --b 5 \
  --metrics_out positives_only_geo_k10_test.json
```

`--split` takes `val`, `test` or `train`. `val` is the split a `preset_by_intf` model was
selected on; `test` is ground it has never seen. Both are honest, neither is a selection
metric. A split the JSON does not hold is an error — the `*_testeval.json` variants carry
their parent's test list under `"val"` and have no `"test"` of their own.

`--test_data_path` still works and is unchanged; exactly one of the two sources must be
given. `--save_test_dataset PATH` pickles the built split so a rescore (or `inspect-run`,
which reads pickles) can use `--test_data_path` instead of rebuilding it.

### Scene level

```bash
sinkholes eval-scenes ... --positives_only
```

Predicts only tiles whose ground truth is non-empty, ANDed with the LiDAR gate. The tile
set is derived from the ground-truth grid `eval-scenes` already builds, so it follows
`--unioned_mask` automatically. `scripts/eval/run_eval_positives.sh` is the wexac
submission wrapper; it drives the same `GROUP`/`SPLIT`/`ARCH` config as `run_eval.sh` and
runs `eval-outputs` afterwards.

## Recovering each run's settings

Training does **not** dump `vars(args)` (unlike `eval-scenes`, which does). Recover from
the run's `.log` header, which records:

- `patch directories: .../data_patches_H200_W100_strpp2_11days_Aligned | ...`
  → gives `--patch_size`, `--stride`, `--train_on_11d_diff`, `--use_cleaned_patches`
- `sequence length T=4` → `--k_prevs 3` (T = k_prevs + 1)
- `partition: preset_by_intf` and the `val intfs (N): [...]` list
  → cross-check against the partition JSON

The architecture flag (`--convlstm_unet` etc.) is recoverable from the checkpoint itself
via `build_from_checkpoint`, so a mismatch surfaces as a load error rather than a silently
wrong number.

## How it is built

`test-patches` mirrors `train.build_datasets` rather than reimplementing it: the same
`resolve_patch_dirs` (shared, in `dataprep/patchify.py`, so the directory naming has one
implementation), the same `find_11day_sequences` chain filter, and the same `SubsiDataset`
arguments. `mode="test"` is what keeps the set positives-only — it is the mode that takes
neither ring nor validation negatives. Everything from the `DataLoader` on is untouched,
so `evaluate(..., mode="test")` computes the number exactly as it always did.

`tests/test_positives_only_eval.py` pins the equivalence: a split built from a partition
must hand `evaluate` byte-identical samples to one that was pickled, single-frame and
temporal.

## Gotchas

- **Temporal runs drop interferograms without full chains.** The build logs a warning
  naming them. On a `geo_*` / `_compatible` partition it must drop nothing — training on
  that partition would have crashed otherwise — so a warning means `--k_prevs` or
  `--partition_file` is wrong for that checkpoint.
- **Positive coords are unioned over the stack** for `--union_temporal_mask` runs: a patch
  positive at *any* timestep is included. Match whatever training did.
- At scene level the confidence of predictions spilling *outside* the positive tiles is
  attenuated (`average="uniform"` divides by `stride**2` regardless of how many tiles
  contributed). Recall is unaffected — a tile holding a ground-truth pixel is positive by
  definition, so every such pixel keeps all of its overlapping tiles.
- Do not change `nonz_only`'s default anywhere. `--positives_only` is off by default and
  full-scene evaluation stays the norm.

## Why this is not the default

`docs/RESULTS.md:50` — models trained only on positive patches (~2.5% of the map) and
then applied to all of it "found nearly everything and flagged far too much". Ring
negatives fixed it: precision +0.10 geo, +0.28 temporal, the project's largest real gain.
`RESULTS.md:85-87` explains that Dice ranked those runs *backwards* precisely because
validation stays positives-only, so the suppressed false positives are invisible to it.

**So: use these numbers to compare against the paper, and never to choose a model.**
When reporting, state the protocol explicitly, because a positives-only precision is not
comparable to a full-scene precision. Both code paths say so in their logs.
