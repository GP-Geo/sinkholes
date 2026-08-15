# Positives-only evaluation for `preset_by_intf` runs

**Status:** not implemented. Handoff brief — written 2026-08-15, to be actioned on wexac.

## Goal

The paper this work is benchmarked against evaluates **only on patches that contain
subsidence** — the task it poses is "delineate subsidence in areas known to have
subsidence", not "find subsidence anywhere on the map". To compare like for like we
need the same protocol: patch-level Dice + object-level precision/recall computed over
positive patches only.

This is **for baseline comparison only**. It is not a model-selection metric — see
[Why this is not the default](#why-this-is-not-the-default) at the end.

## The problem

`sinkholes test-patches` already implements exactly this protocol, but it can only read a
**pickled test split**, and the runs we care about don't have one.

Root cause, `sinkholes/training/train.py:276-281`:

```python
elif args.partition_mode == "preset_by_intf":
    train_list, val_list = load_preset_partition(...)
    test_list = []          # <-- always empty
    mode = "preset_by_intf"
```

`test_list = []` makes `test_set` `None` at `train.py:298-299`, so the
`save_test_dataset` call at `train.py:636` never runs and no `.pkl` is written.

Every run trained with `--partition_mode preset_by_intf` (i.e. everything driven by a
partition JSON, including the `k10split` family) is affected. Confirmed in run logs:

```
test intfs (0): []
partition: preset_by_intf; train/val/test = 4713/1681/0 samples
```

Retraining to produce the pickles is not viable — there are many such runs and the
checkpoints already exist.

## What already exists — do not rebuild any of this

| Piece | Location | Note |
|---|---|---|
| Positives-only selection | `dataprep/dataset.py` `nonz_only=True` | default; the val/test sets already use it |
| Patch metrics | `training/evaluate.py` `mode="test"` | Dice + pixel P/R + object-level P/R |
| Object matching | `evaluate.py:29-50` `object_level_evaluate` | GT object detected when overlap with prediction buffered by `--b` exceeds `--th` |
| CLI wrapper | `inference/patch_test.py` | only the dataset *source* is wrong |
| Partition loader | `dataprep/partition.py:103` `load_preset_partition` | returns `(train, val)` |

The metric code is correct and tested. **The only missing piece is a way to hand
`test-patches` a dataset built from a partition JSON instead of a pickle.** Reusing
`evaluate()` unchanged is what guarantees our number is computed the same way as the
existing ones.

## The change to make

Extend `sinkholes/inference/patch_test.py` so the split can be constructed in place.

### 1. Make `--test_data_path` optional, add an alternative source

New flags, mirroring `train.py` names exactly so they can be copy-pasted from a run log:

```
--partition_file PATH     partition JSON to evaluate
--split {val,train}       which list to use (default: val)
--patches_dir PATH        root holding the {data,mask}_patches_* trees
--patch_size H W          default 200 100
--stride INT              default 2
--train_on_11d_diff       BooleanOptionalAction, default True
--use_cleaned_patches
--add_temporal
--k_prevs INT
--treat_nodata_regions
--union_temporal_mask
--intf_dict_path PATH
--nonz_only               BooleanOptionalAction, default True
```

Require exactly one of `--test_data_path` / `--partition_file`.

### 2. Build the dataset the same way training does

Mirror `train.py:210-250`. The directory naming **must** match or the files won't be
found:

```python
H, W = args.patch_size
days = 11 if args.train_on_11d_diff else None
image_dir = os.path.join(args.patches_dir, patch_dir_name("data", H, W, args.stride, days))
mask_dir  = os.path.join(args.patches_dir, patch_dir_name("mask", H, W, args.stride, days))
if args.use_cleaned_patches:
    image_dir = os.path.join(image_dir, "cleaned")
    mask_dir  = os.path.join(mask_dir,  "cleaned")

train_list, val_list = load_preset_partition(args.partition_file)
intf_list = val_list if args.split == "val" else train_list

seq_dict = None
if args.add_temporal:
    coord_dict = load_coord_dict(args.intf_dict_path)
    seq_dict, intf_list = find_11day_sequences(
        coord_dict, k_prev=args.k_prevs, restrict_to=intf_list)

ds = SubsiDataset(
    image_dir, mask_dir, intf_list, mode="test",
    patch_size=(H, W), stride=args.stride,
    nonz_only=args.nonz_only,
    temporal=args.add_temporal, seq_dict=seq_dict,
    treat_nodata_regions=args.treat_nodata_regions,
    union_temporal_mask=args.union_temporal_mask,
    use_cleaned_patches=args.use_cleaned_patches,
)
```

`SubsiDataset`'s full signature is at `dataprep/dataset.py:96-116`. Note `mode="test"`
matters: it is what suppresses ring negatives and selects test behaviour.

### 3. Leave the rest of `main()` alone

The existing `DataLoader` → `build_from_checkpoint` → `evaluate(..., mode="test")` path
at `patch_test.py:30-64` needs no changes.

### Expected invocation

```bash
sinkholes test-patches \
  --model outputs/convlstm_10prev_h256_b64_lr5e6_40e_k10split_2026-08-03_17h19/checkpoints/best.pt \
  --partition_file assets/partition_20_05_16h53_k10_compatible.json \
  --split val \
  --patches_dir <patch root> \
  --add_temporal --k_prevs 10 --convlstm_unet \
  --patch_size 200 100 --stride 2 \
  --th 0.7 --b 5
```

## Recovering each run's settings

Training does **not** dump `vars(args)` (unlike `eval-scenes`, which does at
`scenes.py:156`). Recover from the run's `.log` header, which records:

- `patch directories: .../data_patches_H200_W100_strpp2_11days_Aligned | ...`
  → gives `--patch_size`, `--stride`, `--train_on_11d_diff`, `--use_cleaned_patches`
- `sequence length T=4` → `--k_prevs 3` (T = k_prevs + 1)
- `partition: preset_by_intf` and the `val intfs (N): [...]` list
  → cross-check against the partition JSON

The architecture flag (`--convlstm_unet` etc.) is recoverable from the checkpoint itself
via `build_from_checkpoint`, so a mismatch will surface as a load error rather than a
silently wrong number.

## Verification

1. **Regression anchor** — pick one of the four runs that *does* have a `.pkl`
   (`random_by_intf` runs, e.g. `convlstm_v1_2026-07-28_22h12`). Run `test-patches` both
   ways: via `--test_data_path` and via `--partition_file` with the same interferogram
   list. **The Dice must match exactly.** This proves the in-place build is equivalent to
   the pickled one.
2. `nonz_indices.json` must cover every evaluated id — `scripts/verify_dataset.py:136-159`
   already checks this.
3. Sanity: the reported patch count should be far smaller than the full grid
   (positives are ~2.5% of the map).

## Gotchas

- **`--split val` is the right choice**, not train. For `preset_by_intf` the val list is
  the held-out data; the paper's numbers are on held-out ground.
- **Temporal runs need `seq_dict`.** `SubsiDataset` raises if `temporal=True` and
  `seq_dict is None` (`dataset.py:127`). `find_11day_sequences` may also *drop*
  interferograms lacking full chains — log how many, since it changes the denominator.
- **Positive coords are unioned over the stack** for temporal runs
  (`dataset.py:228-236`): a patch positive at *any* timestep is included. This is what
  training did, so it is the correct comparison, but it is not the same as "positive in
  the current frame".
- Do not change `nonz_only`'s default anywhere, and do not touch `eval-scenes` —
  full-scene evaluation must stay as it is (see below).

## Optional follow-up: positives-only at scene level

Only needed if georeferenced polygons under the paper protocol are wanted. `eval-scenes`
predicts every tile — `inference/reconstruct.py:159-162` — with exactly one skip
condition, the LiDAR gate at line 173. `nonz_indices.json` is keyed by the same `(i,j)`
grid coordinates as that loop, so it would be one extra skip alongside it, plus a flag
plumbed through `scenes.py`. Deliberately kept separate from the change above.

## Why this is not the default

`docs/RESULTS.md:50` — models trained only on positive patches (~2.5% of the map) and
then applied to all of it "found nearly everything and flagged far too much". Ring
negatives fixed it: precision +0.10 geo, +0.28 temporal, the project's largest real gain.
`RESULTS.md:85-87` explains that Dice ranked those runs *backwards* precisely because
validation stays positives-only, so the suppressed false positives are invisible to it.

**So: use these numbers to compare against the paper, and never to choose a model.**
When reporting, state the protocol explicitly, because a positives-only precision is not
comparable to a full-scene precision.
