# The 11-day dataset

`assets/intf_coord.json` describes **exactly one thing**: the 11-day
interferograms that training uses. It is not a general index of the archive.

Any other duration gets its own file (`intf_coord_44day.json`, …) passed with
`--intf_dict_path` / `--intf_dict`. Durations are never merged into one
dictionary — a mixed file cannot be checked against any single patch tree,
which is how the previous 716-entry version drifted out of sync unnoticed.

## Current contents

| | |
|---|---|
| interferograms | 437 |
| duration | 11 days, all |
| frames | North 229, South 208 |
| years | 2019–2026 |
| labelled (GT `sub_20260701.shp`) | 273, holding 30,729 objects |
| unlabelled | 164 (undigitised; empty masks, 0 positive patches) |
| positive patches | 78,935 |
| `lidar_mask: no_mask` | 112 (from 20240615_20240626; `lidar_intf_mask.txt` stops at 20240605_20240616) |

Matching patch tree:
`deadsea_sinkholes_data/patches/{data,mask}_patches_H200_W100_strpp2_11days_Aligned`

## Scope rules

An interferogram belongs here when all of these hold:

1. **11-day span** — start and end dates exactly 11 days apart.
2. **The `tgeo_int_*` copy**, from the `deadsea_sinkholes_data` root. Scenes
   also exist as `tgeo_ccw_*` and under `004/`/`013/`, but those have different
   raster extents for most ids. The committed dictionaries have always
   described the root `int` copies; mixing them silently misaligns patches.
3. **Alignable to its frame origin** — the raster must contain the fixed
   `FRAME_ORIGINS` point for its frame, or `crop_to_start_xy` raises.

Rule 3 currently excludes 9 scenes, all 2026, all undigitised:

```
20260418_20260429  North   20260601_20260612  North
20260429_20260510  North   20260612_20260623  North
20260510_20260521  North   20260623_20260704  North
20260521_20260601  North   20260715_20260726  North
20260520_20260531  South
```

The 8 North scenes start 1–21 pixels south of latitude 31.79; the South one
starts ~1,998 pixels (~6 km) south of 31.44. As more 2026 data arrives this
drift may continue and will need a real decision — extend the frames, or treat
2026+ as a new frame definition. `FRAME_ORIGINS` must not be edited casually:
it shifts every patch grid and every exported polygon.

## Rebuilding

```bash
D=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data

# 1. one directory, one file per interferogram (skips unalignable scenes)
python scripts/data/link_scenes.py --data_dir $D --out_dir data/scenes_11day --clear

# 2. the dictionary
python -m sinkholes prepare-metadata --intf_dir data/scenes_11day --out_path assets/intf_coord.json

# 3. patches (hours; submit it)
bsub < scripts/data/regenerate_patches.sh
```

`regenerate_patches.sh` refuses to start unless the dictionary matches the
scene directory exactly, then runs `prepare-patches`, `count-positives` and
`verify_dataset.py` in sequence.

If `prepare-patches` dies after writing arrays but before
`nonz_indices.json` (it is written only after the last interferogram), recover
without redoing patch generation:

```bash
bsub < scripts/data/finish_regeneration.sh
```

## Verifying

```bash
python scripts/data/verify_dataset.py --intf_dict assets/intf_coord.json \
  --scene_dir data/scenes_11day --patches_root $D/patches \
  --days_diff 11 --gt_polygon_file_path $D/sub_20260701.shp
```

Checks the dictionary's own consistency, its geometry against the `.ers`
headers, the patch tree's pairing, that `nonz_indices.json` and `nonz_num`
both match the arrays, and that no labelled interferogram has an empty mask.
Exits non-zero on any failure, so it can gate a pipeline.

## Other dictionaries on the volume

- `Rudich_Collaboration/sinkholes/intf_coord.json` — 1,141 entries, the
  complete scene inventory across all durations, from the pre-package version
  of this project. Useful as a record of what exists; its `nonz_num` values are
  stale and it says nothing about alignability or patch coverage.
- `deadsea_sinkholes_data/intf_coord.json` — 716 entries across all durations
  (299 of the 11-day, plus 44/77/33/22-day and longer). **Deliberately not
  overwritten** by the 11-day regeneration: it is the shared tree's own record
  and other people read it. It is not this dataset and must not be passed to
  the 11-day commands — the 437-entry `assets/intf_coord.json` is the only
  dictionary that describes the 2-stride 11-day patch tree.

## Known stale

- **Six stride-4 masks predate the current GT.** `strpp4_11days` holds all 437 ids, but its
  first 331 were cut on 2026-06-10 from the previous GT file, `sub_20231001.shp` (last date
  2023-09-15). Those 331 match the current GT except six scenes the old file had not labelled:
  20230801_20230812, 20231029_20231109, 20231222_20240102, 20231223_20240103,
  20240102_20240113 and 20240103_20240114 (54–207 polygons each now). Their stride-4 masks are
  empty, so an RTh evaluation scores them as scenes without ground truth. They are training
  scenes (stride 2, correct) in every kept partition except 5-t2026, whose val holds three of
  them ([`HPO.md`](HPO.md) §3). Rebuild them after the HPO study: remove the six ids'
  stride-4 data and mask grids and rerun `scripts/data/make_stride4_patches.sh`, which fills
  every id the stride-4 tree lacks.
- `strpp4_77days` was built from `sub_20231001.shp` and does not match this dictionary.
