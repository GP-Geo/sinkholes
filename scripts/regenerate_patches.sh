#!/bin/bash
#BSUB -J regen_patches_11day
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/regen_patches_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/regen_patches_%J.err
#BSUB -q short
#BSUB -R rusage[mem=70GB]
#BSUB -W 24:00
#
# Regenerate the 11-day, 2-stride patch tree IN PLACE from the current ground
# truth, refill the positive-patch counts, and verify the result.
#
# Assumes scripts/link_scenes.py has already built the scene directory:
#   python scripts/link_scenes.py --data_dir <data> --out_dir <scenes> --days_diff 11 --variant int --clear
#
# Submit:  bsub < scripts/regenerate_patches.sh   (from /home/labs/rudich/pinkas/sinkholes)
#
# Scheduler stdout/stderr land in logs/regen_patches_<jobid>.{out,err}.
#
# NOTE: this overwrites the existing tree. prepare-patches writes
# nonz_indices.json only at the very end, so a job that dies partway leaves the
# arrays and that file inconsistent — re-run this script from the start rather
# than resuming. Back up the JSON side-cars before submitting.

set -o pipefail

REPO=/home/labs/rudich/pinkas/sinkholes
DATA=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data
SCENES=/home/labs/rudich/pinkas/scenes_11day
OUT=$DATA/patches
GT=$DATA/sub_20260701.shp
DICT=$REPO/assets/intf_coord.json
TREE=$OUT/data_patches_H200_W100_strpp2_11days_Aligned

cd "$REPO"
source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# (0) preconditions — refuse to touch the tree unless the dictionary covers
# every scene. prepare-patches would otherwise run for hours and then die on a
# KeyError at the first unregistered id, leaving new arrays beside a stale
# nonz_indices.json.
python - "$SCENES" "$DICT" <<'PY'
import json, os, sys
scenes, dict_path = sys.argv[1], sys.argv[2]
meta = json.load(open(dict_path))
ids = {f.split(".")[0][9:17] + f.split(".")[0][24:33]
       for f in os.listdir(scenes) if f.endswith(".unw")}
missing = sorted(ids - set(meta))
extra = sorted(set(meta) - ids)
print(f"=== scenes: {len(ids)}   dictionary: {len(meta)} entries ===")
if missing or extra:
    print(f"!! {len(missing)} scenes not in the dictionary: {missing[:5]}")
    print(f"!! {len(extra)} dictionary entries with no scene: {extra[:5]}")
    print("!! run 'sinkholes prepare-metadata' over the scene directory first")
    raise SystemExit(1)
print("precondition ok: dictionary matches the scene directory exactly")
PY
date

# (1) patches — the long step ------------------------------------------------
python -m sinkholes prepare-patches \
  --input_dir "$SCENES" \
  --output_dir "$OUT" \
  --gt_polygon_file_path "$GT" \
  --patch_size 200 100 \
  --strides_per_patch 2 \
  --days_diff 11 \
  --intf_dict_path "$DICT"
date

# (2) positive-patch counts back into the dictionary --------------------------
python -m sinkholes count-positives \
  --input_patch_dir "$TREE" \
  --intf_dict "$DICT" \
  --out_path "$DICT"

# (3) verify: dictionary <-> headers <-> patch tree ---------------------------
python scripts/verify_dataset.py \
  --intf_dict "$DICT" \
  --scene_dir "$SCENES" \
  --patches_root "$OUT" \
  --patch_size 200 100 \
  --strides_per_patch 2 \
  --days_diff 11 \
  --gt_polygon_file_path "$GT"

echo "done: patch tree regenerated and verified"
echo "next, by hand:  cp $DICT $DATA/intf_coord.json"
