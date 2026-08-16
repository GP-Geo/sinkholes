#!/bin/bash
#BSUB -J finish_regen_11day
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/finish_regen_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/finish_regen_%J.err
#BSUB -q short
#BSUB -R rusage[mem=6GB]
#BSUB -W 4:00
#
# Finish a patch regeneration whose prepare-patches step already wrote every
# array but died before writing nonz_indices.json (it is written only after the
# last interferogram, so any crash loses it).
#
# Does NOT re-run patch generation. It rebuilds the index from the mask grids
# on disk, refills the dictionary's positive-patch counts, and verifies the
# three sources agree.
#
# Requires the scene directory to hold exactly the alignable scenes:
#   python scripts/data/link_scenes.py --data_dir <data> --out_dir <scenes> --clear
#
# Submit:  bsub < scripts/data/finish_regeneration.sh

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

echo "=== scenes: $(ls "$SCENES"/*.unw | wc -l), dictionary: $(python -c "import json;print(len(json.load(open('$DICT'))))") entries ==="
date

# (1) rebuild the positive-patch index from the mask grids --------------------
python scripts/data/rebuild_nonz_indices.py \
  --patches_root "$OUT" \
  --patch_size 200 100 \
  --strides_per_patch 2 \
  --days_diff 11
date

# (2) positive-patch counts back into the dictionary --------------------------
python -m sinkholes count-positives \
  --input_patch_dir "$TREE" \
  --intf_dict "$DICT" \
  --out_path "$DICT"

# (3) verify: dictionary <-> headers <-> patch tree ---------------------------
python scripts/data/verify_dataset.py \
  --intf_dict "$DICT" \
  --scene_dir "$SCENES" \
  --patches_root "$OUT" \
  --patch_size 200 100 \
  --strides_per_patch 2 \
  --days_diff 11 \
  --gt_polygon_file_path "$GT"

echo "done: index rebuilt, counts refilled, dataset verified"
echo "next, by hand:  cp $DICT $DATA/intf_coord.json"
