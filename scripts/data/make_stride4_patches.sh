#!/bin/bash
#BSUB -J make_strpp4_missing
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/strpp4_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/strpp4_%J.err
#BSUB -q short
#BSUB -R rusage[mem=70GB]
#BSUB -W 24:00
#
# Fill the STRIDE-4, 11-DAY patch tree: generate data and mask grids for every
# 11-day interferogram that has stride-2 patches but no stride-4 pair.
#
# Submit:  bsub < scripts/data/make_stride4_patches.sh   (from the repo root)
#
# ---- why this exists -------------------------------------------------------
# Six of the eight eval jobs on 2026-08-19 died on:
#   FileNotFoundError: .../data_patches_H200_W100_strpp4_11days_Aligned/
#                      data_patches_20240319_20240330_H200_W100_strpp4.npy
# The strpp4 tree was built before the 2024-2026 acquisitions were added, so it
# holds 331 of the 437 interferograms the strpp2 tree has. The gap is 106, all
# of them 2024 (48), 2025 (44) and 2026 (14). Every one is already in
# assets/intf_coord.json and already patched at stride 2, so they align and cut
# cleanly -- verified 2026-08-19.
#
# ---- 11-DAY ONLY -----------------------------------------------------------
# The gap is computed against the strpp2 ELEVEN-DAY tree and --days_diff 11 is
# passed to prepare-patches, which skips any id of another duration. The
# separate strpp4_77days tree is a different product and is not touched.
#
# ---- the trap in --by_list, and why step (3) exists ------------------------
# prepare-patches accumulates positive-patch coordinates for the ids it
# processed IN THIS RUN and then overwrites nonz_indices.json with just those
# (prepare_patches.py:117-120). A --by_list run would replace the 331-entry
# index with a 106-entry one. eval-scenes does not read that file (scenes.py:315
# derives the positives from the gt grid instead), so the evals would still
# work -- but training and verify_dataset.py do read it, and a silently
# truncated index is the kind of thing found months later. Step (3) rebuilds it
# for the whole tree from the mask grids on disk.
#
# ---- what is deliberately NOT here -----------------------------------------
# `sinkholes count-positives` is in regenerate_patches.sh but must NOT run
# against the strpp4 tree: it writes positive-patch counts into
# assets/intf_coord.json and those counts are stride-dependent. Running it here
# would overwrite the stride-2 counts every partition and training run uses.
#
# ---- sizing ----------------------------------------------------------------
# DISK: the existing 331 ids occupy 1.7 TB of data + 428 GB of mask, i.e.
# ~6.4 GB per interferogram across both trees. 106 more is ~680 GB. There was
# 13 TB free on the mount on 2026-08-19; the script re-checks before starting.
# MEMORY: prepare-patches holds one interferogram at a time. The stride-2 regen
# measured a 5.2 GB peak (logs/regen_patches_826816.out); stride 4 quadruples
# the patch arrays, so ~21 GB. 70 GB is the same request the stride-2 job used
# and leaves a wide margin.
# WALLTIME: no clean per-scene measurement survives -- both stride-2 regen jobs
# died early. 106 scenes at 4x the stride-2 work is roughly one full stride-2
# regeneration, which was budgeted 24:00. Writing ~680 GB is likely the real
# floor. If it hits the wall, rerun: the gap is recomputed from what is on disk,
# so a second run picks up exactly where the first stopped.

set -o pipefail

REPO=/home/labs/rudich/pinkas/sinkholes
DATA=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data
SCENES=/home/labs/rudich/pinkas/scenes_11day
OUT=$DATA/patches
GT=$DATA/sub_20260701.shp
DICT=$REPO/assets/intf_coord.json

cd "$REPO"
source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# (0) preconditions ----------------------------------------------------------
# The scene directory is symlinks built by link_scenes.py and is easy to lose.
# Fail here rather than after an hour: with no .unw to match, prepare-patches
# writes nothing and EXITS SUCCESSFULLY, which looks like the job worked.
n_unw=$(ls -1 "$SCENES"/*.unw 2>/dev/null | wc -l)
echo "scene directory $SCENES: $n_unw .unw files"
if [ "$n_unw" -eq 0 ]; then
  echo "!! empty. Rebuild it first (the raw scenes are in $DATA):" >&2
  echo "   python scripts/data/link_scenes.py --data_dir $DATA \\" >&2
  echo "       --out_dir $SCENES --days_diff 11 --variant int --clear" >&2
  exit 1
fi

avail_gb=$(df -Pk "$OUT" | awk 'NR==2 {print int($4/1048576)}')
echo "free space on the patches mount: ${avail_gb} GB"
[ "$avail_gb" -ge 800 ] || { echo "!! under 800 GB free; this needs ~680 GB" >&2; exit 1; }

# (1) work out the gap from what is actually on disk --------------------------
# Computed rather than hard-coded so a rerun after a walltime kill does only
# what is still missing. An id counts as done only when BOTH its data and mask
# grids exist -- a half-written pair must be redone, not skipped.
BY_LIST=$(python - "$OUT" "$DICT" <<'PY'
import glob, json, os, sys
out, dict_path = sys.argv[1], sys.argv[2]
def ids(tree, kind, st):
    pre, suf = f"{kind}_patches_", f"_H200_W100_strpp{st}.npy"
    return {os.path.basename(f)[len(pre):-len(suf)]
            for f in glob.glob(f"{out}/{tree}/{kind}_patches_2*.npy")}
s2 = ids("data_patches_H200_W100_strpp2_11days_Aligned", "data", 2)
d4 = ids("data_patches_H200_W100_strpp4_11days_Aligned", "data", 4)
m4 = ids("mask_patches_H200_W100_strpp4_11days_Aligned", "mask", 4)
missing = sorted(s2 - (d4 & m4))
meta = set(json.load(open(dict_path)))
absent = [i for i in missing if i not in meta]
if absent:
    print(f"!! {len(absent)} ids are not in the dictionary: {absent[:5]}", file=sys.stderr)
    print("!! run 'sinkholes prepare-metadata' first", file=sys.stderr)
    raise SystemExit(1)
if len(missing) > 200:
    print(f"!! {len(missing)} missing -- more than expected; check the trees by hand",
          file=sys.stderr)
    raise SystemExit(1)
print(f"stride-2: {len(s2)}   stride-4 complete: {len(d4 & m4)}   to build: {len(missing)}",
      file=sys.stderr)
print(",".join(missing))
PY
)
if [ -z "$BY_LIST" ]; then
  echo "nothing missing -- the stride-4 11-day tree is already complete."
  exit 0
fi
echo "building: $BY_LIST"
date

# (2) the grids --------------------------------------------------------------
python -m sinkholes prepare-patches \
  --input_dir "$SCENES" \
  --output_dir "$OUT" \
  --gt_polygon_file_path "$GT" \
  --patch_size 200 100 \
  --strides_per_patch 4 \
  --days_diff 11 \
  --by_list "$BY_LIST" \
  --intf_dict_path "$DICT"
date

# (3) repair nonz_indices.json for the WHOLE strpp4 tree ---------------------
python scripts/data/rebuild_nonz_indices.py \
  --patches_root "$OUT" \
  --patch_size 200 100 \
  --strides_per_patch 4 \
  --days_diff 11
date

# (4) verify -----------------------------------------------------------------
# ONE CHECK IS EXPECTED TO FAIL and it is not a problem:
#   "dictionary nonz_num matches the arrays"  -- nonz_num in intf_coord.json is
#   the STRIDE-2 positive count, and a stride-4 grid has ~4x the patches, so
#   every id differs. This is exactly why count-positives must not be run here.
#
# What must PASS once the tree is complete: "patch ids == dictionary ids" (both
# 437 after this run), the id-pairing checks (data/mask, full/nonz),
# "nonz_indices covers every patched id", and "nonz_indices lengths match the
# arrays". Those are the ones that catch a half-written tree.
python scripts/data/verify_dataset.py \
  --intf_dict "$DICT" \
  --scene_dir "$SCENES" \
  --patches_root "$OUT" \
  --patch_size 200 100 \
  --strides_per_patch 4 \
  --days_diff 11 \
  --gt_polygon_file_path "$GT"

echo "done: stride-4 11-day tree filled"
echo "next: bash scripts/submit_all.sh eval5           # dry run"
echo "      bash scripts/submit_all.sh eval5 --submit"
