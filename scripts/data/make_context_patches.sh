#!/bin/bash
#BSUB -J make_ctx50_patches
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/ctx50_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/ctx50_%J.err
#BSUB -q short
#BSUB -R rusage[mem=64GB]
#BSUB -W 24:00
#
# Build the LARGE-CONTEXT patch tree: 300x200 data cells on the EXISTING
# 200x100 stride-2 grid, for the overlap-tile experiment (predict the centre,
# see the surroundings).
#
# Submit:  bsub < scripts/data/make_context_patches.sh    (from the repo root)
#
# ---- what this produces ----------------------------------------------------
#   $DATA/patches/data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned/
# and NOTHING ELSE. In particular no mask tree: see "masks are not rewritten".
#
# ---- the design, in one paragraph ------------------------------------------
# A context tree is the SAME GRID as its 200x100 parent with a 50 px margin
# grown around every cell, NOT a new grid. Cell (i, j) here is cell (i, j) of
# data_patches_H200_W100_strpp2_11days_Aligned padded by 50 px on all four
# sides, so its centre 200x100 is bit-identical to the plain patch. That is why
# grid coordinates, nonz_indices.json, count-positives, every partition, the
# AOI windows and the reconstruction canvas all keep their exact meaning.
# Verified end to end before this script was written (see "smoke test" below).
#
# ---- NOT strides_per_patch=2 on a 300x200 patch ----------------------------
# Cutting 300x200 at strpp2 steps by (150, 100), which puts a cell centre at row
# 150i+50 -- an original patch start (100a) only when i is odd, so HALF the grid
# rows would be misaligned and "reconstruct at the original target locations"
# would be undefined for them. The grid that works is the parent's step (100,
# 50) with an origin offset of (-50, -50), which is what --context_margin cuts.
# As a scalar strides_per_patch that would be 3 in rows and 4 in columns, so the
# scalar is not passed at all: --strides_per_patch stays 2 (it names the GRID,
# which is unchanged) and --context_margin adds the margin.
#
# ---- masks are not rewritten -----------------------------------------------
# The target stays the plain 200x100 mask, so the mask tree is bit-identical to
# the one already on disk. --no-write_masks reuses it: 142 GB of I/O saved and,
# more importantly, "samples are selected on the CENTRE mask" is true by
# construction rather than by a cropping convention someone has to remember.
# resolve_patch_dirs() sends the data read to the ctx tree and the mask read to
# the plain tree, so training needs no extra flag.
#
# ---- the --by_list trap, and why step (3) exists ---------------------------
# prepare-patches overwrites nonz_indices.json with only the ids it processed
# in THIS run (prepare_patches.py). This script uses --by_list for
# resumability, so that file would be truncated to whatever the last run did.
# Step (3) installs the correct one by COPYING the parent tree's -- which is
# exact, not an approximation: the grid and the masks are the same, so the
# positive cells are the same. Verified on 20190113_20190124 (221 == 221 ==
# intf_coord.json's nonz_num) before this script was written. rebuild_nonz_-
# indices.py is deliberately NOT used here: it derives tree names without the
# ctx token and would rewrite the PARENT tree's index.
#
# ---- what is deliberately NOT here -----------------------------------------
# `sinkholes count-positives` must not run: it writes stride-dependent counts
# into assets/intf_coord.json, and this tree shares the parent's grid, so the
# parent's counts are already correct. Running it would be a no-op at best.
#
# ---- sizing (measured 2026-09-03, not guessed) -----------------------------
# DISK: the parent tree is 562 GB over 437 ids (mean 1.29 GB, largest 1.33 GB).
#   A context cell is 300*200 / (200*100) = 3x the pixels, so this tree is
#   ~1687 GB plain + ~2% nonz = ~1.72 TB. There were 11854 GB free on the mount
#   on 2026-09-03, leaving ~10.1 TB after. The check below refuses under 2 TB.
# MEMORY: measured 11.22 GB peak RSS on 20190113_20190124 (grid 193x89, a
#   mean-sized scene; the largest is within 3% of it) with /usr/bin/time -l.
#   64 GB is a ~5x margin and matches what the stride-4 job asked for.
# WALLTIME: I/O bound -- ~1.72 TB written plus ~0.58 TB of scenes read. 24:00 is
#   the same budget the stride-4 tree used for ~680 GB, so ONE SUBMISSION MAY
#   NOT FINISH. That is expected and safe: step (1) recomputes the remaining
#   work from what is on disk, so resubmitting continues where this stopped.
#
# ---- smoke test already done ------------------------------------------------
# 20190113_20190124 was cut with exactly these flags on 2026-09-03:
#   shape (193, 89, 300, 200) against the parent's (193, 89, 200, 100);
#   centre == parent patch on 7 sampled cells AND all 89 cells of grid row 96;
#   nonz_indices identical to the parent's (221);
#   only the data tree written, no mask tree.

set -o pipefail

REPO=/home/labs/rudich/pinkas/sinkholes
DATA=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data
SCENES=/home/labs/rudich/pinkas/scenes_11day
OUT=$DATA/patches
GT=$DATA/sub_20260701.shp
DICT=$REPO/assets/intf_coord.json
PARENT=$OUT/data_patches_H200_W100_strpp2_11days_Aligned
CTX=$OUT/data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned

cd "$REPO" || exit 1
source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# (0) preconditions -----------------------------------------------------------
[ -d "$PARENT" ] || { echo "!! parent tree missing: $PARENT" >&2; exit 1; }
[ -f "$GT" ]     || { echo "!! GT polygons missing: $GT" >&2; exit 1; }

# The scene directory is a symlink farm of the 'int' variant -- the one the
# committed dictionary describes. The frame-split copies under 004/ and 013/ are
# a DIFFERENT raster extent (20521x18772 against the dictionary's 20127x16556)
# and cutting from them fails on reshape, so this must be the farm, not a
# frame directory. Rebuilt when empty rather than erroring: it is only symlinks.
n_unw=$(ls -1 "$SCENES"/*.unw 2>/dev/null | wc -l)
if [ "$n_unw" -eq 0 ]; then
  echo "scene farm $SCENES is empty -- rebuilding it (symlinks only)"
  python scripts/data/link_scenes.py --data_dir "$DATA" --out_dir "$SCENES" \
      --days_diff 11 --variant int --clear
  n_unw=$(ls -1 "$SCENES"/*.unw 2>/dev/null | wc -l)
fi
echo "scene directory $SCENES: $n_unw .unw files"
[ "$n_unw" -gt 0 ] || { echo "!! still empty after relinking" >&2; exit 1; }

avail_gb=$(df -Pk "$OUT" | awk 'NR==2 {print int($4/1048576)}')
echo "free space on the patches mount: ${avail_gb} GB   (this tree needs ~1720 GB)"
[ "$avail_gb" -ge 2000 ] || { echo "!! under 2000 GB free; refusing to start" >&2; exit 1; }

mkdir -p "$CTX"

# (1) remaining work, computed from disk so a resubmission continues ----------
BY_LIST=$(python - "$PARENT" "$CTX" "$DICT" <<'PY'
import glob, json, os, sys
parent, ctx, dict_path = sys.argv[1], sys.argv[2], sys.argv[3]
SUF = "_H200_W100_strpp2.npy"
def ids(tree, prefix):
    return {os.path.basename(f)[len(prefix):-len(SUF)]
            for f in glob.glob(f"{tree}/{prefix}2*{SUF}")}
want = ids(parent, "data_patches_")
# An id counts as done only when BOTH its grid and its nonz subset exist: a
# half-written pair from a walltime kill must be redone, not skipped.
done = ids(ctx, "data_patches_") & ids(ctx, "data_patches_nonz_")
missing = sorted(want - done)
meta = set(json.load(open(dict_path)))
absent = [i for i in missing if i not in meta]
if absent:
    print(f"!! {len(absent)} ids are not in the dictionary: {absent[:5]}", file=sys.stderr)
    print("!! run 'sinkholes prepare-metadata' first", file=sys.stderr)
    raise SystemExit(1)
print(f"parent tree: {len(want)}   context tree complete: {len(done)}   to build: {len(missing)}",
      file=sys.stderr)
print(",".join(missing))
PY
)
if [ -z "$BY_LIST" ]; then
  echo "nothing left to build -- the ctx50x50 tree is complete."
else
  echo "building $(echo "$BY_LIST" | tr ',' '\n' | wc -l) interferograms"
  date
  # (2) the grids -------------------------------------------------------------
  # --strides_per_patch 2 names the GRID, which a margin does not move.
  python -m sinkholes prepare-patches \
    --input_dir "$SCENES" \
    --output_dir "$OUT" \
    --gt_polygon_file_path "$GT" \
    --patch_size 200 100 \
    --strides_per_patch 2 \
    --days_diff 11 \
    --context_margin 50 50 \
    --no-write_masks \
    --by_list "$BY_LIST" \
    --intf_dict_path "$DICT"
  date
fi

# (3) install the real nonz_indices.json (see "the --by_list trap" above) -----
python - "$PARENT" "$CTX" <<'PY'
import json, os, sys
parent, ctx = sys.argv[1], sys.argv[2]
src, dst = f"{parent}/nonz_indices.json", f"{ctx}/nonz_indices.json"
index = json.load(open(src))
present = {f.split("data_patches_")[1].split("_H200")[0]
           for f in os.listdir(ctx)
           if f.startswith("data_patches_2") and f.endswith("_H200_W100_strpp2.npy")}
missing = sorted(present - set(index))
if missing:
    raise SystemExit(f"!! {len(missing)} built ids absent from the parent index: {missing[:5]}")
json.dump(index, open(dst, "w"))
print(f"nonz_indices.json: copied {len(index)} ids from the parent tree "
      f"({len(present)} grids present here)")
PY

# (4) report -------------------------------------------------------------------
n_done=$(ls -1 "$CTX"/data_patches_2*_H200_W100_strpp2.npy 2>/dev/null | wc -l)
n_want=$(ls -1 "$PARENT"/data_patches_2*_H200_W100_strpp2.npy 2>/dev/null | wc -l)
echo
echo "context tree: $n_done / $n_want interferograms"
du -sh "$CTX" 2>/dev/null || true
if [ "$n_done" -lt "$n_want" ]; then
  echo "INCOMPLETE -- resubmit this same script to continue; it recomputes what is left."
  exit 1
fi
echo "COMPLETE. Train with:  --context_margin 50 50   (eval-scenes takes the same flag)"
