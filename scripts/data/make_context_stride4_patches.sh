#!/bin/bash
#BSUB -J make_ctx50_strpp4
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/ctx50_s4_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/ctx50_s4_%J.err
#BSUB -q short
#BSUB -R rusage[mem=128GB]
#BSUB -W 24:00
#
# Build the STRIDE-4 LARGE-CONTEXT patch tree, for the interferograms the
# eval6-protocol context evaluations actually read -- and only those.
#
# Submit:  bsub < scripts/data/make_context_stride4_patches.sh   (from the repo root)
#
# ---- what this produces ----------------------------------------------------
#   $DATA/patches/data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned/
# holding ~61 of the 437 interferograms, and NOTHING ELSE. No mask tree (see
# "masks are not rewritten"), and no attempt at the other 376 (see "A SUBSET").
#
# ---- what it unblocks ------------------------------------------------------
# submit_all.sh evalampfix_base30 and evalampfix_ctx30, both marked BLOCKED on
# exactly this directory (they died in seconds on FileNotFoundError, LSF
# 680921/680926). EVAL6_TEMP sets DATA_STRIDE=4 and the context tree existed
# only at strpp2.
#
# WHY NOT JUST USE THE evalctx STRIDE-2 BATCH. It works and it is already
# defined, but it is self-contained: stride 2 AND prob rather than stride 4 and
# rth, so it shares no number with RESULTS.md and needed a third job
# (evalctx_baseline) purely to manufacture a reference. These two run the eval6
# protocol, so they are readable against the 0.7797 anchor directly.
#
# ---- the design is make_context_patches.sh's, unchanged --------------------
# A context tree is the SAME GRID as its plain parent with a 50 px margin grown
# around every cell, NOT a new grid. Cell (i, j) here is cell (i, j) of
# data_patches_H200_W100_strpp4_11days_Aligned padded by 50 px on all four
# sides, so its centre 200x100 is bit-identical to the plain strpp4 patch. Read
# that script's header for the full argument; everything in it holds here with
# strpp2 replaced by strpp4. In particular --strides_per_patch 4 names the GRID,
# which a margin does not move, and prepare-patches re-proves the centre
# invariant on three cells of every grid it cuts (prepare_patches.py:171).
#
# ---- masks are not rewritten -----------------------------------------------
# The target stays the plain 200x100 mask, so --no-write_masks reuses the
# mask_patches_H200_W100_strpp4_11days_Aligned tree already on disk (complete at
# 437 ids, 565 GB). patch_dir_name() RAISES on a margined mask tree -- context
# trees have no masks by construction, not by convention. resolve_patch_dirs()
# and scenes.py send the data read here and the mask read there.
#
# ---- A SUBSET, AND WHY THAT IS SAFE ----------------------------------------
# The full 437-id tree would be 6.55 TB against 9.6 TB free. It is not needed:
# eval-scenes reads the partition's scene list plus each scene's k-previous
# chain, and nothing else. For the evalampfix settings below that is 61 ids,
# 943 GB -- measured from the parent tree's real file sizes, not estimated.
#
#   22 scenes in partition_temporal_k10_testeval_clean.json ["val"]
#   -> 20 after the --min_positives 150 gate
#        (drops 20250121_20250201 at 42 and 20260316_20260327 at 116)
#   -> all 20 have full 5-chains
#   -> 61 unique ids once the chains are unioned in
#
# 20250121_20250201 is dropped as a SCORED scene and is still in the 61 as a
# PREDECESSOR. The chain sets the requirement, not the scene list -- which is
# why step (1) derives the set from the partition with the project's own
# find_11day_sequences instead of anyone typing a list.
#
# ---- THE HAZARD THIS SCRIPT EXISTS TO CONTAIN ------------------------------
# scenes.py:354-357 filters predecessors by os.path.exists and then pads a short
# chain with the CURRENT frame:
#     available = [pid for pid in prev_ids if os.path.exists(grid_path(...))]
#     while len(prevs) < args.k_prevs: prevs.append(cur)
# A missing predecessor grid therefore raises NOTHING. The evaluation runs to
# completion and reports a plausible, wrong number, because a timestep that
# should have been 2025-03-06 was silently the scene itself. Against a complete
# tree that path is unreachable; against a SUBSET tree it is one typo away.
# Step (4) is the whole point of the script: it refuses to report success unless
# every one of the derived ids is present. Do not submit the evals if it fails.
#
# ---- what is deliberately NOT here -----------------------------------------
# `sinkholes count-positives` must not run: it writes stride-dependent positive
# counts into assets/intf_coord.json, and every partition and training run reads
# the stride-2 ones. Same reason it is absent from make_stride4_patches.sh.
#
# `verify_dataset.py` must not run either, and unlike the count-positives case
# it is not merely useless but actively misleading: it checks "patch ids ==
# dictionary ids" and this tree holds 61 of 437 BY DESIGN, so it fails on a
# healthy tree. Step (4) is the targeted check that replaces it.
#
# ---- sizing (measured 2026-09-08, not guessed) -----------------------------
# DISK: the 61 needed ids occupy 311.3 GB in the plain strpp4 tree. A context
#   cell is 300*200 / (200*100) = 3x the pixels, so 934 GB plus ~1% nonz = 943
#   GB. Step (0) recomputes this for whatever is actually left to build and
#   checks df against that number rather than against a constant.
# MEMORY: make_context_patches.sh measured 11.22 GB peak RSS at strpp2. Stride 4
#   quadruples the cell count, so ~45 GB, and prepare_patches.py:171 cuts a
#   SECOND (plain) grid for the centre-alignment assert on top of that. 128 GB
#   is roughly a 2x margin on the extrapolation. This is an extrapolation, not a
#   measurement -- if it dies on memory, the log's peak RSS is the real number.
# WALLTIME: I/O bound -- ~943 GB written plus ~81 GB of scenes read. The
#   stride-4 plain tree budgeted 24:00 for ~680 GB, so ONE SUBMISSION MAY NOT
#   FINISH. That is expected and safe: step (1) recomputes the remaining work
#   from what is on disk, so resubmitting continues where this stopped.

set -o pipefail

REPO=/home/labs/rudich/pinkas/sinkholes
DATA=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data
SCENES=/home/labs/rudich/pinkas/scenes_11day
OUT=$DATA/patches
GT=$DATA/sub_20260701.shp
DICT=$REPO/assets/intf_coord.json
PARENT4=$OUT/data_patches_H200_W100_strpp4_11days_Aligned
MASK4=$OUT/mask_patches_H200_W100_strpp4_11days_Aligned
CTX4=$OUT/data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned

# Which evaluation this tree is being cut for. The defaults are evalampfix's
# settings exactly -- run_eval.sh resolves GEN=3 + SPLIT=test + GROUP=temporal_k10
# to this partition file, K_PREVS=5 names the checkpoint's depth, and MIN_POS
# is run_eval.sh's own default of 150. Change these and the id set changes with
# them; that is the point of deriving it rather than listing it.
PARTITION="${PARTITION:-$REPO/assets/partition_temporal_k10_testeval_clean.json}"
K_PREVS="${K_PREVS:-5}"
MIN_POS="${MIN_POS:-150}"

cd "$REPO" || exit 1
source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

echo "partition $PARTITION"
echo "k_prevs   $K_PREVS   min_positives $MIN_POS"
echo "target    $CTX4"
echo

# (0) preconditions -----------------------------------------------------------
[ -d "$PARENT4" ]  || { echo "!! parent stride-4 tree missing: $PARENT4" >&2; exit 1; }
[ -d "$MASK4" ]    || { echo "!! stride-4 mask tree missing: $MASK4 -- eval-scenes reads its GT from there" >&2; exit 1; }
[ -f "$GT" ]       || { echo "!! GT polygons missing: $GT" >&2; exit 1; }
if [ ! -f "$PARTITION" ]; then
  echo "!! partition missing: $PARTITION" >&2
  # A path starting /assets/ is almost always $REPO expanded to nothing: $REPO is
  # defined in THIS script, not in the submitting shell, so `PARTITION=$REPO/...`
  # typed at a prompt submits a rooted path that cannot exist (LSF 176342).
  case "$PARTITION" in
    /assets/*) echo "!! that looks like \$REPO expanded to empty in the submitting shell." >&2
               echo "!! \$REPO is set INSIDE this script; pass a repo-relative path instead:" >&2
               echo "!!     PARTITION=${PARTITION#/} bsub < $0" >&2 ;;
  esac
  exit 1
fi

# The scene directory is a symlink farm of the 'int' variant -- the one the
# committed dictionary describes. The frame-split copies under 004/ and 013/ are
# a DIFFERENT raster extent (20521x18772 against the dictionary's 20127x16556)
# and cutting from them fails on reshape, so this must be the farm, not a frame
# directory. It was EMPTY on 2026-09-08 and is rebuilt here rather than treated
# as an error: it is only symlinks. This matters more than it looks -- with no
# .unw to match, prepare-patches writes nothing and EXITS SUCCESSFULLY, which
# reads exactly like a job that worked.
#
# COUNTED WITH find, NOT `ls *.unw | wc -l`. On an empty directory the glob
# matches nothing, ls exits non-zero, `set -o pipefail` propagates that through
# wc, and `set -e` then kills the script AT THE ASSIGNMENT -- so the very
# rebuild this block exists to perform is never reached, and the job dies with
# an empty log. Verified on 2026-09-08. make_context_patches.sh:107 carries the
# same line and the same latent bug -- its "rebuilt when empty" branch is
# unreachable and it survived only because its farm was populated at the time.
# make_stride4_patches.sh:77 is the same line again; there the empty farm is
# meant to be fatal, so only its diagnostic is lost, not its behaviour.
# find exits 0 on no matches, so the count is 0 and the branch runs.
n_unw=$(find "$SCENES" -maxdepth 1 -name '*.unw' | wc -l)
if [ "$n_unw" -eq 0 ]; then
  echo "scene farm $SCENES is empty -- rebuilding it (symlinks only)"
  python scripts/data/link_scenes.py --data_dir "$DATA" --out_dir "$SCENES" \
      --days_diff 11 --variant int --clear
  n_unw=$(find "$SCENES" -maxdepth 1 -name '*.unw' | wc -l)
fi
echo "scene directory $SCENES: $n_unw .unw files"
[ "$n_unw" -gt 0 ] || { echo "!! still empty after relinking" >&2; exit 1; }

mkdir -p "$CTX4"

# (1) the id set, and the remaining work ---------------------------------------
# Derived from the partition with the project's own chain code, so it cannot
# drift from what eval-scenes will ask for. Emits two lines: the ids still to
# build, and the GB they need. The FULL set goes to $NEED_FILE for step (4).
NEED_FILE=$(mktemp)
PLAN=$(mktemp)
trap 'rm -f "$NEED_FILE" "$PLAN"' EXIT

python - "$PARENT4" "$MASK4" "$CTX4" "$DICT" "$PARTITION" "$K_PREVS" "$MIN_POS" "$NEED_FILE" \
    > "$PLAN" <<'PY'
import json, os, sys

sys.path.insert(0, os.getcwd())
from sinkholes.meta import find_11day_sequences

parent4, mask4, ctx4, dict_path, partition, k_prev, min_pos, need_file = sys.argv[1:9]
k_prev, min_pos = int(k_prev), int(min_pos)
SUF = "_H200_W100_strpp4.npy"
GB = 1073741824


def ids(tree, prefix):
    """Ids of one file family in a tree.

    The '2' after the prefix is load-bearing: every id starts with a year, so it
    separates data_patches_<id> from data_patches_nonz_<id>, which also starts
    with 'data_patches_'.
    """
    if not os.path.isdir(tree):
        return set()
    return {f[len(prefix):-len(SUF)] for f in os.listdir(tree)
            if f.startswith(prefix + "2") and f.endswith(SUF)}


coord = json.load(open(dict_path))
# --intf_source preset reads ONLY the "val" key (scenes.py:181). A _testeval
# partition holds the parent's TEST list under that key; that indirection is
# run_eval.sh's, and this has to match it or the tree is cut for other scenes.
scenes = json.load(open(partition))["val"]

# filter_by_min_positives (scenes.py:130), including its strict '>'.
kept = [i for i in scenes
        if isinstance(coord.get(i, {}).get("nonz_num"), int)
        and coord[i]["nonz_num"] > min_pos]
dropped = [(i, coord.get(i, {}).get("nonz_num", "none")) for i in scenes if i not in kept]
if dropped:
    print(f"--min_positives {min_pos} drops {len(dropped)} of {len(scenes)} scenes: "
          + ", ".join(f"{i}({n})" for i, n in sorted(dropped)), file=sys.stderr)

prev_dict, with_chains = find_11day_sequences(
    coord, k_prev=k_prev, restrict_to=kept, require_current_nonz_gt0=False)
no_chain = sorted(set(kept) - set(with_chains))
if no_chain:
    # eval-scenes drops these too (scenes.py:322), so not cutting them is correct.
    print(f"{len(no_chain)} scenes have no full {k_prev}-chain and are not scored: "
          f"{no_chain}", file=sys.stderr)

# The scored scenes plus every predecessor eval-scenes will open for them.
need = set(with_chains)
for cur in with_chains:
    need.update(prev_dict[cur]["prevs"][:k_prev])
need = sorted(need)

absent = [i for i in need if i not in coord]
if absent:
    print(f"!! {len(absent)} ids are not in the dictionary: {absent[:5]}", file=sys.stderr)
    print("!! run 'sinkholes prepare-metadata' first", file=sys.stderr)
    raise SystemExit(1)

# Cheap consistency checks against the trees this one is derived from. A needed
# id absent from the plain strpp4 tree is not fatal here -- prepare-patches cuts
# from the .unw, not from that tree -- but it means the two trees disagree about
# what exists, which is worth seeing before 943 GB of writes.
plain_missing = [i for i in need if i not in ids(parent4, "data_patches_")]
if plain_missing:
    print(f"NOTE: {len(plain_missing)} needed ids have no plain strpp4 grid: "
          f"{plain_missing[:5]}", file=sys.stderr)
# This one IS fatal: eval-scenes reads the GT of every SCORED scene from the
# stride-4 mask tree, and a missing mask grid is a hard FileNotFoundError there.
mask_missing = [i for i in with_chains if i not in ids(mask4, "mask_patches_")]
if mask_missing:
    print(f"!! {len(mask_missing)} scored scenes have no stride-4 mask grid: "
          f"{mask_missing[:5]}", file=sys.stderr)
    print("!! run scripts/data/make_stride4_patches.sh first", file=sys.stderr)
    raise SystemExit(1)

# An id counts as done only when BOTH its grid and its nonz subset exist: a
# half-written pair from a walltime kill must be redone, not skipped.
done = ids(ctx4, "data_patches_") & ids(ctx4, "data_patches_nonz_")
missing = [i for i in need if i not in done]

# Size the remaining work from the parent tree's real bytes x3, so a resumed run
# checks df against what IT still needs rather than against the original total.
bytes4 = 0
for i in missing:
    f = os.path.join(parent4, f"data_patches_{i}{SUF}")
    if os.path.exists(f):
        bytes4 += os.path.getsize(f)
need_gb = int(bytes4 * 3 * 1.01 / GB) + 1

print(f"scored scenes: {len(with_chains)}   unique ids needed (with {k_prev}-chains): "
      f"{len(need)}   already built: {len(need) - len(missing)}   to build: {len(missing)}",
      file=sys.stderr)
print(f"remaining work: ~{need_gb} GB", file=sys.stderr)

with open(need_file, "w") as fh:
    fh.write("\n".join(need) + "\n")
print(",".join(missing))
print(need_gb)
PY

BY_LIST=$(sed -n 1p "$PLAN")
NEED_GB=$(sed -n 2p "$PLAN")

if [ -z "$BY_LIST" ]; then
  echo "nothing left to build -- every needed id is already in the ctx50x50 strpp4 tree."
else
  # Checked against what is actually left, plus 300 GB of headroom so a run that
  # just fits does not fill the mount other jobs are writing to.
  avail_gb=$(df -Pk "$OUT" | awk 'NR==2 {print int($4/1048576)}')
  floor=$((NEED_GB + 300))
  echo "free space on the patches mount: ${avail_gb} GB   (need ~${NEED_GB} GB, floor ${floor} GB)"
  [ "$avail_gb" -ge "$floor" ] || { echo "!! under ${floor} GB free; refusing to start" >&2; exit 1; }

  echo "building $(echo "$BY_LIST" | tr ',' '\n' | wc -l) interferograms"
  date
  # (2) the grids -------------------------------------------------------------
  # --strides_per_patch 4 names the GRID, which a margin does not move.
  python -m sinkholes prepare-patches \
    --input_dir "$SCENES" \
    --output_dir "$OUT" \
    --gt_polygon_file_path "$GT" \
    --patch_size 200 100 \
    --strides_per_patch 4 \
    --days_diff 11 \
    --context_margin 50 50 \
    --no-write_masks \
    --by_list "$BY_LIST" \
    --intf_dict_path "$DICT"
  date
fi

# (3) install nonz_indices.json from the parent strpp4 tree --------------------
# The grid and the masks are the parent's, so the parent's positive cells ARE
# this tree's positive cells -- a copy, not an approximation.
#
# prepare-patches no longer truncates the index under --by_list (--index_mode
# defaults to merge, prepare_patches.py:119), so this is belt-and-braces rather
# than the repair make_context_patches.sh step (3) had to be. It is still the
# right file: a merge only ever accumulates the ids THIS tree has, and the
# parent's covers all 437 with identical coordinates.
#
# rebuild_nonz_indices.py is deliberately NOT used: it derives tree names
# without the ctx token and would rewrite the PARENT tree's index.
python - "$PARENT4" "$CTX4" <<'PY'
import json, os, sys
parent4, ctx4 = sys.argv[1], sys.argv[2]
SUF = "_H200_W100_strpp4.npy"
index = json.load(open(f"{parent4}/nonz_indices.json"))
present = {f[len("data_patches_"):-len(SUF)] for f in os.listdir(ctx4)
           if f.startswith("data_patches_2") and f.endswith(SUF)}
missing = sorted(present - set(index))
if missing:
    raise SystemExit(f"!! {len(missing)} built ids absent from the parent index: {missing[:5]}")
json.dump(index, open(f"{ctx4}/nonz_indices.json", "w"))
print(f"nonz_indices.json: copied {len(index)} ids from the parent strpp4 tree "
      f"({len(present)} grids present here)")
PY

# (4) THE CHECK THAT MATTERS --------------------------------------------------
# Not a formality. Read "THE HAZARD THIS SCRIPT EXISTS TO CONTAIN" above: a
# predecessor missing from this tree does not fail the evaluation, it silently
# replaces that timestep with the scene's own frame and reports a number that
# looks fine. Nothing downstream will catch it. So the tree is either complete
# for the derived id set or this script exits non-zero.
python - "$CTX4" "$NEED_FILE" <<'PY'
import os, sys
ctx4, need_file = sys.argv[1], sys.argv[2]
SUF = "_H200_W100_strpp4.npy"
need = [l.strip() for l in open(need_file) if l.strip()]
have = {f[len("data_patches_"):-len(SUF)] for f in os.listdir(ctx4)
        if f.startswith("data_patches_2") and f.endswith(SUF)}
nonz = {f[len("data_patches_nonz_"):-len(SUF)] for f in os.listdir(ctx4)
        if f.startswith("data_patches_nonz_2") and f.endswith(SUF)}
missing = [i for i in need if i not in (have & nonz)]
print(f"\ncontext stride-4 tree: {len(need) - len(missing)} / {len(need)} needed interferograms")
if missing:
    print(f"INCOMPLETE -- {len(missing)} still missing: {missing[:8]}"
          f"{' ...' if len(missing) > 8 else ''}")
    print("Resubmit this same script to continue; it recomputes what is left.")
    print("DO NOT submit the evaluations until this passes: a missing predecessor is")
    print("padded with the current frame (scenes.py:354) and scores silently wrong.")
    raise SystemExit(1)
PY

du -sh "$CTX4" 2>/dev/null || true
echo
echo "COMPLETE -- every interferogram the evaluation will open is present."
echo "next: bash scripts/submit_all.sh evalampfix --only=ctx30      # dry run"
echo "      bash scripts/submit_all.sh evalampfix --only=base30     # dry run"
echo "      (drop the BLOCKED notes in scripts/submit_all.sh, then add --submit)"
