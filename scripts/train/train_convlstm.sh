#!/usr/bin/env bash
# ============================================================================
#  ConvLSTM U-Net training template
#
#  Submit:  bsub < scripts/train/train_convlstm.sh
#
#  EDIT TWO PLACES:
#    1. the "#BSUB -J" line below  -- names the LSF job, the run directory
#       and the log files. It is the single source of truth: the script reads
#       it back through $LSB_JOBNAME, so --job_name can never drift from it.
#    2. the CONFIG block           -- the experiment itself.
#
#  Name the job after what you changed, e.g.
#    convlstm_geo_k5_h256_b128_lr1e6_60e
#    convlstm_temporal_k10_h256_b128_lr1e6_posw4_60e
#  so outputs/ and logs/ stay readable. See PRESETS.md for the exact settings
#  of every run trained so far, and docs/MODEL_RUNS.md for what they scored.
# ============================================================================
#BSUB -J convlstm_geo_k10_h256_b128_lr1e6_60e
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/%J.err
#BSUB -q long-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=64G
#BSUB -R rusage[mem=128GB]
#BSUB -W 14:00

set -euo pipefail

# ------------------------------ CONFIG --------------------------------------
# Edit the defaults here for a one-off run. Each also honours an environment
# variable of the same name, which is how scripts/submit_all.sh drives batches
# without editing the file: SEED=7 bsub -J <name> < this_script
PARTITION="${PARTITION:-assets/partition_geo_k10.json}"  # geo_k5|geo_k10|temporal_k5|temporal_k10
K_PREVS="${K_PREVS:-10}"                  # temporal depth; MUST match the partition's k
HIDDEN="${HIDDEN:-256}"                   # ConvLSTM hidden channels (256 is enough; see PRESETS)
POS_W="${POS_W:-8}"                       # BCE positive weight (code default is 1)
SEED="${SEED:-42}"
LR="${LR:-1e-6}"
SCHEDULE="${SCHEDULE:-plateau}"           # plateau | cosine
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-128}"
PATIENCE="${PATIENCE:-20}"

# --- negative sampling ------------------------------------------------------
# Positives-only training (--nonz_only, the code default and what every run
# before 2026-08-10 used) fits the model on the ~2.5% of the patch grid that
# contains subsidence, then applies it to 100% of the map at inference. That is
# what drives the scene-scale false-positive rate: geo_k10 scores P=0.12 at
# confidence 0.125 (docs/MODEL_RUNS.md). POS_W=8 pushes the same direction on
# top of it, which is why POS_W 8->4 bought +0.023 object-level F1.
#
# RING_NEGS=yes adds all-zero patches drawn from an annulus around the
# positives. Candidates must be empty at EVERY timestep, so a patch that was
# positive last month never enters as a negative (dataset.py:69-97).
RING_NEGS="${RING_NEGS:-no}"              # yes | no
NEG_RING_INNER="${NEG_RING_INNER:-1}"     # annulus radii, in patch-grid units
NEG_RING_OUTER="${NEG_RING_OUTER:-3}"     # 3 = hard near-field only; raise to reach far-field
NEG_PER_POS="${NEG_PER_POS:-1.0}"         # negatives per positive, capped by availability

# --- validation negatives ---------------------------------------------------
# The three knobs above shape TRAINING data. VAL_NEGS=yes puts negatives in the
# VALIDATION set as well, which is what makes val/dice able to see the thing
# they exist to fix -- a false positive on background now costs dice instead of
# being invisible. Its configuration is FIXED in the code (ring 1..3, 1:1, drawn
# once from the validation interferograms with SEED) and has no knobs here on
# purpose: it is the ruler, not the experiment. Any two runs sharing a partition
# and a seed are then scored on identical samples whatever their RING_NEGS
# settings or architecture, so their val/dice can be read against each other.
#
# It does NOT make a run comparable with one trained before it existed: a
# positives-only val set and a 1:1 one are different scales. Compare valneg runs
# with valneg runs.
VAL_NEGS="${VAL_NEGS:-no}"                # yes | no   (needs SEED)

# 'auto' finds this LSF job's own directory again after a WEXAC preemption, and
# only works because a requeued job keeps its id. A job KILLED by TERM_RUNLIMIT
# is not requeued and gets a new id, so 'auto' would start it from scratch --
# point RESUME at the run directory instead to carry on from its last epoch.
RESUME="${RESUME:-auto}"                  # auto | <run dir> | <checkpoint file>
# ----------------------------------------------------------------------------

# K_PREVS and the partition are not independent: partition_*_k5.json only lists
# interferograms with 5-previous chains, _k10 only those with 10. Mixing them
# silently trains on a smaller set than you think.
case "$PARTITION" in
  *_k5.json)  [ "$K_PREVS" = 5 ]  || { echo "K_PREVS=$K_PREVS with a _k5 partition"  >&2; exit 1; } ;;
  *_k10.json) [ "$K_PREVS" = 10 ] || { echo "K_PREVS=$K_PREVS with a _k10 partition" >&2; exit 1; } ;;
esac

# Ring negatives reach the TRAIN dataset only -- train.py passes them to the
# train split and not to val/test. With VAL_NEGS=no, validation therefore stays
# positives-only and val/dice CANNOT see the thing this flag exists to fix:
# expect it flat or slightly down while scene-level precision improves, and
# judge the run with scripts/eval/run_eval.sh (object-level F1) rather than the
# results.csv curve. That is exactly what the 2026-08-10 batch showed -- all
# nine arms inside the +-0.006 noise floor. VAL_NEGS=yes is the fix for it.
NEG_FLAGS=()
case "$RING_NEGS" in
  yes) NEG_FLAGS=(--add_ring_negatives
                  --neg_ring_inner "$NEG_RING_INNER"
                  --neg_ring_outer "$NEG_RING_OUTER"
                  --neg_per_pos "$NEG_PER_POS") ;;
  no)  ;;
  *)   echo "RING_NEGS must be 'yes' or 'no', got '$RING_NEGS'" >&2; exit 1 ;;
esac

case "$VAL_NEGS" in
  yes) NEG_FLAGS+=(--add_val_negatives) ;;
  no)  ;;
  *)   echo "VAL_NEGS must be 'yes' or 'no', got '$VAL_NEGS'" >&2; exit 1 ;;
esac

JOB="${LSB_JOBNAME:-convlstm_manual}"

cd /home/labs/rudich/pinkas/sinkholes
source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sinkholes train \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --learning-rate "$LR" \
  --lr_schedule "$SCHEDULE" \
  --patches_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches \
  --partition_mode preset_by_intf \
  --partition_file "$PARTITION" \
  --patch_size 200 100 \
  --stride 2 \
  --pos_w "$POS_W" \
  --add_temporal \
  --k_prevs "$K_PREVS" \
  --convlstm_unet \
  --convlstm_hidden "$HIDDEN" \
  ${NEG_FLAGS[@]+"${NEG_FLAGS[@]}"} \
  --amp \
  --save_best_only \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --job_name "$JOB" \
  --resume "$RESUME"
