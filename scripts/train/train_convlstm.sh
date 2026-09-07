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
# No default on purpose: assets/ holds three generations of partition
# (assets/PARTITIONS.md). Defaulting to one is how a clean-data run silently
# trains on the 2019-2026 noisy lists, so set it explicitly every time.
PARTITION="${PARTITION:?set PARTITION=assets/partition_<axis>_k<k>[_clean].json -- see assets/PARTITIONS.md}"
K_PREVS="${K_PREVS:-10}"                  # temporal depth; MUST match the partition's k
HIDDEN="${HIDDEN:-256}"                   # ConvLSTM hidden channels (256 is enough; see PRESETS)
POS_W="${POS_W:-8}"                       # BCE positive weight (code default is 1)
SEED="${SEED:-42}"
LR="${LR:-1e-6}"
SCHEDULE="${SCHEDULE:-plateau}"           # plateau | cosine
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-128}"
# Micro-batch x ACCUM is the EFFECTIVE batch, and the effective batch is what a
# run is comparable on. Raise ACCUM and lower BATCH by the same factor when the
# activations no longer fit -- a large-context run is the reason this exists --
# and the optimiser sees what BATCH=128 would have given it. Both are STRICT
# resume keys (resume.py), so a resume cannot quietly change either.
ACCUM="${ACCUM:-1}"
PATIENCE="${PATIENCE:-20}"                # early stop; 0 = off (reporter.py:257)
# plateau: epochs without val/dice improvement before the LR is cut. 5 is the
# code default (train.py:259) and what EVERY run before 2026-09-03 used, since
# this template did not expose the knob at all. train.py argues for keeping it
# short -- on the k10split run the first cut is what broke a five-epoch plateau.
# Raise it when a run is dying at a floored LR with epochs left to spend, and
# say so in the run's row: it is an ADVISORY resume key (resume.py:88), so a
# resume will report the change rather than refuse it.
LR_PATIENCE="${LR_PATIENCE:-5}"
MOMENTUM="${MOMENTUM:-0.999}"        # RMSprop momentum; 0.999 is historical, see --momentum
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-8}"      # 1e-8 is historical and is barely regularisation
                                          # at all; 1e-4..1e-2 is the usual range
AUGMENT="${AUGMENT:-none}"                # none | h | v | hv -- random flips, TRAIN split only

# --- spatial context --------------------------------------------------------
# Empty = the plain 200x100 tree, which is every run before 2026-09-07.
# CTX_MY=50 CTX_MX=50 reads data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned and feeds
# the network 300x200 while still supervising, scoring and reconstructing the
# centre 200x100. The grid, the targets, sample selection, ring negatives, the
# AOI window and the evaluation protocol are all unchanged -- only what
# surrounds each target. Costs ~3x the activations, hence ACCUM above.
# Two variables rather than one "MY MX" string: submit_all.sh applies overrides
# with `read -ra`, which splits on whitespace, so a two-word value cannot
# survive the trip. Set BOTH or NEITHER.
CTX_MY="${CTX_MY:-}"
CTX_MX="${CTX_MX:-}"
CTX_FLAGS=()
if [ -n "$CTX_MY" ] || [ -n "$CTX_MX" ]; then
  if [ -z "$CTX_MY" ] || [ -z "$CTX_MX" ]; then
    echo "set both CTX_MY and CTX_MX, or neither (got '$CTX_MY' / '$CTX_MX')" >&2
    exit 1
  fi
  CTX_FLAGS=(--context_margin "$CTX_MY" "$CTX_MX")
fi

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

# --- validation negatives (DEPRECATED 2026-08-20, DO NOT USE) ---------------
# VAL_NEGS=yes put negatives in the VALIDATION set as well, so val/dice could
# see a false positive on background. It is deprecated: dice_coeff maps an
# empty prediction on an empty mask to 1.0, so at 1:1 roughly half the val
# samples score ~1.0 and the mean becomes about (1 + dice_on_positives)/2 --
# a number that ranks nothing and cannot be read against any run trained
# without it. val/F1, val/P and val/R are pooled from raw pixel counts
# (evaluate.py:300-306) and were already negative-aware, so nothing was gained.
# Precision claims belong to scripts/eval/run_eval.sh at object level.
#
# IT STILL WORKS, for exactly one reason: the five attnfix runs of 2026-08-19
# were trained with it, and `dataset` is a STRICT resume key carrying
# `valneg=1-3x1.0` (resume.py). A resume of one of those runs -- including the
# manual `RESUME=<run dir> bsub -J <same name> < this script` path -- must pass
# VAL_NEGS=yes or it will be refused as an incompatible resume. No new run
# should set it, and nothing in scripts/submit_all.sh does any more.
VAL_NEGS="${VAL_NEGS:-no}"                # yes | no   DEPRECATED: resume only

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
  *_k5.json|*_k5_*.json)  [ "$K_PREVS" = 5 ]  || { echo "K_PREVS=$K_PREVS with a _k5 partition"  >&2; exit 1; } ;;
  *_k10.json|*_k10_*.json) [ "$K_PREVS" = 10 ] || { echo "K_PREVS=$K_PREVS with a _k10 partition" >&2; exit 1; } ;;
esac

# Ring negatives reach the TRAIN dataset only -- train.py passes them to the
# train split and not to val/test. Validation is therefore positives-only and
# val/dice CANNOT see the thing this flag exists to fix: expect it flat or
# slightly down while scene-level precision improves, and judge the run with
# scripts/eval/run_eval.sh (object-level F1) rather than the results.csv curve.
# That is exactly what the 2026-08-10 batch showed -- all nine arms inside the
# +-0.006 noise floor. Read val/F1 rather than val/dice: it is pooled over
# pixels, so it is negative-aware and does move.
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
  yes) echo "WARNING: VAL_NEGS=yes is DEPRECATED and inflates val/dice to roughly" >&2
       echo "         (1 + dice_on_positives)/2. It is kept only so the attnfix runs of" >&2
       echo "         2026-08-19 can be resumed. Do not start a NEW run with it." >&2
       NEG_FLAGS+=(--add_val_negatives) ;;
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
  --accum_steps "$ACCUM" \
  --learning-rate "$LR" \
  --lr_schedule "$SCHEDULE" \
  --lr_patience "$LR_PATIENCE" \
  --momentum "$MOMENTUM" \
  --weight_decay "$WEIGHT_DECAY" \
  --augment_flips "$AUGMENT" \
  --patches_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches \
  --partition_mode preset_by_intf \
  --partition_file "$PARTITION" \
  --patch_size 200 100 \
  ${CTX_FLAGS[@]+"${CTX_FLAGS[@]}"} \
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
