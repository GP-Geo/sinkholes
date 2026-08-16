#!/usr/bin/env bash
# ============================================================================
#  Temporal-attention U-Net training template
#
#  Submit:  bsub < scripts/train/train_tattn.sh
#
#  EDIT TWO PLACES:
#    1. the "#BSUB -J" line below  -- names the LSF job, the run directory
#       and the log files. It is the single source of truth: the script reads
#       it back through $LSB_JOBNAME, so --job_name can never drift from it.
#    2. the CONFIG block           -- the experiment itself.
#
#  Name the job after what you changed, e.g.
#    tattn_temporal_k5_d256_fuse0_posw4_60e
#    tattn_temporal_k5_d256_fuse4_recur-convlstm_60e
#  so outputs/ and logs/ stay readable. See PRESETS.md for the exact settings
#  of every run trained so far, and docs/MODEL_RUNS.md for what they scored.
#
#  WHAT THIS MODEL IS. The ConvLSTM compresses the interferogram sequence into
#  one hidden state and takes its skip connections from the latest frame alone.
#  This one lets the current frame attend over its own history at each
#  bottleneck pixel, and can reuse those attention weights to fuse the skips
#  over time as well. The hypothesis worth testing is about PRECISION, not
#  dice: atmospheric and decorrelation artefacts are temporally inconsistent
#  while sinkhole subsidence is persistent, so a learned weighted average over
#  the chain is a matched filter for the signal and a suppressor for the noise.
#  Judge it with scripts/eval/run_eval.sh, not with the results.csv curve.
# ============================================================================
#BSUB -J tattn_temporal_k5_d256_fuse0_posw4_60e
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/%J.err
#BSUB -q long-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=36G
#BSUB -R rusage[mem=96GB]
#BSUB -W 12:00

set -euo pipefail

# ------------------------------ CONFIG --------------------------------------
# Edit the defaults here for a one-off run. Each also honours an environment
# variable of the same name, which is how scripts/submit_all.sh drives batches
# without editing the file: SEED=7 bsub -J <name> < this_script
#
# Defaults are temporal_k5 at pos_w 4 because that is where the comparison
# lives: convlstm_temporal_k5_h256_posw4_60e is the same-everything-else
# reference, group T has two k5 ConvLSTM runs bracketing the +-0.006 noise
# floor, and k5 carries 34% more training data than k10 for the same score.
PARTITION="${PARTITION:-assets/partition_temporal_k5.json}"  # geo_k5|geo_k10|temporal_k5|temporal_k10
K_PREVS="${K_PREVS:-5}"                   # temporal depth; MUST match the partition's k
POS_W="${POS_W:-4}"                       # BCE positive weight (code default is 1)
SEED="${SEED:-42}"
LR="${LR:-1e-6}"
SCHEDULE="${SCHEDULE:-plateau}"           # plateau | cosine
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-128}"
PATIENCE="${PATIENCE:-20}"

# --- the attention block ----------------------------------------------------
DIM="${DIM:-0}"                           # token width (0 = the default, 256)
HEADS="${HEADS:-8}"                       # must divide 64/128/256/512 when FUSE_SKIPS>0
LAYERS="${LAYERS:-1}"                     # 1 = a single present-queries-past readout
RECURRENCE="${RECURRENCE:-none}"          # none | convlstm  (the hybrid)
HIDDEN="${HIDDEN:-256}"                   # ConvLSTM hidden width, RECURRENCE=convlstm only

# How many skips are fused over time by the attention weights, coarsest first.
#   0  latest frame only -- byte-for-byte the ConvLSTM's skip contract, so a run
#      at 0 is the clean head-to-head: same encoder, same decoder, same skips,
#      the ONLY difference is attention vs recurrence at the bottleneck.
#   2  s4 (25x12) and s3 (50x25). The attention field is computed at 12x6, so
#      these are 2x and 4x upsamples -- the defensible middle.
#   4  every level, including s1 at 200x100 where the same field is stretched
#      16x. Note an 11-day interferogram measures RATE, so a sinkhole active at
#      t-5 and quiescent now shows up in the older frames: temporally averaging
#      the FINE skip can import stale signatures into the layer that decides
#      object boundaries, which is the opposite of what we want from precision.
#      Run 0 and 2 before spending a slot on 4.
FUSE_SKIPS="${FUSE_SKIPS:-0}"             # 0..4

# --- negative sampling ------------------------------------------------------
# Positives-only training (--nonz_only, the code default) fits the model on the
# ~2.5% of the patch grid that contains subsidence, then applies it to 100% of
# the map at inference. That is what drives the scene-scale false-positive rate
# (geo_k10 scores P=0.12 at confidence 0.125, docs/MODEL_RUNS.md), and it is a
# DATA problem this architecture does not touch. If scene-level precision is
# what you are trying to move, RING_NEGS=yes is the more direct lever.
#
# RING_NEGS=yes adds all-zero patches drawn from an annulus around the
# positives. Candidates must be empty at EVERY timestep, so a patch that was
# positive last month never enters as a negative (dataset.py:69-97).
RING_NEGS="${RING_NEGS:-no}"              # yes | no
NEG_RING_INNER="${NEG_RING_INNER:-1}"     # annulus radii, in patch-grid units
NEG_RING_OUTER="${NEG_RING_OUTER:-3}"     # 3 = hard near-field only; raise to reach far-field
NEG_PER_POS="${NEG_PER_POS:-1.0}"         # negatives per positive, capped by availability

# --- validation negatives ---------------------------------------------------
# The knobs above shape TRAINING data. VAL_NEGS=yes puts negatives in the
# VALIDATION set too, so val/dice can finally see a false positive on
# background instead of scoring only the positive patches. Its configuration is
# FIXED in the code (ring 1..3, 1:1, drawn once from the validation
# interferograms with SEED) and deliberately has no knobs here: it is the ruler,
# not the experiment. Any two runs sharing a partition and a seed are scored on
# identical samples whatever their architecture or RING_NEGS settings, which is
# what this model most needs -- its whole claim is a PRECISION claim, and the
# temporal_k5 head-to-head came in at 0.6459 against the ConvLSTM's 0.6460 on a
# val set that could not see precision at scene scale.
#
# It does NOT make a run comparable with one trained before it existed: a
# positives-only val set and a 1:1 one are different scales.
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

case "$RECURRENCE" in
  none|convlstm) ;;
  *) echo "RECURRENCE must be 'none' or 'convlstm', got '$RECURRENCE'" >&2; exit 1 ;;
esac

# k=10 doubles the sequence, and fused skips hold one attention map per level at
# the skip's own resolution -- both land on VRAM. 36G is the estimated k5 rung
# (~24 GiB peak plus 50% margin, see scripts/submit_all.sh); either of these
# pushes past it.
if [ "$K_PREVS" -ge 10 ] || [ "$FUSE_SKIPS" -ge 3 ]; then
  echo "NOTE: K_PREVS=$K_PREVS FUSE_SKIPS=$FUSE_SKIPS -- consider gmem=48G (this file asks for 36G)" >&2
fi

# Ring negatives reach the TRAIN dataset only -- train.py passes them to the
# train split and not to val/test. With VAL_NEGS=no, validation therefore stays
# positives-only and val/dice CANNOT see the thing this flag exists to fix:
# expect it flat or slightly down while scene-level precision improves, and
# judge the run with scripts/eval/run_eval.sh (object-level F1) rather than the
# results.csv curve. VAL_NEGS=yes is the fix for it.
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

JOB="${LSB_JOBNAME:-tattn_manual}"

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
  --tattn_unet \
  --tattn_dim "$DIM" \
  --tattn_heads "$HEADS" \
  --tattn_layers "$LAYERS" \
  --tattn_recurrence "$RECURRENCE" \
  --tattn_fuse_skips "$FUSE_SKIPS" \
  --convlstm_hidden "$HIDDEN" \
  ${NEG_FLAGS[@]+"${NEG_FLAGS[@]}"} \
  --amp \
  --save_best_only \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --job_name "$JOB" \
  --resume "$RESUME"
