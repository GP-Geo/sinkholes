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
# No default on purpose: assets/ holds three generations of partition
# (assets/PARTITIONS.md). Defaulting to one is how a clean-data run silently
# trains on the 2019-2026 noisy lists, so set it explicitly every time.
PARTITION="${PARTITION:?set PARTITION=assets/partition_<axis>_k<k>[_clean].json -- see assets/PARTITIONS.md}"
K_PREVS="${K_PREVS:-5}"                   # temporal depth; MUST match the partition's k
POS_W="${POS_W:-4}"                       # BCE positive weight (code default is 1)
SEED="${SEED:-42}"
MOMENTUM="${MOMENTUM:-0.999}"             # see lrscan: 0.999 diverges on corrected clipping
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-8}"      # 1e-8 is historical and is barely regularisation
                                          # at all; 1e-4..1e-2 is the usual range
AUGMENT="${AUGMENT:-none}"                # none | h | v | hv -- random flips, TRAIN split only
LR_PATIENCE="${LR_PATIENCE:-5}"
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

# --- frame selection --------------------------------------------------------
# Before 2026-08-19 the attention in every run here was DEAD: the weights came
# out at exactly 1/T, so the model averaged its history instead of choosing from
# it (docs/ATTENTION_COLLAPSE.md, 15 of 15 checkpoints). The cause was visible at
# initialisation -- the frame-to-frame differences are 0.09% of the token the
# queries and keys are built from, so the softmax had nothing to separate.
#
# CONTRAST builds q/k from token - mean_over_time(token), so selection runs on
# how the frames DIFFER. QK_NORM unit-norms q/k and puts the logit scale on one
# learned temperature, so selectivity stops riding on projection magnitude --
# without it the same block goes uniform at LR=1e-6 and one-hot at LR=1e-5.
# Neither alone is enough; together they take effective frames used from
# 10.97/11 to 4.43/11 at init and hold it through training.
#
# Set BOTH to no to reproduce a pre-2026-08-19 run exactly.
CONTRAST="${CONTRAST:-yes}"               # yes | no
QK_NORM="${QK_NORM:-yes}"                 # yes | no

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

for v in CONTRAST QK_NORM; do
  case "${!v}" in
    yes|no) ;;
    *) echo "$v must be 'yes' or 'no', got '${!v}'" >&2; exit 1 ;;
  esac
done

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
# train split and not to val/test. Validation is therefore positives-only and
# val/dice CANNOT see the thing this flag exists to fix: expect it flat or
# slightly down while scene-level precision improves, and judge the run with
# scripts/eval/run_eval.sh (object-level F1) rather than the results.csv curve.
# Read val/F1 there too, not val/dice -- it is pooled over pixels and does move.
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
  --lr_patience "$LR_PATIENCE" \
  --momentum "$MOMENTUM" \
  --weight_decay "$WEIGHT_DECAY" \
  --augment_flips "$AUGMENT" \
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
  ${CONTRAST:+$([ "$CONTRAST" = yes ] && echo --tattn_contrast || echo --no-tattn_contrast)} \
  ${QK_NORM:+$([ "$QK_NORM" = yes ] && echo --tattn_qk_norm || echo --no-tattn_qk_norm)} \
  --convlstm_hidden "$HIDDEN" \
  ${NEG_FLAGS[@]+"${NEG_FLAGS[@]}"} \
  --amp \
  --save_best_only \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --job_name "$JOB" \
  --resume "$RESUME"
