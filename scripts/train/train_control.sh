#!/usr/bin/env bash
# ============================================================================
#  Control / ablation training template -- the two non-recurrent baselines
#
#  Submit:  bsub < scripts/train/train_control.sh
#
#  ARCH picks the architecture:
#    single -- one interferogram in, plain U-Net. The floor that shows what
#              temporal context is worth (+0.046 dice on both partitions).
#    stack  -- k+1 frames stacked into the U-Net's input channels instead of
#              consumed recurrently. The control for the ConvLSTM itself; it
#              lost by 0.0143 on the temporal partition, ~2.4x the noise floor.
#
#  Everything else works like train_convlstm.sh: edit "#BSUB -J" and CONFIG.
#  Keep lr/batch identical to the ConvLSTM run you are comparing against --
#  the 2026-08-05 baselines used b64/lr1e-5 against b128/lr1e-6 ConvLSTMs,
#  which confounded architecture with optimizer and cost a clean answer.
# ============================================================================
#BSUB -J baseline_single_geo_k10_b128_lr1e6_60e
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
# without editing the file: ARCH=stack bsub -J <name> < this_script
ARCH="${ARCH:-single}"                    # single | stack
PARTITION="${PARTITION:-assets/partition_geo_k10.json}"  # geo_k5|geo_k10|temporal_k5|temporal_k10
K_PREVS="${K_PREVS:-10}"                  # stack only; ignored when ARCH=single
POS_W="${POS_W:-8}"
SEED="${SEED:-42}"
LR="${LR:-1e-6}"
SCHEDULE="${SCHEDULE:-plateau}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-128}"
PATIENCE="${PATIENCE:-20}"

# RING_NEGS=yes adds all-zero patches drawn from an annulus around the
# positives. Candidates must be empty at EVERY timestep, so a patch that was
# positive last month never enters as a negative (dataset.py:69-97).
#
# This works for ARCH=single as well as ARCH=stack. For a single-frame run the
# exclusion grid is still the union over the K_PREVS chain, and the sampler and
# seed are the same, so the negatives are EXACTLY those a temporal run on this
# partition draws -- which is the whole point of running this as a control.
# K_PREVS therefore matters even when ARCH=single, where it is otherwise
# ignored: it selects the chain the exclusion is built from.
RING_NEGS="${RING_NEGS:-no}"              # yes | no
NEG_RING_INNER="${NEG_RING_INNER:-1}"     # annulus radii, in patch-grid units
NEG_RING_OUTER="${NEG_RING_OUTER:-3}"     # 3 = hard near-field only; raise to reach far-field
NEG_PER_POS="${NEG_PER_POS:-1.0}"         # negatives per positive, capped by availability

# VAL_NEGS=yes additionally puts negatives in the VALIDATION set, so val/dice
# can see a false positive on background rather than scoring positive patches
# alone. Its configuration is FIXED in the code (ring 1..3, 1:1, drawn once from
# the validation interferograms with SEED) and has no knobs here on purpose: it
# is the ruler, not the experiment. Two runs on one partition and seed are then
# scored on identical samples whatever their architecture -- which is the whole
# reason this control exists, so if the ConvLSTM arm sets it, this one must too.
#
# For ARCH=single it makes the VALIDATION loader take the same coordinate-based
# path the training split already takes under RING_NEGS=yes (full grids plus the
# chain's mask grids, rather than the pre-extracted nonz files), so K_PREVS
# matters here for the same reason it matters there.
VAL_NEGS="${VAL_NEGS:-no}"                # yes | no   (needs SEED)
# ----------------------------------------------------------------------------

case "$RING_NEGS" in
  yes) NEG_FLAGS=(--add_ring_negatives
                  --neg_ring_inner "$NEG_RING_INNER"
                  --neg_ring_outer "$NEG_RING_OUTER"
                  --neg_per_pos "$NEG_PER_POS") ;;
  no)  NEG_FLAGS=() ;;
  *)   echo "RING_NEGS must be 'yes' or 'no', got '$RING_NEGS'" >&2; exit 1 ;;
esac

case "$VAL_NEGS" in
  yes) NEG_FLAGS+=(--add_val_negatives) ;;
  no)  ;;
  *)   echo "VAL_NEGS must be 'yes' or 'no', got '$VAL_NEGS'" >&2; exit 1 ;;
esac

# Ring negatives need the coordinate list, which the spatial loader never
# builds -- train.py rejects that combination rather than silently training
# without them, but catch it here too, before a GPU slot is taken.
# VAL_NEGS=yes is included: its negatives are drawn against the same chain, so a
# mismatched K_PREVS would move the VALIDATION set as well as the training one.
if { [ "$RING_NEGS" = yes ] || [ "$VAL_NEGS" = yes ]; } && [ "$ARCH" = single ]; then
  case "$PARTITION" in
    *_k5.json)  [ "$K_PREVS" = 5 ]  || { echo "negatives with ARCH=single need K_PREVS=5 with a _k5 partition (it selects the exclusion chain)" >&2; exit 1; } ;;
    *_k10.json) [ "$K_PREVS" = 10 ] || { echo "negatives with ARCH=single need K_PREVS=10 with a _k10 partition (it selects the exclusion chain)" >&2; exit 1; } ;;
  esac
fi

case "$ARCH" in
  # --k_prevs without --add_temporal does not add frames to the input; it only
  # names the chain that RING_NEGS builds its exclusion grid from. Harmless
  # when RING_NEGS=no, and required for the negatives to match a temporal run.
  single) ARCH_FLAGS=(--k_prevs "$K_PREVS") ;;
  stack)  ARCH_FLAGS=(--add_temporal --k_prevs "$K_PREVS")
          case "$PARTITION" in
            *_k5.json)  [ "$K_PREVS" = 5 ]  || { echo "K_PREVS=$K_PREVS with a _k5 partition"  >&2; exit 1; } ;;
            *_k10.json) [ "$K_PREVS" = 10 ] || { echo "K_PREVS=$K_PREVS with a _k10 partition" >&2; exit 1; } ;;
          esac ;;
  *)      echo "ARCH must be 'single' or 'stack', got '$ARCH'" >&2; exit 1 ;;
esac

JOB="${LSB_JOBNAME:-control_manual}"

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
  ${ARCH_FLAGS[@]+"${ARCH_FLAGS[@]}"} \
  ${NEG_FLAGS[@]+"${NEG_FLAGS[@]}"} \
  --amp \
  --save_best_only \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --job_name "$JOB" \
  --resume auto
