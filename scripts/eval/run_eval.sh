#!/usr/bin/env bash
# ============================================================================
#  Full-scene evaluation template
#
#  Submit:  bsub < scripts/eval/run_eval.sh
#
#  Two stages, on whichever interferograms you point it at:
#    1. eval-scenes   reconstructs each interferogram from its patch grid (+ k
#                     previous frames), LiDAR-gates, thresholds, polygonises.
#                     Writes <intf>_{image,pred,pred_th,gt}.npy and shapefiles.
#    2. eval-outputs  re-thresholds the saved confidence at 0.125/0.25/0.5 and
#                     computes object-level precision/recall into a JSON.
#
#  EDIT TWO PLACES: the "#BSUB -J" line (names the job and the log files) and
#  the CONFIG block. See PRESETS.md for the evaluations run so far.
#
#  ARCH, K_PREVS and the patch geometry must match how the checkpoint was
#  TRAINED, not what you wish it were -- a mismatch either fails to load or
#  silently feeds the network the wrong stack. scripts/train/PRESETS.md records
#  what each run was trained with.
# ============================================================================
#BSUB -J eval_geo_k10_test
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/%J.err
# short-gpu rejects these: "RUNLIMIT: Cannot exceed queue's hard limit(s)".
# Its hard walltime cap is under 8:00, and a full eval measures ~3.0 h.
#BSUB -q long-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=48G
#BSUB -R rusage[mem=80GB]
#BSUB -W 8:00

set -o pipefail

# ------------------------------ CONFIG --------------------------------------
# Edit the defaults here for a one-off run. Each also honours an environment
# variable of the same name, which is how scripts/submit_all.sh drives batches
# without editing the file: GROUP=temporal_k10 K_PREVS=5 bsub -J <name> < this
RUN="${RUN:-outputs/2026-08-06/geo-k10_convlstm-h256_lsf206375}"
CKPT="${CKPT:-best.pt}"     # best.pt | last.pt (last.pt was pruned from most runs)
GROUP="${GROUP:-geo_k10}"   # geo_k5 | geo_k10 | temporal_k5 | temporal_k10
SPLIT="${SPLIT:-test}"      # val = the split the model was selected on
                            # test = the held-out split (never seen)
ARCH="${ARCH:-convlstm}"    # convlstm | tattn | stack | single -- must match training
K_PREVS="${K_PREVS:-10}"    # must match the CHECKPOINT, not the partition
RECON_TH="${RECON_TH:-0.25}"  # threshold on the reconstructed confidence map
OL_TH="${OL_TH:-0.7}"       # object-level overlap threshold (eval-outputs)
BUFFER="${BUFFER:-5}"       # object-matching buffer, pixels
# ----------------------------------------------------------------------------

REPO=/home/labs/rudich/pinkas/sinkholes
PATCHES=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches
# Every path below is relative to the repo root, so a failed cd must not be
# survivable -- otherwise the preflight checks run against the wrong tree.
cd "$REPO" || { echo "cannot cd to $REPO" >&2; exit 1; }

# -- resolve the partition ----------------------------------------------------
# `eval-scenes --intf_source preset` only ever reads a partition's "val" list
# (sinkholes/inference/scenes.py). To evaluate a TEST split we therefore point
# it at the _testeval variant, whose "val" is exactly the parent's "test" list.
case "$SPLIT" in
  val)  PARTITION="assets/partition_${GROUP}.json" ;;
  test) PARTITION="assets/partition_${GROUP}_testeval.json" ;;
  *)    echo "SPLIT must be 'val' or 'test', got '$SPLIT'" >&2; exit 1 ;;
esac

# -- architecture flags -------------------------------------------------------
# The factory DETECTS the architecture from the checkpoint's own tensors
# (sinkholes/models/factory.py:272); these flags are the override, and they
# also carry --k_prevs, which the detector cannot supply because it is a
# property of the data stack rather than the weights.
#
# tattn covers all four 2026-08-10 attention runs, fuse_skips and
# tattn_recurrence included: both are read back out of the checkpoint's
# tattn_unet_config blob, so the hybrid (RECURRENCE=convlstm) rebuilds
# correctly under --tattn_unet and must NOT be evaluated as ARCH=convlstm --
# factory.py:221 orders tattn ahead of convlstm precisely because the hybrid
# carries a ConvLSTM cell and would otherwise match the wrong entry.
case "$ARCH" in
  convlstm) ARCH_FLAGS=(--convlstm_unet --k_prevs "$K_PREVS") ;;
  tattn)    ARCH_FLAGS=(--tattn_unet --k_prevs "$K_PREVS") ;;
  stack)    ARCH_FLAGS=(--k_prevs "$K_PREVS") ;;
  single)   ARCH_FLAGS=(--k_prevs 0) ;;
  *)        echo "ARCH must be 'convlstm', 'tattn', 'stack' or 'single', got '$ARCH'" >&2; exit 1 ;;
esac

# -- preflight: fail here, not 20 minutes into a GPU slot ---------------------
MODEL="$RUN/checkpoints/$CKPT"
[ -f "$MODEL" ]     || { echo "no checkpoint at $MODEL" >&2; exit 1; }
[ -f "$PARTITION" ] || { echo "no partition file $PARTITION" >&2; exit 1; }
# K_PREVS must match the CHECKPOINT, not the partition -- at eval time the
# partition only supplies the list of interferograms to score. Deliberately
# mismatching them is how you compare a k5 model against a k10 model on one
# scene list: temporal_k10's test split is a subset of temporal_k5's (and
# geo_k10's of geo_k5's), and every scene in both has 5- AND 10-previous
# chains, so the k10 list is the common ground. Hence a note, not an error.
if [ "$ARCH" != "single" ]; then
  case "$GROUP" in
    *_k5)  [ "$K_PREVS" = 5 ]  || echo "NOTE: K_PREVS=$K_PREVS on group $GROUP (k5 scene list)." >&2 ;;
    *_k10) [ "$K_PREVS" = 10 ] || echo "NOTE: K_PREVS=$K_PREVS on group $GROUP (k10 scene list) -- expected when scoring a k5 model on the shared list." >&2 ;;
  esac
fi

# Run directories carry their training group somewhere in the name, so the
# run's own group is recoverable. Evaluating a model outside the partition it
# was trained on is a legitimate transfer experiment, but it is far more often
# a typo -- and its score is NOT comparable to that model's usual numbers
# (docs/MODEL_RUNS.md).
#
# Two naming schemes are in play and both must parse. Runs up to 2026-08-09
# lead with the group ("geo-k10_convlstm-h256_posw4_lsf594076"); the
# 2026-08-10 batch leads with the ARCHITECTURE instead
# ("convlstm_geo_k5_h256_posw4_neg1x_ring3_60e_..."), so the old
# `cut -d_ -f1` read "convlstm" as the group and warned on every single run.
# Match the group token wherever it sits; an unrecognised name (the retired
# k10split runs) yields empty and simply skips the check.
RUN_GROUP=$(basename "$RUN" | tr '-' '_' | grep -oE '(geo|temporal)_k[0-9]+' | head -1)
if [ -n "$RUN_GROUP" ] && [ "$RUN_GROUP" != "$GROUP" ]; then
  echo "NOTE: run was trained on '$RUN_GROUP' but you are evaluating on '$GROUP'." >&2
  echo "      Cross-partition scores are not comparable to either group." >&2
fi

JOB="${LSB_JOBNAME:-eval_manual}"
OUTROOT="outputs/predictions/$(basename "$RUN")"

echo "model     $MODEL"
echo "partition $PARTITION  ($SPLIT split of $GROUP)"
echo "arch      $ARCH (k_prevs ${K_PREVS})"
echo "output    $OUTROOT"

source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# -- stage 1: reconstruct, LiDAR-gate, threshold, polygonise ------------------
python -m sinkholes eval-scenes \
  --model "$MODEL" \
  --input_patch_dir "$PATCHES" \
  --intf_source preset --valset_from_partition "$PARTITION" \
  --patch_size 200 100 --data_stride 2 --days_diff 11 \
  --recon_th "$RECON_TH" \
  "${ARCH_FLAGS[@]}" \
  --add_lidar_mask \
  --save_confidence --merge_polygs \
  --job_name "$JOB" --output_dir "$OUTROOT"

# eval-scenes nests under <output_dir>/<checkpoint stem>/<job>_<timestamp>;
# it stamps the directory with its start time, so pick the newest.
EVAL_DIR=$(ls -dt "$OUTROOT/${CKPT%.pt}/${JOB}"_* | head -1)
echo "eval-scenes wrote $EVAL_DIR"

# -- stage 2: object-level metrics + per-interferogram figures ----------------
python -m sinkholes eval-outputs \
  --path "$EVAL_DIR" \
  --th "$OL_TH" --buffer "$BUFFER" \
  --save_figures

echo "done: $EVAL_DIR"
