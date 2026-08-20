#!/usr/bin/env bash
# ============================================================================
#  Positives-only evaluation template  (the benchmark paper's protocol)
#
#  Submit:  bsub < scripts/eval/run_eval_positives.sh
#
#  Scores a checkpoint ONLY on ground that is known to hold subsidence --
#  "delineate subsidence where it is known to be", not "find it anywhere on
#  the map". Three stages:
#    1. test-patches   builds the split in place from the partition JSON (a
#                      preset_by_intf run pickles none) and reports patch Dice
#                      + pixel and object-level P/R over positive patches.
#    2. eval-scenes    the same restriction at scene scale: --positives_only
#                      predicts only tiles whose ground truth is non-empty.
#    3. eval-outputs   re-thresholds the saved confidence into a JSON.
#
#  THESE NUMBERS ARE FOR COMPARING AGAINST THE PAPER AND NOTHING ELSE. They
#  are not comparable with the full-scene numbers from run_eval.sh, and they
#  must never be used to choose a model: a positives-only set cannot see the
#  false positives a model scatters over the 97.5% of the map that holds no
#  subsidence, which is exactly how it ranked the ring-negative runs backwards
#  (docs/RESULTS.md, docs/POSITIVES_ONLY_EVAL.md).
#
#  EDIT TWO PLACES: the "#BSUB -J" line and the CONFIG block. ARCH, K_PREVS and
#  the patch geometry must match how the checkpoint was TRAINED -- recover them
#  from the run's .log header (scripts/train/PRESETS.md records them too).
# ============================================================================
#BSUB -J eval_pos_geo_k10_test
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/%J.err
#BSUB -q long-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=48G
#BSUB -R rusage[mem=80GB]
#BSUB -W 8:00

set -o pipefail

# ------------------------------ CONFIG --------------------------------------
# Each honours an environment variable of the same name, so a batch can be
# driven without editing the file (see scripts/submit_all.sh).
RUN="${RUN:-outputs/2026-08-06/geo_k10_convlstm_posw8}"
CKPT="${CKPT:-best.pt}"
GROUP="${GROUP:-geo_k10}"   # geo_k5 | geo_k10 | temporal_k5 | temporal_k10
SPLIT="${SPLIT:-test}"      # val = the split the model was selected on
                            # test = the held-out split (never seen)
ARCH="${ARCH:-convlstm}"    # convlstm | tattn | stack | single -- must match training
K_PREVS="${K_PREVS:-10}"    # must match the CHECKPOINT, not the partition
RECON_TH="${RECON_TH:-0.25}"
OL_TH="${OL_TH:-0.7}"       # object-level overlap threshold
BUFFER="${BUFFER:-5}"       # object-matching buffer, pixels
SCENES="${SCENES:-1}"       # 0 = patch level only (much faster)
MIN_POS="${MIN_POS:-150}"   # stage-2 SCENE GATE: skip interferograms with this
                            # many positive patches or fewer. Stage 1
                            # (test-patches) is unaffected -- it scores patches,
                            # not scenes, so a thin interferogram contributes
                            # proportionally there rather than as a full scene.
                            # 0 disables. See scripts/eval/run_eval.sh.
# ----------------------------------------------------------------------------

REPO=/home/labs/rudich/pinkas/sinkholes
PATCHES=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches
cd "$REPO" || { echo "cannot cd to $REPO" >&2; exit 1; }

# -- resolve the partition ----------------------------------------------------
# Unlike eval-scenes, `test-patches --partition_file` reads whichever list is
# named by --split, so the parent partition serves every split directly. The
# _testeval variants stay in use for stage 2, whose --intf_source preset only
# ever reads a "val" list.
PARTITION="assets/partition_${GROUP}.json"
case "$SPLIT" in
  val)  SCENE_PARTITION="assets/partition_${GROUP}.json" ;;
  test) SCENE_PARTITION="assets/partition_${GROUP}_testeval.json" ;;
  *)    echo "SPLIT must be 'val' or 'test', got '$SPLIT'" >&2; exit 1 ;;
esac

# -- architecture flags -------------------------------------------------------
# The factory detects the architecture from the checkpoint's own tensors; these
# flags are the override, and they carry --k_prevs, which the detector cannot
# supply because it is a property of the data stack rather than the weights.
# --add_temporal is what makes test-patches build the k-previous stacks.
case "$ARCH" in
  convlstm) ARCH_FLAGS=(--convlstm_unet --k_prevs "$K_PREVS"); TEMPORAL=(--add_temporal) ;;
  tattn)    ARCH_FLAGS=(--tattn_unet --k_prevs "$K_PREVS");    TEMPORAL=(--add_temporal) ;;
  stack)    ARCH_FLAGS=(--k_prevs "$K_PREVS");                 TEMPORAL=(--add_temporal) ;;
  single)   ARCH_FLAGS=(--k_prevs 0);                          TEMPORAL=() ;;
  *)        echo "ARCH must be 'convlstm', 'tattn', 'stack' or 'single', got '$ARCH'" >&2; exit 1 ;;
esac

# -- preflight: fail here, not 20 minutes into a GPU slot ---------------------
MODEL="$RUN/checkpoints/$CKPT"
[ -f "$MODEL" ]     || { echo "no checkpoint at $MODEL" >&2; exit 1; }
[ -f "$PARTITION" ] || { echo "no partition file $PARTITION" >&2; exit 1; }

# K_PREVS must match the CHECKPOINT, not the partition -- at eval time the
# partition only supplies the list of interferograms to score. Deliberately
# mismatching them is how a k5 model is compared against a k10 one on a single
# scene list: geo_k10's test split is a subset of geo_k5's and every scene in it
# has 5- AND 10-previous chains, so the k10 list is the common ground.
if [ "$ARCH" != "single" ]; then
  case "$GROUP" in
    *_k5)  [ "$K_PREVS" = 5 ]  || echo "NOTE: K_PREVS=$K_PREVS on group $GROUP (k5 scene list)." >&2 ;;
    *_k10) [ "$K_PREVS" = 10 ] || echo "NOTE: K_PREVS=$K_PREVS on group $GROUP (k10 scene list) -- expected when scoring a k5 model on the shared list." >&2 ;;
  esac
fi

RUN_GROUP=$(basename "$RUN" | tr '-' '_' | grep -oE '(geo|temporal)_k[0-9]+' | head -1)
if [ -n "$RUN_GROUP" ] && [ "$RUN_GROUP" != "$GROUP" ]; then
  if [ "${RUN_GROUP%_k*}" = "${GROUP%_k*}" ]; then
    # Same partition family, different chain length: the shared-scene-list
    # comparison above. Legitimate, and the scores stay comparable.
    echo "NOTE: '$RUN_GROUP' model scored on the '$GROUP' scene list (the shared list)." >&2
  else
    echo "NOTE: run was trained on '$RUN_GROUP' but you are evaluating on '$GROUP'." >&2
    echo "      Cross-partition scores are not comparable to either group." >&2
  fi
fi

JOB="${LSB_JOBNAME:-eval_pos_manual}"
OUTROOT="outputs/predictions_positives/$(basename "$RUN")"
mkdir -p "$OUTROOT"

echo "model     $MODEL"
echo "partition $PARTITION  ($SPLIT split of $GROUP)"
echo "arch      $ARCH (k_prevs ${K_PREVS})"
echo "protocol  POSITIVES-ONLY -- compare against the paper, never to select a model"

source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# -- stage 1: patch level -----------------------------------------------------
python -m sinkholes test-patches \
  --model "$MODEL" \
  --partition_file "$PARTITION" --split "$SPLIT" \
  --patches_dir "$PATCHES" \
  --patch_size 200 100 --stride 2 \
  "${ARCH_FLAGS[@]}" "${TEMPORAL[@]+"${TEMPORAL[@]}"}" \
  --th "$OL_TH" --b "$BUFFER" \
  --metrics_out "$OUTROOT/${JOB}_patches_${GROUP}_${SPLIT}.json"

echo "patch-level metrics: $OUTROOT/${JOB}_patches_${GROUP}_${SPLIT}.json"
[ "$SCENES" = "1" ] || { echo "done (SCENES=0, patch level only)"; exit 0; }

# -- stage 2: scene level, positives-only gating ------------------------------
python -m sinkholes eval-scenes \
  --model "$MODEL" \
  --input_patch_dir "$PATCHES" \
  --intf_source preset --valset_from_partition "$SCENE_PARTITION" \
  --patch_size 200 100 --data_stride 2 --days_diff 11 \
  --recon_th "$RECON_TH" \
  "${ARCH_FLAGS[@]}" \
  --add_lidar_mask --positives_only \
  --min_positives "$MIN_POS" \
  --save_confidence --merge_polygs \
  --job_name "$JOB" --output_dir "$OUTROOT"

EVAL_DIR=$(ls -dt "$OUTROOT/${CKPT%.pt}/${JOB}"_* | head -1)
echo "eval-scenes wrote $EVAL_DIR"

# -- stage 3: object-level metrics + per-interferogram figures ----------------
python -m sinkholes eval-outputs \
  --path "$EVAL_DIR" \
  --th "$OL_TH" --buffer "$BUFFER" \
  --save_figures

echo "done: $EVAL_DIR"
