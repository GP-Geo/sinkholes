#!/usr/bin/env bash
# ============================================================================
#  submit_all.sh -- the shield in front of bsub.
#
#      scripts/submit_all.sh                     # list every live job (DRY RUN)
#      scripts/submit_all.sh eval6ref            # list one kind
#      scripts/submit_all.sh long200             # the 200-epoch temporal rerun
#      scripts/submit_all.sh eval6ref --submit   # send it
#      scripts/submit_all.sh --only g5_single    # narrow by name, AND-ed with the kind
#
#  DRY RUN IS THE DEFAULT. Nothing reaches LSF without --submit.
#
#  WHAT THIS FILE IS. A table of jobs that have NEVER RUN, plus the preflight
#  that stops a doomed one before it reaches a compute node. It is deliberately
#  not the project's notebook: a batch that has run is recorded in
#  docs/EXPERIMENTS.md and REMOVED from here, so `--submit` can never silently
#  repeat finished GPU-hours. That failure has happened -- the 2026-08-10 batch
#  sat here after completing and `train --submit` would have retrained ~60
#  GPU-hours of finished work.
#
#  To recover any retired batch's exact job definitions:
#      git show d708fba:scripts/submit_all.sh
#
#  WHAT THE PREFLIGHT CATCHES, on the login node rather than 20 minutes into a
#  GPU allocation:
#    * a template that does not exist;
#    * RUN= whose checkpoints/best.pt is gone (deleted or moved by a tidy);
#    * DIR= for a re-score whose *_pred.npy were pruned -- re-scoring reads
#      exactly those, and only a full run_eval.sh can rebuild them.
#
#  RESOURCES are (queue, host GB, GPU GB, walltime). They are measured, not
#  guessed; the comment on each line says what it was measured from.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f scripts/train/train_convlstm.sh ] || [ ! -d assets ]; then
  echo "run this from the repo (cd to it); assets/ and scripts/train/ must be here" >&2
  exit 1
fi

KINDS=(); NAMES=(); TEMPLATES=(); OVERRIDES=(); RESOURCES=(); WHYS=()
job() { KINDS+=("$1"); NAMES+=("$2"); TEMPLATES+=("$3"); OVERRIDES+=("$4"); RESOURCES+=("$5"); WHYS+=("$6"); }

# ---- resources --------------------------------------------------------------
RES_EVAL_S4_G5="long-gpu    144   36  12:00"   # measured 109.6 GiB peak
RES_EVAL_S4_G10="long-gpu   208   48  12:00"   # est 161.5 GiB (144 died at 100%)
RES_EVAL_S4_T5="long-gpu    160   36  16:00"   # geo k5 + larger AOI residual
RES_EVAL_S4_CTRL="long-gpu   48   24   6:00"   # measured 27.8 GiB, single-frame

# ---- eval6ref: the anchors the nineteen are read against --------------------
#
#     submit_all.sh eval6ref --submit     # 4 jobs
#
# Four positives-only runs of 2026-08-19, still carrying best.pt. They are not
# part of the nineteen; they are what makes the nineteen readable.
#
# WHY THEY MATTER. The attnpos batch shipped no single-frame arm and only one
# ConvLSTM (docs/EXPERIMENTS.md), so every one of its rows currently has to be
# read against a pre23 comparator -- across a training-archive boundary. These
# four close that gap with evaluations only, no retraining.
#
# geo_k5_single_ring3_valpos IS UNFINISHED -- killed at epoch 93 of 100 with no
# completion line. Its best.pt is from epoch 59 and the seven lost epochs ran at
# lr 1.95e-08 against a val/dice flat in the fourth decimal, so the checkpoint
# is not in question; only the log line is missing. It is scoreable as it stands.
R_G10_CONVLSTM=outputs/2026-08-19/geo_k10_convlstm_ring3_valpos
R_G5_CONVLSTM=outputs/2026-08-19/geo_k5_convlstm_ring3_valpos
R_G5_SINGLE=outputs/2026-08-19/geo_k5_single_ring3_valpos
R_T5_CONVLSTM=outputs/2026-08-19/temporal_k5_convlstm_ring3_valpos

EVAL6_GEO="GEN=3 GROUP=geo_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_geo_clean_rth"
EVAL6_TEMP="GEN=3 GROUP=temporal_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_clean_rth"

job eval6ref eval6ref_g5_single \
    scripts/eval/run_eval.sh "RUN=$R_G5_SINGLE $EVAL6_GEO ARCH=single" "$RES_EVAL_S4_CTRL" \
    "the highest-value job of the four: on the patch curve this model TOPS its geo batch on both dice and val/F1 while its twin is 0.076 obj F1 behind. One of those two readings is wrong"
job eval6ref eval6ref_g5_convlstm \
    scripts/eval/run_eval.sh "RUN=$R_G5_CONVLSTM $EVAL6_GEO K_PREVS=5" "$RES_EVAL_S4_G5" \
    "THE geo anchor under the current protocol -- every geo row in eval6 is read against this one"
job eval6ref eval6ref_g10_convlstm \
    scripts/eval/run_eval.sh "RUN=$R_G10_CONVLSTM $EVAL6_GEO K_PREVS=10" "$RES_EVAL_S4_G10" \
    "the k10 non-attention reference, so eval6's two geo_k10 attention arms have a same-protocol baseline at their own depth"
job eval6ref eval6ref_t5_convlstm \
    scripts/eval/run_eval.sh "RUN=$R_T5_CONVLSTM $EVAL6_TEMP K_PREVS=5" "$RES_EVAL_S4_T5" \
    "the temporal anchor, completing the pair for every temporal row in eval6"

# ---- long200: the temporal ConvLSTM taken to 200 epochs, early stopping off --
#
#     submit_all.sh long200 --submit     # 1 job
#
# pre23_temporal_k5_convlstm_ring3 is the best temporal model on record
# (obj F1 0.780, RESULTS.md finding 9). Its 2026-08-20 run never finished: the
# log stops mid-epoch at 91/100 with no early-stopping line and no completion
# line, so best.pt is from an interrupted run, not a converged one. This is the
# same config carried to 200 epochs with early stopping switched off.
#
# EVERY HYPER-PARAMETER IS THE ORIGINAL except EPOCHS, PATIENCE and LR_PATIENCE,
# recovered from `git show d708fba:scripts/submit_all.sh` ($P_T5 $NEG_R3_1X
# $C_HYP): LR 1e-5, POS_W 4, ring negatives 1..3 at 1:1, HIDDEN 256, BATCH 128,
# plateau, SEED 42.
#
# PATIENCE=0 IS how early stopping is disabled -- reporter.py:257 gates the
# stop on `bool(self.patience)`, so 0 means never stop. It is also train.py's
# own default; the templates set 20.
#
# LR_PATIENCE=20 IS A REAL CHANGE, AND THE ONE THAT MAKES THE EPOCHS WORTH
# BUYING. The original died flat: by epoch 91 the plateau schedule had taken lr
# to 1.95e-08 -- the --min_lr floor -- and val/dice had sat in the fourth
# decimal for ~30 epochs. At LR_PATIENCE=5 another 100 epochs would run at an lr
# that cannot move the weights. 20 cuts the LR roughly a quarter as often, so
# the run spends its epochs at a live lr instead of on the floor.
#
# It cuts BOTH ways and the comparison should be read knowing it. train.py:259
# argues the short patience is deliberate -- on the k10split run the first cut
# is what broke a five-epoch plateau -- so a longer one can also just sit on a
# plateau it should have escaped. This arm therefore differs from its eval6 row
# by two factors, not one, and a difference in object F1 cannot be attributed to
# the epoch count alone. That is an acceptable trade for a run whose purpose is
# a converged model rather than a controlled comparison, but it does mean this
# is a NEW experiment and not a re-run of the old one.
#
# This template did not expose --lr_patience before 2026-09-03; the knob was
# added for this job, defaulting to the code's 5, so no other job moves.
LONG200_T5="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
LONG200_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
LONG200_HYP="LR=1e-5 EPOCHS=200 PATIENCE=0 LR_PATIENCE=20"

# Measured from the original run, not estimated: 91 epochs in 5h01m = 3.31
# min/epoch, so 200 epochs is ~11h02m. 18:00 is that plus 60% headroom.
#
# HOST MEMORY IS 90GB BY REQUEST, not by measurement. The pre23 batch estimated
# this arm's peak at 44G and asked RES_P_T5=64G, and nothing in this job changes
# the memory profile -- same partition, same ring negatives, same batch 128, and
# epoch count does not affect stored samples. 90G is deliberate headroom over
# that, cheap on long-gpu and harmless if unused. VRAM stays at RES_P_T5's 36G.
RES_LONG200_T5="long-gpu    90   36  18:00"   # 11h02m measured; 90G host by request (est peak 44G)

job long200 long200_t5_convlstm_ring3_200e \
    scripts/train/train_convlstm.sh "$LONG200_T5 $LONG200_NEG $LONG200_HYP" "$RES_LONG200_T5" \
    "the best temporal model rerun to completion: pre23_temporal_k5_convlstm_ring3 config, 200 epochs, early stopping off, LR_PATIENCE 5->20 so the epochs run at a live lr"

# ---- argument parsing -------------------------------------------------------
WANT=all; SUBMIT=no; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    eval6ref|long200|all)   WANT="$1" ;;
    # Retired by name rather than left to select zero jobs, so the mistake is
    # visible instead of looking like an empty batch. See docs/EXPERIMENTS.md.
    train|tattn|control|clean22|valpos|attnfix|attnpos|pre23|training|\
    eval|eval2|eval3|eval4|eval5|eval6|eval7|probe|posonly)
      echo "'$1' has already run -- see docs/EXPERIMENTS.md for what it settled." >&2
      echo "Recover its job definitions with: git show d708fba:scripts/submit_all.sh" >&2
      exit 1 ;;
    valneg)
      echo "the 'valneg' batch is retired: validation is positives-only." >&2; exit 1 ;;
    blend)
      echo "'blend' cannot run: both checkpoints were deleted (see docs/EXPERIMENTS.md)." >&2; exit 1 ;;
    rescore)
      echo "'rescore' cannot run: its eval directory has no *_pred.npy left." >&2; exit 1 ;;
    --submit)       SUBMIT=yes ;;
    --only)         shift; [ $# -gt 0 ] || { echo "--only needs a value" >&2; exit 1; }
                    ONLY+=("$1") ;;
    --only=*)       ONLY+=("${1#--only=}") ;;
    -h|--help)      sed -n '2,/measured from\./p' "$0"; exit 0 ;;
    *)              echo "unknown argument '$1'" >&2; exit 1 ;;
  esac
  shift
done

selected() {
  local kind="$1" name="${2-}"
  if [ "${#ONLY[@]}" -gt 0 ]; then          # --only is an AND on top of the kind
    local hit=no pat
    for pat in "${ONLY[@]}"; do
      case "$name" in *"$pat"*) hit=yes ;; esac
    done
    [ "$hit" = yes ] || return 1
  fi
  [ "$WANT" = all ] || [ "$WANT" = "$kind" ]
}

if [ "$SUBMIT" = yes ] && ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found -- run this on the cluster, not the mounted volume." >&2
  exit 1
fi

# ---- preflight --------------------------------------------------------------
fail=0
for i in "${!NAMES[@]}"; do
  selected "${KINDS[$i]}" "${NAMES[$i]}" || continue
  [ -f "${TEMPLATES[$i]}" ] || { echo "missing template ${TEMPLATES[$i]}" >&2; fail=1; }
  case "${OVERRIDES[$i]}" in
    *RUN=*) r=$(sed 's/.*RUN=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            [ -f "$r/checkpoints/best.pt" ] || { echo "missing checkpoint $r/checkpoints/best.pt" >&2; fail=1; } ;;
    *DIR=*) d=$(sed 's/.*DIR=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            if [ ! -d "$d" ]; then
              echo "missing eval directory $d" >&2; fail=1
            elif [ -z "$(find "$d" -maxdepth 1 -name '*_pred.npy' -print -quit)" ]; then
              echo "no *_pred.npy in $d -- rebuild it with run_eval.sh, not rescore.sh" >&2; fail=1
            fi ;;
  esac
done
[ "$fail" = 0 ] || { echo "preflight failed; nothing submitted" >&2; exit 1; }

# ---- summary ----------------------------------------------------------------
printf "%-46s %-10s %6s %6s %7s\n" "JOB" "QUEUE" "MEM" "VRAM" "WALL"
printf "%-46s %-10s %6s %6s %7s\n" "$(printf '%.0s-' {1..46})" "----------" "------" "------" "-------"
for i in "${!NAMES[@]}"; do
  selected "${KINDS[$i]}" "${NAMES[$i]}" || continue
  read -r q mem vram wall <<<"${RESOURCES[$i]}"
  printf "%-46s %-10s %5sG %5sG %7s\n" "${NAMES[$i]:0:46}" "$q" "$mem" "$vram" "$wall"
done

# ---- run --------------------------------------------------------------------
n=0; ok=0; FAILED=()
for i in "${!NAMES[@]}"; do
  selected "${KINDS[$i]}" "${NAMES[$i]}" || continue
  n=$((n+1))
  read -r q mem vram wall <<<"${RESOURCES[$i]}"
  BSUB_ARGS=(-J "${NAMES[$i]}" -q "$q"
             -R "rusage[mem=${mem}GB]"
             -gpu "num=1:j_exclusive=yes:gmem=${vram}G"
             -W "$wall")
  echo
  echo "[${KINDS[$i]}] ${NAMES[$i]}"
  echo "     why: ${WHYS[$i]}"
  echo "     res: $q  mem=${mem}GB  gmem=${vram}G  W=$wall"
  echo "     cmd: env ${OVERRIDES[$i]} bsub ${BSUB_ARGS[*]} < ${TEMPLATES[$i]}"
  if [ "$SUBMIT" = yes ]; then
    read -ra kv <<<"${OVERRIDES[$i]}"
    # `if` keeps set -e from aborting the batch on one rejection: a bad queue
    # limit on job 2 must not silently cancel jobs 3 and 4.
    if env "${kv[@]}" bsub "${BSUB_ARGS[@]}" < "${TEMPLATES[$i]}"; then
      ok=$((ok+1))
    else
      echo "     !! SUBMISSION FAILED -- continuing with the rest" >&2
      FAILED+=("${NAMES[$i]}")
    fi
  fi
done

echo
if [ "$SUBMIT" = yes ]; then
  echo "submitted $ok of $n job(s). Track with: bjobs -w"
  if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "failed to submit:"; printf '  %s\n' "${FAILED[@]}"
  fi
else
  echo "$n job(s) listed. DRY RUN -- nothing was submitted. Add --submit to send them."
fi
