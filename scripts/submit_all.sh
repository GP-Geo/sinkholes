#!/usr/bin/env bash
# ============================================================================
#  submit_all.sh -- the shield in front of bsub.
#
#      scripts/submit_all.sh                     # list every live job (DRY RUN)
#      scripts/submit_all.sh eval6ref            # list one kind
#      scripts/submit_all.sh long500             # 500-epoch temporal rerun
#      scripts/submit_all.sh ctx50               # the two 300x200 context arms
#      scripts/submit_all.sh evallong200         # score long200 on scenes
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

# ---- long500: does long200's curve actually have more to give? -------------
#
#     submit_all.sh long500 --submit     # 1 job
#
# long200 (LSF 643128, 2026-09-04) ran all 200 epochs and took its best val/dice
# ON THE LAST ONE -- 0.6622 @ 200, having set a new best at 198 as well. Read on
# its own that curve has not flattened, which is the reason for this arm.
#
# READ THE COMPARISON HONESTLY. long200's 0.6622 does NOT beat the run it was
# meant to improve on: the original pre23_temporal_k5_convlstm_ring3 hit 0.6626
# at epoch 51, and 200 epochs bought -0.0004 -- inside the +/-0.006 noise floor
# (RESULTS.md 1). Its best.pt sitting on the final epoch is what argues for
# more epochs; the flat 200-vs-51 result is what argues that they will not buy
# object F1. This arm settles that, and its verdict is the object-level score,
# not the dice curve (RESULTS.md 2).
#
# WHAT CHANGES FROM long200: EPOCHS 200->500, PATIENCE 0->200, LR_PATIENCE
# 20->35. Everything else is long200's config exactly.
#
# PATIENCE=200 rather than 0 (off) is deliberate: 200 epochs of no improvement
# is a real plateau by any reading, and it returns the walltime instead of
# burning 300 more epochs to prove it. LR_PATIENCE=35 continues what long200
# established -- at 20 the schedule cut five times and still ended at 3.13e-07,
# well clear of the 1.95e-08 floor the ORIGINAL died on, so the lr stayed live
# the whole way. 35 cuts less often again.
LONG500_T5="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
LONG500_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
LONG500_HYP="LR=1e-5 EPOCHS=500 PATIENCE=200 LR_PATIENCE=35"

# Measured from long200, not estimated: 200 epochs in 11h44m = 3m31s/epoch, so
# 500 epochs is ~29h20m. THAT DOES NOT FIT ONE ALLOCATION. -W 18:00 is the
# longest walltime this project has had granted; --resume auto (the template
# default) finds the run directory by LSF job id and continues from the last
# completed epoch, so expect ~2 requeues. VRAM is long200's measured 25.8 GiB
# reserved rounded up; host memory is long200's 90G, and epoch count does not
# change the resident sample count.
RES_LONG500_T5="long-gpu    90   36  18:00"   # 3m31s/epoch measured; ~2 requeues expected

job long500 long500_t5_convlstm_ring3_500e \
    scripts/train/train_convlstm.sh "$LONG500_T5 $LONG500_NEG $LONG500_HYP" "$RES_LONG500_T5" \
    "long200 set its best on the final epoch, so the epochs may not be spent: 500 of them, early stopping at 200, LR_PATIENCE 20->35"

# ---- ctx50: 300x200 of context, supervising the same centre 200x100 ---------
#
#     submit_all.sh ctx50 --submit       # 2 jobs
#
# The first runs on data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned
# (437/437 interferograms, 1.7 TB, LSF 640831). The network is fed 300x200 and
# its logits are cropped back to 200x100 inside forward(), so the loss, the
# metrics, the sample grid, the partitions, the ring negatives, the AOI window
# and the whole evaluation protocol are untouched. Only what surrounds each
# target changes, which is what makes these two arms readable against long200.
#
# EVERY HYPER-PARAMETER IS long200's except the context and the batch split.
# The comparator is therefore long200's OWN CURVE at the same epoch -- same
# partition, same negatives, same LR, same seed, same effective batch.
#
# BATCH=64 ACCUM=2, NOT BATCH=64. 300x200 is 3x the activations, so 128 does not
# fit; but batch_size is a STRICT resume key and every preset is 128, so halving
# it alone would confound the context with the optimiser (PLAN_LARGE_CONTEXT.md
# 6). Accumulation holds the effective batch at 128 and gives the optimiser the
# gradient a single batch of 128 would have produced. One difference survives
# and is not removable: BatchNorm sees one micro-batch at a time, so 384 samples
# per step (B x T, T=6) instead of 768.
#
# 50 px IS UNDER WHAT THE ARCHITECTURE COULD USE. The receptive field measured
# in PLAN_LARGE_CONTEXT.md 1 is 220 px for ConvLSTM-current, i.e. a usable
# margin of +/-110. This is the cheap arm at 3x, not the plan's 400x300 (6x) or
# 600x400 (12x); a null result here bounds the cheap end and does not settle
# whether a margin at the receptive-field limit would help.
CTX50_T5="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
CTX50_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
CTX50_GEOM="CTX_MY=50 CTX_MX=50 BATCH=64 ACCUM=2"
CTX50_HYP="LR=1e-5 LR_PATIENCE=20"

# HOST MEMORY IS THE RISK HERE, not VRAM. SubsiDataset still materialises every
# sample's pixels (the lazy loader PLAN_LARGE_CONTEXT.md 3 calls for was NOT
# built), and those pixels are now 3x larger: long200's ~44G estimated peak
# becomes ~90G, since the images scale and the 200x100 masks do not. 180G is
# deliberate headroom over that -- the eval jobs above already run at 208G.
# VRAM: long200 measured 25.8 GiB reserved at 128x200x100; half the batch at
# three times the pixels is ~1.5x that, ~39 GiB, so 56G carries ~40% headroom.
# WALLTIME: 3x the pixels is ~10m33s/epoch off long200's measured 3m31s.
RES_CTX50_SHORT="long-gpu   180   56  18:00"   # 60 ep x ~10m33s = ~10h33m, one allocation
RES_CTX50_LONG="long-gpu    180   56  18:00"   # 200 ep = ~35h10m, ~2 requeues via --resume auto

job ctx50 ctx50_t5_convlstm_ring3_60e \
    scripts/train/train_convlstm.sh "$CTX50_T5 $CTX50_NEG $CTX50_GEOM $CTX50_HYP EPOCHS=60 PATIENCE=0" "$RES_CTX50_SHORT" \
    "the cheap read: does 300x200 of context move the curve at all by epoch 60, against long200's own epoch 60"
job ctx50 ctx50_t5_convlstm_ring3_200e \
    scripts/train/train_convlstm.sh "$CTX50_T5 $CTX50_NEG $CTX50_GEOM $CTX50_HYP EPOCHS=200 PATIENCE=0" "$RES_CTX50_LONG" \
    "the matched arm: 200 epochs, early stopping off, directly against long200 -- same partition, negatives, LR, seed and effective batch, differing only in context"

# ---- evallong200: the object-level score long200 has never had --------------
#
#     submit_all.sh evallong200 --submit     # 1 job
#
# long200 has a dice curve and NO object-level score, and dice is the metric
# RESULTS.md 2 shows ranks these models backwards. Until this runs, "long200
# looks good" is a reading of the one number the project has ruled out.
#
# THE PROTOCOL IS eval6's, EXACTLY, because that is what produced the 0.780 this
# has to be read against: GEN=3 (generation-3 clean partitions), GROUP=temporal_k10
# for the shared 20-scene temporal list, DATA_STRIDE=4 and PROTOCOL=rth.
# K_PREVS=5 names the CHECKPOINT's depth, not the partition's -- run_eval.sh:264
# expects exactly this pairing and says so. Identical to the eval6ref_t5_convlstm
# row above, which is what makes the two directly comparable.
R_LONG200=outputs/long200_t5_convlstm_ring3_200e_2026-09-03_16h22_lsf_643128

job evallong200 evallong200_t5_convlstm \
    scripts/eval/run_eval.sh "RUN=$R_LONG200 $EVAL6_TEMP K_PREVS=5" "$RES_EVAL_S4_T5" \
    "the only reading that settles long200: object F1 on the eval6 protocol, against temporal_k5_pre2023_convlstm_ring3 at 0.780"

# ---- argument parsing -------------------------------------------------------
WANT=all; SUBMIT=no; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    eval6ref|long500|ctx50|evallong200|all)   WANT="$1" ;;
    # Retired by name rather than left to select zero jobs, so the mistake is
    # visible instead of looking like an empty batch. See docs/EXPERIMENTS.md.
    long200|train|tattn|control|clean22|valpos|attnfix|attnpos|pre23|training|\
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
