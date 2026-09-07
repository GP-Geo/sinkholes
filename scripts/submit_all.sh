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

# ---- ampfix: re-baselining on the corrected tree ----------------------------
#
#     submit_all.sh ampfix --submit          # 3 jobs
#     submit_all.sh ampfix --only=30e --submit
#
# WHY THESE REPLACE long500 AND THE FIRST ctx50 PAIR. Two things changed under
# this batch, and both change what a number means:
#
#   1. AMP gradient clipping was corrected (ab23b07). clip_grad_norm_ used to
#      run BEFORE grad_scaler unscaled, so it saw gradients multiplied by the
#      scale factor and renormalised nearly every step instead of thresholding
#      at 1.0. Every run in the project up to 25c4dc5 trained that way.
#      check_config_compatible now REFUSES to resume such a checkpoint, so the
#      earlier long500/ctx50 executions could not have been continued here.
#   2. Single-frame patch normalisation was corrected to frame-v2. ConvLSTM is
#      temporal and is NOT affected by that one -- which is precisely why these
#      arms isolate the clipping change cleanly.
#
# NOTHING HERE IS COMPARABLE WITH 0.780, with long200, or with anything in
# RESULTS.md. This batch re-establishes the temporal baseline on corrected
# numerics; the old figures stay valid for the tree that produced them.
#
# THE 500-EPOCH ARM IS DROPPED. evallong200 (LSF 580743) has now scored long200
# on the eval6 protocol: object F1 0.7529 against the baseline's 0.7797 on the
# convention RESULTS.md 9 quotes (ith0.7_b5, threshold 0.25, mean) -- -0.0268,
# more than four times the +/-0.006 noise floor, in the WRONG direction. The
# premise for spending 500 epochs was that long200's rising dice curve had more
# to give; the object-level score says the extra epochs cost accuracy rather
# than buying it. 200 with LR_PATIENCE=20 is long200's own schedule and is the
# right length to re-measure first.
#
# grad/norm, grad/clip% and amp/scale are now written to results.csv every
# epoch (6bc1181). On the corrected tree grad/clip% should be a real, mostly
# SMALL number; if it comes back near 100% the clip is still normalising and
# these runs need reading again before anything is concluded from them.
AMPFIX_T5="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
AMPFIX_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
AMPFIX_CTX="CTX_MY=50 CTX_MX=50 BATCH=64 ACCUM=2"

# Plain 200x100, long200's config exactly: LR 1e-5, LR_PATIENCE 20, early
# stopping off. The ONLY difference from long200 is the corrected clipping,
# which is what makes it the control for the other two.
RES_AMPFIX_PLAIN="long-gpu    90   36  18:00"   # 3m31s/epoch measured -> ~11h45m, one allocation

# 300x200 context: 3x the pixels, so ~10m33s/epoch off the same measurement.
# Host memory, not VRAM, is the risk -- the lazy loader PLAN_LARGE_CONTEXT.md 3
# calls for was never built, so samples are still fully materialised.
RES_AMPFIX_CTX_LONG="long-gpu   180   56  18:00"   # 200 ep = ~35h10m, ~2 requeues via --resume auto
RES_AMPFIX_CTX_SHORT="long-gpu   180   56  18:00"  # 30 ep = ~5h20m, one allocation

job ampfix ampfix_t5_convlstm_ring3_200e \
    scripts/train/train_convlstm.sh "$AMPFIX_T5 $AMPFIX_NEG LR=1e-5 LR_PATIENCE=20 EPOCHS=200 PATIENCE=0" "$RES_AMPFIX_PLAIN" \
    "the control: long200's exact config on corrected clipping -- the number every other arm here is read against"
job ampfix ampfix_ctx50_t5_convlstm_ring3_200e \
    scripts/train/train_convlstm.sh "$AMPFIX_T5 $AMPFIX_NEG $AMPFIX_CTX LR=1e-5 LR_PATIENCE=20 EPOCHS=200 PATIENCE=0" "$RES_AMPFIX_CTX_LONG" \
    "the matched context arm: identical to the control except 300x200 of context and the batch split that pays for it"
job ampfix ampfix_ctx50_t5_convlstm_ring3_30e \
    scripts/train/train_convlstm.sh "$AMPFIX_T5 $AMPFIX_NEG $AMPFIX_CTX LR=1e-5 LR_PATIENCE=5 EPOCHS=30 PATIENCE=0" "$RES_AMPFIX_CTX_SHORT" \
    "the fast read: 30 epochs at LR_PATIENCE=5, enough to see the corrected clip% and whether context moves the early curve at all"

# THE SINGLE-FRAME FLOOR, AND THE ONE ARM THAT CHANGES ON BOTH COUNTS.
# Every ConvLSTM arm above is affected by the clipping fix ALONE, because a
# temporal stack always had real frames to normalise over. This one is affected
# by the frame-v2 normalisation fix as well: a single 200x100 patch has no
# channel axis, so normalise_channels looped over ROWS and decided the
# "radians or already 0-1?" question 200 times per patch, once per row of 100
# pixels -- which means one patch could be part-converted and part-not.
#
# That makes the recorded single-vs-recurrent gap (RESULTS.md 9: +0.047
# temporal, +0.038 geo) UNSAFE TO QUOTE: some of it may be the U-Net running on
# mangled input rather than recurrence being worth anything. This arm is what
# re-measures that gap honestly, so it is worth more than either context arm.
#
# Config is temporal_k5_pre2023_single_ring3's (which scored 0.733), moved onto
# the control's schedule: LR_PATIENCE=20 and 200 epochs with early stopping off,
# matching ampfix_t5_convlstm_ring3_200e exactly so the two are readable against
# each other. The original ran at LR_PATIENCE=5 (the code default; the template
# never exposed the knob until now) and early-stopped at 87/100.
#
# 2m13s/epoch measured from that run (87 epochs in 3h13m), so 200 is ~7h25m in
# one allocation. Host memory is its 13G peak with headroom.
RES_AMPFIX_SINGLE="long-gpu    24   24  12:00"   # 2m13s/epoch measured -> ~7h25m

job ampfix ampfix_t5_single_ring3_200e \
    scripts/train/train_control.sh "ARCH=single $AMPFIX_T5 $AMPFIX_NEG LR=1e-5 LR_PATIENCE=20 EPOCHS=200 PATIENCE=0" "$RES_AMPFIX_SINGLE" \
    "the single-frame floor on corrected data: the only arm fixed on BOTH counts, and the one that says whether recurrence really buys what RESULTS.md 9 claims"

# ---- lrscan: find a step size the corrected gradients can live with --------
#
#     submit_all.sh lrscan --submit          # 5 jobs, ~30 min each
#
# WHY THIS EXISTS. ampfix (LSF 612826/612827/612828) died identically on both
# the plain and the context arm: epoch 1 healthy (dice 0.469, clip 3.1%, AMP
# scale steady at 65536), then nan on the FIRST step of epoch 2. best.pt from
# epoch 1 is finite but carries a BatchNorm running_var of 1.576e+08 -- the
# activations were already exploding, and the next forward pass overflowed fp16.
#
# It is NOT gradient explosion: median true gradient norm was 0.124 and the clip
# fired on 3.1% of steps. It is STEP SIZE. RMSprop runs at momentum=0.999, which
# amplifies every update by 1/(1-m) = 1000x at steady state. That was survivable
# only while the clipping bug crushed every gradient to a fixed ~1.5e-5 norm.
# With unscaled-v2 clipping the real gradients arrive and the same 1000x
# amplification is far too hot. lr and momentum were co-adapted to the bug.
#
# TWO AXES, because either fixes the step size and they are not equivalent:
# dropping lr scales every update uniformly; dropping momentum shortens the
# window the updates are accumulated over, which also removes the lag that makes
# a 1000x buffer dangerous near a curved minimum. 0.9 is a conventional RMSprop
# momentum and 0 is PyTorch's own default.
#
# READ IT FROM results.csv: an arm is alive if amp/scale stays 65536, the
# non-finite warning never appears, and val/dice climbs. An arm that dies does
# so by epoch 2, so 8 epochs is more than enough to tell.
LRSCAN_BASE="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
LRSCAN_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
LRSCAN_LEN="LR_PATIENCE=20 EPOCHS=8 PATIENCE=0"
RES_LRSCAN="long-gpu    90   36   2:00"   # 8 x 3m31s = ~28m

# THE NEGATIVE CONTROL, and it is expected to FAIL. lr 1e-5 with momentum 0.999
# is exactly what ampfix ran (LSF 612826) before it went nan, so this arm is the
# origin of the grid: every other arm changes one axis away from it.
#
# It is worth a GPU slot for a reason beyond completeness. --momentum was added
# in 3a64c5d with a default of 0.999 to reproduce the value that had been
# hardcoded at train.py:702. This arm is what PROVES that default is faithful.
# If it trains happily, the divergence was never about momentum and the whole
# reading of ampfix is wrong -- so a pass here is the informative outcome, not
# the failure.
#
# Expect: epoch 1 clean (ampfix reached dice 0.469 with a BatchNorm running_var
# already at 1.576e+08), epoch 2 flooded with non-finite gradients and amp/scale
# collapsed to 0, loss nan from epoch 3. It dies well inside the 8.
job lrscan lrscan_lr1e5_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-5 MOMENTUM=0.999" "$RES_LRSCAN" \
    "the negative control: the exact ampfix setting that diverged, re-run to confirm it still does and that --momentum 0.999 reproduces the old hardcoded value"

job lrscan lrscan_lr1e6_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-6 MOMENTUM=0.999" "$RES_LRSCAN" \
    "lr axis: 10x down from the setting that blew up, momentum left at its historical 0.999"
job lrscan lrscan_lr1e7_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-7 MOMENTUM=0.999" "$RES_LRSCAN" \
    "lr axis: 100x down"
job lrscan lrscan_lr1e8_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-8 MOMENTUM=0.999" "$RES_LRSCAN" \
    "lr axis: 1000x down -- roughly the factor momentum contributes, so the floor of this axis"
job lrscan lrscan_lr1e5_m090 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-5 MOMENTUM=0.9" "$RES_LRSCAN" \
    "momentum axis: long200's own lr, momentum 0.999 -> 0.9, so the amplification drops 1000x -> 10x"
job lrscan lrscan_lr1e5_m000 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-5 MOMENTUM=0" "$RES_LRSCAN" \
    "momentum axis: long200's own lr with no momentum at all -- PyTorch's RMSprop default, and the cleanest reading of what the gradients alone do"

# ---- argument parsing -------------------------------------------------------
WANT=all; SUBMIT=no; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    eval6ref|ampfix|lrscan|all)   WANT="$1" ;;
    # Retired by name rather than left to select zero jobs, so the mistake is
    # visible instead of looking like an empty batch. See docs/EXPERIMENTS.md.
    long200|long500|ctx50|evallong200|\
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
