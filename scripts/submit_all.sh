#!/usr/bin/env bash
# ============================================================================
#  submit_all.sh -- the shield in front of bsub.
#
#      scripts/submit_all.sh                     # list every live job (DRY RUN)
#      scripts/submit_all.sh <kind>              # list one kind
#      scripts/submit_all.sh <kind> --submit     # send it
#      scripts/submit_all.sh --only <substr>     # narrow by name, AND-ed with the kind
#
#  DRY RUN IS THE DEFAULT. Nothing reaches LSF without --submit.
#
#  WHAT THIS FILE IS. A table of jobs that have NEVER RUN, plus the preflight
#  that stops a doomed one before it reaches a compute node. A batch that has
#  run is REMOVED from here, so `--submit` can never silently repeat finished
#  GPU-hours; what it settled goes into docs/RESULTS.md. As of 2026-10-15 every
#  batch has run, so the table is empty. The HPO study is not here either:
#  `sinkholes hpo` writes and submits its own jobs (docs/HPO.md).
#
#  Every retired batch's exact rows and reasoning:
#      git show notebook-2026-10-15:scripts/submit_all.sh   (2026-09 -> 10)
#      git show d708fba:scripts/submit_all.sh               (up to 2026-09-03)
#
#  WHAT THE PREFLIGHT CATCHES, on the login node rather than 20 minutes into a
#  GPU allocation:
#    * a template that does not exist;
#    * RUN= whose checkpoints/best.pt is gone (deleted or moved by a tidy);
#    * PARTITION= that does not exist;
#    * DIR= for a re-score whose *_pred.npy were pruned -- re-scoring reads
#      exactly those, and only a full run_eval.sh can rebuild them.
#
#  RESOURCES are (queue, host GB, GPU GB, walltime). They are measured, not
#  guessed; the comment on each line says what it was measured from.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f scripts/train/train_tattn.sh ] || [ ! -d assets ]; then
  echo "run this from the repo (cd to it); assets/ and scripts/train/ must be here" >&2
  exit 1
fi

KINDS=(); NAMES=(); TEMPLATES=(); OVERRIDES=(); RESOURCES=(); WHYS=()
job() { KINDS+=("$1"); NAMES+=("$2"); TEMPLATES+=("$3"); OVERRIDES+=("$4"); RESOURCES+=("$5"); WHYS+=("$6"); }

# ---- the current leader, as building blocks for the next row ----------------
#
# temporal_k10_th350_fill_tattn_posw8_30e (outputs/2026-10-06/), object F1
# 0.8114 on the 24 generation-5 test scenes -- plain temporal attention at k10
# on filled chains (docs/RESULTS.md). Its exact rows, as they ran:
#
#   job <kind> <name> scripts/train/train_tattn.sh \
#       "$FILL_K10_BASE $LEAD_NEG $LEAD_OPT $K10_RUN30 BATCH=32 ACCUM=4 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" \
#       "$RES_FILL_K10_TATTN" "<why>"
#   job <kind> <name> scripts/eval/run_eval.sh \
#       "RUN=<run dir> $EVAL_FILL_K10 K_PREVS=10 ARCH=tattn" "$RES_EVAL_S4_T" "<why>"
#
# The k5 twin (temporal_k5_th350_fill_tattn_posw8_40e, 0.7934) is the same with
# FILL_K5_BASE, K5_RUN40, BATCH=64 ACCUM=2 and RES_K5_TATTN.
FILL_K5_BASE="PARTITION=assets/partition_temporal_k5_clean_fill_th350x200.json K_PREVS=5 POS_W=8"
FILL_K10_BASE="PARTITION=assets/partition_temporal_k10_clean_fill_th350x200.json K_PREVS=10 POS_W=8"
LEAD_NEG="RING_NEGS=yes NEG_RING_INNER=1 NEG_RING_OUTER=10 NEG_PER_POS=1.0"
LEAD_OPT="OPTIM=adamw LR=3e-4 WEIGHT_DECAY=1e-2 SEG_LOSS=dice"
K10_RUN30="SCHEDULE=plateau LR_PATIENCE=8 EPOCHS=30 PATIENCE=0"
K5_RUN40="SCHEDULE=plateau LR_PATIENCE=5 EPOCHS=40 PATIENCE=15"
EVAL_FILL_K5="GEN=fill GROUP=temporal_k5 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_k5_fill_rth"
EVAL_FILL_K10="GEN=fill GROUP=temporal_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_k10_fill_rth"
EVAL_T2026_K10="GEN=t2026 GROUP=temporal_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_k10_t2026_rth"

# ---- resources --------------------------------------------------------------
RES_FILL_K10_TATTN="long-gpu   208   24  16:00"   # gen-4 twin 82.6 GB / 5.7 h (LSF 316673); x1.51 samples -> ~125 GB, ~8.6 h
RES_K5_TATTN="long-gpu      96   24  10:00"       # gen-4 twin 55.0 GB / 4.9 h (LSF 333265); x1.24 samples -> ~68 GB, ~6 h
RES_EVAL_S4_T="long-gpu    160   36  16:00"       # plain temporal evals: k5 48 GB, k10 75 GB (LSF 330515)
RES_EVAL_S4_CTRL="long-gpu  48   24   6:00"       # single-frame evals: 27.8 GiB measured

# ---- jobs that have never run -----------------------------------------------
# (none)

# ---- argument parsing -------------------------------------------------------
WANT=all; SUBMIT=no; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    all)   WANT="$1" ;;
    # Retired by name rather than left to select zero jobs, so the mistake is
    # visible instead of looking like an empty batch.
    eval6ref|ampfix|lrscan|evalampfix|evalctx|reg|dropscan|lossscan|momscan|adamscan|combo|\
    th350|evalth350|posw8|evalposw8|k10|k10plain|evalk10|evalk10plain|k10plain45|k5plain|\
    evalk5plain|fill|evalfill|t2026|dd300|long200|long500|ctx50|evallong200|longreg|longgrid|\
    evallonggrid|train|tattn|control|clean22|valpos|valneg|attnfix|attnpos|pre23|training|\
    blend|rescore|eval|eval2|eval3|eval4|eval5|eval6|eval7|probe|posonly)
      echo "'$1' has run or was superseded -- docs/RESULTS.md says what it settled." >&2
      echo "Its rows: git show notebook-2026-10-15:scripts/submit_all.sh (older: d708fba)." >&2
      exit 1 ;;
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
    *PARTITION=*) pf=$(sed 's/.*PARTITION=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            [ -f "$pf" ] || { echo "missing partition $pf" >&2; fail=1; } ;;
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
