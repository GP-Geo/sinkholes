#!/usr/bin/env bash
# ============================================================================
#  Move finished training runs out of the top level of outputs/ and into the
#  dated folder they belong to, under the name outputs/README.md specifies.
#
#      outputs/clean_geo_k10_tattn_fixed_ring3_2026-08-19_16h20_lsf_768296
#   -> outputs/2026-08-19/geo_k10_tattn_fixed_ring3_valneg
#
#  THE _valneg SUFFIX IS READ OUT OF THE RUN, NOT GUESSED. reporter.log records
#  the validation protocol as either "Val scored on positives only" or
#  "Val scored on N positive + N negative", and a run validated against
#  negatives selected best.pt on a curve where an empty prediction on an empty
#  mask scores dice 1.0. That is not comparable to a positives-only curve and
#  must not be read as if it were, so it goes in the name. A run whose name
#  already ends _valpos or _valneg is left alone.
#
#  A run is BORN with the `_<date>_<time>_lsf_<jobid>` suffix and cannot be
#  born without it: --resume auto finds a requeued execution by globbing
#  outputs/*_lsf_$LSB_JOBID (sinkholes/training/resume.py:312). The suffix is
#  the resume key, so it can only come off once the run will never be requeued.
#  That is what this script is -- the tidy step after a batch settles, not
#  something to wire into training.
#
#  DRY RUN IS THE DEFAULT. Nothing moves without --apply.
#
#  ---------------------------------------------------------------------------
#  DO NOT RUN THIS WHILE ANY EVAL IS QUEUED OR RUNNING against these runs.
#  An eval carries its RUN= path from submit time and preflights
#  `[ -f "$RUN/checkpoints/best.pt" ]`, so a PEND job whose run directory moved
#  underneath it dies on that line. On the cluster, check first:
#
#      bjobs -w | grep -E 'eval_|clean_'
#
#  and only run this when it prints nothing.
#
#  WHAT THE MOVE DOES NOT BREAK. run_eval.sh:275 derives
#  OUTROOT="outputs/predictions/$(basename "$RUN")" from the BASENAME, so
#  evaluations submitted AFTER a tidy land in outputs/predictions/<clean name>/
#  -- which is the convention, and is why tidying before evaluating is worth
#  doing. Directories already under outputs/predictions/ keep their old names
#  and stay valid; this script does not touch them.
#
#  WHAT IT REFUSES TO MOVE. A run with no "Training complete"/"Training
#  interrupted" line in logs/reporter.log is still live, or died without
#  finishing, and keeps its resume key so it can be requeued. Use --force to
#  move one anyway, having decided it will never be resumed -- after that,
#  resuming needs the explicit form:
#      --resume outputs/<date>/<run>/checkpoints/resume.pt
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

APPLY=0
FORCE=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,41p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a (expected --apply and/or --force)" >&2; exit 2 ;;
  esac
done

shopt -s nullglob
runs=(outputs/*_lsf_[0-9]*)
[ ${#runs[@]} -gt 0 ] || { echo "nothing to tidy: no outputs/*_lsf_<id> at the top level"; exit 0; }

moved=0 skipped=0 failed=0
for d in "${runs[@]}"; do
  [ -d "$d" ] || continue
  n=$(basename "$d")

  # <name>_<YYYY-MM-DD>_<HHhMM>_lsf_<id>  ->  date and clean name
  if [[ ! "$n" =~ ^(.+)_([0-9]{4}-[0-9]{2}-[0-9]{2})_[0-9]{2}h[0-9]{2}_lsf_[0-9]+$ ]]; then
    echo "SKIP  $n -- name does not carry <date>_<time>_lsf_<id>"; skipped=$((skipped+1)); continue
  fi
  clean="${BASH_REMATCH[1]}"; date="${BASH_REMATCH[2]}"
  # The clean_ prefix marked the clean-benchmark era; the dated folder and the
  # noisy-data archive already separate the eras, so it is redundant in a name.
  clean="${clean#clean_}"

  log="$d/logs/reporter.log"
  if [ "$FORCE" = 0 ]; then
    if [ ! -f "$log" ]; then
      echo "SKIP  $n -- no logs/reporter.log, cannot tell whether it finished"; skipped=$((skipped+1)); continue
    fi
    if ! grep -qE "Training (complete|interrupted)" "$log"; then
      # `|| true`: grep exits 1 on no match and pipefail would abort the run.
      last=$(grep -oE '^\s+[0-9]+/[0-9]+' "$log" 2>/dev/null | tail -1 | tr -d ' ' || true)
      echo "SKIP  $n -- no completion line (reached ${last:-no epoch}); still resumable, use --force to override"
      skipped=$((skipped+1)); continue
    fi
  fi

  # Protocol marker, straight out of the log the run wrote about itself.
  case "$clean" in
    *_valpos|*_valneg) ;;                       # already carries one
    *)
      # `|| true` for the same reason: a run predating the protocol line has no
      # match, and that is a legitimate case, not a failure.
      proto=$([ -f "$log" ] && grep -h "Val scored on" "$log" 2>/dev/null | tail -1 | sed 's/.*Val scored on *//' || true)
      case "$proto" in
        "positives only")      ;;               # the default; not in the name
        *positive*negative*)   clean="${clean}_valneg" ;;
        "")  echo "NOTE  $n -- no 'Val scored on' line; leaving the protocol out of the name" ;;
        *)   echo "NOTE  $n -- unrecognised protocol '$proto'; leaving it out of the name" ;;
      esac ;;
  esac

  dest="outputs/$date/$clean"
  if [ -e "$dest" ]; then
    echo "SKIP  $n -- $dest already exists, refusing to overwrite"; skipped=$((skipped+1)); continue
  fi

  if [ "$APPLY" = 0 ]; then
    echo "would move  $n -> $dest"; moved=$((moved+1)); continue
  fi

  mkdir -p "outputs/$date"
  # A failing mv on the SMB mount means something still holds the directory
  # open -- a Finder window or a log open in Console.app is enough. Report it
  # and carry on rather than half-moving the batch.
  if mv "$d" "$dest" 2>/dev/null; then
    echo "moved $n -> $dest"; moved=$((moved+1))
  else
    echo "FAIL  $n -- something holds it open. Find it with:  lsof +D $d" >&2
    failed=$((failed+1))
  fi
done

echo
if [ "$APPLY" = 0 ]; then
  echo "DRY RUN: $moved would move, $skipped skipped. Re-run with --apply."
else
  echo "$moved moved, $skipped skipped, $failed failed."
  if [ "$moved" -gt 0 ]; then echo "Record what changed in outputs/README.md."; fi
fi

# Non-zero only when a move was attempted and the filesystem refused it. A
# skip is a decision the script made on purpose and is not an error.
[ "$failed" -eq 0 ]
