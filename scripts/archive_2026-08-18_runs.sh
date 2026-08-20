#!/usr/bin/env bash
# ============================================================================
#  SUPERSEDED 2026-08-20 -- DO NOT RUN. Kept as the record of a one-off that
#  already happened. It expects 14 run dirs at outputs/clean_*_2026-08-18_* and
#  there are none, so it dies SILENTLY with status 1 -- `ls` fails, pipefail
#  propagates it, and set -e kills the script before its own error message
#  runs. The sed blocks below would rewrite paths that have since changed
#  again, so do not lift them out either.
#
#  The general version is scripts/tidy_outputs.sh: same job, any batch, dry-run
#  by default, and it derives the _valpos/_valneg suffix from reporter.log.
#
#  Move the fourteen 2026-08-18 (clean22) training runs into outputs/2026-08-18/,
#  the same retention layout as outputs/2026-08-09 / 2026-08-10 / 2026-08-11.
#
#  DO NOT RUN THIS WHILE ANY EVAL IS QUEUED OR RUNNING. An eval carries its
#  RUN= path in the bsub environment from submit time, and run_eval.sh:227-228
#  preflights it:  [ -f "$RUN/checkpoints/best.pt" ] || exit 1. A PEND job whose
#  run directory moved underneath it dies on that line. Check first:
#
#      bjobs -w | grep -E 'eval_clean|clean_'
#
#  and only run this when it prints nothing.
#
#  WHAT THE MOVE DOES NOT BREAK. run_eval.sh:263 sets
#      OUTROOT="outputs/predictions/$(basename "$RUN")"
#  from the BASENAME, so predictions keep landing in the same place and every
#  directory already under outputs/predictions/ stays valid. The run's own
#  group check (run_eval.sh:256) is basename-derived too. So the only thing
#  that has to change is where RUN= points, which is the sed block below.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=outputs/2026-08-18

live=$(ls -d outputs/clean_*_2026-08-18_* 2>/dev/null | wc -l | tr -d ' ')
[ "$live" = 14 ] || { echo "expected 14 run dirs at outputs/clean_*_2026-08-18_*, found $live" >&2; exit 1; }

mkdir -p "$DEST"
for d in outputs/clean_*_2026-08-18_*; do
  n=$(basename "$d")
  [ -e "$DEST/$n" ] && { echo "$DEST/$n already exists -- refusing to overwrite" >&2; exit 1; }
  # A failing mv on the SMB mount means something still holds the directory
  # open (a Finder window is enough). Stop rather than half-move the batch.
  mv "$d" "$DEST/$n" || { echo "mv failed on $n -- close anything holding it open, then rerun" >&2; exit 1; }
  echo "moved $n"
done

# Repoint the eight CLEAN_* variables the eval5 table reads. They are the only
# references to these paths in the repo -- verified with
#     grep -rn "clean_.*2026-08-18" scripts/ docs/
sed -i.bak -E 's#^(CLEAN_[A-Z0-9_]+=)outputs/(clean_[a-z0-9_]+_2026-08-18_[a-z0-9_]+)$#\1outputs/2026-08-18/\2#' scripts/submit_all.sh
echo
echo "scripts/submit_all.sh repointed (backup at scripts/submit_all.sh.bak):"
grep -n '^CLEAN_[A-Z0-9_]*=' scripts/submit_all.sh
echo
# docs/MODEL_RUNS.md:337 names the batch as the glob `outputs/clean_*_2026-08-18_*/`
# in prose. Nothing reads it, but leaving it pointing at an empty glob is a
# trap for the next person, so it is rewritten to the archived location.
sed -i.bak 's#`outputs/clean_\*_2026-08-18_\*/`#`outputs/2026-08-18/clean_*/`#' docs/MODEL_RUNS.md
echo "docs/MODEL_RUNS.md:337 repointed."
echo
echo "Now dry-run the eval table before submitting anything:"
echo "    bash scripts/submit_all.sh eval5"
