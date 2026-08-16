#!/usr/bin/env bash
# ============================================================================
#  Re-score an existing eval-scenes directory  (stage 2 only)
#
#  Submit:  DIR=<eval dir> bsub < scripts/eval/rescore.sh
#
#  run_eval.sh runs two stages: eval-scenes (GPU inference over every patch of
#  every scene, ~3 h) then eval-outputs (re-threshold the saved confidence and
#  compute object-level P/R, minutes, CPU-only). When only the SCORING needs to
#  change -- a different threshold sweep, a different overlap threshold or
#  buffer -- stage 1 does not have to run again: eval-scenes already wrote
#  <intf>_pred.npy, and eval-outputs reads exactly that.
#
#  Why this exists: the evals of 2026-08-09 12h38 and earlier were scored at
#  0.125/0.25/0.5 only. THRESHOLDS was later extended to add 0.7 and 0.9
#  (sinkholes/inference/outputs.py:26), and the geo models peak above 0.5 --
#  geo_k10 was still at P=0.31 and climbing steeply at the last point it has,
#  so its curve is cut off before its own operating point and its best F1 is
#  unknown. Those JSONs are not comparable to the 5-point ones until re-scored.
#
#  Cheap: no GPU, and the ~57 GB peak of a full eval belongs to stage 1.
# ============================================================================
#BSUB -J rescore
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/%J.err
# short-gpu, whose hard walltime cap is under 8:00 (see run_eval.sh) -- 2:00
# fits comfortably. This stage needs no GPU at all; if the cluster has a
# CPU-only queue, point -q at it instead and stop burning a GPU slot.
#BSUB -q short-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=8G
#BSUB -R rusage[mem=16GB]
#BSUB -W 2:00

set -euo pipefail

# ------------------------------ CONFIG --------------------------------------
DIR="${DIR:?set DIR to an eval-scenes output directory (the one holding <intf>_pred.npy)}"
OL_TH="${OL_TH:-0.7}"           # object-level overlap threshold
BUFFER="${BUFFER:-5}"           # object-matching buffer, pixels
THRESHOLDS="${THRESHOLDS:-0.125 0.25 0.5 0.7 0.9}"   # confidence operating points
FIGURES="${FIGURES:-no}"        # yes = also redraw the per-scene overview PNGs
# ----------------------------------------------------------------------------

REPO=/home/labs/rudich/pinkas/sinkholes
cd "$REPO" || { echo "cannot cd to $REPO" >&2; exit 1; }

# Preflight: the confidence maps are what stage 2 reads. If eval-scenes output
# was pruned, only a full run_eval.sh can rebuild it -- say so here rather than
# failing inside the python.
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }
n_pred=$(find "$DIR" -maxdepth 1 -name '*_pred.npy' | wc -l)
[ "$n_pred" -gt 0 ] || {
  echo "no *_pred.npy in $DIR -- nothing to re-score." >&2
  echo "The confidence maps were pruned; rebuild them with scripts/eval/run_eval.sh." >&2
  exit 1
}
echo "re-scoring $n_pred scene(s) in $DIR"
echo "thresholds: $THRESHOLDS   overlap: $OL_TH   buffer: $BUFFER"

source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes

FIG_FLAG=()
[ "$FIGURES" = yes ] && FIG_FLAG=(--save_figures)

# eval-outputs timestamps its JSON (olm_results_<MMDDHHMM>.json), so the old
# 3-point file is left in place next to the new one rather than overwritten.
# Read the newest; keep the old only as a record of what was reported before.
python -m sinkholes eval-outputs \
  --path "$DIR" \
  --thresholds $THRESHOLDS \
  --th "$OL_TH" --buffer "$BUFFER" \
  ${FIG_FLAG[@]+"${FIG_FLAG[@]}"}

echo "done. newest metrics:"
ls -t "$DIR"/olm_results_*.json | head -1
