#!/usr/bin/env bash
# ============================================================================
#  Re-score saved evaluations against the corrected test labels  (WEXAC)
#
#  Submit (repo root):  bsub < scripts/relabel/eval_v2.sh
#  Other label:         LABEL=v2_<tag> bsub < scripts/relabel/eval_v2.sh
#
#  One array task per eval directory in DIRS below, each running
#  `sinkholes relabel eval run --only_relabelled --exclude_flagged --objects`
#  (docs/RELABELING.md §6): bit-identity check of the saved GT, then the same
#  _pred.npy scored against the original labels, the corrected labels, and the
#  corrected labels with the qc-flagged polygons not counted. No inference, so
#  no GPU. On the Mac the same run is SMB-bound (every scene's 120 MB canvases
#  cross the network several times) and takes hours per 20-scene directory.
#
#  Results: <workspace>/eval/<LABEL>/<model>/best/<job>/ -- the same tree the
#  Mac runs write, so local and cluster tasks can be mixed. Do not run the same
#  directory in both places at once: they write the same files.
# ============================================================================
#BSUB -J relabel_eval[1-3]
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/relabel_eval_%J_%I.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/relabel_eval_%J_%I.err
#BSUB -q short
#BSUB -R rusage[mem=16GB]
#BSUB -W 6:00

set -euo pipefail

# ------------------------------ CONFIG --------------------------------------
LABEL="${LABEL:-v2_20_20261013}"
P=outputs/predictions
# Keep the #BSUB -J array range equal to the number of entries.
DIRS=(
  $P/temporal_k10_th350_fill_tattn_posw8_30e/best/scenes_temporal_k10_fill_rth_10_07_09h53
  $P/temporal_k10_th350_tattn_posw8_30e/best/scenes_temporal_k10_fill_rth_10_06_13h28
  $P/temporal_k10_th350_single_posw8_30e/best/scenes_temporal_k10_th350_rth_09_16_21h35
)
# ----------------------------------------------------------------------------

REPO=/home/labs/rudich/pinkas/sinkholes
DATA=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data
WS=$DATA/relabel_temporal_test_v2
cd "$REPO" || { echo "cannot cd to $REPO" >&2; exit 1; }

i="${LSB_JOBINDEX:?run as an LSF array task (bsub < $0)}"
[ "$i" -le "${#DIRS[@]}" ] || { echo "array index $i > ${#DIRS[@]} directories" >&2; exit 1; }
DIR="${DIRS[$((i - 1))]}"
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }
GT="$WS/exports/gt_test_corrected_${LABEL}.gpkg"
[ -f "$GT" ] || { echo "no corrected GT $GT -- run 'sinkholes relabel export --tag $LABEL'" >&2; exit 1; }
echo "task $i: $DIR  label $LABEL  on $(hostname)"

source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes

python -m sinkholes relabel eval run \
  --workspace "$WS" --data_dir "$DATA" --label "$LABEL" \
  --eval_dirs "$DIR" --only_relabelled --exclude_flagged --objects

echo "done: $WS/eval/$LABEL"
