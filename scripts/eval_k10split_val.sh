#!/bin/bash
#BSUB -J eval_k10split_val
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/eval_k10split_val_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/eval_k10split_val_%J.err
#BSUB -q long-gpu
#BSUB -gpu num=1:j_exclusive=yes
#BSUB -R rusage[mem=64GB]
#BSUB -W 24:00
#
# Full-interferogram evaluation of the ConvLSTM run
#   convlstm_10prev_h256_b64_lr5e6_40e_k10split_2026-08-03_17h19
# on the 19 validation interferograms of its partition (no test split exists).
#
# Submit:  bsub < scripts/eval_k10split_val.sh   (from /home/labs/rudich/pinkas/sinkholes)
# Or run directly inside a GPU session:  bash scripts/eval_k10split_val.sh
#
# Scheduler stdout/stderr land in logs/eval_k10split_val_<jobid>.{out,err}.

set -o pipefail

REPO=/home/labs/rudich/pinkas/sinkholes
VENV=/home/labs/rudich/pinkas/python_envs/sinkholes
PATCHES=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches

RUN=outputs/convlstm_10prev_h256_b64_lr5e6_40e_k10split_2026-08-03_17h19
MODEL=$RUN/checkpoints/best.pt                      # epoch 23, val/dice 0.6694
PARTITION=assets/partition_20_05_16h53_k10_compatible.json
OUTROOT=outputs/predictions/convlstm_10prev_k10split
JOB=val_k10

cd "$REPO"
source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# (3b) scenes: reconstruct, LiDAR-gate, threshold, polygonise ------------------
python -m sinkholes eval-scenes \
  --model "$MODEL" \
  --input_patch_dir "$PATCHES" \
  --intf_source preset --valset_from_partition "$PARTITION" \
  --patch_size 200 100 --data_stride 2 --days_diff 11 \
  --k_prevs 10 --recon_th 0.25 \
  --convlstm_unet \
  --add_lidar_mask \
  --save_confidence --merge_polygs \
  --job_name "$JOB" --output_dir "$OUTROOT"

# eval-scenes stamps the job dir with its start time; pick the newest one.
EVAL_DIR=$(ls -dt "$OUTROOT"/best/${JOB}_* | head -1)
echo "eval-scenes wrote $EVAL_DIR"

# object-level metrics + per-interferogram overview figures --------------------
python -m sinkholes eval-outputs \
  --path "$EVAL_DIR" \
  --th 0.7 --buffer 5 \
  --save_figures

echo "done: $EVAL_DIR"
