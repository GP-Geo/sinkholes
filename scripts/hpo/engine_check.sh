#!/usr/bin/env bash
#BSUB -J hpo_engine_check
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/hpo_engine_check_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/hpo_engine_check_%J.err
#BSUB -q long-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=24G
#BSUB -R rusage[mem=96GB]
#BSUB -W 6:00
#
# Engine check before the HPO study: legacy eval-scenes + eval-outputs against
# eval-snapshots, on the two gappiest validation scenes (2022-23) of the k10
# test2026 partition, whole AOI, with the existing fill k10 checkpoint. Plain geometry
# (the ctx50 stride-4 tree is not complete yet). Selects nothing and touches no
# test scene. Submit:  bsub < scripts/hpo/engine_check.sh   (from the repo root)
#
# Memory: the legacy engine holds all eleven 5.6 GB grids of a scene (legacy
# plain k10 evals measured 75 GB); eval-snapshots far less.
set -euo pipefail
REPO=/home/labs/rudich/pinkas/sinkholes
cd "$REPO"
source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# The env holds an editable install of this checkout; refuse any other copy of the code.
python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import sinkholes.inference.snapshot_eval as e
root = os.path.realpath(os.getcwd())
sys.exit(f"snapshot_eval from {e.__file__}, not {root}") if not os.path.realpath(e.__file__).startswith(root + os.sep) else print("code from", root)
PY
python scripts/hpo/engine_check.py \
  --model /home/labs/rudich/pinkas/sinkholes/outputs/2026-10-06/temporal_k10_th350_fill_tattn_posw8_30e/checkpoints/best.pt \
  --n_scenes 2 --tile_batch 32 \
  --out /home/labs/rudich/pinkas/sinkholes/studies/engine_check_$(date +%Y%m%d_%H%M)
