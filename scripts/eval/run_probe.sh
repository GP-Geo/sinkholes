#!/usr/bin/env bash
# ============================================================================
#  Attention-selectivity probe
#
#  Submit:  bsub < scripts/eval/run_probe.sh
#
#  Answers ONE question about a trained tattn_unet checkpoint: when the model
#  is handed a long, gappy history, does its temporal attention SELECT frames,
#  or does it spread weight uniformly and degenerate into an average?
#
#  WHY THIS IS NOT OPTIONAL. The attention block collapsed to uniform once
#  already, silently: the clean22 tattn arms trained, converged and produced
#  respectable dice while doing temporal AVERAGING, not selection. `contrast`
#  and `qk_norm` are the fix, and the ONLY way to confirm the fix took on a
#  given run is to measure the trained weights on real data. A freshly built
#  model passes every content-sensitivity test and still dies in training, so
#  an architecture flag in the log is not evidence -- see the header of
#  submit_all.sh's `attnfix` block.
#
#  WHAT IT WRITES, into $OUT_DIR:
#    summary.json          effective_frames vs effective_frames_if_uniform for
#                          both conditions; their ratio is the selectivity
#    weights_control_*.csv  the attention profile at the run's own k_prevs
#    weights_probe_k*.csv   the same at LOOKBACK, where selection has room to show
#    dice_vs_depth.csv      only when SCORE_DEPTHS is set
#
#  HOW TO READ IT. effective_frames / effective_frames_if_uniform is the number
#  to look at. 1.0 is exactly uniform, i.e. dead attention. The measured
#  reference points, from the 2026-08-19 valneg runs:
#
#    geo_k5  tattn fixed    0.757   selecting
#    geo_k10 tattn fixed    0.788   selecting
#    geo_k10 tattn hybrid   0.893   marginal
#    temporal_k5 fixed      0.928   effectively uniform
#    geo_k10 prefix (old)   1.000   dead -- the pre-fix control, as designed
#
#  REQUIRE_SELECTIVITY makes the job FAIL rather than report quietly, which is
#  what makes this usable in a batch: a non-zero exit is greppable in bhist.
#  0.9 is the collapse alarm the attnfix block specifies.
# ============================================================================
#BSUB -J probe_manual
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/%J.err
#BSUB -q short-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=24G
#BSUB -R rusage[mem=96GB]
#BSUB -W 3:00

set -o pipefail

# ------------------------------ CONFIG --------------------------------------
# Each honours an environment variable of the same name, which is how
# scripts/submit_all.sh drives batches without editing this file.
RUN="${RUN:-outputs/2026-08-19/geo_k10_tattn_fixed_ring3_valneg}"
CKPT="${CKPT:-best.pt}"
GROUP="${GROUP:-geo_k10}"     # geo_k5 | geo_k10 | temporal_k5 | temporal_k10
SPLIT="${SPLIT:-val}"         # the probe reads where attention LANDS, so val is
                              # the right split: it is the data the checkpoint
                              # was selected on, and no metric is being claimed.
GEN="${GEN:-3}"               # PARTITION FAMILY -- assets/PARTITIONS.md.
                              # 2   = partition_<group>.json          (generation 2)
                              # 3   = partition_<group>_clean.json    (the benchmark)
                              # p23 = partition_<group>_pre2023.json  (2019-2022)
                              # A run must be probed on the partition it was
                              # TRAINED on: this measures the model's behaviour
                              # on its own data, not transfer.
CONTROL_LOOKBACK="${CONTROL_LOOKBACK:-10}"   # SET THIS TO THE RUN'S k_prevs.
                              # The control condition is the model in the shape
                              # it was trained in; leaving the default on a k5
                              # run probes a depth that run never saw and the
                              # selectivity figure stops meaning what the table
                              # above means.
LOOKBACK="${LOOKBACK:-40}"    # probe depth in 11-day slots, ~14 months. The
                              # condition where selection has room to show:
                              # uniform over 40 frames is unmistakable, uniform
                              # over 6 is nearly indistinguishable from selective.
SCHEDULE="${SCHEDULE:-dense}"
MAX_PER_INTF="${MAX_PER_INTF:-16}"   # patches per interferogram. Bounds the
                              # READ, not the compute: the grids are 1.4 GB each
                              # and LOOKBACK of them are held at once, which is
                              # what the memory request is for.
MAX_INTFS="${MAX_INTFS:-0}"   # 0 = the whole split
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
SCORE_DEPTHS="${SCORE_DEPTHS:-}"     # space-separated, e.g. "1 5 10 20 40".
                              # Worth setting whenever the weights come out
                              # uniform: a model that genuinely averages should
                              # suppress temporally inconsistent noise BETTER
                              # the more frames it averages, and that prediction
                              # is testable without retraining anything.
REQUIRE_SELECTIVITY="${REQUIRE_SELECTIVITY:-0.9}"   # empty disables the alarm
OUT_NAME="${OUT_NAME:-}"      # subdirectory under outputs/attention_probe/.
                              # Defaults to the run's own basename, which is the
                              # convention the existing probe dirs follow and is
                              # what keeps run / prediction / probe trees
                              # matching by name (outputs/README.md).
# ----------------------------------------------------------------------------

REPO=/home/labs/rudich/pinkas/sinkholes
PATCHES=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches
cd "$REPO" || { echo "cannot cd to $REPO" >&2; exit 1; }

case "$GEN" in
  2)   GEN_SUFFIX="" ;;
  3)   GEN_SUFFIX="_clean" ;;
  p23) GEN_SUFFIX="_pre2023" ;;
  *)   echo "GEN must be 2, 3 or p23, got '$GEN'" >&2; exit 1 ;;
esac
PARTITION="assets/partition_${GROUP}${GEN_SUFFIX}.json"

MODEL="$RUN/checkpoints/$CKPT"
OUT_DIR="outputs/attention_probe/${OUT_NAME:-$(basename "$RUN")}"

# -- preflight: fail here, not 20 minutes into a GPU slot ---------------------
[ -f "$MODEL" ]     || { echo "no checkpoint at $MODEL" >&2; exit 1; }
[ -f "$PARTITION" ] || { echo "no partition file $PARTITION" >&2; exit 1; }

# The probe is only defined for a temporal-attention checkpoint. Every other
# architecture loads and then has no attention weights to report, which is a
# confusing way to spend a GPU slot -- so say so here.
case "$(basename "$RUN")" in
  *tattn*) : ;;
  *) echo "$(basename "$RUN") is not a tattn run: the probe has nothing to measure." >&2
     echo "  ConvLSTM and single-frame runs have no temporal attention block." >&2
     exit 1 ;;
esac

# CONTROL_LOOKBACK must be the run's own k_prevs or the headline ratio is not
# the quantity the reference table is built from. The run name carries the
# group, and the group carries k -- so a mismatch is checkable and worth a loud
# note. It is a NOTE rather than an error because probing a k5 run at k10 is a
# legitimate (if unusual) question, just not the one the alarm is calibrated for.
case "$GROUP" in
  *_k5)  EXPECT=5 ;;
  *_k10) EXPECT=10 ;;
  *)     EXPECT="" ;;
esac
if [ -n "$EXPECT" ] && [ "$CONTROL_LOOKBACK" != "$EXPECT" ]; then
  echo "NOTE: CONTROL_LOOKBACK=$CONTROL_LOOKBACK on group $GROUP (k_prevs $EXPECT)." >&2
  echo "      The selectivity alarm is calibrated on the run's own depth." >&2
fi

echo "model      $MODEL"
echo "partition  $PARTITION  ($SPLIT split of $GROUP, generation $GEN)"
echo "control    k${CONTROL_LOOKBACK}   probe k${LOOKBACK} ($SCHEDULE)"
echo "alarm      ${REQUIRE_SELECTIVITY:-none}$([ -n "$REQUIRE_SELECTIVITY" ] && echo " (non-zero exit if attention is at or above this fraction of uniform)")"
echo "output     $OUT_DIR"

source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# `if` rather than `[ ... ] && arr=(...)`: under `set -e` a bare test that fails
# IS the command's exit status, so the empty-default cases (SCORE_DEPTHS unset,
# MAX_INTFS=0) would abort the job before it started rather than skip a flag.
REQ_FLAGS=()
if [ -n "$REQUIRE_SELECTIVITY" ]; then
  REQ_FLAGS=(--require_selectivity "$REQUIRE_SELECTIVITY")
fi
DEPTH_FLAGS=()
if [ -n "$SCORE_DEPTHS" ]; then
  # Unquoted on purpose: SCORE_DEPTHS is a space-separated list of depths.
  DEPTH_FLAGS=(--score_depths $SCORE_DEPTHS)
fi
INTF_FLAGS=()
if [ "$MAX_INTFS" != 0 ]; then
  INTF_FLAGS=(--max_intfs "$MAX_INTFS")
fi

python -m sinkholes attention-probe \
  --model "$MODEL" \
  --partition "$PARTITION" \
  --split "$SPLIT" \
  --patches_dir "$PATCHES" \
  --patch_size 200 100 \
  --lookback "$LOOKBACK" \
  --schedule "$SCHEDULE" \
  --control_lookback "$CONTROL_LOOKBACK" \
  --max_per_intf "$MAX_PER_INTF" \
  ${INTF_FLAGS[@]+"${INTF_FLAGS[@]}"} \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --out_dir "$OUT_DIR" \
  ${DEPTH_FLAGS[@]+"${DEPTH_FLAGS[@]}"} \
  ${REQ_FLAGS[@]+"${REQ_FLAGS[@]}"}
