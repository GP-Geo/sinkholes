#!/usr/bin/env bash
# ============================================================================
#  Full-scene evaluation template
#
#  Submit:  bsub < scripts/eval/run_eval.sh
#
#  Two stages, on whichever interferograms you point it at:
#    1. eval-scenes   reconstructs each interferogram from its patch grid (+ k
#                     previous frames), LiDAR-gates, thresholds, polygonises.
#                     Writes <intf>_{image,pred,pred_th,gt}.npy and shapefiles.
#    2. eval-outputs  re-thresholds the saved confidence at 0.125/0.25/0.5 and
#                     computes object-level precision/recall into a JSON.
#
#  EDIT TWO PLACES: the "#BSUB -J" line (names the job and the log files) and
#  the CONFIG block. The defaults score the current leader the current way
#  (generation 5, stride 4, the paper's RTh protocol).
#
#  ARCH, K_PREVS and the patch geometry must match how the checkpoint was
#  TRAINED, not what you wish it were -- a mismatch either fails to load or
#  silently feeds the network the wrong stack. The run's log header records
#  what each run was trained with.
# ============================================================================
#BSUB -J eval_temporal_k10_fill
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/%J.err
# short-gpu rejects these: "RUNLIMIT: Cannot exceed queue's hard limit(s)".
# These directives are the ONE-OFF defaults, sized for a plain temporal model at
# stride 4 (k5 48 GB, k10 75 GB measured, LSF 330515); submit_all.sh overrides
# all four on the bsub command line per job. A single-frame model needs far less
# (27.8 GiB), and the legacy ctx50 k10 engine measured 224 GB.
#BSUB -q long-gpu
#BSUB -gpu num=1:j_exclusive=yes:gmem=36G
#BSUB -R rusage[mem=160GB]
#BSUB -W 16:00

set -o pipefail

# ------------------------------ CONFIG --------------------------------------
# Edit the defaults here for a one-off run. Each also honours an environment
# variable of the same name, which is how scripts/submit_all.sh drives batches
# without editing the file: GROUP=temporal_k10 K_PREVS=5 bsub -J <name> < this
RUN="${RUN:-outputs/2026-10-06/temporal_k10_th350_fill_tattn_posw8_30e}"
CKPT="${CKPT:-best.pt}"     # best.pt | last.pt (last.pt was pruned from most runs)
GROUP="${GROUP:-temporal_k10}"  # temporal_k5 | temporal_k10
SPLIT="${SPLIT:-test}"      # val = the split the model was selected on
                            # test = the held-out split (never seen)
GEN="${GEN:-fill}"          # PARTITION GENERATION -- assets/PARTITIONS.md. Every
                            # one carries a per-split aoi_window, passed to BOTH
                            # stages below (see AOI_FLAGS).
                            # 3 = partition_<group>_clean.json, the AOI benchmark
                            #     with no label-quality threshold.
                            # th350 = partition_<group>_clean_th350x200.json,
                            #     generation 3 with the 350N/200S label-quality
                            #     threshold on every split (generation 4): 20 test
                            #     scenes at k5 (the official relabelled set), 17 at k10.
                            # fill = partition_<group>_clean_fill_th350x200.json,
                            #     th350 with chains that skip missing
                            #     acquisitions (generation 5): 24 test scenes at
                            #     both k5 and k10, a SUPERSET of th350's 20 / 17.
                            #     For like-for-like with th350, re-average the
                            #     per-scene results over the th350 list.
                            # t2026 = partition_<group>_test2026_fill_th350x200.json,
                            #     the fill family re-split by date blocks with
                            #     FULL separation: test = 2026 (10 scenes, a
                            #     SUBSET of fill's 24), val = 2022-2023, train =
                            #     the rest incl. 2024-25. To compare with a fill
                            #     model, re-average its per-scene results over
                            #     these 10 -- same chains, same window.
                            # Generations are NOT comparable: different scene
                            # lists AND different scored ground.
DATA_STRIDE="${DATA_STRIDE:-4}"  # strides per patch of the reconstruction grid.
                            # 4 = QUARTER-patch step, 16 tiles per pixel -- the
                            #     paper's reconstruction geometry, and what every
                            #     RTh number uses. Reads the *_strpp4_* tree:
                            #     5.47 GB per timestep.
                            # 2 = half-patch step, 4 tiles per pixel (1.37 GB per
                            #     timestep); only for PROTOCOL=prob.
ARCH="${ARCH:-tattn}"       # tattn | convlstm | stack | single -- must match training
K_PREVS="${K_PREVS:-10}"    # must match the CHECKPOINT, not the partition
PROTOCOL="${PROTOCOL:-rth}"   # rth | prob -- WHAT THE CONFIDENCE MAP MEANS.
                            # prob = mean predicted probability over the tiles
                            #     covering each pixel, cut at probability
                            #     thresholds. Every number before 2026-08-19.
                            # rth  = the BENCHMARK PAPER'S protocol. Each tile
                            #     is binarised at 0.5 and the pixel value is the
                            #     FRACTION OF OVERLAPPING TILES voting positive
                            #     (the paper's Confidence Factor), swept at its
                            #     Reconstruction Thresholds 0.125/0.25/0.375/0.5
                            #     under both its Intersection Tolerances
                            #     (ITh 0.7/b 5 and the softer ITh 0.5/b 10).
                            #     Pair it with DATA_STRIDE=4 -- RTh 0.25 is
                            #     defined as "4 of 16 tiles agreed", and at
                            #     stride 2 there are only 4 tiles per pixel, so
                            #     the quantisation is 1/4 and the thresholds
                            #     stop meaning what the paper says they mean.
                            # An RTh number and a recon_th number are NOT
                            # comparable: one counts tiles, one cuts a mean.
RECON_TH="${RECON_TH:-0.25}"  # threshold eval-scenes writes polygons at.
                            # Under PROTOCOL=rth this is an RTh, not a
                            # probability. It does NOT affect the metrics --
                            # eval-outputs re-thresholds the saved confidence
                            # map itself -- only the shapefiles from stage 1.
MIN_POS="${MIN_POS:-150}"   # SCENE GATE: skip interferograms with this many
                            # positive patches or fewer (strictly more is kept).
                            # Thin scenes make the object-level mean per-scene
                            # precision/recall noisy -- every scene carries equal
                            # weight, so one miss on a 4-positive scene moves its
                            # recall by 0.25. The count is nonz_num, which is
                            # WHOLE-SCENE and is NOT restricted to the AOI window.
                            # 0 disables. CHANGES THE SCENE LIST.

OL_TH="${OL_TH:-0.7}"       # object-level overlap threshold (eval-outputs)
BUFFER="${BUFFER:-5}"       # object-matching buffer, pixels
BLEND="${BLEND:-none}"      # none | hann -- how overlapping tile predictions are
                            # combined at stitch time. 'none' is a flat average
                            # (every tile votes equally on every pixel it covers,
                            # including the pixels at its own border, which the
                            # network predicted from truncated context). 'hann'
                            # weights each tile by a Hann window, so a pixel is
                            # scored mostly by the tiles that saw it CENTRALLY.
                            # Every number in docs/RESULTS.md is BLEND=none.
GAMMA="${GAMMA:-1.0}"       # window sharpening exponent; BLEND=hann only. >1
                            # concentrates weight further toward tile centres.
# Names the OUTPUT DIRECTORY, independently of the LSF job name, e.g.
# `scenes_temporal_k10_fill_rth` (outputs/README.md). LSF job names have to stay
# unique and greppable in bjobs, so the two cannot be the same string; without
# this override the directory inherits whatever -J was.
JOB_NAME="${JOB_NAME:-}"
# ----------------------------------------------------------------------------

REPO=/home/labs/rudich/pinkas/sinkholes
PATCHES=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches
# Every path below is relative to the repo root, so a failed cd must not be
# survivable -- otherwise the preflight checks run against the wrong tree.
cd "$REPO" || { echo "cannot cd to $REPO" >&2; exit 1; }

# -- resolve the partition ----------------------------------------------------
# `eval-scenes --intf_source preset` only ever reads a partition's "val" list
# (sinkholes/inference/scenes.py). To evaluate a TEST split we therefore point
# it at the _testeval variant, whose "val" is exactly the parent's "test" list.
case "$GEN" in
  3)     GEN_SUFFIX="_clean" ;;
  th350) GEN_SUFFIX="_clean_th350x200" ;;
  fill)  GEN_SUFFIX="_clean_fill_th350x200" ;;
  t2026) GEN_SUFFIX="_test2026_fill_th350x200" ;;
  *)     echo "GEN must be 3, th350, fill or t2026, got '$GEN' (retired generations:" >&2
         echo "git show notebook-2026-10-15:assets/<partition file>)" >&2; exit 1 ;;
esac
case "$SPLIT" in
  val)  PARTITION="assets/partition_${GROUP}${GEN_SUFFIX}.json" ;;
  test) PARTITION="assets/partition_${GROUP}_testeval${GEN_SUFFIX}.json" ;;
  *)    echo "SPLIT must be 'val' or 'test', got '$SPLIT'" >&2; exit 1 ;;
esac

# -- AOI window ---------------------------------------------------------------
# Generation-3 partitions carry a per-split lat/lon box (assets/PARTITIONS.md).
# It has to reach BOTH stages: eval-scenes decides which tiles are PREDICTED,
# eval-outputs crops the canvas it SCORES. If only one gets it, the metrics
# cover different ground than the predictions -- outputs.py:_resolve_aoi says so
# in as many words.
#
# --aoi_split is 'val', NOT 'test', and not the tools' own default of 'test':
#   * the interferogram list read above is always the "val" key, because
#     --intf_source preset reads only that key (scenes.py:114);
#   * the _testeval files hold the parent's TEST list under "val", and have no
#     "test" key at all -- their aoi_window is {train, val, crossview}.
# So 'val' is the window belonging to the scenes actually being scored, in both
# SPLIT modes. Passing 'test' against a _testeval file exits with a message
# rather than mis-scoring, which is the intended failure (partition.py:161).
#
# Every generation still in assets/ is generation 3's generator (th350 adds a
# positive-count threshold, fill longer chains, t2026 a block re-split), so all
# carry the same per-split window and take the same flags.
AOI_FLAGS=(--aoi_from_partition "$PARTITION" --aoi_split val)

# -- architecture flags -------------------------------------------------------
# The factory DETECTS the architecture from the checkpoint's own tensors
# (sinkholes/models/factory.py:272); these flags are the override, and they
# also carry --k_prevs, which the detector cannot supply because it is a
# property of the data stack rather than the weights.
#
# tattn covers all four 2026-08-10 attention runs, fuse_skips and
# tattn_recurrence included: both are read back out of the checkpoint's
# tattn_unet_config blob, so the hybrid (RECURRENCE=convlstm) rebuilds
# correctly under --tattn_unet and must NOT be evaluated as ARCH=convlstm --
# factory.py:221 orders tattn ahead of convlstm precisely because the hybrid
# carries a ConvLSTM cell and would otherwise match the wrong entry.
case "$ARCH" in
  convlstm) ARCH_FLAGS=(--convlstm_unet --k_prevs "$K_PREVS") ;;
  tattn)    ARCH_FLAGS=(--tattn_unet --k_prevs "$K_PREVS") ;;
  stack)    ARCH_FLAGS=(--k_prevs "$K_PREVS") ;;
  single)   ARCH_FLAGS=(--k_prevs 0) ;;
  *)        echo "ARCH must be 'convlstm', 'tattn', 'stack' or 'single', got '$ARCH'" >&2; exit 1 ;;
esac

# -- blending flags -----------------------------------------------------------
# Validated here rather than left to argparse: a typo ("hanning") would be
# rejected by eval-scenes only AFTER the GPU slot is allocated and the patch
# tree loaded, ~20 minutes in.
#
# BLEND_FLAGS is EMPTY under the default, and this script runs `set -u` before
# the python call. On bash < 4.4 a plain "${BLEND_FLAGS[@]}" is an unbound
# variable when the array is empty and would abort the job, so it is expanded
# below as ${BLEND_FLAGS[@]+"..."} -- do not "simplify" that back.
case "$BLEND" in
  none) BLEND_FLAGS=() ;;
  hann) BLEND_FLAGS=(--blend_type hann --window_gamma "$GAMMA") ;;
  *)    echo "BLEND must be 'none' or 'hann', got '$BLEND'" >&2; exit 1 ;;
esac

# -- protocol flags -----------------------------------------------------------
# The two stages must agree: stage 1 decides what the confidence map MEANS
# (mean probability vs tile-vote fraction), stage 2 decides how it is cut.
# Scoring a vote map with probability thresholds, or the reverse, produces
# numbers that look plausible and are not, so both flags come from one switch.
case "$PROTOCOL" in
  prob) SCENE_PROTO_FLAGS=(); OUT_PROTO_FLAGS=() ;;
  rth)  SCENE_PROTO_FLAGS=(--recon_average vote --vote_threshold 0.5)
        OUT_PROTO_FLAGS=(--rth) ;;
  *)    echo "PROTOCOL must be 'prob' or 'rth', got '$PROTOCOL'" >&2; exit 1 ;;
esac

# The paper's RTh values are fractions of SIXTEEN tiles. Stride 2 gives four.
if [ "$PROTOCOL" = rth ] && [ "$DATA_STRIDE" != 4 ]; then
  echo "PROTOCOL=rth expects DATA_STRIDE=4 (16 tiles/pixel); got $DATA_STRIDE." >&2
  echo "      At stride $DATA_STRIDE the Confidence Factor quantises differently and" >&2
  echo "      the paper's 0.125/0.25/0.5 stop meaning 2/4/8 of 16." >&2
  exit 1
fi
if [ "$PROTOCOL" = rth ] && [ "$BLEND" != none ]; then
  echo "PROTOCOL=rth and BLEND=$BLEND are exclusive: a vote is unweighted." >&2
  exit 1
fi

# -- preflight: fail here, not 20 minutes into a GPU slot ---------------------
MODEL="$RUN/checkpoints/$CKPT"
[ -f "$MODEL" ]     || { echo "no checkpoint at $MODEL" >&2; exit 1; }
[ -f "$PARTITION" ] || { echo "no partition file $PARTITION" >&2; exit 1; }
# K_PREVS must match the CHECKPOINT, not the partition -- at eval time the
# partition only supplies the list of interferograms to score. Deliberately
# mismatching them is how you compare a k5 model against a k10 model on one
# scene list: temporal_k10's test split is a subset of temporal_k5's (and
# geo_k10's of geo_k5's), and every scene in both has 5- AND 10-previous
# chains, so the k10 list is the common ground. Hence a note, not an error.
if [ "$ARCH" != "single" ]; then
  case "$GROUP" in
    *_k5)  [ "$K_PREVS" = 5 ]  || echo "NOTE: K_PREVS=$K_PREVS on group $GROUP (k5 scene list)." >&2 ;;
    *_k10) [ "$K_PREVS" = 10 ] || echo "NOTE: K_PREVS=$K_PREVS on group $GROUP (k10 scene list) -- expected when scoring a k5 model on the shared list." >&2 ;;
  esac
fi

# Run directories carry their training group somewhere in the name, so the
# run's own group is recoverable. Evaluating a model outside the partition it
# was trained on is a legitimate transfer experiment, but it is far more often
# a typo -- and its score is NOT comparable to that model's usual numbers
# (docs/RESULTS.md).
#
# Two naming schemes are in play and both must parse. Runs up to 2026-08-09
# lead with the group ("geo_k10_convlstm_posw4"); the
# 2026-08-10 batch leads with the ARCHITECTURE instead
# ("convlstm_geo_k5_h256_posw4_neg1x_ring3_60e_..."), so the old
# `cut -d_ -f1` read "convlstm" as the group and warned on every single run.
# Match the group token wherever it sits; an unrecognised name (the retired
# k10split runs) yields empty and simply skips the check.
RUN_GROUP=$(basename "$RUN" | tr '-' '_' | grep -oE '(geo|temporal)_k[0-9]+' | head -1)
if [ -n "$RUN_GROUP" ] && [ "$RUN_GROUP" != "$GROUP" ]; then
  echo "NOTE: run was trained on '$RUN_GROUP' but you are evaluating on '$GROUP'." >&2
  echo "      Cross-partition scores are not comparable to either group." >&2
fi

JOB="${JOB_NAME:-${LSB_JOBNAME:-eval_manual}}"
OUTROOT="outputs/predictions/$(basename "$RUN")"

echo "model     $MODEL"
echo "partition $PARTITION  ($SPLIT split of $GROUP, generation $GEN)"
echo "arch      $ARCH (k_prevs ${K_PREVS})"
echo "stride    $DATA_STRIDE strides/patch$([ "$DATA_STRIDE" = 4 ] && echo " -- the paper's 16 tiles/pixel geometry")"
echo "protocol  $PROTOCOL$([ "$PROTOCOL" = rth ] && echo " -- Confidence Factor + RTh 0.125/0.25/0.375/0.5, ITh 0.7/b5 and 0.5/b10" || echo " (mean probability)")"
echo "aoi       $([ ${#AOI_FLAGS[@]} -gt 0 ] && echo "from $PARTITION [val]" || echo "none (whole canvas)")"
echo "blend     $BLEND$([ "$BLEND" = hann ] && echo " (gamma $GAMMA)")"
echo "output    $OUTROOT/${CKPT%.pt}/${JOB}_<ts>"

source /apps/easybd/easybuild/amd/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh || exit 1
conda activate /home/labs/rudich/pinkas/.conda/envs/sinkholes || exit 1
set -eu

# -- stage 1: reconstruct, LiDAR-gate, threshold, polygonise ------------------
python -m sinkholes eval-scenes \
  --model "$MODEL" \
  --input_patch_dir "$PATCHES" \
  --intf_source preset --valset_from_partition "$PARTITION" \
  --patch_size 200 100 --data_stride "$DATA_STRIDE" --days_diff 11 \
  --recon_th "$RECON_TH" \
  "${ARCH_FLAGS[@]}" \
  ${BLEND_FLAGS[@]+"${BLEND_FLAGS[@]}"} \
  ${AOI_FLAGS[@]+"${AOI_FLAGS[@]}"} \
  ${SCENE_PROTO_FLAGS[@]+"${SCENE_PROTO_FLAGS[@]}"} \
  --add_lidar_mask \
  --min_positives "$MIN_POS" \
  --save_confidence --merge_polygs \
  --job_name "$JOB" --output_dir "$OUTROOT"

# eval-scenes nests under <output_dir>/<checkpoint stem>/<job>_<timestamp>;
# it stamps the directory with its start time, so pick the newest.
#
# The timestamp is matched EXPLICITLY (scenes.py:164, "%m_%d_%Hh%M") rather than
# with a bare ${JOB}_* glob. Since the rename of 2026-08-17 the job dirs are
# `scenes_geo` and `scenes_geo_hann` in the same parent, so ${JOB}_* with
# JOB=scenes_geo also matches scenes_geo_hann -- an unblended run would then
# hand stage 2 the BLENDED directory and silently re-score the wrong maps.
EVAL_DIR=$(ls -dt "$OUTROOT/${CKPT%.pt}/${JOB}"_[0-9][0-9]_[0-9][0-9]_[0-9][0-9]h[0-9][0-9] | head -1)
echo "eval-scenes wrote $EVAL_DIR"

# -- stage 2: object-level metrics + per-interferogram figures ----------------
# --patch_size/--data_stride are passed explicitly rather than left to the
# argparse defaults (200x100, stride 2): they drive the AOI crop, and a stride-4
# reconstruction cropped with a stride-2 grid lands on the wrong pixels
# (outputs.py:44). They equal the old defaults at DATA_STRIDE=2, so archived
# evaluations re-score unchanged.
python -m sinkholes eval-outputs \
  --path "$EVAL_DIR" \
  --patch_size 200 100 --data_stride "$DATA_STRIDE" \
  --th "$OL_TH" --buffer "$BUFFER" \
  ${AOI_FLAGS[@]+"${AOI_FLAGS[@]}"} \
  ${OUT_PROTO_FLAGS[@]+"${OUT_PROTO_FLAGS[@]}"} \
  --save_figures

echo "done: $EVAL_DIR"
