#!/usr/bin/env bash
# ============================================================================
#  Batch submitter for the planned training and evaluation jobs.
#
#    scripts/submit_all.sh                   # dry run: print the plan, submit nothing
#    scripts/submit_all.sh train             # dry run, training jobs only
#    scripts/submit_all.sh training --submit # submit ConvLSTM + temporal-attention training
#    scripts/submit_all.sh control --submit  # the single-frame U-Net 2x2 control
#    scripts/submit_all.sh clean22 --submit  # THE CLEAN BENCHMARK: six runs, all with
#                                            # ring negatives in TRAINING
#    scripts/submit_all.sh valpos  --submit  # six of the clean22 runs again, with the
#                                            # negatives in TRAINING ONLY
#    scripts/submit_all.sh attnfix --submit  # THE ATTENTION FIX: the clean22 tattn arms
#                                            # re-run with attention that actually
#                                            # selects, plus the paired pre-fix control
#                                            # (DONE 2026-08-20 -- do not resubmit)
#    scripts/submit_all.sh attnpos --submit  # THE QUEUED WORK: four of those arms
#                                            # again on a positives-only val set,
#                                            # plus four new temporal arms
#    scripts/submit_all.sh pre23   --submit  # THE 2019-2022 ARCHIVE: eleven arms on
#                                            # partitions that stop at 2022-12-31,
#                                            # to measure how much of the
#                                            # false-positive rate is the newer
#                                            # years. Five are attention arms
#    scripts/submit_all.sh eval4   --submit  # the 2026-08-11 evals (DONE 2026-08-12)
#    scripts/submit_all.sh eval5   --submit  # THE CLEAN BENCHMARK EVALS: 8 of the 14
#                                            # clean22 runs, GEN=3 partitions at the
#                                            # paper's stride-4 geometry
#    scripts/submit_all.sh posonly --submit  # positives-only, the paper's protocol
#    scripts/submit_all.sh blend   --submit  # Hann-blended stitching, the geo_k5 triple
#                                            # (2 of 3 -- the single-frame arm is done)
#    scripts/submit_all.sh eval    --submit  # the 2026-08-09 eval backlog
#    scripts/submit_all.sh eval2   --submit  # the 2026-08-10 evals  (DONE 2026-08-11)
#    scripts/submit_all.sh eval3   --submit  # second-wave evals     (DONE 2026-08-11)
#    scripts/submit_all.sh rescore --submit  # cheap re-scoring of existing evals
#    scripts/submit_all.sh --submit          # submit everything
#
#    scripts/submit_all.sh --only hybrid --submit   # ONE job, by name substring
#
#  --only narrows to jobs whose name contains the given text, so a single job
#  can be sent without its whole kind riding along (an eval kind is 4-6 jobs at
#  ~3 h of long-gpu each). Repeatable. Combine with a kind to narrow within it.
#
#  `training`, `tattn` and `control` ARE EMPTY. The 2026-08-10 batch is archived
#  under outputs/2026-08-10, and the four jobs of the 2026-08-11 batch (two
#  `tattn` geo arms, two `control` single-frame arms) finished cleanly and now
#  sit at the top level of outputs/. They have been retired from the job table
#  for the same reason every finished batch is: leaving them listed means one
#  stray `training --submit` retrains ~50 GPU-hours of completed work. What they
#  were trained with is kept in the commented blocks below, so any one of them
#  can be resubmitted by uncommenting a single `job` line.
#
#  Four kinds are live:
#    pre23   ELEVEN arms on the 2019-2022 archive (assets/partition_*_pre2023.json,
#            generated 2026-08-20). Same generator, cut, AOI, seed and val share
#            as clean22 -- the archive stops at 2022-12-31 and nothing else
#            moves, so a pre23 run read against its clean22 twin measures the
#            newer years and nothing else. That is the open question section 0a
#            of docs/PLAN_CLEAN_BENCHMARK.md accepted knowingly rather than
#            answered. FIVE of the eleven are attention arms, because
#            attention's hypothesis IS this batch's hypothesis: it claims to
#            suppress temporally inconsistent artefacts, which is what section 1
#            measured accumulating in the post-2022 scenes. ~223 h of walltime
#            requests at the 100-epoch ceiling, materially less with early
#            stopping; the pools are 50-66% of their clean22 twins.
#    attnpos EIGHT arms on a POSITIVES-ONLY validation set: four of the five
#            attnfix arms re-run (the prefix control is not repeated -- attnfix
#            already ran that pair on one protocol), plus FOUR NEW temporal
#            arms that have no positives-only twin. attnfix itself is DONE --
#            all five finished by 2026-08-20 05:41, four of them early-stopping
#            on patience 40 against a padded curve that also chose their
#            best.pt. ~80 GPU-hours at the 100-epoch ceiling, and materially
#            less in practice: early stopping ended four of five attnfix runs,
#            and losing the validation negatives took 35% off the measured
#            epoch time of the one run that exists in both protocols.
#    eval5   the object-level scores for 8 of the 14 clean22 runs, on the
#            generation-3 partitions at stride 4. clean22 itself is DONE --
#            all 14 finished 2026-08-19; do not resubmit it, it is ~50
#            GPU-hours of completed training.
#    valpos  six of those fourteen re-run with the negatives in TRAINING ONLY.
#            clean22's val/dice of 0.72-0.80 is inflated by the validation
#            negatives -- an empty prediction on an empty mask scores dice 1.0,
#            so at 1:1 the mean is roughly (1 + dice_on_positives)/2 and is not
#            comparable to any earlier batch. ~52 GPU-hours. See the block
#            below for what it costs on checkpoint selection.
#
#  attnpos and valpos are the same correction applied to the two training
#  batches that used negative validation. Between them they re-establish one
#  val protocol across everything trained on the generation-3 partitions.
#
#  `valneg` and `eval4` are retired: eval4 completed 2026-08-12, and valneg is
#  gone with negative validation itself (2026-08-20). VALIDATION IS NOW
#  POSITIVES-ONLY FOR EVERY JOB IN THIS FILE -- no template passes
#  --add_val_negatives any more, and the `valneg` kind is rejected by name.
#  What replaces it: keep background patches in TRAINING ($NEG_R3_1X) and
#  settle every precision claim at object level with scripts/eval/run_eval.sh.
#
#  DRY RUN IS THE DEFAULT. Nothing reaches LSF without --submit.
#
#  Each job is one line in the table below: name, template, CONFIG overrides,
#  and RESOURCES. The templates read every CONFIG value as ${VAR:-default}, so
#  overrides need no file edits, and the LSF job name (-J) flows through to
#  --job_name via $LSB_JOBNAME.
#
#  RESOURCES = "queue host_mem_GB gpu_vram_G walltime". They are passed on the
#  bsub COMMAND LINE, which overrides the #BSUB directives baked into each
#  template -- those are static and cannot vary per job, which is exactly why
#  they live here instead.
#
#  ---- where the numbers come from ------------------------------------------
#  Host memory is measured: every finished job's "Max Memory" from its LSF
#  summary in logs/*.out. The pattern is that host RAM tracks the number of
#  FRAMES per sample, not model size -- h1024 peaked at 64.5 GB against h256's
#  65.4 GB, while k5 sits near 41 GB and single-frame near 6 GB:
#
#      single-frame   ~6 GB      k5 ConvLSTM  ~41 GB     k10 ConvLSTM  ~65 GB
#      k10 unet-stack ~66 GB     eval         ~57 GB
#
#  Requests below are the measured peak plus ~40%. Several were wildly
#  over-requested before (the single-frame baselines asked 70 GB and used 6),
#  which only delays scheduling.
#
#  GPU VRAM was NOT measured until 2026-08-10 -- LSF records no gmem usage and
#  no job has ever hit a CUDA OOM, so the ladder was a safe guess (48G standard
#  / 64G for k=10). That guess cost real time: on 2026-08-10 seven jobs sat PEND
#  behind it. Over-requesting VRAM does not just waste memory, it excludes whole
#  classes of GPU -- a 48G request cannot land on a 40G A100 at all, so the job
#  waits for a bigger card instead of running on an idle one.
#
#  The ladder is now an ESTIMATE (see below), and every run logs its own peak
#  through torch.cuda.max_memory_allocated(). Read the real numbers back with
#      grep "peak VRAM" logs/*.out outputs/*/*.log
#  and replace the estimate with measurement as soon as a few runs finish.
#
#  Walltime: longest observed training run was 4.4 h and the eval 3.0 h. The
#  10 h / 8 h below are ~2x that -- deliberately, because a k5 b64 run has
#  already been killed once by TERM_RUNLIMIT at 6.0 h.
# ============================================================================

set -euo pipefail

# This script SUBMITS jobs; it is not one. It carries no #BSUB directives, so
# `bsub < submit_all.sh` would queue the submitter itself onto a compute node
# and waste a slot. Run it with bash on the login node instead -- the three
# templates under train/ and eval/ are the actual job scripts.
if [ -n "${LSB_JOBID:-}" ]; then
  echo "submit_all.sh is running under LSF (job $LSB_JOBID) -- that is not how it works." >&2
  echo "It is the submitter, not a job. On the login node run:" >&2
  echo "    bash scripts/submit_all.sh --submit" >&2
  exit 1
fi

REPO=/home/labs/rudich/pinkas/sinkholes
[ -d "$REPO" ] || REPO="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"
cd "$REPO" 2>/dev/null || { echo "cannot cd to '$REPO'" >&2; exit 1; }
# $0 is unreliable when the script is piped (bash < submit_all.sh makes it
# "bash", so the fallback above lands in the repo's PARENT and every relative
# path silently misses). Confirm we are actually at the repo root.
if [ ! -f scripts/train/train_convlstm.sh ] || [ ! -d assets ]; then
  echo "'$REPO' is not the repo root (no scripts/train/ + assets/)." >&2
  echo "Run as: bash scripts/submit_all.sh   -- not piped from stdin." >&2
  exit 1
fi

TEMPORAL_K5=assets/partition_temporal_k5.json
GEO_K5=assets/partition_geo_k5.json

# The 2026-08-09 batch, archived (docs/MODEL_RUNS.md retention: results.csv,
# log, curves.png, validation/ and best.pt kept; resume.pt and last.pt pruned).
A09=outputs/2026-08-09
G10_SEED7=$A09/geo_k10_convlstm_seed7
G10_POSW4=$A09/geo_k10_convlstm_posw4
G10_SINGLE=$A09/geo_k10_single
G10_STACK=$A09/geo_k10_stack
T5_POSW1=$A09/temporal_k5_convlstm_posw1
T5_POSW2=$A09/temporal_k5_convlstm_posw2

# The 2026-08-10 batch: all 13 jobs (9 negative-sampling + 4 attention) ran to
# completion, every one of them stopping on early-stopping patience rather than
# the walltime, and are archived under the same retention rule -- except that
# best.pt is kept for ALL 13 because none has been scored at object level yet.
# Deleting any of them now would cost a retrain to run the eval below.
#
# Every run directory was renamed on 2026-08-17 to
# <partition>_<arch>[_<variant>], dropping the LSF id, timestamp, epoch count
# and any setting that is the default (h256, pos_w 4, b128, lr1e-6). See
# outputs/README.md. run_eval.sh derives GROUP/ARCH from the flags below, not
# from the directory name, so the rename does not affect it.
A10=outputs/2026-08-10
T5_POSW4=$A10/temporal_k5_convlstm_base
T5_NEG1_R3=$A10/temporal_k5_convlstm_ring3
T5_NEG1_R10=$A10/temporal_k5_convlstm_ring10
T5_NEG3_R3=$A10/temporal_k5_convlstm_ring3_3x
G5_POSW4=$A10/geo_k5_convlstm_base
G5_NEG1_R3=$A10/geo_k5_convlstm_ring3
G5_NEG1_R10=$A10/geo_k5_convlstm_ring10
G5_NEG3_R3=$A10/geo_k5_convlstm_ring3_3x
G10_NEG1_R3=$A10/geo_k10_convlstm_ring3
TA_FUSE0=$A10/temporal_k5_tattn_base
TA_FUSE0_NEG=$A10/temporal_k5_tattn_ring3
TA_FUSE2_NEG=$A10/temporal_k5_tattn_fuse2_ring3
TA_HYBRID_NEG=$A10/temporal_k5_tattn_hybrid_ring3

# The 2026-08-11 batch: the two geo attention arms (LSF 493314/493315) and the
# two single-frame controls (LSF 501433/501434). All four finished cleanly --
# three on early-stopping patience, one on the full 60 epochs, none killed --
# and all four kept best.pt. Moved into outputs/2026-08-11/ on 2026-08-17.
#
# All four now have object-level scores (docs/PREDICTIONS.md): the hybrid ties
# ConvLSTM on geo and pure attention trails it, so best.pt is still worth
# keeping on all four, but the eval4 backlog they existed for is closed.
A11=outputs/2026-08-11
G5_TATTN_FUSE0=$A11/geo_k5_tattn_ring3
G5_TATTN_HYBRID=$A11/geo_k5_tattn_hybrid_ring3
G5_SINGLE_BASE=$A11/geo_k5_single_base
G5_SINGLE_NEG=$A11/geo_k5_single_ring3

# The 2026-08-06 geo_k10 reference: the project's worst over-predictor
# (P=0.121 @ 0.125) and therefore the widest gap between what a model finds and
# what it flags. Kept for the positives-only table below.
G10_OLD=outputs/2026-08-06/geo_k10_convlstm_posw8

# ---- the negative-sampling batch --------------------------------------------
# Every model to date trained on positive patches only (--nonz_only), i.e. the
# ~2.5% of the patch grid that contains subsidence, and was then applied to
# 100% of the map at inference. That is what the object-level evals exposed:
# geo_k10 scores P=0.12 at confidence 0.125, and outputs.py:22-25 predicts
# exactly this ("models trained with --nonz_only ... over-predict at scene
# scale"). --add_ring_negatives puts background patches back into training.
#
# ONE FACTOR PER RUN, and the whole batch is pinned at POS_W=4. That is the
# value the object-level evals picked out: pos_w 8->4 was worth +0.023 F1 on 18
# of 20 test scenes, the best-established result in the project. Holding it
# fixed everywhere also lets the negatives arms be read across partitions
# without pos_w shifting underneath them.
#
# The cost of pinning it: the older runs sit at pos_w 2 (temporal_k5) and pos_w
# 8 (geo_k5), so neither was a valid reference. Their pos_w=4 replacements have
# now been submitted and are intentionally absent from the remaining job table.
#
#   temporal_k5  ->  convlstm_temporal_k5_h256_posw4_60e   (submitted separately)
#   geo_k5       ->  convlstm_geo_k5_h256_posw4_60e        (submitted separately)
#   geo_k10      ->  $G10_POSW4                            (already trained)
#
# geo_k10 needs no new baseline: it was trained at pos_w 4 on 2026-08-09 and
# holds the highest val dice in the project (0.6646).
T5_BASE="PARTITION=$TEMPORAL_K5 K_PREVS=5 POS_W=4"
G5_BASE="PARTITION=$GEO_K5 K_PREVS=5 POS_W=4"
G10_BASE="PARTITION=assets/partition_geo_k10.json K_PREVS=10 POS_W=4"

# The three negative-sampling arms:
#   ring3_1x   near-field hard negatives, balanced 1:1  -- the basic effect
#   ring10_1x  same count, drawn from much further out  -- near vs far field
#   ring3_3x   near-field, 3 negatives per positive     -- how much is enough
NEG_R3_1X="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
NEG_R10_1X="RING_NEGS=yes NEG_RING_OUTER=10 NEG_PER_POS=1.0"
NEG_R3_3X="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=3.0"

# VRAM ladder: 36G standard, 48G for the k=10 jobs. Lowered from 48/64 on
# 2026-08-10, when seven jobs sat PEND waiting for cards big enough to satisfy
# a request nothing had ever needed.
#
# Where 36 and 48 come from. Activation memory is dominated by the encoder,
# which runs on B*T frames, and the per-frame feature maps are exactly
# 2,467,328 elements (64x200x100 + 128x100x50 + 256x50x25 + 512x25x12 +
# 1024x12x6). At batch 128 in fp16, assuming DoubleConv keeps ~4 tensors per
# output for backward, plus the decoder at B (not B*T) and fp32
# param/grad/RMSprop state:
#
#            encoder   decoder   optimizer   total   +40%
#   k=5       14.1 GiB   2.4 GiB    0.6 GiB   17.1    24.0   -> 36G, 50% margin
#   k=10      25.9 GiB   2.4 GiB    0.6 GiB   28.9    40.4   -> 48G, 19% margin
#
# So k=10 is the reason the second rung still exists: at 36G it would be
# running BELOW its estimated peak. Do not collapse the two rungs into one.
# The k=10 margin is the thin one -- if anything OOMs, it is that job, and the
# fix is to put it back on 64G rather than to raise everything.
#
# Ring negatives do not move VRAM either way -- it is set by batch size and
# model, and negatives add SAMPLES, not per-batch memory -- so each negatives
# run sits on the same rung as the baseline it is paired with. The temporal
# attention model is SMALLER than the ConvLSTM (32.1 M vs 43.1 M) and its
# fused skips cost well under a GB at k=5, so it shares the k=5 rung.
#
#                queue     mem  vram  wall     (measured peak -> request)
RES_K10="long-gpu   96   48  10:00"   # ConvLSTM k=10, peak ~65 GB host -- the big one
RES_K5="long-gpu    64   36  10:00"   # ConvLSTM k=5,  peak ~41 GB host
RES_STACK="long-gpu 96   36  10:00"   # stacked U-Net k=10, peak ~66 GB host, no recurrent state
RES_SINGLE="long-gpu 24  36   6:00"   # single-frame, peak ~6 GB host
# RES_K10/RES_STACK/RES_SINGLE are unused by the current table -- kept because
# they carry the measured host-memory peaks for job types that will come back.
# Eval sat on short-gpu until 2026-08-09, when every eval submission bounced:
#   "RUNLIMIT: Cannot exceed queue's hard limit(s). Job not submitted."
# short-gpu's hard walltime cap is under 8:00. The measured eval is 3.0 h, so a
# short-gpu slot would need W well below that cap and leaves little headroom;
# long-gpu accepts 10:00 and is proven. To move back, find the real cap with
#   bqueues -l short-gpu | grep -i runlimit
# and set W under it.
RES_EVAL="long-gpu  80   36   8:00"   # eval, peak ~57 GB host (was 64 -> 87% full)

# ---- sizing the negative-sampling runs --------------------------------------
# Ring negatives ADD samples, so both memory and walltime move. Peak host RAM
# splits into two parts, which scale differently:
#
#   persistent  the dataset's own (T, N, H, W) float32 store, kept for the whole
#               run:  N x T x 200 x 100 x 4 B  (+ the (N, H, W) target)
#               ~= N x 560 KB at T=6. This is the part negatives multiply.
#   transient   one interferogram's full patch grids while it is being indexed,
#               freed before the next: 6 frames x 2 (image+mask) x ~17,088
#               patches x 20,000 px x 4 B ~= 16 GB. Independent of N.
#
# Checked against measurement: temporal_k5 (38,705 samples) predicts 21.7 + 16
# = 38 GB and measured 42.7; geo_k5 (45,458) predicts 25.5 + 16 = 42 GB and
# measured 43.4. Good enough to extrapolate, with the usual ~40% on top.
#
#   1x negatives -> persistent doubles  -> ~60-67 GB  -> request 88 GB
#   3x negatives -> persistent x4       -> ~103-118 GB -> request 144-160 GB
#
# Walltime scales with sample count directly: measured 159 s/epoch on
# temporal_k5 and 243 s/epoch on geo_k5, so 60 epochs at 3x negatives is 10.6 h
# and 16.2 h respectively.
#
# WARNING: geo_k5 at 3x asks for 24:00, which is above the 14:00 the templates
# carry and may exceed long-gpu's hard cap -- if it bounces with
# "RUNLIMIT: Cannot exceed queue's hard limit(s)", either drop that arm to
# NEG_PER_POS=2.0 or let it hit the wall and resume: the run directory keeps
# resume.pt, so `RESUME=<run dir> bsub -J <same name> < train_convlstm.sh`
# continues from the last completed epoch ('auto' will NOT do it -- a
# runlimit kill is not a requeue and the new job gets a new id).
#
#                     queue     mem  vram  wall
RES_NEG1_T5="long-gpu   88   36  10:00"   # temporal_k5 + 1x negatives, ~5.3 h
RES_NEG1_G5="long-gpu   88   36  14:00"   # geo_k5      + 1x negatives, ~8.1 h
RES_NEG3_T5="long-gpu  144   36  16:00"   # temporal_k5 + 3x negatives, ~10.6 h
RES_NEG3_G5="long-gpu  160   36  24:00"   # geo_k5      + 3x negatives, ~16.2 h  <-- may bounce
# geo_k10 carries T=11, not T=6, so both terms roughly double: persistent
# 38,287 x 11 frames = 36.8 GB, transient grid 30.1 GB, predicted 67 GB against
# 69.1 GB measured on lsf594076. With 1x negatives: 73.6 + 30 = ~104 GB. Its
# measured 262 s/epoch doubles to ~8.7 h over 60 epochs.
RES_NEG1_G10="long-gpu 144   48  14:00"   # geo_k10     + 1x negatives, ~8.7 h
RES_RESCORE="short-gpu 16    8   2:00"    # stage-2 re-scoring only, no inference
# Positives-only: the patch stage is trivial (~4,400 positive patches over the
# 18 geo test scenes, against 5,846 in a single validation epoch), and the scene
# stage runs the network on ~2.5% of the tiles. What does NOT shrink is loading
# and normalising every full patch grid, which is the transient ~30 GB term and
# a good share of the 3.0 h a normal eval takes. So host memory stays at the
# eval rung and only walltime comes down -- 6:00 is roughly 2x the I/O floor.
# Measure the first one and tighten:  grep -A3 "Max Memory" logs/*.out
RES_POSONLY="long-gpu 80    36   6:00"

NAMES=(); KINDS=(); TEMPLATES=(); OVERRIDES=(); RESOURCES=(); WHYS=()
job() { KINDS+=("$1"); NAMES+=("$2"); TEMPLATES+=("$3"); OVERRIDES+=("$4"); RESOURCES+=("$5"); WHYS+=("$6"); }

# ---- training: the negative-sampling batch (DONE 2026-08-10) -----------------
# All seven arms plus both no-negatives baselines finished on 2026-08-10 and are
# archived under $A10 with their metrics and best.pt. They are deliberately NO
# LONGER LISTED as jobs: leaving them here meant `submit_all.sh train --submit`
# silently retrained ~60 GPU-hours of finished work. The variables T5_BASE /
# G5_BASE / G10_BASE / NEG_R* above are kept because the eval table and any
# reruns still need them, and because they document what was trained.
#
# What they produced, and how to read it -- val/dice CANNOT see what the
# negatives change. Ring negatives go to the TRAIN split only; validation stays
# positives-only, so suppressing scene-scale false positives has almost no
# upside on that curve. That prediction held exactly: all nine land inside
# 0.639-0.660, i.e. at or under the +-0.006 noise floor, with the geo arms
# slightly DOWN. Nothing was learned from results.csv and nothing was supposed
# to be. The verdict comes from the eval table below.
#
#   temporal_k5  base 0.6391 | ring3 0.6460 | ring10 0.6432 | neg3x 0.6449
#   geo_k5       base 0.6558 | ring3 0.6541 | ring10 0.6461 | neg3x 0.6386
#   geo_k10      base 0.6646 (2026-08-09)   | ring3 0.6596

# ---- evaluation: the 2026-08-09 backlog -------------------------------------
# All six runs of 2026-08-09 were scored on val/dice only and have no
# object-level numbers. Their patch dice is known to mis-rank across
# architecture families (the single-frame baseline placed last on dice and tied
# the ConvLSTMs on object F1), so these are what actually settle that batch.
#
# Scene lists are chosen to PAIR with the evals already in
# outputs/predictions/: the temporal runs go on temporal_k10's 20-scene test
# list (the shared list -- K_PREVS=5 against a _k10 group is expected here and
# run_eval.sh only notes it), the geo runs on geo_k10's 18.
job eval eval_temporal_k5_posw2 \
    scripts/eval/run_eval.sh "RUN=$T5_POSW2 GROUP=temporal_k10 K_PREVS=5" "$RES_EVAL" \
    "group-T val-dice leader (0.6496) -- does it hold up at object level?"
job eval eval_temporal_k5_posw1 \
    scripts/eval/run_eval.sh "RUN=$T5_POSW1 GROUP=temporal_k10 K_PREVS=5" "$RES_EVAL" \
    "far end of the pos_w sweep; biggest precision shift of any run (P 0.674->0.754)"
job eval eval_geo_k10_posw4 \
    scripts/eval/run_eval.sh "RUN=$G10_POSW4 GROUP=geo_k10 K_PREVS=10" "$RES_EVAL" \
    "highest val dice in the project (0.6646); geo is where over-prediction is worst"
# REMOVED 2026-08-11: eval_geo_k10_seed7. Its weights were pruned as a probe
# checkpoint, so this eval can no longer run.
#
#   job eval eval_geo_k10_seed7 \
#       scripts/eval/run_eval.sh "RUN=$G10_SEED7 GROUP=geo_k10 K_PREVS=10" "$RES_EVAL" \
#       "the G10 error bar, at object level -- how big is the noise floor on THIS metric?"
#
# WHAT THIS COSTS: G10's object-level noise floor is now unmeasurable without a
# retrain (~4.4 h). It was never measured, and docs/RESULTS.md flags it as
# needed before any small geo gap is believed -- which matters more now that geo
# is the main line. To restore the question, retrain seed 7 with
#   $G10_BASE SEED=7   (train_convlstm.sh, RES_NEG1_G10)
# and re-add the job above. $G10_SEED7 still holds its metrics, so the
# PATCH-level floor is intact; it is only the object-level one that is gone.
job eval eval_geo_k10_single \
    scripts/eval/run_eval.sh "RUN=$G10_SINGLE GROUP=geo_k10 ARCH=single" "$RES_EVAL" \
    "single-frame at matched b128/lr1e-6: the de-confounded temporal-gain test"
job eval eval_geo_k10_stack \
    scripts/eval/run_eval.sh "RUN=$G10_STACK GROUP=geo_k10 K_PREVS=10 ARCH=stack" "$RES_EVAL" \
    "channel-stacking under the domain shift, at object level"

# ---- rescore: fix the truncated threshold sweeps ----------------------------
# One eval was scored at 0.125/0.25/0.5 only, before THRESHOLDS grew 0.7 and
# 0.9. It is cut off below its own operating point -- geo_k10 is still at
# P=0.31 and climbing at its last point -- so its best F1 is unknown and it
# cannot be compared with the 5-point results. The confidence maps are still on
# disk, so this is stage 2 only: minutes, not another 3 h of inference.
#
# Pruning *_image.npy on 2026-08-11 does NOT affect this: rescore reads
# _pred.npy and _gt.npy, both kept. Verified by re-scoring a pruned directory
# to byte-identical numbers.
RESCORE_G10=outputs/predictions/geo_k10_convlstm_posw8/best/scenes_geo
job rescore rescore_geo_k10_206375 \
    scripts/eval/rescore.sh "DIR=$RESCORE_G10" "$RES_RESCORE" \
    "the 0.6583 geo leader, currently unrankable -- 3 of 5 thresholds missing"
# REMOVED 2026-08-11: rescore_k10split_10prev. Its eval directory was deleted
# with the invalid union-bug model (outputs/2026-08-03/README.md). Nothing to
# re-score, and the numbers would not have been trustworthy anyway.

# ---- temporal attention (kind: tattn) ---------------------------------------
# Its own kind on purpose, so `submit_all.sh train --submit` keeps meaning the
# negative-sampling batch above and these do not ride along with it.
#
# All four are read against convlstm_temporal_k5_h256_posw4_60e -- same
# partition, same pos_w, same optimizer. That baseline was already submitted.
# Host RAM is set by the dataset, not the model, so temporal_k5 sits on the
# same rung as its ConvLSTM sibling; the attention model is 26% SMALLER
# (32.1 M vs 43.1 M) and its temporal block is ~20x cheaper in MACs, so if
# anything these finish sooner.
#
# val/dice will probably not separate them: the top six models already span
# 0.0069 against a +-0.006 floor. The mechanism under test is temporal
# consistency -- artefacts flicker, subsidence persists -- which is a PRECISION
# claim. Score with scripts/eval/run_eval.sh.
TATTN_T5="PARTITION=$TEMPORAL_K5 K_PREVS=5 POS_W=4"

# Every arm here has a ConvLSTM twin in the training table above that differs
# ONLY in the architecture, which is the whole point -- same partition, same
# pos_w 4, same optimizer, same negative sampling:
#
#   tattn_..._fuse0_posw4_60e             <-> convlstm_temporal_k5_h256_posw4_60e
#   tattn_..._fuse0_..._neg1x_ring3_60e   <-> convlstm_temporal_k5_h256_posw4_neg1x_ring3_60e
#
# The no-negatives ConvLSTM baseline was submitted separately; its negatives
# twin remains in the training table above.
# All four finished on 2026-08-10 and are archived under $A10; like the
# negatives batch they are no longer listed as jobs. Their val/dice came in
# exactly as predicted -- inside the noise floor, top to bottom:
#
#   fuse0 no-negs 0.6460 | fuse0 +negs 0.6459 | fuse2 +negs 0.6399 | hybrid 0.6468
#
# The head-to-head under negatives is 0.6459 against the ConvLSTM's 0.6460: a
# dead tie for a model 26% smaller, which is the result the plan called for.
# The one delta pointing anywhere is fuse2 at -0.0060 against fuse0 -- the
# wrong direction, so fuse4 should NOT be submitted (PRESETS.md makes it
# conditional on fuse2 beating fuse0).

# ---- attention on the GEO split (kind: tattn, submitted 2026-08-11) ----------
# The whole attention experiment so far lives on temporal_k5, where it tied the
# ConvLSTM exactly (0.6459 vs 0.6460). These two move it onto geo -- train on
# the north, predict the south -- which is now the main line (docs/RESULTS.md).
#
# WHY IT MIGHT COME OUT DIFFERENTLY, and why it is worth the GPU: the two
# partitions fail in opposite directions. Temporal under-predicts on future
# scenes (object R 0.65 / P 0.51); geo over-predicts badly on unseen ground
# (R 0.90 / P 0.32 at geo_k5, R 0.96 / P 0.18 at geo_k10). Attention's claim is
# a PRECISION claim -- artefacts flicker between frames, real subsidence
# persists, so a weighted average over the chain suppresses one and keeps the
# other. A tie on the partition that already has decent precision says much
# less than a win on the one that does not. Temporal was the wrong place to
# test it.
#
# Both have an exact ConvLSTM twin already trained on 2026-08-10, differing
# ONLY in the architecture -- same partition, same pos_w 4, same 1:1
# near-field negatives, same optimizer:
#
#   tattn_geo_k5_..._fuse0_..._neg1x_ring3   <-> $G5_NEG1_R3  (dice 0.6541)
#   tattn_geo_k5_..._recur-convlstm_...      <-> $G5_NEG1_R3  (the hybrid)
#
# FUSE_SKIPS stays 0: it is the ConvLSTM's exact skip contract, so architecture
# is the only variable. fuse2 is deliberately absent -- it lost 0.0060 on
# temporal and nothing downstream depends on it.
TATTN_G5="PARTITION=$GEO_K5 K_PREVS=5 POS_W=4 FUSE_SKIPS=0"

# Sizing, from measured per-epoch times at neg1x_ring3 (see the 2026-08-10
# logs). Host RAM is set by the DATASET, so geo_k5 + 1x negatives sits on the
# same 88 GB rung as its ConvLSTM twin, which ran there successfully. VRAM is
# 25.6 GiB reserved for attention against the ConvLSTM's 25.8, so 36G holds.
#
# Walltime is the part that needed changing. Per epoch:
#     temporal_k5 + neg1x   convlstm 5m06s   tattn 8m13s   hybrid 9m02s
#     geo_k5      + neg1x   convlstm 9m08s   -> geo runs 1.79x the temporal
# Carrying that factor across: tattn ~14m43s/epoch and the hybrid ~16m10s,
# i.e. 14.7 h and 16.2 h for a full 60 epochs. RES_NEG1_G5's 14:00 would kill
# both before they finish, and a k5 run has already been lost once to
# TERM_RUNLIMIT. 20:00 covers the worst case; the 24:00 of RES_NEG3_G5 was
# accepted by long-gpu, so this is inside the queue's cap. Early stopping has
# fired at 44-56 epochs on every run so far, so expect ~11-15 h in practice.
RES_TATTN_G5="long-gpu  88   36  20:00"

# Negatives are pinned at 1:1 near-field for both, matching $G5_NEG1_R3 exactly.
# A 2:1 arm was considered and dropped: NEG_PER_POS is a request capped by
# annulus availability, and geo_k5's "3x" arm only ever reached 2.42x, so 2.0x
# would sit ~83% of the way to the ratio that already lost 0.017 dice -- while
# leaving no ConvLSTM twin at its own ratio, which would confound architecture
# with negative count. Hold the negatives fixed; vary only the architecture.
# DONE 2026-08-11: both were submitted at 14:15 as LSF 493314 and 493315 and
# both FINISHED -- fuse0 best @16 / stopped @36 (dice 0.6433), the hybrid best
# @22 / stopped @42 (dice 0.6487). RETIRED from the job table on 2026-08-12,
# following the same rule as the 2026-08-10 batch: a finished run that is still
# listed is a ~15 h retrain waiting for one careless `tattn --submit`.
#
# Neither dice number ranks them -- see the eval4 table, which is what settles
# this pairing. To resubmit after a failure, uncomment:
#
#   job tattn tattn_geo_k5_d256_fuse0_posw4_neg1x_ring3_60e \
#       scripts/train/train_tattn.sh "$TATTN_G5 $NEG_R3_1X" "$RES_TATTN_G5" \
#       "attention vs recurrence on the GEO split"
#   job tattn tattn_geo_k5_d256_fuse0_recur-convlstm_posw4_neg1x_ring3_60e \
#       scripts/train/train_tattn.sh "$TATTN_G5 RECURRENCE=convlstm $NEG_R3_1X" "$RES_TATTN_G5" \
#       "the hybrid on geo: ConvLSTM first, attention over ALL its hidden states"

# ---- the U-Net control: does the ARCHITECTURE matter at all? (kind: control) --
# Ring negatives bought the geo_k5 ConvLSTM +0.096 object F1 (0.452 -> 0.548 at
# conf 0.25, measured 2026-08-11). Nobody knows whether that is a ConvLSTM
# effect or something ANY model gets from seeing background. These two close a
# 2x2 on geo_k5 at pos_w 4 whose other half is already trained AND scored:
#
#                    no negatives      + ring negatives 1:1
#   single-frame     <-- these two jobs -->
#   ConvLSTM         F1 0.452           F1 0.548
#
# If single-frame + negatives lands near the ConvLSTM's 0.548, the temporal
# machinery -- ConvLSTM, attention, the hybrid -- is buying much less than
# assumed, and that is worth knowing before more GPU goes into architectures.
# These are also the cheapest jobs in the file: single-frame ran 60 epochs in
# 2h11m on geo_k5.
#
# Neither existing single-frame baseline can serve. Both sit at pos_w 8 (the
# whole current line is pinned at 4, worth +0.023 F1 on its own) and the geo_k5
# one is also at batch 64 / lr 1e-5. Matched settings here: b128, lr 1e-6,
# pos_w 4, K_PREVS=5.
#
# K_PREVS=5 on a single-frame run adds no input frames -- it names the chain the
# negatives' exclusion grid is built from, which is what makes these draw the
# SAME negatives as $G5_NEG1_R3 (see dataset.py::_load_single_ring).
#
# Sizing: measured 33 GB host and ~2m11s/epoch without negatives. Negatives
# double the sample count, so ~4.5 h and a larger persistent store; the loader
# also switches from the pre-extracted nonz files to full grids plus the chain's
# mask grids, ~10 GB transient per interferogram. VRAM is a plain U-Net on one
# input channel, far under the ConvLSTM's measured 25.8 GiB, so 24G schedules on
# more cards; the run logs its own peak, so replace this with measurement.
CONTROL_G5="ARCH=single PARTITION=$GEO_K5 K_PREVS=5 POS_W=4"
RES_CTRL_G5="long-gpu   48   24   6:00"
RES_CTRL_G5_NEG="long-gpu 64   24  10:00"

# DONE 2026-08-11: submitted at 14:36 as LSF 501433 and 501434, both finished
# -- the no-negatives corner ran all 60 epochs (dice 0.6434) and the negatives
# corner stopped @40 (dice 0.6341). RETIRED from the job table on 2026-08-12
# for the same reason as the tattn arms above.
#
# The no-negatives run has ALREADY paid for itself on dice alone: at matched
# b128/lr1e-6/pos_w4 the single-frame/ConvLSTM gap on geo_k5 is +0.0124, not the
# +0.046 the confounded 2026-08-05 baseline showed, and geo_k10 independently
# agrees at +0.0133 (docs/RESULTS.md finding 3). The negatives corner is the one
# still open -- dice went DOWN 0.0093, which is precisely the signal dice cannot
# read, so it needs eval4 before the 2x2 means anything.
#
# To resubmit after a failure, uncomment:
#
#   job control unet_single_geo_k5_posw4_60e \
#       scripts/train/train_control.sh "$CONTROL_G5" "$RES_CTRL_G5" \
#       "the no-negatives corner of the 2x2, at matched b128/lr1e-6/pos_w4"
#   job control unet_single_geo_k5_posw4_neg1x_ring3_60e \
#       scripts/train/train_control.sh "$CONTROL_G5 $NEG_R3_1X" "$RES_CTRL_G5_NEG" \
#       "THE control: single-frame with the same negatives \$G5_NEG1_R3 got"

# ---- the geo_k5 reruns under negative validation (kind: valneg) -------------
# RETIRED 2026-08-20: NEGATIVE VALIDATION IS NO LONGER SUBMITTABLE FROM HERE.
# Validation is positives-only for every job in this file. $VALNEG is gone and
# the five jobs at the end of this section are commented out, so no new run can
# turn negative validation on. The reasoning below is kept as the record of why
# it was tried and what it cost -- read it as history, not as instructions.
#
# WHAT IT COST. The fix worked as designed and bought a metric nobody could
# read across the boundary: half of a post-2026-08-18 val/dice is a
# negative-patch-cleanliness term the older numbers do not contain, so the
# clean22 batch cannot be ranked against anything trained before it. The
# object-level eval already answers the question negative validation was
# introduced to answer, and it answers it on scenes rather than on a sampled
# annulus. See docs/RESULTS.md and docs/MODEL_RUNS.md.
#
# The `--add_val_negatives` flag itself still exists in the trainer, DEPRECATED,
# for one reason only: the five attnfix runs of 2026-08-19 were trained with it
# and their resume.pt fingerprints carry `valneg=1-3x1.0`, a STRICT resume key.
# Removing the flag would make those runs unresumable. It goes when they land.
#
# WHAT WAS BROKEN. Every negative-sampling result in this file was produced
# against a val set of 5846 PURELY POSITIVE patches. Negatives went to the train
# split only, so the false positives they exist to suppress were almost entirely
# outside the thing being measured, and val/dice was structurally unable to rank
# the arms. It did not: all nine arms of 2026-08-10 landed in 0.639-0.660,
# inside the +-0.006 noise floor, and the geo arms came out slightly DOWN while
# object-level F1 on the same models moved +0.096. Every verdict on this line
# has therefore had to come from run_eval.sh at ~3 h a scene list.
#
# WHAT CHANGES. --add_val_negatives (VAL_NEGS=yes) puts negatives in the
# VALIDATION set too, so a false positive on background costs dice. Its
# configuration is FIXED in the code -- ring 1..3, 1:1, drawn once from the
# validation interferograms with SEED, unchanged across every epoch -- and is
# deliberately not exposed as a knob. Partition + seed alone decide the samples,
# so all five runs below are scored on the SAME ~11,700-patch val set whatever
# their architecture or training-negative ratio. LR scheduling, early stopping
# and best.pt still key off val/dice; it is only the set underneath that grew.
#
# WHY RERUN RATHER THAN RESCORE. val/dice selects best.pt and drives the plateau
# schedule and early stopping. A model whose checkpoints were CHOSEN on a
# positives-only curve is not the model this val set would have selected, so
# re-scoring the 2026-08-10/11 weights would answer a different question. These
# five reruns are the same five experiments end to end.
#
# ONE FACTOR CHANGES. Architecture, optimizer, batch, LR, pos_w, partition,
# seed, patience, epochs and the TRAINING negatives are all copied verbatim from
# the runs named below -- verified against each run's own banner in
# outputs/.../logs/reporter.log, not from memory. The only difference is the
# validation set.
#
#   convlstm ..._neg1x_ring3_valneg1x   <-> $G5_NEG1_R3        (dice 0.6541 @30)
#   convlstm ..._neg3x_ring3_valneg1x   <-> $G5_NEG3_R3        (dice 0.6386 @30)
#   convlstm ..._neg1x_ring10_valneg1x  <-> $G5_NEG1_R10       (dice 0.6461 @24)
#   tattn    ..._fuse0_..._valneg1x     <-> $G5_TATTN_FUSE0    (dice 0.6433 @16)
#   unet     single_..._valneg1x        <-> $G5_SINGLE_NEG     (dice 0.6341 @20)
#
# Those five numbers span 0.0200 and are the ones that cannot be trusted to rank
# anything. The reruns' numbers can be compared with each other, and with
# nothing trained before them: a positives-only val set and a 1:1 one are
# different scales, so do not read a rerun against the figure beside it above.
# VALNEG="VAL_NEGS=yes" -- REMOVED 2026-08-20. It was appended to the override
# string of all 24 training jobs below (valneg, clean22 and attnfix); every one
# of them now trains against a positives-only validation set. Do not reinstate
# it: put background patches in TRAINING with $NEG_R3_1X and settle precision
# claims with scripts/eval/run_eval.sh at object level.

# Sizing. Negatives DOUBLE the validation set, and validation runs at batch size
# 1 (train.py's val_loader), which makes it a much larger share of an epoch than
# its sample count suggests. Decomposed from the two single-frame runs, which
# differ only in training samples (45,487 vs 90,916 at a shared 5,846 val):
#   131 s and 182 s per epoch  ->  ~0.0011 s per train sample, ~0.014 s per val
#   sample, i.e. ~80 s of that 131 s epoch was VALIDATION.
# The same decomposition over the two ConvLSTM negatives arms (90,916 vs 155,252
# train) gives ~0.0055 s/train sample and ~52 s of validation per epoch. So:
#
#            measured/epoch   + doubled val   60 epochs   request
#   convlstm 1x   9m08s          ~10m00s        10.0 h    14:00  (RES_NEG1_G5)
#   convlstm 1x r10 8m32s        ~9m24s          9.4 h    14:00  (RES_NEG1_G5)
#   convlstm 3x   14m59s         ~15m51s        15.9 h    24:00  (RES_NEG3_G5)
#   tattn         5m46s          ~6m38s          6.6 h    12:00  (below)
#   unet single   3m02s          ~4m22s          4.4 h    10:00  (RES_CTRL_G5_NEG)
#
# Every existing rung holds, so they are reused unchanged -- except tattn's.
# RES_TATTN_G5 asks 20:00 on an estimate of 14m43s/epoch that the run itself
# disproved: it measured 5m46s and finished 36 epochs in 3h27m. 12:00 is ~1.8x
# the worst case here and stops that job blocking a long-gpu slot it cannot use.
# Host memory: the extra ~5,846 val samples cost 6 frames x 20,000 px x 4 B each
# = ~3.3 GB for a k5 temporal run, ~0.5 GB single-frame. Inside every rung.
# VRAM is unmoved -- measured 25.8 GiB reserved (ConvLSTM), 25.6 (tattn), 8.3
# (single-frame); negatives add SAMPLES, not per-batch memory.
RES_TATTN_G5_VN="long-gpu  88   36  12:00"

# Submit order is deliberate: the 1:1 near-field ConvLSTM arm first. It is the
# reference every other arm is read against, and until it exists the rest are
# individually uninterpretable -- exactly the mistake the 2026-08-10 batch made
# by queueing arms before their baseline.
# job valneg convlstm_geo_k5_h256_posw4_neg1x_ring3_valneg1x_60e \
#     scripts/train/train_convlstm.sh "$G5_BASE $NEG_R3_1X" "$RES_NEG1_G5" \
#     "THE reference arm: 1:1 near-field negatives, now scored on a val set that can see them"
# job valneg convlstm_geo_k5_h256_posw4_neg3x_ring3_valneg1x_60e \
#     scripts/train/train_convlstm.sh "$G5_BASE $NEG_R3_3X" "$RES_NEG3_G5" \
#     "how much is enough: 3:1 against 1:1, on a metric that can tell them apart"
# job valneg convlstm_geo_k5_h256_posw4_neg1x_ring10_valneg1x_60e \
#     scripts/train/train_convlstm.sh "$G5_BASE $NEG_R10_1X" "$RES_NEG1_G5" \
#     "near vs far field: same count, drawn from a 10-cell annulus"
# job valneg tattn_geo_k5_d256_fuse0_posw4_neg1x_ring3_valneg1x_60e \
#     scripts/train/train_tattn.sh "$TATTN_G5 $NEG_R3_1X" "$RES_TATTN_G5_VN" \
#     "attention vs recurrence on geo -- a precision claim, finally on a precision-sensitive metric"
# job valneg unet_single_geo_k5_posw4_neg1x_ring3_valneg1x_60e \
#     scripts/train/train_control.sh "$CONTROL_G5 $NEG_R3_1X" "$RES_CTRL_G5_NEG" \
#     "the control: does the temporal machinery buy anything once background is scored?"

# ---- training: the clean benchmark (kind: clean22) --------------------------
# The six runs of docs/PLAN_CLEAN_BENCHMARK.md section 6, on the generation-3
# partitions written 2026-08-18 (assets/partition_*_clean.json). These are the
# first runs on the new ground rules and NOTHING from the old batches carries
# over -- different scene lists, an AOI window, and a latitude cut that makes
# the geo train and hold-out disjoint for the first time.
#
# EVERY RUN HAS NEGATIVES, in TRAINING:
#   RING_NEGS=yes   background patches in TRAINING  (ring 1-3, the arm that won
#                   on the old data), so the model sees what it must not flag.
#
# AS ORIGINALLY SUBMITTED, all fourteen also carried VAL_NEGS=yes, which put
# background patches in VALIDATION as well. That is why their val/dice reads
# 0.72-0.80 against the 0.63-0.66 of every earlier batch: an empty prediction
# on an empty mask scores dice 1.0, so at 1:1 the mean is roughly
# (1 + dice_on_positives)/2. The numbers in outputs/ are what those runs
# actually scored and are kept as they are -- but the flag was removed on
# 2026-08-20, so RESUBMITTING ANY JOB BELOW NOW TRAINS AGAINST A
# POSITIVES-ONLY VAL SET and its curve will not line up with the recorded one.
# Start such a rerun under a new --job_name rather than resuming.
#
# The partitions carry their own aoi_window, so no AOI flag is needed here --
# train.py reads it from the file and hands it to the datasets. That is the
# point of putting the window in the partition rather than on the command line.
#
# HOST MEMORY: every request is now sized from a measurement, capped at 128 GB.
# SubsiDataset holds every loaded patch in RAM, so ring negatives add SAMPLES
# and therefore add memory -- 1:1 doubles the stored set, 3:1 quadruples it.
#
# Calibration: the old geo_k5 train pool is 109 intfs / 45,487 positives and its
# measured host peak was ~41 GB positives-only, which is 1.61x the raw sample
# bytes (T x 200 x 100 x 4, plus the mask). Estimated peaks, +25% headroom:
#
#   geo_k5   1:1   59G -> 80     geo_k10  1:1   85G -> 112
#   geo_k5   3:1  117G -> 128    temp_k5  1:1   77G -> 96
#   geo_k5   single 17G -> 24    temp_k10 1:1  101G -> 128
#
# THE 3:1 ARM IS THE ONLY ONE NEAR THE CAP: 117G estimated against a 128G
# request is ~9% headroom, where everything else has 25%. If it OOMs, drop
# NEG_PER_POS to 2.0 (est. 88G) rather than raising the request -- "more than
# 1:1" was the question that arm asks, not "exactly 3:1".
#
# WALLTIMES were re-scaled for EPOCHS=100 (x1.6 over the 60-epoch figures).
#
# SEVERAL NOW EXCEED 24:00, which scripts/submit_all.sh already flags as
# possibly above long-gpu's hard cap. That is a cheap failure: an over-long
# request is REJECTED AT SUBMIT TIME with
#   "RUNLIMIT: Cannot exceed queue's hard limit(s)"
# so nothing is wasted and the real cap becomes known immediately. Asking too
# LITTLE is the expensive mistake -- the job dies mid-run at the wall.
#
# If a job bounces, either lower its -W here, or let it run to the wall and
# continue from resume.pt:
#     RESUME=<run dir> bsub -J <same name> < scripts/train/train_convlstm.sh
# RESUME=auto will NOT do it: a runlimit kill is not a requeue, and the new job
# gets a new id.
#
# PATIENCE=40 is what actually decides these runs. The 2026-08-10 batch stopped
# every one of its 13 jobs on patience rather than the epoch budget, and at
# LR=1e-5 the loss should move earlier still -- so 100 epochs is a ceiling most
# runs will not reach.
#
# Resources are re-sized, not inherited: the clean training pools are LARGER
# than the ones the old walltimes were measured on (temporal_k5 goes 106 -> 163
# interferograms, geo_k5 109 -> 127), and 1:1 negatives double the sample count
# on top of that. Walltimes below carry roughly 1.6x headroom over a linear
# scaling of the 2026-08-10 measurements; host memory tracks FRAMES per sample,
# so the k10 arms keep the 144 GB that geo_k10+negatives needed.
GEO_K5_CLEAN=assets/partition_geo_k5_clean.json
GEO_K10_CLEAN=assets/partition_geo_k10_clean.json
TEMP_K5_CLEAN=assets/partition_temporal_k5_clean.json
TEMP_K10_CLEAN=assets/partition_temporal_k10_clean.json

# Shared hyperparameters for the whole batch, set 2026-08-18. Every clean22 job
# carries $C_HYP so the fourteen runs differ in partition, architecture and
# negatives -- and in nothing else. Changing a value here changes all fourteen,
# which is the point: a hyperparameter that drifts between arms makes the
# architecture comparison uninterpretable.
#
#   LR 1e-5     10x the 1e-6 every previous run used. Expect the loss to move
#               much earlier in training; the old curves are not a guide.
#   EPOCHS 100  a ceiling, not a target -- patience decides in practice.
#   PATIENCE 40 doubled with the epoch budget so early stopping stays a
#               proportional rule rather than becoming twice as aggressive.
C_HYP="LR=1e-5 EPOCHS=100 PATIENCE=40"

C_G5="PARTITION=$GEO_K5_CLEAN   K_PREVS=5  POS_W=4"
C_G10="PARTITION=$GEO_K10_CLEAN  K_PREVS=10 POS_W=4"
C_T5="PARTITION=$TEMP_K5_CLEAN  K_PREVS=5  POS_W=4"
C_T10="PARTITION=$TEMP_K10_CLEAN K_PREVS=10 POS_W=4"
C_CTRL_G5="ARCH=single PARTITION=$GEO_K5_CLEAN K_PREVS=5 POS_W=4"
C_CTRL_T5="ARCH=single PARTITION=$TEMP_K5_CLEAN K_PREVS=5 POS_W=4"

#                       queue    hostGB vramG walltime
RES_C_G5="long-gpu       80   36  28:00"   # est peak 59G
RES_C_G5_3X="long-gpu   128   36  48:00"   # est peak 117G -- the ONLY job near the cap
RES_C_G10="long-gpu     112   48  28:00"   # est peak 85G
RES_C_T5="long-gpu       96   36  35:00"   # est peak 77G (biggest pool)
RES_C_T10="long-gpu     128   48  32:00"   # est peak 101G
RES_C_CTRL="long-gpu     24   24   13:00"   # single-frame geo_k5, est peak 17G
RES_C_CTRL_T5="long-gpu  32   24   13:00"   # single-frame temporal_k5, est peak 22G

# Order matters: the geo_k5 1:1 ConvLSTM arm is the anchor every other run is
# read against. Submit it first so a queue that only clears one job still
# clears the interpretable one.
job clean22 clean_geo_k5_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$C_G5 $NEG_R3_1X $C_HYP" "$RES_C_G5" \
    "THE anchor: geo_k5, ConvLSTM h256, 1:1 near-field negatives, negative validation"
job clean22 clean_geo_k5_convlstm_ring3_3x \
    scripts/train/train_convlstm.sh "$C_G5 $NEG_R3_3X $C_HYP" "$RES_C_G5_3X" \
    "best config on the old data (3:1), re-run on clean ground -- does it survive the AOI?"
job clean22 clean_geo_k5_single_ring3 \
    scripts/train/train_control.sh "$C_CTRL_G5 $NEG_R3_1X $C_HYP" "$RES_C_CTRL" \
    "temporal-context control: what does the recurrence buy once background is scored?"
job clean22 clean_geo_k10_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$C_G10 $NEG_R3_1X $C_HYP" "$RES_C_G10" \
    "k5 vs k10 on the geo axis, same negatives"
job clean22 clean_temporal_k5_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$C_T5 $NEG_R3_1X $C_HYP" "$RES_C_T5" \
    "the temporal axis: train <2024, val <2025, test 2025+ -- the year shift, on the AOI"
job clean22 clean_temporal_k10_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$C_T10 $NEG_R3_1X $C_HYP" "$RES_C_T10" \
    "k5 vs k10 there; val is 16 intfs / 4,735 positives, thin but no longer the old 6"
# The control on BOTH axes, not just geo. Without it a "temporal context helps"
# claim on the temporal axis has no baseline of its own: the geo control only
# licenses the statement for geo-split data, and the two axes now differ in
# training pool as well as in split rule.
job clean22 clean_temporal_k5_single_ring3 \
    scripts/train/train_control.sh "$C_CTRL_T5 $NEG_R3_1X $C_HYP" "$RES_C_CTRL_T5" \
    "temporal-context control on the temporal axis -- the missing half of the geo control"

# ---- the attention arms of the clean batch ----------------------------------
# Two architectures beyond the ConvLSTM, on both axes, same negatives and same
# negative validation as everything else in clean22:
#
#   tattn   RECURRENCE=none      the current frame attends over its own history
#                                at each bottleneck pixel. The hypothesis is
#                                about PRECISION: atmospheric and decorrelation
#                                artefacts are temporally inconsistent while
#                                subsidence is persistent, so a learned
#                                weighted average over the chain is a matched
#                                filter for signal and a suppressor for noise.
#   hybrid  RECURRENCE=convlstm  attention over the chain PLUS the recurrent
#                                state -- the arm that tied ConvLSTM on geo in
#                                docs/PREDICTIONS.md while pure attention
#                                trailed it.
#
# Both at FUSE_SKIPS=0, matching the arms those results came from. FUSE_SKIPS=2
# exists (temporal_k5_tattn_fuse2_ring3 on the old data) and is one env var away
# -- add FUSE_SKIPS=2 to an override string -- but it is a third variant of an
# already-four-run block and is deliberately not queued here.
#
# Why both axes rather than geo alone: the precision claim is about suppressing
# temporally inconsistent artefacts, and the temporal axis is where the year
# shift actually stresses that. Running it on geo only would leave the
# architecture's own hypothesis untested.
#
# gmem: pure attention at k5/fuse0 fits 36G, the hybrid carries a 256-wide
# ConvLSTM on top of the attention stack and gets 48G. The template itself
# warns above 36G only for K_PREVS>=10 or FUSE_SKIPS>=3, neither of which
# applies here -- the bump is for the hybrid's extra state, not the warning.
C_TATTN_G5="PARTITION=$GEO_K5_CLEAN K_PREVS=5 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none"
C_HYBRID_G5="PARTITION=$GEO_K5_CLEAN K_PREVS=5 POS_W=4 FUSE_SKIPS=0 RECURRENCE=convlstm HIDDEN=256"
C_TATTN_T5="PARTITION=$TEMP_K5_CLEAN K_PREVS=5 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none"
C_HYBRID_T5="PARTITION=$TEMP_K5_CLEAN K_PREVS=5 POS_W=4 FUSE_SKIPS=0 RECURRENCE=convlstm HIDDEN=256"
# k=10 attention arms. The template itself warns that K_PREVS>=10 wants gmem 48G
# against the 36G its #BSUB line asks for -- that is exactly this case, and the
# RESOURCES below override the directive, which is why they live here.
C_TATTN_G10="PARTITION=$GEO_K10_CLEAN K_PREVS=10 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none"
C_HYBRID_G10="PARTITION=$GEO_K10_CLEAN K_PREVS=10 POS_W=4 FUSE_SKIPS=0 RECURRENCE=convlstm HIDDEN=256"
C_TATTN_T10="PARTITION=$TEMP_K10_CLEAN K_PREVS=10 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none"

RES_C_TATTN_G5="long-gpu   88   36  38:00"   # geo_k5 pool, attention only
RES_C_HYBRID_G5="long-gpu  88   48  44:00"   # + recurrent state (GPU-side, not host)
RES_C_TATTN_T5="long-gpu  104   36  44:00"   # temporal_k5: the biggest pool
RES_C_HYBRID_T5="long-gpu 104   48  51:00"   # biggest pool + recurrent state
RES_C_TATTN_G10="long-gpu  112   48  38:00"   # k=10 attention, est peak 85G
RES_C_HYBRID_G10="long-gpu 112   48  44:00"   # k=10 attention + recurrent state
RES_C_TATTN_T10="long-gpu  128   48  44:00"   # k=10 attention, est peak 101G

job clean22 clean_geo_k5_tattn_ring3 \
    scripts/train/train_tattn.sh "$C_TATTN_G5 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_G5" \
    "attention vs recurrence on geo, on clean ground and a precision-sensitive val set"
job clean22 clean_geo_k5_tattn_hybrid_ring3 \
    scripts/train/train_tattn.sh "$C_HYBRID_G5 $NEG_R3_1X $C_HYP" "$RES_C_HYBRID_G5" \
    "the hybrid tied ConvLSTM on the old geo data -- does it still, once the AOI removes the sea?"
job clean22 clean_temporal_k5_tattn_ring3 \
    scripts/train/train_tattn.sh "$C_TATTN_T5 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_T5" \
    "the artefact-suppression claim, tested where the year shift actually stresses it"
job clean22 clean_temporal_k5_tattn_hybrid_ring3 \
    scripts/train/train_tattn.sh "$C_HYBRID_T5 $NEG_R3_1X $C_HYP" "$RES_C_HYBRID_T5" \
    "hybrid on the temporal axis: recurrence and attention together against the 2025+ test set"

# The k=10 attention arms. Attention over a chain is the case where depth should
# matter MOST -- a longer history is more evidence for what is temporally
# persistent and what is not, which is the whole artefact-suppression argument.
# k5-only attention would leave that untested.
#
# Absent for symmetry: temporal_k10 hybrid. One `job` line away if wanted; left
# out because the geo axis already carries the tattn-vs-hybrid pair at k10 and
# the temporal axis carries it at k5, so both contrasts exist once.
job clean22 clean_geo_k10_tattn_ring3 \
    scripts/train/train_tattn.sh "$C_TATTN_G10 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_G10" \
    "attention at k=10 on geo: does a longer history buy what the k5 arm could not?"
job clean22 clean_geo_k10_tattn_hybrid_ring3 \
    scripts/train/train_tattn.sh "$C_HYBRID_G10 $NEG_R3_1X $C_HYP" "$RES_C_HYBRID_G10" \
    "the tattn-vs-hybrid contrast at k=10, against clean_geo_k10_convlstm_ring3"
job clean22 clean_temporal_k10_tattn_ring3 \
    scripts/train/train_tattn.sh "$C_TATTN_T10 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_T10" \
    "attention at k=10 under the year shift -- the longest history against the 2025+ test set"

# ---- training: the attention that actually selects (kind: attnfix) ----------
# DONE 2026-08-20. All five finished (768296-768300); four early-stopped on
# patience 40. They are left listed rather than commented out because the
# `attnpos` batch below re-runs them and the two must stay readable side by
# side -- but DO NOT RESUBMIT THIS KIND: it is ~30 GPU-hours of completed work,
# and it would now train against a positives-only val set anyway, i.e. it would
# silently become attnpos with the wrong job names. Use `attnpos`.
#
# Every tattn run above this line trained a DEAD attention. Measured on all 15
# checkpoints: the weights came out at exactly 1/T, so the model averaged its
# history instead of choosing from it, and the ~1M attention parameters were
# decoration. docs/ATTENTION_COLLAPSE.md carries the measurements; the short
# version is that the collapse is present before the first gradient step, not
# grown during training -- at init the frame-to-frame differences are 0.09% of
# the token the queries and keys are built from, so the softmax has nothing to
# separate, and RMSprop's normalised steps then grind q/k to zero.
#
# Two fixes, both ON BY DEFAULT in train_tattn.sh, neither sufficient alone:
#   CONTRAST=yes  queries/keys from token - mean_over_time(token), so selection
#                 runs on how frames DIFFER rather than on what they share
#   QK_NORM=yes   unit-norm q/k with one learned temperature, so selectivity
#                 stops riding on projection magnitude -- without it the same
#                 block goes uniform at LR=1e-6 and one-hot at LR=1e-5
#
# WHAT THIS BATCH IS FOR. The clean22 tattn numbers (obj F1 0.7378 geo_k10
# hybrid, 0.7271 geo_k10 plain) are real, but they measure temporal AVERAGING,
# not attention. This batch re-runs the same arms with selection working, so
# the comparison is like for like: same partitions, same negatives, same
# hyper-parameters, same val protocol as clean22. Only the attention differs.
#
# THE PAIRED CONTROL IS NOT OPTIONAL. clean_geo_k10_tattn_prefix_ring3 trains
# the OLD architecture from this same code (CONTRAST=no QK_NORM=no). Without
# it the comparison is against numbers produced by a different commit on
# checkpoints that no longer exist, which confounds the fix with everything
# else that changed. It is one extra run and it is the whole experiment.
#
# CHECK THE RESULT SELECTED, do not assume it. After each run:
#     sinkholes attention-probe --model outputs/<run>/checkpoints/best.pt \
#       --partition assets/partition_geo_k10_clean.json --split val \
#       --patches_dir "$DATA/patches" --control_lookback 10 \
#       --require_selectivity 0.9
# which exits non-zero if the attention is at or above 90% of uniform. That is
# the check that would have caught the original collapse, and it can only be
# made against a trained checkpoint -- a fresh model passes every
# content-sensitivity test and still dies in training.
#
# Resources are copied from the clean22 arms unchanged: the fix adds one
# LayerNorm and one scalar per attention block (~66k parameters on a 32M model),
# so neither VRAM nor epoch time moves measurably.
C_FIX_G5="$C_TATTN_G5 CONTRAST=yes QK_NORM=yes"
C_FIX_G10="$C_TATTN_G10 CONTRAST=yes QK_NORM=yes"
C_FIX_HYBRID_G10="$C_HYBRID_G10 CONTRAST=yes QK_NORM=yes"
C_FIX_T5="$C_TATTN_T5 CONTRAST=yes QK_NORM=yes"
C_PREFIX_G10="$C_TATTN_G10 CONTRAST=no QK_NORM=no"

job attnfix clean_geo_k10_tattn_fixed_ring3 \
    scripts/train/train_tattn.sh "$C_FIX_G10 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_G10" \
    "the headline: does attention that SELECTS beat the averaging it was doing, at the depth where it should matter most"
job attnfix clean_geo_k10_tattn_prefix_ring3 \
    scripts/train/train_tattn.sh "$C_PREFIX_G10 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_G10" \
    "the paired control: the OLD dead-attention architecture from THIS code, so the pair differs in the fix and nothing else"
job attnfix clean_geo_k5_tattn_fixed_ring3 \
    scripts/train/train_tattn.sh "$C_FIX_G5 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_G5" \
    "k5 with selection: k10 beat k5 while merely averaging, so the depth ordering may not survive the fix"
job attnfix clean_geo_k10_tattn_hybrid_fixed_ring3 \
    scripts/train/train_tattn.sh "$C_FIX_HYBRID_G10 $NEG_R3_1X $C_HYP" "$RES_C_HYBRID_G10" \
    "the best clean22 arm (0.7378), re-run with working selection"
job attnfix clean_temporal_k5_tattn_fixed_ring3 \
    scripts/train/train_tattn.sh "$C_FIX_T5 $NEG_R3_1X $C_HYP" "$RES_C_TATTN_T5" \
    "the artefact-suppression claim on the axis that stresses it -- and the claim was always about selecting frames, never averaging them"

# ---- training: the attnfix five, positives-only val (kind: attnpos) ---------
# THE SAME FIVE RUNS AS attnfix, with the validation negatives gone. Nothing
# else moves: same partitions, same architectures, same ring-3 1:1 training
# negatives, same LR/epochs/patience/seed. Only the val set changes, and with
# it which checkpoint gets kept.
#
# WHY RERUN RATHER THAN RESCORE -- the same argument the valneg batch made in
# the opposite direction, and it is the whole reason this costs GPU hours.
# best.pt is selected on val/dice, and val/dice is also what drives the plateau
# schedule and early stopping. Every one of the five below was SELECTED on an
# inflated curve, so re-scoring those weights would answer a different question
# than "what would this experiment have produced under the protocol we now
# use". Four of the five never reached epoch 100 either: they early-stopped on
# patience 40 against that curve, so even the run LENGTH is a product of it.
#
# WHAT THE ORIGINALS DID (all five finished; four early-stopped):
#
#   run                      best val/dice @ epoch   stopped   measured
#   geo_k10  fixed            0.7331 @ 34             74/100    246 s/epoch
#   geo_k10  prefix (control) 0.7271 @ 46             86/100    232 s/epoch
#   geo_k10  hybrid fixed     0.7442 @ 56             96/100    292 s/epoch
#   geo_k5   fixed            0.7233 @ 48             88/100    374 s/epoch
#   temporal_k5 fixed         0.7957 @ 75            100/100    416 s/epoch
#
# PLUS FOUR TEMPORAL ARMS THAT HAVE NO POSITIVES-ONLY TWIN YET. The temporal
# axis (train 2019-22 -> val 2023-24) is the one that stresses the artefact-
# suppression claim, and attnfix covered it with a single k5 run. These four
# complete it: {attention, hybrid} x {k5, k10} for the tattn family, plus the
# ConvLSTM at k10 as the non-attention reference on the same axis.
#
#   clean_temporal_k5_convlstm_ring3_valpos IS ALREADY DONE -- LSF 664204, the
#   full 100 epochs on 2026-08-19, positives-only. It is NOT re-queued here.
#   That run is the ConvLSTM half of the k5 reference; this batch adds k10.
#
# Those dice figures are inflated by roughly (1 + dice_on_positives)/2 -- an
# empty prediction on an empty mask scores dice 1.0 -- so expect this batch to
# land near 0.63-0.66 on the same models. THAT IS NOT A REGRESSION, it is the
# padding coming off. The comparison that means anything is attnpos against
# valpos (both positives-only) and, above all, object-level F1 from run_eval.sh.
#
# WHAT THIS COSTS, unchanged from the valpos batch: positives-only validation
# goes back to selecting a checkpoint that cannot see a false positive on
# background, so expect more recall-heavy weights than the attnfix twins. The
# object-level eval is what absorbs that, not this curve. Read val/F1 rather
# than val/dice while the runs are live -- it is pooled over raw pixel counts
# (evaluate.py:300-306), so it is negative-aware on both protocols and is the
# one column that IS directly comparable to the attnfix runs.
#
# NO PAIRED CONTROL IN THIS BATCH, deliberately. attnfix already ran the
# fixed/prefix pair against each other on ONE protocol (0.7331 vs 0.7271, both
# with validation negatives), so "does the fix beat dead attention" is already
# answered and does not need paying for twice. What is NOT answered there is
# what those architectures produce when the curve that selects best.pt is not
# padded, and that is all this batch is for. The consequence to keep in mind:
# nothing in attnpos isolates the attention fix on its own -- for that
# comparison, read the attnfix pair.
#
# STILL CHECK THE ATTENTION SELECTED, per run, before believing any of it:
#     sinkholes attention-probe --model outputs/<run>/checkpoints/best.pt \
#       --partition assets/partition_geo_k10_clean.json --split val \
#       --patches_dir "$DATA/patches" --control_lookback 10 \
#       --require_selectivity 0.9
#
# THE THREE NEW CONFIGS. C_TATTN_T10 already existed; the temporal k10 hybrid
# did not, because clean22 never ran one. It is the geo k10 hybrid's config on
# the temporal partition and nothing else.
C_HYBRID_T10="PARTITION=$TEMP_K10_CLEAN K_PREVS=10 POS_W=4 FUSE_SKIPS=0 RECURRENCE=convlstm HIDDEN=256"
C_FIX_T10="$C_TATTN_T10 CONTRAST=yes QK_NORM=yes"
C_FIX_HYBRID_T5="$C_HYBRID_T5 CONTRAST=yes QK_NORM=yes"
C_FIX_HYBRID_T10="$C_HYBRID_T10 CONTRAST=yes QK_NORM=yes"

# RESOURCES ARE MEASURED wherever a twin exists -- LSF Max Memory and the
# trainer's own peak-VRAM line, per job. Host GB is the measured peak +~25%,
# and walltime the measured seconds-per-epoch carried to the FULL 100 epochs
# (none of these may early-stop where its twin did) with ~60% on top. Dropping
# the validation negatives cuts the val set in half and validation runs at
# batch size 1, so every walltime here is an upper bound -- measured on the two
# runs that exist in both protocols, temporal_k5 convlstm went 513 -> 331
# s/epoch, a 35% saving, purely from losing the negatives.
#
#   run                          LSF Max Memory / VRAM reserved / s per epoch
#   geo_k10  fixed        768296  85187 MB   44.4 GiB   246   (attnfix)
#   geo_k10  hybrid       768299  83998 MB   45.1 GiB   292   (attnfix)
#   geo_k5   fixed        768298  55832 MB   25.8 GiB   374   (attnfix)
#   temporal_k5 fixed     768300  69437 MB   25.8 GiB   416   (attnfix)
#   temporal_k10 convlstm 372690  98729 MB   44.4 GiB   365   (clean22)
#   temporal_k5 convlstm  664204  64888 MB   25.8 GiB   331   (valpos, pos-only)
#
# The two temporal arms with no twin at all are derived, not measured, and the
# derivation is stated so it can be checked: the hybrid costs ~1.19x the plain
# attention per epoch (292/246, measured on geo_k10), and temporal k10 costs
# ~0.71x temporal k5 per epoch (365/513, measured on the clean22 ConvLSTMs --
# k10 has a smaller pool). Host memory for a k10 temporal arm is taken from the
# k10 ConvLSTM's measured 96.4 GB, since host RAM is dominated by the patch
# grids in memory rather than by the model.
#
# gmem: 36G on the k5 arms, 48G on the k10 arms, and the split is forced by
# measurement rather than chosen. Across all 15 runs that have ever reported a
# peak, the two depths sit in two tight bands and nothing straddles them:
#
#   k5   (convlstm, tattn, hybrid)   25.7 - 25.8 GiB reserved
#   k10  (convlstm, tattn, hybrid)   44.1 - 45.1 GiB reserved
#
# VRAM here is set by batch size and depth, NOT by pool size, which is why geo
# and temporal arms of the same k measure the same and why the temporal k10
# arms are sized off the geo k10 measurements without apology.
#
# The k5 arms were dropped 48G -> 36G on 2026-08-20 to stop them queueing for a
# large-memory card they do not need: 25.8 GiB leaves 10 GiB of headroom at
# 36G, and a 36G card is a far commoner slot. THE K10 ARMS CANNOT FOLLOW. At
# 44-45 GiB measured they are 8-9 GiB OVER a 36G card, so a 36G request does
# not make them schedule sooner -- it makes them wait for a slot and then die
# on CUDA OOM partway through epoch 1. If they must fit 36G, the lever is
# BATCH=64 (halves the activation memory), and that changes the experiment:
# batch size is part of the strict resume fingerprint and moves the results.
#
#                            queue     hostGB vramG walltime
RES_A_G10="long-gpu          104   48  11:00"   # 83.2 GB / 44.4 GiB / 246 s/ep
RES_A_HYBRID_G10="long-gpu   104   48  13:00"   # 82.0 GB / 45.1 GiB / 292 s/ep
RES_A_G5="long-gpu            72   36  17:00"   # 54.5 GB / 25.8 GiB / 374 s/ep
RES_A_T5="long-gpu            88   36  19:00"   # 67.8 GB / 25.7 GiB / 416 s/ep
RES_A_HYBRID_T5="long-gpu     88   36  22:00"   # 67.8 GB meas / 26.2 GiB (geo twin)
RES_A_T10="long-gpu          120   48  14:00"   # 96.4 GB meas / ~296 s/ep est
RES_A_HYBRID_T10="long-gpu   120   48  16:00"   # 96.4 GB meas / ~352 s/ep est
RES_A_CONV_T10="long-gpu     120   48  17:00"   # 96.4 GB / 44.4 GiB / 365 s/ep

# Submit order: the geo arms first because they are the axis the project is
# steering toward and they are the cheapest, then the temporal block, longest
# job last so it is not holding a slot while the readable ones queue behind it.
job attnpos clean_geo_k10_tattn_fixed_ring3_valpos \
    scripts/train/train_tattn.sh "$C_FIX_G10 $NEG_R3_1X $C_HYP" "$RES_A_G10" \
    "the headline, on the protocol we keep: does selection beat averaging when the curve is not padded? (twin: 0.7331 @34, stopped 74)"
job attnpos clean_geo_k5_tattn_fixed_ring3_valpos \
    scripts/train/train_tattn.sh "$C_FIX_G5 $NEG_R3_1X $C_HYP" "$RES_A_G5" \
    "k5 with selection: does the k10-over-k5 ordering survive both the fix and the protocol change? (twin: 0.7233 @48, stopped 88)"
job attnpos clean_geo_k10_tattn_hybrid_fixed_ring3_valpos \
    scripts/train/train_tattn.sh "$C_FIX_HYBRID_G10 $NEG_R3_1X $C_HYP" "$RES_A_HYBRID_G10" \
    "best attnfix arm (0.7442 @56, stopped 96) -- and the arm whose checkpoint selection had the most room to move"
job attnpos clean_temporal_k5_tattn_fixed_ring3_valpos \
    scripts/train/train_tattn.sh "$C_FIX_T5 $NEG_R3_1X $C_HYP" "$RES_A_T5" \
    "the artefact-suppression claim on the axis that stresses it; the only twin that ran all 100 epochs (0.7957 @75)"
job attnpos clean_temporal_k10_tattn_fixed_ring3_valpos \
    scripts/train/train_tattn.sh "$C_FIX_T10 $NEG_R3_1X $C_HYP" "$RES_A_T10" \
    "NEW: does more history help on the axis where history is the whole claim? k5 vs k10 under working attention, temporal side"
job attnpos clean_temporal_k5_tattn_hybrid_fixed_ring3_valpos \
    scripts/train/train_tattn.sh "$C_FIX_HYBRID_T5 $NEG_R3_1X $C_HYP" "$RES_A_HYBRID_T5" \
    "NEW: the hybrid carried geo; this is whether recurrence-plus-attention also carries the temporal axis at k5"
job attnpos clean_temporal_k10_tattn_hybrid_fixed_ring3_valpos \
    scripts/train/train_tattn.sh "$C_FIX_HYBRID_T10 $NEG_R3_1X $C_HYP" "$RES_A_HYBRID_T10" \
    "NEW: completes the temporal 2x2 {attention, hybrid} x {k5, k10} -- a config clean22 never ran at all"
job attnpos clean_temporal_k10_convlstm_ring3_valpos \
    scripts/train/train_convlstm.sh "$C_T10 $NEG_R3_1X $C_HYP" "$RES_A_CONV_T10" \
    "NEW: the non-attention reference at k10, so the temporal attention numbers are not read without one (k5's twin, 664204, is already done)"

# ---- training: negatives in TRAIN ONLY (kind: valpos) -----------------------
# Six of the fourteen clean22 runs, resubmitted with ONE change: the validation
# negatives are dropped, so RING_NEGS=yes still puts background patches in
# TRAINING and the validation split goes back to positives only.
#
# AS OF 2026-08-20 THIS IS NO LONGER A VARIANT -- it is what every job in this
# file does, since --add_val_negatives was removed from all of them. The batch
# is kept under its own name because it is the six runs that ANSWER the
# question; the reasoning below is why the flag went away at all.
#
# WHY. clean22's val/dice landed at 0.72-0.80 against the 0.64-0.66 every
# earlier batch produced, and that jump is an artefact of the metric, not a
# better model. dice_coeff (sinkholes/training/losses.py:12-22) maps an empty
# prediction on an empty mask to (0+eps)/(0+eps) = 1.0, so at 1:1 validation
# negatives roughly half the val samples score ~1.0 and the mean is about
# (1 + dice_on_positives) / 2. Nothing in those runs is comparable to anything
# before them on that column.
#
# WHAT THIS COSTS, and it is not nothing. best.pt is chosen on val/dice
# (train.py:900-915), so positives-only validation goes back to selecting the
# checkpoint that cannot see a false positive -- which is the failure mode
# docs/PLAN_CLEAN_BENCHMARK.md put VAL_NEGS in for, and the reason the whole
# 2026-08-10 batch landed inside its own noise floor. Expect these six to
# select more recall-heavy weights than their clean22 twins. The object-level
# eval, not this curve, is what settles them.
#
# WORTH KNOWING BEFORE SPENDING THE HOURS: val/F1, val/P and val/R in every
# clean22 results.csv are accumulated from raw pixel tp/fp/fn
# (evaluate.py:300-306), not per-sample, so they are ALREADY negative-aware and
# already un-inflated -- the clean22 F1 column reads 0.62-0.74 and ranks the
# arms the same way this batch will. These runs change which checkpoint is
# kept; they do not change what is measurable.
#
# THE SIX. Geo-weighted, because geo (north->south) is the axis the project is
# steering toward, and geo is where over-prediction has always been worst. The
# geo four are a 2x2: {ConvLSTM, hybrid attention} x {k=5, k=10}, so depth and
# architecture are both readable without a third factor moving. Plus the geo
# single-frame control -- last night it placed LAST on dice and MID-PACK on F1
# (0.6906 / 0.6348), the exact disagreement checkpoint selection can move --
# and one temporal ConvLSTM so the geo numbers are not read in isolation.
#
# Left out on purpose: the 3:1 arm (16.9 h measured, and 1:1 vs 3:1 is a
# training-side question this batch does not touch), pure tattn on both axes
# (the hybrid carried the geo contrast last night), and the temporal k10 pair.
#
# RESOURCES ARE NOW MEASURED, not estimated -- all fourteen clean22 jobs
# reported Max Memory and peak VRAM, and every one of them came in far under
# its request. Host GB below is the measured peak +~25%, VRAM is the measured
# RESERVED figure +~25%, walltime is the measured seconds-per-epoch carried to
# the full 100-epoch ceiling with ~60% on top. Dropping the validation
# negatives halves the val set, so all three move DOWN from here, never up.
#
#   run                     measured host / VRAM(res) / s per epoch
#   geo_k5   convlstm        55.9 GB   25.8 GiB   222   (97 epochs, 6.0 h)
#   geo_k5   hybrid          55.8 GB   26.2 GiB   271   (90 epochs, 6.8 h)
#   geo_k5   single          20.8 GB    8.3 GiB   151   (73 epochs, 3.1 h)
#   geo_k10  convlstm        85.4 GB   44.4 GiB   326   (49 epochs, 4.4 h)
#   geo_k10  hybrid          85.5 GB   44.4 GiB   361   (65 epochs, 6.5 h)
#   temp_k5  convlstm        69.2 GB   25.8 GiB   546  (100 epochs, 15.2 h)
#
# The temporal ConvLSTM's 546 s/epoch against the geo ConvLSTM's 222 is larger
# than the pool difference (163 vs 127 interferograms) explains, so some of it
# is card variance. Its 26:00 keeps the margin that gap deserves.
#
#                        queue    hostGB vramG walltime
RES_V_G5="long-gpu        72   32  10:00"   # measured 55.9 GB / 25.8 GiB / 6.0 h
RES_V_HYBRID_G5="long-gpu 72   36  12:00"   # measured 55.8 GB / 26.2 GiB / 6.8 h
RES_V_CTRL_G5="long-gpu   28   16   8:00"   # measured 20.8 GB /  8.3 GiB / 3.1 h
RES_V_G10="long-gpu      108   48  14:00"   # measured 85.4 GB / 44.4 GiB / 4.4 h
RES_V_HYBRID_G10="long-gpu 108 48  15:00"   # measured 85.5 GB / 44.4 GiB / 6.5 h
RES_V_T5="long-gpu        88   32  26:00"   # measured 69.2 GB / 25.8 GiB / 15.2 h

# The anchor first, same rule as clean22: a queue that clears one job should
# clear the interpretable one.
job valpos clean_geo_k5_convlstm_ring3_valpos \
    scripts/train/train_convlstm.sh "$C_G5 $NEG_R3_1X $C_HYP" "$RES_V_G5" \
    "THE anchor, train-side negatives only -- the direct twin of clean_geo_k5_convlstm_ring3"
job valpos clean_geo_k5_tattn_hybrid_ring3_valpos \
    scripts/train/train_tattn.sh "$C_HYBRID_G5 $NEG_R3_1X $C_HYP" "$RES_V_HYBRID_G5" \
    "best geo arm at k=5 last night (F1 0.6395) -- does it stay there on a positives-only curve?"
job valpos clean_geo_k5_single_ring3_valpos \
    scripts/train/train_control.sh "$C_CTRL_G5 $NEG_R3_1X $C_HYP" "$RES_V_CTRL_G5" \
    "the control, and the run whose dice and F1 disagreed most -- 3 h, the cheapest arm here"
job valpos clean_geo_k10_convlstm_ring3_valpos \
    scripts/train/train_convlstm.sh "$C_G10 $NEG_R3_1X $C_HYP" "$RES_V_G10" \
    "k5 vs k10 under ConvLSTM, completing the geo 2x2"
job valpos clean_geo_k10_tattn_hybrid_ring3_valpos \
    scripts/train/train_tattn.sh "$C_HYBRID_G10 $NEG_R3_1X $C_HYP" "$RES_V_HYBRID_G10" \
    "top geo arm overall last night (F1 0.6437); k5 vs k10 under attention, the other half of the 2x2"
job valpos clean_temporal_k5_convlstm_ring3_valpos \
    scripts/train/train_convlstm.sh "$C_T5 $NEG_R3_1X $C_HYP" "$RES_V_T5" \
    "one temporal arm so the geo five are not read without a cross-axis reference"

# ---- training: the 2019-2022 archive (kind: pre23) --------------------------
# ELEVEN RUNS on assets/partition_*_pre2023.json, generated 2026-08-20 -- six
# ConvLSTM/control arms here and five attention arms in the block below. The
# same generator, cut, AOI, seed and val share as the clean22 partitions -- ONE
# factor differs: the archive stops at 2022-12-31.
#
# WHY. Scene-level false positives are concentrated in the post-2022
# interferograms. docs/PLAN_CLEAN_BENCHMARK.md section 1 measured it on
# geo_k5_convlstm_ring3_3x: object-level precision 0.91 on 2021 scenes, 0.44 on
# 2024 and 2025, while recall holds at 0.91-0.93. The model finds the same
# objects; the newer BACKGROUND generates the detections. Section 0a kept those
# years anyway and recorded the mechanism as unexplained and the risk as
# knowingly accepted. This batch is the other arm of that decision: train and
# score with the suspect years absent, so "how much of the false-positive rate
# is the newer archive" becomes a measurement instead of an argument.
#
# WHAT IT IS NOT. It is not a replacement for clean22 and its numbers do not
# supersede anything. Read a pre23 run against its clean22 TWIN -- same
# architecture, same negatives, same hyper-parameters -- and the difference is
# the archive. Read one against a clean22 arm of a different architecture and
# you have confounded the two.
#
# THE HOLD-OUT IS SMALLER, on both axes, and that is the price of the design:
#
#            train intfs / AOI positives      val            test
#   geo_k5        90 / 21,432              24 / 2,576     34 / 3,686
#   geo_k10       71 / 16,690              15 / 1,462     22 / 2,377
#   temp_k5       77 / 24,817              15 / 6,078     48 / 5,652
#   temp_k10      50 / 16,409               6 / 2,632     35 / 3,890
#
# TEMPORAL k10 VAL IS 6 INTERFEROGRAMS. That is the risk PLAN section 11 flags
# as the plan's biggest, and the padding that used to hide it is gone --
# --add_val_negatives was removed on 2026-08-20. Six scenes drive the plateau
# schedule, early stopping and best.pt selection for that arm, so treat its
# val/dice curve as noise and judge it at object level like everything else.
# The documented fallback, if it proves unusable, is to let VAL-ONLY chains
# reach back into train (PLAN section 5), which the generator does not do today.
#
# TEMPORAL BOUNDS ARE NOT THE clean22 ONES. 20240101/20250101 put every val and
# test scene in years this archive does not contain. These files use
# 20210101/20210701 -- train 2019-2020, val 2021H1, test 2021H2 onward -- which
# is the layout PLAN section 5 chose from a sweep of all 320 viable monthly
# boundary pairs, back when the archive ended in 2022. It is the right layout
# for THIS archive and the wrong one for the full archive; section 5a explains
# why, and the two must not be swapped.
PRE23_G5=assets/partition_geo_k5_pre2023.json
PRE23_G10=assets/partition_geo_k10_pre2023.json
PRE23_T5=assets/partition_temporal_k5_pre2023.json
PRE23_T10=assets/partition_temporal_k10_pre2023.json

# $C_HYP verbatim from clean22 (LR=1e-5 EPOCHS=100 PATIENCE=40). Do not give
# this batch hyper-parameters of its own: the comparison it exists to make is
# pre23 against clean22, and a second moving factor destroys it.
P_G5="PARTITION=$PRE23_G5  K_PREVS=5  POS_W=4"
P_G10="PARTITION=$PRE23_G10 K_PREVS=10 POS_W=4"
P_T5="PARTITION=$PRE23_T5  K_PREVS=5  POS_W=4"
P_T10="PARTITION=$PRE23_T10 K_PREVS=10 POS_W=4"
P_CTRL_G5="ARCH=single PARTITION=$PRE23_G5 K_PREVS=5 POS_W=4"
P_CTRL_T5="ARCH=single PARTITION=$PRE23_T5 K_PREVS=5 POS_W=4"
# CONTRAST/QK_NORM are the train_tattn.sh defaults since the attention fix, but
# they are named here anyway: they are STRICT resume keys, and an arm whose
# attention silently reverted would look like an archive effect.
#
# RECURRENCE=none is PURE attention -- the current frame attends over its own
# history and there is no recurrent state. RECURRENCE=convlstm is the hybrid,
# which carries both. Both are needed: the hybrid alone cannot separate "the
# attention suppressed the artefact" from "the recurrence did", and separating
# them is the whole reason attention is in this batch (see the block below).
P_TATTN_G5="PARTITION=$PRE23_G5 K_PREVS=5 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none CONTRAST=yes QK_NORM=yes"
P_TATTN_G10="PARTITION=$PRE23_G10 K_PREVS=10 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none CONTRAST=yes QK_NORM=yes"
P_TATTN_T5="PARTITION=$PRE23_T5 K_PREVS=5 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none CONTRAST=yes QK_NORM=yes"
P_TATTN_T10="PARTITION=$PRE23_T10 K_PREVS=10 POS_W=4 FUSE_SKIPS=0 RECURRENCE=none CONTRAST=yes QK_NORM=yes"
P_HYBRID_G10="PARTITION=$PRE23_G10 K_PREVS=10 POS_W=4 FUSE_SKIPS=0 RECURRENCE=convlstm HIDDEN=256 CONTRAST=yes QK_NORM=yes"

# RESOURCES are the clean22 rungs SCALED BY THE TRAINING POOL, since host memory
# tracks stored samples and epoch time tracks them almost linearly. Pool ratios
# against the clean22 twin: geo_k5 0.66, geo_k10 0.61, temp_k5 0.58, temp_k10
# 0.50. Applied to clean22's estimated peaks, then +25% headroom, then rounded
# up to a rung. VRAM does NOT scale with pool size and is copied unchanged.
#
#                     queue    hostGB vramG walltime   est peak
RES_P_G5="long-gpu       56   36  20:00"   # 39G  (clean22: 59G / 28:00)
RES_P_G10="long-gpu      72   48  19:00"   # 52G  (clean22: 85G / 28:00)
RES_P_T5="long-gpu       64   36  22:00"   # 44G  (clean22: 77G / 35:00)
RES_P_T10="long-gpu      72   48  18:00"   # 51G  (clean22: 101G / 32:00)
RES_P_CTRL_G5="long-gpu   24   24  10:00"  # 11G  (clean22: 17G / 13:00)
RES_P_CTRL_T5="long-gpu   24   24   9:00"  # 13G  (clean22: 22G / 13:00)
# The attention rungs. Host memory is a little above the ConvLSTM twin at the
# same partition for the same reason it is in clean22 (88 vs 80 on geo_k5);
# VRAM follows clean22's split -- 36G for pure attention at k5, 48G at k10 and
# for anything carrying the recurrent state.
RES_P_TATTN_G5="long-gpu    64   36  25:00"  # 39G  (clean22: 88G / 38:00)
RES_P_TATTN_G10="long-gpu   72   48  23:00"  # 52G  (clean22: 112G / 38:00)
RES_P_TATTN_T5="long-gpu    64   36  26:00"  # 44G  (clean22: 104G / 44:00)
RES_P_TATTN_T10="long-gpu   72   48  22:00"  # 51G  (clean22: 128G / 44:00)
RES_P_HYBRID_G10="long-gpu  72   48  29:00"  # 52G  (clean22: 85G / 44:00)
#
# RES_P_HYBRID_G10 asks 29:00 and RES_P_TATTN_T5 26:00, over the 24:00 that this
# file warns may be long-gpu's hard cap. clean22 and attnfix both had accepted
# requests above it (44:00, 51:00), so the cap is higher than 24 h in practice.
# If one bounces, lower its -W here rather than splitting the run: it resumes
# from resume.pt with
#     RESUME=<run dir> bsub -J <same name> < scripts/train/train_tattn.sh

# Order matters, the same rule as clean22: the geo_k5 ConvLSTM anchor first, so
# a queue that clears one job clears the interpretable one.
job pre23 pre23_geo_k5_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$P_G5 $NEG_R3_1X $C_HYP" "$RES_P_G5" \
    "THE anchor: the exact clean_geo_k5_convlstm_ring3 config with 2023+ removed from the archive"
job pre23 pre23_geo_k10_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$P_G10 $NEG_R3_1X $C_HYP" "$RES_P_G10" \
    "k5 vs k10 on the geo axis, so the depth ordering can be read on this archive too"
job pre23 pre23_geo_k5_single_ring3 \
    scripts/train/train_control.sh "$P_CTRL_G5 $NEG_R3_1X $C_HYP" "$RES_P_CTRL_G5" \
    "the temporal-context control: if the newer years were the false positives, the recurrence should buy LESS here, not more"
job pre23 pre23_temporal_k5_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$P_T5 $NEG_R3_1X $C_HYP" "$RES_P_T5" \
    "the temporal axis inside the old archive: train 2019-20, val 2021H1, test 2021H2+ -- a year shift with no suspect years in it"
job pre23 pre23_temporal_k10_convlstm_ring3 \
    scripts/train/train_convlstm.sh "$P_T10 $NEG_R3_1X $C_HYP" "$RES_P_T10" \
    "k5 vs k10 there. Val is SIX interferograms: read this arm at object level only"
job pre23 pre23_temporal_k5_single_ring3 \
    scripts/train/train_control.sh "$P_CTRL_T5 $NEG_R3_1X $C_HYP" "$RES_P_CTRL_T5" \
    "the control on the temporal axis, the missing half of the geo control -- clean22 carries both, so this batch must too"

# ---- the attention arms of the pre23 batch ----------------------------------
# FIVE ARMS, and they are the point of the batch rather than an extra.
#
# Attention's hypothesis IS the hypothesis this batch tests. Atmospheric and
# decorrelation artefacts are temporally inconsistent while subsidence is
# persistent, so a learned weighted average over the chain should be a matched
# filter for the signal and a suppressor for the noise -- i.e. a PRECISION
# claim, about exactly the false positives that section 1 measured piling up in
# the post-2022 scenes. If the newer years' background is what the ConvLSTM was
# flagging, attention is the architecture that should care most about their
# removal, in either direction: it should either gain the least (it was already
# suppressing them) or the most (it never could).
#
# PURE ATTENTION AT ALL FOUR CORNERS, plus the one hybrid. RECURRENCE=none is
# the arm that isolates the claim; the hybrid carries a ConvLSTM as well, so on
# its own it cannot say which half did the work. clean22 shipped both and its
# top two arms were hybrids (0.7378 geo_k10, 0.7350 geo_k5) with pure attention
# a few thousandths behind (0.7271, 0.7246) -- a margin small enough that the
# question of which mechanism earns it is still open, and small enough that
# reading it off one hybrid would be reading noise.
#
# Depth matters most here, which is why k10 is not skipped on either axis: a
# longer history is more evidence for what is temporally persistent and what is
# not, and that is the whole artefact-suppression argument. On the temporal
# axis, note that pre23_temporal_k10_tattn_ring3 selects its checkpoint on SIX
# validation interferograms (see the block above) -- queue it, but settle it
# with scripts/eval/run_eval.sh, never with its val curve.
#
# All five at FUSE_SKIPS=0, matching every clean22 and attnfix arm they pair
# with. FUSE_SKIPS=2 is one env var away and is deliberately not queued.
#
# Absent for symmetry: the temporal hybrids and the geo_k5 hybrid. Each is one
# `job` line away. Left out because the batch already carries pure attention at
# every corner and one hybrid to anchor it against clean22's best arm; adding
# the other three hybrids is a ~90 h decision, not a free one.
job pre23 pre23_geo_k5_tattn_ring3 \
    scripts/train/train_tattn.sh "$P_TATTN_G5 $NEG_R3_1X $C_HYP" "$RES_P_TATTN_G5" \
    "pure attention on the geo anchor: the precision claim, on an archive with the suspect years removed"
job pre23 pre23_geo_k10_tattn_ring3 \
    scripts/train/train_tattn.sh "$P_TATTN_G10 $NEG_R3_1X $C_HYP" "$RES_P_TATTN_G10" \
    "the same at k=10, where a longer history is more evidence for what is temporally persistent"
job pre23 pre23_geo_k10_tattn_hybrid_ring3 \
    scripts/train/train_tattn.sh "$P_HYBRID_G10 $NEG_R3_1X $C_HYP" "$RES_P_HYBRID_G10" \
    "the best clean22 arm (obj F1 0.7378) on the pre-2023 archive -- and the hybrid half of the k10 attention-vs-hybrid pair"
job pre23 pre23_temporal_k5_tattn_ring3 \
    scripts/train/train_tattn.sh "$P_TATTN_T5 $NEG_R3_1X $C_HYP" "$RES_P_TATTN_T5" \
    "artefact suppression on the axis that stresses it, with no post-2022 scene on either side of the boundary"
job pre23 pre23_temporal_k10_tattn_ring3 \
    scripts/train/train_tattn.sh "$P_TATTN_T10 $NEG_R3_1X $C_HYP" "$RES_P_TATTN_T10" \
    "the longest history under the year shift. SIX val interferograms: judge it at object level only"

# ---- evaluation: the 2026-08-10 batch (kind: eval2) -------------------------
# THIS IS THE BATCH THAT SETTLES 2026-08-10. Its own kind so it can be sent
# without dragging the 2026-08-09 backlog along:  submit_all.sh eval2 --submit
#
# Six jobs, ~3 h each. They are chosen as PAIRS, because every question the
# 2026-08-10 batch asked is a difference between two runs and a single eval in
# isolation answers none of them. What each pair buys:
#
#   negatives, temporal:  T5_POSW4  vs  T5_NEG1_R3
#   negatives, geo:       G5_POSW4  vs  G5_NEG1_R3
#   negatives, geo_k10:   $G10_POSW4 (already queued as eval_geo_k10_posw4
#                         in the 2026-08-09 table)  vs  G10_NEG1_R3
#   architecture:         T5_NEG1_R3 vs TA_FUSE0_NEG   (both under negatives)
#                         T5_POSW4   vs TA_FUSE0       (both without)
#
# The number to beat is object-level precision at low confidence, which is what
# --nonz_only training wrecked and what ring negatives exist to repair:
#
#   geo_k10  P=0.121 @ 0.125   (2026-08-06 reference, the worst case)
#   geo_k5   P=0.236 @ 0.125
#   temporal_k5 (pos_w 2)  P=0.440 @ 0.125
#
# Scene lists follow the established pairing: temporal runs on temporal_k10's
# 20-scene test list, geo runs on geo_k10's 18, both being the shared lists
# every model can be scored on. K_PREVS tracks the CHECKPOINT, so K_PREVS=5 on
# a _k10 group is expected here and run_eval.sh only notes it.
job eval2 eval_t5_posw4_base \
    scripts/eval/run_eval.sh "RUN=$T5_POSW4 GROUP=temporal_k10 K_PREVS=5" "$RES_EVAL" \
    "the negatives ANCHOR on temporal: without it the ring3 number below means nothing"
job eval2 eval_t5_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$T5_NEG1_R3 GROUP=temporal_k10 K_PREVS=5" "$RES_EVAL" \
    "THE headline test: does 1:1 near-field ring negatives buy object precision on temporal?"
job eval2 eval_g5_posw4_base \
    scripts/eval/run_eval.sh "RUN=$G5_POSW4 GROUP=geo_k10 K_PREVS=5" "$RES_EVAL" \
    "the negatives anchor on geo, at pos_w 4 (the old geo_k5 eval sat at pos_w 8 and cannot serve)"
job eval2 eval_g5_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$G5_NEG1_R3 GROUP=geo_k10 K_PREVS=5" "$RES_EVAL" \
    "the same test under the north->south shift, where over-prediction is worst (P=0.236 to beat)"
job eval2 eval_g10_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$G10_NEG1_R3 GROUP=geo_k10 K_PREVS=10" "$RES_EVAL" \
    "the worst over-prediction in the project (P=0.121); pairs with eval_geo_k10_posw4"
job eval2 eval_tattn_fuse0_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$TA_FUSE0_NEG GROUP=temporal_k10 K_PREVS=5 ARCH=tattn" "$RES_EVAL" \
    "attention vs recurrence at object level -- they tie on dice (0.6459/0.6460), and precision is the actual claim"

# ---- evaluation: second wave (kind: eval3) ----------------------------------
# Deliberately NOT in eval2. Each of these only becomes worth 3 h of GPU once
# the pair above has landed, and two of them are already arguing against
# themselves on dice. Submit with: submit_all.sh eval3 --submit
job eval3 eval_tattn_fuse0_base \
    scripts/eval/run_eval.sh "RUN=$TA_FUSE0 GROUP=temporal_k10 K_PREVS=5 ARCH=tattn" "$RES_EVAL" \
    "separates architecture from negatives -- only needed if eval_tattn_fuse0_neg1x_ring3 and eval_t5_neg1x_ring3 diverge"
job eval3 eval_tattn_hybrid_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$TA_HYBRID_NEG GROUP=temporal_k10 K_PREVS=5 ARCH=tattn" "$RES_EVAL" \
    "nominal top group-T dice (0.6468) but inside the floor; do attention and recurrence compound at object level?"
job eval3 eval_t5_neg1x_ring10 \
    scripts/eval/run_eval.sh "RUN=$T5_NEG1_R10 GROUP=temporal_k10 K_PREVS=5" "$RES_EVAL" \
    "near vs far field, only meaningful once near-field ring3 is shown to do something"
job eval3 eval_g5_neg3x_ring3 \
    scripts/eval/run_eval.sh "RUN=$G5_NEG3_R3 GROUP=geo_k10 K_PREVS=5" "$RES_EVAL" \
    "3:1 negatives cost 12.5 h and lost 0.017 dice; eval only to confirm more background is not the answer"

# ---- evaluation: the 2026-08-11 batch (kind: eval4) -------------------------
# THE ONLY QUEUED WORK IN THIS FILE, and docs/RESULTS.md's top priority. Four
# jobs, ~3 h each, one per trained-but-unscored run:
#
#     submit_all.sh eval4 --submit
#
# These four are the entire 2026-08-11 batch. Every one of them has best.pt and
# NO object-level number, which after finding 2 in docs/RESULTS.md means we
# currently know nothing about them: dice put the batch inside the noise floor,
# and dice is the metric that ranked the project's best scene-level model LAST.
#
# Scene list: geo_k10's 18-scene test split, the same list the six geo models in
# docs/RESULTS.md section 3 were scored on. That is what makes these numbers
# rank against `$G5_NEG1_R3` (F1 0.712) and `$G5_NEG3_R3` (0.724) rather than
# just against each other. K_PREVS tracks the CHECKPOINT, so K_PREVS=5 on a _k10
# group is expected and run_eval.sh only notes it; ARCH=single ignores K_PREVS
# entirely and forces --k_prevs 0.
#
# ARCH=tattn is REQUIRED for the hybrid, not optional. It carries a ConvLSTM
# cell and would match the convlstm entry of the factory if evaluated as
# ARCH=convlstm -- run_eval.sh:70-77 orders tattn first precisely for this.
#
# Two questions, one pair each, and neither is answerable from half a pair:
#
#   attention vs recurrence ON GEO:  G5_TATTN_FUSE0, G5_TATTN_HYBRID
#       read against $G5_NEG1_R3, already scored at F1 0.712. Same partition,
#       same pos_w 4, same 1:1 near-field negatives, same optimizer -- only the
#       architecture moves. Attention won on temporal by 0.016 (0.642 vs 0.626),
#       which is one pairing on the partition that needed it least; geo is where
#       precision is actually broken, so this is the replication that decides
#       whether "attention is the default architecture" survives.
#
#   does the ARCHITECTURE matter at all:  G5_SINGLE_BASE, G5_SINGLE_NEG
#       the missing half of the 2x2 whose other half is scored. Ring negatives
#       bought the geo_k5 ConvLSTM +0.096 F1; if a plain single-frame U-Net
#       gets the same lift from the same negatives, then the ConvLSTM, the
#       attention block and the hybrid are all buying much less than assumed --
#       which is worth knowing BEFORE any more GPU goes into architectures.
#
# Sizing: RES_EVAL for all four. It is measured on k5/k10 ConvLSTM evals (~57 GB
# host peak) and the tattn arms carry the same 6-frame stack, so it transfers.
# The two single-frame evals will use far LESS -- they load one frame per scene
# instead of six -- but nothing has measured a single-frame eval yet, and the
# scene-reconstruction arrays are frame-independent, so the split is unknown.
# Read the real peak back from the LSF summary and give them their own rung:
#     grep -A3 "Max Memory" logs/*.out
job eval4 eval_tattn_geo_k5_fuse0_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$G5_TATTN_FUSE0 GROUP=geo_k10 K_PREVS=5 ARCH=tattn" "$RES_EVAL" \
    "attention vs recurrence ON GEO -- the head-to-head against \$G5_NEG1_R3 (F1 0.712); the temporal win rests on one pairing"
job eval4 eval_tattn_geo_k5_hybrid_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$G5_TATTN_HYBRID GROUP=geo_k10 K_PREVS=5 ARCH=tattn" "$RES_EVAL" \
    "the hybrid on geo: do attention and recurrence compound where neither alone has fixed precision?"
job eval4 eval_g5_single_posw4_base \
    scripts/eval/run_eval.sh "RUN=$G5_SINGLE_BASE GROUP=geo_k10 ARCH=single" "$RES_EVAL" \
    "the no-negatives corner of the 2x2 -- the object-level floor every temporal claim on geo is measured against"
job eval4 eval_g5_single_neg1x_ring3 \
    scripts/eval/run_eval.sh "RUN=$G5_SINGLE_NEG GROUP=geo_k10 ARCH=single" "$RES_EVAL" \
    "THE control: single-frame with \$G5_NEG1_R3's exact negatives. Near F1 0.712 and the temporal machinery is not what is doing the work"

# ---- evaluation: the clean benchmark (kind: eval5) --------------------------
#
#     submit_all.sh eval5 --submit          # 8 jobs
#     submit_all.sh eval5 --only anchor --submit
#
# THE ONLY QUEUED WORK IN THIS FILE. Eight of the fourteen clean22 runs, scored
# on scenes for the first time. Written 2026-08-19 from the val-curve analysis
# in docs/MODEL_RUNS.md ("The 2026-08-18 batch").
#
# WHY ONLY EIGHT. Fourteen jobs is ~50 GPU-hours and six of them answer a
# question that is already answered or that geo does not depend on. The six
# skipped, each with its reason, are listed at the bottom of this block. If one
# of the eight below produces a surprise, the matching skipped run is the
# cheapest follow-up -- they all still have best.pt.
#
# ---- PROTOCOL, and it is new on three axes ---------------------------------
#
#   GEN=3         the *_clean.json partitions. This is what makes the numbers
#                 mean anything: the generation-2 geo split put the 31.25-31.44
#                 band in train AND in the hold-out (assets/PARTITIONS.md), so
#                 every geo object-level score in docs/PREDICTIONS.md was
#                 measured partly on training ground. These are not comparable
#                 to that table -- they REPLACE it.
#
#   DATA_STRIDE=4 the paper's reconstruction geometry: quarter-patch step, 16
#                 tiles per interior pixel against the 4 every previous number
#                 in this project used.
#
#   PROTOCOL=rth  the paper's RTh, implemented 2026-08-19 (plan section 7(b)
#                 items 1 and 4). Each tile is binarised at 0.5 and the pixel
#                 value is the FRACTION OF OVERLAPPING TILES voting positive --
#                 the paper's Confidence Factor, on {0, 1/16, ..., 1} at stride
#                 4. eval-outputs then sweeps RTh 0.125/0.25/0.375/0.5 under
#                 BOTH of the paper's Intersection Tolerances (ITh 0.7 / b 5 and
#                 the softer ITh 0.5 / b 10) and reports a GT-area-weighted
#                 aggregate alongside the unweighted one.
#
#                 An RTh number is NOT comparable with any recon_th number in
#                 docs/PREDICTIONS.md: one counts tiles, the other cuts a mean
#                 probability. RTh 0.25 means "at least 4 of 16 tiles agreed".
#
#                 Per the paper, RECALL is the primary metric here: full-range
#                 reconstruction scores a great deal of unannotated ground, so
#                 precision partly measures the digitisation rather than the
#                 model.
#
#   AOI           carried by the partition and passed to BOTH stages by
#                 run_eval.sh. Only tiles wholly inside the box are predicted,
#                 and eval-outputs crops to the same box before scoring.
#
# ---- scene lists -----------------------------------------------------------
# Geo on geo_k10's 35-scene test list, temporal on temporal_k10's 22-scene test
# list. Verified 2026-08-19: on generation 3 the k10 test lists are strict
# SUBSETS of the k5 ones on both axes (geo 35/51, temporal 22/27), so the k10
# list is the common ground and a k5 model scored on it is directly comparable
# to a k10 model. That is the same convention the eval2/eval4 batches used, and
# it is why K_PREVS=5 on a _k10 GROUP is expected here, not a mistake --
# run_eval.sh only notes it.
#
# ---- sizing, which is NOT the stride-2 rung --------------------------------
# Measured, not guessed. Tiles actually predicted per scene (AOI-gated,
# computed from sinkholes.geo.grid_window at patch 200x100):
#
#   geo test (South band 31.25-31.40)   stride 2:  2,340    stride 4:   9,360
#   temporal (both frames 31.25-31.75)  stride 2: 3.0k-9.8k stride 4: 12.0k-39.5k
#
# The AOI is what makes stride 4 affordable at all: a whole stride-4 canvas is
# 68,322 tiles, and the geo box keeps 13.7% of it.
#
# HOST MEMORY IS THE BINDING CONSTRAINT, not the GPU. scenes.py:243 loads each
# timestep's WHOLE grid as float32 with no banded streaming (that is section
# 7(b) item 3, also unimplemented), and a stride-4 grid is 5.47 GB per scene per
# timestep against 1.37 GB at stride 2 -- verified on disk. So the input stack
# alone is 6 x 5.47 = 33 GB at k5 and 11 x 5.47 = 60 GB at k10, against the
# ~8 GB it was when the 80 GB rung was measured at a ~57 GB peak.
#
#   k5  stride 4:  ~57 - 8 + 33  = ~82 GB  -> request 112
#   k10 stride 4:  ~57 - 15 + 60 = ~102 GB -> request 144
#
# Walltime scales with tiles at ~29 tiles/s (3.0 h / 18 scenes / 17,177 tiles,
# the pre-AOI stride-2 measurement), plus I/O that is now substantial: 35 scenes
# x 6 timesteps x 5.47 GB is ~1.1 TB read off the mount. Hence the headroom.
#
# ---- RESIZED 2026-08-19 FROM MEASUREMENT -- the estimates above were wrong ---
# The 2026-08-19 batch measured every rung and the k-dependent ones were far too
# small. LSF reports in MiB and treats "112GB" as 114,688 MB, so these compare
# directly:
#
#   g5_convlstm_3x   112,227 / 114,688 MB   97.9%  (died on a missing patch,
#                                                   i.e. BEFORE its true peak)
#   g5_tattn         109,828 / 114,688 MB   95.7%  same
#   g5_tattn_hybrid  104,853 / 114,688 MB   91.4%  killed by owner
#   g10_tattn_hybrid 147,456 / 147,456 MB  100.0%  TERM_MEMLIMIT, on scene 1-2
#   g5_single         28,457 /  40,960 MB   69.4%  many scenes, so trustworthy
#   t5_single         23,161 /  40,960 MB   56.5%
#
# WHY IT IS SO MUCH LARGER THAN THE MODEL ABOVE PREDICTED. scenes.py:280-286
# rebinds the timestep list through a list comprehension:
#     arrays_newest_first = [normalise_phase(p[...]) for p in arrays_newest_first]
# normalise_phase RETURNS A NEW ARRAY (normalise.py:26-39, "the input is not
# modified"), and the old list stays alive until the assignment completes -- so
# both lists coexist and the stack is held TWICE at that moment. With the
# largest stride-4 grid at 394 x 177 x 200 x 100 x 4 B = 5.196 GiB per timestep:
#
#                stack     x2 (the comprehension)   + mask_cur   + residual
#   k5  (T=6)    31.2      62.4                     5.2          42   = 109.6 GiB
#   k10 (T=11)   57.2     114.3                     5.2          42   = 161.5 GiB
#
# The 109.6 GiB for k5 is exactly the 112,227 MB measured, which is what makes
# the 161.5 GiB k10 figure trustworthy -- and it is why a 144 GB k10 request
# died. The 42 GiB residual is everything else (reconstruction canvas,
# confidence maps, LiDAR gate, polygonisation) and is k-independent.
#
# Requests below are those peaks plus ~30%. THE ONE-LINE ALTERNATIVE, if these
# ever fail to schedule: normalising in place, or deleting the old list before
# building the new one, removes the x2 and takes k10 back under 110 GiB. That is
# a change to inference code and is deliberately not bundled with a resize.
#
# T5 IS THE UNMEASURED RUNG. Its only job died in 14 s on a missing patch (58 MB
# peak), so there is no number for it. Its stack is the same size as geo k5's,
# but its AOI is 3.3x the latitude span (31.25-31.75 against 31.25-31.40) and
# its North scenes carry 39.5k tiles against geo's 9,360, which inflates the
# residual term rather than the stack. 160 GB is geo k5's rung plus room for
# that; MEASURE IT and tighten:  grep -A3 "Max Memory" logs/*.out
#                        queue    hostGB vramG walltime
RES_EVAL_S4_G5="long-gpu    144   36  12:00"   # measured 109.6 GiB peak (was 112 -- 98% full)
RES_EVAL_S4_G10="long-gpu   208   48  12:00"   # est 161.5 GiB (was 144 -- died at 100%)
RES_EVAL_S4_T5="long-gpu    160   36  16:00"   # UNMEASURED: geo k5 + larger AOI residual
RES_EVAL_S4_CTRL="long-gpu   48   24   6:00"   # measured 27.8 GiB over many scenes (was 40)

# RETIRED 2026-08-20 -- NONE OF THE EIGHT PATHS BELOW STILL RESOLVE TO WEIGHTS.
# The whole 2026-08-18 batch validated against negatives (reporter.log: "Val
# scored on N positive + N negative"), so best.pt was selected on a curve that
# scores an empty prediction on an empty mask as dice 1.0. Those five run
# directories keep results.csv, curves.png, logs/ and validation/; their
# checkpoints were deleted. The other three (_372694, _372696, _372702) never
# reached outputs/2026-08-18/ at all -- the clean22 tattn arms did not survive.
#
# Five of the eight WERE evaluated before the weights went, and those object
# scores stand in outputs/predictions/clean_*/. Re-submitting this batch now
# fails on a missing best.pt. Point it at positives-only runs instead --
# outputs/2026-08-19/*_valpos -- and give those their own variables.
# Renamed 2026-08-20 to carry the protocol: every clean22 arm validated against
# negatives, so each surviving directory ends _valneg. These five resolve to a
# run directory holding metrics ONLY -- the weights are gone.
CLEAN_G5_CONVLSTM=outputs/2026-08-18/geo_k5_convlstm_ring3_valneg
CLEAN_G5_CONVLSTM_3X=outputs/2026-08-18/geo_k5_convlstm_ring3_3x_valneg
CLEAN_G5_SINGLE=outputs/2026-08-18/geo_k5_single_ring3_valneg
CLEAN_T5_CONVLSTM=outputs/2026-08-18/temporal_k5_convlstm_ring3_valneg
CLEAN_T5_SINGLE=outputs/2026-08-18/temporal_k5_single_ring3_valneg
# These three never reached outputs/2026-08-18/ at all -- the clean22 tattn arms
# did not survive, so there is no directory to rename and nothing to point at.
CLEAN_G5_TATTN=outputs/2026-08-18/geo_k5_tattn_ring3_valneg                # MISSING
CLEAN_G5_HYBRID=outputs/2026-08-18/geo_k5_tattn_hybrid_ring3_valneg        # MISSING
CLEAN_G10_HYBRID=outputs/2026-08-18/geo_k10_tattn_hybrid_ring3_valneg      # MISSING

# JOB_NAME names the OUTPUT DIRECTORY, separately from the LSF -J. outputs/README.md
# fixes the convention at `scenes_geo` / `scenes_temporal` plus a suffix for a
# non-default protocol (`scenes_geo_hann`). This batch is non-default on two
# counts at once -- generation-3 partitions and stride 4 -- so it takes its own
# suffix. Without it each directory would inherit the LSF job name and the eight
# evaluations would be unglobbable as a set.
CLEAN_GEO="GEN=3 GROUP=geo_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_geo_clean_rth"
CLEAN_TEMP="GEN=3 GROUP=temporal_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_clean_rth"

# Order matters, same rule as clean22: the anchor first, so a queue that clears
# one job clears the interpretable one.
job eval5 eval_clean_g5_convlstm_anchor \
    scripts/eval/run_eval.sh "RUN=$CLEAN_G5_CONVLSTM $CLEAN_GEO K_PREVS=5" "$RES_EVAL_S4_G5" \
    "THE anchor -- every other geo row is read against it, and the first honest geo object score the project has"
job eval5 eval_clean_g5_single_control \
    scripts/eval/run_eval.sh "RUN=$CLEAN_G5_SINGLE $CLEAN_GEO ARCH=single" "$RES_EVAL_S4_CTRL" \
    "THE control, and the highest-value job here: on val it MATCHES the anchor on F1/P/R (0.635/0.618/0.653 vs 0.634/0.626/0.643) while losing 0.036 dice. If it matches on scenes too, the recurrence is buying nothing on geo"
job eval5 eval_clean_g5_tattn_hybrid \
    scripts/eval/run_eval.sh "RUN=$CLEAN_G5_HYBRID $CLEAN_GEO K_PREVS=5 ARCH=tattn" "$RES_EVAL_S4_G5" \
    "top of the geo batch on dice (0.7350). ARCH=tattn is REQUIRED -- the hybrid carries a ConvLSTM cell and would match the wrong factory entry as ARCH=convlstm"
job eval5 eval_clean_g5_tattn \
    scripts/eval/run_eval.sh "RUN=$CLEAN_G5_TATTN $CLEAN_GEO K_PREVS=5 ARCH=tattn" "$RES_EVAL_S4_G5" \
    "plain attention. Without it the hybrid's score cannot be attributed -- hybrid = attention + recurrence, so it takes all three arms to say which half moved"
job eval5 eval_clean_g5_convlstm_3x \
    scripts/eval/run_eval.sh "RUN=$CLEAN_G5_CONVLSTM_3X $CLEAN_GEO K_PREVS=5" "$RES_EVAL_S4_G5" \
    "3:1 negatives was THE best model in the project on generation 2 (obj F1 0.724) and that was an object-level-only finding -- dice ranked it LAST. Unretested here, the old recommendation stands on leaked ground"
job eval5 eval_clean_g10_tattn_hybrid \
    scripts/eval/run_eval.sh "RUN=$CLEAN_G10_HYBRID $CLEAN_GEO K_PREVS=10 ARCH=tattn" "$RES_EVAL_S4_G10" \
    "best geo dice overall (0.7378); gives k5-vs-k10 at the winning architecture on the shared 35-scene list"
job eval5 eval_clean_t5_convlstm_anchor \
    scripts/eval/run_eval.sh "RUN=$CLEAN_T5_CONVLSTM $CLEAN_TEMP K_PREVS=5" "$RES_EVAL_S4_T5" \
    "the temporal anchor -- the axis whose val F1 did NOT move between generations, so its object score is the control on the whole geo story"
job eval5 eval_clean_t5_single_control \
    scripts/eval/run_eval.sh "RUN=$CLEAN_T5_SINGLE $CLEAN_TEMP ARCH=single" "$RES_EVAL_S4_CTRL" \
    "the temporal half of the control pair. The geo control alone only licenses a claim about geo-split data; the two axes differ in training pool as well as split rule"

# ---- deliberately NOT submitted, with the reason each ------------------------
# All six keep best.pt under the retention rule (unscored runs are never
# pruned), so any of these is one uncommented line away.
#
#   clean_geo_k10_convlstm_ring3     peaked at epoch 9 of 49 and degraded for 40
#                                    epochs after -- the weakest geo run in the
#                                    batch, and the k10 axis is covered by
#                                    eval_clean_g10_tattn_hybrid above.
#   clean_geo_k10_tattn_ring3        k10 architecture spread is already covered
#                                    by g10_hybrid read against g5_hybrid.
#   clean_temporal_k10_convlstm_ring3 k5-vs-k10 was a dead tie on generation 2
#   clean_temporal_k10_tattn_ring3    (0.6420 vs 0.6427) and temporal is not the
#                                    target axis -- see the geo focus in
#                                    docs/RESULTS.md.
#   clean_temporal_k5_tattn_ring3    attention-vs-recurrence ON TEMPORAL was
#   clean_temporal_k5_tattn_hybrid_ring3  settled on generation 2 (0.642 vs
#                                    0.626, attention ahead at 26% fewer params).
#                                    Geo is where it has never been replicated,
#                                    which is why both geo arms are in and both
#                                    temporal arms are out.

# ---- Hann blending at stitch time (kind: blend) -----------------------------
#
#     submit_all.sh blend --submit                  # the 2 remaining, ~3 h each
#
# WHAT IS BEING TESTED. Every full-scene number in this project was stitched
# with a FLAT average: at stride 2 each pixel is covered by 4 tiles, and all 4
# vote equally -- including the tiles that saw that pixel at their own BORDER,
# where the U-Net predicted it from truncated context and edge padding. Those
# border predictions are the usual source of the spurious blobs that appear
# along tile seams, and seams are everywhere: at 200x100 / stride 2 they fall
# every 100 rows and 50 columns of the canvas.
#
# --blend_type hann weights each tile's contribution by a Hann window
# (sinkholes/inference/reconstruct.py:46,204) and normalises by the accumulated
# weight, so a pixel is scored mostly by the tiles that saw it CENTRALLY and
# barely at all by the tiles that saw it at their edge. The claim is therefore a
# PRECISION claim -- fewer seam artefacts surviving the threshold -- which is
# exactly the axis geo is broken on.
#
# The code has been in the repo since the reconstruction engine was written, is
# covered by tests/test_reconstruction.py:157, and HAS NEVER BEEN RUN: no eval
# script set --blend_type, so BLEND=none is what produced every figure in
# docs/RESULTS.md and docs/MODEL_RUNS.md.
#
# WHY THESE. They are the geo_k5 architecture triple at IDENTICAL training
# settings -- pos_w 4, 1:1 near-field ring-3 negatives, same optimizer, same
# partition -- so only the architecture moves, and all three ALREADY HAVE an
# unblended full-scene eval on the same geo_k10 18-scene test list:
#
#   $G5_TATTN_FUSE0  eval4, 2026-08-12   (R 0.983 @0.125 against P 0.21)
#   $G5_NEG1_R3      eval2, F1 0.712     -- the reference every geo claim uses
#   $G5_SINGLE_NEG   eval4, 2026-08-12   -- the no-temporal-machinery control
#
# That is what makes this batch cheap to interpret: the BLEND=none half of the
# A/B is already on disk, so these jobs complete a paired comparison rather
# than starting one. Run BOTH, not one -- if blending helps only the
# model with the worst precision, that is a different finding from a uniform
# lift, and one job cannot tell those apart.
#
# GAMMA IS PINNED AT 1.0 (a plain Hann window). One factor: blending on/off. A
# gamma sweep sharpens the window further and is the obvious follow-up, but only
# on whichever arm actually moves -- sweeping it now would confound "does
# margin-weighting help" with "how much".
#
# HOW TO READ THE RESULT -- do NOT compare at a single threshold. Unblended
# stitching uses average="uniform", which divides by stride^2 = 4 regardless of
# how many tiles actually contributed, so pixels near the scene border and near
# LiDAR-mask edges are ATTENUATED. Hann divides by the true accumulated weight
# and removes that attenuation, which pushes those confidences up. So a
# fixed-threshold comparison mixes the seam cleanup (precision up) with
# de-attenuation (recall up) and reports their sum as if it were one effect.
# Read the whole 0.125/0.25/0.5/0.7/0.9 sweep eval-outputs writes, and compare
# best-F1 to best-F1 rather than 0.25 to 0.25.
#
# Two things this batch will move that are NOT regressions:
#   - docs/PIPELINE.md:384 pins "464 polygons for 20190205_20190216 (LiDAR
#     gating on, no blending)". Blending changes the polygon count by design.
#   - rescore.sh CANNOT produce these numbers. Blending happens during
#     reconstruction, before _pred.npy is written, so this is a full stage-1
#     re-run and not the cheap re-threshold path.
#
# Sizing: RES_EVAL unchanged. Hann adds one extra float32 canvas (the weight
# accumulator alongside the prediction sum) and one elementwise multiply per
# tile -- a few hundred MB against an 80 GB rung, and no additional GPU work,
# since the network runs on exactly the same tiles either way.
# FIRST ATTEMPT, 2026-08-17 11:30: killed by the directory rename, not by
# anything about blending. The run directories were renamed under the running
# jobs at 11:53 and compatibility symlinks left at the old names; the compute
# node could not traverse them, and both jobs died writing a shapefile
# (pyogrio "Not a directory", scenes.py:299) at 5/18 and 9/18 scenes. Their
# partial output was deleted. The single-frame arm had already finished at
# 11:51 and SURVIVES as predictions/geo_k5_single_ring3/best/scenes_geo_hann --
# it is deliberately no longer listed below, so a resubmit does not repeat 3 h
# of completed work.
#
# The lesson worth keeping: eval-scenes resolves its output path ONCE at
# startup, so renaming a directory an eval is writing into kills the job.
BLEND_HANN="BLEND=hann GAMMA=1.0"

# JOB_NAME names the output DIRECTORY (outputs/README.md's convention:
# scenes_geo + a suffix for a non-default protocol) while the LSF -J name stays
# unique so bjobs and logs/%J.out remain greppable. Both blended results
# therefore land beside their unblended twin as
#   predictions/<model>/best/scenes_geo       <- BLEND=none, already on disk
#   predictions/<model>/best/scenes_geo_hann  <- these jobs
# which is what makes the A/B a diff of two JSONs in one directory.
HANN_JOB="$BLEND_HANN JOB_NAME=scenes_geo_hann"

job blend eval_hann_geo_k5_tattn_ring3 \
    scripts/eval/run_eval.sh "RUN=$G5_TATTN_FUSE0 GROUP=geo_k10 K_PREVS=5 ARCH=tattn $HANN_JOB" "$RES_EVAL" \
    "THE headline: highest recall in the project (0.983) at P 0.21, so almost all its error is ground it should not have flagged -- the most seam artefacts available to remove"
job blend eval_hann_geo_k5_convlstm_ring3 \
    scripts/eval/run_eval.sh "RUN=$G5_NEG1_R3 GROUP=geo_k10 K_PREVS=5 $HANN_JOB" "$RES_EVAL" \
    "the geo REFERENCE (F1 0.712 unblended) -- without it, a moved tattn number cannot be told from a moved protocol"

# ---- positives-only: the paper's protocol (kind: posonly) -------------------
#
#     submit_all.sh posonly --submit                 # all five, ~6 h each
#     submit_all.sh posonly --only tattn_geo_k5_fuse0 --submit   # the headline
#
# The benchmark paper scores ONLY patches that contain subsidence -- "delineate
# subsidence where it is known to be", not "find it anywhere on the map". Every
# number in docs/RESULTS.md is the second question. These five jobs answer the
# first, on models that are already scored, so each one lands beside a number we
# already have rather than on its own (docs/POSITIVES_ONLY_EVAL.md).
#
# READ THE RESULT AS A BASELINE COMPARISON AND NOTHING ELSE. A positives-only
# precision cannot see the false positives a model scatters over the 97.5% of
# the map that holds no subsidence, which is exactly how dice ranked the
# ring-negative runs backwards (finding 2). Never promote a model on it.
#
# Scene list: geo_k10's 18-scene test split for the geo arms and temporal_k10's
# 20 for the temporal one -- the same lists section 3 of RESULTS.md uses, so the
# positives-only number sits directly beside that model's own full-scene one.
# K_PREVS tracks the CHECKPOINT, so K_PREVS=5 on a _k10 group is expected.
#
# The five, and what each is for:
#
#   tattn_geo_k5_fuse0   THE ONE. Object recall 0.983 @0.125 -- the highest in
#       the project -- against precision 0.21. Nearly all of its error is
#       ground it should never have flagged, so this protocol removes the whole
#       failure mode and leaves the question of how well it delineates what is
#       actually there. If any model's number moves, it is this one.
#   tattn_geo_k5_hybrid  its twin at recall 0.980. Attention-vs-recurrence on
#       geo is being decided on ~0.003; stripping the background out of both
#       sides is a fair chance at separating them.
#   g10_old_206375       the widest gap in the project: recall 0.976, precision
#       0.121. The upper bound on what this protocol can flatter.
#   g5_neg3x_ring3       the current best full-scene F1 (0.724). The reference:
#       without it, nothing says whether a good positives-only number means a
#       good model or just an easy protocol.
#   t5_ring10            already at precision 0.726 full-scene, the highest we
#       have. It should move LEAST -- the control that shows the movement in
#       the others is the protocol working rather than an artefact.
job posonly posonly_tattn_geo_k5_fuse0 \
    scripts/eval/run_eval_positives.sh "RUN=$G5_TATTN_FUSE0 GROUP=geo_k10 SPLIT=test K_PREVS=5 ARCH=tattn" "$RES_POSONLY" \
    "THE headline: highest object recall in the project (0.983) at precision 0.21 -- the protocol removes precisely its failure mode"
job posonly posonly_tattn_geo_k5_hybrid \
    scripts/eval/run_eval_positives.sh "RUN=$G5_TATTN_HYBRID GROUP=geo_k10 SPLIT=test K_PREVS=5 ARCH=tattn" "$RES_POSONLY" \
    "its twin at recall 0.980: attention vs recurrence on geo is a 0.003 margin, so remove the background noise from both sides"
job posonly posonly_g10_old_206375 \
    scripts/eval/run_eval_positives.sh "RUN=$G10_OLD GROUP=geo_k10 SPLIT=test K_PREVS=10 ARCH=convlstm" "$RES_POSONLY" \
    "the widest gap in the project (R 0.976 / P 0.121) -- the upper bound on what positives-only can flatter"
job posonly posonly_g5_neg3x_ring3 \
    scripts/eval/run_eval_positives.sh "RUN=$G5_NEG3_R3 GROUP=geo_k10 SPLIT=test K_PREVS=5 ARCH=convlstm" "$RES_POSONLY" \
    "the REFERENCE: best full-scene F1 (0.724). Without it a good positives-only number cannot be told from an easy protocol"
job posonly posonly_t5_ring10 \
    scripts/eval/run_eval_positives.sh "RUN=$T5_NEG1_R10 GROUP=temporal_k10 SPLIT=test K_PREVS=5 ARCH=convlstm" "$RES_POSONLY" \
    "the CONTROL: already at P 0.726 full-scene, so it should move least -- proves the movement elsewhere is the protocol, not an artefact"

# ---- argument parsing -------------------------------------------------------
# --only <text> narrows to jobs whose NAME CONTAINS <text>, so a single job can
# be sent without submitting its whole kind (an eval kind is 4-6 jobs and each
# is ~3 h of long-gpu). Repeatable; matching is a plain substring test, so
# `--only hybrid` is enough for eval_tattn_hybrid_neg1x_ring3.
WANT=all; SUBMIT=no; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    train|tattn|control|clean22|valpos|attnfix|attnpos|pre23|training|eval|eval2|eval3|eval4|eval5|posonly|blend|rescore|all) WANT="$1" ;;
    # Retired 2026-08-20 with negative validation itself. Rejected by name
    # rather than left to select zero jobs, so the mistake is visible.
    valneg)         echo "the 'valneg' batch is retired: validation is positives-only. See the retirement note above its section." >&2; exit 1 ;;
    --submit)       SUBMIT=yes ;;
    --only)         shift; [ $# -gt 0 ] || { echo "--only needs a value" >&2; exit 1; }
                    ONLY+=("$1") ;;
    --only=*)       ONLY+=("${1#--only=}") ;;
    # Print the whole usage block rather than a fixed line range: the range
    # went stale every time a kind was added, and --help silently truncated
    # mid-entry. The block ends at the --only paragraph.
    -h|--help)      sed -n '2,/Combine with a kind to narrow within it\./p' "$0"; exit 0 ;;
    *)              echo "unknown argument '$1'" >&2; exit 1 ;;
  esac
  shift
done

# `training` is the combined training selector: the established ConvLSTM batch
# (`train`) plus the temporal-attention batch (`tattn`). Both are empty as of
# 2026-08-12, as is `control` -- every training job this file has ever listed
# has finished and been retired -- so `training` and `control` now select
# nothing. That is the intended state, not a bug.
#
# The eval kinds are kept apart on purpose, since each is ~3 h of long-gpu:
#   eval   the 2026-08-09 backlog -- still pending, all 5 of them
#   eval2  the 2026-08-10 batch, in decisive pairs -- COMPLETED 2026-08-11
#   eval3  the second wave                         -- COMPLETED 2026-08-11
#   eval4  the 2026-08-11 batch                    -- COMPLETED 2026-08-12
#   eval5  the clean benchmark, 8 of 14 runs       -- LIVE, the queued work.
#          GEN=3 + DATA_STRIDE=4, so it shares NO ground with eval/eval2/eval3/
#          eval4: different partitions, different scene lists, different
#          reconstruction geometry. Its numbers replace docs/PREDICTIONS.md
#          rather than extending it.
#   posonly positives-only, the paper's protocol   -- LIVE
#   blend  Hann-blended stitching on the geo_k5 architecture triple -- LIVE.
#          The single-frame arm completed 2026-08-17; 2 jobs remain.
#          Its BLEND=none halves are already on disk (eval2/eval4), so these
#          three complete a paired comparison rather than starting one.
# eval2 and eval3 are kept listed rather than retired because an eval is cheap
# to repeat and its inputs (best.pt) are all still on disk, so re-running one is
# a legitimate act; a re-run overwrites nothing, it writes a new timestamped
# directory under outputs/predictions/<run>/best/. Check that directory before
# sending either kind again -- 6 jobs is ~18 GPU-hours of work already done.
#
# `all` is now 19 eval jobs plus the rescore -- roughly 57 GPU-hours, ten of
# which have already been spent. Name the kind instead.
selected() {
  local kind="$1" name="${2-}"
  # --only, when given, is an AND on top of the kind filter.
  if [ "${#ONLY[@]}" -gt 0 ]; then
    local hit=no pat
    for pat in "${ONLY[@]}"; do
      case "$name" in *"$pat"*) hit=yes ;; esac
    done
    [ "$hit" = yes ] || return 1
  fi
  [ "$WANT" = all ] || [ "$WANT" = "$kind" ] ||
    { [ "$WANT" = training ] && { [ "$kind" = train ] || [ "$kind" = tattn ]; }; }
}

if [ "$SUBMIT" = yes ] && ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found -- run this on the cluster, not the mounted volume." >&2
  exit 1
fi

# ---- preflight: every template and checkpoint must exist ---------------------
fail=0
for i in "${!NAMES[@]}"; do
  selected "${KINDS[$i]}" "${NAMES[$i]}" || continue
  [ -f "${TEMPLATES[$i]}" ] || { echo "missing template ${TEMPLATES[$i]}" >&2; fail=1; }
  case "${OVERRIDES[$i]}" in
    *RUN=*) r=$(sed 's/.*RUN=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            [ -f "$r/checkpoints/best.pt" ] || { echo "missing checkpoint $r/checkpoints/best.pt" >&2; fail=1; } ;;
    # rescore reads saved confidence maps, not a checkpoint. If eval-scenes
    # output was pruned there is nothing to re-score and only a full
    # run_eval.sh can rebuild it -- catch that here, not on a compute node.
    *DIR=*) d=$(sed 's/.*DIR=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            if [ ! -d "$d" ]; then
              echo "missing eval directory $d" >&2; fail=1
            elif [ -z "$(find "$d" -maxdepth 1 -name '*_pred.npy' -print -quit)" ]; then
              echo "no *_pred.npy in $d -- rebuild it with run_eval.sh, not rescore.sh" >&2; fail=1
            fi ;;
  esac
done
[ "$fail" = 0 ] || { echo "preflight failed; nothing submitted" >&2; exit 1; }

# ---- summary table ----------------------------------------------------------
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
    # `if` keeps set -e from aborting the whole batch on one rejection: a bad
    # queue limit on job 7 must not silently cancel jobs 8-12.
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
    echo "failed to submit:"
    printf '  %s\n' "${FAILED[@]}"
  fi
else
  echo "$n job(s) listed. DRY RUN -- nothing was submitted. Add --submit to send them."
fi
