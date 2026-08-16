#!/usr/bin/env bash
# ============================================================================
#  Batch submitter for the planned training and evaluation jobs.
#
#    scripts/submit_all.sh                   # dry run: print the plan, submit nothing
#    scripts/submit_all.sh train             # dry run, training jobs only
#    scripts/submit_all.sh training --submit # submit ConvLSTM + temporal-attention training
#    scripts/submit_all.sh control --submit  # the single-frame U-Net 2x2 control
#    scripts/submit_all.sh valneg  --submit  # the geo_k5 reruns with negative validation
#    scripts/submit_all.sh eval4   --submit  # the 2026-08-11 evals -- START HERE
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
#  Two kinds are live:
#    valneg  the five geo_k5 reruns under the new negative validation set --
#            same five experiments, a val/dice that can finally rank them.
#    eval4   the object-level scores for the 2026-08-11 runs, which
#            docs/RESULTS.md names the top priority.
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
G10_SEED7=$A09/geo-k10_convlstm-h256_seed7_lsf594073
G10_POSW4=$A09/geo-k10_convlstm-h256_posw4_lsf594076
G10_SINGLE=$A09/geo-k10_unet-single_lsf594074
G10_STACK=$A09/geo-k10_unet-stack_lsf594075
T5_POSW1=$A09/temporal-k5_convlstm-h256_posw1_lsf594080
T5_POSW2=$A09/temporal-k5_convlstm-h256_posw2_lsf594078

# The 2026-08-10 batch: all 13 jobs (9 negative-sampling + 4 attention) ran to
# completion, every one of them stopping on early-stopping patience rather than
# the walltime, and are archived under the same retention rule -- except that
# best.pt is kept for ALL 13 because none has been scored at object level yet.
# Deleting any of them now would cost a retrain to run the eval below.
#
# Note the naming scheme changed with this batch: it leads with the
# ARCHITECTURE ("convlstm_geo_k5_...") where earlier runs led with the group
# ("geo-k5_convlstm-..."). run_eval.sh parses both.
A10=outputs/2026-08-10
T5_POSW4=$A10/convlstm_temporal_k5_h256_posw4_60e_2026-08-10_15h41_lsf_82458
T5_NEG1_R3=$A10/convlstm_temporal_k5_h256_posw4_neg1x_ring3_60e_2026-08-10_16h53_lsf_96429
T5_NEG1_R10=$A10/convlstm_temporal_k5_h256_posw4_neg1x_ring10_60e_2026-08-10_16h53_lsf_96432
T5_NEG3_R3=$A10/convlstm_temporal_k5_h256_posw4_neg3x_ring3_60e_2026-08-10_16h53_lsf_96433
G5_POSW4=$A10/convlstm_geo_k5_h256_posw4_60e_2026-08-10_16h00_lsf_82459
G5_NEG1_R3=$A10/convlstm_geo_k5_h256_posw4_neg1x_ring3_60e_2026-08-10_16h54_lsf_96436
G5_NEG1_R10=$A10/convlstm_geo_k5_h256_posw4_neg1x_ring10_60e_2026-08-10_16h54_lsf_96437
G5_NEG3_R3=$A10/convlstm_geo_k5_h256_posw4_neg3x_ring3_60e_2026-08-10_16h54_lsf_96439
G10_NEG1_R3=$A10/convlstm_geo_k10_h256_posw4_neg1x_ring3_60e_2026-08-10_17h08_lsf_96441
TA_FUSE0=$A10/tattn_temporal_k5_d256_fuse0_posw4_60e_2026-08-10_16h55_lsf_96442
TA_FUSE0_NEG=$A10/tattn_temporal_k5_d256_fuse0_posw4_neg1x_ring3_60e_2026-08-10_16h54_lsf_96445
TA_FUSE2_NEG=$A10/tattn_temporal_k5_d256_fuse2_posw4_neg1x_ring3_60e_2026-08-10_16h57_lsf_96447
TA_HYBRID_NEG=$A10/tattn_temporal_k5_d256_fuse0_recur-convlstm_posw4_neg1x_ring3_60e_2026-08-10_16h57_lsf_96449

# The 2026-08-11 batch: the two geo attention arms (LSF 493314/493315) and the
# two single-frame controls (LSF 501433/501434). All four finished cleanly --
# three on early-stopping patience, one on the full 60 epochs, none killed --
# and all four kept best.pt. They are still LOOSE AT THE TOP LEVEL of outputs/,
# not under a dated folder, so there is no $A11 prefix to hang them on; the
# paths below are deliberately written out in full and must be re-pointed if
# docs/MODEL_RUNS.md's housekeeping (move into outputs/2026-08-11/) is done.
#
# None of the four has an object-level score -- they are exactly the four runs
# the eval4 table below exists to score, so best.pt must not be pruned from any
# of them until that has run.
A11=outputs
G5_TATTN_FUSE0=$A11/tattn_geo_k5_d256_fuse0_posw4_neg1x_ring3_60e_2026-08-11_14h15_lsf_493314
G5_TATTN_HYBRID=$A11/tattn_geo_k5_d256_fuse0_recur-convlstm_posw4_neg1x_ring3_60e_2026-08-11_14h15_lsf_493315
G5_SINGLE_BASE=$A11/unet_single_geo_k5_posw4_60e_2026-08-11_14h36_lsf_501433
G5_SINGLE_NEG=$A11/unet_single_geo_k5_posw4_neg1x_ring3_60e_2026-08-11_14h36_lsf_501434

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
RESCORE_G10=outputs/predictions/convlstm_geo_k10_lsf206375/best/test_geo_k10_08_09_12h38
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
# WHAT IS BROKEN. Every negative-sampling result in this file was produced
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
VALNEG="VAL_NEGS=yes"

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
job valneg convlstm_geo_k5_h256_posw4_neg1x_ring3_valneg1x_60e \
    scripts/train/train_convlstm.sh "$G5_BASE $NEG_R3_1X $VALNEG" "$RES_NEG1_G5" \
    "THE reference arm: 1:1 near-field negatives, now scored on a val set that can see them"
job valneg convlstm_geo_k5_h256_posw4_neg3x_ring3_valneg1x_60e \
    scripts/train/train_convlstm.sh "$G5_BASE $NEG_R3_3X $VALNEG" "$RES_NEG3_G5" \
    "how much is enough: 3:1 against 1:1, on a metric that can tell them apart"
job valneg convlstm_geo_k5_h256_posw4_neg1x_ring10_valneg1x_60e \
    scripts/train/train_convlstm.sh "$G5_BASE $NEG_R10_1X $VALNEG" "$RES_NEG1_G5" \
    "near vs far field: same count, drawn from a 10-cell annulus"
job valneg tattn_geo_k5_d256_fuse0_posw4_neg1x_ring3_valneg1x_60e \
    scripts/train/train_tattn.sh "$TATTN_G5 $NEG_R3_1X $VALNEG" "$RES_TATTN_G5_VN" \
    "attention vs recurrence on geo -- a precision claim, finally on a precision-sensitive metric"
job valneg unet_single_geo_k5_posw4_neg1x_ring3_valneg1x_60e \
    scripts/train/train_control.sh "$CONTROL_G5 $NEG_R3_1X $VALNEG" "$RES_CTRL_G5_NEG" \
    "the control: does the temporal machinery buy anything once background is scored?"

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

# ---- argument parsing -------------------------------------------------------
# --only <text> narrows to jobs whose NAME CONTAINS <text>, so a single job can
# be sent without submitting its whole kind (an eval kind is 4-6 jobs and each
# is ~3 h of long-gpu). Repeatable; matching is a plain substring test, so
# `--only hybrid` is enough for eval_tattn_hybrid_neg1x_ring3.
WANT=all; SUBMIT=no; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    train|tattn|control|valneg|training|eval|eval2|eval3|eval4|rescore|all) WANT="$1" ;;
    --submit)       SUBMIT=yes ;;
    --only)         shift; [ $# -gt 0 ] || { echo "--only needs a value" >&2; exit 1; }
                    ONLY+=("$1") ;;
    --only=*)       ONLY+=("${1#--only=}") ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
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
#   eval4  the 2026-08-11 batch -- THE LIVE ONE, and RESULTS.md's top priority
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
