#!/usr/bin/env bash
# ============================================================================
#  submit_all.sh -- the shield in front of bsub.
#
#      scripts/submit_all.sh                     # list every live job (DRY RUN)
#      scripts/submit_all.sh eval6ref            # list one kind
#      scripts/submit_all.sh long500             # 500-epoch temporal rerun
#      scripts/submit_all.sh ctx50               # the two 300x200 context arms
#      scripts/submit_all.sh evallong200         # score long200 on scenes
#      scripts/submit_all.sh eval6ref --submit   # send it
#      scripts/submit_all.sh --only g5_single    # narrow by name, AND-ed with the kind
#
#  DRY RUN IS THE DEFAULT. Nothing reaches LSF without --submit.
#
#  WHAT THIS FILE IS. A table of jobs that have NEVER RUN, plus the preflight
#  that stops a doomed one before it reaches a compute node. It is deliberately
#  not the project's notebook: a batch that has run is recorded in
#  docs/EXPERIMENTS.md and REMOVED from here, so `--submit` can never silently
#  repeat finished GPU-hours. That failure has happened -- the 2026-08-10 batch
#  sat here after completing and `train --submit` would have retrained ~60
#  GPU-hours of finished work.
#
#  To recover any retired batch's exact job definitions:
#      git show d708fba:scripts/submit_all.sh
#
#  WHAT THE PREFLIGHT CATCHES, on the login node rather than 20 minutes into a
#  GPU allocation:
#    * a template that does not exist;
#    * RUN= whose checkpoints/best.pt is gone (deleted or moved by a tidy);
#    * DIR= for a re-score whose *_pred.npy were pruned -- re-scoring reads
#      exactly those, and only a full run_eval.sh can rebuild them.
#
#  RESOURCES are (queue, host GB, GPU GB, walltime). They are measured, not
#  guessed; the comment on each line says what it was measured from.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f scripts/train/train_convlstm.sh ] || [ ! -d assets ]; then
  echo "run this from the repo (cd to it); assets/ and scripts/train/ must be here" >&2
  exit 1
fi

KINDS=(); NAMES=(); TEMPLATES=(); OVERRIDES=(); RESOURCES=(); WHYS=()
job() { KINDS+=("$1"); NAMES+=("$2"); TEMPLATES+=("$3"); OVERRIDES+=("$4"); RESOURCES+=("$5"); WHYS+=("$6"); }

# ---- resources --------------------------------------------------------------
RES_EVAL_S4_G5="long-gpu    144   36  12:00"   # measured 109.6 GiB peak
RES_EVAL_S4_G10="long-gpu   208   48  12:00"   # est 161.5 GiB (144 died at 100%)
RES_EVAL_S4_T5="long-gpu    160   36  16:00"   # geo k5 + larger AOI residual
RES_EVAL_S4_CTRL="long-gpu   48   24   6:00"   # measured 27.8 GiB, single-frame

# ---- eval6ref: the anchors the nineteen are read against --------------------
#
#     submit_all.sh eval6ref --submit     # 4 jobs
#
# Four positives-only runs of 2026-08-19, still carrying best.pt. They are not
# part of the nineteen; they are what makes the nineteen readable.
#
# WHY THEY MATTER. The attnpos batch shipped no single-frame arm and only one
# ConvLSTM (docs/EXPERIMENTS.md), so every one of its rows currently has to be
# read against a pre23 comparator -- across a training-archive boundary. These
# four close that gap with evaluations only, no retraining.
#
# geo_k5_single_ring3_valpos IS UNFINISHED -- killed at epoch 93 of 100 with no
# completion line. Its best.pt is from epoch 59 and the seven lost epochs ran at
# lr 1.95e-08 against a val/dice flat in the fourth decimal, so the checkpoint
# is not in question; only the log line is missing. It is scoreable as it stands.
R_G10_CONVLSTM=outputs/2026-08-19/geo_k10_convlstm_ring3_valpos
R_G5_CONVLSTM=outputs/2026-08-19/geo_k5_convlstm_ring3_valpos
R_G5_SINGLE=outputs/2026-08-19/geo_k5_single_ring3_valpos
R_T5_CONVLSTM=outputs/2026-08-19/temporal_k5_convlstm_ring3_valpos

EVAL6_GEO="GEN=3 GROUP=geo_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_geo_clean_rth"
EVAL6_TEMP="GEN=3 GROUP=temporal_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_clean_rth"

job eval6ref eval6ref_g5_single \
    scripts/eval/run_eval.sh "RUN=$R_G5_SINGLE $EVAL6_GEO ARCH=single" "$RES_EVAL_S4_CTRL" \
    "the highest-value job of the four: on the patch curve this model TOPS its geo batch on both dice and val/F1 while its twin is 0.076 obj F1 behind. One of those two readings is wrong"
job eval6ref eval6ref_g5_convlstm \
    scripts/eval/run_eval.sh "RUN=$R_G5_CONVLSTM $EVAL6_GEO K_PREVS=5" "$RES_EVAL_S4_G5" \
    "THE geo anchor under the current protocol -- every geo row in eval6 is read against this one"
job eval6ref eval6ref_g10_convlstm \
    scripts/eval/run_eval.sh "RUN=$R_G10_CONVLSTM $EVAL6_GEO K_PREVS=10" "$RES_EVAL_S4_G10" \
    "the k10 non-attention reference, so eval6's two geo_k10 attention arms have a same-protocol baseline at their own depth"
job eval6ref eval6ref_t5_convlstm \
    scripts/eval/run_eval.sh "RUN=$R_T5_CONVLSTM $EVAL6_TEMP K_PREVS=5" "$RES_EVAL_S4_T5" \
    "the temporal anchor, completing the pair for every temporal row in eval6"

# ---- ampfix: re-baselining on the corrected tree ----------------------------
#
#     submit_all.sh ampfix --submit          # 3 jobs
#     submit_all.sh ampfix --only=30e --submit
#
# WHY THESE REPLACE long500 AND THE FIRST ctx50 PAIR. Two things changed under
# this batch, and both change what a number means:
#
#   1. AMP gradient clipping was corrected (ab23b07). clip_grad_norm_ used to
#      run BEFORE grad_scaler unscaled, so it saw gradients multiplied by the
#      scale factor and renormalised nearly every step instead of thresholding
#      at 1.0. Every run in the project up to 25c4dc5 trained that way.
#      check_config_compatible now REFUSES to resume such a checkpoint, so the
#      earlier long500/ctx50 executions could not have been continued here.
#   2. Single-frame patch normalisation was corrected to frame-v2. ConvLSTM is
#      temporal and is NOT affected by that one -- which is precisely why these
#      arms isolate the clipping change cleanly.
#
# NOTHING HERE IS COMPARABLE WITH 0.780, with long200, or with anything in
# RESULTS.md. This batch re-establishes the temporal baseline on corrected
# numerics; the old figures stay valid for the tree that produced them.
#
# THE 500-EPOCH ARM IS DROPPED. evallong200 (LSF 580743) has now scored long200
# on the eval6 protocol: object F1 0.7529 against the baseline's 0.7797 on the
# convention RESULTS.md 9 quotes (ith0.7_b5, threshold 0.25, mean) -- -0.0268,
# more than four times the +/-0.006 noise floor, in the WRONG direction. The
# premise for spending 500 epochs was that long200's rising dice curve had more
# to give; the object-level score says the extra epochs cost accuracy rather
# than buying it. 200 with LR_PATIENCE=20 is long200's own schedule and is the
# right length to re-measure first.
#
# grad/norm, grad/clip% and amp/scale are now written to results.csv every
# epoch (6bc1181). On the corrected tree grad/clip% should be a real, mostly
# SMALL number; if it comes back near 100% the clip is still normalising and
# these runs need reading again before anything is concluded from them.
AMPFIX_T5="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
AMPFIX_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
# lrscan settled this: momentum 0.999 is DEAD on corrected clipping (LSF 636709,
# nan from epoch 1), and 0.9 was the best of the five survivors -- 0.666 in 8
# epochs, above long200's 0.6622 over 200, and the only arm still clipping (12.6%)
# at epoch 8 rather than starved to 0%.
AMPFIX_OPT="LR=1e-5 MOMENTUM=0.9 LR_PATIENCE=20"
AMPFIX_CTX="CTX_MY=50 CTX_MX=50 BATCH=64 ACCUM=2"

# Plain 200x100, long200's config exactly: LR 1e-5, LR_PATIENCE 20, early
# stopping off. The ONLY difference from long200 is the corrected clipping,
# which is what makes it the control for the other two.
RES_AMPFIX_PLAIN="long-gpu    90   36  18:00"   # 3m31s/epoch measured -> ~11h45m, one allocation

# 300x200 context: 3x the pixels, so ~10m33s/epoch off the same measurement.
# Host memory, not VRAM, is the risk -- the lazy loader PLAN_LARGE_CONTEXT.md 3
# calls for was never built, so samples are still fully materialised.
RES_AMPFIX_CTX_LONG="long-gpu   180   56  18:00"   # 200 ep = ~35h10m, ~2 requeues via --resume auto
RES_AMPFIX_CTX_SHORT="long-gpu   180   56  18:00"  # 30 ep = ~5h20m, one allocation

job ampfix ampfix_t5_convlstm_ring3_200e \
    scripts/train/train_convlstm.sh "$AMPFIX_T5 $AMPFIX_NEG $AMPFIX_OPT EPOCHS=200 PATIENCE=0" "$RES_AMPFIX_PLAIN" \
    "the control: long200's exact config on corrected clipping -- the number every other arm here is read against"
job ampfix ampfix_ctx50_t5_convlstm_ring3_200e \
    scripts/train/train_convlstm.sh "$AMPFIX_T5 $AMPFIX_NEG $AMPFIX_CTX $AMPFIX_OPT EPOCHS=200 PATIENCE=0" "$RES_AMPFIX_CTX_LONG" \
    "the matched context arm: identical to the control except 300x200 of context and the batch split that pays for it"
job ampfix ampfix_ctx50_t5_convlstm_ring3_30e \
    scripts/train/train_convlstm.sh "$AMPFIX_T5 $AMPFIX_NEG $AMPFIX_CTX LR=1e-5 MOMENTUM=0.9 LR_PATIENCE=5 EPOCHS=30 PATIENCE=0" "$RES_AMPFIX_CTX_SHORT" \
    "the fast read: 30 epochs at LR_PATIENCE=5, enough to see the corrected clip% and whether context moves the early curve at all"

# THE SINGLE-FRAME FLOOR, AND THE ONE ARM THAT CHANGES ON BOTH COUNTS.
# Every ConvLSTM arm above is affected by the clipping fix ALONE, because a
# temporal stack always had real frames to normalise over. This one is affected
# by the frame-v2 normalisation fix as well: a single 200x100 patch has no
# channel axis, so normalise_channels looped over ROWS and decided the
# "radians or already 0-1?" question 200 times per patch, once per row of 100
# pixels -- which means one patch could be part-converted and part-not.
#
# That makes the recorded single-vs-recurrent gap (RESULTS.md 9: +0.047
# temporal, +0.038 geo) UNSAFE TO QUOTE: some of it may be the U-Net running on
# mangled input rather than recurrence being worth anything. This arm is what
# re-measures that gap honestly, so it is worth more than either context arm.
#
# Config is temporal_k5_pre2023_single_ring3's (which scored 0.733), moved onto
# the control's schedule: LR_PATIENCE=20 and 200 epochs with early stopping off,
# matching ampfix_t5_convlstm_ring3_200e exactly so the two are readable against
# each other. The original ran at LR_PATIENCE=5 (the code default; the template
# never exposed the knob until now) and early-stopped at 87/100.
#
# 2m13s/epoch measured from that run (87 epochs in 3h13m), so 200 is ~7h25m in
# one allocation. Host memory is its 13G peak with headroom.
RES_AMPFIX_SINGLE="long-gpu    24   24  12:00"   # 2m13s/epoch measured -> ~7h25m

# THE NO-NEGATIVES TWIN OF THE 30e ARM, for reading the loss curve itself.
# Identical to ampfix_ctx50_t5_convlstm_ring3_30e in every respect except
# RING_NEGS=no, which drops --add_ring_negatives and leaves the train split
# positives-only.
#
# THE ANSWER IS PARTLY KNOWN AND IT IS A TRAP. RESULTS.md 1 already measured
# this on temporal_k5: baseline P 0.743 / R 0.451 / F1 0.561 against ring3's
# 0.664 / 0.591 / 0.626. And on geo the ordering INVERTS between the two
# metrics -- baseline takes the best dice (0.6558, 1st) and third F1, while
# ring3 3:1 takes the worst dice (0.6386, last) and the best F1 (0.724, 1st).
# So expect this arm to post a LOWER training loss and a BETTER dice while
# being the WORSE detector. A lower loss here is not a better model; it is a
# smaller, easier training set.
#
# WHY RUN IT ANYWAY: val is positives-only in both arms (VAL_NEGS defaults to
# no), so val/dice stays directly comparable while train/loss does not -- and
# the train/bce + train/dice columns are new. They say WHICH term the negatives
# move, which none of the historical runs recorded. If the gap is nearly all
# BCE, the negatives are costing background calibration; if it is the Dice
# term, they are changing what the model is willing to predict at all.
job ampfix ampfix_ctx50_t5_convlstm_base_30e \
    scripts/train/train_convlstm.sh "$AMPFIX_T5 RING_NEGS=no $AMPFIX_CTX LR=1e-5 MOMENTUM=0.9 LR_PATIENCE=5 EPOCHS=30 PATIENCE=0" "$RES_AMPFIX_CTX_SHORT" \
    "the 30e arm with the ring negatives removed: reads which HALF of the loss they move, against a val split that stays positives-only in both"

job ampfix ampfix_t5_single_ring3_200e \
    scripts/train/train_control.sh "ARCH=single $AMPFIX_T5 $AMPFIX_NEG $AMPFIX_OPT EPOCHS=200 PATIENCE=0" "$RES_AMPFIX_SINGLE" \
    "the single-frame floor on corrected data: the only arm fixed on BOTH counts, and the one that says whether recurrence really buys what RESULTS.md 9 claims"

# ---- evalampfix: object-level scores for the corrected-optimiser batch ------
#
#     submit_all.sh evalampfix --submit
#
# THE PROTOCOL IS eval6's, EXACTLY, because that is what produced the 0.7797
# these have to be read against: GEN=3, GROUP=temporal_k10 for the shared
# 20-scene list, DATA_STRIDE=4, PROTOCOL=rth, K_PREVS=5 naming the CHECKPOINT's
# depth. Identical to the evallong200 row, which is what makes all of them
# comparable.
#
# CONTEXT MODELS NEED NO EXTRA FLAG. scenes.py:292 loads the checkpoint before
# scenes.py:294 reads the margin, so io_geometry sets args.context_margin and
# the ctx50 patch tree is selected automatically. That only became true with
# ab23b07; before it, evaluating a context model crashed in reconstruct_scene.
# These are therefore the first scene evaluations of a large-context model.
#
# READ THEM AGAINST 0.7797, NOT AGAINST EACH OTHER'S DICE. val/dice ranked
# long200 above the baseline and the object score reversed it (0.7529 vs
# 0.7797); RESULTS.md 2 says that is the rule, not the exception.
#
# 208G because a ctx50 scene holds 3x the input pixels; evallong200 peaked at
# 110G on plain patches in 71 minutes.
RES_EVAL_AMPFIX="long-gpu   208   36  16:00"

# WAS BLOCKED, NOW RUNNABLE. EVAL6_TEMP sets DATA_STRIDE=4 and the context tree
# existed only at strpp2, so this used to fail in seconds on FileNotFoundError
# (LSF 680921/680926). data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned was
# built on 2026-09-08 by scripts/data/make_context_stride4_patches.sh (LSF
# 203155, 942 GiB, 32 min).
#
# THAT TREE IS A SUBSET -- 61 of 437 interferograms, not the whole grid. It is
# exactly the 20 scenes this partition scores plus their 5-previous chains, and
# it is sufficient for THIS row and no other: GEN=3 + GROUP=temporal_k10 +
# SPLIT=test + K_PREVS=5 + MIN_POS=150, which is what EVAL6_TEMP resolves to.
#
# CHANGE ANY OF THOSE FIVE AND YOU MUST REBUILD FIRST. A different partition,
# split, depth or min_positives asks for interferograms the tree does not hold,
# and eval-scenes DOES NOT FAIL on a missing predecessor: scenes.py:354 filters
# prevs by os.path.exists and then pads the short chain with the CURRENT frame,
# so the run completes and reports a plausible, wrong number. Re-run the builder
# (it derives the id set from the partition and is resumable) and let its final
# check pass before submitting.
job evalampfix evalampfix_base30 \
    scripts/eval/run_eval.sh "RUN=outputs/2026-09-07/ampfix_ctx50_t5_convlstm_base_30e $EVAL6_TEMP K_PREVS=5" "$RES_EVAL_AMPFIX" \
    "the no-ring-negatives context arm, complete at 30/30 -- and the first object-level score of any large-context model"

# ARCH=single, NOT the convlstm default of run_eval.sh:89. This checkpoint is a
# plain U-Net and build_from_checkpoint refuses the mismatch outright (LSF
# 680392/680924: "checkpoint looks like 'unet' but 'convlstm_unet' was
# requested"), which is the check doing its job.
job evalampfix evalampfix_single \
    scripts/eval/run_eval.sh "RUN=outputs/2026-09-07/ampfix_t5_single_ring3_200e $EVAL6_TEMP ARCH=single K_PREVS=5" "$RES_EVAL_AMPFIX" \
    "the single-frame floor on corrected data: the number that says what recurrence is really worth, against RESULTS.md 9's confounded +0.047"

# WAS BLOCKED, NOW RUNNABLE. EVAL6_TEMP sets DATA_STRIDE=4 and the context tree
# existed only at strpp2, so this used to fail in seconds on FileNotFoundError
# (LSF 680921/680926). data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned was
# built on 2026-09-08 by scripts/data/make_context_stride4_patches.sh (LSF
# 203155, 942 GiB, 32 min).
#
# THAT TREE IS A SUBSET -- 61 of 437 interferograms, not the whole grid. It is
# exactly the 20 scenes this partition scores plus their 5-previous chains, and
# it is sufficient for THIS row and no other: GEN=3 + GROUP=temporal_k10 +
# SPLIT=test + K_PREVS=5 + MIN_POS=150, which is what EVAL6_TEMP resolves to.
#
# CHANGE ANY OF THOSE FIVE AND YOU MUST REBUILD FIRST. A different partition,
# split, depth or min_positives asks for interferograms the tree does not hold,
# and eval-scenes DOES NOT FAIL on a missing predecessor: scenes.py:354 filters
# prevs by os.path.exists and then pads the short chain with the CURRENT frame,
# so the run completes and reports a plausible, wrong number. Re-run the builder
# (it derives the id set from the partition and is resumable) and let its final
# check pass before submitting.
job evalampfix evalampfix_ctx30 \
    scripts/eval/run_eval.sh "RUN=outputs/2026-09-07/ampfix_ctx50_t5_convlstm_ring3_30e $EVAL6_TEMP K_PREVS=5" "$RES_EVAL_AMPFIX" \
    "the ring-negative context arm -- its twin against base30, the pair that says what ring negatives buy at object level"

# ---- evalctx: the context comparison, run entirely at DATA_STRIDE=2 ---------
#
#     submit_all.sh evalctx --submit       # 3 jobs
#
# WHY STRIDE 2 AND NOT eval6's 4. The context patch tree exists ONLY at strpp2
# (1.7 TB, LSF 640831); there is no ctx50 tree at strpp4, which is what killed
# LSF 680921/680926 in seconds. Building one would cost ~6.8 TB against 9.8 TB
# free. Stride 2 costs three evaluations instead.
#
# THIS BATCH IS SELF-CONTAINED and shares no number with RESULTS.md. It differs
# from eval6 on BOTH axes -- stride 2 rather than 4, and prob rather than rth --
# so the 0.7797 is not a reference for anything here. evalctx_baseline re-scores
# that same checkpoint under these exact settings to supply the only reference
# the three arms are allowed to use.
#
# READ IT AS: baseline (legacy tree, no context) vs ctx30 (context + ring
# negatives) vs base30 (context, no ring negatives). The first pair says whether
# 300x200 of context buys anything; the second says what ring negatives buy once
# context is present.
#
# The baseline checkpoint predates ab23b07 and cannot be RESUMED here, but it
# evaluates fine: build_from_checkpoint defaults its data_contract to
# legacy-row-v1 and strips it before load_state_dict, and nothing in scenes.py
# gates on it. It is a ConvLSTM, so ARCH stays at run_eval.sh's default.
# PROTOCOL=prob, NOT rth. run_eval.sh:241 refuses rth at any stride but 4, and
# it is right to: under rth the pixel value is the fraction of SIXTEEN
# overlapping tiles voting positive, and the paper's 0.125/0.25/0.5 literally
# mean 2/4/8 of 16. At stride 2 there are four tiles and those thresholds stop
# meaning anything. LSF 688878/688879/688880 died on exactly that check.
# prob -- the mean predicted probability over the covering tiles, cut at
# probability thresholds -- carries no such dependence and is what every
# number before 2026-08-19 used.
EVAL_S2_TEMP="GEN=3 GROUP=temporal_k10 DATA_STRIDE=2 PROTOCOL=prob JOB_NAME=scenes_temporal_clean_prob_s2"

# Stride 2 is a quarter of stride 4's tiles, so well inside evallong200's
# measured 110G/71min even with the context tree's 3x pixels.
RES_EVAL_S2="long-gpu   160   36  12:00"

job evalctx evalctx_baseline \
    scripts/eval/run_eval.sh "RUN=outputs/2026-08-20/temporal_k5_pre2023_convlstm_ring3 $EVAL_S2_TEMP K_PREVS=5" "$RES_EVAL_S2" \
    "the reference the other two are read against: the 0.780 checkpoint re-scored at stride 2, because a stride-4 number cannot be compared with these"
job evalctx evalctx_ctx30 \
    scripts/eval/run_eval.sh "RUN=outputs/2026-09-07/ampfix_ctx50_t5_convlstm_ring3_30e $EVAL_S2_TEMP K_PREVS=5" "$RES_EVAL_S2" \
    "300x200 of context with ring negatives -- the first object-level score of a large-context model in this project"
job evalctx evalctx_base30 \
    scripts/eval/run_eval.sh "RUN=outputs/2026-09-07/ampfix_ctx50_t5_convlstm_base_30e $EVAL_S2_TEMP K_PREVS=5" "$RES_EVAL_S2" \
    "the same context arm without ring negatives: what they buy at object level, where RESULTS.md 1 says dice cannot see it"

# ---- reg: stop the model memorising its training set -----------------------
#
#     submit_all.sh reg --submit             # 6 jobs
#     submit_all.sh reg --only=wd --submit
#
# THE PROBLEM. Every corrected-optimiser arm peaks early and then decays while
# train/loss keeps falling -- ampfix_ctx50_ring3 reached train/loss 0.045 by
# epoch 22 with val/F1 peaking at 14, and ampfix_t5_single peaked at val/F1
# epoch 20 and spent the next 70 epochs drifting down. That is memorisation,
# and no amount of extra epochs fixes it.
#
# WHY IT WAS INEVITABLE. weight_decay was hardcoded at 1e-8 (train.py, before
# it was exposed) -- four orders of magnitude below the usual 1e-4..1e-2, i.e.
# effectively none. And there was no augmentation anywhere in the data path.
# The model had 43M parameters, no regularisation, and one look at each patch.
#
# READ val/F1, NOT val/dice. RESULTS.md 2 is explicit and this batch is exactly
# the case it warns about: ring negatives are added to the TRAIN split only, so
# validation stays positives-only and dice cannot see the false positives being
# suppressed. On every ring-negative arm so far F1 peaked 8-11 epochs AFTER
# dice did.
#
# ONE LEVER PER ARM, all against ampfix_t5_convlstm_ring3_200e (LSF 644244),
# which is this exact config with none of them. Changing four things at once
# would leave no way to tell which one worked.
REG_BASE="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
REG_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
REG_OPT="LR=1e-5 MOMENTUM=0.9 LR_PATIENCE=20 EPOCHS=100 PATIENCE=0"

# 100 epochs at long200's measured 3m31s is ~5h50m; the newer nodes have been
# running this config at ~1m22s, so 12:00 covers either.
#
# long-gpu, and it has to be: at LSF 644244's measured 5m23s/epoch these arms
# are 8h58m plus the ~47min of dataset assembly -- ~9h45m against short-gpu's
# hard cap of under 8:00 (run_eval.sh). The fast-node figure above is not
# something a submission can count on drawing. The dropscan arms below ARE on
# short-gpu because 20 epochs is ~2h35m even on the slow node.
RES_REG="long-gpu    90   36  12:00"

job reg reg_wd1e4 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $REG_OPT WEIGHT_DECAY=1e-4" "$RES_REG" \
    "weight decay 1e-8 -> 1e-4: the setting that was hardcoded out of reach, at four orders of magnitude below normal"
job reg reg_aughv \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $REG_OPT AUGMENT=hv" "$RES_REG" \
    "random horizontal and vertical flips on the train split: roughly 4x the effective data, and the standard answer to memorisation"
job reg reg_wd_aug \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $REG_OPT WEIGHT_DECAY=1e-4 AUGMENT=hv" "$RES_REG" \
    "both together -- worth its own arm because they regularise different things and may not simply add"
job reg reg_ring10 \
    scripts/train/train_convlstm.sh "$REG_BASE RING_NEGS=yes NEG_RING_OUTER=10 NEG_PER_POS=1.0 $REG_OPT" "$RES_REG" \
    "far negatives instead of near: RESULTS.md 1 measured F1 0.698 vs ring3's 0.626 on this partition, the largest gain the project has recorded, and no current arm uses it"
job reg reg_drop01 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $REG_OPT DROPOUT_BOTTLENECK=0.1" "$RES_REG" \
    "dropout on the ConvLSTM's recurrent summary: the one tensor the whole sequence is compressed into, and the only site where dropout cannot manufacture frame-to-frame change"
job reg reg_drop02 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $REG_OPT DROPOUT_BOTTLENECK=0.2" "$RES_REG" \
    "the same lever at twice the rate -- 0.1 vs 0.2 is the whole question, since a 12x6 bottleneck has only so many channels to spare"

# ---- dropscan: does bottleneck dropout do anything, before spending 100e ---
#
#     submit_all.sh dropscan --submit        # 4 jobs, ~2h35m each
#     submit_all.sh dropscan --only=p02 --submit
#
# A 20-EPOCH SCREEN, not an experiment. reg_drop01/reg_drop02 below are the
# real 100-epoch arms; this settles which rate is worth one of those slots
# first, at a quarter of the cost. Do not read an object-level claim out of it.
#
# WHY 20 EPOCHS IS ENOUGH TO SEE IT. On the control (LSF 644244, whose config
# this reproduces exactly) val/loss bottoms at epoch 10 and val/F1 peaks at 14,
# and by epoch 20 train/loss has fallen to 0.3598 against a val/loss of 0.6717
# -- a gap of +0.31, from -0.29 at epoch 1. That gap IS the overfitting.
# Twenty epochs contains the turn, the peak and six epochs of the decay, and
# cost the control 1h46m.
#
# WHAT TO READ, in this order:
#   1. WHERE val/loss BOTTOMS, and whether it is still climbing at 20. This is
#      the number the batch exists for and the only one 20 epochs measures
#      cleanly. Dropout working = the bottom moves later and the climb flattens.
#   2. the train/val loss gap at epoch 20, against the control's +0.31
#      (train 0.3598, val 0.6717). Smaller is the point of the whole batch.
#   3. val/F1 LAST, and only its shape. RESULTS.md (2) is explicit that patch
#      dice ranks these models backwards, +-0.006 is noise, and object-level
#      F1 decides -- which none of these runs measure.
#
# THE TRAP. Dropout may push the val/F1 peak PAST epoch 20, in which case an
# arm that is working looks worse at the cutoff. That is exactly why (1) is the
# read and not the max. If two rates tie on (1), raise EPOCHS to 30 rather than
# guessing from a truncated peak.
#
# LR_PATIENCE=5 -- the code default (train.py) and what every run before
# 2026-09-03 used, rather than REG_OPT's 20. CHECK THE lr COLUMN BEFORE
# COMPARING ARMS: at this patience the plateau scheduler CAN fire inside 20
# epochs, and an arm whose lr was halved at epoch 12 is not on the same footing
# as one that held 1e-5 to the end. On the control's own val/dice curve the
# first cut lands exactly at epoch 20 (best 0.6647 at 14, then six bad epochs),
# and scheduler.step runs before the row is written, so it shows in row 20's lr
# cell while changing no epoch inside the run. The dropout arms have different
# curves and may cut earlier; if two arms cut at different epochs, say so in
# the row rather than reading the tail as a clean comparison.
#
# lr_patience is an ADVISORY resume key (resume.py), not a STRICT one, so this
# differs from the control's 20 without making anything unresumable -- and
# since the first cut lands on the last epoch, p00 still reproduces LSF
# 644244's twenty epochs of training exactly.
#
# p00 IS THE CONTROL, AND IT WILL NOT MATCH LSF 644244. --dropout_bottleneck
# 0.0 builds nn.Identity and is proven bit-identical to the pre-2026-09-08
# MODEL (tests/test_dropout_bottleneck.py), but the OBJECTIVE changed under
# these arms: the region term used to reduce in fp16 and saturated to a
# constant 1.0 with zero gradient above ~2.56% mean predicted probability at
# batch 128, so 644244 optimised BCE alone for its first several epochs
# (train/dice pinned at exactly 1.0 for epochs 1-3). See REGION_LOSS_DTYPE in
# training/losses.py. These arms also run at BATCH=64 ACCUM=2 rather than 644244's
# 128, which changes what BatchNorm sees. All four arms share both changes, so
# the batch is internally consistent; p00 against 644244 is NOT a clean single
# comparison any more and should be read as orientation, not measurement.
DROPSCAN_OPT="LR=1e-5 MOMENTUM=0.9 LR_PATIENCE=5 EPOCHS=20 PATIENCE=0 BATCH=64 ACCUM=2"

# SIZED TO WHAT THIS CONFIG MEASURABLY USES, not to what the 100-epoch arms
# ask for. Eight completed runs of this exact config (ampfix x2, long200,
# long500, lrscan x4 -- all batch 128, t5_pre2023, ring3, plain 200x100) peaked
# at 43.5-46.6 GB of host memory against a 90 GB request. 64 GB is 37% over the
# worst of them. RES_LONG200_T5 says as much in its own comment: "90G host by
# request (est peak 44G)".
#
# gmem 24G, AND THAT IS WHY BATCH=64 ACCUM=2 IS IN DROPSCAN_OPT. Nothing in
# this repo has ever been granted more than 24G on short-gpu -- run_probe.sh
# asks 24G, rescore.sh 8G, and every 36G+ job goes to long-gpu -- so a 36G
# exclusive request here plausibly matches no card in the queue and simply
# waits. Halving the micro-batch halves the encoder activations, which dominate
# because the shared encoder runs on B*T = 768 samples at batch 128; batch 128
# is known to fit 36G, so batch 64 lands near 18-20G and 24G has headroom.
# ACCUM=2 keeps the EFFECTIVE batch at 128, so the optimiser sees the same
# gradient it would have. BatchNorm does not: it sees one 64-sample micro-batch
# at a time (train.py --accum_steps). That is a real difference from batch 128
# and the reason it is applied to EVERY arm in both scan batches at once --
# within-batch comparisons stay clean, cross-run ones against LSF 644244 now
# differ in micro-batch as well as objective.
#
# 20 epochs at the control's measured 5m23s is 1h48m, and 64x2 costs a little
# more per epoch for the same FLOPs; plus the ~47min this partition spends
# assembling the dataset before epoch 1 (LSF 644244: job start 14:33, banner
# 15:19). Call it ~2h45m, so 4:00 keeps real margin -- and 4:00 was already
# accepted by short-gpu, so it is under the queue's hard cap.
RES_DROPSCAN="short-gpu   64   24   4:00"

job dropscan dropscan_p00 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $DROPSCAN_OPT DROPOUT_BOTTLENECK=0.0" "$RES_DROPSCAN" \
    "the control, and the regression check: 0.0 builds nn.Identity, so these 20 rows must reproduce LSF 644244's first 20"
job dropscan dropscan_p01 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $DROPSCAN_OPT DROPOUT_BOTTLENECK=0.1" "$RES_DROPSCAN" \
    "the gentle rate: 26 of 256 recurrent channels dropped per step, enough to matter if the mechanism works at all"
job dropscan dropscan_p02 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $DROPSCAN_OPT DROPOUT_BOTTLENECK=0.2" "$RES_DROPSCAN" \
    "the usual sweet spot for feature dropout, and the rate reg_drop02 would spend 100 epochs on"
job dropscan dropscan_p03 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $DROPSCAN_OPT DROPOUT_BOTTLENECK=0.3" "$RES_DROPSCAN" \
    "the upper bracket: 77 of 256 channels gone every step on a 12x6 map, where the rate should start costing more signal than it buys -- included so the scan has a losing arm and a shape, not two points"

# ---- lossscan: Jaccard against Dice as the region term ---------------------
#
#     submit_all.sh lossscan --submit        # 3 jobs, ~2h35m each
#     submit_all.sh lossscan --only=jac --submit
#
# THREE JOBS, NOT FOUR. This is a 2x2 -- {dice, jaccard} x {seed 42, seed 7} --
# and dropscan_p00 IS the fourth cell: it is this exact config at SEG_LOSS=dice
# (the template default) with SEED=42 and DROPOUT_BOTTLENECK=0.0. Re-running it
# would burn 2h35m to reproduce a number already on disk.
#
# WHY A SEED REPLICATE AND NOT JUST A/B. dice-vs-jaccard is one comparison with
# no historical reference: every earlier run trained the fp16-saturated
# objective (REGION_LOSS_DTYPE in training/losses.py), so RESULTS.md's +-0.006
# noise floor was measured on a different objective and does not transfer. The
# second seed measures THIS batch's own run-to-run spread, and the dice-jaccard
# gap has to beat it to mean anything. Without that, a difference is a guess.
#
# DROPOUT IS PINNED AT 0.0, explicitly rather than by template default: the
# dropscan batch is the dropout question, this one is the loss question, and an
# arm carrying both levers answers neither.
#
# HOW TO READ IT -- the trap is right here. Both region terms are recorded in
# BOTH arms (train/dice, train/jaccard, val/dice, val/jaccard), whichever is the
# objective. So:
#   1. COMPARE ARMS ON THE SAME COLUMN. val/dice in the jaccard arm against
#      val/dice in the dice arm, or val/jaccard against val/jaccard. Reading
#      the jaccard arm's val/jaccard against the dice arm's val/dice compares
#      two different quantities and will show a large fake difference: J =
#      D/(2-D), so J is ~0.19 below D at D=0.65 before anything has happened.
#   2. train/loss IS NOT COMPARABLE ACROSS ARMS. Jaccard loss exceeds Dice loss
#      at every identical prediction, so the jaccard arm reads higher from
#      epoch 1 for a reason that has nothing to do with how it is training.
#   3. Read where val/loss bottoms and the train/val gap at epoch 20, as in
#      dropscan -- those are within-arm readings and stay valid.
#   4. val/dice and val/jaccard are the per-sample pair; val/IoU is the POOLED
#      Jaccard and is exactly val/F1/(2-val/F1), so it says nothing val/F1 does
#      not. Ignore it here.
LOSSSCAN_OPT="LR=1e-5 MOMENTUM=0.9 LR_PATIENCE=5 EPOCHS=20 PATIENCE=0 DROPOUT_BOTTLENECK=0.0 BATCH=64 ACCUM=2"

# Identical to RES_DROPSCAN, and it has to stay identical: dropscan_p00 is the
# fourth cell of this 2x2, so BATCH, ACCUM and everything else must match or
# the shared cell is not shared. See RES_DROPSCAN for where the numbers come from.
RES_LOSSSCAN="short-gpu   64   24   4:00"

job lossscan lossscan_jac_s42 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $LOSSSCAN_OPT SEG_LOSS=jaccard SEED=42" "$RES_LOSSSCAN" \
    "the arm the batch exists for: the direct twin of dropscan_p00, differing only in the region term"
job lossscan lossscan_dice_s7 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $LOSSSCAN_OPT SEG_LOSS=dice SEED=7" "$RES_LOSSSCAN" \
    "the dice control at a second seed -- with dropscan_p00 this is the noise floor the dice-jaccard gap has to beat"
job lossscan lossscan_jac_s7 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $LOSSSCAN_OPT SEG_LOSS=jaccard SEED=7" "$RES_LOSSSCAN" \
    "the jaccard replicate, so both arms have a spread and the comparison is between two distributions rather than two points"

# ---- lrscan: find a step size the corrected gradients can live with --------
#
#     submit_all.sh lrscan --submit          # 5 jobs, ~30 min each
#
# WHY THIS EXISTS. ampfix (LSF 612826/612827/612828) died identically on both
# the plain and the context arm: epoch 1 healthy (dice 0.469, clip 3.1%, AMP
# scale steady at 65536), then nan on the FIRST step of epoch 2. best.pt from
# epoch 1 is finite but carries a BatchNorm running_var of 1.576e+08 -- the
# activations were already exploding, and the next forward pass overflowed fp16.
#
# It is NOT gradient explosion: median true gradient norm was 0.124 and the clip
# fired on 3.1% of steps. It is STEP SIZE. RMSprop runs at momentum=0.999, which
# amplifies every update by 1/(1-m) = 1000x at steady state. That was survivable
# only while the clipping bug crushed every gradient to a fixed ~1.5e-5 norm.
# With unscaled-v2 clipping the real gradients arrive and the same 1000x
# amplification is far too hot. lr and momentum were co-adapted to the bug.
#
# TWO AXES, because either fixes the step size and they are not equivalent:
# dropping lr scales every update uniformly; dropping momentum shortens the
# window the updates are accumulated over, which also removes the lag that makes
# a 1000x buffer dangerous near a curved minimum. 0.9 is a conventional RMSprop
# momentum and 0 is PyTorch's own default.
#
# READ IT FROM results.csv: an arm is alive if amp/scale stays 65536, the
# non-finite warning never appears, and val/dice climbs. An arm that dies does
# so by epoch 2, so 8 epochs is more than enough to tell.
LRSCAN_BASE="PARTITION=assets/partition_temporal_k5_pre2023.json K_PREVS=5 POS_W=4"
LRSCAN_NEG="RING_NEGS=yes NEG_RING_OUTER=3 NEG_PER_POS=1.0"
LRSCAN_LEN="LR_PATIENCE=20 EPOCHS=8 PATIENCE=0"
RES_LRSCAN="long-gpu    90   36   2:00"   # 8 x 3m31s = ~28m

# THE NEGATIVE CONTROL, and it is expected to FAIL. lr 1e-5 with momentum 0.999
# is exactly what ampfix ran (LSF 612826) before it went nan, so this arm is the
# origin of the grid: every other arm changes one axis away from it.
#
# It is worth a GPU slot for a reason beyond completeness. --momentum was added
# in 3a64c5d with a default of 0.999 to reproduce the value that had been
# hardcoded at train.py:702. This arm is what PROVES that default is faithful.
# If it trains happily, the divergence was never about momentum and the whole
# reading of ampfix is wrong -- so a pass here is the informative outcome, not
# the failure.
#
# Expect: epoch 1 clean (ampfix reached dice 0.469 with a BatchNorm running_var
# already at 1.576e+08), epoch 2 flooded with non-finite gradients and amp/scale
# collapsed to 0, loss nan from epoch 3. It dies well inside the 8.
job lrscan lrscan_lr1e5_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-5 MOMENTUM=0.999" "$RES_LRSCAN" \
    "the negative control: the exact ampfix setting that diverged, re-run to confirm it still does and that --momentum 0.999 reproduces the old hardcoded value"

job lrscan lrscan_lr1e6_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-6 MOMENTUM=0.999" "$RES_LRSCAN" \
    "lr axis: 10x down from the setting that blew up, momentum left at its historical 0.999"
job lrscan lrscan_lr1e7_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-7 MOMENTUM=0.999" "$RES_LRSCAN" \
    "lr axis: 100x down"
job lrscan lrscan_lr1e8_m999 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-8 MOMENTUM=0.999" "$RES_LRSCAN" \
    "lr axis: 1000x down -- roughly the factor momentum contributes, so the floor of this axis"
job lrscan lrscan_lr1e5_m090 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-5 MOMENTUM=0.9" "$RES_LRSCAN" \
    "momentum axis: long200's own lr, momentum 0.999 -> 0.9, so the amplification drops 1000x -> 10x"
job lrscan lrscan_lr1e5_m000 \
    scripts/train/train_convlstm.sh "$LRSCAN_BASE $LRSCAN_NEG $LRSCAN_LEN LR=1e-5 MOMENTUM=0" "$RES_LRSCAN" \
    "momentum axis: long200's own lr with no momentum at all -- PyTorch's RMSprop default, and the cleanest reading of what the gradients alone do"

# ---- momscan: momentum 0.99, the decade lrscan never tried ------------------
#
#     submit_all.sh momscan --submit         # 4 jobs, ~2h35m each
#     submit_all.sh momscan --only=lr1e6 --submit
#
# ON HOLD -- DO NOT SUBMIT WITHOUT READING adamscan FIRST. This batch scans the
# lr/(1-m) ladder, and that ladder is an artefact of RMSprop's uncorrected
# momentum buffer. adamscan below removes the amplification outright rather than
# navigating it, so the two batches answer overlapping questions and adamscan
# answers the more useful half. Kept defined, not deleted: if adamw turns out
# worse than dropscan_p00, this is the next thing to try and it is already
# costed and argued.
#
# WHAT lrscan LEFT OPEN. It settled two things (see AMPFIX_OPT): momentum 0.999
# is dead on corrected clipping, and 0.9 was the best of the five survivors.
# But it only ever moved momentum AT lr 1e-5, and 0.99 -- the decade between the
# dead value and the best one -- was never a cell. The momentum axis is
# therefore two points wide with a factor of 100 between them.
#
# AND IT RAN WITHOUT A SCHEDULE. LRSCAN_LEN is "LR_PATIENCE=20 EPOCHS=8", so the
# scheduler could not fire before the run ended: every lrscan number is a
# FIXED-lr survival probe, not a training curve. These arms use LR_PATIENCE=5
# over 20 epochs against ReduceLROnPlateau on val/dice at --lr_factor 0.5
# (train.py:766), so the schedule can cut at most ~3 times -- a 8x range, not a
# decay to nothing. That is a different question -- "can this step size train",
# not "does this step size survive" -- and it is the reason these are not just
# more lrscan arms.
#
# EXCEPT AT THE FLOOR. --min_lr defaults to 1e-8 (train.py:312) and
# train_convlstm.sh does not expose it, so the lr1e8 arm STARTS at the clamp and
# its scheduler can never fire. That arm is a fixed-lr probe whatever this
# paragraph says -- which is exactly what makes it comparable to
# lrscan_lr1e8_m999, also fixed-lr, and the shared endpoint claimed below.
#
# AN EXACT TWIN OF dropscan_p00, WHICH IS THE CONTROL. Same base, same negatives,
# same 20-epoch LR_PATIENCE=5 schedule, same BATCH=64 x ACCUM=2, and SEED,
# DROPOUT_BOTTLENECK and SEG_LOSS all left at the defaults dropscan_p00 uses
# (42 / 0.0 / dice). LR and MOMENTUM are the only things that move. So the
# control costs no GPU slot here -- it has already run.
#
# IT ALSO INHERITS A NOISE FLOOR, which is what makes a small gap readable:
# dropscan_p00 and lossscan_dice_s7 are that same cell at seeds 42 and 7, so the
# seed-to-seed spread is being measured by the batch already on the machine. Do
# not call a difference real until it clears that spread.
#
# THE lr LADDER. RMSprop momentum amplifies the update by 1/(1-m) at steady
# state, so what matters is roughly lr/(1-m), and 0.99 amplifies 100x against
# 0.9's 10x. Every arm here is below 1e-5:
#
#     arm                  lr      m      lr/(1-m)   vs control
#     dropscan_p00       1e-5    0.9       1.0e-4      1x   (the control)
#     momscan_lr3e6_m99  3e-6    0.99      3.0e-4      3x
#     momscan_lr1e6_m99  1e-6    0.99      1.0e-4      1x   <- matched
#     momscan_lr1e7_m99  1e-7    0.99      1.0e-5      0.1x
#     momscan_lr1e8_m99  1e-8    0.99      0.01e-4     0.01x
#
# lr1e6 is the arm the batch is really for: it puts 0.99 at the SAME effective
# step as the best known setting, so it isolates what the longer momentum window
# does on its own, with the step size held fixed. The other three bracket it one
# decade hot and two cold, which is what says whether 0.99 has any usable range
# at all or only works where it happens to imitate 0.9.
#
# READ IT IN TWO PASSES. First alive or dead, the lrscan reading: amp/scale
# stuck at 65536, no non-finite warning, loss not nan. lrscan's dead arm went at
# epoch 1-2, so anything that reaches epoch 5 is alive. Only then read val/F1
# against dropscan_p00, and only against dropscan_p00 -- these share no setting
# with the 100-epoch reg arms and nothing here is an object-level claim.
MOMSCAN_OPT="LR_PATIENCE=5 EPOCHS=20 PATIENCE=0 BATCH=64 ACCUM=2 MOMENTUM=0.99"

# Identical to RES_DROPSCAN, and it has to stay identical for the same reason
# RES_LOSSSCAN does: dropscan_p00 is the control, and a control that ran on
# different hardware limits is not a control. 20 epochs measured ~2h35m there.
RES_MOMSCAN="short-gpu   64   24   4:00"

job momscan momscan_lr3e6_m99 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $MOMSCAN_OPT LR=3e-6" "$RES_MOMSCAN" \
    "0.99 at 3x the control's effective step: the hot end, and the arm that says whether 0.99 has headroom above the matched setting or only survives at it"
job momscan momscan_lr1e6_m99 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $MOMSCAN_OPT LR=1e-6" "$RES_MOMSCAN" \
    "THE arm of the batch: lr/(1-m) identical to dropscan_p00's, so momentum 0.9 -> 0.99 is the only thing that changes and the longer window is read on its own"
job momscan momscan_lr1e7_m99 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $MOMSCAN_OPT LR=1e-7" "$RES_MOMSCAN" \
    "one decade cold: with LR_PATIENCE=5 live, this is where a too-small step should show as a curve that never gets going rather than one that diverges"
job momscan momscan_lr1e8_m99 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $MOMSCAN_OPT LR=1e-8" "$RES_MOMSCAN" \
    "the floor, and a FIXED-lr arm despite LR_PATIENCE=5: it starts at --min_lr, so its scheduler is inert and it is a true like-for-like against lrscan_lr1e8_m999 at 0.999"

# ---- adamscan: change the optimiser instead of tuning around it -------------
#
#     submit_all.sh adamscan --submit        # 5 jobs, ~2h35m each
#     submit_all.sh adamscan --only=lr1e4 --submit
#
# WHY THIS EXISTS, AND WHY IT SUPERSEDES momscan. ampfix, lrscan and momscan are
# all the same problem: PyTorch's RMSprop keeps a RAW momentum buffer, so at
# steady state it converges to grad/(1-m) and the effective step is lr/(1-m) --
# 1000x at the historical 0.999. lr and momentum were therefore never
# independent knobs, and the whole nan cascade of 2026-09-07 was that product
# being far too large once the AMP clipping bug stopped crushing the gradients.
#
# Adam bias-corrects both moments, so m_hat/sqrt(v_hat) is O(1) whatever beta1
# is and the step is ~lr. The coupling does not need tuning; it is gone. That
# is why this is a scan over lr ALONE, with beta1 left at 0.9 -- under Adam
# beta1 sets the averaging window and not the step size, which is exactly the
# property momscan was trying to establish for RMSprop the hard way.
#
# ADAMW, NOT ADAM. --weight_decay under RMSprop and Adam is L2 folded into the
# gradient, so it is scaled by the same 1/sqrt(v) normalisation as everything
# else and is not really weight decay. AdamW decouples it. That matters here
# because the reg batch is spending six 100-epoch slots on wd 1e-8 -> 1e-4, and
# under adamw that question is at least well posed.
#
# STILL AN EXACT TWIN OF dropscan_p00 in everything else -- same base, same
# negatives, same 20-epoch LR_PATIENCE=5 schedule, same BATCH=64 x ACCUM=2,
# same seed/dropout/loss defaults. So it inherits dropscan_p00 as its control
# and the dropscan_p00-vs-lossscan_dice_s7 seed pair as its noise floor, the
# same way momscan does.
#
# NOTHING HERE IS COMPARABLE WITH ANY EXISTING NUMBER beyond that control.
# Every run in RESULTS.md, MODEL_RUNS.md and PREDICTIONS.md is RMSprop. This is
# the same class of break as changing the stride or the protocol, and
# --optimizer is a STRICT resume key for that reason (resume.py): a run cannot
# be continued across it.
#
# THE lr LADDER. Under adamw the step is ~lr, so the ladder is read directly and
# does NOT go through 1/(1-m). dropscan_p00's rmsprop step is lr/(1-m) =
# 1e-5/0.1 = 1e-4, which is what lr1e4 matches:
#
#     arm                   lr      step     vs dropscan_p00's effective step
#     adamscan_lr3e4      3e-4     3e-4      3x
#     adamscan_lr1e4      1e-4     1e-4      1x   <- matched, THE arm
#     adamscan_lr3e5      3e-5     3e-5      0.3x
#     adamscan_lr1e5      1e-5     1e-5      0.1x  (also the literal rmsprop lr)
#
# READ IT IN TWO PASSES, as with lrscan: first alive or dead (amp/scale pinned
# at 65536, no non-finite warning, loss not nan), then val/F1 against
# dropscan_p00. Adam is expected to be far harder to blow up than the rmsprop
# arms were -- if something still diverges here, the problem was never the
# optimiser and both this batch and momscan have been chasing the wrong thing.
ADAMSCAN_OPT="LR_PATIENCE=5 EPOCHS=20 PATIENCE=0 BATCH=64 ACCUM=2 OPTIM=adamw"

# Identical to RES_DROPSCAN for the same reason RES_LOSSSCAN and RES_MOMSCAN are:
# dropscan_p00 is the control and a control on different limits is not a control.
RES_ADAMSCAN="short-gpu   64   24   4:00"

job adamscan adamscan_lr3e4 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $ADAMSCAN_OPT LR=3e-4" "$RES_ADAMSCAN" \
    "the hot end: 3x the control's effective step, and roughly where Adam is usually run on a U-Net -- if this trains cleanly the rmsprop step size was the whole problem"
job adamscan adamscan_lr1e4 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $ADAMSCAN_OPT LR=1e-4" "$RES_ADAMSCAN" \
    "THE arm of the batch: the same effective step as dropscan_p00, so the optimiser is the only thing that changes and adamw is read against rmsprop at matched step size"
job adamscan adamscan_lr3e5 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $ADAMSCAN_OPT LR=3e-5" "$RES_ADAMSCAN" \
    "one step cold, to bracket the anchor from below and show whether the curve is flat in lr or sharply peaked"
job adamscan adamscan_lr1e5 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $ADAMSCAN_OPT LR=1e-5" "$RES_ADAMSCAN" \
    "the literal rmsprop lr under adamw: a 10x smaller step than the control, and the arm that says how much of dropscan_p00's result was the 1/(1-m) amplification rather than the lr on the label"

# WEIGHT DECAY, and the reason it is one arm rather than a second axis. At the
# historical 1e-8 adamw is indistinguishable from adam -- there is nothing for
# the decoupling to do. This arm puts wd at PyTorch's own AdamW default so the
# batch actually exercises the thing it selected adamw for, at the anchor lr so
# it reads directly against adamscan_lr1e4. If it wins, the reg batch's whole
# wd axis should move to adamw before it spends six 100-epoch slots.
job adamscan adamscan_lr1e4_wd1e2 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $ADAMSCAN_OPT LR=1e-4 WEIGHT_DECAY=1e-2" "$RES_ADAMSCAN" \
    "decoupled weight decay at AdamW's own default against the anchor's 1e-8: the only arm where 'adamw rather than adam' means anything, and a cheap read on the reg batch's premise"

# ---- combo: dropout + adamw at 1e-3, with and without 300x200 context -------
#
#     submit_all.sh combo --submit           # 4 jobs
#     submit_all.sh combo --only=ctx --submit
#     submit_all.sh combo --only=p02 --submit
#
# A 2x2: {dropout 0.0, 0.2} x {plain 200x100, ctx50 300x200}, all four on adamw
# at lr 1e-3, 20 epochs, LR_PATIENCE=5, effective batch 128.
#
# WHAT IT MERGES. Three readings from the 2026-09-07/08 batches, none of which
# has been run against the others:
#
#   * DROPOUT_BOTTLENECK=0.2, from dropscan. All three dropout arms reached a
#     LOWER val/loss minimum than p00 (0.5167 / 0.5225 / 0.5321 against 0.5391),
#     and every minimum landed while lr was still 1e-5 -- p01's first cut is at
#     epoch 15 and p02's at 17, both after their own bottoms -- so the four
#     minima are on the same footing. p02 took the best val/dice (0.6704).
#   * OPTIM=adamw, from adamscan. At matched epochs adamw led the rmsprop
#     control everywhere the ladder was warm: at epoch 9, lr3e4 0.6669 and
#     lr1e4_wd1e2 0.6629 against dropscan_p00's 0.6531.
#   * CTX_MY/CTX_MX=50, from ampfix. The context arms took the highest patch
#     dice in the project (0.6891) and the WORST object-level score of any
#     corrected-tree model (0.6948 against the baseline's 0.7724 on the same
#     prob_s2 protocol, all of it lost recall). Patch dice and object F1
#     disagree on this lever harder than on any other, which is why it is worth
#     one more pair rather than being written off -- but see the warning below.
#
# THE lr IS EXTRAPOLATED. 1e-3 is a full decade above adamscan's hottest arm,
# and adamscan had not finished when this was written -- lr3e4 and lr1e4_wd1e2
# were at epoch 9 of 20, the rest at 6-7. The partial ladder is monotone in lr
# and still rising at the top (lr1e5 0.5949 @5, lr3e5 0.6300 @5, lr1e4 0.6719
# @7, lr3e4 0.6669 @9), but the top two are 0.005 apart at different epochs,
# inside this batch's own seed spread. The ladder does not establish that 1e-3
# is better than 3e-4; it establishes that nothing has turned over yet.
#
# BATCH=128, ACCUM=1 -- AND THIS CHANGES THE CONTROL. The scans ran BATCH=64
# ACCUM=2 because short-gpu never grants more than 24G of gmem. At a true batch
# 128 the optimiser sees the same gradient it did there, but BatchNorm sees one
# 128-sample batch instead of two 64s, which is the same micro-batch break the
# dropscan header warns about in the other direction. So dropscan_p00 IS NO
# LONGER A TWIN OF ANYTHING HERE. combo_p00 is this batch's own control, and
# every dropout reading is within-batch. What batch 128 does buy is the
# micro-batch of ampfix and long200, which is the only reason to prefer it.
#
# THE CONTEXT ARMS KEEP ACCUM=2, AND CANNOT NOT. Measured reserved VRAM:
#
#     plain,  micro-batch  64   13.3 GiB   (dropscan_p02)
#     plain,  micro-batch 128   25.8 GiB   (ampfix_t5_convlstm_ring3_200e, long200)
#     ctx50,  micro-batch  64   38.5 GiB   (ampfix_ctx50_t5_convlstm_ring3_30e)
#     ctx50,  micro-batch 128   ~76 GiB    PROJECTED, and the reason for ACCUM=2
#
# Activations dominate and scale with the MICRO-batch: plain doubles 13.3 ->
# 25.8 across that step, so ctx50 at micro-batch 128 lands near 76 GiB. That is
# above every gmem this project has ever been granted (the ceiling in this file
# is 56G) and needs an 80 GB card to itself. At 48G the context arms fit only at
# micro-batch 64, so they run BATCH=64 ACCUM=2 for an effective 128.
#
# THE CONSEQUENCE, STATED PLAINLY: the ctx-vs-plain comparison differs in
# micro-batch (64 vs 128) as well as in context. The two DROPOUT comparisons are
# clean -- combo_p02 vs combo_p00, and combo_ctx_p02 vs combo_ctx_p00, each a
# within-pair reading at identical batching. The CONTEXT comparison is not, and
# should be read as orientation. Making it clean costs either an 80G card or
# dropping the plain arms back to 64x2.
#
# 48G PUTS ALL FOUR ON long-gpu. Nothing in this repo has ever been granted more
# than 24G on short-gpu; every 36G+ job goes to long-gpu. Host memory is sized
# separately and is the real risk on the context arms: the lazy loader
# PLAN_LARGE_CONTEXT.md 3 calls for was never built, so samples are still fully
# materialised, which is why ampfix asked 180G there against 90G plain.
#
# READ THE BATCH IN THIS ORDER:
#   1. ALIVE OR DEAD, on combo_p00 first -- amp/scale pinned at 65536, no
#      non-finite warning, loss not nan. It is the arm most likely to diverge at
#      1e-3 and the cheapest thing to learn.
#   2. combo_p00 against adamscan_lr3e4: does the lr ladder still rise? Read it
#      knowing the micro-batch differs.
#   3. combo_p02 vs combo_p00, then combo_ctx_p02 vs combo_ctx_p00: does dropout
#      still help at 10x the step, and does the answer survive the context?
#      These are the two clean readings and the reason the batch is a 2x2.
#   4. val/loss bottom and the train/val gap, as in dropscan. Patch dice ranked
#      the context arms FIRST and the object metric ranked them LAST, so nothing
#      here decides the context question -- only a scene evaluation does.
#
# THE CLIP THRESHOLD SHOULD BE INERT AT THIS lr, WHICH IS WHY (3) IS READABLE.
# train.py:829 clips at a hardcoded 1.0 and the scans sat on top of it, so
# grad/clip% swung 6% -> 68% inside one run and the lossscan jaccard arms were
# clipped on 92-98% of steps -- normalised gradient descent at a fixed step,
# which confounds their dice-vs-jaccard gap. adamscan runs the other way: the
# hotter the lr the smaller the norm (lr1e5 median 1.12-1.24, clipped 97-100%;
# lr3e4 median 0.37-0.81, clipped ~0%). CHECK grad/clip% IN ALL FOUR ARMS BEFORE
# TRUSTING (3): if it is high, the pairs carry the same confound and the reading
# is not clean.
#
# WEIGHT DECAY STAYS AT THE 1e-8 DEFAULT, deliberately. adamscan_lr1e4_wd1e2 is
# the arm that tests decoupled decay; carrying it here would put a third lever
# on a 2x2 that already has two.
#
# NOTHING HERE IS COMPARABLE WITH RESULTS.md, MODEL_RUNS.md OR PREDICTIONS.md.
# Every published number is RMSprop, --optimizer is a STRICT resume key
# (resume.py), and these also carry the fp32-v2 region term.
COMBO_OPT="LR_PATIENCE=5 EPOCHS=20 PATIENCE=0 OPTIM=adamw LR=1e-3"
COMBO_PLAIN="BATCH=128 ACCUM=1"
COMBO_CTX="CTX_MY=50 CTX_MX=50 BATCH=64 ACCUM=2"

# 5m25s/epoch measured at batch 128 on the corrected tree (ampfix_t5_convlstm_
# ring3_200e) -> ~1h50m for 20, plus the ~47min this partition spends
# assembling the dataset before epoch 1. 90G host: eight completed runs of the
# plain config peaked at 43.5-46.6 GB.
RES_COMBO_PLAIN="long-gpu    90   48   4:00"

# 3m28s/epoch measured at micro-batch 64 (ampfix_ctx50_t5_convlstm_ring3_30e,
# 14 epochs in 48m31s) -> ~1h10m for 20, but assembly is slower at 3x the
# pixels. 180G host is ampfix's measured figure for this tree, NOT a guess:
# samples are fully materialised until the lazy loader is built.
RES_COMBO_CTX="long-gpu   180   48   4:00"

job combo combo_p02 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $COMBO_OPT $COMBO_PLAIN DROPOUT_BOTTLENECK=0.2" "$RES_COMBO_PLAIN" \
    "THE arm of the batch: dropscan's best rate and adamscan's optimiser at one decade past its hottest rung, on ampfix's micro-batch"
job combo combo_p00 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $COMBO_OPT $COMBO_PLAIN DROPOUT_BOTTLENECK=0.0" "$RES_COMBO_PLAIN" \
    "the control this batch supplies for itself, since batch 128 breaks the twinning with dropscan_p00 -- and the missing top rung of the adamscan ladder"
job combo combo_ctx_p02 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $COMBO_OPT $COMBO_CTX DROPOUT_BOTTLENECK=0.2" "$RES_COMBO_CTX" \
    "the same merged arm with 300x200 of context: the first time dropout, adamw and the context lever have been in one run"
job combo combo_ctx_p00 \
    scripts/train/train_convlstm.sh "$REG_BASE $REG_NEG $COMBO_OPT $COMBO_CTX DROPOUT_BOTTLENECK=0.0" "$RES_COMBO_CTX" \
    "the context arm's own dropout-0 control, so the dropout reading inside the context pair is as clean as the plain pair's"

# ---- th350: the four architectures on the label-quality partition -----------
#
#     submit_all.sh th350 --submit            # 4 jobs
#
# The first batch on assets/partition_temporal_k5_clean_th350x200.json, written
# 2026-09-10 by `make-benchmark-partitions --nonz_th 350 200`: an interferogram
# enters train/val/test only when its WHOLE-SCENE positive-patch count clears
# the per-region threshold. 100/11/20 interferograms against clean's 163/21/27,
# and 35,558 in-window train positives against 42,985.
#
# THE THRESHOLD GATES TARGETS ONLY. A dropped scene is still legitimate history
# and still appears in a kept scene's chain (121 such links survive in the
# file), so this batch trains on fewer supervised dates, NOT on shorter stacks.
# Whole-scene is also deliberate: an in-window count would confuse "badly
# digitised" with "few sinkholes in this AOI band".
#
# WHAT IT ANSWERS. Which architecture is worth carrying once the badly
# digitised scenes are out of the supervision. Arm 2 is what makes the
# THRESHOLD itself readable: it is longreg_ring10's configuration exactly --
# ConvLSTM, ctx50, outer=10, hv, dropout 0.2, 1:1 -- and differs from it in the
# partition and in nothing else.
#
# DROPOUT IS ON ARM 2 ONLY, and not by choice. --dropout_bottleneck names the
# single tensor a ConvLSTM compresses its sequence into; train.py:372 REFUSES
# it on any other architecture rather than ignoring it. The other three arms
# are unmatched to the ConvLSTM on that axis, unavoidably. Say so when the four
# are read against each other.
#
# NEG_PER_POS=1.0 matches every live run -- all seven print
# "train/val/test = 85928/5917/0", and 85,928 = 42,985 x 2. Expect
# "71116/3184/0" here: 0.83x, the partition smaller and the ratio unchanged.
# If a log prints anything else, the partition or the annulus is not what this
# block says it is; stop and read it before trusting the run.
TH350_BASE="PARTITION=assets/partition_temporal_k5_clean_th350x200.json K_PREVS=5 POS_W=4"
TH350_NEG="RING_NEGS=yes NEG_RING_INNER=1 NEG_RING_OUTER=10 NEG_PER_POS=1.0"
TH350_CTX="CTX_MY=50 CTX_MX=50 BATCH=64 ACCUM=2"
TH350_RUN="SCHEDULE=plateau LR_PATIENCE=8 EPOCHS=60 PATIENCE=0"
# No DROPOUT_BOTTLENECK here; arm 2 adds it on its own row and the other three
# cannot take it at all. Mirrors the retired longgrid block's LONGGRID_OPT
# (docs/reference/submit_all_retired_2026-09-17.sh), which existed for this reason.
TH350_OPT="OPTIM=adamw LR=3e-4 WEIGHT_DECAY=1e-2 SEG_LOSS=dice"

# RESOURCES. Every host figure below is a request a LARGER job already ran on,
# so these are CARRIED, not scaled: this batch is 0.83x the samples of the runs
# that measured them. gmem is carried unchanged for the opposite reason --
# activations scale with the MICRO-batch and the geometry, not with the sample
# count, so a smaller partition buys nothing back on VRAM.
RES_TH350_TATTN="long-gpu     256   48  12:00"   # 5m13s/ep measured (LSF 769020) -> ~4h20m at 0.83x
RES_TH350_CONVLSTM="long-gpu  256   48  18:00"   # 11m48s/ep measured at ctx50   -> ~9h50m at 0.83x
RES_TH350_SINGLE="long-gpu     96   36  12:00"   # longsingle_ctx50's own request (LSF 721120)
# NEVER RUN. Attention measured 2.3x faster than the ConvLSTM at ctx50 because
# the recurrence is serialised over T and attention is not; the hybrid pays
# both, so ~14h is the estimate and 18:00 the wall with margin. If it hits
# TERM_RUNLIMIT it is NOT requeued -- resume it with RESUME=<run dir>, because
# RESUME=auto would start it over.
RES_TH350_HYBRID="long-gpu    256   48  18:00"

job th350 th350_tattn_ctx50_neg10 \
    scripts/train/train_tattn.sh "$TH350_BASE $TH350_NEG $TH350_OPT $TH350_RUN $TH350_CTX AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_TH350_TATTN" \
    "attention on the thresholded partition, and the only arm here whose attention can be READ -- but note the probe refuses context checkpoints, so selectivity still has to come from a plain-geometry twin"
job th350 th350_convlstm_ctx50_neg10 \
    scripts/train/train_convlstm.sh "$TH350_BASE $TH350_NEG $TH350_OPT $TH350_RUN $TH350_CTX AUGMENT=hv DROPOUT_BOTTLENECK=0.2" "$RES_TH350_CONVLSTM" \
    "the arm that measures the THRESHOLD rather than an architecture: longreg_ring10's exact configuration, differing only in the partition, so the delta between them is the label-quality cut and nothing else"
job th350 th350_single_ctx50_neg10 \
    scripts/train/train_control.sh "ARCH=single $TH350_BASE $TH350_NEG $TH350_OPT $TH350_RUN $TH350_CTX AUGMENT=hv" "$RES_TH350_SINGLE" \
    "the floor: one frame in, no recurrence and no attention, so the other three are only worth their cost by the margin they beat it -- its ring negatives are still excluded against the union over the 5-chain, which is what keeps it a matched control and not a differently-sampled one"
job th350 th350_hybrid_ctx50_neg10 \
    scripts/train/train_tattn.sh "$TH350_BASE $TH350_NEG $TH350_OPT $TH350_RUN $TH350_CTX AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=convlstm HIDDEN=256" "$RES_TH350_HYBRID" \
    "attention AND recurrence at the bottleneck: the only cell that can say whether the two mechanisms are complementary or redundant, and the one arm in this batch with no measured runtime"

# ---- evalth350: object-level scores for the four th350 arms ------------------
#
#     submit_all.sh evalth350 --submit          # 4 jobs
#     submit_all.sh evalth350 --only=single --submit
#
# The four arms of the th350 batch above, all complete at 60/60 (LSF
# 816217/816218/816219/816220), scored on the ONE protocol their partition
# permits.
#
# WHAT val/dice ALREADY SAYS, AND WHY IT IS NOT ENOUGH. Best checkpoint per arm:
#     hybrid    0.7065 @ ep 9    convlstm  0.7035 @ ep 15
#     tattn     0.6970 @ ep 15   single    0.6859 @ ep 16
# Two things in that table matter more than the ordering. The top three sit
# inside 0.0095 of each other against the 0.011 run-to-run spread longreg's
# header establishes -- at dice they are ONE result, not three, and only the
# single-frame floor is separated (0.021 below the best). And EVERY ARM PEAKS
# BEFORE EPOCH 17 AND DECAYS, ending 0.031-0.038 down at 60: the memorisation
# the reg batch was written for is NOT fixed here, and weight decay 1e-2 plus
# dropout on arm 2 did not fix it either. best.pt is early in all four, which is
# what these evaluations score.
#
# RESULTS.md 2 is the standing warning that dice and the object metric disagree
# -- it is what reversed long200 against the baseline -- so no ordering above is
# a result until these four rows land.
#
# THE SCORED LIST IS th350's OWN TEST SPLIT, by request: the test scenes are
# now gated by the SAME 350N/200S threshold as the train and val targets, so a
# scene is scored only if it cleared the bar its supervision did. That is
# GEN=th350 (run_eval.sh resolves it to partition_temporal_k5_testeval_clean_
# th350x200.json), GROUP=temporal_k5, DATA_STRIDE=4, PROTOCOL=rth, K_PREVS=5.
#
# THIS BATCH IS THEREFORE SELF-CONTAINED. It shares NO number with RESULTS.md,
# with the 0.7797 anchor, with evalampfix or with any eval6ref row, because the
# mean is over a different set of scenes. Against the generation-3 list:
#     17 of 20 scenes are common
#      3 are ADDED  (20260304_20260315, 20260326_20260406, 20260406_20260417),
#        which the k10 list cannot hold -- they have no 10-previous chain
#      3 are DROPPED (20250512_20250523, 20250603_20250614, 20250706_20250717),
#        northern scenes at nonz 346/343/339 against the >350 cut
#
# WHAT THE SWAP IS WORTH, MEASURED. Re-averaging the three finished evalampfix
# runs over just the above-threshold scenes moves object F1 by +0.013..+0.024 at
# every RTh, for all three models -- three to four times the +/-0.006 noise
# floor. It is a level shift from an easier scene list, NOT model quality. Do
# not read these four rows against any number produced on the generation-3 list;
# read them against each other, which is what they are for.
#
# THE THRESHOLD ALSO GOES THE OTHER WAY, and it is worth knowing which scenes
# left. The three dropped are the low-precision cluster: on ampfix_ctx30 they
# score 0.628/0.612/0.631 precision where the retained northern scenes score
# 0.82-0.92. Two more of that cluster SURVIVE the cut at nonz 356 and 359 --
# 20250329_20250409 is the worst scene in the set at 0.530 precision and is
# still scored here. 350 is not a clean separation of that band; the gap in the
# data sits between 359 and 496.
#
# LEAKAGE, VERIFIED RATHER THAN ASSUMED -- AND RE-VERIFIED AFTER THE SWAP. The
# scored list is now the th350 family's own, so the check that mattered for the
# generation-3 list (th350 train vs k10_clean test, which was 0) is no longer
# the relevant one. Against partition_temporal_k5_testeval_clean_th350x200.json,
# whose "val" key IS the scored list and equals the parent's "test" key exactly:
#     th350 train (100 intf)   vs  the 20 scored scenes  ->  0
#     th350 val   (11 intf)    vs  the 20 scored scenes  ->  0
#     testeval train (111)     vs  the 20 scored scenes  ->  0
#
# THE PREDECESSORS ARE CLEAN TOO, which is NOT guaranteed by construction and is
# the check worth having. The threshold gates targets only, so a scene may be
# excluded as a target and still be read as history -- the parent th350 file
# carries 121 such links. Of the 58 predecessor ids these 20 scored scenes pull
# in, 0 are th350 train targets and 0 are th350 val targets. No frame this batch
# reads as history was ever a supervised target.
#
# The *_pre23_* prohibition still stands: that list overlaps k5_clean train by 35
# interferograms and th350 is drawn from inside it, so a p23 protocol would leak
# and still return a plausible number.
#
# ---- THE PATCH TREE MUST BE EXTENDED FIRST. THIS IS A HARD BLOCKER ---------
#
# data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned was cut for the
# generation-3 list and holds 62 ids. This list needs 63, and 9 OF THEM ARE
# MISSING -- the 3 added scenes and 6 of their predecessors:
#     20260108_20260119  20260119_20260130  20260130_20260210   (predecessors)
#     20260210_20260221  20260221_20260304  20260315_20260326   (predecessors)
#     20260304_20260315  20260326_20260406  20260406_20260417   (scored scenes)
# All 9 are present in the plain strpp4 parent at 50.7 GB, so the context cut is
# ~152 GB against 8.9 TB free. Build them with the SAME builder, pointed at the
# new partition -- it derives the id set itself and is resumable, so it will cut
# the 9 and leave the 54 it finds:
#
#     PARTITION=assets/partition_temporal_k5_testeval_clean_th350x200.json \
#       bsub < scripts/data/make_context_stride4_patches.sh
#
# THE PATH IS REPO-RELATIVE, WITH NO $REPO ON IT. $REPO is defined INSIDE the
# builder, not in the submitting shell, so writing $REPO/assets/... there
# expands to an empty string and submits PARTITION=/assets/..., which dies at
# the precondition check ("!! partition missing: /assets/...", LSF 176342, 7
# seconds, nothing written). Relative is safe because the builder cd's to the
# repo before anything reads the path.
#
# ITS STEP (4) IS THE GATE. It refuses to report success unless all 63 derived
# ids are on disk. DO NOT SUBMIT THIS BATCH UNTIL THAT PASSES, and the reason is
# that the failure is silent, not loud: scenes.py:354 filters predecessors by
# os.path.exists and then pads the short chain with the CURRENT frame, so three
# scored scenes would evaluate against themselves repeated 6 times and the job
# would finish and report a plausible, wrong number. There is no error to catch.
#
# The 8 ids the old list needed and this one does not are simply left in place;
# they cost disk and nothing else.
#
# MIN_POS=150 IS NOW INERT, and that is a change worth naming. On the
# generation-3 list it did real work, dropping 20250121_20250201 (42 positives)
# and 20260316_20260327 (116) from 22 scenes to 20. Here the 350N/200S threshold
# has already removed everything it would have caught -- the thinnest scene in
# the list is 220 -- so all 20 scenes are scored and the gate never fires. Both
# lists happen to score 20 scenes; they are NOT the same 20.
#
# CONTEXT NEEDS NO FLAG. All four checkpoints are CTX_MY=CTX_MX=50, including the
# single-frame arm, and scenes.py:292 loads the checkpoint before scenes.py:294
# reads the margin -- io_geometry selects the ctx50 tree on its own.
#
# ARCH PER ARM, AND THE HYBRID IS THE TRAP:
#     tattn     ARCH=tattn     RECURRENCE=none
#     convlstm  ARCH=convlstm  (run_eval.sh's default)
#     single    ARCH=single    -> --k_prevs 0; K_PREVS=5 below is inert, and is
#                                 written only to match the evalampfix_single row
#     hybrid    ARCH=tattn     NOT convlstm. It carries a ConvLSTM cell at the
#                                 bottleneck and factory.py:221 orders tattn
#                                 ahead of convlstm for exactly this case; the
#                                 recurrence is read back out of the checkpoint's
#                                 tattn_unet_config blob. ARCH=convlstm here
#                                 rebuilds the wrong network.
#
# WHAT THESE FOUR CANNOT SETTLE YET. th350_convlstm exists to measure the
# THRESHOLD, not an architecture -- it is longreg_ring10's configuration with the
# partition swapped and nothing else. longreg_ring10 WAS KILLED BY OWNER at epoch
# 19 of 60 (LSF 697787, TERM_OWNER, 30157 s); its best.pt is a 19-epoch checkpoint
# and scoring it against a 60-epoch arm measures the epochs, not the labels. The
# threshold delta needs longreg_ring10 finished first. The other three edges --
# the three architectures against each other and against the single-frame floor,
# all four at 60 epochs on one partition -- are fully readable from this batch
# alone.
#
# RESOURCES ARE THE MEASURED ONES, not scaled: an evaluation holds one scene's
# patches at a time, so its footprint follows the scene and the stack depth and
# is indifferent to how large the training partition was.
# GEN=th350 is run_eval.sh's fourth generation, added for this batch: it
# resolves SPLIT=test to partition_temporal_k5_testeval_clean_th350x200.json and
# carries that file's aoi_window into both stages, exactly as generation 3 does.
# GROUP is temporal_k5, NOT k10 -- the threshold family was only ever generated
# at k5, and the three added scenes are precisely the ones k10 has no chain for.
# JOB_NAME is deliberately not scenes_temporal_clean_rth: that directory name
# means the generation-3 list, and outputs/README.md fixes the convention that a
# non-default protocol gets its own suffix. Reusing it would file these under a
# name that claims comparability they do not have.
EVAL_TH350="GEN=th350 GROUP=temporal_k5 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_th350_rth"

RES_EVAL_TH350="long-gpu   208   36  16:00"   # measured 143.0 GB peak / 4875 s (LSF 412393),
                                              # the identical ctx50 + stride-4 + rth + k5 workload.
                                              # 208G is NOT slack: the same job OOM-killed at this
                                              # exact cap three times (233847/262921/276398) before
                                              # the row-streaming reconstruction landed.
# One timestep instead of six, at ctx50's 3x pixels. The plain single-frame arm
# measured 28.0 GB / 3501 s (LSF 688882); 3x its stack puts this near 60-90 GB,
# so 128G is margin over an estimate rather than a measurement -- if it returns a
# peak near the cap, raise it to RES_EVAL_TH350 rather than trimming.
RES_EVAL_TH350_SINGLE="long-gpu   128   36   8:00"

job evalth350 evalth350_tattn \
    scripts/eval/run_eval.sh "RUN=outputs/th350_tattn_ctx50_neg10_2026-09-10_01h16_lsf_816217 $EVAL_TH350 K_PREVS=5 ARCH=tattn" "$RES_EVAL_TH350" \
    "attention alone on the thresholded partition -- best dice 0.6970 at epoch 15, inside the noise floor of both recurrent arms, so the object metric is the only place the three can actually separate"
job evalth350 evalth350_convlstm \
    scripts/eval/run_eval.sh "RUN=outputs/th350_convlstm_ctx50_neg10_2026-09-10_01h16_lsf_816218 $EVAL_TH350 K_PREVS=5" "$RES_EVAL_TH350" \
    "best dice 0.7035 at epoch 15, and the arm that WOULD measure the threshold itself -- but only once longreg_ring10 is finished past its 19/60 kill, since that is the run it differs from in the partition alone"
job evalth350 evalth350_single \
    scripts/eval/run_eval.sh "RUN=outputs/th350_single_ctx50_neg10_2026-09-10_01h16_lsf_816219 $EVAL_TH350 K_PREVS=5 ARCH=single" "$RES_EVAL_TH350_SINGLE" \
    "the floor the other three have to beat to be worth their cost: one frame in, no recurrence and no attention, and the ONLY arm dice separates -- 0.0206 below the best, which is the one gap here wider than the 0.011 noise"
job evalth350 evalth350_hybrid \
    scripts/eval/run_eval.sh "RUN=outputs/th350_hybrid_ctx50_neg10_2026-09-10_01h16_lsf_816220 $EVAL_TH350 K_PREVS=5 ARCH=tattn" "$RES_EVAL_TH350" \
    "attention AND recurrence -- the cell that says whether the two mechanisms are complementary or redundant. Nominal dice leader at 0.7065, but it got there in NINE epochs and leads arm 2 by 0.0030, which is a quarter of the noise floor; ARCH=tattn, never convlstm, or the factory rebuilds the wrong network"

# ---- posw8: the th350 architecture sweep again, at positive weight 8 --------
#
#     submit_all.sh posw8 --only=convlstm --only=hybrid --submit   # what is LEFT
#     submit_all.sh posw8 --submit            # ALL FOUR -- see below first
#
# ---- TWO OF THESE FOUR ARE ALREADY OUT (2026-09-14) ------------------------
#   posw8_single_ctx50_20e   SUBMITTED, running as LSF 182834 (16:10), dir
#                            outputs/posw8_single_ctx50_20e_..._lsf_182834
#   posw8_tattn_ctx50_25e    SUBMITTED, no run directory yet -- expected while
#                            it PENDs on the 256 GB request; confirm with bjobs
#
# A BARE `posw8 --submit` WOULD RESUBMIT BOTH, for ~6 GPU-hours of work already
# in flight. That is the failure this file's header records against the
# 2026-08-10 batch. Use the --only form above until these two land; then move
# all four to docs/EXPERIMENTS.md and DELETE the block, which is the lifecycle
# the header describes and the only thing that makes a bare --submit safe again.
#
# All four th350 architectures at POS_W=8 instead of 4: tattn, convlstm and the
# hybrid for 25 epochs, the single-frame floor for 20. Everything else is
# th350's configuration VERBATIM -- the same partition, ring, context,
# optimiser and schedule strings, REUSED rather than copied so they cannot
# drift. Each arm is therefore a clean single-variable comparison against its
# own th350 twin, and the four together re-ask the architecture question at the
# new weight.
#
# WHAT IT IS READ AGAINST -- th350 at RTh 0.25, ITh 0.7/b5, the 20-scene th350
# list (scenes_temporal_th350_rth):
#     tattn     P 0.874  R 0.707  F1 0.782     hybrid    P 0.881  R 0.680  F1 0.767
#     single    P 0.869  R 0.644  F1 0.740     convlstm  P 0.884  R 0.637  F1 0.741
# tattn leads recall at ALL FOUR reconstruction thresholds (0.725 / 0.707 /
# 0.703 / 0.692) and leads F1; convlstm has the best PRECISION and the worst
# recall, which is exactly the axis pos_w moves, so it is the arm with the most
# to gain and the one most likely to reorder the table.
#
# THE DROPOUT ASYMMETRY IS BACK, AND IT IS NOT A CHOICE. convlstm carries
# DROPOUT_BOTTLENECK=0.2 and the other three carry none, because
# --dropout_bottleneck names the single tensor a ConvLSTM compresses its
# sequence into and train.py:372 RAISES SystemExit on any other architecture
# rather than ignoring it. The hybrid cannot take it either: it is built through
# train_tattn.sh and selects tattn_unet, which that check rejects. So a
# convlstm-vs-anything comparison here is confounded by regularisation, exactly
# as it was in th350. Say so when the four are read against each other. What IS
# clean is each arm against its own th350 twin, since the asymmetry is identical
# on both sides of that comparison.
#
# WHY POS_W IS A REAL LEVER HERE, CHECKED IN THE CODE BEFORE SPENDING THE GPU.
# The objective is bce + dice, and the BCE half is
# nn.BCEWithLogitsLoss(pos_weight=args.pos_w) (train.py:838), so 8 doubles the
# cost of a missed positive against a false alarm. THE TRAP THAT IS NOT HIT:
# losses.py:222 hardcodes pos_w=8.0 and IGNORES --pos_w entirely, but only on
# the treat_nodata_regions branch. That flag is store_true, defaults False, and
# no train template sets it -- verified, 0 occurrences in all three. Had it been
# on, this whole batch would have retrained th350 at its existing weight and
# reported the difference as noise.
#
# ONLY THE BCE HALF MOVES. Dice is unweighted and is half the objective, so 2x
# on BCE is NOT 2x on the gradient the model sees. Expect a real shift in the
# precision/recall balance, not a transformation of it.
#
# THE EPOCH BUDGETS DIFFER AND IT DOES NOT MATTER. Every th350 arm peaked before
# epoch 17 and decayed for the remaining 40+ -- tattn 15, convlstm 15, hybrid 9,
# single 16. SCHEDULE=plateau is ReduceLROnPlateau, which reacts to val/dice and
# NEVER reads args.epochs (only cosine does, via T_max at train.py:806). A
# shorter plateau run is therefore bit-identical to the first N epochs of a
# longer one, so 25 and 20 are two different-sized WINDOWS ON THE SAME
# TRAJECTORY, both well past every observed peak. The extra epochs buy selection
# headroom if pos_w 8 moves a peak later, and cost ~10 min/epoch if it does not.
#
# THERE IS ALSO A FREE RECALL LEVER ALREADY IN HAND, and it is worth spending
# before any GPU. Dropping the reconstruction threshold from RTh 0.25 to 0.125
# buys tattn +0.018 recall for -0.005 precision, and convlstm +0.015 for -0.005
# -- no retraining at all, just re-reading the saved confidence maps, which
# eval-outputs already wrote at all four thresholds. This batch is for what that
# cannot give you.
POSW8_BASE="PARTITION=assets/partition_temporal_k5_clean_th350x200.json K_PREVS=5 POS_W=8"
POSW8_RUN25="SCHEDULE=plateau LR_PATIENCE=8 EPOCHS=25 PATIENCE=0"
POSW8_RUN20="SCHEDULE=plateau LR_PATIENCE=8 EPOCHS=20 PATIENCE=0"
# TH350_NEG / TH350_CTX / TH350_OPT are REUSED, not re-typed: identity with
# th350 is the point of the batch, and a copy could drift while still looking
# right. If th350 is ever retired out of this file these three must be inlined
# here -- under `set -u` their removal fails loudly on the next invocation,
# which is the intended behaviour.

# RESOURCES ARE CARRIED FROM th350's OWN COMPLETED RUNS, same geometry, same
# batch, same partition -- pos_w changes the loss, not the footprint.
#
# ONE LINE FOR ALL THREE TEMPORAL ARMS, because they measured the SAME. Over 60
# epochs: tattn 34729 s (LSF 816217), hybrid 34752 s (816220), convlstm 34217 s
# (816218) -- a 1.6% spread. Peak RSS was 140.6 GB for all three, on a 256 GB
# request. ~570-580 s/epoch puts 25 epochs at ~3h58m-4h01m; wall 8:00 is ~2.0x.
#
# THIS CONTRADICTS th350's OWN HEADER, which expected attention to run ~2.3x
# faster than the ConvLSTM because the recurrence serialises over T and
# attention does not. At ctx50 that did not happen -- the three are the same
# number, and the hybrid pays no measurable penalty for carrying both
# mechanisms. Sizing off the measurement, not the prediction.
RES_POSW8_TEMPORAL="long-gpu   256   48   8:00"
# single (LSF 816219): 33.9 GB peak of a 96 GB request, 22875 s for 60 epochs
#                      -> 381 s/epoch -> ~2h07m at 20. Wall 4:00 is ~1.9x.
RES_POSW8_SINGLE="long-gpu    96   36   4:00"
# The memory figures above are the REQUESTS those jobs actually ran on, not
# their peaks; the peaks are recorded so a future tightening has the evidence
# without another run. Per-epoch extrapolation over-counts slightly -- startup
# and the first data load are a fixed cost paid once, not N times.

job posw8 posw8_tattn_ctx50_25e \
    scripts/train/train_tattn.sh "$POSW8_BASE $TH350_NEG $TH350_OPT $POSW8_RUN25 $TH350_CTX AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_POSW8_TEMPORAL" \
    "the best model in the project at pos_w 8: attention alone, already the recall AND F1 leader at pos_w 4, and the only arm here whose attention can be READ -- though the probe still refuses context checkpoints, so selectivity needs a plain-geometry twin"
job posw8 posw8_convlstm_ctx50_25e \
    scripts/train/train_convlstm.sh "$POSW8_BASE $TH350_NEG $TH350_OPT $POSW8_RUN25 $TH350_CTX AUGMENT=hv DROPOUT_BOTTLENECK=0.2" "$RES_POSW8_TEMPORAL" \
    "the arm with the most to gain: best precision of the four at pos_w 4 (0.884) and the WORST recall (0.637), which is the exact axis the weight moves -- and the only arm that can carry the dropout, so read it against its own th350 twin rather than across the batch"
job posw8 posw8_hybrid_ctx50_25e \
    scripts/train/train_tattn.sh "$POSW8_BASE $TH350_NEG $TH350_OPT $POSW8_RUN25 $TH350_CTX AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=convlstm HIDDEN=256" "$RES_POSW8_TEMPORAL" \
    "attention AND recurrence at pos_w 8: at pos_w 4 the extra recurrence cost -0.027 recall and -0.015 F1 against plain attention, so this says whether that deficit is a property of the architecture or of the weight it was trained at"
job posw8 posw8_single_ctx50_20e \
    scripts/train/train_control.sh "ARCH=single $POSW8_BASE $TH350_NEG $TH350_OPT $POSW8_RUN20 $TH350_CTX AUGMENT=hv" "$RES_POSW8_SINGLE" \
    "the floor at pos_w 8 -- the cheapest arm to run and the one with the most recall headroom to recover, since at pos_w 4 it sits 0.063 below tattn; note train_control.sh ALREADY defaults POS_W to 8, so th350's row was the override and this one is the template's own default made explicit"

# ---- evalposw8: object-level scores for the four pos_w 8 arms ---------------
#
#     submit_all.sh evalposw8 --submit          # 4 jobs
#     submit_all.sh evalposw8 --only=single --submit
#
# The four posw8 arms, all complete (LSF 182833/182850/182851/182834), scored on
# EXACTLY the string the evalth350 rows carry -- $EVAL_TH350, GEN=th350,
# GROUP=temporal_k5, DATA_STRIDE=4, PROTOCOL=rth, K_PREVS=5, the same 20-scene
# th350 list. That identity is the entire point: it is what makes pos_w 4 and
# pos_w 8 a single-variable comparison at the object level.
#
# NO PATCH-TREE WORK THIS TIME. The ctx50 stride-4 tree was extended to 63/63
# for evalth350 (LSF 176814) and this batch reads the same scene list, so it is
# already complete. The preflight below still checks it rather than assuming.
#
# WHAT VALIDATION SAYS, AND WHY IT IS NOT THE ANSWER. val/P and val/R at each
# arm's best.pt, pos_w 4 -> pos_w 8:
#     arm        val/P                    val/R
#     tattn      0.7310 -> 0.7503 (+.019)  0.8375 -> 0.8172 (-.020)
#     convlstm   0.7702 -> 0.7492 (-.021)  0.7987 -> 0.8138 (+.015)
#     hybrid     0.7607 -> 0.7617 (+.001)  0.8274 -> 0.7959 (-.032)
#     single     0.7749 -> 0.7575 (-.017)  0.7833 -> 0.8136 (+.030)
#
# THE WEIGHT DID NOT DO THE SAME THING TO EVERY ARCHITECTURE, and on half of
# them it did the OPPOSITE of what it was raised for. convlstm and single moved
# the intended way -- recall up, precision down, which is what doubling the cost
# of a missed positive is supposed to buy. BOTH ATTENTION ARMS MOVED THE OTHER
# WAY: tattn traded 0.020 of recall for 0.019 of precision, and the hybrid gave
# up 0.032 of recall for essentially nothing (+0.001). If that survives to the
# object level, pos_w is not a recall lever on attention at all, and the arm
# this batch was started for is the one it does not help.
#
# DO NOT CONCLUDE THAT FROM THE TABLE ABOVE. It is patch-level validation P/R on
# 3184 positive patches, and RESULTS.md 2 is the standing warning that it
# disagrees with the object metric -- it is what reversed long200 against the
# baseline. These four rows are what settle it. Note also that val/dice barely
# moved on three of the four (+0.004, +0.000, -0.004); only single gained
# meaningfully (+0.016), so dice cannot separate these at all and the object
# score is doing all the work.
#
# THE EXTRA EPOCHS WERE USED, which is worth recording for the next budget.
# Peaks moved LATER under the heavier weight: tattn 15/60 -> 18/25, convlstm
# 15/60 -> 19/25, hybrid 9/60 -> 17/25, single 16/60 -> 16/20. convlstm's 19 of
# 25 is only six epochs from its ceiling, so that arm is the one that might
# still have been climbing; if its object score lands well and the curve is
# still rising at 25, it is the candidate for a longer rerun.
#
# ARCH PER ARM, AND THE HYBRID IS THE SAME TRAP AS ALWAYS:
#     tattn     ARCH=tattn        convlstm  ARCH=convlstm (the default)
#     single    ARCH=single       hybrid    ARCH=tattn, NEVER convlstm --
#               it carries a ConvLSTM cell at the bottleneck and factory.py:221
#               orders tattn ahead of convlstm for exactly this case.
#
# RESOURCES ARE THE evalth350 MEASUREMENTS, carried because the workload is
# identical -- same scenes, same stride, same protocol, same checkpoint depth.
# Measured on that batch (all four Successfully completed):
#     tattn    144.8 GB peak / 13219 s     convlstm  144.7 GB peak / 12964 s
#     hybrid   144.1 GB peak / 11259 s     single     47.0 GB peak /  4463 s
# The 208 G request is CORRECT and not slack -- the same shape OOM-killed three
# times at exactly that cap before the row-streaming reconstruction landed. The
# walls are carried unchanged: 13219 s against 16:00 is 4.4x, which looks
# generous until you note the same convlstm workload took 4875 s under
# evalampfix (LSF 412393) and 12964 s here -- a 2.7x swing from cluster
# conditions alone. An eval is NOT resumable, so a TERM_RUNLIMIT costs the whole
# run; the margin is deliberate. The single arm's 128 G is now known to be 2.7x
# its real peak and is the one figure worth tightening if scheduling ever bites.
RES_EVAL_POSW8="long-gpu   208   36  16:00"
RES_EVAL_POSW8_SINGLE="long-gpu   128   36   8:00"

job evalposw8 evalposw8_tattn \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_tattn_ctx50_25e_2026-09-14_16h22_lsf_182833 $EVAL_TH350 K_PREVS=5 ARCH=tattn" "$RES_EVAL_POSW8" \
    "the arm the batch was started for: recall leader at pos_w 4 (0.707), and the one whose validation says the heavier weight COST it 0.020 of recall rather than buying any -- this row says whether that is real"
job evalposw8 evalposw8_convlstm \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_convlstm_ctx50_25e_2026-09-14_16h27_lsf_182850 $EVAL_TH350 K_PREVS=5" "$RES_EVAL_POSW8" \
    "the arm with the most to gain and the one validation says actually gained: worst recall of the four at pos_w 4 (0.637), +0.015 val/R here, and still climbing at 19 of 25 epochs"
job evalposw8 evalposw8_hybrid \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_hybrid_ctx50_25e_2026-09-14_16h27_lsf_182851 $EVAL_TH350 K_PREVS=5 ARCH=tattn" "$RES_EVAL_POSW8" \
    "attention and recurrence at pos_w 8 -- it gave up 0.032 of val recall for +0.001 precision, the worst trade of the four, so this row is mostly a check that the object metric agrees before the configuration is dropped"
job evalposw8 evalposw8_single \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_single_ctx50_20e_2026-09-14_16h10_lsf_182834 $EVAL_TH350 K_PREVS=5 ARCH=single" "$RES_EVAL_POSW8_SINGLE" \
    "the floor, and the arm the weight helped most on validation (+0.030 val/R, +0.016 val/dice) -- if that holds at object level the floor moved, which changes what the other three have to beat"

# ---- k10: does a longer history help the attention? --------------------------
#
# ---- ALL THREE HAVE RUN (2026-09-16). DO NOT SUBMIT THIS BATCH AGAIN. -------
#   posw8_tattn_k10_ctx50_30e    DONE 30/30, LSF 254183,
#                                outputs/posw8_tattn_k10_ctx50_30e_2026-09-16_14h01_lsf_254183
#   posw8_single_k10_ctx50_30e   DONE 30/30, LSF 254184,
#                                outputs/posw8_single_k10_ctx50_30e_2026-09-16_14h01_lsf_254184
#   posw8_single_k10_plain_30e   DONE 30/30, LSF 269165,
#                                outputs/posw8_single_k10_plain_30e_2026-09-16_16h00_lsf_269165
# Scored by evalk10 below. Once those land, record the batch in
# docs/EXPERIMENTS.md and delete this block, per the header.
#
# posw8's tattn and single-frame arms again, at k=10 instead of k=5. Everything
# else is posw8's configuration -- TH350_NEG and TH350_OPT are REUSED, as the
# posw8 block reuses them -- except the partition, the depth, the tattn
# micro-batch and the epoch budget, each explained below.
#
# WHY NOW. docs/ATTENTION.md (2026-09-16): the ctx50 k5 tattn runs
# SELECT (~1.8 of 6 frames, 30% of uniform) rather than average, but masking
# every past frame costs them only 0.002-0.004 patch Dice. At k5 the history is
# nearly unused, so a real gain here would be the first evidence it matters.
#
# THE PARTITION. assets/partition_temporal_k10_clean_th350x200.json, written
# 2026-09-16 by the th350 command at k=10:
#     make-benchmark-partitions --axis temporal --k_prevs 10 --nonz_th 350 200 \
#         --suffix clean_th350x200
# 74 / 10 / 17 interferograms and 27,191 train positives, against k5's
# 100 / 11 / 20 and 35,558. Every k10 split is a strict SUBSET of its k5 split.
# The same command at k=5 reproduces the committed k5 file exactly. All 295
# frames the k10 chains need are in the ctx50 stride-2 tree, with masks.
#
# THE CONFOUND. k10 trains on 24% fewer positives than posw8's k5 arms, so k10
# tattn vs posw8_tattn changes depth AND data. The single arm here is the
# matched floor. For depth alone on identical scenes, score the posw8 k5 arms
# on this batch's 17-scene test list, which lies inside their own 20.
#
# BATCH=32 ACCUM=4 ON THE TATTN ARM, not 64x2. The shared encoder runs over
# B*T frames, and posw8_tattn reserved 38.4 GiB at 64x6 (LSF 182833). 64x11
# would need ~70 GiB; 32x11 is ~35 GiB. The effective batch stays 128, and the
# encoder's BatchNorm sees 352 frames per micro-batch against k5's 384. The
# single arm keeps 64x2, identical to posw8_single.
#
# 30 EPOCHS, not 25/20. The smaller partition gives ~24% fewer steps per epoch,
# so posw8_tattn's epoch-18 peak (~10k steps) lands near epoch 23-24 here.
# SCHEDULE=plateau never reads EPOCHS, so extra epochs only add headroom.
#
# RESOURCES ARE ESTIMATES, scaled from posw8's measured runs by T x samples:
#   tattn   host: 140.6 GB at k5 was ~113 GB of samples plus a ~30 GB transient
#           per-interferogram grid load; at k10 that is ~156 + ~60 = ~215 GB of
#           the 256 GB request. TERM_MEMLIMIT is the failure to expect if it
#           is wrong. Time: 679 s/epoch x 1.4 ~ 950 s -> ~8 h for 30; wall 16:00.
#   single  host: 34.3 GB at k5 (LSF 182834), fewer samples here; 96 GB carried.
#           Time: 481 s/epoch x 0.77 ~ 370 s -> ~3 h for 30; wall 6:00.
#
# BEFORE SCORING, the ctx50 stride-4 tree needs 11 more ids for the 17 k10 test
# scenes' 10-chains. A missing one is padded with the current frame, not
# refused, so build them first (resumable, ~190 GB):
#     PARTITION=assets/partition_temporal_k10_testeval_clean_th350x200.json K_PREVS=10 \
#         bsub < scripts/data/make_context_stride4_patches.sh
# LSF 252292 was that command, and it exited in 4 s because this partition did
# not exist yet. It does now.
K10_BASE="PARTITION=assets/partition_temporal_k10_clean_th350x200.json K_PREVS=10 POS_W=8"
K10_RUN30="SCHEDULE=plateau LR_PATIENCE=8 EPOCHS=30 PATIENCE=0"
K10_CTX_TATTN="CTX_MY=50 CTX_MX=50 BATCH=32 ACCUM=4"
RES_K10_TATTN="long-gpu    256   48  16:00"
RES_K10_SINGLE="long-gpu    96   36   6:00"
# THE PLAIN SINGLE ARM, added after the other two went out. With it, the three
# arms separate the two levers on one partition:
#     temporal = tattn_k10_ctx50  - single_k10_ctx50   (history, at ctx50)
#     spatial  = single_k10_ctx50 - single_k10_plain   (context, single frame)
# The grid is NOT complete: there is no tattn_k10_plain, so the spatial effect
# is measured for the single-frame model only, and whether context helps
# attention by the same amount is not measured.
#
# MICRO-BATCH 64x2, NOT 128x1, although nothing forces it at 200x100: context
# must be the ONLY change against single_k10_ctx50, and BatchNorm sees one
# micro-batch at a time. longsingle_plain (LSF 721118) went out at 128x1 and
# carries that confound on its edge -- see the retired longgrid block in
# docs/reference/submit_all_retired_2026-09-17.sh.
#
# Resources scaled from longsingle_plain (LSF 721118, plain single, clean k5,
# ~86k samples): 29.2 GB host, 8.3 GiB VRAM reserved at 128, 211 s/epoch.
# At ~54k samples: ~135 s/epoch -> ~1h10m for 30. Its own request is carried.
RES_K10_SINGLE_PLAIN="long-gpu    48   24   4:00"

job k10 posw8_tattn_k10_ctx50_30e \
    scripts/train/train_tattn.sh "$K10_BASE $TH350_NEG $TH350_OPT $K10_RUN30 $K10_CTX_TATTN AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_K10_TATTN" \
    "posw8_tattn at twice the history: the k5 run selects but barely uses its past frames, so this says whether more of them give the attention something worth choosing"
job k10 posw8_single_k10_ctx50_30e \
    scripts/train/train_control.sh "ARCH=single $K10_BASE $TH350_NEG $TH350_OPT $K10_RUN30 $TH350_CTX AUGMENT=hv" "$RES_K10_SINGLE" \
    "the matched floor on the smaller k10 partition -- K_PREVS=10 so its ring negatives are excluded against the same 10-chain union the tattn arm uses"
job k10 posw8_single_k10_plain_30e \
    scripts/train/train_control.sh "ARCH=single $K10_BASE $TH350_NEG $TH350_OPT $K10_RUN30 BATCH=64 ACCUM=2 AUGMENT=hv" "$RES_K10_SINGLE_PLAIN" \
    "single_k10_ctx50 without the 50 px margin: against it, the spatial-context contribution; against tattn_k10_ctx50 through it, the temporal one"

# ---- k10plain: the fourth cell of the k10 grid ------------------------------
#
# ---- HAS RUN (2026-09-17). DO NOT SUBMIT AGAIN. ------------------------------
#   posw8_tattn_k10_plain_30e    DONE 30/30, LSF 316673, best.pt from epoch 29
#                                (still rising at the end), 82.6 GB host peak,
#                                11.5 GiB VRAM reserved, 20504 s.
#                                outputs/posw8_tattn_k10_plain_30e_2026-09-16_23h47_lsf_316673
#   Scored by evalk10plain below.
#
# Its own kind because the k10 batch above has fully run and must not be
# resubmitted. With this cell the k10 runs form a complete 2x2:
#                  plain 200x100                  ctx50 300x200
#   single     posw8_single_k10_plain_30e     posw8_single_k10_ctx50_30e
#   tattn      posw8_tattn_k10_plain_30e <-   posw8_tattn_k10_ctx50_30e
# so context is measured for BOTH architectures, the temporal edge exists at
# both geometries, and whether context and history add or overlap becomes
# readable. The longgrid square asks the same of k5 on the clean partition.
#
# Everything is posw8_tattn_k10_ctx50_30e's configuration minus CTX_MY/CTX_MX,
# REUSING K10_BASE, TH350_NEG, TH350_OPT and K10_RUN30.
#
# MICRO-BATCH 32x4, NOT 64x2, although plain would fit 64 easily. It matches
# the tattn ctx50 cell, so the tattn CONTEXT edge is single-lever. The tattn-vs-
# single edge at plain then carries the same 32-vs-64 micro-batch difference as
# the one at ctx50 -- identical on both geometries, so the two temporal edges
# stay comparable with each other. (A tattn and a single model cannot be matched
# on BatchNorm anyway: the tattn encoder normalises over B*T = 352 frames.)
#
# RESOURCES, scaled from longgrid_tattn_plain (LSF 811232; plain tattn k5, 64x2,
# 85,928 samples): 13.2 GiB VRAM reserved, 65.8 GB host, 533 s/epoch.
#   VRAM   32x11 frames against 64x6 -> ~12 GiB; gmem 24G.
#   host   ~48 GB of k10 samples + ~7 GB val and masks + ~30 GB transient grid
#          load (11 frames) -> ~85 GB. The same model predicted the ctx50 cell
#          at 215 GB against its measured 206 GB. 146G is an existing tier.
#   wall   1.16x the frames of the k5 run -> ~620 s/epoch, ~5-6 h for 30 at the
#          smaller micro-batch; 12:00.
#
# SCORE IT WITH evalk10's STRING once it lands: $EVAL_K10 K_PREVS=10 ARCH=tattn.
# The plain stride-4 tree it reads is complete at 437 ids.
RES_K10_TATTN_PLAIN="long-gpu    146   24  12:00"

job k10plain posw8_tattn_k10_plain_30e \
    scripts/train/train_tattn.sh "$K10_BASE $TH350_NEG $TH350_OPT $K10_RUN30 BATCH=32 ACCUM=4 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_K10_TATTN_PLAIN" \
    "attention over ten frames at 200x100 -- completes the k10 2x2: context for attention against tattn_k10_ctx50, history at plain against single_k10_plain"

# ---- evalk10: object-level scores for the three k10 arms --------------------
#
# ---- HAS RUN (2026-09-16/17). DO NOT SUBMIT AGAIN. ---------------------------
#   All three scored; results in docs/EXPERIMENTS.md (k10). The fourth k10 cell
#   is scored by evalk10plain below.
#
# The three k10 arms, all complete at 30/30 (LSF 254183/254184/269165), scored
# on evalth350's protocol -- DATA_STRIDE=4, PROTOCOL=rth -- over the k10 th350
# TEST split: GEN=th350 GROUP=temporal_k10, which run_eval.sh resolves to
# partition_temporal_k10_testeval_clean_th350x200.json.
#
# 17 SCENES, NOT 20. The k10 test list is the k5 th350 list minus the three
# scenes with no 10-previous chain, so NO number here is comparable to evalth350
# or evalposw8: the mean runs over different scenes. JOB_NAME keeps them in their
# own directory for that reason. For depth on identical scenes, score the posw8 k5
# arms on this list with K_PREVS=5; run_eval.sh documents GROUP=temporal_k10
# K_PREVS=5 as a supported combination.
#
# val/dice at best.pt, positives-only over 10 val scenes -- a hint, not the answer:
#     tattn_k10_ctx50   0.6986 @ 11/30   P 0.720  R 0.837
#     single_k10_ctx50  0.7060 @ 15/30   P 0.766  R 0.800
#     single_k10_plain  0.6613 @ 21/30   P 0.731  R 0.801
#
# PATCH TREES. The ctx50 stride-4 tree was extended to the 70 ids these scenes'
# 10-chains need (LSF 254228, COMPLETE 70/70). The preflight below re-derives
# that set and blocks --submit if any id is missing, because a missing
# predecessor is padded with the current frame, not refused. The plain arm reads
# the plain stride-4 tree, complete at 437.
#
# RESOURCES. eval-scenes peaks at (k_prevs+1)+1 full grids (scenes.py normalises
# in place). Measured on the real grid files: the worst k10 scene holds 201 GB of
# grids, against 117 GB for the worst k5 th350 scene, whose job peaked at 144.8
# GB (evalposw8_tattn). That is ~28 GB of overhead, so ~228 GB here -- 256 G is
# the largest request in this file and leaves ~11%. TERM_MEMLIMIT is the failure
# to expect if the estimate is wrong. Wall 18:00: k5 took 13219 s for 20 scenes,
# and 17 scenes at 1.83x the frames is ~5.7 h, but this file records a 2.7x swing
# on one workload from cluster conditions alone, and an eval is NOT resumable.
# single_k10_ctx50 loads one timestep -- evalposw8_single's workload, 47.0 GB /
# 4463 s. single_k10_plain is RES_EVAL_S4_CTRL's plain single-frame measurement.
#
# ARCH: tattn -> ARCH=tattn. Both singles -> ARCH=single, where K_PREVS is inert
# and written as 10 only to name the family. Context needs no flag: eval-scenes
# reads the margin from each checkpoint's io_geometry.
EVAL_K10="GEN=th350 GROUP=temporal_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_k10_th350_rth"
RES_EVAL_K10_TATTN="long-gpu   256   36  18:00"

job evalk10 evalk10_tattn_ctx50 \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_tattn_k10_ctx50_30e_2026-09-16_14h01_lsf_254183 $EVAL_K10 K_PREVS=10 ARCH=tattn" "$RES_EVAL_K10_TATTN" \
    "attention over ten past frames -- against single_k10_ctx50, the temporal contribution at matched data and context"
job evalk10 evalk10_single_ctx50 \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_single_k10_ctx50_30e_2026-09-16_14h01_lsf_254184 $EVAL_K10 K_PREVS=10 ARCH=single" "$RES_EVAL_POSW8_SINGLE" \
    "the ctx50 floor on the k10 partition, and the pivot of the three: the temporal edge on one side, the spatial edge on the other"
job evalk10 evalk10_single_plain \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_single_k10_plain_30e_2026-09-16_16h00_lsf_269165 $EVAL_K10 K_PREVS=10 ARCH=single" "$RES_EVAL_S4_CTRL" \
    "the same single-frame model without the 50 px margin -- against single_k10_ctx50, the spatial-context contribution; val/dice has it 0.045 behind"

# ---- evalk10plain: object-level score for the fourth k10 cell ----------------
#
# ---- HAS RUN (2026-09-17, LSF 330515). DO NOT SUBMIT AGAIN. -----------------
#   obj F1 0.7907 (P 0.819 / R 0.764) ith0.7_b5, best of four RTh; 0.9056 at
#   ith0.5_b10. 75.3 GB host peak, 6971 s. Results in docs/EXPERIMENTS.md (k10).
#
# posw8_tattn_k10_plain_30e (LSF 316673) on EXACTLY evalk10's string -- the same
# 17 k10 th350 test scenes, stride 4, rth -- so it completes the k10 2x2 at the
# object level:
#                  plain                     ctx50
#   single     0.7685  (0.789 / 0.749)   0.7427  (0.871 / 0.647)
#   tattn      <- THIS                   0.7748  (0.858 / 0.706)
# (obj F1 ith0.7_b5, best of four RTh; P / R.) Against tattn_k10_ctx50 it is
# context for attention; against single_k10_plain, history at 200x100.
#
# CAVEAT FOR THE READING: its best.pt is from epoch 29 of 30, still rising, so
# the plain tattn cell may be under-trained relative to the other three (best
# epochs 11-21). A win for it is robust to that; a loss is not.
#
# RESOURCES. Plain stride-4 grids are ~5.4 GB each and eval-scenes peaks at
# (k_prevs+1)+1 grids: ~65 GB here. The plain k5 tattn/convlstm evals peaked at
# 47.6 / 48.0 GB with ~38 GB of grids (LSF 316677/316679), so ~10 GB overhead
# -> ~75 GB. RES_EVAL_S4_T5 (160 G) covers it twice over. The plain stride-4
# tree is complete at 437 ids, so no tree check applies. (The same model gave
# 228 GB for evalk10_tattn_ctx50, which measured 224.3 GB, LSF 303197.)
job evalk10plain evalk10_tattn_plain \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_tattn_k10_plain_30e_2026-09-16_23h47_lsf_316673 $EVAL_K10 K_PREVS=10 ARCH=tattn" "$RES_EVAL_S4_T5" \
    "attention over ten frames at 200x100 -- the last cell of the k10 2x2; ctx50 attention scored 0.7748 and plain single 0.7685 on these 17 scenes"

# ---- k10plain45: the plain k10 attention model, trained longer ----------------
#
#     submit_all.sh k10plain45 --submit       # 1 job
#
# posw8_tattn_k10_plain_30e again, from scratch, for 45 epochs with LR_PATIENCE=5.
# Everything else is that run's configuration: K10_BASE, TH350_NEG, TH350_OPT,
# batch 32x4, hv, FUSE_SKIPS=0, RECURRENCE=none, PATIENCE=0 (all 45 epochs run).
#
# WHY. The 30-epoch run is the best k10 model at object level -- F1 0.7907
# (P 0.819 / R 0.764) on the 17 k10 test scenes, +0.016 over its ctx50 twin and
# +0.004 over th350_tattn_ctx50_neg10 re-averaged over the same 17 (0.7863) --
# and it was cut off while still improving: best.pt at epoch 29 of 30, and it
# NEVER CUT ITS LEARNING RATE. val/dice kept setting a new best inside every
# 8-epoch window, so LR_PATIENCE=8 left it at 3e-4 for all 30 epochs.
# LR_PATIENCE=5 makes the first cut likelier, and 45 epochs gives the cut room to
# pay off.
#
# NOT A PURE EXTENSION. Two things change against the 30-epoch run, the length
# and LR_PATIENCE, and against its ctx50 twin (LR_PATIENCE=8, 30 epochs) the
# schedule differs as well as the geometry. SCHEDULE=plateau never reads EPOCHS,
# so the first 30 epochs would repeat the old run exactly IF the patience were
# unchanged; at 5 they diverge from the first epoch that goes 5 without a gain.
# Resuming the 30-epoch run's resume.pt was not used for the same reason: a
# resume would splice two schedules into one run.
#
# RESOURCES ARE THE 30-EPOCH RUN'S MEASUREMENTS (LSF 316673): 82.6 GB host peak,
# 11.5 GiB VRAM reserved, 20504 s for 30 epochs = 683 s/epoch including loading
# -> ~8.5 h for 45. RES_K10_TATTN_PLAIN's 146G / 24G carries; the wall rises from
# 12:00 to 16:00 (~1.9x). A TERM_RUNLIMIT kill is not requeued -- resume with
# RESUME=<run dir>.
#
# SCORE IT WITH evalk10plain's STRING once it lands:
#     RUN=<run dir> $EVAL_K10 K_PREVS=10 ARCH=tattn   at RES_EVAL_S4_T5
# (the 30-epoch run's eval measured 75.3 GB, LSF 330515).
K10PLAIN_RUN45="SCHEDULE=plateau LR_PATIENCE=5 EPOCHS=45 PATIENCE=0"
RES_K10_TATTN_PLAIN_45E="long-gpu    146   24  16:00"

job k10plain45 posw8_tattn_k10_plain_45e \
    scripts/train/train_tattn.sh "$K10_BASE $TH350_NEG $TH350_OPT $K10PLAIN_RUN45 BATCH=32 ACCUM=4 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_K10_TATTN_PLAIN_45E" \
    "the best k10 model (obj F1 0.7907), retrained for 45 epochs at LR_PATIENCE=5 -- its 30-epoch run peaked at epoch 29 and never cut its learning rate"

# ---- k5plain: the generation-4 k5 attention models without context -----------
#
# ---- HAS RUN (2026-09-17). DO NOT SUBMIT AGAIN. ------------------------------
#   th350_tattn_plain_40e  LSF 333264, early-stopped at 39, best.pt @24 (val/dice
#                          0.6728), 56.0 GB host peak, 19968 s.
#                          outputs/th350_tattn_plain_40e_2026-09-17_09h56_lsf_333264
#   posw8_tattn_plain_40e  LSF 333265, stopped at 40 (early stop on the last
#                          epoch), best.pt @25 (val/dice 0.6764), 55.0 GB, 17485 s.
#                          outputs/posw8_tattn_plain_40e_2026-09-17_09h56_lsf_333265
#   Scored by evalk5plain below.
#
# th350_tattn_ctx50_neg10 (pos_w 4) and posw8_tattn_ctx50_25e (pos_w 8) with the
# 50 px margin removed. TH350_BASE / POSW8_BASE, TH350_NEG and TH350_OPT are
# reused verbatim, and the micro-batch stays 64x2 -- TH350_CTX minus its
# CTX_MY/CTX_MX. Context is the main lever inside each pair, but NOT the only
# one: the schedule differs too, see below.
#
# WHY. Every context pair scored at object level so far loses to its plain twin,
# and loses on RECALL (docs/RESULTS.md finding 16, docs/EXPERIMENTS.md longgrid
# and k10): the three generation-3 k5 pairs by 0.008-0.029 F1, the generation-4
# k10 single pair by 0.026. As of 2026-09-17 all eight generation-4 k5 runs
# (th350, posw8) are ctx50, so the project's lead models have no twin to say
# whether context costs them recall too.
#
# WHAT EACH IS READ AGAINST, on the same 20 th350 test scenes (obj F1
# ith0.7_b5, best of four RTh; P / R; F1 at ith0.5_b10):
#     th350_tattn_plain_40e  vs  th350_tattn_ctx50_neg10  0.7905  0.869 / 0.725  0.8888
#     posw8_tattn_plain_40e  vs  posw8_tattn_ctx50_25e    0.7699  0.850 / 0.704  0.8898
#
# THE SCHEDULE IS NOT THE TWINS', ON TWO COUNTS, by request (2026-09-17):
#   LR_PATIENCE=5, not 8. The plateau scheduler halves the LR after 5 epochs
#            without a val/dice gain instead of 8, so from the first cut onward
#            a plain arm is on a DIFFERENT trajectory from its ctx50 twin, not a
#            shorter window on the same one. Each pair therefore differs in
#            context AND in when the LR drops. A gap between them is the two
#            together; it cannot be credited to context alone.
#   EPOCHS=40, PATIENCE=15. Early stopping after 15 epochs without a val/dice
#            gain, where the twins ran 60 and 25 epochs with PATIENCE=0. At
#            pos_w 4 the ctx50 twin peaked at epoch 15, well inside any window
#            here. At pos_w 8 the twin stopped at 25 with its best at 18, so it
#            never searched epochs 26-40 and the plain arm can -- check the
#            best epoch in results.csv before reading a plain win as context.
# PATIENCE monitors val/dice, which ranks these models backwards (RESULTS.md
# finding 2); it only decides when to stop, and best.pt is still what is scored.
#
# RESOURCES, scaled from longgrid_tattn_plain (LSF 811232: plain tattn k5, 64x2,
# 85,928 samples, before its run directory was deleted): 65.8 GB host peak,
# 13.2 GiB VRAM reserved, 533 s/epoch. This partition has 71,074 samples
# (0.83x): ~55 GB host, the same VRAM (it follows the micro-batch), ~440
# s/epoch -> ~4.9 h if all 40 epochs run. pos_w changes the loss, not the
# footprint, so both arms share one line. Wall 10:00 is ~2x; a TERM_RUNLIMIT
# kill is not requeued -- resume with RESUME=<run dir>.
#
# SCORE THEM WITH evalth350's / evalposw8's STRING once they land:
#     RUN=<run dir> $EVAL_TH350 K_PREVS=5 ARCH=tattn   at RES_EVAL_S4_T5
# (plain k5 temporal evals measured 47.6-48.0 GB, LSF 316677/316679). The plain
# stride-4 tree is complete at 437 ids.
K5PLAIN_RUN40="SCHEDULE=plateau LR_PATIENCE=5 EPOCHS=40 PATIENCE=15"
RES_K5_TATTN_PLAIN="long-gpu     96   24  10:00"

job k5plain th350_tattn_plain_40e \
    scripts/train/train_tattn.sh "$TH350_BASE $TH350_NEG $TH350_OPT $K5PLAIN_RUN40 BATCH=64 ACCUM=2 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_K5_TATTN_PLAIN" \
    "th350_tattn_ctx50_neg10 without the 50 px margin -- the best F1 in the project (0.7905), and whether context costs it the recall it has cost every other pair"
job k5plain posw8_tattn_plain_40e \
    scripts/train/train_tattn.sh "$POSW8_BASE $TH350_NEG $TH350_OPT $K5PLAIN_RUN40 BATCH=64 ACCUM=2 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_K5_TATTN_PLAIN" \
    "posw8_tattn_ctx50_25e without the 50 px margin -- the same question at pos_w 8, the weight the k10 runs were trained at"

# ---- evalk5plain: object-level scores for the two k5plain arms ----------------
#
#     submit_all.sh evalk5plain --submit                  # 2 jobs
#     submit_all.sh evalk5plain --only=posw8 --submit     # one arm
#
# The two plain k5 attention twins on EXACTLY evalth350's / evalposw8's string --
# GEN=th350, GROUP=temporal_k5, DATA_STRIDE=4, PROTOCOL=rth, the same 20 th350
# test scenes -- so each is read directly against its ctx50 twin:
#     th350_tattn_plain_40e  vs  th350_tattn_ctx50_neg10  0.7905  0.869 / 0.725  0.8888
#     posw8_tattn_plain_40e  vs  posw8_tattn_ctx50_25e    0.7699  0.850 / 0.704  0.8898
# (obj F1 ith0.7_b5, best of four RTh; P / R; F1 at ith0.5_b10.) Remember the
# k5plain caveat: LR_PATIENCE=5 against the twins' 8, so each pair differs in the
# schedule as well as the context.
#
# val/dice at best.pt, plain vs ctx50 -- plain behind in both, as it was in every
# pair where plain then WON at object level:
#     pos_w 4   0.6728 @24   vs  0.6970 @15
#     pos_w 8   0.6764 @25   vs  0.7010 @18
# The pos_w 8 plain arm peaked at epoch 25, inside the 25 epochs its twin ran,
# so the epoch-budget asymmetry flagged in k5plain did not come into play.
#
# RESOURCES. Plain stride-4 grids, (k+1)+1 = 7 of them at ~5.4 GB: the plain k5
# temporal evals on a 20-scene list measured 47.6 / 48.0 GB (LSF 316677/316679),
# so RES_EVAL_S4_T5 (160 G) is ~3x. The plain stride-4 tree is complete at 437
# ids, so no tree check applies. ARCH=tattn, K_PREVS=5.
job evalk5plain evalk5plain_th350_tattn \
    scripts/eval/run_eval.sh "RUN=outputs/th350_tattn_plain_40e_2026-09-17_09h56_lsf_333264 $EVAL_TH350 K_PREVS=5 ARCH=tattn" "$RES_EVAL_S4_T5" \
    "the plain twin of the best F1 in the project (0.7905): does context cost the lead model recall, as it has in all five pairs so far"
job evalk5plain evalk5plain_posw8_tattn \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_tattn_plain_40e_2026-09-17_09h56_lsf_333265 $EVAL_TH350 K_PREVS=5 ARCH=tattn" "$RES_EVAL_S4_T5" \
    "the plain twin of posw8_tattn_ctx50_25e (0.7699) -- the same question at pos_w 8, the weight where plain attention won at k10"

# ---- fill: the two leading attention models on filled history chains ---------
#
#     submit_all.sh fill --submit             # 2 jobs
#
# Generation 5 (assets/PARTITIONS.md): find_11day_sequences now skips a missing
# acquisition and takes an older one, up to 3k slots back, and the attention is
# told each frame's real age (meta.chain_offsets -> models.temporal.run_model).
# The partitions are generation 4 rebuilt with that rule and NOTHING else --
# same generator, AOI, bounds, seed and 350N/200S threshold:
#     k5   train/val/test 100/11/20 -> 130/12/24   train positives 35,558 -> 44,114
#     k10  train/val/test  74/10/17 -> 121/11/24   train positives 27,191 -> 41,032
# Every generation-4 split is a subset of its generation-5 twin.
#
# EACH ARM IS A SCORED LEADER WITH THE PARTITION SWAPPED, every other variable
# reused verbatim, so the delta against its twin is the filled chains alone:
#     fill_tattn_k5_ctx50_neg10   <- th350_tattn_ctx50_neg10    obj F1 0.7905 (20 scenes)
#     fill_tattn_k10_plain_30e    <- posw8_tattn_k10_plain_30e  obj F1 0.7907 (17 scenes)
# The k10 twin is the 30-epoch run, not k10plain45: it is the one with an object
# score, and LR_PATIENCE is part of what would otherwise differ.
#
# WHAT THEY ANSWER. Whether more supervised dates (the scenes the gap-free rule
# dropped) and real frame ages are worth having. Expect the bigger effect at
# k10, which loses the most to the strict rule (+47 train scenes against +30).
# NOT separable here: more data vs. uneven ages. Both arrive together because
# every newly admitted scene is a gappy one.
#
# RESOURCES, scaled from each twin's measurement by its sample ratio:
#   k5   140.6 GB / 34729 s for 60 epochs on 71,074 samples (LSF 816217).
#        x1.24 -> ~175 GB, ~12 h. 256G carries; wall 12:00 -> 18:00.
#   k10  82.6 GB / 20504 s for 30 epochs (LSF 316673). x1.51 -> ~125 GB, ~8.6 h.
#        146G would sit at ~85%, so the next tier, 208G; wall 12:00 -> 16:00.
# Stride-2 trees (ctx50 and plain, data and mask) hold every id the filled
# train/val chains need: 249 at k5, 297 at k10, checked 2026-10-06.
FILL_K5_BASE="PARTITION=assets/partition_temporal_k5_clean_fill_th350x200.json K_PREVS=5 POS_W=4"
FILL_K10_BASE="PARTITION=assets/partition_temporal_k10_clean_fill_th350x200.json K_PREVS=10 POS_W=8"
RES_FILL_K5_TATTN="long-gpu    256   48  18:00"
RES_FILL_K10_TATTN_PLAIN="long-gpu    208   24  16:00"

job fill fill_tattn_k5_ctx50_neg10 \
    scripts/train/train_tattn.sh "$FILL_K5_BASE $TH350_NEG $TH350_OPT $TH350_RUN $TH350_CTX AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_FILL_K5_TATTN" \
    "th350_tattn_ctx50_neg10 (obj F1 0.7905) on filled k5 chains -- +30 train scenes, and real frame ages for the 49 chains that skip a missing date"
job fill fill_tattn_k10_plain_30e \
    scripts/train/train_tattn.sh "$FILL_K10_BASE $TH350_NEG $TH350_OPT $K10_RUN30 BATCH=32 ACCUM=4 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_FILL_K10_TATTN_PLAIN" \
    "posw8_tattn_k10_plain_30e (obj F1 0.7907) on filled k10 chains -- +47 train scenes, where the gap-free rule cost the most"

# ---- evalfill: the generation-4 leaders scored on the generation-5 test lists --
#
#     submit_all.sh evalfill --submit         # 2 jobs
#
# The fill arms will be scored on 24 test scenes at both depths, a superset of
# th350's 20 (k5) and 17 (k10). Their twins have no score on the added scenes,
# so these two rows score the EXISTING checkpoints on the generation-5 list. The
# twins' predictions on the old scenes do not change -- a gap-free chain is the
# same chain with the same ages -- so this costs nothing in comparability and
# gives the coverage half of the comparison:
#     like-for-like  both models re-averaged over the generation-4 list
#     coverage       both models over all 24
# Add the fill arms' own rows here, on the same strings, once their runs land.
#
# PATCH TREES. The plain stride-4 tree has every id the k10 list needs (89).
# The ctx50 stride-4 tree is SHORT 5 ids for the k5 list -- 20241106_20241117,
# 20251012_20251023, 20251023_20251103, 20251103_20251114, 20251206_20251217 --
# and the preflight below blocks --submit until they exist. Build them first:
#     PARTITION=assets/partition_temporal_k5_testeval_clean_fill_th350x200.json K_PREVS=5 \
#       bsub < scripts/data/make_context_stride4_patches.sh
# RESOURCES are the twins' own eval measurements: RES_EVAL_TH350 for the ctx50
# k5 workload, RES_EVAL_S4_T5 for the plain k10 one (75.3 GB, LSF 330515); 24
# scenes against 20 / 17 only lengthens the run.
EVAL_FILL_K5="GEN=fill GROUP=temporal_k5 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_k5_fill_rth"
EVAL_FILL_K10="GEN=fill GROUP=temporal_k10 DATA_STRIDE=4 PROTOCOL=rth JOB_NAME=scenes_temporal_k10_fill_rth"

job evalfill evalfill_th350_tattn_ctx50 \
    scripts/eval/run_eval.sh "RUN=outputs/th350_tattn_ctx50_neg10_2026-09-10_01h16_lsf_816217 $EVAL_FILL_K5 K_PREVS=5 ARCH=tattn" "$RES_EVAL_TH350" \
    "the k5 leader on the 24-scene generation-5 list: its 20 old scenes reproduce evalth350, the 4 new ones are the baseline for fill_tattn_k5's coverage gain"
job evalfill evalfill_posw8_tattn_k10_plain \
    scripts/eval/run_eval.sh "RUN=outputs/posw8_tattn_k10_plain_30e_2026-09-16_23h47_lsf_316673 $EVAL_FILL_K10 K_PREVS=10 ARCH=tattn" "$RES_EVAL_S4_T5" \
    "the k10 leader on the 24-scene generation-5 list: 17 old scenes reproduce evalk10plain, 7 new ones it could never be scored on before"

# ---- argument parsing -------------------------------------------------------
WANT=all; SUBMIT=no; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    eval6ref|ampfix|lrscan|evalampfix|evalctx|reg|dropscan|lossscan|momscan|adamscan|combo|th350|evalth350|posw8|evalposw8|k10|k10plain|evalk10|evalk10plain|k10plain45|k5plain|evalk5plain|fill|evalfill|all)   WANT="$1" ;;
    # Retired by name rather than left to select zero jobs, so the mistake is
    # visible instead of looking like an empty batch. See docs/EXPERIMENTS.md.
    long200|long500|ctx50|evallong200|\
    train|tattn|control|clean22|valpos|attnfix|attnpos|pre23|training|\
    eval|eval2|eval3|eval4|eval5|eval6|eval7|probe|posonly)
      echo "'$1' has already run -- see docs/EXPERIMENTS.md for what it settled." >&2
      echo "Recover its job definitions with: git show d708fba:scripts/submit_all.sh" >&2
      exit 1 ;;
    longreg|longgrid|evallonggrid)
      echo "'$1' has already run, and its run directories were deleted 2026-09-17 --" >&2
      echo "see docs/EXPERIMENTS.md. These blocks were never committed; their job" >&2
      echo "definitions are archived in docs/reference/submit_all_retired_2026-09-17.sh" >&2
      exit 1 ;;
    valneg)
      echo "the 'valneg' batch is retired: validation is positives-only." >&2; exit 1 ;;
    blend)
      echo "'blend' cannot run: both checkpoints were deleted (see docs/EXPERIMENTS.md)." >&2; exit 1 ;;
    rescore)
      echo "'rescore' cannot run: its eval directory has no *_pred.npy left." >&2; exit 1 ;;
    --submit)       SUBMIT=yes ;;
    --only)         shift; [ $# -gt 0 ] || { echo "--only needs a value" >&2; exit 1; }
                    ONLY+=("$1") ;;
    --only=*)       ONLY+=("${1#--only=}") ;;
    -h|--help)      sed -n '2,/measured from\./p' "$0"; exit 0 ;;
    *)              echo "unknown argument '$1'" >&2; exit 1 ;;
  esac
  shift
done

selected() {
  local kind="$1" name="${2-}"
  if [ "${#ONLY[@]}" -gt 0 ]; then          # --only is an AND on top of the kind
    local hit=no pat
    for pat in "${ONLY[@]}"; do
      case "$name" in *"$pat"*) hit=yes ;; esac
    done
    [ "$hit" = yes ] || return 1
  fi
  [ "$WANT" = all ] || [ "$WANT" = "$kind" ]
}

if [ "$SUBMIT" = yes ] && ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found -- run this on the cluster, not the mounted volume." >&2
  exit 1
fi

# ---- preflight --------------------------------------------------------------
fail=0
for i in "${!NAMES[@]}"; do
  selected "${KINDS[$i]}" "${NAMES[$i]}" || continue
  [ -f "${TEMPLATES[$i]}" ] || { echo "missing template ${TEMPLATES[$i]}" >&2; fail=1; }
  case "${OVERRIDES[$i]}" in
    *RUN=*) r=$(sed 's/.*RUN=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            [ -f "$r/checkpoints/best.pt" ] || { echo "missing checkpoint $r/checkpoints/best.pt" >&2; fail=1; } ;;
    *PARTITION=*) pf=$(sed 's/.*PARTITION=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            [ -f "$pf" ] || { echo "missing partition $pf" >&2; fail=1; } ;;
    *DIR=*) d=$(sed 's/.*DIR=\([^ ]*\).*/\1/' <<<"${OVERRIDES[$i]}")
            if [ ! -d "$d" ]; then
              echo "missing eval directory $d" >&2; fail=1
            elif [ -z "$(find "$d" -maxdepth 1 -name '*_pred.npy' -print -quit)" ]; then
              echo "no *_pred.npy in $d -- rebuild it with run_eval.sh, not rescore.sh" >&2; fail=1
            fi ;;
  esac
done
# -- the ctx50 stride-4 tree must hold every id a ctx50 eval will read --------
# This batch is the one case in the file where an incomplete input does NOT
# fail. scenes.py:354 filters predecessors by os.path.exists and pads the short
# chain with the CURRENT frame, so a missing grid produces a finished job and a
# plausible, wrong number. The id set is DERIVED with the project's own chain
# code (stdlib-only import, so this runs on the login node without conda), never
# typed -- the same derivation make_context_stride4_patches.sh step (1) uses.
#
# If the patch root cannot be found this NOTES and does not block: the dry run
# is routinely done on the mounted volume, where the cluster path does not
# exist, and refusing there would make the file unusable for its main purpose.
# --submit already requires bsub, so a real submission is always on the cluster.
# Each batch contributes the scene list it scores and the chain depth it reads,
# as "<testeval partition>:<k>"; the union of their id sets is what is checked.
needs_tree=no; TREE_SETS=()
for i in "${!NAMES[@]}"; do
  selected "${KINDS[$i]}" "${NAMES[$i]}" || continue
  case "${KINDS[$i]}" in
    evalth350|evalposw8) needs_tree=yes
                         TREE_SETS+=("assets/partition_temporal_k5_testeval_clean_th350x200.json:5") ;;
    evalk10)             needs_tree=yes
                         TREE_SETS+=("assets/partition_temporal_k10_testeval_clean_th350x200.json:10") ;;
    evalfill)            # Only the k5 row reads the ctx50 tree; the k10 row is plain.
                         needs_tree=yes
                         TREE_SETS+=("assets/partition_temporal_k5_testeval_clean_fill_th350x200.json:5") ;;
  esac
done
if [ "$needs_tree" = yes ]; then
  CTX4=""
  for root in "${PATCHES:-}" \
              /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches \
              /Volumes/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches; do
    [ -n "$root" ] || continue
    if [ -d "$root/data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned" ]; then
      CTX4="$root/data_patches_H200_W100_ctx50x50_strpp4_11days_Aligned"; break
    fi
  done
  if [ -z "$CTX4" ]; then
    echo "NOTE: ctx50 stride-4 tree not reachable from here -- the ctx50 evals' patch" >&2
    echo "      coverage was NOT verified. It is checked for real by step (4) of" >&2
    echo "      scripts/data/make_context_stride4_patches.sh; do not submit until" >&2
    echo "      that has passed." >&2
  else
    miss=$(python3 - "$CTX4" "${TREE_SETS[@]}" <<'PYCHK'
import json, os, sys
sys.path.insert(0, os.getcwd())
from sinkholes.meta import find_11day_sequences
ctx4 = sys.argv[1]
SUF = "_H200_W100_strpp4.npy"
def ids(prefix):
    return {f[len(prefix):-len(SUF)] for f in os.listdir(ctx4)
            if f.startswith(prefix + "2") and f.endswith(SUF)}
coord = json.load(open("assets/intf_coord.json"))
done = ids("data_patches_") & ids("data_patches_nonz_")
# One line per scene list that is short of ids: "<partition> <k> <id> <id> ...".
for spec in dict.fromkeys(sys.argv[2:]):
    part, k = spec.rsplit(":", 1)
    k = int(k)
    scenes = json.load(open(part))["val"]
    kept = [i for i in scenes
            if isinstance(coord.get(i, {}).get("nonz_num"), int) and coord[i]["nonz_num"] > 150]
    prev_dict, with_chains = find_11day_sequences(
        coord, k_prev=k, restrict_to=kept, require_current_nonz_gt0=False)
    need = set(with_chains)
    for cur in with_chains:
        need.update(prev_dict[cur]["prevs"][:k])
    if need - done:
        print(part, k, " ".join(sorted(need - done)))
PYCHK
) || miss="CHECK_FAILED"
    if [ "$miss" = "CHECK_FAILED" ]; then
      echo "NOTE: could not derive the ctx50 evals' id set; patch coverage unverified." >&2
    elif [ -n "$miss" ]; then
      while read -r part k ids_missing; do
        n_miss=$(set -- $ids_missing; echo $#)
        echo "$part (k=$k) needs $n_miss id(s) the ctx50 stride-4 tree does not have:" >&2
        printf '    %s\n' $ids_missing >&2
        echo "  build them (derives its own id set, resumable, ~16 GB per id):" >&2
        echo "    PARTITION=$part K_PREVS=$k \\" >&2
        echo "      bsub < scripts/data/make_context_stride4_patches.sh" >&2
      done <<<"$miss"
      echo "  a missing predecessor does NOT fail at eval time -- it is padded with" >&2
      echo "  the current frame and the job reports a wrong number. Hence this gate." >&2
      # BLOCKS --submit, WARNS on a dry run. Unlike a missing checkpoint, this is
      # a known-pending build rather than a broken row, and a bare
      # `submit_all.sh` has to keep working as the catalogue of every live job --
      # failing the listing because one batch's tree is still being cut would
      # make the file's primary use unavailable. Nothing can actually reach LSF
      # while the ids are absent, which is the property that matters.
      [ "$SUBMIT" = yes ] && fail=1
    fi
  fi
fi

[ "$fail" = 0 ] || { echo "preflight failed; nothing submitted" >&2; exit 1; }

# ---- summary ----------------------------------------------------------------
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
    # `if` keeps set -e from aborting the batch on one rejection: a bad queue
    # limit on job 2 must not silently cancel jobs 3 and 4.
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
    echo "failed to submit:"; printf '  %s\n' "${FAILED[@]}"
  fi
else
  echo "$n job(s) listed. DRY RUN -- nothing was submitted. Add --submit to send them."
fi
