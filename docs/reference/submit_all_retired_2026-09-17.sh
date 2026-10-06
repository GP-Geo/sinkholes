# ============================================================================
#  RETIRED submit_all.sh blocks: longreg, longgrid, evallonggrid
#
#  Removed from scripts/submit_all.sh on 2026-09-17, verbatim. Kept here because
#  these blocks were NEVER COMMITTED -- `git show` cannot recover them.
#
#  Why retired: every job below has run, and on 2026-09-17 all eight run
#  directories (longreg_*, longgrid_*, longsingle_*) and their six prediction
#  directories were DELETED. The runs trained on the generation-3 partition,
#  superseded by the generation-4 label-quality threshold. What they settled is
#  recorded in docs/EXPERIMENTS.md (longreg / longsingle / longgrid) and
#  docs/ATTENTION.md (k5, generation 3). Every RUN= path below is gone. The
#  verbatim blocks below cite docs/ATTENTION_COLLAPSE.md, its name until 2026-09-17.
#
#  Reference only. This file is not meant to be sourced or run: it depends on
#  submit_all.sh's job() and on variables defined elsewhere in that file.
# ============================================================================

# ---- longreg: the 60-epoch regularised arms, ctx50 on the all-years archive --
#
#     submit_all.sh longreg --submit         # 3 jobs, ~7h each
#     submit_all.sh longreg --only=aughv --submit
#
#     >>> READ "THE MEMORY REQUEST IS UNPRECEDENTED" BELOW BEFORE --submit. <<<
#
# THE FIRST RUNS IN THIS PROJECT THAT ARE NOT A SCREEN. Every batch since
# 2026-09-07 was 8-30 epochs to rank a lever. These are 60 epochs at the
# settings those screens chose, on the full archive, and they exist to produce
# a MODEL rather than a ranking.
#
# WHAT THE SCREENS SETTLED, and what they did not:
#
#   * adamw over rmsprop, and lr 3e-4. The adamscan ladder is monotone across a
#     decade and a half -- 1e-5 0.6346, 3e-5 0.6530, 1e-4 0.6719, 3e-4 0.6762 --
#     a 0.042 span, the only optimiser result that clears the noise floor. 3e-4
#     is also the ONLY arm where the hardcoded 1.0 gradient clip is inert
#     (grad/clip% 0.5 against 100 at 1e-5), so it is the one lr on the ladder
#     that means what it says.
#   * NOTHING ELSE. dropout rate, jaccard-vs-dice and weight decay all landed
#     inside 0.011, the run-to-run spread measured by the accidental duplicate
#     submission of combo_ctx_p00 (LSF 275654 vs 275798: identical config,
#     identical seed, both 20/20, val/dice 0.6908 vs 0.7017). That figure
#     supersedes the 0.0013 seed pair AND RESULTS.md's +-0.006.
#
# dropout 0.2 AND wd 1e-2 ARE ON SPEC, NOT ON EVIDENCE -- both measured as
# no-ops at 20 epochs. They are carried because the failure mode they target is
# real: over 186 epochs ampfix_t5_convlstm_ring3_200e drove train/loss 0.466 ->
# 0.186 and val/loss 0.533 -> 1.001 while val/dice DID NOT MOVE (0.6647 at 14,
# 0.6517 at 200). That is memorisation, and it is a CALIBRATION failure rather
# than a segmentation one -- which matters because the object metric is scored
# at confidence thresholds, so where the model puts its probability mass moves
# the score even when patch overlap does not.
#
# PLATEAU, NOT COSINE, BY REQUEST -- and it makes the experiment honest. Cosine
# anneals lr to 0 and MANUFACTURES a late best epoch; the last rows win by
# construction. Under plateau the schedule only reacts after val stalls, so a
# peak that moves late here is the regularisation doing it. LR_PATIENCE=8
# rather than the screens' 5: at 60 epochs a 5-epoch fuse fires on noise.
#
# DO NOT READ THE BEST EPOCH AS CONVERGENCE. The val/dice curve is FLAT from
# roughly epoch 7 onward -- least-squares slope over epochs 11-20 is -0.0002 to
# -0.0024 across every arm of every screen -- and best-minus-mean(last 10) is
# 0.014-0.028 everywhere, which is a max of ~20 draws at sigma 0.011. best.pt is
# therefore chosen by overfitting the validation set and its val/dice is
# inflated ~0.02 against what the model does. Judge these at OBJECT level.
#
# ---- why the archive changed, and what it costs -----------------------------
#
# partition_temporal_k5_clean, NOT _pre2023. This is the batch's biggest break
# with everything before it and it buys three things:
#
#     partition            train intfs   train yrs    val yr   positives
#     k5_pre2023  (old)         77       2019-2020     2021      24,817
#     k5_clean    (this)       163       2019-2023     2024      42,985
#
#   1. 1.73x the training positives, 2.1x the interferograms.
#   2. A VALIDATION SET THAT IS ADJACENT TO THE EVALUATION. Under _pre2023 the
#      model trained on 2019-2020, selected best.pt on 2021, and was reported on
#      2025-2026 scenes -- selection and reporting four to five years apart.
#      _clean validates on 2024, one year off the 2025-2026 eval list.
#   3. Training that has seen 2021-2023 at all.
#
# WHAT IT COSTS. Nothing in RESULTS.md, MODEL_RUNS.md or PREDICTIONS.md is on
# this partition, and neither is any run of the 2026-09-07/08 campaign. These
# three arms are readable against EACH OTHER and against nothing else. There is
# no _clean comparator at any depth; ampfix_ctx50_t5_convlstm_ring3_30e shares
# the context and the batching but not the archive.
#
# LEAKAGE, AND IT IS EASY TO TRIP. Verified disjoint against the protocol these
# will be scored on:
#     k5_clean train  vs  k10_clean test (GEN=3 GROUP=temporal_k10)  ->  0
#     k5_clean train  vs  k10_pre2023 test                           -> 35
# Scoring one of these on ANY *_pre23_* protocol leaks all 35 of that list's
# interferograms into training and returns a number that looks fine. Use
# EVAL6_TEMP / EVAL_S2_TEMP only.
#
# ---- THE MEMORY REQUEST IS UNPRECEDENTED ------------------------------------
#
# 256G host is larger than any TRAINING job this project has run. The training
# maximum is 180G (RES_AMPFIX_CTX_LONG / RES_COMBO_CTX, both ctx50).
#
# THE 208G FIGURES IN THIS FILE ARE NOT A PRECEDENT FOR IT. RES_EVAL_S4_G10 and
# RES_EVAL_AMPFIX are EVALUATION jobs, and a scene evaluation holds one scene's
# patches at a time -- it never materialises a training sample set, so its
# footprint scales with the scene and not with N x T. The two workloads are not
# comparable and the eval number says nothing about a training cap.
#
# NO TRAINING-SIDE CEILING IS DOCUMENTED ANYWHERE, so 256G is unprecedented
# rather than known-refused. IT IS STILL NOT KNOWN TO BE SCHEDULABLE. Check
# before submitting -- on the login node:
#
#     bhosts -l | grep -i maxmem        # or: lshosts -w
#
# If no long-gpu host carries ~256G of usable RAM these three jobs PEND
# FOREVER; the preflight in this file checks paths, not cluster limits.
#
# WHERE THE NUMBER COMES FROM. _load_temporal materialises every sample's pixels
# (PLAN_LARGE_CONTEXT.md 3), so peak resident host memory is
# N x T x (ctx_h*ctx_w + 200*100) x 4 B. The formula reproduces both runs on
# disk -- it predicts 44 GiB for the plain screens whose measured peak was
# 43.5-46.6 GB, and 89 GiB for ampfix_ctx50 which asked 180G:
#
#     config                        samples   predicted peak   request
#     pre2023 negpp1.0 ctx50         49,604        89 GiB        180G  (ran)
#     clean   negpp1.0 ctx50         85,970       154 GiB        256G  (this)
#     clean   negpp2.0 ctx50        128,955       231 GiB       ~380G  untried
#
# 256G is 1.66x the predicted peak, against the 1.9x ampfix_ctx50 carried. The
# padding is thinner because the absolute number is already unprecedented.
#
# NEG_PER_POS STAYS AT 1.0 HERE, but is no longer ruled out on principle: at 2.0
# the peak is 231 GiB and the request lands near 380G, which is simply a bigger
# unknown than 256G. If the nodes turn out to be large, that is the next lever
# and it is the cheapest way to grow this dataset further.
#
# IF THESE PEND ON MEMORY, the fallbacks are dropping ctx50 (clean + plain
# patches peaks at 77 GiB, a ~146G request, comfortably inside the 180G the
# project already runs) or building the lazy loader PLAN_LARGE_CONTEXT.md 3
# specifies, which the plan says would use ~22 GB at ANY context size.
#
# THE THIRD ARM SPENDS THE ONE DIVERSITY LEVER THAT IS FREE. NEG_RING_OUTER
# widens the annulus negatives are drawn from without changing how many there
# are, so it buys different ground at CONSTANT memory. 3 is hard near-field
# only; 10 reaches far-field. Never run.
#
# WHAT EACH PAIR READS:
#   longreg_aughv vs longreg_noaug   -> does flip augmentation delay memorisation
#   longreg_aughv vs longreg_ring10  -> does negative diversity do it instead
# Both single-lever. aughv is the candidate model.
#
# AUGMENT=hv HAS NEVER BEEN RUN -- no log in outputs/ mentions augment_flips.
# The implementation is correct (dataset.py: one flip decision per sample
# applied across all T timesteps via ellipsis indexing, mask flipped about the
# same axis, TRAIN split only, torch.rand so DataLoader workers do not share a
# sequence). The open question is physical: these scenes share a fixed radar
# look geometry, so a left-right flip produces an illumination that never occurs
# in the archive. If hv LOSES to noaug, that is the explanation and AUGMENT=v
# is the follow-up.
LONGREG_BASE="PARTITION=assets/partition_temporal_k5_clean.json K_PREVS=5 POS_W=4"
LONGREG_NEG="RING_NEGS=yes NEG_RING_INNER=1 NEG_PER_POS=1.0"
LONGREG_OPT="OPTIM=adamw LR=3e-4 WEIGHT_DECAY=1e-2 DROPOUT_BOTTLENECK=0.2 SEG_LOSS=dice"
LONGREG_RUN="SCHEDULE=plateau LR_PATIENCE=8 EPOCHS=60 PATIENCE=0"
# 300x200 context. BATCH=64 ACCUM=2 is not a preference: ctx50 measured 38.5 GiB
# reserved at micro-batch 64 and activations scale with the MICRO-batch, so
# micro-batch 128 lands near 76 GiB -- above every gmem this project has been
# granted. ACCUM=2 keeps the optimiser's effective batch at 128.
LONGREG_CTX="CTX_MY=50 CTX_MX=50 BATCH=64 ACCUM=2"

# 3m28s/epoch measured on this geometry and batching at 49,604 samples
# (ampfix_ctx50_t5_convlstm_ring3_30e: 14 epochs in 48m31s). x1.73 the data is
# ~6m/epoch -> ~6h for 60, plus a slower assembly at 2.1x the interferograms.
# 18:00 is one allocation with margin and is already accepted on long-gpu
# (RES_AMPFIX_CTX_LONG). --resume auto covers a requeue if it is not.
# 256G host: see THE MEMORY REQUEST IS UNPRECEDENTED above -- VERIFY FIRST.
# gmem 48G against 38.5 GiB reserved, the request combo_ctx ran on successfully.
RES_LONGREG="long-gpu   256   48  18:00"

job longreg longreg_aughv \
    scripts/train/train_convlstm.sh "$LONGREG_BASE $LONGREG_NEG $LONGREG_OPT $LONGREG_RUN $LONGREG_CTX NEG_RING_OUTER=3 AUGMENT=hv" "$RES_LONGREG" \
    "THE candidate: everything the screens chose, on the full archive, plus the one regulariser that adds data instead of constraining the model -- and the first run in the project to use augment_flips at all"
job longreg longreg_noaug \
    scripts/train/train_convlstm.sh "$LONGREG_BASE $LONGREG_NEG $LONGREG_OPT $LONGREG_RUN $LONGREG_CTX NEG_RING_OUTER=3 AUGMENT=none" "$RES_LONGREG" \
    "the control that makes the candidate readable: identical but for the flips, so a later peak is attributable to augmentation and not to the 60-epoch budget, the plateau patience or the bigger archive"
job longreg longreg_ring10 \
    scripts/train/train_convlstm.sh "$LONGREG_BASE $LONGREG_NEG $LONGREG_OPT $LONGREG_RUN $LONGREG_CTX NEG_RING_OUTER=10 AUGMENT=hv" "$RES_LONGREG" \
    "far-field negatives at constant memory: the only way to feed this batch more varied ground once ctx50 has capped the sample count, and never run"

# ---- longgrid: the 2x2 that says WHERE longreg's score comes from -----------
#
#     submit_all.sh longgrid --submit            # 1 job -- FIVE CELLS ARE ALREADY OUT
#
# longreg_aughv changes TWO things at once against everything this project has
# run: it is recurrent AND it sees 300x200. Its score therefore cannot be
# attributed. This grid decomposes the gain -- a 2x2 over {architecture} x
# {geometry}, plus a third architecture in the ctx50 column:
#
#                     plain 200x100                    ctx50 300x200
#   single       longsingle_plain    (LSF 721118)  longsingle_ctx50 (LSF 721120)
#   convlstm     longgrid_convlstm_plain (751271)  longreg_aughv    (LSF 697785)
#   tattn        longgrid_tattn_plain    <- THIS   longgrid_tattn_ctx50  (769020)
#
# EACH COLUMN IS AN ARCHITECTURE LADDER at fixed geometry and fixed everything
# else: no temporal model (single) -> recurrence (convlstm) -> selection (tattn).
# Each ROW is a clean geometry pair. The tattn row exists at both geometries for
# a reason that is not symmetry:
#
#   ONLY THE PLAIN TATTN ARM CAN BE PROBED. attention_probe.py:513 refuses
#   context checkpoints outright ("does not yet support context patches"), and
#   it is the ONLY reader of attention weights in the codebase --
#   forward_with_attention has exactly two callers, the probe and tattn_unet's
#   own forward. test-patches does NOT read them, despite what that error
#   message suggests. So longgrid_tattn_ctx50 currently CANNOT be checked for
#   selectivity, and longgrid_tattn_plain is the only arm that can say whether
#   attention is selecting at all under this optimiser and this archive.
#
#   That makes the plain arm the INTERPRETER for the ctx one. If plain selects
#   and ctx scores well, the ctx arm is plausibly selecting too. If plain
#   collapses, the ctx arm should be reported as a differently-parameterised
#   mean until the probe learns to read it.
#
# NAMES ON DISK. The two single cells were submitted as longsingle_* before this
# batch was reorganised around the square, so their run directories read
# outputs/longsingle_{plain,ctx50}_2026-09-09_13h30_lsf_7211{18,20}. They are the
# square's two single cells; nothing is renamed, because they are being written
# to right now.
#
# WHAT EACH EDGE READS, and the two diagonals are the point:
#   convlstm_plain vs longreg_aughv       -> CONTEXT, holding recurrence
#   single_plain   vs single_ctx50        -> CONTEXT, without recurrence
#   single_plain   vs convlstm_plain      -> RECURRENCE, holding geometry
#   single_ctx50   vs longreg_aughv       -> RECURRENCE, at ctx50
#   the two context edges against each other -> the INTERACTION: whether
#     recurrence is what makes context pay, or the two are simply additive
#
# Only the full square can answer that. With longreg alone, "the temporal model
# is worth it" and "the extra 200x200 pixels are worth it" are the same number.
#
# THE EXISTING SINGLE-FRAME RUN CANNOT STAND IN FOR ANY CELL.
# ampfix_t5_single_ring3_200e is the only recent one and it differs from longreg
# on four axes at once -- partition (_pre2023 vs _clean), optimiser (rmsprop
# 1e-5 momentum 0.9 vs adamw 3e-4), weight decay (1e-8 vs 1e-2) and augmentation
# (none vs hv) -- before geometry is even considered.
#
# AND THE GAP IT IS MEANT TO MEASURE IS ALREADY IN DOUBT. On _pre2023, where a
# single and a ConvLSTM twin WERE run under matched settings, best val/dice is
# 0.6569 (single, epoch 11) against 0.6647 (convlstm, epoch 14): +0.0078, INSIDE
# the 0.011 run-to-run spread longreg's header establishes. RESULTS.md 9's
# +0.047 for recurrence does not survive the corrected data. That is the single
# strongest reason to run this square rather than assume the recurrent cell wins.
#
# ---- every cell runs at MICRO-batch 64, and that is not a detail ------------
#
# longreg_aughv is BATCH=64 ACCUM=2 -- 64 forced by ctx50 activations, ACCUM=2
# restoring the effective 128. The other three cells could each fit 128x1 at the
# same effective batch, and they DO NOT USE IT: BatchNorm sees one MICRO-batch
# at a time (PRESETS.md), so a cell at 128x1 would differ from the running arm
# in its normalisation statistics as well as in the lever under test. The
# running arm fixes the micro-batch for the whole square at 64.
#
# longsingle_plain (LSF 721118) IS THE EXCEPTION, because it was submitted at
# 128x1 before this was settled. That is why longgrid_convlstm_plain below stays
# at 64x2 rather than matching it: given one cell is off-grid, the micro-batch
# difference has to land on ONE edge, and it must not land on
# convlstm_plain-vs-longreg_aughv -- the edge that answers "context or
# recurrence?", the question the whole square exists for. Putting it on the
# recurrence-at-plain edge instead costs nothing extra, because that edge already
# carries the dropout asymmetry and is caveated either way.
#
# ---- the one axis that cannot be matched, on two of the four cells ----------
#
# DROPOUT_BOTTLENECK=0.2 names the tensor a ConvLSTM compresses its sequence
# into. A plain U-Net has no such tensor and train.py:372 REFUSES the flag
# rather than ignoring it. So:
#
#   convlstm_plain vs longreg_aughv   -- BOTH carry dropout 0.2. PERFECTLY
#                                        single-lever: geometry and nothing else.
#   single_plain vs single_ctx50      -- NEITHER carries it. Also single-lever.
#   either RECURRENCE edge             -- dropout present on one side only.
#   tattn_ctx50 vs longreg_aughv      -- dropout on the ConvLSTM side only.
#                                        tattn has no bottleneck state to drop:
#                                        it selects over frames instead of
#                                        collapsing them into one tensor, and
#                                        train.py:387 gates the flag on
#                                        convlstm_unet. Same bounded caveat.
#
# So the two CONTEXT edges are clean and the two RECURRENCE edges carry a
# bounded caveat: dropout landed inside the 0.011 spread at 20 epochs, i.e. it
# was a no-op wherever it has been looked at. Quote the context edges without
# qualification; qualify the recurrence edges.
#
# ---- READ THE TATTN ARM WITH THE PROBE, NOT THE DICE CURVE ------------------
#
# THE ATTENTION IN THIS PROJECT HAS A HISTORY OF NOT WORKING, and the arm most
# like this one is the one that failed. docs/ATTENTION_COLLAPSE.md: 15 of 15
# tattn checkpoints produced weights of EXACTLY 1/T at every timestep, head and
# bottleneck pixel -- an unweighted mean, with ~1M attention parameters as
# decoration. Not a training pathology: at init the tokens are 99.91% a
# component shared across the sequence, so the softmax has nothing to separate.
#
# --tattn_contrast and --tattn_qk_norm fix it and are ON BY DEFAULT here
# (CONTRAST/QK_NORM default to yes), but the fix is HALF-FINISHED by its own
# document: training walks 58% of the initialised selectivity back, and on the
# five attnfix runs the result was two convincing passes, one marginal, and --
#
#     temporal_k5_tattn_ring3   5.566 / 6 frames   92.8% of uniform   COLLAPSED
#
# -- the k5 TEMPORAL arm, which is this arm's own axis, failing the 0.9
# selectivity alarm outright. So the prior for this cell is NOT "attention
# selects and we measure what that buys". It is: attention may again average,
# in which case a tattn score at or below longreg_aughv means nothing about
# selection and the run has measured a differently-parameterised mean.
#
# WHAT MAKES THIS WORTH A SLOT ANYWAY, and it is specific rather than hopeful:
#   1. ctx50 MOVES THE ATTENTION FIELD from 12x6 to 18x12 -- 3x the positions,
#      each selecting over the same 6 frames. Every collapse measurement in that
#      document is at 12x6. Whether a larger field changes the collapse is
#      unmeasured, and it is the one thing this arm asks that no existing run does.
#   2. adamw at 3e-4 is a different optimiser and a 300x LARGER step than the
#      1e-6 rmsprop every collapsed run used, and the document is explicit that
#      the failure is a SCALE problem -- qk_norm's temperature decays under
#      training, and without qk_norm the same block "goes uniform at LR=1e-6 and
#      one-hot at LR=1e-5". No tattn run has ever been trained at 3e-4.
#
# THE ACCEPTANCE CHECK IS NOT val/dice. Before this arm is compared to anything,
# probe it -- and the probe is cheap and needs NO cluster job (MPS + the mounted
# patch tree, ~5 min/checkpoint):
#
#     python -m sinkholes attention-probe \
#         --model outputs/<this run>/checkpoints/best.pt \
#         --require_selectivity 0.9
#
# If it comes back at or near uniform, the arm has NOT tested attention and must
# be reported as "a mean, differently parameterised" -- exactly the error
# ATTENTION_COLLAPSE.md was written to stop being repeated. Probe BEFORE
# spending an eval slot on it.
#
# WHY hv EVERYWHERE, when AUGMENT=hv has never been run at all. It matches
# longreg_aughv. If the flips turn out to hurt (the fixed radar look geometry
# argument in longreg's header), they hurt all four cells, so every edge above
# stays single-lever -- the level moves, the gaps do not. What the square does
# NOT cover is augmentation itself; longreg_noaug has no twin here, and one is
# worth submitting only if the flips move the ConvLSTM.
#
# ---- memory, and why only one of these is expensive -------------------------
#
# longreg's formula, N x T x (ctx_px + 200*100) x 4 B, at N = 85,970:
#
#     cell                       T   ctx      predicted   request   note
#     longgrid_single_plain      1   none      12.8 GiB     48G
#     longgrid_single_ctx50      1   50x50     25.6 GiB     96G
#     longgrid_convlstm_plain    6   none      77   GiB    146G     <- see below
#     longreg_aughv (running)    6   50x50    154   GiB    256G     unprecedented
#
# THE 146G IS NOT MY ESTIMATE -- longreg's own header computes it, as the
# fallback if the 256G arms pend: "dropping ctx50 (clean + plain patches peaks
# at 77 GiB, a ~146G request, comfortably inside the 180G the project already
# runs)". This batch spends that fallback as an experiment rather than a
# retreat, and it is the reason the square is affordable at all.
#
# NOTHING HERE IS UNPRECEDENTED. Every request is inside the 180G this project
# already runs (RES_AMPFIX_CTX_LONG), so these three DO NOT depend on the 256G
# question. If the longreg arms turn out to have pended on memory, ALL THREE OF
# THESE STILL RUN -- and three cells of the square is still a decomposition,
# because longgrid_convlstm_plain alone answers "is the temporal model worth it"
# at the geometry this project has always used.
#
# ---- walltime, all measured ------------------------------------------------
#   single plain     2m13s/epoch at 49,604 (ampfix_t5_single) x1.73 -> ~4m/ep
#   single ctx50     ~3x the pixels at half the micro-batch         -> ~11m30s/ep
#   convlstm plain   3m31s/epoch at 49,604 (RES_AMPFIX_PLAIN) x1.73 -> ~6m05s/ep
# Each budget is ~2x its estimate, which covers the slower assembly at 2.1x the
# interferograms; --resume auto covers a requeue past that.
#
# ---- how these get scored, because both halves are easy to get wrong --------
#
# THE DATA IS EXACTLY MATCHED ACROSS ARCHITECTURES, VERIFIED. On the one
# partition where a single and a ConvLSTM twin were both run under the current
# negative sampling (ampfix_t5_single_ring3_200e vs ampfix_t5_convlstm_ring3_200e),
# the two logs agree line for line: 437 interferograms after the nonz filter,
# 214 with full 5-previous chains, train/val 49604/6072 in both. The
# single-frame path takes a different discovery branch (discovery_nonz is true
# only when --add_temporal is off) and still lands on the same set, so all four
# cells see the same 85,928/5,917 -- the same patches, the same negatives.
#
# EVAL MUST BE STRIDE 2 / PROTOCOL=prob FOR ALL FOUR CELLS, i.e. EVAL_S2_TEMP,
# not eval6's stride 4. The ctx50 patch tree exists ONLY at strpp2 (building one
# at strpp4 costs ~6.8 TB against 9.8 TB free), and run_eval.sh:241 refuses rth
# at any stride but 4 -- correctly, since under rth the thresholds
# 0.125/0.25/0.5 literally mean 2/4/8 of SIXTEEN overlapping tiles, and stride 2
# has four. THE TWO PLAIN CELLS COULD BE SCORED AT STRIDE 4 AND MUST NOT BE: a
# stride-4 number is not comparable with the ctx cells, which is the entire
# point of the square. ARCH=single for the two single cells, where K_PREVS is
# then ignored (run_eval.sh:209 passes --k_prevs 0).
#
# AND IT MUST BE THE temporal_k10 SCENE LIST (GEN=3). Scoring a _clean-trained
# checkpoint on any *_pre23_* protocol leaks all 35 of that list's
# interferograms into training and returns a number that looks fine --
# longreg's header verified k5_clean train vs k10_clean test at 0 overlap and
# vs k10_pre2023 test at 35. The same trap applies here unchanged.
#
# No eval jobs are listed for these yet, on purpose: the preflight checks
# RUN=<dir>/checkpoints/best.pt for every SELECTED job, so a row pointing at an
# unfinished run would make a bare `submit_all.sh` fail for every batch.
# The two single cells' requests (48G/24G/8:00 and 96G/36G/18:00) are recorded
# with their commands below; only the unsubmitted cell needs a live resource.
RES_LONGGRID_CONVLSTM="long-gpu       146   36  12:00"   # 77 GiB predicted, ~6h05m
# Same geometry, same depth, same dataset materialisation as longreg_aughv, so
# the same 154 GiB predicted peak and the same request. THE 256G IS NO LONGER
# UNPRECEDENTED: longreg's header flagged it as "not known to be schedulable",
# and LSF 697785/697786/697787 have since been GRANTED it and are training. That
# question is closed, and this arm is the first to spend the answer.
# Walltime from those same running jobs rather than from an estimate: 11m51s per
# epoch measured (697785 epoch 1 at 11m58s, epoch 2 at 23m49s) plus 1h43m of
# assembly before epoch 1, so 60 epochs is ~13h34m. 18:00 holds it in one
# allocation; --resume auto covers a requeue past that.
RES_LONGGRID_TATTN="long-gpu       256   48  18:00"   # 154 GiB predicted, ~13h34m
# Same host footprint as longgrid_convlstm_plain -- the dataset materialisation
# is architecture-independent, so plain + T=6 is 77 GiB either way.
# WALLTIME IS NOW MEASURED RATHER THAN SCALED, from the three arms of 2026-09-09:
#   longsingle_plain  3h03m/60 =  3m03s/ep      longsingle_ctx50 7h28m/60 = 7m28s/ep
#   longreg_aughv     9h03m/46 = 11m48s/ep
# ctx costs 2.45x at fixed architecture (7m28 / 3m03), so a plain T=6 arm is
# 11m48s / 2.45 = ~4m49s/epoch -> ~4h49m for 60, plus assembly. 12:00 holds it.
# NOTE the longreg header's own ctx estimate (~6m/epoch) came in at 11m48s, so
# scale-from-a-different-geometry has already been wrong by 2x once; these
# numbers are same-batch measurements and should be preferred.
# Estimated from the tattn ARM ITSELF rather than across architectures, which
# is the mistake that made convlstm_plain 1.8x its estimate: tattn ctx50 runs
# at 0.44x the ConvLSTM ctx50 rate (5m13s vs 11m48s), so tattn plain is
# ~0.44 x convlstm_plain's measured 8m50s = ~3m53s/epoch -> ~3h53m for 60.
RES_LONGGRID_TATTN_PLAIN="long-gpu 146   36  12:00"   # 77 GiB predicted, ~3h53m

# Everything longreg_aughv passes except the architecture and the geometry.
# LONGREG_* are reused verbatim rather than copied so that an edit to longreg's
# config follows into the square instead of silently splitting it.
LONGGRID_OPT="OPTIM=adamw LR=3e-4 WEIGHT_DECAY=1e-2 SEG_LOSS=dice"
# Micro-batch 64, effective 128 -- longreg_aughv's batching, at every cell.
LONGGRID_BATCH="BATCH=64 ACCUM=2"

# ARCH=single with RING_NEGS=yes still needs K_PREVS=5: it selects the chain the
# negative-exclusion grid is built from, so the control draws EXACTLY the
# negatives longreg drew (dataset.py:69-97, train.py:507-517). The template
# refuses a mismatch rather than silently drawing a different set.
LONGGRID_SINGLE="ARCH=single $LONGREG_BASE $LONGREG_NEG NEG_RING_OUTER=3 AUGMENT=hv"

# ---- SUBMITTED 2026-09-09, kept as a record and NOT as `job` rows ------------
# Same reason as the single cells below: a live row here would resubmit a
# running job. Their commands, verified against the launched configs:
#
#   env $LONGREG_BASE $LONGREG_NEG $LONGGRID_OPT DROPOUT_BOTTLENECK=0.2 \
#       $LONGREG_RUN $LONGGRID_BATCH NEG_RING_OUTER=3 AUGMENT=hv \
#       bsub -J longgrid_convlstm_plain -q long-gpu -R rusage[mem=146GB] \
#       -gpu num=1:j_exclusive=yes:gmem=36G -W 12:00 < scripts/train/train_convlstm.sh
#   env $LONGREG_BASE $LONGREG_NEG $LONGGRID_OPT $LONGREG_RUN $LONGREG_CTX \
#       NEG_RING_OUTER=3 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none \
#       bsub -J longgrid_tattn_ctx50 -q long-gpu -R rusage[mem=256GB] \
#       -gpu num=1:j_exclusive=yes:gmem=48G -W 18:00 < scripts/train/train_tattn.sh
#
# LSF 751271 longgrid_convlstm_plain -- 64x2, adamw 3e-4, dropout 0.2, T=6. The
#   context edge holding recurrence. 8m50s/epoch measured (7h57m at 54), i.e.
#   ~8h50m for 60 against its 12:00 wall -- 1.8x my 4m49s estimate, which was
#   scaled from the single-arm ctx ratio. Scaling across ARCHITECTURES is now
#   wrong by the same factor scaling across GEOMETRIES was; measure per arm.
# LSF 769020 longgrid_tattn_ctx50 -- 5m13s/epoch (1h39m at 19), so ~5h13m for
#   60 against an 18:00 wall. ATTENTION AT ctx50 IS 2.3x FASTER THAN THE
#   ConvLSTM AT ctx50 (11m48s), which is the one resource surprise in this grid:
#   the recurrence is serialised over T, the attention is not.
job longgrid longgrid_tattn_plain \
    scripts/train/train_tattn.sh "$LONGREG_BASE $LONGREG_NEG $LONGGRID_OPT $LONGREG_RUN $LONGGRID_BATCH NEG_RING_OUTER=3 AUGMENT=hv FUSE_SKIPS=0 RECURRENCE=none" "$RES_LONGGRID_TATTN_PLAIN" \
    "the only arm in the grid whose attention can actually be READ: the probe refuses context checkpoints, so this is where selectivity gets measured -- and it is the tattn row's own geometry control, differing from longgrid_tattn_ctx50 in the 12x6-vs-18x12 attention field and nothing else"
# ---- the two single cells: SUBMITTED 2026-09-09 13:30, NOT jobs any more ----
#
# Kept as a record and NOT as `job` rows, because this file's whole contract is
# that it holds jobs which have NEVER RUN -- a live row here would make
# `submit_all.sh longgrid --submit` resubmit two running jobs. Their exact
# commands, for a requeue or a rerun:
#
#   env ARCH=single $LONGGRID_SINGLE $LONGGRID_OPT $LONGREG_RUN BATCH=128 ACCUM=1 \
#       bsub -J longsingle_plain -q long-gpu -R rusage[mem=48GB] \
#       -gpu num=1:j_exclusive=yes:gmem=24G -W 8:00 < scripts/train/train_control.sh
#   env ARCH=single $LONGGRID_SINGLE $LONGGRID_OPT $LONGREG_RUN $LONGREG_CTX \
#       bsub -J longsingle_ctx50 -q long-gpu -R rusage[mem=96GB] \
#       -gpu num=1:j_exclusive=yes:gmem=36G -W 18:00 < scripts/train/train_control.sh
#
# LSF 721118 longsingle_plain -- MICRO-BATCH 128, NOT 64. It went out before the
# square fixed the micro-batch at 64, so it carries BATCH=128 ACCUM=1: the same
# EFFECTIVE batch as every other cell, but BatchNorm sees 128 samples per update
# where the other three see 64. Both edges it sits on (context-without-recurrence
# against longsingle_ctx50, recurrence-at-plain against longgrid_convlstm_plain)
# therefore carry a micro-batch difference on top of their stated lever.
#
# IT IS NOT WORTH RESTARTING, and the reason is quantitative: the run-to-run
# spread on this configuration is 0.011 (combo_ctx_p00's accidental duplicate,
# LSF 275654 vs 275798), and a BatchNorm sample count of 64 against 128 is well
# inside it -- both are large-batch estimates of the same statistics. Against
# that, a restart costs the ~4h already spent plus a fresh queue wait, and the
# edge it would clean is ALREADY caveated by the dropout asymmetry below. Note
# it when the numbers are read; do not spend GPU-hours on it.
#
# LSF 721120 longsingle_ctx50 -- 64x2, matching longreg exactly. Clean.

# ---- evallonggrid: object-level scores for the longgrid square ---------------
#
# ---- HAS RUN (2026-09-17). DO NOT SUBMIT AGAIN. ------------------------------
#   All six scored; results in docs/EXPERIMENTS.md (longgrid).
#
# The six cells of the longgrid square (see the longgrid block), none of which
# has ever been scored at object level. Each ROW is a single-lever context pair
# -- same partition (temporal_k5_clean), optimiser, pos_w 4, ring 1-3, hv flips
# and micro-batch 64, differing ONLY in plain 200x100 vs ctx50 300x200:
#
#                  plain                               ctx50
#   single     longsingle_plain         (721118)   longsingle_ctx50       (721120)
#   convlstm   longgrid_convlstm_plain  (751271)   longreg_aughv          (697785)
#   tattn      longgrid_tattn_plain     (811232)   longgrid_tattn_ctx50   (769020)
#
# WHY NOW. Best val/dice says context pays in every row, by a similar margin:
#     single   0.6658 -> 0.6900 (+0.024)   convlstm 0.6767 -> 0.6972 (+0.021)
#     tattn    0.6734 -> 0.7025 (+0.029)
# All three are ~2x the 0.011 run-to-run spread, but val/dice is positives-only
# and RESULTS.md 2 records it reversing against the object metric. These six
# rows say whether context really is the lever. Validation also has the plain
# single cell at the highest recall and lowest precision of the six
# (R 0.822, P 0.728) -- the same shape posw8_single_k10_plain showed on k10.
#
# TWO CELLS WERE KILLED BY OWNER, and neither loses its checkpoint.
# longgrid_tattn_ctx50 stopped at 30/60 (TERM_OWNER) with best.pt from epoch 19,
# and longreg_aughv at 52/60 with best.pt from epoch 16. No cell of the square
# improved after epoch 25, and SCHEDULE=plateau never reads EPOCHS, so a killed
# run's epochs 1..N match what the full run would have done.
#
# THE PROTOCOL IS evalampfix's, VERBATIM: $EVAL6_TEMP with K_PREVS=5 -- GEN=3,
# GROUP=temporal_k10 test list, MIN_POS=150 (20 scenes), stride 4, rth. That is
# the list the 0.7797 anchor was scored on and these runs were trained on its
# generation (clean k5, all dates < 20240101), so every row is readable against
# it. The ctx50 stride-4 tree was cut for exactly this list at k=5; the
# preflight re-derives and checks it.
#
# RESOURCES, carried from the same workload:
#   ctx50 temporal   RES_EVAL_AMPFIX   143.0 GB / 4875 s measured (LSF 412393,
#                                      ctx50 convlstm k5 on this list)
#   ctx50 single     RES_EVAL_POSW8_SINGLE  47.0 GB measured on the th350 list
#   plain temporal   RES_EVAL_S4_T5    eval6ref_t5_convlstm's request, same list
#   plain single     RES_EVAL_S4_CTRL  27.8 GiB measured
#
# ARCH: single -> ARCH=single; convlstm -> run_eval.sh's default; tattn ->
# ARCH=tattn. Context needs no flag -- io_geometry sets it per checkpoint.
R_LG_SINGLE_PLAIN=outputs/longsingle_plain_2026-09-09_13h30_lsf_721118
R_LG_SINGLE_CTX=outputs/longsingle_ctx50_2026-09-09_13h30_lsf_721120
R_LG_CONVLSTM_PLAIN=outputs/longgrid_convlstm_plain_2026-09-09_15h44_lsf_751271
R_LG_CONVLSTM_CTX=outputs/longreg_aughv_2026-09-09_11h06_lsf_697785
R_LG_TATTN_PLAIN=outputs/longgrid_tattn_plain_2026-09-10_00h30_lsf_811232
R_LG_TATTN_CTX=outputs/longgrid_tattn_ctx50_2026-09-09_17h52_lsf_769020

job evallonggrid evallonggrid_tattn_plain \
    scripts/eval/run_eval.sh "RUN=$R_LG_TATTN_PLAIN $EVAL6_TEMP K_PREVS=5 ARCH=tattn" "$RES_EVAL_S4_T5" \
    "attention at 200x100 -- the no-context half of the tattn pair, and the one tattn checkpoint the stock probe can read"
job evallonggrid evallonggrid_tattn_ctx50 \
    scripts/eval/run_eval.sh "RUN=$R_LG_TATTN_CTX $EVAL6_TEMP K_PREVS=5 ARCH=tattn" "$RES_EVAL_AMPFIX" \
    "attention at 300x200 -- best val/dice of the square (0.7025) and the widest context gain (+0.029); killed at 30/60, best.pt from epoch 19"
job evallonggrid evallonggrid_convlstm_plain \
    scripts/eval/run_eval.sh "RUN=$R_LG_CONVLSTM_PLAIN $EVAL6_TEMP K_PREVS=5" "$RES_EVAL_S4_T5" \
    "recurrence at 200x100 -- with dropout 0.2 on BOTH sides, this pair is the cleanest context edge in the square"
job evallonggrid evallonggrid_convlstm_ctx50 \
    scripts/eval/run_eval.sh "RUN=$R_LG_CONVLSTM_CTX $EVAL6_TEMP K_PREVS=5" "$RES_EVAL_AMPFIX" \
    "longreg_aughv, the run the square was built to decompose -- killed at 52/60, best.pt from epoch 16"
job evallonggrid evallonggrid_single_plain \
    scripts/eval/run_eval.sh "RUN=$R_LG_SINGLE_PLAIN $EVAL6_TEMP K_PREVS=5 ARCH=single" "$RES_EVAL_S4_CTRL" \
    "the floor at 200x100, and the recall outlier (val R 0.822, P 0.728) -- the k5 check on what single_k10_plain showed; micro-batch 128x1, the square's one off-grid cell"
job evallonggrid evallonggrid_single_ctx50 \
    scripts/eval/run_eval.sh "RUN=$R_LG_SINGLE_CTX $EVAL6_TEMP K_PREVS=5 ARCH=single" "$RES_EVAL_POSW8_SINGLE" \
    "the floor at 300x200 -- against single_plain, context without any temporal model"
