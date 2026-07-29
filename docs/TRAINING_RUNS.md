# Training Runs

Four runs, 27–29 Jul 2026, all on MPS with `pos_weight=8.0`, patches 200×100 (stride 2),
partition `random_by_intf`, patch dir `data_patches_H200_W100_strpp2_11days_Aligned`.

## Results

| Run | Model | Best dice | Ep | Epochs run | Wall |
|---|---|---|---|---|---|
| `convlstm_v1_2026-07-28_15h34` | ConvLSTM-UNet, T=3, hidden 1024 | **0.5913** | 19 | 30/30 | 2h14m |
| `run_v2_2026-07-27_14h44` | UNet, single frame (1 ch) | 0.5648 | 25 | 30/30 | 49m |
| `convlstm_v1_2026-07-28_22h12` | ConvLSTM-UNet, T=3, hidden 1024 | 0.5473 | 68 | 93/100 (early stop) | 5h16m |
| `unet_temporal_v1_2026-07-28_14h25` | UNet, 3 frames stacked as channels | 0.3415 | 16 | 17/30 (interrupted) | 31m |

Metrics at each run's best epoch:

| Run | train/loss | val/loss | dice | IoU | F1 | P | R |
|---|---|---|---|---|---|---|---|
| convlstm 15h34 | 0.667 | 0.757 | 0.5913 | 0.5516 | 0.7110 | 0.6306 | 0.8151 |
| run_v2 | 0.825 | — | 0.5648 | — | — | 0.5603 | 0.8285 |
| convlstm 22h12 | 0.433 | 0.894 | 0.5473 | 0.4790 | 0.6477 | 0.5712 | 0.7480 |
| unet_temporal | 1.258 | 1.154 | 0.3415 | 0.2368 | 0.3830 | 0.2581 | 0.7417 |

`run_v2` predates the val/loss, IoU and F1 columns — its CSV only has dice/P/R.

## Test-set check (best run)

Run on the held-out test set of the best model (`convlstm 15h34`, 637 patches):

```bash
python -m sinkholes test-patches \
    --test_data_path outputs/convlstm_v1_2026-07-28_15h34/test_dataset_convlstm_v1_2026-07-28_15h34.pkl \
    --model outputs/convlstm_v1_2026-07-28_15h34/checkpoints/best.pt \
    --k_prevs 2
```

| metric | value |
|---|---|
| mean dice (per-patch) | 0.546 |
| pixel precision / recall | 0.58 / 0.74 |
| object-level precision / recall | 0.85 / 0.80 |

Test dice 0.546 vs val 0.591 — a modest drop, so the val figure is not badly overfit. Object-level
scores are much stronger than pixel-level, i.e. the model **finds** the right sinkholes but its
boundaries are loose. For an operational detector that is the favourable direction.

## Hyperparameters

| Run | Batch | LR | Train/val/test patches |
|---|---|---|---|
| convlstm 15h34 | 16 | 1e-5 | 8966 / 1153 / 637 |
| convlstm 22h12 | 32 | 1e-5 | 8430 / 1126 / 1200 |
| run_v2 | 8 | 1e-4 | 8469 / 632 / 724 |
| unet_temporal | 8 | 1e-4 | 9574 / 556 / 626 |

## Takeaways

- **ConvLSTM at batch 16 is the best model so far** — top score on every metric (dice 0.591,
  IoU 0.552, F1 0.711), reached by epoch 19 of 30.
- **The longer ConvLSTM run was worse, not better.** At batch 32 for 93 epochs (5h16m) it hit
  only 0.547. Train loss fell much further (0.43 vs 0.67) while val loss rose off its 0.847
  floor — it overfit. Early stopping fired after 25 epochs without improvement. Larger batch +
  more epochs is the wrong direction here.
- **Channel-stacking frames into a plain UNet does not work.** At 0.342 it is far below even the
  single-frame baseline, and training was unstable (train loss spiked 1.20 → 1.84 around
  epoch 14). ConvLSTM's recurrent handling of the time axis is doing real work.
- **The single-frame baseline is more competitive than expected** — 0.565 in 49 minutes, beating
  the 5-hour ConvLSTM run. Temporal modelling only pays off in the batch-16 configuration.
- **All runs are recall-heavy** (R 0.74–0.83 vs P 0.26–0.63), as expected from `pos_weight=8.0`.
  Precision is the bottleneck on the good runs.

⚠️ **Cross-run comparison is approximate.** `random_by_intf` reshuffles per run, so all four have
different train/val/test splits (val ranges 556–1153 patches). Differences of a few points may be
split noise. A fixed split is needed before trusting a ranking this close.

## Artifacts

Per run under `outputs/<run>/`: `results.csv` (per-epoch metrics), `logs/reporter.log`
(full log), `curves.png`, `checkpoints/best.pt` + `last.pt`, `validation/preds/epoch_*.png`,
and a pickled test set. `run_v2` predates the validation-preview feature so it has no
`validation/preds/` (its `curves.png` was backfilled with `train_reporter.py`);
`unet_temporal` also has `checkpoints/interrupted.pt` from the interrupt.

Regenerate any run's curves with:

```bash
python -m sinkholes curves outputs/<run>/
```

Disk use is 3.0 GB total, almost all checkpoints (2.2 GB) and pickled test sets (820 MB);
the validation PNGs are only 26 MB.
