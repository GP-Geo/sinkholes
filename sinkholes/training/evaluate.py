"""Validation/test evaluation: Dice, pixel precision/recall, object-level metrics."""

import logging
from typing import Callable, Optional

import numpy as np
import rasterio.features
import torch
import torch.nn.functional as F
from affine import Affine
from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm import tqdm

from ..device import memory_format_for
from .features import compute_feature
from .losses import dice_coeff, multiclass_dice_coeff

logger = logging.getLogger(__name__)


def calc_precision_recall(y_pred: np.ndarray, y_true: np.ndarray):
    """Pixel precision/recall of one batch (nan on an all-negative batch)."""
    precision = np.sum(y_pred * y_true) / np.sum(y_pred)
    recall = np.sum(y_pred * y_true) / np.sum(y_true)
    return precision, recall


def object_level_evaluate(
    gt: np.ndarray,
    pred: np.ndarray,
    image: Optional[np.ndarray] = None,
    features=(),
    epsilon: float = 1e-7,
    th: float = 0.7,
    buffer: float = 5,
    round_ndigits: Optional[int] = 2,
):
    """Object-level recall/precision over a batch of binary masks.

    Masks are polygonised per patch. A ground-truth object counts as detected
    when the fraction of its area covered by the union of predictions buffered
    by ``buffer`` pixels exceeds ``th``. A predicted object whose fractional
    overlap with the buffered ground-truth union is below ``th`` contributes
    its full area as false positive.

    Batch numbers are weighted by each patch's ground-truth area. With a
    non-empty ``features`` list, per-object features are collected separately
    for detected and undetected ground-truth objects.

    ``round_ndigits`` rounds the per-patch and returned figures, and defaults
    to 2 because every archived number was produced that way. Full-scene
    scoring passes ``None``: at scene scale the rounding happens BEFORE the
    area weighting, so a 0.005 quantum on each scene propagates into the
    aggregate, and model deltas in this project are routinely smaller than
    that. Leave the default alone for the patch-level path, whose numbers are
    published at 2 decimals anyway.

    Returns (recall, precision, batch_gt_area, {feature: [detected, undetected]}).
    """
    def _round(x):
        return x if round_ndigits is None else round(x, round_ndigits)

    feature_lists = {feature: [[], []] for feature in features}
    if image is not None and image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    transform = Affine.identity()
    intersect_recall, intersect_precision, patch_gt_areas = [], [], []

    for i in range(gt.shape[0]):
        gt_polygons = [shape(g) for g, v in rasterio.features.shapes(gt[i], transform=transform) if v > 0]
        pred_polygons = [shape(g) for g, v in rasterio.features.shapes(pred[i], transform=transform) if v > 0]

        total_gt_area = sum(p.area for p in gt_polygons)
        patch_gt_areas.append(total_gt_area)

        buffered_pred_union = unary_union([pp.buffer(buffer) for pp in pred_polygons])
        pred_union = unary_union(pred_polygons)

        intersection_area = 0.0
        for pgt in gt_polygons:
            covered_fraction = pgt.intersection(buffered_pred_union).area / pgt.area
            detected = covered_fraction > th
            if detected:
                intersection_area += pgt.area
            for feat in feature_lists:
                # image[i] is the patch's (T, H, W) input for phase features.
                patch_image = image[i] if image is not None else None
                if patch_image is not None and patch_image.ndim == 2:
                    patch_image = np.expand_dims(patch_image, axis=0)
                feature_lists[feat][0 if detected else 1].append(
                    compute_feature(pgt, feat, patch_image)
                )

        buffered_gt_union = unary_union([pgt.buffer(buffer) for pgt in gt_polygons])
        fp_area = sum(
            pp.area for pp in pred_polygons
            if pp.intersection(buffered_gt_union).area / pp.area < th
        )

        recall_i = _round(intersection_area / (total_gt_area + epsilon))
        precision_i = _round(intersection_area / (intersection_area + fp_area + epsilon))
        intersect_recall.append(recall_i)
        intersect_precision.append(min(precision_i, 1.0))

    batch_gt_area = float(np.sum(patch_gt_areas))
    ol_recall = _round(np.sum(np.array(intersect_recall) * np.array(patch_gt_areas)) / batch_gt_area)
    ol_precision = _round(np.sum(np.array(intersect_precision) * np.array(patch_gt_areas)) / batch_gt_area)
    return ol_recall, ol_precision, batch_gt_area, feature_lists


#: Candidate patches for the sample grid, as (sample index, positive pixels,
#: image, ground truth, probability). Sample index is the dataset index when
#: the loader is unshuffled, and :class:`SubsiDataset` lays patches out in grid
#: order, so neighbouring indices are windows overlapping by half a patch.
#: Index distance is therefore a cheap stand-in for spatial distance. It is
#: conservative in two ways: it also separates across interferogram boundaries
#: (harmless — a few candidates are passed over), and it cannot recognise
#: vertical neighbours, which sit a whole grid row apart.


def _prune_pool(pool, cap: int, min_sep: int) -> None:
    """Trim the candidate pool back to `cap`, keeping its range of GT areas.

    Dropped first is any sample crowded against another pooled one, then the
    most redundant by area: sorted by positive-pixel count, the entry whose
    two neighbours are closest together. Only interior entries are considered,
    so the emptiest and fullest patches seen always survive and what remains
    stays spread across the range.
    """
    while len(pool) > cap:
        order = sorted(range(len(pool)), key=lambda k: (pool[k][1], pool[k][0]))

        def redundancy(p):
            k = order[p]
            crowded = any(abs(pool[k][0] - pool[j][0]) < min_sep
                          for j in range(len(pool)) if j != k)
            return (0 if crowded else 1,
                    pool[order[p + 1]][1] - pool[order[p - 1]][1], k)

        victims = [redundancy(p) for p in range(1, len(order) - 1)]
        pool.pop(min(victims)[2] if victims else len(pool) - 1)


def _spread_samples(pool, n: int, min_sep: int):
    """`n` candidates spanning the pool's positive-pixel range, sparsest first.

    The targets are `n` evenly spaced ranks in the area-sorted pool — so the
    grid runs from a patch with barely any sinkhole to the fullest one seen.
    Each target takes the nearest-ranked candidate still `min_sep` samples
    clear of everything already picked; separation is relaxed only to top up a
    grid that would otherwise come out short.
    """
    ranked = sorted(pool, key=lambda c: (c[1], c[0]))
    if not ranked or n <= 0:
        return []
    if n == 1:
        return [ranked[len(ranked) // 2]]

    used = set()

    def take(target: int, sep: int) -> None:
        for off in range(len(ranked)):
            for p in ((target,) if off == 0 else (target - off, target + off)):
                if 0 <= p < len(ranked) and p not in used and all(
                    abs(ranked[p][0] - ranked[q][0]) >= sep for q in used
                ):
                    used.add(p)
                    return

    targets = [round(i * (len(ranked) - 1) / (n - 1)) for i in range(n)]
    for t in targets:
        take(t, min_sep)
    for t in targets:
        if len(used) >= min(n, len(ranked)):
            break
        take(t, 1)
    return [ranked[p] for p in sorted(used)]


@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    amp: bool = False,
    *,
    mode: str = "val",
    epoch: int = 1,
    out_path: Optional[str] = None,
    save_val: bool = False,
    th: float = 0.7,
    buffer: float = 5,
    metrics_out: Optional[dict] = None,
    loss_fn: Optional[Callable] = None,
    samples_out: Optional[dict] = None,
):
    """One pass over a loader; returns the mean per-batch Dice (macro).

    ``metrics_out``: pass a dict to additionally receive pooled pixel counts
    and precision/recall/IoU/F1. Note the pooled F1 (micro) is *not* the
    returned Dice (macro, per-batch mean) — a patch with 3 positive pixels
    weighs as much as one with 3000 in the latter. Both are correct.

    ``loss_fn``: optional ``f(logits, images, mask_true) -> tensor`` evaluated
    on the pre-threshold logits and averaged into ``metrics_out['loss']``.
    Training passes the very objective it optimises, so val/loss is directly
    comparable with train/loss.

    ``samples_out``: pass ``{'n': k, 'channel': c}`` to receive ``image``,
    ``gt`` and ``prob`` arrays (k, H, W) for a few patches — positives
    preferred, deterministic when the loader is not shuffled. ``channel``
    selects which input channel is shown (the current frame for temporal
    stacks). The k patches are chosen to span the range of ground-truth area
    seen during the pass, ordered sparsest to densest, and ``'min_sep'``
    (default 4) keeps them at least that many samples apart so the grid never
    shows the same sinkhole through overlapping windows. Empty patches are
    used only to top up a grid the positives could not fill.

    ``mode='test'`` additionally reports pixel precision/recall and the
    object-level metrics above.
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    precision = recall = 0
    ol_precision = ol_recall = 0
    b_gts = []
    _tp = _fp = _fn = 0.0
    _loss_sum, _loss_n = 0.0, 0
    _n_samples = int(samples_out.get("n", 4)) if samples_out is not None else 0
    _min_sep = max(1, int(samples_out.get("min_sep", 4))) if samples_out is not None else 1
    # Candidates are kept as (index, positive pixels, image, gt, prob). The cap
    # bounds what is held on the host: a few MB at 200x100, whatever the val
    # set size, while still being wide enough to choose a spread from.
    _pool_cap = max(6 * _n_samples, 24)
    _samp_pos, _samp_any = [], []
    _seen = 0
    pred_batches, image_batches, true_mask_batches = [], [], []

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation round",
                          unit="batch", leave=False):
            image, mask_true = batch["image"], batch["mask"]
            image = image.to(device=device, dtype=torch.float32,
                             memory_format=memory_format_for(device))
            mask_true = mask_true.to(device=device, dtype=torch.long)

            logits = net(image)

            if net.n_classes == 1:
                mask_true = mask_true.unsqueeze(1)
                assert mask_true.min() >= 0 and mask_true.max() <= 1, \
                    "true mask indices must be in [0, 1]"
                mask_prob = F.sigmoid(logits)
                if loss_fn is not None:
                    _l = float(loss_fn(logits, image, mask_true))
                    if np.isfinite(_l):
                        _loss_sum += _l
                        _loss_n += 1
                mask_pred = (mask_prob > 0.5).float()

                mask_pred_np = mask_pred.squeeze(1).cpu().numpy()
                mask_true_np = mask_true.squeeze(1).cpu().numpy()
                image_np = image.squeeze(1).cpu().numpy()

                if _n_samples:
                    _ch = samples_out.get("channel")
                    _ch = image.shape[1] - 1 if _ch is None else int(_ch)
                    _ch = min(max(_ch, 0), image.shape[1] - 1)
                    for _i in range(mask_true_np.shape[0]):
                        # Decide from the counts already on the host, so the
                        # extra device copies happen only for a kept candidate.
                        _idx = _seen + _i
                        _cnt = float(mask_true_np[_i].sum())
                        if _cnt == 0 and (
                            len(_samp_any) >= _n_samples
                            or any(abs(_idx - c[0]) < _min_sep for c in _samp_any)
                        ):
                            continue  # empties are only ever padding
                        _pool = _samp_pos if _cnt > 0 else _samp_any
                        _pool.append((_idx, _cnt, image[_i, _ch].cpu().numpy(),
                                      mask_true_np[_i], mask_prob[_i, 0].cpu().numpy()))
                        if _cnt > 0:
                            _prune_pool(_samp_pos, _pool_cap, _min_sep)
                _seen += mask_true_np.shape[0]

                if save_val:
                    if epoch % 2 == 0:
                        pred_batches.append(mask_pred_np)
                    if epoch == 1:
                        image_batches.append(image_np)
                        true_mask_batches.append(mask_true_np)

                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

                if metrics_out is not None:
                    _p = mask_pred_np.astype(bool)
                    _g = mask_true_np.astype(bool)
                    _tp += float(np.logical_and(_g, _p).sum())
                    _fp += float(np.logical_and(~_g, _p).sum())
                    _fn += float(np.logical_and(_g, ~_p).sum())

                if mode == "test":
                    batch_p, batch_r = calc_precision_recall(mask_pred_np, mask_true_np)
                    olr, olp, b_gt_a, _ = object_level_evaluate(
                        mask_true_np, mask_pred_np, image_np, features=(), th=th, buffer=buffer
                    )
                    precision += batch_p
                    recall += batch_r
                    if not np.isnan(olp) and not np.isnan(olr):
                        ol_precision += olp * b_gt_a
                        ol_recall += olr * b_gt_a
                        b_gts.append(b_gt_a)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes
                mask_true_1h = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_1h = F.one_hot(logits.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(
                    mask_pred_1h[:, 1:], mask_true_1h[:, 1:], reduce_batch_first=False
                )

        if save_val and out_path is not None:
            if epoch == 1 and mode == "val" and image_batches:
                np.save(f"{out_path}/image_valid_test", np.concatenate(image_batches))
                np.save(f"{out_path}/mask_true_valid", np.concatenate(true_mask_batches))
            if epoch % 2 == 0 and pred_batches:
                np.save(f"{out_path}/mask_pred_valid_epoch{epoch}", np.concatenate(pred_batches))

    net.train()
    mean_dice = dice_score / max(num_val_batches, 1)

    if metrics_out is not None:
        metrics_out["tp"], metrics_out["fp"], metrics_out["fn"] = _tp, _fp, _fn
        metrics_out["precision"] = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        metrics_out["recall"] = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        metrics_out["n_batches"] = num_val_batches
        den = _tp + _fp + _fn
        metrics_out["iou"] = _tp / den if den else 0.0
        metrics_out["f1"] = 2 * _tp / (2 * _tp + _fp + _fn) if den else 0.0
        if _loss_n:
            metrics_out["loss"] = _loss_sum / _loss_n

    if samples_out is not None:
        picked = _spread_samples(_samp_pos, _n_samples, _min_sep)
        if len(picked) < _n_samples:  # too few positives — pad with empty patches
            picked = picked + _samp_any[:_n_samples - len(picked)]
        if picked:
            samples_out["image"] = np.stack([p[2] for p in picked])
            samples_out["gt"] = np.stack([p[3] for p in picked])
            samples_out["prob"] = np.stack([p[4] for p in picked])
            samples_out["n_positive"] = sum(1 for p in picked if p[1] > 0)

    if mode == "test":
        results = {
            "mean_dice": float(mean_dice),
            "pixel_precision": round(precision / num_val_batches, 2),
            "pixel_recall": round(recall / num_val_batches, 2),
            "ol_precision": round(ol_precision / sum(b_gts), 2) if b_gts else float("nan"),
            "ol_recall": round(ol_recall / sum(b_gts), 2) if b_gts else float("nan"),
        }
        print(f"mean dice score:  {results['mean_dice']}")
        print(f"Mean pixel level Precision: {results['pixel_precision']}")
        print(f"Mean pixel level Recall: {results['pixel_recall']}")
        print(f"Mean OL Precision: {results['ol_precision']}")
        print(f"Mean OL Recall: {results['ol_recall']}")
        if metrics_out is not None:
            metrics_out["test"] = results

    return mean_dice
