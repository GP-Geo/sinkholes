"""Segmentation losses: Dice, and the masked/unmasked training objectives."""

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-6


# -- Dice -------------------------------------------------------------------------------

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False,
               epsilon: float = 1e-6) -> Tensor:
    """Soft Dice, averaged over the batch (or over one mask)."""
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    return ((inter + epsilon) / (sets_sum + epsilon)).mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False,
                          epsilon: float = 1e-6) -> Tensor:
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# -- masked (validity-aware) losses -----------------------------------------------------

def masked_bce_with_logits(logits, y_float, V, pos_w: float = 1.0):
    """BCE-with-logits averaged over valid pixels only.

    logits, y_float, V: (B, 1, H, W); V is 1 on valid pixels, 0 on no-data.
    """
    pw = torch.as_tensor([pos_w], device=logits.device, dtype=logits.dtype)
    per_pix = F.binary_cross_entropy_with_logits(logits, y_float, reduction="none", pos_weight=pw)
    w = V.detach()
    return (per_pix * w).sum() / w.sum().clamp(min=1.0)


def masked_dice_loss_binary(logits, y_float, V):
    """Soft Dice on probabilities, restricted to valid pixels."""
    p = torch.sigmoid(logits)
    num = (2.0 * (p * y_float * V)).sum()
    den = (p * V).sum() + (y_float * V).sum() + EPS
    return 1.0 - num / den


def masked_ce_plus_softdice_multiclass(logits, y_long, V):
    """Masked CrossEntropy + mean soft Dice over classes.

    logits (B, C, H, W); y_long (B, H, W); V (B, 1, H, W).
    """
    ce_per_pix = F.cross_entropy(logits, y_long, reduction="none")
    ce = (ce_per_pix * V.squeeze(1)).sum() / V.sum().clamp(min=1.0)

    probs = F.softmax(logits, dim=1) * V
    y1 = F.one_hot(y_long, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float() * V
    num = (2.0 * (probs * y1)).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + y1.sum(dim=(0, 2, 3)) + EPS
    dice = 1.0 - (num / den).mean()
    return ce + dice


# -- the training objective -------------------------------------------------------------

def segmentation_loss(logits, images, true_masks, *, n_classes, treat_nodata_regions, criterion):
    """The per-batch objective. Validation calls this same function, so
    val/loss is directly comparable with train/loss.

    ``images`` is (B, T, H, W) — or (B, 2T, H, W) with validity channels, in
    which case the second half is the per-time validity block and ``V_any``
    (valid at any timestep) masks the loss. ``true_masks`` may arrive as
    (B, H, W) from the training loop or (B, 1, H, W) from evaluation.
    """
    if true_masks.dim() == 4 and true_masks.shape[1] == 1:
        true_masks = true_masks.squeeze(1)

    if treat_nodata_regions:
        T = images.shape[1] // 2
        V = images[:, T:, ...]
        V_any = V.max(dim=1, keepdim=True).values  # (B, 1, H, W)
    else:
        V_any = None

    if n_classes == 1:
        y_float = true_masks.float().unsqueeze(1)

        if treat_nodata_regions:
            # pos_w here is fixed at 8.0, deliberately independent of --pos_w:
            # the masked objective was tuned with it and reported runs rely on it.
            loss_bce = masked_bce_with_logits(logits, y_float, V_any, pos_w=8.0)
            loss_dice = masked_dice_loss_binary(logits, y_float, V_any)

            # Suppress false positives inside no-data areas, tolerating small
            # holes near real targets (dilate GT by r_tol pixels first).
            r_tol = 2
            lam = 0.2
            y_dil = F.max_pool2d(y_float, kernel_size=2 * r_tol + 1, stride=1, padding=r_tol)
            M_fp = (1.0 - V_any) * (1.0 - y_dil)
            if M_fp.sum() > 0:
                bce_fp = F.binary_cross_entropy_with_logits(
                    logits, torch.zeros_like(y_float), reduction="none"
                )
                bce_fp = (bce_fp * M_fp).sum() / M_fp.sum().clamp(min=1.0)
            else:
                bce_fp = logits.new_tensor(0.0)
            return loss_bce + loss_dice + lam * bce_fp

        loss = criterion(logits.squeeze(1), y_float.squeeze(1))
        loss += dice_loss(torch.sigmoid(logits.squeeze(1)), y_float.squeeze(1), multiclass=False)
        return loss

    if treat_nodata_regions:
        return masked_ce_plus_softdice_multiclass(logits, true_masks, V_any)
    return criterion(logits, true_masks)
