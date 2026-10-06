"""Segmentation losses: Dice, and the masked/unmasked training objectives."""

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-6

#: Version of the region-term reduction, recorded in ``run_config`` exactly as
#: ``gradient_clipping`` is, because it changes what every AMP run optimises.
#:
#: THE DEFECT (fixed 2026-09-08). The loss is computed outside ``autocast``, so
#: under ``--amp`` the model's fp16 logits reached these functions unchanged and
#: the bare ``input.sum(...)`` in ``dice_coeff`` reduced an fp16 tensor. fp16
#: saturates at 65504, and a 200x100 patch at batch 128 is 2,560,000 elements:
#: the sum overflows to ``inf`` whenever the MEAN predicted probability exceeds
#: 2.56%, the ratio collapses to 0, and ``dice_loss`` returns exactly 1.0 with
#: exactly ZERO gradient. The objective silently became BCE-only.
#:
#: Measured, at the training path's own dtypes and shapes:
#:     fp16   p.sum()=1280413 (inf)  loss=1.000000  |grad|=0.000e+00
#:     fp32   p.sum()=1280322        loss=0.952090  |grad|=2.759e-02
#: and the ceiling is crossed above mean p of 10.2% / 5.1% / 2.56% at batch
#: 32 / 64 / 128. It shows in the logs as ``train/dice`` pinned at exactly 1.0
#: (LSF 644244, batch 128: epochs 1-3 exactly 1.0, 0.9985 at 4-5, decaying to
#: 0.62 by epoch 10 while ``train/bce`` sat flat at 0.058 -> 0.042).
#:
#: THE FIX. Every region term widens to fp32 before it reduces. Widening fp16
#: to fp32 is exact, so the inputs are unchanged and the only difference is
#: that the reduction no longer saturates. ``dice_coeff`` itself is deliberately
#: NOT touched: it is also the ``val/dice`` metric (evaluate.py), which reduces
#: one 20,000-pixel patch at a time and can never reach the ceiling.
REGION_LOSS_DTYPE = "fp32-v2"


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
    """Dice as a LOSS, reduced over the whole batch and always in fp32.

    This is the site the fp16 overflow reached: ``reduce_batch_first=True``
    sums over (B, H, W) at once, which is 2.5M elements at batch 128 against
    fp16's 65504 ceiling. See ``REGION_LOSS_DTYPE``. The cast is a no-op
    without ``--amp``, where the logits are fp32 already.
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input.float(), target.float(), reduce_batch_first=True)


# -- Jaccard (IoU) -----------------------------------------------------------------------

#: The region terms ``--seg_loss`` selects between.
REGION_LOSSES = ("dice", "jaccard")


def jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False,
                  epsilon: float = 1e-6) -> Tensor:
    """Soft Jaccard (IoU), reduced EXACTLY as ``dice_coeff`` is.

    Matching the reduction is what makes ``--seg_loss`` a one-lever change:
    ``reduce_batch_first=True`` sums over (B, H, W) at once, so this is one
    global soft IoU per batch, not a mean of per-sample IoUs, exactly like
    ``dice_coeff``.

    The empty-set guard mirrors Dice's for the same reason. When prediction and
    target are both empty ``union`` is 0, and ``where`` substitutes ``inter``
    so the ratio is ``eps/eps = 1`` -- the "an empty prediction on an empty
    mask is perfect" convention. That is not academic here: ring negatives are
    all-zero patches, so a run that scored them 0 instead of 1 would be
    optimising something quite different from its Dice twin.

    Related to Dice by ``J = D / (2 - D)``, so on ONE sample the two rank
    identically. They do not rank identically once averaged over samples --
    ``J`` is convex in ``D``, so ``mean(J) != J(mean(D))``.
    """
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - inter
    union = torch.where(union == 0, inter, union)
    return ((inter + epsilon) / (union + epsilon)).mean()


def multiclass_jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False,
                             epsilon: float = 1e-6) -> Tensor:
    return jaccard_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def jaccard_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    """Jaccard as a LOSS. fp32 for the same reason as ``dice_loss``.

    The union is ``a.sum() + b.sum() - inter``, i.e. the identical bare sums --
    so this would have inherited the fp16 overflow exactly. See
    ``REGION_LOSS_DTYPE``.
    """
    fn = multiclass_jaccard_coeff if multiclass else jaccard_coeff
    return 1 - fn(input.float(), target.float(), reduce_batch_first=True)


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
    # Widened for the same reason as dice_loss. This path reduces in fp32
    # today only because y_float and V are fp32 and promote the products --
    # true by accident, and not what REGION_LOSS_DTYPE should rest on. The
    # widening is exact, so this path's numbers do not move.
    p = torch.sigmoid(logits).float()
    num = (2.0 * (p * y_float * V)).sum()
    den = (p * V).sum() + (y_float * V).sum() + EPS
    return 1.0 - num / den


def masked_jaccard_loss_binary(logits, y_float, V):
    """Soft Jaccard on probabilities, restricted to valid pixels.

    Deliberately mirrors ``masked_dice_loss_binary``'s own convention rather
    than ``jaccard_coeff``'s: that function has no empty-set guard either, and
    returns a loss of 1.0 when everything is empty. The two masked terms have
    to agree with each other or ``--seg_loss`` stops being one lever.
    """
    p = torch.sigmoid(logits).float()
    inter = (p * y_float * V).sum()
    union = (p * V).sum() + (y_float * V).sum() - inter
    return 1.0 - inter / (union + EPS)


def masked_ce_plus_softdice_multiclass(logits, y_long, V):
    """Masked CrossEntropy + mean soft Dice over classes.

    logits (B, C, H, W); y_long (B, H, W); V (B, 1, H, W).
    """
    ce_per_pix = F.cross_entropy(logits, y_long, reduction="none")
    ce = (ce_per_pix * V.squeeze(1)).sum() / V.sum().clamp(min=1.0)

    probs = F.softmax(logits, dim=1).float() * V  # fp32 reduction, see REGION_LOSS_DTYPE
    y1 = F.one_hot(y_long, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float() * V
    num = (2.0 * (probs * y1)).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + y1.sum(dim=(0, 2, 3)) + EPS
    dice = 1.0 - (num / den).mean()
    return ce + dice


# -- the training objective -------------------------------------------------------------

def segmentation_loss(logits, images, true_masks, *, n_classes, treat_nodata_regions, criterion,
                      components=None, region_loss: str = "dice"):
    """The per-batch objective. Validation calls this same function, so
    val/loss is directly comparable with train/loss.

    ``region_loss`` selects the region term: "dice" (the default, and every run
    before 2026-09-08) or "jaccard". Only the BINARY paths implement the
    choice; ``n_classes > 1`` with "jaccard" raises rather than silently
    training Dice.

    NOTE that ``train/loss`` is NOT comparable across the two. Jaccard loss
    exceeds Dice loss at the same prediction for every J < 1, so a jaccard arm
    reads higher throughout and only shapes and validation metrics compare.

    Pass a dict as ``components`` to receive the individual terms (detached)
    under "bce", "dice" and "jaccard". The total is the only thing training
    uses; the split is what says WHICH term a plateau is stuck on, which the
    total cannot. BOTH region terms are always recorded, whichever is the
    objective, so a dice arm and a jaccard arm can be read against each other
    inside one results.csv -- the one that is not the objective is computed
    under ``no_grad`` and cannot reach the optimiser.

    ``images`` is (B, T, H, W) — or (B, 2T, H, W) with validity channels, in
    which case the second half is the per-time validity block and ``V_any``
    (valid at any timestep) masks the loss. ``true_masks`` may arrive as
    (B, H, W) from the training loop or (B, 1, H, W) from evaluation.
    """
    if region_loss not in REGION_LOSSES:
        raise ValueError(f"region_loss must be one of {REGION_LOSSES}; got {region_loss!r}.")

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
            region_fn = (masked_jaccard_loss_binary if region_loss == "jaccard"
                         else masked_dice_loss_binary)
            loss_region = region_fn(logits, y_float, V_any)

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
            if components is not None:
                components["bce"] = (loss_bce + lam * bce_fp).detach()
                components[region_loss] = loss_region.detach()
                other = "dice" if region_loss == "jaccard" else "jaccard"
                with torch.no_grad():
                    components[other] = (masked_dice_loss_binary if other == "dice"
                                         else masked_jaccard_loss_binary)(logits, y_float, V_any)
            return loss_bce + loss_region + lam * bce_fp

        bce = criterion(logits.squeeze(1), y_float.squeeze(1))
        probs, y = torch.sigmoid(logits.squeeze(1)), y_float.squeeze(1)
        region_fn = jaccard_loss if region_loss == "jaccard" else dice_loss
        region = region_fn(probs, y, multiclass=False)
        if components is not None:
            components["bce"] = bce.detach()
            components[region_loss] = region.detach()
            other = "dice" if region_loss == "jaccard" else "jaccard"
            with torch.no_grad():
                components[other] = (dice_loss if other == "dice" else jaccard_loss)(
                    probs, y, multiclass=False)
        return bce + region

    if region_loss != "dice":
        raise ValueError(
            f"region_loss={region_loss!r} is implemented for binary segmentation only; "
            f"this run has n_classes={n_classes}. The multiclass objectives still use Dice."
        )
    if treat_nodata_regions:
        return masked_ce_plus_softdice_multiclass(logits, true_masks, V_any)
    return criterion(logits, true_masks)
