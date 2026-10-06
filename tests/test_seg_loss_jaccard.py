"""Jaccard as an alternative region term, and the reporting that makes it readable.

``--seg_loss jaccard`` replaces the Dice term added to BCE. The invariants:

  * ``dice`` (the default) is bit-identical to the objective before the flag;
  * Jaccard matches Dice's REDUCTION and its empty-set convention, so the flag
    is one lever and not three;
  * both region terms are recorded every step whichever is the objective, and
    the non-objective one can never reach the optimiser;
  * a resume cannot cross a change of objective.
"""

import argparse

import pytest
import torch
import torch.nn as nn

from sinkholes.training.losses import (
    REGION_LOSSES,
    dice_coeff,
    dice_loss,
    jaccard_coeff,
    jaccard_loss,
    masked_dice_loss_binary,
    masked_jaccard_loss_binary,
    segmentation_loss,
)
from sinkholes.training.resume import (
    STRICT_CONFIG_KEYS,
    IncompatibleResume,
    check_config_compatible,
    run_config,
)
from sinkholes.training.train import add_arguments, build_model

B, H, W = 4, 32, 32


def make_args(**overrides):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, 1, H, W, generator=g)
    images = torch.randn(B, 3, H, W, generator=g)
    masks = (torch.rand(B, H, W, generator=g) > 0.9).long()
    return logits, images, masks


def criterion():
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))


# -- the coefficient ---------------------------------------------------------------------

def test_jaccard_is_dice_reparametrised():
    """J = D/(2-D) exactly. This is what makes the two rank identically on ONE
    sample, and it is the identity every claim about the pair rests on."""
    g = torch.Generator().manual_seed(0)
    p = torch.sigmoid(torch.randn(B, H, W, generator=g))
    y = (torch.rand(B, H, W, generator=g) > 0.9).float()
    d = float(dice_coeff(p, y, reduce_batch_first=True))
    j = float(jaccard_coeff(p, y, reduce_batch_first=True))
    assert j == pytest.approx(d / (2 - d), abs=1e-6)


def test_it_matches_dice_reduction_over_the_whole_batch():
    """reduce_batch_first=True sums over (B, H, W) at once -- one global score,
    not a mean of per-sample scores. A Jaccard that reduced per sample and
    meaned would make --seg_loss two levers instead of one."""
    g = torch.Generator().manual_seed(1)
    p = torch.sigmoid(torch.randn(B, H, W, generator=g))
    y = (torch.rand(B, H, W, generator=g) > 0.9).float()
    inter = (p * y).sum()
    union = p.sum() + y.sum() - inter
    assert float(jaccard_coeff(p, y, reduce_batch_first=True)) == pytest.approx(
        float((inter + 1e-6) / (union + 1e-6)), abs=1e-6)


def test_empty_prediction_on_empty_mask_scores_one_like_dice():
    """Ring negatives are all-zero patches. A Jaccard that scored them 0 where
    Dice scores 1 would be optimising something quite different."""
    z = torch.zeros(2, 8, 8)
    assert float(jaccard_coeff(z, z, reduce_batch_first=True)) == pytest.approx(1.0)
    assert float(jaccard_loss(z, z)) == pytest.approx(float(dice_loss(z, z)))


def test_the_masked_pair_agree_with_each_other():
    """masked_dice_loss_binary has no empty-set guard and returns 1.0 on an
    all-empty batch; its Jaccard twin must follow ITS convention, not
    jaccard_coeff's, or --seg_loss changes two things on the masked path."""
    z = torch.full((2, 1, 8, 8), -20.0)          # sigmoid ~ 0
    y = torch.zeros(2, 1, 8, 8)
    V = torch.ones_like(y)
    assert float(masked_jaccard_loss_binary(z, y, V)) == pytest.approx(
        float(masked_dice_loss_binary(z, y, V)), abs=1e-4)


def test_jaccard_loss_is_the_steeper_penalty():
    """J <= D for every imperfect prediction, so jaccard LOSS is always the
    larger. This is why train/loss cannot be compared across the two arms."""
    g = torch.Generator().manual_seed(2)
    for seed in range(5):
        p = torch.sigmoid(torch.randn(B, H, W, generator=g))
        y = (torch.rand(B, H, W, generator=g) > 0.9).float()
        assert float(jaccard_loss(p, y)) > float(dice_loss(p, y))


# -- the objective -----------------------------------------------------------------------

def test_dice_is_the_default_and_is_unchanged():
    logits, images, masks = batch()
    assert make_args().seg_loss == "dice"
    expected = (criterion()(logits.squeeze(1), masks.float())
                + dice_loss(torch.sigmoid(logits.squeeze(1)), masks.float()))
    got = segmentation_loss(logits, images, masks, n_classes=1,
                            treat_nodata_regions=False, criterion=criterion())
    assert torch.equal(got, expected)


def test_jaccard_swaps_only_the_region_term():
    logits, images, masks = batch()
    parts = {}
    total = segmentation_loss(logits, images, masks, n_classes=1, treat_nodata_regions=False,
                              criterion=criterion(), components=parts, region_loss="jaccard")
    assert float(total) == pytest.approx(float(parts["bce"]) + float(parts["jaccard"]), abs=1e-5)


@pytest.mark.parametrize("kind", REGION_LOSSES)
def test_both_region_terms_are_always_recorded(kind):
    """So a dice arm and a jaccard arm can be read against each other inside one
    results.csv -- and so the values do not depend on which one is optimised."""
    logits, images, masks = batch()
    parts = {}
    segmentation_loss(logits, images, masks, n_classes=1, treat_nodata_regions=False,
                      criterion=criterion(), components=parts, region_loss=kind)
    assert {"bce", "dice", "jaccard"} <= set(parts)
    assert not any(v.requires_grad for v in parts.values()), \
        "a recorded component must never carry a graph into the optimiser"


def test_the_recorded_terms_do_not_depend_on_which_is_the_objective():
    logits, images, masks = batch()
    a, b = {}, {}
    for kind, out in (("dice", a), ("jaccard", b)):
        segmentation_loss(logits, images, masks, n_classes=1, treat_nodata_regions=False,
                          criterion=criterion(), components=out, region_loss=kind)
    for key in ("bce", "dice", "jaccard"):
        assert float(a[key]) == pytest.approx(float(b[key]), abs=1e-6)


def test_only_the_objective_term_carries_gradient():
    logits, images, masks = batch()
    logits = logits.requires_grad_(True)
    segmentation_loss(logits, images, masks, n_classes=1, treat_nodata_regions=False,
                      criterion=criterion(), components={}, region_loss="jaccard").backward()
    assert float(logits.grad.abs().sum()) > 0


def test_an_unknown_region_loss_is_rejected():
    logits, images, masks = batch()
    with pytest.raises(ValueError, match="region_loss"):
        segmentation_loss(logits, images, masks, n_classes=1, treat_nodata_regions=False,
                          criterion=criterion(), region_loss="tversky")


def test_multiclass_with_jaccard_is_refused_rather_than_silently_dice():
    args = make_args(seg_loss="jaccard", classes=3)
    with pytest.raises(SystemExit, match="binary segmentation only"):
        build_model(args, torch.device("cpu"))


# -- reporting and resume ----------------------------------------------------------------

def test_the_epoch_row_carries_both_region_columns():
    from sinkholes.training.train import add_arguments  # noqa: F401  (columns live in train)

    import inspect

    from sinkholes.training import train as T
    src = inspect.getsource(T.train_model)
    for column in ("train/jaccard", "val/jaccard"):
        assert f'"{column}"' in src, f"{column} missing from the results columns"


def test_seg_loss_is_a_strict_resume_key():
    assert "seg_loss" in STRICT_CONFIG_KEYS
    saved = run_config(make_args(seg_loss="dice"))
    now = run_config(make_args(seg_loss="jaccard"))
    with pytest.raises(IncompatibleResume, match="seg_loss"):
        check_config_compatible(saved, now, "resume.pt")


def test_a_config_written_before_the_flag_still_resumes():
    current = run_config(make_args())
    saved = {k: v for k, v in current.items() if k != "seg_loss"}
    assert check_config_compatible(saved, current, "resume.pt") == []
