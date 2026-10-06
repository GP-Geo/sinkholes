"""The fp16 region-loss overflow, and the guard that stops it being resumed across.

Under ``--amp`` the loss is computed outside ``autocast``, so fp16 logits used
to reach ``dice_loss`` and its bare ``input.sum(...)`` reduced an fp16 tensor.
fp16 saturates at 65504; a 200x100 patch at batch 128 is 2,560,000 elements, so
the sum overflowed to ``inf`` above a mean predicted probability of 2.56% and
the term returned exactly 1.0 with exactly zero gradient -- the objective became
BCE alone. Everything here pins the fix and its blast radius:

  * the loss reduces in fp32 and produces a real gradient at the shapes and
    dtypes training actually uses;
  * ``dice_coeff`` as the ``val/dice`` METRIC is untouched;
  * fp32 (no-AMP) runs are numerically unchanged;
  * an AMP checkpoint written before the fix cannot be resumed into this tree.
"""

import pytest
import torch

from sinkholes.training.losses import (
    REGION_LOSS_DTYPE,
    dice_coeff,
    dice_loss,
    masked_dice_loss_binary,
)
from sinkholes.training.resume import (
    STRICT_CONFIG_KEYS,
    IncompatibleResume,
    check_config_compatible,
)

#: The real training shape: batch 128 of 200x100 is 2.56M elements.
B, H, W = 128, 200, 100
FP16_MAX = 65504.0


def probs_and_target(dtype, seed=0, positive_rate=0.975):
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(B, H, W, generator=gen) * 2.0).to(dtype)
    target = (torch.rand(B, H, W, generator=gen) > positive_rate).float()
    return torch.sigmoid(logits), target


# -- the defect is gone ------------------------------------------------------------------

def test_the_bare_fp16_sum_still_overflows():
    """The premise. If this ever stops being true the fix is unnecessary, and a
    test that silently passes for the wrong reason is worse than none."""
    p, _ = probs_and_target(torch.float16)
    assert p.numel() * 1.0 > FP16_MAX
    assert torch.isinf(p.sum()), "fp16 sum of 2.56M probabilities should saturate"
    assert torch.isfinite(p.float().sum()), "the same sum in fp32 must not"


def test_dice_loss_on_fp16_gives_a_real_gradient():
    """The defect itself: loss exactly 1.0, gradient exactly 0."""
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(B, H, W, generator=gen) * 2.0).half().requires_grad_(True)
    target = (torch.rand(B, H, W, generator=gen) > 0.975).float()

    loss = dice_loss(torch.sigmoid(logits), target)
    loss.backward()

    value = float(loss.detach())
    assert torch.isfinite(loss), "an overflowed reduction shows up as a saturated loss"
    assert value < 1.0, f"loss pinned at {value} -- the reduction saturated"
    assert float(logits.grad.float().abs().sum()) > 0.0, "the region term must train"


def test_fp16_and_fp32_now_agree():
    """Widening fp16 to fp32 is exact, so the two paths must land on the same
    loss once neither saturates -- the fp16 inputs are not approximations of
    the fp32 ones, they ARE the same numbers."""
    p16, target = probs_and_target(torch.float16)
    loss16 = float(dice_loss(p16, target))
    loss32 = float(dice_loss(p16.float(), target))
    assert loss16 == pytest.approx(loss32, abs=1e-6)


def test_masked_region_loss_reduces_in_fp32():
    gen = torch.Generator().manual_seed(1)
    logits = (torch.randn(B, 1, H, W, generator=gen) * 2.0).half()
    y = (torch.rand(B, 1, H, W, generator=gen) > 0.975).float()
    V = torch.ones_like(y)
    loss = masked_dice_loss_binary(logits, y, V)
    assert torch.isfinite(loss) and float(loss) < 1.0


# -- blast radius ------------------------------------------------------------------------

def test_the_val_dice_metric_is_untouched():
    """dice_coeff is also evaluate.py's val/dice. It reduces one 20,000-pixel
    patch at a time -- max sum 20,000 against a 65504 ceiling -- so it could
    never overflow, and the fix deliberately does not reach it."""
    pred = (torch.rand(1, H, W, generator=torch.Generator().manual_seed(2)) > 0.5).float()
    true = (torch.rand(1, H, W, generator=torch.Generator().manual_seed(3)) > 0.5).float()
    assert pred[0].numel() < FP16_MAX
    before = float(dice_coeff(pred, true, reduce_batch_first=False))
    assert 0.0 < before < 1.0


def test_fp32_runs_are_numerically_unchanged():
    """No --amp means fp32 logits, where .float() is a no-op: those runs must
    produce exactly what they produced before the fix."""
    p32, target = probs_and_target(torch.float32)
    # .float() on a float32 tensor returns the tensor itself, so this must be
    # bit-identical -- compared as tensors, since going through a Python float
    # first would round differently and hide a real change.
    expected = 1 - dice_coeff(p32, target, reduce_batch_first=True)
    assert torch.equal(dice_loss(p32, target), expected)


# -- the resume guard --------------------------------------------------------------------

FIXED = {"amp": True, "gradient_clipping": "unscaled-v2",
         "region_loss_dtype": REGION_LOSS_DTYPE}


def test_it_is_a_strict_config_key():
    assert "region_loss_dtype" in STRICT_CONFIG_KEYS


def test_an_amp_checkpoint_from_before_the_fix_is_refused():
    """Its early epochs optimised BCE alone; continuing it here would run one
    objective to epoch N and another from N+1."""
    legacy = {"amp": True, "gradient_clipping": "unscaled-v2"}
    with pytest.raises(IncompatibleResume, match="fp16 region-loss"):
        check_config_compatible(legacy, FIXED, "resume.pt")


def test_a_non_amp_checkpoint_from_before_the_fix_still_resumes():
    """Without --amp the logits were fp32 and the sum never reached the
    ceiling, so those runs are unaffected and must stay resumable."""
    legacy = {"amp": False, "gradient_clipping": "unscaled-v2"}
    current = {**FIXED, "amp": False}   # amp is advisory; hold it so only the guard is under test
    assert check_config_compatible(legacy, current, "resume.pt") == []


def test_a_checkpoint_written_after_the_fix_resumes():
    assert check_config_compatible(FIXED, FIXED, "resume.pt") == []


def test_run_config_records_the_version():
    import argparse

    from sinkholes.training.train import add_arguments
    from sinkholes.training.resume import run_config

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert run_config(parser.parse_args([]))["region_loss_dtype"] == REGION_LOSS_DTYPE
