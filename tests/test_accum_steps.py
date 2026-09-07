"""Gradient accumulation: `accum_steps` micro-batches, one optimiser step.

Why it exists: a 300x200 context run cannot hold BATCH=128 in the VRAM its
200x100 twin used, and `batch_size` is a STRICT resume key with every preset at
128. Accumulating lets the large-context arm keep the *effective* batch of the
run it is compared against, so a difference in the result is the context and not
the optimiser -- the confound PLAN_LARGE_CONTEXT.md 6 warns about.

Note what is NOT claimed. `segmentation_loss` adds a Dice term that
`dice_coeff` pools over the WHOLE batch (`reduce_batch_first=True` sums over
(B, H, W)), so it is not decomposable: the Dice of four samples is not the mean
of the Dice of two halves. Accumulation therefore reproduces a single large
batch exactly only for the per-pixel part of the objective, which is what
test_accumulation_reproduces_the_single_batch_gradient pins by substituting a
decomposable loss. The BatchNorm in the real architectures is a second such
difference, stated in --accum_steps' help.
"""
import argparse

import pytest
import torch
import torch.nn as nn

from sinkholes.training import resume as R
from sinkholes.training.train import add_arguments, train_model

PATCH_H, PATCH_W = 16, 16


def make_args(**overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TinySet(list):
    mask_values = [0, 1]

    @classmethod
    def make(cls, n=4, seed=0):
        gen = torch.Generator().manual_seed(seed)
        return cls([{"image": torch.rand(1, PATCH_H, PATCH_W, generator=gen),
                     "mask": (torch.rand(PATCH_H, PATCH_W, generator=gen) > 0.7).long()}
                    for _ in range(n)])


class PlainNet(nn.Module):
    """BatchNorm-free on purpose: BN statistics depend on the micro-batch size,
    which would mask the arithmetic these tests are about."""

    def __init__(self):
        super().__init__()
        self.n_channels = 1
        self.n_classes = 1
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def run(tmp_path, *, batch_size, accum_steps, n=4, loss_fn=None, monkeypatch=None):
    """One epoch of the real loop; returns (final params, optimiser step count)."""
    if loss_fn is not None:
        monkeypatch.setattr("sinkholes.training.train.segmentation_loss", loss_fn)

    steps = []
    real_step = torch.optim.RMSprop.step

    def counting_step(self, *a, **kw):
        steps.append(1)
        return real_step(self, *a, **kw)

    torch.manual_seed(0)
    model = PlainNet()
    args = make_args(epochs=1, batch_size=batch_size, accum_steps=accum_steps, lr=1e-2,
                     patience=0, sample_every=0, job_name="accum", save_best_only=True,
                     seed=0, reporter=False)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(torch.optim.RMSprop, "step", counting_step)
        torch.manual_seed(123)          # fixes the shuffle for both arms
        train_model(args, model, torch.device("cpu"),
                    TinySet.make(n), TinySet.make(2, seed=1), None, str(tmp_path))
    return [p.detach().clone() for p in model.parameters()], len(steps)


def decomposable_loss(logits, images, true_masks, **kw):
    """Per-pixel mean BCE -- a mean over samples, so halves average exactly."""
    y = true_masks.float().unsqueeze(1)
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


# -- the mechanic ------------------------------------------------------------------------

def test_one_optimiser_step_per_accum_steps_batches(tmp_path):
    _, steps = run(tmp_path, batch_size=1, accum_steps=2, n=4)
    assert steps == 2, "4 batches at accum 2 must be 2 optimiser steps"


def test_accum_one_is_the_historical_path(tmp_path):
    _, steps = run(tmp_path, batch_size=1, accum_steps=1, n=4)
    assert steps == 4, "accum 1 must still step once per batch"


def test_a_partial_trailing_group_is_stepped_not_discarded(tmp_path):
    # 5 batches at accum 2 = two full groups plus a leftover. Dropping the
    # leftover would silently throw away a fifth of the epoch's gradient.
    _, steps = run(tmp_path, batch_size=1, accum_steps=2, n=5)
    assert steps == 3


# -- the arithmetic ----------------------------------------------------------------------

def test_accumulation_reproduces_the_single_batch_gradient(tmp_path, monkeypatch):
    """batch 4 x accum 1 == batch 2 x accum 2, for a decomposable objective."""
    big, big_steps = run(tmp_path / "a", batch_size=4, accum_steps=1,
                         loss_fn=decomposable_loss, monkeypatch=monkeypatch)
    split, split_steps = run(tmp_path / "b", batch_size=2, accum_steps=2,
                             loss_fn=decomposable_loss, monkeypatch=monkeypatch)
    assert big_steps == split_steps == 1
    for p_big, p_split in zip(big, split):
        torch.testing.assert_close(p_big, p_split, rtol=1e-5, atol=1e-6)


# -- the fingerprint ---------------------------------------------------------------------

def test_accum_steps_is_strict_so_a_resume_cannot_change_the_effective_batch():
    assert "accum_steps" in R.STRICT_CONFIG_KEYS


def test_accum_steps_is_absent_from_the_fingerprint_when_it_is_one():
    # Otherwise every checkpoint written before accumulation would fail to
    # resume against a config that now carries the key.
    assert R.run_config(make_args(accum_steps=1))["accum_steps"] is None
    assert R.run_config(make_args(accum_steps=2))["accum_steps"] == 2
