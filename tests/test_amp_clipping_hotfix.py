"""Run the actual optimiser boundaries with a real, enabled CPU GradScaler."""
import argparse

import pytest
import torch

from sinkholes.training import train


class ScalarNet(torch.nn.Module):
    n_channels = n_classes = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5))

    def forward(self, x):
        return self.weight * torch.ones_like(x)


class Samples(list):
    mask_values = [0, 1]


@pytest.mark.parametrize("gradient", [.25, 4.0])
@pytest.mark.parametrize("accum,n", [(1, 2), (2, 4), (2, 3)])
def test_amp_clips_true_gradients_at_full_and_partial_boundaries(tmp_path, monkeypatch, gradient, accum, n):
    real_scaler = torch.amp.GradScaler
    scaler = real_scaler("cpu", enabled=True, init_scale=65536., growth_interval=1000)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda **kw: scaler)
    real_loader = train.DataLoader
    def local_loader(*args, **kwargs):
        kwargs.update(num_workers=0, pin_memory=False)
        return real_loader(*args, **kwargs)
    monkeypatch.setattr(train, "DataLoader", local_loader)
    monkeypatch.setattr(train, "segmentation_loss", lambda logits, *a, **kw: logits.mean() * gradient)
    monkeypatch.setattr(train, "evaluate", lambda *a, **kw: torch.tensor(.5))
    before_clip, at_step = [], []
    real_clip = torch.nn.utils.clip_grad_norm_
    real_step = torch.optim.RMSprop.step
    def clip(parameters, *args, **kwargs):
        params = list(parameters)
        before_clip.append(float(params[0].grad))
        return real_clip(params, *args, **kwargs)
    def step(optimizer, *args, **kwargs):
        at_step.append(float(optimizer.param_groups[0]["params"][0].grad))
        return real_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(torch.optim.RMSprop, "step", step)
    parser = argparse.ArgumentParser(); train.add_arguments(parser)
    args = parser.parse_args(["--amp"])
    args.epochs = 1; args.batch_size = 1; args.accum_steps = accum
    args.reporter = False; args.sample_every = 0; args.patience = 0; args.seed = 1
    args.save_best_only = True
    samples = Samples([{"image": torch.ones(1, 4, 4), "mask": torch.ones(4, 4).long()} for _ in range(n)])
    train.train_model(args, ScalarNet(), torch.device("cpu"), samples, samples, None, str(tmp_path))
    # Keep the existing loss/accum weighting, including the trailing group.
    expected = [gradient] * (n // accum)
    if n % accum:
        expected.append(gradient * (n % accum) / accum)
    assert before_clip == pytest.approx(expected)
    clipped = [g * min(1., 1. / (g + 1e-6)) for g in expected]
    assert at_step == pytest.approx(clipped)
    assert min(at_step) > .01   # the old path produced roughly 1 / 65536
    assert scaler.is_enabled() and scaler.get_scale() == 65536.
    saved = torch.load(tmp_path / "checkpoints" / "resume.pt", weights_only=False)
    assert saved["config"]["gradient_clipping"] == "unscaled-v2"
