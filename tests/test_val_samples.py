"""Choice of the per-epoch validation sample grid: spatial separation between
the patches shown, and a spread over their ground-truth area."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sinkholes.training.evaluate import evaluate

DEVICE = torch.device("cpu")
H, W = 20, 10
MIN_SEP = 4


class ConstantNet(torch.nn.Module):
    """Logit 0 everywhere -> probability 0.5 everywhere."""

    n_classes = 1

    def forward(self, x):
        return torch.zeros(x.shape[0], 1, H, W)

    def train(self, mode=True):  # evaluate() puts the net back in train mode
        return self


class FakeVal(Dataset):
    """Patches whose GT area cycles 1..20, so a spread is checkable.

    The image is a constant plane holding the sample's own index, which lets a
    test recover which patches were chosen.
    """

    def __init__(self, n=80, empties=()):
        self.n = n
        self.empties = set(empties)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        mask = np.zeros((H, W), dtype=np.float32)
        if i not in self.empties:
            mask.reshape(-1)[: 1 + (i % 20)] = 1
        return {"image": torch.full((1, H, W), float(i)),
                "mask": torch.as_tensor(mask).long()}


def pick(dataset, n=4, min_sep=MIN_SEP):
    """Run a validation pass; return (indices shown, their GT areas, samples)."""
    samples = {"n": n, "channel": 0, "min_sep": min_sep}
    loader = DataLoader(dataset, shuffle=False, drop_last=True, batch_size=1)
    evaluate(ConstantNet(), loader, DEVICE, False, mode="val", samples_out=samples)
    if "image" not in samples:
        return [], [], samples
    return ([int(img[0, 0]) for img in samples["image"]],
            [int(gt.sum()) for gt in samples["gt"]],
            samples)


def separated(indices, min_sep=MIN_SEP):
    return all(abs(a - b) >= min_sep for a in indices for b in indices if a != b)


def test_samples_are_not_neighbouring_windows():
    indices, _, _ = pick(FakeVal())
    assert len(indices) == 4
    assert separated(indices), indices


def test_samples_span_the_range_of_ground_truth_area():
    _, areas, samples = pick(FakeVal())
    assert areas == sorted(areas), "the grid should read sparsest to densest"
    assert areas[0] <= 2, areas
    assert areas[-1] >= 19, areas
    assert samples["n_positive"] == 4


def test_a_larger_grid_still_spans_and_separates():
    indices, areas, _ = pick(FakeVal(), n=8)
    assert len(areas) == 8
    assert areas == sorted(areas)
    assert areas[0] <= 2 and areas[-1] >= 19, areas
    assert separated(indices), indices


def test_selection_is_deterministic_across_passes():
    assert pick(FakeVal())[0] == pick(FakeVal())[0]


def test_empty_patches_only_pad_a_grid_the_positives_cannot_fill():
    _, areas, samples = pick(FakeVal(n=40, empties=range(2, 40)))
    assert len(areas) == 4
    assert samples["n_positive"] == 2
    assert sorted(areas) == [0, 0, 1, 2]


def test_an_all_empty_validation_set_still_yields_a_separated_grid():
    indices, areas, samples = pick(FakeVal(n=40, empties=range(40)))
    assert areas == [0, 0, 0, 0]
    assert samples["n_positive"] == 0
    assert separated(indices), indices


def test_separation_is_relaxed_rather_than_returning_a_short_grid():
    # Three patches in total cannot be four apart; the grid is filled anyway.
    indices, areas, _ = pick(FakeVal(n=3))
    assert indices == [0, 1, 2]
    assert areas == [1, 2, 3]
