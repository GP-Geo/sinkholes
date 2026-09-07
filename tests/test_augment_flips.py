"""Flip augmentation: the train split only, and the mask must follow the image.

Why it exists: a flip that moves the image but not the target silently trains
the model on wrong labels, and nothing downstream would catch it -- the shapes
still agree. With a context sample the danger is sharper, because image and
mask are different sizes and only the CENTRE of the image corresponds to the
mask at all.
"""

import numpy as np
import pytest
import torch

from sinkholes.dataprep.dataset import SubsiDataset

H, W = 8, 6


def make(mode="train", flips="none", ctx=False, mark=(1, 2)):
    """A dataset stub carrying exactly what __getitem__ reads."""
    ch, cw = (H + 4, W + 4) if ctx else (H, W)
    ds = object.__new__(SubsiDataset)
    # every pixel distinct, so any flip is detectable and reversible
    ds.image_data = [np.arange(ch * cw, dtype=np.float32).reshape(1, 1, ch, cw)]
    msk = np.zeros((1, 1, H, W), dtype=np.float32)
    msk[0, 0, mark[0], mark[1]] = 1.0
    ds.mask_data = [msk]
    ds.index_map = [(0, 0)]
    ds.temporal = False
    ds.mask_values = [0, 1]
    ds.n_value_channels = None
    ds.preprocessing_version = "frame-v2"
    ds.patch_size = (H, W)
    ds.context_size = (ch, cw)
    ds.mode = mode
    ds.augment_flips = flips
    return ds


def image_value_under_mask(sample, ctx):
    """The image value at the pixel the mask marks, through the centre crop."""
    img = sample["image"][0].numpy()
    msk = sample["mask"].numpy()
    r, c = map(int, np.argwhere(msk == 1)[0])
    my = (img.shape[0] - msk.shape[0]) // 2
    mx = (img.shape[1] - msk.shape[1]) // 2
    return float(img[my + r, mx + c])


@pytest.mark.parametrize("ctx", [False, True])
@pytest.mark.parametrize("flips", ["h", "v", "hv"])
def test_mask_follows_the_image_through_every_flip(flips, ctx):
    """Whatever the flip did, the marked pixel still sits on the same value."""
    expected = image_value_under_mask(make(flips="none", ctx=ctx)[0], ctx)
    seen = set()
    for seed in range(60):
        torch.manual_seed(seed)
        sample = make(mode="train", flips=flips, ctx=ctx)[0]
        assert image_value_under_mask(sample, ctx) == expected
        seen.add(tuple(np.argwhere(sample["mask"].numpy() == 1)[0]))
    assert len(seen) > 1, "60 draws produced no flip at all -- augmentation is inert"


@pytest.mark.parametrize("mode", ["val", "test"])
def test_val_and_test_are_never_augmented(mode):
    """A moving validation target makes the epoch-to-epoch curve meaningless."""
    ref = make(mode=mode, flips="none")[0]
    for seed in range(40):
        torch.manual_seed(seed)
        got = make(mode=mode, flips="hv")[0]
        assert torch.equal(got["image"], ref["image"])
        assert torch.equal(got["mask"], ref["mask"])


def test_none_is_the_untouched_path():
    torch.manual_seed(0)
    a = make(mode="train", flips="none")[0]
    torch.manual_seed(1)
    b = make(mode="train", flips="none")[0]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["mask"], b["mask"])


def test_context_geometry_survives_a_flip():
    s = make(mode="train", flips="hv", ctx=True)[0]
    assert tuple(s["image"].shape) == (1, H + 4, W + 4)
    assert tuple(s["mask"].shape) == (H, W)


def test_unknown_flip_spec_is_refused_at_the_cli():
    """The guard a user actually meets. (SubsiDataset.__init__ checks too, but
    the stub above builds via object.__new__ and so cannot reach it.)"""
    import argparse

    from sinkholes.training.train import add_arguments

    p = argparse.ArgumentParser()
    add_arguments(p)
    assert p.parse_args(["--augment_flips", "hv"]).augment_flips == "hv"
    assert p.parse_args([]).augment_flips == "none", "augmentation must be opt-in"
    with pytest.raises(SystemExit):
        p.parse_args(["--augment_flips", "diagonal"])


def test_weight_decay_is_exposed_and_defaults_to_the_historical_value():
    import argparse

    from sinkholes.training.train import add_arguments
    from sinkholes.training.resume import STRICT_CONFIG_KEYS, run_config

    p = argparse.ArgumentParser()
    add_arguments(p)
    assert p.parse_args([]).weight_decay == 1e-8, "no earlier run may change"
    cfg = run_config(p.parse_args(["--weight_decay", "1e-4", "--augment_flips", "h"]))
    assert cfg["weight_decay"] == 1e-4 and cfg["augment_flips"] == "h"
    # both change what the model sees, so a resume must not silently cross them
    assert "weight_decay" in STRICT_CONFIG_KEYS
    assert "augment_flips" in STRICT_CONFIG_KEYS
