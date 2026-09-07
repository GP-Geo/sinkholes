"""Equivalent frame content and ground selection across dataset paths."""
import argparse
import json
import pickle

import numpy as np
import pytest
import torch

from sinkholes.dataprep.dataset import RingNegatives, SubsiDataset, load_test_dataset
from sinkholes.dataprep.patchify import patch_file_name
from sinkholes.geo import FRAME_ORIGINS, PIXEL_DEG, grid_window
from sinkholes.normalise import normalise_channels, PATCH_RANGE_TOL
from sinkholes.training import resume
from sinkholes.training.train import add_arguments

TARGET = (4, 4)
CURRENT = "20200112_20200123"
PREVIOUS = "20200101_20200112"


@pytest.fixture
def tree(tmp_path):
    image_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    image_dir.mkdir(); mask_dir.mkdir()
    raw = np.array([[.2, .4, .6, .8], [-2., 2., -.5, .1],
                    [.1, .2, .3, .4], [-1., 1., 0., 0.]], dtype=np.float32)
    images = np.tile(raw, (3, 3, 1, 1))
    for r in range(3):
        for c in range(3):
            images[r, c, -1, -1] = -(1 + 3*r + c) / 10
    masks = np.ones((3, 3, *TARGET), np.uint8)
    coords = [[r, c] for r in range(3) for c in range(3)]
    for tid in (PREVIOUS, CURRENT):
        for kind, root, array in [("data", image_dir, images), ("mask", mask_dir, masks)]:
            np.save(root / patch_file_name(kind, tid, *TARGET, 2), array)
            np.save(root / patch_file_name(kind, tid, *TARGET, 2, nonz=True), array.reshape(-1, *TARGET))
    (image_dir / "nonz_indices.json").write_text(json.dumps({CURRENT: coords, PREVIOUS: coords}))
    x, y = FRAME_ORIGINS["North"]
    d, eps = PIXEL_DEG, 1e-10
    box = (y - 6*d - eps, y - 2*d + eps, x + 2*d - eps, x + 6*d + eps)
    assert grid_window("North", *box, patch_size=TARGET, stride=(2, 2)) == (1, 2, 1, 2)
    return image_dir, mask_dir, images[1, 1].copy(), box


def dataset(tree, **kw):
    image_dir, mask_dir, raw, box = tree
    options = dict(patch_size=TARGET, mode="val", coord_dict={CURRENT: {"frame": "North"}},
                   seq_dict={CURRENT: {"prevs": [PREVIOUS]}}, aoi_window=box)
    options.update(kw)
    return SubsiDataset(image_dir, mask_dir, [CURRENT], **options)


def test_same_raw_frame_normalises_equally_single_and_temporal(tree):
    single = dataset(tree)
    temporal = dataset(tree, temporal=True)
    assert len(single) == len(temporal) == 1
    torch.testing.assert_close(single[0]["image"][0], temporal[0]["image"][-1], rtol=0, atol=0)
    expected = (tree[2] + np.pi) / (2*np.pi)
    np.testing.assert_array_equal(single[0]["image"][0].numpy(), expected)


@pytest.mark.parametrize("mode", ["train", "val", "test"])
@pytest.mark.parametrize("nonz_only", [True, False])
def test_single_aoi_keeps_same_target_as_temporal(tree, mode, nonz_only):
    single = dataset(tree, mode=mode, nonz_only=nonz_only)
    temporal = dataset(tree, mode=mode, temporal=True)
    assert len(single) == len(temporal) == 1
    torch.testing.assert_close(single[0]["mask"], temporal[0]["mask"])
    assert single.image_data[0].shape[1] == 1
    np.testing.assert_array_equal(single.image_data[0][0, 0], tree[2])


def test_ring_training_and_positives_only_validation_share_aoi(tree):
    # Every cell is positive here: the ring contributes no negative samples.
    train = dataset(tree, mode="train", ring_negatives=RingNegatives(seed=1))
    val = dataset(tree, mode="val", ring_negatives=RingNegatives(seed=1))
    assert len(train) == len(val) == 1


@pytest.mark.parametrize("temporal", [False, True])
def test_outside_aoi_loads_no_samples(tree, temporal):
    with pytest.raises(RuntimeError, match="no patches loaded"):
        dataset(tree, temporal=temporal, aoi_window=(50, 51, 50, 51))


def test_legacy_selection_and_normalisation_remain_explicit(tree):
    legacy = dataset(tree, aoi_selection_version="legacy-v1", preprocessing_version="legacy-row-v1")
    assert len(legacy) == 9
    np.testing.assert_array_equal(legacy[0]["image"][0, 0].numpy(), tree[2][0])
    corrected = dataset(tree, aoi_selection_version="legacy-v1")
    assert not torch.equal(legacy[0]["image"], corrected[0]["image"])


def test_from_arrays_defaults_to_frame_and_old_pickle_keeps_row_policy(tmp_path):
    raw = np.array([[[.2, .4], [-2., 2.]]], np.float32)
    ds = SubsiDataset.from_arrays(raw, np.ones_like(raw))
    expected = normalise_channels(raw[0], range_tol=PATCH_RANGE_TOL)
    np.testing.assert_array_equal(ds[0]["image"][0], expected)
    del ds.preprocessing_version  # schema of historical pickles
    file = tmp_path / "old.pkl"
    file.write_bytes(pickle.dumps(ds))
    old = load_test_dataset(file)
    np.testing.assert_array_equal(old[0]["image"][0, 0], raw[0, 0])


def test_corrected_normalisation_does_not_modify_validity_channels():
    stack = np.array([[[-2., 2.]], [[0., 1.]]], np.float32)
    out = normalise_channels(stack, range_tol=PATCH_RANGE_TOL, n_channels=1)
    np.testing.assert_array_equal(out[1], stack[1])


def config(**overrides):
    p = argparse.ArgumentParser(); add_arguments(p)
    args = p.parse_args([])
    for key, value in overrides.items(): setattr(args, key, value)
    return resume.run_config(args)


def test_old_resume_cannot_silently_change_patch_policies():
    old = config()
    del old["preprocessing_version"]; del old["aoi_selection_version"]
    with pytest.raises(resume.IncompatibleResume, match="preprocessing_version"):
        resume.check_config_compatible(old, config(), "old.pt")
    resume.check_config_compatible(old, config(preprocessing_version="legacy-row-v1",
                                              aoi_selection_version="legacy-v1"), "old.pt")


def test_legacy_amp_resume_is_refused_before_it_can_change_optimisation():
    old = config(amp=True)
    del old["gradient_clipping"]
    with pytest.raises(resume.IncompatibleResume, match="legacy AMP"):
        resume.check_config_compatible(old, config(amp=True), "old.pt")


def test_checkpoint_loader_preserves_data_contract_and_old_defaults():
    from sinkholes.models.factory import build_from_checkpoint
    from sinkholes.models.unet import UNet
    model = UNet(1, 1)
    state = model.state_dict()
    state["data_contract"] = {"preprocessing_version": "frame-v2", "aoi_selection_version": "coordinates-v2"}
    loaded = build_from_checkpoint(state)
    loaded.model.load_state_dict(state)
    assert loaded.data_contract["preprocessing_version"] == "frame-v2"
    old = build_from_checkpoint(model.state_dict())
    assert old.data_contract["preprocessing_version"] == "legacy-row-v1"


@pytest.mark.parametrize("legacy,context", [(True, (4, 4)), (False, (4, 4)), (False, (8, 8))])
def test_patch_test_cli_resolves_checkpoint_policies_and_context(tree, tmp_path, monkeypatch, legacy, context):
    from sinkholes.dataprep.patchify import patch_dir_name
    from sinkholes.inference import patch_test
    from sinkholes.models.factory import LoadedModel
    from test_context_scene_hotfix import CentreIdentity

    images, masks, raw, box = tree
    margin = tuple((c - p)//2 for c, p in zip(context, TARGET))
    if context != TARGET:
        for path in images.glob("*.npy"):
            arr = np.load(path)
            np.save(path, np.pad(arr, [(0, 0)]*(arr.ndim-2) + [(2, 2), (2, 2)]))
    images.rename(tmp_path / patch_dir_name("data", *TARGET, 2, 11, margin))
    masks.rename(tmp_path / patch_dir_name("mask", *TARGET, 2, 11))
    coord = tmp_path / "coord.json"
    coord.write_text(json.dumps({CURRENT: {"frame": "North"}}))
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"val": [CURRENT], "aoi_window": {"val": box}}))
    checkpoint = tmp_path / "model.pt"
    torch.save({}, checkpoint)
    model = CentreIdentity(TARGET)
    loaded = LoadedModel(model, "unet", 1, context_size=context, predict_size=TARGET)
    if not legacy:
        loaded.data_contract = {"preprocessing_version": "frame-v2", "aoi_selection_version": "coordinates-v2"}
    monkeypatch.setattr("sinkholes.models.factory.build_from_checkpoint", lambda *a, **kw: loaded)
    monkeypatch.setattr("sinkholes.device.get_device", lambda: torch.device("cpu"))
    def inspect_samples(net, loader, device, **kw):
        ds = loader.dataset
        assert len(ds) == (9 if legacy else 1)
        sample = ds[0]
        assert sample["image"].shape[-2:] == context
        assert net(sample["image"][None]).shape[-2:] == TARGET
        assert ds.preprocessing_version == ("legacy-row-v1" if legacy else "frame-v2")
        return torch.tensor(1.)
    monkeypatch.setattr("sinkholes.training.evaluate.evaluate", inspect_samples)
    p = argparse.ArgumentParser(); patch_test.add_arguments(p)
    args = p.parse_args(["--model", str(checkpoint), "--partition_file", str(partition),
                         "--patches_dir", str(tmp_path), "--patch_size", "4", "4",
                         "--intf_dict_path", str(coord), "--metrics_out", str(tmp_path / "metrics.json")])
    patch_test.main(args)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["patches"] == (9 if legacy else 1)
    assert metrics["input_size"] == list(context)


def test_missing_context_pixels_are_rejected_instead_of_silently_used(tree):
    ds = dataset(tree, context_size=(8, 8))
    with pytest.raises(ValueError, match="does not match context"):
        ds[0]


def test_pickled_subset_reads_preprocessing_policy_from_its_dataset(tmp_path, monkeypatch):
    from sinkholes.inference import patch_test
    from sinkholes.models.factory import LoadedModel
    from test_context_scene_hotfix import CentreIdentity

    ds = SubsiDataset.from_arrays(np.ones((2, 4, 4), np.float32), np.ones((2, 4, 4), np.uint8))
    subset = torch.utils.data.Subset(ds, [1])
    pickle_path = tmp_path / "split.pkl"
    pickle_path.write_bytes(pickle.dumps(subset))
    ckpt = tmp_path / "model.pt"; torch.save({}, ckpt)
    loaded = LoadedModel(CentreIdentity(TARGET), "unet", 1, context_size=TARGET, predict_size=TARGET,
                         data_contract={"preprocessing_version": "frame-v2", "aoi_selection_version": "coordinates-v2"})
    monkeypatch.setattr("sinkholes.models.factory.build_from_checkpoint", lambda *a, **kw: loaded)
    monkeypatch.setattr("sinkholes.device.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("sinkholes.training.evaluate.evaluate", lambda *a, **kw: torch.tensor(1.))
    p = argparse.ArgumentParser(); patch_test.add_arguments(p)
    args = p.parse_args(["--model", str(ckpt), "--test_data_path", str(pickle_path), "--patch_size", "4", "4"])
    patch_test.main(args)
