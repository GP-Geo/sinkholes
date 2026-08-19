"""Positives-only evaluation of an already-trained model.

Two entry points, both scoring only patches that contain subsidence — the
protocol the benchmark paper uses:

- ``test-patches --partition_file``: the split is built in place from a
  partition JSON, because ``preset_by_intf`` runs pickle no test split.
- ``eval-scenes --positives_only``: the same restriction one level up, as a
  tile gate inside the reconstruction engine.
"""

import argparse
import json

import numpy as np
import pytest
import torch

from sinkholes.dataprep.dataset import SubsiDataset, load_test_dataset, save_test_dataset
from sinkholes.dataprep.partition import load_partition_split
from sinkholes.dataprep.patchify import patch_dir_name, resolve_patch_dirs
from sinkholes.inference.patch_test import add_arguments, build_split_from_partition
from sinkholes.inference.reconstruct import canvas_shape, reconstruct_scene

from test_reconstruction import ConstantNet, grid_of
from test_temporal_data import (
    CHAIN,
    CHAINS,
    CURRENT,
    DISJOINT_POSITIVES,
    PATCH,
    STRIDE,
    write_patch_tree,
)

DEVICE = torch.device("cpu")


# -- the partition JSON ----------------------------------------------------------------

def write_partition(tmp_path, **splits):
    path = tmp_path / "partition.json"
    path.write_text(json.dumps(splits))
    return str(path)


def test_load_partition_split_reads_each_list(tmp_path):
    path = write_partition(tmp_path, train=["a"], val=["b"], test=["c", "d"])
    assert load_partition_split(path, "train") == ["a"]
    assert load_partition_split(path, "val") == ["b"]
    assert load_partition_split(path, "test") == ["c", "d"]


def test_a_missing_split_is_an_error_not_an_empty_list(tmp_path):
    """The _testeval partitions hold a parent's test list under 'val' and have
    no 'test' of their own — scoring the wrong ground silently is the failure
    this guards against."""
    path = write_partition(tmp_path, train=["a"], val=["b"])
    with pytest.raises(SystemExit, match="no 'test' list"):
        load_partition_split(path, "test")


def test_an_unknown_split_name_is_rejected(tmp_path):
    path = write_partition(tmp_path, train=["a"], val=["b"])
    with pytest.raises(SystemExit, match="unknown split"):
        load_partition_split(path, "validation")


# -- building the split in place -------------------------------------------------------

@pytest.fixture
def patch_root(tmp_path):
    """A synthetic patch tree under the directory names the resolver expects."""
    H, W = PATCH
    root = tmp_path / "patches"
    root.mkdir()
    tree = tmp_path / "tree"
    tree.mkdir()
    img_dir, msk_dir = write_patch_tree(tree, DISJOINT_POSITIVES)
    img_dir.rename(root / patch_dir_name("data", H, W, STRIDE, 11))
    msk_dir.rename(root / patch_dir_name("mask", H, W, STRIDE, 11))
    return str(root)


@pytest.fixture
def coord_dict_path(tmp_path):
    """A coordinate dictionary whose 11-day chain is the synthetic CHAIN."""
    path = tmp_path / "intf_coord.json"
    path.write_text(json.dumps({tid: {"frame": "North", "nonz_num": 2} for tid in CHAIN}))
    return str(path)


def test_resolve_patch_dirs_matches_the_naming_scheme(patch_root):
    image_dir, mask_dir = resolve_patch_dirs(patch_root, PATCH, STRIDE)
    assert image_dir.endswith(patch_dir_name("data", *PATCH, STRIDE, 11))
    assert mask_dir.endswith(patch_dir_name("mask", *PATCH, STRIDE, 11))


def test_resolve_patch_dirs_fails_loudly_on_the_wrong_geometry(patch_root):
    with pytest.raises(SystemExit, match="patch directories not found"):
        resolve_patch_dirs(patch_root, PATCH, STRIDE + 1)


def parsed_args(**overrides):
    """A test-patches namespace with the defaults the CLI would apply."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    argv = ["--model", "unused"]
    for key, value in overrides.items():
        if value is True:
            argv.append(f"--{key}")
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            argv += [f"--{key}", *[str(v) for v in value]]
        else:
            argv += [f"--{key}", str(value)]
    return parser.parse_args(argv)


def built_split(tmp_path, patch_root, **overrides):
    partition = write_partition(tmp_path, train=["unused"], val=[CURRENT], test=[CURRENT])
    args = parsed_args(partition_file=partition, patches_dir=patch_root,
                       patch_size=PATCH, stride=STRIDE, **overrides)
    return build_split_from_partition(args)


def test_the_built_split_is_the_positive_patches_of_the_split_list(tmp_path, patch_root):
    ds = built_split(tmp_path, patch_root)
    assert len(ds) == len(DISJOINT_POSITIVES[CURRENT])
    assert ds.ids == [CURRENT]
    assert ds.n_negative == 0, "a positives-only set holds no empty targets"


def test_the_built_split_equals_the_split_training_would_have_pickled(tmp_path, patch_root):
    """The regression the whole change rests on: an in-place build and a
    pickled one must hand ``evaluate`` byte-identical samples."""
    image_dir, mask_dir = resolve_patch_dirs(patch_root, PATCH, STRIDE)
    pickled_source = SubsiDataset(image_dir, mask_dir, [CURRENT], mode="test",
                                  patch_size=PATCH, stride=STRIDE)
    pkl = tmp_path / "test_dataset.pkl"
    save_test_dataset(pickled_source, pkl)
    pickled = load_test_dataset(pkl)

    built = built_split(tmp_path, patch_root)
    assert len(built) == len(pickled)
    for k in range(len(built)):
        assert torch.equal(built[k]["image"], pickled[k]["image"])
        assert torch.equal(built[k]["mask"], pickled[k]["mask"])


def test_the_temporal_build_carries_the_whole_chain(tmp_path, patch_root, coord_dict_path):
    """--add_temporal must reproduce training's stacks, not just its patches."""
    image_dir, mask_dir = resolve_patch_dirs(patch_root, PATCH, STRIDE)
    expected = SubsiDataset(image_dir, mask_dir, [CURRENT], mode="test", temporal=True,
                            seq_dict=CHAINS, patch_size=PATCH, stride=STRIDE)

    built = built_split(tmp_path, patch_root, add_temporal=True,
                        k_prevs=len(CHAIN) - 1, intf_dict_path=coord_dict_path)

    assert len(built) == len(expected)
    for k in range(len(built)):
        assert built[k]["image"].shape[0] == len(CHAIN)
        assert torch.equal(built[k]["image"], expected[k]["image"])
        assert torch.equal(built[k]["mask"], expected[k]["mask"])


def test_an_interferogram_without_a_chain_is_dropped_with_a_warning(
        tmp_path, patch_root, coord_dict_path, caplog):
    partition = write_partition(tmp_path, val=[CURRENT, CHAIN[0]])
    args = parsed_args(partition_file=partition, patches_dir=patch_root, patch_size=PATCH,
                       stride=STRIDE, add_temporal=True, k_prevs=len(CHAIN) - 1,
                       intf_dict_path=coord_dict_path)
    with caplog.at_level("WARNING"):
        built = build_split_from_partition(args)
    assert built.ids == [CURRENT], "only the id with a full chain survives"
    assert "no full" in caplog.text and CHAIN[0] in caplog.text


# -- the scene-level gate --------------------------------------------------------------

def test_positive_tiles_gate_restricts_prediction_to_the_listed_tiles():
    net = ConstantNet()
    stack = [grid_of(0.3)]
    kept = {(1, 1)}
    result = reconstruct_scene(stack, net, PATCH, STRIDE, threshold=0.4,
                               device=DEVICE, positive_tiles=kept)

    step_y, step_x = PATCH[0] // STRIDE, PATCH[1] // STRIDE
    y0, x0 = 1 * step_y, 1 * step_x
    assert result.confidence[y0:y0 + PATCH[0], x0:x0 + PATCH[1]].max() > 0
    predicted = result.confidence > 0
    outside = predicted.copy()
    outside[y0:y0 + PATCH[0], x0:x0 + PATCH[1]] = False
    assert not outside.any(), "no tile outside the positive set may be predicted"


def test_the_gate_leaves_the_image_and_ground_truth_canvases_whole():
    """Gated tiles still accumulate their input and ground truth, so a
    positives-only run's saved arrays stay comparable with a full one."""
    net = ConstantNet()
    stack = [grid_of(0.3)]
    gt_grid = np.ones((3, 3, *PATCH), dtype=np.float32)
    full = reconstruct_scene(stack, net, PATCH, STRIDE, 0.4, device=DEVICE, gt_grid=gt_grid)
    gated = reconstruct_scene(stack, net, PATCH, STRIDE, 0.4, device=DEVICE,
                              gt_grid=gt_grid, positive_tiles={(1, 1)})
    assert np.array_equal(full.image, gated.image)
    assert np.array_equal(full.gt, gated.gt)
    assert gated.confidence.sum() < full.confidence.sum()


def test_an_empty_positive_set_predicts_nothing():
    net = ConstantNet()
    result = reconstruct_scene([grid_of(0.3)], net, PATCH, STRIDE, 0.4,
                               device=DEVICE, positive_tiles=set())
    assert result.confidence.max() == 0.0
    assert result.confidence.shape == canvas_shape(3, 3, PATCH, STRIDE)
