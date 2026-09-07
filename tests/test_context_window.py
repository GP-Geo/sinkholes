"""Large-context geometry: the centre of a context patch IS the plain patch.

The whole design rests on that one invariant -- it is what lets the mask tree,
nonz_indices, every partition, the AOI window and the reconstruction canvas stay
exactly as they were. These tests pin it on synthetic grids, and (when the patch
tree is reachable) on the real ones.
"""
import numpy as np
import pytest
import torch

from sinkholes.dataprep.context import (
    assert_centre_matches, centre_slices, context_margin, context_size,
    context_window, io_geometry,
)
from sinkholes.dataprep.patchify import patch_dir_name, patch_strides, patchify

PH, PW = 200, 100
MARGIN = (50, 50)


def _scene(rows=900, cols=600, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, cols)).astype(np.float32)


# -- geometry ---------------------------------------------------------------------------

def test_context_size_and_margin_round_trip():
    assert context_size((PH, PW), MARGIN) == (300, 200)
    assert context_margin((PH, PW), (300, 200)) == MARGIN


def test_asymmetric_context_is_rejected():
    # An odd difference cannot be centred; rounding it would offset training
    # from inference by half a pixel, which no shape check would ever catch.
    with pytest.raises(ValueError):
        context_margin((PH, PW), (301, 200))
    with pytest.raises(ValueError):
        context_margin((PH, PW), (100, 100))      # smaller than the target


def test_centre_slices_pick_the_middle():
    rs, cs = centre_slices((300, 200), (PH, PW))
    assert (rs.start, rs.stop, cs.start, cs.stop) == (50, 250, 50, 150)


# -- the invariant, on a patchified grid -------------------------------------------------

def test_every_context_cell_centres_on_its_plain_cell():
    scene = _scene()
    stride = patch_strides((PH, PW), 2)
    plain = patchify(scene, (PH, PW), stride, nx=scene.shape[1], offset=0)
    ctx = patchify(scene, (PH, PW), stride, nx=scene.shape[1], offset=0, margin=MARGIN)

    assert plain.shape[:2] == ctx.shape[:2], "a margin must not change the grid"
    assert ctx.shape[2:] == (300, 200)
    for i in range(plain.shape[0]):
        for j in range(plain.shape[1]):
            assert_centre_matches(ctx[i, j], plain[i, j], (PH, PW), where=f"cell ({i},{j})")


def test_context_cells_carry_the_true_neighbourhood_not_padding():
    """Interior cells must hold real scene pixels in the margin, not zeros."""
    scene = _scene()
    stride = patch_strides((PH, PW), 2)
    ctx = patchify(scene, (PH, PW), stride, nx=scene.shape[1], offset=0, margin=MARGIN)
    i, j = 2, 2
    sy, sx = stride
    expected = scene[i * sy - 50: i * sy + PH + 50, j * sx - 50: j * sx + PW + 50]
    assert np.array_equal(ctx[i, j], expected)


def test_edge_cells_are_padded_with_the_no_data_code():
    scene = _scene()
    stride = patch_strides((PH, PW), 2)
    ctx = patchify(scene, (PH, PW), stride, nx=scene.shape[1], offset=0, margin=MARGIN)
    top_left = ctx[0, 0]
    assert np.all(top_left[:50, :] == 0.0), "rows above the scene must be the no-data code"
    assert np.all(top_left[:, :50] == 0.0)
    # ... and the centre is still exactly the plain patch despite the padding.
    assert np.array_equal(top_left[50:250, 50:150], scene[0:PH, 0:PW])


def test_context_window_matches_patchify():
    """The dataset-side cutter and the disk-side cutter must agree exactly."""
    scene = _scene()
    stride = patch_strides((PH, PW), 2)
    ctx = patchify(scene, (PH, PW), stride, nx=scene.shape[1], offset=0, margin=MARGIN)
    for (i, j) in [(0, 0), (1, 3), (4, 2)]:
        assert np.array_equal(
            context_window(scene, i, j, (PH, PW), stride, MARGIN), ctx[i, j])


def test_assert_centre_matches_actually_fails():
    a = np.zeros((300, 200), np.float32)
    b = np.ones((PH, PW), np.float32)
    with pytest.raises(AssertionError):
        assert_centre_matches(a, b, (PH, PW))


# -- naming and the checkpoint contract ---------------------------------------------------

def test_context_tree_has_its_own_name_and_masks_do_not():
    assert patch_dir_name("data", PH, PW, 2, 11, MARGIN) == \
        "data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned"
    # the plain name is unchanged, so existing trees still resolve
    assert patch_dir_name("data", PH, PW, 2, 11) == "data_patches_H200_W100_strpp2_11days_Aligned"
    assert patch_dir_name("data", PH, PW, 2, 11, (0, 0)) == "data_patches_H200_W100_strpp2_11days_Aligned"
    with pytest.raises(ValueError):
        patch_dir_name("mask", PH, PW, 2, 11, MARGIN)


def test_io_geometry_record():
    assert io_geometry((300, 200), (PH, PW)) == {"context": [300, 200], "predict": [200, 100]}
    assert io_geometry(None, (PH, PW)) == {"context": [200, 100], "predict": [200, 100]}


# -- the model crop ------------------------------------------------------------------------

def test_model_crops_logits_to_predict_size():
    from sinkholes.models.convlstm_unet import ConvLSTMUNet
    net = ConvLSTMUNet(n_channels_per_timestep=1).eval()
    x = torch.randn(1, 3, 300, 200)
    with torch.no_grad():
        assert net(x).shape[-2:] == (300, 200)      # unset -> predict everything
        net.predict_size = (PH, PW)
        assert net(x).shape[-2:] == (PH, PW)


def test_crop_is_centred_and_matches_a_manual_crop():
    from sinkholes.models.unet import UNet
    net = UNet(n_channels=1, n_classes=1).eval()
    x = torch.randn(1, 1, 300, 200)
    with torch.no_grad():
        full = net(x)
        net.predict_size = (PH, PW)
        cropped = net(x)
    assert torch.equal(cropped, full[..., 50:250, 50:150])


def test_checkpoint_round_trips_geometry():
    """A saved large-context model must come back cropping, with no flag."""
    from sinkholes.models.convlstm_unet import ConvLSTMUNet
    from sinkholes.models.factory import build_from_checkpoint

    net = ConvLSTMUNet(n_channels_per_timestep=1)
    state = net.state_dict()
    state["io_geometry"] = io_geometry((300, 200), (PH, PW))
    loaded = build_from_checkpoint(state)
    loaded.model.load_state_dict(state)
    assert tuple(loaded.context_size) == (300, 200)
    assert tuple(loaded.predict_size) == (PH, PW)
    assert loaded.model.predict_size == (PH, PW)
    with torch.no_grad():
        assert loaded.model.eval()(torch.randn(1, 2, 300, 200)).shape[-2:] == (PH, PW)


def test_plain_checkpoint_still_loads_and_does_not_crop():
    from sinkholes.models.convlstm_unet import ConvLSTMUNet
    from sinkholes.models.factory import build_from_checkpoint

    net = ConvLSTMUNet(n_channels_per_timestep=1)
    state = net.state_dict()                      # no io_geometry: pre-context checkpoint
    loaded = build_from_checkpoint(state)
    loaded.model.load_state_dict(state)
    assert loaded.model.predict_size is None
    with torch.no_grad():
        assert loaded.model.eval()(torch.randn(1, 2, PH, PW)).shape[-2:] == (PH, PW)
