"""ConvLSTM U-Net: shape contract, sequence handling, temporal ordering.

Pure tensor tests. The project's real patch size (200 x 100) is used wherever
the assertion is about spatial correctness, since neither dimension is a power
of two and the decoder has to land back on them exactly.
"""

import pytest
import torch

from sinkholes.models.convlstm_unet import (
    CONFIG_KEY,
    ConvLSTMCell,
    ConvLSTMUNet,
    build_convlstm_unet,
    pop_model_config,
)

PATCH_H, PATCH_W = 200, 100


def make_model(**kwargs):
    model = ConvLSTMUNet(**kwargs)
    model.eval()
    return model


# -- ConvLSTM cell ----------------------------------------------------------------------

@pytest.mark.parametrize("kernel_size", [3, 5])
def test_convlstm_cell_shapes(kernel_size):
    b, c_in, c_hidden, h, w = 2, 16, 8, 12, 6
    cell = ConvLSTMCell(c_in, c_hidden, kernel_size=kernel_size)
    h_next, c_next = cell(torch.randn(b, c_in, h, w))
    assert h_next.shape == (b, c_hidden, h, w)
    assert c_next.shape == (b, c_hidden, h, w)


def test_convlstm_cell_zero_state_matches_input_device_and_dtype():
    cell = ConvLSTMCell(4, 4, kernel_size=3).double()
    x = torch.randn(1, 4, 5, 7, dtype=torch.float64)
    h0, c0 = cell.init_state(x)
    assert h0.dtype == x.dtype and c0.dtype == x.dtype
    assert torch.count_nonzero(h0) == 0 and torch.count_nonzero(c0) == 0


def test_convlstm_cell_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd kernel_size"):
        ConvLSTMCell(4, 4, kernel_size=2)


# -- forward shapes ---------------------------------------------------------------------

def test_temporal_forward_pass_shape():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 1, PATCH_H, PATCH_W))
    assert logits.shape == (2, 1, PATCH_H, PATCH_W)


def test_dataset_style_input_is_three_single_channel_timesteps():
    """(B, T, H, W) must mean T one-channel timesteps, not one T-channel image."""
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    x4 = torch.randn(2, 3, PATCH_H, PATCH_W)
    with torch.no_grad():
        from_4d = model(x4)
        from_5d = model(x4.unsqueeze(2))
    assert from_4d.shape == (2, 1, PATCH_H, PATCH_W)
    assert torch.allclose(from_4d, from_5d, atol=1e-6)


@pytest.mark.parametrize("t", [1, 2, 4])
def test_variable_sequence_length_on_one_model(t):
    """T is not an architectural parameter: one model, several sequence lengths."""
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        logits = model(torch.randn(1, t, PATCH_H, PATCH_W))
    assert logits.shape == (1, 1, PATCH_H, PATCH_W)


def test_backward_reaches_all_three_stages():
    model = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1)  # train mode
    loss = model(torch.randn(2, 3, 64, 32)).mean()
    loss.backward()
    checks = {
        "encoder": model.inc.double_conv[0].weight,
        "convlstm": model.convlstm.conv.weight,
        "decoder-up": model.up4.conv.double_conv[0].weight,
        "decoder-out": model.outc.conv.weight,
    }
    for name, param in checks.items():
        assert param.grad is not None, f"{name}: no gradient"
        assert torch.count_nonzero(param.grad) > 0, f"{name}: gradient is all zeros"


def test_batch_size_one_keeps_every_dimension():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        logits = model(torch.randn(1, 3, PATCH_H, PATCH_W))
    assert logits.shape == (1, 1, PATCH_H, PATCH_W)


@pytest.mark.parametrize("h,w", [(PATCH_H, PATCH_W), (101, 57)])
def test_decoder_returns_exact_input_spatial_size(h, w):
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with torch.no_grad():
        logits = model(torch.randn(1, 2, h, w))
    assert logits.shape[-2:] == (h, w)


# -- validity-channel block layout ------------------------------------------------------

def test_nodata_block_layout_is_regrouped_correctly():
    """[B, 2T, H, W] is BLOCK-laid-out: [img_t0..img_t{T-1}, V_t0..V_t{T-1}].

    Guards the view+permute against a naive reshape(B, T, 2, H, W), which would
    pair each image with the wrong validity map.
    """
    model = make_model(n_channels_per_timestep=2, n_classes=1)
    b, t, h, w = 2, 3, 32, 16
    imgs = torch.randn(b, t, h, w)
    valid = torch.randint(0, 2, (b, t, h, w)).float()

    flat = torch.cat([imgs, valid], dim=1)
    seq = model._to_sequence(flat)

    assert seq.shape == (b, t, 2, h, w)
    for step in range(t):
        assert torch.equal(seq[:, step, 0], imgs[:, step]), f"image at t={step} misplaced"
        assert torch.equal(seq[:, step, 1], valid[:, step]), f"validity at t={step} misplaced"

    with torch.no_grad():
        assert model(flat).shape == (b, 1, h, w)


def test_nodata_forward_at_project_patch_size():
    model = make_model(n_channels_per_timestep=2, n_classes=1)
    with torch.no_grad():
        logits = model(torch.randn(1, 6, PATCH_H, PATCH_W))  # T = 3, 2 channels each
    assert logits.shape == (1, 1, PATCH_H, PATCH_W)


# -- temporal order is oldest -> newest -------------------------------------------------

def test_convlstm_consumes_timesteps_oldest_first():
    """The ConvLSTM must see timestep 0 first and timestep T-1 last."""
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    b, t = 1, 3
    x = torch.randn(b, t, 64, 32)

    seen = []
    real_cell = model.convlstm

    class RecordingCell(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x_t, state=None):
            seen.append(x_t.detach().clone())
            return self.inner(x_t, state)

    model.convlstm = RecordingCell(real_cell)
    with torch.no_grad():
        model(x)
    model.convlstm = real_cell

    assert len(seen) == t
    with torch.no_grad():
        for step in range(t):
            frame = x[:, step : step + 1]
            expected = model.down4(model.down3(model.down2(model.down1(model.inc(frame)))))
            assert torch.allclose(seen[step], expected, atol=1e-5), (
                f"ConvLSTM call {step} did not receive the encoding of frame {step} — "
                f"the sequence is not consumed oldest -> newest"
            )


def test_reversing_the_sequence_changes_the_output():
    """Temporal order is load-bearing, not accidentally symmetric."""
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    x = torch.randn(1, 3, 64, 32)
    with torch.no_grad():
        forward_order = model(x)
        reversed_order = model(torch.flip(x, dims=[1]))
    assert not torch.allclose(forward_order, reversed_order, atol=1e-4)


def test_skips_come_from_the_latest_timestep():
    """Perturbing the newest frame must move the output more than the oldest."""
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    x = torch.zeros(1, 3, 64, 32)
    with torch.no_grad():
        base = model(x)
        x_oldest = x.clone()
        x_oldest[:, 0] = 1.0
        d_oldest = (model(x_oldest) - base).abs().mean()
        x_newest = x.clone()
        x_newest[:, -1] = 1.0
        d_newest = (model(x_newest) - base).abs().mean()
    assert d_newest > d_oldest


# -- strict interface -------------------------------------------------------------------

def test_indivisible_channel_count_raises_with_shapes_named():
    model = make_model(n_channels_per_timestep=2, n_classes=1)
    with pytest.raises(ValueError) as excinfo:
        model(torch.randn(1, 3, PATCH_H, PATCH_W))
    msg = str(excinfo.value)
    assert "(1, 3, 200, 100)" in msg
    assert "not divisible" in msg
    assert "--add_temporal" in msg


def test_three_dimensional_input_raises():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with pytest.raises(ValueError, match="squeeze"):
        model(torch.randn(3, PATCH_H, PATCH_W))


def test_five_dimensional_input_with_wrong_channel_count_raises():
    model = make_model(n_channels_per_timestep=1, n_classes=1)
    with pytest.raises(ValueError, match="channels per timestep"):
        model(torch.randn(1, 3, 2, 64, 32))


@pytest.mark.parametrize("bad", [0, 3, -1])
def test_unsupported_channels_per_timestep_rejected(bad):
    with pytest.raises(ValueError, match="n_channels_per_timestep"):
        ConvLSTMUNet(n_channels_per_timestep=bad)


# -- checkpoint round-trip --------------------------------------------------------------

def test_checkpoint_config_round_trip():
    """A checkpoint carrying CONFIG_KEY rebuilds the exact architecture and loads."""
    original = ConvLSTMUNet(n_channels_per_timestep=1, n_classes=1,
                            convlstm_hidden_channels=32, convlstm_kernel_size=1)
    sd = original.state_dict()
    sd["mask_values"] = [0, 1]
    sd[CONFIG_KEY] = original.config_dict()

    # Deliberately wrong fallbacks: the stored config must win.
    rebuilt = build_convlstm_unet(sd, n_channels_per_timestep=2, n_classes=1,
                                  convlstm_hidden_channels=None)

    assert rebuilt.convlstm_hidden_channels == 32
    assert rebuilt.convlstm_kernel_size == 1
    assert rebuilt.n_channels_per_timestep == 1
    assert CONFIG_KEY not in sd

    sd.pop("mask_values")
    rebuilt.load_state_dict(sd)


def test_pop_model_config_is_a_noop_on_a_plain_checkpoint():
    sd = {"inc.double_conv.0.weight": torch.zeros(1), "mask_values": [0, 1]}
    assert pop_model_config(sd) is None
    assert set(sd) == {"inc.double_conv.0.weight", "mask_values"}


def test_n_channels_attribute_is_per_timestep():
    model = ConvLSTMUNet(n_channels_per_timestep=2, n_classes=1)
    assert model.n_channels == 2
    assert model.bottleneck_channels == 1024
    assert model.convlstm_hidden_channels == 1024  # defaults to the bottleneck width


def test_bilinear_halves_the_bottleneck_and_still_returns_patch_size():
    model = make_model(n_channels_per_timestep=1, n_classes=1, bilinear=True)
    assert model.bottleneck_channels == 512
    assert model.convlstm_hidden_channels == 512
    with torch.no_grad():
        logits = model(torch.randn(1, 2, PATCH_H, PATCH_W))
    assert logits.shape == (1, 1, PATCH_H, PATCH_W)


def test_narrow_hidden_state_is_projected_back_to_the_bottleneck_width():
    model = make_model(n_channels_per_timestep=1, n_classes=1, convlstm_hidden_channels=64)
    assert isinstance(model.convlstm_proj, torch.nn.Conv2d)
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 32))
    assert logits.shape == (1, 1, 64, 32)
