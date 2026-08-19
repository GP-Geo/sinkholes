"""The shared reconstruction engine: canvas geometry, temporal feed order,
LiDAR AND-gating, averaging modes and blending."""

import numpy as np
import pytest
import torch

from sinkholes.inference.reconstruct import (
    canvas_shape,
    hann_window,
    reconstruct_scene,
)

DEVICE = torch.device("cpu")


class ConstantNet(torch.nn.Module):
    """Logit 0 everywhere -> probability 0.5 everywhere."""

    n_classes = 1

    def forward(self, x):
        return torch.zeros(x.shape[0], 1, *x.shape[-2:])


class RecordingNet(ConstantNet):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return super().forward(x)


def grid_of(value, ny=3, nx=3, h=4, w=4):
    return np.full((ny, nx, h, w), value, dtype=np.float32)


def test_canvas_shape_keeps_the_historical_trailing_band():
    # 192x89 grid of 200x100 patches at stride 2 -> the documented 19400x4550.
    assert canvas_shape(192, 89, (200, 100), 2) == (19400, 4550)
    # Stride 1: patches tile without overlap, no trailing band.
    assert canvas_shape(5, 4, (10, 8), 1) == (50, 32)


def test_hann_window_is_peak_normalised_and_gamma_sharpens():
    win = hann_window(16, 8)
    assert win.shape == (16, 8)
    assert win.max() == pytest.approx(1.0, abs=1e-6)
    sharp = hann_window(16, 8, gamma=2.0)
    assert sharp.sum() < win.sum()


def test_uniform_average_replicates_the_sum_over_stride_squared():
    """Interior pixels covered by stride^2 tiles average to the probability."""
    net = ConstantNet()
    stack = [grid_of(0.3)]
    result = reconstruct_scene(stack, net, (4, 4), 2, threshold=0.4,
                               device=DEVICE, average="uniform")
    out_h, out_w = canvas_shape(3, 3, (4, 4), 2)
    assert result.confidence.shape == (out_h, out_w)
    # Fully covered interior pixel: 4 tiles x 0.5 / 4 = 0.5.
    assert result.confidence[4, 4] == pytest.approx(0.5)
    # A corner pixel is covered by one tile only: 0.5 / 4.
    assert result.confidence[0, 0] == pytest.approx(0.125)
    # The trailing band beyond the last tile stays zero.
    assert result.confidence[-1, -1] == 0.0
    assert result.thresholded[4, 4] == 1.0 and result.thresholded[0, 0] == 0.0


def test_coverage_average_gives_full_probability_at_borders():
    net = ConstantNet()
    result = reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, threshold=0.4,
                               device=DEVICE, average="coverage")
    assert result.confidence[4, 4] == pytest.approx(0.5)
    assert result.confidence[0, 0] == pytest.approx(0.5)  # 1 tile / count 1


def test_image_accumulation_averages_the_input():
    net = ConstantNet()
    result = reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, threshold=0.4,
                               device=DEVICE, average="uniform")
    assert result.image.shape[0] == 1
    assert result.image[0, 4, 4] == pytest.approx(0.3)


def test_gt_grid_is_reconstructed_alongside():
    net = ConstantNet()
    gt = grid_of(1.0)
    result = reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, threshold=0.4,
                               device=DEVICE, gt_grid=gt)
    assert result.gt[4, 4] == pytest.approx(1.0)


def test_lidar_gate_is_an_and_across_timesteps():
    """A tile outside ANY timestep's mask is never predicted — even when the
    current frame's mask covers it."""
    net = ConstantNet()
    stack = [grid_of(0.3), grid_of(0.3)]
    out_shape = canvas_shape(3, 3, (4, 4), 2)

    gates = np.ones((2, *out_shape), dtype=np.uint8)
    result_all = reconstruct_scene(stack, net, (4, 4), 2, 0.4, device=DEVICE,
                                   lidar_gates=gates)
    assert result_all.confidence.max() > 0

    gates_blocked = gates.copy()
    gates_blocked[0] = 0  # the OLDER timestep has no coverage anywhere
    result_blocked = reconstruct_scene(stack, net, (4, 4), 2, 0.4, device=DEVICE,
                                       lidar_gates=gates_blocked)
    assert result_blocked.confidence.max() == 0.0, \
        "one uncovered timestep must gate out every tile"
    # The input image is still accumulated — gating affects prediction only.
    assert result_blocked.image[0, 4, 4] == pytest.approx(0.3)


def test_lidar_gate_requires_the_whole_tile_inside():
    net = ConstantNet()
    stack = [grid_of(0.3, ny=1, nx=1)]
    out_shape = canvas_shape(1, 1, (4, 4), 2)
    gates = np.ones((1, *out_shape), dtype=np.uint8)
    gates[0, 0, 0] = 0  # a single uncovered pixel inside the only tile
    result = reconstruct_scene(stack, net, (4, 4), 2, 0.4, device=DEVICE,
                               lidar_gates=gates)
    assert result.confidence.max() == 0.0


def test_stack_is_fed_chronologically():
    """The network must receive channels [oldest, ..., newest] per tile."""
    net = RecordingNet()
    oldest = grid_of(0.1, ny=1, nx=1)
    middle = grid_of(0.2, ny=1, nx=1)
    newest = grid_of(0.3, ny=1, nx=1)
    reconstruct_scene([oldest, middle, newest], net, (4, 4), 2, 0.4, device=DEVICE)

    assert len(net.seen) == 1
    fed = net.seen[0][0]  # (T, H, W)
    assert fed[0].mean() == pytest.approx(0.1)
    assert fed[1].mean() == pytest.approx(0.2)
    assert fed[2].mean() == pytest.approx(0.3), "the current frame must be the last channel"


def test_validity_channels_are_appended_in_block_layout():
    net = RecordingNet()
    grid = grid_of(0.3, ny=1, nx=1)
    grid[0, 0, 0, 0] = 0.5  # the normalised no-data code
    reconstruct_scene([grid, grid_of(0.4, ny=1, nx=1)], net, (4, 4), 2, 0.4,
                      device=DEVICE, treat_nodata_regions=True)
    fed = net.seen[0][0]
    assert fed.shape[0] == 4  # [img_t0, img_t1, V_t0, V_t1]
    assert fed[2][0, 0] == 0.0, "no-data pixel must be invalid in its own validity map"
    assert fed[2][1, 1] == 1.0
    assert torch.all(fed[3] == 1.0)


def test_hann_blend_weights_overlaps():
    net = ConstantNet()
    result = reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, 0.4,
                               device=DEVICE, blend="hann")
    # A constant probability field stays ~constant under normalised blending
    # wherever any weight was accumulated.
    covered = result.confidence[result.confidence > 0]
    assert np.allclose(covered, 0.5, atol=1e-3)


def test_rejects_unknown_average_mode():
    with pytest.raises(ValueError, match="average"):
        reconstruct_scene([grid_of(0.3)], ConstantNet(), (4, 4), 2, 0.4,
                          device=DEVICE, average="mean")


# ---- the paper's Confidence Factor (average='vote') --------------------------
# Remote Sens. 2026, 18, 211 sec. 3.6.2: "a Confidence Factor as the fraction of
# overlapping patches that assign a positive (1) label to a given pixel, taking
# values in {0, 0.0625, ..., 1}". Thresholding it with RTh gives the binary
# reconstruction; RTh 0.125/0.25/0.5 = at least 2/4/8 of 16 tiles.


class HalfPositiveNet(torch.nn.Module):
    """Positive logit on the top half of each tile, negative on the bottom."""

    n_classes = 1

    def forward(self, x):
        h = x.shape[-2]
        out = torch.full((x.shape[0], 1, *x.shape[-2:]), -10.0)
        out[..., : h // 2, :] = 10.0
        return out


class ProbabilityNet(torch.nn.Module):
    """Emits a fixed probability everywhere, via its logit."""

    n_classes = 1

    def __init__(self, prob):
        super().__init__()
        self.logit = float(np.log(prob / (1 - prob)))

    def forward(self, x):
        return torch.full((x.shape[0], 1, *x.shape[-2:]), self.logit)


def test_vote_is_a_tile_fraction_not_a_mean_probability():
    """The distinguishing property: a confident-but-outvoted tile counts once.

    Every tile here predicts 0.9, well above the 0.5 vote threshold, so the
    Confidence Factor is 1.0 at every predicted pixel -- where 'coverage'
    would report the mean probability 0.9. The two modes must not agree.
    """
    net = ProbabilityNet(0.9)
    vote = reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, threshold=0.25,
                             device=DEVICE, average="vote")
    coverage = reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, threshold=0.25,
                                 device=DEVICE, average="coverage")
    assert vote.confidence[4, 4] == pytest.approx(1.0)
    assert coverage.confidence[4, 4] == pytest.approx(0.9)


def test_vote_below_threshold_scores_zero_however_close():
    """0.49 everywhere is 0 votes, not 0.49 confidence."""
    net = ProbabilityNet(0.49)
    result = reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, threshold=0.125,
                               device=DEVICE, average="vote")
    assert result.confidence.max() == pytest.approx(0.0)
    assert result.thresholded.max() == 0.0


def test_vote_fraction_lands_on_the_papers_quantised_values():
    """At stride 4 a FULLY interior pixel sees 16 tiles, so values are k/16.

    The paper's {0, 0.0625, ..., 1} holds for interior pixels specifically --
    "each interior pixel being included in UP TO sixteen overlapping patches".
    Edge pixels are covered by fewer and are checked in the next test; here the
    grid is chosen so a 6x6 block of pixels reaches the full 16.
    """
    net = HalfPositiveNet()
    grid = grid_of(0.3, ny=6, nx=6, h=8, w=8)
    result = reconstruct_scene([grid], net, (8, 8), 4, threshold=0.25,
                               device=DEVICE, average="vote")

    # Rows/cols 6..11 of the 20x20 canvas are the fully-covered block.
    interior = result.confidence[6:12, 6:12]
    assert interior.size == 36
    sixteenths = np.round(interior * 16) / 16
    assert np.allclose(interior, sixteenths, atol=1e-6), \
        f"interior confidence must be a multiple of 1/16, got {np.unique(interior)}"
    assert 0.0 <= interior.min() and interior.max() <= 1.0


def test_vote_denominator_is_per_pixel_coverage_not_a_fixed_sixteen():
    """An edge pixel is scored on the tiles that reached it, not out of 16.

    A unanimous model must read 1.0 everywhere it predicted -- including
    corners covered by a single tile. Dividing by a fixed 16 would report
    0.0625 there and manufacture a recall cliff at every scene boundary.
    """
    net = ProbabilityNet(0.9)
    result = reconstruct_scene([grid_of(0.3, ny=6, nx=6, h=8, w=8)], net,
                               (8, 8), 4, threshold=0.25, device=DEVICE,
                               average="vote")
    predicted = result.confidence[result.confidence > 0]
    assert np.allclose(predicted, 1.0), \
        "unanimous votes are 1.0 at every coverage depth, corners included"
    assert result.confidence[0, 0] == pytest.approx(1.0)


def test_vote_denominator_is_tiles_that_actually_voted():
    """Gated-out tiles lower the denominator instead of voting zero.

    The paper's 'up to sixteen overlapping patches' -- a pixel at a mask edge
    is scored on the tiles that reached it, so a fully-positive model still
    reads 1.0 there rather than a fraction.
    """
    net = ProbabilityNet(0.9)
    stack = [grid_of(0.3)]
    out_shape = canvas_shape(3, 3, (4, 4), 2)  # 10x10; tiles span rows 0-4, 2-6, 4-8
    gates = np.ones((1, *out_shape), dtype=np.uint8)
    gates[0, :1, :] = 0  # only row 0 -> gates out the i=0 tile row, i=1/i=2 survive

    result = reconstruct_scene(stack, net, (4, 4), 2, 0.25, device=DEVICE,
                               lidar_gates=gates, average="vote")
    predicted = result.confidence[result.confidence > 0]
    assert predicted.size > 0, "some tiles must survive the gate"
    assert np.allclose(predicted, 1.0), \
        "surviving pixels are unanimous, so the fraction is 1.0 regardless of coverage"
    # Rows 0-1 were reachable only by the gated-out tile row: no tile voted.
    assert result.confidence[0, :].max() == 0.0


def test_vote_rth_thresholds_mean_at_least_k_of_n_tiles():
    """RTh 0.5 keeps a pixel only when at least half its tiles voted yes."""
    net = HalfPositiveNet()
    grid = grid_of(0.3, ny=6, nx=6, h=8, w=8)
    low = reconstruct_scene([grid], net, (8, 8), 4, threshold=0.125,
                            device=DEVICE, average="vote")
    high = reconstruct_scene([grid], net, (8, 8), 4, threshold=0.5,
                             device=DEVICE, average="vote")
    # Same confidence map, different cut: RTh is monotone in what it keeps.
    assert np.array_equal(low.confidence, high.confidence)
    assert high.thresholded.sum() < low.thresholded.sum()
    assert np.all(high.thresholded[low.thresholded == 0] == 0)


def test_vote_rejects_hann_blending():
    net = ProbabilityNet(0.9)
    with pytest.raises(ValueError, match="unweighted tile votes"):
        reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, 0.25, device=DEVICE,
                          average="vote", blend="hann")


def test_unknown_average_mode_is_rejected():
    net = ProbabilityNet(0.9)
    with pytest.raises(ValueError, match="uniform"):
        reconstruct_scene([grid_of(0.3)], net, (4, 4), 2, 0.25, device=DEVICE,
                          average="majority")
