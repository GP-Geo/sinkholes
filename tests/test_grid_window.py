"""The spatial window that three consumers must agree on.

`grid_window` decides which patches belong to a split. Every property here is
one a disagreement would break -- most importantly that the two sides of the
31.4 deg cut cannot share a patch, which is what stops a model being scored on
ground it trained on.
"""
import math

import pytest

from sinkholes.geo import (
    DEFAULT_PATCH,
    FRAME_ORIGINS,
    PIXEL_DEG,
    grid_window,
    window_contains,
)

CUT = 31.4
AOI = (31.25, 31.75, 35.38, 35.46)          # plan section 10
PH, PW = DEFAULT_PATCH
SH, SW = PH // 2, PW // 2


def patch_bounds(frame, row, col):
    """(lat_bottom, lat_top, lon_left, lon_right) of one grid patch."""
    lon0, lat0 = FRAME_ORIGINS[frame]
    return (
        lat0 - (row * SH + PH) * PIXEL_DEG,
        lat0 - row * SH * PIXEL_DEG,
        lon0 + col * SW * PIXEL_DEG,
        lon0 + (col * SW + PW) * PIXEL_DEG,
    )


@pytest.mark.parametrize("frame", ["North", "South"])
def test_every_patch_in_window_is_wholly_inside(frame):
    la, lb, lc, ld = AOI
    r0, r1, c0, c1 = grid_window(frame, *AOI)
    assert r1 > r0 and c1 > c0, "the AOI must select something on both frames"
    for row in (r0, (r0 + r1) // 2, r1 - 1):
        for col in (c0, (c0 + c1) // 2, c1 - 1):
            bot, top, left, right = patch_bounds(frame, row, col)
            assert bot >= la - 1e-12 and top <= lb + 1e-12
            assert left >= lc - 1e-12 and right <= ld + 1e-12


@pytest.mark.parametrize("frame", ["North", "South"])
def test_patches_just_outside_are_excluded(frame):
    """The row before and the row after must each break the box."""
    la, lb, _, _ = AOI
    r0, r1, _, _ = grid_window(frame, *AOI)
    if r0 > 0:
        _, top, _, _ = patch_bounds(frame, r0 - 1, 0)
        assert top > lb + 1e-12, "row before the window should overflow the north edge"
    bot, _, _, _ = patch_bounds(frame, r1, 0)
    assert bot < la - 1e-12, "row at the window end should overflow the south edge"


def test_cut_sides_are_disjoint():
    """The whole point of the cut: no patch may sit on both sides."""
    north_train = grid_window("North", CUT, AOI[1], AOI[2], AOI[3])
    north_cross = grid_window("North", AOI[0], CUT, AOI[2], AOI[3])
    tr = {(r, c) for r in range(*north_train[:2]) for c in range(*north_train[2:])}
    cv = {(r, c) for r in range(*north_cross[:2]) for c in range(*north_cross[2:])}
    assert tr and cv
    assert tr.isdisjoint(cv)


def test_straddling_rows_belong_to_neither_side():
    """Rows crossing 31.4 are dropped by both windows, not claimed by one."""
    above = grid_window("North", CUT, 90.0)
    below = grid_window("North", -90.0, CUT)
    gap = set(range(above[1], below[0]))
    assert gap, "there must be a dropped band between the two sides"
    for row in gap:
        bot, top, _, _ = patch_bounds("North", row, 0)
        assert bot < CUT < top, f"row {row} was dropped but does not straddle the cut"


def test_matches_the_plan_geometry():
    """Section 4's hand-derived table, recomputed."""
    # North: rows fully at or above 31.4 end at 138; 139 and 140 straddle.
    assert grid_window("North", CUT, 90.0)[1] == 139
    assert grid_window("North", -90.0, CUT)[0] == 141
    # South: rows fully at or below 31.4 start at 15; 13 and 14 straddle.
    assert grid_window("South", -90.0, CUT)[0] == 15


def test_window_is_half_open():
    r0, r1, c0, c1 = grid_window("North", *AOI)
    assert window_contains((r0, r1, c0, c1), r0, c0)
    assert window_contains((r0, r1, c0, c1), r1 - 1, c1 - 1)
    assert not window_contains((r0, r1, c0, c1), r1, c0)
    assert not window_contains((r0, r1, c0, c1), r0, c1)


def test_unconstrained_window_starts_at_origin():
    assert grid_window("North")[0] == 0
    assert grid_window("South")[2] == 0


def test_box_that_fits_no_patch_is_empty_not_an_error():
    lon0, lat0 = FRAME_ORIGINS["North"]
    tiny = grid_window("North", lat0 - PH * PIXEL_DEG / 2, lat0, lon0, lon0 + PW * PIXEL_DEG)
    assert tiny[0] >= tiny[1] or tiny[2] >= tiny[3]


def test_inverted_box_raises():
    with pytest.raises(ValueError):
        grid_window("North", 31.75, 31.25, 35.38, 35.46)


def test_unknown_frame_raises():
    with pytest.raises(ValueError):
        grid_window("Middle", *AOI)


def test_window_ignores_raw_scene_north():
    """Regression: the window must come from FRAME_ORIGINS.

    `_load_spatial` used to compute its threshold row from a scene's raw
    `north`, while the grids on disk are aligned to FRAME_ORIGINS -- up to ~700
    rows apart. grid_window takes no scene metadata at all, so the bug cannot
    be reintroduced through it.
    """
    import inspect
    sig = inspect.signature(grid_window)
    for forbidden in ("meta", "coord_dict", "north", "scene"):
        assert forbidden not in sig.parameters
