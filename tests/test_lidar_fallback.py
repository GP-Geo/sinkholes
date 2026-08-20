"""The LiDAR gate for interferograms the mapping does not cover.

``assets/lidar_intf_mask.txt`` stops at start date 20240605 and
``assets/lidar_mask_polygs.shp`` holds four surveys (2019-2022), so every
interferogram from ``20240615_20240626`` onward records ``no_mask``. Those
scenes now gate on :data:`LIDAR_FALLBACK_SOURCE` — the last real survey —
rather than reaching ``rasterise_lidar_gates`` as an unrecognised string and
silently widening the gate to the union of all four footprints.
"""

import argparse
import json

import pytest

from sinkholes.meta import (
    LIDAR_FALLBACK_SOURCE,
    NO_LIDAR_MASK,
    intf_meta,
    lidar_source_for,
    resolve_lidar_mask,
)
from sinkholes.paths import asset

#: Covered by the mapping file, and its survey.
MAPPED = ("20190504_20190515", "LiDAR2019")
#: The last id the mapping covers, and the first one it does not.
LAST_MAPPED = "20240605_20240616"
FIRST_UNMAPPED = "20240615_20240626"


@pytest.mark.parametrize("value", [None, "", "none", "None", "null", NO_LIDAR_MASK])
def test_every_spelling_of_absent_resolves_to_the_fallback(value):
    assert resolve_lidar_mask(value) == LIDAR_FALLBACK_SOURCE


def test_a_real_source_is_left_alone():
    assert resolve_lidar_mask("LiDAR2019") == "LiDAR2019"


def test_the_fallback_is_a_source_the_shapefile_actually_has():
    """The whole point of the fallback is a NARROWER gate, not a wider one.

    ``rasterise_lidar_gates`` falls back to ALL polygons for any id it cannot
    find, so a fallback naming a survey that is not in the shapefile would
    quietly restore the union-of-everything behaviour this replaced — and would
    do it while looking like a deliberate choice.
    """
    gpd = pytest.importorskip("geopandas")
    sources = set(
        gpd.read_file(asset("lidar_mask_polygs.shp"))["source"].astype(str).str.strip()
    )
    assert LIDAR_FALLBACK_SOURCE in sources, (
        f"LIDAR_FALLBACK_SOURCE={LIDAR_FALLBACK_SOURCE!r} is not one of {sorted(sources)}; "
        f"scenes using it would be gated by ALL polygons instead"
    )


def test_the_mapping_file_still_reports_what_it_knows():
    """lidar_source_for is the honest reader; resolution happens above it.

    ``prepare-metadata`` warns on ``no_mask``, which is the only signal that the
    mapping has gone stale. Resolving inside this function would delete it.
    """
    assert lidar_source_for(MAPPED[0]) == MAPPED[1]
    assert lidar_source_for(LAST_MAPPED) == LIDAR_FALLBACK_SOURCE
    assert lidar_source_for(FIRST_UNMAPPED) == NO_LIDAR_MASK


def test_intf_meta_hands_consumers_a_survey_never_no_mask():
    assert intf_meta(MAPPED[0]).lidar_mask == MAPPED[1]
    assert intf_meta(FIRST_UNMAPPED).lidar_mask == LIDAR_FALLBACK_SOURCE


def test_no_interferogram_in_the_dictionary_gates_on_no_mask():
    """The dictionary keeps recording no_mask; no consumer ever sees it."""
    with open(asset("intf_coord.json")) as fh:
        coord = json.load(fh)
    assert any(v.get("lidar_mask") == NO_LIDAR_MASK for v in coord.values()), (
        "the fixture assumes the shipped dictionary still has uncovered ids"
    )
    assert all(intf_meta(i).lidar_mask != NO_LIDAR_MASK for i in coord)


# -- --min_positives -------------------------------------------------------------------

def _args(min_positives):
    return argparse.Namespace(min_positives=min_positives, intf_dict_path=None)


def test_min_positives_keeps_strictly_more_than_the_threshold(tmp_path, monkeypatch):
    """'more than 150' is exclusive: a scene with exactly 150 is dropped."""
    from sinkholes.inference import scenes

    coord = {
        "20200101_20200112": {"nonz_num": 151},
        "20200112_20200123": {"nonz_num": 150},
        "20200123_20200203": {"nonz_num": 4},
        "20200203_20200214": {"nonz_num": "none"},
    }
    monkeypatch.setattr(scenes, "load_coord_dict", lambda _path: coord)

    kept = scenes.filter_by_min_positives(list(coord), _args(150))
    assert kept == ["20200101_20200112"]


def test_min_positives_zero_disables_the_filter(monkeypatch):
    from sinkholes.inference import scenes

    ids = ["20200101_20200112", "20200112_20200123"]
    monkeypatch.setattr(scenes, "load_coord_dict",
                        lambda _path: {i: {"nonz_num": 1} for i in ids})
    assert scenes.filter_by_min_positives(ids, _args(0)) == ids


def test_an_id_missing_from_the_dictionary_is_dropped_not_crashed(monkeypatch):
    from sinkholes.inference import scenes

    monkeypatch.setattr(scenes, "load_coord_dict",
                        lambda _path: {"20200101_20200112": {"nonz_num": 900}})
    kept = scenes.filter_by_min_positives(
        ["20200101_20200112", "19990101_19990112"], _args(150))
    assert kept == ["20200101_20200112"]


def test_filtering_everything_away_is_a_clean_exit_not_an_empty_run(monkeypatch):
    """An empty scene list would otherwise fail much later, mid-evaluation."""
    from sinkholes.inference import scenes

    ids = ["20200101_20200112"]
    monkeypatch.setattr(scenes, "load_coord_dict",
                        lambda _path: {i: {"nonz_num": 3} for i in ids})
    with pytest.raises(SystemExit, match="min_positives"):
        scenes.filter_by_min_positives(ids, _args(150))
