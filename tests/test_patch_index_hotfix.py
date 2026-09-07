"""Partial/full regeneration operates only on synthetic temporary scenes."""
import argparse
import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest

from sinkholes.dataprep import prepare_patches as prep
from sinkholes.dataprep.patchify import patch_dir_name
from sinkholes.geo import FRAME_ORIGINS

A, B = "20200101_20200112", "20200112_20200123"


@pytest.fixture
def generation(tmp_path, monkeypatch):
    scenes, root = tmp_path / "raw", tmp_path / "patches"
    scenes.mkdir(); root.mkdir()
    for tid in (A, B):
        start, end = tid.split("_")
        np.ones((4, 4), np.float32).tofile(scenes / f"tgeo_int_{start}T000000_{end}T000000.unw")
    x, y = FRAME_ORIGINS["North"]
    meta = SimpleNamespace(frame="North", nlines=4, ncells=4, byte_order="LSBFirst",
                           east=x, north=y, dx=.0001, dy=.0001)
    monkeypatch.setattr(prep, "intf_meta", lambda *a: meta)
    empty = gpd.GeoDataFrame({"start_date": [], "end_date": []}, geometry=[], crs="EPSG:4326")
    monkeypatch.setattr(gpd, "read_file", lambda *a, **kw: empty)
    p = argparse.ArgumentParser(); prep.add_arguments(p)
    args = p.parse_args(["--input_dir", str(scenes), "--output_dir", str(root),
                         "--gt_polygon_file_path", "synthetic", "--patch_size", "2", "2",
                         "--offset_x", "0", "--nx", "4"])
    folder = root / patch_dir_name("data", 2, 2, 2, 11)
    folder.mkdir()
    index = folder / "nonz_indices.json"
    index.write_text(json.dumps({A: [[0, 0]], B: [[1, 1]]}))
    return args, index


def test_subset_regeneration_preserves_unprocessed_scene(generation):
    args, index = generation
    args.by_list = A
    prep.main(args)
    assert json.loads(index.read_text()) == {A: [], B: [[1, 1]]}


def test_no_matching_subset_leaves_original_index_bytes(generation):
    args, index = generation
    args.by_list = "20300101_20300112"
    before = index.read_bytes()
    prep.main(args)
    assert index.read_bytes() == before


def test_full_replace_is_explicit_and_deterministic(generation):
    args, index = generation
    args.index_mode = "replace"
    index.write_text(json.dumps({"old_scene": [[5, 5]]}))
    prep.main(args)
    assert json.loads(index.read_text()) == {A: [], B: []}
    before = index.read_bytes()
    prep.main(args)
    assert index.read_bytes() == before


def test_replace_rejects_filtered_pass_before_modifying_metadata(generation):
    args, index = generation
    args.index_mode = "replace"; args.by_list = A
    before = index.read_bytes()
    with pytest.raises(ValueError, match="unfiltered"):
        prep.main(args)
    assert index.read_bytes() == before


def test_failed_atomic_publish_preserves_previous_index(tmp_path, monkeypatch):
    index = tmp_path / "nonz_indices.json"
    index.write_text(json.dumps({B: [[1, 1]]}))
    before = index.read_bytes()
    def fail(*a):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(prep.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        prep.write_nonz_index_atomic(index, {A: []})
    assert index.read_bytes() == before
    assert not list(tmp_path.glob(".nonz_indices.json.*"))


def test_malformed_index_fails_before_array_overwrite(generation):
    args, index = generation
    index.write_text("bad json")
    with pytest.raises(json.JSONDecodeError):
        prep.main(args)
    assert not list(index.parent.glob("*.npy"))


def test_finished_scene_index_survives_failure_on_next_scene(generation, monkeypatch):
    args, index = generation
    original = prep.intf_meta
    def fail_on_second(tid, *a):
        if tid == B:
            raise RuntimeError("simulated interruption")
        return original(tid, *a)
    monkeypatch.setattr(prep, "intf_meta", fail_on_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        prep.main(args)
    assert json.loads(index.read_text()) == {A: [], B: [[1, 1]]}
