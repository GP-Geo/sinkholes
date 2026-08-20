"""Rules the benchmark partitions must satisfy, whatever the archive looks like.

Every check here corresponds to a way a partition can silently leak. The
generator exists to make these mechanical instead of a thing someone verifies
by eye once.
"""
import argparse
import json
from datetime import date, timedelta

import numpy as np
import pytest

from sinkholes.dataprep import benchmark_partitions as bp
from sinkholes.dataprep.patchify import patch_dir_name
from sinkholes.geo import FRAME_ORIGINS, PIXEL_DEG, grid_window

PATCH = (200, 100)
SPP = 2
AOI = (31.25, 31.75, 35.38, 35.46)
CUT = 31.4


def chain_ids(start: date, n: int, frame_offset: int = 0):
    """n consecutive 11-day interferograms starting at `start`."""
    out = []
    for i in range(n):
        s = start + timedelta(days=11 * i + frame_offset)
        e = s + timedelta(days=11)
        out.append(f"{s:%Y%m%d}_{e:%Y%m%d}")
    return out


def build_archive(tmp_path, n_north=30, n_south=30):
    """A synthetic dictionary + nonz_indices with known geometry."""
    north = chain_ids(date(2023, 1, 2), n_north)
    south = chain_ids(date(2023, 1, 3), n_south, frame_offset=0)
    south = [s for s in south]

    coord, nonz = {}, {}
    # grid rows for the two bands, from the real geometry
    n_tr = grid_window("North", CUT, AOI[1], AOI[2], AOI[3], patch_size=PATCH, stride=(100, 50))
    n_cv = grid_window("North", AOI[0], CUT, AOI[2], AOI[3], patch_size=PATCH, stride=(100, 50))
    s_ho = grid_window("South", AOI[0], CUT, AOI[2], AOI[3], patch_size=PATCH, stride=(100, 50))

    for i in north:
        coord[i] = {"frame": "North", "dx": PIXEL_DEG, "dy": PIXEL_DEG,
                    "north": FRAME_ORIGINS["North"][1], "east": FRAME_ORIGINS["North"][0],
                    "nonz_num": 4, "nlines": 19500, "ncells": 4500, "lidar_mask": "LiDAR2021"}
        nonz[i] = [[n_tr[0] + 1, n_tr[2] + 1], [n_tr[0] + 2, n_tr[2] + 2],
                   [n_cv[0] + 1, n_cv[2] + 1]]
    for i in south:
        coord[i] = {"frame": "South", "dx": PIXEL_DEG, "dy": PIXEL_DEG,
                    "north": FRAME_ORIGINS["South"][1], "east": FRAME_ORIGINS["South"][0],
                    "nonz_num": 3, "nlines": 19500, "ncells": 4500, "lidar_mask": "LiDAR2021"}
        nonz[i] = [[s_ho[0] + 1, s_ho[2] + 1], [s_ho[0] + 2, s_ho[2] + 2]]

    dict_path = tmp_path / "coord.json"
    dict_path.write_text(json.dumps(coord))
    tree = tmp_path / "patches" / patch_dir_name("data", PATCH[0], PATCH[1], SPP, 11)
    tree.mkdir(parents=True)
    (tree / "nonz_indices.json").write_text(json.dumps(nonz))
    return dict_path, tmp_path / "patches", coord


def run_generator(tmp_path, **overrides):
    dict_path, patches, coord = build_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    p = argparse.ArgumentParser()
    bp.add_arguments(p)
    args = p.parse_args([
        "--intf_dict", str(dict_path), "--patches_dir", str(patches),
        "--out_dir", str(out), "--temporal_bounds", "20230601", "20230801",
    ])
    for k, v in overrides.items():
        setattr(args, k, v)
    bp.main(args)
    files = {f.name: json.loads(f.read_text()) for f in out.glob("*.json")}
    return files, coord


def test_writes_eight_files(tmp_path):
    files, _ = run_generator(tmp_path)
    assert len(files) == 8
    for axis in ("geo", "temporal"):
        for k in (5, 10):
            assert f"partition_{axis}_k{k}_clean.json" in files
            assert f"partition_{axis}_k{k}_testeval_clean.json" in files


def test_splits_are_disjoint(tmp_path):
    files, _ = run_generator(tmp_path)
    for name, d in files.items():
        if "testeval" in name:
            continue
        lists = {s: set(d[s]) for s in ("train", "val", "test") if s in d}
        for a in lists:
            for b in lists:
                if a < b:
                    assert not (lists[a] & lists[b]), f"{name}: {a} and {b} overlap"


def test_k10_is_a_subset_of_k5_per_split(tmp_path):
    files, _ = run_generator(tmp_path)
    for axis in ("geo", "temporal"):
        k5 = files[f"partition_{axis}_k5_clean.json"]
        k10 = files[f"partition_{axis}_k10_clean.json"]
        for split in ("train", "val", "test"):
            assert set(k10[split]) <= set(k5[split]), f"{axis} {split}: k10 not inside k5"


def test_held_out_chains_never_reach_into_train(tmp_path):
    """The rule that keeps a held-out sample's input out of training."""
    from sinkholes.meta import find_11day_sequences

    files, coord = run_generator(tmp_path)
    b1 = date(2023, 6, 1)
    for k in (5, 10):
        d = files[f"partition_temporal_k{k}_clean.json"]
        chains, _ = find_11day_sequences(coord, k_prev=k)
        for split in ("val", "test"):
            for intf in d[split]:
                prevs = chains[intf]["prevs"]
                earliest = min([bp.start_date(p) for p in prevs] + [bp.start_date(intf)])
                assert earliest >= b1, (
                    f"k{k} {split}: {intf}'s chain starts {earliest}, before {b1}"
                )


def test_geo_train_and_holdout_are_on_different_frames(tmp_path):
    files, coord = run_generator(tmp_path)
    for k in (5, 10):
        d = files[f"partition_geo_k{k}_clean.json"]
        assert all(coord[i]["frame"] == "North" for i in d["train"])
        assert all(coord[i]["frame"] == "South" for i in d["val"] + d["test"])


def test_geo_carries_crossview_and_temporal_does_not(tmp_path):
    files, _ = run_generator(tmp_path)
    for k in (5, 10):
        assert files[f"partition_geo_k{k}_clean.json"]["crossview"]
        assert "crossview" not in files[f"partition_temporal_k{k}_clean.json"]


def test_crossview_is_north_and_overlaps_train_on_purpose(tmp_path):
    files, coord = run_generator(tmp_path)
    d = files["partition_geo_k5_clean.json"]
    assert all(coord[i]["frame"] == "North" for i in d["crossview"])
    assert set(d["crossview"]) & set(d["train"]), (
        "crossview is meant to share scenes with train -- see plan 5b"
    )


def test_every_split_has_a_window(tmp_path):
    files, _ = run_generator(tmp_path)
    for name, d in files.items():
        win = d["aoi_window"]
        for split in d:
            if split in ("aoi_window", "provenance"):
                continue
            assert split in win, f"{name}: no window for {split}"
            assert len(win[split]) == 4


def test_geo_train_window_is_above_the_cut_and_holdout_below(tmp_path):
    files, _ = run_generator(tmp_path)
    d = files["partition_geo_k5_clean.json"]
    assert d["aoi_window"]["train"][0] == CUT
    assert d["aoi_window"]["test"][1] == CUT


def test_positives_are_counted_inside_the_window_not_whole_scene(tmp_path):
    """nonz_num would overstate every split; provenance must not use it."""
    files, coord = run_generator(tmp_path)
    d = files["partition_geo_k5_clean.json"]
    pos = d["provenance"]["positives_in_window"]
    # each synthetic North scene has 2 positives above the cut and 1 below,
    # while nonz_num says 4 -- so a whole-scene count would be 4x the scenes.
    assert pos["train"] == 2 * len(d["train"])
    assert pos["crossview"] == 1 * len(d["crossview"])
    assert pos["train"] != 4 * len(d["train"]), "counted from nonz_num, not the window"


def test_testeval_carries_the_parent_test_list_as_val(tmp_path):
    files, _ = run_generator(tmp_path)
    for axis in ("geo", "temporal"):
        for k in (5, 10):
            parent = files[f"partition_{axis}_k{k}_clean.json"]
            te = files[f"partition_{axis}_k{k}_testeval_clean.json"]
            assert te["val"] == parent["test"]
            assert "test" not in te
            assert set(te["train"]) == set(parent["train"]) | set(parent["val"])
            assert te["aoi_window"]["val"] == parent["aoi_window"]["test"]


def test_provenance_identifies_the_generation(tmp_path):
    files, _ = run_generator(tmp_path)
    for name, d in files.items():
        pr = d["provenance"]
        assert pr["generation"] == bp.GENERATION
        assert pr["aoi"] == list(AOI)
        assert pr["seed"] == 0
        assert pr["axis"] in ("geo", "temporal")
        assert pr["k_prevs"] in (5, 10)


def test_dry_run_writes_nothing(tmp_path):
    dict_path, patches, _ = build_archive(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    p = argparse.ArgumentParser()
    bp.add_arguments(p)
    args = p.parse_args(["--intf_dict", str(dict_path), "--patches_dir", str(patches),
                         "--out_dir", str(out), "--dry_run"])
    bp.main(args)
    assert not list(out.glob("*.json"))


def test_missing_nonz_indices_is_a_clear_error(tmp_path):
    dict_path, patches, _ = build_archive(tmp_path)
    (patches / patch_dir_name("data", PATCH[0], PATCH[1], SPP, 11) / "nonz_indices.json").unlink()
    p = argparse.ArgumentParser()
    bp.add_arguments(p)
    args = p.parse_args(["--intf_dict", str(dict_path), "--patches_dir", str(patches),
                         "--out_dir", str(tmp_path), "--dry_run"])
    with pytest.raises(SystemExit, match="nonz_num is whole-scene"):
        bp.main(args)


# ---------------------------------------------------------------- year filter
#
# The 2019-2022 family (assets/partition_*_pre2023.json) exists because the
# post-2022 interferograms drive scene-level false positives -- see
# docs/PLAN_CLEAN_BENCHMARK.md section 1. It is produced by --years alone, so
# these check the two ways that flag can silently lie: a scene outside the range
# surviving into a split, and a restricted file still claiming to be the
# all-years generation.


def test_year_filter_keeps_only_the_requested_years(tmp_path):
    # A two-year archive: 60 interferograms x 11 days runs into 2024.
    dict_path, patches, _ = build_archive(tmp_path, n_north=60, n_south=60)
    out = tmp_path / "out"
    out.mkdir()
    p = argparse.ArgumentParser()
    bp.add_arguments(p)
    args = p.parse_args([
        "--intf_dict", str(dict_path), "--patches_dir", str(patches),
        "--out_dir", str(out), "--temporal_bounds", "20230601", "20231201",
        "--years", "2023", "2023", "--suffix", "y2023",
    ])
    bp.main(args)
    files = {f.name: json.loads(f.read_text()) for f in out.glob("*.json")}
    assert files, "the year filter emptied the archive"
    assert all("_y2023.json" in n for n in files)

    for name, d in files.items():
        for split, ids in d.items():
            if split in ("aoi_window", "provenance"):
                continue
            assert ids, f"{name}: {split} is empty under the year filter"
            assert all(i[:4] == "2023" for i in ids), f"{name}: {split} leaks another year"


def test_year_filter_restamps_the_generation(tmp_path):
    dict_path, patches, _ = build_archive(tmp_path, n_north=60, n_south=60)
    out = tmp_path / "out"
    out.mkdir()
    p = argparse.ArgumentParser()
    bp.add_arguments(p)
    args = p.parse_args([
        "--intf_dict", str(dict_path), "--patches_dir", str(patches),
        "--out_dir", str(out), "--temporal_bounds", "20230601", "20231201",
        "--years", "2023", "2023", "--suffix", "y2023",
    ])
    bp.main(args)
    for f in out.glob("*.json"):
        pr = json.loads(f.read_text())["provenance"]
        # A restricted family must not be mistakable for the all-years one.
        assert pr["generation"] == "2023_2023_clean" != bp.GENERATION
        assert pr["years_filter"] == [2023, 2023]


def test_unrestricted_run_records_no_year_filter(tmp_path):
    files, _ = run_generator(tmp_path)
    for d in files.values():
        assert d["provenance"]["years_filter"] is None
        assert d["provenance"]["generation"] == bp.GENERATION
