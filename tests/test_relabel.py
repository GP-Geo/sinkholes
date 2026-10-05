"""The relabelling workflow: canvas equivalence, change detection, re-scoring.

The load-bearing claim is that a GT canvas built by ``relabel.common.gt_canvas``
is the canvas eval-scenes would have saved, so that re-scoring a saved
``_pred.npy`` against it changes the labels and nothing else. These tests pin
that against the pipeline's own ``reconstruct_scene`` and
``object_level_evaluate`` rather than against a re-statement of them.
"""

import argparse
import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from sinkholes.inference.reconstruct import canvas_shape, reconstruct_scene
from sinkholes.relabel import diff as d
from sinkholes.relabel.common import grid_from_canvas, gt_canvas
from sinkholes.relabel.evaluate import classify_objects
from sinkholes.training.evaluate import object_level_evaluate


class _NoNet:
    def eval(self):
        return self

    def __call__(self, x):  # never reached: every tile is gated out
        raise AssertionError("the GT canvas must not depend on the network")


@pytest.mark.parametrize("stride", [2, 4])
def test_gt_canvas_matches_reconstruct_scene(stride):
    rng = np.random.default_rng(0)
    ph, pw = 40, 20
    sy, sx = ph // stride, pw // stride
    ny, nx = 7, 9
    H, W = (ny - 1) * sy + ph, (nx - 1) * sx + pw
    mask = (rng.random((H + 13, W)) > 0.8).astype(np.uint8)   # extra rows, like a real scene
    grid = np.stack([np.stack([mask[i * sy:i * sy + ph, j * sx:j * sx + pw] for j in range(nx)])
                     for i in range(ny)]).astype(np.float32)
    stack = [np.zeros((ny, nx, ph, pw), np.float32)]
    out = canvas_shape(ny, nx, (ph, pw), stride)
    gates = np.zeros((1, *out), np.uint8)          # gate everything: no inference at all
    ref = reconstruct_scene(stack, _NoNet(), (ph, pw), stride, 0.5, device="cpu",
                            gt_grid=grid, lidar_gates=gates, accumulate_image=False)
    assert grid_from_canvas(out, (ph, pw), stride) == (ny, nx)
    np.testing.assert_array_equal(gt_canvas(mask, out, (ph, pw), stride), ref.gt)


def test_classify_objects_agrees_with_object_level_evaluate():
    rng = np.random.default_rng(1)
    gt = np.zeros((120, 90), np.float32)
    pred = np.zeros_like(gt)
    for _ in range(12):
        y, x = rng.integers(0, 110), rng.integers(0, 80)
        gt[y:y + rng.integers(2, 9), x:x + rng.integers(2, 9)] = 1
    for _ in range(12):
        y, x = rng.integers(0, 110), rng.integers(0, 80)
        pred[y:y + rng.integers(2, 9), x:x + rng.integers(2, 9)] = 1
    pred[gt > 0] = 1 * (rng.random(int((gt > 0).sum())) > 0.3)
    r, p, _, _ = object_level_evaluate(gt[None], pred[None], None, features=(), th=0.7,
                                       buffer=5, round_ndigits=None)
    _, _, r2, p2 = classify_objects(gt, pred, th=0.7, buffer=5)
    assert r == pytest.approx(r2, abs=1e-12)
    assert p == pytest.approx(p2, abs=1e-12)


# --------------------------------------------------------------------------- change detection


def _orig(polys, intf="20250101_20250112"):
    return gpd.GeoDataFrame({"orig_uid": [f"{intf}_O{i}" for i in range(len(polys))],
                             "intf_id": intf}, geometry=polys, crs="EPSG:4326")


def _work(rows, intf="20250101_20250112"):
    return gpd.GeoDataFrame([{"feat_uid": f, "orig_uid": o, "intf_id": intf, "edit_status": s,
                              "edit_reason": None, "geometry": g} for f, o, s, g in rows],
                            crs="EPSG:4326")


def sq(x, y, s=1.0):
    return box(x, y, x + s, y + s)


def test_detect_changes_cases():
    I = "20250101_20250112"
    O = _orig([sq(0, 0), sq(10, 0), sq(20, 0), sq(30, 0), sq(40, 0), sq(50, 0), sq(52, 0), sq(60, 0, 2)])
    W = _work([
        (f"{I}_O0", f"{I}_O0", "unchanged", sq(0, 0)),                 # unchanged
        (f"{I}_O1", f"{I}_O1", "unchanged", sq(10, 0, 1.2)),           # modified, undeclared
        (f"{I}_O2", f"{I}_O2", "deleted", sq(20, 0)),                  # soft delete
        # O3 removed outright                                          # hard delete
        ("u-new", None, "added", sq(70, 0)),                           # addition
        ("u-redraw", None, "added", sq(40.05, 0)),                     # O4 redrawn, uid lost
        ("u-merge", None, "added", box(50, 0, 53, 1)),                 # one polygon over O5+O6
        (f"{I}_O7", f"{I}_O7", "modified", box(60, 0, 60.9, 2)),       # split piece 1 (largest)
        ("u-split", f"{I}_O7", "modified", box(61.1, 0, 62, 2)),       # split piece 2
    ])
    W.loc[W["feat_uid"] == "u-split", "feat_uid"] = "u-split"
    changes, detected = d.detect_changes(O, W)
    by = {(c["change_type"], c["orig_uid"] or c["feat_uid"]): c for c in changes}

    assert detected[f"{I}_O0"] == "unchanged"
    assert ("MOD", f"{I}_O1") in by and not by[("MOD", f"{I}_O1")]["declared_matches"]
    assert by[("DEL", f"{I}_O2")]["match_method"] == "soft_delete"
    assert ("DEL", f"{I}_O3") in by
    assert ("ADD", "u-new") in by and not by[("ADD", "u-new")]["needs_review"]
    redraw = by[("MOD", f"{I}_O4")]
    assert redraw["match_method"].startswith("geometry_iou") and redraw["feat_uid"] == "u-redraw"
    # the merge is never silently matched: two flagged deletions + one flagged addition
    assert by[("DEL", f"{I}_O5")]["needs_review"] and by[("DEL", f"{I}_O6")]["needs_review"]
    assert by[("ADD", "u-merge")]["needs_review"]
    # split: the larger piece continues the original, the other is an addition
    assert by[("MOD", f"{I}_O7")]["feat_uid"] == f"{I}_O7"
    split_add = [c for c in changes if c["feat_uid"] == "u-split"]
    assert split_add and split_add[0]["change_type"] == "ADD"


def test_change_ids_are_stable(tmp_path):
    I = "20250101_20250112"
    O = _orig([sq(0, 0), sq(10, 0)])
    reg = tmp_path / "registry.csv"
    W1 = _work([(f"{I}_O0", f"{I}_O0", "unchanged", sq(0, 0, 1.5)),
                (f"{I}_O1", f"{I}_O1", "unchanged", sq(10, 0)), ("a", None, "added", sq(30, 0))])
    c1, _ = d.detect_changes(O, W1)
    ids1 = {c["change_id"] for c in d.assign_ids(c1, reg)}
    # later: one more addition, the earlier modification reverted
    W2 = _work([(f"{I}_O0", f"{I}_O0", "unchanged", sq(0, 0)),
                (f"{I}_O1", f"{I}_O1", "unchanged", sq(10, 0)), ("a", None, "added", sq(30, 0)),
                ("b", None, "added", sq(40, 0))])
    c2, _ = d.detect_changes(O, W2)
    ids2 = {c["feat_uid"]: c["change_id"] for c in d.assign_ids(c2, reg)}
    add_a = next(i for i in ids1 if "_ADD_" in i)
    assert ids2["a"] == add_a                                     # unchanged id
    assert ids2["b"] not in ids1 and ids2["b"].endswith("_003")   # numbering continues
    rows = d.load_registry(reg)
    assert any(r["state"] == "withdrawn" and "_MOD_" in r["change_id"] for r in rows)


# --------------------------------------------------------------------------- re-scoring


def _outputs_args(path, **kw):
    from sinkholes.inference import outputs

    p = argparse.ArgumentParser()
    outputs.add_arguments(p)
    argv = ["--path", str(path), "--thresholds", "0.5"]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    return p.parse_args(argv)


def test_gt_dir_rescore(tmp_path):
    from sinkholes.inference import outputs

    intf = "20250329_20250409"                       # any id present in the coordinate dict
    src, alt = tmp_path / "src", tmp_path / "alt"
    src.mkdir(), alt.mkdir()
    gt = np.zeros((60, 50), np.float32)
    gt[10:20, 10:20] = 1
    pred = np.zeros_like(gt)
    pred[10:20, 10:20] = 0.9
    pred[40:45, 30:35] = 0.9                          # a "false positive"
    np.save(src / f"{intf}_pred.npy", pred)
    np.save(src / f"{intf}_gt.npy", gt)
    fixed = gt.copy()
    fixed[40:45, 30:35] = 1                           # relabelling adds the missing object
    np.save(alt / f"{intf}_gt.npy", fixed)

    with pytest.raises(SystemExit):                   # would write into the source dir
        outputs.main(_outputs_args(src, gt_dir=alt))
    outputs.main(_outputs_args(src, out_json=tmp_path / "old.json"))
    outputs.main(_outputs_args(src, gt_dir=alt, out_json=tmp_path / "new.json"))
    old = json.load(open(tmp_path / "old.json"))["per_intf"][intf]["0.5"]
    new = json.load(open(tmp_path / "new.json"))["per_intf"][intf]["0.5"]
    assert old["precision"] < 0.9 and new["precision"] > 0.99
    assert sorted(p.name for p in src.iterdir()) == [f"{intf}_gt.npy", f"{intf}_pred.npy"]

    np.save(alt / f"{intf}_gt.npy", fixed[:-1])       # a canvas of the wrong shape is refused
    with pytest.raises(SystemExit):
        outputs.main(_outputs_args(src, gt_dir=alt, out_json=tmp_path / "bad.json"))


def test_export_writer_keeps_date_fields(tmp_path):
    """The corrected GT must be matchable the way prepare-patches matches it."""
    import sqlite3

    import pandas as pd
    import pyogrio

    from sinkholes.relabel.export import check_like_prepare_patches, write_gt

    pytest.importorskip("fiona")
    g = gpd.GeoDataFrame({"start_date": pd.to_datetime(["2025-03-29"] * 2),
                          "end_date": pd.to_datetime(["2025-04-09"] * 2),
                          "timestamp": pd.to_datetime(["2026-10-05", None]),
                          "id": [0, 0], "decor": [np.nan, 0.0], "reporter": ["a", None],
                          "intf_id": "20250329_20250409"},
                         geometry=[sq(35.4, 31.5, 0.001), sq(35.41, 31.5, 0.001)], crs="EPSG:4326")
    shp, gpkg = tmp_path / "x.shp", tmp_path / "x.gpkg"
    write_gt(g, shp)
    write_gt(g, gpkg, layer="gt")
    info = dict(zip(*[pyogrio.read_info(shp)[k] for k in ("fields", "dtypes")]))
    assert info["start_date"].startswith("datetime64")      # a Date field, not a String
    for path in (shp, gpkg):
        check_like_prepare_patches(path, {"20250329_20250409": 2})
    assert sqlite3.connect(gpkg).execute("pragma user_version").fetchone()[0] == 10200
