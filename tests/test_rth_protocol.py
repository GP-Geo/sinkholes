"""The benchmark paper's RTh reconstruction protocol, end to end.

Remote Sens. 2026, 18, 211 sec. 3.6.2 defines the Confidence Factor as the
fraction of overlapping patches assigning a positive label to a pixel, and RTh
as a threshold on that fraction: 0.125/0.25/0.5 = at least 2/4/8 of 16 tiles.
These tests cover the eval-outputs half (the reconstruct.py half lives in
test_reconstruction.py).
"""

import argparse
import json
import os

import numpy as np
import pytest

from sinkholes.inference import outputs

# Real ids, so intf_meta resolves against assets/intf_coord.json.
INTF_HIGH = "20190113_20190124"
INTF_LOW = "20190114_20190125"

BLOBS = [(60, 60), (160, 160), (260, 80)]


def write_scene(d, intf, fraction, shape=(400, 300)):
    """A canvas whose predicted region carries one exact vote fraction."""
    conf = np.zeros(shape, dtype=np.float32)
    gt = np.zeros(shape, dtype=np.float32)
    for r, c in BLOBS:
        gt[r:r + 24, c:c + 24] = 1.0
        conf[r - 3:r + 27, c - 3:c + 27] = fraction
    np.save(os.path.join(d, f"{intf}_pred.npy"), conf)
    np.save(os.path.join(d, f"{intf}_gt.npy"), gt)


def run_outputs(d, argv):
    p = argparse.ArgumentParser()
    outputs.add_arguments(p)
    args = p.parse_args(["--path", str(d)] + argv)
    args.intf_dict_path = None
    outputs.main(args)
    js = [f for f in os.listdir(d) if f.endswith(".json")]
    assert len(js) == 1, js
    with open(os.path.join(d, js[0])) as fh:
        return json.load(fh)


@pytest.fixture
def scene_dir(tmp_path):
    # 12/16 survives every RTh in the sweep; 3/16 survives only RTh 0.125.
    write_scene(tmp_path, INTF_HIGH, 0.75)
    write_scene(tmp_path, INTF_LOW, 0.1875)
    return tmp_path


def test_rth_sweeps_the_papers_four_thresholds(scene_dir):
    res = run_outputs(scene_dir, ["--rth"])
    assert res["thresholds"] == [0.125, 0.25, 0.375, 0.5]
    assert res["threshold_kind"] == "vote"


def test_rth_recall_falls_as_the_vote_requirement_rises(scene_dir):
    """The paper's Table 2 behaviour: raising RTh trades recall for precision.

    Here it is exact rather than statistical -- the 3/16 scene drops out
    between RTh 0.125 and 0.25, halving the aggregate.
    """
    res = run_outputs(scene_dir, ["--rth"])
    s = res["summary"]
    assert s["0.125"]["mean_recall"] == pytest.approx(1.0)
    assert s["0.25"]["mean_recall"] == pytest.approx(0.5)
    assert s["0.5"]["mean_recall"] == pytest.approx(0.5)
    recalls = [s[str(t)]["mean_recall"] for t in res["thresholds"]]
    assert recalls == sorted(recalls, reverse=True), "recall must be monotone in RTh"


def test_both_intersection_tolerances_are_reported(scene_dir):
    res = run_outputs(scene_dir, ["--rth"])
    keys = set(res["summary_by_tolerance"])
    assert keys == {"ith0.7_b5", "ith0.5_b10"}, keys
    assert res["ol_th"] == 0.7 and res["buffer"] == 5, "primary stays the default pair"
    for block in res["summary_by_tolerance"].values():
        assert set(block) == {"0.125", "0.25", "0.375", "0.5"}


def test_primary_tolerance_keeps_the_historical_top_level_schema(scene_dir):
    """A reader written against a single-tolerance file still works."""
    res = run_outputs(scene_dir, ["--rth"])
    primary = res["summary_by_tolerance"]["ith0.7_b5"]
    for th, agg in primary.items():
        assert res["summary"][th]["mean_recall"] == pytest.approx(agg["mean_recall"])
    for intf, per_th in res["per_intf"].items():
        for th, entry in per_th.items():
            assert set(entry) >= {"recall", "precision"}


def test_area_weighted_summary_differs_from_the_unweighted_one(tmp_path):
    """One large scene and one small one must not count equally when weighted."""
    # Big GT area, perfect detection; small GT area, missed entirely at RTh>0.125.
    write_scene(tmp_path, INTF_HIGH, 0.75, shape=(400, 300))
    write_scene(tmp_path, INTF_LOW, 0.1875, shape=(400, 300))
    # Shrink the second scene's ground truth so its area weight is far smaller.
    gt = np.load(os.path.join(tmp_path, f"{INTF_LOW}_gt.npy"))
    gt[:] = 0.0
    gt[60:66, 60:66] = 1.0
    np.save(os.path.join(tmp_path, f"{INTF_LOW}_gt.npy"), gt)

    res = run_outputs(tmp_path, ["--rth"])
    at_25 = res["summary"]["0.25"]
    assert at_25["mean_recall"] == pytest.approx(0.5)
    # Weighted: the failing scene carries 36 px against 3*576, so recall stays high.
    assert at_25["weighted_recall"] > 0.9
    assert at_25["gt_area"] > 0


def test_extra_tolerance_parses_and_rejects_junk(scene_dir):
    res = run_outputs(scene_dir, ["--extra_tolerance", "0.5:10"])
    assert set(res["summary_by_tolerance"]) == {"ith0.7_b5", "ith0.5_b10"}
    assert res["threshold_kind"] == "probability", "no --rth means probability cuts"

    p = argparse.ArgumentParser()
    outputs.add_arguments(p)
    args = p.parse_args(["--path", str(scene_dir), "--extra_tolerance", "nonsense"])
    args.intf_dict_path = None
    with pytest.raises(SystemExit, match="ITH:BUFFER"):
        outputs.main(args)


def test_without_rth_the_historical_thresholds_are_kept(scene_dir):
    res = run_outputs(scene_dir, [])
    assert res["thresholds"] == list(outputs.THRESHOLDS)
    assert res["threshold_kind"] == "probability"
    assert set(res["summary_by_tolerance"]) == {"ith0.7_b5"}
