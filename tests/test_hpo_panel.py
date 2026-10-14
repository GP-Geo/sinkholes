"""The visual panel and precision at matched recall, on synthetic vote maps."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optuna")

from sinkholes.hpo import controller as C
from sinkholes.hpo import panel as P
from sinkholes.hpo.spec import load_spec

REPO = Path(__file__).resolve().parents[1]
SCENES = ["20240101_20240112", "20240201_20240212", "20240301_20240312"]


def make_maps(rng, strength, clutter_amp, clutter):
    """votes/counts: objects at their own strengths, plus smooth background
    clutter. The two models share objects and clutter pattern; the cleaner one
    has less clutter, so at the same recall it must show higher precision."""
    from scipy.ndimage import gaussian_filter

    field = strength + clutter_amp * clutter
    votes = np.clip(np.round(16 * gaussian_filter(field, 1.0)), 0, 16).astype(np.uint8)
    return votes, np.full(field.shape, 16, np.uint8)


@pytest.fixture
def study(tmp_path):
    raw = json.loads((REPO / "assets/hpo/k10t2026_plain_vs_ctx50.json").read_text())
    raw.update(root=str(tmp_path / "study"), repo=str(REPO))
    (tmp_path / "spec.json").write_text(json.dumps(raw))
    spec = load_spec(tmp_path / "spec.json")
    C.init(spec, spec.path)
    led = C.open_ledger(spec)
    rng = np.random.default_rng(0)
    crop = np.array([0, 300, 0, 300])
    from scipy.ndimage import gaussian_filter

    for intf in SCENES:
        gt = np.zeros((300, 300), np.float32)
        strength = np.zeros((300, 300))
        for _ in range(6):
            y, x = rng.integers(10, 270, 2)
            sl = (slice(y, y + rng.integers(8, 30)), slice(x, x + rng.integers(8, 30)))
            gt[sl] = 1
            strength[sl] = rng.uniform(0.4, 0.95)
        clutter = gaussian_filter(rng.normal(0, 1, gt.shape), 5)
        clutter = np.clip(clutter / clutter.max(), 0, 1) * (gt == 0)
        g = Path(spec.root) / "cache" / "gt"
        g.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(g / f"{intf}_s4.npz", gt=gt, crop=crop, canvas=np.array([300, 300]))
        for key, q in (("plain/s0", 0.45), ("ctx50/s0", 0.85)):
            t = led.trial(key)
            out = Path(spec.root) / "evals" / key / "e030"
            d = out / "maps" / "e030"
            d.mkdir(parents=True, exist_ok=True)
            v, c = make_maps(rng, strength, q, clutter)
            np.savez_compressed(d / f"{intf}.npz", votes=v, counts=c, crop=crop,
                                canvas=np.array([300, 300]))
            t["evals"]["e030"] = {"epochs": [30], "out_dir": str(out), "collected": "x"}
            t["objectives"]["e30"] = {"rth": 0.25, "value": 0.8, "by_rth": {}}
    led.save()
    return spec


def test_panel_is_anonymised_and_model_independent(study, tmp_path):
    out = P.panel(study, ["plain/s0@30", "ctx50/s0@30"], n_windows=4, out=str(tmp_path / "a"))
    assert "panel.png" in out
    key = json.loads((tmp_path / "a" / "panel_key.json").read_text())
    assert sorted(v["trial"] for v in key["key"].values()) == ["ctx50/s0", "plain/s0"]
    assert (tmp_path / "a" / "panel.png").stat().st_size > 10_000
    sheet = (tmp_path / "a" / "rating_sheet.csv").read_text().splitlines()
    assert sheet[0].startswith("window,model,boundaries") and len(sheet) == 1 + 4 * 2
    # the windows depend on the ground truth and the seed, never on the models
    P.panel(study, ["ctx50/s0@30", "plain/s0@30"], n_windows=4, out=str(tmp_path / "b"))
    kb = json.loads((tmp_path / "b" / "panel_key.json").read_text())
    assert kb["windows"] == key["windows"]
    assert any(w["kind"] == "background" for w in key["windows"])


def test_precision_at_matched_recall_separates_cleaner_from_quieter(study, tmp_path):
    P.panel(study, ["plain/s0@30", "ctx50/s0@30"], n_windows=2, out=str(tmp_path / "a"))
    mr = json.loads((tmp_path / "a" / "matched_recall.json").read_text())
    table = mr["precision_at_target"]
    assert 0 < mr["target_recall"] <= 1
    assert table["plain/s0@30"]["precision"] > table["ctx50/s0@30"]["precision"]


def test_precision_interpolation():
    curve = {"0.1": {"mean_recall": 0.9, "mean_precision": 0.5},
             "0.3": {"mean_recall": 0.7, "mean_precision": 0.7},
             "0.5": {"mean_recall": 0.5, "mean_precision": 0.9}}
    # recall >= 0.8 is reached at 0.1 (P 0.5) and, interpolated, at R=0.8 (P 0.6)
    assert P.precision_at_recall(curve, 0.8)["precision"] == pytest.approx(0.6)
    assert P.precision_at_recall(curve, 0.95) is None
    # a flat recall: the cleanest operating point that keeps it is credited
    flat = {"0.1": {"mean_recall": 1.0, "mean_precision": 0.2},
            "0.3": {"mean_recall": 1.0, "mean_precision": 0.6},
            "0.5": {"mean_recall": 0.6, "mean_precision": 0.9}}
    assert P.precision_at_recall(flat, 1.0)["precision"] == pytest.approx(0.6)
