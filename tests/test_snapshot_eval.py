"""eval-snapshots reproduces eval-scenes + eval-outputs exactly, on less data.

The legacy pair is the reference: every number the project quotes came from it.
These tests build a small synthetic world -- three dates on one frame, a
mask with real objects, LiDAR gates that differ per survey, an AOI window that
overhangs the grid -- run BOTH pipelines over it, and require the per-scene
object recall/precision at every RTh, and the confidence map itself, to be
identical. Then the pieces the new engine adds: the min/max cache, missing
history, resumption, the diagnostics.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sinkholes.dataprep.patchify import patch_dir_name, patch_file_name
from sinkholes.inference import outputs, predict, scenes
from sinkholes.inference import snapshot_eval as SE
from sinkholes.models.factory import LoadedModel
from sinkholes.models.parts import CentreCropOutput
from sinkholes.training.evaluate import object_level_evaluate

TARGET = (8, 4)
STRIDE = 4
IDS = ["20200101_20200112", "20200112_20200123", "20200123_20200203", "20200203_20200214"]
SCORED = IDS[2:]          # each has a full 2-chain
K = 2
WINDOW = (1, 40, 2, 11)   # rows overhang the grid on purpose: eval-scenes clips them


class TinyTemporal(CentreCropOutput, torch.nn.Module):
    """Age-weighted mean of the frames, plus the tile's own mean -> logit.

    The per-tile term is what makes overlapping tiles disagree about a pixel
    (a purely per-pixel model would vote unanimously and every RTh would score
    the same). Every reduction is within one sample, so batching cannot move it.
    """

    takes_offsets = True

    def __init__(self):
        super().__init__()
        self.predict_size = TARGET
        self.n_classes = 1

    def forward(self, x, offsets=None, valid=None):
        w = torch.exp(-0.4 * offsets) if offsets is not None else torch.ones(x.shape[1])
        m = (x * w.view(1, -1, 1, 1)).sum(1, keepdim=True) / w.sum()
        tile = m.mean(dim=(2, 3), keepdim=True)
        return self.crop_output(12.0 * (m - 0.52) + 40.0 * (tile - 0.52)).contiguous()


def _gates(sources, origin, shape):
    """A different LiDAR footprint per survey, so the AND across time matters."""
    out = np.ones((len(sources), *shape), np.uint8)
    for k, src in enumerate(sources):
        h = sum(map(ord, str(src))) % 7
        out[k, : 6 + h, :] = 0
        out[k, :, -(3 + h % 3):] = 0
    return out


def build_world(tmp_path, context):
    rng = np.random.default_rng(0)
    H, W = 40, 30
    margin = tuple((c - p) // 2 for c, p in zip(context, TARGET))
    data_dir = tmp_path / patch_dir_name("data", *TARGET, STRIDE, 11, margin)
    mask_dir = tmp_path / patch_dir_name("mask", *TARGET, STRIDE, 11)
    data_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:H, :W]
    for n, intf in enumerate(IDS):
        # Different extents per date (the common grid is the intersection), one
        # date in [0, 1] with exact zeros (the other normalisation branch).
        h, w = H - (n % 2) * 2, W - (n % 3)
        if n == 1:
            raw = rng.uniform(0, 1, (h, w)).astype(np.float32)
            raw[raw < 0.1] = 0.0
        else:
            raw = rng.uniform(-np.pi, np.pi, (h, w)).astype(np.float32)
            blob = ((yy[:h, :w] - 18) ** 2 + (xx[:h, :w] - 14) ** 2) < 30
            raw[blob] = 2.9
            # A faint object (near the vote threshold, so RTh decides it) that
            # strengthens over time, and a strong signal with no ground truth.
            raw[30:34, 20:27] = 0.25 + 0.35 * n + rng.normal(0, 0.6, (4, 7))
            raw[12:17, 22:27] = 3.0
        np.save(data_dir / patch_file_name("data", intf, *TARGET, STRIDE),
                np.ascontiguousarray(predict.tile_view(raw, TARGET, STRIDE, context_size=context)))
        m = np.zeros((h, w), np.uint8)
        m[((yy[:h, :w] - 18) ** 2 + (xx[:h, :w] - 14) ** 2) < 25] = 1
        m[4:9, 3:6] = 1
        m[30:34, 20:27] = 1
        np.save(mask_dir / patch_file_name("mask", intf, *TARGET, STRIDE),
                np.ascontiguousarray(predict.tile_view(m, TARGET, STRIDE)))
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"train": [], "val": SCORED, "test": [],
                                     "aoi_window": {"val": [31.25, 31.75, 35.38, 35.46]}}))
    return partition


@pytest.fixture
def world(monkeypatch):
    coord = {i: {"frame": "North", "nonz_num": 500} for i in IDS}
    meta = lambda intf, *a: SimpleNamespace(frame="North", dx=.001, dy=.001,
                                            lidar_mask=f"LiDAR{intf[3]}{intf[7]}")
    model = TinyTemporal()

    def build(state, **kw):
        return LoadedModel(model, "tattn_unet", 1, context_size=build.context,
                           predict_size=TARGET)

    monkeypatch.setattr("sinkholes.models.factory.build_from_checkpoint", build)
    monkeypatch.setattr("sinkholes.device.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("sinkholes.meta.load_coord_dict", lambda *a: coord)
    monkeypatch.setattr("sinkholes.meta.intf_meta", meta)
    monkeypatch.setattr(scenes, "load_coord_dict", lambda *a: coord)
    monkeypatch.setattr(scenes, "intf_meta", meta)
    monkeypatch.setattr(outputs, "intf_meta", meta)
    fixed = lambda *a, **kw: WINDOW
    monkeypatch.setattr(scenes, "grid_window", fixed)
    monkeypatch.setattr(outputs, "grid_window", fixed)
    monkeypatch.setattr(SE, "grid_window", fixed)
    monkeypatch.setattr(scenes, "rasterise_lidar_gates", _gates)
    monkeypatch.setattr(SE, "rasterise_lidar_gates", _gates)
    return build


def run_legacy(tmp_path, partition):
    ckpt = tmp_path / "model.pt"
    torch.save({}, ckpt)
    p = argparse.ArgumentParser(); scenes.add_arguments(p)
    scenes.main(p.parse_args([
        "--model", str(ckpt), "--input_patch_dir", str(tmp_path),
        "--patch_size", *map(str, TARGET), "--data_stride", str(STRIDE),
        "--intf_source", "preset", "--valset_from_partition", str(partition),
        "--k_prevs", str(K), "--tattn_unet", "--recon_average", "vote",
        "--vote_threshold", "0.5", "--aoi_from_partition", str(partition),
        "--aoi_split", "val", "--min_positives", "150", "--save_confidence",
        "--output_dir", str(tmp_path / "legacy")]))
    eval_dir = next(p.parent for p in (tmp_path / "legacy").rglob("*_pred.npy"))
    p = argparse.ArgumentParser(); outputs.add_arguments(p)
    out_json = tmp_path / "olm.json"
    outputs.main(p.parse_args([
        "--path", str(eval_dir), "--patch_size", *map(str, TARGET),
        "--data_stride", str(STRIDE), "--aoi_from_partition", str(partition),
        "--aoi_split", "val", "--rth", "--out_json", str(out_json)]))
    return json.loads(out_json.read_text()), eval_dir


def run_new(tmp_path, partition, tile_batch, out="new", extra=()):
    ckpt = tmp_path / "model.pt"
    torch.save({}, ckpt)
    p = argparse.ArgumentParser(); SE.add_arguments(p)
    SE.main(p.parse_args([
        "--checkpoint", f"a={ckpt}", "--partition", str(partition),
        "--patches_dir", str(tmp_path), "--k_prevs", str(K), "--data_stride", str(STRIDE),
        "--patch_size", *map(str, TARGET),
        "--tile_batch", str(tile_batch), "--out_dir", str(tmp_path / out),
        "--cache_dir", str(tmp_path / "cache"), "--save_maps", *extra]))
    return json.loads((tmp_path / out / "a.json").read_text())


@pytest.mark.parametrize("context", [TARGET, (12, 8)])
@pytest.mark.parametrize("tile_batch", [1, 5])
def test_identical_to_eval_scenes_plus_eval_outputs(tmp_path, world, context, tile_batch):
    world.context = context
    partition = build_world(tmp_path, context)
    legacy, eval_dir = run_legacy(tmp_path, partition)
    new = run_new(tmp_path, partition, tile_batch)

    assert sorted(new["per_scene"]) == sorted(legacy["per_intf"]) == SCORED
    for intf in SCORED:
        for th in ("0.125", "0.25", "0.375", "0.5"):
            a, b = new["per_scene"][intf][th], legacy["per_intf"][intf][th]
            for key in ("recall", "precision", "gt_area"):
                assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key])), \
                    (intf, th, key, a[key], b[key])
        # the confidence map itself, over the scored crop
        z = np.load(tmp_path / "new" / "maps" / "a" / f"{intf}.npz")
        Y0, Y1, X0, X1 = z["crop"]
        legacy_conf = np.load(eval_dir / f"{intf}_pred.npy")[Y0:Y1, X0:X1]
        np.testing.assert_array_equal(SE.confidence_from_votes(z["votes"], z["counts"]),
                                      legacy_conf)
        legacy_gt = np.load(eval_dir / f"{intf}_gt.npy")[Y0:Y1, X0:X1]
        cached = np.load(tmp_path / "cache" / "gt" / f"{intf}_s{STRIDE}.npz")["gt"]
        np.testing.assert_array_equal(cached, legacy_gt)
    for th in ("0.125", "0.25", "0.375", "0.5"):
        assert new["summary"][th]["mean_recall"] == legacy["summary"][th]["mean_recall"]
        assert new["summary"][th]["mean_precision"] == legacy["summary"][th]["mean_precision"]


def test_the_world_is_not_degenerate(tmp_path, world):
    """Guard the guard: the synthetic scenes must exercise what they claim to."""
    world.context = TARGET
    partition = build_world(tmp_path, TARGET)
    new = run_new(tmp_path, partition, 1)
    rec = json.loads((tmp_path / "new" / "partial" / "a" / f"{SCORED[0]}.json").read_text())
    assert set(rec["branches"]) == {"radians", "unit"} or "unit" in str(
        json.loads((tmp_path / "new" / "partial" / "a" / f"{SCORED[1]}.json").read_text()))
    z = np.load(tmp_path / "new" / "maps" / "a" / f"{SCORED[0]}.npz")
    assert z["votes"].max() > 0 and (z["counts"] == 0).any() and (z["counts"] > 1).any()
    for i in SCORED:
        pairs = {(v["recall"], v["precision"]) for v in new["per_scene"][i].values()}
        assert len(pairs) > 1, f"{i}: every RTh scores the same -- votes are unanimous"
        assert all(0 < r < 1 and 0 < p < 1 for r, p in pairs)
        assert all(v["n_background"] > 0 for v in new["per_scene"][i].values())


def test_missing_history_is_refused_by_both_engines(tmp_path, world):
    world.context = TARGET
    partition = build_world(tmp_path, TARGET)
    data_dir = tmp_path / patch_dir_name("data", *TARGET, STRIDE, 11)
    (data_dir / patch_file_name("data", IDS[0], *TARGET, STRIDE)).unlink()
    with pytest.raises(SE.MissingHistory, match=IDS[0]):
        run_new(tmp_path, partition, 1)
    with pytest.raises(SystemExit, match="Refusing to pad"):
        run_legacy(tmp_path, partition)


def test_a_killed_job_continues_from_its_finished_scenes(tmp_path, world, monkeypatch):
    world.context = TARGET
    partition = build_world(tmp_path, TARGET)
    first = run_new(tmp_path, partition, 4)
    (tmp_path / "new" / "a.json").unlink()
    (tmp_path / "new" / "partial" / "a" / f"{SCORED[1]}.json").unlink()
    calls = []
    real = SE.load_scene
    monkeypatch.setattr(SE, "load_scene", lambda intf, *a, **kw: calls.append(intf) or real(intf, *a, **kw))
    again = run_new(tmp_path, partition, 4)
    assert calls == [SCORED[1]]
    assert again["summary"] == first["summary"]


def test_cell_minmax_matches_nanmin_and_is_cached(tmp_path):
    grid = np.random.default_rng(1).normal(size=(5, 6, 4, 3)).astype(np.float32)
    grid[2, 3] = np.nan                     # an all-NaN cell
    grid[0, 0, 0, 0] = np.nan
    path = tmp_path / "d" / "g.npy"
    path.parent.mkdir()
    np.save(path, grid)
    cmin, cmax = SE.cell_minmax(path, tmp_path / "cache")
    for ny, nx in [(5, 6), (3, 4), (5, 2)]:
        mn, mx = SE.block_range(cmin, cmax, ny, nx)
        assert mn == float(np.nanmin(grid[:ny, :nx])) and mx == float(np.nanmax(grid[:ny, :nx]))
    cached = tmp_path / "cache" / "d" / "g.minmax.npz"
    assert cached.exists()
    np.save(path, grid * 3)                 # the file changes: the cache must not be trusted
    cmin2, _ = SE.cell_minmax(path, tmp_path / "cache")
    assert np.nanmin(cmin2) == pytest.approx(3 * np.nanmin(cmin))


@pytest.mark.parametrize("seed", range(6))
def test_object_scores_equal_object_level_evaluate(seed):
    rng = np.random.default_rng(seed)
    gt = np.zeros((60, 50), np.float32)
    pred = np.zeros((60, 50), np.int64)
    for _ in range(rng.integers(0, 6)):
        y, x = rng.integers(0, 55, 2)
        gt[y:y + rng.integers(2, 9), x:x + rng.integers(2, 9)] = rng.choice([1.0, 0.25, 0.5])
    for _ in range(rng.integers(0, 7)):
        y, x = rng.integers(0, 55, 2)
        pred[y:y + rng.integers(1, 8), x:x + rng.integers(1, 8)] = 1
    for th, buf in [(0.7, 5), (0.5, 10)]:
        r, p, area, _ = object_level_evaluate(gt[None], pred[None], None, features=(),
                                              th=th, buffer=buf, round_ndigits=None)
        mine = SE.object_scores(SE.gt_geometry(gt, buf), pred, th, buf)
        for got, want in [(mine["recall"], r), (mine["precision"], p), (mine["gt_area"], area)]:
            assert got == want or (np.isnan(got) and np.isnan(want))


def test_diagnostics_count_what_they_say():
    gt = np.zeros((80, 80), np.float32)
    gt[10:20, 10:20] = 1                      # found, in two pieces
    gt[50:60, 50:60] = 1                      # missed
    pred = np.zeros((80, 80), np.int64)
    pred[10:20, 10:14] = 1
    pred[10:20, 16:20] = 1
    pred[60:75, 5:15] = 1                     # far from any GT: background
    s = SE.object_scores(SE.gt_geometry(gt, 5), pred, 0.7, 5)
    assert s["n_gt"] == 2 and s["n_detected"] == 1
    assert s["n_pred"] == 3 and s["n_background"] == 1 and s["n_fp"] == 1
    assert s["background_area"] == 150 and s["fragments_per_detected"] == 2


def test_aggregate_quotes_the_projects_f1():
    per = {"a": {"0.25": {"recall": 0.8, "precision": 0.6, "gt_area": 10.0}},
           "b": {"0.25": {"recall": 0.6, "precision": 1.0, "gt_area": 30.0}},
           "c": {"0.25": {"recall": float("nan"), "precision": float("nan"), "gt_area": 0.0}}}
    s = SE.aggregate(per, [0.25])["0.25"]
    assert s["n_scenes"] == 2 and s["n_undefined"] == 1
    assert s["mean_recall"] == pytest.approx(0.7) and s["mean_precision"] == pytest.approx(0.8)
    assert s["f1"] == pytest.approx(2 * 0.7 * 0.8 / 1.5)
    assert s["weighted_recall"] == pytest.approx((8 + 18) / 40)
