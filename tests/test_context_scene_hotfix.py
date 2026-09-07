"""Context is input-only: exercise the real tiler, reconstruction and scene CLI."""
import argparse
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import torch

from sinkholes.dataprep.context import centre_slices
from sinkholes.dataprep.patchify import patch_dir_name, patch_file_name, patchify
from sinkholes.inference import predict, scenes
from sinkholes.inference.reconstruct import reconstruct_scene
from sinkholes.models.factory import LoadedModel
from sinkholes.models.parts import CentreCropOutput


class CentreIdentity(CentreCropOutput, torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        self.predict_size = target
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape[-2:]))
        return self.crop_output(x[:, :1]).contiguous()


@pytest.mark.parametrize("average,blend", [("uniform", None), ("coverage", None),
                                          ("vote", None), ("uniform", "hann")])
@pytest.mark.parametrize("context", [(200, 100), (300, 200)])
def test_context_reconstructs_same_target_locations_and_overlap(context, average, blend):
    target = (200, 100)
    raw = np.linspace(-2, 2, 400 * 200, dtype=np.float32).reshape(400, 200)
    plain = predict.tile_view(raw, target, 2)
    expanded = predict.tile_view(raw, target, 2, context_size=context)
    centre = centre_slices(context, target)
    np.testing.assert_array_equal(expanded[..., centre[0], centre[1]], plain)
    kwargs = dict(device=torch.device("cpu"), average=average, blend=blend,
                  gt_grid=(plain > 0).astype(np.float32))
    expected = reconstruct_scene([plain], CentreIdentity(target), target, 2, .25, **kwargs)
    model = CentreIdentity(target)
    actual = reconstruct_scene([expanded], model, target, 2, .25, **kwargs)
    for field in ("image", "confidence", "thresholded", "gt"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert set(model.shapes) == {context}


def test_raw_context_tiler_matches_prepared_grid_at_crop_and_scene_edges():
    raw = np.arange(120, dtype=np.float32).reshape(10, 12)
    target, margin = (4, 4), (2, 2)
    prepared = patchify(raw, target, (2, 2), offset=2, nx=8, margin=margin)
    actual = predict.tile_view(raw[:, 2:10], target, 2, context_size=(8, 8))
    np.testing.assert_array_equal(actual, prepared)


def test_context_preserves_target_based_lidar_and_aoi_gates():
    target = (4, 4)
    raw = np.ones((8, 8), np.float32)
    plain = predict.tile_view(raw, target, 2)
    context = predict.tile_view(raw, target, 2, context_size=(8, 8))
    gates = np.ones((1, 10, 10), np.uint8)
    gates[:, :2] = 0
    kwargs = dict(device=torch.device("cpu"), average="coverage", lidar_gates=gates,
                  tile_window=(0, 3, 1, 3))
    a = reconstruct_scene([plain], CentreIdentity(target), target, 2, .25, **kwargs)
    b = reconstruct_scene([context], CentreIdentity(target), target, 2, .25, **kwargs)
    np.testing.assert_array_equal(a.confidence, b.confidence)


def test_target_geometry_mismatch_is_rejected():
    loaded = LoadedModel(CentreIdentity((4, 4)), "unet", 1,
                         context_size=(8, 8), predict_size=(4, 4))
    with pytest.raises(ValueError, match="target grid"):
        loaded.input_size_for((6, 4))


@pytest.mark.parametrize("context", [(8, 4), (12, 8)])
@pytest.mark.parametrize("source,save", [("preset", True), ("intf_list", False),
                                       ("all", True), ("all", False)])
def test_scene_cli_context_and_one_canonical_confidence_write(tmp_path, monkeypatch, context, source, save):
    target, intf = (8, 4), "20200101_20200112"
    margin = tuple((c - p) // 2 for c, p in zip(context, target))
    raw = np.full((16, 8), .7, np.float32)
    patches = predict.tile_view(raw, target, 2, context_size=context).copy()
    masks = np.ones((*patches.shape[:2], *target), np.uint8)
    for kind, array, pad in [("data", patches, margin), ("mask", masks, (0, 0))]:
        folder = tmp_path / patch_dir_name(kind, *target, 2, 11, pad)
        folder.mkdir()
        np.save(folder / patch_file_name(kind, intf, *target, 2), array)
    checkpoint = tmp_path / "model.pt"
    torch.save({}, checkpoint)
    partition = tmp_path / "partition.json"
    partition.write_text('{"val": ["' + intf + '"]}')
    model = CentreIdentity(target)
    monkeypatch.setattr("sinkholes.models.factory.build_from_checkpoint", lambda *a, **kw:
                        LoadedModel(model, "unet", 1, context_size=context, predict_size=target))
    monkeypatch.setattr("sinkholes.device.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(scenes, "load_coord_dict", lambda *a: {})
    monkeypatch.setattr(scenes, "intf_meta", lambda *a: SimpleNamespace(frame="North", dx=.001, dy=.001))
    parser = argparse.ArgumentParser()
    scenes.add_arguments(parser)
    argv = ["--model", str(checkpoint), "--input_patch_dir", str(tmp_path),
            "--patch_size", "8", "4", "--intf_source", source, "--intf_list", intf,
            "--valset_from_partition", str(partition), "--min_positives", "0",
            "--no-add_lidar_mask", "--output_dir", str(tmp_path / "out")]
    if save:
        argv.append("--save_confidence")
    writes = []
    real_save = np.save
    def count_save(path, *a, **kw):
        writes.append(str(path))
        return real_save(path, *a, **kw)
    monkeypatch.setattr(np, "save", count_save)
    args = parser.parse_args(argv)
    scenes.main(args)
    pred_writes = [p for p in writes if p.endswith("_pred") or p.endswith("_pred.npy")]
    assert len(pred_writes) == int(save or source != "all")
    assert set(model.shapes) == {context}
    assert args.context_margin == margin
    assert len(list((tmp_path / "out").rglob("*_pred.npy"))) == len(pred_writes)
    shp = next((tmp_path / "out").rglob("*_predicted_polygs.shp"))
    assert gpd.read_file(shp).crs.to_epsg() == 4326


@pytest.mark.parametrize("context", [(8, 4), (12, 8)])
def test_predict_cli_uses_checkpoint_context(tmp_path, monkeypatch, context):
    target = (8, 4)
    model = CentreIdentity(target)
    ckpt = tmp_path / "model.pt"
    torch.save({}, ckpt)
    monkeypatch.setattr("sinkholes.models.factory.build_from_checkpoint", lambda *a, **kw:
                        LoadedModel(model, "unet", 1, context_size=context, predict_size=target))
    monkeypatch.setattr("sinkholes.device.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(predict, "load_coord_dict", lambda *a: {})
    meta = SimpleNamespace(east=35., north=32., dx=.001, dy=.001)
    monkeypatch.setattr(predict, "intf_meta", lambda *a: meta)
    raw = np.full((16, 10), .7, np.float32)
    monkeypatch.setattr(predict, "read_unw_scene", lambda *a: raw.copy())
    parser = argparse.ArgumentParser()
    predict.add_arguments(parser)
    args = parser.parse_args(["--model", str(ckpt), "--intfs_list", "20200101_20200112",
                              "--patch_size", "8", "4", "--x_pxls_offset", "2",
                              "--no-add_lidar_mask", "--output_polygs_dir", str(tmp_path / "out")])
    predict.main(args)
    assert set(model.shapes) == {context}
    assert len(model.shapes) == 9   # unchanged 3x3 target grid
    geom = gpd.read_file(next((tmp_path / "out").glob("*.shp")))
    assert geom.total_bounds[0] == pytest.approx(35.002)


def test_predict_preserves_history_context_beyond_shared_target_extent(tmp_path, monkeypatch):
    previous, current = "20200101_20200112", "20200112_20200123"
    class Recorder(CentreIdentity):
        def forward(self, x):
            self.last_input = x.detach().clone()
            return super().forward(x)
    model = Recorder((8, 4))
    ckpt = tmp_path / "model.pt"; torch.save({}, ckpt)
    monkeypatch.setattr("sinkholes.models.factory.build_from_checkpoint", lambda *a, **kw:
                        LoadedModel(model, "convlstm_unet", 1, context_size=(12, 8), predict_size=(8, 4)))
    monkeypatch.setattr("sinkholes.device.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(predict, "load_coord_dict", lambda *a: {})
    monkeypatch.setattr(predict, "find_11day_sequences", lambda *a, **kw:
                        ({current: {"prevs": [previous]}}, [current]))
    meta = SimpleNamespace(east=35., north=32., dx=.001, dy=.001)
    monkeypatch.setattr(predict, "intf_meta", lambda *a: meta)
    history = np.full((20, 12), .9, np.float32)
    present = np.full((16, 8), .7, np.float32)
    monkeypatch.setattr(predict, "read_unw_scene", lambda root, tid, m:
                        history.copy() if tid == previous else present.copy())
    p = argparse.ArgumentParser(); predict.add_arguments(p)
    args = p.parse_args(["--model", str(ckpt), "--intfs_list", current, "--k_prevs", "1",
                         "--patch_size", "8", "4", "--x_pxls_offset", "0",
                         "--no-add_lidar_mask", "--output_polygs_dir", str(tmp_path / "out")])
    predict.main(args)
    assert len(model.shapes) == 9  # common target grid remains 3x3
    torch.testing.assert_close(model.last_input[0, 0, -2:, 2:6], torch.full((2, 4), .9))
    torch.testing.assert_close(model.last_input[0, 1, -2:, 2:6], torch.full((2, 4), .5))
