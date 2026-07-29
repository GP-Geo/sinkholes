# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import argparse
import os
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from affine import Affine
from shapely.geometry import Polygon, shape
from rasterio.features import rasterize, shapes

from unet import *
# EDIT 2026-07-29: this module uses `plt` in 7 places but never imported it — it relied on
# `from unet import *` re-exporting matplotlib, which entry #0 removed from unet.py as
# "unused". That made this script fail at import with NameError, so stage (4) was broken
# for every architecture. Importing it explicitly here. CHANGELOG.md #12
import matplotlib.pyplot as plt
# EDIT 2026-07-29: shared checkpoint -> model factory. CHANGELOG.md #12
from factory import architecture_from_flags, build_from_checkpoint


def crop_to_start_xy(
    intf: np.ndarray,
    mask: np.ndarray | None,
    x0: float,
    y0: float,
    x_star: float,
    y_star: float,
    dx: float = 2.777e-05,
    dy: float = 2.777e-05,
    tol: float = 1e-2,
):
    if mask is not None and intf.shape != mask.shape:
        raise ValueError(f"intf and mask must have same shape, got {intf.shape} vs {mask.shape}")

    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    H, W = intf.shape

    col_f = (x_star - x0) / dx
    row_f = (y0 - y_star) / dy

    col_off = int(round(col_f))
    row_off = int(round(row_f))

    if abs(col_f - col_off) * dx > tol or abs(row_f - row_off) * dy > tol:
        raise ValueError(
            f"(x*,y*) not on pixel grid within tolerance: "
            f"col_off ~ {col_f}, row_off ~ {row_f}"
        )

    if not (0 <= col_off < W) or not (0 <= row_off < H):
        raise ValueError(
            f"Requested start is outside the image: "
            f"row_off={row_off}, col_off={col_off}, shape={intf.shape}"
        )

    width = W - col_off
    height = H - row_off

    intf_c = intf[row_off:row_off + height, col_off:col_off + width]

    if mask is not None:
        mask_c = mask[row_off:row_off + height, col_off:col_off + width]
    else:
        mask_c = None

    new_x0 = x0 + col_off * dx
    new_y0 = y0 - row_off * dy

    return intf_c, mask_c, new_x0, new_y0, (row_off, col_off)


def find_11day_sequences(
    meta: Dict[str, Dict[str, Any]],
    k_prev: int = 2,
    step_days: int = 11,
    restrict_to: Optional[List[str]] = None,
    require_current_nonz_gt0: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:

    def parse_key(k: str):
        s, e = k.split("_")
        sd = datetime.strptime(s, "%Y%m%d").date()
        ed = datetime.strptime(e, "%Y%m%d").date()
        return sd, ed, (ed - sd).days

    def make_key(sd, ed) -> str:
        return sd.strftime("%Y%m%d") + "_" + ed.strftime("%Y%m%d")

    daydiff: Dict[str, int] = {}
    for k in meta.keys():
        sd, ed, dd = parse_key(k)
        daydiff[k] = dd

    all_keys = set(meta.keys())
    curr_keys = list(all_keys if restrict_to is None else (all_keys & set(restrict_to)))

    frame_groups = {"North": set(), "South": set()}
    for k, info in meta.items():
        frm = info.get("frame")
        if frm in frame_groups:
            frame_groups[frm].add(k)

    chains_map: Dict[str, Dict[str, Any]] = {}
    valid_intfs: List[str] = []

    for cur_key in curr_keys:
        info = meta.get(cur_key)
        if not info:
            continue

        frm = info.get("frame")
        if frm not in frame_groups:
            continue

        sd, ed, dd = parse_key(cur_key)
        if dd != step_days:
            continue

        if require_current_nonz_gt0:
            nonz_raw = info.get("nonz_num", 0)
            try:
                nonz = int(nonz_raw)
            except Exception:
                nonz = 0
            if nonz <= 0:
                continue

        group = frame_groups[frm]
        prevs: List[str] = []
        ok = True

        for i in range(k_prev, 0, -1):
            psd = sd - timedelta(days=i * step_days)
            ped = ed - timedelta(days=i * step_days)
            pk = make_key(psd, ped)
            if (pk not in group) or (daydiff.get(pk) != step_days):
                ok = False
                break
            prevs.append(pk)

        if not ok:
            continue

        chains_map[cur_key] = {"prevs": prevs, "frame": frm}
        valid_intfs.append(cur_key)

    return chains_map, valid_intfs


def get_intf_coords(intf_name):
    with open(_bootstrap.asset("intf_coord.json"), "r") as intf_dict_file:
        intf_coords = json.load(intf_dict_file)

    x0 = intf_coords[intf_name]["east"]
    y0 = intf_coords[intf_name]["north"]
    dx = intf_coords[intf_name]["dx"]
    dy = intf_coords[intf_name]["dy"]
    nlines = intf_coords[intf_name]["nlines"]
    ncells = intf_coords[intf_name]["ncells"]
    lidar_mask = intf_coords[intf_name]["lidar_mask"]
    num_nonz_p = intf_coords[intf_name]["nonz_num"]
    bo = intf_coords[intf_name]["byte_order"]
    frame = intf_coords[intf_name]["frame"]

    return (x0, y0, dx, dy, ncells, nlines, lidar_mask, num_nonz_p, bo, frame)


def mask_array_to_polygons(mask_array):
    transform = Affine.identity()
    polygons = shapes(mask_array, transform=transform)
    shapely_polygons = [shape(geom) for geom, value in polygons if value == 1]
    polygons_gpd = gpd.GeoDataFrame(geometry=shapely_polygons, crs="EPSG:4326")
    return polygons_gpd


def plg_indx2longlat(polyg_gdf, intf_coords, x_start):
    x0, y0, dx, dy = intf_coords[0], intf_coords[1], intf_coords[2], intf_coords[3]
    polyg_list = polyg_gdf["geometry"].tolist()
    polyg_longlat = []

    for polyg in polyg_list:
        new_coords = [(x_start + x * dx, y0 - y * dy) for x, y in polyg.exterior.coords]
        polyg_longlat.append(Polygon(new_coords))

    polyg_longlat_gdf = gpd.GeoDataFrame(geometry=polyg_longlat, crs="EPSG:4326")
    return polyg_longlat_gdf


def add_dashes(date_str):
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def build_unified_gt_polygons(gdf, curr_and_prevs_list):
    gt_polygs_list = []

    for intf_name in curr_and_prevs_list:
        start_date = intf_name[:8]
        end_date = intf_name[9:]

        curr_gt_polygs = gdf[
            (gdf["start_date"] == add_dashes(start_date)) &
            (gdf["end_date"] == add_dashes(end_date))
        ]

        if not curr_gt_polygs.empty:
            gt_polygs_list.append(curr_gt_polygs)

    if not gt_polygs_list:
        print(f"No ground truth for interferograms: {curr_and_prevs_list}. Skipping...")
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs), None

    all_gt_polygs = gpd.GeoDataFrame(
        pd.concat(gt_polygs_list, ignore_index=True),
        crs=gdf.crs
    )
    unified_geometry = all_gt_polygs.unary_union
    gt_polygs = gpd.GeoDataFrame(geometry=[unified_geometry], crs=gdf.crs)

    return gt_polygs, unified_geometry


def predict_from_full_images(
    all_data,
    intf_coords,
    net,
    patch_size,
    strdpp,
    rth,
    add_lidar_mask=True,
    plot=False,
    lidar_sources=None,
    overlay_on_preds=True,
    device=None,
    x_offset=3000,
    treat_nodata_regions=False,
    blend_type=None,
    window_gamma=1.0,
    pred_blend_dtype="float32",
):
    x0, y0, dx, dy = intf_coords[:4]
    current_lidar_mask = intf_coords[6]
    patch_H, patch_W = patch_size
    C = len(all_data)

    # --- crop x-offset and match shapes ---
    cropped = [d[:, x_offset:] for d in all_data]
    H = min(d.shape[0] for d in cropped)
    W = min(d.shape[1] for d in cropped)
    cropped = [d[:H, :W].astype(np.float32, copy=False) for d in cropped]

    # normalize once
    cropped = [(d + np.pi) / (2 * np.pi) for d in cropped]

    step_y = patch_H // strdpp
    step_x = patch_W // strdpp

    ny = (H - patch_H) // step_y + 1
    nx = (W - patch_W) // step_x + 1

    out_h = ny * step_y + patch_H - step_y
    out_w = nx * step_x + patch_W - step_x

    use_hann = (blend_type == "hann")

    if use_hann:
        def get_hann_window(h, w, gamma=1.0):
            wy = np.hanning(h) if h > 1 else np.ones(1, dtype=np.float32)
            wx = np.hanning(w) if w > 1 else np.ones(1, dtype=np.float32)
            Wn = (wy[:, None] * wx[None, :]).astype(np.float32)
            if gamma != 1.0:
                Wn = np.power(Wn, gamma, dtype=np.float32)
            return Wn / (Wn.max() + 1e-8)

        dtype_np = np.float16 if pred_blend_dtype == "float16" else np.float32
        Wn = get_hann_window(patch_H, patch_W, gamma=window_gamma).astype(dtype_np, copy=False)
        pred_num = np.zeros((out_h, out_w), dtype=dtype_np)
        pred_den = np.zeros((out_h, out_w), dtype=dtype_np)
    else:
        reconstructed_pred = np.zeros((out_h, out_w), dtype=np.float32)
        pred_counts = np.zeros((out_h, out_w), dtype=np.float32)

    # -------- LiDAR mask(s): AND across time steps --------
    rasterized_polygon_current = None
    mask_polyg_current = None
    mask_all = None

    if add_lidar_mask:
        lidar_gdf = gpd.read_file(_bootstrap.asset("lidar_mask_polygs.shp"))

        if lidar_sources is None:
            srcs = [current_lidar_mask] * C
        else:
            srcs = list(lidar_sources)
            if len(srcs) != C:
                print(f"[warn] lidar_sources len {len(srcs)} != C {C}; using current for all")
                srcs = [current_lidar_mask] * C

        mask_all = np.zeros((C, out_h, out_w), dtype=np.uint8)
        tr = rasterio.transform.from_origin(x0 + x_offset * dx, y0, dx, dy)

        for c, src in enumerate(srcs):
            if src is None or str(src).strip().lower() in ("", "none", "null"):
                polyg = lidar_gdf
            else:
                col = lidar_gdf["source"].astype(str).str.strip().str.lower()
                polyg = lidar_gdf[col == str(src).strip().lower()]
                if polyg.empty:
                    print(f"[warn] LiDAR source '{src}' not in shapefile -> using ALL polygons")
                    polyg = lidar_gdf

            mask_all[c] = rasterize(
                [(g, 1) for g in polyg["geometry"].tolist()],
                out_shape=(out_h, out_w),
                transform=tr,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )

            if c == 0:
                rasterized_polygon_current = mask_all[c].copy()
                mask_polyg_current = polyg

    # -------- main prediction loop --------
    for i in range(ny):
        print(i)
        for j in range(nx):
            y0o = i * step_y
            y1o = y0o + patch_H
            x0o = j * step_x
            x1o = x0o + patch_W

            is_within_mask = True
            if add_lidar_mask and mask_all is not None:
                is_within_mask = mask_all[:, y0o:y1o, x0o:x1o].all()

            if not is_within_mask:
                continue

            x_np = np.stack(
                [d[y0o:y1o, x0o:x1o] for d in cropped],
                axis=0
            )  # (C, H, W)

            if treat_nodata_regions:
                tol = 1e-9
                if np.isnan(x_np).any():
                    x_np = np.nan_to_num(x_np, nan=0.0)
                v_np = (np.abs(x_np - 0.5) > tol).astype(np.float32)
                x_np = np.concatenate([x_np, v_np], axis=0)

            image = torch.from_numpy(x_np[None]).to(
                device=device,
                memory_format=torch.channels_last
            )

            with torch.no_grad():
                logits = net(image)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

            if use_hann:
                prob_cast = prob.astype(pred_num.dtype, copy=False)
                pred_num[y0o:y1o, x0o:x1o] += prob_cast * Wn
                pred_den[y0o:y1o, x0o:x1o] += Wn
            else:
                reconstructed_pred[y0o:y1o, x0o:x1o] += prob
                pred_counts[y0o:y1o, x0o:x1o] += 1.0

    # -------- finalize prediction --------
    if use_hann:
        pred_den[pred_den == 0] = 1.0
        reconstructed_pred = (pred_num / pred_den).astype(np.float32)
    else:
        pred_counts[pred_counts == 0] = 1.0
        reconstructed_pred = reconstructed_pred / pred_counts

    reconstructed_pred_th = (reconstructed_pred > rth).astype(np.float32)

    return reconstructed_pred_th, reconstructed_pred

def get_args():
    parser = argparse.ArgumentParser(description="Predict subsidence polygons for new interferogram")
    parser.add_argument("--intfs_dir", type=str, default="./", help="interferogram path")
    parser.add_argument("--intfs_list", type=str, default="a list of intfs in format YYYYMMDD_YYYYMMDD seperated by comma")
    parser.add_argument("--model_dir", type=str, default="./models/")
    parser.add_argument("--model_file", type=str, help="the .pth model file")
    parser.add_argument("--patch_size", nargs="+", type=int, default=(200, 100), help="patch H, patch W. These have to match the patch size in training")
    parser.add_argument("--strdpp", type=int, default=2, help="strides per patch in both directions")
    parser.add_argument("--plot_polygs", action="store_true", help="")
    parser.add_argument("--output_polygs_dir", type=str, default="./out_polygs")
    parser.add_argument("--rth", type=float, default=0.25)
    parser.add_argument("--add_gt_polygs", action="store_true", default=False)
    parser.add_argument("--gt_polygons_file_path", type=str, default="sub_20231001.shp")
    parser.add_argument("--unified_mask", action="store_true")
    parser.add_argument("--k_prevs", type=int, default=0)
    # EDIT 2026-07-29: architecture is detected from the checkpoint; these are optional
    # overrides. CHANGELOG.md #12
    parser.add_argument("--attn_unet", action="store_true",
                        help="Checkpoint is an AttentionUNet (usually auto-detected)")
    parser.add_argument("--add_attn", action="store_true",
                        help="Checkpoint is a UNet with bottleneck attention (usually auto-detected)")
    parser.add_argument("--convlstm_unet", action="store_true",
                        help="Checkpoint is a ConvLSTM U-Net (usually auto-detected)")
    parser.add_argument("--treat_nodata_regions", action="store_true",
                        help="Checkpoint was trained with validity channels (2 per timestep)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--x_pxls_offset", type=int, default=3000)
    parser.add_argument("--blend_type", type=str, default=None, choices=["hann"])
    return parser.parse_args()


if __name__ == "__main__":
    plt.rcParams["backend"] = "Qt5Agg"

    args = get_args()
    x_pxls_offset = args.x_pxls_offset
    intfs_list = args.intfs_list.split(",")

    if args.k_prevs > 0:
        with open(_bootstrap.asset("intf_coord.json"), "r") as f:
            intf_info = json.load(f)
        prev_dict, updated = find_11day_sequences(
            intf_info,
            k_prev=args.k_prevs,
            restrict_to=intfs_list
        )

    num_c = args.k_prevs + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model_dir + args.model_file, map_location=device)

    # EDIT 2026-07-29: was a hardcoded UNet, so ConvLSTM checkpoints could not be
    # deployed to new scenes at all. The factory detects the architecture from the
    # weights; --attn_unet / --add_attn / --convlstm_unet force it. CHANGELOG.md #12
    loaded = build_from_checkpoint(
        state_dict,
        arch=architecture_from_flags(
            attn_unet=getattr(args, "attn_unet", False),
            add_attn=getattr(args, "add_attn", False),
            convlstm_unet=getattr(args, "convlstm_unet", False),
        ),
        n_classes=1,
        bilinear=False,
        treat_nodata_regions=getattr(args, "treat_nodata_regions", False),
    )
    net = loaded.model
    mask_values = loaded.mask_values
    net.to(device=device)
    net.load_state_dict(state_dict)
    net.eval()
    logging.info(
        f"Model loaded! architecture={loaded.architecture} "
        f"in-channels={loaded.n_channels}"
    )
    if loaded.architecture != "convlstm_unet" and loaded.n_channels not in (None, num_c):
        logging.warning(
            f"checkpoint expects {loaded.n_channels} input channels but --k_prevs "
            f"{args.k_prevs} produces {num_c}; the forward pass will fail. "
            f"Set --k_prevs {loaded.n_channels - 1}."
        )

    for i, intf in enumerate(intfs_list):
        intf_coords = get_intf_coords(intf)
        x0, y0, dx, dy, ncells, nlines, intf_lidar_mask, num_nz, bo, frame = intf_coords

        filename = next(
            f for f in os.listdir(args.intfs_dir)
            if f.endswith(".unw") and intf[:8] in f and intf[9:] in f
        )

        path = os.path.join(args.intfs_dir, filename)
        data = np.fromfile(path, dtype=np.float32).reshape(nlines, ncells)

        if bo == "MSBFirst":
            data = data.byteswap().newbyteorder("<")

        prev_coords, prev_data, prev_lidar = [], [], []

        if args.k_prevs > 0:
            for item in prev_dict[intf]["prevs"]:
                coords = get_intf_coords(item)
                bo_item, ncells_item, nlines_item = coords[8], coords[4], coords[5]

                prev_lidar.append(coords[6])
                prev_coords.append(coords)

                fname = next(
                    f for f in os.listdir(args.intfs_dir)
                    if f.endswith(".unw") and item[:8] in f and item[9:] in f
                )

                p = os.path.join(args.intfs_dir, fname)
                prd = np.fromfile(p, dtype=np.float32).reshape(nlines_item, ncells_item)

                if bo_item == "MSBFirst":
                    prd = prd.byteswap().newbyteorder("<")

                prev_data.append(prd)

        all_data = prev_data.copy()
        all_coords = prev_coords.copy()
        all_lidar = prev_lidar.copy()

        all_data.append(data)
        all_coords.append(intf_coords)
        all_lidar.append(intf_lidar_mask)

        common_x0 = max(c[0] for c in all_coords)
        common_y0 = min(c[1] for c in all_coords)
        intf_coords = (common_x0, common_y0) + intf_coords[2:]
        x_start = common_x0 + args.x_pxls_offset * dx

        d_cropped = []
        for k, d in enumerate(all_data):
            d_c, _, _, _, _ = crop_to_start_xy(
                d,
                None,
                all_coords[k][0],
                all_coords[k][1],
                common_x0,
                common_y0
            )
            d_cropped.append(d_c)

        reconstructed_pred_th, reconstructed_pred = predict_from_full_images(
            d_cropped,
            intf_coords,
            net,
            args.patch_size,
            args.strdpp,
            args.rth,
            add_lidar_mask=True,
            plot=args.plot,
            lidar_sources=all_lidar,
            device=device,
            x_offset=x_pxls_offset,
            blend_type=args.blend_type
        )

        polygons = mask_array_to_polygons(reconstructed_pred_th)
        polygons = plg_indx2longlat(polygons, intf_coords, x_start)

        if not os.path.exists(Path(args.output_polygs_dir)):
            os.makedirs(args.output_polygs_dir)

        out_polyg_f = args.output_polygs_dir + intf + "_predicted_polyogns.shp"
        polygons.to_file(out_polyg_f)

        if args.add_gt_polygs:
            gdf = gpd.read_file(args.gt_polygons_file_path)

            if args.unified_mask and args.k_prevs > 0:
                curr_and_prevs_list = [intf] + prev_dict[intf]["prevs"]
            else:
                curr_and_prevs_list = [intf]

            gt_polygs, unified_geometry = build_unified_gt_polygons(gdf, curr_and_prevs_list)

        if args.plot_polygs:
            data_plot = d_cropped[-1][:, x_pxls_offset:]
            extent = [x_start, x_start + dx * data_plot.shape[1], common_y0 - dy * data_plot.shape[0], common_y0]
            plt.imshow(data_plot, extent=extent)

            for p in polygons.geometry:
                if p is None:
                    continue
                if p.geom_type == "Polygon":
                    x, y = p.exterior.xy
                    plt.plot(x, y, color="red")
                elif p.geom_type == "MultiPolygon":
                    for poly in p.geoms:
                        x, y = poly.exterior.xy
                        plt.plot(x, y, color="red")

            if args.add_gt_polygs and not gt_polygs.empty:
                for gp in gt_polygs.geometry:
                    if gp is None:
                        continue
                    if gp.geom_type == "Polygon":
                        xg, yg = gp.exterior.xy
                        plt.plot(xg, yg, color="blue")
                    elif gp.geom_type == "MultiPolygon":
                        for poly in gp.geoms:
                            xg, yg = poly.exterior.xy
                            plt.plot(xg, yg, color="blue")

            plt.show()