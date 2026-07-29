# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import argparse
import os
from pathlib import Path
from unet import *
# EDIT 2026-07-29: shared checkpoint -> model factory. CHANGELOG.md #12
from factory import architecture_from_flags, build_from_checkpoint
import torch
import logging
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import Polygon
import numpy as np
from affine import Affine
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
import rasterio.features
from shapely.geometry import shape
import json
import datetime
from remove_no_Data_predictions import *
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
def crop_to_start_xy(intf: np.ndarray,
                     mask: np.ndarray | None,
                     x0: float, y0: float,
                     x_star: float, y_star: float,
                     dx: float = 2.777e-05,
                     dy: float = 2.777e-05,
                     tol: float = 1e-2):
    """
    Crop intf (and optionally mask) so that the output starts at (x_star, y_star).

    If mask is None, only intf is cropped and mask_c is returned as None.
    """

    if mask is not None and intf.shape != mask.shape:
        raise ValueError(f"intf and mask must have same shape, got {intf.shape} vs {mask.shape}")

    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    H, W = intf.shape

    # --- compute offsets ---
    col_f = (x_star - x0) / dx
    row_f = (y0 - y_star) / dy  # y decreases downward

    col_off = int(round(col_f))
    row_off = int(round(row_f))

    # --- grid alignment check ---
    if abs(col_f - col_off) * dx > tol or abs(row_f - row_off) * dy > tol:
        raise ValueError(
            f"(x*,y*) not on pixel grid within tolerance: "
            f"col_off ~ {col_f}, row_off ~ {row_f}"
        )

    # --- bounds check ---
    if not (0 <= col_off < W) or not (0 <= row_off < H):
        raise ValueError(
            f"Requested start is outside the image: "
            f"row_off={row_off}, col_off={col_off}, shape={intf.shape}"
        )

    # --- crop size ---
    width = W - col_off
    height = H - row_off

    # --- crop ---
    intf_c = intf[row_off:row_off + height, col_off:col_off + width]

    if mask is not None:
        mask_c = mask[row_off:row_off + height, col_off:col_off + width]
    else:
        mask_c = None

    # --- new origin ---
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
    """
    Build per-interferogram chains:
      returns (chains_map, valid_currents)

      chains_map = {
         current_key: {
            "prevs": [prev_K, ..., prev_1],   # oldest -> most recent
            "frame": "North" or "South"
         },
         ...
      }

      valid_currents = [current_key, ...]     # only those with full K-previous chains

    Rules:
      - Only consider interferograms with duration == step_days (default 11).
      - 'frame' is taken from meta[k]['frame'] and must be 'North' or 'South'.
      - Previous frames are required exactly i*step_days earlier (i=1..K),
        and must belong to the SAME frame as current.
      - If require_current_nonz_gt0=True, current must have nonz_num > 0.
      - If restrict_to is provided, only those current keys are attempted.
    """

    def parse_key(k: str):
        s, e = k.split("_")
        sd = datetime.strptime(s, "%Y%m%d").date()
        ed = datetime.strptime(e, "%Y%m%d").date()
        return sd, ed, (ed - sd).days

    def make_key(sd, ed) -> str:
        return sd.strftime("%Y%m%d") + "_" + ed.strftime("%Y%m%d")

    # Precompute durations
    daydiff: Dict[str, int] = {}
    for k in meta.keys():
        sd, ed, dd = parse_key(k)
        daydiff[k] = dd

    all_keys = set(meta.keys())
    curr_keys = list(all_keys if restrict_to is None else (all_keys & set(restrict_to)))

    # Partition keys by frame
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
            continue  # skip 'Boundary'/missing/invalid

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

        # Build previous list within SAME frame group
        group = frame_groups[frm]
        prevs: List[str] = []
        ok = True
        # build in oldest->newest order
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
    intf_dict_file = open(_bootstrap.asset('intf_coord.json'), 'r')
    intf_coords = json.load(intf_dict_file)
    x0 = intf_coords[intf_name]['east']
    y0 = intf_coords[intf_name]['north']
    dx = intf_coords[intf_name]['dx']
    dy = intf_coords[intf_name]['dy']
    nlines = intf_coords[intf_name]['nlines']
    ncells = intf_coords[intf_name]['ncells']
    lidar_mask = intf_coords[intf_name]['lidar_mask']
    num_nonz_p = intf_coords[intf_name]['nonz_num']
    bo = intf_coords[intf_name]['byte_order']
    frame = intf_coords[intf_name]['frame']


    return (x0, y0, dx, dy,ncells, nlines,lidar_mask,num_nonz_p,bo,frame)

def get_intf_lidar_mask(intf_name):
    with open(_bootstrap.asset('lidar_intf_mask.txt'), 'r') as f:
        mask = 'no_mask'
        for line in f:
            if intf_name[:8] == line[8:16] and intf_name[9:17] == line[24:32]:
                mask = line[40:49]
            elif intf_name[:8] == line[8:16]:
                mask = line[40:49]


    return mask
def patchify(input_array, window_size, stride, mask_array= None,nonz_pathces = True,x_offset=3000):
    if mask_array is not None:
        assert input_array.shape == mask_array.shape, "Mask array should be the same shape as input array"
        mask_array = mask_array[:, x_offset:]


    input_array = input_array[:, x_offset:]
    rows, cols = input_array.shape
    data_patches,mask_patches,data_patches_nonz,mask_patches_nonz = [],[],[],[]

    for i in range(0, rows - window_size[0] + 1, stride[0]):
      data_row , mask_row = [],[]
      for j in range(0, cols - window_size[1] + 1, stride[1]):
         data_patch = input_array[i:i + window_size[0], j:j + window_size[1]]
         data_row.append(data_patch)
         if mask_array is not None:
            mask_patch = mask_array[i:i + window_size[0], j:j + window_size[1]]
            if mask_patch.any() !=0 and nonz_pathces:
                data_patches_nonz.append(data_patch)
                mask_patches_nonz.append(mask_patch)

      data_patches.append(data_row)
      mask_patches.append(mask_row)
    if mask_array is None:
       return np.array(data_patches)
    else:
        return np.array(data_patches),np.array(mask_patches), np.array(data_patches_nonz), np.array(mask_patches_nonz)

def reconstruct_intf_prediction(
    data,                   # (C, ny, nx, H, W) or (ny, nx, H, W) ; channel 0 = current
    intf_coords,            # tuple from get_intf_coords / your tuple logic
    net,                    # torch model
    patch_size,             # (patch_H, patch_W)
    stride,                 # int
    rth,                    # float threshold for reconstructed_pred_th
    mask=None,              # (ny, nx, H, W) or None (already cropped to data)
    add_lidar_mask=True,
    plot=False,
    lidar_sources=None,     # list[str|None] of length C; if None -> replicate current for all
    overlay_on_preds=True,
    device=None,
    x_offset = 3000,
    treat_nodata_regions=False,
    # ---- NEW: optional blending for predictions only ----
    blend=None,             # None | 'hann'
    pred_blend_dtype='float32',
    window_gamma = 1.0,
    blend_space = 'prob'
):
    """
    Returns (if mask is not None):
      reconstructed_intf_all, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs
    Else:
      reconstructed_intf, reconstructed_pred, reconstructed_prevs
    """
    # unpack coords (keep your ordering)
    x0, y0, dx, dy = intf_coords[:4]
    current_lidar_mask = intf_coords[8]
    patch_H, patch_W = patch_size

    # normalize shape
    if data.ndim == 5:  # (C, ny, nx, H, W)
        C, ny, nx, H, W = data.shape
        data_stack = data.astype(np.float32, copy=False)
    else:               # (ny, nx, H, W)
        ny, nx, H, W = data.shape
        C = 1
        data_stack = data[None].astype(np.float32, copy=False)

    # output sizes (original formula)
    out_h = ny * patch_H // stride + patch_H * (1 - 1 // stride)
    out_w = nx * patch_W // stride + patch_W * (1 - 1 // stride)

    # --- accumulators (original for intf & mask) ---
    reconstructed_intf_all = np.zeros((C, out_h, out_w), dtype=np.float32)
    reconstructed_pred_raw = np.zeros((out_h, out_w), dtype=np.float32)  # used only when blend is None
    reconstructed_mask     = np.zeros((out_h, out_w), dtype=np.float32) if mask is not None else None

    # --- optional blending only for predictions ---
    use_hann = (blend == 'hann')
    if use_hann:
        def get_hann_window(h, w,gamma=1.0):
            wy = np.hanning(h) if h > 1 else np.ones(1, dtype=np.float32)
            wx = np.hanning(w) if w > 1 else np.ones(1, dtype=np.float32)
            W  = (wy[:, None] * wx[None, :]).astype(np.float32)
            if gamma != 1.0:
                W = np.power(W, gamma, dtype=np.float32)
            return W / (W.max() + 1e-8)

        Wn = get_hann_window(patch_H, patch_W,gamma=window_gamma)
        # allow reduced precision to cut RAM if needed
        dtype_np = np.float16 if str(pred_blend_dtype).lower() == 'float16' else np.float32
        Wn = Wn.astype(dtype_np, copy=False)
        pred_num = np.zeros((out_h, out_w), dtype=dtype_np)
        pred_den = np.zeros((out_h, out_w), dtype=dtype_np)

    # -------- LiDAR mask(s): AND across time steps --------
    rasterized_polygon_current = None  # for overlay (channel 0)
    mask_polyg_current = None
    mask_all = None  # (C, out_h, out_w) uint8

    if add_lidar_mask:
        lidar_gdf = gpd.read_file(_bootstrap.asset('lidar_mask_polygs.shp'))
        # sources list
        if lidar_sources is None:
            srcs = [current_lidar_mask] * C
        else:
            srcs = list(lidar_sources)
            if len(srcs) != C:
                print(f"[warn] lidar_sources len {len(srcs)} != C {C}; using current for all")
                srcs = [current_lidar_mask] * C

        mask_all = np.zeros((C, out_h, out_w), dtype=np.uint8)
        tr = rasterio.transform.from_origin(x0+x_offset*dx, y0, dx, dy)  # same grid as recon

        for c, src in enumerate(srcs):
            if src is None or str(src).strip().lower() in ("", "none", "null"):
                polyg = lidar_gdf
            else:
                col = lidar_gdf['source'].astype(str).str.strip().str.lower()
                polyg = lidar_gdf[col == str(src).strip().lower()]
                if polyg.empty:
                    print(f"[warn] LiDAR source '{src}' not in shapefile -> using ALL polygons")
                    polyg = lidar_gdf

            mask_all[c] = rasterize(
                [(g, 1) for g in polyg['geometry'].tolist()],
                out_shape=(out_h, out_w),
                transform=tr,
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
            if c == 0:
                rasterized_polygon_current = mask_all[c].copy()
                mask_polyg_current = polyg

    # -------- main reconstruction --------
    step_y = patch_H // stride
    step_x = patch_W // stride
    data_stack = (data_stack+np.pi)/(2*np.pi)
    for i in range(ny):
        print(i)
        for j in range(nx):
            y0o = i * step_y; y1o = y0o + patch_H
            x0o = j * step_x; x1o = x0o + patch_W

            # accumulate all channels (original averaging)
            for c in range(C):
                reconstructed_intf_all[c, y0o:y1o, x0o:x1o] += data_stack[c, i, j] / (stride**2)

            if reconstructed_mask is not None:
                reconstructed_mask[y0o:y1o, x0o:x1o] += mask[i, j] / (stride**2)

            # LiDAR gating: must be inside ALL time-step masks for the whole patch
            is_within_mask = True
            if add_lidar_mask and (mask_all is not None):
                is_within_mask = mask_all[:, y0o:y1o, x0o:x1o].all()

            if not is_within_mask:
                continue

            # Build (1, C or 2C, H, W) for net and predict
            x_np = data_stack[:, i, j]  # (C, H, W)

            if treat_nodata_regions:
                tol = 1e-9
                if np.isnan(x_np).any():
                    x_np = np.nan_to_num(x_np, nan=0.0)
                v_np = (np.abs(x_np - 0.5) > tol).astype(np.float32)  # (C, H, W), 1=valid, 0=no-data
                x_np = np.concatenate([x_np, v_np], axis=0)  # (2C, H, W)

            image = torch.from_numpy(x_np[None]).to(device=device, memory_format=torch.channels_last)
            with torch.no_grad():
                logits = net(image)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)  # (H, W)

            # --- paste predictions
            if use_hann:
                # blend probabilities only (no extra buffers for intf/mask)
                pred_num[y0o:y1o, x0o:x1o] += (prob.astype(pred_blend_dtype) if pred_blend_dtype=='float16' else prob) * Wn
                pred_den[y0o:y1o, x0o:x1o] += Wn
            else:
                # original accumulation
                reconstructed_pred_raw[y0o:y1o, x0o:x1o] += prob / (stride**2)

    # ---- finalize prediction map
    if use_hann:
        eps = 1e-8
        reconstructed_pred = (pred_num / (pred_den + eps)).astype(np.float32)  # back to f32
    else:
        reconstructed_pred = reconstructed_pred_raw

    reconstructed_pred_th = (reconstructed_pred > rth).astype(np.float32)
    reconstructed_intf    = reconstructed_intf_all[0]
    reconstructed_prevs   = reconstructed_intf_all[1:] if C > 1 else None

    # -------- optional quick plot (overlay current LiDAR) --------
    if plot:
        extent = [x0+x_offset*dx , x0 +x_offset*dx + dx * out_w, y0 - dy * out_h, y0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.02)

        def overlay_current(ax):
            if not add_lidar_mask or rasterized_polygon_current is None:
                return
            ax.imshow(rasterized_polygon_current, extent=extent, cmap='Reds',
                      vmin=0, vmax=1, alpha=0.30, interpolation='nearest', zorder=2)
            if mask_polyg_current is not None:
                for geom in mask_polyg_current.geometry:
                    if geom is None: continue
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom)
                    for poly in polys:
                        xg, yg = poly.exterior.xy
                        ax.plot(xg, yg, color='yellow', linewidth=1.2, zorder=3)

        ax1.imshow(reconstructed_intf, extent=extent, cmap='gray', zorder=1)
        overlay_current(ax1)
        ax1.set_title('Reconstructed (current) + LiDAR')

        ax2.imshow(reconstructed_pred_th, extent=extent, cmap='binary', vmin=0, vmax=1, zorder=1)
        if overlay_on_preds:
            overlay_current(ax2)
        ax2.set_title('Prediction + LiDAR')

        for ax in (ax1, ax2):
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        plt.show()

    if reconstructed_mask is None:
        return reconstructed_intf, reconstructed_pred_th, reconstructed_prevs
    else:
        return reconstructed_intf_all, reconstructed_mask, reconstructed_pred_th, reconstructed_pred, reconstructed_prevs

def mask_array_to_polygons(mask_array):
    transform = Affine.identity()    # Generate polygons from the mask array
    polygons = shapes(mask_array, transform=transform)
    shapely_polygons = [shape(geom) for geom, value in polygons if value == 1]
    polygons_gpd = gpd.GeoDataFrame(geometry=shapely_polygons,crs="EPSG:4326")
    return polygons_gpd

def plg_indx2longlat(polyg_gdf, intf_coords,x_start):
    x0,y0, dx, dy = intf_coords[0],intf_coords[1], intf_coords[2], intf_coords[3]
    polyg_list = polyg_gdf['geometry'].tolist()
    polyg_longlat = []
    for polyg in polyg_list:
        new_coords = [(x_start + x * dx, y0 - y * dy) for x, y in polyg.exterior.coords]
        polyg = Polygon(new_coords)
        polyg_longlat.append(polyg)

    polyg_longlat_gdf = gpd.GeoDataFrame(geometry=polyg_longlat, crs='EPSG:4326')
    return polyg_longlat_gdf

def get_args():
    parser = argparse.ArgumentParser(description='Predict subsidence polygons for new interferogram')
    parser.add_argument('--intfs_dir',  type=str, default='./', help='interferogram path')
    parser.add_argument('--intfs_list', type=str, default='a list of intfs in format YYYYMMDD_YYYYMMDD seperated by comma')
    parser.add_argument('--model_dir', type=str,default='./models/')
    parser.add_argument('--model_file', type=str, help='the .pth model file')
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=(200,100), help='patch H, patch W. These have to match the patch size in training')
    parser.add_argument('--strdpp',type = int, default = 2, help='strides per patch in both directions')
    parser.add_argument('--plot_polygs', action='store_true', help='')
    parser.add_argument('--output_polygs_dir', type=str,default="./out_polygs")
    parser.add_argument('--rth',type = float, default=0.25)
    parser.add_argument('--add_gt_polygs', action='store_true', default=False)
    parser.add_argument('--gt_polygons_file_path', type=str, default='sub_20231001.shp')
    parser.add_argument('--align_frames',  action='store_true')
    parser.add_argument('--unioned_mask', action='store_true')
    parser.add_argument('--k_prevs', type=int, default=0)
    # EDIT 2026-07-29: architecture is detected from the checkpoint; these are optional
    # overrides. CHANGELOG.md #12
    parser.add_argument('--attn_unet', action='store_true',
                        help='Checkpoint is an AttentionUNet (usually auto-detected)')
    parser.add_argument('--add_attn', action='store_true',
                        help='Checkpoint is a UNet with bottleneck attention (usually auto-detected)')
    parser.add_argument('--convlstm_unet', action='store_true',
                        help='Checkpoint is a ConvLSTM U-Net (usually auto-detected)')
    parser.add_argument('--treat_nodata_regions', action='store_true',
                        help='Checkpoint was trained with validity channels (2 per timestep)')
    parser.add_argument('--blend_type', type=str, default=None, choices=['hann', 'center-crop'])
    parser.add_argument('--window_gamma', type=float, default=1.0)
    parser.add_argument('--plot',  action='store_true')
    parser.add_argument('--x_pxls_offset', type=int, default=3000)



    return parser.parse_args()


if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    args = get_args()
    x_pxls_offset = args.x_pxls_offset
    stride = (args.patch_size[0]//args.strdpp, args.patch_size[1]//args.strdpp)
    intfs_list = args.intfs_list.split(',')
    if args.k_prevs >0:
        with open(_bootstrap.asset('intf_coord.json'), "r") as f:
            intf_info = json.load(f)
        prev_dict, updated = find_11day_sequences(intf_info, k_prev=args.k_prevs, restrict_to=intfs_list)
    num_c = args.k_prevs + 1
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.model_dir + args.model_file, map_location=device)

    # EDIT 2026-07-29: was a hardcoded UNet; the factory detects the architecture from
    # the weights so ConvLSTM checkpoints load here too. CHANGELOG.md #12
    loaded = build_from_checkpoint(
        state_dict,
        arch=architecture_from_flags(
            attn_unet=getattr(args, 'attn_unet', False),
            add_attn=getattr(args, 'add_attn', False),
            convlstm_unet=getattr(args, 'convlstm_unet', False),
        ),
        n_classes=1,
        bilinear=False,
        treat_nodata_regions=getattr(args, 'treat_nodata_regions', False),
    )
    net = loaded.model
    mask_values = loaded.mask_values
    net.to(device=device)
    net.load_state_dict(state_dict)
    net.eval()
    logging.info(
        f'Model loaded! architecture={loaded.architecture} '
        f'in-channels={loaded.n_channels}'
    )
    if loaded.architecture != 'convlstm_unet' and loaded.n_channels not in (None, num_c):
        logging.warning(
            f'checkpoint expects {loaded.n_channels} input channels but --k_prevs '
            f'{args.k_prevs} produces {num_c}; the forward pass will fail. '
            f'Set --k_prevs {loaded.n_channels - 1}.'
        )

    for i, intf in enumerate(intfs_list):

        # --- current interferogram ---
        intf_coords = get_intf_coords(intf)
        x0, y0, dx, dy, ncells, nlines, intf_lidar_mask, num_nz, bo, frame = intf_coords

        filename = next(
            f for f in os.listdir(args.intfs_dir)
            if f.endswith('.unw') and intf[:8] in f and intf[9:] in f
        )

        path = os.path.join(args.intfs_dir, filename)

        data = np.fromfile(path, dtype=np.float32).reshape(nlines, ncells)

        if bo == 'MSBFirst':
            data = data.byteswap().newbyteorder('<')

        # --- previous interferograms ---
        prev_coords, prev_data, prev_lidar = [], [],[]

        if args.k_prevs > 0:
            for item in prev_dict[intf]['prevs']:
                coords = get_intf_coords(item)
                bo_item, ncells_item, nlines_item = coords[8], coords[4], coords[5]
                prev_lidar.append(coords[6])
                prev_coords.append(coords)

                fname = next(
                    f for f in os.listdir(args.intfs_dir)
                    if f.endswith('.unw') and item[:8] in f and item[9:] in f
                )

                p = os.path.join(args.intfs_dir, fname)

                prd = np.fromfile(p, dtype=np.float32).reshape(nlines_item, ncells_item)

                if bo_item == 'MSBFirst':
                    prd = prd.byteswap().newbyteorder('<')

                prev_data.append(prd)

        # --- aggregate ---
        all_data = prev_data.copy()
        all_coords = prev_coords.copy()
        all_lidar = prev_lidar.copy()

        all_data.append(data)
        all_coords.append(intf_coords)
        all_lidar.append(intf_lidar_mask)

        common_x0 = max(c[0] for c in all_coords)
        common_y0 = min(c[1] for c in all_coords)
        intf_coords = (common_x0, common_y0) + intf_coords[2:]
        x_start = common_x0+args.x_pxls_offset*dx
        d_cropped,patchified_intfs = [],[]
        for i,d in enumerate(all_data):
            d_c,_,_,_,_=crop_to_start_xy(d,None,all_coords[i][0],all_coords[i][1],common_x0,common_y0)
            d_cropped.append(d_c)
            patchified_intfs.append(patchify(d_c, args.patch_size, stride, mask_array=None, nonz_pathces=False,x_offset=x_pxls_offset))

        min_ny = min(p.shape[0] for p in patchified_intfs)
        min_nx = min(p.shape[1] for p in patchified_intfs)
        patchified_intfs = [
            p[:min_ny, :min_nx] for p in patchified_intfs
        ]
        patchified_intf = np.array(patchified_intfs)
        reconstructed_intf,reconstructed_pred,reconstructed_prevs = reconstruct_intf_prediction(patchified_intf,intf_coords,net,args.patch_size,args.strdpp,args.rth,lidar_sources=all_lidar,plot=args.plot,x_offset=x_pxls_offset)


        polygons = mask_array_to_polygons(reconstructed_pred)
        polygons = plg_indx2longlat(polygons,intf_coords,x_start)

        if not os.path.exists(Path(args.output_polygs_dir)):
            os.makedirs(args.output_polygs_dir)
        out_polyg_f = args.output_polygs_dir + intf +'_predicted_polyogns.shp'
        polygons.to_file(out_polyg_f)


        if args.add_gt_polygs:
            gdf = gpd.read_file(args.gt_polygons_file_path)

            if args.unioned_mask and args.k_prevs > 0:
                curr_and_prevs_list = [intf] + prev_dict[intf]['prevs']
            else:
                curr_and_prevs_list = [intf]

            gt_polygs_list = []

            for intf_name in curr_and_prevs_list:
                start_date = intf_name[:8]
                end_date = intf_name[9:]


                def add_dashes(date_str):
                    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                curr_gt_polygs = gdf[
                    (gdf['start_date'] == add_dashes(start_date)) &
                    (gdf['end_date'] == add_dashes(end_date))
                    ]

                if not curr_gt_polygs.empty:
                    gt_polygs_list.append(curr_gt_polygs)

            if not gt_polygs_list:
                print(f'No ground truth for interferograms: {curr_and_prevs_list}. Skipping...')
                gt_polygs = gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
                unified_geometry = None
            else:
                all_gt_polygs = gpd.GeoDataFrame(
                    pd.concat(gt_polygs_list, ignore_index=True),
                    crs=gdf.crs
                )
                unified_geometry = all_gt_polygs.unary_union
                gt_polygs = gpd.GeoDataFrame(geometry=[unified_geometry], crs=gdf.crs)


        if args.plot_polygs:
            data = d_cropped[-1][:, x_pxls_offset:]
            extent = [x_start, x_start + dx * data.shape[1], y0 - dy * data.shape[0], y0]
            plt.imshow(data, extent=extent)

            list_of_polygons = polygons.geometry.tolist()
            for p in list_of_polygons:
                x, y = p.exterior.xy
                plt.plot(x, y, color='red')

            if args.add_gt_polygs and not gt_polygs.empty:
                for gp in gt_polygs.geometry:
                    if gp is None:
                        continue

                    if gp.geom_type == 'Polygon':
                        xg, yg = gp.exterior.xy
                        plt.plot(xg, yg, color='blue')

                    elif gp.geom_type == 'MultiPolygon':
                        for poly in gp.geoms:
                            xg, yg = poly.exterior.xy
                            plt.plot(xg, yg, color='blue')

            plt.show()


