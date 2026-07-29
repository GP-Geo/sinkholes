# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
import argparse
from pathlib import Path
import os
import json

from train_sinkholes_unet import str2bool
from get_intf_info import *

logging.basicConfig(level=logging.INFO)

def patchify(input_array, window_size, stride, mask_array=None, nonz_pathces=True,offset = 0,nx=4500):
    """
    Returns:
      data_patches         -> (Xi, Yi, H, W)
      mask_patches         -> (Xi, Yi, H, W)
      data_patches_nonz    -> (N, H, W)  (N could be 0)
      mask_patches_nonz    -> (N, H, W)  (N could be 0)
      nonz_indices         -> list[[i,j], ...] grid coords in (Xi, Yi)
    """
    if mask_array is not None:
        assert input_array.shape == mask_array.shape, "Mask array should be the same shape as input array"
        mask_array = mask_array[:, offset:offset+nx]

    input_array = input_array[:, offset:offset+nx]
    rows, cols = input_array.shape
    H, W = window_size
    Sy, Sx = stride

    data_patches, mask_patches = [], []
    data_patches_nonz, mask_patches_nonz = [], []
    nonz_indices = []  # [(row_idx_in_patch_grid, col_idx_in_patch_grid), ...]

    idx_i = 0
    for i in range(0, rows - H + 1, Sy):
        data_row, mask_row = [], []
        idx_j = 0
        for j in range(0, cols - W + 1, Sx):
            data_patch = input_array[i:i + H, j:j + W]
            data_row.append(data_patch)

            if mask_array is not None:
                mask_patch = mask_array[i:i + H, j:j + W]
                mask_row.append(mask_patch)

                if nonz_pathces and mask_patch.any():
                    data_patches_nonz.append(data_patch)
                    mask_patches_nonz.append(mask_patch)
                    nonz_indices.append([idx_i, idx_j])

            idx_j += 1
        data_patches.append(data_row)
        if mask_array is not None:
            mask_patches.append(mask_row)
        idx_i += 1

    data_patches = np.array(data_patches)
    mask_patches = np.array(mask_patches) if mask_array is not None else None

    # Ensure empty nonz arrays have consistent shape (0, H, W)
    if len(data_patches_nonz) == 0:
        data_patches_nonz = np.empty((0, H, W), dtype=input_array.dtype)
        mask_patches_nonz = np.empty((0, H, W), dtype=np.uint8)
    else:
        data_patches_nonz = np.array(data_patches_nonz)
        mask_patches_nonz = np.array(mask_patches_nonz)

    if mask_array is None:
        return data_patches
    else:
        return (
            data_patches,
            mask_patches,
            data_patches_nonz,
            mask_patches_nonz,
            nonz_indices,
        )

def get_args():
    parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')
    parser.add_argument('--by_list',  type=str, default=None, help='From an input list')
    parser.add_argument('--input_dir',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/', help='full unw files directory for input')
    parser.add_argument('--output_dir',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='patches output')
    parser.add_argument('--gt_polygon_file_path',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/sub_20231001.shp', help='')
    parser.add_argument('--plot_data',  type=bool, default=False)
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--strides_per_patch',type=int, default=2, help='strides per patch - 2 means half a window stride, 4 means quarter a window stride etc')
    parser.add_argument('--align_frames',  action='store_true', default=True)
    parser.add_argument('--days_diff',  type=int, default=11)
    parser.add_argument('--x0_N', type=float, default=35.37)
    parser.add_argument('--x0_S', type=float, default=35.32)
    parser.add_argument('--offset_x', type=int, default=0)
    parser.add_argument('--nx',type=int,default=4500)
    parser.add_argument('--year_range', nargs=2, type=int, default=None,
                        metavar=('YEAR_START', 'YEAR_END'),
                        help='Only process intfs whose start year is in this range (inclusive)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    req_days_diff = args.days_diff
    gdf = gpd.read_file(args.gt_polygon_file_path)
    patch_size = tuple(args.patch_size)
    patch_H, patch_W = patch_size
    data_output_dir = (
        args.output_dir
        + f"data_patches_H{patch_H}_W{patch_W}_strpp{args.strides_per_patch}"
        + (f"_{req_days_diff}days")
        + ('_Aligned' if args.align_frames else '')
    )
    mask_output_dir = (
        args.output_dir
        + f"mask_patches_H{patch_H}_W{patch_W}_strpp{args.strides_per_patch}"
        + (f"_{req_days_diff}days")
        + ('_Aligned' if args.align_frames else '')
    )

    for item in [data_output_dir, mask_output_dir]:
        if not os.path.exists(item):
            os.makedirs(item)
        else:
            logging.info('Note! These patches already exist! If you want to recreate them - remove the existing dir first')

    if args.by_list is None:
        input_file_paths = list(Path(args.input_dir).glob('*.unw'))
    else:
        input_files = []
        input_file_paths = []
        input_intfs = args.by_list.split(',')
        for filename in os.listdir(args.input_dir):
            if filename.endswith('.unw') and any(
                input_intf[:8] == filename[9:17] and input_intf[9:] == filename[25:33]
                for input_intf in input_intfs
            ):
                input_files.append(filename)
                input_file_paths = [Path(args.input_dir) / file for file in input_files]

    nonzero_mask_inds_by_intf = {}

    for item in input_file_paths:
        gfile_name = item.name
        intfrgrm_name = gfile_name.split('.')[0][9:17] + gfile_name.split('.')[0][24:33]
        start_date = f"{intfrgrm_name[0:4]}-{intfrgrm_name[4:6]}-{intfrgrm_name[6:8]}"
        end_date   = f"{intfrgrm_name[9:13]}-{intfrgrm_name[13:15]}-{intfrgrm_name[15:17]}"

        start_datetime = datetime.strptime(start_date,"%Y-%m-%d")
        end_datetime = datetime.strptime(end_date,"%Y-%m-%d")
        days_diff = (end_datetime - start_datetime).days
        if days_diff != req_days_diff:
            continue
        if args.year_range is not None:
            yr0, yr1 = args.year_range
            if not (yr0 <= start_datetime.year <= yr1):
                continue

        intf_mdata = get_intf_coords(intfrgrm_name)
        x0, y0, dx, dy = intf_mdata[0], intf_mdata[1], intf_mdata[2], intf_mdata[3]
        ncells, nlines = intf_mdata[4], intf_mdata[5]
        bo, frame      = intf_mdata[10], intf_mdata[11]

        data = np.fromfile(os.path.join(args.input_dir, gfile_name), dtype=np.float32).reshape(nlines, ncells)
        if bo == 'MSBFirst':
            data = data.byteswap().newbyteorder('<')

        # Build transform and CRS
        transform = rasterio.transform.from_origin(x0, y0, dx, dy)
        height, width = nlines, ncells

        # Select polygons for this INTF
        subset_polygs = gdf[(gdf['start_date'] == start_date) & (gdf['end_date'] == end_date)]

        # Make mask (handle empty polygons)
        if subset_polygs.empty:
            logging.warning(f"NO GT polygons for {intfrgrm_name} ({start_date}–{end_date}); using empty mask")
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = geometry_mask(
                subset_polygs.geometry,
                transform=transform,
                invert=True,
                out_shape=(height, width)
            ).astype(np.uint8)

        extent = [x0, x0 + dx * ncells, y0 - dy * nlines, y0]
        logging.info('initial long lat range: {}'.format(extent))

        if args.plot_data:
            plt.rcParams['backend'] = 'Qt5Agg'
            plt.imshow(data, extent=extent, cmap='Greys')
            plt.title('Interferogram '+ start_date + ' - ' + end_date)
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.show()

            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True, gridspec_kw={'width_ratios': [1, 1]})
            if not subset_polygs.empty:
                subset_polygs.plot(ax=ax[0], edgecolor='blue', facecolor='none')
            ax[0].set_title('subset polygons')
            mask_im = ax[1].imshow(mask, cmap='gray', extent=extent)
            ax[1].set_title('subset polygons mask')
            fig.colorbar(mask_im, ax=ax[1], shrink=0.8)
            plt.show()

            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True, gridspec_kw={'width_ratios': [1, 1]})
            c0 = ax[0].imshow(data, extent=extent, cmap='gray')
            ax[0].set_title(f'Interferogram {intfrgrm_name}', fontsize=12)
            ax[0].set_xlabel('Longitude (deg)')
            ax[0].set_ylabel('Latitude (deg)')
            ax[1].imshow(mask, extent=extent, cmap='gray', vmin=0)
            fig.colorbar(c0, ax=ax[0], shrink=0.5)
            if not subset_polygs.empty:
                for i in range(len(subset_polygs)):
                    coords_e = subset_polygs.iloc[i]['geometry'].exterior.coords
                    x, y = zip(*list(coords_e))
                    ax[0].plot(x, y)
                    ax[1].plot(x, y)
            plt.tight_layout()
            plt.show()

        # alignment
        if args.align_frames:
            if frame == 'North':
                data, mask, _, _, _ = crop_to_start_xy(data, mask, x0, y0, args.x0_N, 31.79)
            else:
                data, mask, _, _, _ = crop_to_start_xy(data, mask, x0, y0, args.x0_S, 31.44)

        # Patchify (will yield empty nonz arrays if mask has no positives)
        data_patches, mask_patches, data_patches_nonz, mask_patches_nonz, nonz_indices = patchify(
            data,
            patch_size,
            stride=(patch_size[0] // args.strides_per_patch, patch_size[1] // args.strides_per_patch),
            mask_array=mask,
            offset=args.offset_x,
            nx = args.nx
        )
        nonzero_mask_inds_by_intf[intfrgrm_name] = nonz_indices

        counter_nonz = data_patches_nonz.shape[0]
        logging.info(f'intrfrgrm {intfrgrm_name}: number of non-zero patches: {counter_nonz}')

        # Save
        ext = f"{intfrgrm_name}_H{patch_H}_W{patch_W}_strpp{args.strides_per_patch}.npy"
        out_file_names = ['data_patches_', 'mask_patches_', 'data_patches_nonz_', 'mask_patches_nonz_']
        arrays = [data_patches, mask_patches, data_patches_nonz, mask_patches_nonz]

        for name, array in zip(out_file_names, arrays):
            output_dir = mask_output_dir if 'mask' in name else data_output_dir
            out_file_path = os.path.join(output_dir, name + ext)
            np.save(out_file_path, array)
            # sanity check
            loaded = np.load(out_file_path, allow_pickle=False)
            logging.info(f'shape of file saved: {name}{loaded.shape}')

        # Optional preview of non-zero patches only
        if args.plot_data and counter_nonz > 0:
            m_nz = np.load(os.path.join(mask_output_dir, 'mask_patches_nonz_' + ext))
            d_nz = np.load(os.path.join(data_output_dir, 'data_patches_nonz_' + ext))
            for i in range(0, max(0, len(d_nz)-8), 8):
                fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
                axes = axes.flatten()
                full = []
                for j in range(8):
                    axes[2*j].imshow(d_nz[i+j], cmap='gray')
                    axes[2*j+1].imshow(m_nz[i+j], cmap='gray')
                    axes[2*j].set_title(f'Patch {i+j}', fontsize=7)
                    full.append(m_nz[i+j])
                full = np.array(full)
                if np.any(full > 0):
                    plt.show()
                else:
                    plt.close()

    # Write non-zero indices JSON (pretty-chunked lines)
    items_per_line = 50
    out_path = os.path.join(data_output_dir, "nonz_indices.json")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        keys = list(nonzero_mask_inds_by_intf.keys())
        for ki, intf_id in enumerate(keys):
            pairs = nonzero_mask_inds_by_intf[intf_id]
            pairs = [(int(i), int(j)) for (i, j) in pairs]
            f.write(f'  "{intf_id}": [\n')
            n = len(pairs)
            if n == 0:
                f.write("  ]")
            else:
                for i, (pi, pj) in enumerate(pairs):
                    if i % items_per_line == 0:
                        f.write("    ")
                    f.write(f"[{pi},{pj}]")
                    if i != n - 1:
                        f.write(", ")
                    if (i % items_per_line == items_per_line - 1) and (i != n - 1):
                        f.write("\n")
                f.write("\n  ]")
            if ki != len(keys) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("}\n")

    print("Wrote", out_path)
