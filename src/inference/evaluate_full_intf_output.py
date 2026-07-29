# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import glob

import numpy as np
import matplotlib.pyplot as plt
from evaluate import *
import os
import argparse
import geopandas as gpd
from get_intf_info import *
import json
from get_intf_info import *


import json

import geopandas as gpd

from scipy.ndimage import gaussian_filter

def plot_full_intfs_preds_vprevs_wlidarmask(
    image,
    mask_true, mask_pred, mask_pred2, mask_pred3, mask_pred_conf,
    *,
    add_lidar_mask=True,
    intf_lidar_mask=None,            # <-- pass the CURRENT INTF's lidar source id
    lidar_shp_path=_bootstrap.asset('lidar_mask_polygs.shp'),
    overlay_on_preds=True            # overlay lidar on pred/conf panels too
):
    from scipy.ndimage import gaussian_filter
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    import numpy as np
    import matplotlib.pyplot as plt

    # smooth confidence a bit
    mask_pred_conf = gaussian_filter(mask_pred_conf.astype(np.float32), sigma=2)

    # base image & extent (must match rasterization grid)
    base_img = image[0] if (image.ndim == 3 and image.shape[0] >= 1) else image
    H, W = base_img.shape[-2], base_img.shape[-1]
    extent = [x0, x0 + dx * W, y0 - dy * H, y0]

    # ---- LiDAR: read + filter to CURRENT INTF only + rasterize to (H,W) ----
    rasterized_polygon = None
    mask_polyg = None
    if add_lidar_mask:
        try:
            lidar_df = gpd.read_file(lidar_shp_path)
            # filter to CURRENT interferogram only
            mask_polyg = lidar_df if intf_lidar_mask is None else \
                         lidar_df[lidar_df['source'] == intf_lidar_mask]
            if mask_polyg is None or len(mask_polyg) == 0:
                mask_polyg = lidar_df  # fallback: show all

            rasterized_polygon = rasterize(
                [(g, 1) for g in mask_polyg['geometry'].tolist()],
                out_shape=(H, W),
                transform=rasterio.transform.from_origin(x3000, y0, dx, dy),
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
        except Exception as e:
            print(f"[LiDAR overlay skipped] {e}")
            rasterized_polygon, mask_polyg = None, None

    def _overlay_lidar(ax):
        if rasterized_polygon is None or mask_polyg is None:
            return
        # semi-transparent raster
        ax.imshow(rasterized_polygon, extent=extent, cmap='Reds',
                  vmin=0, vmax=1, alpha=0.30, interpolation='nearest', zorder=2)
        # polygon outlines
        try:
            for geom in mask_polyg.geometry:
                if geom is None: continue
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom)
                for poly in polys:
                    xg, yg = poly.exterior.xy
                    ax.plot(xg, yg, color='yellow', linewidth=1.2, zorder=3)
                    for ring in getattr(poly, "interiors", []):
                        rx, ry = zip(*ring.coords[:])
                        ax.plot(rx, ry, color='yellow', linewidth=0.8, linestyle='--', zorder=3)
        except Exception:
            pass
        # lock to image bounds
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # layout
    show_three = (image.ndim == 3 and image.shape[0] == 3)
    ncols = 6 + (2 if show_three else 0)
    fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True,
                             figsize=(14 + 4*(ncols-6), 10),
                             gridspec_kw={'wspace': 0.01})
    axes = list(axes)

    col = 0
    if show_three:
        axes[col].imshow(image[0].astype(np.float32), extent=extent, cmap='jet', zorder=1)
        axes[col].set_title('Current\nInterferogram', fontsize=18); _overlay_lidar(axes[col]); col += 1
        axes[col].imshow(image[1].astype(np.float32), extent=extent, cmap='jet', zorder=1)
        axes[col].set_title('Prev (t-1)\nInterferogram', fontsize=18); _overlay_lidar(axes[col]); col += 1
        axes[col].imshow(image[2].astype(np.float32), extent=extent, cmap='jet', zorder=1)
        axes[col].set_title('Prev (t-2)\nInterferogram', fontsize=18); _overlay_lidar(axes[col]); col += 1
    else:
        axes[col].imshow(base_img.astype(np.float32), extent=extent, cmap='jet', zorder=1)
        axes[col].set_title('Original\nInterferogram', fontsize=18); _overlay_lidar(axes[col]); col += 1

    def _sl2d(a): return a if a.ndim == 2 else a[0]
    mt  = _sl2d(mask_true).astype(np.float32)
    mp1 = _sl2d(mask_pred).astype(np.float32)
    mp2 = _sl2d(mask_pred2).astype(np.float32)
    mp3 = _sl2d(mask_pred3).astype(np.float32)
    mpc = _sl2d(mask_pred_conf).astype(np.float32)

    # show masks/probs with fixed [0,1] range; overlay LiDAR on preds too
    def _show_mask(ax, arr, title, overlay=False, cmap='binary'):
        ax.imshow(arr, extent=extent, vmin=0, vmax=1, cmap=cmap, zorder=1)
        if overlay: _overlay_lidar(ax)
        ax.set_title(title, fontsize=18)

    _show_mask(axes[col], mt,  'True \nMask', overlay=False); col += 1
    _show_mask(axes[col], mp1, 'Pred Mask\n$\\mathbf{(RTh=0.25)}$', overlay=overlay_on_preds); col += 1
    _show_mask(axes[col], mp2, 'Pred Mask\n$\\mathbf{(RTh=0.5)}$',  overlay=overlay_on_preds); col += 1
    _show_mask(axes[col], mp3, 'Pred Mask\n$\\mathbf{(RTh=0.75)}$',   overlay=overlay_on_preds); col += 1
    _show_mask(axes[col], mpc, 'Prediction\nConfidence',             overlay=overlay_on_preds, cmap='viridis')

    # labels & axis cosmetics
    labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)'][:ncols]
    for ax, lab in zip(axes, labels):
        ax.text(0.00, 1.0, lab, transform=ax.transAxes, fontsize=18, fontweight='bold',
                va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, boxstyle='round,pad=0.2'))
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.tick_params(axis='x', labelsize=14, rotation=45)
        ax.tick_params(axis='y', labelsize=14)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    axes[0].set_xlabel('Longitude (°E)', fontsize=18)
    axes[0].set_ylabel('Latitude (°N)', fontsize=18, rotation=90)

    fig.savefig('/Users/galidek/Desktop/paper_figs/full_intfs/' + item + '_june25', dpi=1000)
    plt.show()

def plot_prediction_hist_all(all_feature_lists):
    feature_names = list(all_feature_lists.keys())
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 2, figsize=(10, 4 * n_features))

    det_color = '#2ca02c'   # green
    undet_color = '#d62728' # red
    det_color = '#98df8a'  # lighter green (pastel)
    undet_color = '#ff9896'  # lighter red (salmon/pinkish)

    fig, axes = plt.subplots(n_features, 2, figsize=(10, 4 * n_features), sharex=False)

    for i, feature_name in enumerate(feature_names):
        ...
        ax1, ax2 = axes[i]
        ax2.sharex(ax1)
        all_detected = np.array(all_feature_lists[feature_name][0])
        all_undetected = np.array(all_feature_lists[feature_name][1])

        all_values = np.concatenate([all_detected, all_undetected])
        all_values = all_values[np.isfinite(all_values)]

        bin_count = 50
        if feature_name == 'year':
            bin_count=5
        min_val = np.min(all_values)
        max_val = np.max(all_values)

        # Feature-specific bin limits
        if feature_name == 'area':
            max_val = 2500
        elif feature_name == 'perimeter':
            max_val = 500
        elif feature_name == 'solidity':
            min_val, max_val = 0.7, 1
        elif feature_name == 'roundness':
            min_val, max_val = 0.2, 0.7
        elif feature_name in ['phase_std', 'phase_fft']:
            max_val = 0.5
        if min_val == max_val:
            max_val += 1

        bins = np.linspace(min_val, max_val, bin_count + 1)
        det_counts, _ = np.histogram(all_detected, bins=bins)
        undet_counts, _ = np.histogram(all_undetected, bins=bins)

        total_counts = det_counts + undet_counts
        valid = total_counts > 0

        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_centers = bin_centers[valid]
        width = np.diff(bins)[valid] * 0.9  # thinner bars
        det_counts = det_counts[valid]
        undet_counts = undet_counts[valid]
        total_counts = total_counts[valid]

        ax1, ax2 = axes[i]

        # Count plot
        ax1.bar(bin_centers, det_counts, width=width, label='Detected', color=det_color, edgecolor='black', linewidth=0.3)
        ax1.bar(bin_centers, undet_counts, width=width, bottom=det_counts, label='Undetected', color=undet_color, edgecolor='black', linewidth=0.3)
        ax1.set_ylabel('Count', fontsize=7)
        ax1.set_title(f'{feature_name} – Count', fontsize=7)
        ax1.legend(fontsize=6)
        ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax1.tick_params(labelsize=6)

        # Percentage plot
        det_pct = det_counts / total_counts * 100
        undet_pct = undet_counts / total_counts * 100
        ax2.bar(bin_centers, det_pct, width=width, label='Detected (%)', color=det_color, edgecolor='black', linewidth=0.3)
        ax2.bar(bin_centers, undet_pct, width=width, bottom=det_pct, label='Undetected (%)', color=undet_color, edgecolor='black', linewidth=0.3)
        ax2.set_xlabel(f'{feature_name} bins', fontsize=9)
        ax2.set_ylabel('Percentage (%)', fontsize=7)
        ax2.set_title(f'{feature_name} – Percentage', fontsize=7)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=6)
        ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax2.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def overlay_black_where_value(ax, arr, extent, value=0.5, atol=1e-6):
    # mask True exactly where arr == value (with tolerance for float)
    m = np.isclose(arr, value, atol=atol)
    # masked array: only show where mask is True, elsewhere transparent
    overlay = np.ma.masked_where(~m, arr)
    ax.imshow(overlay, extent=extent, cmap=ListedColormap(['black']),
              interpolation='nearest', zorder=10)

def plot_full_intfs_preds_vprevs(image, mask_true, mask_pred, mask_pred2, mask_pred3, mask_pred_conf, intf_name):
    gdf = gpd.read_file('sub_20231001.shp')
    start = intf_name[:8]
    end = intf_name[9:17]
    start_date = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_date = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
    gsi_sub = gdf[(gdf['start_date'] == start_date) & (gdf['end_date'] == end_date)]

    from scipy.ndimage import gaussian_filter
    mask_pred_conf = gaussian_filter(mask_pred_conf, sigma=2)

    # --- format title from yyyymmdd_yyyymmdd -> dd.mm.yyyy → dd.mm.yyyy
    def _fmt_intf_title(name: str) -> str:
        try:
            a, b = name.strip().split('_', 1)
            d1 = datetime.strptime(a, "%Y%m%d").strftime("%d.%m.%Y")
            d2 = datetime.strptime(b, "%Y%m%d").strftime("%d.%m.%Y")
            return f"{d1} \u2192 {d2}"   # → (RIGHTWARDS ARROW)
        except Exception:
            return name  # fallback (leave as-is)

    # choose a base image for extent
    base_img = image[0] if (image.ndim == 3 and image.shape[0] >= 1) else image
    extent = [x0, x0 + dx * base_img.shape[1], y0 - dy * base_img.shape[0], y0]

    # do we have 3 images to show (current + two prevs)?
    show_three = (image.ndim == 3 and image.shape[0] == 3)

    # total columns: 6 baseline + 2 extra if we have 3 images
    ncols = 6 + (2 if show_three else 0)
    # ncols = 5 + (2 if show_three else 0)
    fig, axes = plt.subplots(
        1, ncols, sharex=True, sharey=True, figsize=(14 + 4*(ncols-6), 10),
        gridspec_kw={'wspace': 0.01}
    )
    # fig, axes = plt.subplots(
    #     1, ncols, sharex=True, sharey=True, figsize=(14 + 4*(ncols-5), 10),
    #     gridspec_kw={'wspace': 0.2}
    # )
    axes = list(axes)  # make indexable

    col = 0

    def _poly_coords_iter(geom):
        """Yield exterior coordinate arrays for Polygon/MultiPolygon."""
        if geom is None:
            return
        if isinstance(geom, Polygon):
            yield geom.exterior.coords
        elif isinstance(geom, MultiPolygon):
            for g in geom.geoms:
                yield g.exterior.coords

    if show_three:
        # three interferograms: current + prev t-1 + prev t-2
        axes[col].imshow(image[0], extent=extent, cmap='jet')
        axes[col].set_title('Current\nInterferogram', fontsize=18);
        for geom in gsi_sub.geometry:
            for coords in _poly_coords_iter(geom):
                xy = np.array(coords)
                axes[col].plot(xy[:, 0], xy[:, 1],
                        c='white', lw=5)


        col += 1

        axes[col].imshow(image[1], extent=extent, cmap='jet')
        axes[col].set_title('Prev (t-1)\nInterferogram', fontsize=18); col += 1

        axes[col].imshow(image[2], extent=extent, cmap='jet')
        axes[col].set_title('Prev (t-2)\nInterferogram', fontsize=18); col += 1
    else:
        # single interferogram
        axes[col].imshow(base_img, extent=extent, cmap='jet')

        axes[col].set_title('Original\nInterferogram', fontsize=14)
        axes[col].imshow(base_img, extent=extent, cmap='jet')
        overlay_black_where_value(axes[col], base_img, extent)
        for geom in gsi_sub.geometry:
            for coords in _poly_coords_iter(geom):
                xy = np.array(coords)
                axes[col].plot(xy[:, 0], xy[:, 1],
                        c='white', lw=1.5)

        col+=1
    # helper to get 2D slice if arrays come as (T,H,W)
    def _sl2d(a):
        return a if a.ndim == 2 else a[0]

    mt  = _sl2d(mask_true)
    mp1 = _sl2d(mask_pred)
    mp2 = _sl2d(mask_pred2)
    mp3 = _sl2d(mask_pred3)
    mpc = _sl2d(mask_pred_conf)

    # True & predicted masks + confidence
    axes[col].imshow(mt * 50, extent=extent, vmin=0, vmax=1); axes[col].set_title('True \nMask', fontsize=14); col += 1
    axes[col].imshow(mp1 * 250, extent=extent, vmin=0, vmax=1); axes[col].set_title('Predicted Mask\n$\\mathbf{(RTh=0.125)}$', fontsize=14); col += 1
    axes[col].imshow(mp2 * 250, extent=extent, vmin=0, vmax=1); axes[col].set_title('Predicted Mask\n$\\mathbf{(RTh=0.25)}$',  fontsize=14); col += 1
    axes[col].imshow(mp3 * 250, extent=extent, vmin=0, vmax=1); axes[col].set_title('Predicted Mask\n$\\mathbf{(RTh=0.5)}$',   fontsize=14); col += 1
    axes[col].imshow(mpc,        extent=extent, vmin=0, vmax=1, cmap='seismic'); axes[col].set_title('Prediction\nConfidence', fontsize=14)


    # panel labels
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'][:ncols]
    for ax, lab in zip(axes, labels):
        ax.text(
            0.00, 1.0, lab,
            transform=ax.transAxes,
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.5,
                boxstyle='round,pad=0.2'
            ),
            zorder=10,  # <<--- make sure text is on top
            clip_on=False  # <<--- avoids clipping at axes border
        )
    # axis cosmetics
    axes[0].set_xlabel('Longitude (°E)', fontsize=18)
    axes[0].set_ylabel('Latitude (°N)', fontsize=18, rotation=90)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=14, rotation=45)
        ax.tick_params(axis='y', labelsize=14)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    # --- title with real dates and a real arrow
    fig.suptitle(f"Interferogram: {_fmt_intf_title(intf_name)}", fontsize=16)
    fig.subplots_adjust(top=0.88)  # keep title clear of panels

    # save & show (kept as in your code; assumes `item` exists in scope)
    # fig.savefig('/Users/galidek/Desktop/paper_figs/full_intfs/' + item + '_nov25', dpi=1000)
    plt.show()

def plot_full_intfs_preds(image,mask_true,mask_pred,mask_pred2,mask_pred3,mask_pred_conf):
    mask_pred_conf = gaussian_filter(mask_pred_conf, sigma=2)

    for i in range(mask_pred.shape[0]):
        extent = [x0, x0+ dx * image[i].shape[1], y0 - dy * image[i].shape[0], y0]

        # fig, (ax1, ax2, ax3, ax4, ax5,ax6) = plt.subplots(
        #     1, 6,
        #     sharex=True,
        #     sharey=True,
        #     figsize=(14, 10),
        #     gridspec_kw={'wspace': 0.01}
        # )
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            1, 5,
            sharex=True,
            sharey=True,
            figsize=(14, 10),
            gridspec_kw={'wspace': 0.01}
        )

        # 🔥 This makes the panels truly wider by minimizing unused margins
        # fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.12)

        # Plot content
        ax1.imshow(image[i], extent=extent, cmap='jet')
        ax1.set_title('Original\nInterferogram\n', fontsize=18)
        ax1.set_xlabel('Longitude (°E)', fontsize=18)
        ax1.set_ylabel('Latitude (°N)', fontsize=18, rotation=90)

        ax2.imshow(mask_true[i]*50 , extent=extent,  vmin=0, vmax=1)
        ax2.set_title('Ground \nTruth \nMask\n', fontsize=18)

        ax3.imshow(mask_pred[i]*250 , extent=extent, vmin=0, vmax=1)
        ax3.set_title('Predicted \nMask\n$\\mathbf{(RTh = 0.375)}$', fontsize=18)

        ax4.imshow(mask_pred2[i]*250 , extent=extent, vmin=0, vmax=1)
        ax4.set_title('Predicted \nMask\n$\\mathbf{(RTh = 0.5)}$', fontsize=18)

        ax5.imshow(mask_pred3[i] *250, extent=extent, vmin=0, vmax=1)
        ax5.set_title('Predicted \nMask\n$\\mathbf{(RTh = 0.625)}$', fontsize=18)
        #
        # ax6.imshow(mask_pred_conf[i], extent=extent, vmin=0, vmax=1,cmap='seismic')
        # ax6.set_title('Prediction \nConfidence \nMap', fontsize=18)
        # for ax, label in zip([ax1, ax2, ax3, ax4, ax5,ax6], ['(a)', '(b)', '(c)', '(d)', '(e)','(f)']):
        for ax, label in zip([ax1, ax2, ax3, ax4, ax5], ['(a)', '(b)', '(c)', '(d)', '(e)']):
            ax.text(
                0.00, 1.0,  # Top-left corner in axes coordinates
                label,
                transform=ax.transAxes,
                fontsize=18,
                fontweight='bold',
                va='top',
                ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, boxstyle='round,pad=0.2')
            )
        # Tick formatting
        # for ax in [ax1, ax2, ax3, ax4, ax5,ax6]:
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.tick_params(axis='x', labelsize=14, rotation=45)
            ax.tick_params(axis='y', labelsize=14
                           )
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

        # Save and show
        fig.savefig('/Users/galidek/Desktop/paper_figs/full_intfs/' + item + '_june25', dpi=1000)
        plt.show()


def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg

parser = argparse.ArgumentParser(description='evaluate full intf output')
# EDIT 2026-07-29: default followed test_full_intf.py out of 'pred_outputs2/'. CHANGELOG.md #14
parser.add_argument('--path',  type=str, default='outputs/predictions/<model>/<job>_<timestamp>/', help='reconstructed interferogram path (a test_full_intf.py output dir)')
parser.add_argument('--th',  type=float, default=0)
parser.add_argument('--k_prevs',  type=int, default=0)
parser.add_argument('--aligned_patches',  action='store_true')
parser.add_argument('--skip_ol_metrics',  action='store_true')

parser.add_argument('--plot',  type=str, default='False')
args = parser.parse_args()
path=args.path

plot=str2bool(args.plot)
#plt.rcParams['backend'] = 'Qt5Agg'
files_list = os.listdir(path)
intf_list = [file[:17] for file in files_list if (file[-3:] !='log' and os.path.isfile(args.path + file))]
unique_intf_list = list(set(intf_list))
#recheck nonz num
now = datetime.now()  # ← no dot before ()
nowstr = now.strftime("%m%d%H%M")
with open(_bootstrap.asset('intf_coord.json'), 'r') as json_file:
    coord_dict = json.load(json_file)

for intf in unique_intf_list:
    print (intf)
    print(coord_dict[intf]['north'])
    print(coord_dict[intf]['nonz_num'])
    print('---------------------')


ol_mean_recalls,ol_mean_precisions = [],[]
images,preds,gts = [],[],[]
recalls,precisions,recalls2,precisions2,recalls3,precisions3 = [],[],[],[],[],[]
for item in unique_intf_list:
    # #     #
    #
    # if '20210623' not in item:
    #     continue

    # # if '0326' not in item and '1210' not in item:
    # if ( '2023' in item or '2022' in item):
    #     continue
    print(item)
    intf_info = get_intf_coords(item)
    if args.aligned_patches:
        if intf_info[11] == 'North':
            intf_info = (35.3, 31.79) + intf_info[2:]
        else:
            intf_info = (35.25, 31.44) + intf_info[2:]

    dx = intf_info[2]
    y0 = intf_info[1]
    dy = intf_info[3]
    x0 = intf_info[0]
    frame = intf_info[11]
    print(f'{frame} Frame')
    if args.aligned_patches:
        if frame == 'North':
            x0 = x0 + 3000*dx
        else:
            x0 = x0 + 3000*dx
    else:
        x0 = x0+3000*dx

    image = np.load( path + item +'_image.npy', allow_pickle=True)
    mask_pred_conf = np.load(path + item+'_pred.npy', allow_pickle=True)
    mask_true = np.load(path + item + '_gt.npy', allow_pickle=True)
    mask_pred = np.where(mask_pred_conf > 0.125, 1, 0)
    mask_pred2 = np.where(mask_pred_conf > 0.25, 1, 0)
    mask_pred3 = np.where(mask_pred_conf > 0.5,1,0)
    if image.ndim ==3:
        im_ydim = 1
    else:
        im_ydim = 0

    if frame =='South':
         half_size_y = image.shape[im_ydim]//2
         if im_ydim==1:
             image = image[:, :half_size_y, :]
         else:
            image = image[:half_size_y, :]

         mask_pred_conf = mask_pred_conf[:half_size_y,:]
         mask_pred = mask_pred[:half_size_y,:]
         mask_pred2 = mask_pred2[:half_size_y,:]
         mask_pred3 = mask_pred3[:half_size_y,:]
         mask_true = mask_true[:half_size_y,]

    # else:
    #    size_y = image.shape[im_ydim]
    #    fifth_size_y = image.shape[im_ydim]//5
    #    half_size_x = image.shape[1+im_ydim]//2
    #    y0 = y0-fifth_size_y*dy
       # if im_ydim ==1:
       #     image = image[:,:size_y-1000, :half_size_x]
       # else:
       #     image = image[:size_y-1000, :half_size_x]
       # mask_pred_conf = mask_pred_conf[:size_y-1000, :half_size_x]
       # mask_pred = mask_pred[:size_y-1000, :half_size_x]
       # mask_pred2 = mask_pred2[:size_y-1000, :half_size_x]
       # mask_pred3 = mask_pred3[:size_y-1000, :half_size_x]
       # mask_true = mask_true[:size_y-1000, :half_size_x]


    if len(mask_pred.shape) <3:
        mask_pred_conf = np.expand_dims(mask_pred_conf,axis=0)
        mask_pred = np.expand_dims(mask_pred, axis=0)
        mask_pred2=np.expand_dims(mask_pred2,axis=0)
        mask_pred3 = np.expand_dims(mask_pred3,axis=0)
        mask_true = np.expand_dims(mask_true, axis=0)
        # image = np.expand_dims(image, axis=0)


    images.append(image)
    preds.append(mask_pred)
    gts.append(mask_true)

    if True:
        # precision = precision1(mask_true,mask_pred)
        # recall = recall1(mask_true,mask_pred)
        print('intf ' + item + ':')
        # print(precision,recall)

        if args.skip_ol_metrics:
            r, p,n,feature_lists = 0,0,0,[]
            r2, p2, n, feature_lists = 0, 0, 0, []
            r3, p3, n, feature_lists = 0, 0, 0, []

        else:
            r, p, n, feature_lists = object_level_evaluate(mask_true, mask_pred, image, features=[], th=0.7,buffer=5)
            r2, p2, n, _ = object_level_evaluate(mask_true, mask_pred2, image, features=[], th=0.7, buffer=5)
            r3, p3, n, _ = object_level_evaluate(mask_true, mask_pred3, image, features=[], th=0.7, buffer=5)


        recalls.append(r)
        precisions.append(p)
        recalls2.append(r2)
        precisions2.append(p2)
        recalls3.append(r3)
        precisions3.append(p3)


    plot = True
    if plot:
        plot_full_intfs_preds_vprevs(image,mask_true,mask_pred,mask_pred2,mask_pred3,mask_pred_conf,item)


recall_np = np.array(recalls)
precision_np = np.array(precisions)
recall2_np = np.array(recalls2)
precision2_np = np.array(precisions2)
recall3_np = np.array(recalls3)
precision3_np = np.array(precisions3)


mean_Recall = np.mean(recall_np)
mean_Precision = np.mean(precision_np)
mean_Recall2 = np.mean(recall2_np)
mean_Precision2 = np.mean(precision2_np)
mean_Recall3 = np.mean(recall3_np)
mean_Precision3 = np.mean(precision3_np)

print('Mean Recall: ', mean_Recall)
print('Mean Precision: ', mean_Precision)
print('Mean Recall 2: ', mean_Recall2)
print('Mean Precision 2: ', mean_Precision2)
print('Mean Recall 3: ', mean_Recall3)
print('Mean Precision 3: ', mean_Precision3)
ol_mean_recalls.append(mean_Recall)
ol_mean_precisions.append(mean_Precision)
olm_results = {
    "recalls": ol_mean_recalls,
    "precisions": ol_mean_precisions
}
with open("olm_results_with22_23.json", "w") as f:
    json.dump(olm_results,f)