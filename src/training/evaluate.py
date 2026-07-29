# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import torch
import torch.nn.functional as F
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.affinity import translate
import matplotlib.pyplot as plt


from dice_score import multiclass_dice_coeff, dice_coeff
from device_utils import memory_format_for  # EDIT 2026-07-27, CHANGELOG.md #1

import rasterio
from rasterio.features import shapes

import rasterio.features
from shapely.geometry import shape

import geopandas as gpd
from unet import *
import logging
from rasterio.features import geometry_mask
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from rasterio.features import geometry_mask
from affine import Affine
from shapely.affinity import translate
from shapely.geometry import box
from rasterio.features import geometry_mask
from skimage.util import img_as_float
from numpy.fft import fft2, fftshift
from skimage.restoration import unwrap_phase
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

def _iter_polygons(geom):
    """
    Yield only Polygon objects (area > 0) from any Shapely geometry:
    Polygon, MultiPolygon, GeometryCollection, etc.
    """
    if geom.is_empty:
        return

    if isinstance(geom, Polygon):
        if geom.area > 0:
            yield geom

    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if g.area > 0:
                yield g

    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            # recurse: some children may be polygons, others points/lines
            yield from _iter_polygons(g)

    # for Point, LineString, MultiPoint, etc. → yield nothing


# ------------------------------------------------------
# INTERSECTION between GT and *BUFFERED* predictions
# ------------------------------------------------------

plt.rcParams['backend'] = 'Qt5Agg'
def draw_polygon_on_image(image, polygon, color='red', linewidth=2):
    # Display the image
    plt.imshow(image, cmap='jet')  # or cmap='gray' if grayscale
    plt.axis('off')

    # Get x, y coordinates of the polygon exterior
    x, y = polygon.exterior.xy
    plt.plot(x, y, color=color, linewidth=linewidth)

    plt.show()



def compute_feature(p,feature,image):


    if feature == 'area':
        return p.area
    elif feature == 'perimeter':
        return p.length
    elif feature == 'solidity':
        return p.area / p.convex_hull.area
    elif feature == 'roundness':
        return 4 * np.pi * p.area / (p.length ** 2)
    elif feature.startswith('phase'):
    # Create a mask for the polygon
        transform = Affine.identity()  # assuming polygon and image are in pixel coordinates
        mask = geometry_mask([p], transform=transform, invert=True, out_shape=image[0].shape)
        masked_values =  image * mask[None, :, :]
        minx, miny, maxx, maxy = map(int, p.bounds)

        pad = 5

        minx = max(minx - pad, 0)

        miny = max(miny - pad, 0)

        maxx = min(maxx + pad, image.shape[2])

        maxy = min(maxy + pad, image.shape[1])

        sub_image = image[:,miny:maxy, minx:maxx]

        # Shift polygon to local coordinates of sub-image

        shifted_p = translate(p, xoff=-minx, yoff=-miny)

        # Create mask for shifted polygon

        transform = Affine.identity()

        submask = geometry_mask([shifted_p], transform=transform, invert=True, out_shape=sub_image[0].shape)

        if feature == 'phase_std':
            vals = np.std(sub_image[:, submask], axis=1)  # shape: (T,)
            return float(np.nanmean(vals))

        elif feature == 'phase_gradient':
            # Create a mask for the polygon
            # transform = Affine.identity()  # Assuming pixel-based polygon
            # mask = geometry_mask([p], transform=transform, invert=True, out_shape=image.shape)
            #
            # # Compute gradients along x and y
            # gy, gx = np.gradient(image)
            # grad_magnitude = np.sqrt(gx ** 2 + gy ** 2)
            #
            # # Masked gradient magnitude
            # masked_grad = grad_magnitude[mask]
            #
            # #return np.mean(masked_grad)  # or np.median(masked_grad), np.std(...), etc.
            return compute_mean_phase_gradient(sub_image, submask,unwrap=True)
        elif feature == 'phase_fft':

            # Get bounding box of polygon with small padding


            fft_ratio = compute_fft_noise_ratio(sub_image, submask)
            #plot_fft_ratio_figure(sub_image, submask)


            # if p.area <50:
            #     plt.imshow(image, cmap='jet')
            #     x, y = p.exterior.xy
            #     minx, miny, maxx, maxy =p.bounds
            #     plt.plot(x, y,color = 'black',linewidth=10)
            #     plt.title('area: ' + str(p.area)+ ' minx: ' + str(minx) + ' miny: ' + str(miny))
            #     plt.show()
            #


            return fft_ratio
        elif feature == 'phase_radial_symmetry':
            return radial_symmetry_score(sub_image, submask)


def radial_symmetry_score(image, mask, center=None, band_width=1):


    y, x = np.indices(image.shape)
    if center is None:
        y_mask, x_mask = np.where(mask)
        center_y = np.mean(y_mask)
        center_x = np.mean(x_mask)
    else:
        center_y, center_x = center

    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r_int = r.astype(int)

    scores = []
    max_r = r_int[mask].max()

    for ri in range(1, max_r):
        ring = (r_int == ri) & mask
        if np.sum(ring) < 10:
            continue
        values = image[ring]
        std = np.std(values)
        mean = np.mean(values)
        if mean != 0:
            scores.append(std / mean)  # coefficient of variation

    if len(scores) == 0:
        return np.nan
    return 1 - np.mean(scores)  #

def plot_fft_ratio_figure(image, mask, pixel_size=3.0, freq_th=0.05, sigma=2):
    ratio, freqs, energy_normalized = compute_fft_noise_ratio(
        image, mask, pixel_size=pixel_size, freq_th=freq_th, sigma=sigma, return_spectrum=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original image
    im0 = axes[0].imshow(image, cmap='jet')
    axes[0].set_title("Phase Image")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Mask
    im1 = axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask (Subsidence Area)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Radial FFT energy spectrum
    axes[2].plot(freqs, energy_normalized, label="Normalized FFT Energy")
    axes[2].axvline(freq_th, color='red', linestyle='--', label=f"Threshold = {freq_th:.2f}")
    axes[2].set_xlabel("Spatial Frequency (cycles/m)")
    axes[2].set_ylabel("Normalized Energy")
    axes[2].grid(True)
    axes[2].legend()
    axes[2].set_title("Radial FFT Spectrum")

    # Overall title
    fig.suptitle(f"High/Low Frequency Energy Ratio = {ratio:.3f}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.fft import fft2, fftshift

def compute_fft_noise_ratio(image, mask, pixel_size=3.0, freq_th=0.05, sigma=2, return_spectrum=False):
    h, w = image.shape
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=sigma)
    windowed_image = image * smoothed_mask
    fft_img = fftshift(fft2(windowed_image))
    fft_energy = np.abs(fft_img) ** 2

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_int = r.astype(int)

    r_max = r_int.max()
    radial_energy = np.zeros(r_max + 1)

    for i in range(r_max + 1):
        ring = (r_int == i)
        if np.any(ring):
            radial_energy[i] = fft_energy[ring].sum()

    energy_normalized = radial_energy / (radial_energy.sum() + 1e-12)
    freqs = np.arange(len(energy_normalized)) / (np.sqrt(h**2 + w**2) * pixel_size)

    high_band = freqs > freq_th
    low_band = freqs <= freq_th
    high_energy = energy_normalized[high_band].sum()
    low_energy = energy_normalized[low_band].sum()
    ratio = high_energy / (low_energy + 1e-12)

    if return_spectrum:
        return ratio, freqs, energy_normalized
    else:
        return ratio
def compute_mean_phase_gradient(image, mask, unwrap=False):

    if image[0].shape != mask.shape:
        raise ValueError("Image and mask must have the same shape.")


    # image: (T,H,W), mask: (H,W) bool or 0/1
    T, H, W = image.shape
    mask_bool = mask.astype(bool)
    assert mask_bool.shape == (H, W), "mask must be (H,W)"

    out_sum = 0.0
    valid_frames = 0

    for i in range(T):
        frame = image[i]
        if unwrap:
            frame_unwrapped_rad = unwrap_phase(2 * np.pi * frame)
            frame = frame_unwrapped_rad / (2 * np.pi)

        # gradients (gy, gx) for 2D frame
        gy, gx = np.gradient(frame)
        grad_mag = np.hypot(gx, gy)  # sqrt(gx**2 + gy**2)

        masked = grad_mag[mask_bool]
        if masked.size == 0:
            continue
        out_sum += float(np.mean(masked))  # cast to Python float
        valid_frames += 1

    if valid_frames == 0:
        return np.nan
    return out_sum / valid_frames
def compute_entropy_region(image, mask, radius=2):

    """Compute average entropy inside a masked region."""

    from skimage.filters.rank import entropy

    from skimage.morphology import disk

    from skimage.util import img_as_ubyte

    # Normalize before converting to 8-bit

    normed = (image - image.min()) / (image.ptp() + 1e-8)

    image_ubyte = img_as_ubyte(normed)

    entropy_img = entropy(image_ubyte, disk(radius))

    return entropy_img[mask].mean()

def compute_roundness(polygon: Polygon):
    area = polygon.area
    perimeter = polygon.length
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)
def object_level_evaluate( gt, pred, image = None, features = ['area', 'roundness','phase_std','phase_gradient'],epsilon = 1e-7,th=0.7,buffer=5,is_local=False,plot=False):
    import numpy as np
    feature_lists = {feature: [[],[]] for feature in features}
    if image.ndim==2:
        image = np.expand_dims(image, axis=0)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    intersect_recall, intersect_precision = [], []
    patch_gt_areas, patch_pred_areas = [], []
    for i in range(gt.shape[0]):
        transform = Affine.identity()  # Create an identity transform

        # Convert binary mask to Shapely polygons
        gt_polygons,pred_polygons = [], []

        ###create polygons
        for geom, val in rasterio.features.shapes(gt[i], transform=transform):
            if val > 0:  # Assuming binary mask has values 0 and 1
                gt_polygons.append(shape(geom))

        for geom, val in rasterio.features.shapes(pred[i], transform=transform):
            if val > 0:  # Assuming binary mask has values 0 and 1
                pred_polygons.append(shape(geom))

        ###start calculation for patch
        intersection_area= 0
        pred_intersection_area= 0
        total_pred_area = sum(pp.area for pp in pred_polygons)
        total_gt_area = sum(pgt.area for pgt in gt_polygons)

        patch_gt_areas.append(total_gt_area)
        patch_pred_areas.append(total_pred_area)
        detected_gt_areas = []
        undetected_gt_areas = []
        detected_gt_feat = []
        undetected_gt_feat = []
        from shapely.ops import unary_union
        for pgt in gt_polygons:
            intersection = 0
            orig_intersection_area = 0
            # for pp in pred_polygons:
            #     buffered_pp = pp.buffer(buffer)
            #     intersection += pgt.intersection(buffered_pp).area/pgt.area
            #     orig_intersection_area +=pgt.intersection(pp).area

            buffered_union = unary_union([pp.buffer(buffer) for pp in pred_polygons])

            # union of original (unbuffered) preds
            orig_union = unary_union(pred_polygons)

            # intersection with GT polygon
            inter_buf_area = pgt.intersection(buffered_union).area
            inter_orig_area = pgt.intersection(orig_union).area

            # your metrics
            intersection = inter_buf_area / pgt.area
            orig_intersection_area = inter_orig_area
            if intersection > th:
                intersection_area += pgt.area
                pred_intersection_area += orig_intersection_area
                if True:
                    for feat in feature_lists:
                        feature_lists[feat][0].append(compute_feature(pgt,feat,image))

            else:
                if True:
                    for feat in feature_lists:
                        feature_lists[feat][1].append(compute_feature(pgt,feat,image))


        fp_area = 0
        for pp in pred_polygons:
            pp_intersection_area = 0
            # for pgt in gt_polygons:
            #     pp_intersection_area += pgt.buffer(buffer).intersection(pp).area
            buffered_union = unary_union([pgt.buffer(buffer) for pgt in gt_polygons])
            inter_buf_area = pp.intersection(buffered_union).area

            #
            if inter_buf_area / pp.area < th:
                fp_area += (pp.area - pp_intersection_area)

        ol_intersection_recall = round(intersection_area / (total_gt_area + epsilon),2)
        ol_intersection_precision = round(pred_intersection_area / (total_pred_area + epsilon),2)
        ol_intersection_precision = round(intersection_area / (intersection_area + fp_area+ epsilon),2)

        if ol_intersection_precision  > 1:
            ol_intersection_precision =1.0

        intersect_recall.append(ol_intersection_recall)
        intersect_precision.append(ol_intersection_precision)

        if plot and is_local:  # and (ol_intersection_recall < 0.7 and ol_intersection_precision > 0.8):

            from math import ceil
            from skimage import measure
            from shapely.geometry import Polygon, MultiPolygon
            from shapely.ops import unary_union

            # --- helper: mask -> shapely polygons in pixel coords ---
            def mask_to_polygons(mask2d, level=0.5, min_points=5):
                contours = measure.find_contours(mask2d.astype(float), level=level)
                polys = []
                for c in contours:
                    if c.shape[0] < min_points:
                        continue
                    # c[:,0] = row (y), c[:,1] = col (x)
                    poly = Polygon(np.column_stack((c[:, 1], c[:, 0])))
                    if poly.is_valid and poly.area > 0:
                        polys.append(poly)
                return polys

            # helper: yield only Polygon objects (skip points/lines)
            def _iter_polygons(geom):
                if geom.is_empty:
                    return
                if isinstance(geom, Polygon):
                    yield geom
                elif isinstance(geom, MultiPolygon):
                    for g in geom.geoms:
                        if isinstance(g, Polygon):
                            yield g

            # --- prepare frames ---
            if image[i].ndim == 3:
                T, H, W = image[i].shape
                cur = image[i][0]
                prev_list = [image[i][k] for k in range(1, T)]
            else:
                cur = image[i]
                H, W = cur.shape
                prev_list = []

            # panels: all prevs + current + GT + Prediction + Polygon diagnostics
            n_prev = len(prev_list)
            n_panels = n_prev + 4  # prevs + cur + GT + pred + diag

            max_cols = 5
            ncols = min(max_cols, n_panels)
            nrows = ceil(n_panels / ncols)

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(3.4 * ncols, 3.4 * 2 * nrows),
                sharex=True, sharey=True
            )
            axes = np.atleast_1d(axes).ravel()

            idx = 0

            # 1) previous frames
            for j, prev in enumerate(prev_list):
                ax = axes[idx]
                ax.imshow(prev, cmap='jet')
                ax.set_title(f'Phase Image', fontsize=11)
                ax.set_xlim(0, W - 1)
                ax.set_ylim(H - 1, 0)
                ax.set_aspect('equal')
                idx += 1

            # 2) current frame
            # 2) current frame
            ax_cur = axes[idx]
            ax_cur.imshow(cur, cmap='jet')
            ax_cur.set_title('Input Patch \n', fontsize=16)
            ax_cur.set_xlim(0, W - 1)
            ax_cur.set_ylim(H - 1, 0)
            ax_cur.set_aspect('equal')

            # ---- PAINT INPUT PIXELS WITH VALUE 0.5 IN BLACK ----
            mask_half = np.isclose(cur, 0.5)

            if mask_half.any():
                # create an overlay that is black where cur == 0.5, transparent elsewhere
                overlay = np.zeros_like(cur, dtype=float)  # 0 = black in 'gray' cmap
                # mask everything that is NOT exactly 0.5
                overlay_ma = np.ma.masked_where(~mask_half, overlay)

                ax_cur.imshow(
                    overlay_ma,
                    cmap='gray',  # 0 -> black
                    vmin=0, vmax=1,
                    alpha=1.0,  # fully black on those pixels
                )

            idx += 1

            # 3) GT mask panel
            ax_gt = axes[idx]
            ax_gt.imshow(gt[i])
            ax_gt.set_title('Ground Truth Mask', fontsize=16)
            ax_gt.set_xlim(0, W - 1)
            ax_gt.set_ylim(H - 1, 0)
            ax_gt.set_aspect('equal')
            idx += 1

            # 4) prediction panel (mask only)
            ax_pred = axes[idx]
            ax_pred.imshow(pred[i])
            ax_pred.set_title('Predicted Mask', fontsize=16)
            ax_pred.set_xlim(0, W - 1)
            ax_pred.set_ylim(H - 1, 0)
            ax_pred.set_aspect('equal')
            idx += 1

            # --- polygons from masks ---
            gt_polys = mask_to_polygons(gt[i] > 0.5)
            pred_polys = mask_to_polygons(pred[i] > 0.5)

            buffer_px = buffer  # buffer radius in PIXELS

            # 5) polygon-only diagnostics panel
            ax_poly = axes[idx]
            ax_poly.set_title('Intersection', fontsize=16)
            ax_poly.set_facecolor('lightgray')  # background color (change if you want)
            ax_poly.set_xlim(0, W - 1)
            ax_poly.set_ylim(H - 1, 0)
            ax_poly.set_aspect('equal')
            idx += 1

            # --- ORIGINAL predicted polygons (black) on polygon panel ---
            for poly in pred_polys:
                for g in _iter_polygons(poly):
                    x_p, y_p = g.exterior.xy

                    # on the current-frame panel (first main panel)
                    # ax_cur.plot(x_p, y_p, color='black', linewidth=3, label='_nolegend_')

                    # on the polygon diagnostics panel
                    ax_poly.plot(x_p, y_p, color='black', linewidth=3, label='_nolegend_')

            # --- GT polygons on BOTH current frame and polygon panel (white) ---
            for poly in gt_polys:
                for g in _iter_polygons(poly):
                    x_g, y_g = g.exterior.xy
                    # ax_cur.plot(x_g, y_g, color='white', linewidth=3, label='_nolegend_')
                    ax_poly.plot(x_g, y_g, color='white', linewidth=3, label='_nolegend_')

            # --- BUFFERED predicted polygons (cyan) on pred & polygon panels ---
            buffered_preds = []
            for poly in pred_polys:
                buff = poly.buffer(buffer_px)
                if buff.is_empty:
                    continue
                for g in _iter_polygons(buff):
                    buffered_preds.append(g)
                    x_b, y_b = g.exterior.xy

                    # draw on prediction panel
                    ax_pred.plot(x_b, y_b, color='cyan', linewidth=3, label='_nolegend_',linestyle = 'dashed')
                    # draw on polygon panel
                    ax_poly.plot(x_b, y_b, color='cyan', linewidth=3, label='_nolegend_',linestyle = 'dashed')

            # --- INTERSECTION between GT and BUFFERED predictions (filled) ---
            if buffered_preds and gt_polys:
                pred_union = unary_union(buffered_preds)

                for poly_gt in gt_polys:

                    # --- 3A. INTERSECTION ---
                    inter = poly_gt.intersection(pred_union)

                    if not inter.is_empty:
                        for g in _iter_polygons(inter):  # safe helper you already have
                            x_i, y_i = g.exterior.xy
                            ax_poly.fill(
                                x_i, y_i,
                                facecolor='blue',  # intersection color
                                edgecolor='none',
                                alpha=0.3,
                            )

                    # --- 3B. NON-INTERSECTING PORTION (GT minus intersection) ---
                    diff = poly_gt.difference(pred_union)

                    if not diff.is_empty:
                        for g in _iter_polygons(diff):
                            x_d, y_d = g.exterior.xy
                            ax_poly.fill(
                                x_d, y_d,
                                facecolor='#CC0000',  # NON-intersection color (change here)
                                edgecolor='none',
                                alpha=0.3
                            )

            # ticks (for all used panels)
            for ax in axes[:idx]:
                ax.set_xticks([0, W // 2, W - 1])
                ax.set_yticks([0, H // 2, H - 1])
                ax.tick_params(axis='both', labelsize=14)

            # hide unused axes, if any
            for ax in axes[idx:]:
                ax.axis('off')

            txt = (r'$\mathrm{Recall}_{\mathrm{OL}}$: ' f'{ol_intersection_recall:.3f}\n'
                   r'$\mathrm{Precision}_{\mathrm{OL}}$: ' f'{ol_intersection_precision:.3f}')

            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.85])
            fig.text(0.80, 0.96, txt, transform=fig.transFigure,
                     ha='left', va='top', fontsize=14,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

            plt.show()

    batch_gt_area = np.sum(np.array(patch_gt_areas))
    ol_batch_recall = round(np.sum(np.array(intersect_recall)*np.array(patch_gt_areas))/batch_gt_area,2)
    ol_batch_precision = round(np.sum(np.array(intersect_precision)*np.array(patch_gt_areas))/batch_gt_area,2)
    # ol_batch_precision = round(sum(intersect_precision)/len(intersect_precision),2)
    print('ol batch recall:',ol_batch_recall)
    print('ol batch precision:',ol_batch_precision)

    return ol_batch_recall, ol_batch_precision,batch_gt_area, feature_lists
def precision1_weighted(gt,pred,epsilon=1e-7):
    gt_sum_per_batch = gt.sum(axis=(1, 2)).astype(float)

    # Normalize weights to avoid scaling issues
    weights = gt_sum_per_batch / (gt_sum_per_batch.sum() + epsilon)

    # Calculate true positives (TP) and false positives (FP) for each batch
    TP = np.logical_and(gt, pred).sum(axis=(1, 2)).astype(float)
    FP = np.logical_and(~gt, pred).sum(axis=(1, 2)).astype(float)

    # Weighted precision for the entire batch
    weighted_TP = (weights * TP).sum()
    weighted_FP = (weights * FP).sum()

    precision = weighted_TP / (weighted_TP + weighted_FP + epsilon)
    return precision


def precision1(gt, pred, epsilon=1e-7):
    # Convert arrays to boolean arrays
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # Calculate true positives (TP), false positives (FP) for the entire batch
    TP = np.logical_and(gt, pred).sum().astype(float)
    FP = np.logical_and(~gt, pred).sum().astype(float)

    # Calculate precision for the entire batch
    precision = TP / (TP + FP + epsilon)

    return precision
def recall1(gt, pred, epsilon=1e-7):
    # Convert arrays to boolean arrays
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # Calculate true positives (TP), false negatives (FN) for the entire batch
    TP = np.logical_and(gt, pred).sum().astype(float)
    FN = np.logical_and(gt, ~pred).sum().astype(float)

    # Calculate recall for the entire batch
    recall = TP / (TP + FN + epsilon)

    return recall
def calc_precision_recall (y_pred, y_true):
    precision = np.sum(y_pred * y_true)/np.sum(y_pred) #tp/tp+tn
    recall = np.sum(y_pred * y_true)/np.sum(y_true) #tp/tp+fn
    return precision, recall

@torch.inference_mode()
# EDIT 2026-07-27: added the optional `metrics_out` dict. CHANGELOG.md #3
# EDIT 2026-07-27: added `loss_fn` and `samples_out`. CHANGELOG.md #5
def evaluate(net, dataloader, device, amp ,is_local,out_path,epoch,mode = 'val',save_val = False,net_aux=None,th=0.7,buffer=5,plot=False,metrics_out=None,loss_fn=None,samples_out=None):
    """`metrics_out`: pass a dict to receive pixel-level precision/recall/TP counts.

    Opt-in and non-breaking — the return value is unchanged, so existing callers
    (test.py, test_full_intf.py, train_sinkholes_unet.py) are unaffected. Only
    binary (n_classes == 1) runs populate it. Counts are accumulated globally
    rather than averaged per batch, which avoids the division-by-zero that
    `calc_precision_recall` hits on an all-negative batch.

    `loss_fn`: optional `f(logits, images, mask_true) -> scalar tensor`, called
    under `no_grad` on the *pre-threshold* logits and averaged over batches into
    `metrics_out['loss']`. The training script passes the very function it
    optimises, so `val/loss` is directly comparable with `train/loss` and cannot
    drift away from it.

    `samples_out`: optional dict; set `{'n': k}` to receive `image`, `gt` and
    `prob` arrays `(k, H, W)` for the first k validation patches, for rendering
    a qualitative grid. Patches whose ground truth has positive pixels are
    preferred — an all-background sample shows nothing useful — and the pick is
    deterministic across epochs as long as the loader is not shuffled.
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    precision = 0
    recall = 0
    ol_precision = 0
    ol_recall = 0
    # EDIT 2026-07-27: global confusion counts for `metrics_out`. CHANGELOG.md #3
    _tp = _fp = _fn = 0.0
    # EDIT 2026-07-27: validation loss and qualitative samples. CHANGELOG.md #5
    _loss_sum, _loss_n = 0.0, 0
    _n_samples = int(samples_out.get('n', 4)) if samples_out is not None else 0
    _samp_pos, _samp_any = [], []  # positive-GT picks, then any-GT fallback

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        image_batches,true_mask_batches, pred_batches, b_gts = [],[],[],[]
        for bn, batch in tqdm(enumerate(dataloader), total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            # EDIT 2026-07-27: memory format chosen per device so validation matches the
            # model's layout on MPS (inference with channels_last does work there, but
            # mixing layouts forces a per-batch copy). See CHANGELOG.md #1.
            image = image.to(device=device, dtype=torch.float32, memory_format=memory_format_for(device))
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            if net_aux is not None:
                mask_pred1 = net_aux(image)

            if net.n_classes == 1:
                mask_true = mask_true.unsqueeze(1)
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # EDIT 2026-07-27: keep the logits and the probability map before
                # thresholding — the loss needs logits, the sample grid needs
                # probabilities, and both used to be discarded here. CHANGELOG.md #5
                logits = mask_pred
                mask_prob = F.sigmoid(logits)
                if loss_fn is not None:
                    with torch.no_grad():  # never build a graph for a reported number
                        _l = float(loss_fn(logits, image, mask_true))
                    if np.isfinite(_l):
                        _loss_sum += _l
                        _loss_n += 1
                mask_pred = (mask_prob > 0.5).float()
                if net_aux is not None:
                    mask_pred1 = (F.sigmoid(mask_pred1) > 0.5).float()

                mask_pred_np = mask_pred.squeeze(1).cpu().detach().numpy() #squeeze to (b,H,W) when saving npy
                mask_true_np = mask_true.squeeze(1).cpu().detach().numpy()
                image_np = image.squeeze(1).cpu().detach().numpy()
                if net_aux is not None:
                    mask_pred1_np = mask_pred1.squeeze(1).cpu().detach().numpy()

                # EDIT 2026-07-27: collect a handful of patches for the qualitative
                # grid. Bounded by `n`, so this stops costing anything after the
                # first few batches. CHANGELOG.md #5
                if _n_samples and len(_samp_pos) < _n_samples:
                    # `image` is (B, C, H, W) with C = T (+T validity channels when
                    # treat_nodata_regions). Channel T-1 is the current
                    # interferogram — `sinkholes_data_loading.py:176` orders the
                    # stack prevs…present — so the caller passes T-1 and we fall
                    # back to the last channel, which is the same thing whenever
                    # no validity channels are appended.
                    _ch = samples_out.get('channel')
                    _ch = image.shape[1] - 1 if _ch is None else int(_ch)
                    _ch = min(max(_ch, 0), image.shape[1] - 1)
                    _img_b = image[:, _ch].cpu().detach().numpy()
                    _prob_b = mask_prob.squeeze(1).cpu().detach().numpy()
                    for _i in range(_img_b.shape[0]):
                        _triple = (_img_b[_i], mask_true_np[_i], _prob_b[_i])
                        if mask_true_np[_i].sum() > 0:
                            if len(_samp_pos) < _n_samples:
                                _samp_pos.append(_triple)
                        elif len(_samp_any) < _n_samples:
                            _samp_any.append(_triple)

                epoch_suf = '_epoch' + str(epoch)
                if epoch % 2 == 0:
                    pred_batches.append(mask_pred_np)
                if epoch == 1:
                    image_batches.append(image_np)
                    true_mask_batches.append(mask_true_np)


                # compute the Dice score

                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # EDIT 2026-07-27: cheap global confusion counts, only when a caller
                # asked for them. No polygonization here, so validation stays fast.
                # CHANGELOG.md #3
                if metrics_out is not None:
                    _p = mask_pred_np.astype(bool)
                    _g = mask_true_np.astype(bool)
                    _tp += float(np.logical_and(_g, _p).sum())
                    _fp += float(np.logical_and(~_g, _p).sum())
                    _fn += float(np.logical_and(_g, ~_p).sum())
                if mode == 'test':
                    batch_precision, batch_recall = calc_precision_recall(mask_pred_np, mask_true_np)
                    olr, olp, b_gt_a,_ = object_level_evaluate(mask_true_np, mask_pred_np,image_np,features=[],th=th,buffer=buffer,is_local=is_local,plot=plot)
                    precision += batch_precision
                    recall += batch_recall
                    if not np.isnan(olp) and not np.isnan(olr):
                        ol_precision += olp * b_gt_a
                        ol_recall +=olr * b_gt_a
                        b_gts.append(b_gt_a)

            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)


        if epoch == 1 and mode == 'val' and save_val:
            image = np.concatenate(image_batches)
            mask_true = np.concatenate(true_mask_batches)
            np.save(out_path + '/image_valid_test', image)
            np.save(out_path + '/mask_true_valid', mask_true)

        if epoch % 2 == 0  and save_val:
            mask_pred = np.concatenate(pred_batches)
            np.save(out_path + '/mask_pred_valid'+epoch_suf,mask_pred)
    net.train()
    mean_dice_score = dice_score / max(num_val_batches, 1)
    # EDIT 2026-07-27: fill the caller's dict. CHANGELOG.md #3
    if metrics_out is not None:
        metrics_out['tp'], metrics_out['fp'], metrics_out['fn'] = _tp, _fp, _fn
        metrics_out['precision'] = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        metrics_out['recall'] = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        metrics_out['n_batches'] = num_val_batches
        # EDIT 2026-07-27: IoU and F1 over the same pooled counts. Note that this
        # F1 is *not* the returned Dice: F1 pools every pixel in the split, while
        # `mean_dice_score` averages per batch, so a patch with 3 positive pixels
        # weighs as much as one with 3000. Both are correct and they will differ;
        # PROJECT_SUMMARY/inspect_run call the same split macro vs micro.
        # CHANGELOG.md #5
        _den_iou = _tp + _fp + _fn
        metrics_out['iou'] = _tp / _den_iou if _den_iou else 0.0
        metrics_out['f1'] = 2 * _tp / (2 * _tp + _fp + _fn) if _den_iou else 0.0
        if _loss_n:
            metrics_out['loss'] = _loss_sum / _loss_n

    # EDIT 2026-07-27: hand back the collected samples, positives first, padded
    # with all-background patches only if there were not enough. CHANGELOG.md #5
    if samples_out is not None:
        _picked = (_samp_pos + _samp_any)[:_n_samples]
        if _picked:
            samples_out['image'] = np.stack([p[0] for p in _picked])
            samples_out['gt'] = np.stack([p[1] for p in _picked])
            samples_out['prob'] = np.stack([p[2] for p in _picked])
            samples_out['n_positive'] = len(_samp_pos)
    if mode == 'test':
        mean_p = round(precision / num_val_batches,2)
        mean_r = round(recall / num_val_batches,2)
        mean_ol_p = round(ol_precision/sum(b_gts),2)
        mean_ol_r = round(ol_recall / sum(b_gts),2)

        print('mean dice score: ', mean_dice_score)
        print('Mean pixel level Precision: {}'.format(mean_p))
        print('Mean pixel level Recall: {}'.format(mean_r))

        print('Mean OL Precision: {}'.format(mean_ol_p))
        print('Mean OL Recall: {}'.format(mean_ol_r))
    return mean_dice_score
