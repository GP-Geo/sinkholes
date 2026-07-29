"""Per-object candidate features (shape and phase statistics).

Used by :func:`sinkholes.training.evaluate.object_level_evaluate` to profile
detected vs undetected ground-truth objects — pass a non-empty ``features``
list to collect them. Geometry features come from the polygon; phase features
from the interferogram pixels inside it.
"""

import numpy as np
from affine import Affine
from numpy.fft import fft2, fftshift
from rasterio.features import geometry_mask
from scipy.ndimage import gaussian_filter
from shapely.affinity import translate
from shapely.geometry import Polygon


def compute_feature(p: Polygon, feature: str, image: np.ndarray):
    """One feature of one ground-truth polygon; `image` is (T, H, W)."""
    if feature == "area":
        return p.area
    if feature == "perimeter":
        return p.length
    if feature == "solidity":
        return p.area / p.convex_hull.area
    if feature == "roundness":
        return 4 * np.pi * p.area / (p.length**2)
    if feature.startswith("phase"):
        minx, miny, maxx, maxy = map(int, p.bounds)
        pad = 5
        minx = max(minx - pad, 0)
        miny = max(miny - pad, 0)
        maxx = min(maxx + pad, image.shape[2])
        maxy = min(maxy + pad, image.shape[1])
        sub_image = image[:, miny:maxy, minx:maxx]
        shifted = translate(p, xoff=-minx, yoff=-miny)
        submask = geometry_mask([shifted], transform=Affine.identity(), invert=True,
                                out_shape=sub_image[0].shape)

        if feature == "phase_std":
            return float(np.nanmean(np.std(sub_image[:, submask], axis=1)))
        if feature == "phase_gradient":
            return compute_mean_phase_gradient(sub_image, submask, unwrap=True)
        if feature == "phase_fft":
            return compute_fft_noise_ratio(sub_image[0], submask)
        if feature == "phase_radial_symmetry":
            return radial_symmetry_score(sub_image[0], submask)
    raise ValueError(f"unknown feature {feature!r}")


def radial_symmetry_score(image, mask, center=None, min_ring_pixels=10):
    """1 - mean coefficient of variation over concentric rings inside the mask.

    A radially symmetric bowl (the classic subsidence signature) scores high.
    """
    y, x = np.indices(image.shape)
    if center is None:
        ys, xs = np.where(mask)
        center = (np.mean(ys), np.mean(xs))
    cy, cx = center

    r_int = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    scores = []
    for ri in range(1, r_int[mask].max()):
        ring = (r_int == ri) & mask
        if np.sum(ring) < min_ring_pixels:
            continue
        values = image[ring]
        mean = np.mean(values)
        if mean != 0:
            scores.append(np.std(values) / mean)
    return np.nan if not scores else 1 - np.mean(scores)


def compute_fft_noise_ratio(image, mask, pixel_size=3.0, freq_th=0.05, sigma=2,
                            return_spectrum=False):
    """High/low spatial-frequency energy ratio of the masked phase.

    Noise-dominated regions carry relatively more high-frequency energy than a
    smooth deformation bowl.
    """
    h, w = image.shape
    windowed = image * gaussian_filter(mask.astype(float), sigma=sigma)
    fft_energy = np.abs(fftshift(fft2(windowed))) ** 2

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r_int = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)

    radial = np.zeros(r_int.max() + 1)
    for i in range(len(radial)):
        ring = r_int == i
        if np.any(ring):
            radial[i] = fft_energy[ring].sum()

    energy = radial / (radial.sum() + 1e-12)
    freqs = np.arange(len(energy)) / (np.sqrt(h**2 + w**2) * pixel_size)
    high = energy[freqs > freq_th].sum()
    low = energy[freqs <= freq_th].sum()
    ratio = high / (low + 1e-12)
    return (ratio, freqs, energy) if return_spectrum else ratio


def compute_mean_phase_gradient(image, mask, unwrap=False):
    """Mean gradient magnitude of the masked phase, averaged over timesteps.

    With ``unwrap`` the [0, 1]-scaled phase is unwrapped first, so the wrap
    seam does not read as an enormous gradient.
    """
    from skimage.restoration import unwrap_phase

    T, H, W = image.shape
    mask_bool = mask.astype(bool)
    assert mask_bool.shape == (H, W)

    out_sum, valid_frames = 0.0, 0
    for i in range(T):
        frame = image[i]
        if unwrap:
            frame = unwrap_phase(2 * np.pi * frame) / (2 * np.pi)
        gy, gx = np.gradient(frame)
        masked = np.hypot(gx, gy)[mask_bool]
        if masked.size == 0:
            continue
        out_sum += float(np.mean(masked))
        valid_frames += 1
    return np.nan if valid_frames == 0 else out_sum / valid_frames


def compute_entropy_region(image, mask, radius=2):
    """Average local entropy inside a masked region."""
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.util import img_as_ubyte

    normed = (image - image.min()) / (image.ptp() + 1e-8)
    return entropy(img_as_ubyte(normed), disk(radius))[mask].mean()
