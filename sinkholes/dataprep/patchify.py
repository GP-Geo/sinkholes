"""Cutting aligned scenes into overlapping patch grids, and the naming scheme
that ties the grids on disk to the code that reads them back."""

import os
from typing import Optional, Tuple

import numpy as np

from ..geo import X_CROP_COLS, X_CROP_OFFSET


def patch_dir_name(kind: str, patch_h: int, patch_w: int, strides_per_patch: int,
                   days_diff: Optional[int] = 11,
                   context_margin: Optional[Tuple[int, int]] = None) -> str:
    """Directory holding one patch tree, e.g. data_patches_H200_W100_strpp2_11days_Aligned.

    ``days_diff=None`` selects the all-durations tree (``_all``). Frames are
    always aligned before patching, hence the unconditional ``_Aligned``.

    ``context_margin`` names a large-context DATA tree, e.g.
    ``data_patches_H200_W100_ctx50x50_strpp2_11days_Aligned``. The name carries
    the **target** size and the margin, never the stored array size, because the
    grid is the plain tree's grid: cell (i, j) is the plain cell (i, j) with a
    margin grown around it. Masks are never context-padded -- the target stays
    the plain patch -- so a margin on ``kind="mask"`` is a programming error.
    """
    assert kind in ("data", "mask")
    days = f"_{days_diff}days" if days_diff is not None else "_all"
    ctx = ""
    if context_margin and tuple(context_margin) != (0, 0):
        if kind == "mask":
            raise ValueError("mask trees are never context-padded: the target is the plain patch")
        ctx = f"_ctx{context_margin[0]}x{context_margin[1]}"
    return f"{kind}_patches_H{patch_h}_W{patch_w}{ctx}_strpp{strides_per_patch}{days}_Aligned"


def patch_file_name(kind: str, intf_id: str, patch_h: int, patch_w: int,
                    strides_per_patch: int, nonz: bool = False, cleaned: bool = False) -> str:
    """One grid file, e.g. data_patches_nonz_20190205_20190216_H200_W100_strpp2.npy."""
    assert kind in ("data", "mask")
    prefix = f"{kind}_patches_nonz_" if nonz else f"{kind}_patches_"
    suffix = "_cleaned" if cleaned else ""
    return f"{prefix}{intf_id}_H{patch_h}_W{patch_w}_strpp{strides_per_patch}{suffix}.npy"


def patch_strides(patch_size: Tuple[int, int], strides_per_patch: int) -> Tuple[int, int]:
    """(Sy, Sx) pixel steps; strides_per_patch=2 means a 50% overlap step."""
    h, w = patch_size
    return h // strides_per_patch, w // strides_per_patch


def resolve_patch_dirs(
    patches_dir,
    patch_size: Tuple[int, int],
    strides_per_patch: int,
    *,
    days_diff: Optional[int] = 11,
    cleaned: bool = False,
    context_margin: Optional[Tuple[int, int]] = None,
) -> Tuple[str, str]:
    """(image_dir, mask_dir) of one patch tree under ``patches_dir``.

    The single place the naming scheme above is turned into the pair of
    directories a dataset is read from: training and evaluation must resolve
    them identically or an evaluation silently reads a different tree from the
    one the checkpoint was trained on. Missing directories are an error here
    rather than a confusing per-file FileNotFoundError later.
    """
    H, W = patch_size
    # The image tree carries the context margin; the mask tree never does, so a
    # context run reads its targets from the SAME mask tree a plain run does.
    # That is what makes "select samples on the centre mask" true by
    # construction instead of by a cropping convention.
    image_dir = os.path.join(patches_dir, patch_dir_name("data", H, W, strides_per_patch,
                                                         days_diff, context_margin))
    mask_dir = os.path.join(patches_dir, patch_dir_name("mask", H, W, strides_per_patch, days_diff))
    if cleaned:
        image_dir = os.path.join(image_dir, "cleaned")
        mask_dir = os.path.join(mask_dir, "cleaned")
    missing = [d for d in (image_dir, mask_dir) if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"patch directories not found: {' | '.join(missing)} — "
                         f"prepare patches first, or check --patch_size/--stride")
    return image_dir, mask_dir


def _padded_window(scene: np.ndarray, i: int, j: int, H: int, W: int,
                   My: int, Mx: int, pad_value: float = 0.0) -> np.ndarray:
    """Rows [i-My, i+H+My) x cols [j-Mx, j+W+Mx), zero-padded past the edges.

    0 is the project's no-data code: ``normalise_phase`` maps it to 0.5 and
    ``--treat_nodata_regions`` then marks it invalid, so padded ground is
    already handled by machinery that exists.
    """
    r0, c0 = i - My, j - Mx
    r1, c1 = i + H + My, j + W + Mx
    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r1, scene.shape[0]), min(c1, scene.shape[1])
    out = np.full((r1 - r0, c1 - c0), pad_value, dtype=scene.dtype)
    if rr0 < rr1 and cc0 < cc1:
        out[rr0 - r0: rr1 - r0, cc0 - c0: cc1 - c0] = scene[rr0:rr1, cc0:cc1]
    return out


def patchify(
    input_array: np.ndarray,
    window_size: Tuple[int, int],
    stride: Tuple[int, int],
    mask_array: Optional[np.ndarray] = None,
    nonz_patches: bool = True,
    offset: int = X_CROP_OFFSET,
    nx: int = X_CROP_COLS,
    margin: Tuple[int, int] = (0, 0),
):
    """Cut a scene (and its mask) into an overlapping patch grid.

    Columns are cropped to [offset, offset + nx) before patching; rows tile
    from 0 with step Sy. With a mask, patches whose mask has any positive pixel
    are additionally collected as the "nonz" subset together with their grid
    coordinates.

    Returns
    -------
    Without a mask: ``data_patches`` of shape (ny, nx_grid, H, W).
    With a mask: ``(data_patches, mask_patches, data_nonz, mask_nonz,
    nonz_indices)`` where the nonz arrays are (N, H, W) and ``nonz_indices``
    is a row-major list of ``[i, j]`` grid coordinates.
    """
    if mask_array is not None:
        assert input_array.shape == mask_array.shape, "mask must have the data's shape"
        mask_array = mask_array[:, offset : offset + nx]
    input_array = input_array[:, offset : offset + nx]

    rows, cols = input_array.shape
    H, W = window_size
    Sy, Sx = stride
    My, Mx = margin

    data_patches, mask_patches = [], []
    data_nonz, mask_nonz, nonz_indices = [], [], []

    # The grid is unchanged by a margin: cells still tile at (Sy, Sx) over the
    # TARGET window, and only what the data patch carries around that window
    # grows. So (idx_i, idx_j), nonz_indices and every partition keep meaning.
    for idx_i, i in enumerate(range(0, rows - H + 1, Sy)):
        data_row, mask_row = [], []
        for idx_j, j in enumerate(range(0, cols - W + 1, Sx)):
            if My or Mx:
                data_patch = _padded_window(input_array, i, j, H, W, My, Mx)
            else:
                data_patch = input_array[i : i + H, j : j + W]
            data_row.append(data_patch)
            if mask_array is not None:
                # Targets are never margined: positivity, and therefore sample
                # selection, is decided on the centre window alone.
                mask_patch = mask_array[i : i + H, j : j + W]
                mask_row.append(mask_patch)
                if nonz_patches and mask_patch.any():
                    data_nonz.append(data_patch)
                    mask_nonz.append(mask_patch)
                    nonz_indices.append([idx_i, idx_j])
        data_patches.append(data_row)
        if mask_array is not None:
            mask_patches.append(mask_row)

    data_patches = np.array(data_patches)
    if mask_array is None:
        return data_patches

    mask_patches = np.array(mask_patches)
    if data_nonz:
        data_nonz = np.array(data_nonz)
        mask_nonz = np.array(mask_nonz)
    else:
        data_nonz = np.empty((0, H, W), dtype=input_array.dtype)
        mask_nonz = np.empty((0, H, W), dtype=np.uint8)
    return data_patches, mask_patches, data_nonz, mask_nonz, nonz_indices
