"""Cutting aligned scenes into overlapping patch grids, and the naming scheme
that ties the grids on disk to the code that reads them back."""

from typing import Optional, Tuple

import numpy as np

from ..geo import X_CROP_COLS, X_CROP_OFFSET


def patch_dir_name(kind: str, patch_h: int, patch_w: int, strides_per_patch: int,
                   days_diff: Optional[int] = 11) -> str:
    """Directory holding one patch tree, e.g. data_patches_H200_W100_strpp2_11days_Aligned.

    ``days_diff=None`` selects the all-durations tree (``_all``). Frames are
    always aligned before patching, hence the unconditional ``_Aligned``.
    """
    assert kind in ("data", "mask")
    days = f"_{days_diff}days" if days_diff is not None else "_all"
    return f"{kind}_patches_H{patch_h}_W{patch_w}_strpp{strides_per_patch}{days}_Aligned"


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


def patchify(
    input_array: np.ndarray,
    window_size: Tuple[int, int],
    stride: Tuple[int, int],
    mask_array: Optional[np.ndarray] = None,
    nonz_patches: bool = True,
    offset: int = X_CROP_OFFSET,
    nx: int = X_CROP_COLS,
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

    data_patches, mask_patches = [], []
    data_nonz, mask_nonz, nonz_indices = [], [], []

    for idx_i, i in enumerate(range(0, rows - H + 1, Sy)):
        data_row, mask_row = [], []
        for idx_j, j in enumerate(range(0, cols - W + 1, Sx)):
            data_patch = input_array[i : i + H, j : j + W]
            data_row.append(data_patch)
            if mask_array is not None:
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
