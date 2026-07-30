"""The training dataset: patch grids on disk -> (image, mask) tensors.

A sample's image is ``(1, H, W)`` for single-frame training, ``(T, H, W)``
with ``--add_temporal`` (chronological, oldest -> newest, current frame last),
or ``(2T, H, W)`` when validity channels are appended — BLOCK layout
``[img_t0..img_t{T-1}, V_t0..V_t{T-1}]``. The mask is ``(H, W)`` int64 class
indices and, for temporal samples, belongs to the **newest** timestep.

Loading is lazy. ``__init__`` reads metadata only — .npy headers through
:func:`load_grid`, ``nonz_indices.json``, the coordinate dictionary — and
builds an index of ``(interferogram stack, grid row, grid column)``; the
pixels of one patch are read in :meth:`SubsiDataset.__getitem__`. A patch grid
is ~1.3 GB and a temporal sample spans ``T`` of them, so eager loading cost
tens of GB per split; the index costs a few tens of bytes per sample.

Pickle compatibility: trained runs pickle their held-out test split, and those
pickles restore instances by attribute without calling ``__init__``. Lazy
splits pickle their index (``groups``, ``samples``); splits pickled before the
lazy rewrite carry their arrays (``image_data``, ``mask_data``, ``index_map``).
``__getitem__`` reads only those attributes and serves both, and
:func:`load_test_dataset` maps the legacy module path onto this class.
"""

import io
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from ..normalise import PATCH_RANGE_TOL, normalise_channels
from .patchify import patch_file_name

#: Validity is derived from raw values: a pixel is data iff |raw| > this.
RAW_VALIDITY_TOL = 1e-9

#: Open memory maps kept alive. A temporal sample touches 2T files (T images +
#: T masks), so this holds a few consecutive stacks. Each entry costs a file
#: descriptor and a virtual mapping, not resident memory: the pages a patch
#: touches are file-backed and reclaimable by the kernel.
GRID_CACHE_SIZE = 32

#: Patch masks are written binary by ``sinkholes prepare-patches``; class index
#: i is the position in this list. See :meth:`SubsiDataset._binary_mask_values`.
BINARY_MASK_VALUES: Tuple[int, int] = (0, 1)


@lru_cache(maxsize=GRID_CACHE_SIZE)
def load_grid(path: str) -> np.ndarray:
    """A patch grid as a read-only memory map (only its header is read).

    Cached because consecutive samples come from the same interferograms.
    Module level, not per instance: the cache never enters a pickle, and each
    DataLoader worker process keeps its own.
    """
    return np.load(path, mmap_mode="r")


def clear_grid_cache() -> None:
    """Drop every cached memory map (tests, and after rewriting patch files)."""
    load_grid.cache_clear()


def temporal_target_mask(masks_per_t: Sequence[np.ndarray], union: bool = False) -> np.ndarray:
    """Ground truth for a temporal sample.

    ``masks_per_t`` is chronological oldest -> newest. The default target is
    the newest interferogram's mask — that is the state the model is asked to
    predict; the history only provides context. ``union=True`` restores the
    legacy target (positive wherever any timestep was positive).
    """
    if union:
        return (np.stack(masks_per_t, axis=0) > 0).any(axis=0).astype(np.float32)
    return (masks_per_t[-1] > 0).astype(np.float32)


def has_consecutive_zeros(arr: np.ndarray, min_consecutive: int = 10) -> bool:
    """True when any row or column contains a run of `min_consecutive` zeros."""
    kernel = np.ones(min_consecutive, dtype=int)

    def check(axis):
        return np.apply_along_axis(
            lambda x: np.any(np.convolve(x == 0, kernel, "valid") == min_consecutive),
            axis=axis,
            arr=arr,
        )

    return bool(np.any(check(1)) or np.any(check(0)))


@dataclass(frozen=True)
class RingNegatives:
    """All-zero patches sampled in an annulus around positive patches.

    Radii are in patch-grid units. Candidates must be empty at *every*
    timestep (the union grid) — a "negative" that was positive last month is
    not a negative.
    """

    inner: int = 1
    outer: int = 3
    per_pos: float = 1.0
    seed: Optional[int] = None

    def sample(self, positives, union_grid) -> List[Tuple[int, int]]:
        pos_grid = np.zeros_like(union_grid, dtype=bool)
        for (i, j) in positives:
            pos_grid[i, j] = True

        k_inner = np.ones((2 * self.inner + 1,) * 2, dtype=bool) if self.inner > 0 else np.ones((1, 1), dtype=bool)
        k_outer = np.ones((2 * self.outer + 1,) * 2, dtype=bool)
        ring = binary_dilation(pos_grid, structure=k_outer)
        ring &= ~(binary_dilation(pos_grid, structure=k_inner) if self.inner > 0 else pos_grid)

        candidates = list(zip(*np.where(ring & ~union_grid)))
        rng = np.random.default_rng(self.seed)
        k = min(len(candidates), int(self.per_pos * len(positives)))
        return [tuple(c) for c in rng.choice(candidates, size=k, replace=False).tolist()]


@dataclass(frozen=True)
class PatchSource:
    """Where one interferogram's (or one temporal stack's) patches live.

    ``tids`` is chronological, oldest -> newest, so ``tids[-1]`` is the frame a
    sample is asked to predict. ``row_offsets[t]`` is the grid row that sample
    row 0 maps to in ``tids[t]``'s file — 0 everywhere except spatial splits,
    where each grid is cut at its own latitude line. ``flat`` marks the
    pre-extracted ``*_nonz_*.npy`` files, which are ``(N, H, W)`` and indexed by
    column alone.
    """

    tids: Tuple[str, ...]
    image_paths: Tuple[str, ...]
    mask_paths: Tuple[str, ...]
    row_offsets: Tuple[int, ...] = ()
    flat: bool = False


class SubsiDataset(Dataset):
    """Patches of one or more interferograms, ready for a DataLoader.

    Construction indexes; :meth:`__getitem__` reads. ``__init__`` never loads a
    patch grid into memory — it takes shapes from the .npy headers and positive
    patch coordinates from ``nonz_indices.json`` — so a split's footprint scales
    with its number of samples, not with the bytes on disk.
    """

    #: Mask patches probed at construction to confirm the binary vocabulary.
    MASK_PROBES = 16

    def __init__(
        self,
        image_dir,
        mask_dir,
        intf_ids: Sequence[str],
        *,
        patch_size: Tuple[int, int] = (200, 100),
        stride: int = 2,
        mode: str = "train",
        nonz_only: bool = True,
        temporal: bool = False,
        seq_dict: Optional[Dict[str, Any]] = None,
        treat_nodata_regions: bool = False,
        union_temporal_mask: bool = False,
        add_nulls_to_train: bool = False,
        use_cleaned_patches: bool = False,
        ring_negatives: Optional[RingNegatives] = None,
        spatial: bool = False,
        thresh_lat: float = 31.4,
        coord_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        if not (self.image_dir.exists() and self.mask_dir.exists()):
            raise FileNotFoundError(
                f"patch directories missing: {self.image_dir} / {self.mask_dir}"
            )
        if not intf_ids:
            raise RuntimeError(f"empty interferogram list for {mode} set from {image_dir}")
        if temporal and seq_dict is None:
            raise ValueError("temporal datasets need seq_dict (the 11-day chains)")
        if spatial and coord_dict is None:
            raise ValueError("spatial partitioning needs coord_dict for the latitude line")

        self.ids = list(intf_ids)
        self.mode = mode
        self.temporal = temporal
        self.seq_dict = seq_dict
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.union_temporal_mask = union_temporal_mask
        self.treat_nodata_regions = treat_nodata_regions
        # Index at which validity channels begin (== T), or None when the
        # samples carry none. Only temporal samples ever carry them.
        self.n_value_channels: Optional[int] = None

        H, W = self.patch_size
        # The nonz (positive-only) files can serve directly only when nothing
        # requires the full grids: spatial splits, temporal stacks and
        # null-patch sampling all index into the (ny, nx, H, W) grid.
        use_nonz_files = nonz_only and not spatial and not add_nulls_to_train and not temporal

        def path_of(kind, tid, nonz=False):
            d = self.image_dir if kind == "data" else self.mask_dir
            name = patch_file_name(kind, tid, H, W, stride, nonz=nonz, cleaned=use_cleaned_patches)
            return join(d, name)

        # Positive patch coordinates, written by `sinkholes prepare-patches`.
        # Temporal stacks cannot be indexed without them. A spatial split only
        # takes the shortcut when the masks it reads are the ones the index was
        # written from — cleaning drops polygons — and scans otherwise.
        nonz_indices: Dict[str, Any] = {}
        index_file = self.image_dir / "nonz_indices.json"
        spatial_shortcut = spatial and nonz_only and not use_cleaned_patches
        if temporal or (spatial_shortcut and index_file.exists()):
            with open(index_file) as fh:
                nonz_indices = json.load(fh)

        #: One entry per interferogram (or temporal stack) that has samples.
        self.groups: List[PatchSource] = []
        #: (group, grid row, grid column) per sample — the whole dataset.
        self.samples: List[Tuple[int, int, int]] = []

        for intf_id in self.ids:
            if spatial:
                self._index_spatial(intf_id, path_of, thresh_lat, coord_dict,
                                    nonz_only, nonz_indices, ring_negatives)
            elif temporal:
                self._index_temporal(intf_id, path_of, nonz_indices, ring_negatives)
            elif use_nonz_files:
                self._index_flat(intf_id, path_of)
            else:
                self._index_grid(intf_id, path_of, nonz_only=nonz_only,
                                 add_nulls=add_nulls_to_train)

        if not self.samples:
            raise RuntimeError(f"no patches loaded for the {mode} set from {image_dir}")

        if temporal and treat_nodata_regions:
            # BLOCK layout: [images chronological..., validity chronological...].
            self.n_value_channels = len(self.groups[0].tids)
        self.mask_values = self._binary_mask_values()

    # -- per-interferogram indexing (metadata only, no pixels) -------------------------

    def _add_group(self, tids, image_paths, mask_paths, row_offsets=None, flat=False) -> int:
        self.groups.append(PatchSource(
            tids=tuple(tids),
            image_paths=tuple(image_paths),
            mask_paths=tuple(mask_paths),
            row_offsets=tuple(row_offsets if row_offsets is not None else (0,) * len(tids)),
            flat=flat,
        ))
        return len(self.groups) - 1

    def _index_flat(self, intf_id, path_of) -> None:
        """Index a pre-extracted ``*_nonz_*.npy`` file: every patch it holds."""
        img_path, msk_path = path_of("data", intf_id, True), path_of("mask", intf_id, True)
        n = load_grid(img_path).shape[0]
        if not n:
            return
        g = self._add_group([intf_id], [img_path], [msk_path], flat=True)
        self.samples.extend((g, -1, j) for j in range(n))

    def _index_grid(self, intf_id, path_of, *, rows=None, nonz_only=False,
                    add_nulls=False, nonz_coords=None) -> None:
        """Index one (ny, nx, H, W) grid in row-major order.

        ``rows`` restricts the grid rows to a half-open range (spatial split).
        ``nonz_coords``, when given, lists this interferogram's positive
        patches and spares a scan of its mask grid.
        """
        img_path, msk_path = path_of("data", intf_id), path_of("mask", intf_id)
        grid = load_grid(img_path)
        ny, nx = grid.shape[0], grid.shape[1]
        lo, hi = rows if rows is not None else (0, ny)
        lo, hi = max(0, lo), min(ny, hi)
        coords = [(i, j) for i in range(lo, hi) for j in range(nx)]

        if nonz_only and add_nulls:
            coords = self._sample_null_patches(grid, load_grid(msk_path), coords)
        elif nonz_only:
            if nonz_coords is not None:
                positive = {(int(i), int(j)) for i, j in nonz_coords}
                coords = [c for c in coords if c in positive]
            else:
                masks = load_grid(msk_path)
                coords = [c for c in coords if (masks[c] > 0).any()]
        if not coords:
            return

        g = self._add_group([intf_id], [img_path], [msk_path])
        self.samples.extend((g, i, j) for (i, j) in coords)

    def _index_temporal(self, intf_id, path_of, nonz_indices, ring_negatives,
                        spans=None) -> None:
        """Index one temporal stack: the positives of every timestep, unioned.

        ``spans`` maps tid -> (row offset, row count) after the spatial cut;
        None means each grid is used whole.
        """
        tids = list(self.seq_dict[intf_id]["prevs"]) + [intf_id]  # oldest -> newest
        img_paths = [path_of("data", tid) for tid in tids]
        msk_paths = [path_of("mask", tid) for tid in tids]
        shapes = [load_grid(p).shape for p in img_paths]
        if spans is None:
            spans = {tid: (0, s[0]) for tid, s in zip(tids, shapes)}

        # Different dates cover slightly different extents; clip every grid to
        # the common intersection so (i, j) means the same ground in all of them.
        ny = min(spans[tid][1] for tid in tids)
        nx = min(s[1] for s in shapes)

        # Positive coordinates: union over the whole stack, so a patch that was
        # positive at any timestep is trained on.
        rc, seen = [], set()
        for tid in tids:
            off = spans[tid][0]
            for ij in nonz_indices.get(tid, []):
                i, j = int(ij[0]) - off, int(ij[1])
                if 0 <= i < ny and 0 <= j < nx and (i, j) not in seen:
                    rc.append((i, j))
                    seen.add((i, j))
        if not rc:
            return

        if ring_negatives is not None and self.mode == "train":
            # Ring negatives are checked against the union over time on
            # purpose: a negative patch must be empty at every timestep — which
            # is exactly what the coordinates collected above cover.
            union_grid = np.zeros((ny, nx), dtype=bool)
            for (i, j) in rc:
                union_grid[i, j] = True
            rc = rc + ring_negatives.sample(rc, union_grid)

        g = self._add_group(tids, img_paths, msk_paths,
                            [spans[tid][0] for tid in tids])
        self.samples.extend((g, i, j) for (i, j) in rc)

    def _index_spatial(self, intf_id, path_of, thresh_lat, coord_dict,
                       nonz_only, nonz_indices, ring_negatives) -> None:
        """Split one interferogram at a latitude line: train north of it, val/test south."""
        H, _ = self.patch_size
        dy = coord_dict[intf_id]["dy"]
        # One grid row advances the top-left by Sy = H // 2 pixels southward.
        stride_deg = (H // 2) * dy
        thresh_line = int(np.floor((coord_dict[intf_id]["north"] - thresh_lat) / stride_deg))

        ref_rows = load_grid(path_of("data", intf_id)).shape[0]
        thresh_line = max(0, min(thresh_line, ref_rows))
        train_split = self.mode == "train"

        if self.temporal:
            # nonz_indices hold pre-cut grid rows; the offsets convert them.
            spans = {}
            for tid in list(self.seq_dict[intf_id]["prevs"]) + [intf_id]:
                rows_t = load_grid(path_of("data", tid)).shape[0]
                tl = min(thresh_line, rows_t)
                spans[tid] = (0, tl) if train_split else (tl, rows_t - tl)
            self._index_temporal(intf_id, path_of, nonz_indices, ring_negatives, spans=spans)
            return

        rows = (0, thresh_line) if train_split else (thresh_line, ref_rows)
        self._index_grid(intf_id, path_of, rows=rows, nonz_only=nonz_only,
                         nonz_coords=nonz_indices.get(intf_id) if nonz_indices else None)

    @staticmethod
    def _sample_null_patches(grid, masks, coords):
        """Positives plus a random sample of hard empty patches (no-data streaks).

        Patches are read one at a time through the memory maps and only their
        coordinates are kept. The draw order follows the row-major scan, so a
        seeded run selects the same patches it always did.
        """
        import random as _random

        keep = []
        for c in coords:
            if bool(np.any(masks[c] > 0)):
                keep.append(c)
                continue
            img = np.asarray(grid[c])
            if (_random.choice([True, False])
                    and np.sum(img == 0) < 1000
                    and has_consecutive_zeros(img)):
                keep.append(c)
        return keep

    def _binary_mask_values(self) -> List[int]:
        """The label vocabulary, confirmed against a bounded sample of patches.

        Patch masks are written binary by ``sinkholes prepare-patches``, so the
        vocabulary is known without reading the whole mask tree (~98 GB). A
        handful of patches spread over the index are still probed, so a
        non-binary tree fails loudly here rather than silently mapping every
        label onto class 0.
        """
        n = len(self.samples)
        step = max(1, n // self.MASK_PROBES)
        for idx in list(range(0, n, step))[: self.MASK_PROBES]:
            extra = set(np.unique(self._mask_patch(idx)).tolist()) - {0.0, 1.0}
            if extra:
                raise RuntimeError(
                    f"patch masks must be binary, but sample {idx} of the {self.mode} "
                    f"set holds {sorted(extra)} — re-run `sinkholes prepare-patches`"
                )
        return list(BINARY_MASK_VALUES)

    def sample_spec(self, idx) -> Dict[str, Any]:
        """What sample ``idx`` reads: the frames it stacks and where."""
        g, row, col = self.samples[idx]
        src = self.groups[g]
        return {"target": src.tids[-1], "prevs": list(src.tids[:-1]),
                "row": row, "col": col}

    @classmethod
    def from_arrays(cls, image_data, mask_data, ids=None, mode="test") -> "SubsiDataset":
        """A dataset directly over in-memory (N, H, W) arrays (non-overlap splits)."""
        self = cls.__new__(cls)
        Dataset.__init__(self)
        self.ids = list(ids or [])
        self.mode = mode
        self.temporal = False
        self.n_value_channels = None
        self.image_data = [np.expand_dims(image_data, axis=0)]
        self.mask_data = [np.expand_dims(mask_data, axis=0)]
        self.index_map = [[0, j] for j in range(image_data.shape[0])]
        uniques = np.unique(mask_data)
        self.mask_values = [int(v) if float(v).is_integer() else float(v) for v in sorted(uniques.tolist())]
        return self

    # -- sample access (reads only pickle-stable attributes) ---------------------------

    def __len__(self):
        samples = getattr(self, "samples", None)
        return len(self.index_map) if samples is None else len(samples)

    @staticmethod
    def preprocess(mask_values, img, is_mask, n_value_channels=None):
        """Normalise an image patch, or map a mask's values to class indices.

        For images, normalisation runs over axis 0 — timesteps of a (T, H, W)
        stack — and stops at ``n_value_channels`` so validity maps stay
        strictly {0, 1}.
        """
        if is_mask:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        return normalise_channels(img, range_tol=PATCH_RANGE_TOL, n_channels=n_value_channels)

    def _mask_patch(self, idx) -> np.ndarray:
        """The raw mask patch of sample ``idx``'s newest timestep."""
        g, row, col = self.samples[idx]
        src = self.groups[g]
        grid = load_grid(src.mask_paths[-1])
        return np.asarray(grid[col] if src.flat else grid[row + src.row_offsets[-1], col])

    def _lazy_sample(self, idx):
        """Read one sample's pixels: (image, target mask), unnormalised."""
        g, row, col = self.samples[idx]
        src = self.groups[g]
        H, W = self.patch_size

        if not self.temporal:
            grid = load_grid(src.image_paths[0])
            img = grid[col] if src.flat else grid[row, col]
            return np.asarray(img, dtype=np.float32), \
                np.asarray(self._mask_patch(idx), dtype=np.float32)

        image = np.stack(  # (T, H, W), chronological, current frame last
            [np.asarray(load_grid(p)[row + off, col][:H, :W], dtype=np.float32)
             for p, off in zip(src.image_paths, src.row_offsets)],
            axis=0,
        )
        if self.union_temporal_mask:
            masks_per_t = [np.asarray(load_grid(p)[row + off, col][:H, :W], dtype=np.float32)
                           for p, off in zip(src.mask_paths, src.row_offsets)]
        else:  # the target is the newest timestep — the rest need not be read
            masks_per_t = [np.asarray(self._mask_patch(idx)[:H, :W], dtype=np.float32)]
        target = temporal_target_mask(masks_per_t, union=self.union_temporal_mask)

        if self.treat_nodata_regions:
            valid = (np.abs(image) > RAW_VALIDITY_TOL).astype(np.float32)
            if np.isnan(image).any():
                valid = valid * ~np.isnan(image)
                image = np.nan_to_num(image, nan=0.0)
            # BLOCK layout: [images chronological..., validity chronological...].
            image = np.concatenate([image, valid], axis=0).astype(np.float32)
        return image, target

    def _eager_sample(self, idx):
        """Read one sample from in-memory arrays (``from_arrays``, old pickles)."""
        intf_idx, patch_idx = self.index_map[idx]
        if getattr(self, "temporal", False):
            img = self.image_data[intf_idx][:, patch_idx].astype(np.float32)  # (T|2T, H, W)
        else:
            img = self.image_data[intf_idx][0, patch_idx].astype(np.float32)  # (H, W)
        return img, self.mask_data[intf_idx][0, patch_idx].astype(np.float32)

    def __getitem__(self, sample):
        # Lazy datasets carry an index; arrays mean `from_arrays` or a test
        # split pickled before the lazy rewrite.
        if hasattr(self, "image_data"):
            img, msk = self._eager_sample(sample)
        else:
            img, msk = self._lazy_sample(sample)

        img = self.preprocess(self.mask_values, img, 0,
                              n_value_channels=getattr(self, "n_value_channels", None))
        msk = self.preprocess(self.mask_values, msk, 1)

        assert img.shape[-2:] == msk.shape[-2:], \
            f"spatial size mismatch: image {img.shape} vs mask {msk.shape}"

        if getattr(self, "temporal", False):
            img_t = torch.as_tensor(img.copy()).float().contiguous()
        else:
            img_t = torch.as_tensor(img.copy()).unsqueeze(0).float().contiguous()
        msk_t = torch.as_tensor(msk.copy()).long().contiguous()
        return {"image": img_t, "mask": msk_t}


# -- saved test sets --------------------------------------------------------------------

#: Module paths under which SubsiDataset was pickled by earlier versions.
_LEGACY_DATASET_MODULES = {"sinkholes_data_loading"}


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SubsiDataset" and module in _LEGACY_DATASET_MODULES:
            return SubsiDataset
        return super().find_class(module, name)


def load_test_dataset(path):
    """Load a pickled test split, including ones written before the rewrite."""
    with open(path, "rb") as fh:
        return _CompatUnpickler(fh).load()


def save_test_dataset(dataset, path) -> float:
    """Pickle a test split; returns its size in GB.

    A lazy split pickles its index — paths and coordinates — not its pixels,
    so the file is small and reloading it re-reads the patch tree in place.
    """
    buffer = io.BytesIO()
    pickle.dump(dataset, buffer)
    size_gb = buffer.tell() / (1024**3)
    with open(path, "wb") as fh:
        fh.write(buffer.getvalue())
    return size_gb
