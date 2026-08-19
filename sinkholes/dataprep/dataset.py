"""The training dataset: patch grids on disk -> (image, mask) tensors.

A sample's image is ``(1, H, W)`` for single-frame training, ``(T, H, W)``
with ``--add_temporal`` (chronological, oldest -> newest, current frame last),
or ``(2T, H, W)`` when validity channels are appended — BLOCK layout
``[img_t0..img_t{T-1}, V_t0..V_t{T-1}]``. The mask is ``(H, W)`` int64 class
indices and, for temporal samples, belongs to the **newest** timestep.

Both paths cover the **same patches**: a sample exists where the interferogram
being predicted has a positive mask, so a temporal and a single-frame run over
one partition hold the same coordinates and differ only in input depth. Only
the target's own definition moves that set — ``--union_temporal_mask`` targets
the union over the stack and samples the union's coordinates to match.

Negatives — all-zero patches from an annulus around the positives — are added
on top of that set by :class:`RingNegatives`, and the two kinds are kept apart
by which split they reach. ``ring_negatives`` is the experiment's *training*
negatives (train split only, radii and ratio swept from the CLI);
``val_negatives`` is the fixed measurement set (val split only, configuration
pinned by :func:`validation_negatives`).

Pickle compatibility: trained runs pickle their held-out test split, and those
pickles restore instances by attribute (``image_data``, ``mask_data``,
``index_map``, ``temporal``, ``mask_values``, ``n_value_channels``) without
calling ``__init__``. ``__getitem__`` therefore reads only those attributes,
and :func:`load_test_dataset` maps the legacy module path onto this class.
"""

import io
import json
import pickle
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from ..geo import FRAME_ORIGINS, grid_window
from ..normalise import PATCH_RANGE_TOL, normalise_channels
from .patchify import patch_file_name

#: Validity is derived from raw values: a pixel is data iff |raw| > this.
RAW_VALIDITY_TOL = 1e-9


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

    The draw depends only on ``seed`` and on the candidate set, never on the
    global RNG: :func:`numpy.random.default_rng` is built fresh per call. Two
    runs over one partition therefore draw identical negatives whatever else
    differs between them, which is what :func:`validation_negatives` relies on.
    """

    inner: int = 1
    outer: int = 3
    per_pos: float = 1.0
    seed: Optional[int] = None

    def sample(self, positives, union_grid, allowed=None) -> List[Tuple[int, int]]:
        """Negatives around ``positives``, optionally confined to ``allowed``.

        ``allowed`` is a boolean grid of cells the split may use — the AOI
        window. Without it a negative could be drawn outside the split's own
        ground, which on the geo axis means drawing training background out of
        the hold-out band.
        """
        pos_grid = np.zeros_like(union_grid, dtype=bool)
        for (i, j) in positives:
            pos_grid[i, j] = True

        k_inner = np.ones((2 * self.inner + 1,) * 2, dtype=bool) if self.inner > 0 else np.ones((1, 1), dtype=bool)
        k_outer = np.ones((2 * self.outer + 1,) * 2, dtype=bool)
        ring = binary_dilation(pos_grid, structure=k_outer)
        ring &= ~(binary_dilation(pos_grid, structure=k_inner) if self.inner > 0 else pos_grid)

        usable = ring & ~union_grid
        if allowed is not None:
            usable &= allowed
        candidates = list(zip(*np.where(usable)))
        rng = np.random.default_rng(self.seed)
        k = min(len(candidates), int(self.per_pos * len(positives)))
        return [tuple(c) for c in rng.choice(candidates, size=k, replace=False).tolist()]


#: The validation-negative configuration, fixed on purpose (--add_val_negatives).
#:
#: Training negatives are an experimental variable — ``--neg_ring_outer`` and
#: ``--neg_per_pos`` are swept, and that is the point of them. Validation
#: negatives are the opposite: they are part of the *ruler*, so they are pinned
#: here rather than exposed as flags. Two runs on one partition and one seed are
#: then scored on byte-identical validation samples however their training
#: negatives were configured and whatever architecture they use, which is what
#: makes their val/dice directly comparable.
VAL_NEGATIVE_INNER = 1
VAL_NEGATIVE_OUTER = 3
VAL_NEGATIVE_PER_POS = 1.0


def validation_negatives(seed: int) -> RingNegatives:
    """The fixed 1:1 validation-negative sampler for a run seed.

    ``seed`` is the run seed, and it is the only thing that moves: the radii
    and the ratio are constants. The sampler is applied once, when the
    validation dataset is built, so the negatives are drawn from the validation
    interferograms alone and stay fixed for every epoch of the run.
    """
    if seed is None:
        raise ValueError("validation negatives need a seed to be reproducible")
    return RingNegatives(inner=VAL_NEGATIVE_INNER, outer=VAL_NEGATIVE_OUTER,
                         per_pos=VAL_NEGATIVE_PER_POS, seed=int(seed))


class SubsiDataset(Dataset):
    """Patches of one or more interferograms, ready for a DataLoader."""

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
        val_negatives: Optional[RingNegatives] = None,
        spatial: bool = False,
        thresh_lat: float = 31.4,
        coord_dict: Optional[Dict[str, Any]] = None,
        aoi_window: Optional[Tuple[float, float, float, float]] = None,
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
        if aoi_window is not None and spatial:
            raise ValueError(
                "aoi_window and spatial= are two different ways to restrict ground and "
                "cannot be combined: the spatial path slices the grids by latitude, so a "
                "window applied afterwards would be interpreted in post-slice rows and "
                "land on the wrong ground. Use a partition file carrying aoi_window."
            )
        if aoi_window is not None:
            if coord_dict is None:
                raise ValueError(
                    "aoi_window needs coord_dict: the window is per frame, and only "
                    "the coordinate dictionary says which frame an interferogram is on"
                )
            if len(aoi_window) != 4:
                raise ValueError(
                    f"aoi_window must be (lat_min, lat_max, lon_min, lon_max), got {aoi_window!r}"
                )
        self.aoi_window = tuple(aoi_window) if aoi_window is not None else None
        self.coord_dict = coord_dict
        self.patch_size = tuple(patch_size)
        self._grid_stride = (patch_size[0] // stride, patch_size[1] // stride)

        self.ids = list(intf_ids)
        self.mode = mode
        self.temporal = temporal
        self.seq_dict = seq_dict
        # Index at which validity channels begin (== T), or None when the
        # samples carry none. Set where the concatenation happens so the two
        # can never disagree.
        self.n_value_channels: Optional[int] = None

        # Which negatives, if any, this split gets. ``ring_negatives`` are the
        # experiment's training negatives and reach the train split only;
        # ``val_negatives`` are the fixed measurement set and reach the val
        # split only. Resolving both to one object here means the loaders never
        # have to know which split they are serving, and neither kind can leak
        # into the other's set.
        negatives = (val_negatives if self.mode == "val"
                     else ring_negatives if self.mode == "train" else None)

        H, W = patch_size
        # The nonz (positive-only) files can serve directly only when nothing
        # requires the full grids: spatial splits, temporal stacks, null-patch
        # sampling and negatives all index into the (ny, nx, H, W) grid.
        use_nonz_files = (nonz_only and not spatial and not add_nulls_to_train
                          and not temporal and negatives is None)

        def path_of(kind, tid, nonz):
            d = self.image_dir if kind == "data" else self.mask_dir
            name = patch_file_name(kind, tid, H, W, stride, nonz=nonz, cleaned=use_cleaned_patches)
            return join(d, name)

        # Negatives need the coordinate list too: the single-frame path
        # normally reads the pre-extracted nonz files, which carry patches but
        # not their (i, j) positions, and an annulus cannot be drawn without
        # positions.
        nonz_indices: Dict[str, Any] = {}
        if temporal or negatives is not None:
            with open(self.image_dir / "nonz_indices.json") as fh:
                nonz_indices = json.load(fh)

        self.image_data: List[np.ndarray] = []
        self.mask_data: List[np.ndarray] = []
        self.index_map: List[List[int]] = []

        for intf_idx, intf_id in enumerate(self.ids):
            if spatial:
                image_data, mask_data = self._load_spatial(
                    intf_id, path_of, patch_size, thresh_lat, coord_dict,
                    nonz_only, nonz_indices, negatives,
                    treat_nodata_regions, union_temporal_mask,
                )
            elif temporal:
                image_data, mask_data = self._load_temporal(
                    intf_id, path_of, patch_size, nonz_indices, negatives,
                    treat_nodata_regions, union_temporal_mask, row_offsets=None,
                )
            elif negatives is not None and nonz_only:
                # Single-frame + negatives: training negatives on the train
                # split, validation negatives on the val split. Both use the
                # same coordinate maths, so a single-frame val set and a
                # temporal one hold exactly the same patches.
                image_data, mask_data = self._load_single_ring(
                    intf_id, path_of, patch_size, nonz_indices, negatives,
                )
            elif use_nonz_files:
                image_data = np.load(path_of("data", intf_id, nonz=True))
                mask_data = np.load(path_of("mask", intf_id, nonz=True))
            else:
                image_data = np.load(path_of("data", intf_id, nonz=False))
                mask_data = np.load(path_of("mask", intf_id, nonz=False))
                image_data = image_data.reshape(-1, image_data.shape[2], image_data.shape[3])
                mask_data = mask_data.reshape(-1, mask_data.shape[2], mask_data.shape[3])
                if nonz_only and add_nulls_to_train:
                    image_data, mask_data = self._add_null_patches(image_data, mask_data)

            if image_data is None or image_data.size == 0:
                continue
            if image_data.ndim == 3:  # (N, H, W) -> (1, N, H, W)
                image_data = np.expand_dims(image_data, axis=0)
            mask_data = np.expand_dims(mask_data, axis=0)

            self.image_data.append(image_data)
            self.mask_data.append(mask_data)
            n = image_data.shape[1]
            self.index_map.extend([[len(self.image_data) - 1, j] for j in range(n)])

        if not self.index_map:
            raise RuntimeError(f"no patches loaded for the {mode} set from {image_dir}")

        # Label vocabulary over everything actually loaded (binary data: [0, 1]).
        uniques = np.unique(np.concatenate([m.reshape(-1) for m in self.mask_data]))
        self.mask_values = [int(v) if float(v).is_integer() else float(v) for v in sorted(uniques.tolist())]

        # Samples whose target is empty. Zero on a positives-only set; on a val
        # set built with --add_val_negatives it is the negative count, which is
        # what a run has to report to be readable against another.
        self.n_negative = int(sum(int((~(m[0] > 0).any(axis=(-2, -1))).sum())
                                  for m in self.mask_data))

    # -- per-interferogram loaders -----------------------------------------------------

    def _window_for(self, intf_id, ny, nx):
        """The split's grid window for one interferogram, clamped to (ny, nx).

        ``None`` means unrestricted. The window is per *frame*, because the two
        frames have different aligned origins, so the same lat/lon box lands on
        different grid rows in each.
        """
        if self.aoi_window is None:
            return None
        meta = self.coord_dict.get(intf_id)
        if meta is None or meta.get("frame") not in FRAME_ORIGINS:
            raise KeyError(
                f"{intf_id} has no usable frame in the coordinate dictionary, "
                f"so its AOI window cannot be resolved"
            )
        r0, r1, c0, c1 = grid_window(
            meta["frame"], *self.aoi_window,
            patch_size=self.patch_size, stride=self._grid_stride,
        )
        return r0, min(r1, ny), c0, min(c1, nx)

    def _allowed_grid(self, window, ny, nx):
        """Boolean (ny, nx) of cells the split may draw negatives from."""
        if window is None:
            return None
        allowed = np.zeros((ny, nx), dtype=bool)
        r0, r1, c0, c1 = window
        allowed[r0:r1, c0:c1] = True
        return allowed

    def _load_temporal(self, intf_id, path_of, patch_size, nonz_indices,
                       negatives, treat_nodata, union_mask, row_offsets):
        """Build (T, N, H, W) images + (N, H, W) target for one interferogram.

        Samples are the positive patches of ``intf_id`` itself — the same set
        the single-frame path loads from the nonz files — each carrying the
        full T-frame stack as input. Coordinates absent from an earlier grid
        are dropped: a stack needs the location to exist at every timestep.

        ``row_offsets`` maps tid -> rows already sliced off the top of its grid
        (spatial mode); None means the full grids are used as-is.
        """
        tids = list(self.seq_dict[intf_id]["prevs"]) + [intf_id]  # oldest -> newest
        H, W = patch_size

        if row_offsets is None:
            img_pa = [np.load(path_of("data", tid, nonz=False)).astype(np.float32) for tid in tids]
            msk_pa = [np.load(path_of("mask", tid, nonz=False)).astype(np.float32) for tid in tids]
            row_offsets = {tid: 0 for tid in tids}
        else:
            img_pa, msk_pa = row_offsets.pop("_arrays")

        # Different dates cover slightly different extents; clip every grid to
        # the common intersection so (i, j) means the same ground in all of them.
        ny = min(p.shape[0] for p in img_pa)
        nx = min(p.shape[1] for p in img_pa)
        img_pa = [p[:ny, :nx, :H, :W] for p in img_pa]
        msk_pa = [p[:ny, :nx, :H, :W] for p in msk_pa]

        # Sample coordinates come from whichever interferograms define the
        # target, so every sample carries a non-empty one: the current frame
        # by default, the whole stack only when the target is its union.
        # Earlier timesteps are input context and never contribute
        # coordinates of their own — a patch that was positive last month but
        # is empty now would otherwise enter the set with an all-zero target,
        # which the single-frame path never yields and which per-patch Dice
        # scores 1.0 for predicting nothing.
        coord_tids = tids if union_mask else [intf_id]
        # The split's AOI window, in post-slice grid coordinates. Applied to the
        # sample coordinates rather than to the loaded grids so the arrays stay
        # whole -- _load_spatial's row offsets and the negative sampler both
        # index the full grid.
        window = self._window_for(intf_id, ny, nx)
        rc, seen = [], set()
        for tid in coord_tids:
            off = row_offsets[tid]
            for ij in nonz_indices.get(tid, []):
                i, j = int(ij[0]) - off, int(ij[1])
                if not (0 <= i < ny and 0 <= j < nx) or (i, j) in seen:
                    continue
                if window is not None and not (window[0] <= i < window[1]
                                               and window[2] <= j < window[3]):
                    continue
                rc.append((i, j))
                seen.add((i, j))
        if not rc:
            return None, None

        if negatives is not None:
            # Negatives are checked against the union over time on purpose: a
            # negative patch must be empty at every timestep, which also makes
            # its target — the current frame — empty. Validation negatives go
            # through this same test, so they satisfy exactly the temporal
            # validity a training negative does.
            union_grid = np.zeros((ny, nx), dtype=bool)
            for m in msk_pa:
                union_grid |= (m > 0).any(axis=(-2, -1))
            rc = rc + negatives.sample(rc, union_grid,
                                       allowed=self._allowed_grid(window, ny, nx))

        image = np.stack(
            [np.stack([p[i, j] for (i, j) in rc], axis=0) for p in img_pa], axis=0
        ).astype(np.float32)  # (T, N, H, W)
        masks_per_t = [np.stack([m[i, j] for (i, j) in rc], axis=0) for m in msk_pa]
        target = temporal_target_mask(masks_per_t, union=union_mask)

        if treat_nodata:
            valid = (np.abs(image) > RAW_VALIDITY_TOL).astype(np.float32)
            if np.isnan(image).any():
                valid = valid * ~np.isnan(image)
                image = np.nan_to_num(image, nan=0.0)
            # BLOCK layout: [images chronological..., validity chronological...].
            image = np.concatenate([image, valid], axis=0).astype(np.float32)
            self.n_value_channels = len(tids)

        return image, target

    def _load_single_ring(self, intf_id, path_of, patch_size, nonz_indices, negatives):
        """One interferogram's positive patches plus negatives, single frame.

        This exists so a single-frame U-Net can be trained on the *same* data a
        temporal run sees, minus the history — the control that separates "what
        the negatives bought" from "what the recurrence bought". The validation
        split reaches it too, under ``--add_val_negatives``, for the same
        reason: a single-frame run and a temporal one must be *scored* on the
        same patches, not merely trained on matching ones.

        The coordinate maths deliberately mirrors ``_load_temporal``: same
        positive set, same clipping to the chain's common extent, same
        union-over-time exclusion grid, same sampler and seed. Given one
        partition, both paths therefore draw the **same** negative patches, so
        the pair differs in input depth and nothing else. Diverging here — for
        instance excluding only the current frame's positives — would confound
        the architecture with which negatives it happened to get.

        Only the chain's MASK grids are read for the exclusion; its image grids
        are not, which is what keeps this much cheaper than a temporal load.
        ``patchify`` writes data and mask grids at the same (ny, nx), so the
        clip agrees with the one ``_load_temporal`` computes from the images.
        """
        H, W = patch_size
        img = np.load(path_of("data", intf_id, nonz=False)).astype(np.float32)

        # A negative must be empty at every timestep — a patch that was positive
        # last month is not background. Without a chain this degrades to the
        # current frame alone, which is weaker but still correct for its own
        # target; the log line in train.py says which applied.
        tids = (list(self.seq_dict[intf_id]["prevs"]) + [intf_id]
                if self.seq_dict and intf_id in self.seq_dict else [intf_id])
        msk_pa = [np.load(path_of("mask", t, nonz=False)).astype(np.float32) for t in tids]

        ny = min([p.shape[0] for p in msk_pa] + [img.shape[0]])
        nx = min([p.shape[1] for p in msk_pa] + [img.shape[1]])
        img = img[:ny, :nx, :H, :W]
        msk_pa = [p[:ny, :nx, :H, :W] for p in msk_pa]
        msk = msk_pa[-1]  # tids ends with intf_id

        # Mirrors _load_temporal's window handling exactly: the single-frame
        # control must be scored on the same patches as the temporal run, so a
        # difference in which cells the window admits would confound the
        # architecture comparison this path exists to make.
        window = self._window_for(intf_id, ny, nx)
        rc, seen = [], set()
        for ij in nonz_indices.get(intf_id, []):
            i, j = int(ij[0]), int(ij[1])
            if not (0 <= i < ny and 0 <= j < nx) or (i, j) in seen:
                continue
            if window is not None and not (window[0] <= i < window[1]
                                           and window[2] <= j < window[3]):
                continue
            rc.append((i, j))
            seen.add((i, j))
        if not rc:
            return None, None

        union_grid = np.zeros((ny, nx), dtype=bool)
        for m in msk_pa:
            union_grid |= (m > 0).any(axis=(-2, -1))
        rc = rc + negatives.sample(rc, union_grid,
                                   allowed=self._allowed_grid(window, ny, nx))

        image = np.stack([img[i, j] for (i, j) in rc], axis=0).astype(np.float32)
        target = np.stack([msk[i, j] for (i, j) in rc], axis=0).astype(np.float32)
        return image, target

    def _load_spatial(self, intf_id, path_of, patch_size, thresh_lat, coord_dict,
                      nonz_only, nonz_indices, negatives, treat_nodata, union_mask):
        """Split one interferogram at a latitude line: train north of it, val/test south.

        The threshold row comes from :func:`grid_window`, i.e. from the frame's
        **aligned** origin. It used to be computed from the scene's raw
        ``north``, which is wrong: every grid on disk was cropped to
        ``FRAME_ORIGINS`` first, and the two differ by up to ~700 pixel rows —
        so the line landed far from the requested latitude and train/val were
        not actually split where they claimed to be.

        Patches straddling the line belong to neither side. The old arithmetic
        assigned every row to one side or the other, which let a patch spanning
        the line sit in both.
        """
        H, W = patch_size
        meta = coord_dict.get(intf_id)
        if meta is None or meta.get("frame") not in FRAME_ORIGINS:
            raise KeyError(f"{intf_id} has no usable frame; cannot place the spatial split line")

        ref = np.load(path_of("data", intf_id, nonz=False), mmap_mode="r")
        ny = ref.shape[0]
        is_train = self.mode == "train"
        # Above the line for train, below it for val/test; the straddling rows
        # fall out of both windows and are dropped.
        side = (thresh_lat, 90.0) if is_train else (-90.0, thresh_lat)
        r0, r1, _, _ = grid_window(
            meta["frame"], side[0], side[1],
            patch_size=(H, W), stride=(H // 2, W // 2),
        )
        r0, r1 = max(0, r0), min(r1, ny)
        def load_sliced(kind, tid):
            arr = np.load(path_of(kind, tid, nonz=False)).astype(np.float32)
            lo, hi = min(r0, arr.shape[0]), min(r1, arr.shape[0])
            return arr[lo:hi]

        image_data = load_sliced("data", intf_id)
        mask_data = load_sliced("mask", intf_id)

        if self.temporal:
            tids = list(self.seq_dict[intf_id]["prevs"]) + [intf_id]
            img_pa = [load_sliced("data", tid) for tid in tids]
            msk_pa = [load_sliced("mask", tid) for tid in tids]
            # nonz_indices hold pre-slice grid rows; convert to post-slice.
            # Both sides now slice from r0, so the offset is r0 for train too —
            # it just happens to be 0 whenever the window starts at the top.
            offsets = {
                tid: min(r0, np.load(path_of("data", tid, nonz=False), mmap_mode="r").shape[0])
                for tid in tids
            }
            offsets["_arrays"] = (img_pa, msk_pa)
            return self._load_temporal(
                intf_id, path_of, patch_size, nonz_indices, negatives,
                treat_nodata, union_mask, row_offsets=offsets,
            )

        image_data = image_data.reshape(-1, image_data.shape[2], image_data.shape[3])
        mask_data = mask_data.reshape(-1, mask_data.shape[2], mask_data.shape[3])
        if nonz_only:
            keep = [(m > 0).any() for m in mask_data]
            image_data = image_data[np.array(keep, dtype=bool)]
            mask_data = mask_data[np.array(keep, dtype=bool)]
        return image_data, mask_data

    @staticmethod
    def _add_null_patches(image_data, mask_data):
        """Positives plus a random sample of hard empty patches (no-data streaks)."""
        import random as _random

        keep_img, keep_msk = [], []
        for img, msk in zip(image_data, mask_data):
            positive = bool(np.any(msk > 0))
            add_null = (
                not positive
                and _random.choice([True, False])
                and np.sum(img == 0) < 1000
                and has_consecutive_zeros(img)
            )
            if positive or add_null:
                keep_img.append(img)
                keep_msk.append(msk)
        return np.array(keep_img), np.array(keep_msk)

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
        self.n_negative = int((~(mask_data > 0).any(axis=(-2, -1))).sum())
        uniques = np.unique(mask_data)
        self.mask_values = [int(v) if float(v).is_integer() else float(v) for v in sorted(uniques.tolist())]
        return self

    # -- sample access (reads only pickle-stable attributes) ---------------------------

    def __len__(self):
        return len(self.index_map)

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

    def __getitem__(self, sample):
        intf_idx, patch_idx = self.index_map[sample]
        if getattr(self, "temporal", False):
            img = self.image_data[intf_idx][:, patch_idx].astype(np.float32)  # (T|2T, H, W)
            msk = self.mask_data[intf_idx][0, patch_idx].astype(np.float32)
        else:
            img = self.image_data[intf_idx][0, patch_idx].astype(np.float32)  # (H, W)
            msk = self.mask_data[intf_idx][0, patch_idx].astype(np.float32)

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
    """Pickle a test split; returns its size in GB (it can be large)."""
    buffer = io.BytesIO()
    pickle.dump(dataset, buffer)
    size_gb = buffer.tell() / (1024**3)
    with open(path, "wb") as fh:
        fh.write(buffer.getvalue())
    return size_gb
