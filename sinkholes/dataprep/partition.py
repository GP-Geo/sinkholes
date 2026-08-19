"""Which interferograms exist, which are usable, and how they split.

Four partition modes:

- ``random_by_patch``  — random split over individual patches. Overlapping
  patches leak across the splits (stride-2 patches share half their pixels),
  so metrics are optimistic; kept for continuity.
- ``random_by_intf``   — whole interferograms per split (recommended).
- ``spatial``          — a latitude line inside each interferogram separates
  train (north of it) from val/test (south); handled by the dataset itself.
- ``preset_by_intf``   — a saved partition JSON ({"train": [...], "val": [...]}).
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..meta import INTF_ID_RE

#: The fixed 2021 temporal hold-out (--preset_test_val_21): train on the rest,
#: validate and test on these.
PRESET_21_VAL = ["20210407_20210418", "20210418_20210429", "20210304_20210315", "20210326_20210406"]
PRESET_21_TEST = ["20210806_20210817", "20210623_20210704", "20210622_20210703", "20210714_20210725"]


def discover_intf_ids(patch_dir, *, nonz: bool) -> List[str]:
    """Interferogram ids present in a patch directory.

    ``nonz`` selects between the positive-only files and the full grids —
    they can differ, since scenes without positives have an empty nonz file
    but a real grid.
    """
    ids = []
    for name in os.listdir(patch_dir):
        if not name.endswith(".npy") or ("nonz" in name) != nonz:
            continue
        m = INTF_ID_RE.search(name)
        if m:
            ids.append(m.group(0))
    return sorted(set(ids))


def filter_by_nonz_count(
    intf_ids: Sequence[str],
    coord_dict: Dict[str, Any],
    nonz_th: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Drop ids without patches on disk (nonz_num == 'none'), and optionally
    ids below a per-region positive-patch threshold (north, south) — the
    region boundary is latitude 31.5."""
    out = [i for i in intf_ids if coord_dict[i]["nonz_num"] != "none"]
    if nonz_th is not None:
        th_north, th_south = nonz_th
        out = [
            i for i in out
            if coord_dict[i]["nonz_num"] > (th_north if coord_dict[i]["north"] > 31.5 else th_south)
        ]
    return out


def split_random_by_intf(
    intf_ids: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    exclude_test: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle whole interferograms into train/val/test lists.

    ``exclude_test`` pins the test list to a previous run's interferograms and
    splits only train/val over the rest. The shuffle uses the global ``random``
    state, so seed it (``--seed``) for a reproducible split.
    """
    ids = list(intf_ids)
    if exclude_test is None:
        random.shuffle(ids)
        n_val = int(len(ids) * val_fraction)
        n_test = int(len(ids) * test_fraction)
        if n_val == 0:
            raise SystemExit("not enough interferograms to carve out a validation split")
        n_train = len(ids) - n_val - n_test
        return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]

    test_list = list(exclude_test)
    rest = list(set(ids) - set(test_list))
    random.shuffle(rest)
    n_val = int(len(ids) * val_fraction)
    if n_val == 0:
        raise SystemExit("not enough interferograms to carve out a validation split")
    n_train = len(rest) - n_val
    return rest[:n_train], rest[n_train:], test_list


def split_preset_21(intf_ids: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """The fixed 2021 hold-out: everything else trains."""
    train = list(set(intf_ids) - set(PRESET_21_TEST) - set(PRESET_21_VAL))
    return train, list(PRESET_21_VAL), list(PRESET_21_TEST)


def load_preset_partition(path) -> Tuple[List[str], List[str]]:
    """(train, val) lists from a saved partition JSON."""
    with open(path) as fh:
        data = json.load(fh)
    return data["train"], data["val"]


#: Keys a partition JSON may hold as interferogram lists. Everything else in
#: the file is provenance (``derived_from``, ``val_samples``, ...).
#:
#: ``crossview`` is the geo memorisation probe of the plan's section 5b: the
#: North band below the 31.4 deg cut, the same ground the South hold-out
#: covers, seen from the other frame. It shares most of its interferograms
#: with ``train`` -- the one deliberate breach of one-interferogram-one-split
#: in the benchmark -- so it is EVALUATION ONLY. Training must never load it,
#: it must never select a checkpoint, and it must never be averaged into test.
PARTITION_SPLITS = ("train", "val", "test", "crossview")

#: Splits a model may be trained or checkpoint-selected on.
TRAINABLE_SPLITS = ("train", "val")

#: Key holding the per-split lat/lon box, when the partition carries one.
AOI_WINDOW_KEY = "aoi_window"


def load_partition_split(path, split: str = "val") -> List[str]:
    """One named interferogram list from a partition JSON.

    Training reads only train/val (:func:`load_preset_partition`) and drops the
    ``test`` list, so evaluation is where it is finally read. A missing key is
    an error rather than an empty list: the ``*_testeval.json`` variants hold a
    parent's test list under ``"val"`` and have no ``"test"`` of their own, and
    scoring the wrong ground silently is exactly what must not happen.
    """
    if split not in PARTITION_SPLITS:
        raise SystemExit(f"unknown split {split!r}; expected one of {', '.join(PARTITION_SPLITS)}")
    with open(path) as fh:
        data = json.load(fh)
    if split not in data:
        available = [k for k in PARTITION_SPLITS if k in data]
        raise SystemExit(f"{path} has no '{split}' list (it holds: {', '.join(available)}). "
                         f"The _testeval partitions carry their parent's test list as 'val'.")
    return list(data[split])


def load_partition_window(path, split: str = "val"):
    """The (lat_min, lat_max, lon_min, lon_max) box a split is restricted to.

    Returns ``None`` when the partition carries no window — the pre-AOI
    generations, which are scored on the whole canvas. Callers must treat
    ``None`` as "no restriction" rather than substituting a default, so an old
    partition keeps reproducing its old numbers.

    The box travels inside the partition file on purpose: the three consumers
    that must agree on it (``dataprep/dataset.py``, ``inference/scenes.py``,
    ``inference/outputs.py``) then read it from one source instead of each
    taking a flag that could be passed inconsistently.
    """
    if split not in PARTITION_SPLITS:
        raise SystemExit(f"unknown split {split!r}; expected one of {', '.join(PARTITION_SPLITS)}")
    with open(path) as fh:
        data = json.load(fh)
    windows = data.get(AOI_WINDOW_KEY)
    if not windows:
        return None
    if split not in windows:
        raise SystemExit(
            f"{path} has an '{AOI_WINDOW_KEY}' but no entry for split {split!r} "
            f"(it holds: {', '.join(sorted(windows))}). Scoring a split on the wrong "
            f"ground is exactly what this key exists to prevent, so this is an error."
        )
    box = windows[split]
    if len(box) != 4:
        raise SystemExit(
            f"{path}: {AOI_WINDOW_KEY}[{split!r}] must be "
            f"[lat_min, lat_max, lon_min, lon_max], got {box!r}"
        )
    return tuple(float(v) for v in box)


def add_make_partition_arguments(p) -> None:
    p.add_argument("--patches_dir", type=str, required=True,
                   help="root holding the data_patches_* tree")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--strides_per_patch", type=int, default=2)
    p.add_argument("--days_diff", type=int, default=11)
    p.add_argument("--val_percent", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_path", type=str, default=None,
                   help="partition JSON to write (default: partition_<ts>.json)")


def make_partition_main(args) -> None:
    """Write a train/val partition JSON, splitting whole interferograms so the
    validation side holds ~``--val_percent`` of the positive patches.

    Run as ``sinkholes make-partition``; the result feeds
    ``--partition_mode preset_by_intf --partition_file <json>``.
    """
    import logging
    import os
    from datetime import datetime

    from .patchify import patch_dir_name, patch_file_name

    logging.basicConfig(level=logging.INFO)
    if args.seed is not None:
        random.seed(args.seed)
    H, W = args.patch_size
    in_path = os.path.join(args.patches_dir,
                           patch_dir_name("data", H, W, args.strides_per_patch, args.days_diff))
    if not os.path.exists(in_path):
        raise SystemExit(f"patch directory not found: {in_path}")

    ids = discover_intf_ids(in_path, nonz=False)
    random.shuffle(ids)
    logging.info(f"{len(ids)} interferograms, shuffled")

    counts = [
        np.load(os.path.join(in_path, patch_file_name("data", i, H, W, args.strides_per_patch,
                                                      nonz=True)), mmap_mode="r").shape[0]
        for i in ids
    ]
    total = sum(counts)
    cut = int(np.where(np.cumsum(counts) / total >= (100 - args.val_percent) / 100)[0][0])
    partition = {"train": ids[: cut + 1], "val": ids[cut + 1 :]}

    out_path = args.out_path or f"partition_{datetime.now().strftime('%d_%m_%Hh%M')}.json"
    with open(out_path, "w") as fh:
        json.dump(partition, fh, indent=4)
    val_share = sum(counts[cut + 1 :]) / total
    logging.info(f"wrote {out_path}: {len(partition['train'])} train / "
                 f"{len(partition['val'])} val interferograms "
                 f"({100 * val_share:.1f}% of positive patches in val)")


def split_nonoverlap_patches(
    image_grid: np.ndarray,
    mask_grid: np.ndarray,
    test_fraction: float,
    min_grid_gap: int = 2,
):
    """Patch-level split of one (ny, nx, H, W) grid without stride overlap.

    Positive patches are sampled into the test set; a training patch is kept
    only when its grid coordinate is at Chebyshev distance >= ``min_grid_gap``
    from every test patch, so no training pixel appears in a test patch. The
    default gap of 2 is correct for stride 2 (50% overlap).

    Returns (train_images, train_masks, test_images, test_masks) as (N, H, W).
    """
    nz_patches, nz_masks, nz_indices = [], [], []
    for i in range(image_grid.shape[0]):
        for j in range(image_grid.shape[1]):
            if np.any(mask_grid[i, j] > 0):
                nz_patches.append(image_grid[i, j])
                nz_masks.append(mask_grid[i, j])
                nz_indices.append((i, j))

    n_test = int(test_fraction * len(nz_indices))
    test_picks = sorted(random.sample(range(len(nz_indices)), n_test))
    test_set = {nz_indices[p] for p in test_picks}

    train_images, train_masks = [], []
    for n, (i, j) in enumerate(nz_indices):
        if not any(abs(ti - i) < min_grid_gap and abs(tj - j) < min_grid_gap for (ti, tj) in test_set):
            train_images.append(nz_patches[n])
            train_masks.append(nz_masks[n])

    test_images = [nz_patches[p] for p in test_picks]
    test_masks = [nz_masks[p] for p in test_picks]
    return (
        np.array(train_images), np.array(train_masks),
        np.array(test_images), np.array(test_masks),
    )
