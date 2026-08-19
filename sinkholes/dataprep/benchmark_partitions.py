"""Generate the benchmark partitions described in docs/PLAN_CLEAN_BENCHMARK.md.

Writes eight files -- ``partition_{geo,temporal}_k{5,10}_clean.json`` and a
``_testeval`` variant of each -- carrying, besides the interferogram lists:

``aoi_window``
    The lat/lon box each split owns. Read by ``dataprep/dataset.py``,
    ``inference/scenes.py`` and ``inference/outputs.py``; see
    :func:`sinkholes.geo.grid_window` for why one source matters.
``crossview`` (geo only)
    The North band below the cut -- section 5b's memorisation probe. It shares
    interferograms with ``train`` on purpose and is evaluation-only.
``provenance``
    Everything needed to explain the file without this script.

Two rules the generator exists to enforce, both easy to get wrong by hand:

* **Positives are counted inside the window**, never from ``nonz_num`` (which is
  whole-scene). A partition adopted on whole-scene counts would claim ground it
  does not own.
* **k10 is a subset of k5 per split.** Splits are assigned at k5 and intersected
  with the k10 chain-valid set, so the two depths differ in chain length and
  nothing else.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import date

import numpy as np

from ..geo import grid_window
from ..meta import find_11day_sequences, load_coord_dict

#: Written into every file so a reader can tell the generation apart.
GENERATION = "all_years_clean"


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--intf_dict", type=str, default=None,
                   help="coordinate dictionary (default: the committed asset)")
    p.add_argument("--patches_dir", type=str, required=True,
                   help="root holding the data_patches_* tree (for nonz_indices.json)")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--strides_per_patch", type=int, default=2)
    p.add_argument("--cut_lat", type=float, default=31.4,
                   help="geo split line; train = North above it, hold-out = South below")
    p.add_argument("--aoi", nargs=4, type=float, default=[31.25, 31.75, 35.38, 35.46],
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    p.add_argument("--geo_val_share", type=float, default=0.4,
                   help="fraction of the geo hold-out used for val (rest is test)")
    p.add_argument("--temporal_bounds", nargs=2, type=str, default=["20240101", "20250101"],
                   metavar=("TRAIN_END", "VAL_END"),
                   help="train < first, val < second, test >= second")
    p.add_argument("--k_prevs", nargs="+", type=int, default=[5, 10])
    p.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1"),
                   help="optional year filter; default is every year in the dictionary")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="assets")
    p.add_argument("--suffix", type=str, default="clean",
                   help="file name suffix: partition_<axis>_k<k>_<suffix>.json")
    p.add_argument("--dry_run", action="store_true",
                   help="print the tables without writing anything")


# ---------------------------------------------------------------- counting


def load_nonz_indices(patches_dir, patch_size, strides_per_patch):
    from .patchify import patch_dir_name

    name = patch_dir_name("data", patch_size[0], patch_size[1], strides_per_patch,
                          days_diff=11)
    path = os.path.join(patches_dir, name, "nonz_indices.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. The generator counts positives inside the AOI window, "
            f"which needs the per-scene grid coordinates -- nonz_num is whole-scene and "
            f"would overstate every split."
        )
    with open(path) as fh:
        return json.load(fh)


class PositiveCounter:
    """Positives of one interferogram inside a lat/lon box, on the patch grid."""

    def __init__(self, nonz_indices, coord, patch_size, strides_per_patch):
        self.nonz = nonz_indices
        self.coord = coord
        self.patch_size = tuple(patch_size)
        self.stride = (patch_size[0] // strides_per_patch,
                       patch_size[1] // strides_per_patch)
        self._cache = {}

    def frame(self, intf):
        return self.coord[intf]["frame"]

    def count(self, intf, box):
        key = (intf, box)
        if key in self._cache:
            return self._cache[key]
        rc = self.nonz.get(intf)
        if not rc:
            self._cache[key] = 0
            return 0
        r0, r1, c0, c1 = grid_window(self.frame(intf), *box,
                                     patch_size=self.patch_size, stride=self.stride)
        a = np.asarray(rc, dtype=int)
        n = int(((a[:, 0] >= r0) & (a[:, 0] < r1)
                 & (a[:, 1] >= c0) & (a[:, 1] < c1)).sum())
        self._cache[key] = n
        return n

    def total(self, intfs, box):
        return sum(self.count(i, box) for i in intfs)


# ---------------------------------------------------------------- splitting


def start_date(intf):
    return date(int(intf[:4]), int(intf[4:6]), int(intf[6:8]))


def geo_split(valid, counter, cut_lat, aoi, val_share, seed):
    """Train = North above the cut, hold-out = South below it, split val/test.

    The hold-out is divided by *positive mass* rather than by scene count, so
    the val share means what it says whatever the per-scene spread is.
    """
    la, lb, lc, ld = aoi
    train_box = (cut_lat, lb, lc, ld)
    hold_box = (la, cut_lat, lc, ld)

    north = [i for i in valid if counter.frame(i) == "North"]
    south = [i for i in valid if counter.frame(i) == "South"]

    train = sorted(i for i in north if counter.count(i, train_box) > 0)
    crossview = sorted(i for i in north if counter.count(i, hold_box) > 0)
    hold = sorted(i for i in south if counter.count(i, hold_box) > 0)

    # Greedy fill of the val side up to the requested share of positives, over a
    # seeded shuffle so the choice is reproducible but not date-ordered (taking
    # the earliest scenes would make val a different season from test).
    rng = np.random.default_rng(seed)
    order = list(hold)
    rng.shuffle(order)
    target = val_share * counter.total(hold, hold_box)
    val, running = [], 0
    for intf in order:
        if running >= target:
            break
        val.append(intf)
        running += counter.count(intf, hold_box)
    val = sorted(val)
    test = sorted(set(hold) - set(val))
    return {"train": train, "val": val, "test": test, "crossview": crossview}, \
           {"train": train_box, "val": hold_box, "test": hold_box, "crossview": hold_box}


def temporal_split(valid, chains, counter, bounds, aoi, k):
    """Train before the first bound, val before the second, test after.

    Chain containment: a held-out interferogram is kept only when its whole
    chain starts on or after the *val* boundary, so no held-out sample's input
    was ever a training target. Chains reaching from test back into val are
    allowed -- val is held out too, so nothing leaks from training.
    """
    b1, b2 = (date(int(b[:4]), int(b[4:6]), int(b[6:8])) for b in bounds)
    train, val, test = [], [], []
    for intf in valid:
        s = start_date(intf)
        if s < b1:
            train.append(intf)
            continue
        prevs = chains.get(intf, {}).get("prevs", [])
        earliest = min([start_date(p) for p in prevs] + [s])
        if earliest < b1:
            continue                      # chain reaches into train -> drop
        (val if s < b2 else test).append(intf)
    box = tuple(aoi)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}, \
           {"train": box, "val": box, "test": box}


# ---------------------------------------------------------------- writing


def drop_leaky_chains(splits, chains, bounds, axis, k):
    """Remove held-out scenes whose chain at this depth reaches into train."""
    b1 = date(int(bounds[0][:4]), int(bounds[0][4:6]), int(bounds[0][6:8]))
    out = dict(splits)
    for split in ("val", "test"):
        kept, dropped = [], []
        for intf in splits.get(split, []):
            prevs = chains.get(intf, {}).get("prevs", [])
            earliest = min([start_date(p) for p in prevs] + [start_date(intf)])
            (kept if earliest >= b1 else dropped).append(intf)
        if dropped:
            logging.info(f"{axis}_k{k} {split}: dropped {len(dropped)} scene(s) whose "
                         f"k={k} chain reaches before {bounds[0]}: {sorted(dropped)}")
        out[split] = sorted(kept)
    return out


def testeval_variant(splits, windows):
    """The parent's test list carried under 'val', as the convention requires."""
    out = {"train": list(splits["train"]) + list(splits["val"]), "val": list(splits["test"])}
    win = {"train": windows["train"], "val": windows["test"]}
    if "crossview" in splits:
        out["crossview"] = list(splits["crossview"])
        win["crossview"] = windows["crossview"]
    return out, win


def write_partition(path, splits, windows, provenance, dry_run=False):
    doc = {k: list(v) for k, v in splits.items()}
    doc["aoi_window"] = {k: list(v) for k, v in windows.items()}
    doc["provenance"] = provenance
    if dry_run:
        return
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=4)
        fh.write("\n")


def check_invariants(axis, k, splits, counter, windows):
    """Fail loudly rather than ship a partition that leaks."""
    lists = {s: set(v) for s, v in splits.items() if s != "crossview"}
    for a in lists:
        for b in lists:
            if a < b and lists[a] & lists[b]:
                raise SystemExit(
                    f"{axis}_k{k}: {a} and {b} share interferograms: "
                    f"{sorted(lists[a] & lists[b])}"
                )
    if axis == "geo":
        # No train patch may sit inside the hold-out band.
        for intf in splits["train"]:
            if counter.count(intf, windows["test"]) and counter.frame(intf) == "South":
                raise SystemExit(f"{axis}_k{k}: train scene {intf} has positives in the hold-out band")
        for intf in splits["test"]:
            if counter.frame(intf) != "South":
                raise SystemExit(f"{axis}_k{k}: test scene {intf} is not on the South frame")


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    coord = load_coord_dict(args.intf_dict)
    if args.years:
        y0, y1 = args.years
        coord = {i: m for i, m in coord.items() if y0 <= int(i[:4]) <= y1}
    nonz = load_nonz_indices(args.patches_dir, args.patch_size, args.strides_per_patch)
    counter = PositiveCounter(nonz, coord, args.patch_size, args.strides_per_patch)
    aoi = tuple(args.aoi)
    os.makedirs(args.out_dir, exist_ok=True)

    ks = sorted(args.k_prevs, reverse=True)          # largest first, for the subset check
    chains_by_k, valid_by_k = {}, {}
    for k in ks:
        ch, valid = find_11day_sequences(coord, k_prev=k)
        chains_by_k[k], valid_by_k[k] = ch, sorted(valid)
        logging.info(f"k={k}: {len(valid)} chain-valid interferograms")

    # Splits are assigned at the SHALLOWEST chain depth and the deeper ones are
    # intersected with it, which is what makes k10 a subset of k5 per split --
    # so the two depths differ in chain length and in nothing else.
    k_base = min(ks)
    written = []
    for axis in ("geo", "temporal"):
        base_splits = base_windows = None
        for k in sorted(ks):
            valid = valid_by_k[k]
            if axis == "geo":
                splits, windows = geo_split(valid, counter, args.cut_lat, aoi,
                                            args.geo_val_share, args.seed)
            else:
                splits, windows = temporal_split(valid, chains_by_k[k], counter,
                                                 args.temporal_bounds, aoi, k)
            if k == k_base:
                base_splits, base_windows = splits, windows
            else:
                # k10 subset of k5, per split: assign once, then intersect.
                valid_set = set(valid)
                splits = {s: sorted(set(v) & valid_set) for s, v in base_splits.items()}
                windows = base_windows
                if axis == "temporal":
                    # Chain containment must be re-checked at the deeper k, and
                    # it OUTRANKS the subset rule. A scene whose 5-step chain
                    # starts after the val boundary can still have a 10-step
                    # chain reaching back into train -- three do, at these
                    # bounds -- and keeping it would put a training target into
                    # a held-out sample's input. Dropping is safe for the subset
                    # rule because it only ever removes.
                    splits = drop_leaky_chains(splits, chains_by_k[k],
                                               args.temporal_bounds, axis, k)

            check_invariants(axis, k, splits, counter, windows)

            prov = {
                "generation": GENERATION,
                "axis": axis,
                "k_prevs": k,
                "years": sorted({int(i[:4]) for v in splits.values() for i in v}) or None,
                "cut_lat": args.cut_lat if axis == "geo" else None,
                "aoi": list(aoi),
                "temporal_bounds": list(args.temporal_bounds) if axis == "temporal" else None,
                "geo_val_share": args.geo_val_share if axis == "geo" else None,
                "seed": args.seed,
                "positives_in_window": {s: counter.total(v, windows[s])
                                        for s, v in splits.items()},
                "intfs": {s: len(v) for s, v in splits.items()},
                "generated_by": "sinkholes make-benchmark-partitions",
            }
            if prov["years"]:
                prov["years"] = [prov["years"][0], prov["years"][-1]]

            name = f"partition_{axis}_k{k}_{args.suffix}.json"
            write_partition(os.path.join(args.out_dir, name), splits, windows, prov, args.dry_run)
            written.append((name, splits, windows, prov))

            te_splits, te_windows = testeval_variant(splits, windows)
            te_prov = dict(prov, derived_from=name,
                           note="parent's test list carried under 'val'",
                           intfs={s: len(v) for s, v in te_splits.items()},
                           positives_in_window={s: counter.total(v, te_windows[s])
                                                for s, v in te_splits.items()})
            te_name = f"partition_{axis}_k{k}_testeval_{args.suffix}.json"
            write_partition(os.path.join(args.out_dir, te_name), te_splits, te_windows,
                            te_prov, args.dry_run)
            written.append((te_name, te_splits, te_windows, te_prov))

    logging.info("")
    logging.info(f"{'file':>46} | {'split':>10} {'intfs':>6} {'positives in window':>20}")
    logging.info("-" * 92)
    for name, splits, windows, prov in written:
        for s in splits:
            logging.info(f"{name if s == list(splits)[0] else '':>46} | {s:>10} "
                         f"{len(splits[s]):6d} {prov['positives_in_window'][s]:20,}")
    logging.info("")
    logging.info(f"{'(dry run -- nothing written)' if args.dry_run else 'wrote'} "
                 f"{len(written)} files to {args.out_dir}")
