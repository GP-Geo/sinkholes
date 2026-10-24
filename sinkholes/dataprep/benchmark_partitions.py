"""Generate the benchmark partitions described in assets/PARTITIONS.md.

Writes four files -- ``partition_temporal_k{5,10}_<suffix>.json`` and a
``_testeval`` variant of each -- carrying, besides the interferogram lists:

``aoi_window``
    The lat/lon box each split owns. Read by ``dataprep/dataset.py``,
    ``inference/scenes.py`` and ``inference/outputs.py``; see
    :func:`sinkholes.geo.grid_window` for why one source matters.
``provenance``
    Everything needed to explain the file without this script.

Two rules the generator exists to enforce, both easy to get wrong by hand:

* **Positives are counted inside the window**, never from ``nonz_num`` (which is
  whole-scene). A partition adopted on whole-scene counts would claim ground it
  does not own.
* **k10 is a subset of k5 per split.** Splits are assigned at k5 and intersected
  with the k10 chain-valid set, so the two depths differ in chain length and
  nothing else.
* **``--nonz_th`` gates targets only.** A scene below the threshold never enters
  train/val/test, but stays available as a *predecessor*: the chains are built
  over the whole dictionary before the filter runs, so a thinly labelled scene
  can still supply an input frame. Label quality is judged WHOLE-SCENE
  (``nonz_num``) and not inside the AOI on purpose -- an in-window count would
  confuse "badly digitised" with "few sinkholes in this band". Since an
  in-window count can never exceed the whole-scene one, the filter cannot drop
  a scene that has more than the threshold's worth of positives in the window.
* **``--temporal_blocks`` separates fully.** A block layout holds out date
  blocks and trains on everything around them, so chains cross set boundaries
  in both directions. Every target's inputs must then be targets of its own
  set or of none: train scenes that feed a held-out block, or take one as
  input, are dropped (:func:`separate_fully`). The ``--temporal_bounds`` rule
  is weaker -- a test chain may reach into val -- and stays as it was.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import date

import numpy as np

from ..geo import grid_window
from ..meta import FILL_LOOKBACK_FACTOR, find_11day_sequences, load_coord_dict
from .partition import filter_by_nonz_count

#: Written into every file so a reader can tell the generation apart. ``_fill``
#: marks chains that reach past missing acquisitions (meta.find_11day_sequences);
#: the committed ``all_years_clean`` files were built with gap-free chains only.
GENERATION = "all_years_clean_fill"


def generation_label(years, nonz_th=None):
    """The generation stamp for this run.

    A ``--years``-restricted run is *not* :data:`GENERATION`. Stamping it
    ``all_years_clean`` would make the year-restricted family indistinguishable
    from the full one in the only place a reader is told to trust
    (assets/PARTITIONS.md: "trust the file's own provenance"). A
    ``--nonz_th``-restricted run is a separate family for the same reason: it
    holds fewer scenes than the bare name promises.
    """
    label = GENERATION if not years else f"{years[0]}_{years[1]}_clean_fill"
    return label if not nonz_th else f"{label}_nonz{nonz_th[0]}-{nonz_th[1]}"


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--intf_dict", type=str, default=None,
                   help="coordinate dictionary (default: the committed asset)")
    p.add_argument("--patches_dir", type=str, required=True,
                   help="root holding the data_patches_* tree (for nonz_indices.json)")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--strides_per_patch", type=int, default=2)
    p.add_argument("--aoi", nargs=4, type=float, default=[31.25, 31.75, 35.38, 35.46],
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    p.add_argument("--temporal_bounds", nargs=2, type=str, default=["20240101", "20250101"],
                   metavar=("TRAIN_END", "VAL_END"),
                   help="train < first, val < second, test >= second "
                        "(ignored when --temporal_blocks is given)")
    p.add_argument("--temporal_blocks", nargs="+", type=str, default=None,
                   metavar="SPLIT:START:END",
                   help="held-out date blocks, e.g. val:20220101:20231231 test:20260101:20261231 "
                        "(inclusive, on the interferogram's start date). Everything outside "
                        "them is train, and every target's 11-day chain must lie in its own "
                        "set: train scenes next to a block are dropped. Replaces "
                        "--temporal_bounds")
    p.add_argument("--k_prevs", nargs="+", type=int, default=[5, 10])
    p.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1"),
                   help="optional year filter; default is every year in the dictionary")
    p.add_argument("--nonz_th", nargs=2, type=int, default=None, metavar=("NORTH", "SOUTH"),
                   help="per-region whole-scene positive-patch threshold for TARGETS: a scene "
                        "enters train/val/test only when nonz_num > this (north when the frame "
                        "origin is above lat 31.5, else south). Predecessors are exempt. "
                        "Default: no threshold (generations 4 and 5 used 350 200)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="assets")
    p.add_argument("--suffix", type=str, default="clean_fill",
                   help="file name suffix: partition_temporal_k<k>_<suffix>.json")
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


def drop_leaky_chains(splits, chains, bounds, k):
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
            logging.info(f"temporal_k{k} {split}: dropped {len(dropped)} scene(s) whose "
                         f"k={k} chain reaches before {bounds[0]}: {sorted(dropped)}")
        out[split] = sorted(kept)
    return out


def parse_temporal_blocks(specs):
    """``SPLIT:YYYYMMDD:YYYYMMDD`` strings -> [(split, start, end)] by start date."""
    blocks = []
    for spec in specs:
        try:
            split, a, b = spec.split(":")
            start, end = (date(int(x[:4]), int(x[4:6]), int(x[6:8])) for x in (a, b))
        except ValueError:
            raise SystemExit(f"--temporal_blocks: {spec!r} is not SPLIT:YYYYMMDD:YYYYMMDD")
        if split not in ("val", "test"):
            raise SystemExit(f"--temporal_blocks: {spec!r} -- split must be val or test; "
                             f"everything outside the blocks is train")
        if end < start:
            raise SystemExit(f"--temporal_blocks: {spec!r} ends before it starts")
        blocks.append((split, start, end))
    blocks.sort(key=lambda blk: blk[1])
    for (_, _, end), (split, start, _) in zip(blocks, blocks[1:]):
        if start <= end:
            raise SystemExit(f"--temporal_blocks: the {split} block starting {start} "
                             f"overlaps the block before it")
    return blocks


def separate_fully(splits, chains, label=""):
    """Make every target's chain belong to its own split, in every direction.

    A target's ``prevs`` may be targets of its own split or of none, never of
    another. Held-out blocks are kept whole and train pays: a train scene is
    dropped when it is an input of a held-out target (the lead-in before a
    block) or takes a held-out target as input (the lead-out after it). A
    held-out scene whose inputs are targets of the OTHER held-out split is
    dropped too -- that only happens when a val and a test block sit closer
    than one chain.
    """
    def prevs(intf):
        return set(chains.get(intf, {}).get("prevs", []))

    held = {i: s for s in ("val", "test") for i in splits.get(s, [])}
    crossed = sorted(i for i, s in held.items() if any(held.get(p, s) != s for p in prevs(i)))
    for i in crossed:
        del held[i]
    feeds_held = set().union(*(prevs(i) for i in held))
    train, dropped = [], []
    for intf in splits["train"]:
        (dropped if intf in feeds_held or prevs(intf) & set(held) else train).append(intf)
    if crossed:
        logging.info(f"{label}: dropped {len(crossed)} held-out scene(s) whose chain holds "
                     f"the other held-out split: {crossed}")
    if dropped:
        logging.info(f"{label}: dropped {len(dropped)} train scene(s) whose chain touches "
                     f"a held-out block: {sorted(dropped)}")
    out = dict(splits)
    out["train"] = sorted(train)
    for s in ("val", "test"):
        out[s] = sorted(i for i, x in held.items() if x == s)
    return out


def block_split(valid, chains, blocks, aoi, label=""):
    """Train = everything outside the held-out blocks, fully separated."""
    splits = {"train": [], "val": [], "test": []}
    for intf in valid:
        s = start_date(intf)
        splits[next((sp for sp, a, b in blocks if a <= s <= b), "train")].append(intf)
    box = tuple(aoi)
    return separate_fully(splits, chains, label), {"train": box, "val": box, "test": box}


def check_full_separation(label, splits, chains):
    """Fail loudly if any target takes an input that is another split's target."""
    owner = {i: s for s, v in splits.items() for i in v}
    for intf, s in owner.items():
        foreign = sorted(p for p in chains.get(intf, {}).get("prevs", [])
                         if owner.get(p, s) != s)
        if foreign:
            raise SystemExit(f"{label}: {s} scene {intf} takes inputs that are "
                             f"{owner[foreign[0]]} targets: {foreign}")


def testeval_variant(splits, windows):
    """The parent's test list carried under 'val', as the convention requires."""
    out = {"train": list(splits["train"]) + list(splits["val"]), "val": list(splits["test"])}
    win = {"train": windows["train"], "val": windows["test"]}
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


def check_invariants(label, splits):
    """Fail loudly rather than ship a partition that leaks."""
    lists = {s: set(v) for s, v in splits.items()}
    for a in lists:
        for b in lists:
            if a < b and lists[a] & lists[b]:
                raise SystemExit(
                    f"{label}: {a} and {b} share interferograms: "
                    f"{sorted(lists[a] & lists[b])}"
                )


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    coord = load_coord_dict(args.intf_dict)
    if args.years:
        y0, y1 = args.years
        coord = {i: m for i, m in coord.items() if y0 <= int(i[:4]) <= y1}
    nonz = load_nonz_indices(args.patches_dir, args.patch_size, args.strides_per_patch)
    counter = PositiveCounter(nonz, coord, args.patch_size, args.strides_per_patch)
    aoi = tuple(args.aoi)
    blocks = parse_temporal_blocks(args.temporal_blocks) if args.temporal_blocks else None
    os.makedirs(args.out_dir, exist_ok=True)

    ks = sorted(args.k_prevs, reverse=True)          # largest first, for the subset check
    chains_by_k, valid_by_k = {}, {}
    for k in ks:
        ch, valid = find_11day_sequences(coord, k_prev=k)
        chains_by_k[k], valid_by_k[k] = ch, sorted(valid)
        logging.info(f"k={k}: {len(valid)} chain-valid interferograms")
        if args.nonz_th:
            # Targets only. ``ch`` is deliberately left whole: it was built over
            # the entire dictionary just above, so a scene dropped here still
            # appears in a kept scene's ``prevs`` and still supplies an input
            # frame. Filtering the chains too would silently shorten every stack
            # that reaches back across a thinly labelled date.
            kept = filter_by_nonz_count(valid_by_k[k], coord, tuple(args.nonz_th))
            dropped = len(valid_by_k[k]) - len(kept)
            valid_by_k[k] = sorted(kept)
            logging.info(f"k={k}: --nonz_th {args.nonz_th[0]} {args.nonz_th[1]} dropped "
                         f"{dropped} of {len(valid)} as targets; they stay available "
                         f"as predecessors")

    # Splits are assigned at the SHALLOWEST chain depth and the deeper ones are
    # intersected with it, which is what makes k10 a subset of k5 per split --
    # so the two depths differ in chain length and in nothing else.
    k_base = min(ks)
    written = []
    base_splits = base_windows = None
    for k in sorted(ks):
        label = f"temporal_k{k}"
        valid = valid_by_k[k]
        if blocks:
            # Assigned once, at the shallowest depth: the deeper ones are
            # intersected with it below and re-separated on their own chains.
            if k == k_base:
                splits, windows = block_split(valid, chains_by_k[k], blocks, aoi, label)
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
            if blocks:
                # Full separation re-checked on the deeper chains: a k10
                # chain reaches further than a k5 one, so the gaps around
                # each block widen. Only train shrinks unless two held-out
                # blocks sit within one chain of each other.
                splits = separate_fully(splits, chains_by_k[k], label)
            else:
                # Chain containment must be re-checked at the deeper k, and
                # it OUTRANKS the subset rule. A scene whose 5-step chain
                # starts after the val boundary can still have a 10-step
                # chain reaching back into train -- three do, at these
                # bounds -- and keeping it would put a training target into
                # a held-out sample's input. Dropping is safe for the subset
                # rule because it only ever removes.
                splits = drop_leaky_chains(splits, chains_by_k[k], args.temporal_bounds, k)

        check_invariants(label, splits)
        if blocks:
            check_full_separation(label, splits, chains_by_k[k])

        prov = {
            "generation": generation_label(args.years, args.nonz_th),
            "axis": "temporal",
            "k_prevs": k,
            "max_lookback": FILL_LOOKBACK_FACTOR * k,
            # ``years`` is what the splits actually contain; ``years_filter``
            # is what was asked for. They differ when the archive is thinner
            # than the request, so both are recorded.
            "years": sorted({int(i[:4]) for v in splits.values() for i in v}) or None,
            "years_filter": list(args.years) if args.years else None,
            # Target-side only; predecessors were never filtered by it.
            "nonz_th": list(args.nonz_th) if args.nonz_th else None,
            "aoi": list(aoi),
            "temporal_bounds": list(args.temporal_bounds) if not blocks else None,
            "temporal_blocks": ([f"{sp}:{a:%Y%m%d}:{b:%Y%m%d}" for sp, a, b in blocks]
                                if blocks else None),
            # "full": no target takes an input that is another split's
            # target. "held_out_chains": held-out chains never reach into
            # train, but a test chain may reach into val.
            "separation": "full" if blocks else "held_out_chains",
            "seed": args.seed,
            "positives_in_window": {s: counter.total(v, windows[s])
                                    for s, v in splits.items()},
            "intfs": {s: len(v) for s, v in splits.items()},
            "generated_by": "sinkholes make-benchmark-partitions",
        }
        if prov["years"]:
            prov["years"] = [prov["years"][0], prov["years"][-1]]

        name = f"partition_temporal_k{k}_{args.suffix}.json"
        write_partition(os.path.join(args.out_dir, name), splits, windows, prov, args.dry_run)
        written.append((name, splits, windows, prov))

        te_splits, te_windows = testeval_variant(splits, windows)
        te_prov = dict(prov, derived_from=name,
                       note="parent's test list carried under 'val'",
                       intfs={s: len(v) for s, v in te_splits.items()},
                       positives_in_window={s: counter.total(v, te_windows[s])
                                            for s, v in te_splits.items()})
        te_name = f"partition_temporal_k{k}_testeval_{args.suffix}.json"
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
