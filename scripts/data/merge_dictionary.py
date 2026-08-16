"""Merge a scoped coordinate dictionary into a broader one.

The shared dictionary in the data directory spans every interferogram duration,
while a regeneration produces a dictionary for one duration only. This folds
the regenerated entries into the shared file and leaves every other entry
untouched, so the shared copy reflects newly processed interferograms without
losing the durations the regeneration never looked at.

Reports what would change and writes nothing unless ``--apply`` is given.

    python scripts/data/merge_dictionary.py \
        --base /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/intf_coord.json \
        --update assets/intf_coord.json --apply
"""

import argparse
import json
import math
import os
import shutil
from datetime import datetime

#: Geometry fields that must agree for two entries to describe the same raster.
_GEOM_FIELDS = ("east", "north", "dx", "dy", "ncells", "nlines", "byte_order", "frame")


def _same(a, b) -> bool:
    """Field equality, tolerating JSON float round-trip noise.

    Coordinates round-trip through JSON at ~1e-15 relative precision, so
    31.807581685 and 31.807581685000002 are the same origin. The tolerance is
    many orders of magnitude below one pixel (dx = 2.777e-05 degrees).
    """
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="dictionary to merge into (the broader one)")
    p.add_argument("--update", required=True, help="dictionary to merge from (the regenerated one)")
    p.add_argument("--out_path", default=None, help="default: overwrite --base")
    p.add_argument("--apply", action="store_true", help="write; otherwise report only")
    p.add_argument("--no_backup", action="store_true",
                   help="skip the timestamped copy of --base taken before writing")
    args = p.parse_args()

    base = json.load(open(args.base))
    update = json.load(open(args.update))
    out_path = args.out_path or args.base

    added = sorted(set(update) - set(base))
    common = sorted(set(update) & set(base))
    untouched = sorted(set(base) - set(update))

    changed, identical, geom_changed = [], [], []
    for i in common:
        keys = set(base[i]) | set(update[i])
        if all(_same(base[i].get(k), update[i].get(k)) for k in keys):
            identical.append(i)
            continue
        changed.append(i)
        if any(not _same(base[i].get(k), update[i].get(k)) for k in _GEOM_FIELDS):
            geom_changed.append(i)

    print(f"base   {args.base}: {len(base)} entries")
    print(f"update {args.update}: {len(update)} entries")
    print()
    print(f"  added (new to base)      : {len(added)}")
    print(f"  updated (values differ)  : {len(changed)}")
    print(f"     of which geometry too : {len(geom_changed)}")
    print(f"  already identical        : {len(identical)}")
    print(f"  left untouched in base   : {len(untouched)}")
    print(f"  -> result                : {len(base) + len(added)} entries")

    if geom_changed:
        print(f"\n!! geometry differs for {len(geom_changed)} shared ids — the two "
              f"dictionaries describe different raster copies:")
        for i in geom_changed[:5]:
            print(f"     {i}: base ncells/nlines="
                  f"{base[i].get('ncells')}/{base[i].get('nlines')} vs update "
                  f"{update[i].get('ncells')}/{update[i].get('nlines')}")
        raise SystemExit("refusing to merge across different scene copies")

    if not args.apply:
        print("\nreport only; pass --apply to write")
        return

    if not args.no_backup and os.path.exists(out_path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{out_path}.{stamp}.bak"
        shutil.copyfile(out_path, backup)
        print(f"\nbacked up {out_path} -> {backup}")

    merged = dict(base)
    merged.update(update)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(merged, fh, indent=4)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}: {len(merged)} entries")


if __name__ == "__main__":
    main()
