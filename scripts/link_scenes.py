"""Build a single-copy scene directory for prepare-metadata / prepare-patches.

The raw scene tree holds several thousand headers: ids of every duration, two
processing variants per id (``tgeo_int_*`` and ``tgeo_ccw_*``), and frame-split
copies under 004/ and 013/ whose raster extents differ from the root copies.
Both ``prepare-metadata`` and ``prepare-patches`` take a single ``--input_dir``
and process everything in it, so they need one directory holding exactly one
file per interferogram.

This symlinks the chosen variant of the chosen duration into ``--out_dir``
(nothing is copied; re-running is idempotent).

    python scripts/link_scenes.py \
        --data_dir /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data \
        --out_dir /home/labs/rudich/pinkas/scenes_11day \
        --days_diff 11 --variant int
"""

import argparse
import glob
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinkholes.geo import FRAME_ORIGINS  # noqa: E402
from sinkholes.meta import intf_id_from_filename, parse_ers_header  # noqa: E402


def span_days(intf_id):
    s = datetime.strptime(intf_id[:8], "%Y%m%d")
    e = datetime.strptime(intf_id[9:], "%Y%m%d")
    return (e - s).days


def alignment_offsets(header):
    """(row_off, col_off) of the frame origin within this scene, or None if the
    scene cannot be aligned to it.

    prepare-patches crops every scene to its frame's fixed origin so patch
    (i, j) covers the same ground across dates. A scene whose raster starts
    south or east of that origin has no such crop and raises in
    crop_to_start_xy — hours into a run, since the failure is per scene.
    """
    frame = "North" if header["north"] > 31.6 else "South"
    x_star, y_star = FRAME_ORIGINS[frame]
    col_off = round((x_star - header["east"]) / header["dx"])
    row_off = round((header["north"] - y_star) / header["dy"])
    if not (0 <= col_off < header["ncells"]) or not (0 <= row_off < header["nlines"]):
        return None
    return row_off, col_off


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="directory holding the raw scenes")
    p.add_argument("--out_dir", required=True, help="scene directory to build")
    p.add_argument("--days_diff", type=int, default=11,
                   help="keep only interferograms of exactly this duration (0 = all)")
    p.add_argument("--variant", default="int", choices=("int", "ccw"),
                   help="processing variant to link (default: int, the one the "
                        "committed coordinate dictionaries describe)")
    p.add_argument("--clear", action="store_true",
                   help="remove any existing links in --out_dir first")
    p.add_argument("--dry_run", action="store_true", help="report only, link nothing")
    p.add_argument("--keep_unalignable", action="store_true",
                   help="link scenes that cannot be cropped to their frame origin "
                        "(they will make prepare-patches raise; off by default)")
    args = p.parse_args()

    pattern = os.path.join(args.data_dir, f"tgeo_{args.variant}_*.unw.ers")
    headers = sorted(glob.glob(pattern))
    if not headers:
        raise SystemExit(f"no headers matched {pattern}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.clear and not args.dry_run:
        for f in os.listdir(args.out_dir):
            path = os.path.join(args.out_dir, f)
            if os.path.islink(path):
                os.remove(path)

    spans = Counter()
    linked, missing_unw, seen, unalignable = 0, [], {}, []
    for ers in headers:
        intf_id = intf_id_from_filename(os.path.basename(ers))
        try:
            days = span_days(intf_id)
        except ValueError:
            print(f"  skip (unparseable id): {os.path.basename(ers)}")
            continue
        spans[days] += 1
        if args.days_diff and days != args.days_diff:
            continue
        unw = ers[:-4]
        if not os.path.exists(unw):
            missing_unw.append(intf_id)
            continue
        if not args.keep_unalignable:
            header = parse_ers_header(ers)
            if alignment_offsets(header) is None:
                frame = "North" if header["north"] > 31.6 else "South"
                unalignable.append((intf_id, frame, header["north"]))
                continue
        if intf_id in seen:
            print(f"  duplicate id {intf_id}: keeping {os.path.basename(seen[intf_id])}, "
                  f"skipping {os.path.basename(ers)}")
            continue
        seen[intf_id] = ers
        if not args.dry_run:
            for src in (unw, ers):
                dst = os.path.join(args.out_dir, os.path.basename(src))
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
        linked += 1

    print(f"\n{len(headers)} '{args.variant}' headers in {args.data_dir}")
    print(f"durations present: {dict(sorted(spans.items()))}")
    if missing_unw:
        print(f"!! {len(missing_unw)} headers with no .unw: {missing_unw[:5]}")
    if unalignable:
        print(f"skipped {len(unalignable)} scene(s) that cannot be cropped to their "
              f"frame origin (use --keep_unalignable to link them anyway):")
        for intf_id, frame, north in unalignable:
            print(f"    {intf_id}  {frame:5s} north={north} "
                  f"(origin {FRAME_ORIGINS[frame][1]})")
    verb = "would link" if args.dry_run else "linked"
    print(f"{verb} {linked} interferograms"
          f"{f' at {args.days_diff} days' if args.days_diff else ''} -> {args.out_dir}")

    if not args.dry_run:
        n_unw = len(glob.glob(os.path.join(args.out_dir, "*.unw")))
        n_ers = len(glob.glob(os.path.join(args.out_dir, "*.unw.ers")))
        broken = [f for f in os.listdir(args.out_dir)
                  if os.path.islink(os.path.join(args.out_dir, f))
                  and not os.path.exists(os.path.join(args.out_dir, f))]
        print(f"directory now holds {n_unw} .unw and {n_ers} .ers, "
              f"{len(broken)} broken links")
        if n_unw != linked or n_ers != linked or broken:
            raise SystemExit("!! directory contents do not match what was linked")


if __name__ == "__main__":
    main()
