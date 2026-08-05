"""Reconcile the coordinate dictionary, the .ers headers and the patch tree.

Run after a patch regeneration to confirm the three agree. Every check is
read-only. Exits non-zero if any check fails, so it can gate a pipeline.

    python scripts/verify_dataset.py \
        --intf_dict assets/intf_coord.json \
        --scene_dir /home/labs/rudich/pinkas/scenes_11day \
        --patches_root /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches \
        --days_diff 11

``--gt_polygon_file_path`` additionally reports how many interferograms carry
ground-truth polygons (needs geopandas; skipped when not given).
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinkholes.dataprep.patchify import patch_dir_name, patch_file_name  # noqa: E402
from sinkholes.meta import intf_id_from_filename, parse_ers_header  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)
    return ok


def span_days(intf_id):
    s = datetime.strptime(intf_id[:8], "%Y%m%d")
    e = datetime.strptime(intf_id[9:], "%Y%m%d")
    return (e - s).days


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intf_dict", required=True)
    p.add_argument("--scene_dir", required=True, help="directory of .unw/.ers scenes")
    p.add_argument("--patches_root", required=True, help="root holding the patch trees")
    p.add_argument("--patch_size", nargs=2, type=int, default=[200, 100], metavar=("H", "W"))
    p.add_argument("--strides_per_patch", type=int, default=2)
    p.add_argument("--days_diff", type=int, default=11)
    p.add_argument("--gt_polygon_file_path", default=None)
    args = p.parse_args()

    H, W = args.patch_size
    meta = json.load(open(args.intf_dict))

    # ---- 1. the dictionary itself -------------------------------------------
    print("\n1. coordinate dictionary")
    print(f"   {args.intf_dict}: {len(meta)} entries")
    wrong_span = [i for i in meta if span_days(i) != args.days_diff]
    check(f"every entry spans {args.days_diff} days", not wrong_span,
          f"{len(wrong_span)} others, e.g. {wrong_span[:3]}" if wrong_span else "")
    required = {"east", "north", "dx", "dy", "ncells", "nlines", "byte_order", "frame"}
    incomplete = [i for i, v in meta.items() if not required <= set(v)]
    check("all entries have the full geometry fields", not incomplete,
          f"{len(incomplete)} incomplete" if incomplete else "")
    frames = Counter(v.get("frame") for v in meta.values())
    check("frames are North/South only", set(frames) <= {"North", "South"}, str(dict(frames)))
    no_lidar = [i for i, v in meta.items() if v.get("lidar_mask", "no_mask") == "no_mask"]
    print(f"   note: {len(no_lidar)} entries have lidar_mask='no_mask' "
          f"(only matters for --add_lidar_mask at eval time)")

    # ---- 2. dictionary vs the scenes ----------------------------------------
    print("\n2. dictionary vs .ers headers")
    scene_ids = {}
    for fn in sorted(os.listdir(args.scene_dir)):
        if fn.endswith(".ers"):
            scene_ids[intf_id_from_filename(fn)] = os.path.join(args.scene_dir, fn)
    scenes = {i for i in scene_ids if span_days(i) == args.days_diff}
    print(f"   {args.scene_dir}: {len(scene_ids)} headers ({len(scenes)} at "
          f"{args.days_diff} days)")
    check("dictionary covers every scene", not (scenes - set(meta)),
          f"{len(scenes - set(meta))} scenes unregistered")
    check("no dictionary entry lacks a scene", not (set(meta) - scenes),
          f"{len(set(meta) - scenes)} entries with no scene")

    mismatched = []
    for i in sorted(scenes & set(meta)):
        h = parse_ers_header(scene_ids[i])
        for k in ("ncells", "nlines", "byte_order"):
            if h[k] != meta[i][k]:
                mismatched.append((i, k, h[k], meta[i][k]))
                break
    check("geometry matches the headers", not mismatched,
          f"{len(mismatched)} differ, e.g. {mismatched[:2]}" if mismatched else
          f"{len(scenes & set(meta))} checked")

    # ---- 3. the patch tree ---------------------------------------------------
    print("\n3. patch tree")
    ddir = os.path.join(args.patches_root,
                        patch_dir_name("data", H, W, args.strides_per_patch, args.days_diff))
    mdir = os.path.join(args.patches_root,
                        patch_dir_name("mask", H, W, args.strides_per_patch, args.days_diff))
    print(f"   {ddir}")
    if not os.path.isdir(ddir) or not os.path.isdir(mdir):
        check("patch directories exist", False, "data or mask tree missing")
        return finish()

    def ids_in(d, nonz):
        out = set()
        for fn in os.listdir(d):
            if not fn.endswith(".npy"):
                continue
            if ("_nonz_" in fn) != nonz:
                continue
            if "_cleaned" in fn:
                continue
            out.add(fn.split("_patches_")[1].split("_H")[0].replace("nonz_", ""))
        return out

    d_full, d_nonz = ids_in(ddir, False), ids_in(ddir, True)
    m_full, m_nonz = ids_in(mdir, False), ids_in(mdir, True)
    check("data full == data nonz ids", d_full == d_nonz,
          f"{len(d_full ^ d_nonz)} unpaired")
    check("data ids == mask ids", d_full == m_full, f"{len(d_full ^ m_full)} unpaired")
    check("mask full == mask nonz ids", m_full == m_nonz, f"{len(m_full ^ m_nonz)} unpaired")
    check("patch ids == dictionary ids", d_full == set(meta),
          f"patches-only={len(d_full - set(meta))}, dict-only={len(set(meta) - d_full)}")

    # ---- 4. nonz counts ------------------------------------------------------
    print("\n4. positive-patch counts")
    ni_path = os.path.join(ddir, "nonz_indices.json")
    ni = json.load(open(ni_path)) if os.path.exists(ni_path) else {}
    check("nonz_indices.json exists", bool(ni), ni_path)
    check("nonz_indices covers every patched id", set(ni) == d_full,
          f"missing={len(d_full - set(ni))}, extra={len(set(ni) - d_full)}")

    bad_ni, bad_meta, empty, dm = [], [], [], []
    for i in sorted(d_full):
        n_d = np.load(os.path.join(ddir, patch_file_name(
            "data", i, H, W, args.strides_per_patch, nonz=True)), mmap_mode="r").shape[0]
        n_m = np.load(os.path.join(mdir, patch_file_name(
            "mask", i, H, W, args.strides_per_patch, nonz=True)), mmap_mode="r").shape[0]
        if n_d != n_m:
            dm.append((i, n_d, n_m))
        if n_d == 0:
            empty.append(i)
        if i in ni and len(ni[i]) != n_d:
            bad_ni.append((i, len(ni[i]), n_d))
        mv = meta.get(i, {}).get("nonz_num", "none")
        if (mv if isinstance(mv, int) else 0) != n_d:
            bad_meta.append((i, mv, n_d))

    check("data and mask nonz lengths agree", not dm, f"{len(dm)} differ, e.g. {dm[:2]}")
    check("nonz_indices lengths match the arrays", not bad_ni,
          f"{len(bad_ni)} differ, e.g. {bad_ni[:2]}")
    check("dictionary nonz_num matches the arrays", not bad_meta,
          f"{len(bad_meta)} differ, e.g. {bad_meta[:3]}")
    print(f"   {len(empty)} interferograms have 0 positive patches "
          f"(expected: the unlabelled ones)")
    print(f"   total positive patches: {sum(len(v) for v in ni.values())}")

    # ---- 5. against the ground truth ----------------------------------------
    if args.gt_polygon_file_path:
        print("\n5. ground truth")
        import geopandas as gpd

        gdf = gpd.read_file(args.gt_polygon_file_path)
        pairs = Counter(zip(gdf["start_date"].dt.strftime("%Y-%m-%d"),
                            gdf["end_date"].dt.strftime("%Y-%m-%d")))
        labelled, objects = set(), 0
        for i in d_full:
            n = pairs.get((f"{i[:4]}-{i[4:6]}-{i[6:8]}", f"{i[9:13]}-{i[13:15]}-{i[15:17]}"), 0)
            if n:
                labelled.add(i)
                objects += n
        print(f"   {len(labelled)}/{len(d_full)} interferograms labelled, {objects} objects")
        lab_empty = sorted(labelled & set(empty))
        check("no labelled interferogram has an empty mask", not lab_empty,
              f"{len(lab_empty)}: {lab_empty[:5]}")
        unlab_nonempty = sorted((d_full - labelled) - set(empty))
        check("no unlabelled interferogram has positive patches", not unlab_nonempty,
              f"{len(unlab_nonempty)}: {unlab_nonempty[:5]}")

    return finish()


def finish():
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
