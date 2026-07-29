"""Fill ``nonz_num`` (positive-patch counts) into the coordinate dictionary.

Run as ``sinkholes count-positives`` after generating patches — training skips
any interferogram whose ``nonz_num`` is ``'none'``, so a stale dictionary
silently excludes data. Reads the counts off the ``*_nonz_*`` files' shapes;
interferograms without a nonz file in the patch directory are set to ``'none'``.

Reads and writes are explicit paths (``--intf_dict`` in, ``--out_path`` out;
they may be the same file).
"""

import argparse
import json
import logging
import os

import numpy as np

from ..meta import INTF_ID_RE
from ..paths import asset


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input_patch_dir", type=str, required=True,
                   help="a data_patches_* directory holding *_nonz_* files")
    p.add_argument("--intf_dict", type=str, default=None,
                   help="coordinate dictionary to update (default: the committed asset)")
    p.add_argument("--out_path", type=str, default=None,
                   help="where to write the updated dictionary (default: overwrite --intf_dict)")


def main(args) -> None:
    logging.basicConfig(level=logging.INFO)
    dict_path = args.intf_dict or asset("intf_coord.json")
    with open(dict_path) as fh:
        meta = json.load(fh)

    for entry in meta.values():
        entry.setdefault("nonz_num", "none")

    counted = 0
    for name in sorted(os.listdir(args.input_patch_dir)):
        if "nonz" not in name or not name.endswith(".npy"):
            continue
        m = INTF_ID_RE.search(name)
        if not m or m.group(0) not in meta:
            continue
        intf_id = m.group(0)
        n = np.load(os.path.join(args.input_patch_dir, name), mmap_mode="r").shape[0]
        region = "north" if meta[intf_id]["north"] > 31.5 else "south"
        logging.info(f"{intf_id} ({region}): {n} positive patches")
        meta[intf_id]["nonz_num"] = int(n)
        counted += 1

    out_path = args.out_path or dict_path
    with open(out_path, "w") as fh:
        json.dump(meta, fh, indent=4)
    logging.info(f"wrote {out_path}: {counted} interferograms counted, "
                 f"{sum(1 for e in meta.values() if e['nonz_num'] == 'none')} set to 'none'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
