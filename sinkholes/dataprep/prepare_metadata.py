"""Build the interferogram coordinate dictionary from the .ers headers, and
merge the per-year LiDAR coverage shapefiles.

Run as ``sinkholes prepare-metadata`` (one-off per data drop; the result is
committed as ``assets/intf_coord.json``) and ``sinkholes merge-lidar`` (one-off;
result committed as ``assets/lidar_mask_polygs.shp``).
"""

import argparse
import json
import logging
import os


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--intf_dir", type=str, required=True, help="directory of the .ers headers")
    p.add_argument("--out_path", type=str, default="intf_coord.json")
    p.add_argument("--lidar_mapping", type=str, default=None,
                   help="lidar_intf_mask.txt (default: the committed asset)")


def main(args) -> None:
    from ..meta import intf_id_from_filename, lidar_source_for, parse_ers_header

    logging.basicConfig(level=logging.INFO)
    intf_dict = {}
    for filename in sorted(os.listdir(args.intf_dir)):
        if not filename.endswith(".ers"):
            continue
        intf_id = intf_id_from_filename(filename)
        entry = parse_ers_header(os.path.join(args.intf_dir, filename))
        entry["lidar_mask"] = lidar_source_for(intf_id, args.lidar_mapping)
        if entry["lidar_mask"] == "no_mask":
            logging.warning(f"no LiDAR mask mapping for {intf_id}")
        # Frame from the scene origin: North scenes start around latitude
        # 31.79-31.82, South scenes around 31.45-31.47 — cleanly separated.
        entry["frame"] = "North" if entry["north"] > 31.6 else "South"
        intf_dict[intf_id] = entry
        logging.info(f"{intf_id}: {entry['frame']} {entry['ncells']}x{entry['nlines']}")

    with open(args.out_path, "w") as fh:
        json.dump(intf_dict, fh, indent=4)
    logging.info(f"wrote {args.out_path} ({len(intf_dict)} interferograms). "
                 f"Run 'sinkholes count-positives' after patch generation to fill nonz_num.")


def add_merge_lidar_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shapefiles", nargs="+", required=True,
                   metavar="SOURCE=PATH",
                   help="per-year shapefiles as source=path pairs, "
                        "e.g. LiDAR2019=LiDAR2019_polyg.shp")
    p.add_argument("--out_path", type=str, default="lidar_mask_polygs.shp")


def merge_lidar_main(args) -> None:
    import geopandas as gpd
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    parts = []
    for pair in args.shapefiles:
        source, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"expected SOURCE=PATH, got {pair!r}")
        gdf = gpd.read_file(path)
        gdf["source"] = source
        parts.append(gdf)
    merged = pd.concat(parts, ignore_index=True)
    merged.to_file(args.out_path)
    logging.info(f"wrote {args.out_path}: {len(merged)} polygons from {len(parts)} sources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
