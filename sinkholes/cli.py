"""The one command-line surface: ``sinkholes <command>`` (or ``python -m sinkholes``).

Each subcommand lives in its own module exposing ``add_arguments(parser)`` and
``main(args)``; heavy imports (torch, geopandas) happen inside ``main`` so
``sinkholes --help`` stays instant.
"""

import argparse
import sys
from importlib import import_module

#: command -> (module, add_arguments attr, main attr, help)
_COMMANDS = {
    # data preparation
    "prepare-metadata": ("sinkholes.dataprep.prepare_metadata", "add_arguments", "main",
                         "parse .ers headers into the interferogram coordinate dictionary"),
    "merge-lidar": ("sinkholes.dataprep.prepare_metadata", "add_merge_lidar_arguments",
                    "merge_lidar_main",
                    "merge per-year LiDAR coverage shapefiles into one"),
    "prepare-patches": ("sinkholes.dataprep.prepare_patches", "add_arguments", "main",
                        "cut .unw scenes + GT polygons into aligned patch grids"),
    "count-positives": ("sinkholes.dataprep.count_positives", "add_arguments", "main",
                        "fill nonz_num counts into the coordinate dictionary"),
    "clean-patches": ("sinkholes.dataprep.clean_patches", "add_arguments", "main",
                      "drop no-data / edge-sliver mask polygons (-> cleaned/)"),
    "make-partition": ("sinkholes.dataprep.partition", "add_make_partition_arguments",
                       "make_partition_main",
                       "write a reproducible train/val partition JSON"),
    "make-benchmark-partitions": ("sinkholes.dataprep.benchmark_partitions", "add_arguments",
                                  "main",
                                  "write the geo/temporal benchmark partitions (AOI + cut)"),
    "prepare-local-subset": ("sinkholes.dataprep.local_subset", "add_arguments", "main",
                             "derive nonz files + corrected metadata for a partial download"),
    # training
    "train": ("sinkholes.training.train", "add_arguments", "main",
              "train a segmentation model"),
    # evaluation / inference
    "test-patches": ("sinkholes.inference.patch_test", "add_arguments", "main",
                     "patch-level metrics on a pickled split or a partition JSON"),
    "eval-scenes": ("sinkholes.inference.scenes", "add_arguments", "main",
                    "full-scene reconstruction, polygons and saved arrays"),
    "eval-outputs": ("sinkholes.inference.outputs", "add_arguments", "main",
                     "metrics and figures over saved eval-scenes outputs"),
    "predict": ("sinkholes.inference.predict", "add_arguments", "main",
                "predict polygons on new raw .unw scenes (no ground truth)"),
    "inspect-run": ("sinkholes.inference.inspect_run", "add_arguments", "main",
                    "learning curve + threshold sweep for a finished run"),
    "attention-probe": ("sinkholes.inference.attention_probe", "add_arguments", "main",
                        "where a tattn_unet's attention lands when given a long, gappy history"),
    "curves": (None, None, None, "regenerate curves.png for a finished run"),
    "architectures": (None, None, None, "list the registered model architectures"),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="sinkholes",
        description="Dead Sea sinkhole detection in InSAR interferograms.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    parsers = {}
    for name, (module, _, _, help_text) in _COMMANDS.items():
        parsers[name] = sub.add_parser(name, help=help_text)

    parsers["curves"].add_argument("run_dir", help="a training run directory")
    # The two trivial commands are handled inline; the rest defer to modules.

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        parser.print_help()
        raise SystemExit(1)
    command = argv[0]
    if command in ("-h", "--help"):
        parser.print_help()
        return
    if command not in _COMMANDS:
        parser.print_help()
        raise SystemExit(f"unknown command {command!r}")

    if command == "architectures":
        from .models.factory import describe

        print("Registered architectures (match order):")
        print(describe())
        return

    if command == "curves":
        args = parsers["curves"].parse_args(argv[1:])
        from pathlib import Path

        from .training.reporter import plot_curves

        png = plot_curves(Path(args.run_dir) / "results.csv", Path(args.run_dir) / "curves.png")
        print(f"wrote {png}" if png else f"nothing to plot in {args.run_dir}")
        raise SystemExit(0 if png else 1)

    module_name, add_args_attr, main_attr, help_text = _COMMANDS[command]
    module = import_module(module_name)
    cmd_parser = argparse.ArgumentParser(prog=f"sinkholes {command}",
                                         description=help_text)
    getattr(module, add_args_attr)(cmd_parser)
    args = cmd_parser.parse_args(argv[1:])
    getattr(module, main_attr)(args)


if __name__ == "__main__":
    main()
