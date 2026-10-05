"""``sinkholes relabel <step>`` -- the test-set relabelling workflow (docs/RELABELING.md).

Steps, in the order they are used:

  init      build the workspace (original + working GeoPackages, rasters, manifest)
  validate  check the working GT (dates, geometry, scene, duplicates, ...)
  diff      detect added / deleted / modified polygons; stable change ids
  export    Output A: the corrected GT (gpkg + shp)
  review    Output B: the change-review package for an external reviewer
  eval      run: re-score saved predictions on corrected labels; compare: tables
"""

from importlib import import_module

_STEPS = {
    "init": "sinkholes.relabel.workspace",
    "validate": "sinkholes.relabel.validate",
    "diff": "sinkholes.relabel.diff",
    "export": "sinkholes.relabel.export",
    "review": "sinkholes.relabel.review",
    "eval": "sinkholes.relabel.evaluate",
}


def add_arguments(p):
    sub = p.add_subparsers(dest="step", required=True, metavar="step")
    for name, mod in _STEPS.items():
        doc = (import_module(mod).__doc__ or "").strip().splitlines()[0].split(" -- ", 1)[-1]
        import_module(mod).add_arguments(sub.add_parser(name, help=doc, description=doc))


def main(args):
    import_module(_STEPS[args.step]).main(args)
