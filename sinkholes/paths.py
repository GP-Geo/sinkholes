"""Locations of the committed data assets and default output roots.

Everything the pipeline opens by name rather than by a CLI argument resolves
through here, so commands work from any working directory. The repo layout is
assumed (assets/ next to the package); ``SINKHOLES_ASSETS`` overrides it for
installs that keep the assets elsewhere.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = Path(os.environ.get("SINKHOLES_ASSETS", REPO_ROOT / "assets"))

#: Default root for training runs and full-scene prediction outputs.
DEFAULT_OUTPUTS_DIR = "outputs"
DEFAULT_PREDICTIONS_DIR = "outputs/predictions"

#: The WEXAC data root most defaults point at.
WEXAC_DATA_DIR = "/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/"


def asset(name: str) -> str:
    """Absolute path of a committed asset (intf_coord.json, lidar_mask_polygs.shp, ...)."""
    path = ASSETS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"asset {name!r} not found under {ASSETS_DIR} "
            f"(set SINKHOLES_ASSETS if the assets live elsewhere)"
        )
    return str(path)
