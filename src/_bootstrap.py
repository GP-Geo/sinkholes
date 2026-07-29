"""Keeps the repo's flat imports working after the move into ``src/`` subfolders.

Every module here still imports its neighbours by bare name (``from unet import UNet``,
``from device_utils import get_device``) regardless of which subfolder they live in.
Python only puts the *running script's* directory on ``sys.path``, so importing this
module adds every ``src/*`` folder to the path and makes those bare imports resolve.

Scripts pick it up via a three-line header (see any entry point in ``src/``)::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import _bootstrap  # noqa: F401,E402

Importing it more than once is harmless. The repo root goes on the path too, so
``assets/`` and ``data/`` stay reachable by relative path when a script is launched
from the root.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent


def _prepend(path: Path) -> None:
    entry = str(path)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


for _pkg in sorted(_SRC.iterdir()):
    if _pkg.is_dir() and not _pkg.name.startswith((".", "_")):
        _prepend(_pkg)

if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

REPO_ROOT = _ROOT
SRC_ROOT = _SRC
ASSETS = _ROOT / "assets"


def asset(name: str) -> str:
    """Absolute path to a committed data asset in ``assets/``.

    These files (``intf_coord.json``, ``lidar_mask_polygs.shp``, the partition JSONs)
    used to sit at the repo root and were opened by bare relative name, which only
    worked when the script was launched from there. Resolving them here makes the
    scripts runnable from any working directory.
    """
    return str(ASSETS / name)
