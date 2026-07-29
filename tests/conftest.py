"""Pytest configuration for the sinkholes test suite.

EDIT 2026-07-28: new file. CHANGELOG.md #6
EDIT 2026-07-29: modules moved from the repo root into ``src/`` subfolders, so the
repo root alone no longer resolves them. Delegate to ``src/_bootstrap.py``, which
puts every ``src/*`` folder on ``sys.path`` and keeps the flat imports
(``from convlstm_unet import ...``) working here and in the scripts alike.

Under pytest's default (`prepend`) import mode only the directory holding the test file
lands on `sys.path`, so `import convlstm_unet` would fail without this.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import _bootstrap  # noqa: E402,F401
