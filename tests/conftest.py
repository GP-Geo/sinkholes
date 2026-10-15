"""Make the in-repo ``sinkholes`` package importable without installation."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _in_process_dataloaders(monkeypatch):
    """Load batches in the test process instead of a worker per loader.

    ``train_model`` builds its loaders with ``num_workers=1``. On macOS every
    iterator then spawns a fresh interpreter that re-imports torch and the
    package -- seconds per loader per epoch, over SMB more -- and that was most
    of the suite's runtime. The tiny in-memory test sets need no worker.
    """
    import torch.utils.data as tud

    from sinkholes.training import train

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return tud.DataLoader(*args, **kwargs)

    monkeypatch.setattr(train, "DataLoader", loader)


_SOURCE_FINGERPRINTS = {}


@pytest.fixture(autouse=True)
def _cached_source_fingerprint(monkeypatch):
    """Hash the package source once per session, not once per ``train_model``.

    Every execution writes a provenance record that reads every package ``.py``
    and runs three ``git`` commands; the code cannot change inside one test
    session, so the first answer is the answer.
    """
    from sinkholes.training import provenance

    compute = provenance.source_fingerprint

    def cached(root=provenance.PACKAGE_ROOT):
        key = str(root)
        if key not in _SOURCE_FINGERPRINTS:
            _SOURCE_FINGERPRINTS[key] = compute(root)
        return dict(_SOURCE_FINGERPRINTS[key])

    monkeypatch.setattr(provenance, "source_fingerprint", cached)
