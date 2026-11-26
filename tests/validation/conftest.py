"""Opt-in validation tests for the validation code in ``scripts``.

The modules tested here are validation and exploration code that is not part of
the installed package.  ``scripts/hooks/run_tests.py`` collects only
``tests/unit``, so these tests never run in the unit gate.  Run them explicitly
from the repository root::

    PYTHONPATH=src MPLBACKEND=Agg python -m pytest tests/validation -o addopts= -q

Every test collected from this directory carries the ``validation`` marker, so
a run over the whole ``tests`` directory can exclude them with
``-m "not validation"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

VALIDATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_DIR.parents[1]

# ``scripts`` is importable from the repository root only; this also holds
# when pytest is started without ``python -m``.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    """Mark every test collected from this directory as a validation test."""
    for item in items:
        if VALIDATION_DIR in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.validation)
