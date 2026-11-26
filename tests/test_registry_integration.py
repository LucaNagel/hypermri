"""Integration tests for direct plotting with real data."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

import hypermri

matplotlib.use("Agg")


REAL_DATASET = "brukerapi_pv601_lego_phantom"


@pytest.mark.integration
def test_real_csi_uses_direct_viewer(real_dataset_path):
    """Verify real CSI scans construct the direct viewer on request."""
    data_path: Path = real_dataset_path(REAL_DATASET)
    csi = hypermri.create_experiment(data_path / "30", load_data=True)

    assert "_plotter" not in vars(csi)
    assert csi.plot() is not None
