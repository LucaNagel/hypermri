"""Visualization integration test using the public BrukerAPI phantom corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

import hypermri
from hypermri.visualization import PlotOptions, plot

REAL_DATASET = "brukerapi_pv601_lego_phantom"


@pytest.mark.integration
def test_real_csi_plot_initializes(real_dataset_path):
    """Check that the browser viewer opens on real CSI data, with and without anatomy."""
    data_path: Path = real_dataset_path(REAL_DATASET)
    csi = hypermri.create_experiment(data_path / "30", load_data=True)
    anatomy = hypermri.create_experiment(data_path / "47", load_data=True)

    csi_plot = plot(csi)
    overlay_plot = plot(csi, PlotOptions(anatomy=anatomy))

    assert csi_plot is not None
    assert overlay_plot is not None
