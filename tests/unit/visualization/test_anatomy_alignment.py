"""The disabled reference-RPS viewer alignment; standalone RPS geometry controls remain in utils."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from hypermri.visualization.render_data import PlotDataModel
from hypermri.visualization.state import PlotState


@pytest.mark.parametrize("with_sources", [False, True])
def test_deferred_alignment_rejects_model_construction_without_geometry_reads(
    monkeypatch, with_sources
):
    from hypermri.visualization import render_data as model_module

    compute = Mock(side_effect=AssertionError("Deferred alignment computed geometry"))
    monkeypatch.setattr(model_module, "compute_scan_alignment", compute)
    source = object() if with_sources else None
    with pytest.raises(
        NotImplementedError, match="anatomy_align=True is not available"
    ):
        PlotDataModel(source, anatomy_obj=source, anatomy_align=True)
    compute.assert_not_called()


def test_alignment_cannot_be_reenabled_by_mutating_the_model_flag(monkeypatch):
    from hypermri.visualization import render_data as model_module

    compute = Mock(side_effect=AssertionError("Deferred alignment computed geometry"))
    monkeypatch.setattr(model_module, "compute_scan_alignment", compute)
    model = PlotDataModel()
    model.anatomy_align = True
    with pytest.raises(
        NotImplementedError, match="anatomy_align=True is not available"
    ):
        model._get_alignment()
    compute.assert_not_called()


def test_ordinary_plot_and_cache_invalidation_preserve_data_without_rps_alignment(
    monkeypatch,
):
    from hypermri.visualization import render_data as model_module

    # Invented complex samples and mm axes, with no scanner acquisition involved.
    data = xr.DataArray(
        (np.arange(6).reshape(2, 3, 1) + 1j).astype(np.complex64),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": [0.0, 1.0],
            "phase_idx": [0.0, 2.0, 4.0],
            "slice_idx": [0.0],
        },
        attrs={"units": "a.u.", "invented_fixture": True},
    )
    before = data.copy(deep=True)
    compute = Mock(
        side_effect=AssertionError("Ordinary plotting computed RPS alignment")
    )
    monkeypatch.setattr(model_module, "compute_scan_alignment", compute)
    source = SimpleNamespace(primary_data=data, is_spectroscopic=False)
    model = PlotDataModel(source, anatomy_obj=source)
    state = PlotState()
    first = model.get_render_payload(data, state)
    model.invalidate_alignment()
    second = model.get_render_payload(data, state)
    assert first.spatial_error is None
    assert second.spatial_error is None
    np.testing.assert_array_equal(second.spatial_array, first.spatial_array)
    np.testing.assert_allclose(
        second.spatial_extent, first.spatial_extent, rtol=0, atol=1e-3
    )
    xr.testing.assert_identical(data, before)
    compute.assert_not_called()
