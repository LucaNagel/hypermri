"""Verify the supplied-domain magnitude ROI contract with invented arrays.

Exact quadratures and small integer signals give independent weighted means.
The 1e-12 a.u. tolerance covers float64 reduction; float32 editor coverage
is retained, while magnitude and averaging use double precision.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.visualization.contracts import RenderPayload
from hypermri.visualization.state import PlotState


@pytest.mark.parametrize("with_coordinates", [False, True])
def test_volume_estimate_without_widgets_preserves_spacing_and_inputs(with_coordinates):
    from hypermri.analysis.anatomical.roi import mask_volume_ml

    data = xr.DataArray(
        np.zeros((3, 4, 2)), dims=("read_idx", "phase_idx", "slice_idx")
    )
    if with_coordinates:
        data = data.assign_coords(
            {
                "read_idx": ("read_idx", [4, 2, 0], {"units": "mm"}),
                "phase_idx": ("phase_idx", [-3, 0, 3, 6], {"units": "mm"}),
                "slice_idx": ("slice_idx", [0, 5], {"units": "mm"}),
            }
        )
    source = SimpleNamespace(fov_mm=[6, 12, 10], matrix=[3, 4, 2], slice_thickness_mm=5)
    mask = xr.DataArray(
        np.zeros((4, 2, 3), dtype=np.float32),
        dims=("phase_idx", "slice_idx", "read_idx"),
    )
    mask.values[1, 1, 0] = 0.75
    original_data, original_mask = data.copy(deep=True), mask.copy(deep=True)
    # Three quarters of a 2x3x5 mm cell, independently converted to mL.
    assert mask_volume_ml(mask, data, source) == pytest.approx(0.0225, abs=1e-12)
    xr.testing.assert_identical(data, original_data)
    xr.testing.assert_identical(mask, original_mask)


def test_unresolved_volume_is_not_replaced_by_default_geometry():
    from hypermri.analysis.anatomical.roi import mask_volume_ml

    data = xr.DataArray(np.zeros((2, 3)), dims=("read_idx", "phase_idx"))
    assert mask_volume_ml(xr.ones_like(data), data) is None
