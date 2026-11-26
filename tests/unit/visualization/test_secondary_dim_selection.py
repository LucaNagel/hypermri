"""Protect the secondary-dimension choice and image orientation of one slice.

The invented fixture mimics a single-slice diffusion scan: a square 4 x 4
read/phase plane, one ``slice_idx`` entry and five ``bvalue_idx`` entries.
Values are arbitrary integers in a.u.; the comparisons are exact because the
spatial view only selects and transposes samples.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from ipywidgets import Output

from hypermri.visualization import PlotOptions, plot
from hypermri.visualization.render_data import PlotDataModel
from hypermri.visualization.state import PlotState


def _single_slice_diffusion() -> xr.DataArray:
    values = np.arange(4 * 4 * 1 * 5, dtype=float).reshape(4, 4, 1, 5)
    return xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "slice_idx", "bvalue_idx"),
        coords={
            "read_idx": np.arange(4),
            "phase_idx": np.arange(4),
            "slice_idx": np.arange(1),
            "bvalue_idx": np.array([0.0, 50.0, 100.0, 500.0, 1000.0]),
        },
        attrs={"units": "a.u.", "invented_fixture": True},
    )


@pytest.mark.parametrize("secondary_dim", ["slice_idx", "bvalue_idx", None])
def test_spatial_orientation_does_not_depend_on_secondary_dim(secondary_dim):
    data = _single_slice_diffusion()
    state = PlotState()
    state.spatial.slicing_dim = "slice_idx"
    state.spatial.slice_indices = {"slice_idx": 0, "bvalue_idx": 3}
    state.spectral.secondary_dim = secondary_dim

    payload = PlotDataModel(SimpleNamespace(primary_data=data)).get_render_payload(
        data, state
    )

    assert payload.spatial_error is None
    assert payload.spatial_dims == ["phase_idx", "read_idx"]
    expected = data.isel(slice_idx=0, bvalue_idx=3).transpose("phase_idx", "read_idx")
    np.testing.assert_array_equal(payload.spatial_array, expected.values)


@pytest.mark.parametrize(
    ("slicing_dim", "expected"),
    [
        ("slice_idx", ["phase_idx", "read_idx"]),
        ("read_idx", ["slice_idx", "phase_idx"]),
        ("phase_idx", ["slice_idx", "read_idx"]),
    ],
)
def test_view_plane_order_applies_after_its_dimension_was_selected(
    slicing_dim, expected
):
    data = _single_slice_diffusion().isel({slicing_dim: 0, "bvalue_idx": 0})
    state = PlotState()
    state.spatial.slicing_dim = slicing_dim

    dims = PlotDataModel(SimpleNamespace(primary_data=data))._determine_spatial_dims(
        data, state, "ds0"
    )

    assert dims == expected
