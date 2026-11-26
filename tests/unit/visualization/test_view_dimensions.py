"""Check the one owner of image-plane and curve-dimension choices.

Fixtures are invented zero arrays; only dimension names and sizes matter, so
all comparisons are exact.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.visualization.render_data import PlotDataModel
from hypermri.visualization.state import PlotState
from hypermri.visualization.view_dimensions import (
    curve_dim_choices,
    default_curve_dim,
    displayed_spatial_dims,
    view_plane_choices,
)


def _zeros(**sizes: int) -> xr.DataArray:
    return xr.DataArray(np.zeros(tuple(sizes.values())), dims=tuple(sizes))


@pytest.mark.parametrize(
    ("sizes", "slicing_dim", "expected"),
    [
        (
            {"read_idx": 4, "phase_idx": 5, "slice_idx": 3},
            None,
            ["phase_idx", "read_idx"],
        ),
        (
            {"read_idx": 4, "phase_idx": 5, "slice_idx": 3},
            "read_idx",
            ["slice_idx", "phase_idx"],
        ),
        (
            {"read_idx": 4, "phase_idx": 5, "slice_idx": 3},
            "phase_idx",
            ["slice_idx", "read_idx"],
        ),
        ({"read_idx": 4, "phase_idx": 6}, None, ["phase_idx", "read_idx"]),
        ({"read_idx": 6, "phase_idx": 4}, None, ["read_idx", "phase_idx"]),
        ({"read_idx": 4, "phase_idx": 4}, None, ["read_idx", "phase_idx"]),
        ({"slice_idx": 4, "phase_idx": 4}, "read_idx", ["slice_idx", "phase_idx"]),
        ({"read_idx": 4, "acquisition_idx": 8}, None, ["read_idx"]),
        ({"acquisition_idx": 8}, "slice_idx", []),
    ],
)
def test_displayed_spatial_dims(sizes, slicing_dim, expected):
    assert displayed_spatial_dims(_zeros(**sizes), slicing_dim) == expected


def test_view_plane_choices_need_three_spatial_dims():
    assert view_plane_choices(_zeros(read_idx=2, phase_idx=2, slice_idx=2)) == [
        "slice_idx",
        "read_idx",
        "phase_idx",
    ]
    assert view_plane_choices(_zeros(read_idx=2, phase_idx=2, echo_idx=3)) == []


def test_curve_choices_skip_image_and_single_entry_dims():
    data = _zeros(read_idx=4, phase_idx=4, slice_idx=1, repetition_idx=3, echo_idx=2)
    assert curve_dim_choices(data, ["phase_idx", "read_idx"]) == [
        "echo_idx",
        "repetition_idx",
    ]


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"repetition_idx": 3, "echo_idx": 2}, "echo_idx"),
        ({"acquisition_idx": 8, "inversion_idx": 4}, "inversion_idx"),
        ({"acquisition_idx": 8, "repetition_idx": 4}, "acquisition_idx"),
        ({"zeta_idx": 3, "alpha_idx": 2}, "alpha_idx"),
        ({"repetition_idx": 1}, None),
    ],
)
def test_default_curve_dim(extra, expected):
    data = _zeros(read_idx=4, phase_idx=4, **extra)
    assert default_curve_dim(data, ["phase_idx", "read_idx"]) == expected


def test_render_data_follows_the_shared_choice():
    """The drawn curve runs along the dimension a menu offers as default."""
    data = _zeros(read_idx=4, phase_idx=4, acquisition_idx=8, inversion_idx=3)
    state = PlotState()
    payload = PlotDataModel(SimpleNamespace(primary_data=data)).get_render_payload(
        data, state
    )
    assert payload.spatial_dims == ["read_idx", "phase_idx"]
    assert len(payload.spectral_y) == 3


def test_single_entry_curve_falls_back_to_the_remaining_dim():
    data = _zeros(read_idx=4, phase_idx=4, repetition_idx=1)
    model = PlotDataModel(SimpleNamespace(primary_data=data))
    assert (
        model._determine_secondary_target_dim(
            data, PlotState(), ["read_idx", "phase_idx"]
        )
        == "repetition_idx"
    )
