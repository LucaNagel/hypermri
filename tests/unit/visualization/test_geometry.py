"""Pure viewer geometry on invented millimetre coordinates.

Coordinates have known Cartesian step sizes; extent/position comparisons use
the shared stage-D physical position tolerance. Acquired values are invented
and coordinate permutation/reversal must preserve their storage precision.
"""

import copy
import importlib
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.utils.coordinates.transformations import (
    _POSITION_ATOL_MM as POSITION_ATOL_MM,
)
from hypermri.visualization.geometry import read_phase_to_plot_coordinates
from hypermri.visualization.render_data import PlotDataModel
from hypermri.visualization.state import PlotState

_CASES = list(product([False, True], repeat=4))


@pytest.mark.parametrize("dims", [("read_idx", "phase_idx"), ("phase_idx", "read_idx")])
@pytest.mark.parametrize("rotation_k", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("flip_ud, flip_lr", list(product([False, True], repeat=2)))
def test_read_phase_to_plot_coordinates_matches_numpy_image_transform(
    dims, rotation_k, flip_ud, flip_lr
):
    """A labelled voxel remains at the returned plot position after display transforms."""
    shape_by_dim = {"read_idx": 3, "phase_idx": 5}
    shape = tuple(shape_by_dim[dim] for dim in dims)
    values = np.zeros(shape, dtype=np.uint8)
    read_idx, phase_idx = 2, 3
    position_by_dim = {"read_idx": read_idx, "phase_idx": phase_idx}
    values[tuple(position_by_dim[dim] for dim in dims)] = 1
    image = xr.DataArray(
        values,
        dims=dims,
        coords={"read_idx": [10, 20, 30], "phase_idx": [4, 8, 12, 16, 20]},
    )

    displayed = np.rot90(image.values, k=rotation_k)
    if flip_ud:
        displayed = np.flipud(displayed)
    if flip_lr:
        displayed = np.fliplr(displayed)

    x, y = read_phase_to_plot_coordinates(
        30, 16, image, rotation_k=rotation_k, flip_ud=flip_ud, flip_lr=flip_lr
    )

    assert displayed[y, x] == 1


def test_read_phase_to_plot_coordinates_requires_a_displayable_plane():
    """Non-singleton dimensions outside the displayed read/phase plane are explicit."""
    image = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("read_idx", "phase_idx", "echo_idx"),
    )

    with pytest.raises(ValueError, match="Select every non-singleton dimension"):
        read_phase_to_plot_coordinates(1, 2, image)


def _image(swapped, reverse_x, reverse_y, permuted, dtype=np.complex64):
    dims = ("read_idx", "phase_idx", "slice_idx")
    shape = (3, 2, 3)
    values = (np.arange(18).reshape(shape) + 1j).astype(dtype)
    data = xr.DataArray(values, dims=dims, attrs={"units": "a.u.", "invented": True})
    physical = {
        "read_idx": np.array([-2.0, 0.0, 2.0]),
        "phase_idx": np.array([4.0, 6.0]),
        "slice_idx": np.array([0.0, 3.0, 6.0]),
    }
    x_dim, y_dim = ("phase_idx", "read_idx") if swapped else ("read_idx", "phase_idx")
    if reverse_x:
        physical[x_dim] = physical[x_dim][::-1]
    if reverse_y:
        physical[y_dim] = physical[y_dim][::-1]
    grids = dict(
        zip(dims, np.meshgrid(*(physical[d] for d in dims), indexing="ij"), strict=True)
    )
    for coord, dim in [("x", x_dim), ("y", y_dim), ("z", "slice_idx")]:
        data = data.assign_coords({coord: (dims, grids[dim])})
        data.coords[coord].attrs["units"] = "mm"
    if permuted:
        data = data.transpose("phase_idx", "slice_idx", "read_idx")
    state = PlotState()
    state.spatial.coordinate_mode = "xyz"
    state.spatial.slicing_dim = "slice_idx"
    state.spatial.slice_indices = {"slice_idx": 1, "read_idx": 1, "phase_idx": 0}
    plane = data.isel(slice_idx=1)
    expected = plane.transpose(y_dim, x_dim)
    if reverse_x:
        expected = expected.isel({x_dim: slice(None, None, -1)})
    if reverse_y:
        expected = expected.isel({y_dim: slice(None, None, -1)})
    bounds = [3.0, 7.0, -3.0, 3.0] if swapped else [-3.0, 3.0, 3.0, 7.0]
    return data, plane, state, expected, bounds


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("case", _CASES)
def test_existing_geometry_orientation_extent_and_input_preservation(case, dtype):
    data, plane, state, expected, bounds = _image(*case, dtype=dtype)
    before = data.copy(deep=True)
    state_before = copy.deepcopy(state.to_dict())
    model = PlotDataModel(SimpleNamespace(primary_data=data))
    oriented, dims = model._orient_spatial_slice_for_xyz(
        plane, list(plane.dims), state, geometry_data=data
    )
    xr.testing.assert_identical(oriented, expected)
    assert dims == list(expected.dims)
    extent, origin, x_label, y_label = model._calculate_xyz_spatial_extent(
        oriented, dims, geometry_data=data
    )
    np.testing.assert_allclose(extent, bounds, rtol=0, atol=POSITION_ATOL_MM)
    assert (origin, x_label, y_label) == ("lower", "x [mm]", "y [mm]")
    assert model._xyz_plane_signature(
        oriented, dims, geometry_data=data, slicing_dim="slice_idx"
    ) == ("z", 3.0)
    xr.testing.assert_identical(data, before)
    assert state.to_dict() == state_before


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("case", _CASES)
def test_pure_geometry_matches_characterized_model_without_widgets(case, dtype):
    geometry = importlib.import_module("hypermri.visualization.geometry")
    data, plane, state, expected, bounds = _image(*case, dtype=dtype)
    before = data.copy(deep=True)
    oriented, dims = geometry.orient_xyz_slice(
        plane, list(plane.dims), state, geometry_data=data
    )
    xr.testing.assert_identical(oriented, expected)
    extent = geometry.xyz_spatial_extent(oriented, dims, geometry_data=data)
    np.testing.assert_allclose(extent[0], bounds, rtol=0, atol=POSITION_ATOL_MM)
    assert extent[1:] == ("lower", "x [mm]", "y [mm]")
    assert geometry.nearest_slice_index(data, "z", "slice_idx", 4.5) == 1
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize(
    "positions, position, expected",
    [
        ([0.0, 3.0, 6.0], 4.5, 1),
        ([6.0, 3.0, 0.0], 4.5, 0),
        ([np.nan, 3.0, 6.0], 0.0, 1),
        ([np.nan, np.nan, np.nan], 0.0, None),
    ],
)
def test_existing_nearest_slice_ties_and_nonfinite_coordinates(
    positions, position, expected
):
    data, _, _, _, _ = _image(False, False, False, False)
    data = data.assign_coords(z=("slice_idx", positions))
    assert (
        PlotDataModel()._nearest_slice_index_for_xyz_position(
            data, "z", "slice_idx", position
        )
        == expected
    )


def test_geometry_dependency_boundary_and_missing_coordinate_contract():
    geometry = importlib.import_module("hypermri.visualization.geometry")
    data = xr.DataArray(
        np.ones((2, 3), dtype=np.float32), dims=("phase_idx", "read_idx")
    )
    state = PlotState()
    state.spatial.coordinate_mode = "xyz"
    oriented, dims = geometry.orient_xyz_slice(data, list(data.dims), state)
    xr.testing.assert_identical(oriented, data)
    assert dims == list(data.dims)
    assert geometry.xyz_spatial_extent(data, dims) is None
    assert geometry.nearest_slice_index(data, "z", "slice_idx", 0.0) is None
    # The geometry owner contains ordinary data functions and never imports the
    # model/controller, interactive packages or numerical analysis.
    import ast
    from pathlib import Path

    tree = ast.parse(Path(geometry.__file__).read_text())
    names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(
        any(
            s in (name or "")
            for s in ("controller", "model", "ipywidgets", "plotly", "analysis")
        )
        for name in names
    )


def test_rendered_centres_and_clicks_retain_directed_bounds_without_widgets():
    from hypermri.visualization.geometry import (
        display_voxel_centres,
        snap_display_point,
    )

    extent, shape = [5, -1, -4.5, 7.5], (4, 3)
    x, y = display_voxel_centres(extent, shape)
    np.testing.assert_allclose(x, [4, 2, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(y, [-3, 0, 3, 6], rtol=0, atol=1e-12)
    assert snap_display_point(2.6, 0.7, extent, shape) == (2, 0)
    assert snap_display_point(3, 1.5, extent, shape) == (4, 0)
    assert snap_display_point(6, 0, extent, shape) is None
    assert snap_display_point(np.nan, 0, extent, shape) is None


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("permuted", [False, True])
def test_pure_mask_mapping_restores_labels_and_coordinates(rotation, permuted):
    from hypermri.visualization.geometry import (
        mask_coverage_in_data_order,
        mask_coverage_in_display_order,
    )

    dims = ["read_idx", "phase_idx"]
    coords = {"read_idx": np.array([4, 2, 0]), "phase_idx": np.array([-3, 0, 3, 6])}
    coverage = np.zeros((3, 4), dtype=np.float32)
    coverage[0, 1] = 0.75
    displayed = {"read_idx": coords["read_idx"][::-1], "phase_idx": coords["phase_idx"]}
    # Independent grid reference: orient the descending read line, then rotate
    # clockwise like the viewer; the rotation reverses the displayed y (read)
    # line at 90 degrees, x (phase) at 270 and both at 180.
    expected = np.rot90(coverage[::-1], k=-(rotation // 90))
    if rotation in (90, 180):
        displayed["read_idx"] = displayed["read_idx"][::-1]
    if rotation in (180, 270):
        displayed["phase_idx"] = displayed["phase_idx"][::-1]
    stored, stored_dims = (coverage.T, dims[::-1]) if permuted else (coverage, dims)
    before = stored.copy()
    actual = mask_coverage_in_display_order(
        stored,
        stored_dims,
        dims,
        rotation,
        display_coords=displayed,
        source_coords=coords,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    restored = mask_coverage_in_data_order(
        actual,
        dims,
        stored_dims,
        rotation,
        display_coords=displayed,
        source_coords=coords,
    )
    np.testing.assert_allclose(restored, stored, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(stored, before)
    assert actual.dtype == restored.dtype == np.float32


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_positional_mask_mapping_follows_viewer_rotation(rotation):
    """Regression: without coordinates, 90/270 degrees used the opposite turn.

    The reference is the plane the viewer itself renders
    (``PlotDataModel.get_spatial_slice``), so a voxel marked on screen must
    land on the source voxel shown there. Invented 3 x 4 grid, no coordinates.
    """
    from types import SimpleNamespace

    from hypermri.visualization.geometry import (
        mask_coverage_in_data_order,
        mask_coverage_in_display_order,
    )
    from hypermri.visualization.render_data import PlotDataModel
    from hypermri.visualization.state import PlotState

    dims = ["phase_idx", "read_idx"]
    source = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=dims)
    state = PlotState()
    state.spatial.rotation = rotation
    model = PlotDataModel(SimpleNamespace(primary_data=source))
    shown = model.get_spatial_slice(source, state, spatial_dims=dims).values
    for row, column in np.ndindex(shown.shape):
        drawn = np.zeros(shown.shape, dtype=np.float32)
        drawn[row, column] = 1.0
        stored = mask_coverage_in_data_order(drawn, dims, dims, rotation)
        assert source.values[stored == 1.0].tolist() == [shown[row, column]]
        np.testing.assert_array_equal(
            mask_coverage_in_display_order(stored, dims, dims, rotation), drawn
        )


def test_pure_mask_mapping_rejects_rotation_and_ambiguous_grids():
    from hypermri.visualization.geometry import (
        mask_coverage_in_data_order,
        mask_coverage_in_display_order,
    )

    coverage, dims = np.zeros((2, 3)), ["read_idx", "phase_idx"]
    with pytest.raises(ValueError, match="90-degree"):
        mask_coverage_in_display_order(coverage, dims, dims, 45)
    with pytest.raises(ValueError, match="acquired grid"):
        mask_coverage_in_data_order(
            coverage,
            dims,
            dims,
            0,
            display_coords={"read_idx": np.array([1, 1])},
            source_coords={"read_idx": np.array([1, 2])},
        )
