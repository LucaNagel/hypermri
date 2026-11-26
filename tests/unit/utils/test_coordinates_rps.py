"""Slice positions of the RPS axes and extent for 2D and 3D scans.

The scan parameters are synthetic.  The 2D case follows the ParaVision reference water-phantom
series, five 1 mm slices with 4 mm gaps, whose DICOM slice centres are 5 mm
apart; ``tests/unit/utils/test_coordinates_paravision_reference.py`` checks the
same rule against the DICOM exports.  ``fov_mm`` holds the slice field of view
as ``BrukerHandler`` builds it, from the first to the last slice edge.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.utils.coordinates.grids import (
    define_grid_lines,
    define_xyz_grid,
    get_rps_axes,
    get_rps_extent,
    get_rps_grid,
    select_rps_extent_by_axis,
)

# The position tolerance of the ParaVision reference tests: DICOM positions
# carry six decimals in mm, and the slice step error of 0.8 mm exceeds it by
# almost three orders of magnitude.
POSITION_ATOL_MM = 1e-3

SLICE_OFFSET_MM = 3.0


class _ScanParameters(SimpleNamespace):
    """Synthetic scan parameters with the ``get`` access of the vendor handlers."""

    def get(self, name, default=None):
        return getattr(self, name, default)


def _scan(spatial_dimension, slice_fov_mm, n_slices, thickness_mm, gap_mm):
    fov_mm = np.array([32.0, 48.0, slice_fov_mm])
    matrix = np.array([32, 48, n_slices])
    return _ScanParameters(
        spatial_dimension=spatial_dimension,
        fov_mm=fov_mm,
        matrix=matrix,
        resolution_mm=fov_mm / matrix,
        n_slices=n_slices,
        slice_thickness_mm=thickness_mm,
        slice_gap_mm=gap_mm,
        read_offset_mm=4.0,
        phase_offset_mm=-2.0,
        slice_offset_mm=SLICE_OFFSET_MM,
    )


def test_2d_slice_axis_steps_by_slice_thickness_plus_gap():
    """Five 1 mm slices with 4 mm gaps lie 5 mm apart around the slice offset."""
    scan = _scan("2D", slice_fov_mm=21.0, n_slices=5, thickness_mm=1.0, gap_mm=4.0)

    slice_axis = get_rps_axes(scan)[2]
    extent = get_rps_extent(scan)

    np.testing.assert_allclose(
        slice_axis,
        SLICE_OFFSET_MM + np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_allclose(
        extent[4:],
        [SLICE_OFFSET_MM - 12.5, SLICE_OFFSET_MM + 12.5],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize(
    ("spatial_dimension", "slice_fov_mm", "n_slices", "thickness_mm", "gap_mm"),
    [
        pytest.param("2D", 5.0, 5, 1.0, 0.0, id="2d-without-gap"),
        pytest.param("2D", 1.0, 1, 1.0, 4.0, id="2d-single-slice"),
        pytest.param("3D", 20.0, 40, 20.0, 4.0, id="3d-slab"),
    ],
)
def test_slice_axis_divides_the_slice_field_of_view(
    spatial_dimension, slice_fov_mm, n_slices, thickness_mm, gap_mm
):
    """Without gaps between several 2D slices the step is the field of view per slice."""
    scan = _scan(spatial_dimension, slice_fov_mm, n_slices, thickness_mm, gap_mm)
    step = slice_fov_mm / n_slices

    slice_axis = get_rps_axes(scan)[2]
    extent = get_rps_extent(scan)

    np.testing.assert_allclose(
        slice_axis,
        SLICE_OFFSET_MM + (np.arange(n_slices) - (n_slices - 1) / 2) * step,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_allclose(
        extent[4:],
        [SLICE_OFFSET_MM - slice_fov_mm / 2, SLICE_OFFSET_MM + slice_fov_mm / 2],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_rps_extent_without_resolution_names_the_half_voxel_correction():
    """Regression: a missing voxel size is reported instead of printed and skipped."""
    scan = _scan("2D", slice_fov_mm=5.0, n_slices=5, thickness_mm=1.0, gap_mm=0.0)
    scan.resolution_mm = None

    with pytest.raises(ValueError, match="with_correction=False"):
        get_rps_extent(scan)


def test_rps_extent_without_correction_needs_no_resolution():
    """The field-of-view edges around the offsets need no voxel size."""
    scan = _scan("2D", slice_fov_mm=5.0, n_slices=5, thickness_mm=1.0, gap_mm=0.0)
    scan.resolution_mm = None

    extent = get_rps_extent(scan, with_correction=False)

    np.testing.assert_allclose(
        extent,
        [-20.0, 12.0, -22.0, 26.0, SLICE_OFFSET_MM - 2.5, SLICE_OFFSET_MM + 2.5],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize("function", [get_rps_extent, select_rps_extent_by_axis])
def test_rps_extent_rejects_the_removed_grid_argument(function):
    """The limits come from the scan parameters; a grid is no longer accepted."""
    scan = _scan("2D", slice_fov_mm=21.0, n_slices=5, thickness_mm=1.0, gap_mm=4.0)

    with pytest.raises(TypeError, match="rps_grid_mm"):
        function(scan, rps_grid_mm=get_rps_grid(scan))


@pytest.mark.parametrize(
    "name",
    [
        "get_rps_extent_from_grid",
        "convert_rps_to_xyz_grid",
        "convert_rps_to_xyz_matrix_based",
    ],
)
def test_removed_rps_helpers_are_absent(name):
    """The removed grid helpers are gone from the coordinates package."""
    from hypermri.utils import coordinates
    from hypermri.utils.coordinates import grids, transformations

    assert name not in coordinates.__all__
    for module in (coordinates, grids, transformations):
        assert not hasattr(module, name)


def _three_package_scan():
    """Three packages of seven 4 mm slices, like the ParaVision reference localizers."""
    scan = _scan("2D", slice_fov_mm=84.0, n_slices=21, thickness_mm=4.0, gap_mm=0.0)
    scan.slices_per_pack = (7, 7, 7)
    return scan


@pytest.mark.parametrize(
    "rps_function",
    [
        pytest.param(get_rps_axes, id="axes"),
        pytest.param(get_rps_grid, id="grid"),
        pytest.param(get_rps_extent, id="extent"),
        pytest.param(lambda scan: get_rps_extent(scan, old=True), id="extent-old"),
        pytest.param(select_rps_extent_by_axis, id="extent-by-axis"),
    ],
)
def test_several_slice_packages_have_no_rps_coordinates(rps_function):
    """Regression: the first package's offsets were used for all packages."""
    with pytest.raises(ValueError, match=r"3 slice packages \(7, 7, 7\).*get_xyz_grid"):
        rps_function(_three_package_scan())


def test_one_slice_package_keeps_its_rps_coordinates():
    """A single package gives the axes of a scan without package counts."""
    scan = _scan("2D", slice_fov_mm=21.0, n_slices=5, thickness_mm=1.0, gap_mm=4.0)
    expected_axes, expected_extent = get_rps_axes(scan), get_rps_extent(scan)
    scan.slices_per_pack = (5,)

    for axis, expected in zip(get_rps_axes(scan), expected_axes, strict=True):
        np.testing.assert_array_equal(axis, expected)
    assert get_rps_extent(scan) == expected_extent


@pytest.mark.parametrize("operation", [get_rps_axes, get_rps_extent])
@pytest.mark.parametrize(
    "field,value",
    [
        ("fov_mm", None),
        ("fov_mm", [32]),
        ("fov_mm", [32, np.nan, 5]),
        ("matrix", None),
        ("matrix", [32]),
        ("matrix", [32, 48, 1.5]),
    ],
)
def test_missing_or_incomplete_geometry_names_the_operation_and_parameter(
    operation, field, value
):
    scan = _scan("2D", 5, 5, 1, 0)
    setattr(scan, field, value)
    with pytest.raises(ValueError, match=operation.__name__ + ".*" + field):
        operation(scan)


def test_two_axis_geometry_requires_declared_slice_count_and_thickness():
    scan = _scan("2D", 5, 5, 1, 0)
    scan.fov_mm = scan.fov_mm[:2]
    scan.matrix = scan.matrix[:2]
    before = (scan.fov_mm.copy(), scan.matrix.copy())
    axes = get_rps_axes(scan)
    np.testing.assert_allclose(
        axes[2], np.arange(-2, 3) + SLICE_OFFSET_MM, rtol=0, atol=POSITION_ATOL_MM
    )
    np.testing.assert_array_equal(scan.fov_mm, before[0])
    np.testing.assert_array_equal(scan.matrix, before[1])
    scan.n_slices = None
    with pytest.raises(ValueError, match="n_slices"):
        get_rps_axes(scan)
    scan.n_slices = 5
    scan.slice_thickness_mm = None
    with pytest.raises(ValueError, match="slice_thickness_mm"):
        get_rps_axes(scan)


def test_explicit_rps_grid_bypasses_missing_scan_metadata():
    expected = tuple(np.arange(2, dtype=float) for _ in range(3))
    grids = tuple(np.meshgrid(*expected, indexing="ij"))
    for result, reference in zip(
        get_rps_axes(rps_grid_mm=grids), expected, strict=True
    ):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize(
    "axis,indices",
    [
        ("read_idx", [2, 3, 4, 5]),
        ("phase_idx", [0, 1, 4, 5]),
        ("slice_idx", [0, 1, 2, 3]),
    ],
)
def test_extent_selection_always_returns_four_flat_limits(axis, indices):
    scan = _scan("2D", 5, 5, 1, 0)
    expected = np.asarray(get_rps_extent(scan))[indices]
    result = select_rps_extent_by_axis(scan, axis=axis)
    assert np.shape(result) == (4,)
    np.testing.assert_allclose(result, expected, rtol=0, atol=POSITION_ATOL_MM)


def test_unknown_extent_axis_is_rejected_before_reading_geometry():
    with pytest.raises(
        ValueError, match="unknown axis.*read_idx.*phase_idx.*slice_idx"
    ):
        select_rps_extent_by_axis(axis="invalid")


@pytest.mark.parametrize(
    "field,kwargs", [("fov_mm", {"mat": (2, 2, 1)}), ("matrix", {"fov": (2, 2, 1)})]
)
def test_legacy_xyz_grid_requires_declared_missing_geometry(field, kwargs):
    with pytest.raises(ValueError, match="define_xyz_grid.*" + field):
        define_xyz_grid(_ScanParameters(), **kwargs)


def test_single_axis_explicit_pairs_need_no_scan_geometry():
    scan = _ScanParameters()
    np.testing.assert_allclose(define_grid_lines(scan, fov=4, res=1), [-2, -1, 0, 1])
    np.testing.assert_allclose(define_grid_lines(scan, mat=4, res=1), [-2, -1, 0, 1])
