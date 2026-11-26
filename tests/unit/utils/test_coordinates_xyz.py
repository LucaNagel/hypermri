import logging
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.mixins import UniformDataPropertiesMixin
from hypermri.utils.coordinates import grids
from hypermri.utils.coordinates.grids import (
    assign_xyz_coordinates,
    define_grid_lines,
    define_xyz_grid,
    get_xyz_axes,
    get_xyz_extent,
    get_xyz_grid_from_visu,
)

# The position tolerance of the ParaVision reference tests: DICOM positions
# carry six decimals in mm, and a wrong slice step of 1 mm exceeds it by three
# orders of magnitude.
POSITION_ATOL_MM = 1e-3


def test_define_grid_lines_keeps_the_scalar_fov_and_matrix_rule():
    grid_lines = define_grid_lines(fov=100.0, mat=4)

    np.testing.assert_allclose(grid_lines, [-50.0, -25.0, 0.0, 25.0])


def test_define_grid_lines_rejects_missing_geometry_instead_of_inventing_it():
    class Handler:
        method = {}

        def get(self, _name, default=None):
            return default

    with pytest.raises(ValueError, match="define_grid_lines.*fov_mm.*matrix"):
        define_grid_lines(scan=Handler())


def test_define_grid_lines_rejects_multi_axis_fov_values():
    with pytest.raises(ValueError, match="one axis.*scalar fov"):
        define_grid_lines(fov=np.array([100.0, 80.0]), mat=np.array([64, 32]))


def test_get_xyz_grid_from_visu_axis_aligned():
    handler = SimpleNamespace(
        matrix=np.array([4, 3, 2]),
        visucore_position=np.array([[-2.0, -1.0, -5.0], [-2.0, -1.0, 0.0]]),
        visucore_orientation=np.array(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * 2
        ),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([4.0, 3.0]),
    )

    x, y, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    assert x.shape == (4, 3, 2)
    np.testing.assert_allclose(x[:, 0, 0], [-2.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(y[0, :, 0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(z[0, 0, :], [-5.0, 0.0])


def test_get_xyz_grid_from_visu_maps_swapped_inplane_axes():
    handler = SimpleNamespace(
        matrix=np.array([3, 4, 1]),
        visucore_position=np.array([[10.0, 20.0, 30.0]]),
        visucore_orientation=np.array(
            [[0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]]
        ),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([8.0, 6.0]),
    )

    x, y, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    assert x.shape == (3, 4, 1)
    np.testing.assert_allclose(x[:, :, 0], 10.0)
    np.testing.assert_allclose(y[0, :, 0], [20.0, 22.0, 24.0, 26.0])
    np.testing.assert_allclose(z[:, 0, 0], [30.0, 28.0, 26.0])


def test_get_xyz_grid_from_visu_ignores_leading_spectral_size_axis():
    handler = SimpleNamespace(
        matrix=np.array([2, 3, 1]),
        n_acquisition=4,
        visucore_position=np.array([[1.0, 2.0, 3.0]]),
        visucore_orientation=np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        visucore_size=np.array([4, 2, 3]),
        visucore_extent=np.array([7.0, 2.0, 6.0]),
    )

    x, y, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    assert x.shape == (2, 3, 1)
    np.testing.assert_allclose(x[:, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(y[0, :, 0], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(z[..., 0], 3.0)


def test_get_xyz_grid_from_visu_ignores_middle_spectral_size_axis():
    handler = SimpleNamespace(
        matrix=np.array([2, 3, 1]),
        n_acquisition=4,
        visucore_position=np.array([[1.0, 2.0, 3.0]]),
        visucore_orientation=np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        visucore_size=np.array([2, 4, 3]),
        visucore_extent=np.array([2.0, 7.0, 6.0]),
    )

    x, y, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    assert x.shape == (2, 3, 1)
    np.testing.assert_allclose(x[:, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(y[0, :, 0], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(z[..., 0], 3.0)


def test_get_xyz_grid_from_visu_3d_single_frame_uses_volume_slice_spacing():
    handler = SimpleNamespace(
        matrix=np.array([128, 128, 128]),
        spatial_dimension="3D",
        fov_mm=np.array([40.0, 40.0, 40.0]),
        slice_thickness_mm=40.0,
        slice_gap_mm=0.0,
        visucore_position=np.array([[-20.0, -20.0, -19.84375]]),
        visucore_orientation=np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        visucore_size=np.array([128, 128, 128]),
        visucore_extent=np.array([40.0, 40.0, 40.0]),
    )

    _, _, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    assert z.shape == (128, 128, 128)
    np.testing.assert_allclose(z[0, 0, 0], -19.84375)
    np.testing.assert_allclose(z[0, 0, 1] - z[0, 0, 0], 0.3125)
    np.testing.assert_allclose(z[0, 0, -1], 19.84375)


def test_get_xyz_grid_from_visu_uses_acquisition_directions_for_square_matrix():
    handler = SimpleNamespace(
        matrix=np.array([2, 2, 1]),
        acquisition_matrix=np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        visucore_position=np.array([[-1.0, 2.0, 3.0]]),
        visucore_orientation=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]]),
        visucore_size=np.array([2, 2]),
        visucore_extent=np.array([2.0, 2.0]),
    )

    x, y, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    np.testing.assert_allclose(x[0, :, 0], [-1.0, 0.0])
    np.testing.assert_allclose(z[:, 0, 0], [3.0, 2.0])
    np.testing.assert_allclose(y[..., 0], 2.0)


@pytest.mark.parametrize(
    "disk_slice_order",
    [
        {},
        {"visucore_disk_slice_order": "disk_normal_slice_order"},
        {"visucore_disk_slice_order": "disk_reverse_slice_order"},
        {"visu": {"VisuCoreDiskSliceOrder": "disk_reverse_slice_order"}},
    ],
    ids=["absent", "normal", "reverse", "reverse_in_visu_pars"],
)
def test_get_xyz_grid_from_visu_ignores_disk_slice_order(disk_slice_order):
    """Regression: the slices follow VisuCorePosition for every disk order.

    Previously ``disk_reverse_slice_order`` reversed the slice positions to
    ``[12, 11, 10]``.
    """
    handler = SimpleNamespace(
        matrix=np.array([2, 2, 3]),
        spatial_dimension="3D",
        fov_mm=np.array([2.0, 2.0, 3.0]),
        visucore_position=np.array([[0.0, 0.0, 10.0]]),
        visucore_orientation=np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        visucore_size=np.array([2, 2, 3]),
        visucore_extent=np.array([2.0, 2.0, 3.0]),
        **disk_slice_order,
    )

    _, _, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    np.testing.assert_allclose(z[0, 0, :], [10.0, 11.0, 12.0])


def test_get_xyz_grid_from_visu_matches_dicom_of_reverse_disk_order_scan():
    """Regression: a 3D scan stored in reverse disk order matches its DICOM.

    The geometry values come from a local 3D FLASH scan (Head_Supine, coronal,
    read H_F) with ``VisuCoreDiskSliceOrder`` ``disk_reverse_slice_order`` and
    a ParaVision DICOM export of 60 images per cycle.  Only these values and the
    ``Image Position (Patient)`` of the first and the last image are copied.
    The images are 1 mm apart along the slice normal ``(0, 1, 0)``.  Slice
    ``k`` of the ``2dseq`` and image ``k`` agree voxel by voxel (correlation
    1.000000, against image ``59 - k`` 0.525).  Previously the grid placed
    slice 0 at the position of image 59, 59 mm away.
    """
    handler = SimpleNamespace(
        spatial_dimension="3D",
        visucore_disk_slice_order="disk_reverse_slice_order",
        visucore_position=np.array([[-26.0, -23.5, 25.0]]),
        visucore_orientation=np.array([[1, 0, 0, 0, 0, -1, 0, 1, 0]]),
        visucore_size=np.array([48, 48, 60]),
        visucore_extent=np.array([48.0, 48.0, 60.0]),
    )

    grid = get_xyz_grid_from_visu(handler, spatial_shape=(48, 48, 60), dtype=np.float64)

    slice_origins = np.stack(grid, axis=-1)[0, 0]
    np.testing.assert_allclose(
        slice_origins[[0, -1]],
        [[-26.0, -23.5, 25.0], [-26.0, 35.5, 25.0]],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_allclose(
        np.diff(slice_origins, axis=0),
        np.tile([0.0, 1.0, 0.0], (59, 1)),
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


class _VisuScan(SimpleNamespace):
    """Synthetic ``VisuCore`` metadata with the ``get`` access of the handlers."""

    def get(self, name, default=None):
        return getattr(self, name, default)


AXIAL_ROWS = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# Read along x, phase along -z and the slice normal along y.
CORONAL_ROWS = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
# Read and phase rotated by 30 degrees about z.
OBLIQUE_ROWS = [np.sqrt(3) / 2, 0.5, 0.0, -0.5, np.sqrt(3) / 2, 0.0, 0.0, 0.0, 1.0]


# Read along y, phase along -z and the slice normal along -x.
SAGITTAL_ROWS = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]


def _two_package_scan(reco_transposition, **extra):
    """Two square frames of 3 x 3 voxels of 2 mm: axial at 0 and sagittal at x 10.

    The layout follows the multi-package localizers of the ParaVision reference fixture, whose
    axial and sagittal packages store read along the columns and the rows.
    """
    return SimpleNamespace(
        matrix=np.array([3, 3, 2]),
        reco_transposition=np.array(reco_transposition),
        visucore_position=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        visucore_orientation=np.array([AXIAL_ROWS, SAGITTAL_ROWS]),
        visucore_size=np.array([3, 3]),
        visucore_extent=np.array([6.0, 6.0]),
        **extra,
    )


def _two_package_reference(reco_transposition):
    """Voxel centres in mm, built from the DICOM rule of each frame."""
    read, phase = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    frames = []
    for origin, orientation, transposed in zip(
        ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
        (AXIAL_ROWS, SAGITTAL_ROWS),
        reco_transposition,
        strict=True,
    ):
        column, row = (phase, read) if transposed else (read, phase)
        frames.append(
            np.asarray(origin)
            + 2.0 * column[..., None] * np.asarray(orientation[:3])
            + 2.0 * row[..., None] * np.asarray(orientation[3:6])
        )
    return np.stack(frames, axis=2)


@pytest.mark.parametrize("reco_transposition", [[0, 0], [0, 1], [1, 0], [1, 1]])
def test_get_xyz_grid_from_visu_places_read_per_frame(reco_transposition):
    """Regression: ``RECO_transposition`` places read in each frame.

    Previously one mapping, the columns for these square frames, was applied
    to all frames.  The frames with ``RECO_transposition`` 1 of the two ParaVision reference
    localizers then lay up to 89.803 mm from their DICOM voxel centres.
    """
    grid = get_xyz_grid_from_visu(
        _two_package_scan(reco_transposition), dtype=np.float64
    )

    np.testing.assert_allclose(
        np.stack(grid, axis=-1),
        _two_package_reference(reco_transposition),
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_get_xyz_grid_from_visu_prefers_reco_transposition_to_acquisition_matrix():
    """``RECO_transposition`` decides before the read direction of the scan."""
    read_along_x = np.eye(3)
    scan = _two_package_scan([0, 1], acquisition_matrix=read_along_x)

    grid = get_xyz_grid_from_visu(scan, dtype=np.float64)

    np.testing.assert_allclose(
        np.stack(grid, axis=-1),
        _two_package_reference([0, 1]),
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_get_xyz_grid_from_visu_applies_one_transposition_to_all_frames():
    """One value of a single 3D frame applies to every slice of the volume."""
    handler = SimpleNamespace(
        matrix=np.array([2, 2, 3]),
        spatial_dimension="3D",
        reco_transposition=np.array([1]),
        visucore_position=np.array([[0.0, 0.0, 0.0]]),
        visucore_orientation=np.array([AXIAL_ROWS]),
        visucore_size=np.array([2, 2, 3]),
        visucore_extent=np.array([2.0, 2.0, 3.0]),
    )

    x, y, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    for k in range(3):
        np.testing.assert_allclose(x[:, :, k], [[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(y[:, :, k], [[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(z[:, :, k], float(k))


@pytest.mark.parametrize(
    "reco_transposition",
    [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2], None],
    ids=["echo_frames_0", "echo_frames_1", "other_value", "absent"],
)
def test_get_xyz_grid_from_visu_uses_one_mapping_for_uniform_transposition(
    reco_transposition,
):
    """Characterization: uniform values place read like the in-plane sizes.

    Two slices with two echoes each have four ``2dseq`` frames but two
    positions.  Value 2 swaps other frame dimensions and leaves the size
    rule in charge, as for a local 3D scan.
    """
    transposed = reco_transposition is not None and reco_transposition[0] == 1
    handler = SimpleNamespace(
        matrix=np.array([3, 2, 2]),
        reco_transposition=reco_transposition,
        visucore_position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        visucore_orientation=np.array([AXIAL_ROWS, AXIAL_ROWS]),
        visucore_size=np.array([2, 3] if transposed else [3, 2]),
        visucore_extent=np.array([2.0, 3.0] if transposed else [3.0, 2.0]),
    )

    x, y, z = get_xyz_grid_from_visu(handler, dtype=np.float64)

    read_axis, phase_axis = (y, x) if transposed else (x, y)
    np.testing.assert_allclose(read_axis[:, 0, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(phase_axis[0, :, 0], [0.0, 1.0])
    np.testing.assert_allclose(z[0, 0, :], [0.0, 5.0])


@pytest.mark.parametrize(
    "reco_transposition, message",
    [
        ([0, 1, 0, 1], "has 4 values for 2 VisuCorePosition frames"),
        ([0, 2], r"differs between frames \(0, 2\); only 0 and 1"),
        (["a", "b"], "is not numeric"),
    ],
    ids=["count", "other_value", "not_numeric"],
)
def test_get_xyz_grid_from_visu_rejects_unassignable_transposition(
    reco_transposition, message
):
    """Mixed values that do not fit the frames raise instead of guessing."""
    with pytest.raises(ValueError, match=message):
        get_xyz_grid_from_visu(_two_package_scan(reco_transposition))


def test_get_xyz_grid_from_visu_rejects_transposition_against_inplane_sizes():
    """Read along the rows needs the row size of VisuCoreSize to be read."""
    handler = SimpleNamespace(
        matrix=np.array([3, 2, 1]),
        reco_transposition=np.array([1]),
        visucore_position=np.array([[0.0, 0.0, 0.0]]),
        visucore_orientation=np.array([AXIAL_ROWS]),
        visucore_size=np.array([3, 2]),
        visucore_extent=np.array([3.0, 2.0]),
    )

    with pytest.raises(
        ValueError,
        match=r"RECO_transposition 1 of frame 0 places read along the DICOM rows, "
        r"but VisuCoreSize \(3, 2\) does not fit the spatial shape \(3, 2\)",
    ):
        get_xyz_grid_from_visu(handler)


def _xyz_axes_warnings(caplog, scan):
    with caplog.at_level(logging.WARNING, logger=grids.__name__):
        axes = get_xyz_axes(scan)
    messages = [r.message for r in caplog.records if r.name == grids.__name__]
    return axes, messages


def test_xyz_axes_of_a_coronal_scan_log_no_warning(caplog):
    """Regression: image axes parallel to x, y and z are not oblique.

    Previously every ``acquisition_matrix`` other than the identity logged the
    oblique warning, which in the ParaVision reference study is every orientation except axial
    with read L_R.
    """
    scan = _VisuScan(
        matrix=np.array([4, 3, 2]),
        acquisition_matrix=np.reshape(CORONAL_ROWS, (3, 3)),
        visucore_position=np.array([[-2.0, -1.0, 5.0], [-2.0, 1.0, 5.0]]),
        visucore_orientation=np.array([CORONAL_ROWS] * 2),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([4.0, 3.0]),
    )

    (x_axis, y_axis, z_axis), messages = _xyz_axes_warnings(caplog, scan)

    assert messages == []
    np.testing.assert_allclose(
        x_axis, [-2.0, -1.0, 0.0, 1.0], rtol=0, atol=POSITION_ATOL_MM
    )
    np.testing.assert_allclose(y_axis, [-1.0, 1.0], rtol=0, atol=POSITION_ATOL_MM)
    np.testing.assert_allclose(z_axis, [3.0, 4.0, 5.0], rtol=0, atol=POSITION_ATOL_MM)


def test_xyz_axes_of_an_oblique_scan_log_a_warning(caplog):
    """In a rotated slice x and y depend on read and phase, so every value is kept."""
    scan = _VisuScan(
        matrix=np.array([4, 3, 1]),
        acquisition_matrix=np.reshape(OBLIQUE_ROWS, (3, 3)),
        visucore_position=np.array([[0.0, 0.0, 0.0]]),
        visucore_orientation=np.array([OBLIQUE_ROWS]),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([4.0, 3.0]),
    )

    (x_axis, y_axis, z_axis), messages = _xyz_axes_warnings(caplog, scan)

    assert len(messages) == 1
    assert "get_xyz_axes" in messages[0]
    assert (len(x_axis), len(y_axis), len(z_axis)) == (12, 12, 1)


# Orientation of a coronal scan with read along z, phase along -x and the
# slice normal along y, with the entries that ParaVision writes for a local
# 320 x 320 x 9 scan: the zero entries are cos(90 degrees), 6.1e-17, and
# its square.  Multiplied by the index and the voxel size they move positions
# by one unit in the last place, about 3.6e-15 mm at 24 mm.
_COS_90_DEG = 6.12323399573677e-17
CORONAL_COS90_ROWS = [
    -1.0,
    3.74939945665464e-33,
    -_COS_90_DEG,
    -_COS_90_DEG,
    -_COS_90_DEG,
    1.0,
    0.0,
    1.0,
    _COS_90_DEG,
]


def _cos90_coronal_scan(n_slices=3, origin_mm=(24.0, -2.0, -24.0)):
    """Synthetic coronal stack of 32 x 32 voxels of 1.5 mm with 1 mm slice steps.

    Read lies along the ``2dseq`` rows (``RECO_transposition`` 1), as in the
    local scan that showed the rounding problem.  The origin is synthetic.
    """
    rows = np.asarray(CORONAL_COS90_ROWS)
    origin = np.asarray(origin_mm) + np.arange(n_slices)[:, None] * rows[6:9]
    return _VisuScan(
        matrix=np.array([32, 32, n_slices]),
        visucore_position=origin,
        visucore_orientation=np.array([rows] * n_slices),
        visucore_size=np.array([32, 32]),
        visucore_extent=np.array([48.0, 48.0]),
        reco_transposition=np.array([1] * n_slices),
    )


def test_cos90_coronal_scan_is_axis_aligned_within_rounding(caplog):
    """The fixture of the rounding regression differs from its nominal grid by rounding.

    x follows ``phase_idx``, y ``slice_idx`` and z ``read_idx``; along the
    other index axes the positions vary by less than 1e-14 mm, far below the
    geometry position tolerance, so ``get_xyz_axes`` logs no oblique warning.
    """
    scan = _cos90_coronal_scan()
    read, phase, slice_ = np.meshgrid(
        np.arange(32), np.arange(32), np.arange(3), indexing="ij"
    )

    x_grid, y_grid, z_grid = grids.get_xyz_grid(scan)
    _, messages = _xyz_axes_warnings(caplog, scan)

    np.testing.assert_allclose(x_grid, 24.0 - 1.5 * phase, rtol=0, atol=1e-14)
    np.testing.assert_allclose(y_grid, -2.0 + 1.0 * slice_, rtol=0, atol=1e-14)
    np.testing.assert_allclose(z_grid, -24.0 + 1.5 * read, rtol=0, atol=1e-14)
    assert messages == []


@pytest.mark.parametrize(
    ("n_slices", "origin_mm"),
    [
        (3, (24.0, -2.0, -24.0)),
        (1, (24.0, -2.0, -24.0)),
        # Positions half-way between two multiples of 1e-6 mm, where rounding
        # to that grid separates values that differ by one unit in the last
        # place.
        (3, (24.0000005, -2.0000005, -24.0000005)),
        (3, (23.9999995, -2.0, -23.9999995)),
    ],
    ids=["stack", "single_slice", "rounding_boundary_up", "rounding_boundary_down"],
)
def test_xyz_axes_merge_rounding_of_an_axis_aligned_scan(n_slices, origin_mm):
    """Regression: one value per image coordinate of an axis-aligned scan.

    Found on a local 320 x 320 x 9 coronal scan gave 2064, 25 and 635
    values for x, y and z instead of 320, 9 and 320.  Previously this
    32 x 32 x 3 fixture gave 150, 133 and 64 values.  Rounding to multiples
    of 1e-6 mm instead of merging within the tolerance would still give 4
    and 64 values at the two rounding boundaries.
    """
    origin_x, origin_y, origin_z = origin_mm

    x_axis, y_axis, z_axis = get_xyz_axes(_cos90_coronal_scan(n_slices, origin_mm))

    assert (len(x_axis), len(y_axis), len(z_axis)) == (32, n_slices, 32)

    np.testing.assert_allclose(
        x_axis, origin_x - 1.5 * np.arange(32)[::-1], rtol=0, atol=POSITION_ATOL_MM
    )
    np.testing.assert_allclose(
        y_axis, origin_y + np.arange(n_slices), rtol=0, atol=POSITION_ATOL_MM
    )
    np.testing.assert_allclose(
        z_axis, origin_z + 1.5 * np.arange(32), rtol=0, atol=POSITION_ATOL_MM
    )


def test_xyz_axes_merge_rounding_independent_of_slice_order_and_dtype():
    """The merged axes do not depend on the frame order and are cast once.

    The reversed fixture lists the same three slices in the opposite frame
    order.  float32 axes are the float64 result cast once, so they keep the
    number of positions.
    """
    scan = _cos90_coronal_scan()
    reversed_scan = _cos90_coronal_scan()
    reversed_scan.visucore_position = scan.visucore_position[::-1].copy()

    axes = get_xyz_axes(scan)
    float32_axes = get_xyz_axes(scan, dtype=np.float32)

    for axis, again, reverse, float32_axis in zip(
        axes,
        get_xyz_axes(scan),
        get_xyz_axes(reversed_scan),
        float32_axes,
        strict=True,
    ):
        assert axis.dtype == np.float64
        np.testing.assert_array_equal(again, axis)
        np.testing.assert_array_equal(reverse, axis)
        assert float32_axis.dtype == np.float32
        np.testing.assert_array_equal(float32_axis, axis.astype(np.float32))


def test_xyz_axes_keep_exact_positions_bit_identical():
    """Positions whose voxels agree exactly keep their value bit for bit.

    The coronal fixture of ``test_xyz_axes_of_a_coronal_scan_log_no_warning``
    has exact orientation entries, so merging returns the exactly unique
    grid values.
    """
    scan = _VisuScan(
        matrix=np.array([4, 3, 2]),
        visucore_position=np.array([[-2.0, -1.0, 5.0], [-2.0, 1.0, 5.0]]),
        visucore_orientation=np.array([CORONAL_ROWS] * 2),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([4.0, 3.0]),
    )

    for axis, grid in zip(get_xyz_axes(scan), grids.get_xyz_grid(scan), strict=True):
        np.testing.assert_array_equal(axis, np.unique(grid))


def test_xyz_axes_reject_voxel_steps_within_the_position_tolerance():
    """Voxels of 5e-4 mm chain into one 7.5e-3 mm wide group and are rejected.

    Synthetic axial slice of 16 x 2 voxels; x changes along ``read_idx`` in
    steps of half the position tolerance.  The handler property
    raises as well: it returns None only for absent ``VisuCore`` geometry.
    """
    geometry = {
        "matrix": np.array([16, 2, 1]),
        "visucore_position": np.array([[0.0, 0.0, 0.0]]),
        "visucore_orientation": np.array([AXIAL_ROWS]),
        "visucore_size": np.array([16, 2]),
        "visucore_extent": np.array([16 * 5e-4, 2.0]),
    }

    with pytest.raises(ValueError, match=r"get_xyz_axes: .* chain over 0\.0075 mm"):
        get_xyz_axes(_VisuScan(**geometry))
    with pytest.raises(ValueError, match=r"get_xyz_axes: .* chain over 0\.0075 mm"):
        _ScanWithProperties(**geometry).xyz_axes_mm


def test_xyz_axes_keep_voxels_two_micrometres_apart():
    """Voxels of 2e-3 mm, twice the position tolerance, stay distinct."""
    scan = _VisuScan(
        matrix=np.array([8, 4, 2]),
        visucore_position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2e-3]]),
        visucore_orientation=np.array([AXIAL_ROWS] * 2),
        visucore_size=np.array([8, 4]),
        visucore_extent=np.array([16e-3, 8e-3]),
    )

    x_axis, y_axis, z_axis = get_xyz_axes(scan)

    np.testing.assert_allclose(x_axis, 2e-3 * np.arange(8), rtol=0, atol=1e-9)
    np.testing.assert_allclose(y_axis, 2e-3 * np.arange(4), rtol=0, atol=1e-9)
    np.testing.assert_allclose(z_axis, [0.0, 2e-3], rtol=0, atol=1e-9)


def test_xyz_axes_of_several_slice_packages_log_a_warning(caplog):
    """Regression: an axial and a coronal package make y vary along two axes.

    Previously the identity ``acquisition_matrix`` of the first package
    suppressed the warning, as for the three-package localizers.
    """
    scan = _VisuScan(
        matrix=np.array([4, 3, 2]),
        acquisition_matrix=np.eye(3),
        visucore_position=np.array([[-2.0, -1.0, 0.0], [-2.0, 4.0, 5.0]]),
        visucore_orientation=np.array([AXIAL_ROWS, CORONAL_ROWS]),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([4.0, 3.0]),
    )

    (_, y_axis, _), messages = _xyz_axes_warnings(caplog, scan)

    assert len(messages) == 1
    assert "slice_idx" in messages[0]
    np.testing.assert_allclose(
        y_axis, [-1.0, 0.0, 1.0, 4.0], rtol=0, atol=POSITION_ATOL_MM
    )


def test_assign_xyz_coordinates_preserves_rps_dims_and_adds_aux_coords():
    handler = SimpleNamespace(
        matrix=np.array([2, 2, 1]),
        visucore_position=np.array([[1.0, 2.0, 3.0]]),
        visucore_orientation=np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        visucore_size=np.array([2, 2]),
        visucore_extent=np.array([2.0, 2.0]),
    )
    data = xr.DataArray(
        np.zeros((2, 2, 1)),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    result = assign_xyz_coordinates(data, scan=handler, dtype=np.float64)

    assert result.dims == ("read_idx", "phase_idx", "slice_idx")
    for coord_name in ("x", "y", "z"):
        assert coord_name in result.coords
        assert result.coords[coord_name].dims == (
            "read_idx",
            "phase_idx",
            "slice_idx",
        )
        assert result.coords[coord_name].attrs["unit"] == "mm"


def test_assign_xyz_coordinates_supports_single_slice_2d_data_without_slice_dim():
    handler = SimpleNamespace(
        matrix=np.array([2, 3, 1]),
        visucore_position=np.array([[1.0, 2.0, 3.0]]),
        visucore_orientation=np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        visucore_size=np.array([2, 3]),
        visucore_extent=np.array([2.0, 6.0]),
    )
    data = xr.DataArray(
        np.zeros((4, 2, 3)),
        dims=("acquisition_idx", "read_idx", "phase_idx"),
    )

    result = assign_xyz_coordinates(data, scan=handler, dtype=np.float64)

    assert result.dims == ("acquisition_idx", "read_idx", "phase_idx")
    for coord_name in ("x", "y", "z"):
        assert coord_name in result.coords
        assert result.coords[coord_name].dims == ("read_idx", "phase_idx")
        assert result.coords[coord_name].shape == (2, 3)
        assert result.coords[coord_name].attrs["unit"] == "mm"

    np.testing.assert_allclose(result.coords["x"].values[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(result.coords["y"].values[0, :], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(result.coords["z"].values, 3.0)


def test_coordinate_precision_has_one_float64_owner():
    """Coordinates in mm are stored as float64 by one owner."""
    from hypermri.utils import coordinates
    from hypermri.utils.coordinates import grids, transformations

    assert transformations.COORDINATE_PRECISION is np.float64
    assert grids.COORDINATE_PRECISION is transformations.COORDINATE_PRECISION
    assert coordinates.COORDINATE_PRECISION is transformations.COORDINATE_PRECISION


def test_position_tolerance_has_one_owner():
    """The geometry checks share the position tolerance of 1e-3 mm.

    The value is the tolerance of the ParaVision reference tests (``POSITION_ATOL_MM``
    of this module); the modules must not keep copies that can diverge.
    """
    from hypermri.reconstruction import consistency
    from hypermri.utils.coordinates import alignment, transformations

    assert transformations._POSITION_ATOL_MM == POSITION_ATOL_MM
    for module in (grids, alignment, consistency):
        assert module._POSITION_ATOL_MM is transformations._POSITION_ATOL_MM
        assert not [
            name
            for name in vars(module)
            if name.endswith("ATOL_MM") and name != "_POSITION_ATOL_MM"
        ]


def test_assign_xyz_coordinates_default_resolves_sub_mm_offsets_far_from_isocentre():
    """Default coordinates keep 0.1 mm voxel positions about 100 mm off-centre.

    Synthetic axis-aligned VisuCore geometry with 0.1 mm voxels whose first
    voxel lies near (100, -100, 99) mm.  float16 resolves only 0.0625 mm at
    this distance (errors up to about 0.03 mm, a third of a voxel).  The
    tolerance of 1e-9 mm is far below any MR voxel size and far above float64
    rounding (about 1e-14 mm at 100 mm).
    """
    origin = np.array([100.03, -100.07, 99.01])
    handler = SimpleNamespace(
        matrix=np.array([4, 3, 2]),
        visucore_position=np.array([origin, origin + [0.0, 0.0, 0.1]]),
        visucore_orientation=np.array(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * 2
        ),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([0.4, 0.3]),
    )
    data = xr.DataArray(
        np.zeros((4, 3, 2)),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    result = assign_xyz_coordinates(data, scan=handler)

    for coord_name in ("x", "y", "z"):
        assert result.coords[coord_name].dtype == np.float64
    np.testing.assert_allclose(
        result.coords["x"].values[:, 0, 0],
        origin[0] + 0.1 * np.arange(4),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.coords["y"].values[0, :, 0],
        origin[1] + 0.1 * np.arange(3),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.coords["z"].values[0, 0, :],
        origin[2] + 0.1 * np.arange(2),
        rtol=0,
        atol=1e-9,
    )


class _ScanParameters(SimpleNamespace):
    """Synthetic scan parameters with the ``get`` access of the vendor handlers."""

    def get(self, name, default=None):
        return getattr(self, name, default)


class _ScanWithProperties(UniformDataPropertiesMixin, _ScanParameters):
    """Synthetic scan parameters with the cached properties of the handlers."""


AXIAL = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _axial_stack(slice_z_mm, **extra):
    """Return an axial stack of 4 x 2 voxels of 1 mm at the given slice z."""
    n_slices = len(slice_z_mm)
    return _ScanParameters(
        matrix=np.array([4, 2, n_slices]),
        visucore_position=np.array([[-2.0, -1.0, z] for z in slice_z_mm]),
        visucore_orientation=np.array([AXIAL] * n_slices),
        visucore_size=np.array([4, 2]),
        visucore_extent=np.array([4.0, 2.0]),
        **extra,
    )


def test_get_xyz_extent_includes_half_the_slice_distance_beyond_outer_slices():
    """Regression: the slice gap belongs to the voxel box of a 2D stack.

    Synthetic axial stack of three 1 mm slices with 4 mm gaps, 5 mm apart.
    The RPS parameters describe the same stack as ``BrukerHandler`` builds
    them, with the slice field of view from the first to the last slice edge
    (11 mm).  Previously the box ended 0.5 mm beyond the outer slice centres
    instead of half the slice distance, 2.5 mm, as the DICOM voxel box of the
    ParaVision reference series does.
    """
    scan = _axial_stack(
        [-5.0, 0.0, 5.0],
        visucore_thickness=1.0,
        fov_mm=np.array([4.0, 2.0, 11.0]),
        read_offset_mm=-0.5,
        phase_offset_mm=-0.5,
        slice_offset_mm=0.0,
    )

    extent = get_xyz_extent(scan)

    np.testing.assert_allclose(
        extent,
        [[-2.5, 1.5], [-1.5, 0.5], [-7.5, 7.5]],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_get_xyz_extent_of_single_slice_uses_frame_thickness_along_normal():
    """A single slice spans ``VisuCoreFrameThickness`` along its normal.

    Synthetic sagittal slice with the geometry of
    ``test_get_xyz_grid_from_visu_maps_swapped_inplane_axes``: read runs
    along -z, phase along +y and the slice normal along -x, with 2 mm voxels.
    """
    scan = _ScanParameters(
        matrix=np.array([3, 4, 1]),
        visucore_position=np.array([[10.0, 20.0, 30.0]]),
        visucore_orientation=np.array(
            [[0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]]
        ),
        visucore_size=np.array([4, 3]),
        visucore_extent=np.array([8.0, 6.0]),
        visucore_thickness=0.5,
    )

    extent = get_xyz_extent(scan)

    np.testing.assert_allclose(
        extent,
        [[9.75, 10.25], [19.0, 27.0], [25.0, 31.0]],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_get_xyz_extent_of_oblique_slice_encloses_the_tilted_voxels():
    """The box is aligned with x, y and z and encloses tilted voxels.

    Synthetic single 1 mm slice of 2 x 2 voxels of 1 mm, rotated by 45
    degrees about z.  Each tilted voxel reaches ``(cos 45 + sin 45) / 2``
    beyond its centre along x and y.
    """
    c = np.sqrt(0.5)
    scan = _ScanParameters(
        matrix=np.array([2, 2, 1]),
        visucore_position=np.array([[0.0, 0.0, 0.0]]),
        visucore_orientation=np.array([[c, c, 0.0, -c, c, 0.0, 0.0, 0.0, 1.0]]),
        visucore_size=np.array([2, 2]),
        visucore_extent=np.array([2.0, 2.0]),
        visucore_thickness=1.0,
    )

    x_extent, y_extent, z_extent = get_xyz_extent(scan)

    np.testing.assert_allclose(x_extent, [-c - c, c + c], rtol=0, atol=1e-9)
    np.testing.assert_allclose(y_extent, [-c, 2 * c + c], rtol=0, atol=1e-9)
    np.testing.assert_allclose(z_extent, [-0.5, 0.5], rtol=0, atol=1e-9)


def test_get_xyz_extent_rejects_single_slice_without_frame_thickness():
    """Without ``VisuCoreFrameThickness`` a single slice has no known extent."""
    scan = _axial_stack([0.0])

    with pytest.raises(ValueError, match="VisuCoreFrameThickness"):
        get_xyz_extent(scan)


def test_get_xyz_extent_rejects_unequally_spaced_slices():
    """Slices of several packages do not form one stack with one slice step."""
    scan = _axial_stack([0.0, 5.0, 7.0], visucore_thickness=1.0)

    with pytest.raises(ValueError, match="several slice packages"):
        get_xyz_extent(scan)


def test_get_xyz_extent_rejects_a_single_voxel_along_read():
    """One voxel along read leaves the read voxel size unknown."""
    scan = _ScanParameters(
        matrix=np.array([1, 2, 1]),
        visucore_position=np.array([[0.0, 0.0, 0.0]]),
        visucore_orientation=np.array([AXIAL]),
        visucore_size=np.array([1, 2]),
        visucore_extent=np.array([1.0, 2.0]),
        visucore_thickness=1.0,
    )

    with pytest.raises(ValueError, match="at least two voxels"):
        get_xyz_extent(scan)


def test_xyz_extent_is_unavailable_without_visucore_geometry():
    """Regression: without ``VisuCore`` there is no XYZ extent.

    Previously the extent came from the RPS parameters and the identity matrix
    when the ``VisuCore`` geometry was missing.
    """
    scan = _ScanWithProperties(
        matrix=np.array([4, 2, 1]),
        fov_mm=np.array([4.0, 2.0, 1.0]),
        read_offset_mm=0.0,
        phase_offset_mm=0.0,
        slice_offset_mm=0.0,
    )

    with pytest.raises(ValueError, match="VisuCore"):
        get_xyz_extent(scan)
    assert scan.xyz_extent_mm is None


def test_define_xyz_grid_centres_each_line_half_a_voxel_below_zero():
    """The documented example: zero-centred lines moved by half a voxel."""
    lines = define_xyz_grid(fov=(4.0, 6.0, 2.0), mat=(4, 3, 1))

    for line, expected in zip(
        lines, ([-2.0, -1.0, 0.0, 1.0], [-3.0, -1.0, 1.0], [-1.0]), strict=True
    ):
        np.testing.assert_allclose(line, expected, rtol=0, atol=POSITION_ATOL_MM)


def test_xyz_grid_scan_defaults_pair_fov_and_matrix_in_zxy_order():
    """Scan-derived FOV and matrix vectors both use z/x/y order."""
    scan = SimpleNamespace(
        matrix=np.array([32, 48, 5]),
        fov_mm=np.array([32.0, 48.0, 21.0]),
        spatial_dimension="3D",
        patient_position="Head_Prone",
        slice_orientation="axial",
        read_orientation="L_R",
    )
    original_matrix, original_fov = scan.matrix.copy(), scan.fov_mm.copy()
    first = define_xyz_grid(scan)
    second = define_xyz_grid(scan)
    expected_endpoints = [(-10.5, 6.3), (-16.0, 15.0), (-24.0, 23.0)]
    former_mismatched_endpoints = [(-16.0, 9.6), (-24.0, 22.5), (-10.5, 10.0625)]
    assert tuple(line.size for line in first) == (5, 32, 48)
    for line, repeated, endpoints, former_endpoints in zip(
        first, second, expected_endpoints, former_mismatched_endpoints, strict=True
    ):
        np.testing.assert_allclose(
            line[[0, -1]], endpoints, rtol=0, atol=POSITION_ATOL_MM
        )
        np.testing.assert_array_equal(line, repeated)
        assert line.dtype == np.float64
        assert not np.allclose(line[[0, -1]], former_endpoints, rtol=0, atol=0)
    np.testing.assert_array_equal(scan.matrix, original_matrix)
    np.testing.assert_array_equal(scan.fov_mm, original_fov)


@pytest.mark.parametrize(
    ("explicit", "expected_endpoints"),
    [
        ("fov", [(-10.5, 6.3), (-16.0, 15.0), (-24.0, 23.0)]),
        ("mat", [(-10.5, 6.3), (-16.0, 15.0), (-24.0, 23.0)]),
    ],
)
def test_xyz_grid_mixed_explicit_and_scan_defaults_use_matching_zxy_order(
    explicit, expected_endpoints
):
    scan = SimpleNamespace(
        matrix=np.array([32, 48, 5]),
        fov_mm=np.array([32.0, 48.0, 21.0]),
        spatial_dimension="3D",
        patient_position="Head_Prone",
        slice_orientation="axial",
        read_orientation="L_R",
    )
    original_matrix, original_fov = scan.matrix.copy(), scan.fov_mm.copy()
    if explicit == "fov":
        supplied = np.array([21.0, 32.0, 48.0])
        lines = define_xyz_grid(scan, fov=supplied)
        repeated = define_xyz_grid(scan, fov=supplied)
    else:
        supplied = np.array([5, 32, 48])
        lines = define_xyz_grid(scan, mat=supplied)
        repeated = define_xyz_grid(scan, mat=supplied)

    assert tuple(line.size for line in lines) == (5, 32, 48)
    for line, repeat_line, endpoints in zip(
        lines, repeated, expected_endpoints, strict=True
    ):
        np.testing.assert_allclose(
            line[[0, -1]], endpoints, rtol=0, atol=POSITION_ATOL_MM
        )
        np.testing.assert_array_equal(line, repeat_line)
        assert line.dtype == np.float64
    np.testing.assert_array_equal(
        supplied, [21.0, 32.0, 48.0] if explicit == "fov" else [5, 32, 48]
    )
    np.testing.assert_array_equal(scan.matrix, original_matrix)
    np.testing.assert_array_equal(scan.fov_mm, original_fov)


@pytest.mark.parametrize("patient_position", ["Head_Prone", "Foot_Prone"])
@pytest.mark.parametrize("explicit", [None, "mat", "fov"])
def test_coronal_lr_grid_defaults_use_approved_fov_pairing(patient_position, explicit):
    """Derived FOV is corrected; caller-supplied FOV is retained verbatim."""
    scan = SimpleNamespace(
        matrix=np.array([4, 3, 2]),
        fov_mm=np.array([8.0, 9.0, 6.0]),
        spatial_dimension="3D",
        patient_position=patient_position,
        slice_orientation="coronal",
        read_orientation="L_R",
        read_offset_mm=100.0,
        phase_offset_mm=200.0,
    )
    supplied = (
        np.array([12.0, 16.0, 10.0]) if explicit == "fov" else np.array([3, 4, 2])
    )
    options = {} if explicit is None else {explicit: supplied}
    expected = (
        ([-6.0, -2.0, 2.0], [-8.0, -4.0, 0.0, 4.0], [-5.0, 0.0])
        if explicit == "fov"
        else ([-4.5, -1.5, 1.5], [-4.0, -2.0, 0.0, 2.0], [-3.0, 0.0])
    )
    first = define_xyz_grid(scan, **options)
    for actual, repeated, reference in zip(
        first, define_xyz_grid(scan, **options), expected, strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual, repeated)
        assert actual.dtype == np.float64
    np.testing.assert_array_equal(scan.fov_mm, [8.0, 9.0, 6.0])
    np.testing.assert_array_equal(scan.matrix, [4, 3, 2])
    np.testing.assert_array_equal(
        supplied, [12.0, 16.0, 10.0] if explicit == "fov" else [3, 4, 2]
    )
