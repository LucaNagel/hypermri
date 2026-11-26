"""Reference tests of scan geometry against ParaVision DICOM geometry.

The ParaVision reference fixture ``tests/reference/paravision_water_phantom_geometry.json``
holds, for 51 water-phantom series, the DICOM geometry exported by ParaVision
and the ParaVision parameters that hypermri reads.  Each test writes those
parameters back as ParaVision parameter files, loads the scan with
``load_scan`` and compares the grid of ``get_xyz_grid_from_visu`` or
``get_xyz_grid`` with the DICOM voxel centres.  Without ``visu_pars`` no
scanner grid is derived: the grid functions raise and
``assign_xyz_coordinates`` attaches no ``x``, ``y`` or ``z``.  The
tests compare the RPS axes, grid and extent, the XYZ extent and the scan
alignment with values derived from the DICOM voxel centres; scans with
several slice packages have no RPS positions.

Reference convention, in DICOM LPS millimetres:

* ``ImagePositionPatient`` is the centre of the first voxel of a frame, so the
  voxel in DICOM row ``i`` and column ``j`` of frame ``k`` lies at
  ``IPP[k] + j * PixelSpacing[1] * IOP[k][:3] + i * PixelSpacing[0] * IOP[k][3:]``.
* Frames in ``InstanceNumber`` order are the ``slice_idx`` order.
* ``RECO_transposition == 0`` stores ``read_idx`` along the DICOM columns and
  ``phase_idx`` along the rows; ``RECO_transposition == 1`` swaps them.  The
  ParaVision reference fixture test checks this rule against the DICOM matrix shape.  The
  multi-package localizers mix both values, so the rule applies per frame
 .
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from hypermri.loading.scans import load_scan
from hypermri.utils.coordinates import grids
from hypermri.utils.coordinates.alignment import compute_scan_alignment
from hypermri.utils.coordinates.grids import (
    assign_xyz_coordinates,
    get_xyz_axes,
    get_xyz_grid,
    get_xyz_grid_from_visu,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "reference" / "paravision_water_phantom_geometry.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

# The geometry position tolerance.  The DICOM positions carry six decimals in mm
# and the pixel spacings are 0.25 mm or more, so half a voxel or a wrong slice
# step exceeds it by two orders of magnitude.
POSITION_ATOL_MM = 1e-3

PARAMETER_FILES = {
    "method": Path("method"),
    "acqp": Path("acqp"),
    "visu_pars": Path("visu_pars"),
    "reco": Path("pdata") / "1" / "reco",
}
HANDLER_PARAMETER_ATTRIBUTES = {
    "method": "method",
    "acqp": "acqp",
    "visu_pars": "visu",
    "reco": "reco",
}

MISSING_VISUCORE = "Missing VisuCore geometry metadata"


def _is_multi_package(series: dict[str, Any]) -> bool:
    return isinstance(series["paravision"]["method"]["PVM_SPackArrNSlices"], list)


def _series_params():
    return [pytest.param(name, id=name.replace("/", "-")) for name in FIXTURE["series"]]


# --------------------------------------------------------------------------
# Reference grid and rebuilt ParaVision scans
# --------------------------------------------------------------------------


def _dicom_reference_grid(series: dict[str, Any]) -> np.ndarray:
    """Return DICOM voxel centres with shape ``(read, phase, slice, 3)`` in mm."""
    dicom = series["dicom"]
    positions = np.asarray(dicom["image_position_patient_mm"], dtype=np.float64)
    orientations = np.asarray(dicom["image_orientation_patient"], dtype=np.float64)
    row_spacing, column_spacing = dicom["pixel_spacing_mm"]
    rows = np.arange(dicom["rows"], dtype=np.float64)[:, None, None]
    columns = np.arange(dicom["columns"], dtype=np.float64)[None, :, None]
    transposition = series["paravision"]["reco"]["RECO_transposition"]

    slices = []
    for position, orientation, transposed in zip(
        positions, orientations, transposition, strict=True
    ):
        frame = (
            position
            + columns * column_spacing * orientation[:3]
            + rows * row_spacing * orientation[3:]
        )
        slices.append(frame if transposed else frame.transpose(1, 0, 2))
    return np.stack(slices, axis=2)


@pytest.fixture(scope="module")
def handlers(tmp_path_factory, write_paravision_parameters):
    """Load every fixture series twice: with and without ``visu_pars``."""
    root = tmp_path_factory.mktemp("paravision_reference")
    loaded = {}
    for with_visu_pars in (True, False):
        variant_root = root / ("visu" if with_visu_pars else "no_visu")
        for name, series in FIXTURE["series"].items():
            scan_dir = variant_root / name
            for section, relative_path in PARAMETER_FILES.items():
                if section != "visu_pars" or with_visu_pars:
                    write_paravision_parameters(
                        scan_dir / relative_path, series["paravision"][section]
                    )
            experiment = load_scan(
                scan_dir, mode="lazy", resolve_sequence=False, log_mode="critical"
            )
            loaded[name, with_visu_pars] = experiment.handler
    return loaded


def _spatial_shape(series: dict[str, Any]) -> tuple[int, int, int]:
    read_size, phase_size = series["paravision"]["method"]["PVM_Matrix"]
    return read_size, phase_size, len(series["dicom"]["instance_numbers"])


def _assert_grid_matches_dicom(grid, series: dict[str, Any], name: str) -> None:
    """Compare a grid with the DICOM reference: first voxel, slice order, all voxels."""
    actual = np.stack(grid, axis=-1)
    expected = _dicom_reference_grid(series)
    assert actual.shape == expected.shape, name

    first_position = series["dicom"]["image_position_patient_mm"][0]
    np.testing.assert_allclose(
        actual[0, 0, 0],
        first_position,
        rtol=0,
        atol=POSITION_ATOL_MM,
        err_msg=f"{name}: first voxel centre differs from ImagePositionPatient",
    )
    np.testing.assert_allclose(
        actual[0, 0, :],
        expected[0, 0, :],
        rtol=0,
        atol=POSITION_ATOL_MM,
        err_msg=f"{name}: slice positions differ from the InstanceNumber order",
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
        err_msg=f"{name}: voxel centres differ from the DICOM reference",
    )


# --------------------------------------------------------------------------
# Rebuilt scans
# --------------------------------------------------------------------------


def test_rebuilt_scans_reproduce_the_fixture_parameters(handlers):
    """The handler parses the written files back to the fixture values."""
    for name, series in FIXTURE["series"].items():
        handler = handlers[name, True]
        for section, attribute in HANDLER_PARAMETER_ATTRIBUTES.items():
            parsed = getattr(handler, attribute)
            for parameter, expected in series["paravision"][section].items():
                value = parsed.get(parameter)
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                assert value == expected, (name, section, parameter)
        assert handlers[name, False].visu == {}, name


def test_dicom_reference_grid_places_the_first_voxel_at_image_position():
    """The reference uses voxel centres: index 0 lies at ``ImagePositionPatient``."""
    series = FIXTURE["series"]["head_prone/14"]
    grid = _dicom_reference_grid(series)
    dicom = series["dicom"]
    row_spacing, column_spacing = dicom["pixel_spacing_mm"]
    orientation = np.asarray(dicom["image_orientation_patient"][0])

    assert series["paravision"]["reco"]["RECO_transposition"][0] == 1
    assert grid.shape == (dicom["rows"], dicom["columns"], 5, 3)
    np.testing.assert_array_equal(grid[0, 0, :], dicom["image_position_patient_mm"])
    np.testing.assert_allclose(
        grid[1, 0, 0] - grid[0, 0, 0], row_spacing * orientation[3:]
    )
    np.testing.assert_allclose(
        grid[0, 1, 0] - grid[0, 0, 0], column_spacing * orientation[:3]
    )


# --------------------------------------------------------------------------
# Grids against the DICOM reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid_function", [get_xyz_grid_from_visu, get_xyz_grid], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("name", _series_params())
def test_visucore_grid_matches_dicom_voxel_centres(handlers, grid_function, name):
    """``VisuCore`` geometry reproduces the DICOM voxel centres and slice order.

    Previously the frames with ``RECO_transposition`` 1 of the two
    multi-package localizers lay up to 89.803 mm from their voxel centres.
    """
    series = FIXTURE["series"][name]
    grid = grid_function(
        scan=handlers[name, True], spatial_shape=_spatial_shape(series)
    )
    _assert_grid_matches_dicom(grid, series, name)


# --------------------------------------------------------------------------
# Scans without VisuCore geometry
# --------------------------------------------------------------------------
#
# Previously ``get_xyz_grid`` fell back to an RPS grid labelled ``dicom_lps``
# that deviated from the DICOM reference by 1.6 mm (16 series), 32 to 56 mm
# (33 series, two reversed index directions) and 71.05 mm (localizers).


@pytest.mark.parametrize("name", _series_params())
def test_xyz_grid_without_visucore_raises(handlers, name):
    """Without ``visu_pars`` the grid functions raise instead of guessing."""
    series = FIXTURE["series"][name]
    handler = handlers[name, False]
    spatial_shape = _spatial_shape(series)

    for grid_function in (get_xyz_grid_from_visu, get_xyz_grid):
        with pytest.raises(ValueError, match=MISSING_VISUCORE):
            grid_function(scan=handler, spatial_shape=spatial_shape)
    with pytest.raises(ValueError, match=MISSING_VISUCORE):
        get_xyz_axes(scan=handler)
    assert handler.xyz_grid_mm is None
    assert handler.xyz_axes_mm is None


@pytest.mark.parametrize("name", _series_params())
def test_assign_xyz_coordinates_without_visucore_attaches_nothing(
    handlers, name, caplog
):
    """Without ``visu_pars`` the data keep only their index coordinates."""
    series = FIXTURE["series"][name]
    data = xr.DataArray(
        np.zeros(_spatial_shape(series), dtype=np.float32),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    with caplog.at_level(logging.WARNING, logger=grids.__name__):
        result = assign_xyz_coordinates(data, scan=handlers[name, False])

    assert not {"x", "y", "z"} & set(result.coords)
    assert result.identical(data)
    assert MISSING_VISUCORE in caplog.text


def test_xyz_grid_without_offsets_raises(handlers):
    """``with_offsets=False`` has no scanner-space meaning and is rejected."""
    series = FIXTURE["series"]["head_prone/14"]
    with pytest.raises(ValueError, match="with_offsets=False"):
        get_xyz_grid(
            scan=handlers["head_prone/14", True],
            with_offsets=False,
            spatial_shape=_spatial_shape(series),
        )


# --------------------------------------------------------------------------
# RPS geometry, XYZ extent and scan alignment
# --------------------------------------------------------------------------
#
# The expected values come from the DICOM voxel centres alone.  The rows of
# the index directions are the unit vectors of increasing ``read_idx``,
# ``phase_idx`` and ``slice_idx``; the RPS position of a voxel is its DICOM
# position projected onto them.  Extents follow the pixel-centre convention:
# the first and last voxel centres widened by half a step.  The multi-package
# localizers have no RPS positions.


def _single_package_params():
    return [
        pytest.param(name, id=name.replace("/", "-"))
        for name, series in FIXTURE["series"].items()
        if not _is_multi_package(series)
    ]


def _multi_package_params():
    return [
        pytest.param(name, id=name.replace("/", "-"))
        for name, series in FIXTURE["series"].items()
        if _is_multi_package(series)
    ]


def _alignment_pairs(parallel: bool):
    """Return ``(reference, source)`` pairs of one study, one series per orientation.

    Parallel pairs share the slice orientation; the others have crossing
    slice planes.
    """
    representatives = {}
    for name, series in FIXTURE["series"].items():
        if _is_multi_package(series):
            continue
        method = series["paravision"]["method"]
        key = (
            name.split("/")[0],
            method["PVM_SPackArrSliceOrient"],
            method["PVM_SPackArrReadOrient"],
        )
        representatives.setdefault(key, name)
    pairs = []
    for (reference_key, reference), (source_key, source) in itertools.permutations(
        representatives.items(), 2
    ):
        if reference_key[0] != source_key[0]:
            continue
        if (reference_key[1] == source_key[1]) == parallel:
            pairs.append(
                pytest.param(
                    reference,
                    source,
                    id=f"{reference}-from-{source}".replace("/", "-"),
                )
            )
    return pairs


def _index_geometry(series: dict[str, Any]):
    """Return DICOM voxel centres, unit index directions and index steps in mm."""
    grid = _dicom_reference_grid(series)
    assert min(grid.shape[:3]) > 1
    steps = np.stack(
        [
            grid[1, 0, 0] - grid[0, 0, 0],
            grid[0, 1, 0] - grid[0, 0, 0],
            grid[0, 0, 1] - grid[0, 0, 0],
        ]
    )
    spacing = np.linalg.norm(steps, axis=1)
    return grid, steps / spacing[:, None], spacing


def _expected_rps(series: dict[str, Any]):
    """Return the expected RPS voxel centres ``(read, phase, slice, 3)`` and extent."""
    grid, directions, spacing = _index_geometry(series)
    rps = grid @ directions.T
    extent = []
    for axis in range(3):
        extent += [
            rps[..., axis].min() - spacing[axis] / 2,
            rps[..., axis].max() + spacing[axis] / 2,
        ]
    return rps, np.asarray(extent)


def _assert_rps_axes(handler, expected_rps, axes, name):
    actual_grid = np.stack(handler.rps_grid_mm, axis=-1)
    expected_axes = (
        expected_rps[:, 0, 0, 0],
        expected_rps[0, :, 0, 1],
        expected_rps[0, 0, :, 2],
    )
    for axis in axes:
        np.testing.assert_allclose(
            actual_grid[..., axis],
            expected_rps[..., axis],
            rtol=0,
            atol=POSITION_ATOL_MM,
            err_msg=f"{name}: rps_grid_mm axis {axis}",
        )
        np.testing.assert_allclose(
            handler.rps_axes_mm[axis],
            expected_axes[axis],
            rtol=0,
            atol=POSITION_ATOL_MM,
            err_msg=f"{name}: rps_axes_mm axis {axis}",
        )


@pytest.mark.parametrize("name", _single_package_params())
def test_rps_read_and_phase_are_dicom_positions_along_the_index_directions(
    handlers, name
):
    """Read and phase RPS coordinates project the DICOM voxel centres."""
    expected_rps, _ = _expected_rps(FIXTURE["series"][name])
    _assert_rps_axes(handlers[name, True], expected_rps, (0, 1), name)


@pytest.mark.parametrize("name", _single_package_params())
def test_rps_slice_is_the_dicom_position_along_the_slice_direction(handlers, name):
    """Slice RPS coordinates step by slice thickness plus gap."""
    expected_rps, _ = _expected_rps(FIXTURE["series"][name])
    _assert_rps_axes(handlers[name, True], expected_rps, (2,), name)


@pytest.mark.parametrize("name", _single_package_params())
def test_rps_extent_read_and_phase_bound_the_dicom_voxels(handlers, name):
    """The in-plane RPS extent is the voxel-centre range widened by half a voxel."""
    _, expected_extent = _expected_rps(FIXTURE["series"][name])
    np.testing.assert_allclose(
        handlers[name, True].rps_extent_mm[:4],
        expected_extent[:4],
        rtol=0,
        atol=POSITION_ATOL_MM,
        err_msg=name,
    )


@pytest.mark.parametrize("name", _single_package_params())
def test_rps_extent_slice_bounds_the_dicom_slices(handlers, name):
    """The slice RPS extent widens the slice-centre range by half a slice step."""
    _, expected_extent = _expected_rps(FIXTURE["series"][name])
    np.testing.assert_allclose(
        handlers[name, True].rps_extent_mm[4:],
        expected_extent[4:],
        rtol=0,
        atol=POSITION_ATOL_MM,
        err_msg=name,
    )


@pytest.mark.parametrize("with_visu_pars", [True, False])
@pytest.mark.parametrize("name", _multi_package_params())
def test_rps_positions_of_several_slice_packages_are_rejected(
    handlers, name, with_visu_pars
):
    """Regression test: one RPS frame cannot hold three packages.

    The localizers place an axial, a sagittal and a coronal package of seven
    slices each.  Previously the RPS axes, grid and extent described the
    first package only; projected onto the index directions of the DICOM
    frames, the positions of the other packages deviated by up to 28.755 mm
    along the slice and 6.34 mm along the phase direction.  Now the RPS
    functions raise and the handler properties are None.
    """
    handler = handlers[name, with_visu_pars]
    for function in (grids.get_rps_axes, grids.get_rps_grid, grids.get_rps_extent):
        with pytest.raises(ValueError, match=r"3 slice packages \(7, 7, 7\)"):
            function(handler)
    assert handler.rps_axes_mm is None, name
    assert handler.rps_grid_mm is None, name
    assert handler.rps_extent_mm is None, name


@pytest.mark.parametrize("name", _single_package_params())
def test_xyz_extent_bounds_the_dicom_voxels(handlers, name):
    """The XYZ extent is the bounding box of the voxels, each a half step around its centre."""
    grid, directions, spacing = _index_geometry(FIXTURE["series"][name])
    half_width = np.abs(directions).T @ (spacing / 2)
    centres = grid.reshape(-1, 3)
    expected = np.stack(
        [centres.min(axis=0) - half_width, centres.max(axis=0) + half_width], axis=1
    )

    extent = handlers[name, True].xyz_extent_mm
    assert extent is not None, name
    np.testing.assert_allclose(
        np.asarray(extent, dtype=np.float64),
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
        err_msg=name,
    )


def _expected_alignment(reference_series, source_series):
    """Return the expected axis order, aligned grid and extent of a parallel pair.

    The source voxel centres are projected onto the reference index directions
    and their spatial axes are reordered to the reference axes.  The extent
    holds the edges on the side of the first and of the last source voxel,
    so a reversed axis gives a decreasing pair.
    """
    _, reference_directions, _ = _index_geometry(reference_series)
    source_grid, source_directions, _ = _index_geometry(source_series)
    rotation = reference_directions @ source_directions.T
    order = tuple(int(np.argmax(np.abs(row))) for row in rotation)
    aligned = np.transpose(source_grid @ reference_directions.T, (*order, 3))

    extent = []
    for axis in range(3):
        index = [0, 0, 0]
        first = aligned[tuple(index)][axis]
        index[axis] = -1
        last = aligned[tuple(index)][axis]
        step = (last - first) / (aligned.shape[axis] - 1)
        extent += [first - step / 2, last + step / 2]
    return order, aligned, np.asarray(extent)


def _assert_alignment(handlers, reference, source, axes):
    order, expected_grid, expected_extent = _expected_alignment(
        FIXTURE["series"][reference], FIXTURE["series"][source]
    )
    result = compute_scan_alignment(
        source_obj=handlers[source, True], reference_obj=handlers[reference, True]
    )
    assert result.transpose_order == order, (reference, source)
    actual_grid = np.stack(result.rps_grid_mm_transformed, axis=-1)
    for axis in axes:
        np.testing.assert_allclose(
            actual_grid[..., axis],
            expected_grid[..., axis],
            rtol=0,
            atol=POSITION_ATOL_MM,
            err_msg=f"{reference} from {source}: aligned grid axis {axis}",
        )
        np.testing.assert_allclose(
            result.extent_mm[2 * axis : 2 * axis + 2],
            expected_extent[2 * axis : 2 * axis + 2],
            rtol=0,
            atol=POSITION_ATOL_MM,
            err_msg=f"{reference} from {source}: extent axis {axis}",
        )


@pytest.mark.parametrize(("reference", "source"), _alignment_pairs(parallel=True))
def test_alignment_of_parallel_planes_in_the_reference_plane(
    handlers, reference, source
):
    """Parallel slice planes align read and phase, including reversed directions."""
    _assert_alignment(handlers, reference, source, axes=(0, 1))


@pytest.mark.parametrize(("reference", "source"), _alignment_pairs(parallel=True))
def test_alignment_of_parallel_planes_along_the_reference_slice(
    handlers, reference, source
):
    """Parallel slice planes align the slice positions and slice extent."""
    _assert_alignment(handlers, reference, source, axes=(2,))


@pytest.mark.parametrize(("reference", "source"), _alignment_pairs(parallel=False))
def test_alignment_of_crossing_planes_raises(handlers, reference, source):
    """Crossing slice planes have no 2D overlay and are rejected."""
    with pytest.raises(ValueError, match="not parallel"):
        compute_scan_alignment(
            source_obj=handlers[source, True], reference_obj=handlers[reference, True]
        )
