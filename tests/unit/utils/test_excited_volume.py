"""Tests for the excited volume of single-voxel and slice spectroscopy.

All geometry is invented. The conversion from the Bruker gradient frame to
DICOM LPS for Head_Prone (flip x and y) was checked on local, Git-ignored
imaging scans; these tests fix that convention on synthetic parameters.
Positions are compared with 1e-9 mm, well below any voxel size.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.utils.coordinates.excited_volume import (
    ExcitedVolume,
    excited_volume,
    image_slice_volume,
    outline_in_plane,
)
from hypermri.vendors.bruker import BrukerHandler

ATOL_MM = 1e-9


def _rotation_about_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _press(gradient=None, rps=(2.0, -3.0, 4.0), size=(4.5, 3.5, 7.5)):
    gradient = np.eye(3) if gradient is None else gradient
    method = {
        "PVM_VoxArrSize": [list(size)],
        "PVM_VoxArrPositionRPS": [list(rps)],
        "PVM_VoxArrGradOrient": [np.asarray(gradient).tolist()],
    }
    return SimpleNamespace(method=method, patient_position="Head_Prone")


def test_voxel_centre_and_axes_follow_the_head_prone_convention():
    volume = excited_volume(_press())

    assert volume.kind == "voxel"
    # Head_Prone flips x and y of the gradient frame.
    np.testing.assert_allclose(volume.centre_mm, [-2.0, 3.0, 4.0], atol=ATOL_MM)
    np.testing.assert_allclose(np.abs(volume.axes), np.eye(3), atol=ATOL_MM)
    np.testing.assert_allclose(volume.size_mm, [4.5, 3.5, 7.5])


def test_rotated_voxel_contains_its_corners_only():
    gradient = _rotation_about_z(30.0)
    volume = excited_volume(_press(gradient=gradient))
    half = volume.size_mm / 2
    corner = volume.centre_mm + half @ volume.axes
    beyond = volume.centre_mm + 1.01 * half @ volume.axes

    assert volume.contains(np.stack([volume.centre_mm, corner])).all()
    assert not volume.contains(beyond)
    # The axes stay orthonormal.
    np.testing.assert_allclose(volume.axes @ volume.axes.T, np.eye(3), atol=1e-12)


def test_scan_object_is_read_through_its_handler():
    scan = SimpleNamespace(handler=_press())

    np.testing.assert_allclose(
        excited_volume(scan).centre_mm, [-2.0, 3.0, 4.0], atol=ATOL_MM
    )


def test_voxel_from_paravision_parameter_files(tmp_path, write_paravision_parameters):
    scan_dir = tmp_path / "1"
    write_paravision_parameters(
        scan_dir / "method",
        {
            "Method": "<Bruker:PRESS>",
            "PVM_EncSpectroscopy": "Yes",
            "PVM_VoxArrSize": [[4.5, 3.5, 7.5]],
            "PVM_VoxArrPositionRPS": [[1.0, 2.0, -0.5]],
            "PVM_VoxArrGradOrient": [np.eye(3).tolist()],
            "PVM_NVoxels": 1,
            "PVM_SpecMatrix": [2048],
            "PVM_SpecSWH": [6000.0],
            "PVM_SpecSW": [20.0],
        },
    )
    write_paravision_parameters(
        scan_dir / "acqp",
        {"ACQ_grad_matrix": [np.eye(3).tolist()], "ACQ_patient_pos": "Head_Prone"},
    )
    write_paravision_parameters(scan_dir / "pdata" / "1" / "reco", {})
    handler = BrukerHandler(scan_dir, log_mode="critical")

    volume = excited_volume(handler)

    np.testing.assert_allclose(volume.centre_mm, [-1.0, -2.0, -0.5], atol=ATOL_MM)
    np.testing.assert_allclose(volume.size_mm, [4.5, 3.5, 7.5])


def test_slab_is_unlimited_in_plane():
    method = {
        "PVM_SliceThick": 2.0,
        "PVM_SPackArrGradOrient": [np.eye(3).tolist()],
        "PVM_SPackArrSliceOffset": [5.0],
        "PVM_NSPacks": 1,
        "PVM_SPackArrNSlices": [1],
    }
    volume = excited_volume(
        SimpleNamespace(method=method, patient_position="Head_Prone")
    )

    assert volume.kind == "slab"
    np.testing.assert_allclose(volume.centre_mm, [0.0, 0.0, 5.0], atol=ATOL_MM)
    assert np.isinf(volume.size_mm[:2]).all() and volume.size_mm[2] == 2.0


@pytest.mark.parametrize(
    ("method", "position", "match"),
    [
        ({"PVM_VoxArrSize": [[1, 1, 1]]}, "Head_Supine", "Head_Supine"),
        ({"ExcPulse1Enum": "<bp>"}, "Head_Prone", "neither a voxel"),
        (
            {
                "PVM_SliceThick": 2.0,
                "PVM_SPackArrGradOrient": [np.eye(3).tolist()] * 2,
                "PVM_NSPacks": 2,
            },
            "Head_Prone",
            "PVM_NSPacks",
        ),
        (
            {
                "PVM_VoxArrSize": [[1, 1, 1]],
                "PVM_VoxArrPositionRPS": [[0, 0, 0]],
            },
            "Head_Prone",
            "PVM_VoxArrGradOrient",
        ),
    ],
)
def test_unsupported_geometry_raises_an_actionable_error(method, position, match):
    with pytest.raises(ValueError, match=match):
        excited_volume(SimpleNamespace(method=method, patient_position=position))


def _plane(z_mm=0.0):
    """Image plane z = z_mm with 1 mm pixels; pixel (r, c) at x = c, y = r."""
    return dict(
        origin_mm=np.array([0.0, 0.0, z_mm]),
        row_step_mm=np.array([0.0, 1.0, 0.0]),
        column_step_mm=np.array([1.0, 0.0, 0.0]),
        shape=(20, 30),
    )


def _area(polygon: np.ndarray) -> float:
    rows, columns = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(
        np.dot(rows, np.roll(columns, 1)) - np.dot(columns, np.roll(rows, 1))
    )


def test_axis_aligned_voxel_outline_is_its_rectangle():
    volume = ExcitedVolume(
        np.array([10.0, 5.0, 0.0]), np.eye(3), np.array([4.0, 2.0, 3.0]), "voxel"
    )
    corners = outline_in_plane(volume, **_plane())

    np.testing.assert_allclose(corners[:, 0].min(), 4.0)  # rows = y
    np.testing.assert_allclose(corners[:, 0].max(), 6.0)
    np.testing.assert_allclose(corners[:, 1].min(), 8.0)  # columns = x
    np.testing.assert_allclose(corners[:, 1].max(), 12.0)
    assert _area(corners) == pytest.approx(8.0)


def test_rotated_voxel_outline_keeps_its_area_and_lies_on_the_box():
    axes = _rotation_about_z(45.0).T
    volume = ExcitedVolume(
        np.array([15.0, 10.0, 0.0]), axes, np.array([6.0, 4.0, 3.0]), "voxel"
    )
    corners = outline_in_plane(volume, **_plane(z_mm=1.0))

    assert _area(corners) == pytest.approx(24.0)
    points = np.column_stack([corners[:, 1], corners[:, 0], np.ones(len(corners))])
    offsets = np.abs((points - volume.centre_mm) @ volume.axes.T)
    # Every corner lies on the surface of the box in read or phase.
    assert np.all(np.isclose(offsets[:, :2], volume.size_mm[:2] / 2).any(axis=1))


def test_plane_beyond_the_voxel_has_no_outline():
    volume = ExcitedVolume(
        np.array([10.0, 5.0, 0.0]), np.eye(3), np.array([4.0, 2.0, 3.0]), "voxel"
    )

    assert outline_in_plane(volume, **_plane(z_mm=1.6)).shape == (0, 2)
    assert len(outline_in_plane(volume, **_plane(z_mm=1.4)))


def test_slab_across_the_plane_is_a_band_clipped_to_the_image():
    # Slab normal along x: a band of columns over all rows.
    slab = ExcitedVolume(
        np.array([10.0, 0.0, 0.0]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.array([np.inf, np.inf, 2.0]),
        "slab",
    )
    corners = outline_in_plane(slab, **_plane())

    np.testing.assert_allclose(sorted(set(corners[:, 1])), [9.0, 11.0])
    np.testing.assert_allclose(sorted(set(corners[:, 0])), [-0.5, 19.5])


def test_image_slice_is_a_box_around_the_pixel_centres():
    # Invented coronal slice: rows along z (2 mm), columns along x (1 mm),
    # at y = 5 mm, 3 mm thick.
    volume = image_slice_volume(
        origin_mm=[0.0, 5.0, -4.0],
        row_step_mm=[0.0, 0.0, 2.0],
        column_step_mm=[1.0, 0.0, 0.0],
        shape=(5, 10),
        thickness_mm=3.0,
    )

    assert volume.kind == "slice"
    np.testing.assert_allclose(volume.centre_mm, [4.5, 5.0, 0.0], atol=ATOL_MM)
    np.testing.assert_allclose(volume.size_mm, [10.0, 10.0, 3.0], atol=ATOL_MM)
    np.testing.assert_allclose(np.abs(volume.axes[2]), [0.0, 1.0, 0.0], atol=ATOL_MM)


def test_image_slice_cut_with_a_crossing_plane_is_a_band_of_its_thickness():
    volume = image_slice_volume(
        [0.0, 5.0, -4.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0], (5, 10), 2.0
    )
    # Axial plane z = 0 with 1 mm pixels, rows along y, columns along x.
    corners = outline_in_plane(
        volume, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], (15, 20)
    )

    np.testing.assert_allclose(
        [corners[:, 0].min(), corners[:, 0].max()], [4.0, 6.0], atol=ATOL_MM
    )
    np.testing.assert_allclose(
        [corners[:, 1].min(), corners[:, 1].max()], [-0.5, 9.5], atol=ATOL_MM
    )


@pytest.mark.parametrize(
    ("row_step", "column_step", "shape", "thickness", "match"),
    [
        ([1.0, 0, 0], [0, 1.0, 0], (0, 3), 1.0, "shape"),
        ([1.0, 0, 0], [0, 1.0, 0], (3, 3), -1.0, "thickness"),
        ([1.0, 0, 0], [2.0, 0, 0], (3, 3), 1.0, "parallel|zero"),
    ],
)
def test_image_slice_rejects_invalid_geometry(
    row_step, column_step, shape, thickness, match
):
    with pytest.raises(ValueError, match=match):
        image_slice_volume([0.0, 0, 0], row_step, column_step, shape, thickness)
