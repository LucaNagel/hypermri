"""Field of view shift derived from the slice package offsets.

The fixtures are synthetic attribute containers with the names that the
Bruker handler provides.  Offsets and resolutions are dyadic numbers, so
every division and sign change is exact in float64 and the expected shifts
can be compared exactly.  Head_Prone axial scans with read orientation L_R
have the orientation multipliers +1, +1 and +1 for read, phase and slice
(see ``get_orientation_flip_multipliers``), so the shift has the sign of
the offset.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.utils.coordinates.image_shifts import (
    _fov_shift_from_offsets,
    determine_FOV_shift,
    get_orientation_flip_multipliers,
)


def _scan(**overrides):
    values = {
        "type": "FLASH",
        "read_offset_mm": 2.0,
        "phase_offset_mm": 3.0,
        "slice_offset_mm": 4.0,
        "resolution_mm": np.array([0.5, 0.25, 1.0]),
        "patient_pos": "Head_Prone",
        "slice_orientation": "axial",
        "read_orientation": "L_R",
        "spatial_dimension": "2D",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_2d_shift_moves_phase_only_and_applies_the_multipliers():
    shift_vox, shift_mm = determine_FOV_shift(scan=_scan())

    np.testing.assert_array_equal(shift_vox, [0.0, 12.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0])
    assert shift_vox.dtype == np.float64
    assert shift_mm.dtype == np.float64


def test_3d_shift_also_returns_the_slice_entry():
    shift_vox, shift_mm = determine_FOV_shift(scan=_scan(spatial_dimension="3D"))

    np.testing.assert_array_equal(shift_vox, [0.0, 12.0, 4.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0, 4.0])


def test_3d_shift_without_slice_resolution_keeps_the_slice_voxels_at_zero():
    scan = _scan(spatial_dimension="3D", resolution_mm=np.array([0.5, 0.25]))

    shift_vox, shift_mm = determine_FOV_shift(scan=scan)

    np.testing.assert_array_equal(shift_vox, [0.0, 12.0, 0.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0, 4.0])


@pytest.mark.parametrize("sequence_type", ["CSI", "<Bruker:CSI>", "custom_csi"])
def test_csi_shift_also_moves_the_read_axis(sequence_type):
    shift_vox, shift_mm = determine_FOV_shift(scan=_scan(type=sequence_type))

    np.testing.assert_array_equal(shift_vox, [4.0, 12.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0])


def test_csi_is_recognised_by_the_class_name_without_a_sequence_type():
    class CsiScan(SimpleNamespace):
        pass

    scan = CsiScan(**vars(_scan(type=None)))

    shift_vox, _ = determine_FOV_shift(scan=scan)

    np.testing.assert_array_equal(shift_vox, [4.0, 12.0])


def test_scan_without_orientation_attributes_keeps_the_offset_signs():
    shift_vox, shift_mm = determine_FOV_shift(
        scan=_scan(
            slice_orientation=None, read_orientation=None, spatial_dimension="3D"
        )
    )

    np.testing.assert_array_equal(shift_vox, [0.0, 12.0, 4.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"patient_pos": "Head_Left"},
        {"patient_pos": None},
        {"patient_pos": "Head_Supine", "read_orientation": "H_F"},
        {"patient_pos": "Head_Prone", "slice_orientation": "oblique"},
        {"slice_orientation": None},
    ],
)
def test_orientation_outside_the_flip_table_raises_instead_of_changing_the_signs(
    overrides,
):
    """These scans have no reviewed table entry, so they raise."""
    with pytest.raises(ValueError, match="get_orientation_flip_multipliers"):
        determine_FOV_shift(scan=_scan(**overrides))


def test_missing_slice_offset_is_read_as_zero():
    scan = _scan(spatial_dimension="3D")
    del scan.slice_offset_mm

    shift_vox, shift_mm = determine_FOV_shift(scan=scan)

    np.testing.assert_array_equal(shift_vox, [0.0, 12.0, 0.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0, 0.0])


def test_missing_data_object_gives_zero_shifts():
    shift_vox, shift_mm = determine_FOV_shift(scan=None)

    np.testing.assert_array_equal(shift_vox, [0, 0])
    np.testing.assert_array_equal(shift_mm, [0.0, 0.0])


@pytest.mark.parametrize(
    "missing", ["read_offset_mm", "phase_offset_mm", "resolution_mm"]
)
def test_missing_offset_or_resolution_gives_zero_shifts_and_warns(missing, caplog):
    scan = _scan()
    delattr(scan, missing)

    with caplog.at_level(logging.WARNING, logger="hypermri"):
        shift_vox, shift_mm = determine_FOV_shift(scan=scan)

    np.testing.assert_array_equal(shift_vox, [0, 0])
    np.testing.assert_array_equal(shift_mm, [0.0, 0.0])
    assert missing in caplog.text


def test_missing_sequence_type_raises_attribute_error():
    scan = _scan()
    del scan.type

    with pytest.raises(AttributeError, match="type"):
        determine_FOV_shift(scan=scan)


def test_slice_packages_with_equal_multipliers_use_the_common_multipliers():
    scan = _scan(
        slice_orientation=np.array(["axial", "sagittal", "coronal"]),
        read_orientation=np.array(["L_R", "A_P", "H_F"]),
    )

    shift_vox, shift_mm = determine_FOV_shift(scan=scan)

    np.testing.assert_array_equal(shift_vox, [0.0, 12.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0])


@pytest.mark.parametrize(
    ("shift_read", "expected_vox"),
    [(False, [0.0, 12.0, -4.0]), (True, [-4.0, 12.0, -4.0])],
)
def test_offsets_are_converted_to_voxels_and_multiplied(shift_read, expected_vox):
    offsets_mm = [2.0, -3.0, 4.0]
    resolution_mm = np.array([0.5, 0.25, 1.0])

    shift_vox, shift_mm = _fov_shift_from_offsets(
        offsets_mm, resolution_mm, np.array([-1, -1, -1]), shift_read=shift_read
    )

    np.testing.assert_array_equal(shift_vox, expected_vox)
    np.testing.assert_array_equal(shift_mm, [-2.0, 3.0, -4.0])
    assert shift_vox.dtype == shift_mm.dtype == np.float64
    assert offsets_mm == [2.0, -3.0, 4.0]
    np.testing.assert_array_equal(resolution_mm, [0.5, 0.25, 1.0])


def test_offsets_without_slice_resolution_keep_the_slice_voxels_at_zero():
    shift_vox, shift_mm = _fov_shift_from_offsets(
        (2.0, 3.0, 4.0), (0.5, 0.25), (1, 1, 1), shift_read=True
    )

    np.testing.assert_array_equal(shift_vox, [4.0, 12.0, 0.0])
    np.testing.assert_array_equal(shift_mm, [2.0, 3.0, 4.0])


def _two_package_scan(**overrides):
    # Package 0 is Head_Prone axial L_R and package 1 Head_Prone coronal L_R
    # with 3D encoding; both have the multipliers +1, +1 and +1.
    values = {
        "spatial_dimension": "3D",
        "slice_orientation": np.array(["axial", "coronal"]),
        "read_orientation": np.array(["L_R", "L_R"]),
        "slices_per_pack": (1, 1),
        "slice_pack_offsets_mm": np.array([[2.0, 3.0, 4.0], [1.0, -1.0, 0.5]]),
    }
    values.update(overrides)
    return _scan(**values)


@pytest.mark.parametrize(
    ("sequence_type", "slice_pack", "expected_vox", "expected_mm"),
    [
        ("FLASH", 0, [0.0, 12.0, 4.0], [2.0, 3.0, 4.0]),
        ("FLASH", 1, [0.0, -4.0, 0.5], [1.0, -1.0, 0.5]),
        ("CSI", 1, [2.0, -4.0, 0.5], [1.0, -1.0, 0.5]),
    ],
)
def test_slice_pack_shift_uses_the_offsets_and_orientation_of_that_package(
    sequence_type, slice_pack, expected_vox, expected_mm
):
    scan = _two_package_scan(type=sequence_type)

    shift_vox, shift_mm = determine_FOV_shift(scan=scan, slice_pack=slice_pack)

    np.testing.assert_array_equal(shift_vox, expected_vox)
    np.testing.assert_array_equal(shift_mm, expected_mm)


def test_slice_pack_shift_applies_a_single_orientation_to_every_package():
    scan = _two_package_scan(
        spatial_dimension="2D", slice_orientation="axial", read_orientation="L_R"
    )

    shift_vox, shift_mm = determine_FOV_shift(scan=scan, slice_pack=1)

    np.testing.assert_array_equal(shift_vox, [0.0, -4.0])
    np.testing.assert_array_equal(shift_mm, [1.0, -1.0])


def test_slice_pack_shift_without_package_offsets_gives_zero_shifts(caplog):
    scan = _two_package_scan()
    del scan.slice_pack_offsets_mm

    with caplog.at_level(logging.WARNING, logger="hypermri"):
        shift_vox, shift_mm = determine_FOV_shift(scan=scan, slice_pack=0)

    np.testing.assert_array_equal(shift_vox, [0, 0])
    np.testing.assert_array_equal(shift_mm, [0.0, 0.0])
    assert "slice_pack_offsets_mm" in caplog.text


def test_slice_pack_outside_the_packages_raises_index_error():
    with pytest.raises(IndexError):
        determine_FOV_shift(scan=_two_package_scan(), slice_pack=2)


@pytest.mark.parametrize(
    (
        "patient_pos",
        "slice_orientation",
        "read_orientation",
        "spatial_dimension",
        "expected",
    ),
    [
        ("Head_Prone", "axial", "L_R", "2D", [1, 1, 1]),
        ("Head_Prone", "axial", "A_P", "2D", [1, 1, 1]),
        ("Head_Prone", "sagittal", "A_P", "2D", [1, 1, 1]),
        ("Head_Prone", "sagittal", "H_F", "2D", [1, 1, 1]),
        ("Head_Prone", "coronal", "H_F", "2D", [1, 1, 1]),
        ("Head_Prone", "coronal", "L_R", "3D", [1, 1, 1]),
        ("Head_Supine", "axial", "L_R", "2D", [1, 1, 1]),
        ("Head_Supine", "axial", "A_P", "2D", [1, 1, 1]),
        ("Head_Supine", "sagittal", "H_F", "2D", [1, 1, 1]),
        ("Head_Supine", "sagittal", "A_P", "2D", [1, 1, 1]),
        ("Head_Supine", "coronal", "H_F", "2D", [1, 1, 1]),
        ("Head_Supine", "coronal", "L_R", "3D", [1, 1, 1]),
        ("Foot_Prone", "axial", "L_R", "2D", [1, 1, 1]),
        ("Foot_Supine", "coronal", "H_F", "3D", [1, 1, 1]),
    ],
)
def test_flip_table_entries_keep_their_multipliers(
    patient_pos, slice_orientation, read_orientation, spatial_dimension, expected
):
    """Characterization of the table entries before and after missing entries became errors.

    The entries were compared with the function previously for every
    combination of the four patient positions, three slice orientations,
    three read orientations and both encodings (2D and 3D).
    """
    scan = SimpleNamespace(
        patient_pos=patient_pos,
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
        spatial_dimension=spatial_dimension,
    )

    np.testing.assert_array_equal(get_orientation_flip_multipliers(scan), expected)


@pytest.mark.parametrize(
    ("patient_pos", "expected"),
    [
        ("Head_Prone", [1, 1, 1]),
        ("Head_Supine", [1, 1, 1]),
        (None, [1, 1, 1]),
        ("Foot_Prone", [1, 1, 1]),
    ],
)
def test_scan_without_orientation_attributes_has_no_table_lookup(patient_pos, expected):
    """Non-imaging raw data has a ``slice_idx`` axis but no orientation."""
    scan = SimpleNamespace(patient_pos=patient_pos)

    np.testing.assert_array_equal(get_orientation_flip_multipliers(scan), expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"patient_pos": "Head_Left"}, "unknown patient_pos 'Head_Left'"),
        ({"patient_pos": "Feet_Prone"}, "unknown patient_pos 'Feet_Prone'"),
        ({"patient_pos": None}, "patient_pos 'None'"),
        ({"read_orientation": "H_F"}, "no flip table entry"),
        ({"slice_orientation": "oblique"}, "slice_orientation 'oblique'"),
        ({"read_orientation": None}, "read_orientation 'None'"),
    ],
)
def test_flip_table_rejects_orientations_without_an_entry(overrides, message):
    """Each of these gave +1 for every axis before; Head_Supine axial H_F warned."""
    values = {
        "patient_pos": "Head_Supine",
        "slice_orientation": "axial",
        "read_orientation": "L_R",
        "spatial_dimension": "2D",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        get_orientation_flip_multipliers(SimpleNamespace(**values))
