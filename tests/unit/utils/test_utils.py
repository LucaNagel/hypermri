from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.analysis.spectroscopy.single_voxel import define_extent_press
from hypermri.utils.coordinates.transformations import define_imagematrix_parameters
from hypermri.utils.image.transforms import (
    define_imageFOV_parameters,
    get_extent,
)


def _geometry(**overrides):
    values = {
        "patient_position": "Head_Prone",
        "read_orientation": "L_R",
        "slice_orientation": "axial",
        "fov_mm": [20.0, 30.0],
        "matrix": [2, 3],
        "spatial_dimension": "2D",
        "n_slices": 3,
        "slice_thickness_mm": 2.0,
        "slice_gap_mm": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_define_imagefov_parameters_uses_standardized_geometry_attributes():
    data_obj = _geometry()

    assert define_imageFOV_parameters(data_obj) == (7.0, 20.0, 30.0)


def test_define_imagematrix_parameters_uses_standardized_geometry_attributes():
    data_obj = _geometry()

    assert define_imagematrix_parameters(data_obj) == (3, 2, 3)


def test_define_extent_press_uses_standardized_geometry_attributes():
    data_obj = _geometry(read_offset_mm=1.0, phase_1_offset_mm=2.0)

    assert define_extent_press(data_obj) == [11.0, -9.0, -13.0, 17.0]


def test_define_imagefov_parameters_uses_3d_fov_slice_extent():
    data_obj = _geometry(
        read_orientation="A_P",
        slice_orientation="sagittal",
        fov_mm=[20.0, 30.0, 40.0],
        spatial_dimension="3D",
    )

    assert define_imageFOV_parameters(data_obj) == (30.0, 40.0, 20.0)


def test_define_imagefov_parameters_handles_head_supine_coronal_lr():
    data_obj = _geometry(
        patient_position="Head_Supine",
        read_orientation="L_R",
        slice_orientation="coronal",
    )

    assert define_imageFOV_parameters(data_obj) == (30.0, 20.0, 7.0)


@pytest.mark.parametrize("patient_position", ["Head_Prone", "Foot_Prone"])
@pytest.mark.parametrize("spatial_dimension", ["2D", "3D"])
def test_coronal_lr_fov_and_matrix_preserve_read_phase_pairing(
    patient_position, spatial_dimension
):
    """Unequal physical lengths follow their own unequal voxel counts."""
    scan = _geometry(
        patient_position=patient_position,
        slice_orientation="coronal",
        read_orientation="L_R",
        fov_mm=np.array([8.0, 9.0] if spatial_dimension == "2D" else [8.0, 9.0, 6.0]),
        matrix=np.array([4, 3] if spatial_dimension == "2D" else [4, 3, 2]),
        spatial_dimension=spatial_dimension,
        n_slices=2,
        slice_thickness_mm=3.0,
        slice_gap_mm=0.0,
    )
    original_fov, original_matrix = scan.fov_mm.copy(), scan.matrix.copy()
    expected_fov = [9.0, 8.0, 6.0]
    for _ in range(2):
        actual_fov = define_imageFOV_parameters(scan)
        np.testing.assert_allclose(actual_fov, expected_fov, rtol=0, atol=1e-12)
        assert not np.allclose(actual_fov, [8.0, 9.0, 6.0], rtol=0, atol=1e-12)
        assert define_imagematrix_parameters(scan) == (3, 4, 2)
    np.testing.assert_array_equal(scan.fov_mm, original_fov)
    np.testing.assert_array_equal(scan.matrix, original_matrix)


@pytest.mark.parametrize("patient_position", ["Head_Prone", "Foot_Prone"])
def test_coronal_lr_legacy_extents_use_approved_fov_lengths(patient_position):
    scan = _geometry(
        patient_position=patient_position,
        slice_orientation="coronal",
        read_orientation="L_R",
        fov_mm=[8.0, 9.0, 6.0],
        spatial_dimension="3D",
        read_offset_mm=1.5,
        phase_1_offset_mm=1.0,
    )
    # Independent 8-mm read, 9-mm phase and 6-mm slice lengths; legacy signs.
    expected = (
        [-2.5, 5.5, -3.0, 3.0],
        [-5.5, 3.5, -3.0, 3.0],
        [-5.5, 3.5, -5.5, 2.5],
    )
    np.testing.assert_allclose(get_extent(scan), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (define_imageFOV_parameters, (7.0, 20.0, 30.0)),
        (define_imagematrix_parameters, (3, 2, 3)),
    ],
)
def test_foot_prone_keeps_the_head_prone_field_of_view_and_matrix(function, expected):
    assert function(_geometry(patient_position="Foot_Prone")) == expected


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (define_imageFOV_parameters, "unknown patient_position: Feet_Prone"),
        (define_imagematrix_parameters, "Unknown patient position: Feet_Prone"),
    ],
)
def test_feet_prone_is_an_unknown_patient_position(function, message):
    """Regression test: ParaVision writes Foot_Prone, not Feet_Prone.

    ``ACQ_patient_pos`` of feet-first ParaVision scans reads ``Foot_Prone``.
    An earlier fix had kept ``"Feet_Prone"`` and rejected ``"Foot_Prone"``.
    """
    with pytest.raises(Exception, match=message):
        function(_geometry(patient_position="Feet_Prone"))


@pytest.mark.parametrize("name", ["reorient_anat", "shift_anat"])
def test_placeholder_anatomy_transforms_are_removed(name):
    """The no-op ``reorient_anat`` and ``shift_anat`` placeholders are gone."""
    import hypermri.utils.image.transforms as transforms

    assert not hasattr(transforms, name)
