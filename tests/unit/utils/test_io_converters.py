import inspect
import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import xarray as xr
from pydicom.pixels import apply_rescale

from hypermri.utils.io import dicom_export, export_to_dicom, load_dicom_series


def test_export_to_dicom_uses_acquisition_idx_as_instances(tmp_path):
    n_acq = 200
    n_read = 10
    n_phase = 8

    data = np.arange(n_acq * n_read * n_phase, dtype=np.int16).reshape(
        n_acq, n_read, n_phase, 1, 1, 1
    )
    da = xr.DataArray(
        data,
        dims=(
            "acquisition_idx",
            "read_idx",
            "phase_idx",
            "slice_idx",
            "repetition_idx",
            "channel_idx",
        ),
    )

    with pytest.warns(UserWarning, match="identity geometry"):
        success = export_to_dicom(
            data=da,
            output_dir=tmp_path,
            filename_prefix="CSI",
            series_description="CSI sampling",
            allow_identity_geometry=True,
        )

    assert success

    exported = sorted(tmp_path.glob("CSI*.dcm"))
    assert len(exported) == n_acq

    first = pydicom.dcmread(exported[0])
    second = pydicom.dcmread(exported[1])

    assert first.Rows == n_phase
    assert first.Columns == n_read

    expected0 = data[0, :, :, 0, 0, 0].T
    expected1 = data[1, :, :, 0, 0, 0].T

    np.testing.assert_array_equal(first.pixel_array, expected0)
    np.testing.assert_array_equal(second.pixel_array, expected1)


def _xyz_image_data() -> xr.DataArray:
    read_size, phase_size, slice_size = 2, 3, 2
    values = np.arange(read_size * phase_size * slice_size, dtype=np.float32).reshape(
        read_size, phase_size, slice_size
    )
    read = np.arange(read_size)[:, None, None, None]
    phase = np.arange(phase_size)[None, :, None, None]
    slices = np.asarray([[10.0, 20.0, 30.0], [10.0, 22.0, 30.0]])[None, None, :, :]
    xyz = (
        slices
        + read * np.asarray([0.0, 0.0, -1.5])
        + phase * np.asarray([2.0, 0.0, 0.0])
    )
    return xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            name: (
                ("read_idx", "phase_idx", "slice_idx"),
                xyz[..., component],
                {"unit": "mm", "coordinate_system": "dicom_lps"},
            )
            for component, name in enumerate("xyz")
        },
    )


def test_export_to_dicom_uses_xarray_xyz_geometry_without_references(tmp_path):
    data = _xyz_image_data()

    success = export_to_dicom(
        data,
        tmp_path,
        filename_prefix="map",
        series_description="Derived map",
        unit="s",
    )

    assert success
    exported = sorted(tmp_path.glob("map*.dcm"))
    assert len(exported) == 2
    first = pydicom.dcmread(exported[0])
    second = pydicom.dcmread(exported[1])
    assert first.Rows == 2
    assert first.Columns == 3
    assert first.PixelSpacing == [1.5, 2.0]
    assert first.ImageOrientationPatient == [1, 0, 0, 0, 0, -1]
    assert first.ImagePositionPatient == [10, 20, 30]
    assert second.ImagePositionPatient == [10, 22, 30]
    assert first.InPlanePhaseEncodingDirection == "ROW"
    assert first.ImageType == ["DERIVED", "SECONDARY"]
    assert first.SeriesDescription == "Derived map"
    assert "ImageComments" not in first
    assert first.RescaleSlope == second.RescaleSlope
    assert first.RescaleIntercept == second.RescaleIntercept
    np.testing.assert_allclose(
        apply_rescale(first.pixel_array, first),
        data.isel(slice_idx=0),
        atol=float(first.RescaleSlope),
    )

    reloaded = load_dicom_series(tmp_path)
    assert reloaded.dims == ("read_idx", "phase_idx", "slice_idx")
    np.testing.assert_allclose(
        reloaded.values, data.values, atol=float(first.RescaleSlope)
    )
    for coord_name in "xyz":
        np.testing.assert_allclose(reloaded.coords[coord_name], data.coords[coord_name])


def test_export_to_dicom_accepts_separate_geometry_data(tmp_path):
    geometry_data = _xyz_image_data()
    values_only = xr.DataArray(
        geometry_data.values,
        dims=geometry_data.dims,
    )

    success = export_to_dicom(
        values_only,
        tmp_path,
        geometry=geometry_data,
    )

    assert success
    first = pydicom.dcmread(sorted(tmp_path.glob("MRIm*.dcm"))[0])
    assert first.ImagePositionPatient == [10, 20, 30]
    assert first.PixelSpacing == [1.5, 2.0]


def _mixed_package_geometry():
    """Invented axial/sagittal two-slice packages, with 2 mm voxel spacing."""
    from hypermri.utils.coordinates.grids import assign_xyz_coordinates

    handler = SimpleNamespace(
        matrix=np.array([3, 3, 4]),
        slices_per_pack=(2, 2),
        reco_transposition=np.zeros(4, dtype=int),
        visucore_position=np.array(
            [[0, 0, 0], [0, 0, 2], [10, 0, 0], [12, 0, 0]], dtype=float
        ),
        visucore_orientation=np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 2 + [[0, 1, 0, 0, 0, -1, -1, 0, 0]] * 2,
            dtype=float,
        ),
        visucore_size=np.array([3, 3]),
        visucore_extent=np.array([6.0, 6.0]),
    )
    values = xr.DataArray(
        np.arange(36, dtype=np.int16).reshape(3, 3, 4),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    return handler, assign_xyz_coordinates(values, scan=handler)


@pytest.mark.parametrize("source_kind", ["data", "array", "handler", "scan"])
@pytest.mark.parametrize("allow_identity_geometry", [False, True])
def test_mixed_packages_report_export_remedy_before_writing(
    tmp_path, source_kind, allow_identity_geometry
):
    """Every XYZ route rejects the same unsupported mixed stack."""
    handler, positioned = _mixed_package_geometry()
    original = positioned.copy(deep=True)
    values = xr.DataArray(positioned.values, dims=positioned.dims)
    geometry = {
        "data": handler,
        "array": positioned,
        "handler": handler,
        "scan": SimpleNamespace(handler=handler, primary_data=positioned),
    }[source_kind]
    output = tmp_path / "out"

    with pytest.raises(
        ValueError, match="export each slice package separately"
    ) as error:
        dicom_export.export_to_dicom(
            positioned if source_kind == "data" else values,
            output,
            geometry=geometry,
            allow_identity_geometry=allow_identity_geometry,
        )

    if source_kind != "array":
        assert "package 1 (2 slices), package 2 (2 slices)" in str(error.value)
    assert not output.exists()
    xr.testing.assert_identical(positioned, original)


def test_public_export_logs_slice_packages_and_returns_false(tmp_path, caplog):
    handler, positioned = _mixed_package_geometry()

    with caplog.at_level(logging.ERROR):
        success = export_to_dicom(positioned, tmp_path / "out", geometry=handler)

    assert success is False
    assert "package 1 (2 slices), package 2 (2 slices)" in caplog.text
    assert "export each slice package separately" in caplog.text
    assert not (tmp_path / "out").exists()


def test_selected_slice_packages_export_with_their_own_xyz_coordinates(tmp_path):
    """The suggested remedy preserves each package's orientation and values."""
    _, positioned = _mixed_package_geometry()
    for package_index in range(2):
        package = positioned.isel(
            slice_idx=slice(2 * package_index, 2 * package_index + 2)
        )
        output = tmp_path / str(package_index)
        assert export_to_dicom(package, output)
        files = sorted(output.glob("*.dcm"))
        assert len(files) == 2
        for slice_index, path in enumerate(files):
            dataset = pydicom.dcmread(path)
            plane = package.isel(slice_idx=slice_index)
            np.testing.assert_array_equal(dataset.pixel_array, plane.values)
            origin = np.array([plane.coords[name][0, 0].item() for name in "xyz"])
            read_step = (
                np.array([plane.coords[name][1, 0].item() for name in "xyz"]) - origin
            )
            phase_step = (
                np.array([plane.coords[name][0, 1].item() for name in "xyz"]) - origin
            )
            np.testing.assert_allclose(dataset.ImagePositionPatient, origin, atol=1e-6)
            np.testing.assert_allclose(
                dataset.ImageOrientationPatient,
                np.concatenate([phase_step / 2.0, read_step / 2.0]),
                atol=1e-6,
            )


def test_export_to_dicom_accepts_reference_directory_path(tmp_path):
    geometry_data = _xyz_image_data()
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "derived"
    assert export_to_dicom(geometry_data, reference_dir)

    values_only = xr.DataArray(
        geometry_data.values,
        dims=geometry_data.dims,
    )
    success = export_to_dicom(
        values_only,
        output_dir,
        reference_dicoms=reference_dir,
    )

    assert success
    first = pydicom.dcmread(sorted(output_dir.glob("MRIm*.dcm"))[0])
    assert first.ImagePositionPatient == [10, 20, 30]
    assert first.ImageOrientationPatient == [1, 0, 0, 0, 0, -1]
    assert first.ImageType == ["DERIVED", "SECONDARY"]


# Data without geometry.  The arrays below are synthetic (``np.arange``
# values on a 10 x 8 single-slice grid); no scanner data are involved.


def _values_without_geometry() -> xr.DataArray:
    return xr.DataArray(
        np.arange(10 * 8, dtype=np.int16).reshape(10, 8, 1),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )


def test_fallback_geometry_is_a_centred_1mm_identity_grid():
    geometry = dicom_export._fallback_geometry(_values_without_geometry())

    assert geometry.source == "fallback_identity"
    assert (geometry.row_dim, geometry.column_dim) == ("phase_idx", "read_idx")
    np.testing.assert_array_equal(
        geometry.image_orientation_patient, [1, 0, 0, 0, 1, 0]
    )
    np.testing.assert_allclose(
        geometry.image_positions_patient, [[-4.5, -3.5, 0.0]], atol=1e-3
    )
    assert geometry.pixel_spacing == (1.0, 1.0)
    assert geometry.slice_thickness == 1.0


def test_export_to_dicom_rejects_data_without_geometry(tmp_path):
    with pytest.raises(ValueError, match="allow_identity_geometry=True"):
        dicom_export.export_to_dicom(_values_without_geometry(), tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_export_to_dicom_rejects_numpy_data_without_geometry(tmp_path):
    with pytest.raises(ValueError, match="no spatial geometry"):
        dicom_export.export_to_dicom(np.ones((4, 3)), tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_export_to_dicom_rejects_geometry_source_without_geometry(tmp_path):
    class ScanWithoutVisuCore:
        pass

    with pytest.raises(ValueError, match="ScanWithoutVisuCore"):
        dicom_export.export_to_dicom(
            _values_without_geometry(),
            tmp_path / "out",
            geometry=ScanWithoutVisuCore(),
        )

    assert not (tmp_path / "out").exists()


def test_export_to_dicom_rejects_references_without_image_position(tmp_path):
    reference = pydicom.Dataset()
    reference.InstanceNumber = 1
    reference.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    reference.PixelSpacing = [1.0, 1.0]

    with pytest.raises(ValueError, match="ImagePositionPatient"):
        dicom_export.export_to_dicom(
            _values_without_geometry(),
            tmp_path / "out",
            reference_dicoms=[reference],
        )

    assert not (tmp_path / "out").exists()


def test_public_export_to_dicom_returns_false_without_geometry(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        success = export_to_dicom(_values_without_geometry(), tmp_path / "out")

    assert success is False
    assert not (tmp_path / "out").exists()
    assert "allow_identity_geometry=True" in caplog.text


def test_public_export_to_dicom_offers_identity_geometry_opt_in():
    parameter = inspect.signature(export_to_dicom).parameters["allow_identity_geometry"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def test_identity_geometry_opt_in_labels_the_series(tmp_path):
    with pytest.warns(UserWarning, match="identity geometry"):
        success = export_to_dicom(
            _values_without_geometry(),
            tmp_path,
            series_description="Derived map",
            allow_identity_geometry=True,
        )

    assert success
    exported = sorted(tmp_path.glob("MRIm*.dcm"))
    assert len(exported) == 1
    first = pydicom.dcmread(exported[0])
    assert first.SeriesDescription == "Derived map [identity geometry]"
    assert "not the patient position" in first.ImageComments
    assert first.DerivationDescription.endswith("geometry=fallback_identity")
    assert first.ImagePositionPatient == [-4.5, -3.5, 0.0]
    assert first.ImageOrientationPatient == [1, 0, 0, 0, 1, 0]
    assert first.PixelSpacing == [1.0, 1.0]
    assert (first.Rows, first.Columns) == (8, 10)


def test_identity_geometry_label_keeps_the_series_description_length(tmp_path):
    with pytest.warns(UserWarning, match="identity geometry"):
        assert export_to_dicom(
            _values_without_geometry(),
            tmp_path,
            series_description="d" * 64,
            allow_identity_geometry=True,
        )

    first = pydicom.dcmread(sorted(tmp_path.glob("MRIm*.dcm"))[0])
    assert len(first.SeriesDescription) == 64
    assert first.SeriesDescription.endswith(" [identity geometry]")


def test_identity_geometry_opt_in_prefers_available_geometry(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert export_to_dicom(
            _xyz_image_data(),
            tmp_path,
            series_description="Derived map",
            allow_identity_geometry=True,
        )

    first = pydicom.dcmread(sorted(tmp_path.glob("MRIm*.dcm"))[0])
    assert first.SeriesDescription == "Derived map"
    assert "ImageComments" not in first
    assert first.ImagePositionPatient == [10, 20, 30]
