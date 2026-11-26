from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from hypermri.utils.io import discover_dicom_series, load_dicom_series


def _write_dicom(
    path: Path,
    *,
    series_uid: str,
    study_uid: str,
    acquisition: int,
    slice_index: int,
    pixels: np.ndarray,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = study_uid
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.Modality = "MR"
    dataset.SeriesDescription = "Synthetic DCE"
    dataset.InstanceNumber = acquisition * 10 + slice_index
    dataset.AcquisitionNumber = acquisition
    dataset.InPlanePhaseEncodingDirection = "ROW"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.PixelSpacing = [1.5, 2.0]
    dataset.SliceThickness = 2.0
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
    dataset.ImagePositionPatient = [10, 20 + 2 * slice_index, 30]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 0.5
    dataset.RescaleIntercept = 1.0
    dataset.PixelData = np.asarray(pixels, dtype=np.int16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_discover_and_load_classic_dynamic_dicom_series(tmp_path):
    dicom_dir = tmp_path / "pdata" / "1" / "custom_dicoms"
    dicom_dir.mkdir(parents=True)
    series_uid = generate_uid()
    study_uid = generate_uid()

    for acquisition in (1, 2):
        for slice_index in range(3):
            pixels = np.arange(6, dtype=np.int16).reshape(2, 3)
            pixels += acquisition * 100 + slice_index * 10
            _write_dicom(
                dicom_dir / f"image_{acquisition}_{slice_index}.dcm",
                series_uid=series_uid,
                study_uid=study_uid,
                acquisition=acquisition,
                slice_index=slice_index,
                pixels=pixels,
            )

    discovered = discover_dicom_series(tmp_path)

    assert len(discovered) == 1
    assert discovered[0].series_instance_uid == series_uid
    assert discovered[0].number_of_instances == 6

    data = load_dicom_series(discovered[0])

    assert data.dims == (
        "read_idx",
        "phase_idx",
        "slice_idx",
        "repetition_idx",
    )
    assert data.shape == (2, 3, 3, 2)
    assert list(data.repetition_idx.values) == [1, 2]
    expected = (np.arange(6).reshape(2, 3) + 100) * 0.5 + 1.0
    np.testing.assert_allclose(data.isel(slice_idx=0, repetition_idx=0), expected)

    origin = np.array(
        [data.coords[name].isel(read_idx=0, phase_idx=0, slice_idx=0) for name in "xyz"]
    )
    read_step = (
        np.array(
            [
                data.coords[name].isel(read_idx=1, phase_idx=0, slice_idx=0)
                for name in "xyz"
            ]
        )
        - origin
    )
    phase_step = (
        np.array(
            [
                data.coords[name].isel(read_idx=0, phase_idx=1, slice_idx=0)
                for name in "xyz"
            ]
        )
        - origin
    )
    np.testing.assert_allclose(origin, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(read_step, [0.0, 0.0, -1.5])
    np.testing.assert_allclose(phase_step, [2.0, 0.0, 0.0])
