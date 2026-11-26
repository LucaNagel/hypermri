"""Tests for the ParaVision geometry reference fixture and its extraction.

The committed fixture ``tests/reference/paravision_water_phantom_geometry.json``
was produced by ``scripts/extract_geometry_fixture.py`` from local,
Git-ignored ParaVision DICOM exports of a water phantom.  The
fixture tests check that it holds only whitelisted, non-identifying geometry
and that its DICOM reference is internally consistent.  The script tests use
small synthetic DICOM files written in ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_RELATIVE_PATH = "tests/reference/paravision_water_phantom_geometry.json"
FIXTURE_PATH = PROJECT_ROOT / FIXTURE_RELATIVE_PATH

# Every string value in the fixture must be one of these geometry enums.  A new
# value needs a privacy review before it is added here.
ALLOWED_STRING_VALUES = frozenset(
    {
        "<2D>",
        "A_P",
        "H_F",
        "HFP",
        "HFS",
        "Head_Prone",
        "Head_Supine",
        "L_R",
        "QUADRUPED",
        "axial",
        "coronal",
        "sagittal",
    }
)
SERIES_KEY_RE = re.compile(r"^(head_prone|head_supine)/[1-9][0-9]*$")
DATE_LIKE_RE = re.compile(r"\d{8}|\d{4}[-./]\d{2}[-./]\d{2}")

# VisuCore and DICOM values are decimal strings written by ParaVision for the
# same geometry; they agree exactly in the source exports.
VISU_DICOM_ATOL_MM = 1e-6
# Direction cosines are exported as exact 0 and +-1 for these acquisitions.
DIRECTION_ATOL = 1e-6
# The geometry position tolerance; positions carry six decimals in mm.
POSITION_ATOL_MM = 1e-3


def _load_script(name: str) -> ModuleType:
    script_path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"hypermri_{name}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extract = _load_script("extract_geometry_fixture")
audit = _load_script("audit_public_snapshot")


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _string_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)
    elif isinstance(value, str):
        yield value


def _single_pack_series(fixture: dict):
    for name, series in fixture["series"].items():
        if np.size(series["paravision"]["method"]["PVM_SPackArrNSlices"]) == 1:
            yield name, series


# --------------------------------------------------------------------------
# Committed fixture: whitelist and privacy
# --------------------------------------------------------------------------


def test_fixture_keys_match_extraction_whitelist(fixture):
    assert set(fixture) == {"provenance", "series"}
    assert fixture["provenance"] == extract.PROVENANCE

    dicom_keys = set(extract.PER_SERIES_DICOM_FIELDS.values()) | set(
        extract.PER_FRAME_DICOM_FIELDS.values()
    )
    for name, series in fixture["series"].items():
        assert set(series) == {"dicom", "paravision"}, name
        assert set(series["dicom"]) == dicom_keys, name
        assert {
            section: tuple(values) for section, values in series["paravision"].items()
        } == extract.PARAVISION_FIELDS, name


def test_fixture_contains_only_non_identifying_values(fixture):
    assert all(SERIES_KEY_RE.fullmatch(name) for name in fixture["series"])
    assert set(_string_values(fixture["series"])) <= ALLOWED_STRING_VALUES
    assert DATE_LIKE_RE.search(FIXTURE_PATH.read_text(encoding="utf-8")) is None
    assert audit.audit_file(PROJECT_ROOT, FIXTURE_RELATIVE_PATH) == []


def test_fixture_file_is_the_unedited_script_output(fixture):
    text = FIXTURE_PATH.read_text(encoding="utf-8")

    assert text == extract.format_fixture(fixture) + "\n"


# --------------------------------------------------------------------------
# Committed fixture: DICOM reference consistency
# --------------------------------------------------------------------------


def test_fixture_covers_both_patient_positions(fixture):
    positions = {
        "head_prone": ("HFP", "Head_Prone"),
        "head_supine": ("HFS", "Head_Supine"),
    }
    counts = {label: 0 for label in positions}
    for name, series in fixture["series"].items():
        label = name.split("/")[0]
        counts[label] += 1
        dicom_position, paravision_position = positions[label]
        assert series["dicom"]["patient_position"] == dicom_position, name
        assert series["dicom"]["anatomical_orientation_type"] == "QUADRUPED", name
        assert series["paravision"]["acqp"]["ACQ_patient_pos"] == paravision_position

    assert counts == {"head_prone": 26, "head_supine": 25}


def test_fixture_frame_counts_agree_between_dicom_and_paravision(fixture):
    for name, series in fixture["series"].items():
        dicom, paravision = series["dicom"], series["paravision"]
        n_frames = int(np.sum(paravision["method"]["PVM_SPackArrNSlices"]))

        assert dicom["instance_numbers"] == list(range(1, n_frames + 1)), name
        assert np.shape(dicom["image_position_patient_mm"]) == (n_frames, 3), name
        assert np.shape(dicom["image_orientation_patient"]) == (n_frames, 6), name
        assert np.shape(paravision["reco"]["RECO_transposition"]) == (n_frames,)
        assert np.shape(paravision["acqp"]["ACQ_grad_matrix"]) == (n_frames, 3, 3)
        assert np.shape(paravision["visu_pars"]["VisuCorePosition"]) == (n_frames, 3)


def test_fixture_image_orientations_are_orthonormal(fixture):
    for name, series in fixture["series"].items():
        orientation = np.asarray(series["dicom"]["image_orientation_patient"])
        columns, rows = orientation[:, :3], orientation[:, 3:]

        np.testing.assert_allclose(
            np.linalg.norm(columns, axis=1), 1.0, atol=DIRECTION_ATOL, err_msg=name
        )
        np.testing.assert_allclose(
            np.linalg.norm(rows, axis=1), 1.0, atol=DIRECTION_ATOL, err_msg=name
        )
        np.testing.assert_allclose(
            np.einsum("ij,ij->i", columns, rows), 0.0, atol=DIRECTION_ATOL, err_msg=name
        )


def test_fixture_visucore_geometry_equals_dicom_export(fixture):
    for name, series in fixture["series"].items():
        dicom, visu = series["dicom"], series["paravision"]["visu_pars"]
        orientation = np.asarray(visu["VisuCoreOrientation"]).reshape(-1, 3, 3)
        column_spacing, row_spacing = np.divide(
            visu["VisuCoreExtent"], visu["VisuCoreSize"]
        )

        np.testing.assert_allclose(
            visu["VisuCorePosition"],
            dicom["image_position_patient_mm"],
            rtol=0,
            atol=VISU_DICOM_ATOL_MM,
            err_msg=name,
        )
        np.testing.assert_allclose(
            orientation[:, :2].reshape(-1, 6),
            dicom["image_orientation_patient"],
            rtol=0,
            atol=DIRECTION_ATOL,
            err_msg=name,
        )
        assert visu["VisuCoreSize"] == [dicom["columns"], dicom["rows"]], name
        np.testing.assert_allclose(
            [row_spacing, column_spacing],
            dicom["pixel_spacing_mm"],
            rtol=0,
            atol=VISU_DICOM_ATOL_MM,
            err_msg=name,
        )
        assert visu["VisuCoreFrameThickness"] == dicom["slice_thickness_mm"], name
        assert visu["VisuCoreDiskSliceOrder"] is None, name


def test_fixture_transposition_matches_dicom_matrix_shape(fixture):
    """Independent shape evidence for the ParaVision read/phase index rule.

    For a non-square ``PVM_Matrix = (read, phase)``, the DICOM frame shape
    shows which in-plane axis carries the read direction:
    ``RECO_transposition == 0`` stores read along DICOM columns and
    ``RECO_transposition == 1`` along DICOM rows.
    """
    checked = 0
    for name, series in _single_pack_series(fixture):
        dicom, paravision = series["dicom"], series["paravision"]
        read_size, phase_size = paravision["method"]["PVM_Matrix"]
        transposition = set(paravision["reco"]["RECO_transposition"])
        if read_size == phase_size:
            continue

        assert len(transposition) == 1, name
        if transposition == {0}:
            assert (dicom["columns"], dicom["rows"]) == (read_size, phase_size), name
        else:
            assert transposition == {1}, name
            assert (dicom["rows"], dicom["columns"]) == (read_size, phase_size), name
        checked += 1

    assert checked == 48


def test_fixture_slice_positions_follow_slice_distance(fixture):
    """Instance order is a monotonic walk with the ParaVision slice distance.

    The slice distance is ``PVM_SliceThick + PVM_SPackArrSliceGap``.  The walk
    runs along the DICOM normal ``column x row`` for axial and sagittal series
    and against it for coronal series, so only its magnitude and a constant
    sign are asserted.
    """
    for name, series in _single_pack_series(fixture):
        dicom, method = series["dicom"], series["paravision"]["method"]
        position = np.asarray(dicom["image_position_patient_mm"])
        orientation = np.asarray(dicom["image_orientation_patient"])
        normal = np.cross(orientation[0, :3], orientation[0, 3:])
        steps = np.diff(position, axis=0) @ normal
        distance = method["PVM_SliceThick"] + method["PVM_SPackArrSliceGap"]

        np.testing.assert_allclose(
            np.abs(steps), distance, rtol=0, atol=POSITION_ATOL_MM, err_msg=name
        )
        assert len(set(np.sign(steps))) == 1, name
        in_plane_shift = np.diff(position, axis=0) - np.outer(steps, normal)
        np.testing.assert_allclose(
            in_plane_shift, 0.0, rtol=0, atol=POSITION_ATOL_MM, err_msg=name
        )


# --------------------------------------------------------------------------
# Extraction script with synthetic DICOM files
# --------------------------------------------------------------------------


def _write_synthetic_frame(
    path: Path, instance_number: int, omit: tuple[str, ...] = (), **overrides
) -> None:
    """Write a synthetic MR frame with identifying elements and pixel data.

    All values are invented for this test.  The identifying elements must not
    reach the extracted geometry.
    """
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPClassUID = MRImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.{instance_number}"
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = f"1.2.3.4.{instance_number}"
    dataset.PatientName = "Synthetic^Subject"
    dataset.PatientID = "SYNTHETIC-ID"
    dataset.StudyDate = "20000101"
    dataset.InstitutionName = "Synthetic Institute"
    dataset.StationName = "SYNTH-STATION"
    dataset.InstanceNumber = instance_number
    dataset.ImagePositionPatient = [-24.0, -16.0, -10.0 + 5.0 * (instance_number - 1)]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.PixelSpacing = [0.25, 0.25]
    dataset.SliceThickness = 1.0
    dataset.PatientPosition = "HFP"
    dataset.AnatomicalOrientationType = "QUADRUPED"
    dataset.BitsAllocated = 16
    dataset.PixelData = np.arange(6, dtype=np.uint16).tobytes()
    for keyword, value in overrides.items():
        setattr(dataset, keyword, value)
    for keyword in omit:
        delattr(dataset, keyword)
    dataset.save_as(path, enforce_file_format=True)


def test_read_dicom_frame_geometry_reads_only_whitelisted_elements(tmp_path):
    path = tmp_path / "frame.dcm"
    _write_synthetic_frame(path, instance_number=3)

    geometry = extract.read_dicom_frame_geometry(path)

    assert geometry == {
        "InstanceNumber": 3,
        "ImagePositionPatient": [-24.0, -16.0, 0.0],
        "ImageOrientationPatient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "Rows": 2,
        "Columns": 3,
        "PixelSpacing": [0.25, 0.25],
        "SliceThickness": 1.0,
        "PatientPosition": "HFP",
        "AnatomicalOrientationType": "QUADRUPED",
    }
    text = json.dumps(geometry)
    for identifying in ("Synthetic", "SYNTHETIC", "20000101", "1.2.3.4"):
        assert identifying not in text


def test_read_dicom_frame_geometry_rejects_missing_element(tmp_path):
    path = tmp_path / "frame.dcm"
    _write_synthetic_frame(path, instance_number=1, omit=("PixelSpacing",))

    with pytest.raises(ValueError, match="PixelSpacing is missing"):
        extract.read_dicom_frame_geometry(path)


def test_extract_dicom_series_orders_frames_by_instance_number(tmp_path):
    _write_synthetic_frame(tmp_path / "a.dcm", instance_number=2)
    _write_synthetic_frame(tmp_path / "b.dcm", instance_number=1)

    series = extract.extract_dicom_series(tmp_path)

    assert series["instance_numbers"] == [1, 2]
    assert series["image_position_patient_mm"] == [
        [-24.0, -16.0, -10.0],
        [-24.0, -16.0, -5.0],
    ]
    assert series["pixel_spacing_mm"] == [0.25, 0.25]
    assert set(series) == set(extract.PER_SERIES_DICOM_FIELDS.values()) | set(
        extract.PER_FRAME_DICOM_FIELDS.values()
    )


def test_extract_dicom_series_rejects_duplicate_instance_numbers(tmp_path):
    _write_synthetic_frame(tmp_path / "a.dcm", instance_number=1)
    _write_synthetic_frame(tmp_path / "b.dcm", instance_number=1)

    with pytest.raises(ValueError, match="InstanceNumber values are not unique"):
        extract.extract_dicom_series(tmp_path)


def test_extract_dicom_series_rejects_inconsistent_series_value(tmp_path):
    _write_synthetic_frame(tmp_path / "a.dcm", instance_number=1)
    _write_synthetic_frame(tmp_path / "b.dcm", instance_number=2, Rows=4)

    with pytest.raises(ValueError, match="Rows differs between frames"):
        extract.extract_dicom_series(tmp_path)


def test_extract_study_keys_series_by_label_and_skips_scans_without_export(
    tmp_path, monkeypatch
):
    study = tmp_path / "2000.01.01 - initials - study"
    (study / "7" / "pdata" / "1" / "dicom").mkdir(parents=True)
    _write_synthetic_frame(study / "7" / "pdata" / "1" / "dicom" / "f.dcm", 1)
    (study / "3" / "pdata" / "1").mkdir(parents=True)
    (study / "AdjResult").mkdir()
    monkeypatch.setattr(
        extract,
        "extract_paravision_parameters",
        lambda scan_dir: {"scan": scan_dir.name},
    )

    series = extract.extract_study("head_prone", study)

    assert list(series) == ["head_prone/7"]
    assert series["head_prone/7"]["paravision"] == {"scan": "7"}
    assert "initials" not in json.dumps(series)


@pytest.mark.parametrize(
    "label",
    ["2000.01.01 - initials - study", "Head_Prone", "head prone", "", "1study"],
)
def test_parse_study_argument_rejects_labels_that_are_not_identifiers(tmp_path, label):
    with pytest.raises(ValueError, match="must match"):
        extract.parse_study_argument(f"{label}={tmp_path}")


def test_parse_study_argument_requires_existing_directory(tmp_path):
    assert extract.parse_study_argument(f"head_prone={tmp_path}") == (
        "head_prone",
        tmp_path,
    )
    with pytest.raises(ValueError, match="expected LABEL=PATH"):
        extract.parse_study_argument(str(tmp_path))
    with pytest.raises(ValueError, match="is not a directory"):
        extract.parse_study_argument(f"head_prone={tmp_path / 'missing'}")


def test_to_json_value_keeps_shapes_and_enum_text():
    value = {
        "matrix": np.array([[1, 0], [0, 1]]),
        "offset": np.float64(-6.33962),
        "enum": "<2D>",
        "packs": ["axial", np.int64(2)],
    }

    converted = {key: extract.to_json_value(item) for key, item in value.items()}

    assert converted == {
        "matrix": [[1, 0], [0, 1]],
        "offset": -6.33962,
        "enum": "<2D>",
        "packs": ["axial", 2],
    }
    assert type(converted["packs"][1]) is int
    with pytest.raises(TypeError, match="Cannot convert"):
        extract.to_json_value(object())


def test_format_fixture_round_trips_and_keeps_short_lists_inline():
    value = {
        "a": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "b": [[float(i), 0.0, 0.0] for i in range(12)],
        "c": {},
    }

    text = extract.format_fixture(value)

    assert json.loads(text) == value
    assert '"a": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]' in text
    assert "\n    [11.0, 0.0, 0.0]\n  ]" in text
