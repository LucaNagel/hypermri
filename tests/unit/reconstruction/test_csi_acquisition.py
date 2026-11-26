"""Tests for the vendor-neutral CSI acquisition description.

The Bruker scans are synthetic: a parameter-only ParaVision folder (``method``,
``acqp`` and an empty ``pdata``) with an invented 3 x 2 CSI matrix, no raw
data and no identifying metadata.  Every field of the description is compared
with the value the existing code reads from the parsed handler, so the
description adds no numerical change.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy
from hypermri.vendors.bruker import BrukerHandler
from hypermri.vendors.bruker_acquisition import bruker_to_csi_acquisition

_CSI_METHOD = {
    "Method": "<Bruker:CSI>",
    "PVM_Matrix": [3, 2],
    "PVM_Fov": [30.0, 20.0],
    "PVM_SliceThick": 4.0,
    "PVM_EncOrder": ["LINEAR_ENC", "LINEAR_ENC"],
    "PVM_EncSteps0": [0, -1, 1],
    "PVM_EncSteps1": [0, -1],
    "PVM_SpecMatrix": 64,
    "PVM_DigDw": 0.1,
    "PVM_FrqRef": [75.4, 0, 0, 0, 0, 0, 0, 0],
    "PVM_FrqWorkPpm": [171.0, 0, 0, 0, 0, 0, 0, 0],
    "PVM_SPackArrReadOffset": [1.5],
    "PVM_SPackArrPhase1Offset": [-2.0],
    "PVM_SPackArrSliceOffset": [0.5],
    "PVM_NRepetitions": 1,
    "PVM_EncNReceivers": 1,
}


def _csi_handler(tmp_path, write_paravision_parameters, **method) -> BrukerHandler:
    """Return a parsed Bruker handler of a parameter-only synthetic CSI scan."""
    scan = tmp_path / "scan"
    write_paravision_parameters(scan / "method", {**_CSI_METHOD, **method})
    write_paravision_parameters(scan / "acqp", {"ACQ_scan_name": "<synthetic>"})
    (scan / "pdata").mkdir()
    return BrukerHandler(str(scan))


def _valid_fields(**changes) -> dict:
    fields = {
        "encoding_order": "LINEAR_ENC LINEAR_ENC",
        "matrix": (3, 2),
        "fov_mm": (30.0, 20.0),
        "fov_offset_mm": (0.0, 0.0, 0.0),
        "read_positions": [0, 1, 2, 0, 1, 2],
        "phase_positions": [0, 0, 0, 1, 1, 1],
        "n_points": 64,
        "dwell_time_s": 1e-4,
        "reference_frequency_mhz": 75.4,
        "center_ppm": 171.0,
    }
    fields.update(changes)
    return fields


def test_bruker_csi_description_matches_every_value_the_code_reads(
    tmp_path, write_paravision_parameters
):
    handler = _csi_handler(tmp_path, write_paravision_parameters)
    order, read, phase = BrukerCsiRecoStrategy().get_phase_encoding_order(handler)

    acquisition = bruker_to_csi_acquisition(handler)

    assert acquisition.encoding_order == order == "LINEAR_ENC LINEAR_ENC"
    assert acquisition.matrix == tuple(int(n) for n in handler.matrix[:2]) == (3, 2)
    assert acquisition.fov_mm == tuple(float(v) for v in handler.fov_mm[:2])
    assert acquisition.fov_mm == (30.0, 20.0)
    assert acquisition.fov_offset_mm == (
        handler.read_offset_mm,
        handler.phase_offset_mm,
        handler.slice_offset_mm,
    )
    assert acquisition.fov_offset_mm == (1.5, -2.0, 0.5)
    np.testing.assert_array_equal(acquisition.read_positions, read)
    np.testing.assert_array_equal(acquisition.phase_positions, phase)
    np.testing.assert_array_equal(acquisition.read_positions, [1, 0, 2, 1, 0, 2])
    np.testing.assert_array_equal(acquisition.phase_positions, [1, 1, 1, 0, 0, 0])
    assert acquisition.n_fids == 6
    assert acquisition.n_points == handler.n_acquisition == 64
    assert acquisition.dwell_time_s == handler.dwell_time_s == pytest.approx(1e-4)
    assert acquisition.reference_frequency_mhz == handler.reference_frequency_mhz
    assert acquisition.reference_frequency_mhz == 75.4
    assert acquisition.center_ppm == handler.center_ppm == 171.0
    assert {field.name for field in dataclasses.fields(acquisition)} == set(
        _valid_fields()
    )


def test_bruker_csi_description_does_not_change_the_handler(
    tmp_path, write_paravision_parameters
):
    handler = _csi_handler(tmp_path, write_paravision_parameters)
    method_before = {key: np.copy(value) for key, value in handler.method.items()}
    matrix_before = handler.matrix.copy()

    bruker_to_csi_acquisition(handler)

    assert handler.method.keys() == method_before.keys()
    for key, value in method_before.items():
        np.testing.assert_array_equal(handler.method[key], value)
    np.testing.assert_array_equal(handler.matrix, matrix_before)


def test_handler_accessor_returns_a_new_equal_description_each_call(
    tmp_path, write_paravision_parameters
):
    handler = _csi_handler(tmp_path, write_paravision_parameters)

    first = handler.to_csi_acquisition()
    second = handler.to_csi_acquisition()
    expected = bruker_to_csi_acquisition(handler)

    assert first is not second
    for field in dataclasses.fields(CSIAcquisition):
        np.testing.assert_array_equal(
            getattr(first, field.name), getattr(expected, field.name)
        )
        np.testing.assert_array_equal(
            getattr(second, field.name), getattr(expected, field.name)
        )


def test_bruker_csi_description_reports_an_unsupported_order(
    tmp_path, write_paravision_parameters
):
    handler = _csi_handler(
        tmp_path,
        write_paravision_parameters,
        PVM_EncOrder=["LINEAR_ENC", "CENTRIC_ENC"],
    )

    with pytest.raises(ValueError, match="PVM_EncOrder"):
        bruker_to_csi_acquisition(handler)


def test_description_is_immutable_and_stores_read_only_int64_positions():
    acquisition = CSIAcquisition(**_valid_fields())

    assert acquisition.read_positions.dtype == np.int64
    assert not acquisition.read_positions.flags.writeable
    assert not acquisition.phase_positions.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        acquisition.n_points = 128
    with pytest.raises(ValueError):
        acquisition.read_positions[0] = 2


def test_description_copies_the_position_input():
    read = np.array([0, 1, 2, 0, 1, 2])
    acquisition = CSIAcquisition(**_valid_fields(read_positions=read))

    read[0] = 2

    assert acquisition.read_positions[0] == 0


def test_description_allows_missing_dwell_time_reference_frequency_and_fov():
    acquisition = CSIAcquisition(
        **_valid_fields(dwell_time_s=None, reference_frequency_mhz=None, fov_mm=None)
    )

    assert acquisition.dwell_time_s is None
    assert acquisition.reference_frequency_mhz is None
    assert acquisition.fov_mm is None


def test_bruker_scan_without_field_of_view_keeps_its_positions(
    tmp_path, write_paravision_parameters
):
    scan = tmp_path / "scan"
    method = {key: value for key, value in _CSI_METHOD.items() if key != "PVM_Fov"}
    write_paravision_parameters(scan / "method", method)
    write_paravision_parameters(scan / "acqp", {"ACQ_scan_name": "<synthetic>"})
    (scan / "pdata").mkdir()
    handler = BrukerHandler(str(scan))

    acquisition = bruker_to_csi_acquisition(handler)

    assert handler.fov_mm is None
    assert acquisition.fov_mm is None
    np.testing.assert_array_equal(acquisition.read_positions, [1, 0, 2, 1, 0, 2])
    np.testing.assert_array_equal(acquisition.phase_positions, [1, 1, 1, 0, 0, 0])


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"encoding_order": ""}, "encoding_order"),
        ({"matrix": (3,)}, "matrix"),
        ({"matrix": (3.5, 2)}, "matrix"),
        ({"fov_mm": (30.0, 0.0)}, "fov_mm"),
        ({"fov_offset_mm": (0.0, np.nan, 0.0)}, "fov_offset_mm"),
        ({"read_positions": [0, 1, 3, 0, 1, 2]}, "read_positions"),
        ({"phase_positions": [0, 0, 0, 1, 1, -1]}, "phase_positions"),
        ({"phase_positions": [0, 0, 0, 1, 1]}, "read_positions and phase_positions"),
        ({"read_positions": [0.5, 1, 2, 0, 1, 2]}, "read_positions"),
        ({"n_points": 0}, "n_points"),
        ({"dwell_time_s": -1e-4}, "dwell_time_s"),
        ({"reference_frequency_mhz": np.inf}, "reference_frequency_mhz"),
        ({"center_ppm": np.nan}, "center_ppm"),
    ],
)
def test_description_rejects_invalid_fields_and_names_them(changes, field):
    with pytest.raises(ValueError, match=f"CSIAcquisition.{field}"):
        CSIAcquisition(**_valid_fields(**changes))
