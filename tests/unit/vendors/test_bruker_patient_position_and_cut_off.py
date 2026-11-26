"""Regression tests for Bruker patient position and FID cut-off metadata.

All parameter files are synthetic. The patient position strings follow the
``ACQ_patient_pos`` values of a ParaVision 7.0.0 study (28 October 2026) whose
17 scans all read ``Foot_Prone``. The CPMG echo window of 10 points and the
70-point FID cut-off of stock spectroscopy are taken from the same study.
"""

import numpy as np
import pytest

import hypermri.vendors.bruker as bruker_module
from hypermri.vendors.bruker import BrukerHandler


def _handler(tmp_path, write_parameters, *, method, patient_pos="Head_Prone"):
    scan_dir = tmp_path / "1"
    write_parameters(scan_dir / "method", method)
    write_parameters(
        scan_dir / "acqp",
        {"ACQ_grad_matrix": [np.eye(3).tolist()], "ACQ_patient_pos": patient_pos},
    )
    write_parameters(scan_dir / "pdata" / "1" / "reco", {})
    return BrukerHandler(scan_dir, log_mode="critical")


def _spectroscopy_method(method_name, n_points):
    return {
        "Method": f"<Bruker:{method_name}>",
        "PVM_EncSpectroscopy": "Yes",
        "PVM_SpecMatrix": [n_points],
        "PVM_SpecSWH": [25000.0],
        "PVM_SpecSW": [200.0],
    }


@pytest.mark.parametrize(
    ("patient_pos", "expected"),
    [
        ("Foot_Prone", [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        ("Foot_Supine", np.eye(3)),
        ("Head_Prone", [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    ],
)
def test_paravision_patient_positions_select_their_position_matrix(
    tmp_path, write_paravision_parameters, patient_pos, expected
):
    """Previously ``Foot_Prone`` fell through to the identity matrix."""
    handler = _handler(
        tmp_path,
        write_paravision_parameters,
        method=_spectroscopy_method("SINGLEPULSE", 2048),
        patient_pos=patient_pos,
    )

    np.testing.assert_array_equal(handler.position_matrix, expected)


def test_unknown_patient_position_warns_and_keeps_the_identity(
    tmp_path, write_paravision_parameters, monkeypatch
):
    # The handler's log_mode silences its logger, so the call is recorded.
    messages = []
    monkeypatch.setattr(
        bruker_module.logger,
        "warning",
        lambda message, *args: messages.append(message % args),
    )
    handler = _handler(
        tmp_path,
        write_paravision_parameters,
        method=_spectroscopy_method("SINGLEPULSE", 2048),
        patient_pos="Feet_Prone",
    )

    np.testing.assert_array_equal(handler.position_matrix, np.eye(3))
    assert any("unknown ACQ_patient_pos 'Feet_Prone'" in m for m in messages)


@pytest.mark.parametrize(
    ("method_name", "n_points", "expected_cut_off"),
    [("SINGLEPULSE", 2048, 70), ("NSPECT", 4096, 70), ("CPMG", 10, 0)],
)
def test_stock_cut_off_applies_to_fids_but_not_to_cpmg_echoes(
    tmp_path, write_paravision_parameters, method_name, n_points, expected_cut_off
):
    """Previously a 10-point CPMG echo got the 70-point FID cut-off."""
    handler = _handler(
        tmp_path,
        write_paravision_parameters,
        method=_spectroscopy_method(method_name, n_points),
    )

    assert handler.cut_off == expected_cut_off
    assert handler.n_acquisition - handler.cut_off > 0
