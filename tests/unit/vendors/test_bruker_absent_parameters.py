"""Check how ``BrukerHandler`` reads optional slice parameters and sample counts.

The scans are the synthetic two-slice FLASH series of ``synthetic_flash_scan``
with invented ``method`` parameters. The values each case expects are the
ones the handler gave before its broad ``BaseException`` handlers were
narrowed, so the change keeps every result.
"""

import logging

import numpy as np
import pytest

from hypermri.vendors.bruker import BrukerHandler, _has_samples


def _handler(scan, write_paravision_parameters, parameters, log_mode="error"):
    write_paravision_parameters(
        scan / "method", {"Method": "<Bruker:FLASH>", **parameters}
    )
    return BrukerHandler(scan, log_mode=log_mode)


@pytest.mark.parametrize(
    ("parameters", "n_slices"),
    [({"SliceNum": 3}, 3), ({}, 1), ({"PVM_SPackArrNSlices": [2, 1]}, 3)],
)
def test_slice_count_without_packages_falls_back_to_slice_num(
    synthetic_flash_scan, write_paravision_parameters, parameters, n_slices
):
    handler = _handler(synthetic_flash_scan, write_paravision_parameters, parameters)

    assert handler.n_slices == n_slices


@pytest.mark.parametrize(
    ("parameters", "slice_gap_mm"),
    [
        ({"PVM_SPackArrSliceGap": [0.5, 1.5]}, 0.5),
        ({"PVM_SPackArrSliceGap": 0.25}, 0.25),
        ({}, 0),
    ],
)
def test_slice_gap_reads_the_first_package(
    synthetic_flash_scan, write_paravision_parameters, parameters, slice_gap_mm
):
    handler = _handler(synthetic_flash_scan, write_paravision_parameters, parameters)

    assert handler.slice_gap_mm == slice_gap_mm


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, False),
        (np.zeros(0), False),
        ([], False),
        (np.zeros((2, 2)), True),
        ([0], True),
        (np.float64(1.0), True),
        (np.array(1.0), True),
        (1.0, True),
    ],
)
def test_loaded_data_with_samples(data, expected):
    # A 0-d array has a length method but no length; it holds one sample.
    assert _has_samples(data) is expected


def test_sample_check_does_not_hide_other_errors():
    class _BrokenLength:
        def __len__(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _has_samples(_BrokenLength())


def _acqp_with_time(scan, write_paravision_parameters, acq_time):
    parameters = {
        "ACQ_scan_name": "<synthetic>",
        "ACQ_grad_matrix": [np.eye(3, dtype=int).tolist()] * 2,
    }
    if acq_time is not None:
        parameters["ACQ_time"] = acq_time
    write_paravision_parameters(scan / "acqp", parameters)


def test_measurement_time_is_read_from_acq_time(
    synthetic_flash_scan, write_paravision_parameters
):
    _acqp_with_time(
        synthetic_flash_scan,
        write_paravision_parameters,
        "<2024-03-01T12:34:56,789+0100>",
    )

    handler = _handler(synthetic_flash_scan, write_paravision_parameters, {})

    assert handler.measurement_time.isoformat() == "2024-03-01T12:34:56.789000+01:00"


def test_measurement_time_without_acq_time_is_none(
    synthetic_flash_scan, write_paravision_parameters
):
    _acqp_with_time(synthetic_flash_scan, write_paravision_parameters, None)

    handler = _handler(synthetic_flash_scan, write_paravision_parameters, {})

    assert handler.measurement_time is None


def test_measurement_time_in_another_format_is_none_with_a_warning(
    synthetic_flash_scan, write_paravision_parameters, caplog
):
    # Invented value in the clock-first style of older ParaVision versions.
    _acqp_with_time(
        synthetic_flash_scan, write_paravision_parameters, "<12:34:56  1 Mar 2024>"
    )

    with caplog.at_level(logging.WARNING):
        handler = _handler(
            synthetic_flash_scan, write_paravision_parameters, {}, log_mode="warning"
        )

    assert handler.measurement_time is None
    assert "ACQ_time" in caplog.text
    assert str(synthetic_flash_scan) in caplog.text


def test_method_parameter_returns_the_stored_value(
    synthetic_flash_scan, write_paravision_parameters
):
    handler = _handler(
        synthetic_flash_scan, write_paravision_parameters, {"EchoSpacing": 5.0}
    )

    assert handler.method_parameter("EchoSpacing") == handler.method["EchoSpacing"]


def test_method_parameter_names_parameter_and_scan_when_missing(
    synthetic_flash_scan, write_paravision_parameters
):
    handler = _handler(synthetic_flash_scan, write_paravision_parameters, {})

    with pytest.raises(KeyError) as error:
        handler.method_parameter("NEchoes")

    # KeyError.__str__ quotes its argument and doubles Windows backslashes.
    message = error.value.args[0]
    assert "'NEchoes'" in message
    assert str(handler.expnum) in message
    assert str(synthetic_flash_scan) in message
