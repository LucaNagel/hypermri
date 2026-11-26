"""Check the static field ``b0_tesla`` against its one definition.

Every vendor handler sets ``b0_tesla`` to the frequency of 0 ppm
(``reference_frequency_mhz``) divided by the gyromagnetic ratio of the scan's
nucleus from :func:`hypermri.utils.physics.get_gmr`. The owner's check value
is the 13C CSI example 75.4901665 MHz / 10.7084 MHz/T = 7.0496 T. All scans
here are synthetic parameter files with invented values; the Bruker scan is
the two-slice FLASH series of ``synthetic_flash_scan``.
"""

import warnings

import numpy as np
import pytest

from hypermri import load_scan
from hypermri.exceptions import DataLoadingError
from hypermri.utils.physics import b0_tesla_from_reference_frequency, get_gmr
from hypermri.vendors.agilent import AgilentHandler
from hypermri.vendors.bruker import BrukerHandler

# Ratios agree with the published values to the six digits stored in get_gmr.
_RELATIVE_TOLERANCE = 1e-12


@pytest.mark.parametrize("nucleus", ["13C", "C13", "13c"])
def test_owner_example_gives_seven_tesla(nucleus):
    b0 = b0_tesla_from_reference_frequency(75.4901665, nucleus)

    assert b0 == pytest.approx(75.4901665 / 10.7084, rel=_RELATIVE_TOLERANCE)
    assert round(b0, 4) == 7.0496


def test_element_first_deuterium_uses_the_deuterium_ratio():
    b0 = b0_tesla_from_reference_frequency(46.007062, "H2")

    assert b0 == pytest.approx(46.007062 / get_gmr("2H", unit="MHz_T"))


@pytest.mark.parametrize(
    ("frequency", "nucleus"),
    [(None, "13C"), (75.49, None), (75.49, ""), (75.49, "unknown")],
)
def test_missing_frequency_or_nucleus_gives_no_field(frequency, nucleus):
    assert b0_tesla_from_reference_frequency(frequency, nucleus) is None


@pytest.mark.parametrize("frequency", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_frequency_is_rejected(frequency):
    with pytest.raises(ValueError, match="reference_frequency_mhz"):
        b0_tesla_from_reference_frequency(frequency, "13C")


def test_unknown_nucleus_is_rejected():
    with pytest.raises(ValueError, match="'99Xx'.*get_gmr"):
        b0_tesla_from_reference_frequency(75.49, "99Xx")


def _bruker_scan(scan, write_paravision_parameters, nucleus, frequencies):
    write_paravision_parameters(
        scan / "method",
        {
            "Method": "<Bruker:FLASH>",
            "PVM_Nucleus1": f"<{nucleus}>",
            "PVM_FrqRef": frequencies,
        },
    )
    return scan


def test_bruker_field_uses_the_first_reference_frequency(
    synthetic_flash_scan, write_paravision_parameters
):
    _bruker_scan(
        synthetic_flash_scan,
        write_paravision_parameters,
        "13C",
        [75.4901665, 300.2, 0, 0, 0, 0, 0, 0],
    )
    handler = BrukerHandler(synthetic_flash_scan)

    assert handler.reference_frequency_mhz == 75.4901665
    assert handler.b0_tesla == pytest.approx(75.4901665 / 10.7084)
    assert handler.params["b0_tesla"] == handler.b0_tesla


def test_bruker_unknown_nucleus_names_the_scan(
    synthetic_flash_scan, write_paravision_parameters
):
    _bruker_scan(synthetic_flash_scan, write_paravision_parameters, "99Xx", [75.49])

    with pytest.raises(DataLoadingError, match="'99Xx'.*get_gmr") as caught:
        BrukerHandler(synthetic_flash_scan)

    assert str(synthetic_flash_scan) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


def _procpar_entry(name, value):
    if isinstance(value, str):
        return f'{name} 2 2 8 0 0 2 1 0 1 64\n1 "{value}"\n0\n'
    return f"{name} 1 1 1e+18 -1e+18 0 2 1 0 1 64\n1 {value}\n0\n"


def _agilent_scan(tmp_path, parameters):
    scan = tmp_path / "s_2D.fid"
    scan.mkdir()
    (scan / "procpar").write_text(
        "".join(_procpar_entry(name, value) for name, value in parameters.items()),
        encoding="utf-8",
    )
    (scan / "fid").write_bytes(b"")
    return scan


@pytest.mark.parametrize(
    ("nucleus", "reffrq", "canonical"),
    [("H1", 299.7766, "1H"), ("C13", 75.3815, "13C"), ("H2", 46.0070, "2H")],
)
def test_agilent_field_follows_the_reference_frequency(
    tmp_path, nucleus, reffrq, canonical
):
    # The procpar B0 (gauss) and sfrq differ on purpose: neither may set the
    # field any more.
    scan = _agilent_scan(
        tmp_path,
        {"tn": nucleus, "reffrq": reffrq, "sfrq": reffrq + 0.001, "B0": 70000.0},
    )
    handler = AgilentHandler(scan)

    assert handler.reference_frequency_mhz == reffrq
    assert handler.b0_tesla == pytest.approx(
        reffrq / get_gmr(canonical, unit="MHz_T"), rel=_RELATIVE_TOLERANCE
    )


def test_agilent_without_reference_frequency_has_no_field(tmp_path):
    scan = _agilent_scan(tmp_path, {"tn": "C13", "sfrq": 75.3815, "B0": 70000.0})

    assert AgilentHandler(scan).b0_tesla is None


def test_scan_reads_field_and_frequency_axes_without_warning(
    synthetic_flash_scan, write_paravision_parameters
):
    _bruker_scan(synthetic_flash_scan, write_paravision_parameters, "13C", [75.4901665])
    scan = load_scan(synthetic_flash_scan, mode="lazy", log_mode="critical")
    handler = scan.handler

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert scan.reference_frequency_mhz == handler.reference_frequency_mhz
        assert scan.b0_tesla == handler.b0_tesla
        frequency_axes = {
            "freq_axis_ppm": scan.freq_axis_ppm,
            "freq_axis_hz": scan.freq_axis_hz,
            "ppm": scan.ppm,
            "hz": scan.hz,
        }

    # The FLASH series has no spectral bandwidth, so every axis is empty.
    assert frequency_axes == {name: [] for name in frequency_axes}
    assert handler.freq_axis_ppm == handler.freq_axis_hz == []
    assert scan.__dict__["_data_loaded"] is False
    assert scan.__dict__["_data"] is None


def test_scan_frequency_axes_equal_the_handler_axes(
    synthetic_flash_scan, write_paravision_parameters
):
    _bruker_scan(synthetic_flash_scan, write_paravision_parameters, "13C", [75.4901665])
    scan = load_scan(synthetic_flash_scan, mode="lazy", log_mode="critical")
    # Invented spectral metadata: 8 Hz over 4 points at 2 Hz/ppm.
    spectral = {
        "bandwidth_hz": 8.0,
        "bandwidth_ppm": 4.0,
        "dwell_time_s": 0.125,
        "n_acquisition": 4,
        "cut_off": 0,
        "center_ppm": 0.0,
    }
    for name, value in spectral.items():
        setattr(scan.handler, name, value)
        scan.params[name] = value

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        np.testing.assert_array_equal(scan.freq_axis_hz, scan.handler.freq_axis_hz)
        np.testing.assert_array_equal(scan.freq_axis_ppm, scan.handler.freq_axis_ppm)
        np.testing.assert_array_equal(scan.hz, scan.freq_axis_hz)
        np.testing.assert_array_equal(scan.ppm, scan.freq_axis_ppm)

    np.testing.assert_array_equal(scan.freq_axis_hz, [-4.0, -2.0, 0.0, 2.0])


_COMMON_ACQUISITION_VALUES = (
    "nucleus",
    "reference_frequency_mhz",
    "b0_tesla",
    "bandwidth_hz",
    "dwell_time_s",
    "center_ppm",
    "n_acquisition",
    "n_repetitions",
    "sampling_time_axis_s",
    "repetition_time_ms",
    "echo_time_ms",
    "fov_mm",
    "matrix",
    "resolution_mm",
)


@pytest.mark.parametrize("name", _COMMON_ACQUISITION_VALUES)
def test_scan_reads_common_acquisition_values_like_the_handler(
    name, synthetic_flash_scan, write_paravision_parameters
):
    _bruker_scan(synthetic_flash_scan, write_paravision_parameters, "13C", [75.4901665])
    scan = load_scan(synthetic_flash_scan, mode="lazy", log_mode="critical")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        value = getattr(scan, name)

    np.testing.assert_equal(value, getattr(scan.handler, name))
    assert scan.__dict__["_data_loaded"] is False
