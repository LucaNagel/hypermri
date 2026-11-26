"""Protect frequency-axis errors and successful FFT-bin references."""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.analysis.spectroscopy.frequency import get_freq_axis


def _context(**settings):
    metadata = {"center_ppm": 170.0, "dwell_time_s": 0.001, **settings}
    return SimpleNamespace(
        bandwidth_hz=1000.0,
        bandwidth_ppm=10.0,
        n_acquisition=8,
        get=metadata.get,
    )


@pytest.mark.parametrize("npoints", [7, 8])
@pytest.mark.parametrize("centered", [False, True])
def test_explicit_hz_axis_preserves_shifted_bins_and_midpoint(npoints, centered):
    # Invented acquisition at 1 kHz, expected unnormalized DFT bin frequencies.
    expected = np.fft.fftshift(np.fft.fftfreq(npoints, d=0.001))
    if centered:
        expected = expected - (expected[0] + expected[-1]) / 2
    actual = get_freq_axis(
        unit="Hz", dwell_time_s=0.001, npoints=npoints, centered_at_0=centered
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("unit", ["Hz", "ppm"])
def test_metadata_axis_retains_calibration_cutoff_and_frequency_direction(unit):
    context = _context(cut_off=1)
    expected = np.fft.fftshift(np.fft.fftfreq(7, d=0.001))
    if unit == "ppm":
        expected = expected / 100.0 + 170.0
    actual = get_freq_axis(context, unit=unit)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(actual) > 0)


@pytest.mark.parametrize("arguments", [{}, {"npoints": 8}, {"dwell_time_s": 0.001}])
def test_missing_explicit_sampling_raises_actionable_error(arguments):
    with pytest.raises(ValueError, match="get_freq_axis.*dwell_time_s.*npoints"):
        get_freq_axis(unit="Hz", **arguments)


@pytest.mark.parametrize("context", [None, _context()])
def test_unknown_unit_raises_before_axis_preparation(context):
    with pytest.raises(ValueError, match="get_freq_axis.*unit.*kHz.*Hz.*ppm"):
        get_freq_axis(context, unit="kHz", dwell_time_s=0.001, npoints=8)


def test_explicit_ppm_still_requires_acquisition_calibration():
    with pytest.raises(ValueError, match="scan=None"):
        get_freq_axis(unit="ppm", dwell_time_s=0.001, npoints=8)
