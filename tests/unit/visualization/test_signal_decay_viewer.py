"""Synthetic signal-decay diagnostics, scientific settings and owned UI events.

Eight-point quadrature tones and exact labelled grids need 1e-12 a.u.
tolerance in complex128, with 1e-7 only for float32 mask coverage.
Band-power tones sit exactly on FFT bins, so their analytic power holds to
1e-12 relative in complex128. Noise checks use seeded invented Gaussian noise
and a four-standard-error tolerance.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.anatomical.roi import raw_masked_mean
from hypermri.analysis.spectroscopy.signal_decay import (
    signal_decay_curves,
    spectral_band_power,
)
from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.sequences.base_spectroscopy import BaseSpectroscopy
from hypermri.sequences.CSI import CSI
from hypermri.visualization import PlotOptions
from hypermri.visualization.contracts import RenderPayload
from hypermri.visualization.signal_decay import _mean_preview, _SignalDecayData


def test_tail_fit_subtraction_and_invalid_tail_without_widgets():
    matrix = np.array([[8, 6, 4, 2], [8j, 6j, 4j, 2j]])
    curves = signal_decay_curves(matrix, end_index=2, subtract_tail=True)
    np.testing.assert_allclose(curves["magnitude"], 0, atol=1e-12)
    np.testing.assert_allclose(curves["magnitude_tail"], 0, atol=1e-12)
    assert signal_decay_curves(matrix, end_index=3)["magnitude_tail"].size == 0


def _bin_tone(amplitude, frequency_hz, n_samples=64, dwell_time_s=1e-3):
    """Return an invented complex tone whose frequency is an exact FFT bin."""
    t = np.arange(n_samples) * dwell_time_s
    return amplitude * np.exp(2j * np.pi * frequency_hz * t + 0.3j)


def test_spectral_band_power_recovers_tone_power_and_parseval():
    """A 64-point tone at 250 Hz (1 ms dwell, 15.625 Hz bins) has power N a^2 in its bin."""
    matrix = np.stack([_bin_tone(3.0, 250.0), _bin_tone(2.0, -125.0)], axis=1)
    # [200, 300] Hz holds the bins 200 + k * 15.625 Hz for k = 0..6.
    power = spectral_band_power(matrix, dwell_time_s=1e-3, bands_hz=[(200, 300)])
    np.testing.assert_allclose(power[0], 64 * 3.0**2 / 7, rtol=1e-12)
    assert abs(power[1]) < 1e-12 * power[0]
    # Without bands the value equals mean |s|^2 over the FID (Parseval).
    whole = spectral_band_power(matrix, dwell_time_s=1e-3)
    np.testing.assert_allclose(whole, [3.0**2, 2.0**2], rtol=1e-12)
    # Overlapping bands select [200, 320] Hz = 8 bins, each once.
    overlap = spectral_band_power(
        matrix, dwell_time_s=1e-3, bands_hz=[(200, 300), (250, 320)]
    )
    np.testing.assert_allclose(overlap[0], 64 * 3.0**2 / 8, rtol=1e-12)
    assert power.dtype == np.float64


def test_spectral_band_power_of_white_noise_is_two_sigma_squared():
    """Seeded invented noise: any band has expectation 2 sigma^2 per complex sample."""
    rng = np.random.default_rng(3)
    sigma, n, points = 2.5, 350, 400
    noise = rng.normal(0, sigma, (n, points)) + 1j * rng.normal(0, sigma, (n, points))
    dwell = 1 / 3000.0
    bands = [(-330, -150), (200, 420)]
    power = spectral_band_power(noise, dwell_time_s=dwell, bands_hz=bands)
    frequency = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    n_bins = sum(((frequency >= lo) & (frequency <= hi)).sum() for lo, hi in bands)
    standard_error = 2 * sigma**2 / np.sqrt(n_bins * points)
    assert abs(power.mean() - 2 * sigma**2) < 4 * standard_error


def test_band_power_finds_weak_late_signal_the_magnitude_averages_away():
    """A weak in-band tone in 1000 of 2000 noisy FIDs stands out more in band power."""
    rng = np.random.default_rng(7)
    n, points, dwell = 128, 2000, 1e-3
    matrix = rng.normal(0, 1, (n, points)) + 1j * rng.normal(0, 1, (n, points))
    matrix[:, :1000] += _bin_tone(0.15, 250.0, n_samples=n)[:, None]

    def separation(quantity, **options):
        curve = signal_decay_curves(matrix, quantity=quantity, **options)[quantity]
        signal, noise = curve[:1000], curve[1000:]
        return (signal.mean() - noise.mean()) / (noise.std() * np.sqrt(2 / 1000))

    magnitude = separation("magnitude")
    band = separation("band_power", dwell_time_s=dwell, bands_hz=[(220, 280)])
    assert band > 6
    assert band > 2 * magnitude


def test_band_power_curves_keep_the_decay_curve_contract():
    matrix = np.stack([_bin_tone(a, 250.0) for a in (4.0, 3.0, 2.0, 1.0)], axis=1)
    curves = signal_decay_curves(
        matrix,
        end_index=2,
        quantity="band_power",
        dwell_time_s=1e-3,
        bands_hz=[(240, 260)],
    )
    assert set(curves) == {
        "band_power",
        "band_power_tail",
        "cumulative",
        "cumulative_tail",
    }
    np.testing.assert_allclose(curves["band_power"], 64 * np.array([16, 9, 4, 1.0]))
    assert curves["cumulative"][-1] == pytest.approx(1.0)
    assert curves["band_power_tail"].size == 4


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"bands_hz": [(300, 200)]}, "low < high"),
        ({"bands_hz": [(np.nan, 200)]}, "finite"),
        ({"bands_hz": [(200,)]}, "pair in Hz"),
        ({"bands_hz": []}, "at least one"),
        ({"bands_hz": [(201, 202)]}, "no frequency sample"),
        ({"dwell_time_s": 0.0}, "positive dwell_time_s"),
    ],
)
def test_spectral_band_power_rejects_invalid_sampling_and_bands(options, message):
    arguments = {"dwell_time_s": 1e-3, **options}
    with pytest.raises(ValueError, match=message):
        spectral_band_power(_bin_tone(1.0, 250.0)[:, None], **arguments)


def test_signal_decay_curves_reject_unknown_quantity_and_missing_dwell():
    matrix = _bin_tone(1.0, 250.0)[:, None]
    with pytest.raises(ValueError, match="'magnitude' or 'band_power'"):
        signal_decay_curves(matrix, quantity="energy")
    with pytest.raises(ValueError, match="requires dwell_time_s"):
        signal_decay_curves(matrix, quantity="band_power")
