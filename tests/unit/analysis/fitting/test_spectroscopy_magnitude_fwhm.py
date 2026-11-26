"""Absorption and magnitude FWHM of fitted spectroscopy lines.

A Lorentz line with T2* has the absorption FWHM 1/(pi*T2*) and the
magnitude FWHM sqrt(3)/(pi*T2*); both hold up to rounding (1e-12 relative).
Voigt magnitude widths are compared with the magnitude spectrum of the model
line, Fourier-transformed numerically: 0.1 ms sampling, 2**21 points (about
210 s, far beyond every decay), the first point weighted 1/2 (trapezoid
rule) and the half-maximum crossing interpolated linearly on the 0.005 Hz
grid. Discretization and interpolation errors are below 1e-5 relative, so
1e-4 relative is the tolerance. Fit fixtures are noise-free synthetic FIDs:
256 samples at 4 kHz, lines at -300, 0 and 250 Hz with areas 10, 30 and 3.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy import fid_model
from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv

DT_S = 1e-4
N_POINTS = 2**21
TIME_S = np.arange(256) / 4000.0
FREQS_HZ = np.array([-300.0, 0.0, 250.0])
AMPLITUDES = np.array([10.0, 30.0, 3.0]) * np.exp(1j * np.array([0.3, -0.2, 0.5]))


def transformed_magnitude_fwhm_hz(lorentz_fwhm_hz, gauss_fwhm_hz):
    """Return the magnitude FWHM of the numerically transformed model line."""
    t = np.arange(N_POINTS) * DT_S
    fid = np.exp(
        -np.pi * lorentz_fwhm_hz * t
        - (np.pi * gauss_fwhm_hz * t) ** 2 / (4 * np.log(2))
    )
    fid[0] *= 0.5
    magnitude = np.abs(np.fft.fft(fid))[: N_POINTS // 2]
    half = magnitude[0] / 2
    k = int(np.argmax(magnitude < half))
    fraction = (magnitude[k - 1] - half) / (magnitude[k - 1] - magnitude[k])
    return 2 * (k - 1 + fraction) / (N_POINTS * DT_S)


def test_lorentz_magnitude_width_is_sqrt3_times_the_absorption_width():
    t2s_s = np.array([0.01, 0.03, 0.2])
    absorption = 1 / (np.pi * t2s_s)

    magnitude = fid_model._magnitude_fwhm_hz(absorption, 0.0)

    np.testing.assert_allclose(magnitude, np.sqrt(3) / (np.pi * t2s_s), rtol=1e-12)


@pytest.mark.parametrize(
    ("lorentz_fwhm_hz", "gauss_fwhm_hz"),
    [(5.0, 0.0), (0.0, 10.0), (4.0, 8.0), (20.0, 3.0), (1.0, 30.0)],
)
def test_magnitude_width_matches_the_transformed_model_line(
    lorentz_fwhm_hz, gauss_fwhm_hz
):
    expected = transformed_magnitude_fwhm_hz(lorentz_fwhm_hz, gauss_fwhm_hz)

    width = fid_model._magnitude_fwhm_hz(lorentz_fwhm_hz, gauss_fwhm_hz)

    assert width == pytest.approx(expected, rel=1e-4)


def test_magnitude_width_keeps_labels_and_nan():
    lorentz = xr.DataArray(
        [[2.0, np.nan], [3.0, 4.0]],
        dims=("x", "metabolite"),
        coords={"x": [0, 1], "metabolite": ["a", "b"]},
    )
    gauss = xr.DataArray([np.nan, 6.0], dims="x", coords={"x": [0, 1]})

    width = fid_model._magnitude_fwhm_hz(lorentz, gauss)

    assert width.dims == ("x", "metabolite")
    assert np.isnan(width.values[0]).all()
    expected = [fid_model._magnitude_fwhm_hz(value, 6.0) for value in (3.0, 4.0)]
    np.testing.assert_allclose(width.values[1], expected, rtol=1e-12)


def three_line_fit(**options):
    """Return the noise-free three-line fit with the given lineshape options."""
    gauss_fwhm_hz = 14.0 if options.get("lineshape") == "voigt" else 0.0
    lines = AMPLITUDES[:, None] * np.exp(
        2j * np.pi * FREQS_HZ[:, None] * TIME_S
        - TIME_S / np.array([0.05, 0.04, 0.06])[:, None]
    )
    signal = lines.sum(0) * np.exp(
        -((np.pi * gauss_fwhm_hz * TIME_S) ** 2) / (4 * np.log(2))
    )
    data = xr.DataArray(signal[None], dims=["x", "acquisition_idx"])
    data.attrs["sampling_time_axis_s"] = TIME_S
    params = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b", "c"],
            "metabs_freqs_hz": list(FREQS_HZ + 1.0),
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 10.0,
            **options,
        },
    )
    return fit_data_pseudo_inv(data, params)


def test_lorentz_fit_reports_absorption_and_magnitude_widths():
    result = three_line_fit()

    np.testing.assert_allclose(
        result.fit_fwhm_hz, 1 / (np.pi * result.fit_t2s), rtol=1e-12
    )
    np.testing.assert_allclose(
        result.fit_magnitude_fwhm_hz, np.sqrt(3) * result.fit_fwhm_hz, rtol=1e-12
    )
    assert result.fit_magnitude_fwhm_hz.dims == result.fit_fwhm_hz.dims
    assert result.fit_magnitude_fwhm_hz.attrs["unit"] == "Hz"
    assert "magnitude" in result.fit_magnitude_fwhm_hz.attrs["definition"]
    assert "absorption" in result.fit_fwhm_hz.attrs["definition"]


def test_voigt_fit_reports_the_magnitude_width_of_the_whole_line():
    result = three_line_fit(lineshape="voigt")

    expected = fid_model._magnitude_fwhm_hz(
        result.fit_lorentz_fwhm_hz, result.fit_gauss_fwhm_hz
    )

    np.testing.assert_allclose(result.fit_magnitude_fwhm_hz, expected, rtol=1e-12)
    assert np.all(
        result.fit_magnitude_fwhm_hz > np.sqrt(3) * result.fit_lorentz_fwhm_hz
    )
    assert "absorption" in result.fit_fwhm_hz.attrs["definition"]
