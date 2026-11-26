"""Local spectroscopy interval contracts for synthetic Lorentzian FIDs."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.stats import chi2, f, norm, t

from hypermri.analysis.fitting.spectroscopy import def_fit_params
from hypermri.analysis.spectroscopy import (
    estimate_fit_local_uncertainty,
    fit_spectrum,
)


def _case(
    *,
    amplitude=10 + 0j,
    sigma=0.1,
    frozen=False,
    frequency_width=70.0,
    t2_width=0.04,
    min_t2=0.005,
    max_t2=0.07,
):
    """Return one exact synthetic fit with externally supplied noise scale."""
    time = np.arange(96) * 0.001
    frequency = 40.0
    t2_star = 0.025
    signal = amplitude * np.exp(-time / t2_star + 2j * np.pi * time * frequency)
    data = xr.DataArray(
        signal,
        dims="acquisition_idx",
        coords={
            "acquisition_idx": np.arange(time.size),
            "sampling_time_axis_s": ("acquisition_idx", time),
        },
        attrs={"signal_domain": "temporal", "unit": "a.u."},
    )
    params = def_fit_params(
        data,
        {
            "metabs": ["peak"],
            "metabs_freqs_hz": [frequency],
            "initial_t2s_s": [t2_star],
            "metabs_freq_ranges": [0.0 if frozen else frequency_width],
            "metabs_t2_ranges": [0.0 if frozen else t2_width],
            "min_t2_s": [min_t2],
            "max_t2_s": [max_t2],
        },
    )
    fit = fit_spectrum(data, params, noise=sigma)
    return params, fit


def test_known_sigma_amplitude_ellipse_and_fixed_parameter_intervals():
    """An isotropic fixed-basis amplitude ellipse has exact radial bounds."""
    params, fit = _case(amplitude=10 + 3j, frozen=True)
    result = estimate_fit_local_uncertainty(fit, params)
    radius = (
        np.sqrt(chi2.ppf(0.95, 2))
        * result.parameter_std.sel(fit_parameter="amplitude_real:peak").item()
    )

    assert result.amplitude_abs_interval_low.item() == pytest.approx(
        abs(10 + 3j) - radius
    )
    assert result.amplitude_abs_interval_high.item() == pytest.approx(
        abs(10 + 3j) + radius
    )
    assert result.amplitude_abs_interval_status.item() == "success"
    assert result.frequency_interval_status.item() == "fixed_by_assumption"
    assert result.t2_interval_status.item() == "fixed_by_assumption"
    assert result.frequency_interval_low.item() == pytest.approx(40.0)
    assert result.frequency_interval_high.item() == pytest.approx(40.0)
    assert result.t2_interval_low.item() == pytest.approx(0.025)
    assert result.t2_interval_high.item() == pytest.approx(0.025)
    assert np.isinf(result.noise_degrees_of_freedom)
    assert result.interval_critical_value.item() == pytest.approx(norm.ppf(0.975))
    assert result.attrs["noise_scale_scope"] == "known_sigma"


def test_independent_noise_degrees_of_freedom_change_critical_values():
    """An estimated independent sigma uses t and scaled-F critical values."""
    params, fit = _case(frozen=True)
    result = estimate_fit_local_uncertainty(
        fit,
        params,
        noise_degrees_of_freedom=18,
    )

    assert result.interval_critical_value.item() == pytest.approx(t.ppf(0.975, 18))
    assert result.amplitude_ellipse_radius_squared.item() == pytest.approx(
        2 * f.ppf(0.95, 2, 18)
    )
    assert result.noise_degrees_of_freedom.item() == 18
    assert result.attrs["noise_scale_scope"] == "independent_estimated_sigma"


def test_log_t2_and_frequency_wald_intervals_match_documented_formulas():
    """Regular free parameters use marginal frequency and positive T2* bounds."""
    params, fit = _case()
    result = estimate_fit_local_uncertainty(
        fit,
        params,
        max_grid_step_fraction=1e6,
    )
    critical = norm.ppf(0.975)
    expected_frequency_half_width = critical * result.frequency_std.item()
    expected_log_t2_half_width = critical * result.t2_std.item() / 0.025

    assert result.frequency_interval_status.item() == "success"
    assert result.t2_interval_status.item() == "success"
    assert result.frequency_interval_low.item() == pytest.approx(
        40.0 - expected_frequency_half_width
    )
    assert result.frequency_interval_high.item() == pytest.approx(
        40.0 + expected_frequency_half_width
    )
    assert result.t2_interval_low.item() == pytest.approx(
        0.025 * np.exp(-expected_log_t2_half_width)
    )
    assert result.t2_interval_high.item() == pytest.approx(
        0.025 * np.exp(expected_log_t2_half_width)
    )
    assert result.t2_interval_low.item() > 0
    assert result.fwhm_interval_low.item() == pytest.approx(
        1 / (np.pi * result.t2_interval_high.item())
    )
    for side in ("low", "high"):
        magnitude = result[f"magnitude_fwhm_interval_{side}"]
        assert magnitude.item() == pytest.approx(
            np.sqrt(3) * result[f"fwhm_interval_{side}"].item(), rel=1e-12
        )
        assert magnitude.attrs["unit"] == "Hz"
        assert magnitude.attrs["confidence_level"] == result.confidence_level


def test_near_zero_amplitude_suppresses_free_frequency_and_t2_intervals():
    """A weak component cannot receive identifiable nonlinear parameters."""
    params, fit = _case(amplitude=0.001 + 0j)
    result = estimate_fit_local_uncertainty(
        fit,
        params,
        max_grid_step_fraction=1e6,
    )

    assert result.amplitude_abs_interval_status.item() == "includes_zero"
    assert result.amplitude_abs_interval_low.item() == 0
    assert np.isfinite(result.amplitude_abs_interval_high)
    assert result.frequency_interval_status.item() == "amplitude_near_zero"
    assert result.t2_interval_status.item() == "amplitude_near_zero"
    assert np.isnan(result.frequency_interval_low)
    assert np.isnan(result.t2_interval_high)


def test_grid_and_parameter_bound_diagnostics_suppress_local_intervals():
    """Numerical resolution and intervals crossing bounds request a fallback."""
    params, fit = _case()
    grid_limited = estimate_fit_local_uncertainty(
        fit, params, max_grid_step_fraction=1e-6
    )
    assert grid_limited.frequency_interval_status.item() == "grid_limited"
    assert grid_limited.t2_interval_status.item() == "grid_limited"
    assert np.isnan(grid_limited.frequency_interval_low)

    params, fit = _case(
        amplitude=1 + 0j,
        frequency_width=0.1,
        t2_width=0.001,
        min_t2=0.0245,
        max_t2=0.0255,
    )
    intersects = estimate_fit_local_uncertainty(
        fit,
        params,
        max_grid_step_fraction=1e6,
    )
    assert intersects.frequency_interval_status.item() == "interval_intersects_bound"
    assert intersects.t2_interval_status.item() == "interval_intersects_bound"
    assert np.isnan(intersects.frequency_interval_high)
    assert np.isnan(intersects.t2_interval_low)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"confidence_level": 0}, "confidence_level"),
        ({"confidence_level": True}, "confidence_level"),
        ({"noise_degrees_of_freedom": 0}, "degrees_of_freedom"),
        ({"noise_degrees_of_freedom": np.nan}, "degrees_of_freedom"),
        ({"max_grid_step_fraction": 0}, "max_grid_step_fraction"),
    ],
)
def test_invalid_interval_options_are_rejected(kwargs, match):
    params, fit = _case(frozen=True)
    with pytest.raises((TypeError, ValueError), match=match):
        estimate_fit_local_uncertainty(fit, params, **kwargs)


def test_labelled_batches_preserve_dimensions_and_noise_dof_coordinates():
    """Interval maps follow arbitrary fitted spectrum dimension order."""
    params, fit = _case(frozen=True)
    batch = fit.expand_dims(voxel=["left", "right"]).transpose(
        "metabolite", "voxel", ...
    )
    degrees = xr.DataArray(
        [18.0, 38.0], dims="voxel", coords={"voxel": ["left", "right"]}
    )
    result = estimate_fit_local_uncertainty(
        batch,
        params,
        noise_degrees_of_freedom=degrees,
    )

    assert result.amplitude_abs_interval_low.dims == ("metabolite", "voxel")
    assert result.frequency_interval_status.dims == ("metabolite", "voxel")
    np.testing.assert_array_equal(result.noise_degrees_of_freedom, [18, 38])
    assert result.interval_critical_value.sel(
        voxel="left"
    ) > result.interval_critical_value.sel(voxel="right")
