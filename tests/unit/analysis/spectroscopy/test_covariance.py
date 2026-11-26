"""Joint local error contracts using synthetic Lorentzian FIDs only.

Analytical single-peak Fisher moments are independent of the implementation.
Finite-difference checks use a separately written signal model; Monte Carlo
checks use independent Gaussian quadratures with a fixed seed and small noise.
No measured data or identifying acquisition metadata are used.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import def_fit_params
from hypermri.analysis.spectroscopy import (
    calculate_spectroscopy_fit_diagnostics,
    estimate_fit_covariance,
    estimate_fit_local_uncertainty,
    estimate_fit_uncertainty,
    fit_spectrum,
)
from hypermri.analysis.spectroscopy.covariance import _joint_jacobian


def _case(*, amplitudes=(10.0,), frequencies=(40.0,), frozen=False, **options):
    time = np.arange(96) * 0.001
    amplitudes = np.asarray(amplitudes, dtype=complex)
    frequencies = np.asarray(frequencies, dtype=float)
    signal = np.sum(
        amplitudes
        * np.exp(-time[:, None] / 0.025 + 2j * np.pi * time[:, None] * frequencies),
        axis=1,
    )
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
            "metabs": [f"peak{i}" for i in range(amplitudes.size)],
            "metabs_freqs_hz": frequencies,
            "initial_t2s_s": 0.025,
            "metabs_freq_ranges": 0.0 if frozen else 70.0,
            "metabs_t2_ranges": 0.0 if frozen else 0.04,
            "min_t2_s": 0.005,
            "max_t2_s": 0.07,
            "npoints": 3,
            "rep_fitting": 1,
            **options,
        },
    )
    fit = fit_spectrum(data, params, noise=0.1)
    return data, params, fit


def test_joint_covariance_matches_analytical_single_peak_fisher_moments():
    """Check all four variances and nonzero nuisance covariances analytically."""
    data, params, fit = _case()
    result = estimate_fit_covariance(fit, params)
    time = data.sampling_time_axis_s.values
    s0, s1, s2 = [
        np.sum(time**power * np.exp(-2 * time / 0.025)) for power in (0, 1, 2)
    ]
    determinant = s0 * s2 - s1 * s1
    expected = np.zeros((4, 4))
    expected[0, 0] = expected[1, 1] = 0.1**2 * s2 / determinant
    expected[2, 2] = 0.1**2 * s0 / ((2 * np.pi * 10) ** 2 * determinant)
    expected[3, 3] = 0.1**2 * 0.025**4 * s0 / (10**2 * determinant)
    expected[0, 3] = expected[3, 0] = -(0.1**2) * 0.025**2 * s1 / (10 * determinant)
    expected[1, 2] = expected[2, 1] = -(0.1**2) * s1 / (2 * np.pi * 10 * determinant)
    # Float64 SVD should preserve analytical covariance to much better than 1 ppm.
    np.testing.assert_allclose(
        result.parameter_covariance, expected, rtol=1e-10, atol=1e-14
    )
    assert result.covariance_status.item() == "success"
    assert result.jacobian_rank.item() == 4
    assert result.n_free_parameters.item() == 4
    assert result.residual_degrees_of_freedom.item() == 188
    assert result.amplitude_abs_std.item() == pytest.approx(
        np.sqrt(expected[0, 0]), rel=1e-10
    )
    assert result.t2_std.item() == pytest.approx(np.sqrt(expected[3, 3]), rel=1e-10)
    assert result.fwhm_std.item() == pytest.approx(
        result.t2_std.item() / (np.pi * 0.025**2)
    )
    assert result.magnitude_fwhm_std.item() == pytest.approx(
        np.sqrt(3) * result.fwhm_std.item(), rel=1e-12
    )
    assert result.magnitude_fwhm_std.attrs["unit"] == "Hz"
    assert (
        result.amplitude_abs_std.item()
        > 1.35 * fit.amplitude_std_quadrature_conditional.item()
    )


def test_fixed_basis_matches_exact_complex_linear_covariance():
    """Known basis gives exact amplitude variance and zero assumed f/T2 variance."""
    data, params, fit = _case(frozen=True, amplitudes=(10 + 3j,))
    result = estimate_fit_covariance(fit, params)
    variance = 0.1**2 / np.sum(np.exp(-2 * data.sampling_time_axis_s.values / 0.025))
    np.testing.assert_allclose(
        result.parameter_covariance,
        np.diag([variance, variance, 0, 0]),
        rtol=1e-12,
        atol=1e-16,
    )
    np.testing.assert_array_equal(result.parameter_free, [True, True, False, False])
    assert result.frequency_std.item() == 0
    assert result.t2_std.item() == 0
    assert np.isnan(result.parameter_correlation.isel(fit_parameter=2)).all()
    np.testing.assert_allclose(
        result.amplitude_abs_std, fit.amplitude_std_quadrature_conditional, rtol=1e-12
    )


def test_analytic_jacobian_matches_independent_finite_differences():
    """Check units, phase and cross-metabolite column ordering at nonzero origin."""
    time = 0.005 + np.arange(64) * 0.0006
    theta = np.array([2.0, 1.0, 0.5, -0.7, 30.0, 82.0, 0.02, 0.04])

    def model(values):
        signal = sum(
            (values[k] + 1j * values[k + 2])
            * np.exp(2j * np.pi * values[k + 4] * time - time / values[k + 6])
            for k in range(2)
        )
        return np.r_[signal.real, signal.imag]

    finite = []
    for i, step in enumerate([1e-5] * 4 + [1e-4] * 2 + [1e-8] * 2):
        delta = np.zeros(8)
        delta[i] = step
        finite.append((model(theta + delta) - model(theta - delta)) / (2 * step))
    expected = np.column_stack(finite)
    actual = _joint_jacobian(theta[:2] + 1j * theta[2:4], theta[4:6], theta[6:], time)
    # Central finite differences at these physical scales are accurate to < 1 ppm.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_sigma_scaling_repeatability_units_and_no_input_mutation():
    _, params, fit = _case()
    original_fit, original_params = fit.copy(deep=True), params.copy(deep=True)
    first = estimate_fit_covariance(fit, params)
    second = estimate_fit_covariance(fit, params, noise=0.2)
    np.testing.assert_allclose(
        second.parameter_covariance, 4 * first.parameter_covariance
    )
    np.testing.assert_allclose(second.parameter_std, 2 * first.parameter_std)
    np.testing.assert_allclose(
        second.parameter_correlation, first.parameter_correlation
    )
    xr.testing.assert_identical(first, estimate_fit_covariance(fit, params))
    xr.testing.assert_identical(fit, original_fit)
    xr.testing.assert_identical(params, original_params)
    assert first.parameter_covariance.dtype == np.float64
    assert first.frequency_std.attrs["unit"] == "Hz"
    assert first.t2_std.attrs["unit"] == "s"
    assert first.parameter_unit.values.tolist() == ["a.u.", "a.u.", "Hz", "s"]
    assert "uncertainty_valid" not in first


def test_labelled_maps_and_dimension_permutations():
    _, params, fit = _case()
    batch = fit.expand_dims(voxel=["left", "right"], slice_idx=[3, 4])
    sigma = xr.DataArray([0.1, 0.2], dims="voxel", coords={"voxel": ["left", "right"]})
    expected = estimate_fit_covariance(batch, params, noise=sigma)
    transposed = batch.transpose("metabolite", "slice_idx", "voxel", ...)
    actual = estimate_fit_covariance(transposed, params, noise=sigma)
    for name in expected.data_vars:
        xr.testing.assert_allclose(
            actual[name].transpose(*expected[name].dims), expected[name]
        )
    np.testing.assert_allclose(
        expected.parameter_covariance.sel(voxel="right"),
        4 * expected.parameter_covariance.sel(voxel="left"),
    )
    with pytest.raises(ValueError, match="coordinates must exactly match"):
        estimate_fit_covariance(batch, params, noise=sigma.sel(voxel=["right", "left"]))
    local_params = params.expand_dims(voxel=["right", "left"])
    with pytest.raises(ValueError, match="coordinates must exactly match"):
        estimate_fit_covariance(batch, local_params, noise=sigma)


def test_skipped_components_are_nan_and_do_not_poison_active_covariance():
    _, params, fit = _case(
        amplitudes=(10.0, 0.0), frequencies=(40.0, 200.0), skip_fitting=[False, True]
    )
    fit.fit_freqs.loc[{"metabolite": "peak1"}] = np.nan
    fit.fit_t2s.loc[{"metabolite": "peak1"}] = np.nan
    result = estimate_fit_covariance(fit, params)
    assert result.covariance_status.item() == "success"
    assert result.amplitude_abs_status.sel(metabolite="peak1").item() == "skipped"
    assert np.isnan(result.amplitude_abs_std.sel(metabolite="peak1"))
    assert np.isnan(result.parameter_covariance.isel(fit_parameter=[1, 3, 5, 7])).all()
    assert np.isfinite(result.amplitude_abs_std.sel(metabolite="peak0"))
    assert result.n_free_parameters.item() == 4


@pytest.mark.parametrize("frozen", [False, True])
def test_zero_amplitude_never_gets_a_magnitude_error(frozen):
    _, params, fit = _case(amplitudes=(0.0,), frozen=frozen)
    result = estimate_fit_covariance(fit, params)
    assert np.isnan(result.amplitude_abs_std).all()
    if frozen:
        assert result.covariance_status.item() == "success"
        assert result.amplitude_abs_status.item() == "near_zero"
    else:
        assert result.covariance_status.item() == "rank_deficient"
        assert np.isnan(result.parameter_covariance).all()


def test_weak_amplitude_is_explicitly_marked_near_zero():
    _, params, fit = _case(amplitudes=(0.001,), frozen=True)
    result = estimate_fit_covariance(fit, params)
    assert result.amplitude_abs_status.item() == "near_zero"
    assert np.isnan(result.amplitude_abs_std).all()
    assert np.isfinite(result.parameter_std.isel(fit_parameter=slice(0, 2))).all()


@pytest.mark.parametrize(
    "case,expected",
    [
        ("bound", "parameter_bound"),
        ("failed", "invalid_fit"),
        ("nonfinite", "invalid_parameters"),
        ("ill_conditioned", "ill_conditioned"),
        ("skip_all", "no_active_metabolites"),
    ],
)
def test_unusable_fit_statuses_suppress_covariance(case, expected):
    _, params, fit = _case()
    kwargs = {}
    if case == "bound":
        fit.fit_t2s[:] = params.min_t2_s.values
    elif case == "failed":
        fit["fit_status"] = xr.DataArray("fit_failed")
    elif case == "nonfinite":
        fit.fit_amps[:] = np.nan + 1j * np.nan
    elif case == "ill_conditioned":
        kwargs["max_condition_number"] = 1.01
    else:
        params.skip_fitting[:] = True
        fit.fit_skipped[:] = True
    result = estimate_fit_covariance(fit, params, **kwargs)
    assert result.covariance_status.item() == expected
    assert np.isnan(result.parameter_covariance).all()


def test_degenerate_peaks_fail_rank_check_even_with_finite_fitted_values():
    _, params, fit = _case(
        amplitudes=(10.0, 10.0),
        frequencies=(40.0, 40.0),
        frozen=True,
        min_peak_separation_hz=0.0,
    )
    result = estimate_fit_covariance(fit, params)
    assert result.covariance_status.item() == "rank_deficient"
    assert np.isnan(result.parameter_covariance).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("amp_solver", "ridge"),
        ("amp_ridge_lambda", 0.1),
        ("use_sse_mag", True),
        ("snr_threshold", 2.0),
        ("detection_target", "peak0"),
    ],
)
def test_unsupported_estimation_contract_is_rejected(field, value):
    _, params, fit = _case()
    fit.attrs[field] = value
    with pytest.raises(ValueError, match="requires an ordinary complex-RSS"):
        estimate_fit_covariance(fit, params)


def test_sampling_noise_and_mask_mismatches_are_rejected():
    _, params, fit = _case()
    changed = params.assign_attrs(
        sampling_time_axis_s=params.attrs["sampling_time_axis_s"] * 2
    )
    with pytest.raises(ValueError, match="sampling_time_axis_s"):
        estimate_fit_covariance(fit, changed)
    changed = params.copy(deep=True)
    changed.skip_fitting[:] = True
    with pytest.raises(ValueError, match="skip_fitting must match"):
        estimate_fit_covariance(fit, changed)
    for noise in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="finite positive"):
            estimate_fit_covariance(fit, params, noise=noise)
    with pytest.raises(ValueError, match="Supply noise"):
        estimate_fit_covariance(fit.drop_vars("noise_sigma_quadrature"), params)
    reference = xr.Dataset(
        {"noise_sigma_quadrature": 0.1, "noise_likelihood_compatible": False}
    )
    with pytest.raises(ValueError, match="white-noise likelihood"):
        estimate_fit_covariance(fit, params, noise=reference)


def test_dask_evaluation_stays_lazy_and_matches_eager():
    pytest.importorskip("dask.array")
    _, params, fit = _case()
    batch = fit.expand_dims(voxel=["left", "right"])
    expected = estimate_fit_covariance(batch, params)
    lazy = estimate_fit_covariance(batch.chunk({"voxel": 1}), params)
    assert lazy.parameter_covariance.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), expected)


def test_overlapping_peaks_have_joint_cross_covariance():
    """Overlapping resonances must keep cross-metabolite nuisance information."""
    _, params, fit = _case(amplitudes=(10 + 3j, 8 - 1j), frequencies=(40.0, 47.0))
    joint = estimate_fit_covariance(fit, params)
    assert joint.covariance_status.item() == "success"
    assert joint.n_free_parameters.item() == 8
    assert (
        abs(
            joint.parameter_correlation.sel(
                fit_parameter="amplitude_real:peak0",
                fit_parameter_other="amplitude_real:peak1",
            ).item()
        )
        > 0.5
    )
    covariance = joint.parameter_covariance.values
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_active_peak_separation_constraint_is_not_treated_as_an_interior_fit():
    _, params, fit = _case(amplitudes=(10.0, 8.0), frequencies=(40.0, 45.0))
    result = estimate_fit_covariance(fit, params)
    assert result.covariance_status.item() == "parameter_bound"
    assert np.isnan(result.parameter_covariance).all()


def test_equal_bounds_are_fixed_even_with_nonzero_search_width():
    _, params, fit = _case(min_t2_s=0.025, max_t2_s=0.025)
    result = estimate_fit_covariance(fit, params)
    assert result.covariance_status.item() == "success"
    assert result.n_free_parameters.item() == 3
    assert not result.parameter_free.sel(fit_parameter="t2:peak0").item()
    assert result.t2_std.item() == 0


@pytest.mark.parametrize(
    "operation",
    [
        "estimate_fit_covariance",
        "estimate_fit_local_uncertainty",
        "calculate_spectroscopy_fit_diagnostics",
        "estimate_fit_uncertainty",
    ],
)
def test_stale_frequency_bounds_follow_start_and_range(operation):
    """Readers use start -/+ half range, not bounds left behind by an edit."""
    data, params, fit = _case()
    stale = params.copy(deep=True)
    # A stored upper bound at the fitted 40 Hz would mark the line as bounded.
    stale["freq_range_upper_bound_hz"] = fit.fit_freqs.copy()
    calls = {
        "estimate_fit_covariance": lambda p: estimate_fit_covariance(fit, p),
        "estimate_fit_local_uncertainty": lambda p: estimate_fit_local_uncertainty(
            fit, p, noise=0.1
        ),
        "calculate_spectroscopy_fit_diagnostics": (
            lambda p: calculate_spectroscopy_fit_diagnostics(fit, p, 0.1)
        ),
        "estimate_fit_uncertainty": lambda p: estimate_fit_uncertainty(
            data, p, fit, noise=0.1
        ),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        expected = calls[operation](params)
    with pytest.warns(UserWarning, match=rf"^{operation}: .*\['peak0'\]"):
        result = calls[operation](stale)
    xr.testing.assert_identical(result, expected)
    assert stale.freq_range_upper_bound_hz.item() == fit.fit_freqs.item()


def test_independent_high_snr_simulations_match_predicted_parameter_scatter():
    """Check statistical scale with 160 outer draws; no nested bootstrap.

    A=10, T2*=25 ms, 96 samples at 1 ms and sigma=0.5 per quadrature.
    The relative sampling SD of an empirical variance is about 11% for 160
    Gaussian draws; 35% tolerance allows approximately three such SDs plus
    small nonlinear/grid effects. This is a high-SNR regression, not a
    validation of confidence coverage near zero or overlapping resonances.
    """
    data, _, _ = _case()
    rng = np.random.default_rng(20260913)
    noise = 0.5 * (
        rng.normal(size=(160, data.size)) + 1j * rng.normal(size=(160, data.size))
    )
    observed = data.expand_dims(spectrum=np.arange(160)) + xr.DataArray(
        noise,
        dims=("spectrum", "acquisition_idx"),
        coords={"spectrum": np.arange(160), "acquisition_idx": data.acquisition_idx},
    )
    observed.attrs = dict(data.attrs)
    params = def_fit_params(
        observed,
        {
            "metabs": ["peak0"],
            "metabs_freqs_hz": [40.0],
            "initial_t2s_s": [0.025],
            "metabs_freq_ranges": [70.0],
            "metabs_t2_ranges": [0.04],
        },
    )
    fit = fit_spectrum(observed, params, noise=0.5)
    errors = estimate_fit_covariance(fit, params)
    assert (errors.covariance_status == "success").all()
    estimates = np.column_stack(
        [
            fit.fit_amps.values[:, 0].real,
            fit.fit_amps.values[:, 0].imag,
            fit.fit_freqs.values[:, 0],
            fit.fit_t2s.values[:, 0],
        ]
    )
    empirical = np.cov(estimates, rowvar=False, ddof=1)
    predicted = errors.parameter_covariance.mean("spectrum").values
    np.testing.assert_allclose(
        np.diag(empirical), np.diag(predicted), rtol=0.35, atol=0
    )
    assert empirical[0, 3] < 0  # Amplitude and decay-time estimates trade off.
    assert empirical[1, 2] < 0  # Phase and frequency estimates trade off.


@pytest.mark.parametrize("value", [True, 1.0, np.inf, np.nan, "large", [1e8]])
def test_invalid_condition_limit_is_rejected(value):
    _, params, fit = _case()
    with pytest.raises(ValueError, match="max_condition_number"):
        estimate_fit_covariance(fit, params, max_condition_number=value)
