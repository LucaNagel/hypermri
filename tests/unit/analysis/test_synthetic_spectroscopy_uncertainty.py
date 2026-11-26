"""Scientific recovery, noise and uncertainty contracts on synthetic spectra.

Fixtures use seed 20261002, two 256-sample complex peaks at 0.4 ms dwell,
a 2048-sample RF-off reference and 199 bootstrap draws. Tolerances below are
statistical limits for these fixtures, not general measured-data guarantees.
"""

import runpy
from pathlib import Path

import numpy as np
import pytest

_Z_95 = 1.959963984540054


@pytest.fixture(scope="module")
def guide_namespace():
    """Compute the scientific fixture independently of documentation files."""
    path = (
        Path(__file__).resolve().parents[2]
        / "fixtures/synthetic_spectroscopy_uncertainty.py"
    )
    return runpy.run_path(str(path))


def test_minimal_fit_recovers_the_synthetic_truth(guide_namespace):
    """Point estimates lie within four local standard errors of the truth.

    Four standard errors keep the two-sided false-failure probability per
    parameter below 1e-4 under the local normal approximation; a wrong
    frequency sign, unit or component order moves an estimate by many
    standard errors.
    """
    ns = guide_namespace
    fit = ns["fit"]
    errors = ns["local_errors"]
    assert fit.fit_status.item() == "success"
    assert errors.covariance_status.item() == "success"

    amplitude_z = (np.abs(fit.fit_amps.values) - ns["true_amplitudes"]) / (
        errors.amplitude_abs_std.values
    )
    frequency_z = (fit.fit_freqs.values - ns["true_frequencies_hz"]) / (
        errors.frequency_std.values
    )
    t2_z = (fit.fit_t2s.values - ns["true_t2s_s"]) / errors.t2_std.values
    assert np.all(np.abs(amplitude_z) < 4.0)
    assert np.all(np.abs(frequency_z) < 4.0)
    assert np.all(np.abs(t2_z) < 4.0)


def test_noise_adds_diagnostics_without_changing_the_fit(guide_namespace):
    """Supplied noise changes no point estimate and is estimated to 5 %.

    The point estimates come from the same deterministic fit, so identity is
    the contract and exact equality is intended. For 2048 complex samples the
    relative standard deviation of the quadrature sigma estimate is
    1 / (2 * sqrt(2048)) = 1.1 %, so 5 % is about 4.5 standard deviations.
    """
    ns = guide_namespace
    fit = ns["fit"]
    fit_with_noise = ns["fit_with_noise"]
    for name in ("fit_amps", "fit_freqs", "fit_t2s", "fit_rss"):
        np.testing.assert_array_equal(fit_with_noise[name].values, fit[name].values)

    estimated = ns["noise"].item()
    assert abs(estimated / ns["true_noise_sigma"] - 1.0) < 0.05
    assert fit_with_noise.noise_sigma_quadrature.item() == pytest.approx(
        estimated, rel=1e-12
    )
    assert np.all(fit_with_noise.amplitude_snr_conditional.values > 10.0)
    assert ns["noise_from_pipeline"].item() == pytest.approx(estimated, rel=1e-12)


def test_relevance_reports_the_expected_rss_improvement(guide_namespace):
    """Leaving out lactate raises the RSS by the lactate signal energy.

    Expected: RSS_M1 is about 2 N sigma^2 = 1.28 and the lactate energy is
    0.3^2 / (1 - exp(-2 dt / T2*)) = 2.3, giving a relative improvement near
    0.64. The noise in RSS_M1 is about 1 / sqrt(N) = 6 % for N = 256, so 0.15
    absolute covers more than three standard deviations.
    """
    relevance = guide_namespace["relevance"]
    assert relevance.m0_fit_status.sel(target="lactate").item() == "success"
    improvement = relevance.relative_rss_improvement.sel(target="lactate").item()
    assert improvement == pytest.approx(0.64, abs=0.15)


def test_bootstrap_widths_agree_with_local_errors(guide_namespace):
    """Bootstrap and local 95 % widths agree for two strong, isolated peaks.

    The Monte Carlo standard deviation of a 2.5 % or 97.5 % percentile is
    about 2.67 sigma / sqrt(B), so the relative standard deviation of the
    interval width is about 0.96 / sqrt(B) = 6.8 % for B = 199. A tolerance of
    25 % is 3.7 standard deviations and still detects a sqrt(2) error in the
    sigma convention, which would give ratios of 1.41 or 0.71.
    """
    ns = guide_namespace
    bootstrap = ns["bootstrap"]
    errors = ns["local_errors"]
    assert bootstrap.bootstrap_status.item() == "success"
    assert bootstrap.bootstrap_requested.item() == 199
    assert bootstrap.bootstrap_successful.item() == 199
    assert bool(bootstrap.uncertainty_valid.all())

    for interval, std in (
        ("amplitude_abs", "amplitude_abs_std"),
        ("frequency", "frequency_std"),
        ("t2", "t2_std"),
    ):
        width = bootstrap[f"{interval}_ci_high"] - bootstrap[f"{interval}_ci_low"]
        ratio = width.values / (2.0 * _Z_95 * errors[std].values)
        np.testing.assert_allclose(ratio, 1.0, atol=0.25, err_msg=interval)


def test_local_intervals_are_reported_with_success_status(guide_namespace):
    """Strong isolated peaks give regular local intervals around the estimate."""
    ns = guide_namespace
    intervals = ns["local_intervals"]
    fit = ns["fit"]
    for status in (
        "amplitude_abs_interval_status",
        "frequency_interval_status",
        "t2_interval_status",
    ):
        assert list(intervals[status].values) == ["success", "success"]
    assert np.all(intervals.t2_interval_low.values < fit.fit_t2s.values)
    assert np.all(fit.fit_t2s.values < intervals.t2_interval_high.values)
    assert intervals.noise_degrees_of_freedom.item() == 2 * 2048
    assert np.isfinite(ns["frequency_t2_covariance"].item())
