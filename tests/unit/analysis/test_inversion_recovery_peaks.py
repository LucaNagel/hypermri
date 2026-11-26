"""Exercise the exact legacy spectral model without plots or scanner objects."""

from importlib import import_module

import numpy as np
import pytest

from hypermri.analysis.spectroscopy.inversion_recovery import (
    fit_inversion_recovery_peaks,
)


def _data(dtype=np.complex128):
    # Invented quadrature spectra: two exact isolated bins, no scanner metadata.
    times = np.arange(0, 1001, 50, dtype=float)
    peaks = np.stack(
        [
            np.abs(0.8 - 1.6 * np.exp(-times / 240)) + 0.05,
            2 * (np.abs(0.8 - 1.6 * np.exp(-times / 420)) + 0.05),
        ]
    )
    spectra = np.zeros((len(times), 8))
    spectra[:, 1] = peaks[0]
    spectra[:, 5] = peaks[1]
    raw = np.fft.ifft(np.fft.ifftshift(spectra, axes=-1), axis=-1).astype(dtype).ravel()
    return raw, times, spectra, peaks


def _fit(raw, times, intervals=(1, 2, 5, 6), bounds=None):
    return fit_inversion_recovery_peaks(
        raw,
        n_inversions=len(times),
        n_acquisition=8,
        inversion_times_ms=times,
        intervals=intervals,
        bounds=bounds or ([0.3, -3, 50, 0], [1.2, -0.3, 900, 0.2]),
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("intervals", [(1, 2, 5, 6), (-7, -6, 5, 6), (1, 2, 5, 6, 7)])
def test_direct_recovery_precision_covariance_reconstruction_repeatability_and_inputs(
    dtype, intervals
):
    raw, times, spectra, peaks = _data(dtype)
    raw_before = raw.copy()
    times_before = times.copy()
    first = _fit(raw, times, intervals)
    second = _fit(raw, times, intervals)
    np.testing.assert_allclose(
        [c[2] for c in first.coefficients], [240, 420], rtol=0, atol=0.1
    )
    np.testing.assert_allclose(
        first.normalized_spectra, spectra / spectra.max(), rtol=0, atol=2e-6
    )
    np.testing.assert_allclose(
        first.normalized_integrals,
        peaks / peaks.max(axis=1, keepdims=True),
        rtol=0,
        atol=2e-6,
    )
    assert first.normalized_integrals.dtype == np.float64
    for i, (coeff, cov) in enumerate(
        zip(first.coefficients, first.covariances, strict=True)
    ):
        assert coeff.dtype == cov.dtype == np.float64
        np.testing.assert_allclose(
            coeff, second.coefficients[i], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(cov, second.covariances[i], rtol=1e-12, atol=1e-12)
        expected = (
            np.abs(coeff[0] + coeff[1] * np.exp(-first.fit_times_ms / coeff[2]))
            + coeff[3]
        )
        np.testing.assert_allclose(first.fit_signals[i], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first.inversion_times_ms, times_before)
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_array_equal(times, times_before)
    assert not np.shares_memory(first.inversion_times_ms, times)
    assert first.fit_signals.shape == (2, 1000)


def test_extra_raw_samples_are_ignored_as_in_legacy_workflow():
    raw, times, *_ = _data()
    first = _fit(raw, times)
    second = _fit(np.append(raw, [np.nan + 1j, 999]), times)
    np.testing.assert_allclose(
        first.normalized_spectra, second.normalized_spectra, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        first.coefficients, second.coefficients, rtol=1e-12, atol=1e-12
    )


def test_original_time_order_and_explicit_bounds_reach_solver_once_per_peak(
    monkeypatch,
):
    raw, times, spectra, peaks = _data()
    order = np.arange(len(times))[::-1]
    raw = raw.reshape(-1, 8)[order].ravel()
    times = times[order]
    bounds = (np.array([0.3, -3, 50, 0]), np.array([1.2, -0.3, 900, 0.2]))
    calls = []

    def solver(model, x, y, *, bounds):
        calls.append((x, y.copy(), bounds))
        return np.array([0.8, -1.6, 240, 0.05]), np.eye(4)

    monkeypatch.setattr(
        import_module("hypermri.analysis.spectroscopy.inversion_recovery"),
        "curve_fit",
        solver,
    )
    result = _fit(raw, times, bounds=bounds)
    assert len(calls) == 2
    for i, (x, y, effective_bounds) in enumerate(calls):
        assert x is times and effective_bounds is bounds
        np.testing.assert_allclose(
            y, (peaks[i] / peaks[i].max())[order], rtol=0, atol=1e-12
        )
    np.testing.assert_array_equal(result.inversion_times_ms, times)
    np.testing.assert_allclose(
        result.normalized_spectra, (spectra / spectra.max())[order], rtol=0, atol=1e-12
    )
    assert result.fit_times_ms[0] == 0 and result.fit_times_ms[-1] == 1000
    assert result.intervals == (1, 2, 5, 6)
    for saved, supplied in zip(result.bounds, bounds, strict=True):
        np.testing.assert_array_equal(saved, supplied)
        assert not np.shares_memory(saved, supplied)


@pytest.mark.parametrize("intervals", [(2, 2), (30, 40)])
def test_empty_peak_retains_normalization_warning_and_fit_failure(intervals):
    raw, times, *_ = _data()
    before = raw.copy()
    with pytest.warns(RuntimeWarning, match="invalid value"):
        with pytest.raises(ValueError, match="infs or NaNs"):
            _fit(raw, times, intervals)
    np.testing.assert_array_equal(raw, before)


def test_solver_failure_propagates_without_changing_inputs(monkeypatch):
    raw, times, *_ = _data()
    before = raw.copy()

    def fail(*args, **options):
        raise RuntimeError("invented optimizer failure")

    monkeypatch.setattr(
        import_module("hypermri.analysis.spectroscopy.inversion_recovery"),
        "curve_fit",
        fail,
    )
    with pytest.raises(RuntimeError, match="invented optimizer failure"):
        _fit(raw, times)
    np.testing.assert_array_equal(raw, before)


def test_empty_peak_list_returns_empty_fit_arrays():
    raw, times, *_ = _data()
    result = _fit(raw, times, ())
    assert result.coefficients == result.covariances == []
    assert result.normalized_integrals.shape == (0, len(times))
    assert result.fit_signals.shape == (0, 1000)


def test_invalid_bounds_retain_scipy_error():
    raw, times, *_ = _data()
    with pytest.raises(ValueError, match="lower bound"):
        _fit(raw, times, bounds=([1] * 4, [0] * 4))
