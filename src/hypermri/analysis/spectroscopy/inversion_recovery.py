"""Fit legacy spectral inversion-recovery integrals without interactive controls.

Notes
-----
This domain workflow preserves the four-parameter absolute exponential used by
IVRecovery multi-peak controls. It differs from quantitative MRI's T1 grid
model and therefore does not delegate to that scientifically different solver.
Calls are eager and deterministic; imports construct no widgets or plots and
retain no signal, reconstruction or fit cache. Acquired inputs are not mutated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["InversionRecoveryPeakFit", "fit_inversion_recovery_peaks"]


@dataclass(frozen=True)
class InversionRecoveryPeakFit:
    """Hold the calculated signals and fits of the legacy multi-peak workflow.

    Parameters
    ----------
    inversion_times_ms : numpy.ndarray
        Original inversion times of shape (n_inversions,), in milliseconds and
        supplied order. This is an independent copy, without coordinate sorting.
    normalized_spectra : numpy.ndarray
        Shifted Fourier magnitudes of shape (n_inversions, n_acquisition), divided
        by the maximum magnitude over all inversions and spectral bins. Values
        are dimensionless and retain NumPy FFT/magnitude precision.
    normalized_integrals : numpy.ndarray
        Per-peak absolute spectral sums divided by each peak's maximum across
        inversions. Shape (n_peaks, n_inversions), float64, dimensionless. Empty
        intervals retain the legacy nonfinite normalization and fitting failure.
    coefficients : list of numpy.ndarray
        One SciPy float64 four-parameter result per peak in (M0, A, T1, offset)
        order. T1 is in ms; the other parameters follow dimensionless normalized
        signal. Bounds and optimizer defaults are those supplied to SciPy.
    covariances : list of numpy.ndarray
        One float64 (4, 4) SciPy covariance matrix per peak in parameter order.
        Mixed units are the products of the corresponding parameter units;
        these are covariances, not standard errors. Singular fits can yield inf.
    fit_times_ms : numpy.ndarray
        Float64 millisecond display axis of shape (1000,), from zero to the
        maximum supplied inversion time, retaining the legacy sampling rule.
    fit_signals : numpy.ndarray
        Reconstructed dimensionless float64 curves of shape (n_peaks, 1000),
        calculated with each peak's four fitted coefficients.
    intervals : tuple of int
        Original supplied paired spectral indices, including any ignored trailing
        index, retained as scientific selection provenance.
    bounds : tuple of numpy.ndarray
        Independent copies of supplied lower/upper bounds in parameter order,
        retaining scalar versus vector shape and the documented parameter units.

    Notes
    -----
    Fields cannot be rebound, but the owned arrays/lists remain writable. No
    array shares acquired input storage. No status or uncertainty approximation
    is invented: optimizer failures propagate and covariance comes from SciPy.
    """

    inversion_times_ms: np.ndarray
    normalized_spectra: np.ndarray
    normalized_integrals: np.ndarray
    coefficients: list[np.ndarray]
    covariances: list[np.ndarray]
    fit_times_ms: np.ndarray
    fit_signals: np.ndarray
    intervals: tuple[int, ...]
    bounds: tuple[np.ndarray, np.ndarray]


def _absolute_inversion_recovery(time_ms, M0, A, T1, offset):
    """Evaluate the existing absolute exponential in millisecond time units."""
    return np.abs(M0 + A * np.exp(-time_ms / T1)) + offset


def fit_inversion_recovery_peaks(
    raw_job: np.ndarray,
    *,
    n_inversions: int,
    n_acquisition: int,
    inversion_times_ms: np.ndarray,
    intervals: Sequence[int],
    bounds: tuple[Sequence[float], Sequence[float]],
) -> InversionRecoveryPeakFit:
    """Fit raw spectral peak integrals to the legacy absolute exponential.

    Contiguous FIDs are Fourier transformed separately, shifted and converted to
    magnitudes. Paired spectral intervals give absolute sums, normalized to each
    peak's maximum, before SciPy fits ``abs(M0 + A*exp(-t/T1)) + offset``. This
    preserves the injected IVRecovery multi-peak workflow without preparing ppm
    metadata, reading a handler, plotting or constructing interactive controls.

    Parameters
    ----------
    raw_job : numpy.ndarray
        One-dimensional acquired real/complex samples containing n_inversions
        contiguous FIDs of n_acquisition samples each. Extra trailing samples
        are ignored as before. Input units cancel during normalization; FFT and
        magnitude calculations retain NumPy's storage-dependent precision policy.
    n_inversions : int
        Number of FIDs taken from the beginning of raw_job, normally scanner NoIT.
    n_acquisition : int
        Number of samples per FID; also the length of the shifted spectral axis.
    inversion_times_ms : numpy.ndarray
        Times in milliseconds, one per inversion, in original supplied order.
        No conversion, sorting, interpolation or metadata inference occurs.
    intervals : sequence of int
        Flat paired start/stop indices into shifted spectra, one pair per peak.
        Stops are exclusive. Negative indices follow NumPy slicing; a trailing
        unpaired index is ignored as in the legacy method. No ppm calibration.
    bounds : tuple of sequence of float
        Lower and upper SciPy bounds for (M0, A, T1, offset), each four entries
        (or a scalar accepted by SciPy). T1 bounds use ms; others follow the
        normalized dimensionless signal. Passed through unchanged; no extra
        initial guess, weighting, clipping or optimizer options are imposed.

    Returns
    -------
    InversionRecoveryPeakFit
        Normalized spectra/integrals, fitted parameters, covariance, original
        timing and reconstructed 1000-point curves with the documented shapes
        and units. Empty intervals/invalid fits do not return partial results.

    Raises
    ------
    ValueError
        If sample shapes, times, intervals or bounds prevent NumPy/SciPy
        calculation, including nonfinite normalized integrals for empty peaks.
    RuntimeError
        If SciPy's existing nonlinear optimizer cannot converge.
    TypeError
        If spectral interval endpoints cannot be used as NumPy slice indices.

    Notes
    -----
    Deterministic, eager and non-mutating. Normalization/solver warnings and
    exceptions are preserved. Covariance is the solver estimate; there is no
    separate fit-quality policy or invented statistical uncertainty. The
    quantitative T1 grid model has a different scientific contract and is not
    a replacement for this legacy four-parameter model.

    Examples
    --------
    With supplied scanner times and paired spectral bins::

        result = fit_inversion_recovery_peaks(
            raw_job, n_inversions=len(times_ms), n_acquisition=1024,
            inversion_times_ms=times_ms, intervals=[100, 120, 180, 200],
            bounds=([0.3, -3, 10, 0], [1.2, -0.3, 1000, 0.2]),
        )
        t1_ms = [parameters[2] for parameters in result.coefficients]
    """
    spectra, integrals = _prepare_inversion_recovery_peaks(
        raw_job, n_inversions, n_acquisition, intervals
    )
    normalized_spectra = _normalize_inversion_recovery_signal(spectra)
    coefficients = []
    covariances = []
    normalized_integrals = np.empty_like(integrals)
    fit_signals = np.empty((len(integrals), 1000), dtype=float)
    fit_times_ms = np.linspace(0, np.max(inversion_times_ms), 1000)
    for idx, integral in enumerate(integrals):
        ydata = _normalize_inversion_recovery_signal(integral)
        coeff, covariance, fit_times_ms, curve = _fit_inversion_recovery_peak(
            ydata, inversion_times_ms, bounds
        )
        normalized_integrals[idx] = ydata
        fit_signals[idx] = curve
        coefficients.append(coeff)
        covariances.append(covariance)

    return InversionRecoveryPeakFit(
        inversion_times_ms=np.asarray(inversion_times_ms).copy(),
        normalized_spectra=normalized_spectra,
        normalized_integrals=normalized_integrals,
        coefficients=coefficients,
        covariances=covariances,
        fit_times_ms=fit_times_ms,
        fit_signals=fit_signals,
        intervals=tuple(intervals),
        bounds=(np.array(bounds[0], copy=True), np.array(bounds[1], copy=True)),
    )


def _prepare_inversion_recovery_peaks(raw_job, n_inversions, n_acquisition, intervals):
    """Transform contiguous FIDs and integrate the paired shifted spectral bins."""
    n_peaks = int(len(intervals) / 2)
    spectra = []
    integrals = np.empty([n_peaks, n_inversions], dtype=float)
    for n in range(n_inversions):
        fid = raw_job[n * n_acquisition : n_acquisition + n * n_acquisition]
        spectra.append(np.abs(np.fft.fftshift(np.fft.fft(fid))))
    spectra = np.array(spectra)
    for n in range(n_inversions):
        for i in range(n_peaks):
            integrals[i, n] = np.abs(
                np.sum(spectra[n][intervals[i * 2] : intervals[i * 2 + 1]])
            )
    return spectra, integrals


def _normalize_inversion_recovery_signal(signal):
    """Normalize the supplied spectrum grid or one integral by its maximum."""
    return signal / np.max(signal)


def _fit_inversion_recovery_peak(normalized_integral, inversion_times_ms, bounds):
    """Fit one normalized peak and reconstruct the legacy millisecond curve."""
    coeff, covariance = curve_fit(
        _absolute_inversion_recovery,
        inversion_times_ms,
        normalized_integral,
        bounds=bounds,
    )
    fit_times_ms = np.linspace(0, np.max(inversion_times_ms), 1000)
    fit_signal = _absolute_inversion_recovery(fit_times_ms, *coeff)
    return coeff, covariance, fit_times_ms, fit_signal
