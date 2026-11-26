"""Iterative complex sinusoidal fitting for single-voxel B1 calibration.

The calibration retains the historical centered period search and its complete
residual trajectories. Quantitative B1 map fitting uses its existing separate
period-sweep model. Importing this module initializes no plotting or widgets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["fit_b1_calibration"]

_DEFAULT_PERIOD_SAMPLES = 1000
_DEFAULT_REFINEMENT_ITERATIONS = 10
_DEFAULT_ZOOM_FACTOR = 1.75


def _resolve_b1_calibration_window(center, extent):
    """Validate and clamp the historical search window in sqrt(W)."""
    center = 3.0 if center is None else center
    extent = 3.0 if extent is None else extent
    for name, value in (
        ("initial_guess_cent", center),
        ("initial_guess_extent", extent),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"B1 calibration {name} must be a finite positive real number."
            )
    return max(center, 0.25), max(extent, 1e-6)


def fit_b1_calibration(
    values: ArrayLike,
    refpower_W: ArrayLike,
    *,
    initial_guess_cent: float | None = None,
    initial_guess_extent: float | None = None,
    npoints: int = _DEFAULT_PERIOD_SAMPLES,
    niter: int = _DEFAULT_REFINEMENT_ITERATIONS,
    zoom_factor: float = _DEFAULT_ZOOM_FACTOR,
    stop_index: int | None = None,
    stop_index_extra: int = 0,
) -> tuple[
    complex,
    float,
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fit a complex B1 calibration curve with iterative period refinement.

    The model is ``A * sin(pi / period * sqrt(refpower_W))``. Each iteration
    solves the complex amplitude analytically for every candidate period,
    retains the lowest residual sum of squares seen so far, recenters on that
    period and divides the search half-width by ``zoom_factor``. The historical
    minimum candidate period is 0.25 sqrt(W). This eager, deterministic operation
    preserves both inputs, converts RF powers to float64 and calculates signals
    in complex128. It performs no pulse-shape or pulse-duration adjustment.

    Parameters
    ----------
    values : array_like
        Finite real or complex signal samples with shape ``(n_powers,)`` and at
        least three entries, in arbitrary signal units. Samples are paired with
        RF powers in their supplied order; no sorting or filtering is performed.
    refpower_W : array_like
        Finite, nonnegative RF power samples in watts, shape ``(n_powers,)``.
    initial_guess_cent : float or None, optional
        Positive initial period-search center in sqrt(W). ``None`` selects the
        historical center of 3.0 sqrt(W); centers below 0.25 are clamped to 0.25.
    initial_guess_extent : float or None, optional
        Positive initial search half-width in sqrt(W). ``None`` selects the
        historical half-width of 3.0 sqrt(W). Values below 1e-6 are clamped to
        1e-6, as in the historical calculation.
    npoints : int, optional
        Number of candidate periods in each iteration; at least two.
    niter : int, optional
        Number of period-search iterations; at least one.
    zoom_factor : float, optional
        Finite factor greater than one by which the search half-width shrinks.
    stop_index : int or None, optional
        Exclusive end of the fitted segment, between three and ``n_powers``.
        ``None`` includes every sample. The fitted curve is zero after this
        index; residuals use only the fitted segment.
    stop_index_extra : int, optional
        Retained historical compatibility setting. The original single-voxel
        calculation accepted this value without changing the fitted segment;
        this function preserves that behavior. Set ``stop_index`` explicitly to
        select a segment. This setting does not infer a flip-angle cutoff.

    Returns
    -------
    amplitude : complex
        Best complex amplitude in the input signal units.
    period : float
        Best period parameter in sqrt(W), corresponding to the model's
        180-degree crossing. An all-zero fitted segment returns NaN.
    fit_data : numpy.ndarray
        Complex128 fitted samples, shape ``(n_powers,)``, zero outside the fitted
        segment. An all-zero fitted segment returns an all-zero array.
    local_r2 : numpy.ndarray
        Float64 residual sums of squared complex magnitudes, in squared signal
        units, shape ``(niter, npoints)``. The retained historical name denotes
        RSS, not the coefficient of determination. All values are NaN for an
        all-zero fitted segment.
    local_per : numpy.ndarray
        Float64 candidate periods in sqrt(W), shape ``(niter, npoints)``, in
        iteration and candidate order. All values are NaN for an all-zero fitted
        segment.

    Raises
    ------
    ValueError
        If sample arrays are not matching one-dimensional finite arrays with at
        least three entries, powers are negative, or a search option is outside
        its documented range.

    Examples
    --------
    >>> powers = np.linspace(0.0, 16.0, 17)
    >>> signal = (2.0 + 0.5j) * np.sin(np.pi / 4.0 * np.sqrt(powers))
    >>> amplitude, period, curve, rss, periods = fit_b1_calibration(signal, powers)
    >>> bool(abs(period - 4.0) < 1e-4)
    True
    """
    signal = np.asarray(values)
    power = np.asarray(refpower_W, dtype=float)
    if signal.ndim != 1 or power.ndim != 1 or signal.size != power.size:
        raise ValueError(
            "B1 calibration requires matching one-dimensional values and "
            "refpower_W arrays."
        )
    if signal.size < 3:
        raise ValueError("B1 calibration requires at least three RF-power samples.")
    if not np.isfinite(signal).all():
        raise ValueError("B1 calibration values must contain only finite samples.")
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError(
            "B1 calibration refpower_W must contain finite, nonnegative powers in watts."
        )
    for name, value, minimum in (("npoints", npoints, 2), ("niter", niter, 1)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"B1 calibration {name} must be an integer >= {minimum}.")
    percent, perextent = _resolve_b1_calibration_window(
        initial_guess_cent, initial_guess_extent
    )
    if not np.isscalar(zoom_factor) or not np.isfinite(zoom_factor) or zoom_factor <= 0:
        raise ValueError("B1 calibration zoom_factor must be finite and positive.")
    if zoom_factor <= 1:
        raise ValueError("B1 calibration zoom_factor must be greater than one.")
    if stop_index is None:
        stop_index = signal.size
    if (
        isinstance(stop_index, (bool, np.bool_))
        or not isinstance(stop_index, (int, np.integer))
        or not 3 <= stop_index <= signal.size
    ):
        raise ValueError(
            "B1 calibration stop_index must be an integer between three and "
            "the number of RF-power samples."
        )
    if isinstance(stop_index_extra, (bool, np.bool_)) or not isinstance(
        stop_index_extra, (int, np.integer)
    ):
        raise ValueError("B1 calibration stop_index_extra must be an integer.")

    voxel_data = signal[:stop_index]
    sqrt_refpower = np.sqrt(power)[:stop_index]
    local_r2 = np.full((niter, npoints), np.nan, dtype=float)
    local_per = np.full((niter, npoints), np.nan, dtype=float)
    fit_data = np.zeros(signal.size, dtype=complex)
    if np.nansum(np.abs(voxel_data)) == 0:
        return 0.0 + 0.0j, np.nan, fit_data, local_r2, local_per

    best_period = np.nan
    best_amp = 0.0 + 0.0j
    best_rss = np.inf
    best_fit_segment = np.zeros(stop_index, dtype=complex)
    b = voxel_data.astype(complex)
    for iter_idx in range(niter):
        lower = max(0.1, percent - perextent)
        upper = percent + perextent
        per = np.linspace(lower, upper, npoints, dtype=float)
        per = np.clip(per, 0.25, None)
        sin_matrix = np.sin(np.pi / per[np.newaxis, :] * sqrt_refpower[:, np.newaxis])
        numerator = np.sum(sin_matrix.conj() * b[:, np.newaxis], axis=0)
        denom = np.sum(np.abs(sin_matrix) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            amps = np.divide(
                numerator,
                denom,
                out=np.zeros_like(numerator, dtype=complex),
                where=denom > 0,
            )
        residuals = np.sum(
            np.abs(b[:, np.newaxis] - sin_matrix * amps[np.newaxis, :]) ** 2,
            axis=0,
        )
        local_r2[iter_idx, :] = residuals
        local_per[iter_idx, :] = per
        if np.all(~np.isfinite(residuals)):
            break
        r2_min_ind = int(np.nanargmin(residuals))
        rss_candidate = residuals[r2_min_ind]
        if rss_candidate < best_rss:
            best_rss = rss_candidate
            best_period = per[r2_min_ind]
            best_amp = amps[r2_min_ind]
            best_fit_segment = sin_matrix[:, r2_min_ind] * best_amp
        perextent /= zoom_factor
        percent = best_period if np.isfinite(best_period) else percent

    fit_data[:stop_index] = best_fit_segment
    period = float(best_period) if np.isfinite(best_period) else np.nan
    return complex(best_amp), period, fit_data, local_r2, local_per
