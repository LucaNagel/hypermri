"""Trapezoidal convolution of a sampled DCE input function with a model kernel.

The Tofts, extended Tofts, Sourbron and 2C-IRF models evaluate

``(f * h)(t_n) = integral from t_0 to t_n of f(u) h(t_n - u) du``

on the acquisition times ``t_0 < t_1 < ... < t_{n_time-1}`` with the
trapezoidal rule in ``u``.  The rule uses the actual interval lengths, so it is
valid for non-uniform sampling, and it does not add the half-sample bias of the
rectangle rule ``dt * sum(f_i h_{n-i})``.  The Patlak integral uses the same
rule through ``scipy.integrate.cumulative_trapezoid``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _validated_time_axis(time_s: np.ndarray) -> np.ndarray:
    """Return ``time_s`` as a finite, strictly increasing 1D float64 array."""
    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1:
        raise ValueError(
            f"trapezoidal convolution needs a 1D time axis in s, got shape "
            f"{time.shape}."
        )
    if not np.all(np.isfinite(time)):
        raise ValueError("trapezoidal convolution needs finite sample times in s.")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError(
            "trapezoidal convolution needs strictly increasing sample times in s; "
            "sort the time axis and remove repeated samples."
        )
    return time


def trapezoidal_convolution_weights_s(time_s: np.ndarray) -> np.ndarray:
    """Return the quadrature weights of the trapezoidal convolution.

    Parameters
    ----------
    time_s : numpy.ndarray
        Acquisition times in s, shape ``(n_time,)``, finite and strictly
        increasing.  The spacing may be non-uniform.

    Returns
    -------
    numpy.ndarray
        Weights in s, float64, shape ``(n_time, n_time)``.  Row ``n`` holds
        the trapezoid weights of the integral from ``t_0`` to ``t_n``:
        ``(t_1 - t_0) / 2`` for ``i = 0``, ``(t_{i+1} - t_{i-1}) / 2`` for
        ``0 < i < n``, ``(t_n - t_{n-1}) / 2`` for ``i = n`` and 0 for
        ``i > n``.  Row 0 is zero.

    Raises
    ------
    ValueError
        If ``time_s`` is not 1D, not finite or not strictly increasing.

    Examples
    --------
    >>> trapezoidal_convolution_weights_s(np.array([0.0, 2.0, 6.0]))
    array([[0., 0., 0.],
           [1., 1., 0.],
           [1., 3., 2.]])
    """
    time = _validated_time_axis(time_s)
    n_time = time.size
    intervals_s = np.diff(time)
    row = np.arange(n_time)[:, None]
    column = np.arange(n_time)[None, :]
    # Interval i -> i+1 contributes half its length to sample i (when i < n)
    # and half to sample i+1 (when i+1 <= n).
    interval_after_s = np.append(intervals_s, 0.0)[None, :]
    interval_before_s = np.insert(intervals_s, 0, 0.0)[None, :]
    weights_s = np.where(column < row, interval_after_s, 0.0)
    weights_s += np.where((column >= 1) & (column <= row), interval_before_s, 0.0)
    return 0.5 * weights_s


def trapezoidal_convolution(
    input_values: np.ndarray,
    time_s: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Convolve a sampled input function with a kernel by the trapezoidal rule.

    Evaluates ``c[n] = sum_i w[n, i] * f[i] * h(t_n - t_i)`` with the weights
    of :func:`trapezoidal_convolution_weights_s`.  The input is treated as
    known only at the samples; the kernel is evaluated exactly at the sample
    lags.  The error is second order in the sampling interval when ``f`` and
    ``h`` are smooth on the scale of that interval.

    Parameters
    ----------
    input_values : numpy.ndarray
        Input function at ``time_s``, shape ``(n_time,)``, for example a
        plasma concentration in mM.
    time_s : numpy.ndarray
        Acquisition times in s, shape ``(n_time,)``, finite and strictly
        increasing.  The spacing may be non-uniform.
    kernel : callable
        Impulse response as a function of the lag in s.  It receives a
        float64 array of shape ``(n_time, n_time)`` with ``max(t_n - t_i, 0)``
        and must return values of the same shape.  Entries with ``i > n``
        have lag 0 and weight 0.

    Returns
    -------
    numpy.ndarray
        Convolution at ``time_s``, float64, shape ``(n_time,)``, in the unit
        of ``input_values`` times the unit of the kernel times s.  The first
        value is 0 because the integral starts at ``t_0``.

    Raises
    ------
    ValueError
        If the time axis is invalid, ``input_values`` has a different length
        or the kernel returns an array of the wrong shape.

    Examples
    --------
    The convolution of ``f(t) = t`` with ``h = 1`` is ``t**2 / 2``, which the
    rule reproduces exactly for a linear input on any grid:

    >>> t = np.array([0.0, 1.0, 3.0, 4.0])
    >>> trapezoidal_convolution(t, t, np.ones_like)
    array([0. , 0.5, 4.5, 8. ])
    """
    time = _validated_time_axis(time_s)
    values = np.asarray(input_values, dtype=np.float64)
    if values.shape != time.shape:
        raise ValueError(
            f"trapezoidal convolution needs input values of shape {time.shape} "
            f"matching time_s, got {values.shape}."
        )
    lag_s = np.clip(time[:, None] - time[None, :], 0.0, None)
    kernel_values = np.asarray(kernel(lag_s), dtype=np.float64)
    if kernel_values.shape != lag_s.shape:
        raise ValueError(
            f"trapezoidal convolution kernel must return shape {lag_s.shape}, "
            f"got {kernel_values.shape}."
        )
    weights_s = trapezoidal_convolution_weights_s(time)
    return np.sum(weights_s * kernel_values * values[None, :], axis=1)
