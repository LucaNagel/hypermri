"""
Mathematical calculations and curve fitting utilities.

This module contains basic mathematical functions extracted from utils_general.py
as part of the Phase 2 refactoring to create focused utility modules.
"""

import operator as _operator

import numpy as np
import xarray as xr

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def get_histogram(data=None, number_of_bins: str = 10):
    """Compute histogram bin centers and counts.

    Parameters
    ----------
    data : array_like
        Values to bin.
    number_of_bins : int, optional
        Number of histogram bins. Default is 10.

    Returns
    -------
    x_data : numpy.ndarray
        Bin centers.
    y_data : numpy.ndarray
        Sample count per bin.
    bin_size : float
        Width of each bin.

    Notes
    -----
    NaN values in ``data`` are ignored when accumulating the histogram counts.
    """
    if np.isnan(data).any():
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
    else:
        min_val = np.min(data)
        max_val = np.max(data)
    bin_size = (max_val - min_val) / number_of_bins
    x_data = np.linspace(min_val, max_val, number_of_bins)
    y_data = np.zeros_like(x_data)

    if len(data.shape) > 1:
        for i in np.arange(0, data.shape[0], 1):
            for j in np.arange(0, data.shape[1], 1):
                if np.isnan(data[i, j]):
                    # handling nan values so that they don't contribute to the
                    # histogram
                    pass
                else:
                    idx = np.abs(x_data - data[i, j]).argmin()
                    y_data[idx] += 1
    else:
        for j in np.arange(0, data.shape[0], 1):
            idx = (np.abs(x_data - data[j])).argmin()
            y_data[idx] += 1
    return x_data, y_data, bin_size


def gaussian(x, amplitude, mean, sigma):
    """Evaluate a Gaussian curve.

    Parameters
    ----------
    x : array_like
        Independent variable.
    amplitude : float
        Peak amplitude of the Gaussian.
    mean : float
        Center position of the Gaussian.
    sigma : float
        Standard deviation (spread) of the Gaussian.

    Returns
    -------
    numpy.ndarray
        Gaussian values evaluated at ``x``.
    """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def compute_weighted_mean_and_std(data=None, weights=None):
    """Compute weighted statistics for a sample.

    Parameters
    ----------
    data : array_like
        Sample values.
    weights : array_like
        Non-negative weights assigned to ``data``.

    Returns
    -------
    tuple of float
        ``(weighted_mean, weighted_std)`` giving the weighted mean and
        standard deviation.
    """
    data = np.asarray(data)
    weights = np.asarray(weights)

    # Compute weighted mean
    weighted_mean = np.average(data, weights=weights)

    # Compute weighted variance
    weighted_var = np.average((data - weighted_mean) ** 2, weights=weights)
    weighted_std = np.sqrt(weighted_var)

    return weighted_mean, weighted_std


def find_max_and_index(data, *, magnitude=False):
    """Locate the maximum value and its indices.

    Parameters
    ----------
    data : array_like or xarray.DataArray
        Input array, optionally multi-dimensional.
    magnitude : bool, optional
        If True, find the maximum absolute value also for real input. Default
        False preserves signed real maxima. Values with a nonzero imaginary part
        use magnitude, retaining the existing complex-input behavior.

    Returns
    -------
    tuple
        ``(max_value, max_indices)`` where ``max_value`` is the maximum element
        and ``max_indices`` is a tuple of the indices at which it occurs.

    Examples
    --------
    >>> test_array = np.random.normal(loc=0, scale=1, size=(50,50))
    >>> highest_val, indices = find_max_and_index(data=test_array)
    >>> print(highest_val, indices, test_array[indices[0], indices[1]])
    """
    if magnitude:
        data = np.abs(data)
    if isinstance(data, xr.DataArray):
        if np.iscomplex(data).any():
            data = np.abs(data)
            import warnings

            warnings.warn(
                "data is complex, returned maxima will be calculated from magnitude of data."
            )
        max_value = data.max().item()
        # Get flat index and unravel using shape
        flat_idx = np.argmax(data.values).item()
        max_index = np.unravel_index(flat_idx, data.shape)
    else:
        if np.iscomplex(data).any():
            data = np.abs(data)
            import warnings

            warnings.warn(
                "data is complex, returned maxima will be calculated from magnitude of data."
            )
        max_value = np.max(data)
        max_index = np.unravel_index(np.argmax(data), data.shape)

    return max_value, max_index


def calc_sampling_time_axis(
    dwell_time_s: float | None = None,
    npoints: int | None = 0,
    scan=None,
    bw_hz: float | None = None,
) -> np.ndarray:
    """Adapt legacy sampling-axis inputs to the canonical temporal calculation.

    Parameters
    ----------
    dwell_time_s : float, optional
        Finite positive interval in seconds. Mutually exclusive with bw_hz.
        When both timing inputs are omitted, retain the legacy 1 ms default;
        explicitly supplied zero or negative values are invalid.
    npoints : int, optional
        Nonnegative sample count. Zero without a context gives an empty axis;
        zero or None with a context uses n_acquisition minus cut_off.
    scan : object, optional
        Metadata context supplying n_acquisition and optional cut_off. Its
        precomputed time axis is not used by this legacy entry point.
    bw_hz : float, optional
        Finite positive spectral bandwidth in Hz, used instead of dwell_time_s.
        Specifying both timing inputs is an error.

    Returns
    -------
    numpy.ndarray
        Float64 seconds from zero, shape (npoints,). Inputs are preserved;
        calculation is eager, deterministic and uncached. The canonical owner
        is hypermri.utils.time.temporal.calc_sampling_time_axis.

    Raises
    ------
    ValueError
        If the resolved sample count is negative, explicit timing inputs conflict,
        or timing is nonpositive/nonfinite.
    TypeError
        If the sample count is not an integer or timing is not a real scalar.
    AttributeError
        If context-derived sample count is requested without n_acquisition.

    Examples
    --------
    >>> calc_sampling_time_axis(npoints=3).tolist()
    [0.0, 0.001, 0.002]
    """
    if dwell_time_s is None and bw_hz is None:
        dwell_time_s = 1e-3

    if scan is not None:
        if npoints == 0 or npoints is None:
            npoints = scan.n_acquisition - getattr(scan, "cut_off", 0)

    if isinstance(npoints, (bool, np.bool_, np.ndarray)):
        raise TypeError("calc_sampling_time_axis requires npoints to be an integer.")
    npoints = _operator.index(npoints)
    if npoints < 0:
        raise ValueError("calc_sampling_time_axis requires a nonnegative npoints.")

    from hypermri.utils.time.temporal import calc_sampling_time_axis as sampling_axis

    return sampling_axis(dwell_time_s=dwell_time_s, no_points=npoints, bw_hz=bw_hz)


def polynomial_fit(x, y, degree=1, weights=None):
    """Fit a polynomial to data with optional weighting.

    Parameters
    ----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.
    degree : int, optional
        Degree of the polynomial to fit. Default is 1.
    weights : array_like, optional
        Relative weights applied to each observation. Default is ``None``.

    Returns
    -------
    tuple
        ``(coefficients, polynomial)`` where ``coefficients`` are the fitted
        polynomial coefficients and ``polynomial`` is a ``numpy.poly1d``
        callable representing the fit.
    """
    coeffs = np.polyfit(x, y, degree, w=weights)
    poly_func = np.poly1d(coeffs)
    return coeffs, poly_func


def moving_average(data, window_size):
    """Compute a simple moving average.

    Parameters
    ----------
    data : array_like
        Input data sequence.
    window_size : int
        Length of the averaging window.

    Returns
    -------
    numpy.ndarray
        Moving average values. The output length is ``len(data) - window_size + 1``.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def apply_gaussian_smoothing(data=None, sigma: float = 1.0):
    """Apply Gaussian smoothing to data.

    Parameters
    ----------
    data : array_like
        Input data to smooth.
    sigma : float, optional
        Standard deviation of the Gaussian kernel. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Smoothed data array.
    """
    from scipy import ndimage

    return ndimage.gaussian_filter(data, sigma=sigma)


def calculate_fwhm(x, y, method="interpolation"):
    """Compute the full width at half maximum of a peak.

    Parameters
    ----------
    x : array_like
        Sample positions.
    y : array_like
        Sampled peak values.
    method : {"interpolation", "fit"}, optional
        Algorithm used to determine the FWHM. Default is ``"interpolation"``.

    Returns
    -------
    float
        Estimated full width at half maximum.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.
    """
    # Find peak
    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    half_max = peak_val / 2

    if method == "interpolation":
        # Find points where y crosses half maximum
        indices = np.where(y >= half_max)[0]
        if len(indices) < 2:
            return np.nan

        left_idx = indices[0]
        right_idx = indices[-1]

        # Interpolate for more precise FWHM
        if left_idx > 0:
            left_x = np.interp(
                half_max, [y[left_idx - 1], y[left_idx]], [x[left_idx - 1], x[left_idx]]
            )
        else:
            left_x = x[left_idx]

        if right_idx < len(x) - 1:
            right_x = np.interp(
                half_max,
                [y[right_idx], y[right_idx + 1]],
                [x[right_idx], x[right_idx + 1]],
            )
        else:
            right_x = x[right_idx]

        return right_x - left_x

    elif method == "fit":
        # Fit Gaussian and extract FWHM
        try:
            from scipy.optimize import curve_fit

            # Initial guess
            amplitude_guess = peak_val
            mean_guess = x[peak_idx]
            sigma_guess = (x[-1] - x[0]) / 10  # Rough estimate

            popt, _ = curve_fit(
                gaussian, x, y, p0=[amplitude_guess, mean_guess, sigma_guess]
            )

            # FWHM = 2.355 * sigma for Gaussian
            return 2.355 * abs(popt[2])

        except Exception as e:
            logger.warning(f"Gaussian fit failed for FWHM calculation: {e}")
            return np.nan

    else:
        raise ValueError("Method must be 'interpolation' or 'fit'")


def part_func(data=None, signal_part="Magnitude"):
    """Extract a chosen component from a complex-valued signal.

    Parameters
    ----------
    data : array_like, optional
        Complex-valued samples from which to extract a component.
    signal_part : {"Real", "Imaginary", "Phase", "Magnitude"}, optional
        Component to extract. Any other value defaults to ``"Magnitude"``.

    Returns
    -------
    numpy.ndarray
        Requested component of ``data``. Returns ``None`` when
        ``data`` is ``None``.

    Notes
    -----
    The function silently falls back to the magnitude if an unsupported
    ``signal_part`` is provided.
    """
    if data is None:
        return None

    match signal_part:
        case "Real":
            return np.real(data)
        case "Imaginary":
            return np.imag(data)
        case "Phase":
            return np.angle(data)
        case "Magnitude":
            return np.abs(data)
        case _:
            return np.abs(data)
