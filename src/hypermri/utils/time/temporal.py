"""
Temporal processing and time axis calculations.

This module contains all time-related functions extracted from utils_general.py
as part of the Phase 2 refactoring to create focused utility modules.
"""

import operator as _operator
from numbers import Real as _Real
from typing import Any, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def get_dwell_time_s(
    scan: Any = None,
    bw_hz: float = None,
    sampling_time_axis_s: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the sampling duration per point in [s] when a spectrum/FID was acquired.

    Parameters
    ----------
        scan: scan, optional
            THe data object containing information about the sampling
        bw_hz: float
            spectral bandwidth in [Hz]
        sampling_time_axis_s: ndarray
            sampling time axis in [s]

    Returns
    -------
    sampling duration per point

    """
    dwell_time_s = 0
    if scan is None:
        if bw_hz is not None and bw_hz > 0:
            dwell_time_s = 1 / bw_hz
        elif sampling_time_axis_s is not None and isinstance(
            sampling_time_axis_s, np.ndarray
        ):
            dwell_time_s = sampling_time_axis_s[1] - sampling_time_axis_s[0]
        else:
            logger.critical("Pass either scan or bw_hz!")
    else:
        # CSI:
        dwell_time_s = scan.get("dwell_time_s", 0)

    return dwell_time_s


def _sampling_scalar(value, name, *, positive=False):
    """Validate a finite real scalar used by the sampling-axis calculation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, _Real):
        raise TypeError(f"calc_sampling_time_axis requires {name} to be a real scalar.")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(
            f"calc_sampling_time_axis requires {name} to be {requirement}."
        )
    return value


def _sampling_count(value):
    """Validate a nonnegative integer sample count without rounding."""
    if isinstance(value, (bool, np.bool_, np.ndarray)):
        raise TypeError("calc_sampling_time_axis requires npoints to be an integer.")
    count = _operator.index(value)
    if count < 0:
        raise ValueError("calc_sampling_time_axis requires nonnegative npoints.")
    return count


def calc_sampling_time_axis(
    scan: Any = None,
    dwell_time_s: float | None = None,
    npoints: int | None = None,
    bw_hz: float | None = None,
    no_points: int | None = None,
    bandwidth_hz: float | None = None,
    start_with_0: bool = True,
    t_echo_s: float | None = None,
) -> np.ndarray:
    """Construct a sampling axis from one explicit interval or valid metadata.

    Supply either a positive dwell time in seconds or a positive bandwidth in
    Hz. Their simultaneous explicit specification is an error, even when they
    are reciprocals or a cached axis exists. No timing default is invented.

    Parameters
    ----------
    scan : object, optional
        Metadata context with an optional one-dimensional, finite, strictly
        increasing ``sampling_time_axis_s``. A nonempty cached axis wins when
        its length matches the requested count, or no count is supplied. Its
        offsets and irregular spacing are preserved. Otherwise, missing count
        is ``n_acquisition - cut_off`` and missing explicit timing is the
        positive ``dwell_time_s`` attribute or mapping-style ``get`` value.
    dwell_time_s : float, optional
        Finite positive sampling interval in seconds. ``None`` means omitted;
        zero, negative and nonfinite values are invalid.
    npoints : int, optional
        Nonnegative sample count. Zero produces an empty axis. Without metadata,
        an omitted count defaults to 512; with metadata it is resolved above.
    bw_hz : float, optional
        Finite positive bandwidth in Hz, giving an interval of ``1 / bw_hz``.
        Mutually exclusive with ``dwell_time_s`` and ``bandwidth_hz``.
    no_points : int, optional
        Alias of ``npoints``; specifying both count names is an error.
    bandwidth_hz : float, optional
        Alias of ``bw_hz``; specifying both bandwidth names is an error.
    start_with_0 : bool, optional
        For a context-free axis, start at zero if True or at one dwell interval
        if False. Metadata-derived axes retain their historical zero origin;
        cached axes retain their own origin.
    t_echo_s : float, optional
        Finite offset in seconds added to context-free axes. Negative offsets
        are supported for a time origin preceding the first sample. Metadata
        and cached axes retain their existing origin and ignore this offset.

    Returns
    -------
    numpy.ndarray
        Seconds with shape ``(npoints,)``. Generated axes have float64 dtype;
        cached axes retain their real numeric dtype and may share input storage.
        The eager calculation is deterministic, does not reconstruct data or
        mutate inputs, and does not update the metadata object's cache.

    Raises
    ------
    TypeError
        If timing or offset inputs are not real scalars, the count is not an
        integer, or ``start_with_0`` is not boolean.
    ValueError
        If explicit timing sources or aliases conflict, timing is unavailable,
        a scalar is nonfinite or timing is nonpositive, the count is negative
        or unavailable, or a selected cached axis is not a valid sampling axis.

    Notes
    -----
    This canonical API rejects the former zero-dwell placeholder. The legacy
    ``utils.mathematics.calculations.calc_sampling_time_axis`` adapter alone
    retains a 1 ms default when neither timing source is supplied.

    Examples
    --------
    >>> calc_sampling_time_axis(npoints=3, bw_hz=1000).tolist()
    [0.0, 0.001, 0.002]
    >>> calc_sampling_time_axis(no_points=3, dwell_time_s=0.25,
    ...                         start_with_0=False, t_echo_s=1.0).tolist()
    [1.25, 1.5, 1.75]
    """
    if npoints is not None and no_points is not None:
        raise ValueError(
            "calc_sampling_time_axis: supply only one of npoints and no_points."
        )
    if bw_hz is not None and bandwidth_hz is not None:
        raise ValueError(
            "calc_sampling_time_axis: supply only one of bw_hz and bandwidth_hz."
        )
    bandwidth = bandwidth_hz if bandwidth_hz is not None else bw_hz
    if dwell_time_s is not None and bandwidth is not None:
        raise ValueError(
            "calc_sampling_time_axis: supply either dwell_time_s or bandwidth, not both."
        )
    if dwell_time_s is not None:
        dwell_time_s = _sampling_scalar(dwell_time_s, "dwell_time_s", positive=True)
    if bandwidth is not None:
        bandwidth = _sampling_scalar(bandwidth, "bandwidth_hz", positive=True)
    if not isinstance(start_with_0, (bool, np.bool_)):
        raise TypeError("calc_sampling_time_axis requires boolean start_with_0.")
    if t_echo_s is not None:
        t_echo_s = _sampling_scalar(t_echo_s, "t_echo_s")

    count = no_points if no_points is not None else npoints
    if count is not None:
        count = _sampling_count(count)

    if scan is not None:
        cached = getattr(scan, "sampling_time_axis_s", None)
        if cached is not None:
            cached = np.asarray(cached)
            if cached.size and (
                count is None or cached.ndim != 1 or cached.size == count
            ):
                if (
                    cached.ndim != 1
                    or cached.dtype.kind not in "fiu"
                    or not np.all(np.isfinite(cached))
                    or np.any(cached[1:] <= cached[:-1])
                ):
                    raise ValueError(
                        "calc_sampling_time_axis requires a finite, real, "
                        "one-dimensional, strictly increasing sampling_time_axis_s."
                    )
                return cached
        if count is None:
            try:
                count = _sampling_count(scan.n_acquisition - scan.cut_off)
            except AttributeError as exc:
                raise ValueError(
                    "calc_sampling_time_axis cannot determine npoints from scan; "
                    "provide npoints or n_acquisition and cut_off metadata."
                ) from exc
    elif count is None:
        count = 512

    if dwell_time_s is None and bandwidth is None:
        if scan is not None:
            dwell_time_s = getattr(scan, "dwell_time_s", None)
            if dwell_time_s is None and hasattr(scan, "get"):
                dwell_time_s = scan.get("dwell_time_s", None)
        if dwell_time_s is None:
            raise ValueError(
                "calc_sampling_time_axis requires dwell_time_s or bandwidth, "
                "or scan with a valid sampling axis or dwell_time_s."
            )
        dwell_time_s = _sampling_scalar(
            dwell_time_s, "scan.dwell_time_s", positive=True
        )
    dt = dwell_time_s if dwell_time_s is not None else 1.0 / bandwidth
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            "calc_sampling_time_axis requires a finite positive sampling interval."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        if scan is not None:
            axis = np.linspace(0, dt * (count - 1), count)
        else:
            indices = np.arange(count) if start_with_0 else np.arange(1, count + 1)
            axis = indices * dt
            if t_echo_s is not None:
                axis += t_echo_s
    if not np.all(np.isfinite(axis)) or np.any(axis[1:] <= axis[:-1]):
        raise ValueError(
            "calc_sampling_time_axis inputs do not produce finite, increasing float64 times."
        )
    return axis


def calc_per_image_timeaxis_s(scan=None, start_with_0=True):
    """Calculate the time range [in s] on which the data was acquired.

    Parameters
    ----------
    scan : object, optional
        Data object containing timing parameters (original interface)

    """
    if scan is None:
        return None

    try:
        nr = scan.n_repetitions
        image_repetition_time = scan.scan_duration_s / nr

        if start_with_0:
            return np.linspace(0, (nr - 1) * image_repetition_time, nr)
        else:
            return np.linspace(image_repetition_time, nr * image_repetition_time, nr)
    except BaseException:
        return None


def calc_inversion_time_multi_slice(scan):
    """Calculate time increment on the inversion time for each slice in FAIR sequence."""
    if scan is None:
        raise ValueError("Data object is None")

    ref_pulse_time = scan.get("refpulse_duration_ms", 0)
    if ref_pulse_time == 0 or not ref_pulse_time:
        return 0

    fat_sup_time = scan.method.get("PVM_FatSupModuleTime", 0)
    mag_trans_time = scan.method.get("PVM_MagTransModuleTime", 0)
    fov_sat_time = scan.method.get("PVM_FovSatModuleTime", 0)
    rep_spoiler_time = scan.method.get("RepetitionSpoiler", [0, 0, 0])[2]

    amp_enable_time = scan.method.get("PVM_FatSupAmpEnable", 0)
    rf_pulse_time = scan.get("rfpulse_duration_ms")
    if rf_pulse_time is None:
        return 0
    exc_pulse_half_time = rf_pulse_time / 2
    rare_factor = scan.rare_factor
    echo_time = scan.echo_time_ms

    grad_stab_time = scan.method.get("GradStab", 0)
    rise_time = 0.169  # Bruker default rise time in ms

    # This is taken from the FAIR_RARE Bruker sequence. other sequences will have different slice_TI_increment
    slice_TI_increment = (
        0.02
        + fat_sup_time
        + mag_trans_time
        + fov_sat_time
        + rep_spoiler_time
        + amp_enable_time
        + exc_pulse_half_time
        + echo_time * rare_factor
        + (echo_time - ref_pulse_time - 2 * grad_stab_time) / 2
        + 2.0
        + rise_time
    )
    return slice_TI_increment


def create_time_axis(
    start_time: float, end_time: float, dt: float, include_end: bool = True
) -> np.ndarray:
    """
    Create a time axis with specified parameters.

    Parameters
    ----------
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    dt : float
        Time step in seconds
    include_end : bool, optional
        Whether to include the end time point. Default is True.

    Returns
    -------
    array_like
        Time axis in seconds

    """
    if include_end:
        n_points = int((end_time - start_time) / dt) + 1
        time_axis = np.linspace(start_time, end_time, n_points)
    else:
        time_axis = np.arange(start_time, end_time, dt)

    return time_axis


def resample_time_series(
    time_old: np.ndarray,
    data_old: np.ndarray,
    time_new: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Resample time series data to new time axis.

    Parameters
    ----------
    time_old : array_like
        Original time axis
    data_old : array_like
        Original data values
    time_new : array_like
        New time axis for resampling
    method : str, optional
        Interpolation method ('linear', 'cubic', 'nearest'). Default is 'linear'.

    Returns
    -------
    array_like
        Resampled data on new time axis

    """
    # Handle multi-dimensional data
    if data_old.ndim > 1:
        # Assume time is first dimension
        data_new = np.zeros((len(time_new),) + data_old.shape[1:])

        # Flatten extra dimensions for vectorized interpolation
        original_shape = data_old.shape
        data_flat = data_old.reshape(original_shape[0], -1)
        data_new_flat = np.zeros((len(time_new), data_flat.shape[1]))

        for i in range(data_flat.shape[1]):
            interp_func = interp1d(
                time_old, data_flat[:, i], kind=method, bounds_error=False, fill_value=0
            )
            data_new_flat[:, i] = interp_func(time_new)

        data_new = data_new_flat.reshape((len(time_new),) + original_shape[1:])
    else:
        # 1D data
        interp_func = interp1d(
            time_old, data_old, kind=method, bounds_error=False, fill_value=0
        )
        data_new = interp_func(time_new)

    return data_new


def interpolate_temporal_data(
    time_axis: np.ndarray, data: np.ndarray, target_dt: float, method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate temporal data to achieve target time resolution.

    Parameters
    ----------
    time_axis : array_like
        Original time axis
    data : array_like
        Data to interpolate (time should be first dimension)
    target_dt : float
        Target time step in seconds
    method : str, optional
        Interpolation method. Default is 'linear'.

    Returns
    -------
    tuple
        (new_time_axis, interpolated_data)

    """
    # Create new time axis with target resolution
    time_start = time_axis[0]
    time_end = time_axis[-1]
    new_time_axis = create_time_axis(time_start, time_end, target_dt, include_end=True)

    # Interpolate data
    interpolated_data = resample_time_series(time_axis, data, new_time_axis, method)

    return new_time_axis, interpolated_data


def smooth_temporal_data(
    data: np.ndarray, window_size: int, axis: int = 0, method: str = "moving_average"
) -> np.ndarray:
    """
    Smooth temporal data using specified method.

    Parameters
    ----------
    data : array_like
        Data to smooth
    window_size : int
        Size of smoothing window
    axis : int, optional
        Temporal axis. Default is 0.
    method : str, optional
        Smoothing method ('moving_average', 'gaussian'). Default is 'moving_average'.

    Returns
    -------
    array_like
        Smoothed data

    """
    if method == "moving_average":
        # Use convolution for moving average
        kernel = np.ones(window_size) / window_size

        if data.ndim == 1:
            smoothed = np.convolve(data, kernel, mode="same")
        else:
            # Apply along specified axis
            smoothed = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode="same"), axis, data
            )

    elif method == "gaussian":
        from scipy import ndimage

        sigma = window_size / 4.0  # Convert window size to sigma
        smoothed = ndimage.gaussian_filter1d(data, sigma, axis=axis)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return smoothed
