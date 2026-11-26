"""Construct spectroscopy frequency axes and calibrate Hz/ppm positions.

The shifted FFT bins, axis-calibrated regressions and signed interval helpers
retain their existing scientific rules. Explicit centering translates monotonic
axes to endpoint midpoint zero without changing their direction. Operations
read metadata or supplied
axes, never signal samples, and construct no plots, widgets or result caches.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from ...compatibility import resolve_frequency_axis
from ...reconstruction.axes import center_freq_axis, get_freq_axis
from ...utils.system.logging import debug_log

# Retain one owner for the existing spectroscopy diagnostic defaults.
DEBUG_FILE = "/tmp/analysis_spec_core.log"
DEBUG_LOG = False

__all__ = [
    "add_fit_freqs_ppm",
    "center_freq_axis",
    "freq_hz_to_ppm",
    "freq_ppm_to_hz",
    "freq_to_index",
    "get_bw_from_freq_axis",
    "get_freq_axis",
]


def debug_log_spec_core(
    message: str = "", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG
):
    """Write optional diagnostics using the historical spectroscopy defaults.

    Parameters
    ----------
    message : str, optional
        Diagnostic text passed unchanged to the shared logging helper.
    file_name : str or pathlib.Path, optional
        Destination when writing is enabled; retains the historical spectroscopy
        path. The default is captured at definition time.
    write_to_file : bool, optional
        Enable file output; default False is a no-op. When True, the shared
        helper creates parent directories and appends timestamped text.

    Returns
    -------
    None
        No scientific data are read or mutated and no values are cached.
        Timestamps make enabled diagnostic output nondeterministic.
    """
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


def freq_to_index(freq: float | None, freq_range: np.ndarray | None) -> int:
    """Find the nearest frequency sample using the first minimum distance.

    Parameters
    ----------
    freq : float or None
        Requested frequency, in the same units as ``freq_range``. A numeric
        scalar is required for the normal path; None retains arithmetic failure.
    freq_range : numpy.ndarray or None
        Nonempty one-dimensional frequency samples in Hz, ppm or another shared
        unit. Ascending and descending order are accepted without sorting.
        None retains arithmetic failure.

    Returns
    -------
    int
        NumPy integer positional index of the nearest sample; ties select the
        first array occurrence. Out-of-range requests select the nearest
        endpoint. Deterministic and eager, with no input mutation or caching.

    Raises
    ------
    TypeError
        If inputs do not support numeric subtraction.
    ValueError
        If the axis is empty. Nonfinite inputs retain NumPy argmin behavior.

    Examples
    --------
    >>> int(freq_to_index(-150.0, np.array([-200.0, -100.0, 0.0])))
    0
    """
    index = np.argmin(np.abs(freq_range - freq))

    return index


def get_bw_from_freq_axis(
    freq_axis: np.ndarray | None, scan: Any = None, unit: str = "Hz"
) -> float:
    """Calculate endpoint span plus the first frequency-sample spacing.

    Parameters
    ----------
    freq_axis : numpy.ndarray or None
        One-dimensional numeric frequency axis with at least two samples, in
        Hz or ppm. Pass None to derive the axis from ``scan``. Ascending
        and descending axes use their existing first-step rule without sorting.
    scan : object or None, optional
        Acquisition context accepted by :func:`get_freq_axis`. Mutually exclusive
        with an explicit axis; no signal samples are accessed.
    unit : str, optional
        Hz (default) or ppm when deriving the axis. Ignored for an explicit axis;
        no unit conversion is applied to supplied values.

    Returns
    -------
    float
        Absolute endpoint span plus the absolute first step, in axis units.
        NumPy input arithmetic determines precision without explicit casting.
        Inputs are preserved; computation is eager, deterministic and uncached.

    Raises
    ------
    ValueError
        If neither or both sources are supplied, or axis derivation rejects
        acquisition settings.
    IndexError, TypeError
        If the supplied/derived axis has fewer than two samples or is unavailable.

    Notes
    -----
    This is a sampled uniform-axis bandwidth convention. No uniformity validation
    is added: for nonuniform axes the first step still defines sample width,
    so reversing such an axis can change the returned bandwidth.

    Examples
    --------
    >>> float(get_bw_from_freq_axis(np.array([-500.0, -250.0, 0.0, 250.0])))
    1000.0
    """
    # Ensure that either freq_axis or scan is provided, but not both
    if freq_axis is None and scan is None:
        raise ValueError("have to pass either freq_axis OR scan")

    if freq_axis is not None and scan is not None:
        raise ValueError("Do not pass freq_axis AND scan")

    # If freq_axis is provided directly, calculate the bandwidth using it
    if freq_axis is not None:
        # Calculate the sampling step as the absolute difference between the
        # first two points
        freq_axis_sampling_step = np.abs(freq_axis[1] - freq_axis[0])
        # Calculate the bandwidth as the absolute difference between the last
        # and first points
        freq_axis_bw = np.abs(freq_axis[-1] - freq_axis[0])
        # Adjust the bandwidth by adding the width of one sampling point
        bw = freq_axis_bw + freq_axis_sampling_step
        return bw

    # If a data object is provided, extract the frequency axis from it first
    if scan is not None:
        # Extract frequency axis from data object, considering the specified
        # unit
        freq_axis = get_freq_axis(scan=scan, unit=unit)
        # Repeat the bandwidth calculation as done for the directly provided
        # freq_axis
        freq_axis_sampling_step = np.abs(freq_axis[1] - freq_axis[0])
        freq_axis_bw = np.abs(freq_axis[-1] - freq_axis[0])
        bw = freq_axis_bw + freq_axis_sampling_step
        return bw


def freq_hz_to_ppm(
    freq_hz: float = 0.0,
    hz_axis: np.array | None = None,
    ppm_axis: np.array | None = None,
    ppm_axis_flipped: bool = False,
    scan: Any = None,
    ppm_centered_at_0: bool = False,
) -> float | np.ndarray | None:
    """Convert Hz positions to ppm using corresponding calibration-axis samples.

    The calibration is the ordinary least-squares line from Hz to ppm, including
    its intercept. Axis order is retained unless an explicit flip is requested.
    For a ``scan`` without a usable spectral axis (imaging scans such as
    bSSFP, or fewer than two acquisition points) the line is taken from
    metadata instead: ``ppm = center_ppm + Hz / reference frequency [MHz]``.

    Parameters
    ----------
    freq_hz : float or array_like, optional
        Absolute frequency positions in Hz; scalar, list or numeric NumPy array
        of any shape. Default is zero Hz. No signal samples are accessed.
    hz_axis : numpy.ndarray or None, optional
        One-dimensional Hz calibration samples corresponding point-for-point to
        ``ppm_axis``. None derives this axis from ``scan``.
    ppm_axis : numpy.ndarray or None, optional
        One-dimensional ppm samples, ascending or descending, with the same
        length as ``hz_axis``. None derives this axis from ``scan``.
    ppm_axis_flipped : bool, optional
        Reverse the supplied/derived ppm samples before regression. This changes
        calibration direction; it does not sort or reorder the Hz samples.
    scan : object or None, optional
        Metadata context accepted by :func:`get_freq_axis`. Used only for missing
        axes; when both axes are supplied the context is ignored and preserved.
        If both axes are missing and the context cannot provide two or more
        samples of each, ``center_ppm`` and ``reference_frequency_mhz`` of the
        context are used instead.
    ppm_centered_at_0 : bool, optional
        Apply :func:`center_freq_axis` independently to both calibration axes
        before regression, preserving their directions and moving each endpoint
        midpoint to zero. Both axes must then be finite and strictly monotonic.
        Default False preserves their absolute intercept and validation behavior.
        With the metadata line this drops ``center_ppm``, so 0 Hz maps to 0 ppm.

    Returns
    -------
    float or numpy.ndarray or list
        Positions in ppm. Python list input returns a list; NumPy input retains
        its shape and returns an ndarray; scalar input returns a NumPy scalar.
        Labelled inputs are converted with ``numpy.asarray`` and lose labels and
        attrs. SciPy regression/NumPy arithmetic determine calculation precision.
        Inputs remain unchanged. Execution is eager, deterministic and uncached.

    Raises
    ------
    ValueError
        If calibration axes are missing without a context, or SciPy rejects their
        lengths or a constant Hz axis. When centering is requested, also raised
        for empty, nonvector, nonfinite or nonmonotonic calibration axes.
        Also raised if the metadata line is needed but ``center_ppm`` or a
        positive ``reference_frequency_mhz`` is missing. Metadata construction
        errors propagate.
    TypeError
        If supplied values or unavailable derived axes cannot be used numerically.

    Notes
    -----
    Absolute conversion includes the axis intercept. Use the relative ppm helper
    for intervals. The Larmor frequency is used only by the metadata line, taken
    from ``reference_frequency_mhz``; no gyromagnetic ratio is inferred.
    Without explicit centering, invalid/nonfinite calibration retains SciPy's
    existing behavior. Explicit centering validates both axes before regression;
    the supplied frequency positions are interpreted relative to the centered
    Hz axis and are not translated separately.

    Examples
    --------
    >>> hz = np.array([-200.0, -100.0, 0.0, 100.0])
    >>> ppm = np.array([168.0, 169.0, 170.0, 171.0])
    >>> np.testing.assert_allclose(freq_hz_to_ppm(50.0, hz, ppm), 170.5)
    >>> np.testing.assert_allclose(freq_hz_to_ppm([-100.0, 100.0], hz, ppm), [169., 171.])
    """
    use_metadata_line = False
    if hz_axis is None or ppm_axis is None:
        if scan is None:
            raise ValueError("Provide either hz_axis and ppm_axis or scan!")
        both_missing = hz_axis is None and ppm_axis is None
        if hz_axis is None:
            hz_axis = get_freq_axis(scan=scan, unit="Hz")
        if ppm_axis is None:
            ppm_axis = get_freq_axis(scan=scan, unit="ppm")
        use_metadata_line = both_missing and not _has_spectral_axes(hz_axis, ppm_axis)

    if use_metadata_line:
        hz_per_ppm, center_ppm = _hz_ppm_calibration_from_metadata(
            scan, "freq_hz_to_ppm"
        )
        slope = 1.0 / hz_per_ppm
        intercept = 0.0 if ppm_centered_at_0 else center_ppm
    else:
        if ppm_axis_flipped:
            ppm_axis = np.flip(ppm_axis)

        if ppm_centered_at_0:
            ppm_axis = center_freq_axis(freq_axis=ppm_axis)
            hz_axis = center_freq_axis(freq_axis=hz_axis)

        # calculate a "conversion function" using linear regression
        from scipy.stats import linregress

        f_hz_to_ppm = linregress(hz_axis, ppm_axis)
        slope, intercept = f_hz_to_ppm.slope, f_hz_to_ppm.intercept

    input_was_list = isinstance(freq_hz, list)  # Check if input is a list

    # Ensure freq_hz is an array for vectorized operations
    freq_hz = np.asarray(freq_hz)

    # Convert using the linear regression results
    freq_ppm = intercept + slope * freq_hz

    # Return the output in the same type as the input
    if input_was_list:
        return freq_ppm.tolist()  # Convert back to list if the input was a list
    elif isinstance(freq_hz, np.ndarray):
        return freq_ppm  # Keep as numpy array if the input was an array
    else:
        return float(freq_ppm)  # Convert to float if the input was a scalar


def _hz_ppm_calibration_from_metadata(scan: Any, operation: str) -> tuple[float, float]:
    """Return Hz-per-ppm and the centre position from scan metadata.

    The spectral axes of :func:`get_freq_axis` place 0 Hz at ``center_ppm`` and
    advance by the reference frequency per ppm, i.e. ``hz = (ppm - center_ppm) *
    reference_frequency_mhz``. Evaluating that line directly also works for
    scans without a spectral axis (imaging, single-point acquisitions).

    Parameters
    ----------
    scan : object
        Scan or metadata context. ``center_ppm`` is read through ``scan.get``
        (ppm; the position of 0 Hz, i.e. the transmit centre frequency).
        ``reference_frequency_mhz`` (MHz, equal to Hz/ppm) is read from
        ``scan.handler`` when present, otherwise from ``scan``.
    operation : str
        Name of the calling public function, used in the error message.

    Returns
    -------
    tuple of float
        ``(reference_frequency_mhz, center_ppm)``; the first value is the slope
        in Hz/ppm, the second the ppm position of 0 Hz.

    Raises
    ------
    ValueError
        If ``center_ppm`` or a positive, finite ``reference_frequency_mhz`` is
        not available on ``scan``.
    """
    source = getattr(scan, "handler", None) or scan
    reference_mhz = getattr(source, "reference_frequency_mhz", None)
    center_ppm = scan.get("center_ppm", None) if hasattr(scan, "get") else None
    try:
        reference_mhz = float(reference_mhz)
        center_ppm = float(center_ppm)
    except (TypeError, ValueError):
        reference_mhz = center_ppm = float("nan")
    if not (
        np.isfinite(reference_mhz) and reference_mhz > 0 and np.isfinite(center_ppm)
    ):
        raise ValueError(
            f"{operation}: the data object has no usable spectral axis and "
            "lacks 'reference_frequency_mhz' (> 0) or 'center_ppm' for the "
            "metadata-based conversion. Pass hz_axis and ppm_axis explicitly."
        )
    return reference_mhz, center_ppm


def _has_spectral_axes(hz_axis: Any, ppm_axis: Any) -> bool:
    """Return whether both derived calibration axes hold at least two samples."""
    return np.size(hz_axis) >= 2 and np.size(ppm_axis) >= 2


def freq_ppm_to_hz(
    freq_ppm: float = 0,
    hz_axis: np.ndarray = None,
    ppm_axis: np.ndarray = None,
    ppm_axis_flipped: bool = False,
    scan: Any = None,
):
    """Convert ppm positions to Hz relative to the scan centre frequency.

    With calibration axes, the ordinary least-squares regression from ppm to Hz
    includes the calibration intercept. Its legacy list/scalar policy is
    deliberately retained. For a ``scan`` without a usable spectral axis
    (imaging scans such as bSSFP, or fewer than two acquisition points) the line
    is taken from metadata instead: ``Hz = (ppm - center_ppm) * reference
    frequency [MHz]``.

    Parameters
    ----------
    freq_ppm : float or array_like, optional
        Absolute chemical-shift positions in ppm. Python numeric scalars and
        one-dimensional iterables are supported; iterable entries may be None.
        Default is zero ppm. Use the array helper for arbitrary-dimensional arrays.
    hz_axis : numpy.ndarray or None, optional
        One-dimensional Hz calibration samples paired with ``ppm_axis``.
        None derives the missing axis from ``scan`` when available.
    ppm_axis : numpy.ndarray or None, optional
        One-dimensional ppm calibration samples with the same length as
        ``hz_axis``. Both ascending and descending order are retained.
    ppm_axis_flipped : bool, optional
        Reverse only the ppm samples before regression; default False.
    scan : object or None, optional
        Metadata context accepted by :func:`get_freq_axis`, used for missing
        calibration axes only. If both axes are missing and the context cannot
        provide two or more samples of each, ``center_ppm`` and
        ``reference_frequency_mhz`` of the context are used instead. The context
        and supplied arrays remain unchanged.

    Returns
    -------
    float or list
        Positions in Hz, where 0 Hz is the centre frequency of the acquisition
        (``center_ppm``). One result returns its NumPy scalar value or None,
        including for a one-element iterable. Multiple results return a list,
        retaining None entries; an empty iterable returns an empty list.
        SciPy regression and NumPy arithmetic determine precision. No labels or
        attrs are returned. Computation is eager, deterministic and uncached.

    Raises
    ------
    ValueError
        If neither both axes nor ``scan`` are given; if SciPy rejects axis
        lengths or a constant ppm axis; or if the metadata fallback is needed
        but ``center_ppm`` or a positive ``reference_frequency_mhz`` is missing.
        Errors from deriving a metadata axis propagate unchanged.
    TypeError
        If input values are not iterable/supported numeric scalars, or unavailable
        derived axes cannot be used by SciPy.

    Notes
    -----
    This direction independently regresses Hz against ppm. On non-affine paired
    axes it is not necessarily the exact inverse of :func:`freq_hz_to_ppm`.
    Nonfinite calibration retains SciPy's existing behavior.

    The metadata fallback agrees with the axis regression of spectroscopic scans
    to the precision of the stored bandwidth ratio (about 2e-5 relative for
    Bruker CSI, i.e. 0.02 Hz at 836 Hz).

    Examples
    --------
    >>> hz = np.array([-200.0, -100.0, 0.0, 100.0])
    >>> ppm = np.array([168.0, 169.0, 170.0, 171.0])
    >>> np.testing.assert_allclose(freq_ppm_to_hz(170.5, hz, ppm), 50.0)
    >>> converted = freq_ppm_to_hz([170.0, None, 171.0], hz, ppm)
    >>> converted[1] is None
    True
    >>> np.testing.assert_allclose([converted[0], converted[2]], [0.0, 100.0])
    """
    use_metadata_line = False
    if hz_axis is None or ppm_axis is None:
        if scan is None:
            raise ValueError("Provide either hz_axis and ppm_axis or scan!")
        both_missing = hz_axis is None and ppm_axis is None
        if hz_axis is None:
            hz_axis = get_freq_axis(scan=scan, unit="Hz")
        if ppm_axis is None:
            ppm_axis = get_freq_axis(scan=scan, unit="ppm")
        use_metadata_line = both_missing and not _has_spectral_axes(hz_axis, ppm_axis)

    if use_metadata_line:
        slope, center_ppm = _hz_ppm_calibration_from_metadata(scan, "freq_ppm_to_hz")
        intercept = -center_ppm * slope
    else:
        # check if passed ppm axis was flipped:
        if ppm_axis_flipped:
            ppm_axis = np.flip(ppm_axis)

        from scipy.stats import linregress

        f_ppm_to_hz = linregress(ppm_axis, hz_axis)
        slope, intercept = f_ppm_to_hz.slope, f_ppm_to_hz.intercept

    # make iterable:
    if isinstance(freq_ppm, float) or isinstance(freq_ppm, int):
        freq_ppm = [freq_ppm]

    freq_hz = [None if f is None else intercept + slope * f for f in freq_ppm]

    if len(freq_hz) == 1:
        return freq_hz[0]
    return freq_hz


def _convert_ppm_ndarray(
    array_val: np.ndarray,
    *,
    hz_axis: np.ndarray,
    ppm_axis: np.ndarray,
) -> np.ndarray:
    """Convert an array of absolute ppm positions to Hz.

    The conversion is calibrated from corresponding points on ``ppm_axis``
    and ``hz_axis``. The returned array has the same shape as ``array_val``;
    only its values and dtype change.

    Parameters
    ----------
    array_val : numpy.ndarray
        Absolute chemical-shift positions in ppm.
    hz_axis : numpy.ndarray
        Frequency axis in Hz used for the calibration.
    ppm_axis : numpy.ndarray
        Chemical-shift axis in ppm corresponding point-for-point to
        ``hz_axis``.

    Returns
    -------
    numpy.ndarray
        Frequencies in Hz with ``dtype=float`` and the same shape as the input.

    Raises
    ------
    ValueError
        If either calibration axis is missing.
    """
    if hz_axis is None or ppm_axis is None:
        raise ValueError(
            "Frequency axis information is required to convert from ppm to Hz."
        )

    ppm_values = np.asarray(array_val)
    converted = freq_ppm_to_hz(
        freq_ppm=ppm_values.ravel().tolist(),
        hz_axis=hz_axis,
        ppm_axis=ppm_axis,
    )
    return np.asarray(converted, dtype=float).reshape(ppm_values.shape)


def _convert_ppm_value(
    value: float | np.ndarray | xr.DataArray | None,
    *,
    hz_axis: np.ndarray,
    ppm_axis: np.ndarray,
) -> float | np.ndarray | None:
    """Convert absolute ppm position(s) to Hz while preserving value shape.

    Scalars are returned as ``float`` values, NumPy arrays and xarray data
    arrays are returned as NumPy arrays with the same shape, and ``None`` is
    passed through unchanged. Use :func:`_ppm_delta_to_hz` for ppm intervals
    or offsets, because absolute conversion includes the axis intercept.

    Parameters
    ----------
    value : float, numpy.ndarray, xarray.DataArray, or None
        Absolute chemical-shift position or positions in ppm.
    hz_axis : numpy.ndarray
        Frequency axis in Hz used for the calibration.
    ppm_axis : numpy.ndarray
        Chemical-shift axis in ppm corresponding point-for-point to
        ``hz_axis``.

    Returns
    -------
    float, numpy.ndarray, or None
        Absolute frequency position(s) in Hz.
    """
    if value is None:
        return None
    if isinstance(value, xr.DataArray):
        return _convert_ppm_ndarray(
            np.asarray(value.data),
            hz_axis=hz_axis,
            ppm_axis=ppm_axis,
        )
    if isinstance(value, np.ndarray):
        return _convert_ppm_ndarray(
            value,
            hz_axis=hz_axis,
            ppm_axis=ppm_axis,
        )

    converted = _convert_ppm_ndarray(
        np.asarray(value),
        hz_axis=hz_axis,
        ppm_axis=ppm_axis,
    )
    return float(converted)


def _ppm_delta_to_hz(
    delta_ppm: float | np.ndarray | xr.DataArray,
    *,
    hz_axis: np.ndarray,
    ppm_axis: np.ndarray,
) -> float | np.ndarray:
    """Convert a relative interval in ppm to its signed width in Hz.

    Unlike an absolute chemical-shift conversion, an interval must not include
    the offset between the ppm and Hz axes. This function therefore subtracts
    the Hz position of 0 ppm from the converted endpoint. The sign of the axis
    calibration is preserved, which also supports descending ppm axes.

    Parameters
    ----------
    delta_ppm : float, numpy.ndarray, or xarray.DataArray
        Relative ppm interval or intervals to convert.
    hz_axis : numpy.ndarray
        Frequency axis in Hz used for the calibration.
    ppm_axis : numpy.ndarray
        Chemical-shift axis in ppm corresponding point-for-point to
        ``hz_axis``.

    Returns
    -------
    float or numpy.ndarray
        Signed interval(s) in Hz, matching the scalar or array shape of
        ``delta_ppm``.
    """
    delta_hz = _convert_ppm_value(
        delta_ppm,
        hz_axis=hz_axis,
        ppm_axis=ppm_axis,
    )
    zero_hz = _convert_ppm_value(
        0.0,
        hz_axis=hz_axis,
        ppm_axis=ppm_axis,
    )
    return delta_hz - zero_hz


def add_fit_freqs_ppm(fit_results: xr.Dataset) -> xr.Dataset:
    """Return spectroscopy fit results with ``fit_freqs_ppm`` from ``fit_freqs``.

    ``fit_freqs_ppm`` is the linear Hz-to-ppm conversion of ``fit_freqs``
    with the Hz and ppm frequency axes stored in ``fit_results.attrs``, see
    :func:`freq_hz_to_ppm`. An existing ``fit_freqs_ppm`` is recomputed, so
    the variable follows frequencies changed after the fit, for example by a
    review refit or an added metabolite.

    Parameters
    ----------
    fit_results : xr.Dataset
        Fit results with ``fit_freqs`` in Hz and the ``freq_axis_hz`` and
        ``freq_axis_ppm`` axes as coordinates or attributes. It is not
        modified.

    Returns
    -------
    xr.Dataset
        A shallow copy with ``fit_freqs_ppm`` (unit ppm, same dimensions and
        coordinates as ``fit_freqs``). Without ``fit_freqs`` or without both
        calibration axes the input is returned unchanged.
    """
    hz_axis = resolve_frequency_axis(fit_results, "Hz")
    ppm_axis = resolve_frequency_axis(fit_results, "ppm")
    if hz_axis is None or ppm_axis is None or "fit_freqs" not in fit_results:
        return fit_results
    freqs_hz = fit_results["fit_freqs"]
    freqs_ppm = xr.apply_ufunc(
        freq_hz_to_ppm,
        freqs_hz,
        kwargs={"hz_axis": np.asarray(hz_axis), "ppm_axis": np.asarray(ppm_axis)},
    )
    freqs_ppm.attrs = {**freqs_hz.attrs, "unit": "ppm"}
    return fit_results.assign(fit_freqs_ppm=freqs_ppm)
