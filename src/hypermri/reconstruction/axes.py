"""Time and frequency axes of reconstructed spectroscopy data.

This module owns the spectral frequency axis: the FFT-shifted bins in Hz or
ppm (:func:`get_freq_axis`), their translation to a zero midpoint
(:func:`center_freq_axis`), and the axes attached to a reconstructed FID
after leading samples were removed. It reads acquisition metadata, never
signal samples, and imports nothing from analysis or visualization, so
vendor handlers and reconstruction strategies can use it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from hypermri.compatibility import assign_frequency_axis

__all__ = [
    "apply_processed_axis_metadata",
    "center_freq_axis",
    "frequency_axes",
    "get_freq_axis",
    "processed_sampling_axis",
    "resolve_cut_off",
]


def get_freq_axis(
    scan: Any = None,
    unit: str = "ppm",
    cut_off: int | None = None,
    npoints: int | tuple[int] | None = None,
    centered_at_0: bool | None = False,
    dwell_time_s: float | None = None,
) -> np.ndarray | list[float] | None:
    """Construct the existing shifted Fourier frequency axis in Hz or ppm.

    Parameters
    ----------
    scan
        Acquisition context exposing bandwidth_hz, bandwidth_ppm, n_acquisition
        and get(name, default). Dwell time in seconds, optional cut_off and
        centre offset in ppm come from that context. None requires explicit
        dwell_time_s and npoints and supports only Hz.
    unit
        Hz or ppm, case-insensitive. Unknown units raise before preparation.
    cut_off
        Number of acquisition points excluded when deriving the length from
        metadata; None reads the context's cut_off, defaulting to zero.
        Explicit npoints takes precedence. With no context this value does not
        shorten the explicitly requested axis.
    npoints
        Positive length of the frequency axis. None uses n_acquisition minus
        cut_off. A one-element tuple is accepted for legacy explicit Hz inputs.
    centered_at_0
        Translate the axis so its two endpoint frequencies have midpoint zero.
        False or None retains shifted FFT bins and, for ppm, the centre offset.
    dwell_time_s
        Positive sampling interval in seconds for explicit Hz construction.
        A supplied context uses its own dwell time instead.

    Returns
    -------
    numpy.ndarray or list of float or None
        One-dimensional float64 axis of length npoints (or the metadata-derived
        length), increasing for positive sampling/calibration bandwidths, in
        the requested units. The ppm calibration
        uses bandwidth_hz / bandwidth_ppm and adds center_ppm. Existing missing
        metadata behavior is retained: absent bandwidth_hz or unavailable ppm
        calibration returns None; bandwidth_hz=None returns an empty list.

    Raises
    ------
    ValueError
        If unit is unsupported, explicit sampling arguments are missing, ppm
        is requested without acquisition calibration, or FFT length is invalid,
        including a metadata-derived length n_acquisition - cut_off below one.
        Explicit centering also rejects nonfinite or nonmonotonic axes.
    ZeroDivisionError
        If the sampling interval used to construct FFT bins is zero.
    AttributeError, TypeError
        If acquisition metadata lack required attributes/values or an unsupported
        FFT length type is supplied. No new metadata validation is introduced.

    Notes
    -----
    Construction is eager and deterministic. It does not read signal samples,
    edit acquisition metadata or cache an axis. Existing FFT-bin orientation,
    midpoint translation and conversion precision are preserved.

    Examples
    --------
    >>> get_freq_axis(unit="Hz", dwell_time_s=0.001, npoints=4)
    array([-500., -250.,    0.,  250.])
    """
    if unit.lower() not in {"hz", "ppm"}:
        raise ValueError(f"get_freq_axis: unsupported unit {unit!r}; use Hz or ppm.")

    if scan is None:
        if dwell_time_s is None or npoints is None:
            raise ValueError(
                "get_freq_axis: if scan=None, pass dwell_time_s and npoints."
            )
        if unit.lower() == "ppm":
            raise ValueError("if scan=None this can only return unit=Hz axis!")
        if isinstance(npoints, tuple) and len(npoints) == 1:
            npoints = npoints[0]
        hz_axis = np.fft.fftshift(np.fft.fftfreq(npoints, d=dwell_time_s))
        if centered_at_0 is True:
            hz_axis = center_freq_axis(freq_axis=hz_axis)
        return hz_axis

    if cut_off is None:
        cut_off = scan.get("cut_off", 0)
    if not hasattr(scan, "bandwidth_hz"):
        return None
    bw_hz = scan.bandwidth_hz
    if bw_hz is None:
        return []

    from hypermri.utils.time.temporal import get_dwell_time_s

    dt = get_dwell_time_s(scan=scan)
    if npoints is None:
        freq_points = int(scan.n_acquisition - cut_off)
        if freq_points <= 0:
            raise ValueError(
                f"get_freq_axis: cut_off={cut_off} leaves no points of "
                f"n_acquisition={scan.n_acquisition}. Pass npoints or a "
                "cut_off smaller than n_acquisition."
            )
    else:
        freq_points = npoints
    hz_axis = np.fft.fftshift(np.fft.fftfreq(freq_points, d=dt))

    if unit.lower() == "hz":
        return center_freq_axis(freq_axis=hz_axis) if centered_at_0 else hz_axis

    bw_ppm = getattr(scan, "bandwidth_ppm", None)
    if bw_ppm is None or bw_ppm == 0:
        return None
    hz_per_ppm = bw_hz / bw_ppm
    ppm_axis = hz_axis / hz_per_ppm + scan.get("center_ppm", 0)
    return center_freq_axis(freq_axis=ppm_axis) if centered_at_0 else ppm_axis


def center_freq_axis(
    freq_axis: ArrayLike | None = None,
) -> np.ndarray | xr.DataArray:
    """Translate a monotonic frequency axis to endpoint midpoint zero.

    Parameters
    ----------
    freq_axis : array_like or xarray.DataArray or None, optional
        Nonempty one-dimensional finite real axis in Hz, ppm or another
        consistent unit. Multiple samples must be strictly increasing or
        strictly decreasing; uneven spacing is accepted. A single sample is
        translated to zero. None is retained as the legacy default but is not
        a valid axis. No acquisition metadata are required.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Axis of the same shape, order and units, shifted by the midpoint of its
        first and last samples. Spacing and ascending/descending direction are
        preserved. Floating inputs retain NumPy's arithmetic precision, including
        float32/float64 dtype. Integer inputs return float64. Their translation
        uses native integer arithmetic unless the span would overflow the signed
        integer range, in which case calculation promotes to float64.
        DataArray inputs retain their name, dimension order and coordinate
        values; attrs follow xarray's existing keep_attrs setting. Other inputs
        return an unlabelled array. The input is preserved. Execution is eager,
        deterministic for the active attribute setting, and uncached.

    Raises
    ------
    TypeError
        If the input is None or contains nonnumeric, boolean or complex values.
    ValueError
        If the axis is empty, not one-dimensional, nonfinite or, for multiple
        samples, not strictly monotonic. Repeated coordinates are rejected.

    Notes
    -----
    The axis is translated in two steps: subtract its minimum, then half its
    span. This preserves the existing ascending-axis rounding policy while
    centering descending axes too. Samples are never sorted or reversed.
    Zero denotes the midpoint of the frequency range, which need not coincide
    with a sampled frequency. No signal amplitudes or absolute calibration
    metadata are modified.

    Examples
    --------
    >>> center_freq_axis(np.array([-200.0, -100.0, 0.0, 100.0]))
    array([-150.,  -50.,   50.,  150.])
    >>> center_freq_axis(np.array([100.0, 0.0, -100.0, -200.0]))
    array([ 150.,   50.,  -50., -150.])
    """
    axis = np.asarray(freq_axis)
    if not np.issubdtype(axis.dtype, np.number) or np.iscomplexobj(axis):
        raise TypeError("center_freq_axis requires real numeric frequency values.")
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("center_freq_axis requires a nonempty one-dimensional axis.")
    if not np.all(np.isfinite(axis)):
        raise ValueError("center_freq_axis requires finite frequency values.")
    if axis.size > 1 and not (
        np.all(axis[1:] > axis[:-1]) or np.all(axis[1:] < axis[:-1])
    ):
        raise ValueError(
            "center_freq_axis requires a strictly monotonic axis; provide increasing "
            "or decreasing coordinates without repeats. The axis is never sorted."
        )
    values = freq_axis if isinstance(freq_axis, xr.DataArray) else axis
    if np.issubdtype(axis.dtype, np.signedinteger) and (
        abs(int(axis[-1]) - int(axis[0])) > np.iinfo(axis.dtype).max
    ):
        values = values.astype(np.float64)
    translated = values - np.min(values)
    return translated - np.max(translated) / 2.0


def _label_cpmg_echo_train(data, handler):
    """Label CPMG echo trains identically without changing spectral samples.

    Only the known singleton localization placeholders are removed. Echoes
    use EffectiveTE in seconds when complete, otherwise integer indices.
    Scanner processing may have a different acquisition length or bandwidth;
    those axes and their provenance are deliberately preserved.
    """
    method = getattr(handler, "method", None) or {}
    labels = (getattr(handler, "type", ""), method.get("Method", ""))
    if not any(
        str(label).strip("<>").lower().split(":")[-1] == "cpmg" for label in labels
    ):
        return data
    if "echo_idx" not in data.dims and "phase_idx" in data.dims:
        data = data.rename({"phase_idx": "echo_idx"})
    if "echo_idx" not in data.dims:
        return data
    for dim in ("read_idx", "phase_idx", "slice_idx"):
        if dim in data.dims and data.sizes[dim] == 1:
            data = data.squeeze(dim, drop=True)
    times = np.asarray(method.get("EffectiveTE", []), dtype=float).ravel()
    size = data.sizes["echo_idx"]
    if times.size == size:
        data = data.assign_coords(echo_idx=("echo_idx", times / 1000.0, {"unit": "s"}))
    elif "echo_idx" not in data.coords:
        data = data.assign_coords(
            echo_idx=("echo_idx", np.arange(size), {"unit": "index"})
        )
    order = ("acquisition_idx", "echo_idx", "channel_idx", "repetition_idx")
    for dim in order[2:]:
        if dim not in data.dims:
            data = data.expand_dims({dim: [0]})
        if dim not in data.coords:
            data = data.assign_coords({dim: np.arange(data.sizes[dim])})
    return data.transpose(*(dim for dim in order if dim in data.dims), ...)


def _echo_frequency_axes(handler, echo_times_ms):
    """Calculate frequency bins from explicit echo times without handler writes."""
    echo_times_ms = np.atleast_1d(echo_times_ms)
    spacing_ms = getattr(handler, "echo_spacing_ms", 0.0)
    if not spacing_ms and echo_times_ms.size > 1:
        spacing_ms = float(echo_times_ms[1] - echo_times_ms[0])
    dwell_s = spacing_ms / 1000.0 if spacing_ms > 0 else 1.0
    axes = {"Hz": np.fft.fftshift(np.fft.fftfreq(echo_times_ms.size, d=dwell_s))}
    reference_mhz = getattr(handler, "reference_frequency_mhz", None)
    if reference_mhz:
        axes["ppm"] = axes["Hz"] / reference_mhz + handler.get("center_ppm", 0)
    return axes


def resolve_cut_off(
    handler: Any,
    requested: Any,
    n_acquisition: int,
    *,
    context: str = "Spectroscopy",
) -> int:
    """Resolve and validate the number of leading points removed from each FID.

    Parameters
    ----------
    handler : Any
        Vendor handler whose ``cut_off`` attribute (default 0) is used when
        ``requested`` is None.
    requested : Any
        Number of leading FID samples to remove, or None for the handler's
        value. Integral floats such as ``3.0`` are accepted.
    n_acquisition : int
        Number of acquired samples per FID.
    context : str, optional
        Name of the calling reconstruction, used in error messages.

    Returns
    -------
    int
        Number of samples to remove, ``0 <= cut_off < n_acquisition``.

    Raises
    ------
    ValueError
        If the value is not an integer, is negative, or leaves no sample.
    """
    value = getattr(handler, "cut_off", 0) if requested is None else requested
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} cut_off must be an integer, got {value!r}."
        ) from exc

    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{context} cut_off must be an integer, got {value!r}.")
    if resolved < 0:
        raise ValueError(f"{context} cut_off must be non-negative, got {resolved}.")
    if resolved >= n_acquisition:
        raise ValueError(
            f"{context} cut_off must leave at least one FID point: "
            f"cut_off={resolved}, n_acquisition={n_acquisition}."
        )
    return resolved


def processed_sampling_axis(
    handler: Any,
    cut_off: int,
    npoints: int,
) -> tuple[np.ndarray, str]:
    """Return a zero-based sampling axis matching the processed FID.

    The axis counts from the first kept sample, not from excitation: after a
    cut-off of ``cut_off`` leading points the first kept sample is labelled
    0 s, although it was acquired ``cut_off`` dwell times after the first
    acquired sample. Callers that need the delay since excitation add that
    offset themselves (as ``phase_band_tau_s_from_scan`` does).

    Parameters
    ----------
    handler : Any
        Source of ``sampling_time_axis_s`` (acquired axis in seconds, shape
        ``(n_acquisition,)``) or, as fallback, ``dwell_time_s`` in seconds.
    cut_off : int
        Non-negative number of leading acquired samples removed.
    npoints : int
        Positive number of samples remaining after processing.

    Returns
    -------
    axis : numpy.ndarray
        Shape ``(npoints,)``, starting at 0. Float seconds when the acquired
        axis (sliced after ``cut_off`` and shifted to start at 0) or a
        positive finite dwell time is available, else integer sample indices.
    unit : str
        ``"s"`` for a time axis, ``"index"`` for the positional fallback.
    """
    raw_axis = getattr(handler, "sampling_time_axis_s", None)
    if raw_axis is not None:
        raw_axis = np.asarray(raw_axis, dtype=float).reshape(-1)
        if raw_axis.size >= cut_off + npoints:
            axis = raw_axis[cut_off : cut_off + npoints].copy()
        elif raw_axis.size == npoints:
            axis = raw_axis.copy()
        else:
            axis = None
        if axis is not None and axis.size == npoints and np.all(np.isfinite(axis)):
            return axis - axis[0], "s"

    dwell_time_s = getattr(handler, "dwell_time_s", None)
    try:
        dwell_time_s = float(dwell_time_s)
    except (TypeError, ValueError):
        dwell_time_s = None
    if dwell_time_s is not None and np.isfinite(dwell_time_s) and dwell_time_s > 0:
        return np.arange(npoints, dtype=float) * dwell_time_s, "s"

    return np.arange(npoints), "index"


def frequency_axes(handler: Any, npoints: int, cut_off: int) -> dict[str, np.ndarray]:
    """Calculate spectral bins for the effective rather than acquired FID size.

    Parameters
    ----------
    handler : VendorHandler
        Source of spectral bandwidth, reference frequency and ppm centre.
        Missing bandwidth leaves the corresponding axis unavailable.
    npoints : int
        Positive number of samples remaining after processing.
    cut_off : int
        Non-negative number of leading acquired samples removed.

    Returns
    -------
    dict of str to numpy.ndarray
        Available ``"Hz"`` and ``"ppm"`` axes, each with shape ``(npoints,)``.
        Orientation and centring follow ``get_freq_axis``; an unavailable axis
        is omitted. Arrays are computed eagerly without caching or mutation.

    Raises
    ------
    ValueError
        If supplied calibration metadata cannot define a frequency axis.
    TypeError
        If calibration metadata have an invalid type.
    """
    axes = {}
    for unit in ("ppm", "Hz"):
        axis = get_freq_axis(
            scan=handler,
            unit=unit,
            cut_off=cut_off,
            npoints=npoints,
        )
        if axis is not None and len(np.atleast_1d(axis)) == npoints:
            axes[unit] = np.asarray(axis)
    return axes


def apply_processed_axis_metadata(
    data: xr.DataArray,
    handler: Any,
    cut_off: int,
    sampling_axis: np.ndarray,
    sampling_unit: str,
    *,
    raw_n_acquisition: int | None = None,
) -> xr.DataArray:
    """Attach time and frequency axes aligned with reconstructed spectroscopy data.

    Parameters
    ----------
    data : xarray.DataArray
        Reconstructed spectroscopy data with an ``acquisition_idx`` dimension
        holding the samples kept after ``cut_off``.
    handler : Any
        Vendor handler that supplies bandwidth, reference frequency, centre
        chemical shift and ``n_acquisition``.
    cut_off : int
        Number of leading acquired samples that were removed.
    sampling_axis : numpy.ndarray
        Processed sampling axis with one value per kept sample, as returned by
        :func:`processed_sampling_axis`.
    sampling_unit : str
        ``"s"`` for a time axis or ``"index"`` for sample indices.
    raw_n_acquisition : int or None, optional
        Number of acquired samples before ``cut_off``; None reads
        ``handler.n_acquisition``.

    Returns
    -------
    xarray.DataArray
        New array with the ``acquisition_idx`` coordinate set to the sampling
        axis, an ``acquisition_time_s`` coordinate and ``dwell_time_s``
        attribute for a time axis, the frequency axes in Hz and ppm where the
        metadata define them, and the attributes ``cut_off`` and
        ``raw_n_acquisition``. Signal samples are not changed.

    Raises
    ------
    ValueError
        If ``sampling_axis`` does not have one value per sample.
    """
    npoints = data.sizes.get("acquisition_idx", 0)
    sampling_axis = np.asarray(sampling_axis)
    if sampling_axis.size != npoints:
        raise ValueError(
            "Processed spectroscopy sampling axis does not match reconstructed data: "
            f"{sampling_axis.size} != {npoints}."
        )

    data = data.assign_coords(
        acquisition_idx=(
            "acquisition_idx",
            sampling_axis,
            {"unit": sampling_unit},
        )
    )
    if sampling_unit == "s":
        data = data.assign_coords(
            acquisition_time_s=(
                "acquisition_idx",
                sampling_axis,
                {"unit": "s"},
            )
        )
        data.attrs["sampling_time_axis_s"] = sampling_axis
        if npoints > 1:
            data.attrs["dwell_time_s"] = float(np.median(np.diff(sampling_axis)))
    else:
        data.attrs.pop("sampling_time_axis_s", None)

    for unit, axis in frequency_axes(handler, npoints, cut_off).items():
        data = assign_frequency_axis(data, axis, unit)

    data.attrs["cut_off"] = cut_off
    if raw_n_acquisition is None:
        raw_n_acquisition = int(getattr(handler, "n_acquisition", 0))
    data.attrs["raw_n_acquisition"] = int(raw_n_acquisition)
    return data
