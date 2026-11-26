"""Window and truncate CSI samples for :meth:`BaseSpectroscopy.filter_signal`.

The NumPy kernels take arrays, an integer axis and explicit settings; they
know no dimension names or attributes. The labelled helpers resolve CSI
dimensions, call a kernel once and restore dimensions, coordinates and
attributes. The raw helpers keep the legacy vector layout of acquired
point/FID sample and reconstruct filtered raw data with the canonical CSI
strategy. None of these functions mutates its input or a handler cache.
"""

import copy
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..utils.system.logging import init_default_logger

logger = init_default_logger(__name__)

FILTER_TYPES = ("None", "Exponential", "Hann", "Hamming")
SPATIAL_ORDER_MODES = ("auto", "native", "csi_acquisition_order")
_ZERO = 0 + 1j * 0
_KSPACE_DIMS = ("acquisition_idx", "phase_idx", "read_idx")
_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


@dataclass(frozen=True)
class SampleWindow:
    """Settings of the window and truncation along one filter axis.

    Attributes mirror the parameters of ``filter_signal``. ``decay_constant``
    is the exponential attenuation per sample index, ``alpha`` the
    dimensionless Hamming coefficient and ``center_position`` the sample index
    of the Hamming/Hann maximum (None: the middle). ``signal_end`` is the
    number of samples kept, counted along the filter axis. ``flipangle_deg``
    is the resolved flip angle in degrees when ``apply_flipangle_correction``
    is True and is ignored otherwise.

    Raises
    ------
    ValueError
        If ``filter_type`` is not one of ``FILTER_TYPES``.
    """

    filter_type: str
    decay_constant: float
    alpha: float
    center_position: int | None
    signal_end: int | None
    remove_lowest_signal: bool
    apply_flipangle_correction: bool
    flipangle_deg: float | None

    def __post_init__(self):
        if self.filter_type not in FILTER_TYPES:
            valid = ", ".join(repr(name) for name in FILTER_TYPES)
            raise ValueError(
                f"filter_signal: filter_type {self.filter_type!r} is not supported; "
                f"use one of {valid}."
            )


def _window_values(size, window):
    """Return the window along an axis of ``size`` samples, or None."""
    if window.filter_type == "Exponential":
        return np.exp(-window.decay_constant * np.arange(size))
    if window.filter_type not in ("Hamming", "Hann"):
        return None
    center = (
        size // 2 if window.center_position is None else int(window.center_position)
    )
    shifted = np.clip(np.arange(size) - center + size // 2, 0, size - 1)
    phase = 2 * np.pi * shifted / max(size - 1, 1)
    if window.filter_type == "Hamming":
        return window.alpha - (1 - window.alpha) * np.cos(phase)
    return 0.5 * (1 - np.cos(phase))


def _along_axis(vector, ndim, axis):
    shape = [1] * ndim
    shape[axis] = vector.size
    return vector.reshape(shape)


def _correct_flipangle(values, axis, flipangle_deg):
    """Divide each sample by the longitudinal magnetization before its pulse."""
    from hypermri.utils.physics import hyperpolarized_Mz_flipangle

    n_pulses = values.shape[axis]
    # Longitudinal magnetization before each excitation, cos(alpha)**n.
    correction, _ = hyperpolarized_Mz_flipangle(
        M0_au=1, n_excitations=n_pulses, flipangle_deg=flipangle_deg
    )
    return values / _along_axis(correction, values.ndim, axis)


def _zero_lowest_mean_signal(values, axis, start, stop, keep_count):
    """Zero, in place, the samples in ``[start, stop)`` with lowest mean |x|."""
    n_available = stop - start
    n_to_remove = n_available - int(np.clip(keep_count, 0, n_available))
    if n_to_remove <= 0:
        return
    segment = [slice(None)] * values.ndim
    segment[axis] = slice(start, stop)
    other_axes = tuple(i for i in range(values.ndim) if i != axis)
    mean_signal = np.mean(np.abs(values[tuple(segment)]), axis=other_axes)
    zeroed = [slice(None)] * values.ndim
    zeroed[axis] = np.argsort(mean_signal)[:n_to_remove] + start
    values[tuple(zeroed)] = _ZERO


def window_and_truncate(values, axis, window, n_blocks=1):
    """Apply flip-angle correction, window and truncation along one axis.

    Parameters
    ----------
    values : numpy.ndarray
        Samples of any shape; not mutated.
    axis : int
        Non-negative index of the filter axis.
    window : SampleWindow
        Window and truncation settings.
    n_blocks : int, optional
        Number of equal repetition blocks along ``axis``. With more than one
        block that divides the axis length, truncation keeps
        ``signal_end`` samples in each block. Default 1.

    Returns
    -------
    numpy.ndarray
        Filtered samples with the shape of ``values``. The window and the
        correction promote the calculation dtype as NumPy arithmetic does.
    """
    filtered = np.array(values, copy=True)
    if window.apply_flipangle_correction:
        filtered = _correct_flipangle(filtered, axis, window.flipangle_deg)
    length = filtered.shape[axis]
    decay = _window_values(length, window)
    if decay is not None:
        filtered = filtered * _along_axis(decay, filtered.ndim, axis)
    if window.signal_end is None:
        return filtered
    keep = int(np.clip(window.signal_end, 0, length))
    if keep >= length:
        return filtered

    if n_blocks <= 1 or length % n_blocks != 0:
        n_blocks = 1
    block_size = length // n_blocks
    keep_per_block = int(np.clip(keep, 0, block_size))
    for start in range(0, length, block_size):
        stop = start + block_size
        if window.remove_lowest_signal:
            _zero_lowest_mean_signal(filtered, axis, start, stop, keep_per_block)
            continue
        tail = [slice(None)] * filtered.ndim
        tail[axis] = slice(start + keep_per_block, stop)
        filtered[tuple(tail)] = _ZERO
    return filtered


def _transform_spatial_dims(data, transform, domain):
    """Apply a centred CSI FFT over the present spatial dimensions."""
    dims = [dim for dim in _SPATIAL_DIMS if dim in data.dims]
    if not dims:
        return data.copy(deep=True)
    values = transform(
        np.asarray(data.values).copy(), [data.get_axis_num(d) for d in dims]
    )
    result = data.copy(data=values)
    result.attrs = dict(data.attrs)
    result.attrs["domain"] = domain
    return result


def _csi_acquisition_order(sequence, kspace_values):
    """Return the measured-centre acquisition permutation of the CSI points."""
    if getattr(sequence, "handler", None) is None:
        raise ValueError(
            "Cannot resolve CSI spatial acquisition order without handler."
        )
    from ..analysis.spectroscopy.signal_decay import csi_order_indices

    acquisition = sequence.handler.to_csi_acquisition()
    dims = list(_KSPACE_DIMS) + [
        f"group_{axis}" for axis in range(3, kspace_values.ndim)
    ]
    return csi_order_indices(
        xr.DataArray(kspace_values, dims=dims),
        acquisition.read_positions,
        acquisition.phase_positions,
    )


def _acquisition_order_or_fallback(sequence, kspace_values, spatial_order_mode):
    """Resolve the permutation and the effective order mode name."""
    n_points = kspace_values.shape[1] * kspace_values.shape[2]
    if spatial_order_mode == "native":
        return np.arange(n_points), "native_order"
    try:
        return _csi_acquisition_order(sequence, kspace_values), "encoding_order"
    except Exception as order_error:
        logger.debug(
            "Failed to get CSI encoding order for filter_signal; reason: %s",
            order_error,
        )
        if spatial_order_mode == "csi_acquisition_order":
            raise ValueError(
                "CSI acquisition order was requested but could not be "
                "resolved from phase-encoding metadata."
            ) from order_error
        # Magnitude sorting is not a valid substitute for acquisition order:
        # it changes which points count as acquired before signal_end. Auto
        # mode keeps the native flattened k-space order instead.
        return np.arange(n_points), "native_order_fallback"


def filter_kspace_in_acquired_order(data, sequence, window, spatial_order_mode):
    """Filter labelled CSI k-space points in the order they were acquired.

    Parameters
    ----------
    data : xarray.DataArray
        CSI samples with ``acquisition_idx``, ``phase_idx`` and ``read_idx``
        and optional further dimensions in any order. ``domain="spatial"``
        is moved to k-space first and returned to the spatial domain after
        filtering; other domains are treated as k-space.
    sequence : object
        Owner of ``handler.to_csi_acquisition()``, which supplies the
        acquired encoding positions.
    window : SampleWindow
        Window and truncation along the acquired points; ``signal_end``
        counts FIDs.
    spatial_order_mode : {"auto", "native", "csi_acquisition_order"}
        Order source; "auto" falls back to native storage order when the
        acquisition description is unavailable.

    Returns
    -------
    xarray.DataArray
        Filtered samples with the input dimensions, coordinates and
        attributes, plus ``filter_fid_order_mode`` and
        ``filter_kspace_order_mode``.

    Raises
    ------
    ValueError
        If "csi_acquisition_order" is requested and cannot be resolved.
    """
    from ..reconstruction.centred_dft import centered_fft, centered_ifft

    was_spatial = str(data.attrs.get("domain", "")).lower() == "spatial"
    kspace = (
        _transform_spatial_dims(data, centered_fft, "kspace")
        if was_spatial
        else data.copy(deep=True)
    )
    working = kspace.transpose(
        *_KSPACE_DIMS, *[d for d in kspace.dims if d not in _KSPACE_DIMS]
    )
    values = np.asarray(working.values).copy()
    n_acquisition, n_phase, n_read, *trailing = values.shape
    n_groups = int(np.prod(trailing)) if trailing else 1
    points = values.reshape(n_acquisition, n_phase * n_read, n_groups)

    order, order_mode = _acquisition_order_or_fallback(
        sequence, values, spatial_order_mode
    )
    filtered = window_and_truncate(np.take(points, order, axis=1), 1, window)
    filtered = np.take(filtered, np.argsort(order), axis=1)

    result = working.copy(data=filtered.reshape(values.shape)).transpose(*kspace.dims)
    result.attrs = dict(kspace.attrs)
    if was_spatial:
        result = _transform_spatial_dims(result, centered_ifft, "spatial")
    result = result.transpose(*data.dims)
    result.attrs = dict(data.attrs)
    result.attrs["filter_fid_order_mode"] = order_mode
    result.attrs["filter_kspace_order_mode"] = order_mode
    return result


def filter_along_dim(data, dim, window):
    """Filter labelled samples along one named dimension.

    Parameters
    ----------
    data : xarray.DataArray
        Samples of any dimensions; not mutated.
    dim : str
        Dimension along which the window and truncation apply.
    window : SampleWindow
        Window and truncation settings.

    Returns
    -------
    xarray.DataArray
        Filtered samples with the dimensions, coordinates and attributes of
        ``data``.
    """
    filtered = window_and_truncate(
        np.asarray(data.values), data.get_axis_num(dim), window
    )
    return data.copy(data=filtered)


def reconstruct_filtered_raw(sequence, filtered_vector, spatial_provenance):
    """Reconstruct filtered raw CSI samples with the canonical CSI strategy.

    Parameters
    ----------
    sequence : object
        Owner of ``handler`` and, optionally, ``data`` whose dimension
        order the result follows.
    filtered_vector : numpy.ndarray
        Filtered raw vector in acquired point/FID sample order.
    spatial_provenance : dict or None
        Physical spatial-filter provenance stored as ``spatial_filter``.

    Returns
    -------
    xarray.DataArray
        Reconstructed CSI data; the handler is shallow-copied, not changed.

    Raises
    ------
    ValueError
        If no handler or raw source exists or the reconstruction returns None.
    """
    from ..reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy

    if getattr(sequence, "handler", None) is None:
        raise ValueError(
            "Cannot reconstruct filtered data: no handler available on object."
        )
    handler = copy.copy(sequence.handler)
    if hasattr(handler, "rawdatajob0"):
        handler.rawdatajob0 = filtered_vector
    elif hasattr(handler, "fid"):
        handler.fid = filtered_vector
    else:
        raise ValueError(
            "Cannot reconstruct filtered data: handler has neither rawdatajob0 nor fid."
        )

    reconstructed = BrukerCsiRecoStrategy().reconstruct(handler=handler)
    if reconstructed is None:
        raise ValueError("Reconstruction of filtered data returned None.")

    reco_data = getattr(sequence, "data", None)
    if reco_data is not None:
        leading = [d for d in reco_data.dims if d in reconstructed.dims]
        if leading:
            reconstructed = reconstructed.transpose(
                *leading, *[d for d in reconstructed.dims if d not in leading]
            )
    if spatial_provenance is not None:
        reconstructed = reconstructed.assign_attrs(spatial_filter=spatial_provenance)
    return reconstructed


def normalize_axis(axis, ndim):
    """Return ``axis`` as a non-negative index for an array of ``ndim`` axes.

    Raises
    ------
    ValueError
        If the axis is outside the array.
    """
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"filter_axis={axis} out of bounds for input with ndim={ndim}")
    return axis


def resolve_filter_dim(data, filter_dim, filter_axis):
    """Return the dimension filtered when acquired-order filtering is not used.

    An explicit ``filter_dim`` must exist. Otherwise ``filter_axis`` selects a
    dimension by position, except that ``filter_axis=0`` selects
    ``acquisition_idx`` wherever that dimension is.

    Raises
    ------
    ValueError
        If ``filter_dim`` is missing from ``data`` or ``filter_axis`` is out
        of bounds.
    """
    if filter_dim is not None:
        if filter_dim not in data.dims:
            raise ValueError(
                f"filter_dim='{filter_dim}' not found in input dims {data.dims}"
            )
        return filter_dim
    dim = data.dims[normalize_axis(filter_axis, data.ndim)]
    if "acquisition_idx" in data.dims and filter_axis == 0:
        return "acquisition_idx"
    return dim


def raw_samples_from_handler(handler):
    """Return a copy of the handler's raw CSI samples.

    ``rawdatajob0`` is preferred over ``fid``; the handler is not changed.

    Raises
    ------
    ValueError
        If the handler holds neither.
    """
    for name in ("rawdatajob0", "fid"):
        samples = getattr(handler, name, None)
        if samples is not None:
            logger.debug("Using %s from handler", name)
            return samples.copy()
    raise ValueError(
        "No raw data available for filtering. Ensure CSI data is loaded properly."
    )


def filter_raw_fids(values, n_acquisition, filter_axis, window, n_repetitions):
    """Filter a raw CSI vector or matrix and return the flattened raw layout.

    Parameters
    ----------
    values : numpy.ndarray
        Either a vector in acquired point/FID sample order, reshaped with
        ``n_acquisition`` samples per FID, or a matrix of shape
        (FID sample, acquired point).
    n_acquisition : int
        FID samples per acquired point; used for vectors only.
    filter_axis : int
        Matrix axis filtered: 1 for acquired points, 0 for FID samples.
    window : SampleWindow
        Window and truncation settings.
    n_repetitions : int
        Repetition blocks along the acquired points; truncation keeps
        ``signal_end`` points per block when the count divides them.

    Returns
    -------
    numpy.ndarray
        Filtered vector in acquired point/FID sample order.
    """
    if values.ndim == 1:
        n_points = len(values) // n_acquisition
        values = np.reshape(values, (n_points, n_acquisition)).T
    axis = normalize_axis(filter_axis, values.ndim)
    n_blocks = n_repetitions if axis == 1 else 1
    return window_and_truncate(values, axis, window, n_blocks).T.flatten()
