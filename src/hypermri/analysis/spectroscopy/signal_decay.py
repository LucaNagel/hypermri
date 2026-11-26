"""CSI acquisition-order diagnostics without widgets or rendering.

These operations retain the signal-decay tool's mean-magnitude, normalized
cumulative sum and linear tail-fit conventions. Spectral band power is a second
per-point quantity that measures only the frequencies where metabolites sit.
They do not filter samples.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr

from hypermri.utils.mri.fourier import spectral_fft


def csi_order_indices(
    kspace: xr.DataArray, read_indices: np.ndarray, phase_indices: np.ndarray
) -> np.ndarray:
    """Resolve the existing measured-centre CSI acquisition permutation.

    Parameters
    ----------
    kspace : xarray.DataArray
        Complex k-space samples with read_idx and phase_idx. All other dimensions
        contribute to the magnitude metric locating the measured centre. Labels,
        coordinates, units and storage dtype are preserved without mutation.
    read_indices, phase_indices : numpy.ndarray
        Scanner encoding positions, one per acquired FID. Their order is retained
        and translated by the measured minus nominal centre, modulo matrix size.

    Returns
    -------
    numpy.ndarray
        Integer permutation into flattened (phase_idx, read_idx) storage.

    Raises
    ------
    ValueError
        If positions do not form an in-bounds permutation of the spatial grid,
        or no finite magnitude is available to locate the centre.

    Notes
    -----
    Deterministic; uses the existing first-maximum tie rule. This retains the
    filtering convention and makes no new scanner encoding correction.

    Examples
    --------
    >>> grid = xr.DataArray([[1, 0], [0, 0]], dims=("phase_idx", "read_idx"))
    >>> csi_order_indices(grid, np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
    array([3, 2, 1, 0])
    """
    n_phase, n_read = kspace.sizes["phase_idx"], kspace.sizes["read_idx"]
    read = np.asarray(read_indices).reshape(-1).astype(int)
    phase = np.asarray(phase_indices).reshape(-1).astype(int)
    if read.size != n_phase * n_read or phase.size != read.size:
        raise ValueError("CSI encoding order size must match the read/phase grid.")
    if np.any((read < 0) | (read >= n_read)) or np.any(
        (phase < 0) | (phase >= n_phase)
    ):
        raise ValueError("CSI encoding order indices are out of bounds.")
    working = kspace.transpose("phase_idx", "read_idx", ...)
    axes = tuple(range(2, working.ndim))
    metric = np.mean(np.abs(working.values), axis=axes)
    measured_phase, measured_read = np.unravel_index(np.nanargmax(metric), metric.shape)
    read = (read + measured_read - n_read // 2) % n_read
    phase = (phase + measured_phase - n_phase // 2) % n_phase
    order = np.ravel_multi_index((phase, read), (n_phase, n_read))
    if np.unique(order).size != order.size:
        raise ValueError("CSI encoding order must form a spatial permutation.")
    return order


def _band_limits(
    bands_hz: Sequence[tuple[float, float]] | None,
) -> tuple[tuple[float, float], ...] | None:
    """Validate frequency bands as finite (low, high) pairs in Hz."""
    if bands_hz is None:
        return None
    limits = []
    for band in bands_hz:
        try:
            low, high = (float(value) for value in band)
        except (TypeError, ValueError):
            raise ValueError(
                f"Spectral band {band!r} must be a (low, high) pair in Hz."
            ) from None
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError(
                f"Spectral band ({low:g}, {high:g}) Hz needs finite limits with low < high."
            )
        limits.append((low, high))
    if not limits:
        raise ValueError(
            "Spectral bands must contain at least one (low, high) pair in Hz; "
            "pass None for the whole spectrum."
        )
    return tuple(limits)


def _band_selection(
    n_samples: int,
    dwell_time_s: float,
    bands_hz: Sequence[tuple[float, float]] | None,
) -> np.ndarray:
    """Select the shifted-spectrum samples inside inclusive bands in Hz.

    The frequency axis is ``fftshift(fftfreq(n_samples, dwell_time_s))``; None
    selects every sample. Raises ValueError for invalid or empty bands.
    """
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(n_samples, dwell_time_s))
    limits = _band_limits(bands_hz)
    if limits is None:
        return np.ones(n_samples, dtype=bool)
    selected = np.zeros(n_samples, dtype=bool)
    for low, high in limits:
        selected |= (frequency_hz >= low) & (frequency_hz <= high)
    if not selected.any():
        raise ValueError(
            f"Spectral bands {limits} Hz contain no frequency sample of the "
            f"axis {frequency_hz[0]:.1f} to {frequency_hz[-1]:.1f} Hz."
        )
    return selected


def spectral_band_power(
    matrix: np.ndarray,
    *,
    dwell_time_s: float,
    bands_hz: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Calculate the mean spectral power inside frequency bands per acquired point.

    Each column holds one acquired FID. Its spectrum is the shifted FFT used by
    the spectroscopy viewer, on the frequency axis
    ``fftshift(fftfreq(n_samples, dwell_time_s))`` in Hz. The power of the
    selected frequency samples is averaged and divided by the number of FID
    samples. Restricting the bands to the metabolite lines detects weak late
    signal that the whole-FID magnitude averages away, because noise outside
    the bands no longer contributes.

    Parameters
    ----------
    matrix : numpy.ndarray
        Complex (FID sample, acquired k-space point) values in acquired signal
        units. FID samples start at time zero.
    dwell_time_s : float
        Positive FID sampling interval in seconds.
    bands_hz : sequence of (float, float) or None, optional
        Inclusive (low, high) frequency ranges in Hz on the axis above, for
        example ``[(-330, -150), (200, 420)]``. Overlapping ranges select each
        frequency sample once. None selects the whole spectral width.

    Returns
    -------
    numpy.ndarray
        Float64 power per acquired point, shape (n_points,), in squared signal
        units. Pure complex white noise has the expected value
        ``E|n|^2 = 2 sigma^2`` (sigma per real component) for any band. With
        ``bands_hz=None`` the value equals the mean ``|s|^2`` over the FID
        (Parseval's theorem).

    Raises
    ------
    ValueError
        For an empty or non-matrix input, a non-positive or non-finite dwell
        time, an invalid band, or bands that contain no frequency sample.

    Notes
    -----
    Eager, deterministic and free of input mutation and caching.

    Examples
    --------
    >>> t = np.arange(8)
    >>> fid = np.exp(2j * np.pi * 250.0 * t * 1e-3)[:, None]
    >>> spectral_band_power(fid, dwell_time_s=1e-3, bands_hz=[(200, 300)])
    array([8.])
    >>> power = spectral_band_power(fid, dwell_time_s=1e-3, bands_hz=[(-300, -200)])
    >>> bool(np.allclose(power, 0.0))
    True
    """
    values = np.asarray(matrix)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(
            "Spectral band power requires a nonempty (FID sample, acquired point) matrix."
        )
    dwell = float(dwell_time_s)
    if not (np.isfinite(dwell) and dwell > 0):
        raise ValueError(
            f"Spectral band power requires a positive dwell_time_s, got {dwell_time_s!r}."
        )
    n_samples = values.shape[0]
    selected = _band_selection(n_samples, dwell, bands_hz)
    spectrum = spectral_fft(values.astype(np.complex128, copy=False), axis=0)
    return np.mean(np.abs(spectrum[selected]) ** 2, axis=0) / n_samples


def _repetition_block(
    matrix: np.ndarray, n_repetitions: int, repetition_index: int
) -> np.ndarray:
    """Return one repetition of a (FID sample, acquired point) matrix.

    Repetitions are contiguous equal blocks along the second axis. A matrix
    whose point count is not divisible by ``n_repetitions`` is returned whole,
    the legacy decay diagnostic.
    """
    values = np.asarray(matrix)
    if values.ndim != 2 or 0 in values.shape or n_repetitions < 1:
        raise ValueError(
            "Signal decay requires a nonempty matrix and positive repetition count."
        )
    if values.shape[1] % n_repetitions == 0:
        if not 0 <= repetition_index < n_repetitions:
            raise ValueError(
                "Signal decay repetition_index is outside the acquired blocks."
            )
        values = values.reshape(values.shape[0], n_repetitions, -1)[:, repetition_index]
    return values


def signal_decay_curves(
    matrix: np.ndarray,
    *,
    n_repetitions: int = 1,
    repetition_index: int = 0,
    end_index: int = 0,
    subtract_tail: bool = False,
    quantity: str = "magnitude",
    dwell_time_s: float | None = None,
    bands_hz: Sequence[tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Calculate per-point and cumulative CSI decay curves with their linear tails.

    Parameters
    ----------
    matrix : numpy.ndarray
        Complex or real (FID sample, acquired k-space point) values. Repetitions
        are contiguous equal blocks along the second axis. Mean absolute values
        retain signal units; the normalized cumulative sum is dimensionless.
    n_repetitions : int, optional
        Positive number of blocks. An indivisible matrix retains the legacy
        whole-matrix diagnostic rather than splitting points incorrectly.
    repetition_index : int, optional
        Zero-based repetition to display, within the available block range.
    end_index : int, optional
        Start of the linear tail fit, clipped to the displayed curve's bounds.
        This does not truncate or filter input data.
    subtract_tail : bool, optional
        Subtract each curve's own extrapolated linear tail for display only.
    quantity : {"magnitude", "band_power"}, optional
        Per-point value: the mean absolute value over the FID (default) or
        :func:`spectral_band_power` of the selected repetition block.
    dwell_time_s : float or None, optional
        FID sampling interval in seconds; required for ``"band_power"`` and
        ignored for ``"magnitude"``.
    bands_hz : sequence of (float, float) or None, optional
        Frequency ranges in Hz for ``"band_power"``; None selects the whole
        spectral width. Ignored for ``"magnitude"``.

    Returns
    -------
    dict of str to numpy.ndarray
        The per-point curve under the ``quantity`` name, ``cumulative``, and
        their fitted lines under ``<name>_tail``. Fits use float64 arithmetic
        and are empty with fewer than two finite tail points. Subtracted fitted
        lines are zero. Magnitude/reduction retain NumPy's existing calculation
        precision; band power is float64; input storage is unchanged.

    Raises
    ------
    ValueError
        For empty/non-matrix input, invalid repetition count or block selection,
        an unknown quantity, or invalid band-power sampling and bands.

    Notes
    -----
    Eager, deterministic and free of input mutation, caching and UI dependencies.
    Cumulative sums are normalized only when their maximum is positive.

    Examples
    --------
    >>> curves = signal_decay_curves(np.array([[2j, 4j], [2, 4]]))
    >>> curves["magnitude"]
    array([2., 4.])
    """
    values = _repetition_block(matrix, n_repetitions, repetition_index)
    if quantity == "magnitude":
        per_point = np.mean(np.abs(values), axis=0)
    elif quantity == "band_power":
        if dwell_time_s is None:
            raise ValueError("Signal decay band power requires dwell_time_s.")
        per_point = spectral_band_power(
            values, dwell_time_s=dwell_time_s, bands_hz=bands_hz
        )
    else:
        raise ValueError(
            f"Signal decay quantity {quantity!r} is unknown; use 'magnitude' or 'band_power'."
        )
    cumulative = np.cumsum(per_point)
    if np.max(cumulative) > 0:
        cumulative = cumulative / np.max(cumulative)
    result = {}
    for name, curve in ((quantity, per_point), ("cumulative", cumulative)):
        x = np.arange(curve.size)
        start = int(np.clip(end_index, 0, curve.size - 1))
        finite = np.isfinite(curve[start:])
        tail = np.array([])
        if np.count_nonzero(finite) >= 2:
            slope, intercept = np.polyfit(
                x[start:][finite], np.asarray(curve[start:], dtype=float)[finite], 1
            )
            tail = slope * x + intercept
        result[name] = curve - tail if subtract_tail and tail.size else curve
        result[name + "_tail"] = np.zeros_like(tail) if subtract_tail else tail
    return result
