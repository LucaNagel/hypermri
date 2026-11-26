"""Frequency alignment of spectroscopic data from fitted peak positions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from hypermri.utils.mri.fourier import spectral_fft, spectral_ifft

__all__ = [
    "align_spectra_by_fit_frequency",
    "shift_frequency",
]


def _sampling_time_axis_seconds(
    npoints: int,
    *,
    sampling_time_axis_s: np.ndarray | None,
    dwell_time_s: float | None,
) -> np.ndarray:
    """Resolve a sampling-time axis for phase-ramp frequency shifting.

    An explicit sampling-time axis takes precedence over the dwell time. A
    shorter axis is extended with its median spacing; a longer one is cut.
    """
    if npoints <= 0:
        raise ValueError("Cannot frequency-shift data with an empty sampling axis.")

    if sampling_time_axis_s is not None:
        axis = np.asarray(sampling_time_axis_s, dtype=float).reshape(-1)
        if axis.size >= npoints:
            return axis[:npoints]
        if axis.size > 1:
            dt = float(np.median(np.diff(axis)))
            if np.isfinite(dt) and dt > 0:
                tail = axis[-1] + dt * np.arange(1, npoints - axis.size + 1)
                return np.concatenate([axis, tail])
        raise ValueError(
            "sampling_time_axis_s must contain at least two finite, increasing "
            "samples when it is shorter than the sampling axis."
        )

    if dwell_time_s is None:
        raise ValueError(
            "Could not determine sampling time axis. Provide "
            "'sampling_time_axis_s' or 'dwell_time_s'."
        )

    dwell_time_s = float(dwell_time_s)
    if not np.isfinite(dwell_time_s) or dwell_time_s <= 0:
        raise ValueError(f"Invalid dwell_time_s={dwell_time_s}.")
    return np.arange(npoints, dtype=float) * dwell_time_s


def shift_frequency(
    values: np.ndarray,
    offsets_hz: float | np.ndarray,
    *,
    sampling_time_axis_s: np.ndarray | None = None,
    dwell_time_s: float | None = None,
    axis: int = -1,
    signal_domain: str = "temporal",
) -> np.ndarray:
    """Shift spectra in frequency by a phase ramp on the FID.

    Each FID is multiplied by ``exp(2j * pi * offset_hz * t)``, which moves
    every peak by ``offset_hz`` without interpolation. Spectral input is
    transformed to the time domain with
    :func:`hypermri.utils.mri.fourier.spectral_ifft`, shifted, and transformed
    back with :func:`hypermri.utils.mri.fourier.spectral_fft`. This is the
    plain NumPy operation behind :func:`align_spectra_by_fit_frequency`; it
    reads no metadata and works on any array layout.

    Parameters
    ----------
    values : numpy.ndarray
        Complex signal with one sampling axis (selected by ``axis``) and any
        number of independent other axes, for example voxels or repetitions.
    offsets_hz : float or numpy.ndarray
        Frequency shift in Hz. Positive values move peaks to higher
        frequencies. A scalar shifts every spectrum by the same amount; an
        array gives one shift per spectrum and must broadcast to the shape of
        ``values`` without its sampling axis.
    sampling_time_axis_s : numpy.ndarray, optional
        Sampling times in seconds, starting at the first acquired point. It
        takes precedence over ``dwell_time_s``. A longer axis is cut to the
        number of samples; a shorter axis with at least two increasing
        samples is extended with its median spacing.
    dwell_time_s : float, optional
        Time between samples in seconds, used as ``t = n * dwell_time_s``
        when ``sampling_time_axis_s`` is not given. Must be finite and
        positive.
    axis : int, optional
        Sampling axis of ``values`` (time points or frequency bins).
    signal_domain : {"temporal", "spectral"}, optional
        Whether ``values`` holds FIDs or centred spectra in ``fftshift``
        order, case-insensitive. The result is returned in the same domain.

    Returns
    -------
    numpy.ndarray
        Shifted signal with the shape and axis order of ``values`` and a
        complex dtype. The input is not mutated. Evaluation is eager and
        deterministic.

    Raises
    ------
    ValueError
        If ``signal_domain`` is unknown, the sampling axis is empty, no valid
        timing is given, or ``offsets_hz`` does not broadcast to the spectra.

    Examples
    --------
    Shift a single FID by 10 Hz and a stack of spectra by one offset each::

        shifted_fid = shift_frequency(fid, 10.0, dwell_time_s=1e-4)
        shifted_spectra = shift_frequency(
            spectra,  # shape (n_voxels, n_points)
            offsets_hz=np.array([5.0, -3.0, 0.0]),
            dwell_time_s=1e-4,
            signal_domain="spectral",
        )
    """
    domain = str(signal_domain).lower()
    if domain not in ("temporal", "spectral"):
        raise ValueError(
            f"signal_domain must be 'temporal' or 'spectral', got {signal_domain!r}."
        )

    values = np.moveaxis(np.asarray(values), axis, -1)
    time_axis_s = _sampling_time_axis_seconds(
        values.shape[-1],
        sampling_time_axis_s=sampling_time_axis_s,
        dwell_time_s=dwell_time_s,
    )
    offsets = np.asarray(offsets_hz, dtype=float)
    try:
        np.broadcast_shapes(offsets.shape, values.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            f"offsets_hz with shape {offsets.shape} does not broadcast to the "
            f"spectra shape {values.shape[:-1]} (values without its sampling axis)."
        ) from exc

    temporal = spectral_ifft(values, axis=-1) if domain == "spectral" else values
    phase_ramp = np.exp(1j * 2.0 * np.pi * offsets[..., np.newaxis] * time_axis_s)
    shifted = temporal * phase_ramp
    if domain == "spectral":
        shifted = spectral_fft(shifted, axis=-1)
    return np.moveaxis(shifted, -1, axis)


def _compute_weighted_frequency_alignment(
    fit_results: xr.Dataset,
) -> tuple[float, xr.DataArray, int]:
    """Return shared fitted frequency and its per-voxel offsets."""
    required = ("fit_freqs", "fit_amps")
    missing = [name for name in required if name not in fit_results]
    if missing:
        raise ValueError(
            "fit_results must contain both 'fit_freqs' and 'fit_amps'; "
            f"missing {missing}."
        )

    fit_freqs, fit_amps = xr.align(
        fit_results["fit_freqs"], fit_results["fit_amps"], join="exact"
    )
    if "metabolite" not in fit_freqs.dims or "metabolite" not in fit_amps.dims:
        raise ValueError("fit_freqs and fit_amps must include a 'metabolite' dim.")

    weights = np.abs(fit_amps)
    valid = np.isfinite(fit_freqs) & np.isfinite(weights) & (weights > 0)
    weights = xr.where(valid, weights, 0.0)
    weighted_freq = xr.where(valid, fit_freqs * weights, 0.0)

    total_weight = float(weights.sum(skipna=True).values)
    if total_weight <= 0:
        raise ValueError(
            "No valid fitted amplitudes/frequencies available for weighted averaging."
        )

    global_mean_hz = float(weighted_freq.sum(skipna=True).values / total_weight)
    voxel_weight = weights.sum(dim="metabolite", skipna=True)
    voxel_frequency_hz = xr.where(
        voxel_weight > 0,
        weighted_freq.sum(dim="metabolite", skipna=True) / voxel_weight,
        np.nan,
    )
    offsets_hz = xr.where(
        np.isfinite(voxel_frequency_hz), global_mean_hz - voxel_frequency_hz, 0.0
    )
    n_valid_voxels = int(np.count_nonzero(np.asarray(voxel_weight.values) > 0))
    return global_mean_hz, offsets_hz, n_valid_voxels


def _validated_frequency_offsets(
    frequency_offsets_hz: object,
    *,
    source_name: str,
) -> xr.DataArray:
    """Return a defensive copy of explicitly supplied frequency offsets."""
    if not isinstance(frequency_offsets_hz, xr.DataArray):
        raise TypeError(f"{source_name} must be an xarray.DataArray.")
    if not np.all(np.isfinite(np.asarray(frequency_offsets_hz.values))):
        raise ValueError(f"{source_name} must contain only finite values.")
    return frequency_offsets_hz.copy(deep=True)


def _resolve_frequency_alignment(
    fit_results: xr.Dataset,
    *,
    frequency_offsets_hz: xr.DataArray | None,
    alignment_info: Mapping[str, object] | None,
) -> tuple[float, xr.DataArray, int, dict[str, object]]:
    """Resolve computed or explicitly supplied alignment state."""
    if frequency_offsets_hz is not None and alignment_info is not None:
        raise ValueError(
            "Pass either frequency_offsets_hz or alignment_info, not both."
        )

    if alignment_info is not None:
        required = (
            "global_mean_frequency_hz",
            "frequency_offsets_hz",
            "n_valid_voxels",
        )
        missing = [name for name in required if name not in alignment_info]
        if missing:
            raise ValueError(f"alignment_info is missing required entries: {missing}.")

        global_mean_hz = float(alignment_info["global_mean_frequency_hz"])
        if not np.isfinite(global_mean_hz):
            raise ValueError(
                "alignment_info['global_mean_frequency_hz'] must be finite."
            )
        offsets_hz = _validated_frequency_offsets(
            alignment_info["frequency_offsets_hz"],
            source_name="alignment_info['frequency_offsets_hz']",
        )
        n_valid_voxels = int(alignment_info["n_valid_voxels"])
        resolved_info = dict(alignment_info)
        resolved_info["global_mean_frequency_hz"] = global_mean_hz
        resolved_info["frequency_offsets_hz"] = offsets_hz
        resolved_info["n_valid_voxels"] = n_valid_voxels
        return global_mean_hz, offsets_hz, n_valid_voxels, resolved_info

    global_mean_hz, computed_offsets_hz, n_valid_voxels = (
        _compute_weighted_frequency_alignment(fit_results)
    )
    offsets_hz = (
        computed_offsets_hz
        if frequency_offsets_hz is None
        else _validated_frequency_offsets(
            frequency_offsets_hz,
            source_name="frequency_offsets_hz",
        )
    )
    resolved_info = {
        "global_mean_frequency_hz": global_mean_hz,
        "frequency_offsets_hz": offsets_hz,
        "n_valid_voxels": n_valid_voxels,
    }
    return global_mean_hz, offsets_hz, n_valid_voxels, resolved_info


def _broadcast_offsets(
    offsets_hz: xr.DataArray,
    target: xr.DataArray,
    *,
    allow_new_dims: bool = False,
) -> xr.DataArray:
    """Broadcast per-voxel offsets to a data or fit-parameter target."""
    extra_dims = [dim for dim in offsets_hz.dims if dim not in target.dims]
    non_singleton_extra_dims = [dim for dim in extra_dims if offsets_hz.sizes[dim] != 1]
    if non_singleton_extra_dims and not allow_new_dims:
        raise ValueError(
            "Frequency offsets have dimensions not present in the target: "
            f"{non_singleton_extra_dims}."
        )
    if extra_dims and not allow_new_dims:
        offsets_hz = offsets_hz.isel({dim: 0 for dim in extra_dims}, drop=True)
    try:
        return offsets_hz.broadcast_like(target)
    except ValueError as exc:
        raise ValueError(
            "Could not align frequency offsets with the target coordinates."
        ) from exc


_APPLIED_OFFSET_COORDINATE = "frequency_alignment_offset_hz"


def _with_applied_offsets(target, offsets_hz, template):
    """Add per-voxel offsets to the coordinate of the total applied shift.

    ``template`` holds the voxel dimensions of ``target`` in their order. The
    coordinate accumulates over repeated alignments, so applying the negative
    offsets of an earlier call returns it to zero.
    """
    offsets = _broadcast_offsets(offsets_hz, template).transpose(*template.dims)
    values = np.asarray(offsets.values, dtype=np.float64)
    if _APPLIED_OFFSET_COORDINATE in target.coords:
        previous = target.coords[_APPLIED_OFFSET_COORDINATE]
        values = values + np.asarray(
            previous.transpose(*template.dims).values, dtype=np.float64
        )
    applied = xr.DataArray(
        values,
        dims=template.dims,
        attrs={
            "unit": "Hz",
            "definition": (
                "total frequency shift applied to this voxel by "
                "align_spectra_by_fit_frequency; fitted frequencies of the "
                "unaligned data are fit_freqs - frequency_alignment_offset_hz"
            ),
        },
    )
    return target.assign_coords({_APPLIED_OFFSET_COORDINATE: applied})


def _shift_data(
    data: xr.DataArray,
    offsets_hz: xr.DataArray,
    *,
    global_mean_hz: float,
    sampling_time_axis_s: np.ndarray | None,
    dwell_time_s: float | None,
    signal_domain: str | None = None,
) -> xr.DataArray:
    """Apply per-voxel offsets with :func:`shift_frequency` on labelled data.

    Missing timing and ``signal_domain`` are read from ``data.attrs``.
    Dimension order, coordinates and attributes are preserved;
    ``global_mean_hz`` is only recorded as provenance.
    """
    spectral_dim = "acquisition_idx"
    if spectral_dim not in data.dims:
        raise ValueError(
            f"Input data is missing required spectral dimension '{spectral_dim}'."
        )

    loop_dims = [dim for dim in data.dims if dim != spectral_dim]
    aligned = data.transpose(*loop_dims, spectral_dim)
    offsets = _broadcast_offsets(offsets_hz, aligned.isel({spectral_dim: 0}, drop=True))

    if signal_domain is None:
        attr_domain = str(data.attrs.get("signal_domain", "temporal")).lower()
        signal_domain = "spectral" if attr_domain == "spectral" else "temporal"
    if sampling_time_axis_s is None:
        sampling_time_axis_s = data.attrs.get("sampling_time_axis_s")
    if sampling_time_axis_s is None and dwell_time_s is None:
        dwell_time_s = data.attrs.get("dwell_time_s")

    shifted_values = shift_frequency(
        aligned.values,
        offsets.values,
        sampling_time_axis_s=sampling_time_axis_s,
        dwell_time_s=dwell_time_s,
        signal_domain=signal_domain,
    )

    shifted = aligned.copy(data=shifted_values).transpose(*data.dims)
    shifted.attrs = dict(data.attrs)
    shifted.attrs["frequency_alignment_method"] = "fit_amp_weighted_mean"
    shifted.attrs["frequency_alignment_global_mean_hz"] = float(global_mean_hz)
    return _with_applied_offsets(
        shifted, offsets_hz, data.isel({spectral_dim: 0}, drop=True)
    )


def _shift_fit_results(
    fit_results: xr.Dataset,
    offsets_hz: xr.DataArray,
) -> xr.Dataset:
    """Shift fitted frequencies without mutating the caller's dataset."""
    updated = fit_results.copy(deep=True)
    updated["fit_freqs"] = updated["fit_freqs"] + _broadcast_offsets(
        offsets_hz, updated["fit_freqs"]
    )
    return _with_applied_offsets(
        updated, offsets_hz, updated["fit_freqs"].isel(metabolite=0, drop=True)
    )


def _shift_fit_params(
    fit_params: xr.Dataset | None,
    offsets_hz: xr.DataArray,
) -> xr.Dataset | None:
    """Shift fitted frequency initial values and bounds, if present."""
    if fit_params is None:
        return None

    updated = fit_params.copy(deep=True)
    if "initial_freqs_hz" not in updated:
        return updated

    param_offsets = _broadcast_offsets(
        offsets_hz, updated["initial_freqs_hz"], allow_new_dims=True
    )
    updated["initial_freqs_hz"] = updated["initial_freqs_hz"] + param_offsets
    for name in ("freq_range_lower_bound_hz", "freq_range_upper_bound_hz"):
        if name in updated:
            updated[name] = updated[name] + _broadcast_offsets(
                offsets_hz, updated[name], allow_new_dims=True
            )
    return updated


def align_spectra_by_fit_frequency(
    data: xr.DataArray,
    fit_results: xr.Dataset,
    fit_params: xr.Dataset | None = None,
    *,
    sampling_time_axis_s: np.ndarray | None = None,
    dwell_time_s: float | None = None,
    frequency_offsets_hz: xr.DataArray | None = None,
    alignment_info: Mapping[str, object] | None = None,
    signal_domain: str | None = None,
) -> tuple[xr.DataArray, xr.Dataset, xr.Dataset | None, dict[str, object]]:
    """Align spectra to the amplitude-weighted mean fitted frequency.

    Every voxel is shifted in the temporal domain by the difference between
    its amplitude-weighted fitted frequency and the global weighted mean.
    The shift itself is :func:`shift_frequency`, which also works on plain
    NumPy arrays.  Spectral-domain input is transformed to temporal data and
    back automatically.  The function returns copies and never mutates its inputs.

    Parameters
    ----------
    data
        Spectroscopy data with an ``acquisition_idx`` dimension.  Set its
        ``signal_domain`` attribute to ``"spectral"`` when the input already
        represents spectra; all other values are treated as temporal data.
    fit_results
        Dataset containing voxel-wise ``fit_freqs`` and complex or real
        ``fit_amps``, each with a ``metabolite`` dimension.
    fit_params
        Optional fitting parameters to shift along with the data.  Frequency
        initial values and bounds are updated when present.
    sampling_time_axis_s, dwell_time_s
        Optional explicit timing metadata.  They override the corresponding
        attributes on ``data`` and are useful when timing lives on a
        parent scan object.
    frequency_offsets_hz
        Optional per-voxel offsets to apply instead of estimating new offsets
        from ``fit_results``.  Pass the negative offsets from an earlier call
        to undo that alignment.  Cannot be combined with ``alignment_info``.
    alignment_info
        Optional alignment state returned by an earlier call.  Its stored
        offsets and diagnostic values are reused without recomputing them.
        Cannot be combined with ``frequency_offsets_hz``.
    signal_domain
        ``"temporal"`` or ``"spectral"`` (case-insensitive). Overrides the
        ``signal_domain`` attribute of ``data`` when given.

    Returns
    -------
    aligned_data : xarray.DataArray
        Frequency-aligned data in the same domain and dimension order as the
        input. The coordinate ``frequency_alignment_offset_hz`` (float64, Hz,
        voxel dimensions in input order) holds the total shift applied so far:
        it adds up over repeated calls and returns to zero when an alignment
        is undone. Fits of this data inherit the coordinate.
    aligned_fit_results : xarray.Dataset
        Copy of ``fit_results`` with ``fit_freqs`` shifted to the shared
        reference frequency and the same ``frequency_alignment_offset_hz``
        coordinate.
    aligned_fit_params : xarray.Dataset or None
        Copy of ``fit_params`` with initial frequencies and bounds shifted by
        the same per-voxel offsets, or ``None`` if no parameters were passed.
    alignment_info : dict
        ``global_mean_frequency_hz`` (float), ``frequency_offsets_hz``
        (xarray.DataArray), and ``n_valid_voxels`` (int).

    Examples
    --------
    Align data and retain the adjusted fitting state and diagnostic values::

        aligned_data, aligned_results, aligned_params, alignment_info = (
            align_spectra_by_fit_frequency(data, fit_results, fit_params)
        )
        mean_hz = alignment_info["global_mean_frequency_hz"]
        offsets_hz = alignment_info["frequency_offsets_hz"]
        n_valid_voxels = alignment_info["n_valid_voxels"]

    Undo the alignment by applying the negative stored offsets to the aligned
    data and fitting state::

        restored_data, restored_results, restored_params, _ = (
            align_spectra_by_fit_frequency(
                aligned_data,
                aligned_results,
                aligned_params,
                frequency_offsets_hz=-alignment_info["frequency_offsets_hz"],
            )
        )
    """
    global_mean_hz, offsets_hz, _n_valid_voxels, resolved_alignment_info = (
        _resolve_frequency_alignment(
            fit_results,
            frequency_offsets_hz=frequency_offsets_hz,
            alignment_info=alignment_info,
        )
    )
    shifted_data = _shift_data(
        data,
        offsets_hz,
        global_mean_hz=global_mean_hz,
        sampling_time_axis_s=sampling_time_axis_s,
        dwell_time_s=dwell_time_s,
        signal_domain=signal_domain,
    )
    return (
        shifted_data,
        _shift_fit_results(fit_results, offsets_hz),
        _shift_fit_params(fit_params, offsets_hz),
        resolved_alignment_info,
    )
