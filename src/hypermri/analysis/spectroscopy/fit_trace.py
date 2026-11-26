"""Record the optimization path of one FID fit for step-by-step inspection.

The FID fit (:func:`~hypermri.analysis.spectroscopy.fit_data_pseudo_inv`)
can store every candidate and every accepted state of its grid search and
joint refinement (``store_optimization_history``). This module refits one
selected spectrum with that history enabled and arranges the records as a
timeline of complete model states: measured and model FID and spectrum,
frequencies and T2* inside their bounds, complex amplitudes and the phase
law of the phase band. The result contains no widgets or plots; the fit
monitor page of :mod:`hypermri.visualization.fit_monitor` renders it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy import fid_model
from hypermri.analysis.fitting.models.spectroscopy.diagnostics import (
    HISTORY_EVENTS,
    HISTORY_FIELDS,
    HISTORY_REASONS,
)
from hypermri.analysis.fitting.spectroscopy import fit_data_pseudo_inv

FIT_TRACE_VERSION = 1

_EVENT_NAMES = {code: name for name, code in HISTORY_EVENTS.items()}
_REASON_NAMES = {code: name for name, code in HISTORY_REASONS.items()}
_FIELD = {name: index for index, name in enumerate(HISTORY_FIELDS)}
_GRID_EVENTS = {
    "frequency_candidate": "frequency",
    "t2_candidate": "t2",
    "t2_band_leader_candidate": "t2",
}
_STATE_EVENTS = (
    "initial",
    "frequency_step",
    "t2_step",
    "joint_refinement",
    "verified",
    "returned",
)

# Flags a stored voxel result that differs from the traced refit, for example
# after a review refit or edited start values. Not a scientific tolerance: an
# unchanged deterministic refit reproduces the stored values to rounding.
_STORED_MATCH_RTOL = 1e-6
_STORED_MATCH_ATOL = 1e-9


def fit_spectrum_trace(
    data: xr.DataArray,
    *,
    fit_params: xr.Dataset,
    selection: Mapping[str, int],
    fit_results: xr.Dataset | None = None,
) -> dict[str, object]:
    """Refit one spectrum and return its complete optimization path.

    The selected spectrum is fitted again with
    :func:`~hypermri.analysis.spectroscopy.fit_data_pseudo_inv` and
    ``store_optimization_history=True``; all other settings come unchanged
    from ``fit_params``. The FID fit is deterministic, so the traced path ends
    in the result the full fit returns for this spectrum with the same
    parameters.

    Parameters
    ----------
    data : xarray.DataArray
        Complex spectroscopy data of the full fit, with the spectral dimension
        named by ``fit_params.attrs['spectral_dim']`` (default
        ``acquisition_idx``) and any non-spectral dimensions. The domain
        (``signal_domain`` attribute of ``fit_params``, else of ``data``,
        default ``'temporal'``) is interpreted as in the fit. Only the
        selected spectrum is read.
    fit_params : xarray.Dataset
        Prepared Lorentz FID fit parameters from ``def_fit_params``, with a
        ``metabolite`` dimension and optional parameter maps over the
        non-spectral dimensions of ``data``. Not modified.
    selection : mapping of str to int
        Positional index for each non-spectral dimension of ``data``. A
        missing dimension selects index 0; keys that are not dimensions of
        ``data`` are ignored, so a viewer's full index state can be passed.
        The same indices select the maps of ``fit_params`` and
        ``fit_results``.
    fit_results : xarray.Dataset, optional
        Stored fit of the same data. When given, the selected voxel's
        ``fit_freqs``, ``fit_t2s`` and ``fit_amps`` are compared with the
        traced result (``stored_comparison`` in the output).

    Returns
    -------
    dict of str to object
        JSON-compatible except for NumPy arrays, which the caller encodes:

        ``version``
            Layout version of this dictionary (``FIT_TRACE_VERSION``).
        ``metabolites``, ``selection``, ``signal_domain``, ``objective_name``, ``amplitude_unit``
            Metabolite labels in ``fit_params`` order (the solver component
            order), the resolved positional selection, the input domain, the
            objective minimized by the grid search and the signal unit.
        ``time_s``, ``frequency_hz``, ``ppm_conversion``
            Fitted sampling times in seconds (``n_samples``), the
            spectrum axis ``fftshift(fftfreq(n_samples, dwell))`` in Hz and,
            when the parameters carry a ppm axis, ``{"slope", "offset"}`` with
            ``ppm = slope * hz + offset``; else ``None``.
        ``measured``
            ``fid`` and ``spectrum`` (complex, ``n_samples``): the samples the
            fit compares with the model, and ``fftshift(fft(fid))``.
        ``lines``
            One dictionary per metabolite: ``name``, ``skipped``, start values
            ``initial_frequency_hz`` and ``initial_t2_s``, full search widths
            ``frequency_range_hz`` and ``t2_range_s``, absolute bounds
            ``frequency_lower_hz``/``frequency_upper_hz`` (start -/+ half the
            width) and ``t2_min_s``/``t2_max_s``, ``phase_tolerance_deg``
            (``None`` outside the phase band) and ``t2_coupling`` (``None`` or
            ``{"leader", "ratio"}`` with a number or a ``[low, high]`` band).
        ``options``
            Search settings: ``n_iterations``, ``n_points``,
            ``n_repetitions``, ``zoom_factor``, ``min_peak_separation_hz``,
            ``phase_band_tau_s``, ``search_order``, ``joint_refinement``,
            ``use_magnitude_objective``.
        ``entries``
            Timeline in solver order. Each entry is one complete model state:
            ``kind`` (``initial``, ``grid``, ``frequency_step``,
            ``t2_step``, ``joint_refinement``, ``verified`` or ``returned``),
            ``history_row``, ``fit_round``, ``repetition``, ``iteration``,
            ``line`` (metabolite index moved by a grid, else -1), ``grid``
            (index into ``grids`` or -1), ``accepted``, ``rss`` and
            ``objective`` (squared signal units), ``frequency_hz``,
            ``t2_s``, ``amplitude`` (complex, signal units), ``active`` and,
            per line, ``phase_law_rad`` = ``angle(a) - 2*pi*f*tau`` (wrapped
            to (-pi, pi]) and the entry's ``phase_reference_rad``.
        ``model_fid``, ``model_spectrum``
            Complex arrays ``(n_entries, n_samples)``: the summed model of
            each entry, in the time domain and as ``fftshift(fft(fid))``.
        ``grids``
            One dictionary per grid search: ``parameter`` (``frequency`` or
            ``t2``), ``event``, ``fit_round``, ``repetition``,
            ``iteration``, ``line``, candidate ``values`` (Hz or s),
            ``objective`` (``inf`` where rejected), ``evaluated``,
            ``rejection_reason`` names, ``accepted_index`` (-1 if the line
            stayed at its centre) and ``center_index``.
        ``result``
            ``status``, ``failure_reason``, ``rss``, ``objective`` and the
            returned ``frequency_hz``, ``t2_s`` and ``amplitude`` per label.
        ``stored_comparison``
            ``None`` without ``fit_results``; else ``matches`` and the largest
            ``frequency_difference_hz``, ``t2_relative_difference`` and
            ``amplitude_relative_difference``.

    Raises
    ------
    ValueError
        If the spectral dimension is missing, a selected index is outside
        its dimension, ``fit_params`` keeps an unselected non-metabolite
        dimension, or the parameters describe a Voigt fit (the optimization
        history is not recorded for ``lineshape='voigt'``).

    Notes
    -----
    Eager and deterministic; neither ``data`` nor the datasets are modified
    and nothing is cached. The history does not store the solver's shared
    phase ``phi0``. The phase-law reference of an entry estimates it for
    display: the amplitude-weighted circular mean of ``phase_law_rad`` over
    the active lines of the phase band, moved to the nearest phase at which
    every such line lies within its tolerance when that interval exists, and
    to the middle of the conflicting limits otherwise. The deviations
    ``phase_law_rad - phase_reference_rad`` are then inside the tolerances
    whenever some ``phi0`` allows it. The reference is NaN without
    a phase band.
    Entries and grids keep the solver's component order. With
    ``keep_fit_order=False`` the returned labels (``result``) can exchange
    components, see ``fit_component_index`` of the fit result.

    Examples
    --------
    >>> trace = fit_spectrum_trace(  # doctest: +SKIP
    ...     data, fit_params=fit_params, selection={"x_idx": 3, "y_idx": 5}
    ... )
    >>> trace["entries"][-1]["kind"]  # doctest: +SKIP
    'returned'
    """
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(
            f"fit_spectrum_trace: data has no spectral dimension {spectral_dim!r}; "
            "pass the data of the fit."
        )
    if fid_model._uses_voigt_lineshape(fit_params.attrs):
        raise ValueError(
            "fit_spectrum_trace: lineshape='voigt' does not record an "
            "optimization history; trace a Lorentz fit."
        )
    resolved = _resolve_selection(data, spectral_dim, selection)
    spectrum = data.isel(resolved)
    params = _select_maps(fit_params, resolved, "fit_params").copy(deep=True)
    params.attrs["store_optimization_history"] = True
    params.attrs["show_tqdm"] = False

    result = fit_data_pseudo_inv(spectrum, params)

    times_s = np.asarray(result.attrs["sampling_time_axis_s"], dtype=float)
    n_samples = times_s.size
    domain = str(
        params.attrs.get("signal_domain", data.attrs.get("signal_domain", "temporal"))
    ).lower()
    samples = np.asarray(spectrum.values)
    measured_fid = (
        np.fft.ifft(np.fft.ifftshift(samples)) if domain == "spectral" else samples
    )[:n_samples].astype(np.complex128)
    dwell_s = float(times_s[1] - times_s[0]) if n_samples > 1 else 1.0

    metabolites = [str(name) for name in params.metabolite.values]
    lines = _describe_lines(params, metabolites)
    tau_s = params.attrs.get("phase_band_tau_s")
    band_deg = np.array(
        [
            np.nan
            if line["phase_tolerance_deg"] is None
            else line["phase_tolerance_deg"]
            for line in lines
        ],
        dtype=float,
    )

    entries, grids = _timeline(result)
    amplitudes = np.array(
        [entry["amplitude"] for entry in entries], dtype=np.complex128
    ).reshape(len(entries), len(metabolites))
    frequencies = np.array([entry["frequency_hz"] for entry in entries]).reshape(
        amplitudes.shape
    )
    t2s = np.array([entry["t2_s"] for entry in entries]).reshape(amplitudes.shape)
    model_fid = _model_fids(amplitudes, frequencies, t2s, times_s)
    phase_law, reference = _phase_law(
        amplitudes, frequencies, 0.0 if tau_s is None else float(tau_s), band_deg
    )
    for index, entry in enumerate(entries):
        entry["phase_law_rad"] = phase_law[index]
        entry["phase_reference_rad"] = float(reference[index])

    return {
        "version": FIT_TRACE_VERSION,
        "metabolites": metabolites,
        "selection": resolved,
        "signal_domain": domain,
        "objective_name": result.attrs.get("fit_objective_name"),
        "amplitude_unit": str(data.attrs.get("units", "a.u.")),
        "time_s": times_s,
        "frequency_hz": np.fft.fftshift(np.fft.fftfreq(n_samples, dwell_s)),
        "ppm_conversion": _ppm_conversion(params.attrs),
        "measured": {
            "fid": measured_fid,
            "spectrum": np.fft.fftshift(np.fft.fft(measured_fid)),
        },
        "lines": lines,
        "options": {
            "n_iterations": int(params.attrs.get("niter", 1)),
            "n_points": int(params.attrs.get("npoints", 31)),
            "n_repetitions": int(params.attrs.get("rep_fitting", 11)),
            "zoom_factor": float(params.attrs.get("zoomfactor", 1.5)),
            "min_peak_separation_hz": float(
                params.attrs.get("min_peak_separation_hz", 0.0) or 0.0
            ),
            "phase_band_tau_s": None if tau_s is None else float(tau_s),
            "search_order": str(params.attrs.get("search_order", "amplitude")),
            "joint_refinement": bool(params.attrs.get("joint_refinement", True)),
            "use_magnitude_objective": bool(params.attrs.get("use_sse_mag", False)),
        },
        "entries": entries,
        "model_fid": model_fid,
        "model_spectrum": np.fft.fftshift(np.fft.fft(model_fid, axis=1), axes=1),
        "grids": grids,
        "result": {
            "status": str(result.fit_status.values),
            "failure_reason": str(result.fit_failure_reason.values),
            "rss": float(result.fit_rss),
            "objective": float(result.fit_objective),
            "frequency_hz": result.fit_freqs.values.astype(float),
            "t2_s": result.fit_t2s.values.astype(float),
            "amplitude": result.fit_amps.values.astype(np.complex128),
        },
        "stored_comparison": (
            None
            if fit_results is None
            else _compare_with_stored(fit_results, resolved, result, metabolites)
        ),
    }


def _resolve_selection(
    data: xr.DataArray, spectral_dim: str, selection: Mapping[str, int]
) -> dict[str, int]:
    """Return one checked positional index per non-spectral data dimension."""
    resolved = {}
    for dim in data.dims:
        if dim == spectral_dim:
            continue
        index = int(selection.get(dim, 0))
        if not 0 <= index < data.sizes[dim]:
            raise ValueError(
                f"fit_spectrum_trace: index {index} of {dim!r} is outside "
                f"0..{data.sizes[dim] - 1}."
            )
        resolved[str(dim)] = index
    return resolved


def _select_maps(dataset: xr.Dataset, resolved: dict[str, int], name: str):
    """Select one voxel of a parameter or result dataset."""
    selected = dataset.isel({d: i for d, i in resolved.items() if d in dataset.dims})
    extra = [d for d in selected.dims if d != "metabolite" and selected.sizes[d] > 1]
    if extra:
        raise ValueError(
            f"fit_spectrum_trace: {name} has dimensions {extra} that the data "
            "selection does not index; pass maps over the data's dimensions."
        )
    return selected.squeeze([d for d in selected.dims if d != "metabolite"], drop=True)


def _describe_lines(params: xr.Dataset, metabolites: list[str]) -> list[dict]:
    """Collect start values, search widths, bounds and constraints per line."""
    bounded = fid_model._derive_frequency_bounds(params, operation="fit_spectrum_trace")
    coupling = params.attrs.get("t2_coupling") or {}
    tolerances = params.attrs.get("phase_band_tolerance_deg") or {}

    def value(name: str, index: int) -> float:
        if name not in bounded:
            return float("nan")
        return float(bounded[name].values.reshape(-1)[index])

    lines = []
    for index, name in enumerate(metabolites):
        couple = coupling.get(name)
        if couple is not None:
            leader, ratio = (couple, 1.0) if isinstance(couple, str) else couple
            ratio = [float(r) for r in ratio] if np.ndim(ratio) else float(ratio)
            couple = {"leader": str(leader), "ratio": ratio}
        tolerance = tolerances.get(name)
        lines.append(
            {
                "name": name,
                "skipped": bool(
                    bounded["skip_fitting"].values.reshape(-1)[index]
                    if "skip_fitting" in bounded
                    else False
                ),
                "initial_frequency_hz": value("initial_freqs_hz", index),
                "frequency_range_hz": value("metabs_freq_ranges", index),
                "frequency_lower_hz": value("freq_range_lower_bound_hz", index),
                "frequency_upper_hz": value("freq_range_upper_bound_hz", index),
                "initial_t2_s": value("initial_t2s_s", index),
                "t2_range_s": value("metabs_t2_ranges", index),
                "t2_min_s": value("min_t2_s", index),
                "t2_max_s": value("max_t2_s", index),
                "phase_tolerance_deg": None if tolerance is None else float(tolerance),
                "t2_coupling": couple,
            }
        )
    return lines


def _timeline(result: xr.Dataset) -> tuple[list[dict], list[dict]]:
    """Split the history rows into model-state entries and grid searches."""
    count = int(result.fit_history_count)
    info = result.fit_history_info.values[:count]
    amps = result.fit_history_amps.values[:count]
    freqs = result.fit_history_freqs.values[:count]
    t2s = result.fit_history_t2s.values[:count]
    active = result.fit_history_active.values[:count]
    rss = result.fit_history_rss.values[:count]
    objective = result.fit_history_objective.values[:count]

    def entry(row: int, kind: str, *, line=-1, grid=-1, accepted=None) -> dict:
        fields = info[row]
        return {
            "kind": kind,
            "history_row": row,
            "fit_round": int(fields[_FIELD["fit_round"]]),
            "repetition": int(fields[_FIELD["repetition"]]),
            "iteration": int(fields[_FIELD["iteration"]]),
            "line": int(line),
            "grid": int(grid),
            "accepted": bool(fields[_FIELD["accepted"]])
            if accepted is None
            else accepted,
            "rss": float(rss[row]),
            "objective": float(objective[row]),
            "frequency_hz": freqs[row].astype(float),
            "t2_s": t2s[row].astype(float),
            "amplitude": amps[row].astype(np.complex128),
            "active": active[row].astype(bool),
        }

    entries: list[dict] = []
    grids: list[dict] = []
    row = 0
    while row < count:
        event = _EVENT_NAMES.get(int(info[row, _FIELD["event"]]), "unknown")
        if event in _GRID_EVENTS:
            end = row + 1
            key = info[row, [0, 1, 2, 3, 4]]
            while end < count and np.array_equal(info[end, [0, 1, 2, 3, 4]], key):
                end += 1
            line = int(info[row, _FIELD["metabolite_index"]])
            values_by_row = freqs if _GRID_EVENTS[event] == "frequency" else t2s
            values = values_by_row[row:end, line].astype(float)
            accepted = np.flatnonzero(info[row:end, _FIELD["accepted"]] == 1)
            previous = (
                entries[-1][
                    "frequency_hz" if _GRID_EVENTS[event] == "frequency" else "t2_s"
                ][line]
                if entries
                else np.nan
            )
            centre = np.flatnonzero(values == previous)
            grid = {
                "parameter": _GRID_EVENTS[event],
                "event": event,
                "fit_round": int(info[row, _FIELD["fit_round"]]),
                "repetition": int(info[row, _FIELD["repetition"]]),
                "iteration": int(info[row, _FIELD["iteration"]]),
                "line": line,
                "values": values,
                "objective": objective[row:end].astype(float),
                "evaluated": info[row:end, _FIELD["evaluated"]] == 1,
                "rejection_reason": [
                    _REASON_NAMES.get(int(code), "unknown")
                    for code in info[row:end, _FIELD["rejection_reason"]]
                ],
                "accepted_index": int(accepted[0]) if accepted.size else -1,
                "center_index": int(centre[0]) if centre.size else -1,
            }
            grids.append(grid)
            chosen = grid["accepted_index"]
            if chosen < 0:
                chosen = grid["center_index"]
            if chosen >= 0 and grid["evaluated"][chosen]:
                entries.append(
                    entry(
                        row + chosen,
                        "grid",
                        line=line,
                        grid=len(grids) - 1,
                        accepted=grid["accepted_index"] >= 0,
                    )
                )
            row = end
            continue
        if event in _STATE_EVENTS and bool(info[row, _FIELD["evaluated"]]):
            entries.append(entry(row, event))
        row += 1
    return entries, grids


def _model_fids(amplitudes, frequencies, t2s, times_s) -> np.ndarray:
    """Evaluate the summed model FID of every entry with the fit's model."""
    if amplitudes.shape[0] == 0:
        return np.zeros((0, times_s.size), dtype=np.complex128)
    unused = ~np.isfinite(amplitudes) | (amplitudes == 0)
    components = fid_model.reconstruct_fid_signal(
        amplitude=xr.DataArray(
            np.where(unused, 0.0, amplitudes), dims=("entry", "metabolite")
        ),
        freq_hz=xr.DataArray(
            np.where(unused, 0.0, frequencies), dims=("entry", "metabolite")
        ),
        t2_star_s=xr.DataArray(
            np.where(unused, 1.0, t2s), dims=("entry", "metabolite")
        ),
        time_axis_s=xr.DataArray(times_s, dims="sample"),
    )
    return (
        components.sum("metabolite")
        .transpose("entry", "sample")
        .values.astype(np.complex128)
    )


def _phase_law(amplitudes, frequencies, tau_s: float, band_deg: np.ndarray):
    """Remove the phase-law frequency term and estimate a display reference."""
    law = np.angle(amplitudes * np.exp(-2j * np.pi * frequencies * tau_s))
    band = np.isfinite(band_deg)
    reference = np.full(amplitudes.shape[0], np.nan)
    tolerance = np.deg2rad(band_deg)
    for row in range(amplitudes.shape[0]):
        lines = band & np.isfinite(amplitudes[row]) & (amplitudes[row] != 0)
        if not lines.any():
            continue
        weights = np.abs(amplitudes[row, lines])
        mean = np.angle(np.sum(weights * np.exp(1j * law[row, lines])))
        offset = np.angle(np.exp(1j * (law[row, lines] - mean)))
        low = np.max(offset - tolerance[lines])
        high = np.min(offset + tolerance[lines])
        # Without a common interval (a spread beyond the tolerances, or one at
        # exactly twice the tolerance that rounding pushes over), the midpoint
        # keeps the largest deviation as small as possible.
        shift = np.clip(0.0, low, high) if low <= high else 0.5 * (low + high)
        reference[row] = np.angle(np.exp(1j * (mean + shift)))
    return law, reference


def _ppm_conversion(attrs: Mapping) -> dict[str, float] | None:
    """Return the linear Hz-to-ppm map defined by the parameters' axes."""
    hz, ppm = attrs.get("freq_axis_hz"), attrs.get("freq_axis_ppm")
    if hz is None or ppm is None:
        return None
    hz, ppm = np.asarray(hz, dtype=float), np.asarray(ppm, dtype=float)
    if hz.shape != ppm.shape or hz.size < 2 or not np.all(np.isfinite(hz + ppm)):
        return None
    slope, offset = np.polyfit(hz, ppm, 1)
    return {"slope": float(slope), "offset": float(offset)}


def _compare_with_stored(fit_results, resolved, result, metabolites) -> dict:
    """Report how far the stored voxel result is from the traced refit."""
    missing = [n for n in ("fit_freqs", "fit_t2s", "fit_amps") if n not in fit_results]
    if missing:
        raise ValueError(
            f"fit_spectrum_trace: fit_results lacks {missing}; pass the dataset "
            "returned by fit_data_pseudo_inv."
        )
    stored = _select_maps(
        fit_results[["fit_freqs", "fit_t2s", "fit_amps"]], resolved, "fit_results"
    ).sel(metabolite=metabolites)
    traced = result.sel(metabolite=metabolites)

    def largest(difference) -> float:
        difference = np.asarray(difference, dtype=float)
        finite = difference[np.isfinite(difference)]
        return float(finite.max()) if finite.size else 0.0

    pairs = [
        (stored[name].values, traced[name].values)
        for name in ("fit_freqs", "fit_t2s", "fit_amps")
    ]
    matches = all(
        np.allclose(
            a, b, rtol=_STORED_MATCH_RTOL, atol=_STORED_MATCH_ATOL, equal_nan=True
        )
        for a, b in pairs
    )
    (sf, tf), (st, tt), (sa, ta) = pairs
    scale_t2 = np.maximum(np.abs(tt), np.finfo(float).tiny)
    scale_amp = np.maximum(np.abs(ta), np.finfo(float).tiny)
    return {
        "matches": bool(matches),
        "frequency_difference_hz": largest(np.abs(sf - tf)),
        "t2_relative_difference": largest(np.abs(st - tt) / scale_t2),
        "amplitude_relative_difference": largest(np.abs(sa - ta) / scale_amp),
    }
