"""Refit one labelled Lorentz or Voigt spectrum through the established fitter.

This module owns review-specific parameter preparation, time-axis resolution
and final amplitude refinement. It reads no widget, controller or session state.
Callers own selection, session commits and result persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from ..fitting.models.spectroscopy import fid_model
from .schema import _sampling_time_axis_from_data

__all__ = ["SpectroscopyRefitEdit", "refit_spectroscopy_voxel"]


@dataclass(frozen=True)
class SpectroscopyRefitEdit:
    """Express one reviewed metabolite's edits in scientific units.

    Parameters
    ----------
    initial_freq_hz : float or None, optional
        Reviewed initial frequency in Hz. None keeps the dataset's initial
        frequency, falling back to the earlier fitted frequency when absent.
        Applied with an explicit freq_search_range_hz row.
    freq_search_range_hz : float or None, optional
        Full frequency search width in Hz, finite and non-negative. The
        bounds are the initial frequency -/+ half this width; zero freezes
        the frequency at its start, as in the full fit. None keeps the
        voxel's ``metabs_freq_ranges`` value instead of a reviewed row.
    initial_t2_s, min_t2_s, max_t2_s : float or None, optional
        Reviewed initial T2-star and bounds in seconds; for Voigt fits this is
        the Lorentz part of the line. All three must
        accompany an explicit frequency-width row. Without that row, the
        earlier fitted T2 is the start and the selected voxel's ``min_t2_s``
        and ``max_t2_s`` maps are the bounds. The T2-star search width is
        always the voxel's ``metabs_t2_ranges`` value.
    enabled : bool, optional
        Whether this component participates in fitting; defaults True. A
        disabled component keeps its earlier amplitude, frequency and T2.

    Notes
    -----
    Construction stores values without reading samples or fitting. UI callers
    translate displayed milliseconds to seconds before constructing the edit.
    No fit default, result cache or mutable widget reference is stored here.

    Examples
    --------
    >>> edit = SpectroscopyRefitEdit(
    ...     initial_freq_hz=15, freq_search_range_hz=4,
    ...     initial_t2_s=0.03, min_t2_s=0.005, max_t2_s=0.1,
    ... )
    """

    initial_freq_hz: float | None = None
    freq_search_range_hz: float | None = None
    initial_t2_s: float | None = None
    min_t2_s: float | None = None
    max_t2_s: float | None = None
    enabled: bool = True


def refit_spectroscopy_voxel(
    data: xr.DataArray,
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    *,
    voxel_indices: Mapping[str, int],
    data_indices: Mapping[str, int] | None = None,
    metabolites: Sequence[str] | None = None,
    edits: Mapping[str, SpectroscopyRefitEdit] | None = None,
    scan: Any = None,
) -> dict[str, object]:
    """Refit one selected spectrum and refine amplitudes without a review UI.

    Every line's frequency bounds are its start -/+ half its full search
    width; stored bounds are not read. Disabled
    metabolites retain their earlier values. After the established frequency/T2
    search, amplitudes are solved again at its final parameters, retaining the
    fitter amplitudes if that optional refinement fails.

    Parameters
    ----------
    data : xarray.DataArray
        Complex signal containing acquisition_idx and arbitrary non-spectral
        dimensions in any order. Samples stay in their acquired storage dtype
        and are read eagerly only for the selected spectrum. No preprocessing
        or domain conversion is added; fit_params.attrs.signal_domain selects
        the established temporal/spectral fitter interpretation.
    fit_params : xarray.Dataset
        Existing labelled Lorentz or Voigt parameter contract in metabolite
        order. A Voigt refit keeps one shared Gaussian FWHM per spectrum,
        starting from the voxel's earlier fitted ``fit_gauss_fwhm_hz`` and
        using the ``gauss_max_fwhm_hz`` and ``gauss_search_width_hz`` attrs.
        Initial frequencies and widths use Hz; T2 values use seconds. Each
        line's search window comes from the selected voxel's maps, as in the
        full fit: ``metabs_freq_ranges`` (full frequency width) and, for
        lines without a reviewed row, ``min_t2_s`` and ``max_t2_s``;
        ``metabs_t2_ranges`` (full T2-star width) applies to every line.
        A width of 0 freezes that parameter. The attr of the same name is
        used only when a map is absent or shorter than ``metabolites``;
        invalid map values are passed on and rejected by the fitter.
        Without map and attr, or with a non-finite attr, the full fit's
        defaults apply: ``fid_model.METABS_FREQ_RANGE_DEFAULT``,
        ``MIN_T2_S_DEFAULT``, ``MAX_T2_S_DEFAULT`` (Lorentz) or
        ``VOIGT_MAX_T2_S_DEFAULT`` (Voigt) and ``METABS_T2_RANGE_DEFAULT``.
        Global attrs retain search counts, amplitude options, sampling times,
        cut_off and nsamplepoints.
    fit_results : xarray.Dataset
        Earlier fit_amps, fit_freqs and fit_t2s with a metabolite coordinate and
        selected spatial dimensions. Their storage dtype, variable attrs and
        scalar selected coordinates are preserved in the returned voxel result.
    voxel_indices : mapping of str to int
        Positional selectors for result dimensions. The same applicable names
        select parameter dimensions. Select all non-metabolite result dimensions
        with size greater than one to obtain a single voxel.
    data_indices : mapping of str to int or None, optional
        Positional signal selectors, including channel/repetition when needed.
        None uses voxel_indices. Missing non-spectral indices select zero and
        supplied indices clamp to the corresponding signal size, as in review.
    metabolites : sequence of str or None, optional
        Existing fit-session metabolite order. None uses the result coordinate.
        Parameter/result arrays must match that order.
    edits : mapping of str to SpectroscopyRefitEdit or None, optional
        Explicit reviewed rows or enable flags. Names must occur in metabolites;
        absent rows keep the dataset/earlier-fit preparation rules.
    scan : object or None, optional
        Metadata-only source for sampling_time_axis, sampling_time_axis_s or
        dwell_time_s. Array-aligned seconds coordinates/attrs take precedence,
        then parameter metadata and source candidates. Legacy full axes are
        cut once; valid dwell times reconstruct a seconds axis. The existing
        final fallback is a positional unit-step axis when timing is unavailable.

    Returns
    -------
    dict of str to object
        fit_results is a new selected-voxel Dataset with updated fit_amps in
        signal units, fit_freqs in Hz and fit_t2s in seconds. Voigt results
        also update fit_gauss_fwhm_hz, fit_lorentz_fwhm_hz and
        fit_gauss_fwhm_status. Present fit_fwhm_hz and fit_magnitude_fwhm_hz
        are recomputed from the refitted fit_t2s for both lineshapes for the
        voxel. Present fit_rss, fit_objective, fit_n_free_parameters,
        fit_residual_degrees_of_freedom and fit_reduced_chi_squared are
        recomputed for the refitted model of the enabled lines with their
        effective search windows; the reduced chi-squared uses the selected
        spectrum's ``noise_sigma_quadrature`` coordinate of ``data``
        (``noise_source="acquisition_tail"``) and is NaN with
        ``noise_source="none"`` without it. Present fit_freq_status and
        fit_t2_status of the enabled lines are recomputed against the refit's
        frequency bounds and T2* bounds. Other variables and provenance
        attrs are retained. status is 'refitted' or
        'no_enabled_metabolites'; this is operation status, not a fit-quality
        classification. active_metabolites records fitted order.
        sampling_time_axis_s is the used float64 axis. effective_options records
        global scientific settings and the active per-metabolite starts, search
        widths and bounds, with data_indices/voxel_indices recording selection.

        frequency_search_diagnostics, t2_search_diagnostics and
        amplitude_filter_flags retain the fitter's optional outputs in active
        metabolite order. Search arrays use (metabolite, repetition, iteration,
        candidate, 3); the last axis holds candidate frequency in Hz or T2 in
        seconds, complex-FID residual sum of squares and magnitude-spectrum
        residual sum of squares in the respective squared signal units. They
        retain float64 calculation precision, NaN uncomputed entries and infinite
        failed-candidate scores. Filter flags are booleans per active metabolite;
        this review path does not enable amplitude thresholding. Absent outputs
        are None. amplitude_refinement_error is
        the caught exception or None, allowing the caller to retain its reporting
        policy. reconstructed_signal and uncertainty are None: this operation
        does not add either calculation to the earlier review workflow.

    Raises
    ------
    TypeError
        If an edit is not a SpectroscopyRefitEdit.
    ValueError
        If acquisition_idx is absent, ``data`` carries a
        ``noise_sigma_quadrature`` coordinate of another noise source,
        fit_params and fit_results carry
        different lineshapes, an edit
        names an unknown metabolite, an explicit row lacks its T2 fields or
        has a negative or non-finite ``freq_search_range_hz``, or supplied
        selector dimensions are absent.
        Numerical fit/configuration errors propagate from the established fitter.
    KeyError
        If required fit variables are absent.

    Warns
    -----
    UserWarning
        If stored ``freq_range_lower_bound_hz`` or ``freq_range_upper_bound_hz``
        differ from ``initial_freqs_hz`` -/+ ``metabs_freq_ranges`` / 2, as in
        the full fit. The refit derives its bounds from start and width and
        does not read the stored ones; ``fit_params`` itself is not modified.

    Notes
    -----
    This eager operation does not mutate inputs, initialize UI, reconstruct a
    scan, persist results or change a fit session. Calculation precision remains
    with the numerical fitter; returned estimates use the earlier result storage
    dtype. Frequencies/T2 for disabled components and input coordinate direction,
    units and provenance are retained. No cache is introduced. A fitter failure propagates; amplitude-refinement failure keeps
    the fitter result and is explicitly reported in the return value.
    The ``t2_coupling`` and phase-band options of ``fit_params`` apply to the
    refit and to the final amplitude solve, as in the full fit;
    ``search_order`` applies to the refit's grid search, and
    ``joint_refinement`` (default True) to the refinement after it.
    Returned timing arrays may share input metadata; treat them as read-only.

    Examples
    --------
    >>> from hypermri.analysis.spectroscopy import refit_spectroscopy_voxel
    >>> reviewed = refit_spectroscopy_voxel(  # doctest: +SKIP
    ...     data, fit_params, fit_results, voxel_indices={'read_idx': 1},
    ... )
    >>> voxel_results = reviewed['fit_results']  # doctest: +SKIP
    """
    if "acquisition_idx" not in data.dims:
        raise ValueError(
            "refit_spectroscopy_voxel requires an acquisition_idx dimension."
        )
    voigt = fid_model._require_same_lineshape(
        "refit_spectroscopy_voxel", fit_params, fit_results
    )
    metabs = (
        [str(name) for name in fit_results.metabolite.values]
        if metabolites is None
        else list(metabolites)
    )
    edits = {} if edits is None else edits
    for name, edit in edits.items():
        if name not in metabs:
            raise ValueError(
                f"refit_spectroscopy_voxel edit names unknown metabolite {name!r}."
            )
        if not isinstance(edit, SpectroscopyRefitEdit):
            raise TypeError(f"Refit edit for {name!r} must be SpectroscopyRefitEdit.")
        if edit.freq_search_range_hz is not None and any(
            value is None for value in (edit.initial_t2_s, edit.min_t2_s, edit.max_t2_s)
        ):
            raise ValueError(
                f"Refit edit for {name!r} requires initial_t2_s, min_t2_s and max_t2_s with freq_search_range_hz."
            )
        if edit.freq_search_range_hz is not None and not (
            np.isfinite(edit.freq_search_range_hz) and edit.freq_search_range_hz >= 0
        ):
            raise ValueError(
                f"Refit edit for {name!r} has freq_search_range_hz="
                f"{edit.freq_search_range_hz!r}; use a finite full width >= 0 Hz "
                "(0 freezes the frequency)."
            )
    global_attrs = fit_params.attrs
    source_indices = voxel_indices if data_indices is None else data_indices
    data_sel = {}
    for d in data.dims:
        if d == "acquisition_idx":
            continue
        idx = source_indices.get(d, 0)
        data_sel[d] = max(0, min(idx, data.sizes[d] - 1))

    raw_fid = data.isel(data_sel).values

    voxel_results = fit_results.isel(voxel_indices)
    initial_t2s = voxel_results["fit_t2s"].values.astype(float)

    params_spatial_sel = {
        d: v for d, v in voxel_indices.items() if d in fit_params.dims
    }
    cp = fit_params.isel(params_spatial_sel) if params_spatial_sel else fit_params
    # Warn about stale stored bounds as the full fit does; the refit derives
    # its bounds below from the start and width of each line.
    fid_model._derive_frequency_bounds(cp, operation="refit_spectroscopy_voxel")

    n_metabs = len(metabs)
    if "initial_freqs_hz" in cp:
        initial_freqs = (
            np.asarray(cp["initial_freqs_hz"].values, dtype=float).ravel().copy()
        )
    else:
        initial_freqs = voxel_results["fit_freqs"].values.astype(float)
    if initial_freqs.size != n_metabs:
        resized = np.zeros(n_metabs, dtype=float)
        copy_n = min(initial_freqs.size, n_metabs)
        if copy_n > 0:
            resized[:copy_n] = initial_freqs[:copy_n]
        initial_freqs = resized

    freq_ranges = np.empty(n_metabs)
    t2_ranges = np.empty(n_metabs)
    min_t2 = np.empty(n_metabs)
    max_t2 = np.empty(n_metabs)
    # Each line's search window comes from the selected voxel's maps, as in
    # the full fit; the attrs only stand in for a line without a map value.
    voxel_maps = {
        key: (
            np.asarray(cp[key].values, dtype=float).ravel()
            if key in cp
            else np.empty(0)
        )
        for key in ("metabs_freq_ranges", "metabs_t2_ranges", "min_t2_s", "max_t2_s")
    }

    def window_value(key: str, mi: int, default: float) -> float:
        """Return one line's search-window value from the voxel map or attr.

        Parameters
        ----------
        key : str
            Window parameter name, used for the map and the attr.
        mi : int
            Zero-based line position in ``metabs``.
        default : float
            The full fit's default from ``fid_model``, used when neither the
            map nor the attr provides a finite value.

        Returns
        -------
        float
            The map value when the map covers ``mi``, unchanged even when
            invalid, else the attr value or ``default``.
        """
        values = voxel_maps[key]
        if mi < values.size:
            return values[mi]
        return _coerce_metab_attr_scalar(global_attrs.get(key), mi, default)

    for mi, mname in enumerate(metabs):
        edit = edits.get(mname)
        if edit is not None and edit.freq_search_range_hz is not None:
            if edit.initial_freq_hz is not None:
                initial_freqs[mi] = float(edit.initial_freq_hz)
            freq_ranges[mi] = edit.freq_search_range_hz
            min_t2[mi] = edit.min_t2_s
            max_t2[mi] = edit.max_t2_s
            initial_t2s[mi] = edit.initial_t2_s
        else:
            freq_ranges[mi] = window_value(
                "metabs_freq_ranges", mi, fid_model.METABS_FREQ_RANGE_DEFAULT
            )
            min_t2[mi] = window_value("min_t2_s", mi, fid_model.MIN_T2_S_DEFAULT)
            max_t2[mi] = window_value(
                "max_t2_s",
                mi,
                fid_model.VOIGT_MAX_T2_S_DEFAULT
                if voigt
                else fid_model.MAX_T2_S_DEFAULT,
            )
        # A reviewed row has no T2* width; zero freezes T2* as in the full fit.
        t2_ranges[mi] = window_value(
            "metabs_t2_ranges", mi, fid_model.METABS_T2_RANGE_DEFAULT
        )

    # Start and full width own every line's bounds, as in
    # fid_model._update_frequency_bounds; stored bounds are not read.
    freq_lower = initial_freqs - freq_ranges / 2
    freq_upper = initial_freqs + freq_ranges / 2

    cut_off = int(global_attrs.get("cut_off", 0))
    sampling_time = _resolve_refit_sampling_time_axis(
        data,
        global_attrs,
        expected_size=len(raw_fid),
        scan=scan,
    )
    requested_nsamplepoints = int(global_attrs.get("nsamplepoints", len(sampling_time)))
    nsamplepoints = max(
        1,
        min(requested_nsamplepoints, len(raw_fid), len(sampling_time)),
    )
    sampling_time = sampling_time[:nsamplepoints]

    active_metab_indices = [
        mi
        for mi, mname in enumerate(metabs)
        if mname not in edits or edits[mname].enabled
    ]
    if not active_metab_indices:
        return {
            "fit_results": voxel_results.copy(deep=True),
            "status": "no_enabled_metabolites",
            "active_metabolites": [],
            "sampling_time_axis_s": sampling_time,
            "effective_options": {
                "data_indices": data_sel,
                "voxel_indices": dict(voxel_indices),
            },
            "amplitude_refinement_error": None,
            "frequency_search_diagnostics": None,
            "t2_search_diagnostics": None,
            "amplitude_filter_flags": None,
            "reconstructed_signal": None,
            "uncertainty": None,
        }

    metabs_active = [metabs[i] for i in active_metab_indices]
    initial_freqs_active = initial_freqs[active_metab_indices]
    initial_t2s_active = initial_t2s[active_metab_indices]
    freq_ranges_active = freq_ranges[active_metab_indices]
    t2_ranges_active = t2_ranges[active_metab_indices]
    min_t2_active = min_t2[active_metab_indices]
    max_t2_active = max_t2[active_metab_indices]
    freq_lower_active = freq_lower[active_metab_indices]
    freq_upper_active = freq_upper[active_metab_indices]

    fit_kwargs = {
        "metabs": metabs_active,
        "rep_fitting": int(global_attrs.get("rep_fitting", 3)),
        "zoomfactor": float(global_attrs.get("zoomfactor", 2.0)),
        "niter": int(global_attrs.get("niter", 10)),
        "npoints": int(global_attrs.get("npoints", 50)),
        "pseudo_inv": bool(global_attrs.get("pseudo_inv", True)),
        "amp_solver": global_attrs.get("amp_solver", None),
        "amp_ridge_lambda": float(global_attrs.get("amp_ridge_lambda", 0.0)),
        "min_peak_separation_hz": float(
            global_attrs.get("min_peak_separation_hz", 5.0)
        ),
        "signal_domain": str(global_attrs.get("signal_domain", "temporal")),
        "sampling_time_axis_s": sampling_time,
        "nsamplepoints": nsamplepoints,
        "cut_off": cut_off,
        "t2_coupling": fid_model._normalize_t2_coupling(
            global_attrs.get("t2_coupling"), metabs
        ),
        "phase_band_tolerance_deg": global_attrs.get("phase_band_tolerance_deg"),
        "phase_band_tau_s": global_attrs.get("phase_band_tau_s"),
        "search_order": fid_model._normalize_search_order(
            global_attrs.get("search_order", fid_model.SEARCH_ORDER_DEFAULT)
        ),
        "joint_refinement": bool(
            global_attrs.get("joint_refinement", fid_model.JOINT_REFINEMENT_DEFAULT)
        ),
    }

    initial_gauss_fwhm_hz = None
    if voigt:
        fit_kwargs.update(_voigt_refit_options(global_attrs))
        initial_gauss_fwhm_hz = _voigt_refit_start_hz(
            voxel_results,
            cp,
            fit_kwargs["gauss_initial_fwhm_hz"],
            fit_kwargs["gauss_max_fwhm_hz"],
        )
        voigt_solver_kwargs = {
            "lineshape": "voigt",
            "gauss_max_fwhm_hz": fit_kwargs["gauss_max_fwhm_hz"],
            "gauss_search_width_hz": fit_kwargs["gauss_search_width_hz"],
            "initial_gauss_fwhm_hz": initial_gauss_fwhm_hz,
        }
    else:
        voigt_solver_kwargs = {}

    result = fid_model._fit_single_spectrum_pseudo_inv(
        data_slice=raw_fid,
        initial_freq_hz=initial_freqs_active,
        initial_t2_s=initial_t2s_active,
        freq_search_range_hz=freq_ranges_active,
        t2_search_range_s=t2_ranges_active,
        min_t2_s=min_t2_active,
        max_t2_s=max_t2_active,
        freq_search_range_lower_bound_hz=freq_lower_active,
        freq_search_range_upper_bound_hz=freq_upper_active,
        metabs=fit_kwargs["metabs"],
        rep_fitting=fit_kwargs["rep_fitting"],
        zoomfactor=fit_kwargs["zoomfactor"],
        niter=fit_kwargs["niter"],
        npoints=fit_kwargs["npoints"],
        pseudo_inv=fit_kwargs["pseudo_inv"],
        amp_solver=fit_kwargs["amp_solver"],
        amp_ridge_lambda=fit_kwargs["amp_ridge_lambda"],
        min_peak_separation_hz=fit_kwargs["min_peak_separation_hz"],
        signal_domain=fit_kwargs["signal_domain"],
        sampling_time_axis_s=fit_kwargs["sampling_time_axis_s"],
        nsamplepoints=fit_kwargs["nsamplepoints"],
        cut_off=fit_kwargs["cut_off"],
        t2_coupling=fit_kwargs["t2_coupling"],
        phase_band_tolerance_deg=fit_kwargs["phase_band_tolerance_deg"],
        phase_band_tau_s=fit_kwargs["phase_band_tau_s"],
        search_order=fit_kwargs["search_order"],
        joint_refinement=fit_kwargs["joint_refinement"],
        **voigt_solver_kwargs,
    )

    best_amps, best_freqs, best_t2s = result[0], result[1], result[2]
    best_gauss_fwhm_hz = float(result[6]) if voigt else None
    # Re-solve amplitudes for the final fitted frequency/T2* parameters.
    # This avoids zero-amplitude lock-in for newly added metabolites.
    amplitude_refinement_error = None
    try:
        _, refined_amps = fid_model.solve_for_complex_amplitudes(
            values=raw_fid,
            freq_hz=np.asarray(best_freqs, dtype=float),
            t2_star_s=np.asarray(best_t2s, dtype=float),
            time_axis_s=np.asarray(sampling_time, dtype=float),
            signal_domain=str(global_attrs.get("signal_domain", "temporal")),
            pseudo_inv=bool(global_attrs.get("pseudo_inv", True)),
            amp_solver=global_attrs.get("amp_solver", None),
            amp_ridge_lambda=float(global_attrs.get("amp_ridge_lambda", 0.0)),
            gauss_fwhm_hz=best_gauss_fwhm_hz,
            **fid_model._phase_band_solver_kwargs(fit_kwargs),
        )
        best_amps = np.asarray(refined_amps)
    except Exception as amp_err:
        amplitude_refinement_error = amp_err

    best_amps_full = voxel_results["fit_amps"].values.copy()
    best_freqs_full = voxel_results["fit_freqs"].values.copy()
    best_t2s_full = voxel_results["fit_t2s"].values.copy()

    for i, mi in enumerate(active_metab_indices):
        best_amps_full[mi] = best_amps[i]
        best_freqs_full[mi] = best_freqs[i]
        best_t2s_full[mi] = best_t2s[i]

    updated_results = voxel_results.copy(deep=True)
    for name, values in (
        ("fit_amps", best_amps_full),
        ("fit_freqs", best_freqs_full),
        ("fit_t2s", best_t2s_full),
    ):
        updated_results[name].values[...] = values
    _refresh_refit_linewidths(updated_results, best_gauss_fwhm_hz, fit_kwargs)
    _refresh_refit_fit_quality(
        updated_results,
        fid=raw_fid,
        amplitudes=best_amps,
        frequencies=best_freqs,
        t2s=best_t2s,
        gauss_fwhm_hz=best_gauss_fwhm_hz,
        fit_options={
            **fit_kwargs,
            "use_sse_mag": bool(global_attrs.get("use_sse_mag", False)),
        },
        count_options=dict(global_attrs),
        metabolites=metabs,
        active=np.isin(np.arange(n_metabs), active_metab_indices),
        windows=(freq_ranges, t2_ranges, min_t2, max_t2, freq_lower, freq_upper),
        noise=_selected_carried_noise(data, data_sel),
    )
    _refresh_refit_bound_statuses(
        updated_results,
        metabolites=metabs,
        active=np.isin(np.arange(n_metabs), active_metab_indices),
        windows=(freq_ranges, t2_ranges, min_t2, max_t2, freq_lower, freq_upper),
        t2_coupling=fit_kwargs["t2_coupling"],
    )
    return {
        "fit_results": updated_results,
        "status": "refitted",
        "active_metabolites": metabs_active,
        "sampling_time_axis_s": sampling_time,
        "effective_options": {
            **fit_kwargs,
            "initial_freq_hz": initial_freqs_active,
            "initial_t2_s": initial_t2s_active,
            "freq_search_range_hz": freq_ranges_active,
            "t2_search_range_s": t2_ranges_active,
            "min_t2_s": min_t2_active,
            "max_t2_s": max_t2_active,
            "freq_search_range_lower_bound_hz": freq_lower_active,
            "freq_search_range_upper_bound_hz": freq_upper_active,
            **({"initial_gauss_fwhm_hz": initial_gauss_fwhm_hz} if voigt else {}),
            "data_indices": data_sel,
            "voxel_indices": dict(voxel_indices),
        },
        "amplitude_refinement_error": amplitude_refinement_error,
        "frequency_search_diagnostics": result[3] if len(result) > 3 else None,
        "t2_search_diagnostics": result[4] if len(result) > 4 else None,
        "amplitude_filter_flags": result[5] if len(result) > 5 else None,
        "reconstructed_signal": None,
        "uncertainty": None,
    }


def _voigt_refit_options(global_attrs: Mapping) -> dict[str, object]:
    """Return the effective Voigt settings of a prepared fit in Hz."""
    return {
        "lineshape": "voigt",
        "gauss_initial_fwhm_hz": float(
            global_attrs.get(
                "gauss_initial_fwhm_hz", fid_model.GAUSS_INITIAL_FWHM_HZ_DEFAULT
            )
        ),
        "gauss_max_fwhm_hz": float(
            global_attrs.get("gauss_max_fwhm_hz", fid_model.GAUSS_MAX_FWHM_HZ_DEFAULT)
        ),
        "gauss_search_width_hz": float(
            global_attrs.get(
                "gauss_search_width_hz", fid_model.GAUSS_SEARCH_WIDTH_HZ_DEFAULT
            )
        ),
    }


def _first_finite(values) -> float | None:
    """Return the first finite real value of an array-like, or None."""
    flat = np.real(np.asarray(values, dtype=complex)).astype(float).ravel()
    finite = flat[np.isfinite(flat)]
    return float(finite[0]) if finite.size else None


def _voigt_refit_start_hz(
    voxel_results: xr.Dataset,
    voxel_params: xr.Dataset,
    global_start_hz: float,
    max_fwhm_hz: float,
) -> float:
    """Return the start Gaussian FWHM in Hz for refitting one Voigt spectrum.

    The earlier fitted ``fit_gauss_fwhm_hz`` of the voxel comes first, as the
    earlier fitted T2* start the Lorentz part. The prepared
    ``initial_gauss_fwhm_hz`` map and then the global ``gauss_initial_fwhm_hz``
    follow. The value is clipped to ``[0, max_fwhm_hz]``.
    """
    for source, name in (
        (voxel_results, "fit_gauss_fwhm_hz"),
        (voxel_params, "initial_gauss_fwhm_hz"),
    ):
        if name in source:
            value = _first_finite(source[name].values)
            if value is not None:
                return float(np.clip(value, 0.0, max_fwhm_hz))
    return float(np.clip(global_start_hz, 0.0, max_fwhm_hz))


def _selected_carried_noise(
    data: xr.DataArray, data_sel: Mapping[str, int]
) -> xr.DataArray | None:
    """Return the selected spectrum's acquisition-tail sigma, or None.

    The coordinate is read and validated by the same function as in
    ``fit_data_pseudo_inv``; a sigma of another source raises ValueError.
    """
    from ..fitting.spectroscopy import _carried_acquisition_noise

    carried = _carried_acquisition_noise(data)
    if carried is None:
        return None
    return carried.isel({dim: data_sel[dim] for dim in carried.dims}, drop=True)


def _refresh_refit_fit_quality(
    results: xr.Dataset,
    *,
    fid: np.ndarray,
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    t2s: np.ndarray,
    gauss_fwhm_hz: float | None,
    fit_options: Mapping,
    count_options: Mapping,
    metabolites: Sequence[str],
    active: np.ndarray,
    windows: tuple[np.ndarray, ...],
    noise: xr.DataArray | None,
) -> None:
    """Recompute the fit-quality outputs of one refitted voxel in place.

    ``fit_rss`` and ``fit_objective`` describe the refitted model of the
    enabled lines on the fitted samples, as in the full fit.
    ``fit_n_free_parameters`` counts the enabled lines with their effective
    search windows; ``fit_residual_degrees_of_freedom`` and
    ``fit_reduced_chi_squared`` follow from them. Without carried
    acquisition-tail noise the reduced chi-squared is NaN with
    ``noise_source="none"``. Variables absent from ``results`` are skipped.

    Parameters
    ----------
    results : xarray.Dataset
        Selected-voxel fit results; modified in place.
    fid : numpy.ndarray
        The selected input spectrum in the fit's signal domain.
    amplitudes, frequencies, t2s : numpy.ndarray
        Refitted complex areas, Hz and s of the enabled lines.
    gauss_fwhm_hz : float or None
        Refitted Voigt Gaussian FWHM in Hz; None for Lorentz.
    fit_options : Mapping
        The refit's options with ``sampling_time_axis_s`` and
        ``signal_domain``.
    count_options : Mapping
        The fit-parameter attrs used by ``count_free_parameters``.
    metabolites : sequence of str
        All lines in result order.
    active : numpy.ndarray
        Boolean per line, True for enabled lines.
    windows : tuple of numpy.ndarray
        Full frequency width (Hz), full T2* width (s), min and max T2* (s),
        lower and upper frequency bound (Hz) per line, in result order.
    noise : xarray.DataArray or None
        Scalar acquisition-tail sigma of the selected spectrum.
    """
    from .fit_diagnostics import (
        REDUCED_CHI_SQUARED_DEFINITION,
        calculate_reduced_chi_squared,
    )

    rss, objective = fid_model.final_fit_errors(
        fid,
        amplitudes,
        frequencies,
        t2s,
        fit_options=fit_options,
        gauss_fwhm_hz=gauss_fwhm_hz,
    )
    n_free = fid_model.count_free_parameters(
        active, *windows, metabolites=metabolites, fit_options=dict(count_options)
    )
    dof = 2 * fid_model.fitted_sample_count(len(fid), fit_options) - n_free
    for name, value in (
        ("fit_rss", rss),
        ("fit_objective", objective),
        ("fit_n_free_parameters", n_free),
        ("fit_residual_degrees_of_freedom", dof),
    ):
        if name in results:
            results[name].values[...] = value
    if "fit_reduced_chi_squared" not in results:
        return
    if noise is None:
        results["fit_reduced_chi_squared"].values[...] = np.nan
        results["fit_reduced_chi_squared"].attrs = {
            "unit": "dimensionless",
            "definition": REDUCED_CHI_SQUARED_DEFINITION,
            "noise_source": "none",
        }
        return
    results["fit_reduced_chi_squared"] = calculate_reduced_chi_squared(results, noise)


def _refresh_refit_bound_statuses(
    results: xr.Dataset,
    *,
    metabolites: Sequence[str],
    active: np.ndarray,
    windows: tuple[np.ndarray, ...],
    t2_coupling: Mapping[str, tuple[str, object]] | None,
) -> None:
    """Recompute ``fit_freq_status`` and ``fit_t2_status`` of refitted lines.

    The refitted frequency is compared with the refit's frequency bounds and
    the refitted T2* with its min and max T2*, by the rules of the full fit:
    a zero width is ``fixed`` and a fixed-ratio ``t2_coupling`` dependent
    follows its leader's T2* search. Disabled lines keep their earlier
    labels, as they keep their earlier values. Variables absent from
    ``results`` are skipped; attrs are kept.

    Parameters
    ----------
    results : xarray.Dataset
        Selected-voxel fit results with the refitted ``fit_freqs`` (Hz) and
        ``fit_t2s`` (s); modified in place.
    metabolites : sequence of str
        All lines in result order.
    active : numpy.ndarray
        Boolean per line, True for refitted lines.
    windows : tuple of numpy.ndarray
        Full frequency width (Hz), full T2* width (s), min and max T2* (s),
        lower and upper frequency bound (Hz) per line, in result order.
    t2_coupling : Mapping or None
        Normalized ``{dependent: (leader, t2_ratio)}`` coupling of the refit.
    """
    from ..fitting.spectroscopy import (
        _FREQ_BOUND_TOLERANCE_HZ,
        _T2_BOUND_TOLERANCE_S,
        _bound_status_labels,
    )

    freq_ranges, t2_ranges, min_t2, max_t2, freq_lower, freq_upper = windows
    names = [str(name) for name in metabolites]
    t2_searched = np.asarray(t2_ranges) > 0
    for dependent, (leader, ratio) in (t2_coupling or {}).items():
        if not isinstance(ratio, (tuple, list)):
            t2_searched[names.index(dependent)] = t2_searched[names.index(leader)]
    freq_searched = (np.asarray(freq_ranges) > 0) & (freq_lower < freq_upper)
    outputs = (
        (
            "fit_freq_status",
            "fit_freqs",
            freq_lower,
            freq_upper,
            freq_searched,
            _FREQ_BOUND_TOLERANCE_HZ,
        ),
        (
            "fit_t2_status",
            "fit_t2s",
            min_t2,
            max_t2,
            t2_searched,
            _T2_BOUND_TOLERANCE_S,
        ),
    )
    for name, source, lower, upper, searched, tolerance in outputs:
        if name not in results:
            continue
        values = results[source].transpose("metabolite", ...)
        shape = (len(names),) + (1,) * (values.ndim - 1)
        labels = _bound_status_labels(
            values.values,
            np.reshape(lower, shape),
            np.reshape(upper, shape),
            np.reshape(searched, shape),
            True,
            tolerance=tolerance,
        )
        previous = results[name].transpose("metabolite", ...)
        updated = np.where(
            np.reshape(active, shape), labels, previous.values.astype(str)
        )
        # Rebuild the array: a fixed-width string dtype could truncate names.
        results[name] = previous.copy(data=updated).transpose(*results[name].dims)


def _refresh_refit_linewidths(
    results: xr.Dataset, gauss_fwhm_hz: float | None, options: Mapping
) -> None:
    """Update linewidth outputs of one refitted voxel in place.

    Both lineshapes recompute the widths that are present from the refitted
    ``fit_t2s`` with the definitions of the full fit, keeping each variable's
    attrs. Lorentz results update ``fit_fwhm_hz`` (absorption width
    ``1/(pi*T2*)``) and ``fit_magnitude_fwhm_hz`` (``sqrt(3)/(pi*T2*)``).
    Voigt results also store the refitted Gaussian FWHM and recompute
    ``fit_lorentz_fwhm_hz``, ``fit_fwhm_hz`` (Olivero-Longbothum total
    absorption width), ``fit_magnitude_fwhm_hz`` (exact magnitude width) and
    ``fit_gauss_fwhm_status``.

    Parameters
    ----------
    results : xarray.Dataset
        Selected-voxel fit results with the refitted ``fit_t2s`` in seconds;
        modified in place.
    gauss_fwhm_hz : float or None
        Refitted Gaussian FWHM in Hz for a Voigt refit; ``None`` for Lorentz.
    options : Mapping
        Voigt refit options with ``gauss_max_fwhm_hz`` and
        ``gauss_search_width_hz`` in Hz; unused for Lorentz.
    """
    if gauss_fwhm_hz is None:
        lorentz_fwhm_hz = 1.0 / (np.pi * results["fit_t2s"].astype(float))
        if "fit_fwhm_hz" in results:
            results["fit_fwhm_hz"].values[...] = lorentz_fwhm_hz.transpose(
                *results["fit_fwhm_hz"].dims
            ).values
        if "fit_magnitude_fwhm_hz" in results:
            results["fit_magnitude_fwhm_hz"].values[...] = (
                fid_model.MAGNITUDE_LORENTZ_FWHM_FACTOR
                * lorentz_fwhm_hz.transpose(*results["fit_magnitude_fwhm_hz"].dims)
            ).values
        return
    gauss = results["fit_gauss_fwhm_hz"]
    gauss.values[...] = gauss_fwhm_hz
    lorentz_fwhm_hz = 1.0 / (np.pi * results["fit_t2s"].astype(float))
    if "fit_lorentz_fwhm_hz" in results:
        results["fit_lorentz_fwhm_hz"].values[...] = lorentz_fwhm_hz.values
    if "fit_fwhm_hz" in results:
        results["fit_fwhm_hz"].values[...] = fid_model._voigt_fwhm_hz(
            lorentz_fwhm_hz, gauss.astype(float)
        ).values
    if "fit_magnitude_fwhm_hz" in results:
        results["fit_magnitude_fwhm_hz"].values[...] = fid_model._magnitude_fwhm_hz(
            lorentz_fwhm_hz, gauss.astype(float)
        ).values
    if "fit_gauss_fwhm_status" in results:
        upper_hz = (
            float(options["gauss_max_fwhm_hz"]) - fid_model._GAUSS_BOUND_TOLERANCE_HZ
        )
        if float(options["gauss_search_width_hz"]) == 0:
            status = "fixed"
        elif gauss_fwhm_hz <= fid_model._GAUSS_BOUND_TOLERANCE_HZ:
            status = "at_lower_bound"
        elif gauss_fwhm_hz >= upper_hz:
            status = "at_upper_bound"
        else:
            status = "estimated"
        previous = results["fit_gauss_fwhm_status"]
        new_status = np.full(previous.shape, status, dtype=object)
        if "fit_status" in results:
            fit_status = (
                results["fit_status"].broadcast_like(previous).transpose(*previous.dims)
            )
            new_status[np.asarray(fit_status.values) != "success"] = "not_fitted"
        # Rebuild the array: a fixed-width string dtype could truncate names.
        results["fit_gauss_fwhm_status"] = previous.copy(data=new_status.astype(str))


def _coerce_metab_attr_scalar(value, metab_index: int, default: float) -> float:
    """Return a finite scalar for one metabolite from scalar/list/array attrs."""
    if value is None:
        return float(default)

    try:
        scalar = float(value)
        if np.isfinite(scalar):
            return scalar
    except (TypeError, ValueError):
        pass

    arr = None
    if isinstance(value, xr.DataArray):
        try:
            arr = np.asarray(value.values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            arr = None
    elif isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            arr = None

    if arr is None or arr.size == 0:
        return float(default)

    if arr.size == 1:
        candidate = float(arr[0])
    else:
        idx = min(max(int(metab_index), 0), int(arr.size) - 1)
        candidate = float(arr[idx])

    if not np.isfinite(candidate):
        return float(default)
    return candidate


def _resolve_refit_sampling_time_axis(
    data: xr.DataArray,
    global_attrs: dict,
    expected_size: int,
    scan: Any = None,
) -> np.ndarray:
    """Return a time axis aligned with the FID currently being refitted."""
    expected_size = max(0, int(expected_size))
    if expected_size == 0:
        return np.array([], dtype=float)

    cut_off = max(0, int(global_attrs.get("cut_off", 0) or 0))
    candidates = [_sampling_time_axis_from_data(data)]
    candidates.append(global_attrs.get("sampling_time_axis_s"))

    if scan is not None:
        try:
            candidates.append(getattr(scan, "sampling_time_axis", None))
        except Exception:
            pass
        try:
            candidates.append(getattr(scan, "sampling_time_axis_s", None))
        except Exception:
            pass

    inferred_dwell_time_s = None
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            axis = np.asarray(candidate, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if not axis.size or not np.all(np.isfinite(axis)):
            continue
        if axis.size > 1 and inferred_dwell_time_s is None:
            diffs = np.abs(np.diff(axis))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                inferred_dwell_time_s = float(np.median(diffs))
        if axis.size == expected_size:
            return axis
        if axis.size >= cut_off + expected_size:
            return axis[cut_off : cut_off + expected_size]

    dwell_candidates = [
        getattr(data, "attrs", {}).get("dwell_time_s"),
        global_attrs.get("dwell_time_s"),
        getattr(scan, "dwell_time_s", None),
        inferred_dwell_time_s,
    ]
    for candidate in dwell_candidates:
        try:
            dwell_time_s = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(dwell_time_s) and dwell_time_s > 0:
            return np.arange(expected_size, dtype=float) * dwell_time_s

    return np.arange(expected_size, dtype=float)
