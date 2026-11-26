"""Model comparisons between fits with and without single lines.

``compare_metabolite_relevance`` reports one descriptive M1/M0 RSS comparison
per target. ``eliminate_unnecessary_lines`` removes lines of one overlapping
group by backward elimination with an information criterion.
``add_line_from_residual`` is the forward step: it proposes one line at the
largest residual peak and keeps it by the same criterion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _prepare_single_spectrum_data,
    _promote_t2_coupling,
    _require_same_lineshape,
    _uses_voigt_lineshape,
    fitted_sample_count,
)
from hypermri.analysis.fitting.spectroscopy import (
    _carried_acquisition_noise,
    add_metab_to_fit_params,
    fit_data_pseudo_inv,
    initialise_fit_params_from_results,
    reconstruct_fitted_signal,
)

from .fit_diagnostics import NoiseInput, _fit_parameter_like, noise_sigma_for_fit
from .fitting import _noise_template, _prepare_fit_params, fit_spectrum


def _relevance_selection(
    data: xr.DataArray,
    fit_params: xr.Dataset,
    selection: xr.DataArray | str | None,
) -> xr.DataArray:
    """Return a labelled mask over spectra; None means all spectra."""
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(
            f"Input data is missing the spectral dimension '{spectral_dim}'."
        )
    template = data.isel({spectral_dim: 0}, drop=True)
    if selection is None:
        return template.copy(data=np.ones(template.shape, dtype=bool))
    if isinstance(selection, str):
        if selection != "all":
            raise ValueError("String selection must be exactly 'all'.")
        return template.copy(data=np.ones(template.shape, dtype=bool))
    if not isinstance(selection, xr.DataArray) or selection.dtype.kind != "b":
        raise TypeError("selection must be a boolean xarray.DataArray or 'all'.")
    unknown = set(selection.dims) - set(template.dims)
    if unknown:
        raise ValueError(
            "Selection dimensions are absent from the fitted spectra: "
            f"{sorted(unknown)}."
        )
    try:
        selection, template = xr.align(selection, template, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "Selection coordinates must exactly match the fitted spectra."
        ) from exc
    result = selection.broadcast_like(template).transpose(*template.dims)
    if not bool(np.asarray(result.any().values)):
        raise ValueError("Relevance selection does not contain any spectra.")
    return result


def compare_metabolite_relevance(
    data: xr.DataArray,
    fit_params: xr.Dataset | Mapping[str, object],
    fit_results: xr.Dataset | None = None,
    *,
    targets: str | Sequence[str],
    selection: xr.DataArray | str | None = None,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
) -> xr.Dataset:
    """Compare the full M1 fit with one refitted M0 model per target.

    This is a quick descriptive relevance comparison.  It returns the observed
    RSS improvement but does not attach a p-value or call it a confidence
    measure.  A calibrated detection decision still requires a null
    distribution. M0 refits use the lineshape of ``fit_params``, so a Voigt
    M0 keeps fitting the shared Gaussian FWHM.

    Parameters
    ----------
    data : xr.DataArray
        Complex temporal FIDs.
    fit_params : xr.Dataset or Mapping
        Ordinary fit parameters, or a mapping for ``def_fit_params``, which
        rejects unknown keys with ``ValueError``.
    fit_results : xr.Dataset, optional
        Existing M1 results of ``fit_params``; fitted when omitted.
    targets : str or sequence of str
        Unique metabolites removed one at a time to form each M0.
    selection : xr.DataArray or str, optional
        Boolean labelled selection of spectra, or ``'all'``.
    n_jobs, batch_size, parallel_backend
        Execution options forwarded unchanged to ``fit_data_pseudo_inv``.

    Returns
    -------
    xr.Dataset
        ``rss_m1``, ``rss_m0``, ``rss_improvement`` (RSS unit),
        ``relative_rss_improvement`` and ``m0_fit_status`` with a ``target``
        dimension, and the boolean ``relevance_selected``.

    Raises
    ------
    TypeError
        If ``data`` or ``fit_results`` has the wrong type.
    ValueError
        If ``detection_target`` is set, the targets are invalid, the
        selection is invalid, or ``fit_results`` uses another lineshape than
        ``fit_params``.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray.")
    prepared = _prepare_fit_params(data, fit_params)
    if prepared.attrs.get("detection_target") is not None:
        raise ValueError(
            "Quick relevance requires ordinary fit parameters without "
            "detection_target settings."
        )
    target_names = [targets] if isinstance(targets, str) else [str(v) for v in targets]
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("targets must contain unique metabolite names.")
    available = [str(value) for value in prepared.metabolite.values.tolist()]
    missing = [target for target in target_names if target not in available]
    if missing:
        raise ValueError(f"Unknown relevance targets: {missing}.")

    selected = _relevance_selection(data, prepared, selection)
    if fit_results is None:
        fit_results = fit_spectrum(
            data,
            prepared,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
        )
    elif not isinstance(fit_results, xr.Dataset):
        raise TypeError("fit_results must be an xarray.Dataset.")
    else:
        _require_same_lineshape("compare_metabolite_relevance", prepared, fit_results)

    spectral_dim = prepared.attrs.get("spectral_dim", "acquisition_idx")
    selected_input = data.where(selected.broadcast_like(data), other=0.0 + 0.0j)
    m0_rss = []
    m0_status = []
    for target in target_names:
        m0_params = prepared.copy(deep=True)
        target_mask = m0_params.metabolite == target
        m0_params["skip_fitting"] = m0_params.skip_fitting | target_mask
        m0_fit = fit_data_pseudo_inv(
            selected_input,
            m0_params,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
            show_tqdm=False,
        )
        m0_rss.append(m0_fit.fit_rss.where(selected))
        m0_status.append(m0_fit.fit_status.where(selected, other="not_selected"))

    target_coord = xr.IndexVariable("target", target_names)
    rss_m0 = xr.concat(m0_rss, dim=target_coord)
    rss_m1 = fit_results.fit_rss.where(selected).expand_dims(target=target_coord)
    status_m0 = xr.concat(m0_status, dim=target_coord)
    improvement = rss_m0 - rss_m1
    relative = xr.where(rss_m0 > 0, improvement / rss_m0, np.nan)
    result = xr.Dataset(
        {
            "rss_m1": rss_m1,
            "rss_m0": rss_m0,
            "rss_improvement": improvement,
            "relative_rss_improvement": relative,
            "m0_fit_status": status_m0,
            "relevance_selected": selected,
        }
    )
    for name in ("rss_m1", "rss_m0", "rss_improvement"):
        result[name].attrs["unit"] = fit_results.fit_rss.attrs.get("unit", "a.u.^2")
    result.rss_improvement.attrs["definition"] = "rss_m0 - rss_m1; no clipping"
    result.relative_rss_improvement.attrs["definition"] = (
        "(rss_m0 - rss_m1) / rss_m0; descriptive, not a p-value"
    )
    result.attrs.update(
        relevance_method="one fully refitted M0 model per target",
        relevance_scope=(
            "Observed RSS improvement only; not a p-value, no null calibration "
            "and no parameter confidence interval."
        ),
        spectral_dim=spectral_dim,
    )
    return result


_MODEL_COMPARISON_UNSUPPORTED_OPTIONS = {
    "detection_target": (
        "the information criterion compares ordinary fits; run detection separately"
    ),
    "use_sse_mag": (
        "the information criterion needs fits that minimise the temporal "
        "complex RSS; set use_sse_mag=False"
    ),
    "store_optimization_history": (
        "refits of different spectra are combined into one result; set "
        "store_optimization_history=False"
    ),
}


def _necessity_candidates(candidates: Sequence[str], names: list[str]) -> list[str]:
    """Return validated candidate names in the order given."""
    if isinstance(candidates, str):
        raise TypeError(
            "eliminate_unnecessary_lines: candidates must be a sequence of at "
            f"least two line names, not the single string {candidates!r}."
        )
    candidate_names = [str(name) for name in candidates]
    if len(candidate_names) < 2 or len(set(candidate_names)) != len(candidate_names):
        raise ValueError(
            "eliminate_unnecessary_lines: candidates must contain at least two "
            f"unique line names; got {candidate_names}."
        )
    unknown = [name for name in candidate_names if name not in names]
    if unknown:
        raise ValueError(
            f"eliminate_unnecessary_lines: unknown candidates {unknown}; "
            f"fit_params contains {names}."
        )
    return candidate_names


def _n_free_parameters(
    names: list[str], active: tuple[bool, ...], coupling: dict | None, voigt: bool
) -> int:
    """Count the real parameters of the lines a fit actually used.

    Each line has a frequency and a complex amplitude (3). A line adds its own
    T2* (1) unless it follows a fitted leader under ``t2_coupling``, after the
    leader promotion of the fitter. A Voigt fit adds one shared Gaussian FWHM.
    """
    if not any(active):
        return 0
    promoted, _ = _promote_t2_coupling(
        {"metabs": names, "t2_coupling": coupling},
        np.asarray(active),
        np.zeros(len(names)),
    )
    dependents = set(promoted["t2_coupling"] or {})
    own_t2 = sum(
        keep and name not in dependents
        for name, keep in zip(names, active, strict=True)
    )
    return 3 * sum(active) + own_t2 + int(voigt)


def _rows(values: xr.DataArray, loop_dims: list[str], *trailing: str) -> np.ndarray:
    """Return labelled values as rows of spectra in ``loop_dims`` order."""
    array = values.transpose(*loop_dims, *trailing).values
    return array.reshape(-1, *array.shape[len(loop_dims) :])


def _fitted_lines(result: xr.Dataset, loop_dims: list[str]) -> np.ndarray:
    """Return which lines each spectrum fitted, as (spectra, metabolites)."""
    excluded = (result.fit_skipped | result.fit_low_snr_refitted).broadcast_like(
        result.fit_amps
    )
    return ~_rows(excluded, loop_dims, "metabolite").astype(bool)


def _take_spectra(
    current: xr.Dataset, trial: xr.Dataset, take: xr.DataArray
) -> xr.Dataset:
    """Return ``current`` with the spectra marked in ``take`` from ``trial``."""
    merged = current.copy()
    for name, values in current.data_vars.items():
        if set(take.dims) <= set(values.dims):
            merged[name] = trial[name].where(take, values).transpose(*values.dims)
    return merged


def eliminate_unnecessary_lines(
    data: xr.DataArray,
    fit_params: xr.Dataset | Mapping[str, object],
    *,
    candidates: Sequence[str],
    noise: NoiseInput,
    selection: xr.DataArray | str | None = None,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
) -> xr.Dataset:
    """Remove lines of one overlapping group that the data do not need.

    Overlapping lines can trade signal with each other: an extra line may
    lower the RSS only by cancelling part of a neighbour. This backward
    elimination keeps a line only if removing it makes the fit clearly worse.
    Each spectrum is handled on its own:

    1. Fit the full model M1 with ``fit_params``.
    2. For every candidate line still fitted, refit without it (M0) from the
       start parameters of ``fit_params``, not from the M1 result.
    3. Compute ``delta_chi2 = (RSS_M0 - RSS_M1) / s2`` with
       ``s2 = max(sigma**2, RSS_M1 / (2 N - k_M1))``. ``N`` is the number of
       fitted complex samples and ``k_M1`` the number of real parameters of
       M1 (three per line for frequency and complex amplitude, one per own
       T2* and one shared Gaussian FWHM for Voigt fits). The larger of the
       two variances keeps model errors from making every line look needed.
    4. Remove the line with the smallest ``delta_chi2 - delta_k * ln(2 N)``
       if this value is negative, where ``delta_k = k_M1 - k_M0`` is the
       number of parameters the line adds. This is the Bayesian information
       criterion with ``2 N`` real data values. Its refit becomes the new M1.
    5. Repeat until no line is removed or one candidate remains.

    The candidates are assumed to belong to a signal that is present, so at
    least one of them always stays. Whether the group is present at all is a
    separate question for :func:`compare_metabolite_relevance` or the
    detection test. Without a phase constraint, spurious lines that cancel
    each other can still look necessary; combine this test with
    ``phase_band_tolerance_deg`` (see ``def_fit_params``) to tie the lines
    to one phase. Lines not in ``candidates`` are fitted but never tested.

    Parameters
    ----------
    data : xr.DataArray
        Complex temporal FIDs with the spectral dimension of ``fit_params``.
    fit_params : xr.Dataset or Mapping
        Ordinary fit parameters, or a mapping for ``def_fit_params``, which
        rejects unknown keys with ``ValueError``. Its ``t2_coupling`` and
        phase-band options apply to every fit.
    candidates : sequence of str
        At least two unique names of one group of overlapping lines, for
        example the components of one multiplet.
    noise : float, xr.DataArray or xr.Dataset
        Standard deviation per real or imaginary quadrature of the noise in
        the fitted temporal FID samples, in the signal unit; a scalar or one
        value per spectrum (see
        :func:`~hypermri.analysis.spectroscopy.fit_diagnostics.noise_sigma_for_fit`).
        It must be finite and positive for every selected spectrum.
    selection : xr.DataArray or str, optional
        Boolean labelled selection of spectra, or ``'all'`` (default when
        ``None``). Other spectra are not fitted and keep ``fit_status``
        ``'skipped'``.
    n_jobs, batch_size, parallel_backend
        Execution options forwarded unchanged to every fit.

    Returns
    -------
    xr.Dataset
        The fit results of ``fit_spectrum`` for the final model of each
        spectrum; removed lines appear as skipped (``fit_skipped`` True).
        Added variables:

        ``necessity_removed`` (spectra x metabolite, bool)
            Line removed by this test.
        ``necessity_removal_step`` (spectra x metabolite, int16)
            Step at which the line was removed, counting from 1; 0 if kept.
        ``necessity_delta_chi2`` (spectra x metabolite, dimensionless)
            ``delta_chi2`` of the line's last test: at its removal for removed
            lines, otherwise from the last step that tested it. NaN if the
            line was never tested or its refit failed.
        ``necessity_threshold`` (spectra x metabolite, dimensionless)
            ``delta_k * ln(2 N)`` belonging to ``necessity_delta_chi2``.
        ``necessity_tested`` (spectra, bool)
            Spectrum selected, M1 fit successful and at least two candidates
            fitted, so that at least one test ran.
        ``necessity_noise_sigma`` (spectra)
            The supplied quadrature noise SD.

        Attributes ``necessity_candidates``, ``necessity_n_samples`` (``N``),
        ``necessity_method`` and ``necessity_criterion`` record the settings.

    Raises
    ------
    TypeError
        If ``data``, ``candidates``, ``noise`` or ``selection`` has the
        wrong type.
    ValueError
        If the candidates are invalid, the noise is not finite and positive
        for a selected spectrum, the selection does not match the spectra,
        ``fit_params`` sets ``detection_target``, ``use_sse_mag`` or
        ``store_optimization_history``, or a fit option is invalid.

    Notes
    -----
    The test runs one full fit plus, per step, one refit per remaining
    candidate over the spectra still being tested; spectra that are finished
    are skipped in later refits. It is deterministic and does not modify
    its inputs. ``fit_low_snr_refitted`` lines count as not fitted.

    Examples
    --------
    >>> result = eliminate_unnecessary_lines(
    ...     fids,
    ...     params,
    ...     candidates=["OMPD1_1", "OMPD1_2", "OMPD1_3"],
    ...     noise=0.02,
    ... )  # doctest: +SKIP
    >>> result.necessity_removed.sel(metabolite="OMPD1_3")  # doctest: +SKIP
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray.")
    prepared = _prepare_fit_params(data, fit_params)
    for option, reason in _MODEL_COMPARISON_UNSUPPORTED_OPTIONS.items():
        if prepared.attrs.get(option):
            raise ValueError(
                f"eliminate_unnecessary_lines does not support {option}: {reason}."
            )
    names = [str(value) for value in prepared.metabolite.values.tolist()]
    candidate_names = _necessity_candidates(candidates, names)
    candidate_index = [names.index(name) for name in candidate_names]

    template = _noise_template(data, prepared)
    loop_dims = list(template.dims[:-1])
    spectra_template = template.isel(metabolite=0, drop=True)
    selected = _rows(_relevance_selection(data, prepared, selection), loop_dims).astype(
        bool
    )
    sigma = noise_sigma_for_fit(noise, template)
    sigma_values = _rows(sigma, loop_dims)
    if not np.all(np.isfinite(sigma_values[selected]) & (sigma_values[selected] > 0)):
        raise ValueError(
            "eliminate_unnecessary_lines: noise must be finite and positive for "
            "every selected spectrum; restrict selection or correct the noise map."
        )
    spectral_dim = prepared.attrs.get("spectral_dim", "acquisition_idx")
    n_samples = _prepare_single_spectrum_data(
        np.zeros(data.sizes[spectral_dim], dtype=complex), dict(prepared.attrs)
    )[0].size
    log_n = np.log(2 * n_samples)
    base_skip = _rows(
        _fit_parameter_like(prepared, "skip_fitting", template), loop_dims, "metabolite"
    ).astype(bool)
    coupling = prepared.attrs.get("t2_coupling")
    voigt = _uses_voigt_lineshape(prepared.attrs)
    parameter_counts: dict[tuple[bool, ...], int] = {}

    def n_parameters(fitted: np.ndarray) -> np.ndarray:
        """Return the parameter count of each row of fitted lines."""
        counts = []
        for row in fitted:
            key = tuple(bool(value) for value in row)
            if key not in parameter_counts:
                parameter_counts[key] = _n_free_parameters(names, key, coupling, voigt)
            counts.append(parameter_counts[key])
        return np.asarray(counts, dtype=float)

    def fit_without(skip: np.ndarray, spectra: np.ndarray) -> xr.Dataset:
        """Fit ``spectra`` with the lines in ``skip`` skipped; others not at all."""
        params = prepared.copy(deep=True)
        skip = np.where(spectra[:, None], skip, True)
        params["skip_fitting"] = xr.DataArray(
            skip.reshape(template.shape), dims=template.dims, coords=template.coords
        )
        return fit_spectrum(
            data,
            params,
            show_tqdm=False,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
        )

    def labelled(values: np.ndarray, *trailing: str) -> xr.DataArray:
        like = template if trailing else spectra_template
        return xr.DataArray(
            values.reshape(like.shape), dims=like.dims, coords=like.coords
        )

    removed = np.zeros_like(base_skip)
    current = fit_without(base_skip, selected)
    fitted = _fitted_lines(current, loop_dims)
    rss = _rows(current.fit_rss, loop_dims).astype(float)
    testing = selected & (_rows(current.fit_status, loop_dims) == "success")
    testing &= fitted[:, candidate_index].sum(axis=1) >= 2
    tested = testing.copy()
    removal_step = np.zeros(base_skip.shape, dtype=np.int16)
    delta_chi2 = np.full(base_skip.shape, np.nan)
    threshold = np.full(base_skip.shape, np.nan)

    step = 0
    while np.any(testing):
        step += 1
        k_m1 = n_parameters(fitted)
        with np.errstate(divide="ignore", invalid="ignore"):
            s2 = np.maximum(sigma_values**2, rss / (2 * n_samples - k_m1))
        score = np.full((len(rss), len(candidate_index)), np.inf)
        trials = []
        for column, line in enumerate(candidate_index):
            spectra = testing & fitted[:, line]
            if not np.any(spectra):
                trials.append(None)
                continue
            skip = base_skip | removed
            skip[:, line] = True
            trial = fit_without(skip, spectra)
            trial_fitted = _fitted_lines(trial, loop_dims)
            trial_rss = _rows(trial.fit_rss, loop_dims).astype(float)
            valid = (
                spectra
                & (_rows(trial.fit_status, loop_dims) == "success")
                & np.isfinite(trial_rss)
            )
            line_delta_chi2 = (trial_rss[valid] - rss[valid]) / s2[valid]
            line_threshold = (k_m1[valid] - n_parameters(trial_fitted[valid])) * log_n
            delta_chi2[valid, line] = line_delta_chi2
            threshold[valid, line] = line_threshold
            score[valid, column] = line_delta_chi2 - line_threshold
            trials.append((trial, trial_fitted, trial_rss))

        best = np.argmin(score, axis=1)
        testing &= score[np.arange(len(best)), best] < 0
        for column, line in enumerate(candidate_index):
            take = testing & (best == column)
            if not np.any(take):
                continue
            trial, trial_fitted, trial_rss = trials[column]
            current = _take_spectra(current, trial, labelled(take))
            fitted[take] = trial_fitted[take]
            rss[take] = trial_rss[take]
            removed[take, line] = True
            removal_step[take, line] = step
        testing &= fitted[:, candidate_index].sum(axis=1) >= 2

    result = current.assign(
        necessity_removed=labelled(removed, "metabolite"),
        necessity_removal_step=labelled(removal_step, "metabolite"),
        necessity_delta_chi2=labelled(delta_chi2, "metabolite"),
        necessity_threshold=labelled(threshold, "metabolite"),
        necessity_tested=labelled(tested),
        necessity_noise_sigma=sigma,
    )
    result.necessity_removed.attrs["description"] = (
        "Line removed by the necessity backward elimination"
    )
    result.necessity_removal_step.attrs["description"] = (
        "Elimination step of the removal, counting from 1; 0 if kept"
    )
    result.necessity_delta_chi2.attrs.update(
        unit="1",
        definition=(
            "(RSS_M0 - RSS_M1) / max(sigma**2, RSS_M1 / (2 N - k_M1)) of the "
            "line's last test"
        ),
    )
    result.necessity_threshold.attrs.update(
        unit="1", definition="(k_M1 - k_M0) * ln(2 N); the line is removable below"
    )
    result.necessity_tested.attrs["description"] = (
        "Selected spectrum with a successful full fit and at least two fitted "
        "candidates"
    )
    result.attrs.update(
        necessity_candidates=candidate_names,
        necessity_n_samples=int(n_samples),
        necessity_method=(
            "backward elimination: refit without each remaining candidate from "
            "the start parameters and remove the least necessary line"
        ),
        necessity_criterion=(
            "remove the line with the smallest delta_chi2 - (k_M1 - k_M0) * "
            "ln(2 N) while it is negative; at least one candidate stays"
        ),
    )
    return result


_RESIDUAL_ZERO_FILL = 8
"""Zero-filling factor of the residual spectrum that places a proposed line.

It only sets the grid of start frequencies (an eighth of a spectral point);
the nested fit refines the frequency inside the new line's window.
"""


def _search_range(
    search_range_hz: tuple[float, float],
    window_hz: float,
    axis_hz: np.ndarray,
) -> tuple[float, float]:
    """Return the validated search range; proposals and windows stay inside."""
    try:
        low, high = (float(value) for value in search_range_hz)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "add_line_from_residual: search_range_hz must be two frequencies "
            f"(low, high) in Hz; got {search_range_hz!r}."
        ) from exc
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError(
            "add_line_from_residual: search_range_hz must be finite with "
            f"low < high; got ({low}, {high}) Hz."
        )
    lowest = float(np.min(axis_hz)) + window_hz / 2
    highest = float(np.max(axis_hz)) - window_hz / 2
    if low < lowest or high > highest:
        raise ValueError(
            f"add_line_from_residual: search_range_hz ({low:.6g}, {high:.6g}) "
            f"Hz with a {window_hz:.6g} Hz line window leaves the spectrum; "
            f"use a range inside ({lowest:.6g}, {highest:.6g}) Hz."
        )
    return low, high


def _residual_peaks(
    residual: np.ndarray,
    dwell_time_s: float,
    *,
    search_range_hz: tuple[float, float],
    fitted_freqs_hz: np.ndarray,
    min_separation_hz: float,
) -> np.ndarray:
    """Return the largest residual peak per row in Hz; NaN if none is allowed.

    ``residual`` has the rows of spectra and the fitted FID samples. The
    magnitude spectrum is zero-filled by ``_RESIDUAL_ZERO_FILL``. Frequencies
    closer than ``min_separation_hz`` to a fitted line of that row
    (``fitted_freqs_hz``, NaN for lines not fitted) are not allowed, so the
    proposed start keeps the fit's minimum peak separation.
    """
    n_points = _RESIDUAL_ZERO_FILL * residual.shape[1]
    freqs = np.fft.fftfreq(n_points, dwell_time_s)
    magnitude = np.abs(np.fft.fft(residual, n=n_points, axis=1))
    low, high = search_range_hz
    allowed = np.broadcast_to((freqs >= low) & (freqs <= high), magnitude.shape)
    with np.errstate(invalid="ignore"):
        near = (
            np.abs(freqs[None, None, :] - fitted_freqs_hz[:, :, None])
            < min_separation_hz
        )
    allowed = allowed & ~near.any(axis=1) & np.isfinite(magnitude)
    score = np.where(allowed, magnitude, -np.inf)
    best = np.argmax(score, axis=1)
    found = np.isfinite(score[np.arange(len(best)), best])
    return np.where(found, freqs[best], np.nan)


def add_line_from_residual(
    data: xr.DataArray,
    fit_params: xr.Dataset | Mapping[str, object],
    *,
    search_range_hz: tuple[float, float],
    noise: NoiseInput | None = None,
    new_metab_name: str = "residual_line",
    line_window_hz: float | None = None,
    selection: xr.DataArray | str | None = None,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
) -> xr.Dataset:
    """Add one line at the largest residual peak where the data need it.

    This forward step is the counterpart of
    :func:`eliminate_unnecessary_lines` and uses the same information
    criterion. Each spectrum is handled on its own:

    1. Fit the parent model M0 with ``fit_params``, extended by the new line
       but with that line skipped. A skipped line does not enter the fit, so
       M0 is the fit of ``fit_params`` itself.
    2. Take the magnitude spectrum of the residual (data minus M0 over the
       fitted FID samples, zero-filled 8 times) and propose the new line at
       its largest value inside ``search_range_hz``. Frequencies closer than
       the fit's ``min_peak_separation_hz`` to a fitted line are excluded.
    3. Fit M1 from a nested start: every line starts at its M0 result and the
       new line at the proposed frequency with the ``def_fit_params`` T2*
       defaults and a frequency window of ``line_window_hz``. The search
       accepts only improvements, so M1 never ends above M0's RSS.
    4. Compute ``delta_chi2 = (RSS_M0 - RSS_M1) / s2`` with
       ``s2 = max(sigma**2, RSS_M1 / (2 N - k_M1))`` and keep the line if
       ``delta_chi2 > (k_M1 - k_M0) * ln(2 N)``, the Bayesian information
       criterion with ``2 N`` real data values. ``N`` is the number of fitted
       complex samples and ``k`` the number of real parameters (three per
       line for frequency and complex amplitude, one per own T2*, one shared
       Gaussian FWHM for Voigt fits), so the new line adds four.

    The residual peak is the most likely place of a missing line but not
    always the best one: a misfit of an existing line can leave the largest
    residual. Restrict ``search_range_hz`` to where a line is expected, and
    review accepted lines. Whether the line belongs to a known metabolite is
    a separate, scientific decision.

    Parameters
    ----------
    data : xr.DataArray
        Complex temporal FIDs with the spectral dimension of ``fit_params``.
    fit_params : xr.Dataset or Mapping
        Ordinary fit parameters of the parent model, or a mapping for
        ``def_fit_params``, which rejects unknown keys with ``ValueError``.
        Its lineshape, ``t2_coupling`` and phase-band options apply to both
        fits; the new line is not coupled.
    search_range_hz : tuple of float
        ``(low, high)`` frequencies in Hz, in the convention of
        ``initial_freqs_hz`` (the fit's Hz axis), where the new line may
        start. The range widened by half the line window on each side must
        lie inside the spectrum.
    noise : float, xr.DataArray, xr.Dataset or None, optional
        Standard deviation per real or imaginary quadrature of the noise in
        the fitted temporal FID samples, in the signal unit; a scalar or one
        value per spectrum (see
        :func:`~hypermri.analysis.spectroscopy.fit_diagnostics.noise_sigma_for_fit`).
        None uses the acquisition-tail noise the input carries as coordinate
        ``noise_sigma_quadrature`` (from
        :func:`~hypermri.reconstruction.csi_noise.propagate_acquisition_tail_noise`).
        It must be finite and positive for every selected spectrum.
    new_metab_name : str, optional
        Name of the new line; it must not be in ``fit_params``.
    line_window_hz : float or None, optional
        Full frequency search width of the new line in Hz around its proposed
        start. None uses two spectral points of the fitted FID,
        ``2 / (N * dwell time)``, the narrowest window that lets the line
        move one point to either side.
    selection : xr.DataArray or str, optional
        Boolean labelled selection of spectra, or ``'all'`` (default when
        ``None``). Other spectra are not fitted and keep ``fit_status``
        ``'skipped'``.
    n_jobs, batch_size, parallel_backend
        Execution options forwarded unchanged to both fits.

    Returns
    -------
    xr.Dataset
        The fit results of ``fit_spectrum`` with the new line as the last
        metabolite: the M1 fit where the line is kept, the M0 fit (new line
        skipped, ``fit_skipped`` True, amplitude 0) elsewhere. Added
        variables over the spectra:

        ``line_search_accepted`` (bool)
            The new line is kept.
        ``line_search_tested`` (bool)
            Spectrum selected, M0 fit successful, an allowed proposal found
            and the M1 fit successful, so that the criterion was evaluated.
        ``line_search_start_freq_hz`` (Hz)
            Proposed start frequency; NaN without a proposal.
        ``line_search_delta_chi2`` (dimensionless)
            ``delta_chi2`` as defined above; NaN if not tested.
        ``line_search_threshold`` (dimensionless)
            ``(k_M1 - k_M0) * ln(2 N)``; NaN if not tested.
        ``line_search_noise_sigma``
            The quadrature noise SD that was used.

        Attributes ``line_search_new_line``, ``line_search_range_hz``,
        ``line_search_window_hz``, ``line_search_n_samples`` (``N``),
        ``line_search_method`` and ``line_search_criterion`` record the
        settings.

    Raises
    ------
    TypeError
        If ``data``, ``search_range_hz``, ``noise`` or ``selection``
        has the wrong type.
    ValueError
        If no noise is given or carried, the noise is not finite and positive
        for a selected spectrum, the search range or window is invalid or
        leaves the spectrum, ``new_metab_name`` already exists, the data are
        spectral, ``fit_params`` sets ``detection_target``, ``use_sse_mag``
        or ``store_optimization_history``, or a fit option is invalid.

    Notes
    -----
    Two fits over the selected spectra; deterministic; the inputs are not
    modified. ``fit_low_snr_refitted`` lines count as not fitted.

    Examples
    --------
    >>> result = add_line_from_residual(
    ...     fids,
    ...     params,
    ...     search_range_hz=(330.0, 360.0),
    ...     new_metab_name="OMPD5_4",
    ... )  # doctest: +SKIP
    >>> result.line_search_accepted.sum()  # doctest: +SKIP
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray.")
    prepared = _prepare_fit_params(data, fit_params)
    for option, reason in _MODEL_COMPARISON_UNSUPPORTED_OPTIONS.items():
        if prepared.attrs.get(option):
            raise ValueError(
                f"add_line_from_residual does not support {option}: {reason}."
            )
    if str(prepared.attrs.get("signal_domain", "temporal")).lower() != "temporal":
        raise ValueError(
            "add_line_from_residual needs temporal FIDs; fit_params has "
            f"signal_domain={prepared.attrs.get('signal_domain')!r}."
        )
    names = [str(value) for value in prepared.metabolite.values.tolist()]
    if new_metab_name in names:
        raise ValueError(
            f"add_line_from_residual: new_metab_name {new_metab_name!r} is "
            f"already a line of fit_params {names}; choose another name."
        )
    times_s = prepared.attrs.get("sampling_time_axis_s")
    if times_s is None or np.size(times_s) < 2:
        raise ValueError(
            "add_line_from_residual: fit_params has no sampling_time_axis_s; "
            "build it with def_fit_params from the input data."
        )
    spectral_dim = prepared.attrs.get("spectral_dim", "acquisition_idx")
    n_samples = fitted_sample_count(data.sizes[spectral_dim], dict(prepared.attrs))
    dwell_time_s = float(np.mean(np.diff(np.asarray(times_s, dtype=float))))
    if line_window_hz is None:
        window_hz = 2.0 / (n_samples * dwell_time_s)
    else:
        window_hz = float(line_window_hz)
        if not (np.isfinite(window_hz) and window_hz > 0):
            raise ValueError(
                "add_line_from_residual: line_window_hz must be finite and "
                f"positive; got {line_window_hz!r}."
            )
    axis_hz = prepared.attrs.get("freq_axis_hz")
    if axis_hz is None:
        axis_hz = np.array([-0.5, 0.5]) / dwell_time_s
    low, high = _search_range(search_range_hz, window_hz, np.asarray(axis_hz))

    template = _noise_template(data, prepared)
    loop_dims = list(template.dims[:-1])
    spectra_template = template.isel(metabolite=0, drop=True)
    selected = _rows(_relevance_selection(data, prepared, selection), loop_dims).astype(
        bool
    )
    if noise is None:
        noise = _carried_acquisition_noise(data)
        if noise is None:
            raise ValueError(
                "add_line_from_residual: pass noise, or give input data that "
                "carry the acquisition-tail noise coordinate "
                "'noise_sigma_quadrature'."
            )
    sigma = noise_sigma_for_fit(noise, template)
    sigma_values = _rows(sigma, loop_dims)
    if not np.all(np.isfinite(sigma_values[selected]) & (sigma_values[selected] > 0)):
        raise ValueError(
            "add_line_from_residual: noise must be finite and positive for "
            "every selected spectrum; restrict selection or correct the noise map."
        )

    def labelled(values: np.ndarray, *trailing: str) -> xr.DataArray:
        like = extended_template if trailing else spectra_template
        return xr.DataArray(
            values.reshape(like.shape), dims=like.dims, coords=like.coords
        )

    def fit(params: xr.Dataset) -> xr.Dataset:
        return fit_spectrum(
            data,
            params,
            show_tqdm=False,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
        )

    # The line is added while every line is skipped, so that its placeholder
    # start cannot violate the minimum separation; the skip map is set after.
    base_skip = _rows(
        _fit_parameter_like(prepared, "skip_fitting", template), loop_dims, "metabolite"
    ).astype(bool)
    all_skipped = prepared.copy(deep=True)
    all_skipped["skip_fitting"] = xr.ones_like(template, dtype=bool)
    extended = add_metab_to_fit_params(
        0.5 * (low + high),
        new_metab_name=new_metab_name,
        fit_params=all_skipped,
        new_freq_range=window_hz,
        freeze_other_freqs=False,
        freeze_other_t2s=False,
    )
    extended_template = _noise_template(data, extended)
    new_line = np.asarray(extended.metabolite.values == new_metab_name)
    names = [str(value) for value in extended.metabolite.values.tolist()]

    def skip_map(old_lines: np.ndarray, new: np.ndarray) -> xr.DataArray:
        values = np.empty((len(old_lines), len(names)), dtype=bool)
        values[:, ~new_line] = old_lines
        values[:, new_line] = new[:, None]
        return labelled(values, "metabolite")

    m0_params = extended.copy()
    m0_params["skip_fitting"] = skip_map(
        np.where(selected[:, None], base_skip, True), np.ones(len(selected), bool)
    )
    m0 = fit(m0_params)
    m0_rss = _rows(m0.fit_rss, loop_dims).astype(float)
    m0_ok = selected & (_rows(m0.fit_status, loop_dims) == "success")
    m0_fitted = _fitted_lines(m0, loop_dims)

    fitted_samples = data.isel({spectral_dim: slice(0, n_samples)})
    fitted_samples.attrs = {
        **fitted_samples.attrs,
        "sampling_time_axis_s": m0.attrs["sampling_time_axis_s"],
    }
    residual = _rows(
        fitted_samples - reconstruct_fitted_signal(m0, fitted_samples),
        loop_dims,
        spectral_dim,
    )
    fitted_freqs = np.where(
        m0_fitted, _rows(m0.fit_freqs, loop_dims, "metabolite"), np.nan
    )
    start_freqs = np.full(len(selected), np.nan)
    start_freqs[m0_ok] = _residual_peaks(
        residual[m0_ok],
        dwell_time_s,
        search_range_hz=(low, high),
        fitted_freqs_hz=fitted_freqs[m0_ok],
        min_separation_hz=float(prepared.attrs.get("min_peak_separation_hz", 0.0)),
    )
    proposed = np.isfinite(start_freqs)

    seeds = m0.copy()
    seeds["fit_freqs"] = m0.fit_freqs.where(
        ~(labelled(proposed) & (m0.metabolite == new_metab_name)),
        labelled(np.where(proposed, start_freqs, 0.0)),
    ).transpose(*m0.fit_freqs.dims)
    m1_params = initialise_fit_params_from_results(m0_params, seeds)
    m1_params["skip_fitting"] = skip_map(
        np.where(proposed[:, None], base_skip, True), ~proposed
    )
    m1 = fit(m1_params)
    m1_rss = _rows(m1.fit_rss, loop_dims).astype(float)
    tested = (
        proposed
        & (_rows(m1.fit_status, loop_dims) == "success")
        & np.isfinite(m1_rss)
        & ~_rows(m1.fit_low_snr_refitted, loop_dims, "metabolite")[:, new_line][:, 0]
    )

    coupling = extended.attrs.get("t2_coupling")
    voigt = _uses_voigt_lineshape(extended.attrs)
    log_n = np.log(2 * n_samples)
    k_m0 = np.array(
        [_n_free_parameters(names, tuple(row), coupling, voigt) for row in m0_fitted]
    )
    k_m1 = np.array(
        [
            _n_free_parameters(names, tuple(row), coupling, voigt)
            for row in _fitted_lines(m1, loop_dims)
        ]
    )
    delta_chi2 = np.full(len(selected), np.nan)
    threshold = np.full(len(selected), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.maximum(sigma_values**2, m1_rss / (2 * n_samples - k_m1))
        delta_chi2[tested] = (m0_rss[tested] - m1_rss[tested]) / s2[tested]
    threshold[tested] = (k_m1[tested] - k_m0[tested]) * log_n
    accepted = tested & (delta_chi2 > threshold)

    result = _take_spectra(m0, m1, labelled(accepted)).assign(
        line_search_accepted=labelled(accepted),
        line_search_tested=labelled(tested),
        line_search_start_freq_hz=labelled(start_freqs),
        line_search_delta_chi2=labelled(delta_chi2),
        line_search_threshold=labelled(threshold),
        line_search_noise_sigma=sigma,
    )
    result.line_search_accepted.attrs["description"] = (
        f"Line {new_metab_name!r} kept by the information criterion"
    )
    result.line_search_tested.attrs["description"] = (
        "Selected spectrum with successful M0 and M1 fits and an allowed residual peak"
    )
    result.line_search_start_freq_hz.attrs.update(
        unit="Hz", description="Largest allowed residual peak; start of the new line"
    )
    result.line_search_delta_chi2.attrs.update(
        unit="1",
        definition="(RSS_M0 - RSS_M1) / max(sigma**2, RSS_M1 / (2 N - k_M1))",
    )
    result.line_search_threshold.attrs.update(
        unit="1", definition="(k_M1 - k_M0) * ln(2 N); the line is kept above"
    )
    result.attrs.update(
        line_search_new_line=new_metab_name,
        line_search_range_hz=[low, high],
        line_search_window_hz=window_hz,
        line_search_n_samples=int(n_samples),
        line_search_method=(
            "forward step: propose one line at the largest residual peak in "
            "the search range and refit from the parent result (nested start)"
        ),
        line_search_criterion=("keep the line if delta_chi2 > (k_M1 - k_M0) * ln(2 N)"),
    )
    return result


__all__ = [
    "add_line_from_residual",
    "compare_metabolite_relevance",
    "eliminate_unnecessary_lines",
]
