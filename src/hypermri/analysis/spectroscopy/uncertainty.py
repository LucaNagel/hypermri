"""Parametric-bootstrap uncertainty for fitted spectroscopy parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    UNCERTAINTY_BOOTSTRAP_SAMPLES_DEFAULT,
    UNCERTAINTY_CONFIDENCE_LEVEL_DEFAULT,
    UNCERTAINTY_SEED_DEFAULT,
    _derive_frequency_bounds,
    _prepare_global_fit_options,
    _require_same_lineshape,
)
from hypermri.analysis.fitting.spectroscopy import (
    fit_data_pseudo_inv,
    reconstruct_fitted_signal,
)

from .fit_diagnostics import NoiseInput, noise_sigma_for_fit
from .fitting import _prepare_fit_params


def _uncertainty_options(fit_params: xr.Dataset) -> tuple[int, float, int]:
    """Read and revalidate the effective bootstrap settings."""
    options = _prepare_global_fit_options(fit_params.attrs)
    return (
        int(
            options.get(
                "uncertainty_bootstrap_samples",
                UNCERTAINTY_BOOTSTRAP_SAMPLES_DEFAULT,
            )
        ),
        float(
            options.get(
                "uncertainty_confidence_level",
                UNCERTAINTY_CONFIDENCE_LEVEL_DEFAULT,
            )
        ),
        int(options.get("uncertainty_seed", UNCERTAINTY_SEED_DEFAULT)),
    )


def _uncertainty_selection(
    data: xr.DataArray,
    fit_params: xr.Dataset,
    selection: xr.DataArray | str | None,
) -> xr.DataArray:
    """Return a boolean mask over the original non-spectral dimensions."""
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(
            f"Input data is missing the spectral dimension '{spectral_dim}'."
        )
    template = data.isel({spectral_dim: 0}, drop=True)
    if selection is None:
        if template.size != 1:
            raise ValueError(
                "Multi-spectrum uncertainty requires a labelled selection or "
                "selection='all'."
            )
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
        raise ValueError("Uncertainty selection does not contain any spectra.")
    return result


def _local_dataset(dataset: xr.Dataset, indexers: dict[str, int]) -> xr.Dataset:
    """Select one spectrum from variables that carry its loop dimensions."""
    applicable = {dim: index for dim, index in indexers.items() if dim in dataset.dims}
    return dataset.isel(applicable, drop=True)


def _local_dataarray(
    data: xr.DataArray,
    indexers: dict[str, int],
) -> xr.DataArray:
    """Select one spectrum from a labelled array."""
    applicable = {dim: index for dim, index in indexers.items() if dim in data.dims}
    return data.isel(applicable, drop=True)


def _filled_like(template: xr.DataArray, value, dtype) -> xr.DataArray:
    """Allocate an eager output while retaining all labelled coordinates."""
    return template.copy(data=np.full(template.shape, value, dtype=dtype))


def _bootstrap_input(
    input_template: xr.DataArray,
    fitted_temporal: xr.DataArray,
    *,
    sigma: float,
    samples: int,
    seed: int,
    flat_spectrum_index: int,
    spectral_dim: str,
) -> xr.DataArray:
    """Generate temporal complex bootstrap FIDs for one selected spectrum."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, flat_spectrum_index]))
    fitted = np.asarray(fitted_temporal.values, dtype=np.complex128)
    noise = sigma * (
        rng.standard_normal((samples, fitted.size))
        + 1j * rng.standard_normal((samples, fitted.size))
    )
    coords = {
        name: coord
        for name, coord in input_template.coords.items()
        if set(coord.dims).issubset({spectral_dim})
    }
    result = xr.DataArray(
        fitted[np.newaxis, :] + noise,
        dims=("bootstrap", spectral_dim),
        coords={"bootstrap": np.arange(samples), **coords},
        attrs=dict(input_template.attrs),
        name="bootstrap_fid",
    )
    result.attrs["signal_domain"] = "temporal"
    return result


def estimate_fit_uncertainty(
    data: xr.DataArray,
    fit_params: xr.Dataset | Mapping[str, object],
    fit_results: xr.Dataset,
    *,
    noise: NoiseInput | None = None,
    selection: xr.DataArray | str | None = None,
    metabolites: Sequence[str] | None = None,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
) -> xr.Dataset:
    """Estimate percentile intervals by fully refitting noisy M1 signals.

    The number of draws ``B`` is read from
    ``fit_params.attrs['uncertainty_bootstrap_samples']``, with a default of
    999 when the option is absent.  Supplying the option does not affect the
    ordinary fit; work starts only when this function is called.

    One spectrum is selected implicitly only when the input contains exactly
    one spectrum.  Multi-spectrum data require a boolean labelled selection or
    an explicit ``selection='all'``.  Every draw fits the complete active model,
    even when intervals are returned for only selected metabolites.

    Voigt fits are supported when ``fit_params`` and ``fit_results`` share the
    lineshape: every draw refits the shared Gaussian FWHM, so the amplitude
    and frequency intervals include its uncertainty. ``t2_ci_*`` then refer
    to the Lorentz part of the linewidth; no interval is returned for the
    Gaussian FWHM itself. Draws are deterministic for a fixed
    ``uncertainty_seed``.

    Parameters
    ----------
    data : xr.DataArray
        Complex temporal FIDs with exactly the samples used by the fit.
    fit_params : xr.Dataset or Mapping
        Fit parameters of ``fit_results``, or a mapping for ``def_fit_params``,
        which rejects unknown keys with ``ValueError``.
    fit_results : xr.Dataset
        Ordinary M1 fit results used as the bootstrap mean.
    noise : NoiseInput, optional
        Real/imaginary quadrature sigma of the temporal FID. Defaults to
        ``fit_results['noise_sigma_quadrature']`` when present.
    selection : xr.DataArray or str, optional
        Boolean labelled selection of spectra, or ``'all'``.
    metabolites : sequence of str, optional
        Metabolites whose intervals are returned; all by default.
    n_jobs, batch_size, parallel_backend
        Execution options forwarded unchanged to ``fit_data_pseudo_inv``.

    Returns
    -------
    xr.Dataset
        Per spectrum and requested metabolite: ``amplitude_abs_ci_low/high``
        (amplitude unit), ``frequency_ci_low/high`` (Hz), ``t2_ci_low/high``
        (s), bound-hit fractions and ``uncertainty_assessed/available/valid``;
        per spectrum: ``bootstrap_requested``, ``bootstrap_successful``,
        ``bootstrap_failure_fraction`` and ``bootstrap_status``.

    Raises
    ------
    TypeError
        If ``data`` or ``fit_results`` has the wrong type.
    ValueError
        If ``detection_target`` is set, the lineshapes of ``fit_params`` and
        ``fit_results`` differ, no valid noise is available, the selection or
        metabolites are invalid, or the samples differ from the fit.

    Warns
    -----
    UserWarning
        If stored ``freq_range_lower_bound_hz`` or ``freq_range_upper_bound_hz``
        differ from ``initial_freqs_hz`` -/+ ``metabs_freq_ranges`` / 2. The
        bounds always follow start and full search width, so the derived
        bounds are used; ``fit_params`` itself is not modified.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray.")
    if not isinstance(fit_results, xr.Dataset):
        raise TypeError("fit_results must be an xarray.Dataset.")
    fit_params = _derive_frequency_bounds(
        _prepare_fit_params(data, fit_params),
        operation="estimate_fit_uncertainty",
    )
    if fit_params.attrs.get("detection_target") is not None:
        raise ValueError(
            "Parameter uncertainty requires an ordinary M1 fit without "
            "detection_target settings."
        )
    _require_same_lineshape("estimate_fit_uncertainty", fit_params, fit_results)

    bootstrap_samples, confidence_level, seed = _uncertainty_options(fit_params)
    selected = _uncertainty_selection(data, fit_params, selection)
    noise_input = (
        fit_results["noise_sigma_quadrature"]
        if noise is None and "noise_sigma_quadrature" in fit_results
        else noise
    )
    if noise_input is None:
        raise ValueError(
            "Parameter uncertainty requires noise or a fit containing "
            "'noise_sigma_quadrature'."
        )
    sigma_map = noise_sigma_for_fit(
        noise_input, fit_results.fit_amps, require_valid=True
    )

    available_metabolites = [
        str(value) for value in fit_results.metabolite.values.tolist()
    ]
    requested_metabolites = (
        available_metabolites
        if metabolites is None
        else [str(value) for value in metabolites]
    )
    missing = [
        value for value in requested_metabolites if value not in available_metabolites
    ]
    if missing:
        raise ValueError(f"Unknown metabolites requested: {missing}.")
    parameter_template = fit_results.fit_amps.sel(metabolite=requested_metabolites).real
    loop_template = fit_results.fit_rss

    outputs = {
        name: _filled_like(parameter_template, np.nan, float)
        for name in (
            "amplitude_abs_ci_low",
            "amplitude_abs_ci_high",
            "frequency_ci_low",
            "frequency_ci_high",
            "t2_ci_low",
            "t2_ci_high",
            "frequency_bound_hit_fraction",
            "t2_bound_hit_fraction",
        )
    }
    outputs.update(
        {
            "uncertainty_assessed": _filled_like(parameter_template, False, bool),
            "uncertainty_available": _filled_like(parameter_template, False, bool),
            "uncertainty_valid": _filled_like(parameter_template, False, bool),
            "bootstrap_requested": _filled_like(loop_template, 0, np.int64),
            "bootstrap_successful": _filled_like(loop_template, 0, np.int64),
            "bootstrap_failure_fraction": _filled_like(loop_template, np.nan, float),
            "bootstrap_status": _filled_like(loop_template, "not_selected", "U32"),
        }
    )

    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    time = np.asarray(fit_params.attrs.get("sampling_time_axis_s"), dtype=float)
    if data.sizes[spectral_dim] != time.size:
        raise ValueError(
            "data must contain exactly the temporal samples used by the fit "
            "before parameter uncertainty is estimated."
        )
    selected_flat = np.flatnonzero(np.asarray(selected.values).reshape(-1))
    tail = (1.0 - confidence_level) / 2.0
    successful_statuses = {"success", "zero_signal"}

    for flat_index in selected_flat:
        position = np.unravel_index(flat_index, selected.shape)
        indexers = dict(zip(selected.dims, position, strict=True))
        parameter_index = position + (slice(None),)
        outputs["uncertainty_assessed"].data[parameter_index] = True
        outputs["bootstrap_requested"].data[position] = bootstrap_samples

        observed_status = str(_local_dataarray(fit_results.fit_status, indexers).item())
        if observed_status not in successful_statuses:
            outputs["bootstrap_status"].data[position] = "invalid_observed_fit"
            continue

        local_input = _local_dataarray(data, indexers)
        local_fit = _local_dataset(fit_results, indexers)
        local_params = _local_dataset(fit_params, indexers)
        temporal_template = local_input.copy(deep=False)
        temporal_template.attrs = dict(local_input.attrs)
        temporal_template.attrs["signal_domain"] = "temporal"
        fitted_temporal = reconstruct_fitted_signal(local_fit, temporal_template)
        local_sigma = float(_local_dataarray(sigma_map, indexers).item())
        bootstrap_data = _bootstrap_input(
            local_input,
            fitted_temporal,
            sigma=local_sigma,
            samples=bootstrap_samples,
            seed=seed,
            flat_spectrum_index=int(flat_index),
            spectral_dim=spectral_dim,
        )
        local_params = local_params.assign_attrs(
            {**local_params.attrs, "signal_domain": "temporal"}
        )
        bootstrap_fit = fit_data_pseudo_inv(
            bootstrap_data,
            local_params,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
            show_tqdm=False,
        )
        statuses = np.asarray(bootstrap_fit.fit_status.values).astype(str)
        success = np.isin(statuses, list(successful_statuses))
        successful = int(np.count_nonzero(success))
        outputs["bootstrap_successful"].data[position] = successful
        outputs["bootstrap_failure_fraction"].data[position] = (
            1.0 - successful / bootstrap_samples
        )
        if successful != bootstrap_samples:
            outputs["bootstrap_status"].data[position] = "bootstrap_failed"
            continue

        target_fit = bootstrap_fit.sel(metabolite=requested_metabolites)
        draws = {
            "amplitude_abs": np.abs(np.asarray(target_fit.fit_amps.values)),
            "frequency": np.asarray(target_fit.fit_freqs.values, dtype=float),
            "t2": np.asarray(target_fit.fit_t2s.values, dtype=float),
        }
        for parameter in ("amplitude_abs", "frequency", "t2"):
            low, high = np.quantile(draws[parameter], [tail, 1.0 - tail], axis=0)
            outputs[f"{parameter}_ci_low"].data[parameter_index] = low
            outputs[f"{parameter}_ci_high"].data[parameter_index] = high

        local_targets = local_params.sel(metabolite=requested_metabolites)
        frequency_active = (
            np.asarray(local_targets.metabs_freq_ranges.values, dtype=float) > 0
        )
        frequency_bound = frequency_active[np.newaxis, :] & (
            np.isclose(
                draws["frequency"],
                np.asarray(local_targets.freq_range_lower_bound_hz.values)[
                    np.newaxis, :
                ],
                rtol=1e-7,
                atol=1e-12,
            )
            | np.isclose(
                draws["frequency"],
                np.asarray(local_targets.freq_range_upper_bound_hz.values)[
                    np.newaxis, :
                ],
                rtol=1e-7,
                atol=1e-12,
            )
        )
        t2_active = np.asarray(local_targets.metabs_t2_ranges.values, dtype=float) > 0
        t2_bound = t2_active[np.newaxis, :] & (
            np.isclose(
                draws["t2"],
                np.asarray(local_targets.min_t2_s.values)[np.newaxis, :],
                rtol=1e-7,
                atol=1e-12,
            )
            | np.isclose(
                draws["t2"],
                np.asarray(local_targets.max_t2_s.values)[np.newaxis, :],
                rtol=1e-7,
                atol=1e-12,
            )
        )
        outputs["frequency_bound_hit_fraction"].data[parameter_index] = (
            frequency_bound.mean(axis=0)
        )
        outputs["t2_bound_hit_fraction"].data[parameter_index] = t2_bound.mean(axis=0)
        outputs["uncertainty_available"].data[parameter_index] = True
        outputs["uncertainty_valid"].data[parameter_index] = True
        outputs["bootstrap_status"].data[position] = "success"

    result = xr.Dataset(outputs)
    for prefix, unit in (
        ("amplitude_abs", fit_results.fit_amps.attrs.get("unit", "a.u.")),
        ("frequency", "Hz"),
        ("t2", "s"),
    ):
        for suffix in ("low", "high"):
            result[f"{prefix}_ci_{suffix}"].attrs["unit"] = unit
    result.attrs.update(
        uncertainty_method="M1 parametric percentile bootstrap with full refit",
        uncertainty_scope=(
            "Conditional on the fitted model, supplied per-quadrature noise and "
            "unchanged fit settings; noise-estimation uncertainty is not propagated."
        ),
        uncertainty_bootstrap_samples=bootstrap_samples,
        uncertainty_confidence_level=confidence_level,
        uncertainty_seed=seed,
        uncertainty_rng=(
            "PCG64(SeedSequence([uncertainty_seed, original_flat_spectrum_index]))"
        ),
        bootstrap_failure_policy="all requested draws must succeed",
    )
    return result


__all__ = ["estimate_fit_uncertainty"]
