"""Labelled post-fit diagnostics for spectroscopy.

The numerical fitter owns point estimation.  This module adds diagnostics that
need a supplied noise scale without changing any fitted parameter: the
reduced chi-squared and the conditional amplitude diagnostics.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _derive_frequency_bounds,
    _require_lorentz_lineshape,
)
from hypermri.analysis.fitting.models.spectroscopy.model_comparison import (
    conditional_fit_diagnostics,
)

NoiseInput = Real | xr.DataArray | xr.Dataset


def _fit_parameter_like(
    fit_params: xr.Dataset,
    name: str,
    template: xr.DataArray,
) -> xr.DataArray:
    """Return one labelled fit-parameter map aligned to a fit result."""
    if name not in fit_params:
        raise KeyError(f"fit_params is missing required variable '{name}'.")
    value = fit_params[name]
    extra = set(value.dims) - set(template.dims)
    for dim in list(extra):
        if value.sizes[dim] != 1:
            raise ValueError(
                f"fit_params['{name}'] varies along unsupported dimension '{dim}'."
            )
        value = value.isel({dim: 0}, drop=True)
    value, _ = xr.align(value, template, join="exact", exclude="metabolite")
    return value.broadcast_like(template).transpose(*template.dims)


def noise_sigma_for_fit(
    noise: NoiseInput,
    template: xr.DataArray,
    *,
    require_valid: bool = False,
) -> xr.DataArray:
    """Align scalar or labelled per-spectrum quadrature sigma to a fit.

    Parameters
    ----------
    noise
        Positive scalar, a DataArray containing one sigma per spectrum, or a
        Dataset with ``noise_sigma_quadrature``.  The Dataset form keeps the
        existing advanced noise-reference contract usable.
    template
        Fitted values with a ``metabolite`` dimension.
    require_valid
        Reject non-finite or non-positive values when True.  Advanced quality
        evaluation leaves this False so a missing/rejected reference can be
        represented by NaN and classified explicitly.
    """
    if "metabolite" not in template.dims:
        raise ValueError("template must contain a 'metabolite' dimension.")

    if isinstance(noise, xr.Dataset):
        if "noise_sigma_quadrature" not in noise:
            raise KeyError("Noise Dataset must contain 'noise_sigma_quadrature'.")
        sigma = noise["noise_sigma_quadrature"]
    elif isinstance(noise, xr.DataArray):
        sigma = noise
    elif isinstance(noise, Real) and not isinstance(noise, (bool, np.bool_)):
        sigma = xr.DataArray(float(noise))
    else:
        raise TypeError(
            "noise must be a real scalar, xarray.DataArray, or xarray.Dataset."
        )

    if np.iscomplexobj(sigma):
        raise ValueError("Noise sigma must be real-valued.")

    allowed = set(template.dims) - {"metabolite"}
    for dim in list(set(sigma.dims) - allowed):
        if sigma.sizes[dim] != 1:
            raise ValueError(
                "Noise sigma has unresolved dimensions; provide one scalar per "
                "spectrum. Time-dependent or channel covariance needs an explicit "
                "weighted likelihood."
            )
        sigma = sigma.isel({dim: 0}, drop=True)

    loop_template = template.isel(metabolite=0, drop=True)
    try:
        sigma, loop_template = xr.align(sigma, loop_template, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "Noise sigma coordinates must exactly match the fitted spectra."
        ) from exc
    sigma = sigma.broadcast_like(loop_template).transpose(*loop_template.dims)
    sigma = sigma.astype(float)

    if require_valid:
        values = np.asarray(sigma.values)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Noise sigma must contain only finite positive values.")

    sigma.name = "noise_sigma_quadrature"
    sigma.attrs.update(
        {
            "definition": (
                "Standard deviation of either the real or imaginary component "
                "of proper complex noise in the temporal FID samples used by "
                "the fit; E[abs(noise)**2] = 2*sigma**2."
            ),
            "unit": template.attrs.get("unit", "a.u."),
        }
    )
    return sigma


REDUCED_CHI_SQUARED_DEFINITION = (
    "fit_rss / noise_sigma_quadrature**2 / fit_residual_degrees_of_freedom; "
    "fit_rss = sum(abs(measured_FID - fitted_FID)**2) over the fitted complex "
    "samples, degrees of freedom = 2 * fitted complex samples - free real "
    "parameters. Expected value 1 for white noise of the given sigma and a "
    "correct model."
)


def calculate_reduced_chi_squared(
    fit_results: xr.Dataset, noise: NoiseInput
) -> xr.DataArray:
    """Return the reduced chi-squared of every fitted spectrum.

    ``chi2 / N_eff = fit_rss / sigma**2 / N_eff``. Each complex residual
    sample contributes two real residuals (real and imaginary part), each with
    noise variance ``sigma**2``, so ``chi2 = fit_rss / sigma**2`` and
    ``N_eff = 2 * n_fitted_samples - n_free_parameters`` real degrees of
    freedom, the value stored in ``fit_residual_degrees_of_freedom`` by
    :func:`~hypermri.analysis.fitting.spectroscopy.fit_data_pseudo_inv`. For
    white noise of the given sigma and a correct model the value scatters
    around 1 with standard deviation ``sqrt(2 / N_eff)``; values above 1 mean
    model error or underestimated noise. The residual is the temporal complex
    residual also when the fit minimised the spectral magnitude objective.
    The calculation is eager for NumPy data, deterministic and does not
    modify its inputs.

    Parameters
    ----------
    fit_results : xr.Dataset
        FID fit results with ``fit_rss`` (squared signal units),
        ``fit_residual_degrees_of_freedom`` and ``fit_amps`` (whose
        ``metabolite`` dimension identifies the spectra).
    noise : NoiseInput
        Real/imaginary quadrature sigma of the temporal FID samples used by
        the fit, in signal units; see :func:`noise_sigma_for_fit`. Attributes
        of a labelled sigma that start with ``noise_`` (for example the
        provenance of
        :func:`~hypermri.analysis.spectroscopy.noise_estimation.propagate_acquisition_tail_noise`)
        are copied to the result.

    Returns
    -------
    xr.DataArray
        ``fit_reduced_chi_squared`` (dimensionless, float64) with the loop
        dimensions of ``fit_rss`` in their order. NaN where the fit has no
        finite RSS, the sigma is not finite and positive, or
        ``N_eff <= 0``.

    Raises
    ------
    KeyError
        If a required result variable is missing.
    ValueError
        If the noise cannot be aligned to the fitted spectra.

    Examples
    --------
    >>> sigma = propagate_acquisition_tail_noise(
    ...     tail, point_weights=weights, n_spatial_points=n_points
    ... )  # doctest: +SKIP
    >>> chi2 = calculate_reduced_chi_squared(fit_results, sigma)  # doctest: +SKIP
    """
    required = {"fit_rss", "fit_residual_degrees_of_freedom", "fit_amps"}
    missing = required - set(fit_results)
    if missing:
        raise KeyError(
            f"calculate_reduced_chi_squared: fit_results is missing {sorted(missing)}; "
            "fit with fit_data_pseudo_inv to obtain them."
        )
    sigma = noise_sigma_for_fit(noise, fit_results.fit_amps)
    sigma = sigma.reset_coords(drop=True)
    rss = fit_results.fit_rss.reset_coords(drop=True)
    dof = fit_results.fit_residual_degrees_of_freedom.reset_coords(drop=True)
    valid = np.isfinite(sigma) & (sigma > 0) & (dof > 0)
    chi2 = (rss / sigma.where(valid) ** 2 / dof.where(valid)).astype(float)
    chi2 = chi2.transpose(*fit_results.fit_rss.dims)
    chi2 = chi2.assign_coords(
        {name: fit_results.fit_rss.coords[name] for name in fit_results.fit_rss.coords}
    )
    chi2.name = "fit_reduced_chi_squared"
    chi2.attrs = {
        "unit": "dimensionless",
        "definition": REDUCED_CHI_SQUARED_DEFINITION,
        "noise_sigma_convention": (
            "SD per temporal real/imaginary quadrature; E[abs(noise)**2] = 2*sigma**2"
        ),
        **{
            key: value for key, value in sigma.attrs.items() if key.startswith("noise_")
        },
    }
    chi2.attrs.setdefault("noise_source", "supplied")
    return chi2


def calculate_spectroscopy_fit_diagnostics(
    fit_results: xr.Dataset,
    fit_params: xr.Dataset,
    noise: NoiseInput,
) -> xr.Dataset:
    """Calculate conditional amplitude and basis diagnostics without fitting.

    The amplitude standard deviation treats fitted frequencies and T2* values
    as fixed.  It is not a full parameter confidence interval. The
    calculation is eager, deterministic and does not modify its inputs.

    Parameters
    ----------
    fit_results : xr.Dataset
        Lorentz FID fit results with ``fit_amps``, ``fit_freqs`` (Hz),
        ``fit_t2s`` (s) and ``fit_skipped``, each with a ``metabolite``
        dimension.
    fit_params : xr.Dataset
        Prepared fit parameters that produced ``fit_results``, with the search
        and bound maps and ``attrs['sampling_time_axis_s']`` in seconds.
    noise : NoiseInput
        Real/imaginary quadrature sigma of the temporal FID, see
        :func:`noise_sigma_for_fit`.

    Returns
    -------
    xr.Dataset
        ``basis_condition_number`` and ``basis_rank`` per spectrum and
        ``amplitude_std_quadrature_conditional``,
        ``amplitude_snr_conditional``, ``frequency_bound_hit`` and
        ``t2_bound_hit`` per spectrum and metabolite.

    Raises
    ------
    KeyError
        If required fit variables or parameter maps are missing.
    ValueError
        If ``fit_amps`` has no metabolite dimension, the time axis is not a
        finite 1-D array, the noise is invalid, or either dataset describes a
        ``lineshape='voigt'`` fit (the diagnostic basis is Lorentz only).

    Warns
    -----
    UserWarning
        If stored ``freq_range_lower_bound_hz`` or ``freq_range_upper_bound_hz``
        differ from ``initial_freqs_hz`` -/+ ``metabs_freq_ranges`` / 2. The
        bounds always follow start and full search width, so the derived
        bounds are used; ``fit_params`` itself is not modified.
    """
    _require_lorentz_lineshape(
        "calculate_spectroscopy_fit_diagnostics", fit_results, fit_params
    )
    required = {"fit_amps", "fit_freqs", "fit_t2s", "fit_skipped"}
    missing = required - set(fit_results)
    if missing:
        raise KeyError(f"fit_results is missing required variables: {sorted(missing)}")
    fit_params = _derive_frequency_bounds(
        fit_params, operation="calculate_spectroscopy_fit_diagnostics"
    )
    template = fit_results.fit_amps
    if "metabolite" not in template.dims:
        raise ValueError("fit_amps must contain a metabolite dimension.")
    parameters = [
        _fit_parameter_like(fit_params, name, template)
        for name in (
            "metabs_freq_ranges",
            "metabs_t2_ranges",
            "min_t2_s",
            "max_t2_s",
            "freq_range_lower_bound_hz",
            "freq_range_upper_bound_hz",
        )
    ]
    sigma = noise_sigma_for_fit(noise, template)
    time = np.asarray(fit_params.attrs.get("sampling_time_axis_s"), dtype=float)
    if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("fit_params requires a finite one-dimensional time axis.")
    outputs = xr.apply_ufunc(
        conditional_fit_diagnostics,
        template,
        fit_results.fit_freqs,
        fit_results.fit_t2s,
        fit_results.fit_skipped,
        *parameters,
        sigma,
        input_core_dims=[["metabolite"]] * 10 + [[]],
        output_core_dims=[
            [],
            [],
            ["metabolite"],
            ["metabolite"],
            ["metabolite"],
            ["metabolite"],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, np.int16, float, float, bool, bool],
        dask_gufunc_kwargs={
            "output_sizes": {"metabolite": template.sizes["metabolite"]}
        },
        kwargs={"sampling_time_axis_s": time},
    )
    names = (
        "basis_condition_number",
        "basis_rank",
        "amplitude_std_quadrature_conditional",
        "amplitude_snr_conditional",
        "frequency_bound_hit",
        "t2_bound_hit",
    )
    result = xr.Dataset(dict(zip(names, outputs, strict=True)))
    result.amplitude_std_quadrature_conditional.attrs.update(
        definition=(
            "sigma * sqrt(diag(pinv(X) @ pinv(X).conj().T)); conditional on "
            "fitted frequencies/T2*. Not a confidence interval."
        ),
        unit=fit_results.fit_amps.attrs.get("unit", "a.u."),
    )
    result.amplitude_snr_conditional.attrs["definition"] = (
        "abs(fit_amps) / amplitude_std_quadrature_conditional"
    )
    result.basis_rank.attrs["definition"] = "numerical matrix rank of active M1 basis"
    result.basis_condition_number.attrs["definition"] = (
        "2-norm condition number of active M1 basis"
    )
    result.attrs["postfit_diagnostics_version"] = "1"
    return result


__all__ = [
    "NoiseInput",
    "REDUCED_CHI_SQUARED_DEFINITION",
    "calculate_reduced_chi_squared",
    "calculate_spectroscopy_fit_diagnostics",
    "noise_sigma_for_fit",
]


_BOUND_STATUS_LABELS = {"fit_freq_status": "frequency", "fit_t2_status": "T2*"}


def summarize_fit_bound_flags(fit_results: xr.Dataset) -> xr.DataArray:
    """Describe, per spectrum, which fitted lines ended at a search bound.

    Reads the bound statuses ``fit_freq_status`` and ``fit_t2_status`` that
    :func:`~hypermri.analysis.fitting.spectroscopy.fit_data_pseudo_inv`
    stores and lists every line whose frequency or T2* is ``at_lower_bound``
    or ``at_upper_bound``. ``fixed``, ``not_fitted`` and ``estimated`` are
    not flags. The calculation is eager, deterministic and does not modify
    its input.

    Parameters
    ----------
    fit_results : xr.Dataset
        Fit results with at least one of ``fit_freq_status`` and
        ``fit_t2_status`` (strings with a ``metabolite`` dimension).

    Returns
    -------
    xr.DataArray
        ``fit_bound_flags``: one string per spectrum, with the non-metabolite
        dimensions and coordinates of the statuses in their order. Each flag
        reads ``"<line> <frequency|T2*> <status>"``; flags are joined with
        ``"; "`` in metabolite order, frequency before T2*. An empty string
        means no line is at a bound.

    Raises
    ------
    KeyError
        If neither status variable is present (results from before the
        statuses existed); refit to obtain them.

    Examples
    --------
    >>> flags = summarize_fit_bound_flags(fit_results)  # doctest: +SKIP
    >>> flags.isel(x=3, y=5).item()  # doctest: +SKIP
    'pyruvate frequency at_upper_bound'
    """
    names = [name for name in _BOUND_STATUS_LABELS if name in fit_results]
    if not names:
        raise KeyError(
            "summarize_fit_bound_flags: fit_results has neither fit_freq_status "
            "nor fit_t2_status; refit with fit_data_pseudo_inv to obtain them."
        )
    template = fit_results[names[0]].isel(metabolite=0, drop=True)
    metabolites = [str(m) for m in fit_results[names[0]].coords["metabolite"].values]
    statuses = {
        name: fit_results[name].transpose(*template.dims, "metabolite").values
        for name in names
    }
    flags = np.empty(template.shape, dtype=object)
    for index in np.ndindex(template.shape):
        entries = [
            f"{metabolite} {_BOUND_STATUS_LABELS[name]} {statuses[name][index][m]}"
            for m, metabolite in enumerate(metabolites)
            for name in names
            if str(statuses[name][index][m]).startswith("at_")
        ]
        flags[index] = "; ".join(entries)
    return xr.DataArray(
        flags.astype(str),
        dims=template.dims,
        coords=template.coords,
        name="fit_bound_flags",
        attrs={
            "definition": (
                "lines whose fitted frequency or T2* ended at a search bound, "
                "from fit_freq_status and fit_t2_status; empty when none"
            )
        },
    )


def order_fit_problem_voxels(fit_results: xr.Dataset) -> xr.Dataset:
    """List fitted spectra from most to least suspicious for review.

    Spectra with at least one bound flag (see
    :func:`summarize_fit_bound_flags`) come first, then the others; within
    each group the reduced chi-squared ``fit_reduced_chi_squared`` decides,
    highest first, and spectra without a finite value follow in index order.
    Spectra without a finite ``fit_rss`` and without flags (not fitted, for
    example outside the mask) are left out. Ties keep C index order, so the
    order is deterministic. Results without bound statuses are ordered by
    reduced chi-squared alone. The input is not modified.

    Parameters
    ----------
    fit_results : xr.Dataset
        Fit results with ``fit_rss`` and ``fit_freqs`` (whose non-metabolite
        dimensions are the spectrum dimensions), and optionally
        ``fit_reduced_chi_squared`` (dimensionless), ``fit_freq_status`` and
        ``fit_t2_status``.

    Returns
    -------
    xr.Dataset
        One entry per listed spectrum along the dimension ``problem_voxel``,
        most suspicious first, with one int64 index variable per spectrum
        dimension (named like the dimension, positional index into
        ``fit_results``), ``fit_reduced_chi_squared`` (float64, NaN when not
        available) and ``fit_bound_flags`` (str). Use
        ``{dim: int(order[dim][k]) for dim in order.attrs["spectrum_dims"]}``
        as an ``isel`` selector of entry ``k``.

    Raises
    ------
    KeyError
        If ``fit_rss`` or ``fit_freqs`` is missing.

    Examples
    --------
    >>> order = order_fit_problem_voxels(fit_results)  # doctest: +SKIP
    >>> first = {d: int(order[d][0]) for d in order.attrs["spectrum_dims"]}  # doctest: +SKIP
    >>> fit_results.fit_amps.isel(first)  # doctest: +SKIP
    """
    missing = {"fit_rss", "fit_freqs"} - set(fit_results)
    if missing:
        raise KeyError(
            f"order_fit_problem_voxels: fit_results is missing {sorted(missing)}; "
            "fit with fit_data_pseudo_inv to obtain them."
        )
    dims = tuple(d for d in fit_results.fit_freqs.dims if d != "metabolite")
    rss = fit_results.fit_rss.transpose(*dims).values
    if "fit_reduced_chi_squared" in fit_results:
        chi2 = fit_results.fit_reduced_chi_squared.transpose(*dims).values
    else:
        chi2 = np.full(rss.shape, np.nan)
    chi2 = np.asarray(chi2, dtype=np.float64)
    if any(name in fit_results for name in _BOUND_STATUS_LABELS):
        flags = summarize_fit_bound_flags(fit_results).transpose(*dims).values
    else:
        flags = np.full(rss.shape, "", dtype=str)

    flat_flags = flags.ravel()
    flat_chi2 = chi2.ravel()
    flagged = flat_flags != ""
    listed = flagged | np.isfinite(np.asarray(rss, dtype=np.float64).ravel())
    finite = np.isfinite(flat_chi2)
    # lexsort sorts by the last key first and is stable, so ties keep C order.
    order = np.lexsort((np.where(finite, -flat_chi2, 0.0), ~finite, ~flagged))
    order = order[listed[order]]
    positions = np.unravel_index(order, rss.shape)

    result = xr.Dataset(
        {
            dim: ("problem_voxel", np.asarray(position, dtype=np.int64))
            for dim, position in zip(dims, positions, strict=True)
        }
    )
    result["fit_reduced_chi_squared"] = ("problem_voxel", flat_chi2[order])
    result["fit_reduced_chi_squared"].attrs["unit"] = "dimensionless"
    result["fit_bound_flags"] = ("problem_voxel", flat_flags[order].astype(str))
    result.attrs["spectrum_dims"] = list(dims)
    result.attrs["ordering"] = (
        "spectra with bound flags first, then descending fit_reduced_chi_squared, "
        "spectra without a finite value last, ties in C index order; spectra "
        "without a finite fit_rss and without flags are omitted"
    )
    return result
