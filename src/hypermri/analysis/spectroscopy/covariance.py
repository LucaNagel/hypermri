"""Local joint covariance for complex Lorentzian spectroscopy fits.

Notes
-----
The full real/imaginary model Jacobian propagates supplied white-noise sigma
through complex amplitudes, frequencies and T2*. This deterministic post-fit
operation uses float64/complex128, performs no refitting and does not estimate
noise, calibrate detection or establish confidence-interval coverage.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    DEFAULT_MIN_PEAK_SEPARATION_HZ,
    MAGNITUDE_LORENTZ_FWHM_FACTOR,
    _derive_frequency_bounds,
    _require_lorentz_lineshape,
    fid_basefunc,
)

from .fit_diagnostics import NoiseInput, noise_sigma_for_fit

_PARAMETER_KINDS = ("amplitude_real", "amplitude_imag", "frequency", "t2")
_MAP_NAMES = (
    "metabs_freq_ranges",
    "metabs_t2_ranges",
    "min_t2_s",
    "max_t2_s",
    "freq_range_lower_bound_hz",
    "freq_range_upper_bound_hz",
)
# Chi-square(2) 95% quantile: a local ellipse diagnostic, not a detection test.
_AMPLITUDE_ELLIPSE_RADIUS_SQUARED = -2.0 * np.log(0.05)
_COVARIANCE_STATUSES = (
    "success",
    "no_active_metabolites",
    "invalid_fit",
    "invalid_parameters",
    "parameter_bound",
    "rank_deficient",
    "ill_conditioned",
    "numerical_failure",
)
_AMPLITUDE_STATUSES = (*_COVARIANCE_STATUSES, "near_zero", "skipped")


def _joint_jacobian(amplitudes, frequencies, t2s, time_s):
    """Stack real and imaginary derivatives, grouped by parameter kind."""
    basis = fid_basefunc(frequencies, t2s, time_s).T
    signal_components = basis * amplitudes
    complex_jacobian = np.concatenate(
        (
            basis,
            1j * basis,
            2j * np.pi * time_s[:, None] * signal_components,
            time_s[:, None] * signal_components / t2s**2,
        ),
        axis=1,
    )
    return np.concatenate((complex_jacobian.real, complex_jacobian.imag), axis=0)


def _at_bound(values, lower, upper):
    """Identify the fitter's absolute boundaries at floating-point tolerance."""
    return np.isclose(values, lower, rtol=1e-7, atol=1e-12) | np.isclose(
        values, upper, rtol=1e-7, atol=1e-12
    )


def _spectrum_covariance(
    sigma,
    amplitudes,
    frequencies,
    t2s,
    skipped,
    frequency_widths,
    t2_widths,
    min_t2s,
    max_t2s,
    frequency_lower,
    frequency_upper,
    fit_status,
    *,
    time_s,
    min_peak_separation_hz,
    max_condition_number,
):
    """Propagate one spectrum's noise through its active unconstrained tangent."""
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    t2s = np.asarray(t2s, dtype=np.float64)
    active = ~np.asarray(skipped, dtype=bool)
    count = active.size
    frequency_free = (
        active & (frequency_widths > 0) & (frequency_upper > frequency_lower)
    )
    t2_free = active & (t2_widths > 0) & (max_t2s > min_t2s)
    free = np.concatenate((active, active, frequency_free, t2_free))
    n_free = int(free.sum())
    covariance = np.full((4 * count, 4 * count), np.nan)
    amplitude_std = np.full(count, np.nan)
    amplitude_status = np.full(count, "skipped", dtype="U32")
    rank, condition = 0, np.nan

    def result(status):
        if status != "success":
            amplitude_status[active] = status
        return (
            covariance,
            amplitude_std,
            np.array(
                [_AMPLITUDE_STATUSES.index(value) for value in amplitude_status],
                dtype=np.int16,
            ),
            free,
            _COVARIANCE_STATUSES.index(status),
            rank,
            condition,
            n_free,
        )

    if not np.any(active):
        return result("no_active_metabolites")
    if str(fit_status) not in {"success", "zero_signal"}:
        return result("invalid_fit")
    parameters = np.stack(
        (
            amplitudes.real,
            amplitudes.imag,
            frequencies,
            t2s,
            frequency_widths,
            t2_widths,
            min_t2s,
            max_t2s,
            frequency_lower,
            frequency_upper,
        )
    )[:, active]
    if (
        not np.isfinite(parameters).all()
        or np.any(t2s[active] <= 0)
        or np.any(min_t2s[active] <= 0)
        or np.any(t2s[active] < min_t2s[active])
        or np.any(t2s[active] > max_t2s[active])
        or np.any(frequencies[active] < frequency_lower[active])
        or np.any(frequencies[active] > frequency_upper[active])
        or np.any(frequency_widths[active] < 0)
        or np.any(t2_widths[active] < 0)
    ):
        return result("invalid_parameters")
    if np.any(
        frequency_free & _at_bound(frequencies, frequency_lower, frequency_upper)
    ) or np.any(t2_free & _at_bound(t2s, min_t2s, max_t2s)):
        return result("parameter_bound")
    for left in np.flatnonzero(active):
        for right in np.flatnonzero(active & (np.arange(count) > left)):
            distance = abs(frequencies[left] - frequencies[right])
            if distance < min_peak_separation_hz:
                return result("invalid_parameters")
            if (frequency_free[left] or frequency_free[right]) and np.isclose(
                distance, min_peak_separation_hz, rtol=1e-7, atol=1e-12
            ):
                return result("parameter_bound")

    # Build only active basis functions; excluded components may contain NaNs.
    active_columns = np.flatnonzero(np.tile(active, 4))
    active_free = free[active_columns]
    try:
        jacobian = _joint_jacobian(
            amplitudes[active], frequencies[active], t2s[active], time_s
        )[:, active_free]
        scales = np.linalg.norm(jacobian, axis=0)
        if not np.isfinite(jacobian).all() or not np.isfinite(scales).all():
            return result("numerical_failure")
        scaled = jacobian / np.where(scales > 0, scales, 1.0)
        _, singular, vh = np.linalg.svd(scaled, full_matrices=False)
        tolerance = np.finfo(float).eps * max(scaled.shape) * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        if rank < n_free:
            condition = np.inf
            return result("rank_deficient")
        condition = float(singular[0] / singular[-1])
        if condition > max_condition_number:
            return result("ill_conditioned")
        # Cov = sigma^2 V diag(1/s^2) V.T, with column scaling undone.
        factor = (vh.T / singular) / scales[:, None] * float(sigma)
        free_covariance = factor @ factor.T
        if not np.isfinite(free_covariance).all():
            return result("numerical_failure")
        covariance[np.ix_(active_columns, active_columns)] = 0.0
        free_columns = np.flatnonzero(free)
        covariance[np.ix_(free_columns, free_columns)] = free_covariance
        for metabolite in np.flatnonzero(active):
            indices = [metabolite, count + metabolite]
            amplitude = amplitudes[metabolite]
            cartesian = np.array([amplitude.real, amplitude.imag])
            marginal = covariance[np.ix_(indices, indices)]
            ellipse_distance = cartesian @ np.linalg.solve(marginal, cartesian)
            if ellipse_distance <= _AMPLITUDE_ELLIPSE_RADIUS_SQUARED:
                amplitude_status[metabolite] = "near_zero"
                continue
            gradient = cartesian / abs(amplitude)
            amplitude_std[metabolite] = np.sqrt(gradient @ marginal @ gradient)
            amplitude_status[metabolite] = "success"
    except np.linalg.LinAlgError:
        covariance[:] = np.nan
        amplitude_std[:] = np.nan
        return result("numerical_failure")
    return result("success")


def _status_labels(codes, *, labels):
    """Decode fixed-size integer gufunc statuses without NumPy U1 truncation."""
    return np.asarray(labels, dtype="U32")[codes]


def _aligned_map(value, template, name):
    """Broadcast a map by names, rejecting missing or reordered coordinates."""
    if set(value.dims) - set(template.dims):
        raise ValueError(
            f"{name} has dimensions absent from the fitted parameter maps."
        )
    try:
        value, _ = xr.align(value, template, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(f"{name} coordinates must exactly match fit_results.") from exc
    return value.broadcast_like(template).transpose(*template.dims)


def _validate_fit_contract(fit_results, fit_params):
    """Require the unregularized Lorentz complex-FID model and matching times."""
    for name, dataset in (("fit_results", fit_results), ("fit_params", fit_params)):
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"{name} must be an xarray.Dataset.")
        options = dataset.attrs
        if options.get("model_name") != "FID":
            raise ValueError(f"{name} must describe the FID spectroscopy model.")
        if (
            options.get("use_sse_mag", False)
            or options.get("fit_objective_name", "temporal_complex_rss")
            != "temporal_complex_rss"
            or options.get("amp_solver", "pinv") not in {"pinv", "normal_eq"}
            or options.get("amp_ridge_lambda", 0) != 0
            or options.get("snr_threshold") not in (None, 0)
            or options.get("detection_target") is not None
        ):
            raise ValueError(
                f"{name}: local covariance requires an ordinary complex-RSS FID "
                "fit with unregularized amplitudes and no amplitude filtering or "
                "detection_target."
            )
        if options.get("t2_coupling"):
            raise ValueError(
                f"{name}: local covariance treats every T2* as a free parameter "
                "and does not support t2_coupling; use the bootstrap uncertainty "
                "instead."
            )
        if options.get("phase_band_tolerance_deg"):
            raise ValueError(
                f"{name}: local covariance treats every complex amplitude as free "
                "and does not support phase_band_tolerance_deg; use the bootstrap "
                "uncertainty instead."
            )
    _require_lorentz_lineshape("Local covariance estimation", fit_results, fit_params)
    time_s = np.asarray(fit_results.attrs.get("sampling_time_axis_s"), dtype=float)
    parameter_time = np.asarray(
        fit_params.attrs.get("sampling_time_axis_s"), dtype=float
    )
    if (
        time_s.ndim != 1
        or not time_s.size
        or not np.isfinite(time_s).all()
        or np.any(time_s < 0)
        or np.any(np.diff(time_s) <= 0)
        or parameter_time.shape != time_s.shape
        or not np.array_equal(parameter_time, time_s)
    ):
        raise ValueError(
            "sampling_time_axis_s must be identical in fit_results and fit_params, "
            "finite, non-negative and strictly increasing in seconds."
        )
    return time_s


def estimate_fit_covariance(
    fit_results: xr.Dataset,
    fit_params: xr.Dataset,
    *,
    noise: NoiseInput | None = None,
    max_condition_number: float = 1e8,
) -> xr.Dataset:
    """Estimate joint local spectroscopy errors without refitting spectra.

    Propagate white complex Gaussian noise through the full model Jacobian,
    including correlations of complex amplitudes, frequencies and T2*.
    This is a local Gauss-Newton approximation, not a calibrated confidence
    interval or detection test. Input fits and their point estimates are unchanged.

    Parameters
    ----------
    fit_results
        Ordinary unregularized FID results from ``fit_spectrum`` or
        ``fit_data_pseudo_inv``. Required variables are ``fit_amps`` (complex
        FID amplitudes at model time zero), ``fit_freqs`` (Hz), ``fit_t2s`` (s),
        ``fit_skipped`` (boolean), and per-spectrum ``fit_status``. Parameter
        arrays have a ``metabolite`` axis and any labelled spectrum axes, in
        any dimension order. Metadata must identify ``model_name='FID'`` and
        the actual ``sampling_time_axis_s`` used by the fit.
    fit_params
        The unchanged prepared Dataset used for this fit, including absolute
        bounds, search widths and ``skip_fitting``. Shared maps may broadcast
        over spectra; shared coordinates must match exactly, including order.
        Zero search widths or equal bounds fix parameters; their variance is
        zero by assumption, not a measured precision. Skipped components have
        NaN covariance rows/columns and NaN standard errors.
    noise
        Positive finite sigma per real/imaginary quadrature of the fitted
        temporal samples, with ``E[abs(noise)**2] = 2*sigma**2``. Accepts a
        scalar, labelled map over spectra, or Dataset containing
        ``noise_sigma_quadrature``. By default use that variable from the fit.
        Reference and signal must have matching scaling, reconstruction and
        filtering. Time/channel covariance is not used; a Dataset declaring
        ``noise_likelihood_compatible=False`` is rejected. Noise-estimation
        uncertainty is not propagated and residual variance never rescales sigma.
    max_condition_number
        Finite limit greater than one on the 2-norm condition of the
        column-normalized free Jacobian. Above this limit (default 1e8),
        suppress all covariance estimates. This is a numerical safeguard,
        not a calibrated threshold for scientific identifiability.

    Returns
    -------
    xr.Dataset
        ``parameter_covariance`` and ``parameter_correlation`` have dimensions
        ``(*spectrum_dims, fit_parameter, fit_parameter_other)``. Both parameter
        axes group all metabolites by ``amplitude_real``, ``amplitude_imag``,
        ``frequency``, ``t2``; companion kind, metabolite and unit coordinates
        describe each entry. Covariance units are the product of the two units.
        Spectrum dimensions follow ``fit_amps`` with ``metabolite`` removed.

        ``parameter_std`` and ``parameter_free`` use the first parameter axis.
        ``amplitude_abs_std``, ``frequency_std``, ``t2_std``, ``fwhm_std`` and
        ``magnitude_fwhm_std`` have ``(*spectrum_dims, metabolite)`` dimensions
        and units of fitted amplitude, Hz, s, Hz and Hz. ``fwhm_std`` belongs
        to the absorption FWHM ``1/(pi*T2*)``, ``magnitude_fwhm_std`` to the
        magnitude FWHM ``sqrt(3)/(pi*T2*)`` and is ``sqrt(3) * fwhm_std``. Correlations involving
        fixed/skipped parameters are NaN. ``amplitude_abs_status='near_zero'``
        suppresses the magnitude standard error when the local 95% Cartesian
        amplitude ellipse contains zero; this diagnostic is not a peak test.

        ``covariance_status`` reports ``success``, ``no_active_metabolites``,
        ``invalid_fit``, ``invalid_parameters``, ``parameter_bound``,
        ``rank_deficient``, ``ill_conditioned`` or ``numerical_failure`` per
        spectrum. Only success has covariance values. Rank, scaled condition,
        free-parameter count and real residual degrees of freedom accompany it.
        ``success`` does not guarantee that a nonlinear local approximation is
        adequate. No bootstrap-validity or Q2-classification flags are set.

    Raises
    ------
    TypeError
        If inputs are not Datasets or fitted masks are not boolean.
    KeyError
        If required parameter maps or result variables are missing.
    ValueError
        If coordinates, times, noise or numerical options are invalid, if
        skip masks disagree, or the objective/model is unsupported, including
        ``lineshape='voigt'`` fits (the Jacobian has no Gaussian FWHM) and
        fits with ``t2_coupling`` (the Jacobian treats every T2* as free)
        and fits with ``phase_band_tolerance_deg`` (it treats every complex
        amplitude as free).

    Warns
    -----
    UserWarning
        If stored ``freq_range_lower_bound_hz`` or ``freq_range_upper_bound_hz``
        differ from ``initial_freqs_hz`` -/+ ``metabs_freq_ranges`` / 2. The
        bounds always follow start and full search width, so the derived
        bounds are used; ``fit_params`` itself is not modified.

    Notes
    -----
    The real Jacobian has 2*N rows for N complex observations. Scaled SVD
    computes ``sigma**2 * inv(J.T @ J)`` on the full-rank free subspace,
    without forming the normal equations. Rank deficiency and active absolute
    or peak-separation bounds invalidate the unconstrained approximation.
    Model mismatch, uncertain preprocessing/alignment, spatial cross-spectrum
    covariance and post-selection effects are outside this approximation.

    Calculation/storage use float64 and complex128 irrespective of input
    precision. The operation is deterministic, has no cache or input mutation,
    and performs no fitting, plotting or raw-data reads. Metadata, masks and
    sigma are validated eagerly; Dask-backed fit calculations remain lazy.
    Each Dask core ``metabolite`` axis must be a single chunk. Memory per
    spectrum grows quadratically with the number of metabolites.

    Examples
    --------
    Reuse a fit containing supplied-noise diagnostics::

        from hypermri.analysis.spectroscopy import estimate_fit_covariance
        errors = estimate_fit_covariance(fit_results, fit_params)
        amplitude_error = errors.amplitude_abs_std
        frequency_error_hz = errors.frequency_std
        t2_error_s = errors.t2_std
        usable = errors.covariance_status == "success"

    The returned errors are one-standard-deviation local estimates, not
    automatically valid 95% intervals. Read ``amplitude_abs_status`` as well
    when reporting magnitude errors.
    """
    time_s = _validate_fit_contract(fit_results, fit_params)
    fit_params = _derive_frequency_bounds(
        fit_params, operation="estimate_fit_covariance"
    )
    if (
        isinstance(max_condition_number, (bool, np.bool_))
        or not isinstance(max_condition_number, Real)
        or not np.isfinite(max_condition_number)
        or max_condition_number <= 1
    ):
        raise ValueError("max_condition_number must be finite and greater than one.")
    for dataset, names in (
        (
            fit_results,
            ("fit_amps", "fit_freqs", "fit_t2s", "fit_skipped", "fit_status"),
        ),
        (fit_params, (*_MAP_NAMES, "skip_fitting")),
    ):
        missing = set(names) - set(dataset)
        if missing:
            raise KeyError(f"Local covariance requires variables {sorted(missing)}.")
    template = fit_results.fit_amps
    if "metabolite" not in template.dims or not template.sizes["metabolite"]:
        raise ValueError("fit_amps requires a nonempty labelled metabolite dimension.")
    for dim in template.dims:
        if dim not in template.coords or not template.get_index(dim).is_unique:
            raise ValueError(f"fit_amps requires unique coordinates for '{dim}'.")
    loop_dims = [dim for dim in template.dims if dim != "metabolite"]
    if {"fit_parameter", "fit_parameter_other"} & set(template.dims):
        raise ValueError(
            "Fit spectrum dimensions conflict with covariance parameter axes."
        )
    template = template.transpose(*loop_dims, "metabolite")
    fitted_maps = [
        _aligned_map(fit_results[name], template, name)
        for name in ("fit_freqs", "fit_t2s", "fit_skipped")
    ]
    skipped = fitted_maps[-1]
    expected_skip = _aligned_map(fit_params.skip_fitting, template, "skip_fitting")
    if skipped.dtype.kind != "b" or expected_skip.dtype.kind != "b":
        raise TypeError("fit_skipped and skip_fitting must be boolean.")
    if not bool((skipped == expected_skip).all().compute().item()):
        raise ValueError("fit_params skip_fitting must match the original fit_skipped.")
    maps = [_aligned_map(fit_params[name], template, name) for name in _MAP_NAMES]
    status = _aligned_map(
        fit_results.fit_status, template.isel(metabolite=0, drop=True), "fit_status"
    )
    if noise is None:
        if "noise_sigma_quadrature" not in fit_results:
            raise ValueError("Supply noise or a fit containing noise_sigma_quadrature.")
        noise = fit_results.noise_sigma_quadrature
    if isinstance(noise, xr.Dataset) and "noise_likelihood_compatible" in noise:
        if not bool(noise.noise_likelihood_compatible.all().compute().item()):
            raise ValueError(
                "Noise reference does not support the white-noise likelihood."
            )
    sigma = noise_sigma_for_fit(noise, template, require_valid=True)
    minimum_separation = float(
        fit_results.attrs.get("min_peak_separation_hz", DEFAULT_MIN_PEAK_SEPARATION_HZ)
    )
    if not np.isfinite(minimum_separation) or minimum_separation < 0:
        raise ValueError("min_peak_separation_hz must be finite and non-negative.")
    count = template.sizes["metabolite"]
    core_dims = [
        ["fit_parameter", "fit_parameter_other"],
        ["metabolite"],
        ["metabolite"],
        ["fit_parameter"],
        [],
        [],
        [],
        [],
    ]
    outputs = xr.apply_ufunc(
        _spectrum_covariance,
        sigma,
        template,
        *fitted_maps,
        *maps,
        status,
        input_core_dims=[[]] + [["metabolite"]] * 10 + [[]],
        output_core_dims=core_dims,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[
            float,
            float,
            np.int16,
            bool,
            np.int16,
            np.int64,
            float,
            np.int64,
        ],
        dask_gufunc_kwargs={
            "output_sizes": {
                "fit_parameter": 4 * count,
                "fit_parameter_other": 4 * count,
            }
        },
        kwargs={
            "time_s": time_s,
            "min_peak_separation_hz": minimum_separation,
            "max_condition_number": float(max_condition_number),
        },
    )
    names = (
        "parameter_covariance",
        "amplitude_abs_std",
        "amplitude_abs_status",
        "parameter_free",
        "covariance_status",
        "jacobian_rank",
        "jacobian_condition_number",
        "n_free_parameters",
    )
    result = xr.Dataset(dict(zip(names, outputs, strict=True)))
    for name, labels in (
        ("covariance_status", _COVARIANCE_STATUSES),
        ("amplitude_abs_status", _AMPLITUDE_STATUSES),
    ):
        result[name] = xr.apply_ufunc(
            _status_labels,
            result[name],
            kwargs={"labels": labels},
            dask="parallelized",
            output_dtypes=[np.dtype("U32")],
        )
    metabolites = [str(value) for value in template.metabolite.values]
    labels = [
        f"{kind}:{metabolite}"
        for kind in _PARAMETER_KINDS
        for metabolite in metabolites
    ]
    amplitude_unit = template.attrs.get("unit", "a.u.")
    units = np.repeat([amplitude_unit, amplitude_unit, "Hz", "s"], count)
    for axis, suffix in (("fit_parameter", ""), ("fit_parameter_other", "_other")):
        result = result.assign_coords(
            {
                axis: labels,
                f"parameter{suffix}_kind": (axis, np.repeat(_PARAMETER_KINDS, count)),
                f"parameter{suffix}_metabolite": (axis, np.tile(metabolites, 4)),
                f"parameter{suffix}_unit": (axis, units),
            }
        )
    diagonal = xr.apply_ufunc(
        np.diagonal,
        result.parameter_covariance,
        input_core_dims=[["fit_parameter", "fit_parameter_other"]],
        output_core_dims=[["fit_parameter"]],
        kwargs={"axis1": -2, "axis2": -1},
        dask="parallelized",
        output_dtypes=[float],
    )
    result["parameter_std"] = np.sqrt(diagonal.clip(min=0))
    # Work with a coordinate-free partner to avoid conflicts with the first axis.
    partner_std = xr.DataArray(
        result.parameter_std.data,
        dims=(*loop_dims, "fit_parameter_other"),
        coords={
            **{dim: result[dim] for dim in loop_dims},
            "fit_parameter_other": labels,
        },
    )
    denominator = result.parameter_std * partner_std
    result["parameter_correlation"] = result.parameter_covariance / denominator.where(
        denominator > 0
    )
    for index, name in ((2, "frequency_std"), (3, "t2_std")):
        values = result.parameter_std.isel(
            fit_parameter=slice(index * count, (index + 1) * count)
        )
        result[name] = xr.DataArray(
            values.data, dims=template.dims, coords=template.coords
        )
    result["fwhm_std"] = result.t2_std / (np.pi * fitted_maps[1] ** 2)
    result["magnitude_fwhm_std"] = MAGNITUDE_LORENTZ_FWHM_FACTOR * result.fwhm_std
    result["noise_sigma_quadrature"] = sigma
    result["residual_degrees_of_freedom"] = 2 * time_s.size - result.n_free_parameters
    for name, unit in (
        ("amplitude_abs_std", amplitude_unit),
        ("frequency_std", "Hz"),
        ("t2_std", "s"),
        ("fwhm_std", "Hz"),
        ("magnitude_fwhm_std", "Hz"),
    ):
        result[name].attrs.update(
            unit=unit,
            definition="Local one-standard-deviation error; not a confidence interval.",
        )
    result.parameter_covariance.attrs["unit"] = "parameter_unit * parameter_other_unit"
    result.parameter_std.attrs["unit"] = "See parameter_unit coordinate."
    result.parameter_correlation.attrs["unit"] = "dimensionless"
    result.attrs.update(
        covariance_method="Full real/imaginary FID Jacobian; column-scaled SVD",
        covariance_version="1",
        max_condition_number=float(max_condition_number),
        min_peak_separation_hz=minimum_separation,
        amplitude_ellipse_level=0.95,
        sampling_time_axis_s=time_s.copy(),
        covariance_scope="Local approximation conditional on the fitted model, fixed parameters and supplied white quadrature noise; excludes noise-estimation, model and selection uncertainty.",
        residual_variance_rescaling=False,
        source_model_name=fit_results.attrs["model_name"],
        source_numerical_contract_version=fit_results.attrs.get(
            "numerical_contract_version", "unspecified"
        ),
    )
    if fit_results.attrs.get("git_commit_hash") is not None:
        result.attrs["source_git_commit_hash"] = fit_results.attrs["git_commit_hash"]
    return result


__all__ = ["estimate_fit_covariance"]
