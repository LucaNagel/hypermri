"""Report local spectroscopy parameter intervals without repeated fits.

The module turns the joint covariance of a complex FID fit into labelled local
intervals.  It keeps non-identifiable, boundary-limited and grid-limited cases
explicit instead of returning apparently precise symmetric error bars.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, f, norm, t

from ..fitting.models.spectroscopy.fid_model import (
    MAGNITUDE_LORENTZ_FWHM_FACTOR,
    _derive_frequency_bounds,
)
from .covariance import _aligned_map, estimate_fit_covariance
from .fit_diagnostics import NoiseInput

_INTERVAL_STATUSES = (
    "success",
    "includes_zero",
    "fixed_by_assumption",
    "amplitude_near_zero",
    "interval_intersects_bound",
    "grid_limited",
    "skipped",
    "no_active_metabolites",
    "invalid_fit",
    "invalid_parameters",
    "parameter_bound",
    "rank_deficient",
    "ill_conditioned",
    "numerical_failure",
)


def _ellipse_magnitude_bounds(
    amplitude_real,
    amplitude_imag,
    variance_real,
    covariance_real_imag,
    variance_imag,
    radius_squared,
):
    """Return radial projection bounds of one Cartesian confidence ellipse."""
    values = np.asarray(
        [
            amplitude_real,
            amplitude_imag,
            variance_real,
            covariance_real_imag,
            variance_imag,
            radius_squared,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or radius_squared <= 0:
        return np.nan, np.nan
    covariance = np.array(
        [[variance_real, covariance_real_imag], [covariance_real_imag, variance_imag]]
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0:
        return np.nan, np.nan
    center = np.array([amplitude_real, amplitude_imag])
    factor = np.linalg.cholesky(covariance) * np.sqrt(radius_squared)

    def squared_radius(angle):
        direction = np.array([np.cos(angle), np.sin(angle)])
        point = center + factor @ direction
        return float(point @ point)

    # The objective contains only first and second angular harmonics.  A dense
    # deterministic bracket therefore locates every relevant extremum before a
    # bounded scalar refinement, without Monte Carlo sampling.
    count = 64
    step = 2 * np.pi / count
    angles = np.arange(count) * step
    squared = np.array([squared_radius(angle) for angle in angles])

    def refine(index, sign):
        center_angle = angles[index]
        optimized = minimize_scalar(
            lambda angle: sign * squared_radius(angle),
            bounds=(center_angle - step, center_angle + step),
            method="bounded",
            options={"xatol": 1e-13},
        )
        return squared_radius(optimized.x)

    maximum = max(float(squared.max()), refine(int(np.argmax(squared)), -1.0))
    mahalanobis = center @ np.linalg.solve(covariance, center)
    if mahalanobis <= radius_squared:
        minimum = 0.0
    else:
        minimum = min(float(squared.min()), refine(int(np.argmin(squared)), 1.0))
    return np.sqrt(max(minimum, 0.0)), np.sqrt(max(maximum, 0.0))


def _status_labels(codes):
    """Decode local interval status codes without Unicode truncation."""
    return np.asarray(_INTERVAL_STATUSES, dtype="U32")[codes]


def _status_codes(
    skipped,
    covariance_status,
    amplitude_status,
    free,
    intersects_bound,
    grid_limited,
):
    """Classify one nonlinear parameter interval in precedence order."""
    if bool(skipped):
        status = "skipped"
    elif str(covariance_status) != "success":
        status = str(covariance_status)
    elif not bool(free):
        status = "fixed_by_assumption"
    elif str(amplitude_status) == "near_zero":
        status = "amplitude_near_zero"
    elif bool(intersects_bound):
        status = "interval_intersects_bound"
    elif bool(grid_limited):
        status = "grid_limited"
    else:
        status = "success"
    return np.int16(_INTERVAL_STATUSES.index(status))


def _parameter_kind_map(
    values: xr.DataArray,
    *,
    kind: str,
    template: xr.DataArray,
) -> xr.DataArray:
    """Select one parameter kind and restore the fitted metabolite layout."""
    labels = [f"{kind}:{value}" for value in template.metabolite.values]
    selected = values.sel(fit_parameter=labels)
    loop_dims = tuple(dim for dim in template.dims if dim != "metabolite")
    restored = xr.DataArray(
        selected.data,
        dims=(*loop_dims, "metabolite"),
        coords={
            **{dim: template[dim] for dim in loop_dims},
            "metabolite": template.metabolite,
        },
    )
    return restored.transpose(*template.dims)


def _align_degrees_of_freedom(
    value: Real | xr.DataArray,
    template: xr.DataArray,
) -> xr.DataArray:
    """Validate and align independent noise-estimate degrees of freedom."""
    loop_template = template.isel(metabolite=0, drop=True)
    if isinstance(value, xr.DataArray):
        degrees = _aligned_map(value, loop_template, "noise_degrees_of_freedom")
    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        degrees = xr.full_like(loop_template.real, float(value), dtype=float)
    else:
        raise TypeError(
            "noise_degrees_of_freedom must be a real scalar or xarray.DataArray."
        )
    values = np.asarray(degrees.values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(
            "noise_degrees_of_freedom must contain only finite positive values."
        )
    return degrees.astype(float)


def _grid_step(
    width: xr.DataArray,
    fit_results: xr.Dataset,
    *,
    template: xr.DataArray,
) -> xr.DataArray:
    """Return the final spacing of the current alternating grid estimator."""
    npoints = int(fit_results.attrs.get("npoints", 0))
    repetitions = int(fit_results.attrs.get("rep_fitting", 0))
    zoom = float(fit_results.attrs.get("zoomfactor", np.nan))
    if npoints < 2 or repetitions < 1 or not np.isfinite(zoom) or zoom <= 0:
        raise ValueError(
            "fit_results must record npoints >= 2, rep_fitting >= 1 and a "
            "finite positive zoomfactor for interval grid diagnostics."
        )
    aligned = _aligned_map(width, template, width.name or "search_width")
    return aligned / (zoom ** (repetitions - 1) * (npoints - 1))


def estimate_fit_local_uncertainty(
    fit_results: xr.Dataset,
    fit_params: xr.Dataset,
    *,
    noise: NoiseInput | None = None,
    confidence_level: float = 0.95,
    noise_degrees_of_freedom: Real | xr.DataArray | None = None,
    max_condition_number: float = 1e8,
    max_grid_step_fraction: float = 0.2,
) -> xr.Dataset:
    """Estimate reportable local parameter intervals without repeated fits.

    Compute the full joint covariance of an ordinary complex-RSS FID fit and
    convert it into labelled confidence intervals.  Complex-amplitude magnitude
    intervals are radial projections of their two-dimensional Cartesian
    ellipses.  Frequency intervals use a marginal Wald approximation.  Positive
    T2* intervals use a log-scale delta approximation, and absorption and
    magnitude FWHM intervals are transformed from the same T2* endpoints.

    Parameters
    ----------
    fit_results
        Results returned by :func:`fit_spectrum`.  Complex amplitudes are FID
        amplitudes at model time zero; frequencies are in Hz and T2* values are
        in seconds.  Arbitrary labelled spectrum dimensions are preserved.
    fit_params
        The unchanged prepared parameter Dataset used for ``fit_results``.
        Absolute bounds, search widths, skipped components and sampling metadata
        must match the fit exactly.
    noise
        Optional known or independently estimated standard deviation per real or
        imaginary FID quadrature.  The accepted forms and white-noise assumptions
        are those of :func:`estimate_fit_covariance`.  By default reuse
        ``noise_sigma_quadrature`` from ``fit_results``.
    confidence_level
        Two-sided local confidence level strictly between zero and one.  The
        default is 0.95.
    noise_degrees_of_freedom
        Optional positive effective degrees of freedom of an independent noise
        variance estimate.  A scalar or labelled per-spectrum map is accepted.
        Supplying it uses Student-t marginal critical values and a scaled-F
        Cartesian amplitude ellipse.  Leave as ``None`` when sigma is treated as
        known.  For ``M`` independent proper complex samples, the estimator
        ``sum(abs(z)**2)/(2*M)`` has ``2*M`` degrees of freedom; subtract two
        when an independently justified complex mean was first estimated.
    max_condition_number
        Finite upper condition-number limit delegated to
        :func:`estimate_fit_covariance`.  It is a numerical safeguard rather
        than a calibrated identifiability threshold.
    max_grid_step_fraction
        Positive maximum ratio of the final frequency or T2* grid spacing to
        its local standard error.  The default 0.2 requires at least five grid
        steps per standard error.  Coarser free estimates receive
        ``"grid_limited"`` and no reportable interval.

    Returns
    -------
    xr.Dataset
        All variables and coordinates from :func:`estimate_fit_covariance`, plus
        ``amplitude_abs_interval_low/high``, ``frequency_interval_low/high``,
        ``t2_interval_low/high``, ``fwhm_interval_low/high`` and
        ``magnitude_fwhm_interval_low/high``.  Their dimensions match
        ``fit_amps`` and their units are amplitude, Hz, seconds, Hz and Hz.
        The absorption FWHM interval is ``1/(pi*T2*)`` and the magnitude FWHM
        interval ``sqrt(3)/(pi*T2*)`` of the T2* endpoints.

        ``amplitude_abs_interval_status``, ``frequency_interval_status`` and
        ``t2_interval_status`` must be read before reporting an interval.
        ``"success"`` marks a locally reportable estimate;
        ``"includes_zero"`` is a finite amplitude interval with lower bound
        zero; ``"fixed_by_assumption"`` is a point interval, not measured
        precision.  Free nonlinear parameters are suppressed as
        ``"amplitude_near_zero"``, ``"interval_intersects_bound"`` or
        ``"grid_limited"`` when the regular local interpretation fails.  The
        covariance failure statuses and ``"skipped"`` are propagated.

        ``interval_critical_value`` and
        ``amplitude_ellipse_radius_squared`` are per-spectrum maps.  The output
        records whether sigma was treated as known or as an independent variance
        estimate.  No fit, resampling, detection test or input mutation occurs.

    Raises
    ------
    TypeError
        If the confidence, grid or degrees-of-freedom inputs have unsupported
        types, or if delegated covariance inputs violate their type contract.
    KeyError
        If required fit variables or parameter maps are absent.
    ValueError
        If numerical options, coordinates, parameter bounds, fit metadata or
        noise inputs are invalid or inconsistent, or if the fit uses
        ``lineshape='voigt'``, which the local model does not support.

    Warns
    -----
    UserWarning
        If stored ``freq_range_lower_bound_hz`` or ``freq_range_upper_bound_hz``
        differ from ``initial_freqs_hz`` -/+ ``metabs_freq_ranges`` / 2. The
        bounds always follow start and full search width, so the derived
        bounds are used; ``fit_params`` itself is not modified.

    Notes
    -----
    These intervals remain local approximations conditional on the selected
    model, fitted peak set, supplied white-noise likelihood and fixed parameters.
    A successful status does not establish repeated-sampling coverage under
    model mismatch, colored noise, preprocessing uncertainty or nonlinear
    multimodality.  Boundary and other nonregular cases require a profile-
    likelihood fallback or an explicit ``not identifiable`` result.

    Examples
    --------
    Produce parameter estimates and local 95% intervals from one existing fit::

        from hypermri.analysis.spectroscopy import estimate_fit_local_uncertainty

        uncertainty = estimate_fit_local_uncertainty(fit, fit_params)
        usable_frequency = uncertainty.frequency_interval_status == "success"
        frequency_low_hz = uncertainty.frequency_interval_low.where(
            usable_frequency
        )
        frequency_high_hz = uncertainty.frequency_interval_high.where(
            usable_frequency
        )
    """
    if (
        isinstance(confidence_level, (bool, np.bool_))
        or not isinstance(confidence_level, Real)
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError(
            "confidence_level must be finite and strictly between 0 and 1."
        )
    if (
        isinstance(max_grid_step_fraction, (bool, np.bool_))
        or not isinstance(max_grid_step_fraction, Real)
        or not np.isfinite(max_grid_step_fraction)
        or max_grid_step_fraction <= 0
    ):
        raise ValueError("max_grid_step_fraction must be finite and positive.")

    fit_params = _derive_frequency_bounds(
        fit_params, operation="estimate_fit_local_uncertainty"
    )
    result = estimate_fit_covariance(
        fit_results,
        fit_params,
        noise=noise,
        max_condition_number=max_condition_number,
    )
    template = fit_results.fit_amps
    loop_template = template.isel(metabolite=0, drop=True).real
    probability = (1.0 + float(confidence_level)) / 2.0
    if noise_degrees_of_freedom is None:
        degrees = xr.full_like(loop_template, np.inf, dtype=float)
        critical = xr.full_like(loop_template, norm.ppf(probability), dtype=float)
        ellipse_radius_squared = xr.full_like(
            loop_template, chi2.ppf(confidence_level, 2), dtype=float
        )
        noise_scale_scope = "known_sigma"
    else:
        degrees = _align_degrees_of_freedom(noise_degrees_of_freedom, template)
        critical = xr.apply_ufunc(
            lambda value: t.ppf(probability, value),
            degrees,
            dask="parallelized",
            output_dtypes=[float],
        )
        ellipse_radius_squared = xr.apply_ufunc(
            lambda value: 2.0 * f.ppf(confidence_level, 2, value),
            degrees,
            dask="parallelized",
            output_dtypes=[float],
        )
        noise_scale_scope = "independent_estimated_sigma"

    metabolites = list(template.metabolite.values)
    real_labels = [f"amplitude_real:{value}" for value in metabolites]
    imag_labels = [f"amplitude_imag:{value}" for value in metabolites]
    variance_real = result.parameter_covariance.sel(
        fit_parameter=real_labels, fit_parameter_other=real_labels
    )
    variance_imag = result.parameter_covariance.sel(
        fit_parameter=imag_labels, fit_parameter_other=imag_labels
    )
    covariance_real_imag = result.parameter_covariance.sel(
        fit_parameter=real_labels, fit_parameter_other=imag_labels
    )
    diagonal_index = xr.DataArray(
        np.arange(len(metabolites)),
        dims="metabolite",
        coords={"metabolite": metabolites},
    )
    variance_real = variance_real.isel(
        fit_parameter=diagonal_index, fit_parameter_other=diagonal_index
    )
    variance_imag = variance_imag.isel(
        fit_parameter=diagonal_index, fit_parameter_other=diagonal_index
    )
    covariance_real_imag = covariance_real_imag.isel(
        fit_parameter=diagonal_index, fit_parameter_other=diagonal_index
    )
    amplitude_low, amplitude_high = xr.apply_ufunc(
        _ellipse_magnitude_bounds,
        template.real,
        template.imag,
        variance_real,
        covariance_real_imag,
        variance_imag,
        ellipse_radius_squared,
        input_core_dims=[[]] * 6,
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    amplitude_status = xr.where(
        result.amplitude_abs_status == "near_zero",
        "includes_zero",
        result.amplitude_abs_status,
    )
    amplitude_low = amplitude_low.where(
        amplitude_status.isin(["success", "includes_zero"])
    )
    amplitude_high = amplitude_high.where(
        amplitude_status.isin(["success", "includes_zero"])
    )

    frequency_free = _parameter_kind_map(
        result.parameter_free, kind="frequency", template=template
    )
    t2_free = _parameter_kind_map(result.parameter_free, kind="t2", template=template)
    frequency_raw_low = fit_results.fit_freqs - critical * result.frequency_std
    frequency_raw_high = fit_results.fit_freqs + critical * result.frequency_std
    log_t2_half_width = critical * result.t2_std / fit_results.fit_t2s
    t2_raw_low = fit_results.fit_t2s * np.exp(-log_t2_half_width)
    t2_raw_high = fit_results.fit_t2s * np.exp(log_t2_half_width)

    frequency_lower = _aligned_map(
        fit_params.freq_range_lower_bound_hz,
        template,
        "freq_range_lower_bound_hz",
    )
    frequency_upper = _aligned_map(
        fit_params.freq_range_upper_bound_hz,
        template,
        "freq_range_upper_bound_hz",
    )
    t2_lower = _aligned_map(fit_params.min_t2_s, template, "min_t2_s")
    t2_upper = _aligned_map(fit_params.max_t2_s, template, "max_t2_s")
    frequency_intersects = frequency_free & (
        (frequency_raw_low < frequency_lower) | (frequency_raw_high > frequency_upper)
    )
    t2_intersects = t2_free & ((t2_raw_low < t2_lower) | (t2_raw_high > t2_upper))

    frequency_step = _grid_step(
        fit_params.metabs_freq_ranges.rename("metabs_freq_ranges"),
        fit_results,
        template=template,
    )
    t2_step = _grid_step(
        fit_params.metabs_t2_ranges.rename("metabs_t2_ranges"),
        fit_results,
        template=template,
    )
    frequency_grid_limited = frequency_free & (
        frequency_step > max_grid_step_fraction * result.frequency_std
    )
    t2_grid_limited = t2_free & (t2_step > max_grid_step_fraction * result.t2_std)
    covariance_status = result.covariance_status.broadcast_like(template)

    def parameter_status(free, intersects, limited):
        codes = xr.apply_ufunc(
            _status_codes,
            fit_results.fit_skipped,
            covariance_status,
            result.amplitude_abs_status,
            free,
            intersects,
            limited,
            input_core_dims=[[]] * 6,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int16],
        )
        return xr.apply_ufunc(
            _status_labels,
            codes,
            dask="parallelized",
            output_dtypes=[np.dtype("U32")],
        )

    frequency_status = parameter_status(
        frequency_free, frequency_intersects, frequency_grid_limited
    )
    t2_status = parameter_status(t2_free, t2_intersects, t2_grid_limited)
    frequency_reportable = frequency_status.isin(["success", "fixed_by_assumption"])
    t2_reportable = t2_status.isin(["success", "fixed_by_assumption"])

    result["amplitude_abs_interval_low"] = amplitude_low
    result["amplitude_abs_interval_high"] = amplitude_high
    result["amplitude_abs_interval_status"] = amplitude_status
    result["frequency_interval_low"] = frequency_raw_low.where(frequency_reportable)
    result["frequency_interval_high"] = frequency_raw_high.where(frequency_reportable)
    result["frequency_interval_status"] = frequency_status
    result["t2_interval_low"] = t2_raw_low.where(t2_reportable)
    result["t2_interval_high"] = t2_raw_high.where(t2_reportable)
    result["t2_interval_status"] = t2_status
    result["fwhm_interval_low"] = 1.0 / (np.pi * result.t2_interval_high)
    result["fwhm_interval_high"] = 1.0 / (np.pi * result.t2_interval_low)
    for side in ("low", "high"):
        result[f"magnitude_fwhm_interval_{side}"] = (
            MAGNITUDE_LORENTZ_FWHM_FACTOR * result[f"fwhm_interval_{side}"]
        )
    result["frequency_grid_step"] = frequency_step
    result["t2_grid_step"] = t2_step
    result["interval_critical_value"] = critical
    result["amplitude_ellipse_radius_squared"] = ellipse_radius_squared
    result["noise_degrees_of_freedom"] = degrees

    amplitude_unit = template.attrs.get("unit", "a.u.")
    for name, unit in (
        ("amplitude_abs_interval_low", amplitude_unit),
        ("amplitude_abs_interval_high", amplitude_unit),
        ("frequency_interval_low", "Hz"),
        ("frequency_interval_high", "Hz"),
        ("t2_interval_low", "s"),
        ("t2_interval_high", "s"),
        ("fwhm_interval_low", "Hz"),
        ("fwhm_interval_high", "Hz"),
        ("magnitude_fwhm_interval_low", "Hz"),
        ("magnitude_fwhm_interval_high", "Hz"),
        ("frequency_grid_step", "Hz"),
        ("t2_grid_step", "s"),
    ):
        result[name].attrs["unit"] = unit
    for name in (
        "amplitude_abs_interval_low",
        "amplitude_abs_interval_high",
        "frequency_interval_low",
        "frequency_interval_high",
        "t2_interval_low",
        "t2_interval_high",
        "fwhm_interval_low",
        "fwhm_interval_high",
        "magnitude_fwhm_interval_low",
        "magnitude_fwhm_interval_high",
    ):
        result[name].attrs["confidence_level"] = float(confidence_level)
        result[name].attrs["scope"] = "local_conditional_approximation"
    result.interval_critical_value.attrs.update(
        unit="dimensionless",
        distribution="normal" if noise_degrees_of_freedom is None else "student_t",
    )
    result.amplitude_ellipse_radius_squared.attrs.update(
        unit="dimensionless",
        distribution=(
            "chi_squared_2"
            if noise_degrees_of_freedom is None
            else "2_times_F_2_noise_degrees_of_freedom"
        ),
    )
    result.noise_degrees_of_freedom.attrs.update(
        unit="dimensionless",
        definition="Infinity means supplied sigma was treated as known.",
    )
    result.attrs.update(
        local_uncertainty_version="1",
        confidence_level=float(confidence_level),
        noise_scale_scope=noise_scale_scope,
        max_grid_step_fraction=float(max_grid_step_fraction),
        interval_scope=(
            "Local approximation conditional on the fitted model, selected peak set, "
            "fixed parameters and supplied white-noise likelihood; successful status "
            "does not by itself validate repeated-sampling coverage."
        ),
        amplitude_interval_method="Radial projection of joint Cartesian ellipse",
        frequency_interval_method="Marginal Wald interval",
        t2_interval_method="Log-scale delta interval",
        fwhm_interval_method="Monotone transform of T2* interval",
    )
    return result


__all__ = ["estimate_fit_local_uncertainty"]
