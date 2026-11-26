"""Direct quantitative map fits using the existing scientific implementations.

Named functions own preparation and choose one existing voxel or special solver
path. Generic and legacy APIs delegate here. Signal values, units, numerical
precision, labelled coordinates and algorithms retain their established contracts.
Calls are eager and create no result cache; importing this module fits no data.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.fitting.models.quantitative import adc, b1, ideal, t1, t2
from hypermri.analysis.fitting.voxelwise import VoxelwiseExecutor

from .options import (
    ADCFitOptions,
    B1FitOptions,
    IDEALFitOptions,
    T1FitOptions,
    T2FitOptions,
    _options_mapping,
)


def _fit_t1_voxel(signal, dynamic_coords, fit_params, indices):
    return t1.fit_t1_inversion_recovery_helper(
        signal=signal, inversion_times_ms=dynamic_coords, fit_params=fit_params
    ), indices


def _fit_adc_voxel(signal, dynamic_coords, fit_params, indices):
    return adc.fit_adc_monoexponential_helper(
        signal=signal, fit_params=fit_params
    ), indices


def _fit_t2_voxel(signal, dynamic_coords, fit_params, indices):
    return t2._fit_decay(
        signal, dynamic_coords, fit_params["signal_threshold"]
    ), indices


def _fit_b1_voxel(signal, dynamic_coords, fit_params, indices):
    return b1.fit_single_voxel_b1_period_sweep_helper(
        signal=signal,
        sqrt_refpower=np.sqrt(np.clip(dynamic_coords, 0, None)),
        fit_params=fit_params,
    ), indices


_QUANTITATIVE_VOXELS = {
    "t1": _fit_t1_voxel,
    "adc": _fit_adc_voxel,
    "b1": _fit_b1_voxel,
    "t2": _fit_t2_voxel,
}


def _normalize_options(model_id, fit_params):
    option_type = _QUANTITATIVE_MODELS[model_id][2]
    if isinstance(fit_params, option_type):
        if (
            model_id == "t2"
            and fit_params.execution is not None
            and not isinstance(fit_params.execution, ExecutionConfig)
        ):
            raise TypeError("T2 execution must be ExecutionConfig or None.")
        return _options_mapping(fit_params)
    if fit_params is None or isinstance(fit_params, dict):
        return dict(fit_params or {})
    if isinstance(fit_params, xr.Dataset):
        return dict(fit_params.attrs)
    raise TypeError(
        f"{model_id} fitting requires {option_type.__name__}, dict, prepared Dataset or None."
    )


def _prepare_quantitative_params(data, model_id, fit_params, return_params_info=False):
    module = _QUANTITATIVE_MODELS[model_id][0]
    if isinstance(fit_params, xr.Dataset):
        return getattr(module, f"prepare_{model_id}_fit_params_dataset")(
            data, fit_params, return_params_info
        )
    return getattr(module, f"prepare_{model_id}_fit_params_dataset")(
        data, _normalize_options(model_id, fit_params), return_params_info
    )


def _fit_prepared_quantitative_maps(data, model_id, fit_params):
    params = _normalize_options(model_id, fit_params)
    executor = VoxelwiseExecutor()
    module = _QUANTITATIVE_MODELS[model_id][0]
    if model_id == "ideal":
        fitted = ideal.fit_ideal_dataset(data, params)[0]
    elif model_id == "b1" and params.get("use_region_growing", False):
        fitted = b1.fit_b1_region_growing_dataset(data, params, executor)
    else:
        fitted = executor.execute(
            data=data,
            model_id=model_id,
            fit_params=params,
            single_voxel_fit_func=_QUANTITATIVE_VOXELS[model_id],
            output_param_names=module.OUTPUT_PARAM_NAMES,
            dynamic_dim=params.get(
                "dynamic_dim",
                {
                    "t1": "inversion_idx",
                    "adc": "bvalue_idx",
                    "b1": "rfpower_idx",
                    "t2": "echo_idx",
                }[model_id],
            ),
            model_name=module.RESULT_MODEL_NAME,
            model_equation=t2.MODEL_EQUATION if model_id == "t2" else None,
        )
    if model_id == "t2":
        t2._attach_metadata(fitted, data, params)
    from hypermri import __version__

    fitted.attrs["fit_prepared_options"] = deepcopy(params)
    fitted.attrs["fit_package_version"] = __version__
    return fitted


def _fit_quantitative_maps(data, model_id, fit_params, return_params_info):
    if isinstance(fit_params, xr.Dataset):
        if model_id == "t2":
            t2._validate_prepared(data, fit_params.attrs)
    else:
        fit_params = _prepare_quantitative_params(data, model_id, fit_params)
    if return_params_info:
        if model_id == "ideal":
            ideal._validate_ideal_solver_options(fit_params.attrs)
        return None
    return _fit_prepared_quantitative_maps(data, model_id, fit_params)


def fit_t1(
    data: xr.DataArray,
    fit_params: T1FitOptions | dict | xr.Dataset | None = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Fit inversion-recovery T1 maps with the established grid-search solver.

    Parameters
    ----------
    data : xarray.DataArray
        Real or complex signal with an ``inversion_idx`` dimension (or supplied
        ``dynamic_dim``). Its numerical coordinate is inversion time in ms; all
        other dimensions identify voxels. Sample order and input values are kept.
    fit_params : T1FitOptions or dict or xarray.Dataset or None, optional
        Supplied scientific grid/options or a prepared attribute dataset. Trial
        ``t1_values_s`` are seconds; fitted T1 is ms. A mapping is prepared once;
        a Dataset is already prepared. None retains existing model defaults.
        Supplied signal weighting, absolute/relative coefficient bounds,
        automatic bounds and ridge regularization reach the existing solver.
    return_params_info : bool, optional
        Return None after existing preparation/validation, without numerical fitting.

    Returns
    -------
    xarray.Dataset or None
        ``t1`` in ms, signal coefficients ``A``/``B`` in input signal units and
        ``rss_t1`` in squared signal units, weighted when requested and excluding
        any regularization penalty. Maps retain non-dynamic input order
        and compatible labels. Numeric maps are float64; complex A/B retain input
        complex dtype. Failed fits retain existing NaN outputs; no new status or
        uncertainty is inferred. Reconstruction inputs/units and existing provenance
        remain, with ``fit_prepared_options`` and ``fit_package_version`` added.
        Eager and deterministic for fixed input/options/backend; no retained cache.

    Raises
    ------
    TypeError
        If the option container belongs to another model or has an unsupported type.
    KeyError
        If the requested sample coordinate is missing.
    ValueError
        If sampling, scientific settings or execution options are incompatible.

    Notes
    -----
    Inputs and option mappings are preserved. Real floating signals are promoted
    to float64 by the existing executor. The model retains its original baseline,
    phase/complex handling, grid traversal and NaN failure behavior.
    Relative bounds override absolute bounds; automatic bounds override supplied
    relative bounds. Finite bounds select bounded least squares, which does not
    apply ridge regularization. Complex coefficients use the same bounds for
    their real and imaginary parts independently.

    Examples
    --------
    >>> result = fit_t1(data, T1FitOptions(t1_values_s=(0.5, 0.7, 0.9)))  # doctest: +SKIP
    """
    return _fit_quantitative_maps(data, "t1", fit_params, return_params_info)


def fit_adc(
    data: xr.DataArray,
    fit_params: ADCFitOptions | dict | xr.Dataset | None = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Fit magnitude ADC maps with the established log-linear diffusion model.

    Parameters
    ----------
    data : xarray.DataArray
        Signal with ``bvalue_idx`` (or supplied ``dynamic_dim``); its numerical
        coordinate supplies b-values in s/mm². Remaining dimensions identify voxels.
        Complex input is fitted by magnitude; labels/order and acquired values remain.
    fit_params : ADCFitOptions or dict or xarray.Dataset or None, optional
        Supplied threshold/axis/execution settings or prepared attributes. The
        threshold has input signal units. None retains the model's defaults;
        mappings are prepared once and Datasets are treated as prepared.
    return_params_info : bool, optional
        Return None after existing preparation/validation, without fitting.

    Returns
    -------
    xarray.Dataset or None
        Float64 ``adc`` in mm²/s, ``s0`` in input signal units and ``rss_adc``
        in squared log-magnitude units, in non-dynamic input dimension order.
        Too few valid samples, deficient designs and negative diffusion estimates
        retain existing NaN behavior. No new status or uncertainty is inferred.
        Existing coordinates/reconstruction metadata/provenance remain; prepared
        options and package version are recorded in two additional attributes.
        Calls are eager, deterministic and cache-free.

    Raises
    ------
    TypeError
        If the option container type is unsupported or belongs to another model.
    KeyError
        If the sample coordinate is absent.
    ValueError
        If options or sample dimensions are incompatible.

    Notes
    -----
    Input data and settings are preserved. The voxel coordinate remains the
    numerical b-value axis even when legacy preparation supplies ``b_values``.
    Thresholding, magnitude conversion and regression retain their current rules.

    Examples
    --------
    >>> result = fit_adc(data, ADCFitOptions(signal_threshold=1e-5))  # doctest: +SKIP
    """
    return _fit_quantitative_maps(data, "adc", fit_params, return_params_info)


def fit_t2(
    data: xr.DataArray,
    fit_params: T2FitOptions | dict | xr.Dataset | None = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Fit monoexponential transverse relaxation in magnitude signal space.

    Parameters
    ----------
    data : xarray.DataArray
        Real or complex echo images. The dynamic coordinate (default echo_idx)
        must declare unit='s', contain at least three finite nonnegative echo
        times and at least two distinct times. All other dimensions are fitted
        independently, retaining their order, coordinates and orientation.
    fit_params : T2FitOptions or dict or xarray.Dataset or None, optional
        Signal threshold, dynamic_dim and shared worker options. None uses
        threshold 1e-5 in input units and existing worker defaults. Ordinary
        settings are prepared once; prepared datasets are validated and reused.
    return_params_info : bool, optional
        Validate settings/sampling and return None without reading/fitting signal.

    Returns
    -------
    xarray.Dataset or None
        Float64 t2 in ms (maximum 5000 ms), s0 in input signal units and rss_t2,
        the sum of squared magnitude signal residuals over included echoes, in
        squared signal units.
        echo_times_s supports model reconstruction. Compatible coordinates and
        input metadata, effective options and package version are retained.
        Insufficient valid samples or unsuccessful numerical fits yield NaNs.
        Calls are eager, deterministic and preserve inputs/options without cache.

    Raises
    ------
    TypeError, ValueError
        Unsupported/unknown options, invalid coordinate units or echo sampling.

    Notes
    -----
    Uses unweighted nonlinear least squares, no offset, all echoes, S0 >= 0 and
    0 < T2 <= 5 s; S0 has no finite upper bound. The fixed model bound is
    applied during optimization and recorded as maximum_t2_s in result attributes.
    Only finite magnitudes strictly above threshold participate; at least three
    valid echoes and two distinct times are required per voxel. A log regression
    initializes the signal-space solver.
    Signal normalization improves conditioning without changing the RSS objective.
    Internal times and optimization use seconds; one model-owned conversion
    reports ms. No uncertainty or statistical/physiological quality is inferred.
    The solver uses relative tolerances 1e-10 and at most 2000 evaluations.

    Examples
    --------
    >>> result = fit_t2(echo_images, T2FitOptions(signal_threshold=1e-5))  # doctest: +SKIP
    >>> prediction = result.hp.reconstruct()  # doctest: +SKIP
    """
    return _fit_quantitative_maps(data, "t2", fit_params, return_params_info)


def fit_b1(
    data: xr.DataArray,
    fit_params: B1FitOptions | dict | xr.Dataset | None = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Fit an RF-power sweep through the existing sinusoidal B1 solver.

    Parameters
    ----------
    data : xarray.DataArray
        Signal with an RF-power sample dimension (normally ``rfpower_idx``),
        whose numerical coordinate is W. Other dimensions identify voxels.
        Region growing requires the existing two-dimensional spatial layout;
        extra spatial dimensions must be singleton. Coordinates are preserved.
    fit_params : B1FitOptions or dict or xarray.Dataset or None, optional
        Pulse calibration (``fa_deg``, ``trf_ms``, dimensionless ``sint``),
        period/seed/region settings and execution options, or prepared attributes.
        Periods use sqrt(W). Mappings are prepared once; prepared Datasets retain
        advanced sweep/region settings. Omitted settings retain model defaults.
    return_params_info : bool, optional
        Return None after existing preparation/validation, without fitting.

    Returns
    -------
    xarray.Dataset or None
        ``amplitude`` in signal units (input complex dtype when complex),
        float64 ``period`` in sqrt(W), ``refpower_90deg_W`` in W, ``rss_b1``
        in squared signal units and float64 sample/stop maps. Ordinary fitting
        retains non-dynamic input order. Region debug maps retain their existing
        two-dimensional layout. NaN fits remain NaN; no new statistical status
        or uncertainty is inferred. Existing metadata/provenance remain, with
        prepared options and package version recorded in additional attributes.
        Calls are eager, deterministic for fixed settings/backend and cache-free.

    Raises
    ------
    TypeError
        If the option container type is unsupported or belongs to another model.
    KeyError
        If the numerical RF-power coordinate is absent during execution.
    ValueError
        If power, calibration, seed/layout or execution settings are incompatible.

    Notes
    -----
    Inputs and supplied options are preserved. Square-root power conversion,
    period refinement, fit-window selection, seed order and absolute calibration
    remain in the existing scientific model. This operation fits signal data;
    it does not reconstruct acquired B1 raw data.

    Examples
    --------
    >>> result = fit_b1(data, B1FitOptions(fa_deg=90.0, trf_ms=1.0, sint=1.0))  # doctest: +SKIP
    """
    return _fit_quantitative_maps(data, "b1", fit_params, return_params_info)


def fit_ideal(
    data: xr.DataArray,
    fit_params: IDEALFitOptions | dict | xr.Dataset | None = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Fit labelled echo component amplitudes with the established IDEAL solver.

    Parameters
    ----------
    data : xarray.DataArray
        Complex echo signal with ``echo_idx`` (or supplied ``echo_dim``).
        Non-singleton remaining dimensions must be among ``read_idx``,
        ``phase_idx``, ``slice_idx`` and ``repetition_idx``. The solver uses
        centered equally spaced times from ``echo_spacing_s`` in seconds.
    fit_params : IDEALFitOptions or dict or xarray.Dataset or None, optional
        Relative component frequencies in Hz, echo spacing in seconds, names,
        execution options or prepared attributes. Both historical frequency
        spellings retain their existing precedence/conflict rules. A mapping is
        prepared once; supplied solve selection, field/Theta initialization,
        convergence thresholds and iteration cap reach the existing solver.
        None retains defaults, including optimization of both Field_offset and
        Theta. Initialization aliases and units are documented in IDEALFitOptions.
    return_params_info : bool, optional
        Return None after existing preparation/validation, without solving echoes.

    Returns
    -------
    xarray.Dataset or None
        Complex128 ``fit_amps`` in signal units with ``metabolite`` last, and
        float64 ``field_offset`` in Hz when selected by the solver. Spatial axes
        follow the solver's standard order; singleton axes are squeezed as before.
        Compatible labels/attributes remain. No new status, uncertainty or generic
        reconstruction is introduced. Prepared options and package version are
        recorded in two additional attributes. Eager and deterministic for fixed
        solver options/backend; no cache or acquired-input mutation occurs.

    Raises
    ------
    TypeError
        If the option container type is unsupported or belongs to another model.
    ValueError
        If echo calibration, labels, dimension layout or solver settings are invalid.
        T2-star option requests are rejected because that physics is not implemented;
        remove those options or use a separate T2-star relaxation fitting model.
    numpy.linalg.LinAlgError
        If the existing solver encounters a singular design.

    Notes
    -----
    The existing solver computes in complex128, preserves inputs and retains its
    existing console/progress output. Numerical initialization, field/transient
    options and convergence rules remain owned by the IDEAL model/solver.
    The existing schema exposes no Theta map, even when Theta is optimized.

    Examples
    --------
    >>> result = fit_ideal(data, IDEALFitOptions(
    ...     relative_frequencies_hz=(0.0, 120.0), echo_spacing_s=0.001
    ... ))  # doctest: +SKIP
    """
    return _fit_quantitative_maps(data, "ideal", fit_params, return_params_info)


_QUANTITATIVE_MODELS = {
    "t1": (t1, fit_t1, T1FitOptions),
    "adc": (adc, fit_adc, ADCFitOptions),
    "b1": (b1, fit_b1, B1FitOptions),
    "ideal": (ideal, fit_ideal, IDEALFitOptions),
    "t2": (t2, fit_t2, T2FitOptions),
}


def _find_quantitative_model(name):
    for key, model in _QUANTITATIVE_MODELS.items():
        if name in (key, model[0].RESULT_MODEL_NAME):
            return key, model
    return None


__all__ = ["fit_t1", "fit_adc", "fit_b1", "fit_ideal", "fit_t2"]
