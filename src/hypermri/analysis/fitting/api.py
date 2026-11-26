"""Compatibility fitting facade delegating to scientific domain operations.

DCE and quantitative models use explicit domain dictionaries; spectroscopy and
custom registered models retain compatible adapters. Named APIs own units, labelled dimensions,
precision, scientific defaults and status meaning. Calls preserve acquired
values, are eager and deterministic for fixed inputs/options/backends, and
create no result cache. Importing this module does not fit or reconstruct data.
"""

from __future__ import annotations

import xarray as xr

from .registry import get_registry


def prepare_fit_params(
    data,
    model: str,
    fit_params: dict | xr.Dataset | None = None,
    return_params_info: bool = False,
) -> xr.Dataset | None:
    """Prepare scientific fit parameters through their owning domain.

    Parameters
    ----------
    data : xarray.DataArray or None
        Labelled acquired samples and model-specific sampling coordinates/units.
        None is permitted only by preparers that can describe options without data.
    model : str
        Canonical model key or supported result-name alias. DCE selection uses its
        explicit dictionary; T1, ADC, B1 and IDEAL use their quantitative domain
        dictionary. Other models retain their compatible registry adapters.
    fit_params : dict or xarray.Dataset or None, optional
        Scientific and execution settings in the selected model's documented units.
        Dataset attributes provide legacy settings; no acquired values are modified.
    return_params_info : bool, optional
        Request the preparer's existing description/printing behavior instead of a
        fitting parameter dataset. This does not run a numerical fit.

    Returns
    -------
    xarray.Dataset or None
        Prepared parameter values/attributes in the model's existing schema and
        precision, or None for description-only behavior. DCE times are seconds and
        concentration inputs mM. Preparation is eager and creates no result cache.

    Raises
    ------
    KeyError
        If the model key or alias is unknown.
    ValueError
        If required scientific inputs are invalid, or the selected model has no
        legacy parameter preparer. Use the named 2C-IRF or empirical domain function
        for those concentration workflows.

    Examples
    --------
    >>> params = prepare_fit_params(
    ...     concentrations, "patlak", {"aif_signal": plasma, "time_points_s": times_s}
    ... )  # doctest: +SKIP
    """
    from hypermri.analysis.dce.execution import _prepare_dce_params
    from hypermri.analysis.dce.model_selection import _find_dce_model

    dce_model = _find_dce_model(model)
    if dce_model is not None:
        return _prepare_dce_params(data, dce_model[0], fit_params, return_params_info)
    from hypermri.analysis.quantitative.fitting import (
        _find_quantitative_model,
        _prepare_quantitative_params,
    )

    quantitative_model = _find_quantitative_model(model)
    if quantitative_model is not None:
        return _prepare_quantitative_params(
            data, quantitative_model[0], fit_params, return_params_info
        )
    spec = get_registry().get(model)
    if spec.prepare_params is None:
        raise ValueError(f"Model '{model}' does not provide parameter preparation.")
    return spec.prepare_params(data, fit_params, return_params_info)


def fit(
    data,
    model: str,
    fit_params: dict | xr.Dataset | None = None,
    return_params_info: bool = False,
    **kwargs,
) -> xr.Dataset | None:
    """Fit labelled scientific samples through the selected domain operation.

    Parameters
    ----------
    data : xarray.DataArray or None
        Acquired data with the selected model's labelled dynamic/spatial dimensions
        and units. A numerical fit requires data; None retains existing information-
        only behavior where the model preparer supports it.
    model : str
        Canonical model key or existing result-name alias. DCE models with a legacy
        preparer call concentration-map execution directly, without registry fitting.
        T1, ADC, B1 and IDEAL call their named quantitative functions directly.
    fit_params : dict or xarray.Dataset or None, optional
        Scientific settings in model-specific units and shared worker options. A
        dataset is already prepared; a mapping is prepared once. Scientific defaults,
        bounds, status meaning and precision remain owned by the selected model.
    return_params_info : bool, optional
        Return None instead of running a fit after existing parameter validation.
    **kwargs
        Legacy option entries merged into a dictionary; explicit entries here override
        matching fit_params keys. Cannot be combined with a prepared dataset. They
        are retained for compatibility, not forwarded as numerical function keywords.

    Returns
    -------
    xarray.Dataset or None
        Model-specific parameter/status/uncertainty variables with labelled spatial
        dimensions, coordinates and reconstruction/provenance metadata. DCE numeric
        calculations/maps are float64; Sourbron counts are int32 and its success flag
        boolean. Other precision and units follow the named model API. None retains
        information-only behavior. Calls are eager and deterministic for fixed
        inputs, scientific options and backend, and create no result cache.
        Quantitative maps retain their existing schema/precision and add copied
        ``fit_prepared_options`` and the actual ``fit_package_version``.

    Raises
    ------
    KeyError
        If a model or required dynamic coordinate is unknown/missing.
    ValueError
        If scientific inputs/options are inconsistent or the model has no generic
        parameter-preparation/fit contract. Use the named DCE workflows for 2C-IRF
        and empirical concentration maps.
    TypeError
        If keyword options are combined with a prepared parameter dataset.

    Examples
    --------
    >>> fitted = fit(
    ...     concentrations, "patlak", {"aif_signal": plasma, "time_points_s": times_s}
    ... )  # doctest: +SKIP
    """
    if kwargs:
        if fit_params is None:
            fit_params = dict(kwargs)
        elif isinstance(fit_params, dict):
            fit_params = {**fit_params, **kwargs}
        else:
            raise TypeError(
                "Keyword fit parameters are only supported when fit_params is a dict or None."
            )

    from hypermri.analysis.dce.execution import _fit_dce_maps
    from hypermri.analysis.dce.model_selection import _find_dce_model

    dce_model = _find_dce_model(model)
    if dce_model is not None:
        return _fit_dce_maps(data, dce_model[0], fit_params, return_params_info)
    from hypermri.analysis.quantitative.fitting import _find_quantitative_model

    quantitative_model = _find_quantitative_model(model)
    if quantitative_model is not None:
        return quantitative_model[1][1](data, fit_params, return_params_info)
    if not isinstance(fit_params, xr.Dataset):
        fit_params = prepare_fit_params(data, model, fit_params)

    spec = get_registry().get(model)
    if spec.fit is None:
        raise ValueError(f"Model '{model}' does not provide a fit entry point.")
    return spec.fit(data, fit_params, return_params_info)


def reconstruct_fit(
    result_ds: xr.Dataset,
    voxel_coords: dict | None = None,
    **kwargs,
) -> xr.DataArray:
    """Reconstruct fitted signals through the model's scientific reconstructor.

    Parameters
    ----------
    result_ds : xarray.Dataset
        Fitted model parameters with model_name metadata and the sampling/AIF inputs
        required by its model. DCE sampling uses seconds and concentrations mM.
    voxel_coords : dict or None, optional
        Positional spatial selectors passed to Dataset.isel; None selects all voxels.
    **kwargs
        Existing accessor reconstruction options, including model-specific sampling,
        Tofts signal/concentration space and supported legacy aliases. The accessor
        translates known values to the selected scientific function.

    Returns
    -------
    xarray.DataArray
        Eager reconstructed samples with model-specific spatial/dynamic dimensions,
        coordinates, units and calculation precision. DCE concentration predictions
        are float64 mM; Tofts signal predictions use the acquired signal scale.
        FID reconstruction retains complex metabolite components. Input data and
        fitted parameters are preserved; no reconstruction is cached.

    Raises
    ------
    KeyError
        If an unknown model or required fitted parameter is missing.
    ValueError
        If the model's required sampling, AIF or conversion inputs are unavailable.
    NotImplementedError
        If the selected registered model has no reconstruction implementation.

    Examples
    --------
    >>> prediction = reconstruct_fit(fitted, voxel_coords={"read_idx": 0})  # doctest: +SKIP
    """
    from hypermri.analysis.dce.model_selection import _find_dce_result_model

    if _find_dce_result_model(result_ds.attrs.get("model_name")) is None:
        get_registry().resolve_result_model(result_ds)
    return result_ds.hp.reconstruct(voxel_coords=voxel_coords, **kwargs)
