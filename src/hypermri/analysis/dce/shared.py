"""Shared helpers for DCE analysis workflows."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

from hypermri.analysis.anatomical.masks import align_mask_to_data_grid

from .fit_quality import _dataarray_isclose as _dataarray_isclose
from .result_metadata import _set_var_attrs as _set_var_attrs
from .sampling import (
    _coerce_numeric_1d as _coerce_numeric_1d,
)
from .sampling import (
    _to_numpy as _to_numpy,
)
from .sampling import (
    _validate_dce_time_axis as _validate_dce_time_axis,
)


def _spatial_template(data: xr.DataArray) -> xr.DataArray:
    if "repetition_idx" not in data.dims:
        raise ValueError("DCE data must contain a 'repetition_idx' dimension.")
    template = data.isel(repetition_idx=0, drop=True)
    if "channel_idx" in template.dims and template.sizes["channel_idx"] == 1:
        template = template.squeeze("channel_idx", drop=True)
    return template


def _spatial_dims(data: xr.DataArray) -> list[str]:
    return [dim for dim in data.dims if dim not in {"repetition_idx", "channel_idx"}]


def _align_mask_weights_to_template(
    mask,
    template: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    """Align boolean or fractional mask data to a spatial template."""
    return align_mask_to_data_grid(
        mask,
        target_data=template,
        source_data=template,
        spatial_dims=template.dims,
        name=name,
    )


def _mask_weight_sum(mask_da: xr.DataArray) -> float:
    return float(np.asarray(mask_da.values, dtype=np.float64).sum())


def _weighted_sum_over_mask(
    data: xr.DataArray,
    mask_weights: xr.DataArray,
    *,
    dims: tuple[str, ...] | list[str],
) -> xr.DataArray:
    return (data * mask_weights).sum(dim=dims, skipna=True)


def _weighted_mean_over_mask(
    data: xr.DataArray,
    mask_weights: xr.DataArray,
    *,
    dims: tuple[str, ...] | list[str],
) -> xr.DataArray:
    numerator = _weighted_sum_over_mask(data, mask_weights, dims=dims)
    valid_weights = mask_weights * xr.where(np.isfinite(data), 1.0, 0.0)
    denominator = valid_weights.sum(dim=dims, skipna=True)
    return numerator / denominator.where(denominator > 0)


def _weighted_values_from_mask(
    data: xr.DataArray,
    mask_weights: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data.values, dtype=np.float64)
    weights = np.asarray(mask_weights.values, dtype=np.float64)
    weights = np.clip(
        np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return values[valid], weights[valid]


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or weights.size == 0:
        return float("nan")

    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return float("nan")

    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * total
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def _merge_fit_params(
    fit_params_pk: dict | None = None,
    fit_params_tofts: dict | None = None,
    *,
    default_model_mode: str = "both",
) -> dict:
    def normalize_params(name: str, raw_params) -> dict:
        if raw_params is None:
            return {}
        if isinstance(raw_params, Mapping):
            return dict(raw_params)
        if (
            isinstance(raw_params, (list, tuple))
            and len(raw_params) == 1
            and isinstance(raw_params[0], Mapping)
        ):
            warnings.warn(
                f"{name} was passed as a single-element tuple containing a dict. "
                "This usually comes from a trailing comma after the dict literal. "
                "The inner dict was unwrapped automatically.",
                stacklevel=3,
            )
            return dict(raw_params[0])
        try:
            return dict(raw_params)
        except Exception as exc:
            raise TypeError(
                f"{name} must be a mapping of fit parameters, got {type(raw_params).__name__}."
            ) from exc

    params = {}
    params.update(normalize_params("fit_params_tofts", fit_params_tofts))
    params.update(normalize_params("fit_params_pk", fit_params_pk))

    if "fit_stop_index" in params and "fit_end_index" not in params:
        params["fit_end_index"] = params["fit_stop_index"]
    if "fit_stop_time_s" in params and "fit_end_time_s" not in params:
        params["fit_end_time_s"] = params["fit_stop_time_s"]

    params.setdefault("n_precontrast", 5)
    params.setdefault("model_mode", default_model_mode)
    params.setdefault("aif_threshold", 1e-6)
    params.setdefault("insufficient_signal_threshold", 1e-6)
    return params


def _apply_explicit_aif_options(
    fit_params: Mapping[str, object] | None = None,
    *,
    aif_convert_blood_to_plasma: bool | None = None,
    aif_hematocrit: float | None = None,
) -> dict:
    params = dict(fit_params or {})
    if aif_convert_blood_to_plasma is not None:
        params["aif_convert_blood_to_plasma"] = bool(aif_convert_blood_to_plasma)
    if aif_hematocrit is not None:
        params["aif_hematocrit"] = float(aif_hematocrit)
    return params


def _normalize_legacy_dce_request(
    fit_type: str,
    fit_params: dict | None = None,
) -> tuple[str, dict, str]:
    """Interpret legacy model_mode values that were used as top-level fit selectors."""
    normalized_fit_type = str(fit_type).lower()
    params = dict(fit_params or {})
    requested_mode = str(params.get("model_mode", "")).lower()

    from .model_selection import _DCE_MODELS

    for canonical, model in _DCE_MODELS.items():
        for alias, _fitter, mode, deprecated in model.request_aliases:
            if normalized_fit_type != alias:
                continue
            if mode in {"1comp", "2comp"}:
                params["model_mode"] = mode
            else:
                params.pop("model_mode", None)
            if deprecated:
                warnings.warn(
                    f"fit_type='{normalized_fit_type}' is deprecated because it is mathematically "
                    "equivalent to 'extended_tofts_plasma'. Routing to the extended Tofts model.",
                    stacklevel=3,
                )
                params["_deprecated_equivalent_fit_type"] = "extended_tofts_plasma"
            return canonical, params, mode
    model = _DCE_MODELS.get(normalized_fit_type)
    if model is not None and not getattr(
        model.module, "_KEEP_LEGACY_MODEL_MODE", False
    ):
        params.pop("model_mode", None)
        return normalized_fit_type, params, model.module._DEFAULT_MODEL_MODE

    inferred_fit_type = None
    if normalized_fit_type == "tofts":
        if requested_mode == "patlak":
            inferred_fit_type = "patlak"
    if inferred_fit_type is not None:
        warnings.warn(
            "run_dce_pipeline interpreted fit_params_tofts['model_mode'] as a legacy fit selector. "
            f"Proceeding with fit_type='{inferred_fit_type}'. "
            f"Use fit_type='{inferred_fit_type}' or run_renal_dce_pipeline(...) in new code.",
            stacklevel=3,
        )
        params.pop("model_mode", None)
        return inferred_fit_type, params, "1comp"

    if normalized_fit_type == "tofts" and requested_mode == "sourbron":
        raise ValueError(
            "fit_params_tofts['model_mode'] = 'sourbron' is no longer supported. "
            "Use fit_type='extended_tofts_plasma' or set model_mode='2comp'."
        )

    if normalized_fit_type == "patlak" and requested_mode == "patlak":
        params.pop("model_mode", None)

    default_model_mode = (
        model.module._DEFAULT_MODEL_MODE if model is not None else "1comp"
    )
    return normalized_fit_type, params, default_model_mode


def _build_fit_dispatch_metadata(
    *,
    requested_fit_type: str,
    requested_fit_params: Mapping | None,
    effective_fit_type: str,
    effective_fit_params: Mapping | None,
    default_model_mode: str,
) -> dict[str, str | None]:
    requested_fit_type = str(requested_fit_type).lower()
    requested_fit_params = dict(requested_fit_params or {})
    effective_fit_params = dict(effective_fit_params or {})

    requested_model_mode = requested_fit_params.get("model_mode")
    requested_model_mode = (
        str(requested_model_mode).lower() if requested_model_mode is not None else None
    )

    from .execution import _fit_dce_maps
    from .model_selection import _DCE_MODELS, _get_pipeline_fitter, _public_fit_name

    if effective_fit_type not in _DCE_MODELS:
        raise ValueError(
            "effective_fit_type must be 'tofts', 'patlak', 'empirical_log_logistic', 'sourbron', or '2c_irf'."
        )
    model = _DCE_MODELS[effective_fit_type].module
    select_mode = getattr(model, "_effective_model_mode", None)
    effective_model_mode = (
        select_mode(effective_fit_params, default_model_mode)
        if select_mode is not None
        else effective_fit_type
    )
    fitter = _get_pipeline_fitter(effective_fit_type, requested_fit_type)
    fit_function = _public_fit_name(fitter)
    fit_backend_fit_type = effective_fit_type
    fit_dispatch_summary = f"run_dce_pipeline -> {fitter.__name__} -> " + getattr(
        model, "_BACKEND_DESCRIPTION", "DCE concentration-map execution"
    )
    backend = (
        _fit_dce_maps
        if getattr(model, "_prepare_fit_params", None) is not None
        else fitter
    )
    metadata = {
        "requested_fit_type": requested_fit_type,
        "requested_model_mode": requested_model_mode,
        "effective_fit_type": effective_fit_type,
        "effective_model_mode": effective_model_mode,
        "fit_function": fit_function,
        "fit_backend_function": _public_fit_name(backend),
        "fit_backend_fit_type": fit_backend_fit_type,
        "fit_dispatch_summary": fit_dispatch_summary,
    }
    deprecated_equivalent = effective_fit_params.get("_deprecated_equivalent_fit_type")
    if deprecated_equivalent is not None:
        metadata["deprecated_fit_type_alias"] = requested_fit_type
        metadata["equivalent_fit_type"] = str(deprecated_equivalent)
        metadata["fit_type_equivalence_note"] = (
            f"'{requested_fit_type}' is treated as an alias of '{deprecated_equivalent}' "
            "because the implemented equations are mathematically equivalent."
        )
    return metadata


def _slice_series_from_index(series, start_index: int):
    if series is None or start_index <= 0:
        return series
    if isinstance(series, xr.DataArray):
        target_dim = (
            "repetition_idx" if "repetition_idx" in series.dims else series.dims[0]
        )
        return series.isel({target_dim: slice(start_index, None)})
    return series[start_index:]
