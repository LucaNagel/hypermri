"""Legacy fitting registry containing adapters to scientific domain workflows.

DCE and quantitative defaults are compatibility views of explicit domain maps;
named fits and the generic facade use domain execution directly. Registry
aliases, such as the result model names, are supported model names. Registry
construction reads no acquired data, fits no curves and creates no result cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import xarray as xr

PrepareParamsCallable = Callable[
    [xr.DataArray | None, dict | xr.Dataset | None, bool],
    xr.Dataset | None,
]
FitCallable = Callable[
    [xr.DataArray | None, dict | xr.Dataset | None, bool],
    xr.Dataset | None,
]
ReconstructCallable = Callable[..., xr.DataArray]


@dataclass(slots=True)
class FittingModelSpec:
    """Registry description for a fitting model."""

    model_id: str
    aliases: tuple[str, ...] = ()
    prepare_params: PrepareParamsCallable | None = None
    fit: FitCallable | None = None
    reconstruct: ReconstructCallable | None = None
    describe_params: Callable[..., object] | None = None


@dataclass(slots=True)
class FittingRegistry:
    """In-memory registry for fitting models and aliases."""

    _models: dict[str, FittingModelSpec] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)

    def register(self, spec: FittingModelSpec) -> FittingModelSpec:
        self._models[spec.model_id] = spec
        self._aliases[spec.model_id] = spec.model_id
        for alias in spec.aliases:
            self._aliases[alias] = spec.model_id
        return spec

    def get(self, model_name: str) -> FittingModelSpec:
        canonical_name = self._aliases.get(model_name)
        if canonical_name is None:
            raise KeyError(model_name)
        return self._models[canonical_name]

    def resolve_name(self, model_name: str) -> str:
        return self.get(model_name).model_id

    def resolve_result_model(self, result_ds: xr.Dataset) -> FittingModelSpec:
        model_name = result_ds.attrs.get("model_name")
        if not model_name:
            raise KeyError("model_name")
        return self.get(model_name)

    def list_models(self) -> tuple[str, ...]:
        return tuple(self._models)


_REGISTRY = FittingRegistry()
_DEFAULTS_REGISTERED = False


def _normalize_quantitative_fit_params(
    fit_params: dict | xr.Dataset | None,
) -> dict | None:
    if fit_params is None:
        return None
    if isinstance(fit_params, xr.Dataset):
        return dict(fit_params.attrs)
    return fit_params


def register_default_models() -> None:
    """Populate the legacy registry with adapters for existing fitting models.

    Returns
    -------
    None
        The shared registry is populated on the first call; later calls are
        deterministic no-ops. DCE and quantitative fit adapters delegate to their
        domain dictionaries and contain no numerical implementation. Each
        model is also registered under its documented aliases.
        This imports model definitions without fitting or reading acquired data.
    """
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    from hypermri.analysis.dce.model_selection import _DCE_MODELS
    from hypermri.analysis.quantitative.fitting import _QUANTITATIVE_MODELS

    from . import spectroscopy
    from .models.quantitative import adc, b1, t1, t2

    quantitative_modules = {
        "t1": _QUANTITATIVE_MODELS["t1"][0],
        "adc": _QUANTITATIVE_MODELS["adc"][0],
        **{
            key: model.module
            for key, model in _DCE_MODELS.items()
            if getattr(model.module, "_prepare_fit_params", None) is not None
        },
        "b1": _QUANTITATIVE_MODELS["b1"][0],
        "ideal": _QUANTITATIVE_MODELS["ideal"][0],
        "t2": _QUANTITATIVE_MODELS["t2"][0],
    }
    reconstructors = {
        "t1": t1.inversion_recovery_model,
        "adc": adc.monoexponential_adc_model,
        **{
            key: model.module._reconstruct
            for key, model in _DCE_MODELS.items()
            if getattr(model.module, "_prepare_fit_params", None) is not None
        },
        "b1": b1.b1_sinusoidal_model,
        "ideal": None,
        "t2": t2.monoexponential_t2_model,
    }

    for model_id, mod in quantitative_modules.items():
        prepare_params = getattr(mod, "_prepare_fit_params", None)
        if prepare_params is None:
            prepare_params = getattr(mod, f"prepare_{model_id}_fit_params_dataset")
        aliases = (mod.RESULT_MODEL_NAME,) if mod.RESULT_MODEL_NAME != model_id else ()

        _REGISTRY.register(
            FittingModelSpec(
                model_id=model_id,
                aliases=aliases,
                prepare_params=lambda data, fit_params=None, return_params_info=False, *, _prepare=prepare_params: (
                    _prepare(data, fit_params, return_params_info)
                ),
                fit=lambda data, fit_params=None, return_params_info=False, *, _model=model_id: (
                    None
                    if return_params_info
                    else _fit_registered_prepared(data, _model, fit_params)
                ),
                reconstruct=reconstructors[model_id],
                describe_params=(
                    mod._describe_params
                    if hasattr(mod, "_describe_params")
                    else getattr(mod, f"get_{model_id}_params_info")
                ),
            )
        )

    from .models.spectroscopy import fid_model as fid_pseudo_inv

    _REGISTRY.register(
        FittingModelSpec(
            model_id="fid_pseudo_inv",
            aliases=("FID", "fid", "fit_data_pseudo_inv"),
            prepare_params=lambda data, fit_params=None, return_params_info=False: (
                fit_params
                if isinstance(fit_params, xr.Dataset) and not return_params_info
                else spectroscopy.def_fit_params(
                    data=data,
                    fit_params=fit_params
                    if isinstance(fit_params, dict) or fit_params is None
                    else dict(fit_params.attrs),
                    return_params_info=return_params_info,
                )
            ),
            fit=lambda data, fit_params=None, return_params_info=False: (
                None
                if return_params_info
                else spectroscopy.fit_data_pseudo_inv(
                    data=data,
                    fit_params=fit_params
                    if isinstance(fit_params, xr.Dataset)
                    else spectroscopy.def_fit_params(
                        data=data,
                        fit_params=fit_params,
                    ),
                )
            ),
            reconstruct=fid_pseudo_inv.fid_model,
            describe_params=fid_pseudo_inv.get_fid_pseudo_inv_params_info,
        )
    )

    for model_id, model in _DCE_MODELS.items():
        if getattr(model.module, "_prepare_fit_params", None) is not None:
            continue
        _REGISTRY.register(
            FittingModelSpec(
                model_id=model_id,
                aliases=(),
                reconstruct=model.module._reconstruct,
            )
        )

    _DEFAULTS_REGISTERED = True


def _fit_registered_prepared(data, model_id, fit_params):
    """Translate prepared legacy attributes and delegate to the scientific owner."""
    from hypermri.analysis.dce.execution import _fit_prepared_dce_maps
    from hypermri.analysis.dce.model_selection import _DCE_MODELS
    from hypermri.analysis.quantitative.fitting import _QUANTITATIVE_MODELS

    params = _normalize_quantitative_fit_params(fit_params)
    if model_id in _DCE_MODELS:
        return _fit_prepared_dce_maps(data, model_id, params)
    return _QUANTITATIVE_MODELS[model_id][1](data, xr.Dataset(attrs=params))


def get_registry() -> FittingRegistry:
    """Return the singleton fitting registry with defaults loaded."""
    register_default_models()
    return _REGISTRY
