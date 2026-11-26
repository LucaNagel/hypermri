"""Explicit selection of DCE concentration workflows and model contracts.

One ordinary dictionary links stable model keys to existing model modules and
named dataset fit functions. Modules own numerical calculations, scientific
options, units, metadata, reconstruction translation and precision. Selection
is deterministic, performs no fitting or data reads and creates no registry or
result cache. Aliases retain the existing compatibility warnings and windows.

A model module supplies its result name, voxel callback, ordered outputs/dtypes,
reconstruction parameter translation, reconstructor, quality window/inputs and
output annotation. A legacy parameter preparer and metadata hooks are optional.
The named dataset function and these existing functions are the extension
contract; no base class, registration method or broad keyword forwarding is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from .fitting_support import (
    fit_empirical_log_logistic_model,
    fit_extended_tofts_model,
    fit_patlak_model,
    fit_sourbron_model,
    fit_standard_tofts_model,
    fit_tofts_model,
    fit_two_c_irf_model,
)
from .models import empirical_log_logistic, patlak, sourbron, tofts, two_c_irf


@dataclass(frozen=True)
class _DceModel:
    """An existing model module and its concentration-map entry point."""

    module: ModuleType
    fit_function: Callable
    requires_aif: bool = True
    request_aliases: tuple[tuple[str, Callable, str, bool], ...] = ()


_DCE_MODELS = {
    "patlak": _DceModel(patlak, fit_patlak_model),
    "tofts": _DceModel(
        tofts,
        fit_tofts_model,
        request_aliases=(
            ("standard_tofts", fit_standard_tofts_model, "1comp", False),
            ("standard_ktrans_ve", fit_standard_tofts_model, "1comp", False),
            ("extended_tofts", fit_extended_tofts_model, "2comp", False),
            ("extended_tofts_plasma", fit_extended_tofts_model, "2comp", False),
            ("extended_plasma_tofts", fit_extended_tofts_model, "2comp", False),
            ("kidney_two_compartment_outflow", fit_extended_tofts_model, "2comp", True),
            ("renal_two_compartment_outflow", fit_extended_tofts_model, "2comp", True),
            ("kidney_2comp_outflow", fit_extended_tofts_model, "2comp", True),
        ),
    ),
    "sourbron": _DceModel(sourbron, fit_sourbron_model),
    "2c_irf": _DceModel(two_c_irf, fit_two_c_irf_model),
    "empirical_log_logistic": _DceModel(
        empirical_log_logistic,
        fit_empirical_log_logistic_model,
        requires_aif=False,
        request_aliases=(
            (
                "log_logistic_empirical",
                fit_empirical_log_logistic_model,
                "empirical_log_logistic",
                False,
            ),
        ),
    ),
}


def _find_dce_model(name: str) -> tuple[str, _DceModel] | None:
    """Resolve canonical/legacy result names for the fitting facade."""
    for key, model in _DCE_MODELS.items():
        if name in (key, model.module.RESULT_MODEL_NAME):
            return key, model
    return None


def _find_dce_result_model(name: str) -> _DceModel | None:
    """Resolve only established result names for accessor reconstruction."""
    return next(
        (
            model
            for model in _DCE_MODELS.values()
            if name == model.module.RESULT_MODEL_NAME
        ),
        None,
    )


def _get_pipeline_fitter(fit_type: str, requested_type: str):
    """Choose the existing named workflow, including variant aliases."""
    model = _DCE_MODELS[fit_type]
    return next(
        (
            fitter
            for alias, fitter, _mode, _deprecated in model.request_aliases
            if alias == requested_type
        ),
        model.fit_function,
    )


def _public_fit_name(fitter: Callable) -> str:
    """Keep established public aliases; new model functions name their module."""
    owner = fitter.__module__
    if owner == "hypermri.analysis.dce.fitting_support":
        owner = "hypermri.analysis.dce.fitting"
    return owner + "." + fitter.__name__


def _fit_pipeline_model(
    concentration_map,
    *,
    fit_type,
    requested_type,
    aif_info,
    time_points_s,
    fit_params,
    use_multiprocessing,
):
    """Call the selected named concentration workflow once."""
    model = _DCE_MODELS[fit_type]
    fitter = _get_pipeline_fitter(fit_type, requested_type)
    if model.requires_aif:
        return fitter(
            concentration_map,
            aif_concentration=aif_info["aif_concentration"],
            aif_time_s=aif_info["aif_time_s"],
            fit_params=fit_params,
            use_multiprocessing=use_multiprocessing,
        )
    return fitter(
        concentration_map,
        fit_params={**fit_params, "time_points_s": time_points_s},
        use_multiprocessing=use_multiprocessing,
    )
