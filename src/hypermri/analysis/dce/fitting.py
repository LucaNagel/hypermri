"""Canonical public import surface for DCE fitting workflows."""

from .aif import prepare_aif, validate_aif
from .fitting_support import (
    compute_fit_quality_metrics,
    convert_signal_to_concentration_map,
    fit_empirical_log_logistic_model,
    fit_extended_tofts_model,
    fit_patlak_model,
    fit_sourbron_model,
    fit_standard_tofts_model,
    fit_tofts_model,
    fit_two_c_irf_model,
    select_fit_window,
    two_c_irf_model,
)
from .pipeline import run_dce_pipeline  # , run_dce_pipeline_legacy_signal_fit
from .preprocessing import (
    align_t1_to_dce_grid,
    combine_multichannel_data,
    compute_s0_map,
    prepare_dce_metadata,
    standardize_t1_map_units,
)

__all__ = [
    "align_t1_to_dce_grid",
    "combine_multichannel_data",
    "compute_fit_quality_metrics",
    "compute_s0_map",
    "convert_signal_to_concentration_map",
    "fit_empirical_log_logistic_model",
    "fit_extended_tofts_model",
    "fit_patlak_model",
    "fit_two_c_irf_model",
    "fit_sourbron_model",
    "fit_standard_tofts_model",
    "fit_tofts_model",
    "prepare_aif",
    "prepare_dce_metadata",
    "two_c_irf_model",
    "run_dce_pipeline",
    # "run_dce_pipeline_legacy_signal_fit",
    "select_fit_window",
    "standardize_t1_map_units",
    "validate_aif",
]
