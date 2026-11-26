"""Generic fitting import surface."""

from __future__ import annotations

from . import accessors as _accessors  # noqa: F401
from .api import fit, prepare_fit_params, reconstruct_fit
from .fit_maps import fit_quantitative_maps
from .spectroscopy import (
    add_metab_to_fit_params,
    fid_basefunc,
    fid_model,
    fit_best_of_starts,
    fit_data_pseudo_inv,
    initialise_fit_params_from_results,
    phase_fit_results,
    reconstruct_fitted_signal,
    refit_gauss_width_from_neighbours,
    remove_metab_from_fit_params,
    solve_for_complex_amplitudes,
)
from .spectroscopy import (
    def_fit_params as def_fit_params_spectroscopy,
)

# Preserve legacy naming where both modules exported def_fit_params
# In the dynamic version, the last one imported wins.
def_fit_params = def_fit_params_spectroscopy

__all__ = [
    "fit",
    "prepare_fit_params",
    "reconstruct_fit",
    "def_fit_params",
    "fit_quantitative_maps",
    "add_metab_to_fit_params",
    "fid_basefunc",
    "fid_model",
    "fit_best_of_starts",
    "fit_data_pseudo_inv",
    "initialise_fit_params_from_results",
    "phase_fit_results",
    "reconstruct_fitted_signal",
    "refit_gauss_width_from_neighbours",
    "remove_metab_from_fit_params",
    "solve_for_complex_amplitudes",
]
