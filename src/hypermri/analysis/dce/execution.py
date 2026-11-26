"""Execute prepared DCE concentration maps without generic facade dispatch.

Existing model preparers own scientific defaults and validation. The shared
voxel executor retains float64 calculation, model-specific status/count dtypes,
labelled dimensions, coordinates and reconstruction metadata. Functions are
eager, deterministic for fixed options/backends and create no result cache.
"""

from __future__ import annotations

import xarray as xr

from hypermri.analysis.fitting.voxelwise import VoxelwiseExecutor


def _prepare_dce_params(data, model_id, fit_params=None, return_params_info=False):
    """Delegate once to the model's established parameter preparer."""
    from .model_selection import _DCE_MODELS

    prepare = getattr(_DCE_MODELS[model_id].module, "_prepare_fit_params", None)
    if prepare is None:
        raise ValueError(f"Model '{model_id}' does not provide parameter preparation.")
    return prepare(data, fit_params, return_params_info)


def _fit_dce_maps(data, model_id, fit_params=None, return_params_info=False):
    """Prepare once and run the existing model callback through one voxel loop."""
    if not isinstance(fit_params, xr.Dataset):
        fit_params = _prepare_dce_params(data, model_id, fit_params)
    if return_params_info:
        return None
    return _fit_prepared_dce_maps(data, model_id, dict(fit_params.attrs))


def _fit_prepared_dce_maps(data, model_id, params):
    """Run an already-prepared legacy or domain parameter mapping once."""
    from .model_selection import _DCE_MODELS

    model = _DCE_MODELS[model_id].module
    return VoxelwiseExecutor().execute(
        data=data,
        model_id=model_id,
        fit_params=params,
        single_voxel_fit_func=model._fit_voxel,
        output_param_names=model._output_param_names(params),
        dynamic_dim=params.get("dynamic_dim", "repetition_idx"),
        model_name=model.RESULT_MODEL_NAME,
    )
