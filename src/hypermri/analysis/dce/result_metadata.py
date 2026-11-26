"""Attach existing DCE descriptions and labelled reconstruction inputs.

Time arrays use seconds and plasma concentrations mM, calculated in float64.
Helpers update caller-owned fit datasets eagerly, retaining acquisition labels
when lengths match and the existing positional fallback otherwise. They do not
fit, change scientific options, mutate acquired input data or cache results.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.result_metadata import set_reconstruction_variable


def _set_var_attrs(
    dataset: xr.Dataset,
    name: str,
    *,
    long_name: str,
    description: str,
    units: str,
    model: str,
) -> None:
    """Update scientific descriptions of existing variables in place."""
    if name in dataset:
        dataset[name].attrs.update(
            {
                "long_name": long_name,
                "description": description,
                "units": units,
                "model": model,
            }
        )


def _attach_dce_reconstruction_arrays(
    output_dataset: xr.Dataset,
    fit_params: dict,
    dynamic_dim_name: str,
    data: xr.DataArray,
    *,
    aif_times_key: str,
) -> None:
    """Retain the established Patlak/Tofts/Sourbron reconstruction schema."""
    dynamic_coord = data.coords.get(dynamic_dim_name)
    time_points_s = fit_params.get("time_points_s")
    if time_points_s is not None:
        time_points_s = np.asarray(time_points_s, dtype=np.float64).squeeze()
        if (
            dynamic_coord is not None
            and getattr(dynamic_coord, "ndim", 1) == 1
            and len(dynamic_coord) == len(time_points_s)
        ):
            time_coords = {dynamic_dim_name: dynamic_coord}
        else:
            time_coords = {dynamic_dim_name: np.arange(len(time_points_s), dtype=int)}
        set_reconstruction_variable(
            output_dataset,
            "time_points_s",
            time_points_s,
            dims=[dynamic_dim_name],
            coords=time_coords,
            attrs={"units": "s"},
        )
    aif_signal = fit_params.get("aif_signal")
    aif_times = fit_params.get(aif_times_key, fit_params.get("time_points_s"))
    if aif_signal is not None:
        aif_signal = np.asarray(aif_signal, dtype=np.float64).squeeze()
        aif_idx = np.arange(len(aif_signal), dtype=int)
        set_reconstruction_variable(
            output_dataset,
            "aif_signal",
            aif_signal,
            dims=["aif_idx"],
            coords={"aif_idx": aif_idx},
            attrs={"units": "mM"},
        )
        if aif_times is not None:
            aif_times = np.asarray(aif_times, dtype=np.float64).squeeze()
            set_reconstruction_variable(
                output_dataset,
                "aif_time_s",
                aif_times,
                dims=["aif_idx"],
                coords={"aif_idx": aif_idx},
                attrs={"units": "s"},
            )
