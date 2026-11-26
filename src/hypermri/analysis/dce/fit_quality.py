"""Pure labelled parameter-bound checks for DCE quality evaluation.

Bounds retain their model-specific units. Comparisons preserve coordinates,
input values and the existing absolute/relative tolerances; flags are boolean.
No optimizer, scientific defaults, rendering or persistent state is created.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _dataarray_isclose(
    data: xr.DataArray, value: float, *, atol: float, rtol: float = 1e-05
) -> xr.DataArray:
    """Preserve xarray dims/coords when checking fitted parameters against bounds."""
    return xr.apply_ufunc(
        np.isclose,
        data,
        float(value),
        kwargs={"atol": atol, "rtol": rtol},
        dask="allowed",
    )


def _parameters_at_bounds(
    fit_results: xr.Dataset,
    parameter_names: tuple[str, ...],
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
    *,
    atol: float,
) -> xr.DataArray:
    """Flag voxels where any named parameter lies at its lower or upper bound."""
    lower_bounds, upper_bounds = bounds
    hit_bound = xr.zeros_like(fit_results[parameter_names[0]], dtype=bool)
    for name, lower, upper in zip(
        parameter_names, lower_bounds, upper_bounds, strict=True
    ):
        hit_bound = (
            hit_bound
            | _dataarray_isclose(fit_results[name], float(lower), atol=atol)
            | _dataarray_isclose(fit_results[name], float(upper), atol=atol)
        )
    return hit_bound
