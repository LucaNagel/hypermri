"""Pure array and acquisition-time validation for DCE concentration models.

Numeric 1D inputs retain sample order and are calculated in float64. Strict
DCE time validation requires finite, increasing seconds. Coercion alone keeps
its historical shape-only validation, including non-finite concentration data.
These functions read no files, construct no UI and retain no mutable state.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr


def _to_numpy(values) -> np.ndarray:
    if isinstance(values, xr.DataArray):
        return np.asarray(values.values)
    return np.asarray(values)


def _coerce_numeric_1d(name: str, values) -> np.ndarray:
    arr = _to_numpy(values).astype(np.float64, copy=False).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeezing. Got shape {arr.shape}.")
    return arr


def _validate_dce_time_axis(name: str, values) -> np.ndarray:
    """Return a finite, strictly increasing DCE time axis."""
    time_points = _coerce_numeric_1d(name, values)
    if time_points.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(time_points)):
        raise ValueError(f"{name} must contain only finite values.")
    if time_points.size > 1 and np.any(np.diff(time_points) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return time_points


def _derive_time_points_from_concentration_map(
    concentration_map: xr.DataArray,
    fit_params: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Resolve the DCE time axis from fit params, coordinates, or metadata."""
    fit_params = dict(fit_params or {})

    time_points_s = fit_params.get("time_points_s")
    if time_points_s is not None:
        return _coerce_numeric_1d("time_points_s", time_points_s)

    for coord_name in ("time_s", "time", "acquisition_time_s"):
        if coord_name not in concentration_map.coords:
            continue
        coord = concentration_map.coords[coord_name]
        if "repetition_idx" in coord.dims or coord.ndim == 1:
            return _coerce_numeric_1d(coord_name, coord)

    if "repetition_idx" not in concentration_map.dims:
        raise ValueError(
            "concentration_map must contain 'repetition_idx' when time_points_s is not supplied."
        )

    n_repetitions = int(concentration_map.sizes["repetition_idx"])
    rep_coord = concentration_map.coords.get("repetition_idx")
    if rep_coord is None:
        return np.arange(n_repetitions, dtype=np.float64)

    axis_s = rep_coord.attrs.get("axis_s")
    if axis_s is not None:
        axis_values = _coerce_numeric_1d("repetition_idx.attrs['axis_s']", axis_s)
        if len(axis_values) != n_repetitions:
            raise ValueError(
                "repetition_idx.attrs['axis_s'] length does not match repetition_idx."
            )
        return axis_values

    for attr_name in ("frame_time_s", "time_points_s"):
        if attr_name not in concentration_map.attrs:
            continue
        attr_values = _coerce_numeric_1d(
            f"concentration_map.attrs['{attr_name}']",
            concentration_map.attrs[attr_name],
        )
        if len(attr_values) != n_repetitions:
            raise ValueError(
                f"concentration_map.attrs['{attr_name}'] length does not match repetition_idx."
            )
        return attr_values

    rep_values = np.asarray(rep_coord.values, dtype=np.float64).squeeze()
    if rep_values.ndim != 1:
        raise ValueError(
            "repetition_idx coordinate must be 1D when used as a time axis fallback."
        )

    return rep_values
