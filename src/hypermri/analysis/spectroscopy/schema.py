"""Canonical data layout for spectroscopy arrays.

This module contains the spectroscopy dimension contract.  These names are
domain invariants used throughout HyperMRI, not user-configurable defaults.
Keeping the aliases and normalization in one small module makes the accepted
array layout visible without having to inspect the spectroscopy algorithms.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

STANDARD_SPECTROSCOPY_DIMS = (
    "acquisition_idx",
    "slice_idx",
    "read_idx",
    "phase_idx",
    "repetition_idx",
    "channel_idx",
)

SPECTROSCOPY_DIM_ALIASES = {
    "t": "acquisition_idx",
    "time": "acquisition_idx",
    "fid": "acquisition_idx",
    "spectral": "acquisition_idx",
    "spectroscopic": "acquisition_idx",
    "echoes": "acquisition_idx",
    "sampling": "acquisition_idx",
    "spec": "acquisition_idx",
    "z": "slice_idx",
    "z-axis": "slice_idx",
    "x": "read_idx",
    "x-axis": "read_idx",
    "y": "phase_idx",
    "y-axis": "phase_idx",
    "r": "repetition_idx",
    "repetitions": "repetition_idx",
    "reps": "repetition_idx",
    "c": "channel_idx",
    "chan": "channel_idx",
    "ch": "channel_idx",
    "chans": "channel_idx",
    "channels": "channel_idx",
    "channel": "channel_idx",
}
SPECTROSCOPY_DIM_ALIASES.update(
    {dimension: dimension for dimension in STANDARD_SPECTROSCOPY_DIMS}
)


def _resolve_dim_position(dimension: str | int) -> int | None:
    if isinstance(dimension, int):
        return dimension
    if not isinstance(dimension, str):
        return None

    canonical_name = SPECTROSCOPY_DIM_ALIASES.get(dimension)
    if canonical_name is None:
        return None
    return STANDARD_SPECTROSCOPY_DIMS.index(canonical_name)


def _resolve_standard_dim_positions(
    provided_dims: tuple | list | None,
    ndim: int,
) -> list[int]:
    if ndim > len(STANDARD_SPECTROSCOPY_DIMS):
        raise ValueError(
            f"Expected at most {len(STANDARD_SPECTROSCOPY_DIMS)} dimensions, got {ndim}."
        )

    if provided_dims is None:
        positions = list(range(ndim))
    else:
        positions = [_resolve_dim_position(dimension) for dimension in provided_dims]

    if len(positions) != ndim:
        raise ValueError(
            "provided_dims must describe each input dimension exactly once."
        )
    if any(position is None for position in positions):
        raise ValueError(f"Could not resolve provided_dims={provided_dims!r}.")

    resolved_positions = [int(position) for position in positions]
    if any(
        position < 0 or position >= len(STANDARD_SPECTROSCOPY_DIMS)
        for position in resolved_positions
    ):
        raise ValueError(
            "provided_dims must contain indices between 0 and "
            f"{len(STANDARD_SPECTROSCOPY_DIMS) - 1}."
        )
    if len(set(resolved_positions)) != len(resolved_positions):
        raise ValueError("provided_dims must not contain duplicate dimension mappings.")

    return resolved_positions


def _canonicalize_spectroscopy_dim_name(dim_name: str) -> str:
    return SPECTROSCOPY_DIM_ALIASES.get(dim_name, dim_name)


def ensure_standard_spectroscopy_dataarray(
    data: xr.DataArray | np.ndarray | None = None,
    provided_dims: tuple | list | None = None,
    name: str | None = None,
) -> xr.DataArray:
    """Return data with the canonical spectroscopy dimensions and order.

    Missing dimensions are added as singleton axes. Existing aliases such as
    ``"spec"`` or ``"reps"`` are renamed to their canonical counterparts.
    """
    if data is None:
        raise ValueError("data cannot be None")

    if isinstance(data, xr.DataArray):
        data_array = data.copy(deep=False)
        rename_map: dict[str, str] = {}

        if provided_dims is not None:
            positions = _resolve_standard_dim_positions(provided_dims, data_array.ndim)
            for source_dim, position in zip(data_array.dims, positions, strict=False):
                target_dim = STANDARD_SPECTROSCOPY_DIMS[position]
                if source_dim != target_dim:
                    rename_map[source_dim] = target_dim
        else:
            for source_dim in data_array.dims:
                target_dim = _canonicalize_spectroscopy_dim_name(source_dim)
                if (
                    target_dim != source_dim
                    and target_dim not in data_array.dims
                    and target_dim not in rename_map.values()
                ):
                    rename_map[source_dim] = target_dim

        if rename_map:
            data_array = data_array.rename(rename_map)
    else:
        source_array = np.asarray(data)
        positions = _resolve_standard_dim_positions(provided_dims, source_array.ndim)
        source_dims = [f"input_dim_{index}" for index in range(source_array.ndim)]
        rename_map = {
            source_dim: STANDARD_SPECTROSCOPY_DIMS[position]
            for source_dim, position in zip(source_dims, positions, strict=False)
        }
        data_array = xr.DataArray(source_array, dims=source_dims, name=name)
        data_array = data_array.rename(rename_map)

    missing_dims = {
        dim_name: [0]
        for dim_name in STANDARD_SPECTROSCOPY_DIMS
        if dim_name not in data_array.dims
    }
    if missing_dims:
        data_array = data_array.expand_dims(missing_dims)

    return data_array.transpose(*STANDARD_SPECTROSCOPY_DIMS)


def _sampling_time_axis_from_data(
    data: xr.DataArray | None,
) -> np.ndarray | None:
    """Return a sampling axis that is aligned with ``data``.

    Reconstructed spectroscopy arrays can contain fewer acquisition points than
    the vendor handler after removal of leading FID samples.  Prefer coordinates
    and attributes carried by the array itself, but only when their length
    matches the displayed acquisition dimension.
    """
    if not isinstance(data, xr.DataArray) or "acquisition_idx" not in data.dims:
        return None

    expected_size = int(data.sizes["acquisition_idx"])
    candidates = []
    for coord_name in ("acquisition_time_s", "sampling_time_axis_s"):
        if coord_name in data.coords:
            candidates.append(data.coords[coord_name].values)
    for attr_name in ("sampling_time_axis_s", "time_axis_s"):
        candidates.append(data.attrs.get(attr_name))

    acquisition_coord = data.coords.get("acquisition_idx")
    if acquisition_coord is not None:
        unit = str(
            acquisition_coord.attrs.get(
                "unit", acquisition_coord.attrs.get("units", "")
            )
        ).casefold()
        if unit in {"s", "sec", "second", "seconds"}:
            candidates.append(acquisition_coord.values)

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            axis = np.asarray(candidate, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if axis.size == expected_size and np.all(np.isfinite(axis)):
            return axis
    return None


__all__ = [
    "SPECTROSCOPY_DIM_ALIASES",
    "STANDARD_SPECTROSCOPY_DIMS",
    "ensure_standard_spectroscopy_dataarray",
]
