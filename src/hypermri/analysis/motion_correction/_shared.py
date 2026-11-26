from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr


def infer_motion_spatial_dims(
    data: xr.DataArray,
    spatial_dims: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    if spatial_dims is None:
        inferred = tuple(
            dim for dim in ("read_idx", "phase_idx", "slice_idx") if dim in data.dims
        )
    else:
        inferred = tuple(spatial_dims)

    if len(inferred) not in {2, 3}:
        raise ValueError(
            "Motion correction requires either 2 or 3 spatial dimensions. "
            f"Got {inferred!r}."
        )

    missing = [dim for dim in inferred if dim not in data.dims]
    if missing:
        raise ValueError(
            "data is missing required spatial dimensions for motion correction: "
            f"{missing}."
        )

    return inferred


def prepare_motion_registration_series(
    data: xr.DataArray,
    *,
    time_dim: str,
    spatial_dims: tuple[str, ...],
) -> xr.DataArray:
    registration_data = data

    if (
        "channel_idx" in registration_data.dims
        and registration_data.sizes["channel_idx"] > 1
    ):
        registration_data = np.sqrt(
            (np.abs(registration_data) ** 2).sum(dim="channel_idx")
        )

    for dim in list(registration_data.dims):
        if dim in spatial_dims or dim == time_dim:
            continue
        if registration_data.sizes[dim] == 1:
            registration_data = registration_data.isel({dim: 0}, drop=True)
        else:
            registration_data = registration_data.mean(dim=dim)

    registration_data = np.abs(registration_data).astype(np.float64, copy=False)
    return registration_data.transpose(*spatial_dims, time_dim)


def prepare_motion_reference_volume(
    reference_volume: xr.DataArray | np.ndarray,
    *,
    time_dim: str,
    spatial_dims: tuple[str, ...],
) -> np.ndarray:
    if isinstance(reference_volume, xr.DataArray):
        ref = reference_volume
        if time_dim in ref.dims:
            ref = ref.isel({time_dim: 0}, drop=True)
        for dim in list(ref.dims):
            if dim in spatial_dims:
                continue
            if ref.sizes[dim] == 1:
                ref = ref.isel({dim: 0}, drop=True)
            else:
                ref = ref.mean(dim=dim)
        ref = np.abs(ref.transpose(*spatial_dims)).astype(np.float64, copy=False).values
    else:
        ref = np.asarray(reference_volume, dtype=np.float64)

    if ref.ndim != len(spatial_dims):
        raise ValueError(
            "reference_volume must match the number of spatial dimensions after reduction. "
            f"Expected {len(spatial_dims)} dimensions, got shape {ref.shape}."
        )
    return np.nan_to_num(ref, copy=False)


def validate_motion_input(
    data: xr.DataArray,
    *,
    time_dim: str,
    spatial_dims: tuple[str, ...] | list[str] | None = None,
    reference_index: int = 0,
) -> tuple[tuple[str, ...], int]:
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray.")
    if time_dim not in data.dims:
        raise ValueError(f"data must contain the time dimension '{time_dim}'.")
    if data.sizes[time_dim] == 0:
        raise ValueError(f"data.{time_dim} must contain at least one frame.")

    motion_spatial_dims = infer_motion_spatial_dims(data, spatial_dims)
    n_frames = int(data.sizes[time_dim])
    if reference_index < 0 or reference_index >= n_frames:
        raise IndexError(
            f"reference_index={reference_index} is out of bounds for {time_dim} with "
            f"{n_frames} frames."
        )
    return motion_spatial_dims, n_frames


def frame_geometry(
    data: xr.DataArray,
    *,
    time_dim: str,
    spatial_dims: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    frame_dim_order = [dim for dim in data.dims if dim != time_dim]
    frame_extra_dims = [dim for dim in frame_dim_order if dim not in spatial_dims]
    return frame_dim_order, frame_extra_dims


def reshape_frame_for_motion(
    frame: xr.DataArray,
    *,
    spatial_dims: tuple[str, ...],
    frame_extra_dims: list[str],
) -> tuple[np.ndarray, list[str], tuple[int, ...]]:
    working_dim_order = frame_extra_dims + list(spatial_dims)
    frame = frame.transpose(*working_dim_order)
    frame_shape = tuple(frame.sizes[dim] for dim in working_dim_order)
    reshaped = frame.values.reshape(
        (-1,) + tuple(frame.sizes[dim] for dim in spatial_dims)
    )
    return reshaped, working_dim_order, frame_shape


def restore_frame_from_motion(
    corrected: np.ndarray,
    *,
    input_frame: xr.DataArray,
    working_dim_order: list[str],
    frame_shape: tuple[int, ...],
    frame_dim_order: list[str],
) -> xr.DataArray:
    corrected = corrected.reshape(frame_shape)
    return xr.DataArray(
        corrected,
        dims=working_dim_order,
        coords={dim: input_frame.coords[dim] for dim in working_dim_order},
        attrs=dict(input_frame.attrs),
        name=input_frame.name,
    ).transpose(*frame_dim_order)


def build_motion_outputs(
    data: xr.DataArray,
    *,
    corrected_frames: list[np.ndarray],
    metadata: Mapping[str, object],
    time_dim: str,
    return_metadata: bool,
) -> xr.DataArray | tuple[xr.DataArray, dict[str, object]]:
    corrected_stack = np.stack(corrected_frames, axis=data.get_axis_num(time_dim))
    corrected_data = xr.DataArray(
        corrected_stack.astype(data.dtype, copy=False),
        dims=data.dims,
        coords=data.coords,
        attrs=dict(data.attrs),
        name=data.name,
    )
    corrected_data.attrs["motion_correction_parameters"] = dict(metadata)

    metadata_dict = dict(metadata)
    if return_metadata:
        return corrected_data, metadata_dict
    return corrected_data
