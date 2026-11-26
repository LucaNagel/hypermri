from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

from ._shared import (
    build_motion_outputs,
    frame_geometry,
    prepare_motion_reference_volume,
    prepare_motion_registration_series,
    reshape_frame_for_motion,
    restore_frame_from_motion,
    validate_motion_input,
)


def _apply_shift_to_volume(
    volume: np.ndarray,
    shift_vector: np.ndarray,
    *,
    interpolation_order: int,
    mode: str,
    cval: float,
) -> np.ndarray:
    shift_vector = tuple(float(value) for value in shift_vector)
    if np.iscomplexobj(volume):
        real = ndi_shift(
            np.asarray(np.real(volume), dtype=np.float64),
            shift=shift_vector,
            order=interpolation_order,
            mode=mode,
            cval=cval,
            prefilter=interpolation_order > 1,
        )
        imag = ndi_shift(
            np.asarray(np.imag(volume), dtype=np.float64),
            shift=shift_vector,
            order=interpolation_order,
            mode=mode,
            cval=cval,
            prefilter=interpolation_order > 1,
        )
        return real + 1j * imag

    return ndi_shift(
        np.asarray(volume, dtype=np.float64),
        shift=shift_vector,
        order=interpolation_order,
        mode=mode,
        cval=cval,
        prefilter=interpolation_order > 1,
    )


def skimage_dce_motion_correction(
    data: xr.DataArray,
    *,
    time_dim: str = "repetition_idx",
    spatial_dims: tuple[str, ...] | list[str] | None = None,
    reference_index: int = 0,
    reference_volume: xr.DataArray | np.ndarray | None = None,
    upsample_factor: int = 10,
    normalization: str | None = None,
    interpolation_order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    return_metadata: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, dict[str, object]]:
    """
    Motion-correct a DCE series using scikit-image phase cross-correlation.

    This backend estimates one translation vector per frame relative to a fixed
    reference image and applies that translation to the full DCE frame. It is
    lightweight and well suited for small rigid drifts, but it does not model
    rotation or nonrigid deformation.

    Parameters
    ----------
    data
        Dynamic DCE data stored as an ``xarray.DataArray``.
    time_dim
        Name of the temporal dimension to register frame by frame.
    spatial_dims
        Spatial dimensions used for registration. When omitted, they are inferred
        from ``read_idx``, ``phase_idx``, and optionally ``slice_idx``.
    reference_index
        Reference frame used when ``reference_volume`` is not supplied.
    reference_volume
        Optional external fixed image used for all registrations.
    upsample_factor
        Subpixel refinement factor for ``phase_cross_correlation``.
    normalization
        Optional normalization mode forwarded to ``phase_cross_correlation``.
    interpolation_order
        Spline interpolation order used by ``scipy.ndimage.shift``.
    mode, cval
        Boundary handling settings used while resampling corrected frames.
    return_metadata
        When ``True``, also return backend-specific registration metadata.
    """
    motion_spatial_dims, n_frames = validate_motion_input(
        data,
        time_dim=time_dim,
        spatial_dims=spatial_dims,
        reference_index=reference_index,
    )
    registration_series = prepare_motion_registration_series(
        data,
        time_dim=time_dim,
        spatial_dims=motion_spatial_dims,
    )

    if reference_volume is None:
        static = registration_series.isel({time_dim: reference_index}, drop=True).values
        reference_source = f"{time_dim}[{reference_index}]"
    else:
        static = prepare_motion_reference_volume(
            reference_volume,
            time_dim=time_dim,
            spatial_dims=motion_spatial_dims,
        )
        reference_source = "reference_volume"

    static = np.nan_to_num(np.asarray(static, dtype=np.float64), copy=False)
    frame_values: list[np.ndarray] = []
    estimated_shifts: list[np.ndarray] = []
    registration_errors: list[float] = []
    phasediff_values: list[float] = []
    frame_dim_order, frame_extra_dims = frame_geometry(
        data,
        time_dim=time_dim,
        spatial_dims=motion_spatial_dims,
    )

    for frame_index in range(n_frames):
        if reference_volume is None and frame_index == reference_index:
            shift_vector = np.zeros(static.ndim, dtype=np.float64)
            error = 0.0
            phasediff = 0.0
        else:
            moving = registration_series.isel({time_dim: frame_index}, drop=True).values
            shift_vector, error, phasediff = phase_cross_correlation(
                reference_image=static,
                moving_image=np.nan_to_num(
                    np.asarray(moving, dtype=np.float64), copy=False
                ),
                upsample_factor=int(upsample_factor),
                normalization=normalization,
            )
            shift_vector = np.asarray(shift_vector, dtype=np.float64)
            error = float(error)
            phasediff = float(phasediff)

        estimated_shifts.append(shift_vector)
        registration_errors.append(error)
        phasediff_values.append(phasediff)

        frame = data.isel({time_dim: frame_index}, drop=True)
        reshaped, working_dim_order, frame_shape = reshape_frame_for_motion(
            frame,
            spatial_dims=motion_spatial_dims,
            frame_extra_dims=frame_extra_dims,
        )
        corrected = np.empty_like(
            reshaped,
            dtype=np.result_type(frame.values.dtype, np.float64),
        )

        for volume_index in range(reshaped.shape[0]):
            corrected[volume_index] = _apply_shift_to_volume(
                reshaped[volume_index],
                shift_vector,
                interpolation_order=int(interpolation_order),
                mode=mode,
                cval=float(cval),
            )

        corrected_da = restore_frame_from_motion(
            corrected,
            input_frame=frame,
            working_dim_order=working_dim_order,
            frame_shape=frame_shape,
            frame_dim_order=frame_dim_order,
        )
        frame_values.append(corrected_da.values)

    metadata = {
        "method": "skimage",
        "time_dim": time_dim,
        "spatial_dims": motion_spatial_dims,
        "reference_index": int(reference_index),
        "reference_source": reference_source,
        "upsample_factor": int(upsample_factor),
        "normalization": normalization,
        "interpolation_order": int(interpolation_order),
        "mode": mode,
        "cval": float(cval),
        "estimated_shifts": np.stack(estimated_shifts, axis=0),
        "registration_error": np.asarray(registration_errors, dtype=np.float64),
        "phasediff": np.asarray(phasediff_values, dtype=np.float64),
    }
    return build_motion_outputs(
        data,
        corrected_frames=frame_values,
        metadata=metadata,
        time_dim=time_dim,
        return_metadata=return_metadata,
    )
