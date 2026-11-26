from __future__ import annotations

import numpy as np
import xarray as xr

from ._shared import (
    build_motion_outputs,
    frame_geometry,
    prepare_motion_reference_volume,
    prepare_motion_registration_series,
    reshape_frame_for_motion,
    restore_frame_from_motion,
    validate_motion_input,
)

try:
    import SimpleITK
except ImportError:
    SimpleITK = None


def _require_simpleitk():
    if SimpleITK is None:
        raise ImportError(
            "SimpleITK is required for correction_method='simpleitk'. "
            "Install SimpleITK or choose a different correction_method."
        )
    return SimpleITK


def _simpleitk_interpolator(interpolation: str):
    sitk_mod = _require_simpleitk()
    mapping = {
        "linear": sitk_mod.sitkLinear,
        "nearest": sitk_mod.sitkNearestNeighbor,
        "nearestneighbor": sitk_mod.sitkNearestNeighbor,
        "bspline": sitk_mod.sitkBSpline,
    }
    try:
        return mapping[str(interpolation).lower()]
    except KeyError as exc:
        raise ValueError(
            "simpleitk interpolation must be one of 'linear', 'nearest', or 'bspline'."
        ) from exc


def _create_simpleitk_transform(
    fixed_image,
    *,
    transform_type: str,
    bspline_grid_size: int,
):
    sitk_mod = _require_simpleitk()
    dim = fixed_image.GetDimension()
    transform_type = str(transform_type).lower()

    if transform_type == "translation":
        return sitk_mod.TranslationTransform(dim)
    if transform_type == "rigid":
        return sitk_mod.Euler2DTransform() if dim == 2 else sitk_mod.Euler3DTransform()
    if transform_type == "affine":
        return sitk_mod.AffineTransform(dim)
    if transform_type == "bspline":
        mesh_size = [int(bspline_grid_size)] * dim
        return sitk_mod.BSplineTransformInitializer(
            image1=fixed_image,
            transformDomainMeshSize=mesh_size,
            order=3,
        )

    raise ValueError(
        "simpleitk_transform_type must be one of 'translation', 'rigid', "
        "'affine', or 'bspline'."
    )


def _estimate_simpleitk_transform(
    static: np.ndarray,
    moving: np.ndarray,
    *,
    transform_type: str,
    metric: str,
    shrink_factors: tuple[int, ...],
    smoothing_sigmas: tuple[float, ...],
    learning_rate: float,
    min_step: float,
    number_of_iterations: int,
    gradient_magnitude_tolerance: float,
    sampling_percentage: float | None,
    bspline_grid_size: int,
):
    sitk_mod = _require_simpleitk()
    fixed_image = sitk_mod.GetImageFromArray(
        np.asarray(static, dtype=np.float32),
        isVector=False,
    )
    moving_image = sitk_mod.GetImageFromArray(
        np.asarray(moving, dtype=np.float32),
        isVector=False,
    )

    registration_method = sitk_mod.ImageRegistrationMethod()
    metric_name = str(metric).lower()
    if metric_name in {"mi", "mattes", "mattesmi"}:
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric_name in {"meansquares", "ms"}:
        registration_method.SetMetricAsMeanSquares()
    else:
        raise ValueError("simpleitk metric must be 'mattes'/'mi' or 'meansquares'.")

    if sampling_percentage is not None:
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(float(sampling_percentage))

    registration_method.SetInterpolator(sitk_mod.sitkLinear)
    registration_method.SetShrinkFactorsPerLevel(
        shrinkFactors=[int(value) for value in shrink_factors]
    )
    registration_method.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[float(value) for value in smoothing_sigmas]
    )
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=float(min_step),
        numberOfIterations=int(number_of_iterations),
        gradientMagnitudeTolerance=float(gradient_magnitude_tolerance),
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    initial_transform = _create_simpleitk_transform(
        fixed_image,
        transform_type=transform_type,
        bspline_grid_size=bspline_grid_size,
    )
    if str(transform_type).lower() in {"rigid", "affine"}:
        initial_transform = sitk_mod.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            initial_transform,
            sitk_mod.CenteredTransformInitializerFilter.GEOMETRY,
        )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    return final_transform, fixed_image


def _apply_simpleitk_transform_to_volume(
    transform,
    *,
    fixed_image,
    volume: np.ndarray,
    interpolation: str,
    default_value: float,
) -> np.ndarray:
    sitk_mod = _require_simpleitk()

    def _resample_component(component: np.ndarray) -> np.ndarray:
        moving_image = sitk_mod.GetImageFromArray(
            np.asarray(component, dtype=np.float32),
            isVector=False,
        )
        moving_image.CopyInformation(fixed_image)
        resampled = sitk_mod.Resample(
            moving_image,
            fixed_image,
            transform,
            _simpleitk_interpolator(interpolation),
            float(default_value),
            moving_image.GetPixelID(),
        )
        return sitk_mod.GetArrayFromImage(resampled).astype(np.float64, copy=False)

    if np.iscomplexobj(volume):
        real = _resample_component(np.real(volume))
        imag = _resample_component(np.imag(volume))
        return real + 1j * imag

    return _resample_component(np.asarray(volume, dtype=np.float64))


def simpleitk_dce_motion_correction(
    data: xr.DataArray,
    *,
    time_dim: str = "repetition_idx",
    spatial_dims: tuple[str, ...] | list[str] | None = None,
    reference_index: int = 0,
    reference_volume: xr.DataArray | np.ndarray | None = None,
    simpleitk_transform_type: str = "rigid",
    metric: str = "mattes",
    shrink_factors: tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
    learning_rate: float = 2.0,
    min_step: float = 1e-4,
    number_of_iterations: int = 200,
    gradient_magnitude_tolerance: float = 1e-8,
    sampling_percentage: float | None = 0.2,
    bspline_grid_size: int = 8,
    interpolation: str = "linear",
    default_value: float = 0.0,
    return_metadata: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, dict[str, object]]:
    """
    Motion-correct a DCE series with frame-wise SimpleITK registration.

    The registration is performed on a reduced magnitude image for each time
    point and the resulting transform is then applied to the full frame. This
    backend supports translation, rigid, affine, and BSpline transforms via
    SimpleITK's registration framework.

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
    simpleitk_transform_type
        Transform family to optimize. Supported values are ``"translation"``,
        ``"rigid"``, ``"affine"``, and ``"bspline"``.
    metric
        Registration metric. Supported values are ``"mattes"``/``"mi"`` and
        ``"meansquares"``.
    shrink_factors, smoothing_sigmas
        Multi-resolution pyramid settings used during registration.
    learning_rate, min_step, number_of_iterations, gradient_magnitude_tolerance
        Optimizer settings for ``RegularStepGradientDescent``.
    sampling_percentage
        Optional fraction of voxels sampled randomly for the metric.
    bspline_grid_size
        Mesh size used when ``simpleitk_transform_type="bspline"``.
    interpolation, default_value
        Resampling settings for the corrected output.
    return_metadata
        When ``True``, also return backend-specific registration metadata.
    """
    _require_simpleitk()
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
    transform_parameters: list[np.ndarray] = []
    fixed_parameters: list[np.ndarray] = []
    frame_dim_order, frame_extra_dims = frame_geometry(
        data,
        time_dim=time_dim,
        spatial_dims=motion_spatial_dims,
    )

    for frame_index in range(n_frames):
        if reference_volume is None and frame_index == reference_index:
            fixed_image = _require_simpleitk().GetImageFromArray(
                np.asarray(static, dtype=np.float32),
                isVector=False,
            )
            transform = _create_simpleitk_transform(
                fixed_image,
                transform_type=simpleitk_transform_type,
                bspline_grid_size=bspline_grid_size,
            )
        else:
            moving = registration_series.isel({time_dim: frame_index}, drop=True).values
            transform, fixed_image = _estimate_simpleitk_transform(
                static=static,
                moving=np.nan_to_num(np.asarray(moving, dtype=np.float64), copy=False),
                transform_type=simpleitk_transform_type,
                metric=metric,
                shrink_factors=shrink_factors,
                smoothing_sigmas=smoothing_sigmas,
                learning_rate=learning_rate,
                min_step=min_step,
                number_of_iterations=number_of_iterations,
                gradient_magnitude_tolerance=gradient_magnitude_tolerance,
                sampling_percentage=sampling_percentage,
                bspline_grid_size=bspline_grid_size,
            )

        transform_parameters.append(
            np.asarray(transform.GetParameters(), dtype=np.float64)
        )
        fixed_parameters.append(
            np.asarray(transform.GetFixedParameters(), dtype=np.float64)
        )

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
            corrected[volume_index] = _apply_simpleitk_transform_to_volume(
                transform,
                fixed_image=fixed_image,
                volume=reshaped[volume_index],
                interpolation=interpolation,
                default_value=default_value,
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
        "method": "simpleitk",
        "time_dim": time_dim,
        "spatial_dims": motion_spatial_dims,
        "reference_index": int(reference_index),
        "reference_source": reference_source,
        "simpleitk_transform_type": str(simpleitk_transform_type).lower(),
        "metric": str(metric).lower(),
        "interpolation": str(interpolation).lower(),
        "transform_parameters": transform_parameters,
        "fixed_parameters": fixed_parameters,
    }
    return build_motion_outputs(
        data,
        corrected_frames=frame_values,
        metadata=metadata,
        time_dim=time_dim,
        return_metadata=return_metadata,
    )
