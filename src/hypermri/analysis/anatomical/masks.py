"""Mask and ROI utilities for anatomical analysis.

Labelled native-grid extraction and scanner-space projection share one spatial
weighting operation. Complex values retain their phase unless magnitude is
explicitly requested. These helpers do not reconstruct, Fourier transform or
plot signals; the caller selects the input domain. Inputs are never mutated.

Notes
-----
Masks represent dimensionless fractional voxel coverage in [0, 1]. Spatial
reductions preserve non-spatial dimensions, coordinates and signal units.
Native-grid extraction needs matching labels; projection uses scanner-space
coordinates in millimeters. Weighting uses float64 calculation precision.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np
import xarray as xr

from ...utils.coordinates.grids import assign_xyz_coordinates, has_xyz_coordinates
from ...utils.image.slices import get_overlapping_slices

__all__ = [
    "align_mask_to_data_grid",
    "apply_aligned_mask",
    "apply_projected_mask",
    "extract_values_with_aligned_mask",
    "extract_masked_data",
    "project_mask_to_data_grid",
]


_DEFAULT_NON_SPATIAL_DIMS = {
    "acquisition_idx",
    "repetition_idx",
    "time",
    "channel_idx",
    "inversion_idx",
    "echo_idx",
    "rfpower_idx",
    "b_value_idx",
}

_XYZ_COORD_NAMES = ("x", "y", "z")


def _select_dataarray(
    data: xr.DataArray | xr.Dataset,
    *,
    variable: str | None,
    argument_name: str,
) -> xr.DataArray:
    if isinstance(data, xr.DataArray):
        return data

    if isinstance(data, xr.Dataset):
        if variable is None:
            if len(data.data_vars) != 1:
                raise ValueError(
                    f"{argument_name} is a Dataset with multiple variables. "
                    "Pass variable=... to select one."
                )
            variable = next(iter(data.data_vars))
        if variable not in data:
            raise ValueError(f"{argument_name} does not contain variable {variable!r}.")
        return data[variable]

    raise TypeError(f"{argument_name} must be an xarray DataArray or Dataset.")


def _infer_spatial_dims(
    data: xr.DataArray,
    *,
    spatial_dims: Iterable[str] | None,
) -> tuple[str, ...]:
    if spatial_dims is not None:
        dims = tuple(spatial_dims)
    else:
        dims = tuple(dim for dim in data.dims if dim not in _DEFAULT_NON_SPATIAL_DIMS)

    missing = [dim for dim in dims if dim not in data.dims]
    if missing:
        raise ValueError(
            f"Spatial dimensions {missing} are not present in data dims {data.dims}."
        )
    if not dims:
        raise ValueError("Could not infer spatial dimensions.")
    return dims


def _spatial_template(
    data: xr.DataArray,
    *,
    spatial_dims: tuple[str, ...],
) -> xr.DataArray:
    selectors = {
        dim: 0 for dim in data.dims if dim not in spatial_dims and data.sizes[dim] > 0
    }
    template = data.isel(selectors, drop=True) if selectors else data
    return template.transpose(*spatial_dims)


def _coerce_mask_to_source_grid(
    mask: xr.DataArray | np.ndarray,
    *,
    source_template: xr.DataArray,
    spatial_dims: tuple[str, ...],
    name: str,
) -> xr.DataArray:
    if isinstance(mask, xr.DataArray):
        mask_da = mask.astype(np.float64)
        for dim in list(mask_da.dims):
            if dim not in spatial_dims:
                if mask_da.sizes[dim] == 1:
                    mask_da = mask_da.isel({dim: 0}, drop=True)
                else:
                    raise ValueError(
                        f"Mask {name!r} has unsupported non-spatial dimension {dim!r}."
                    )

        source_spatial = source_template.transpose(*spatial_dims)
        try:
            mask_da = mask_da.broadcast_like(source_spatial).transpose(*spatial_dims)
        except Exception as exc:
            raise ValueError(
                f"Mask {name!r} could not be broadcast to source spatial shape "
                f"{source_spatial.shape}."
            ) from exc
        if mask_da.shape != source_spatial.shape:
            raise ValueError(
                f"Mask {name!r} has shape {mask_da.shape}, "
                f"expected source spatial shape {source_spatial.shape}."
            )
        return xr.DataArray(
            mask_da.values,
            dims=source_spatial.dims,
            coords=source_spatial.coords,
            name=mask_da.name or name,
            attrs=mask_da.attrs,
        )

    mask_arr = np.asarray(mask, dtype=np.float64)
    if mask_arr.shape != source_template.shape:
        raise ValueError(
            f"Mask {name!r} has shape {mask_arr.shape}, "
            f"expected source spatial shape {source_template.shape}."
        )

    return xr.DataArray(
        mask_arr,
        dims=source_template.dims,
        coords=source_template.coords,
        name=name,
    ).transpose(*spatial_dims)


def _clip_mask(mask: xr.DataArray) -> xr.DataArray:
    clipped = xr.apply_ufunc(
        np.clip,
        xr.apply_ufunc(
            np.nan_to_num,
            mask.astype(np.float64),
            kwargs={"nan": 0.0, "posinf": 1.0, "neginf": 0.0},
            dask="allowed",
        ),
        0.0,
        1.0,
        dask="allowed",
    )
    clipped.name = mask.name
    clipped.attrs.update(mask.attrs)
    return clipped


def _select_dataarray_or_primary(
    data: xr.DataArray | xr.Dataset | Any,
    *,
    variable: str | None,
    argument_name: str,
) -> xr.DataArray:
    """Return a DataArray from xarray inputs or MRIExperiment-like objects."""
    if isinstance(data, xr.DataArray | xr.Dataset):
        return _select_dataarray(
            data,
            variable=variable,
            argument_name=argument_name,
        )

    for attr_name in ("primary_data", "data", "scanner_data"):
        value = getattr(data, attr_name, None)
        if isinstance(value, xr.DataArray):
            return value

    raise TypeError(
        f"{argument_name} must be an xarray DataArray, Dataset, or an object "
        "with primary_data, data or scanner_data."
    )


def _ensure_xyz_coordinates(
    data: xr.DataArray,
    *,
    scan: Any = None,
    argument_name: str,
) -> xr.DataArray:
    """Ensure an xarray image carries scanner-space x/y/z coordinates."""
    if has_xyz_coordinates(data):
        return data

    if scan is not None:
        data = assign_xyz_coordinates(data, scan=scan, dtype=np.float64)
        if has_xyz_coordinates(data):
            return data

    raise ValueError(
        f"{argument_name} does not carry x/y/z scanner coordinates. "
        "Pass source_obj=... or target_obj=... so they can be derived from "
        "the scan geometry."
    )


def _broadcast_xyz_coord(data: xr.DataArray, coord_name: str) -> np.ndarray:
    coord = data.coords[coord_name]
    try:
        coord = coord.broadcast_like(data).transpose(*data.dims)
    except Exception as exc:
        raise ValueError(
            f"Coordinate {coord_name!r} cannot be broadcast to dims {data.dims}."
        ) from exc
    return np.asarray(coord.values, dtype=np.float64)


def _xyz_stack(data: xr.DataArray) -> np.ndarray:
    return np.stack(
        [_broadcast_xyz_coord(data, coord_name) for coord_name in _XYZ_COORD_NAMES],
        axis=-1,
    )


def _coord_attr_float(data: xr.DataArray, dim: str, key: str) -> float | None:
    if dim not in data.coords:
        return None
    value = data.coords[dim].attrs.get(key)
    if value is None:
        return None
    try:
        return float(np.atleast_1d(value)[0])
    except Exception:
        return None


def _estimate_xyz_affine(data: xr.DataArray) -> dict[str, Any]:
    """Estimate affine voxel-index to scanner-space vectors from x/y/z coords."""
    xyz = _xyz_stack(data)
    origin_index = (0,) * data.ndim
    origin = np.asarray(xyz[origin_index], dtype=np.float64)

    vectors = np.zeros((data.ndim, 3), dtype=np.float64)
    dim_resolutions = np.full(data.ndim, np.nan, dtype=np.float64)
    active_axes: list[int] = []

    for axis, dim in enumerate(data.dims):
        if data.sizes[dim] > 1:
            indexer = [0] * data.ndim
            indexer[axis] = 1
            vector = np.asarray(xyz[tuple(indexer)], dtype=np.float64) - origin
            if np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 0:
                vectors[axis] = vector
                dim_resolutions[axis] = float(np.linalg.norm(vector))
                active_axes.append(axis)
                continue

        resolution = _coord_attr_float(data, dim, "resolution_mm")
        if resolution is not None and np.isfinite(resolution) and resolution > 0:
            dim_resolutions[axis] = resolution

    if not active_axes:
        raise ValueError("Cannot derive scanner-space voxel vectors from x/y/z coords.")

    return {
        "dims": tuple(data.dims),
        "shape": tuple(int(data.sizes[dim]) for dim in data.dims),
        "origin": origin,
        "vectors": vectors,
        "active_axes": tuple(active_axes),
        "dim_resolutions": dim_resolutions,
        "slice_dimension_missing": "slice_idx" not in data.dims,
        "slice_thickness_mm": data.attrs.get("slice_thickness_mm"),
    }


def _default_plane_tolerance_mm(source_geometry: dict[str, Any]) -> float:
    if (
        source_geometry["slice_dimension_missing"]
        and len(source_geometry["active_axes"]) == 2
    ):
        thickness = source_geometry["slice_thickness_mm"]
        try:
            values = np.asarray(thickness, dtype=float)
            if values.size == 1:
                value = float(values.reshape(-1)[0])
                if np.isfinite(value) and value > 0:
                    return value / 2.0
        except (TypeError, ValueError):
            pass
        raise ValueError(
            "Mask projection without slice_idx requires a positive scalar "
            "source slice_thickness_mm attribute or explicit plane_tolerance_mm."
        )
    inactive_resolutions = [
        float(resolution)
        for axis, resolution in enumerate(source_geometry["dim_resolutions"])
        if axis not in source_geometry["active_axes"] and np.isfinite(resolution)
    ]
    if inactive_resolutions:
        return max(inactive_resolutions) / 2.0
    return 1e-3


def _continuous_source_indices(
    points_xyz: np.ndarray,
    source_geometry: dict[str, Any],
    *,
    plane_tolerance_mm: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map scanner-space points to continuous source voxel indices."""
    active_axes = source_geometry["active_axes"]
    vectors = source_geometry["vectors"]
    active_vectors = np.stack([vectors[axis] for axis in active_axes], axis=1)

    delta = np.asarray(points_xyz, dtype=np.float64) - source_geometry["origin"]
    coeffs, *_ = np.linalg.lstsq(active_vectors, delta.T, rcond=None)
    reconstructed = (active_vectors @ coeffs).T
    residual = np.linalg.norm(delta - reconstructed, axis=1)

    indices = np.zeros((points_xyz.shape[0], len(source_geometry["dims"])))
    for coeff_axis, source_axis in enumerate(active_axes):
        indices[:, source_axis] = coeffs[coeff_axis]

    inside = np.all(np.isfinite(indices), axis=1)
    inside &= np.all(np.isfinite(points_xyz), axis=1)
    for source_axis in active_axes:
        size = source_geometry["shape"][source_axis]
        inside &= indices[:, source_axis] >= -0.5
        inside &= indices[:, source_axis] <= size - 0.5

    if len(active_axes) < 3:
        tolerance = (
            _default_plane_tolerance_mm(source_geometry)
            if plane_tolerance_mm is None
            else float(plane_tolerance_mm)
        )
        inside &= residual <= tolerance

    return indices, inside


def _normalise_coverage_samples(
    coverage_samples: int | Iterable[int],
    *,
    ndim: int,
) -> tuple[int, ...]:
    if np.isscalar(coverage_samples):
        count = int(coverage_samples)
        if count < 1:
            raise ValueError("coverage_samples must be >= 1.")
        return (count,) * ndim

    counts = tuple(int(value) for value in coverage_samples)
    if len(counts) != ndim:
        raise ValueError(
            f"coverage_samples must contain {ndim} values for spatial dims."
        )
    if any(count < 1 for count in counts):
        raise ValueError("coverage_samples values must be >= 1.")
    return counts


def _build_subvoxel_offsets(
    target_geometry: dict[str, Any],
    *,
    coverage_samples: int | Iterable[int],
) -> np.ndarray:
    counts = _normalise_coverage_samples(
        coverage_samples,
        ndim=len(target_geometry["dims"]),
    )
    per_axis_offsets = []
    for axis, count in enumerate(counts):
        vector = target_geometry["vectors"][axis]
        if count == 1 or not np.any(vector):
            per_axis_offsets.append(np.zeros((1, 3), dtype=np.float64))
            continue
        coefficients = (np.arange(count, dtype=np.float64) + 0.5) / count - 0.5
        per_axis_offsets.append(coefficients[:, None] * vector[None, :])

    offsets = np.zeros((1, 3), dtype=np.float64)
    for axis_offsets in per_axis_offsets:
        offsets = (offsets[:, None, :] + axis_offsets[None, :, :]).reshape(-1, 3)
    return offsets


def _interpolate_mask_values(
    mask_da: xr.DataArray,
    points_xyz: np.ndarray,
    *,
    source_geometry: dict[str, Any],
    method: str,
    plane_tolerance_mm: float | None,
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    method = str(method).lower()
    if method not in {"nearest", "linear"}:
        raise ValueError("method must be 'nearest' or 'linear'.")
    if method == "linear" and any(size < 2 for size in mask_da.shape):
        method = "nearest"

    axes = tuple(np.arange(size, dtype=np.float64) for size in mask_da.shape)
    interpolator = RegularGridInterpolator(
        axes,
        np.asarray(mask_da.values, dtype=np.float64),
        method=method,
        bounds_error=False,
        fill_value=0.0,
    )
    source_indices, inside = _continuous_source_indices(
        points_xyz,
        source_geometry,
        plane_tolerance_mm=plane_tolerance_mm,
    )

    values = np.zeros(points_xyz.shape[0], dtype=np.float64)
    if np.any(inside):
        values[inside] = interpolator(source_indices[inside])
    return np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)


def project_mask_to_data_grid(
    mask: xr.DataArray | np.ndarray,
    target_data: xr.DataArray | xr.Dataset | Any,
    *,
    source_data: xr.DataArray | xr.Dataset | Any | None = None,
    source_obj: Any = None,
    target_obj: Any = None,
    target_variable: str | None = None,
    source_variable: str | None = None,
    spatial_dims: Iterable[str] | None = None,
    method: str = "nearest",
    coverage_samples: int | Iterable[int] = 3,
    threshold: float | None = None,
    plane_tolerance_mm: float | None = None,
    batch_size: int = 250_000,
    name: str | None = None,
) -> xr.DataArray:
    """Project a source-grid ROI mask onto a target image grid using x/y/z coords.

    The returned mask lives on the target spatial grid and contains fractional
    coverage values in ``[0, 1]``. Fractional values are estimated by sampling
    each target voxel at ``coverage_samples`` points per spatial direction and
    averaging the source-mask values at those scanner-space positions.

    Parameters
    ----------
    mask : xr.DataArray or np.ndarray
        Mask values on the source spatial grid. Boolean, integer, and fractional
        masks are accepted. Non-finite values are replaced and all values are
        clipped to ``[0, 1]`` before projection. A NumPy mask requires
        ``source_data`` to define its dimensions and geometry.
    target_data : xr.DataArray, xr.Dataset, or MRIExperiment-like object
        Data whose spatial grid receives the projected mask. Experiment-like
        objects must expose their image data through ``primary_data``. For an
        xarray Dataset with multiple variables, specify ``target_variable``.
    source_data : xr.DataArray, xr.Dataset, or MRIExperiment-like object, optional
        Data defining the spatial grid on which ``mask`` was created. If omitted,
        ``mask`` itself must be an xarray DataArray and is used as the source-grid
        template. For a Dataset with multiple variables, specify
        ``source_variable``.
    source_obj : object, optional
        Scan object carrying the source acquisition geometry. It is used to
        derive scanner-space ``x``, ``y``, and ``z`` coordinates when
        ``source_data`` does not already contain them. This is normally the
        source ``MRIExperiment``.
    target_obj : object, optional
        Scan object carrying the target acquisition geometry. It is used to
        derive scanner-space ``x``, ``y``, and ``z`` coordinates when
        ``target_data`` does not already contain them. When omitted and
        ``target_data`` is an experiment-like object, that object is used.
    target_variable : str, optional
        Variable selected from ``target_data`` when it is an xarray Dataset.
        This is required if the Dataset contains more than one data variable.
    source_variable : str, optional
        Variable selected from ``source_data`` when it is an xarray Dataset.
        This is required if the Dataset contains more than one data variable.
    spatial_dims : iterable of str, optional
        Spatial dimension names and their desired order. The same names are
        used for source and target grids. If omitted, known acquisition, time,
        repetition, channel, inversion, echo, RF-power, and b-value dimensions
        are excluded and all remaining dimensions are treated as spatial.
    method : {"nearest", "linear"}, default "nearest"
        Interpolation method used to evaluate the source mask at each sampled
        scanner-space position. ``"nearest"`` preserves hard mask boundaries;
        ``"linear"`` interpolates fractional values. Linear interpolation falls
        back to nearest-neighbor interpolation if a source spatial dimension has
        fewer than two voxels.
    coverage_samples : int or iterable of int, default 3
        Number of regularly spaced subvoxel samples per target spatial
        dimension. A scalar applies the same count to every dimension; an
        iterable must contain one positive count per target spatial dimension.
        The total samples per target voxel are the product of these counts.
        Use ``1`` to evaluate only the target voxel center.
    threshold : float, optional
        If provided, convert fractional target coverage to a binary mask after
        projection: coverage greater than or equal to ``threshold`` becomes
        ``1`` and lower coverage becomes ``0``. If omitted, fractional coverage
        is retained.
    plane_tolerance_mm : float, optional
        Maximum scanner-space distance in millimeters from a lower-dimensional
        source mask plane. It is only used when the source geometry has fewer
        than three active spatial axes. By default, half the known inactive-axis
        voxel thickness is used. If the source's slice dimension was removed,
        half its positive scalar ``slice_thickness_mm`` attribute is used;
        without that thickness, an explicit tolerance is required. Explicit
        tolerances must be finite and nonnegative and take precedence. Other
        lower-dimensional legacy paths retain their ``1e-3`` mm fallback.
    batch_size : int, default 250000
        Maximum number of target voxel centers processed in one batch. Reducing
        this value lowers temporary memory use but may increase runtime. Must be
        at least one.
    name : str, optional
        Name assigned to the returned DataArray. Defaults to ``mask.name`` when
        available, otherwise ``"mask"``.

    Returns
    -------
    xr.DataArray
        Float32 mask with the target spatial dimensions and coordinates. Values
        lie in ``[0, 1]`` unless ``threshold`` makes them binary. Projection
        settings and source/target spatial dimensions are recorded in its
        attributes.

    Raises
    ------
    ValueError
        If source/target geometry or mask dimensions are invalid, the explicit
        plane tolerance is negative/nonfinite, or a source plane without
        ``slice_idx`` has neither known thickness nor an explicit tolerance.
    TypeError
        If source or target inputs cannot supply a labelled DataArray.

    Notes
    -----
    Projection is performed in scanner-space ``x/y/z`` coordinates, not by
    matching ``read_idx``, ``phase_idx``, or ``slice_idx`` values. Source and
    target may therefore differ in position, orientation, FOV, and resolution.
    Xarray selection retains array attributes: keep ``slice_thickness_mm`` when
    using ``isel(slice_idx=..., drop=True)``. Source data own the thickness even
    when a separately supplied mask has no attributes. Inputs are not mutated;
    projection is eager and deterministic, using float64 intermediate weights.
    """
    target_obj = target_data if target_obj is None else target_obj
    target = _select_dataarray_or_primary(
        target_data,
        variable=target_variable,
        argument_name="target_data",
    )
    target_spatial_dims = _infer_spatial_dims(target, spatial_dims=spatial_dims)
    target_template = _spatial_template(target, spatial_dims=target_spatial_dims)
    target_template = _ensure_xyz_coordinates(
        target_template,
        scan=target_obj,
        argument_name="target_data",
    )

    if source_data is None:
        if not isinstance(mask, xr.DataArray):
            raise ValueError(
                "source_data is required when mask is not an xarray DataArray."
            )
        source = mask
        source_obj_for_xyz = source_obj
    else:
        source_obj_for_xyz = source_data if source_obj is None else source_obj
        source = _select_dataarray_or_primary(
            source_data,
            variable=source_variable,
            argument_name="source_data",
        )

    source_spatial_dims = _infer_spatial_dims(source, spatial_dims=spatial_dims)
    source_template = _spatial_template(source, spatial_dims=source_spatial_dims)
    source_template = _ensure_xyz_coordinates(
        source_template,
        scan=source_obj_for_xyz,
        argument_name="source_data",
    )

    mask_name = name or getattr(mask, "name", None) or "mask"
    mask_da = _coerce_mask_to_source_grid(
        mask,
        source_template=source_template,
        spatial_dims=source_spatial_dims,
        name=mask_name,
    )
    mask_da = _clip_mask(mask_da).transpose(*source_template.dims)

    source_geometry = _estimate_xyz_affine(source_template)
    target_geometry = _estimate_xyz_affine(target_template)

    if plane_tolerance_mm is not None:
        plane_tolerance_mm = float(plane_tolerance_mm)
        if not np.isfinite(plane_tolerance_mm) or plane_tolerance_mm < 0:
            raise ValueError("plane_tolerance_mm must be finite and nonnegative.")
    elif len(source_geometry["active_axes"]) < 3:
        plane_tolerance_mm = _default_plane_tolerance_mm(source_geometry)

    target_xyz = _xyz_stack(target_template).reshape(-1, 3)
    offsets = _build_subvoxel_offsets(
        target_geometry,
        coverage_samples=coverage_samples,
    )
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    projected_flat = np.zeros(target_xyz.shape[0], dtype=np.float64)
    for start in range(0, target_xyz.shape[0], batch_size):
        stop = min(start + batch_size, target_xyz.shape[0])
        centres = target_xyz[start:stop]
        batch_values = np.zeros(stop - start, dtype=np.float64)
        for offset in offsets:
            batch_values += _interpolate_mask_values(
                mask_da,
                centres + offset,
                source_geometry=source_geometry,
                method=method,
                plane_tolerance_mm=plane_tolerance_mm,
            )
        projected_flat[start:stop] = batch_values / float(len(offsets))

    projected = np.clip(projected_flat.reshape(target_template.shape), 0.0, 1.0)
    if threshold is not None:
        projected = (projected >= float(threshold)).astype(np.float64)

    result = xr.DataArray(
        projected.astype(np.float32),
        dims=target_template.dims,
        coords=target_template.coords,
        name=mask_name,
        attrs=dict(getattr(mask, "attrs", {})),
    )
    result.attrs.update(
        {
            "mask_projection": "xyz",
            "source_spatial_dims": tuple(source_template.dims),
            "target_spatial_dims": tuple(target_template.dims),
            "interpolation_method": str(method).lower(),
            "coverage_samples": coverage_samples,
        }
    )
    return _clip_mask(result)


def _validate_mask_reduction(reduction: str, signal_part: str) -> None:
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(
            f"Mask reduction {reduction!r} is unsupported; "
            "use reduction='none', 'sum', or 'mean'."
        )
    if signal_part not in ("complex", "magnitude"):
        raise ValueError(
            f"Mask signal_part {signal_part!r} is unsupported; "
            "use signal_part='complex' to retain phase or 'magnitude'."
        )


def _apply_mask_weights(
    target: xr.DataArray,
    mask: xr.DataArray,
    *,
    spatial_dims: tuple[str, ...],
    reduction: str,
    signal_part: str,
    fill_value: float,
) -> xr.DataArray:
    weights = mask.broadcast_like(target).astype(np.float64)
    values = (
        abs(target.astype(np.result_type(target.dtype, np.float64)))
        if signal_part == "magnitude"
        else target
    )
    if reduction == "none":
        return (values * weights).where(weights > 0, fill_value)
    finite_weights = weights.where(np.isfinite(values), 0.0)
    numerator = (values * finite_weights).sum(dim=spatial_dims, skipna=True)
    if reduction == "sum":
        return numerator
    denominator = finite_weights.sum(dim=spatial_dims, skipna=True)
    return (numerator / denominator).where(denominator > 0)


def _record_mask_reduction(
    result: xr.DataArray,
    target: xr.DataArray,
    mask: xr.DataArray,
    *,
    reduction: str,
    signal_part: str,
) -> None:
    result.name = target.name
    # Signal metadata comes from the target, never from arithmetic mask operands.
    result.attrs = dict(target.attrs)
    result.attrs.update(
        applied_mask=mask.name,
        mask_reduction=reduction,
        mask_signal_part=signal_part,
    )


def extract_masked_data(
    data: xr.DataArray,
    mask: xr.DataArray | np.ndarray,
    *,
    reduction: Literal["none", "sum", "mean"] = "none",
    signal_part: Literal["complex", "magnitude"] = "complex",
    spatial_dims: Iterable[str] | None = None,
    fill_value: float = 0.0,
) -> xr.DataArray:
    """Extract weighted ROI data, a coherent sum/mean or a magnitude sum/mean.

    The mask must already occupy the input grid. Magnitude is taken per voxel
    before spatial reduction; complex reduction preserves phase cancellation.
    The caller supplies raw, reconstructed or Fourier-domain values explicitly.
    This function does not perform FFT, preprocessing or scanner-space projection.

    Parameters
    ----------
    data : xr.DataArray
        Numeric signal with labelled spatial and non-spatial dimensions in any
        order. Samples, units, coordinate values/direction and provenance are
        preserved. Complex inputs are supported without dropping quadratures.
    mask : xr.DataArray or np.ndarray
        Dimensionless fractional coverage on the input grid. Labelled masks may
        omit spatial dimensions, broadcasting along those dimensions; present
        dimensions must have matching sizes and coordinate values/direction.
        Shared spatial auxiliary coordinates must also match. NumPy masks must
        match the full spatial shape in ``spatial_dims`` order. Boolean masks
        select voxels unchanged. Fractional weights scale their contribution.
        Weights are clipped to [0, 1]; NaN/negative infinity become zero and
        positive infinity becomes one, as in scanner-space mask projection.
    reduction : {"none", "sum", "mean"}, default "none"
        ``"none"`` keeps the complete grid without averaging or flattening.
        ``"sum"`` returns the coverage-weighted spatial sum; ``"mean"`` divides
        that sum by the effective coverage weight. Non-finite samples contribute
        neither signal nor weight to reductions. Empty/all-invalid ROIs return
        zero for sums and NaN for means, independently per non-spatial position.
    signal_part : {"complex", "magnitude"}, default "complex"
        ``"complex"`` uses values as supplied, retaining sign and phase.
        ``"magnitude"`` takes absolute values before weighting/reduction. Thus
        opposite phases cancel in a complex sum/mean but not in a magnitude one.
        Magnitude is calculated at least in float64/complex128 precision.
    spatial_dims : iterable of str, optional
        Unique dimensions to reduce or retain under the mask. If omitted,
        exclude the known acquisition, time, repetition, channel, inversion,
        echo, RF-power and b-value index dimensions. Pass explicit names for
        other dimension conventions. Non-spatial dimensions retain their order.
    fill_value : float, default 0.0
        Outside-mask value for ``reduction="none"``; use NaN to mark missing
        voxels. Non-finite in-mask samples are preserved without reduction.
        Ignored by sums and means.

    Returns
    -------
    xr.DataArray
        Weighted grid or reduced signal in the input's units and domain. Weight
        calculation is float64; float32/complex64 inputs produce float64/complex128
        results, and higher input precision is retained. Unreduced dimensions and
        coordinates keep their order and direction; reductions drop coordinates
        that depend on spatial dimensions. Input name/attributes are retained
        with ``applied_mask``, ``mask_reduction`` and ``mask_signal_part`` metadata.
        Mask attributes are not copied into signal results, independently of
        xarray's ``keep_attrs`` setting; the input owns signal metadata.

    Raises
    ------
    TypeError
        If data are not a numeric DataArray or mask weights are not real numeric
        or boolean values.
    ValueError
        If options, spatial dimensions, mask shape or shared coordinates are
        inconsistent. Project masks from other grids with
        :func:`project_mask_to_data_grid` before calling this function.

    Notes
    -----
    Deterministic, with no input mutation, cache, reconstruction or UI creation.
    Operations preserve xarray's lazy arithmetic where supported. A sum counts
    fractional voxel contributions; it is not a physical volume integral.

    Examples
    --------
    Retain individual complex values and mark voxels outside the ROI as missing::

        selected = extract_masked_data(data, mask, fill_value=np.nan)

    Select coherent summation or a mean of per-voxel magnitudes explicitly::

        coherent = extract_masked_data(data, mask, reduction="sum")
        magnitude = extract_masked_data(
            data, mask, reduction="mean", signal_part="magnitude",
        )
    """
    _validate_mask_reduction(reduction, signal_part)
    if not isinstance(data, xr.DataArray) or not np.issubdtype(data.dtype, np.number):
        raise TypeError("extract_masked_data requires a numeric xarray DataArray.")
    dims = _infer_spatial_dims(data, spatial_dims=spatial_dims)
    if len(set(dims)) != len(dims):
        raise ValueError("extract_masked_data spatial_dims must contain unique names.")
    if not isinstance(mask, xr.DataArray):
        values = np.asarray(mask)
        expected = tuple(data.sizes[dim] for dim in dims)
        if values.shape != expected:
            raise ValueError(
                f"extract_masked_data mask shape {values.shape} must match "
                f"spatial_dims {dims} with shape {expected}."
            )
        mask = xr.DataArray(
            values,
            dims=dims,
            coords={dim: data.coords[dim] for dim in dims if dim in data.coords},
            name="mask",
        )
    if not (
        np.issubdtype(mask.dtype, np.number) or np.issubdtype(mask.dtype, np.bool_)
    ) or np.issubdtype(mask.dtype, np.complexfloating):
        raise TypeError("extract_masked_data mask must contain real coverage weights.")
    for dim in mask.dims:
        if dim not in dims or mask.sizes[dim] != data.sizes[dim]:
            raise ValueError(
                f"extract_masked_data mask dimension {dim!r} must match the input "
                f"spatial grid {dims}; project a mask from another grid first."
            )
    for coord_name in set(mask.coords) & set(data.coords):
        coordinate = mask.coords[coord_name]
        target_coordinate = data.coords[coord_name]
        if set(coordinate.dims).issubset(dims) and (
            not coordinate.variable.equals(target_coordinate.variable)
            or (
                "units" in coordinate.attrs
                and "units" in target_coordinate.attrs
                and coordinate.attrs["units"] != target_coordinate.attrs["units"]
            )
        ):
            raise ValueError(
                f"extract_masked_data mask coordinate {coord_name!r} must match "
                "the input grid and direction; project a mask from another grid first."
            )
    result = _apply_mask_weights(
        data,
        _clip_mask(mask),
        spatial_dims=dims,
        reduction=reduction,
        signal_part=signal_part,
        fill_value=fill_value,
    )
    _record_mask_reduction(
        result, data, mask, reduction=reduction, signal_part=signal_part
    )
    return result


def apply_projected_mask(
    target_data: xr.DataArray | xr.Dataset | Any,
    mask: xr.DataArray | np.ndarray,
    *,
    source_data: xr.DataArray | xr.Dataset | Any | None = None,
    source_obj: Any = None,
    target_obj: Any = None,
    target_variable: str | None = None,
    source_variable: str | None = None,
    spatial_dims: Iterable[str] | None = None,
    method: str = "nearest",
    coverage_samples: int | Iterable[int] = 3,
    threshold: float | None = None,
    fill_value: float = 0.0,
    mean_along_space: bool = False,
    reduction: Literal["none", "sum", "mean"] | None = None,
    signal_part: Literal["complex", "magnitude"] = "complex",
    return_mask: bool = False,
    plane_tolerance_mm: float | None = None,
    batch_size: int = 250_000,
    name: str | None = None,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Apply an x/y/z-projected ROI mask to target data.

    Outside the projected mask, values are set to ``fill_value``. Voxels with
    partial mask coverage are multiplied by their fractional coverage weight.
    Choose an explicit complex or per-voxel magnitude sum/mean with ``reduction``
    and ``signal_part``, or keep the full masked grid. Existing calls without
    these options retain their original numerical and metadata behavior.

    Parameters
    ----------
    target_data : xr.DataArray, xr.Dataset, or MRIExperiment-like object
        Data to mask. Spatial dimensions are retained unless
        a sum or mean is requested; all non-spatial dimensions, such as
        acquisition, repetition, or channel dimensions, are preserved.
        Experiment-like objects must expose ``primary_data``.
    mask : xr.DataArray or np.ndarray
        Mask values on the source grid. Values may be boolean, binary, or
        fractional and are clipped to ``[0, 1]``. A NumPy mask requires
        ``source_data``.
    source_data : xr.DataArray, xr.Dataset, or MRIExperiment-like object, optional
        Data defining the source grid on which ``mask`` was created. If omitted,
        an xarray ``mask`` is used as the source-grid template.
    source_obj : object, optional
        Source scan object used to derive scanner-space ``x/y/z`` coordinates
        when they are absent from ``source_data``. Normally the source
        ``MRIExperiment``.
    target_obj : object, optional
        Target scan object used to derive scanner-space ``x/y/z`` coordinates
        when they are absent from ``target_data``. When omitted and
        ``target_data`` is experiment-like, that object is used.
    target_variable : str, optional
        Variable selected from ``target_data`` when it is an xarray Dataset
        containing multiple data variables.
    source_variable : str, optional
        Variable selected from ``source_data`` when it is an xarray Dataset
        containing multiple data variables.
    spatial_dims : iterable of str, optional
        Spatial dimensions used for projection and, when requested, averaging.
        If omitted, they are inferred by excluding known non-spatial dimensions.
    method : {"nearest", "linear"}, default "nearest"
        Source-mask interpolation method. See
        :func:`project_mask_to_data_grid` for interpolation details.
    coverage_samples : int or iterable of int, default 3
        Subvoxel samples per target spatial dimension used to estimate target
        voxel coverage. A scalar applies to all dimensions; an iterable must
        provide one positive value per target spatial dimension.
    threshold : float, optional
        Optional cutoff applied to projected fractional coverage. Coverage at
        or above the threshold becomes ``1`` and lower coverage becomes ``0``.
        If omitted, fractional weights are retained.
    fill_value : float, default 0.0
        Value assigned where projected mask coverage is zero when returning
        spatially masked data. Use ``np.nan`` to mark outside-mask voxels as
        missing. This parameter does not affect sums or means.
    mean_along_space : bool, default False
        If False, return target data multiplied by fractional mask coverage,
        with outside-mask values replaced by ``fill_value``. If True, return
        ``sum(target * coverage) / sum(coverage)`` over all spatial dimensions.
        Non-finite target values are excluded from both numerator and effective
        weight; an empty ROI produces ``NaN``.
        Retained for compatibility; use ``reduction="mean"`` in new calls.
    reduction : {"none", "sum", "mean"} or None, optional
        Explicit spatial reduction, as in :func:`extract_masked_data`. None
        selects the existing ``mean_along_space`` behavior. ``"none"`` retains
        the complete grid, ``"sum"`` returns weighted sums and ``"mean"``
        returns weighted means. ``mean_along_space=True`` may only be combined
        with None or ``"mean"``. Empty sums are zero and empty means are NaN.
    signal_part : {"complex", "magnitude"}, default "complex"
        Complex retains input values/phase. Magnitude takes each voxel's absolute
        value before spatial weighting/reduction. No Fourier transform or other
        preprocessing is performed; the input defines the scientific domain.
    return_mask : bool, default False
        If True, return ``(result, projected_mask)``. If False, return only the
        masked data or weighted mean.
    plane_tolerance_mm : float, optional
        Distance tolerance in millimeters for lower-dimensional source mask
        planes. See :func:`project_mask_to_data_grid` for the default behavior.
    batch_size : int, default 250000
        Maximum target voxel centers projected per batch. Smaller batches use
        less temporary memory. Must be at least one.
    name : str, optional
        Name assigned to the projected mask. Defaults to ``mask.name`` when
        available, otherwise ``"mask"``.

    Returns
    -------
    xr.DataArray or tuple[xr.DataArray, xr.DataArray]
        Without reduction, an array with the same dimensions and
        coordinates as the selected target data. With a sum or mean,
        an array containing only the target's non-spatial dimensions. If
        ``return_mask=True``, the second tuple element is the fractional mask on
        the target spatial grid.

    Raises
    ------
    TypeError
        If target or source data do not provide a supported labelled array.
    ValueError
        If options conflict, reduction/signal part is unsupported, variables or
        spatial dimensions are inconsistent, scanner geometry is unavailable,
        or projection options are invalid.

    Notes
    -----
    Projects once and delegates weighting to the native-grid extraction owner.
    Deterministic without input mutation, caches or plotting. Projection eagerly
    samples mask geometry; signal arithmetic retains supported xarray laziness.
    Coordinates in millimeters define projection; signal units/domain are kept.
    Float64 weights promote float32/complex64 to float64/complex128. Explicit
    reduction or magnitude requests retain input attributes/name and record
    ``applied_mask``, ``mask_reduction`` and ``mask_signal_part``. Legacy calls
    retain their original metadata: masked grids have ``applied_mask`` and target
    attributes; reduced means have no attributes unless xarray's global
    ``keep_attrs=True`` is explicitly selected, in which case they retain only
    target attributes. ``keep_attrs=False`` and ``"default"`` both drop attributes
    from legacy means. Projection metadata stays on the returned projected mask;
    mask attributes never supply signal units or provenance.

    Examples
    --------
    Keep the complete target shape and mark values outside the ROI as missing::

        masked = apply_projected_mask(
            target,
            source_mask,
            source_data=source,
            fill_value=np.nan,
        )

    Return a coverage-weighted curve over all spatial dimensions::

        curve = apply_projected_mask(
            target,
            source_mask,
            source_data=source,
            reduction="mean",
            signal_part="magnitude",
        )
    """
    effective_reduction = (
        ("mean" if mean_along_space else "none") if reduction is None else reduction
    )
    _validate_mask_reduction(effective_reduction, signal_part)
    if mean_along_space and effective_reduction != "mean":
        raise ValueError(
            "apply_projected_mask mean_along_space=True conflicts with "
            f"reduction={reduction!r}; use reduction='mean' or omit mean_along_space."
        )
    target_obj = target_data if target_obj is None else target_obj
    target = _select_dataarray_or_primary(
        target_data,
        variable=target_variable,
        argument_name="target_data",
    )
    projected_mask = project_mask_to_data_grid(
        mask,
        target,
        source_data=source_data,
        source_obj=source_obj,
        target_obj=target_obj,
        source_variable=source_variable,
        spatial_dims=spatial_dims,
        method=method,
        coverage_samples=coverage_samples,
        threshold=threshold,
        plane_tolerance_mm=plane_tolerance_mm,
        batch_size=batch_size,
        name=name,
    )
    target_spatial_dims = tuple(
        dim for dim in projected_mask.dims if dim in target.dims
    )
    result = _apply_mask_weights(
        target,
        projected_mask,
        spatial_dims=target_spatial_dims,
        reduction=effective_reduction,
        signal_part=signal_part,
        fill_value=fill_value,
    )
    if reduction is not None or signal_part != "complex":
        _record_mask_reduction(
            result,
            target,
            projected_mask,
            reduction=effective_reduction,
            signal_part=signal_part,
        )
    elif not mean_along_space:
        result.attrs = dict(target.attrs)
        result.attrs["applied_mask"] = projected_mask.name
    else:
        # Preserve the legacy default even when xarray changes reduction defaults.
        result.attrs = (
            dict(target.attrs) if xr.get_options()["keep_attrs"] is True else {}
        )

    if return_mask:
        return result, projected_mask
    return result


def align_mask_to_data_grid(
    mask: xr.DataArray | np.ndarray,
    target_data: xr.DataArray | xr.Dataset,
    *,
    source_data: xr.DataArray | xr.Dataset | None = None,
    target_variable: str | None = None,
    source_variable: str | None = None,
    spatial_dims: Iterable[str] | None = None,
    slice_dim: str = "slice_idx",
    tol_mm: float = 0.1,
    name: str | None = None,
) -> xr.DataArray:
    """Align an ROI mask from one xarray image grid to another image grid.

    This is intended for the common case where two datasets have the same
    in-plane FOV/resolution/orientation, but one dataset has more slices. Slices
    are matched through ``slice_idx.attrs["axis_mm"]`` and non-overlapping
    target slices are filled with zeros.

    Parameters
    ----------
    mask : xr.DataArray or np.ndarray
        Source-grid mask. Values are converted to floating point and clipped to
        ``[0, 1]``. A NumPy mask without ``source_data`` is assumed to already
        have the target spatial shape.
    target_data : xr.DataArray or xr.Dataset
        Data defining the target grid. Select a Dataset variable with
        ``target_variable`` when more than one data variable is present.
    source_data : xr.DataArray or xr.Dataset, optional
        Data defining the source grid. If omitted, an xarray ``mask`` defines
        its own source grid; a NumPy mask is interpreted on the target grid.
    target_variable : str, optional
        Variable selected from a multi-variable target Dataset.
    source_variable : str, optional
        Variable selected from a multi-variable source Dataset.
    spatial_dims : iterable of str, optional
        Spatial dimension names and order. If omitted, known non-spatial
        dimensions are excluded and all remaining dimensions are used.
    slice_dim : str, default "slice_idx"
        Spatial dimension whose physical positions are matched through its
        coordinate attribute ``axis_mm``.
    tol_mm : float, default 0.1
        Maximum slice-position difference in millimeters for two slices to be
        considered overlapping.
    name : str, optional
        Name assigned to the aligned mask. Defaults to ``mask.name`` when
        available, otherwise ``"mask"``.

    Returns
    -------
    xr.DataArray
        Floating-point mask on the target spatial grid. Non-overlapping target
        slices contain zero and all values are clipped to ``[0, 1]``.

    Notes
    -----
    This legacy helper does not perform scanner-space resampling. Source and
    target must have matching in-plane dimensions, resolution, FOV, and
    orientation. Use :func:`project_mask_to_data_grid` when grids differ.
    """
    target = _select_dataarray(
        target_data,
        variable=target_variable,
        argument_name="target_data",
    )
    target_spatial_dims = _infer_spatial_dims(target, spatial_dims=spatial_dims)
    target_template = _spatial_template(target, spatial_dims=target_spatial_dims)

    if source_data is None:
        if not isinstance(mask, xr.DataArray):
            source_template = target_template
        else:
            source_template = mask
    else:
        source = _select_dataarray(
            source_data,
            variable=source_variable,
            argument_name="source_data",
        )
        source_template = _spatial_template(source, spatial_dims=target_spatial_dims)

    mask_name = name or getattr(mask, "name", None) or "mask"
    mask_da = _coerce_mask_to_source_grid(
        mask,
        source_template=source_template,
        spatial_dims=target_spatial_dims,
        name=mask_name,
    )

    target_mask = xr.zeros_like(target_template, dtype=np.float64).rename(mask_name)

    if slice_dim not in target_spatial_dims:
        if mask_da.shape != target_template.shape:
            raise ValueError(
                f"Mask {mask_name!r} has shape {mask_da.shape}, "
                f"expected target shape {target_template.shape}."
            )
        target_mask.values = mask_da.values
        return _clip_mask(target_mask)

    can_match_slices = (
        slice_dim in mask_da.dims
        and slice_dim in target_template.dims
        and slice_dim in mask_da.coords
        and slice_dim in target_template.coords
        and "axis_mm" in mask_da.coords[slice_dim].attrs
        and "axis_mm" in target_template.coords[slice_dim].attrs
    )

    if not can_match_slices:
        if mask_da.shape != target_template.shape:
            raise ValueError(
                f"Mask {mask_name!r} and target have different shapes "
                f"({mask_da.shape} vs {target_template.shape}) and cannot be "
                f"slice-matched without {slice_dim}.attrs['axis_mm']."
            )
        target_mask.values = mask_da.values
        return _clip_mask(target_mask)

    source_slice_indices, target_slice_indices = get_overlapping_slices(
        data_obj_1=mask_da,
        data_obj_2=target_template,
        tol_mm=tol_mm,
    )
    if not source_slice_indices:
        raise ValueError(
            f"No overlapping {slice_dim} positions were found between mask and target."
        )

    source_subset = mask_da.isel({slice_dim: source_slice_indices})
    target_block = target_template.isel({slice_dim: target_slice_indices})
    if source_subset.shape != target_block.shape:
        raise ValueError(
            f"Overlapping mask block shape {source_subset.shape} does not match "
            f"target block shape {target_block.shape}. This helper currently "
            "requires matching in-plane resolution and FOV."
        )

    full_values = target_mask.values.copy()
    slice_axis = target_mask.dims.index(slice_dim)
    slicer = [slice(None)] * target_mask.ndim
    slicer[slice_axis] = target_slice_indices
    full_values[tuple(slicer)] = source_subset.transpose(*target_mask.dims).values
    target_mask.values = full_values
    return _clip_mask(target_mask)


def apply_aligned_mask(
    target_data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | np.ndarray,
    *,
    source_data: xr.DataArray | xr.Dataset | None = None,
    target_variable: str | None = None,
    source_variable: str | None = None,
    spatial_dims: Iterable[str] | None = None,
    slice_dim: str = "slice_idx",
    tol_mm: float = 0.1,
    name: str | None = None,
) -> xr.DataArray:
    """Apply a source-grid mask to target data after grid/slice alignment.

    Parameters
    ----------
    target_data : xr.DataArray or xr.Dataset
        Data to multiply by the aligned mask. For a Dataset with multiple data
        variables, select one with ``target_variable``.
    mask : xr.DataArray or np.ndarray
        Source-grid mask. Boolean, integer, and fractional values are accepted
        and clipped to ``[0, 1]`` during alignment.
    source_data : xr.DataArray or xr.Dataset, optional
        Data defining the grid on which ``mask`` was created. If omitted, an
        xarray mask defines its own source grid.
    target_variable : str, optional
        Variable selected from a multi-variable target Dataset.
    source_variable : str, optional
        Variable selected from a multi-variable source Dataset.
    spatial_dims : iterable of str, optional
        Spatial dimension names and order. If omitted, they are inferred by
        excluding known non-spatial dimensions.
    slice_dim : str, default "slice_idx"
        Dimension matched by physical slice position using ``axis_mm``.
    tol_mm : float, default 0.1
        Maximum slice-position difference in millimeters for a match.
    name : str, optional
        Name assigned to the intermediate aligned mask.

    Returns
    -------
    xr.DataArray
        Target data multiplied by the aligned mask. Outside-mask values are
        zero; fractional mask values proportionally weight target values. The
        target's original dimensions and non-spatial coordinates are retained.

    Notes
    -----
    This function uses :func:`align_mask_to_data_grid` and therefore requires
    matching in-plane grids. Use :func:`apply_projected_mask` for differing
    position, orientation, FOV, or resolution.
    """
    target = _select_dataarray(
        target_data,
        variable=target_variable,
        argument_name="target_data",
    )
    aligned_mask = align_mask_to_data_grid(
        mask,
        target,
        source_data=source_data,
        source_variable=source_variable,
        spatial_dims=spatial_dims,
        slice_dim=slice_dim,
        tol_mm=tol_mm,
        name=name,
    )
    return target * aligned_mask


def extract_values_with_aligned_mask(
    target_data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | np.ndarray,
    *,
    source_data: xr.DataArray | xr.Dataset | None = None,
    target_variable: str | None = None,
    source_variable: str | None = None,
    spatial_dims: Iterable[str] | None = None,
    slice_dim: str = "slice_idx",
    tol_mm: float = 0.1,
    name: str | None = None,
    min_weight: float = 0.0,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, xr.DataArray]:
    """Return finite target values inside a source-grid ROI mask.

    Fractional masks are supported; voxels are included when the aligned mask
    weight is greater than ``min_weight``. For weighted statistics, use the
    returned mask together with the target data.

    Parameters
    ----------
    target_data : xr.DataArray or xr.Dataset
        Data to sample after mask alignment. If a Dataset is supplied, select
        the variable with ``target_variable`` unless it contains exactly one
        data variable.
    mask : xr.DataArray or np.ndarray
        ROI mask drawn on the source grid. Boolean, integer, and fractional
        masks are accepted; values are clipped to the interval ``[0, 1]``.
    source_data : xr.DataArray or xr.Dataset, optional
        Dataset or image array defining the source grid on which ``mask`` was
        drawn. Pass this when the mask itself does not carry complete physical
        coordinates, for example masks created from ``create_mask``.
    target_variable : str, optional
        Variable name to extract from ``target_data`` when ``target_data`` is an
        xarray Dataset with multiple variables.
    source_variable : str, optional
        Variable name to extract from ``source_data`` when ``source_data`` is an
        xarray Dataset with multiple variables.
    spatial_dims : iterable of str, optional
        Spatial dimensions to align over. If omitted, dimensions commonly used
        for time, channels, inversion, echo, RF power, and b-values are ignored
        and all remaining dimensions are treated as spatial.
    slice_dim : str, default "slice_idx"
        Dimension used for slice-position matching.
    tol_mm : float, default 0.1
        Maximum absolute slice-position difference in millimeters for two
        slices to be considered overlapping.
    name : str, optional
        Name assigned to the aligned mask. Defaults to the mask name when
        available, otherwise ``"mask"``.
    min_weight : float, default 0.0
        Minimum aligned mask weight required for a voxel to be included in the
        returned values.
    return_mask : bool, default False
        If True, return ``(values, aligned_mask)``. If False, return only the
        one-dimensional values array.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, xr.DataArray]
        Finite target values inside the aligned mask. If ``return_mask=True``,
        the aligned target-grid mask is returned as the second tuple element.
    """
    target = _select_dataarray(
        target_data,
        variable=target_variable,
        argument_name="target_data",
    )
    aligned_mask = align_mask_to_data_grid(
        mask,
        target,
        source_data=source_data,
        source_variable=source_variable,
        spatial_dims=spatial_dims,
        slice_dim=slice_dim,
        tol_mm=tol_mm,
        name=name,
    )
    selected = target.where(aligned_mask > float(min_weight))
    values = np.asarray(selected.values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if return_mask:
        return values, aligned_mask
    return values
