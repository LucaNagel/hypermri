import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.anatomical.masks import (
    align_mask_to_data_grid,
    apply_aligned_mask,
    apply_projected_mask,
    extract_values_with_aligned_mask,
    project_mask_to_data_grid,
)


def _dataarray(
    values,
    *,
    slice_positions,
    name,
):
    values = np.asarray(values, dtype=np.float64)
    data = xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": np.arange(values.shape[0]),
            "phase_idx": np.arange(values.shape[1]),
            "slice_idx": np.arange(values.shape[2]),
        },
        name=name,
    )
    data.coords["slice_idx"].attrs["axis_mm"] = np.asarray(
        slice_positions,
        dtype=np.float64,
    )
    return data


def _mask_like_without_slice_position_attrs(source):
    return xr.DataArray(
        np.zeros(source.shape, dtype=np.float64),
        dims=source.dims,
        coords={dim: source.coords[dim].values for dim in source.dims},
    )


def _xyz_dataarray(values, *, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    values = np.asarray(values, dtype=np.float64)
    dims = ("read_idx", "phase_idx", "slice_idx")
    read = origin[0] + np.arange(values.shape[0]) * spacing[0]
    phase = origin[1] + np.arange(values.shape[1]) * spacing[1]
    slice_ = origin[2] + np.arange(values.shape[2]) * spacing[2]
    x, y, z = np.meshgrid(read, phase, slice_, indexing="ij")
    data = xr.DataArray(
        values,
        dims=dims,
        coords={
            "read_idx": np.arange(values.shape[0]),
            "phase_idx": np.arange(values.shape[1]),
            "slice_idx": np.arange(values.shape[2]),
            "x": (dims, x),
            "y": (dims, y),
            "z": (dims, z),
        },
    )
    for dim, resolution in zip(dims, spacing, strict=True):
        data.coords[dim].attrs["resolution_mm"] = resolution
    return data


def _xyz_csi_dataarray(
    values,
    *,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
):
    values = np.asarray(values, dtype=np.float64)
    dims = (
        "acquisition_idx",
        "read_idx",
        "phase_idx",
        "slice_idx",
        "repetition_idx",
        "channel_idx",
    )
    spatial_dims = ("read_idx", "phase_idx", "slice_idx")
    read = origin[0] + np.arange(values.shape[1]) * spacing[0]
    phase = origin[1] + np.arange(values.shape[2]) * spacing[1]
    slice_ = origin[2] + np.arange(values.shape[3]) * spacing[2]
    x, y, z = np.meshgrid(read, phase, slice_, indexing="ij")
    data = xr.DataArray(
        values,
        dims=dims,
        coords={
            "acquisition_idx": np.arange(values.shape[0], dtype=np.float64),
            "repetition_idx": np.arange(values.shape[4]),
            "channel_idx": np.arange(values.shape[5]),
            "x": (spatial_dims, x),
            "y": (spatial_dims, y),
            "z": (spatial_dims, z),
        },
    )
    for dim, resolution in zip(spatial_dims, spacing, strict=True):
        data.coords[dim].attrs["resolution_mm"] = resolution
    return data


@pytest.mark.parametrize("mask_kind", ["numpy", "labelled", "self_contained"])
def test_selected_plane_projects_physical_thickness_and_explicit_override(mask_kind):
    source = _xyz_dataarray(np.ones((2, 2, 1)))
    source.attrs["slice_thickness_mm"] = 2.0
    source = source.isel(slice_idx=0, drop=True)
    before = source.copy(deep=True)
    target = _xyz_dataarray(
        np.zeros((2, 2, 7)), origin=(0, 0, -1.5), spacing=(1, 1, 0.5)
    )
    if mask_kind == "numpy":
        mask = np.ones(source.shape)
    else:
        mask = xr.ones_like(source)
        if mask_kind == "labelled":
            mask.attrs.clear()
    arguments = {} if mask_kind == "self_contained" else {"source_data": source}
    result = project_mask_to_data_grid(mask, target, coverage_samples=1, **arguments)
    # Analytic slab [-1, 1] mm, including both boundaries. The former 0.001 mm
    # fallback covered only the central probe instead of these five planes.
    np.testing.assert_allclose(
        result, np.broadcast_to([0, 1, 1, 1, 1, 1, 0], target.shape), rtol=0, atol=1e-7
    )
    thin = project_mask_to_data_grid(
        mask, target, coverage_samples=1, plane_tolerance_mm=1e-3, **arguments
    )
    np.testing.assert_allclose(
        thin, np.broadcast_to([0, 0, 0, 1, 0, 0, 0], target.shape), rtol=0, atol=1e-7
    )
    xr.testing.assert_identical(source, before)


@pytest.mark.parametrize("thickness", [None, np.nan, 0, -1, [1, 2]])
def test_selected_plane_without_supported_thickness_requires_explicit_tolerance(
    thickness,
):
    source = _xyz_dataarray(np.ones((2, 2, 1))).isel(slice_idx=0, drop=True)
    source.attrs["slice_thickness_mm"] = thickness
    target = _xyz_dataarray(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="slice_thickness_mm.*plane_tolerance_mm"):
        project_mask_to_data_grid(source, target, coverage_samples=1)
    result = project_mask_to_data_grid(
        source, target, coverage_samples=1, plane_tolerance_mm=1.0
    )
    np.testing.assert_allclose(result, 1, rtol=0, atol=1e-7)


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_projection_rejects_invalid_explicit_plane_tolerance(tolerance):
    source = _xyz_dataarray(np.ones((2, 2, 1)))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        project_mask_to_data_grid(source, source, plane_tolerance_mm=tolerance)


def test_selected_plane_fractional_coverage_uses_its_thickness():
    source = _xyz_dataarray(np.ones((2, 2, 1)))
    source.attrs["slice_thickness_mm"] = 2.0
    source = source.isel(slice_idx=0, drop=True)
    target = _xyz_dataarray(np.zeros((2, 2, 2)), origin=(0, 0, 0.9))
    result = project_mask_to_data_grid(source, target, coverage_samples=(1, 1, 3))
    # Two of three z subcentres of the 0.9 mm voxel fall within [-1, 1] mm.
    np.testing.assert_allclose(
        result, np.broadcast_to([2 / 3, 0], target.shape), rtol=0, atol=1e-7
    )


def test_align_mask_to_data_grid_maps_overlapping_source_slice_to_target_slice():
    source = _dataarray(
        np.zeros((2, 3, 3)),
        slice_positions=(0.0, 1.0, 2.0),
        name="dce",
    )
    target = _dataarray(
        np.arange(6).reshape(2, 3, 1) + 100.0,
        slice_positions=(1.0,),
        name="t1",
    )
    mask = _mask_like_without_slice_position_attrs(source)
    mask.loc[dict(read_idx=[0, 1], phase_idx=[1], slice_idx=[1])] = 1.0

    aligned = align_mask_to_data_grid(
        mask,
        target,
        source_data=source,
        name="left_cortex",
    )

    expected = np.zeros((2, 3, 1), dtype=np.float64)
    expected[:, 1, 0] = 1.0
    np.testing.assert_array_equal(aligned.values, expected)
    assert aligned.dims == target.dims
    assert aligned.name == "left_cortex"
    np.testing.assert_array_equal(
        aligned.coords["slice_idx"].attrs["axis_mm"],
        np.array([1.0]),
    )


def test_extract_values_with_aligned_mask_returns_t1_values_inside_roi():
    source = _dataarray(
        np.zeros((2, 3, 3)),
        slice_positions=(0.0, 1.0, 2.0),
        name="dce",
    )
    target = _dataarray(
        np.array([[[10.0], [20.0], [30.0]], [[40.0], [50.0], [60.0]]]),
        slice_positions=(1.0,),
        name="t1",
    )
    mask = _mask_like_without_slice_position_attrs(source)
    mask.loc[dict(read_idx=[0, 1], phase_idx=[1], slice_idx=[1])] = 1.0

    values, aligned_mask = extract_values_with_aligned_mask(
        xr.Dataset({"t1": target}),
        mask,
        source_data=source,
        target_variable="t1",
        return_mask=True,
    )
    masked = apply_aligned_mask(target, mask, source_data=source)

    np.testing.assert_array_equal(values, np.array([20.0, 50.0]))
    assert float(np.nanmean(values)) == 35.0
    masked_roi_values = masked.where(aligned_mask > 0).values
    np.testing.assert_array_equal(
        masked_roi_values[np.isfinite(masked_roi_values)],
        np.array([20.0, 50.0]),
    )


def test_align_mask_to_data_grid_broadcasts_missing_spatial_dims_on_same_grid():
    target = _dataarray(
        np.zeros((2, 3, 2)),
        slice_positions=(1.0, 2.0),
        name="t1",
    )
    mask = xr.DataArray(
        np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]]),
        dims=("read_idx", "phase_idx"),
        coords={
            "read_idx": target.coords["read_idx"],
            "phase_idx": target.coords["phase_idx"],
        },
    )

    aligned = align_mask_to_data_grid(mask, target, source_data=target)

    expected = np.zeros((2, 3, 2), dtype=np.float64)
    expected[:, 1, :] = np.array([[1.0, 1.0], [0.5, 0.5]])
    np.testing.assert_array_equal(aligned.values, expected)


def test_align_mask_to_data_grid_raises_when_no_slices_overlap():
    source = _dataarray(
        np.zeros((2, 3, 1)),
        slice_positions=(0.0,),
        name="dce",
    )
    target = _dataarray(np.zeros((2, 3, 1)), slice_positions=(5.0,), name="t1")
    mask = xr.ones_like(source)

    with pytest.raises(ValueError, match="No overlapping slice_idx positions"):
        align_mask_to_data_grid(mask, target, source_data=source)


def test_project_mask_to_data_grid_returns_fractional_target_voxel_coverage():
    source = _xyz_dataarray(np.zeros((4, 4, 4)))
    mask = xr.zeros_like(source)
    mask.loc[dict(read_idx=[0])] = 1.0
    target = _xyz_dataarray(
        np.ones((2, 2, 2)),
        origin=(0.5, 0.5, 0.5),
        spacing=(2.0, 2.0, 2.0),
    )

    projected = project_mask_to_data_grid(
        mask,
        target,
        source_data=source,
        coverage_samples=2,
    )

    expected = np.zeros((2, 2, 2), dtype=np.float32)
    expected[0, :, :] = 0.5
    np.testing.assert_allclose(projected.values, expected)
    assert projected.dims == target.dims
    assert projected.attrs["mask_projection"] == "xyz"


def test_apply_projected_mask_weights_partial_voxels_and_supports_nan_fill():
    source = _xyz_dataarray(np.zeros((4, 4, 4)))
    mask = xr.zeros_like(source)
    mask.loc[dict(read_idx=[0])] = 1.0
    target = _xyz_dataarray(
        np.full((2, 2, 2), 10.0),
        origin=(0.5, 0.5, 0.5),
        spacing=(2.0, 2.0, 2.0),
    )

    masked, projected = apply_projected_mask(
        target,
        mask,
        source_data=source,
        coverage_samples=2,
        fill_value=np.nan,
        return_mask=True,
    )

    np.testing.assert_allclose(projected.values[0, :, :], 0.5)
    np.testing.assert_allclose(masked.values[0, :, :], 5.0)
    assert np.isnan(masked.values[1, :, :]).all()


def test_apply_projected_mask_mean_along_space_uses_mask_weights():
    source = _xyz_dataarray(np.zeros((4, 4, 4)))
    mask = xr.zeros_like(source)
    mask.loc[dict(read_idx=[0])] = 1.0
    target = _xyz_dataarray(
        np.full((2, 2, 2), 10.0),
        origin=(0.5, 0.5, 0.5),
        spacing=(2.0, 2.0, 2.0),
    )

    mean_value = apply_projected_mask(
        target,
        mask,
        source_data=source,
        coverage_samples=2,
        mean_along_space=True,
    )

    assert mean_value.dims == ()
    assert float(mean_value) == pytest.approx(10.0)


def test_apply_projected_mask_mean_along_space_preserves_csi_acquisition_axis():
    source = _xyz_dataarray(np.zeros((4, 4, 4)))
    mask = xr.zeros_like(source)
    mask.loc[dict(read_idx=[0])] = 1.0

    acquisition_values = np.arange(3, dtype=np.float64)[:, None, None, None, None, None]
    repetition_values = (
        10.0 * np.arange(2, dtype=np.float64)[None, None, None, None, :, None]
    )
    target_values = np.broadcast_to(
        100.0 + acquisition_values + repetition_values,
        (3, 2, 2, 2, 2, 1),
    )
    target = _xyz_csi_dataarray(
        target_values,
        origin=(0.5, 0.5, 0.5),
        spacing=(2.0, 2.0, 2.0),
    )

    curve = apply_projected_mask(
        target,
        mask,
        source_data=source,
        coverage_samples=2,
        mean_along_space=True,
    )

    assert curve.dims == ("acquisition_idx", "repetition_idx", "channel_idx")
    assert curve.shape == (3, 2, 1)
    expected = 100.0 + np.arange(3)[:, None, None] + 10.0 * np.arange(2)[None, :, None]
    np.testing.assert_allclose(curve.values, expected)
