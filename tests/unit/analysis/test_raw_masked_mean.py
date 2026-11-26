"""Verify raw ROI analysis independently of any widget/controller construction."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.anatomical.roi import raw_masked_mean


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_direct_mean_is_repeatable_preserves_inputs_and_original_axis_order(
    dtype, preview, permuted
):
    # Invented 3/9 a.u. quadratures with 1:3 coverage; slice 1 doubles the signal.
    data = xr.DataArray(
        np.array([[[3, 6], [6, 12]], [[9, 12], [18, 24]]], dtype=dtype),
        dims=("read_idx", "slice_idx", "echo_idx"),
        coords={
            "echo_idx": xr.DataArray([20, 10], dims="echo_idx", attrs={"units": "ms"})
        },
        attrs={"units": "a.u.", "signal_domain": "temporal", "fixture": "invented"},
    )
    if np.issubdtype(dtype, np.complexfloating):
        data = data * 1j
    mask = xr.DataArray([[0.25, 0.25], [0.75, 0.75]], dims=("read_idx", "slice_idx"))
    if preview:
        mask = mask.isel(slice_idx=1, drop=True)
    if permuted:
        # Dimension names, including square shapes, determine voxel matching.
        data = data.transpose("echo_idx", "read_idx", "slice_idx")
    before = data.copy(deep=True)
    mask_before = mask.copy(deep=True)
    expected = [15, 21] if preview else [11.25, 15.75]
    with xr.set_options(keep_attrs=False):
        first = raw_masked_mean(
            data,
            mask,
            secondary_dim="echo_idx",
            selectors={},
            preview_slice=("slice_idx", 1) if preview else None,
        )
    with xr.set_options(keep_attrs=True):
        second = raw_masked_mean(
            data,
            mask,
            secondary_dim="echo_idx",
            selectors={},
            preview_slice=("slice_idx", 1) if preview else None,
        )
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(second, first, rtol=0, atol=1e-12)
    assert first.dtype == np.float64
    xr.testing.assert_identical(data, before)
    xr.testing.assert_identical(mask, mask_before)


def test_direct_empty_or_invalid_signal_positions_are_nan():
    data = xr.DataArray([[np.nan, 4], [np.inf, 8]], dims=("read_idx", "echo_idx"))
    mask = xr.DataArray([0.25, 0.75], dims="read_idx")
    result = raw_masked_mean(data, mask, secondary_dim="echo_idx", selectors={})
    np.testing.assert_allclose(result, [np.nan, 7], rtol=0, atol=1e-12, equal_nan=True)
    empty = raw_masked_mean(data, mask * 0, secondary_dim="echo_idx", selectors={})
    assert np.isnan(empty).all()


def test_direct_spatial_preview_covers_only_the_selected_plane():
    data = xr.DataArray([[3, 9]], dims=("read_idx", "slice_idx"))
    mask = xr.DataArray([1], dims="read_idx")
    result = raw_masked_mean(
        data,
        mask,
        secondary_dim="slice_idx",
        selectors={},
        preview_slice=("slice_idx", 1),
    )
    np.testing.assert_allclose(result, [np.nan, 9], rtol=0, atol=1e-12, equal_nan=True)


def test_direct_failed_data_selection_yields_nan_without_mutating_selectors():
    data = xr.DataArray(np.ones((1, 2)), dims=("read_idx", "echo_idx"))
    selectors = {"read_idx": 5}
    result = raw_masked_mean(
        data,
        xr.DataArray([1], dims="read_idx"),
        secondary_dim="echo_idx",
        selectors=selectors,
    )
    assert np.isnan(result).all()
    assert selectors == {"read_idx": 5}


def test_direct_invalid_secondary_dimension_has_actionable_error():
    data = xr.DataArray([1], dims="read_idx")
    with pytest.raises(ValueError, match="raw_masked_mean secondary_dim 'missing'"):
        raw_masked_mean(data, data, secondary_dim="missing", selectors={})


def test_direct_incompatible_committed_mask_index_still_raises():
    data = xr.DataArray([[1, 2]], dims=("read_idx", "echo_idx"))
    mask = xr.DataArray([[1]], dims=data.dims)
    with pytest.raises(IndexError):
        raw_masked_mean(data, mask, secondary_dim="echo_idx", selectors={})


@pytest.mark.parametrize("square", [False, True])
def test_direct_named_complex_alignment_preserves_quadratures(square):
    """Independent quadrature reference exposes both historical mapping errors."""
    shape = (2, 2) if square else (2, 3)
    data = xr.DataArray(
        np.arange(1, np.prod(shape) + 1).reshape(*shape, 1) + 20j,
        dims=("read_idx", "phase_idx", "echo_idx"),
    )
    weights = np.zeros(shape[::-1])
    weights[-1, 0] = 1
    mask = xr.DataArray(weights, dims=("phase_idx", "read_idx"))
    result = raw_masked_mean(data, mask, secondary_dim="echo_idx", selectors={})
    expected = np.sqrt((2 if square else 3) ** 2 + 20**2)
    np.testing.assert_allclose(result, [expected], rtol=0, atol=1e-12)


def test_direct_same_shape_different_coordinate_grid_is_rejected():
    data = xr.DataArray(
        [[3j], [-9j]], dims=("read_idx", "echo_idx"), coords={"read_idx": [0, 1]}
    )
    mask = xr.DataArray([1, 0], dims="read_idx", coords={"read_idx": [1, 0]})
    with pytest.raises(ValueError, match="mask coordinate.*read_idx"):
        raw_masked_mean(data, mask, secondary_dim="echo_idx", selectors={})


def test_direct_spatial_profile_of_single_voxels_uses_absolute_values():
    data = xr.DataArray([-3, 9], dims="read_idx")
    result = raw_masked_mean(
        data, xr.ones_like(data), secondary_dim="read_idx", selectors={}
    )
    np.testing.assert_allclose(result, [3, 9], rtol=0, atol=1e-12)


@pytest.mark.parametrize("weights", [[1j, 1], ["0", "1"]])
def test_direct_invalid_weight_type_is_rejected_before_editor_cast(weights):
    """Coverage cannot discard quadratures or interpret strings as numbers."""
    data = xr.DataArray([[3j], [-9j]], dims=("read_idx", "echo_idx"))
    with pytest.raises(TypeError, match="raw_masked_mean mask.*real coverage"):
        raw_masked_mean(
            data,
            xr.DataArray(weights, dims="read_idx"),
            secondary_dim="echo_idx",
            selectors={},
        )
