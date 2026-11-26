"""Characterize the old plot helpers before assigning domain modules."""

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr


def _plot():
    return importlib.import_module("hypermri.utils.plot.matplotlib_helpers")


def _slices(values):
    data = xr.DataArray(
        np.zeros(len(values)),
        dims="slice_idx",
        coords={"slice_idx": np.arange(len(values))},
    )
    data.slice_idx.attrs["axis_mm"] = np.array(values)
    return data


def test_slice_matching_preserves_order_ties_and_strict_tolerance():
    first = _slices([2.0, 0.0, 2.05, 9.0])
    second = _slices([0.05, 2.0])
    a, b = first.copy(deep=True), second.copy(deep=True)
    overlap = _plot().get_overlapping_slices
    assert overlap(first, second, tol_mm=0.1) == ([0, 1, 2], [1, 0, 1])
    assert overlap(first, second, tol_mm=-0.1) == ([0, 1, 2], [1, 0, 1])
    assert overlap(first, second, tol_mm=0) == ([], [])
    assert overlap(_slices([1.0]), _slices([0.0, 2.0]), tol_mm=2) == ([0], [0])
    assert overlap(_slices([0.0]), _slices([0.5]), tol_mm=0.5) == ([], [])
    xr.testing.assert_identical(first, a)
    xr.testing.assert_identical(second, b)


@pytest.mark.parametrize(
    "count, expected",
    [(0, (0, 0)), (1, (1, 1)), (2, (1, 2)), (6, (2, 3)), (10, (3, 4))],
)
def test_subplot_layout_keeps_the_existing_grid(count, expected):
    assert _plot().subplot_auto_size(count) == expected


def test_removing_empty_axes_preserves_populated_axes_and_return_shape():
    fig, axes = plt.subplots(2, 2)
    try:
        axes[0, 0].plot([1, 2])
        axes[1, 0].imshow([[1]])
        returned_fig, returned_axes = _plot().remove_empty_subplots(fig, axes)
        assert returned_fig is fig
        assert returned_axes.shape == (4,)
        assert fig.axes == [axes[0, 0], axes[1, 0]]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("discrete", [False, True])
def test_colormap_preserves_color_order_and_nan_color(discrete):
    cmap = _plot().create_colormap(
        ["#ff0000", "#0000ff"],
        discrete=discrete,
        register=False,
        reverse=True,
        nan_color="yellow",
    )
    np.testing.assert_allclose(cmap(0.0), [0, 0, 1, 1], atol=1e-15)
    np.testing.assert_allclose(cmap(1.0), [1, 0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(cmap(np.nan), [1, 1, 0, 1], atol=1e-15)


def test_three_dimensional_plot_keeps_values_threshold_and_colorbar_label():
    data = np.array([1.0, 100001.0]).reshape(1, 1, 2)
    snapshot = data.copy()
    axis = _plot().plot_3d(data)
    try:
        np.testing.assert_array_equal(axis.collections[0].get_array(), [1, 100001])
        np.testing.assert_array_equal(axis.collections[0].get_sizes(), [10, 0])
        assert axis.figure.axes[1].get_ylabel() == "Frequency [Hz]"
        np.testing.assert_array_equal(data, snapshot)
    finally:
        plt.close(axis.figure)
