from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.utils.mathematics.calculations import (
    apply_gaussian_smoothing,
    calc_sampling_time_axis,
    compute_weighted_mean_and_std,
    find_max_and_index,
    gaussian,
    get_histogram,
    moving_average,
    part_func,
    polynomial_fit,
)


def test_get_histogram_ignores_nan_values_in_two_dimensional_data():
    centers, counts, bin_size = get_histogram(
        np.array([[0.0, np.nan], [1.0, 2.0]]), number_of_bins=3
    )

    np.testing.assert_allclose(centers, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(counts, [1.0, 1.0, 1.0])
    assert bin_size == pytest.approx(2 / 3)


def test_basic_curve_and_statistics_helpers_return_expected_values():
    np.testing.assert_allclose(
        gaussian(np.array([0.0, 1.0]), 2.0, 0.0, 1.0), [2.0, 2 / np.e**0.5]
    )

    mean, std = compute_weighted_mean_and_std([1.0, 3.0], [1.0, 3.0])
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(np.sqrt(0.75))

    coefficients, polynomial = polynomial_fit([0, 1, 2], [1, 3, 5])
    np.testing.assert_allclose(coefficients, [2.0, 1.0])
    assert polynomial(3) == pytest.approx(7.0)


def test_find_max_and_index_can_explicitly_use_absolute_values():
    values = np.array([[1.0, -4.0], [3.0, 2.0]])
    data_array = xr.DataArray(values, dims=("row", "column"))

    assert find_max_and_index(values, magnitude=True) == (4.0, (0, 1))
    assert find_max_and_index(data_array, magnitude=True) == (4.0, (0, 1))


def test_find_max_and_index_preserves_signed_real_default():
    values = np.array([[1.0, -4.0], [3.0, 2.0]])
    for data in (values, xr.DataArray(values, dims=("row", "column"))):
        assert find_max_and_index(data) == (3.0, (1, 0))
        assert find_max_and_index(data, magnitude=False) == (3.0, (1, 0))
    for data in (-values - 5, xr.DataArray(-values - 5)):
        assert find_max_and_index(data) == (-1.0, (0, 1))


def test_find_max_and_index_keeps_complex_magnitude_behavior():
    values = np.array([[1j, -3 - 4j], [3.0, 2.0]], dtype=complex)
    for data in (values, xr.DataArray(values, dims=("row", "column"))):
        with pytest.warns(UserWarning, match="magnitude"):
            assert find_max_and_index(data) == (5.0, (0, 1))
        assert find_max_and_index(data, magnitude=True) == (5.0, (0, 1))


def test_sampling_axis_uses_explicit_dwell_time_or_data_object_metadata():
    np.testing.assert_allclose(
        calc_sampling_time_axis(dwell_time_s=0.002, npoints=3),
        [0.0, 0.002, 0.004],
    )

    data_obj = SimpleNamespace(n_acquisition=5, cut_off=1)
    np.testing.assert_allclose(
        calc_sampling_time_axis(dwell_time_s=0.01, scan=data_obj),
        [0.0, 0.01, 0.02, 0.03],
    )


def test_signal_component_and_smoothing_helpers():
    data = np.array([3 + 4j, -1j])

    np.testing.assert_array_equal(part_func(data, "Real"), [3.0, 0.0])
    np.testing.assert_array_equal(part_func(data, "Imaginary"), [4.0, -1.0])
    np.testing.assert_allclose(part_func(data, "Phase"), np.angle(data))
    np.testing.assert_allclose(part_func(data, "unsupported"), np.abs(data))
    assert part_func(None) is None
    np.testing.assert_allclose(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    impulse = np.zeros(5)
    impulse[2] = 1
    smoothed = apply_gaussian_smoothing(impulse, sigma=1)
    assert 0 < smoothed[2] < 1
    assert smoothed.sum() == pytest.approx(1.0)
