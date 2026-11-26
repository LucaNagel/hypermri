"""Protect the endpoint-midpoint centring of frequency axes.

All axes are invented; Hz/ppm calibration uses 100 Hz per ppm. Numerical
references are endpoint midpoints and affine calibration, independent of the
implementation. Float64/calibration tolerances are 1e-12; float32 uses 1e-6
for rounding at these small frequency spans. No acquired signals are used.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.frequency import (
    center_freq_axis,
    freq_hz_to_ppm,
    get_freq_axis,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint16])
@pytest.mark.parametrize("descending", [False, True])
def test_nonuniform_axis_keeps_order_spacing_precision_and_input(dtype, descending):
    axis = np.array([100, 120, 180, 300], dtype=dtype)
    expected = np.array([-100, -80, -20, 100])
    if descending:
        axis = axis[::-1]
        expected = expected[::-1]
    before = axis.copy()
    result = center_freq_axis(axis)
    tolerance = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        np.diff(result), np.diff(axis.astype(float)), rtol=tolerance, atol=tolerance
    )
    assert result.dtype == (np.float32 if dtype == np.float32 else np.float64)
    assert not np.shares_memory(result, axis)
    np.testing.assert_array_equal(axis, before)
    np.testing.assert_array_equal(center_freq_axis(axis), result)


@pytest.mark.parametrize("value", [-170.0, 0.0, 170.0])
def test_single_sample_has_zero_relative_frequency(value):
    axis = np.array([value], dtype=np.float32)
    result = center_freq_axis(axis)
    np.testing.assert_allclose(result, [0.0], rtol=1e-12, atol=1e-12)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(axis, [value])


@pytest.mark.parametrize("descending", [False, True])
def test_integer_translation_avoids_span_overflow(descending):
    # The 60 kHz span exceeds int16, but all frequencies and centered results fit.
    axis = np.array([-30000, 0, 30000], dtype=np.int16)
    if descending:
        axis = axis[::-1]
    result = center_freq_axis(axis)
    np.testing.assert_allclose(result, axis, rtol=1e-12, atol=1e-12)
    assert result.dtype == np.float64


def test_integer_translation_retains_small_spacing_at_large_offset():
    # Removing an exactly represented integer offset precedes float64 output.
    axis = np.array([2**60, 2**60 + 2, 2**60 + 4], dtype=np.int64)
    np.testing.assert_allclose(
        center_freq_axis(axis), [-2.0, 0.0, 2.0], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    "axis",
    [
        [0.0, 2.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0],
    ],
)
def test_nonmonotonic_and_repeated_coordinates_raise_without_sorting(axis):
    axis = np.asarray(axis)
    before = axis.copy()
    with pytest.raises(ValueError, match="center_freq_axis.*strictly monotonic"):
        center_freq_axis(axis)
    np.testing.assert_array_equal(axis, before)


@pytest.mark.parametrize("axis", [[], [[1.0, 2.0]], 1.0])
def test_empty_or_nonvector_axes_raise_actionable_error(axis):
    with pytest.raises(ValueError, match="center_freq_axis.*nonempty.*one-dimensional"):
        center_freq_axis(axis)


@pytest.mark.parametrize("axis", [[0.0, np.nan], [0.0, np.inf], [-np.inf, 0.0]])
def test_nonfinite_axes_raise_actionable_error(axis):
    with pytest.raises(ValueError, match="center_freq_axis.*finite"):
        center_freq_axis(axis)


@pytest.mark.parametrize("axis", [None, ["1", "2"], [1j, 2j], [False, True]])
def test_nonreal_or_nonnumeric_axes_raise_actionable_error(axis):
    with pytest.raises(TypeError, match="center_freq_axis.*real numeric"):
        center_freq_axis(axis)


@pytest.mark.parametrize("reverse_hz", [False, True])
@pytest.mark.parametrize("reverse_ppm", [False, True])
@pytest.mark.parametrize("flip_ppm", [False, True])
def test_centered_calibration_keeps_sign_for_each_axis_order(
    reverse_hz, reverse_ppm, flip_ppm
):
    hz = np.array([-200.0, -100.0, 0.0, 100.0])
    ppm = np.array([168.0, 169.0, 170.0, 171.0])
    if reverse_hz:
        hz = hz[::-1]
    if reverse_ppm:
        ppm = ppm[::-1]
    hz_before, ppm_before = hz.copy(), ppm.copy()
    sign = -1.0 if reverse_hz ^ reverse_ppm ^ flip_ppm else 1.0
    values = np.array([[-50.0, 0.0, 50.0]])
    actual = freq_hz_to_ppm(
        values,
        hz,
        ppm,
        ppm_axis_flipped=flip_ppm,
        ppm_centered_at_0=True,
    )
    np.testing.assert_allclose(actual, sign * values / 100, rtol=1e-12, atol=1e-12)
    # Default absolute calibration retains its existing paired-axis intercept.
    absolute = freq_hz_to_ppm(values, hz, ppm, ppm_axis_flipped=flip_ppm)
    intercept = 170.0 if sign > 0 else 169.0
    np.testing.assert_allclose(
        absolute, intercept + sign * values / 100, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(hz, hz_before)
    np.testing.assert_array_equal(ppm, ppm_before)


@pytest.mark.parametrize("bad_axis", ["hz", "ppm"])
def test_centered_calibration_rejects_nonmonotonic_axis(bad_axis):
    hz = np.array([-200.0, -100.0, 0.0, 100.0])
    ppm = np.array([168.0, 169.0, 170.0, 171.0])
    if bad_axis == "hz":
        hz = hz[[0, 2, 1, 3]]
    else:
        ppm = ppm[[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="center_freq_axis.*strictly monotonic"):
        freq_hz_to_ppm(0.0, hz, ppm, ppm_centered_at_0=True)


def test_metadata_centering_handles_descending_ppm_without_changing_defaults():
    metadata = {"center_ppm": 170.0, "dwell_time_s": 0.001}
    context = SimpleNamespace(
        bandwidth_hz=1000.0,
        bandwidth_ppm=-10.0,
        n_acquisition=4,
        get=metadata.get,
    )
    np.testing.assert_allclose(
        get_freq_axis(context, unit="ppm"),
        [175.0, 172.5, 170.0, 167.5],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        get_freq_axis(context, unit="ppm", centered_at_0=True),
        [3.75, 1.25, -1.25, -3.75],
        rtol=1e-12,
        atol=1e-12,
    )
    assert metadata == {"center_ppm": 170.0, "dwell_time_s": 0.001}


@pytest.mark.parametrize("keep_attrs", [False, True])
def test_labelled_axis_preserves_coordinates_and_existing_attribute_policy(keep_attrs):
    axis = xr.DataArray(
        np.array([100.0, 0.0, -100.0, -200.0], dtype=np.float32),
        dims="sample",
        coords={"sample": [4, 3, 2, 1]},
        name="frequency",
        attrs={"unit": "Hz", "provenance": "invented axis"},
    )
    before = axis.copy(deep=True)
    expected = xr.DataArray(
        np.array([150.0, 50.0, -50.0, -150.0], dtype=np.float32),
        dims="sample",
        coords={"sample": [4, 3, 2, 1]},
        name="frequency",
        attrs=axis.attrs.copy() if keep_attrs else {},
    )
    with xr.set_options(keep_attrs=keep_attrs):
        result = center_freq_axis(axis)
    xr.testing.assert_identical(result, expected)
    xr.testing.assert_identical(axis, before)
