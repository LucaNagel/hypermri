import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.frequency import (
    _convert_ppm_ndarray,
    _convert_ppm_value,
    _ppm_delta_to_hz,
)

HZ_AXIS = np.array([-200.0, -100.0, 0.0, 100.0])
PPM_AXIS = np.array([168.0, 169.0, 170.0, 171.0])


def test_convert_ppm_ndarray_preserves_shape():
    ppm_values = np.array([[168.0, 169.0], [170.0, 171.0]])

    converted = _convert_ppm_ndarray(
        ppm_values,
        hz_axis=HZ_AXIS,
        ppm_axis=PPM_AXIS,
    )

    assert converted.shape == ppm_values.shape
    assert np.allclose(converted, HZ_AXIS.reshape(2, 2))


def test_convert_ppm_value_preserves_supported_value_kinds():
    assert _convert_ppm_value(
        170.0,
        hz_axis=HZ_AXIS,
        ppm_axis=PPM_AXIS,
    ) == pytest.approx(0.0)

    ppm_data = xr.DataArray([169.0, 171.0], dims=["metabolite"])
    converted = _convert_ppm_value(
        ppm_data,
        hz_axis=HZ_AXIS,
        ppm_axis=PPM_AXIS,
    )
    assert isinstance(converted, np.ndarray)
    assert np.allclose(converted, [-100.0, 100.0])

    assert (
        _convert_ppm_value(
            None,
            hz_axis=HZ_AXIS,
            ppm_axis=PPM_AXIS,
        )
        is None
    )


def test_ppm_delta_to_hz_removes_axis_intercept_and_preserves_sign():
    assert _ppm_delta_to_hz(
        1.0,
        hz_axis=HZ_AXIS,
        ppm_axis=PPM_AXIS,
    ) == pytest.approx(100.0)

    descending_ppm_axis = PPM_AXIS[::-1]
    assert _ppm_delta_to_hz(
        1.0,
        hz_axis=HZ_AXIS,
        ppm_axis=descending_ppm_axis,
    ) == pytest.approx(-100.0)


def test_ppm_conversion_requires_both_calibration_axes():
    with pytest.raises(ValueError, match="Frequency axis information"):
        _convert_ppm_ndarray(
            np.array([170.0]),
            hz_axis=None,
            ppm_axis=PPM_AXIS,
        )
