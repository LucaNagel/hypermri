"""Unit-contract tests for T1 fitting and reconstruction."""

import numpy as np
import xarray as xr

from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.models.quantitative.t1 import get_t1_params_info


def test_t1_grid_seconds_produces_millisecond_result_and_roundtrip():
    inversion_times_ms = np.array([80.0, 250.0, 600.0, 1200.0, 2400.0])
    t1_true_ms = 1200.0
    amplitude = 75.0
    signal = amplitude * (1.0 - 2.0 * np.exp(-inversion_times_ms / t1_true_ms))
    input_data = xr.DataArray(
        signal[None, None, :],
        dims=("read_idx", "phase_idx", "inversion_idx"),
        coords={
            "read_idx": [0],
            "phase_idx": [0],
            "inversion_idx": inversion_times_ms,
        },
    )
    input_data.coords["inversion_idx"].attrs["units"] = "ms"

    prepared = api.prepare_fit_params(
        input_data,
        "t1",
        {
            "t1_values_s": [0.8, 1.2, 1.6],
            "use_multiprocessing": False,
        },
    )
    result = api.fit(input_data, "t1", prepared)
    reconstructed = api.reconstruct_fit(result)

    np.testing.assert_allclose(prepared.attrs["t1_values_s"], [0.8, 1.2, 1.6])
    np.testing.assert_allclose(prepared.attrs["t1_values_ms"], [800.0, 1200.0, 1600.0])
    assert prepared.attrs["t1_grid_input_units"] == "s"
    assert prepared.attrs["t1_output_units"] == "ms"

    np.testing.assert_allclose(result["t1"], t1_true_ms)
    assert result["t1"].attrs["unit"] == "ms"
    assert result["t1"].attrs["units"] == "ms"
    assert result.attrs["t1_grid_input_units"] == "s"
    assert result.attrs["t1_output_units"] == "ms"
    assert result["inversion_times_ms"].attrs["units"] == "ms"
    np.testing.assert_allclose(
        reconstructed.transpose(*input_data.dims), input_data, atol=1e-12
    )


def test_t1_parameter_info_matches_millisecond_output_contract():
    info = get_t1_params_info()

    assert "seconds" in info["required_fit_params"]["t1_values_s"]
    assert "milliseconds" in info["output_parameters"]["t1"]
