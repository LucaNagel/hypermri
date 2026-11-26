"""Empirical half-rise regression references for invented mM curves and absolute seconds.

An independent analytic log-logistic curve fixes the half-rise time at 72 s.
The initial vector equals the known parameters to isolate the output contract
from timing identifiability. Absolute tolerance 1e-6 s covers optimizer rounding
and is far below the 2--4 s sampling intervals. The recovery ensemble separately
checks noisy recovery on uniform and irregular grids.
"""

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri.analysis.dce.fitting_support import fit_empirical_log_logistic_model
from hypermri.analysis.dce.models.empirical_log_logistic import (
    empirical_log_logistic_model,
    fit_empirical_log_logistic_curve,
)

TIME_S = np.r_[50.0, 54.0, 58.0, np.arange(60.0, 102.0, 2.0), 106.0, 110.0]
PARAMETERS = np.array([62.0, 10.0, 4.0, 0.01, 0.5, 0.03])
TAU_S = np.maximum(TIME_S - PARAMETERS[0], 0)
REFERENCE_MM = PARAMETERS[5] + PARAMETERS[4] * (
    TAU_S ** PARAMETERS[2] / (PARAMETERS[1] ** PARAMETERS[2] + TAU_S ** PARAMETERS[2])
) * np.exp(-PARAMETERS[3] * TAU_S)


def test_curve_reports_absolute_half_rise_of_rise_term_before_baseline_and_washout():
    curve = REFERENCE_MM.copy()
    result = fit_empirical_log_logistic_curve(
        curve, time_points_s=TIME_S, fit_params={"x0": PARAMETERS}
    )
    assert result["half_rise_time_s"] == pytest.approx(72.0, abs=1e-6)
    assert result["half_rise_time_s"] == pytest.approx(
        result["bolus_arrival_s"] + result["rise_time_s"], abs=1e-12
    )
    assert isinstance(result["half_rise_time_s"], float)
    predicted_at_half = empirical_log_logistic_model(
        np.array([result["half_rise_time_s"]]),
        result["bolus_arrival_s"],
        result["rise_time_s"],
        result["rise_shape"],
        result["washout_rate_s"],
        result["peak_amplitude"],
        result["baseline_offset"],
    )
    assert predicted_at_half.item() == pytest.approx(
        0.03 + 0.25 * np.exp(-0.01 * 10), abs=1e-10
    )
    assert result["fit_success"]
    np.testing.assert_array_equal(curve, REFERENCE_MM)


@pytest.mark.parametrize("cause", ["short", "nonfinite", "window"])
def test_missing_fit_returns_nan_half_rise_with_existing_failure_contract(cause):
    time = TIME_S[:5] if cause == "short" else TIME_S
    curve = REFERENCE_MM[:5].copy() if cause == "short" else REFERENCE_MM.copy()
    options = {"fit_end_index": 5} if cause == "window" else {}
    if cause == "nonfinite":
        curve[:] = np.nan
    result = fit_empirical_log_logistic_curve(
        curve, time_points_s=time, fit_params=options
    )
    assert np.isnan(result["half_rise_time_s"])
    assert not result["fit_success"]
    assert np.isnan(result["bolus_arrival_s"])
    assert np.isnan(result["rise_time_s"])


@pytest.mark.parametrize("temporal_first", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_half_rise_map_keeps_grid_units_precision_and_failure_values(
    temporal_first, parallel
):
    data = xr.DataArray(
        np.stack([REFERENCE_MM, np.full(len(TIME_S), np.nan)]),
        dims=("read_idx", "repetition_idx"),
        coords={
            "read_idx": [4.0, -4.0],
            "repetition_idx": np.arange(8, 8 + len(TIME_S)),
            "time_s": ("repetition_idx", TIME_S),
        },
        attrs={"units": "mM", "fixture_provenance": "invented analytic curve"},
    )
    data.read_idx.attrs["units"] = "mm"
    if temporal_first:
        data = data.transpose("repetition_idx", "read_idx")
    original = data.copy(deep=True)
    with parallel_config(backend="threading"):
        result = fit_empirical_log_logistic_model(
            data,
            fit_params={"x0": PARAMETERS, "number_of_cpu_cores": 2},
            use_multiprocessing=parallel,
        )
    half = result["half_rise_time_s"]
    assert half.dims == ("read_idx",)
    assert half.dtype == np.float64
    np.testing.assert_array_equal(half.read_idx, data.read_idx)
    assert half.read_idx.attrs == data.read_idx.attrs
    assert half.sel(read_idx=4.0).item() == pytest.approx(72, abs=1e-6)
    assert np.isnan(half.sel(read_idx=-4.0).item())
    assert half.attrs["units"] == "s"
    assert "rise term" in half.attrs["description"]
    assert result["fit_success"].values.tolist() == [True, False]
    xr.testing.assert_identical(data, original)
