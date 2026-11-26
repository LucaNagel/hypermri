"""Tests of the curve-fit tool (ADC, T2, T1 along the curve axis).

All fixtures are small synthetic noise-free signals; recovered parameters
are compared with a relative tolerance of 1e-3, which covers the log-linear
ADC fit, the least-squares T2 fit and the 1 ms T1 grid used here.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.visualization.curve_fit_tool import (
    CURVE_MODELS,
    CurveFitState,
    check_curve_axis,
    curve_fit_curves,
    curve_fit_settings,
    curve_fit_summary,
    curve_model,
    model_signal,
    run_curve_fit,
    set_curve_fit_option,
    voxel_parameters,
)
from hypermri.visualization.state import PlotState

SERIAL = ExecutionConfig(use_multiprocessing=False)


def _signal(dim, axis, unit, curve):
    """Return a 2x3 voxel signal along ``dim`` with value ``curve(axis, s0)``."""
    s0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    values = curve(np.asarray(axis)[None, None, :], s0[..., None])
    return xr.DataArray(
        values,
        dims=("y", "x", dim),
        coords={dim: (dim, np.asarray(axis, dtype=float), {"unit": unit})},
    )


def test_model_follows_curve_axis():
    data = _signal("bvalue_idx", [0, 500, 1000], "s/mm2", lambda b, s: s * 1.0 + 0 * b)
    assert curve_model(data, "bvalue_idx").key == "adc"
    assert curve_model(data, "echo_idx") is None
    assert curve_model(data, "x") is None
    assert curve_model(data, None) is None
    assert {m.key for m in CURVE_MODELS.values()} == {"adc", "t2", "t1"}


def test_adc_fit_recovers_diffusion_coefficient():
    adc = 1.5e-3  # mm²/s
    data = _signal(
        "bvalue_idx", [0, 200, 500, 1000], "s/mm2", lambda b, s: s * np.exp(-b * adc)
    )
    model = curve_model(data, "bvalue_idx")
    result = run_curve_fit(data, model, CurveFitState(), execution=SERIAL)
    np.testing.assert_allclose(result["adc"].values, adc, rtol=1e-3)
    np.testing.assert_allclose(result["s0"].values[1, 2], 6.0, rtol=1e-3)


def test_t2_fit_recovers_decay_time():
    t2_ms = 40.0
    data = _signal(
        "echo_idx",
        [0.01, 0.02, 0.04, 0.08],
        "s",
        lambda t, s: s * np.exp(-t / (t2_ms / 1000.0)),
    )
    model = curve_model(data, "echo_idx")
    result = run_curve_fit(data, model, CurveFitState(), execution=SERIAL)
    np.testing.assert_allclose(result["t2"].values, t2_ms, rtol=1e-3)


def test_t1_fit_recovers_recovery_time_with_baseline_option():
    t1_ms = 1200.0
    data = _signal(
        "inversion_idx",
        [50.0, 300.0, 800.0, 1500.0, 3000.0, 6000.0],
        "ms",
        lambda ti, s: s * (1 - 2 * np.exp(-ti / t1_ms)),
    )
    model = curve_model(data, "inversion_idx")
    state = CurveFitState()
    set_curve_fit_option(state, "fit_baseline", True)
    assert state.fit_baseline is True
    result = run_curve_fit(data, model, state, execution=SERIAL)
    # The default trial grid is 10 ms .. 10 s in 500 steps (about 20 ms).
    np.testing.assert_allclose(result["t1"].values, t1_ms, rtol=0.02)


@pytest.mark.parametrize(
    ("coord", "message"),
    [
        (("echo_idx", [0.01, 0.02, 0.04], {"unit": "ms"}), "unit"),
        (("echo_idx", [0.01, 0.01, 0.04], {"unit": "s"}), "distinct"),
        (("echo_idx", [0.01, np.nan, 0.04], {"unit": "s"}), "finite"),
    ],
)
def test_axis_check_names_the_problem(coord, message):
    data = xr.DataArray(
        np.ones((2, 3)), dims=("x", "echo_idx"), coords={"echo_idx": coord}
    )
    with pytest.raises(ValueError, match=message):
        check_curve_axis(data, CURVE_MODELS["echo_idx"])


def test_axis_check_needs_a_coordinate():
    data = xr.DataArray(np.ones((2, 3)), dims=("x", "echo_idx"))
    with pytest.raises(ValueError, match="no coordinate 'echo_idx'"):
        run_curve_fit(data, CURVE_MODELS["echo_idx"], CurveFitState())


def test_unknown_option_is_rejected():
    with pytest.raises(ValueError, match="unknown setting"):
        set_curve_fit_option(CurveFitState(), "t1_values_s", 1.0)


def test_model_signal_uses_the_model_equations():
    x = np.array([0.0, 1000.0])
    adc = model_signal(CURVE_MODELS["bvalue_idx"], {"s0": 2.0, "adc": 1e-3}, x)
    np.testing.assert_allclose(adc, [2.0, 2.0 * np.exp(-1.0)])
    t2 = model_signal(CURVE_MODELS["echo_idx"], {"s0": 1.0, "t2": 50.0}, [0.05])
    np.testing.assert_allclose(t2, [np.exp(-1.0)])
    t1 = model_signal(
        CURVE_MODELS["inversion_idx"], {"A": 1.0, "B": 0.5, "t1": 100.0}, [0.0]
    )
    np.testing.assert_allclose(t1, [-0.5])


def test_curves_hold_dense_fit_and_residual_below_signal():
    model = CURVE_MODELS["bvalue_idx"]
    x = np.array([0.0, 500.0, 1000.0])
    parameters = {"s0": 2.0, "adc": 1e-3, "rss_adc": 0.0}
    measured = model_signal(model, parameters, x) + np.array([0.0, 0.1, 0.0])
    curves = curve_fit_curves(model, parameters, x, measured, PlotState())
    names = [curve["name"] for curve in curves]
    assert names[0].startswith("Fit")
    assert len(curves[0]["x"]) == len(curves[0]["y"]) == 200
    assert names[1].startswith("Residual") and curves[2]["dash"] == "dash"
    residual = np.array(curves[1]["y"])
    # The residual keeps its shape and lies below the lowest curve sample.
    np.testing.assert_allclose(np.diff(residual), [0.1, -0.1], atol=1e-12)
    assert residual.max() < measured.min()


def test_failed_voxel_gives_no_curves():
    model = CURVE_MODELS["bvalue_idx"]
    parameters = {"s0": np.nan, "adc": np.nan}
    assert (
        curve_fit_curves(model, parameters, [0.0, 1.0], [1.0, 1.0], PlotState()) == []
    )


def test_voxel_parameters_summary_and_settings():
    result = xr.Dataset(
        {
            "adc": (("y", "x"), [[1e-3, np.nan], [2e-3, 3e-3]]),
            "s0": (("y", "x"), np.ones((2, 2))),
        }
    )
    assert voxel_parameters(result, {"y": 1, "x": 0}) == {"adc": 2e-3, "s0": 1.0}
    assert voxel_parameters(result, {})["adc"] == 1e-3
    model = CURVE_MODELS["bvalue_idx"]
    assert "3 of 4 voxels" in curve_fit_summary(model, result)
    assert "0.002 mm²/s" in curve_fit_summary(model, result)
    settings = curve_fit_settings(CurveFitState(), model)
    assert settings["offered"] and settings["model"] == "adc"
    assert curve_fit_settings(CurveFitState(), None) == {
        "fit_baseline": False,
        "message": "",
        "offered": False,
    }
