"""Approved signal-space T2 model on synthetic echoes, with no scanner inputs.

Noiseless S0=2, T2=80 ms at TE=10..160 ms recovers T2 to 1e-6 ms and S0
to 1e-8 signal units: solver/float64 rounding allowances, not clinical claims.
Route/parallel comparisons use rtol=1e-12, atol=1e-14 structurally.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.fitting.models.quantitative import t2
from hypermri.analysis.quantitative import T2FitOptions, fit_t2


def _echo_data():
    times = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
    values = np.stack([2 * np.exp(-times / 0.08), 1.5 * np.exp(-times / 0.12)])
    data = xr.DataArray(
        values,
        dims=("voxel", "echo_idx"),
        coords={"voxel": [4, -2], "echo_idx": times},
        attrs={"unit": "a.u.", "synthetic": True},
    )
    data.echo_idx.attrs["unit"] = "s"
    return data


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("complex_storage", [False, True])
def test_t2_recovery_labels_reconstruction_and_preservation(transpose, complex_storage):
    data = _echo_data()
    if transpose:
        data = data.transpose("echo_idx", "voxel")
    if complex_storage:
        data = data * np.exp(0.4j)
        data.echo_idx.attrs["unit"] = "s"
    before = data.copy(deep=True)
    options = T2FitOptions(execution=ExecutionConfig(False, 1))
    result = fit_t2(data, options)
    np.testing.assert_allclose(result.t2, [80, 120], rtol=0, atol=1e-6)
    np.testing.assert_allclose(result.s0, [2, 1.5], rtol=0, atol=1e-8)
    assert result.t2.attrs["unit"] == "ms"
    assert result.rss_t2.attrs["unit"] == "(a.u.)^2"
    assert result.attrs["fit_uncertainty"] == "not calculated"
    assert result.attrs["minimum_valid_echoes"] == 3
    assert result.attrs["fit_package_version"]
    prediction = result.hp.reconstruct()
    xr.testing.assert_allclose(
        prediction, abs(data).transpose("voxel", "echo_idx"), rtol=1e-8, atol=1e-10
    )
    xr.testing.assert_allclose(
        result.hp.reconstruct(voxel_coords={"voxel": 0}), prediction.isel(voxel=0)
    )
    np.testing.assert_allclose(result.rss_t2, 0, atol=1e-18)
    xr.testing.assert_identical(data, before)
    with pytest.raises(FrozenInstanceError):
        options.signal_threshold = 1


@pytest.mark.parametrize("sampling_keyword", ["dynamic_dim", "echo_times_s"])
def test_t2_reconstruction_accepts_replacement_echo_times_in_seconds(sampling_keyword):
    result = fit_t2(_echo_data(), T2FitOptions(execution=ExecutionConfig(False, 1)))
    before = result.copy(deep=True)
    dense_times = np.linspace(0.01, 0.16, 17)
    prediction = result.hp.reconstruct(
        voxel_coords={"voxel": 1}, **{sampling_keyword: dense_times}
    )
    assert prediction.dims == ("echo_idx",)
    np.testing.assert_array_equal(prediction.echo_idx, dense_times)
    np.testing.assert_allclose(
        prediction,
        result.s0[1].item() * np.exp(-dense_times / (result.t2[1].item() / 1000)),
    )
    xr.testing.assert_identical(result, before)


def test_t2_reconstruction_retains_named_unlabelled_echo_axis():
    result = fit_t2(_echo_data(), T2FitOptions(execution=ExecutionConfig(False, 1)))
    times = np.linspace(0.01, 0.16, 17)
    prediction = result.hp.reconstruct(echo_times_s=times, dynamic_dim="echo_time_s")
    assert prediction.dims == ("voxel", "echo_time_s")
    np.testing.assert_array_equal(prediction.echo_time_s, times)


def test_t2_serial_threaded_and_generic_routes_agree():
    data = _echo_data()
    serial = fit_t2(data, T2FitOptions(execution=ExecutionConfig(False, 1)))
    with parallel_config(backend="threading"):
        threaded = fit_t2(data, T2FitOptions(execution=ExecutionConfig(True, 2)))
    for result in (
        api.fit(data, "t2", {"use_multiprocessing": False}),
        api.fit(
            data,
            "monoexponential_t2",
            api.prepare_fit_params(data, "t2", {"use_multiprocessing": False}),
        ),
        threaded,
    ):
        xr.testing.assert_allclose(result, serial, rtol=1e-12, atol=1e-14)
        assert result.t2.dims == ("voxel",)


@pytest.mark.parametrize(
    "values", [[0, 0, 0, 0, 0], [np.nan] * 5, [np.inf] * 5, [1, 0.5, 0, 0, 0]]
)
def test_t2_invalid_or_insufficient_echo_signal_yields_nan(values):
    data = _echo_data().isel(voxel=0).copy()
    data.values[:] = values
    result = fit_t2(data, T2FitOptions(execution=ExecutionConfig(False, 1)))
    assert all(np.isnan(result[name]).all() for name in ("t2", "s0", "rss_t2"))


def test_t2_threshold_omits_invalid_echoes_and_rss_is_signal_space():
    data = _echo_data().isel(voxel=0).copy()
    data.values[-1] = 1e-5
    result = fit_t2(data, T2FitOptions(execution=ExecutionConfig(False, 1)))
    assert result.t2.item() == pytest.approx(80, abs=1e-6)
    noisy = _echo_data().isel(voxel=0).copy()
    noisy.values[:] += [0.01, -0.02, 0.01, 0.02, -0.01]
    fitted = fit_t2(noisy, T2FitOptions(execution=ExecutionConfig(False, 1)))
    expected = float(np.sum((fitted.hp.reconstruct().values - noisy.values) ** 2))
    assert fitted.rss_t2.item() == pytest.approx(expected, rel=1e-12, abs=1e-14)
    # The approved signal-space optimum improves on the log-linear solution.
    slope, intercept = np.polyfit(noisy.echo_idx.values, np.log(noisy.values), 1)
    log_fit_rss = np.sum(
        (np.exp(intercept + slope * noisy.echo_idx.values) - noisy.values) ** 2
    )
    assert fitted.rss_t2.item() < log_fit_rss


@pytest.mark.parametrize(
    "change", ["unit", "negative", "nonfinite", "short", "constant"]
)
def test_t2_rejects_invalid_echo_coordinates(change):
    data = _echo_data()
    if change == "unit":
        data.echo_idx.attrs["unit"] = "ms"
    elif change == "negative":
        data = data.assign_coords(
            echo_idx=("echo_idx", [-1, 0.02, 0.04, 0.08, 0.16], {"unit": "s"})
        )
    elif change == "nonfinite":
        data = data.assign_coords(
            echo_idx=("echo_idx", [np.nan, 0.02, 0.04, 0.08, 0.16], {"unit": "s"})
        )
    elif change == "short":
        data = data.isel(echo_idx=slice(2))
    elif change == "constant":
        data = data.assign_coords(echo_idx=("echo_idx", [0.01] * 5, {"unit": "s"}))
    with pytest.raises(ValueError):
        fit_t2(data)


@pytest.mark.parametrize(
    "options",
    [
        {"signal_threshold": -1},
        {"signal_threshold": np.nan},
        {"number_of_cpu_cores": 0},
        {"offset": True},
        {"dynamic_dim": 3},
    ],
)
def test_t2_rejects_invalid_options_before_fit(options, monkeypatch):
    monkeypatch.setattr(t2, "_fit_decay", lambda *a: pytest.fail("fitter called"))
    with pytest.raises((TypeError, ValueError)):
        fit_t2(_echo_data(), options)


def test_t2_nonconvergence_is_nan_and_info_mode_does_not_fit(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        t2,
        "least_squares",
        lambda *a, **k: SimpleNamespace(success=False, x=np.array([2, 0.08])),
    )
    assert fit_t2(_echo_data(), return_params_info=True) is None
    result = fit_t2(_echo_data(), T2FitOptions(execution=ExecutionConfig(False, 1)))
    assert np.isnan(result.t2).all()


@pytest.mark.parametrize("info", [False, True])
def test_t2_prepared_path_validates_sampling_and_required_settings(info):
    data = _echo_data()
    prepared = api.prepare_fit_params(data, "t2", {"use_multiprocessing": False})
    data.echo_idx.attrs["unit"] = "ms"
    with pytest.raises(ValueError, match="unit"):
        fit_t2(data, prepared, return_params_info=info)
    with pytest.raises(ValueError, match="Prepared T2"):
        fit_t2(_echo_data(), xr.Dataset(), return_params_info=info)


def test_t2_rejects_invalid_execution_container():
    with pytest.raises(TypeError, match="ExecutionConfig"):
        fit_t2(_echo_data(), T2FitOptions(execution="invalid"))


@pytest.mark.parametrize("curve", ["slow_decay", "flat", "increasing"])
def test_t2_fit_is_bounded_at_five_seconds(curve):
    """A 5 s optimizer bound is 5000 ms in the stored map, not a display cap."""
    data = _echo_data().isel(voxel=0).copy()
    if curve == "slow_decay":
        data.values[:] = 2 * np.exp(-data.echo_idx.values / 300.0)
    elif curve == "flat":
        data.values[:] = 2
    else:
        data.values[:] = np.linspace(1, 2, data.size)
    result = fit_t2(data, T2FitOptions(execution=ExecutionConfig(False, 1)))
    assert np.isfinite(result.t2.item())
    assert 4999.0 < result.t2.item() <= 5000.0
    assert result.attrs["maximum_t2_s"] == 5.0
    prediction = result.hp.reconstruct()
    assert result.rss_t2.item() == pytest.approx(
        float(np.sum((prediction.values - data.values) ** 2)), rel=1e-12, abs=1e-14
    )
    # At the bound, S0 must be re-estimated by the signal-space objective.
    basis = np.exp(-data.echo_idx.values / (result.t2.item() / 1000))
    expected_s0 = np.dot(basis, data.values) / np.dot(basis, basis)
    assert result.s0.item() == pytest.approx(expected_s0, rel=1e-7, abs=1e-9)


def test_t2_bound_applies_to_generic_prepared_and_threaded_fits():
    data = _echo_data().copy()
    data.values[:] = 2
    prepared = api.prepare_fit_params(data, "t2", {"use_multiprocessing": False})
    results = [
        api.fit(data, "t2", prepared),
        api.fit(data, "t2", {"use_multiprocessing": False}),
    ]
    with parallel_config(backend="threading"):
        results.append(fit_t2(data, T2FitOptions(execution=ExecutionConfig(True, 2))))
    for result in results:
        assert np.isfinite(result.t2).all()
        assert float(result.t2.max()) <= 5000.0
        assert result.attrs["maximum_t2_s"] == 5.0
