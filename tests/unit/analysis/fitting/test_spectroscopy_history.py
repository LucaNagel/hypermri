"""Reconstruct full optimization states and test numerical status across backends."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import api, spectroscopy
from hypermri.analysis.fitting.models.spectroscopy import fid_model as model
from hypermri.analysis.fitting.models.spectroscopy.diagnostics import HISTORY_EVENTS


def problem(*, history=True, domain="temporal", magnitude=False, threshold=None):
    """Small complex, noisy signal with a manual exclusion between two active peaks."""
    time = np.arange(32) * 0.001
    frequencies = np.array([-35.0, 90.0, 210.0])
    amplitudes = np.array([1.0 + 0.4j, 0.0, 0.018 - 0.006j])
    basis = np.exp(2j * np.pi * time[:, None] * frequencies - time[:, None] / 0.025)
    rng = np.random.default_rng(1295)
    fid = basis @ amplitudes + 0.001 * (rng.normal(size=32) + 1j * rng.normal(size=32))
    data = xr.DataArray(
        fid if domain == "temporal" else np.fft.fftshift(np.fft.fft(fid)),
        dims="acquisition_idx",
        coords={"acquisition_idx": time},
        attrs={"signal_domain": domain},
    )
    params = spectroscopy.def_fit_params(
        data,
        {
            "metabs": ["a", "masked", "b"],
            "metabs_freqs_hz": frequencies + 2,
            "initial_t2s_s": [0.025] * 3,
            "skip_fitting": [False, True, False],
            "metabs_freq_ranges": [30, 0, 40],
            "metabs_t2_ranges": [0.06, 0, 0.07],
            "min_t2_s": [0.015] * 3,
            "max_t2_s": [0.035] * 3,
            "npoints": 4,
            "niter": 2,
            "rep_fitting": 2,
            "use_sse_mag": magnitude,
            "snr_threshold": threshold,
            "store_optimization_history": history,
        },
    )
    return data, params, fid


@pytest.mark.parametrize("domain", ["temporal", "spectral"])
@pytest.mark.parametrize("magnitude", [False, True])
@pytest.mark.parametrize("threshold", [None, 0.1])
def test_history_reconstructs_every_evaluated_state(domain, magnitude, threshold):
    data, params, fid = problem(domain=domain, magnitude=magnitude, threshold=threshold)
    original_data, original_params = data.copy(deep=True), params.copy(deep=True)
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert result.fit_status.item() == "success"
    count = result.fit_history_count.item()
    history = result.isel(history_step=slice(0, count))
    assert (
        history.fit_history_info.sel(history_field="event")[-1]
        == HISTORY_EVENTS["returned"]
    )
    np.testing.assert_array_equal(history.fit_history_amps[-1], result.fit_amps)
    np.testing.assert_array_equal(history.fit_history_freqs[-1], result.fit_freqs)
    np.testing.assert_array_equal(history.fit_history_t2s[-1], result.fit_t2s)
    assert history.fit_history_rss[-1] == result.fit_rss
    assert history.fit_history_objective[-1] == result.fit_objective
    assert np.any(history.fit_history_info.sel(history_field="rejection_reason") == 1)
    for index in range(count):
        state = history.isel(history_step=index)
        info = state.fit_history_info
        if not info.sel(history_field="evaluated"):
            continue
        amps, freqs, t2s = (
            state[f"fit_history_{name}"].values for name in ("amps", "freqs", "t2s")
        )
        basis = np.exp(
            2j * np.pi * data.acquisition_idx.values[:, None] * freqs
            - data.acquisition_idx.values[:, None] / t2s
        )
        fitted = basis @ amps
        active = state.fit_history_active.values
        np.testing.assert_array_equal(amps[~active], 0)
        reference = np.linalg.lstsq(basis[:, active], fid, rcond=None)[0]
        np.testing.assert_allclose(amps[active], reference, rtol=1e-10, atol=1e-12)
        rss = np.sum(np.abs(fid - fitted) ** 2)
        objective = (
            np.sum((np.abs(np.fft.fft(fid)) - np.abs(np.fft.fft(fitted))) ** 2)
            if magnitude
            else rss
        )
        assert state.fit_history_rss.item() == pytest.approx(rss, abs=1e-12)
        assert state.fit_history_objective.item() == pytest.approx(objective, abs=1e-10)
        assert np.all(freqs[active] >= params.freq_range_lower_bound_hz.values[active])
        assert np.all(freqs[active] <= params.freq_range_upper_bound_hz.values[active])
        assert np.all(t2s[active] >= params.min_t2_s.values[active])
        assert np.all(t2s[active] <= params.max_t2_s.values[active])

    info = history.fit_history_info
    events = info.sel(history_field="event")
    outer = np.isin(
        events,
        [
            HISTORY_EVENTS[name]
            for name in ("initial", "frequency_step", "t2_step", "verified")
        ],
    )
    accepted = info.sel(history_field="accepted").values == 1
    for fit_round in (0, 1):
        selected = (
            outer & accepted & (info.sel(history_field="fit_round").values == fit_round)
        )
        assert np.all(np.diff(history.fit_history_objective.values[selected]) <= 1e-10)
    assert np.all(result.fit_history_info.values[count:] == -1)
    assert np.all(np.isnan(result.fit_history_amps.values[count:]))
    if threshold is not None:
        assert bool(result.fit_low_snr_refitted.sel(metabolite="b"))
        assert np.any(info.sel(history_field="fit_round") == 1)
    assert not history.fit_history_active.sel(metabolite="masked").any()
    xr.testing.assert_identical(data, original_data)
    xr.testing.assert_identical(params, original_params)


def status_problem(history):
    data, params, _ = problem(history=history)
    voxels = ["good", "zero", "invalid", "skipped", "failed"]
    data = data.expand_dims(voxel=voxels).copy(deep=True)
    data.loc[{"voxel": "zero"}] = 0
    data.loc[{"voxel": "invalid", "acquisition_idx": data.acquisition_idx[3]}] = np.nan
    # Finite samples whose squared residuals overflow float64 fail numerically
    # in this spectrum only; start values closer than min_peak_separation_hz
    # no longer reach the fitter, because fit preparation rejects them.
    data.loc[{"voxel": "failed"}] = data.sel(voxel="failed") * 1e160
    params = spectroscopy.def_fit_params(data, params.attrs).copy(deep=True)
    params.skip_fitting.loc[{"voxel": "skipped"}] = True
    return data, params


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize(
    "backend", ["progress", "threading", "loky", "dask", "dask-single-worker"]
)
def test_numerical_status_and_history_match_all_backends(history, backend, monkeypatch):
    data, params = status_problem(history)
    expected = spectroscopy.fit_data_pseudo_inv(data, params)
    assert expected.fit_status.values.tolist() == [
        "success",
        "zero_signal",
        "invalid_input",
        "skipped",
        "fit_failed",
    ]
    assert expected.fit_failure_reason.values.tolist() == [
        "none",
        "none",
        "nonfinite_samples",
        "none",
        "nonfinite_objective",
    ]
    assert np.all(
        np.isnan(expected.fit_amps.sel(voxel="failed", metabolite=["a", "b"]))
    )
    assert expected.fit_amps.sel(voxel="failed", metabolite="masked") == 0
    if backend == "progress":
        result = spectroscopy.fit_data_pseudo_inv(data, params, show_tqdm=True)
    elif backend.startswith("dask"):
        pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        def unexpected_joblib(*args, **kwargs):
            pytest.fail("Dask execution must not create a nested joblib pool.")

        monkeypatch.setattr(
            spectroscopy, "_fit_data_pseudo_inv_with_joblib", unexpected_joblib
        )
        executed = []
        with Callback(pretask=lambda *args: executed.append(args[0])):
            with pytest.warns(UserWarning, match="Joblib options are ignored"):
                result = spectroscopy.fit_data_pseudo_inv(
                    data.chunk({"voxel": 2, "acquisition_idx": 8}),
                    params,
                    n_jobs=2,
                )
        assert not executed, "Constructing the lazy result must not execute any fits."
        assert result.fit_status.chunks is not None
        if history:
            assert result.fit_history_amps.chunks is not None
        result = result.compute(
            scheduler="threads",
            num_workers=1 if backend.endswith("single-worker") else 2,
        )
    else:
        result = spectroscopy.fit_data_pseudo_inv(
            data, params, n_jobs=2, parallel_backend=backend, batch_size=1
        )
    xr.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
    assert any(name.startswith("fit_history_") for name in result) == history


def test_single_spectrum_joblib_keeps_scalar_diagnostics():
    data, params, _ = problem()
    expected = api.fit(data, "FID", params)
    result = spectroscopy.fit_data_pseudo_inv(data, params, n_jobs=2)
    assert result.fit_status.dims == ()
    xr.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", ["linalg", "nonfinite", "verification"])
def test_numerical_failure_does_not_abort_neighbor(kind, monkeypatch):
    data, params, _ = problem()
    data = data.expand_dims(voxel=["ok", "bad"]).copy(deep=True)
    data.loc[{"voxel": "bad"}] *= 2
    params = spectroscopy.def_fit_params(
        data, {**params.attrs, "npoints": 1, "niter": 1, "rep_fitting": 1}
    )
    original = model._score_fit_parameters
    count = 0

    def score(values, freqs, t2s, options):
        nonlocal count
        objective, amps = original(values, freqs, t2s, options)
        if abs(values[0]) > 1.5:
            count += 1
            if kind == "linalg":
                raise np.linalg.LinAlgError("injected SVD failure")
            if kind == "nonfinite":
                return np.nan, amps
            # Calls: initial state, frequency step, T2* step, joint
            # refinement, final verification.
            if count == 5:
                return objective + 1.0, amps
        return objective, amps

    monkeypatch.setattr(model, "_score_fit_parameters", score)
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert result.fit_status.values.tolist() == ["success", "fit_failed"]
    assert (
        result.fit_failure_reason.sel(voxel="bad").item()
        == {
            "linalg": "linear_algebra_error",
            "nonfinite": "nonfinite_objective",
            "verification": "inconsistent_final_state",
        }[kind]
    )
    assert np.isnan(result.fit_rss.sel(voxel="bad"))
    assert result.fit_history_count.sel(voxel="bad") >= 1


def test_history_flag_is_validated_without_changing_manual_settings():
    data, params, _ = problem()
    with pytest.raises(ValueError, match="store_optimization_history must be boolean"):
        spectroscopy.def_fit_params(
            data, {**params.attrs, "store_optimization_history": "yes"}
        )
    edited = params.copy(deep=True)
    edited.attrs["store_optimization_history"] = "yes"
    with pytest.raises(ValueError, match="store_optimization_history must be boolean"):
        spectroscopy.fit_data_pseudo_inv(data, edited)
    plain = params.copy(deep=True)
    plain.attrs["store_optimization_history"] = False
    with_history = spectroscopy.fit_data_pseudo_inv(data, params)
    without_history = spectroscopy.fit_data_pseudo_inv(data, plain)
    for name in without_history:
        xr.testing.assert_identical(with_history[name], without_history[name])
