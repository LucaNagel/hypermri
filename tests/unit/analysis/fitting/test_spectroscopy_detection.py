"""Independent reconstruction, calibration bookkeeping and execution parity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import api, spectroscopy
from hypermri.analysis.fitting.models.spectroscopy import model_comparison as comparison
from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _update_frequency_bounds,
)


def problem(*, fixed=False, domain="temporal", n_null=19, history=False):
    time = np.arange(32) * 0.001
    true_freq = np.array([-31.0, 145.0])
    basis = np.exp(2j * np.pi * time[:, None] * true_freq - time[:, None] / 0.025)
    fid = basis @ np.array([1.0 + 0.7j, 0.2 - 0.3j])
    rng = np.random.default_rng(9172)  # independent of calibration seed
    fid += 0.03 * (rng.standard_normal(32) + 1j * rng.standard_normal(32))
    data = xr.DataArray(
        fid if domain == "temporal" else np.fft.fftshift(np.fft.fft(fid)),
        dims="acquisition_idx",
        coords={"acquisition_idx": time},
        attrs={"signal_domain": domain},
    )
    params = spectroscopy.def_fit_params(
        data,
        {
            "metabs": ["Pyruvate", "Bicarbonate"],
            "metabs_freqs_hz": true_freq + (0 if fixed else 3),
            "initial_t2s_s": [0.025, 0.025],
            "metabs_freq_ranges": 0 if fixed else 18,
            "metabs_t2_ranges": 0 if fixed else 0.02,
            "npoints": 3,
            "rep_fitting": 1,
            "detection_target": "Bicarbonate",
            "detection_sigma": 0.03,
            "detection_null_samples": n_null,
            "detection_seed": 85411,
            "store_optimization_history": history,
        },
    )
    return data, params


def reconstruct(result, time, *, m0=False):
    prefix = "m0" if m0 else "fit"
    basis = np.exp(
        2j * np.pi * time[:, None] * result[f"{prefix}_freqs"].values
        - time[:, None] / result[f"{prefix}_t2s"].values
    )
    return basis @ result[f"{prefix}_amps"].values


@pytest.mark.parametrize("domain", ["temporal", "spectral"])
def test_observed_models_are_full_refits_on_same_samples(domain):
    data, params = problem(domain=domain, history=True)
    params.attrs["nsamplepoints"] = 24
    params.attrs["sampling_time_axis_s"] = params.attrs["sampling_time_axis_s"][:24]
    original = params.copy(deep=True)
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    numerical = params.copy(deep=True)
    numerical.attrs = {
        k: v for k, v in numerical.attrs.items() if not k.startswith("detection_")
    }
    numerical.skip_fitting.loc[{"metabolite": "Bicarbonate"}] = True
    independently_fitted_m0 = spectroscopy.fit_data_pseudo_inv(data, numerical)
    np.testing.assert_allclose(result.m0_amps, independently_fitted_m0.fit_amps)
    np.testing.assert_allclose(result.m0_freqs, independently_fitted_m0.fit_freqs)
    np.testing.assert_allclose(result.m0_t2s, independently_fitted_m0.fit_t2s)
    fid = (
        data.values
        if domain == "temporal"
        else np.fft.ifft(np.fft.ifftshift(data.values))
    )
    for m0 in (False, True):
        model = reconstruct(result, params.attrs["sampling_time_axis_s"], m0=m0)
        rss = np.sum(np.abs(fid[:24] - model) ** 2)
        assert result["rss_m0" if m0 else "rss_m1"].item() == pytest.approx(rss)
    assert result.test_statistic_raw.item() == pytest.approx(
        (result.rss_m0.item() - result.rss_m1.item()) / 0.03**2
    )
    assert result.detection_status.item() == "detected"
    assert result.accepted.values.tolist() == [False, True]
    assert np.isnan(result.fit_amps_accepted.values[0])
    assert result.fit_amps_accepted.values[1] == result.fit_amps.values[1]
    last = result.fit_history_count.item() - 1
    np.testing.assert_array_equal(
        result.fit_history_amps.isel(history_step=last), result.fit_amps
    )
    np.testing.assert_array_equal(
        result.fit_history_freqs.isel(history_step=last), result.fit_freqs
    )
    xr.testing.assert_identical(params, original)


def test_finite_calibration_matches_independent_complex_lstsq_reference():
    """Fixed f/T2 here is a manually selected oracle test, never an API restriction."""
    data, params = problem(fixed=True)
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    t = data.acquisition_idx.values
    basis = np.exp(
        2j * np.pi * t[:, None] * params.initial_freqs_hz.values - t[:, None] / 0.025
    )
    null_basis = basis[:, :1]
    fit0 = null_basis @ np.linalg.lstsq(null_basis, data.values, rcond=None)[0]
    fit1 = basis @ np.linalg.lstsq(basis, data.values, rcond=None)[0]

    def rss(y, yhat):
        return np.sum(np.abs(y - yhat) ** 2)

    observed = (rss(data.values, fit0) - rss(data.values, fit1)) / 0.03**2
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([85411, 0])))
    statistics = []
    for _ in range(19):
        y = fit0 + 0.03 * (rng.standard_normal(32) + 1j * rng.standard_normal(32))
        r0 = rss(y, null_basis @ np.linalg.lstsq(null_basis, y, rcond=None)[0])
        r1 = rss(y, basis @ np.linalg.lstsq(basis, y, rcond=None)[0])
        statistics.append((r0 - r1) / 0.03**2)
    k = np.count_nonzero(np.asarray(statistics) >= observed)
    assert result.detection_n_extreme.item() == k == 0
    assert result.detection_p_value.item() == (k + 1) / 20 == 0.05
    assert result.detection_mc_tail_low.item() == 0
    assert result.detection_mc_tail_high.item() > 0.05
    assert result.detection_n_null.item() == result.detection_n_valid_null.item() == 19
    std = 0.03 * np.sqrt(np.diag(np.linalg.inv(basis.conj().T @ basis)).real)
    np.testing.assert_allclose(result.amplitude_std_quadrature_conditional, std)
    np.testing.assert_allclose(
        result.amplitude_snr_conditional, np.abs(result.fit_amps) / std
    )
    assert (
        not result.fit_bound_hit.any()
    )  # manually frozen parameters are not boundary hits


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"detection_sigma": None}, "detection_sigma"),
        ({"detection_sigma": -1}, "detection_sigma"),
        ({"detection_sigma": np.inf}, "detection_sigma"),
        ({"detection_sigma": 1e300}, "detection_sigma"),
        ({"detection_sigma": 1e-300}, "detection_sigma"),
        ({"detection_sigma": [0.1, 0.2]}, "detection_sigma"),
        ({"detection_null_samples": 0}, "detection_null_samples"),
        ({"detection_null_samples": 3.5}, "detection_null_samples"),
        ({"detection_seed": None}, "detection_seed"),
        ({"detection_seed": True}, "detection_seed"),
        ({"detection_alpha": 1}, "detection_alpha"),
        ({"detection_target": ["Pyruvate", "Bicarbonate"]}, "exactly one"),
        ({"detection_target": "absent"}, "exactly one"),
        ({"snr_threshold": 0.0}, "snr_threshold=None"),
        ({"use_sse_mag": True}, "complex RSS"),
        ({"amp_solver": "ridge"}, "unregularized"),
        ({"detection_weak_only": True}, "Unsupported detection settings"),
        ({"detection_covariance": np.eye(2)}, "Unsupported detection settings"),
    ],
)
def test_invalid_settings_rejected_in_preparation_and_edited_maps(settings, match):
    data, params = problem()
    params.attrs.update(settings)
    with pytest.raises(ValueError, match=match):
        spectroscopy.def_fit_params(data, params.attrs)
    with pytest.raises(ValueError, match=match):
        spectroscopy.fit_data_pseudo_inv(data, params)


@pytest.mark.parametrize(
    "backend", ["progress", "threading", "loky", "dask1", "dask2", "api"]
)
def test_identical_results_seeds_and_provenance_across_backends(backend, monkeypatch):
    data, params = problem(n_null=3)
    data = data.expand_dims(voxel=[10, 20])
    params = spectroscopy.def_fit_params(data, params.attrs)
    expected = spectroscopy.fit_data_pseudo_inv(data, params)
    assert expected.detection_stream.values.tolist() == [0, 1]
    assert expected.calibration_id.values[0] != expected.calibration_id.values[1]
    if backend.startswith("dask"):
        pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        executed = []

        def unexpected(*args, **kwargs):
            pytest.fail("Dask must not enter joblib")

        monkeypatch.setattr(
            spectroscopy, "_fit_data_pseudo_inv_with_joblib", unexpected
        )
        with Callback(pretask=lambda *args: executed.append(1)):
            result = spectroscopy.fit_data_pseudo_inv(
                data.chunk({"voxel": 1, "acquisition_idx": 8}), params
            )
            assert not executed
            result = result.compute(scheduler="threads", num_workers=int(backend[-1]))
        assert executed
    elif backend == "progress":
        result = spectroscopy.fit_data_pseudo_inv(data, params, show_tqdm=True)
    elif backend == "api":
        result = api.fit(data, "fid_pseudo_inv", params)
    else:
        result = spectroscopy.fit_data_pseudo_inv(
            data, params, n_jobs=2, batch_size=1, parallel_backend=backend
        )
    xr.testing.assert_identical(result, expected)


def test_zero_nonfinite_and_skipped_are_separate_from_nondetection():
    data, params = problem(fixed=True)
    data = data.expand_dims(voxel=["zero", "invalid", "skipped"]).copy(deep=True)
    data.loc[{"voxel": "zero"}] = 0
    data.loc[{"voxel": "invalid", "acquisition_idx": data.acquisition_idx[2]}] = np.nan
    params = spectroscopy.def_fit_params(data, params.attrs).copy(deep=True)
    params.skip_fitting.loc[{"voxel": "skipped", "metabolite": "Bicarbonate"}] = True
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert result.detection_status.values.tolist() == [
        "not_detected",
        "fit_failed",
        "not_tested",
    ]
    assert result.detection_p_value.sel(voxel="zero").item() == 1
    assert result.detection_n_null.values.tolist() == [19, 0, 0]
    assert not result.accepted.any()
    assert np.isnan(result.fit_amps_accepted).all()
    assert result.weak_candidate.sel(voxel="zero").all()


def test_low_budget_reports_resolution_and_never_stops_at_significance():
    data, params = problem(n_null=3)
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert result.detection_status.item() == "inconclusive"
    assert result.detection_reason.item() == "mc_resolution"
    assert result.detection_n_null.item() == 3
    assert result.detection_p_value.item() == 0.25


def test_recovery_keeps_bounds_freezes_masks_and_raw_negative_statistic(monkeypatch):
    data, params = problem(n_null=3)
    params = params.copy(deep=True)
    params.metabs_freq_ranges.values[0] = 0
    params.metabs_t2_ranges.values[1] = 0
    params = _update_frequency_bounds(params)
    actual = spectroscopy._fit_single_spectrum_numerical_outputs
    observed_m1 = []

    def numerical(signal, *maps, **options):
        result = actual(signal, *maps, **options)
        if np.array_equal(signal, data.values) and not maps[8][1]:
            observed_m1.append(tuple(np.array(m) for m in maps))
            if len(observed_m1) == 1:
                result = (*result[:8], 1000.0, 1000.0, *result[10:])
        return result

    monkeypatch.setattr(
        spectroscopy, "_fit_single_spectrum_numerical_outputs", numerical
    )
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert result.m1_recovery_attempted.item() and result.m1_recovery_used.item()
    assert result.test_statistic_initial.item() < 0
    assert result.test_statistic_raw.item() > 0
    assert len(observed_m1) == 2
    for before, after in zip(observed_m1[0][2:], observed_m1[1][2:], strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(observed_m1[1][0][0], result.m0_freqs.values[0])
    assert observed_m1[1][0][1] == params.initial_freqs_hz.values[1]
    assert result.fit_freqs.values[0] == params.initial_freqs_hz.values[0]
    assert result.fit_t2s.values[1] == params.initial_t2s_s.values[1]


@pytest.mark.parametrize("failure", ["observed_ordering", "null_failure"])
def test_unresolved_ordering_and_failed_nulls_cannot_produce_acceptance(
    failure, monkeypatch
):
    data, params = problem(n_null=3)
    actual = comparison.fit_model_pair

    def pair(signal, *args):
        result = actual(signal, *args)
        observed = np.array_equal(signal, data.values)
        if failure == "observed_ordering" and observed:
            m0, m1, initial, attempted, used, _ = result
            m1 = (*m1[:8], m0[8] + 1, m0[8] + 1, *m1[10:])
            return m0, m1, initial, True, used, "m1_worse_than_m0"
        if failure == "null_failure" and not observed:
            return (*result[:-1], "m1_failed")
        return result

    monkeypatch.setattr(comparison, "fit_model_pair", pair)
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert result.detection_status.item() == "inconclusive"
    assert np.isnan(result.detection_p_value.item())
    assert not result.accepted.any()
    if failure == "observed_ordering":
        assert result.test_statistic_raw.item() < 0
        assert result.detection_n_null.item() == 0
    else:
        assert (
            result.detection_n_null.item() == result.detection_n_failed_null.item() == 3
        )
        assert result.detection_n_valid_null.item() == 0


def test_provenance_matches_sources_and_changes_with_noise_seed_and_bounds():
    data, params = problem(n_null=1)
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    manifest = json.loads(result.attrs["detection_source_manifest"])
    path = Path(comparison.__file__).resolve()
    assert (
        manifest[path.relative_to(path.parents[4]).as_posix()]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    for key, value in (("detection_sigma", 0.04), ("detection_seed", 85412)):
        updated = params.copy(deep=True)
        updated.attrs[key] = value
        changed = spectroscopy.fit_data_pseudo_inv(data, updated)
        assert changed.calibration_id.item() != result.calibration_id.item()
    changed = params.copy(deep=True)
    changed.metabs_t2_ranges[:] = 0
    assert (
        spectroscopy.fit_data_pseudo_inv(data, changed).calibration_id.item()
        != result.calibration_id.item()
    )


def test_weak_flags_do_not_select_tests_or_change_amplitudes():
    data, params = problem()
    low_threshold = spectroscopy.fit_data_pseudo_inv(data, params)
    params.attrs["detection_weak_snr"] = 1e9
    high_threshold = spectroscopy.fit_data_pseudo_inv(data, params)
    assert high_threshold.weak_candidate.all()
    assert not low_threshold.weak_candidate.all()
    for name in (
        "fit_amps",
        "fit_freqs",
        "fit_t2s",
        "detection_n_null",
        "detection_p_value",
        "calibration_id",
    ):
        xr.testing.assert_identical(high_threshold[name], low_threshold[name])


def test_five_peak_bicarbonate_null_and_manual_exclusion_smoke():
    time = np.arange(64) * 0.0004
    frequencies = np.array([-70, 857, 566, 340, -830], dtype=float)
    basis = np.exp(2j * np.pi * time[:, None] * frequencies - time[:, None] / 0.025)
    rng = np.random.default_rng(291005)  # independent null-validation input
    fid = basis @ np.array([1 + 0.7j, 0.8 - 0.2j, 0, 0.05 + 0.01j, 0])
    fid += 0.02 * (rng.standard_normal(64) + 1j * rng.standard_normal(64))
    data = xr.DataArray(fid, dims="acquisition_idx", coords={"acquisition_idx": time})
    params = spectroscopy.def_fit_params(
        data,
        {
            "metabs": [
                "Pyruvate",
                "Lactate",
                "PyruvateHydrate",
                "Alanine",
                "Bicarbonate",
            ],
            "metabs_freqs_hz": frequencies,
            "skip_fitting": [False, False, True, False, False],
            "metabs_freq_ranges": 20,
            "metabs_t2_ranges": 0.02,
            "npoints": 3,
            "rep_fitting": 2,
            "detection_target": "Bicarbonate",
            "detection_sigma": 0.02,
            "detection_seed": 863217,
            "detection_null_samples": 3,
        },
    )
    before = params.copy(deep=True)
    result = spectroscopy.fit_data_pseudo_inv(data, params, n_jobs=2)
    assert result.fit_amps.values[2] == result.m0_amps.values[2] == 0
    assert result.m0_amps.values[4] == 0
    assert result.fit_amps.values[4] != 0  # raw noise peak remains complex and free
    assert np.any(np.abs(result.fit_amps.values.imag) > 0.1)
    assert result.m0_skipped.values.tolist() == [False, False, True, False, True]
    assert result.detection_n_null.item() == 3
    xr.testing.assert_identical(params, before)


def test_ill_conditioned_basis_yields_diagnostic_without_constraints():
    data, params = problem(fixed=True, n_null=3)
    params = params.copy(deep=True)
    params.attrs["min_peak_separation_hz"] = 0
    params.initial_freqs_hz.values[:] = -31
    params.freq_range_lower_bound_hz.values[:] = -31
    params.freq_range_upper_bound_hz.values[:] = -31
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert (
        result.weak_reason.values & comparison.WEAK_REASONS["ill_conditioned"]
    ).all()
    assert np.isnan(result.amplitude_std_quadrature_conditional).all()
    assert result.detection_n_null.item() == 3
    np.testing.assert_array_equal(result.fit_freqs, [-31, -31])


def test_nonfinite_normalized_statistic_cannot_be_accepted():
    data, params = problem(fixed=True, n_null=3)
    params.attrs["detection_sigma"] = 1e-160
    result = spectroscopy.fit_data_pseudo_inv(data, params)
    assert result.detection_status.item() == "fit_failed"
    assert result.detection_reason.item() == "nonfinite_statistic"
    assert result.detection_n_null.item() == 0
    assert not result.accepted.any()


@pytest.mark.parametrize("backend", ["serial", "joblib", "dask"])
@pytest.mark.parametrize("store_null", [False, True])
def test_optional_null_draws_preserve_decisions_and_counting(backend, store_null):
    data, params = problem(fixed=True)
    expected = spectroscopy.fit_data_pseudo_inv(data, params)
    params.attrs["detection_store_null"] = store_null
    kwargs = {}
    if backend == "joblib":
        kwargs["n_jobs"] = 2
    elif backend == "dask":
        pytest.importorskip("dask.array")
        data = data.chunk({"acquisition_idx": -1})
    result = spectroscopy.fit_data_pseudo_inv(data, params, **kwargs).compute()
    for name in [
        "fit_amps",
        "test_statistic_raw",
        "detection_p_value",
        "calibration_id",
    ]:
        xr.testing.assert_equal(result[name], expected[name])
    xr.testing.assert_equal(result.accepted, expected.accepted)
    if not store_null:
        assert "detection_null_statistic" not in result
        return
    draws = result.detection_null_statistic.values
    assert draws.shape == (params.attrs["detection_null_samples"],)
    assert np.all(result.detection_null_reason.values == 0)
    extreme = np.count_nonzero(draws >= result.test_statistic_raw.item())
    assert extreme == result.detection_n_extreme.item()
    assert result.detection_p_value.item() == (extreme + 1) / (len(draws) + 1)


def test_optional_null_draws_mark_unattempted_and_failed(monkeypatch):
    data, params = problem(n_null=3)
    params.attrs["detection_store_null"] = True
    params.skip_fitting.loc[{"metabolite": "Bicarbonate"}] = True
    skipped = spectroscopy.fit_data_pseudo_inv(data, params)
    assert np.all(skipped.detection_null_reason.values == -1)
    assert np.all(np.isnan(skipped.detection_null_statistic.values))
    params.skip_fitting.loc[{"metabolite": "Bicarbonate"}] = False
    original = comparison.fit_model_pair

    def failing_null(signal, *args):
        pair = original(signal, *args)
        if not np.array_equal(signal, data.values):
            return (*pair[:-1], "m1_failed")
        return pair

    monkeypatch.setattr(comparison, "fit_model_pair", failing_null)
    failed = spectroscopy.fit_data_pseudo_inv(data, params)
    assert np.all(
        failed.detection_null_reason.values
        == comparison.DETECTION_REASONS.index("m1_failed")
    )
    assert failed.detection_status.item() == "inconclusive"
    assert np.isnan(failed.detection_p_value.item())


def test_null_storage_option_requires_boolean():
    data, params = problem()
    params.attrs["detection_store_null"] = "yes"
    with pytest.raises(ValueError, match="detection_store_null must be boolean"):
        spectroscopy.fit_data_pseudo_inv(data, params)


@pytest.mark.parametrize("backend", ["serial", "joblib", "dask"])
def test_labelled_sigma_and_stream_maps_match_independent_spectra(backend):
    data, params = problem(fixed=True, n_null=3)
    data = xr.concat([data, data * 1.4], dim=xr.IndexVariable("sample_id", ["a", "b"]))
    params["detection_stream_id"] = xr.DataArray(
        [71, 93], dims="sample_id", coords={"sample_id": ["a", "b"]}
    )
    params["detection_sigma_map"] = xr.DataArray(
        [0.03, 0.05], dims="sample_id", coords={"sample_id": ["a", "b"]}
    )
    params.attrs["detection_store_null"] = True
    kwargs = {"n_jobs": 2} if backend == "joblib" else {}
    if backend == "dask":
        pytest.importorskip("dask.array")
        data = data.chunk({"sample_id": 1, "acquisition_idx": -1})
    batched = spectroscopy.fit_data_pseudo_inv(data, params, **kwargs).compute()
    for label in ["a", "b"]:
        single = spectroscopy.fit_data_pseudo_inv(
            data.sel(sample_id=label).compute(), params.sel(sample_id=label)
        ).compute()
        for name in [
            "fit_amps",
            "detection_null_statistic",
            "detection_p_value",
            "calibration_id",
            "noise_sigma_quadrature",
        ]:
            xr.testing.assert_equal(batched[name].sel(sample_id=label), single[name])
    np.testing.assert_array_equal(batched.detection_stream.values, [71, 93])
    np.testing.assert_allclose(batched.noise_sigma_quadrature.values, [0.03, 0.05])
    np.testing.assert_allclose(
        batched.test_statistic_raw,
        (batched.rss_m0 - batched.rss_m1) / batched.noise_sigma_quadrature**2,
    )


@pytest.mark.parametrize(
    "name,values",
    [
        ("detection_stream_id", [1, 1]),
        ("detection_stream_id", [-1, 2]),
        ("detection_stream_id", [1.0, 2.0]),
        ("detection_stream_id", [True, False]),
        ("detection_sigma_map", [0.0, 0.03]),
        ("detection_sigma_map", [np.nan, 0.03]),
        ("detection_sigma_map", [1e300, 0.03]),
        ("detection_sigma_map", [1e-300, 0.03]),
    ],
)
def test_invalid_detection_maps_fail_before_fitting(name, values):
    data, params = problem(fixed=True, n_null=3)
    data = xr.concat([data, data], dim="sample_id")
    params[name] = xr.DataArray(values, dims="sample_id")
    with pytest.raises(ValueError, match=name):
        spectroscopy.fit_data_pseudo_inv(data, params)
