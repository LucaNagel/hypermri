"""Deterministic scientific regression fixture.

Five-component complex FIDs with null/signal conditions and fixed
simulation/fit seeds; retained M0/M1 detection and labelled reconstruction
assertions. Calls shipped domain APIs; no acquired data, plotting, downloads or
file writes.
"""

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
    reconstruct_fitted_signal,
)

component_names = ["Pyruvate", "Lactate", "PyruvateHydrate", "Alanine", "Bicarbonate"]
true_frequencies_hz = np.array([-80.0, -15.0, 50.0, 105.0, 175.0])
true_t2s_s = np.array([0.026, 0.025, 0.02, 0.027, 0.025])
time_s = np.arange(256) * 0.001
sigma_per_quadrature = 0.03
true_amplitudes = np.array(
    [
        [1.0 + 0.2j, 0.42 - 0.15j, 0.18 + 0.08j, 0.25 - 0.1j, 0.0 + 0j],
        [1.0 + 0.2j, 0.42 - 0.15j, 0.18 + 0.08j, 0.25 - 0.1j, 0.2 + 0.06j],
    ]
)
basis = np.exp(
    2j * np.pi * time_s[:, None] * true_frequencies_hz - time_s[:, None] / true_t2s_s
)
data_rng = np.random.default_rng(20260905)
noise = sigma_per_quadrature * (
    data_rng.standard_normal((2, time_s.size))
    + 1j * data_rng.standard_normal((2, time_s.size))
)
complex_fids = true_amplitudes @ basis.T + noise
data = xr.DataArray(
    complex_fids,
    dims=("condition", "sample"),
    coords={"condition": ["null", "with_bicarbonate"], "sample": time_s},
    attrs={
        "signal_domain": "temporal",
        "description": "Synthetic non-CSI FIDs for the M0/M1 tutorial",
    },
    name="complex_fid",
)
data.coords["sample"].attrs["unit"] = "s"
initial_frequencies_hz = true_frequencies_hz + np.array([2.0, -2.0, 2.0, -2.0, 2.0])
fit_params = def_fit_params(
    data,
    {
        "spectral_dim": "sample",
        "metabs": component_names,
        "metabs_freqs_hz": initial_frequencies_hz,
        "initial_t2s_s": true_t2s_s,
        "metabs_freq_ranges": 12.0,
        "metabs_t2_ranges": 0.01,
        "min_peak_separation_hz": 5.0,
        "npoints": 5,
        "niter": 1,
        "rep_fitting": 1,
        "amp_solver": "pinv",
        "snr_threshold": None,
        "detection_target": "Bicarbonate",
        "detection_sigma": sigma_per_quadrature,
        "detection_alpha": 0.05,
        "detection_null_samples": 39,
        "detection_seed": 550921,
        "detection_store_null": True,
    },
)
result = fit_data_pseudo_inv(data, fit_params, n_jobs=1)
target = "Bicarbonate"
summary = xr.Dataset(
    {
        "raw_target_amplitude": result.fit_amps.sel(metabolite=target),
        "accepted_target_amplitude": result.fit_amps_accepted.sel(metabolite=target),
        "target_accepted": result.accepted.sel(metabolite=target),
        "p_value": result.detection_p_value,
        "mc_tail_low": result.detection_mc_tail_low,
        "mc_tail_high": result.detection_mc_tail_high,
        "test_statistic_raw": result.test_statistic_raw,
        "rss_m0": result.rss_m0,
        "rss_m1": result.rss_m1,
        "fit_status": result.fit_status,
        "detection_status": result.detection_status,
        "detection_reason": result.detection_reason,
    }
)
assert result.detection_status.sel(condition="null").item() == "not_detected"
assert result.detection_status.sel(condition="with_bicarbonate").item() == "detected"
assert not result.accepted.sel(condition="null", metabolite=target).item()
assert result.accepted.sel(condition="with_bicarbonate", metabolite=target).item()
assert np.isnan(
    result.fit_amps_accepted.sel(condition="null", metabolite=target).item()
)
assert result.detection_p_value.sel(condition="with_bicarbonate").item() == 0.025
null_diagnostics = result[
    [
        "detection_null_statistic",
        "detection_null_reason",
        "detection_null_recovery_used",
    ]
]
assert result.sizes["null_replicate"] == 39
fitted = reconstruct_fitted_signal(result, data)
residual = data - fitted
rss_from_residual = (np.abs(residual) ** 2).sum("sample")
np.testing.assert_allclose(rss_from_residual, result.fit_rss, rtol=1e-12, atol=1e-14)
xr.Dataset(
    {
        "fit_rss": result.fit_rss,
        "rss_recomputed": rss_from_residual,
        "max_abs_residual": np.abs(residual).max("sample"),
    }
)
