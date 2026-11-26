"""Deterministic scientific regression fixture.

RF-off: 64x256 independent proper complex Gaussian samples, sigma 0.03 a.u.,
seed 2026091101. Signal and bootstrap seeds 2026091103/2026091105; 99 bootstrap
draws. Calls shipped domain APIs; no acquired data, plotting, downloads or file
writes.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chi2

from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
    reconstruct_fitted_signal,
)
from hypermri.analysis.spectroscopy import (
    evaluate_spectroscopy_fit_quality,
    make_rf_off_noise_reference,
)

SIGMA_TRUE = 0.03
time_s = np.arange(256) * 0.0004
reference_rng = np.random.default_rng(2026091101)
z = SIGMA_TRUE * (
    reference_rng.normal(size=(64, time_s.size))
    + 1j * reference_rng.normal(size=(64, time_s.size))
)
rf_off = xr.DataArray(
    z,
    dims=("reference", "sample"),
    coords={"reference": np.arange(64), "sample": time_s},
    attrs={"signal_domain": "temporal", "unit": "a.u."},
)
metadata = {
    "sampling_time_axis_s": time_s.tolist(),
    "dwell_time_s": float(time_s[1] - time_s[0]),
    "channel_labels": ["synthetic_single_channel"],
    "n_receivers": 1,
    "reconstruction_model": "identity_no_filter_no_averaging",
    "acquisition_model": "independent_proper_complex_white_gaussian",
}
noise_reference = make_rf_off_noise_reference(
    rf_off,
    signal_fingerprint="synthetic-signal-seed-2026091103",
    reference_fingerprint="synthetic-rf-off-seed-2026091101",
    signal_acquisition=metadata,
    reference_acquisition=metadata,
)
sigma_hat = float(noise_reference.noise_sigma_quadrature)
sample_count = z.size
energy = np.sum(np.abs(z) ** 2)
sigma_interval = np.sqrt(energy / chi2.ppf([0.975, 0.025], df=2 * sample_count))
pd.Series(
    {
        "Sigma wahr": SIGMA_TRUE,
        "Sigma geschätzt": sigma_hat,
        "Referenz-Sigma 95 % unten": sigma_interval[0],
        "Referenz-Sigma 95 % oben": sigma_interval[1],
        "unabhängige komplexe Samples": sample_count,
    }
)
tail_rng = np.random.default_rng(2026091102)
long_signal = 0.3 * np.exp(2j * np.pi * 90 * time_s - time_s / 0.08)
tail_noise = SIGMA_TRUE * (
    tail_rng.normal(size=(64, time_s.size))
    + 1j * tail_rng.normal(size=(64, time_s.size))
)
contaminated = long_signal[None, :] + tail_noise
block_start = np.arange(0, time_s.size, 32)
block_time_ms = np.array([time_s[k : k + 32].mean() * 1000 for k in block_start])
block_sigma = np.array(
    [
        np.sqrt(np.mean(np.abs(contaminated[:, k : k + 32]) ** 2) / 2)
        for k in block_start
    ]
)
tail_sigma = np.sqrt(np.mean(np.abs(contaminated[:, -64:]) ** 2) / 2)
labels = ["ohne_peak", "schwach", "stark"]
true_amplitudes = SIGMA_TRUE * np.array([0.0, 2.0, 6.0]) * np.exp(0.4j)
true_frequency_hz = 80.0
true_t2_s = 0.025
clean = true_amplitudes[:, None] * np.exp(
    2j * np.pi * true_frequency_hz * time_s - time_s / true_t2_s
)
signal_rng = np.random.default_rng(2026091103)
y = clean + SIGMA_TRUE * (
    signal_rng.normal(size=clean.shape) + 1j * signal_rng.normal(size=clean.shape)
)
data = xr.DataArray(
    y,
    dims=("condition", "sample"),
    coords={"condition": labels, "sample": time_s},
    attrs={"signal_domain": "temporal"},
)
settings = {
    "spectral_dim": "sample",
    "metabs": ["Peak"],
    "metabs_freqs_hz": [82.0],
    "initial_t2s_s": [0.03],
    "metabs_freq_ranges": 30.0,
    "metabs_t2_ranges": 0.03,
    "npoints": 9,
    "niter": 1,
    "rep_fitting": 3,
    "amp_solver": "pinv",
    "snr_threshold": None,
}
fit_params = def_fit_params(data, settings)
fit = fit_data_pseudo_inv(data, fit_params, n_jobs=1)
fitted = reconstruct_fitted_signal(fit, data)
residual = data - fitted
parameter_table = pd.DataFrame(
    {
        "Fall": labels,
        "Amplitude wahr": np.abs(true_amplitudes),
        "Amplitude Fit": np.abs(fit.fit_amps.sel(metabolite="Peak").values),
        "Frequenz wahr / Hz": [np.nan, true_frequency_hz, true_frequency_hz],
        "Frequenz Fit / Hz": fit.fit_freqs.sel(metabolite="Peak").values,
        "T2* wahr / ms": [np.nan, true_t2_s * 1000, true_t2_s * 1000],
        "T2* Fit / ms": fit.fit_t2s.sel(metabolite="Peak").values * 1000,
        "Fitstatus": fit.fit_status.values,
    }
)
quality = evaluate_spectroscopy_fit_quality(
    data,
    fit_params,
    noise_reference,
    fit_results=fit,
    config={"detection_null_samples": 39, "detection_seed": 2026091104},
    run_detection=False,
)
quality[
    [
        "amplitude_snr_conditional",
        "peak_quality_class",
        "peak_quality_status",
        "peak_detection_basis",
        "full_detection_test_required",
    ]
].to_dataframe()
factors = np.array([0.8, 1.0, 1.2])
strong_fit_snr = float(
    quality.amplitude_snr_conditional.sel(condition="stark", metabolite="Peak")
)
sensitivity = pd.DataFrame(
    {
        "verwendetes Sigma / Referenz-Sigma": factors,
        "Fit-SNR starker Peak": strong_fit_snr / factors,
        "Faktor der Teststatistik bei gleicher RSS-Differenz": 1 / factors**2,
    }
)
BOOTSTRAP_SAMPLES = 99
bootstrap_rng = np.random.default_rng(2026091105)
strong_model = fitted.sel(condition="stark").values
bootstrap_values = strong_model[None, :] + sigma_hat * (
    bootstrap_rng.normal(size=(BOOTSTRAP_SAMPLES, time_s.size))
    + 1j * bootstrap_rng.normal(size=(BOOTSTRAP_SAMPLES, time_s.size))
)
bootstrap_data = xr.DataArray(
    bootstrap_values,
    dims=("draw", "sample"),
    coords={"draw": np.arange(BOOTSTRAP_SAMPLES), "sample": time_s},
    attrs={"signal_domain": "temporal"},
)
bootstrap_params = def_fit_params(bootstrap_data, settings)
bootstrap_fit = fit_data_pseudo_inv(bootstrap_data, bootstrap_params, n_jobs=1)
draws = pd.DataFrame(
    {
        "Amplitude / a.u.": np.abs(
            bootstrap_fit.fit_amps.sel(metabolite="Peak").values
        ),
        "Frequenz / Hz": bootstrap_fit.fit_freqs.sel(metabolite="Peak").values,
        "T2* / ms": bootstrap_fit.fit_t2s.sel(metabolite="Peak").values * 1000,
    }
)
if not bool((bootstrap_fit.fit_status == "success").all()):
    raise RuntimeError(
        "Mindestens ein Bootstrap-Fit fehlgeschlagen; keine Intervallausgabe."
    )
intervals = draws.quantile([0.025, 0.5, 0.975]).T
intervals.columns = ["2,5-%-Quantil", "Median", "97,5-%-Quantil"]
