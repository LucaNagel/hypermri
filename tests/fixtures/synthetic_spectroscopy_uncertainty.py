"""Two-peak scientific uncertainty fixture.

256 complex samples at 0.4 ms: amplitudes 1.0/0.3, 0/920 Hz, T2 25/20 ms,
quadrature sigma 0.05 a.u.; 2048-sample RF-off reference. Seed 20261002; 199
bootstrap draws. Numerical assertions live in the associated unit tests. Calls
shipped domain APIs; no acquired data, plotting, downloads or file writes.
"""

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import reconstruct_fitted_signal
from hypermri.analysis.spectroscopy import fit_spectrum

dwell_time_s = 0.0004
time_s = np.arange(256) * dwell_time_s
true_amplitudes = np.array([1.0, 0.3])
true_frequencies_hz = np.array([0.0, 920.0])
true_t2s_s = np.array([0.025, 0.02])
true_noise_sigma = 0.05
rng = np.random.default_rng(20261002)
noise_free = sum(
    (
        amplitude * np.exp(2j * np.pi * frequency * time_s) * np.exp(-time_s / t2)
        for amplitude, frequency, t2 in zip(
            true_amplitudes, true_frequencies_hz, true_t2s_s, strict=False
        )
    )
)
noisy = noise_free + true_noise_sigma * (
    rng.standard_normal(time_s.size) + 1j * rng.standard_normal(time_s.size)
)
data = xr.DataArray(
    noisy, dims="sample", coords={"sample": time_s}, attrs={"signal_domain": "temporal"}
)
fit_params = {
    "spectral_dim": "sample",
    "metabs": ["pyruvate", "lactate"],
    "metabs_freqs_hz": [3.0, 915.0],
    "initial_t2s_s": [0.02, 0.02],
    "metabs_freq_ranges": 20.0,
    "metabs_t2_ranges": 0.02,
}
fit = fit_spectrum(data, fit_params)
residual = data - reconstruct_fitted_signal(fit, data)
from hypermri.analysis.spectroscopy import estimate_noise

rf_off_fids = xr.DataArray(
    true_noise_sigma * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)),
    dims="sample",
    coords={"sample": np.arange(2048) * dwell_time_s},
)
noise = estimate_noise(rf_off_fids, method="rf_off")
fit_with_noise = fit_spectrum(data, fit_params, noise=noise)
noise_pipeline = estimate_noise(rf_off_fids, method="rf_off", return_pipeline=True)
noise_from_pipeline = noise_pipeline["noise_sigma_quadrature"]
distribution_fits = noise_pipeline[
    [
        "fit_location",
        "fit_scale",
        "fit_distribution_mean",
        "fit_distribution_std",
        "fit_noise_sigma",
        "fit_sample_count",
        "fit_formula",
        "fit_is_primary",
    ]
]
steps = noise_pipeline["processing_description"]
from hypermri.analysis.spectroscopy import compare_metabolite_relevance

relevance = compare_metabolite_relevance(
    data, fit_params, fit_results=fit, targets="lactate"
)
from hypermri.analysis.spectroscopy import estimate_fit_uncertainty

uncertainty_params = {
    **fit_params,
    "uncertainty_bootstrap_samples": 199,
    "uncertainty_confidence_level": 0.95,
    "uncertainty_seed": 20261002,
}
bootstrap = estimate_fit_uncertainty(data, uncertainty_params, fit_with_noise)
from hypermri.analysis.fitting.spectroscopy import def_fit_params
from hypermri.analysis.spectroscopy import estimate_fit_covariance

prepared_params = def_fit_params(data, fit_params)
local_errors = estimate_fit_covariance(fit_with_noise, prepared_params)
frequency_t2_covariance = local_errors.parameter_covariance.sel(
    fit_parameter="frequency:pyruvate", fit_parameter_other="t2:pyruvate"
)
from hypermri.analysis.spectroscopy import estimate_fit_local_uncertainty

local_intervals = estimate_fit_local_uncertainty(
    fit_with_noise,
    prepared_params,
    confidence_level=0.95,
    noise_degrees_of_freedom=2 * rf_off_fids.size,
)
