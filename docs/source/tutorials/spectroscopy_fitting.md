---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Spectroscopy fitting

{func}`~hypermri.analysis.spectroscopy.fit_spectrum` fits a sum of damped
complex exponentials (Lorentzian lines) to free induction decays (FIDs). This
tutorial fits a synthetic signal whose true parameters are known, so every
step can be checked. It needs no downloaded data.

## A synthetic FID

Two metabolites, called pyruvate and lactate here, with known amplitude,
frequency and T2*, sampled 256 times at a dwell time of 0.4 ms, plus complex
Gaussian noise:

```{code-cell} ipython3
import numpy as np
import xarray as xr

dwell_time_s = 0.0004
time_s = np.arange(256) * dwell_time_s
true = {
    "amplitude": np.array([1.0, 0.3]),
    "frequency_hz": np.array([0.0, 920.0]),
    "t2_s": np.array([0.025, 0.020]),
}
noise_sigma = 0.05
rng = np.random.default_rng(20261002)

clean = sum(
    a * np.exp(2j * np.pi * f * time_s - time_s / t2)
    for a, f, t2 in zip(true["amplitude"], true["frequency_hz"], true["t2_s"])
)
noise = noise_sigma * (rng.standard_normal(256) + 1j * rng.standard_normal(256))

fid = xr.DataArray(
    clean + noise,
    dims="sample",
    coords={"sample": time_s},
    attrs={"signal_domain": "temporal"},
)
```

The attribute `signal_domain="temporal"` tells the fit that the samples are
an FID, not a spectrum. Data from a scan carry it already.

## Fit

The fit options name the dimension that holds the FID, the metabolites, their
starting frequencies and T2* values, and how far the fit may move from them:

```{code-cell} ipython3
from hypermri.analysis.spectroscopy import fit_spectrum

fit_params = {
    "spectral_dim": "sample",
    "metabs": ["pyruvate", "lactate"],
    "metabs_freqs_hz": [3.0, 915.0],
    "initial_t2s_s": [0.02, 0.02],
    "metabs_freq_ranges": 20.0,
    "metabs_t2_ranges": 0.02,
}
fit = fit_spectrum(fid, fit_params)
```

The result is an {class}`xarray.Dataset` with one entry per metabolite. Every
other dimension of the input (voxels, repetitions) would be kept, so the same
call fits a whole CSI data set.

```{code-cell} ipython3
import pandas as pd

pd.DataFrame(
    {
        "true |amplitude|": true["amplitude"],
        "fit |amplitude|": abs(fit.fit_amps.values),
        "true frequency (Hz)": true["frequency_hz"],
        "fit frequency (Hz)": fit.fit_freqs.values,
        "true T2* (ms)": 1e3 * true["t2_s"],
        "fit T2* (ms)": 1e3 * fit.fit_t2s.values,
    },
    index=fit.metabolite.values,
).round(3)
```

Each value comes with a status, and the options actually used are stored as
attributes, so a saved result documents how it was made:

```{code-cell} ipython3
fit.fit_freq_status.values, fit.attrs["fit_type"], fit.attrs["lineshape"]
```

## Check the fit

{func}`~hypermri.analysis.fitting.spectroscopy.reconstruct_fitted_signal`
rebuilds the fitted FID. Its spectrum should match the data, and the residual
should look like noise:

```{code-cell} ipython3
import matplotlib.pyplot as plt

from hypermri.analysis.fitting.spectroscopy import reconstruct_fitted_signal

model = reconstruct_fitted_signal(fit, fid)


def spectrum(signal):
    return np.fft.fftshift(np.fft.fft(np.asarray(signal)))


frequency_hz = np.fft.fftshift(np.fft.fftfreq(fid.size, d=dwell_time_s))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(frequency_hz, spectrum(fid).real, color="0.6", label="data")
ax.plot(frequency_hz, spectrum(model).real, color="C0", label="fit")
ax.plot(frequency_hz, spectrum(fid - model).real - 15, color="C3", label="residual (shifted)")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("real part (a.u.)")
ax.legend(frameon=False)
fig.tight_layout()
```

## Noise

The noise level is needed for uncertainties. With a noise-only measurement
(an acquisition without excitation), {func}`~hypermri.analysis.spectroscopy.estimate_noise`
estimates it; passing it to the fit attaches it to the result:

```{code-cell} ipython3
from hypermri.analysis.spectroscopy import estimate_noise

noise_only = xr.DataArray(
    noise_sigma * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)),
    dims="sample",
    coords={"sample": np.arange(2048) * dwell_time_s},
)
noise_estimate = estimate_noise(noise_only, method="rf_off")
fit = fit_spectrum(fid, fit_params, noise=noise_estimate)
noise_estimate
```

## Uncertainties

{func}`~hypermri.analysis.spectroscopy.estimate_fit_local_uncertainty` turns
the noise level and the local curvature of the fit into confidence intervals.
It is fast and suited to well-separated peaks:

```{code-cell} ipython3
from hypermri.analysis.fitting.spectroscopy import def_fit_params
from hypermri.analysis.spectroscopy import estimate_fit_local_uncertainty

intervals = estimate_fit_local_uncertainty(
    fit,
    def_fit_params(fid, fit_params),
    confidence_level=0.95,
    noise_degrees_of_freedom=2 * noise_only.size,
)
intervals[
    ["frequency_interval_low", "frequency_interval_high", "t2_interval_low", "t2_interval_high"]
].to_dataframe().round(4)
```

Both true frequencies (0 and 920 Hz) and T2* values (25 and 20 ms) lie inside
their intervals. For overlapping peaks or low SNR,
{func}`~hypermri.analysis.spectroscopy.estimate_fit_uncertainty` repeats the
fit on resampled noise (a bootstrap) and needs no local approximation; it
takes longer.

## Next

- {doc}`viewer`: fit spectra interactively with the fit tool.
- {doc}`../concepts/data_model`: names and dimensions of fit results.
