"""Deterministic scientific regression fixture.

Noiseless CSI: 128 samples at 1 ms, 3x2x1 voxels, -40/60 Hz, T2 25/35 ms.
Recovery limits: 0.5 Hz, 1 ms, 0.01 a.u.; complex RMSE < 0.002 a.u. Calls
shipped domain APIs; no acquired data, plotting, downloads or file writes.
"""

import numpy as np
import xarray as xr

from hypermri.analysis.fitting import fit, prepare_fit_params
from hypermri.analysis.fitting.spectroscopy import reconstruct_fitted_signal
from hypermri.analysis.spectroscopy import generate_fid

time_s = np.arange(128) * 0.001
true_frequency_hz = np.array([-40.0, 60.0])
true_t2_s = np.array([0.025, 0.035])
true_amplitude = np.array(
    [[[1.0, 0.3], [1.2, 0.4]], [[1.4, 0.5], [1.6, 0.6]], [[1.8, 0.7], [2.0, 0.8]]]
)
basis = generate_fid(
    time_axis_s=time_s,
    amplitude=1.0,
    t2_s=true_t2_s,
    freq_hz=true_frequency_hz,
    phase_rad=0.0,
    noise_std=0.0,
    sum_fids=False,
)
values = np.einsum("tm,rpm->trp", basis, true_amplitude)[..., None]
data = xr.DataArray(
    values,
    dims=("acquisition_idx", "read_idx", "phase_idx", "slice_idx"),
    coords={
        "acquisition_idx": time_s,
        "read_idx": [0, 1, 2],
        "phase_idx": [0, 1],
        "slice_idx": [0],
    },
    attrs={
        "signal_domain": "temporal",
        "unit": "a.u.",
        "sampling_time_axis_s": time_s,
        "dwell_time_s": 0.001,
        "provenance": "invented noiseless two-component CSI volume",
    },
)
data.coords["acquisition_idx"].attrs["units"] = "s"
original = data.copy(deep=True)
params = prepare_fit_params(
    data,
    model="fid_pseudo_inv",
    fit_params={
        "metabs": ["component_a", "component_b"],
        "metabs_freqs_hz": true_frequency_hz,
        "initial_t2s_s": true_t2_s,
        "metabs_freq_ranges": [4.0, 4.0],
        "metabs_t2_ranges": [0.004, 0.004],
        "min_t2_s": [0.02, 0.03],
        "max_t2_s": [0.03, 0.04],
        "npoints": 5,
        "niter": 1,
        "rep_fitting": 1,
        "amp_solver": "pinv",
        "snr_threshold": None,
    },
)
result = fit(data, model="fid_pseudo_inv", fit_params=params)
fitted = reconstruct_fitted_signal(result, data)
residual = data - fitted
signal_rmse = float(np.sqrt(np.mean(np.abs(residual.values) ** 2)))
assert bool((result.fit_status == "success").all())
xr.testing.assert_identical(data, original)
assert fitted.dims == data.dims
for dim in data.dims:
    xr.testing.assert_identical(fitted.coords[dim], data.coords[dim])
assert signal_rmse < 0.002
for index, metabolite in enumerate(["component_a", "component_b"]):
    np.testing.assert_allclose(
        result.fit_freqs.sel(metabolite=metabolite),
        true_frequency_hz[index],
        atol=0.5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.fit_t2s.sel(metabolite=metabolite),
        true_t2_s[index],
        atol=0.001,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.abs(result.fit_amps.sel(metabolite=metabolite).squeeze("slice_idx")),
        true_amplitude[..., index],
        atol=0.01,
        rtol=0.0,
    )
