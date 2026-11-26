"""Small numerical baseline for simplifying spectroscopy fitting.

The acquisition and signal parameters follow cells 5 and 7 of
``08_si_fit_quality_estimation_model_comparison.ipynb``: 256 samples at
0.4 ms, five metabolites, T2* = 25 ms, and independent Gaussian noise in
each complex quadrature. Frequencies are rounded to the six decimal places
printed in the notebook. No measured arrays or scan files are needed.

Only two bicarbonate amplitudes (0 and 0.05) and one paired noise realization
at SNR = 50 are used. This is a regression baseline, not a detection-power
study. Both the canonical ``metabs_freqs_hz`` key and the notebook spelling
``metabs_freqs_Hz`` are checked, including input without frequency coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _update_frequency_bounds,
)
from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
    reconstruct_fitted_signal,
)
from hypermri.analysis.spectroscopy.signals import generate_fid

METABOLITES = ["Pyruvate", "Lactate", "PyruvateHydrate", "Alanine", "Bicarbonate"]
FREQUENCIES_HZ = np.array([-69.754216, 856.829390, 566.186822, 339.712093, -830.407339])
T2_S = 0.025
NOISE_STD = 1.0 / 50.0


@pytest.fixture(scope="module")
def notebook_fids() -> xr.Dataset:
    """Return clean/noisy FIDs and true amplitudes for absent/present bicarbonate.

    Noise is drawn once after summing all metabolite FIDs, with sigma = 0.02
    per quadrature and seed 20260912. Reusing it for both bicarbonate amplitudes
    follows the notebook's paired simulation. The ppm coordinate is derived
    from its pyruvate (171.076 ppm) and bicarbonate (161 ppm) frequency pairs;
    this allows comparison of calibrated and Hz-only parameter preparation.
    """
    time_s = np.arange(256) * 0.0004
    amplitudes = np.array([[1.0, 0.8, 0.1, 0.05, bic] for bic in (0.0, 0.05)])
    clean = np.stack(
        [
            generate_fid(
                time_axis_s=time_s,
                amplitude=amps,
                freq_hz=FREQUENCIES_HZ,
                t2_s=T2_S,
                noise_std=0.0,
                sum_fids=True,
            )
            for amps in amplitudes
        ]
    )
    rng = np.random.default_rng(20260912)
    noise = NOISE_STD * (rng.normal(size=256) + 1j * rng.normal(size=256))
    frequency_axis_hz = np.fft.fftshift(np.fft.fftfreq(256, d=0.0004))
    hz_per_ppm = (FREQUENCIES_HZ[0] - FREQUENCIES_HZ[-1]) / (171.076 - 161.0)
    frequency_axis_ppm = 171.076 + (frequency_axis_hz - FREQUENCIES_HZ[0]) / hz_per_ppm
    data = xr.Dataset(
        {
            "clean": (("bicarbonate_amplitude", "acquisition_idx"), clean),
            "noisy": (("bicarbonate_amplitude", "acquisition_idx"), clean + noise),
            "true_amplitudes": (("bicarbonate_amplitude", "metabolite"), amplitudes),
        },
        coords={
            "bicarbonate_amplitude": [0.0, 0.05],
            "metabolite": METABOLITES,
            "acquisition_idx": np.arange(256),
            "acquisition_time_s": ("acquisition_idx", time_s),
            "sampling_time_axis_s": ("acquisition_idx", time_s),
            "freq_axis_hz": ("acquisition_idx", frequency_axis_hz),
            "freq_axis_ppm": ("acquisition_idx", frequency_axis_ppm),
        },
    )
    for name in ("clean", "noisy"):
        data[name].attrs.update(
            signal_domain="temporal", simulation_source="notebook_five_metabolites"
        )
    data["noisy"].attrs.update(random_seed=20260912, noise_std=NOISE_STD)
    return data


def _prepare_notebook_fit_params(
    data: xr.DataArray, frequency_key: str = "metabs_freqs_hz"
) -> xr.Dataset:
    """Prepare the notebook's five-peak fit with explicit Hz and second units.

    Keep all 31 grid points and 11 alternating frequency/T2* passes. The small
    number of spectra bounds runtime without changing the numerical method.
    ``frequency_key`` selects the canonical Hz key or the notebook alias.
    """
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": METABOLITES,
            frequency_key: FREQUENCIES_HZ.tolist(),
            "initial_t2s_s": [T2_S] * 5,
            "metabs_freq_ranges": [70.0] * 5,
            "metabs_t2_ranges": [0.10] * 5,
            "min_t2_s": [0.001] * 5,
            "max_t2_s": [0.16] * 5,
            "signal_domain": "temporal",
            "snr_threshold": None,
            "npoints": 31,
            "niter": 1,
            "rep_fitting": 11,
            "zoomfactor": 1.5,
            "amp_solver": "pinv",
            "amp_ridge_lambda": 0.0,
            "min_peak_separation_hz": 5.0,
        },
    )


@pytest.fixture(scope="module")
def noisy_fit(notebook_fids: xr.Dataset) -> xr.Dataset:
    """Fit the two noisy FIDs once as the serial execution reference."""
    data = notebook_fids["noisy"]
    return fit_data_pseudo_inv(
        data, _prepare_notebook_fit_params(data), n_jobs=1, show_tqdm=False
    )


def test_clean_fids_recover_parameters_from_shifted_start(notebook_fids):
    """Recover known peaks from imperfect initial values, including a null peak."""
    data = notebook_fids["clean"]
    params = _prepare_notebook_fit_params(data)
    params["initial_freqs_hz"] = params["initial_freqs_hz"] + 3.0
    params["initial_t2s_s"] = xr.full_like(params["initial_t2s_s"], 0.03)
    params = _update_frequency_bounds(params)
    result = fit_data_pseudo_inv(data, params, n_jobs=1, show_tqdm=False)

    # Grid-search tolerances: 0.2% of pyruvate amplitude, 0.3 Hz, and 0.3 ms.
    # A zero-amplitude peak has no identifiable frequency or T2*.
    np.testing.assert_allclose(
        result["fit_amps"], notebook_fids["true_amplitudes"], rtol=0, atol=0.002
    )
    present = notebook_fids["true_amplitudes"].values > 0
    expected_frequencies = np.broadcast_to(FREQUENCIES_HZ, present.shape)
    np.testing.assert_allclose(
        result["fit_freqs"].values[present],
        expected_frequencies[present],
        rtol=0,
        atol=0.3,
    )
    np.testing.assert_allclose(
        result["fit_t2s"].values[present], T2_S, rtol=0, atol=0.0003
    )
    reconstructed = result.reconstruct().sum("metabolite")
    # The existing reconstruction accessor labels acquisition_idx in seconds.
    np.testing.assert_allclose(
        reconstructed.coords["acquisition_idx"], data.coords["sampling_time_axis_s"]
    )
    relative_error = np.linalg.norm(reconstructed.values - data.values, axis=-1)
    relative_error /= np.linalg.norm(data.values, axis=-1)
    assert np.all(relative_error < 0.003)


def test_noisy_fids_preserve_signal_and_result_contract(notebook_fids, noisy_fit):
    """Check signal recovery, bounds, units and provenance at quadrature SNR 50."""
    data = notebook_fids["noisy"]
    assert noisy_fit["fit_amps"].dims == ("bicarbonate_amplitude", "metabolite")
    assert noisy_fit.coords["metabolite"].values.tolist() == METABOLITES
    xr.testing.assert_identical(
        noisy_fit.coords["bicarbonate_amplitude"], data.coords["bicarbonate_amplitude"]
    )
    for name, unit in (("fit_amps", "a.u."), ("fit_freqs", "Hz"), ("fit_t2s", "s")):
        assert noisy_fit[name].attrs["unit"] == unit
        assert np.isfinite(noisy_fit[name]).all()
    for name in ("signal_domain", "simulation_source", "random_seed", "noise_std"):
        assert noisy_fit.attrs[name] == data.attrs[name]
    assert noisy_fit.attrs["model_name"] == "FID"
    assert noisy_fit.attrs["amp_solver"] == "pinv"
    assert noisy_fit.attrs["rep_fitting"] == 11
    assert ((noisy_fit["fit_t2s"] >= 0.001) & (noisy_fit["fit_t2s"] <= 0.16)).all()
    assert (np.abs(noisy_fit["fit_freqs"].values - FREQUENCIES_HZ) <= 35.0).all()

    # For populated peaks, allow 1.5% of pyruvate's amplitude for this draw.
    # The absent peak can fit noise with a short T2* and a nonzero amplitude;
    # its effect is covered by the reconstructed-signal check below.
    present = notebook_fids["true_amplitudes"].values > 0
    np.testing.assert_allclose(
        noisy_fit["fit_amps"].values[present],
        notebook_fids["true_amplitudes"].values[present],
        rtol=0,
        atol=0.015,
    )
    reconstructed = reconstruct_fitted_signal(noisy_fit, data)
    legacy_signal = noisy_fit.reconstruct().sum("metabolite")
    np.testing.assert_array_equal(reconstructed.values, legacy_signal.values)
    residual = data - reconstructed
    assert residual.sizes == data.sizes
    assert np.isfinite(residual).all()
    clean = notebook_fids["clean"]
    error = np.linalg.norm((reconstructed - clean).values, axis=-1)
    assert np.all(error / np.linalg.norm(clean.values, axis=-1) < 0.03)


@pytest.mark.parametrize(
    ("n_jobs", "show_tqdm"),
    [(1, False), (1, True), (2, False)],
    ids=["serial", "serial-progress", "parallel-loky"],
)
def test_transposed_fids_match_serial_reference(
    notebook_fids, noisy_fit, n_jobs, show_tqdm
):
    """Preserve numerical outputs across axis orders, progress mode and workers."""
    data = notebook_fids["noisy"].transpose("acquisition_idx", "bicarbonate_amplitude")
    original = data.copy(deep=True)
    result = fit_data_pseudo_inv(
        data,
        _prepare_notebook_fit_params(data),
        n_jobs=n_jobs,
        batch_size=1,
        parallel_backend="loky",
        show_tqdm=show_tqdm,
    )
    # Allow floating-point differences between serial and worker BLAS libraries.
    xr.testing.assert_allclose(result, noisy_fit, rtol=1e-9, atol=1e-10)
    xr.testing.assert_identical(data, original)


@pytest.mark.parametrize("entry_point", ["api", "workflow"])
def test_fitting_entry_points_match_notebook_results(
    notebook_fids, noisy_fit, entry_point
):
    """Use one execution path for explicit bounds, numerical outputs and metadata."""
    data = notebook_fids["noisy"]
    params = _prepare_notebook_fit_params(data)
    if entry_point == "api":
        result = api.fit(data, model="FID", fit_params=params)
    else:
        result = fit_data_pseudo_inv(data, params)

    xr.testing.assert_identical(result, noisy_fit)


def test_notebook_hz_alias_without_frequency_axes_preserves_fit(
    notebook_fids, noisy_fit
):
    """Recover the calibrated reference using only Hz values and the sampling axis."""
    data = notebook_fids["noisy"].drop_vars(["freq_axis_hz", "freq_axis_ppm"])
    # Earlier parameter preparation may have cached frequency axes in attrs.
    data.attrs = {"signal_domain": "temporal"}
    params = _prepare_notebook_fit_params(data, frequency_key="metabs_freqs_Hz")
    result = fit_data_pseudo_inv(data, params, n_jobs=1, show_tqdm=False)

    for name in noisy_fit.data_vars:
        xr.testing.assert_identical(result[name], noisy_fit[name])
    assert result.attrs["metabs_freqs_ppm"] == [None] * len(METABOLITES)
    assert result.attrs["freq_axis_ppm"] is None
