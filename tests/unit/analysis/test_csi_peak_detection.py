"""Exercise widget-free CSI detection on invented seed-11 Gaussian spectra.

The two centres are -20 and 30 Hz. Recovery uses a 1.3 Hz absolute tolerance
for the 1.26 Hz frequency grid; repeated/permuted calculations use 1e-12
relative tolerance for unchanged float64 arithmetic. Complex storage is tested
at both precisions. This is a structural reference, not a lineshape validation.
"""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy import (
    CSIPeakDetectionOptions,
    detect_csi_peaks,
    pipeline,
)


def _spectra(dtype=np.complex64, domain="spectral", permuted=False):
    frequency_hz = np.linspace(-80.0, 80.0, 128)
    rng = np.random.default_rng(11)
    signal = np.exp(-0.5 * ((frequency_hz + 20) / 2) ** 2) + 0.7 * np.exp(
        -0.5 * ((frequency_hz - 30) / 2.5) ** 2
    )
    values = np.stack(
        [signal + 0.01 + rng.normal(0, 0.001, signal.shape) for _ in range(4)], axis=1
    ).astype(dtype)
    if domain == "temporal":
        values = np.fft.ifft(np.fft.ifftshift(values, axes=0), axis=0)
    data = xr.DataArray(
        values,
        dims=("acquisition_idx", "read_idx"),
        coords={"acquisition_idx": np.arange(128) * 0.001, "read_idx": [3, 2, 1, 0]},
        attrs={
            "signal_domain": domain,
            "freq_axis_hz": frequency_hz,
            "freq_axis_ppm": np.linspace(160, 190, 128),
            "units": "a.u.",
            "dwell_time_s": 0.001,
            "provenance": "invented seed-11 Gaussian spectra",
        },
    )
    return data.transpose("read_idx", "acquisition_idx") if permuted else data


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("domain", ["temporal", "spectral"])
@pytest.mark.parametrize("permuted", [False, True])
def test_detection_retains_units_axes_precision_input_and_repeatability(
    dtype, domain, permuted
):
    data = _spectra(dtype, domain, permuted)
    before = data.copy(deep=True)
    options = CSIPeakDetectionOptions(max_peaks=2)
    first = detect_csi_peaks(data, options)
    repeated = detect_csi_peaks(data, options)
    canonical = detect_csi_peaks(data.transpose("acquisition_idx", "read_idx"), options)
    for result in (first, repeated, canonical):
        assert result["options"] is options
        assert result["spectral_data"].dims == (
            "acquisition_idx",
            "slice_idx",
            "read_idx",
            "phase_idx",
            "repetition_idx",
            "channel_idx",
        )
        assert result["spectral_data"].dtype == dtype
        assert result["spectral_data"].attrs["signal_domain"] == "spectral"
        assert result["spectral_data"].attrs["units"] == "a.u."
        assert result["spectral_data"].attrs["provenance"] == data.attrs["provenance"]
        np.testing.assert_array_equal(result["spectral_data"].read_idx, [3, 2, 1, 0])
        np.testing.assert_array_equal(
            result["freq_axis_hz"], data.attrs["freq_axis_hz"]
        )
        np.testing.assert_array_equal(
            result["freq_axis_ppm"], data.attrs["freq_axis_ppm"]
        )
        assert result["freq_axis_hz"].dtype == np.float64
        assert result["peak_detection"]["n_spectra_total"] == 4
        assert np.isnan(result["temporal_noise_std"])
        assert result["snr_threshold"] is None
        np.testing.assert_allclose(
            [item["center_hz"] for item in result["cluster_summary"]],
            [-20, 30],
            atol=1.3,
            rtol=0,
        )
        for key in ("freq_ranges_hz", "initial_t2s_s", "t2_ranges_s"):
            np.testing.assert_allclose(result[key], first[key], rtol=1e-12)
        assert result["cluster_summary"] == first["cluster_summary"]
    xr.testing.assert_identical(data, before)


def test_empty_detection_is_a_result_without_a_workflow_decision():
    data = _spectra()
    data.values[:] = 0
    result = detect_csi_peaks(data)
    for key in (
        "candidates",
        "clusters",
        "cluster_summary",
        "freq_ranges_hz",
        "initial_t2s_s",
        "t2_ranges_s",
    ):
        assert result[key] == []
    assert result["n_clusters_after_filtering"] == 0


def test_missing_ppm_is_reported_without_inventing_calibration():
    data = _spectra()
    del data.attrs["freq_axis_ppm"]
    result = detect_csi_peaks(data, CSIPeakDetectionOptions(max_peaks=1))
    assert result["freq_axis_ppm"] is None
    assert np.isnan(result["cluster_summary"][0]["center_ppm"])


@pytest.mark.parametrize("legacy_factor", [None, 4.0])
def test_detection_prepares_each_required_domain_once_without_ui_or_fitting(
    monkeypatch, legacy_factor
):
    calls = []
    original = pipeline._to_standard_signal_domain

    def record(data, target):
        calls.append(target)
        return original(data, target)

    def forbidden(*args, **kwargs):
        raise AssertionError("Detection must not fit, plot or construct widgets.")

    from ipywidgets import Widget

    monkeypatch.setattr(pipeline, "_to_standard_signal_domain", record)
    monkeypatch.setattr(pipeline, "fit_data_pseudo_inv", forbidden)
    monkeypatch.setattr(pipeline, "_render_peak_review_plot", forbidden)
    monkeypatch.setattr(Widget, "__init__", forbidden)
    result = detect_csi_peaks(
        _spectra(), CSIPeakDetectionOptions(snr_threshold_factor=legacy_factor)
    )
    assert calls == (
        ["spectral"] if legacy_factor is None else ["spectral", "temporal"]
    )
    if legacy_factor is not None:
        np.testing.assert_allclose(
            result["snr_threshold"], 4 * result["temporal_noise_std"], rtol=1e-12
        )


def test_options_are_immutable_and_reject_unknown_fields():
    options = CSIPeakDetectionOptions()
    with pytest.raises(FrozenInstanceError):
        options.max_peaks = 2
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        CSIPeakDetectionOptions(unknown_option=2)


def test_unknown_option_object_fails_before_reading_samples():
    with pytest.raises(TypeError, match="CSIPeakDetectionOptions"):
        detect_csi_peaks(None, {"max_peaks": 2})


def test_missing_frequency_metadata_names_the_required_calibration():
    data = _spectra()
    del data.attrs["freq_axis_hz"]
    del data.attrs["dwell_time_s"]
    with pytest.raises(ValueError, match="freq_axis_hz.*provide"):
        detect_csi_peaks(data)


def test_detection_leaves_the_input_unchanged():
    data = _spectra()
    before = data.copy(deep=True)
    detect_csi_peaks(data)
    xr.testing.assert_identical(data, before)


def test_auto_pipeline_delegates_once_preserving_disabled_tail_and_result_contract(
    monkeypatch,
):
    data = _spectra()
    before = data.copy(deep=True)
    calls = []
    original = pipeline._detect_csi_peaks_from_spectra

    def record(data, spectra, frequency_hz, frequency_ppm, step_hz, options):
        result = original(data, spectra, frequency_hz, frequency_ppm, step_hz, options)
        calls.append((options, result))
        return result

    monkeypatch.setattr(pipeline, "_detect_csi_peaks_from_spectra", record)
    monkeypatch.setattr(
        pipeline,
        "fit_data_pseudo_inv",
        lambda data, fit_params, show_tqdm=None: xr.Dataset(attrs={"fit": "stub"}),
    )
    monkeypatch.setattr(
        pipeline,
        "initialise_fit_params_from_results",
        lambda *, fit_params, fit_results, freq_ranges_hz, t2_ranges_s: fit_params.copy(
            deep=True
        ),
    )
    output = pipeline.run_csi_auto_fitting_pipeline(data, max_peaks=1, verbose=False)
    assert len(calls) == 1
    options, detected = calls[0]
    assert options.max_peaks == 1
    assert options.cluster_min_separation_hz is None
    assert options.snr_threshold_factor is None
    assert output["auto_diagnostics"]["cluster_summary"] == detected["cluster_summary"]
    assert output["auto_fit_config"]["metabs_freq_ranges"] == detected["freq_ranges_hz"]
    assert output["auto_diagnostics"]["snr_threshold"] is None
    xr.testing.assert_identical(data, before)
