"""Exercise calibration options on invented receiver-resolved RF-power curves.

Only the spectral fit and its reconstruction are replaced; the real
parameter preparation runs. The iterative B1 calculation
and pulse adjustment run normally. Comparisons with the canonical calculation
use rtol=1e-12 and atol=1e-14 for identical float64 arithmetic, while known
noiseless recovery allows 1e-4 W for the discretized period search.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.anatomical import b1_mapping
from hypermri.analysis.quantitative import fit_b1_calibration


@pytest.fixture(params=[1, 2], ids=["one_receiver", "two_receivers"])
def calibration_case(request, monkeypatch):
    from hypermri.analysis.fitting import spectroscopy

    powers = np.linspace(0.0, 16.0, 9)
    signal = (1.4 - 0.6j) * np.sin(np.pi / 4.25 * np.sqrt(powers))
    amplitudes = signal[:, None] * np.array([1.0, 0.7])[: request.param]
    fid = np.ones((9, request.param, 8), dtype=np.complex64)
    fid.flags.writeable = False
    method = {"PulseVectorStart": 0.0, "PulseVectorEnd": 16.0, "Method": "invented"}
    scan = SimpleNamespace(
        n_repetitions=9,
        n_receivers=request.param,
        rfpulse_duration_ms=1.2,
        rfpulse_sint=0.8,
        method=method,
        method_parameter=method.__getitem__,
        fid=fid,
        dwell_times_s=1e-3,
    )
    spectral_calls = []
    voxel_calls = []

    def fit_spectrum(data, fit_params, n_jobs=1):
        spectral_calls.append({"data": data, "n_jobs": n_jobs})
        dims = ("repetition_idx", "channel_idx", "metabolite")
        return xr.Dataset({"fit_amps": (dims, amplitudes[:, :, None])})

    original_voxel_fit = b1_mapping.fit_single_voxel

    def record_voxel_fit(**kwargs):
        result = original_voxel_fit(**kwargs)
        voxel_calls.append((kwargs, result))
        return result

    monkeypatch.setattr(spectroscopy, "fit_data_pseudo_inv", fit_spectrum)
    monkeypatch.setattr(
        spectroscopy, "reconstruct_fitted_signal", lambda fit, data: data * 0
    )
    monkeypatch.setattr(b1_mapping, "fit_single_voxel", record_voxel_fit)
    return SimpleNamespace(
        scan=scan,
        powers=powers,
        amplitudes=amplitudes,
        spectral_calls=spectral_calls,
        voxel_calls=voxel_calls,
    )


def _run(case, **options):
    before = case.scan.fid.copy()
    result = b1_mapping.b1_calibration(
        case.scan, plot=False, db=False, n_jobs=1, **options
    )
    np.testing.assert_array_equal(case.scan.fid, before)
    assert case.scan.fid.dtype == np.complex64
    assert len(case.spectral_calls) == 1
    assert case.spectral_calls[0]["n_jobs"] == 1
    # local_stop applies after spectral fitting; no FID samples are discarded.
    assert case.spectral_calls[0]["data"].size == case.scan.fid.size
    assert len(case.voxel_calls) == case.scan.n_receivers
    return result


def _assert_calculation(case, result, *, end, center, extent):
    expected_last_power = None
    for channel, (options, actual) in enumerate(case.voxel_calls):
        expected_signal = case.amplitudes[:end, channel]
        expected_powers = case.powers[:end]
        np.testing.assert_array_equal(options["values"], expected_signal)
        np.testing.assert_array_equal(options["refpower_W"], expected_powers)
        expected = fit_b1_calibration(
            expected_signal,
            expected_powers,
            initial_guess_cent=center,
            initial_guess_extent=extent,
        )
        for actual_value, expected_value in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=1e-12, atol=1e-14
            )
        expected_last_power = (expected[1] / 2) ** 2 * (1.2 * 0.8) ** 2
    np.testing.assert_allclose(result, expected_last_power, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("center", [3.1, 2.6, None, np.float32(3.1)])
def test_implicit_width_preserves_historical_window_and_last_sample_exclusion(
    calibration_case, center
):
    result = _run(calibration_case, initial_guess_cent=center)
    _assert_calculation(calibration_case, result, end=8, center=center, extent=center)


@pytest.mark.parametrize(
    "local_stop,end", [(-2, 7), (np.int64(-2), 7), (3, 3), (None, 9), (9, 9), (99, 9)]
)
def test_explicit_selection_preserves_python_slice_semantics(
    calibration_case, local_stop, end
):
    result = _run(calibration_case, local_stop=local_stop)
    _assert_calculation(calibration_case, result, end=end, center=3.1, extent=3.1)


def test_explicit_width_reaches_each_receiver_and_controls_numerical_search(
    calibration_case,
):
    result = _run(calibration_case, initial_guess_cent=3.1, initial_guess_extent=0.2)
    _assert_calculation(calibration_case, result, end=8, center=3.1, extent=0.2)
    for options, actual in calibration_case.voxel_calls:
        assert options["initial_guess_extent"] == 0.2
        historical_ignored_width = fit_b1_calibration(
            options["values"],
            options["refpower_W"],
            initial_guess_cent=3.1,
            initial_guess_extent=3.1,
        )
        # A deliberately narrow window excludes the invented 4.25 sqrt(W)
        # solution. A >0.1 sqrt(W) difference proves the option affects fitting,
        # beyond float64 roundoff or grid-resolution differences.
        assert abs(actual[1] - historical_ignored_width[1]) > 0.1


def test_none_center_preserves_domain_default_with_independent_explicit_width(
    calibration_case,
):
    result = _run(calibration_case, initial_guess_cent=None, initial_guess_extent=1.3)
    _assert_calculation(calibration_case, result, end=8, center=None, extent=1.3)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"initial_guess_cent": 0.0}, "initial_guess_cent"),
        ({"initial_guess_cent": np.nan}, "initial_guess_cent"),
        ({"initial_guess_cent": True}, "initial_guess_cent"),
        ({"initial_guess_cent": 3.1 + 0.1j}, "initial_guess_cent"),
        ({"initial_guess_extent": -1.0}, "initial_guess_extent"),
        ({"initial_guess_extent": np.inf}, "initial_guess_extent"),
        ({"initial_guess_extent": "3.1"}, "initial_guess_extent"),
        ({"n_jobs": True}, "n_jobs"),
        ({"n_jobs": 1.0}, "n_jobs"),
        ({"n_jobs": "1"}, "n_jobs"),
        ({"local_stop": False}, "local_stop"),
        ({"local_stop": 3.5}, "local_stop"),
        ({"local_stop": "3"}, "local_stop"),
    ],
)
def test_invalid_options_fail_before_reading_acquired_data(options, message):
    with pytest.raises(ValueError, match=message):
        b1_mapping.b1_calibration(None, plot=False, **options)


@pytest.mark.parametrize("local_stop", [2, 0, -99])
def test_too_short_selection_fails_before_spectral_fitting(
    calibration_case, local_stop
):
    fit_options = {}
    with pytest.raises(ValueError, match="local_stop.*at least three"):
        b1_mapping.b1_calibration(
            calibration_case.scan, fit_options, plot=False, local_stop=local_stop
        )
    assert calibration_case.spectral_calls == []
    assert calibration_case.voxel_calls == []
    assert fit_options == {}


@pytest.mark.parametrize("name", ["local_stpo", "niter", "stop_index"])
def test_unknown_keywords_are_rejected_instead_of_silently_ignored(name):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
        b1_mapping.b1_calibration(None, plot=False, **{name: 3})


def test_shorter_spectral_output_does_not_silently_change_power_pairing(
    calibration_case, monkeypatch
):
    from hypermri.analysis.fitting import spectroscopy

    monkeypatch.setattr(
        spectroscopy,
        "fit_data_pseudo_inv",
        lambda data, fit_params, n_jobs=1: xr.Dataset(
            {
                "fit_amps": (
                    ("repetition_idx", "channel_idx", "metabolite"),
                    calibration_case.amplitudes[:-1, :, None],
                )
            }
        ),
    )
    # Eight amplitudes for nine powers must not be paired with unrelated
    # power labels.
    with pytest.raises(ValueError, match="one fitted amplitude per repetition"):
        b1_mapping.b1_calibration(calibration_case.scan, plot=False)


@pytest.mark.parametrize("local_stop,end", [(-2, 7), (None, 9)])
def test_plot_uses_normalized_cutoff_and_handles_all_samples(
    calibration_case, monkeypatch, local_stop, end
):
    monkeypatch.setattr(b1_mapping.plt, "show", lambda: None)
    before = set(b1_mapping.plt.get_fignums())
    try:
        result = b1_mapping.b1_calibration(
            calibration_case.scan,
            plot=True,
            local_stop=local_stop,
        )
        _assert_calculation(calibration_case, result, end=end, center=3.1, extent=3.1)
        power_axes = [
            axis
            for figure_id in set(b1_mapping.plt.get_fignums()) - before
            for axis in b1_mapping.plt.figure(figure_id).axes
            if axis.get_xlabel() == "refpower [W]"
        ]
        assert len(power_axes) == calibration_case.scan.n_receivers
        for axis in power_axes:
            cutoffs = [
                line
                for line in axis.lines
                if line.get_label().startswith("Stop index =")
            ]
            if end < calibration_case.scan.n_repetitions:
                assert len(cutoffs) == 1
                np.testing.assert_array_equal(
                    cutoffs[0].get_xdata(), [calibration_case.powers[end]] * 2
                )
            else:
                assert cutoffs == []
    finally:
        for figure_id in set(b1_mapping.plt.get_fignums()) - before:
            b1_mapping.plt.close(figure_id)


@pytest.mark.parametrize("n_receivers", [1, 2])
def test_calibration_runs_the_real_spectral_fit_end_to_end(n_receivers, monkeypatch):
    # Invented single-line FIDs (0 Hz, T2* 20 ms, 256 samples at 0.5 ms) whose
    # amplitude follows sin(pi / 4.25 * sqrt(P)); the second receiver sees 0.7x
    # the signal with another phase. Nothing is mocked. P90 = (4.25 / 2)^2
    # scaled by pulse duration and integral; 1e-3 relative covers the
    # discretized period search and the noiseless spectral fit.
    powers = np.linspace(0.0, 16.0, 9)
    times_s = np.arange(256) * 5e-4
    line = np.exp(-times_s / 0.02)
    gains = np.array([1.4 - 0.6j, 0.7 * (0.3 + 1.0j)])[:n_receivers]
    amplitudes = np.sin(np.pi / 4.25 * np.sqrt(powers))[:, None] * gains
    fid = amplitudes[:, :, None] * line
    method = {"PulseVectorStart": 0.0, "PulseVectorEnd": 16.0, "Method": "invented"}
    scan = SimpleNamespace(
        n_repetitions=9,
        n_receivers=n_receivers,
        rfpulse_duration_ms=1.2,
        rfpulse_sint=0.8,
        method=method,
        method_parameter=method.__getitem__,
        fid=fid.copy(),
        dwell_times_s=5e-4,
    )
    fit_params = {"metabs": ["peak1"], "metabs_freqs_hz": 0, "cut_off": 0}
    result = b1_mapping.b1_calibration(
        scan, dict(fit_params), plot=False, local_stop=None
    )
    expected = (4.25 / 2.0) ** 2 * (1.2 * 0.8) ** 2
    np.testing.assert_allclose(result, expected, rtol=1e-3)
    np.testing.assert_array_equal(scan.fid, fid)
    monkeypatch.setattr(b1_mapping.plt, "show", lambda: None)
    before = set(b1_mapping.plt.get_fignums())
    try:
        # Default options (cut_off 70) with plots of measured and fitted spectra.
        default = b1_mapping.b1_calibration(scan, plot=True, local_stop=None)
        added = set(b1_mapping.plt.get_fignums()) - before
        spectra_axes = [
            axis
            for figure_id in added
            for axis in b1_mapping.plt.figure(figure_id).axes
            if axis.get_xlabel() == "Frequency [Hz]"
        ]
        assert len(spectra_axes) == 9 * n_receivers
        assert all(len(axis.lines) == 2 for axis in spectra_axes)
    finally:
        for figure_id in set(b1_mapping.plt.get_fignums()) - before:
            b1_mapping.plt.close(figure_id)
    np.testing.assert_allclose(default, expected, rtol=1e-3)


def test_calibration_rejects_spectral_input_domain_and_missing_dwell_time():
    method = {"PulseVectorStart": 1.0, "PulseVectorEnd": 4.0, "Method": "invented"}
    scan = SimpleNamespace(
        n_repetitions=4,
        n_receivers=1,
        rfpulse_duration_ms=1.0,
        rfpulse_sint=1.0,
        method=method,
        method_parameter=method.__getitem__,
        fid=np.ones(32, dtype=complex),
        dwell_times_s=0,
    )
    with pytest.raises(ValueError, match="signal_domain"):
        b1_mapping.b1_calibration(
            scan, {"metabs": ["peak1"], "signal_domain": "spectral"}, plot=False
        )
    with pytest.raises(ValueError, match="positive dwell time"):
        b1_mapping.b1_calibration(scan, plot=False)
