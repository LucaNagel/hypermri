"""Characterize the retained iterative calibration on invented complex samples.

The frozen reference comes from a0fa88d6f4e045c71c2e8817bb691f7df412223c,
before the incompatible single-sweep adapter replaced the iterative solver.
rtol=1e-12 and atol=1e-14 preserve its float64 arithmetic and diagnostics;
they do not define physiological accuracy or noisy-data recovery accuracy.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.anatomical import b1_mapping
from hypermri.analysis.quantitative import fit_b1_calibration


def _reference():
    path = Path(__file__).resolve().parents[3] / "reference"
    return json.loads((path / "b1_calibration_iterative.json").read_text())


def _reference_inputs():
    reference = _reference()
    signal = np.array(reference["signal_real"]) + 1j * np.array(
        reference["signal_imag"]
    )
    return reference, signal, np.array(reference["power_W"])


def _assert_reference(result, expected):
    amplitude, period, curve, rss, periods = result
    expected_amplitude = complex(expected["amplitude_real"], expected["amplitude_imag"])
    expected_curve = np.array(expected["fit_real"]) + 1j * np.array(
        expected["fit_imag"]
    )
    for actual, reference in (
        (amplitude, expected_amplitude),
        (period, expected["period"]),
        (curve, expected_curve),
        (rss, expected["rss"]),
        (periods, expected["periods"]),
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)


def test_single_voxel_wrapper_preserves_historical_fit_and_trajectories():
    reference, signal, power = _reference_inputs()
    result = b1_mapping.fit_single_voxel(signal, power, **reference["options"])
    _assert_reference(result, reference["cases"][0])


@pytest.mark.parametrize("case_index", [0, 1])
def test_calibration_preserves_pinned_numerical_results(case_index):
    reference, signal, power = _reference_inputs()
    case = reference["cases"][case_index]
    result = fit_b1_calibration(
        signal, power, stop_index=case["stop_index"], **reference["options"]
    )
    _assert_reference(result, case)
    if case["stop_index"] is not None:
        assert np.count_nonzero(result[2][case["stop_index"] :]) == 0


@pytest.mark.parametrize("storage_dtype", [np.float32, np.complex64, np.complex128])
def test_calibration_preserves_inputs_precision_shapes_and_repeatability(storage_dtype):
    reference, signal, power = _reference_inputs()
    signal = (signal.real if storage_dtype is np.float32 else signal).astype(
        storage_dtype
    )
    power = power.astype(np.float32)
    original_signal, original_power = signal.copy(), power.copy()
    signal.flags.writeable = False
    power.flags.writeable = False
    first = fit_b1_calibration(signal, power, **reference["options"])
    second = fit_b1_calibration(signal, power, **reference["options"])
    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert isinstance(first[0], complex)
    assert isinstance(first[1], float)
    assert first[2].dtype == np.complex128
    assert first[2].shape == signal.shape
    for trajectory in first[3:]:
        assert trajectory.dtype == np.float64
        assert trajectory.shape == (4, 11)
    np.testing.assert_array_equal(signal, original_signal)
    np.testing.assert_array_equal(power, original_power)


def test_calibration_recovers_known_complex_sinusoid_and_default_guesses():
    # Discretized ten-iteration search resolves the invented period within
    # 1e-4 sqrt(W) and amplitude within 1e-4 signal units (no noise or MRI data).
    power = np.linspace(0.0, 16.0, 17)
    amplitude = 2.0 + 0.5j
    signal = amplitude * np.sin(np.pi / 4.0 * np.sqrt(power))
    result = fit_b1_calibration(signal, power)
    np.testing.assert_allclose(result[0], amplitude, rtol=0, atol=1e-4)
    np.testing.assert_allclose(result[1], 4.0, rtol=0, atol=1e-4)
    explicit = fit_b1_calibration(
        signal, power, initial_guess_cent=3.0, initial_guess_extent=3.0
    )
    for actual, expected in zip(result, explicit, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_single_voxel_delegates_once_with_every_search_option(monkeypatch):
    reference, signal, power = _reference_inputs()
    calls = []
    expected = (1.0 + 0.0j, 4.0, signal, np.zeros((4, 11)), np.ones((4, 11)))

    def capture(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(b1_mapping, "_fit_b1_calibration", capture)
    result = b1_mapping.fit_single_voxel(
        signal, power, stop_index_extra=2, **reference["options"]
    )
    assert result is expected
    assert len(calls) == 1
    assert calls[0]["values"] is signal
    assert calls[0]["refpower_W"] is power
    assert {key: calls[0][key] for key in reference["options"]} == reference["options"]
    assert calls[0]["stop_index_extra"] == 2


def test_historical_extra_index_option_does_not_invent_a_cutoff():
    reference, signal, power = _reference_inputs()
    result = fit_b1_calibration(
        signal, power, stop_index_extra=4, **reference["options"]
    )
    _assert_reference(result, reference["cases"][0])


def test_zero_signal_retains_historical_nonfit_sentinels():
    result = fit_b1_calibration(
        np.zeros(5, dtype=complex), np.arange(5), niter=3, npoints=7
    )
    assert result[0] == 0j
    assert np.isnan(result[1])
    assert np.count_nonzero(result[2]) == 0
    assert result[2].dtype == np.complex128
    for trajectory in result[3:]:
        assert trajectory.shape == (3, 7)
        assert np.isnan(trajectory).all()


@pytest.mark.parametrize(
    ("signal", "power", "message"),
    [
        ([[1.0, 2.0, 3.0]], [0.0, 1.0, 2.0], "one-dimensional"),
        ([1.0, 2.0, 3.0], [[0.0, 1.0, 2.0]], "one-dimensional"),
        ([1.0, 2.0, 3.0], [0.0, 1.0], "matching"),
        ([], [], "at least three"),
        ([1.0, 2.0], [0.0, 1.0], "at least three"),
        ([1.0, np.nan, 3.0], [0.0, 1.0, 2.0], "values.*finite"),
        ([1.0, 2.0, 3.0], [0.0, -1.0, 2.0], "refpower_W"),
        ([1.0, 2.0, 3.0], [0.0, np.inf, 2.0], "refpower_W"),
    ],
)
def test_invalid_sample_arrays_fail_with_actionable_error(signal, power, message):
    with pytest.raises(ValueError, match=message):
        fit_b1_calibration(signal, power)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("initial_guess_cent", 0.0),
        ("initial_guess_cent", np.nan),
        ("initial_guess_extent", -1.0),
        ("initial_guess_extent", np.inf),
        ("npoints", 1),
        ("npoints", 2.5),
        ("npoints", True),
        ("niter", 0),
        ("zoom_factor", 1.0),
        ("zoom_factor", np.nan),
        ("stop_index", 2),
        ("stop_index", 10),
        ("stop_index", 3.5),
        ("stop_index_extra", 1.5),
    ],
)
def test_invalid_search_options_fail_with_actionable_error(name, value):
    with pytest.raises(ValueError, match=name):
        fit_b1_calibration([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], **{name: value})


def test_plot_displays_fit_and_all_trajectories_without_refitting(monkeypatch):
    reference, signal, power = _reference_inputs()
    calls = []
    shown = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return fit_b1_calibration(*args, **kwargs)

    monkeypatch.setattr(b1_mapping, "_fit_b1_calibration", capture)
    monkeypatch.setattr(b1_mapping.plt, "show", lambda: shown.append(True))
    result = b1_mapping.fit_single_voxel(
        signal, power, plot=True, **reference["options"]
    )
    figure = b1_mapping.plt.gcf()
    try:
        assert len(calls) == 1
        assert shown == [True]
        assert len(figure.axes) == 2
        assert len(figure.axes[0].lines) == 6
        for iteration, line in enumerate(figure.axes[1].lines[:4]):
            np.testing.assert_allclose(line.get_xdata(), result[4][iteration])
            np.testing.assert_allclose(line.get_ydata(), result[3][iteration])
    finally:
        b1_mapping.plt.close(figure)


@pytest.mark.parametrize("n_channels", [1, 2])
def test_spectroscopic_calibration_caller_can_consume_restored_searches(
    monkeypatch, n_channels
):
    from hypermri.analysis.fitting import spectroscopy

    # Spectroscopy is isolated so this tests the calibration handoff, channel
    # iteration and diagnostic-history consumer without changing its fit model.
    power = np.linspace(0.0, 16.0, 9)
    signal = (1.4 - 0.6j) * np.sin(np.pi / 4.25 * np.sqrt(power))
    amplitudes = signal if n_channels == 1 else np.column_stack([signal, 0.7 * signal])
    method = {"PulseVectorStart": 0.0, "PulseVectorEnd": 16.0, "Method": "invented"}
    data = SimpleNamespace(
        n_repetitions=9,
        n_receivers=n_channels,
        rfpulse_duration_ms=1.2,
        rfpulse_sint=0.8,
        method=method,
        method_parameter=method.__getitem__,
        fid=np.ones((9, n_channels, 8), dtype=np.complex64),
        dwell_times_s=1e-3,
    )
    original_fid = data.fid.copy()

    def fit_spectrum(data, fit_params, n_jobs=1):
        assert n_jobs == 1
        values = np.reshape(amplitudes, (9, n_channels, 1))
        dims = ("repetition_idx", "channel_idx", "metabolite")
        return xr.Dataset({"fit_amps": (dims, values)})

    monkeypatch.setattr(spectroscopy, "fit_data_pseudo_inv", fit_spectrum)
    before = set(b1_mapping.plt.get_fignums())
    try:
        result = b1_mapping.b1_calibration(data, plot=False, db=True, n_jobs=1)
        # For the model above, P90 = (period / 2)^2, with the existing pulse
        # duration/integral scaling. 1e-4 W allows discretized grid resolution.
        expected = (4.25 / 2.0) ** 2 * (1.2 * 0.8) ** 2
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(data.fid, original_fid)
        added = set(b1_mapping.plt.get_fignums()) - before
        assert len(added) == n_channels
        for figure_id in added:
            figure = b1_mapping.plt.figure(figure_id)
            histories = [axis for axis in figure.axes if axis.lines]
            assert len(histories) == 10
            assert all(len(axis.lines) == 1 for axis in histories)
    finally:
        for figure_id in set(b1_mapping.plt.get_fignums()) - before:
            b1_mapping.plt.close(figure_id)
