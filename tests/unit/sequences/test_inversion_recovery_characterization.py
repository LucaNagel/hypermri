"""Characterize the injected legacy multi-peak workflow previously extraction.

Invented noiseless FIDs contain two isolated spectral bins and an absolute
exponential inversion curve. Signal/plot tolerances are 2e-6 a.u. for complex64
storage; T1 recovery uses 0.1 ms, well below the 50 ms sampling interval.
"""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.sequences.IVRecovery import IVRecovery


class _Axes:
    def __init__(self):
        self.lines = []
        self.points = []
        self.title = None
        self.xlabel = None
        self.ylabel = None
        self.legends = 0

    def plot(self, x, y, **options):
        self.lines.append((np.array(x), np.array(y), options))

    def scatter(self, x, y, **options):
        self.points.append((np.array(x), np.array(y), options))

    def set_title(self, title):
        self.title = title

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def legend(self):
        self.legends += 1


def _fixture(dtype=np.complex128):
    times = np.arange(0, 1001, 50, dtype=float)
    peaks = np.stack(
        [
            np.abs(0.8 - 1.6 * np.exp(-times / 240)) + 0.05,
            2 * (np.abs(0.8 - 1.6 * np.exp(-times / 420)) + 0.05),
        ]
    )
    spectra = np.zeros((len(times), 8))
    spectra[:, 1] = peaks[0]
    spectra[:, 5] = peaks[1]
    fids = np.fft.ifft(np.fft.ifftshift(spectra, axes=-1), axis=-1).astype(dtype)
    instance = IVRecovery.__new__(IVRecovery)
    method = {"NoIT": len(times), "InversionTime": times}
    instance.handler = SimpleNamespace(
        rawdatajob0=fids.ravel().copy(),
        method=method,
        method_parameter=method.__getitem__,
    )
    instance.n_acquisition = 8
    preparations = []
    instance.get_ppm = lambda index: preparations.append(index)
    return instance, times, spectra, peaks, preparations


_BOUNDS = ([0.3, -3, 50, 0], [1.2, -0.3, 900, 0.2])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("intervals", [[1, 2, 5, 6], [1, 2, 5, 6, 7], [-7, -6, 5, 6]])
def test_multi_peak_recovery_normalization_plots_and_acquired_input(dtype, intervals):
    instance, times, spectra, peaks, preparations = _fixture(dtype)
    before = instance.handler.rawdatajob0.copy()
    axes = [_Axes() for _ in range(3)]
    coeffs, covariances = instance.InversionRecovery_Calculate_T1_multi_peak(
        intervals, _BOUNDS, axes
    )
    assert preparations == [0]
    assert len(coeffs) == len(covariances) == 2
    np.testing.assert_allclose([c[2] for c in coeffs], [240, 420], rtol=0, atol=0.1)
    np.testing.assert_array_equal(instance.handler.rawdatajob0, before)
    assert len(axes[0].lines) == len(times)
    for i, (_, y, options) in enumerate(axes[0].lines):
        np.testing.assert_allclose(y, spectra[i] / np.max(spectra), atol=2e-6, rtol=0)
        assert options == {"label": f"TI= {times[i]} ms"}
    for i in range(2):
        x, y, _ = axes[i + 1].points[0]
        np.testing.assert_array_equal(x, times)
        np.testing.assert_allclose(y, peaks[i] / np.max(peaks[i]), rtol=0, atol=2e-6)
        curve_x, curve_y, options = axes[i + 1].lines[0]
        assert len(curve_x) == len(curve_y) == 1000
        assert curve_x[0] == 0 and curve_x[-1] == 1000
        c = coeffs[i]
        expected = np.abs(c[0] + c[1] * np.exp(-curve_x / c[2])) + c[3]
        np.testing.assert_allclose(curve_y, expected, rtol=0, atol=1e-12)
        assert covariances[i].shape == (4, 4)
        assert options == {"label": "Fit"}
        assert axes[i + 1].xlabel == "InversionTime [ms]"
        assert axes[i + 1].ylabel == "Integrated spectrum [a.U.]"
        assert axes[i + 1].legends == 1


def test_legacy_method_still_constructs_axes_when_omitted(monkeypatch):
    instance, *_ = _fixture()
    axes = [_Axes() for _ in range(3)]
    calls = []

    def subplots(*args, **options):
        calls.append((args, options))
        return object(), axes

    monkeypatch.setattr(import_module("matplotlib.pyplot"), "subplots", subplots)
    instance.InversionRecovery_Calculate_T1_multi_peak([1, 2, 5, 6], _BOUNDS)
    assert calls == [((1, 3), {"figsize": (8, 5)})]


def test_invalid_fit_bounds_still_raise_after_preparation_without_changing_raw():
    instance, *_, preparations = _fixture()
    before = instance.handler.rawdatajob0.copy()
    axes = [_Axes(), _Axes()]
    with pytest.raises(ValueError, match="lower bound"):
        instance.InversionRecovery_Calculate_T1_multi_peak(
            [1, 2], ([1] * 4, [0] * 4), axes
        )
    assert preparations == [0]
    np.testing.assert_array_equal(instance.handler.rawdatajob0, before)
    assert len(axes[0].lines) == instance.handler.method["NoIT"]
    assert len(axes[1].points) == 1
    assert axes[1].lines == []


def test_empty_interval_still_reports_nonfinite_normalized_fit_input():
    instance, *_ = _fixture()
    with pytest.warns(RuntimeWarning, match="invalid value"):
        with pytest.raises(ValueError, match="infs or NaNs"):
            instance.InversionRecovery_Calculate_T1_multi_peak(
                [2, 2], _BOUNDS, [_Axes(), _Axes()]
            )


def test_interactive_callback_preserves_slider_bounds_axis_clearing_and_output(
    monkeypatch, capsys
):
    instance, *_ = _fixture()
    widgets = import_module("ipywidgets")
    display_module = import_module("IPython.display")
    captured = {}
    cleared = []
    axes = [SimpleNamespace(clear=lambda i=i: cleared.append(i)) for i in range(3)]

    def bind(callback, controls):
        captured["callback"] = callback
        captured["controls"] = controls
        return object()

    def calculate(intervals, bounds, supplied_axes):
        assert cleared == [0, 1, 2]
        assert intervals == [1, 2, 5, 6] and supplied_axes is axes
        np.testing.assert_array_equal(bounds, [[-10, -10, 0, -5], [10, 10, 10000, 5]])
        return [np.array([1, -2, 240, 0.1]), np.array([2, -3, 420, 0.2])], []

    monkeypatch.setattr(widgets, "interactive_output", bind)
    monkeypatch.setattr(display_module, "display", lambda *args: None)
    instance.InversionRecovery_Calculate_T1_multi_peak = calculate
    instance.Interactive_Fitting([1, 2, 5, 6], axes)
    controls = captured["controls"]
    captured["callback"](**{key: widget.value for key, widget in controls.items()})
    output = capsys.readouterr().out
    assert "T1: 240.0 and 420.0" in output
    for widget in controls.values():
        widget.close()
