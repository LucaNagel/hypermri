"""Characterize Rayleigh fitting on invented deterministic magnitude data."""

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import rayleigh


def _module():
    return importlib.import_module("hypermri.analysis.fitting.distributions")


def _signal():
    # Equal-probability quantiles avoid RNG uncertainty in the recovery control.
    probability = (np.arange(20000) + 0.5) / 20000
    magnitude = rayleigh.ppf(probability, loc=0.35, scale=1.75)
    return (magnitude * np.exp(1j * np.linspace(-np.pi, np.pi, 20000))).reshape(
        100, 200
    )


def test_rayleigh_pdf_matches_independent_closed_form():
    x = np.array([0.35, 0.8, 1.2, 2.5, 4.0])
    shifted = x - 0.35
    expected = shifted / 1.75**2 * np.exp(-(shifted**2) / (2 * 1.75**2))
    np.testing.assert_allclose(
        _module().rayleigh_pdf_fit(x, 0.35, 1.75), expected, rtol=1e-14
    )


def test_rayleigh_fit_recovers_invented_parameters_and_preserves_input():
    signal = _signal()
    snapshot = signal.copy()
    figures = plt.get_fignums()
    options = {"bins": 40, "p0": [0.3, 1.7]}
    params, covariance = _module().fit_rayleigh(signal, plot=False, **options)
    again, again_covariance = _module().fit_rayleigh(signal, plot=False, **options)
    # Histogram fitting approximates a continuous density, hence finite error.
    np.testing.assert_allclose(params, [0.35, 1.75], atol=0.01, rtol=0)
    assert covariance.shape == (2, 2)
    assert np.isfinite(covariance).all()
    assert (np.diag(covariance) >= 0).all()
    np.testing.assert_allclose(again, params, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(again_covariance, covariance, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(signal, snapshot)
    assert plt.get_fignums() == figures


def test_rayleigh_plot_preserves_fit_and_creates_two_diagnostic_axes():
    signal = _signal()
    module = _module()
    expected, _ = module.fit_rayleigh(signal, plot=False, bins=40, p0=[0.3, 1.7])
    figures = set(plt.get_fignums())
    try:
        params, _ = module.fit_rayleigh(signal, plot=True, bins=40, p0=[0.3, 1.7])
        np.testing.assert_allclose(params, expected, rtol=1e-12, atol=1e-12)
        created = set(plt.get_fignums()) - figures
        assert len(created) == 1
        axes = plt.figure(created.pop()).axes
        assert [axis.get_title() for axis in axes] == [
            "Signal Magnitude",
            "Histogram Magnitude",
        ]
        assert len(axes[0].lines[0].get_ydata()) == signal.size
    finally:
        for figure in set(plt.get_fignums()) - figures:
            plt.close(figure)


def test_rayleigh_missing_data_retains_actionable_failure():
    with pytest.raises(ValueError, match="data has to be passed"):
        _module().fit_rayleigh(plot=False)
