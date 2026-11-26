"""Check which spectroscopy analyses accept or reject Voigt fits.

Analyses built on a Lorentz basis must reject ``lineshape='voigt'`` inputs
before any computation; bootstrap uncertainty and relevance accept Voigt fits
whose parameters and results share the lineshape. The fixture is a synthetic
two-line FID (0 and 200 Hz, areas 20 and 5, Gaussian FWHM 10 Hz, 128 samples
at 2 kHz, quadrature noise sigma 0.02) fitted with three short passes.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy as fitting
from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv
from hypermri.analysis.spectroscopy import fitting as spectrum_fitting
from hypermri.analysis.spectroscopy import pipeline
from hypermri.analysis.spectroscopy.covariance import estimate_fit_covariance
from hypermri.analysis.spectroscopy.fit_diagnostics import (
    calculate_spectroscopy_fit_diagnostics,
)
from hypermri.analysis.spectroscopy.fitting import fit_spectrum
from hypermri.analysis.spectroscopy.local_uncertainty import (
    estimate_fit_local_uncertainty,
)
from hypermri.analysis.spectroscopy.quality import evaluate_spectroscopy_fit_quality
from hypermri.analysis.spectroscopy.relevance import compare_metabolite_relevance
from hypermri.analysis.spectroscopy.uncertainty import estimate_fit_uncertainty

LORENTZ_ONLY = "supports only lineshape='lorentz'"
NOISE_SIGMA = 0.02


def options(**extra):
    """Return plain fit options for the two synthetic lines."""
    return {
        "metabs": ["a", "b"],
        "metabs_freqs_hz": [1.0, 201.0],
        "initial_t2s_s": 0.03,
        "metabs_freq_ranges": 6.0,
        "rep_fitting": 3,
        "npoints": 11,
        **extra,
    }


@pytest.fixture(scope="module")
def data():
    """Return two noisy Voigt FIDs of the synthetic lines."""
    rng = np.random.default_rng(3)
    t = np.arange(128) / 2000.0
    lines = np.array([20.0, 5.0])[:, None] * np.exp(
        2j * np.pi * np.array([0.0, 200.0])[:, None] * t - t / 0.05
    )
    signal = lines.sum(0) * np.exp(-((np.pi * 10.0 * t) ** 2) / (4 * np.log(2)))
    noise = NOISE_SIGMA * (
        rng.standard_normal((2, t.size)) + 1j * rng.standard_normal((2, t.size))
    )
    values = xr.DataArray(
        signal + noise, dims=["x", "acquisition_idx"], coords={"x": [0, 1]}
    )
    values.attrs["sampling_time_axis_s"] = t
    return values


@pytest.fixture(scope="module")
def voigt(data):
    """Return Voigt parameters and results for the synthetic data."""
    params = def_fit_params(data=data, fit_params=options(lineshape="voigt"))
    return params, fit_data_pseudo_inv(data, params)


@pytest.fixture(scope="module")
def lorentz_params(data):
    """Return Lorentz parameters for the synthetic data."""
    return def_fit_params(data=data, fit_params=options())


@pytest.fixture
def no_fitting(monkeypatch):
    """Fail if a rejected analysis reaches the spectroscopy fit."""

    def unexpected_fit(*args, **kwargs):
        pytest.fail("A rejected Voigt analysis must not fit spectra.")

    for module in (fitting, spectrum_fitting, pipeline):
        monkeypatch.setattr(module, "fit_data_pseudo_inv", unexpected_fit)


@pytest.mark.parametrize(
    "analysis",
    [
        lambda d, p, r: estimate_fit_covariance(r, p, noise=NOISE_SIGMA),
        lambda d, p, r: estimate_fit_local_uncertainty(r, p, noise=NOISE_SIGMA),
        lambda d, p, r: calculate_spectroscopy_fit_diagnostics(r, p, NOISE_SIGMA),
        lambda d, p, r: fit_spectrum(d, p, noise=NOISE_SIGMA),
        lambda d, p, r: evaluate_spectroscopy_fit_quality(d, p),
        lambda d, p, r: evaluate_spectroscopy_fit_quality(d, p, fit_results=r),
    ],
    ids=[
        "covariance",
        "local-uncertainty",
        "diagnostics",
        "fit-spectrum-noise",
        "quality",
        "quality-with-results",
    ],
)
def test_lorentz_only_analyses_reject_voigt_before_fitting(
    analysis, data, voigt, no_fitting
):
    """Lorentz-basis analyses reject Voigt fits instead of misreporting them."""
    params, results = voigt
    with pytest.raises(ValueError, match=LORENTZ_ONLY):
        analysis(data, params, results)


def test_lorentz_only_analyses_reject_voigt_results_alone(data, lorentz_params, voigt):
    """A Voigt result is rejected even with Lorentz parameters."""
    _, results = voigt
    with pytest.raises(ValueError, match=LORENTZ_ONLY):
        estimate_fit_covariance(results, lorentz_params, noise=NOISE_SIGMA)


def test_pipeline_quality_rejects_voigt_before_fitting(data, no_fitting):
    """run_quality=True fails before the pipeline fits a Voigt model."""
    with pytest.raises(ValueError, match=r"run_quality=True\) supports only"):
        pipeline.run_csi_fitting_pipeline(
            data, options(lineshape="voigt"), run_quality=True, verbose=False
        )


def test_bootstrap_requires_matching_lineshape(data, lorentz_params, voigt):
    """Bootstrap draws cannot mix Lorentz parameters with Voigt results."""
    _, results = voigt
    with pytest.raises(ValueError, match="same lineshape"):
        estimate_fit_uncertainty(data, lorentz_params, results, noise=NOISE_SIGMA)


def test_relevance_requires_matching_lineshape(data, lorentz_params, voigt):
    """Relevance cannot compare Lorentz M0 fits with a Voigt M1 fit."""
    _, results = voigt
    with pytest.raises(ValueError, match="same lineshape"):
        compare_metabolite_relevance(data, lorentz_params, results, targets="b")


def test_relevance_accepts_voigt_fits(data, voigt):
    """Removing the strong line of a Voigt fit raises the Voigt M0 residual."""
    params, results = voigt
    relevance = compare_metabolite_relevance(data, params, results, targets="a")
    assert np.all(relevance.rss_m0.values > 10 * relevance.rss_m1.values)
    assert np.all(relevance.m0_fit_status.values == "success")
