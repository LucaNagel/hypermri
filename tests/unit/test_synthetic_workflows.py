"""Scientific workflow regressions independent of tutorial documents."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hypermri.visualization import PlotOptions, plot

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_synthetic_metabolite_detection_and_reconstruction():
    """The fixed null/signal simulations retain M0/M1 decisions and labels."""
    scope = runpy.run_path(str(_FIXTURES / "synthetic_metabolite_detection.py"))
    assert scope["result"].detection_status.values.tolist() == [
        "not_detected",
        "detected",
    ]
    assert scope["summary"].target_accepted.values.tolist() == [False, True]
    assert scope["fitted"].dims == scope["data"].dims


def test_synthetic_noise_reference_and_bootstrap():
    """Retain 5% quadrature-sigma recovery and all 99 successful bootstrap fits."""
    scope = runpy.run_path(str(_FIXTURES / "synthetic_noise_quantification.py"))
    np.testing.assert_allclose(scope["sigma_hat"], scope["SIGMA_TRUE"], rtol=0.05)
    assert bool((scope["bootstrap_fit"].fit_status == "success").all())
    assert np.isfinite(scope["intervals"].values).all()
