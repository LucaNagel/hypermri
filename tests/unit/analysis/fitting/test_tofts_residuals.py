"""Residual sum of squares stored by the Tofts fit.

The fixture is synthetic: a piecewise-linear AIF in mM on an 8 s grid with
1 s sampling and a Tofts tissue curve with known parameters, perturbed by
Gaussian noise with a fixed seed so that the residuals are not zero.
"""

import numpy as np
import pytest
from scipy.interpolate import interp1d

from hypermri.analysis.dce.models.tofts import (
    extended_tofts_model_2comp,
    fit_single_voxel_tofts_helper,
    tofts_model_1comp,
)

TIMES_S = np.arange(8, dtype=np.float64)
AIF_MM = np.array([0.0, 0.0, 0.4, 0.9, 0.6, 0.3, 0.15, 0.05], dtype=np.float64)
NOISE_SD_MM = 0.01

# The stored and the recomputed RSS evaluate the same model at the same
# parameters, so they differ only by floating-point rounding (about 1e-16
# relative).  1e-9 is far below the factor of two that the test guards against.
RSS_RTOL = 1e-9


def _cp_func():
    return interp1d(TIMES_S, AIF_MM, kind="linear", fill_value="extrapolate")


def _noisy_curve(clean_curve):
    rng = np.random.default_rng(20261002)
    return clean_curve + rng.normal(0.0, NOISE_SD_MM, size=clean_curve.shape)


def test_tofts_1comp_stores_residual_sum_of_squares():
    cp_func = _cp_func()
    measured = _noisy_curve(tofts_model_1comp(TIMES_S, 0.18, 0.55, cp_func))

    params, rss = fit_single_voxel_tofts_helper(TIMES_S, measured, cp_func, "1comp")

    residual = tofts_model_1comp(TIMES_S, params[0], params[1], cp_func) - measured
    assert rss > 0.0
    assert rss == pytest.approx(np.sum(residual**2), rel=RSS_RTOL)


def test_tofts_2comp_stores_residual_sum_of_squares():
    cp_func = _cp_func()
    measured = _noisy_curve(
        extended_tofts_model_2comp(TIMES_S, 0.18, 0.55, 0.05, cp_func)
    )

    params, rss = fit_single_voxel_tofts_helper(TIMES_S, measured, cp_func, "2comp")

    residual = (
        extended_tofts_model_2comp(TIMES_S, params[0], params[1], params[2], cp_func)
        - measured
    )
    assert rss > 0.0
    assert rss == pytest.approx(np.sum(residual**2), rel=RSS_RTOL)


def test_tofts_rss_uses_only_fit_window_samples():
    cp_func = _cp_func()
    measured = _noisy_curve(tofts_model_1comp(TIMES_S, 0.18, 0.55, cp_func))
    fit_mask = TIMES_S < 6.0

    params, rss = fit_single_voxel_tofts_helper(
        TIMES_S, measured, cp_func, "1comp", fit_mask=fit_mask
    )

    residual = tofts_model_1comp(TIMES_S, params[0], params[1], cp_func) - measured
    assert rss == pytest.approx(np.sum(residual[fit_mask] ** 2), rel=RSS_RTOL)


@pytest.mark.parametrize("model_type", ["1comp", "2comp"])
def test_tofts_nonpositive_curve_stores_rss_of_zero_model(model_type):
    """A curve without uptake is fitted with zero parameters, not zero RSS."""
    measured = np.array([0.0, -0.02, -0.01, 0.0, -0.03, -0.01, 0.0, -0.02])

    params, rss = fit_single_voxel_tofts_helper(
        TIMES_S, measured, _cp_func(), model_type
    )

    assert np.all(np.asarray(params) == 0.0)
    assert rss == pytest.approx(np.sum(measured**2), rel=RSS_RTOL)
