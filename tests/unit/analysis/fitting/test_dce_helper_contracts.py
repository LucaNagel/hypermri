"""Characterize the distinct DCE window and residual contracts previously.

All inputs are invented concentrations in mM and irregular times in seconds.
Masks are exact discrete choices. Scalar metrics use relative tolerance 1e-12,
well above float64 rounding for six samples and below any selection error.
Patlak RSS is deliberately measured in the AIF-normalized graphical space.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from hypermri.analysis.dce.fitting_support import (
    compute_fit_quality_metrics,
    select_fit_window,
)
from hypermri.analysis.dce.models.patlak import (
    fit_single_voxel_patlak_helper,
)
from hypermri.analysis.dce.models.sourbron import (
    _build_sourbron_fit_mask,
)
from hypermri.analysis.dce.models.tofts import build_tofts_fit_mask

TIMES_S = np.array([0.0, 2.0, 5.0, 9.0, 14.0, 20.0])


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, [True] * 6),
        (
            {"fit_start_index": 2, "fit_stop_index": 5},
            [False, False, True, True, True, False],
        ),
        (
            {"fit_start_time_s": 5, "fit_end_time_s": 14},
            [False, False, True, True, True, False],
        ),
        ({"fit_indices": [-1, 1, 3]}, [False, True, False, True, False, True]),
        ({"fit_times_s": [3.5, 12]}, [False, True, False, False, True, False]),
        ({"fit_end_index": -2}, [True, True, True, True, False, False]),
        ({"fit_end_index": -9}, [False] * 6),
    ],
)
def test_tofts_window_contract(options, expected):
    original = TIMES_S.copy()
    actual = build_tofts_fit_mask(TIMES_S, options)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(TIMES_S, original)


@pytest.mark.parametrize("fit_type", ["tofts", "patlak", "sourbron"])
def test_public_window_uses_absolute_end_and_nearest_time_selection(fit_type):
    mask = select_fit_window(TIMES_S, {"fit_end_index": -2}, fit_type=fit_type)
    assert not mask.any()
    mask = select_fit_window(TIMES_S, {"fit_times_s": [3.5, 12]}, fit_type=fit_type)
    np.testing.assert_array_equal(mask, [False, True, False, False, True, False])
    assert mask.dims == ("repetition_idx",)
    assert mask.name == "fit_window_mask"
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask.repetition_idx, np.arange(6))


def test_sourbron_window_keeps_its_threshold_and_ignores_explicit_times():
    aif = np.array([0, 1e-7, 0.5, 0.2, 0.1, 0.05])
    mask = _build_sourbron_fit_mask(
        TIMES_S,
        fit_params={"fit_times_s": [20], "fit_indices": [0, 1, 2, 4]},
        aif_concentration=aif,
    )
    np.testing.assert_array_equal(mask, [False, True, True, False, True, False])
    mask = _build_sourbron_fit_mask(TIMES_S, fit_params={"fit_end_index": -2})
    assert not mask.any()


def test_patlak_keeps_graphical_rss_and_its_model_specific_selection():
    aif = np.array([0.1, 0.3, 0.8, 0.4, 0.2, 0.1])
    integral = cumulative_trapezoid(aif, TIMES_S, initial=0)
    signal = (
        0.02 * integral
        + 0.1 * aif
        + np.array([0.001, -0.001, 0.002, -0.002, 0.001, -0.001])
    )
    result = fit_single_voxel_patlak_helper(
        signal, aif, TIMES_S, {"fit_times_s": [20], "fit_indices": [0]}
    )
    reference = fit_single_voxel_patlak_helper(signal, aif, TIMES_S)
    assert result == reference
    graphical_prediction = result["K_trans"] * integral / aif + result["Vp"]
    graphical_rss = np.sum((signal / aif - graphical_prediction) ** 2)
    np.testing.assert_allclose(result["rss_patlak"], graphical_rss, rtol=1e-12)
    concentration_rss = np.sum((signal - graphical_prediction * aif) ** 2)
    assert not np.isclose(result["rss_patlak"], concentration_rss, rtol=1e-3, atol=0)
    invalid = fit_single_voxel_patlak_helper(
        signal, aif, TIMES_S, {"fit_end_index": -2}
    )
    assert all(np.isnan(value) for value in invalid.values())


@pytest.mark.parametrize("constant", [False, True])
def test_quality_metrics_keep_concentration_space_and_flat_curve_rules(constant):
    observed = np.ones(6) if constant else np.arange(1.0, 7.0)
    concentration = xr.DataArray(
        observed[None, :],
        dims=("read_idx", "repetition_idx"),
        coords={"read_idx": [4.0], "repetition_idx": [8, 9, 10, 11, 12, 13]},
        attrs={"unit": "mM"},
    )
    parameters = xr.Dataset(
        {"K_trans": ("read_idx", [0.0]), "Vp": ("read_idx", [0.0])},
        coords={"read_idx": [4.0]},
    )
    metrics = compute_fit_quality_metrics(
        parameters,
        concentration,
        fit_type="patlak",
        aif_concentration=np.ones(6),
        aif_time_s=TIMES_S,
    )
    rss = 6.0 if constant else 91.0
    assert metrics.rss.item() == rss
    assert metrics.rmse.item() == pytest.approx(np.sqrt(rss / 6), rel=1e-12)
    if constant:
        assert np.isnan(metrics.r_squared.item())
    else:
        assert metrics.r_squared.item() == pytest.approx(-4.2, rel=1e-12)
    assert metrics.aic.item() == pytest.approx(6 * np.log(rss / 6) + 4, rel=1e-12)
    assert metrics.bic.item() == pytest.approx(
        6 * np.log(rss / 6) + 2 * np.log(6), rel=1e-12
    )
    np.testing.assert_array_equal(
        metrics.fit_window_mask.repetition_idx, concentration.repetition_idx
    )
