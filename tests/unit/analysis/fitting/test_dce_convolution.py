"""Tests for the single trapezoidal DCE convolution rule.

The fixtures are small synthetic curves written for these tests: a gamma-variate
plasma input in mM on a uniform grid with 4 s steps and on a non-uniform grid
with 2 s to 8 s steps.  The references are independent evaluations of the same
quadrature (``np.convolve`` with the trapezoid end-point correction on uniform
grids, ``scipy.integrate.cumulative_trapezoid`` per output time on any grid),
so agreement is limited only by float64 summation order and is checked with
``rtol=1e-12``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from hypermri.analysis.dce.fitting_support import (
    _predict_two_c_irf_curve,
    _prepare_two_c_irf_prediction,
    two_c_irf_retention_function,
)
from hypermri.analysis.dce.models.sourbron import (
    sourbron_separable_compartment_model,
)
from hypermri.analysis.dce.models.tofts import (
    extended_tofts_model_2comp,
    tofts_model_1comp,
)
from hypermri.analysis.fitting.models.quantitative.convolution import (
    trapezoidal_convolution,
    trapezoidal_convolution_weights_s,
)

# Summation-order tolerance for two evaluations of the same float64 quadrature.
SAME_RULE_RTOL = 1e-12
SAME_RULE_ATOL_MM = 1e-14

UNIFORM_TIME_S = np.arange(0.0, 120.0, 4.0)
NONUNIFORM_TIME_S = np.concatenate(
    [[0.0, 4.0, 8.0], np.arange(10.0, 40.0, 2.0), np.arange(40.0, 120.0, 8.0)]
)
TIME_GRIDS_S = {"uniform": UNIFORM_TIME_S, "nonuniform": NONUNIFORM_TIME_S}

TWO_C_IRF_PARAMETERS = {
    "gfr_s": 0.010,
    "rpf_s": 0.035,
    "vascular_delay_s": 3.0,
    "vascular_dispersion_rate_s": 0.2,
    "tubular_delay_s": 20.0,
    "tubular_dispersion_rate_s": 0.05,
}


def _aif_mM(time_s: np.ndarray) -> np.ndarray:
    """Gamma-variate plasma input with arrival at 8 s and a 5 mM peak scale."""
    shifted_s = np.clip(time_s - 8.0, 0.0, None)
    return 5.0 * (shifted_s / 6.0) ** 2 * np.exp(-shifted_s / 6.0 + 2.0) / 4.0


def _cumulative_trapezoid_reference(
    values: np.ndarray, time_s: np.ndarray, kernel
) -> np.ndarray:
    """Integrate ``f(u) h(t_n - u)`` from ``t_0`` to ``t_n`` for every ``n``."""
    lag_s = time_s[:, None] - time_s[None, :]
    integrand = values[None, :] * kernel(np.clip(lag_s, 0.0, None))
    cumulative = cumulative_trapezoid(integrand, time_s, axis=1, initial=0.0)
    return np.diag(cumulative).copy()


def _uniform_np_convolve_reference(
    values: np.ndarray, time_s: np.ndarray, kernel
) -> np.ndarray:
    """Trapezoid rule on a uniform grid via ``np.convolve`` and end corrections."""
    step_s = time_s[1] - time_s[0]
    kernel_values = kernel(time_s - time_s[0])
    rectangle = np.convolve(values, kernel_values)[: time_s.size] * step_s
    return rectangle - 0.5 * step_s * (
        values[0] * kernel_values + values * kernel_values[0]
    )


def test_weights_match_the_documented_example() -> None:
    weights_s = trapezoidal_convolution_weights_s(np.array([0.0, 2.0, 6.0]))
    np.testing.assert_allclose(
        weights_s, [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 3.0, 2.0]]
    )


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
def test_weights_of_each_row_sum_to_the_elapsed_time(grid: str) -> None:
    time_s = TIME_GRIDS_S[grid]
    weights_s = trapezoidal_convolution_weights_s(time_s)
    np.testing.assert_allclose(
        weights_s.sum(axis=1), time_s - time_s[0], rtol=SAME_RULE_RTOL
    )
    assert np.all(np.triu(weights_s, k=1) == 0.0)


def test_linear_input_with_constant_kernel_is_exact_on_a_nonuniform_grid() -> None:
    # The trapezoid rule integrates the linear integrand u exactly: t**2 / 2.
    time_s = NONUNIFORM_TIME_S
    result = trapezoidal_convolution(time_s, time_s, np.ones_like)
    np.testing.assert_allclose(result, 0.5 * time_s**2, rtol=SAME_RULE_RTOL)


def test_uniform_grid_matches_np_convolve_with_end_point_correction() -> None:
    time_s = UNIFORM_TIME_S
    aif = _aif_mM(time_s)

    def kernel(lag_s):
        return np.exp(-0.02 * lag_s)

    np.testing.assert_allclose(
        trapezoidal_convolution(aif, time_s, kernel),
        _uniform_np_convolve_reference(aif, time_s, kernel),
        rtol=SAME_RULE_RTOL,
        atol=SAME_RULE_ATOL_MM,
    )


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
def test_any_grid_matches_cumulative_trapezoid_per_output_time(grid: str) -> None:
    time_s = TIME_GRIDS_S[grid]
    aif = _aif_mM(time_s)

    def kernel(lag_s):
        return np.exp(-lag_s / 30.0)

    np.testing.assert_allclose(
        trapezoidal_convolution(aif, time_s, kernel),
        _cumulative_trapezoid_reference(aif, time_s, kernel),
        rtol=SAME_RULE_RTOL,
        atol=SAME_RULE_ATOL_MM,
    )


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
def test_tofts_models_use_the_trapezoid_rule(grid: str) -> None:
    # Regression: the rectangle rule np.convolve(...) * (t[1] - t[0])
    # differed from this reference by dt / 2 * (f_0 h_n + f_n h_0) on uniform
    # grids and used the wrong step on non-uniform grids.
    time_s = TIME_GRIDS_S[grid]
    ktrans_per_s, kep_per_s, vp = 0.005, 0.02, 0.05
    expected_mM = ktrans_per_s * _cumulative_trapezoid_reference(
        _aif_mM(time_s), time_s, lambda lag_s: np.exp(-kep_per_s * lag_s)
    )
    np.testing.assert_allclose(
        tofts_model_1comp(time_s, ktrans_per_s, kep_per_s, _aif_mM),
        expected_mM,
        rtol=SAME_RULE_RTOL,
        atol=SAME_RULE_ATOL_MM,
    )
    np.testing.assert_allclose(
        extended_tofts_model_2comp(time_s, ktrans_per_s, kep_per_s, vp, _aif_mM),
        expected_mM + vp * _aif_mM(time_s),
        rtol=SAME_RULE_RTOL,
        atol=SAME_RULE_ATOL_MM,
    )


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
def test_reduced_sourbron_model_uses_the_trapezoid_rule(grid: str) -> None:
    # Regression: on uniform grids the numpy and xarray branches used
    # the rectangle rule with the mean step.
    time_s = TIME_GRIDS_S[grid]
    aif = _aif_mM(time_s)
    vp, ft_per_s, tt_s = 0.25, 0.03, 60.0
    expected_mM = vp * aif + ft_per_s * _cumulative_trapezoid_reference(
        aif, time_s, lambda lag_s: np.exp(-lag_s / tt_s)
    )
    result = sourbron_separable_compartment_model(
        time_s,
        aif,
        plasma_volume_fraction=vp,
        tubular_flow_s=ft_per_s,
        tubular_transit_time_s=tt_s,
    )
    np.testing.assert_allclose(
        result, expected_mM, rtol=SAME_RULE_RTOL, atol=SAME_RULE_ATOL_MM
    )


@pytest.mark.parametrize("plasma_transit_time_s", [None, 5.0])
@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
def test_sourbron_xarray_branch_matches_numpy_branch(
    grid: str, plasma_transit_time_s: float | None
) -> None:
    time_s = TIME_GRIDS_S[grid]
    aif = _aif_mM(time_s)
    vp = np.array([0.1, 0.25])
    ft_per_s = np.array([0.03, 0.01])
    tt_s = np.array([60.0, 20.0])

    def as_voxels(values):
        return xr.DataArray(values, dims=["voxel"])

    batched = sourbron_separable_compartment_model(
        time_s,
        aif,
        plasma_volume_fraction=as_voxels(vp),
        tubular_flow_s=as_voxels(ft_per_s),
        tubular_transit_time_s=as_voxels(tt_s),
        plasma_transit_time_s=(
            None
            if plasma_transit_time_s is None
            else as_voxels(np.full(2, plasma_transit_time_s))
        ),
    )
    for voxel in range(2):
        single = sourbron_separable_compartment_model(
            time_s,
            aif,
            plasma_volume_fraction=float(vp[voxel]),
            tubular_flow_s=float(ft_per_s[voxel]),
            tubular_transit_time_s=float(tt_s[voxel]),
            plasma_transit_time_s=plasma_transit_time_s,
        )
        np.testing.assert_allclose(
            batched.isel(voxel=voxel).transpose("repetition_idx").values,
            single,
            rtol=SAME_RULE_RTOL,
            atol=SAME_RULE_ATOL_MM,
        )


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
def test_two_c_irf_prediction_is_unchanged_by_the_shared_rule(grid: str) -> None:
    # 2C-IRF used the trapezoid rule previously: np.convolve with end-point
    # correction on uniform grids and cumulative_trapezoid otherwise.
    time_s = TIME_GRIDS_S[grid]
    aif = _aif_mM(time_s)

    def kernel(lag_s):
        return two_c_irf_retention_function(lag_s, **TWO_C_IRF_PARAMETERS)

    if grid == "uniform":
        previous_mM = _uniform_np_convolve_reference(aif, time_s, kernel)
        previous_mM[0] = 0.0
    else:
        previous_mM = _cumulative_trapezoid_reference(aif, time_s, kernel)
    prepared = _prepare_two_c_irf_prediction(time_s, aif)
    np.testing.assert_allclose(
        _predict_two_c_irf_curve(prepared, **TWO_C_IRF_PARAMETERS),
        previous_mM,
        rtol=SAME_RULE_RTOL,
        atol=SAME_RULE_ATOL_MM,
    )


@pytest.mark.parametrize(
    ("time_s", "message"),
    [
        (np.zeros((2, 2)), "1D time axis"),
        (np.array([0.0, np.nan, 2.0]), "finite sample times"),
        (np.array([0.0, 2.0, 2.0]), "strictly increasing"),
        (np.array([0.0, 4.0, 2.0]), "strictly increasing"),
    ],
)
def test_invalid_time_axis_is_rejected(time_s: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        trapezoidal_convolution_weights_s(time_s)


def test_mismatched_input_and_kernel_shapes_are_rejected() -> None:
    time_s = np.array([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="input values of shape"):
        trapezoidal_convolution(np.ones(2), time_s, np.ones_like)
    with pytest.raises(ValueError, match="kernel must return shape"):
        trapezoidal_convolution(np.ones(3), time_s, lambda lag_s: lag_s[0])
