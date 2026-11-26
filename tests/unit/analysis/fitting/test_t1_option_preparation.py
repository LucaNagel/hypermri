"""Protect supplied T1 settings using invented, perturbed IR curves.

The reference is the existing solver reached through explicit prepared settings.
rtol=1e-12 and atol=1e-14 protect routing equivalence in each output's units;
these are not claims about physiological recovery accuracy. The 10 ms grid
separates known requested-option effects on the invented curve.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config
from test_quantitative_domain import (
    _assert_complete_equal,
    _assert_metadata_equal,
)

from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.fitting.models.quantitative import t1
from hypermri.analysis.quantitative import T1FitOptions, fit_t1

_CASES = [
    ("weight_by_signal", True, {}),
    ("bounds_A", (0.0, 1.0), {}),
    ("bounds_B", (-0.02, 0.02), {"fit_baseline": True}),
    ("regularization_alpha", 0.5, {}),
    ("auto_calc_rel_bound", True, {}),
    ("rel_bounds_A", (0.9, 1.1), {}),
    ("rel_bounds_B", (-0.1, 0.1), {"fit_baseline": True}),
    ("inversion_imperfection", 0.3, {"auto_calc_rel_bound": True}),
]


def _invented_ir(dtype=np.float64, transpose=False):
    times = np.array([50.0, 150.0, 400.0, 900.0, 1500.0])
    curve = 2.0 * (1.0 - 2.0 * np.exp(-times / 700.0))
    curve += np.array([0.15, -0.35, 0.05, 0.2, -0.1])
    if np.issubdtype(dtype, np.complexfloating):
        curve = curve + 0.4j * curve
    data = xr.DataArray(
        np.stack([curve, 0.8 * curve]).astype(dtype)[:, None, :],
        dims=("read_idx", "phase_idx", "inversion_idx"),
        coords={"read_idx": [7, 3], "phase_idx": [2], "inversion_idx": times},
        attrs={"unit": "a.u.", "invented_fixture": True},
    )
    if transpose:
        data = data.transpose("inversion_idx", "phase_idx", "read_idx")
    return data


def _base_options():
    return {
        "t1_values_s": np.linspace(0.2, 2.0, 181),
        "use_multiprocessing": False,
        "number_of_cpu_cores": -1,
    }


@pytest.mark.parametrize("name,value,extra", _CASES)
def test_preparation_preserves_each_supported_t1_option(name, value, extra):
    data = _invented_ir()
    settings = {**_base_options(), **extra, name: value}
    before = deepcopy(settings)
    prepared = api.prepare_fit_params(data, "t1", settings)
    assert name in prepared.attrs
    _assert_metadata_equal(prepared.attrs[name], value)
    _assert_metadata_equal(settings, before)
    assert name in t1.get_t1_params_info()["optional_fit_params"]


@pytest.mark.parametrize("name,value,extra", _CASES)
def test_mapping_and_immutable_options_match_existing_prepared_solver(
    name, value, extra
):
    data = _invented_ir()
    before = data.copy(deep=True)
    base = _base_options()
    prepared = api.prepare_fit_params(data, "t1", base)
    prepared.attrs.update(extra)
    prepared.attrs[name] = value
    prepared_before = prepared.copy(deep=True)
    expected = fit_t1(data, prepared)
    requested = {**base, **extra, name: value}
    actual = api.fit(data, "t1", requested)
    typed = T1FitOptions(
        t1_values_s=base["t1_values_s"],
        execution=ExecutionConfig(use_multiprocessing=False),
        **extra,
        **{name.lower(): value},
    )
    _assert_complete_equal(actual, expected)
    _assert_complete_equal(fit_t1(data, typed), expected)
    _assert_complete_equal(prepared, prepared_before)
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize(
    "settings,expected_ms",
    [
        ({"weight_by_signal": True}, 720.0),
        ({"bounds_A": (0.0, 1.0)}, 610.0),
        ({"fit_baseline": True, "bounds_B": (-0.02, 0.02)}, 680.0),
        ({"regularization_alpha": 0.5}, 690.0),
    ],
)
def test_requested_settings_change_the_characterized_grid_solution(
    settings, expected_ms
):
    result = fit_t1(_invented_ir(), {**_base_options(), **settings})
    np.testing.assert_allclose(
        result.t1.isel(read_idx=0), expected_ms, rtol=0, atol=1e-8
    )
    assert result.t1.attrs["unit"] == "ms"
    if "bounds_A" in settings:
        assert result.A.isel(read_idx=0).item() <= 1.0 + 1e-12
    if "bounds_B" in settings:
        assert abs(result.B.isel(read_idx=0).item()) <= 0.02 + 1e-12


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("transpose", [False, True])
def test_weighted_regularized_fits_preserve_precision_labels_and_inputs(
    dtype, transpose
):
    data = _invented_ir(dtype, transpose)
    before = data.copy(deep=True)
    settings = {
        **_base_options(),
        "weight_by_signal": True,
        "regularization_alpha": 0.5,
    }
    expected = fit_t1(data, settings)
    with parallel_config(backend="threading"):
        actual = fit_t1(
            data, {**settings, "use_multiprocessing": True, "number_of_cpu_cores": 2}
        )
    for result in (actual, expected):
        result.attrs["fit_prepared_options"].pop("use_multiprocessing")
        result.attrs["fit_prepared_options"].pop("number_of_cpu_cores", None)
    _assert_complete_equal(actual, expected)
    assert actual.t1.dims == tuple(d for d in data.dims if d != "inversion_idx")
    assert actual.t1.dtype == np.float64
    assert actual.A.dtype == (
        dtype if np.issubdtype(dtype, np.complexfloating) else np.float64
    )
    reconstructed = api.reconstruct_fit(actual)
    reference = (
        actual.A * (1.0 - 2.0 * np.exp(-data.inversion_idx / actual.t1)) + actual.B
    )
    xr.testing.assert_allclose(reconstructed, reference.transpose(*reconstructed.dims))
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("relative", [0.1, (0.9, 1.1)])
def test_t1_options_freeze_bounds_without_changing_scalar_relative_settings(relative):
    bounds = [0.0, 1.0]
    options = T1FitOptions(bounds_a=bounds, rel_bounds_a=relative)
    bounds[1] = 99.0
    assert options.bounds_a == (0.0, 1.0)
    assert options.rel_bounds_a == relative
    with pytest.raises(FrozenInstanceError):
        options.weight_by_signal = True


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_weighted_fit_retains_nan_failure_contract(dtype):
    data = _invented_ir(dtype)
    result = fit_t1(
        data,
        {
            "t1_values_s": [-1.0, 0.0],
            "weight_by_signal": True,
            "use_multiprocessing": False,
        },
    )
    for name in ("t1", "A", "B", "rss_t1"):
        assert np.isnan(result[name]).all()


def test_omitted_scientific_settings_keep_existing_defaults():
    base = _base_options()
    data = _invented_ir()
    prepared = api.prepare_fit_params(data, "t1", base)
    assert not ({case[0] for case in _CASES} & prepared.attrs.keys())
    expected = fit_t1(data, prepared)
    actual = fit_t1(data, base)
    _assert_complete_equal(actual, expected)
    np.testing.assert_allclose(actual.t1.isel(read_idx=0), 700.0, rtol=0, atol=1e-8)


def test_preparation_copies_mutable_supplied_bounds():
    bounds = [0.0, 1.0]
    prepared = api.prepare_fit_params(_invented_ir(), "t1", {"bounds_A": bounds})
    bounds[1] = 99.0
    assert prepared.attrs["bounds_A"] == [0.0, 1.0]


def test_weighted_rss_matches_reconstructed_residuals():
    data = _invented_ir()
    result = fit_t1(data, {**_base_options(), "weight_by_signal": True})
    predicted = api.reconstruct_fit(result)
    weights = abs(data) / (abs(data).max("inversion_idx") + 1e-8)
    reference_rss = (weights * abs(data - predicted) ** 2).sum("inversion_idx")
    xr.testing.assert_allclose(result.rss_t1, reference_rss, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "selected,overridden",
    [
        ({"bounds_A": (0.0, 1.0)}, {"regularization_alpha": 0.5}),
        ({"rel_bounds_A": (0.9, 1.1)}, {"bounds_A": (0.0, 1.0)}),
        ({"auto_calc_rel_bound": True}, {"rel_bounds_A": (0.2, 0.3)}),
    ],
)
def test_existing_bound_and_regularization_precedence_is_preserved(
    selected, overridden
):
    data = _invented_ir()
    expected = fit_t1(data, {**_base_options(), **selected})
    actual = fit_t1(data, {**_base_options(), **selected, **overridden})
    xr.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    for key, value in overridden.items():
        _assert_metadata_equal(actual.attrs["fit_prepared_options"][key], value)
