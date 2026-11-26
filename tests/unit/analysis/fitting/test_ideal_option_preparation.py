"""IDEAL option routing on invented complex echoes, without acquired fixtures.

The existing explicitly prepared solver is the routing reference, compared at
rtol=1e-12 and atol=1e-14 in Hz/signal units. These tolerances protect the same
execution, not accuracy of a nonlinear fit from arbitrary initial guesses.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config
from test_quantitative_domain import _assert_complete_equal, _assert_metadata_equal

from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.fitting.ideal import IDEAL_Solver
from hypermri.analysis.fitting.models.quantitative import ideal
from hypermri.analysis.quantitative import IDEALFitOptions, fit_ideal

_CASES = [
    ("solve_for", ["Field_offset"], {}),
    ("solve_for", ["Theta"], {}),
    ("init_f_offset_Hz", 100.0, {"max_iterations": 1}),
    ("stop_f_offset_hz", 1e6, {"solve_for": ["Field_offset"]}),
    ("max_iterations", 1, {}),
    ("init_Theta", 0.1 + 0.04j, {"max_iterations": 1}),
    ("stop_theta", 1e3 + 1e3j, {"stop_f_offset_hz": 1e6}),
]
_TYPED_NAMES = {"init_f_offset_Hz": "init_f_offset_hz", "init_Theta": "init_theta"}


def _invented_echoes(dtype=np.complex128, transpose=False):
    indices = np.arange(9)
    times_s = (indices - (indices.size - 1) / 2) * 0.001
    curve = (1.5 + 0.2j) + (0.4 - 0.1j) * np.exp(2j * np.pi * 120.0 * times_s)
    curve *= np.exp(
        2j * np.pi * 60.0 * times_s + (0.15 + 0.08j) * (-1.0) ** (indices + 1)
    )
    curve += 0.006 * np.array(
        [
            1 + 0.2j,
            -0.2 + 0.5j,
            0.4 - 0.8j,
            -0.6 + 0.3j,
            0.2 - 0.3j,
            0.5 + 0.6j,
            -0.8 - 0.4j,
            0.1 + 0.9j,
            0.7 - 0.1j,
        ]
    )
    data = xr.DataArray(
        np.stack([curve, 0.8 * curve]).astype(dtype)[:, None, :],
        dims=("read_idx", "phase_idx", "echo_idx"),
        coords={"read_idx": [7, 3], "phase_idx": [2], "echo_idx": indices},
        attrs={"unit": "a.u.", "invented_fixture": True},
    )
    return data.transpose("echo_idx", "phase_idx", "read_idx") if transpose else data


def _base_options():
    return {
        "relative_frequencies_hz": [0.0, 120.0],
        "echo_spacing_s": 0.001,
        "metabolite_names": ["invented_a", "invented_b"],
        "use_multiprocessing": False,
        "number_of_cpu_cores": -1,
    }


@pytest.mark.parametrize("name,value,extra", _CASES)
def test_ideal_preparation_preserves_each_active_setting(name, value, extra):
    settings = {**_base_options(), **extra, name: value}
    before = deepcopy(settings)
    prepared = api.prepare_fit_params(_invented_echoes(), "ideal", settings)
    assert name in prepared.attrs
    _assert_metadata_equal(prepared.attrs[name], value)
    _assert_metadata_equal(settings, before)
    assert name in ideal.get_ideal_params_info()["optional_fit_params"]


@pytest.mark.parametrize("name,value,extra", _CASES)
def test_mapping_and_typed_ideal_options_match_existing_prepared_solver(
    name, value, extra
):
    data = _invented_echoes()
    data_before = data.copy(deep=True)
    base = _base_options()
    prepared = api.prepare_fit_params(data, "ideal", base)
    prepared.attrs.update(extra)
    prepared.attrs[name] = value
    prepared_before = prepared.copy(deep=True)
    expected = fit_ideal(data, prepared)
    actual = api.fit(data, "ideal", {**base, **extra, name: value})
    typed = IDEALFitOptions(
        relative_frequencies_hz=base["relative_frequencies_hz"],
        echo_spacing_s=base["echo_spacing_s"],
        metabolite_names=base["metabolite_names"],
        execution=ExecutionConfig(use_multiprocessing=False),
        **{_TYPED_NAMES.get(k, k): v for k, v in {**extra, name: value}.items()},
    )
    _assert_complete_equal(actual, expected)
    _assert_complete_equal(fit_ideal(data, typed), expected)
    _assert_complete_equal(prepared, prepared_before)
    xr.testing.assert_identical(data, data_before)


@pytest.mark.parametrize(
    "selection", [["Field_offset"], ["Theta"], ["Field_offset", "Theta"]]
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("transpose", [False, True])
def test_ideal_selection_preserves_labels_precision_and_process_results(
    selection, dtype, transpose
):
    data = _invented_echoes(dtype, transpose)
    before = data.copy(deep=True)
    settings = {
        **_base_options(),
        "solve_for": selection,
        "init_f_offset_Hz": 100.0,
        "init_Theta": 0.1 + 0.04j,
        "stop_f_offset_hz": 1e-6,
        "stop_theta": 1e-6 + 1e-6j,
        "max_iterations": 3,
    }
    expected = fit_ideal(data, settings)
    with parallel_config(backend="loky"):
        actual = fit_ideal(
            data, {**settings, "use_multiprocessing": True, "number_of_cpu_cores": 2}
        )
    for result in (actual, expected):
        result.attrs["fit_prepared_options"].pop("use_multiprocessing")
        result.attrs["fit_prepared_options"].pop("number_of_cpu_cores")
    _assert_complete_equal(actual, expected)
    assert ("field_offset" in actual) == ("Field_offset" in selection)
    assert actual.fit_amps.dtype == np.complex128
    assert actual.fit_amps.dims == ("read_idx", "metabolite")
    np.testing.assert_array_equal(actual.read_idx, data.read_idx)
    assert actual.read_idx.dtype == data.read_idx.dtype
    _assert_metadata_equal(actual.read_idx.attrs, data.read_idx.attrs)
    assert actual.echo_idx.item() == data.echo_idx.isel(echo_idx=0).item()
    assert actual.metabolite.values.tolist() == settings["metabolite_names"]
    if "field_offset" in actual:
        assert actual.field_offset.dtype == np.float64
        assert actual.field_offset.attrs["units"] == "Hz"
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("key", ["init_T2_star_s", "stop_T2_star_s"])
@pytest.mark.parametrize(
    "route",
    [
        "prepare",
        "mapping",
        "prepared",
        "model",
        "information_only",
        "prepared_information_only",
    ],
)
def test_unsupported_t2_star_requests_fail_before_the_solver(key, route, monkeypatch):
    data = _invented_echoes()
    settings = {**_base_options(), key: 0.1}

    def rejected(*args, **kwargs):
        pytest.fail("An unsupported T2-star request reached the solver")

    monkeypatch.setattr(IDEAL_Solver, "__init__", rejected)
    with pytest.raises(ValueError, match=key + ".*not.*implemented.*[Rr]emove"):
        if route == "prepare":
            api.prepare_fit_params(data, "ideal", settings)
        elif route == "mapping":
            fit_ideal(data, settings)
        elif route in ("prepared", "prepared_information_only"):
            prepared = api.prepare_fit_params(data, "ideal", _base_options())
            prepared.attrs[key] = 0.1
            fit_ideal(
                data, prepared, return_params_info=route == "prepared_information_only"
            )
        elif route == "information_only":
            fit_ideal(data, settings, return_params_info=True)
        else:
            ideal.fit_ideal_dataset(data, settings)


@pytest.mark.parametrize(
    "selection", [[], ["T2_star"], ["Field_offset", "Field_offset"], "Theta"]
)
@pytest.mark.parametrize("prepared", [False, True])
def test_invalid_solve_selection_is_rejected_before_solver_construction(
    selection, prepared, monkeypatch
):
    def rejected(*args, **kwargs):
        pytest.fail("Invalid solve selection reached solver construction")

    monkeypatch.setattr(IDEAL_Solver, "__init__", rejected)
    data = _invented_echoes()
    settings = (
        api.prepare_fit_params(data, "ideal", _base_options())
        if prepared
        else _base_options()
    )
    if prepared:
        settings.attrs["solve_for"] = selection
    else:
        settings["solve_for"] = selection
    with pytest.raises(ValueError, match="IDEAL.*solve_for.*Field_offset.*Theta"):
        fit_ideal(data, settings)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_invalid_iteration_cap_is_rejected_without_starting_a_loop(
    iterations, prepared, monkeypatch
):
    def rejected(*args, **kwargs):
        pytest.fail("Invalid iteration cap reached solver construction")

    monkeypatch.setattr(IDEAL_Solver, "__init__", rejected)
    data = _invented_echoes()
    settings = (
        api.prepare_fit_params(data, "ideal", _base_options())
        if prepared
        else _base_options()
    )
    if prepared:
        settings.attrs["max_iterations"] = iterations
    else:
        settings["max_iterations"] = iterations
    with pytest.raises(ValueError, match="IDEAL.*max_iterations.*positive integer"):
        fit_ideal(data, settings)


def test_solve_selection_is_copied_by_preparation_and_frozen_options():
    selection = ["Field_offset"]
    prepared = api.prepare_fit_params(
        _invented_echoes(), "ideal", {**_base_options(), "solve_for": selection}
    )
    options = IDEALFitOptions(solve_for=selection)
    selection.append("Theta")
    assert prepared.attrs["solve_for"] == ["Field_offset"]
    assert options.solve_for == ("Field_offset",)
    with pytest.raises(FrozenInstanceError):
        options.max_iterations = 1


def test_omitted_and_none_settings_retain_the_actual_solver_defaults():
    data = _invented_echoes()
    base = _base_options()
    prepared = api.prepare_fit_params(data, "ideal", base)
    assert not ({name for name, _, _ in _CASES} & prepared.attrs.keys())
    default = fit_ideal(data, base)
    explicit = fit_ideal(data, {**base, "solve_for": ["Field_offset", "Theta"]})
    xr.testing.assert_allclose(default, explicit, rtol=1e-12, atol=1e-14)
    with_none = fit_ideal(data, {**base, **{name: None for name, _, _ in _CASES}})
    _assert_complete_equal(default, with_none)
    assert (
        "['Field_offset', 'Theta']"
        in ideal.get_ideal_params_info()["optional_fit_params"]["solve_for"]
    )


def test_requested_single_iteration_matches_characterized_reference():
    result = fit_ideal(_invented_echoes(), {**_base_options(), "max_iterations": 1})
    np.testing.assert_allclose(
        result.field_offset.isel(read_idx=0), -6.981282238960745, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("selection", [["Field_offset"], ["Field_offset", "Theta"]])
def test_active_settings_preserve_noise_free_synthetic_recovery(selection):
    # Independent forward equation, 3 Hz from a nearby start; tolerances protect
    # rounding-level recovery in this well-conditioned case, not arbitrary starts.
    indices = np.arange(9)
    times_s = (indices - 4) * 0.001
    amplitudes = np.array([1.5 + 0.2j, 0.4 - 0.1j])
    signal = np.sum(
        amplitudes
        * np.exp(2j * np.pi * (np.array([0.0, 120.0]) + 3.0) * times_s[:, None]),
        axis=1,
    )
    data = xr.DataArray(signal, dims="echo_idx", coords={"echo_idx": indices})
    before = data.copy(deep=True)
    result = fit_ideal(
        data,
        {
            **_base_options(),
            "solve_for": selection,
            "stop_f_offset_hz": 1e-6,
            "stop_theta": 1e-6 + 1e-6j,
        },
    )
    np.testing.assert_allclose(result.field_offset, 3.0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.fit_amps, amplitudes, rtol=0, atol=1e-9)
    xr.testing.assert_identical(data, before)


def test_singular_ideal_design_keeps_the_existing_exception_contract():
    data = _invented_echoes()
    before = data.copy(deep=True)
    with pytest.raises(np.linalg.LinAlgError):
        fit_ideal(
            data,
            {
                **_base_options(),
                "relative_frequencies_hz": [0.0, 0.0],
                "solve_for": ["Field_offset"],
            },
        )
    xr.testing.assert_identical(data, before)
