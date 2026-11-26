"""Routing contracts on invented quantitative signals, with no scanner fixtures.

Structural comparisons use rtol=1e-12 and atol=1e-14 in the model's units.
These tolerances protect identical execution, not physiological fit accuracy.
"""

import contextlib
import io

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.fit_maps import fit_quantitative_maps


def _invented_input(model, transpose=False, complex_storage=False):
    if model == "t1":
        axis = np.array([50.0, 150.0, 400.0, 900.0, 1500.0])
        signal = 2.0 * (1.0 - 2.0 * np.exp(-axis / 700.0))
        dynamic = "inversion_idx"
        options = {"t1_values_s": np.array([0.5, 0.7, 0.9])}
    elif model == "adc":
        axis = np.array([0.0, 100.0, 300.0, 700.0, 1000.0])
        signal = 2.0 * np.exp(-axis * 0.0012)
        dynamic = "bvalue_idx"
        options = {}
    elif model.startswith("b1"):
        axis = np.linspace(0.25, 9.0, 8)
        signal = 2.0 * np.sin(np.pi / 4.0 * np.sqrt(axis))
        dynamic = "rfpower_idx"
        options = {
            "fa_deg": 90.0,
            "trf_ms": 1.0,
            "sint": 1.0,
            "period_samples": 40,
        }
        if model == "b1_region":
            options.update(
                use_region_growing=True,
                seed_voxel={"read_idx": 0, "phase_idx": 0, "slice_idx": 0},
                return_debug_info=True,
            )
    else:
        axis = np.arange(6)
        times = (axis - (axis.size - 1) / 2) * 0.001
        signal = 1.5 + 0.4 * np.exp(2j * np.pi * 120.0 * times)
        dynamic = "echo_idx"
        options = {
            "relative_frequencies_hz": np.array([0.0, 120.0]),
            "echo_spacing_s": 0.001,
            "metabolite_names": ["invented_a", "invented_b"],
        }
    dtype = np.complex64 if complex_storage else np.float32
    if model == "ideal":
        dtype = np.complex64 if complex_storage else np.complex128
    signal = np.stack([signal, 0.8 * signal]).astype(dtype)
    data = xr.DataArray(
        signal[:, None, None, :],
        dims=("read_idx", "phase_idx", "slice_idx", dynamic),
        coords={"read_idx": [7, 3], "phase_idx": [2], "slice_idx": [1], dynamic: axis},
        attrs={"unit": "a.u.", "invented_fixture": True},
    )
    if transpose:
        data = data.transpose(dynamic, "slice_idx", "phase_idx", "read_idx")
    return data, options


def _assert_metadata_equal(actual, expected):
    if isinstance(actual, xr.DataArray):
        xr.testing.assert_identical(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_metadata_equal(actual[key], expected[key])
    else:
        np.testing.assert_equal(actual, expected)


def _assert_complete_equal(actual, expected):
    xr.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert dict(actual.sizes) == dict(expected.sizes)
    assert list(actual) == list(expected)
    _assert_metadata_equal(actual.attrs, expected.attrs)
    xr.testing.assert_identical(
        xr.Dataset(coords=actual.coords), xr.Dataset(coords=expected.coords)
    )
    for name in actual:
        assert actual[name].dims == expected[name].dims
        assert actual[name].dtype == expected[name].dtype
        _assert_metadata_equal(actual[name].attrs, expected[name].attrs)


@pytest.mark.parametrize("model", ["t1", "adc", "b1", "b1_region", "ideal"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("complex_storage", [False, True])
def test_quantitative_legacy_and_prepared_routes_preserve_full_results(
    model, transpose, complex_storage
):
    data, options = _invented_input(model, transpose, complex_storage)
    before = data.copy(deep=True)
    canonical = "b1" if model == "b1_region" else model
    options["use_multiprocessing"] = False
    warnings = (
        pytest.warns(np.exceptions.ComplexWarning)
        if model.startswith("b1") and not complex_storage
        else contextlib.nullcontext()
    )
    with (
        warnings,
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        prepared = api.prepare_fit_params(data, canonical, options)
        expected = api.fit(data, canonical, prepared)
        actual = fit_quantitative_maps(data, canonical, options)
    _assert_complete_equal(actual, expected)
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("model", ["t1", "adc", "b1", "ideal"])
def test_quantitative_threaded_and_serial_routes_agree(model):
    data, options = _invented_input(model, complex_storage=True)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        expected = api.fit(data, model, {**options, "use_multiprocessing": False})
        with parallel_config(backend="threading"):
            actual = api.fit(
                data,
                model,
                {**options, "use_multiprocessing": True, "number_of_cpu_cores": 2},
            )
    assert actual.attrs["fit_prepared_options"]["use_multiprocessing"] is True
    assert expected.attrs["fit_prepared_options"]["use_multiprocessing"] is False
    for result in (actual, expected):
        for key in ("use_multiprocessing", "number_of_cpu_cores"):
            result.attrs["fit_prepared_options"].pop(key, None)
    _assert_complete_equal(actual, expected)


def _domain_function(model):
    from hypermri.analysis import quantitative

    return getattr(quantitative, f"fit_{model}")


@pytest.mark.parametrize("model", ["t1", "adc", "b1", "ideal"])
def test_named_fits_do_not_call_facade_registry_or_compatibility(monkeypatch, model):
    from hypermri.analysis.fitting import registry

    data, options = _invented_input(model, complex_storage=True)
    options["use_multiprocessing"] = False
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        expected = api.fit(data, model, options)

    def rejected(*args, **kwargs):
        pytest.fail("Named fit returned through compatibility dispatch")

    monkeypatch.setattr(api, "fit", rejected)
    monkeypatch.setattr(api, "prepare_fit_params", rejected)
    monkeypatch.setattr(registry, "get_registry", rejected)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        actual = _domain_function(model)(data, options)
    _assert_complete_equal(actual, expected)


@pytest.mark.parametrize("model", ["t1", "adc", "b1", "ideal"])
def test_named_fit_prepares_once_and_prepared_registry_bridge_delegates(
    monkeypatch, model
):
    from hypermri.analysis.fitting import registry
    from hypermri.analysis.quantitative.fitting import _QUANTITATIVE_MODELS

    data, options = _invented_input(model, complex_storage=True)
    options["use_multiprocessing"] = False
    module, fitter, _ = _QUANTITATIVE_MODELS[model]
    prepare_name = f"prepare_{model}_fit_params_dataset"
    original = getattr(module, prepare_name)
    count = 0

    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, prepare_name, counted)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        expected = fitter(data, options)
        assert count == 1
        prepared = original(data, options)
        before = prepared.copy(deep=True)
        actual = registry.get_registry().get(model).fit(data, prepared)
    assert count == 1
    _assert_complete_equal(actual, expected)
    _assert_complete_equal(prepared, before)
    assert actual.attrs["fit_package_version"]
    _assert_metadata_equal(actual.attrs["fit_prepared_options"], dict(prepared.attrs))


@pytest.mark.parametrize("model", ["t1", "adc", "b1", "ideal"])
def test_immutable_options_match_mapping_and_copy_sequences(model):
    from dataclasses import FrozenInstanceError

    from hypermri.analysis.fitting.execution import ExecutionConfig
    from hypermri.analysis.quantitative import (
        ADCFitOptions,
        B1FitOptions,
        IDEALFitOptions,
        T1FitOptions,
    )

    data, mapping = _invented_input(model, complex_storage=True)
    mapping["use_multiprocessing"] = False
    execution = ExecutionConfig(use_multiprocessing=False)
    mapping["number_of_cpu_cores"] = execution.number_of_cpu_cores
    if model == "t1":
        grid = [0.5, 0.7, 0.9]
        options = T1FitOptions(t1_values_s=grid, execution=execution)
        grid[0] = 10.0
        assert options.t1_values_s == (0.5, 0.7, 0.9)
    elif model == "adc":
        options = ADCFitOptions(execution=execution)
    elif model == "b1":
        options = B1FitOptions(
            fa_deg=90.0, trf_ms=1.0, sint=1.0, period_samples=40, execution=execution
        )
    else:
        options = IDEALFitOptions(
            relative_frequencies_hz=(0.0, 120.0),
            echo_spacing_s=0.001,
            metabolite_names=("invented_a", "invented_b"),
            execution=execution,
        )
        mapping["relative_frequencies_hz"] = [0.0, 120.0]
    with pytest.raises(FrozenInstanceError):
        options.execution = None
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        expected = api.fit(data, model, mapping)
        actual = _domain_function(model)(data, options)
    _assert_complete_equal(actual, expected)


@pytest.mark.parametrize("model", ["t1", "adc", "b1", "ideal"])
def test_wrong_scientific_option_container_is_rejected(model):
    from hypermri.analysis.quantitative import ADCFitOptions, T1FitOptions

    data, _ = _invented_input(model)
    wrong = T1FitOptions() if model == "adc" else ADCFitOptions()
    with pytest.raises(TypeError, match=f"{model} fitting requires"):
        _domain_function(model)(data, wrong)


def test_recorded_prepared_options_are_independent_of_supplied_parameters():
    data, mapping = _invented_input("t1")
    mapping["use_multiprocessing"] = False
    prepared = api.prepare_fit_params(data, "t1", mapping)
    before = prepared.copy(deep=True)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = _domain_function("t1")(data, prepared)
    result.attrs["fit_prepared_options"]["t1_values_s"][0] = 42.0
    _assert_complete_equal(prepared, before)


@pytest.mark.parametrize(
    "route", ["named", "facade", "legacy", "registry", "surface_coil"]
)
@pytest.mark.parametrize("two_iterations", [False, True])
def test_b1_region_outputs_and_options_survive_all_public_routes(route, two_iterations):
    """Protect the complete saved B1 schema, beyond agreement between routes."""
    from hypermri import __version__
    from hypermri.analysis.anatomical.b1_mapping import fit_b1_surface_coil
    from hypermri.analysis.fitting.registry import get_registry
    from hypermri.analysis.quantitative import fit_b1

    data, options = _invented_input("b1_region", complex_storage=True)
    options.update(
        use_multiprocessing=False,
        use_two_iteration_fit=two_iterations,
    )
    prepared = api.prepare_fit_params(data, "b1", options)
    before = prepared.copy(deep=True)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        if route == "named":
            result = fit_b1(data, prepared)
        elif route == "facade":
            result = api.fit(data, "b1", options)
        elif route == "legacy":
            result = fit_quantitative_maps(data, "b1", options)
        elif route == "registry":
            result = get_registry().get("b1").fit(data, prepared)
        else:
            result = fit_b1_surface_coil(data, fit_params=options)

    base_variables = {
        "amplitude",
        "period",
        "refpower_90deg_W",
        "rss_b1",
        "stop_index",
        "n_valid_samples",
    }
    expected_variables = base_variables | {
        "fit_order_map",
        "stop_index_map_iter1",
        "region_mask",
    }
    if two_iterations:
        expected_variables |= {f"{name}_iter1" for name in base_variables}
    assert set(result.data_vars) == expected_variables
    assert result.fit_order_map.dims == ("read_idx", "phase_idx")
    xr.testing.assert_identical(result.read_idx, data.read_idx)
    xr.testing.assert_identical(result.phase_idx, data.phase_idx)
    np.testing.assert_array_equal(result.fit_order_map.values[:, 0], [0.0, 1.0])
    assert result.region_mask.dtype == bool
    assert result.attrs["has_init_params_map"] is True
    assert result.attrs["fit_package_version"] == __version__
    for name, expected in (
        ("model_name", "b1_sinusoidal"),
        ("dynamic_dim", "rfpower_idx"),
        ("fa_deg", 90.0),
        ("trf_ms", 1.0),
        ("sint", 1.0),
        ("unit", "a.u."),
        ("invented_fixture", True),
    ):
        assert result.attrs[name] == expected
    np.testing.assert_array_equal(result.attrs["refpower_W"], data.rfpower_idx)
    _assert_metadata_equal(result.attrs["fit_prepared_options"], dict(prepared.attrs))
    _assert_complete_equal(prepared, before)

    # Object snapshots must retain both the map and nested provenance independently.
    snapshot = result.copy(deep=True)
    _assert_complete_equal(snapshot, result)
    snapshot.attrs["fit_prepared_options"]["seed_voxel"]["read_idx"] = 99
    assert result.attrs["fit_prepared_options"]["seed_voxel"]["read_idx"] == 0


def test_b1_legacy_stacked_repetition_mean_retains_order_and_options():
    """Reproduce the reported call with invented data and a smaller seed position."""
    data, options = _invented_input("b1_region", complex_storage=True)
    stacked = data.expand_dims(
        acquisition_idx=[0], channel_idx=[0], repetition_idx=[0, 1]
    )
    averaged = np.mean(stacked, axis=stacked.get_axis_num("repetition_idx"))
    options.update(
        period_samples=11111,
        seed_voxel={
            "read_idx": 1,
            "phase_idx": 0,
            "acquisition_idx": 0,
            "slice_idx": 0,
            "channel_idx": 0,
        },
        region_growing_priority="snr",
        min_stop_index=3,
        neighbor_period_search_width=0.1,
        use_two_iteration_fit=True,
        use_neighbor_init_values=True,
        stop_index_extra=-1,
        use_multiprocessing=False,
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = fit_quantitative_maps(data=averaged, fit_type="b1", fit_params=options)

    assert result.fit_order_map.dims == ("read_idx", "phase_idx")
    np.testing.assert_array_equal(result.fit_order_map.values[:, 0], [1.0, 0.0])
    assert result.amplitude.dims == tuple(
        dim for dim in averaged.dims if dim != "rfpower_idx"
    )
    for name, value in options.items():
        _assert_metadata_equal(result.attrs["fit_prepared_options"][name], value)
