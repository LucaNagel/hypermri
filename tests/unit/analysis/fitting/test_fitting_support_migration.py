"""Protect fitting support previously moves its four implementation modules.

All signals, coordinates and metadata are invented. SPGR inversion is checked
against the closed-form forward signal at 1e-12 mM absolute tolerance. Voxel
assembly compares simple sums at 1e-12 signal units; no optimizer is involved.
Parallel controls use joblib's threading backend to test the same dispatcher
without requiring process creation in restricted environments.
"""

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri.analysis.dce.conversions import (
    concentration_to_signal,
    signal_to_concentration,
)
from hypermri.analysis.fitting import execution
from hypermri.analysis.fitting.result_metadata import (
    dataset_reconstruction_array,
    dataset_reconstruction_values,
    inherit_input_xarray_metadata,
    select_reconstruction_array,
    set_reconstruction_variable,
)
from hypermri.analysis.fitting.voxelwise import VoxelwiseExecutor


def test_execution_config_normalizes_only_worker_options():
    params = {"number_of_cpu_cores": None, "use_multiprocessing": 0, "r1": 4.0}
    before = deepcopy(params)
    config = execution.ExecutionConfig.from_mapping(params)
    assert config.as_fit_params() == {
        "use_multiprocessing": False,
        "number_of_cpu_cores": -1,
    }
    assert params == before
    assert execution.ExecutionConfig.from_mapping(None).as_fit_params() == {
        "use_multiprocessing": True,
        "number_of_cpu_cores": -1,
    }
    target = {"r1": 3.5}
    assert execution.copy_execution_params(params, target) is target
    assert target == {"r1": 3.5, **{key: params[key] for key in config.as_fit_params()}}
    assert execution.copy_execution_params(None, target) is target


@pytest.mark.parametrize("parallel", [False, True])
def test_index_dispatch_keeps_all_results_and_empty_input(parallel):
    config = execution.ExecutionConfig(parallel, 2)
    with parallel_config(backend="threading"):
        result = execution.run_indexed_tasks(
            iter([3, 1, 2]),
            lambda index: (index, index**2),
            execution_config=config,
            parallel_desc="parallel",
            sequential_desc="serial",
        )
    assert sorted(result) == [(1, 1), (2, 4), (3, 9)]
    if not parallel:
        assert [index for index, _ in result] == [3, 1, 2]
    assert (
        execution.run_indexed_tasks(
            [],
            lambda _: pytest.fail("empty dispatch called a task"),
            execution_config=config,
            parallel_desc="parallel",
            sequential_desc="serial",
        )
        == []
    )


@pytest.mark.parametrize("old_joblib", [False, True])
def test_dispatch_permission_fallback_warns_once(monkeypatch, old_joblib):
    calls = []

    def unavailable_parallel(**kwargs):
        calls.append(kwargs)
        if old_joblib and "return_as" in kwargs:
            raise TypeError("unexpected keyword argument 'return_as'")
        raise PermissionError("invented unavailable process backend")

    monkeypatch.setattr(execution, "Parallel", unavailable_parallel)
    with pytest.warns(RuntimeWarning, match="Falling back to sequential") as records:
        result = execution.run_indexed_tasks(
            [2, 1],
            lambda index: index + 1,
            execution_config=execution.ExecutionConfig(True, 2),
            parallel_desc="parallel",
            sequential_desc="serial",
        )
    assert result == [3, 2]
    assert len(records) == 1
    assert len(calls) == (2 if old_joblib else 1)


def test_dispatch_propagates_task_typeerror(monkeypatch):
    def broken_parallel(**kwargs):
        raise TypeError("invented task error")

    monkeypatch.setattr(execution, "Parallel", broken_parallel)
    with pytest.raises(TypeError, match="invented task error"):
        execution.run_indexed_tasks(
            [0, 1],
            lambda index: index,
            execution_config=execution.ExecutionConfig(True, 2),
            parallel_desc="parallel",
            sequential_desc="serial",
        )


def test_reconstruction_metadata_precedence_selection_and_storage():
    dataset = xr.Dataset(
        {"baseline": ("read_idx", [10.0, 20.0])},
        coords={"read_idx": [4.0, -4.0], "time_s": ("sample", [1.0, 3.0])},
        attrs={"baseline": np.array([-1.0]), "legacy": [7.0, 8.0]},
    )
    before = dataset.copy(deep=True)
    xr.testing.assert_identical(
        dataset_reconstruction_array(dataset, "baseline"), dataset.baseline
    )
    assert (
        select_reconstruction_array(
            dataset, "baseline", {"read_idx": 1, "absent": 0}
        ).item()
        == 20
    )
    np.testing.assert_array_equal(
        dataset_reconstruction_values(dataset, "legacy"), [7.0, 8.0]
    )
    xr.testing.assert_identical(
        dataset_reconstruction_array(dataset, "time_s"), dataset.time_s
    )
    assert dataset_reconstruction_array(dataset, "missing") is None
    assert dataset_reconstruction_values(dataset, "missing") is None
    xr.testing.assert_identical(dataset, before)
    assert (
        set_reconstruction_variable(dataset, "ignored", None, dims=["sample"]) is None
    )
    set_reconstruction_variable(
        dataset,
        "aif",
        np.array([0.1, 0.2], dtype=np.float32),
        dims=["sample"],
        attrs={"units": "mM"},
    )
    assert dataset.aif.dtype == np.float32
    assert dataset.aif.dims == ("sample",)
    assert dataset.aif.attrs == {"units": "mM"}
    assert "ignored" not in dataset


def test_inherited_metadata_precedence_and_deep_attrs():
    source = xr.DataArray(
        np.ones((2, 3)),
        dims=("read_idx", "sample"),
        coords={"read_idx": [4.0, -4.0], "sample": [0, 1, 2], "fixture": "invented"},
        attrs={"units": "signal", "nested": {"value": [1]}, "owner": "input"},
    )
    source.read_idx.attrs = {"units": "mm", "nested": {"value": [2]}}
    output = xr.Dataset(
        {"amplitude": ("read_idx", [1.0, 2.0])},
        coords={"read_idx": [4.0, -4.0]},
        attrs={"owner": "result"},
    )
    output.read_idx.attrs["units"] = "custom"
    before = source.copy(deep=True)
    result = inherit_input_xarray_metadata(output, source)
    assert result is not output  # assign_coords creates a new dataset.
    assert result.attrs["owner"] == "result"
    assert result.read_idx.attrs["units"] == "custom"
    assert result.fixture.item() == "invented"
    assert "sample" not in result.coords
    result.attrs["nested"]["value"].append(3)
    result.read_idx.attrs["nested"]["value"].append(4)
    xr.testing.assert_identical(source, before)


@pytest.mark.parametrize("labelled", [False, True])
def test_spgr_roundtrip_matches_independent_forward_equation(labelled):
    concentration = np.array([[0.0, 0.1, 0.5], [0.0, 0.3, 0.8]])
    baseline, t10, tr, angle, r1 = 100.0, 1.3, 0.004, np.deg2rad(15), 3.5
    e10 = np.exp(-tr / t10)
    e1 = np.exp(-tr * (1 / t10 + r1 * concentration))
    reference = (
        baseline
        * (1 - np.cos(angle) * e10)
        / (1 - e10)
        * (1 - e1)
        / (1 - np.cos(angle) * e1)
    )
    if labelled:
        concentration = xr.DataArray(
            concentration,
            dims=("read_idx", "time_s"),
            coords={"read_idx": [4.0, -4.0], "time_s": [0.0, 2.0, 5.0]},
        )
    before = concentration.copy()
    signal = concentration_to_signal(concentration, baseline, t10, tr, 15.0, r1)
    np.testing.assert_allclose(signal, reference, rtol=1e-12, atol=1e-12)
    recovered = signal_to_concentration(signal, baseline, t10, tr, 15.0, r1)
    np.testing.assert_allclose(recovered, concentration, rtol=1e-12, atol=1e-12)
    if labelled:
        assert recovered.dims == concentration.dims
        xr.testing.assert_identical(
            recovered.coords.to_dataset(), concentration.coords.to_dataset()
        )
        xr.testing.assert_identical(concentration, before)
    else:
        np.testing.assert_array_equal(concentration, before)


@pytest.mark.parametrize("mode", ["signal_enhancement", "relative_signal_enhancement"])
def test_linear_eq8_and_plasma_units(mode):
    signal = np.array([100.0, 110.0, 130.0])
    actual = signal_to_concentration(
        signal,
        100.0,
        1.25,
        r1=4.0,
        mode=mode,
        linear_scale="eq8",
        convert_blood_to_plasma=True,
        hematocrit=0.4,
    )
    np.testing.assert_allclose(
        actual, (signal / 100.0 - 1) / (4 * 1.25 * 0.6), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("behaviour", ["clip", "nan", "nan_all"])
def test_spgr_saturation_diagnostics_and_last_axis_rule(behaviour):
    signal = xr.DataArray([[100.0, 1e6], [100.0, 110.0]], dims=("read_idx", "sample"))
    before = signal.copy(deep=True)
    result, info = signal_to_concentration(
        signal,
        100.0,
        1.3,
        0.004,
        15.0,
        3.5,
        saturation_behavior=behaviour,
        return_saturation_info=True,
    )
    assert info["saturation_detected"]
    assert info["saturation_count"] == 1
    assert info["saturation_fraction"] == 0.25
    np.testing.assert_array_equal(
        info["saturation_mask"], [[False, True], [False, False]]
    )
    expected_nan = {
        "clip": [[False, False], [False, False]],
        "nan": [[False, True], [False, False]],
        "nan_all": [[True, True], [False, False]],
    }[behaviour]
    np.testing.assert_array_equal(np.isnan(result), expected_nan)
    xr.testing.assert_identical(signal, before)


def test_signal_conversion_retains_nan_failure_contract():
    result, info = signal_to_concentration(
        np.ones(3), mode="invalid", return_saturation_info=True
    )
    assert np.isnan(result)
    assert info == {}


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("dims", [("read_idx", "sample"), ("sample", "read_idx")])
def test_voxel_assembly_preserves_order_precision_missing_results_and_provenance(
    parallel, dtype, dims
):
    data = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
        dims=("read_idx", "sample"),
        coords={
            "read_idx": [4.0, -4.0],
            "sample": [10.0, 20.0, 40.0],
            "fixture": "invented",
        },
        attrs={"units": "signal", "git_commit_hash": "synthetic-h3"},
    ).transpose(*dims)
    data.read_idx.attrs["units"] = "mm"
    before = data.copy(deep=True)
    params = {
        "use_multiprocessing": parallel,
        "number_of_cpu_cores": 2,
        "guess": np.array([1.0], dtype=np.float32),
    }

    def fit_voxel(*, signal, dynamic_coords, fit_params, indices):
        assert signal.dtype == (np.float64 if dtype == np.float32 else np.complex64)
        np.testing.assert_array_equal(dynamic_coords, [10, 20, 40])
        assert fit_params["guess"].dtype == np.float64
        if indices == (1,):
            return None, indices
        return {"amplitude": signal.sum(), "discarded": np.ones(100)}, indices

    with parallel_config(backend="threading"):
        result = VoxelwiseExecutor().execute(
            data,
            "b1",
            params,
            fit_voxel,
            ["amplitude"],
            "sample",
            model_name="invented",
            model_equation="sum(signal)",
        )
    assert result.amplitude.dims == ("read_idx",)
    assert result.amplitude.dtype == (
        np.float64 if dtype == np.float32 else np.complex64
    )
    np.testing.assert_allclose(
        result.amplitude, [6.0, np.nan], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    assert "discarded" not in result
    xr.testing.assert_identical(result.read_idx, data.read_idx)
    assert result.attrs["git_commit_hash"] == "synthetic-h3"
    assert result.attrs["model_name"] == "invented"
    assert result.attrs["dynamic_dim"] == "sample"
    assert result.attrs["fa_deg"] == 90
    assert params["guess"].dtype == np.float64  # Established dictionary side effect.
    xr.testing.assert_identical(data, before)
