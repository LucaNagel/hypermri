"""Regress the approved additive equivalent-volume contract with invented data.

Analytical parameter products/quotients are compared at rtol 1e-12, atol 1e-14.
No optimizer reparameterization or recovery claim is involved. All metadata and
coordinates are synthetic; old clipped outputs remain part of the contract.
"""

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri.analysis.dce.fitting_support import (
    _add_clear_model_aliases,
    fit_extended_tofts_model,
    fit_sourbron_model,
    fit_tofts_model,
)
from hypermri.analysis.dce.models.sourbron import sourbron_separable_compartment_model


def _volume_dataset(model):
    flow = xr.DataArray(
        [[0.02, 0.03, np.nan], [0.0, -0.02, 0.025]],
        dims=("read_idx", "phase_idx"),
        coords={"read_idx": [4.0, -4.0], "phase_idx": [3, 2, 1]},
    )
    transit = xr.DataArray(
        [[15, 60, 40], [40, 15, 40]], dims=flow.dims, coords=flow.coords
    )
    if model == "sourbron":
        return xr.Dataset({"vp": xr.full_like(flow, 0.25), "Ft": flow, "Tt": transit})
    return xr.Dataset(
        {
            "Ktrans_2c": flow,
            "kep_2c": 1 / transit,
            "vp_2c": xr.full_like(flow, 0.25),
            "ve_2c": (flow / (1 / transit)).clip(min=0, max=1),
        }
    )


@pytest.mark.parametrize("model", ["tofts", "sourbron"])
@pytest.mark.parametrize("transpose", [False, True])
def test_additive_volume_reports_unbounded_value_clipping_and_preserves_inputs(
    model, transpose
):
    source = _volume_dataset(model)
    if transpose:
        source = source.transpose("phase_idx", "read_idx")
    before = source.copy(deep=True)
    result = _add_clear_model_aliases(source)
    expected = xr.DataArray(
        [[0.3, 1.8, np.nan], [0.0, -0.3, 1.0]],
        dims=("read_idx", "phase_idx"),
        coords=source.coords,
    ).transpose(*source["vp" if model == "sourbron" else "vp_2c"].dims)
    np.testing.assert_allclose(
        result["ve_equivalent"], expected, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        result["ve_2c"], expected.clip(min=0, max=1), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        result["ve_extended"], result["ve_2c"], rtol=1e-12, atol=1e-14
    )
    assert result["ve_equivalent"].dims == expected.dims
    assert result["ve_equivalent"].dtype == np.float64
    assert result["ve_equivalent"].attrs["units"] == "1"
    assert result["ve_was_clipped"].dtype == bool
    np.testing.assert_array_equal(
        result["ve_was_clipped"], (expected < 0) | (expected > 1)
    )
    assert "NaN" in result["ve_was_clipped"].attrs["description"]
    assert (
        "ve_equivalent = Ft * Tt" in result.attrs["parameterization_equivalence_note"]
    )
    xr.testing.assert_identical(source, before)
    xr.testing.assert_identical(_add_clear_model_aliases(result), result)


def _volume_fit_inputs():
    times = np.arange(45, dtype=np.float64) * 4
    lag = np.maximum(times - 8, 0) / 8
    plasma = lag**3 * np.exp(-lag)
    plasma /= plasma.max()
    tissue = sourbron_separable_compartment_model(
        times,
        plasma,
        plasma_volume_fraction=0.25,
        tubular_flow_s=0.03,
        tubular_transit_time_s=60,
    )
    data = xr.DataArray(
        np.stack([tissue, np.full_like(tissue, np.nan)]),
        dims=("read_idx", "repetition_idx"),
        coords={
            "read_idx": [4.0, -4.0],
            "repetition_idx": np.arange(len(times)),
            "time_s": ("repetition_idx", times),
        },
        attrs={"units": "mM", "git_commit_hash": "invented-volume-fixture"},
    )
    return data, times, plasma


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("model", ["tofts", "extended_tofts", "sourbron"])
def test_public_fit_volume_outputs_preserve_unavailable_status_and_dimension_order(
    model, parallel
):
    data, times, plasma = _volume_fit_inputs()
    data = data.transpose("repetition_idx", "read_idx")
    before = data.copy(deep=True)
    fitter = {
        "tofts": fit_tofts_model,
        "extended_tofts": fit_extended_tofts_model,
        "sourbron": fit_sourbron_model,
    }[model]
    options = {"number_of_cpu_cores": 2, "model_mode": "2comp"}
    if model == "sourbron":
        options["x0"] = [0.25, 0.03, 60]
    with parallel_config(backend="threading"):
        result = fitter(
            data,
            aif_concentration=plasma,
            aif_time_s=times,
            fit_params=options,
            use_multiprocessing=parallel,
        )
    np.testing.assert_allclose(
        result["ve_equivalent"], [1.8, np.nan], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(result["ve_2c"], [1.0, np.nan], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(result["ve_was_clipped"], [True, False])
    assert result["ve_equivalent"].dims == ("read_idx",)
    assert result["ve_equivalent"].attrs["units"] == "1"
    xr.testing.assert_identical(result.coords["read_idx"], data.coords["read_idx"])
    xr.testing.assert_identical(data, before)
    if model == "sourbron":
        np.testing.assert_array_equal(result["fit_success"], [True, False])


def test_finite_plasma_transit_keeps_its_separate_output_contract():
    source = _volume_dataset("sourbron")
    source["Tp"] = xr.full_like(source["vp"], 5.0)
    result = _add_clear_model_aliases(source)
    assert "ve_equivalent" not in result
    assert "ve_was_clipped" not in result
    assert "ve_2c" not in result


def test_unavailable_decay_does_not_invent_a_volume_or_clipping_status():
    source = xr.Dataset(
        {
            "Ktrans_2c": ("read_idx", [0.03, 0.03, 0.03, 0.03]),
            "kep_2c": ("read_idx", [0.0, -0.1, np.nan, np.inf]),
            "vp_2c": ("read_idx", [0.25] * 4),
            "ve_2c": ("read_idx", [np.nan] * 4),
        }
    )
    result = _add_clear_model_aliases(source)
    assert np.isnan(result["ve_equivalent"]).all()
    assert not result["ve_was_clipped"].any()
