"""Characterize 2C-IRF execution before replacing its separate voxel loop.

All curves and metadata are invented. Model-generated curves isolate execution
and labelled-output behavior; independent numerical references remain in ``test_dce_recovery.py``.
Serial/parallel comparisons use rtol 1e-12, atol 1e-14 because they perform the
same deterministic float64 fit, rather than testing parameter recovery.
"""

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri.analysis.dce.fitting_support import fit_two_c_irf_model
from hypermri.analysis.dce.models.two_c_irf import fit_two_c_irf_curve, two_c_irf_model


def _two_c_inputs(dtype=np.float64):
    times = 17.0 + np.array([0, 2, 5, 8, 12, 17, 23, 30, 38, 47, 57, 68])
    aif = np.array([0, 0.1, 0.4, 0.8, 0.7, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03])
    parameters = np.array([0.018, 0.014, 4.0, 0.16, 8.0, 0.08])
    curve = two_c_irf_model(
        times,
        aif,
        gfr_s=parameters[0],
        rpf_s=parameters[0] + parameters[1],
        vascular_delay_s=parameters[2],
        vascular_dispersion_rate_s=parameters[3],
        tubular_delay_s=parameters[4],
        tubular_dispersion_rate_s=parameters[5],
    )
    curves = np.stack([curve, 0.7 * curve, np.full_like(curve, np.nan), curve])
    curves[3, 5:] = np.nan
    data = xr.DataArray(
        curves.astype(dtype).reshape(2, 2, len(times)),
        dims=("read_idx", "phase_idx", "repetition_idx"),
        coords={
            "read_idx": xr.DataArray(
                [4.0, -4.0], dims="read_idx", attrs={"units": "mm"}
            ),
            "phase_idx": [3, 1],
            "repetition_idx": np.arange(len(times)) + 10,
            "time_s": xr.DataArray(times, dims="repetition_idx", attrs={"units": "s"}),
            "voxel_volume_ml": (("read_idx", "phase_idx"), [[0.1, 0.2], [0.3, 0.4]]),
            "fixture_source": "invented",
        },
        attrs={"units": "mM", "git_commit_hash": "synthetic-reference"},
    )
    return data, times, aif, {"x0": parameters, "number_of_cpu_cores": 2}


def _fit_map(data, times, aif, params, parallel):
    with parallel_config(backend="threading"):
        return fit_two_c_irf_model(
            data,
            aif_concentration=aif,
            aif_time_s=times,
            fit_params=params,
            use_multiprocessing=parallel,
        )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "dims",
    [
        ("read_idx", "phase_idx", "repetition_idx"),
        ("repetition_idx", "phase_idx", "read_idx"),
        ("phase_idx", "repetition_idx", "read_idx"),
    ],
)
def test_two_c_voxel_execution_preserves_labelled_contract(parallel, dtype, dims):
    data, times, aif, params = _two_c_inputs(dtype)
    data = data.transpose(*dims)
    before = data.copy(deep=True)
    params_before = deepcopy(params)
    aif_before = aif.copy()
    result = _fit_map(data, times, aif, params, parallel)
    spatial_dims = tuple(dim for dim in dims if dim != "repetition_idx")

    assert result["GFR"].dims == spatial_dims
    assert result["GFR"].dtype == np.float64
    assert result["fit_success"].dtype == bool
    assert result["n_points_fit"].dtype == np.int64
    xr.testing.assert_identical(result.coords["read_idx"], data.coords["read_idx"])
    xr.testing.assert_identical(
        result.coords["voxel_volume_ml"], data.coords["voxel_volume_ml"]
    )
    xr.testing.assert_identical(result.coords["time_s"], data.coords["time_s"])
    assert result.coords["fixture_source"].item() == "invented"
    assert result.attrs == {
        "dynamic_dim": "repetition_idx",
        "fit_type": "2c_irf",
        "model_name": "2c_irf",
        "model_equation": (
            "Ccortex(t) = Cp(t) * Rcortex(t), where Rcortex is the cortical "
            "piecewise-exponential 2C-IRF retention kernel of Chen et al. (2014)."
        ),
        "aif_concentration_space": "plasma",
    }
    assert result["GFR"].attrs == {}
    assert result["time_points_s"].attrs == {"units": "s"}
    assert result["aif_time_s"].attrs == {"units": "s"}
    assert result["aif_concentration"].attrs == {"concentration_space": "plasma"}
    assert not any("covariance" in name or "uncertainty" in name for name in result)
    np.testing.assert_array_equal(
        result["fit_success"].transpose("read_idx", "phase_idx"),
        [[True, True], [False, False]],
    )
    np.testing.assert_array_equal(
        result["n_points_fit"].transpose("read_idx", "phase_idx"), [[12, 12], [0, 5]]
    )
    np.testing.assert_array_equal(
        result["fit_window_mask"], np.ones(len(times), dtype=bool)
    )
    for read in range(2):
        for phase in range(2):
            reference = fit_two_c_irf_curve(
                data.isel(read_idx=read, phase_idx=phase).values.astype(np.float64),
                aif_concentration=aif,
                time_points_s=times,
                fit_params=params,
            )
            for name in (
                "GFR",
                "RPF",
                "rss",
                "rmse",
                "vascular_delay_s",
                "tubular_delay_s",
            ):
                np.testing.assert_allclose(
                    result[name].isel(read_idx=read, phase_idx=phase),
                    reference[name],
                    rtol=1e-12,
                    atol=1e-14,
                    equal_nan=True,
                )
    xr.testing.assert_identical(data, before)
    np.testing.assert_array_equal(aif, aif_before)
    np.testing.assert_array_equal(params["x0"], params_before["x0"])
    assert params.keys() == params_before.keys()


def test_two_c_voxel_execution_supports_scalar_curve_and_repeated_calls():
    data, times, aif, params = _two_c_inputs()
    curve = data.isel(read_idx=0, phase_idx=0, drop=True)
    first = _fit_map(curve, times, aif, params, False)
    second = _fit_map(curve, times, aif, params, True)
    assert first["GFR"].dims == ()
    assert first["fit_success"].item()
    xr.testing.assert_identical(first, second)


@pytest.mark.parametrize("parallel", [False, True])
def test_two_c_voxel_execution_preserves_text_sample_labels(parallel):
    data, times, aif, params = _two_c_inputs()
    reference = _fit_map(data, times, aif, params, False)
    labels = [f"sample-{index}" for index in range(len(times))]
    labelled = data.assign_coords(repetition_idx=labels)
    before = labelled.copy(deep=True)
    result = _fit_map(labelled, times, aif, params, parallel)
    xr.testing.assert_identical(result, reference.assign_coords(repetition_idx=labels))
    xr.testing.assert_identical(labelled, before)


@pytest.mark.parametrize("parallel", [False, True])
def test_two_c_voxel_execution_keeps_finite_unsuccessful_fit(parallel):
    data, times, aif, params = _two_c_inputs()
    params.update({"x0": [0.012, 0.02, 3, 0.2, 6, 0.1], "max_nfev": 1})
    result = _fit_map(data, times, aif, params, parallel)
    assert not result["fit_success"].any()
    assert np.isfinite(result["GFR"].isel(read_idx=0)).all()
    assert np.isnan(result["GFR"].isel(read_idx=1)).all()


@pytest.mark.parametrize("parallel", [False, True])
def test_two_c_voxel_execution_propagates_invalid_optimizer_option(parallel):
    data, times, aif, params = _two_c_inputs()
    params["max_nfev"] = 0
    with pytest.raises(ValueError, match="max_nfev"):
        _fit_map(data, times, aif, params, parallel)
