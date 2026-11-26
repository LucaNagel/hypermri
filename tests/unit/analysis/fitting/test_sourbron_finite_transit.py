"""Finite-Tp integration contracts using invented constant plasma input.

The analytical double-convolution reference is independent of the production
quadrature. An absolute tolerance of 0.1% of peak concentration allows the
approved 0.1 s integration resolution, including acquisition samples between
integration points. Structural comparisons use rtol 1e-12, atol 1e-14 mM.
No scanner data is used.
"""

import numpy as np
import pytest
import xarray as xr
from joblib import parallel_config

from hypermri.analysis.dce.fitting import fit_sourbron_model
from hypermri.analysis.dce.models import sourbron
from hypermri.analysis.dce.models.sourbron import sourbron_separable_compartment_model


@pytest.mark.parametrize("tp_s", [5.0, 60.0])
@pytest.mark.parametrize("offset_s", [0.0, 7.137])
def test_finite_transit_matches_analytical_constant_input(tp_s, offset_s):
    lags = np.array([0.0, 0.817, 2.319, 8.173, 17.147, 43.271, 83.519])
    times = offset_s + lags
    plasma = np.ones_like(times, dtype=np.float32)
    before = plasma.copy()
    tt_s = 60.0
    vp, ft_s = 0.25, 0.03
    plasma_reference = -np.expm1(-lags / tp_s)
    cross = (
        lags * np.exp(-lags / tt_s)
        if tp_s == tt_s
        else (np.exp(-lags / tt_s) - np.exp(-lags / tp_s)) / (1 / tp_s - 1 / tt_s)
    )
    tubular_reference = tt_s * -np.expm1(-lags / tt_s) - cross
    expected = vp * plasma_reference + ft_s * tubular_reference
    actual = sourbron_separable_compartment_model(
        times,
        plasma,
        plasma_volume_fraction=vp,
        tubular_flow_s=ft_s,
        tubular_transit_time_s=tt_s,
        plasma_transit_time_s=tp_s,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=np.max(expected) * 1e-3)
    assert actual.dtype == np.float64
    assert actual[0] == 0
    np.testing.assert_array_equal(plasma, before)


def test_finite_transit_retains_mixed_parameter_dimensions_and_coordinates():
    vp = xr.DataArray([0.2, 0.3], dims="read_idx", coords={"read_idx": [9, -9]})
    ft = xr.DataArray([0.02, 0.03], dims="phase_idx", coords={"phase_idx": [3, 1]})
    tt = xr.DataArray([30.0, 60.0], dims="slice_idx", coords={"slice_idx": [4, 0]})
    tp = xr.DataArray([5.0, 10.0], dims="echo_idx", coords={"echo_idx": [0, 1]})
    inputs = [field.copy(deep=True) for field in (vp, ft, tt, tp)]
    times = np.array([7.0, 9.0, 12.0, 17.0])
    plasma = np.ones_like(times)
    result = sourbron_separable_compartment_model(
        times,
        plasma,
        plasma_volume_fraction=vp,
        tubular_flow_s=ft,
        tubular_transit_time_s=tt,
        plasma_transit_time_s=tp,
    )
    # Characterized previously: vascular dimensions precede tubular dimensions.
    assert result.dims == (
        "read_idx",
        "repetition_idx",
        "echo_idx",
        "phase_idx",
        "slice_idx",
    )
    assert result.dtype == np.float64
    for idx in np.ndindex(2, 2, 2, 2):
        r, p, s, e = idx
        expected = sourbron_separable_compartment_model(
            times,
            plasma,
            plasma_volume_fraction=vp[r].item(),
            tubular_flow_s=ft[p].item(),
            tubular_transit_time_s=tt[s].item(),
            plasma_transit_time_s=tp[e].item(),
        )
        actual = result.isel(read_idx=r, phase_idx=p, slice_idx=s, echo_idx=e)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    for field, before in zip((vp, ft, tt, tp), inputs, strict=True):
        xr.testing.assert_identical(field, before)
        xr.testing.assert_identical(result[field.dims[0]], field[field.dims[0]])
    np.testing.assert_array_equal(result.repetition_idx, times)


@pytest.mark.parametrize("invalid_tp_s", [0.0, -1.0])
def test_finite_transit_retains_invalid_scalar_transit_error(invalid_tp_s):
    with pytest.raises(ValueError, match="must be positive"):
        sourbron_separable_compartment_model(
            np.array([0.0, 1.0, 2.0]),
            np.ones(3),
            plasma_volume_fraction=0.25,
            tubular_flow_s=0.03,
            tubular_transit_time_s=60.0,
            plasma_transit_time_s=invalid_tp_s,
        )


@pytest.mark.parametrize("transpose", [False, True])
def test_finite_transit_maps_are_repeatable_across_serial_and_parallel_execution(
    transpose,
):
    times = np.arange(0.0, 100.0, 2.0)
    plasma = np.exp(-(((times - 20.0) / 8.0) ** 2))
    curve = sourbron_separable_compartment_model(
        times,
        plasma,
        plasma_volume_fraction=0.25,
        tubular_flow_s=0.03,
        tubular_transit_time_s=60.0,
        plasma_transit_time_s=5.0,
    )
    data = xr.DataArray(
        np.stack([curve, curve, np.full_like(curve, np.nan), curve]).reshape(2, 2, -1),
        dims=("read_idx", "phase_idx", "repetition_idx"),
        coords={
            "read_idx": [9.0, -9.0],
            "phase_idx": [3, 1],
            "repetition_idx": np.arange(times.size),
        },
        attrs={"units": "mM", "invented_fixture": True},
    )
    if transpose:
        data = data.transpose("repetition_idx", "phase_idx", "read_idx")
    before = data.copy(deep=True)
    options = {"estimate_tp": True, "fit_start_index": 0, "number_of_cpu_cores": 2}

    def fit(parallel):
        with parallel_config(backend="threading"):
            return fit_sourbron_model(
                data,
                aif_concentration=plasma,
                aif_time_s=times,
                fit_params=options,
                use_multiprocessing=parallel,
            )

    serial = fit(False)
    repeated = fit(False)
    parallel = fit(True)
    xr.testing.assert_identical(serial, repeated)
    xr.testing.assert_allclose(serial, parallel, rtol=1e-12, atol=1e-14)
    assert serial.Tp.dtype == np.float64
    assert serial.fit_success.dtype == bool
    assert serial.n_points_fit.dtype == np.int32
    assert serial.Tp.dims == tuple(dim for dim in data.dims if dim != "repetition_idx")
    unavailable = serial.isel(read_idx=1, phase_idx=0)
    assert not unavailable.fit_success.item()
    assert np.isnan(unavailable.Tp.item())
    assert np.isnan(serial.hp.reconstruct().isel(read_idx=1, phase_idx=0)).all()
    for dim in ("read_idx", "phase_idx"):
        xr.testing.assert_identical(serial[dim], data[dim])
    xr.testing.assert_identical(data, before)
    assert options == {
        "estimate_tp": True,
        "fit_start_index": 0,
        "number_of_cpu_cores": 2,
    }


def test_curve_fit_interpolates_its_arterial_input_once(monkeypatch):
    times = np.arange(0.0, 100.0, 2.0)
    plasma = np.exp(-(((times - 20.0) / 8.0) ** 2))
    curve = sourbron_separable_compartment_model(
        times,
        plasma,
        plasma_volume_fraction=0.25,
        tubular_flow_s=0.03,
        tubular_transit_time_s=60.0,
        plasma_transit_time_s=5.0,
    )
    prepare = sourbron._prepare_finite_transit_input
    calls = []

    def record_preparation(times, plasma):
        calls.append(None)
        return prepare(times, plasma)

    monkeypatch.setattr(sourbron, "_prepare_finite_transit_input", record_preparation)
    result = sourbron.fit_sourbron_separable_compartment_curve(
        curve,
        aif_concentration=plasma,
        time_s=times,
        fit_params={"estimate_tp": True, "fit_start_index": 0},
    )
    assert result["fit_success"]
    assert len(calls) == 1
