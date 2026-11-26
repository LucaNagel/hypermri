"""Characterize the shared equation and current derived-volume clipping for F4.

Inputs are invented. An independently written trapezoid reference checks the
discrete equation at rtol 1e-13, atol 1e-14 mM, allowing floating-point operation
order differences without claiming continuous-time sampling accuracy.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid

from hypermri.analysis.dce.fitting_support import _add_clear_model_aliases
from hypermri.analysis.dce.models.sourbron import sourbron_separable_compartment_model
from hypermri.analysis.dce.models.tofts import extended_tofts_model_2comp


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("plasma_transit_s", [None, 5.0])
def test_sourbron_labelled_equation_matches_scalar_voxels(transpose, plasma_transit_s):
    times = np.array([17, 19, 22, 25, 29, 34, 40, 47, 55, 64, 74, 85], dtype=float)
    plasma = np.exp(-(((times - 29) / 12) ** 2))
    volume = xr.DataArray(
        [[0.1, 0.25], [0.3, 0.15]],
        dims=("read_idx", "phase_idx"),
        coords={"read_idx": [4.0, -4.0], "phase_idx": [3, 1]},
    )
    flow, transit = volume * 0 + 0.03, volume * 0 + 60
    if transpose:
        volume, flow, transit = [value.transpose() for value in (volume, flow, transit)]
    before = volume.copy(deep=True)
    result = sourbron_separable_compartment_model(
        times,
        plasma,
        plasma_volume_fraction=volume,
        tubular_flow_s=flow,
        tubular_transit_time_s=transit,
        plasma_transit_time_s=plasma_transit_s,
    )
    assert result.dims == (*volume.dims, "repetition_idx")
    assert result.dtype == np.float64
    for read in range(2):
        for phase in range(2):
            scalar = sourbron_separable_compartment_model(
                times,
                plasma,
                plasma_volume_fraction=float(
                    volume.isel(read_idx=read, phase_idx=phase)
                ),
                tubular_flow_s=0.03,
                tubular_transit_time_s=60,
                plasma_transit_time_s=plasma_transit_s,
            )
            np.testing.assert_allclose(
                result.isel(read_idx=read, phase_idx=phase),
                scalar,
                rtol=1e-13,
                atol=1e-14,
            )
    xr.testing.assert_identical(volume, before)


def test_f4_keeps_model_specific_default_parameterizations_and_optimizers(monkeypatch):
    from hypermri.analysis.dce.models import sourbron, tofts

    times = np.arange(12, dtype=float) * 4
    plasma = np.ones_like(times)
    tissue = 0.05 + 0.01 * times
    calls = []

    def capture(residuals, *, x0, bounds, **options):
        calls.append((np.asarray(x0), np.asarray(bounds), options))
        residual = residuals(np.asarray(x0))
        return SimpleNamespace(
            x=np.asarray(x0),
            fun=residual,
            cost=float(np.sum(residual**2) / 2),
            success=True,
        )

    monkeypatch.setattr(tofts, "least_squares", capture)
    monkeypatch.setattr(sourbron, "least_squares", capture)
    tofts.fit_single_voxel_tofts_helper(times, tissue, lambda _times: plasma, "2comp")
    sourbron.fit_sourbron_separable_compartment_curve(
        tissue, aif_concentration=plasma, time_s=times
    )
    np.testing.assert_allclose(calls[0][0], [0.01, 0.1, 0.01], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        calls[0][1], [[0, 0, 0], [5, 5, 1]], rtol=1e-13, atol=1e-14
    )
    assert calls[0][2] == {"ftol": 1e-3}
    np.testing.assert_allclose(calls[1][0], [0.1, 0.01, 20], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        calls[1][1], [[0, 0, 2], [1, 0.5, 132]], rtol=1e-13, atol=1e-14
    )
    assert calls[1][2] == {"loss": "linear", "f_scale": 1.0, "max_nfev": 300}


def test_extended_tofts_evaluates_plasma_once_and_retains_storage_precision():
    times = np.array([17, 19, 22, 25, 29, 34], dtype=np.float32)
    plasma = np.array([0.1234567, 0.2, 0.7, 1.0, 0.8, 0.5], dtype=np.float32)
    before = plasma.copy()
    calls = []

    def plasma_at(requested_times):
        calls.append(requested_times)
        return plasma

    result = extended_tofts_model_2comp(times, 0.03, 0.1, 0.25, plasma_at)
    reference = (0.25 * plasma).astype(np.float64)
    time64, plasma64 = times.astype(np.float64), plasma.astype(np.float64)
    reference += 0.03 * np.array(
        [
            trapezoid(
                plasma64[: index + 1] * np.exp(-0.1 * (time - time64[: index + 1])),
                time64[: index + 1],
            )
            for index, time in enumerate(time64)
        ]
    )
    assert len(calls) == 1
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(plasma, before)


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("transit_s", [5.0, 60.0, 300.0])
def test_extended_equation_parameter_mapping_before_consolidation(irregular, transit_s):
    times = (
        np.array([17, 19, 22, 25, 29, 34, 40, 47, 55, 64, 74, 85], dtype=np.float64)
        if irregular
        else np.arange(12, dtype=np.float64) * 4 + 17
    )
    plasma = np.array([0.05, 0.1, 0.4, 0.8, 0.7, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03])
    before = plasma.copy()
    transfer_s, vascular_fraction = 0.03, 0.25
    reference = np.array(
        [
            vascular_fraction * plasma[index]
            + transfer_s
            * trapezoid(
                plasma[: index + 1] * np.exp(-(time - times[: index + 1]) / transit_s),
                times[: index + 1],
            )
            for index, time in enumerate(times)
        ]
    )
    tofts = extended_tofts_model_2comp(
        times, transfer_s, 1 / transit_s, vascular_fraction, lambda _times: plasma
    )
    sourbron = sourbron_separable_compartment_model(
        times,
        plasma,
        plasma_volume_fraction=vascular_fraction,
        tubular_flow_s=transfer_s,
        tubular_transit_time_s=transit_s,
    )
    for result in (tofts, sourbron):
        assert result.shape == times.shape
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(plasma, before)


def test_current_equivalent_volume_aliases_clip_values_above_one():
    flow = xr.DataArray(
        [0.02, 0.03, np.nan], dims="read_idx", coords={"read_idx": [4.0, 0.0, -4.0]}
    )
    transit = xr.DataArray([15.0, 60.0, 40.0], dims="read_idx", coords=flow.coords)
    sourbron = xr.Dataset({"vp": xr.full_like(flow, 0.25), "Ft": flow, "Tt": transit})
    tofts = xr.Dataset(
        {
            "Ktrans_2c": flow,
            "kep_2c": 1 / transit,
            "vp_2c": sourbron["vp"],
            "ve_2c": (flow / (1 / transit)).clip(min=0.0, max=1.0),
        }
    )
    before = sourbron.copy(deep=True)
    for source in (sourbron, tofts):
        result = _add_clear_model_aliases(source)
        np.testing.assert_allclose(
            result["ve_2c"], [0.3, 1.0, np.nan], rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(
            result["ve_extended"], result["ve_2c"], rtol=1e-13, atol=1e-14
        )
        assert result["Ft"].sel(read_idx=0).item() == 0.03
        assert result["Tt"].sel(read_idx=0).item() == 60.0
        assert (
            "ve_equivalent = Ft * Tt"
            in result.attrs["parameterization_equivalence_note"]
        )
    xr.testing.assert_identical(sourbron, before)
