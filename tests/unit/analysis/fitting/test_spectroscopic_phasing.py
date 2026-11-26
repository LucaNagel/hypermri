import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import phase_fit_results


def test_phase_fit_results_rotates_each_complex_amplitude_to_real():
    amplitudes = xr.DataArray(
        np.array([[1j, -2.0], [1.0 + 1.0j, -3.0j]], dtype=np.complex128),
        dims=["metabolite", "voxel"],
        coords={"metabolite": ["A", "B"], "voxel": [0, 1]},
    )
    fit_results = xr.Dataset({"fit_amps": amplitudes})

    phased = phase_fit_results(fit_results)

    np.testing.assert_allclose(
        np.real(phased["fit_amps"].values),
        np.abs(amplitudes.values),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.imag(phased["fit_amps"].values),
        0.0,
        atol=1e-12,
    )
    assert "fit_phase_correction_rad" in phased
    assert phased.attrs["phased_to_real"] is True
    assert np.iscomplexobj(fit_results["fit_amps"].values)


def test_phase_fit_results_can_apply_shared_metabolite_phase():
    amplitudes = xr.DataArray(
        np.array([[1j, 2j], [1.0 + 1.0j, 2.0 + 2.0j]], dtype=np.complex128),
        dims=["metabolite", "voxel"],
        coords={"metabolite": ["A", "B"], "voxel": [0, 1]},
    )
    fit_results = xr.Dataset({"fit_amps": amplitudes})

    phased = phase_fit_results(fit_results, reduce_dims="voxel")

    assert phased["fit_phase_correction_rad"].dims == ("metabolite",)
    summed = phased["fit_amps"].sum("voxel")
    np.testing.assert_allclose(np.imag(summed.values), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.real(summed.values),
        np.abs(amplitudes.sum("voxel").values),
        rtol=1e-12,
        atol=1e-12,
    )


def test_phase_fit_results_rejects_real_amplitudes():
    fit_results = xr.Dataset(
        {"fit_amps": (["metabolite"], np.array([1.0, 2.0], dtype=float))}
    )

    with pytest.raises(ValueError, match="complex-valued"):
        phase_fit_results(fit_results)
