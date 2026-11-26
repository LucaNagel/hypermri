"""Check fitted-signal values and coordinate alignment without running an optimizer."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import reconstruct_fitted_signal


@pytest.fixture
def fitted_fids() -> tuple[xr.Dataset, xr.DataArray]:
    """Provide two known complex FIDs with sample labels distinct from model times."""
    times_s = np.arange(8) * 0.001
    amplitudes = np.array([[1.0 + 1.0j, 0.25], [2.0 - 0.5j, 0.5j]])
    frequencies_hz = np.array([0.0, 125.0])
    t2_s = np.array([0.02, 0.03])
    basis = np.exp(2j * np.pi * frequencies_hz[:, None] * times_s)
    basis *= np.exp(-times_s / t2_s[:, None])
    values = amplitudes @ basis
    result = xr.Dataset(
        {
            "fit_amps": (("voxel", "metabolite"), amplitudes),
            "fit_freqs": (("voxel", "metabolite"), np.tile(frequencies_hz, (2, 1))),
            "fit_t2s": (("voxel", "metabolite"), np.tile(t2_s, (2, 1))),
        },
        coords={"voxel": [4, 9], "metabolite": ["pyruvate", "lactate"]},
        attrs={
            "model_name": "FID",
            "spectral_dim": "sample",
            "sampling_time_axis_s": times_s,
            "signal_domain": "temporal",
        },
    )
    result["fit_amps"].attrs["unit"] = "a.u."
    data = xr.DataArray(
        values.T,
        dims=("sample", "voxel"),
        coords={
            "sample": np.arange(100, 108),
            "voxel": [4, 9],
            "sampling_time_axis_s": ("sample", 12.0 + times_s),
            "position_mm": ("voxel", [0.0, 2.0]),
            "channel_idx": 2,
        },
        attrs={"signal_domain": "temporal", "simulation_source": "known_fids"},
    )
    data.coords["sampling_time_axis_s"].attrs["unit"] = "s"
    return result, data


@pytest.mark.parametrize("domain", ["temporal", "spectral"])
def test_total_signal_matches_reference_values_coordinates_and_domain(
    fitted_fids, domain
):
    """Permit direct subtraction for either representation and a transposed FID axis."""
    result, data = fitted_fids
    if domain == "spectral":
        data = data.copy(data=np.fft.fftshift(np.fft.fft(data.values, axis=0), axes=0))
        data.attrs["signal_domain"] = domain
    original = data.copy(deep=True)
    reconstructed = reconstruct_fitted_signal(result, data)

    assert reconstructed.dims == data.dims
    xr.testing.assert_identical(
        reconstructed.coords.to_dataset(), data.coords.to_dataset()
    )
    np.testing.assert_allclose(
        reconstructed.values, data.values, rtol=1e-12, atol=1e-12
    )
    residual = data - reconstructed
    assert residual.sizes == data.sizes
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    assert reconstructed.attrs["signal_domain"] == domain
    assert reconstructed.attrs["simulation_source"] == "known_fids"
    assert reconstructed.attrs["unit"] == "a.u."
    xr.testing.assert_identical(data, original)


@pytest.mark.parametrize("mismatch", ["length", "voxel", "sampling"])
def test_mismatched_reference_is_rejected(fitted_fids, mismatch):
    """Reject references that would hide lost samples or mislabel fitted values."""
    result, data = fitted_fids
    if mismatch == "length":
        data = data.isel(sample=slice(1, None))
    elif mismatch == "voxel":
        data = data.assign_coords(voxel=[0, 1])
    else:
        data = data.assign_coords(sampling_time_axis_s=("sample", np.arange(8) * 0.002))

    with pytest.raises(ValueError, match="data"):
        reconstruct_fitted_signal(result, data)


def test_invalid_metabolite_is_not_silently_omitted_from_total(fitted_fids):
    """Propagate failed component values instead of reporting a partial sum as a fit."""
    result, data = fitted_fids
    result["fit_amps"].values[0, 0] = np.nan
    reconstructed = reconstruct_fitted_signal(result, data)
    assert np.isnan(reconstructed.sel(voxel=4)).all()
    assert np.isfinite(reconstructed.sel(voxel=9)).all()


def test_legacy_reconstruction_keeps_component_and_time_coordinate_contract(
    fitted_fids,
):
    """Keep the established component-wise accessor usable by existing plots."""
    result, data = fitted_fids
    components = result.reconstruct()
    assert components.dims == ("voxel", "metabolite", "acquisition_idx")
    np.testing.assert_array_equal(
        components.coords["acquisition_idx"], result.attrs["sampling_time_axis_s"]
    )
    np.testing.assert_allclose(components.sum("metabolite").values, data.values.T)
