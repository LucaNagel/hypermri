import numpy as np
import pytest
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from hypermri.analysis.dce.conversions import concentration_to_signal
from hypermri.analysis.fitting.api import fit
from hypermri.analysis.fitting.fit_maps import fit_quantitative_maps


def test_fit_results_reconstruct_alias_matches_hp_accessor():
    times_s = np.arange(6, dtype=np.float64) * 5.0
    aif_concentration = np.array([0.0, 0.2, 0.5, 0.3, 0.1, 0.05], dtype=np.float64)
    integral_aif = cumulative_trapezoid(aif_concentration, times_s, initial=0.0)

    slopes = np.array([0.03, 0.015], dtype=np.float64)
    intercepts = np.array([0.12, 0.07], dtype=np.float64)
    tissue_concentration = (
        slopes[:, None] * integral_aif[None, :]
        + intercepts[:, None] * aif_concentration[None, :]
    )

    signal = concentration_to_signal(
        tissue_concentration,
        S_0=100.0,
        T10_s=1.2,
        TR_s=0.05,
        flip_angle_deg=15.0,
        r1=3.5,
    )[:, None, None, :]

    concentration_map = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(signal.shape[0]),
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(signal.shape[-1], dtype=int),
        },
    )

    fit_result = fit(
        concentration_map,
        model="patlak",
        fit_params={
            "aif_signal": aif_concentration,
            "time_points_s": times_s,
            "use_multiprocessing": False,
        },
    )

    reconstructed = fit_result.reconstruct()
    reconstructed_hp = fit_result.hp.reconstruct()

    assert reconstructed.dims == reconstructed_hp.dims
    assert np.allclose(reconstructed.values, reconstructed_hp.values, atol=1e-6)


def test_fit_quantitative_maps_rejects_removed_2cm_fit_type():
    data = xr.DataArray(
        np.ones((1, 1, 1, 4), dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={"repetition_idx": np.arange(4, dtype=int)},
    )

    with pytest.raises(ValueError, match="Unsupported fit_type"):
        fit_quantitative_maps(
            data,
            fit_type="2cm",
            fit_params={
                "aif_signal": np.ones(4, dtype=np.float64),
                "time_points_s": np.arange(4, dtype=np.float64),
                "use_multiprocessing": False,
            },
        )


def test_t1_reconstruct_accepts_voxel_alias_and_maps_parameter_names():
    inversion_times_ms = np.array([120.0, 400.0, 900.0, 1500.0], dtype=np.float64)
    fit_result = xr.Dataset(
        {
            "t1": (
                ("read_idx", "phase_idx", "slice_idx"),
                np.full((2, 2, 2), 1000.0, dtype=np.float64),
            ),
            "A": (
                ("read_idx", "phase_idx", "slice_idx"),
                np.full((2, 2, 2), 2.0, dtype=np.float64),
            ),
            "B": (
                ("read_idx", "phase_idx", "slice_idx"),
                np.full((2, 2, 2), 0.1, dtype=np.float64),
            ),
        },
        coords={
            "read_idx": np.arange(2, dtype=int),
            "phase_idx": np.arange(2, dtype=int),
            "slice_idx": np.arange(2, dtype=int),
            "inversion_idx": inversion_times_ms,
        },
        attrs={
            "model_name": "t1_inversion_recovery",
            "dynamic_dim": "inversion_idx",
        },
    )
    fit_result["inversion_times_ms"] = xr.DataArray(
        inversion_times_ms,
        dims=("inversion_idx",),
        coords={"inversion_idx": inversion_times_ms},
    )

    reconstructed_full = fit_result.reconstruct()
    expected_curve = 2.0 * (1.0 - 2.0 * np.exp(-inversion_times_ms / 1000.0)) + 0.1

    assert reconstructed_full.dims == (
        "read_idx",
        "phase_idx",
        "slice_idx",
        "inversion_idx",
    )
    np.testing.assert_allclose(
        reconstructed_full.isel(read_idx=0, phase_idx=0, slice_idx=0).values,
        expected_curve,
        atol=1e-10,
    )

    voxel = {"read_idx": 1, "phase_idx": 0, "slice_idx": 1}
    reconstructed_voxel_alias = fit_result.reconstruct(voxel=voxel)
    reconstructed_voxel_coords = fit_result.hp.reconstruct(voxel_coords=voxel)

    assert reconstructed_voxel_alias.dims == ("inversion_idx",)
    np.testing.assert_allclose(
        reconstructed_voxel_alias.values,
        reconstructed_voxel_coords.values,
        atol=1e-12,
    )
