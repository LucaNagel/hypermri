from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import hypermri.analysis.fitting as fitting
from hypermri.analysis.dce.models import patlak as patlak_model
from hypermri.analysis.dce.models import sourbron as sourbron_model
from hypermri.analysis.dce.models import tofts as tofts_model
from hypermri.analysis.fitting import api, fit_maps, spectroscopy
from hypermri.analysis.fitting.models.quantitative import adc as adc_model
from hypermri.analysis.fitting.models.quantitative import b1 as b1_model
from hypermri.analysis.fitting.models.quantitative import ideal as ideal_model
from hypermri.analysis.fitting.models.quantitative import t1 as t1_model
from hypermri.analysis.fitting.registry import get_registry


def test_analysis_fitting_import_surfaces_remain_compatible():
    """Keep existing imports as the same functions exported by canonical modules."""
    assert fitting.fit_quantitative_maps is fit_maps.fit_quantitative_maps
    assert fitting.def_fit_params is spectroscopy.def_fit_params
    assert fitting.fit_data_pseudo_inv is spectroscopy.fit_data_pseudo_inv
    assert fitting.reconstruct_fitted_signal is spectroscopy.reconstruct_fitted_signal


@pytest.mark.parametrize("model", ["adc", "b1", "patlak", "sourbron", "t1", "tofts"])
def test_removed_configure_fit_functions_are_absent(model):
    """``prepare_fit_params`` replaces the removed ``configure_*_fit`` helpers."""
    from hypermri import compatibility

    name = f"configure_{model}_fit"
    model_modules = {
        "adc": [adc_model],
        "b1": [b1_model],
        "t1": [t1_model],
        "patlak": [patlak_model],
        "sourbron": [sourbron_model],
        "tofts": [tofts_model],
    }[model]

    assert name not in fitting.__all__
    for module in [fitting, fit_maps, *model_modules]:
        assert not hasattr(module, name)
    assert not hasattr(compatibility, "deprecation_warning")


def test_registry_resolves_public_model_aliases():
    registry = get_registry()

    assert registry.resolve_name("t1") == "t1"
    assert registry.resolve_name("t1_inversion_recovery") == "t1"
    assert registry.resolve_name("monoexponential_adc") == "adc"
    assert registry.resolve_name("FID") == "fid_pseudo_inv"
    assert registry.resolve_name("2c_irf") == "2c_irf"
    with pytest.raises(KeyError):
        registry.resolve_name("renal_2c_irf")


_REMOVED_FIT_MAPS_NAMES = (
    "DB_MSG",
    "DEBUG_FILE",
    "DEBUG_LOG",
    "debug_log_fit_maps",
    "logger",
    "get_registry",
    "two_c_irf_reconstruction",
    "empirical_log_logistic_reconstruction",
    "concentration_to_signal",
    "signal_to_concentration",
    "inversion_recovery_model",
    "get_t1_params_info",
    "prepare_t1_fit_params_dataset",
    "monoexponential_adc_model",
    "get_adc_params_info",
    "prepare_adc_fit_params_dataset",
    "b1_sinusoidal_model",
    "get_b1_params_info",
    "prepare_b1_fit_params_dataset",
    "region_growing_b1_fitting",
    "fit_ideal_dataset",
    "get_ideal_params_info",
    "patlak_model",
    "get_patlak_params_info",
    "prepare_patlak_fit_params_dataset",
    "tofts_model_1comp",
    "extended_tofts_model_2comp",
    "tofts_model_reconstruction",
    "get_tofts_params_info",
    "sourbron_separable_compartment_model",
    "fit_sourbron_separable_compartment_curve",
    "sourbron_model_reconstruction",
    "get_sourbron_params_info",
    "prepare_sourbron_fit_params_dataset",
)


@pytest.mark.parametrize("name", _REMOVED_FIT_MAPS_NAMES)
def test_removed_fit_maps_reexports_are_absent(name):
    """Regression: model functions are imported only from their owner modules."""
    assert name not in fitting.__all__
    assert not hasattr(fitting, name)
    if name != "get_registry":
        # fit_maps uses get_registry internally; it is not re-exported.
        assert not hasattr(fit_maps, name)


def test_fit_maps_keeps_only_its_model_key_entry_points():
    public = {name for name in vars(fit_maps) if not name.startswith("_")}
    assert public - {"annotations", "xr"} == {
        "def_fit_params",
        "fit_quantitative_maps",
        "get_registry",
    }
    assert get_registry is fit_maps.get_registry


def test_central_api_wraps_existing_quantitative_entry_points():
    inversion_times_ms = np.array([50.0, 150.0, 400.0, 900.0], dtype=np.float64)
    t1_true_s = np.array([0.7], dtype=np.float64)[:, None, None, None]
    signal = 100.0 * (
        1.0
        - 2.0 * np.exp(-(inversion_times_ms[None, None, None, :] / 1000.0) / t1_true_s)
    )
    input_data = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "inversion_idx"),
        coords={
            "read_idx": np.array([0], dtype=int),
            "phase_idx": np.array([0], dtype=int),
            "slice_idx": np.array([0], dtype=int),
            "inversion_idx": inversion_times_ms,
        },
    )

    prepared = api.prepare_fit_params(
        data=input_data,
        model="t1",
        fit_params={
            "t1_values_s": np.linspace(0.2, 2.0, 40, dtype=np.float64),
            "use_multiprocessing": False,
        },
    )
    result = api.fit(
        data=input_data,
        model="t1_inversion_recovery",
        fit_params=prepared,
    )
    reconstructed = api.reconstruct_fit(result)

    assert prepared.attrs["fit_type"] == "t1"
    assert result.attrs["model_name"] == "t1_inversion_recovery"
    assert reconstructed.dims == ("read_idx", "phase_idx", "slice_idx", "inversion_idx")


def test_prepare_fit_params_preserves_execution_config():
    """All registry-backed quantitative preparers keep shared worker settings."""
    execution_params = {"use_multiprocessing": False, "number_of_cpu_cores": 2}
    time_points_s = np.arange(4, dtype=np.float64)
    spatial_coords = {
        "read_idx": np.array([0], dtype=int),
        "phase_idx": np.array([0], dtype=int),
    }
    repetition_data = xr.DataArray(
        np.ones((1, 1, 4), dtype=np.float64),
        dims=("read_idx", "phase_idx", "repetition_idx"),
        coords={**spatial_coords, "repetition_idx": time_points_s},
    )

    cases = [
        (
            "t1",
            repetition_data.rename({"repetition_idx": "inversion_idx"}),
            {"t1_values_s": np.linspace(0.2, 2.0, 8, dtype=np.float64)},
        ),
        (
            "adc",
            repetition_data.rename({"repetition_idx": "bvalue_idx"}),
            {"b_values": time_points_s},
        ),
        (
            "patlak",
            repetition_data,
            {"aif_signal": np.ones(4), "time_points_s": time_points_s},
        ),
        (
            "tofts",
            repetition_data,
            {
                "aif_signal": np.ones(4),
                "aif_times": time_points_s,
                "time_points_s": time_points_s,
            },
        ),
        (
            "sourbron",
            repetition_data,
            {"aif_signal": np.ones(4), "time_points_s": time_points_s},
        ),
        (
            "b1",
            repetition_data.rename({"repetition_idx": "rfpower_idx"}).assign_coords(
                rfpower_idx=np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
            ),
            {"fa_deg": 90, "trf_ms": 1.0, "sint": 1.0},
        ),
        (
            "ideal",
            repetition_data.rename({"repetition_idx": "echo_idx"}),
            {"relative_frequencies_hz": np.array([0.0]), "echo_spacing_s": 0.001},
        ),
    ]

    for model, input_data, fit_params in cases:
        prepared = api.prepare_fit_params(
            data=input_data,
            model=model,
            fit_params={**fit_params, **execution_params},
        )
        assert prepared.attrs["use_multiprocessing"] is False
        assert prepared.attrs["number_of_cpu_cores"] == 2


def test_prepare_ideal_params_accepts_lowercase_hz_alias():
    input_data = xr.DataArray(
        np.ones((1, 3), dtype=np.complex128),
        dims=("read_idx", "echo_idx"),
    )
    frequencies_hz = np.array([0.0, 120.0], dtype=np.float64)

    prepared = api.prepare_fit_params(
        data=input_data,
        model="ideal",
        fit_params={
            "relative_frequencies_hz": frequencies_hz,
            "echo_spacing_s": 0.001,
        },
    )

    np.testing.assert_array_equal(
        prepared.attrs["relative_frequencies_Hz"], frequencies_hz
    )

    direct_prepared = ideal_model.prepare_ideal_fit_params_dataset(
        input_data,
        {
            "relative_frequencies_hz": frequencies_hz,
            "echo_spacing_s": 0.001,
        },
    )
    np.testing.assert_array_equal(
        direct_prepared.attrs["relative_frequencies_Hz"], frequencies_hz
    )


def test_b1_region_growing_runs_once_without_registry_recursion(monkeypatch):
    input_data = xr.DataArray(
        np.ones((2, 1, 1, 4), dtype=np.complex128),
        dims=("read_idx", "phase_idx", "slice_idx", "rfpower_idx"),
        coords={
            "read_idx": np.arange(2, dtype=int),
            "phase_idx": np.array([0], dtype=int),
            "slice_idx": np.array([0], dtype=int),
            "rfpower_idx": np.array([1.0, 4.0, 9.0, 16.0]),
        },
    )
    fit_params = {
        "fa_deg": 90.0,
        "trf_ms": 1.0,
        "sint": 1.0,
        "seed_voxel": {"read_idx": 0, "phase_idx": 0},
        "period_samples": 10,
        "use_region_growing": True,
        "use_multiprocessing": False,
    }
    region_calls = 0

    def fake_region_growing(data, fit_params):
        nonlocal region_calls
        region_calls += 1
        return {
            (read_idx, 0): {
                "amplitude": complex(read_idx + 1),
                "period": 5.0,
                "refpower_90deg_W": 6.25,
                "rss_b1": 0.0,
                "stop_index": 3.0,
                "n_valid_samples": 4.0,
            }
            for read_idx in range(2)
        }

    monkeypatch.setattr(b1_model, "region_growing_b1_fitting", fake_region_growing)
    monkeypatch.setattr(
        fit_maps,
        "fit_quantitative_maps",
        lambda *args, **kwargs: pytest.fail("recursive compatibility call"),
    )

    result = api.fit(data=input_data, model="b1", fit_params=fit_params)

    assert region_calls == 1
    assert result.attrs["model_name"] == "b1_sinusoidal"
    assert result["amplitude"].dims == ("read_idx", "phase_idx", "slice_idx")
    np.testing.assert_array_equal(result["amplitude"].values[:, 0, 0], [1.0, 2.0])


def _synthetic_b1_volume(n_read=4, n_phase=5):
    """Noise-free sinusoidal B1 sweep; period 4-6 sqrt(W), amplitude 1-2."""
    refpower_w = np.linspace(0.25, 9.0, 8)
    sqrt_power = np.sqrt(refpower_w)
    period = 4.0 + 0.1 * np.add.outer(np.arange(n_read), np.arange(n_phase))
    amplitude = 1.0 + 0.05 * np.add.outer(np.arange(n_read), np.arange(n_phase))
    signal = amplitude[..., None] * np.sin(np.pi / period[..., None] * sqrt_power)
    return xr.DataArray(
        signal[:, :, None, :].astype(np.complex128),
        dims=("read_idx", "phase_idx", "slice_idx", "rfpower_idx"),
        coords={
            "read_idx": np.arange(n_read),
            "phase_idx": np.arange(n_phase),
            "slice_idx": [0],
            "rfpower_idx": refpower_w,
        },
    )


@pytest.mark.parametrize("priority", ["fifo", "snr"])
def test_b1_region_growing_fit_order_map(priority):
    input_data = _synthetic_b1_volume()
    seed = {"read_idx": 1, "phase_idx": 2, "slice_idx": 0}
    result = api.fit(
        data=input_data,
        model="b1",
        fit_params={
            "fa_deg": 90.0,
            "trf_ms": 1.0,
            "sint": 1.0,
            "seed_voxel": seed,
            "period_samples": 200,
            "use_region_growing": True,
            "region_growing_priority": priority,
            "return_debug_info": True,
            "use_multiprocessing": False,
        },
    )

    order = result["fit_order_map"]
    assert order.dims == ("read_idx", "phase_idx")
    np.testing.assert_array_equal(order["read_idx"], input_data["read_idx"])
    np.testing.assert_array_equal(order["phase_idx"], input_data["phase_idx"])

    values = order.values
    n_voxels = values.size
    np.testing.assert_array_equal(np.sort(values.ravel()), np.arange(n_voxels))
    assert values[seed["read_idx"], seed["phase_idx"]] == 0

    for parameter in b1_model.OUTPUT_PARAM_NAMES:
        seed_first_iteration = result[f"{parameter}_iter1"].isel(
            read_idx=seed["read_idx"], phase_idx=seed["phase_idx"]
        )
        assert np.isfinite(seed_first_iteration.item())

    # Each voxel is fitted only after at least one 4-connected neighbour.
    for read_idx, phase_idx in np.ndindex(values.shape):
        if values[read_idx, phase_idx] == 0:
            continue
        earlier_neighbours = [
            values[r, p]
            for r, p in (
                (read_idx - 1, phase_idx),
                (read_idx + 1, phase_idx),
                (read_idx, phase_idx - 1),
                (read_idx, phase_idx + 1),
            )
            if 0 <= r < values.shape[0] and 0 <= p < values.shape[1]
        ]
        assert min(earlier_neighbours) < values[read_idx, phase_idx]

    if priority == "fifo":
        distance = np.abs(np.arange(values.shape[0])[:, None] - seed["read_idx"]) + (
            np.abs(np.arange(values.shape[1])[None, :] - seed["phase_idx"])
        )
        flat = np.argsort(values.ravel())
        assert np.all(np.diff(distance.ravel()[flat]) >= 0)


def test_b1_region_growing_fit_order_map_requires_debug_info():
    result = api.fit(
        data=_synthetic_b1_volume(3, 3),
        model="b1",
        fit_params={
            "fa_deg": 90.0,
            "trf_ms": 1.0,
            "sint": 1.0,
            "seed_voxel": {"read_idx": 0, "phase_idx": 0, "slice_idx": 0},
            "period_samples": 100,
            "use_region_growing": True,
            "use_multiprocessing": False,
        },
    )
    assert "fit_order_map" not in result


def test_b1_region_growing_rejects_non_singleton_extra_dimensions():
    input_data = xr.DataArray(
        np.ones((2, 1, 2, 4), dtype=np.complex128),
        dims=("read_idx", "phase_idx", "slice_idx", "rfpower_idx"),
    )

    with pytest.raises(ValueError, match="two-dimensional"):
        b1_model.fit_b1_region_growing_dataset(
            input_data,
            {"dynamic_dim": "rfpower_idx"},
        )


def test_b1_fit_preserves_public_metadata_and_reconstruction(monkeypatch):
    input_data = xr.DataArray(
        np.ones((2, 1, 1, 4), dtype=np.complex128),
        dims=("read_idx", "phase_idx", "slice_idx", "rfpower_idx"),
        coords={
            "read_idx": np.arange(2, dtype=int),
            "phase_idx": np.array([0], dtype=int),
            "slice_idx": np.array([0], dtype=int),
            "rfpower_idx": np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64),
        },
    )

    result = fit_maps.fit_quantitative_maps(
        data=input_data,
        fit_type="b1",
        fit_params={
            "fa_deg": 90.0,
            "trf_ms": 1.0,
            "sint": 1.0,
            "period_grid": np.array([2.0, 5.0, 8.0], dtype=np.float64),
            "use_multiprocessing": False,
        },
    )

    assert result.attrs["model_name"] == "b1_sinusoidal"
    assert result.attrs["dynamic_dim"] == "rfpower_idx"
    assert result.attrs["refpower_W"] == [1.0, 4.0, 9.0, 16.0]
    assert set(result.data_vars) >= {
        "amplitude",
        "period",
        "refpower_90deg_W",
        "rss_b1",
        "stop_index",
        "n_valid_samples",
    }

    reconstructed = result.hp.reconstruct()
    assert reconstructed.dims == ("read_idx", "phase_idx", "slice_idx", "rfpower_idx")
    np.testing.assert_allclose(
        reconstructed.coords["rfpower_idx"], [1.0, 4.0, 9.0, 16.0]
    )


def test_spectroscopy_fit_preserves_public_metadata_and_reconstruction(monkeypatch):
    input_data = xr.DataArray(
        np.ones((2, 1, 8), dtype=np.complex64),
        dims=("read_idx", "phase_idx", "acquisition_idx"),
        coords={
            "read_idx": np.arange(2, dtype=int),
            "phase_idx": np.array([0], dtype=int),
            "acquisition_idx": np.arange(8, dtype=int),
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8, dtype=np.float64) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )

    fit_params = spectroscopy.def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "rep_fitting": 2,
            "niter": 1,
            "npoints": 3,
        },
    )

    def fake_fit_single(
        data_slice,
        initial_freq_hz,
        initial_t2_s,
        freq_search_range_hz,
        t2_search_range_s,
        min_t2_s,
        max_t2_s,
        freq_search_range_lower_bound_hz,
        freq_search_range_upper_bound_hz,
        skip_fitting_mask=None,
        db=False,
        **kwargs,
    ):
        del (
            data_slice,
            freq_search_range_hz,
            t2_search_range_s,
            min_t2_s,
            max_t2_s,
            freq_search_range_lower_bound_hz,
            freq_search_range_upper_bound_hz,
            skip_fitting_mask,
            db,
        )
        rep_fitting = kwargs["rep_fitting"]
        niter = kwargs["niter"]
        npoints = kwargs["npoints"]
        n_metabs = len(initial_freq_hz)
        return (
            np.full(n_metabs, 2.0 + 1.0j, dtype=np.complex128),
            initial_freq_hz + 1.5,
            initial_t2_s + 0.005,
            np.zeros((n_metabs, rep_fitting, niter, npoints, 3), dtype=np.float64),
            np.zeros((n_metabs, rep_fitting, niter, npoints, 3), dtype=np.float64),
            np.zeros(n_metabs, dtype=bool),
        )

    monkeypatch.setattr(
        spectroscopy,
        "_fit_single_spectrum_pseudo_inv",
        fake_fit_single,
    )

    result = spectroscopy.fit_data_pseudo_inv(
        data=input_data,
        fit_params=fit_params,
        show_tqdm=False,
    )

    assert result.attrs["model_name"] == "FID"
    assert result.attrs["spectral_dim"] == "acquisition_idx"
    assert result.attrs["signal_domain"] == "temporal"
    assert set(result.data_vars) >= {
        "fit_amps",
        "fit_freqs",
        "fit_t2s",
        "fit_fwhm_hz",
        "fit_r2_freq",
        "fit_r2_t2",
        "fit_low_snr_refitted",
    }

    reconstructed = result.reconstruct()
    reconstructed_hp = result.hp.reconstruct()

    assert reconstructed.dims == (
        "read_idx",
        "phase_idx",
        "metabolite",
        "acquisition_idx",
    )
    assert reconstructed.dims == reconstructed_hp.dims
    np.testing.assert_allclose(reconstructed.values, reconstructed_hp.values)


def test_central_api_wraps_existing_spectroscopy_entry_points(monkeypatch):
    input_data = xr.DataArray(
        np.ones((1, 8), dtype=np.complex64),
        dims=("read_idx", "acquisition_idx"),
        coords={
            "read_idx": np.array([0], dtype=int),
            "acquisition_idx": np.arange(8, dtype=int),
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8, dtype=np.float64) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )

    prepared = api.prepare_fit_params(
        data=input_data,
        model="fid",
        fit_params={"metabs": ["pyruvate"], "rep_fitting": 2, "niter": 1, "npoints": 3},
    )

    def fake_fit_single(*args, **kwargs):
        initial_freq_hz = args[1]
        initial_t2_s = args[2]
        rep_fitting = kwargs["rep_fitting"]
        niter = kwargs["niter"]
        npoints = kwargs["npoints"]
        n_metabs = len(initial_freq_hz)
        return (
            np.full(n_metabs, 1.0 + 0.0j, dtype=np.complex128),
            initial_freq_hz,
            initial_t2_s,
            np.zeros((n_metabs, rep_fitting, niter, npoints, 3), dtype=np.float64),
            np.zeros((n_metabs, rep_fitting, niter, npoints, 3), dtype=np.float64),
            np.zeros(n_metabs, dtype=bool),
        )

    monkeypatch.setattr(
        spectroscopy,
        "_fit_single_spectrum_pseudo_inv",
        fake_fit_single,
    )

    result = api.fit(
        data=input_data,
        model="FID",
        fit_params=prepared,
    )

    assert prepared.attrs["spectral_dim"] == "acquisition_idx"
    assert result.attrs["model_name"] == "FID"
