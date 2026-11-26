from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy as spectroscopy_fitting
from hypermri.analysis.fitting.models.spectroscopy import fid_model as fid_pseudo_inv
from hypermri.analysis.fitting.spectroscopy import def_fit_params
from hypermri.analysis.spectroscopy import pipeline as spectroscopy_pipeline
from hypermri.analysis.spectroscopy.metabolites import get_metab_cs_ppm


def _patch_grid_searches(monkeypatch, frequency, decay):
    """Route the alternating search's grid steps to fake 4-output searches.

    ``frequency`` and ``decay`` take ``(data, options)`` like the public
    ``fit_freq_pseudo_inv``/``fit_t2_pseudo_inv``; the adapter appends the
    identity line permutation that ``_fit_parameter_grid`` also returns.
    """

    def grid(data, options, *, parameter):
        search = frequency if parameter == "frequency" else decay
        return (*search(data, options), np.arange(len(options["metabs"])))

    monkeypatch.setattr(fid_pseudo_inv, "_fit_parameter_grid", grid)


def _append_fake_metabolite(
    fit_params: xr.Dataset,
    metabolite: str,
) -> xr.Dataset:
    expanded = fit_params.copy(deep=True)
    for var_name, data_array in fit_params.data_vars.items():
        if "metabolite" not in data_array.dims:
            continue
        template = (
            xr.zeros_like(data_array.isel(metabolite=0, drop=True))
            if data_array.dtype != bool
            else xr.zeros_like(
                data_array.isel(metabolite=0, drop=True),
                dtype=bool,
            )
        )
        template = template.expand_dims(metabolite=[metabolite])
        expanded[var_name] = xr.concat([expanded[var_name], template], dim="metabolite")
    expanded.attrs["metabs"] = spectroscopy_pipeline._get_dataset_metabolites(expanded)
    return expanded


def test_run_csi_fitting_pipeline_chains_metabolite_additions():
    input_data = xr.DataArray(
        np.ones((8, 1, 1, 1), dtype=np.complex64),
        dims=["acquisition_idx", "read_idx", "phase_idx", "repetition_idx"],
        coords={
            "acquisition_idx": np.arange(8),
            "read_idx": [0],
            "phase_idx": [0],
            "repetition_idx": [0],
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8) * 1e-3,
            "freq_axis_hz": np.linspace(-100.0, 100.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
            "bw_hz": 200.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate", "lactate", "pyruvatehydrate"],
            "metabs_freq_ranges": [150.0, 30.0, 30.0],
            "metabs_t2_ranges": [0.04, 0.01, 0.01],
            "initial_t2s_s": [0.02, 0.03, 0.04],
            "min_t2_s": [0.01, 0.01, 0.01],
            "max_t2_s": [0.04, 0.04, 0.04],
        },
    )

    fit_calls: list[list[str]] = []
    add_calls: list[dict[str, object]] = []
    initialise_calls: list[dict[str, object]] = []

    def fake_fit_data_pseudo_inv(
        data: xr.DataArray,
        fit_params: xr.Dataset,
        show_tqdm: bool | None = None,
    ) -> xr.Dataset:
        del data, show_tqdm
        fit_calls.append(spectroscopy_pipeline._get_dataset_metabolites(fit_params))
        return xr.Dataset(attrs={"call_index": len(fit_calls)})

    def fake_add_metab_to_fit_params(**kwargs) -> xr.Dataset:
        add_calls.append(kwargs)
        existing_metabs = spectroscopy_pipeline._get_dataset_metabolites(
            kwargs["fit_params"]
        )
        return spectroscopy_pipeline._subset_fit_params_dataset(
            fit_params,
            existing_metabs + [kwargs["new_metab_name"]],
        )

    def fake_initialise_fit_params_from_results(**kwargs) -> xr.Dataset:
        initialise_calls.append(kwargs)
        refined = kwargs["fit_params"].copy(deep=True)
        refined.attrs["refined"] = True
        return refined

    original_fit = spectroscopy_pipeline.fit_data_pseudo_inv
    original_add = spectroscopy_pipeline.add_metab_to_fit_params
    original_initialise = spectroscopy_pipeline.initialise_fit_params_from_results
    try:
        spectroscopy_pipeline.fit_data_pseudo_inv = fake_fit_data_pseudo_inv
        spectroscopy_pipeline.add_metab_to_fit_params = fake_add_metab_to_fit_params
        spectroscopy_pipeline.initialise_fit_params_from_results = (
            fake_initialise_fit_params_from_results
        )

        result = spectroscopy_pipeline.run_csi_fitting_pipeline(
            data=input_data,
            fit_params=fit_params,
            initial_metabolite="pyruvate",
            metabolites_to_add=["lactate", "pyruvatehydrate"],
            freeze_other_freqs=[True, False],
            freeze_other_t2s=False,
            final_freq_ranges_hz=5.0,
            final_t2_ranges_s=0.01,
        )
    finally:
        spectroscopy_pipeline.fit_data_pseudo_inv = original_fit
        spectroscopy_pipeline.add_metab_to_fit_params = original_add
        spectroscopy_pipeline.initialise_fit_params_from_results = original_initialise

    assert fit_calls == [
        ["pyruvate"],
        ["pyruvate", "lactate", "pyruvatehydrate"],
        ["pyruvate", "lactate", "pyruvatehydrate"],
    ]

    assert [call["new_metab_name"] for call in add_calls] == [
        "lactate",
        "pyruvatehydrate",
    ]
    assert [call["freeze_other_freqs"] for call in add_calls] == [True, False]
    assert [call["freeze_other_t2s"] for call in add_calls] == [False, False]
    assert add_calls[0]["new_freq_range"] == 30.0
    assert add_calls[1]["new_t2_s"] == 0.04
    assert add_calls[0]["freq_offset"] == (
        get_metab_cs_ppm("lactate") - get_metab_cs_ppm("pyruvate")
    )
    assert add_calls[1]["freq_offset"] == (
        get_metab_cs_ppm("pyruvatehydrate") - get_metab_cs_ppm("pyruvate")
    )

    assert len(initialise_calls) == 1
    assert initialise_calls[0]["freq_ranges_hz"] == 5.0
    assert initialise_calls[0]["t2_ranges_s"] == 0.01

    assert result["fit_params"].attrs["refined"] is True
    assert result["fit_results"].attrs["call_index"] == 3


def test_fit_data_pseudo_inv_show_tqdm_runs_voxel_loop(monkeypatch):
    input_data = xr.DataArray(
        np.ones((2, 2, 16), dtype=np.complex64),
        dims=["read_idx", "phase_idx", "acquisition_idx"],
        coords={
            "read_idx": [0, 1],
            "phase_idx": [0, 1],
            "acquisition_idx": np.arange(16),
            "x": (
                ("read_idx", "phase_idx"),
                np.array([[-1.0, -1.0], [1.0, 1.0]]),
            ),
            "y": (
                ("read_idx", "phase_idx"),
                np.array([[-2.0, 2.0], [-2.0, 2.0]]),
            ),
            "z": (
                ("read_idx", "phase_idx"),
                np.zeros((2, 2), dtype=float),
            ),
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "rep_fitting": 2,
            "niter": 1,
            "npoints": 3,
            "min_t2_s": [0.01],
            "max_t2_s": [0.05],
        },
    )

    class FakeTqdm:
        instances: list["FakeTqdm"] = []

        def __init__(self, total: int, desc: str, leave: bool):
            self.total = total
            self.desc = desc
            self.leave = leave
            self.updated = 0
            type(self).instances.append(self)

        def __enter__(self) -> "FakeTqdm":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def update(self, value: int) -> None:
            self.updated += value

    def fake_fit_single(
        data_slice: np.ndarray,
        initial_freqs_hz: np.ndarray,
        initial_t2s_s: np.ndarray,
        metabs_freq_ranges: np.ndarray,
        metabs_t2_ranges: np.ndarray,
        min_t2_s: np.ndarray,
        max_t2_s: np.ndarray,
        freq_range_lower_bound_hz: np.ndarray,
        freq_range_upper_bound_hz: np.ndarray,
        skip_fitting_mask: np.ndarray | None = None,
        db: bool = False,
        **kwargs,
    ) -> tuple:
        del (
            data_slice,
            metabs_freq_ranges,
            metabs_t2_ranges,
            min_t2_s,
            max_t2_s,
            freq_range_lower_bound_hz,
            freq_range_upper_bound_hz,
            skip_fitting_mask,
            db,
        )
        n_metabs = len(initial_freqs_hz)
        rep_fitting = kwargs["rep_fitting"]
        niter = kwargs["niter"]
        npoints = kwargs["npoints"]
        return (
            np.full(n_metabs, 2.0 + 1.0j, dtype=np.complex128),
            initial_freqs_hz + 1.0,
            initial_t2s_s + 0.01,
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 7.0),
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 8.0),
            np.zeros(n_metabs, dtype=bool),
        )

    monkeypatch.setattr(
        "hypermri.analysis.fitting.spectroscopy._fit_single_spectrum_pseudo_inv",
        fake_fit_single,
    )
    monkeypatch.setattr("tqdm.auto.tqdm", FakeTqdm)

    result = spectroscopy_fitting.fit_data_pseudo_inv(
        data=input_data,
        fit_params=fit_params,
        show_tqdm=True,
    )

    assert len(FakeTqdm.instances) == 1
    assert FakeTqdm.instances[0].total == 4
    assert FakeTqdm.instances[0].updated == 4
    assert result["fit_freqs"].shape == (2, 2, 1)
    assert np.allclose(
        result["fit_freqs"].values,
        fit_params["initial_freqs_hz"].values + 1.0,
    )
    assert np.allclose(
        result["fit_t2s"].values,
        fit_params["initial_t2s_s"].values + 0.01,
    )
    assert result["fit_r2_freq"].shape[-4:] == (2, 1, 3, 3)
    for coord in ("x", "y", "z"):
        xr.testing.assert_equal(result.coords[coord], input_data.coords[coord])


@pytest.mark.parametrize(
    ("n_jobs", "show_tqdm"),
    [(1, True), (2, False)],
    ids=["serial-progress", "parallel-threading"],
)
def test_fit_data_pseudo_inv_execution_modes_match_serial(
    monkeypatch, n_jobs, show_tqdm
):
    """Keep voxel-dependent parameters and metadata aligned in every eager mode."""
    data_values = np.arange(2 * 3 * 16, dtype=np.float64).reshape(2, 3, 16)
    input_data = xr.DataArray(
        data_values + 1j * data_values[::-1],
        dims=["snr", "noise_realization", "acquisition_idx"],
        coords={
            "snr": [50.0, 100.0],
            "noise_realization": [10, 11, 12],
            "acquisition_idx": np.arange(16),
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate", "bicarbonate"],
            "rep_fitting": 2,
            "niter": 1,
            "npoints": 3,
            "min_t2_s": [0.01, 0.01],
            "max_t2_s": [0.05, 0.05],
        },
    )
    voxel_offset = input_data.real.isel(acquisition_idx=0, drop=True)
    fit_params["initial_freqs_hz"] = fit_params["initial_freqs_hz"] + voxel_offset
    fit_params["initial_t2s_s"] = fit_params["initial_t2s_s"] + voxel_offset * 1e-5
    fit_params = fid_pseudo_inv._update_frequency_bounds(fit_params)
    # Keep parameter maps in their original order while moving both input axes.
    input_data = input_data.transpose("noise_realization", "acquisition_idx", "snr")
    original_data = input_data.copy(deep=True)
    original_params = fit_params.copy(deep=True)

    def fake_fit_single(
        data_slice: np.ndarray,
        initial_freqs_hz: np.ndarray,
        initial_t2s_s: np.ndarray,
        metabs_freq_ranges: np.ndarray,
        metabs_t2_ranges: np.ndarray,
        min_t2_s: np.ndarray,
        max_t2_s: np.ndarray,
        freq_range_lower_bound_hz: np.ndarray,
        freq_range_upper_bound_hz: np.ndarray,
        skip_fitting_mask: np.ndarray | None = None,
        **kwargs,
    ) -> tuple:
        del (
            metabs_freq_ranges,
            metabs_t2_ranges,
            min_t2_s,
            max_t2_s,
            freq_range_lower_bound_hz,
            freq_range_upper_bound_hz,
            skip_fitting_mask,
        )
        n_metabs = len(initial_freqs_hz)
        rep_fitting = kwargs["rep_fitting"]
        niter = kwargs["niter"]
        npoints = kwargs["npoints"]
        signal_marker = float(np.real(data_slice[0]))
        r2_shape = (n_metabs, rep_fitting, niter, npoints, 3)
        return (
            signal_marker + 1j * np.arange(n_metabs),
            initial_freqs_hz + signal_marker,
            initial_t2s_s + signal_marker * 1e-4,
            np.full(r2_shape, signal_marker),
            np.full(r2_shape, signal_marker + 1.0),
            np.arange(n_metabs) % 2 == 0,
        )

    monkeypatch.setattr(
        "hypermri.analysis.fitting.spectroscopy._fit_single_spectrum_pseudo_inv",
        fake_fit_single,
    )

    serial_result = spectroscopy_fitting.fit_data_pseudo_inv(
        data=input_data,
        fit_params=fit_params,
        n_jobs=1,
    )
    result = spectroscopy_fitting.fit_data_pseudo_inv(
        data=input_data,
        fit_params=fit_params,
        n_jobs=n_jobs,
        batch_size=2,
        parallel_backend="threading",
        show_tqdm=show_tqdm,
    )

    xr.testing.assert_identical(result, serial_result)
    assert result["fit_amps"].dims == (
        "noise_realization",
        "snr",
        "metabolite",
    )
    expected_freqs = (fit_params["initial_freqs_hz"] + voxel_offset).transpose(
        "noise_realization", "snr", "metabolite"
    )
    xr.testing.assert_allclose(result["fit_freqs"], expected_freqs)
    xr.testing.assert_identical(input_data, original_data)
    xr.testing.assert_identical(fit_params, original_params)


def test_fit_data_pseudo_inv_rejects_invalid_joblib_options():
    input_data = xr.DataArray(
        np.ones((2, 8), dtype=np.complex128),
        dims=["noise_realization", "acquisition_idx"],
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={"metabs": ["pyruvate"]},
    )

    with pytest.raises(ValueError, match="n_jobs=0"):
        spectroscopy_fitting.fit_data_pseudo_inv(
            data=input_data,
            fit_params=fit_params,
            n_jobs=0,
        )

    with pytest.raises(ValueError, match="batch_size"):
        spectroscopy_fitting.fit_data_pseudo_inv(
            data=input_data,
            fit_params=fit_params,
            n_jobs=2,
            batch_size=0,
        )


def test_def_fit_params_produces_real_numeric_maps():
    input_data = xr.DataArray(
        np.ones((2, 16), dtype=np.complex64),
        dims=["read_idx", "acquisition_idx"],
        coords={
            "read_idx": [0, 1],
            "acquisition_idx": np.arange(16),
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "skip_fitting": [True],
            "min_t2_s": [0.01],
            "max_t2_s": [0.05],
        },
    )

    numeric_vars = [
        "initial_freqs_hz",
        "initial_t2s_s",
        "metabs_freq_ranges",
        "metabs_t2_ranges",
        "min_t2_s",
        "max_t2_s",
        "freq_range_lower_bound_hz",
        "freq_range_upper_bound_hz",
    ]
    for name in numeric_vars:
        assert np.issubdtype(fit_params[name].dtype, np.floating)
        assert not np.iscomplexobj(fit_params[name].values)

    assert fit_params["skip_fitting"].dtype == bool
    assert bool(np.all(fit_params["skip_fitting"].values))


def test_def_fit_params_uses_attr_sampling_axis_for_spectral_domain():
    input_data = xr.DataArray(
        np.ones((8,), dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(8)},
        attrs={
            "signal_domain": "spectral",
            "sampling_time_axis_s": np.arange(10, dtype=np.float64) * 1e-3,
            "freq_axis_hz": np.linspace(-500.0, 500.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
        },
    )

    fit_params = def_fit_params(
        data=input_data,
        fit_params={"metabs": ["pyruvate"]},
    )

    sampling_time_axis_s = np.asarray(fit_params.attrs["sampling_time_axis_s"])
    assert sampling_time_axis_s.shape == (8,)
    assert np.isclose(sampling_time_axis_s[0], 2e-3)
    assert np.isclose(sampling_time_axis_s[-1], 9e-3)
    assert np.isclose(fit_params.attrs["dwell_time_s"], 1e-3)
    assert int(fit_params.attrs["cut_off"]) == 2


def test_def_fit_params_respects_nsamplepoints_override():
    input_data = xr.DataArray(
        np.ones((16,), dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(16)},
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16, dtype=np.float64) * 1e-3,
            "freq_axis_hz": np.linspace(-500.0, 500.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
        },
    )

    fit_params = def_fit_params(
        data=input_data,
        fit_params={"metabs": ["pyruvate"], "nsamplepoints": 8},
    )

    sampling_time_axis_s = np.asarray(fit_params.attrs["sampling_time_axis_s"])
    assert sampling_time_axis_s.shape == (8,)
    assert int(fit_params.attrs["nsamplepoints"]) == 8


def test_def_fit_params_defaults_to_5_hz_min_peak_separation():
    input_data = xr.DataArray(
        np.ones((16,), dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(16)},
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16, dtype=np.float64) * 1e-3,
            "freq_axis_hz": np.linspace(-500.0, 500.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
        },
    )

    fit_params = def_fit_params(
        data=input_data,
        fit_params={"metabs": ["pyruvate"]},
    )

    assert np.isclose(fit_params.attrs["min_peak_separation_hz"], 5.0)


def test_fit_data_pseudo_inv_trims_input_to_nsamplepoints(monkeypatch):
    input_data = xr.DataArray(
        np.ones((16,), dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(16)},
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16, dtype=np.float64) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "rep_fitting": 2,
            "niter": 1,
            "npoints": 3,
            "nsamplepoints": 8,
        },
    )

    seen: dict[str, int] = {}

    def fake_fit_single(
        data_slice: np.ndarray,
        initial_freqs_hz: np.ndarray,
        initial_t2s_s: np.ndarray,
        metabs_freq_ranges: np.ndarray,
        metabs_t2_ranges: np.ndarray,
        min_t2_s: np.ndarray,
        max_t2_s: np.ndarray,
        freq_range_lower_bound_hz: np.ndarray,
        freq_range_upper_bound_hz: np.ndarray,
        skip_fitting_mask: np.ndarray | None = None,
        db: bool = False,
        **kwargs,
    ) -> tuple:
        del (
            metabs_freq_ranges,
            metabs_t2_ranges,
            min_t2_s,
            max_t2_s,
            freq_range_lower_bound_hz,
            freq_range_upper_bound_hz,
            skip_fitting_mask,
            db,
        )
        seen["data_slice_size"] = int(data_slice.size)
        seen["sampling_time_size"] = int(
            np.asarray(kwargs["sampling_time_axis_s"]).size
        )
        n_metabs = len(initial_freqs_hz)
        rep_fitting = kwargs["rep_fitting"]
        niter = kwargs["niter"]
        npoints = kwargs["npoints"]
        return (
            np.full(n_metabs, 2.0 + 1.0j, dtype=np.complex128),
            initial_freqs_hz + 1.0,
            initial_t2s_s + 0.01,
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 7.0),
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 8.0),
            np.zeros(n_metabs, dtype=bool),
        )

    monkeypatch.setattr(
        "hypermri.analysis.fitting.spectroscopy._fit_single_spectrum_pseudo_inv",
        fake_fit_single,
    )

    _ = spectroscopy_fitting.fit_data_pseudo_inv(
        data=input_data,
        fit_params=fit_params,
        show_tqdm=False,
    )

    assert seen["data_slice_size"] == 8
    assert seen["sampling_time_size"] == 8


def test_fit_data_pseudo_inv_apply_ufunc_filters_positional_kwarg_collisions(
    monkeypatch,
):
    input_data = xr.DataArray(
        np.ones((16,), dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(16)},
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "rep_fitting": 2,
            "niter": 1,
            "npoints": 3,
            "min_t2_s": [0.01],
            "max_t2_s": [0.05],
        },
    )

    seen_kwargs: dict[str, object] = {}

    def fake_fit_single(
        data_slice: np.ndarray,
        initial_freqs_hz: np.ndarray,
        initial_t2s_s: np.ndarray,
        metabs_freq_ranges: np.ndarray,
        metabs_t2_ranges: np.ndarray,
        min_t2_s: np.ndarray,
        max_t2_s: np.ndarray,
        freq_range_lower_bound_hz: np.ndarray,
        freq_range_upper_bound_hz: np.ndarray,
        skip_fitting_mask: np.ndarray | None = None,
        db: bool = False,
        **kwargs,
    ) -> tuple:
        del (
            data_slice,
            metabs_freq_ranges,
            metabs_t2_ranges,
            min_t2_s,
            max_t2_s,
            freq_range_lower_bound_hz,
            freq_range_upper_bound_hz,
            skip_fitting_mask,
            db,
        )
        seen_kwargs.update(kwargs)
        n_metabs = len(initial_freqs_hz)
        rep_fitting = kwargs["rep_fitting"]
        niter = kwargs["niter"]
        npoints = kwargs["npoints"]
        return (
            np.full(n_metabs, 2.0 + 1.0j, dtype=np.complex128),
            initial_freqs_hz + 1.0,
            initial_t2s_s + 0.01,
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 7.0),
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 8.0),
            np.zeros(n_metabs, dtype=bool),
        )

    monkeypatch.setattr(
        "hypermri.analysis.fitting.spectroscopy._fit_single_spectrum_pseudo_inv",
        fake_fit_single,
    )

    result = spectroscopy_fitting.fit_data_pseudo_inv(
        data=input_data,
        fit_params=fit_params,
        show_tqdm=False,
    )

    assert "min_t2_s" not in seen_kwargs
    assert "max_t2_s" not in seen_kwargs
    assert np.allclose(result["fit_freqs"].values, fit_params["initial_freqs_hz"] + 1.0)


def test_fit_single_spectrum_pseudo_inv_default_rep_fitting_is_11(monkeypatch):
    def fake_fit_freq_pseudo_inv(values: np.ndarray, fit_params: dict) -> tuple:
        del values
        n_metabs = len(fit_params["metabs"])
        niter = int(fit_params.get("niter", 1))
        npoints = int(fit_params.get("npoints", 31))
        return (
            np.zeros((8, n_metabs), dtype=np.complex128),
            np.asarray(fit_params["metabs_freqs_hz"], dtype=np.float64),
            np.ones(n_metabs, dtype=np.complex128),
            np.zeros((n_metabs, niter, npoints, 3), dtype=np.float64),
        )

    def fake_fit_t2_pseudo_inv(values: np.ndarray, fit_params: dict) -> tuple:
        del values
        n_metabs = len(fit_params["metabs"])
        niter = int(fit_params.get("niter", 1))
        npoints = int(fit_params.get("npoints", 31))
        return (
            np.zeros((8, n_metabs), dtype=np.complex128),
            np.asarray(fit_params["metabs_t2s"], dtype=np.float64),
            np.ones(n_metabs, dtype=np.complex128),
            np.zeros((n_metabs, niter, npoints, 3), dtype=np.float64),
        )

    _patch_grid_searches(monkeypatch, fake_fit_freq_pseudo_inv, fake_fit_t2_pseudo_inv)

    _, _, _, r2_freq, r2_t2, _ = fid_pseudo_inv._fit_single_spectrum_pseudo_inv(
        data_slice=np.ones(8, dtype=np.complex128),
        initial_freq_hz=np.array([10.0], dtype=np.float64),
        initial_t2_s=np.array([0.02], dtype=np.float64),
        freq_search_range_hz=np.array([20.0], dtype=np.float64),
        t2_search_range_s=np.array([0.01], dtype=np.float64),
        min_t2_s=np.array([0.005], dtype=np.float64),
        max_t2_s=np.array([0.07], dtype=np.float64),
        freq_search_range_lower_bound_hz=np.array([0.0], dtype=np.float64),
        freq_search_range_upper_bound_hz=np.array([20.0], dtype=np.float64),
        metabs=["pyruvate"],
        signal_domain="temporal",
        sampling_time_axis_s=np.linspace(0.0, 7e-3, 8),
        niter=1,
        npoints=3,
    )

    assert r2_freq.shape == (1, 11, 1, 3, 3)
    assert r2_t2.shape == (1, 11, 1, 3, 3)


def test_def_fit_params_initialises_all_runtime_defaults():
    """Check runtime defaults with an explicit, usable frequency in Hz."""
    input_data = xr.DataArray(
        np.ones(8, dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(8, dtype=np.float64) * 1e-3},
        attrs={"signal_domain": "temporal"},
    )

    fit_params = def_fit_params(
        data=input_data,
        fit_params={"metabs": ["pyruvate"], "metabs_freqs_hz": [0.0]},
    )

    assert fit_params.attrs["niter"] == fid_pseudo_inv.NITER_DEFAULT
    assert fit_params.attrs["npoints"] == fid_pseudo_inv.NPOINTS_DEFAULT
    assert fit_params.attrs["pseudo_inv"] is True
    assert fit_params.attrs["rep_fitting"] == fid_pseudo_inv.REP_FITTING_DEFAULT
    assert fit_params.attrs["show_tqdm"] is False
    assert fit_params.attrs["snr_threshold"] is None
    assert fit_params.attrs["use_sse_mag"] is False
    assert fit_params.attrs["zoomfactor"] == fid_pseudo_inv.ZOOMFACTOR_DEFAULT


@pytest.mark.parametrize("threshold", [0.5, 1.0, 20.0])
def test_fit_single_spectrum_applies_amplitude_threshold_and_refits(
    monkeypatch, threshold
):
    """Refit only a retained subset, with original widths and filtering disabled.

    Equality with the threshold retains a component; excluding all components
    zeros amplitudes without starting a fit on an empty metabolite list.
    """
    fit_call_sizes: list[int] = []

    def fake_score_fit_parameters(
        values: np.ndarray,
        freq_hz: np.ndarray,
        t2_star_s: np.ndarray,
        fit_params: dict,
    ) -> tuple[float, np.ndarray]:
        """Return known amplitudes before and after excluding the weak peak."""
        del values, t2_star_s, fit_params
        amplitudes = (
            np.array([10.0 + 0.0j, 0.5 + 0.0j])
            if len(freq_hz) == 2
            else np.array([12.0 + 0.0j])
        )
        return 0.0, amplitudes

    def fake_fit_freq_pseudo_inv(values: np.ndarray, fit_params: dict) -> tuple:
        """Check that a subset refit starts with original, unshrunk settings."""
        del values
        n_metabs = len(fit_params["metabs"])
        fit_call_sizes.append(n_metabs)
        if n_metabs == 1:
            assert fit_params["metabs"] == ["pyruvate"]
            assert fit_params["snr_threshold"] is None
            np.testing.assert_array_equal(fit_params["metabs_freq_ranges"], [20.0])
            np.testing.assert_array_equal(fit_params["metabs_t2_ranges"], [0.01])
            np.testing.assert_array_equal(
                fit_params["freq_range_lower_bound_hz"], [0.0]
            )
            np.testing.assert_array_equal(
                fit_params["freq_range_upper_bound_hz"], [20.0]
            )
        return (
            np.zeros((8, n_metabs), dtype=np.complex128),
            np.asarray(fit_params["metabs_freqs_hz"], dtype=np.float64),
            np.ones(n_metabs, dtype=np.complex128),
            np.zeros((n_metabs, 1, 3, 3), dtype=np.float64),
        )

    def fake_fit_t2_pseudo_inv(values: np.ndarray, fit_params: dict) -> tuple:
        """Keep the supplied T2* guesses and provide valid diagnostic shapes."""
        del values
        n_metabs = len(fit_params["metabs"])
        return (
            np.zeros((8, n_metabs), dtype=np.complex128),
            np.asarray(fit_params["metabs_t2s"], dtype=np.float64),
            np.ones(n_metabs, dtype=np.complex128),
            np.zeros((n_metabs, 1, 3, 3), dtype=np.float64),
        )

    monkeypatch.setattr(
        fid_pseudo_inv, "_score_fit_parameters", fake_score_fit_parameters
    )
    _patch_grid_searches(monkeypatch, fake_fit_freq_pseudo_inv, fake_fit_t2_pseudo_inv)

    fit_amps, _, _, _, _, low_snr_refitted = (
        fid_pseudo_inv._fit_single_spectrum_pseudo_inv(
            data_slice=np.ones(8, dtype=np.complex128),
            initial_freq_hz=np.array([10.0, 30.0]),
            initial_t2_s=np.array([0.02, 0.02]),
            freq_search_range_hz=np.array([20.0, 20.0]),
            t2_search_range_s=np.array([0.01, 0.01]),
            min_t2_s=np.array([0.005, 0.005]),
            max_t2_s=np.array([0.07, 0.07]),
            freq_search_range_lower_bound_hz=np.array([0.0, 20.0]),
            freq_search_range_upper_bound_hz=np.array([20.0, 40.0]),
            metabs=["pyruvate", "lactate"],
            signal_domain="temporal",
            sampling_time_axis_s=np.linspace(0.0, 7e-3, 8),
            rep_fitting=1,
            niter=1,
            npoints=3,
            snr_threshold=threshold,
        )
    )

    if threshold == 1.0:
        np.testing.assert_allclose(fit_amps, [12.0, 0.0])
        np.testing.assert_array_equal(low_snr_refitted, [False, True])
        assert fit_call_sizes == [2, 1]
    else:
        excluded = np.array([10.0, 0.5]) < threshold
        np.testing.assert_array_equal(
            fit_amps, [0.0, 0.0] if np.all(excluded) else [10.0, 0.5]
        )
        np.testing.assert_array_equal(low_snr_refitted, excluded)
        assert fit_call_sizes == [2]


@pytest.mark.parametrize("snr_threshold", [-1.0, np.inf, "invalid"])
def test_def_fit_params_rejects_invalid_snr_threshold(snr_threshold):
    """Reject an invalid amplitude threshold with otherwise usable fit settings."""
    input_data = xr.DataArray(
        np.ones(8, dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(8, dtype=np.float64) * 1e-3},
        attrs={"signal_domain": "temporal"},
    )

    with pytest.raises(ValueError, match="snr_threshold"):
        def_fit_params(
            data=input_data,
            fit_params={
                "metabs": ["pyruvate"],
                "metabs_freqs_hz": [0.0],
                "snr_threshold": snr_threshold,
            },
        )


def test_fit_freq_and_t2_pseudo_inv_ignore_imaginary_parts_without_warnings():
    sampling_time_axis_s = np.linspace(0.0, 0.015, 16)
    input_data = (
        np.exp(-sampling_time_axis_s / 0.02)
        * np.exp(1j * 2.0 * np.pi * 12.0 * sampling_time_axis_s)
    ).astype(np.complex128)

    fit_params = {
        "metabs": ["pyruvate"],
        "npoints": 5,
        "niter": 1,
        "signal_domain": "temporal",
        "sampling_time_axis_s": sampling_time_axis_s,
        "pseudo_inv": True,
        "use_sse_mag": False,
        "metabs_freqs_hz": np.array([12.0 + 0.5j], dtype=np.complex128),
        "initial_t2s_s": np.array([0.02 + 0.01j], dtype=np.complex128),
        "metabs_freq_ranges": np.array([20.0 + 1.0j], dtype=np.complex128),
        "freq_range_lower_bound_hz": np.array([2.0 + 0.2j], dtype=np.complex128),
        "freq_range_upper_bound_hz": np.array([22.0 + 0.2j], dtype=np.complex128),
        "metabs_t2_ranges": np.array([0.02 + 0.01j], dtype=np.complex128),
        "min_t2_s": np.array([0.005 + 0.001j], dtype=np.complex128),
        "max_t2_s": np.array([0.08 + 0.001j], dtype=np.complex128),
    }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fid_pseudo_inv.fit_freq_pseudo_inv(values=input_data, fit_params=fit_params)
        fid_pseudo_inv.fit_t2_pseudo_inv(values=input_data, fit_params=fit_params)

    complex_warnings = [
        warning
        for warning in captured
        if issubclass(warning.category, np.ComplexWarning)
    ]
    assert complex_warnings == []


def test_fit_freq_pseudo_inv_enforces_min_peak_separation_hz():
    sampling_time_axis_s = np.linspace(0.0, 0.031, 64)
    true_freqs_hz = np.array([0.0, 24.0], dtype=np.float64)
    t2_star_s = np.array([0.035, 0.035], dtype=np.float64)
    true_amps = np.array([1.0 + 0.0j, 0.8 + 0.0j], dtype=np.complex128)
    basis = fid_pseudo_inv.fid_basefunc(true_freqs_hz, t2_star_s, sampling_time_axis_s)
    input_data = np.sum(basis.T * true_amps, axis=1)

    fit_params = {
        "metabs": ["peak_1", "peak_2"],
        "npoints": 9,
        "niter": 1,
        "signal_domain": "temporal",
        "sampling_time_axis_s": sampling_time_axis_s,
        "pseudo_inv": True,
        "use_sse_mag": False,
        "metabs_freqs_hz": np.array([0.0, 5.0], dtype=np.float64),
        "initial_t2s_s": t2_star_s,
        "metabs_freq_ranges": np.array([40.0, 40.0], dtype=np.float64),
        "freq_range_lower_bound_hz": np.array([-20.0, -15.0], dtype=np.float64),
        "freq_range_upper_bound_hz": np.array([20.0, 25.0], dtype=np.float64),
        "min_peak_separation_hz": 12.0,
    }

    _, fitted_freqs_hz, _, r2_freq = fid_pseudo_inv.fit_freq_pseudo_inv(
        values=input_data,
        fit_params=fit_params,
    )

    assert abs(fitted_freqs_hz[1] - fitted_freqs_hz[0]) >= 12.0
    invalid_scores = r2_freq[..., 1][~np.isfinite(r2_freq[..., 1])]
    assert invalid_scores.size > 0


def test_score_fit_parameters_rejects_close_frequency_pair():
    sampling_time_axis_s = np.linspace(0.0, 0.015, 16)
    input_data = np.ones_like(sampling_time_axis_s, dtype=np.complex128)

    score, amps = fid_pseudo_inv._score_fit_parameters(
        values=input_data,
        freq_hz=np.array([0.0, 5.0], dtype=np.float64),
        t2_star_s=np.array([0.02, 0.02], dtype=np.float64),
        fit_params={
            "signal_domain": "temporal",
            "sampling_time_axis_s": sampling_time_axis_s,
            "min_peak_separation_hz": 10.0,
        },
    )

    assert score == np.inf
    assert np.allclose(amps, 0.0)


def test_solve_for_complex_amplitudes_ridge_shrinks_amplitude_norm():
    time_axis_s = np.linspace(0.0, 0.012, 32)
    freqs_hz = np.array([-500.0, -495.0], dtype=np.float64)
    t2_star_s = np.array([0.02, 0.02], dtype=np.float64)
    true_amps = np.array([1.0 + 0.0j, 0.8 + 0.2j], dtype=np.complex128)

    basis = fid_pseudo_inv.fid_basefunc(freqs_hz, t2_star_s, time_axis_s)
    temporal_signal = np.sum(basis.T * true_amps, axis=1)

    _, beta_pinv = fid_pseudo_inv.solve_for_complex_amplitudes(
        values=temporal_signal,
        freq_hz=freqs_hz,
        t2_star_s=t2_star_s,
        time_axis_s=time_axis_s,
        signal_domain="temporal",
        pseudo_inv=True,
    )
    _, beta_ridge = fid_pseudo_inv.solve_for_complex_amplitudes(
        values=temporal_signal,
        freq_hz=freqs_hz,
        t2_star_s=t2_star_s,
        time_axis_s=time_axis_s,
        signal_domain="temporal",
        amp_solver="ridge",
        amp_ridge_lambda=10.0,
    )

    assert np.linalg.norm(beta_ridge) < np.linalg.norm(beta_pinv)
    assert np.all(np.isfinite(beta_ridge))


def test_def_fit_params_rejects_unknown_amp_solver():
    input_data = xr.DataArray(
        np.ones((8,), dtype=np.complex64),
        dims=["acquisition_idx"],
        coords={"acquisition_idx": np.arange(8)},
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8, dtype=np.float64) * 1e-3,
            "freq_axis_hz": np.linspace(-100.0, 100.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
        },
    )

    with pytest.raises(ValueError, match="Unknown amp_solver"):
        _ = def_fit_params(
            data=input_data,
            fit_params={"metabs": ["pyruvate"], "amp_solver": "not_a_solver"},
        )


def test_fit_data_pseudo_inv_accepts_averaged_input_with_constant_fit_params(
    monkeypatch,
):
    input_data = xr.DataArray(
        np.ones((3, 16), dtype=np.complex64),
        dims=["repetition_idx", "acquisition_idx"],
        coords={
            "repetition_idx": [0, 1, 2],
            "acquisition_idx": np.arange(16),
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "rep_fitting": 2,
            "niter": 1,
            "npoints": 3,
        },
    )
    averaged_data = input_data.mean(dim="repetition_idx")

    def fake_fit_single(
        data_slice: np.ndarray,
        initial_freqs_hz: np.ndarray,
        initial_t2s_s: np.ndarray,
        metabs_freq_ranges: np.ndarray,
        metabs_t2_ranges: np.ndarray,
        min_t2_s: np.ndarray,
        max_t2_s: np.ndarray,
        freq_range_lower_bound_hz: np.ndarray,
        freq_range_upper_bound_hz: np.ndarray,
        skip_fitting_mask: np.ndarray | None = None,
        db: bool = False,
        **kwargs,
    ) -> tuple:
        del (
            data_slice,
            metabs_freq_ranges,
            metabs_t2_ranges,
            min_t2_s,
            max_t2_s,
            freq_range_lower_bound_hz,
            freq_range_upper_bound_hz,
            skip_fitting_mask,
            db,
        )
        n_metabs = len(initial_freqs_hz)
        rep_fitting = kwargs["rep_fitting"]
        niter = kwargs["niter"]
        npoints = kwargs["npoints"]
        return (
            np.full(n_metabs, 2.0 + 1.0j, dtype=np.complex128),
            initial_freqs_hz + 1.0,
            initial_t2s_s + 0.01,
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 7.0),
            np.full((n_metabs, rep_fitting, niter, npoints, 3), 8.0),
            np.zeros(n_metabs, dtype=bool),
        )

    monkeypatch.setattr(
        "hypermri.analysis.fitting.spectroscopy._fit_single_spectrum_pseudo_inv",
        fake_fit_single,
    )

    result = spectroscopy_fitting.fit_data_pseudo_inv(
        data=averaged_data,
        fit_params=fit_params,
    )

    assert result["fit_freqs"].dims == ("metabolite",)
    assert np.allclose(
        result["fit_freqs"].values,
        fit_params["initial_freqs_hz"].isel(repetition_idx=0).values + 1.0,
    )


def test_fit_data_pseudo_inv_rejects_averaged_input_with_varying_fit_params():
    input_data = xr.DataArray(
        np.ones((3, 16), dtype=np.complex64),
        dims=["repetition_idx", "acquisition_idx"],
        coords={
            "repetition_idx": [0, 1, 2],
            "acquisition_idx": np.arange(16),
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(16) * 1e-3,
            "freq_axis_hz": np.linspace(-150.0, 150.0, 16),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 16),
            "bw_hz": 300.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "rep_fitting": 2,
            "niter": 1,
            "npoints": 3,
        },
    )
    repetition_offsets = xr.DataArray(
        [0.0, 1.0, 2.0],
        dims=["repetition_idx"],
        coords={"repetition_idx": input_data.coords["repetition_idx"]},
    )
    fit_params["initial_freqs_hz"] = fit_params["initial_freqs_hz"] + repetition_offsets
    fit_params = fid_pseudo_inv._update_frequency_bounds(fit_params)
    averaged_data = input_data.mean(dim="repetition_idx")

    with pytest.raises(ValueError, match="still varies along 'repetition_idx'"):
        spectroscopy_fitting.fit_data_pseudo_inv(
            data=averaged_data,
            fit_params=fit_params,
        )


def test_run_csi_fitting_pipeline_allows_added_metabolites_missing_from_fit_params():
    input_data = xr.DataArray(
        np.ones((8, 1, 1, 1), dtype=np.complex64),
        dims=["acquisition_idx", "read_idx", "phase_idx", "repetition_idx"],
        coords={
            "acquisition_idx": np.arange(8),
            "read_idx": [0],
            "phase_idx": [0],
            "repetition_idx": [0],
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8) * 1e-3,
            "freq_axis_hz": np.linspace(-100.0, 100.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
            "bw_hz": 200.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={
            "metabs": ["pyruvate"],
            "metabs_freq_ranges": [150.0],
            "min_t2_s": [0.01],
            "max_t2_s": [0.04],
        },
    )

    add_calls: list[dict[str, object]] = []

    def fake_fit_data_pseudo_inv(
        data: xr.DataArray,
        fit_params: xr.Dataset,
        show_tqdm: bool | None = None,
    ) -> xr.Dataset:
        del data, show_tqdm
        return xr.Dataset(
            attrs={"metabs": spectroscopy_pipeline._get_dataset_metabolites(fit_params)}
        )

    def fake_add_metab_to_fit_params(**kwargs) -> xr.Dataset:
        add_calls.append(kwargs)
        return _append_fake_metabolite(kwargs["fit_params"], kwargs["new_metab_name"])

    original_fit = spectroscopy_pipeline.fit_data_pseudo_inv
    original_add = spectroscopy_pipeline.add_metab_to_fit_params
    try:
        spectroscopy_pipeline.fit_data_pseudo_inv = fake_fit_data_pseudo_inv
        spectroscopy_pipeline.add_metab_to_fit_params = fake_add_metab_to_fit_params

        result = spectroscopy_pipeline.run_csi_fitting_pipeline(
            data=input_data,
            fit_params=fit_params,
            initial_metabolite="pyruvate",
            metabolites_to_add=["lactate", "pyruvatehydrate"],
            run_final_refinement=False,
        )
    finally:
        spectroscopy_pipeline.fit_data_pseudo_inv = original_fit
        spectroscopy_pipeline.add_metab_to_fit_params = original_add

    assert [call["new_metab_name"] for call in add_calls] == [
        "lactate",
        "pyruvatehydrate",
    ]
    assert add_calls[0]["new_t2_s"] == 0.02
    assert add_calls[0]["new_freq_range"] == 30.0
    assert add_calls[0]["new_t2_range_s"] == 0.04
    assert add_calls[0]["new_min_t2_s"] == 0.005
    assert add_calls[0]["new_max_t2_s"] == 0.1
    assert isinstance(result["fit_params"], xr.Dataset)


def test_cluster_peak_candidates_keeps_same_spectrum_neighbours_separate():
    candidates = [
        {"spectrum_id": 0, "freq_hz": 10.0, "height_snr": 8.0, "width_hz": 4.0},
        {"spectrum_id": 0, "freq_hz": 16.0, "height_snr": 7.5, "width_hz": 4.0},
        {"spectrum_id": 1, "freq_hz": 11.0, "height_snr": 8.2, "width_hz": 4.2},
        {"spectrum_id": 1, "freq_hz": 17.0, "height_snr": 7.8, "width_hz": 4.1},
    ]

    clusters = spectroscopy_pipeline._cluster_peak_candidates(
        candidates,
        tolerance_hz=8.0,
        min_cluster_support=2,
    )

    assert len(clusters) == 2
    assert np.allclose(
        [cluster["center_hz"] for cluster in clusters],
        [10.5, 16.5],
        atol=0.02,
    )
    assert [cluster["support"] for cluster in clusters] == [2, 2]


def test_suppress_close_clusters_keeps_stronger_cluster():
    clusters = [
        {
            "center_hz": 10.0,
            "support": 5,
            "median_height_snr": 6.0,
        },
        {
            "center_hz": 18.0,
            "support": 2,
            "median_height_snr": 3.0,
        },
        {
            "center_hz": 45.0,
            "support": 4,
            "median_height_snr": 4.0,
        },
    ]

    filtered = spectroscopy_pipeline._suppress_close_clusters(
        clusters,
        min_separation_hz=12.0,
    )

    assert [round(cluster["center_hz"], 3) for cluster in filtered] == [10.0, 45.0]


def test_resolve_selected_peak_indices_accepts_names_and_indices():
    peak_names = ["peak_1", "peak_2", "peak_3"]
    selected = spectroscopy_pipeline._resolve_selected_peak_indices(
        ["2", "peak_3"],
        peak_names=peak_names,
    )
    assert selected == [1, 2]


def test_run_csi_auto_fitting_pipeline_detects_stable_peak_clusters():
    acquisition_idx = np.arange(128) * 1e-3
    freq_axis_hz = np.linspace(-80.0, 80.0, 128)
    freq_axis_ppm = np.linspace(160.0, 190.0, 128)
    read_positions = np.arange(4)
    spectra = []
    peak_pairs = [(-20.0, 30.0), (-18.0, 32.0), (-15.0, 29.0), (-19.0, 31.0)]
    rng = np.random.default_rng(7)
    for peak_a, peak_b in peak_pairs:
        spectrum = (
            np.exp(-0.5 * ((freq_axis_hz - peak_a) / 2.0) ** 2)
            + 0.7 * np.exp(-0.5 * ((freq_axis_hz - peak_b) / 2.5) ** 2)
            + 0.01
            + rng.normal(0.0, 0.002, size=freq_axis_hz.shape)
        )
        spectra.append(spectrum.astype(np.complex64))

    input_data = xr.DataArray(
        np.stack(spectra, axis=1),
        dims=["acquisition_idx", "read_idx"],
        coords={
            "acquisition_idx": acquisition_idx,
            "read_idx": read_positions,
        },
        attrs={
            "signal_domain": "spectral",
            "freq_axis_hz": freq_axis_hz,
            "freq_axis_ppm": freq_axis_ppm,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )

    fit_calls: list[list[str]] = []
    threshold_calls: list[float | None] = []
    add_calls: list[str] = []

    def fake_fit_data_pseudo_inv(
        data: xr.DataArray,
        fit_params: xr.Dataset,
        show_tqdm: bool | None = None,
    ) -> xr.Dataset:
        del data, show_tqdm
        fit_calls.append(spectroscopy_pipeline._get_dataset_metabolites(fit_params))
        threshold_calls.append(fit_params.attrs.get("snr_threshold"))
        return xr.Dataset(attrs={"call_index": len(fit_calls)})

    def fake_add_metab_to_fit_params(**kwargs) -> xr.Dataset:
        add_calls.append(kwargs["new_metab_name"])
        return _append_fake_metabolite(kwargs["fit_params"], kwargs["new_metab_name"])

    def fake_initialise_fit_params_from_results(**kwargs) -> xr.Dataset:
        refined = kwargs["fit_params"].copy(deep=True)
        refined.attrs["refined"] = True
        return refined

    original_fit = spectroscopy_pipeline.fit_data_pseudo_inv
    original_add = spectroscopy_pipeline.add_metab_to_fit_params
    original_initialise = spectroscopy_pipeline.initialise_fit_params_from_results
    try:
        spectroscopy_pipeline.fit_data_pseudo_inv = fake_fit_data_pseudo_inv
        spectroscopy_pipeline.add_metab_to_fit_params = fake_add_metab_to_fit_params
        spectroscopy_pipeline.initialise_fit_params_from_results = (
            fake_initialise_fit_params_from_results
        )

        result = spectroscopy_pipeline.run_csi_auto_fitting_pipeline(
            data=input_data,
            b0_shift_tolerance_hz=6.0,
            max_peaks=2,
            verbose=False,
        )
    finally:
        spectroscopy_pipeline.fit_data_pseudo_inv = original_fit
        spectroscopy_pipeline.add_metab_to_fit_params = original_add
        spectroscopy_pipeline.initialise_fit_params_from_results = original_initialise

    assert fit_calls[0] == ["peak_1"]
    assert len(fit_calls) == 3
    assert add_calls == ["peak_2"]
    assert len(result["auto_diagnostics"]["cluster_summary"]) == 2
    centers = [
        item["center_hz"] for item in result["auto_diagnostics"]["cluster_summary"]
    ]
    assert np.allclose(centers, [-18.0, 30.5], atol=2.0)
    assert result["auto_diagnostics"]["snr_threshold"] is None
    assert result["auto_diagnostics"]["legacy_fid_tail_noise_used"] is False
    assert threshold_calls == [None, None, None]
    assert result["auto_diagnostics"]["rep_fitting"] >= 1
    assert result["auto_diagnostics"]["npoints"] >= 31
    assert result["fit_params"].attrs["refined"] is True


def test_run_csi_auto_fitting_pipeline_filters_selected_peaks():
    acquisition_idx = np.arange(128) * 1e-3
    freq_axis_hz = np.linspace(-80.0, 80.0, 128)
    freq_axis_ppm = np.linspace(160.0, 190.0, 128)
    read_positions = np.arange(4)
    spectra = []
    peak_pairs = [(-20.0, 30.0), (-18.0, 32.0), (-15.0, 29.0), (-19.0, 31.0)]
    rng = np.random.default_rng(7)
    for peak_a, peak_b in peak_pairs:
        spectrum = (
            np.exp(-0.5 * ((freq_axis_hz - peak_a) / 2.0) ** 2)
            + 0.7 * np.exp(-0.5 * ((freq_axis_hz - peak_b) / 2.5) ** 2)
            + 0.01
            + rng.normal(0.0, 0.002, size=freq_axis_hz.shape)
        )
        spectra.append(spectrum.astype(np.complex64))

    input_data = xr.DataArray(
        np.stack(spectra, axis=1),
        dims=["acquisition_idx", "read_idx"],
        coords={
            "acquisition_idx": acquisition_idx,
            "read_idx": read_positions,
        },
        attrs={
            "signal_domain": "spectral",
            "freq_axis_hz": freq_axis_hz,
            "freq_axis_ppm": freq_axis_ppm,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )

    fit_calls: list[list[str]] = []

    def fake_fit_data_pseudo_inv(
        data: xr.DataArray,
        fit_params: xr.Dataset,
        show_tqdm: bool | None = None,
    ) -> xr.Dataset:
        del data, show_tqdm
        fit_calls.append(spectroscopy_pipeline._get_dataset_metabolites(fit_params))
        return xr.Dataset(attrs={"call_index": len(fit_calls)})

    def fake_initialise_fit_params_from_results(**kwargs) -> xr.Dataset:
        refined = kwargs["fit_params"].copy(deep=True)
        refined.attrs["refined"] = True
        return refined

    original_fit = spectroscopy_pipeline.fit_data_pseudo_inv
    original_initialise = spectroscopy_pipeline.initialise_fit_params_from_results
    try:
        spectroscopy_pipeline.fit_data_pseudo_inv = fake_fit_data_pseudo_inv
        spectroscopy_pipeline.initialise_fit_params_from_results = (
            fake_initialise_fit_params_from_results
        )

        result = spectroscopy_pipeline.run_csi_auto_fitting_pipeline(
            data=input_data,
            b0_shift_tolerance_hz=6.0,
            max_peaks=2,
            selected_peaks=[2],
            verbose=False,
        )
    finally:
        spectroscopy_pipeline.fit_data_pseudo_inv = original_fit
        spectroscopy_pipeline.initialise_fit_params_from_results = original_initialise

    assert fit_calls == [["peak_2"], ["peak_2"]]
    assert result["auto_diagnostics"]["selected_peak_indices"] == [1]
    assert result["auto_diagnostics"]["n_clusters_after_filtering"] == 2
    assert result["auto_diagnostics"]["n_clusters_after_selection"] == 1


def test_run_csi_fitting_pipeline_forwards_quality_options_unchanged(monkeypatch):
    input_data = xr.DataArray(
        np.ones((8, 1), dtype=np.complex64),
        dims=["acquisition_idx", "repetition_idx"],
        coords={"acquisition_idx": np.arange(8), "repetition_idx": [0]},
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8) * 1e-3,
            "freq_axis_hz": np.linspace(-100.0, 100.0, 8),
            "bw_hz": 200.0,
            "dwell_time_s": 1e-3,
        },
    )
    fit_params = def_fit_params(
        data=input_data,
        fit_params={"metabs": ["pyruvate"], "metabs_freqs_hz": [0.0]},
    )
    quality_calls: list[dict[str, object]] = []

    def fake_evaluate(data, fit_params, noise_references, **kwargs):
        del data, fit_params, noise_references
        quality_calls.append(kwargs)
        return xr.Dataset(attrs={"quality": True})

    monkeypatch.setattr(
        spectroscopy_pipeline,
        "fit_data_pseudo_inv",
        lambda data, fit_params, show_tqdm=None: xr.Dataset(),
    )
    monkeypatch.setattr(
        spectroscopy_pipeline,
        "initialise_fit_params_from_results",
        lambda **kwargs: kwargs["fit_params"],
    )
    monkeypatch.setattr(
        spectroscopy_pipeline, "evaluate_spectroscopy_fit_quality", fake_evaluate
    )
    options = {"detection_fit_snr": 8.0, "detection_null_samples": 99}
    result = spectroscopy_pipeline.run_csi_fitting_pipeline(
        data=input_data,
        fit_params=fit_params,
        verbose=False,
        run_quality=True,
        quality_config=options,
    )

    assert len(quality_calls) == 1
    assert quality_calls[0]["config"] is options
    assert result["quality_results"].attrs["quality"] is True


def _two_voxel_fid(nan_voxel: int) -> xr.DataArray:
    """Invented 8-sample FID in two voxels; one voxel has a single NaN sample."""
    values = np.ones((8, 2, 1, 1), dtype=np.complex64)
    values[3, nan_voxel] = np.nan
    return xr.DataArray(
        values,
        dims=["acquisition_idx", "read_idx", "phase_idx", "repetition_idx"],
        coords={
            "acquisition_idx": np.arange(8),
            "read_idx": [0, 1],
            "phase_idx": [0],
            "repetition_idx": [0],
        },
        attrs={
            "signal_domain": "temporal",
            "sampling_time_axis_s": np.arange(8) * 1e-3,
            "freq_axis_hz": np.linspace(-100.0, 100.0, 8),
            "freq_axis_ppm": np.linspace(160.0, 190.0, 8),
            "bw_hz": 200.0,
            "dwell_time_s": 1e-3,
            "b0": 7.0,
            "nucleus": "13c",
        },
    )


_NAN_TEST_OPTIONS = {
    "metabs": ["pyruvate", "lactate"],
    "metabs_freq_ranges": [30.0, 30.0],
    "metabs_t2_ranges": [0.04, 0.01],
    "initial_t2s_s": [0.02, 0.03],
    "min_t2_s": [0.01, 0.01],
    "max_t2_s": [0.04, 0.04],
}


def test_def_fit_params_is_nan_where_the_data_are_nan_by_default():
    data = _two_voxel_fid(nan_voxel=1)

    params = def_fit_params(data, dict(_NAN_TEST_OPTIONS))

    float_vars = [
        name
        for name, var in params.data_vars.items()
        if np.issubdtype(var.dtype, np.floating)
        or np.issubdtype(var.dtype, np.complexfloating)
    ]
    assert "initial_freqs_hz" in float_vars
    for name in float_vars:
        assert np.isnan(params[name].sel(read_idx=1)).all(), name
        assert np.isfinite(params[name].sel(read_idx=0)).all(), name

    kept = def_fit_params(data, dict(_NAN_TEST_OPTIONS), mask_nonfinite_input=False)
    for name in float_vars:
        assert np.isfinite(kept[name]).all(), name


def test_fit_with_default_params_marks_the_nan_voxel_invalid():
    data = _two_voxel_fid(nan_voxel=0)

    params = def_fit_params(data, dict(_NAN_TEST_OPTIONS))
    results = spectroscopy_fitting.fit_data_pseudo_inv(data, params)

    assert results["fit_status"].sel(read_idx=0).item() == "invalid_input"
    assert results["fit_status"].sel(read_idx=1).item() != "invalid_input"
