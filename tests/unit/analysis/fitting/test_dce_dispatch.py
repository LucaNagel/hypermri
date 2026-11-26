"""Characterize F5 dispatch on invented labelled concentration maps.

The reconstruction cases compare complete public accessor and named model
outputs at rtol 1e-12, atol 1e-14 mM. This is structural equivalence evidence,
not a claim of physiological parameter recovery. No scanner data is used.
"""

from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.dce import fitting_support
from hypermri.analysis.dce.models import (
    empirical_log_logistic,
    patlak,
    sourbron,
    tofts,
    two_c_irf,
)
from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.registry import get_registry


def _fitted_case(case, transpose=False):
    times = np.array([0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77], dtype=float)
    plasma = np.exp(-(((times - 20) / 12) ** 2))
    field = xr.DataArray(
        [[1.0, 1.2], [0.8, 1.1]],
        dims=("read_idx", "phase_idx"),
        coords={"read_idx": [9.0, -9.0], "phase_idx": [3, 1]},
    )
    if transpose:
        field = field.transpose()
    names = {
        "patlak": {"K_trans": 0.003, "Vp": 0.08},
        "tofts_1comp": {"Ktrans_1c": 0.005, "kep_1c": 0.02, "rss_1c": 0.01},
        "tofts_2comp": {
            "Ktrans_2c": 0.005,
            "kep_2c": 0.02,
            "vp_2c": 0.08,
            "ve_2c": 0.25,
            "rss_2c": 0.001,
        },
        "sourbron": {"vp": 0.25, "Ft": 0.03, "Tt": 60.0},
        "sourbron_tp": {"vp": 0.25, "Ft": 0.03, "Tt": 60.0, "Tp": 5.0},
        "2c_irf": {
            "GFR": 0.01,
            "RPF": 0.035,
            "vascular_delay_s": 3.0,
            "vascular_dispersion_rate_s": 0.2,
            "tubular_delay_s": 20.0,
            "tubular_dispersion_rate_s": 0.05,
        },
        "empirical_log_logistic": {
            "bolus_arrival_s": 12.0,
            "rise_time_s": 10.0,
            "rise_shape": 4.0,
            "washout_rate_s": 0.01,
            "peak_amplitude": 0.5,
            "baseline_offset": 0.0,
        },
    }
    names["tofts_both"] = {**names["tofts_1comp"], **names["tofts_2comp"]}
    dataset = xr.Dataset({name: value * field for name, value in names[case].items()})
    model = case.split("_")[0] if case.startswith(("tofts_", "sourbron_")) else case
    result_names = {
        "tofts": "tofts_model",
        "sourbron": "sourbron_separable_compartment",
    }
    dataset.attrs.update(
        model_name=result_names.get(model, model), dynamic_dim="repetition_idx"
    )
    if model == "tofts":
        dataset.attrs["model_mode"] = case.removeprefix("tofts_")
    dataset["time_points_s"] = xr.DataArray(times, dims="repetition_idx")
    dataset["aif_signal"] = xr.DataArray(plasma, dims="aif_idx")
    dataset["aif_time_s"] = xr.DataArray(times, dims="aif_idx")
    dataset["aif_concentration"] = xr.DataArray(plasma, dims="aif_idx")
    return dataset, times, plasma, model


def _named_reconstruction(dataset, times, plasma, model):
    if model == "patlak":
        return patlak.patlak_model(times, plasma, dataset.K_trans, dataset.Vp)
    if model == "tofts":
        return tofts.tofts_model_reconstruction(
            times,
            plasma,
            aif_times=times,
            model_mode=dataset.attrs["model_mode"],
            **{
                name: dataset.get(name)
                for name in (
                    "Ktrans_1c",
                    "kep_1c",
                    "Ktrans_2c",
                    "kep_2c",
                    "vp_2c",
                    "ve_2c",
                    "rss_1c",
                    "rss_2c",
                )
            },
        )
    if model == "sourbron":
        return sourbron.sourbron_model_reconstruction(
            times, plasma, dataset.vp, dataset.Ft, dataset.Tt, dataset.get("Tp")
        )
    if model == "2c_irf":
        return two_c_irf.two_c_irf_reconstruction(
            times,
            plasma,
            dataset.GFR,
            dataset.RPF,
            dataset.vascular_delay_s,
            dataset.vascular_dispersion_rate_s,
            dataset.tubular_delay_s,
            dataset.tubular_dispersion_rate_s,
        )
    return empirical_log_logistic.empirical_log_logistic_reconstruction(
        times,
        dataset.bolus_arrival_s,
        dataset.rise_time_s,
        dataset.rise_shape,
        dataset.washout_rate_s,
        dataset.peak_amplitude,
        dataset.baseline_offset,
    )


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "patlak",
        "tofts_1comp",
        "tofts_2comp",
        "tofts_both",
        "sourbron",
        "sourbron_tp",
        "2c_irf",
        "empirical_log_logistic",
    ],
)
def test_dce_accessor_matches_named_reconstruction_before_dispatch_move(
    case, transpose
):
    dataset, times, plasma, model = _fitted_case(case, transpose)
    before = dataset.copy(deep=True)
    expected = _named_reconstruction(dataset, times, plasma, model)
    result = dataset.hp.reconstruct()
    assert result.dims == expected.dims
    assert result.dtype == expected.dtype
    xr.testing.assert_identical(
        xr.Dataset(coords=result.coords), xr.Dataset(coords=expected.coords)
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
    selected = dataset.hp.reconstruct(voxel_coords={"read_idx": 1})
    np.testing.assert_allclose(
        selected, expected.isel(read_idx=1), rtol=1e-12, atol=1e-14
    )
    xr.testing.assert_identical(dataset, before)


@pytest.mark.parametrize("model", ["2c_irf", "empirical_log_logistic"])
def test_generic_dce_preparation_limit_is_preserved(model):
    with pytest.raises(ValueError, match="does not provide parameter preparation"):
        api.prepare_fit_params(None, model)
    assert get_registry().get(model).reconstruct is not None


@pytest.mark.parametrize("model", ["patlak", "tofts", "sourbron"])
def test_named_dce_fit_does_not_dispatch_back_through_generic_facade(
    model, monkeypatch
):
    def recursive_dispatch(**_options):
        raise AssertionError("A named DCE fit must call its domain execution path.")

    monkeypatch.setattr(fitting_support, "_api_fit", recursive_dispatch)
    monkeypatch.setattr(api, "fit", recursive_dispatch)
    times = np.arange(12, dtype=float) * 4
    plasma = np.exp(-(((times - 16) / 8) ** 2))
    tissue = xr.DataArray(
        0.1 * plasma + 0.005 * times,
        dims="repetition_idx",
        coords={"repetition_idx": times},
    )
    fitted = getattr(fitting_support, "fit_" + model + "_model")(
        tissue,
        aif_concentration=plasma,
        aif_time_s=times,
        fit_params={"fit_start_index": 0},
        use_multiprocessing=False,
    )
    assert fitted.attrs["model_name"]
    from hypermri.analysis.dce.shared import _build_fit_dispatch_metadata

    provenance = _build_fit_dispatch_metadata(
        requested_fit_type=model,
        requested_fit_params={},
        effective_fit_type=model,
        effective_fit_params={},
        default_model_mode="both",
    )
    assert provenance["fit_backend_function"] == (
        "hypermri.analysis.dce.execution._fit_dce_maps"
    )
    assert provenance["fit_dispatch_summary"].endswith(
        "DCE concentration-map execution"
    )


@pytest.fixture
def dictionary_model(monkeypatch):
    """Provide one invented constant model using only the dictionary contract."""
    from hypermri.analysis.dce.execution import _fit_dce_maps
    from hypermri.analysis.dce.model_selection import _DCE_MODELS, _DceModel

    get_registry()
    module = ModuleType("invented_dce_fixture")
    module.RESULT_MODEL_NAME = "invented_dce_fixture_result"
    module._DEFAULT_MODEL_MODE = "invented_dce_fixture"
    module._BACKEND_DESCRIPTION = "invented constant concentration fixture"
    module._OUTPUT_DTYPES = {"fit_success": bool, "n_points_fit": np.int16}
    calls = {"prepare": 0, "voxel": 0, "annotate": 0}

    def prepare(data, options=None, return_params_info=False):
        calls["prepare"] += 1
        if return_params_info:
            return None
        return xr.Dataset(
            attrs={**dict(options or {}), "dynamic_dim": "repetition_idx"}
        )

    def fit_voxel(signal, dynamic_coords, fit_params, indices):
        calls["voxel"] += 1
        return {
            "level": float(np.mean(signal)),
            "fit_success": True,
            "n_points_fit": len(signal),
        }, indices

    def reconstruct(level, time_points_s):
        time = xr.DataArray(
            time_points_s,
            dims="repetition_idx",
            coords={"repetition_idx": time_points_s},
        )
        return level + xr.zeros_like(time)

    def reconstruction_params(data, options, voxel_coords=None):
        times = options.get("time_points_s", data["time_points_s"].values)
        return {"level": data.level, "time_points_s": times}

    def quality_inputs(
        fitted,
        concentrations,
        *,
        aif_concentration,
        aif_time_s,
        time_points_for_fit,
        fit_params,
        rep_coord,
    ):
        prediction = reconstruct(fitted.level, time_points_for_fit).assign_coords(
            repetition_idx=rep_coord
        )
        if "time_s" in concentrations.coords:
            prediction = prediction.assign_coords(time_s=concentrations.time_s)
        return (
            prediction,
            1,
            np.isfinite(fitted.level),
            xr.zeros_like(fitted.level, dtype=bool),
            None,
        )

    def annotate(dataset):
        calls["annotate"] += 1
        dataset["level"].attrs.update(units="mM", fixture="invented")

    def fit_fixture_dce_model(data, *, fit_params=None, use_multiprocessing=True):
        return _fit_dce_maps(
            data,
            "invented_dce_fixture",
            {**dict(fit_params or {}), "use_multiprocessing": use_multiprocessing},
        )

    module._prepare_fit_params = prepare
    module._describe_params = lambda: {"output_parameters": {"level": "mM"}}
    module._fit_voxel = fit_voxel
    module._output_param_names = lambda options: (
        "level",
        "fit_success",
        "n_points_fit",
    )
    module._reconstruct = reconstruct
    module._reconstruction_parameters = reconstruction_params
    module._quality_window = lambda times, plasma, options: np.ones(
        len(times), dtype=bool
    )
    module._quality_inputs = quality_inputs
    module._annotate_model_output = annotate
    monkeypatch.setitem(
        _DCE_MODELS,
        "invented_dce_fixture",
        _DceModel(module, fit_fixture_dce_model, requires_aif=False),
    )
    times = np.arange(8, dtype=float) * 4
    data = xr.DataArray(
        np.full((2, 8), 2.0, dtype=np.float32),
        dims=("read_idx", "repetition_idx"),
        coords={
            "read_idx": [7.0, -7.0],
            "repetition_idx": times,
            "time_s": ("repetition_idx", times),
        },
        attrs={"units": "mM"},
    )
    return module, calls, data, times


def test_one_dictionary_entry_supplies_facade_accessor_precision_and_quality(
    dictionary_model,
):
    module, calls, data, times = dictionary_model
    # Existing legacy defaults are already loaded; the domain entry needs no registration.
    get_registry()
    before = data.copy(deep=True)
    prepared = api.prepare_fit_params(
        data, "invented_dce_fixture", {"use_multiprocessing": False}
    )
    fitted = api.fit(data, "invented_dce_fixture", prepared)
    assert calls["prepare"] == 1
    assert calls["voxel"] == 2
    assert fitted.level.dtype == np.float64
    assert fitted.fit_success.dtype == bool
    assert fitted.n_points_fit.dtype == np.int16
    fitted["time_points_s"] = xr.DataArray(times, dims="repetition_idx")
    prediction = api.reconstruct_fit(fitted)
    np.testing.assert_allclose(prediction, 2.0, rtol=1e-12, atol=1e-14)
    quality = fitting_support.compute_fit_quality_metrics(
        fitted,
        data,
        fit_type="invented_dce_fixture",
        aif_concentration=np.zeros_like(times),
        aif_time_s=times,
        fit_params={"fit_start_index": 0},
    )
    np.testing.assert_allclose(quality.rss, 0.0, atol=1e-14)
    annotated = fitting_support._annotate_output_dataset(
        fitted, fit_type="invented_dce_fixture"
    )
    assert annotated.level.attrs["fixture"] == "invented"
    assert calls["annotate"] == 1
    assert fitted.attrs["model_name"] == module.RESULT_MODEL_NAME
    xr.testing.assert_identical(data, before)


def test_one_dictionary_entry_supplies_pipeline_selection_and_annotation(
    dictionary_model,
):
    _module, calls, data, times = dictionary_model
    from hypermri.analysis.dce.pipeline import run_dce_pipeline

    result = run_dce_pipeline(
        dce_data=data,
        time_points_s=times,
        fit_type="invented_dce_fixture",
        fit_params_pk={"n_precontrast": 2},
        use_multiprocessing=False,
    )
    assert calls["prepare"] == 1
    assert calls["voxel"] == 2
    assert calls["annotate"] == 1
    assert result.level.attrs["fixture"] == "invented"
    assert result.attrs["effective_fit_type"] == "invented_dce_fixture"
    assert result.attrs["fit_function"].endswith("fit_fixture_dce_model")
    np.testing.assert_allclose(result.level, 2.0, rtol=1e-12, atol=1e-14)
