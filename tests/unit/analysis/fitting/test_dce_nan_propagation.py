import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from hypermri.analysis.dce.conversions import concentration_to_signal
from hypermri.analysis.dce.fitting import run_dce_pipeline
from hypermri.analysis.dce.fitting_support import empirical_log_logistic_model
from hypermri.analysis.dce.models.patlak import fit_single_voxel_patlak_helper
from hypermri.analysis.dce.models.tofts import tofts_model_1comp


def _make_synthetic_tofts_inputs():
    times_s = np.arange(8, dtype=np.float64)
    aif_concentration = np.array(
        [0.0, 0.0, 0.4, 0.9, 0.6, 0.3, 0.15, 0.05], dtype=np.float64
    )
    cp_func = interp1d(
        times_s,
        aif_concentration,
        kind="linear",
        fill_value="extrapolate",
    )
    tissue_concentration = tofts_model_1comp(times_s, 0.18, 0.55, cp_func)

    s0 = 120.0
    t10_s = 1.35
    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5

    signal = concentration_to_signal(
        tissue_concentration,
        S_0=s0,
        T10_s=t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
    ).reshape(1, 1, 1, -1)

    dce = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": [0],
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(signal.shape[-1], dtype=int),
        },
        name="dce_signal",
    )
    fitted_t1 = xr.Dataset(
        {
            "t1": xr.DataArray(
                np.full((1, 1, 1), t10_s * 1000.0),
                dims=("read_idx", "phase_idx", "slice_idx"),
                coords={
                    "read_idx": dce.coords["read_idx"],
                    "phase_idx": dce.coords["phase_idx"],
                    "slice_idx": dce.coords["slice_idx"],
                },
                attrs={"units": "ms"},
            )
        }
    )

    return {
        "dce": dce,
        "fitted_t1": fitted_t1,
        "aif_concentration": aif_concentration,
        "times_s": times_s,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
    }


def test_run_dce_pipeline_tofts_returns_nan_params_when_signal_contains_nan():
    synthetic = _make_synthetic_tofts_inputs()
    dce_with_nan = synthetic["dce"].copy()
    dce_with_nan.loc[dict(read_idx=0, phase_idx=0, slice_idx=0, repetition_idx=3)] = (
        np.nan
    )

    result = run_dce_pipeline(
        dce_data=dce_with_nan,
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={"model_mode": "1comp", "n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert np.isnan(
        float(result["Ktrans_1c"].isel(read_idx=0, phase_idx=0, slice_idx=0))
    )
    assert np.isnan(float(result["kep_1c"].isel(read_idx=0, phase_idx=0, slice_idx=0)))
    assert np.isnan(float(result["rss_1c"].isel(read_idx=0, phase_idx=0, slice_idx=0)))


def test_run_dce_pipeline_empirical_log_logistic_does_not_require_t1():
    times_s = np.arange(8, dtype=np.float64)
    signal = empirical_log_logistic_model(
        times_s,
        bolus_arrival_s=1.0,
        rise_time_s=2.0,
        rise_shape=4.0,
        washout_rate_s=0.08,
        peak_amplitude=5.0,
        baseline_offset=100.0,
    ).reshape(1, 1, 1, -1)
    dce = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": [0],
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(signal.shape[-1], dtype=int),
        },
        name="dce_signal",
    )

    result = run_dce_pipeline(
        dce_data=dce,
        time_points_s=times_s,
        fit_type="empirical_log_logistic",
        use_multiprocessing=False,
    )

    assert "t1_map" not in result
    assert "concentration_map" not in result
    assert "signal_curve_map" in result
    assert result.attrs["fit_input_space"] == "signal"
    assert bool(result["fit_success"].isel(read_idx=0, phase_idx=0, slice_idx=0))


def test_fit_single_voxel_patlak_returns_nan_when_any_input_contains_nan():
    result = fit_single_voxel_patlak_helper(
        signal=np.array([0.0, np.nan, 0.2, 0.3], dtype=np.float64),
        aif_signal=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        time_points_s=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        fit_params={"fit_start_index": 0},
    )

    assert np.isnan(result["K_trans"])
    assert np.isnan(result["Vp"])
    assert np.isnan(result["rss_patlak"])
