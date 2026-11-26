import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

import hypermri.analysis.dce.fitting_support as fitting_support
from hypermri.analysis.dce.aif import prepare_aif
from hypermri.analysis.dce.conversions import (
    concentration_to_signal,
    signal_to_concentration,
)
from hypermri.analysis.dce.fitting import (
    compute_s0_map,
    convert_signal_to_concentration_map,
    fit_empirical_log_logistic_model,
    fit_extended_tofts_model,
    fit_patlak_model,
    fit_sourbron_model,
    fit_two_c_irf_model,
    run_dce_pipeline,
    # run_dce_pipeline_legacy_signal_fit,
    select_fit_window,
    standardize_t1_map_units,
    two_c_irf_model,
    validate_aif,
)
from hypermri.analysis.dce.models.sourbron import (
    sourbron_separable_compartment_model,
)
from hypermri.analysis.dce.models.tofts import (
    extended_tofts_model_2comp,
    tofts_model_1comp,
)
from hypermri.analysis.dce.renal import (
    collect_extended_tofts_roi_metrics,
    run_renal_dce_pipeline,
    save_extended_tofts_roi_metrics_csv,
)
from hypermri.analysis.fitting.api import fit


def _make_synthetic_renal_inputs(slice_positions=(0.0,), r1=3.5):
    times_s = np.arange(6, dtype=np.float64) * 5.0
    aif_concentration = np.array([0.0, 0.2, 0.5, 0.3, 0.1, 0.05], dtype=np.float64)
    integral_aif = cumulative_trapezoid(aif_concentration, times_s, initial=0.0)

    slopes = np.array([0.03, 0.015], dtype=np.float64)
    intercepts = np.array([0.12, 0.07], dtype=np.float64)
    tissue_concentration = (
        slopes[:, None] * integral_aif[None, :]
        + intercepts[:, None] * aif_concentration[None, :]
    )

    s0 = 100.0
    t10_s = 1.2
    tr_s = 0.05
    fa_deg = 15.0

    signal = concentration_to_signal(
        tissue_concentration,
        S_0=s0,
        T10_s=t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
    )
    signal = signal[:, None, None, :]

    dce = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(signal.shape[0]),
            "phase_idx": np.arange(signal.shape[1]),
            "slice_idx": np.arange(signal.shape[2], dtype=int),
            "repetition_idx": np.arange(signal.shape[3], dtype=int),
        },
        name="dce_signal",
        attrs={"repetition_time_ms": tr_s * 1000.0, "flipangle_deg": fa_deg},
    )
    dce.coords["slice_idx"].attrs["axis_mm"] = np.asarray(
        slice_positions, dtype=np.float64
    )

    t1_map = xr.DataArray(
        np.full((signal.shape[0], signal.shape[1], signal.shape[2]), t10_s * 1000.0),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )
    t1_map.coords["slice_idx"].attrs["axis_mm"] = np.asarray(
        slice_positions, dtype=np.float64
    )
    fitted_t1 = xr.Dataset({"t1": t1_map})

    return {
        "dce": dce,
        "fitted_t1": fitted_t1,
        "aif_concentration": aif_concentration,
        "times_s": times_s,
        "slopes": slopes,
        "intercepts": intercepts,
        "tissue_concentration": tissue_concentration,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "r1": r1,
    }


def _make_synthetic_tofts_inputs():
    times_s = np.arange(8, dtype=np.float64) * 5.0
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
        attrs={"repetition_time_ms": tr_s * 1000.0, "flipangle_deg": fa_deg},
    )
    t1_map = xr.DataArray(
        np.full((1, 1, 1), t10_s * 1000.0),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )

    return {
        "dce": dce,
        "fitted_t1": xr.Dataset({"t1": t1_map}),
        "aif_concentration": aif_concentration,
        "times_s": times_s,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
    }


def _make_signal_conversion_alignment_inputs(*, misaligned_t1: bool = False):
    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5
    s0_values = np.array([100.0, 120.0], dtype=np.float64)
    t1_aligned_ms = np.array([1000.0, 1500.0], dtype=np.float64)
    expected_concentration = np.array(
        [[0.0, 0.08], [0.0, 0.12]],
        dtype=np.float64,
    )

    signal = np.stack(
        [
            concentration_to_signal(
                expected_concentration[idx],
                S_0=s0_values[idx],
                T10_s=t1_aligned_ms[idx] / 1000.0,
                TR_s=tr_s,
                flip_angle_deg=fa_deg,
                r1=r1,
            )
            for idx in range(len(s0_values))
        ],
        axis=0,
    ).reshape(1, 1, len(s0_values), -1)

    dce = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": [0],
            "phase_idx": [0],
            "slice_idx": [0, 1],
            "repetition_idx": [0, 1],
        },
    )
    dce.coords["slice_idx"].attrs.update(
        {
            "unit": "mm",
            "resolution_mm": 5.0,
            "axis_mm": np.array([0.0, 5.0], dtype=np.float64),
        }
    )

    s0_map = xr.DataArray(
        s0_values.reshape(1, 1, -1),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
    )
    s0_map.coords["slice_idx"].attrs.update(dict(dce.coords["slice_idx"].attrs))

    t1_values = t1_aligned_ms if not misaligned_t1 else t1_aligned_ms[::-1]
    t1_axis_mm = np.array([0.0, 5.0], dtype=np.float64)
    if misaligned_t1:
        t1_axis_mm = t1_axis_mm[::-1]

    t1_map = xr.DataArray(
        t1_values.reshape(1, 1, -1),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )
    t1_map.coords["slice_idx"].attrs.update(
        {
            "unit": "mm",
            "resolution_mm": 5.0,
            "axis_mm": t1_axis_mm,
        }
    )

    aligned_t1_map = xr.DataArray(
        t1_aligned_ms.reshape(1, 1, -1),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )
    aligned_t1_map.coords["slice_idx"].attrs.update(dict(dce.coords["slice_idx"].attrs))

    return {
        "dce": dce,
        "s0_map": s0_map,
        "t1_map": t1_map,
        "aligned_t1_map": aligned_t1_map,
        "expected_concentration": expected_concentration.reshape(1, 1, 2, 2),
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "r1": r1,
    }


def _make_synthetic_extended_tofts_inputs():
    times_s = np.arange(10, dtype=np.float64) * 5.0
    aif_concentration = np.array(
        [0.0, 0.0, 0.35, 0.8, 0.6, 0.35, 0.18, 0.09, 0.04, 0.02],
        dtype=np.float64,
    )
    cp_func = interp1d(
        times_s,
        aif_concentration,
        kind="linear",
        fill_value="extrapolate",
    )
    ktrans = 0.16
    kep = 0.45
    vp = 0.08
    tissue_concentration = extended_tofts_model_2comp(
        times_s,
        ktrans,
        kep,
        vp,
        cp_func,
    )

    s0 = 125.0
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
        attrs={"repetition_time_ms": tr_s * 1000.0, "flipangle_deg": fa_deg},
    )
    t1_map = xr.DataArray(
        np.full((1, 1, 1), t10_s * 1000.0),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )

    return {
        "dce": dce,
        "fitted_t1": xr.Dataset({"t1": t1_map}),
        "aif_concentration": aif_concentration,
        "times_s": times_s,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "ktrans": ktrans,
        "kep": kep,
        "vp": vp,
    }


def _make_synthetic_full_sourbron_inputs():
    times_s = np.arange(12, dtype=np.float64) * 5.0
    aif_concentration = np.array(
        [0.0, 0.0, 0.25, 0.72, 0.98, 0.74, 0.48, 0.29, 0.17, 0.1, 0.06, 0.03],
        dtype=np.float64,
    )
    vp = 0.09
    ft = 0.11
    tt = 14.0
    tp = 4.0
    tissue_concentration = sourbron_separable_compartment_model(
        times_s,
        aif_concentration,
        plasma_volume_fraction=vp,
        tubular_flow_s=ft,
        tubular_transit_time_s=tt,
        plasma_transit_time_s=tp,
    )

    s0 = 118.0
    t10_s = 1.28
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
        attrs={"repetition_time_ms": tr_s * 1000.0, "flipangle_deg": fa_deg},
    )
    t1_map = xr.DataArray(
        np.full((1, 1, 1), t10_s * 1000.0),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )

    return {
        "dce": dce,
        "fitted_t1": xr.Dataset({"t1": t1_map}),
        "aif_concentration": aif_concentration,
        "times_s": times_s,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "vp": vp,
        "ft": ft,
        "tt": tt,
        "tp": tp,
    }


def _make_synthetic_two_c_irf_inputs():
    times_s = np.arange(12, dtype=np.float64) * 5.0
    aif_concentration = np.array(
        [0.0, 0.0, 0.22, 0.62, 0.95, 0.72, 0.46, 0.28, 0.18, 0.11, 0.07, 0.04],
        dtype=np.float64,
    )
    gfr_s = 0.018
    rpf_s = 0.032
    vascular_delay_s = 4.0
    vascular_dispersion_rate_s = 0.16
    tubular_delay_s = 8.0
    tubular_dispersion_rate_s = 0.08
    tissue_concentration = two_c_irf_model(
        times_s,
        aif_concentration,
        gfr_s=gfr_s,
        rpf_s=rpf_s,
        vascular_delay_s=vascular_delay_s,
        vascular_dispersion_rate_s=vascular_dispersion_rate_s,
        tubular_delay_s=tubular_delay_s,
        tubular_dispersion_rate_s=tubular_dispersion_rate_s,
    )

    s0 = 110.0
    t10_s = 1.25
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
        attrs={"repetition_time_ms": tr_s * 1000.0, "flipangle_deg": fa_deg},
    )
    t1_map = xr.DataArray(
        np.full((1, 1, 1), t10_s * 1000.0),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )

    return {
        "dce": dce,
        "fitted_t1": xr.Dataset({"t1": t1_map}),
        "aif_concentration": aif_concentration,
        "times_s": times_s,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "gfr_s": gfr_s,
        "rpf_s": rpf_s,
    }


def _make_synthetic_empirical_inputs():
    times_s = np.arange(12, dtype=np.float64) * 4.0
    bolus_arrival_s = 4.0
    rise_time_s = 10.0
    rise_shape = 5.0
    washout_rate_s = 0.045
    peak_amplitude = 0.9
    baseline_offset = 0.02

    tau = np.clip(times_s - bolus_arrival_s, 0.0, None)
    safe_tau = np.maximum(tau, np.finfo(np.float64).tiny)
    inverse_power = np.exp(
        np.clip(-rise_shape * np.log(safe_tau / rise_time_s), -700.0, 700.0)
    )
    rise = np.where(
        tau > 0.0,
        1.0 / (1.0 + inverse_power),
        0.0,
    )
    tissue_concentration = baseline_offset + peak_amplitude * rise * np.exp(
        -washout_rate_s * tau
    )

    s0 = 115.0
    t10_s = 1.3
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
            "time_s": ("repetition_idx", times_s),
        },
        name="dce_signal",
        attrs={"repetition_time_ms": tr_s * 1000.0, "flipangle_deg": fa_deg},
    )
    t1_map = xr.DataArray(
        np.full((1, 1, 1), t10_s * 1000.0),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )

    return {
        "dce": dce,
        "fitted_t1": xr.Dataset({"t1": t1_map}),
        "times_s": times_s,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "bolus_arrival_s": bolus_arrival_s,
        "rise_time_s": rise_time_s,
        "rise_shape": rise_shape,
        "washout_rate_s": washout_rate_s,
        "peak_amplitude": peak_amplitude,
        "baseline_offset": baseline_offset,
    }


def _make_masked_aif_pipeline_inputs():
    times_s = np.arange(8, dtype=np.float64) * 5.0
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

    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5
    aif_t10_s = 1.55
    tissue_t10_s = 1.35

    aif_signal = concentration_to_signal(
        aif_concentration,
        S_0=150.0,
        T10_s=aif_t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
    )
    tissue_signal = concentration_to_signal(
        tissue_concentration,
        S_0=120.0,
        T10_s=tissue_t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
    )

    signal = np.stack([aif_signal, tissue_signal], axis=0)[:, None, None, :]
    dce = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(signal.shape[0]),
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(signal.shape[-1], dtype=int),
        },
        name="dce_signal",
        attrs={"repetition_time_ms": tr_s * 1000.0, "flipangle_deg": fa_deg},
    )

    t1_map = xr.DataArray(
        np.array([[[aif_t10_s * 1000.0]], [[tissue_t10_s * 1000.0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="t1",
        attrs={"units": "ms"},
    )
    fitted_t1 = xr.Dataset(
        {
            "t1": t1_map,
            "rss_t1": xr.DataArray(
                np.array([[[0.25]], [[4.0]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
                coords=t1_map.coords,
                name="rss_t1",
            ),
        }
    )

    aif_mask = xr.DataArray(
        np.array([[[True]], [[False]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": dce.coords["read_idx"],
            "phase_idx": dce.coords["phase_idx"],
            "slice_idx": dce.coords["slice_idx"],
        },
        name="AIF",
    )

    return {
        "dce": dce,
        "fitted_t1": fitted_t1,
        "aif_mask": aif_mask,
        "aif_concentration": aif_concentration,
        "times_s": times_s,
        "tr_s": tr_s,
        "fa_deg": fa_deg,
        "tissue_ktrans": 0.18,
        "tissue_kep": 0.55,
    }


def test_validate_aif_rejects_mismatched_time_axis():
    with pytest.raises(ValueError, match="does not match"):
        validate_aif(
            aif_time_s=np.array([0.0, 1.0, 2.0]),
            aif_concentration=np.array([0.0, 0.1]),
        )


def test_validate_aif_rejects_non_monotonic_time_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_aif(
            aif_time_s=np.array([0.0, 1.0, 1.0]),
            aif_concentration=np.array([0.0, 0.1, 0.2]),
        )


def test_compute_s0_map_uses_precontrast_frames_only():
    data = xr.DataArray(
        np.array([[[[10.0, 12.0, 80.0, 90.0]]]]),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map, _ = compute_s0_map(data, n_precontrast=2)
    assert np.isclose(float(s0_map.values.squeeze()), 11.0)


def test_standardize_t1_map_units_converts_seconds_to_ms():
    t1_map_s = xr.DataArray(
        np.array([[[1.2]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
        attrs={"units": "s"},
    )
    t1_map_ms = standardize_t1_map_units(t1_map_s)
    assert np.isclose(float(t1_map_ms.values.squeeze()), 1200.0)
    assert t1_map_ms.attrs["units"] == "ms"


def test_convert_signal_to_concentration_round_trip():
    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5
    t10_s = 1.2
    s0 = 100.0
    concentration = np.array([0.0, 0.05, 0.12, 0.08], dtype=np.float64)

    signal = concentration_to_signal(
        concentration,
        S_0=s0,
        T10_s=t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
    )
    signal_da = xr.DataArray(
        signal.reshape(1, 1, 1, -1),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map = xr.DataArray(
        np.array([[[s0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    t1_map = xr.DataArray(
        np.array([[[t10_s * 1000.0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    recovered = convert_signal_to_concentration_map(
        signal_da,
        s0_map,
        t1_map,
        tr_s=tr_s,
        fa_deg=fa_deg,
        r1=r1,
        time_points_s=np.arange(len(concentration), dtype=np.float64),
    )

    assert np.allclose(
        recovered.values.squeeze(),
        concentration,
        atol=1e-6,
        equal_nan=False,
    )


def test_convert_signal_to_concentration_map_skips_t1_realign_when_grids_match(
    monkeypatch,
):
    case = _make_signal_conversion_alignment_inputs(misaligned_t1=False)

    def _unexpected_align(*args, **kwargs):
        raise AssertionError("align_t1_to_dce_grid should not be called")

    monkeypatch.setattr(fitting_support, "align_t1_to_dce_grid", _unexpected_align)

    recovered = convert_signal_to_concentration_map(
        case["dce"],
        case["s0_map"],
        case["t1_map"],
        tr_s=case["tr_s"],
        fa_deg=case["fa_deg"],
        r1=case["r1"],
    )

    assert np.allclose(
        recovered.values,
        case["expected_concentration"],
        atol=1e-6,
        equal_nan=False,
    )


def test_convert_signal_to_concentration_map_realigns_t1_when_slice_grid_differs(
    monkeypatch,
):
    case = _make_signal_conversion_alignment_inputs(misaligned_t1=True)
    calls = []

    def _fake_align(dce_data, t1_data=None, **kwargs):
        calls.append(
            {
                "dce_data": dce_data,
                "t1_data": t1_data,
                "kwargs": kwargs,
            }
        )
        return case["aligned_t1_map"], None

    monkeypatch.setattr(fitting_support, "align_t1_to_dce_grid", _fake_align)

    recovered = convert_signal_to_concentration_map(
        case["dce"],
        case["s0_map"],
        case["t1_map"],
        tr_s=case["tr_s"],
        fa_deg=case["fa_deg"],
        r1=case["r1"],
    )

    assert len(calls) == 1
    assert calls[0]["dce_data"] is case["dce"]
    assert calls[0]["t1_data"] is None
    assert calls[0]["kwargs"]["fitted_t1"] is case["t1_map"]
    assert calls[0]["kwargs"]["use_multiprocessing"] is False
    assert np.allclose(
        recovered.values,
        case["expected_concentration"],
        atol=1e-6,
        equal_nan=False,
    )


def test_convert_signal_to_concentration_map_preserves_dce_coordinate_attrs():
    case = _make_signal_conversion_alignment_inputs(misaligned_t1=False)

    recovered = convert_signal_to_concentration_map(
        case["dce"],
        case["s0_map"],
        case["t1_map"],
        tr_s=case["tr_s"],
        fa_deg=case["fa_deg"],
        r1=case["r1"],
        time_points_s=np.array([0.0, 5.0], dtype=np.float64),
    )

    assert recovered.coords["slice_idx"].attrs == case["dce"].coords["slice_idx"].attrs
    assert (
        recovered.coords["repetition_idx"].attrs
        == case["dce"].coords["repetition_idx"].attrs
    )


def test_signal_to_concentration_reports_spgr_saturation():
    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5
    t10_s = 1.2
    s0 = 100.0
    theta = np.deg2rad(fa_deg)
    e10 = np.exp(-tr_s / t10_s)
    b = (1.0 - e10) / (1.0 - np.cos(theta) * e10)
    max_s_rel = 0.98 / b
    signal = np.array([s0, s0 * max_s_rel * 1.01], dtype=np.float64)

    recovered, saturation_info = signal_to_concentration(
        S_t=signal,
        S_0=s0,
        T10_s=t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
        return_saturation_info=True,
    )

    assert saturation_info["saturation_detected"] is True
    assert saturation_info["saturation_count"] == 1
    assert saturation_info["saturation_mask"].tolist() == [False, True]
    assert np.all(np.isfinite(recovered))


def test_signal_to_concentration_can_invalidate_entire_curve_on_saturation():
    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5
    t10_s = 1.2
    s0 = 100.0
    theta = np.deg2rad(fa_deg)
    e10 = np.exp(-tr_s / t10_s)
    b = (1.0 - e10) / (1.0 - np.cos(theta) * e10)
    max_s_rel = 0.98 / b
    signal = np.array([s0, s0 * max_s_rel * 1.01], dtype=np.float64)

    recovered, saturation_info = signal_to_concentration(
        S_t=signal,
        S_0=s0,
        T10_s=t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
        saturation_behavior="nan_all",
        return_saturation_info=True,
    )

    assert saturation_info["saturation_detected"] is True
    assert np.isnan(recovered).all()


def test_convert_signal_to_concentration_supports_signal_enhancement_mode():
    signal = xr.DataArray(
        np.array([[[[10.0, 11.5, 13.0, 9.5]]]]),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map = xr.DataArray(
        np.array([[[10.0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    t1_map = xr.DataArray(
        np.array([[[1200.0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    recovered = convert_signal_to_concentration_map(
        signal,
        s0_map,
        t1_map,
        tr_s=0.05,
        fa_deg=15.0,
        r1=3.5,
        mode="signal_enhancement",
        linear_scale=2.0,
    )

    assert np.allclose(
        recovered.values.squeeze(),
        np.array([0.0, 3.0, 6.0, -1.0], dtype=np.float64),
        atol=1e-12,
    )
    assert recovered.attrs["signal_to_concentration_mode"] == "signal_enhancement"
    assert np.isclose(recovered.attrs["signal_to_concentration_scale"], 2.0)


def test_convert_signal_to_concentration_map_records_saturation_metadata():
    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5
    t10_s = 1.2
    s0 = 100.0
    theta = np.deg2rad(fa_deg)
    e10 = np.exp(-tr_s / t10_s)
    b = (1.0 - e10) / (1.0 - np.cos(theta) * e10)
    max_s_rel = 0.98 / b

    signal = xr.DataArray(
        np.array([[[[s0, s0 * max_s_rel * 1.01]]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map = xr.DataArray(
        np.array([[[s0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    t1_map = xr.DataArray(
        np.array([[[t10_s * 1000.0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    recovered = convert_signal_to_concentration_map(
        signal,
        s0_map,
        t1_map,
        tr_s=tr_s,
        fa_deg=fa_deg,
        r1=r1,
        saturation_behavior="nan",
    )

    assert recovered.attrs["signal_saturation_detected"] is True
    assert recovered.attrs["signal_saturation_count"] == 1
    assert np.isclose(recovered.attrs["signal_saturation_fraction"], 0.5)
    assert recovered.attrs["signal_saturation_behavior"] == "nan"
    assert np.isnan(recovered.values[0, 0, 0, 1])


def test_convert_signal_to_concentration_map_nan_all_is_applied_per_voxel_curve():
    tr_s = 0.05
    fa_deg = 15.0
    r1 = 3.5
    t10_s = 1.2
    s0 = 100.0
    theta = np.deg2rad(fa_deg)
    e10 = np.exp(-tr_s / t10_s)
    b = (1.0 - e10) / (1.0 - np.cos(theta) * e10)
    max_s_rel = 0.98 / b

    safe_concentration = np.array([0.0, 0.05], dtype=np.float64)
    safe_signal = concentration_to_signal(
        safe_concentration,
        S_0=s0,
        T10_s=t10_s,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
    )
    saturated_signal = np.array([s0, s0 * max_s_rel * 1.01], dtype=np.float64)

    signal = xr.DataArray(
        np.array(
            [
                [[[safe_signal[0], safe_signal[1]]]],
                [[[saturated_signal[0], saturated_signal[1]]]],
            ],
            dtype=np.float64,
        ),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map = xr.DataArray(
        np.array([[[s0]], [[s0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    t1_map = xr.DataArray(
        np.array([[[t10_s * 1000.0]], [[t10_s * 1000.0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    recovered = convert_signal_to_concentration_map(
        signal,
        s0_map,
        t1_map,
        tr_s=tr_s,
        fa_deg=fa_deg,
        r1=r1,
        saturation_behavior="nan_all",
    )

    assert np.allclose(
        recovered.values[0, 0, 0, :],
        safe_concentration,
        atol=1e-6,
        equal_nan=False,
    )
    assert np.isnan(recovered.values[1, 0, 0, :]).all()


def test_convert_signal_to_concentration_supports_relative_signal_enhancement_mode():
    signal = xr.DataArray(
        np.array([[[[10.0, 11.5, 13.0, 9.5]]]]),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map = xr.DataArray(
        np.array([[[10.0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    t1_map = xr.DataArray(
        np.array([[[1200.0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )

    recovered = convert_signal_to_concentration_map(
        signal,
        s0_map,
        t1_map,
        tr_s=0.05,
        fa_deg=15.0,
        r1=3.5,
        mode="relative_signal_enhancement",
        linear_scale=4.0,
    )

    assert np.allclose(
        recovered.values.squeeze(),
        np.array([0.0, 0.6, 1.2, -0.2], dtype=np.float64),
        atol=1e-12,
    )
    assert (
        recovered.attrs["signal_to_concentration_mode"] == "relative_signal_enhancement"
    )
    assert np.isclose(recovered.attrs["signal_to_concentration_scale"], 4.0)


def test_relative_signal_enhancement_supports_eq8_auto_scale():
    signal = xr.DataArray(
        np.array([[[[10.0, 11.5, 13.0, 9.5]]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map = xr.DataArray(
        np.array([[[10.0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    t1_map = xr.DataArray(
        np.array([[[1200.0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    r1 = 3.5
    expected = (
        (signal - s0_map.broadcast_like(signal)) / s0_map.broadcast_like(signal)
    ) / (r1 * (t1_map.broadcast_like(signal) / 1000.0))

    recovered = convert_signal_to_concentration_map(
        signal,
        s0_map,
        t1_map,
        tr_s=0.05,
        fa_deg=15.0,
        r1=r1,
        mode="relative_signal_enhancement",
        linear_scale="eq8",
    )

    assert np.allclose(recovered.values, expected.values, atol=1e-12)
    assert recovered.attrs["signal_to_concentration_scale"] == "eq8"


def test_signal_enhancement_supports_eq8_auto_scale():
    signal = xr.DataArray(
        np.array([[[[10.0, 11.5, 13.0, 9.5]]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    s0_map = xr.DataArray(
        np.array([[[10.0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    t1_map = xr.DataArray(
        np.array([[[1200.0]]], dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    r1 = 3.5
    expected = (signal - s0_map.broadcast_like(signal)) / (
        s0_map.broadcast_like(signal) * r1 * (t1_map.broadcast_like(signal) / 1000.0)
    )

    recovered = convert_signal_to_concentration_map(
        signal,
        s0_map,
        t1_map,
        tr_s=0.05,
        fa_deg=15.0,
        r1=r1,
        mode="signal_enhancement",
        linear_scale="eq8",
    )

    assert np.allclose(recovered.values, expected.values, atol=1e-12)
    assert recovered.attrs["signal_to_concentration_scale"] == "eq8"


def test_select_fit_window_supports_standardized_and_legacy_end_names():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask_index = select_fit_window(
        times,
        {"fit_start_index": 1, "fit_end_index": 4},
    )
    assert np.array_equal(mask_index.values, np.array([False, True, True, True, False]))

    mask_legacy = select_fit_window(
        times,
        {"fit_start_time_s": 1.5, "fit_stop_time_s": 3.2},
    )
    assert np.array_equal(
        mask_legacy.values, np.array([False, False, True, True, False])
    )


def test_fit_patlak_model_recovers_synthetic_parameters():
    synthetic = _make_synthetic_renal_inputs()
    concentration_map = xr.DataArray(
        synthetic["tissue_concentration"].reshape(2, 1, 1, -1),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(2),
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(len(synthetic["times_s"])),
        },
    )

    fit_result = fit_patlak_model(
        concentration_map,
        aif_concentration=synthetic["aif_concentration"],
        aif_time_s=synthetic["times_s"],
        fit_params={"fit_start_index": 0},
        use_multiprocessing=False,
    )

    assert np.isclose(
        float(fit_result["K_trans"].isel(read_idx=0, phase_idx=0, slice_idx=0)),
        synthetic["slopes"][0],
        atol=5e-4,
    )
    assert np.isclose(
        float(fit_result["Vp"].isel(read_idx=1, phase_idx=0, slice_idx=0)),
        synthetic["intercepts"][1],
        atol=5e-4,
    )


def test_run_renal_dce_pipeline_preserves_dims_and_adds_aif_qc_variables():
    synthetic = _make_synthetic_renal_inputs()
    result = run_renal_dce_pipeline(
        synthetic["dce"],
        fitted_t1=synthetic["fitted_t1"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        fit_type="patlak",
        fit_params_pk={"n_precontrast": 1},
        use_multiprocessing=False,
    )

    assert result["t1_map"].dims == ("read_idx", "phase_idx", "slice_idx")
    assert result["s0_map"].dims == ("read_idx", "phase_idx", "slice_idx")
    assert result["concentration_map"].dims == (
        "read_idx",
        "phase_idx",
        "slice_idx",
        "repetition_idx",
    )
    assert "aif_signal_raw" in result
    assert "aif_concentration" in result
    assert "aif_time_s" in result
    assert "patlak_slope" in result
    assert "filtration_index" in result
    assert np.isclose(
        float(result["patlak_slope"].isel(read_idx=0, phase_idx=0, slice_idx=0)),
        synthetic["slopes"][0],
        atol=2e-3,
    )


def test_run_dce_pipeline_uses_explicit_r1_for_signal_conversion():
    """The ``r1`` argument reaches the signal-to-concentration conversion.

    Synthetic Patlak tissue curves are converted to SPGR signal with
    r1 = 4.5 1/(mM s) and fitted with the same r1; the AIF is given as
    concentration, so r1 only enters the tissue conversion.  The fixture is
    noise-free and the linear Patlak fit recovers the slope to about 1e-10 1/s.
    The tolerance of 2e-3 1/s matches the renal Patlak pipeline test on this
    six-point grid and is four times smaller than the bias the test guards
    against: before the fix the argument was ignored, r1 = 3.5 was used, and
    K_trans was 0.0386 1/s instead of 0.0300 1/s (factor 4.5 / 3.5).
    """
    r1 = 4.5
    synthetic = _make_synthetic_renal_inputs(r1=r1)

    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        r1=r1,
        fit_type="patlak",
        fit_params_pk={"n_precontrast": 1},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["r1"] == pytest.approx(r1)
    assert float(
        result["K_trans"].isel(read_idx=0, phase_idx=0, slice_idx=0)
    ) == pytest.approx(synthetic["slopes"][0], abs=2e-3)


def test_run_dce_pipeline_rejects_conflicting_r1():
    """``r1`` and ``fit_params_pk["r1"]`` must not disagree."""
    synthetic = _make_synthetic_renal_inputs(r1=4.5)

    with pytest.raises(ValueError, match=r"r1=4\.5 and fit_params_pk\['r1'\]=3\.5"):
        run_dce_pipeline(
            dce_data=synthetic["dce"],
            aif_signal_concentration=synthetic["aif_concentration"],
            aif_times=synthetic["times_s"],
            tr_s=synthetic["tr_s"],
            fa_deg=synthetic["fa_deg"],
            r1=4.5,
            fit_type="patlak",
            fit_params_pk={"n_precontrast": 1, "r1": 3.5},
            fitted_t1=synthetic["fitted_t1"],
            use_multiprocessing=False,
        )


def test_run_renal_dce_pipeline_reads_r1_from_fit_params_without_argument():
    """Without an ``r1`` argument the renal pipeline uses ``fit_params_pk``.

    This guards the renal default: a numeric default for the ``r1`` argument
    would now take precedence over ``fit_params_pk["r1"]`` or raise a conflict.
    Fixture and tolerance as in
    ``test_run_dce_pipeline_uses_explicit_r1_for_signal_conversion``.
    """
    r1 = 4.5
    synthetic = _make_synthetic_renal_inputs(r1=r1)

    result = run_renal_dce_pipeline(
        synthetic["dce"],
        fitted_t1=synthetic["fitted_t1"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        fit_type="patlak",
        fit_params_pk={"n_precontrast": 1, "r1": r1},
        use_multiprocessing=False,
    )

    assert result.attrs["r1"] == pytest.approx(r1)
    assert float(
        result["patlak_slope"].isel(read_idx=0, phase_idx=0, slice_idx=0)
    ) == pytest.approx(synthetic["slopes"][0], abs=2e-3)


def test_run_dce_pipeline_interprets_legacy_patlak_model_mode():
    synthetic = _make_synthetic_renal_inputs()
    with pytest.warns(UserWarning, match="legacy fit selector"):
        result = run_dce_pipeline(
            dce_data=synthetic["dce"],
            aif_signal_concentration=synthetic["aif_concentration"],
            aif_times=synthetic["times_s"],
            tr_s=synthetic["tr_s"],
            fa_deg=synthetic["fa_deg"],
            fit_params_tofts={"model_mode": "patlak", "n_precontrast": 1},
            fitted_t1=synthetic["fitted_t1"],
            use_multiprocessing=False,
        )

    assert result.attrs["primary_fit_type"] == "patlak"
    assert result.attrs["requested_fit_type"] == "tofts"
    assert result.attrs["requested_model_mode"] == "patlak"
    assert result.attrs["effective_fit_type"] == "patlak"
    assert result.attrs["effective_model_mode"] == "patlak"
    assert (
        result.attrs["fit_function"] == "hypermri.analysis.dce.fitting.fit_patlak_model"
    )
    assert result.attrs["fit_backend_fit_type"] == "patlak"
    assert "patlak_slope" not in result
    assert "Ktrans_1c" not in result
    assert np.isclose(
        float(result["K_trans"].isel(read_idx=0, phase_idx=0, slice_idx=0)),
        synthetic["slopes"][0],
        atol=2e-3,
    )


def test_fit_sourbron_model_recovers_extended_tofts_equivalent_parameters():
    synthetic = _make_synthetic_extended_tofts_inputs()
    fit_params = {
        "fit_start_index": 0,
        "x0": [synthetic["vp"], synthetic["ktrans"], 1.0 / synthetic["kep"]],
        "lower_bounds": [0.0, 0.0, 0.2],
    }
    s0, _ = compute_s0_map(synthetic["dce"], n_precontrast=2)
    concentration_map = convert_signal_to_concentration_map(
        synthetic["dce"],
        s0,
        synthetic["fitted_t1"]["t1"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        time_points_s=synthetic["times_s"],
    )
    fit_result = fit_sourbron_model(
        concentration_map,
        aif_concentration=synthetic["aif_concentration"],
        aif_time_s=synthetic["times_s"],
        fit_params=fit_params,
        use_multiprocessing=False,
    )

    assert np.isclose(
        float(fit_result["Ft"].values.squeeze()),
        synthetic["ktrans"],
        atol=3e-2,
    )
    assert np.isclose(
        float(fit_result["vp"].values.squeeze()),
        synthetic["vp"],
        atol=3e-2,
    )
    assert np.isclose(
        float(fit_result["Tt"].values.squeeze()),
        1.0 / synthetic["kep"],
        atol=2.0,
    )
    assert np.isclose(
        float(fit_result["Ktrans_extended"].values.squeeze()),
        synthetic["ktrans"],
        atol=3e-2,
    )


def test_fit_extended_tofts_model_is_invariant_to_common_concentration_scaling():
    synthetic = _make_synthetic_extended_tofts_inputs()
    s0, _ = compute_s0_map(synthetic["dce"], n_precontrast=2)
    concentration_map = convert_signal_to_concentration_map(
        synthetic["dce"],
        s0,
        synthetic["fitted_t1"]["t1"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        time_points_s=synthetic["times_s"],
    )
    fit_params = {"fit_start_index": 0}

    reference = fit_extended_tofts_model(
        concentration_map,
        aif_concentration=synthetic["aif_concentration"],
        aif_time_s=synthetic["times_s"],
        fit_params=fit_params,
        use_multiprocessing=False,
    )

    scale = 7.0
    scaled = fit_extended_tofts_model(
        concentration_map * scale,
        aif_concentration=synthetic["aif_concentration"] * scale,
        aif_time_s=synthetic["times_s"],
        fit_params=fit_params,
        use_multiprocessing=False,
    )

    for var_name in (
        "Ktrans_extended",
        "kep_extended",
        "vp_plasma",
        "ve_extended",
        "Ft",
        "Tt",
        "tubular_flow_ml_min_100ml",
    ):
        assert np.isclose(
            float(reference[var_name].values.squeeze()),
            float(scaled[var_name].values.squeeze()),
            atol=1e-6,
        )


def test_fit_sourbron_model_recovers_tp_when_enabled():
    synthetic = _make_synthetic_full_sourbron_inputs()
    fit_params = {
        "fit_start_index": 0,
        "estimate_tp": True,
        "x0": [synthetic["vp"], synthetic["ft"], synthetic["tt"], synthetic["tp"]],
        "lower_bounds": [0.0, 0.0, 2.0, 1.0],
    }
    s0, _ = compute_s0_map(synthetic["dce"], n_precontrast=2)
    concentration_map = convert_signal_to_concentration_map(
        synthetic["dce"],
        s0,
        synthetic["fitted_t1"]["t1"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        time_points_s=synthetic["times_s"],
    )
    fit_result = fit_sourbron_model(
        concentration_map,
        aif_concentration=synthetic["aif_concentration"],
        aif_time_s=synthetic["times_s"],
        fit_params=fit_params,
        use_multiprocessing=False,
    )

    assert "Tp" in fit_result
    assert "Fp" in fit_result
    assert "extraction_fraction" in fit_result
    assert np.isclose(
        float(fit_result["vp"].values.squeeze()), synthetic["vp"], atol=4e-2
    )
    assert np.isclose(
        float(fit_result["Ft"].values.squeeze()), synthetic["ft"], atol=4e-2
    )
    assert np.isclose(
        float(fit_result["Tt"].values.squeeze()), synthetic["tt"], atol=4.0
    )
    assert np.isclose(
        float(fit_result["Tp"].values.squeeze()), synthetic["tp"], atol=2.0
    )

    reconstructed = fit_result.hp.reconstruct()
    assert reconstructed.dims == concentration_map.dims
    assert np.allclose(reconstructed.values, concentration_map.values, atol=2e-2)


def test_run_dce_pipeline_rejects_removed_sourbron_model_mode():
    synthetic = _make_synthetic_extended_tofts_inputs()
    with pytest.raises(
        ValueError,
        match="fit_params_tofts\\['model_mode'\\] = 'sourbron' is no longer supported",
    ):
        run_dce_pipeline(
            dce_data=synthetic["dce"],
            aif_signal_concentration=synthetic["aif_concentration"],
            aif_times=synthetic["times_s"],
            tr_s=synthetic["tr_s"],
            fa_deg=synthetic["fa_deg"],
            fit_params_tofts={"model_mode": "sourbron", "n_precontrast": 2},
            fitted_t1=synthetic["fitted_t1"],
            use_multiprocessing=False,
        )


def test_run_dce_pipeline_accepts_sourbron_fit_type():
    synthetic = _make_synthetic_extended_tofts_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="sourbron",
        fit_params_pk={"n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["requested_fit_type"] == "sourbron"
    assert result.attrs["effective_fit_type"] == "sourbron"
    assert result.attrs["effective_model_mode"] == "sourbron"
    assert "vp" in result
    assert "Ft" in result
    assert "Tt" in result
    assert "Ktrans_extended" in result
    np.testing.assert_allclose(
        result["ve_equivalent"],
        result["Ft"] * result["Tt"],
        rtol=1e-12,
        atol=1e-14,
    )
    assert result["ve_equivalent"].attrs["units"] == "1"
    assert result["ve_was_clipped"].dtype == bool
    assert np.isclose(
        float(result["Ft"].values.squeeze()),
        float(result["Ktrans_extended"].values.squeeze()),
        atol=1e-10,
    )
    assert np.isclose(
        float(result["Tt"].values.squeeze()),
        float(result["tubular_transit_time_s"].values.squeeze()),
        atol=1e-10,
    )
    assert np.isclose(
        float(result["vp"].values.squeeze()),
        float(result["vp_plasma"].values.squeeze()),
        atol=1e-10,
    )


def test_run_dce_pipeline_accepts_sourbron_tp_variant_and_can_skip_quality_metrics():
    synthetic = _make_synthetic_full_sourbron_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="sourbron",
        fit_params_pk={
            "n_precontrast": 2,
            "estimate_tp": True,
            "x0": [synthetic["vp"], synthetic["ft"], synthetic["tt"], synthetic["tp"]],
            "lower_bounds": [0.0, 0.0, 2.0, 1.0],
        },
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
        skip_fit_quality_metrics=True,
    )

    assert result.attrs["requested_fit_type"] == "sourbron"
    assert result.attrs["effective_fit_type"] == "sourbron"
    assert result.attrs["fit_quality_metrics_skipped"] is True
    assert result.attrs["estimate_tp"] is True
    assert result.attrs["sourbron_model_variant"] == "full_with_tp"
    assert result.attrs["sourbron_convolution_method"] == "oversampled_trapezoidal"
    assert result.attrs["sourbron_integration_max_step_s"] == 0.1
    assert result.attrs["sourbron_aif_interpolation"] == "linear"
    assert "Tp" in result
    assert "Fp" in result
    assert "plasma_flow_ml_min_100ml" in result
    assert "aic" not in result
    assert "converged" not in result
    assert "equation_equivalent_fit_types" not in result.attrs


def test_run_dce_pipeline_accepts_aif_mask():
    synthetic = _make_masked_aif_pipeline_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_mask=synthetic["aif_mask"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={
            "model_mode": "1comp",
            "n_precontrast": 2,
            "aif_convert_blood_to_plasma": False,
        },
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert np.allclose(
        np.asarray(result["aif_concentration"].values, dtype=np.float64),
        synthetic["aif_concentration"],
        atol=1e-6,
    )
    assert np.isclose(
        float(result["Ktrans_1c"].isel(read_idx=1, phase_idx=0, slice_idx=0)),
        synthetic["tissue_ktrans"],
        atol=2e-2,
    )
    assert np.isclose(
        float(result["kep_1c"].isel(read_idx=1, phase_idx=0, slice_idx=0)),
        synthetic["tissue_kep"],
        atol=5e-2,
    )
    assert result.attrs["aif_t1_weighting"] == "inverse_rss_t1"
    assert np.isclose(result.attrs["aif_t1_ms_used"], 1550.0)


def test_prepare_aif_uses_fractional_mask_weights_for_signal_and_t1_summary():
    synthetic = _make_masked_aif_pipeline_inputs()
    weighted_mask = xr.DataArray(
        np.array([[[0.75]], [[0.25]]], dtype=np.float32),
        dims=synthetic["aif_mask"].dims,
        coords=synthetic["aif_mask"].coords,
        name="weighted_aif",
    )

    aif_info = prepare_aif(
        synthetic["dce"],
        synthetic["fitted_t1"]["t1"],
        aif_times=synthetic["times_s"],
        aif_mask=weighted_mask,
        n_precontrast=2,
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        r1=3.5,
        aif_t1_weight_map=synthetic["fitted_t1"]["rss_t1"],
        fit_params={"aif_convert_blood_to_plasma": False},
    )

    signal = np.asarray(synthetic["dce"].values, dtype=np.float64)
    expected_raw = 0.75 * signal[0, 0, 0, :] + 0.25 * signal[1, 0, 0, :]
    expected_t1_ms = ((0.75 / 0.25) * 1550.0 + (0.25 / 4.0) * 1350.0) / (
        (0.75 / 0.25) + (0.25 / 4.0)
    )

    assert np.allclose(
        np.asarray(aif_info["aif_signal_raw"], dtype=np.float64),
        expected_raw,
        atol=1e-10,
    )
    assert np.isclose(float(aif_info["aif_t1_ms_used"]), expected_t1_ms, atol=1e-10)
    assert np.isclose(float(aif_info["aif_mask_voxel_count"]), 1.0)


def test_run_dce_pipeline_defaults_raw_aif_to_plasma_with_default_hematocrit():
    synthetic = _make_masked_aif_pipeline_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_mask=synthetic["aif_mask"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={"model_mode": "1comp", "n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    expected_plasma_aif = synthetic["aif_concentration"] / (1.0 - 0.41)
    assert np.allclose(
        np.asarray(result["aif_concentration"].values, dtype=np.float64),
        expected_plasma_aif,
        atol=1e-6,
    )
    assert result.attrs["aif_concentration_kind"] == "plasma"
    assert bool(result.attrs["aif_convert_blood_to_plasma"]) is True
    assert np.isclose(float(result.attrs["aif_hematocrit"]), 0.41)


def test_run_dce_pipeline_can_convert_raw_aif_from_blood_to_plasma():
    synthetic = _make_masked_aif_pipeline_inputs()
    hematocrit = 0.42
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_mask=synthetic["aif_mask"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={
            "model_mode": "1comp",
            "n_precontrast": 2,
            "aif_convert_blood_to_plasma": True,
            "aif_hematocrit": hematocrit,
        },
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    expected_plasma_aif = synthetic["aif_concentration"] / (1.0 - hematocrit)
    assert np.allclose(
        np.asarray(result["aif_concentration"].values, dtype=np.float64),
        expected_plasma_aif,
        atol=1e-6,
    )
    assert result.attrs["aif_concentration_kind"] == "plasma"
    assert bool(result.attrs["aif_convert_blood_to_plasma"]) is True
    assert np.isclose(float(result.attrs["aif_hematocrit"]), hematocrit)


def test_run_dce_pipeline_top_level_aif_options_override_fit_params():
    synthetic = _make_masked_aif_pipeline_inputs()
    hematocrit = 0.37
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_mask=synthetic["aif_mask"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={
            "model_mode": "1comp",
            "n_precontrast": 2,
            "aif_convert_blood_to_plasma": False,
        },
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
        blood_to_plasma=True,
        hematocrit=hematocrit,
    )

    expected_plasma_aif = synthetic["aif_concentration"] / (1.0 - hematocrit)
    assert np.allclose(
        np.asarray(result["aif_concentration"].values, dtype=np.float64),
        expected_plasma_aif,
        atol=1e-6,
    )
    assert bool(result.attrs["aif_convert_blood_to_plasma"]) is True
    assert np.isclose(float(result.attrs["aif_hematocrit"]), hematocrit)


def test_run_dce_pipeline_applies_relative_signal_enhancement_to_tissue_and_aif():
    synthetic = _make_masked_aif_pipeline_inputs()
    scale = 3.0
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_mask=synthetic["aif_mask"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={
            "model_mode": "1comp",
            "n_precontrast": 2,
            "signal_to_concentration_mode": "relative_signal_enhancement",
            "signal_to_concentration_scale": scale,
            "aif_convert_blood_to_plasma": False,
        },
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    signal = np.asarray(synthetic["dce"].values, dtype=np.float64)
    aif_signal = signal[0, 0, 0, :]
    tissue_signal = signal[1, 0, 0, :]
    expected_aif = scale * (aif_signal - aif_signal[0]) / aif_signal[0]
    expected_tissue = scale * (tissue_signal - tissue_signal[0]) / tissue_signal[0]

    assert np.allclose(
        np.asarray(result["aif_concentration"].values, dtype=np.float64),
        expected_aif,
        atol=1e-12,
    )
    assert np.allclose(
        np.asarray(
            result["concentration_map"]
            .isel(read_idx=1, phase_idx=0, slice_idx=0)
            .values,
            dtype=np.float64,
        ),
        expected_tissue,
        atol=1e-12,
    )
    assert result.attrs["signal_to_concentration_mode"] == "relative_signal_enhancement"
    assert np.isclose(float(result.attrs["signal_to_concentration_scale"]), scale)
    assert (
        result.attrs["aif_signal_to_concentration_mode"]
        == "relative_signal_enhancement"
    )
    assert np.isclose(float(result.attrs["aif_signal_to_concentration_scale"]), scale)


def test_run_dce_pipeline_unwraps_single_dict_tuple_fit_params():
    synthetic = _make_synthetic_renal_inputs()
    with pytest.warns(UserWarning, match="single-element tuple containing a dict"):
        result = run_dce_pipeline(
            dce_data=synthetic["dce"],
            aif_signal_concentration=synthetic["aif_concentration"],
            aif_times=synthetic["times_s"],
            tr_s=synthetic["tr_s"],
            fa_deg=synthetic["fa_deg"],
            fit_type="patlak",
            fit_params_pk=({"fit_start_index": 0, "n_precontrast": 1},),
            fitted_t1=synthetic["fitted_t1"],
            use_multiprocessing=False,
        )

    assert result.attrs["primary_fit_type"] == "patlak"
    assert np.isclose(
        float(result["K_trans"].isel(read_idx=0, phase_idx=0, slice_idx=0)),
        synthetic["slopes"][0],
        atol=2e-3,
    )


def test_run_dce_pipeline_tofts_1comp_retains_named_qc_dims():
    synthetic = _make_synthetic_tofts_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={"model_mode": "1comp", "n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["primary_fit_type"] == "tofts"
    assert result.attrs["effective_fit_type"] == "tofts"
    assert result.attrs["fit_model_mode"] == "1comp"
    assert result.attrs["effective_model_mode"] == "1comp"
    assert (
        result.attrs["fit_function"] == "hypermri.analysis.dce.fitting.fit_tofts_model"
    )
    assert result.attrs["fit_backend_fit_type"] == "tofts"
    assert result.attrs["fit_input_space"] == "concentration"
    assert result["hit_bound"].dims == ("read_idx", "phase_idx", "slice_idx")
    assert result["failed"].dims == ("read_idx", "phase_idx", "slice_idx")
    assert result["fit_window_mask"].dims == ("repetition_idx",)


def test_run_dce_pipeline_accepts_standard_tofts_alias():
    synthetic = _make_synthetic_tofts_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="standard_tofts",
        fit_params_pk={"n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["requested_fit_type"] == "standard_tofts"
    assert result.attrs["effective_fit_type"] == "tofts"
    assert result.attrs["effective_model_mode"] == "1comp"
    assert "Ktrans_standard" in result
    assert "ve_standard" in result
    assert np.isclose(
        float(result["Ktrans_standard"].values.squeeze()),
        0.18,
        atol=2e-2,
    )


def test_run_dce_pipeline_accepts_extended_tofts_alias():
    synthetic = _make_synthetic_extended_tofts_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="extended_tofts_plasma",
        fit_params_pk={"n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["requested_fit_type"] == "extended_tofts_plasma"
    np.testing.assert_allclose(
        result["ve_equivalent"],
        result["Ktrans_2c"] / result["kep_2c"],
        rtol=1e-12,
        atol=1e-14,
    )
    assert result["ve_equivalent"].attrs["units"] == "1"
    assert result["ve_was_clipped"].dtype == bool
    assert result.attrs["effective_fit_type"] == "tofts"
    assert result.attrs["effective_model_mode"] == "2comp"
    assert "Ktrans_extended" in result
    assert "vp_plasma" in result
    assert "Ft" in result
    assert "Tt" in result
    assert "tubular_flow_ml_min_100ml" in result
    assert np.isclose(
        float(result["Ktrans_extended"].values.squeeze()),
        synthetic["ktrans"],
        atol=3e-2,
    )
    assert np.isclose(
        float(result["vp_plasma"].values.squeeze()),
        synthetic["vp"],
        atol=3e-2,
    )
    assert (
        result.attrs["equation_equivalent_fit_types"]
        == "extended_tofts_plasma,kidney_two_compartment_outflow,sourbron"
    )
    assert np.isclose(
        float(result["Ft"].values.squeeze()),
        synthetic["ktrans"],
        atol=3e-2,
    )
    assert np.isclose(
        float(result["Tt"].values.squeeze()),
        1.0 / synthetic["kep"],
        atol=2.0,
    )


def test_run_renal_dce_pipeline_accepts_sourbron_fit_type():
    synthetic = _make_synthetic_extended_tofts_inputs()
    result = run_renal_dce_pipeline(
        synthetic["dce"],
        fitted_t1=synthetic["fitted_t1"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="sourbron",
        fit_params_pk={"n_precontrast": 2},
        use_multiprocessing=False,
    )

    assert result.attrs["requested_fit_type"] == "sourbron"
    assert result.attrs["effective_fit_type"] == "sourbron"
    assert "Ft" in result
    assert "Tt" in result


def test_run_dce_pipeline_prefers_fit_params_pk_model_mode_over_fit_params_tofts():
    synthetic = _make_synthetic_extended_tofts_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="tofts",
        fit_params_tofts={"model_mode": "1comp", "n_precontrast": 2},
        fit_params_pk={"model_mode": "2comp"},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["fit_model_mode"] == "2comp"
    assert "vp_plasma" in result
    assert "Ktrans_1c" not in result
    assert np.isclose(
        float(result["Ktrans_extended"].values.squeeze()),
        synthetic["ktrans"],
        atol=3e-2,
    )
    assert np.isclose(
        float(result["vp_plasma"].values.squeeze()),
        synthetic["vp"],
        atol=3e-2,
    )


def test_deprecated_kidney_outflow_alias_routes_to_extended_tofts():
    synthetic = _make_synthetic_extended_tofts_inputs()
    with pytest.warns(UserWarning, match="deprecated"):
        result = run_dce_pipeline(
            dce_data=synthetic["dce"],
            aif_signal_concentration=synthetic["aif_concentration"],
            aif_times=synthetic["times_s"],
            tr_s=synthetic["tr_s"],
            fa_deg=synthetic["fa_deg"],
            fit_type="kidney_two_compartment_outflow",
            fit_params_pk={"n_precontrast": 2},
            fitted_t1=synthetic["fitted_t1"],
            use_multiprocessing=False,
        )

    assert result.attrs["requested_fit_type"] == "kidney_two_compartment_outflow"
    assert result.attrs["effective_fit_type"] == "tofts"
    assert result.attrs["effective_model_mode"] == "2comp"
    assert result.attrs["equivalent_fit_type"] == "extended_tofts_plasma"
    assert "Ktrans_extended" in result
    assert "vp_plasma" in result


def test_fit_two_c_irf_model_recovers_synthetic_parameters():
    synthetic = _make_synthetic_two_c_irf_inputs()
    s0, _ = compute_s0_map(synthetic["dce"], n_precontrast=2)
    concentration_map = convert_signal_to_concentration_map(
        synthetic["dce"],
        s0,
        synthetic["fitted_t1"]["t1"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        time_points_s=synthetic["times_s"],
    )
    fit_result = fit_two_c_irf_model(
        concentration_map,
        aif_concentration=synthetic["aif_concentration"],
        aif_time_s=synthetic["times_s"],
        fit_params={"fit_start_index": 0},
        use_multiprocessing=False,
    )

    assert np.isclose(
        float(fit_result["GFR"].values.squeeze()),
        synthetic["gfr_s"],
        atol=8e-3,
    )
    assert np.isclose(
        float(fit_result["RPF"].values.squeeze()),
        synthetic["rpf_s"],
        atol=1.2e-2,
    )
    assert bool(fit_result["fit_success"].values.squeeze())
    assert "gfr_ml_min_100ml" in fit_result
    assert "filtration_fraction" in fit_result


def test_fit_two_c_irf_model_accepts_nonterminal_repetition_axis():
    synthetic = _make_synthetic_two_c_irf_inputs()
    s0, _ = compute_s0_map(synthetic["dce"], n_precontrast=2)
    concentration_map = convert_signal_to_concentration_map(
        synthetic["dce"],
        s0,
        synthetic["fitted_t1"]["t1"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        time_points_s=synthetic["times_s"],
    ).transpose("repetition_idx", "read_idx", "phase_idx", "slice_idx")

    fit_result = fit_two_c_irf_model(
        concentration_map,
        aif_concentration=synthetic["aif_concentration"],
        aif_time_s=synthetic["times_s"],
        fit_params={"fit_start_index": 0},
        use_multiprocessing=False,
    )

    assert fit_result["GFR"].dims == ("read_idx", "phase_idx", "slice_idx")
    assert np.isclose(
        float(fit_result["GFR"].values.squeeze()),
        synthetic["gfr_s"],
        atol=8e-3,
    )
    assert np.isclose(
        float(fit_result["RPF"].values.squeeze()),
        synthetic["rpf_s"],
        atol=1.2e-2,
    )
    assert bool(fit_result["fit_success"].values.squeeze())


def test_fit_two_c_irf_model_hp_reconstruct_matches_concentration_map():
    synthetic = _make_synthetic_two_c_irf_inputs()
    s0, _ = compute_s0_map(synthetic["dce"], n_precontrast=2)
    concentration_map = convert_signal_to_concentration_map(
        synthetic["dce"],
        s0,
        synthetic["fitted_t1"]["t1"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        time_points_s=synthetic["times_s"],
    )
    fit_result = fit_two_c_irf_model(
        concentration_map,
        aif_concentration=synthetic["aif_concentration"],
        aif_time_s=synthetic["times_s"],
        fit_params={"fit_start_index": 0},
        use_multiprocessing=False,
    )

    reconstructed = fit_result.hp.reconstruct()
    assert reconstructed.dims == concentration_map.dims
    assert np.allclose(reconstructed.values, concentration_map.values, atol=5e-3)


def test_run_dce_pipeline_accepts_2c_irf_and_exposes_gfr_outputs():
    synthetic = _make_synthetic_two_c_irf_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="2c_irf",
        fit_params_pk={"n_precontrast": 2, "fit_start_index": 0},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["requested_fit_type"] == "2c_irf"
    assert result.attrs["effective_fit_type"] == "2c_irf"
    assert result.attrs["effective_model_mode"] == "2c_irf"
    assert result.attrs["fit_input_space"] == "concentration"
    assert "GFR" in result
    assert "RPF" in result
    assert "gfr_ml_min_100ml" in result
    assert "rpf_ml_min_100ml" in result
    assert np.isclose(
        float(result["gfr_ml_min_100ml"].values.squeeze()),
        synthetic["gfr_s"] * 6000.0,
        atol=50.0,
    )
    voxel_volume_ml = 0.02
    voxel_gfr_ml_min = (
        float(result["gfr_ml_min_100ml"].values.squeeze()) * voxel_volume_ml / 100.0
    )
    assert np.isclose(
        voxel_gfr_ml_min, synthetic["gfr_s"] * voxel_volume_ml * 60.0, atol=2e-4
    )


@pytest.mark.parametrize(
    "fit_type",
    [
        "renal_2c_irf",
        "renal-2c-irf",
        "2c-irf",
        "modified_two_compartment_irf",
        "two_c_irf",
    ],
)
def test_run_dce_pipeline_rejects_removed_2c_irf_aliases(fit_type):
    dce = xr.DataArray(
        np.ones((1, 1, 1, 2), dtype=np.float64),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )

    with pytest.raises(ValueError, match="effective_fit_type"):
        run_dce_pipeline(dce_data=dce, fit_type=fit_type)


def test_fit_empirical_log_logistic_model_recovers_synthetic_parameters():
    synthetic = _make_synthetic_empirical_inputs()
    s0, _ = compute_s0_map(synthetic["dce"], n_precontrast=2)
    concentration_map = convert_signal_to_concentration_map(
        synthetic["dce"],
        s0,
        synthetic["fitted_t1"]["t1"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        time_points_s=synthetic["times_s"],
    )
    fit_result = fit_empirical_log_logistic_model(
        concentration_map,
        fit_params={"time_points_s": synthetic["times_s"], "fit_start_index": 0},
        use_multiprocessing=False,
    )

    assert np.isclose(
        float(fit_result["bolus_arrival_s"].values.squeeze()),
        synthetic["bolus_arrival_s"],
        atol=4.0,
    )
    assert np.isclose(
        float(fit_result["washout_rate_s"].values.squeeze()),
        synthetic["washout_rate_s"],
        atol=3e-2,
    )
    assert bool(fit_result["fit_success"].values.squeeze())


def test_fit_empirical_log_logistic_model_uses_repetition_axis_seconds():
    synthetic = _make_synthetic_empirical_inputs()
    concentration_map = synthetic["dce"].drop_vars("time_s")
    concentration_map.coords["repetition_idx"].attrs["axis_s"] = synthetic["times_s"]

    fit_result = fit_empirical_log_logistic_model(
        concentration_map,
        fit_params={"fit_start_index": 0},
        use_multiprocessing=False,
    )
    reconstructed = fit_result.hp.reconstruct(
        voxel_coords={"read_idx": 0, "phase_idx": 0, "slice_idx": 0}
    )

    assert np.allclose(fit_result["time_points_s"].values, synthetic["times_s"])
    assert np.allclose(reconstructed.coords["time_s"].values, synthetic["times_s"])


def test_run_dce_pipeline_accepts_empirical_log_logistic_alias_without_aif():
    synthetic = _make_synthetic_empirical_inputs()
    result = run_dce_pipeline(
        dce_data=synthetic["dce"],
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="empirical_log_logistic",
        fit_params_pk={"n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.attrs["requested_fit_type"] == "empirical_log_logistic"
    assert result.attrs["effective_fit_type"] == "empirical_log_logistic"
    assert "bolus_arrival_s" in result
    assert "half_rise_time_s" in result
    np.testing.assert_allclose(
        result["half_rise_time_s"],
        result["bolus_arrival_s"] + result["rise_time_s"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert result["half_rise_time_s"].attrs["units"] == "s"
    assert "fit_window_mask" in result
    assert "aif_concentration" not in result
    assert np.isfinite(float(result["washout_rate_s"].values.squeeze()))
    assert bool(result["fit_success"].values.squeeze())

    reconstructed = result.hp.reconstruct(
        voxel_coords={"read_idx": 0, "phase_idx": 0, "slice_idx": 0}
    )
    assert reconstructed.sizes["repetition_idx"] == len(synthetic["times_s"])
    assert np.all(np.isfinite(reconstructed.values))


def test_run_dce_pipeline_empirical_preserves_trimmed_repetition_coordinates():
    synthetic = _make_synthetic_empirical_inputs()
    trimmed_dce = (
        synthetic["dce"]
        .isel(repetition_idx=slice(1, None))
        .assign_coords(time=("repetition_idx", synthetic["times_s"][1:]))
    )

    result = run_dce_pipeline(
        dce_data=trimmed_dce,
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_type="empirical_log_logistic",
        fit_params_pk={"n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.sizes["repetition_idx"] == trimmed_dce.sizes["repetition_idx"]
    assert np.array_equal(
        result.coords["repetition_idx"].values,
        trimmed_dce.coords["repetition_idx"].values,
    )
    assert np.all(np.isfinite(result["time_points_s"].values))
    assert np.array_equal(
        result["time_points_s"].coords["repetition_idx"].values,
        result.coords["repetition_idx"].values,
    )
    assert np.array_equal(
        result["fit_window_mask"].coords["repetition_idx"].values,
        result.coords["repetition_idx"].values,
    )


def test_run_dce_pipeline_preserves_trimmed_repetition_coordinates():
    synthetic = _make_synthetic_tofts_inputs()
    trimmed_dce = synthetic["dce"].isel(repetition_idx=slice(1, None))
    trimmed_times = synthetic["times_s"][1:]
    trimmed_aif = synthetic["aif_concentration"][1:]

    result = run_dce_pipeline(
        dce_data=trimmed_dce,
        aif_signal_concentration=trimmed_aif,
        aif_times=trimmed_times,
        tr_s=synthetic["tr_s"],
        fa_deg=synthetic["fa_deg"],
        fit_params_tofts={"model_mode": "1comp", "n_precontrast": 2},
        fitted_t1=synthetic["fitted_t1"],
        use_multiprocessing=False,
    )

    assert result.sizes["repetition_idx"] == trimmed_dce.sizes["repetition_idx"]
    assert np.array_equal(
        result.coords["repetition_idx"].values,
        trimmed_dce.coords["repetition_idx"].values,
    )
    assert np.all(np.isfinite(result["time_points_s"].values))
    assert np.array_equal(
        result["time_points_s"].coords["repetition_idx"].values,
        result.coords["repetition_idx"].values,
    )

    reconstructed = result.hp.reconstruct(
        voxel_coords={"read_idx": 0, "phase_idx": 0, "slice_idx": 0},
        model_mode="1comp",
    )
    assert reconstructed.sizes["repetition_idx"] == trimmed_dce.sizes["repetition_idx"]
    assert np.all(np.isfinite(reconstructed.values))


def test_fit_quantitative_maps_uses_dataset_variables_for_reconstruction_schema():
    synthetic = _make_synthetic_renal_inputs()
    concentration_map = xr.DataArray(
        synthetic["tissue_concentration"].reshape(2, 1, 1, -1),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(2),
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(len(synthetic["times_s"]), dtype=int),
        },
    )

    fit_result = fit(
        concentration_map,
        model="patlak",
        fit_params={
            "aif_signal": synthetic["aif_concentration"],
            "time_points_s": synthetic["times_s"],
            "use_multiprocessing": False,
        },
    )

    assert "time_points_s" in fit_result
    assert "aif_signal" in fit_result
    assert "aif_time_s" in fit_result
    assert "time_points_s" not in fit_result.attrs
    assert "aif_signal" not in fit_result.attrs


def test_patlak_hp_reconstruct_matches_full_concentration_map():
    synthetic = _make_synthetic_renal_inputs()
    concentration_map = xr.DataArray(
        synthetic["tissue_concentration"].reshape(2, 1, 1, -1),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(2),
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(len(synthetic["times_s"]), dtype=int),
        },
    )

    fit_result = fit(
        concentration_map,
        model="patlak",
        fit_params={
            "aif_signal": synthetic["aif_concentration"],
            "time_points_s": synthetic["times_s"],
            "use_multiprocessing": False,
        },
    )

    reconstructed = fit_result.hp.reconstruct()
    assert reconstructed.dims == concentration_map.dims
    assert np.allclose(reconstructed.values, concentration_map.values, atol=1e-6)


def test_tofts_hp_reconstruct_matches_full_concentration_map():
    synthetic = _make_synthetic_tofts_inputs()
    concentration_map = xr.DataArray(
        synthetic["dce"].copy().values * np.nan,
        dims=synthetic["dce"].dims,
        coords=synthetic["dce"].coords,
    )
    concentration_map.values = tofts_model_1comp(
        synthetic["times_s"],
        0.18,
        0.55,
        interp1d(
            synthetic["times_s"],
            synthetic["aif_concentration"],
            kind="linear",
            fill_value="extrapolate",
        ),
    ).reshape(1, 1, 1, -1)

    fit_result = fit(
        concentration_map,
        model="tofts",
        fit_params={
            "aif_signal": synthetic["aif_concentration"],
            "aif_times": synthetic["times_s"],
            "time_points_s": synthetic["times_s"],
            "input_space": "concentration",
            "model_mode": "1comp",
            "use_multiprocessing": False,
        },
    )

    reconstructed = fit_result.hp.reconstruct()
    assert reconstructed.dims == concentration_map.dims
    assert np.allclose(reconstructed.values, concentration_map.values, atol=1e-4)


# def test_run_dce_pipeline_legacy_signal_fit_matches_legacy_direct_tofts_fit_quantitative_maps_path():
#     synthetic = _make_synthetic_tofts_inputs()
#     fit_params = {"model_mode": "1comp", "n_precontrast": 2}

#     result = run_dce_pipeline_legacy_signal_fit(
#         dce_data=synthetic["dce"],
#         aif_signal_concentration=synthetic["aif_concentration"],
#         aif_times=synthetic["times_s"],
#         tr_s=synthetic["tr_s"],
#         fa_deg=synthetic["fa_deg"],
#         fit_params_tofts=fit_params,
#         fitted_t1=synthetic["fitted_t1"],
#         use_multiprocessing=False,
#     )

#     assert result.attrs["fit_input_space"] == "signal"

#     legacy_direct = fit_quantitative_maps(
#         input_data=synthetic["dce"],
#         fit_type="tofts",
#         fit_params={
#             "aif_signal": synthetic["aif_concentration"],
#             "aif_times": synthetic["times_s"],
#             "time_points_s": synthetic["times_s"],
#             "t1_map": synthetic["fitted_t1"]["t1"],
#             "tr_s": synthetic["tr_s"],
#             "fa_deg": synthetic["fa_deg"],
#             "r1": 3.5,
#             "use_multiprocessing": False,
#             **fit_params,
#         },
#     )

#     assert np.isclose(
#         float(result["Ktrans_1c"].values.squeeze()),
#         float(legacy_direct["Ktrans_1c"].values.squeeze()),
#         atol=1e-10,
#     )
#     assert np.isclose(
#         float(result["kep_1c"].values.squeeze()),
#         float(legacy_direct["kep_1c"].values.squeeze()),
#         atol=1e-10,
#     )


def test_run_renal_dce_pipeline_adds_roi_fits_and_summaries():
    synthetic = _make_synthetic_renal_inputs()
    whole_kidney_mask = xr.DataArray(
        np.array([[[True]], [[True]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
        coords={
            "read_idx": synthetic["dce"].coords["read_idx"],
            "phase_idx": synthetic["dce"].coords["phase_idx"],
            "slice_idx": synthetic["dce"].coords["slice_idx"],
        },
    )

    result = run_renal_dce_pipeline(
        synthetic["dce"],
        fitted_t1=synthetic["fitted_t1"],
        aif_signal_concentration=synthetic["aif_concentration"],
        aif_times=synthetic["times_s"],
        fit_type="patlak",
        fit_params_pk={"n_precontrast": 1},
        renal_masks={"whole_kidney": whole_kidney_mask},
        use_multiprocessing=False,
    )

    assert "roi_patlak_slope" in result
    assert "filtration_index_summary" in result
    assert result["roi_concentration_curve"].dims == ("roi_label", "repetition_idx")


def test_run_renal_dce_pipeline_raises_on_missing_overlap():
    synthetic = _make_synthetic_renal_inputs(slice_positions=(10.0,))
    dce = synthetic["dce"].copy()
    dce.coords["slice_idx"].attrs["axis_mm"] = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="No overlapping slice positions"):
        run_renal_dce_pipeline(
            dce,
            fitted_t1=synthetic["fitted_t1"],
            aif_signal_concentration=synthetic["aif_concentration"],
            aif_times=synthetic["times_s"],
            fit_type="patlak",
            fit_params_pk={"n_precontrast": 1},
            use_multiprocessing=False,
        )


def test_save_extended_tofts_roi_metrics_csv_overwrites_existing_subject_row(tmp_path):
    masks = xr.Dataset(
        {
            "left_cortex": xr.DataArray(
                np.array([[[True]], [[False]], [[False]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
            "right_cortex": xr.DataArray(
                np.array([[[False]], [[True]], [[False]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
            "tumor": xr.DataArray(
                np.array([[[False]], [[False]], [[True]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
        }
    )
    fit_results = xr.Dataset(
        {
            "tubular_flow_ml_min_100ml": xr.DataArray(
                np.array([[[10.0]], [[20.0]], [[5.0]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
            "vp": xr.DataArray(
                np.array([[[0.1]], [[0.2]], [[0.05]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
        }
    )
    fit_results_2comp = xr.Dataset(
        {
            "Ktrans_2c": xr.DataArray(
                np.array([[[1.0e-6]], [[2.0e-6]], [[3.0e-6]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
            "kep_2c": xr.DataArray(
                np.array([[[4.0e-3]], [[5.0e-3]], [[6.0e-3]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
            "vp_2c": xr.DataArray(
                np.array([[[0.10]], [[0.20]], [[0.30]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
            "ve_2c": xr.DataArray(
                np.array([[[0.40]], [[0.50]], [[0.60]]]),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
        }
    )

    row = collect_extended_tofts_roi_metrics(
        fit_results,
        masks,
        voxel_volume_ml=0.25,
        spatial_dims=("read_idx", "phase_idx", "slice_idx"),
        subject_name="mouse_01",
        fit_results_2comp=fit_results_2comp,
        spatial_dims_2comp=("read_idx", "phase_idx", "slice_idx"),
    )
    assert row["animal_id"] == "mouse_01"
    assert np.isclose(row["left_gfr_ml_min"], 0.025)
    assert np.isclose(row["right_gfr_ml_min"], 0.05)
    assert np.isclose(row["mean_gfr_ml_min"], 0.0375)
    assert np.isclose(row["left_cortex_mean_ktrans_2c_x1e6"], 1.0)
    assert np.isclose(row["left_cortex_mean_kep_2c_x1e3"], 4.0)
    assert np.isclose(row["left_cortex_mean_vp_2c_percent"], 10.0)
    assert np.isclose(row["left_cortex_mean_ve_2c_percent"], 40.0)

    csv_path = tmp_path / "renal_metrics.csv"
    save_extended_tofts_roi_metrics_csv(
        fit_results,
        masks,
        voxel_volume_ml=0.25,
        spatial_dims=("read_idx", "phase_idx", "slice_idx"),
        subject_name="mouse_01",
        csv_path=csv_path,
        fit_results_2comp=fit_results_2comp,
        spatial_dims_2comp=("read_idx", "phase_idx", "slice_idx"),
    )

    updated_fit_results = fit_results.copy(deep=True)
    updated_fit_results["tubular_flow_ml_min_100ml"] = xr.DataArray(
        np.array([[[40.0]], [[20.0]], [[5.0]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    updated_fit_results_2comp = fit_results_2comp.copy(deep=True)
    updated_fit_results_2comp["Ktrans_2c"] = xr.DataArray(
        np.array([[[8.0e-6]], [[2.0e-6]], [[3.0e-6]]]),
        dims=("read_idx", "phase_idx", "slice_idx"),
    )
    save_extended_tofts_roi_metrics_csv(
        updated_fit_results,
        masks,
        voxel_volume_ml=0.25,
        spatial_dims=("read_idx", "phase_idx", "slice_idx"),
        subject_name="mouse_01",
        csv_path=csv_path,
        fit_results_2comp=updated_fit_results_2comp,
        spatial_dims_2comp=("read_idx", "phase_idx", "slice_idx"),
    )
    save_extended_tofts_roi_metrics_csv(
        fit_results,
        masks,
        voxel_volume_ml=0.25,
        spatial_dims=("read_idx", "phase_idx", "slice_idx"),
        subject_name="mouse_02",
        csv_path=csv_path,
        fit_results_2comp=fit_results_2comp,
        spatial_dims_2comp=("read_idx", "phase_idx", "slice_idx"),
    )

    saved = pd.read_csv(csv_path)
    assert list(saved["animal_id"]) == ["mouse_01", "mouse_02"]
    mouse_01 = saved.loc[saved["animal_id"] == "mouse_01"].iloc[0]
    assert np.isclose(mouse_01["left_cortex_gfr_ml_min"], 0.1)
    assert np.isclose(mouse_01["left_gfr_ml_min"], 0.1)
    assert np.isclose(mouse_01["right_gfr_ml_min"], 0.05)
    assert np.isclose(mouse_01["mean_gfr_ml_min"], 0.075)
    assert np.isclose(mouse_01["left_cortex_mean_ktrans_2c_x1e6"], 8.0)
    assert np.isclose(mouse_01["right_cortex_mean_kep_2c_x1e3"], 5.0)
    assert np.isclose(mouse_01["tumor_mean_vp_2c_percent"], 30.0)
    assert np.isclose(mouse_01["tumor_mean_ve_2c_percent"], 60.0)


def test_collect_extended_tofts_roi_metrics_uses_fractional_mask_weights():
    masks = xr.Dataset(
        {
            "left_cortex": xr.DataArray(
                np.array([[[0.5]], [[0.25]], [[0.0]]], dtype=np.float32),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
        }
    )
    fit_results = xr.Dataset(
        {
            "tubular_flow_ml_min_100ml": xr.DataArray(
                np.array([[[10.0]], [[30.0]], [[50.0]]], dtype=np.float64),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
            "vp": xr.DataArray(
                np.array([[[0.1]], [[0.3]], [[0.5]]], dtype=np.float64),
                dims=("read_idx", "phase_idx", "slice_idx"),
            ),
        }
    )

    row = collect_extended_tofts_roi_metrics(
        fit_results,
        masks,
        voxel_volume_ml=0.25,
        spatial_dims=("read_idx", "phase_idx", "slice_idx"),
        subject_name="mouse_weighted",
    )

    expected_weight_sum = 0.75
    expected_mean_flow = (0.5 * 10.0 + 0.25 * 30.0) / expected_weight_sum
    expected_signal = 0.5 * 10.0 + 0.25 * 30.0
    expected_volume_ml = expected_weight_sum * 0.25
    expected_gfr = expected_volume_ml * expected_signal / 100.0
    expected_mean_vp = (0.5 * 0.1 + 0.25 * 0.3) / expected_weight_sum

    assert np.isclose(row["left_cortex_voxel_count"], expected_weight_sum)
    assert np.isclose(
        row["left_cortex_mean_tubular_flow_ml_min_100ml"], expected_mean_flow
    )
    assert np.isclose(row["left_cortex_signal"], expected_signal)
    assert np.isclose(row["left_cortex_volume_ml"], expected_volume_ml)
    assert np.isclose(row["left_cortex_mean_plasma_volume_fraction"], expected_mean_vp)
    assert np.isclose(row["left_cortex_gfr_ml_min"], expected_gfr)
    assert np.isclose(row["left_gfr_ml_min"], expected_gfr)
    assert np.isclose(row["mean_gfr_ml_min"], expected_gfr)


def test_run_renal_dce_pipeline_rejects_removed_legacy_fit_type():
    synthetic = _make_synthetic_renal_inputs()

    with pytest.raises(
        ValueError,
        match="only supports fit_type='patlak', 'tofts', or 'sourbron'",
    ):
        run_renal_dce_pipeline(
            dce_data=synthetic["dce"],
            fitted_t1=synthetic["fitted_t1"],
            aif_signal_concentration=synthetic["aif_concentration"],
            aif_times=synthetic["times_s"],
            tr_s=synthetic["tr_s"],
            fa_deg=synthetic["fa_deg"],
            fit_type="renal_2comp",
            fit_params_pk={"n_precontrast": 2},
            use_multiprocessing=False,
        )
