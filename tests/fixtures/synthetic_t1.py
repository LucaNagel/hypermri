"""Deterministic scientific regression fixture.

Two-compartment 12x10 inversion-recovery phantom: T1 800/1350 ms, seven TI
values, Gaussian sigma 7 a.u., seed 20260728. T1 MAE < 50 ms; signal RMSE < 10
a.u. Calls shipped domain APIs; no acquired data, plotting, downloads or file
writes.
"""

import numpy as np
import xarray as xr

from hypermri.analysis.fitting import fit, reconstruct_fit

rng = np.random.default_rng(20260728)
n_read, n_phase = (12, 10)
inversion_times_ms = np.array([80.0, 180.0, 350.0, 650.0, 1100.0, 1800.0, 3000.0])
read_axis_mm = np.linspace(-11.0, 11.0, n_read)
phase_axis_mm = np.linspace(-9.0, 9.0, n_phase)
read_grid, phase_grid = np.meshgrid(read_axis_mm, phase_axis_mm, indexing="ij")
phantom_values = (read_grid / 9.5) ** 2 + (phase_grid / 7.5) ** 2 <= 1.0
inner_compartment = (read_grid / 4.0) ** 2 + (phase_grid / 3.2) ** 2 <= 1.0
true_t1_ms_values = np.where(inner_compartment, 800.0, 1350.0)
true_t1_ms_values = np.where(phantom_values, true_t1_ms_values, 1000.0)
amplitude = np.where(phantom_values, 900.0 * (1.0 + 0.1 * read_grid / 11.0), 0.0)
baseline = np.where(phantom_values, 15.0, 0.0)
clean_signal = (
    amplitude[..., None]
    * (
        1.0
        - 2.0
        * np.exp(-inversion_times_ms[None, None, :] / true_t1_ms_values[..., None])
    )
    + baseline[..., None]
)
measured_values = clean_signal + rng.normal(0.0, 7.0, clean_signal.shape)
measured = xr.DataArray(
    measured_values,
    dims=("read_idx", "phase_idx", "inversion_idx"),
    coords={
        "read_idx": np.arange(n_read),
        "phase_idx": np.arange(n_phase),
        "inversion_idx": inversion_times_ms,
    },
    name="synthetic_ir_signal",
    attrs={"unit": "a.u.", "domain": "spatial", "signal_domain": "temporal"},
)
measured.coords["read_idx"].attrs.update(
    unit="index",
    axis_mm=read_axis_mm,
    resolution_mm=float(np.diff(read_axis_mm).mean()),
)
measured.coords["phase_idx"].attrs.update(
    unit="index",
    axis_mm=phase_axis_mm,
    resolution_mm=float(np.diff(phase_axis_mm).mean()),
)
measured.coords["inversion_idx"].attrs["unit"] = "ms"
phantom_mask = xr.DataArray(
    phantom_values,
    dims=("read_idx", "phase_idx"),
    coords={"read_idx": measured.read_idx, "phase_idx": measured.phase_idx},
    name="phantom_mask",
)
true_t1_ms = xr.DataArray(
    true_t1_ms_values,
    dims=("read_idx", "phase_idx"),
    coords={"read_idx": measured.read_idx, "phase_idx": measured.phase_idx},
    name="true_t1_ms",
)
assert measured.dims == ("read_idx", "phase_idx", "inversion_idx")
assert measured.coords["inversion_idx"].attrs["unit"] == "ms"
fit_results = fit(
    measured,
    model="t1",
    fit_params={
        "t1_values_s": np.linspace(0.4, 1.8, 141),
        "fit_baseline": True,
        "use_multiprocessing": False,
    },
)
required_variables = {"t1", "A", "B", "rss_t1"}
assert required_variables.issubset(fit_results.data_vars)
assert fit_results.attrs["model_name"] == "t1_inversion_recovery"
assert fit_results.attrs["dynamic_dim"] == "inversion_idx"
assert fit_results["t1"].attrs["units"] == "ms"
assert fit_results["inversion_times_ms"].attrs["units"] == "ms"
reconstructed = reconstruct_fit(fit_results)
t1_mae_ms = float(np.abs(fit_results["t1"] - true_t1_ms).where(phantom_mask).mean())
signal_rmse = float(
    np.sqrt(((reconstructed - measured) ** 2).where(phantom_mask).mean())
)
assert reconstructed.dims == measured.dims
assert np.array_equal(reconstructed.inversion_idx, measured.inversion_idx)
assert t1_mae_ms < 50.0
assert signal_rmse < 10.0
