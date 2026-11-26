"""Shared extended-compartment equation and equivalent parameter aliases.

The reduced Sourbron/extended-Tofts equation has plasma fraction vp, transfer
rate Ktrans = Ft in 1/s and decay rate kep = 1/Tt. Times are seconds and plasma
and tissue concentrations mM. Float64 trapezoid integration retains uniform or
non-uniform acquisition samples; the vascular multiplication retains its input
precision. Labelled evaluation preserves parameter coordinates and dimension
order. Model-local kernels retain multiplication/division operation order.

The equation preserves inputs and is eager and deterministic for a deterministic
kernel. Private alias helpers modify a caller-owned dataset in place; new fields
retain the existing scientific units and clipping contract. No optimizer,
scientific defaults, finite plasma-transit calculation or result cache lives here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.quantitative.convolution import (
    trapezoidal_convolution,
    trapezoidal_convolution_weights_s,
)

from .two_c_irf import _PER_S_TO_ML_MIN_PER_100ML

__all__ = []


def _extended_compartment_concentration(
    time_s: np.ndarray,
    plasma_concentration: np.ndarray | xr.DataArray,
    *,
    plasma_volume_fraction: float | xr.DataArray,
    transfer_rate_s: float | xr.DataArray,
    retention_kernel: Callable,
) -> np.ndarray | xr.DataArray:
    """Evaluate vp*Cp + transfer*convolution(Cp, retention) on validated samples.

    A labelled plasma input selects labelled integration for map parameters;
    scalar callers supply a NumPy vector. Kernel lags are seconds, retention is
    dimensionless, concentrations mM and transfer 1/s. Callers own validation
    and the native rate/time kernel arithmetic. Inputs are preserved.
    """
    if isinstance(plasma_concentration, xr.DataArray):
        time_da = xr.DataArray(
            time_s, dims=["repetition_idx"], coords={"repetition_idx": time_s}
        )
        lag_s = time_da - time_da.rename({"repetition_idx": "integration_idx"})
        weights_s = xr.DataArray(
            trapezoidal_convolution_weights_s(time_s),
            dims=["repetition_idx", "integration_idx"],
            coords={"repetition_idx": time_s, "integration_idx": time_s},
        )
        kernel = xr.where(lag_s >= 0, retention_kernel(lag_s), 0.0)
        integrand = (
            plasma_concentration.rename({"repetition_idx": "integration_idx"}) * kernel
        )
        convolution = (integrand * weights_s).sum(dim="integration_idx")
    else:
        convolution = trapezoidal_convolution(
            plasma_concentration, time_s, retention_kernel
        )
    vascular_term = plasma_volume_fraction * plasma_concentration
    exchange_term = transfer_rate_s * convolution
    return vascular_term + exchange_term


_EXTENDED_TOFTS_SOURBRON_EQUIVALENT_FIT_TYPES = (
    "extended_tofts_plasma,kidney_two_compartment_outflow,sourbron"
)
_EXTENDED_TOFTS_SOURBRON_EQUIVALENCE_NOTE = (
    "The implemented extended Tofts 2-compartment form is mathematically "
    "equivalent to the Sourbron separable renal compartment parameterization. "
    "Parameter correspondence: vp_2c = vp, Ktrans_2c = Ft, kep_2c = 1/Tt, "
    "and ve_equivalent = Ft * Tt. Existing ve_2c and ve_extended outputs "
    "retain clipping to [0, 1]; ve_was_clipped marks values outside that interval."
)

_EQUIVALENT_VOLUME_ATTRS = {
    "ve_equivalent": {
        "long_name": "Unbounded equivalent extravascular volume",
        "description": (
            "Derived Ktrans/kep or Ft*Tt for extended Tofts/reduced Sourbron, "
            "without clipping; not an additional fitted parameter."
        ),
        "units": "1",
    },
    "ve_was_clipped": {
        "long_name": "Equivalent volume clipping indicator",
        "description": (
            "True when ve_equivalent lies outside [0, 1], the interval used "
            "by legacy ve outputs. False for NaN/unavailable values; "
            "this is not a fit-success or physical-validity flag."
        ),
        "units": "1",
    },
}


def _add_equivalent_volume_outputs(dataset: xr.Dataset) -> xr.Dataset:
    """Add derived volumes in place to a caller-owned fit dataset, once.

    Only the extended Tofts and reduced Sourbron parameterizations participate;
    a finite plasma-transit parameter retains its separate contract. NaN or
    non-positive decay times/rates cannot define this equivalent volume.
    """
    if "Tp" in dataset:
        return dataset
    if {"Ktrans_2c", "kep_2c"}.issubset(dataset):
        if "ve_equivalent" not in dataset:
            transfer = dataset["Ktrans_2c"].astype(np.float64)
            decay = dataset["kep_2c"].astype(np.float64)
            dataset["ve_equivalent"] = xr.where(
                np.isfinite(transfer) & np.isfinite(decay) & (decay > 0),
                transfer / decay,
                np.nan,
            )
    elif {"Ft", "Tt"}.issubset(dataset):
        if "ve_equivalent" not in dataset:
            transfer = dataset["Ft"].astype(np.float64)
            transit = dataset["Tt"].astype(np.float64)
            dataset["ve_equivalent"] = xr.where(
                np.isfinite(transfer) & np.isfinite(transit) & (transit > 0),
                transfer * transit,
                np.nan,
            )
    else:
        return dataset
    if "ve_was_clipped" not in dataset:
        dataset["ve_was_clipped"] = (dataset["ve_equivalent"] < 0) | (
            dataset["ve_equivalent"] > 1
        )
    for name, attrs in _EQUIVALENT_VOLUME_ATTRS.items():
        dataset[name].attrs.update(attrs)
    return dataset


def _add_extended_compartment_aliases(dataset: xr.Dataset) -> xr.Dataset:
    """Add equivalent extended/renal aliases to a caller-owned dataset in place."""
    has_tp_parameter = "Tp" in dataset.data_vars
    if {"Ktrans_2c", "kep_2c", "vp_2c"}.issubset(dataset.data_vars):
        dataset["Ktrans_extended"] = dataset["Ktrans_2c"].copy(deep=True)
        dataset["kep_extended"] = dataset["kep_2c"].copy(deep=True)
        dataset["vp_plasma"] = dataset["vp_2c"].copy(deep=True)
        dataset["ve_extended"] = dataset.get(
            "ve_2c",
            xr.where(
                dataset["kep_2c"] > 0,
                dataset["Ktrans_2c"] / dataset["kep_2c"],
                np.nan,
            ),
        ).copy(deep=True)
        transit_time_equivalent = xr.where(
            dataset["kep_2c"] > 0,
            1.0 / dataset["kep_2c"],
            np.nan,
        )
        if "vp" not in dataset:
            dataset["vp"] = dataset["vp_2c"].copy(deep=True)
        if "Ft" not in dataset:
            dataset["Ft"] = dataset["Ktrans_2c"].copy(deep=True)
        if "Tt" not in dataset:
            dataset["Tt"] = transit_time_equivalent.copy(deep=True)
        if "tubular_flow_s" not in dataset:
            dataset["tubular_flow_s"] = dataset["Ktrans_2c"].copy(deep=True)
        if "tubular_flow_ml_min_100ml" not in dataset:
            dataset["tubular_flow_ml_min_100ml"] = (
                dataset["Ktrans_2c"] * _PER_S_TO_ML_MIN_PER_100ML
            ).astype(np.float64)
        if "tubular_transit_time_s" not in dataset:
            dataset["tubular_transit_time_s"] = transit_time_equivalent.copy(deep=True)

    if {"vp", "Ft", "Tt"}.issubset(dataset.data_vars) and not has_tp_parameter:
        kep_equivalent = xr.where(
            dataset["Tt"] > 0,
            1.0 / dataset["Tt"],
            np.nan,
        )
        ve_equivalent = xr.where(
            dataset["Tt"] > 0,
            (dataset["Ft"] * dataset["Tt"]).clip(min=0.0, max=1.0),
            np.nan,
        )
        if "Ktrans_2c" not in dataset:
            dataset["Ktrans_2c"] = dataset["Ft"].copy(deep=True)
        if "kep_2c" not in dataset:
            dataset["kep_2c"] = kep_equivalent.copy(deep=True)
        if "vp_2c" not in dataset:
            dataset["vp_2c"] = dataset["vp"].copy(deep=True)
        if "ve_2c" not in dataset:
            dataset["ve_2c"] = ve_equivalent.astype(np.float64)
        if "Ktrans_extended" not in dataset:
            dataset["Ktrans_extended"] = dataset["Ft"].copy(deep=True)
        if "kep_extended" not in dataset:
            dataset["kep_extended"] = kep_equivalent.copy(deep=True)
        if "vp_plasma" not in dataset:
            dataset["vp_plasma"] = dataset["vp"].copy(deep=True)
        if "ve_extended" not in dataset:
            dataset["ve_extended"] = ve_equivalent.astype(np.float64)
        if "tubular_flow_s" not in dataset:
            dataset["tubular_flow_s"] = dataset["Ft"].copy(deep=True)
        if "tubular_flow_ml_min_100ml" not in dataset:
            dataset["tubular_flow_ml_min_100ml"] = (
                dataset["Ft"] * _PER_S_TO_ML_MIN_PER_100ML
            ).astype(np.float64)
        if "tubular_transit_time_s" not in dataset:
            dataset["tubular_transit_time_s"] = dataset["Tt"].copy(deep=True)

    if {"vp", "Ft", "Tt"}.issubset(dataset.data_vars):
        dataset["kidney_plasma_volume_fraction"] = dataset["vp"].copy(deep=True)
        dataset["kidney_outflow_rate_s"] = dataset["Ft"].copy(deep=True)
        dataset["kidney_transit_time_s"] = dataset["Tt"].copy(deep=True)
        if "tubular_flow_ml_min_100ml" in dataset:
            dataset["kidney_outflow_ml_min_100ml"] = dataset[
                "tubular_flow_ml_min_100ml"
            ].copy(deep=True)
        if not has_tp_parameter:
            dataset.attrs.setdefault(
                "equation_equivalent_fit_types",
                _EXTENDED_TOFTS_SOURBRON_EQUIVALENT_FIT_TYPES,
            )
            dataset.attrs.setdefault(
                "equivalent_parameterizations",
                "extended_tofts_plasma,sourbron",
            )
            dataset.attrs.setdefault(
                "parameterization_equivalence_note",
                _EXTENDED_TOFTS_SOURBRON_EQUIVALENCE_NOTE,
            )

    return dataset
