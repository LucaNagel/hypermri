"""Mono-exponential ADC fitting model."""

from __future__ import annotations

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.execution import copy_execution_params

MODEL_ID = "adc"
RESULT_MODEL_NAME = "monoexponential_adc"
MODEL_EQUATION = "s0 * exp(-b * adc)"
OUTPUT_PARAM_NAMES = ["adc", "s0", "rss_adc"]


def get_adc_params_info() -> dict[str, object]:
    """Return the public parameter-info entry for the ADC model."""
    return {
        "required_fit_params": {
            "b_values": "Array of b-values (e.g., in s/mm^2) corresponding to the diffusion weighting",
        },
        "optional_fit_params": {
            "signal_threshold": "Minimum signal value to include in fit (avoids log(0)). Default: 1e-5",
            "use_multiprocessing": "Enable parallel processing. Default: True",
            "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
        },
        "output_parameters": {
            "adc": "Apparent diffusion coefficient (typically in mm^2/s)",
            "s0": "Signal at b=0 (baseline signal intensity)",
            "rss_adc": "Residual sum of squares for the fit (in log-space)",
        },
        "model_equation": MODEL_EQUATION,
    }


def prepare_adc_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
    return_params_info: bool = False,
    *,
    base_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Prepare the public quantitative fit-params dataset for ADC fitting."""
    if return_params_info:
        return None
    fit_params = fit_params or {}
    attrs = dict(base_attrs or {})
    dynamic_dim = fit_params.get("dynamic_dim", "bvalue_idx")
    attrs.update(
        {
            "fit_type": "adc",
            "model_id": "adc",
            "model_name": RESULT_MODEL_NAME,
            "dynamic_dim": dynamic_dim,
            "signal_threshold": fit_params.get("signal_threshold", 1e-5),
            "b_values": fit_params.get("b_values", data.coords.get(dynamic_dim, None)),
        }
    )
    copy_execution_params(fit_params, attrs)
    return xr.Dataset(attrs=attrs)


def fit_adc_monoexponential_helper(
    signal: np.ndarray = None,
    b_values: np.ndarray = None,
    signal_threshold: float = 1e-5,
    fit_params: dict | None = None,
):
    """Fit ADC using a mono-exponential model via log-linear regression."""
    fit_params = fit_params or {}
    b_values = fit_params.get("b_values", b_values)
    signal_threshold = fit_params.get("signal_threshold", signal_threshold)
    signal = fit_params.get("signal", signal)

    magnitude_signal = np.abs(signal)
    valid_indices = (magnitude_signal > signal_threshold) & (b_values >= 0)

    if np.sum(valid_indices) < 2:
        return {"adc": np.nan, "s0": np.nan, "rss_adc": np.nan}

    b_values_filt = b_values[valid_indices]
    log_signal_filt = np.log(magnitude_signal[valid_indices])
    design = np.vstack(
        [np.ones_like(b_values_filt, dtype=float), -b_values_filt.astype(float)]
    ).T

    try:
        coeffs, _, rank, _ = np.linalg.lstsq(design, log_signal_filt, rcond=None)

        if rank < 2:
            return {"adc": np.nan, "s0": np.nan, "rss_adc": np.nan}

        log_s0 = coeffs[0]
        adc_val = coeffs[1]
        s0_val = np.exp(log_s0)
        predicted_log_signals = design @ coeffs
        rss = np.sum((log_signal_filt - predicted_log_signals) ** 2)

        if adc_val < 0:
            adc_val = np.nan

        return {"adc": adc_val, "s0": s0_val, "rss_adc": rss}

    except (np.linalg.LinAlgError, ValueError):
        return {"adc": np.nan, "s0": np.nan, "rss_adc": np.nan}


def monoexponential_adc_model(
    b_values: np.ndarray,
    s0: xr.DataArray,
    adc: xr.DataArray,
    db_mode: bool = False,
) -> xr.DataArray:
    """Reconstruct the signal for the mono-exponential ADC model."""
    if db_mode:
        print("Calculating mono-exponential ADC model:")
        print(f"  - b_values: {b_values}")
        print(f"  - s0: {s0}")
        print(f"  - adc: {adc}")

    b_da = xr.DataArray(b_values, dims=["bvalue_idx"], coords={"bvalue_idx": b_values})
    if db_mode:
        print(f"  - b_da: {b_da}")
    return s0 * np.exp(-b_da * adc)
