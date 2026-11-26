"""B1 sinusoidal fitting model."""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.execution import copy_execution_params
from hypermri.analysis.fitting.voxelwise import VoxelwiseExecutor
from hypermri.utils.system.logging import debug_log

MODEL_ID = "b1"
RESULT_MODEL_NAME = "b1_sinusoidal"
MODEL_EQUATION = "S(sqrt(P)) = A * sin(π / period * sqrt(P))"
OUTPUT_PARAM_NAMES = [
    "amplitude",
    "period",
    "refpower_90deg_W",
    "rss_b1",
    "stop_index",
    "n_valid_samples",
]
DEBUG_FILE = "/tmp/fit_maps_debug.log"
DEBUG_LOG = False


def debug_log_b1(
    message="",
    file_name=DEBUG_FILE,
    write_to_file=DEBUG_LOG,
    numpy_full_print=False,
):
    """Mirror fit_maps debug logging for the extracted B1 model code."""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
        numpy_full_print=numpy_full_print,
    )


def get_b1_params_info() -> dict[str, object]:
    """Return the public parameter-info entry for the B1 model."""
    return {
        "required_fit_params": {
            "refpower_W": "Array of reference power values in Watts corresponding to the RF power sweep",
            "fa_deg": "Flip angle in degrees (e.g., 90 for 90° pulse). Required for absolute B1 calibration",
            "trf_ms": "RF pulse duration in milliseconds. Required for reference power calculation",
            "sint": "Integration factor (dimensionless, related to RF pulse shape). Required for reference power calculation",
        },
        "optional_fit_params": {
            "period_samples": "Number of period values tested in each iteration. Default: 500",
            "use_region_growing": "Use region-growing algorithm to propagate initial guesses from neighbors. Default: False",
            "seed_voxel": "Seed voxel specification (tuple or dict indexed by xarray dims, e.g. {'read_idx': 32, 'phase_idx': 32}). Default: auto-detect",
            "stop_index_extra": "Additional samples beyond the inferred 180° cutoff to include in the fit window. Default: 0",
            "stage1_niter": "Iterations used for the first (full) sweep of the two-stage fit. Default: max(niter, 10)",
            "stage1_npoints": "Period samples used for the first sweep. Default: max(period_samples, 1500)",
            "store_debug_info": "Store iteration details (stop_indices, r2 maps) for debugging. Default: False",
            "return_debug_info": "With region growing, add debug maps to the result: stop_index_map_iter1, region_mask and fit_order_map. Default: False",
            "use_multiprocessing": "Enable parallel processing. Default: True",
            "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
        },
        "output_parameters": {
            "amplitude": "Complex signal amplitude from sinusoidal fit",
            "period": "Fitted period parameter (in sqrt(power) units, related to B1 field strength)",
            "refpower_90deg_W": "Reference power required for 90° flip angle (Watts), adjusted for pulse parameters",
            "rss_b1": "Residual sum of squares for the sinusoidal fit (minimum R² from optimization)",
            "fit_order_map": "Only with region growing and return_debug_info=True: zero-based order in which voxels were fitted (seed = 0, NaN if never fitted)",
        },
        "model_equation": MODEL_EQUATION,
    }


def prepare_b1_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
    return_params_info: bool = False,
    *,
    base_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Prepare the public quantitative fit-params dataset for B1 fitting."""
    if return_params_info:
        return None

    fit_params = fit_params or {}
    fa_deg = fit_params.get("fa_deg", None)
    trf_ms = fit_params.get("trf_ms", None)
    sint = fit_params.get("sint", None)
    dynamic_dim = fit_params.get("dynamic_dim")
    if dynamic_dim is None:
        candidate_dims = fit_params.get(
            "dynamic_dim_candidates",
            ["rfpower_idx", "power_idx", "rf_power_idx", "rfpower"],
        )
        dynamic_dim = next((dim for dim in candidate_dims if dim in data.dims), None)
        if dynamic_dim is None:
            raise ValueError(
                "Unable to determine dynamic dimension for B1 parameters. "
                "Provide 'dynamic_dim' in fit_params."
            )

    coord_refpower = data.coords.get(dynamic_dim, None)
    if coord_refpower is None:
        refpower_values = fit_params.get("refpower_W", None)
    else:
        if coord_refpower.ndim == 1 and coord_refpower.dims[0] == dynamic_dim:
            refpower_values = coord_refpower.values
        else:
            coord_sel = {dim: 0 for dim in coord_refpower.dims if dim != dynamic_dim}
            coord_sel[dynamic_dim] = slice(None)
            refpower_values = coord_refpower.isel(coord_sel).values

    if refpower_values is None:
        raise ValueError(f"No RF power coordinate found for '{dynamic_dim}'.")

    power_values = np.asarray(refpower_values, dtype=float).squeeze()
    period_min = float(fit_params.get("period_min", 0.2))
    period_max = fit_params.get("period_max", None)
    period_samples = int(fit_params.get("period_samples", 1000))

    if period_max is None:
        sqrt_power_values = np.sqrt(np.clip(power_values, a_min=0.0, a_max=None))
        finite_sqrt = sqrt_power_values[np.isfinite(sqrt_power_values)]
        period_max = (
            float(np.nanmax(finite_sqrt) * 2.5) if finite_sqrt.size > 0 else 2.5
        )

    period_grid = np.linspace(period_min, period_max, period_samples)

    attrs = dict(base_attrs or {})
    attrs.update(
        {
            "fit_type": "b1",
            "model_id": "b1",
            "model_name": RESULT_MODEL_NAME,
            "dynamic_dim": dynamic_dim,
            "refpower_W": power_values,
            "fa_deg": fa_deg,
            "trf_ms": trf_ms,
            "sint": sint,
            "period_min": period_min,
            "period_max": period_max,
            "period_samples": period_samples,
            "_b1_period_grid": period_grid,
        }
    )
    copy_execution_params(fit_params, attrs)
    if fit_params.get("use_region_growing", False):
        configured_params, *_ = _configure_b1_fit(data, fit_params)
        attrs.update(configured_params)
    return xr.Dataset(attrs=attrs)


def _configure_b1_fit(
    data: xr.DataArray,
    fit_params: dict | None = None,
) -> tuple[dict, str, str, str, list[str]]:
    """Normalize B1 fit inputs for the legacy quantitative executor."""
    fit_params = dict(fit_params or {})
    dynamic_dim = fit_params.get("dynamic_dim")
    if dynamic_dim is None:
        candidate_dims = fit_params.get(
            "dynamic_dim_candidates",
            ["rfpower_idx", "power_idx", "rf_power_idx", "rfpower"],
        )
        dynamic_dim = next((dim for dim in candidate_dims if dim in data.dims), None)
        if dynamic_dim is None:
            raise ValueError(
                "Unable to determine the dynamic RF power dimension for B1 fitting. "
                "Please supply 'dynamic_dim' in fit_params."
            )

    if dynamic_dim not in data.dims:
        raise ValueError(
            f"Dynamic dimension '{dynamic_dim}' not found in data dims {data.dims}."
        )

    power_coord = data.coords.get(dynamic_dim, None)
    power_values = None
    if power_coord is not None:
        if power_coord.ndim == 1 and power_coord.dims[0] == dynamic_dim:
            power_values = power_coord.values.astype(float)
        else:
            coord_sel = {dim: 0 for dim in power_coord.dims if dim != dynamic_dim}
            coord_sel[dynamic_dim] = slice(None)
            power_values = power_coord.isel(coord_sel).values.astype(float)

        power_values = np.asarray(power_values, dtype=float).squeeze()
        if power_values.size == 0:
            raise ValueError(
                "RF power coordinate appears empty after selection. "
                "Check the dynamic dimension setup."
            )
        fit_params.setdefault("refpower_W", power_values)
    elif "refpower_W" in fit_params:
        power_values = np.asarray(fit_params["refpower_W"], dtype=float).squeeze()
    else:
        raise ValueError(
            f"No RF power coordinate found for '{dynamic_dim}'. Provide "
            "'refpower_W' in fit_params."
        )

    period_grid = fit_params.get("period_grid", None)
    if period_grid is None:
        period_min = float(fit_params.get("period_min", 0.2))
        period_max = fit_params.get("period_max", None)
        period_samples = int(fit_params.get("period_samples", 1000))

        if period_samples < 2:
            raise ValueError(
                "B1 fitting requires at least two period samples. "
                "Increase 'period_samples' in fit_params."
            )

        sqrt_power_values = np.sqrt(np.clip(power_values, a_min=0.0, a_max=None))
        finite_sqrt = sqrt_power_values[np.isfinite(sqrt_power_values)]
        debug_log_b1(f"B1 fit: finite sqrt power values: {finite_sqrt}")
        if finite_sqrt.size == 0:
            finite_sqrt = np.array([1.0], dtype=float)

        if period_max is None:
            period_max = float(np.nanmax(finite_sqrt) * 2.5)
            debug_log_b1(f"B1 fit: auto-set period_max to {period_max} based on data")
            if not np.isfinite(period_max) or period_max <= 0:
                period_max = 2.5

        if period_min <= 0:
            period_min = 0.1

        if period_max <= period_min:
            period_max = period_min * 1.5

        period_grid = np.linspace(period_min, period_max, period_samples)
        debug_log_b1(
            f"B1 fit: generated period_grid from {period_min} to {period_max} with {period_samples} samples"
        )
    else:
        period_grid = np.asarray(period_grid, dtype=float).squeeze()
        if period_grid.ndim != 1 or period_grid.size < 2:
            raise ValueError(
                "'period_grid' must be a 1D array with at least two entries."
            )

    fit_params["_b1_period_grid"] = np.asarray(period_grid, dtype=float)
    fit_params["dynamic_dim"] = dynamic_dim
    fit_params.setdefault("min_valid_samples", 3)
    fit_params.setdefault("period_min", float(period_grid[0]))
    fit_params.setdefault("period_max", float(period_grid[-1]))
    fit_params.setdefault("period_samples", int(period_grid.size))

    seed_voxel = fit_params.get("seed_voxel", None)
    fit_params.get("use_region_growing", False)

    debug_log_b1("=" * 80)
    debug_log_b1("===== B1 Fitting: Seed Initialization =====")

    if seed_voxel is not None:
        seed_voxel_signal = data.isel(seed_voxel).squeeze().values
        debug_log_b1(
            f"B1 fit: seed voxel at {seed_voxel} has signal {seed_voxel_signal}"
        )

        sqrt_power_values = np.sqrt(np.clip(power_values, a_min=0.0, a_max=None))
        seed_first_iteration_result = None
        for _k in range(4):
            seed_fit_results = fit_single_voxel_b1_period_sweep_helper(
                signal=seed_voxel_signal,
                sqrt_refpower=sqrt_power_values,
                fit_params=fit_params,
            )
            if _k == 0:
                # Keep the full-sweep seed fit so its values populate the
                # exported *_iter1 maps just as they do for neighbour voxels.
                seed_first_iteration_result = dict(seed_fit_results)
            n_valid_samples = seed_fit_results.get("n_valid_samples", len(power_values))
            period = seed_fit_results.get("period", None)
            stop_index_extra = fit_params.get("stop_index_extra", 0)
            period_idx = np.searchsorted(sqrt_power_values, period)
            stop_index = period_idx + stop_index_extra
            print(
                f"B1 fit: seed voxel initial period fit result: {period}, stop_index: {stop_index}"
            )
            if stop_index > n_valid_samples:
                stop_index = n_valid_samples

            debug_log_b1(
                f"B1 fit: seed voxel initial period fit result: {period}, stop_index: {stop_index}"
            )
            fit_params["stop_index"] = stop_index

        seed_fit_results["stop_index"] = stop_index
        seed_fit_results["_b1_period_grid"] = fit_params["_b1_period_grid"]
        fit_params["seed_fit_results"] = seed_fit_results
        if fit_params.get("use_two_iteration_fit", True):
            fit_params["seed_first_iteration_result"] = seed_first_iteration_result

    if fit_params.get("db_mode", False):
        print(f"Fitting {MODEL_ID} with the following parameters:")
        print(f"  - dynamic_dim: {dynamic_dim}")
        print(f"  - period_min: {fit_params['period_min']}")
        print(f"  - period_max: {fit_params['period_max']}")
        print(f"  - period_samples: {fit_params['period_samples']}")

    return (
        fit_params,
        RESULT_MODEL_NAME,
        MODEL_EQUATION,
        dynamic_dim,
        list(OUTPUT_PARAM_NAMES),
    )


def fit_single_voxel_b1_period_sweep_helper(
    signal: np.ndarray,
    sqrt_refpower: np.ndarray,
    fit_params: dict,
) -> dict:
    """Fit the sinusoidal B1 model by sweeping candidate periods."""
    period_grid = fit_params["_b1_period_grid"]
    min_valid_samples = int(fit_params.get("min_valid_samples", 3))

    neighbor_period = fit_params.get("period", None)
    neighbor_period_search_width = float(
        fit_params.get("neighbor_period_search_width", 0.3)
    )

    if neighbor_period is not None and np.isfinite(neighbor_period):
        period_samples = int(fit_params.get("period_samples", 1000))
        period_min_local = neighbor_period * (1.0 - neighbor_period_search_width)
        period_max_local = neighbor_period * (1.0 + neighbor_period_search_width)

        global_period_min = fit_params.get("period_min", None)
        global_period_max = fit_params.get("period_max", None)
        if global_period_min is not None:
            period_min_local = max(period_min_local, global_period_min)
        if global_period_max is not None:
            period_max_local = min(period_max_local, global_period_max)

        period_grid = np.linspace(period_min_local, period_max_local, period_samples)
    else:
        debug_log_b1("Using full period grid (no neighbor information)")

    signal = np.asarray(signal, dtype=complex).squeeze()
    sqrt_refpower = np.asarray(sqrt_refpower, dtype=float).squeeze()

    finite_mask = np.isfinite(signal) & np.isfinite(sqrt_refpower)
    signal_valid = signal[finite_mask]
    sqrt_refpower_valid = sqrt_refpower[finite_mask]
    n_valid = signal_valid.size

    stop_index = int(fit_params.get("stop_index", n_valid))
    min_stop_index = int(fit_params.get("min_stop_index", 2))
    if stop_index < min_stop_index:
        stop_index = min_stop_index

    signal_valid = signal_valid[: stop_index + 1]
    sqrt_refpower_valid = sqrt_refpower_valid[: stop_index + 1]

    if n_valid < min_valid_samples or period_grid.size == 0:
        debug_log_b1("Insufficient valid samples or empty period grid.")
        return {
            "amplitude": np.nan,
            "period": np.nan,
            "refpower_90deg_W": np.nan,
            "rss_b1": np.nan,
            "stop_index": np.nan,
            "n_valid_samples": float(n_valid),
        }

    debug_log_b1(f"sqrt_refpower_valid: {sqrt_refpower_valid}")

    sin_matrix = np.sin(
        np.pi / period_grid[np.newaxis, :] * sqrt_refpower_valid[:, np.newaxis]
    )
    numerator = np.sum(np.conjugate(sin_matrix) * signal_valid[:, np.newaxis], axis=0)
    denom = np.sum(np.abs(sin_matrix) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        amps = np.divide(
            numerator,
            denom,
            out=np.zeros_like(numerator, dtype=complex),
            where=denom > 0,
        )

    residuals = np.sum(
        np.abs(signal_valid[:, np.newaxis] - sin_matrix * amps[np.newaxis, :]) ** 2,
        axis=0,
    )

    valid_period_mask = denom > 0
    if not np.any(valid_period_mask):
        return {
            "amplitude": np.nan,
            "period": np.nan,
            "refpower_90deg_W": np.nan,
            "rss_b1": np.nan,
            "stop_index": np.nan,
            "n_valid_samples": float(n_valid),
        }

    masked_residuals = np.where(valid_period_mask, residuals, np.inf)
    best_idx = int(np.argmin(masked_residuals))

    best_period = float(period_grid[best_idx])
    best_amp = amps[best_idx]
    best_residual = float(residuals[best_idx])

    if neighbor_period is not None and np.isfinite(neighbor_period):
        debug_log_b1(
            "Using neighbor-informed period search: "
            f"center={neighbor_period:.3f}, range=[{period_min_local:.3f}, {period_max_local:.3f}], best_period={best_period:.3f}",
        )

    from hypermri.analysis.anatomical.b1_mapping import (
        refpow_from_sin_fit,
        refpow_parameter_adjustment,
    )

    refpower_90deg = refpow_from_sin_fit(periods=best_period)
    debug_log_b1(
        f"1/ refpower_90deg (pre-adjustment) = {refpower_90deg:.3f} W",
        write_to_file=False,
    )
    refpower_90deg = refpow_parameter_adjustment(
        refpower_W=refpower_90deg,
        trf_ms=fit_params.get("trf_ms", 1),
        sint=fit_params.get("sint", 1),
        fa_deg=fit_params.get("fa_deg", 90),
    )
    debug_log_b1(
        f"2/ refpower_90deg (pre-adjustment) = {refpower_90deg:.3f} W",
        write_to_file=False,
    )

    if np.any(finite_mask):
        finite_indices = np.nonzero(finite_mask)[0]
        stop_index = float(finite_indices.max())
    else:
        stop_index = np.nan

    return {
        "amplitude": best_amp,
        "period": best_period,
        "refpower_90deg_W": refpower_90deg,
        "rss_b1": best_residual,
        "stop_index": stop_index,
        "n_valid_samples": float(n_valid),
    }


def b1_sinusoidal_model(
    refpower_W: np.ndarray,
    amplitude: xr.DataArray,
    period: xr.DataArray,
    db_mode: bool = False,
) -> xr.DataArray:
    """Reconstruct B1 signal from fitted sinusoidal parameters."""
    if db_mode:
        print("Calculating B1 sinusoidal model:")
        print(f"  - refpower_W: {refpower_W}")
        print(f"  - amplitude shape: {amplitude.shape}")
        print(f"  - period shape: {period.shape}")

    sqrt_refpow_da = xr.DataArray(
        np.sqrt(refpower_W), dims=["rfpower_idx"], coords={"rfpower_idx": refpower_W}
    )

    if db_mode:
        print(f"  - sqrt_refpow_da shape: {sqrt_refpow_da.shape}")

    signal = amplitude * np.sin(np.pi / period * sqrt_refpow_da)

    if db_mode:
        print(f"  - output signal shape: {signal.shape}")

    return signal


_NEIGHBOUR_OFFSETS = {
    4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    8: ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
}

# Start values a grown voxel takes over from its fitted neighbours.
_NEIGHBOUR_START_KEYS = ("amplitude", "period", "refpower_90deg_W", "stop_index")

# Fit settings recorded per voxel in the debug ``init_params_map``.
_RECORDED_START_KEYS = (
    "amplitude",
    "period",
    "refpower_90deg_W",
    "stop_index",
    "use_neighbor_init_values",
    "neighbor_period_search_width",
)

_SIGNAL_PRIORITY_MODES = {"snr", "signal", "max_signal", "max_snr"}


def _seed_indices(seed_voxel) -> tuple[int, int]:
    """Return ``(read_idx, phase_idx)`` of the region-growing seed voxel."""
    if seed_voxel is None:
        raise ValueError("region_growing_b1_fitting requires a 'seed_voxel' entry.")

    if isinstance(seed_voxel, dict):
        seed_read_idx = seed_voxel.get("read_idx", None)
        seed_phase_idx = seed_voxel.get("phase_idx", None)
    else:
        try:
            seed_read_idx, seed_phase_idx = seed_voxel
        except (TypeError, ValueError):
            raise ValueError(
                "seed_voxel must be either a dict with 'read_idx'/'phase_idx' keys or a tuple of (read_idx, phase_idx)."
            )

    if seed_read_idx is None or seed_phase_idx is None:
        raise ValueError(
            "Seed voxel indices could not be determined. "
            "Ensure 'seed_voxel' provides both 'read_idx' and 'phase_idx'."
        )
    return seed_read_idx, seed_phase_idx


def _median_neighbour_fit(
    fit_results: dict, read_idx: int, phase_idx: int, shape: tuple[int, int]
) -> dict:
    """Return the median fit values of the fitted voxels around one voxel.

    The 3 x 3 neighbourhood, including the voxel itself, contributes every
    fitted voxel with a finite period. Keys without contributions are NaN;
    ``n_neighbors`` counts the contributing voxels.
    """
    read_size, phase_size = shape
    neighbour_values = {key: [] for key in _NEIGHBOUR_START_KEYS}
    for dr in (-1, 0, 1):
        for dp in (-1, 0, 1):
            nbr_r = read_idx + dr
            nbr_p = phase_idx + dp
            if not (
                0 <= nbr_r < read_size
                and 0 <= nbr_p < phase_size
                and (nbr_r, nbr_p) in fit_results
            ):
                continue
            fit_res = fit_results[(nbr_r, nbr_p)]
            if np.isnan(fit_res["period"]):
                continue
            neighbour_values["amplitude"].append(np.abs(fit_res["amplitude"]))
            neighbour_values["period"].append(fit_res["period"])
            neighbour_values["refpower_90deg_W"].append(fit_res["refpower_90deg_W"])
            if "stop_index" in fit_res and not np.isnan(fit_res["stop_index"]):
                neighbour_values["stop_index"].append(fit_res["stop_index"])

    median_params = {
        key: np.nanmedian(values) if len(values) > 0 else np.nan
        for key, values in neighbour_values.items()
    }
    median_params["n_neighbors"] = sum(
        1 for val in neighbour_values["period"] if not np.isnan(val)
    )
    debug_log_b1(f"Computed mean neighbor params: {median_params}", write_to_file=False)
    return median_params


def _with_neighbour_start(base_params: dict, neighbour_params: dict) -> dict:
    """Return the fit settings of one voxel with its neighbours' start values.

    ``use_neighbor_init_values=False`` in ``base_params`` keeps the base
    settings. NaN neighbour values never replace a base setting.
    """
    merged = dict(base_params)
    if not base_params.get("use_neighbor_init_values", True):
        return merged
    for key, value in neighbour_params.items():
        if key == "n_neighbors" or value is None:
            continue
        if isinstance(value, float | int | np.floating | np.integer) and np.isnan(
            value
        ):
            continue
        merged[key] = value
    return merged


def _fit_voxel_with_refined_stop(
    signal: np.ndarray, sqrt_refpower: np.ndarray, voxel_params: dict
) -> tuple[dict, float, dict | None]:
    """Fit one voxel, then refit up to the stop index its first period gives.

    With ``use_two_iteration_fit`` (default True) and a finite first period,
    the stop index is the first reference power at or above that period plus
    ``stop_index_extra``, clipped to the last sample and to at least
    ``min_stop_index``, and the voxel is fitted again up to it.

    Returns
    -------
    tuple
        The final fit, its stop index, and the first fit (None without the
        second iteration).
    """
    use_two_iterations = voxel_params.get("use_two_iteration_fit", True)
    period_grid = voxel_params.get("_b1_period_grid")

    first_result = fit_single_voxel_b1_period_sweep_helper(
        signal=signal,
        sqrt_refpower=sqrt_refpower,
        fit_params=dict(voxel_params),
    )
    refined_result = dict(first_result)
    refined_result["_b1_period_grid"] = period_grid
    stop_index_val = float(first_result.get("stop_index", np.nan))
    first_iteration_result = dict(first_result) if use_two_iterations else None

    period = first_result.get("period", np.nan)
    if np.isfinite(period) and use_two_iterations:
        stop_index = np.searchsorted(sqrt_refpower, period) + int(
            voxel_params.get("stop_index_extra", 0)
        )
        if sqrt_refpower.size > 0:
            stop_index = min(stop_index, sqrt_refpower.size - 1)
        stop_index = max(stop_index, int(voxel_params.get("min_stop_index", 2)))

        refined_params = dict(voxel_params)
        refined_params["stop_index"] = stop_index
        refined_result = fit_single_voxel_b1_period_sweep_helper(
            signal=signal,
            sqrt_refpower=sqrt_refpower,
            fit_params=refined_params,
        )
        refined_result["_b1_period_grid"] = period_grid
        stop_index_val = float(stop_index)

    refined_result["stop_index"] = float(stop_index_val)
    return refined_result, stop_index_val, first_iteration_result


def _requested_voxels(index_iterator) -> set | None:
    """Return the ``(read_idx, phase_idx)`` pairs named by ``index_iterator``."""
    if index_iterator is None:
        return set()
    pending_voxels = set()
    for indices in index_iterator:
        if isinstance(indices, tuple) and len(indices) >= 2:
            pending_voxels.add((indices[0], indices[1]))
        elif isinstance(indices, dict):
            r = indices.get("read_idx")
            p = indices.get("phase_idx")
            if r is not None and p is not None:
                pending_voxels.add((r, p))
    return pending_voxels or None


class _GrowthQueue:
    """Voxels whose neighbours are still to be fitted.

    In FIFO order by default; with a signal priority the voxel with the
    largest mean signal magnitude comes first.
    """

    def __init__(self, by_signal: bool):
        self._by_signal = by_signal
        self._items = [] if by_signal else deque()

    def __bool__(self) -> bool:
        return len(self._items) > 0

    def push(self, coord: tuple[int, int], signal: np.ndarray) -> None:
        if not self._by_signal:
            self._items.append(coord)
            return
        values = np.abs(np.asarray(signal).ravel())
        finite = np.isfinite(values)
        priority = float(np.nanmean(values[finite])) if finite.any() else float("-inf")
        heapq.heappush(self._items, (-priority, coord))

    def pop(self) -> tuple[int, int]:
        if not self._by_signal:
            return self._items.popleft()
        return heapq.heappop(self._items)[1]


class _GrowthProgressPlot:
    """Optional live plot of the region mask and stop-index map."""

    def __init__(self, region_mask: np.ndarray, stop_index_map: np.ndarray):
        self.active = False
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display

            self._plt = plt
            self._fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            self._region_im = axes[0].imshow(region_mask, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Region Mask")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            self._stop_index_im = axes[1].imshow(stop_index_map, cmap="viridis")
            axes[1].set_title("Stop Index")
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            plt.tight_layout()
            self._display_handle = display(self._fig, display_id=True)
            self.active = True
        except Exception:
            self.active = False

    def update(
        self, region_mask: np.ndarray, stop_index_map: np.ndarray, pause_s: float
    ) -> None:
        self._region_im.set_data(region_mask)
        self._stop_index_im.set_data(stop_index_map)
        finite_stop = np.isfinite(stop_index_map)
        if np.any(finite_stop):
            vmin = float(np.nanmin(stop_index_map[finite_stop]))
            vmax = float(np.nanmax(stop_index_map[finite_stop]))
            if vmax > vmin:
                self._stop_index_im.set_clim(vmin=vmin, vmax=vmax)
        if self._display_handle is not None:
            self._display_handle.update(self._fig)
        else:
            self._plt.pause(pause_s)


def region_growing_b1_fitting(
    data: xr.DataArray,
    fit_params: dict,
    max_iterations: int = None,
    connectivity: int = 4,
):
    """
    Perform region-growing B1 fitting starting from a seed voxel.

    Starting from the seed, the fit grows into neighbouring voxels. Each new
    voxel starts from the median fit of its already fitted 3 x 3 neighbours
    and is fitted twice (see ``use_two_iteration_fit``). When the queue runs
    empty before every voxel is fitted, growth restarts from a voxel of
    ``index_iterator`` or the first unfitted voxel in read-major order.

    Parameters
    ----------
    data : xr.DataArray
        Complex-valued MRI data with dimensions ``read_idx``, ``phase_idx``
        and ``rfpower_idx``; other dimensions must be singleton.
    fit_params : dict
        Fitting parameters. Required: ``seed_voxel`` (``(read_idx,
        phase_idx)`` or a dict with these keys), ``seed_fit_results`` (the
        seed's fit), ``refpower_W`` (reference powers in W along
        ``rfpower_idx``) and ``_b1_period_grid``. Optional:
        ``use_two_iteration_fit`` (default True), ``stop_index_extra``
        (default 0), ``min_stop_index`` (default 2),
        ``use_neighbor_init_values`` (default True),
        ``region_growing_priority`` (``"fifo"`` by default, or ``"snr"``,
        ``"signal"``, ``"max_signal"``, ``"max_snr"`` to grow from the
        strongest voxel first), ``index_iterator``,
        ``seed_first_iteration_result``, ``return_debug_info`` (default
        False) and ``plot_region_growing_progress`` (default False).
    max_iterations : int, optional
        Maximum number of queue voxels whose neighbours are fitted. If None,
        processes all voxels.
    connectivity : int, default 4
        Neighbor connectivity (4 or 8).

    Returns
    -------
    fit_results : dict
        Dictionary mapping (read_idx, phase_idx) to final fit results.
    first_iteration_results : dict, optional
        Only returned if 'use_two_iteration_fit' is True and results are available.
    debug_info : dict, optional
        Only returned if 'return_debug_info' is True. Contains
        ``stop_index_map`` (float, shape ``(read_idx, phase_idx)``),
        ``init_params_map``, ``region_mask`` (bool) and ``fit_order_map``
        (float, shape ``(read_idx, phase_idx)``). ``fit_order_map`` holds the
        zero-based position at which each voxel was fitted: the seed voxel is
        0 and the last fitted voxel is ``n_fitted - 1``. Voxels that were
        never fitted (e.g. when ``max_iterations`` stops the growth early) are
        NaN.

    Raises
    ------
    ValueError
        If the seed voxel or its fit is missing or malformed, or if
        ``connectivity`` is neither 4 nor 8.
    """
    seed = _seed_indices(fit_params.get("seed_voxel", None))
    seed_fit_result = fit_params.get("seed_fit_results", None)
    return_debug_info = fit_params.get("return_debug_info", False)
    use_two_iteration_fit = fit_params.get("use_two_iteration_fit", True)
    if seed_fit_result is None:
        raise ValueError(
            "region_growing_b1_fitting requires precomputed 'seed_fit_results' for the seed voxel."
        )

    squeeze_dims = [
        d
        for d in data.dims
        if d not in ["read_idx", "phase_idx", "rfpower_idx"] and data.sizes[d] == 1
    ]
    debug_log_b1(f"Squeezing dimensions for 2D extraction: {squeeze_dims}")
    data_2d = data.squeeze(dim=squeeze_dims, drop=True) if squeeze_dims else data
    shape = (data_2d.sizes["read_idx"], data_2d.sizes["phase_idx"])

    region_mask = np.zeros(shape, dtype=bool)
    fit_order_map = np.full(shape, np.nan, dtype=float)
    stop_index_map = np.full(shape, np.nan, dtype=float)
    init_params_map = {} if return_debug_info else None
    fit_results = {seed: dict(seed_fit_result)}
    first_iteration_results = {}
    seed_first_iter = fit_params.get("seed_first_iteration_result", None)
    if use_two_iteration_fit and seed_first_iter is not None:
        first_iteration_results[seed] = dict(seed_first_iter)
    region_mask[seed] = True
    fit_order_map[seed] = 0.0
    stop_index_map[seed] = float(seed_fit_result.get("stop_index", np.nan))
    n_fitted = 1

    sqrt_refpower = np.sqrt(
        np.clip(
            np.asarray(fit_params["refpower_W"], dtype=float), a_min=0.0, a_max=None
        )
    )
    if connectivity not in _NEIGHBOUR_OFFSETS:
        raise ValueError("connectivity must be 4 or 8")
    offsets = _NEIGHBOUR_OFFSETS[connectivity]

    pending_voxels = _requested_voxels(fit_params.get("index_iterator", None))
    priority_mode = fit_params.get("region_growing_priority", "fifo")
    priority_mode = priority_mode.lower() if isinstance(priority_mode, str) else "fifo"
    queue = _GrowthQueue(by_signal=priority_mode in _SIGNAL_PRIORITY_MODES)
    queue.push(seed, data_2d.isel(read_idx=seed[0], phase_idx=seed[1]).values)

    progress = None
    if fit_params.get("plot_region_growing_progress", False):
        progress = _GrowthProgressPlot(region_mask, stop_index_map)
        if not progress.active:
            progress = None

    def fit_and_record(coord: tuple[int, int], pause_s: float) -> None:
        nonlocal n_fitted
        neighbour_params = _median_neighbour_fit(fit_results, *coord, shape)
        voxel_params = _with_neighbour_start(fit_params, neighbour_params)
        if return_debug_info:
            init_params_map[coord] = {
                "neighbor_params": dict(neighbour_params),
                "merged_fit_params": {
                    k: v for k, v in voxel_params.items() if k in _RECORDED_START_KEYS
                },
            }
        signal = data_2d.isel(read_idx=coord[0], phase_idx=coord[1]).values
        fit_res, stop_idx_val, first_iter_res = _fit_voxel_with_refined_stop(
            signal, sqrt_refpower, voxel_params
        )
        fit_results[coord] = fit_res
        if first_iter_res is not None:
            first_iteration_results[coord] = first_iter_res
        region_mask[coord] = True
        fit_order_map[coord] = n_fitted
        n_fitted += 1
        stop_index_map[coord] = stop_idx_val
        queue.push(coord, signal)
        if pending_voxels and coord in pending_voxels:
            pending_voxels.discard(coord)
        if progress is not None:
            progress.update(region_mask, stop_index_map, pause_s)

    iterations = 0
    while queue and (max_iterations is None or iterations < max_iterations):
        curr_read, curr_phase = queue.pop()
        debug_log_b1(
            f"Processing voxel at read_idx={curr_read}, phase_idx={curr_phase}",
            write_to_file=False,
        )
        if pending_voxels and (curr_read, curr_phase) in pending_voxels:
            pending_voxels.discard((curr_read, curr_phase))

        for dr, dp in offsets:
            nbr = (curr_read + dr, curr_phase + dp)
            if (
                0 <= nbr[0] < shape[0]
                and 0 <= nbr[1] < shape[1]
                and not region_mask[nbr]
            ):
                fit_and_record(nbr, pause_s=0.0001)

        iterations += 1
        if queue and max_iterations is not None and iterations >= max_iterations:
            debug_log_b1(
                f"Reached max_iterations={max_iterations}; stopping region growing."
            )
            break

        if not queue:
            next_voxel = _next_region_start(pending_voxels, region_mask)
            if next_voxel is not None:
                debug_log_b1(
                    f"Queue empty; starting new region from voxel {next_voxel}."
                )
                fit_and_record(next_voxel, pause_s=0.01)

    if return_debug_info:
        debug_info = {
            "stop_index_map": stop_index_map,
            "init_params_map": init_params_map,
            "region_mask": region_mask,
            "fit_order_map": fit_order_map,
        }
        if use_two_iteration_fit and first_iteration_results:
            return fit_results, first_iteration_results, debug_info
        return fit_results, debug_info
    if use_two_iteration_fit and first_iteration_results:
        return fit_results, first_iteration_results
    return fit_results


def _next_region_start(pending_voxels, region_mask: np.ndarray):
    """Return a voxel to restart growth from, or None when none is left.

    A still requested voxel comes first; otherwise the first unfitted voxel
    in read-major order.
    """
    if pending_voxels:
        return next(iter(pending_voxels), None)
    unfitted = np.argwhere(~region_mask)
    if unfitted.size == 0:
        return None
    return (int(unfitted[0, 0]), int(unfitted[0, 1]))


def fit_b1_region_growing_dataset(
    data: xr.DataArray,
    fit_params: dict,
    executor: VoxelwiseExecutor | None = None,
) -> xr.Dataset:
    """Run region growing and assemble its spatial B1 parameter maps."""
    dynamic_dim = str(fit_params.get("dynamic_dim", "rfpower_idx"))
    if dynamic_dim not in data.dims:
        raise ValueError(
            f"Dynamic dimension '{dynamic_dim}' not found in data dims {data.dims}."
        )

    non_dynamic_dims = [dim for dim in data.dims if dim != dynamic_dim]
    unsupported_dims = [
        dim
        for dim in non_dynamic_dims
        if dim not in {"read_idx", "phase_idx"} and data.sizes[dim] != 1
    ]
    if unsupported_dims:
        raise ValueError(
            "B1 region growing is two-dimensional; additional dimensions must "
            f"be singleton. Non-singleton dimensions: {unsupported_dims}."
        )

    try:
        read_idx_pos = non_dynamic_dims.index("read_idx")
        phase_idx_pos = non_dynamic_dims.index("phase_idx")
    except ValueError as exc:
        raise ValueError(
            "B1 region growing requires 'read_idx' and 'phase_idx' dimensions."
        ) from exc

    region_output = region_growing_b1_fitting(data, fit_params)
    fit_results = region_output
    first_iteration_results = None
    debug_info = None
    if isinstance(region_output, tuple):
        if len(region_output) == 3:
            fit_results, first_iteration_results, debug_info = region_output
        elif len(region_output) == 2:
            fit_results, optional_results = region_output
            if fit_params.get("return_debug_info", False):
                debug_info = optional_results
            else:
                first_iteration_results = optional_results
        else:
            fit_results = region_output[0]

    executor = executor or VoxelwiseExecutor()

    def assemble(results: dict[tuple[int, int], dict]) -> xr.Dataset:
        def lookup_result(*, signal, dynamic_coords, fit_params, indices):
            del signal, dynamic_coords, fit_params
            coordinate = (indices[read_idx_pos], indices[phase_idx_pos])
            return results.get(coordinate), indices

        return executor.execute(
            data=data,
            model_id=MODEL_ID,
            fit_params=fit_params,
            single_voxel_fit_func=lookup_result,
            output_param_names=OUTPUT_PARAM_NAMES,
            dynamic_dim=dynamic_dim,
            model_name=RESULT_MODEL_NAME,
        )

    result = assemble(fit_results)
    if first_iteration_results:
        first_iteration = assemble(first_iteration_results)
        for param_name in OUTPUT_PARAM_NAMES:
            result[f"{param_name}_iter1"] = first_iteration[param_name]

    if debug_info:
        stop_index_map = np.asarray(debug_info.get("stop_index_map", []))
        if stop_index_map.ndim == 2:
            debug_dims = ("read_idx", "phase_idx")
            debug_coords = {}
            for axis, dim in enumerate(debug_dims):
                coordinate = data.coords.get(dim)
                debug_coords[dim] = (
                    coordinate
                    if coordinate is not None and coordinate.dims == (dim,)
                    else np.arange(stop_index_map.shape[axis])
                )
            result["stop_index_map_iter1"] = xr.DataArray(
                stop_index_map,
                coords=debug_coords,
                dims=debug_dims,
            )
            if "region_mask" in debug_info:
                result["region_mask"] = xr.DataArray(
                    np.asarray(debug_info["region_mask"], dtype=bool),
                    coords=debug_coords,
                    dims=debug_dims,
                )
            if "fit_order_map" in debug_info:
                result["fit_order_map"] = xr.DataArray(
                    np.asarray(debug_info["fit_order_map"], dtype=float),
                    coords=debug_coords,
                    dims=debug_dims,
                    attrs={
                        "long_name": "Region-growing fit order",
                        "description": (
                            "Zero-based position at which each voxel was fitted; "
                            "the seed voxel is 0. NaN for voxels that were not fitted."
                        ),
                    },
                )
        if "init_params_map" in debug_info:
            result.attrs["has_init_params_map"] = True
    return result
