"""DCE preprocessing helpers shared by fitting workflows."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.api import fit
from hypermri.utils.image.slices import get_overlapping_slices
from hypermri.utils.mri.channels import combine_channels

from .shared import (
    _coerce_numeric_1d,
    _spatial_template,
    _to_numpy,
)


def prepare_dce_metadata(
    dce_data: xr.DataArray,
    *,
    tr_s: float | None = None,
    fa_deg: float | None = None,
    aif_times: np.ndarray | xr.DataArray | None = None,
    time_points_s: np.ndarray | xr.DataArray | None = None,
    require_acquisition_params: bool = True,
) -> dict:
    """Collect DCE timing and acquisition metadata."""
    if tr_s is None:
        tr_ms = dce_data.attrs.get("repetition_time_ms")
        if tr_ms is not None:
            tr_s = float(tr_ms) / 1000.0

    if fa_deg is None:
        fa_deg = dce_data.attrs.get("flipangle_deg")

    if require_acquisition_params and (tr_s is None or fa_deg is None):
        raise ValueError(
            "TR (seconds) and flip angle (degrees) must be provided or present in dce_data.attrs."
        )

    if "repetition_idx" not in dce_data.dims:
        raise ValueError("dce_data must contain a 'repetition_idx' dimension.")

    n_repetitions = int(dce_data.sizes["repetition_idx"])
    resolved_time_points_s = None

    if time_points_s is not None:
        resolved_time_points_s = _coerce_numeric_1d("time_points_s", time_points_s)
        if aif_times is not None:
            legacy_time_points_s = _coerce_numeric_1d("aif_times", aif_times)
            if len(legacy_time_points_s) != len(
                resolved_time_points_s
            ) or not np.allclose(legacy_time_points_s, resolved_time_points_s):
                raise ValueError(
                    "time_points_s and aif_times were both provided but do not match."
                )
    elif aif_times is not None:
        resolved_time_points_s = _coerce_numeric_1d("aif_times", aif_times)
    elif "time" in dce_data.coords:
        resolved_time_points_s = _coerce_numeric_1d(
            "dce_data.coords['time']", dce_data.coords["time"]
        )
    elif "frame_time_s" in dce_data.attrs:
        resolved_time_points_s = _coerce_numeric_1d(
            "dce_data.attrs['frame_time_s']", dce_data.attrs["frame_time_s"]
        )
    elif hasattr(dce_data, "frame_time_s"):
        resolved_time_points_s = _coerce_numeric_1d(
            "dce_data.frame_time_s", dce_data.frame_time_s
        )
    elif "repetition_idx" in dce_data.coords:
        repetition_coord = _to_numpy(dce_data.coords["repetition_idx"]).squeeze()
        unit = str(dce_data.coords["repetition_idx"].attrs.get("unit", "")).lower()
        if (
            repetition_coord.ndim == 1
            and len(repetition_coord) == n_repetitions
            and unit in {"s", "sec", "second", "seconds"}
        ):
            resolved_time_points_s = repetition_coord.astype(np.float64, copy=False)

    if resolved_time_points_s is None and tr_s is None:
        raise ValueError("A DCE time axis must be provided when TR is unavailable.")
    if resolved_time_points_s is None:
        resolved_time_points_s = np.arange(n_repetitions, dtype=np.float64) * float(
            tr_s
        )

    if len(resolved_time_points_s) != n_repetitions:
        raise ValueError(
            "The supplied time axis length does not match dce_data.repetition_idx."
        )

    if not np.all(np.isfinite(resolved_time_points_s)):
        raise ValueError("DCE time points must be finite.")

    return {
        "tr_s": np.nan if tr_s is None else float(tr_s),
        "fa_deg": np.nan if fa_deg is None else float(fa_deg),
        "time_points_s": resolved_time_points_s.astype(np.float64, copy=False),
    }


def standardize_t1_map_units(t1_map: xr.DataArray) -> xr.DataArray:
    """Return a T1 map in milliseconds with explicit units metadata."""
    t1_ms = t1_map.astype(np.float64, copy=False)
    units = str(t1_ms.attrs.get("units", "")).strip().lower()

    finite = np.asarray(t1_ms.values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    finite_median = float(np.nanmedian(np.abs(finite))) if finite.size else np.nan

    convert_from_seconds = units in {"s", "sec", "second", "seconds"}
    heuristic_seconds = (
        units == "" and np.isfinite(finite_median) and 0 < finite_median < 20
    )

    if convert_from_seconds or heuristic_seconds:
        t1_ms = t1_ms * 1000.0
        if heuristic_seconds:
            warnings.warn(
                "T1 map units were not declared and values looked like seconds. "
                "They were converted to milliseconds for DCE analysis.",
                stacklevel=2,
            )

    t1_ms.attrs["units"] = "ms"
    return t1_ms


def _align_subset_to_template(
    template: xr.DataArray,
    subset: xr.DataArray,
    *,
    target_slice_indices: list[int] | None = None,
) -> xr.DataArray:
    subset = subset.astype(np.float64, copy=False)

    if (
        target_slice_indices is None
        or "slice_idx" not in template.dims
        or "slice_idx" not in subset.dims
    ):
        # try:
        #     template_slice_indices, subset_slice_indices = get_overlapping_slices(
        #         data_obj_1=template,
        #         data_obj_2=subset,
        #     )
        aligned = subset.broadcast_like(template)
        return aligned.transpose(*template.dims)

    target = xr.full_like(template, np.nan, dtype=np.float64)
    common_dims = [dim for dim in target.dims if dim in subset.dims]
    subset_aligned = subset.transpose(*common_dims, ...)
    subset_values = subset_aligned.values
    target_block = target.isel(slice_idx=target_slice_indices)

    if subset_values.shape != target_block.shape:
        try:
            subset_values = subset_values.reshape(target_block.shape)
        except ValueError as exc:
            raise ValueError(
                "Could not align the T1 subset to the DCE template. "
                f"Subset shape {subset_values.shape}, target shape {target_block.shape}."
            ) from exc

    full_values = target.values.copy()
    slice_axis = target.dims.index("slice_idx")
    slicer = [slice(None)] * target.ndim
    slicer[slice_axis] = target_slice_indices
    full_values[tuple(slicer)] = subset_values
    target.values = full_values
    return target


def align_t1_to_dce_grid(
    dce_data: xr.DataArray,
    t1_data: xr.DataArray | None = None,
    *,
    fitted_t1: xr.Dataset | xr.DataArray | None = None,
    fit_params_t1: dict | None = None,
    use_multiprocessing: bool = True,
    return_debug_maps: bool = False,
) -> (
    tuple[xr.DataArray, xr.DataArray | None]
    | tuple[xr.DataArray, xr.DataArray | None, dict[str, xr.DataArray]]
):
    """Fit or accept T1 data and align it to the DCE spatial grid.

    Parameters
    ----------
    dce_data : xr.DataArray
        The DCE data array, used to determine the spatial template and slice positions.
    t1_data : xr.DataArray, optional
        Raw T1 data to be fitted. Ignored if fitted_t1 is provided.
    fitted_t1 : xr.Dataset or xr.DataArray, optional
        Pre-fitted T1 map dataset or data array. If a dataset, it must contain a 't1' variable. If provided, this will be used directly without fitting.
    fit_params_t1 : dict, optional
        Parameters to pass to the T1 fitting function if t1_data is provided and fitted_t1 is not. Ignored if fitted_t1 is provided.
    use_multiprocessing : bool, default True
        Whether to use multiprocessing for T1 fitting when t1_data is provided and fitted_t1 is not. Ignored if fitted_t1 is provided.
    return_debug_maps : bool, default False
        Whether to return additional debug maps (e.g., RSS of T1 fit) when fitted_t1 is provided and contains them.
         Ignored if fitted_t1 is not provided.

    Returns
    -------
    tuple
        If return_debug_maps is False:
            - t1_map_aligned: xr.DataArray
                The T1 map aligned to the DCE spatial grid.
            - weights_aligned: xr.DataArray or None
                The channel combination weights aligned to the DCE spatial grid, or None if not applicable.
        If return_debug_maps is True:
            - t1_map_aligned: xr.DataArray
                The T1 map aligned to the DCE spatial grid.
            - weights_aligned: xr.DataArray or None
                The channel combination weights aligned to the DCE spatial grid, or None if not applicable.
            - debug_maps: dict[str, xr.DataArray]
                A dictionary of additional debug maps aligned to the DCE spatial grid, such as the RSS.
    """
    if t1_data is None and fitted_t1 is None:
        raise ValueError("Provide either raw t1_data or fitted_t1.")

    fit_params_t1 = dict(fit_params_t1 or {})
    template_spatial = _spatial_template(dce_data)

    if fitted_t1 is not None:
        if isinstance(fitted_t1, xr.Dataset):
            if "t1" not in fitted_t1:
                raise ValueError("fitted_t1 dataset must contain a 't1' variable.")
            fitted_t1_ds = fitted_t1
            t1_reference = fitted_t1["t1"]
        elif isinstance(fitted_t1, xr.DataArray):
            fitted_t1_ds = xr.Dataset({"t1": fitted_t1})
            t1_reference = fitted_t1
        else:
            raise TypeError("fitted_t1 must be an xarray Dataset or DataArray.")
    else:
        fitted_t1_ds = None
        t1_reference = t1_data

    dce_slice_indices = None
    t1_slice_indices = None

    if "slice_idx" in dce_data.dims and "slice_idx" in t1_reference.dims:
        try:
            dce_slice_indices, t1_slice_indices = get_overlapping_slices(
                data_obj_1=dce_data,
                data_obj_2=t1_reference,
            )
        except Exception:
            if dce_data.sizes["slice_idx"] == t1_reference.sizes["slice_idx"]:
                dce_slice_indices = list(range(dce_data.sizes["slice_idx"]))
                t1_slice_indices = list(range(t1_reference.sizes["slice_idx"]))
            else:
                raise

        if not dce_slice_indices:
            raise ValueError(
                "No overlapping slice positions were found between DCE and T1 data."
            )

    if fitted_t1_ds is None:
        t1_input = t1_data
        if t1_slice_indices is not None:
            t1_input = t1_data.isel(slice_idx=t1_slice_indices)

        fit_params_t1["use_multiprocessing"] = use_multiprocessing
        t1_results_subset = fit(
            data=t1_input,
            model="t1",
            fit_params=fit_params_t1,
        )
    else:
        t1_results_subset = fitted_t1_ds
        if t1_slice_indices is not None and "slice_idx" in t1_results_subset.dims:
            t1_results_subset = t1_results_subset.isel(slice_idx=t1_slice_indices)

    t1_map_subset = t1_results_subset["t1"]
    weights_subset = None

    if (
        "channel_idx" in t1_results_subset.dims
        and t1_results_subset.sizes["channel_idx"] > 1
    ):
        weights_subset = np.abs(t1_results_subset["A"])
        t1_map_subset = combine_channels(
            data=t1_results_subset["t1"],
            method="weighted_mean",
            weights=weights_subset,
        )

    if "channel_idx" in t1_map_subset.dims and t1_map_subset.sizes["channel_idx"] == 1:
        t1_map_subset = t1_map_subset.isel(channel_idx=0, drop=True)

    t1_map_subset = standardize_t1_map_units(t1_map_subset)
    t1_map = _align_subset_to_template(
        template_spatial,
        t1_map_subset,
        target_slice_indices=dce_slice_indices,
    )

    weights = None
    if weights_subset is not None:
        weight_template = dce_data.isel(repetition_idx=0, drop=True)
        if (
            "channel_idx" in weight_template.dims
            and "channel_idx" not in weights_subset.dims
        ):
            weight_template = weight_template.isel(channel_idx=0, drop=True)
        weights = _align_subset_to_template(
            weight_template.astype(np.float64),
            weights_subset.astype(np.float64),
            target_slice_indices=dce_slice_indices,
        )

    if not return_debug_maps:
        return t1_map, weights

    debug_maps: dict[str, xr.DataArray] = {}
    if fitted_t1_ds is not None and "rss_t1" in fitted_t1_ds:
        rss_t1_subset = fitted_t1_ds["rss_t1"]
        if t1_slice_indices is not None and "slice_idx" in rss_t1_subset.dims:
            rss_t1_subset = rss_t1_subset.isel(slice_idx=t1_slice_indices)
        rss_t1_subset = rss_t1_subset.astype(np.float64, copy=False)
        if (
            "channel_idx" in rss_t1_subset.dims
            and rss_t1_subset.sizes["channel_idx"] > 1
        ):
            if weights_subset is not None:
                try:
                    rss_t1_subset = combine_channels(
                        data=rss_t1_subset,
                        method="weighted_mean",
                        weights=weights_subset,
                    )
                except Exception:
                    rss_t1_subset = combine_channels(rss_t1_subset, method="mean")
            else:
                rss_t1_subset = combine_channels(rss_t1_subset, method="mean")
        if (
            "channel_idx" in rss_t1_subset.dims
            and rss_t1_subset.sizes["channel_idx"] == 1
        ):
            rss_t1_subset = rss_t1_subset.isel(channel_idx=0, drop=True)

        debug_maps["rss_t1_map"] = _align_subset_to_template(
            template_spatial.astype(np.float64),
            rss_t1_subset,
            target_slice_indices=dce_slice_indices,
        )

    return t1_map, weights, debug_maps


def combine_multichannel_data(
    dce_data: xr.DataArray,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Combine multi-channel DCE data using T1-derived weights when available."""
    if "channel_idx" not in dce_data.dims:
        return dce_data

    if dce_data.sizes.get("channel_idx", 1) <= 1:
        return dce_data.isel(channel_idx=0, drop=True)

    if weights is not None:
        try:
            return combine_channels(
                data=dce_data,
                method="weighted_mean",
                weights=weights,
            )
        except Exception as exc:
            warnings.warn(
                f"Weighted DCE channel combination failed ({exc}). Falling back to RSS.",
                stacklevel=2,
            )

    return combine_channels(dce_data, method="rss")


def compute_s0_map(
    dce_combined: xr.DataArray,
    *,
    n_precontrast: int = 5,
    method: str = "mean",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Estimate baseline S0 and its voxelwise pre-contrast standard deviation."""
    if "repetition_idx" not in dce_combined.dims:
        raise ValueError("dce_combined must contain 'repetition_idx'.")

    # Number of frames
    n_frames = int(dce_combined.sizes["repetition_idx"])
    # First n_pre frames to calculate s0 from
    n_pre = max(1, min(int(n_precontrast), n_frames))

    if method == "median":
        return dce_combined.isel(repetition_idx=slice(0, n_pre)).median(
            dim="repetition_idx"
        ), dce_combined.isel(repetition_idx=slice(0, n_pre)).std(dim="repetition_idx")
    else:
        return dce_combined.isel(repetition_idx=slice(0, n_pre)).mean(
            dim="repetition_idx"
        ), dce_combined.isel(repetition_idx=slice(0, n_pre)).std(dim="repetition_idx")
