from __future__ import annotations

"""Canonical renal DCE workflows and post-processing helpers."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .aif import _detect_bolus_arrival_index
from .fitting_support import fit_patlak_model, fit_sourbron_model, fit_tofts_model
from .shared import (
    _align_mask_weights_to_template,
    _apply_explicit_aif_options,
    _build_fit_dispatch_metadata,
    _mask_weight_sum,
    _merge_fit_params,
    _set_var_attrs,
    _spatial_dims,
    _weighted_mean_over_mask,
    _weighted_quantile,
    _weighted_sum_over_mask,
    _weighted_values_from_mask,
)


def _align_mask_to_template(
    mask,
    template: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    return _align_mask_weights_to_template(mask, template, name=name)


def _canonical_mask_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _prepare_renal_masks(
    renal_masks: dict[str, xr.DataArray | np.ndarray] | None,
    template: xr.DataArray,
) -> dict[str, xr.DataArray]:
    if not renal_masks:
        return {}

    aligned_masks: dict[str, xr.DataArray] = {}
    for name, mask in renal_masks.items():
        aligned_masks[_canonical_mask_name(name)] = _align_mask_to_template(
            mask,
            template,
            name=name,
        )

    all_masks = dict(aligned_masks)

    left_masks = [mask for name, mask in aligned_masks.items() if "left" in name]
    right_masks = [mask for name, mask in aligned_masks.items() if "right" in name]
    cortex_masks = [mask for name, mask in aligned_masks.items() if "cortex" in name]
    medulla_masks = [mask for name, mask in aligned_masks.items() if "medulla" in name]

    if "left_kidney" not in all_masks and left_masks:
        all_masks["left_kidney"] = xr.concat(left_masks, dim="_mask_union").max(
            dim="_mask_union"
        )
    if "right_kidney" not in all_masks and right_masks:
        all_masks["right_kidney"] = xr.concat(right_masks, dim="_mask_union").max(
            dim="_mask_union"
        )
    if "cortex" not in all_masks and cortex_masks:
        all_masks["cortex"] = xr.concat(cortex_masks, dim="_mask_union").max(
            dim="_mask_union"
        )
    if "medulla" not in all_masks and medulla_masks:
        all_masks["medulla"] = xr.concat(medulla_masks, dim="_mask_union").max(
            dim="_mask_union"
        )

    whole_candidates = list(all_masks.values())
    if "whole_kidney" not in all_masks and whole_candidates:
        all_masks["whole_kidney"] = xr.concat(whole_candidates, dim="_mask_union").max(
            dim="_mask_union"
        )

    preferred_order = [
        "whole_kidney",
        "left_kidney",
        "right_kidney",
        "cortex",
        "medulla",
    ]
    ordered: dict[str, xr.DataArray] = {}
    for name in preferred_order:
        if name in all_masks:
            ordered[name] = all_masks[name]
    for name, mask in all_masks.items():
        if name not in ordered:
            ordered[name] = mask

    return ordered


def _fit_roi_models(
    concentration_map: xr.DataArray,
    *,
    fit_type: str,
    fit_params: dict,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    renal_masks: dict[str, xr.DataArray | np.ndarray] | None,
) -> xr.Dataset:
    template = concentration_map.isel(repetition_idx=0, drop=True)
    masks = _prepare_renal_masks(renal_masks, template)
    if not masks:
        return xr.Dataset()

    spatial_dims = _spatial_dims(concentration_map)
    roi_labels: list[str] = []
    roi_curves: list[np.ndarray] = []
    roi_counts: list[int] = []
    roi_values: dict[str, list[float]] = {}

    for label, mask in masks.items():
        voxel_count = _mask_weight_sum(mask)
        if voxel_count <= 0:
            continue

        curve = _weighted_mean_over_mask(
            concentration_map,
            mask,
            dims=spatial_dims,
        )
        if not np.any(np.isfinite(curve.values)):
            continue

        roi_input = xr.DataArray(
            curve.values.astype(np.float64),
            dims=["repetition_idx"],
            coords={"repetition_idx": concentration_map.coords["repetition_idx"]},
            name=f"{label}_concentration_curve",
        )
        if "time_s" in concentration_map.coords:
            roi_input = roi_input.assign_coords(
                time_s=concentration_map.coords["time_s"]
            )

        if fit_type == "patlak":
            roi_fit = fit_patlak_model(
                roi_input,
                aif_concentration=aif_concentration,
                aif_time_s=aif_time_s,
                fit_params=fit_params,
                use_multiprocessing=False,
            )
        elif fit_type == "tofts":
            roi_fit = fit_tofts_model(
                roi_input,
                aif_concentration=aif_concentration,
                aif_time_s=aif_time_s,
                fit_params=fit_params,
                use_multiprocessing=False,
            )
        elif fit_type == "sourbron":
            roi_fit = fit_sourbron_model(
                roi_input,
                aif_concentration=aif_concentration,
                aif_time_s=aif_time_s,
                fit_params=fit_params,
                use_multiprocessing=False,
            )
        else:
            continue

        roi_labels.append(label)
        roi_curves.append(curve.values.astype(np.float64))
        roi_counts.append(voxel_count)
        for var_name in roi_fit.data_vars:
            if roi_fit[var_name].ndim != 0:
                continue
            roi_values.setdefault(var_name, []).append(
                float(np.asarray(roi_fit[var_name].values).squeeze())
            )

    if not roi_labels:
        return xr.Dataset()

    roi_dataset = xr.Dataset(
        {
            "roi_concentration_curve": xr.DataArray(
                np.stack(roi_curves, axis=0),
                dims=["roi_label", "repetition_idx"],
                coords={
                    "roi_label": roi_labels,
                    "repetition_idx": concentration_map.coords["repetition_idx"],
                },
            ),
            "roi_voxel_count": xr.DataArray(
                np.asarray(roi_counts, dtype=np.float64),
                dims=["roi_label"],
                coords={"roi_label": roi_labels},
            ),
        }
    )

    if "time_s" in concentration_map.coords:
        roi_dataset["roi_concentration_curve"] = roi_dataset[
            "roi_concentration_curve"
        ].assign_coords(time_s=concentration_map.coords["time_s"])

    for var_name, values in roi_values.items():
        roi_dataset[f"roi_{var_name}"] = xr.DataArray(
            np.asarray(values, dtype=np.float64),
            dims=["roi_label"],
            coords={"roi_label": roi_labels},
        )

    if fit_type == "patlak":
        if "roi_K_trans" in roi_dataset:
            roi_dataset["roi_patlak_slope"] = roi_dataset["roi_K_trans"].copy(deep=True)
            roi_dataset["roi_filtration_index"] = roi_dataset["roi_K_trans"].copy(
                deep=True
            )
        if "roi_Vp" in roi_dataset:
            roi_dataset["roi_patlak_intercept"] = roi_dataset["roi_Vp"].copy(deep=True)
    return roi_dataset


def summarize_renal_parameters(
    dataset: xr.Dataset,
    *,
    renal_masks: dict[str, xr.DataArray | np.ndarray] | None = None,
    parameter_names: list[str] | None = None,
    gfr_conversion: dict | None = None,
) -> xr.Dataset:
    """Add kidney-level summary variables to a fitted renal DCE dataset."""

    if not renal_masks:
        return dataset

    spatial_template = None
    for candidate in ("t1_map", "s0_map", "patlak_slope", "K_trans", "Fp"):
        if candidate in dataset and "repetition_idx" not in dataset[candidate].dims:
            spatial_template = dataset[candidate]
            break

    if spatial_template is None:
        return dataset

    masks = _prepare_renal_masks(renal_masks, spatial_template)
    if not masks:
        return dataset

    if parameter_names is None:
        primary_fit_type = str(dataset.attrs.get("primary_fit_type", "patlak"))
        if primary_fit_type == "patlak":
            parameter_names = [
                "patlak_slope",
                "patlak_intercept",
                "filtration_index",
                "rss",
                "rmse",
                "r_squared",
            ]
        elif primary_fit_type == "tofts":
            parameter_names = [
                "Ktrans_1c",
                "kep_1c",
                "Ktrans_2c",
                "kep_2c",
                "vp_2c",
                "ve_2c",
                "rss",
                "rmse",
                "r_squared",
            ]
        elif primary_fit_type == "sourbron":
            parameter_names = [
                "vp",
                "Ft",
                "Tt",
                "tubular_flow_s",
                "tubular_flow_ml_min_100ml",
                "tubular_transit_time_s",
                "rss",
                "rmse",
                "r_squared",
            ]
        else:
            parameter_names = [
                "renal_plasma_flow",
                "renal_extravascular_fraction",
                "renal_blood_volume_fraction",
                "rss",
                "rmse",
                "r_squared",
            ]

    region_names = list(masks.keys())
    stat_names = ["mean", "median", "std", "voxel_count"]
    template_dims = tuple(spatial_template.dims)

    for parameter_name in parameter_names:
        if parameter_name not in dataset:
            continue

        data_array = dataset[parameter_name]
        if "repetition_idx" in data_array.dims:
            continue

        try:
            data_array = data_array.broadcast_like(spatial_template).transpose(
                *template_dims
            )
        except Exception:
            continue

        summary_values = np.full(
            (len(region_names), len(stat_names)), np.nan, dtype=np.float64
        )

        for region_index, (region_name, mask_da) in enumerate(masks.items()):
            finite, weights = _weighted_values_from_mask(data_array, mask_da)
            if finite.size == 0:
                continue

            weight_sum = float(np.sum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                continue

            weighted_mean = float(np.sum(finite * weights) / weight_sum)
            weighted_var = float(
                np.sum(weights * (finite - weighted_mean) ** 2) / weight_sum
            )
            summary_values[region_index, 0] = weighted_mean
            summary_values[region_index, 1] = _weighted_quantile(finite, weights, 0.5)
            summary_values[region_index, 2] = float(np.sqrt(max(weighted_var, 0.0)))
            summary_values[region_index, 3] = weight_sum

        dataset[f"{parameter_name}_summary"] = xr.DataArray(
            summary_values,
            dims=["renal_region", "summary_stat"],
            coords={"renal_region": region_names, "summary_stat": stat_names},
        )

    if gfr_conversion:
        parameter_name = gfr_conversion.get("parameter_name", "filtration_index")
        scale_factor = gfr_conversion.get("scale_factor_ml_min_per_unit", None)
        assumptions = gfr_conversion.get("assumptions", None)
        normalization = gfr_conversion.get("normalization", "user_supplied")

        summary_name = f"{parameter_name}_summary"
        if parameter_name in dataset and scale_factor is not None and assumptions:
            kidney_regions = [name for name in region_names if "kidney" in name]
            if kidney_regions:
                mean_values = dataset[summary_name].sel(
                    summary_stat="mean", renal_region=kidney_regions
                )
                gfr_estimate = mean_values * float(scale_factor)
                dataset["single_kidney_gfr_estimate"] = gfr_estimate
                _set_var_attrs(
                    dataset,
                    "single_kidney_gfr_estimate",
                    long_name="Single-kidney GFR Estimate",
                    description=(
                        "Scaled estimate derived from a renal summary parameter. "
                        f"Assumptions: {assumptions}. Normalization: {normalization}."
                    ),
                    units=str(gfr_conversion.get("units", "mL/min")),
                    model="scaled_filtration_index",
                )

    return dataset


def _to_scalar(value) -> float:
    array = np.asarray(getattr(value, "values", value), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(array.squeeze())


def collect_extended_tofts_roi_metrics(
    fit_results: xr.Dataset,
    masks: xr.Dataset | Mapping[str, xr.DataArray | np.ndarray],
    *,
    voxel_volume_ml: float,
    spatial_dims: tuple[str, ...] | list[str],
    subject_name: str,
    fit_results_2comp: xr.Dataset | None = None,
    spatial_dims_2comp: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, float | int | str]:
    """Collect ROI metrics from extended Tofts renal-flow aliases and 2-compartment fits."""
    if "tubular_flow_ml_min_100ml" not in fit_results:
        raise KeyError("fit_results must contain 'tubular_flow_ml_min_100ml'.")
    if "vp" not in fit_results:
        raise KeyError("fit_results must contain 'vp'.")

    if isinstance(masks, xr.Dataset):
        mask_items = list(masks.data_vars.items())
    else:
        mask_items = list(masks.items())

    row: dict[str, float | int | str] = {"animal_id": str(subject_name)}
    gfr_by_mask: dict[str, float] = {}
    template = fit_results["tubular_flow_ml_min_100ml"]
    voxel_volume_ml = _to_scalar(voxel_volume_ml)

    for mask_name, mask in mask_items:
        mask_da = _align_mask_to_template(mask, template, name=mask_name)
        tubular_flow_masked_sum = _weighted_sum_over_mask(
            fit_results["tubular_flow_ml_min_100ml"],
            mask_da,
            dims=spatial_dims,
        )
        tubular_flow_mean = _to_scalar(
            _weighted_mean_over_mask(
                fit_results["tubular_flow_ml_min_100ml"],
                mask_da,
                dims=spatial_dims,
            )
        )
        plasma_volume_fraction_mean = _to_scalar(
            _weighted_mean_over_mask(
                fit_results["vp"],
                mask_da,
                dims=spatial_dims,
            )
        )

        roi_voxel_count = _mask_weight_sum(mask_da)
        roi_volume_ml = float(roi_voxel_count * voxel_volume_ml)
        signal = _to_scalar(tubular_flow_masked_sum)
        gfr_ml_min = roi_volume_ml * signal / 100.0
        gfr_by_mask[mask_name] = gfr_ml_min

        row[f"{mask_name}_voxel_count"] = roi_voxel_count
        row[f"{mask_name}_volume_ml"] = roi_volume_ml
        row[f"{mask_name}_mean_tubular_flow_ml_min_100ml"] = tubular_flow_mean
        row[f"{mask_name}_mean_plasma_volume_fraction"] = plasma_volume_fraction_mean
        row[f"{mask_name}_signal"] = signal
        row[f"{mask_name}_gfr_ml_min"] = gfr_ml_min

        if fit_results_2comp is not None:
            required_2comp = ("Ktrans_2c", "kep_2c", "vp_2c", "ve_2c")
            missing_2comp = [
                var_name
                for var_name in required_2comp
                if var_name not in fit_results_2comp
            ]
            if missing_2comp:
                missing_str = ", ".join(missing_2comp)
                raise KeyError(
                    f"fit_results_2comp is missing required variables: {missing_str}."
                )

            dims_2comp = spatial_dims_2comp
            if dims_2comp is None:
                dims_2comp = tuple(dim for dim in template.dims if dim in mask_da.dims)

            row[f"{mask_name}_mean_ktrans_2c_x1e6"] = (
                _to_scalar(
                    _weighted_mean_over_mask(
                        fit_results_2comp["Ktrans_2c"],
                        mask_da,
                        dims=dims_2comp,
                    )
                )
                * 1e6
            )
            row[f"{mask_name}_mean_kep_2c_x1e3"] = (
                _to_scalar(
                    _weighted_mean_over_mask(
                        fit_results_2comp["kep_2c"],
                        mask_da,
                        dims=dims_2comp,
                    )
                )
                * 1e3
            )
            row[f"{mask_name}_mean_vp_2c_percent"] = (
                _to_scalar(
                    _weighted_mean_over_mask(
                        fit_results_2comp["vp_2c"],
                        mask_da,
                        dims=dims_2comp,
                    )
                )
                * 100.0
            )
            row[f"{mask_name}_mean_ve_2c_percent"] = (
                _to_scalar(
                    _weighted_mean_over_mask(
                        fit_results_2comp["ve_2c"],
                        mask_da,
                        dims=dims_2comp,
                    )
                )
                * 100.0
            )

    left_candidates = [
        name
        for name in gfr_by_mask
        if "left" in name.lower() and "cortex" in name.lower()
    ]
    right_candidates = [
        name
        for name in gfr_by_mask
        if "right" in name.lower() and "cortex" in name.lower()
    ]
    if not left_candidates:
        left_candidates = [name for name in gfr_by_mask if "left" in name.lower()]
    if not right_candidates:
        right_candidates = [name for name in gfr_by_mask if "right" in name.lower()]

    left_gfr = gfr_by_mask[left_candidates[0]] if left_candidates else float("nan")
    right_gfr = gfr_by_mask[right_candidates[0]] if right_candidates else float("nan")

    if np.isfinite(left_gfr) or np.isfinite(right_gfr):
        mean_gfr = float(np.nanmean([left_gfr, right_gfr]))
    elif gfr_by_mask:
        mean_gfr = float(np.nanmean(list(gfr_by_mask.values())))
    else:
        mean_gfr = float("nan")

    row["mean_gfr_ml_min"] = mean_gfr
    row["left_gfr_ml_min"] = left_gfr
    row["right_gfr_ml_min"] = right_gfr

    ordered_row = {"animal_id": row["animal_id"]}
    for column_name in ("mean_gfr_ml_min", "left_gfr_ml_min", "right_gfr_ml_min"):
        ordered_row[column_name] = row[column_name]
    for key, value in row.items():
        if key not in ordered_row:
            ordered_row[key] = value

    return ordered_row


def upsert_metrics_row_to_csv(
    row: Mapping[str, float | int | str],
    csv_path: str | Path,
    *,
    id_column: str = "animal_id",
) -> pd.DataFrame:
    """Insert or replace a metrics row in a CSV based on the subject identifier."""
    if id_column not in row:
        raise KeyError(f"row must contain '{id_column}'.")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([dict(row)])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if id_column in existing.columns:
            existing = existing[existing[id_column].astype(str) != str(row[id_column])]
        output = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        output = new_row

    output.to_csv(csv_path, index=False)
    return output


def save_extended_tofts_roi_metrics_csv(
    fit_results: xr.Dataset,
    masks: xr.Dataset | Mapping[str, xr.DataArray | np.ndarray],
    *,
    voxel_volume_ml: float,
    spatial_dims: tuple[str, ...] | list[str],
    subject_name: str,
    csv_path: str | Path,
    fit_results_2comp: xr.Dataset | None = None,
    spatial_dims_2comp: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, float | int | str]:
    """Collect ROI metrics and upsert them into a single CSV file."""
    row = collect_extended_tofts_roi_metrics(
        fit_results,
        masks,
        voxel_volume_ml=voxel_volume_ml,
        spatial_dims=spatial_dims,
        subject_name=subject_name,
        fit_results_2comp=fit_results_2comp,
        spatial_dims_2comp=spatial_dims_2comp,
    )
    upsert_metrics_row_to_csv(row, csv_path)
    return row


def _assess_aif_delay_and_curve_stability(
    concentration_map: xr.DataArray,
    aif_info: dict,
    *,
    renal_masks: dict[str, xr.DataArray | np.ndarray] | None = None,
    n_precontrast: int = 5,
) -> list[str]:
    messages = list(aif_info.get("qc_messages", []))
    template = concentration_map.isel(repetition_idx=0, drop=True)
    masks = _prepare_renal_masks(renal_masks, template)
    spatial_dims = _spatial_dims(concentration_map)

    if "whole_kidney" in masks:
        kidney_curve = _weighted_mean_over_mask(
            concentration_map,
            masks["whole_kidney"],
            dims=spatial_dims,
        )
    else:
        kidney_curve = concentration_map.mean(dim=spatial_dims, skipna=True)

    kidney_curve_np = np.asarray(kidney_curve.values, dtype=np.float64)
    kidney_bolus_index = _detect_bolus_arrival_index(
        kidney_curve_np,
        n_precontrast=n_precontrast,
    )
    aif_bolus_index = aif_info.get("bolus_arrival_index")

    if kidney_bolus_index is not None and aif_bolus_index is not None:
        time_points_s = np.asarray(aif_info["aif_time_s"], dtype=np.float64)
        delay_s = float(
            time_points_s[kidney_bolus_index] - time_points_s[aif_bolus_index]
        )
        if delay_s < -1.0:
            messages.append(
                "Kidney enhancement appears to start before the AIF bolus. "
                "Check temporal alignment and possible motion-related inconsistency."
            )
        elif delay_s > float(np.median(np.diff(time_points_s)) * 5.0):
            messages.append(
                "Kidney enhancement lags the AIF by several frames. "
                "A vascular delay mismatch may bias voxel-wise fitting."
            )

    temporal_diff = concentration_map.diff("repetition_idx")
    if temporal_diff.size > 0:
        instability = np.nanmean(np.abs(temporal_diff.values))
        if np.isfinite(instability) and instability > 5.0 * max(
            float(np.nanmedian(np.abs(aif_info["aif_concentration"]))),
            1e-6,
        ):
            messages.append(
                "Voxel-wise concentration curves show large frame-to-frame jumps relative to the AIF. "
                "Residual motion or unstable voxels are likely present."
            )

    return messages


def _apply_model_specific_outputs(dataset: xr.Dataset, fit_type: str) -> xr.Dataset:
    if fit_type == "patlak":
        dataset["patlak_slope"] = dataset["K_trans"].copy(deep=True)
        dataset["patlak_intercept"] = dataset["Vp"].copy(deep=True)
        dataset["filtration_index"] = dataset["patlak_slope"].copy(deep=True)
    return dataset


def run_renal_dce_pipeline(
    dce_data: xr.DataArray,
    t1_data: xr.DataArray | None = None,
    *,
    fitted_t1: xr.Dataset | xr.DataArray | None = None,
    aif_signal: np.ndarray | xr.DataArray | None = None,
    aif_signal_concentration: np.ndarray | xr.DataArray | None = None,
    aif_times: np.ndarray | xr.DataArray | None = None,
    aif_voxel: Mapping[str, int] | None = None,
    aif_mask: xr.DataArray | np.ndarray | None = None,
    tr_s: float | None = None,
    fa_deg: float | None = None,
    r1: float | None = None,
    fit_params_t1: dict | None = None,
    fit_params_pk: dict | None = None,
    fit_params_tofts: dict | None = None,
    fit_type: str = "patlak",
    use_multiprocessing: bool = True,
    motion_correct_dce=None,
    renal_masks: dict[str, xr.DataArray | np.ndarray] | None = None,
    gfr_conversion: dict | None = None,
    aif_t1_ms: float | None = None,
    aif_t1_s: float | None = None,
    aif_convert_blood_to_plasma: bool | None = None,
    aif_hematocrit: float | None = None,
    skip_fit_quality_metrics: bool = False,
    debug_display: bool = False,
) -> xr.Dataset:
    """
    Run the renal-focused DCE pipeline with optional ROI summaries and QC output.

    This wrapper specializes :func:`hypermri.analysis.dce.pipeline.run_dce_pipeline`
    for renal DCE workflows. In addition to the core voxel-wise fit, it can add:

    - ROI mean concentration curves for supplied renal masks
    - ROI parameter summaries
    - kidney-focused aliases such as ``patlak_slope`` and
      ``filtration_index`` for Patlak outputs
    - AIF and curve-stability QC messages

    Supported fit types are ``"patlak"``, ``"tofts"``, and ``"sourbron"``.

    The fitter used underneath depends on ``fit_type``:

    - ``"patlak"`` routes to :func:`hypermri.analysis.dce.fitting_support.fit_patlak_model`
    - ``"tofts"`` routes to :func:`hypermri.analysis.dce.fitting_support.fit_tofts_model`
    - ``"sourbron"`` routes to :func:`hypermri.analysis.dce.fitting_support.fit_sourbron_model`

    Parameters
    ----------
    dce_data : xr.DataArray
        DCE signal series with a ``repetition_idx`` dimension and spatial
        dimensions that are kept in the output.  The input is not modified.
    t1_data : xr.DataArray, optional
        T1 data to fit when ``fitted_t1`` is not given.
    fitted_t1 : xr.Dataset or xr.DataArray, optional
        Pre-fitted T1 map, or a dataset with a ``t1`` variable.
    aif_signal : np.ndarray or xr.DataArray, optional
        Raw arterial signal over time, converted to concentration with ``r1``.
    aif_signal_concentration : np.ndarray or xr.DataArray, optional
        Arterial concentration in mM over time, used without conversion.
    aif_times : np.ndarray or xr.DataArray, optional
        Acquisition time of each repetition in s.
    aif_voxel : Mapping[str, int], optional
        Spatial dimension name to index of the AIF voxel.
    aif_mask : xr.DataArray or np.ndarray, optional
        Boolean AIF mask on the spatial grid of ``dce_data``.
    tr_s : float, optional
        Repetition time in s; read from ``dce_data.attrs`` when omitted.
    fa_deg : float, optional
        Flip angle in degrees; read from ``dce_data.attrs`` when omitted.
    r1 : float, optional
        Longitudinal relaxivity in 1/(mM s).  May instead be given as
        ``fit_params_pk["r1"]``; when both are given they must be equal.
        Defaults to 3.5 1/(mM s) when neither is given.
    fit_params_t1 : dict, optional
        Options for the T1 fit of ``t1_data``.
    fit_params_pk : dict, optional
        Model options; copied and not modified.
    fit_params_tofts : dict, optional
        Legacy model options, merged before ``fit_params_pk``.
    fit_type : {"patlak", "tofts", "sourbron"}, default "patlak"
        Kinetic model.
    use_multiprocessing : bool, default True
        Fit voxels in parallel processes.
    motion_correct_dce : optional
        Motion corrector applied to ``dce_data`` before fitting.
    renal_masks : dict, optional
        Named boolean masks (for example cortex and medulla) on the spatial
        grid for ROI curves and parameter summaries.
    gfr_conversion : dict, optional
        User-supplied conversion of a filtration parameter to GFR, with keys
        such as ``parameter_name``, ``scale_factor_ml_min_per_unit``,
        ``units`` and ``assumptions``.
    aif_t1_ms, aif_t1_s : float, optional
        Blood T1 for the AIF conversion in ms or s.
    aif_convert_blood_to_plasma : bool, optional
        Convert a blood AIF to plasma concentration.
    aif_hematocrit : float, optional
        Hematocrit fraction for the blood-to-plasma conversion.
    skip_fit_quality_metrics : bool, default False
        Skip the residual-based QC metrics.
    debug_display : bool, default False
        Accepted for backward compatibility; has no effect.

    Returns
    -------
    xr.Dataset
        Output of :func:`hypermri.analysis.dce.pipeline.run_dce_pipeline` plus
        renal aliases, ROI summaries and QC messages.  ``attrs["r1"]`` holds
        the effective relaxivity in 1/(mM s).

    Raises
    ------
    ValueError
        If ``fit_type`` is not supported, if ``r1`` and
        ``fit_params_pk["r1"]`` disagree, or if required acquisition
        parameters are missing.
    """
    from .pipeline import _run_dce_pipeline_core

    _ = debug_display  # Reserved for compatibility with the older API.
    fit_type = str(fit_type).lower()
    if fit_type not in {"patlak", "tofts", "sourbron"}:
        raise ValueError(
            "run_renal_dce_pipeline only supports fit_type='patlak', 'tofts', or 'sourbron'."
        )

    merged_pk_params = _merge_fit_params(
        fit_params_pk=fit_params_pk,
        fit_params_tofts=fit_params_tofts,
        default_model_mode="1comp",
    )
    merged_pk_params = _apply_explicit_aif_options(
        merged_pk_params,
        aif_convert_blood_to_plasma=aif_convert_blood_to_plasma,
        aif_hematocrit=aif_hematocrit,
    )
    dispatch_metadata = _build_fit_dispatch_metadata(
        requested_fit_type=fit_type,
        requested_fit_params=merged_pk_params,
        effective_fit_type=fit_type,
        effective_fit_params=merged_pk_params,
        default_model_mode="1comp",
    )

    return _run_dce_pipeline_core(
        dce_data,
        t1_data=t1_data,
        fitted_t1=fitted_t1,
        aif_signal=aif_signal,
        aif_signal_concentration=aif_signal_concentration,
        aif_times=aif_times,
        aif_voxel=aif_voxel,
        aif_mask=aif_mask,
        tr_s=tr_s,
        fa_deg=fa_deg,
        r1=r1,
        fit_params_t1=fit_params_t1,
        fit_params_pk=merged_pk_params,
        fit_type=fit_type,
        use_multiprocessing=use_multiprocessing,
        motion_corrector=motion_correct_dce,
        renal_masks=renal_masks,
        gfr_conversion=gfr_conversion,
        default_model_mode="1comp",
        aif_t1_ms=aif_t1_ms,
        aif_t1_s=aif_t1_s,
        requested_fit_type=fit_type,
        dispatch_metadata=dispatch_metadata,
        apply_renal_postprocessing=True,
        skip_fit_quality_metrics=skip_fit_quality_metrics,
        debug_display=debug_display,
    )


__all__ = [
    "collect_extended_tofts_roi_metrics",
    "run_renal_dce_pipeline",
    "save_extended_tofts_roi_metrics_csv",
    "summarize_renal_parameters",
    "upsert_metrics_row_to_csv",
]
