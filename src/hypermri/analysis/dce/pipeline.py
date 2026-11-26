"""Canonical DCE pipeline orchestration helpers and public entry points."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping

import numpy as np
import xarray as xr

from hypermri.analysis.dce.conversions import resolve_relaxivity
from hypermri.analysis.motion_correction import apply_motion_corrector

from .aif import prepare_aif
from .fitting_support import (
    _add_clear_model_aliases,
    _annotate_output_dataset,
    compute_fit_quality_metrics,
    convert_signal_to_concentration_map,
    fit_empirical_log_logistic_model,
    fit_extended_tofts_model,
    fit_patlak_model,
    fit_sourbron_model,
    fit_standard_tofts_model,
    fit_tofts_model,
    fit_two_c_irf_model,
)
from .model_selection import _DCE_MODELS, _fit_pipeline_model
from .preprocessing import (
    align_t1_to_dce_grid,
    combine_multichannel_data,
    compute_s0_map,
    prepare_dce_metadata,
)
from .shared import (
    _apply_explicit_aif_options,
    _build_fit_dispatch_metadata,
    _merge_fit_params,
    _normalize_legacy_dce_request,
    _weighted_mean_over_mask,
)


def _warn_if_all_nonfinite(data: xr.DataArray, *, name: str, hint: str) -> None:
    values = np.asarray(data.values, dtype=np.float64)
    if not np.any(np.isfinite(values)):
        warnings.warn(
            f"{name} contains no finite values. {hint}",
            stacklevel=3,
        )


def _select_debug_slice_idx(data: xr.DataArray) -> int | None:
    if "slice_idx" not in data.dims or data.sizes.get("slice_idx", 0) == 0:
        return None
    return int(data.sizes["slice_idx"] // 2)


def _extract_debug_view(
    data: xr.DataArray,
    *,
    repetition_idx: int | None = None,
    slice_idx: int | None = None,
) -> xr.DataArray:
    view = data
    if repetition_idx is not None and "repetition_idx" in view.dims:
        view = view.isel(repetition_idx=int(repetition_idx))
    if slice_idx is not None and "slice_idx" in view.dims:
        view = view.isel(slice_idx=int(slice_idx))
    return view.squeeze()


def _debug_print_aif_summary(
    *,
    aif_info: Mapping[str, object],
    n_precontrast: int,
    tr_s: float,
    fa_deg: float,
) -> None:
    raw_scale = aif_info.get("aif_signal_to_concentration_scale", np.nan)
    try:
        formatted_scale = f"{float(raw_scale):.6g}"
    except (TypeError, ValueError):
        formatted_scale = str(raw_scale)

    print("AIF debug summary:")
    print(
        f"  source={aif_info.get('aif_source', 'unknown')}, "
        f"n_precontrast={int(n_precontrast)}, TR={float(tr_s):.5g} s, "
        f"FA={float(fa_deg):.5g} deg"
    )
    print(
        f"  AIF S0 used={float(aif_info.get('aif_s0_used', np.nan)):.6g}, "
        f"AIF T1 used={float(aif_info.get('aif_t1_ms_used', np.nan)):.6g} ms "
        f"({aif_info.get('aif_t1_weighting', 'unknown')})"
    )
    print(
        "  "
        f"AIF concentration kind={aif_info.get('aif_concentration_kind', 'unknown')}, "
        f"signal_to_concentration_mode={aif_info.get('aif_signal_to_concentration_mode', 'unknown')}, "
        f"signal_to_concentration_scale={formatted_scale}, "
        f"blood_to_plasma={bool(aif_info.get('aif_convert_blood_to_plasma', False))}, "
        f"hematocrit={float(aif_info.get('aif_hematocrit', np.nan)):.6g}"
    )
    if int(aif_info.get("aif_mask_voxel_count", 0) or 0) > 0:
        print(
            f"  AIF mask voxels={int(aif_info.get('aif_mask_voxel_count', 0))}, "
            f"valid T1 voxels={int(aif_info.get('aif_valid_t1_voxel_count', 0))}, "
            f"weighted voxels={int(aif_info.get('aif_t1_weighted_voxel_count', 0))}"
        )


def _debug_plot_dce_pipeline(
    *,
    dce_combined: xr.DataArray,
    s0_map: xr.DataArray,
    t1_map: xr.DataArray,
    concentration_map: xr.DataArray | None,
    aif_info: Mapping[str, object],
    n_precontrast: int,
    fit_type: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Failed to import matplotlib for DCE debug plots: {exc}")
        return

    slice_idx = _select_debug_slice_idx(dce_combined)
    n_show = max(1, min(int(n_precontrast), 6))
    fig, axes = plt.subplots(1, n_show + 2, figsize=(4 * (n_show + 2), 4))
    axes = np.atleast_1d(axes)

    for panel_idx in range(n_show):
        ax = axes[panel_idx]
        view = _extract_debug_view(
            dce_combined,
            repetition_idx=panel_idx,
            slice_idx=slice_idx,
        )
        if view.ndim == 2:
            im = ax.imshow(np.asarray(view.values, dtype=np.float64), cmap="gray")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.plot(np.ravel(np.asarray(view.values, dtype=np.float64)))
        ax.set_title(f"S0 frame {panel_idx}")

    s0_view = _extract_debug_view(s0_map, slice_idx=slice_idx)
    ax = axes[-2]
    if s0_view.ndim == 2:
        im = ax.imshow(np.asarray(s0_view.values, dtype=np.float64), cmap="gray")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.plot(np.ravel(np.asarray(s0_view.values, dtype=np.float64)))
    ax.set_title("S0 map")

    t1_view = _extract_debug_view(t1_map, slice_idx=slice_idx)
    ax = axes[-1]
    if t1_view.ndim == 2:
        im = ax.imshow(np.asarray(t1_view.values, dtype=np.float64), cmap="hot")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.plot(np.ravel(np.asarray(t1_view.values, dtype=np.float64)))
    ax.set_title("T1 map (ms)")

    fig.suptitle(f"DCE debug: S0 frames used for {fit_type}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = np.atleast_1d(axes)
    time_points_s = np.asarray(aif_info["aif_time_s"], dtype=np.float64)
    axes[0].plot(
        time_points_s,
        np.asarray(aif_info["aif_signal_raw"], dtype=np.float64),
        label="AIF raw signal",
    )
    axes[0].set_title("AIF raw signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend()

    axes[1].plot(
        time_points_s,
        np.asarray(aif_info["aif_concentration"], dtype=np.float64),
        label="AIF conc from raw mean",
    )
    if concentration_map is not None and aif_info.get("aif_mask") is not None:
        mask_da = aif_info["aif_mask"]
        mask_mean_conc = _weighted_mean_over_mask(
            concentration_map,
            mask_da,
            dims=tuple(mask_da.dims),
        )
        axes[1].plot(
            time_points_s,
            np.asarray(mask_mean_conc.values, dtype=np.float64),
            "--",
            label="Mean voxelwise concentration in AIF mask",
        )
        diff = np.asarray(mask_mean_conc.values, dtype=np.float64) - np.asarray(
            aif_info["aif_concentration"], dtype=np.float64
        )
        if np.any(np.isfinite(diff)):
            print(
                "  Mask AIF concentration comparison: "
                f"mean(|voxelwise_mean - mean_signal_based|)="
                f"{float(np.nanmean(np.abs(diff))):.6g}"
            )
    axes[1].set_title("AIF concentration")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    fig.suptitle("DCE debug: AIF signal and concentration")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _run_dce_pipeline_core(
    dce_data: xr.DataArray,
    *,
    t1_data: xr.DataArray | None = None,
    fitted_t1: xr.Dataset | xr.DataArray | None = None,
    aif_signal: np.ndarray | xr.DataArray | None = None,
    aif_signal_concentration: np.ndarray | xr.DataArray | None = None,
    aif_times: np.ndarray | xr.DataArray | None = None,
    time_points_s: np.ndarray | xr.DataArray | None = None,
    aif_voxel: Mapping[str, int] | tuple | None = None,
    aif_mask: xr.DataArray | np.ndarray | None = None,
    tr_s: float | None = None,
    fa_deg: float | None = None,
    r1: float = None,
    fit_params_t1: dict | None = None,
    fit_params_pk: dict | None = None,
    fit_type: str = "tofts",
    use_multiprocessing: bool = True,
    motion_corrector=None,
    renal_masks: dict[str, xr.DataArray | np.ndarray] | None = None,
    gfr_conversion: dict | None = None,
    default_model_mode: str = "both",
    aif_t1_ms: float | None = None,
    aif_t1_s: float | None = None,
    requested_fit_type: str | None = None,
    dispatch_metadata: Mapping[str, object] | None = None,
    apply_renal_postprocessing: bool = False,
    skip_fit_quality_metrics: bool = False,
    debug_display: bool = False,
) -> xr.Dataset:
    fit_type = str(fit_type).lower()
    requested_fit_label = str(
        (dispatch_metadata or {}).get(
            "requested_fit_type", requested_fit_type or fit_type
        )
    ).lower()
    if fit_type == "auto":
        raise NotImplementedError(
            "fit_type='auto' was left out of this refactor to keep the renal layer auditable. "
            "Use 'patlak', 'tofts', 'sourbron', 'empirical_log_logistic', or '2c_irf'."
        )
    if fit_type not in _DCE_MODELS:
        raise ValueError(
            f"Unsupported DCE fit_type '{fit_type}'; choose one of {tuple(_DCE_MODELS)}."
        )
    model = _DCE_MODELS[fit_type]
    can_fit_empirical_without_t1 = (
        not model.requires_aif and t1_data is None and fitted_t1 is None
    )

    fit_params_pk = _merge_fit_params(
        fit_params_pk=fit_params_pk,
        default_model_mode=default_model_mode,
    )
    metadata = prepare_dce_metadata(
        dce_data,
        tr_s=tr_s,
        fa_deg=fa_deg,
        aif_times=aif_times,
        time_points_s=time_points_s,
        require_acquisition_params=not can_fit_empirical_without_t1,
    )

    if motion_corrector is not None:
        dce_data = apply_motion_corrector(dce_data, motion_corrector)

    if can_fit_empirical_without_t1:
        t1_map = None
        weights = None
        t1_debug_maps = {}
    else:
        t1_map, weights, t1_debug_maps = align_t1_to_dce_grid(
            dce_data,
            t1_data,
            fitted_t1=fitted_t1,
            fit_params_t1=fit_params_t1,
            use_multiprocessing=use_multiprocessing,
            return_debug_maps=True,
        )
        _warn_if_all_nonfinite(
            t1_map,
            name="Aligned T1 map",
            hint="Check fitted_t1 contents, slice overlap between DCE and T1, and T1 units.",
        )

    dce_combined = combine_multichannel_data(dce_data, weights=weights)
    s0_map, s0_std_map = compute_s0_map(
        dce_combined,
        n_precontrast=int(fit_params_pk.get("n_precontrast", 5)),
    )

    normalized_fit_type = fit_type
    requires_aif = model.requires_aif

    r1 = resolve_relaxivity(
        r1, fit_params_pk, operation="run_dce_pipeline", fit_params_name="fit_params_pk"
    )
    # The AIF conversion and the model fitters read the effective value here.
    fit_params_pk["r1"] = r1

    aif_info = None
    if requires_aif:
        aif_info = prepare_aif(
            dce_combined,
            t1_map,
            aif_signal=aif_signal,
            aif_signal_concentration=aif_signal_concentration,
            aif_times=metadata["time_points_s"],
            aif_voxel=aif_voxel,
            aif_mask=aif_mask,
            n_precontrast=int(fit_params_pk.get("n_precontrast", 5)),
            tr_s=metadata["tr_s"],
            fa_deg=metadata["fa_deg"],
            r1=r1,
            weights=weights,
            aif_t1_weight_map=t1_debug_maps.get("rss_t1_map"),
            fit_params=fit_params_pk,
            aif_t1_ms=aif_t1_ms,
            aif_t1_s=aif_t1_s,
        )
    if debug_display and aif_info is not None:
        _debug_print_aif_summary(
            aif_info=aif_info,
            n_precontrast=int(fit_params_pk.get("n_precontrast", 5)),
            tr_s=metadata["tr_s"],
            fa_deg=metadata["fa_deg"],
        )

    if can_fit_empirical_without_t1:
        concentration_map = dce_combined.astype(np.float64, copy=False)
        concentration_map.name = "signal_curve_map"
        if "repetition_idx" in concentration_map.dims:
            concentration_map = concentration_map.assign_coords(
                time_s=(
                    "repetition_idx",
                    np.asarray(metadata["time_points_s"], dtype=np.float64),
                )
            )
        concentration_map.attrs.update(
            {
                "conversion_model": "none",
                "fit_input_space": "signal",
            }
        )
    else:
        concentration_map = convert_signal_to_concentration_map(
            dce_combined,
            s0_map,
            t1_map,
            tr_s=metadata["tr_s"],
            fa_deg=metadata["fa_deg"],
            r1=r1,
            mode=str(fit_params_pk.get("signal_to_concentration_mode", "spgr")),
            linear_scale=fit_params_pk.get("signal_to_concentration_scale", 1.0),
            time_points_s=metadata["time_points_s"],
            saturation_threshold=float(fit_params_pk.get("saturation_threshold", 0.98)),
            saturation_behavior=fit_params_pk.get("saturation_behavior", "nan_all"),
            warn_on_saturation=fit_params_pk.get("warn_on_saturation", False),
        )
        _warn_if_all_nonfinite(
            concentration_map,
            name="Concentration map",
            hint="Voxel-wise kinetic fitting will fail until T1 alignment, baseline S0, and signal scaling are valid.",
        )
        if debug_display:
            if aif_info is not None:
                _debug_plot_dce_pipeline(
                    dce_combined=dce_combined,
                    s0_map=s0_map,
                    t1_map=t1_map,
                    concentration_map=concentration_map,
                    aif_info=aif_info,
                    n_precontrast=int(fit_params_pk.get("n_precontrast", 5)),
                    fit_type=fit_type,
                )

    fit_results = _fit_pipeline_model(
        concentration_map,
        fit_type=normalized_fit_type,
        requested_type=requested_fit_label,
        aif_info=aif_info,
        time_points_s=metadata["time_points_s"],
        fit_params=fit_params_pk,
        use_multiprocessing=use_multiprocessing,
    )

    quality_time_s = (
        np.asarray(aif_info["aif_time_s"], dtype=np.float64)
        if aif_info is not None
        else np.asarray(metadata["time_points_s"], dtype=np.float64)
    )
    quality_aif = (
        np.asarray(aif_info["aif_concentration"], dtype=np.float64)
        if aif_info is not None
        else np.zeros_like(quality_time_s, dtype=np.float64)
    )
    if skip_fit_quality_metrics:
        final_ds = _add_clear_model_aliases(fit_results)
    else:
        fit_quality = compute_fit_quality_metrics(
            fit_results,
            concentration_map,
            fit_type=fit_type,
            aif_concentration=quality_aif,
            aif_time_s=quality_time_s,
            fit_params=fit_params_pk,
        )
        final_ds = xr.merge(
            [_add_clear_model_aliases(fit_results), fit_quality], compat="override"
        )
    final_ds = _add_clear_model_aliases(final_ds)
    if t1_map is not None:
        final_ds["t1_map"] = t1_map
    final_ds["s0_map"] = s0_map
    final_ds["s0_std_map"] = s0_std_map

    if can_fit_empirical_without_t1:
        final_ds["signal_curve_map"] = concentration_map
    else:
        final_ds["concentration_map"] = concentration_map

    if aif_info is not None:
        aif_idx = np.arange(len(aif_info["aif_time_s"]), dtype=int)
        final_ds["aif_signal_raw"] = xr.DataArray(
            np.asarray(aif_info["aif_signal_raw"], dtype=np.float64),
            dims=["aif_idx"],
            coords={"aif_idx": aif_idx},
        )
        final_ds["aif_concentration"] = xr.DataArray(
            np.asarray(aif_info["aif_concentration"], dtype=np.float64),
            dims=["aif_idx"],
            coords={"aif_idx": aif_idx},
        )
        final_ds["aif_time_s"] = xr.DataArray(
            np.asarray(aif_info["aif_time_s"], dtype=np.float64),
            dims=["aif_idx"],
            coords={"aif_idx": aif_idx},
        )

    raw_signal_scale = fit_params_pk.get("signal_to_concentration_scale", 1.0)
    try:
        serialized_signal_scale: float | str = float(raw_signal_scale)
    except (TypeError, ValueError):
        serialized_signal_scale = str(raw_signal_scale).strip().lower()

    raw_aif_scale = (
        aif_info.get("aif_signal_to_concentration_scale", np.nan)
        if aif_info is not None
        else np.nan
    )
    try:
        serialized_aif_scale: float | str = float(raw_aif_scale)
    except (TypeError, ValueError):
        serialized_aif_scale = str(raw_aif_scale).strip().lower()

    final_ds.attrs.update(
        {
            "requested_fit_type": str(
                (dispatch_metadata or {}).get(
                    "requested_fit_type",
                    requested_fit_type or fit_type,
                )
            ).lower(),
            "primary_fit_type": fit_type,
            "fit_model_mode": dispatch_metadata.get(
                "effective_model_mode",
                fit_params_pk.get("model_mode"),
            )
            if dispatch_metadata is not None
            else fit_params_pk.get("model_mode"),
            "tr_s": metadata["tr_s"],
            "fa_deg": metadata["fa_deg"],
            "r1": float(r1),
            "motion_corrected": bool(motion_corrector is not None),
            "fit_input_space": "signal"
            if can_fit_empirical_without_t1
            else "concentration",
            "signal_to_concentration_mode": str(
                "none"
                if can_fit_empirical_without_t1
                else fit_params_pk.get("signal_to_concentration_mode", "spgr")
            )
            .strip()
            .lower(),
            "signal_to_concentration_scale": serialized_signal_scale,
            "fit_quality_metrics_skipped": bool(skip_fit_quality_metrics),
            "estimate_tp": False,
        }
    )
    if aif_info is not None:
        final_ds.attrs.update(
            {
                "aif_bolus_arrival_index": aif_info.get("bolus_arrival_index"),
                "aif_bolus_arrival_time_s": aif_info.get("bolus_arrival_time_s"),
                "aif_source": aif_info.get("aif_source"),
                "aif_s0_used": float(aif_info.get("aif_s0_used", np.nan)),
                "aif_t1_ms_used": float(aif_info.get("aif_t1_ms_used", np.nan)),
                "aif_t1_weighting": str(aif_info.get("aif_t1_weighting", "unknown")),
                "aif_concentration_kind": str(
                    aif_info.get("aif_concentration_kind", "unknown")
                ),
                "aif_signal_to_concentration_mode": str(
                    aif_info.get("aif_signal_to_concentration_mode", "unknown")
                ),
                "aif_signal_to_concentration_scale": serialized_aif_scale,
                "aif_convert_blood_to_plasma": bool(
                    aif_info.get("aif_convert_blood_to_plasma", False)
                ),
                "aif_hematocrit": float(aif_info.get("aif_hematocrit", np.nan)),
                "aif_mask_voxel_count": int(
                    aif_info.get("aif_mask_voxel_count", 0) or 0
                ),
                "aif_valid_t1_voxel_count": int(
                    aif_info.get("aif_valid_t1_voxel_count", 0) or 0
                ),
                "aif_t1_weighted_voxel_count": int(
                    aif_info.get("aif_t1_weighted_voxel_count", 0) or 0
                ),
            }
        )
    if dispatch_metadata:
        final_ds.attrs.update(dispatch_metadata)

    finalize = getattr(model.module, "_finalize_pipeline_metadata", None)
    if finalize is not None:
        finalize(final_ds, fit_params_pk)

    if apply_renal_postprocessing:
        if aif_info is None:
            warnings.warn(
                "Renal postprocessing requires an AIF-supported model. Skipping ROI/AIF-specific renal outputs.",
                stacklevel=2,
            )
            final_ds = _annotate_output_dataset(final_ds, fit_type=fit_type)
            return final_ds

        from hypermri.analysis.dce.renal import (
            _apply_model_specific_outputs,
            _assess_aif_delay_and_curve_stability,
            _fit_roi_models,
            summarize_renal_parameters,
        )

        final_ds = _apply_model_specific_outputs(final_ds, fit_type)

        roi_dataset = _fit_roi_models(
            concentration_map,
            fit_type=fit_type,
            fit_params=fit_params_pk,
            aif_concentration=np.asarray(
                aif_info["aif_concentration"], dtype=np.float64
            ),
            aif_time_s=np.asarray(aif_info["aif_time_s"], dtype=np.float64),
            renal_masks=renal_masks,
        )
        if roi_dataset.data_vars:
            final_ds = xr.merge([final_ds, roi_dataset], compat="override")

        qc_messages = _assess_aif_delay_and_curve_stability(
            concentration_map,
            aif_info,
            renal_masks=renal_masks,
            n_precontrast=int(fit_params_pk.get("n_precontrast", 5)),
        )
        if qc_messages:
            final_ds.attrs["qc_messages"] = qc_messages

        final_ds = summarize_renal_parameters(
            final_ds,
            renal_masks=renal_masks,
            gfr_conversion=gfr_conversion,
        )

    final_ds = _annotate_output_dataset(final_ds, fit_type=fit_type)
    return final_ds


def run_dce_pipeline(
    dce_data: xr.DataArray,
    t1_data: xr.DataArray = None,
    aif_signal: np.ndarray = None,
    aif_signal_concentration: np.ndarray = None,
    aif_times: np.ndarray = None,
    time_points_s: np.ndarray = None,
    aif_voxel: dict = None,
    aif_mask: xr.DataArray | np.ndarray = None,
    tr_s: float = None,
    fa_deg: float = None,
    r1: float = None,
    fit_params_t1: dict = None,
    fit_params_tofts: dict = None,
    fit_params_pk: dict = None,
    fit_type: str = "tofts",
    use_multiprocessing: bool = True,
    debug_display: bool = False,
    aif_convert_blood_to_plasma: bool | None = None,
    aif_hematocrit: float | None = None,
    skip_fit_quality_metrics: bool | None = None,
    *,
    fitted_t1: xr.Dataset | xr.DataArray | None = None,
    motion_correct_dce: Callable[[xr.DataArray], xr.DataArray]
    | Mapping[str, object]
    | str
    | None = None,
    renal_masks: dict[str, xr.DataArray | np.ndarray] | None = None,
    gfr_conversion: dict | None = None,
    aif_t1_ms: float | None = None,
    aif_t1_s: float | None = None,
    blood_to_plasma: bool | None = None,
    hematocrit: float | None = None,
) -> xr.Dataset:
    """
    Run the canonical high-level DCE fitting pipeline.

    This is the main public entry point for general DCE analysis in the
    refactored DCE package. The workflow:

    1. prepares metadata and optional motion correction
    2. aligns or loads the T1 map
    3. computes the baseline ``s0_map``
    4. prepares the AIF when the selected model requires one
    5. converts the DCE series from signal space to concentration space
    6. dispatches to a model-specific fitter
    7. computes residual-based QC metrics and attaches audit metadata

    When the AIF is recovered from raw signal data (for example from
    ``aif_signal``, ``aif_voxel``, or ``aif_mask``), the raw AIF is treated as
    blood-pool signal by default and converted to plasma concentration using a
    default hematocrit of ``0.41``. Override this via
    ``aif_convert_blood_to_plasma`` / ``aif_hematocrit`` or the legacy
    ``blood_to_plasma`` / ``hematocrit`` keyword aliases.

    Supported ``fit_type`` values are:

    - ``"patlak"``
    - ``"tofts"``
    - ``"standard_tofts"``
    - ``"extended_tofts"``
    - ``"extended_tofts_plasma"``
    - ``"extended_plasma_tofts"``
    - ``"kidney_two_compartment_outflow"`` and related deprecated aliases,
      which are normalized to extended Tofts
    - ``"sourbron"``
    - ``"empirical_log_logistic"``
    - ``"2c_irf"``

    For the model-specific public wrappers, see:

    - :func:`hypermri.analysis.dce.fitting_support.fit_patlak_model`
    - :func:`hypermri.analysis.dce.fitting_support.fit_tofts_model`
    - :func:`hypermri.analysis.dce.fitting_support.fit_standard_tofts_model`
    - :func:`hypermri.analysis.dce.fitting_support.fit_extended_tofts_model`
    - :func:`hypermri.analysis.dce.fitting_support.fit_sourbron_model`
    - :func:`hypermri.analysis.dce.fitting_support.fit_empirical_log_logistic_model`
    - :func:`hypermri.analysis.dce.fitting_support.fit_two_c_irf_model`

    Parameters
    ----------
    dce_data : xr.DataArray
        DCE signal series in arbitrary scanner units.  Must contain the
        ``repetition_idx`` dimension; the spatial dimensions (for example
        ``read_idx``, ``phase_idx``, ``slice_idx``) are kept in the output.  An
        optional ``channel_idx`` dimension is combined before fitting.  The
        input is not modified.
    t1_data : xr.DataArray, optional
        Variable-flip-angle or inversion-recovery T1 data to fit when
        ``fitted_t1`` is not given.
    aif_signal : np.ndarray, optional
        Raw arterial signal over time, one value per repetition, converted to
        concentration with ``r1``.
    aif_signal_concentration : np.ndarray, optional
        Arterial concentration in mM over time, used without conversion.
    aif_times : np.ndarray, optional
        Legacy name for ``time_points_s``: acquisition time of each repetition
        in s.  Must agree with ``time_points_s`` when both are given.
    time_points_s : np.ndarray, optional
        Acquisition time of each repetition in s.  When omitted it is derived
        from ``dce_data``.
    aif_voxel : dict, optional
        Mapping from spatial dimension name to integer index of the voxel whose
        signal is used as AIF.
    aif_mask : xr.DataArray or np.ndarray, optional
        Boolean mask on the spatial grid of ``dce_data``; the AIF is the mean
        signal inside the mask.
    tr_s : float, optional
        Repetition time in s.  Read from ``dce_data.attrs["repetition_time_ms"]``
        when omitted.
    fa_deg : float, optional
        Excitation flip angle in degrees.  Read from
        ``dce_data.attrs["flipangle_deg"]`` when omitted.
    r1 : float, optional
        Longitudinal relaxivity of the contrast agent in 1/(mM s), used for the
        tissue and AIF signal-to-concentration conversion.  May instead be given
        as ``fit_params_pk["r1"]``; when both are given they must be equal.
        Defaults to 3.5 1/(mM s) when neither is given.  The effective value is
        stored in ``attrs["r1"]`` of the result.
    fit_params_t1 : dict, optional
        Options for the T1 fit of ``t1_data``.
    fit_params_tofts : dict, optional
        Legacy model options, merged before ``fit_params_pk``.
    fit_params_pk : dict, optional
        Model options such as ``n_precontrast``, ``model_mode``,
        ``fit_start_index``, ``signal_to_concentration_mode`` and ``r1``.  The
        mapping is copied and not modified.
    fit_type : str, default "tofts"
        Kinetic model; see the list of supported values above.
    use_multiprocessing : bool, default True
        Fit voxels in parallel processes.
    debug_display : bool, default False
        Accepted for backward compatibility; has no effect.
    aif_convert_blood_to_plasma : bool, optional
        Convert a blood AIF to plasma concentration with ``aif_hematocrit``.
    aif_hematocrit : float, optional
        Hematocrit as a fraction between 0 and 1 for the blood-to-plasma
        conversion.  Defaults to 0.41.
    skip_fit_quality_metrics : bool, optional
        Skip the residual-based QC metrics.  Defaults to the
        ``skip_fit_quality_metrics`` entry of the fit options, else ``False``.
    fitted_t1 : xr.Dataset or xr.DataArray, optional
        Pre-fitted spatial T1 map in ms, or a dataset containing ``t1``. This
        takes precedence over fitting ``t1_data``; existing spatial alignment,
        coordinate direction and unit conversion rules remain in force.
    motion_correct_dce : callable or Mapping or str, optional
        Existing motion-correction specification: a callable accepting and
        returning the labelled signal, a backend name, or a mapping of backend
        settings. None skips correction. The callable must preserve its input.
    renal_masks : dict of str to xr.DataArray or np.ndarray, optional
        Labelled or positional spatial masks for applicable model postprocessing.
        Existing mask selection and geometry rules remain unchanged.
    gfr_conversion : dict, optional
        Existing renal GFR conversion settings for applicable postprocessing,
        with ``parameter_name``, ``scale_factor_ml_min_per_unit``, ``assumptions``
        and ``normalization``. The supplied scale converts the named parameter
        to mL/min. None skips this user-supplied conversion.
    aif_t1_ms : float, optional
        Supplied blood T1 in ms for AIF signal-to-concentration conversion.
    aif_t1_s : float, optional
        Supplied blood T1 in seconds. If both blood-T1 units are supplied, the
        existing AIF validation requires consistent values.
    blood_to_plasma : bool, optional
        Compatibility alias used only when ``aif_convert_blood_to_plasma`` is
        None. An explicit canonical False or True retains precedence.
    hematocrit : float, optional
        Compatibility fraction alias used only when ``aif_hematocrit`` is None;
        a supplied canonical value retains precedence.

    Returns
    -------
    xr.Dataset
        Model parameter maps on the spatial dimensions of ``dce_data``, the
        concentration curves, AIF information and QC metrics.  ``attrs``
        record the effective fit type, model mode, ``tr_s``, ``fa_deg``,
        ``r1`` in 1/(mM s), the signal-to-concentration mode and AIF
        provenance.
        Concentration calculations and continuous DCE maps use float64;
        model-specific count/boolean fields retain their documented precision.
        Spatial dimension order and compatible coordinates follow the selected
        model's existing contract. No handler result cache is introduced.

    Raises
    ------
    ValueError
        If ``r1`` and ``fit_params_pk["r1"]`` disagree, if the relaxivity is
        not positive and finite, if TR or flip angle are missing, or if the
        timing inputs are inconsistent.
    NotImplementedError
        If ``fit_type="auto"`` is requested.
    TypeError
        If an unknown keyword is supplied. Python rejects it before metadata,
        conversion, motion correction or fitting; the message names that keyword.

    Notes
    -----
    Processing is eager and preserves acquired data and option mappings.
    Fixed inputs, scientific settings, deterministic correction and numerical
    backend retain repeatability. Optional motion correction and debug displays
    run only when explicitly selected. Auxiliary options are explicit keyword-only
    arguments; legacy positional arguments and known alias precedence are retained.

    Examples
    --------
    >>> result = run_dce_pipeline(  # doctest: +SKIP
    ...     dce_data=dce.scanner_data,
    ...     aif_mask=dce.masks["AIF"],
    ...     r1=3.5,
    ...     fit_type="patlak",
    ...     fitted_t1=t1_fit,
    ... )
    >>> result.attrs["r1"]  # doctest: +SKIP
    3.5
    """
    _ = debug_display  # Kept for backwards compatibility.
    if aif_convert_blood_to_plasma is None:
        aif_convert_blood_to_plasma = blood_to_plasma
    if aif_hematocrit is None:
        aif_hematocrit = hematocrit

    merged_fit_params = _merge_fit_params(
        fit_params_pk=fit_params_pk,
        fit_params_tofts=fit_params_tofts,
        default_model_mode="both",
    )
    merged_fit_params = _apply_explicit_aif_options(
        merged_fit_params,
        aif_convert_blood_to_plasma=aif_convert_blood_to_plasma,
        aif_hematocrit=aif_hematocrit,
    )
    effective_fit_type, effective_fit_params, default_model_mode = (
        _normalize_legacy_dce_request(
            fit_type,
            merged_fit_params,
        )
    )
    if skip_fit_quality_metrics is None:
        skip_fit_quality_metrics = bool(
            effective_fit_params.get("skip_fit_quality_metrics", False)
        )
    dispatch_metadata = _build_fit_dispatch_metadata(
        requested_fit_type=fit_type,
        requested_fit_params=merged_fit_params,
        effective_fit_type=effective_fit_type,
        effective_fit_params=effective_fit_params,
        default_model_mode=default_model_mode,
    )

    return _run_dce_pipeline_core(
        dce_data,
        t1_data=t1_data,
        fitted_t1=fitted_t1,
        aif_signal=aif_signal,
        aif_signal_concentration=aif_signal_concentration,
        aif_times=aif_times,
        time_points_s=time_points_s,
        aif_voxel=aif_voxel,
        aif_mask=aif_mask,
        tr_s=tr_s,
        fa_deg=fa_deg,
        r1=r1,
        fit_params_t1=fit_params_t1,
        fit_params_pk=effective_fit_params,
        fit_type=effective_fit_type,
        use_multiprocessing=use_multiprocessing,
        motion_corrector=motion_correct_dce,
        renal_masks=renal_masks,
        gfr_conversion=gfr_conversion,
        default_model_mode=default_model_mode,
        aif_t1_ms=aif_t1_ms,
        aif_t1_s=aif_t1_s,
        requested_fit_type=fit_type,
        dispatch_metadata=dispatch_metadata,
        skip_fit_quality_metrics=bool(skip_fit_quality_metrics),
        debug_display=debug_display,
    )


__all__ = [
    "run_dce_pipeline",
]
