"""AIF preparation and validation helpers for DCE workflows."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from hypermri.analysis.dce.conversions import (
    resolve_relaxivity,
    signal_to_concentration,
)
from hypermri.utils.mri.channels import combine_channels

from .shared import (
    _align_mask_weights_to_template,
    _coerce_numeric_1d,
    _mask_weight_sum,
    _weighted_mean_over_mask,
)


def _interpolate_nan_series(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError("Series contains only NaN values.")
    if np.all(valid):
        return values

    indices = np.arange(len(values), dtype=np.float64)
    if int(valid.sum()) == 1:
        fill_value = float(values[valid][0])
        return np.full_like(values, fill_value, dtype=np.float64)

    interpolator = interp1d(
        indices[valid],
        values[valid],
        kind="linear",
        fill_value="extrapolate",
    )
    return interpolator(indices).astype(np.float64)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _detect_bolus_arrival_index(
    curve: np.ndarray, *, n_precontrast: int = 5
) -> int | None:
    finite_curve = np.asarray(curve, dtype=np.float64)
    if finite_curve.size == 0 or not np.any(np.isfinite(finite_curve)):
        return None

    n_base = max(1, min(int(n_precontrast), len(finite_curve)))
    baseline = finite_curve[:n_base]
    baseline = baseline[np.isfinite(baseline)]
    if baseline.size == 0:
        return None

    peak = float(np.nanmax(finite_curve))
    if not np.isfinite(peak):
        return None

    baseline_mean = float(np.nanmean(baseline))
    baseline_std = float(np.nanstd(baseline))
    dynamic_threshold = baseline_mean + max(
        3.0 * baseline_std, 0.05 * max(peak - baseline_mean, 0.0)
    )
    above = np.where(finite_curve > dynamic_threshold)[0]
    if above.size == 0:
        return None
    return int(above[0])


def validate_aif(
    *,
    aif_time_s,
    aif_signal_raw=None,
    aif_concentration=None,
    n_precontrast: int = 5,
    interpolate_nan: bool = True,
    smoothing_window: int = 1,
    clip_negative_concentration: bool = False,
    spike_z_threshold: float = 8.0,
) -> dict:
    """Validate and optionally clean AIF inputs."""
    time_points_s = _coerce_numeric_1d("aif_time_s", aif_time_s)
    if len(time_points_s) < 3:
        raise ValueError("AIF time axis must contain at least three samples.")
    if not np.all(np.isfinite(time_points_s)):
        raise ValueError("AIF time axis contains non-finite values.")
    if np.any(np.diff(time_points_s) <= 0):
        raise ValueError("AIF time axis must be strictly increasing.")

    qc_messages: list[str] = []
    outputs: dict[str, np.ndarray | None] = {
        "aif_signal_raw": None,
        "aif_concentration": None,
    }

    for name, series in (
        ("aif_signal_raw", aif_signal_raw),
        ("aif_concentration", aif_concentration),
    ):
        if series is None:
            continue

        arr = _coerce_numeric_1d(name, series)
        if len(arr) != len(time_points_s):
            raise ValueError(
                f"{name} length ({len(arr)}) does not match aif_time_s length ({len(time_points_s)})."
            )

        if np.all(np.isnan(arr)):
            raise ValueError(f"{name} contains only NaN values.")

        if np.any(np.isnan(arr)):
            if not interpolate_nan:
                raise ValueError(f"{name} contains NaN values.")
            arr = _interpolate_nan_series(arr)
            qc_messages.append(f"{name} contained NaNs and was linearly interpolated.")

        if smoothing_window > 1:
            arr = _smooth_series(arr, int(smoothing_window))
            qc_messages.append(
                f"{name} was smoothed with a moving-average window of {int(smoothing_window)}."
            )

        outputs[name] = arr.astype(np.float64, copy=False)

    concentration = outputs["aif_concentration"]
    spike_indices = np.array([], dtype=np.int64)
    if concentration is not None:
        negative_idx = np.where(concentration < 0)[0]
        if negative_idx.size:
            qc_messages.append(
                f"AIF concentration contains {negative_idx.size} negative samples."
            )
            if clip_negative_concentration:
                concentration = concentration.copy()
                concentration[negative_idx] = 0.0
                qc_messages.append(
                    "Negative AIF concentration samples were clipped to zero."
                )
                outputs["aif_concentration"] = concentration

        diffs = np.diff(concentration)
        if diffs.size >= 3:
            median_diff = float(np.nanmedian(diffs))
            mad = float(np.nanmedian(np.abs(diffs - median_diff)))
            if mad > 0:
                robust_sigma = 1.4826 * mad
                spike_indices = (
                    np.where(
                        np.abs(diffs - median_diff) > spike_z_threshold * robust_sigma
                    )[0]
                    + 1
                )
                if spike_indices.size:
                    qc_messages.append(
                        "AIF concentration contains abrupt spikes at indices "
                        f"{spike_indices.tolist()}."
                    )

    bolus_index = None
    if concentration is not None:
        bolus_index = _detect_bolus_arrival_index(
            concentration,
            n_precontrast=n_precontrast,
        )

    bolus_time_s = (
        float(time_points_s[bolus_index]) if bolus_index is not None else np.nan
    )

    return {
        "aif_time_s": time_points_s,
        "aif_signal_raw": outputs["aif_signal_raw"],
        "aif_concentration": outputs["aif_concentration"],
        "qc_messages": qc_messages,
        "bolus_arrival_index": bolus_index,
        "bolus_arrival_time_s": bolus_time_s,
        "spike_indices": spike_indices,
    }


def _align_aif_mask_to_template(
    mask,
    *,
    template: xr.DataArray,
) -> xr.DataArray | None:
    if mask is None:
        return None

    mask_da = _align_mask_weights_to_template(mask, template, name="aif_mask")
    if _mask_weight_sum(mask_da) <= 0:
        raise ValueError("AIF mask contains no selected voxels.")
    return mask_da


def _prepare_aif_series_input(
    series,
    *,
    weights: xr.DataArray | None = None,
    aif_voxel: Mapping[str, int] | None = None,
) -> np.ndarray | None:
    if series is None:
        return None

    if isinstance(series, xr.DataArray):
        series_da = series
        if "channel_idx" in series_da.dims:
            if weights is not None and aif_voxel is not None:
                weight_sel = {
                    dim: idx for dim, idx in aif_voxel.items() if dim in weights.dims
                }
                try:
                    voxel_weights = weights.isel(**weight_sel)
                    series_da = combine_channels(
                        data=series_da,
                        method="weighted_mean",
                        weights=voxel_weights,
                    )
                except Exception:
                    series_da = combine_channels(series_da, method="rss")
            else:
                series_da = combine_channels(series_da, method="rss")
        return _coerce_numeric_1d("AIF series", series_da)

    return _coerce_numeric_1d("AIF series", series)


def _compute_masked_t1_summary(
    t1_map: xr.DataArray,
    mask_da: xr.DataArray,
    *,
    t1_weight_map: xr.DataArray | None = None,
) -> dict[str, float | int | str]:
    mask_weights = np.asarray(mask_da.values, dtype=np.float64)
    mask_weights = np.clip(
        np.nan_to_num(mask_weights, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    t1_values = np.asarray(t1_map.values, dtype=np.float64)
    valid_t1 = (mask_weights > 0) & np.isfinite(t1_values)
    masked_voxel_count = float(mask_weights.sum())
    valid_t1_voxel_count = float(mask_weights[valid_t1].sum())

    if valid_t1_voxel_count <= 0:
        return {
            "aif_t1_ms_used": np.nan,
            "aif_t1_weighting": "mean",
            "aif_mask_voxel_count": masked_voxel_count,
            "aif_valid_t1_voxel_count": valid_t1_voxel_count,
            "aif_t1_weight_sum": np.nan,
            "aif_t1_weighted_voxel_count": 0,
        }

    if t1_weight_map is not None:
        rss_values = np.asarray(t1_weight_map.values, dtype=np.float64)
        valid_rss = valid_t1 & np.isfinite(rss_values) & (rss_values >= 0)
        if np.any(valid_rss):
            safe_rss = np.maximum(rss_values[valid_rss], np.finfo(np.float64).tiny)
            quality_weights = mask_weights[valid_rss] / safe_rss
            weight_sum = float(np.sum(quality_weights))
            if np.isfinite(weight_sum) and weight_sum > 0:
                weighted_t1_ms = float(
                    np.sum(t1_values[valid_rss] * quality_weights) / weight_sum
                )
                return {
                    "aif_t1_ms_used": weighted_t1_ms,
                    "aif_t1_weighting": "inverse_rss_t1",
                    "aif_mask_voxel_count": masked_voxel_count,
                    "aif_valid_t1_voxel_count": valid_t1_voxel_count,
                    "aif_t1_weight_sum": weight_sum,
                    "aif_t1_weighted_voxel_count": float(mask_weights[valid_rss].sum()),
                }

    return {
        "aif_t1_ms_used": float(
            np.sum(t1_values[valid_t1] * mask_weights[valid_t1]) / valid_t1_voxel_count
        ),
        "aif_t1_weighting": "mean",
        "aif_mask_voxel_count": masked_voxel_count,
        "aif_valid_t1_voxel_count": valid_t1_voxel_count,
        "aif_t1_weight_sum": np.nan,
        "aif_t1_weighted_voxel_count": 0,
    }


def prepare_aif(
    dce_combined: xr.DataArray,
    t1_map: xr.DataArray,
    *,
    aif_signal: np.ndarray | xr.DataArray | None = None,
    aif_signal_concentration: np.ndarray | xr.DataArray | None = None,
    aif_times: np.ndarray | xr.DataArray,
    aif_voxel: Mapping[str, int] | tuple | None = None,
    aif_mask: xr.DataArray | np.ndarray | None = None,
    n_precontrast: int = 5,
    tr_s: float,
    fa_deg: float,
    r1: float | None = None,
    weights: xr.DataArray | None = None,
    aif_t1_weight_map: xr.DataArray | None = None,
    fit_params: dict | None = None,
    aif_t1_ms: float | None = None,
    aif_t1_s: float | None = None,
) -> dict:
    """Extract or validate the AIF and convert raw AIF signal to concentration.

    The AIF is taken, in this order of precedence, from
    ``aif_signal_concentration``, ``aif_signal``, the voxel ``aif_voxel`` of
    ``dce_combined`` or the weighted mean of ``dce_combined`` over
    ``aif_mask``.  A raw signal is converted to concentration with
    :func:`~hypermri.analysis.dce.conversions.signal_to_concentration`
    using the baseline of the first ``n_precontrast`` frames, the AIF T1 and
    the relaxivity ``r1``.  The series is then checked and cleaned with
    :func:`validate_aif`.  The inputs are not modified.

    Parameters
    ----------
    dce_combined : xarray.DataArray
        Channel-combined dynamic signal in arbitrary units with the spatial
        dimensions of ``t1_map`` and the dynamic dimension ``repetition_idx``.
    t1_map : xarray.DataArray
        Pre-contrast T1 map in ms on the spatial grid of ``dce_combined``.
        Read only when the AIF T1 comes from ``aif_voxel`` or ``aif_mask``.
    aif_signal : numpy.ndarray or xarray.DataArray, optional
        Raw AIF signal in arbitrary units, one value per time point.  A
        ``channel_idx`` dimension is combined first.
    aif_signal_concentration : numpy.ndarray or xarray.DataArray, optional
        AIF concentration in mM, one value per time point.  Takes precedence
        over every raw-signal source and is not converted.
    aif_times : numpy.ndarray or xarray.DataArray
        Strictly increasing AIF time points in s; at least three samples.
    aif_voxel : Mapping[str, int] or tuple, optional
        Indices of the AIF voxel by dimension name.  A one-element tuple
        holding the mapping is accepted.
    aif_mask : xarray.DataArray or numpy.ndarray, optional
        Boolean or fractional AIF mask on the spatial grid of
        ``dce_combined``; fractional values in [0, 1] weight the voxels.
    n_precontrast : int, default 5
        Number of leading frames used for the baseline signal and for the
        bolus-arrival detection.
    tr_s : float
        Repetition time in s.
    fa_deg : float
        Flip angle in degrees.
    r1 : float, optional
        Longitudinal relaxivity of the contrast agent in 1/(mM s).  May be
        given here or as ``fit_params["r1"]``; when both are given they must
        be equal.  Defaults to
        :data:`~hypermri.analysis.dce.conversions.DEFAULT_R1_PER_MM_S`.
    weights : xarray.DataArray, optional
        Channel weights used to combine a multi-channel ``aif_signal`` or
        ``aif_signal_concentration`` at ``aif_voxel``.
    aif_t1_weight_map : xarray.DataArray, optional
        Residual sum of squares of the T1 fit on the grid of ``t1_map``.  With
        ``aif_mask``, the mask T1 is weighted by the inverse of this map.
    fit_params : dict, optional
        Conversion and cleaning options.  Read keys are ``"r1"``,
        ``"signal_to_concentration_mode"``, ``"signal_to_concentration_scale"``,
        ``"aif_convert_blood_to_plasma"`` (default True for raw signal),
        ``"aif_hematocrit"`` (default 0.41), ``"saturation_threshold"``,
        ``"saturation_behavior"``, ``"warn_on_saturation"``,
        ``"return_saturation_info"``, ``"interpolate_aif_nan"``,
        ``"aif_smoothing_window"``, ``"clip_negative_aif_concentration"`` and
        ``"aif_spike_z_threshold"``.  The mapping is not modified.
    aif_t1_ms : float, optional
        Explicit AIF T1 in ms; overrides the T1 from ``aif_voxel`` or
        ``aif_mask``.
    aif_t1_s : float, optional
        Explicit AIF T1 in s; takes precedence over ``aif_t1_ms``.

    Returns
    -------
    dict
        The result of :func:`validate_aif` (time points in s, raw signal,
        concentration in mM as float64 arrays of the AIF length, QC messages,
        bolus arrival and spike indices) extended with ``"aif_source"``,
        ``"aif_s0_used"`` (baseline signal, NaN for concentration input),
        ``"aif_mask"`` and the AIF T1 summary keys ``"aif_t1_ms_used"``,
        ``"aif_t1_weighting"``, ``"aif_mask_voxel_count"``,
        ``"aif_valid_t1_voxel_count"``, ``"aif_t1_weight_sum"``,
        ``"aif_t1_weighted_voxel_count"``, ``"aif_convert_blood_to_plasma"``,
        ``"aif_hematocrit"``, ``"aif_concentration_kind"``,
        ``"aif_signal_to_concentration_mode"`` and
        ``"aif_signal_to_concentration_scale"``.

    Raises
    ------
    ValueError
        If no AIF source is given, if ``r1`` and ``fit_params["r1"]`` differ
        or the relaxivity is not positive and finite, if a raw signal has no
        T1 source, if the AIF voxel or mask lies outside the valid T1 region,
        if the baseline signal is not positive, or if :func:`validate_aif`
        rejects the series.
    """
    fit_params = dict(fit_params or {})
    r1 = resolve_relaxivity(r1, fit_params, operation="prepare_aif")
    signal_to_concentration_mode = (
        str(fit_params.get("signal_to_concentration_mode", "spgr")).strip().lower()
    )
    raw_signal_scale = fit_params.get("signal_to_concentration_scale", 1.0)
    try:
        signal_to_concentration_scale: float | str = float(raw_signal_scale)
    except (TypeError, ValueError):
        signal_to_concentration_scale = str(raw_signal_scale).strip().lower()
    requested_convert_blood_to_plasma = fit_params.get(
        "aif_convert_blood_to_plasma", None
    )
    requested_hematocrit = fit_params.get("aif_hematocrit", None)
    t1_summary: dict[str, float | int | str] = {
        "aif_t1_ms_used": np.nan,
        "aif_t1_weighting": "none",
        "aif_mask_voxel_count": 0,
        "aif_valid_t1_voxel_count": 0,
        "aif_t1_weight_sum": np.nan,
        "aif_t1_weighted_voxel_count": 0,
        "aif_convert_blood_to_plasma": False,
        "aif_hematocrit": np.nan,
        "aif_concentration_kind": "unknown",
        "aif_signal_to_concentration_mode": signal_to_concentration_mode,
        "aif_signal_to_concentration_scale": signal_to_concentration_scale,
    }
    aif_source = "unknown"

    if (
        isinstance(aif_voxel, tuple)
        and len(aif_voxel) == 1
        and isinstance(aif_voxel[0], Mapping)
    ):
        aif_voxel = aif_voxel[0]

    mask_template = dce_combined.isel(repetition_idx=0, drop=True)
    aif_mask_da = _align_aif_mask_to_template(aif_mask, template=mask_template)

    raw_signal = _prepare_aif_series_input(
        aif_signal,
        weights=weights,
        aif_voxel=aif_voxel if isinstance(aif_voxel, Mapping) else None,
    )
    concentration = _prepare_aif_series_input(
        aif_signal_concentration,
        weights=weights,
        aif_voxel=aif_voxel if isinstance(aif_voxel, Mapping) else None,
    )

    if (
        raw_signal is None
        and concentration is None
        and aif_voxel is None
        and aif_mask_da is None
    ):
        raise ValueError(
            "Provide aif_signal, aif_signal_concentration, aif_voxel, or aif_mask to define the AIF."
        )

    if raw_signal is None and concentration is None and aif_voxel is not None:
        voxel_signal = dce_combined.isel(**dict(aif_voxel)).squeeze()
        raw_signal = _coerce_numeric_1d("aif_signal_raw", voxel_signal)
        aif_source = "voxel"
    elif raw_signal is None and concentration is None and aif_mask_da is not None:
        mask_dims = tuple(aif_mask_da.dims)
        masked_signal = _weighted_mean_over_mask(
            dce_combined,
            aif_mask_da,
            dims=mask_dims,
        )
        raw_signal = _coerce_numeric_1d("aif_signal_raw", masked_signal)
        aif_source = "mask"
    elif concentration is not None:
        aif_source = "concentration_input"
    elif raw_signal is not None:
        aif_source = "signal_input"

    if concentration is None and raw_signal is not None:
        convert_blood_to_plasma = (
            True
            if requested_convert_blood_to_plasma is None
            else bool(requested_convert_blood_to_plasma)
        )
        hematocrit = (
            0.41
            if requested_hematocrit is None and convert_blood_to_plasma
            else requested_hematocrit
        )
        t1_summary["aif_convert_blood_to_plasma"] = convert_blood_to_plasma
        t1_summary["aif_hematocrit"] = (
            float(hematocrit) if hematocrit is not None else np.nan
        )
        t1_summary["aif_concentration_kind"] = (
            "plasma" if convert_blood_to_plasma else "blood"
        )

        if aif_t1_s is not None:
            t10_s = float(aif_t1_s)
            t1_summary["aif_t1_ms_used"] = t10_s * 1000.0
            t1_summary["aif_t1_weighting"] = "explicit_seconds"
        elif aif_t1_ms is not None:
            t10_s = float(aif_t1_ms) / 1000.0
            t1_summary["aif_t1_ms_used"] = float(aif_t1_ms)
            t1_summary["aif_t1_weighting"] = "explicit_milliseconds"
        elif aif_voxel is not None:
            try:
                t1_value_ms = float(t1_map.isel(**dict(aif_voxel)).values.item())
            except Exception as exc:
                raise ValueError(
                    f"AIF voxel {aif_voxel} is out of bounds for the aligned T1 map."
                ) from exc
            if not np.isfinite(t1_value_ms):
                raise ValueError(
                    "The selected AIF voxel lies outside the valid T1 overlap region."
                )
            t10_s = t1_value_ms / 1000.0
            t1_summary["aif_t1_ms_used"] = t1_value_ms
            t1_summary["aif_t1_weighting"] = "voxel"
        elif aif_mask_da is not None:
            t1_summary.update(
                _compute_masked_t1_summary(
                    t1_map,
                    aif_mask_da,
                    t1_weight_map=aif_t1_weight_map,
                )
            )
            t1_value_ms = float(t1_summary["aif_t1_ms_used"])
            if not np.isfinite(t1_value_ms):
                raise ValueError(
                    "The selected AIF mask lies outside the valid T1 overlap region."
                )
            t10_s = t1_value_ms / 1000.0
        else:
            raise ValueError(
                "Raw AIF signal was supplied without aif_voxel, aif_mask, or explicit aif_t1_ms/aif_t1_s."
            )

        s0_vox = float(np.nanmean(raw_signal[: max(1, int(n_precontrast))]))
        if not np.isfinite(s0_vox) or s0_vox <= 0:
            raise ValueError(
                "AIF baseline S0 could not be estimated from the raw AIF signal."
            )
        concentration = signal_to_concentration(
            S_t=raw_signal,
            S_0=s0_vox,
            T10_s=t10_s,
            TR_s=tr_s,
            flip_angle_deg=fa_deg,
            r1=r1,
            mode=signal_to_concentration_mode,
            linear_scale=signal_to_concentration_scale,
            convert_blood_to_plasma=convert_blood_to_plasma,
            hematocrit=hematocrit,
            saturation_threshold=fit_params.get("saturation_threshold", 0.98),
            saturation_behavior=fit_params.get("saturation_behavior", "nan_all"),
            warn_on_saturation=fit_params.get("warn_on_saturation", False),
            return_saturation_info=fit_params.get("return_saturation_info", False),
        )
    else:
        s0_vox = np.nan

    if concentration is not None and raw_signal is None:
        t1_summary["aif_concentration_kind"] = str(
            fit_params.get("aif_concentration_kind", "unknown")
        )
        t1_summary["aif_signal_to_concentration_mode"] = str(
            fit_params.get("aif_signal_to_concentration_mode", "input_concentration")
        )
        raw_aif_scale = fit_params.get("aif_signal_to_concentration_scale", np.nan)
        try:
            t1_summary["aif_signal_to_concentration_scale"] = float(raw_aif_scale)
        except (TypeError, ValueError):
            t1_summary["aif_signal_to_concentration_scale"] = str(raw_aif_scale).strip()

    aif_info = validate_aif(
        aif_time_s=aif_times,
        aif_signal_raw=raw_signal,
        aif_concentration=concentration,
        n_precontrast=n_precontrast,
        interpolate_nan=bool(fit_params.get("interpolate_aif_nan", True)),
        smoothing_window=int(fit_params.get("aif_smoothing_window", 1)),
        clip_negative_concentration=bool(
            fit_params.get("clip_negative_aif_concentration", False)
        ),
        spike_z_threshold=float(fit_params.get("aif_spike_z_threshold", 8.0)),
    )

    if aif_info["aif_signal_raw"] is None:
        aif_info["aif_signal_raw"] = np.full_like(
            aif_info["aif_concentration"],
            np.nan,
            dtype=np.float64,
        )

    aif_info["aif_source"] = aif_source
    aif_info["aif_s0_used"] = float(s0_vox)
    aif_info["aif_mask"] = aif_mask_da
    aif_info.update(t1_summary)

    return aif_info
