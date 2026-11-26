"""Fit labelled DCE concentration data and describe derived model outputs.

Times are seconds, concentrations mM and normalized flows 1/s. Models retain
their scientific defaults and precision contracts; derived dataset outputs do
not add optimizer parameters. Calls preserve acquired inputs and retain no cache.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid as trapezoid
from scipy.optimize import least_squares as least_squares

from hypermri.analysis.dce.models.patlak import (
    _build_patlak_fit_mask,
    patlak_model,
)
from hypermri.analysis.dce.models.sourbron import (
    _build_sourbron_fit_mask,
    _default_sourbron_bounds,
    sourbron_model_reconstruction,
)
from hypermri.analysis.dce.models.tofts import (
    TOFTS_1COMP_BOUNDS,
    TOFTS_2COMP_BOUNDS,
    build_tofts_fit_mask,
    tofts_model_reconstruction,
)
from hypermri.analysis.fitting import api as _fitting_api

from .execution import _fit_dce_maps
from .fit_quality import _parameters_at_bounds as _parameters_at_bounds
from .models.empirical_log_logistic import (
    _reconstruct_empirical_log_logistic_prediction as _reconstruct_empirical_log_logistic_prediction,
)
from .models.tofts import _TOFTS_1COMP_NAMES as _TOFTS_1COMP_NAMES
from .models.tofts import _TOFTS_2COMP_NAMES as _TOFTS_2COMP_NAMES
from .models.two_c_irf import (
    _reconstruct_two_c_irf_prediction as _reconstruct_two_c_irf_prediction,
)
from .sampling import (
    _derive_time_points_from_concentration_map as _derive_time_points_from_concentration_map,
)

_api_fit = _fitting_api.fit
from hypermri.analysis.dce.conversions import (
    DEFAULT_R1_PER_MM_S,
    signal_to_concentration,
)
from hypermri.analysis.fitting.models.quantitative.convolution import (
    trapezoidal_convolution as trapezoidal_convolution,
)
from hypermri.analysis.fitting.voxelwise import VoxelwiseExecutor

from .fit_metrics import _residual_statistics
from .fit_window import select_fit_window as select_fit_window
from .models.empirical_log_logistic import (
    _EMPIRICAL_HALF_RISE_TIME_ATTRS,
)
from .models.empirical_log_logistic import (
    _EMPIRICAL_LOG_LOGISTIC_MODEL_EQUATION as _EMPIRICAL_LOG_LOGISTIC_MODEL_EQUATION,
)
from .models.empirical_log_logistic import (
    _EMPIRICAL_LOG_LOGISTIC_OUTPUT_NAMES as _EMPIRICAL_LOG_LOGISTIC_OUTPUT_NAMES,
)
from .models.empirical_log_logistic import (
    _default_empirical_log_logistic_bounds as _default_empirical_log_logistic_bounds,
)
from .models.empirical_log_logistic import (
    _fit_empirical_log_logistic_voxel as _fit_empirical_log_logistic_voxel,
)
from .models.empirical_log_logistic import (
    _reconstruct_empirical_log_logistic_curve_scalar as _reconstruct_empirical_log_logistic_curve_scalar,
)
from .models.empirical_log_logistic import (
    empirical_log_logistic_model as empirical_log_logistic_model,
)
from .models.empirical_log_logistic import (
    empirical_log_logistic_reconstruction as empirical_log_logistic_reconstruction,
)
from .models.empirical_log_logistic import (
    fit_empirical_log_logistic_curve as fit_empirical_log_logistic_curve,
)
from .models.extended_compartment import (
    _EQUIVALENT_VOLUME_ATTRS as _EQUIVALENT_VOLUME_ATTRS,
)
from .models.extended_compartment import (
    _EXTENDED_TOFTS_SOURBRON_EQUIVALENCE_NOTE as _EXTENDED_TOFTS_SOURBRON_EQUIVALENCE_NOTE,
)
from .models.extended_compartment import (
    _EXTENDED_TOFTS_SOURBRON_EQUIVALENT_FIT_TYPES as _EXTENDED_TOFTS_SOURBRON_EQUIVALENT_FIT_TYPES,
)
from .models.extended_compartment import (
    _add_equivalent_volume_outputs as _add_equivalent_volume_outputs,
)
from .models.extended_compartment import (
    _add_extended_compartment_aliases,
)
from .models.two_c_irf import (
    _PER_S_TO_ML_MIN_PER_100ML as _PER_S_TO_ML_MIN_PER_100ML,
)
from .models.two_c_irf import (
    _TWO_C_IRF_OUTPUT_NAMES,
    _fit_two_c_irf_voxel,
)
from .models.two_c_irf import (
    _default_two_c_irf_bounds as _default_two_c_irf_bounds,
)
from .models.two_c_irf import (
    _fit_two_c_irf_curve_tuple as _fit_two_c_irf_curve_tuple,
)
from .models.two_c_irf import (
    _predict_two_c_irf_curve as _predict_two_c_irf_curve,
)
from .models.two_c_irf import (
    _prepare_two_c_irf_prediction as _prepare_two_c_irf_prediction,
)
from .models.two_c_irf import (
    _reconstruct_two_c_irf_curve_scalar as _reconstruct_two_c_irf_curve_scalar,
)
from .models.two_c_irf import (
    fit_two_c_irf_curve as fit_two_c_irf_curve,
)
from .models.two_c_irf import (
    two_c_irf_model as two_c_irf_model,
)
from .models.two_c_irf import (
    two_c_irf_reconstruction as two_c_irf_reconstruction,
)
from .models.two_c_irf import (
    two_c_irf_retention_function as two_c_irf_retention_function,
)
from .preprocessing import align_t1_to_dce_grid
from .shared import (
    _coerce_numeric_1d,
    _dataarray_isclose,
    _set_var_attrs,
    _spatial_template,
)
from .shared import (
    _validate_dce_time_axis as _validate_dce_time_axis,
)


def _slice_axis_mm_matches_template(
    data: xr.DataArray,
    template: xr.DataArray,
) -> bool:
    if "slice_idx" not in data.dims or "slice_idx" not in template.dims:
        return True

    data_axis = data.coords["slice_idx"].attrs.get("axis_mm")
    template_axis = template.coords["slice_idx"].attrs.get("axis_mm")
    if data_axis is None or template_axis is None:
        return True

    data_axis = np.asarray(data_axis, dtype=np.float64).squeeze()
    template_axis = np.asarray(template_axis, dtype=np.float64).squeeze()
    return data_axis.shape == template_axis.shape and np.allclose(
        data_axis,
        template_axis,
        atol=1e-6,
        rtol=1e-6,
        equal_nan=False,
    )


def _coerce_spatial_map_to_template(
    data: xr.DataArray,
    template: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    spatial = data.astype(np.float64, copy=False)

    for dim in list(spatial.dims):
        if dim in template.dims:
            continue
        if spatial.sizes[dim] != 1:
            raise ValueError(
                f"{name} has unsupported non-spatial dimension '{dim}' with size "
                f"{spatial.sizes[dim]}."
            )
        spatial = spatial.isel({dim: 0}, drop=True)

    try:
        aligned = spatial.broadcast_like(template).transpose(*template.dims)
    except Exception:
        aligned = None

    if aligned is not None and aligned.shape == template.shape:
        coords_match = all(
            np.array_equal(aligned.coords[dim].values, template.coords[dim].values)
            for dim in template.dims
        )
        if coords_match:
            aligned.name = data.name
            aligned.attrs = dict(data.attrs)
            return aligned

    common_dims = [dim for dim in template.dims if dim in spatial.dims]
    spatial = spatial.transpose(*common_dims)

    for dim in common_dims:
        if spatial.sizes[dim] not in {1, template.sizes[dim]}:
            raise ValueError(
                f"{name} shape {spatial.shape} is not compatible with the DCE spatial "
                f"grid {template.shape}."
            )

    target_shape = tuple(
        spatial.sizes[dim] if dim in spatial.dims else 1 for dim in template.dims
    )
    reshaped = np.asarray(spatial.values, dtype=np.float64).reshape(target_shape)
    values = np.broadcast_to(reshaped, template.shape).copy()
    return xr.DataArray(
        values,
        dims=template.dims,
        coords=template.coords,
        name=data.name,
        attrs=dict(data.attrs),
    )


def _spatial_map_matches_template(
    data: xr.DataArray,
    template: xr.DataArray,
) -> bool:
    try:
        _coerce_spatial_map_to_template(data, template, name="candidate")
    except ValueError:
        return False
    return _slice_axis_mm_matches_template(data, template)


def _copy_coord_attrs(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    for coord_name in target.coords:
        if coord_name in source.coords:
            target.coords[coord_name].attrs = dict(source.coords[coord_name].attrs)
    return target


def convert_signal_to_concentration_map(
    dce_combined: xr.DataArray,
    s0_map: xr.DataArray,
    t1_map_ms: xr.DataArray,
    *,
    tr_s: float,
    fa_deg: float,
    r1: float = DEFAULT_R1_PER_MM_S,
    mode: str = "spgr",
    linear_scale: float | str = 1.0,
    saturation_threshold: float = 0.98,
    saturation_behavior: str = "clip",
    warn_on_saturation: bool = False,
    time_points_s: np.ndarray | None = None,
    contrast_agent_metadata: Mapping[str, object] | None = None,
) -> xr.DataArray:
    """Convert a dynamic DCE signal series into a concentration map.

    The conversion is done voxelwise with
    :func:`~hypermri.analysis.dce.conversions.signal_to_concentration`.
    A T1 map on another grid is first resampled onto the DCE grid with
    :func:`~hypermri.analysis.dce.preprocessing.align_t1_to_dce_grid`.  The
    inputs are not modified.

    Parameters
    ----------
    dce_combined : xarray.DataArray
        Channel-combined signal in arbitrary units with a ``repetition_idx``
        dimension; a singleton ``channel_idx`` is allowed.
    s0_map : xarray.DataArray
        Baseline signal in arbitrary units on the spatial grid of
        ``dce_combined``.
    t1_map_ms : xarray.DataArray
        Pre-contrast T1 in ms.
    tr_s : float
        Repetition time in s.
    fa_deg : float
        Flip angle in degrees.
    r1 : float, default DEFAULT_R1_PER_MM_S
        Contrast-agent relaxivity in 1/(mM s), see
        :data:`~hypermri.analysis.dce.conversions.DEFAULT_R1_PER_MM_S`.
    mode : {"spgr", "signal_enhancement", "relative_signal_enhancement"}, default "spgr"
        Conversion model, see ``signal_to_concentration``.
    linear_scale : float or {"eq8", "schabel_eq8"}, default 1.0
        Scale of the linear conversion modes; ignored for ``"spgr"``.
    saturation_threshold : float, default 0.98
        Relative SPGR saturation limit in (0, 1).
    saturation_behavior : {"clip", "nan", "nan_all"}, default "clip"
        Handling of saturated SPGR samples.
    warn_on_saturation : bool, default False
        Emit a ``RuntimeWarning`` when saturation is detected.
    time_points_s : numpy.ndarray, optional
        Sample times in s, stored as the ``time_s`` coordinate along
        ``repetition_idx``.
    contrast_agent_metadata : Mapping[str, object], optional
        Stored as attributes ``"contrast_agent_<key>"``.

    Returns
    -------
    xarray.DataArray
        float64 concentration in mM named ``"concentration_map"`` with the
        dimensions and coordinates of ``dce_combined``.  Its attributes
        record the conversion model and scale, the saturation summary,
        ``"tr_s"``, ``"fa_deg"`` and ``"r1"``.

    Raises
    ------
    ValueError
        If ``dce_combined`` has no ``repetition_idx`` dimension, a map cannot
        be matched to the DCE grid or a conversion option is invalid.
    """
    template_spatial = _spatial_template(dce_combined)
    s0_map = _coerce_spatial_map_to_template(
        s0_map,
        template_spatial,
        name="s0_map",
    )

    if _spatial_map_matches_template(t1_map_ms, template_spatial):
        t1_map_ms = _coerce_spatial_map_to_template(
            t1_map_ms,
            template_spatial,
            name="t1_map",
        )
    else:
        t1_map_ms, _ = align_t1_to_dce_grid(
            dce_combined,
            fitted_t1=t1_map_ms,
            use_multiprocessing=False,
        )
        t1_map_ms = _coerce_spatial_map_to_template(
            t1_map_ms,
            template_spatial,
            name="t1_map",
        )

    concentration_map, saturation_info = signal_to_concentration(
        S_t=dce_combined,
        S_0=s0_map.broadcast_like(dce_combined),
        T10_s=t1_map_ms.broadcast_like(dce_combined) / 1000.0,
        TR_s=tr_s,
        flip_angle_deg=fa_deg,
        r1=r1,
        mode=mode,
        linear_scale=linear_scale,
        saturation_threshold=saturation_threshold,
        saturation_behavior=saturation_behavior,
        warn_on_saturation=warn_on_saturation,
        return_saturation_info=True,
    )

    if not isinstance(concentration_map, xr.DataArray):
        concentration_map = xr.DataArray(
            concentration_map,
            dims=dce_combined.dims,
            coords=dce_combined.coords,
            name="concentration_map",
        )

    concentration_map = concentration_map.astype(np.float64, copy=False)
    concentration_map.name = "concentration_map"
    concentration_map = _copy_coord_attrs(dce_combined, concentration_map)

    if time_points_s is not None and "repetition_idx" in concentration_map.dims:
        concentration_map = concentration_map.assign_coords(
            time_s=("repetition_idx", np.asarray(time_points_s, dtype=np.float64))
        )
        concentration_map = _copy_coord_attrs(dce_combined, concentration_map)

    try:
        serialized_linear_scale: float | str = float(linear_scale)
    except (TypeError, ValueError):
        serialized_linear_scale = str(linear_scale).strip().lower()

    concentration_map.attrs.update(
        {
            "conversion_model": (
                "spgr_signal_to_concentration"
                if str(mode).strip().lower() == "spgr"
                else str(mode).strip().lower()
            ),
            "signal_to_concentration_mode": str(mode).strip().lower(),
            "signal_to_concentration_scale": serialized_linear_scale,
            "signal_saturation_detected": bool(
                saturation_info.get("saturation_detected", False)
            ),
            "signal_saturation_count": int(saturation_info.get("saturation_count", 0)),
            "signal_saturation_fraction": float(
                saturation_info.get("saturation_fraction", 0.0)
            ),
            "signal_saturation_threshold": float(
                saturation_info.get("saturation_threshold", saturation_threshold)
            ),
            "signal_saturation_behavior": str(
                saturation_info.get("saturation_behavior", saturation_behavior)
            ),
            "tr_s": float(tr_s),
            "fa_deg": float(fa_deg),
            "r1": float(r1),
        }
    )
    if contrast_agent_metadata:
        for key, value in dict(contrast_agent_metadata).items():
            concentration_map.attrs[f"contrast_agent_{key}"] = value

    return concentration_map


def _add_clear_model_aliases(dataset: xr.Dataset) -> xr.Dataset:
    """Attach consistently named aliases for the supported DCE models."""
    dataset = dataset.copy()
    dataset = _add_equivalent_volume_outputs(dataset)
    has_tp_parameter = "Tp" in dataset.data_vars

    if "Ktrans_1c" in dataset and "kep_1c" in dataset:
        dataset["Ktrans_standard"] = dataset["Ktrans_1c"].copy(deep=True)
        dataset["kep_standard"] = dataset["kep_1c"].copy(deep=True)
        dataset["ve_standard"] = xr.where(
            dataset["kep_1c"] > 0,
            dataset["Ktrans_1c"] / dataset["kep_1c"],
            np.nan,
        )

    dataset = _add_extended_compartment_aliases(dataset)

    if has_tp_parameter:
        if "plasma_transit_time_s" not in dataset:
            dataset["plasma_transit_time_s"] = dataset["Tp"].copy(deep=True)
        if "Fp" not in dataset:
            dataset["Fp"] = xr.where(
                dataset["Tp"] > 0, dataset["vp"] / dataset["Tp"], np.nan
            )
        if "plasma_flow_s" not in dataset:
            dataset["plasma_flow_s"] = dataset["Fp"].copy(deep=True)
        if "plasma_flow_ml_min_100ml" not in dataset:
            dataset["plasma_flow_ml_min_100ml"] = (
                dataset["Fp"] * _PER_S_TO_ML_MIN_PER_100ML
            ).astype(np.float64)
        if "extraction_fraction" not in dataset:
            dataset["extraction_fraction"] = xr.where(
                dataset["Fp"] > 0.0,
                dataset["Ft"] / dataset["Fp"],
                np.nan,
            ).astype(np.float64)

    if {"GFR", "RPF"}.issubset(dataset.data_vars):
        if "gfr_s" not in dataset:
            dataset["gfr_s"] = dataset["GFR"].copy(deep=True)
        if "rpf_s" not in dataset:
            dataset["rpf_s"] = dataset["RPF"].copy(deep=True)
        if "gfr_ml_min_100ml" not in dataset:
            dataset["gfr_ml_min_100ml"] = (
                dataset["GFR"] * _PER_S_TO_ML_MIN_PER_100ML
            ).astype(np.float64)
        if "rpf_ml_min_100ml" not in dataset:
            dataset["rpf_ml_min_100ml"] = (
                dataset["RPF"] * _PER_S_TO_ML_MIN_PER_100ML
            ).astype(np.float64)
        if "filtration_fraction" not in dataset:
            dataset["filtration_fraction"] = xr.where(
                dataset["RPF"] > 0.0,
                dataset["GFR"] / dataset["RPF"],
                np.nan,
            )

    return dataset


def fit_patlak_model(
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    fit_params: dict | None = None,
    use_multiprocessing: bool = True,
) -> xr.Dataset:
    """Fit the Patlak graphical model to labelled concentration curves.

    Parameters
    ----------
    concentration_map : xarray.DataArray
        Tissue concentration in mM with numeric repetition_idx coordinates. Other
        dimensions index voxels in their original order, and the dynamic axis can
        occur anywhere. Compatible coordinates and input attributes are retained.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM, shape (n_time,), sampled on the tissue axis.
        This API applies no blood-to-plasma conversion.
    aif_time_s : numpy.ndarray
        Finite, strictly increasing seconds, shape (n_time,), matching both curves.
        Non-uniform sampling and integer-valued seconds are supported.
    fit_params : dict or None, optional
        Patlak window/AIF-threshold options and shared worker settings. Selection
        retains the model's existing threshold and absolute-index-end rules; explicit
        index/time lists are ignored. The mapping is copied, and numerical defaults
        remain owned by the graphical fitter.
    use_multiprocessing : bool, optional
        Use the shared joblib dispatcher; False runs the same voxel callback serially.
        Overrides the matching mapping entry; number_of_cpu_cores selects workers.

    Returns
    -------
    xarray.Dataset
        Float64 K_trans/Vp/rss_patlak maps in the non-temporal input dimension order.
        K_trans is 1/s and Vp dimensionless. Graphical RSS is dimensionless after AIF
        normalization; concentration-space quality RSS is a separate metric. Retains
        compatible coordinates, input/model metadata and seconds/mM reconstruction
        arrays. Insufficient selected samples give NaN fits. Acquired input values
        are preserved. Calls are eager and deterministic for fixed options/backend;
        no result cache or parameter uncertainty estimate is added.

    Raises
    ------
    ValueError
        If AIF/time shapes or required sampling/scientific options are inconsistent.
    KeyError
        If the dynamic coordinate is absent.

    Examples
    --------
    >>> fitted = fit_patlak_model(
    ...     concentrations, aif_concentration=plasma, aif_time_s=times_s,
    ...     use_multiprocessing=False
    ... )  # doctest: +SKIP

    References
    ----------
    Patlak CS, Blasberg RG, Fenstermacher JD. Graphical evaluation of
    blood-to-brain transfer constants from multiple-time uptake data.
    J Cereb Blood Flow Metab. 1983;3(1):1-7. DOI: 10.1038/jcbfm.1983.1
    Patlak CS, Blasberg RG. Generalizations. J Cereb Blood Flow Metab.
    1985;5(4):584-590. DOI: 10.1038/jcbfm.1985.87
    """
    params = dict(fit_params or {})
    params.update(
        {
            "aif_signal": np.asarray(aif_concentration, dtype=np.float64),
            "time_points_s": np.asarray(aif_time_s, dtype=np.float64),
            "use_multiprocessing": use_multiprocessing,
            "_allow_integer_second_time_axis": True,
            "fit_end_index": params.get(
                "fit_end_index", params.get("fit_stop_index", None)
            ),
            "fit_end_time_s": params.get(
                "fit_end_time_s", params.get("fit_stop_time_s", np.inf)
            ),
        }
    )
    return _fit_dce_maps(
        data=concentration_map,
        model_id="patlak",
        fit_params=params,
    )


def fit_tofts_model(
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    fit_params: dict | None = None,
    use_multiprocessing: bool = True,
) -> xr.Dataset:
    """Fit Tofts-family compartment models to labelled concentration curves.

    Parameters
    ----------
    concentration_map : xarray.DataArray
        Tissue concentration in mM, with a numeric ``repetition_idx`` coordinate.
        The dynamic dimension can occur anywhere; other dimensions index voxels
        in their original order. Compatible coordinates and input attributes are
        retained. Real calculations use float64 without modifying acquired data.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM, shape ``(n_time,)``, on the tissue time axis.
        No blood-to-plasma conversion is applied by this concentration-space API.
    aif_time_s : numpy.ndarray
        Finite, strictly increasing times in seconds, shape ``(n_time,)``.
        Non-uniform sampling is supported. Match the AIF and tissue sample count;
        the complete axis is retained for convolution when fitting a subset.
    fit_params : dict or None, optional
        Scientific/execution mapping. ``model_mode`` is ``"1comp"``, ``"2comp"``
        or ``"both"`` (default). Window bounds and explicit indices/times follow
        ``build_tofts_fit_mask``; negative end indices retain Python slicing.
        ``number_of_cpu_cores`` selects joblib workers. This wrapper copies the
        mapping and retains the existing Tofts starts, bounds and optimizer.
    use_multiprocessing : bool, optional
        Run independent voxels through joblib; False uses the same serial loop.
        Overrides the matching mapping entry. Backend follows joblib configuration.

    Returns
    -------
    xarray.Dataset
        Float64 parameter/RSS maps with the non-temporal input dimension order,
        compatible coordinates, model metadata and reconstruction inputs.
        Ktrans/kep are in 1/s, vp/ve dimensionless and RSS in mM².
        The extended fit retains ``ve_2c`` clipped to [0, 1] and additionally
        reports unbounded ``ve_equivalent = Ktrans_2c / kep_2c`` (float64) and
        ``ve_was_clipped`` (boolean), both with units ``1``. NaN/invalid parameters
        give NaN equivalent volume and false clipping; the flag is not a success
        or physical-validity assessment. Standard-only fits have no such outputs.
        No extra optimizer parameter or covariance estimate is added. Calls are
        eager, deterministic for fixed inputs/options and retain no result cache.

    Raises
    ------
    ValueError
        If required AIF/time metadata or scientific options are inconsistent.
    KeyError
        If the dynamic coordinate is absent.

    Examples
    --------
    >>> fitted = fit_tofts_model(
    ...     concentrations, aif_concentration=plasma, aif_time_s=times_s,
    ...     fit_params={"model_mode": "2comp"}, use_multiprocessing=False
    ... )  # doctest: +SKIP
    >>> equivalent_volume = fitted["ve_equivalent"]  # doctest: +SKIP

    References
    ----------
    Tofts PS, Brix G, Buckley DL, et al. Estimating kinetic parameters from
    dynamic contrast-enhanced T1-weighted MRI of a diffusable tracer:
    standardized quantities and symbols.
    J Magn Reson Imaging. 1999;10(3):223-232.
    DOI: https://doi.org/10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S
    """
    params = dict(fit_params or {})
    params.update(
        {
            "aif_signal": np.asarray(aif_concentration, dtype=np.float64),
            "aif_times": np.asarray(aif_time_s, dtype=np.float64),
            "time_points_s": np.asarray(aif_time_s, dtype=np.float64),
            "input_space": "concentration",
            "use_multiprocessing": use_multiprocessing,
            "fit_end_index": params.get(
                "fit_end_index", params.get("fit_stop_index", None)
            ),
            "fit_end_time_s": params.get(
                "fit_end_time_s", params.get("fit_stop_time_s", np.inf)
            ),
            "aif_start_index": 0,
            "aif_end_index": None,
        }
    )
    return _add_equivalent_volume_outputs(
        _fit_dce_maps(
            data=concentration_map,
            model_id="tofts",
            fit_params=params,
        )
    )


def fit_standard_tofts_model(
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    fit_params: dict | None = None,
    use_multiprocessing: bool = True,
) -> xr.Dataset:
    """
    Fit the standard Tofts model with explicit public output aliases.

    This is a thin wrapper around :func:`fit_tofts_model` that forces
    ``model_mode="1comp"`` and adds clearer aliases such as
    ``Ktrans_standard``, ``kep_standard``, and ``ve_standard``.

    The underlying equation is implemented by
    :func:`hypermri.analysis.dce.models.tofts.tofts_model_1comp`.

    References
    ----------
    Tofts PS, Brix G, Buckley DL, et al. Estimating kinetic parameters from
    dynamic contrast-enhanced T1-weighted MRI of a diffusable tracer:
    standardized quantities and symbols.
    J Magn Reson Imaging. 1999;10(3):223-232.
    DOI: https://doi.org/10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S
    """
    params = dict(fit_params or {})
    params["model_mode"] = "1comp"
    return _add_clear_model_aliases(
        fit_tofts_model(
            concentration_map,
            aif_concentration=aif_concentration,
            aif_time_s=aif_time_s,
            fit_params=params,
            use_multiprocessing=use_multiprocessing,
        )
    )


def fit_extended_tofts_model(
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    fit_params: dict | None = None,
    use_multiprocessing: bool = True,
) -> xr.Dataset:
    """Fit extended Tofts and expose equivalent renal parameter names.

    This wrapper forces the existing Tofts ``model_mode="2comp"``. The reduced
    Sourbron equation is equivalent with Ft = Ktrans and Tt = 1/kep, but its
    independent starts, bounds and optimizer are not substituted here.

    Parameters
    ----------
    concentration_map : xarray.DataArray
        Tissue concentration in mM with a numeric ``repetition_idx`` coordinate.
        Other dimensions identify voxels in their original order; the dynamic
        axis can occur anywhere. Coordinates, orientation and input attributes
        follow ``fit_tofts_model``. Input data are preserved; calculation is float64.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM, shape ``(n_time,)`` on the tissue axis.
    aif_time_s : numpy.ndarray
        Finite, strictly increasing sample times in seconds, shape ``(n_time,)``.
        Uniform and non-uniform sampling are supported; convolution uses the full
        axis even when fitting selected frames.
    fit_params : dict or None, optional
        Window and execution entries accepted by ``fit_tofts_model``. The mapping
        is copied and the submodel forced to extended Tofts; scientific defaults
        retain their existing owner. ``number_of_cpu_cores`` controls joblib workers.
    use_multiprocessing : bool, optional
        Use joblib for independent voxels, or the same serial loop when False.
        Overrides the legacy mapping entry; backend follows joblib configuration.

    Returns
    -------
    xarray.Dataset
        Float64 maps in non-temporal input dimension order, with retained
        coordinates/provenance. ``Ktrans_extended``/``Ft`` and ``kep_extended``
        are 1/s, vp/ve dimensionless, ``Tt`` seconds and RSS mM². Converted
        flow aliases are mL/min/100 mL, not absolute volume flow.
        Existing ``ve_2c``/``ve_extended`` retain clipping to [0, 1]. The added
        ``ve_equivalent`` is the unbounded quotient, and boolean ``ve_was_clipped``
        marks values outside that interval. Both have units ``1``. Unavailable or
        invalid decay parameters give NaN equivalent volume and false clipping;
        the flag is not a fit-success or physical-validity assessment.
        Fits and predictions are unchanged, and covariance is not estimated.
        Calculation is eager, deterministic for fixed inputs and has no cache.

    Raises
    ------
    ValueError
        If required AIF/time metadata or scientific options are inconsistent.
    KeyError
        If the dynamic coordinate is absent.

    Examples
    --------
    >>> fitted = fit_extended_tofts_model(
    ...     concentrations, aif_concentration=plasma, aif_time_s=times_s,
    ...     use_multiprocessing=False
    ... )  # doctest: +SKIP
    >>> unbounded_volume = fitted["ve_equivalent"]  # doctest: +SKIP

    References
    ----------
    Tofts PS, Brix G, Buckley DL, et al. Estimating kinetic parameters from
    dynamic contrast-enhanced T1-weighted MRI of a diffusable tracer:
    standardized quantities and symbols.
    J Magn Reson Imaging. 1999;10(3):223-232.
    DOI: https://doi.org/10.1002/(SICI)1522-2586(199909)10:3<223::AID-JMRI2>3.0.CO;2-S
    """
    params = dict(fit_params or {})
    params["model_mode"] = "2comp"
    return _add_clear_model_aliases(
        fit_tofts_model(
            concentration_map,
            aif_concentration=aif_concentration,
            aif_time_s=aif_time_s,
            fit_params=params,
            use_multiprocessing=use_multiprocessing,
        )
    )


def fit_sourbron_model(
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    fit_params: dict | None = None,
    use_multiprocessing: bool = True,
) -> xr.Dataset:
    """Fit the separable renal compartment model to concentration maps.

    The existing optimizer uses vp/Ft/Tt directly. The reduced model shares its
    equation with extended Tofts, while optional Tp retains the separate sampled
    plasma-transit convolution: both stages use a linearly interpolated AIF
    and one internal uniform grid with step at most 0.1 s before returning
    acquisition samples. The reduced model retains acquisition-grid quadrature.
    Finite integration accuracy and noisy parameter identifiability remain
    distinct; fit success does not guarantee accurate physiological parameters.

    Parameters
    ----------
    concentration_map : xarray.DataArray
        Tissue concentration in mM with a numeric ``repetition_idx`` coordinate.
        The dynamic axis can occur anywhere. Other dimensions identify voxels;
        their order, orientation, compatible coordinates and attributes are
        retained. Real calculations use float64 and preserve acquired values.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM, shape ``(n_time,)`` on the tissue time axis.
        The reduced model treats it as tissue-plasma input; the Tp extension
        treats it as arterial plasma input. No conversion is applied here.
    aif_time_s : numpy.ndarray
        Finite, strictly increasing times in seconds, shape ``(n_time,)``.
        Non-uniform sampling is supported; convolution retains the complete axis.
    fit_params : dict or None, optional
        Window/initial/bound/optimizer entries of
        ``fit_sourbron_separable_compartment_curve``. Parameter order is vp, Ft,
        Tt (fraction, 1/s, s), plus Tp in seconds when ``estimate_tp=True``.
        ``number_of_cpu_cores`` selects joblib workers. The mapping is copied;
        existing scientific defaults and AIF threshold remain unchanged.
    use_multiprocessing : bool, optional
        Use joblib for independent voxels or the same serial loop when False.
        Overrides the mapping entry; backend follows joblib configuration.

    Returns
    -------
    xarray.Dataset
        Float64 maps in non-temporal input dimension order; vp is a fraction,
        Ft is 1/s and Tt/Tp seconds. Converted flow aliases are mL/min/100 mL;
        RSS is mM², RMSE mM and R² dimensionless. Success is boolean and selected
        sample counts int32. Covariance is not estimated; check ``fit_success``
        before using finite unsuccessful results.
        The reduced model retains clipped ``ve_2c``/``ve_extended`` and adds
        unbounded ``ve_equivalent = Ft * Tt`` (float64) plus ``ve_was_clipped``
        (boolean), with units ``1``. The flag marks values outside [0, 1]; NaN or
        invalid parameters yield NaN volume and false clipping. It does not assess
        success or physical validity. The Tp extension has no equivalent-volume
        outputs. Calls are eager, deterministic and cache-free; inputs and mapping
        values are not modified.

    Raises
    ------
    ValueError
        If AIF/time shapes, scientific starts/bounds or optimizer options disagree.
    KeyError
        If the dynamic coordinate is absent.
    IndexError
        If explicit selected indices are outside the time array.

    Examples
    --------
    >>> fitted = fit_sourbron_model(
    ...     concentrations, aif_concentration=plasma, aif_time_s=times_s,
    ...     use_multiprocessing=False
    ... )  # doctest: +SKIP
    >>> unbounded_volume = fitted["ve_equivalent"]  # doctest: +SKIP

    References
    ----------
    Sourbron SP, Michaely HJ, Reiser MF, Schoenberg SO.
    MRI-measurement of perfusion and glomerular filtration in the human kidney
    with a separable compartment model.
    Invest Radiol. 2008;43(1):40-48.
    DOI: https://doi.org/10.1097/RLI.0b013e31815597c5
    """
    params = dict(fit_params or {})
    params.update(
        {
            "aif_signal": np.asarray(aif_concentration, dtype=np.float64),
            "aif_times": np.asarray(aif_time_s, dtype=np.float64),
            "aif_concentration": np.asarray(aif_concentration, dtype=np.float64),
            "time_points_s": np.asarray(aif_time_s, dtype=np.float64),
            "use_multiprocessing": use_multiprocessing,
            "fit_end_index": params.get(
                "fit_end_index", params.get("fit_stop_index", None)
            ),
            "fit_end_time_s": params.get(
                "fit_end_time_s", params.get("fit_stop_time_s", np.inf)
            ),
        }
    )
    return _add_clear_model_aliases(
        _fit_dce_maps(
            data=concentration_map,
            model_id="sourbron",
            fit_params=params,
        )
    )


def fit_two_c_irf_model(
    concentration_map: xr.DataArray,
    *,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    fit_params: dict | None = None,
    use_multiprocessing: bool = True,
) -> xr.Dataset:
    """Fit the cortical renal 2C-IRF model to labelled concentration curves.

    The shared voxel executor calls the canonical concentration-space fitter
    once per voxel. The direct wrapper retains its existing model/AIF metadata
    schema, while higher-level DCE workflows add their own provenance.

    Parameters
    ----------
    concentration_map : xarray.DataArray
        Tissue concentration in mM, with ``repetition_idx`` and its coordinate.
        Other dimensions index voxels in their original order; descending
        coordinates and compatible auxiliary/scalar coordinates are preserved.
        Sample coordinates can contain labels; ``aif_time_s`` supplies seconds.
        The dynamic dimension can occur anywhere. Real values are calculated
        in float64 without modifying the input.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM, shape ``(n_time,)`` after squeezing, sampled
        on the tissue time axis. No blood-to-plasma conversion is applied.
    aif_time_s : numpy.ndarray
        Finite, strictly increasing times in seconds, shape ``(n_time,)`` after
        squeezing. Non-uniform sampling and an absolute time offset are allowed.
        Must match both tissue and AIF lengths; the complete axis is retained
        for convolution even when the fit window selects fewer samples.
    fit_params : dict or None, optional
        Canonical curve fit options: window indices/times, six-element ``x0``,
        ``lower_bounds``/``upper_bounds``, ``loss``, ``f_scale`` and ``max_nfev``.
        See ``fit_two_c_irf_curve`` and ``TwoCIrfFitOptions`` for their units,
        parameter order and defaults. ``number_of_cpu_cores`` sets joblib workers.
        The mapping is copied; inputs are not modified.
    use_multiprocessing : bool, optional
        Use the shared joblib dispatcher when more than one voxel is present.
        False runs the same callback serially. This explicit argument overrides
        the matching legacy mapping entry; backend follows joblib configuration.

    Returns
    -------
    xarray.Dataset
        Float64 maps with the non-temporal input dimensions in their original
        order. ``GFR``/``RPF`` and their ``gfr_s``/``rpf_s`` aliases are normalized
        flow in 1/s; converted flows are mL/min/100 mL. Absolute mL/min requires
        multiplication by tissue volume. Delays and mean transit times are
        seconds, dispersion rates 1/s and filtration fraction dimensionless.
        RSS is mM² and RMSE mM. ``fit_success`` is boolean and ``n_points_fit``
        int64. Unavailable fits have NaN parameters and false success; finite
        unsuccessful optimizer results remain available and must be interpreted
        with the status. Covariance and parameter uncertainty are not estimated.
        Temporal variables retain seconds, plasma AIF and the requested boolean
        ``fit_window_mask``; voxel counts additionally exclude non-finite samples.
        The wrapper records model/AIF attributes; generic input attributes are
        not copied by this direct API. Calls are eager and deterministic for a
        fixed input/options/backend, with no result cache.

    Raises
    ------
    ValueError
        If time/curve lengths, scientific bounds/initial values or optimizer
        settings are inconsistent. Model errors propagate in both execution modes.
    KeyError
        If the ``repetition_idx`` coordinate is absent.
    IndexError
        If explicitly selected sample indices are outside the time array.

    Notes
    -----
    Each voxel uses a single bounded optimizer start. ``fit_success`` is an
    optimizer termination flag; neither it nor high R² establishes identifiable
    or accurate GFR/RPF. Delay and dispersion parameters can trade off with the
    flows, and sample-grid quadrature can bias their recovery. The synthetic
    limitations documented by ``fit_two_c_irf_curve`` also apply to these maps
    and pipeline results. No multi-start selection or parameter uncertainty
    estimate is applied.

    Examples
    --------
    >>> fitted = fit_two_c_irf_model(
    ...     concentrations, aif_concentration=plasma, aif_time_s=times_s,
    ...     use_multiprocessing=False
    ... )  # doctest: +SKIP
    >>> normalized_flow = fitted["GFR"]  # doctest: +SKIP
    """
    params = dict(fit_params or {})
    time_points_s = _validate_dce_time_axis("aif_time_s", aif_time_s)
    aif = _coerce_numeric_1d("aif_concentration", aif_concentration)
    if len(aif) != len(time_points_s):
        raise ValueError("aif_concentration and aif_time_s must have matching lengths.")
    dynamic_dim = "repetition_idx"
    params["use_multiprocessing"] = use_multiprocessing
    params["time_points_s"] = time_points_s
    params["aif_concentration"] = aif
    dataset = VoxelwiseExecutor()._execute_fits(
        data=concentration_map.astype(np.float64, copy=False),
        model_id="2c_irf",
        fit_params=params,
        single_voxel_fit_func=_fit_two_c_irf_voxel,
        output_param_names=_TWO_C_IRF_OUTPUT_NAMES,
        dynamic_dim=dynamic_dim,
        dynamic_coords=time_points_s,
    )

    fit_window_mask = select_fit_window(
        time_points_s,
        params,
        fit_type="2c_irf",
    ).assign_coords(repetition_idx=concentration_map.coords["repetition_idx"])
    if "time_s" in concentration_map.coords:
        fit_window_mask = fit_window_mask.assign_coords(
            time_s=concentration_map.coords["time_s"]
        )

    dataset["fit_window_mask"] = fit_window_mask.astype(bool)
    dataset["time_points_s"] = xr.DataArray(
        np.asarray(time_points_s, dtype=np.float64),
        dims=["repetition_idx"],
        coords={"repetition_idx": concentration_map.coords["repetition_idx"]},
        attrs={"units": "s"},
    )
    dataset["aif_time_s"] = xr.DataArray(
        np.asarray(time_points_s, dtype=np.float64),
        dims=["repetition_idx"],
        coords={"repetition_idx": concentration_map.coords["repetition_idx"]},
        attrs={"units": "s"},
    )
    dataset["aif_concentration"] = xr.DataArray(
        np.asarray(aif, dtype=np.float64),
        dims=["repetition_idx"],
        coords={"repetition_idx": concentration_map.coords["repetition_idx"]},
        attrs={"concentration_space": "plasma"},
    )
    if "time_s" in concentration_map.coords:
        dataset["time_points_s"] = dataset["time_points_s"].assign_coords(
            time_s=concentration_map.coords["time_s"]
        )
        dataset["aif_time_s"] = dataset["aif_time_s"].assign_coords(
            time_s=concentration_map.coords["time_s"]
        )
        dataset["aif_concentration"] = dataset["aif_concentration"].assign_coords(
            time_s=concentration_map.coords["time_s"]
        )

    dataset.attrs.update(
        {
            "dynamic_dim": "repetition_idx",
            "fit_type": "2c_irf",
            "model_name": "2c_irf",
            "model_equation": (
                "Ccortex(t) = Cp(t) * Rcortex(t), where Rcortex is the cortical "
                "piecewise-exponential 2C-IRF retention kernel of Chen et al. (2014)."
            ),
            "aif_concentration_space": "plasma",
        }
    )
    return _add_clear_model_aliases(dataset)


def fit_empirical_log_logistic_model(
    concentration_map: xr.DataArray,
    *,
    fit_params: dict | None = None,
    use_multiprocessing: bool = True,
) -> xr.Dataset:
    """Fit the empirical log-logistic model to labelled concentration curves.

    Parameters
    ----------
    concentration_map : xarray.DataArray
        Real tissue concentration in mM with ``repetition_idx`` and any labelled
        spatial dimensions in any order. Coordinate directions are preserved.
    fit_params : dict or None, optional
        Scientific overrides of ``fit_empirical_log_logistic_curve`` plus
        ``number_of_cpu_cores`` for execution. Time in seconds comes from
        ``time_points_s``, acquisition coordinates or the existing time-axis
        fallback. Sampling-dependent initial values and bounds remain owned by
        the curve fitter; the six-parameter fit requires no AIF.
    use_multiprocessing : bool, default True
        Run independent voxels with the existing shared joblib executor.

    Returns
    -------
    xarray.Dataset
        Six fitted parameter maps, derived ``time_to_peak_s`` and
        ``half_rise_time_s`` in absolute acquisition seconds, RSS in mM², RMSE
        in mM, boolean ``fit_success`` and int64 ``n_points_fit``. Maps follow
        the non-temporal input dimension order and coordinates; floating maps
        use float64. The half-rise map equals arrival plus rise duration and
        refers to the log-logistic rise term before baseline and washout, not
        the half-height of the complete curve. It has units ``s`` and is NaN
        when a fit is unavailable. Check success before interpreting finite
        results; covariance and derived-time uncertainty are not estimated.
        The dataset also carries the temporal ``fit_window_mask`` and seconds
        axis, model and existing input/provenance metadata. Calculation is
        eager, deterministic and preserves the inputs without caching.

    Raises
    ------
    ValueError
        If the time axis, scientific options, bounds or starting values are
        incompatible with the data.
    KeyError
        If ``repetition_idx`` is absent from the input grid.

    Examples
    --------
    >>> fitted = fit_empirical_log_logistic_model(
    ...     concentrations, fit_params={"time_points_s": times_s},
    ...     use_multiprocessing=False
    ... )  # doctest: +SKIP
    >>> half_rise = fitted["half_rise_time_s"]  # doctest: +SKIP
    """
    params = dict(fit_params or {})
    time_points_s = _derive_time_points_from_concentration_map(
        concentration_map,
        params,
    )
    params["time_points_s"] = np.asarray(time_points_s, dtype=np.float64)
    params["use_multiprocessing"] = use_multiprocessing

    dataset = VoxelwiseExecutor().execute(
        data=concentration_map.astype(np.float64, copy=False),
        model_id="empirical_log_logistic",
        fit_params=params,
        single_voxel_fit_func=_fit_empirical_log_logistic_voxel,
        output_param_names=_EMPIRICAL_LOG_LOGISTIC_OUTPUT_NAMES,
        dynamic_dim="repetition_idx",
        model_name="empirical_log_logistic",
        model_equation=_EMPIRICAL_LOG_LOGISTIC_MODEL_EQUATION,
    )

    fit_window_mask = select_fit_window(
        time_points_s,
        params,
        fit_type="empirical_log_logistic",
    )
    fit_window_mask = fit_window_mask.assign_coords(
        repetition_idx=concentration_map.coords["repetition_idx"]
    )
    if "time_s" in concentration_map.coords:
        fit_window_mask = fit_window_mask.assign_coords(
            time_s=concentration_map.coords["time_s"]
        )
    dataset["fit_window_mask"] = fit_window_mask.astype(bool)
    dataset["time_points_s"] = xr.DataArray(
        np.asarray(time_points_s, dtype=np.float64),
        dims=["repetition_idx"],
        coords={"repetition_idx": concentration_map.coords["repetition_idx"]},
        attrs={"units": "s"},
    )
    if "time_s" in concentration_map.coords:
        dataset["time_points_s"] = dataset["time_points_s"].assign_coords(
            time_s=concentration_map.coords["time_s"]
        )

    dataset.attrs.update(
        {
            "dynamic_dim": "repetition_idx",
            "fit_type": "empirical_log_logistic",
            "model_name": "empirical_log_logistic",
            "model_equation": _EMPIRICAL_LOG_LOGISTIC_MODEL_EQUATION,
        }
    )
    dataset["half_rise_time_s"].attrs.update(_EMPIRICAL_HALF_RISE_TIME_ATTRS)
    return dataset


def compute_fit_quality_metrics(
    fit_results: xr.Dataset,
    concentration_map: xr.DataArray,
    *,
    fit_type: str,
    aif_concentration: np.ndarray,
    aif_time_s: np.ndarray,
    fit_params: dict | None = None,
) -> xr.Dataset:
    """Compute concentration-space residual metrics and fit-quality flags.

    Parameters
    ----------
    fit_results : xarray.Dataset
        Model parameter maps on the concentration map's non-temporal grid.
        Required variable names follow the selected public model fitter.
    concentration_map : xarray.DataArray
        Real concentration in mM with ``repetition_idx`` and any labelled spatial
        dimensions. Acquisition coordinate values, directions and units are
        retained; their order need not put repetition last.
    fit_type : {"patlak", "tofts", "sourbron", "2c_irf", "empirical_log_logistic"}
        Model used to reconstruct predictions. Tofts parameter count and model
        selection follow ``model_mode`` and available RSS/parameter maps.
    aif_concentration : numpy.ndarray
        Plasma concentration in mM on the acquisition times. Unused by the
        empirical model.
    aif_time_s : numpy.ndarray
        Acquisition times in seconds. The empirical model instead derives its
        axis from the concentration-map coordinates/options.
    fit_params : dict or None, optional
        Model window options/bounds and ``insufficient_signal_threshold``
        (default 1e-6 mM). Selection follows the selected fitter: Tofts keeps
        Python slicing for a negative exclusive end; Patlak ignores
        ``fit_indices`` and ``fit_times_s``; Sourbron ignores ``fit_times_s``.
        Their AIF thresholds and index/time bounds also follow the fitter.
        2C-IRF and empirical fitting use ``select_fit_window``. Pass the same
        options and acquisition inputs used for fitting.

    Returns
    -------
    xarray.Dataset
        Float residual maps ``rss`` (mM²), ``rmse`` (mM), ``r_squared``, ``aic``
        and ``bic`` on the non-temporal grid. RSS sums non-NaN squared residuals;
        observation counts use finite measured samples. RMSE and information
        criteria floor RSS at float64 tiny, preserving exact zero stored RSS.
        Constant observations yield NaN R². AIC/BIC are NaN when observations
        do not outnumber fitted parameters. Boolean ``converged``, ``failed``,
        ``hit_bound`` and ``insufficient_signal`` maps describe quality, with
        ``fit_window_mask`` on repetition and optional ``selected_tofts_model``.
        This eager, deterministic calculation preserves all input arrays/maps.

    Raises
    ------
    ValueError
        If the model is unsupported, required time/window inputs are invalid
        or prediction dimensions cannot be aligned.
    KeyError
        If a required model parameter map is absent.

    Examples
    --------
    >>> quality = compute_fit_quality_metrics(
    ...     fitted, concentrations, fit_type="patlak",
    ...     aif_concentration=plasma, aif_time_s=times_s
    ... )  # doctest: +SKIP
    """
    fit_params = dict(fit_params or {})
    normalized_fit_type = fit_type
    time_points_for_fit = np.asarray(aif_time_s, dtype=np.float64)
    from .model_selection import _DCE_MODELS

    if normalized_fit_type not in _DCE_MODELS:
        raise ValueError(f"Unsupported fit_type for metric calculation: {fit_type}")
    model = _DCE_MODELS[normalized_fit_type].module
    derive_time = getattr(model, "_quality_time_points", None)
    if derive_time is not None:
        time_points_for_fit = derive_time(concentration_map, fit_params)
    rep_coord = concentration_map.coords.get(
        "repetition_idx", np.arange(len(time_points_for_fit))
    )

    mask = model._quality_window(time_points_for_fit, aif_concentration, fit_params)
    fit_window_mask = xr.DataArray(
        mask, dims=["repetition_idx"], name="fit_window_mask"
    )
    fit_window_mask = fit_window_mask.assign_coords(repetition_idx=rep_coord)
    if "time_s" in concentration_map.coords:
        fit_window_mask = fit_window_mask.assign_coords(
            time_s=concentration_map.coords["time_s"]
        )

    prediction, parameter_count, finite_params, hit_bound, selected_tofts_model = (
        model._quality_inputs(
            fit_results,
            concentration_map,
            aif_concentration=aif_concentration,
            aif_time_s=aif_time_s,
            time_points_for_fit=time_points_for_fit,
            fit_params=fit_params,
            rep_coord=rep_coord,
        )
    )

    observed = concentration_map.where(fit_window_mask)
    predicted = prediction.where(fit_window_mask)
    statistics = _residual_statistics(
        observed,
        predicted,
        sample_dim="repetition_idx",
        rss_floor=np.finfo(np.float64).tiny,
    )
    n_obs = statistics["n_observations"]
    rss = statistics["rss"]
    rss_safe = statistics["rss_safe"]
    rmse = statistics["rmse"]
    r_squared = statistics["r_squared"]

    if isinstance(parameter_count, xr.DataArray):
        param_count_da = parameter_count.astype(np.float64)
    else:
        param_count_da = xr.full_like(rss, float(parameter_count), dtype=np.float64)

    aic = xr.where(
        n_obs > param_count_da,
        n_obs * np.log(rss_safe / xr.where(n_obs > 0, n_obs, np.nan))
        + 2.0 * param_count_da,
        np.nan,
    )
    bic = xr.where(
        n_obs > param_count_da,
        n_obs * np.log(rss_safe / xr.where(n_obs > 0, n_obs, np.nan))
        + param_count_da * np.log(xr.where(n_obs > 0, n_obs, np.nan)),
        np.nan,
    )

    peak_signal = np.abs(observed).max(dim="repetition_idx", skipna=True)
    minimum_required_samples = param_count_da + 1.0
    insufficient_signal = (
        peak_signal <= float(fit_params.get("insufficient_signal_threshold", 1e-6))
    ) | (n_obs < minimum_required_samples)
    converged = finite_params & np.isfinite(rmse) & ~insufficient_signal
    failed = ~converged

    metrics = xr.Dataset(
        {
            "rss": rss,
            "rmse": rmse,
            "r_squared": r_squared,
            "aic": aic,
            "bic": bic,
            "converged": converged.astype(bool),
            "failed": failed.astype(bool),
            "hit_bound": hit_bound.astype(bool),
            "insufficient_signal": insufficient_signal.astype(bool),
            "fit_window_mask": fit_window_mask.astype(bool),
        }
    )

    if selected_tofts_model is not None:
        metrics["selected_tofts_model"] = selected_tofts_model

    return metrics


def _annotate_output_dataset(dataset: xr.Dataset, *, fit_type: str) -> xr.Dataset:
    _set_var_attrs(
        dataset,
        "t1_map",
        long_name="Aligned T1 Map",
        description="Pre-contrast T1 map aligned to the DCE grid.",
        units="ms",
        model="t1_inversion_recovery",
    )
    _set_var_attrs(
        dataset,
        "s0_map",
        long_name="Baseline Signal Map",
        description="Baseline S0 estimated from the pre-contrast DCE frames.",
        units="a.u.",
        model="baseline_signal",
    )
    _set_var_attrs(
        dataset,
        "concentration_map",
        long_name="Concentration Map",
        description="Signal-to-concentration conversion of the DCE series using aligned T1 and S0 maps.",
        units="mM",
        model="spgr_signal_to_concentration",
    )
    _set_var_attrs(
        dataset,
        "aif_signal_raw",
        long_name="AIF Raw Signal",
        description="Raw arterial input function signal used for audit and QC.",
        units="a.u.",
        model="arterial_input_function",
    )
    _set_var_attrs(
        dataset,
        "aif_concentration",
        long_name="AIF Concentration",
        description="Arterial input function in concentration units after validation and cleaning.",
        units="mM",
        model="arterial_input_function",
    )
    _set_var_attrs(
        dataset,
        "aif_time_s",
        long_name="AIF Time Axis",
        description="Time axis associated with the arterial input function.",
        units="s",
        model="arterial_input_function",
    )
    _set_var_attrs(
        dataset,
        "fit_window_mask",
        long_name="Fit Window Mask",
        description="Boolean mask identifying the dynamic frames included in the model fit.",
        units="bool",
        model="fit_window",
    )
    _set_var_attrs(
        dataset,
        "rss",
        long_name="Residual Sum of Squares",
        description="Residual sum of squares computed on the selected fit window in concentration space.",
        units="mM^2",
        model=fit_type,
    )
    _set_var_attrs(
        dataset,
        "rmse",
        long_name="Root Mean Squared Error",
        description="Root mean squared residual in concentration space.",
        units="mM",
        model=fit_type,
    )
    _set_var_attrs(
        dataset,
        "r_squared",
        long_name="Pseudo R-Squared",
        description="Residual-based pseudo-R² computed on the selected fit window.",
        units="1",
        model=fit_type,
    )
    _set_var_attrs(
        dataset,
        "aic",
        long_name="Akaike Information Criterion",
        description="AIC computed from concentration-space residuals on the fit window.",
        units="1",
        model=fit_type,
    )
    _set_var_attrs(
        dataset,
        "bic",
        long_name="Bayesian Information Criterion",
        description="BIC computed from concentration-space residuals on the fit window.",
        units="1",
        model=fit_type,
    )

    for status_name in ["converged", "failed", "hit_bound", "insufficient_signal"]:
        _set_var_attrs(
            dataset,
            status_name,
            long_name=status_name.replace("_", " ").title(),
            description=f"Boolean fit-status flag: {status_name}.",
            units="bool",
            model=fit_type,
        )

    from .model_selection import _DCE_MODELS

    model = _DCE_MODELS.get(fit_type)
    if model is not None:
        model.module._annotate_model_output(dataset)

    for var_name, long_name, units, description, model in (
        (
            "Ktrans_standard",
            "Standard Tofts Ktrans",
            "s^-1",
            "Clear-name alias of the standard Tofts transfer parameter.",
            "tofts",
        ),
        (
            "kep_standard",
            "Standard Tofts kep",
            "s^-1",
            "Clear-name alias of the standard Tofts efflux rate constant.",
            "tofts",
        ),
        (
            "ve_standard",
            "Standard Tofts ve",
            "1",
            "Extracellular volume fraction derived from the standard Tofts parameterization.",
            "tofts",
        ),
        (
            "Ktrans_extended",
            "Extended Tofts Ktrans",
            "s^-1",
            "Clear-name alias of the extended Tofts transfer parameter.",
            "tofts",
        ),
        (
            "kep_extended",
            "Extended Tofts kep",
            "s^-1",
            "Clear-name alias of the extended Tofts efflux rate constant.",
            "tofts",
        ),
        (
            "ve_extended",
            "Extended Tofts ve",
            "1",
            "Clear-name alias of the legacy extracellular volume clipped to [0, 1]; "
            "ve_equivalent retains the unbounded value.",
            "tofts",
        ),
        (
            "vp_plasma",
            "Extended Tofts vp",
            "1",
            "Clear-name alias of the extended Tofts plasma volume fraction.",
            "tofts",
        ),
        (
            "vp",
            "Sourbron-Equivalent vp",
            "1",
            "Plasma volume fraction exposed through the Sourbron-equivalent parameterization of extended Tofts.",
            "sourbron",
        ),
        (
            "Ft",
            "Sourbron-Equivalent Ft",
            "s^-1",
            "Transfer parameter exposed as the Sourbron-equivalent tubular flow term.",
            "sourbron",
        ),
        (
            "Tt",
            "Sourbron-Equivalent Tt",
            "s",
            "Transit time exposed through the Sourbron-equivalent parameterization of extended Tofts.",
            "sourbron",
        ),
        (
            "tubular_flow_s",
            "Tubular Flow",
            "s^-1",
            "Tubular flow term in per-second units.",
            "sourbron",
        ),
        (
            "tubular_flow_ml_min_100ml",
            "Tubular Flow",
            "mL/min/100mL",
            "Tubular flow converted to mL/min/100mL.",
            "sourbron",
        ),
        (
            "tubular_transit_time_s",
            "Tubular Transit Time",
            "s",
            "Tubular transit time in seconds.",
            "sourbron",
        ),
        (
            "kidney_plasma_volume_fraction",
            "Kidney Plasma Volume Fraction",
            "1",
            "Clear-name alias of the renal two-compartment plasma volume fraction.",
            "sourbron",
        ),
        (
            "kidney_outflow_rate_s",
            "Kidney Outflow Rate",
            "s^-1",
            "Clear-name alias of the renal outflow term.",
            "sourbron",
        ),
        (
            "kidney_outflow_ml_min_100ml",
            "Kidney Outflow",
            "mL/min/100mL",
            "Renal outflow term converted to mL/min/100mL.",
            "sourbron",
        ),
        (
            "kidney_transit_time_s",
            "Kidney Transit Time",
            "s",
            "Clear-name alias of the renal transit-time parameter.",
            "sourbron",
        ),
        (
            "gfr_s",
            "2C-IRF GFR",
            "s^-1",
            "Clear-name alias of the normalized 2C-IRF GFR term.",
            "2c_irf",
        ),
        (
            "rpf_s",
            "2C-IRF RPF",
            "s^-1",
            "Clear-name alias of the normalized 2C-IRF RPF term.",
            "2c_irf",
        ),
        (
            "gfr_ml_min_100ml",
            "2C-IRF GFR",
            "mL/min/100mL",
            "Normalized 2C-IRF GFR converted to mL/min/100mL.",
            "2c_irf",
        ),
        (
            "rpf_ml_min_100ml",
            "2C-IRF RPF",
            "mL/min/100mL",
            "Normalized 2C-IRF RPF converted to mL/min/100mL.",
            "2c_irf",
        ),
        (
            "filtration_fraction",
            "2C-IRF Filtration Fraction",
            "1",
            "Derived filtration fraction from the 2C-IRF fit.",
            "2c_irf",
        ),
    ):
        _set_var_attrs(
            dataset,
            var_name,
            long_name=long_name,
            description=description,
            units=units,
            model=model,
        )

    return dataset
