"""Convert DCE signals and contrast concentrations with existing model rules.

SPGR and linear enhancement formulas retain their scientific defaults and
saturation policies. Operations are eager, deterministic and do not mutate
inputs or create caches. Importing the module performs no conversion or fitting.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

_EQ8_LINEAR_SCALE_ALIASES = frozenset({"eq8", "schabel_eq8"})

#: Longitudinal relaxivity of the contrast agent in 1/(mM s) used by the DCE
#: signal conversion, AIF preparation and model fitters when no relaxivity is
#: given.
DEFAULT_R1_PER_MM_S = 3.5


def resolve_relaxivity(
    r1: float | None,
    fit_params: Mapping[str, object],
    *,
    operation: str,
    fit_params_name: str = "fit_params",
) -> float:
    """Return the one effective contrast-agent relaxivity for an operation.

    DCE entry points accept the relaxivity both as an explicit ``r1`` argument
    and as ``fit_params["r1"]``.  Both spellings name the same setting, so
    either may be given, and when both are given they must agree.  When
    neither is given, :data:`DEFAULT_R1_PER_MM_S` is used and a message is
    printed.

    Parameters
    ----------
    r1 : float or None
        Explicit relaxivity in 1/(mM s), or ``None`` when the caller did not
        pass one.
    fit_params : Mapping[str, object]
        Fit parameters of the operation.  Only the key ``"r1"`` is read; the
        mapping is not modified.
    operation : str
        Name of the public operation, used in error messages.
    fit_params_name : str, default "fit_params"
        Name of the fit-parameter argument of ``operation``, used in error
        messages.

    Returns
    -------
    float
        Effective relaxivity in 1/(mM s); positive and finite. Deterministic
        scalar conversion without caching or input mutation; prints the default
        selection when neither spelling supplies a value.

    Raises
    ------
    ValueError
        If ``r1`` and ``fit_params["r1"]`` are both given and differ, or if the
        effective relaxivity is not positive and finite.

    Examples
    --------
    >>> resolve_relaxivity(None, {"r1": 4.5}, operation="prepare_aif")
    4.5
    >>> resolve_relaxivity(4.5, {"r1": 4.5}, operation="prepare_aif")
    4.5
    """
    fit_params_r1 = fit_params.get("r1")
    if r1 is not None and fit_params_r1 is not None:
        if float(r1) != float(fit_params_r1):
            raise ValueError(
                f"{operation} received r1={r1} and {fit_params_name}['r1']="
                f"{fit_params_r1}. Pass the relaxivity in 1/(mM s) only once, "
                f"either as the r1 argument or in {fit_params_name}."
            )
    resolved = r1 if r1 is not None else fit_params_r1
    if resolved is None:
        resolved = DEFAULT_R1_PER_MM_S
        print(f"No r1 was passed to the fit, setting it to {resolved} 1 / (mM * s)")
    resolved = float(resolved)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(
            f"{operation} requires a positive finite relaxivity r1 in "
            f"1/(mM s), got {resolved}."
        )
    return resolved


def _contains_true(values: object) -> bool:
    """Return True when the input contains at least one truthy element."""
    if isinstance(values, xr.DataArray):
        return bool(np.any(np.asarray(values.values, dtype=bool)))
    return bool(np.any(np.asarray(values, dtype=bool)))


def _count_true(values: object) -> int:
    """Return the number of truthy elements in the input."""
    if isinstance(values, xr.DataArray):
        return int(np.count_nonzero(np.asarray(values.values, dtype=bool)))
    return int(np.count_nonzero(np.asarray(values, dtype=bool)))


def _where_like(condition: object, x: object, y: object):
    """Dispatch ``where`` to xarray when needed to preserve labeled arrays."""
    if any(isinstance(value, xr.DataArray) for value in (condition, x, y)):
        return xr.where(condition, x, y)
    return np.where(condition, x, y)


def _expand_curve_mask(mask: object, target: object):
    """Broadcast a per-curve mask across the full signal array shape."""
    if isinstance(target, xr.DataArray):
        if isinstance(mask, xr.DataArray):
            expanded = mask
        else:
            expanded = xr.DataArray(np.asarray(mask, dtype=bool))
        return expanded.broadcast_like(target)
    mask_arr = np.asarray(mask, dtype=bool)
    target_shape = np.asarray(target).shape
    while mask_arr.ndim < len(target_shape):
        mask_arr = np.expand_dims(mask_arr, axis=-1)
    return np.broadcast_to(mask_arr, target_shape)


def signal_to_concentration(
    S_t: np.ndarray = None,
    S_0: np.ndarray = None,
    T10_s: np.ndarray = None,
    TR_s: float = None,
    flip_angle_deg: float = None,
    r1: float = None,
    *,
    mode: str = "spgr",
    linear_scale: float | str = 1.0,
    convert_blood_to_plasma: bool = False,
    hematocrit: float | None = None,
    saturation_threshold: float = 0.98,
    saturation_behavior: str = "clip",
    warn_on_saturation: bool = False,
    return_saturation_info: bool = False,
):
    """Convert acquired DCE signals to contrast concentration.

    Parameters
    ----------
    S_t : numpy.ndarray or xarray.DataArray
        Real signal in arbitrary units, with any broadcast-compatible shape.
        For ``"nan_all"`` saturation handling the last result axis is the
        sample/time axis; xarray uses its last dimension, without reordering.
    S_0 : numpy.ndarray or xarray.DataArray or float
        Precontrast baseline signal in the same units as ``S_t``; must broadcast
        with it. None retains the existing NaN failure behavior.
    T10_s : numpy.ndarray or xarray.DataArray or float
        Precontrast longitudinal relaxation time in seconds; broadcast-compatible
        with the signal. Required for SPGR and ``"eq8"`` scaling.
    TR_s : float
        Repetition time in seconds, required for SPGR conversion.
    flip_angle_deg : float
        Acquisition flip angle in degrees, required for SPGR conversion.
    r1 : float or None, optional
        Contrast-agent relaxivity in 1/(mM s), required for SPGR and ``"eq8"``.
        This low-level function does not substitute the workflow default when
        omitted; use :func:`resolve_relaxivity` at workflow entry points.
    mode : {"spgr", "signal_enhancement", "relative_signal_enhancement"}, optional
        Conversion mode. ``"spgr"`` uses the nonlinear spoiled gradient-echo
        inversion. ``"signal_enhancement"`` uses the simplified linear mapping
        ``C(t) = a * (S(t) - S0)``. ``"relative_signal_enhancement"`` uses the
        simplified relative enhancement mapping
        ``C(t) = a * (S(t) - S0) / S0``.
    linear_scale : float or {"eq8", "schabel_eq8"}, optional
        Scaling coefficient ``a`` used for the simplified linear conversion
        modes. Ignored when ``mode="spgr"``. For
        ``mode="relative_signal_enhancement"``, ``linear_scale="eq8"`` applies
        the Schabel-Parker low-concentration approximation
        ``a = 1 / (r1 * T10)``. For ``mode="signal_enhancement"``, the
        equivalent raw-signal scaling ``a = 1 / (S0 * r1 * T10)`` is used.
    convert_blood_to_plasma : bool, optional
        If True, convert the recovered whole-blood concentration to plasma
        concentration by dividing by ``1 - hematocrit``. This should only be
        used for blood-pool signals such as an AIF ROI, not for tissue voxels.
    hematocrit : float, optional
        Fractional hematocrit in the range ``[0, 1)``. Required when
        ``convert_blood_to_plasma=True``.
    saturation_threshold : float, optional
        Relative SPGR saturation limit used for ``mode="spgr"``. Saturation is
        flagged when ``B * S(t) / S0 >= saturation_threshold``. Must lie in the
        open interval ``(0, 1)``.
    saturation_behavior : {"clip", "nan", "nan_all"}, optional
        Behavior when SPGR saturation is detected. ``"clip"`` clamps to the saturation
        threshold.
        ``"nan"`` sets only saturated samples to ``NaN``. ``"nan_all"``
        invalidates the whole returned time course when any saturation is
        detected.
    warn_on_saturation : bool, optional
        If True, emit a ``RuntimeWarning`` when saturation is detected in
        ``mode="spgr"``.
    return_saturation_info : bool, optional
        If True, return ``(C_t, info)`` where ``info`` contains saturation
        diagnostics.

    Returns
    -------
    C_t : numpy.ndarray or xarray.DataArray or scalar
        Concentration in mM with the broadcast shape. xarray arithmetic retains
        aligned dimension labels, coordinate values and direction; attrs follow
        xarray's active keep-attrs policy. NumPy/xarray arithmetic determines
        calculation precision without an explicit dtype cast. Invalid options
        or caught conversion errors return scalar NaN, regardless of input shape.
    info : dict, optional
        Returned only when ``return_saturation_info=True``. Contains the boolean
        ``saturation_detected``, broadcast ``saturation_mask``, sample count and
        fraction, threshold and normalized behavior. Linear modes report zero
        saturation; caught errors return an empty dictionary.

    Notes
    -----
    This function preserves the legacy broad exception-to-NaN contract; it does
    not raise validation errors for invalid scientific options. NumPy floating
    warnings may still occur. Computation is eager and deterministic, preserves
    inputs and creates no cache. Saturation warnings are opt-in RuntimeWarning.
    No physical validity beyond the existing formulas is inferred.

    Examples
    --------
    >>> signal_to_concentration(
    ...     np.array([100.0, 125.0]), 100.0,
    ...     mode="relative_signal_enhancement", linear_scale=2.0,
    ... )
    array([0. , 0.5])
    """
    try:
        normalized_mode = str(mode).strip().lower()
        saturation_threshold = float(saturation_threshold)
        normalized_saturation_behavior = str(saturation_behavior).strip().lower()
        saturation_mask = False
        saturation_detected = False
        resolved_linear_scale: float | np.ndarray | xr.DataArray

        if not 0.0 < saturation_threshold < 1.0:
            raise ValueError(
                "saturation_threshold must be in the open interval (0, 1)."
            )
        if normalized_saturation_behavior not in {"clip", "nan", "nan_all"}:
            raise ValueError(
                "saturation_behavior must be one of 'clip', 'nan', or 'nan_all'."
            )

        if normalized_mode == "spgr":
            theta = np.deg2rad(flip_angle_deg)
            S_rel = S_t / S_0
            E10 = np.exp(-TR_s / T10_s)
            B = (1 - E10) / (1 - np.cos(theta) * E10)
            max_s_rel = saturation_threshold / _where_like(B > 0, B, np.nan)
            saturation_mask = S_rel >= max_s_rel
            saturation_detected = _contains_true(saturation_mask)

            if warn_on_saturation and saturation_detected:
                warnings.warn(
                    (
                        "SPGR signal-to-concentration conversion reached the "
                        f"saturation limit; applying saturation_behavior="
                        f"'{normalized_saturation_behavior}'."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

            if normalized_saturation_behavior == "clip":
                S_rel_eval = _where_like(saturation_mask, max_s_rel, S_rel)
            else:
                S_rel_eval = S_rel

            num = 1 - B * S_rel_eval * np.cos(theta)
            den = 1 - B * S_rel_eval
            log_arg = num / den
            log_arg = _where_like(log_arg > 1e-6, log_arg, np.nan)

            if normalized_saturation_behavior == "nan":
                log_arg = _where_like(saturation_mask, np.nan, log_arg)
            C_t = 1 / (r1 * TR_s) * (np.log(log_arg) - (TR_s / T10_s))
            if normalized_saturation_behavior == "nan_all" and saturation_detected:
                if isinstance(saturation_mask, xr.DataArray):
                    curve_mask = saturation_mask.any(dim=saturation_mask.dims[-1])
                else:
                    curve_mask = np.any(
                        np.asarray(saturation_mask, dtype=bool), axis=-1
                    )
                C_t = _where_like(_expand_curve_mask(curve_mask, C_t), np.nan, C_t)
        elif normalized_mode == "signal_enhancement":
            if isinstance(linear_scale, str):
                normalized_linear_scale = linear_scale.strip().lower()
                if normalized_linear_scale in _EQ8_LINEAR_SCALE_ALIASES:
                    resolved_linear_scale = 1.0 / (S_0 * r1 * T10_s)
                else:
                    resolved_linear_scale = float(linear_scale)
            else:
                resolved_linear_scale = float(linear_scale)
            C_t = resolved_linear_scale * (S_t - S_0)
        elif normalized_mode == "relative_signal_enhancement":
            if isinstance(linear_scale, str):
                normalized_linear_scale = linear_scale.strip().lower()
                if normalized_linear_scale in _EQ8_LINEAR_SCALE_ALIASES:
                    resolved_linear_scale = 1.0 / (r1 * T10_s)
                else:
                    resolved_linear_scale = float(linear_scale)
            else:
                resolved_linear_scale = float(linear_scale)
            C_t = resolved_linear_scale * (S_t - S_0) / S_0
        else:
            raise ValueError(
                "mode must be one of 'spgr', 'signal_enhancement', or "
                "'relative_signal_enhancement'."
            )

        if convert_blood_to_plasma:
            if hematocrit is None:
                raise ValueError(
                    "hematocrit must be provided when convert_blood_to_plasma=True."
                )
            if not 0.0 <= float(hematocrit) < 1.0:
                raise ValueError("hematocrit must be in the range [0, 1).")
            C_t = C_t / (1.0 - float(hematocrit))
    except Exception:
        return (np.nan, {}) if return_saturation_info else np.nan

    if return_saturation_info:
        total_count = int(np.size(C_t))
        info = {
            "saturation_detected": bool(saturation_detected),
            "saturation_mask": saturation_mask,
            "saturation_count": (
                _count_true(saturation_mask) if normalized_mode == "spgr" else 0
            ),
            "saturation_fraction": (
                float(_count_true(saturation_mask) / total_count)
                if normalized_mode == "spgr" and total_count > 0
                else 0.0
            ),
            "saturation_threshold": saturation_threshold,
            "saturation_behavior": normalized_saturation_behavior,
        }
        return C_t, info
    return C_t


def concentration_to_signal(C_t, S_0, T10_s, TR_s, flip_angle_deg, r1):
    """Convert concentration to signal with the spoiled gradient echo model.

    Parameters
    ----------
    C_t : numpy.ndarray or xarray.DataArray or scalar
        Contrast concentration in mM, with any broadcast-compatible shape.
    S_0 : numpy.ndarray or xarray.DataArray or float
        Precontrast baseline signal in arbitrary signal units.
    T10_s : numpy.ndarray or xarray.DataArray or float
        Precontrast longitudinal relaxation time in seconds.
    TR_s : float
        Repetition time in seconds.
    flip_angle_deg : float
        Acquisition flip angle in degrees.
    r1 : float
        Contrast-agent longitudinal relaxivity in 1/(mM s).

    Returns
    -------
    numpy.ndarray or xarray.DataArray or scalar
        SPGR signal in baseline signal units, with the broadcast shape. Labelled
        arithmetic retains aligned dimensions, coordinates and direction; attrs
        follow xarray's active policy. Precision follows input NumPy/xarray
        arithmetic, without explicit casts. Deterministic and eager, with no
        input mutation, cache, clipping or saturation handling.

    Raises
    ------
    TypeError, ValueError
        If inputs do not support the arithmetic or cannot be broadcast/aligned.
        Singular physical parameters retain NumPy's nonfinite/warning behavior.

    Notes
    -----
    This is the inverse of the SPGR branch of :func:`signal_to_concentration`
    only when saturation handling does not alter the recovered signal. It does
    not invert the linear enhancement modes or plasma correction.

    Examples
    --------
    >>> baseline = concentration_to_signal(0.0, 100.0, 1.3, 0.004, 15.0, 3.5)
    >>> np.testing.assert_allclose(baseline, 100.0, rtol=1e-12)
    """
    theta = np.deg2rad(flip_angle_deg)
    R10 = 1.0 / T10_s
    R1_t = R10 + r1 * C_t
    E1_t = np.exp(-TR_s * R1_t)
    E10 = np.exp(-TR_s * R10)

    # Calculate M0 from baseline signal S_0 and T10
    M0 = (S_0 * (1 - np.cos(theta) * E10)) / (np.sin(theta) * (1 - E10))

    # Calculate signal S_t from concentration-dependent E1_t
    S_t = M0 * np.sin(theta) * (1 - E1_t) / (1 - np.cos(theta) * E1_t)
    return S_t
