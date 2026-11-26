"""Calibrated pH conversion from spectroscopy shifts and frequency differences.

The established Henderson-Hasselbalch equations, endpoint calibrations and
13C reference-frequency selection are preserved. Conversion is eager and pure:
no data reads, fitting, plotting, caching or mutation of acquired values occurs.
The historical LAMOR_FREQUENCY_* constant names remain spelled as published.
"""

from typing import Tuple, Union

import numpy as np
import xarray as xr

from hypermri.analysis.spectroscopy.frequency import freq_hz_to_ppm, freq_ppm_to_hz
from hypermri.utils.physics import get_larmor_freq

# Larmor Frequencies (MHz) at 7 Tesla from here:
# https://kodu.ut.ee/~laurit/AK2/NMR_tables_Bruker2012.pdf
LAMOR_FREQUENCY_H1 = 300.130
LAMOR_FREQUENCY_C13 = 75.468
LAMOR_FREQUENCY_P31 = 121.495

# Position of the 13C urea resonance (ppm) that the calibrations below refer to.
UREA_PPM = 163.5

# [pK, δHA, δA = δHA - δ_range] - Reference: Urea
PARAMS_OMPD_C1 = 6.5450, 8.4919, 8.4919 - 0.4644

# [pK, δHA, δA = δHA + δ_range] - Reference: Urea
PARAMS_OMPD_C5 = 6.5450, 12.4801, 12.4801 + 3.2975

# [pK, δHA, δA]
PARAMS_PHOSPHOR = [6.77, 3.23, 5.70]

# [pK, δHA, δA] # Reference Urea?
PARAMS_DIMEZA = [7.3, 172.5, 177.71]

# Z-OMPD C5 minus C1 shift (no urea reference needed). The urea reference cancels
# in the difference, so the logistic calibrations of C5 and C1 combine to
# (pK, 3.9882 ppm at low pH, 7.7501 ppm at high pH).
PARAMS_OMPD_C5_MINUS_C1 = (
    PARAMS_OMPD_C1[0],
    PARAMS_OMPD_C5[1] - PARAMS_OMPD_C1[1],
    PARAMS_OMPD_C5[2] - PARAMS_OMPD_C1[2],
)

#
PARAMS_DIMEZA_NO_UREA = [7.3, 172.5 - 163.5, 177.71 - 163.5]


def pH_to_ppm(ph_value, params):
    """Convert pH to a chemical-shift difference using calibrated endpoints.

    Parameters
    ----------
    ph_value : float or array_like
        Dimensionless pH values. NumPy arrays and labelled xarray arrays retain
        their shape; arithmetic follows the input's ordinary dtype rules.
    params : sequence of three float
        Calibration (pKa, protonated_shift_ppm, deprotonated_shift_ppm), using the
        same chemical-shift reference for both endpoints.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Chemical-shift difference in ppm, with input shape and labelled coordinates
        when present. Inputs and calibration values are not modified.

    Raises
    ------
    ValueError
        If params cannot be unpacked into three calibration values.

    Examples
    --------
    >>> pH_to_ppm(6.545, PARAMS_OMPD_C5)
    14.12885
    """
    pK, dHA, dA = params
    return dHA + (dA - dHA) / (1 + 10 ** (-(ph_value - pK)))


def ppm_to_pH(
    freq_diff_ppm: Union[float, np.ndarray] = None,
    params: Tuple[float, float, float] = None,
) -> Union[float, np.ndarray]:
    """Convert a chemical-shift difference in ppm to calibrated pH.

    Parameters
    ----------
    freq_diff_ppm : float or numpy.ndarray
        Shift relative to the calibration reference, between the two endpoint
        shifts. Labelled xarray input is also supported by existing arithmetic.
    params : tuple of float
        Calibration (pKa, protonated_shift_ppm, deprotonated_shift_ppm). Both
        arguments must be supplied despite their historical None defaults.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Dimensionless pH values with the input shape and labelled coordinates.
        NumPy/xarray arithmetic determines floating precision and attribute
        propagation. No input is modified or cached.

    Raises
    ------
    TypeError
        If a required argument is None or cannot participate in the arithmetic.
    ValueError
        If params cannot be unpacked into three calibration values.

    Notes
    -----
    The original logarithm is retained without clipping or new endpoint validation.
    Endpoint or out-of-range shifts can return infinity or NaN and emit NumPy
    warnings; these values do not establish a valid pH calibration.

    Examples
    --------
    >>> shift = pH_to_ppm(7.0, PARAMS_OMPD_C5)
    >>> float(ppm_to_pH(shift, PARAMS_OMPD_C5))
    7.0
    """
    pKa, dHA_ppm, dA_ppm = params
    return pKa + np.log10((freq_diff_ppm - dHA_ppm) / (dA_ppm - freq_diff_ppm))


def _positive_scalar(value, name):
    """Validate a field strength or reference frequency with explicit units."""
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array) or array.dtype == bool:
        raise ValueError(f"{name} must be a finite positive scalar.")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar.") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return result


def _resolve_larmor_frequency_mhz(fieldstrength_t, fieldstrength_T, frequency_mhz):
    """Resolve one 13C reference, without guessing Tesla/MHz from its value."""
    field = None
    for value, name in (
        (fieldstrength_t, "fieldstrength_t"),
        (fieldstrength_T, "fieldstrength_T"),
    ):
        if value is not None:
            value = _positive_scalar(value, name)
            if field is not None and value != field:
                raise ValueError("Conflicting fieldstrength_t and fieldstrength_T.")
            field = value
    if frequency_mhz is not None:
        if field is not None:
            raise ValueError("Supply either a field strength or larmor_frequency_mhz.")
        return _positive_scalar(frequency_mhz, "larmor_frequency_mhz")
    return (
        get_larmor_freq(
            nucleus="13c", fieldstrength_T=7.0 if field is None else field, unit="Hz"
        )
        * 1e-6
    )


def freq_to_pH(
    frequency_hz,
    params,
    fieldstrength_t=None,
    *,
    larmor_frequency_mhz=None,
    fieldstrength_T=None,
):
    """Convert a frequency difference in Hz to calibrated pH.

    Parameters
    ----------
    frequency_hz : float or array_like
        Difference relative to the calibration reference, in Hz. Arrays retain
        their shape and labelled xarray dimensions/coordinates when present.
    params : sequence of three float
        Calibration (pKa, protonated_shift_ppm, deprotonated_shift_ppm).
    fieldstrength_t : float, optional
        Finite positive field strength in Tesla; the historical third positional
        argument. Defaults to 7 T when no reference option is supplied.
    larmor_frequency_mhz : float, optional
        Measured positive 13C reference frequency in MHz, mutually exclusive with
        both field-strength options. Hz divided by MHz gives ppm.
    fieldstrength_T : float, optional
        Historical field-strength alias in Tesla. If both aliases are supplied,
        their finite positive scalar values must agree.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Dimensionless pH, following ppm_to_pH's shape, labels, arithmetic precision
        and undefined-endpoint behavior. No input is changed or cached.

    Raises
    ------
    ValueError
        If reference options conflict, are nonpositive, nonfinite or nonscalar,
        or the calibration does not contain three values.
    TypeError
        If signal or calibration values cannot participate in the arithmetic.

    Examples
    --------
    >>> shift_hz = pH_to_freq(7.0, PARAMS_OMPD_C5, larmor_frequency_mhz=75.468)
    >>> float(freq_to_pH(shift_hz, PARAMS_OMPD_C5, larmor_frequency_mhz=75.468))
    7.0
    """
    reference_mhz = _resolve_larmor_frequency_mhz(
        fieldstrength_t, fieldstrength_T, larmor_frequency_mhz
    )
    return ppm_to_pH(frequency_hz / reference_mhz, params)


def pH_to_freq(
    ppm,
    params,
    fieldstrength_t=None,
    *,
    larmor_frequency_mhz=None,
    fieldstrength_T=None,
):
    """Convert calibrated pH to a frequency difference in Hz.

    Parameters
    ----------
    ppm : float or array_like
        Dimensionless pH values; ppm is the retained historical argument name.
        NumPy and labelled xarray inputs retain their shape and coordinates.
    params : sequence of three float
        Calibration (pKa, protonated_shift_ppm, deprotonated_shift_ppm).
    fieldstrength_t : float, optional
        Positive scalar field strength in Tesla, retaining the third positional
        argument. Defaults to 7 T when all reference options are absent.
    larmor_frequency_mhz : float, optional
        Positive measured 13C reference in MHz, mutually exclusive with both
        field-strength aliases. Output is shift_ppm multiplied by reference_MHz.
    fieldstrength_T : float, optional
        Historical Tesla alias; if both field-strength aliases are present,
        their positive finite scalar values must agree.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Frequency difference in Hz, using ordinary NumPy/xarray precision and
        attribute propagation. Coordinates/dimensions are preserved; inputs are
        neither changed nor cached.

    Raises
    ------
    ValueError
        If reference options conflict, are nonpositive, nonfinite or nonscalar,
        or params does not contain three calibration values.
    TypeError
        If pH or calibration values cannot participate in the arithmetic.

    Examples
    --------
    >>> pH_to_freq(6.545, PARAMS_OMPD_C5, larmor_frequency_mhz=100.0)
    1412.885
    """
    reference_mhz = _resolve_larmor_frequency_mhz(
        fieldstrength_t, fieldstrength_T, larmor_frequency_mhz
    )
    return pH_to_ppm(ppm, params) * reference_mhz


def OMPD5_freq2pH(
    frequency_hz,
    params=PARAMS_OMPD_C5,
    fieldstrength_t=None,
    *,
    larmor_frequency_mhz=None,
    fieldstrength_T=None,
):
    """Convert an OMPD C5 frequency difference in Hz to calibrated pH.

    Parameters
    ----------
    frequency_hz : float or array_like
        Difference from the calibration reference in Hz, including labelled
        xarray arrays with arbitrary input dimensions.
    params : sequence of three float, optional
        Calibration (pKa, protonated_shift_ppm, deprotonated_shift_ppm).
        Defaults to the established PARAMS_OMPD_C5 values.
    fieldstrength_t : float, optional
        Positive field strength in Tesla; defaults to 7 T when no reference is
        supplied. Retains its historical positional interpretation.
    larmor_frequency_mhz : float, optional
        Measured positive 13C reference frequency in MHz. Cannot be combined with
        either field-strength alias.
    fieldstrength_T : float, optional
        Historical Tesla alias. If both field-strength aliases are specified,
        their positive finite scalar values must agree.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Dimensionless pH from freq_to_pH with its existing shape, label, precision
        and undefined-endpoint behavior. Does not read, mutate or cache data.

    Raises
    ------
    ValueError
        If the reference options are invalid/conflicting or calibration length
        is not three.
    TypeError
        If frequency or calibration values cannot participate in the arithmetic.

    Examples
    --------
    >>> hz = pH_to_freq(7.0, PARAMS_OMPD_C5, larmor_frequency_mhz=75.468)
    >>> float(OMPD5_freq2pH(hz, larmor_frequency_mhz=75.468))
    7.0
    """
    return freq_to_pH(
        frequency_hz=frequency_hz,
        params=params,
        fieldstrength_t=fieldstrength_t,
        larmor_frequency_mhz=larmor_frequency_mhz,
        fieldstrength_T=fieldstrength_T,
    )


def OMPD_pH_from_c1_c5(
    c1_position,
    c5_position,
    *,
    unit="ppm",
    fieldstrength_t=None,
    larmor_frequency_mhz=None,
    fieldstrength_T=None,
):
    """Determine pH from the C1 and C5 peak positions of [1,5-13C2]Z-OMPD.

    The C1 resonance serves as internal frequency reference (Grashei et al.,
    Z-OMPD pH sensor): the C5 - C1 distance follows the scaled logistic
    calibration with pKa = 6.545 and needs no co-injected urea. Any common
    offset of the frequency axis (for example an uncalibrated ppm axis)
    cancels in the difference.

    Parameters
    ----------
    c1_position : float or array_like
        Peak position of the C1 resonance in the unit given by ``unit``.
        Scalars, NumPy arrays of any shape (e.g. 2D voxel maps) and labelled
        xarray arrays are supported.
    c5_position : float or array_like
        Peak position of the C5 resonance, same unit and a shape that
        broadcasts against ``c1_position``. C5 lies at higher frequency than C1.
    unit : {"ppm", "hz"}, optional
        Unit of both positions. ``"hz"`` requires the 13C reference frequency
        via ``fieldstrength_t`` or ``larmor_frequency_mhz`` (7 T if none is
        given); ``"ppm"`` ignores them.
    fieldstrength_t : float, optional
        Field strength in Tesla, only used for ``unit="hz"``.
    larmor_frequency_mhz : float, optional
        Measured 13C reference frequency in MHz, only used for ``unit="hz"``;
        exclusive with the field-strength arguments.
    fieldstrength_T : float, optional
        Historical alias of ``fieldstrength_t``.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Dimensionless pH with the broadcast shape of the inputs (labels are
        kept for xarray input). Distances outside the calibrated open range
        3.9882 to 7.7501 ppm give NaN (or inf at the exact endpoints); no input
        is modified.

    Raises
    ------
    ValueError
        If ``unit`` is not ``"ppm"`` or ``"hz"``, or the Hz reference options
        are invalid or conflicting.

    Examples
    --------
    >>> float(OMPD_pH_from_c1_c5(170.0, 170.0 + 5.8))  # doctest: +ELLIPSIS
    6.5...
    """
    unit_key = str(unit).lower()
    if unit_key not in ("ppm", "hz"):
        raise ValueError(f"unit must be 'ppm' or 'hz', got {unit!r}.")
    distance = c5_position - c1_position
    with np.errstate(divide="ignore", invalid="ignore"):
        if unit_key == "ppm":
            return ppm_to_pH(distance, PARAMS_OMPD_C5_MINUS_C1)
        return freq_to_pH(
            distance,
            PARAMS_OMPD_C5_MINUS_C1,
            fieldstrength_t,
            larmor_frequency_mhz=larmor_frequency_mhz,
            fieldstrength_T=fieldstrength_T,
        )


def OMPD_pH_from_peak_shift(
    position_reference,
    position,
    reference_ph,
    *,
    peak="C1",
    unit="ppm",
    scan=None,
    larmor_frequency_mhz=None,
):
    """Determine the pH of a Z-OMPD peak map from its shift against a reference map.

    The reference map is the same resonance (C1 or C5) measured at the known
    pH ``reference_ph``. The shift ``position - position_reference`` is added to
    the calibrated reference position ``delta(reference_ph)`` and the result is
    inverted with the logistic calibration of that peak. A common offset of the
    frequency axis cancels. The calibration is relative to urea, so absolute
    positions are not needed.

    Parameters
    ----------
    position_reference : float or array_like
        Peak positions at ``reference_ph`` in the unit given by ``unit``
        (ppm, or Hz relative to the scan centre). Scalars, NumPy arrays of any
        shape (e.g. 2D voxel maps) and labelled xarray arrays are supported.
    position : float or array_like
        Peak positions whose pH is wanted, same unit and a shape that
        broadcasts against ``position_reference``.
    reference_ph : float or array_like
        Dimensionless pH at which ``position_reference`` was measured. Must
        broadcast against the maps.
    peak : {"C1", "C5"}, optional
        Resonance used for the calibration. C1 shifts by only 0.46 ppm over
        the whole pH range, C5 by 3.30 ppm, so C1 is far more noise sensitive.
    unit : {"ppm", "hz"}, optional
        Unit of both position maps.
    scan : object or None, optional
        Scan context for the Hz to ppm conversion (Hz per ppm from
        :func:`freq_ppm_to_hz`). Used only for ``unit="hz"``.
    larmor_frequency_mhz : float, optional
        13C reference frequency in MHz, used for ``unit="hz"`` when
        ``scan`` is None; 7 T if neither is given.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Dimensionless pH of ``position`` with the broadcast shape of the
        inputs (labels kept for xarray input). Shifts that leave the calibrated
        open range of the peak give NaN; no input is modified.

    Raises
    ------
    ValueError
        If ``peak`` or ``unit`` is unknown, or ``unit="hz"`` is combined with
        both ``scan`` and ``larmor_frequency_mhz``.

    Examples
    --------
    >>> ref = pH_to_ppm(7.4, PARAMS_OMPD_C5)
    >>> float(OMPD_pH_from_peak_shift(ref, ref - 1.0, 7.4, peak="C5"))  # doctest: +ELLIPSIS
    6.67...
    """
    peaks = {"C1": PARAMS_OMPD_C1, "C5": PARAMS_OMPD_C5}
    if peak not in peaks:
        raise ValueError(f"peak must be 'C1' or 'C5', got {peak!r}.")
    unit_key = str(unit).lower()
    if unit_key not in ("ppm", "hz"):
        raise ValueError(f"unit must be 'ppm' or 'hz', got {unit!r}.")
    params = peaks[peak]
    shift = position - position_reference
    if unit_key == "hz":
        if scan is not None and larmor_frequency_mhz is not None:
            raise ValueError("Supply either scan or larmor_frequency_mhz.")
        if scan is not None:
            hz = freq_ppm_to_hz([0.0, 1.0], scan=scan)
            hz_per_ppm = float(hz[1] - hz[0])
        else:
            hz_per_ppm = _resolve_larmor_frequency_mhz(None, None, larmor_frequency_mhz)
        shift = shift / hz_per_ppm
    with np.errstate(divide="ignore", invalid="ignore"):
        return ppm_to_pH(pH_to_ppm(reference_ph, params) + shift, params)


def OMPD_pH_from_absolute_position(
    position,
    *,
    peak="C1",
    unit="ppm",
    urea_ppm=UREA_PPM,
    scan=None,
):
    """Determine pH from the absolute position of one Z-OMPD peak.

    The calibrations are relative to urea, so the position on the Urea scale is
    ``position_ppm - urea_ppm`` and is inverted with the logistic calibration of
    the chosen peak. This is only as accurate as the frequency axis: an axis
    error of 0.1 ppm gives about 0.2 to 0.3 pH units for C1 and a smaller error
    for C5. The C5 - C1 distance (:func:`OMPD_pH_from_c1_c5`) is independent of
    the axis offset.

    Parameters
    ----------
    position : float or array_like
        Absolute peak position in ``unit``. Scalars, NumPy arrays of any shape
        and labelled xarray arrays are supported.
    peak : {"C1", "C5"}, optional
        Resonance whose calibration is used.
    unit : {"ppm", "hz"}, optional
        ``"ppm"`` is the absolute chemical-shift axis. ``"hz"`` is the frequency
        relative to the scan centre and requires ``scan`` (absolute
        conversion with :func:`freq_hz_to_ppm`, including ``center_ppm``).
    urea_ppm : float, optional
        Position of urea on the same ppm axis; 163.5 ppm by default
        (:data:`UREA_PPM`). Change it if the axis is calibrated differently.
    scan : object or None, optional
        Scan context for the Hz to ppm conversion; required for ``unit="hz"``.

    Returns
    -------
    float or numpy.ndarray or xarray.DataArray
        Dimensionless pH with the input shape (labels kept for xarray input).
        Positions outside the calibrated open range of the peak give NaN. No
        input is modified.

    Raises
    ------
    ValueError
        If ``peak`` or ``unit`` is unknown, or ``unit="hz"`` is used without
        ``scan``.

    Examples
    --------
    >>> float(OMPD_pH_from_absolute_position(163.5 + pH_to_ppm(7.0, PARAMS_OMPD_C5), peak="C5"))
    7.0
    """
    peaks = {"C1": PARAMS_OMPD_C1, "C5": PARAMS_OMPD_C5}
    if peak not in peaks:
        raise ValueError(f"peak must be 'C1' or 'C5', got {peak!r}.")
    unit_key = str(unit).lower()
    if unit_key not in ("ppm", "hz"):
        raise ValueError(f"unit must be 'ppm' or 'hz', got {unit!r}.")
    ppm = position
    if unit_key == "hz":
        if scan is None:
            raise ValueError(
                "OMPD_pH_from_absolute_position: unit='hz' needs scan for the "
                "absolute Hz to ppm conversion; pass scan or use unit='ppm'."
            )
        converted = freq_hz_to_ppm(np.asarray(position, dtype=float), scan=scan)
        ppm = (
            position.copy(data=converted)
            if isinstance(position, xr.DataArray)
            else converted
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return ppm_to_pH(ppm - urea_ppm, peaks[peak])
