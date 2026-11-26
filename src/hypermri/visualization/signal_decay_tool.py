"""Settings, decay curves and confirmation of the browser "signal_decay" tool.

The tool of :class:`~hypermri.visualization.scan_view.ScanView` for CSI raw
data, opened by :func:`~hypermri.visualization.signal_decay.plot_signal_decay`.
It shows how the signal decays over the acquired k-space points (in the
order of acquisition, not FID time) and lets the user choose a time-domain
filter for the acquired points and a spatial k-space filter:

* The decay curve is the mean magnitude of each acquired FID, or its band
  power (:func:`~hypermri.analysis.spectroscopy.signal_decay.spectral_band_power`)
  inside chosen frequency bands, optionally as a normalized cumulative sum.
  The curves of the unfiltered and the filtered data are shown together
  with the straight line fitted to the unfiltered curve from the signal end
  on ("tail fit") and the suggested signal end, where the curve of the
  unfiltered data reaches the noise
  (:func:`~hypermri.analysis.spectroscopy.suggest_signal_end`).
* The filter settings are those of
  :meth:`~hypermri.sequences.base_spectroscopy.BaseSpectroscopy.filter_signal`
  along the acquired points (``filter_axis=1``). The view can show the data
  reconstructed with the filter or without it.
* "Confirm" writes ``signal_end`` and ``filter_params`` to the scan. The
  acquired data are never changed.

This module holds no widget code. :class:`SignalDecayState` keeps the
choices; the functions read and change it and compute what the browser
shows. Filtering and reconstruction stay with the prepared data of
:mod:`hypermri.visualization.signal_decay`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hypermri.analysis.spatial_filtering import SpatialFilterOptions
from hypermri.analysis.spectroscopy.signal_decay import (
    _band_limits,
    signal_decay_curves,
)

if TYPE_CHECKING:
    from hypermri.visualization.signal_decay import _SignalDecayData

__all__ = [
    "DECAY_CURVES",
    "FILTER_OPTIONS",
    "FILTER_TYPES",
    "SPATIAL_FILTERS",
    "SignalDecayState",
    "confirm_signal_decay",
    "new_signal_decay_state",
    "set_signal_decay_option",
    "signal_decay_filter_params",
    "signal_decay_settings",
]

#: Time-domain filters along the acquired points, as named by ``filter_signal``.
FILTER_TYPES = ("None", "Exponential", "Hamming", "Hann")

#: Spatial k-space filters: none, Gaussian or Fermi (widths in cycles/mm).
SPATIAL_FILTERS = ("none", "gaussian", "fermi")

#: Decay curve name -> (per-point quantity, shown curve) of
#: :func:`~hypermri.analysis.spectroscopy.signal_decay.signal_decay_curves`.
DECAY_CURVES = {
    "magnitude": ("magnitude", "magnitude"),
    "cumulative": ("magnitude", "cumulative"),
    "band_power": ("band_power", "band_power"),
    "cumulative_band_power": ("band_power", "cumulative"),
}

# Filter settings that ``filter_signal`` reads for each filter type.
_FILTER_FIELDS = {
    "None": ("signal_end",),
    "Exponential": ("decay_constant",),
    "Hamming": ("alpha", "center_position"),
    "Hann": ("center_position",),
}

#: Settings that change the filtered data; the others change only the curves.
FILTER_OPTIONS = frozenset(
    {
        "filter_type",
        "signal_end",
        "decay_constant",
        "alpha",
        "center_position",
        "remove_lowest_signal",
        "apply_flipangle_correction",
        "flipangle_deg",
        "spatial_filter",
        "sigma_per_mm",
        "radius_per_mm",
        "transition_per_mm",
    }
)


@dataclass
class SignalDecayState:
    """Choices of the signal-decay tool in one view.

    The view changes the fields only through
    :func:`set_signal_decay_option`, which checks every value.

    Attributes
    ----------
    n_points : int
        Number of acquired k-space points per repetition; the valid range of
        ``signal_end`` and ``center_position`` is ``0 ... n_points - 1``.
    filter_type : str
        Time-domain filter along the acquired points, one of
        :data:`FILTER_TYPES`.
    signal_end : int
        Number of kept acquired points for ``filter_type="None"`` and start
        of the tail fit of the decay curves. "Confirm" writes it to the scan
        for every filter type.
    decay_constant : float
        Exponential attenuation per acquired point (dimensionless, not Hz).
    alpha : float
        Hamming coefficient, normally in [0, 1].
    center_position : int
        Centre index of the Hamming or Hann window in acquired points.
    remove_lowest_signal : bool
        Passed to ``filter_signal``.
    apply_flipangle_correction : bool
        Correct the signal loss of repeated excitations with ``flipangle_deg``.
    flipangle_deg : float
        Flip angle in degrees for the correction, in [0, 90].
    spatial_filter : str
        Spatial k-space filter, one of :data:`SPATIAL_FILTERS`.
    sigma_per_mm : float
        Width of the Gaussian filter in cycles/mm, positive.
    radius_per_mm, transition_per_mm : float
        Radius and edge width of the Fermi filter in cycles/mm, positive.
    curve : str
        Shown decay curve, a key of :data:`DECAY_CURVES`.
    bands_hz : str
        Frequency bands of the band power as ``low:high`` ranges in Hz
        separated by commas, for example ``"-330:-150, 200:420"``; empty
        selects the whole spectrum.
    show_tail : bool
        Draw the tail fit of the unfiltered curve.
    subtract_tail : bool
        Subtract the tail fit of the unfiltered curve from both curves (for
        display only).
    show_filtered : bool
        Show the data reconstructed with the filter in the view, instead of
        the unfiltered data.
    message : str
        Last result or error shown below the inputs.
    """

    n_points: int
    filter_type: str = "None"
    signal_end: int = 0
    decay_constant: float = 0.01
    alpha: float = 0.54
    center_position: int = 0
    remove_lowest_signal: bool = False
    apply_flipangle_correction: bool = False
    flipangle_deg: float = 90.0
    spatial_filter: str = "none"
    sigma_per_mm: float = 0.1
    radius_per_mm: float = 0.1
    transition_per_mm: float = 0.02
    curve: str = "magnitude"
    bands_hz: str = ""
    show_tail: bool = True
    subtract_tail: bool = False
    show_filtered: bool = True
    message: str = ""


def new_signal_decay_state(
    prepared: _SignalDecayData,
    bands_hz: Sequence[tuple[float, float]] | None = None,
) -> SignalDecayState:
    """Return the starting choices of the tool for prepared CSI data.

    Parameters
    ----------
    prepared : _SignalDecayData
        Sorted raw data of the scan from
        :mod:`hypermri.visualization.signal_decay`.
    bands_hz : sequence of (float, float) or None, optional
        Frequency bands in Hz. When given, the tool starts with the band
        power curve of these bands; otherwise with the mean magnitude.

    Returns
    -------
    SignalDecayState
        Signal end and window centre at half the acquired points (as in the
        earlier viewer), the flip angle of the scan (90 degrees when it has
        none) and no filter.

    Raises
    ------
    ValueError
        If ``bands_hz`` is not a list of increasing (low, high) pairs.
    """
    n = int(prepared.n_points)
    state = SignalDecayState(
        n_points=n,
        signal_end=n // 2,
        center_position=n // 2,
        flipangle_deg=float(getattr(prepared.sequence, "flipangle_deg", 90.0)),
    )
    bands = _band_limits(bands_hz)
    if bands is not None:
        state.curve = "band_power"
        state.bands_hz = _format_bands_hz(bands)
    return state


def set_signal_decay_option(state: SignalDecayState, name: str, value: Any) -> None:
    """Check and set one choice of the tool.

    Parameters
    ----------
    state : SignalDecayState
        Choices of the view; changed in place.
    name : str
        Field name of :class:`SignalDecayState` except ``n_points`` and
        ``message``.
    value : Any
        New value as sent by the browser: a number, bool or string.

    Raises
    ------
    ValueError
        If the name is unknown or the value is outside its range: an index
        outside ``0 ... n_points - 1``, a negative decay constant, a flip
        angle outside [0, 90] degrees, a filter width that is not positive,
        an unknown filter or curve name, or bands that are not ``low:high``
        ranges. The state is unchanged then.
    """
    if name in ("signal_end", "center_position"):
        number = _number(name, value)
        if number != int(number) or not 0 <= number < state.n_points:
            raise ValueError(
                f"Signal decay: {name} must be a whole number from 0 to "
                f"{state.n_points - 1} (acquired points); got {value!r}."
            )
        setattr(state, name, int(number))
    elif name in ("decay_constant", "alpha"):
        number = _number(name, value)
        if number < 0:
            raise ValueError(f"Signal decay: {name} must not be negative.")
        setattr(state, name, number)
    elif name == "flipangle_deg":
        number = _number(name, value)
        if not 0 <= number <= 90:
            raise ValueError("Signal decay: the flip angle must be 0 to 90 degrees.")
        state.flipangle_deg = number
    elif name in ("sigma_per_mm", "radius_per_mm", "transition_per_mm"):
        number = _number(name, value)
        if number <= 0:
            raise ValueError(
                f"Signal decay: {name} must be a positive width in cycles/mm."
            )
        setattr(state, name, number)
    elif name == "filter_type":
        state.filter_type = _choice(name, value, FILTER_TYPES)
    elif name == "spatial_filter":
        state.spatial_filter = _choice(name, value, SPATIAL_FILTERS)
    elif name == "curve":
        state.curve = _choice(name, value, tuple(DECAY_CURVES))
    elif name == "bands_hz":
        text = str(value)
        _parse_bands_hz(text)
        state.bands_hz = text.strip()
    elif name in (
        "remove_lowest_signal",
        "apply_flipangle_correction",
        "show_tail",
        "subtract_tail",
        "show_filtered",
    ):
        setattr(state, name, bool(value))
    else:
        raise ValueError(f"Signal decay: unknown setting {name!r}.")


def signal_decay_filter_params(
    state: SignalDecayState,
    spatial_spacing_mm: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return the keyword arguments of ``filter_signal`` for the chosen filter.

    Parameters
    ----------
    state : SignalDecayState
        Choices of the view.
    spatial_spacing_mm : tuple of float or None, optional
        Read and phase voxel spacing in mm for the spatial filter, when the
        data have no calibrated axes; None lets ``filter_signal`` use the
        field of view of the scan.

    Returns
    -------
    dict
        ``filter_type``, ``remove_lowest_signal``,
        ``apply_flipangle_correction``, ``filter_axis=1``, ``spatial_filter``
        (a :class:`~hypermri.analysis.spatial_filtering.SpatialFilterOptions`
        or None) and the settings the chosen filter reads: ``signal_end``
        for "None", ``decay_constant`` for "Exponential", ``alpha`` and
        ``center_position`` for "Hamming", ``center_position`` for "Hann",
        plus ``flipangle_deg`` with the flip-angle correction and
        ``spatial_spacing_mm`` with a spatial filter and given spacing.
        This is what "Confirm" stores as ``scan.filter_params``.
    """
    params: dict[str, Any] = {
        "filter_type": state.filter_type,
        "remove_lowest_signal": state.remove_lowest_signal,
        "apply_flipangle_correction": state.apply_flipangle_correction,
        "filter_axis": 1,
        "spatial_filter": _spatial_filter(state),
    }
    if params["spatial_filter"] is not None and spatial_spacing_mm is not None:
        params["spatial_spacing_mm"] = spatial_spacing_mm
    if state.apply_flipangle_correction:
        params["flipangle_deg"] = state.flipangle_deg
    for name in _FILTER_FIELDS[state.filter_type]:
        params[name] = getattr(state, name)
    return params


def signal_decay_settings(
    state: SignalDecayState,
    prepared: _SignalDecayData,
    repetition_index: int,
    *,
    filter_error: str | None = None,
) -> dict[str, Any]:
    """Return choices and decay curves of the tool for the browser.

    Parameters
    ----------
    state : SignalDecayState
        Choices of the view.
    prepared : _SignalDecayData
        Unfiltered and filtered data in acquisition order.
    repetition_index : int
        Zero-based repetition whose acquired points are shown.
    filter_error : str or None, optional
        Why the last filter could not be applied. The filtered curve is left
        out and "Confirm" is disabled while it is set.

    Returns
    -------
    dict
        The fields of :class:`SignalDecayState`, ``"offered": True``,
        ``"can_confirm"``, ``"filter_types"``, ``"y_label"``,
        ``"suggested_end"`` (an acquired point or None), ``"curves"`` as a
        list of ``{"name", "y"}`` over the acquired points ``0 ...`` (named
        "Original", "Filtered" and "Tail fit"; non-finite values are None)
        and ``"curve_error"`` when the curves cannot be computed, for
        example for band power without dwell time. The message names the
        suggestion, or why there is none, unless another message is set.
    """
    settings: dict[str, Any] = {
        **{
            name: getattr(state, name) for name in SignalDecayState.__dataclass_fields__
        },
        "offered": True,
        "can_confirm": filter_error is None,
        "filter_types": list(FILTER_TYPES),
        "curves": [],
        "suggested_end": None,
        "curve_error": "",
    }
    quantity, shown = DECAY_CURVES[state.curve]
    unit = prepared.original.attrs.get("units", "a.u.")
    settings["y_label"] = {
        "magnitude": f"mean magnitude [{unit}]",
        "band_power": f"mean band power [({unit})²]",
        "cumulative": "normalized cumulative "
        + ("magnitude" if quantity == "magnitude" else "band power"),
    }[shown]
    try:
        bands = _parse_bands_hz(state.bands_hz)
        options: dict[str, Any] = {
            "n_repetitions": prepared.n_repetitions,
            "repetition_index": int(repetition_index),
            "end_index": state.signal_end,
            "quantity": quantity,
        }
        if quantity == "band_power":
            options["dwell_time_s"] = prepared.band_power_dwell_time_s()
            options["bands_hz"] = bands
        original = signal_decay_curves(prepared.original_matrix, **options)
        tail = original[shown + "_tail"]
        curves = [("Original", original[shown])]
        if filter_error is None:
            filtered = signal_decay_curves(prepared.filtered_matrix, **options)
            curves.append(("Filtered", filtered[shown]))
        if state.subtract_tail and tail.size:
            curves = [(name, values - tail) for name, values in curves]
            tail = np.zeros_like(tail)
        if state.show_tail and tail.size:
            curves.append(("Tail fit", tail))
    except ValueError as error:
        settings["curve_error"] = f"Decay curve: {error}"
        return settings
    settings["curves"] = [
        {"name": name, "y": _json_values(values)} for name, values in curves
    ]
    try:
        settings["suggested_end"] = int(
            prepared.suggested_signal_end(int(repetition_index), bands)
        )
    except ValueError as error:
        note = f"No signal_end suggestion: {error}"
    else:
        note = (
            f"Suggested signal_end {settings['suggested_end']}: noise onset of "
            "the unfiltered acquisition."
        )
    if not state.message:
        settings["message"] = note
    return settings


def confirm_signal_decay(state: SignalDecayState, prepared: _SignalDecayData) -> str:
    """Write the chosen signal end and filter to the scan.

    Parameters
    ----------
    state : SignalDecayState
        Choices of the view.
    prepared : _SignalDecayData
        Prepared data whose ``sequence`` receives the choice.

    Returns
    -------
    str
        Message for the tool.

    Notes
    -----
    Sets ``sequence.signal_end`` to ``state.signal_end`` and
    ``sequence.filter_params`` to :func:`signal_decay_filter_params`. The
    acquired data and reconstructions stored on the scan are not changed;
    ``scan.filter_signal(**scan.filter_params)`` applies the filter later.
    """
    sequence = prepared.sequence
    sequence.signal_end = state.signal_end
    sequence.filter_params = signal_decay_filter_params(
        state, prepared.spatial_spacing_mm
    )
    return (
        f"Confirmed: signal_end {state.signal_end} and the {state.filter_type} "
        "filter are stored on the scan (signal_end, filter_params)."
    )


def _spatial_filter(state: SignalDecayState) -> SpatialFilterOptions | None:
    """Return the chosen spatial k-space filter, or None."""
    if state.spatial_filter == "gaussian":
        return SpatialFilterOptions(sigma_per_mm=state.sigma_per_mm)
    if state.spatial_filter == "fermi":
        return SpatialFilterOptions(
            kind="fermi",
            radius_per_mm=state.radius_per_mm,
            transition_per_mm=state.transition_per_mm,
        )
    return None


def _number(name: str, value: Any) -> float:
    """Return a finite float or raise a ValueError naming the setting."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if not np.isfinite(number):
        raise ValueError(f"Signal decay: {name} needs a number; got {value!r}.")
    return number


def _choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    """Return ``value`` if it is one of ``choices``, else raise a ValueError."""
    if value not in choices:
        raise ValueError(
            f"Signal decay: {name} must be one of {list(choices)}; got {value!r}."
        )
    return str(value)


def _json_values(values: np.ndarray) -> list[float | None]:
    """Return floats for JSON, with None for values that are not finite."""
    return [
        float(value) if np.isfinite(value) else None
        for value in np.asarray(values, dtype=float)
    ]


def _format_bands_hz(bands_hz: Sequence[tuple[float, float]]) -> str:
    """Write (low, high) Hz pairs in the ``low:high, low:high`` text syntax."""
    return ", ".join(f"{low:g}:{high:g}" for low, high in bands_hz)


def _parse_bands_hz(text: str) -> tuple[tuple[float, float], ...] | None:
    """Read ``low:high`` Hz ranges separated by commas; empty text is None."""
    if not text.strip():
        return None
    bands = []
    for part in text.split(","):
        limits = part.split(":")
        try:
            if len(limits) != 2:
                raise ValueError
            bands.append(tuple(float(value) for value in limits))
        except ValueError:
            raise ValueError(
                f"Signal decay: band {part.strip()!r} is not low:high; write "
                "ranges such as -330:-150, 200:420, or leave the field empty "
                "for the whole spectrum."
            ) from None
    return _band_limits(bands)
