"""State and steps of the "IDEAL" tool of the browser viewer.

IDEAL separates chemical components (for example pyruvate and lactate) in
complex multi-echo images. Each voxel signal along the echo axis is modelled
as a sum of components with known frequencies (in Hz, relative to the
scanner reference frequency) and one common field offset:

``S(t_n) = sum_m rho_m * exp(2 pi i (f_m + f_offset) t_n)``

The tool is offered when the curve axis is ``echo_idx``. The user enters the
component names and frequencies; the echo spacing comes from the echo axis
(coordinate in s or ms, equally spaced) or is entered in ms. The fit itself
is :func:`~hypermri.analysis.quantitative.fit_ideal`; this module only keeps
the settings, checks them and prepares what the browser shows. It builds no
widgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.quantitative import IDEALFitOptions, fit_ideal

__all__ = [
    "ECHO_DIM",
    "FIELD_OFFSET_VARIABLE",
    "IdealToolState",
    "echo_spacing_from_axis",
    "ideal_maps",
    "ideal_offered",
    "ideal_options",
    "ideal_settings",
    "ideal_summary",
    "run_ideal",
    "set_ideal_option",
]

#: Dimension along which IDEAL separates the components.
ECHO_DIM = "echo_idx"

#: Result variable of the field offset map in Hz.
FIELD_OFFSET_VARIABLE = "field_offset"

# Factor from a unit of the echo coordinate to seconds.
_SECONDS_PER_UNIT = {"s": 1.0, "ms": 1e-3}
# Relative tolerance for equally spaced echo times.
_SPACING_RTOL = 1e-6


@dataclass
class IdealToolState:
    """Settings and last note of the IDEAL tool.

    The object is changed in place by :func:`set_ideal_option` and by the
    viewer, which writes ``message``.

    Attributes
    ----------
    component_names : tuple of str
        Names of the components, for example ``("pyruvate", "lactate")``.
        They become the names of the amplitude maps.
    frequencies_hz : tuple of float
        Frequency of each component in Hz, relative to the scanner reference
        frequency, in the order of ``component_names``.
    echo_spacing_ms : float or None
        Time between two echoes in ms. None takes it from the echo axis
        (:func:`echo_spacing_from_axis`).
    fit_field_offset : bool
        Fit the common field offset (in Hz) of each voxel.
    fit_theta : bool
        Fit the alternating-echo modulation Theta (for example of bipolar
        read-out gradients). It is fitted but not returned as a map.
    message : str
        Last note or error shown below the tool.
    """

    component_names: tuple[str, ...] = ()
    frequencies_hz: tuple[float, ...] = ()
    echo_spacing_ms: float | None = None
    fit_field_offset: bool = True
    fit_theta: bool = True
    message: str = ""


def ideal_offered(data: xr.DataArray, curve_dim: str | None) -> bool:
    """Return whether the IDEAL tool applies to the shown data.

    Parameters
    ----------
    data : xarray.DataArray
        Data shown by the viewer. Not modified.
    curve_dim : str or None
        Dimension shown on the curve axis.

    Returns
    -------
    bool
        True when the curve axis is :data:`ECHO_DIM` and ``data`` has it.
    """
    return curve_dim == ECHO_DIM and ECHO_DIM in data.dims


def echo_spacing_from_axis(data: xr.DataArray) -> float | None:
    """Return the echo spacing in seconds read from the echo coordinate.

    Parameters
    ----------
    data : xarray.DataArray
        Data with dimension :data:`ECHO_DIM`. Not modified.

    Returns
    -------
    float or None
        The common step of the echo coordinate in seconds, when the
        coordinate is one-dimensional, has the ``unit`` ``"s"`` or ``"ms"``,
        at least two finite values and equal positive steps (relative
        tolerance 1e-6). None otherwise.
    """
    coord = data.coords.get(ECHO_DIM)
    if coord is None or coord.dims != (ECHO_DIM,) or coord.size < 2:
        return None
    factor = _SECONDS_PER_UNIT.get(coord.attrs.get("unit", coord.attrs.get("units")))
    if factor is None:
        return None
    steps = np.diff(np.asarray(coord.values, dtype=float)) * factor
    if not np.all(np.isfinite(steps)) or steps[0] <= 0:
        return None
    if not np.allclose(steps, steps[0], rtol=_SPACING_RTOL, atol=0.0):
        return None
    return float(steps[0])


def _split(text: Any) -> list[str]:
    """Split a comma-separated text into its stripped, non-empty entries."""
    if isinstance(text, (list, tuple)):
        return [str(entry).strip() for entry in text if str(entry).strip()]
    return [entry.strip() for entry in str(text or "").split(",") if entry.strip()]


def set_ideal_option(state: IdealToolState, name: str, value: Any) -> None:
    """Change one setting of the IDEAL tool.

    Parameters
    ----------
    state : IdealToolState
        Tool state; the attribute ``name`` is replaced.
    name : str
        ``"component_names"`` or ``"frequencies_hz"`` (comma-separated text
        or a sequence), ``"echo_spacing_ms"`` (a positive number, or None or
        an empty text for the echo axis), ``"fit_field_offset"`` or
        ``"fit_theta"`` (converted to bool).
    value : Any
        New value.

    Raises
    ------
    ValueError
        If the setting does not exist, a name repeats or is
        :data:`FIELD_OFFSET_VARIABLE`, a frequency is not a finite number or
        the echo spacing is not a positive number.
    """
    if name == "component_names":
        names = tuple(_split(value))
        if len(set(names)) != len(names) or FIELD_OFFSET_VARIABLE in names:
            raise ValueError(
                "IDEAL: component names must differ from each other and from "
                f"{FIELD_OFFSET_VARIABLE!r}."
            )
        state.component_names = names
    elif name == "frequencies_hz":
        try:
            frequencies = tuple(float(entry) for entry in _split(value))
        except ValueError:
            raise ValueError(
                "IDEAL: frequencies must be numbers in Hz, separated by commas, "
                "for example '0, 390'."
            ) from None
        if not all(np.isfinite(frequencies)):
            raise ValueError("IDEAL: frequencies must be finite numbers in Hz.")
        state.frequencies_hz = frequencies
    elif name == "echo_spacing_ms":
        if value is None or str(value).strip() == "":
            state.echo_spacing_ms = None
            return
        spacing = float(value)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError(
                "IDEAL: the echo spacing must be a positive number in ms; leave "
                "it empty to take it from the echo axis."
            )
        state.echo_spacing_ms = spacing
    elif name in ("fit_field_offset", "fit_theta"):
        setattr(state, name, bool(value))
    else:
        raise ValueError(
            f"IDEAL: unknown setting {name!r}; choose 'component_names', "
            "'frequencies_hz', 'echo_spacing_ms', 'fit_field_offset' or "
            "'fit_theta'."
        )


def ideal_options(
    data: xr.DataArray,
    state: IdealToolState,
    execution: ExecutionConfig | None = None,
) -> IDEALFitOptions:
    """Return the fit options of the IDEAL tool after checking them.

    Parameters
    ----------
    data : xarray.DataArray
        Complex echo data with dimension :data:`ECHO_DIM`. Not modified.
    state : IdealToolState
        Tool settings. Not modified.
    execution : ExecutionConfig or None, optional
        Parallel execution of the fit; None uses the solver default.

    Returns
    -------
    IDEALFitOptions
        Component names and frequencies in Hz, echo spacing in s (entered
        value, else the echo axis), the fitted unknowns as ``solve_for`` and
        ``execution``. Initial values, stop thresholds and the iteration
        limit stay at the solver defaults.

    Raises
    ------
    ValueError
        If the data lack the echo dimension or are not complex, no component
        is given, names and frequencies differ in number, there are fewer
        echoes than components, no echo spacing is known or neither field
        offset nor Theta is fitted.
    """
    if ECHO_DIM not in data.dims:
        raise ValueError(f"IDEAL: the data have no echo dimension {ECHO_DIM!r}.")
    if not np.iscomplexobj(data):
        raise ValueError(
            "IDEAL needs complex echo images (magnitude and phase); the shown "
            "data are real. Reconstruct the scan without taking the magnitude."
        )
    names, frequencies = state.component_names, state.frequencies_hz
    if not frequencies:
        raise ValueError(
            "IDEAL: enter the component frequencies in Hz, for example '0, 390'."
        )
    if names and len(names) != len(frequencies):
        raise ValueError(
            f"IDEAL: {len(names)} component names but {len(frequencies)} "
            "frequencies; give one name per frequency."
        )
    n_echoes = data.sizes[ECHO_DIM]
    if n_echoes < len(frequencies):
        raise ValueError(
            f"IDEAL: {len(frequencies)} components need at least as many echoes; "
            f"the data have {n_echoes}."
        )
    if state.echo_spacing_ms is not None:
        spacing_s = state.echo_spacing_ms * 1e-3
    else:
        spacing_s = echo_spacing_from_axis(data)
        if spacing_s is None:
            raise ValueError(
                "IDEAL: the echo axis gives no equal echo spacing (coordinate "
                f"{ECHO_DIM!r} in s or ms); enter the echo spacing in ms."
            )
    solve_for = tuple(
        unknown
        for unknown, chosen in (
            ("Field_offset", state.fit_field_offset),
            ("Theta", state.fit_theta),
        )
        if chosen
    )
    if not solve_for:
        raise ValueError("IDEAL: fit the field offset, Theta or both.")
    return IDEALFitOptions(
        relative_frequencies_hz=frequencies,
        echo_spacing_s=spacing_s,
        metabolite_names=names or None,
        echo_dim=ECHO_DIM,
        execution=execution,
        solve_for=solve_for,
    )


def run_ideal(
    data: xr.DataArray,
    state: IdealToolState,
    execution: ExecutionConfig | None = None,
) -> xr.Dataset:
    """Separate the components in every voxel with IDEAL.

    Parameters
    ----------
    data : xarray.DataArray
        Complex echo data with dimension :data:`ECHO_DIM`; the other
        non-singleton dimensions must be among ``read_idx``, ``phase_idx``,
        ``slice_idx`` and ``repetition_idx``. Not modified.
    state : IdealToolState
        Tool settings, see :func:`ideal_options`. Not modified.
    execution : ExecutionConfig or None, optional
        Parallel execution of the fit.

    Returns
    -------
    xarray.Dataset
        The result of :func:`~hypermri.analysis.quantitative.fit_ideal`:
        complex ``fit_amps`` (signal units, dimension ``metabolite`` last)
        and, when fitted, ``field_offset`` in Hz.

    Raises
    ------
    ValueError
        If :func:`ideal_options` or the fit rejects the data or settings.
    numpy.linalg.LinAlgError
        If the solver meets a singular design, for example components with
        equal frequencies.
    """
    return fit_ideal(data, ideal_options(data, state, execution))


def ideal_maps(result: xr.Dataset) -> xr.Dataset:
    """Return the IDEAL result as real maps for display.

    Parameters
    ----------
    result : xarray.Dataset
        Result of :func:`run_ideal`. Not modified.

    Returns
    -------
    xarray.Dataset
        One magnitude map per component, named after the component, in the
        signal units of the data, and ``field_offset`` in Hz when it was
        fitted. All maps have the voxel dimensions of ``result``; scalar
        leftover coordinates are dropped.
    """
    amplitudes = result["fit_amps"]
    maps = {
        str(name): np.abs(amplitudes.sel(metabolite=name, drop=True))
        for name in amplitudes.coords["metabolite"].values
    }
    if FIELD_OFFSET_VARIABLE in result:
        maps[FIELD_OFFSET_VARIABLE] = result[FIELD_OFFSET_VARIABLE]
    scalars = [name for name, coord in result.coords.items() if coord.ndim == 0]
    return xr.Dataset(maps).drop_vars(scalars, errors="ignore")


def ideal_summary(result: xr.Dataset) -> str:
    """Return a one-line note on a finished IDEAL fit.

    Parameters
    ----------
    result : xarray.Dataset
        Result of :func:`run_ideal`. Not modified.

    Returns
    -------
    str
        Components, number of voxels and the median field offset in Hz.
    """
    names = ", ".join(str(name) for name in result.coords["metabolite"].values)
    n_voxels = int(np.prod(result["fit_amps"].shape[:-1]))
    note = f"IDEAL: {names} in {n_voxels} voxels"
    if FIELD_OFFSET_VARIABLE in result:
        offset = np.asarray(result[FIELD_OFFSET_VARIABLE].values, dtype=float)
        if np.any(np.isfinite(offset)):
            note += f", median field offset {np.nanmedian(offset):.3g} Hz"
    return note + "."


def ideal_settings(
    state: IdealToolState, data: xr.DataArray, curve_dim: str | None
) -> dict[str, Any]:
    """Return the IDEAL tool state as sent to the browser.

    Parameters
    ----------
    state : IdealToolState
        Tool settings. Not modified.
    data : xarray.DataArray
        Data shown by the viewer. Not modified.
    curve_dim : str or None
        Dimension shown on the curve axis.

    Returns
    -------
    dict
        ``offered`` (:func:`ideal_offered`); when offered also the settings
        (names and frequencies as comma-separated text), ``axis_spacing_ms``
        (echo spacing of the axis in ms, or None), ``n_echoes`` and
        ``complex``. Always ``message``.
    """
    if not ideal_offered(data, curve_dim):
        return {"offered": False, "message": state.message}
    axis_spacing_s = echo_spacing_from_axis(data)
    return {
        "offered": True,
        "component_names": ", ".join(state.component_names),
        "frequencies_hz": ", ".join(f"{f:g}" for f in state.frequencies_hz),
        "echo_spacing_ms": state.echo_spacing_ms,
        "axis_spacing_ms": None if axis_spacing_s is None else axis_spacing_s * 1e3,
        "fit_field_offset": state.fit_field_offset,
        "fit_theta": state.fit_theta,
        "n_echoes": int(data.sizes[ECHO_DIM]),
        "complex": bool(np.iscomplexobj(data)),
        "message": state.message,
    }
