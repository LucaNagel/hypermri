"""Prepare labelled IDEAL echoes and call the existing complex128 solver.

Frequencies use Hz and echo spacing uses seconds. The solver uses centered,
equally spaced times and can optimize field offset, alternating-echo Theta, or
both. Supplied initialization and convergence settings are preserved; omitted
settings retain solver defaults. Quantitative APIs reject T2-star settings
because this solver implements no T2-star correction or optimization.
Preparation reads no signal, and calls mutate no acquired data or retain a cache.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.execution import copy_execution_params

MODEL_ID = "ideal"
RESULT_MODEL_NAME = "ideal"
MODEL_EQUATION = "S(t_n) = sum_m [ rho_m * exp(2*pi*i*(f_m + f_offset)*t_n) ]"
_ACTIVE_SOLVER_SETTINGS = (
    "solve_for",
    "init_f_offset_Hz",
    "init_Theta",
    "stop_f_offset_hz",
    "stop_theta",
    "max_iterations",
)


def _resolve_relative_frequencies(fit_params):
    """Accept the public Hz spellings and reject conflicting calibrations."""
    canonical = fit_params.get("relative_frequencies_hz")
    legacy = fit_params.get("relative_frequencies_Hz")
    if canonical is not None and legacy is not None:
        if not np.array_equal(np.asarray(canonical), np.asarray(legacy)):
            raise ValueError(
                "Conflicting relative_frequencies_hz and relative_frequencies_Hz."
            )
    return canonical if canonical is not None else legacy


def _validate_ideal_solver_options(fit_params):
    """Reject unsupported physics and settings that cannot define a bounded solve."""
    for name in ("init_T2_star_s", "stop_T2_star_s"):
        if name in fit_params:
            raise ValueError(
                f"IDEAL option '{name}': T2-star correction/fitting is not implemented. "
                "Remove this option to fit field offset/Theta, or use a separate "
                "T2-star fitting model for relaxation."
            )
    selection = fit_params.get("solve_for")
    if selection is not None:
        try:
            names = tuple(selection)
        except TypeError:
            names = ()
        if (
            isinstance(selection, str)
            or not names
            or not all(
                isinstance(name, str) and name in ("Field_offset", "Theta")
                for name in names
            )
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "IDEAL solve_for requires a nonempty sequence of distinct names "
                "chosen from 'Field_offset' and 'Theta'; supply one or both."
            )
    iterations = fit_params.get("max_iterations")
    if iterations is not None and (
        isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or iterations < 1
    ):
        raise ValueError(
            "IDEAL max_iterations must be a positive integer; supply 1 or more."
        )


def get_ideal_params_info() -> dict[str, object]:
    """Describe supported IDEAL calibration, transient settings and outputs.

    Returns
    -------
    dict of str to object
        New parameter-information mapping. Component/field frequencies use Hz,
        spacing uses seconds, Theta is a dimensionless complex alternating-echo
        modulation, and amplitudes retain input signal units. Omitted solve
        selection optimizes both Field_offset and Theta in the existing solver.

    Notes
    -----
    Deterministic metadata only, without input reads, mutation, solving or caches.
    T2-star correction and optimization are unsupported in this IDEAL model.
    """
    return {
        "required_fit_params": {
            "relative_frequencies_hz": "Array of metabolite frequencies relative to the reference frequency [Hz]; relative_frequencies_Hz is an accepted alias",
            "echo_spacing_s": "Echo spacing in seconds",
        },
        "optional_fit_params": {
            "echo_dim": "Name of the echo dimension in input_data (default: 'echo_idx')",
            "metabolite_names": "List of metabolite name strings, length must equal number of frequencies",
            "solve_for": "Nonempty sequence of distinct parameters to optimize: 'Field_offset' and/or 'Theta'. Default: ['Field_offset', 'Theta']",
            "init_f_offset_Hz": "Initial field-offset guess in Hz (default: 0.0)",
            "init_Theta": "Initial dimensionless complex alternating-echo modulation (default: 0.0+0.0j)",
            "stop_f_offset_hz": "Convergence threshold for field offset in Hz (default: 1.0)",
            "stop_theta": "Dimensionless complex threshold: real/imaginary parts bound the respective Theta increments (default: 0.1+0.1j)",
            "max_iterations": "Positive integer maximum IDEAL iterations per voxel (default: 25)",
            "use_multiprocessing": "Enable joblib parallel processing (default: True)",
            "number_of_cpu_cores": "Number of joblib workers (-1 uses all). Default: -1",
        },
        "output_parameters": {
            "field_offset": "B0 field offset map in Hz, present only when Field_offset is selected",
            "<metabolite_name>": "Complex signal-unit amplitude entries in fit_amps, indexed by the metabolite coordinate",
        },
        "model_equation": MODEL_EQUATION,
    }


def prepare_ideal_fit_params_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
    return_params_info: bool = False,
    *,
    base_attrs: dict[str, object] | None = None,
) -> xr.Dataset | None:
    """Prepare IDEAL calibration and preserve supported supplied solver settings.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled complex echoes. Preparation reads no values or coordinates; echo
        dimension/order and calibration are interpreted by the solver at fitting.
    fit_params : dict or None, optional
        Component ``relative_frequencies_hz`` (legacy alias
        ``relative_frequencies_Hz``) in Hz, ``echo_spacing_s`` in seconds,
        ``echo_dim`` (default ``echo_idx``) and optional component names.
        Calibration is required when fitting. The six supplied ``solve_for``,
        ``init_f_offset_Hz``, ``init_Theta``, ``stop_f_offset_hz``, ``stop_theta``
        and ``max_iterations`` settings are copied; None/omission retains solver
        defaults. See :func:`get_ideal_params_info` for units/defaults. Shared
        worker settings are retained. Input settings are not modified.
    return_params_info : bool, optional
        Return None immediately without constructing parameters or solving data.
        Use :func:`get_ideal_params_info` for descriptive metadata.
    base_attrs : dict of str to object or None, optional
        Initial result attributes; explicit model/calibration fields take precedence.

    Returns
    -------
    xarray.Dataset or None
        Attribute-only calibration/model dataset with deep-copied supplied active
        solver settings and existing worker metadata. None for information-only
        behavior. Preparation is eager, deterministic and cache-free; no signal
        conversion or fitting occurs and labelled inputs are preserved.

    Raises
    ------
    ValueError
        For conflicting frequency aliases, unsupported T2-star options, invalid
        solve selection or a nonpositive/noninteger iteration cap. Remove the
        named unsupported option; T2-star relaxation needs a separate fitting model.

    Examples
    --------
    >>> params = prepare_ideal_fit_params_dataset(data, {
    ...     "relative_frequencies_hz": [0.0, 120.0], "echo_spacing_s": 0.001,
    ...     "solve_for": ["Field_offset"], "max_iterations": 10,
    ... })  # doctest: +SKIP
    """
    if return_params_info:
        return None
    fit_params = fit_params or {}
    attrs = dict(base_attrs or {})
    attrs.update(
        {
            "fit_type": "ideal",
            "model_id": "ideal",
            "model_name": RESULT_MODEL_NAME,
            "relative_frequencies_Hz": _resolve_relative_frequencies(fit_params),
            "echo_spacing_s": fit_params.get("echo_spacing_s"),
            "echo_dim": fit_params.get("echo_dim", "echo_idx"),
            "metabolite_names": fit_params.get("metabolite_names"),
        }
    )
    for name in _ACTIVE_SOLVER_SETTINGS:
        if fit_params.get(name) is not None:
            attrs[name] = deepcopy(fit_params[name])
    _validate_ideal_solver_options({**attrs, **fit_params})
    copy_execution_params(fit_params, attrs)
    return xr.Dataset(attrs=attrs)


def fit_ideal_dataset(
    data: xr.DataArray,
    fit_params: dict | None = None,
) -> tuple[xr.Dataset, str, str, str]:
    """Fit calibrated complex echoes with the existing IDEAL solver.

    Parameters
    ----------
    data : xarray.DataArray
        Real/complex echo values in signal units with ``echo_idx`` or supplied
        ``echo_dim``. Non-singleton other dimensions must be among ``read_idx``,
        ``phase_idx``, ``slice_idx`` and ``repetition_idx``. Values are promoted
        to complex128; centered equally spaced times come from echo spacing in
        seconds, independently of echo-coordinate values. Input is not mutated.
    fit_params : dict or None, optional
        Calibration and prepared solver settings described by
        :func:`prepare_ideal_fit_params_dataset`. Frequencies and field offsets
        use Hz. Theta is dimensionless complex alternating-echo modulation.
        Supplied active settings reach the solver; None/omitted settings retain
        its defaults, including optimizing both Field_offset and Theta. Legacy
        ``field_map``, ``save_path`` and ``correct_chemical_shift`` settings are
        accepted on this prepared path; save_path writes the existing solver's
        output file. No T2-star correction/optimization is implemented.

    Returns
    -------
    xarray.Dataset
        Complex128 ``fit_amps`` in signal units with ``metabolite`` last and
        optional float64 ``field_offset`` in Hz when selected. Spatial dimensions
        follow read, phase, slice, repetition order, with singleton axes squeezed
        as before; compatible labels/input metadata remain. Theta is fitted
        internally but the existing result schema exposes no Theta map, status,
        uncertainty or generic reconstruction.
    str
        Existing model identifier ``ideal``.
    str
        Existing model-equation metadata string.
    str
        Effective echo dimension name.

    Raises
    ------
    ValueError
        For missing/conflicting calibration, echo dimensions, component labels,
        unsupported T2-star settings, solve selection or iteration cap.
    numpy.linalg.LinAlgError
        If the existing nonlinear solver encounters a singular design.

    Notes
    -----
    Eager and cache-free. Valid input/settings are preserved; the existing solver
    retains numerical initialization, iteration, phase and console/progress
    behavior. Process jobs use the existing worker implementation. No new
    scientific defaults or solver algorithm are introduced.

    Examples
    --------
    >>> result, model, equation, echo_dim = fit_ideal_dataset(data, {
    ...     "relative_frequencies_hz": [0.0, 120.0], "echo_spacing_s": 0.001,
    ...     "solve_for": ["Field_offset"], "use_multiprocessing": False,
    ... })  # doctest: +SKIP
    """
    fit_params = dict(fit_params or {})
    _validate_ideal_solver_options(fit_params)
    from hypermri.analysis.fitting.ideal import IDEAL_Solver

    relative_frequencies_hz = _resolve_relative_frequencies(fit_params)
    echo_spacing_s = fit_params.get("echo_spacing_s")
    echo_dim = fit_params.get("echo_dim", "echo_idx")
    metabolite_names = fit_params.get("metabolite_names", None)

    if relative_frequencies_hz is None:
        raise ValueError(
            "IDEAL fitting requires 'relative_frequencies_hz' (or 'relative_frequencies_Hz') in fit_params."
        )
    if echo_spacing_s is None:
        raise ValueError("IDEAL fitting requires 'echo_spacing_s' in fit_params.")
    if echo_dim not in data.dims:
        raise ValueError(
            f"echo_dim='{echo_dim}' not found in data.dims {data.dims}. "
            "Set fit_params['echo_dim'] to the correct dimension name."
        )

    relative_frequencies_hz = np.asarray(relative_frequencies_hz, dtype=np.float64)
    n_echoes = data.sizes[echo_dim]
    dummy_signal = np.zeros((n_echoes, 1, 1, 1, 1), dtype=np.complex128)

    solver = IDEAL_Solver(
        relative_frequencies_Hz=relative_frequencies_hz,
        signal_data=dummy_signal,
        echo_spacing_s=echo_spacing_s,
    )

    solver_kwarg_keys = set(_ACTIVE_SOLVER_SETTINGS) | {
        "use_multiprocessing",
        "number_of_cpu_cores",
        "field_map",
        "save_path",
        "correct_chemical_shift",
    }
    solver_kwargs = {
        k: v
        for k, v in fit_params.items()
        if k in solver_kwarg_keys
        and (k not in _ACTIVE_SOLVER_SETTINGS or v is not None)
    }

    print(
        f"Running IDEAL solver on {data.sizes[echo_dim]} echoes, "
        f"{relative_frequencies_hz.size} metabolites..."
    )

    result_ds = solver.solve_xarray(
        data=data,
        echo_dim=echo_dim,
        metabolite_names=metabolite_names,
        **solver_kwargs,
    )
    from ...result_metadata import inherit_input_xarray_metadata

    result_ds = inherit_input_xarray_metadata(result_ds, data)
    return result_ds, RESULT_MODEL_NAME, MODEL_EQUATION, echo_dim
