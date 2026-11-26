"""Measurement parameters of a scan as rows of a table for the browser viewer.

:func:`parameter_rows` turns the standardized parameters of a scan
(``scan.get_parameters()``) into short text rows, grouped as in the scan's
``PARAMETER_SCHEMA`` (core, subject, geometry, acquisition, field, ...).
The browser viewer shows them below the plots when "show parameters" is on;
"original names" adds the vendor's own name of each parameter (for example
``PVM_Matrix`` for ``matrix``) where the scan's handler lists it in
``PARAMETER_SOURCES``.
Nothing is loaded or reconstructed: only the stored parameter mapping is read.

Examples
--------
>>> class Scan:
...     PARAMETER_SCHEMA = {"acquisition": ("echo_time_ms", "n_averages")}
...     def get_parameters(self):
...         return {"echo_time_ms": 1.2, "n_averages": 4, "note": None}
>>> parameter_rows(Scan())
[{'group': 'acquisition', 'name': 'echo_time_ms', 'value': '1.2', 'source': ''}, \
{'group': 'acquisition', 'name': 'n_averages', 'value': '4', 'source': ''}]
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["parameter_rows"]

# Arrays with at most this many entries are written out in full.
_MAX_LISTED_VALUES = 8
# Longer texts (for example nested pulse settings) are cut to this length.
_MAX_TEXT_LENGTH = 160
# Group of parameters that the schema does not list.
_OTHER_GROUP = "other"


def parameter_rows(scan: Any | None) -> list[dict[str, str]]:
    """Return the measurement parameters of a scan as table rows.

    Rows follow the groups and order of ``scan.PARAMETER_SCHEMA``; keys that
    the schema does not list come last, sorted, in the group ``"other"``.
    Parameters whose value is None are left out.

    Parameters
    ----------
    scan : object or None
        Scan with ``get_parameters()`` (the standardized parameter mapping)
        and optionally ``PARAMETER_SCHEMA``. ``PARAMETER_SOURCES`` (canonical
        name to vendor parameter name) is read from the scan's ``handler``,
        or from the scan itself when it has no handler. Not modified. None,
        or an object without ``get_parameters``, gives no rows.

    Returns
    -------
    list of dict
        One ``{"group", "name", "value", "source"}`` per parameter, all
        strings; ``source`` is the vendor's name of the parameter, or empty
        when it is not known. Units
        are part of the parameter names (for example ``echo_time_ms``).
        Floats are written with 6 significant digits; arrays with up to 8
        entries in full, longer arrays as shape and value range.
    """
    get_parameters = getattr(scan, "get_parameters", None)
    if get_parameters is None:
        return []
    parameters = get_parameters()
    if not isinstance(parameters, Mapping):
        return []
    schema = getattr(scan, "PARAMETER_SCHEMA", {}) or {}
    handler = getattr(scan, "handler", None) or scan
    sources = getattr(handler, "PARAMETER_SOURCES", {}) or {}
    rows: list[dict[str, str]] = []
    listed: set[str] = set()
    for group, names in schema.items():
        for name in names:
            listed.add(name)
            if parameters.get(name) is not None:
                rows.append(_row(str(group), name, parameters[name], sources))
    for name in sorted(str(key) for key in parameters):
        if name not in listed and parameters[name] is not None:
            rows.append(_row(_OTHER_GROUP, name, parameters[name], sources))
    return rows


def _row(
    group: str, name: str, value: Any, sources: Mapping[str, str]
) -> dict[str, str]:
    """Return one table row."""
    return {
        "group": group,
        "name": name,
        "value": _text(value),
        "source": str(sources.get(name, "")),
    }


def _text(value: Any) -> str:
    """Return a short text for one parameter value."""
    if isinstance(value, (bool, np.bool_, str)):
        text = str(value)
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        text = f"{float(value):.6g}"
    elif isinstance(value, (list, tuple, np.ndarray)):
        text = _array_text(value)
    else:
        text = str(value)
    if len(text) > _MAX_TEXT_LENGTH:
        text = text[: _MAX_TEXT_LENGTH - 1] + "…"
    return text


def _array_text(value: Any) -> str:
    """Return an array in full when short, else its shape and value range."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return str(value)
    if array.dtype == object or array.dtype.kind in "USb":
        return str(value)
    if array.size <= _MAX_LISTED_VALUES:
        return "[" + ", ".join(_text(item) for item in array.ravel()) + "]"
    finite = array[np.isfinite(array)] if array.dtype.kind in "fc" else array
    shape = "×".join(str(size) for size in array.shape)
    if not finite.size or array.dtype.kind == "c":
        return f"array {shape}"
    return f"array {shape}, {_text(finite.min())} … {_text(finite.max())}"
