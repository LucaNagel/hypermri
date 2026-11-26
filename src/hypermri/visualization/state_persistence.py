"""Persist viewer state and restore recognized fields with presence information.

Snapshots live on the source object. Loading applies display settings eagerly;
it reads no scientific arrays and initializes no reconstruction or widgets.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from hypermri.visualization.state import PlotState

_STORE_ATTR = "_saved_plot_settings"


def save_plot_state(scan: Any, state: PlotState, context: str) -> bool:
    """Snapshot a viewer state on its calling source object.

    Parameters
    ----------
    scan : object or None
        Mutable source that owns the in-memory saved-settings mapping.
    state : PlotState
        Display settings to snapshot recursively; field units follow their owners.
    context : str
        Nonempty key under which an existing snapshot is replaced.

    Returns
    -------
    bool
        True if the snapshot was assigned; False for absent source/context or
        rejected attribute assignment. No filesystem persistence is performed.

    Raises
    ------
    TypeError
        If state is not a dataclass accepted by dataclasses.asdict.

    Notes
    -----
    Snapshots are eager and independent of subsequent state mutations. The source's
    settings mapping changes; acquired data, reconstruction and widgets are untouched.
    """
    if scan is None or not context:
        return False

    store = getattr(scan, _STORE_ATTR, None)
    if not isinstance(store, dict):
        store = {}

    store[str(context)] = asdict(state)

    try:
        setattr(scan, _STORE_ATTR, store)
    except Exception:
        return False
    return True


def load_plot_state(scan: Any, state: PlotState, context: str) -> bool:
    """Apply saved viewer fields and record which settings were present.

    Parameters
    ----------
    scan : object or None
        Source owning the in-memory saved-view mapping. None has no saved state.
        Only that attribute is read; no scientific arrays are loaded.
    state : PlotState
        Mutable target. Recognized saved fields update this instance in place;
        omitted fields are left unchanged. Field units retain their state owners.
    context : str
        Saved-view key, normally ``plot`` or the enabled tool's context.

    Returns
    -------
    bool
        True when a dictionary snapshot was applied, otherwise False. Unknown
        legacy keys remain ignored. Qualified paths of recognized applied fields
        are recorded on state for sequence-default precedence checks.

    Raises
    ------
    AttributeError
        If a source or state attribute cannot be read or a target field cannot be
        assigned. Malformed snapshots retain their existing assignment behavior.

    Notes
    -----
    Application is eager and deterministic. Target state and its presence marker
    are updated; the saved snapshot, acquired values, dimensions, coordinates and
    storage dtype are not modified. No reconstruction, fitting or widgets occur.

    Examples
    --------
    >>> state = PlotState()
    >>> restored = load_plot_state(scan, state, "plot")  # doctest: +SKIP
    """
    if scan is None or not context:
        return False

    store = getattr(scan, _STORE_ATTR, None)
    if not isinstance(store, dict):
        return False

    state_dict = store.get(str(context))
    if not isinstance(state_dict, dict):
        return False

    state._loaded_state_overrides = _apply_state_dict(state, state_dict)
    return True


def _apply_state_dict(
    target: Any, values: dict[str, Any], prefix: str = ""
) -> set[str]:
    """Recursively update a PlotState dataclass from a nested dictionary."""
    applied = set()
    for key, value in values.items():
        if not hasattr(target, key):
            continue

        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            applied.update(_apply_state_dict(current, value, prefix=f"{prefix}{key}."))
        else:
            setattr(target, key, value)
            applied.add(f"{prefix}{key}")
    return applied
