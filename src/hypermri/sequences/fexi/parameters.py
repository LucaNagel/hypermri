"""FEXI parameter collection helpers."""

from __future__ import annotations

from typing import Any

FEXI_PARAMETER_KEYS = (
    "method",
    "sequence_name",
    "n_slices",
    "matrix",
    "field_of_view_mm",
    "resolution_mm",
)


def collect_fexi_parameters(sequence: Any) -> dict[str, Any]:
    """Collect commonly useful parameters from a FEXI sequence object."""
    params = dict(getattr(sequence, "params", {}) or {})
    collected: dict[str, Any] = {}

    for key in FEXI_PARAMETER_KEYS:
        value = getattr(sequence, key, params.get(key, None))
        if value is not None:
            collected[key] = value

    return collected
