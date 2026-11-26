"""Lightweight FEXI computation/layout helpers."""

from __future__ import annotations

from typing import Any


def describe_fexi_layout(sequence: Any) -> dict[str, Any]:
    """Return a basic layout summary for a FEXI sequence object."""
    data_array = getattr(sequence, "_scanner_data", None)
    reco_data = getattr(sequence, "_data", None)

    return {
        "sequence": "fexi",
        "scanner_reconstruction_dims": getattr(data_array, "dims", None),
        "raw_reconstruction_dims": getattr(reco_data, "dims", None),
        "parameters": dict(getattr(sequence, "fexi_parameters", {}) or {}),
    }
