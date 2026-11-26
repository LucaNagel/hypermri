"""Which reconstruction ``data`` returns and what its first access starts.

All arrays are small synthetic ``xarray`` objects; no scan is read.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hypermri.exceptions import ReconstructionError
from hypermri.mixins import StandardizedDataArrayMixin
from hypermri.sequences.FLASH import FLASH


def _array(value: float) -> xr.DataArray:
    return xr.DataArray(np.full((2, 2), value), dims=("read_idx", "phase_idx"))


class _CountingData(StandardizedDataArrayMixin):
    """Mixin user whose lazy loaders return fixed arrays and count their calls."""

    def __init__(self, raw=None, scanner=None):
        super().__init__()
        self.raw = raw
        self.scanner = scanner
        self.calls = []

    def _load_data(self):
        self.calls.append("data")
        return self.raw

    def _load_scanner_data(self):
        self.calls.append("scanner_data")
        return self.scanner


def test_data_starts_the_raw_reconstruction_and_returns_it():
    raw, scanner = _array(0.0), _array(1.0)
    obj = _CountingData(raw=raw, scanner=scanner)

    assert obj.primary_data is raw
    assert obj.primary_data is raw
    assert obj.calls == ["data"]
    assert obj.primary_data is raw


def test_data_returns_the_scanner_reconstruction_without_a_raw_one():
    scanner = _array(1.0)
    obj = _CountingData(raw=None, scanner=scanner)

    assert obj.primary_data is scanner
    assert obj.primary_data is scanner
    assert obj.calls == ["data", "scanner_data"]


def test_data_returns_none_without_any_reconstruction():
    obj = _CountingData()

    assert obj.primary_data is None
    assert obj.calls == ["data", "scanner_data"]


def test_data_does_not_start_a_reconstruction_when_raw_data_are_stored():
    raw, scanner = _array(0.0), _array(1.0)
    obj = _CountingData(scanner=scanner)
    obj.data = raw

    assert obj.primary_data is raw
    assert obj.calls == []


class _DummyHandler:
    def __init__(self):
        self.params = {}
        self.type = "dummy"
        self.id = "dummy-id"
        self.fid = None
        self.is_spectroscopic = False
        self.method = {}

    def get_parameters(self):
        return {}


@pytest.fixture
def flash(monkeypatch):
    """Return a FLASH object whose ``reconstruct`` is replaced by a recorder."""
    scan = FLASH(_DummyHandler())
    scan.reconstruct_calls = []
    scan.reconstruct_results = {}

    def reconstruct(reco_type, **kwargs):
        scan.reconstruct_calls.append(reco_type)
        result = scan.reconstruct_results[reco_type]
        if isinstance(result, Exception):
            raise result
        return result

    scan.reconstruct = reconstruct
    return scan


def test_anatomy_data_after_a_minimal_load_starts_the_raw_reconstruction(flash):
    """After ``mode="minimal"`` only ``2dseq`` is stored; ``data`` is raw."""
    scanner, raw = _array(1.0), _array(0.0)
    flash.scanner_data = scanner
    flash.reconstruct_results["raw"] = raw

    assert flash.primary_data is raw
    assert flash.reconstruct_calls == ["raw"]
    assert flash.scanner_data is scanner


def test_anatomy_data_preserves_raw_errors_without_scanner_fallback(flash):
    scanner = _array(1.0)
    flash.reconstruct_results["raw"] = ReconstructionError("synthetic failure")
    flash.reconstruct_results["seq2d"] = scanner

    for _ in range(2):
        with pytest.raises(ReconstructionError, match="synthetic failure"):
            _ = flash.primary_data
    assert flash.reconstruct_calls == ["raw"]
    assert flash.scanner_data is scanner
    assert flash.reconstruct_calls == ["raw", "seq2d"]
    raw = _array(0.0)
    flash.data = raw
    assert flash.primary_data is raw


def test_anatomy_data_falls_back_when_the_raw_strategy_returns_nothing(flash):
    scanner = _array(1.0)
    flash.reconstruct_results["raw"] = None
    flash.reconstruct_results["seq2d"] = scanner

    assert flash.primary_data is scanner
    assert flash.reconstruct_calls == ["raw", "seq2d"]
