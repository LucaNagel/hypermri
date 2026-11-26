"""The one owner of the ``data_precision`` option.

The expected dtypes restate the precision rules that the reconstruction
strategies and the Bruker handler applied before this option had one owner: the acquired complex FID
is only reduced for ``float16``, stored complex results are ``complex64`` for
``float16`` and ``float32``, and real arrays take the requested type.  A reconstruction converts a copy and leaves the acquired ``handler.fid``
unchanged.  The arrays are small synthetic values; conversions to ``float16``, ``float32`` and
``complex64`` are compared with the rounding of NumPy's own ``astype``.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize(
    "precision,dtype",
    [("float16", np.complex64), ("float32", np.complex64), ("float64", np.complex128)],
)
@pytest.mark.parametrize("complex_input", [False, True])
def test_explicit_complex_storage_preserves_reader_values(
    precision, dtype, complex_input
):
    """FID readers retain complex signals, even with zero imaginary parts."""
    from hypermri.reconstruction.precision import convert_complex_for_storage

    samples = np.array([0.0, 1.25, -7.5], dtype=np.float64)
    if complex_input:
        samples = samples.astype(np.complex128)
    acquired = samples.copy()
    result = convert_complex_for_storage(samples, precision)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, samples.astype(dtype))
    np.testing.assert_array_equal(samples, acquired)
    assert convert_complex_for_storage(result, precision) is result


def test_former_fitting_precision_constant_is_removed():
    """The unused DATA_PRECISION constant was removed on 15 November 2026."""
    from hypermri.analysis import fitting
    from hypermri.analysis.fitting import fit_maps

    assert "DATA_PRECISION" not in fitting.__all__
    for module in (fitting, fit_maps):
        with pytest.raises(AttributeError, match="DATA_PRECISION"):
            module.DATA_PRECISION


@pytest.mark.parametrize("precision", ["float16", "float32", "float64"])
def test_siemens_labelled_storage_uses_explicit_precision(precision):
    from hypermri.reconstruction.precision import convert_for_storage
    from hypermri.vendors.siemens import SiemensHandler

    handler = SiemensHandler.__new__(SiemensHandler)
    samples = np.array([[1.25 + 0j, 2.5 + 0j]], dtype=np.complex128)
    coords = {"read_idx": ("read_idx", [1.0, 2.0], {"unit": "mm"})}
    handler.set_seq2d_rps_xarray(
        samples, ["phase_idx", "read_idx"], coords, data_precision=precision
    )
    result = handler.seq2d_rps_xarray
    np.testing.assert_array_equal(result, convert_for_storage(samples, precision))
    assert result.dtype == convert_for_storage(samples, precision).dtype
    assert result.read_idx.attrs["unit"] == "mm"


import hypermri.reconstruction.strategies as strategies_package
from hypermri.reconstruction.precision import (
    convert_for_calculation,
    convert_for_storage,
    parse_data_precision,
)
from hypermri.reconstruction.strategies.bruker.bssfp import BrukerBssfpRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy
from hypermri.reconstruction.strategies.bruker.dti import BrukerDtiRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.epi import BrukerEpiRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.igflash import (
    BrukerIgflashRawRecoStrategy,
)
from hypermri.reconstruction.strategies.bruker.rare import BrukerRareRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.raw import (
    BrukerMGERawRecoStrategy,
    BrukerRawRecoStrategy,
)
from hypermri.reconstruction.strategies.bruker.seq2d import Brukerseq2dRecoStrategy
from hypermri.reconstruction.strategies.bruker.spectroscopy import (
    BrukerSpectroRecoStrategy,
)
from hypermri.vendors.bruker import BrukerHandler

_COMPLEX = np.array([1.0 + 1.0e-4j, 2.0e5 + 0.0j], dtype=np.complex128)
_REAL = np.array([1.0e-4, 6.0e4], dtype=np.float64)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("float16", np.float16),
        (" Float32 ", np.float32),
        ("FLOAT64", np.float64),
        (np.float32, np.float32),
        (np.dtype("float16"), np.float16),
        (np.dtype(np.float64), np.float64),
    ],
)
def test_every_spelling_of_a_precision_gives_the_same_numpy_type(value, expected):
    assert parse_data_precision(value) is expected


@pytest.mark.parametrize(
    "value",
    ["float8", "single", "", None, float, np.complex64, np.int32, 32],
)
def test_unsupported_precision_raises_instead_of_falling_back_to_float16(value):
    with pytest.raises(ValueError, match="Unsupported data_precision") as info:
        parse_data_precision(value)

    assert "'float16', 'float32', 'float64'" in str(info.value)


@pytest.mark.parametrize(
    ("precision", "dtype"),
    [("float16", np.complex64), ("float32", np.complex64), ("float64", np.complex128)],
)
def test_storage_keeps_complex_data_complex(precision, dtype):
    stored = convert_for_storage(_COMPLEX, precision)

    assert stored.dtype == dtype
    np.testing.assert_array_equal(stored, _COMPLEX.astype(dtype))


def test_storage_keeps_complex_dtype_with_zero_imaginary_parts():
    data = np.array([1.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128)

    stored = convert_for_storage(data, np.float32)

    assert stored.dtype == np.complex64
    np.testing.assert_array_equal(stored, data)


@pytest.mark.parametrize("precision", [np.float16, np.float32, np.float64])
def test_storage_and_calculation_give_real_data_the_requested_type(precision):
    for convert in (convert_for_storage, convert_for_calculation):
        converted = convert(_REAL, precision)

        assert converted.dtype == precision
        np.testing.assert_array_equal(converted, _REAL.astype(precision))


def test_calculation_reduces_a_complex_fid_only_for_float16():
    reduced = convert_for_calculation(_COMPLEX, "float16")

    assert reduced.dtype == np.complex64
    np.testing.assert_array_equal(reduced, _COMPLEX.astype(np.complex64))
    assert convert_for_calculation(_COMPLEX, "float32") is _COMPLEX
    assert convert_for_calculation(_COMPLEX, "float64") is _COMPLEX


@pytest.mark.parametrize("convert", [convert_for_storage, convert_for_calculation])
def test_conversion_does_not_modify_its_input(convert):
    data = _COMPLEX.copy()

    convert(data, "float16")

    np.testing.assert_array_equal(data, _COMPLEX)
    assert data.dtype == np.complex128


@pytest.mark.parametrize("convert", [convert_for_storage, convert_for_calculation])
def test_missing_and_empty_arrays_are_returned_unchanged(convert):
    empty = np.array([], dtype=np.complex128)

    assert convert(None, "float32") is None
    assert convert(empty, "float32") is empty


@pytest.mark.parametrize("convert", [convert_for_storage, convert_for_calculation])
def test_conversion_rejects_an_unsupported_precision(convert):
    with pytest.raises(ValueError, match="'float8'"):
        convert(_REAL, "float8")


def test_calculation_keeps_non_finite_values_and_warns(caplog):
    caplog.set_level(logging.WARNING, logger="hypermri.reconstruction.precision")
    data = np.array([np.nan, 1.0], dtype=np.float64)

    converted = convert_for_calculation(data, "float32")

    assert np.isnan(converted[0]) and converted[1] == 1.0
    assert "NaN or infinite" in caplog.text


class _StageReachedError(Exception):
    """Raised by a patched stage once the FID it received has been recorded."""


def _record_fid_then_stop(strategy, monkeypatch, stage):
    """Make ``stage`` of ``strategy`` record its input FID and stop."""
    received = []

    def record(values=None, handler=None):
        received.append(values)
        raise _StageReachedError

    monkeypatch.setattr(strategy, stage, record)
    return received


@pytest.mark.parametrize(
    "strategy_class", [BrukerRawRecoStrategy, BrukerMGERawRecoStrategy]
)
def test_raw_reconstruction_computes_in_float16_without_changing_the_handler(
    strategy_class, monkeypatch
):
    strategy = strategy_class()
    monkeypatch.setattr(
        strategy, "postprocess_fid_file", lambda values, handler: values
    )
    received = _record_fid_then_stop(strategy, monkeypatch, "reshape_fid_to_kspace")
    handler = SimpleNamespace(
        fid=_COMPLEX.copy(),
        enc_matrix=np.array([_COMPLEX.size, 1, 1]),
        seq2d=np.float32([1.0e5]),
        expnum=1,
    )

    with pytest.raises(_StageReachedError):
        strategy.reconstruct(handler, data_precision="float16")

    assert received[0].dtype == np.complex64
    np.testing.assert_array_equal(received[0], _COMPLEX.astype(np.complex64))
    assert handler.fid.dtype == np.complex128
    np.testing.assert_array_equal(handler.fid, _COMPLEX)
    assert handler.seq2d.dtype == np.float32


def test_raw_reconstruction_converts_an_explicit_processed_fid():
    handler = SimpleNamespace(fid=_COMPLEX.copy())
    processed = _REAL.copy()

    fid = BrukerRawRecoStrategy._fid_for_calculation(
        handler, False, "float16", data=processed
    )

    assert fid.dtype == np.float16
    np.testing.assert_array_equal(fid, _REAL.astype(np.float16))
    assert processed.dtype == np.float64


def test_rare_sorting_computes_in_float16_without_changing_the_handler(monkeypatch):
    strategy = BrukerRareRawRecoStrategy()
    received = _record_fid_then_stop(strategy, monkeypatch, "postprocess_fid_file")
    handler = SimpleNamespace(fid=_COMPLEX.copy(), expnum=1)

    with pytest.raises(_StageReachedError):
        strategy.postprocess_fid_rare(handler=handler, data_precision="float16")

    assert received[0].dtype == np.complex64
    np.testing.assert_array_equal(handler.fid, _COMPLEX)
    assert handler.fid.dtype == np.complex128


def _spectroscopy_handler() -> SimpleNamespace:
    """Synthetic single-voxel FID of 6 points without leading filter points."""
    dwell_time_s = 0.125
    handler = SimpleNamespace(
        id=1,
        expnum=1,
        fid=np.array([1.0 + 1.0e-4j, 2.0e5, 3.0, 4.0, 5.0, 6.0], dtype=np.complex128),
        n_repetitions=1,
        n_averages=1,
        n_slices=1,
        n_receivers=1,
        n_acquisition=6,
        method={},
        reco={},
        cut_off=0,
        dwell_time_s=dwell_time_s,
        sampling_time_axis_s=np.arange(6, dtype=float) * dwell_time_s,
        bandwidth_hz=1.0 / dwell_time_s,
        bandwidth_ppm=4.0,
        center_ppm=0.0,
        freq_axis_ppm=None,
        freq_axis_hz=None,
        is_spectroscopic=True,
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    return handler


def _spectroscopy_strategy(monkeypatch) -> BrukerSpectroRecoStrategy:
    strategy = BrukerSpectroRecoStrategy()
    monkeypatch.setattr(
        strategy, "reco_rawdata", lambda handler, **kwargs: kwargs["kspace_xarray"]
    )
    return strategy


def test_spectroscopy_computes_in_float16_without_changing_the_acquired_fid(
    monkeypatch,
):
    strategy = _spectroscopy_strategy(monkeypatch)
    handler = _spectroscopy_handler()
    acquired = handler.fid.copy()

    result = strategy.reconstruct(handler)

    assert not hasattr(handler, "fid_pp")
    expected = np.fft.fftshift(np.fft.fft(acquired.astype(np.complex64)))
    np.testing.assert_array_equal(result.squeeze(), expected)
    assert handler.fid.dtype == np.complex128
    np.testing.assert_array_equal(handler.fid, acquired)
    assert result.dtype == np.complex64


def test_spectroscopy_result_does_not_depend_on_an_earlier_precision(monkeypatch):
    """A float64 request after a float16 one starts from the acquired FID."""
    strategy = _spectroscopy_strategy(monkeypatch)
    handler = _spectroscopy_handler()
    fresh = strategy.reconstruct(_spectroscopy_handler(), data_precision="float64")

    strategy.reconstruct(handler, data_precision="float16")
    after_float16 = strategy.reconstruct(handler, data_precision="float64")

    assert after_float16.dtype == np.complex128
    xr.testing.assert_identical(after_float16, fresh)


@pytest.mark.parametrize(
    "strategy_class",
    [
        BrukerRawRecoStrategy,
        BrukerMGERawRecoStrategy,
        BrukerRareRawRecoStrategy,
        BrukerCsiRecoStrategy,
        BrukerSpectroRecoStrategy,
        BrukerBssfpRawRecoStrategy,
        BrukerDtiRawRecoStrategy,
        BrukerEpiRawRecoStrategy,
        BrukerIgflashRawRecoStrategy,
        Brukerseq2dRecoStrategy,
    ],
    ids=lambda strategy_class: strategy_class.__name__,
)
def test_strategy_rejects_an_unsupported_precision_before_reconstructing(
    strategy_class,
):
    handler = SimpleNamespace(fid=_COMPLEX.copy(), expnum=1)

    with pytest.raises(ValueError, match="Unsupported data_precision 'float8'"):
        strategy_class().reconstruct(handler, data_precision="float8")

    np.testing.assert_array_equal(handler.fid, _COMPLEX)


def test_bruker_2dseq_reader_rejects_an_unsupported_precision(tmp_path):
    handler = object.__new__(BrukerHandler)
    handler.path = tmp_path

    with pytest.raises(ValueError, match="Unsupported data_precision"):
        handler.load_2dseq_file(data_precision="float8")


def test_reconstruction_strategies_import_names_explicitly():
    package = Path(strategies_package.__file__).parent
    wildcard_imports = [
        f"{path.relative_to(package)}:{node.lineno}"
        for path in sorted(package.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
    ]

    assert wildcard_imports == []
