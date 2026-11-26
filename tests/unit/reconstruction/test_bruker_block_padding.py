"""Characterize ParaVision block trimming with invented complex sample labels.

Explicit byte-boundary cases describe the storage format independently of the
implementation. Integer-valued labels in complex64/128 are exactly represented;
trimming performs no floating-point arithmetic, so equality must be exact.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.reconstruction.strategies.bruker._block_padding import (
    _strip_block_padding,
)
from hypermri.reconstruction.strategies.bruker.raw import BrukerRawRecoStrategy

# (stored scalar format, scalar readout count, receivers, padded complex width).
# 1024-byte blocks hold 256 complex int16 pairs or 128 complex int32/float32
# pairs. Receiver samples share the block; each row has its own trailing pad.
_BOUNDARY_CASES = [
    ("GO_16BIT_SGN_INT", 8, 1, 256),
    ("GO_32BIT_SGN_INT", 8, 1, 128),
    ("GO_32BIT_FLOAT", 8, 1, 128),
    ("GO_16BIT_SGN_INT", 512, 1, 256),
    ("GO_16BIT_SGN_INT", 514, 1, 512),
    ("GO_32BIT_SGN_INT", 256, 1, 128),
    ("GO_32BIT_SGN_INT", 258, 1, 256),
    ("GO_32BIT_FLOAT", 258, 1, 256),
    ("GO_32BIT_SGN_INT", 128, 3, 256),
    ("GO_16BIT_SGN_INT", 254, 2, 256),
    ("GO_16BIT_SGN_INT", 256, 2, 256),
    ("GO_16BIT_SGN_INT", 258, 2, 512),
]


@pytest.mark.parametrize("format_name,readout,receivers,padded_width", _BOUNDARY_CASES)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("strided", [False, True])
def test_line_labels_survive_storage_boundaries_and_acquisition_loops(
    format_name, readout, receivers, padded_width, padded, dtype, strided
):
    """Trimming keeps every loop/receiver label and leaves read-only input intact."""
    # Six higher-dimensional positions, two objects, two repetitions: 24 lines.
    acqp = {
        "ACQ_size": np.array([readout, 3, 2]),
        "NI": 2,
        "NR": 2,
        "GO_raw_data_format": format_name,
        "GO_block_size": "Standard_KBlock_Format" if padded else "continuous",
    }
    acquired_width = readout * receivers // 2
    stored_width = padded_width if padded else acquired_width
    expected = np.arange(24 * acquired_width).reshape(24, acquired_width) + 3j
    stored = np.full((24, stored_width), -777 - 999j, dtype=dtype)
    stored[:, :acquired_width] = expected
    raw = stored.ravel()
    if strided:
        backing = np.empty(raw.size * 2, dtype=dtype)
        backing[::2] = raw
        backing[1::2] = 123 + 456j
        raw = backing[::2]
    before = raw.copy()
    metadata_before = acqp["ACQ_size"].copy()
    raw.flags.writeable = False

    actual = _strip_block_padding(raw, acqp, receivers)

    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(acqp["ACQ_size"], metadata_before)
    assert np.shares_memory(actual, raw)
    np.testing.assert_array_equal(_strip_block_padding(raw, acqp, receivers), actual)


@pytest.mark.parametrize("readout", [[8], (8,), np.array([8])])
def test_one_dimensional_acquisition_has_only_object_and_repetition_lines(readout):
    """A single spectral dimension still retains all NI/NR lines."""
    acqp = {
        "ACQ_size": readout,
        "NI": 2,
        "NR": 3,
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "GO_block_size": "continuous",
    }
    raw = np.arange(24).astype(np.complex128)
    np.testing.assert_array_equal(_strip_block_padding(raw, acqp, 1), raw.reshape(6, 4))


@pytest.mark.parametrize(
    "mode,receivers", [("SingleExperiment", 1), ("SingleExperiment", 2), ("Other", 2)]
)
@pytest.mark.parametrize("use_handler_fid", [False, True])
def test_public_postprocess_keeps_selected_receiver_branch(
    mode, receivers, use_handler_fid
):
    """Retain the existing SingleExperiment multi-receiver bypass explicitly."""
    acqp = {
        "ACQ_experiment_mode": mode,
        "ACQ_size": [8, 2],
        "NI": 1,
        "NR": 1,
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "GO_block_size": "Standard_KBlock_Format",
    }
    raw = np.arange(256).astype(np.complex64)
    original = raw.copy()
    method = {"PVM_EncNReceivers": receivers}
    handler = SimpleNamespace(
        fid=raw,
        acqp=acqp,
        method=method,
        method_parameter=method.__getitem__,
    )
    strategy = BrukerRawRecoStrategy()

    actual = strategy.postprocess_fid_file(
        None if use_handler_fid else raw, handler=handler
    )

    if mode == "SingleExperiment" and receivers > 1:
        assert actual is raw
    else:
        np.testing.assert_array_equal(
            actual, original.reshape(2, 128)[:, : 4 * receivers]
        )
    np.testing.assert_array_equal(raw, original)


def test_padding_preserves_legacy_four_byte_fallback():
    """The array-only adapter keeps its existing fallback; file decoding validates."""
    acqp = {
        "ACQ_size": [8],
        "NI": 1,
        "NR": 1,
        "GO_raw_data_format": "legacy_unknown",
        "GO_block_size": "Standard_KBlock_Format",
    }
    raw = np.arange(128).astype(np.complex64)
    np.testing.assert_array_equal(_strip_block_padding(raw, acqp, 1), raw[None, :4])
