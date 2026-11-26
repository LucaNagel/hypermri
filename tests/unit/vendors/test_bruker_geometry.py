"""Focused tests for Bruker acquisition geometry metadata."""

from pathlib import Path

import numpy as np
import pytest

from hypermri.vendors.bruker import BrukerHandler


def test_stock_csi_is_spectroscopic_despite_paravision_encoding_flag():
    assert BrukerHandler._is_spectroscopic_acquisition("<Bruker:CSI>", "No")


def test_non_csi_imaging_with_disabled_spectroscopy_is_not_spectroscopic():
    assert not BrukerHandler._is_spectroscopic_acquisition("<Bruker:FLASH>", "No")


def test_slice_pack_counts_preserve_single_pack_scalar():
    assert BrukerHandler._slice_pack_counts(9) == (9,)


def test_slice_pack_counts_include_every_multi_pack_entry():
    counts = BrukerHandler._slice_pack_counts(np.array([5, 3, 5]))

    assert counts == (5, 3, 5)
    assert sum(counts) == 13


def test_slice_pack_counts_ignore_missing_and_nonpositive_values():
    assert BrukerHandler._slice_pack_counts(None) == ()
    assert BrukerHandler._slice_pack_counts([0, -1, 3]) == (3,)


def test_first_parameter_value_normalises_scalar_and_array_fields_independently():
    assert BrukerHandler._first_parameter_value(np.array([1.5, 2.5])) == 1.5
    assert BrukerHandler._first_parameter_value(3.5) == 3.5
    assert BrukerHandler._first_parameter_value(None) == 0


def _handler_with_path(path: Path) -> BrukerHandler:
    handler = object.__new__(BrukerHandler)
    handler.path = path
    return handler


def test_has_raw_data_rejects_missing_and_empty_sources(tmp_path: Path):
    (tmp_path / "fid").touch()
    handler = _handler_with_path(tmp_path)

    assert handler.has_raw_data is False


def test_has_raw_data_accepts_nonempty_file_or_loaded_array(tmp_path: Path):
    handler = _handler_with_path(tmp_path)
    (tmp_path / "ser").write_bytes(b"raw")
    assert handler.has_raw_data is True
    assert handler.has_fid_data is True

    (tmp_path / "ser").unlink()
    handler.fid = np.ones(1, dtype=np.complex64)
    assert handler.has_raw_data is True
    assert handler.has_fid_data is True


def test_rawdata_job_does_not_masquerade_as_fid_data(tmp_path: Path):
    handler = _handler_with_path(tmp_path)
    (tmp_path / "rawdata.job0").write_bytes(b"raw")

    assert handler.has_raw_data is True
    assert handler.has_fid_data is False


def _handler_with_method(method: dict, slices_per_pack: tuple) -> BrukerHandler:
    handler = object.__new__(BrukerHandler)
    handler.method = method
    handler.slices_per_pack = slices_per_pack
    return handler


def test_slice_pack_offsets_hold_one_row_per_package():
    # The offsets of a three-plane localizer, with a read offset added to the
    # second package; values are illustrative.
    handler = _handler_with_method(
        {
            "PVM_SPackArrReadOffset": [0, 1.5, 0],
            "PVM_SPackArrPhase1Offset": [0.0, 0.0, -6.5],
            "PVM_SPackArrSliceOffset": [0.0, -4.0, -0.75],
        },
        (7, 7, 7),
    )

    offsets = handler.slice_pack_offsets_mm

    np.testing.assert_array_equal(
        offsets, [[0.0, 0.0, 0.0], [1.5, 0.0, -4.0], [0.0, -6.5, -0.75]]
    )
    assert offsets.dtype == np.float64


def test_slice_pack_offsets_of_one_package_come_from_scalar_parameters():
    handler = _handler_with_method(
        {
            "PVM_SPackArrReadOffset": 1.5,
            "PVM_SPackArrPhase1Offset": 2.5,
            "PVM_SPackArrSliceOffset": 3.5,
        },
        (9,),
    )

    np.testing.assert_array_equal(handler.slice_pack_offsets_mm, [[1.5, 2.5, 3.5]])


def test_missing_slice_pack_offsets_are_read_as_zero():
    np.testing.assert_array_equal(
        _handler_with_method({}, (2, 3)).slice_pack_offsets_mm, np.zeros((2, 3))
    )
    np.testing.assert_array_equal(
        _handler_with_method({}, ()).slice_pack_offsets_mm, np.zeros((1, 3))
    )


def test_slice_pack_offsets_reject_another_number_of_values_than_packages():
    handler = _handler_with_method({"PVM_SPackArrPhase1Offset": [0.0, 1.0]}, (7, 7, 7))

    with pytest.raises(ValueError, match="PVM_SPackArrPhase1Offset .* 2 values"):
        handler.slice_pack_offsets_mm
