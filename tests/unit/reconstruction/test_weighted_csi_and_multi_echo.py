"""Density-weighted CSI positions and multi-echo bSSFP echo loops.

All handlers and samples are invented: a 3 x 2 CSI matrix with signed steps
``[0, -1, 1]`` and ``[0, -1]`` (positions ``[1, 0, 2]`` and ``[1, 0]``), and a
4 x 2 bSSFP matrix with three echoes. No scanner data or metadata are used.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.csi_raw import sort_csi_encoding_grid
from hypermri.reconstruction.strategies.bruker.bssfp import (
    BrukerBssfpRawRecoStrategy,
    multi_echo_times_s,
)
from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy
from hypermri.sequence_mapping import method_name_to_sequence_id
from hypermri.sequences.BSSFP import BSSFP

WEIGHTED = "<User:ssfpCSI>"
MULTI_ECHO = "<User:nmrsuFISP_H2_Fb>"
STOCK_READ = [1, 0, 2, 1, 0, 2]
STOCK_PHASE = [1, 1, 1, 0, 0, 0]


def _csi_handler(average_list):
    values = {
        "Method": WEIGHTED,
        "PVM_Matrix": np.array([3, 2]),
        "PVM_EncOrder": ["LINEAR_ENC", "LINEAR_ENC"],
        "PVM_EncSteps0": np.array([0, -1, 1]),
        "PVM_EncSteps1": np.array([0, -1]),
        "AverageList": average_list,
    }
    return SimpleNamespace(method=values, method_parameter=values.__getitem__)


def test_weighted_method_repeats_each_position_by_average_list():
    counts = [1, 3, 2, 1, 1, 2]
    _, read, phase = BrukerCsiRecoStrategy().get_phase_encoding_order(
        _csi_handler(np.array(counts))
    )
    np.testing.assert_array_equal(read, np.repeat(STOCK_READ, counts))
    np.testing.assert_array_equal(phase, np.repeat(STOCK_PHASE, counts))


@pytest.mark.parametrize(
    "bad", [[1, 1, 1], [1, 0, 1, 1, 1, 1], [1, 1.5, 1, 1, 1, 1], [1, -2, 1, 1, 1, 1]]
)
def test_invalid_average_list_names_the_method(bad):
    with pytest.raises(ValueError, match="AverageList") as error:
        BrukerCsiRecoStrategy().get_phase_encoding_order(_csi_handler(np.array(bad)))
    assert WEIGHTED in str(error.value)


def _repeated_payload():
    """Two FIDs at position (0, 0) with values 1 and 3, one at (1, 0) with 5."""
    payload = np.array([1.0, 3.0, 5.0], dtype=complex).reshape(1, 1, 3, 1, 1)
    return payload, [0, 0, 1], [0, 0, 0]


def test_repeated_positions_are_averaged_by_default_and_summed_on_request():
    payload, read, phase = _repeated_payload()
    mean, mean_counts = sort_csi_encoding_grid(payload, read, phase, (2, 1))
    total, total_counts = sort_csi_encoding_grid(
        payload, read, phase, (2, 1), repeated_positions="sum"
    )
    np.testing.assert_allclose(mean.squeeze(), [2.0, 5.0])
    np.testing.assert_allclose(total.squeeze(), [4.0, 5.0])
    np.testing.assert_array_equal(mean_counts.squeeze(), [2, 1])
    np.testing.assert_array_equal(total_counts.squeeze(), [2, 1])
    np.testing.assert_array_equal(payload.squeeze(), [1.0, 3.0, 5.0])


def test_unknown_repeated_position_rule_is_rejected():
    payload, read, phase = _repeated_payload()
    with pytest.raises(ValueError, match="repeated_positions"):
        sort_csi_encoding_grid(
            payload, read, phase, (2, 1), repeated_positions="median"
        )


def test_strategy_accepts_the_option_and_rejects_bad_values_before_reading():
    strategy = BrukerCsiRecoStrategy()
    assert "repeated_positions" in strategy.accepted_options
    unread = SimpleNamespace()
    with pytest.raises(ValueError, match="repeated_positions"):
        strategy.sort_kspace(unread, repeated_positions="median")


def test_weighted_and_multi_echo_methods_keep_their_sequence_ids():
    assert method_name_to_sequence_id(WEIGHTED) == "csi"
    assert method_name_to_sequence_id(MULTI_ECHO) == "bssfp"


def test_echo_times_are_returned_in_seconds():
    times = multi_echo_times_s({"DelayArrayForTE": [1.5, 3.0, 4.5]}, 3)
    np.testing.assert_allclose(times, [0.0015, 0.003, 0.0045])


@pytest.mark.parametrize(
    "delays", [[1.5, 3.0], [1.5, 1.5, 3.0], [0.0, 1.0, 2.0], [1.0, np.nan, 2.0]]
)
def test_inconsistent_echo_times_are_rejected(delays):
    with pytest.raises(ValueError, match="DelayArrayForTE"):
        multi_echo_times_s({"DelayArrayForTE": delays}, 3)


def _multi_echo_handler(monkeypatch, *, n_slices=1):
    """Patch the reader to return three echoes labelled 1, 2 and 3."""
    raw = np.zeros((4, 2, 3, 1, 1), dtype=complex)
    raw[:, :, :, 0, 0] = np.arange(1, 4)
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.bssfp.Dataset",
        lambda path: SimpleNamespace(data=raw),
    )
    monkeypatch.setattr(
        BrukerBssfpRawRecoStrategy,
        "_finish_reconstruction",
        lambda self, h, k, dims, **kw: xr.DataArray(k, dims=dims),
    )
    method = {
        "Method": MULTI_ECHO,
        "PVM_EncSteps1": [-1, 0],
        "PVM_NMovieFrames": 1,
        "DelayArrayForTE": [1.0, 2.0, 3.0],
    }
    return SimpleNamespace(
        path=".",
        matrix=np.array([4, 2, 1]),
        n_slices=n_slices,
        spatial_dimension="2D",
        method=method,
        acqp={"ACQ_obj_order": [0, 1, 2]},
    )


def test_multi_echo_objects_become_echo_times(monkeypatch, tmp_path):
    handler = _multi_echo_handler(monkeypatch)
    (tmp_path / "fid").write_bytes(b"x")
    handler.path = tmp_path
    image = BrukerBssfpRawRecoStrategy().reconstruct(handler)
    assert "echo_idx" in image.dims and "acquisition_idx" not in image.dims
    np.testing.assert_allclose(image.echo_idx, [0.001, 0.002, 0.003])
    assert image.echo_idx.attrs["unit"] == "s"
    np.testing.assert_array_equal(
        image.isel(read_idx=0, phase_idx=0).squeeze(), [1, 2, 3]
    )


def test_multi_echo_with_several_slices_is_refused(monkeypatch, tmp_path):
    handler = _multi_echo_handler(monkeypatch, n_slices=3)
    (tmp_path / "fid").write_bytes(b"x")
    handler.path = tmp_path
    with pytest.raises(ValueError, match="seq2d"):
        BrukerBssfpRawRecoStrategy().reconstruct(handler)


def test_bssfp_counts_echoes_from_the_echo_delays():
    method = {"Method": MULTI_ECHO, "DelayArrayForTE": [1.0, 2.0, 3.0, 4.0, 5.0]}
    handler = SimpleNamespace(method=method, method_parameter=method.__getitem__)
    scan = BSSFP.__new__(BSSFP)
    scan.handler = handler
    assert scan.get_nechoes() == 5
