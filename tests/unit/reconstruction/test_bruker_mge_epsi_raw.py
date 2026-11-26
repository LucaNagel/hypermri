"""Raw k-space ordering of Bruker 2D multislice, MGE and EPSI scans.

The FIDs are synthetic: every sample carries a code for its loop indices, so
the expected positions follow from the ParaVision loop order alone. Loop
orders come from ParaVision method documentation and were confirmed on a
Bruker phantom study (MGE raw against ``2dseq`` correlation 1.000, EPSI
0.99984 after the spectral transform). Reordering is exact, so codes are
compared exactly; the EPSI phase estimate is compared within 1e-9 relative,
far below the rounding of complex128 FFTs of 16 points.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.centred_dft import centered_fft, centered_ifft
from hypermri.reconstruction.selection import find_reconstruction
from hypermri.reconstruction.strategies.bruker.epsi import (
    BrukerEpsiRawRecoStrategy,
    correct_odd_even_echo_phase,
    epsi_dwell_time_ms,
    epsi_segment_count,
)
from hypermri.reconstruction.strategies.bruker.raw import (
    BrukerMGERawRecoStrategy,
    _mge_readout_is_bipolar,
    _reshape_2d_multislice_lines,
)


def _code(slice_index, phase, echo, read):
    return 1000 * slice_index + 100 * phase + 10 * echo + read


@pytest.mark.parametrize("phase_factor", [1, 2, 4])
def test_2d_multislice_lines_follow_phase_factor(phase_factor):
    n_phase, n_slices, n_read = 4, 3, 2
    lines = []
    for segment in range(n_phase // phase_factor):
        for slice_index in range(n_slices):
            for row in range(phase_factor):
                phase = segment * phase_factor + row
                lines.append([_code(slice_index, phase, 0, r) for r in range(n_read)])
    handler = SimpleNamespace(
        enc_matrix=(n_read, n_phase, n_slices),
        n_repetitions=1,
        acqp={"ACQ_phase_factor": phase_factor},
    )

    kspace = _reshape_2d_multislice_lines(
        np.asarray(lines, dtype=complex).ravel(),
        handler,
        line_shape=(1, n_read),
        label="test",
    )

    assert kspace.shape == (1, 1, n_slices, n_phase, 1, n_read)
    for s in range(n_slices):
        for p in range(n_phase):
            expected = [_code(s, p, 0, r) for r in range(n_read)]
            np.testing.assert_array_equal(kspace[0, 0, s, p, 0].real, expected)


def test_2d_multislice_lines_reject_phase_factor_not_dividing_matrix():
    handler = SimpleNamespace(
        enc_matrix=(2, 4, 1), n_repetitions=1, acqp={"ACQ_phase_factor": 3}
    )
    with pytest.raises(ValueError, match="ACQ_phase_factor"):
        _reshape_2d_multislice_lines(
            np.zeros(8, complex), handler, line_shape=(1, 2), label="test"
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("allEchoes", True), ("positiveEchoes", False), (None, False)],
)
def test_mge_readout_mode(mode, expected):
    method = {} if mode is None else {"EchoAcqMode": mode}
    assert _mge_readout_is_bipolar(SimpleNamespace(method=method)) is expected


def test_mge_readout_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="EchoAcqMode"):
        _mge_readout_is_bipolar(SimpleNamespace(method={"EchoAcqMode": "other"}))


def _mge_handler(mode):
    n_read, n_phase, n_slices, n_echoes = 4, 2, 2, 3
    fid = []
    for phase in range(n_phase):
        for slice_index in range(n_slices):
            for echo in range(n_echoes):
                read = np.arange(n_read)
                if mode == "allEchoes" and echo % 2:
                    read = read[::-1]
                fid.extend(_code(slice_index, phase, echo, r) for r in read)
    return SimpleNamespace(
        fid=np.asarray(fid, dtype=complex),
        enc_matrix=(n_read, n_phase, n_slices),
        matrix=(n_read, n_phase, n_slices),
        n_echoes=n_echoes,
        n_receivers=1,
        n_repetitions=1,
        spatial_dimension="2D",
        acqp={"ACQ_phase_factor": 1},
        method={"EchoAcqMode": mode, "PVM_EncSteps1": [-1, 0]},
    )


@pytest.mark.parametrize("mode", ["allEchoes", "positiveEchoes"])
def test_mge_kspace_restores_read_order_of_every_echo(mode):
    handler = _mge_handler(mode)
    kspace = BrukerMGERawRecoStrategy().reshape_fid_to_kspace(handler=handler)

    assert kspace.shape == (1, 1, 2, 2, 3, 1, 4)
    for s in range(2):
        for p in range(2):
            for e in range(3):
                expected = [_code(s, p, e, r) for r in range(4)]
                np.testing.assert_array_equal(kspace[0, 0, s, p, e, 0].real, expected)


def test_epsi_dwell_time_and_segment_count():
    method = {
        "PVM_EpiEchoSpacing": 0.48,
        "NSegments": 4,
        "PVM_EpiNShots": 4,
        "PVM_EpiEchoShiftA": [0.0, 0.12, 0.24, 0.36],
        "PVM_EpiEffBandwidth": 8333.33,
    }
    assert epsi_segment_count(method) == 4
    assert epsi_dwell_time_ms(method) == pytest.approx(0.12)
    assert epsi_segment_count({}) == 1


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ({"NSegments": 2, "PVM_EpiNShots": 4}, "disagree"),
        ({"NSegments": 0}, "positive integer"),
        ({"NSegments": 1.5}, "positive integer"),
        ({"NSegments": 2}, "PVM_EpiEchoSpacing"),
        (
            {"NSegments": 2, "PVM_EpiEchoSpacing": 0.4, "PVM_EpiEchoShiftA": [0, 0.1]},
            "PVM_EpiEchoShiftA",
        ),
        (
            {"NSegments": 2, "PVM_EpiEchoSpacing": 0.4, "PVM_EpiEffBandwidth": 1000},
            "PVM_EpiEffBandwidth",
        ),
    ],
)
def test_epsi_timing_rejects_inconsistent_parameters(method, message):
    with pytest.raises(ValueError, match=message):
        epsi_dwell_time_ms(method)


def _hybrid_row(n_read):
    """Return an asymmetric, positive read profile (image space)."""
    return 1.0 + np.arange(n_read) / n_read + (np.arange(n_read) == 3)


def test_odd_even_phase_correction_removes_injected_linear_phase():
    n_read, n_echoes = 16, 8
    position = np.arange(n_read) - n_read // 2
    injected = (0.4, -0.05)
    row = _hybrid_row(n_read)
    hybrid = np.tile(row, (2, n_echoes, 1)).astype(complex)
    hybrid[:, 1::2] *= np.exp(1j * (injected[0] + injected[1] * position))
    kspace = centered_fft(hybrid, (2,))

    corrected, provenance = correct_odd_even_echo_phase(
        kspace, echo_axis=1, read_axis=2
    )

    np.testing.assert_allclose(provenance["epsi_odd_even_phase_rad"], injected)
    np.testing.assert_allclose(
        centered_ifft(corrected, (2,)), np.tile(row, (2, n_echoes, 1)), atol=1e-9
    )


def test_epsi_interleaves_segments_into_one_spectral_fid():
    n_read, n_phase, n_segments, n_echoes = 8, 2, 2, 4
    n_spectral = n_segments * n_echoes
    read_kspace = centered_fft(_hybrid_row(n_read).astype(complex), (0,))
    # Real, positive spectral samples keep the odd/even phase estimate at zero.
    spectral = 1.0 + np.arange(n_spectral)
    phase_scale = np.array([1.0, 10.0])
    fid = []
    for phase in range(n_phase):
        for segment in range(n_segments):
            for echo in range(n_echoes):
                line = (
                    phase_scale[phase]
                    * spectral[echo * n_segments + segment]
                    * read_kspace
                )
                fid.extend(line[::-1] if echo % 2 else line)
    handler = SimpleNamespace(
        fid=np.asarray(fid),
        enc_matrix=(n_read, n_phase, 1),
        matrix=(n_read, n_phase, 1),
        n_echoes=n_spectral,
        n_receivers=1,
        n_repetitions=1,
        spatial_dimension="2D",
        method={
            "NSegments": n_segments,
            "PVM_EpiNEchoes": n_echoes,
            "PVM_EpiEchoSpacing": 0.48,
            "PVM_EncSteps1": [-1, 0],
        },
    )

    kspace = BrukerEpsiRawRecoStrategy().reshape_fid_to_kspace(handler=handler)

    expected = (
        phase_scale[:, None, None] * spectral[None, :, None] * read_kspace[None, None]
    )
    assert kspace.shape == (1, 1, 1, n_phase, n_spectral, 1, n_read)
    np.testing.assert_allclose(
        kspace[0, 0, 0, :, :, 0], expected, rtol=0, atol=1e-9 * abs(expected).max()
    )
    np.testing.assert_allclose(
        BrukerEpsiRawRecoStrategy()._echo_times_ms(handler), 0.24 * np.arange(8)
    )


def test_epsi_rejects_several_slices():
    handler = SimpleNamespace(
        fid=np.zeros(16, complex),
        enc_matrix=(2, 2, 2),
        n_echoes=2,
        n_receivers=1,
        n_repetitions=1,
        method={"NSegments": 1, "PVM_EpiNEchoes": 2, "PVM_EpiEchoSpacing": 0.5},
    )
    with pytest.raises(ValueError, match="one slice"):
        BrukerEpsiRawRecoStrategy().reshape_fid_to_kspace(handler=handler)


def test_epsi_raw_strategy_is_selected():
    strategy, _ = find_reconstruction("bruker", "<bruker:epsi>", "raw")
    assert strategy is BrukerEpsiRawRecoStrategy


def test_mge_kspace_has_one_echo_frequency_axis(monkeypatch):
    """The echo axis is stored once, as ``freq_axis_hz``, with no ``Hz`` copy.

    Three echoes at 2, 5 and 8 ms give a 3 ms spacing, so the axis is
    ``fftshift(fftfreq(3, 3e-3))`` in Hz, the convention of the other
    sequences; exact values, because both sides use the same NumPy call.
    """
    strategy = BrukerMGERawRecoStrategy()
    handler = _mge_handler("positiveEchoes")
    handler.method["EffectiveTE"] = [2.0, 5.0, 8.0]
    handler.rps_grid_mm = np.zeros((3, 4, 2, 2))
    monkeypatch.setattr(
        strategy,
        "set_kspace_xarray",
        lambda handler, data, dims, coords: xr.DataArray(
            data, dims=dims, coords=coords
        ),
    )
    kspace = strategy.reshape_fid_to_kspace(handler=handler)

    labelled = strategy._label_raw_kspace(handler, kspace)

    assert "Hz" not in labelled.coords
    np.testing.assert_array_equal(
        labelled["freq_axis_hz"].values,
        np.fft.fftshift(np.fft.fftfreq(3, d=3e-3)),
    )
    assert labelled["freq_axis_hz"].attrs["unit"] == "Hz"
