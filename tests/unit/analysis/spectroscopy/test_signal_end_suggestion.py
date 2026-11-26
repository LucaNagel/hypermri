"""Suggested CSI ``signal_end`` from the acquisition-tail noise onset.

Fixtures are synthetic: a tone of constant amplitude in every acquired point
up to a known end, plus white complex Gaussian noise of unit sigma per
quadrature. The noise onset detector is specified to locate the end within
one acquired point when the in-band signal is far above the noise.
"""

import numpy as np
import pytest

from hypermri.analysis.spectroscopy import suggest_signal_end

DWELL_S = 1e-3
N_FID = 64


def _acquisition(signal_end, n_points=400, amplitude=20.0, seed=0, n_repetitions=1):
    rng = np.random.default_rng(seed)
    shape = (N_FID, n_points * n_repetitions)
    noise = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    t = np.arange(N_FID) * DWELL_S
    tone = amplitude * np.exp(2j * np.pi * 50.0 * t)[:, None]
    signal = np.tile(np.arange(n_points) < signal_end, n_repetitions)
    return noise + tone * signal[None, :]


@pytest.mark.parametrize(("signal_end", "seed"), [(150, 0), (220, 1), (300, 2)])
def test_suggestion_is_the_known_signal_end(signal_end, seed):
    matrix = _acquisition(signal_end, seed=seed)
    suggestion = suggest_signal_end(
        matrix, dwell_time_s=DWELL_S, bands_hz=[(0.0, 100.0)]
    )
    assert abs(suggestion - signal_end) <= 1


def test_margin_is_added_and_clamped_to_the_acquisition():
    matrix = _acquisition(150)
    base = suggest_signal_end(matrix, dwell_time_s=DWELL_S, bands_hz=[(0, 100)])
    assert (
        suggest_signal_end(
            matrix, dwell_time_s=DWELL_S, bands_hz=[(0, 100)], margin_points=20
        )
        == base + 20
    )
    assert (
        suggest_signal_end(
            matrix, dwell_time_s=DWELL_S, bands_hz=[(0, 100)], margin_points=10_000
        )
        == 400
    )


def test_each_repetition_block_gets_its_own_suggestion():
    first = _acquisition(150, n_points=300, seed=3)
    second = _acquisition(200, n_points=300, seed=4)
    matrix = np.concatenate([first, second], axis=1)
    suggestions = [
        suggest_signal_end(
            matrix,
            dwell_time_s=DWELL_S,
            bands_hz=[(0, 100)],
            n_repetitions=2,
            repetition_index=index,
        )
        for index in (0, 1)
    ]
    assert abs(suggestions[0] - 150) <= 1
    assert abs(suggestions[1] - 200) <= 1


def test_suggestion_does_not_mutate_the_acquisition():
    matrix = _acquisition(150)
    before = matrix.copy()
    suggest_signal_end(matrix, dwell_time_s=DWELL_S)
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"margin_points": -1}, "margin_points"),
        ({"margin_points": 1.5}, "margin_points"),
        ({"n_repetitions": 2, "repetition_index": 2}, "repetition_index"),
    ],
)
def test_invalid_suggestion_options_fail_with_actionable_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        suggest_signal_end(_acquisition(150), dwell_time_s=DWELL_S, **kwargs)


def test_acquisition_too_short_for_a_noise_tail_gives_no_suggestion():
    # 40 acquired points cannot hold the required 50 noise-only points.
    with pytest.raises(ValueError, match="min_noise_points"):
        suggest_signal_end(_acquisition(20, n_points=40), dwell_time_s=DWELL_S)
