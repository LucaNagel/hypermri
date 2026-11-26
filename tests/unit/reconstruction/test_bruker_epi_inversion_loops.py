"""Invented loop labels characterize T1 EPI inversion storage.

The opt-in phantom declares one slice, NI inversion objects and NR repetitions.
Integer fixture values identify every loop exactly; no scanner data are used.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.reconstruction.strategies.bruker.epi import BrukerEpiRawRecoStrategy


@pytest.mark.parametrize("repetitions", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 2])
def test_t1_epi_object_inversions_preserve_repetitions_and_receivers(
    repetitions, channels
):
    times = [20.0, 180.0, 720.0]
    handler = SimpleNamespace(
        method={"Method": "<Bruker:T1_EPI>"},
        n_slices=1,
        n_echoes=1,
        n_inversion_time=3,
        inversion_time_ms=times,
        b_values=[],
    )
    raw = (
        np.arange(4 * 6 * 3 * repetitions * channels)
        .reshape(4, 6, 3, repetitions, channels)
        .astype(complex)
    )
    original = raw.copy()
    result, dims = BrukerEpiRawRecoStrategy._semantic_2d_layout(handler, raw)
    assert dims == ["read_idx", "phase_idx", "slice_idx", "inversion_idx"] + (
        ["acquisition_idx"] if repetitions > 1 else []
    ) + ["channel_idx"]
    expected = raw[:, :, None, :, :, :]
    if repetitions == 1:
        expected = expected[:, :, :, :, 0, :]
    np.testing.assert_array_equal(result, expected)
    coords = BrukerEpiRawRecoStrategy._coords(handler, dims, result.shape)
    np.testing.assert_array_equal(coords["inversion_idx"][1], times)
    assert coords["inversion_idx"][2] == {"unit": "ms"}
    np.testing.assert_array_equal(raw, original)


def test_unrelated_epi_objects_do_not_become_inversions_by_size():
    handler = SimpleNamespace(
        method={"Method": "<Bruker:EPI>"},
        n_slices=1,
        n_echoes=1,
        n_inversion_time=3,
        b_values=[],
    )
    _, dims = BrukerEpiRawRecoStrategy._semantic_2d_layout(
        handler, np.ones((4, 6, 3, 1, 1))
    )
    assert "inversion_idx" not in dims
    assert "acquisition_idx" in dims
