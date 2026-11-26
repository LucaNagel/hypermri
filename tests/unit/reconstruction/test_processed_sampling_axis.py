"""Pin the first time label of the processed spectroscopy sampling axis.

Invented handlers with a 0.5 ms dwell time and an acquired axis that starts
at 0.2 ms. The processed axis counts from the first kept sample, so its first
label is 0 s for every cut-off (decided on 13 November 2026); later
labels keep the acquired spacing. Comparisons use 1e-12 relative tolerance
for float64 rounding of the subtraction.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.reconstruction.axes import (
    processed_sampling_axis,
)

DWELL_S = 0.0005
ACQUIRED_AXIS_S = 0.0002 + np.arange(16) * DWELL_S


@pytest.mark.parametrize("cut_off", [0, 3])
@pytest.mark.parametrize(
    "handler",
    [
        SimpleNamespace(sampling_time_axis_s=ACQUIRED_AXIS_S, dwell_time_s=DWELL_S),
        SimpleNamespace(sampling_time_axis_s=None, dwell_time_s=DWELL_S),
    ],
    ids=["acquired-axis", "dwell-time"],
)
def test_axis_starts_at_zero_at_the_first_kept_sample(handler, cut_off):
    npoints = 16 - cut_off
    axis, unit = processed_sampling_axis(handler, cut_off, npoints)
    assert unit == "s"
    assert axis.shape == (npoints,)
    assert axis[0] == 0.0
    np.testing.assert_allclose(axis, np.arange(npoints) * DWELL_S, rtol=1e-12)


def test_axis_without_timing_falls_back_to_indices():
    handler = SimpleNamespace(sampling_time_axis_s=None, dwell_time_s=None)
    axis, unit = processed_sampling_axis(handler, 2, 5)
    assert unit == "index"
    np.testing.assert_array_equal(axis, np.arange(5))
