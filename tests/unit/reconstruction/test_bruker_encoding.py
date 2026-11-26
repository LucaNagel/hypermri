"""Independent exact k-space placement references for Bruker encoding.

Invented acquired lines identify each encoding step by an integer amplitude.
No Fourier reference or scanner agreement is inferred from these grid tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.reconstruction.strategies.bruker.rare import BrukerRareRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.raw import (
    BrukerMGERawRecoStrategy,
    BrukerRawRecoStrategy,
)


@pytest.mark.parametrize("mge", [False, True])
@pytest.mark.parametrize("spatial", ["2D", "3D"])
@pytest.mark.parametrize("target", [7, 8])
def test_asymmetric_phase_steps_occupy_declared_positions(mge, spatial, target):
    # Three acquired phase rows are deliberately asymmetric and not sorted.
    steps = np.array([2, 0, 1])
    phase2 = np.array([1, 0])
    echoes = 2 if mge else 1
    slices = 2
    shape = (1, 1, slices, 3, echoes, 1, 2) if mge else (1, 1, slices, 3, 1, 2)
    acquired = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
    acquired = (acquired + 1j * acquired).astype(np.complex64)
    before = acquired.copy()
    handler = SimpleNamespace(
        n_repetitions=1,
        n_receivers=1,
        n_echoes=echoes,
        enc_matrix=np.array([2, 3, slices]),
        matrix=np.array([4, target, 6 if spatial == "3D" else slices]),
        spatial_dimension=spatial,
        acqp={"ACQ_phase_factor": 3},
        method={"PVM_EncSteps1": steps, "PVM_EncSteps2": phase2},
    )
    strategy = BrukerMGERawRecoStrategy() if mge else BrukerRawRecoStrategy()
    actual = strategy.reshape_fid_to_kspace(acquired.ravel(), handler)
    expected = np.zeros(actual.shape, dtype=acquired.dtype)
    for s in range(slices):
        target_slice = phase2[s] + 3 if spatial == "3D" else s
        for p, step in enumerate(steps):
            expected[0, 0, target_slice, step + target // 2, ..., 1:3] = acquired[
                0, 0, s, p
            ]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(acquired, before)
    np.testing.assert_array_equal(
        strategy.reshape_fid_to_kspace(acquired.ravel(), handler), actual
    )
    assert actual.dtype == np.complex64


def test_rare_missing_steps_align_odd_acquired_centre_in_even_target():
    # The acquired DC index 1 must land at target DC index 4, hence start 3.
    actual = BrukerRareRawRecoStrategy._phase_target_indices(
        SimpleNamespace(method={}),
        final_phase_size=8,
        encoded_phase_size=3,
    )
    np.testing.assert_array_equal(actual, [3, 4, 5])


@pytest.mark.parametrize(
    "steps, message",
    [
        ([0, 1], "length"),
        ([0, 0, 1], "repeat"),
        ([0, 1, 0.5], "integer"),
        ([0, 1, np.nan], "integer"),
        ([0, 1, 4], "outside"),
        ([0, 1, 1e100], "outside"),
    ],
)
def test_generic_phase_placement_rejects_invalid_declared_steps(steps, message):
    handler = SimpleNamespace(
        n_repetitions=1,
        n_receivers=1,
        enc_matrix=[2, 3, 1],
        matrix=[2, 8, 1],
        spatial_dimension="2D",
        acqp={"ACQ_phase_factor": 1},
        method={"PVM_EncSteps1": steps},
    )
    source = np.ones(6, dtype=np.complex64)
    with pytest.raises(ValueError, match=f"PVM_EncSteps1.*{message}"):
        BrukerRawRecoStrategy().reshape_fid_to_kspace(source, handler)
    np.testing.assert_array_equal(source, np.ones(6, dtype=np.complex64))
