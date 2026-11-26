"""Field map input of the legacy IDEAL solver.

The solver indexes ``field_map.seq2d`` by axis position.  The fixtures are
synthetic arrays; no recorded data is used.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.ideal import IDEAL_Solver, _field_map_array

LABELLED_DIMS = (
    "phase_idx",
    "echo_idx",
    "read_idx",
    "waterfat_idx",
    "slice_idx",
    "bvalue_idx",
)


def _numpy_seq2d():
    """Return values that differ in every voxel, ordered read, phase, slice, frame."""
    return np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4, 1)


def _labelled_seq2d():
    """Return the numpy fixture with the labelled layout of ``scanner_data``."""
    volume = xr.DataArray(
        _numpy_seq2d()[..., 0], dims=("read_idx", "phase_idx", "slice_idx")
    )
    for dim in set(LABELLED_DIMS) - set(volume.dims):
        volume = volume.expand_dims(dim)
    return volume.transpose(*LABELLED_DIMS)


def test_numpy_seq2d_is_returned_unchanged():
    seq2d = _numpy_seq2d()

    assert _field_map_array(SimpleNamespace(seq2d=seq2d)) is seq2d


def test_labelled_seq2d_is_ordered_by_name_whatever_the_dimension_order():
    labelled = _labelled_seq2d()
    assert labelled.dims == LABELLED_DIMS

    selected = _field_map_array(SimpleNamespace(seq2d=labelled))

    np.testing.assert_array_equal(selected, _numpy_seq2d())


def test_labelled_seq2d_with_a_further_long_dimension_is_rejected():
    labelled = xr.DataArray(
        np.zeros((3, 2, 4, 5)),
        dims=("read_idx", "phase_idx", "slice_idx", "echo_idx"),
    )

    with pytest.raises(ValueError, match=r"echo_idx.*length greater than 1"):
        _field_map_array(SimpleNamespace(seq2d=labelled))


def test_labelled_seq2d_without_a_spatial_dimension_is_rejected():
    labelled = xr.DataArray(np.zeros((3, 2)), dims=("read_idx", "phase_idx"))

    with pytest.raises(ValueError, match="lacks .*slice_idx"):
        _field_map_array(SimpleNamespace(seq2d=labelled))


# Synthetic recovery of the legacy solver.  Signal model of the solver:
# s_n = sum_m a_m exp(2 pi i (f_m + df) t_n) with t_n = -((N - 1 - 2 n) / 2) * dTE.
ECHO_SPACING_S = 1e-3
N_ECHOES = 6
RELATIVE_FREQUENCIES_HZ = np.array([0.0, 440.0])
AMPLITUDES = np.array([1.0 + 0.5j, 0.4 - 0.2j])
FIELD_OFFSET_HZ = 3.0
# Both amplitudes are recovered to rounding; the measured error is below 1e-12.
ATOL = 1e-9


def _echo_signal(field_offset_hz, shape):
    """Return ``(echoes, *shape)`` echoes of the two-species signal model."""
    t_n = -((N_ECHOES - 1 - 2 * np.arange(N_ECHOES)) / 2) * ECHO_SPACING_S
    echoes = (
        AMPLITUDES[None, :]
        * np.exp(
            2j
            * np.pi
            * (RELATIVE_FREQUENCIES_HZ[None, :] + field_offset_hz)
            * t_n[:, None]
        )
    ).sum(axis=1)
    return np.broadcast_to(
        echoes.reshape((-1,) + (1,) * len(shape)), (N_ECHOES, *shape)
    ).copy()


def _solver(signal):
    return IDEAL_Solver(
        RELATIVE_FREQUENCIES_HZ, signal_data=signal, echo_spacing_s=ECHO_SPACING_S
    )


@pytest.mark.parametrize("solve_for", [["Field_offset"], ["Field_offset", "Theta"]])
def test_solve_recovers_field_offset_and_amplitudes_of_every_voxel(solve_for):
    """The serial solve fails previously for numpy >= 2.1 and scrambled voxels."""
    signal = _echo_signal(FIELD_OFFSET_HZ, (3, 2, 2, 2))
    signal[:, 1, 0, 1, 1] *= 2  # one voxel with doubled amplitudes

    solution = _solver(signal).solve(
        use_multiprocessing=False, stop_f_offset_hz=1e-6, solve_for=solve_for
    )

    assert solution.shape == (len(solve_for) + 2, 3, 2, 2, 2)
    expected = np.concatenate([[FIELD_OFFSET_HZ], AMPLITUDES])
    expected_per_voxel = np.broadcast_to(
        expected.reshape(-1, 1, 1, 1, 1), (3, 3, 2, 2, 2)
    ).astype(complex)
    expected_per_voxel[1:, 1, 0, 1, 1] *= 2
    np.testing.assert_allclose(solution[-3:], expected_per_voxel, atol=ATOL)
    if "Theta" in solve_for:
        np.testing.assert_allclose(solution[0], 0, atol=ATOL)


def test_serial_and_job_based_solve_agree():
    signal = _echo_signal(FIELD_OFFSET_HZ, (3, 2, 2, 2))
    signal[:, 0, 1, 0, 0] *= 0.5j

    serial = _solver(signal).solve(
        use_multiprocessing=False, stop_f_offset_hz=1e-6, solve_for=["Field_offset"]
    )
    jobs = _solver(signal).solve(
        use_multiprocessing=True,
        number_of_cpu_cores=1,
        stop_f_offset_hz=1e-6,
        solve_for=["Field_offset"],
    )

    np.testing.assert_array_equal(serial, jobs)


def test_easy_solve_recovers_the_amplitudes_with_a_given_field_map():
    """A uniform field map of the true offset leaves only the amplitudes."""
    signal = _echo_signal(FIELD_OFFSET_HZ, (3, 2, 2, 1))
    field_map = SimpleNamespace(seq2d=np.full((2, 3, 2, 1), FIELD_OFFSET_HZ))

    solution = _solver(signal).easy_solve(field_map)

    assert solution.shape == (2, 3, 2, 2, 1)
    expected = np.broadcast_to(AMPLITUDES.reshape(2, 1, 1, 1, 1), solution.shape)
    np.testing.assert_allclose(solution, expected, atol=ATOL)
