"""Source validation and volume assembly, using invented signals."""

from types import SimpleNamespace

import numpy as np
import pytest
from joblib import parallel_config

from hypermri.analysis.fitting.phase_ideal import PHIDEALSolver


def _source(shape=(8, 2, 2, 1)):
    return SimpleNamespace(
        data=np.ones(shape, dtype=np.complex64),
        TEs=np.arange(shape[0], dtype=float),
        unwrapped_phase=np.zeros(shape),
        _fmap=np.full(shape[1:], 16.0),
        mask_dict={"snr": np.zeros(shape[1:], dtype=bool)},
    )


def _solver(source, **kwargs):
    with pytest.warns(UserWarning, match="Loaded"):
        return PHIDEALSolver(uMGE_object=source, **kwargs)


def test_standalone_array_requires_prepared_source():
    source = _source()
    with pytest.raises(ValueError, match="prepared uMGE_object"):
        PHIDEALSolver(signal_data=source.data, echo_timings=source.TEs / 1000)


@pytest.mark.parametrize("limit", [0, 4, True, 1.5])
def test_invalid_component_limits_are_rejected(limit):
    with pytest.warns(UserWarning, match="Loaded"):
        with pytest.raises(ValueError, match="max_num_frequencies must"):
            PHIDEALSolver(uMGE_object=_source(), max_num_frequencies=limit)


@pytest.mark.parametrize("limit", [1, 2])
def test_legacy_evaluation_rejects_unsupported_component_limits(limit):
    solver = _solver(_source(), max_num_frequencies=limit)
    with pytest.raises(ValueError, match="evaluation_solve requires.*=3"):
        solver.evaluation_solve()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("data", np.ones((8, 2, 2)), "data ordered echo"),
        ("TEs", None, "TEs in milliseconds"),
        ("TEs", np.arange(7), "one finite.*per echo"),
        ("TEs", np.full(8, np.nan), "one finite.*per echo"),
    ],
)
def test_invalid_source_metadata_is_actionable(field, value, message):
    source = _source()
    setattr(source, field, value)
    with pytest.raises(ValueError, match=message):
        PHIDEALSolver(uMGE_object=source)


def test_constructor_override_preserves_source_context_and_time_units():
    source = _source()
    override = 2 * source.data
    solver = _solver(source, signal_data=override, echo_timings=source.TEs / 1000)
    assert solver.uMGE is source
    assert solver.signal_data is override
    np.testing.assert_allclose(solver.echo_timings, source.TEs / 1000, atol=1e-12)
    np.testing.assert_array_equal(source.data, np.ones(source.data.shape))


@pytest.mark.parametrize("times", [np.arange(8), np.arange(7) / 1000])
def test_override_rejects_incompatible_echo_times(times):
    source = _source()
    with pytest.raises(ValueError, match="echo_timings must match"):
        PHIDEALSolver(uMGE_object=source, signal_data=source.data, echo_timings=times)


@pytest.mark.parametrize("method", ["solve", "evaluation_solve"])
@pytest.mark.parametrize("shape", [(7, 2, 2, 1), (8, 1, 2, 1), (8, 2, 2)])
def test_optional_signal_requires_same_echo_and_spatial_shape(method, shape):
    solver = _solver(_source())
    with pytest.raises(ValueError, match="optional_signal_data must match"):
        getattr(solver, method)(optional_signal_data=np.ones(shape))


@pytest.mark.parametrize(
    "field,value,method,message",
    [
        ("unwrapped_phase", None, "solve", "unwrapped_phase"),
        ("unwrapped_phase", np.zeros((8, 1, 1, 1)), "solve", "unwrapped_phase"),
        ("_fmap", None, "solve", "_fmap in Hz"),
        ("_fmap", np.ones((5, 1)), "evaluation_solve", "_fmap in Hz"),
        ("mask_dict", {"snr": np.zeros(5)}, "solve", "mask_dict"),
    ],
)
def test_operation_checks_only_required_source_fields(field, value, method, message):
    source = _source()
    setattr(source, field, value)
    solver = _solver(source)
    with pytest.raises(ValueError, match=message):
        getattr(solver, method)()


@pytest.mark.parametrize("method", ["solve", "evaluation_solve"])
def test_optional_signal_is_used_without_mutating_or_replacing_inputs(
    monkeypatch, method
):
    source = _source()
    before = {
        name: value.copy()
        for name, value in vars(source).items()
        if isinstance(value, np.ndarray)
    }
    solver = _solver(source)
    seen = []

    def voxel_solution(signal_data_one_voxel, init_frequency_estimates, **kwargs):
        seen.append(signal_data_one_voxel.copy())
        count = init_frequency_estimates.size
        return np.r_[init_frequency_estimates, np.ones(count), 1].astype(complex)

    monkeypatch.setattr(solver, "solve_one_voxel", voxel_solution)
    monkeypatch.setattr(solver, "check_result", lambda *args, **kwargs: True)
    override = 3 * source.data
    with parallel_config(backend="threading"):
        with pytest.warns(UserWarning, match="Solve with optional data"):
            first = getattr(solver, method)(
                optional_signal_data=override,
                use_UMPRIRE_freq_for_init=False,
                discard_similar=False,
            )
        assert all(np.all(signal == 3) for signal in seen)
        seen.clear()
        second = getattr(solver, method)(
            use_UMPRIRE_freq_for_init=False, discard_similar=False
        )
    assert all(np.all(signal == 1) for signal in seen)
    assert first.shape == second.shape
    assert solver.signal_data is source.data
    np.testing.assert_array_equal(override, np.full(source.data.shape, 3))
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(source, name), values)


@pytest.mark.parametrize("t2_star", [None, 0.05])
def test_evaluation_keeps_voxel_order_component_padding_masks_and_persistence(
    monkeypatch, tmp_path, t2_star
):
    source = _source((8, 2, 2, 2))
    markers = np.arange(1, 9).reshape(2, 2, 2)
    source.data[:] = markers
    source.mask_dict["snr"][1, 0, 1] = True
    solver = _solver(source)

    def voxel_solution(signal_data_one_voxel, init_frequency_estimates, **kwargs):
        count = init_frequency_estimates.size
        values = [*init_frequency_estimates, *([signal_data_one_voxel[0]] * count)]
        if t2_star is not None:
            values.append(t2_star)
        return np.array([*values, 7], dtype=complex)

    monkeypatch.setattr(solver, "solve_one_voxel", voxel_solution)
    path = tmp_path / "evaluation.npy"
    with pytest.warns(UserWarning, match="Taking UMPIRE"):
        result = solver.evaluation_solve(t2_star_init=t2_star, save_path=path)
    assert result.shape == (2, 2, 2, 3, 7 + (t2_star is not None))
    for idx in np.ndindex(markers.shape):
        if source.mask_dict["snr"][idx]:
            assert np.isnan(result[idx]).all()
        else:
            np.testing.assert_allclose(result[idx][0, 3:6], markers[idx], atol=1e-12)
            np.testing.assert_allclose(result[idx][:, -1], 7, atol=1e-12)
            if t2_star is not None:
                np.testing.assert_allclose(result[idx][:, -2], t2_star, atol=1e-12)
    np.testing.assert_array_equal(np.load(path), result)
    with pytest.warns(UserWarning, match="Taking UMPIRE"):
        repeated = solver.evaluation_solve(t2_star_init=t2_star)
    np.testing.assert_allclose(repeated, result, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_evaluation_recovers_analytic_three_frequency_signals():
    # Noiseless Fourier mixtures with exact initial frequencies. The 1e-8
    # tolerance allows double-precision linear algebra roundoff, not fit bias.
    times_s = np.arange(32) * 0.001
    frequencies = np.array([-90.7, 16.0, 55.9])
    amplitudes = np.array(
        [[2 + 0.1j, 1 + 0.5j, 0.3 - 0.1j], [1 + 0.2j, 2 + 0.3j, 0.5 - 0.2j]]
    )
    signal = np.exp(2j * np.pi * times_s[:, None] * frequencies) @ amplitudes.T
    source = _source((32, 2, 1, 1))
    source.data = signal[:, :, None, None]
    solver = _solver(source)
    result = solver.evaluation_solve(use_UMPRIRE_freq_for_init=False)
    np.testing.assert_allclose(
        result[:, 0, 0, 0, :3],
        np.broadcast_to(frequencies, (2, 3)),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        result[:, 0, 0, 0, 3:6], amplitudes, atol=1e-8, rtol=1e-8
    )
