"""Protect legacy warning emission without changing solver results."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from joblib import parallel_config

from hypermri.analysis.fitting.ideal import IDEAL_Solver
from hypermri.analysis.fitting.phase_ideal import PHIDEALSolver


def _phase_source():
    # Invented eight-echo volume; echo times are milliseconds on the source.
    return SimpleNamespace(
        data=np.ones((8, 1, 1, 1), dtype=np.complex64),
        TEs=np.arange(8, dtype=float),
        _fmap=np.ones((1, 1, 1)) * 100.0,
        unwrapped_phase=np.zeros((8, 1, 1, 1)),
        mask_dict={"snr": np.zeros((1, 1, 1), dtype=bool)},
    )


@pytest.mark.parametrize("from_object", [False, True])
def test_phase_loading_warning_preserves_input_and_seconds(from_object):
    source = _phase_source()
    before = source.data.copy()
    message = "Loaded data from uMGE" if from_object else "Loaded separate data"
    with pytest.warns(UserWarning, match=message):
        if from_object:
            solver = PHIDEALSolver(uMGE_object=source)
        else:
            solver = PHIDEALSolver(
                signal_data=source.data,
                echo_timings=source.TEs / 1000.0,
                uMGE_object=source,
            )
    assert solver.signal_data is source.data
    np.testing.assert_array_equal(source.data, before)
    np.testing.assert_allclose(
        solver.echo_timings, source.TEs / 1000.0, rtol=1e-12, atol=1e-12
    )


def test_phase_field_map_initialization_warns_and_keeps_existing_selection():
    with pytest.warns(UserWarning, match="Loaded data"):
        solver = PHIDEALSolver(uMGE_object=_phase_source())
    with pytest.warns(UserWarning, match="Taking UMPIRE field map"):
        assert solver._PHIDEALSolver__check_init_frequency_estimates(None, True)


@pytest.mark.parametrize("t2_star", [None, 0.05])
def test_singular_phase_design_warns_and_retains_nan_failure_result(t2_star):
    with pytest.warns(UserWarning, match="Loaded data"):
        solver = PHIDEALSolver(uMGE_object=_phase_source())
    frequencies = np.zeros(2, dtype=float)
    before = frequencies.copy()
    with pytest.warns(UserWarning, match="Singular matrix"):
        result = solver.solve_one_voxel(
            np.ones(8, dtype=np.complex64), frequencies, t2_star_init=t2_star
        )
    assert len(result) == (5 if t2_star is None else 6)
    assert np.isnan(result).all()
    np.testing.assert_array_equal(frequencies, before)


@pytest.mark.parametrize(
    "selection,echoes,message",
    [
        (["Field_offset", "unknown"], 6, "Unknown transient unknown"),
        (["Field_offset", "Theta"], 3, "echoes are not enough"),
    ],
)
def test_ideal_option_warnings_keep_the_existing_voxel_result(
    monkeypatch, selection, echoes, message
):
    signal = np.ones((echoes, 1, 1, 1, 1), dtype=np.complex64)
    before = signal.copy()
    solver = IDEAL_Solver(np.array([0.0]), signal_data=signal, echo_spacing_s=0.001)
    # Isolate warning/orchestration behavior from the pre-existing invalid-option
    # numerical path. The invented voxel result has one discarded iteration field.
    reference = np.array([10.0, 0.01, 0.0, 2 + 3j, 1], dtype=np.complex128)
    monkeypatch.setattr(solver, "solve_one_voxel_parallel", lambda *a, **kw: reference)
    with pytest.warns(UserWarning, match=message):
        result = solver.solve(solve_for=selection, use_multiprocessing=False)
    np.testing.assert_array_equal(result[:, 0, 0, 0, 0], reference[:-1])
    np.testing.assert_array_equal(signal, before)


def test_phase_discarding_close_frequencies_warns_and_preserves_nan_positions(
    monkeypatch,
):
    source = _phase_source()
    with pytest.warns(UserWarning, match="Loaded data"):
        solver = PHIDEALSolver(max_num_frequencies=2, uMGE_object=source)
    reference = np.array([10.0, 11.0, 2 + 1j, 3 + 1j, 1], dtype=np.complex128)
    monkeypatch.setattr(solver, "solve_one_voxel", lambda **kw: reference.copy())
    monkeypatch.setattr(solver, "check_result", lambda *a, **kw: True)
    with parallel_config(backend="threading"):
        with pytest.warns(UserWarning, match="Calculated frequency values were close"):
            result = solver.solve(
                use_UMPRIRE_freq_for_init=False,
                frequ_inits=np.array([0.0, 10.0, 11.0]),
            )
    assert result.shape == (4, 1, 1, 1)
    np.testing.assert_array_equal(result[[0, 2], 0, 0, 0], reference[[0, 2]])
    assert np.isnan(result[[1, 3], 0, 0, 0]).all()


def test_no_warning_objects_are_silently_constructed_in_source():
    root = Path(__file__).resolve().parents[3] / "src" / "hypermri"
    unissued = []
    for file in root.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Warning"
                and not isinstance(parents[node], ast.Raise)
            ):
                unissued.append((str(file.relative_to(root)), node.lineno))
    assert unissued == []
