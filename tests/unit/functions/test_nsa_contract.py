"""Analytic NSA controls before separating calculation from notebook controls."""

import numpy as np
import pytest

from hypermri.analysis.spectroscopy.nsa import calculate_nsa, get_freq_hz


def test_single_species_nsa_is_echo_count_or_sum_of_squared_decay():
    times = np.arange(4) * 0.001
    before = times.copy()
    for decay_s in [-1, 0.02]:
        nsa, names, offsets = calculate_nsa(
            ["lactate", "none"], "lactate", 3.0, times, [decay_s, -1], False, True
        )
        expected = 4 if decay_s < 0 else np.exp(-2 * times / decay_s).sum()
        np.testing.assert_allclose(nsa, [expected], rtol=1e-12, atol=1e-12)
        assert names == ["Lactate"]
        assert offsets == ["0.0"]
    np.testing.assert_array_equal(times, before)


def test_orthogonal_reeder_species_each_retain_two_effective_averages():
    # Two echo vectors differ by pi radians for the stored -602 Hz offset.
    result, names, offsets = calculate_nsa(
        ["pyruvate", "lactate"],
        "lactate",
        3.0,
        np.array([0, 1 / 1204]),
        [-1, -1],
        False,
        True,
    )
    np.testing.assert_allclose(result, [2, 2], rtol=1e-12, atol=1e-12)
    assert names == ["Pyruvate", "Lactate"]
    assert offsets == ["-602.0", "0.0"]


def test_coupled_pyruvate_retains_weights_and_absorbs_hydrate_duplicates():
    times = np.array([0.0, 0.001, 0.002])
    result, names, offsets = calculate_nsa(
        ["pyruvate", "pyruvatehydrate", "pyruvate"],
        "lactate",
        3.0,
        times,
        [-1, -1, -1],
        True,
        True,
    )
    signal = 0.61 * np.exp(-2j * np.pi * 602 * times) + 0.39 * np.exp(
        -2j * np.pi * 242 * times
    )
    np.testing.assert_allclose(
        result, [np.sum(np.abs(signal) ** 2)], rtol=1e-12, atol=1e-12
    )
    assert names == ["Pyruvate (Cpl)"]
    assert offsets == ["Main: -602.0, Hyd: -242.0"]


@pytest.mark.parametrize(
    "species,couple", [(["none"], False), (["pyruvatehydrate"], True)]
)
def test_empty_active_design_retains_empty_result(species, couple):
    result, names, offsets = calculate_nsa(
        species, "lactate", 3.0, np.arange(3), [-1], couple, True
    )
    assert result.shape == (0,)
    assert names == offsets == []


def test_singular_design_retains_zero_nsa():
    result, _, _ = calculate_nsa(
        ["lactate", "lactate"],
        "lactate",
        3.0,
        np.array([0.0, 0.001]),
        [-1, -1],
        False,
        True,
    )
    np.testing.assert_array_equal(result, np.zeros(2))


def test_reeder_frequency_scales_field_and_subtracts_selected_center():
    assert get_freq_hz("pyruvate", 6.0, True, "alanine") == -784.0
    assert get_freq_hz("unknown", 3.0, False, "unknown") == 0.0


def test_dashboard_no_longer_exports_numerical_nsa_functions():
    """The dashboard aliases were removed on 15 November 2026."""
    import importlib
    import pickle

    dashboard = importlib.import_module("hypermri.visualization.nsa_dashboard")
    assert dashboard.__all__ == ["interactive_nsa_simulation"]
    for name, canonical in [
        ("calculate_nsa", calculate_nsa),
        ("get_freq_hz", get_freq_hz),
    ]:
        assert not hasattr(dashboard, name)
        assert pickle.loads(pickle.dumps(canonical)) is canonical


def test_nsa_analysis_import_needs_no_notebook_controls(package_subprocess_env):
    import subprocess
    import sys

    code = (
        "import sys; from hypermri.analysis.spectroscopy.nsa import calculate_nsa; "
        "assert not {'ipywidgets', 'IPython', 'hypermri.visualization.nsa_dashboard'} "
        ".intersection(sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env=package_subprocess_env,
        check=True,
        capture_output=True,
        text=True,
    )
