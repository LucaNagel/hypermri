"""Tests of the IDEAL tool of the browser viewer.

The fixture is a small invented noise-free two-component echo signal
(0 and 390 Hz, 8 echoes 0.8 ms apart, field offset 15 Hz). With only the
field offset fitted, amplitudes and offset are recovered to a relative
tolerance of 1e-6 and an absolute one of 1e-6 Hz.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.visualization.ideal_tool import (
    IdealToolState,
    echo_spacing_from_axis,
    ideal_maps,
    ideal_offered,
    ideal_options,
    ideal_settings,
    ideal_summary,
    run_ideal,
    set_ideal_option,
)

SERIAL = ExecutionConfig(use_multiprocessing=False)
FREQUENCIES_HZ = (0.0, 390.0)
SPACING_S = 0.0008
OFFSET_HZ = 15.0


def _echoes(n_echoes=8, unit="s", complex_values=True):
    """Return invented echoes on a 3 x 4 grid with amplitudes 1..12 and 2."""
    t = (np.arange(n_echoes) - (n_echoes - 1) / 2) * SPACING_S
    rho = np.zeros((3, 4, 2), complex)
    rho[..., 0] = np.arange(12).reshape(3, 4) + 1
    rho[..., 1] = 2.0
    phase = np.exp(
        2j * np.pi * (np.asarray(FREQUENCIES_HZ)[:, None] + OFFSET_HZ) * t[None, :]
    )
    signal = np.einsum("rpm,mt->rpt", rho, phase)
    if not complex_values:
        signal = np.abs(signal)
    scale = 1e3 if unit == "ms" else 1.0
    return xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "echo_idx"),
        coords={
            "echo_idx": (
                "echo_idx",
                np.arange(n_echoes) * SPACING_S * scale,
                {"unit": unit},
            )
        },
    )


def _state(**changes):
    state = IdealToolState()
    set_ideal_option(state, "component_names", "pyruvate, lactate")
    set_ideal_option(state, "frequencies_hz", "0, 390")
    for name, value in changes.items():
        set_ideal_option(state, name, value)
    return state


def test_offered_only_along_the_echo_axis():
    data = _echoes()
    assert ideal_offered(data, "echo_idx")
    assert not ideal_offered(data, "read_idx")
    assert not ideal_offered(data, None)
    assert ideal_settings(IdealToolState(), data, None) == {
        "offered": False,
        "message": "",
    }


def test_echo_spacing_comes_from_the_axis_in_s_or_ms():
    assert echo_spacing_from_axis(_echoes()) == pytest.approx(SPACING_S)
    assert echo_spacing_from_axis(_echoes(unit="ms")) == pytest.approx(SPACING_S)
    unequal = _echoes().assign_coords(
        echo_idx=("echo_idx", [0, 1, 2, 4, 5, 6, 7, 8], {"unit": "ms"})
    )
    assert echo_spacing_from_axis(unequal) is None
    no_unit = _echoes().assign_coords(echo_idx=np.arange(8))
    assert echo_spacing_from_axis(no_unit) is None


def test_ideal_recovers_amplitudes_and_field_offset():
    data = _echoes()
    state = _state(fit_theta=False)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = run_ideal(data, state, execution=SERIAL)
    maps = ideal_maps(result)
    assert set(maps.data_vars) == {"pyruvate", "lactate", "field_offset"}
    assert maps["pyruvate"].dims == ("read_idx", "phase_idx")
    np.testing.assert_allclose(
        maps["pyruvate"].values, np.arange(12).reshape(3, 4) + 1, rtol=1e-6
    )
    np.testing.assert_allclose(maps["lactate"].values, 2.0, rtol=1e-6)
    np.testing.assert_allclose(maps["field_offset"].values, OFFSET_HZ, atol=1e-6)
    assert "echo_idx" not in maps.coords
    summary = ideal_summary(result)
    assert "pyruvate, lactate in 12 voxels" in summary
    assert "15 Hz" in summary
    # The shown data are not changed.
    xr.testing.assert_identical(data, _echoes())


def test_options_use_entered_spacing_and_chosen_unknowns():
    options = ideal_options(_echoes(), _state(echo_spacing_ms=1.5, fit_theta=False))
    assert options.echo_spacing_s == pytest.approx(1.5e-3)
    assert options.solve_for == ("Field_offset",)
    assert options.relative_frequencies_hz == FREQUENCIES_HZ
    assert options.metabolite_names == ("pyruvate", "lactate")
    axis = ideal_options(_echoes(), _state())
    assert axis.echo_spacing_s == pytest.approx(SPACING_S)
    assert axis.solve_for == ("Field_offset", "Theta")


@pytest.mark.parametrize(
    ("data", "changes", "message"),
    [
        (_echoes(complex_values=False), {}, "complex"),
        (_echoes(), {"component_names": "pyruvate"}, "1 component names but 2"),
        (_echoes(n_echoes=1), {}, "at least as many echoes"),
        (_echoes(), {"fit_field_offset": False, "fit_theta": False}, "field offset"),
        (
            _echoes().assign_coords(echo_idx=np.arange(8)),
            {},
            "enter the echo spacing",
        ),
    ],
)
def test_options_name_the_problem(data, changes, message):
    with pytest.raises(ValueError, match=message):
        ideal_options(data, _state(**changes))


def test_options_need_frequencies():
    with pytest.raises(ValueError, match="frequencies in Hz"):
        ideal_options(_echoes(), IdealToolState())


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("component_names", "a, a", "differ"),
        ("component_names", "field_offset", "differ"),
        ("frequencies_hz", "0, pyruvate", "numbers in Hz"),
        ("frequencies_hz", "0, inf", "finite"),
        ("echo_spacing_ms", -1.0, "positive"),
        ("max_iterations", 3, "unknown setting"),
    ],
)
def test_invalid_settings_are_rejected(name, value, message):
    with pytest.raises(ValueError, match=message):
        set_ideal_option(IdealToolState(), name, value)


def test_settings_for_the_browser():
    state = _state(echo_spacing_ms="")
    assert state.echo_spacing_ms is None
    settings = ideal_settings(state, _echoes(), "echo_idx")
    assert settings["offered"] and settings["complex"]
    assert settings["component_names"] == "pyruvate, lactate"
    assert settings["frequencies_hz"] == "0, 390"
    assert settings["axis_spacing_ms"] == pytest.approx(0.8)
    assert settings["n_echoes"] == 8
