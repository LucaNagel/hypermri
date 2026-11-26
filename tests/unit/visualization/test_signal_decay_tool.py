"""Signal-decay tool of the browser viewer (choices, curves, confirm).

Uses the invented eight-point quadrature-tone CSI fixture of
``test_signal_decay_viewer.py`` (3 x 4 voxels, 2 slices, 2 repetitions,
dwell time 1 ms). Curves are compared with ``signal_decay_curves`` at 1e-12
relative in complex128.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.signal_decay import signal_decay_curves
from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.sequences.base_spectroscopy import BaseSpectroscopy
from hypermri.visualization.scan_view import ScanView
from hypermri.visualization.signal_decay import _SignalDecayData, plot_signal_decay
from hypermri.visualization.signal_decay_tool import (
    SignalDecayState,
    confirm_signal_decay,
    new_signal_decay_state,
    set_signal_decay_option,
    signal_decay_filter_params,
    signal_decay_settings,
)


def _csi_data():
    tone = 20 * np.array([1, 1j, -1, -1j] * 2, dtype=complex)
    return xr.DataArray(
        np.broadcast_to(tone[:, None, None, None, None], (8, 3, 4, 2, 2)).copy(),
        dims=(
            "acquisition_idx",
            "read_idx",
            "phase_idx",
            "slice_idx",
            "repetition_idx",
        ),
        coords={
            "acquisition_idx": xr.DataArray(
                np.arange(8) * 0.001, dims="acquisition_idx", attrs={"units": "s"}
            ),
            "read_idx": xr.DataArray([4, 2, 0], dims="read_idx", attrs={"units": "mm"}),
            "phase_idx": xr.DataArray(
                [-3, 0, 3, 6], dims="phase_idx", attrs={"units": "mm"}
            ),
            "slice_idx": xr.DataArray([0, 5], dims="slice_idx", attrs={"units": "mm"}),
        },
        attrs={
            "signal_domain": "temporal",
            "domain": "spatial",
            "units": "a.u.",
            "dwell_time_s": 0.001,
            "invented_fixture": True,
        },
    )


def _sequence():
    read, phase = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    acquisition = CSIAcquisition(
        encoding_order="LINEAR_ENC LINEAR_ENC",
        matrix=(3, 4),
        fov_mm=(30.0, 40.0),
        fov_offset_mm=(0.0, 0.0, 0.0),
        read_positions=read.ravel(),
        phase_positions=phase.ravel(),
        n_points=8,
        dwell_time_s=0.001,
        reference_frequency_mhz=None,
        center_ppm=0.0,
    )
    sequence = SimpleNamespace(
        handler=SimpleNamespace(
            matrix=np.array([3, 4, 2]),
            get=lambda key, default=None: default,
            to_csi_acquisition=lambda: acquisition,
        ),
        n_acquisition=8,
        n_repetitions=2,
        flipangle_deg=8.0,
        is_spectroscopic=True,
        sampling_time_axis_s=np.arange(8) * 0.001,
        masks=xr.Dataset(),
    )
    sequence.filter_signal = Mock(
        wraps=BaseSpectroscopy.filter_signal.__get__(sequence)
    )
    return sequence


@pytest.fixture
def prepared():
    return _SignalDecayData(_sequence(), _csi_data())


@pytest.fixture
def view():
    sequence = _sequence()
    data = _csi_data()
    before = data.copy(deep=True)
    view = plot_signal_decay(sequence, data)
    yield view, sequence
    xr.testing.assert_identical(data, before)


def _message(view, **content):
    view._handle_custom_message(view, {"type": "signal_decay", **content}, [])


def test_new_state_starts_at_half_the_acquired_points(prepared):
    state = new_signal_decay_state(prepared)
    assert state.n_points == prepared.n_points
    assert state.signal_end == state.center_position == prepared.n_points // 2
    assert state.flipangle_deg == 8.0
    assert state.filter_type == "None"
    assert state.curve == "magnitude"

    banded = new_signal_decay_state(prepared, [(-300, -100), (100, 300)])
    assert banded.curve == "band_power"
    assert banded.bands_hz == "-300:-100, 100:300"
    with pytest.raises(ValueError):
        new_signal_decay_state(prepared, [(300, 100)])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("signal_end", -1),
        ("signal_end", 1000),
        ("signal_end", 2.5),
        ("center_position", "x"),
        ("decay_constant", -0.1),
        ("flipangle_deg", 91),
        ("sigma_per_mm", 0),
        ("filter_type", "Gauss"),
        ("spatial_filter", "box"),
        ("curve", "spectrum"),
        ("bands_hz", "100"),
        ("bands_hz", "300:100"),
        ("n_points", 3),
    ],
)
def test_invalid_options_raise_and_keep_the_state(name, value):
    state = SignalDecayState(n_points=8)
    before = SignalDecayState(n_points=8)
    with pytest.raises(ValueError, match="Signal decay|Spectral band"):
        set_signal_decay_option(state, name, value)
    assert state == before


def test_valid_options_are_converted():
    state = SignalDecayState(n_points=8)
    set_signal_decay_option(state, "signal_end", 5.0)
    set_signal_decay_option(state, "filter_type", "Hamming")
    set_signal_decay_option(state, "show_tail", 0)
    set_signal_decay_option(state, "bands_hz", " -10:10 ")
    assert state.signal_end == 5 and isinstance(state.signal_end, int)
    assert state.filter_type == "Hamming"
    assert state.show_tail is False
    assert state.bands_hz == "-10:10"


def test_filter_params_carry_only_the_settings_of_the_chosen_filter():
    state = SignalDecayState(n_points=8, signal_end=5, center_position=3)
    assert signal_decay_filter_params(state) == {
        "filter_type": "None",
        "remove_lowest_signal": False,
        "apply_flipangle_correction": False,
        "filter_axis": 1,
        "spatial_filter": None,
        "signal_end": 5,
    }
    state.filter_type = "Hamming"
    state.apply_flipangle_correction = True
    state.spatial_filter = "gaussian"
    params = signal_decay_filter_params(state, (2.0, 3.0))
    assert params["alpha"] == state.alpha
    assert params["center_position"] == 3
    assert params["flipangle_deg"] == state.flipangle_deg
    assert params["spatial_spacing_mm"] == (2.0, 3.0)
    assert params["spatial_filter"] is not None
    assert "signal_end" not in params


def test_settings_curves_match_the_analysis_function(prepared):
    state = new_signal_decay_state(prepared)
    settings = signal_decay_settings(state, prepared, 1)
    assert settings["offered"] and settings["can_confirm"]
    names = [curve["name"] for curve in settings["curves"]]
    assert names == ["Original", "Filtered", "Tail fit"]
    expected = signal_decay_curves(
        prepared.original_matrix,
        n_repetitions=prepared.n_repetitions,
        repetition_index=1,
        end_index=state.signal_end,
        quantity="magnitude",
    )
    np.testing.assert_allclose(
        np.array(settings["curves"][0]["y"], dtype=float),
        expected["magnitude"],
        rtol=1e-12,
    )

    failed = signal_decay_settings(state, prepared, 0, filter_error="broken")
    assert not failed["can_confirm"]
    assert [curve["name"] for curve in failed["curves"]] == ["Original", "Tail fit"]


def test_confirm_stores_signal_end_and_filter_on_the_scan(prepared):
    state = new_signal_decay_state(prepared)
    state.filter_type = "Hann"
    message = confirm_signal_decay(state, prepared)
    assert prepared.sequence.signal_end == state.signal_end
    assert prepared.sequence.filter_params == signal_decay_filter_params(state)
    assert "Hann" in message


def test_plot_signal_decay_offers_the_tool(view):
    view, _ = view
    assert isinstance(view, ScanView)
    assert view.tool == "signal_decay"
    assert view._signal_decay_tool["offered"]
    assert view._signal_decay_tool["curves"]


def test_tool_is_not_offered_in_ordinary_views():
    from hypermri.visualization.scan_view import plot_scan

    plain = plot_scan(_csi_data())
    assert plain._signal_decay_tool == {}
    assert plain.tool != "signal_decay"


def test_option_messages_refilter_and_confirm(view):
    view, sequence = view
    sequence.filter_signal.reset_mock()
    _message(view, action="option", name="filter_type", value="Hann")
    sequence.filter_signal.assert_called_once()
    assert view._signal_decay_tool["filter_type"] == "Hann"

    sequence.filter_signal.reset_mock()
    _message(view, action="option", name="show_filtered", value=False)
    _message(view, action="option", name="curve", value="cumulative")
    sequence.filter_signal.assert_not_called()

    _message(view, action="option", name="signal_end", value=99)
    assert "got 99" in view._signal_decay_tool["message"]

    _message(view, action="confirm")
    assert sequence.filter_params["filter_type"] == "Hann"
    assert "Confirmed" in view._signal_decay_tool["message"]


def test_filter_failure_disables_confirm(view):
    view, sequence = view
    sequence.filter_signal.side_effect = RuntimeError("no raw data")
    _message(view, action="option", name="filter_type", value="Hann")
    tool = view._signal_decay_tool
    assert not tool["can_confirm"]
    assert "no raw data" in tool["message"]
    _message(view, action="confirm")
    assert not hasattr(sequence, "filter_params")
    assert "cannot be confirmed" in view._signal_decay_tool["message"]
