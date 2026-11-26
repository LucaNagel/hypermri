"""Check ``plot``, ``scan.plot`` and ``PlotOptions`` of the browser viewer.

The data are invented: one damped line per voxel whose amplitude grows with
the voxel position, so the strongest voxel is known. No scanner data are used.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from traitlets import TraitError

from hypermri.loading.experiment import MRIExperiment
from hypermri.visualization import PlotOptions, plot
from hypermri.visualization.state import PlotState

pytest.importorskip("anywidget")
from hypermri.visualization.scan_view import ScanView  # noqa: E402


def _csi(n_read=4, n_phase=5, n_slice=3, n_spec=32):
    time_s = np.arange(n_spec) * 2e-4
    line = np.exp(2j * np.pi * 300 * time_s - time_s / 0.03)
    scale = (
        1
        + np.arange(n_read)[:, None, None, None]
        + 10 * np.arange(n_phase)[None, :, None, None]
        + 100 * np.arange(n_slice)[None, None, :, None]
    )
    return xr.DataArray(
        (scale * line).astype(np.complex64),
        dims=("read_idx", "phase_idx", "slice_idx", "acquisition_idx"),
        attrs={"units": "a.u.", "signal_domain": "temporal"},
    )


def _scan(data):
    return SimpleNamespace(primary_data=data, is_spectroscopic=True)


def test_plot_returns_the_browser_view_at_the_strongest_voxel():
    view = plot(_csi())

    assert isinstance(view, ScanView)
    assert view.plot_height == 400
    indices = view.state.spatial.slice_indices
    assert (indices["read_idx"], indices["phase_idx"]) == (3, 4)


def test_plot_height_and_start_position_are_applied():
    options = PlotOptions(
        plot_height=520,
        slice_indices={"slice_idx": 2, "read_idx": 1, "phase_idx": 0},
    )

    view = plot(_csi(), options)

    assert view.plot_height == 520
    indices = view.state.spatial.slice_indices
    assert (indices["slice_idx"], indices["read_idx"], indices["phase_idx"]) == (
        2,
        1,
        0,
    )
    assert view.selection["row"] in (0, 1) and view.selection["column"] in (0, 1)


@pytest.mark.parametrize(
    ("slice_indices", "message"),
    [
        ({"echo_idx": 0}, "not a dimension"),
        ({"acquisition_idx": 0}, "not a dimension"),
        ({"slice_idx": 3}, "from 0 to 2"),
        ({"slice_idx": -1}, "from 0 to 2"),
        ({"slice_idx": 1.5}, "integer"),
    ],
)
def test_start_position_outside_the_data_is_rejected(slice_indices, message):
    with pytest.raises(ValueError, match=message):
        plot(_csi(), PlotOptions(slice_indices=slice_indices))


def test_too_small_plot_height_is_rejected():
    with pytest.raises(TraitError, match="160 or more"):
        plot(_csi(), PlotOptions(plot_height=100))


def test_plot_height_can_be_changed_on_the_view():
    view = plot(_csi())
    view.plot_height = 300
    assert view.plot_height == 300
    with pytest.raises(TraitError, match="too small"):
        view.plot_height = 159


@pytest.mark.parametrize(
    ("name", "hint"),
    [
        ("return_controller", "returns the ScanView"),
        ("enable_mask", "view.tool = 'masks'"),
        ("quick_fit", "view.tool = 'fit'"),
        ("anatomy_align", "coordinate_mode='xyz'"),
        ("state_overrides", "PlotState"),
        ("backend", "Plotly"),
    ],
)
def test_removed_options_name_their_replacement(name, hint):
    with pytest.raises(ValueError, match="no longer available") as error:
        PlotOptions(**{name: True})
    assert name in str(error.value) and hint in str(error.value)


def test_unknown_option_lists_the_valid_ones():
    with pytest.raises(TypeError, match="unknown option.*cmap.*plot_height"):
        PlotOptions(cmap="viridis")


def test_options_are_read_only_and_state_must_be_a_plot_state():
    options = PlotOptions(slice_indices={"slice_idx": 1})
    with pytest.raises(TypeError):
        options.slice_indices["slice_idx"] = 2
    with pytest.raises(ValueError, match="PlotState"):
        PlotOptions(state={"cmap": "viridis"})


def test_given_state_is_kept_by_the_view():
    state = PlotState()
    state.spatial.cmap = "viridis"

    view = plot(_csi(), PlotOptions(state=state))

    assert view.state is state
    assert view.state.spatial.cmap == "viridis"


def test_plot_rejects_something_other_than_plot_options():
    with pytest.raises(TypeError, match="PlotOptions"):
        plot(_csi(), {"plot_height": 500})


def test_scan_plot_takes_options_or_keywords_but_not_both():
    data = _csi()
    scan = _scan(data)

    view = MRIExperiment.plot(scan, plot_height=480, slice_indices={"slice_idx": 1})
    assert view.plot_height == 480
    assert view.state.spatial.slice_indices["slice_idx"] == 1
    assert view._data.equals(data)

    with pytest.raises(TypeError, match="not both"):
        MRIExperiment.plot(scan, options=PlotOptions(), plot_height=480)
    with pytest.raises(ValueError, match="returns the ScanView"):
        MRIExperiment.plot(scan, return_controller=True)
    with pytest.raises(TypeError, match="unknown option"):
        MRIExperiment.plot(scan, cmap="viridis")


def test_scan_shortcuts_open_their_tool():
    scan = _scan(_csi())

    assert MRIExperiment.create_mask(scan).tool == "masks"
    assert MRIExperiment.interactive_fit(scan).tool == "fit"
    with pytest.raises(ValueError, match="together"):
        MRIExperiment.interactive_fit(scan, fit_results=xr.Dataset())


def _tail_noise(data, sigma):
    spatial = [dim for dim in data.dims if dim != "acquisition_idx"]
    return xr.DataArray(
        np.full([data.sizes[dim] for dim in spatial], sigma, dtype=float),
        dims=spatial,
        attrs={"noise_source": "acquisition_tail"},
    )


def test_interactive_fit_attaches_tail_noise_without_changing_the_scan():
    data = _csi()
    scan = _scan(data)

    view = MRIExperiment.interactive_fit(scan, noise=_tail_noise(data, 0.5))

    shown = view._data
    np.testing.assert_array_equal(shown.noise_sigma_quadrature, 0.5)
    assert shown.noise_sigma_quadrature.attrs["noise_source"] == "acquisition_tail"
    xr.testing.assert_identical(shown.drop_vars("noise_sigma_quadrature"), data)
    assert "noise_sigma_quadrature" not in data.coords


def test_interactive_fit_rejects_invalid_noise_before_building_the_view(
    monkeypatch,
):
    data = _csi()

    def must_not_plot(*args, **kwargs):
        raise AssertionError("the view was built before the noise was checked")

    monkeypatch.setattr("hypermri.visualization.api.plot", must_not_plot)
    with pytest.raises(ValueError, match="positive"):
        MRIExperiment.interactive_fit(_scan(data), noise=_tail_noise(data, -1.0))
