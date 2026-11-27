"""Check the curve block contract and the browser view without a browser.

All arrays are invented: three damped lines per voxel whose amplitude encodes
the voxel position, so a displayed row/column can be traced back to its source
voxel. No scanner data or metadata are used.
"""

import copy
import re
from importlib.resources import files as resource_files
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from traitlets import TraitError

from hypermri.analysis.spectroscopy.peak_fit_setup import frequency_axis_hz
from hypermri.visualization.render_data import (
    PlotDataModel,
    apply_display_defaults,
    curve_range_mean,
)
from hypermri.visualization.state import PlotState

pytest.importorskip("anywidget")
from cmap import Colormap  # noqa: E402

from hypermri.visualization.layout import (  # noqa: E402
    BROWSER_PARTS,
    browser_layout,
    browser_script,
)
from hypermri.visualization.mask_editing import MASK_DIMS  # noqa: E402
from hypermri.visualization.scan_view import (  # noqa: E402
    ScanView,
    plot_scan,
    plot_side_by_side,
)
from hypermri.visualization.scan_view.side_by_side import (  # noqa: E402
    SIDE_BY_SIDE_CURVE_ASPECT,
    SIDE_BY_SIDE_CURVE_WIDTH_PX,
    _side_by_side_width,
)

DIMS = (
    "read_idx",
    "phase_idx",
    "slice_idx",
    "repetition_idx",
    "channel_idx",
    "acquisition_idx",
)


def _csi(n_read=5, n_phase=7, n_rep=3, n_spec=64):
    time_s = np.arange(n_spec) * 2e-4
    lines = sum(
        amplitude * np.exp(2j * np.pi * freq_hz * time_s - time_s / 0.03)
        for freq_hz, amplitude in ((-800, 1.0), (300, 0.4), (1200, 0.2))
    )
    scale = (
        1
        + np.arange(n_read)[:, None, None, None, None, None]
        + 10 * np.arange(n_phase)[None, :, None, None, None, None]
        + 100 * np.arange(n_rep)[None, None, None, :, None, None]
    )
    values = (scale * lines).astype(np.complex64)
    return xr.DataArray(
        values,
        dims=DIMS,
        attrs={"units": "a.u.", "signal_domain": "temporal", "invented_fixture": True},
    )


def _model(data):
    return PlotDataModel(SimpleNamespace(primary_data=data, is_spectroscopic=True))


def _state(rotation=0, part="Magnitude"):
    state = PlotState()
    state.spatial.rotation = rotation
    state.spectral.spec_part = part
    state.spatial.slice_indices = {
        "read_idx": 1,
        "phase_idx": 2,
        "slice_idx": 0,
        "repetition_idx": 1,
        "channel_idx": 0,
        "acquisition_idx": 10,
    }
    state.spectral.selected_spectral_index = 10
    return state


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("part", ["Magnitude", "Real", "Phase"])
def test_curve_block_matches_map_spectrum_and_source_indices(rotation, part):
    data = _csi()
    before = data.copy(deep=True)
    state = _state(rotation, part)
    payload = _model(data).get_render_payload(data, state, include_curve_block=True)

    block = payload.curve_block
    assert block.shape == (*payload.spatial_array.shape, data.sizes["acquisition_idx"])
    # The block keeps the processing precision of complex64 input.
    assert block.dtype == np.float32
    # Same preparation path: the map and the spectrum are exact slices.
    np.testing.assert_array_equal(block[:, :, 10], payload.spatial_array)
    row_dim, column_dim = payload.spatial_dims
    indices = payload.spatial_source_indices
    assert sorted(indices) == sorted(payload.spatial_dims)
    row = int(
        np.flatnonzero(indices[row_dim] == state.spatial.slice_indices[row_dim])[0]
    )
    column = int(
        np.flatnonzero(indices[column_dim] == state.spatial.slice_indices[column_dim])[
            0
        ]
    )
    np.testing.assert_array_equal(block[row, column], payload.spectral_y)
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_spatial_source_indices_follow_reversed_read_coordinates(rotation):
    # Display preprocessing shows descending read_idx coordinates ascending;
    # the source indices must still name the source voxel of each spectrum.
    plain = _csi()
    reversed_read = plain.assign_coords(
        read_idx=10.0 - 2.5 * np.arange(plain.sizes["read_idx"])
    )
    state = _state(rotation)
    reference = _model(plain).get_render_payload(plain, state, include_curve_block=True)
    payload = _model(reversed_read).get_render_payload(
        reversed_read, _state(rotation), include_curve_block=True
    )
    row_dim, column_dim = payload.spatial_dims
    ref_rows = list(reference.spatial_source_indices[row_dim])
    ref_columns = list(reference.spatial_source_indices[column_dim])
    for row, source_row in enumerate(payload.spatial_source_indices[row_dim]):
        for column, source_column in enumerate(
            payload.spatial_source_indices[column_dim]
        ):
            np.testing.assert_array_equal(
                payload.curve_block[row, column],
                reference.curve_block[
                    ref_rows.index(source_row), ref_columns.index(source_column)
                ],
            )


def test_curve_block_is_only_built_on_request():
    data = _csi()
    payload = _model(data).get_render_payload(data, _state())
    assert payload.curve_block is None
    # The map's source indices do not depend on the block.
    assert sorted(payload.spatial_source_indices) == sorted(payload.spatial_dims)


def test_spatial_slice_keeps_requested_spectral_dimension_last():
    data = _csi()
    model = _model(data)
    state = _state()
    plane = model.get_spatial_slice(
        data,
        state,
        spatial_dims=["phase_idx", "read_idx"],
        keep_dims=("acquisition_idx",),
    )
    assert plane.dims == ("phase_idx", "read_idx", "acquisition_idx")
    np.testing.assert_array_equal(
        plane.values,
        data.isel(slice_idx=0, repetition_idx=1, channel_idx=0).values.transpose(
            1, 0, 2
        ),
    )


def test_display_defaults_start_spectroscopic_view():
    data = _csi()
    state = PlotState()
    apply_display_defaults(state, data, _model(data))
    assert state.spatial.cmap == "turbo"
    assert state.spatial.range_mode == "Min/Max"
    assert state.spatial.interpolation is False
    assert state.spatial.slicing_dim == "slice_idx"
    assert state.spatial.slice_indices["read_idx"] == 2
    assert state.spatial.slice_indices["phase_idx"] == 3
    # The strongest voxel is the last repetition; its FFT peak sets the index.
    assert state.spatial.slice_indices["repetition_idx"] == 2
    assert (
        state.spectral.selected_spectral_index
        == (state.spatial.slice_indices["acquisition_idx"])
    )


def _view(data=None, state=None):
    data = _csi() if data is None else data
    state = _state() if state is None else state
    return ScanView(data, _model(data), state), data


def test_scan_view_sends_block_and_initial_selection():
    view, data = _view()
    n_rows, n_columns, n_spec = view._view["shape"]
    block = np.frombuffer(view._block, dtype=np.float32).reshape(
        n_rows, n_columns, n_spec
    )
    payload = _model(data).get_render_payload(data, _state(), include_curve_block=True)
    np.testing.assert_array_equal(block, payload.curve_block)
    assert [view._view["row_dim"], view._view["column_dim"]] == payload.spatial_dims
    row_dim = view._view["row_dim"]
    column_dim = view._view["column_dim"]
    selection = view.selection
    assert (
        view._view["row_indices"][selection["row"]]
        == view.state.spatial.slice_indices[row_dim]
    )
    assert (
        view._view["column_indices"][selection["column"]]
        == view.state.spatial.slice_indices[column_dim]
    )
    assert selection["curve_index"] == 10
    # Combined channels (the default) have no channel index to choose.
    assert view._view["other_dims"] == {"slice_idx": 1, "repetition_idx": 3}
    assert len(view._view["curve_x"]) == n_spec


def test_browser_selection_updates_state_without_recomputing(monkeypatch):
    view, _ = _view()
    block_before = view._block

    def fail(*_args, **_kwargs):
        raise AssertionError("a selection must not recompute the block")

    monkeypatch.setattr(view._model, "get_render_payload", fail)
    view.selection = {"row": 0, "column": 4, "curve_index": 20}
    indices = view.state.spatial.slice_indices
    assert indices[view._view["row_dim"]] == view._view["row_indices"][0]
    assert indices[view._view["column_dim"]] == view._view["column_indices"][4]
    assert view.state.spectral.selected_spectral_index == 20
    assert view.state.spectral.selected_spectral_value == view._view["curve_x"][20]
    assert view._block is block_before


def test_controls_recompute_block_for_new_repetition_and_part():
    view, data = _view()
    controls = dict(view._controls)
    controls["indices"] = {**controls["indices"], "repetition_idx": 2}
    controls["spec_part"] = "Real"
    view._controls = controls

    assert view.state.spatial.slice_indices["repetition_idx"] == 2
    assert view.state.spectral.spec_part == "Real"
    expected = _model(data).get_render_payload(
        data, view.state, include_curve_block=True
    )
    block = np.frombuffer(view._block, dtype=np.float32).reshape(view._view["shape"])
    np.testing.assert_array_equal(block, expected.curve_block)


def test_controls_reject_unknown_part():
    view, _ = _view()
    with pytest.raises(Exception, match="spec_part must be one of"):
        view._controls = {**view._controls, "spec_part": "Absolute"}


def test_plotly_request_is_answered_with_packaged_bundle(monkeypatch):
    view, _ = _view()
    sent = []
    monkeypatch.setattr(
        view, "send", lambda content, buffers=None: sent.append((content, buffers))
    )
    view._handle_custom_message(view, {"type": "request_plotly"}, [])
    assert sent[0][0] == {"type": "plotly"}
    assert sent[0][1][0][:200].lstrip().startswith(b"/**")


def test_plot_scan_accepts_scan_and_rejects_a_second_data_obj():
    data = _csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan)
    assert isinstance(view, ScanView)
    assert view.state.spatial.cmap == "turbo"
    assert view._view["spectroscopic"] is True

    with pytest.raises(ValueError, match="only valid with an explicit DataArray"):
        plot_scan(scan, scan=scan)


IMAGE_DIMS = ("read_idx", "phase_idx", "slice_idx")


def _image(extra=None):
    """Invented image whose value encodes read, phase, slice and extra index."""
    sizes = {"read_idx": 8, "phase_idx": 6, "slice_idx": 3}
    if extra is not None:
        sizes[extra] = 5
    grids = np.meshgrid(*(np.arange(n) for n in sizes.values()), indexing="ij")
    weights = (1, 10, 100, 1000)
    values = 1 + sum(w * g for w, g in zip(weights[: len(grids)], grids, strict=True))
    coords = {} if extra is None else {extra: 2.0 * np.arange(5)}
    image = xr.DataArray(values.astype(np.float32), dims=tuple(sizes), coords=coords)
    return SimpleNamespace(primary_data=image, is_spectroscopic=False)


def _block(view):
    return np.frombuffer(view._block, dtype=np.float32).reshape(view._view["shape"])


def _source_value(view, row, column, curve_index=0):
    """Return the invented value of the shown voxel, for any shown plane."""
    indices = dict(view.state.spatial.slice_indices)
    indices[view._view["row_dim"]] = view._view["row_indices"][row]
    indices[view._view["column_dim"]] = view._view["column_indices"][column]
    if view._curve_dim is not None:
        indices[view._curve_dim] = curve_index
    data = view._data
    return float(data.isel({dim: indices.get(dim, 0) for dim in data.dims}))


@pytest.mark.parametrize("extra", ["echo_idx", "repetition_idx"])
def test_image_shows_a_curve_along_echoes_or_repetitions(extra):
    view = plot_scan(_image(extra))

    assert view._curve_dim == extra
    assert view._view["spectroscopic"] is False
    assert view._view["curve_x"] == [0.0, 2.0, 4.0, 6.0, 8.0]
    block = _block(view)
    assert block.shape[2] == 5
    for row, column, k in [(0, 0, 0), (2, 3, 4), (5, 7, 1)]:
        assert block[row, column, k] == pytest.approx(
            _source_value(view, row, column, k)
        )
    assert view._time_dim is None

    view.selection = {"row": 1, "column": 2, "curve_index": 3}
    assert view.state.spatial.slice_indices[extra] == 3


def test_image_with_one_acquisition_point_is_shown_as_image():
    # Images loaded from a scan keep acquisition_idx with one entry; one
    # point is not a spectrum.
    image = _image().primary_data.expand_dims(acquisition_idx=1, axis=-1)

    view = plot_scan(SimpleNamespace(primary_data=image, is_spectroscopic=False))

    assert not view._spectroscopic
    assert view._curve_dim is None
    assert _block(view)[2, 3, 0] == pytest.approx(_source_value(view, 2, 3))


def test_image_without_curve_dimension_is_shown_as_image_only():
    view = plot_scan(_image())

    assert view._curve_dim is None
    assert view._view["curve_dim"] is None
    assert view._view["curve_label"] == ""
    assert view._view["other_dims"] == {"slice_idx": 3}
    block = _block(view)
    assert block.shape[2] == 1
    first_slice = view.state.spatial.slice_indices["slice_idx"]
    assert block[2, 3, 0] == pytest.approx(_source_value(view, 2, 3))

    view.selection = {"row": 2, "column": 3, "curve_index": 0}
    assert (
        view.state.spatial.slice_indices[view._view["row_dim"]]
        == (view._view["row_indices"][2])
    )
    other = (first_slice + 1) % 3
    view._controls = {**view._controls, "indices": {"slice_idx": other}}
    assert view.state.spatial.slice_indices["slice_idx"] == other
    assert _block(view)[2, 3, 0] == pytest.approx(_source_value(view, 2, 3))

    view.add_mask("roi")
    with pytest.raises(ValueError, match="needs a curve panel"):
        view.show_roi_mean("roi")


def _interleaved_image(sequence_type="User:fisp_grad_1metab_CS"):
    """Invented image whose five repetitions alternate between two metabolites."""
    scan = _image("repetition_idx")
    scan.type = sequence_type
    return scan


def _original_value(scan, view, row, column, repetition):
    """Return the invented value of a shown voxel at an unsplit repetition."""
    indices = dict(view.state.spatial.slice_indices)
    indices[view._view["row_dim"]] = view._view["row_indices"][row]
    indices[view._view["column_dim"]] = view._view["column_indices"][column]
    indices["repetition_idx"] = repetition
    data = scan.primary_data
    return float(data.isel({dim: indices[dim] for dim in data.dims}))


def test_interleaved_sequence_is_split_into_two_metabolites():
    scan = _interleaved_image()
    view = plot_scan(scan)

    # Five repetitions give two pairs; the last one has no partner.
    assert view._data.sizes["metabolite_idx"] == 2
    assert view._data.sizes["repetition_idx"] == 2
    assert scan.primary_data.sizes["repetition_idx"] == 5
    assert view._curve_dim == "repetition_idx"
    assert view._view["other_dims"]["metabolite_idx"] == 2
    for metabolite in (0, 1):
        view._controls = {
            **view._controls,
            "indices": {**view._controls["indices"], "metabolite_idx": metabolite},
        }
        block = _block(view)
        for row, column, k in [(0, 0, 0), (2, 3, 1), (5, 7, 1)]:
            assert block[row, column, k] == pytest.approx(
                _original_value(scan, view, row, column, 2 * k + metabolite)
            )


def test_split_metabolites_can_be_switched_off_or_forced():
    assert (
        "metabolite_idx"
        not in plot_scan(_interleaved_image(), split_metabolites=False)._data.dims
    )
    assert "metabolite_idx" not in plot_scan(_image("repetition_idx"))._data.dims
    forced = plot_scan(_image("repetition_idx"), split_metabolites=True)
    assert forced._data.sizes["metabolite_idx"] == 2
    with pytest.raises(ValueError, match="needs at least two entries along"):
        plot_scan(_image("echo_idx"), split_metabolites=True)


def test_split_metabolites_switch_changes_the_shown_data():
    scan = _interleaved_image()
    view = plot_scan(scan, split_metabolites=False)
    assert view._controls["split_metabolites"] is False

    # The browser checkbox sends the switch with the other controls.
    view._controls = {**view._controls, "split_metabolites": True}
    assert view._controls["split_metabolites"] is True
    assert view._data.sizes["metabolite_idx"] == 2
    assert view._view["other_dims"]["metabolite_idx"] == 2
    block = _block(view)
    assert block[2, 3, 1] == pytest.approx(_original_value(scan, view, 2, 3, 2))

    view.set_split_metabolites(False)
    assert "metabolite_idx" not in view._data.dims
    assert view._data.sizes["repetition_idx"] == 5
    assert view._controls["split_metabolites"] is False

    plain = plot_scan(_image("repetition_idx"))
    assert "split_metabolites" not in plain._controls
    with pytest.raises(ValueError, match="offers no metabolite split"):
        plain.set_split_metabolites(True)


def test_view_plane_control_shows_the_chosen_plane():
    view = plot_scan(_image())
    assert view._controls["view_plane"] == "slice_idx"
    assert [dim for _, dim in view._view["view_plane_options"]] == [
        "slice_idx",
        "read_idx",
        "phase_idx",
    ]

    # The browser menu sends the plane with the other controls.
    view._controls = {**view._controls, "view_plane": "read_idx"}
    assert view.state.spatial.slicing_dim == "read_idx"
    assert view._controls["view_plane"] == "read_idx"
    assert {view._view["row_dim"], view._view["column_dim"]} == {
        "phase_idx",
        "slice_idx",
    }
    assert "read_idx" in view._view["other_dims"]
    block = _block(view)
    for row, column in [(0, 0), (2, 1), (block.shape[0] - 1, block.shape[1] - 1)]:
        assert block[row, column, 0] == _source_value(view, row, column)

    with pytest.raises(ValueError, match="slicing_dim must be one of"):
        view.set_view_plane("repetition_idx")
    flat = plot_scan(_image().primary_data.isel(slice_idx=0))
    assert "view_plane" not in flat._controls


def test_display_switches_follow_the_state_and_the_browser():
    view = plot_scan(_image())
    # The switches show the state, which starts from the image defaults of plot.
    spatial, settings = view.state.spatial, view.state.global_settings
    assert view._display == {
        "smoothing": spatial.zsmooth_mode if spatial.interpolation else "off",
        "min_to_nan": spatial.min_to_nan,
        "show_scalebar": spatial.show_scalebar,
        "show_colorbar": spatial.show_colorbar,
        "show_grid_lines": settings.show_grid_lines,
        "show_tooltips": settings.show_tooltips,
        "show_crosshair": settings.show_crosshair,
        "map_cmap": spatial.cmap,
        "anatomy_cmap": view.state.anatomy.cmap,
        "show_slice_lines": False,
        "show_masks": True,
        "show_mask_outlines": True,
    }
    assert view._view["smoothing_options"] == ["off", "fast", "best"]

    # The browser controls send all switches at once; they only redraw.
    block = view._block
    view._display = {
        **view._display,
        "smoothing": "fast",
        "min_to_nan": False,
        "show_grid_lines": True,
        "show_tooltips": True,
        "show_crosshair": False,
    }
    assert (spatial.interpolation, spatial.zsmooth_mode) == (True, "fast")
    assert spatial.min_to_nan is False
    assert settings.show_grid_lines and settings.show_tooltips
    assert settings.show_crosshair is False
    assert view._block is block

    view.set_display(smoothing="off")
    assert spatial.interpolation is False
    assert view._display["smoothing"] == "off"
    with pytest.raises(ValueError, match="smoothing must be one of"):
        view.set_display(smoothing="cubic")


def test_the_colour_bar_starts_on_only_for_fitted_maps():
    # A signal map starts without its colour bar.
    view = plot_scan(_csi())
    assert view._display["show_colorbar"] is False
    view.set_display(show_colorbar=True)
    assert view.state.spatial.show_colorbar is True

    # Fitted maps need the scale of their values.
    fit = _fit_results(_csi(), with_coordinates=False)
    assert plot_scan(fit)._display["show_colorbar"] is True
    # A state given by the caller keeps its choice.
    assert plot_scan(fit, state=_state())._display["show_colorbar"] is False


def test_colormaps_of_map_and_anatomy_follow_the_state_and_the_browser():
    view = plot_scan(_stacked_anatomy_scan(), anatomy=_stacked_anatomy_scan())
    # The choices hold the current colormaps; an image over an anatomy is
    # "turbo" on a "bone" anatomy.
    assert ["turbo", "turbo"] in view._view["map_cmaps"]
    assert view._view["anatomy_cmaps"] == view._view["map_cmaps"]
    assert view._view["anatomy"]["colorscale"] == Colormap("bone").to_plotly()

    # The browser sends a new map colormap; Python sends the new colour scale.
    view._display = {**view._display, "map_cmap": "inferno"}
    assert view.state.spatial.cmap == "inferno"
    assert view._view["colorscale"] == Colormap("inferno").to_plotly()

    # From Python: the anatomy below gets its own colormap, the map keeps its.
    view.set_display(anatomy_cmap="viridis")
    assert view.state.anatomy.cmap == "viridis"
    assert view._display["anatomy_cmap"] == "viridis"
    assert view._view["anatomy"]["colorscale"] == Colormap("viridis").to_plotly()
    assert view._view["colorscale"] == Colormap("inferno").to_plotly()

    # A name outside the short list is offered once it is in use.
    view.set_display(map_cmap="cividis")
    assert ["cividis", "cividis"] in view._view["map_cmaps"]
    with pytest.raises(ValueError, match="unknown colormap map_cmap='no-such-map'"):
        view.set_display(map_cmap="no-such-map")
    assert view.state.spatial.cmap == "cividis"


def test_scalebar_needs_a_physical_voxel_spacing():
    # Index axes have no length: switching the scalebar on shows none.
    view = plot_scan(_image())
    view.set_display(show_scalebar=True)
    assert view.state.spatial.show_scalebar is True
    assert view._view["scalebar"] is None

    # Invented positions 2 mm apart (8 x 2 mm = 16 mm field of view).
    image = _image().primary_data
    image = image.assign_coords(
        {
            dim: (dim, 1.0 + 2.0 * np.arange(n), {"units": "mm"})
            for dim, n in image.sizes.items()
        }
    )
    view = plot_scan(SimpleNamespace(primary_data=image, is_spectroscopic=False))
    view.set_display(show_scalebar=True)
    bar = view._view["scalebar"]
    assert bar is not None
    assert bar["label"].endswith("mm")
    assert 0 < bar["length"] <= 16
    assert bar["color"] == "#FFFFFF"  # "w" in the state, as in plot
    view.set_display(show_scalebar=False)
    assert view._view["scalebar"] is None


def test_image_block_keeps_values_below_the_colour_minimum():
    # The browser hides them ("min -> NaN"), so lowering the minimum later
    # shows them again without Python.
    view = plot_scan(_image())
    view.set_display(min_to_nan=True)
    plane_min = float(_block(view).min())
    view.set_map_range("Absolute", vmin=plane_min + 30.0, vmax=plane_min + 60.0)
    view.set_rotation(90)
    block = _block(view)
    assert np.isfinite(block).all()
    assert block.min() == plane_min
    assert view.state.spatial.min_to_nan is True


def test_rotation_control_turns_the_image_and_keeps_voxel_indices():
    view = plot_scan(_image())
    assert view._controls["rotation"] == 0
    shape = _block(view).shape

    view._controls = {**view._controls, "rotation": 90}
    assert view.state.spatial.rotation == 90
    assert view._controls["rotation"] == 90
    block = _block(view)
    assert block.shape[:2] == shape[:2][::-1]
    for row, column in [(0, 0), (1, 4), (block.shape[0] - 1, block.shape[1] - 1)]:
        assert block[row, column, 0] == _source_value(view, row, column)

    view.set_rotation(180)
    assert _block(view).shape[:2] == shape[:2]
    with pytest.raises(ValueError, match="0, 90, 180 or 270"):
        view.set_rotation(45)


def test_split_view_shows_the_other_metabolite_at_the_selected_voxel():
    view = plot_scan(_interleaved_image())
    assert view._controls["other_metabolite"] is True
    selected = view._controls["indices"]["metabolite_idx"]
    other = view._other_metabolite
    assert other["name"] == f"metabolite_idx {1 - selected}"
    assert (other["row"], other["column"]) == (
        view.selection["row"],
        view.selection["column"],
    )
    row, column = other["row"], other["column"]
    shown = _block(view)[row, column]

    # The other curve equals the shown curve after moving the slider.
    indices = view._controls["indices"]
    view._controls = {
        **view._controls,
        "indices": {**indices, "metabolite_idx": 1 - selected},
    }
    np.testing.assert_allclose(_block(view)[row, column], other["y"], rtol=1e-6)
    assert view._other_metabolite["name"] == f"metabolite_idx {selected}"
    np.testing.assert_allclose(view._other_metabolite["y"], shown, rtol=1e-6)

    # A new voxel gets its own curve; the switch turns it off.
    view.selection = {**view.selection, "row": 1, "column": 2}
    assert (view._other_metabolite["row"], view._other_metabolite["column"]) == (1, 2)
    view._controls = {**view._controls, "other_metabolite": False}
    assert view._other_metabolite == {}
    view.set_split_metabolites(False)
    assert "other_metabolite" in view._controls
    assert view._other_metabolite == {}


def test_image_curve_along_slices_only_on_request():
    view = plot_scan(_image(), curve_dim="slice_idx")

    assert view._curve_dim == "slice_idx"
    assert _block(view).shape[2] == 3
    with pytest.raises(ValueError, match="curve_dim must be one of"):
        plot_scan(_image(), curve_dim="echo_idx")


def test_data_without_spatial_plane_are_rejected():
    line = xr.DataArray(
        np.ones((8, 5), dtype=np.float32), dims=("read_idx", "echo_idx")
    )

    with pytest.raises(ValueError, match="two-dimensional spatial plane"):
        plot_scan(SimpleNamespace(primary_data=line, is_spectroscopic=False))


def test_frontend_files_are_packaged():
    browser = resource_files("hypermri.visualization").joinpath("browser")
    for name in (*BROWSER_PARTS, "viewer.css"):
        assert browser.joinpath(name).is_file()


def test_view_loads_joined_browser_script_and_layout():
    view, _ = _view()

    assert view._esm == browser_script()
    assert view._layout == browser_layout()


def test_tool_menu_chooses_no_tool_or_a_listed_tool():
    view, _ = _view()

    assert view.tool == ""
    view.tool = "masks"
    assert view.tool == "masks"
    view.tool = "fit"
    assert view.tool == "fit"
    with pytest.raises(TraitError, match="unknown tool 'peaks'"):
        view.tool = "peaks"
    assert view.tool == "fit"
    view.tool = ""
    assert view.tool == ""


# ── masks ──────────────────────────────────────────────────────────────


def _voxel_square(view, row, column):
    """Return a square outline inside the displayed voxel (row, column)."""
    x = np.asarray(view._view["x"])
    y = np.asarray(view._view["y"])
    half_x = 0.3 * abs(x[1] - x[0])
    half_y = 0.3 * abs(y[1] - y[0])
    cx, cy = x[column], y[row]
    return [
        (cx - half_x, cy - half_y),
        (cx + half_x, cy - half_y),
        (cx + half_x, cy + half_y),
        (cx - half_x, cy + half_y),
    ]


def _with_descending_coords(data):
    return data.assign_coords(
        read_idx=10.0 - 2.5 * np.arange(data.sizes["read_idx"]),
        phase_idx=-4.0 + 1.5 * np.arange(data.sizes["phase_idx"]),
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("with_coords", [False, True])
def test_drawn_mask_lands_on_the_shown_source_voxel(rotation, with_coords):
    data = _csi()
    if with_coords:
        data = _with_descending_coords(data)
    view, _ = _view(data, _state(rotation))
    view.add_mask("roi")
    row, column = 1, 3
    view.draw_polygon(_voxel_square(view, row, column), mask="roi")

    mask = view.masks["roi"]
    source = {
        view._view["row_dim"]: view._view["row_indices"][row],
        view._view["column_dim"]: view._view["column_indices"][column],
    }
    covered = np.argwhere(mask.values > 0)
    assert covered.shape[0] == 1
    hit = dict(zip(mask.dims, covered[0], strict=True))
    assert hit["read_idx"] == source["read_idx"]
    assert hit["phase_idx"] == source["phase_idx"]
    assert mask.values.max() == pytest.approx(0.36, abs=1e-5)
    # The browser overlay shows the coverage at the drawn display position.
    plane = np.asarray(view._mask_planes[0]["plane"])
    assert np.argwhere(plane > 0).tolist() == [[row, column]]


def test_erase_and_undo_restore_previous_planes():
    view, _ = _view()
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    drawn = view.masks["roi"].values.copy()
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi", erase=True)
    assert not view.masks["roi"].values.any()
    assert view.undo_mask_edit() is True
    np.testing.assert_array_equal(view.masks["roi"].values, drawn)
    assert view.undo_mask_edit() is True
    assert not view.masks["roi"].values.any()
    assert view.undo_mask_edit() is False


@pytest.mark.parametrize("rotation", [0, 90])
def test_drawn_outline_is_kept_and_shown_where_it_was_drawn(rotation):
    view, _ = _view(_with_descending_coords(_csi()), _state(rotation))
    view.add_mask("roi")
    drawn = _voxel_square(view, 1, 3)
    view.draw_polygon(drawn, mask="roi")

    (outline,) = view.mask_outlines["roi"]
    centre = np.mean(outline["points"], axis=0)
    source = {
        view._view["row_dim"]: view._view["row_indices"][1],
        view._view["column_dim"]: view._view["column_indices"][3],
    }
    # The square is centred in its voxel: centre at data index + 0.5.
    np.testing.assert_allclose(
        centre, [source[dim] + 0.5 for dim in outline["dims"]], atol=1e-6
    )
    (shown,) = view._mask_planes[0]["outlines"]
    np.testing.assert_allclose(shown["x"], [p[0] for p in drawn], atol=1e-6)
    np.testing.assert_allclose(shown["y"], [p[1] for p in drawn], atol=1e-6)


def test_undo_and_remove_forget_the_drawn_outline():
    view, _ = _view()
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi", erase=True)
    assert [o["erase"] for o in view.mask_outlines["roi"]] == [False, True]
    # The square is erased completely: no boundary is left to show.
    assert view._mask_planes[0]["outlines"] == []
    view.undo_mask_edit()
    assert [o["erase"] for o in view.mask_outlines["roi"]] == [False]
    view.remove_mask("roi")
    assert view.mask_outlines == {}

    view.add_mask("roi")
    assert view.mask_outlines == {"roi": []}
    view.set_display(show_mask_outlines=False)
    assert view._display["show_mask_outlines"] is False
    assert view._show_mask_outlines is False


def test_mask_names_styles_and_outlines_are_validated():
    view, _ = _view()
    with pytest.raises(ValueError, match="non-empty"):
        view.add_mask("")
    view.add_mask("roi")
    with pytest.raises(ValueError, match="exists"):
        view.add_mask("roi")
    with pytest.raises(ValueError, match="colour"):
        view.set_mask_style("roi", color="red")
    with pytest.raises(ValueError, match="opacity"):
        view.set_mask_style("roi", alpha=1.5)
    with pytest.raises(ValueError, match="no mask named 'other'"):
        view.draw_polygon(_voxel_square(view, 0, 0), mask="other")
    with pytest.raises(ValueError, match="finite"):
        view.draw_polygon([(0, 0), (np.nan, 1), (1, 1)], mask="roi")
    with pytest.raises(ValueError, match="covers no voxel"):
        view.draw_polygon([(1e6, 1e6), (2e6, 1e6), (2e6, 2e6)], mask="roi")
    view.remove_mask("roi")
    assert list(view.masks.data_vars) == []


def test_masks_are_stored_on_the_scan_and_read_back():
    data = _csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = ScanView(data, PlotDataModel(scan), _state())
    view.add_mask("roi", color="#00ff00", alpha=0.6)
    view.draw_polygon(_voxel_square(view, 2, 2), mask="roi")

    stored = scan.masks
    assert isinstance(stored, xr.Dataset)
    assert stored["roi"].attrs["color"] == "#00ff00"
    assert stored["roi"].attrs["alpha"] == pytest.approx(0.6)
    assert stored["roi"].attrs["weighted_voxel_count"] == pytest.approx(0.36, abs=1e-5)

    reopened = ScanView(data, PlotDataModel(scan), _state())
    np.testing.assert_array_equal(reopened.masks["roi"].values, stored["roi"].values)
    assert reopened._mask_planes[0]["color"] == "#00ff00"
    assert reopened.mask_outlines == view.mask_outlines
    assert len(reopened._mask_planes[0]["outlines"]) == 1


def test_stored_mask_with_other_grid_is_rejected():
    data = _csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    scan.masks = xr.Dataset(
        {"roi": xr.DataArray(np.zeros((2, 2, 1), dtype=np.float32), dims=MASK_DIMS)}
    )
    with pytest.raises(ValueError, match="stored mask 'roi'"):
        ScanView(data, PlotDataModel(scan), _state())


def test_masks_survive_a_recompute_and_draw_only_on_the_shown_slice():
    data = xr.concat([_csi(), 2 * _csi()], dim="slice_idx")
    view, _ = _view(data)
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    controls = view._controls
    view._controls = {
        **controls,
        "indices": {**controls["indices"], "slice_idx": 1},
    }
    assert view.state.spatial.slice_indices["slice_idx"] == 1
    assert not np.asarray(view._mask_planes[0]["plane"]).any()
    view.draw_polygon(_voxel_square(view, 1, 1), mask="roi")
    per_slice = view.masks["roi"].sum(dim=["read_idx", "phase_idx"]).values
    np.testing.assert_allclose(per_slice, [0.36, 0.36], atol=1e-5)


def test_browser_mask_messages_and_errors(monkeypatch):
    view, _ = _view()
    sent = []
    monkeypatch.setattr(view, "send", sent.append)
    view._handle_custom_message(
        view, {"type": "mask", "action": "add", "mask": "roi", "color": "#0000ff"}, []
    )
    vertices = [list(point) for point in _voxel_square(view, 0, 1)]
    view._handle_custom_message(
        view,
        {"type": "mask", "action": "draw", "mask": "roi", "vertices": vertices},
        [],
    )
    assert view.masks["roi"].values.sum() == pytest.approx(0.36, abs=1e-5)
    view._handle_custom_message(
        view, {"type": "mask", "action": "add", "mask": "roi"}, []
    )
    assert sent == [{"type": "mask_error", "message": sent[0]["message"]}]
    assert "exists" in sent[0]["message"]
    view._handle_custom_message(
        view,
        {
            "type": "mask",
            "action": "draw",
            "mask": "roi",
            "vertices": vertices,
            "erase": True,
        },
        [],
    )
    assert not view.masks["roi"].values.any()
    # The browser has no undo button any more; the action is unknown.
    view._handle_custom_message(view, {"type": "mask", "action": "undo"}, [])
    assert "unknown mask action 'undo'" in sent[-1]["message"]


def _source_voxel(view, row, column):
    shown = view._view
    return {
        shown["row_dim"]: shown["row_indices"][row],
        shown["column_dim"]: shown["column_indices"][column],
    }


def test_roi_mean_follows_mask_edits_indices_and_removal():
    view, data = _view()
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    view.draw_polygon(_voxel_square(view, 2, 3), mask="roi")
    view.show_roi_mean("roi")
    voxels = [_source_voxel(view, 0, 0), _source_voxel(view, 2, 3)]

    def expected(repetition, chosen):
        # Both voxels carry the same coverage, so the weighted mean is the
        # plain mean of the per-voxel FID magnitudes (supplied samples, no FFT).
        selected = data.isel(repetition_idx=repetition, channel_idx=0, slice_idx=0)
        return np.mean([np.abs(selected.isel(v).values) for v in chosen], axis=0)

    curve = view.roi_mean
    np.testing.assert_allclose(curve.y, expected(1, voxels), rtol=1e-6)
    np.testing.assert_array_equal(curve.x, np.arange(data.sizes["acquisition_idx"]))
    assert curve.x_label == "acquisition_idx [idx]"
    assert curve.y_label == "Mean magnitude [a.u.]"
    assert view._roi_mean["mask"] == "roi"
    assert view._roi_mean["y"] == pytest.approx(list(curve.y))

    controls = view._controls
    view._controls = {
        **controls,
        "indices": {**controls["indices"], "repetition_idx": 2},
    }
    np.testing.assert_allclose(view.roi_mean.y, expected(2, voxels), rtol=1e-6)

    view.undo_mask_edit()
    np.testing.assert_allclose(view.roi_mean.y, expected(2, voxels[:1]), rtol=1e-6)

    view.remove_mask("roi")
    assert view.roi_mean is None
    assert view._roi_mean == {}


def test_roi_mean_of_empty_mask_sends_gaps_and_browser_messages(monkeypatch):
    view, _ = _view()
    sent = []
    monkeypatch.setattr(view, "send", sent.append)
    view.add_mask("empty")
    view._handle_custom_message(view, {"type": "roi_mean", "mask": "empty"}, [])
    assert np.isnan(view.roi_mean.y).all()
    assert all(value is None for value in view._roi_mean["y"])
    view._handle_custom_message(view, {"type": "roi_mean", "mask": "missing"}, [])
    assert sent[-1]["type"] == "mask_error"
    assert "show_roi_mean" in sent[-1]["message"]
    view._handle_custom_message(view, {"type": "roi_mean", "mask": None}, [])
    assert view.roi_mean is None


def _with_scanner_coordinates(data):
    """Attach invented x/y/z voxel centres; y runs against phase_idx."""
    grid_dims = ("read_idx", "phase_idx", "slice_idx")
    shape = tuple(data.sizes[dim] for dim in grid_dims)
    read, phase, _ = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return data.assign_coords(
        x=(grid_dims, 2.0 * read, {"unit": "mm"}),
        y=(grid_dims, 10.0 - 3.0 * phase, {"unit": "mm"}),
        z=(grid_dims, np.zeros(shape), {"unit": "mm"}),
    )


@pytest.mark.parametrize("rotation", [0, 90])
def test_masks_land_on_shown_voxel_in_scanner_coordinates(rotation):
    # In xyz mode the map axis order follows the scanner coordinates, so the
    # shown rows run against phase_idx here; masks must follow the same
    # shown-to-source mapping as the spectra.
    data = _with_scanner_coordinates(_csi())
    state = _state(rotation=rotation)
    state.spatial.coordinate_mode = "xyz"
    view, _ = _view(data, state)
    assert {view._view["x_label"], view._view["y_label"]} == {"x [mm]", "y [mm]"}
    view.add_mask("roi")
    for row, column in [(0, 0), (1, 3)]:
        view.draw_polygon(_voxel_square(view, row, column), mask="roi")
        stored = view.masks["roi"].isel(_source_voxel(view, row, column))
        np.testing.assert_allclose(float(stored.max()), 0.36, atol=1e-6)
    assert float(view.masks["roi"].sum()) == pytest.approx(0.72, abs=1e-5)
    shown = np.asarray(view._mask_planes[0]["plane"])
    np.testing.assert_allclose(shown[[0, 1], [0, 3]], 0.36, atol=1e-4)
    assert float(shown.sum()) == pytest.approx(0.72, abs=1e-3)


def _anatomy_scan():
    """Return an invented anatomy scan that covers more than the CSI plane."""
    grid_dims = ("read_idx", "phase_idx", "slice_idx")
    x_mm = -2.0 + 2.0 * np.arange(12)
    y_mm = -12.0 + 3.0 * np.arange(10)
    x, y, z = np.meshgrid(x_mm, y_mm, [0.0], indexing="ij")
    values = (x + 100.0 * y).astype(float)
    anatomy = xr.DataArray(
        values,
        dims=grid_dims,
        coords={
            "x": (grid_dims, x, {"unit": "mm"}),
            "y": (grid_dims, y, {"unit": "mm"}),
            "z": (grid_dims, z, {"unit": "mm"}),
        },
        attrs={"invented_fixture": True},
    )
    return SimpleNamespace(primary_data=anatomy, is_spectroscopic=False)


def test_plot_scan_draws_anatomy_under_the_map_in_scanner_coordinates():
    data = _with_scanner_coordinates(_csi())
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    anatomy = _anatomy_scan()
    view = plot_scan(scan, anatomy=anatomy, state=_state())
    assert view.state.spatial.coordinate_mode == "xyz"

    reference_state = _state()
    reference_state.spatial.coordinate_mode = "xyz"
    model = PlotDataModel(scan, anatomy_obj=anatomy)
    apply_display_defaults(reference_state, data, model)
    payload = model.get_render_payload(data, reference_state)
    layer = view._view["anatomy"]
    image = np.frombuffer(view._anatomy, dtype=np.float32).reshape(layer["shape"])
    np.testing.assert_allclose(image, payload.spatial_background, rtol=1e-6)
    assert layer["extent"] == pytest.approx(payload.background_extent)
    assert [layer["zmin"], layer["zmax"]] == pytest.approx(
        [np.nanmin(image), np.nanmax(image)]
        if payload.background_clim is None
        else list(payload.background_clim)
    )
    # The axes show the whole anatomy field of view, keeping their direction.
    x0, x1, y0, y1 = layer["extent"]
    assert sorted(view._view["x_range"]) == pytest.approx(sorted([x0, x1]))
    assert sorted(view._view["y_range"]) == pytest.approx(sorted([y0, y1]))
    assert view._anatomy_opacity == pytest.approx(view.state.anatomy.opacity)


def test_anatomy_opacity_is_kept_in_state_across_recomputes():
    data = _with_scanner_coordinates(_csi())
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan, anatomy=_anatomy_scan(), state=_state())
    view.set_anatomy_opacity(0.25)
    assert view.state.anatomy.opacity == pytest.approx(0.25)
    view._anatomy_opacity = 0.8  # as written by the browser slider
    assert view.state.anatomy.opacity == pytest.approx(0.8)
    view._controls = {**view._controls, "spec_part": "Real"}
    assert view._anatomy_opacity == pytest.approx(0.8)
    assert view._view["anatomy"]["shape"]
    for invalid in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="set_anatomy_opacity"):
            view.set_anatomy_opacity(invalid)


def test_anatomy_range_is_kept_in_state_and_sent_in_percent():
    data = _with_scanner_coordinates(_csi())
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan, anatomy=_anatomy_scan(), state=_state())
    layer = view._view["anatomy"]
    image = np.frombuffer(view._anatomy, dtype=np.float32)
    # The browser computes percentage limits from the slice's value range.
    assert layer["value_range"] == pytest.approx([np.nanmin(image), np.nanmax(image)])
    view.set_anatomy_range(10.0, 80.0)
    anatomy = view.state.anatomy
    assert (anatomy.range_mode, anatomy.vmin, anatomy.vmax) == (
        "Percentage",
        10.0,
        80.0,
    )
    assert view._anatomy_range == [10.0, 80.0]
    view._anatomy_range = [5.0, 95.0]  # as written by the browser slider
    assert (anatomy.vmin, anatomy.vmax) == (5.0, 95.0)
    view._controls = {**view._controls, "spec_part": "Real"}
    assert view._anatomy_range == [5.0, 95.0]
    low, high = view._view["anatomy"]["value_range"]
    assert [view._view["anatomy"]["zmin"], view._view["anatomy"]["zmax"]] == (
        pytest.approx([low + 0.05 * (high - low), low + 0.95 * (high - low)])
    )
    for invalid in ((50.0, 50.0), (-1.0, 50.0), (10.0, 101.0), (float("nan"), 50.0)):
        with pytest.raises(ValueError, match="anatomy range"):
            view.set_anatomy_range(*invalid)
    assert (anatomy.vmin, anatomy.vmax) == (5.0, 95.0)


def test_new_view_starts_map_and_anatomy_at_0_to_80_percent():
    data = _with_scanner_coordinates(_csi())
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan, anatomy=_anatomy_scan())
    for layer in (view.state.spatial, view.state.anatomy):
        assert (layer.range_mode, layer.vmin, layer.vmax) == (
            "Percentage",
            0.0,
            80.0,
        )
    assert view._anatomy_range == [0.0, 80.0]


def test_plot_scan_coordinate_mode_choice():
    data = _with_scanner_coordinates(_csi())
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan, anatomy=_anatomy_scan(), coordinate_mode="rps")
    assert view.state.spatial.coordinate_mode == "rps"
    plain = plot_scan(scan)
    assert plain.state.spatial.coordinate_mode == "rps"
    assert plain._view["anatomy"] == {}
    assert plain._anatomy == b""
    with pytest.raises(ValueError, match="coordinate_mode"):
        plot_scan(scan, coordinate_mode="scanner")


GRID_DIMS = ("read_idx", "phase_idx", "slice_idx")
METABOLITES = ("pyruvate", "lactate")


def _fit_results(data, *, with_coordinates):
    """Return invented fit results on the CSI grid.

    ``fit_amps`` names its voxel, metabolite and repetition
    (``1 + read + 10 * phase + 1000 * metabolite + 10000 * repetition``), so
    an overlay plane can be traced back to its source voxels.
    ``fit_spectra`` has spectra and is therefore not offered as a layer.
    """
    shape = tuple(data.sizes[dim] for dim in (*GRID_DIMS, "repetition_idx"))
    read, phase, _, repetition = np.meshgrid(
        *(np.arange(n) for n in shape), indexing="ij"
    )
    voxel = 1.0 + read + 10.0 * phase + 10000.0 * repetition
    amps = np.stack([voxel + 1000.0 * k for k in range(len(METABOLITES))], axis=-1)
    dims = (*GRID_DIMS, "repetition_idx", "metabolite")
    fit = xr.Dataset(
        {
            "fit_amps": (dims, amps),
            "fit_freqs": (dims, np.zeros_like(amps)),
            "fit_spectra": (
                (*GRID_DIMS, "repetition_idx", "acquisition_idx"),
                np.zeros((*shape, data.sizes["acquisition_idx"])),
            ),
        },
        coords={"metabolite": list(METABOLITES)},
        attrs={"invented_fixture": True},
    )
    if with_coordinates:
        fit = fit.assign_coords({name: data.coords[name] for name in ("x", "y", "z")})
    return fit


def _reconstructable_fit(data):
    """Return invented FID fit results that ``fit_results.hp`` reconstructs.

    One line at -800 Hz with T2* 30 ms whose amplitude is the voxel's
    ``1 + read + 10 * phase``, so the fitted spectrum of each voxel differs.
    """
    shape = tuple(data.sizes[dim] for dim in (*GRID_DIMS, "repetition_idx"))
    read, phase, _, _ = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    dims = (*GRID_DIMS, "repetition_idx", "metabolite")
    amps = (1.0 + read + 10.0 * phase)[..., None]
    return xr.Dataset(
        {
            "fit_amps": (dims, amps),
            "fit_freqs": (dims, np.full_like(amps, -800.0)),
            "fit_t2s": (dims, np.full_like(amps, 0.03)),
        },
        coords={"metabolite": ["pyruvate"]},
        attrs={
            "model_name": "FID",
            "sampling_time_axis_s": np.arange(data.sizes["acquisition_idx"]) * 2e-4,
            "invented_fixture": True,
        },
    )


def _plot_fit_curve(data, view, fit):
    """Return the "Fit" curve that plot shows for the view's selected voxel."""
    payload = _model(data).get_render_payload(
        data, view.state, overlays={"overlay": fit}
    )
    return payload.overlays["Fit"]


def test_fitted_spectrum_of_the_selected_voxel_is_drawn_as_in_plot():
    data = _csi()
    fit = _reconstructable_fit(data)
    view = plot_scan(data, overlay=fit, state=_state())

    sent = view._fit_curves
    assert (sent["row"], sent["column"]) == (
        view.selection["row"],
        view.selection["column"],
    )
    # The residual and its zero line follow the fit, as in plot.
    names = [curve["name"] for curve in sent["curves"]]
    assert names[0] == "Fit" and names[1].startswith("Residual") and len(names) == 3
    np.testing.assert_allclose(sent["x"], view._curve_x)
    first = np.asarray(sent["curves"][0]["y"])
    np.testing.assert_allclose(first, _plot_fit_curve(data, view, fit), rtol=1e-6)

    # Another voxel: its own fitted spectrum, scaled by its amplitude.
    old = dict(view.state.spatial.slice_indices)
    target = {"row": 0, "column": 0, "curve_index": 3}
    view.selection = target
    new = view.state.spatial.slice_indices
    sent = view._fit_curves
    assert (sent["row"], sent["column"]) == (0, 0)
    second = np.asarray(sent["curves"][0]["y"])
    np.testing.assert_allclose(second, _plot_fit_curve(data, view, fit), rtol=1e-6)
    amplitude = lambda idx: 1 + idx["read_idx"] + 10 * idx["phase_idx"]  # noqa: E731
    np.testing.assert_allclose(
        second.max() / first.max(), amplitude(new) / amplitude(old), rtol=1e-6
    )


def test_roi_mean_overlays_mean_spectrum_and_mean_fitted_spectrum():
    data = _csi()
    fit = _reconstructable_fit(data)
    view = plot_scan(data, overlay=fit, state=_state())
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    view.draw_polygon(_voxel_square(view, 2, 3), mask="roi")
    view.show_roi_mean("roi")
    sent = view._roi_mean

    # Both voxels are fully covered, so the weighted means are plain means of
    # what the voxel curve panel shows for each of them.
    shown = view._plane_curves
    np.testing.assert_allclose(
        sent["shown"], (shown[0, 0] + shown[2, 3]) / 2, rtol=1e-6
    )
    voxel_fits = []
    for row, column in ((0, 0), (2, 3)):
        state = copy.deepcopy(view.state)
        state.spatial.slice_indices.update(_source_voxel(view, row, column))
        payload = _model(data).get_render_payload(
            data, state, overlays={"overlay": fit}
        )
        voxel_fits.append(payload.overlays["Fit"])
    assert [curve["name"] for curve in sent["fits"]] == ["Fit (mean)"]
    np.testing.assert_allclose(sent["fit_x"], view._curve_x)
    np.testing.assert_allclose(
        sent["fits"][0]["y"], np.mean(voxel_fits, axis=0), rtol=1e-6
    )
    # The mean FID magnitude of plot stays, and the selected voxel is unchanged.
    assert sent["y"] == pytest.approx(list(view.roi_mean.y))
    assert view.state.spatial.slice_indices == _state().spatial.slice_indices

    view.state.spectral.show_fit = False
    view.show_roi_mean("roi")
    assert "fits" not in view._roi_mean
    assert "shown" in view._roi_mean


def test_no_fitted_spectrum_without_fit_results_or_with_show_fit_off():
    data = _csi()
    assert plot_scan(data, state=_state())._fit_curves == {}

    state = _state()
    state.spectral.show_fit = False
    assert (
        plot_scan(data, overlay=_reconstructable_fit(data), state=state)._fit_curves
        == {}
    )


def _overlay_plane(view):
    layer = view._view["overlay"]
    return np.frombuffer(view._overlay, dtype=np.float32).reshape(layer["shape"])


def _expected_fit_plane(view, *, metabolite=0, repetition=1):
    """Return fit_amps of the source voxels behind every shown map pixel."""
    row_dim, column_dim = view._view["row_dim"], view._view["column_dim"]
    expected = np.empty(view._view["shape"][:2])
    for row, row_index in enumerate(view._view["row_indices"]):
        for column, column_index in enumerate(view._view["column_indices"]):
            index = {"read_idx": 0, "phase_idx": 0, row_dim: row_index}
            index[column_dim] = column_index
            expected[row, column] = (
                1.0
                + index["read_idx"]
                + 10.0 * index["phase_idx"]
                + 1000.0 * metabolite
                + 10000.0 * repetition
            )
    return expected


@pytest.mark.parametrize("coordinate_mode", ["xyz", "rps"])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_plot_scan_draws_fit_overlay_on_the_shown_voxels(rotation, coordinate_mode):
    data = _with_scanner_coordinates(_csi())
    fit = _fit_results(data, with_coordinates=True)
    view = plot_scan(
        data,
        overlay=fit,
        coordinate_mode=coordinate_mode,
        state=_state(rotation=rotation),
    )

    assert view.state.overlay.active_variable == "overlay_fit_amps"
    options = view._view["overlay_options"]
    assert options["variables"] == [
        ["fit_amps", "overlay_fit_amps"],
        ["fit_freqs", "overlay_fit_freqs"],
    ]
    assert options["metabolites"] == list(METABOLITES)
    np.testing.assert_array_equal(_overlay_plane(view), _expected_fit_plane(view))
    layer = view._view["overlay"]
    assert layer["x"] == pytest.approx(view._view["x"])
    assert layer["y"] == pytest.approx(view._view["y"])
    # Percentage 0-100 % of the whole pyruvate layer over all repetitions.
    pyruvate = fit["fit_amps"].sel(metabolite="pyruvate").values
    assert [layer["zmin"], layer["zmax"]] == pytest.approx(
        [pyruvate.min(), pyruvate.max()]
    )
    assert layer["colorscale"]
    assert view._overlay_opacity == pytest.approx(1.0)
    assert view.state.overlay.alpha == pytest.approx(1.0)
    # Same plane as the shared preparation used by plot.
    model = view._model
    payload = model.get_render_payload(
        data, view.state, spatial_overlay=view._overlay_layer()
    )
    np.testing.assert_array_equal(
        _overlay_plane(view), payload.spatial_overlays["overlay_fit_amps"]
    )


def test_overlay_without_scanner_coordinates_inherits_the_csi_grid():
    data = _with_scanner_coordinates(_csi())
    fit = _fit_results(data, with_coordinates=False)
    index_coords = {dim: np.arange(data.sizes[dim]) for dim in GRID_DIMS}
    view = plot_scan(
        data.assign_coords(index_coords),
        overlay=fit.assign_coords(index_coords),
        state=_state(),
    )
    assert view.state.spatial.coordinate_mode == "xyz"
    np.testing.assert_array_equal(_overlay_plane(view), _expected_fit_plane(view))


def test_overlay_controls_change_layer_metabolite_range_and_visibility():
    data = _csi()
    view = plot_scan(
        data,
        overlay=_fit_results(data, with_coordinates=False),
        coordinate_mode="rps",
        state=_state(),
    )

    overlay = view._controls["overlay"]
    view._controls = {**view._controls, "overlay": {**overlay, "metabolite": "lactate"}}
    assert view.state.overlay.active_metabolite == "lactate"
    np.testing.assert_array_equal(
        _overlay_plane(view), _expected_fit_plane(view, metabolite=1)
    )
    lactate = _fit_results(data, with_coordinates=False)["fit_amps"].sel(
        metabolite="lactate"
    )
    low, high = float(lactate.min()), float(lactate.max())

    overlay = view._controls["overlay"]
    view._controls = {**view._controls, "overlay": {**overlay, "vmin": 25, "vmax": 75}}
    layer = view._view["overlay"]
    assert [layer["zmin"], layer["zmax"]] == pytest.approx(
        [low + 0.25 * (high - low), low + 0.75 * (high - low)]
    )

    # Another range mode starts from the whole layer, here in overlay units.
    view.set_overlay(range_mode="Min/Max")
    assert (view.state.overlay.vmin, view.state.overlay.vmax) == pytest.approx(
        (low, high)
    )
    view.set_overlay(vmin=low + 5.0)
    assert view._view["overlay"]["zmin"] == pytest.approx(low + 5.0)
    assert view._controls["overlay"]["vmin"] == pytest.approx(low + 5.0)

    # The browser sends a new range mode without limits (browser/controls_overlay.js), so the
    # limits restart from the whole layer instead of keeping the old numbers.
    overlay = view._controls["overlay"]
    view._controls = {
        **view._controls,
        "overlay": {**overlay, "range_mode": "Percentage", "vmin": None, "vmax": None},
    }
    assert (view.state.overlay.vmin, view.state.overlay.vmax) == (0.0, 100.0)
    # The completed limits go back to the browser for its number inputs.
    assert view._controls["overlay"]["vmin"] == 0.0
    assert view._controls["overlay"]["vmax"] == 100.0

    view.set_overlay("overlay_fit_freqs", range_mode="Min/Max")
    assert (view.state.overlay.vmin, view.state.overlay.vmax) == (0.0, 0.0)
    np.testing.assert_array_equal(_overlay_plane(view), 0.0)

    view.set_overlay_opacity(0.3)
    assert view.state.overlay.alpha == pytest.approx(0.3)
    view._overlay_opacity = 0.7  # as written by the browser slider
    view.set_overlay(visible=False)
    assert view._overlay == b""
    assert view._view["overlay"] == {}
    assert view._overlay_opacity == pytest.approx(0.7)
    view.set_overlay(visible=True)
    assert view._view["overlay"]["shape"]


def test_overlay_settings_are_validated():
    data = _csi()
    view = plot_scan(
        data, overlay=_fit_results(data, with_coordinates=False), state=_state()
    )
    with pytest.raises(ValueError, match="cannot draw layer 'overlay_fit_spectra'"):
        view.set_overlay("overlay_fit_spectra")
    with pytest.raises(ValueError, match="no metabolite 'alanine'"):
        view.set_overlay(metabolite="alanine")
    with pytest.raises(ValueError, match="range_mode must be one of"):
        view.set_overlay(range_mode="Absolute")
    with pytest.raises(ValueError, match="vmax must be a finite number"):
        view.set_overlay(vmax=float("nan"))
    with pytest.raises(ValueError, match="set_overlay_opacity"):
        view.set_overlay_opacity(1.5)
    plain = plot_scan(data, state=_state())
    assert plain._view["overlay"] == {}
    assert plain._view["overlay_options"] == {}
    assert plain._controls["overlay"] is None
    with pytest.raises(ValueError, match="no overlay was given"):
        plain.set_overlay("overlay_fit_amps")


def test_plot_scan_rejects_overlays_it_cannot_draw():
    data = _csi()
    spectra = xr.DataArray(
        np.zeros((5, 7, 1, data.sizes["acquisition_idx"])),
        dims=(*GRID_DIMS, "acquisition_idx"),
    )
    with pytest.raises(ValueError, match="maps without acquisition_idx"):
        plot_scan(data, overlay=spectra, state=_state())
    state = _state()
    state.overlay.mode = "subtract"
    with pytest.raises(ValueError, match="mode 'normal' only"):
        plot_scan(data, overlay=_fit_results(data, with_coordinates=False), state=state)


def test_default_layer_with_spectra_falls_back_to_a_map():
    data = _csi()
    spectra = xr.DataArray(
        np.zeros((5, 7, 1, data.sizes["acquisition_idx"])),
        dims=(*GRID_DIMS, "acquisition_idx"),
    )
    voxel_map = xr.DataArray(np.ones((5, 7, 1)), dims=GRID_DIMS)
    view = plot_scan(
        data, overlay={"spectra": spectra, "map": voxel_map}, state=_state()
    )
    assert view.state.overlay.active_variable == "map"
    assert view._view["overlay_options"]["variables"] == [["map", "map"]]


def test_overlay_layer_is_prepared_once_per_layer_selection(monkeypatch):
    data = _csi()
    view = plot_scan(
        data, overlay=_fit_results(data, with_coordinates=False), state=_state()
    )
    calls = []
    reduce = view._model.reduce_spatial_overlay

    def counting_reduce(*args, **kwargs):
        calls.append(1)
        return reduce(*args, **kwargs)

    monkeypatch.setattr(view._model, "reduce_spatial_overlay", counting_reduce)
    controls = view._controls
    view._controls = {
        **controls,
        "indices": {**controls["indices"], "repetition_idx": 2},
    }
    # Unchanged overlay controls keep the layer and its colour range.
    assert calls == []
    assert (view.state.overlay.vmin, view.state.overlay.vmax) == (0.0, 100.0)
    np.testing.assert_array_equal(
        _overlay_plane(view), _expected_fit_plane(view, repetition=2)
    )
    view.set_overlay(metabolite="lactate")
    assert len(calls) == 1


def _block_range(view):
    block = np.frombuffer(view._block, dtype=np.float32)
    finite = block[np.isfinite(block)]
    return float(finite.min()), float(finite.max())


def test_map_range_starts_full_and_follows_the_range_modes_of_plot(monkeypatch):
    data = _csi()
    view = plot_scan(data, state=_state())
    # Spectroscopic data start in Min/Max over the finite range of the plane.
    spatial = view.state.spatial
    assert spatial.range_mode == "Min/Max"
    assert (spatial.vmin, spatial.vmax) == pytest.approx(_block_range(view))
    assert view._view["plane_range"] == pytest.approx(list(_block_range(view)))
    assert view._map_range == {
        "range_mode": "Min/Max",
        "vmin": spatial.vmin,
        "vmax": spatial.vmax,
    }

    # Colour limits only redraw: nothing is recomputed.
    payload = view._model.get_render_payload

    def fail(*_args, **_kwargs):
        raise AssertionError("a colour range must not recompute the block")

    monkeypatch.setattr(view._model, "get_render_payload", fail)
    view.set_map_range(vmin=2.0)
    assert spatial.vmin == 2.0
    assert spatial.vmax == pytest.approx(_block_range(view)[1])
    view._map_range = {"range_mode": "Percentage", "vmin": 10.0, "vmax": 90.0}
    assert (spatial.range_mode, spatial.vmin, spatial.vmax) == ("Percentage", 10, 90)
    view.set_map_range("Min/Max")
    assert (spatial.vmin, spatial.vmax) == pytest.approx(_block_range(view))
    monkeypatch.setattr(view._model, "get_render_payload", payload)

    # Min/Max returns to the full range when the shown values change ...
    view.set_map_range(vmin=1.0, vmax=2.0)
    view._controls = {**view._controls, "spec_part": "Real"}
    real_range = _block_range(view)
    assert real_range[0] < 0
    assert (spatial.vmin, spatial.vmax) == pytest.approx(real_range)
    assert view._map_range["vmin"] == pytest.approx(real_range[0])
    # ... but not for another repetition, whose values share the scale.
    view.set_map_range(vmin=1.0, vmax=2.0)
    controls = view._controls
    view._controls = {
        **controls,
        "indices": {**controls["indices"], "repetition_idx": 2},
    }
    assert (spatial.vmin, spatial.vmax) == (1.0, 2.0)
    # Absolute keeps its limits when the shown values change.
    view.set_map_range("Absolute", vmin=1.0, vmax=2.0)
    view._controls = {**view._controls, "line_broadening_hz": 5.0}
    assert (spatial.range_mode, spatial.vmin, spatial.vmax) == ("Absolute", 1.0, 2.0)


def test_map_range_is_validated_and_a_supplied_range_is_kept():
    data = _csi()
    state = _state()
    state.spatial.range_mode = "Absolute"
    state.spatial.vmin, state.spatial.vmax = 5.0, 50.0
    view = plot_scan(data, state=state)
    assert state.spatial.range_mode == "Absolute"
    assert (state.spatial.vmin, state.spatial.vmax) == (5.0, 50.0)

    with pytest.raises(ValueError, match="range_mode must be one of"):
        view.set_map_range("Log")
    with pytest.raises(ValueError, match=r"vmin \(60.0\) must be below vmax"):
        view.set_map_range(vmin=60.0)
    with pytest.raises(ValueError, match="vmax must be a finite number"):
        view.set_map_range(vmax=float("inf"))
    with pytest.raises(ValueError, match="must be below vmax"):
        view._map_range = {"range_mode": "Absolute", "vmin": 3.0, "vmax": 3.0}
    assert (state.spatial.vmin, state.spatial.vmax) == (5.0, 50.0)
    view.set_map_range("Percentage")
    assert (state.spatial.range_mode, state.spatial.vmin, state.spatial.vmax) == (
        "Percentage",
        0.0,
        100.0,
    )
    state = _state()
    state.spatial.range_mode = "Log"
    with pytest.raises(ValueError, match="range_mode must be one of"):
        ScanView(data, _model(data), state)


def test_overlay_colormap_and_limits_are_validated_without_partial_changes():
    from cmap import Colormap

    data = _csi()
    view = plot_scan(
        data, overlay=_fit_results(data, with_coordinates=False), state=_state()
    )
    names = [name for _, name in view._view["overlay_options"]["cmaps"]]
    assert "viridis" in names
    assert view.state.overlay.cmap in names
    view.set_overlay(cmap="viridis")
    assert view.state.overlay.cmap == "viridis"
    assert view._controls["overlay"]["cmap"] == "viridis"
    assert view._view["overlay"]["colorscale"] == Colormap("viridis").to_plotly()

    # The browser sends a colormap through the overlay controls.
    overlay = view._controls["overlay"]
    view._controls = {**view._controls, "overlay": {**overlay, "cmap": "magma"}}
    assert view.state.overlay.cmap == "magma"

    before = (view.state.overlay.active_variable, view.state.overlay.cmap)
    with pytest.raises(ValueError, match="unknown colormap 'no_such_map'"):
        view.set_overlay("overlay_fit_freqs", cmap="no_such_map")
    with pytest.raises(ValueError, match=r"vmin \(5.0\) must be below vmax \(1.0\)"):
        view.set_overlay("overlay_fit_freqs", vmin=5.0, vmax=1.0)
    with pytest.raises(ValueError, match="must be below vmax"):
        view.set_overlay(vmin=view.state.overlay.vmax)
    assert (view.state.overlay.active_variable, view.state.overlay.cmap) == before
    assert (view.state.overlay.vmin, view.state.overlay.vmax) == (0.0, 100.0)


def test_saved_view_settings_are_restored_by_the_next_plot_scan(monkeypatch):
    data = _csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    fit = _fit_results(data, with_coordinates=False)
    view = plot_scan(scan, overlay=fit)
    assert view._view["can_save_settings"] is True
    view.set_map_range("Absolute", vmin=1.0, vmax=2.0)
    view._controls = {**view._controls, "spec_part": "Real"}
    view.selection = {"row": 0, "column": 1, "curve_index": 20}
    view.set_overlay("overlay_fit_freqs", cmap="viridis")
    view.set_overlay_opacity(0.4)
    sent = []
    monkeypatch.setattr(
        view, "send", lambda content, buffers=None: sent.append(content)
    )
    view._handle_custom_message(view, {"type": "save_settings"}, [])
    assert sent == [{"type": "settings_saved", "ok": True, "message": ""}]

    reopened = plot_scan(scan, overlay=fit)
    spatial = reopened.state.spatial
    assert (spatial.range_mode, spatial.vmin, spatial.vmax) == ("Absolute", 1.0, 2.0)
    assert reopened.state.spectral.spec_part == "Real"
    assert reopened.selection == view.selection
    assert reopened.state.overlay.active_variable == "overlay_fit_freqs"
    assert reopened.state.overlay.cmap == "viridis"
    assert reopened._overlay_opacity == pytest.approx(0.4)

    # Another context, an explicit state or an overlay without the saved
    # layer start without (all of) the saved settings.
    other = plot_scan(scan, settings_context="other")
    assert other.state.spatial.range_mode == "Percentage"
    explicit = plot_scan(scan, state=_state())
    assert explicit.state.spectral.spec_part == "Magnitude"
    single = plot_scan(scan, overlay=fit["fit_amps"])
    assert single.state.overlay.active_variable == "overlay"
    assert single.state.spatial.range_mode == "Absolute"

    plain = plot_scan(data, state=_state())
    assert plain._view["can_save_settings"] is False
    with pytest.raises(ValueError, match="Call plot_scan\\(scan\\)"):
        plain.save_settings()
    sent.clear()
    monkeypatch.setattr(
        plain, "send", lambda content, buffers=None: sent.append(content)
    )
    plain._handle_custom_message(plain, {"type": "save_settings"}, [])
    assert sent[0]["ok"] is False and "plot_scan(scan)" in sent[0]["message"]


def _stacked_csi(n_slices=3, slice_spacing_mm=4.0):
    """Return the invented CSI with several slices and x/y/z centres in mm."""
    data = xr.concat([_csi()] * n_slices, "slice_idx").transpose(*DIMS)
    shape = tuple(data.sizes[dim] for dim in GRID_DIMS)
    read, phase, slice_ = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return data.assign_coords(
        x=(GRID_DIMS, 2.0 * read, {"unit": "mm"}),
        y=(GRID_DIMS, 10.0 - 3.0 * phase, {"unit": "mm"}),
        z=(GRID_DIMS, slice_spacing_mm * slice_, {"unit": "mm"}),
    )


def _thin_slice_overlay(data, n_slices=6, slice_spacing_mm=3.0):
    """Return an invented overlay on the CSI in-plane grid with thinner slices.

    Values are ``100 * slice + read``, so a drawn plane names its slice.
    """
    shape = (data.sizes["read_idx"], data.sizes["phase_idx"], n_slices)
    read, phase, slice_ = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return xr.DataArray(
        100.0 * slice_ + read,
        dims=GRID_DIMS,
        coords={
            "x": (GRID_DIMS, 2.0 * read, {"unit": "mm"}),
            "y": (GRID_DIMS, 10.0 - 3.0 * phase, {"unit": "mm"}),
            "z": (GRID_DIMS, slice_spacing_mm * slice_, {"unit": "mm"}),
        },
        attrs={"invented_fixture": True},
    )


def _drawn_overlay_slice(view):
    """Return the overlay slice whose values are drawn (values name it)."""
    return int(np.nanmax(_overlay_plane(view)) // 100)


def test_overlay_slice_choice_moves_the_map_in_scanner_coordinates():
    # CSI slices at z = 0, 4, 8 mm; overlay slices at z = 0, 3, ..., 15 mm.
    data = _stacked_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan, overlay=_thin_slice_overlay(data), state=_state())
    assert view.state.spatial.coordinate_mode == "xyz"
    options = view._view["overlay_options"]
    assert (options["slice_dim"], options["slice_count"]) == ("slice_idx", 6)
    # Following the map at z = 0 mm draws overlay slice 0.
    assert view._controls["overlay"]["slice"] is None
    assert view._view["overlay"]["slice_index"] == 0
    assert _drawn_overlay_slice(view) == 0

    # Overlay slice 3 (z = 9 mm) moves the map to its nearest slice (z = 8 mm).
    view.set_overlay_slice(3)
    assert view.state.spatial.slice_indices["slice_idx"] == 2
    assert view._controls["indices"]["slice_idx"] == 2
    assert view._controls["overlay"]["slice"] == 3
    assert _drawn_overlay_slice(view) == 3
    assert view._view["overlay"]["warning"] == ""

    # Overlay slice 5 (z = 15 mm) lies outside the CSI slices: warn.
    controls = view._controls
    view._controls = {**controls, "overlay": {**controls["overlay"], "slice": 5}}
    assert _drawn_overlay_slice(view) == 5
    assert "outside primary FOV" in view._view["overlay"]["warning"]
    assert view.state.overlay.slice_overlap_status == "secondary_outside_primary"

    # Moving the map lets the overlay follow it again, even though the
    # browser resends its previous slice choice with the new index.
    controls = view._controls
    view._controls = {**controls, "indices": {**controls["indices"], "slice_idx": 1}}
    assert view.state.overlay.slicing_dim is None
    assert view._controls["overlay"]["slice"] is None
    assert view._view["overlay"]["warning"] == ""
    # The map at z = 4 mm is nearest to overlay slice 1 (z = 3 mm).
    assert _drawn_overlay_slice(view) == 1


def test_overlay_slice_choice_stays_in_rps_and_is_validated():
    data = _stacked_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(
        scan,
        overlay=_thin_slice_overlay(data),
        state=_state(),
        coordinate_mode="rps",
    )
    view.set_overlay_slice(4)
    assert view.state.spatial.slice_indices["slice_idx"] == 0
    assert _drawn_overlay_slice(view) == 4
    controls = view._controls
    view._controls = {**controls, "indices": {**controls["indices"], "slice_idx": 2}}
    assert view._controls["overlay"]["slice"] == 4
    assert _drawn_overlay_slice(view) == 4
    view.set_overlay_slice(None)
    assert view._controls["overlay"]["slice"] is None
    assert view.state.overlay.slice_indices == {}

    for bad in (6, -1, 1.5, True):
        with pytest.raises(ValueError, match="overlay slice"):
            view.set_overlay_slice(bad)
    assert view.state.overlay.slicing_dim is None
    no_overlay, _ = _view()
    with pytest.raises(ValueError, match="no overlay was given"):
        no_overlay.set_overlay_slice(0)
    # A one-slice overlay offers no slice choice.
    flat = plot_scan(_csi(), overlay=_fit_results(_csi(), with_coordinates=False))
    assert flat._view["overlay_options"]["slice_count"] == 1
    with pytest.raises(ValueError, match="single slice"):
        flat.set_overlay_slice(0)


def test_anatomy_slice_choice_keeps_a_single_slice_map_and_warns_outside_it():
    # A 2D CSI at z = 0 mm with an invented 6 mm slice over an anatomy with
    # slices at z = -4, -2, 0, 2, 4 mm.
    data = _with_scanner_coordinates(_csi())
    data.attrs["slice_thickness_mm"] = 6.0
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan, anatomy=_stacked_anatomy_scan(), state=_state())
    anatomy = view._view["anatomy"]
    assert (anatomy["slice_dim"], anatomy["slice_count"]) == ("slice_idx", 5)
    # Following the map draws the anatomy slice at z = 0 mm.
    assert view._controls["anatomy"] == {"slice": None}
    assert anatomy["slice_index"] == 2

    # Anatomy slice 3 (z = 2 mm) lies inside the CSI slice; the map stays.
    view.set_anatomy_slice(3)
    assert view.state.spatial.slice_indices["slice_idx"] == 0
    assert view._controls["anatomy"] == {"slice": 3}
    assert view._view["anatomy"]["slice_index"] == 3
    assert view._view["anatomy"]["warning"] == ""

    # Anatomy slice 4 (z = 4 mm) lies outside it (|4| > 6 / 2): warn. The
    # browser sends the choice with its other controls.
    controls = view._controls
    view._controls = {**controls, "anatomy": {"slice": 4}}
    assert view._view["anatomy"]["slice_index"] == 4
    assert "outside primary FOV" in view._view["anatomy"]["warning"]
    assert view.state.anatomy.slice_overlap_status == "secondary_outside_primary"

    # None follows the map again.
    view.set_anatomy_slice(None)
    assert view.state.anatomy.slicing_dim is None
    assert view._controls["anatomy"] == {"slice": None}
    assert view._view["anatomy"]["slice_index"] == 2
    assert view._view["anatomy"]["warning"] == ""


def test_anatomy_slice_choice_moves_a_stacked_map_and_is_validated():
    # CSI slices at z = 0, 4, 8 mm; anatomy slices at z = -4..4 mm.
    data = _stacked_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan, anatomy=_stacked_anatomy_scan(), state=_state())
    # Anatomy slice 4 (z = 4 mm) moves the map to its slice at z = 4 mm.
    view.set_anatomy_slice(4)
    assert view.state.spatial.slice_indices["slice_idx"] == 1
    assert view._controls["indices"]["slice_idx"] == 1
    assert view._view["anatomy"]["slice_index"] == 4
    assert view._view["anatomy"]["warning"] == ""

    # Moving the map lets the anatomy follow it again, even though the
    # browser resends its previous slice choice with the new index.
    controls = view._controls
    view._controls = {**controls, "indices": {**controls["indices"], "slice_idx": 0}}
    assert view.state.anatomy.slicing_dim is None
    assert view._controls["anatomy"] == {"slice": None}
    assert view._view["anatomy"]["slice_index"] == 2

    for bad in (5, -1, 1.5, True):
        with pytest.raises(ValueError, match="anatomy slice"):
            view.set_anatomy_slice(bad)
    assert view.state.anatomy.slicing_dim is None
    no_anatomy, _ = _view()
    assert "anatomy" not in no_anatomy._controls
    with pytest.raises(ValueError, match="no anatomy is drawn"):
        no_anatomy.set_anatomy_slice(0)
    # A one-slice anatomy offers no slice choice.
    flat = plot_scan(
        SimpleNamespace(
            primary_data=_with_scanner_coordinates(_csi()), is_spectroscopic=True
        ),
        anatomy=_anatomy_scan(),
        state=_state(),
    )
    assert flat._view["anatomy"]["slice_count"] < 2
    with pytest.raises(ValueError, match="single slice"):
        flat.set_anatomy_slice(0)


class _OverlayScan:
    """Invented overlay scan with three reconstructions on the CSI grid.

    ``data`` holds ``read``, ``scanner_data`` ``read + 100`` and the
    DICOM series ``read + 200``; the DICOM series is read on request only.
    """

    is_spectroscopic = False
    has_dicom_reco = True

    def __init__(self, data):
        shape = tuple(data.sizes[dim] for dim in GRID_DIMS)
        read = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")[0]

        def layer(offset, **attrs):
            return xr.DataArray(
                read + offset, dims=GRID_DIMS, attrs={"invented_fixture": True, **attrs}
            )

        self.data = layer(0.0)
        self.scanner_data = layer(100.0)
        self._dicom_data = layer(200.0, source="dicom")
        self.dicom_load_count = 0

    @property
    def primary_data(self):
        return self.data

    def load_dicom_data(self):
        self.dicom_load_count += 1
        return self._dicom_data


def test_overlay_source_switch_reads_dicom_lazily_and_resets_range():
    data = _csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    overlay_scan = _OverlayScan(data)
    view = plot_scan(scan, overlay=overlay_scan, state=_state())
    options = view._view["overlay_options"]
    assert [key for _, key in options["data_sources"]] == [
        "data",
        "scanner_data",
        "dicom",
    ]
    assert view._controls["overlay"]["data_source"] == "data"
    assert overlay_scan.dicom_load_count == 0
    assert float(np.nanmin(_overlay_plane(view))) == 0.0

    view.set_overlay("overlay", range_mode="Min/Max", vmin=1.0, vmax=2.0)
    view.set_overlay_source("scanner_data")
    assert view.state.overlay.data_source == "scanner_data"
    assert float(np.nanmin(_overlay_plane(view))) == 100.0
    assert (view.state.overlay.vmin, view.state.overlay.vmax) == (100.0, 104.0)
    assert overlay_scan.dicom_load_count == 0

    # The browser source control; the stale limits sent with it are ignored.
    controls = view._controls
    view._controls = {
        **controls,
        "overlay": {**controls["overlay"], "data_source": "dicom", "vmin": 0.5},
    }
    assert overlay_scan.dicom_load_count == 1
    assert float(np.nanmin(_overlay_plane(view))) == 200.0
    assert view._controls["overlay"]["vmin"] == 200.0
    view.set_overlay_source("data")
    view.set_overlay_source("dicom")
    assert overlay_scan.dicom_load_count == 1

    with pytest.raises(ValueError, match="unknown source 'pdata'"):
        view.set_overlay_source("pdata")
    assert view.state.overlay.data_source == "dicom"
    fit_view = plot_scan(scan, overlay=_fit_results(data, with_coordinates=False))
    assert fit_view._view["overlay_options"]["data_sources"] == []
    with pytest.raises(ValueError, match="pass the overlay as a scan"):
        fit_view.set_overlay_source("dicom")


def test_saved_overlay_source_and_slice_are_restored():
    data = _stacked_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    overlay_scan = _OverlayScan(data)
    view = plot_scan(scan, overlay=overlay_scan, coordinate_mode="rps")
    view.set_overlay_source("scanner_data")
    view.set_overlay_slice(2)
    view.save_settings()

    reopened = plot_scan(scan, overlay=overlay_scan, coordinate_mode="rps")
    assert reopened._controls["overlay"]["data_source"] == "scanner_data"
    assert reopened._controls["overlay"]["slice"] == 2
    assert float(np.nanmin(_overlay_plane(reopened))) == 100.0
    assert overlay_scan.dicom_load_count == 0


# ── curve along another dimension ─────────────────────────────────────


def _repetition_state():
    state = _state()
    state.spectral.secondary_dim = "repetition_idx"
    return state


def test_curve_block_runs_along_requested_curve_dimension():
    data = _csi()
    model = _model(data)
    state = _repetition_state()
    payload = model.get_render_payload(data, state, include_curve_block=True)

    block = payload.curve_block
    assert block.shape == (*payload.spatial_array.shape, data.sizes["repetition_idx"])
    # The selected voxel's curve is the payload's repetition curve.
    row_dim, column_dim = payload.spatial_dims
    indices = payload.spatial_source_indices
    row = int(
        np.flatnonzero(indices[row_dim] == state.spatial.slice_indices[row_dim])[0]
    )
    column = int(
        np.flatnonzero(indices[column_dim] == state.spatial.slice_indices[column_dim])[
            0
        ]
    )
    np.testing.assert_array_equal(block[row, column], payload.spectral_y)
    # Each curve position is the map shown at that repetition.
    for repetition in range(data.sizes["repetition_idx"]):
        shown = _repetition_state()
        shown.spatial.slice_indices["repetition_idx"] = repetition
        plane = model.get_render_payload(data, shown).spatial_array
        np.testing.assert_array_equal(block[:, :, repetition], plane)


def test_plot_scan_curve_over_repetitions():
    data = _csi()
    view = plot_scan(data, curve_dim="repetition_idx")
    assert view.state.spectral.secondary_dim == "repetition_idx"
    assert view._view["curve_dim"] == "repetition_idx"
    assert view._view["shape"][2] == data.sizes["repetition_idx"]
    assert len(view._view["curve_x"]) == data.sizes["repetition_idx"]
    # The spectral position becomes a slider like the other dimensions.
    assert view._view["other_dims"]["acquisition_idx"] == data.sizes["acquisition_idx"]
    assert "repetition_idx" not in view._view["other_dims"]
    assert (
        view.selection["curve_index"]
        == view.state.spatial.slice_indices["repetition_idx"]
    )


def test_curve_selection_and_spectral_slider_over_repetitions(monkeypatch):
    data = _csi()
    view = plot_scan(data, curve_dim="repetition_idx")
    spectral_before = view.state.spectral.selected_spectral_index
    block_before = view._block

    def fail(*_args, **_kwargs):
        raise AssertionError("a selection must not recompute the block")

    with monkeypatch.context() as patch:
        patch.setattr(view._model, "get_render_payload", fail)
        view.selection = {**view.selection, "curve_index": 2}
    assert view.state.spatial.slice_indices["repetition_idx"] == 2
    assert view.state.spectral.selected_spectral_index == spectral_before
    assert view._block is block_before

    controls = view._controls
    view._controls = {
        **controls,
        "indices": {**controls["indices"], "acquisition_idx": 20},
    }
    assert view.state.spectral.selected_spectral_index == 20
    assert view.state.spectral.selected_spectral_value is None
    assert view._controls["indices"]["acquisition_idx"] == 20
    expected = _model(data).get_render_payload(
        data, view.state, include_curve_block=True
    )
    block = np.frombuffer(view._block, dtype=np.float32).reshape(view._view["shape"])
    np.testing.assert_array_equal(block, expected.curve_block)
    assert view.selection["curve_index"] == 2


def test_averaging_the_curve_dimension_keeps_the_curve():
    # As in plot: with the curve along repetitions, averaging reduces the map
    # only. The curves keep every repetition and the averaged ones are marked.
    data = _csi()
    view = plot_scan(data, curve_dim="repetition_idx")
    curves = np.frombuffer(view._block, dtype=np.float32).copy()

    view.set_processing(repetitions="average", repetition_range=(1, 2))
    n_rep = data.sizes["repetition_idx"]
    assert view._view["curve_dim"] == "repetition_idx"
    assert view._view["shape"][2] == n_rep
    np.testing.assert_array_equal(np.frombuffer(view._block, dtype=np.float32), curves)
    assert view._view["map_fixed"] is True
    assert view._view["average_range"] == [1.0, 2.0]
    expected = _model(data).get_render_payload(
        data, view.state, include_curve_block=False
    )
    shown_map = np.frombuffer(view._map_plane, dtype=np.float32)
    np.testing.assert_array_equal(
        shown_map.reshape(view._view["shape"][:2]),
        expected.spatial_array.astype(np.float32),
    )
    # The averaged map is not one of the repetitions.
    block = curves.reshape(view._view["shape"])
    assert not any(
        np.array_equal(block[..., k], shown_map.reshape(block.shape[:2]))
        for k in range(n_rep)
    )

    view.set_processing(repetitions="off")
    assert view._view["map_fixed"] is False
    assert view._view["average_range"] is None
    assert view._map_plane == b""


def test_roi_mean_runs_along_curve_dimension():
    data = _csi()
    view = plot_scan(data, curve_dim="repetition_idx")
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    view.show_roi_mean("roi")
    voxel = _source_voxel(view, 0, 0)
    # Other dimensions come from state.spatial.slice_indices, as in plot.
    indices = view.state.spatial.slice_indices
    selected = data.isel(
        acquisition_idx=indices["acquisition_idx"], channel_idx=0, slice_idx=0
    )
    np.testing.assert_allclose(
        view.roi_mean.y, np.abs(selected.isel(voxel).values), rtol=1e-6
    )
    assert view.roi_mean.x_label == "repetition_idx [idx]"


def test_plot_scan_curve_dim_choice_and_saved_curve_dim():
    data = _csi()
    with pytest.raises(ValueError, match="curve_dim must be one of"):
        plot_scan(data, curve_dim="channel_idx")
    with pytest.raises(ValueError, match="curve_dim must be one of"):
        plot_scan(data, curve_dim="read_idx")

    state = PlotState()
    state.spectral.secondary_dim = "repetition_idx"
    assert plot_scan(data, state=state)._view["curve_dim"] == "repetition_idx"
    # A kept curve dimension the data cannot show falls back to the spectrum.
    state = PlotState()
    state.spectral.secondary_dim = "echo_idx"
    assert plot_scan(data, state=state)._view["curve_dim"] == "acquisition_idx"


def _repetition_curves(data, state):
    """Return the curve block of every repetition, shape (rows, cols, spec, rep)."""
    blocks = []
    for repetition in range(data.sizes["repetition_idx"]):
        shown = _state()
        shown.spectral.spec_part = state.spectral.spec_part
        shown.spatial.slice_indices["repetition_idx"] = repetition
        payload = _model(data).get_render_payload(data, shown, include_curve_block=True)
        blocks.append(payload.curve_block)
    return np.stack(blocks, axis=-1)


def _time_block(view):
    n_rows, n_columns, _ = view._view["shape"]
    return np.frombuffer(view._time_block, dtype=np.float32).reshape(
        n_rows, n_columns, -1
    )


def test_curve_time_block_holds_the_curve_block_of_every_repetition():
    data = _csi()
    state = _state()
    payload = _model(data).get_render_payload(
        data, state, include_curve_block=True, time_dim="repetition_idx"
    )
    np.testing.assert_array_equal(
        payload.curve_time_block, _repetition_curves(data, state)
    )
    # The shown repetition of the time block is the curve block itself.
    np.testing.assert_array_equal(payload.curve_time_block[..., 1], payload.curve_block)
    plain = _model(data).get_render_payload(data, state, include_curve_block=True)
    assert plain.curve_time_block is None


def test_time_course_follows_the_selected_spectral_point(monkeypatch):
    view, data = _view()
    curves = _repetition_curves(data, view.state)
    time_view = view._view["time"]
    assert time_view["dim"] == "repetition_idx"
    assert time_view["x"] == [0.0, 1.0, 2.0]
    assert time_view["index"] == 1
    assert view.time_course_dim == "repetition_idx"
    np.testing.assert_allclose(_time_block(view), curves[:, :, 10], rtol=1e-6)
    assert view._time_course["follows_point"] is True
    assert view._time_course["range"] == [10, 10]

    def fail(*_args, **_kwargs):
        raise AssertionError("moving the spectral point must not recompute")

    monkeypatch.setattr(view._model, "get_render_payload", fail)
    view.selection = {**view.selection, "curve_index": 20}
    np.testing.assert_allclose(_time_block(view), curves[:, :, 20], rtol=1e-6)
    assert view._time_course["range"] == [20, 20]


def test_time_course_range_is_the_mean_over_spectral_points():
    view, data = _view()
    curves = _repetition_curves(data, view.state)
    x = view._view["curve_x"]
    # Limits in display units, in either order, snap to the nearest point.
    view.set_time_course_range(x[8], x[5])
    assert view.state.spectral.time_course_range == (5, 8)
    np.testing.assert_allclose(
        _time_block(view), curves[:, :, 5:9].mean(axis=2), rtol=1e-5
    )
    assert view._time_course["follows_point"] is False
    assert view._time_course["range_values"] == [x[5], x[8]]
    # A fixed range stays when the spectral point moves.
    view.selection = {**view.selection, "curve_index": 30}
    assert view._time_course["range"] == [5, 8]
    view.set_time_course_range()
    assert view.state.spectral.time_course_range is None
    assert view._time_course["range"] == [30, 30]
    with pytest.raises(ValueError, match="needs both limits"):
        view.set_time_course_range(x[3])
    with pytest.raises(ValueError, match="finite number"):
        view.set_time_course_range(float("nan"), x[3])


def test_time_course_inside_a_mask_follows_edits_and_removal():
    view, data = _view()
    curves = _repetition_curves(data, view.state)
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    view.draw_polygon(_voxel_square(view, 2, 3), mask="roi")
    view.show_time_course_mask("roi")
    # Equal coverage: the weighted mean is the plain mean of both voxels.
    expected = (curves[0, 0, 10] + curves[2, 3, 10]) / 2
    np.testing.assert_allclose(view._time_course["mask_y"], expected, rtol=1e-5)
    assert view._time_course["mask"] == "roi"
    view.set_time_course_range(view._view["curve_x"][4], view._view["curve_x"][6])
    expected = (curves[0, 0, 4:7] + curves[2, 3, 4:7]).mean(axis=0) / 2
    np.testing.assert_allclose(view._time_course["mask_y"], expected, rtol=1e-5)
    view.undo_mask_edit()
    np.testing.assert_allclose(
        view._time_course["mask_y"], curves[0, 0, 4:7].mean(axis=0), rtol=1e-5
    )
    view.remove_mask("roi")
    assert view._time_course["mask"] is None
    assert view._time_course["mask_y"] is None
    with pytest.raises(ValueError, match="show_time_course_mask"):
        view.show_time_course_mask("missing")


def test_time_course_browser_messages_and_dimension_choice(monkeypatch):
    data = _csi().isel(channel_idx=[0, 0]).assign_coords(channel_idx=[0, 1])
    state = _state()
    state.spectral.combine_channels = False
    view, _ = _view(data, state)
    sent = []
    monkeypatch.setattr(view, "send", sent.append)
    x = view._view["curve_x"]
    view._handle_custom_message(
        view, {"type": "time_course", "range": [x[2], x[4]]}, []
    )
    assert view.state.spectral.time_course_range == (2, 4)
    view._handle_custom_message(view, {"type": "time_course", "range": None}, [])
    assert view.state.spectral.time_course_range is None
    assert view._view["time"]["choices"] == ["channel_idx", "repetition_idx"]
    view._handle_custom_message(view, {"type": "time_course", "dim": "channel_idx"}, [])
    assert view.time_course_dim == "channel_idx"
    assert view.state.spectral.time_course_dim == "channel_idx"
    assert _time_block(view).shape[2] == 2
    view._handle_custom_message(view, {"type": "time_course", "mask": "missing"}, [])
    assert sent[-1]["type"] == "time_course_error"
    assert "show_time_course_mask" in sent[-1]["message"]


def test_no_time_course_for_other_curves_or_single_repetitions():
    view = plot_scan(_csi(), curve_dim="repetition_idx")
    assert view._view["time"] is None
    assert view._time_block == b""
    assert view.time_course_dim is None
    with pytest.raises(ValueError, match="needs the time course panel"):
        view.set_time_course_range(1.0, 2.0)
    single, _ = _view(_csi(n_rep=1), _state())
    assert single._view["time"] is None
    # Combined channels and averaged repetitions are not offered.
    combined, _ = _view(
        _csi().isel(channel_idx=[0, 0]).assign_coords(channel_idx=[0, 1])
    )
    assert combined._view["time"]["choices"] == ["repetition_idx"]
    averaged_state = _state()
    averaged_state.spectral.average_repetitions = True
    averaged, _ = _view(_csi(), averaged_state)
    assert averaged._view["time"] is None


def test_curve_range_mean_sorts_and_clips_the_range():
    curves = np.arange(2 * 1 * 4 * 3, dtype=np.float32).reshape(2, 1, 4, 3)
    np.testing.assert_allclose(
        curve_range_mean(curves, 2, 1), curves[:, :, 1:3].mean(axis=2)
    )
    np.testing.assert_array_equal(curve_range_mean(curves, 3, 3), curves[:, :, 3])
    np.testing.assert_allclose(
        curve_range_mean(curves, -5, 99), curves.mean(axis=2), rtol=1e-6
    )
    assert curve_range_mean(curves, 0, 1).dtype == np.float32


def _single_voxel():
    """Invented single-voxel FIDs whose amplitude encodes the repetition."""
    return _csi(n_read=1, n_phase=1).isel(
        read_idx=0, phase_idx=0, slice_idx=0, channel_idx=0
    )


@pytest.mark.parametrize("with_plane", [False, True])
def test_single_voxel_spectra_show_spectrum_and_time_course(with_plane):
    data = _single_voxel()
    if with_plane:
        data = data.expand_dims(read_idx=1, phase_idx=1, slice_idx=1)
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = plot_scan(scan)

    assert view._view["single_voxel"] is True
    assert view._curve_dim == "acquisition_idx"
    assert view._time_dim == "repetition_idx"
    assert _block(view).shape == (1, 1, data.sizes["acquisition_idx"])
    # The browser curve is the spectrum that plot shows for the same scan.
    shown_by_plot = _model(data).get_render_payload(data, view.state).spectral_y
    np.testing.assert_allclose(_block(view)[0, 0], shown_by_plot, rtol=1e-5)
    # The scan's samples keep their dimensions.
    assert scan.primary_data.dims == data.dims

    view.selection = {"row": 0, "column": 0, "curve_index": 5}
    assert view.state.spectral.selected_spectral_index == 5

    repetitions = plot_scan(scan, curve_dim="repetition_idx")
    assert _block(repetitions).shape == (1, 1, data.sizes["repetition_idx"])


def test_multi_voxel_view_is_not_single_voxel():
    view, _ = _view()

    assert view._view["single_voxel"] is False


def _stacked_anatomy_scan():
    """Invented 3D anatomy, 1 mm pixels, slices 2 mm apart at z = -4..4 mm.

    The pixel value ``x + 100 * y`` (mm) tells where each displayed pixel
    lies, so the test can find it without the display mapping.
    """
    grid_dims = ("read_idx", "phase_idx", "slice_idx")
    x, y, z = np.meshgrid(
        np.arange(20.0), np.arange(15.0), [-4.0, -2.0, 0.0, 2.0, 4.0], indexing="ij"
    )
    anatomy = xr.DataArray(
        x + 100.0 * y,
        dims=grid_dims,
        coords={
            name: (grid_dims, value, {"unit": "mm"})
            for name, value in (("x", x), ("y", y), ("z", z))
        },
        attrs={"invented_fixture": True},
    )
    return SimpleNamespace(primary_data=anatomy, is_spectroscopic=False)


def _press_scan():
    """Invented PRESS scan: 5 x 3 x 1 mm voxel centred at LPS (6, 5, 2) mm."""
    method = {
        "PVM_VoxArrSize": [[5.0, 3.0, 1.0]],
        # Head_Prone: LPS = (-read, -phase, slice).
        "PVM_VoxArrPositionRPS": [[-6.0, -5.0, 2.0]],
        "PVM_VoxArrGradOrient": [np.eye(3).tolist()],
    }
    return SimpleNamespace(
        primary_data=_single_voxel(),
        is_spectroscopic=True,
        handler=SimpleNamespace(method=method, patient_position="Head_Prone"),
    )


def _pixels_in_outline(view):
    """Return the LPS (x, y) of the displayed pixels inside the outline box."""
    layer, outline = view._view["anatomy"], view._view["outline"]
    image = np.frombuffer(view._anatomy, dtype=np.float32).reshape(layer["shape"])
    columns, rows = np.meshgrid(layer["x"], layer["y"])
    inside = (
        (columns > min(outline["x"]))
        & (columns < max(outline["x"]))
        & (rows > min(outline["y"]))
        & (rows < max(outline["y"]))
    )
    values = image[inside].astype(int)
    return set(zip((values % 100).tolist(), (values // 100).tolist(), strict=True))


def test_single_voxel_outline_marks_the_voxel_on_the_nearest_anatomy_slice():
    view = plot_scan(_press_scan(), anatomy=_stacked_anatomy_scan())
    outline = view._view["outline"]

    assert view._view["single_voxel"] is True
    assert outline["kind"] == "voxel"
    assert (outline["slice_dim"], outline["slice_index"]) == ("slice_idx", 3)
    assert outline["n_slices"] == 5
    assert outline["x"][0] == outline["x"][-1] and outline["y"][0] == outline["y"][-1]
    # Pixel centres of the box x 4..8 mm, y 4..6 mm lie inside; the next
    # pixels lie half a pixel outside.
    assert _pixels_in_outline(view) == {
        (x, y) for x in range(4, 9) for y in range(4, 7)
    }
    assert view._view["anatomy"]["warning"] == ""
    assert view._controls["anatomy_slice"] == 3


def test_anatomy_slice_slider_moves_the_plane_under_the_outline():
    view = plot_scan(_press_scan(), anatomy=_stacked_anatomy_scan())

    view._controls = {**view._controls, "anatomy_slice": 0}

    outline = view._view["outline"]
    assert outline["slice_index"] == 0
    assert outline["x"] == [] and outline["y"] == []
    assert "does not cross" in view._view["anatomy"]["warning"]
    # Out-of-range slider values stay on the anatomy.
    view._controls = {**view._controls, "anatomy_slice": 99}
    assert view._view["outline"]["slice_index"] == 4


def test_anatomy_under_the_outline_uses_its_value_range():
    view = plot_scan(_press_scan(), anatomy=_stacked_anatomy_scan())
    layer = view._view["anatomy"]

    # Pixel values x + 100 * y span 0..1419 on every slice; a fixed 0..100
    # range would draw almost the whole image in the top colour.
    assert (layer["zmin"], layer["zmax"]) == pytest.approx((0.0, 1419.0))


def test_single_voxel_without_anatomy_has_no_outline():
    view = plot_scan(_press_scan())

    assert view._view["outline"] == {}
    assert "anatomy_slice" not in view._controls


def test_single_voxel_outline_needs_scan_geometry():
    with pytest.raises(ValueError, match="excited_volume"):
        plot_scan(_single_voxel(), anatomy=_stacked_anatomy_scan())


def _shown_fit_curves(view, fit, variable):
    """Return the invented fit values of every shown voxel along the curve."""
    indices = view.state.spatial.slice_indices
    values = fit[variable].isel(
        {
            dim: indices.get(dim, 0)
            for dim in fit[variable].dims
            if dim not in GRID_DIMS[:2]
        }
        | {"repetition_idx": slice(None)}
    )
    rows = view._view["row_indices"]
    columns = view._view["column_indices"]
    return values.isel(
        {view._view["row_dim"]: xr.Variable("r", rows)}
        | {view._view["column_dim"]: xr.Variable("c", columns)}
    ).transpose("r", "c", "repetition_idx")


def test_fit_result_dataset_is_shown_as_maps_of_fit_amplitudes():
    fit = _fit_results(_csi(), with_coordinates=False)
    view = plot_scan(fit, state=_state())

    assert view._controls["variable"] == "fit_amps"
    # Fitted spectra are no map and are not offered.
    assert view._view["variable_options"] == ["fit_amps", "fit_freqs"]
    assert view._curve_dim == "repetition_idx"
    assert view._view["other_values"] == {"metabolite": list(METABOLITES)}
    np.testing.assert_array_equal(
        _block(view), _shown_fit_curves(view, fit, "fit_amps")
    )

    # The metabolite slider selects the lactate map.
    view._controls = {
        **view._controls,
        "indices": {**view._controls["indices"], "metabolite": 1},
    }
    assert view.state.spatial.slice_indices["metabolite"] == 1
    np.testing.assert_array_equal(
        _block(view), _shown_fit_curves(view, fit, "fit_amps")
    )


def test_fit_result_variable_menu_switches_the_shown_map():
    fit = _fit_results(_csi(), with_coordinates=False)
    fit["fit_freqs"] = -fit["fit_amps"]
    view = plot_scan(fit, state=_state(part="Real"))
    view._controls = {
        **view._controls,
        "indices": {**view._controls["indices"], "metabolite": 1},
    }

    # The browser sends the new variable with the old slider indices.
    view._controls = {**view._controls, "variable": "fit_freqs"}

    assert view._controls["variable"] == "fit_freqs"
    assert view.state.spatial.slice_indices["metabolite"] == 1
    np.testing.assert_array_equal(
        _block(view), _shown_fit_curves(view, fit, "fit_freqs")
    )
    # The colour range follows the new map.
    assert (view.state.spatial.vmin, view.state.spatial.vmax) == pytest.approx(
        _block_range(view)
    )
    view.set_variable("fit_amps")
    np.testing.assert_array_equal(
        _block(view), _shown_fit_curves(view, fit, "fit_amps")
    )


def test_fit_result_variable_can_be_chosen_and_is_checked():
    fit = _fit_results(_csi(), with_coordinates=False)

    view = plot_scan(fit, variable="fit_freqs", state=_state())
    assert view._controls["variable"] == "fit_freqs"
    with pytest.raises(ValueError, match="fit_spectra"):
        plot_scan(fit, variable="fit_spectra")
    with pytest.raises(ValueError, match="set_variable"):
        view.set_variable("fit_widths")
    with pytest.raises(ValueError, match="only valid with a Dataset"):
        plot_scan(_csi(), variable="fit_amps")
    with pytest.raises(ValueError, match="no map"):
        plot_scan(fit[["fit_spectra"]])


def test_real_fit_maps_start_with_the_real_part():
    fit = _fit_results(_csi(), with_coordinates=False)
    fit["fit_freqs"] = -fit["fit_amps"]

    view = plot_scan(fit, variable="fit_freqs")

    assert view.state.spectral.spec_part == "Real"
    assert _block_range(view)[1] < 0


def test_plain_data_offer_no_variable_menu():
    view = plot_scan(_csi(), state=_state())

    assert "variable" not in view._controls
    assert view._view["variable_options"] == []
    assert view._view["other_values"] == {}
    with pytest.raises(ValueError, match="set_variable"):
        view.set_variable("fit_amps")


# ── slice lines of other views and side-by-side views ──────────────────


def _coronal_scan():
    """Invented coronal stack: rows along z (1 mm), columns along x (1 mm).

    Slices at y = 3, 5 and 7 mm (2 mm apart); the middle one (y = 5 mm) is
    shown first. It crosses ``_stacked_anatomy_scan`` (axial, 1 mm pixels).
    """
    grid_dims = ("read_idx", "phase_idx", "slice_idx")
    x, z, y = np.meshgrid(
        np.arange(20.0), np.arange(-4.0, 5.0), [3.0, 5.0, 7.0], indexing="ij"
    )
    coronal = xr.DataArray(
        x + 100.0 * z,
        dims=grid_dims,
        coords={
            name: (grid_dims, value, {"unit": "mm"})
            for name, value in (("x", x), ("y", y), ("z", z))
        },
        attrs={"invented_fixture": True},
    )
    return SimpleNamespace(primary_data=coronal, is_spectroscopic=False)


def test_view_plane_for_anatomy_is_the_plane_of_its_slices():
    from hypermri.visualization.geometry import view_plane_for_anatomy

    primary = _stacked_anatomy_scan().primary_data
    choices = ["read_idx", "phase_idx", "slice_idx"]
    # Coronal slices (normal y) lie in the phase plane, axial slices
    # (normal z) in the slice plane.
    coronal = _coronal_scan().primary_data
    assert view_plane_for_anatomy(primary, coronal, choices) == "phase_idx"
    assert view_plane_for_anatomy(primary, primary, choices) == "slice_idx"
    with pytest.raises(ValueError, match="at least one view plane"):
        view_plane_for_anatomy(primary, coronal, [])


def test_show_anatomy_plane_turns_to_the_plane_of_the_anatomy():
    view = plot_scan(_stacked_anatomy_scan(), anatomy=_coronal_scan())
    view.show_anatomy_plane()
    assert view.state.spatial.slicing_dim == "phase_idx"
    assert view._anatomy != b""

    # Without an anatomy the plane stays.
    alone = plot_scan(_stacked_anatomy_scan())
    plane = alone.state.spatial.slicing_dim
    alone.show_anatomy_plane()
    assert alone.state.spatial.slicing_dim == plane


def test_image_only_view_has_only_the_image_panel():
    view = plot_scan(_image("repetition_idx"), image_only=True)
    assert view._layout["panels"] == ["image"]
    assert "curve" in plot_scan(_image("repetition_idx"))._layout["panels"]


def test_linked_frames_show_the_same_repetition():
    first = plot_scan(_image("repetition_idx"))
    second = plot_scan(_image("repetition_idx"), image_only=True)
    first.link_frames(second)
    assert first._frame_partners == [second] and second._frame_partners == [first]

    first.selection = {**first.selection, "curve_index": 3}
    assert second.selection["curve_index"] == 3
    second.selection = {**second.selection, "curve_index": 1}
    assert first.selection["curve_index"] == 1

    first.unlink()
    assert first._frame_partners == [] and second._frame_partners == []
    first.selection = {**first.selection, "curve_index": 4}
    assert second.selection["curve_index"] == 1
    with pytest.raises(ValueError, match="other ScanView"):
        first.link_frames(first)


def test_slice_lines_start_off_and_can_be_switched():
    view = plot_scan(_image())
    assert view._display["show_slice_lines"] is False
    assert view._show_slice_lines is False
    view.set_display(show_slice_lines=True)
    assert view._display["show_slice_lines"] is True
    view.set_display(show_slice_lines=False)
    assert view._display["show_slice_lines"] is False
    assert view._show_slice_lines is False
    view._display = {**view._display, "show_slice_lines": True}
    assert view._show_slice_lines is True


def test_masks_can_be_hidden_and_are_kept():
    view = plot_scan(_image())
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    assert view._display["show_masks"] is True

    view.set_display(show_masks=False)
    assert view._display["show_masks"] is False and view._show_masks is False
    assert "roi" in view.masks
    view._display = {**view._display, "show_masks": True}
    assert view._show_masks is True


def test_image_over_anatomy_is_turbo_on_bone():
    over = plot_scan(_stacked_anatomy_scan(), anatomy=_coronal_scan())
    assert over.state.spatial.cmap == "turbo"
    assert over.state.anatomy.cmap == "bone"
    assert plot_scan(_stacked_anatomy_scan()).state.spatial.cmap == "bone"


def _band_rows(view):
    """Return the source row indices at the edges of the slice band."""
    (line,) = view._slice_lines
    rows = np.asarray(view._view["row_indices"], dtype=float)
    y = np.asarray(view._view["y"], dtype=float)
    slope, offset = np.polyfit(rows, y, 1)
    edges = (np.asarray(line["y"]) - offset) / slope
    return sorted({round(float(edge), 6) for edge in edges})


def test_slice_line_marks_where_the_other_slice_crosses_the_image():
    axial = plot_scan(_stacked_anatomy_scan())
    coronal = plot_scan(_coronal_scan())
    axial.show_slices_of(coronal, colours=["#123456"])

    assert axial._view["row_dim"] == "phase_idx"
    (line,) = axial._slice_lines
    assert line["colour"] == "#123456"
    assert line["x"][0] == line["x"][-1] and line["y"][0] == line["y"][-1]
    # Coronal slice at y = 5 mm, 2 mm thick: axial rows (1 mm, y = row) 4..6.
    assert _band_rows(axial) == [4.0, 6.0]
    # Across the whole image width (columns -0.5 .. 19.5).
    assert min(line["x"]) == pytest.approx(min(axial._view["x"]) - 0.5)
    assert max(line["x"]) == pytest.approx(max(axial._view["x"]) + 0.5)


def test_slice_line_follows_the_other_view_and_can_be_removed():
    axial = plot_scan(_stacked_anatomy_scan())
    coronal = plot_scan(_coronal_scan())
    axial.show_slices_of(coronal)

    coronal._controls = {
        **coronal._controls,
        "indices": {**coronal._controls["indices"], "slice_idx": 2},
    }
    assert _band_rows(axial) == [6.0, 8.0]

    axial.show_slices_of()
    assert axial._slice_lines == []
    assert axial not in coronal._slice_viewers


def test_single_slice_uses_its_slice_thickness():
    coronal = _coronal_scan()
    coronal.primary_data = coronal.primary_data.isel(slice_idx=[1])
    coronal.primary_data.attrs["slice_thickness_mm"] = 1.0
    axial = plot_scan(_stacked_anatomy_scan())
    axial.show_slices_of(plot_scan(coronal))

    assert _band_rows(axial) == [4.5, 5.5]


def test_parallel_slices_draw_no_line():
    axial = plot_scan(_stacked_anatomy_scan())
    axial.show_slices_of(plot_scan(_stacked_anatomy_scan()))

    assert axial._slice_lines == []


def test_slice_lines_need_scanner_coordinates_and_valid_colours():
    axial = plot_scan(_stacked_anatomy_scan())
    plain = plot_scan(_image())

    with pytest.raises(ValueError, match="scanner coordinates"):
        axial.show_slices_of(plain)
    with pytest.raises(ValueError, match="one colour per view"):
        axial.show_slices_of(plot_scan(_coronal_scan()), colours=[])
    with pytest.raises(ValueError, match="#rrggbb"):
        axial.show_slices_of(plot_scan(_coronal_scan()), colours=["red"])
    with pytest.raises(ValueError, match="other ScanView"):
        axial.show_slices_of(axial)


def test_side_by_side_views_show_each_others_slices():
    box = plot_side_by_side(_stacked_anatomy_scan(), _coronal_scan())
    (row,) = box._grids
    axial, coronal = row.children

    assert isinstance(axial, ScanView) and isinstance(coronal, ScanView)
    # A grid, whose rows line up the plots of the views (viewer.css).
    assert row.layout.grid_template_columns.startswith("minmax(0, ")
    axial_colour = axial.layout.border_top.split()[-1]
    coronal_colour = coronal.layout.border_top.split()[-1]
    assert axial_colour != coronal_colour
    assert [line["colour"] for line in axial._slice_lines] == [coronal_colour]
    assert [line["colour"] for line in coronal._slice_lines] == [axial_colour]
    # The axial slice z = 0 mm (2 mm apart) on coronal rows z = row - 4.
    assert _band_rows(coronal) == [3.0, 5.0]


def test_plot_scan_shows_the_given_name_above_the_plots():
    assert plot_scan(_image()).name == ""
    view = plot_scan(_image(), name="E3 · anatomy")
    assert view.name == "E3 · anatomy"


def _column_shares(row):
    """The fr shares of the columns of one side-by-side grid, left to right."""
    return [
        int(share) for share in re.findall(r"(\d+)fr", row.layout.grid_template_columns)
    ]


def test_side_by_side_shares_the_width_by_what_each_view_needs():
    # Each column gets the width its image needs at the full plot height,
    # plus room for a curve, so that both images fill the same height.
    box = plot_side_by_side(
        [[plot_scan(_csi()), _coronal_scan()]], show_slices=False, share_masks=False
    )
    (row,) = box._grids
    csi, coronal = row.children

    assert _column_shares(row) == [
        _side_by_side_width(csi),
        _side_by_side_width(coronal),
    ]
    span = abs(np.subtract(*coronal._view["x_range"])) / abs(
        np.subtract(*coronal._view["y_range"])
    )
    assert _side_by_side_width(coronal) == round(coronal.plot_height * span)
    assert _side_by_side_width(csi) > SIDE_BY_SIDE_CURVE_WIDTH_PX


def test_side_by_side_curve_asks_for_width_in_proportion_to_plot_height():
    # A spectrum asks for a 3:2 landscape at the plot height (at least its
    # least width), so that one plane does not take the width of a row from
    # views that also show spectra or time courses.
    csi = plot_scan(_csi())
    shown = csi._view
    span = abs(np.subtract(*shown["x_range"])) / abs(np.subtract(*shown["y_range"]))
    curve = max(
        SIDE_BY_SIDE_CURVE_WIDTH_PX, SIDE_BY_SIDE_CURVE_ASPECT * csi.plot_height
    )

    assert _side_by_side_width(csi) == round(csi.plot_height * span + curve)


def test_side_by_side_puts_views_with_curves_in_rows_of_their_own():
    # Image-only views share the first row; each view with a spectrum gets
    # the width of a row below, in the given order.
    first, second = plot_scan(_csi()), plot_scan(_csi())
    axial, coronal = plot_scan(_stacked_anatomy_scan()), plot_scan(_coronal_scan())
    box = plot_side_by_side(
        first, axial, second, coronal, show_slices=False, share_masks=False
    )
    assert box.rows == ((axial, coronal), (first,), (second,))
    # Colours and the mask tool still follow the given order.
    assert first.layout.border_top != axial.layout.border_top

    curves = plot_side_by_side(
        plot_scan(_csi()), plot_scan(_csi()), show_slices=False, share_masks=False
    )
    assert [len(row) for row in curves.rows] == [1, 1]


def test_side_by_side_takes_rows_as_one_list_of_lists():
    axial, coronal, csi = (
        plot_scan(_stacked_anatomy_scan()),
        plot_scan(_coronal_scan()),
        plot_scan(_csi()),
    )
    box = plot_side_by_side(
        [[axial, csi], [coronal]], show_slices=False, share_masks=False
    )
    assert box.rows == ((axial, csi), (coronal,))

    with pytest.raises(ValueError, match="row 2 must be a non-empty list"):
        plot_side_by_side([[axial], []], show_slices=False, share_masks=False)
    with pytest.raises(ValueError, match="row 1 must be a non-empty list"):
        plot_side_by_side([axial, coronal], show_slices=False, share_masks=False)
    with pytest.raises(ValueError, match="at least two"):
        plot_side_by_side([[axial]])


def test_side_by_side_switches_between_rows_side_by_side_and_stacked():
    # The buttons "arrange" move the same views; links and settings stay.
    axial, csi = plot_scan(_stacked_anatomy_scan()), plot_scan(_csi())
    coronal = plot_scan(_coronal_scan())
    box = plot_side_by_side(axial, csi, coronal, show_slices=False, share_masks=False)
    switch, _menus, _status, rows_box = box.children
    assert switch.value == "rows"
    assert box.rows == ((axial, coronal), (csi,))

    switch.value = "side by side"
    assert box.arrangement == "side by side"
    assert box.rows == ((axial, csi, coronal),)
    assert tuple(tuple(grid.children) for grid in rows_box.children) == box.rows
    assert _column_shares(box._grids[0]) == [
        _side_by_side_width(view) for view in (axial, csi, coronal)
    ]

    box.arrangement = "stacked"
    assert switch.value == "stacked"
    assert box.rows == ((axial,), (csi,), (coronal,))
    assert box.views == (axial, csi, coronal)
    # Columns of the shown grids still follow a rotation.
    coronal.set_rotation(90)
    assert _column_shares(box._grids[2]) == [_side_by_side_width(coronal)]

    started = plot_side_by_side(
        axial, coronal, show_slices=False, share_masks=False, arrangement="stacked"
    )
    assert started.rows == ((axial,), (coronal,))
    with pytest.raises(ValueError, match="arrangement: 'grid' is unknown"):
        plot_side_by_side(axial, coronal, arrangement="grid")


def test_timings_are_shown_only_when_asked_for():
    # The status line shows redraw and block times only with show_timings;
    # plot_side_by_side sets them for every view, and a view made by the
    # menu "under" takes the setting of its upper view.
    assert plot_scan(_coronal_scan()).show_timings is False
    assert plot_scan(_coronal_scan(), show_timings=True).show_timings is True

    upper, axial = plot_scan(_stacked_anatomy_scan()), plot_scan(_coronal_scan())
    kept = plot_side_by_side(upper, axial, show_slices=False, share_masks=False)
    assert [view.show_timings for view in kept.views] == [False, False]

    box = plot_side_by_side(
        _stacked_anatomy_scan(), _stacked_anatomy_scan(), show_timings=True
    )
    assert [view.show_timings for view in box.views] == [True, True]
    box.set_underlays({1: 0})
    (over,) = box.views
    assert over not in box.given_views and over.show_timings is True


def test_side_by_side_lays_a_view_under_another():
    # The menu "under" of a view lays it under another view: that one is
    # made anew with the lower image underneath, and the lower view hides.
    axial_scan, coronal_scan = _stacked_anatomy_scan(), _coronal_scan()
    upper = plot_scan(_stacked_anatomy_scan(), name="upper")
    axial = plot_scan(axial_scan, name="axial")
    coronal = plot_scan(coronal_scan)
    box = plot_side_by_side(upper, axial, coronal)
    _switch, menus_row, status, _rows_box = box.children
    menus = menus_row.children
    assert [menu.description for menu in menus] == [
        "upper under",
        "axial under",
        "view 3 under",
    ]

    menus[1].value = 0
    over, shown_coronal = box.views
    assert shown_coronal is coronal and over is not upper
    assert over._model.anatomy_obj is axial_scan and over.name == "upper over axial"
    assert box.underlays == {1: 0} and box.given_views == (upper, axial, coronal)
    assert axial not in [view for row in box.rows for view in row]
    # The shown views are linked again; the hidden ones are not linked.
    assert [source for source, _ in over._slice_sources] == [coronal]
    assert axial._slice_sources == [] and upper._slice_sources == []
    assert over._mask_partners == [coronal]

    # Both anatomies under the upper view: one view over each.
    box.set_underlays({1: 0, 2: 0})
    over_axial, over_coronal = box.views
    assert over_coronal._model.anatomy_obj is coronal_scan
    assert over_axial._frame_partners == [over_coronal]
    assert [menu.value for menu in menus] == [-1, 0, 0]
    box.set_underlays({1: 0, 2: 0})
    assert box.views == (over_axial, over_coronal)

    # "none" shows the given views again, linked as before.
    menus[1].value = -1
    menus[2].value = -1
    assert box.views == (upper, axial, coronal) and box.underlays == {}
    assert upper._anatomy == b""
    assert [source for source, _ in upper._slice_sources] == [axial, coronal]
    assert box.rows == ((upper, axial, coronal),)

    # A view under another one cannot have one underneath; nothing changes.
    menus[1].value = 0
    menus[0].value = 2
    assert menus[0].value == -1 and box.underlays == {1: 0}
    assert "Could not lay the views" in status.value
    with pytest.raises(ValueError, match="view 0 has view 1 underneath"):
        box.set_underlays({1: 0, 0: 2})
    with pytest.raises(ValueError, match="5 is not the index of a given view"):
        box.set_underlays({5: 0})
    with pytest.raises(ValueError, match="cannot lie under itself"):
        box.set_underlays({1: 1})

    hidden = plot_side_by_side(
        axial, coronal, show_slices=False, share_masks=False, under_menus=False
    )
    assert len(hidden.children) == 3


def test_side_by_side_columns_follow_rotation_and_plot_height():
    # A rotated image needs another width; the column follows, so that the
    # rotated image is not shrunk into the column of the old plane.
    box = plot_side_by_side(
        _stacked_anatomy_scan(), _coronal_scan(), show_slices=False, share_masks=False
    )
    (row,) = box._grids
    axial, coronal = row.children
    before = _column_shares(row)

    coronal.set_rotation(90)
    rotated = _column_shares(row)
    assert rotated == [_side_by_side_width(axial), _side_by_side_width(coronal)]
    # The coronal plane is wider than high (fixture), so it narrows.
    assert rotated[1] < before[1]

    axial.plot_height = 2 * axial.plot_height
    assert _column_shares(row)[0] == _side_by_side_width(axial)


def test_side_by_side_keeps_given_views_and_needs_two_images():
    axial = plot_scan(_stacked_anatomy_scan())
    box = plot_side_by_side(
        axial, plot_scan(_image()), show_slices=False, share_masks=False
    )

    assert list(box.views)[0] is axial
    assert axial._slice_lines == [] and axial._mask_link == ""
    with pytest.raises(ValueError, match="scanner coordinates"):
        plot_side_by_side(axial, plot_scan(_image()), show_slices=False)
    with pytest.raises(ValueError, match="at least two"):
        plot_side_by_side(axial)


def _assert_pointer_plane_maps_pixels_to_lps(view, scan):
    """Every displayed pixel centre maps to the LPS position of its voxel."""
    plane = view._pointer_plane
    shown = view._view
    data = scan.primary_data
    slice_index = view.state.spatial.slice_indices.get("slice_idx", 0)
    for row, y in zip(shown["row_indices"], shown["y"], strict=True):
        for column, x in zip(shown["column_indices"], shown["x"], strict=True):
            voxel = data.isel(
                {shown["row_dim"]: row, shown["column_dim"]: column}
            ).isel(slice_idx=slice_index)
            expected = [float(voxel[name]) for name in ("x", "y", "z")]
            lps = (
                np.asarray(plane["origin_mm"])
                + x * np.asarray(plane["x_step_mm"])
                + y * np.asarray(plane["y_step_mm"])
            )
            # Geometry of an invented 1 mm grid; only rounding error expected.
            np.testing.assert_allclose(lps, expected, atol=1e-9)


def test_side_by_side_views_share_the_pointer_in_scanner_coordinates():
    axial_scan, coronal_scan = _stacked_anatomy_scan(), _coronal_scan()
    box = plot_side_by_side(axial_scan, coronal_scan)
    axial, coronal = list(box.views)

    assert axial._pointer_link and axial._pointer_link == coronal._pointer_link
    _assert_pointer_plane_maps_pixels_to_lps(axial, axial_scan)
    _assert_pointer_plane_maps_pixels_to_lps(coronal, coronal_scan)
    assert axial._pointer_plane["thickness_mm"] == pytest.approx(2.0)

    # The map follows a new slice and a rotated image.
    axial._controls = {
        **axial._controls,
        "indices": {**axial._controls["indices"], "slice_idx": 4},
    }
    axial.set_rotation(90)
    _assert_pointer_plane_maps_pixels_to_lps(axial, axial_scan)


def test_link_pointer_replaces_and_removes_links_and_checks_views():
    axial = plot_scan(_stacked_anatomy_scan())
    coronal = plot_scan(_coronal_scan())
    assert axial._pointer_link == "" and axial._pointer_plane == {}

    axial.link_pointer(coronal)
    first = axial._pointer_link
    axial.link_pointer(coronal)
    assert axial._pointer_link == coronal._pointer_link != first

    axial.link_pointer()
    assert axial._pointer_link == "" and axial._pointer_plane == {}
    with pytest.raises(ValueError, match="scanner coordinates"):
        axial.link_pointer(plot_scan(_image()))
    with pytest.raises(ValueError, match="other ScanView"):
        axial.link_pointer(axial)


def _click_voxel(view, **source_index):
    """Select the shown voxel with these read/phase indices, as a click does."""
    shown = view._view
    row = shown["row_indices"].index(source_index[shown["row_dim"]])
    column = shown["column_indices"].index(source_index[shown["column_dim"]])
    view.selection = {**view.selection, "row": row, "column": column}


def test_a_click_reaches_a_csi_whose_slab_misses_the_clicked_slice():
    # Coronal anatomy, single-slice CSI (slab z = -1..1 mm) and axial
    # anatomy, in this order as a study shows them by measurement time.
    csi_data = _with_scanner_coordinates(_csi())
    csi_data.attrs["slice_thickness_mm"] = 2.0
    csi_scan = SimpleNamespace(primary_data=csi_data, is_spectroscopic=True)
    box = plot_side_by_side(_coronal_scan(), csi_scan, _stacked_anatomy_scan())
    (coronal, axial), (csi,) = box.rows
    axial.state.spatial.slice_indices["slice_idx"] = 0  # z = -4 mm

    # x = 6, y = 1 mm: CSI read 3 (x = 2 * read), phase 3 (y = 10 - 3 * phase).
    _click_voxel(axial, read_idx=6, phase_idx=1)
    indices = csi.state.spatial.slice_indices
    assert (indices["read_idx"], indices["phase_idx"], indices["slice_idx"]) == (
        3,
        3,
        0,
    )
    # The coronal view takes its nearest slice y = 3 mm at z = -4 mm.
    indices = coronal.state.spatial.slice_indices
    assert (indices["read_idx"], indices["phase_idx"], indices["slice_idx"]) == (
        6,
        0,
        0,
    )

    # x = 12 mm lies beside the CSI plane (x = 0..8 mm): its selection stays.
    before = dict(csi.state.spatial.slice_indices)
    _click_voxel(axial, read_idx=12, phase_idx=1)
    assert csi.state.spatial.slice_indices == before


def test_a_clicked_voxel_is_selected_in_pointer_linked_views():
    axial, coronal = plot_side_by_side(_stacked_anatomy_scan(), _coronal_scan()).views
    axial_slice = axial.state.spatial.slice_indices["slice_idx"]  # z = 0 mm

    # Axial voxel at x = 6, y = 7 mm lies in the coronal slice y = 7 mm
    # (index 2) at x = 6 mm (read 6) and z = 0 mm (phase 4).
    _click_voxel(axial, read_idx=6, phase_idx=7)
    indices = coronal.state.spatial.slice_indices
    assert (indices["read_idx"], indices["phase_idx"], indices["slice_idx"]) == (
        6,
        4,
        2,
    )
    shown = coronal._view
    assert shown["row_indices"][coronal.selection["row"]] == indices[shown["row_dim"]]
    assert (
        shown["column_indices"][coronal.selection["column"]]
        == indices[shown["column_dim"]]
    )

    # The other way round the axial view moves to the slice z = 2 mm.
    _click_voxel(coronal, read_idx=3, phase_idx=6)
    indices = axial.state.spatial.slice_indices
    assert (indices["read_idx"], indices["phase_idx"], indices["slice_idx"]) == (
        3,
        7,
        3,
    )
    assert axial_slice == 2

    # y = 14 mm lies beyond the coronal stack (y = 3, 5, 7 mm): the coronal
    # view takes its nearest slice y = 7 mm at x = 6 mm and z = 2 mm (phase 6).
    _click_voxel(axial, read_idx=6, phase_idx=14)
    indices = coronal.state.spatial.slice_indices
    assert (indices["read_idx"], indices["phase_idx"], indices["slice_idx"]) == (
        6,
        6,
        2,
    )
    before = dict(indices)

    coronal.unlink()
    _click_voxel(axial, read_idx=9, phase_idx=7)
    assert coronal.state.spatial.slice_indices == before
    assert axial._pointer_partners == []


def _draw_box(view, mask, erase=False):
    """Draw a box over displayed columns 2..7 and rows 1..6 of the map."""
    x, y = view._view["x"], view._view["y"]
    view.draw_polygon(
        [(x[2], y[1]), (x[7], y[1]), (x[7], y[6]), (x[2], y[6])],
        mask=mask,
        erase=erase,
    )


def _axial_mask_on_coronal_grid(axial_mask):
    """Expected coronal coverage: each coronal voxel takes the axial voxel at
    its centre. Axial slices lie at z = -4..4 mm, 2 mm apart, so coronal rows
    z = 2 j - 4 - 1 and 2 j - 4 (mm) fall into axial slice j; columns and
    pixels share x = 0..19 mm, and coronal slices y = 3, 5, 7 mm are axial
    rows 3, 5, 7."""
    expected = np.zeros((20, 9, 3), dtype=np.float32)
    for row, z in enumerate(np.arange(-4.0, 5.0)):
        axial_slice = int(np.floor((z + 4.0) / 2.0 + 0.5))
        if axial_slice < 5:
            expected[:, row, :] = axial_mask[:, [3, 5, 7], axial_slice]
    return expected


def test_side_by_side_views_share_masks_on_both_scans():
    axial_scan, coronal_scan = _stacked_anatomy_scan(), _coronal_scan()
    axial, coronal = plot_side_by_side(axial_scan, coronal_scan).views
    assert axial._mask_link and axial._mask_link == coronal._mask_link
    # One mask tool: only the first image shows its inputs.
    assert axial._mask_tool_shown and not coronal._mask_tool_shown
    coronal.link_masks()
    assert coronal._mask_tool_shown and axial._mask_tool_shown
    axial.link_masks(coronal)
    assert axial._mask_link == coronal._mask_link

    axial.add_mask("lesion", color="#00ff00", alpha=0.5)
    assert coronal.masks["lesion"].attrs["color"] == "#00ff00"
    assert float(coronal.masks["lesion"].sum()) == 0.0

    _draw_box(axial, "lesion")
    drawn = axial.masks["lesion"].values
    shared = coronal.masks["lesion"].values
    assert drawn.sum() > 0 and shared.sum() > 0
    np.testing.assert_array_equal(shared, _axial_mask_on_coronal_grid(drawn))
    # Each scan stores the mask on its own grid.
    assert coronal_scan.masks["lesion"].shape == (20, 9, 3)
    assert axial_scan.masks["lesion"].shape == (20, 15, 5)

    # An outline drawn on the coronal view reaches the axial scan.
    _draw_box(coronal, "lesion")
    assert axial.masks["lesion"].values.sum() > drawn.sum()

    coronal.undo_mask_edit()
    axial.undo_mask_edit()
    assert float(axial.masks["lesion"].sum()) == 0.0
    assert float(coronal.masks["lesion"].sum()) == 0.0

    coronal.set_mask_style("lesion", color="#0000ff")
    assert axial.masks["lesion"].attrs["color"] == "#0000ff"
    axial.remove_mask("lesion")
    assert "lesion" not in coronal.masks and "lesion" not in coronal_scan.masks


def test_linked_views_share_the_mask_tool_and_existing_masks():
    axial = plot_scan(_stacked_anatomy_scan())
    coronal = plot_scan(_coronal_scan())
    axial.add_mask("early")
    _draw_box(axial, "early")

    axial.link_masks(coronal)
    # Views linked outside plot_side_by_side keep their own tool inputs.
    assert axial._mask_tool_shown and coronal._mask_tool_shown
    np.testing.assert_array_equal(
        coronal.masks["early"].values,
        _axial_mask_on_coronal_grid(axial.masks["early"].values),
    )
    coronal.tool = "masks"
    assert axial.tool == "masks"
    axial.tool = ""
    assert coronal.tool == ""

    axial.link_masks()
    assert axial._mask_link == "" and coronal._mask_partners == []
    axial.add_mask("alone")
    assert "alone" not in coronal.masks
    with pytest.raises(ValueError, match="scanner coordinates"):
        axial.link_masks(plot_scan(_image()))
    with pytest.raises(ValueError, match="other ScanView"):
        axial.link_masks(axial)


def test_image_with_another_image_underneath():
    # Invented second image of the axial stack: 2 mm x 1.5 mm pixels.
    grid_dims = ("read_idx", "phase_idx", "slice_idx")
    x, y, z = np.meshgrid(
        np.arange(0.0, 20.0, 2.0),
        np.arange(0.0, 15.0, 1.5),
        [-2.0, 0.0, 2.0],
        indexing="ij",
    )
    second = xr.DataArray(
        -(x + y),
        dims=grid_dims,
        coords={
            name: (grid_dims, value, {"unit": "mm"})
            for name, value in (("x", x), ("y", y), ("z", z))
        },
    )
    view = plot_scan(_stacked_anatomy_scan(), anatomy=second)

    assert view.state.spatial.coordinate_mode == "xyz"
    layer = view._view["anatomy"]
    underneath = np.frombuffer(view._anatomy, dtype=np.float32)
    assert underneath.size == np.prod(layer["shape"])
    # Only the second image (values <= 0) lies underneath.
    finite = underneath[np.isfinite(underneath)]
    assert finite.size and np.all(finite <= 0)
    view.set_anatomy_opacity(0.3)
    assert view._anatomy_opacity == 0.3


def _block_at_repetition(repetition):
    state = _state()
    state.spatial.slice_indices["repetition_idx"] = repetition
    view, _ = _view(state=state)
    return _block(view).astype(float)


def test_repetition_average_and_cumulative_sum_over_a_range():
    # The invented amplitudes grow linearly with the repetition at one phase,
    # so the magnitude of the mean (sum) is the mean (sum) of the magnitudes.
    single = [_block_at_repetition(rep) for rep in range(3)]
    view, _ = _view()

    view.set_processing(repetitions="average", repetition_range=(1, 2))
    spectral = view.state.spectral
    assert spectral.average_repetitions and not spectral.cum_sum_repetitions
    assert (spectral.rep_avg_start, spectral.rep_avg_end) == (1, None)
    assert "repetition_idx" not in view._view["other_dims"]
    assert "repetition_idx" not in view._controls["indices"]
    np.testing.assert_allclose(_block(view), (single[1] + single[2]) / 2, rtol=1e-5)

    view.set_processing(repetitions="cumulative sum", repetition_range=(0, 2))
    assert spectral.cum_sum_repetitions and not spectral.average_repetitions
    assert view._view["other_dims"]["repetition_idx"] == 3
    # The shown repetition is 1: the running sum of repetitions 0 and 1.
    np.testing.assert_allclose(_block(view), single[0] + single[1], rtol=1e-5)
    assert view._controls["processing"]["repetition_range"] == [0, 2]


def test_number_of_points_and_fid_view():
    view, data = _view()
    view.set_processing(npoints=128)
    assert view.state.spectral.npoints == 128
    assert view._view["shape"][2] == 128 and len(view._view["curve_x"]) == 128
    assert view._controls["processing"]["npoints"] == 128
    view.set_processing(npoints=64)
    assert view.state.spectral.npoints is None

    view.set_processing(show_fid=True)
    assert view.state.spectral.show_fid
    block = _block(view)
    row = view._view["row_indices"].index(1)
    column = view._view["column_indices"].index(2)
    # The FID magnitude of the shown voxel at repetition 1 is the acquired one.
    acquired = np.abs(data.isel(read_idx=1, phase_idx=2, slice_idx=0, channel_idx=0))
    rows_are_read = view._view["row_dim"] == "read_idx"
    voxel = block[row, column] if rows_are_read else block[column, row]
    np.testing.assert_allclose(voxel, acquired.isel(repetition_idx=1), rtol=1e-5)
    assert np.all(np.diff(view._view["curve_x"]) > 0)


def test_channel_combination_hides_the_channel_slider():
    data = xr.concat([_csi(), 2 * _csi()], dim="channel_idx").transpose(*DIMS)
    view, _ = _view(data)
    assert view._controls["processing"]["combine_channels"]
    assert "channel_idx" not in view._view["other_dims"]
    combined = _block(view).copy()

    view.set_processing(combine_channels=False)
    assert view._view["other_dims"]["channel_idx"] == 2
    first_channel = _block(view)
    # Root sum of squares of channels with amplitude 1 and 2 at one phase.
    np.testing.assert_allclose(combined, np.sqrt(5) * first_channel, rtol=1e-5)

    view.set_processing(combine_channels=True, combine_method="sum")
    assert view.state.spectral.combine_method == "sum"
    np.testing.assert_allclose(_block(view), 3 * first_channel, rtol=1e-5)


def test_processing_is_validated_without_partial_changes():
    view, _ = _view()
    before = copy.deepcopy(view.state.spectral)
    with pytest.raises(ValueError, match="combine_channels does not apply"):
        view.set_processing(combine_channels=False)
    with pytest.raises(ValueError, match="npoints must be at least 2"):
        view.set_processing(npoints=1, repetitions="average")
    with pytest.raises(ValueError, match="repetition_range must lie within 0 and 2"):
        view.set_processing(repetitions="average", repetition_range=(0, 3))
    with pytest.raises(ValueError, match="repetitions must be one of"):
        view.set_processing(repetitions="median")
    with pytest.raises(ValueError, match="denoise_components must be 0 or more"):
        view.set_processing(denoise_components=-1)
    assert view.state.spectral == before
    image = _image()
    image_view = plot_scan(image)
    assert "show_fid" not in image_view._controls["processing"]
    with pytest.raises(ValueError, match="show_fid does not apply"):
        image_view.set_processing(show_fid=True)


def test_browser_processing_control_recomputes_the_block():
    view, data = _view()
    controls = copy.deepcopy(view._controls)
    controls["processing"]["repetitions"] = "average"
    view._controls = controls
    assert view.state.spectral.average_repetitions
    expected = _model(data).get_render_payload(
        data, view.state, include_curve_block=True
    )
    np.testing.assert_array_equal(_block(view), expected.curve_block)
    with pytest.raises(Exception, match="repetitions must be one of"):
        bad = copy.deepcopy(view._controls)
        bad["processing"]["repetitions"] = "median"
        view._controls = bad


def test_curve_scale_follows_the_state_and_the_browser():
    view, _ = _view()
    assert view._curve_axes == {
        "scale": "auto",
        "window": None,
        "window_unit": "ppm",
        "fit": True,
        "residual": True,
    }
    block = view._block

    # "map colours" is couple_ylims of plot and is stored in the state.
    view.set_curve_scale("map")
    assert view.state.spectral.couple_ylims is True
    assert view._curve_axes["scale"] == "map"
    # "whole plane" belongs to the view only.
    view._curve_axes = {**view._curve_axes, "scale": "plane"}
    assert view.state.spectral.couple_ylims is False
    assert view._curve_axes_settings()["scale"] == "plane"
    # The scale only redraws the curve panel in the browser.
    assert view._block is block

    with pytest.raises(ValueError, match="curve scale must be one of"):
        view.set_curve_scale("shared")
    assert view._curve_axes["scale"] == "plane"


def test_frequency_window_is_stored_in_the_shown_unit():
    view, _ = _view()
    block = view._block
    view.set_frequency_window((3.0, -2.0))
    spectral = view.state.spectral
    assert spectral.display_range == (3.0, -2.0)
    assert spectral.display_range_unit == spectral.spec_unit == "ppm"
    assert view._curve_axes["window"] == [3.0, -2.0]
    assert view._block is block

    view.set_frequency_window((-100, 250), unit="Hz")
    assert spectral.display_range == (-100.0, 250.0)
    assert spectral.display_range_unit == "Hz"
    view.set_frequency_window(None)
    assert spectral.display_range is None

    # A state with a window starts the view with it.
    state = _state()
    state.spectral.display_range = (1.0, 2.0)
    state.spectral.couple_ylims = True
    started, _ = _view(state=state)
    assert started._curve_axes == {
        "scale": "map",
        "window": [1.0, 2.0],
        "window_unit": "ppm",
        "fit": True,
        "residual": True,
    }


@pytest.mark.parametrize(
    ("window", "unit", "message"),
    [
        ((1.0, 1.0), None, "two different limits"),
        ((1.0, float("nan")), None, "two finite limits"),
        ((1.0, 2.0, 3.0), None, "two finite limits"),
        ((1.0, 2.0), "kHz", "unit must be one of"),
    ],
)
def test_invalid_frequency_window_is_rejected_without_changes(window, unit, message):
    view, _ = _view()
    view.set_frequency_window((0.0, 5.0))
    with pytest.raises(ValueError, match=message):
        view.set_frequency_window(window, unit=unit)
    assert view.state.spectral.display_range == (0.0, 5.0)
    assert view._curve_axes["window"] == [0.0, 5.0]


def test_frequency_window_needs_a_spectrum():
    view = plot_scan(_image(extra="repetition_idx"))
    with pytest.raises(ValueError, match="needs data with a spectrum"):
        view.set_frequency_window((0.0, 1.0))
    # The curve scale applies to every curve.
    view.set_curve_scale("plane")
    assert view._curve_axes["scale"] == "plane"


# ── plot_study ──────────────────────────────────────────────────────────


class _Study:
    """Study stand-in: invented scans keyed by experiment number, with the
    summary columns that name a scan in the list (invented values)."""

    def __init__(self, scans, times=None):
        import pandas as pd

        keys = list(scans)
        self._scans = scans
        self.dataframe = pd.DataFrame(
            {
                "Scan Name": [f"scan {key}" for key in keys],
                "SequenceClass": ["Invented"] * len(keys),
                "Measurement Time": times or [f"10:0{i}:00" for i in range(len(keys))],
            },
            index=keys,
        )

    def __getitem__(self, key):
        return self._scans[key]


def _study():
    return _Study(
        {
            1: _stacked_anatomy_scan(),
            2: _coronal_scan(),
            3: _image(),
            4: SimpleNamespace(primary_data=None, is_spectroscopic=False),
        }
    )


def test_plot_study_lists_scans_and_shows_none_at_first():
    from hypermri.visualization.study_view import plot_study, scan_label

    study = _study()
    view = plot_study(study)

    assert view.views == {}
    assert scan_label(study, 1) == "E1 · scan 1 · Invented · 10:00:00"
    assert view._list.get_title(0) == "Scans (0 of 4 shown)"
    assert [view._ticks[key].description for key in (1, 2)] == [
        "E1 · scan 1 · Invented · 10:00:00",
        "E2 · scan 2 · Invented · 10:01:00",
    ]
    assert all(under.disabled for under in view._unders.values())


def test_plot_study_links_shown_scans_and_keeps_their_views():
    from hypermri.visualization.study_view import plot_study

    view = plot_study(_study(), scans=[2, 1])
    axial, coronal = view.views[1], view.views[2]
    assert list(view.views) == [1, 2]  # the study's order
    assert axial._mask_link and axial._mask_link == coronal._mask_link
    assert axial._slice_lines and coronal._slice_lines
    assert axial._mask_tool_shown and not coronal._mask_tool_shown
    assert view._ticks[1].value and view._ticks[2].value
    assert view._list.get_title(0) == "Scans (2 of 4 shown)"
    assert axial.name == "E1 · scan 1 · Invented · 10:00:00"

    view.select([1])
    assert view.views == {1: axial}  # same view, settings kept
    assert axial._mask_link == "" and axial._slice_lines == []
    assert axial._pointer_link == "" and axial._mask_tool_shown
    assert view._area.children == (axial,)
    assert not view._ticks[2].value and view._unders[2].disabled

    # Ticking a scan in the list shows it next to the kept view.
    view._ticks[2].value = True
    assert view.views[1] is axial and axial._mask_link == view.views[2]._mask_link


def test_plot_study_keeps_the_chosen_arrangement():
    from hypermri.visualization.study_view import plot_study

    view = plot_study(_study(), scans=[1, 2])
    (box,) = view._area.children
    box.arrangement = "stacked"

    view._ticks[3].value = True
    (shown,) = view._area.children
    assert shown is not box and shown.arrangement == "stacked"
    assert len(shown.rows) == 3


def test_plot_study_orders_the_list_by_measurement_time():
    from hypermri.visualization.study_view import plot_study, scans_by_time

    study = _Study(
        {key: _image() for key in (5, 6, 7, 8)},
        times=["16:00:00", "MISSING", "09:30:00", "11:15:00"],
    )
    assert scans_by_time(study) == [7, 8, 5, 6]
    assert plot_study(study)._keys == [7, 8, 5, 6]


def test_plot_study_lays_a_scan_under_another():
    from hypermri.visualization.study_view import NO_UNDERLAY, plot_study

    view = plot_study(_study(), scans=[1])
    alone = view.views[1]
    assert alone._anatomy == b""

    # Python: scan 2 under scan 1 is one view of scan 1, none of scan 2.
    view.select([1], {1: 2})
    upper = view.views[1]
    assert list(view.views) == [1] and upper is not alone and upper._anatomy != b""
    assert view._ticks[2].value and view._unders[2].value == 1
    assert view._unders[1].value == NO_UNDERLAY
    assert view._list.get_title(0) == "Scans (2 of 4 shown)"

    # List: the row of the lower scan names the scan it lies under.
    view.select([])
    view._ticks[2].value = True
    view._unders[2].value = 1
    assert list(view.views) == [1] and view.views[1]._anatomy != b""
    assert view._ticks[1].value

    # Unticking the upper scan shows the lower one alone.
    view._ticks[1].value = False
    assert list(view.views) == [2] and view.views[2]._anatomy == b""
    assert view._unders[2].value == NO_UNDERLAY


def test_plot_study_lays_several_scans_under_one():
    from hypermri.visualization.study_view import plot_study

    study = _Study(
        {1: _stacked_anatomy_scan(), 2: _coronal_scan(), 3: _stacked_anatomy_scan()}
    )
    view = plot_study(study, scans=[3], underlays={3: [2, 1]})
    # One image of scan 3 over each scan below; the time curve only once.
    assert list(view.views) == [3]
    assert list(view.underlay_views) == [(3, 2), (3, 1)]
    over_coronal, over_axial = view.underlay_views.values()
    assert view.views[3] is over_coronal
    assert over_coronal._model.anatomy_obj is study[2]
    assert over_axial._model.anatomy_obj is study[1]
    assert over_coronal.state.spatial.slicing_dim == "phase_idx"
    assert over_axial.state.spatial.slicing_dim == "slice_idx"
    assert over_coronal._layout["panels"] != ["image"]
    assert over_axial._layout["panels"] == ["image"]
    assert over_coronal._frame_partners == [over_axial]
    assert over_coronal.state.spatial.cmap == "turbo"
    assert view._unders[1].value == 3 and view._unders[2].value == 3
    assert view._list.get_title(0) == "Scans (3 of 3 shown)"

    # The same choice in the list keeps the views.
    view._ticks[3].value = False
    view._ticks[3].value = True
    view._ticks[2].value = True
    view._unders[2].value = 3
    view._ticks[1].value = True
    view._unders[1].value = 3
    assert view._status.value == ""
    assert list(view.views) == [3]
    assert {under for _, under in view.underlay_views} == {1, 2}


def test_plot_study_shows_scans_without_coordinates_unlinked():
    from hypermri.visualization.study_view import plot_study

    view = plot_study(_study(), scans=[1, 3])
    assert set(view.views) == {1, 3}
    assert view.views[1]._mask_link == "" and view.views[1]._slice_lines == []
    assert "Views not linked" in view._status.value


@pytest.mark.parametrize(
    ("scans", "underlays", "message"),
    [
        ([9], None, "not a scan of the study"),
        ([1], {1: 1}, "cannot lie under itself"),
        ([1], {2: 1}, "needs 2 to be shown"),
        ([1], {1: 9}, "not a scan of the study"),
        ([1, 2], {1: 2}, "cannot also be shown on its own"),
        ([1, 3], {1: 2, 3: 2}, "can lie under only one scan"),
        ([1], {1: [2, 2]}, "can lie under only one scan"),
        ([1], {1: [2, 1]}, "cannot lie under itself"),
    ],
)
def test_plot_study_rejects_invalid_choices_without_changes(scans, underlays, message):
    from hypermri.visualization.study_view import plot_study

    view = plot_study(_study(), scans=[2])
    shown = view.views
    with pytest.raises(ValueError, match=message):
        view.select(scans, underlays)
    assert view.views == shown and view._ticks[2].value and not view._ticks[1].value


def test_plot_study_reports_a_scan_that_cannot_be_shown():
    from hypermri.visualization.study_view import plot_study

    view = plot_study(_study(), scans=[1])
    kept = view.views[1]
    with pytest.raises(ValueError, match="primary_data"):
        view.select([1, 4])
    assert view.views == {1: kept}

    view._ticks[4].value = True
    assert view.views == {1: kept} and not view._ticks[4].value
    assert "Could not show the scans" in view._status.value


def _fitted_csi():
    """Invented CSI FIDs with the time axis and dwell time a fit needs."""
    data = _csi(n_read=3, n_phase=2, n_rep=1, n_spec=128)
    time_s = np.arange(data.sizes["acquisition_idx"]) * 2e-4
    return data.assign_coords(
        sampling_time_axis_s=("acquisition_idx", time_s)
    ).assign_attrs(dwell_time_s=2e-4)


def _fit_message(view, action, **content):
    view._handle_custom_message(view, {"type": "fit", "action": action, **content}, [])
    return view._fit_tool


def test_fit_tool_is_offered_only_for_spectra():
    view, _ = _view(_fitted_csi())
    assert view._fit_tool["offered"] is True
    assert view._fit_tool["peaks"] == []
    image_view = plot_scan(_image(extra="repetition_idx"))
    assert image_view._fit_tool["offered"] is False
    settings = _fit_message(image_view, "detect")
    assert "not spectra" in settings["message"]


def _diffusion_image(adc_mm2_s=1.5e-3):
    """Invented diffusion images: s0 encodes read and phase, one ADC for all."""
    b_values = np.array([0.0, 250.0, 500.0, 1000.0])
    s0 = 1.0 + np.arange(8)[:, None, None] + 10.0 * np.arange(6)[None, :, None]
    values = s0[..., None] * np.exp(-b_values * adc_mm2_s)
    image = xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "slice_idx", "bvalue_idx"),
        coords={"bvalue_idx": ("bvalue_idx", b_values, {"unit": "s/mm2"})},
    )
    return SimpleNamespace(primary_data=image, is_spectroscopic=False)


def _curve_fit_message(view, action, **content):
    message = {"type": "curve_fit", "action": action, **content}
    view._handle_custom_message(view, message, [])
    return view._curve_fit_tool


def test_curve_fit_tool_follows_the_curve_axis():
    view = plot_scan(_diffusion_image())
    assert view._curve_dim == "bvalue_idx"
    assert view._curve_fit_tool["offered"] is True
    assert view._curve_fit_tool["model"] == "adc"
    assert plot_scan(_image(extra="repetition_idx"))._curve_fit_tool == {
        "fit_baseline": False,
        "message": "",
        "offered": False,
        "fitted": False,
    }
    spectra, _ = _view(_fitted_csi())
    assert spectra._curve_fit_tool["offered"] is False
    settings = _curve_fit_message(spectra, "run")
    assert "curve axis" in settings["message"]


def test_curve_fit_tool_run_shows_adc_map_fit_curve_and_residual():
    view = plot_scan(_diffusion_image(adc_mm2_s=1.5e-3))
    settings = _curve_fit_message(view, "run")

    assert settings["fitted"] is True
    assert "48 of 48 voxels" in settings["message"]
    assert view.state.overlay.active_variable == "curve_fit_adc"
    assert view.state.overlay.visible is True
    # Noise-free data: the log-linear fit recovers the ADC to rounding.
    np.testing.assert_allclose(view.curve_fit_result["adc"].values, 1.5e-3, rtol=1e-6)
    names = [curve["name"] for curve in view._fit_curves["curves"]]
    assert names[0].startswith("Fit ADC")
    assert any(name.startswith("Residual") for name in names)
    # The spectral fit candidates leave the curve-fit maps out.
    assert "curve_fit" not in view._fit_datasets()

    view.show_fit_curves(fit=False)
    assert view._fit_curves == {}
    settings = _curve_fit_message(view, "option", name="t1_values_s", value=1)
    assert "unknown setting" in settings["message"]


def _echo_image(offset_hz=15.0):
    """Invented complex echoes of two components (0 and 390 Hz) on 4 x 3 voxels.

    The first component encodes read and phase, the second is 2 everywhere;
    eight echoes are 0.8 ms apart and all voxels share one field offset.
    """
    spacing_s, frequencies = 0.0008, np.array([0.0, 390.0])
    t = (np.arange(8) - 3.5) * spacing_s
    first = 1.0 + np.arange(4)[:, None] + 10.0 * np.arange(3)[None, :]
    phase = np.exp(2j * np.pi * (frequencies[:, None] + offset_hz) * t)
    values = first[..., None, None] * phase[0] + 2.0 * phase[1]
    image = xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "slice_idx", "echo_idx"),
        coords={"echo_idx": ("echo_idx", np.arange(8) * spacing_s, {"unit": "s"})},
    )
    return SimpleNamespace(primary_data=image, is_spectroscopic=False)


def _ideal_message(view, action, **content):
    message = {"type": "ideal", "action": action, **content}
    view._handle_custom_message(view, message, [])
    return view._ideal_tool


def test_ideal_tool_separates_components_into_overlay_maps():
    view = plot_scan(_echo_image())
    assert view._curve_dim == "echo_idx"
    settings = view._ideal_tool
    assert settings["offered"] is True and settings["complex"] is True
    assert settings["axis_spacing_ms"] == pytest.approx(0.8)
    assert plot_scan(_diffusion_image())._ideal_tool["offered"] is False

    settings = _ideal_message(view, "run")
    assert "frequencies in Hz" in settings["message"]
    _ideal_message(view, "option", name="component_names", value="pyr, lac")
    _ideal_message(view, "option", name="frequencies_hz", value="0, 390")
    _ideal_message(view, "option", name="fit_theta", value=False)
    settings = _ideal_message(view, "run")

    assert settings["fitted"] is True
    assert "pyr, lac in 12 voxels" in settings["message"]
    assert view.state.overlay.active_variable == "ideal_pyr"
    # Noise-free echoes: amplitudes and offset are recovered to rounding.
    result = view.ideal_result
    np.testing.assert_allclose(
        np.abs(result["fit_amps"].sel(metabolite="lac").values), 2.0, rtol=1e-6
    )
    np.testing.assert_allclose(result["field_offset"].values, 15.0, atol=1e-6)
    assert "ideal" not in view._fit_datasets()
    settings = _ideal_message(view, "option", name="max_iterations", value=3)
    assert "unknown setting" in settings["message"]


def test_curve_fit_is_saved_on_the_scan_and_shown_when_plotted_again():
    scan = _diffusion_image(adc_mm2_s=1.5e-3)
    view = plot_scan(scan)
    settings = _curve_fit_message(view, "save", name="kidney")
    assert "has no fit yet" in settings["message"]
    _curve_fit_message(view, "run")

    settings = _curve_fit_message(view, "save", name=" kidney ")
    assert "'kidney' in scan.saved_fits" in settings["message"]
    saved = scan.saved_fits["kidney"]
    assert saved.kind == "adc" and saved.params is None
    xr.testing.assert_equal(saved.results, view.curve_fit_result)
    assert saved.results is not view.curve_fit_result
    assert view.saved_fits is scan.saved_fits
    # The saved overlay stays when the tool runs again.
    _curve_fit_message(view, "run")
    assert "kidney_adc" in view._overlay_names()

    again = plot_scan(scan)
    assert "kidney_adc" in again._overlay_names()
    assert again.state.overlay.active_variable == "kidney_adc"
    assert again.state.overlay.visible is False


def test_save_fit_checks_name_and_tool_and_numbers_unnamed_fits():
    view = plot_scan(_diffusion_image())
    _curve_fit_message(view, "run")
    for name, error in (
        ("2nd", "not valid"),
        ("a b", "not valid"),
        ("curve_fit", "already exists"),
    ):
        assert error in _curve_fit_message(view, "save", name=name)["message"]
    assert view.save_fit() == "adc_1"
    assert view.save_fit("", tool="curve") == "adc_2"
    assert view.save_fit("adc_1") == "adc_1"
    assert sorted(view.saved_fits) == ["adc_1", "adc_2"]
    with pytest.raises(ValueError, match="no fit yet"):
        view.save_fit("x", tool="ideal")
    with pytest.raises(ValueError, match="must be one of"):
        view.save_fit("x", tool="spectrum")


def test_ideal_fit_is_saved_on_a_plotted_dataarray():
    image = _echo_image().primary_data
    view = plot_scan(image)
    _ideal_message(view, "option", name="component_names", value="pyr, lac")
    _ideal_message(view, "option", name="frequencies_hz", value="0, 390")
    _ideal_message(view, "run")
    settings = _ideal_message(view, "save", name="sep")

    assert 'attrs["saved_fits"]' in settings["message"]
    assert image.attrs["saved_fits"]["sep"].kind == "ideal"
    assert "sep_pyr" in plot_scan(image)._overlay_names()


def test_saved_fit_of_other_voxels_is_not_shown():
    scan = _diffusion_image()
    view = plot_scan(scan)
    _curve_fit_message(view, "run")
    view.save_fit("kidney")
    smaller = scan.primary_data.isel(read_idx=slice(0, 4))
    other = SimpleNamespace(
        primary_data=smaller, is_spectroscopic=False, saved_fits=scan.saved_fits
    )
    assert "kidney_adc" not in plot_scan(other)._overlay_names()


def test_fit_tool_adds_a_line_at_the_spectral_cursor():
    data = _fitted_csi()
    view, _ = _view(data)
    view.state.spectral.selected_spectral_index = 40
    settings = _fit_message(view, "add_at_cursor")
    assert settings["peaks"][0]["freq_hz"] == pytest.approx(frequency_axis_hz(data)[40])
    settings = _fit_message(view, "edit", row=0, column="freq_hz", value="x")
    assert "freq_hz" in settings["message"]
    settings = _fit_message(view, "jump")
    assert "unknown action" in settings["message"]


def test_fit_tool_run_shows_maps_and_fit_curve_and_stores_the_fit():
    data = _fitted_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = ScanView(data, PlotDataModel(scan), _state())
    _fit_message(view, "option", name="n_jobs", value=1)
    settings = _fit_message(view, "detect")
    assert len(settings["peaks"]) == 3
    settings = _fit_message(view, "run")

    assert "Fit done." in settings["message"]
    assert view.state.overlay.active_variable.startswith("fit_")
    assert view.state.overlay.visible is True
    assert view._fit_curves
    assert scan.fit_results_review is scan.interactive_fit_results
    found = np.sort(scan.fit_results_review.fit_freqs.values[0, 0, 0, 0, 0])
    np.testing.assert_allclose(found, [-800.0, 300.0, 1200.0], atol=1.0)
    # The result table shows the selected voxel's lines.
    table = view._fit_table
    assert len(table["rows"]) == 3
    np.testing.assert_allclose(
        sorted(row["freq_hz"] for row in table["rows"]),
        [-800.0, 300.0, 1200.0],
        atol=1.0,
    )
    assert table["gauss_fwhm_hz"] is None
    # A drawn mask gets the mean fitted values of its voxels.
    assert view._mask_statistics == {}
    view.add_mask("roi")
    view.draw_polygon(_voxel_square(view, 0, 0), mask="roi")
    statistics = view._mask_statistics["rows"]
    assert [row["mask"] for row in statistics] == ["roi"] * 3
    assert all(row["n_voxels"] == 1 for row in statistics)
    np.testing.assert_allclose(
        sorted(row["freq_hz"][0] for row in statistics),
        [-800.0, 300.0, 1200.0],
        atol=1.0,
    )

    settings = _fit_message(view, "save", name="pyr_lac")
    assert "'pyr_lac' in scan.saved_fits" in settings["message"]
    saved = scan.saved_fits["pyr_lac"]
    assert saved.kind == "spectral"
    assert saved.results.identical(scan.fit_results_review)
    assert saved.params.identical(scan.fit_params_review)
    assert "pyr_lac_fit_amps" in view._overlay_names()


def test_fit_tool_couplings_and_phase_band_come_from_browser_messages():
    data = _fitted_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = ScanView(data, PlotDataModel(scan), _state())
    _fit_message(view, "option", name="n_jobs", value=1)
    names = [peak["name"] for peak in _fit_message(view, "detect")["peaks"]]
    settings = _fit_message(view, "edit", row=1, column="t2_leader", value=names[0])
    assert settings["peaks"][1]["t2_leader"] == names[0]
    _fit_message(view, "edit", row=0, column="phase_band", value=True)

    # The invented scan has no timing to derive the phase-band delay from.
    settings = _fit_message(view, "run")
    assert "Enter tau in ms" in settings["message"]
    assert not hasattr(scan, "fit_results_review")
    settings = _fit_message(view, "option", name="phase_band_tau_ms", value=-1)
    assert "must not be negative" in settings["message"]


def test_fit_tool_aligns_the_shown_spectra_and_undoes_it():
    data = _fitted_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = ScanView(data, PlotDataModel(scan), _state())
    _fit_message(view, "option", name="n_jobs", value=1)
    settings = _fit_message(view, "align")
    assert "run the fit before" in settings["message"]
    assert settings["fitted"] is False and settings["aligned"] is False

    _fit_message(view, "detect")
    assert _fit_message(view, "run")["fitted"] is True
    settings = _fit_message(view, "align")
    assert "Spectra aligned to" in settings["message"]
    assert settings["aligned"] is True
    assert "frequency_alignment_offset_hz" in view._data.coords
    assert scan.primary_data is data
    assert scan.data_review.identical(view._data)
    assert view._overlays["fit"] is scan.fit_results_review

    settings = _fit_message(view, "undo_align")
    assert settings["message"] == "Alignment undone." and settings["aligned"] is False
    assert view._data is data
    assert "frequency_alignment_offset_hz" not in scan.fit_results_review.coords


def test_magnitude_spectra_have_no_fid_view():
    # Scanner (2dseq) spectra are real magnitudes; their phase and FID are lost.
    spectra = np.abs(_csi()).astype(np.float32)
    spectra.attrs.update(signal_domain="spectral")
    state = _state()
    state.spectral.show_fid = True
    view, _ = _view(spectra, state)
    assert not view.state.spectral.show_fid
    with pytest.raises(ValueError, match="show_fid=True needs complex spectra"):
        view.set_processing(show_fid=True)
    assert not view.state.spectral.show_fid


def test_stored_fit_is_a_hidden_overlay_and_gives_lines_and_start_values():
    data = _fitted_csi()
    first = ScanView(
        data,
        PlotDataModel(SimpleNamespace(primary_data=data, is_spectroscopic=True)),
        _state(),
    )
    _fit_message(first, "option", name="n_jobs", value=1)
    _fit_message(first, "detect")
    assert _fit_message(first, "run")["earlier_fit"] == "last fit"

    scan = SimpleNamespace(
        primary_data=data,
        is_spectroscopic=True,
        fit_results_review=first._fit[1],
        fit_params_review=first._fit[0],
    )
    view = ScanView(data, PlotDataModel(scan), _state())
    settings = view._fit_tool
    assert settings["earlier_fit"] == "fit_results_review"
    assert "shown as overlay 'fit'" in settings["message"]
    assert view._overlays["fit"] is scan.fit_results_review
    assert view.state.overlay.visible is False
    assert view.state.overlay.active_variable.startswith("fit_")

    _fit_message(view, "option", name="n_jobs", value=1)
    settings = _fit_message(view, "lines_from_fit")
    assert len(settings["peaks"]) == 3
    assert all(row["freq_from_fit"] and row["t2_from_fit"] for row in settings["peaks"])
    # One search band per line: start frequency +/- half the search width.
    for row, band in zip(settings["peaks"], settings["search_bands"], strict=True):
        half = row["freq_range_hz"] / 2
        assert band["Hz"] == pytest.approx(
            [row["freq_hz"] - half, row["freq_hz"] + half]
        )
    settings = _fit_message(view, "run")
    assert "Start values from the earlier fit" in settings["message"]


def test_stored_fit_of_other_voxels_is_not_used():
    data = _fitted_csi()
    view, _ = _view(data)
    _fit_message(view, "option", name="n_jobs", value=1)
    _fit_message(view, "detect")
    _fit_message(view, "run")
    other = _csi(n_read=4, n_phase=2, n_rep=1, n_spec=128)
    scan = SimpleNamespace(
        primary_data=other, is_spectroscopic=True, fit_results=view._fit[1]
    )
    other_view = ScanView(other, PlotDataModel(scan), _state())
    assert other_view._fit_tool["earlier_fit"] == ""
    assert "Stored fit fit_results not used" in other_view._fit_tool["message"]
    assert "fit" not in other_view._overlays


def test_fit_review_buttons_refit_and_step_through_problem_voxels():
    data = _fitted_csi()
    scan = SimpleNamespace(primary_data=data, is_spectroscopic=True)
    view = ScanView(data, PlotDataModel(scan), _state())
    _fit_message(view, "option", name="n_jobs", value=1)
    _fit_message(view, "detect")
    settings = _fit_message(view, "refit_voxel")
    assert settings["message"] == "Refit voxel: fit all voxels first."

    _fit_message(view, "run")
    settings = _fit_message(view, "refit_voxel")
    assert settings["message"].startswith("Refitted voxel")
    assert scan.fit_results_review is scan.interactive_fit_results

    settings = _fit_message(view, "next_problem")
    assert settings["message"].startswith("Problem voxel 1/6")
    selected = view.state.spatial.slice_indices
    assert 0 <= selected["read_idx"] < 3 and 0 <= selected["phase_idx"] < 2

    # Invented data carry no acquisition-tail noise: the reason is shown.
    settings = _fit_message(view, "residual_line", low_hz=500.0, high_hz=900.0)
    assert "noise_sigma_quadrature" in settings["message"]


def test_fit_residual_curves_are_shown_and_hidden():
    data = _fitted_csi()
    view, _ = _view(data)
    _fit_message(view, "option", name="n_jobs", value=1)
    _fit_message(view, "detect")
    _fit_message(view, "run")
    view.show_fit_curves(fit=True, residual=True)
    curves = view._fit_curves["curves"]
    zero = [curve for curve in curves if curve.get("dash") == "dash"]
    assert len(zero) == 1 and len(zero[0]["x"]) == len(zero[0]["y"]) == 2
    with_residual = len(curves)

    view.show_fit_curves(residual=False)
    assert view.state.spectral.show_fit_residual is False
    assert len(view._fit_curves["curves"]) == with_residual - 2
    assert view._curve_axes["residual"] is False
    view.show_fit_curves(fit=False)
    assert view._fit_curves == {}


def test_measurement_parameters_are_sent_once_for_the_parameters_group():
    data = _fitted_csi()

    class Scan:
        primary_data = data
        is_spectroscopic = True
        PARAMETER_SCHEMA = {"acquisition": ("echo_time_ms",)}

        def get_parameters(self):
            return {"echo_time_ms": 1.25, "extra": [1, 2]}

    view = ScanView(data, PlotDataModel(Scan()), _state())
    assert view._parameters == [
        {
            "group": "acquisition",
            "name": "echo_time_ms",
            "value": "1.25",
            "source": "",
        },
        {"group": "other", "name": "extra", "value": "[1, 2]", "source": ""},
    ]
    assert _view(data)[0]._parameters == []


def test_a_new_view_starts_at_the_strongest_voxel_in_percent_of_the_map():
    data = _csi().copy()
    # A hot spot away from the corner of the scale ramp (invented values).
    data[{"read_idx": 3, "phase_idx": 1}] = data.isel(read_idx=3, phase_idx=1) * 1e3
    view = plot_scan(data)
    indices = view.state.spatial.slice_indices
    assert (indices["read_idx"], indices["phase_idx"]) == (3, 1)
    block = np.frombuffer(view._block, dtype=np.float32).reshape(view._view["shape"])
    strongest = np.abs(block).max(axis=2)
    row, column = view.selection["row"], view.selection["column"]
    assert strongest[row, column] == strongest.max()
    # Spectroscopic data start in percent of the shown map, not in the
    # Min/Max plane range that plot uses.
    spatial = view.state.spatial
    assert (spatial.range_mode, spatial.vmin, spatial.vmax) == ("Percentage", 0, 80)

    # A given state keeps its voxel and colour range.
    given = plot_scan(data, state=_state()).state.spatial
    assert given.range_mode == "Min/Max"
    assert (given.slice_indices["read_idx"], given.slice_indices["phase_idx"]) == (1, 2)


def test_overlay_layer_labels_drop_the_source_name_unless_ambiguous():
    from hypermri.visualization.scan_view.overlay import _layer_labels

    amps = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    overlays = {
        "fit": xr.Dataset({"fit_amps": amps, "fit_freqs": amps}),
        "other": xr.Dataset({"fit_amps": amps}),
        "map": amps,
    }
    names = ["fit_fit_amps", "fit_fit_freqs", "other_fit_amps", "map"]
    assert _layer_labels(overlays, names) == [
        ["fit_fit_amps", "fit_fit_amps"],
        ["fit_freqs", "fit_fit_freqs"],
        ["other_fit_amps", "other_fit_amps"],
        ["map", "map"],
    ]
    assert _layer_labels({"fit": overlays["fit"]}, names[:2]) == [
        ["fit_amps", "fit_fit_amps"],
        ["fit_freqs", "fit_fit_freqs"],
    ]
