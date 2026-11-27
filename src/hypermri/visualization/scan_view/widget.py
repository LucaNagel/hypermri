"""The browser viewer widget :class:`ScanView`.

The class holds the synchronised traits, the trait observers, the message
dispatch from the browser and the rendering of the shown plane. Its other
methods are grouped by topic in the sibling modules of this package and
joined here.
"""

from __future__ import annotations

from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.analysis.spatial_filtering import (
    SpatialFilterOptions,
)
from hypermri.utils.mri.fourier import fid_is_recoverable
from hypermri.visualization.curve_fit_tool import (
    CurveFitState,
    CurveModel,
)
from hypermri.visualization.data_sources import DataSourceCatalog
from hypermri.visualization.fit_tool import (
    FitToolState,
)
from hypermri.visualization.ideal_tool import (
    IdealToolState,
)
from hypermri.visualization.layout import browser_layout, browser_script
from hypermri.visualization.mask_editing import (
    MaskedMeanCurve,
)
from hypermri.visualization.render_data import (
    PlotDataModel,
)
from hypermri.visualization.scan_parameters import parameter_rows
from hypermri.visualization.signal_decay_tool import (
    SignalDecayState,
)
from hypermri.visualization.spatial_overlay import (
    SpatialOverlayLayer,
)
from hypermri.visualization.state import (
    PlotState,
    _primary_range_was_supplied,
)
from hypermri.visualization.volume_outline import (
    OutlineSource,
)

_import_optional("anywidget")
_import_optional("cmap")

import anywidget  # noqa: E402
from traitlets import (  # noqa: E402
    Bool,
    Bytes,
    Dict,
    Float,
    Int,
    List,
    TraitError,
    Unicode,
    observe,
    validate,
)

from hypermri.visualization.scan_view.data_choice import (  # noqa: E402
    _DataChoice,  # noqa: E402
    _has_spectrum,
)
from hypermri.visualization.scan_view.display import _Display  # noqa: E402
from hypermri.visualization.scan_view.fit_results import (  # noqa: E402
    _FitResults,  # noqa: E402
)
from hypermri.visualization.scan_view.linking import _Linking  # noqa: E402
from hypermri.visualization.scan_view.masks import _Masks  # noqa: E402
from hypermri.visualization.scan_view.options import (  # noqa: E402
    _DISPLAY_SWITCHES,
    _SPATIAL_FILTER_WIDTHS,
    SCAN_SETTINGS_CONTEXT,
    SPATIAL_DIMS,
    SPEC_PART_OPTIONS,
    SPEC_UNIT_OPTIONS,
    SPECTRAL_DIM,
    _checked_curve_axes,
)
from hypermri.visualization.scan_view.overlay import _Overlay  # noqa: E402
from hypermri.visualization.scan_view.rendering import _Rendering  # noqa: E402
from hypermri.visualization.scan_view.secondary_slices import (  # noqa: E402
    _SecondarySlices,
)
from hypermri.visualization.scan_view.time_course import _TimeCourse  # noqa: E402
from hypermri.visualization.scan_view.tools import (  # noqa: E402
    _Tools,  # noqa: E402
)

_HERE = Path(__file__).parent.parent


def _plotly_js_version() -> str:
    """Return the version of the Plotly.js bundled with the plotly package."""
    from plotly.offline import get_plotlyjs_version

    return get_plotlyjs_version()


def _plotly_js_source() -> bytes:
    """Return the Plotly.js bundle shipped with the installed plotly package."""
    return (
        resource_files("plotly").joinpath("package_data", "plotly.min.js").read_bytes()
    )


class ScanView(
    _DataChoice,
    _Rendering,
    _Tools,
    _FitResults,
    _Linking,
    _Display,
    _Overlay,
    _SecondarySlices,
    _Masks,
    _TimeCourse,
    anywidget.AnyWidget,
):
    """Notebook widget showing a CSI map and the spectrum of the selected voxel.

    Create it with :func:`plot_scan`. The widget keeps the display state and
    re-runs the shared data preparation only when a processing setting
    changes. Voxel and spectral selections made in the browser are copied
    back into :attr:`state` without recomputation.

    Masks drawn in the browser call the same methods as Python code
    (:meth:`add_mask`, :meth:`draw_polygon`, :meth:`undo_mask_edit`, ...).
    When the source is a scan, every mask change is written to
    ``scan.masks``, and masks already stored there are shown at start.
    :meth:`show_roi_mean` replaces the voxel spectrum by means inside a
    mask.

    A fit result Dataset among the overlays (or ``scan.fit_results``) adds
    the fitted spectrum of the selected voxel to the curve panel, computed
    as in ``plot`` by the fit's reconstructor (``fit_results.hp``) and drawn
    over the measured spectrum (``state.spectral.show_fit`` and
    ``fit_mode``).

    Overlay layers (parameter maps without ``acquisition_idx``) are drawn
    above the map. Layer, metabolite, visibility, colormap and colour range
    are read from ``state.overlay`` and changed in the browser or with
    :meth:`set_overlay`; the opacity with :meth:`set_overlay_opacity`. The
    overlay slice follows the shown map unless one is chosen with
    :meth:`set_overlay_slice`; an overlay given as a scan can be shown from
    another reconstruction with :meth:`set_overlay_source`. Likewise the
    anatomy slice under the map follows the map unless one is chosen with
    :meth:`set_anatomy_slice`.

    The colour range of the map is read from ``state.spatial`` and changed in
    the browser or with :meth:`set_map_range` without recomputation.
    :meth:`save_settings` stores the whole display state on the scan.

    Parameters
    ----------
    data : xarray.DataArray
        Samples with at least two of ``read_idx``, ``phase_idx`` and
        ``slice_idx``, with or without ``acquisition_idx``. Not modified.
        The curve runs along ``state.spectral.secondary_dim``. When it is
        None, spectroscopic data show the spectrum along ``acquisition_idx``
        and images show no curve panel (image only).
    model : PlotDataModel
        Data preparation owner holding the acquisition context.
    state : PlotState
        Display state, completed with :func:`apply_display_defaults` before
        construction. It is mutated by later interactions.
    overlays : dict of str to Any, optional
        Overlays registered by
        :func:`~hypermri.visualization.spatial_overlay.normalize_overlays`,
        already in the coordinate mode of ``state``. None or empty shows no
        overlay. Not modified.
    overlay_sources : dict of str to DataSourceCatalog, optional
        Data sources of the overlays given as scans, by registered name
        (:func:`~hypermri.visualization.spatial_overlay.overlay_source_catalogs`).
        None or empty offers no source choice.
    settings_context : str, optional
        Key under which :meth:`save_settings` stores the display state on the
        scan; see :func:`plot_scan`.
    outline : OutlineSource, optional
        Anatomy and excited volume of a single-voxel or slice spectroscopy
        scan (:func:`~hypermri.visualization.volume_outline.outline_source`).
        The image panel then shows the anatomy slice with the outline of the
        volume; the browser slider changes ``outline.state`` in place. None
        hides the image panel of a single voxel.
    variables : dict of str to xarray.DataArray, optional
        Map variables of a fit result by name, one of which is ``data``.
        The browser then offers a variable menu; :meth:`set_variable`
        switches between them. None or empty offers no choice.
    metabolite_split : tuple of two xarray.DataArray, optional
        The data as acquired and with the interleaved metabolites split into
        ``metabolite_idx``
        (:func:`~hypermri.visualization.metabolite_split.split_interleaved_metabolites`),
        one of which is ``data``. The browser then offers a "split
        metabolites" switch; :meth:`set_split_metabolites` changes it. None
        offers no switch.
    image_only : bool, optional
        Show only the image panel, without the curve and time panels, for
        example for a second view of the same scan over another anatomy
        whose curve position follows the first view (:meth:`link_frames`).
        The selected voxel and the curve dimension are kept as usual.

    Attributes
    ----------
    selection : dict
        Current selection, synchronised with the browser: ``row`` and
        ``column`` of the displayed map (zero-based, top-left origin) and
        ``curve_index`` into the displayed curve axis (the spectral axis
        unless another curve dimension was chosen).
    tool : str
        Tool whose inputs the browser shows below the other
        controls, synchronised with the browser's "tool" menu: ``""`` for
        none (the default) or a name of
        :data:`~hypermri.visualization.layout.TOOLS`, for example
        ``"masks"`` or ``"fit"``. Setting another name raises
        ``traitlets.TraitError``. Only the shown inputs change; masks stay
        drawn on the map. The browser offers ``"fit"`` only for spectra
        (see :mod:`hypermri.visualization.fit_tool`).
    name : str
        Name shown in bold above the plots, synchronised with the browser,
        for example ``"E32 · CSI kidney"``;
        :func:`~hypermri.visualization.study_view.plot_study` sets the
        scan's entry of the scan list. The empty string (the default) shows
        no name. Changing it changes only the text.
    plot_height : int
        Height of the image and curve plots in pixels, synchronised with
        the browser; default 400. Values below 160 raise
        ``traitlets.TraitError``. Changing it resizes the plots without
        new data from Python.
    show_timings : bool
        Show in the status line below the plots how long the browser took
        to redraw (ms) and how long Python took for the last data block
        (ms, size in MB); synchronised with the browser. Default False:
        the status line shows only the hints for the mouse and keys.
    state : PlotState
        Display state; ``state.spatial.slice_indices`` holds the source index
        of the selected voxel for every dimension.

    Raises
    ------
    ValueError
        If the data have no curve block to show, i.e. no curves along
        ``state.spectral.secondary_dim`` (default ``acquisition_idx``) in a
        two-dimensional spatial plane, if
        masks stored on the scan do not match its spatial grid, or if the
        overlay settings in ``state.overlay`` cannot be shown (see
        :meth:`set_overlay`).
    """

    _esm = browser_script()
    _css = _HERE / "browser" / "viewer.css"

    _block = Bytes(b"").tag(sync=True)
    # The map while the display processing reduces the curve dimension
    # (rows x columns, float32); empty when the map is a point of the curves.
    _map_plane = Bytes(b"").tag(sync=True)
    _anatomy = Bytes(b"").tag(sync=True)
    _anatomy_opacity = Float(0.6).tag(sync=True)
    # Anatomy colour limits in percent of the shown slice's value range;
    # empty when the state holds limits in another range mode.
    _anatomy_range = List().tag(sync=True)
    _overlay = Bytes(b"").tag(sync=True)
    _overlay_opacity = Float(1.0).tag(sync=True)
    _map_range = Dict().tag(sync=True)
    _display = Dict().tag(sync=True)
    _curve_axes = Dict().tag(sync=True)
    _view = Dict().tag(sync=True)
    _controls = Dict().tag(sync=True)
    selection = Dict().tag(sync=True)
    _mask_planes = List().tag(sync=True)
    _roi_mean = Dict().tag(sync=True)
    _fit_curves = Dict().tag(sync=True)
    # Fitted values of the selected voxel from the fit tool (voxel_fit_table).
    _fit_table = Dict().tag(sync=True)
    # Mean and spread of the fitted values inside each mask (mask_fit_table).
    _mask_statistics = Dict().tag(sync=True)
    _other_metabolite = Dict().tag(sync=True)
    _time_block = Bytes(b"").tag(sync=True)
    _time_course = Dict().tag(sync=True)
    _layout = Dict().tag(sync=True)
    _slice_lines = List().tag(sync=True)
    _pointer_link = Unicode("").tag(sync=True)
    _pointer_plane = Dict().tag(sync=True)
    _mask_link = Unicode("").tag(sync=True)
    # False: the masks group of this linked view shows only the chosen mask
    # and its volume, because another view next to it holds the inputs.
    _mask_tool_shown = Bool(True).tag(sync=True)
    # Peak table and settings of the "fit" tool (fit_tool_settings), with
    # "offered" False when the shown data cannot be fitted, "fitted" True
    # after a fit and "aligned" True while aligned spectra are shown.
    _fit_tool = Dict().tag(sync=True)
    # Model and settings of the "curve_fit" tool (curve_fit_settings), with
    # "offered" False when no model belongs to the curve axis and "fitted"
    # True when a result of the shown model exists.
    _curve_fit_tool = Dict().tag(sync=True)
    # Settings of the "ideal" tool (ideal_settings), with "offered" False
    # unless the curve axis is echo_idx and "fitted" True after a run.
    _ideal_tool = Dict().tag(sync=True)
    # Choices and decay curves of the "signal_decay" tool
    # (signal_decay_settings); empty unless plot_signal_decay opened the view.
    _signal_decay_tool = Dict().tag(sync=True)
    # Measurement parameters of the scan (parameter_rows), set once; the
    # browser shows them when "show parameters" is checked.
    _parameters = List().tag(sync=True)
    tool = Unicode("").tag(sync=True)
    name = Unicode("").tag(sync=True)
    plot_height = Int(400).tag(sync=True)
    show_timings = Bool(False).tag(sync=True)
    # Version of the Plotly.js bundled with the plotly package; the browser
    # fetches this version from the web when no kernel sends its copy.
    _plotly_version = Unicode(_plotly_js_version()).tag(sync=True)

    def __init__(
        self,
        data: xr.DataArray,
        model: PlotDataModel,
        state: PlotState,
        overlays: dict[str, Any] | None = None,
        overlay_sources: dict[str, DataSourceCatalog] | None = None,
        settings_context: str = SCAN_SETTINGS_CONTEXT,
        outline: OutlineSource | None = None,
        variables: dict[str, xr.DataArray] | None = None,
        metabolite_split: tuple[xr.DataArray, xr.DataArray] | None = None,
        image_only: bool = False,
    ) -> None:
        super().__init__()
        # Without the curve and time panels the view shows only its image;
        # the curve position then comes from link_frames.
        self._layout = (
            browser_layout(panels=("image",)) if image_only else browser_layout()
        )
        self._data = data
        self._variables = dict(variables or {})
        self._variable = next(
            (name for name, values in self._variables.items() if values is data),
            None,
        )
        if self._variables and self._variable is None:
            raise ValueError(
                "ScanView data must be one of the given variables "
                f"{list(self._variables)}."
            )
        self._metabolite_split = metabolite_split
        if metabolite_split is not None and not any(
            values is data for values in metabolite_split
        ):
            raise ValueError(
                "ScanView data must be the acquired or the split data given as "
                "metabolite_split."
            )
        self._model = model
        self.state = state
        self._overlays = dict(overlays or {})
        self._overlay_sources = dict(overlay_sources or {})
        self._overlay_plane: dict[str, int] = {}
        self._anatomy_plane: dict[str, int] = {}
        self._settings_context = settings_context
        self._outline = outline
        self._cmaps: list[tuple[str, str]] | None = None
        self._plane_range = (0.0, 1.0)
        self._overlay_cache: tuple[tuple[Any, ...], SpatialOverlayLayer] | None = None
        self._block_indices: dict[str, np.ndarray] = {}
        self._plane_curves = np.zeros((0, 0, 0), dtype=np.float32)
        self._spatial_dims: list[str] = []
        self._spectroscopic = _has_spectrum(data)
        if not fid_is_recoverable(data):
            # Real (magnitude) spectra have lost their phase, so no FID.
            state.spectral.show_fid = False
        # None: image only, without a curve panel.
        self._curve_dim = self._shown_curve_dim()
        self._curve_x = np.empty(0)
        self._applying_python_selection = False
        self._scan = (
            None if isinstance(model.data_obj, xr.DataArray) else model.data_obj
        )
        self._mask_store: dict[str, xr.DataArray] = {}
        self._mask_colors: dict[str, str] = {}
        self._mask_alphas: dict[str, float] = {}
        self._mask_outlines: dict[str, list[dict[str, Any]]] = {}
        # (mask, plane indexer, previous plane coverage, stored outline or None)
        self._mask_undo: list[
            tuple[str, tuple[Any, ...], np.ndarray, dict[str, Any] | None]
        ] = []
        self._mask_plane: dict[str, Any] = {}
        self._roi_mask: str | None = None
        self._roi_curve: MaskedMeanCurve | None = None
        self._time_dim: str | None = None
        self._time_curves: np.ndarray | None = None
        self._time_mask: str | None = None
        self._fit_voxel: tuple[int, int] | None = None
        self._other_voxel: tuple[int, int] | None = None
        # Views whose shown slice this view draws, with the line colour, and
        # views that draw this view's slice (see show_slices_of).
        self._slice_sources: list[tuple[ScanView, str]] = []
        self._slice_viewers: list[ScanView] = []
        # (displayed data, the same with scanner coordinates x, y, z)
        self._lps_data: tuple[xr.DataArray, xr.DataArray] | None = None
        # Views whose scans share the masks of this view (see link_masks).
        self._mask_partners: list[ScanView] = []
        # Views of link_pointer: a voxel clicked here is selected there too.
        self._pointer_partners: list[ScanView] = []
        # Views of link_frames: they show the same repetition (and the
        # same position along every other non-spatial dimension).
        self._frame_partners: list[ScanView] = []
        # Bands of the slices of other views (show_slices_of) are drawn;
        # off until switched on, so linked views start uncluttered.
        self._show_slice_lines = False
        # Masks and their drawn outlines are drawn over the map
        # (set_display(show_masks=..., show_mask_outlines=...)).
        self._show_masks = True
        self._show_mask_outlines = True
        # y axis of the curve panel over the whole plane ("plane" in
        # set_curve_scale); a choice of this view, not stored in the state.
        self._plane_curve_scale = False
        self._fit_tool_state = FitToolState()
        self._curve_fit_state = CurveFitState()
        # (model, result) of the last curve-fit run.
        self._curve_fit: tuple[CurveModel, xr.Dataset] | None = None
        self._ideal_state = IdealToolState()
        # Result of the last IDEAL run (fit_ideal), or None.
        self._ideal: xr.Dataset | None = None
        # (prepared CSI raw data, choices) of the signal-decay tool, set by
        # plot_signal_decay, and why the last filter failed (or None).
        self._signal_decay: tuple[Any, SignalDecayState] | None = None
        self._signal_decay_error: str | None = None
        # Spatial k-space filter of the shown data (set_kspace), the last
        # (source data, filter, k-space shown, shown data) and why the last
        # browser change of these failed.
        self._spatial_filter: SpatialFilterOptions | None = None
        # Widths the browser shows for the filter kinds not chosen; they
        # start as in the signal-decay tool.
        self._spatial_filter_widths = {
            name: float(getattr(SignalDecayState, name))
            for name in _SPATIAL_FILTER_WIDTHS
        }
        self._shown_data_cache: (
            tuple[xr.DataArray, SpatialFilterOptions | None, bool, xr.DataArray] | None
        ) = None
        self._kspace_error = ""
        # (fit_params, fit_results) of the last fit tool run or alignment.
        self._fit: tuple[xr.Dataset, xr.Dataset] | None = None
        # (data, fit_params, fit_results) before the first alignment, so that
        # "Undo align" shows the acquired spectra and their fit again.
        self._before_alignment: tuple[xr.DataArray, xr.Dataset, xr.Dataset] | None = (
            None
        )
        # (name, fit_results) whose frequencies and T2* seed the next fit:
        # the fit stored on the scan, then the last fit of this view.
        self._earlier_fit: tuple[str, xr.Dataset] | None = None
        self._load_stored_masks()
        self._load_stored_fit()
        self._load_saved_fits()
        self._parameters = parameter_rows(self._scan)
        self._check_overlay_settings()
        self._check_map_range_mode(state.spatial.range_mode)
        self.on_msg(self._handle_custom_message)
        self._render(
            reset_map_range=state.spatial.range_mode != "Percentage"
            and not _primary_range_was_supplied(state)
        )
        self._controls = self._current_controls()
        self._curve_axes = self._curve_axes_settings()
        self._send_fit_tool()

    def _selection_from_state(self, *, n_curve: int) -> dict[str, int]:
        """Translate the selected source voxel and curve position for the browser."""
        indices = self.state.spatial.slice_indices
        position = {}
        for dim in self._spatial_dims:
            matches = np.flatnonzero(self._block_indices[dim] == indices.get(dim, 0))
            position[dim] = int(matches[0]) if matches.size else 0
        if self._curve_dim is None:
            curve_index = 0
        elif self._curve_dim == SPECTRAL_DIM:
            curve_index = self.state.spectral.selected_spectral_index or 0
        else:
            curve_index = indices.get(self._curve_dim, 0)
        return {
            "row": position[self._spatial_dims[0]],
            "column": position[self._spatial_dims[1]],
            "curve_index": int(min(max(curve_index, 0), n_curve - 1)),
        }

    def _select_strongest_voxel(self) -> None:
        """Select the voxel of the shown plane with the strongest signal.

        The signal of a voxel is the largest magnitude along its shown curve
        (spectrum, time course or the image value itself). The curve position
        stays; a plane without finite values keeps the current voxel.
        """
        strength = np.abs(self._plane_curves)
        finite = np.isfinite(strength)
        if not finite.any():
            return
        peak = np.where(finite, strength, -np.inf).max(axis=2)
        row, column = np.unravel_index(int(np.argmax(peak)), peak.shape)
        self.selection = {
            **self.selection,
            "row": int(row),
            "column": int(column),
        }

    @validate("plot_height")
    def _check_plot_height(self, proposal: dict[str, Any]) -> int:
        """Accept plot heights of at least 160 pixels."""
        height = proposal["value"]
        if height < 160:
            raise TraitError(
                f"ScanView.plot_height: {height} px is too small; give 160 or more."
            )
        return height

    @validate("tool")
    def _check_tool(self, proposal: dict[str, Any]) -> str:
        """Accept no tool or a tool listed in the browser layout."""
        name = proposal["value"]
        tools = self._layout.get("tools", [])
        if name and name not in tools:
            raise TraitError(
                f"ScanView.tool: unknown tool {name!r}; choose '' or one of {tools}."
            )
        return name

    @observe("selection")
    def _on_selection(self, change: dict[str, Any]) -> None:
        """Copy a browser selection into the display state without recomputing."""
        if self._applying_python_selection:
            return
        selection = change["new"]
        row_dim, column_dim = self._spatial_dims
        indices = self.state.spatial.slice_indices
        indices[row_dim] = int(self._block_indices[row_dim][selection["row"]])
        indices[column_dim] = int(self._block_indices[column_dim][selection["column"]])
        curve_index = int(selection["curve_index"])
        self._update_fit_curves()
        self._update_other_metabolite()
        old = change["old"] or {}
        if (old.get("row"), old.get("column")) != (
            selection["row"],
            selection["column"],
        ):
            self._select_in_pointer_partners()
        if self._curve_dim is None:
            return
        if self._curve_dim == SPECTRAL_DIM:
            moved = curve_index != self.state.spectral.selected_spectral_index
            self._select_spectral_index(curve_index, self._curve_x[curve_index])
            if moved and self.state.spectral.time_course_range is None:
                # The time course follows the selected spectral point.
                self._send_time_course()
        else:
            indices[self._curve_dim] = curve_index
        self._show_frame_in_partners()

    def _select_spectral_index(self, index: int, value: float | None) -> None:
        """Store the spectral position of the map in the display state.

        ``value`` is the position in the displayed spectral unit; None leaves
        the position to the index alone (it is then taken from the index on
        the next block, as in ``plot``).
        """
        self.state.spectral.selected_spectral_index = index
        self.state.spectral.selected_spectral_value = (
            None if value is None else float(value)
        )

    @observe("_controls")
    def _on_controls(self, change: dict[str, Any]) -> None:
        """Apply processing settings from the browser and send a new block."""
        old, new = change["old"], change["new"]
        if not old or new == self._current_controls():
            return
        if self._outline is not None and "anatomy_slice" in new:
            outline = self._outline.state.spatial
            n_slices = self._outline.data.sizes.get(outline.slicing_dim, 1)
            outline.slice_indices[outline.slicing_dim] = int(
                min(max(int(new["anatomy_slice"]), 0), n_slices - 1)
            )
        sent_indices = new.get("indices", {})
        variable = new.get("variable")
        if variable is not None and variable != self._variable:
            # The other settings sent with a new variable describe the
            # previous one; its slider indices may name other dimensions.
            self._change_variable(variable)
            self._controls = self._current_controls()
            return
        split = new.get("split_metabolites")
        if split is not None and split != old.get("split_metabolites"):
            # As for a new variable, the slider indices describe the old data.
            self.set_split_metabolites(bool(split))
            return
        plane = new.get("view_plane")
        if plane is not None and plane != old.get("view_plane"):
            # The slider indices sent with a new plane describe the old one.
            self.set_view_plane(plane)
            return
        if "rotation" in new and int(new["rotation"]) != self.state.spatial.rotation:
            self.set_rotation(int(new["rotation"]) % 360)
            return
        if "other_metabolite" in new:
            self.state.spectral.show_secondary_metabolite_curve = bool(
                new["other_metabolite"]
            )
        indices = self.state.spatial.slice_indices
        plane_moved = False
        for dim, index in sent_indices.items():
            if dim not in self._data.dims:
                raise ValueError(f"ScanView control names unknown dimension {dim!r}.")
            value = int(min(max(int(index), 0), self._data.sizes[dim] - 1))
            if dim == SPECTRAL_DIM:
                if value != self._other_index(dim):
                    self._select_spectral_index(value, None)
                continue
            plane_moved |= dim in SPATIAL_DIMS and value != indices.get(dim)
            indices[dim] = value
        if plane_moved and self._model._uses_xyz_coordinates(self.state):
            # As in plot: in scanner coordinates the anatomy and overlay
            # planes follow the shown map again once the map moves.
            self._follow_map_with_layers()
        spectral = self.state.spectral
        shown_values = (
            spectral.spec_part,
            float(spectral.line_broadening_hz),
            self._processing_settings(),
        )
        if new.get("processing") is not None:
            self._apply_processing(new["processing"], "ScanView controls")
        part = new.get("spec_part", self.state.spectral.spec_part)
        unit = new.get("spec_unit", self.state.spectral.spec_unit)
        if part not in SPEC_PART_OPTIONS or unit not in SPEC_UNIT_OPTIONS:
            raise ValueError(
                f"ScanView controls: spec_part must be one of {SPEC_PART_OPTIONS} "
                f"and spec_unit one of {SPEC_UNIT_OPTIONS}."
            )
        self.state.spectral.spec_part = part
        self.state.spectral.spec_unit = unit
        self.state.spectral.line_broadening_hz = float(
            new.get("line_broadening_hz", self.state.spectral.line_broadening_hz)
        )
        if self._overlays and new.get("overlay") is not None:
            overlay = new["overlay"]
            source = overlay.get("data_source")
            if source is not None and source != self.state.overlay.data_source:
                # A new source chooses its own layer, range and slice; the
                # other settings sent with it describe the previous source.
                self._change_overlay_source(source)
            else:
                self._apply_overlay_settings(
                    variable=overlay.get("variable"),
                    metabolite=overlay.get("metabolite"),
                    visible=overlay.get("visible"),
                    range_mode=overlay.get("range_mode"),
                    vmin=overlay.get("vmin"),
                    vmax=overlay.get("vmax"),
                    cmap=overlay.get("cmap"),
                )
                # Only a slice the user changed counts; the browser resends
                # its previous choice with every other control.
                sent_before = (old.get("overlay") or {}).get("slice")
                if "slice" in overlay and overlay["slice"] != sent_before:
                    self._change_overlay_slice(overlay["slice"])
        anatomy = new.get("anatomy")
        if anatomy is not None and self._outline is None:
            # As for the overlay slice, only a changed choice counts.
            sent_before = (old.get("anatomy") or {}).get("slice")
            if "slice" in anatomy and anatomy["slice"] != sent_before:
                self._change_anatomy_slice(anatomy["slice"])
        values_changed = shown_values != (
            spectral.spec_part,
            float(spectral.line_broadening_hz),
            self._processing_settings(),
        )
        # Settings Python completed, such as limits reset for a new overlay
        # layer, go back to the browser before the block that uses them.
        effective = self._current_controls()
        if new != effective:
            self._controls = effective
        self._render(
            reset_map_range=values_changed
            and self.state.spatial.range_mode == "Min/Max"
        )

    def to_html(
        self,
        path: str | Path | None = None,
        *,
        plotly_js: Literal["inline", "cdn"] = "inline",
    ) -> str:
        """Save the shown plane as a standalone HTML page that needs no Python.

        The page contains the viewer's browser code, Plotly.js and the
        values this view has sent to the browser: the block of curves of the
        shown plane, the map, anatomy and overlay layers, the masks, the
        selection, the colour range and the name. In any browser, without a
        kernel, a click or the arrow keys choose a voxel, hover shows values,
        and the colour range, curve axes and display switches work as in the
        notebook. Every slice of an anatomy drawn under the map is saved
        too, so the anatomy slice slider and "follow map" work; the view is
        rendered once per anatomy slice for this and then shows what it
        showed before. Other controls that ask Python for a new block (map
        slices, processing, overlays, masks) are shown switched off, and the
        tools are left out.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            File to write, replaced if it exists. ``None`` (the default)
            writes nothing and only returns the page.
        plotly_js : {"inline", "cdn"}, optional
            ``"inline"`` (the default) embeds the Plotly.js of the installed
            plotly package, about 4.6 MB, so the page also opens offline.
            ``"cdn"`` loads the same Plotly.js version from jsDelivr, which
            keeps the file small but needs the web.

        Returns
        -------
        str
            The HTML document.

        Raises
        ------
        ValueError
            If *plotly_js* is not ``"inline"`` or ``"cdn"``.

        Examples
        --------
        >>> view = scan.plot()  # doctest: +SKIP
        >>> view.to_html("csi_kidney.html")  # doctest: +SKIP
        """
        from hypermri.visualization.scan_view.snapshot import snapshot_html

        page = snapshot_html(self, plotly_js=plotly_js)
        if path is not None:
            Path(path).write_text(page, encoding="utf-8")
        return page

    def _handle_custom_message(
        self, widget: Any, content: dict[str, Any], buffers: list[Any]
    ) -> None:
        """Answer the Plotly.js request and apply browser actions.

        Mask edits and saving the settings are applied with the public
        methods. A mask edit that fails is reported back to the browser as a
        ``mask_error`` message, and every save request is answered with a
        ``settings_saved`` message, instead of raising inside the comm
        handler. A fit tool action that fails shows its error in the tool.
        """
        kind = content.get("type")
        if kind == "request_plotly":
            self.send({"type": "plotly"}, buffers=[_plotly_js_source()])
        elif kind == "save_settings":
            try:
                self.save_settings()
            except ValueError as error:
                self.send(
                    {"type": "settings_saved", "ok": False, "message": str(error)}
                )
            else:
                self.send({"type": "settings_saved", "ok": True, "message": ""})
        elif kind in ("mask", "roi_mean"):
            try:
                if kind == "mask":
                    self._apply_browser_mask_action(content)
                else:
                    self.show_roi_mean(content.get("mask") or None)
            except ValueError as error:
                self.send({"type": "mask_error", "message": str(error)})
        elif kind == "time_course":
            try:
                self._apply_browser_time_course(content)
            except ValueError as error:
                self.send({"type": "time_course_error", "message": str(error)})
        elif kind == "fit":
            try:
                self._apply_browser_fit_action(content)
            except ValueError as error:
                self._fit_tool_state.message = str(error)
            self._send_fit_tool()
        elif kind == "curve_fit":
            try:
                self._apply_browser_curve_fit_action(content)
            except ValueError as error:
                self._curve_fit_state.message = str(error)
            self._send_curve_fit_tool()
        elif kind == "ideal":
            try:
                self._apply_browser_ideal_action(content)
            except (ValueError, np.linalg.LinAlgError) as error:
                self._ideal_state.message = str(error)
            self._send_ideal_tool()
        elif kind == "kspace":
            self._apply_browser_kspace(content)
        elif kind == "signal_decay" and self._signal_decay is not None:
            state = self._signal_decay[1]
            try:
                self._apply_browser_signal_decay_action(content)
            except ValueError as error:
                state.message = str(error)
            self._send_signal_decay_tool()

    @observe("tool")
    def _on_tool(self, change: dict[str, Any]) -> None:
        """Open or close the mask tool in the linked views too."""
        if "masks" not in (change["old"], change["new"]):
            return
        for partner in self._mask_partners:
            if partner.tool != change["new"]:
                partner.tool = change["new"]

    @observe("_anatomy_opacity")
    def _on_anatomy_opacity(self, change: dict[str, Any]) -> None:
        """Keep the opacity set in the browser for the next recomputation."""
        self.state.anatomy.opacity = float(change["new"])

    @observe("_anatomy_range")
    def _on_anatomy_range(self, change: dict[str, Any]) -> None:
        """Keep the anatomy colour range set in the browser in the state."""
        if change["new"]:
            self._store_anatomy_range(*change["new"])

    @observe("_map_range")
    def _on_map_range(self, change: dict[str, Any]) -> None:
        """Store the map range set in Python or the browser in the state."""
        settings = change["new"]
        self._check_map_range(settings)
        spatial = self.state.spatial
        spatial.range_mode = settings["range_mode"]
        spatial.vmin = float(settings["vmin"])
        spatial.vmax = float(settings["vmax"])

    @observe("_display")
    def _on_display(self, change: dict[str, Any]) -> None:
        """Store the display switches set in Python or the browser."""
        settings = change["new"]
        self._check_display(settings)
        spatial = self.state.spatial
        spatial.interpolation = settings["smoothing"] != "off"
        if spatial.interpolation:
            spatial.zsmooth_mode = settings["smoothing"]
        # Python sends the scalebar length and the colour scales with the view.
        redraw = (
            bool(settings["show_scalebar"]) != spatial.show_scalebar
            or settings["map_cmap"] != spatial.cmap
            or settings["anatomy_cmap"] != self.state.anatomy.cmap
        )
        for name, (section, field) in _DISPLAY_SWITCHES.items():
            setattr(getattr(self.state, section), field, bool(settings[name]))
        spatial.cmap = settings["map_cmap"]
        self.state.anatomy.cmap = settings["anatomy_cmap"]
        self._show_slice_lines = bool(settings.get("show_slice_lines", False))
        self._show_masks = bool(settings.get("show_masks", True))
        self._show_mask_outlines = bool(settings.get("show_mask_outlines", True))
        if redraw:
            self._render()

    @observe("_curve_axes")
    def _on_curve_axes(self, change: dict[str, Any]) -> None:
        """Check and store the curve-axis settings set in Python or the browser."""
        settings = change["new"]
        window = _checked_curve_axes(settings, "ScanView curve axes")
        spectral = self.state.spectral
        spectral.couple_ylims = settings["scale"] == "map"
        self._plane_curve_scale = settings["scale"] == "plane"
        spectral.display_range = window
        spectral.display_range_unit = settings["window_unit"]
        fit = bool(settings.get("fit", spectral.show_fit))
        residual = bool(settings.get("residual", spectral.show_fit_residual))
        if (fit, residual) != (spectral.show_fit, spectral.show_fit_residual):
            spectral.show_fit = fit
            spectral.show_fit_residual = residual
            self._update_fit_curves(force=True)

    @observe("_overlay_opacity")
    def _on_overlay_opacity(self, change: dict[str, Any]) -> None:
        """Keep the opacity set in the browser for the next recomputation."""
        self.state.overlay.alpha = float(change["new"])
