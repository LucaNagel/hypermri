"""Rendering of the shown plane: shown dimensions, prepared payload and the layers sent to the browser.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_Rendering`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.utils.mri.fourier import fid_is_recoverable
from hypermri.visualization.geometry import (
    display_voxel_centres,
    get_spatial_dim_label,
)
from hypermri.visualization.mask_editing import (
    mask_dims,
)
from hypermri.visualization.view_dimensions import (
    view_plane_choices,
)

_import_optional("anywidget")
_import_optional("cmap")

from cmap import Colormap  # noqa: E402

from hypermri.visualization.scan_view.data_choice import (  # noqa: E402
    _average_range,
    _chosen_curve_dim,
    _has_spectrum,
)
from hypermri.visualization.scan_view.layers import (  # noqa: E402
    _anatomy_layer,
    _cell_edges,
    _finite_range,
    _outline_layer,
    _overlay_layer_view,
    _scalebar_view,
    _widened_range,
)
from hypermri.visualization.scan_view.options import (  # noqa: E402
    CHANNEL_COMBINE_METHODS,
    MAP_RANGE_MODES,
    MAP_SMOOTHING_OPTIONS,
    REPETITION_MODES,
    SPEC_PART_OPTIONS,
    SPEC_UNIT_OPTIONS,
    SPECTRAL_DIM,
)


class _Rendering:
    """Rendering of the shown plane: shown dimensions, prepared payload and the layers sent to the browser of :class:`ScanView`."""

    def _current_controls(self) -> dict[str, Any]:
        """Return the processing settings shown as browser controls."""
        controls = {
            "indices": {dim: self._other_index(dim) for dim in self._other_dims()},
            "spec_part": self.state.spectral.spec_part,
            "spec_unit": self.state.spectral.spec_unit,
            "line_broadening_hz": float(self.state.spectral.line_broadening_hz),
            "overlay": self._overlay_controls(),
            "rotation": int(self.state.spatial.rotation),
            "processing": self._processing_settings(),
        }
        if len(view_plane_choices(self._data)) > 1:
            controls["view_plane"] = self._view_plane()
        if self._variable is not None:
            controls["variable"] = self._variable
        if self._metabolite_split is not None:
            controls["split_metabolites"] = self._data is self._metabolite_split[1]
            controls["other_metabolite"] = bool(
                self.state.spectral.show_secondary_metabolite_curve
            )
        if self._outline is not None:
            controls["anatomy_slice"] = self._outline_slice()
        anatomy = self._anatomy_controls()
        if anatomy is not None:
            controls["anatomy"] = anatomy
        return controls

    def _outline_slice(self) -> int:
        """Return the anatomy slice shown under the volume outline."""
        spatial = self._outline.state.spatial
        return int(spatial.slice_indices.get(spatial.slicing_dim, 0))

    def _other_dims(self) -> list[str]:
        """Return dimensions that are neither displayed nor the curve.

        Repetitions averaged and channels combined by the display processing
        have no index to choose and are left out.
        """
        reduced = self._reduced_dims()
        return [
            dim
            for dim in self._data.dims
            if dim not in self._spatial_dims
            and dim != self._curve_dim
            and dim not in reduced
        ]

    def _shown_curve_dim(self) -> str | None:
        """Return the dimension the curve panel runs along, or None.

        The curve follows ``state.spectral.secondary_dim``, or the spectrum
        when none is set. While the display processing reduces that
        dimension (averaged repetitions, combined channels), the curve keeps
        every entry and the map shows the processed image, as in ``plot``.
        """
        return self.state.spectral.secondary_dim or (
            SPECTRAL_DIM if self._spectroscopic else None
        )

    def _reduced_dims(self) -> set[str]:
        """Return the dimensions the display processing reduces to one entry."""
        spectral = self.state.spectral
        reduced = set()
        if spectral.combine_channels:
            reduced.add("channel_idx")
        if spectral.average_repetitions and not spectral.cum_sum_repetitions:
            reduced.add("repetition_idx")
        return reduced

    def _other_value_names(self) -> dict[str, list[str]]:
        """Return the names of slider positions with text coordinates.

        A ``metabolite`` slider, for example, shows the metabolite name next
        to its index. Numeric coordinates are not repeated.
        """
        names = {}
        for dim in self._other_dims():
            coord = self._data.coords.get(dim)
            if coord is not None and coord.dims == (dim,) and coord.dtype.kind in "OUS":
                names[dim] = [str(value) for value in coord.values]
        return names

    def _show_data(self, data: xr.DataArray) -> None:
        """Replace the shown data, choose its curve dimension again and render."""
        self._data = data
        self._spectroscopic = _has_spectrum(data)
        spectral = self.state.spectral
        if not fid_is_recoverable(data):
            spectral.show_fid = False
        spectral.secondary_dim = _chosen_curve_dim(
            data, self.state, None, explicit=False
        )
        self._curve_dim = self._shown_curve_dim()
        indices = self.state.spatial.slice_indices
        for dim in list(indices):
            if dim in data.dims:
                indices[dim] = int(min(max(int(indices[dim]), 0), data.sizes[dim] - 1))
        self._overlay_cache = None
        self._render(reset_map_range=True)

    def _other_index(self, dim: str) -> int:
        """Return the selected index of a dimension that is neither image nor curve.

        The spectral position of the map is kept in
        ``state.spectral.selected_spectral_index`` (as in ``plot``); every
        other dimension in ``state.spatial.slice_indices``.
        """
        if dim == SPECTRAL_DIM:
            return int(self.state.spectral.selected_spectral_index or 0)
        return int(self.state.spatial.slice_indices.get(dim, 0))

    def _render(self, *, reset_map_range: bool = False) -> None:
        """Prepare a new curve block and send it with its view description.

        With ``reset_map_range`` and an absolute map range mode, the map
        colour limits become the finite value range of the new plane.
        """
        t0 = time.perf_counter()
        kspace = self._kspace_shown()
        # Overlays, outlines, masks and fits belong to image space.
        overlay_layer = None if kspace else self._overlay_layer()
        self._curve_dim = self._shown_curve_dim()
        self._send_curve_fit_tool()
        self._send_ideal_tool()
        self._send_signal_decay_tool()
        time_dim, time_choices = self._chosen_time_dim()
        image_only = self._curve_dim is None
        # The browser hides values below the colour minimum itself
        # (``min_to_nan``), so the block keeps every value and the colour
        # range can change without Python.
        min_to_nan = self.state.spatial.min_to_nan
        self.state.spatial.min_to_nan = False
        try:
            payload = self._model.get_render_payload(
                self._shown_data(),
                self.state,
                include_curve_block=not image_only,
                time_dim=time_dim,
                spatial_overlay=overlay_layer,
            )
        finally:
            self.state.spatial.min_to_nan = min_to_nan
        if image_only:
            # The block holds the shown image as one "curve point" per voxel.
            curves = (
                None
                if payload.spatial_array is None
                else payload.spatial_array[..., None]
            )
            self._curve_x = np.zeros(1)
        else:
            curves = payload.curve_block
            self._curve_x = np.asarray(payload.spectral_x, dtype=float)
        # Averaged repetitions or combined channels along the curve: the map
        # is the processed image, the curves keep every entry.
        map_plane = (
            payload.spatial_array
            if not image_only and self._curve_dim in self._reduced_dims()
            else None
        )
        if curves is None or np.ndim(curves) != 3:
            reason = payload.spectral_error or payload.spatial_error
            wanted = "an image" if image_only else f"curves along {self._curve_dim}"
            raise ValueError(
                f"plot_scan needs {wanted} in a two-dimensional spatial "
                f"plane{f' ({reason})' if reason else ''}; images need at "
                "least two of read_idx, phase_idx and slice_idx."
            )
        block = np.ascontiguousarray(curves, dtype=np.float32)
        self._plane_curves = block
        self._spatial_dims = list(payload.spatial_dims)
        if self._curve_x.shape != block.shape[2:]:
            raise ValueError(
                f"plot_scan: the curve axis has {self._curve_x.size} points but "
                f"the curves along {self._curve_dim} have {block.shape[2]}; "
                "choose a curve dimension that the display processing keeps."
            )
        self._block_indices = {
            dim: np.asarray(idx, dtype=int)
            for dim, idx in payload.spatial_source_indices.items()
        }
        self._plane_range = _finite_range(block)
        if payload.curve_time_block is None:
            self._time_dim, self._time_curves = None, None
        else:
            self._time_dim = time_dim
            self._time_curves = np.asarray(payload.curve_time_block, dtype=np.float32)
        if reset_map_range and self.state.spatial.range_mode != "Percentage":
            self.state.spatial.vmin, self.state.spatial.vmax = self._plane_range
        n_rows, n_columns, _ = block.shape
        x, y = display_voxel_centres(payload.spatial_extent, (n_rows, n_columns))
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        extent = payload.spatial_extent
        x_range = _cell_edges(x, extent[0], extent[1])
        y_edges = _cell_edges(y, extent[2], extent[3])
        y_range = y_edges[::-1] if payload.spatial_origin == "upper" else y_edges
        anatomy, anatomy_view = _anatomy_layer(payload)
        self._anatomy_plane = dict(
            payload.metadata.get("anatomy_plane_selection") or {}
        )
        if anatomy_view:
            anatomy_view.update(self._anatomy_slice_view())
            # Show the whole anatomy field of view, as plot does.
            x0, x1, y0, y1 = anatomy_view["extent"]
            x_range = _widened_range(x_range, x0, x1)
            y_range = _widened_range(y_range, y0, y1)
        outline_view: dict[str, Any] = {}
        if self._outline is not None and not kspace:
            # Single voxel or slab: the anatomy plane with the cut of the
            # excited volume replaces the map of one voxel.
            anatomy, anatomy_view, outline_view = _outline_layer(
                self._outline.render(), self._outline.volume.kind
            )
        self._overlay_plane = dict(
            payload.metadata.get("overlay_plane_selection") or {}
        )
        overlay, overlay_view = _overlay_layer_view(payload, overlay_layer)
        if overlay_view and self._overlay_plane:
            overlay_view["slice_index"] = next(iter(self._overlay_plane.values()))
        compute_ms = (time.perf_counter() - t0) * 1e3

        row_dim, column_dim = self._spatial_dims
        view = {
            "shape": [int(n) for n in block.shape],
            # One voxel (single-voxel spectroscopy): no image panel.
            "single_voxel": block.shape[:2] == (1, 1),
            "x": x.tolist(),
            "y": y.tolist(),
            "x_range": x_range,
            "y_range": y_range,
            "x_label": payload.labels.get("spatial_x", column_dim),
            "y_label": payload.labels.get("spatial_y", row_dim),
            "row_dim": row_dim,
            "column_dim": column_dim,
            "view_plane_options": [
                [get_spatial_dim_label(dim), dim]
                for dim in view_plane_choices(self._data)
            ],
            "row_indices": self._block_indices[row_dim].tolist(),
            "column_indices": self._block_indices[column_dim].tolist(),
            "curve_dim": self._curve_dim,
            "spectroscopic": self._spectroscopic,
            "curve_x": self._curve_x.tolist(),
            "curve_label": "" if image_only else payload.spectral_label or "",
            "curve_invert_x": bool(payload.spectral_invert_x),
            "map_fixed": map_plane is not None,
            "average_range": _average_range(payload, self._curve_x),
            "value_label": payload.labels.get("spectral_y", "Intensity"),
            "colorscale": Colormap(payload.spatial_cmap).to_plotly(),
            "plane_range": list(self._plane_range),
            "range_modes": list(MAP_RANGE_MODES),
            "can_save_settings": self._scan is not None,
            "title": payload.title,
            "scalebar": _scalebar_view(payload.spatial_scalebar),
            "smoothing_options": list(MAP_SMOOTHING_OPTIONS),
            "map_cmaps": self._cmap_choices(self.state.spatial.cmap),
            "anatomy_cmaps": self._cmap_choices(self.state.anatomy.cmap),
            "anatomy": anatomy_view,
            "outline": outline_view,
            "overlay": overlay_view,
            "overlay_options": self._overlay_options(),
            "other_dims": {
                dim: int(self._data.sizes[dim]) for dim in self._other_dims()
            },
            "other_values": self._other_value_names(),
            "variable_options": list(self._variables),
            "spec_part_options": list(SPEC_PART_OPTIONS),
            "spec_unit_options": list(SPEC_UNIT_OPTIONS),
            "combine_methods": list(CHANNEL_COMBINE_METHODS),
            "repetition_modes": list(REPETITION_MODES),
            "time": self._time_view(time_choices),
            "compute_ms": round(compute_ms, 1),
            "block_megabytes": round(block.nbytes / 1e6, 2),
            "kspace": self._kspace_view(),
        }
        self._mask_plane = (
            {"unavailable": "masks are drawn in image space; show the image."}
            if kspace
            else self._plane_context(payload, (n_rows, n_columns))
        )
        view["mask_unavailable_reason"] = self._mask_plane.get("unavailable", "")
        selection = self._selection_from_state(n_curve=block.shape[2])
        with self.hold_sync():
            self._block = block.tobytes()
            self._map_plane = (
                b""
                if map_plane is None
                else np.ascontiguousarray(map_plane, dtype=np.float32).tobytes()
            )
            self._anatomy = anatomy
            self._anatomy_opacity = float(self.state.anatomy.opacity)
            anatomy_state = self.state.anatomy
            self._anatomy_range = (
                [float(anatomy_state.vmin), float(anatomy_state.vmax)]
                if anatomy_state.range_mode == "Percentage"
                else []
            )
            self._overlay = overlay
            self._overlay_opacity = float(self.state.overlay.alpha)
            self._map_range = self._map_range_settings()
            self._display = self._display_settings()
            # The browser builds its sliders when the view arrives; a new
            # plane or data set brings new slider dimensions and indices.
            self._controls = self._current_controls()
            self._view = view
            self._mask_planes = self._displayed_mask_planes()
            self._update_roi_mean()
            self._send_time_course()
            self._applying_python_selection = True
            try:
                self.selection = selection
            finally:
                self._applying_python_selection = False
            self._update_fit_curves(force=True)
            self._update_other_metabolite(force=True)
            self._slice_lines = self._current_slice_lines()
            self._pointer_plane = self._current_pointer_plane()
        # Views that show this view's slice follow the new plane.
        for viewer in self._slice_viewers:
            viewer._slice_lines = viewer._current_slice_lines()
        self._show_frame_in_partners()

    def _plane_context(self, payload: Any, shape: tuple[int, int]) -> dict[str, Any]:
        """Describe the shown plane for drawing, or why masks are unavailable.

        Shown voxels are mapped to source voxels through the curve block
        indices, the same mapping that selects the spectra. It holds for every
        rotation, coordinate direction and coordinate mode (``rps`` or
        ``xyz``) because it does not depend on coordinate values.
        """
        shown_dims = list(payload.spatial_dims or [])
        dims = mask_dims(self._data)
        if len(shown_dims) != 2 or not set(shown_dims).issubset(dims):
            return {
                "unavailable": (
                    f"masks need a plane of {', '.join(dims) or 'spatial'} "
                    f"dimensions; the map shows {shown_dims}."
                )
            }
        others = [dim for dim in dims if dim not in shown_dims]
        slicing_dim = others[0] if others else None
        row_dim, column_dim = shown_dims
        return {
            "extent": list(payload.spatial_extent),
            "shape": shape,
            "row_dim": row_dim,
            "column_dim": column_dim,
            "source_indices": {
                dim: self._block_indices[dim].copy() for dim in shown_dims
            },
            "slicing_dim": slicing_dim,
            "slice_index": (
                int(self.state.spatial.slice_indices.get(slicing_dim, 0))
                if slicing_dim
                else None
            ),
        }
