"""Python half of the analysis tools: fit, curve fit, signal decay and IDEAL.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_Tools`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.analysis.spectroscopy.frequency import freq_hz_to_ppm
from hypermri.analysis.spectroscopy.peak_fit_setup import frequency_axis_hz
from hypermri.visualization.curve_fit_tool import (
    CurveModel,
    curve_fit_curves,
    curve_fit_settings,
    curve_fit_summary,
    curve_model,
    run_curve_fit,
    set_curve_fit_option,
    voxel_parameters,
)
from hypermri.visualization.fit_tool import (
    add_peak,
    add_residual_line,
    align_spectra,
    detect_peaks,
    edit_peak,
    fit_tool_settings,
    lines_from_fit,
    next_problem_voxel,
    refit_voxel,
    remove_peak,
    run_fit,
    set_fit_option,
    store_fit_on_scan,
    stored_fit,
)
from hypermri.visualization.ideal_tool import (
    ideal_maps,
    ideal_settings,
    ideal_summary,
    run_ideal,
    set_ideal_option,
)
from hypermri.visualization.signal_decay_tool import (
    FILTER_OPTIONS,
    SignalDecayState,
    confirm_signal_decay,
    set_signal_decay_option,
    signal_decay_filter_params,
    signal_decay_settings,
)
from hypermri.visualization.spatial_overlay import (
    normalize_overlays,
    preferred_fit_variable,
)

_import_optional("anywidget", "visualization")
_import_optional("cmap", "visualization")

from hypermri.visualization.scan_view.options import (  # noqa: E402
    CURVE_FIT_OVERLAY_NAME,
    FIT_OVERLAY_NAME,
    IDEAL_OVERLAY_NAME,
    SPECTRAL_DIM,
)
from hypermri.visualization.scan_view.secondary_slices import (  # noqa: E402
    _follow_map,
)


def _stored_fit_mismatch(fit_results: xr.Dataset, data: xr.DataArray) -> str | None:
    """Return why a stored fit does not belong to the voxels of ``data``.

    Every dimension of ``fit_freqs`` other than ``metabolite`` must have the
    size of the same dimension of ``data``, or size 1. Returns None when the
    fit matches, else a short reason for the tool message.
    """
    for dim, size in fit_results["fit_freqs"].sizes.items():
        if dim == "metabolite" or size == 1:
            continue
        if data.sizes.get(dim) != size:
            return (
                f"its {dim} has {size} entries but the shown data have "
                f"{data.sizes.get(dim, 0)}"
            )
    return None


class _Tools:
    """Python half of the analysis tools: fit, curve fit, signal decay and IDEAL of :class:`ScanView`."""

    # ── fit tool ───────────────────────────────────────────────────────

    def _fit_tool_offered(self) -> bool:
        """Whether the shown data are spectra the fit tool can fit."""
        return (
            self._spectroscopic
            and self._variable is None
            and self._curve_dim == SPECTRAL_DIM
        )

    def _send_fit_tool(self) -> None:
        """Send the peak table and settings of the fit tool to the browser."""
        self._fit_tool = {
            **fit_tool_settings(self._fit_tool_state),
            "offered": self._fit_tool_offered(),
            "fitted": self._fit is not None,
            "aligned": self._before_alignment is not None,
            "earlier_fit": "" if self._earlier_fit is None else self._earlier_fit[0],
            "search_bands": self._search_bands(),
        }

    def _search_bands(self) -> list[dict[str, list[float] | None]]:
        """Return the frequency search range of every table line.

        Per row ``{"Hz": [low, high], "ppm": [low, high] or None}``: the start
        frequency plus and minus half the search width, in both spectral
        units so the curve panel can draw it in the shown one. ``ppm`` is None
        when the data have no ppm calibration.
        """
        bands = []
        for peak in self._fit_tool_state.peaks:
            half = 0.5 * float(peak.freq_range_hz)
            hz = [float(peak.freq_hz) - half, float(peak.freq_hz) + half]
            try:
                ppm = freq_hz_to_ppm(freq_hz=np.asarray(hz), scan=self._model.data_obj)
                ppm = None if ppm is None else [float(value) for value in ppm]
            except (TypeError, ValueError, KeyError, AttributeError):
                ppm = None
            bands.append({"Hz": hz, "ppm": ppm})
        return bands

    def _load_stored_fit(self) -> None:
        """Show a spectral fit stored on the scan and keep it as start values.

        The fit of :func:`~hypermri.visualization.fit_tool.stored_fit` becomes
        the hidden overlay "fit" (its maps and fitted spectra) and the source
        of start values for the fit tool. A fit whose voxel dimensions do not
        match the shown data, or a fit of aligned spectra while the acquired
        spectra are shown, is not used; the tool message says why.
        """
        if not self._fit_tool_offered():
            return
        found = stored_fit(self._scan)
        if found is None:
            return
        name, fit_results, fit_params = found
        reason = _stored_fit_mismatch(fit_results, self._data)
        if reason is None and "frequency_alignment_offset_hz" in getattr(
            getattr(self._scan, "data_review", None), "coords", {}
        ):
            reason = "it belongs to aligned spectra"
        if reason is not None:
            self._fit_tool_state.message = f"Stored fit {name} not used: {reason}."
            return
        self._earlier_fit = (name, fit_results)
        if fit_params is not None:
            self._fit = (fit_params, fit_results)
        if FIT_OVERLAY_NAME not in self._overlays:
            overlays, _ = normalize_overlays(
                {FIT_OVERLAY_NAME: fit_results},
                self.state.spatial.coordinate_mode,
                geometry_reference=self._data,
            )
            first_overlay = not self._overlays
            self._overlays = {**self._overlays, **overlays}
            self._overlay_cache = None
            overlay = self.state.overlay
            if overlay.active_variable not in self._overlay_names():
                overlay.active_variable = (
                    f"{FIT_OVERLAY_NAME}_"
                    f"{preferred_fit_variable(fit_results.data_vars)}"
                )
            if first_overlay:
                # Shown only when the overlay is switched on.
                overlay.visible = False
                overlay.alpha = 1.0
        self._fit_tool_state.message = (
            f"Stored fit {name} shown as overlay {FIT_OVERLAY_NAME!r}; "
            "'Lines from fit' takes its lines and start values."
        )

    def _apply_browser_fit_action(self, content: dict[str, Any]) -> None:
        """Dispatch one fit tool message from the browser."""
        if not self._fit_tool_offered():
            raise ValueError(
                "Fit tool: the shown data are not spectra along acquisition_idx."
            )
        tool = self._fit_tool_state
        action = content.get("action")
        if action == "detect":
            detect_peaks(tool, self._data)
        elif action == "add_at_cursor":
            add_peak(tool, self._cursor_frequency_hz())
        elif action == "edit":
            edit_peak(
                tool, int(content["row"]), str(content["column"]), content["value"]
            )
        elif action == "remove":
            remove_peak(tool, int(content["row"]))
        elif action == "option":
            set_fit_option(tool, str(content["name"]), content["value"])
        elif action == "run":
            self._run_fit_tool()
        elif action == "lines_from_fit":
            if self._earlier_fit is None:
                raise ValueError(
                    "Fit tool: there is no earlier fit to take lines from."
                )
            lines_from_fit(tool, self._earlier_fit[1])
        elif action == "save":
            self._save_fit_from_browser(content, "spectral", tool)
        elif action == "refit_voxel":
            self._refit_selected_voxel()
        elif action == "next_problem":
            self._select_next_problem_voxel()
        elif action == "residual_line":
            fit_params, _ = self._shown_fit("Line from residual")
            add_residual_line(
                tool,
                self._data,
                fit_params,
                self.state.spatial.slice_indices,
                (float(content["low_hz"]), float(content["high_hz"])),
            )
        elif action == "align":
            self._align_spectra()
        elif action == "undo_align":
            self._undo_alignment()
        else:
            raise ValueError(f"Fit tool: unknown action {action!r}.")

    def _cursor_frequency_hz(self) -> float:
        """Return the frequency in Hz of the spectral cursor."""
        axis = frequency_axis_hz(self._data)
        index = self.state.spectral.selected_spectral_index or 0
        return float(axis[min(max(int(index), 0), axis.size - 1)])

    def _run_fit_tool(self) -> None:
        """Fit all voxels, store the fit on the scan and show it."""
        fit_params, fit_results = run_fit(
            self._fit_tool_state,
            self._data,
            scan=self._scan,
            earlier_fit=None if self._earlier_fit is None else self._earlier_fit[1],
        )
        self._show_fit(fit_params, fit_results)

    def _shown_fit(self, step: str) -> tuple[xr.Dataset, xr.Dataset]:
        """Return the fit of the shown spectra, or fail naming ``step``."""
        if self._fit is None:
            raise ValueError(f"{step}: fit all voxels first.")
        return self._fit

    def _refit_selected_voxel(self) -> None:
        """Fit the selected voxel again with the table and show the result."""
        fit_params, fit_results = self._shown_fit("Refit voxel")
        updated = refit_voxel(
            self._fit_tool_state,
            self._data,
            fit_params,
            fit_results,
            self.state.spatial.slice_indices,
            scan=self._scan,
        )
        self._show_fit(fit_params, updated)

    def _select_next_problem_voxel(self) -> None:
        """Select the next voxel in the order of fit problems.

        A voxel in another slice moves the map to that slice, as a linked
        pointer does.
        """
        _, fit_results = self._shown_fit("Next problem voxel")
        indices = self.state.spatial.slice_indices
        target = next_problem_voxel(self._fit_tool_state, fit_results, indices)
        plane_moved = any(
            indices.get(dim, 0) != index
            for dim, index in target.items()
            if dim not in self._spatial_dims
        )
        indices.update(target)
        if plane_moved and self._model._uses_xyz_coordinates(self.state):
            self._follow_map_with_layers()
        self._render()

    def _align_spectra(self) -> None:
        """Show the spectra shifted to the weighted mean fitted frequency.

        The view shows the aligned copy and its shifted fit; the scan keeps
        its acquired data and gets the aligned fit and samples under the
        review names of :func:`store_fit_on_scan`.
        """
        if self._fit is None:
            raise ValueError("Fit tool: run the fit before aligning the spectra.")
        if self._metabolite_split is not None:
            raise ValueError(
                "Fit tool: spectra of a view with a metabolite split cannot be aligned."
            )
        fit_params, fit_results = self._fit
        aligned, aligned_params, aligned_results = align_spectra(
            self._fit_tool_state, self._data, fit_params, fit_results, scan=self._scan
        )
        if self._before_alignment is None:
            self._before_alignment = (self._data, fit_params, fit_results)
        self._data = aligned
        self._show_fit(aligned_params, aligned_results)

    def _undo_alignment(self) -> None:
        """Show the spectra before the first alignment and their fit again."""
        if self._before_alignment is None:
            raise ValueError("Fit tool: the spectra are not aligned.")
        data, fit_params, fit_results = self._before_alignment
        self._before_alignment = None
        self._data = data
        self._show_fit(fit_params, fit_results)
        self._fit_tool_state.message = "Alignment undone."

    def _show_fit(self, fit_params: xr.Dataset, fit_results: xr.Dataset) -> None:
        """Keep a fit of the shown data, store it on the scan and show it.

        The fit also becomes the source of start values for the next fit.
        """
        self._fit = (fit_params, fit_results)
        self._earlier_fit = ("last fit", fit_results)
        if self._scan is not None:
            store_fit_on_scan(self._scan, fit_params, fit_results, self._data)
        self._show_result_maps(
            FIT_OVERLAY_NAME,
            fit_results,
            preferred_fit_variable(fit_results.data_vars),
        )

    def _show_result_maps(self, name: str, result: xr.Dataset, variable: str) -> None:
        """Show fit result maps as the overlay ``name`` and redraw.

        The overlay replaces an earlier one of the same name; ``variable``
        becomes the shown overlay map with its full value range.
        """
        overlays, _ = normalize_overlays(
            {name: result},
            self.state.spatial.coordinate_mode,
            geometry_reference=self._data,
        )
        first_overlay = not self._overlays
        self._overlays = {**self._overlays, **overlays}
        self._overlay_cache = None
        overlay = self.state.overlay
        if first_overlay:
            overlay.alpha = 1.0
        overlay.active_variable = f"{name}_{variable}"
        overlay.visible = True
        _follow_map(overlay)
        overlay.vmin, overlay.vmax = self._full_overlay_range()
        self._render()
        self._controls = self._current_controls()
        self._update_fit_curves(force=True)
        self._update_roi_mean()

    # ── curve-fit tool ─────────────────────────────────────────────────

    @property
    def curve_fit_result(self) -> xr.Dataset | None:
        """Result of the last run of the curve-fit tool, or None.

        Returns
        -------
        xarray.Dataset or None
            The fit result of :func:`~hypermri.visualization.curve_fit_tool.run_curve_fit`
            (for example ``adc`` in mm²/s, ``t2`` or ``t1`` in ms), with the
            voxel dimensions of the shown data. It is kept in the view until
            the next run; :meth:`save_fit` stores a copy on the plotted object.
        """
        return None if self._curve_fit is None else self._curve_fit[1]

    def _shown_curve_model(self) -> CurveModel | None:
        """Return the curve-fit model of the shown curve axis, or None."""
        if self._variable is not None:
            return None
        return curve_model(self._data, self._curve_dim)

    def _send_curve_fit_tool(self) -> None:
        """Send model and settings of the curve-fit tool to the browser."""
        model = self._shown_curve_model()
        self._curve_fit_tool = {
            **curve_fit_settings(self._curve_fit_state, model),
            "fitted": self._curve_fit is not None
            and model is not None
            and self._curve_fit[0] == model,
        }

    def _apply_browser_curve_fit_action(self, content: dict[str, Any]) -> None:
        """Dispatch one curve-fit message from the browser."""
        action = content.get("action")
        if action == "option":
            set_curve_fit_option(
                self._curve_fit_state, str(content.get("name")), content.get("value")
            )
        elif action == "run":
            self._run_curve_fit()
        elif action == "save":
            self._save_fit_from_browser(content, "curve", self._curve_fit_state)
        else:
            raise ValueError(f"ScanView: unknown curve-fit action {action!r}.")

    def _run_curve_fit(self) -> None:
        """Fit the model of the curve axis in all voxels and show the maps."""
        model = self._shown_curve_model()
        if model is None:
            raise ValueError(
                "Curve fit: choose b-values, echoes or inversion times as curve axis."
            )
        result = run_curve_fit(self._data, model, self._curve_fit_state)
        self._curve_fit = (model, result)
        self._curve_fit_state.message = curve_fit_summary(model, result)
        self._show_result_maps(CURVE_FIT_OVERLAY_NAME, result, model.result_variable)

    def _curve_fit_curves(self) -> list[dict[str, Any]]:
        """Return fit curve and residual of the selected voxel for the curve panel."""
        if self._curve_fit is None:
            return []
        model, result = self._curve_fit
        if model.dim != self._curve_dim or self._variable is not None:
            return []
        selection = self._selection_from_state(n_curve=max(self._curve_x.size, 1))
        measured = self._plane_curves[selection["row"], selection["column"]]
        parameters = voxel_parameters(result, self.state.spatial.slice_indices)
        return curve_fit_curves(model, parameters, self._curve_x, measured, self.state)

    # ── signal-decay tool ──────────────────────────────────────────────

    def _offer_signal_decay(self, prepared: Any, state: SignalDecayState) -> None:
        """Open the signal-decay tool for prepared CSI raw data.

        Used once by
        :func:`~hypermri.visualization.signal_decay.plot_signal_decay`, right
        after the view of ``prepared.original`` is built. Applies the
        starting filter, shows the filtered or unfiltered data and chooses
        the tool.
        """
        self._signal_decay = (prepared, state)
        self._filter_signal_decay()
        self.tool = "signal_decay"

    def _send_signal_decay_tool(self) -> None:
        """Send choices and decay curves of the signal-decay tool."""
        if self._signal_decay is None:
            self._signal_decay_tool = {}
            return
        prepared, state = self._signal_decay
        repetition = int(self.state.spatial.slice_indices.get("repetition_idx", 0))
        self._signal_decay_tool = signal_decay_settings(
            state, prepared, repetition, filter_error=self._signal_decay_error
        )

    def _apply_browser_signal_decay_action(self, content: dict[str, Any]) -> None:
        """Dispatch one signal-decay message from the browser."""
        prepared, state = self._signal_decay
        action = content.get("action")
        if action == "option":
            name = str(content.get("name"))
            set_signal_decay_option(state, name, content.get("value"))
            state.message = ""
            if name in FILTER_OPTIONS or name == "show_filtered":
                self._filter_signal_decay(refilter=name != "show_filtered")
        elif action == "confirm":
            if self._signal_decay_error is not None:
                raise ValueError(
                    "Signal decay: the filter cannot be applied, so it cannot "
                    f"be confirmed: {self._signal_decay_error}"
                )
            state.message = confirm_signal_decay(state, prepared)
        else:
            raise ValueError(f"ScanView: unknown signal-decay action {action!r}.")

    def _filter_signal_decay(self, *, refilter: bool = True) -> None:
        """Filter the raw data with the chosen filter and show the result.

        The view shows the filtered reconstruction, or the unfiltered one
        when ``show_filtered`` is off or the filter fails; the failure is
        kept for the tool's message and disables "Confirm".
        """
        prepared, state = self._signal_decay
        if refilter:
            try:
                prepared.update(
                    signal_decay_filter_params(state, prepared.spatial_spacing_mm)
                )
            except (ValueError, RuntimeError, NotImplementedError) as error:
                self._signal_decay_error = str(error)
                state.message = f"Filter failed: {error}"
            else:
                self._signal_decay_error = None
        shown = (
            prepared.filtered
            if state.show_filtered and self._signal_decay_error is None
            else prepared.original
        )
        self._show_data(shown)

    # ── IDEAL tool ─────────────────────────────────────────────────────

    @property
    def ideal_result(self) -> xr.Dataset | None:
        """Result of the last run of the IDEAL tool, or None.

        Returns
        -------
        xarray.Dataset or None
            The result of :func:`~hypermri.visualization.ideal_tool.run_ideal`:
            complex ``fit_amps`` (signal units, dimension ``metabolite``
            last) and, when fitted, ``field_offset`` in Hz. It is kept in the
            view until the next run; :meth:`save_fit` stores a copy on the
            plotted object. The overlay ``"ideal"``
            shows the magnitude maps of it (:func:`~hypermri.visualization.ideal_tool.ideal_maps`).
        """
        return self._ideal

    def _send_ideal_tool(self) -> None:
        """Send the settings of the IDEAL tool to the browser."""
        # A shown fit-result variable is no echo signal.
        curve_dim = self._curve_dim if self._variable is None else None
        self._ideal_tool = {
            **ideal_settings(self._ideal_state, self._data, curve_dim),
            "fitted": self._ideal is not None,
        }

    def _apply_browser_ideal_action(self, content: dict[str, Any]) -> None:
        """Dispatch one IDEAL message from the browser."""
        action = content.get("action")
        if action == "option":
            set_ideal_option(
                self._ideal_state, str(content.get("name")), content.get("value")
            )
        elif action == "run":
            self._run_ideal()
        elif action == "save":
            self._save_fit_from_browser(content, "ideal", self._ideal_state)
        else:
            raise ValueError(f"ScanView: unknown IDEAL action {action!r}.")

    def _run_ideal(self) -> None:
        """Separate the components in all voxels and show their maps."""
        if self._variable is not None or self._curve_dim != "echo_idx":
            raise ValueError("IDEAL: choose the echoes (echo_idx) as curve axis.")
        result = run_ideal(self._data, self._ideal_state)
        self._ideal = result
        self._ideal_state.message = ideal_summary(result)
        maps = ideal_maps(result)
        self._show_result_maps(IDEAL_OVERLAY_NAME, maps, next(iter(maps.data_vars)))
