"""Anatomy opacity, map range, display switches, curve axes, processing and saved view settings.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_Display`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypermri._optional import _import_optional
from hypermri.utils.mri.fourier import fid_is_recoverable
from hypermri.visualization.colormap_policy import (
    COMMON_CMAPS,
    is_cmap_available,
)
from hypermri.visualization.state_persistence import (
    save_plot_state,
)

_import_optional("anywidget")
_import_optional("cmap")

from hypermri.visualization.scan_view.options import (  # noqa: E402
    _DISPLAY_SWITCHES,
    CHANNEL_COMBINE_METHODS,
    MAP_RANGE_MODES,
    MAP_SMOOTHING_OPTIONS,
    REPETITION_MODES,
    SPECTRAL_DIM,
    _check_limits,
    _checked_curve_axes,
)


class _Display:
    """Anatomy opacity, map range, display switches, curve axes, processing and saved view settings of :class:`ScanView`."""

    # ── anatomy ────────────────────────────────────────────────────────

    def set_anatomy_opacity(self, opacity: float) -> None:
        """Set how opaque the CSI map is drawn above the anatomy image.

        As in ``plot``, the anatomy image is always fully opaque and this
        value is the opacity of the CSI map layer: 0 shows only the anatomy,
        1 hides it under the map. The value is stored in
        ``state.anatomy.opacity`` and kept across recomputations. Nothing is
        recomputed.

        Parameters
        ----------
        opacity : float
            Opacity of the CSI map in [0, 1].

        Raises
        ------
        ValueError
            If ``opacity`` is not a finite number in [0, 1].
        """
        value = float(opacity)
        if not (np.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(
                f"ScanView.set_anatomy_opacity needs a value in [0, 1]; got {opacity!r}."
            )
        self._anatomy_opacity = value

    def set_anatomy_range(self, low_percent: float, high_percent: float) -> None:
        """Set the colour range of the anatomy in percent of the shown slice.

        0 % is the minimum and 100 % the maximum of the finite values of the
        anatomy slice drawn under the map, as the range mode "Percentage" of
        ``plot``. The limits are stored in ``state.anatomy`` (``vmin``,
        ``vmax`` and ``range_mode = "Percentage"``) and kept for other
        slices and planes. Only the anatomy is redrawn; nothing is
        recomputed.

        Parameters
        ----------
        low_percent, high_percent : float
            Lower and upper colour limit in percent, with
            ``0 <= low_percent < high_percent <= 100``.

        Raises
        ------
        ValueError
            If a limit is not finite, lies outside [0, 100], or the lower
            limit is not below the upper one. The state is then unchanged.

        Examples
        --------
        >>> view = plot_scan(csi, anatomy=rare)  # doctest: +SKIP
        >>> view.set_anatomy_range(0, 60)  # brighten a dark anatomy  # doctest: +SKIP
        """
        self._store_anatomy_range(low_percent, high_percent)
        self._anatomy_range = [float(low_percent), float(high_percent)]

    def _store_anatomy_range(self, low_percent: Any, high_percent: Any) -> None:
        """Check percentage limits of the anatomy and store them in the state."""
        low, high = float(low_percent), float(high_percent)
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low < high <= 100.0):
            raise ValueError(
                "ScanView anatomy range needs 0 <= low < high <= 100 percent; "
                f"got {low_percent!r} and {high_percent!r}."
            )
        anatomy = self.state.anatomy
        anatomy.range_mode = "Percentage"
        anatomy.vmin, anatomy.vmax = low, high

    # ── map colour range ───────────────────────────────────────────────

    def set_map_range(
        self,
        range_mode: str | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Set the colour range of the CSI map.

        The settings are stored in ``state.spatial`` and only redraw the map;
        nothing is recomputed. Choosing another range mode without limits
        starts from its full range: 0-100 % or the finite value range of the
        shown plane. In the browser, the colour range menu also offers
        percentages of the whole plane instead of the shown map and converts
        the limits when the mode changes, so the colours stay the same as far
        as percentages within 0-100 allow.

        Parameters
        ----------
        range_mode : {"Percentage", "Min/Max", "Absolute"} or None, optional
            ``"Percentage"``: vmin and vmax are percentages (0-100) of the
            finite range of the shown map, as in ``plot``. ``"Min/Max"`` and
            ``"Absolute"``: vmin and vmax are limits in signal units of the
            shown complex part; ``"Min/Max"`` returns to the full plane range
            when the complex part or the line broadening changes,
            ``"Absolute"`` keeps the limits. None keeps the current mode.
        vmin, vmax : float or None, optional
            Lower and upper colour limit in the unit of ``range_mode``. None
            keeps the current limit, or the full range after a mode change.

        Raises
        ------
        ValueError
            If ``range_mode`` is unknown, a limit is not finite, or vmin is
            not below vmax.

        Examples
        --------
        >>> view.set_map_range("Absolute", vmin=0.0, vmax=5e4)  # doctest: +SKIP
        >>> view.set_map_range("Percentage", vmin=10, vmax=90)  # doctest: +SKIP
        """
        spatial = self.state.spatial
        mode = spatial.range_mode if range_mode is None else range_mode
        self._check_map_range_mode(mode)
        if mode == spatial.range_mode:
            low, high = spatial.vmin, spatial.vmax
        elif mode == "Percentage":
            low, high = 0.0, 100.0
        else:
            low, high = self._plane_range
        settings = {
            "range_mode": mode,
            "vmin": low if vmin is None else vmin,
            "vmax": high if vmax is None else vmax,
        }
        self._check_map_range(settings)
        self._map_range = settings

    @staticmethod
    def _check_map_range_mode(range_mode: str) -> None:
        """Fail early on a map range mode this view cannot draw."""
        if range_mode not in MAP_RANGE_MODES:
            raise ValueError(
                f"ScanView map range_mode must be one of {MAP_RANGE_MODES}; "
                f"got {range_mode!r}."
            )

    def _check_map_range(self, settings: dict[str, Any]) -> None:
        """Validate map range settings from Python or the browser."""
        self._check_map_range_mode(settings["range_mode"])
        _check_limits("ScanView map range", settings["vmin"], settings["vmax"])

    def _map_range_settings(self) -> dict[str, Any]:
        """Return the map range of ``state.spatial`` as sent to the browser."""
        spatial = self.state.spatial
        return {
            "range_mode": spatial.range_mode,
            "vmin": float(spatial.vmin),
            "vmax": float(spatial.vmax),
        }

    # ── display switches ───────────────────────────────────────────────

    def set_display(
        self,
        *,
        smoothing: str | None = None,
        min_to_nan: bool | None = None,
        show_scalebar: bool | None = None,
        show_colorbar: bool | None = None,
        show_grid_lines: bool | None = None,
        show_tooltips: bool | None = None,
        show_crosshair: bool | None = None,
        map_cmap: str | None = None,
        anatomy_cmap: str | None = None,
        show_slice_lines: bool | None = None,
        show_masks: bool | None = None,
        show_mask_outlines: bool | None = None,
        overlay_smoothing: str | None = None,
        show_overlay_tooltips: bool | None = None,
    ) -> None:
        """Set how the CSI map is drawn, as the display checkboxes of ``plot``.

        The settings are stored in :attr:`state` and only redraw the map,
        except the scalebar, whose length Python chooses for the shown plane.
        None keeps a setting. The browser's "display" controls set the same
        values; the two overlay settings are in its "overlay" controls.

        Parameters
        ----------
        smoothing : {"off", "fast", "best"} or None, optional
            Smoothing between voxel centres: ``"off"`` shows each voxel as a
            square, ``"fast"`` and ``"best"`` are Plotly's heatmap smoothing.
            Stored as ``state.spatial.interpolation`` (False for ``"off"``)
            and ``state.spatial.zsmooth_mode``. Voxel values and indices are
            unchanged; only the drawing is smoothed.
        min_to_nan : bool or None, optional
            Leave voxels below the lower colour limit transparent
            (``state.spatial.min_to_nan``), so an anatomy below shows through.
        show_scalebar : bool or None, optional
            Draw a scalebar in the lower right corner
            (``state.spatial.show_scalebar``). It appears only when the image
            axes have a physical voxel spacing, not for plain indices.
        show_colorbar : bool or None, optional
            Draw the colour bar beside the map (``state.spatial.show_colorbar``).
            It starts on for fitted maps and off otherwise.
        show_grid_lines : bool or None, optional
            Draw the axis grid lines over the map
            (``state.global_settings.show_grid_lines``).
        show_tooltips : bool or None, optional
            Show voxel indices, position, value and mask coverage when the
            mouse rests on the map, and the values under the mouse on the
            spectrum and time course (``state.global_settings.show_tooltips``,
            off by default). Clicking selects a voxel or curve position
            either way.
        show_crosshair : bool or None, optional
            Draw the dotted lines through the selected voxel
            (``state.global_settings.show_crosshair``). The box around the
            voxel stays.
        map_cmap : str or None, optional
            Colormap of the map (``state.spatial.cmap``), a name the ``cmap``
            package knows, such as ``"bone"``, ``"viridis"`` or ``"inferno"``.
            The browser offers a short list; any other known name works too.
        anatomy_cmap : str or None, optional
            Colormap of the anatomy under the map (``state.anatomy.cmap``).
            With an image laid over another image, two different colormaps
            tell the two apart.
        show_slice_lines : bool or None, optional
            Draw the shown slices of other views as coloured bands
            (:meth:`show_slices_of`, :func:`plot_side_by_side`). A choice
            of this view, not stored in :attr:`state`; off by default. The
            links stay, so switching it on shows the current slices.
        show_masks : bool or None, optional
            Draw the masks over the map. A choice of this view, not stored
            in :attr:`state`; on by default. Hidden masks are kept and still
            count for the ROI mean and the mask coverage in the tooltips.
        show_mask_outlines : bool or None, optional
            Draw the outlines drawn into each mask (:attr:`mask_outlines`)
            as lines in the mask colour, at the precision they were drawn;
            where an outline erased area, the boundary of the remaining
            drawn area is shown instead. A choice of this view,
            not stored in :attr:`state`; on by default. Only shown while
            ``show_masks`` is on.
        overlay_smoothing : {"off", "fast", "best"} or None, optional
            Smoothing of the overlay drawn above the map, as ``smoothing``
            for the map (``state.overlay.smoothing``, ``"off"`` by default).
            The overlay values are unchanged.
        show_overlay_tooltips : bool or None, optional
            Add the overlay value, with the overlay's colour-bar label, to
            the hover label of the map (``state.overlay.show_tooltips``, off
            by default). The value is that of the overlay voxel nearest to
            the centre of the map voxel under the mouse; outside the overlay
            none is shown. With ``show_tooltips`` off the label shows the
            position and the overlay value only.

        Raises
        ------
        ValueError
            If ``smoothing`` or ``overlay_smoothing`` is not one of
            ``"off"``, ``"fast"``, ``"best"``, or a colormap name is unknown.

        Examples
        --------
        >>> view.set_display(smoothing="best", min_to_nan=False)  # doctest: +SKIP
        >>> view.set_display(show_scalebar=True, show_tooltips=True)  # doctest: +SKIP
        >>> view.set_display(map_cmap="inferno", anatomy_cmap="gray")  # doctest: +SKIP
        >>> view.set_display(show_masks=False)  # doctest: +SKIP
        >>> view.set_display(overlay_smoothing="best", show_overlay_tooltips=True)  # doctest: +SKIP
        """
        given = {
            "smoothing": smoothing,
            "min_to_nan": min_to_nan,
            "show_scalebar": show_scalebar,
            "show_colorbar": show_colorbar,
            "show_grid_lines": show_grid_lines,
            "show_tooltips": show_tooltips,
            "show_crosshair": show_crosshair,
            "map_cmap": map_cmap,
            "anatomy_cmap": anatomy_cmap,
            "show_slice_lines": show_slice_lines,
            "show_masks": show_masks,
            "show_mask_outlines": show_mask_outlines,
            "overlay_smoothing": overlay_smoothing,
            "show_overlay_tooltips": show_overlay_tooltips,
        }
        settings = dict(self._display_settings())
        settings.update({k: value for k, value in given.items() if value is not None})
        self._check_display(settings)
        self._display = settings

    @staticmethod
    def _check_display(settings: dict[str, Any]) -> None:
        """Fail early on display settings this view cannot draw."""
        for name in ("smoothing", "overlay_smoothing"):
            if settings[name] not in MAP_SMOOTHING_OPTIONS:
                raise ValueError(
                    f"ScanView.set_display: {name} must be one of "
                    f"{', '.join(MAP_SMOOTHING_OPTIONS)}; got {settings[name]!r}."
                )
        for name in ("map_cmap", "anatomy_cmap"):
            if not is_cmap_available(settings[name]):
                raise ValueError(
                    f"ScanView.set_display: unknown colormap {name}="
                    f"{settings[name]!r}; choose for example one of "
                    f"{', '.join(COMMON_CMAPS)}."
                )

    def _display_settings(self) -> dict[str, Any]:
        """Return the display switches of the state as sent to the browser."""
        spatial = self.state.spatial
        smoothing = spatial.zsmooth_mode if spatial.interpolation else "off"
        settings: dict[str, Any] = {"smoothing": smoothing}
        for name, (section, field) in _DISPLAY_SWITCHES.items():
            settings[name] = bool(getattr(getattr(self.state, section), field))
        settings["map_cmap"] = spatial.cmap
        settings["anatomy_cmap"] = self.state.anatomy.cmap
        settings["overlay_smoothing"] = self.state.overlay.smoothing
        settings["show_slice_lines"] = self._show_slice_lines
        settings["show_masks"] = self._show_masks
        settings["show_mask_outlines"] = self._show_mask_outlines
        return settings

    # ── curve axes ─────────────────────────────────────────────────────

    def set_curve_scale(self, mode: str) -> None:
        """Choose how the y axis of the curve panel is scaled.

        Only the curve panel is redrawn; the shown values do not change.

        Parameters
        ----------
        mode : {"auto", "plane", "map"}
            ``"auto"`` fits the y axis to the curves of the selected voxel
            (inside the frequency window, if one is set). ``"plane"`` uses
            the value range of all voxels of the shown plane, so voxels can
            be compared. ``"map"`` uses the colour limits of the map, as
            "Couple Y-Limits" in ``plot``: for the real or imaginary part
            the axis is symmetric around zero up to the larger absolute
            limit (at most pi for the phase), for the magnitude it starts at
            zero or above. ``"map"`` is stored as
            ``state.spectral.couple_ylims``; ``"plane"`` belongs to this
            view only and is not saved with :meth:`save_settings`.

        Raises
        ------
        ValueError
            If ``mode`` is not one of the three choices.

        Examples
        --------
        >>> view.set_curve_scale("map")  # doctest: +SKIP
        """
        settings = {**self._curve_axes_settings(), "scale": mode}
        _checked_curve_axes(settings, "ScanView.set_curve_scale")
        self._curve_axes = settings

    def set_frequency_window(
        self, window: tuple[float, float] | None, unit: str | None = None
    ) -> None:
        """Show only a frequency range of the spectrum, as "Limit" in ``plot``.

        The window stays when another voxel is selected. With the "auto"
        curve scale the y axis is fitted to the curves inside the window.
        It applies while the spectrum is shown in ``unit``; with the other
        unit or the FID the full axis is shown and the window is kept for
        later. Stored as ``state.spectral.display_range`` and
        ``display_range_unit``; only the curve panel is redrawn.

        Parameters
        ----------
        window : tuple of two floats or None
            Start and end of the shown range in ``unit``; their order does
            not matter. None shows the full axis again.
        unit : {"ppm", "Hz"} or None, optional
            Unit of ``window``; None uses the unit currently shown
            (``state.spectral.spec_unit``).

        Raises
        ------
        ValueError
            If the data have no spectrum, ``unit`` is unknown, or the two
            limits are not finite and different.

        Examples
        --------
        >>> view.set_frequency_window((160.0, 185.0))  # ppm  # doctest: +SKIP
        >>> view.set_frequency_window(None)  # full axis  # doctest: +SKIP
        """
        if not self._spectroscopic:
            raise ValueError(
                "ScanView.set_frequency_window needs data with a spectrum; "
                "this view shows curves along "
                f"{self._curve_dim or 'no dimension'}."
            )
        settings = self._curve_axes_settings()
        settings["window"] = None if window is None else list(window)
        settings["window_unit"] = unit or self.state.spectral.spec_unit
        _checked_curve_axes(settings, "ScanView.set_frequency_window")
        self._curve_axes = settings

    def _curve_axes_settings(self) -> dict[str, Any]:
        """Return the curve-axis settings of the state as sent to the browser."""
        spectral = self.state.spectral
        if spectral.couple_ylims:
            scale = "map"
        else:
            scale = "plane" if self._plane_curve_scale else "auto"
        window = spectral.display_range
        return {
            "scale": scale,
            "window": None if window is None else [float(x) for x in window],
            "window_unit": spectral.display_range_unit,
            "fit": bool(spectral.show_fit),
            "residual": bool(spectral.show_fit_residual),
        }

    def show_fit_curves(
        self, fit: bool | None = None, residual: bool | None = None
    ) -> None:
        """Show or hide the fitted spectrum and its residual in the curve panel.

        The fitted spectrum of the selected voxel comes from the fit overlays
        (the fit tool's result, a stored fit or ``scan.fit_results``). The
        residual is the shown signal minus the summed fit, drawn below the
        curves with a dashed zero line, as in ``plot``; it is not drawn for
        the phase. The choices are ``state.spectral.show_fit`` and
        ``state.spectral.show_fit_residual``; only the curve panel changes.

        Parameters
        ----------
        fit : bool or None, optional
            Draw the fitted spectrum; None keeps the current choice.
        residual : bool or None, optional
            Draw the residual (only together with the fitted spectrum); None
            keeps the current choice.

        Examples
        --------
        >>> view.show_fit_curves(residual=False)  # doctest: +SKIP
        """
        settings = self._curve_axes_settings()
        if fit is not None:
            settings["fit"] = bool(fit)
        if residual is not None:
            settings["residual"] = bool(residual)
        self._curve_axes = settings

    # ── signal processing ──────────────────────────────────────────────

    def set_processing(
        self,
        *,
        show_fid: bool | None = None,
        npoints: int | None = None,
        denoise_components: int | None = None,
        repetitions: str | None = None,
        repetition_range: tuple[int, int] | None = None,
        combine_channels: bool | None = None,
        combine_method: str | None = None,
    ) -> None:
        """Set how the shown signal is processed, as the controls of ``plot``.

        The settings are stored in :attr:`state` (``state.spectral``) and the
        map, curves and time course are computed again by the same
        processing as ``plot`` (``PlotDataModel.get_render_payload``). The
        acquired data are not changed. None keeps a setting. The browser's
        "spectrum" controls set the same values.

        Parameters
        ----------
        show_fid : bool or None, optional
            Show the acquired time signal (FID) instead of the spectrum; the
            curve axis is then the time in ms. Only for spectroscopic data
            with complex spectra; real (magnitude) spectra have no FID.
        npoints : int or None, optional
            Number of spectral points (at least 2). More points than acquired
            add zeros to the end of the FID before the Fourier transform
            (zero filling), which interpolates the spectrum; fewer points use
            only the first part of the FID. Only for spectroscopic data.
        denoise_components : int or None, optional
            Number of principal components kept by the PCA denoising of
            ``plot``; 0 switches denoising off.
        repetitions : {"off", "average", "cumulative sum"} or None, optional
            ``"average"`` shows the mean over ``repetition_range`` (the
            repetition slider disappears); ``"cumulative sum"`` shows at each
            repetition the sum from the first repetition of the range up to
            it. Only for data with more than one repetition.
        repetition_range : tuple of int or None, optional
            Inclusive first and last repetition index used by ``repetitions``.
        combine_channels : bool or None, optional
            Combine the receive channels with ``combine_method`` (the channel
            slider disappears). Only for data with more than one channel.
        combine_method : {"rss", "sum", "mean", "adaptive"} or None, optional
            Channel combination: root sum of squares, sum, mean or adaptive
            combination.

        Raises
        ------
        ValueError
            If a setting does not apply to the data (for example the FID of
            an image or repetitions of data without repetitions) or is out of
            range.

        Examples
        --------
        >>> view.set_processing(npoints=2048, repetitions="average")  # doctest: +SKIP
        >>> view.set_processing(repetition_range=(2, 10))  # doctest: +SKIP
        >>> view.set_processing(show_fid=True)  # doctest: +SKIP
        """
        given = {
            "show_fid": show_fid,
            "npoints": npoints,
            "denoise_components": denoise_components,
            "repetitions": repetitions,
            "repetition_range": repetition_range,
            "combine_channels": combine_channels,
            "combine_method": combine_method,
        }
        settings = {k: value for k, value in given.items() if value is not None}
        before = self._processing_settings()
        self._apply_processing(settings, "ScanView.set_processing")
        if self._processing_settings() == before:
            return
        self._render(reset_map_range=self.state.spatial.range_mode == "Min/Max")

    def _processing_settings(self) -> dict[str, Any]:
        """Return the processing settings that apply to the shown data."""
        spectral = self.state.spectral
        sizes = self._data.sizes
        settings: dict[str, Any] = {
            "denoise_components": int(spectral.denoise_components)
        }
        if self._spectroscopic:
            settings["show_fid"] = bool(spectral.show_fid)
            settings["npoints"] = int(spectral.npoints or sizes[SPECTRAL_DIM])
        if sizes.get("repetition_idx", 1) > 1:
            if spectral.cum_sum_repetitions:
                mode = "cumulative sum"
            elif spectral.average_repetitions:
                mode = "average"
            else:
                mode = "off"
            last = sizes["repetition_idx"] - 1
            end = last if spectral.rep_avg_end is None else spectral.rep_avg_end
            settings["repetitions"] = mode
            settings["repetition_range"] = [
                int(spectral.rep_avg_start),
                int(min(end, last)),
            ]
        if sizes.get("channel_idx", 1) > 1:
            settings["combine_channels"] = bool(spectral.combine_channels)
            settings["combine_method"] = spectral.combine_method
        return settings

    def _apply_processing(self, settings: dict[str, Any], operation: str) -> None:
        """Check processing settings and store them in ``state.spectral``.

        All settings are checked before any is stored, so a rejected setting
        leaves the state unchanged.
        """
        available = self._processing_settings()
        unknown = sorted(set(settings) - set(available))
        if unknown:
            raise ValueError(
                f"{operation}: {', '.join(unknown)} does not apply to this data; "
                f"settings that apply: {', '.join(sorted(available))}."
            )
        merged = {**available, **settings}
        if "npoints" in settings and int(merged["npoints"]) < 2:
            raise ValueError(f"{operation}: npoints must be at least 2.")
        if int(merged["denoise_components"]) < 0:
            raise ValueError(f"{operation}: denoise_components must be 0 or more.")
        if merged.get("repetitions", "off") not in REPETITION_MODES:
            raise ValueError(
                f"{operation}: repetitions must be one of "
                f"{', '.join(REPETITION_MODES)}; got {merged['repetitions']!r}."
            )
        if merged.get("combine_method", "rss") not in CHANNEL_COMBINE_METHODS:
            raise ValueError(
                f"{operation}: combine_method must be one of "
                f"{', '.join(CHANNEL_COMBINE_METHODS)}; "
                f"got {merged['combine_method']!r}."
            )
        if "repetition_range" in merged:
            last = self._data.sizes["repetition_idx"] - 1
            first, end = sorted(int(i) for i in merged["repetition_range"])
            if first < 0 or end > last:
                raise ValueError(
                    f"{operation}: repetition_range must lie within 0 and {last}; "
                    f"got {tuple(merged['repetition_range'])}."
                )
            merged["repetition_range"] = [first, end]
        fid_available = fid_is_recoverable(self._data)
        if settings.get("show_fid") and self._spectroscopic and not fid_available:
            raise ValueError(
                f"{operation}: show_fid=True needs complex spectra, but these "
                "spectra are real (scanner magnitude), so their phase and the "
                "FID are lost. Use the raw reconstruction (scan.data) for the FID."
            )
        spectral = self.state.spectral
        spectral.denoise_components = int(merged["denoise_components"])
        if self._spectroscopic:
            spectral.show_fid = bool(merged["show_fid"]) and fid_available
            n_acquired = self._data.sizes[SPECTRAL_DIM]
            npoints = int(merged["npoints"])
            spectral.npoints = None if npoints == n_acquired else npoints
        if "repetitions" in merged:
            spectral.average_repetitions = merged["repetitions"] == "average"
            spectral.cum_sum_repetitions = merged["repetitions"] == "cumulative sum"
            first, end = merged["repetition_range"]
            # None (the last repetition) stays None, so the state and the
            # overlay layer cache do not change when the range is resent.
            last = self._data.sizes["repetition_idx"] - 1
            spectral.rep_avg_start = first
            spectral.rep_avg_end = None if end == last else end
        if "combine_channels" in merged:
            spectral.combine_channels = bool(merged["combine_channels"])
            spectral.combine_method = merged["combine_method"]

    # ── saved settings ─────────────────────────────────────────────────

    def save_settings(self) -> None:
        """Save the display state on the scan for the next :func:`plot_scan` call.

        A snapshot of :attr:`state` (selected voxel and slice, complex part,
        unit, line broadening, colour ranges, anatomy and overlay settings)
        is stored in memory on the scan under the settings context of this
        view (``"plot_scan"`` unless :func:`plot_scan` was given another
        ``settings_context``). A later :func:`plot_scan` call on the same scan
        without ``state`` starts from it. Nothing is written to disk.

        Raises
        ------
        ValueError
            If the view shows an explicit DataArray instead of a scan, or the
            scan does not accept the snapshot.

        Examples
        --------
        >>> view = plot_scan(scan)  # doctest: +SKIP
        >>> view.set_map_range("Absolute", vmin=0.0, vmax=5e4)  # doctest: +SKIP
        >>> view.save_settings()  # doctest: +SKIP
        >>> plot_scan(scan)  # starts with the saved colour range  # doctest: +SKIP
        """
        if self._scan is None:
            raise ValueError(
                "ScanView.save_settings stores settings on a scan; this view shows "
                "a DataArray. Call plot_scan(scan) to save view settings."
            )
        if not save_plot_state(self._scan, self.state, self._settings_context):
            raise ValueError(
                f"ScanView.save_settings could not store settings on the "
                f"{type(self._scan).__name__} object."
            )
