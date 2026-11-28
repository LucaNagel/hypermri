"""Parameter-map overlay: layer, source, slice, opacity and colour range.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_Overlay`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
from typing import Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.visualization.colormap_policy import (
    build_cmap_options,
    is_cmap_available,
)
from hypermri.visualization.data_sources import DataSourceCatalog
from hypermri.visualization.spatial_overlay import (
    SpatialOverlayLayer,
    active_overlay_source_catalog,
    load_overlay_source,
    overlay_value_range,
    split_overlay_variable,
)

_import_optional("anywidget")
_import_optional("cmap")

from hypermri.visualization.scan_view.layers import (  # noqa: E402
    _drawable_overlay_names,
    _overlay_source,
)
from hypermri.visualization.scan_view.options import (  # noqa: E402
    OVERLAY_RANGE_MODES,
    _check_limits,
)
from hypermri.visualization.scan_view.secondary_slices import (  # noqa: E402
    _chosen_slice,
    _follow_map,
    _slice_axis,
)


class _Overlay:
    """Parameter-map overlay: layer, source, slice, opacity and colour range of :class:`ScanView`."""

    # ── overlay ────────────────────────────────────────────────────────

    def set_overlay(
        self,
        variable: str | None = None,
        *,
        metabolite: str | None = None,
        visible: bool | None = None,
        range_mode: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | None = None,
    ) -> None:
        """Change the shown overlay layer, its colours or colour range and redraw.

        Arguments left at None keep their current value in ``state.overlay``.
        Choosing another layer or metabolite, or another range mode, resets
        the colour range to the whole value range of the layer (0-100 % or
        the layer's minimum and maximum). The map is recomputed once.

        Parameters
        ----------
        variable : str or None, optional
            Layer name as listed in the browser, for example
            ``"overlay_fit_amps"`` for the ``fit_amps`` variable of a Dataset
            passed as ``overlay``; see
            :mod:`hypermri.visualization.spatial_overlay`.
        metabolite : str or None, optional
            Metabolite of a layer with a ``metabolite`` dimension.
        visible : bool or None, optional
            Whether the overlay is drawn.
        range_mode : {"Percentage", "Min/Max"} or None, optional
            ``"Percentage"``: vmin and vmax are percentages (0-100) of the
            layer's finite value range. ``"Min/Max"``: vmin and vmax are
            colour limits in overlay units.
        vmin, vmax : float or None, optional
            Lower and upper colour limit in the unit of ``range_mode``.
        cmap : str or None, optional
            Colormap name, for example ``"viridis"`` or ``"lipari"``; the
            browser offers the colormaps of ``plot``.

        Raises
        ------
        ValueError
            If the view has no overlay, the layer, metabolite, range mode or
            colormap is unknown, the layer has spectra (``acquisition_idx``),
            a colour limit is not finite or vmin is not below vmax. The
            overlay settings are then left unchanged.
        """
        if not self._overlays:
            raise ValueError(
                "ScanView.set_overlay: no overlay was given; use plot_scan(..., overlay=...)."
            )
        self._apply_overlay_settings(
            variable=variable,
            metabolite=metabolite,
            visible=visible,
            range_mode=range_mode,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        self._render()
        self._controls = self._current_controls()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set the opacity of the overlay drawn above the map.

        The value is stored in ``state.overlay.alpha`` and kept across
        recomputations. Nothing is recomputed.

        Parameters
        ----------
        opacity : float
            Overlay opacity in [0, 1]; 0 hides it, 1 hides the map under it.

        Raises
        ------
        ValueError
            If ``opacity`` is not a finite number in [0, 1].
        """
        value = float(opacity)
        if not (np.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(
                f"ScanView.set_overlay_opacity needs a value in [0, 1]; got {opacity!r}."
            )
        self._overlay_opacity = value

    def set_overlay_slice(self, index: int | None) -> None:
        """Choose the overlay slice drawn above the map, or let it follow the map.

        By default the overlay plane matches the shown map: the same plane in
        ``"rps"`` coordinates, the nearest one at the same scanner position in
        ``"xyz"`` coordinates. A chosen slice is kept in ``state.overlay``
        (``slicing_dim`` and ``slice_indices``). In ``"xyz"`` coordinates the
        map then moves to its slice nearest to the chosen overlay slice, as in
        ``plot``; a warning is shown when the overlay slice lies outside the
        map's slices. Moving the map afterwards makes the overlay follow it
        again. In ``"rps"`` coordinates the choice stays until ``None`` is
        set. The plane is recomputed once.

        Parameters
        ----------
        index : int or None
            Zero-based slice of the overlay along the dimension normal to the
            shown plane (``state`` and the browser name it, for example
            ``"slice_idx"``); None lets the overlay follow the map.

        Raises
        ------
        ValueError
            If the view has no overlay, the overlay has a single slice along
            that dimension, or ``index`` is outside ``[0, number of slices)``.
            The state is then unchanged.

        Examples
        --------
        >>> view = plot_scan(scan, overlay=fit_result)  # doctest: +SKIP
        >>> view.set_overlay_slice(2)  # doctest: +SKIP
        >>> view.set_overlay_slice(None)  # follow the map again  # doctest: +SKIP
        """
        if not self._overlays:
            raise ValueError(
                "ScanView.set_overlay_slice: no overlay was given; use "
                "plot_scan(..., overlay=...)."
            )
        self._change_overlay_slice(index)
        self._render()
        self._controls = self._current_controls()

    def set_overlay_source(self, key: str) -> None:
        """Show the overlay scan from another of its reconstructions.

        Available when ``overlay`` was given as a scan object that offers
        several data sources: ``"data"`` (reconstructed in hypermri),
        ``"scanner_data"`` (scanner reconstruction) and ``"dicom"`` (exported
        DICOM series, read on first use). The key is stored in
        ``state.overlay.data_source``. If the shown layer does not exist in
        the new source, its first drawable layer is shown. The colour range
        is reset to the whole value range and the overlay slice follows the
        map again. The plane is recomputed once.

        Parameters
        ----------
        key : str
            Source key as offered in the browser, see above.

        Raises
        ------
        ValueError
            If no overlay scan offers several sources, the key is unknown,
            the source cannot be loaded or has no layer without spectra. The
            overlay is then unchanged.

        Examples
        --------
        >>> view = plot_scan(scan, overlay=other_scan)  # doctest: +SKIP
        >>> view.set_overlay_source("dicom")  # doctest: +SKIP
        """
        self._change_overlay_source(key)
        self._render()
        self._controls = self._current_controls()

    def _active_overlay_sources(self) -> tuple[str | None, DataSourceCatalog | None]:
        """Return the scan overlay whose source the source control changes."""
        return active_overlay_source_catalog(
            self._overlays, self._overlay_sources, self.state.overlay.active_variable
        )

    def _overlay_slice_axis(self) -> tuple[str | None, int]:
        """Return the overlay dimension normal to the shown plane and its size."""
        layer = self._overlay_layer(require_visible=False)
        data = None if layer is None else layer.data
        return _slice_axis(self.state.overlay, data, self._overlay_plane)

    def _change_overlay_slice(self, index: Any) -> None:
        """Store a chosen overlay slice and, in xyz, move the map to it."""
        layer = self._overlay_layer(require_visible=False)
        self._change_secondary_slice(
            self.state.overlay,
            None if layer is None else layer.data,
            self._overlay_plane,
            index,
            label="overlay",
        )

    def _change_overlay_source(self, key: str) -> None:
        """Load another source of the overlay scan and choose a drawable layer."""
        name, catalog = self._active_overlay_sources()
        if catalog is None or not catalog.has_canonical_sources:
            raise ValueError(
                "ScanView overlay source: no overlay scan offers several data "
                "sources; pass the overlay as a scan to plot_scan."
            )
        if key not in catalog.keys:
            raise ValueError(
                f"ScanView overlay source: unknown source {key!r}; choose one of "
                f"{list(catalog.keys)}."
            )
        previous_key = catalog.current_key
        try:
            selected = load_overlay_source(
                catalog,
                key,
                self.state.spatial.coordinate_mode,
                geometry_reference=self._data,
            )
        except KeyError as error:
            raise ValueError(
                f"ScanView overlay source: cannot load source {key!r}: {error}"
            ) from error
        overlays = {**self._overlays, name: selected}
        drawable = _drawable_overlay_names(overlays)
        own_layers = [
            layer
            for layer in drawable
            if split_overlay_variable(overlays, layer)[0] == name
        ]
        if not own_layers:
            catalog.current_key = previous_key
            raise ValueError(
                f"ScanView overlay source {key!r} of {name!r} has no map without "
                f"acquisition_idx to draw."
            )
        self._overlays = overlays
        self._overlay_cache = None
        overlay = self.state.overlay
        overlay.data_source = key
        if overlay.active_variable not in drawable:
            overlay.active_variable = own_layers[0]
        _follow_map(self.state.overlay)
        overlay.vmin, overlay.vmax = self._full_overlay_range()

    def _overlay_names(self) -> list[str]:
        """Return the layers this view can draw."""
        return _drawable_overlay_names(self._overlays)

    def _overlay_source(self, variable: str) -> xr.DataArray:
        """Return the registered samples of one layer before any reduction."""
        return _overlay_source(self._overlays, variable)

    def _overlay_metabolites(self, variable: str | None) -> list[str]:
        """Return the metabolites of a layer, or an empty list."""
        if not variable:
            return []
        data = self._overlay_source(variable)
        if "metabolite" not in data.dims:
            return []
        return [str(value) for value in data["metabolite"].values]

    def _overlay_options(self) -> dict[str, Any]:
        """Return the choices offered by the browser overlay controls."""
        if not self._overlays:
            return {}
        slice_dim, slice_count = self._overlay_slice_axis()
        _, catalog = self._active_overlay_sources()
        return {
            "variables": _layer_labels(self._overlays, self._overlay_names()),
            "metabolites": self._overlay_metabolites(
                self.state.overlay.active_variable
            ),
            "range_modes": list(OVERLAY_RANGE_MODES),
            "value_range": list(self._full_overlay_range()),
            "cmaps": self._cmap_choices(self.state.overlay.cmap),
            "slice_dim": slice_dim,
            "slice_count": slice_count,
            "data_sources": (
                [[label, key] for label, key in catalog.options()]
                if catalog is not None and catalog.has_canonical_sources
                else []
            ),
        }

    def _cmap_names(self) -> list[tuple[str, str]]:
        """Return the (label, name) pairs of the colormaps that load.

        Built on first use, since checking each colormap loads it.
        """
        if self._cmaps is None:
            self._cmaps = build_cmap_options()
        return self._cmaps

    def _cmap_choices(self, current: str) -> list[list[str]]:
        """Return (label, name) colormap pairs, including ``current``."""
        choices = [[label, name] for label, name in self._cmap_names()]
        if current not in (name for _, name in choices):
            choices.append([current, current])
        return choices

    def _overlay_controls(self) -> dict[str, Any] | None:
        """Return the overlay settings shown as browser controls."""
        if not self._overlays:
            return None
        overlay = self.state.overlay
        return {
            "variable": overlay.active_variable,
            "metabolite": self._shown_metabolite(),
            "visible": bool(overlay.visible),
            "range_mode": overlay.range_mode,
            "vmin": float(overlay.vmin),
            "vmax": float(overlay.vmax),
            "cmap": overlay.cmap,
            "slice": _chosen_slice(self.state.overlay),
            "data_source": overlay.data_source,
        }

    def _shown_metabolite(self) -> str | None:
        """Return the metabolite drawn, as chosen by ``reduce_spatial_overlay``."""
        metabolites = self._overlay_metabolites(self.state.overlay.active_variable)
        if not metabolites:
            return None
        metabolite = self.state.overlay.active_metabolite
        return metabolite if metabolite in metabolites else metabolites[0]

    def _check_overlay_settings(self) -> None:
        """Fail early on overlay state this view cannot draw."""
        if not self._overlays:
            return
        overlay = self.state.overlay
        if not self._overlay_names():
            raise ValueError(
                "plot_scan overlays must be maps without acquisition_idx (or "
                "with one acquisition point), such as fit amplitudes or a field "
                "map; use hypermri.plot to overlay spectra."
            )
        if overlay.mode != "normal":
            raise ValueError(
                f"plot_scan draws overlays in mode 'normal' only; state.overlay.mode "
                f"is {overlay.mode!r}. Use hypermri.plot for difference overlays."
            )
        if overlay.range_mode not in OVERLAY_RANGE_MODES:
            raise ValueError(
                f"plot_scan overlay range_mode must be one of {OVERLAY_RANGE_MODES}; "
                f"got {overlay.range_mode!r}."
            )
        if overlay.active_variable not in self._overlay_names():
            raise ValueError(
                f"plot_scan cannot draw overlay layer {overlay.active_variable!r}; "
                f"choose one of {self._overlay_names()} with ScanView.set_overlay."
            )

    def _apply_overlay_settings(
        self,
        *,
        variable: str | None,
        metabolite: str | None,
        visible: bool | None,
        range_mode: str | None,
        vmin: float | None,
        vmax: float | None,
        cmap: str | None = None,
    ) -> None:
        """Validate overlay settings and store them in ``state.overlay``.

        Invalid settings raise ValueError and leave ``state.overlay``
        unchanged.
        """
        overlay = self.state.overlay
        previous = replace(overlay)
        try:
            self._change_overlay_settings(
                variable=variable,
                metabolite=metabolite,
                visible=visible,
                range_mode=range_mode,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        except ValueError:
            for field in fields(overlay):
                setattr(overlay, field.name, getattr(previous, field.name))
            raise

    def _change_overlay_settings(
        self,
        *,
        variable: str | None,
        metabolite: str | None,
        visible: bool | None,
        range_mode: str | None,
        vmin: float | None,
        vmax: float | None,
        cmap: str | None,
    ) -> None:
        """Store overlay settings in ``state.overlay``, failing on invalid ones."""
        overlay = self.state.overlay
        layer_changed = False
        if variable is not None and variable != overlay.active_variable:
            if variable not in self._overlay_names():
                raise ValueError(
                    f"ScanView overlay: cannot draw layer {variable!r}; choose one "
                    f"of {self._overlay_names()}."
                )
            overlay.active_variable = variable
            layer_changed = True
        if metabolite is not None and metabolite != self._shown_metabolite():
            metabolites = self._overlay_metabolites(overlay.active_variable)
            if metabolite not in metabolites:
                raise ValueError(
                    f"ScanView overlay: layer {overlay.active_variable!r} has no "
                    f"metabolite {metabolite!r}; choose one of {metabolites}."
                )
            overlay.active_metabolite = metabolite
            layer_changed = True
        if visible is not None:
            overlay.visible = bool(visible)
        if cmap is not None and cmap != overlay.cmap:
            if not is_cmap_available(cmap):
                names = [name for _, name in self._cmap_names()]
                raise ValueError(
                    f"ScanView overlay: unknown colormap {cmap!r}; choose for "
                    f"example one of {names}."
                )
            overlay.cmap = cmap
        if range_mode is not None and range_mode != overlay.range_mode:
            if range_mode not in OVERLAY_RANGE_MODES:
                raise ValueError(
                    f"ScanView overlay range_mode must be one of "
                    f"{OVERLAY_RANGE_MODES}; got {range_mode!r}."
                )
            overlay.range_mode = range_mode
            layer_changed = True
        if layer_changed:
            overlay.vmin, overlay.vmax = self._full_overlay_range()
        if vmin is not None or vmax is not None:
            low = overlay.vmin if vmin is None else vmin
            high = overlay.vmax if vmax is None else vmax
            _check_limits("ScanView overlay", low, high)
            overlay.vmin, overlay.vmax = float(low), float(high)

    def _full_overlay_range(self) -> tuple[float, float]:
        """Return colour limits covering the whole layer in the current range mode."""
        if self.state.overlay.range_mode == "Percentage":
            return 0.0, 100.0
        layer = self._overlay_layer(require_visible=False)
        if layer is None or layer.value_range is None:
            return 0.0, 1.0
        return layer.value_range

    def _overlay_layer(
        self, *, require_visible: bool = True
    ) -> SpatialOverlayLayer | None:
        """Return the selected layer reduced and preprocessed as in ``plot``.

        The layer and its whole value range are cached until the layer,
        metabolite or repetition selection changes, so slice changes only
        cut a new plane.
        """
        overlay = self.state.overlay
        variable = overlay.active_variable
        if not (self._overlays and variable):
            return None
        if require_visible and not overlay.visible:
            return None
        spectral = self.state.spectral
        key = (
            variable,
            overlay.active_metabolite,
            bool(spectral.average_repetitions),
            bool(spectral.cum_sum_repetitions),
            spectral.rep_avg_start,
            spectral.rep_avg_end,
        )
        if self._overlay_cache is not None and self._overlay_cache[0] == key:
            return self._overlay_cache[1]
        reduced = self._model.reduce_spatial_overlay(
            self._overlay_source(variable), self.state
        )
        data = self._model.preprocess_spatial_overlay(reduced, self.state)
        layer = SpatialOverlayLayer(
            name=variable, data=data, value_range=overlay_value_range(data)
        )
        self._overlay_cache = (key, layer)
        return layer


def _layer_labels(overlays: dict[str, Any], names: list[str]) -> list[list[str]]:
    """Return ``[label, name]`` pairs of the overlay layers for the browser menu.

    The label of a Dataset layer is its variable (``fit_amps`` for the layer
    ``fit_fit_amps``), since the overlay section already says where it comes
    from. Labels that would be equal for two layers keep the full name.
    """
    short = {name: split_overlay_variable(overlays, name)[1] or name for name in names}
    counts = Counter(short.values())
    return [[short[name] if counts[short[name]] == 1 else name, name] for name in names]
