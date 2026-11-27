"""Saved fits and the fitted curves drawn in the curve panel.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_FitResults`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.visualization.fit_tool import mask_fit_table, voxel_fit_table
from hypermri.visualization.ideal_tool import (
    ideal_maps,
)
from hypermri.visualization.metabolite_split import (
    METABOLITE_SPLIT_DIM,
)
from hypermri.visualization.render_data import (
    FIT_RESIDUAL_OVERLAY,
    FIT_RESIDUAL_ZERO_OVERLAY,
    add_fit_residual_overlay,
)
from hypermri.visualization.saved_fits import (
    SavedFit,
    check_fit_name,
    save_fit,
    saved_fits,
)
from hypermri.visualization.spatial_overlay import (
    normalize_overlays,
    preferred_fit_variable,
)

_import_optional("anywidget")
_import_optional("cmap")

from hypermri.visualization.scan_view.options import (  # noqa: E402
    _SAVE_FIT_TOOLS,
    _TOOL_OVERLAY_NAMES,
    CURVE_FIT_OVERLAY_NAME,
    FIT_OVERLAY_NAME,
    IDEAL_OVERLAY_NAME,
    SPECTRAL_DIM,
    _json_values,
)


def _saved_fit_mismatch(maps: xr.Dataset, data: xr.DataArray) -> bool:
    """Return whether saved fit maps do not belong to the voxels of ``data``.

    Every dimension of the maps that ``data`` also has must have the same
    size or size 1, and the maps must share at least one dimension with
    ``data`` other than ``acquisition_idx``.
    """
    shared = False
    for variable in maps.data_vars.values():
        for dim, size in variable.sizes.items():
            if dim not in data.dims or dim == SPECTRAL_DIM:
                continue
            if size != 1 and data.sizes[dim] != size:
                return True
            shared = True
    return not shared


class _FitResults:
    """Saved fits and the fitted curves drawn in the curve panel of :class:`ScanView`."""

    # ── saved fits ─────────────────────────────────────────────────────

    @property
    def saved_fits(self) -> dict[str, SavedFit]:
        """Fits saved on the plotted object, by name.

        Returns
        -------
        dict of str to SavedFit
            ``scan.saved_fits`` of a plotted scan or ``attrs["saved_fits"]``
            of a plotted DataArray (see
            :func:`~hypermri.visualization.saved_fits.saved_fits`); empty
            before the first :meth:`save_fit`.
        """
        return saved_fits(self._model.data_obj)

    def save_fit(self, name: str = "", tool: str | None = None) -> str:
        """Save the last fit of a fit tool on the plotted object.

        The result of the last run of the spectral fit tool, the curve-fit
        tool or the IDEAL tool is copied into ``saved_fits`` of the plotted
        object (``scan.saved_fits``, or ``data.attrs["saved_fits"]`` of a
        plotted DataArray) under ``name``. The view also keeps its maps as
        the overlay ``name``, which stays when the tool runs again. A view
        opened later on the same object shows the saved fits as overlays.
        Nothing is written to disk.

        Parameters
        ----------
        name : str, optional
            Name of the fit: a letter, then letters, digits, ``_`` or ``-``.
            Empty chooses ``"<kind>_<n>"``, for example ``"spectral_1"`` or
            ``"t2_1"``. A fit saved before under the same name is replaced.
        tool : {"spectral", "curve", "ideal"} or None, optional
            Tool whose fit is saved. None takes the only tool with a fit.

        Returns
        -------
        str
            The name the fit is saved under.

        Raises
        ------
        ValueError
            If the tool has no fit yet, ``tool`` is None while several tools
            have one, the name is not valid, or the name is taken by an
            overlay that is not a saved fit.

        Examples
        --------
        >>> view = plot_scan(scan)  # doctest: +SKIP
        >>> # run the fit tool in the browser, then
        >>> view.save_fit("pyruvate_lactate")  # doctest: +SKIP
        'pyruvate_lactate'
        >>> scan.saved_fits["pyruvate_lactate"].results  # doctest: +SKIP
        """
        fits = self._tool_fits()
        if tool is None:
            if len(fits) != 1:
                raise ValueError(
                    "ScanView.save_fit: "
                    + (
                        "run a fit first."
                        if not fits
                        else f"choose the tool, one of {sorted(fits)}."
                    )
                )
            tool = next(iter(fits))
        if tool not in _SAVE_FIT_TOOLS:
            raise ValueError(
                f"ScanView.save_fit tool must be one of {_SAVE_FIT_TOOLS}; got {tool!r}."
            )
        if tool not in fits:
            raise ValueError(
                f"ScanView.save_fit: the {tool} tool has no fit yet; run it first."
            )
        fit, overlay_name = fits[tool]
        name = (
            check_fit_name(name)
            if str(name).strip()
            else self._default_fit_name(fit.kind)
        )
        if name in _TOOL_OVERLAY_NAMES or (
            name in self._overlays and name not in self.saved_fits
        ):
            raise ValueError(
                f"ScanView.save_fit: the overlay {name!r} already exists; "
                "choose another name."
            )
        save_fit(self._model.data_obj, name, fit)
        self._overlays = {**self._overlays, name: self._overlays[overlay_name]}
        self._overlay_cache = None
        self._render()
        return name

    def _tool_fits(self) -> dict[str, tuple[SavedFit, str]]:
        """Return the last fit of each tool that has one, with its overlay name."""
        fits: dict[str, tuple[SavedFit, str]] = {}
        if self._fit is not None and FIT_OVERLAY_NAME in self._overlays:
            fit_params, fit_results = self._fit
            fits["spectral"] = (
                SavedFit("spectral", fit_results, fit_params),
                FIT_OVERLAY_NAME,
            )
        if self._curve_fit is not None and CURVE_FIT_OVERLAY_NAME in self._overlays:
            model, result = self._curve_fit
            fits["curve"] = (SavedFit(model.key, result), CURVE_FIT_OVERLAY_NAME)
        if self._ideal is not None and IDEAL_OVERLAY_NAME in self._overlays:
            fits["ideal"] = (SavedFit("ideal", self._ideal), IDEAL_OVERLAY_NAME)
        return fits

    def _default_fit_name(self, kind: str) -> str:
        """Return ``"<kind>_<n>"`` with the smallest free number ``n``."""
        taken = set(self.saved_fits) | set(self._overlays)
        number = 1
        while f"{kind}_{number}" in taken:
            number += 1
        return f"{kind}_{number}"

    def _save_fit_from_browser(
        self, content: dict[str, Any], tool: str, state: Any
    ) -> None:
        """Save a tool's fit under the typed name and note where it is."""
        name = self.save_fit(str(content.get("name") or ""), tool)
        where = (
            'attrs["saved_fits"] of the data'
            if isinstance(self._model.data_obj, xr.DataArray | xr.Dataset)
            else "scan.saved_fits"
        )
        state.message = (
            f"Fit saved as {name!r} in {where}; overlay {name!r} keeps its maps."
        )

    def _load_saved_fits(self) -> None:
        """Show the fits saved on the plotted object as overlays.

        A saved fit whose voxel dimensions do not match the shown data, or
        whose name is already an overlay, is left out. When the view had no
        overlay before, the first saved map is chosen but not shown.
        """
        shown: dict[str, xr.Dataset] = {}
        for name, fit in self.saved_fits.items():
            if not isinstance(fit, SavedFit) or name in self._overlays:
                continue
            maps = ideal_maps(fit.results) if fit.kind == "ideal" else fit.results
            if _saved_fit_mismatch(maps, self._data):
                continue
            shown[name] = maps
        if not shown:
            return
        overlays, _ = normalize_overlays(
            shown,
            self.state.spatial.coordinate_mode,
            geometry_reference=self._data,
        )
        first_overlay = not self._overlays
        self._overlays = {**self._overlays, **overlays}
        self._overlay_cache = None
        if first_overlay:
            name, maps = next(iter(shown.items()))
            variable = (
                self.saved_fits[name].kind
                if self.saved_fits[name].kind in maps
                else preferred_fit_variable(maps.data_vars)
            )
            overlay = self.state.overlay
            overlay.active_variable = f"{name}_{variable}"
            overlay.visible = False
            overlay.alpha = 1.0

    # ── Other split metabolite ────────────────────────────────────────

    def _update_other_metabolite(self, *, force: bool = False) -> None:
        """Send the curve of the other split metabolite at the selected voxel.

        With the interleaved metabolites split (two entries of
        ``metabolite_idx``) and ``state.spectral.show_secondary_metabolite_curve``
        on, the curve panel also shows the curve of the metabolite the slider
        does not select, at the same voxel, as ``plot`` does. It comes from one
        render payload with the other ``metabolite_idx``; the state itself is
        not changed. The curve is recomputed only when the selected voxel
        changes, unless ``force``.
        """
        if (
            self._curve_dim is None
            or self._kspace_shown()
            or self._metabolite_split is None
            or self._data is not self._metabolite_split[1]
            or not self.state.spectral.show_secondary_metabolite_curve
        ):
            self._other_voxel = None
            self._other_metabolite = {}
            return
        indices = self.state.spatial.slice_indices
        voxel = tuple(int(indices.get(dim, 0)) for dim in self._spatial_dims)
        if voxel == self._other_voxel and not force:
            return
        self._other_voxel = voxel
        shown = min(max(int(indices.get(METABOLITE_SPLIT_DIM, 0)), 0), 1)
        other = {**indices, METABOLITE_SPLIT_DIM: 1 - shown}
        state = replace(
            self.state, spatial=replace(self.state.spatial, slice_indices=other)
        )
        payload = self._model.get_render_payload(self._data, state)
        if payload.spectral_y is None:
            self._other_metabolite = {}
            return
        selection = self._selection_from_state(n_curve=max(self._curve_x.size, 1))
        self._other_metabolite = {
            "row": selection["row"],
            "column": selection["column"],
            "name": f"{METABOLITE_SPLIT_DIM} {1 - shown}",
            "y": _json_values(np.asarray(payload.spectral_y, dtype=float)),
        }

    # ── Fitted spectrum ────────────────────────────────────────────────

    def _fit_datasets(self) -> dict[str, xr.Dataset]:
        """Return the fit result Datasets whose fitted spectra can be shown.

        These are the Dataset overlays and ``scan.fit_results``, the same
        candidates ``plot`` reconstructs its "Fit" curve from.
        """
        fits = {
            name: value
            for name, value in self._overlays.items()
            if isinstance(value, xr.Dataset)
        }
        own = getattr(self._model.data_obj, "fit_results", None)
        # The overlay "fit" already shows the fit of this view or the stored one.
        fits.pop(CURVE_FIT_OVERLAY_NAME, None)
        fits.pop(IDEAL_OVERLAY_NAME, None)
        if isinstance(own, xr.Dataset) and FIT_OVERLAY_NAME not in fits:
            fits["fit_results"] = own
        return fits

    def _update_fit_curves(self, *, force: bool = False) -> None:
        """Send the fitted spectrum of the selected voxel to the curve panel.

        The curves come from the same render payload as the spectrum of
        ``plot`` (:meth:`PlotDataModel.get_render_payload` with the fit
        Datasets as overlays), so reconstruction, FFT, complex part and
        metabolite mode match it. Without fit results, or when the curve is
        not a spectrum, nothing is sent. The result is recomputed only when
        the selected voxel changes, unless ``force``. The result table of
        the fit tool and the mask statistics follow the selected voxel, too.
        """
        self._update_fit_table()
        self._update_mask_statistics()
        if self._kspace_shown():
            self._fit_voxel = None
            self._fit_curves = {}
            return
        if self._curve_fit is not None and self._curve_dim == self._curve_fit[0].dim:
            self._update_curve_fit_curves(force=force)
            return
        fits = self._fit_datasets() if self._curve_dim == SPECTRAL_DIM else {}
        if not fits or not self.state.spectral.show_fit:
            self._fit_voxel = None
            self._fit_curves = {}
            return
        indices = self.state.spatial.slice_indices
        voxel = tuple(int(indices.get(dim, 0)) for dim in self._spatial_dims)
        if voxel == self._fit_voxel and not force:
            return
        self._fit_voxel = voxel
        x, curves = self._fit_curves_at({}, fits, residual=True)
        for curve in curves:
            curve["y"] = _json_values(curve["y"])
            if "x" in curve:
                curve["x"] = _json_values(curve["x"])
        selection = self._selection_from_state(n_curve=max(self._curve_x.size, 1))
        self._fit_curves = (
            {
                "row": selection["row"],
                "column": selection["column"],
                "x": x.tolist(),
                "curves": curves,
            }
            if curves
            else {}
        )

    def _update_fit_table(self) -> None:
        """Send the fit tool's fitted values of the selected voxel to the browser.

        Empty without a fit of the fit tool. The values come from
        :func:`~hypermri.visualization.fit_tool.voxel_fit_table`.
        """
        if self._fit is None:
            self._fit_table = {}
            return
        self._fit_table = voxel_fit_table(
            self._fit[1], self.state.spatial.slice_indices
        )

    def _update_mask_statistics(self) -> None:
        """Send mean and spread of the fitted values inside each mask.

        The fit is the fit tool's, else ``scan.fit_results``; the values come
        from :func:`~hypermri.visualization.fit_tool.mask_fit_table` at the
        selected position of the dimensions the masks do not cover. Empty
        without masks or without a fit.
        """
        fit_results = self._fit[1] if self._fit is not None else None
        if fit_results is None:
            own = getattr(self._model.data_obj, "fit_results", None)
            fit_results = own if isinstance(own, xr.Dataset) else None
        if (
            fit_results is None
            or "fit_freqs" not in fit_results
            or not self._mask_store
        ):
            self._mask_statistics = {}
            return
        self._mask_statistics = mask_fit_table(
            fit_results, self._mask_store, self.state.spatial.slice_indices
        )

    def _update_curve_fit_curves(self, *, force: bool = False) -> None:
        """Send the curve fit of the selected voxel to the curve panel."""
        if not self.state.spectral.show_fit:
            self._fit_voxel = None
            self._fit_curves = {}
            return
        indices = self.state.spatial.slice_indices
        voxel = tuple(sorted((dim, int(i)) for dim, i in indices.items()))
        if voxel == self._fit_voxel and not force:
            return
        self._fit_voxel = voxel
        curves = self._curve_fit_curves()
        selection = self._selection_from_state(n_curve=max(self._curve_x.size, 1))
        self._fit_curves = (
            {
                "row": selection["row"],
                "column": selection["column"],
                "x": self._curve_x.tolist(),
                "curves": curves,
            }
            if curves
            else {}
        )

    def _fit_curves_at(
        self,
        voxel: dict[str, int],
        fits: dict[str, xr.Dataset],
        *,
        residual: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Return the fitted spectra of one voxel as the voxel curve panel shows them.

        ``voxel`` replaces entries of ``state.spatial.slice_indices`` (empty
        for the selected voxel); the state itself is not changed. The curves
        come from :meth:`PlotDataModel.get_render_payload` with the fit
        Datasets as overlays, as in ``plot``. Each curve is a dict with
        ``name``, ``y`` (float array on the returned axis) and ``color``.
        With ``residual`` and ``state.spectral.show_fit_residual``, the fit
        residual and its zero line follow, shifted below the curves as in
        ``plot`` (:func:`add_fit_residual_overlay`); a curve on its own axis,
        such as the two-point zero line, carries that axis as ``x``.
        """
        state = self.state
        if voxel:
            indices = {**state.spatial.slice_indices, **voxel}
            state = replace(
                state, spatial=replace(state.spatial, slice_indices=indices)
            )
        payload = self._model.get_render_payload(self._data, state, overlays=fits)
        if residual:
            add_fit_residual_overlay(payload, state)
        x = payload.metadata.get("fit_x_axis")
        x = np.asarray(payload.spectral_x if x is None else x, dtype=float)
        colours = payload.metadata.get("spectral_overlay_colors", {})
        labels = payload.metadata.get("spectral_overlay_legend_labels", {})
        curves = []
        for name, y in payload.overlays.items():
            if y is None:
                continue
            values = np.asarray(y, dtype=float).ravel()
            own_x = payload.metadata.get(f"overlay_x_{name}")
            curve = {
                "name": str(
                    labels.get(name, name)
                    if name in (FIT_RESIDUAL_OVERLAY, FIT_RESIDUAL_ZERO_OVERLAY)
                    else name
                ),
                "y": values,
                "color": colours.get(name, "#d62728"),
            }
            if own_x is not None and np.size(own_x) == values.size:
                if values.size != x.size or not np.array_equal(own_x, x):
                    curve["x"] = np.asarray(own_x, dtype=float).ravel()
            elif values.size != x.size:
                continue
            if name == FIT_RESIDUAL_ZERO_OVERLAY:
                curve["dash"] = "dash"
            curves.append(curve)
        return x, curves
