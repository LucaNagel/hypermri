"""Masks drawn on the map, their undo history and the ROI mean curve.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_Masks`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.visualization.mask_editing import (
    DEFAULT_MASK_ALPHA,
    DEFAULT_MASK_COLOR,
    MaskedMeanCurve,
    combine_plane_coverage,
    empty_mask,
    mask_dims,
    masked_mean_curve,
    masks_from_dataset,
    masks_to_dataset,
    outline_to_shown,
    outline_to_source,
    plane_masked_mean,
    polygon_coverage,
    shown_coverage_to_source,
    source_coverage_to_shown,
    update_mask_measurements,
)

_import_optional("anywidget")
_import_optional("cmap")

from hypermri.visualization.scan_view.options import (  # noqa: E402
    SPECTRAL_DIM,
    _check_mask_style,
    _json_values,
)


class _Masks:
    """Masks drawn on the map, their undo history and the ROI mean curve of :class:`ScanView`."""

    # ── masks ──────────────────────────────────────────────────────────

    @property
    def masks(self) -> xr.Dataset:
        """Return a copy of all masks with their display style.

        Returns
        -------
        xarray.Dataset
            One float32 variable per mask with the coverage of each voxel in
            [0, 1] over ``read_idx``, ``phase_idx`` and ``slice_idx`` (as far
            as present), and attrs ``color``, ``alpha``,
            ``weighted_voxel_count``, when the voxel spacing is known,
            ``volume_ml``, and for masks with drawn outlines ``outlines``
            (a JSON string, see :attr:`mask_outlines`).
        """
        return masks_to_dataset(
            self._mask_store,
            self._mask_colors,
            self._mask_alphas,
            data=self._data,
            scan=self._scan,
            outlines=self._mask_outlines,
        )

    @property
    def mask_outlines(self) -> dict[str, list[dict[str, Any]]]:
        """Return a copy of the outlines drawn into each mask.

        The coverage of a mask keeps only the fraction of each voxel inside
        the drawn outlines; these outlines keep the drawn shape at full
        precision and are drawn as lines over the map.

        Returns
        -------
        dict of str to list of dict
            Outlines by mask name, oldest first, each with ``dims`` (two
            spatial dimensions), ``points`` (vertex positions in voxel units
            of the data grid, in the order of ``dims``; voxel ``k`` spans
            ``[k, k + 1]``), ``slicing_dim``, ``slice_index`` and ``erase``.
            Masks without drawn outlines map to an empty list.

        Notes
        -----
        Outlines are kept for outlines drawn with :meth:`draw_polygon` on a
        plane whose shown rows and columns follow the data grid in steps of
        one voxel; masks copied from linked views get no outline.
        """
        return {
            name: [dict(outline) for outline in self._mask_outlines.get(name, [])]
            for name in self._mask_store
        }

    def add_mask(
        self,
        name: str,
        *,
        color: str = DEFAULT_MASK_COLOR,
        alpha: float = DEFAULT_MASK_ALPHA,
    ) -> None:
        """Create an empty mask, also in views linked with :meth:`link_masks`.

        Parameters
        ----------
        name : str
            New mask name; not empty and not used by another mask.
        color : str, optional
            Display colour as ``#rrggbb``.
        alpha : float, optional
            Display opacity in [0, 1].

        Raises
        ------
        ValueError
            If the name is empty or taken, or the style is invalid.
        """
        name = str(name).strip()
        if not name:
            raise ValueError("ScanView.add_mask needs a non-empty mask name.")
        if name in self._mask_store:
            raise ValueError(f"ScanView.add_mask: a mask named {name!r} exists.")
        _check_mask_style(color, alpha)
        self._mask_store[name] = empty_mask(self._data)
        self._mask_colors[name] = color
        self._mask_alphas[name] = float(alpha)
        self._masks_changed()
        self._share_mask(name, None)

    def remove_mask(self, name: str) -> None:
        """Delete a mask and its undo steps, also in linked views.

        Parameters
        ----------
        name : str
            Existing mask name.

        Raises
        ------
        ValueError
            If no mask has this name.
        """
        self._require_mask(name, "remove_mask")
        self._drop_mask(name)
        self._masks_changed()
        self._share_mask(name, None)

    def _drop_mask(self, name: str) -> None:
        """Forget a mask, its style and its undo steps."""
        del self._mask_store[name]
        self._mask_colors.pop(name, None)
        self._mask_alphas.pop(name, None)
        self._mask_outlines.pop(name, None)
        self._mask_undo = [step for step in self._mask_undo if step[0] != name]
        if self._roi_mask == name:
            self._roi_mask = None

    def set_mask_style(
        self, name: str, *, color: str | None = None, alpha: float | None = None
    ) -> None:
        """Change the display colour or opacity of a mask, also in linked views.

        Parameters
        ----------
        name : str
            Existing mask name.
        color : str or None, optional
            New colour as ``#rrggbb``; None keeps the current one.
        alpha : float or None, optional
            New opacity in [0, 1]; None keeps the current one.

        Raises
        ------
        ValueError
            If no mask has this name or the style is invalid.
        """
        self._require_mask(name, "set_mask_style")
        color = self._mask_colors[name] if color is None else color
        alpha = self._mask_alphas[name] if alpha is None else alpha
        _check_mask_style(color, alpha)
        self._mask_colors[name] = color
        self._mask_alphas[name] = float(alpha)
        self._masks_changed()
        self._share_mask(name, None)

    def draw_polygon(
        self,
        vertices: Sequence[tuple[float, float]],
        *,
        mask: str,
        erase: bool = False,
    ) -> None:
        """Add an outline drawn on the shown map to a mask, or remove it.

        Parameters
        ----------
        vertices : sequence of (float, float)
            Outline points as (x, y) in the units of the map axes. Three or
            more points form a closed polygon whose overlap with each voxel
            gives its coverage; two points give a one-voxel-wide line.
        mask : str
            Name of an existing mask.
        erase : bool, optional
            False adds the area (voxel-wise maximum of old and new coverage);
            True subtracts it.

        Raises
        ------
        ValueError
            If the shown plane cannot hold masks, the mask does not exist, a
            point is not finite, or the outline covers no voxel.

        Notes
        -----
        Only the shown plane (the current slice of the third spatial
        dimension) changes. The previous plane is kept for
        :meth:`undo_mask_edit`. The outline itself is kept in
        :attr:`mask_outlines` and drawn over the map. Views linked with :meth:`link_masks` get the
        changed plane copied onto their grid.
        """
        self._require_mask(mask, "draw_polygon")
        plane = self._mask_plane
        if plane.get("unavailable"):
            raise ValueError(f"ScanView.draw_polygon: {plane['unavailable']}")
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("ScanView.draw_polygon needs at least two (x, y) points.")
        if not np.all(np.isfinite(points)):
            raise ValueError("ScanView.draw_polygon: all points must be finite.")
        coverage = polygon_coverage(
            [tuple(point) for point in points], plane["extent"], plane["shape"]
        )
        if float(coverage.sum()) <= 0:
            raise ValueError("ScanView.draw_polygon: the outline covers no voxel.")
        stored = self._mask_store[mask]
        slicing_dim = plane["slicing_dim"]
        target_dims = [dim for dim in stored.dims if dim != slicing_dim]
        drawn = shown_coverage_to_source(
            coverage,
            [plane["row_dim"], plane["column_dim"]],
            plane["source_indices"],
            self._data.sizes,
            target_dims,
        )
        indexer = tuple(
            plane["slice_index"] if dim == slicing_dim else slice(None)
            for dim in stored.dims
        )
        previous = np.array(stored.values[indexer], dtype=np.float32)
        stored.values[indexer] = combine_plane_coverage(previous, drawn, erase=erase)
        outline = outline_to_source(points, plane, erase=erase)
        if outline is not None:
            self._mask_outlines.setdefault(mask, []).append(outline)
        self._mask_undo.append((mask, indexer, previous, outline))
        self._masks_changed()
        self._share_mask(mask, self._mask_region(mask, indexer))

    def undo_mask_edit(self) -> bool:
        """Restore the plane changed by the last outline and forget the outline.

        Views linked with :meth:`link_masks` get the restored plane copied
        onto their grid; their own undo steps are kept.

        Returns
        -------
        bool
            True if an edit was undone, False if there was none.
        """
        if not self._mask_undo:
            return False
        name, indexer, previous, outline = self._mask_undo.pop()
        self._mask_store[name].values[indexer] = previous
        if outline is not None:
            self._mask_outlines[name].remove(outline)
        self._masks_changed()
        self._share_mask(name, self._mask_region(name, indexer))
        return True

    def _mask_region(self, name: str, indexer: tuple[Any, ...]) -> np.ndarray:
        """Return a boolean array of the mask voxels selected by ``indexer``."""
        region = np.zeros(self._mask_store[name].shape, dtype=bool)
        region[indexer] = True
        return region

    def _require_mask(self, name: str, operation: str) -> None:
        """Raise a ValueError naming the operation for an unknown mask."""
        if name not in self._mask_store:
            raise ValueError(
                f"ScanView.{operation}: no mask named {name!r}; existing masks: "
                f"{sorted(self._mask_store)}."
            )

    def _load_stored_masks(self) -> None:
        """Take over masks already stored on the scan, checking their grid."""
        stored = getattr(self._scan, "masks", None)
        if not isinstance(stored, xr.Dataset):
            return
        masks, colors, alphas, outlines = masks_from_dataset(stored)
        reference = empty_mask(self._data)
        for name, mask in masks.items():
            if mask.dims != reference.dims or mask.shape != reference.shape:
                raise ValueError(
                    f"plot_scan: stored mask {name!r} has dimensions "
                    f"{dict(mask.sizes)}, the data {dict(reference.sizes)}; "
                    "remove it from scan.masks or use matching data."
                )
        self._mask_store.update(masks)
        self._mask_colors.update(colors)
        self._mask_alphas.update(alphas)
        self._mask_outlines.update(outlines)

    def _masks_changed(self) -> None:
        """Refresh measurements, store the masks on the scan and send them."""
        for mask in self._mask_store.values():
            update_mask_measurements(mask, self._data, self._scan)
        if self._scan is not None:
            self._scan.masks = self.masks
        self._mask_planes = self._displayed_mask_planes()
        self._update_roi_mean()
        self._update_mask_statistics()
        self._send_time_course()

    # ── ROI mean ───────────────────────────────────────────────────────

    @property
    def roi_mean(self) -> MaskedMeanCurve | None:
        """Return the ROI mean curve shown in the spectrum panel, if any.

        Returns
        -------
        MaskedMeanCurve or None
            Mean magnitude inside the mask chosen with :meth:`show_roi_mean`
            along the curve dimension, or None when the voxel curve is
            shown.
        """
        return self._roi_curve

    def show_roi_mean(self, mask: str | None) -> None:
        """Show means inside a mask instead of the voxel curve.

        The panel shows up to three kinds of curves:

        * the mean of the shown curves (for spectra: after FFT, line
          broadening, in the chosen complex part and unit) over the mask
          voxels of the shown plane, each voxel weighted by its coverage
          (:func:`~hypermri.visualization.mask_editing.plane_masked_mean`).
          For the magnitude part this is the mean of magnitudes, not the
          magnitude of the mean spectrum;
        * with fit results (as for the voxel curve) and
          ``state.spectral.show_fit``, the mean of the fitted spectra of the
          same voxels with the same weights, named ``"<curve> (mean)"``.
          Every covered voxel costs one render payload (about 10 ms each);
        * the "ROI mean magnitude" of ``plot`` on its own axes (top and
          right).

        The last curve is the coverage-
        weighted mean of the magnitude of the supplied samples over the whole
        mask volume (all slices), along the curve dimension (normally
        ``acquisition_idx``), at the current repetition, channel and other
        selected indices of ``state.spatial.slice_indices``, as in ``plot``. It uses the supplied
        domain without FFT or line broadening, so for FID data it is the mean
        FID magnitude over the acquisition axis. It follows later mask edits
        and index changes.

        Parameters
        ----------
        mask : str or None
            Name of an existing mask, or None to show the voxel spectrum
            again.

        Raises
        ------
        ValueError
            If no mask has this name, or if an image is shown without a
            curve panel.
        """
        if mask is not None:
            self._require_mask(mask, "show_roi_mean")
            if self._curve_dim is None:
                raise ValueError(
                    "ScanView.show_roi_mean needs a curve panel; this image has "
                    "no dimension to draw a curve along. The mask size is in "
                    "ScanView.masks."
                )
        self._roi_mask = mask
        self._update_roi_mean()

    def _update_roi_mean(self) -> None:
        """Recompute the ROI mean curve for the chosen mask and send it.

        Nothing is shown while the map shows k-space; the chosen mask stays.
        """
        if self._roi_mask is not None and self._roi_mask not in self._mask_store:
            self._roi_mask = None
        if self._roi_mask is None or self._kspace_shown():
            self._roi_curve = None
            self._roi_mean = {}
            return
        spatial = set(mask_dims(self._data))
        selectors = {
            dim: int(self.state.spatial.slice_indices.get(dim, 0))
            for dim in self._data.dims
            if dim not in spatial and dim != self._curve_dim
        }
        curve = masked_mean_curve(
            self._data,
            self._mask_store[self._roi_mask],
            secondary_dim=self._curve_dim,
            selectors=selectors,
        )
        self._roi_curve = curve
        roi_mean: dict[str, Any] = {
            "mask": self._roi_mask,
            "x": [float(value) for value in curve.x],
            "y": _json_values(curve.y),
            "x_label": curve.x_label,
            "y_label": curve.y_label,
            "invert_x": curve.invert_x,
        }
        coverage = self._shown_coverage(self._mask_store[self._roi_mask])
        if coverage is not None and np.any(coverage > 0):
            roi_mean["shown"] = _json_values(
                plane_masked_mean(self._plane_curves, coverage)
            )
            fit_x, fits = self._roi_fit_means(coverage)
            if fits:
                roi_mean["fit_x"] = fit_x
                roi_mean["fits"] = fits
        self._roi_mean = roi_mean

    def _roi_fit_means(
        self, coverage: np.ndarray
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Return the coverage-weighted mean fitted spectra of the shown plane.

        Every voxel of the shown plane that the mask covers gets its fitted
        spectra as in the voxel curve panel (:meth:`_fit_curves_at`); their
        mean uses the same weights as the mean of the measured spectra
        (:func:`~hypermri.visualization.mask_editing.plane_masked_mean`). Each
        voxel costs one render payload, about 10 ms for 350 spectral points.
        """
        if self._curve_dim != SPECTRAL_DIM or not self.state.spectral.show_fit:
            return [], []
        fits = self._fit_datasets()
        if not fits:
            return [], []
        row_dim, column_dim = self._spatial_dims
        rows, columns = np.nonzero(coverage > 0)
        stacks: dict[str, np.ndarray] = {}
        colours: dict[str, str] = {}
        x: np.ndarray | None = None
        for row, column in zip(rows, columns, strict=True):
            voxel = {
                row_dim: int(self._block_indices[row_dim][row]),
                column_dim: int(self._block_indices[column_dim][column]),
            }
            voxel_x, curves = self._fit_curves_at(voxel, fits)
            if x is None:
                x = voxel_x
            for curve in curves:
                if "x" in curve:
                    continue  # a curve on its own axis has no voxel mean here
                name = curve["name"]
                if name not in stacks:
                    stacks[name] = np.full((*coverage.shape, x.size), np.nan)
                    colours[name] = curve["color"]
                stacks[name][row, column] = curve["y"]
        if x is None:
            return [], []
        means = [
            {
                "name": f"{name} (mean)",
                "y": _json_values(plane_masked_mean(stack, coverage)),
                "color": colours[name],
            }
            for name, stack in stacks.items()
        ]
        return x.tolist(), means

    def _apply_browser_mask_action(self, content: dict[str, Any]) -> None:
        """Dispatch one mask message from the browser to the public methods."""
        action = content.get("action")
        name = str(content.get("mask", ""))
        if action == "add":
            self.add_mask(
                name,
                color=content.get("color", DEFAULT_MASK_COLOR),
                alpha=float(content.get("alpha", DEFAULT_MASK_ALPHA)),
            )
        elif action == "remove":
            self.remove_mask(name)
        elif action == "style":
            self.set_mask_style(
                name, color=content.get("color"), alpha=content.get("alpha")
            )
        elif action == "draw":
            self.draw_polygon(
                content.get("vertices", []),
                mask=name,
                erase=bool(content.get("erase", False)),
            )
        elif action == "undo":
            self.undo_mask_edit()
        else:
            raise ValueError(f"ScanView: unknown mask action {action!r}.")

    def _displayed_mask_planes(self) -> list[dict[str, Any]]:
        """Return each mask's style, measurements, shown-plane coverage and outlines."""
        result = []
        for name, mask in self._mask_store.items():
            entry = {
                "name": name,
                "color": self._mask_colors.get(name, DEFAULT_MASK_COLOR),
                "alpha": self._mask_alphas.get(name, DEFAULT_MASK_ALPHA),
                "weighted_voxel_count": mask.attrs.get("weighted_voxel_count", 0.0),
                "volume_ml": mask.attrs.get("volume_ml"),
                "plane": None,
                "outlines": [],
            }
            shown = self._shown_coverage(mask)
            if shown is not None:
                entry["plane"] = np.round(shown.astype(float), 4).tolist()
                entry["outlines"] = [
                    drawn
                    for outline in self._mask_outlines.get(name, [])
                    if (drawn := outline_to_shown(outline, self._mask_plane))
                ]
            result.append(entry)
        return result

    def _shown_coverage(self, mask: xr.DataArray) -> np.ndarray | None:
        """Return a mask's coverage of the shown plane, or None without a plane."""
        plane = self._mask_plane
        if not plane or plane.get("unavailable"):
            return None
        slicing_dim = plane["slicing_dim"]
        section = (
            mask.isel({slicing_dim: plane["slice_index"]}) if slicing_dim else mask
        )
        return source_coverage_to_shown(
            section,
            [plane["row_dim"], plane["column_dim"]],
            plane["source_indices"],
        )
