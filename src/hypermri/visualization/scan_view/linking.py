"""Slice lines, shared pointer, frames and masks between several views.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_Linking`` use the traits and state that ``ScanView`` sets up, so the
class is only meant as a base of ``ScanView``.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.utils.coordinates.grids import assign_xyz_coordinates, has_xyz_coordinates
from hypermri.visualization.geometry import (
    _spatial_dim_resolution_mm,
    view_plane_for_anatomy,
)
from hypermri.visualization.mask_editing import (
    empty_mask,
    mask_dims,
    transfer_coverage,
)
from hypermri.visualization.slice_lines import (
    PlaneGrid,
    VoxelGrid,
    axes_to_lps,
    nearest_voxel,
    plane_grid,
    slice_line,
    voxel_grid,
)
from hypermri.visualization.view_dimensions import (
    view_plane_choices,
)

_import_optional("anywidget", "visualization")
_import_optional("cmap", "visualization")

from hypermri.visualization.scan_view.options import (  # noqa: E402
    SLICE_LINE_COLOURS,
    SPATIAL_DIMS,
    SPECTRAL_DIM,
)

if TYPE_CHECKING:
    from hypermri.visualization.scan_view.widget import ScanView


def _is_view(obj: object) -> bool:
    """Return whether ``obj`` is a :class:`ScanView` (imported late: no cycle)."""
    from hypermri.visualization.scan_view.widget import ScanView

    return isinstance(obj, ScanView)


class _Linking:
    """Slice lines, shared pointer, frames and masks between several views of :class:`ScanView`."""

    # ── slice lines of other views ─────────────────────────────────────

    def show_slices_of(
        self, *views: ScanView, colours: Sequence[str] | None = None
    ) -> None:
        """Draw where the shown slices of other views cross this image.

        Each other view's current slice is a thin box in scanner
        coordinates (its thickness is the spacing of its slices, or its
        ``slice_thickness_mm`` for a single slice). Where the box crosses the
        plane shown here, a semi-transparent band in the view's colour is
        drawn on the map, like the slice position lines of a scanner console.
        The band follows the other view whenever it shows a new slice. Views
        whose plane is parallel to this one, or whose slice misses this
        image, draw nothing. Calling the method again replaces the previous
        views; no views remove the lines.

        Parameters
        ----------
        *views : ScanView
            Views of images of the same session (same patient coordinate
            system) whose shown slice is drawn here.
        colours : sequence of str or None, optional
            One colour per view as ``"#rrggbb"``. None uses a fixed palette.

        Raises
        ------
        ValueError
            If a view is this view, the number of colours does not match, a
            colour is not ``"#rrggbb"``, or
            this view or another one has no scanner coordinates ``x``, ``y``,
            ``z`` over its displayed dimensions.

        Notes
        -----
        Masks are not shared: a mask drawn on one view belongs to that
        view's scan and grid.

        Examples
        --------
        >>> axial = plot_scan(axial_scan)  # doctest: +SKIP
        >>> coronal = plot_scan(coronal_scan)  # doctest: +SKIP
        >>> axial.show_slices_of(coronal)  # doctest: +SKIP
        """
        if colours is None:
            colours = [
                SLICE_LINE_COLOURS[i % len(SLICE_LINE_COLOURS)]
                for i in range(len(views))
            ]
        if len(colours) != len(views):
            raise ValueError(
                f"ScanView.show_slices_of got {len(views)} views but "
                f"{len(colours)} colours; give one colour per view."
            )
        for colour in colours:
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(colour)):
                raise ValueError(
                    f"ScanView.show_slices_of colours must be '#rrggbb'; got {colour!r}."
                )
        for view in views:
            if not _is_view(view) or view is self:
                raise ValueError(
                    "ScanView.show_slices_of needs other ScanView objects, for "
                    "example from plot_scan."
                )
            # Fail now, not on the next slice change.
            view._shown_plane()
        self._shown_plane()
        for view, _ in self._slice_sources:
            if self in view._slice_viewers:
                view._slice_viewers.remove(self)
        self._slice_sources = [
            (view, str(c)) for view, c in zip(views, colours, strict=True)
        ]
        for view in views:
            view._slice_viewers.append(self)
        self._slice_lines = self._current_slice_lines()

    def _lps(self) -> xr.DataArray:
        """Return the displayed data with scanner coordinates x, y, z."""
        if self._lps_data is None or self._lps_data[0] is not self._data:
            data = self._data
            if not has_xyz_coordinates(data) and self._scan is not None:
                data = assign_xyz_coordinates(data, scan=self._scan)
            if not has_xyz_coordinates(data):
                raise ValueError(
                    "Linked ScanViews need scanner coordinates x, y, z "
                    f"(VisuCore geometry) for {self._view.get('title') or 'the view'}; "
                    "pass the scan itself or data with these coordinates."
                )
            self._lps_data = (self._data, data)
        return self._lps_data[1]

    def _shown_plane(self) -> PlaneGrid:
        """Return the shown plane of this view in scanner coordinates."""
        lps = self._lps()
        row_dim, column_dim = self._spatial_dims
        slicing_dim = next(
            (d for d in SPATIAL_DIMS if d not in self._spatial_dims), "slice_idx"
        )
        return plane_grid(
            lps,
            row_dim,
            column_dim,
            self.state.spatial.slice_indices,
            _spatial_dim_resolution_mm(lps, slicing_dim, scan=self._scan),
        )

    def _current_slice_lines(self) -> list[dict[str, Any]]:
        """Return the slice line of every linked view for the browser."""
        if (
            not self._slice_sources
            or self._view.get("single_voxel", True)
            or self._kspace_shown()
        ):
            return []
        shown = self._shown_plane()
        lines = []
        for view, colour in self._slice_sources:
            x, y = slice_line(
                view._shown_plane(),
                shown,
                self._block_indices[self._spatial_dims[0]],
                self._block_indices[self._spatial_dims[1]],
                np.asarray(self._view["x"]),
                np.asarray(self._view["y"]),
            )
            if x:
                lines.append(
                    {
                        "x": x,
                        "y": y,
                        "colour": colour,
                        "name": view._view.get("title") or "slice",
                    }
                )
        return lines

    def link_pointer(self, *views: ScanView) -> None:
        """Show the mouse pointer of this view and others in all of them.

        While the mouse is over the map of one linked view, the other views
        mark the same position in scanner coordinates with a ring: a filled
        ring when the position lies inside their shown slice, an open ring
        when it lies in front of or behind it (the ring then shows the
        position projected onto their plane). The pointer is shared in the
        browser only; Python is not involved while the mouse moves. A click
        that selects a voxel selects, in every linked view whose voxels
        contain that position, the voxel there, and moves that view to the
        slice of the voxel (for example a click in an anatomy chooses the
        CSI voxel and its spectrum). Calling the method again links this
        view with the new views instead.

        Parameters
        ----------
        *views : ScanView
            Views of images of the same session (same patient coordinate
            system). No views unlink this view.

        Raises
        ------
        ValueError
            If a view is this view or no ScanView, or a view has no scanner
            coordinates ``x``, ``y``, ``z`` over its displayed dimensions.

        Examples
        --------
        >>> axial = plot_scan(axial_scan)  # doctest: +SKIP
        >>> coronal = plot_scan(coronal_scan)  # doctest: +SKIP
        >>> axial.link_pointer(coronal)  # doctest: +SKIP
        """
        for view in views:
            if not _is_view(view) or view is self:
                raise ValueError(
                    "ScanView.link_pointer needs other ScanView objects, for "
                    "example from plot_scan."
                )
        for view in (self, *views):
            # Fail before any view is linked.
            view._shown_plane()
        link = uuid.uuid4().hex if views else ""
        group = [self, *views] if views else [self]
        for view in group:
            view._pointer_link = link
            view._pointer_plane = view._current_pointer_plane()
            view._pointer_partners = [other for other in group if other is not view]

    def link_frames(self, *views: ScanView) -> None:
        """Show the same repetition in this view and in others.

        Meant for several views of one scan, for example over different
        anatomies (see :func:`plot_study`): choosing a curve position or a
        slider index in one view shows the same index in the others. Every
        dimension outside the spatial ones (``read_idx``, ``phase_idx``,
        ``slice_idx``) that two views share with the same size is followed,
        whether a view shows it as its curve or as a slider. The selected
        voxel is not shared (see :meth:`link_pointer` for that). Calling the
        method again links this view with the new views instead.

        Parameters
        ----------
        *views : ScanView
            Views to link. No views unlink this view.

        Raises
        ------
        ValueError
            If a view is this view or no ScanView.

        Examples
        --------
        >>> over_axial = plot_scan(bssfp, anatomy=axial)  # doctest: +SKIP
        >>> over_coronal = plot_scan(bssfp, anatomy=coronal, image_only=True)  # doctest: +SKIP
        >>> over_axial.link_frames(over_coronal)  # doctest: +SKIP
        """
        for view in views:
            if not _is_view(view) or view is self:
                raise ValueError(
                    "ScanView.link_frames needs other ScanView objects, for "
                    "example from plot_scan."
                )
        group = [self, *views]
        for view in group:
            for old in view._frame_partners:
                if view in old._frame_partners:
                    old._frame_partners.remove(view)
        for view in group:
            view._frame_partners = (
                [other for other in group if other is not view] if views else []
            )
        self._show_frame_in_partners()

    def _show_frame_in_partners(self) -> None:
        """Give the views of :meth:`link_frames` this view's non-spatial indices.

        A partner whose curve shows the dimension gets a new selection (the
        browser redraws from its block); any other index needs a new block.
        A partner applies the indices without sending them back, since its
        own indices then equal this view's.
        """
        for partner in self._frame_partners:
            new_block = False
            for dim in partner._data.dims:
                if (
                    dim in SPATIAL_DIMS
                    or dim not in self._data.dims
                    or partner._data.sizes[dim] != self._data.sizes[dim]
                ):
                    continue
                index = self._other_index(dim)
                if index == partner._other_index(dim):
                    continue
                if dim == partner._curve_dim:
                    partner.selection = {**partner.selection, "curve_index": index}
                elif dim == SPECTRAL_DIM:
                    partner._select_spectral_index(index, None)
                    new_block = True
                else:
                    partner.state.spatial.slice_indices[dim] = index
                    new_block = True
            if new_block:
                partner._render()

    def show_anatomy_plane(self) -> None:
        """Show the view plane in which the anatomy's slices lie.

        For a three-dimensional image over an anatomy, the plane is chosen
        whose normal is most parallel to the anatomy's slice direction
        (:func:`~hypermri.visualization.geometry.view_plane_for_anatomy`),
        so that the anatomy appears in its acquired slices with their full
        in-plane resolution: a coronal anatomy under a 3D image shows the
        coronal plane. Data with one view plane or no anatomy are left
        unchanged.

        Raises
        ------
        ValueError
            If the data or the anatomy have no scanner coordinates ``x``,
            ``y``, ``z``.

        Examples
        --------
        >>> view = plot_scan(bssfp_3d, anatomy=coronal_t2)  # doctest: +SKIP
        >>> view.show_anatomy_plane()  # doctest: +SKIP
        """
        anatomy = self._model.anatomy_obj
        choices = view_plane_choices(self._data)
        if anatomy is None or len(choices) < 2:
            return
        samples = anatomy if isinstance(anatomy, xr.DataArray) else anatomy.primary_data
        if not has_xyz_coordinates(samples) and not isinstance(anatomy, xr.DataArray):
            samples = assign_xyz_coordinates(samples, scan=anatomy)
        if not has_xyz_coordinates(samples):
            raise ValueError(
                "ScanView.show_anatomy_plane needs scanner coordinates x, y, z "
                "on the anatomy; pass the anatomical scan itself or a "
                "DataArray with these coordinates."
            )
        self.set_view_plane(view_plane_for_anatomy(self._lps(), samples, choices))

    def _selected_position_mm(self) -> np.ndarray:
        """Return the LPS centre of the selected voxel, shape (3,), mm."""
        lps = self._lps()
        indices = self.state.spatial.slice_indices
        position = []
        for name in ("x", "y", "z"):
            coordinate = lps.coords[name]
            at = {dim: indices.get(dim, 0) for dim in coordinate.dims}
            position.append(float(coordinate.isel(at)))
        return np.array(position)

    def _select_in_pointer_partners(self) -> None:
        """Select the voxel at the clicked scanner position in linked views.

        A linked view whose voxels do not contain the position keeps its
        selection. A linked view that shows another slice moves to the slice
        that contains the position.
        """
        if not self._pointer_partners or self._view.get("single_voxel", True):
            return
        position = self._selected_position_mm()
        for partner in self._pointer_partners:
            partner._select_position(position)

    def _select_position(self, position_mm: np.ndarray) -> None:
        """Select the voxel under an LPS position, in the nearest slice.

        The position must lie inside the shown plane's area; along the
        slicing axis (the grid axes that are not shown) the nearest slice is
        taken, so a click selects a voxel even when this stack does not reach
        the clicked image's slice.
        """
        if self._view.get("single_voxel", True):
            return
        dims = mask_dims(self._data)
        slicing_axes = tuple(
            axis
            for axis in range(3)
            if axis >= len(dims) or dims[axis] not in self._spatial_dims
        )
        voxel = nearest_voxel(self._mask_voxel_grid(), position_mm, slicing_axes)
        if voxel[0] < 0:
            return
        indices = self.state.spatial.slice_indices
        chosen = {dim: int(index) for dim, index in zip(dims, voxel, strict=False)}
        if all(indices.get(dim, 0) == index for dim, index in chosen.items()):
            return
        plane_moved = any(
            indices.get(dim, 0) != index
            for dim, index in chosen.items()
            if dim not in self._spatial_dims
        )
        indices.update(chosen)
        if plane_moved and self._model._uses_xyz_coordinates(self.state):
            self._follow_map_with_layers()
        # Python sends the new selection (and slice) with the next block.
        self._render()

    def _current_pointer_plane(self) -> dict[str, Any]:
        """Return the map from plot axes to LPS for a linked pointer, or {}."""
        if (
            not self._pointer_link
            or self._view.get("single_voxel", True)
            or self._kspace_shown()
        ):
            return {}
        return axes_to_lps(
            self._shown_plane(),
            self._block_indices[self._spatial_dims[0]],
            self._block_indices[self._spatial_dims[1]],
            np.asarray(self._view["x"]),
            np.asarray(self._view["y"]),
        )

    def link_masks(self, *views: ScanView) -> None:
        """Share the masks of this view with other views of the same session.

        Linked views have the same mask names and styles. An outline drawn
        on one view, an undo and a deleted mask change the masks of all of
        them, so each scan stores the mask in ``scan.masks`` on its own grid
        and uses it for its ROI mean and volume. A drawn outline is copied
        to another image voxel by voxel: each voxel of that image takes the
        coverage of the voxel of the drawn image that contains its centre
        (in scanner coordinates); only voxels of the edited slice are
        copied. Opening or closing the mask tool, the chosen mask and the
        drawing mode are also shared in the browser, so one tool draws on
        every view.

        A mask that exists in one view only is copied to the others when the
        views are linked. Calling the method again links this view with the
        new views instead; no views unlink it.

        Parameters
        ----------
        *views : ScanView
            Views of images of the same session (same patient coordinate
            system).

        Raises
        ------
        ValueError
            If a view is this view or no ScanView, or a view has no scanner
            coordinates ``x``, ``y``, ``z`` or no grid with at least two
            dimensions of more than one voxel (a single slice needs its
            slice thickness).

        Notes
        -----
        The copy is nearest-neighbour sampling: with images of different
        resolution or orientation it approximates the drawn outline on the
        other grid. Each view keeps its own undo steps; undoing in one view
        copies the restored slice to the others again.

        Examples
        --------
        >>> axial = plot_scan(axial_scan)  # doctest: +SKIP
        >>> coronal = plot_scan(coronal_scan)  # doctest: +SKIP
        >>> axial.link_masks(coronal)  # doctest: +SKIP
        >>> axial.add_mask("tumour")  # coronal gets an empty mask "tumour"  # doctest: +SKIP
        """
        for view in views:
            if not _is_view(view) or view is self:
                raise ValueError(
                    "ScanView.link_masks needs other ScanView objects, for "
                    "example from plot_scan."
                )
        group = [self, *views]
        for view in group:
            # Fail before any view is linked.
            view._mask_voxel_grid()
        for view in group:
            for old in view._mask_partners:
                if view in old._mask_partners:
                    old._mask_partners.remove(view)
                if not old._mask_partners:
                    old._mask_tool_shown = True
        link = uuid.uuid4().hex if views else ""
        for view in group:
            view._mask_partners = [other for other in group if other is not view]
            view._mask_link = link
            view._mask_tool_shown = True
        for view in group:
            for name in list(view._mask_store):
                for partner in view._mask_partners:
                    if name not in partner._mask_store:
                        partner._take_mask(view, name, None)

    def unlink(self) -> None:
        """Remove every link of this view to other views.

        Undoes :meth:`show_slices_of`, :meth:`link_pointer`,
        :meth:`link_frames` and :meth:`link_masks` in both directions: this
        view draws no slice bands of other views, the other views draw no
        band of this view, its mouse pointer and repetition are no longer
        shared, and its masks are no longer copied to or from other views. Links among the other views stay.
        The masks already stored on each scan are kept. Use it before a view
        is closed or shown alone, for example by :func:`plot_study`.

        Returns
        -------
        None
            The view and the views it was linked with are changed in place.
        """
        for source, _ in self._slice_sources:
            if self in source._slice_viewers:
                source._slice_viewers.remove(self)
        for viewer in self._slice_viewers:
            viewer._slice_sources = [
                (view, colour)
                for view, colour in viewer._slice_sources
                if view is not self
            ]
            viewer._slice_lines = viewer._current_slice_lines()
        self._slice_sources = []
        self._slice_viewers = []
        self._slice_lines = []
        self._pointer_link = ""
        self._pointer_plane = {}
        for partner in self._pointer_partners:
            if self in partner._pointer_partners:
                partner._pointer_partners.remove(self)
        self._pointer_partners = []
        for partner in self._frame_partners:
            if self in partner._frame_partners:
                partner._frame_partners.remove(self)
        self._frame_partners = []
        for partner in self._mask_partners:
            if self in partner._mask_partners:
                partner._mask_partners.remove(self)
            if not partner._mask_partners:
                partner._mask_link = ""
                partner._mask_tool_shown = True
        self._mask_partners = []
        self._mask_link = ""
        self._mask_tool_shown = True

    def _mask_voxel_grid(self) -> VoxelGrid:
        """Return the LPS grid of the masks of this view."""
        lps = self._lps()
        dims = mask_dims(self._data)
        single = [d for d in dims if lps.sizes[d] == 1]
        single += [d for d in SPATIAL_DIMS if d not in dims]
        thickness = (
            _spatial_dim_resolution_mm(lps, single[0], scan=self._scan)
            if single
            else None
        )
        try:
            return voxel_grid(lps, dims, thickness)
        except ValueError as error:
            raise ValueError(
                f"ScanView masks of {self._view.get('title') or 'this view'} "
                f"cannot be shared: {error}"
            ) from None

    def _share_mask(self, name: str, changed: np.ndarray | None) -> None:
        """Give the linked views the current state of one mask.

        ``changed`` marks the voxels whose coverage changed; None copies the
        style (and a new or deleted mask) only.
        """
        for partner in self._mask_partners:
            partner._take_mask(self, name, changed)

    def _take_mask(
        self, source: ScanView, name: str, changed: np.ndarray | None
    ) -> None:
        """Take one mask of a linked view, without sharing it further."""
        if name not in source._mask_store:
            if name in self._mask_store:
                self._drop_mask(name)
                self._masks_changed()
            return
        created = name not in self._mask_store
        if created:
            self._mask_store[name] = empty_mask(self._data)
        self._mask_colors[name] = source._mask_colors[name]
        self._mask_alphas[name] = source._mask_alphas[name]
        if created or changed is not None:
            target = self._mask_store[name]
            target.values[...] = transfer_coverage(
                source._mask_store[name].values,
                source._mask_voxel_grid(),
                target.values,
                self._mask_voxel_grid(),
                changed,
            )
        self._masks_changed()
