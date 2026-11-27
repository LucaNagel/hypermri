"""Slices of the anatomy and overlay layers drawn with the map.

Part of :class:`~hypermri.visualization.scan_view.ScanView`: the methods of
``_SecondarySlices`` use the traits and state that ``ScanView`` sets up, so
the class is only meant as a base of ``ScanView``.

By default the anatomy and the overlay show the plane that matches the
shown map. A chosen slice of one of them is kept in its layer state
(``slicing_dim`` and ``slice_indices``); in scanner (``"xyz"``) coordinates
the map then moves to its slice nearest to the chosen one, and moving the
map makes both layers follow it again.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from hypermri._optional import _import_optional

_import_optional("anywidget")


class _SecondarySlices:
    """Choice of the anatomy and overlay slices shown with the map."""

    def set_anatomy_slice(self, index: int | None) -> None:
        """Choose the anatomy slice drawn under the map, or let it follow the map.

        By default the anatomy plane matches the shown map: the same plane
        in ``"rps"`` coordinates, the nearest one at the same scanner
        position in ``"xyz"`` coordinates. A chosen slice is kept in
        ``state.anatomy`` (``slicing_dim`` and ``slice_indices``). In
        ``"xyz"`` coordinates a map with several slices then moves to its
        slice nearest to the chosen anatomy slice; a map with a single slice
        (for example a 2D CSI) stays. A warning is shown when the anatomy
        slice lies outside the map's slices. Moving the map afterwards makes
        the anatomy follow it again. In ``"rps"`` coordinates the choice
        stays until ``None`` is set. The plane is recomputed once.

        Parameters
        ----------
        index : int or None
            Zero-based slice of the anatomy along its dimension normal to
            the shown plane (the browser names it, for example
            ``"slice_idx"``); None lets the anatomy follow the map.

        Raises
        ------
        ValueError
            If the view has no anatomy drawn under a map, the anatomy has a
            single slice along that dimension, or ``index`` is outside
            ``[0, number of slices)``. The state is then unchanged.

        Examples
        --------
        >>> view = plot_scan(csi, anatomy=rare)  # doctest: +SKIP
        >>> view.set_anatomy_slice(5)  # doctest: +SKIP
        >>> view.set_anatomy_slice(None)  # follow the map again  # doctest: +SKIP
        """
        if self._anatomy_data() is None or self._outline is not None:
            raise ValueError(
                "ScanView.set_anatomy_slice: no anatomy is drawn under a map; "
                "use plot_scan(..., anatomy=...)."
            )
        self._change_anatomy_slice(index)
        self._render()
        self._controls = self._current_controls()

    def _anatomy_data(self) -> xr.DataArray | None:
        """Return the anatomy samples drawn under the map, or None."""
        return self._model.get_anatomy_data()

    def _anatomy_slice_axis(self) -> tuple[str | None, int]:
        """Return the anatomy dimension normal to the shown plane and its size."""
        return _slice_axis(
            self.state.anatomy, self._anatomy_data(), self._anatomy_plane
        )

    def _anatomy_slice_view(self) -> dict[str, Any]:
        """Return the anatomy slice axis and the slice drawn, for the browser.

        ``slice_dim`` and ``slice_count`` name the anatomy dimension normal
        to the shown plane (``None`` and 0 without one); ``slice_index`` is
        the slice drawn by the last render, or None.
        """
        dim, count = self._anatomy_slice_axis()
        return {
            "slice_dim": dim,
            "slice_count": count,
            "slice_index": self._anatomy_plane.get(dim) if dim else None,
        }

    def _anatomy_controls(self) -> dict[str, Any] | None:
        """Return the anatomy setting sent as a browser control, or None."""
        if self._anatomy_data() is None or self._outline is not None:
            return None
        return {"slice": _chosen_slice(self.state.anatomy)}

    def _change_anatomy_slice(self, index: Any) -> None:
        """Store a chosen anatomy slice and, in xyz, move the map to it."""
        data = self._anatomy_data()
        if index is not None and data is not None:
            data = self._model._ensure_xyz_coordinates(
                data, self.state, scan=self._model.anatomy_obj
            )
        self._change_secondary_slice(
            self.state.anatomy, data, self._anatomy_plane, index, label="anatomy"
        )

    def _follow_map_with_layers(self) -> None:
        """Let the anatomy and overlay planes follow the shown map again."""
        _follow_map(self.state.anatomy)
        _follow_map(self.state.overlay)

    def _change_secondary_slice(
        self,
        layer_state: Any,
        data: xr.DataArray | None,
        plane: dict[str, int],
        index: Any,
        *,
        label: str,
    ) -> None:
        """Store a chosen anatomy or overlay slice and, in xyz, move the map.

        ``layer_state`` is ``state.anatomy`` or ``state.overlay``, ``data``
        the layer's samples and ``plane`` the slice cut by the last render.
        None lets the layer follow the map. The state is unchanged when the
        index is rejected.
        """
        if index is None:
            _follow_map(layer_state)
            return
        dim, count = _slice_axis(layer_state, data, plane)
        if dim is None or count < 2:
            raise ValueError(
                f"ScanView {label} slice: the {label} has a single slice in the "
                f"shown plane; there is no slice to choose."
            )
        if isinstance(index, bool) or not float(index).is_integer():
            raise ValueError(
                f"ScanView {label} slice must be an integer index; got {index!r}."
            )
        value = int(index)
        if not 0 <= value < count:
            raise ValueError(
                f"ScanView {label} slice {value} is outside {dim} 0..{count - 1}."
            )
        _follow_map(layer_state)
        layer_state.slicing_dim = dim
        layer_state.slice_indices = {dim: value}
        match = self._model.primary_slice_for_secondary_slice(
            self._data, self.state, data, dim, value, label=label
        )
        if match is None:
            return
        primary_dim, primary_index, warning = match
        if primary_index is not None and primary_dim in self._other_dims():
            self.state.spatial.slice_indices[primary_dim] = int(primary_index)
        layer_state.slice_warning = warning
        layer_state.slice_overlap_status = (
            "secondary_outside_primary" if warning else None
        )


def _slice_axis(
    layer_state: Any, data: xr.DataArray | None, plane: dict[str, int]
) -> tuple[str | None, int]:
    """Return a layer's dimension normal to the shown plane and its size.

    The dimension is the chosen one, otherwise the one cut by the last
    render. Returns ``(None, 0)`` when the layer has no such dimension.
    """
    dim = layer_state.slicing_dim or next(iter(plane), None)
    if data is None or dim is None or dim not in data.dims:
        return None, 0
    return dim, int(data.sizes[dim])


def _chosen_slice(layer_state: Any) -> int | None:
    """Return a layer's chosen slice, or None while it follows the map."""
    if layer_state.slicing_dim is None:
        return None
    index = layer_state.slice_indices.get(layer_state.slicing_dim)
    return None if index is None else int(index)


def _follow_map(layer_state: Any) -> None:
    """Let a layer's plane follow the shown map again."""
    layer_state.slicing_dim = None
    layer_state.slice_indices = {}
    layer_state.slice_warning = None
    layer_state.slice_overlap_status = None
