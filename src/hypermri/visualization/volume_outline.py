"""Anatomy plane with the outline of a single-voxel volume or slab.

Single-voxel and slice spectroscopy have no image of their own. To show where
the spectrum comes from, the browser viewer draws one slice of an anatomical
scan and on it the cut of the excited volume (see
:mod:`hypermri.utils.coordinates.excited_volume`) with that slice. The
anatomy plane is prepared by the same :class:`PlotDataModel` path as any
image; this module only chooses the slice and converts the outline from LPS
millimetres to the axes of the shown plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hypermri.utils.coordinates.excited_volume import ExcitedVolume, outline_in_plane
from hypermri.utils.coordinates.grids import assign_xyz_coordinates, has_xyz_coordinates
from hypermri.visualization.geometry import display_voxel_centres
from hypermri.visualization.render_data import PlotDataModel, apply_display_defaults
from hypermri.visualization.state import PlotState

_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


@dataclass(frozen=True)
class AnatomyOutline:
    """One anatomy plane and the outline of the excited volume on it.

    Attributes
    ----------
    image : numpy.ndarray
        Displayed anatomy plane, shape (rows, columns), float32, in the
        orientation of ``plot``.
    extent : list of float
        Cell-edge bounds [x_min, x_max, y_min, y_max] of ``image`` in the
        axis units of the plane.
    origin : str
        ``"upper"`` when row 0 is drawn at the top, else ``"lower"``.
    cmap : str
        Colormap name of the anatomy.
    clim : tuple of float or None
        Colour limits of the anatomy; None uses the value range.
    interpolate : bool
        Whether the anatomy is drawn smoothed.
    x_label, y_label : str
        Axis labels of the plane.
    outline_x, outline_y : list of float
        Closed polygon (first corner repeated at the end) of the cut of the
        volume with the plane, in axis units. Empty if the volume does not
        cross the plane.
    slice_dim : str or None
        Anatomy dimension that selects the plane, or None for a single plane.
    slice_index : int
        Selected index along ``slice_dim``.
    n_slices : int
        Number of planes along ``slice_dim``.
    """

    image: np.ndarray
    extent: list[float]
    origin: str
    cmap: str
    clim: tuple[float, float] | None
    interpolate: bool
    x_label: str
    y_label: str
    outline_x: list[float]
    outline_y: list[float]
    slice_dim: str | None
    slice_index: int
    n_slices: int


def anatomy_data(anatomy: Any) -> xr.DataArray:
    """Return the anatomy samples with LPS coordinates ``x``, ``y`` and ``z``.

    Parameters
    ----------
    anatomy : object or xarray.DataArray
        Anatomical scan with ``primary_data`` or a DataArray with dimensions
        ``read_idx`` and ``phase_idx`` (and optionally ``slice_idx``).

    Returns
    -------
    xarray.DataArray
        The samples, with the scan's VisuCore geometry attached as
        coordinates when they had none. The input is not modified.

    Raises
    ------
    ValueError
        If no samples or no scanner geometry are available.
    """
    data = anatomy if isinstance(anatomy, xr.DataArray) else None
    if data is None:
        data = getattr(anatomy, "primary_data", None)
    if not isinstance(data, xr.DataArray):
        raise ValueError(
            "plot_scan anatomy needs a DataArray or a scan with primary_data."
        )
    if not has_xyz_coordinates(data) and not isinstance(anatomy, xr.DataArray):
        data = assign_xyz_coordinates(data, scan=anatomy)
    if not has_xyz_coordinates(data):
        raise ValueError(
            "plot_scan cannot place the voxel outline on the anatomy: the "
            "anatomy has no scanner coordinates x, y, z (VisuCore geometry)."
        )
    return data


def _lps_positions(data: xr.DataArray, plane: dict[str, int]) -> np.ndarray:
    """Return LPS voxel centres of one plane, shape (..., 3), in index order."""
    coords = [data.coords[name] for name in ("x", "y", "z")]
    present = [dim for dim in _SPATIAL_DIMS if dim in coords[0].dims]
    selection = {dim: index for dim, index in plane.items() if dim in present}
    return np.stack(
        [
            np.asarray(
                c.isel(selection).transpose(*[d for d in present if d not in selection])
            )
            for c in coords
        ],
        axis=-1,
    )


def nearest_slice(
    data: xr.DataArray, slice_dim: str | None, volume: ExcitedVolume
) -> int:
    """Return the anatomy plane closest to the centre of the volume.

    Parameters
    ----------
    data : xarray.DataArray
        Anatomy with LPS coordinates (see :func:`anatomy_data`).
    slice_dim : str or None
        Dimension that selects the plane; None returns 0.
    volume : ExcitedVolume
        Excited volume in LPS mm.

    Returns
    -------
    int
        Index along ``slice_dim`` whose plane centre lies nearest to
        ``volume.centre_mm``.
    """
    if slice_dim is None or slice_dim not in data.coords["x"].dims:
        return 0
    distances = [
        np.min(
            np.linalg.norm(
                _lps_positions(data, {slice_dim: index}).reshape(-1, 3)
                - volume.centre_mm,
                axis=-1,
            )
        )
        for index in range(data.sizes[slice_dim])
    ]
    return int(np.argmin(distances))


def _index_to_axis(source_indices: np.ndarray, centres: np.ndarray):
    """Return a function from a source index to the plot axis value.

    The displayed cells of a regular grid lie on a straight line in the
    source index; one cell keeps its position for every index.
    """
    source = np.asarray(source_indices, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if source.size < 2:
        return lambda index: np.full_like(np.asarray(index, float), centres[0])
    slope, offset = np.polyfit(source, centres, 1)
    return lambda index: offset + slope * np.asarray(index, dtype=float)


def anatomy_outline(
    anatomy: Any,
    data: xr.DataArray,
    state: PlotState,
    volume: ExcitedVolume,
) -> AnatomyOutline:
    """Render one anatomy plane and the cut of the excited volume with it.

    Parameters
    ----------
    anatomy : object or xarray.DataArray
        Anatomical scan, used as acquisition context of the plane.
    data : xarray.DataArray
        Anatomy samples with LPS coordinates (see :func:`anatomy_data`).
    state : PlotState
        Display state of the anatomy plane; ``state.spatial.slicing_dim``
        and ``state.spatial.slice_indices`` choose the plane. It is not
        modified.
    volume : ExcitedVolume
        Excited volume in LPS mm.

    Returns
    -------
    AnatomyOutline
        The plane as drawn by ``plot`` and the outline in its axis units.

    Raises
    ------
    ValueError
        If the anatomy has no two-dimensional plane.
    """
    model = PlotDataModel(anatomy if not isinstance(anatomy, xr.DataArray) else data)
    payload = model.get_render_payload(data, state)
    if payload.spatial_array is None or np.ndim(payload.spatial_array) != 2:
        raise ValueError(
            "plot_scan anatomy needs a two-dimensional image plane"
            f"{f' ({payload.spatial_error})' if payload.spatial_error else ''}."
        )
    image = np.asarray(payload.spatial_array, dtype=np.float32)
    row_dim, column_dim = payload.spatial_dims
    slice_dim = (
        state.spatial.slicing_dim if state.spatial.slicing_dim in data.dims else None
    )
    slice_index = int(state.spatial.slice_indices.get(slice_dim, 0)) if slice_dim else 0

    # LPS centres of the plane on the source grid, as (row source index,
    # column source index).
    positions = _lps_positions(data, {slice_dim: slice_index} if slice_dim else {})
    present = [
        d for d in _SPATIAL_DIMS if d in data.coords["x"].dims and d != slice_dim
    ]
    if present.index(row_dim) > present.index(column_dim):
        positions = np.swapaxes(positions, 0, 1)
    n_rows, n_columns = positions.shape[:2]
    origin = positions[0, 0]
    row_step = positions[1, 0] - origin if n_rows > 1 else np.zeros(3)
    column_step = positions[0, 1] - origin if n_columns > 1 else np.zeros(3)
    corners = outline_in_plane(
        volume, origin, row_step, column_step, (n_rows, n_columns)
    )

    x, y = display_voxel_centres(payload.spatial_extent, image.shape)
    to_x = _index_to_axis(payload.spatial_source_indices[column_dim], x)
    to_y = _index_to_axis(payload.spatial_source_indices[row_dim], y)
    if len(corners):
        corners = np.vstack([corners, corners[:1]])
    clim = payload.spatial_clim
    return AnatomyOutline(
        image=image,
        extent=[float(value) for value in payload.spatial_extent],
        origin=payload.spatial_origin,
        cmap=payload.spatial_cmap,
        clim=None if clim is None else (float(clim[0]), float(clim[1])),
        interpolate=bool(payload.spatial_interpolation),
        x_label=payload.labels.get("spatial_x", column_dim),
        y_label=payload.labels.get("spatial_y", row_dim),
        outline_x=to_x(corners[:, 1]).tolist() if len(corners) else [],
        outline_y=to_y(corners[:, 0]).tolist() if len(corners) else [],
        slice_dim=slice_dim,
        slice_index=slice_index,
        n_slices=int(data.sizes[slice_dim]) if slice_dim else 1,
    )


def anatomy_outline_state(
    anatomy: Any, data: xr.DataArray, volume: ExcitedVolume
) -> PlotState:
    """Return the start display state of the anatomy plane under an outline.

    The defaults of ``plot`` for the anatomy apply; the plane is the one
    nearest the centre of the volume. In an absolute range mode
    ("Min/Max", "Absolute") the colour limits start at the finite value
    range of that plane, as the CSI map does.

    Parameters
    ----------
    anatomy : object or xarray.DataArray
        Anatomical scan.
    data : xarray.DataArray
        Anatomy samples with LPS coordinates.
    volume : ExcitedVolume
        Excited volume in LPS mm.

    Returns
    -------
    PlotState
        New display state; nothing else is modified.
    """
    state = PlotState()
    model = PlotDataModel(anatomy if not isinstance(anatomy, xr.DataArray) else data)
    apply_display_defaults(state, data, model)
    slice_dim = state.spatial.slicing_dim
    if slice_dim in data.dims:
        state.spatial.slice_indices[slice_dim] = nearest_slice(data, slice_dim, volume)
    if state.spatial.range_mode != "Percentage":
        plane = model.get_render_payload(data, state).spatial_array
        finite = np.asarray(plane, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            state.spatial.vmin, state.spatial.vmax = (
                float(finite.min()),
                float(finite.max()),
            )
    return state


@dataclass(frozen=True)
class OutlineSource:
    """Everything needed to redraw the anatomy plane under an outline.

    Attributes
    ----------
    anatomy : object or xarray.DataArray
        Anatomical scan.
    data : xarray.DataArray
        Anatomy samples with LPS coordinates.
    state : PlotState
        Display state of the anatomy plane. The browser slice slider changes
        ``state.spatial.slice_indices`` in place.
    volume : ExcitedVolume
        Excited volume of the spectroscopy scan.
    """

    anatomy: Any
    data: xr.DataArray
    state: PlotState
    volume: ExcitedVolume

    def render(self) -> AnatomyOutline:
        """Return the current anatomy plane and outline (see :func:`anatomy_outline`)."""
        return anatomy_outline(self.anatomy, self.data, self.state, self.volume)


def outline_source(anatomy: Any, volume: ExcitedVolume) -> OutlineSource:
    """Prepare the anatomy under the outline of an excited volume.

    Parameters
    ----------
    anatomy : object or xarray.DataArray
        Anatomical scan or samples with scanner coordinates.
    volume : ExcitedVolume
        Excited volume in LPS mm.

    Returns
    -------
    OutlineSource
        Anatomy with LPS coordinates and a start state showing the plane
        nearest the volume centre.

    Raises
    ------
    ValueError
        If the anatomy has no samples or no scanner geometry.
    """
    data = anatomy_data(anatomy)
    return OutlineSource(
        anatomy, data, anatomy_outline_state(anatomy, data, volume), volume
    )


__all__ = [
    "AnatomyOutline",
    "OutlineSource",
    "outline_source",
    "anatomy_data",
    "anatomy_outline",
    "anatomy_outline_state",
    "nearest_slice",
]
