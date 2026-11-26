"""Where the shown slice of one image crosses the shown slice of another.

When two images of the same session are viewed side by side (for example an
axial and a coronal anatomy), each image can show where the other image's
current slice crosses it, as the slice position lines of a scanner console.
The shown slice of the other image is a thin box in DICOM patient
coordinates (LPS, mm, see
:func:`hypermri.utils.coordinates.excited_volume.image_slice_volume`); its
cut with the shown plane (:func:`~hypermri.utils.coordinates.excited_volume.outline_in_plane`)
is a narrow band, drawn as a closed polygon in the axis units of the plot.

The same views share the mouse pointer: :func:`axes_to_lps` gives the affine
map from the plot axes of a shown plane to LPS, with which the browser places
a pointer of one view in the others, and :func:`voxel_grid` with
:func:`nearest_voxel` give, for every voxel of one image, the voxel of another
image at the same place, with which a mask drawn on one image is copied to the
other.

Everything here is geometry on the scanner coordinates ``x``, ``y`` and
``z`` of the samples. Nothing is rendered or modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from hypermri.utils.coordinates.excited_volume import (
    ExcitedVolume,
    image_slice_volume,
    outline_in_plane,
)
from hypermri.visualization.volume_outline import _index_to_axis

_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")

# Planes whose normals are closer than this (|cos| of the angle) are treated
# as parallel: they do not cross in a line inside the image.
_PARALLEL_COSINE = 0.999


@dataclass(frozen=True)
class PlaneGrid:
    """Regular grid of one shown image plane in LPS.

    Attributes
    ----------
    origin_mm : numpy.ndarray
        LPS position of the centre of source pixel (0, 0), shape (3,), mm.
    row_step_mm, column_step_mm : numpy.ndarray
        LPS displacement to the next pixel along the row and the column
        dimension, shape (3,), mm. Zero when the plane has one row or
        column.
    shape : tuple of int
        Number of source rows and columns.
    thickness_mm : float
        Slice thickness: the spacing of neighbouring planes along the
        dimension that selects the plane, or the scan's
        ``slice_thickness_mm`` for a single plane; 0 when neither is known.
    """

    origin_mm: np.ndarray
    row_step_mm: np.ndarray
    column_step_mm: np.ndarray
    shape: tuple[int, int]
    thickness_mm: float

    @property
    def normal(self) -> np.ndarray:
        """Unit normal of the plane, shape (3,); zeros if it has no area."""
        normal = np.cross(self.row_step_mm, self.column_step_mm)
        length = np.linalg.norm(normal)
        return normal / length if length else normal

    def slice_volume(self) -> ExcitedVolume:
        """Return the plane with its thickness as a box (see ``image_slice_volume``)."""
        return image_slice_volume(
            self.origin_mm,
            self.row_step_mm,
            self.column_step_mm,
            self.shape,
            self.thickness_mm,
        )


def _positions(
    data: xr.DataArray, dims: list[str], plane: dict[str, int]
) -> np.ndarray:
    """Return LPS centres of one plane, shape (len(dims[0]), len(dims[1]), 3)."""
    selection = {
        dim: index for dim, index in plane.items() if dim in data.coords["x"].dims
    }
    return np.stack(
        [
            np.asarray(data.coords[name].isel(selection).transpose(*dims), dtype=float)
            for name in ("x", "y", "z")
        ],
        axis=-1,
    )


def plane_grid(
    data: xr.DataArray,
    row_dim: str,
    column_dim: str,
    plane: dict[str, int],
    slice_thickness_mm: float | None = None,
) -> PlaneGrid:
    """Return the LPS grid of the plane spanned by two spatial dimensions.

    Parameters
    ----------
    data : xarray.DataArray
        Samples with scanner coordinates ``x``, ``y`` and ``z`` (LPS, mm)
        over the spatial dimensions, for example from
        :func:`hypermri.visualization.volume_outline.anatomy_data`.
    row_dim, column_dim : str
        Spatial dimensions along the rows and columns of the plane, in the
        source index order (not the displayed, possibly rotated order).
    plane : dict of str to int
        Index of every other spatial dimension of the coordinates (the
        shown slice). Missing dimensions use index 0.
    slice_thickness_mm : float or None, optional
        Thickness of a single plane, in mm, used when no neighbouring plane
        gives the spacing. None gives 0.

    Returns
    -------
    PlaneGrid
        Origin, steps, shape and thickness of the plane.

    Raises
    ------
    ValueError
        If the coordinates do not span ``row_dim`` and ``column_dim``.
    """
    coord_dims = list(data.coords["x"].dims) if "x" in data.coords else []
    if row_dim not in coord_dims or column_dim not in coord_dims:
        raise ValueError(
            f"slice lines need scanner coordinates x, y, z over {row_dim} and "
            f"{column_dim}; they span {coord_dims or 'nothing'}."
        )
    other = [d for d in coord_dims if d not in (row_dim, column_dim)]
    if any(d not in _SPATIAL_DIMS for d in other):
        raise ValueError(
            "slice lines need scanner coordinates over spatial dimensions only; "
            f"x spans {coord_dims}."
        )
    selection = {d: int(plane.get(d, 0)) for d in other}
    positions = _positions(data, [row_dim, column_dim], selection)
    n_rows, n_columns = positions.shape[:2]
    origin = positions[0, 0]
    row_step = positions[1, 0] - origin if n_rows > 1 else np.zeros(3)
    column_step = positions[0, 1] - origin if n_columns > 1 else np.zeros(3)
    thickness = 0.0
    if other and data.sizes[other[0]] > 1:
        # Spacing of neighbouring planes at the shown one.
        index = selection[other[0]]
        neighbour = index + 1 if index + 1 < data.sizes[other[0]] else index - 1
        moved = _positions(
            data, [row_dim, column_dim], {**selection, other[0]: neighbour}
        )
        thickness = float(np.linalg.norm(moved[0, 0] - origin))
    elif slice_thickness_mm is not None and np.isfinite(slice_thickness_mm):
        thickness = max(float(slice_thickness_mm), 0.0)
    return PlaneGrid(origin, row_step, column_step, (n_rows, n_columns), thickness)


def slice_line(
    other: PlaneGrid,
    shown: PlaneGrid,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Return where another image's slice crosses the shown plane.

    Parameters
    ----------
    other : PlaneGrid
        The other image's shown slice.
    shown : PlaneGrid
        The shown plane, with rows and columns in source index order.
    row_indices, column_indices : numpy.ndarray
        Source index of every displayed row and column, as the view sends
        them (after rotation).
    x, y : numpy.ndarray
        Displayed axis value of every displayed column and row centre.

    Returns
    -------
    tuple of list of float
        Closed polygon ``(x, y)`` in displayed axis units (first corner
        repeated at the end); empty lists when the slices do not cross inside
        the shown image or are parallel.
    """
    if (
        not np.any(shown.normal)
        or not np.any(other.normal)
        or abs(float(shown.normal @ other.normal)) > _PARALLEL_COSINE
    ):
        return [], []
    corners = outline_in_plane(
        other.slice_volume(),
        shown.origin_mm,
        shown.row_step_mm,
        shown.column_step_mm,
        shown.shape,
    )
    if not len(corners):
        return [], []
    corners = np.vstack([corners, corners[:1]])
    to_x = _index_to_axis(column_indices, x)
    to_y = _index_to_axis(row_indices, y)
    return to_x(corners[:, 1]).tolist(), to_y(corners[:, 0]).tolist()


def _axis_to_index(
    source_indices: np.ndarray, centres: np.ndarray
) -> tuple[float, float]:
    """Return ``(offset, slope)`` of the source index as a function of the axis."""
    source = np.asarray(source_indices, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if source.size < 2:
        return float(source[0]), 0.0
    slope, offset = np.polyfit(centres, source, 1)
    return float(offset), float(slope)


def axes_to_lps(
    shown: PlaneGrid,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, list[float] | float]:
    """Return the affine map from the plot axes of a shown plane to LPS.

    A point at plot position ``(x, y)`` lies at
    ``origin_mm + x * x_step_mm + y * y_step_mm`` in DICOM patient
    coordinates. The browser uses the map in both directions to show the
    mouse pointer of one view at the same place in linked views.

    Parameters
    ----------
    shown : PlaneGrid
        The shown plane, with rows and columns in source index order.
    row_indices, column_indices : numpy.ndarray
        Source index of every displayed row and column, as the view sends
        them (after rotation).
    x, y : numpy.ndarray
        Displayed axis value of every displayed column and row centre.

    Returns
    -------
    dict
        ``origin_mm``, ``x_step_mm`` and ``y_step_mm`` as lists of three
        floats (LPS, mm; the steps per axis unit), and ``thickness_mm`` of
        the plane. A step is zero along an axis with a single pixel.
    """
    column_offset, column_slope = _axis_to_index(column_indices, x)
    row_offset, row_slope = _axis_to_index(row_indices, y)
    origin = (
        shown.origin_mm
        + row_offset * shown.row_step_mm
        + column_offset * shown.column_step_mm
    )
    return {
        "origin_mm": [float(value) for value in origin],
        "x_step_mm": [float(value) for value in column_slope * shown.column_step_mm],
        "y_step_mm": [float(value) for value in row_slope * shown.row_step_mm],
        "thickness_mm": float(shown.thickness_mm),
    }


@dataclass(frozen=True)
class VoxelGrid:
    """Regular three-dimensional voxel grid of one image in LPS.

    Attributes
    ----------
    origin_mm : numpy.ndarray
        LPS centre of voxel (0, 0, 0), shape (3,), mm.
    steps_mm : numpy.ndarray
        Row ``i`` is the LPS displacement to the next voxel along grid axis
        ``i``, shape (3, 3), mm. Along an axis with a single voxel it is the
        plane normal times the slice thickness.
    shape : tuple of int
        Number of voxels along the three grid axes.
    """

    origin_mm: np.ndarray
    steps_mm: np.ndarray
    shape: tuple[int, int, int]

    def centres_mm(self) -> np.ndarray:
        """Return the LPS centre of every voxel, shape ``shape + (3,)``, mm."""
        indices = np.indices(self.shape, dtype=float)
        return self.origin_mm + np.tensordot(
            np.moveaxis(indices, 0, -1), self.steps_mm, axes=1
        )


def voxel_grid(
    data: xr.DataArray,
    dims: list[str],
    slice_thickness_mm: float | None = None,
) -> VoxelGrid:
    """Return the LPS voxel grid of ``data`` over its spatial dimensions.

    The voxel centres are assumed to lie on a regular grid, as for images
    reconstructed on the scanner: the steps are taken from the first two
    voxels along each dimension.

    Parameters
    ----------
    data : xarray.DataArray
        Samples with scanner coordinates ``x``, ``y`` and ``z`` (LPS, mm)
        over ``dims``.
    dims : list of str
        Two or three spatial dimensions in the order of the grid axes, for
        example the dimensions of a mask. With two, the grid gets a third
        axis of size 1 along the plane normal.
    slice_thickness_mm : float or None, optional
        Thickness of a single plane, in mm, used along an axis with one
        voxel. None or a value <= 0 is accepted only when no such axis
        exists.

    Returns
    -------
    VoxelGrid
        Origin, steps and shape (``dims`` sizes, padded with 1 to three).

    Raises
    ------
    ValueError
        If the coordinates do not span ``dims``, more than one axis has a
        single voxel, or an axis has a single voxel and the thickness is
        unknown.
    """
    coord_dims = list(data.coords["x"].dims) if "x" in data.coords else []
    if not 2 <= len(dims) <= 3 or any(d not in coord_dims for d in dims):
        raise ValueError(
            f"a voxel grid needs scanner coordinates x, y, z over two or three "
            f"spatial dimensions {dims}; they span {coord_dims or 'nothing'}."
        )
    shape = tuple(int(data.sizes[d]) for d in dims) + (1,) * (3 - len(dims))
    other = {d: 0 for d in coord_dims if d not in dims}
    if len(dims) == 2:
        centres = _positions(data, list(dims), other)
    else:
        centres = np.stack(
            [
                np.asarray(data.coords[name].isel(other).transpose(*dims), dtype=float)
                for name in ("x", "y", "z")
            ],
            axis=-1,
        )
    centres = centres.reshape(shape + (3,))
    origin = centres[0, 0, 0]
    steps = np.zeros((3, 3))
    for axis in range(3):
        if shape[axis] > 1:
            steps[axis] = np.take(centres, 1, axis=axis)[0, 0] - origin
    single = [axis for axis in range(3) if shape[axis] == 1]
    if len(single) > 1:
        raise ValueError(
            f"a voxel grid needs at least two dimensions with more than one "
            f"voxel; {dims} have sizes {shape[: len(dims)]}."
        )
    if single:
        first, second = (axis for axis in range(3) if axis != single[0])
        normal = np.cross(steps[first], steps[second])
        length = np.linalg.norm(normal)
        if (
            slice_thickness_mm is None
            or not np.isfinite(slice_thickness_mm)
            or slice_thickness_mm <= 0
            or length == 0
        ):
            raise ValueError(
                "a voxel grid of a single slice needs its slice thickness; "
                "set slice_thickness_mm on the scan or the data attrs."
            )
        steps[single[0]] = normal / length * float(slice_thickness_mm)
    return VoxelGrid(origin, steps, shape)


def nearest_voxel(
    source: VoxelGrid, points_mm: np.ndarray, clamp_axes: tuple[int, ...] = ()
) -> np.ndarray:
    """Return the index of the ``source`` voxel that contains each point.

    Along the grid axes in ``clamp_axes`` a point beyond the grid takes the
    nearest edge voxel instead of counting as outside. A click in one image
    then selects the nearest slice of an image whose slab does not contain
    the clicked position, as long as the in-plane position lies inside.

    Parameters
    ----------
    source : VoxelGrid
        Grid whose voxels are looked up.
    points_mm : numpy.ndarray
        LPS positions, shape ``(..., 3)``, mm.
    clamp_axes : tuple of int, optional
        Grid axes (0, 1 or 2) along which points are moved onto the nearest
        voxel. Default: none, so every axis must contain the point.

    Returns
    -------
    numpy.ndarray
        int64 indices of shape ``(..., 3)`` along the three grid axes; -1 in
        all three for a point outside the grid. A voxel contains the points
        less than half a step from its centre along every axis.
    """
    points = np.asarray(points_mm, dtype=float)
    fractional = np.linalg.solve(
        source.steps_mm.T, (points - source.origin_mm).reshape(-1, 3).T
    ).T
    indices = np.floor(fractional + 0.5).astype(np.int64)
    for axis in clamp_axes:
        indices[:, axis] = np.clip(indices[:, axis], 0, source.shape[axis] - 1)
    inside = np.all((indices >= 0) & (indices < np.asarray(source.shape)), axis=1)
    indices[~inside] = -1
    return indices.reshape(points.shape)


__all__ = [
    "PlaneGrid",
    "VoxelGrid",
    "axes_to_lps",
    "nearest_voxel",
    "plane_grid",
    "slice_line",
    "voxel_grid",
]
