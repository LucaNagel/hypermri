"""Draw spatial masks from polygons without widgets.

The mask tool of the browser viewer
(:class:`~hypermri.visualization.scan_view.ScanView`) uses these functions; they
need no widget, so the stored coverage of a drawn outline can be tested and
reproduced without a browser.

A mask is a float32 :class:`xarray.DataArray` over the spatial dimensions of
the data (``read_idx``, ``phase_idx`` and ``slice_idx``, in that order, as far
as present), holding the fraction of each voxel inside the drawn outline, in
[0, 1]. Masks of a scan are stored together as an :class:`xarray.Dataset` in
``scan.masks``; each variable carries its display ``color`` and ``alpha`` and
the measurements ``weighted_voxel_count`` and, when the voxel spacing is known,
``volume_ml``. Outlines drawn at full precision are kept beside the
coverage in the attr ``outlines`` (see :func:`outline_to_source`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hypermri.analysis.anatomical.roi import (
    _sanitize_mask_values,
    mask_volume_ml,
    raw_masked_mean,
)
from hypermri.visualization.slice_lines import VoxelGrid, nearest_voxel

MASK_DIMS = ("read_idx", "phase_idx", "slice_idx")
DEFAULT_MASK_COLOR = "#ff0000"
DEFAULT_MASK_ALPHA = 0.4


def mask_dims(data: xr.DataArray) -> list[str]:
    """Return the spatial dimensions a mask of ``data`` covers.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled samples.

    Returns
    -------
    list of str
        The dimensions of :data:`MASK_DIMS` present in ``data``, in the order
        ``read_idx``, ``phase_idx``, ``slice_idx``.
    """
    return [dim for dim in MASK_DIMS if dim in data.dims]


def empty_mask(data: xr.DataArray) -> xr.DataArray:
    """Create a mask with zero coverage on the spatial grid of ``data``.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled samples with at least one of ``read_idx``, ``phase_idx`` and
        ``slice_idx``. Not modified.

    Returns
    -------
    xarray.DataArray
        float32 zeros with dimensions :func:`mask_dims` of ``data`` and deep
        copies of every coordinate of ``data`` that depends only on these
        dimensions and is named after one of them or ``x``, ``y`` or ``z``.
    """
    dims = mask_dims(data)
    coords = {
        name: coord.copy(deep=True)
        for name, coord in data.coords.items()
        if (name in dims or name in {"x", "y", "z"}) and set(coord.dims).issubset(dims)
    }
    return xr.DataArray(
        np.zeros(tuple(data.sizes[dim] for dim in dims), dtype=np.float32),
        dims=dims,
        coords=coords,
    )


def polygon_coverage(
    vertices: Sequence[tuple[float, float]],
    extent: Sequence[float],
    shape: tuple[int, int],
    *,
    subpixel_resolution: int = 8,
) -> np.ndarray:
    """Rasterize a drawn outline into the covered fraction of each voxel.

    Parameters
    ----------
    vertices : sequence of (float, float)
        Outline points as (x, y) in the displayed axis units of the plane.
    extent : sequence of float
        Directed cell-edge bounds ``[xmin, xmax, ymin, ymax]`` of the
        displayed plane, as in :attr:`RenderPayload.spatial_extent`.
    shape : tuple of int
        Displayed (rows, columns) of the plane.
    subpixel_resolution : int, optional
        Samples per voxel edge for the line stroke of a two-point outline.

    Returns
    -------
    numpy.ndarray
        float32 coverage in [0, 1] with ``shape``, in displayed row/column
        order. Row ``i`` spans y from ``ymin + i * dy`` to
        ``ymin + (i + 1) * dy`` with ``dy = (ymax - ymin) / rows``.

    Raises
    ------
    ValueError
        If the extent has zero width or height.

    Notes
    -----
    Three or more vertices form a closed polygon whose exact overlap area
    with each voxel is the coverage. Two vertices, or a polygon without area,
    give a one-voxel-wide line. Fewer than two vertices give zeros. Pure and
    deterministic.
    """
    if len(vertices) < 2:
        return np.zeros(shape, dtype=np.float32)
    pixel_vertices = _vertices_to_pixel_space(vertices, extent, shape)
    if len(vertices) >= 3:
        fill = _closed_polygon_coverage(pixel_vertices, shape)
        if np.any(fill > 0):
            return _sanitize_mask_values(fill)
    return _sanitize_mask_values(
        _polyline_coverage(
            pixel_vertices, shape=shape, subpixel_resolution=subpixel_resolution
        )
    )


def combine_plane_coverage(
    previous: np.ndarray, coverage: np.ndarray, *, erase: bool
) -> np.ndarray:
    """Combine a stored mask plane with newly drawn coverage.

    Parameters
    ----------
    previous : numpy.ndarray
        Stored coverage of the plane, in [0, 1].
    coverage : numpy.ndarray
        Drawn coverage of the same plane in the same dimension order.
    erase : bool
        False adds the drawn area (voxel-wise maximum); True removes it
        (``previous - coverage``, clipped to [0, 1]).

    Returns
    -------
    numpy.ndarray
        New float32 plane coverage in [0, 1]. Inputs are not modified.
    """
    previous = np.asarray(previous, dtype=np.float32)
    if erase:
        combined = np.clip(previous - coverage, 0.0, 1.0)
    else:
        combined = np.maximum(previous, coverage)
    return _sanitize_mask_values(combined)


def transfer_coverage(
    source: np.ndarray,
    source_grid: VoxelGrid,
    target: np.ndarray,
    target_grid: VoxelGrid,
    changed: np.ndarray | None = None,
) -> np.ndarray:
    """Copy mask coverage from one image to another image of the same session.

    Every target voxel whose centre lies in a source voxel takes that source
    voxel's coverage (nearest-neighbour sampling in LPS). Target voxels
    outside the source grid, or whose source voxel is not ``changed``, keep
    their coverage. With images of different resolution this is an
    approximation of the drawn outline on the target grid, not an exact
    overlap.

    Parameters
    ----------
    source : numpy.ndarray
        Coverage in [0, 1] on the source grid; its shape padded with 1 to
        three axes equals ``source_grid.shape``. Not modified.
    source_grid, target_grid : VoxelGrid
        LPS grids of the two masks (see
        :func:`hypermri.visualization.slice_lines.voxel_grid`).
    target : numpy.ndarray
        Coverage on the target grid, shape matching ``target_grid`` in the
        same way. Not modified.
    changed : numpy.ndarray or None, optional
        Boolean array of the shape of ``source``: the source voxels whose
        coverage is copied. None copies every source voxel.

    Returns
    -------
    numpy.ndarray
        New float32 target coverage of the shape of ``target``.
    """
    source_values = np.asarray(source, dtype=np.float32).reshape(source_grid.shape)
    result = np.array(target, dtype=np.float32).reshape(target_grid.shape)
    indices = nearest_voxel(source_grid, target_grid.centres_mm())
    take = indices[..., 0] >= 0
    if changed is not None:
        selected = np.asarray(changed, dtype=bool).reshape(source_grid.shape)
        take[take] = selected[tuple(indices[take].T)]
    result[take] = source_values[tuple(indices[take].T)]
    return result.reshape(np.shape(target))


def shown_coverage_to_source(
    coverage: np.ndarray,
    shown_dims: Sequence[str],
    source_indices: Mapping[str, np.ndarray],
    source_sizes: Mapping[str, int],
    target_dims: Sequence[str],
) -> np.ndarray:
    """Place coverage drawn on the shown plane on the data grid.

    Parameters
    ----------
    coverage : numpy.ndarray
        Coverage in shown order, shape (n_rows, n_columns), values in [0, 1].
    shown_dims : sequence of str
        Shown [row, column] dimensions, as ``RenderPayload.spatial_dims``.
    source_indices : mapping of str to numpy.ndarray
        Data-grid position of every shown row and column, keyed by the two
        shown dimensions, as ``RenderPayload.spatial_source_indices``. They
        hold rotation, scanner orientation and any display reversal.
    source_sizes : mapping of str to int
        Size of each of the two dimensions on the data grid.
    target_dims : sequence of str
        The two shown dimensions in the order of the stored mask plane.

    Returns
    -------
    numpy.ndarray
        float32 coverage of the data-grid plane in target_dims order. Voxels
        that are not shown get 0. Inputs are not modified.

    Raises
    ------
    ValueError
        If the dimensions do not match, an index is missing, or the coverage
        shape differs from the number of shown rows and columns.

    Examples
    --------
    A plane shown with its rows reversed (as for a descending scanner axis):

    >>> shown = np.array([[1.0, 0.0], [0.0, 0.0]])
    >>> shown_coverage_to_source(
    ...     shown,
    ...     ["phase_idx", "read_idx"],
    ...     {"phase_idx": np.array([1, 0]), "read_idx": np.array([0, 1])},
    ...     {"phase_idx": 2, "read_idx": 2},
    ...     ["read_idx", "phase_idx"],
    ... )
    array([[0., 1.],
           [0., 0.]], dtype=float32)
    """
    rows, columns = _shown_plane_indices(
        "shown_coverage_to_source", shown_dims, source_indices, target_dims
    )
    coverage = np.asarray(coverage, dtype=np.float32)
    if coverage.shape != (len(rows), len(columns)):
        raise ValueError(
            "shown_coverage_to_source: coverage shape "
            f"{coverage.shape} differs from the shown plane "
            f"({len(rows)}, {len(columns)})."
        )
    row_dim, column_dim = shown_dims
    source = np.zeros(
        (int(source_sizes[row_dim]), int(source_sizes[column_dim])),
        dtype=np.float32,
    )
    source[np.ix_(rows, columns)] = coverage
    return source if list(target_dims) == list(shown_dims) else source.T


def source_coverage_to_shown(
    plane: xr.DataArray,
    shown_dims: Sequence[str],
    source_indices: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return a stored mask plane in shown order.

    Parameters
    ----------
    plane : xarray.DataArray
        Stored coverage of one plane, dimensions equal to the two shown
        dimensions in any order.
    shown_dims : sequence of str
        Shown [row, column] dimensions, as ``RenderPayload.spatial_dims``.
    source_indices : mapping of str to numpy.ndarray
        Data-grid position of every shown row and column, as
        ``RenderPayload.spatial_source_indices``.

    Returns
    -------
    numpy.ndarray
        float32 coverage, shape (n_rows, n_columns), in shown order. The
        inverse of :func:`shown_coverage_to_source` on the shown voxels.

    Raises
    ------
    ValueError
        If the dimensions do not match or an index is missing.
    """
    rows, columns = _shown_plane_indices(
        "source_coverage_to_shown", shown_dims, source_indices, list(plane.dims)
    )
    values = np.asarray(plane.transpose(*shown_dims).values, dtype=np.float32)
    return values[np.ix_(rows, columns)]


def _shown_plane_indices(
    operation: str,
    shown_dims: Sequence[str],
    source_indices: Mapping[str, np.ndarray],
    other_dims: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the row and column source indices after checking the dimensions."""
    shown = list(shown_dims)
    if len(shown) != 2 or sorted(shown) != sorted(other_dims):
        raise ValueError(
            f"{operation}: the shown plane {shown} and the mask plane "
            f"{list(other_dims)} must have the same two dimensions."
        )
    missing = [dim for dim in shown if dim not in source_indices]
    if missing:
        raise ValueError(
            f"{operation}: no source indices for {missing}; use the "
            "spatial_source_indices of the render payload."
        )
    return (
        np.asarray(source_indices[shown[0]], dtype=int),
        np.asarray(source_indices[shown[1]], dtype=int),
    )


def outline_to_source(
    vertices: Sequence[tuple[float, float]],
    plane: Mapping[str, Any],
    *,
    erase: bool = False,
) -> dict[str, Any] | None:
    """Convert an outline drawn on the shown plane to data-grid coordinates.

    The coverage of a mask only keeps the fraction of each voxel inside the
    outline. The outline itself, stored with this function, keeps the drawn
    shape at full precision, so it can be shown again on top of the coarse
    voxels.

    Parameters
    ----------
    vertices : sequence of (float, float)
        Outline points as (x, y) in the displayed axis units of the plane.
    plane : mapping
        The shown plane: ``extent`` (directed cell-edge bounds
        ``[xmin, xmax, ymin, ymax]``), ``shape`` (rows, columns),
        ``row_dim``, ``column_dim``, ``source_indices`` (data-grid index of
        every shown row and column, by dimension), ``slicing_dim`` (the third
        spatial dimension or None) and ``slice_index`` (its shown index or
        None).
    erase : bool, optional
        Whether the outline removed area from the mask.

    Returns
    -------
    dict or None
        JSON-safe outline with ``dims`` (the two shown dimensions),
        ``points`` (one ``[position, position]`` pair per vertex in the
        order of ``dims``), ``slicing_dim``, ``slice_index`` and ``erase``.
        A position is in voxel units of the data grid: voxel ``k`` spans
        ``[k, k + 1]``, so 2.5 is the centre of voxel 2. None if a shown
        axis is not an evenly stepped run of data-grid indices (step +1 or
        -1), because then no straight line on screen is a straight line on
        the data grid.

    Raises
    ------
    ValueError
        If the extent has zero width or height.

    Examples
    --------
    A 2 x 2 plane, 1 mm voxels from 0 to 2 mm, rows shown in reversed order:

    >>> plane = {
    ...     "extent": [0.0, 2.0, 0.0, 2.0], "shape": (2, 2),
    ...     "row_dim": "phase_idx", "column_dim": "read_idx",
    ...     "source_indices": {"phase_idx": np.array([1, 0]),
    ...                        "read_idx": np.array([0, 1])},
    ...     "slicing_dim": None, "slice_index": None,
    ... }
    >>> outline_to_source([(0.5, 0.5), (1.5, 0.5), (1.0, 1.5)], plane)["points"]
    [[1.5, 0.5], [1.5, 1.5], [0.5, 1.0]]
    """
    pixel = _vertices_to_pixel_space(vertices, plane["extent"], plane["shape"])
    dims = [plane["row_dim"], plane["column_dim"]]
    steps = [_index_step(plane["source_indices"][dim]) for dim in dims]
    if None in steps:
        return None
    # pixel holds (column, row) edge positions; the outline keeps (row, column).
    shown = (pixel[:, 1], pixel[:, 0])
    columns = [
        _shown_edge_to_source(position, start, step)
        for position, (start, step) in zip(shown, steps, strict=True)
    ]
    points = np.column_stack(columns)
    return {
        "dims": dims,
        "points": np.round(points, 6).tolist(),
        "slicing_dim": plane["slicing_dim"],
        "slice_index": plane["slice_index"],
        "erase": bool(erase),
    }


def outline_to_shown(
    outline: Mapping[str, Any], plane: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a stored outline in the axis units of the shown plane.

    Parameters
    ----------
    outline : mapping
        Outline as returned by :func:`outline_to_source`.
    plane : mapping
        The shown plane, with the keys described in
        :func:`outline_to_source`.

    Returns
    -------
    dict or None
        ``{"x": [...], "y": [...], "erase": bool}`` in the displayed axis
        units, the inverse of :func:`outline_to_source`. None if the outline
        lies in another slice or plane, or a shown axis is not an evenly
        stepped run of data-grid indices.
    """
    shown_dims = [plane["row_dim"], plane["column_dim"]]
    if (
        sorted(outline["dims"]) != sorted(shown_dims)
        or outline["slicing_dim"] != plane["slicing_dim"]
        or outline["slice_index"] != plane["slice_index"]
    ):
        return None
    points = np.asarray(outline["points"], dtype=float).reshape(-1, 2)
    edges = []
    for dim in shown_dims:
        step_info = _index_step(plane["source_indices"][dim])
        if step_info is None:
            return None
        start, step = step_info
        source = points[:, list(outline["dims"]).index(dim)]
        edges.append(source - start if step > 0 else start + 1 - source)
    row_pos, col_pos = edges
    xmin, xmax, ymin, ymax = plane["extent"]
    n_rows, n_columns = plane["shape"]
    x = xmin + col_pos * (xmax - xmin) / n_columns
    y = ymin + row_pos * (ymax - ymin) / n_rows
    return {
        "x": np.round(x, 6).tolist(),
        "y": np.round(y, 6).tolist(),
        "erase": bool(outline["erase"]),
    }


def drawn_area_boundaries(
    outlines: Sequence[Mapping[str, Any]],
    extent: Sequence[float],
    *,
    resolution: int = 600,
) -> list[dict[str, list[float]]]:
    """Return the boundary of the area left by the drawn outlines of one plane.

    The outlines are applied in the order they were drawn: an outline in add
    mode adds its area, one in erase mode removes it. Without an erasing
    outline the drawn polygons are returned unchanged. Otherwise the area is
    sampled on a fine grid over the shown plane and its boundary traced, so
    an erased part leaves neither its own line nor the cut-away part of an
    earlier outline.

    Parameters
    ----------
    outlines : sequence of mapping
        Outlines of the shown plane as returned by :func:`outline_to_shown`:
        ``x`` and ``y`` in the displayed axis units and ``erase``. Outlines
        of three or more points are closed polygons; two points are a line,
        which is kept when it added area and left out when it erased.
    extent : sequence of float
        Directed cell-edge bounds ``[xmin, xmax, ymin, ymax]`` of the shown
        plane, in the displayed axis units.
    resolution : int, optional
        Number of grid samples along the longer side of the plane when an
        outline erased area. The traced boundary is accurate to about one
        sample spacing.

    Returns
    -------
    list of dict
        Lines ``{"x": [...], "y": [...]}`` in the displayed axis units. A
        traced boundary is closed (its last point repeats the first).
        Deterministic; the inputs are not modified.
    """
    lines = [o for o in outlines if len(o["x"]) == 2 and not o["erase"]]
    polygons = [o for o in outlines if len(o["x"]) > 2]
    if not any(o["erase"] for o in polygons):
        return [{"x": list(o["x"]), "y": list(o["y"])} for o in polygons + lines]
    # Imported here: only planes with an erasing outline need the tracing.
    from matplotlib.path import Path
    from skimage.measure import approximate_polygon, find_contours

    x_low, x_high = sorted(extent[:2])
    y_low, y_high = sorted(extent[2:])
    width, height = x_high - x_low, y_high - y_low
    spacing = max(width, height) / resolution
    n_x, n_y = max(2, round(width / spacing)), max(2, round(height / spacing))
    step_x, step_y = width / n_x, height / n_y
    centres_x = x_low + (np.arange(n_x) + 0.5) * step_x
    centres_y = y_low + (np.arange(n_y) + 0.5) * step_y
    grid_x, grid_y = np.meshgrid(centres_x, centres_y)
    samples = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    area = np.zeros(n_x * n_y, dtype=bool)
    for outline in polygons:
        inside = Path(np.column_stack([outline["x"], outline["y"]])).contains_points(
            samples
        )
        area = area & ~inside if outline["erase"] else area | inside
    # A border of empty samples closes boundaries that touch the plane edge.
    padded = np.pad(area.reshape(n_y, n_x).astype(float), 1)
    boundaries = []
    for contour in find_contours(padded, 0.5):
        contour = approximate_polygon(contour, tolerance=0.5)
        boundaries.append(
            {
                "x": np.round(x_low + (contour[:, 1] - 0.5) * step_x, 6).tolist(),
                "y": np.round(y_low + (contour[:, 0] - 0.5) * step_y, 6).tolist(),
            }
        )
    return boundaries + [{"x": list(o["x"]), "y": list(o["y"])} for o in lines]


def _index_step(indices: np.ndarray) -> tuple[int, int] | None:
    """Return (first index, step) of an evenly stepped index run, else None."""
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return None
    if indices.size == 1:
        return int(indices[0]), 1
    steps = np.diff(indices)
    if steps[0] not in (1, -1) or np.any(steps != steps[0]):
        return None
    return int(indices[0]), int(steps[0])


def _shown_edge_to_source(position: np.ndarray, start: int, step: int) -> np.ndarray:
    """Map a shown edge position to a data-grid edge position.

    Shown voxel ``r`` is data-grid voxel ``start + step * r``. For step +1
    shown edge ``f`` is data-grid edge ``start + f``; for step -1 the axis is
    reversed and shown edge ``f`` is data-grid edge ``start + 1 - f``.
    """
    position = np.asarray(position, dtype=float)
    return start + position if step > 0 else start + 1 - position


def update_mask_measurements(
    mask: xr.DataArray, data: xr.DataArray | None, scan: Any | None
) -> None:
    """Store the weighted voxel count and the volume of a mask in its attrs.

    Parameters
    ----------
    mask : xarray.DataArray
        Mask coverage; its ``attrs`` are updated in place.
    data : xarray.DataArray or None
        Labelled source grid used for the voxel spacing; None gives no volume.
    scan : object or None
        Acquisition context for spacing not resolved from ``data``, see
        :func:`~hypermri.analysis.anatomical.roi.mask_volume_ml`.

    Returns
    -------
    None
        ``attrs["weighted_voxel_count"]`` is the sum of the coverage;
        ``attrs["volume_ml"]`` is the volume in mL, removed when the spacing
        is unknown.
    """
    mask.attrs["weighted_voxel_count"] = float(
        np.sum(_sanitize_mask_values(mask.values))
    )
    volume = (
        mask_volume_ml(mask, data, scan) if isinstance(data, xr.DataArray) else None
    )
    if volume is None:
        mask.attrs.pop("volume_ml", None)
    else:
        mask.attrs["volume_ml"] = volume


@dataclass(frozen=True)
class MaskedMeanCurve:
    """Coverage-weighted mean magnitude of a mask along one axis, ready to plot.

    Attributes
    ----------
    x : numpy.ndarray
        Axis values in ascending order, shape ``(n,)``: the coordinate of the
        secondary dimension, or positions ``0 .. n-1`` without a coordinate.
    y : numpy.ndarray
        Float64 means in the signal units of the data, shape ``(n,)``, in the
        order of ``x``. NaN where the mask covers no finite sample.
    sort_order : numpy.ndarray
        Positions along the secondary dimension in the order of ``x``.
    x_label : str
        ``"<dimension> [<unit>]"``, with ``idx`` when the unit is unknown.
    y_label : str
        ``"Mean magnitude"`` plus the signal unit in brackets when known.
    invert_x : bool
        True when the axis unit is ppm, which is drawn decreasing.
    """

    x: np.ndarray
    y: np.ndarray
    sort_order: np.ndarray
    x_label: str
    y_label: str
    invert_x: bool


def masked_mean_curve(
    data: xr.DataArray,
    mask: xr.DataArray,
    *,
    secondary_dim: str,
    selectors: Mapping[str, int],
    preview_slice: tuple[str, int] | None = None,
) -> MaskedMeanCurve:
    """Return the mean magnitude inside a mask along one axis for display.

    Both mask viewers show this curve as "ROI mean magnitude". The scientific
    reduction has one owner,
    :func:`~hypermri.analysis.anatomical.roi.raw_masked_mean`: the magnitude
    of the supplied samples per voxel, weighted by the coverage, without FFT
    or display processing. For time-domain CSI data along ``acquisition_idx``
    this is the mean FID magnitude. This function only adds the sorted axis
    and the labels.

    Parameters
    ----------
    data : xarray.DataArray
        Supplied samples; not modified.
    mask : xarray.DataArray
        Coverage in [0, 1] on the spatial grid of ``data``.
    secondary_dim : str
        Dimension along which the curve runs, e.g. ``acquisition_idx``.
    selectors : mapping of str to int
        Positional indices of the other non-spatial dimensions.
    preview_slice : tuple of str and int, optional
        Shown slicing dimension and index for a two-dimensional preview mask;
        None averages over the whole mask volume.

    Returns
    -------
    MaskedMeanCurve
        Axis, means and labels. Multi-dimensional coordinates of
        ``secondary_dim`` use their first ``n`` flattened values.

    Raises
    ------
    TypeError, ValueError, IndexError
        As :func:`~hypermri.analysis.anatomical.roi.raw_masked_mean`.
    """
    means = raw_masked_mean(
        data,
        mask,
        secondary_dim=secondary_dim,
        selectors=selectors,
        preview_slice=preview_slice,
    )
    n = data.sizes[secondary_dim]
    if secondary_dim in data.coords:
        coord = data.coords[secondary_dim]
        x_values = np.asarray(coord.values)
        if x_values.ndim > 1:
            x_values = x_values.ravel()[:n]
        sort_order = np.argsort(x_values)
        x_values = x_values[sort_order]
        unit = coord.attrs.get("units", coord.attrs.get("unit", "idx"))
    else:
        sort_order = np.arange(n)
        x_values = sort_order.astype(float)
        unit = "idx"
    signal_unit = data.attrs.get("units", data.attrs.get("unit"))
    return MaskedMeanCurve(
        x=x_values,
        y=means[sort_order],
        sort_order=sort_order,
        x_label=f"{secondary_dim} [{unit}]",
        y_label="Mean magnitude" + (f" [{signal_unit}]" if signal_unit else ""),
        invert_x=str(unit).lower() == "ppm",
    )


def plane_masked_mean(curves: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Return the coverage-weighted mean curve of the voxels of one plane.

    Every voxel's curve counts with its mask coverage, so partly covered
    edge voxels count partly. Voxels whose curve point is not finite are left
    out of that point's mean.

    Parameters
    ----------
    curves : numpy.ndarray
        Curves of the shown plane, shape (rows, columns, n), for example
        time courses. Not modified.
    coverage : numpy.ndarray
        Mask coverage in [0, 1] of the same plane, shape (rows, columns).

    Returns
    -------
    numpy.ndarray
        Mean curve of shape (n,) as float64; NaN where no covered voxel has
        a finite value.

    Raises
    ------
    ValueError
        If ``coverage`` does not have the shape of the first two axes of
        ``curves``.

    Examples
    --------
    >>> curves = np.array([[[1.0, 2.0], [3.0, 6.0]]])
    >>> plane_masked_mean(curves, np.array([[1.0, 0.5]])).tolist()
    [1.6666666666666667, 3.3333333333333335]
    """
    weights = np.asarray(coverage, dtype=float)
    if weights.shape != curves.shape[:2]:
        raise ValueError(
            f"plane_masked_mean: coverage shape {weights.shape} does not match "
            f"the plane {curves.shape[:2]}; use the coverage of the shown plane."
        )
    values = np.asarray(curves, dtype=float)
    finite = np.isfinite(values)
    weight = np.where(finite, weights[..., None], 0.0)
    total = weight.sum(axis=(0, 1))
    weighted = np.where(finite, values, 0.0) * weight
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, weighted.sum(axis=(0, 1)) / total, np.nan)


def coerce_mask(mask: xr.DataArray) -> xr.DataArray:
    """Return a float32 deep copy of a mask with coverage clipped to [0, 1].

    Parameters
    ----------
    mask : xarray.DataArray
        Stored or loaded mask; not modified.

    Returns
    -------
    xarray.DataArray
        Copy with the same dimensions, coordinates and attrs; NaN becomes 0.
    """
    copied = mask.astype(np.float32).copy(deep=True)
    copied.values = _sanitize_mask_values(copied.values)
    return copied


def masks_to_dataset(
    masks: Mapping[str, xr.DataArray],
    colors: Mapping[str, str],
    alphas: Mapping[str, float],
    *,
    data: xr.DataArray | None,
    scan: Any | None,
    outlines: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> xr.Dataset:
    """Collect named masks with their display style into one Dataset.

    Parameters
    ----------
    masks : mapping of str to xarray.DataArray
        Mask coverage by name. Their measurement attrs are refreshed in place.
    colors, alphas : mapping
        Display colour (``#rrggbb``) and opacity in [0, 1] by mask name;
        missing names use :data:`DEFAULT_MASK_COLOR` and
        :data:`DEFAULT_MASK_ALPHA`.
    data, scan : optional
        Source grid and acquisition context for the volume, see
        :func:`update_mask_measurements`.
    outlines : mapping, optional
        Drawn outlines by mask name, each a list of outlines as returned by
        :func:`outline_to_source`.

    Returns
    -------
    xarray.Dataset
        One float32 variable per mask with attrs ``color``, ``alpha``,
        ``weighted_voxel_count``, when known ``volume_ml``, and for masks
        with drawn outlines ``outlines``: a JSON string of the outline list,
        so the Dataset can be written to netCDF. The variables are copies of
        ``masks``.
    """
    variables = {}
    for name, mask in masks.items():
        update_mask_measurements(mask, data, scan)
        stored = coerce_mask(mask)
        stored.attrs["color"] = colors.get(name, DEFAULT_MASK_COLOR)
        stored.attrs["alpha"] = float(alphas.get(name, DEFAULT_MASK_ALPHA))
        drawn = list((outlines or {}).get(name, []))
        if drawn:
            stored.attrs["outlines"] = json.dumps(drawn)
        variables[name] = stored
    return xr.Dataset(variables)


def masks_from_dataset(
    dataset: xr.Dataset,
) -> tuple[
    dict[str, xr.DataArray],
    dict[str, str],
    dict[str, float],
    dict[str, list[dict[str, Any]]],
]:
    """Split a stored mask Dataset into masks, colours, opacities and outlines.

    Parameters
    ----------
    dataset : xarray.Dataset
        Masks as written by :func:`masks_to_dataset`; not modified.

    Returns
    -------
    tuple of dict
        Coerced mask copies (without the ``outlines`` attr), colours,
        opacities and drawn outlines, each keyed by mask name. Missing style
        attrs use the defaults; a mask without outlines gets an empty list.

    Raises
    ------
    ValueError
        If the ``outlines`` attr of a mask is not a JSON list.
    """
    masks, colors, alphas, outlines = {}, {}, {}, {}
    for variable in dataset.data_vars:
        name = str(variable)
        masks[name] = coerce_mask(dataset[name])
        colors[name] = dataset[name].attrs.get("color", DEFAULT_MASK_COLOR)
        alphas[name] = float(dataset[name].attrs.get("alpha", DEFAULT_MASK_ALPHA))
        outlines[name] = _read_outlines(name, masks[name].attrs.pop("outlines", "[]"))
    return masks, colors, alphas, outlines


def _read_outlines(name: str, text: Any) -> list[dict[str, Any]]:
    """Parse the JSON ``outlines`` attr of a stored mask."""
    try:
        drawn = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"masks_from_dataset: the 'outlines' attr of mask {name!r} is not "
            f"JSON ({error}); remove the attr to keep only the coverage."
        ) from error
    if not isinstance(drawn, list):
        raise ValueError(
            f"masks_from_dataset: the 'outlines' attr of mask {name!r} must be "
            "a JSON list; remove the attr to keep only the coverage."
        )
    return drawn


def _vertices_to_pixel_space(
    vertices: Sequence[tuple[float, float]],
    extent: Sequence[float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Map plot coordinates to pixel-edge coordinates; voxel j spans [j, j+1]."""
    disp_h, disp_w = shape
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / disp_w
    dy = (ymax - ymin) / disp_h
    if dx == 0 or dy == 0:
        raise ValueError("Spatial extent has zero width or height.")
    verts = np.asarray(vertices, dtype=float)
    col_pos = (verts[:, 0] - xmin) / dx
    row_pos = (verts[:, 1] - ymin) / dy
    return np.column_stack([col_pos, row_pos])


def _polygon_area(vertices: np.ndarray) -> float:
    """Return the absolute area of a polygon in pixel units."""
    poly = np.asarray(vertices, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_polygon_to_boundary(
    vertices: np.ndarray, *, axis: int, bound: float, keep_lower: bool
) -> np.ndarray:
    """Clip a polygon against a single axis-aligned half-plane."""
    poly = np.asarray(vertices, dtype=float)
    if poly.ndim != 2 or poly.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    eps = 1e-12

    def inside(point: np.ndarray) -> bool:
        coord = float(point[axis])
        if keep_lower:
            return coord <= bound + eps
        return coord >= bound - eps

    def intersect(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        delta = float(end[axis] - start[axis])
        if abs(delta) <= eps:
            clipped = start.copy()
            clipped[axis] = bound
            return clipped
        t = (bound - float(start[axis])) / delta
        return start + t * (end - start)

    output: list[np.ndarray] = []
    prev = poly[-1]
    prev_inside = inside(prev)
    for curr in poly:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                output.append(intersect(prev, curr))
            output.append(curr.copy())
        elif prev_inside:
            output.append(intersect(prev, curr))
        prev = curr
        prev_inside = curr_inside

    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _clip_polygon_to_rect(
    vertices: np.ndarray, *, left: float, right: float, bottom: float, top: float
) -> np.ndarray:
    """Clip a polygon to an axis-aligned pixel cell."""
    clipped = _clip_polygon_to_boundary(vertices, axis=0, bound=left, keep_lower=False)
    clipped = _clip_polygon_to_boundary(clipped, axis=0, bound=right, keep_lower=True)
    clipped = _clip_polygon_to_boundary(clipped, axis=1, bound=bottom, keep_lower=False)
    return _clip_polygon_to_boundary(clipped, axis=1, bound=top, keep_lower=True)


def _closed_polygon_coverage(
    polygon_verts: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Compute exact polygon-cell overlap for a closed polygon."""
    coverage = np.zeros(shape, dtype=np.float32)
    poly = np.asarray(polygon_verts, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return coverage
    if _polygon_area(poly) <= 1e-12:
        return coverage

    disp_h, disp_w = shape
    min_col = max(int(np.floor(np.min(poly[:, 0]))), 0)
    max_col = min(int(np.ceil(np.max(poly[:, 0]))), disp_w)
    min_row = max(int(np.floor(np.min(poly[:, 1]))), 0)
    max_row = min(int(np.ceil(np.max(poly[:, 1]))), disp_h)

    for row in range(min_row, max_row):
        for col in range(min_col, max_col):
            clipped = _clip_polygon_to_rect(
                poly,
                left=float(col),
                right=float(col + 1),
                bottom=float(row),
                top=float(row + 1),
            )
            area = _polygon_area(clipped)
            if area > 0:
                coverage[row, col] = float(area)
    return coverage


def _polyline_coverage(
    poly_verts: np.ndarray, *, shape: tuple[int, int], subpixel_resolution: int = 8
) -> np.ndarray:
    """Approximate one-voxel-wide coverage for open line segments."""
    disp_h, disp_w = shape
    offsets = (np.arange(subpixel_resolution, dtype=float) + 0.5) / float(
        subpixel_resolution
    )
    rows = (
        np.arange(disp_h, dtype=float)[:, None, None, None]
        + offsets[None, :, None, None]
    )
    cols = (
        np.arange(disp_w, dtype=float)[None, None, :, None]
        + offsets[None, None, None, :]
    )
    full = (disp_h, subpixel_resolution, disp_w, subpixel_resolution)
    sample_x = np.broadcast_to(cols, full).ravel()
    sample_y = np.broadcast_to(rows, full).ravel()

    line_covered = np.zeros(sample_x.shape, dtype=bool)
    stroke_radius_sq = 0.25
    for start, end in zip(poly_verts[:-1], poly_verts[1:], strict=False):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            dist_sq = (sample_x - start[0]) ** 2 + (sample_y - start[1]) ** 2
        else:
            t = ((sample_x - start[0]) * dx + (sample_y - start[1]) * dy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj_x = start[0] + t * dx
            proj_y = start[1] + t * dy
            dist_sq = (sample_x - proj_x) ** 2 + (sample_y - proj_y) ** 2
        line_covered |= dist_sq <= stroke_radius_sq

    line_coverage = line_covered.reshape(full).mean(axis=(1, 3))
    return np.asarray(line_coverage, dtype=np.float32)


__all__ = [
    "DEFAULT_MASK_ALPHA",
    "DEFAULT_MASK_COLOR",
    "MASK_DIMS",
    "MaskedMeanCurve",
    "coerce_mask",
    "combine_plane_coverage",
    "empty_mask",
    "mask_dims",
    "masked_mean_curve",
    "plane_masked_mean",
    "masks_from_dataset",
    "masks_to_dataset",
    "outline_to_shown",
    "outline_to_source",
    "polygon_coverage",
    "shown_coverage_to_source",
    "source_coverage_to_shown",
    "transfer_coverage",
    "update_mask_measurements",
]
