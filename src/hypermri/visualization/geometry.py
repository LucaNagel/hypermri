"""Pure scanner-space display geometry shared by the viewer.

Labelled coordinates, voxel centres, physical extents and nearest-plane
selection follow the scanner-coordinate grid and alignment conventions of
:mod:`hypermri.utils.coordinates`. Functions construct
no widgets, perform no fitting and create no caches. Signal storage precision,
coordinate units/direction and input arrays are preserved; coordinate arithmetic
uses NumPy float arrays as before. The render data preparation supplies scan
metadata explicitly and owns state updates and alignment lifetime.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from hypermri.utils.coordinates.alignment import get_2d_extent_from_alignment
from hypermri.utils.coordinates.grids import assign_xyz_coordinates, has_xyz_coordinates
from hypermri.visualization.state import PlotState

_SPATIAL_DIM_LABELS = {"read_idx": "Read", "phase_idx": "Phase", "slice_idx": "Slice"}
_XYZ_COORD_NAMES = ("x", "y", "z")
_VALID_COORDINATE_MODES = {"rps", "xyz"}


def read_phase_to_plot_coordinates(
    read_idx: Any,
    phase_idx: Any,
    data: xr.DataArray,
    *,
    rotation_k: int = 0,
    flip_ud: bool = False,
    flip_lr: bool = False,
) -> tuple[int, int]:
    """Map labelled read/phase indices to ``matplotlib`` image coordinates.

    The mapping matches an image made from ``data.squeeze()`` by applying
    ``np.rot90(image, k=rotation_k)``, then optionally ``np.flipud`` and
    ``np.fliplr`` in that order.  The order of ``read_idx`` and ``phase_idx``
    in ``data`` is detected automatically, so a stored
    ``(phase_idx, read_idx)`` image needs no special case.

    Parameters
    ----------
    read_idx : hashable
        Label of the voxel along the ``read_idx`` coordinate.  With the usual
        zero-based index coordinate this is its positional index.
    phase_idx : hashable
        Label of the voxel along the ``phase_idx`` coordinate.  With the usual
        zero-based index coordinate this is its positional index.
    data : xarray.DataArray
        Array used to make the image.  After singleton dimensions are removed,
        it must have exactly the dimensions ``read_idx`` and ``phase_idx``.
        Coordinate labels are used to find their positions.
    rotation_k : int, default: 0
        Number of 90-degree counter-clockwise rotations, with the same meaning
        as the ``k`` argument of :func:`numpy.rot90`.  Negative values rotate
        clockwise.
    flip_ud : bool, default: False
        Apply the same vertical flip as ``np.flipud`` after rotation.
    flip_lr : bool, default: False
        Apply the same horizontal flip as ``np.fliplr`` after rotation and an
        optional vertical flip.

    Returns
    -------
    tuple of int
        ``(x, y)`` positional coordinates for ``Axes.plot(x, y)`` on the
        transformed image.

    Raises
    ------
    ValueError
        If the array is not a two-dimensional read/phase image after squeezing,
        or either supplied coordinate label is absent or non-unique.

    Examples
    --------
    >>> image = xr.DataArray(
    ...     np.zeros((64, 64)), dims=("read_idx", "phase_idx")
    ... )
    >>> read_phase_to_plot_coordinates(40, 16, image, rotation_k=-1)
    (23, 16)

    Notes
    -----
    This function maps array indices, not physical scanner coordinates.  It is
    pure and does not modify ``data`` or transform the displayed image.
    """
    image = data.squeeze()
    required_dims = {"read_idx", "phase_idx"}
    if set(image.dims) != required_dims or image.ndim != 2:
        raise ValueError(
            "read_phase_to_plot_coordinates requires a DataArray that becomes "
            "a two-dimensional read_idx/phase_idx image after squeezing. "
            "Select every non-singleton dimension before calling it."
        )

    read_position = _coordinate_position(image, "read_idx", read_idx)
    phase_position = _coordinate_position(image, "phase_idx", phase_idx)
    row_dim, column_dim = image.dims
    positions = {"read_idx": read_position, "phase_idx": phase_position}
    row, column = positions[row_dim], positions[column_dim]
    rows, columns = image.shape

    match rotation_k % 4:
        case 0:
            output_rows, output_columns = rows, columns
        case 1:
            row, column = columns - 1 - column, row
            output_rows, output_columns = columns, rows
        case 2:
            row, column = rows - 1 - row, columns - 1 - column
            output_rows, output_columns = rows, columns
        case 3:
            row, column = column, rows - 1 - row
            output_rows, output_columns = columns, rows

    if flip_ud:
        row = output_rows - 1 - row
    if flip_lr:
        column = output_columns - 1 - column
    return column, row


def _coordinate_position(data: xr.DataArray, dim: str, label: Any) -> int:
    """Return one unique positional index for a labelled dimension."""
    try:
        location = data.get_index(dim).get_loc(label)
    except KeyError as error:
        raise ValueError(
            f"{dim} label {label!r} is not present in the image."
        ) from error
    if not isinstance(location, (int, np.integer)):
        raise ValueError(
            f"{dim} label {label!r} is not unique and cannot identify one pixel."
        )
    return int(location)


def display_voxel_centres(
    extent: list[float] | None, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell centres used to draw a spatial heatmap.

    Parameters
    ----------
    extent : list of float or None
        Display cell-edge bounds [xmin, xmax, ymin, ymax], in plot-axis units.
        Descending bounds retain their direction. None uses positional centres.
    shape : tuple of int
        Positive displayed (row, column) counts after orientation and rotation.

    Returns
    -------
    tuple of numpy.ndarray
        Float64 x and y centres of lengths columns and rows. This follows the
        rendered extent, including nonuniform source coordinates represented by
        that extent; it does not introduce a new resampling/geometry convention.

    Raises
    ------
    ValueError
        For empty shapes or non-finite/degenerate bounds.

    Notes
    -----
    Pure and deterministic; input arrays, units and coordinate order are intact.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("Display voxel centres require a nonempty spatial plane.")
    if extent is None:
        return np.arange(w, dtype=float), np.arange(h, dtype=float)
    bounds = np.asarray(extent, dtype=float)
    if (
        bounds.shape != (4,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] == bounds[1]
        or bounds[2] == bounds[3]
    ):
        raise ValueError("Display extent must have finite nonzero width and height.")
    xmin, xmax, ymin, ymax = bounds
    return xmin + (np.arange(w) + 0.5) * (xmax - xmin) / w, ymin + (
        np.arange(h) + 0.5
    ) * (ymax - ymin) / h


def snap_display_point(
    x: float, y: float, extent: list[float], shape: tuple[int, int]
) -> tuple[float, float] | None:
    """Place an in-plane click at its nearest rendered voxel centre.

    Parameters
    ----------
    x, y : float
        Click in the display axes' physical units or positional indices.
    extent : list of float
        Directed cell-edge bounds [xmin, xmax, ymin, ymax] of the drawn plane.
    shape : tuple of int
        Displayed (row, column) counts, including any rotation.

    Returns
    -------
    tuple of float or None
        Nearest x/y cell centre, or None for non-finite/outside-plane clicks.
        Ties select the first position in display order.

    Raises
    ------
    ValueError
        For invalid shape or bounds, as in display_voxel_centres.

    Notes
    -----
    No widgets, state updates or input mutation. Numeric vertex editing can
    retain subvoxel coordinates independently of this click-placement policy.
    """
    xs, ys = display_voxel_centres(extent, shape)
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if not min(extent[:2]) <= x <= max(extent[:2]) or not min(extent[2:]) <= y <= max(
        extent[2:]
    ):
        return None
    return float(xs[np.argmin(np.abs(xs - x))]), float(ys[np.argmin(np.abs(ys - y))])


def mask_coverage_in_data_order(
    coverage: np.ndarray,
    unrotated_dims: list[str],
    target_dims: list[str],
    rotation: int,
    *,
    display_coords: dict[str, np.ndarray] | None = None,
    source_coords: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Restore drawn coverage to the acquired plane's labelled index order.

    Parameters
    ----------
    coverage : numpy.ndarray
        Two-dimensional fractional coverage in rendered row/column order.
    unrotated_dims : list of str
        Ordered [y, x] plane dimensions before display rotation.
    target_dims : list of str
        Same two dimensions in stored mask order.
    rotation : int
        Clockwise display rotation in degrees, a multiple of 90, as applied
        by the viewer (``rotation=90`` shows ``np.rot90(plane, k=-1)``);
        undone before reordering.
    display_coords, source_coords : dict of str to numpy.ndarray or None, optional
        Displayed and acquired one-dimensional coordinate lines, in their
        original units. Coordinates restore any reversal from viewer orientation.
        Missing lines retain positional geometry. Present lines must describe
        exactly the same labelled grid, with unique coordinate values.

    Returns
    -------
    numpy.ndarray
        Coverage in target dimension and coordinate order, with unchanged dtype.
        Inputs are not mutated; the result may share their storage.

    Raises
    ------
    ValueError
        For invalid dimensions/rotation or mismatched labelled coordinate grids.

    Notes
    -----
    Pure and deterministic. This maps the drawn primary plane only, without
    anatomy alignment, mask projection or a new coordinate/resampling policy.

    Examples
    --------
    >>> stored = np.array([[1., 0.], [0.5, 0.]])
    >>> dims = ["read_idx", "phase_idx"]
    >>> mask_coverage_in_data_order(np.rot90(stored, k=-1), dims, dims, 90)
    array([[1. , 0. ],
           [0.5, 0. ]])
    """
    if (
        np.ndim(coverage) != 2
        or len(target_dims) != 2
        or len(unrotated_dims) != 2
        or len(set(unrotated_dims)) != 2
        or set(target_dims) != set(unrotated_dims)
        or rotation % 90
    ):
        raise ValueError(
            "Mask plane requires two matching dimensions and a 90-degree rotation."
        )
    restored = np.rot90(coverage, k=(rotation % 360) // 90)
    for axis, dim in enumerate(unrotated_dims):
        displayed = (display_coords or {}).get(dim)
        source = (source_coords or {}).get(dim)
        if displayed is None or source is None:
            continue
        displayed, source = np.asarray(displayed), np.asarray(source)
        if displayed.ndim != 1 or source.ndim != 1:
            continue
        if (
            rotation % 360 == 180
            or (rotation % 360 == 90 and axis == 0)
            or (rotation % 360 == 270 and axis == 1)
        ):
            displayed = displayed[::-1]
        if len(displayed) != restored.shape[axis] or len(source) != len(displayed):
            raise ValueError(
                f"Mask coordinate sizes for {dim} do not match the drawn plane."
            )
        matching = source[:, None] == displayed[None, :]
        if not np.all(matching.sum(axis=0) == 1) or not np.all(
            matching.sum(axis=1) == 1
        ):
            raise ValueError(
                f"Mask coordinates for {dim} do not match the acquired grid."
            )
        restored = np.take(restored, matching.argmax(axis=1), axis=axis)
    return restored if target_dims == unrotated_dims else restored.T


def mask_coverage_in_display_order(
    coverage: np.ndarray,
    source_dims: list[str],
    unrotated_dims: list[str],
    rotation: int,
    *,
    display_coords: dict[str, np.ndarray] | None = None,
    source_coords: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Place stored mask coverage on the viewer's directed and rotated plane.

    Parameters
    ----------
    coverage : numpy.ndarray
        Two-dimensional fractional coverage in source_dims order.
    source_dims : list of str
        Stored [row, column] dimension names.
    unrotated_dims : list of str
        The same two names in the viewer's unrotated [y, x] order.
    rotation : int
        Clockwise display rotation in degrees, a multiple of 90, as applied
        by the viewer (``rotation=90`` shows ``np.rot90(plane, k=-1)``).
    display_coords, source_coords : dict of str to numpy.ndarray or None, optional
        Displayed and acquired coordinate lines in their original units; missing
        lines retain positional geometry. Present lines must match exactly.

    Returns
    -------
    numpy.ndarray
        Coverage in rendered row/column order, retaining dtype and values.

    Raises
    ------
    ValueError
        For mismatched plane dimensions, rotation or labelled grids.

    Notes
    -----
    Deterministic inverse of mask_coverage_in_data_order; no widgets, projection,
    anatomy alignment, input mutation or interpolation is introduced.

    Examples
    --------
    >>> dims = ["read_idx", "phase_idx"]
    >>> mask_coverage_in_display_order(np.array([[1., 0.], [0.5, 0.]]), dims, dims, 90)
    array([[0.5, 1. ],
           [0. , 0. ]])
    """
    if rotation % 90:
        raise ValueError("Mask plane requires a 90-degree rotation.")
    displayed = dict(display_coords or {})
    for axis, dim in enumerate(unrotated_dims):
        if dim in displayed and (
            rotation % 360 == 180
            or (rotation % 360 == 90 and axis == 0)
            or (rotation % 360 == 270 and axis == 1)
        ):
            displayed[dim] = np.asarray(displayed[dim])[::-1]
    oriented = mask_coverage_in_data_order(
        coverage,
        source_dims,
        unrotated_dims,
        0,
        display_coords=source_coords,
        source_coords=displayed,
    )
    return np.rot90(oriented, k=-((rotation % 360) // 90))


def extract_1d_coord(data: xr.DataArray, dim_name: str) -> np.ndarray:
    """Extract a labelled coordinate line without changing its direction.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled array; dimension coordinates can be one-dimensional or meshgrids.
    dim_name : str
        Existing dimension to extract. Other coordinate dimensions are fixed at
        positional index zero; missing coordinates give integer positional indices.

    Returns
    -------
    numpy.ndarray
        Coordinate line of length data.sizes[dim_name], retaining input units,
        orientation and dtype, or integer indices without a coordinate.

    Raises
    ------
    KeyError
        If a missing coordinate also has no corresponding dimension.

    Notes
    -----
    Eager and deterministic; reads coordinate values only, with no mutation/cache.

    Examples
    --------
    >>> data = xr.DataArray([1, 2], dims="read_idx", coords={"read_idx": [2., 0.]})
    >>> extract_1d_coord(data, "read_idx").tolist()
    [2.0, 0.0]
    """
    if dim_name not in data.coords:
        return np.arange(data.sizes[dim_name])

    coord = data.coords[dim_name]
    if coord.ndim <= 1:
        return coord.values

    # Fix all dims except dim_name to index 0
    selector = {d: 0 for d in coord.dims if d != dim_name}
    result = coord.isel(selector).values
    return result


def _looks_like_default_index_axis(values: np.ndarray) -> bool:
    """Return True when coordinate values appear to be plain sequential indices."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size <= 1:
        return False
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError):
        return False
    zero_based = np.arange(len(numeric), dtype=np.float64)
    one_based = np.arange(1, len(numeric) + 1, dtype=np.float64)
    return bool(
        np.allclose(numeric, zero_based, rtol=0.01, atol=1e-9)
        or np.allclose(numeric, one_based, rtol=0.01, atol=1e-9)
    )


def get_spatial_dim_label(dim_name: str) -> str:
    """Return the existing human-readable label for a spatial dimension.

    Parameters
    ----------
    dim_name : str
        Dimension name; read_idx, phase_idx and slice_idx use established labels.

    Returns
    -------
    str
        Read, Phase or Slice, otherwise the title-cased name with underscores
        replaced by spaces. No data access, mutation, units or cache are involved.
    """
    return _SPATIAL_DIM_LABELS.get(dim_name, dim_name.replace("_", " ").title())


def format_spatial_axis_label(dim_name: str, unit: str | None = None) -> str:
    """Format the existing spatial-dimension label with a declared unit.

    Parameters
    ----------
    dim_name : str
        Existing labelled spatial dimension.
    unit : str or None, optional
        Display unit such as mm; None or an empty string leaves the label unitless.

    Returns
    -------
    str
        Label with its unit in square brackets when supplied. Deterministic and
        eager; no data conversion, mutation, coordinate reversal or cache occurs.
    """
    base_label = get_spatial_dim_label(dim_name)
    if not unit or unit == "idx":
        return base_label
    return f"{base_label} [{unit}]"


def normalize_coordinate_mode(mode: Any) -> str:
    """Return the existing coordinate-mode identifier with legacy fallback.

    Parameters
    ----------
    value : str or None
        Coordinate-mode request; case and surrounding whitespace are ignored.

    Returns
    -------
    str
        xyz or rps; unknown/omitted identifiers retain the existing rps fallback.
        This utility does not transform samples or coordinates and keeps no cache.

    Notes
    -----
    The canonical plot entry validates supplied modes before calling this utility;
    its unsupported requests raise rather than using this legacy state fallback.
    """
    normalized = str(mode or "rps").strip().lower()
    return normalized if normalized in _VALID_COORDINATE_MODES else "rps"


def format_xyz_axis_label(coord_name: str, unit: str | None = "mm") -> str:
    """Format a scanner-coordinate label with its physical unit.

    Parameters
    ----------
    coord_name : str
        Scanner coordinate name, normally x, y or z.
    unit : str, optional
        Declared physical unit; existing default mm. An empty string omits it.

    Returns
    -------
    str
        Coordinate name with the unit in brackets. No numerical unit conversion,
        data access, mutation or cache occurs.
    """
    if not unit:
        return str(coord_name)
    return f"{coord_name} [{unit}]"


def _nice_scalebar_length(target: float) -> float:
    """Pick a visually clean scalebar length close to the requested target."""
    if not np.isfinite(target) or target <= 0:
        return 0.0

    magnitude = 10 ** np.floor(np.log10(target))
    candidates = magnitude * np.array([1, 2, 5, 10], dtype=float)
    valid = candidates[candidates <= target]
    if valid.size:
        return float(valid[-1])
    return float(candidates[0])


def _uses_xyz_coordinates(state: PlotState) -> bool:
    """Return True when spatial plotting should use scanner-space XYZ coords."""
    return normalize_coordinate_mode(state.spatial.coordinate_mode) == "xyz"


def _ensure_xyz_coordinates(
    data: xr.DataArray, state: PlotState, *, scan: Any | None = None
) -> xr.DataArray:
    """Attach scanner-space coordinates on demand when geometry is available."""
    if (
        not _uses_xyz_coordinates(state)
        or not isinstance(data, xr.DataArray)
        or has_xyz_coordinates(data)
    ):
        return data

    source_obj = scan
    if source_obj is None:
        return data

    try:
        return assign_xyz_coordinates(data, scan=source_obj)
    except Exception:
        return data


def _coord_unit(coord: xr.DataArray, default: str = "mm") -> str:
    """Return the declared coordinate unit."""
    return str(coord.attrs.get("unit") or coord.attrs.get("units") or default)


def _coord_values_in_display_order(
    data: xr.DataArray, coord_name: str, spatial_dims: list[str]
) -> np.ndarray | None:
    """Return a 2D coordinate array ordered like the displayed image."""
    if coord_name not in data.coords or len(spatial_dims) != 2:
        return None

    coord = data.coords[coord_name]
    if not all(dim in coord.dims for dim in spatial_dims):
        return None

    try:
        coord = coord.transpose(*spatial_dims)
    except Exception:
        return None

    values = np.asarray(coord.values, dtype=float)
    if values.ndim != 2 or values.size == 0:
        return None
    return values


def _axis_line_values(values: np.ndarray, axis: int) -> np.ndarray | None:
    """Extract representative coordinate values along one display axis."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if axis == 1:
        return np.ravel(arr[arr.shape[0] // 2, :])
    if axis == 0:
        return np.ravel(arr[:, arr.shape[1] // 2])
    return None


def _finite_span(values: np.ndarray | None) -> float:
    """Return the finite coordinate span, or -1 when unavailable."""
    if values is None:
        return -1.0
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0 if finite.size == 1 else -1.0
    return float(np.nanmax(finite) - np.nanmin(finite))


def _line_is_descending(values: np.ndarray | None) -> bool:
    """Return True when finite coordinate values run high-to-low."""
    if values is None:
        return False
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return False
    return bool(finite[-1] < finite[0])


def _xyz_display_coordinate_data(
    spatial_slice: xr.DataArray, spatial_dims: list[str]
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, float]]] | None:
    """Return XYZ arrays and their spans along the two display axes."""
    if len(spatial_dims) != 2 or not has_xyz_coordinates(spatial_slice):
        return None

    coord_arrays: dict[str, np.ndarray] = {}
    spans: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for coord_name in _XYZ_COORD_NAMES:
        values = _coord_values_in_display_order(
            spatial_slice,
            coord_name,
            spatial_dims,
        )
        if values is None:
            continue
        coord_arrays[coord_name] = values
        for axis in (0, 1):
            spans[axis][coord_name] = _finite_span(_axis_line_values(values, axis=axis))

    return (coord_arrays, spans) if coord_arrays else None


def _select_xyz_display_coords(
    spatial_slice: xr.DataArray,
    spatial_dims: list[str],
    *,
    geometry_data: xr.DataArray | None = None,
    scan: Any | None = None,
) -> tuple[str, str, dict[str, np.ndarray]] | None:
    """Choose scanner coordinate names for display ``(x_axis, y_axis)``."""
    coordinate_data = _xyz_display_coordinate_data(
        spatial_slice,
        spatial_dims,
    )
    if coordinate_data is None:
        return None
    coord_arrays, spans = coordinate_data

    geometry_data = geometry_data if geometry_data is not None else spatial_slice
    if len(spatial_dims) == 2:
        mapped_x = _dominant_xyz_coord_for_dim(
            geometry_data,
            spatial_dims[1],
            scan=scan,
        )
        mapped_y = _dominant_xyz_coord_for_dim(
            geometry_data,
            spatial_dims[0],
            scan=scan,
        )
        if (
            mapped_x in coord_arrays
            and mapped_y in coord_arrays
            and mapped_x != mapped_y
        ):
            return mapped_x, mapped_y, coord_arrays

    x_candidates = {name: span for name, span in spans[1].items() if np.isfinite(span)}
    y_candidates = {name: span for name, span in spans[0].items() if np.isfinite(span)}
    if not x_candidates or not y_candidates:
        return None

    x_coord = max(x_candidates, key=x_candidates.get)
    y_candidates_without_x = {
        name: span for name, span in y_candidates.items() if name != x_coord
    }
    y_coord = max(
        y_candidates_without_x or y_candidates,
        key=(y_candidates_without_x or y_candidates).get,
    )

    if x_candidates[x_coord] <= 0 and y_candidates[y_coord] <= 0:
        return None
    return x_coord, y_coord, coord_arrays


def _xyz_axis_coord_by_display_axis(
    spatial_slice: xr.DataArray,
    spatial_dims: list[str],
    *,
    geometry_data: xr.DataArray | None = None,
    scan: Any | None = None,
) -> tuple[str, str] | None:
    """Return scanner coords varying along display rows and columns."""
    coordinate_data = _xyz_display_coordinate_data(
        spatial_slice,
        spatial_dims,
    )
    if coordinate_data is None:
        return None
    _, axis_spans = coordinate_data

    geometry_data = geometry_data if geometry_data is not None else spatial_slice
    if len(spatial_dims) == 2:
        row_coord = _dominant_xyz_coord_for_dim(
            geometry_data,
            spatial_dims[0],
            scan=scan,
        )
        col_coord = _dominant_xyz_coord_for_dim(
            geometry_data,
            spatial_dims[1],
            scan=scan,
        )
        if row_coord is not None and col_coord is not None and row_coord != col_coord:
            return row_coord, col_coord

    if not axis_spans[0] or not axis_spans[1]:
        return None

    row_coord = max(axis_spans[0], key=axis_spans[0].get)
    col_coord = max(axis_spans[1], key=axis_spans[1].get)
    if axis_spans[0][row_coord] <= 0 or axis_spans[1][col_coord] <= 0:
        return None
    if row_coord == col_coord:
        return None
    return row_coord, col_coord


def _canonical_xyz_display_pair(row_coord: str, col_coord: str) -> tuple[str, str]:
    """Return canonical ``(vertical, horizontal)`` scanner coordinates."""
    varying = {row_coord, col_coord}
    ordered = [coord for coord in _XYZ_COORD_NAMES if coord in varying]
    if len(ordered) != 2:
        return row_coord, col_coord
    horizontal = ordered[0]
    vertical = ordered[1]
    return vertical, horizontal


def _orient_spatial_slice_for_xyz(
    spatial_slice: xr.DataArray,
    spatial_dims: list[str],
    state: PlotState,
    *,
    geometry_data: xr.DataArray | None = None,
    scan: Any | None = None,
) -> tuple[xr.DataArray, list[str]]:
    """Orient an already selected image plane in scanner coordinates.

    Parameters
    ----------
    spatial_slice : xarray.DataArray
        Selected plane with labelled scanner x/y/z coordinate grids in mm.
        Further non-spatial dimensions, such as acquisition_idx of a spectral
        block, follow the two spatial dimensions unchanged. Signal
        units/dtype remain untouched.
    spatial_dims : list of str
        Ordered displayed row/column dimensions of the supplied plane.
    state : PlotState
        Existing coordinate mode and requested display rotation. Canonical XYZ
        orientation is applied only in xyz mode with zero explicit rotation.
    geometry_data : xarray.DataArray or None, optional
        Full source grid for direction inference, including singleton dimensions.
        None uses the selected plane.
    scan : object or None, optional
        Explicit acquisition context with resolution_mm and slice_thickness_mm
        metadata in mm when coordinate steps cannot provide a direction/spacing.

    Returns
    -------
    xarray.DataArray
        The original plane, or its existing coordinate-driven transpose/reversal.
        Labels, attrs, coordinate direction relative to data and storage dtype are
        preserved; no resampling, interpolation or signal conversion occurs.
    list of str
        Resulting ordered row/column dimension names. Missing/unsupported geometry
        or a requested rotation leaves the input orientation unchanged.

    Notes
    -----
    Eager, deterministic and independent of widgets. Coordinate arithmetic retains
    NumPy float precision. Inputs and state are not mutated; no cache is created.
    Scanner-coordinate interpretation remains with :mod:`hypermri.utils.coordinates`. This stage
    preserves the existing behavior of separately requested anatomy alignment.

    Examples
    --------
    >>> oriented, dims = orient_xyz_slice(plane, list(plane.dims), state)  # doctest: +SKIP
    """
    if (
        not _uses_xyz_coordinates(state)
        or state.spatial.rotation % 360 != 0
        or len(spatial_dims) != 2
    ):
        return spatial_slice, spatial_dims

    axis_coords = _xyz_axis_coord_by_display_axis(
        spatial_slice,
        spatial_dims,
        geometry_data=geometry_data,
        scan=scan,
    )
    if axis_coords is None:
        return spatial_slice, spatial_dims

    row_coord, col_coord = axis_coords
    target_vertical, target_horizontal = _canonical_xyz_display_pair(
        row_coord,
        col_coord,
    )

    if row_coord == target_vertical and col_coord == target_horizontal:
        oriented_slice = spatial_slice
        oriented_dims = list(spatial_dims)
    elif row_coord == target_horizontal and col_coord == target_vertical:
        transposed_dims = [spatial_dims[1], spatial_dims[0]]
        try:
            oriented_slice = spatial_slice.transpose(*transposed_dims, ...)
            oriented_dims = transposed_dims
        except Exception:
            return spatial_slice, spatial_dims
    else:
        return spatial_slice, spatial_dims

    flip_selectors = {}
    row_values = _coord_values_in_display_order(
        oriented_slice,
        target_vertical,
        oriented_dims,
    )
    if _line_is_descending(_axis_line_values(row_values, axis=0)):
        flip_selectors[oriented_dims[0]] = slice(None, None, -1)

    col_values = _coord_values_in_display_order(
        oriented_slice,
        target_horizontal,
        oriented_dims,
    )
    if _line_is_descending(_axis_line_values(col_values, axis=1)):
        flip_selectors[oriented_dims[1]] = slice(None, None, -1)

    if flip_selectors:
        try:
            oriented_slice = oriented_slice.isel(flip_selectors)
        except Exception:
            return spatial_slice, spatial_dims

    return oriented_slice, oriented_dims


def _axis_extent_from_coord_values(
    values: np.ndarray, *, axis: int, fallback_step: float | None = None
) -> list[float] | None:
    """Calculate imshow edge bounds from scanner coordinate centers."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    line = _axis_line_values(arr, axis=axis)
    if line is None:
        return None
    line = np.asarray(line, dtype=float)
    line = line[np.isfinite(line)]

    if line.size >= 2:
        diffs = np.diff(line)
        diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
        step = float(np.nanmedian(np.abs(diffs))) if diffs.size else 0.0
        increasing = bool(line[-1] >= line[0])
    else:
        step = 0.0
        increasing = True

    if step <= 0 and fallback_step is not None:
        try:
            candidate_step = abs(float(fallback_step))
        except (TypeError, ValueError):
            candidate_step = 0.0
        if np.isfinite(candidate_step) and candidate_step > 0:
            step = candidate_step
    if step <= 0:
        step = 1.0

    half_step = step / 2.0
    lower = float(np.nanmin(finite))
    upper = float(np.nanmax(finite))
    if increasing:
        return [lower - half_step, upper + half_step]
    return [upper + half_step, lower - half_step]


def _calculate_xyz_spatial_extent(
    spatial_slice: xr.DataArray,
    spatial_dims: list[str],
    *,
    geometry_data: xr.DataArray | None = None,
    scan: Any | None = None,
) -> tuple[list[float], str, str, str] | None:
    """Calculate scanner-space image bounds from labelled voxel centres.

    Parameters
    ----------
    spatial_slice : xarray.DataArray
        Two-dimensional displayed image with x/y/z coordinate grids in mm.
    spatial_dims : list of str
        Ordered row/column dimension names matching the displayed image.
    geometry_data : xarray.DataArray or None, optional
        Full source grid for direction/spacing inference; None uses spatial_slice.
    scan : object or None, optional
        Acquisition context providing resolution_mm or slice_thickness_mm in mm.
        Coordinate-declared spacing retains precedence.

    Returns
    -------
    tuple or None
        (extent, origin, x_label, y_label), where extent holds horizontal then
        vertical edge bounds in coordinate units, origin is lower, and labels
        include declared units (legacy fallback mm). None for unavailable XYZ
        geometry. Existing singleton-spacing and edge rules remain unchanged.

    Notes
    -----
    Uses the existing NumPy float coordinate arithmetic, eagerly and deterministically.
    Signal values, storage dtype, input coordinates and attrs are not modified.
    No scan read or cache is introduced; context properties are read only for
    existing geometry metadata.

    Examples
    --------
    >>> extent = xyz_spatial_extent(oriented, dims, geometry_data=source)  # doctest: +SKIP
    """
    geometry_data = geometry_data if geometry_data is not None else spatial_slice
    selected = _select_xyz_display_coords(
        spatial_slice,
        spatial_dims,
        geometry_data=geometry_data,
        scan=scan,
    )
    if selected is None:
        return None

    x_coord, y_coord, coord_arrays = selected
    x_bounds = _axis_extent_from_coord_values(
        coord_arrays[x_coord],
        axis=1,
        fallback_step=_spatial_dim_resolution_mm(
            geometry_data,
            spatial_dims[1],
            scan=scan,
        ),
    )
    y_bounds = _axis_extent_from_coord_values(
        coord_arrays[y_coord],
        axis=0,
        fallback_step=_spatial_dim_resolution_mm(
            geometry_data,
            spatial_dims[0],
            scan=scan,
        ),
    )
    if x_bounds is None or y_bounds is None:
        return None

    x_unit = _coord_unit(spatial_slice.coords[x_coord])
    y_unit = _coord_unit(spatial_slice.coords[y_coord])
    return (
        [*x_bounds, *y_bounds],
        "lower",
        format_xyz_axis_label(x_coord, x_unit),
        format_xyz_axis_label(y_coord, y_unit),
    )


def _coord_line_along_dim(
    data: xr.DataArray, coord_name: str, dim_name: str
) -> np.ndarray | None:
    """Extract scanner-coordinate values along one data dimension."""
    if coord_name not in data.coords:
        return None
    coord = data.coords[coord_name]
    if dim_name not in coord.dims:
        return None

    try:
        selector = {
            dim: int(coord.sizes[dim] // 2) for dim in coord.dims if dim != dim_name
        }
        return np.asarray(coord.isel(selector).values, dtype=float).reshape(-1)
    except Exception:
        return None


def _data_obj_value(scan: Any | None, name: str) -> Any:
    if scan is None:
        return None
    value = getattr(scan, name, None)
    if value is not None:
        return value
    if hasattr(scan, "get"):
        try:
            return scan.get(name, None)
        except Exception:
            return None
    return None


def _spatial_dim_resolution_mm(
    data: xr.DataArray, dim_name: str, *, scan: Any | None = None
) -> float | None:
    """Return physical voxel spacing for one spatial dimension."""
    if dim_name in data.coords:
        value = data.coords[dim_name].attrs.get("resolution_mm")
        try:
            resolution = abs(float(np.atleast_1d(value)[0]))
        except (TypeError, ValueError, IndexError):
            resolution = 0.0
        if np.isfinite(resolution) and resolution > 0:
            return resolution

    if dim_name == "slice_idx":
        value = data.attrs.get("slice_thickness_mm")
        if value is None:
            value = _data_obj_value(scan, "slice_thickness_mm")
        try:
            resolution = abs(float(np.atleast_1d(value)[0]))
        except (TypeError, ValueError, IndexError):
            resolution = 0.0
        if np.isfinite(resolution) and resolution > 0:
            return resolution

    resolutions = _data_obj_value(scan, "resolution_mm")
    dim_order = ("read_idx", "phase_idx", "slice_idx")
    if resolutions is not None and dim_name in dim_order:
        try:
            resolution = abs(
                float(np.atleast_1d(resolutions)[dim_order.index(dim_name)])
            )
        except (TypeError, ValueError, IndexError):
            resolution = 0.0
        if np.isfinite(resolution) and resolution > 0:
            return resolution
    return None


def _direct_xyz_vector_for_dim(data: xr.DataArray, dim_name: str) -> np.ndarray | None:
    """Return an XYZ voxel step vector when coordinate centres expose it."""
    if not has_xyz_coordinates(data) or int(data.sizes.get(dim_name, 0)) <= 1:
        return None

    components = []
    for coord_name in _XYZ_COORD_NAMES:
        line = _coord_line_along_dim(data, coord_name, dim_name)
        if line is None or line.size < 2:
            return None
        components.append(float(line[1] - line[0]))
    vector = np.asarray(components, dtype=float)
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 0:
        return None
    return vector


def _xyz_vector_for_dim(
    data: xr.DataArray, dim_name: str, *, scan: Any | None = None
) -> np.ndarray | None:
    """Return an XYZ voxel vector, including for singleton spatial axes."""
    direct = _direct_xyz_vector_for_dim(data, dim_name)
    if direct is not None:
        return direct

    spatial_dims = ("read_idx", "phase_idx", "slice_idx")
    if dim_name not in spatial_dims or dim_name not in data.dims:
        return None
    known = {
        dim: _direct_xyz_vector_for_dim(data, dim)
        for dim in spatial_dims
        if dim in data.dims and dim != dim_name
    }
    vector = None
    if (
        dim_name == "slice_idx"
        and known.get("read_idx") is not None
        and known.get("phase_idx") is not None
    ):
        vector = np.cross(known["read_idx"], known["phase_idx"])
    elif (
        dim_name == "read_idx"
        and known.get("phase_idx") is not None
        and known.get("slice_idx") is not None
    ):
        vector = np.cross(known["phase_idx"], known["slice_idx"])
    elif (
        dim_name == "phase_idx"
        and known.get("slice_idx") is not None
        and known.get("read_idx") is not None
    ):
        vector = np.cross(known["slice_idx"], known["read_idx"])

    if vector is None or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    resolution = _spatial_dim_resolution_mm(
        data,
        dim_name,
        scan=scan,
    )
    return vector / norm * (resolution if resolution is not None else 1.0)


def _dominant_xyz_coord_for_dim(
    data: xr.DataArray, dim_name: str, *, scan: Any | None = None
) -> str | None:
    """Return the scanner coordinate varying most along a data dimension."""
    if not has_xyz_coordinates(data):
        return None

    vector = _xyz_vector_for_dim(data, dim_name, scan=scan)
    if vector is not None:
        component = int(np.argmax(np.abs(vector)))
        if abs(float(vector[component])) > 0:
            return _XYZ_COORD_NAMES[component]

    spans = {}
    for coord_name in _XYZ_COORD_NAMES:
        spans[coord_name] = _finite_span(
            _coord_line_along_dim(data, coord_name, dim_name)
        )
    valid = {name: span for name, span in spans.items() if np.isfinite(span)}
    if not valid:
        return None
    coord_name = max(valid, key=valid.get)
    return coord_name if valid[coord_name] > 0 else None


def _slice_xyz_position(
    data: xr.DataArray, coord_name: str, dim_name: str, index: int
) -> float | None:
    """Return the representative scanner coordinate of a slice plane."""
    if coord_name not in data.coords or dim_name not in data.dims:
        return None
    coord = data.coords[coord_name]
    if dim_name not in coord.dims:
        return None

    try:
        idx = max(0, min(int(index), int(data.sizes[dim_name]) - 1))
        values = np.asarray(coord.isel({dim_name: idx}).values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(np.nanmedian(finite))
    except Exception:
        return None


def _slice_xyz_positions(
    data: xr.DataArray, coord_name: str, dim_name: str
) -> list[float | None]:
    """Return representative scanner coordinates for all slices along a dim."""
    if dim_name not in data.dims:
        return []
    return [
        _slice_xyz_position(data, coord_name, dim_name, idx)
        for idx in range(int(data.sizes[dim_name]))
    ]


def _finite_slice_positions(positions: list[float | None]) -> np.ndarray:
    """Return finite slice positions as a float array."""
    finite = [
        float(pos) for pos in positions if pos is not None and np.isfinite(float(pos))
    ]
    return np.asarray(finite, dtype=float)


def _nearest_slice_index_for_xyz_position(
    data: xr.DataArray, coord_name: str, dim_name: str, position: float
) -> int | None:
    """Select the nearest labelled scanner-coordinate slice plane.

    Parameters
    ----------
    data : xarray.DataArray
        Target array with a one-dimensional or meshgrid scanner coordinate in mm.
    coord_name : str
        Scanner coordinate normal to the requested plane, normally x, y or z.
    dim_name : str
        Target dimension whose slices are searched, in original positional order.
    position : float
        Requested finite scanner position in the coordinate's physical unit.

    Returns
    -------
    int or None
        Positional index with the nearest finite representative slice coordinate;
        ties retain the first original index. None if the dimension/coordinate has
        no usable slice positions. Nonfinite coordinate samples are ignored.

    Raises
    ------
    TypeError
        If position is not numeric and cannot participate in coordinate arithmetic.

    Notes
    -----
    Each representative position retains the existing finite median rule. NumPy
    float calculations preserve dimension/coordinate orientation and signal dtype.
    Eager and deterministic, with no input mutation, widget creation or cache.

    Examples
    --------
    >>> data = xr.DataArray([0, 0, 0], dims="slice_idx", coords={"z": ("slice_idx", [6., 3., 0.])})
    >>> nearest_slice_index(data, "z", "slice_idx", 4.5)
    0
    """
    positions = _slice_xyz_positions(data, coord_name, dim_name)
    finite_indices = [
        (idx, pos)
        for idx, pos in enumerate(positions)
        if pos is not None and np.isfinite(pos)
    ]
    if not finite_indices:
        return None
    return min(finite_indices, key=lambda item: abs(float(item[1]) - position))[0]


def _position_outside_slice_fov(position: float, positions: list[float | None]) -> bool:
    """Return True when *position* lies outside the covered slice centers.

    The margin uses half the median slice spacing when multiple target slices
    are available, approximating the through-plane FOV from slice centers.
    """
    finite = _finite_slice_positions(positions)
    if finite.size == 0 or not np.isfinite(position):
        return False

    sorted_positions = np.sort(np.unique(finite))
    margin = 0.0
    if sorted_positions.size > 1:
        diffs = np.diff(sorted_positions)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            margin = float(np.nanmedian(diffs)) / 2.0

    return bool(
        float(position) < float(sorted_positions[0]) - margin
        or float(position) > float(sorted_positions[-1]) + margin
    )


def _xyz_plane_signature(
    spatial_slice: xr.DataArray,
    spatial_dims: list[str],
    *,
    geometry_data: xr.DataArray | None = None,
    slicing_dim: str | None = None,
    scan: Any | None = None,
) -> tuple[str, float] | None:
    """Return the scanner coordinate normal to a displayed 2D plane."""
    if len(spatial_dims) != 2 or not has_xyz_coordinates(spatial_slice):
        return None

    geometry_data = geometry_data if geometry_data is not None else spatial_slice
    if slicing_dim is not None and slicing_dim not in spatial_dims:
        normal_coord = _dominant_xyz_coord_for_dim(
            geometry_data,
            slicing_dim,
            scan=scan,
        )
        if normal_coord in spatial_slice.coords:
            values = np.asarray(spatial_slice.coords[normal_coord].values, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                return normal_coord, float(np.nanmedian(finite))

    spans: dict[str, float] = {}
    medians: dict[str, float] = {}
    for coord_name in _XYZ_COORD_NAMES:
        values = _coord_values_in_display_order(
            spatial_slice,
            coord_name,
            spatial_dims,
        )
        if values is None:
            continue
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        spans[coord_name] = float(np.nanmax(finite) - np.nanmin(finite))
        medians[coord_name] = float(np.nanmedian(finite))

    if len(spans) < 3:
        return None

    normal_coord = min(spans, key=spans.get)
    return normal_coord, medians[normal_coord]


def _dominant_anatomy_dim_for_xyz_coord(
    anat_data: xr.DataArray, coord_name: str
) -> str | None:
    """Return the anatomy spatial dimension that changes most along coord."""
    if not has_xyz_coordinates(anat_data):
        return None

    candidates = [
        dim for dim in ("read_idx", "phase_idx", "slice_idx") if dim in anat_data.dims
    ]
    if not candidates:
        return None

    exact = [
        dim
        for dim in candidates
        if _dominant_xyz_coord_for_dim(anat_data, dim) == coord_name
    ]
    if exact:
        return exact[0]

    spans = {
        dim: _finite_span(_coord_line_along_dim(anat_data, coord_name, dim))
        for dim in candidates
    }
    valid = {dim: span for dim, span in spans.items() if np.isfinite(span)}
    if not valid:
        return None
    best_dim = max(valid, key=valid.get)
    return best_dim if valid[best_dim] > 0 else None


def _target_plane_from_primary_xyz(
    primary_spatial_slice: xr.DataArray | None,
    primary_spatial_dims: list[str] | None,
    target_data: xr.DataArray,
    *,
    primary_data: xr.DataArray | None = None,
    primary_slicing_dim: str | None = None,
    primary_data_obj: Any | None = None,
) -> tuple[dict[str, int], list[str]] | None:
    """Select a target plane matching the displayed primary XYZ plane."""
    if primary_spatial_slice is None or primary_spatial_dims is None:
        return None
    if not has_xyz_coordinates(target_data):
        return None

    signature = _xyz_plane_signature(
        primary_spatial_slice,
        primary_spatial_dims,
        geometry_data=primary_data,
        slicing_dim=primary_slicing_dim,
        scan=primary_data_obj,
    )
    if signature is None:
        return None

    normal_coord, target_position = signature
    spatial_dims = [
        dim for dim in ("read_idx", "phase_idx", "slice_idx") if dim in target_data.dims
    ]
    slicing_dim = _dominant_anatomy_dim_for_xyz_coord(
        target_data,
        normal_coord,
    )
    if slicing_dim is None:
        singleton_dims = [
            dim for dim in spatial_dims if int(target_data.sizes.get(dim, 0)) == 1
        ]
        if len(singleton_dims) == 1:
            slicing_dim = singleton_dims[0]
    if slicing_dim is None or slicing_dim not in target_data.dims:
        return None
    keep_dims = [dim for dim in spatial_dims if dim != slicing_dim]
    if len(keep_dims) != 2:
        return None

    selectors: dict[str, int] = {}
    if target_data.sizes.get(slicing_dim, 1) > 1:
        target_idx = _nearest_slice_index_for_xyz_position(
            target_data,
            normal_coord,
            slicing_dim,
            target_position,
        )
        if target_idx is None:
            return None
        selectors[slicing_dim] = target_idx

    return selectors, keep_dims


def _nearest_slice_from_xyz(
    primary_data: xr.DataArray,
    target_data: xr.DataArray,
    slicing_dim: str,
    primary_index: int,
) -> int | None:
    """Map a primary slice index to the nearest target slice using XYZ coords."""
    coord_name = _dominant_xyz_coord_for_dim(primary_data, slicing_dim)
    if coord_name is None or coord_name not in target_data.coords:
        return None
    if slicing_dim not in target_data.dims:
        return None

    target = _slice_xyz_position(
        primary_data,
        coord_name,
        slicing_dim,
        primary_index,
    )
    if target is None:
        return None

    return _nearest_slice_index_for_xyz_position(
        target_data,
        coord_name,
        slicing_dim,
        target,
    )


def _xyz_display_point(
    data: xr.DataArray,
    state: PlotState,
    spatial_dims: list[str],
    *,
    scan: Any | None = None,
) -> tuple[float, float] | None:
    """Return the current voxel position in scanner display coordinates."""
    if len(spatial_dims) != 2 or not has_xyz_coordinates(data):
        return None

    try:
        plane_selectors = {
            dim: max(
                0,
                min(
                    int(state.spatial.slice_indices.get(dim, 0)),
                    int(data.sizes[dim]) - 1,
                ),
            )
            for dim in data.dims
            if dim not in spatial_dims and dim in data.sizes
        }
        plane = data.isel(plane_selectors)
        selected = _select_xyz_display_coords(
            plane,
            spatial_dims,
            geometry_data=data,
            scan=scan,
        )
        if selected is None:
            return None
        x_coord, y_coord, _ = selected

        def _value(coord_name: str) -> float | None:
            coord = data.coords[coord_name]
            selectors = {
                dim: max(
                    0,
                    min(
                        int(state.spatial.slice_indices.get(dim, 0)),
                        int(data.sizes[dim]) - 1,
                    ),
                )
                for dim in coord.dims
                if dim in data.dims
            }
            value = np.asarray(coord.isel(selectors).values, dtype=float)
            if value.size != 1 or not np.isfinite(value.item()):
                return None
            return float(value.item())

        x_value = _value(x_coord)
        y_value = _value(y_coord)
        if x_value is None or y_value is None:
            return None
        return x_value, y_value
    except Exception:
        return None


def _build_spatial_scalebar(
    spatial_slice: xr.DataArray,
    extent: list[float] | None,
    plot_dim_x: str,
    state: PlotState,
) -> dict[str, Any] | None:
    """Build backend-agnostic scalebar metadata when physical spacing is known."""
    if not state.spatial.show_scalebar or extent is None:
        return None
    width = abs(float(extent[1]) - float(extent[0]))
    if _uses_xyz_coordinates(state) and has_xyz_coordinates(spatial_slice):
        n_pixels = int(spatial_slice.shape[-1]) if spatial_slice.ndim >= 2 else 0
        pixel_size = width / n_pixels if n_pixels > 0 else 0.0
        units = "mm"
    else:
        if plot_dim_x not in spatial_slice.coords:
            return None

        x_coord = spatial_slice.coords[plot_dim_x]
        x_vals = extract_1d_coord(spatial_slice, plot_dim_x)
        if len(x_vals) < 2 or _looks_like_default_index_axis(x_vals):
            return None

        diffs = np.diff(np.asarray(x_vals, dtype=float))
        pixel_size = float(np.nanmedian(np.abs(diffs)))
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            return None

        units = x_coord.attrs.get("unit") or x_coord.attrs.get("units") or "mm"

    if not np.isfinite(pixel_size) or pixel_size <= 0:
        return None

    length = (
        float(state.spatial.scalebar_fixed_value)
        if state.spatial.scalebar_fixed_value is not None
        else _nice_scalebar_length(width * 0.2)
    )
    if not np.isfinite(length) or length <= 0 or length > width:
        return None

    return {
        "pixel_size": pixel_size,
        "length": length,
        "units": units,
        "label": f"{length:g} {units}",
        "color": state.spatial.scalebar_color,
        "location": state.spatial.scalebar_location,
        "fontsize": state.spatial.scalebar_fontsize,
    }


def _infer_secondary_slicing_dim(target_da, current_dim, primary_signature):
    """Infer the existing secondary plane from explicit, XYZ or fallback geometry."""
    if current_dim in target_da.dims:
        return current_dim
    if primary_signature is not None:
        _, normal_coord, _ = primary_signature
        target_dim = _dominant_anatomy_dim_for_xyz_coord(target_da, normal_coord)
        if target_dim in target_da.dims:
            return target_dim
    spatial_dims = [
        dim for dim in ("read_idx", "phase_idx", "slice_idx") if dim in target_da.dims
    ]
    singleton_dims = [
        dim for dim in spatial_dims if int(target_da.sizes.get(dim, 0)) == 1
    ]
    if len(spatial_dims) == 3 and len(singleton_dims) == 1:
        # A singleton through-plane dimension keeps the two displayed axes intact.
        return singleton_dims[0]
    for dim in ("slice_idx", "read_idx", "phase_idx"):
        if dim in target_da.dims and target_da.sizes.get(dim, 1) > 1:
            return dim
    return None


def _format_slice_position(position):
    return f"{float(position):.4g}"


def _build_slice_warning(*, source_label, target_label, coord_name, position):
    return (
        f"Warning: {source_label} slice is outside {target_label} FOV "
        f"({coord_name}={_format_slice_position(position)} mm). "
        f"Showing nearest {target_label} slice."
    )


def _nearest_primary_slice_from_secondary(
    primary_data,
    target_da,
    target_dim,
    target_idx,
    *,
    source_label,
    primary_signature,
    primary_thickness_mm=None,
):
    """Return the primary slice nearest to a chosen anatomy or overlay slice.

    The scanner coordinate that varies most along ``target_dim`` gives the
    position of slice ``target_idx`` of ``target_da``; the primary slice
    along the primary dimension dominated by the same coordinate with the
    nearest position is returned. ``primary_data`` carries x/y/z
    coordinates. ``primary_signature`` is ``(data, normal_coordinate,
    position)`` of the shown primary plane or None; it is used for the
    warning when the primary data have a single slice. That slice then
    covers ``primary_thickness_mm`` (mm) around its position; without a
    known thickness only its position counts as covered.

    Returns None when the slice has no scanner position; otherwise
    ``(primary_dim, primary_index, warning)``, where the first two are None
    when the primary data have no slice dimension along that coordinate and
    ``warning`` names the position when it lies outside the primary slices.
    """
    coord_name = _dominant_xyz_coord_for_dim(target_da, target_dim)
    if coord_name is None:
        return None
    position = _slice_xyz_position(target_da, coord_name, target_dim, target_idx)
    if position is None:
        return None

    def warning_text():
        return _build_slice_warning(
            source_label=source_label,
            target_label="primary",
            coord_name=coord_name,
            position=position,
        )

    primary_dim = _dominant_anatomy_dim_for_xyz_coord(primary_data, coord_name)
    if primary_dim in primary_data.dims and primary_data.sizes.get(primary_dim, 1) > 1:
        primary_idx = _nearest_slice_index_for_xyz_position(
            primary_data, coord_name, primary_dim, position
        )
        if primary_idx is None:
            return None, None, None
        positions = _slice_xyz_positions(primary_data, coord_name, primary_dim)
        warning = (
            warning_text() if _position_outside_slice_fov(position, positions) else None
        )
        return primary_dim, int(primary_idx), warning
    if primary_signature is None or primary_signature[1] != coord_name:
        return None, None, None
    centre = float(primary_signature[2])
    if primary_thickness_mm is not None and primary_thickness_mm > 0:
        outside = abs(float(position) - centre) > primary_thickness_mm / 2.0
    else:
        outside = _position_outside_slice_fov(position, [centre])
    return None, None, warning_text() if outside else None


def _nearest_secondary_slice_from_primary(
    primary_signature, target_da, current_dim, target_label
):
    """Match a secondary plane and report the existing non-overlap warning."""
    if primary_signature is None:
        return None
    _, normal_coord, primary_position = primary_signature
    target_dim = _dominant_anatomy_dim_for_xyz_coord(target_da, normal_coord)
    if target_dim is None:
        target_dim = _infer_secondary_slicing_dim(
            target_da, current_dim, primary_signature
        )
    if target_dim is None or target_dim not in target_da.dims:
        return None
    target_idx = _nearest_slice_index_for_xyz_position(
        target_da, normal_coord, target_dim, primary_position
    )
    if target_idx is None:
        return None
    target_positions = _slice_xyz_positions(target_da, normal_coord, target_dim)
    warning = None
    if _position_outside_slice_fov(primary_position, target_positions):
        warning = _build_slice_warning(
            source_label="primary",
            target_label=target_label,
            coord_name=normal_coord,
            position=primary_position,
        )
    return target_dim, int(target_idx), warning


def _explicit_secondary_plane(target_data, secondary_state):
    """Retain explicit secondary slice selection before primary preparation."""
    spatial_dims = [
        dim for dim in ("read_idx", "phase_idx", "slice_idx") if dim in target_data.dims
    ]
    explicit_dim = secondary_state.slicing_dim
    if explicit_dim not in target_data.dims:
        explicit_dim = next(
            (dim for dim in secondary_state.slice_indices if dim in spatial_dims),
            None,
        )
    if explicit_dim in secondary_state.slice_indices:
        index = int(secondary_state.slice_indices[explicit_dim])
        index = max(0, min(index, int(target_data.sizes[explicit_dim]) - 1))
        return {explicit_dim: index}, [
            dim for dim in spatial_dims if dim != explicit_dim
        ]
    return None


def _secondary_plane_from_primary(
    target_data,
    state,
    *,
    primary_data=None,
    primary_spatial_slice=None,
    primary_spatial_dims=None,
    scan=None,
):
    """Match a prepared primary plane through the existing XYZ/RPS rules."""
    spatial_dims = [
        dim for dim in ("read_idx", "phase_idx", "slice_idx") if dim in target_data.dims
    ]
    use_xyz = _uses_xyz_coordinates(state)
    if use_xyz:
        target_plane = _target_plane_from_primary_xyz(
            primary_spatial_slice,
            primary_spatial_dims,
            target_data,
            primary_data=primary_data,
            primary_slicing_dim=state.spatial.slicing_dim,
            primary_data_obj=scan,
        )
        if target_plane is not None:
            return target_plane
    slicing_dim = state.spatial.slicing_dim
    if (
        slicing_dim not in target_data.dims
        or slicing_dim not in state.spatial.slice_indices
    ):
        return {}, [dim for dim in spatial_dims if dim != slicing_dim]
    primary_index = int(state.spatial.slice_indices[slicing_dim])
    target_index = primary_index
    xyz_index = None
    if use_xyz and primary_data is not None and slicing_dim in primary_data.dims:
        xyz_index = _nearest_slice_from_xyz(
            primary_data,
            target_data,
            slicing_dim,
            primary_index,
        )
    if xyz_index is not None:
        target_index = xyz_index
    elif primary_data is not None and slicing_dim in primary_data.dims:
        try:
            primary_coords = extract_1d_coord(primary_data, slicing_dim)
            target_coords = extract_1d_coord(target_data, slicing_dim)
            if primary_index < len(primary_coords):
                target_index = int(
                    np.argmin(
                        np.abs(target_coords - float(primary_coords[primary_index]))
                    )
                )
        except Exception:
            pass
    target_index = max(
        0,
        min(target_index, int(target_data.sizes[slicing_dim]) - 1),
    )
    return {slicing_dim: target_index}, [
        dim for dim in spatial_dims if dim != slicing_dim
    ]


def _index_selection_coordinates(data, actual_spatial_dims):
    """Attach original positional indices before a displayed slice is prepared."""
    index_coord_names: dict[str, str] = {}
    selection_data = data
    for dim in actual_spatial_dims:
        if dim not in selection_data.dims:
            return None
        base_name = f"__hypermri_plot_index_{dim}"
        coord_name = base_name
        suffix = 1
        while coord_name in selection_data.coords:
            coord_name = f"{base_name}_{suffix}"
            suffix += 1
        index_coord_names[dim] = coord_name
        selection_data = selection_data.assign_coords(
            {coord_name: (dim, np.arange(selection_data.sizes[dim], dtype=int))}
        )
    return selection_data, index_coord_names


def _xyz_indices_from_prepared_slice(
    data,
    selection_data,
    spatial_slice,
    actual_spatial_dims,
    index_coord_names,
    state,
    x,
    y,
    *,
    scan=None,
):
    """Resolve original voxel indices from an already prepared display slice."""
    display_spatial_dims = list(actual_spatial_dims)
    if state.spatial.rotation % 180 != 0:
        display_spatial_dims = [display_spatial_dims[1], display_spatial_dims[0]]
    spatial_slice, display_spatial_dims = _orient_spatial_slice_for_xyz(
        spatial_slice,
        display_spatial_dims,
        state,
        geometry_data=selection_data,
        scan=scan,
    )
    if len(display_spatial_dims) != 2:
        return None
    selected = _select_xyz_display_coords(
        spatial_slice,
        display_spatial_dims,
        geometry_data=selection_data,
        scan=scan,
    )
    if selected is None:
        return None
    x_coord, y_coord, coord_arrays = selected
    x_values = np.asarray(coord_arrays[x_coord], dtype=float)
    y_values = np.asarray(coord_arrays[y_coord], dtype=float)
    if x_values.shape != y_values.shape or x_values.ndim != 2:
        return None
    distance = (x_values - float(x)) ** 2 + (y_values - float(y)) ** 2
    finite = np.isfinite(distance)
    if not np.any(finite):
        return None
    row, col = np.unravel_index(
        int(np.argmin(np.where(finite, distance, np.inf))),
        distance.shape,
    )
    dim_y, dim_x = display_spatial_dims

    def _original_index(dim: str, position: int) -> int | None:
        coord_name = index_coord_names.get(dim)
        if coord_name is None or coord_name not in spatial_slice.coords:
            return None
        coord = spatial_slice.coords[coord_name]
        if coord.ndim != 1 or coord.dims != (dim,):
            return None
        values = np.asarray(coord.values)
        if position < 0 or position >= values.size:
            return None
        value = values[position]
        if not np.isfinite(value):
            return None
        return int(value)

    idx_y = _original_index(dim_y, row)
    idx_x = _original_index(dim_x, col)
    if idx_y is None or idx_x is None:
        return None
    indices = {dim_y: idx_y, dim_x: idx_x}
    for dim, idx in list(indices.items()):
        if dim not in data.sizes:
            return None
        indices[dim] = max(0, min(int(idx), int(data.sizes[dim]) - 1))
    return indices


def _calculate_rps_spatial_extent(spatial_slice, spatial_dims, *, alignment=None):
    """Retain RPS/index edge bounds and an explicitly supplied scan alignment."""
    ny, nx = spatial_slice.shape[-2:]
    plot_dim_y, plot_dim_x = spatial_dims[0], spatial_dims[1]
    extent, origin, xlabel, ylabel = (
        [-0.5, nx - 0.5, ny - 0.5, -0.5],
        "upper",
        format_spatial_axis_label(plot_dim_x),
        format_spatial_axis_label(plot_dim_y),
    )
    try:
        if plot_dim_x in spatial_slice.coords and plot_dim_y in spatial_slice.coords:
            x_vals, y_vals = (
                extract_1d_coord(spatial_slice, plot_dim_x),
                extract_1d_coord(spatial_slice, plot_dim_y),
            )
            if len(x_vals) > 1 and len(y_vals) > 1:
                dx, dy = x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]
                extent, origin = (
                    [
                        float(x_vals[0] - dx / 2),
                        float(x_vals[-1] + dx / 2),
                        float(y_vals[0] - dy / 2),
                        float(y_vals[-1] + dy / 2),
                    ],
                    "lower",
                )
                x_unit = spatial_slice.coords[plot_dim_x].attrs.get(
                    "unit"
                ) or spatial_slice.coords[plot_dim_x].attrs.get("units")
                y_unit = spatial_slice.coords[plot_dim_y].attrs.get(
                    "unit"
                ) or spatial_slice.coords[plot_dim_y].attrs.get("units")
                if not x_unit and not _looks_like_default_index_axis(x_vals):
                    x_unit = "mm"
                if not y_unit and not _looks_like_default_index_axis(y_vals):
                    y_unit = "mm"
                xlabel, ylabel = (
                    format_spatial_axis_label(plot_dim_x, x_unit),
                    format_spatial_axis_label(plot_dim_y, y_unit),
                )
    except Exception:
        pass
    # Alignment positions the source extent; permuting the data is not supported.
    if alignment is not None:
        aligned_extent = get_2d_extent_from_alignment(
            alignment, [plot_dim_y, plot_dim_x]
        )
        if aligned_extent:
            extent = aligned_extent
            origin = "lower"
    return extent, origin, xlabel, ylabel


def view_plane_for_anatomy(
    primary: xr.DataArray,
    anatomy: xr.DataArray,
    choices: Sequence[str],
) -> str:
    """Return the view plane of the primary data in which an anatomy's slices lie.

    Each anatomy was acquired as slices perpendicular to its ``slice_idx``.
    Of the view planes ``choices`` of ``primary`` (each named by the
    dimension perpendicular to it), the one whose normal is most parallel
    to that slice direction is returned, so that the anatomy is shown in its
    acquired slices (with their full in-plane resolution) instead of a
    resampled cut across its thicker slices. Directions are read from the
    scanner coordinates ``x``, ``y``, ``z`` (mm, LPS) along each dimension;
    a dimension of size one gets its direction from the other two (see
    ``_xyz_vector_for_dim``). The sign of a direction does not matter.
    Nothing is read beyond the coordinates and nothing is changed.

    Parameters
    ----------
    primary : xarray.DataArray
        Shown data with the dimensions in ``choices`` and scanner coordinates
        ``x``, ``y``, ``z``.
    anatomy : xarray.DataArray
        Anatomy with ``slice_idx`` and scanner coordinates ``x``, ``y``, ``z``.
    choices : sequence of str
        View planes of ``primary``, each a dimension among ``read_idx``,
        ``phase_idx`` and ``slice_idx`` perpendicular to the plane, in order
        of preference.

    Returns
    -------
    str
        The chosen entry of ``choices``. Equal alignment keeps the earlier
        entry; the first entry when no direction can be determined.

    Raises
    ------
    ValueError
        If ``choices`` is empty.

    Examples
    --------
    >>> view_plane_for_anatomy(bssfp_3d, coronal, ["slice_idx", "phase_idx", "read_idx"])  # doctest: +SKIP
    'phase_idx'
    """
    if not choices:
        raise ValueError("view_plane_for_anatomy needs at least one view plane.")
    direction = _xyz_vector_for_dim(anatomy, "slice_idx")
    if direction is None:
        return choices[0]
    direction = direction / np.linalg.norm(direction)
    best, best_score = choices[0], -1.0
    for dim in choices:
        normal = _xyz_vector_for_dim(primary, dim)
        if normal is None:
            continue
        score = abs(float(np.dot(direction, normal / np.linalg.norm(normal))))
        # A small margin keeps the earlier plane for equal alignment.
        if score > best_score + 1e-6:
            best, best_score = dim, score
    return best


orient_xyz_slice = _orient_spatial_slice_for_xyz
xyz_spatial_extent = _calculate_xyz_spatial_extent
nearest_slice_index = _nearest_slice_index_for_xyz_position

__all__ = [
    "view_plane_for_anatomy",
    "mask_coverage_in_display_order",
    "mask_coverage_in_data_order",
    "display_voxel_centres",
    "snap_display_point",
    "orient_xyz_slice",
    "xyz_spatial_extent",
    "nearest_slice_index",
    "extract_1d_coord",
    "get_spatial_dim_label",
    "format_spatial_axis_label",
    "normalize_coordinate_mode",
    "format_xyz_axis_label",
]
