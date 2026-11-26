"""
Spatial grid generation and management for MRI data.

This module handles creation and manipulation of coordinate grids:
- RPS (read, phase, slice) grids
- XYZ physical coordinate grids
- Grid line definitions and extents

Extracted from utils_general.py as part of Phase 2 refactoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..system.logging import init_default_logger
from .transformations import (
    _POSITION_ATOL_MM,
    COORDINATE_PRECISION,
    define_imagematrix_parameters,
    find_indexing_scanner_dim,
)

logger = init_default_logger(__name__)


class GeometryUnavailableError(ValueError):
    """Signal that a scan declares no geometry of the requested kind.

    The grid functions raise this subclass of ``ValueError`` when the
    metadata needed for the requested coordinates is absent, as opposed to
    present but invalid:

    * read, phase and slice positions (``get_rps_*``): ``fov_mm``,
      ``matrix``, ``n_slices``, ``slice_thickness_mm`` or, for the
      half-voxel correction of :func:`get_rps_extent`, ``resolution_mm`` is
      None, or the scan has several slice packages, which no single read,
      phase and slice frame describes;
    * scanner x, y and z positions (``get_xyz_*``): none of the four
      ``VisuCore`` geometry parameters is available.

    The coordinate properties of
    :class:`hypermri.mixins.UniformDataPropertiesMixin` return None for this
    error only; every other error propagates.  Callers that catch
    ``ValueError`` also catch this error.
    """


def has_xyz_coordinates(data: Any, coord_names=("x", "y", "z")) -> bool:
    """Return whether *data* exposes every requested scanner-space coordinate."""
    coords = getattr(data, "coords", None)
    return coords is not None and all(name in coords for name in coord_names)


def define_grid_lines(
    scan: Any = None,
    fov=None,
    mat=None,
    res=None,
    offset=0.0,
    slice_direction=False,
):
    """
    Define grid lines for a given FOV, matrix size, or resolution.

    Parameters
    ----------
    scan : MRIExperiment, optional
        Scan or vendor handler.  If it, or the vendor handler
        ``scan.handler`` of a scan, has a ParaVision ``method``, missing
        ``fov`` and ``mat`` are read from ``fov_mm`` and ``matrix`` without
        inventing missing geometry. This helper creates one axis, so values read
        from a data object must contain exactly one value; use
        :func:`get_rps_axes` for all scan axes.
    fov : float, optional
        Scalar field of view in mm.
    mat : int, optional
        Scalar matrix size in voxels.
    res : float, optional
        Resolution in mm/voxel
    offset : float, optional
        Offset to apply to the grid lines in mm
    slice_direction: boolean, optional
        Flag indicating if the grid line is in slice direction

    Returns
    -------
    array_like
        Grid line positions, with applied offset and half-voxel correction (Bruker)

    Raises
    ------
    ValueError
        If fewer than two of fov, mat and res are available, a supplied value
        is nonscalar/nonpositive/nonfinite, or mat is not an integer count.

    Notes
    -----
    Apply negative offset. Metadata are read only on this explicit request;
    inputs are not mutated and results are deterministic NumPy float arrays.

    """
    if scan is not None:
        if fov is None and not (mat is not None and res is not None):
            fov = _get_data_obj_value(scan, "fov_mm")
        if mat is None and not (fov is not None and res is not None):
            mat = _get_data_obj_value(scan, "matrix")

    def _scalar(value, name):
        """Return the one value accepted for a single grid axis."""
        values = np.asarray(value)
        if values.size != 1:
            raise ValueError(
                "define_grid_lines creates one axis and requires a scalar "
                f"{name}; got {values.size} values. Pass one axis explicitly "
                "or use get_rps_axes for all scan axes."
            )
        try:
            scalar = float(values.item())
        except (TypeError, ValueError):
            raise ValueError(f"define_grid_lines requires numeric {name}.") from None
        if not np.isfinite(scalar) or scalar <= 0:
            raise ValueError(f"define_grid_lines requires finite positive {name}.")
        if name == "mat":
            if scalar != int(scalar):
                raise ValueError("define_grid_lines mat must be a positive integer.")
            return int(scalar)
        return scalar

    if fov is not None and mat is not None:
        fov = _scalar(fov, "fov")
        mat = _scalar(mat, "mat")
        # Calculate resolution from FOV and matrix
        res = fov / mat
        # Calculate grid from FOV and matrix
        pixel_size = fov / mat
        half_fov = fov / 2.0
        grid_line = np.linspace(
            -half_fov + pixel_size / 2, half_fov - pixel_size / 2, mat
        )
    elif fov is not None and res is not None:
        fov = _scalar(fov, "fov")
        res = _scalar(res, "res")
        # Calculate grid from FOV and resolution
        mat = int(fov / res)
        if mat < 1:
            raise ValueError("define_grid_lines res must not exceed fov.")
        half_fov = fov / 2.0
        grid_line = np.linspace(-half_fov + res / 2, half_fov - res / 2, mat)
    elif mat is not None and res is not None:
        mat = _scalar(mat, "mat")
        res = _scalar(res, "res")
        # Calculate grid from matrix and resolution
        fov = mat * res
        half_fov = fov / 2.0
        grid_line = np.linspace(-half_fov + res / 2, half_fov - res / 2, mat)
    else:
        raise ValueError(
            "define_grid_lines requires at least two of fov (fov_mm), mat "
            "(matrix), res; supply missing geometry explicitly."
        )

    # To make Bruker data overlay:
    if slice_direction:
        pass
    else:
        grid_line = grid_line - res / 2

    # Apply offset if provided (negative)
    grid_line = grid_line - offset

    return grid_line


# ------------------------------------------------------------
# ------------------ RPS Coordinate System -------------------
# ------------------------------------------------------------


def _slice_centre_span_mm(scan: Any, slice_fov_mm: float, matrix) -> float:
    """Return the span over which the slice centres of a scan are spaced.

    The slice field of view of a 2D scan, ``fov_mm[2]``, runs from the first to
    the last slice edge, ``n * thickness + (n - 1) * gap``, while the slice
    centres lie one thickness plus gap apart.  The span ``n * (thickness +
    gap)`` has the same centre and gives that distance when divided by the
    slice count ``n = matrix[2]``.  3D scans, single slices and scans without
    a positive slice thickness keep ``slice_fov_mm``; ``matrix`` is only read
    for 2D scans.
    """
    if _get_spatial_dimension(scan) != "2D":
        return slice_fov_mm
    n_slices = int(matrix[2])
    if n_slices <= 1:
        return slice_fov_mm
    thickness_mm = _get_data_obj_value(scan, "slice_thickness_mm", None)
    if thickness_mm is None or float(thickness_mm) <= 0:
        return slice_fov_mm
    gap_mm = _get_data_obj_value(scan, "slice_gap_mm", None) or 0.0
    return n_slices * (float(thickness_mm) + float(gap_mm))


def _reject_several_slice_packages(scan: Any, operation: str) -> None:
    """Raise :class:`GeometryUnavailableError` for several slice packages.

    The offsets, slice thickness and slice gap that the RPS functions read
    describe one slice package, and ParaVision can place each package at
    other offsets and in another orientation, as in the three-plane
    localizers.  The package sizes come from ``slices_per_pack``, which the
    Bruker handler reads from ``PVM_SPackArrNSlices``; data objects without
    it count as one package.
    """
    counts = np.atleast_1d(_get_data_obj_value(scan, "slices_per_pack", ()))
    if counts.size > 1:
        raise GeometryUnavailableError(
            f"{operation}: the scan has {counts.size} slice packages "
            f"{tuple(int(count) for count in counts)}, but read, phase and slice "
            "positions in mm are defined for a single package only. Select "
            "an individual package for uniform slice geometry. Use "
            "get_xyz_grid for the scanner positions of the voxels of every "
            "package."
        )


def _geometry_vector(value, name, operation, *, sizes=(3,), integer=False):
    """Validate declared positive axis geometry without substituting defaults.

    A value of None is absent geometry and raises
    :class:`GeometryUnavailableError`; a declared but invalid value raises
    ``ValueError``.  Scalars count as one-element vectors.
    """
    lengths = " or ".join(str(size) for size in sizes)
    kind = "positive integer counts" if integer else "finite positive lengths in mm"
    message = (
        f"{operation}: missing or incomplete {name}; supply {lengths} "
        f"{kind} in the operation's axis order."
    )
    if value is None:
        raise GeometryUnavailableError(message)
    try:
        values = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        values = np.asarray([])
    if (
        values.ndim != 1
        or values.size not in sizes
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or (integer and np.any(values != np.floor(values)))
    ):
        raise ValueError(message)
    return values.astype(int) if integer else values.copy()


def _rps_geometry(scan, operation):
    """Read RPS geometry, completing two-axis declarations only from metadata."""
    fov = _geometry_vector(
        _get_data_obj_value(scan, "fov_mm"), "fov_mm", operation, sizes=(2, 3)
    )
    matrix = _geometry_vector(
        _get_data_obj_value(scan, "matrix"),
        "matrix",
        operation,
        sizes=(2, 3),
        integer=True,
    )
    if matrix.size == 2:
        count = _geometry_vector(
            _get_data_obj_value(scan, "n_slices"),
            "n_slices",
            operation,
            sizes=(1,),
            integer=True,
        )[0]
        matrix = np.append(matrix, count)
    if fov.size == 2:
        thickness = _geometry_vector(
            _get_data_obj_value(scan, "slice_thickness_mm"),
            "slice_thickness_mm",
            operation,
            sizes=(1,),
        )[0]
        fov = np.append(fov, thickness * matrix[2])
    fov[2] = _slice_centre_span_mm(scan, fov[2], matrix)
    return fov, matrix


def get_rps_axes(
    scan: Any = None,
    dtype=COORDINATE_PRECISION,
    with_offsets: bool = True,
    rps_grid_mm: tuple = None,
):
    """
    Return the voxel-centre positions along the read, phase and slice axes.

    The RPS position of a voxel is its scanner position projected onto the
    directions of increasing ``read_idx``, ``phase_idx`` and ``slice_idx``.
    For Bruker scans with ``VisuCore`` geometry these axes match the DICOM
    voxel centres projected onto the index directions within 1e-3 mm
    (``tests/unit/utils/test_coordinates_paravision_reference.py``).

    Read and phase voxel ``k`` lies at ``k * fov_mm / matrix - fov_mm / 2``
    minus ``read_offset_mm`` or ``phase_offset_mm``.  The slice centres lie
    symmetrically around plus ``slice_offset_mm``.  The slices of a 2D scan
    lie one ``slice_thickness_mm`` plus ``slice_gap_mm`` apart; 3D
    partitions and single slices lie ``fov_mm[2] / matrix[2]`` apart.

    Parameters
    ----------
    scan : Any
        Scan or vendor handler that provides ``fov_mm`` and ``matrix`` as
        attributes and, through ``get``, the offsets in mm, the slice
        thickness and gap in mm, ``n_slices`` and the spatial dimension
        (``"2D"`` or ``"3D"``).  A two-element ``fov_mm`` or ``matrix`` is
        completed with the slices of a 2D scan.
    dtype : numpy.dtype, optional
        Floating-point dtype of the returned positions.  Defaults to
        ``COORDINATE_PRECISION`` (``float64``).
    with_offsets : bool, optional
        If True (default), shift the axes by the read, phase and slice
        offsets as described above.  If False, centre each axis on zero.
    rps_grid_mm : tuple of numpy.ndarray, optional
        Read, phase and slice position grids of shape
        ``(read, phase, slice)`` in mm.  If given, the axes are read from its
        first row along each dimension and ``scan`` is not evaluated.

    Returns
    -------
    read_axis, phase_axis, slice_axis : numpy.ndarray
        One-dimensional voxel-centre positions in mm with ``matrix[0]``,
        ``matrix[1]`` and ``matrix[2]`` elements, in index order.

    Raises
    ------
    ValueError
        If ``scan`` is None, required ``fov_mm`` or ``matrix`` geometry is
        missing/incomplete, or a two-axis declaration lacks its slice count or
        thickness. Also if its ``slices_per_pack`` lists more
        than one slice package.  The offsets and the slice spacing belong to
        a single package, so the axes of such scans are undefined; use
        :func:`get_xyz_grid` for their voxel positions.
    GeometryUnavailableError
        The subclass of ``ValueError`` raised when a required value is None
        or the scan has several slice packages; a declared but invalid value
        raises plain ``ValueError``.

    Notes
    -----
    Geometry is read only on request and is never invented. Inputs remain
    unchanged; calculation uses float64 and the output is cast to dtype.
    Explicit grids bypass scan metadata and retain their existing axis values.

    Examples
    --------
    Five 1 mm slices with 4 mm gaps lie 5 mm apart:

    >>> from types import SimpleNamespace
    >>> import numpy as np
    >>> class Scan(SimpleNamespace):
    ...     def get(self, name, default=None):
    ...         return getattr(self, name, default)
    >>> scan = Scan(
    ...     spatial_dimension="2D",
    ...     fov_mm=np.array([32.0, 32.0, 21.0]),
    ...     matrix=np.array([4, 4, 5]),
    ...     n_slices=5,
    ...     slice_thickness_mm=1.0,
    ...     slice_gap_mm=4.0,
    ... )
    >>> get_rps_axes(scan)[2]
    array([-10.,  -5.,   0.,   5.,  10.])
    """
    if rps_grid_mm is not None:
        return get_rps_axes_from_grid(rps_grid_mm=rps_grid_mm)
    if scan is None:
        raise ValueError("get_rps_axes: scan with fov_mm and matrix is required.")
    _reject_several_slice_packages(scan, "get_rps_axes")

    fov, matrix = _rps_geometry(scan, "get_rps_axes")

    read_offset = 0
    phase_offset = 0
    slice_offset = 0

    if with_offsets:
        # Add offset like in grid definition
        read_offset = _get_data_obj_value(scan, "read_offset_mm", 0.0)
        phase_offset = _get_data_obj_value(scan, "phase_offset_mm", 0.0)
        slice_offset = -_get_data_obj_value(scan, "slice_offset_mm", 0.0)

    logger.debug("=== get_rps_axes: Generating RPS axes ===")
    logger.debug(f"  FOV: read={fov[0]:.2f}, phase={fov[1]:.2f}, slice={fov[2]:.2f} mm")
    logger.debug(f"  Matrix: read={matrix[0]}, phase={matrix[1]}, slice={matrix[2]}")
    logger.debug(
        f"  Offsets: read={read_offset:.2f}, phase={phase_offset:.2f}, slice={slice_offset:.2f} mm"
    )

    # Create axes
    read_axis = define_grid_lines(fov=fov[0], mat=matrix[0], offset=read_offset).astype(
        dtype
    )
    phase_axis = define_grid_lines(
        fov=fov[1], mat=matrix[1], offset=phase_offset
    ).astype(dtype)
    slice_axis = define_grid_lines(
        fov=fov[2],
        mat=matrix[2],
        offset=slice_offset,
        slice_direction=True,
    ).astype(dtype)

    logger.debug(
        f"  RPS axis ranges: read=[{read_axis.min():.2f}, {read_axis.max():.2f}] mm ({len(read_axis)} points)"
    )
    logger.debug(
        f"                   phase=[{phase_axis.min():.2f}, {phase_axis.max():.2f}] mm ({len(phase_axis)} points)"
    )
    logger.debug(
        f"                   slice=[{slice_axis.min():.2f}, {slice_axis.max():.2f}] mm ({len(slice_axis)} points)"
    )

    return read_axis, phase_axis, slice_axis


def get_rps_axes_from_grid(rps_grid_mm=None):
    """
    Get RPS axes from the passed grid.

    Parameters
    ----------
    rps_grid_mm: tuple
        The read, phase slice grid in milimeter. Can be used to determine the extent.

    Returns
    -------
    read_axis, phase_axis, slice_axis

    """
    # Extract axes from the grid tuple
    read_axis = rps_grid_mm[0][:, 0, 0]
    phase_axis = rps_grid_mm[1][0, :, 0]
    slice_axis = rps_grid_mm[2][0, 0, :]

    return read_axis, phase_axis, slice_axis


def get_rps_extent(
    scan: Any = None,
    dtype=COORDINATE_PRECISION,
    old: bool = False,
    with_offsets: bool = True,
    with_correction: bool = True,
):
    """
    Return the limits of the voxel grid along the read, phase and slice axes.

    The limits follow the pixel-centre convention of :func:`get_rps_axes`:
    with the default options they are the first and last voxel centres
    widened by half a voxel.  Along read and phase this is
    ``[-fov_mm / 2, fov_mm / 2] - offset - resolution_mm / 2``.  Along the
    slice axis it is half the slice-centre span on either side of plus
    ``slice_offset_mm``, where the span is ``n * (thickness + gap)`` for the
    ``n`` slices of a 2D scan and ``fov_mm[2]`` otherwise.  For Bruker scans
    with ``VisuCore`` geometry these limits bound the DICOM voxels within
    1e-3 mm (``tests/unit/utils/test_coordinates_paravision_reference.py``).

    Parameters
    ----------
    scan : Any
        Scan or vendor handler that provides, through ``get``, ``fov_mm`` and
        ``matrix`` with three elements, the offsets, the slice thickness and
        gap in mm, ``resolution_mm`` and the spatial dimension.
    dtype : numpy.dtype, optional
        Floating-point dtype passed to :func:`get_rps_axes` when ``old`` is
        True.  The default path returns Python floats.
    old : bool, optional
        If True, return the first and last voxel centres from
        :func:`get_rps_axes` without the half-voxel widening.  Kept for
        legacy callers; defaults to False.
    with_offsets : bool, optional
        If True (default), shift the limits by the read, phase and slice
        offsets as :func:`get_rps_axes` does.  If False, centre them on zero.
    with_correction : bool, optional
        If True (default), shift the read and phase limits by half a voxel,
        ``resolution_mm / 2``, towards negative positions so that they bound
        the voxel centres of :func:`get_rps_axes`.  If False, they are the
        field-of-view edges around the offset.  The slice limits are not
        shifted.

    Returns
    -------
    list of float
        ``[read_min, read_max, phase_min, phase_max, slice_min, slice_max]``
        in mm.

    Raises
    ------
    ValueError
        If declared field-of-view/matrix geometry is missing or incomplete,
        if ``with_correction`` is True and ``scan`` provides no
        ``resolution_mm`` along read and phase, or if the
        ``slices_per_pack`` of ``scan`` lists more than one slice
        package, whose limits are undefined as for :func:`get_rps_axes`.
    GeometryUnavailableError
        The subclass of ``ValueError`` raised when a required value,
        including ``resolution_mm`` for the correction, is None or the scan
        has several slice packages.

    Notes
    -----
    This deterministic metadata-only operation does not mutate scan metadata
    or reconstruct data. Missing geometry is reported on this request only.
    """
    _reject_several_slice_packages(scan, "get_rps_extent")
    if old:
        read_axis, phase_axis, slice_axis = get_rps_axes(
            scan, dtype, with_offsets=with_offsets
        )

        read_extent = [read_axis.min(), read_axis.max()]
        phase_extent = [phase_axis.min(), phase_axis.max()]
        slice_extent = [slice_axis.min(), slice_axis.max()]
    else:
        fov, _ = _rps_geometry(scan, "get_rps_extent")
        read_fov, phase_fov, slice_fov = fov

        if with_offsets:
            read_offset = _get_data_obj_value(scan, "read_offset_mm", 0.0)
            phase_offset = _get_data_obj_value(scan, "phase_offset_mm", 0.0)
            slice_offset = -_get_data_obj_value(scan, "slice_offset_mm", 0.0)
        else:
            read_offset = 0.0
            phase_offset = 0.0
            slice_offset = 0.0

        # Add offset (negative)
        read_extent = [-read_fov / 2 - read_offset, read_fov / 2 - read_offset]
        phase_extent = [-phase_fov / 2 - phase_offset, phase_fov / 2 - phase_offset]
        slice_extent = [-slice_fov / 2 - slice_offset, slice_fov / 2 - slice_offset]

        # define_grid_lines moves the read and phase voxel centres of
        # get_rps_axes half a voxel towards negative positions, but not the
        # slice centres.
        if with_correction:
            resolution_mm = _get_data_obj_value(scan, "resolution_mm")
            resolution_values = np.atleast_1d(resolution_mm)
            if resolution_mm is None or resolution_values.size < 2:
                error = (
                    GeometryUnavailableError if resolution_mm is None else ValueError
                )
                raise error(
                    "get_rps_extent: the half-voxel correction "
                    "(with_correction=True) needs resolution_mm along read and "
                    f"phase, but the scan provides {resolution_mm!r}.  Pass "
                    "with_correction=False for the field-of-view edges."
                )
            resolution_mm = _geometry_vector(
                resolution_values[:2], "resolution_mm", "get_rps_extent", sizes=(2,)
            )
            read_extent = [edge - resolution_mm[0] / 2 for edge in read_extent]
            phase_extent = [edge - resolution_mm[1] / 2 for edge in phase_extent]

    # Flatten the extents into a single list
    return [*read_extent, *phase_extent, *slice_extent]


def select_rps_extent_by_axis(
    scan: Any = None,
    dtype=COORDINATE_PRECISION,
    with_offsets: bool = True,
    with_correction: bool = True,
    axis: str = "slice_idx",
):
    """
    Return the limits of the two RPS axes perpendicular to ``axis``.

    The limits are those of :func:`get_rps_extent`, in the order read,
    phase, slice with ``axis`` left out.

    Parameters
    ----------
    scan : Any
        Scan or vendor handler as for :func:`get_rps_extent`.
    dtype : numpy.dtype, optional
        Passed to :func:`get_rps_extent`.  Defaults to
        ``COORDINATE_PRECISION`` (``float64``).
    with_offsets : bool, optional
        Passed to :func:`get_rps_extent`; defaults to True.
    with_correction : bool, optional
        Passed to :func:`get_rps_extent`; defaults to True.
    axis : str, optional
        Axis perpendicular to the returned limits: ``"read_idx"``,
        ``"phase_idx"`` or ``"slice_idx"`` (default).

    Returns
    -------
    list of float
        For ``"slice_idx"`` ``[read_min, read_max, phase_min, phase_max]``
        and for ``"read_idx"`` ``[phase_min, phase_max, slice_min,
        slice_max]`` in mm. For ``"phase_idx"`` the flat list
        ``[read_min, read_max, slice_min, slice_max]``. Unknown axes raise.

    Raises
    ------
    ValueError
        If axis is unknown, or as raised by :func:`get_rps_extent`, including
        missing geometry or scans with more than one slice package.

    Notes
    -----
    Reads metadata without reconstruction or mutation. Every valid axis returns
    four deterministic limits with the same precision as get_rps_extent.
    """
    if axis not in ("read_idx", "phase_idx", "slice_idx"):
        raise ValueError(
            f"select_rps_extent_by_axis: unknown axis {axis!r}; "
            "choose read_idx, phase_idx or slice_idx."
        )
    rps_extent_mm = get_rps_extent(
        scan=scan,
        dtype=dtype,
        with_correction=with_correction,
        with_offsets=with_offsets,
    )

    valid_axis_names = ["read_idx", "phase_idx", "slice_idx"]
    if axis in valid_axis_names:
        if axis == "read_idx":
            return rps_extent_mm[2:]
        if axis == "phase_idx":
            return [*rps_extent_mm[:2], *rps_extent_mm[-2:]]
        if axis == "slice_idx":
            return rps_extent_mm[:-2]


def get_rps_grid(
    scan: Any = None,
    dtype=COORDINATE_PRECISION,
    with_offsets: bool = True,
    read_axis=None,
    phase_axis=None,
    slice_axis=None,
):
    """
    Return the read, phase and slice positions of every voxel.

    The grids are the meshgrid, in ``indexing="ij"`` order, of the
    voxel-centre axes of :func:`get_rps_axes` or of the axes passed in.

    Parameters
    ----------
    scan : Any
        Scan or vendor handler as for :func:`get_rps_axes`.  Only evaluated
        if ``read_axis`` is None.
    dtype : numpy.dtype, optional
        Floating-point dtype of the returned coordinates in mm.  Defaults to
        ``COORDINATE_PRECISION`` (``float64``).
    with_offsets : bool, optional
        Passed to :func:`get_rps_axes`.  If True (default), the axes are
        shifted by the read, phase and slice offsets.
    read_axis, phase_axis, slice_axis : array_like, optional
        One-dimensional voxel-centre positions in mm, in index order.  If
        ``read_axis`` is given, the three axes are used instead of
        :func:`get_rps_axes`, and ``scan`` and ``with_offsets`` are
        ignored.

    Returns
    -------
    read_grid, phase_grid, slice_grid : numpy.ndarray
        Positions in mm along the read, phase and slice axes with shape
        ``(read, phase, slice)``.

    Raises
    ------
    ValueError
        As raised by :func:`get_rps_axes` if ``read_axis`` is None,
        including :class:`GeometryUnavailableError` for absent geometry and
        for scans with more than one slice package.
    """
    if read_axis is None:
        read_axis, phase_axis, slice_axis = get_rps_axes(
            scan, dtype, with_offsets=with_offsets
        )

    # Create 3D meshgrids
    read_grid, phase_grid, slice_grid = np.meshgrid(
        read_axis, phase_axis, slice_axis, indexing="ij"
    )

    return (read_grid.astype(dtype), phase_grid.astype(dtype), slice_grid.astype(dtype))


def _get_data_obj_value(scan: Any, name: str, default=None):
    """Return a metadata value from mapping-like or attribute-like data objects."""
    if scan is None:
        return default

    attrs = getattr(scan, "attrs", None)
    if isinstance(attrs, dict) and name in attrs:
        return attrs[name]

    if hasattr(scan, "get"):
        try:
            return scan.get(name, default)
        except Exception:
            pass

    return getattr(scan, name, default)


def _normalise_spatial_shape(scan: Any = None, spatial_shape=None):
    """Return the canonical ``(read, phase, slice)`` shape for a spatial grid."""
    if spatial_shape is not None:
        if isinstance(spatial_shape, dict):
            try:
                return tuple(
                    int(spatial_shape[dim])
                    for dim in ("read_idx", "phase_idx", "slice_idx")
                )
            except KeyError as exc:
                raise ValueError(
                    "spatial_shape dict must contain read_idx, phase_idx, and slice_idx"
                ) from exc

        shape = tuple(int(v) for v in spatial_shape)
        if len(shape) != 3:
            raise ValueError("spatial_shape must contain read, phase, and slice sizes")
        return shape

    matrix = _get_data_obj_value(scan, "matrix", None)
    if matrix is None:
        raise ValueError("Cannot determine spatial shape without matrix metadata")

    matrix = np.atleast_1d(matrix).astype(int)
    if matrix.size == 2:
        n_slices = _get_data_obj_value(scan, "n_slices", 1)
        matrix = np.append(matrix, int(np.atleast_1d(n_slices)[0]))

    if matrix.size < 3:
        raise ValueError("matrix metadata must contain at least read and phase sizes")

    return tuple(int(v) for v in matrix[:3])


def _get_spectral_axis_size(scan: Any = None) -> int | None:
    """Return the spectroscopic axis size when Bruker metadata exposes one."""
    n_acquisition = _get_data_obj_value(scan, "n_acquisition", None)
    if n_acquisition is None:
        method = _get_data_obj_value(scan, "method", None) or {}
        if hasattr(method, "get"):
            n_acquisition = method.get("PVM_SpecMatrix", None)

    try:
        size = int(np.atleast_1d(n_acquisition)[0])
    except Exception:
        return None
    return size if size > 1 else None


def _normalise_spatial_dimension(value: Any) -> str:
    """Return a normalized Bruker spatial dimension label such as ``2D`` or ``3D``."""
    if value is None:
        return ""
    try:
        value = np.atleast_1d(value)[0]
    except Exception:
        pass
    return str(value).strip().strip("<>").upper()


def _get_spatial_dimension(scan: Any = None) -> str:
    """Return the acquisition spatial dimensionality from object or method metadata."""
    spatial_dimension = _get_data_obj_value(scan, "spatial_dimension", None)
    normalized = _normalise_spatial_dimension(spatial_dimension)
    if normalized:
        return normalized

    method = _get_data_obj_value(scan, "method", None) or {}
    if hasattr(method, "get"):
        return _normalise_spatial_dimension(method.get("PVM_SpatDimEnum", None))
    return ""


def _infer_single_frame_slice_spacing(
    scan: Any,
    visu_size: np.ndarray,
    visu_extent: np.ndarray,
    *,
    slice_size: int,
) -> float | None:
    """Infer through-plane spacing when VisuCore stores one frame for a volume."""
    spatial_dimension = _get_spatial_dimension(scan)
    if spatial_dimension == "3D":
        if visu_size.size >= 3 and visu_extent.size >= 3:
            slice_axis_size = int(round(float(visu_size[2])))
            if slice_axis_size > 0:
                return float(visu_extent[2]) / slice_axis_size

        fov = _get_data_obj_value(scan, "fov_mm", None)
        if fov is not None:
            fov = np.atleast_1d(fov).astype(float)
            if fov.size >= 3 and slice_size > 0:
                return float(fov[2]) / slice_size

    slice_spacing = None
    slice_thickness = _get_data_obj_value(scan, "slice_thickness_mm", None)
    slice_gap = _get_data_obj_value(scan, "slice_gap_mm", None)
    if slice_thickness is not None:
        slice_spacing = float(np.atleast_1d(slice_thickness)[0])
        if slice_gap is not None:
            slice_spacing += float(np.atleast_1d(slice_gap)[0])

    if slice_spacing is not None:
        return slice_spacing

    fov = _get_data_obj_value(scan, "fov_mm", None)
    if fov is not None:
        fov = np.atleast_1d(fov).astype(float)
        if fov.size >= 3:
            return float(fov[2]) / max(slice_size - 1, 1)

    return None


def _select_visucore_inplane_geometry(
    scan: Any,
    visu_size: np.ndarray,
    visu_extent: np.ndarray,
    *,
    read_size: int,
    phase_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two VisuCore axes that describe read/phase geometry."""
    axis_count = min(int(visu_size.size), int(visu_extent.size))
    if axis_count < 2:
        raise ValueError("VisuCoreSize and VisuCoreExtent must contain two values")

    axis_indices = list(range(axis_count))
    if axis_count == 2:
        return visu_size[:2], visu_extent[:2]

    spectral_size = _get_spectral_axis_size(scan)
    filtered_indices = [
        idx
        for idx in axis_indices
        if spectral_size is None or int(round(float(visu_size[idx]))) != spectral_size
    ]
    candidate_pools = [filtered_indices, axis_indices]
    target_shapes = {
        (int(read_size), int(phase_size)),
        (int(phase_size), int(read_size)),
    }

    seen_pools = set()
    for pool in candidate_pools:
        pool_key = tuple(pool)
        if len(pool) < 2 or pool_key in seen_pools:
            continue
        seen_pools.add(pool_key)
        for i, first_idx in enumerate(pool):
            for second_idx in pool[i + 1 :]:
                pair_shape = (
                    int(round(float(visu_size[first_idx]))),
                    int(round(float(visu_size[second_idx]))),
                )
                if pair_shape in target_shapes:
                    selected = [first_idx, second_idx]
                    return visu_size[selected], visu_extent[selected]

    raise ValueError(
        "Cannot map VisuCore row/column sizes to read/phase dimensions: "
        f"visu_size={tuple(int(round(float(v))) for v in visu_size[:axis_count])}, "
        f"spatial_shape=({read_size}, {phase_size})"
    )


def _resolve_visucore_inplane_mapping(
    scan: Any,
    orientations: np.ndarray,
    *,
    read_size: int,
    phase_size: int,
    col_size: int,
    row_size: int,
) -> str:
    """Return whether ``read_idx`` follows DICOM columns or rows."""
    read_matches_col = read_size == col_size and phase_size == row_size
    read_matches_row = read_size == row_size and phase_size == col_size

    if read_matches_col and not read_matches_row:
        return "column"
    if read_matches_row and not read_matches_col:
        return "row"
    if not read_matches_col and not read_matches_row:
        raise ValueError(
            "Cannot map VisuCore row/column sizes to read/phase dimensions: "
            f"visu_size=({col_size}, {row_size}), "
            f"spatial_shape=({read_size}, {phase_size})"
        )

    # Equal in-plane sizes are ambiguous. Resolve them from the scanner read
    # direction and DICOM/VisuCore direction cosines when available.
    acquisition_matrix = _get_data_obj_value(scan, "acquisition_matrix", None)
    if acquisition_matrix is not None:
        try:
            read_direction = np.asarray(acquisition_matrix, dtype=float).reshape(-1, 3)[
                0
            ]
            col_direction = np.asarray(orientations[0, 0], dtype=float)
            row_direction = np.asarray(orientations[0, 1], dtype=float)
            col_alignment = abs(float(np.dot(read_direction, col_direction)))
            row_alignment = abs(float(np.dot(read_direction, row_direction)))
            if row_alignment > col_alignment + 1e-6:
                return "row"
            if col_alignment > row_alignment + 1e-6:
                return "column"
        except (TypeError, ValueError, IndexError):
            pass

    return "column"


def _resolve_frame_read_along_rows(
    scan: Any,
    orientations: np.ndarray,
    *,
    frame_count: int,
    read_size: int,
    phase_size: int,
    col_size: int,
    row_size: int,
) -> np.ndarray:
    """Return per ``VisuCorePosition`` frame whether ``read_idx`` follows rows.

    ``reco_transposition`` decides when its values are 0 or 1: one value
    applies to all frames, and values that differ between frames need one
    value per frame.  Otherwise :func:`_resolve_visucore_inplane_mapping`
    decides for all frames.  Values that differ between frames but cannot
    be assigned to the frames raise instead of falling back to one mapping.
    """
    transposition = _get_data_obj_value(scan, "reco_transposition", None)
    if transposition is not None:
        try:
            values = np.asarray(transposition, dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"RECO_transposition {transposition!r} in the reco file is not "
                "numeric; repair or re-export the reco file."
            ) from error
        distinct = sorted(set(values.tolist()))
        if values.size and set(distinct) <= {0.0, 1.0}:
            if len(distinct) == 1:
                read_along_rows = np.full(frame_count, distinct[0] == 1.0)
            elif values.size == frame_count:
                read_along_rows = values == 1.0
            else:
                raise ValueError(
                    "RECO_transposition differs between frames but has "
                    f"{values.size} values for {frame_count} VisuCorePosition "
                    "frames, so read and phase cannot be placed per frame.  Use "
                    "the read_idx, phase_idx and slice_idx coordinates for "
                    "this scan."
                )
            for frame, along_rows in enumerate(read_along_rows):
                expected = (row_size, col_size) if along_rows else (col_size, row_size)
                if (read_size, phase_size) != expected:
                    raise ValueError(
                        f"RECO_transposition {int(along_rows)} of frame {frame} "
                        "places read along the DICOM "
                        f"{'rows' if along_rows else 'columns'}, but VisuCoreSize "
                        f"({col_size}, {row_size}) does not fit the spatial shape "
                        f"({read_size}, {phase_size}); check that reco and "
                        "visu_pars belong to the same reconstruction."
                    )
            return read_along_rows
        if len(distinct) > 1:
            raise ValueError(
                "RECO_transposition differs between frames "
                f"({', '.join(f'{value:g}' for value in distinct)}); only 0 "
                "and 1 place read and phase per frame.  Use the read_idx, "
                "phase_idx and slice_idx coordinates for this scan."
            )

    mapping = _resolve_visucore_inplane_mapping(
        scan,
        orientations,
        read_size=read_size,
        phase_size=phase_size,
        col_size=col_size,
        row_size=row_size,
    )
    return np.full(frame_count, mapping == "row")


def get_xyz_grid_from_visu(
    scan: Any = None,
    spatial_shape=None,
    dtype=COORDINATE_PRECISION,
):
    """
    Return scanner coordinates of the voxel centres from ParaVision geometry.

    ``VisuCorePosition`` holds the centre of the first voxel of each stored
    frame and ``VisuCoreOrientation`` holds, per frame, the column direction,
    the row direction and the slice normal.  ParaVision writes the same values
    to ``Image Position (Patient)`` and ``Image Orientation (Patient)`` of its
    DICOM export.  Voxel ``(read, phase, slice)`` lies at the position of frame
    ``slice`` plus its column index times the column spacing along the column
    direction plus its row index times the row spacing along the row direction.
    The spacings are ``VisuCoreExtent / VisuCoreSize`` of the two in-plane
    axes.

    ``read`` follows the columns or the rows of each frame as
    ``RECO_transposition`` states: 0 the columns and 1 the rows, where
    ParaVision stored phase before read.  One value applies to all frames.
    Values that differ between frames, as in scans with several slice
    packages, need one value per ``VisuCorePosition`` frame.  Without
    ``RECO_transposition``, or with values other than 0 and 1, ``read``
    follows the columns or the rows in all frames, whichever in-plane size
    matches ``read``.  When both in-plane sizes are equal, the axis closer to
    the read direction of ``acquisition_matrix`` is used, and the columns
    when that is unavailable.  When ``VisuCorePosition`` has one position but the shape has
    several slices, as for 3D volumes, slice ``k`` lies ``k`` slice spacings
    from that position along the slice normal.

    ``VisuCoreDiskSliceOrder`` is not read.  Position and orientation already
    describe the stored slice order: the ParaVision DICOM export of a scan
    with ``disk_reverse_slice_order`` places stored slice ``k`` at the
    position given here.

    Parameters
    ----------
    scan : object
        Bruker handler or another object whose metadata is read from an
        ``attrs`` dictionary, a ``get`` method or attributes, in this order.
        Required are ``visucore_position`` with shape ``(n_frames, 3)`` or
        ``(3,)`` in mm, ``visucore_orientation`` with shape ``(n_frames, 9)``
        or ``(9,)`` (column direction, row direction, slice normal), and
        ``visucore_size`` and ``visucore_extent`` with at least the two
        in-plane values, extents in mm.  Optional are ``matrix`` and
        ``n_slices`` when ``spatial_shape`` is omitted,
        ``reco_transposition`` with one value per ``2dseq`` frame (see
        :attr:`hypermri.vendors.bruker.BrukerHandler.reco_transposition`),
        ``acquisition_matrix`` (3 x 3, first row the read direction) for equal
        in-plane sizes, ``n_acquisition`` or ``method["PVM_SpecMatrix"]`` to
        skip a spectroscopic axis of ``visucore_size``, and
        ``spatial_dimension``, ``fov_mm``, ``slice_thickness_mm`` and
        ``slice_gap_mm`` for the slice spacing of a single position.
    spatial_shape : tuple or dict, optional
        Sizes of ``(read, phase, slice)``, or a dictionary with the keys
        ``read_idx``, ``phase_idx`` and ``slice_idx``.  When omitted, the
        shape is taken from ``scan.matrix``, with ``n_slices`` (default 1)
        as slice size when ``matrix`` has two entries.
    dtype : numpy.dtype, optional
        Floating-point dtype of the returned coordinates.  The calculation is
        done in ``float64``.  Defaults to ``COORDINATE_PRECISION``
        (``float64``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(x_grid, y_grid, z_grid)`` in mm in the patient coordinate system of
        DICOM ``Image Position (Patient)``, each with shape
        ``(read, phase, slice)``.

    Raises
    ------
    ValueError
        If the position, orientation, size or extent is missing, if
        ``VisuCoreSize`` or ``VisuCoreExtent`` has fewer than two values, if
        the in-plane sizes do not match ``read`` and ``phase`` as
        ``RECO_transposition`` places them, if ``RECO_transposition`` is not
        numeric or differs between frames without one value of 0 or 1 per
        ``VisuCorePosition`` frame, if position or orientation has fewer
        frames than slices, if the slice spacing of a single position cannot
        be inferred, or if the spatial shape cannot be determined.
    GeometryUnavailableError
        The subclass of ``ValueError`` raised when all four ``VisuCore``
        geometry parameters are missing; if only some are missing, the
        geometry is incomplete and plain ``ValueError`` is raised.

    Notes
    -----
    The function reads metadata only; it does not load pixel data or modify
    ``scan``.

    Examples
    --------
    A 3D coronal volume with one position and a slice spacing of 1 mm:

    >>> from types import SimpleNamespace
    >>> import numpy as np
    >>> volume = SimpleNamespace(
    ...     spatial_dimension="3D",
    ...     visucore_position=np.array([[-26.0, -23.5, 25.0]]),
    ...     visucore_orientation=np.array([[1, 0, 0, 0, 0, -1, 0, 1, 0]]),
    ...     visucore_size=np.array([48, 40, 60]),
    ...     visucore_extent=np.array([48.0, 40.0, 60.0]),
    ... )
    >>> x_grid, y_grid, z_grid = get_xyz_grid_from_visu(
    ...     volume, spatial_shape=(48, 40, 60)
    ... )
    >>> x_grid.shape
    (48, 40, 60)
    >>> float(y_grid[0, 0, 0]), float(y_grid[0, 0, -1])
    (-23.5, 35.5)
    """
    positions = _get_data_obj_value(scan, "visucore_position", None)
    orientations = _get_data_obj_value(scan, "visucore_orientation", None)
    visu_size = _get_data_obj_value(scan, "visucore_size", None)
    visu_extent = _get_data_obj_value(scan, "visucore_extent", None)

    missing = [
        name
        for name, value in (
            ("visucore_position", positions),
            ("visucore_orientation", orientations),
            ("visucore_size", visu_size),
            ("visucore_extent", visu_extent),
        )
        if value is None
    ]
    if missing:
        # All four absent: the scan has no scanner geometry.  Some absent:
        # the geometry is incomplete, which is a defect of the metadata.
        error = GeometryUnavailableError if len(missing) == 4 else ValueError
        raise error(
            f"Missing VisuCore geometry metadata: {', '.join(missing)} not "
            "available.  Scanner x/y/z coordinates are read only from "
            "VisuCorePosition, VisuCoreOrientation, VisuCoreSize and "
            "VisuCoreExtent in the visu_pars file of the scan; keep that file "
            "with the scan or use the read_idx, phase_idx and slice_idx "
            "coordinates instead."
        )

    spatial_shape = _normalise_spatial_shape(scan, spatial_shape)
    read_size, phase_size, slice_size = spatial_shape

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    orientations = np.asarray(orientations, dtype=np.float64)
    if orientations.ndim == 1:
        orientations = orientations.reshape(1, -1)
    orientations = orientations.reshape(-1, 3, 3)

    visu_size = np.asarray(visu_size, dtype=np.float64).reshape(-1)
    visu_extent = np.asarray(visu_extent, dtype=np.float64).reshape(-1)
    if visu_size.size < 2 or visu_extent.size < 2:
        raise ValueError("VisuCoreSize and VisuCoreExtent must contain two values")

    all_visu_size = visu_size.copy()
    all_visu_extent = visu_extent.copy()
    visu_size, visu_extent = _select_visucore_inplane_geometry(
        scan,
        visu_size,
        visu_extent,
        read_size=read_size,
        phase_size=phase_size,
    )
    col_size = int(round(float(visu_size[0])))
    row_size = int(round(float(visu_size[1])))
    col_spacing = float(visu_extent[0]) / col_size
    row_spacing = float(visu_extent[1]) / row_size

    read_along_rows = _resolve_frame_read_along_rows(
        scan,
        orientations,
        frame_count=positions.shape[0],
        read_size=read_size,
        phase_size=phase_size,
        col_size=col_size,
        row_size=row_size,
    )

    if positions.shape[0] == 1 and slice_size > 1:
        slice_spacing = _infer_single_frame_slice_spacing(
            scan,
            all_visu_size,
            all_visu_extent,
            slice_size=slice_size,
        )
        if slice_spacing is None:
            raise ValueError("Cannot infer slice spacing from VisuCore metadata")
        positions = positions[0] + (
            np.arange(slice_size, dtype=np.float64)[:, None]
            * slice_spacing
            * orientations[0, 2]
        )
        read_along_rows = np.repeat(read_along_rows, slice_size)
    elif positions.shape[0] < slice_size:
        raise ValueError(
            f"VisuCorePosition contains {positions.shape[0]} frames, "
            f"but spatial shape requires {slice_size} slices"
        )
    else:
        positions = positions[:slice_size]
        read_along_rows = read_along_rows[:slice_size]

    if orientations.shape[0] == 1 and slice_size > 1:
        orientations = np.repeat(orientations, slice_size, axis=0)
    elif orientations.shape[0] < slice_size:
        raise ValueError(
            f"VisuCoreOrientation contains {orientations.shape[0]} frames, "
            f"but spatial shape requires {slice_size} slices"
        )
    else:
        orientations = orientations[:slice_size]

    read_index = np.arange(read_size, dtype=np.float64)[:, None, None]
    phase_index = np.arange(phase_size, dtype=np.float64)[None, :, None]
    along_rows = read_along_rows[None, None, :]
    col_index = np.where(along_rows, phase_index, read_index)
    row_index = np.where(along_rows, read_index, phase_index)

    origins = positions[None, None, :, :]
    col_dirs = orientations[:, 0, :][None, None, :, :]
    row_dirs = orientations[:, 1, :][None, None, :, :]

    xyz_grid = origins + col_index[..., None] * col_spacing * col_dirs
    xyz_grid = xyz_grid + row_index[..., None] * row_spacing * row_dirs

    return (
        xyz_grid[..., 0].astype(dtype),
        xyz_grid[..., 1].astype(dtype),
        xyz_grid[..., 2].astype(dtype),
    )


# ------------------------------------------------------------
# ------------------ XYZ Coordinate System -------------------
# ------------------------------------------------------------


def define_xyz_grid(scan: Any = None, fov=None, mat=None):
    """
    Return three zero-centred grid lines from a field of view and a matrix.

    Each line is :func:`define_grid_lines` of one ``fov`` and ``mat``
    element: ``mat`` voxel centres spaced ``fov / mat`` apart, centred on
    zero and then moved half a voxel towards negative positions.  The lines
    carry no offsets, directions or patient position, so they are not
    scanner coordinates; :func:`get_xyz_grid` gives the voxel positions.

    Parameters
    ----------
    scan : Any, optional
        Scan or vendor handler for the arguments that are not passed.
        The default ``fov`` comes from
        :func:`~hypermri.utils.image.transforms.define_imageFOV_parameters`
        and the default ``mat`` from
        :func:`define_imagematrix_parameters`; both are in z, x and y order.
        Required geometry and supported orientation metadata must be present.
    fov : array_like, optional
        Three field-of-view lengths in mm, paired element by element with
        ``mat``.
    mat : array_like, optional
        Three voxel counts, paired element by element with ``fov``.

    Returns
    -------
    tuple of numpy.ndarray
        Three 1D arrays in mm with ``mat[0]``, ``mat[1]`` and ``mat[2]``
        elements.

    Raises
    ------
    ValueError
        If either vector is missing or invalid, or if scan-derived geometry
        lacks the required orientation, field of view, matrix or slice data.

    Notes
    -----
    When either ``fov`` or ``mat`` is explicit, that vector is paired
    element by element with the other vector; callers supplying one vector
    must therefore use z, x and y order to match the scan-derived default.
    When both vectors are explicit, their existing numerical convention is
    unchanged. This metadata-only operation is deterministic, does not modify
    its inputs and returns NumPy floating-point arrays without reconstruction
    or plotting. The image field-of-view helper is imported only when this
    operation derives a missing ``fov`` from ``scan`` metadata.

    Examples
    --------
    >>> define_xyz_grid(fov=(4.0, 6.0, 2.0), mat=(4, 3, 1))
    (array([-2., -1.,  0.,  1.]), array([-3., -1.,  1.]), array([-1.]))
    """
    if scan is not None:
        if fov is None:
            from ..image.transforms import (
                define_imageFOV_parameters as _define_imageFOV_parameters,
            )

            _geometry_vector(
                _get_data_obj_value(scan, "fov_mm"),
                "fov_mm (fov)",
                "define_xyz_grid",
            )
            fov = _define_imageFOV_parameters(scan=scan)
        if mat is None:
            _geometry_vector(
                _get_data_obj_value(scan, "matrix"),
                "matrix",
                "define_xyz_grid",
                sizes=(2, 3),
                integer=True,
            )
            mat = np.array(define_imagematrix_parameters(scan=scan))

    fov = _geometry_vector(fov, "fov_mm (fov)", "define_xyz_grid")
    mat = _geometry_vector(mat, "matrix (mat)", "define_xyz_grid", integer=True)

    # Create grid lines for each dimension
    grid_x = define_grid_lines(fov=fov[0], mat=mat[0])
    grid_y = define_grid_lines(fov=fov[1], mat=mat[1])
    grid_z = define_grid_lines(fov=fov[2], mat=mat[2])

    return grid_x, grid_y, grid_z


def get_xyz_grid(
    scan=None,
    dtype=COORDINATE_PRECISION,
    with_offsets=True,
    spatial_shape=None,
):
    """
    Return scanner coordinates of the voxel centres of a scan.

    The coordinates come from the ParaVision ``VisuCore`` geometry through
    :func:`get_xyz_grid_from_visu`, which describes the convention.  There is
    no fallback: the read, phase and slice parameters of ``method`` and
    ``acqp`` alone do not reproduce the DICOM positions, so a scan without
    ``VisuCore`` geometry raises ``ValueError`` and has no scanner
    coordinates.

    Parameters
    ----------
    scan : object
        Bruker handler or another object with the ``VisuCore`` metadata that
        :func:`get_xyz_grid_from_visu` requires.
    dtype : numpy.dtype, optional
        Floating-point dtype of the returned coordinates.  The calculation is
        done in ``float64``.  Defaults to ``COORDINATE_PRECISION``
        (``float64``).
    with_offsets : bool, optional
        Must be ``True`` (the default).  Scanner coordinates always include the
        position of the field of view; a grid relative to the field of view is
        given by the ``read_idx``, ``phase_idx`` and ``slice_idx``
        coordinates.  ``False`` raises ``ValueError``.
    spatial_shape : tuple or dict, optional
        Sizes of ``(read, phase, slice)``, or a dictionary with the keys
        ``read_idx``, ``phase_idx`` and ``slice_idx``.  When omitted, the shape
        is taken from the metadata as in :func:`get_xyz_grid_from_visu`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x_grid, y_grid, z_grid)`` in mm in the patient coordinate system of
        DICOM ``Image Position (Patient)`` (LPS), each with shape
        ``(read, phase, slice)``.

    Raises
    ------
    ValueError
        If ``scan`` is None, if ``with_offsets`` is False, if the
        ``VisuCore`` geometry is missing, or if it does not match the spatial
        shape (see :func:`get_xyz_grid_from_visu`).
    GeometryUnavailableError
        If the scan has none of the ``VisuCore`` geometry parameters (see
        :func:`get_xyz_grid_from_visu`).

    Notes
    -----
    The function reads metadata only; it does not load pixel data or modify
    ``scan``.

    Examples
    --------
    A scan without ``VisuCore`` geometry has no scanner coordinates:

    >>> from types import SimpleNamespace
    >>> try:
    ...     get_xyz_grid(SimpleNamespace(matrix=[32, 32], n_slices=1))
    ... except ValueError as error:
    ...     print(str(error).split(":")[0])
    Missing VisuCore geometry metadata
    """
    if scan is None:
        raise ValueError("Data object required for XYZ grid")
    if not with_offsets:
        raise ValueError(
            "get_xyz_grid: with_offsets=False is not supported.  Scanner x/y/z "
            "coordinates always include the position of the field of view; use "
            "the read_idx, phase_idx and slice_idx coordinates for positions "
            "relative to it."
        )
    return get_xyz_grid_from_visu(
        scan=scan,
        spatial_shape=spatial_shape,
        dtype=dtype,
    )


def assign_xyz_coordinates(
    data,
    scan=None,
    dtype=COORDINATE_PRECISION,
    coord_names=("x", "y", "z"),
):
    """
    Attach scanner-space ``x``, ``y`` and ``z`` coordinates to an xarray object.

    The existing ``read_idx``, ``phase_idx`` and ``slice_idx`` dimensions and
    coordinates are left unchanged.  The new coordinates are auxiliary 3D
    coordinates with dimensions ``(read_idx, phase_idx, slice_idx)``.
    Single-slice 2D arrays that only expose ``read_idx`` and ``phase_idx`` get
    2D auxiliary coordinates with dimensions ``(read_idx, phase_idx)``.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled image data with at least the dimensions ``read_idx`` and
        ``phase_idx``.  Other dimensions are allowed and keep their order.
    scan : object, optional
        Handler or experiment with the Bruker ``VisuCore`` geometry that
        :func:`get_xyz_grid` requires.  Without it the input is returned
        unchanged.
    dtype : numpy.dtype, optional
        Floating-point dtype of the attached coordinates in mm.  Defaults to
        ``COORDINATE_PRECISION`` (``float64``).
    coord_names : tuple of str, optional
        Names of the three attached coordinates, in scanner ``x``, ``y``,
        ``z`` order.

    Returns
    -------
    xarray.DataArray
        A new object with the auxiliary coordinates attached (attributes
        ``unit="mm"`` and ``coordinate_system="dicom_lps"``), or the unchanged
        input when the geometry is unavailable or does not match the spatial
        shape.  The input is not modified.

    Notes
    -----
    A scan without ``VisuCore`` geometry gets no ``x``, ``y`` and ``z``
    coordinates; the reason is logged as a warning of this module's logger.
    """
    canonical_spatial_dims = ("read_idx", "phase_idx", "slice_idx")
    if scan is None or not hasattr(data, "dims"):
        return data
    if not all(dim in data.dims for dim in ("read_idx", "phase_idx")):
        return data

    has_slice_dim = "slice_idx" in data.dims
    coord_dims = canonical_spatial_dims if has_slice_dim else ("read_idx", "phase_idx")
    spatial_shape = (
        int(data.sizes["read_idx"]),
        int(data.sizes["phase_idx"]),
        int(data.sizes["slice_idx"]) if has_slice_dim else 1,
    )

    try:
        xyz_grid = get_xyz_grid(
            scan=scan,
            dtype=dtype,
            with_offsets=True,
            spatial_shape=spatial_shape,
        )
    except Exception as exc:
        logger.warning("No x/y/z coordinates attached: %s", exc)
        return data

    if not has_slice_dim:
        xyz_grid = tuple(np.asarray(grid)[..., 0] for grid in xyz_grid)
        expected_shape = tuple(int(data.sizes[dim]) for dim in coord_dims)
    else:
        expected_shape = spatial_shape

    if any(np.shape(grid) != expected_shape for grid in xyz_grid):
        logger.debug(
            "Skipping XYZ coordinates because grid shapes do not match %s: %s",
            expected_shape,
            [np.shape(grid) for grid in xyz_grid],
        )
        return data

    attrs = {
        "unit": "mm",
        "coordinate_system": "dicom_lps",
        "description": (
            "Scanner-space coordinate derived from Bruker VisuCore/DICOM "
            "geometry when available."
        ),
    }
    coords = {
        name: (coord_dims, grid, attrs)
        for name, grid in zip(coord_names, xyz_grid, strict=False)
    }
    try:
        return data.assign_coords(coords)
    except Exception as exc:
        logger.debug("Failed to attach XYZ coordinates: %s", exc)
        return data


# A scanner coordinate that changes by no more than the geometry position
# tolerance along an index axis is constant along that axis.  Local scans whose
# image axes are parallel to x, y and z vary by at most 4e-15 mm along the
# other index axes, oblique scans and scans with several slice packages by at
# least 4 mm.  Coordinates of such an axis that differ by no more than this
# tolerance are one position; the smallest local voxel step is 0.15 mm.
def _index_axes_varied_by(coordinate_grid: np.ndarray) -> int:
    """Return the number of index axes along which a coordinate grid changes."""
    return sum(
        bool(np.ptp(coordinate_grid, axis=axis).max() > _POSITION_ATOL_MM)
        for axis in range(coordinate_grid.ndim)
    )


def _merge_within_position_tolerance(coordinate_grid: np.ndarray) -> np.ndarray:
    """Return the sorted coordinates of a grid with rounding differences merged.

    Sorted float64 values whose neighbours differ by at most
    ``_POSITION_ATOL_MM`` form one position, represented by the median of
    its voxel values.  The median keeps a position whose voxels agree exactly
    bit-identical and does not depend on the voxel order.
    """
    values = np.sort(coordinate_grid.ravel())
    groups = np.split(values, np.flatnonzero(np.diff(values) > _POSITION_ATOL_MM) + 1)
    widest_mm = max(float(np.ptp(group)) for group in groups)
    if widest_mm > _POSITION_ATOL_MM:
        raise ValueError(
            "get_xyz_axes: coordinates within the position tolerance of "
            f"{_POSITION_ATOL_MM} mm chain over {widest_mm:.3g} mm, so the "
            "voxel steps are not resolved as distinct positions.  Use "
            "get_xyz_grid for the voxel positions."
        )
    return np.array([np.median(group) for group in groups])


def get_xyz_axes(scan=None, dtype=COORDINATE_PRECISION):
    """
    Return the sorted distinct scanner coordinates along x, y and z.

    The values come from the voxel-centre grids of :func:`get_xyz_grid`,
    evaluated in ``float64``.  A coordinate that changes along at most one of
    ``read_idx``, ``phase_idx`` and ``slice_idx`` by more than the geometry
    position tolerance of 1e-3 mm gives the positions of one image axis.
    Its sorted values are split where neighbours differ by more than 1e-3 mm,
    and each resulting position is the median of its voxel values.  This
    merges the rounding differences of order 1e-15 mm that ``VisuCore``
    orientations with ``cos(90 deg)`` entries leave in axis-aligned scans,
    while positions whose voxels agree exactly keep their value bit for bit.

    A coordinate that changes along two or more index axes by more than
    1e-3 mm, as in oblique scans and in scans with several slice packages of
    different orientation or position, is returned as its exactly unique
    values.  These arrays hold every distinct value of the volume and are not
    image axes; a warning is then logged and :func:`get_xyz_grid` gives the
    voxel positions.

    The result is deterministic and independent of the voxel order.  The
    function does not modify ``scan`` and does not cache its result.

    Parameters
    ----------
    scan : object
        Bruker handler or another object with the ``VisuCore`` metadata that
        :func:`get_xyz_grid` requires.
    dtype : numpy.dtype, optional
        Floating-point storage dtype of the returned coordinates in mm.
        Merging is always computed in ``float64``; the result is cast to
        ``dtype`` once at the end.  Defaults to ``COORDINATE_PRECISION``
        (``float64``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(x_axis, y_axis, z_axis)``: 1D arrays of the sorted coordinates in
        mm in the DICOM patient coordinate system (LPS).  For an axis-aligned
        scan their lengths are the numbers of distinct positions along the
        image axis parallel to x, y and z, and neighbouring values differ by
        more than 1e-3 mm.

    Raises
    ------
    ValueError
        If the ``VisuCore`` geometry is missing or inconsistent (see
        :func:`get_xyz_grid`), or if values of a coordinate that changes along
        at most one index axis chain over more than 1e-3 mm because the voxel
        steps do not exceed the tolerance.

    Examples
    --------
    For a 320 x 320 x 9 coronal scan, ``x_axis``, ``y_axis`` and ``z_axis``
    have 320, 9 and 320 entries:

    >>> x_axis, y_axis, z_axis = get_xyz_axes(scan)  # doctest: +SKIP
    """
    xyz_grid = get_xyz_grid(scan, np.float64, with_offsets=True)

    if any(_index_axes_varied_by(grid) > 1 for grid in xyz_grid):
        logger.warning(
            "get_xyz_axes: x, y or z changes along more than one of read_idx, "
            "phase_idx and slice_idx, as in oblique scans and scans with several "
            "slice packages, so the returned arrays hold every distinct value "
            "and are not image axes.  Use get_xyz_grid for the voxel positions."
        )

    return tuple(
        (
            _merge_within_position_tolerance(grid)
            if _index_axes_varied_by(grid) <= 1
            else np.unique(grid.ravel())
        ).astype(dtype)
        for grid in xyz_grid
    )


def _xyz_index_steps_mm(scan: Any, centres: np.ndarray) -> np.ndarray:
    """Return the scanner-space steps of increasing read, phase and slice index.

    ``centres`` holds the voxel centres in mm with shape
    ``(read, phase, slice, 3)``.  The read and phase steps are the differences
    of neighbouring centres.  The slice step of a stack is the difference of
    neighbouring slice centres, which includes any slice gap; a single slice
    spans ``VisuCoreFrameThickness`` along the slice normal of
    ``VisuCoreOrientation``.  Returns an array of shape ``(3, 3)`` with one
    step vector per index axis.
    """
    read_size, phase_size, slice_size = centres.shape[:3]
    if read_size < 2 or phase_size < 2:
        raise ValueError(
            "get_xyz_extent: the voxel size along read_idx and phase_idx is "
            "taken from neighbouring VisuCore voxel centres and needs at least "
            "two voxels along each, but the spatial shape is "
            f"({read_size}, {phase_size}, {slice_size}).  Use get_xyz_grid for "
            "the voxel centres instead."
        )

    read_steps = centres[1, 0] - centres[0, 0]
    phase_steps = centres[0, 1] - centres[0, 0]
    if slice_size > 1:
        slice_steps = np.diff(centres[0, 0], axis=0)
    else:
        thickness_mm = _get_data_obj_value(scan, "visucore_thickness", None)
        if thickness_mm is None:
            raise ValueError(
                "get_xyz_extent: a single slice spans VisuCoreFrameThickness, "
                "which is missing from the visu_pars file of the scan.  Use "
                "get_xyz_grid for the voxel centres instead."
            )
        orientations = _get_data_obj_value(scan, "visucore_orientation", None)
        slice_normal = np.asarray(orientations, dtype=np.float64).reshape(-1, 3, 3)[
            0, 2
        ]
        slice_steps = (float(np.atleast_1d(thickness_mm)[0]) * slice_normal)[None]

    steps = []
    for name, axis_steps in (
        ("read_idx", read_steps),
        ("phase_idx", phase_steps),
        ("slice_idx", slice_steps),
    ):
        # Neighbouring voxel steps that differ by more than the geometry
        # position tolerance do not belong to one stack of equally spaced
        # slices.
        if not np.allclose(axis_steps, axis_steps[0], rtol=0, atol=_POSITION_ATOL_MM):
            raise ValueError(
                f"get_xyz_extent: the step along {name} differs between "
                "slices, so the VisuCore frames do not form one stack of "
                "parallel, equally spaced slices, as in scans with several "
                "slice packages.  Use get_xyz_grid for the voxel centres "
                "instead."
            )
        steps.append(axis_steps[0])
    return np.stack(steps)


def get_xyz_extent(scan=None, dtype=COORDINATE_PRECISION):
    """
    Return the scanner-space bounding box of all voxels of a scan.

    The voxel centres come from the ``VisuCore`` geometry through
    :func:`get_xyz_grid`.  Each voxel spans half an index step on either side
    of its centre along the read, phase and slice directions, the
    pixel-centre convention of :func:`get_rps_extent`.  The read and phase
    steps are the distances of neighbouring voxel centres.  The slice step of
    a stack is the distance of neighbouring slice centres, which includes any
    gap between the slices; a single slice spans ``VisuCoreFrameThickness``.
    For oblique scans the box is aligned with x, y and z and encloses the
    tilted voxels.  For the ParaVision reference series the box matches the box of
    the DICOM voxels within 1e-3 mm
    (``tests/unit/utils/test_coordinates_paravision_reference.py``).

    Parameters
    ----------
    scan : object
        Bruker handler or another object with the ``VisuCore`` metadata that
        :func:`get_xyz_grid` requires.  A single slice also requires
        ``visucore_thickness`` in mm.
    dtype : numpy.dtype, optional
        Floating-point precision to which the limits are rounded before they
        are returned as Python floats.  The calculation is done in
        ``float64``.  Defaults to ``COORDINATE_PRECISION`` (``float64``).

    Returns
    -------
    x_extent, y_extent, z_extent : list of float
        ``[min, max]`` in mm along x, y and z of the DICOM patient coordinate
        system (LPS).

    Raises
    ------
    ValueError
        If the ``VisuCore`` geometry is missing or does not match the spatial
        shape (see :func:`get_xyz_grid`), if read or phase has fewer than two
        voxels, if a single slice has no ``visucore_thickness``, or if the
        frames do not form one stack of parallel, equally spaced slices, as
        in scans with several slice packages.

    Notes
    -----
    The function reads metadata only; it does not load pixel data or modify
    ``scan``.

    Examples
    --------
    A single 2 mm slice of 4 x 2 voxels of 1 mm whose first voxel centre lies
    at (-2, -1, 0) mm:

    >>> from types import SimpleNamespace
    >>> import numpy as np
    >>> scan = SimpleNamespace(
    ...     matrix=np.array([4, 2, 1]),
    ...     visucore_position=np.array([[-2.0, -1.0, 0.0]]),
    ...     visucore_orientation=np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]]),
    ...     visucore_size=np.array([4, 2]),
    ...     visucore_extent=np.array([4.0, 2.0]),
    ...     visucore_thickness=2.0,
    ... )
    >>> get_xyz_extent(scan)
    ([-2.5, 1.5], [-1.5, 0.5], [-1.0, 1.0])
    """
    centres = np.stack(get_xyz_grid(scan, dtype=np.float64), axis=-1)
    half_width = np.abs(_xyz_index_steps_mm(scan, centres)).sum(axis=0) / 2
    flat_centres = centres.reshape(-1, 3)
    limits = np.stack(
        [flat_centres.min(axis=0) - half_width, flat_centres.max(axis=0) + half_width],
        axis=1,
    ).astype(dtype)
    x_extent, y_extent, z_extent = limits.tolist()
    return x_extent, y_extent, z_extent


def _astype_float64_if_numeric(coord_array):
    """
    Return a float64 view of numeric coordinate data to avoid float16 overflow.

    Parameters
    ----------
    coord_array : xarray.DataArray or np.ndarray
        Coordinate array that may need promotion to float64 for reductions.

    Returns
    -------
    Same type as input
        ``coord_array`` converted to float64 when numeric, otherwise untouched.

    """
    try:
        dtype = coord_array.dtype
    except AttributeError:
        return coord_array

    try:
        if np.issubdtype(dtype, np.number):
            return coord_array.astype(np.float64, copy=False)
    except TypeError:
        return coord_array

    return coord_array


def _reduce_axis_to_object_dimension(axis_coord, object_dim):
    """
    Reduce a coordinate grid to a 1D array along the requested object dimension.

    Parameters
    ----------
    axis_coord : xarray.DataArray
        Coordinate grid (e.g., x, y, z) defined on the imaging volume.
    object_dim : str
        Name of the object dimension (read_idx, phase_idx, or slice_idx).

    Returns
    -------
    np.ndarray or None
        1D array of coordinate values along ``object_dim`` or ``None`` if the
        coordinate does not depend on ``object_dim``.

    """
    axis_coord = _astype_float64_if_numeric(axis_coord)
    if object_dim not in axis_coord.dims:
        return None

    # Average across all other dimensions so only the requested dimension remains.
    reduce_dims = [dim for dim in axis_coord.dims if dim != object_dim]
    reduced = axis_coord
    if reduce_dims:
        reduced = reduced.mean(dim=reduce_dims)

    # In case averaging still leaves extraneous singleton dimensions, squeeze them.
    reduced = reduced.squeeze()

    if object_dim not in reduced.dims:
        return None

    return np.asarray(reduced, dtype=np.float64)


def _compute_spacing_from_values(values, eps=1e-9):
    """
    Estimate the smallest non-zero spacing present in the array of values.

    Parameters
    ----------
    values : np.ndarray
        Array with coordinate values (flattened or otherwise).
    eps : float
        Tolerance below which differences are treated as zero.

    Returns
    -------
    float or None
        Estimated spacing or ``None`` if it cannot be determined.

    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size < 2:
        return None

    sorted_vals = np.unique(finite)
    if sorted_vals.size < 2:
        return None

    diffs = np.diff(sorted_vals)
    valid_diffs = np.abs(diffs[~np.isclose(diffs, 0.0, atol=eps)])

    if valid_diffs.size == 0:
        return None

    return float(valid_diffs.min())


def _compute_extent_with_padding(values, object_line, object_index, eps=1e-9):
    """
    Compute min/max extent for a coordinate plane with optional padding.

    Parameters
    ----------
    values : np.ndarray
        Coordinate values for the selected plane.
    object_line : np.ndarray or None
        1D coordinate values along the object dimension to fall back on when
        the plane itself does not provide spacing information (e.g., constant
        slice).
    object_index : int
        Index along the object dimension.
    eps : float
        Tolerance for detecting zero spacing.

    Returns
    -------
    list
        ``[min_val, max_val]`` describing the extent along the coordinate axis.

    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return [np.nan, np.nan]

    min_val = float(finite.min())
    max_val = float(finite.max())

    spacing = _compute_spacing_from_values(finite, eps=eps)

    lower_pad = upper_pad = None
    if spacing is not None and spacing > eps:
        half_spacing = spacing / 2.0
        lower_pad = half_spacing
        upper_pad = half_spacing
    elif object_line is not None and object_line.size > object_index:
        center_val = float(object_line[object_index])
        prev_val = float(object_line[object_index - 1]) if object_index > 0 else None
        next_val = (
            float(object_line[object_index + 1])
            if object_index < object_line.size - 1
            else None
        )

        if prev_val is not None:
            lower_pad = abs(center_val - prev_val) / 2.0
        if next_val is not None:
            upper_pad = abs(next_val - center_val) / 2.0

        if lower_pad is None and upper_pad is not None:
            lower_pad = upper_pad
        if upper_pad is None and lower_pad is not None:
            upper_pad = lower_pad

        # When the plane is effectively constant, use the center value
        # and pads derived from neighbouring slices to create thickness.
        if spacing is None or spacing <= eps:
            min_val = center_val
            max_val = center_val

    if lower_pad is not None:
        min_val -= lower_pad
    if upper_pad is not None:
        max_val += upper_pad

    return [float(min_val), float(max_val)]


def get_xyz_extent_for_coordinate(
    data,
    object_coord_name,
    object_index,
    scanner_axes=("x", "y", "z"),
    plane_axes=None,
):
    """
    Determine the physical XYZ extents for a specific plane in an xarray object.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data structure containing spatial coordinates (``x``, ``y``, ``z``) and
        object coordinates (``read_idx``, ``phase_idx``, ``slice_idx``).
    object_coord_name : str
        Name of the object coordinate to index along (``read_idx``,
        ``phase_idx``, or ``slice_idx``).
    object_index : int
        Zero-based index along ``object_coord_name`` identifying the plane for
        which the extents should be calculated.
    scanner_axes : tuple, optional
        Coordinate names that define the scanner space axes. Defaults to
        ``("x", "y", "z")``.
    plane_axes : sequence of str, optional
        Explicit pair of scanner axes that span the plane to be visualised (for
        example, ``("x", "y")``). When not provided, the function derives the
        in-plane axes by excluding the dominant axis associated with the
        selected object coordinate. If the dominant axis cannot be determined,
        the first two axes from ``scanner_axes`` are used as a fallback.

    Returns
    -------
    dict
        Dictionary containing:
            - ``coordinate``: selected object coordinate name
            - ``index``: integer index that was used
            - ``coordinate_value``: mean physical value of the selected plane
            - ``extents``: mapping of each scanner axis to ``[min, max]``
            - ``plane_mapping``: best-fit scanner axis associated with the
              object coordinate (if determinable), provided as a dict with
              keys ``axis``, ``axis_index``, and ``axis_value``
            - ``plane_axes``: mapping of ``horizontal`` / ``vertical`` axes used
              for imaging representations (``None`` if unavailable)
            - ``imshow_extent``: extent list ``[xmin, xmax, ymin, ymax]`` suitable
              for ``matplotlib.pyplot.imshow`` (``None`` if unavailable)
            - ``imshow_origin``: suggested ``origin`` argument for ``imshow``.

    Raises
    ------
    TypeError
        If the input object does not expose xarray-like coordinates.
    ValueError
        If required coordinates are missing.
    IndexError
        If ``object_index`` is out of bounds for ``object_coord_name``.

    """
    if not hasattr(data, "coords"):
        raise TypeError("data must be an xarray Dataset or DataArray")

    if not isinstance(object_index, int | np.integer):
        raise TypeError("object_index must be an integer index")

    if object_coord_name not in getattr(data, "sizes", {}):
        raise ValueError(
            f"Coordinate '{object_coord_name}' not present in the input data dimensions"
        )

    coord_size = int(data.sizes[object_coord_name])
    if object_index < 0 or object_index >= coord_size:
        raise IndexError(
            f"Index {object_index} is out of bounds for '{object_coord_name}' "
            f"with size {coord_size}"
        )

    for axis in scanner_axes:
        if axis not in data.coords:
            raise ValueError(f"Scanner axis '{axis}' not found in data.coords")

    selector = {object_coord_name: int(object_index)}

    # Determine mean coordinate value for the selected plane
    coordinate_value = np.nan
    if object_coord_name in data.coords:
        coord_array = _astype_float64_if_numeric(data.coords[object_coord_name])
        try:
            plane_coord = coord_array.isel(**selector)
        except ValueError:
            plane_coord = coord_array
            for dim, idx in selector.items():
                if dim in plane_coord.dims:
                    plane_coord = plane_coord.isel({dim: idx})
        coordinate_value = float(
            np.nanmean(np.asarray(plane_coord.values, dtype=np.float64))
        )

    # Pre-compute 1D lines along the object dimension for each scanner axis.
    axis_lines = {}
    for axis in scanner_axes:
        axis_lines[axis] = _reduce_axis_to_object_dimension(
            _astype_float64_if_numeric(data.coords[axis]), object_coord_name
        )

    extents = {}
    for axis in scanner_axes:
        axis_coord = _astype_float64_if_numeric(data.coords[axis])
        try:
            plane_values = axis_coord.isel(**selector)
        except ValueError:
            plane_values = axis_coord
            for dim, idx in selector.items():
                if dim in plane_values.dims:
                    plane_values = plane_values.isel({dim: idx})

        extents[axis] = _compute_extent_with_padding(
            plane_values.values, axis_lines[axis], int(object_index)
        )

    plane_mapping = None
    try:
        mapping_result = find_indexing_scanner_dim(
            data=data,
            object_coord_name=object_coord_name,
            object_idx=int(object_index),
        )
        if mapping_result:
            plane_mapping = {
                "axis": mapping_result[0],
                "axis_index": int(mapping_result[1]),
                "axis_value": float(mapping_result[2]),
            }
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        logger.debug(
            "Failed to determine dominant scanner axis for %s[%d]: %s",
            object_coord_name,
            object_index,
            exc,
        )

    # Determine plane axes for plotting/imshow representations
    def _validate_plane_axes(candidate):
        if candidate is None:
            return None
        if len(candidate) != 2:
            raise ValueError("plane_axes must contain exactly two axis names")
        cleaned = []
        for ax in candidate:
            if ax not in scanner_axes:
                raise ValueError(
                    f"Plane axis '{ax}' not recognised. Available axes: {scanner_axes}"
                )
            cleaned.append(ax)
        if cleaned[0] == cleaned[1]:
            raise ValueError("plane_axes entries must be distinct axis names")
        return cleaned

    resolved_plane_axes = _validate_plane_axes(plane_axes)

    if resolved_plane_axes is None:
        normal_axis = plane_mapping["axis"] if plane_mapping else None
        preferred_order = ["x", "y", "z"]
        candidate_axes = []

        for axis in preferred_order:
            if axis in scanner_axes and axis != normal_axis:
                candidate_axes.append(axis)

        for axis in scanner_axes:
            if axis != normal_axis and axis not in candidate_axes:
                candidate_axes.append(axis)

        if len(candidate_axes) >= 2:
            resolved_plane_axes = candidate_axes[:2]

    if resolved_plane_axes is None:
        resolved_plane_axes = list(scanner_axes)[:2]

    imshow_extent = None
    plane_axes_info = None

    if resolved_plane_axes and len(resolved_plane_axes) == 2:
        horiz_axis, vert_axis = resolved_plane_axes
        horiz_extent = extents.get(horiz_axis)
        vert_extent = extents.get(vert_axis)

        if horiz_extent and vert_extent:
            imshow_extent = [
                float(horiz_extent[0]),
                float(horiz_extent[1]),
                float(vert_extent[0]),
                float(vert_extent[1]),
            ]
            plane_axes_info = {
                "horizontal": horiz_axis,
                "vertical": vert_axis,
                "normal": plane_mapping["axis"] if plane_mapping else None,
            }

    imshow_origin = "lower" if imshow_extent is not None else None

    return {
        "coordinate": object_coord_name,
        "index": int(object_index),
        "coordinate_value": coordinate_value,
        "extents": extents,
        "plane_mapping": plane_mapping,
        "plane_axes": plane_axes_info,
        "imshow_extent": imshow_extent,
        "imshow_origin": imshow_origin,
    }


def get_imshow_extent_from_coords(
    data,
    y_dim: str = "read_idx",
    x_dim: str = "phase_idx",
) -> list[float]:
    """
    Construct an ``imshow`` extent directly from xarray coordinates.

    Parameters
    ----------
    data : xr.DataArray
        Array containing coordinate grids (e.g., ``read_idx``/``phase_idx`` or
        ``x``/``y``) as xarray coordinates.
    y_dim : str, default ``\"read_idx\"``
        Coordinate name to map to the vertical axis of the plot.
    x_dim : str, default ``\"phase_idx\"``
        Coordinate name to map to the horizontal axis of the plot.

    Returns
    -------
    list
        Extent in the form ``[xmin, xmax, ymin, ymax]`` suitable for
        ``matplotlib.pyplot.imshow``.

    Raises
    ------
    TypeError
        If ``data`` is not xarray-like.
    ValueError
        If the requested coordinate names are missing.

    """
    if not hasattr(data, "coords"):
        raise TypeError("data_array must be an xarray DataArray with coordinates")

    if x_dim not in data.coords or y_dim not in data.coords:
        raise ValueError(
            f"Coordinates '{y_dim}' and/or '{x_dim}' not present in DataArray"
        )

    x_vals = np.asarray(data.coords[x_dim].values, dtype=np.float64)
    y_vals = np.asarray(data.coords[y_dim].values, dtype=np.float64)

    return [
        float(np.nanmin(x_vals)),
        float(np.nanmax(x_vals)),
        float(np.nanmin(y_vals)),
        float(np.nanmax(y_vals)),
    ]


def create_isotropic_grid(center, fov, resolution):
    """
    Create an isotropic grid centered at given position.

    Parameters
    ----------
    center : array_like
        Center position [x, y, z] in mm
    fov : float
        Field of view in mm (same for all dimensions)
    resolution : float
        Voxel resolution in mm

    Returns
    -------
    tuple
        (x_grid, y_grid, z_grid) coordinate grids

    """
    center = np.asarray(center)
    if len(center) != 3:
        raise ValueError("Center must be 3-element array")

    # Calculate number of voxels
    n_vox = int(fov / resolution)
    half_fov = fov / 2.0

    # Create 1D axes
    axis = np.linspace(-half_fov, half_fov, n_vox)

    # Create 3D grids centered at specified position
    x_axis = axis + center[0]
    y_axis = axis + center[1]
    z_axis = axis + center[2]

    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    return x_grid, y_grid, z_grid


def resample_grid(original_grid, original_axes, new_axes):
    """
    Resample data from one grid to another using interpolation.

    Parameters
    ----------
    original_grid : array_like
        Original data on the source grid
    original_axes : tuple
        (x_axis, y_axis, z_axis) of original grid
    new_axes : tuple
        (x_axis, y_axis, z_axis) of target grid

    Returns
    -------
    array_like
        Resampled data on new grid

    """
    from scipy.interpolate import RegularGridInterpolator

    # Create interpolator
    interpolator = RegularGridInterpolator(
        original_axes, original_grid, method="linear", bounds_error=False, fill_value=0
    )

    # Create target grid points
    new_x, new_y, new_z = np.meshgrid(*new_axes, indexing="ij")
    target_points = np.stack([new_x.ravel(), new_y.ravel(), new_z.ravel()], axis=1)

    # Interpolate
    resampled_flat = interpolator(target_points)
    resampled_grid = resampled_flat.reshape(new_x.shape)

    return resampled_grid


def get_scanner_coords_extent(
    data, scanner_coord_name, object_coord_name="slice_idx", object_idx=0
):
    """

    Parameters
    ----------
    data : xarray DataArray
        Input data containing scanner coordinates
    scanner_coord_name : str
        Name of the scanner coordinate (e.g., 'x', 'y', 'z')
    object_coord_name : str, optional
        Name of the object coordinate (default is 'slice_idx')
    object_idx : int, optional
        Index of the object coordinate to analyze (default is 0)

    Returns
    -------
    tuple
        (min_x, max_x, min_y, max_y) coordinates for the specified object index, the outermost
        coordinates in the scanner coordinate system at the specified object index.

    """
    return (
        data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[0, 0],
        data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[-1, 0],
        data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[0, -1],
        data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[-1, -1],
    )
