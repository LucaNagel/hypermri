"""Express the grid of one scan in the read, phase and slice frame of another.

These standalone helpers express a source scan, for example a
low-resolution spectroscopy scan, on a reference scan, for example an
anatomical scan.  Positions are in mm along the read, phase and slice (RPS)
axes of the reference scan.  The slice planes of both scans must be parallel,
and within them read and phase may only be swapped or reversed.
The separate viewer option ``anatomy_align=True`` is disabled.
Ordinary anatomy/overlay plotting uses the shared XYZ geometry;
the helpers here remain available for independent geometry verification.

Examples
--------
::

    from hypermri.utils.coordinates.alignment import (
        apply_alignment_to_data,
        compute_scan_alignment,
    )

    result = compute_scan_alignment(source_obj=csi, reference_obj=t2_big)
    result.extent_mm  # source field of view in reference RPS mm
    result.transpose_order  # (1, 0, 2) if read and phase are swapped

    # Source data with the spatial axes in reference order
    data_aligned = apply_alignment_to_data(csi.data.values, result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import xarray as xr

from .transformations import _POSITION_ATOL_MM


@dataclass
class ScanAlignmentResult:
    """Geometry of a source scan in the RPS frame of a reference scan.

    Returned by :func:`compute_scan_alignment`.  All positions are in mm
    along the read, phase and slice (RPS) axes of the reference scan.
    """

    transformation_matrix: np.ndarray
    """3×3 rotation matrix ``T`` such that
    ``reference_rps_vec = T @ source_rps_vec``."""

    transpose_order: tuple[int, int, int]
    """Axis permutation for the spatial ``(read, phase, slice)`` axes of the
    source data so that axis *i* of the permuted array corresponds to
    reference axis *i*.  It is ``(0, 1, 2)`` when read and phase keep their
    axes and ``(1, 0, 2)`` when they are swapped; a reversed axis does not
    change it."""

    extent_mm: list[float]
    """Source field of view along the reference axes,
    ``[read_0, read_1, phase_0, phase_1, slice_0, slice_1]``: for each
    reference axis the positions of the edges of the source field of view on
    the side of the first and of the last source voxel.  A source axis that
    runs against the reference axis gives a decreasing pair."""

    rps_grid_mm_transformed: tuple[np.ndarray, np.ndarray, np.ndarray]
    """Full voxel-level position grids ``(read_grid, phase_grid, slice_grid)``
    of the source scan re-expressed in the reference coordinate frame and
    rearranged to reference-axis order.  Each array has the same shape as
    the source voxel grid after axis permutation."""

    slice_positions_mm: np.ndarray
    """1-D array of unique source slice-centre positions along the reference
    slice axis, useful for matching source slices to reference slices."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_scan_geometry(obj: Any, role: str) -> dict:
    """Extract acquisition matrix, RPS grid, and extent from a scan object.

    Accepts:
    - A ``BrukerExp`` (or any object with ``acquisition_matrix``,
      ``rps_grid_mm``, and ``rps_extent_mm`` properties).
    - An ``xr.DataArray`` with ``read_direction`` / ``phase_direction`` /
      ``slice_direction`` attrs and ``read_idx`` / ``phase_idx`` /
      ``slice_idx`` physical-mm coordinate arrays.
    - Any object with a ``primary_data`` property that returns a DataArray.

    ``role`` names the scan, ``"source"`` or ``"reference"``, in errors.
    """
    # BrukerExp-style: properties available directly
    if hasattr(obj, "acquisition_matrix") and hasattr(obj, "rps_grid_mm"):
        missing = [
            name
            for name in ("acquisition_matrix", "rps_grid_mm", "rps_extent_mm")
            if getattr(obj, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"compute_scan_alignment: {', '.join(missing)} of the {role} "
                "scan is None, so its read, phase and slice axes are unknown.  "
                "Plot it without anatomy_align=True."
            )
        acq_matrix = np.asarray(obj.acquisition_matrix, dtype=float)
        rps_grid = tuple(np.asarray(g, dtype=float) for g in obj.rps_grid_mm)
        rps_extent = list(float(v) for v in obj.rps_extent_mm)
        return {
            "acquisition_matrix": acq_matrix,
            "rps_grid_mm": rps_grid,
            "rps_extent_mm": rps_extent,
        }

    # Unwrap experiment wrappers to their primary DataArray
    if not isinstance(obj, xr.DataArray) and hasattr(obj, "primary_data"):
        obj = obj.primary_data

    if not isinstance(obj, xr.DataArray):
        raise TypeError(
            f"compute_scan_alignment: cannot extract the geometry of the {role} "
            f"scan from {type(obj).__name__}.  Pass a BrukerExp instance or an "
            "xr.DataArray with direction-vector attrs and physical-mm "
            "coordinate arrays."
        )

    da = obj
    required_attrs = ("read_direction", "phase_direction", "slice_direction")
    missing = [k for k in required_attrs if k not in da.attrs]
    if missing:
        raise ValueError(
            f"compute_scan_alignment: the data of the {role} scan lack the "
            f"direction-vector attributes {missing}, which reconstruction "
            "stores in DataArray.attrs.  Plot it without anatomy_align=True."
        )

    acq_matrix = np.array(
        [
            da.attrs["read_direction"],
            da.attrs["phase_direction"],
            da.attrs["slice_direction"],
        ],
        dtype=float,
    )

    coord_names = ("read_idx", "phase_idx", "slice_idx")
    missing_coords = [c for c in coord_names if c not in da.coords]
    if missing_coords:
        raise ValueError(
            f"compute_scan_alignment: the data of the {role} scan lack the "
            f"physical-mm coordinates {missing_coords}, which reconstruction "
            "sets from the scanner geometry.  Plot it without anatomy_align=True."
        )

    rps_grid = tuple(np.asarray(da.coords[c].values, dtype=float) for c in coord_names)
    rps_extent = [
        float(rps_grid[0].min()),
        float(rps_grid[0].max()),
        float(rps_grid[1].min()),
        float(rps_grid[1].max()),
        float(rps_grid[2].min()),
        float(rps_grid[2].max()),
    ]
    return {
        "acquisition_matrix": acq_matrix,
        "rps_grid_mm": rps_grid,
        "rps_extent_mm": rps_extent,
    }


def _build_endpoint_grid(
    rps_extent_mm: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2×2×2 grid of FOV corner points from a 6-value extent."""
    r = np.array([rps_extent_mm[0], rps_extent_mm[1]])
    p = np.array([rps_extent_mm[2], rps_extent_mm[3]])
    s = np.array([rps_extent_mm[4], rps_extent_mm[5]])
    rg, pg, sg = np.meshgrid(r, p, s, indexing="ij")
    return rg, pg, sg


def _rotate_grid(grid: tuple[np.ndarray, ...], T: np.ndarray) -> tuple[np.ndarray, ...]:
    """Apply a 3×3 rotation matrix T to a voxel grid expressed as a 3-tuple.

    Parameters
    ----------
    grid :
        Three arrays of identical shape ``(N1, N2, N3)`` representing the
        read, phase, and slice coordinates of each voxel.
    T :
        3×3 rotation matrix.  Output coordinate *i* = row *i* of T dotted
        with the input coordinate vector.

    Returns
    -------
    tuple of three arrays with the same shape as the input.
    """
    arr = np.asarray(grid, dtype=float)  # (3, N1, N2, N3)
    # Equivalent to applying T to each voxel's (read, phase, slice) vector
    rotated = np.dot(arr.transpose(1, 2, 3, 0), T.T).transpose(3, 0, 1, 2)
    return tuple(rotated)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_scan_alignment(
    source_obj: Any,
    reference_obj: Any,
) -> ScanAlignmentResult:
    """Express the voxel grid of a source scan in the RPS frame of a reference scan.

    The rotation ``T = R_ref @ R_src.T`` from the direction matrices
    ``ACQ_grad_matrix`` (rows: read, phase and slice directions) maps source
    RPS positions to reference RPS positions.  The source grid and the
    corners of its field of view are rotated with ``T``, and the spatial axes
    are permuted so that axis *i* follows reference axis *i*.  The result is
    a 2D overlay of source slices on reference slices, so the slice planes of
    both scans must be parallel.  The extent describes the source field of
    view as a rectangle along the reference read and phase axes, and the
    source data are not resampled, so read and phase may only be swapped or
    reversed between the scans: within the slice plane the source read
    direction must be parallel or perpendicular to the reference read
    direction.  For the 12 pairs of ParaVision reference series with parallel planes
    the aligned grid and extent match the DICOM voxel centres within 1e-3 mm
    (``tests/unit/utils/test_coordinates_paravision_reference.py``).

    Parameters
    ----------
    source_obj : Any
        Scan to express in the reference frame, for example a CSI scan.  A
        sequence object with ``acquisition_matrix``, ``rps_grid_mm`` and
        ``rps_extent_mm``, an object whose ``primary_data`` is such a
        DataArray, or an ``xr.DataArray`` with the ``read_direction``,
        ``phase_direction`` and ``slice_direction`` attributes and the
        ``read_idx``, ``phase_idx`` and ``slice_idx`` coordinates in mm.
    reference_obj : Any
        Scan that defines the target frame, for example an anatomical scan.
        Accepts the same types as ``source_obj``.

    Returns
    -------
    ScanAlignmentResult
        Rotation matrix, axis permutation, source extent and source grid in
        the reference RPS frame in mm, and the source slice positions along
        the reference slice axis.

    Raises
    ------
    ValueError
        If the slice planes of the two scans are not parallel, that is if
        the corners of the source field of view at one slice limit lie more
        than 1e-3 mm apart along the reference slice axis.  If the planes are
        parallel but read and phase are rotated within them by an angle other
        than a multiple of 90 degrees, that is if the corners of one edge of
        the source field of view lie more than 1e-3 mm apart along the
        reference read or phase axis; over a 40 mm field of view this allows
        about 0.0014 degrees.  Also if a scan lacks its direction matrix, RPS
        grid or RPS extent.
    TypeError
        If a scan is neither a sequence object nor a DataArray.

    Notes
    -----
    The function reads geometry metadata only.  It may access
    ``primary_data`` of an object without ``acquisition_matrix``, which can
    trigger its reconstruction.  Neither scan is modified.

    Examples
    --------
    Standalone::

        result = compute_scan_alignment(csi, t2_big)
        # Use result.extent_mm with imshow to position csi on top of t2_big
        ax.imshow(csi_slice,
                  extent=[result.extent_mm[2], result.extent_mm[3],   # phase
                          result.extent_mm[0], result.extent_mm[1]])  # read

    Ordinary XYZ plotting with an anatomical background::

        from hypermri.visualization import PlotOptions, plot
        plot(csi, PlotOptions(anatomy=t2_big, coordinate_mode="xyz"))

    The viewer aligns the layers in scanner coordinates, not with this
    standalone read/phase transform.
    """
    src_geo = _get_scan_geometry(source_obj, "source")
    ref_geo = _get_scan_geometry(reference_obj, "reference")

    src_matrix = src_geo["acquisition_matrix"]  # rows: [read_dir, phase_dir, slice_dir]
    ref_matrix = ref_geo["acquisition_matrix"]

    # T maps a vector in source-RPS space → reference-RPS space
    T = ref_matrix @ src_matrix.T  # (3, 3)

    # On parallel planes all in-plane corners of one source slice limit lie
    # at one position along the reference slice axis.  A tilt beyond the
    # geometry position tolerance, like the in-plane spread checked below, is
    # not rounding of the direction matrices: parallel pairs of local scans
    # tilt by at most 1e-12 mm, crossing pairs by at least 10 mm.
    src_endpoints = _build_endpoint_grid(src_geo["rps_extent_mm"])
    ep_rotated = _rotate_grid(src_endpoints, T)
    tilt_mm = float(np.ptp(ep_rotated[2].reshape(4, 2), axis=0).max())
    if tilt_mm > _POSITION_ATOL_MM:
        angle_deg = float(np.degrees(np.arccos(min(1.0, abs(float(T[2, 2]))))))
        raise ValueError(
            "compute_scan_alignment: the slice planes of the source and the "
            f"reference scan are not parallel; their normals differ by "
            f"{angle_deg:.2f} degrees and the source field of view spans "
            f"{tilt_mm:.3f} mm along the reference slice axis (tolerance "
            f"{_POSITION_ATOL_MM} mm).  A 2D overlay needs a reference "
            "scan with the slice orientation of the source; otherwise plot "
            "without anatomy_align=True."
        )

    # Source axis j = perm[i] runs along reference axis i.  The extent is a
    # box along the reference axes, so read and phase may only be swapped or
    # reversed: each edge of the source field of view must lie at one
    # position along the reference axis that it crosses.
    perm = tuple(int(np.argmax(np.abs(row))) for row in T)
    in_plane_spread_mm = max(
        float(np.ptp(ep_rotated[axis], axis=1 - perm[axis]).max()) for axis in (0, 1)
    )
    if sorted(perm) != [0, 1, 2] or in_plane_spread_mm > _POSITION_ATOL_MM:
        read_row = np.abs(T[0, :2])
        angle_deg = float(np.degrees(np.arctan2(read_row.min(), read_row.max())))
        raise ValueError(
            "compute_scan_alignment: within the parallel slice planes, the read "
            f"direction of the source scan is rotated by {angle_deg:.4f} degrees "
            "from the nearest read or phase direction of the reference scan, so "
            f"an edge of the source field of view spans {in_plane_spread_mm:.3f} "
            "mm along the reference read or phase axis (tolerance "
            f"{_POSITION_ATOL_MM} mm).  A 2D overlay places the source image as "
            "an unrotated rectangle along the reference read and phase axes; "
            "use a reference scan whose read direction is parallel or "
            "perpendicular to that of the source, or plot without "
            "anatomy_align=True."
        )

    # Rotate the full voxel position grid
    src_grid = src_geo["rps_grid_mm"]
    grid_rotated = _rotate_grid(src_grid, T)

    # Rearrange spatial axes to match reference frame order
    grid_rearranged = tuple(
        np.transpose(grid_rotated[i], (perm[0], perm[1], perm[2])) for i in range(3)
    )

    # Rearrange the 8 rotated FOV corner points to get the new extent
    ep_rearranged = tuple(
        np.transpose(ep_rotated[i], (perm[0], perm[1], perm[2])) for i in range(3)
    )

    extent_mm = [
        float(ep_rearranged[0][0, 0, 0]),
        float(ep_rearranged[0][1, 0, 0]),  # read
        float(ep_rearranged[1][0, 0, 0]),
        float(ep_rearranged[1][0, 1, 0]),  # phase
        float(ep_rearranged[2][0, 0, 0]),
        float(ep_rearranged[2][0, 0, 1]),  # slice
    ]

    slice_positions_mm = np.unique(grid_rearranged[2])

    return ScanAlignmentResult(
        transformation_matrix=T,
        transpose_order=perm,
        extent_mm=extent_mm,
        rps_grid_mm_transformed=grid_rearranged,
        slice_positions_mm=slice_positions_mm,
    )


def apply_alignment_to_data(
    data: Union[xr.DataArray, np.ndarray],
    alignment: ScanAlignmentResult,
) -> np.ndarray:
    """Permute the spatial axes of *data* to match the reference coordinate frame.

    Only needed when ``alignment.transpose_order != (0, 1, 2)``, i.e. when
    the two scans have different orientations.

    Parameters
    ----------
    data :
        Source data array.  Expected layouts:

        - **2-D** ``(read, phase)`` — a single already-sliced spatial plane.
        - **Full array** with standard HyperMRI layout
          ``(acquisition, read, phase, slice, repetition, channel)``.

        A DataArray is automatically converted via ``np.asarray``.
    alignment :
        Result from :func:`compute_scan_alignment`.

    Returns
    -------
    np.ndarray
        Data with spatial axes permuted to match the reference frame.
    """
    arr = np.asarray(data)
    perm = alignment.transpose_order

    if perm == (0, 1, 2):
        return arr  # identity — no copy needed

    if arr.ndim == 2:
        # Single 2D spatial slice: axes are (read, phase)
        if perm[0] == 1 and perm[1] == 0:
            return arr.T
        return arr  # unsupported permutation for 2D — return unchanged

    # Full array: spatial dims occupy positions 1, 2, 3
    spatial_axes = tuple(1 + p for p in perm)
    full_perm = (0,) + spatial_axes + tuple(range(4, arr.ndim))
    return np.transpose(arr, full_perm)


def get_2d_extent_from_alignment(
    alignment: ScanAlignmentResult,
    display_dims: list[str],
) -> list[float]:
    """Extract the 2-D imshow-compatible extent from an alignment result.

    Parameters
    ----------
    alignment :
        Result from :func:`compute_scan_alignment`.
    display_dims :
        Two dimension names in ``[dim_y, dim_x]`` order (as used by the
        plotter), e.g. ``['read_idx', 'phase_idx']``.

    Returns
    -------
    list of float
        ``[x_0, x_1, y_0, y_1]`` in mm, where *x* is the last display dim
        and *y* is the first.  Each pair comes from
        :attr:`ScanAlignmentResult.extent_mm` and decreases along a reversed
        axis, so an image drawn with ``origin="lower"`` is flipped there.
        An empty list if a display dim is not ``read_idx``, ``phase_idx`` or
        ``slice_idx``.
    """
    _dim_to_slice = {"read_idx": (0, 1), "phase_idx": (2, 3), "slice_idx": (4, 5)}
    dim_y, dim_x = display_dims[0], display_dims[1]
    ix = _dim_to_slice.get(dim_x)
    iy = _dim_to_slice.get(dim_y)
    if ix is None or iy is None:
        return []
    return [
        alignment.extent_mm[ix[0]],
        alignment.extent_mm[ix[1]],
        alignment.extent_mm[iy[0]],
        alignment.extent_mm[iy[1]],
    ]
