"""Position and size of the volume a localized spectroscopy scan excites.

Single-voxel spectroscopy (PRESS, STEAM) measures one box; slice
spectroscopy measures one slab of finite thickness and unlimited in-plane
size. :func:`excited_volume` reads that volume from the Bruker ``method``
parameters in DICOM patient coordinates (LPS, mm), and
:func:`outline_in_plane` intersects it with an image plane, so that the
volume can be drawn as an outline on an anatomical image.

The geometry is the nominal one planned on the scanner; chemical shift
displacement of the excited volume is not included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Head_Prone: a vector in the Bruker gradient frame (rows of the gradient
# orientation matrix) becomes DICOM LPS by flipping x and y. Checked on local
# imaging scans of the same session: VisuCore voxel centres agree with
# P @ G.T @ offsets up to the half-voxel VisuCore grid convention.
_GRADIENT_TO_LPS = {"Head_Prone": np.diag([-1.0, -1.0, 1.0])}


@dataclass(frozen=True)
class ExcitedVolume:
    """Box excited by a localized spectroscopy scan, in DICOM LPS.

    Attributes
    ----------
    centre_mm : numpy.ndarray
        Centre of the box, shape (3,), in mm along DICOM x (left), y
        (posterior) and z (superior).
    axes : numpy.ndarray
        Shape (3, 3). Row ``i`` is the unit vector of the read, phase and
        slice direction of the box in LPS. The sign of each row is
        arbitrary; the box is symmetric about its centre.
    size_mm : numpy.ndarray
        Edge lengths along ``axes``, shape (3,), in mm. ``inf`` marks an
        unlimited direction (the in-plane size of a slab).
    kind : str
        ``"voxel"`` for single-voxel spectroscopy, ``"slab"`` for slice
        spectroscopy, ``"slice"`` for one slice of an image (see
        :func:`image_slice_volume`).
    """

    centre_mm: np.ndarray
    axes: np.ndarray
    size_mm: np.ndarray
    kind: str

    def contains(self, points_mm: np.ndarray) -> np.ndarray:
        """Return whether points lie inside the box.

        Parameters
        ----------
        points_mm : numpy.ndarray
            Positions in LPS mm, shape (..., 3).

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (...); points on the surface count as
            inside.
        """
        offsets = (np.asarray(points_mm, dtype=float) - self.centre_mm) @ self.axes.T
        return np.all(np.abs(offsets) <= self.size_mm / 2 + 1e-9, axis=-1)


def _first_row(value: Any, name: str, size: int) -> np.ndarray:
    """Return the first ``size`` values of a per-package parameter."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < size:
        raise ValueError(
            f"excited_volume: method parameter {name} has {array.size} values; "
            f"expected at least {size}."
        )
    return array[:size]


def _voxel_entry(
    method: dict, name: str, voxel: int, shape: tuple[int, ...]
) -> np.ndarray:
    """Return the entry of one voxel from a per-voxel ``PVM_VoxArr*`` parameter."""
    entries = np.asarray(method.get(name, []), dtype=float).reshape(-1, *shape)
    if not 0 <= voxel < len(entries):
        raise ValueError(
            f"excited_volume: method parameter {name} has no entry for voxel "
            f"{voxel}; it holds {len(entries)}."
        )
    return entries[voxel]


def excited_volume(scan: Any, voxel: int = 0) -> ExcitedVolume:
    """Return the volume a single-voxel or slice spectroscopy scan excites.

    A scan with ``PVM_VoxArrSize`` is a single-voxel scan (PRESS, STEAM):
    the box has the size ``PVM_VoxArrSize``, its centre
    ``PVM_VoxArrPositionRPS`` and its orientation ``PVM_VoxArrGradOrient``.
    Otherwise a scan with one slice package (``PVM_SliceThick``,
    ``PVM_SPackArrSliceOffset``, ``PVM_SPackArrGradOrient``) is a slab of
    that thickness, unlimited in read and phase. The offsets are converted
    from the gradient frame to DICOM LPS for the patient position stored in
    ``ACQ_patient_pos``.

    Parameters
    ----------
    scan : object
        Bruker scan or handler. The ``method`` dictionary and
        ``patient_position`` are read from ``scan.handler`` when it
        exists, else from ``scan`` itself. Nothing is modified.
    voxel : int, optional
        Voxel index for scans that define several voxels. Default 0.

    Returns
    -------
    ExcitedVolume
        Centre, axes and edge lengths in LPS mm.

    Raises
    ------
    ValueError
        If the patient position is not one whose conversion was validated
        (currently only ``Head_Prone``), if the scan has neither voxel nor
        slice geometry (for example a non-selective pulse), if it has several
        slice packages or slices, or if a parameter has the wrong size.

    Examples
    --------
    >>> volume = excited_volume(press_scan)  # doctest: +SKIP
    >>> volume.kind, volume.size_mm  # doctest: +SKIP
    ('voxel', array([4.5, 3.5, 7.5]))
    """
    source = getattr(scan, "handler", scan)
    method = getattr(source, "method", None)
    if not isinstance(method, dict):
        raise ValueError(
            "excited_volume needs a Bruker scan or handler with a method "
            f"parameter dictionary; got {type(scan).__name__}."
        )
    position = getattr(source, "patient_position", None)
    to_lps = _GRADIENT_TO_LPS.get(position)
    if to_lps is None:
        raise ValueError(
            f"excited_volume: patient position {position!r} (ACQ_patient_pos) "
            f"is not supported; only {sorted(_GRADIENT_TO_LPS)} has a validated "
            "conversion from the gradient frame to patient coordinates."
        )
    if "PVM_VoxArrSize" in method:
        gradient = _voxel_entry(method, "PVM_VoxArrGradOrient", voxel, (3, 3))
        offsets = _voxel_entry(method, "PVM_VoxArrPositionRPS", voxel, (3,))
        size = _voxel_entry(method, "PVM_VoxArrSize", voxel, (3,))
        kind = "voxel"
    elif "PVM_SliceThick" in method and "PVM_SPackArrGradOrient" in method:
        packages = int(method.get("PVM_NSPacks", 1))
        slices = int(np.atleast_1d(method.get("PVM_SPackArrNSlices", 1))[0])
        if packages != 1 or slices != 1:
            raise ValueError(
                "excited_volume: a slab needs one slice package with one slice; "
                f"the scan has {packages} packages (PVM_NSPacks) and {slices} "
                "slices (PVM_SPackArrNSlices)."
            )
        gradient = _first_row(
            method["PVM_SPackArrGradOrient"], "PVM_SPackArrGradOrient", 9
        ).reshape(3, 3)
        offsets = np.array(
            [
                float(np.atleast_1d(method.get(name, 0.0))[0])
                for name in (
                    "PVM_SPackArrReadOffset",
                    "PVM_SPackArrPhase1Offset",
                    "PVM_SPackArrSliceOffset",
                )
            ]
        )
        thickness = float(np.atleast_1d(method["PVM_SliceThick"])[0])
        size = np.array([np.inf, np.inf, thickness])
        kind = "slab"
    else:
        raise ValueError(
            "excited_volume: the scan defines neither a voxel (PVM_VoxArrSize) "
            "nor a slice (PVM_SliceThick, PVM_SPackArrGradOrient); a "
            "non-selective excitation has no outline."
        )
    if not np.all(size > 0):
        raise ValueError(f"excited_volume: edge lengths {size} mm must be positive.")
    axes = gradient @ to_lps
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    return ExcitedVolume(
        centre_mm=to_lps @ gradient.T @ offsets,
        axes=axes,
        size_mm=size,
        kind=kind,
    )


def _clip(polygon: list[np.ndarray], normal: np.ndarray, limit: float) -> list:
    """Keep the part of a convex polygon where ``normal @ point <= limit``."""
    kept = []
    for index, point in enumerate(polygon):
        previous = polygon[index - 1]
        inside, was_inside = normal @ point <= limit, normal @ previous <= limit
        if inside != was_inside:
            fraction = (limit - normal @ previous) / (normal @ (point - previous))
            kept.append(previous + fraction * (point - previous))
        if inside:
            kept.append(point)
    return kept


def outline_in_plane(
    volume: ExcitedVolume,
    origin_mm: np.ndarray,
    row_step_mm: np.ndarray,
    column_step_mm: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return the cut of an excited volume with an image plane.

    The image plane is a regular grid: the centre of pixel ``(r, c)`` lies
    at ``origin_mm + r * row_step_mm + c * column_step_mm``. The cut of the
    box with this plane is a convex polygon, clipped to the image
    (``-0.5 <= r <= n_rows - 0.5`` and likewise for columns).

    Parameters
    ----------
    volume : ExcitedVolume
        Box in LPS mm.
    origin_mm : numpy.ndarray
        LPS position of the centre of pixel (0, 0), shape (3,), in mm.
    row_step_mm, column_step_mm : numpy.ndarray
        LPS displacement from one pixel to the next along rows and columns,
        shape (3,), in mm.
    shape : tuple of int
        Number of rows and columns of the image.

    Returns
    -------
    numpy.ndarray
        Corners of the polygon as continuous (row, column) pixel indices,
        shape (n, 2), in order around the polygon (not closed). Empty
        (shape (0, 2)) if the volume does not reach the image.
    """
    n_rows, n_columns = shape
    polygon = [
        np.array(corner, dtype=float)
        for corner in (
            (-0.5, -0.5),
            (-0.5, n_columns - 0.5),
            (n_rows - 0.5, n_columns - 0.5),
            (n_rows - 0.5, -0.5),
        )
    ]
    steps = np.stack([row_step_mm, column_step_mm]).astype(float)  # (2, 3)
    for axis, size in zip(volume.axes, volume.size_mm, strict=True):
        if not np.isfinite(size):
            continue
        # Offset of pixel (r, c) from the centre along this axis:
        # start + normal @ (r, c).
        start = float(axis @ (np.asarray(origin_mm, float) - volume.centre_mm))
        normal = steps @ axis
        for sign in (1.0, -1.0):
            # sign * (start + normal @ p) <= size / 2
            polygon = _clip(polygon, sign * normal, size / 2 - sign * start)
            if not polygon:
                return np.empty((0, 2))
    return np.array(polygon)


def image_slice_volume(
    origin_mm: np.ndarray,
    row_step_mm: np.ndarray,
    column_step_mm: np.ndarray,
    shape: tuple[int, int],
    thickness_mm: float,
) -> ExcitedVolume:
    """Return one slice of an image as a box in LPS.

    The slice is the regular grid of :func:`outline_in_plane` (pixel
    ``(r, c)`` centred at ``origin_mm + r * row_step_mm + c * column_step_mm``)
    extended by half a pixel on every side and by half the slice thickness on
    both sides of the plane. Its cut with another image plane
    (:func:`outline_in_plane`) is where the two slices cross, as the slice
    position lines of a scanner console.

    Parameters
    ----------
    origin_mm : numpy.ndarray
        LPS position of the centre of pixel (0, 0), shape (3,), in mm.
    row_step_mm, column_step_mm : numpy.ndarray
        LPS displacement from one pixel to the next along rows and columns,
        shape (3,), in mm. They must be non-zero and not parallel.
    shape : tuple of int
        Number of rows and columns of the slice.
    thickness_mm : float
        Slice thickness in mm, >= 0. Zero gives a plane without thickness,
        whose cut with another plane is a line.

    Returns
    -------
    ExcitedVolume
        Box of the slice with ``kind`` ``"slice"``; ``axes`` are the row,
        column and normal directions.

    Raises
    ------
    ValueError
        If the steps do not span a plane or the thickness is negative or not
        finite.

    Examples
    --------
    >>> import numpy as np
    >>> box = image_slice_volume(
    ...     np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 2.0, 0]), (4, 3), 1.5
    ... )
    >>> box.centre_mm.tolist(), box.size_mm.tolist()
    ([1.5, 2.0, 0.0], [4.0, 6.0, 1.5])
    """
    row_step = np.asarray(row_step_mm, dtype=float)
    column_step = np.asarray(column_step_mm, dtype=float)
    n_rows, n_columns = (int(n) for n in shape)
    if n_rows < 1 or n_columns < 1:
        raise ValueError(f"image_slice_volume: shape {shape} needs at least one pixel.")
    thickness = float(thickness_mm)
    if not np.isfinite(thickness) or thickness < 0:
        raise ValueError(
            f"image_slice_volume: thickness_mm must be finite and >= 0; got {thickness_mm!r}."
        )
    row_length, column_length = np.linalg.norm(row_step), np.linalg.norm(column_step)
    normal = np.cross(row_step, column_step)
    if row_length == 0 or column_length == 0 or np.linalg.norm(normal) == 0:
        raise ValueError(
            "image_slice_volume: the row and column steps must be non-zero and "
            f"not parallel; got {row_step.tolist()} and {column_step.tolist()} mm."
        )
    centre = (
        np.asarray(origin_mm, dtype=float)
        + (n_rows - 1) / 2 * row_step
        + (n_columns - 1) / 2 * column_step
    )
    return ExcitedVolume(
        centre_mm=centre,
        axes=np.stack(
            [
                row_step / row_length,
                column_step / column_length,
                normal / np.linalg.norm(normal),
            ]
        ),
        size_mm=np.array([n_rows * row_length, n_columns * column_length, thickness]),
        kind="slice",
    )


__all__ = ["ExcitedVolume", "excited_volume", "image_slice_volume", "outline_in_plane"]
