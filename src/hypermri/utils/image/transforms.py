"""
Image transformation and spatial utility functions.

Contains functions for defining the field of view and calculating plotting
extents from scan geometry attributes.
"""

import numpy as np

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def _get_geometry_attr(scan, *names, default=None):
    """Return the first available standardized geometry attribute."""
    for name in names:
        if hasattr(scan, name):
            value = getattr(scan, name)
            if value is not None:
                return value
    return default


def _as_first_float(value, default=None):
    if value is None:
        return default
    values = np.atleast_1d(value)
    if values.size == 0:
        return default
    return float(values[0])


def _as_first_int(value, default=None):
    first_value = _as_first_float(value, default=default)
    if first_value is None:
        return default
    return int(first_value)


def _normalise_spatial_dimension(value):
    if value is None:
        return None
    return str(value).strip("<>")


def _slice_fov_mm(scan, fov_mm):
    spatial_dimension = _normalise_spatial_dimension(
        _get_geometry_attr(scan, "spatial_dimension")
    )

    if spatial_dimension == "3D" and len(fov_mm) > 2:
        return fov_mm[2]

    slice_thickness_mm = _as_first_float(_get_geometry_attr(scan, "slice_thickness_mm"))
    n_slices = _as_first_int(_get_geometry_attr(scan, "n_slices"))
    slice_gap_mm = _as_first_float(
        _get_geometry_attr(scan, "slice_gap_mm"), default=0.0
    )

    if slice_thickness_mm is not None and n_slices is not None:
        return slice_thickness_mm * n_slices + slice_gap_mm * max(n_slices - 1, 0)

    if len(fov_mm) > 2:
        return fov_mm[2]

    raise ValueError(
        "Missing slice FOV information: need slice_thickness_mm and n_slices, "
        "or a 3-element fov_mm."
    )


def define_imageFOV_parameters(
    scan: object | None = None,
) -> tuple[float | None, float | None, float | None]:
    """
    Return the field of view along z, x and y from the orientation names.

    The read and phase field of view and the slice extent are assigned to
    the scanner axes by the slice and read orientation.  The slice extent is
    the third ``fov_mm`` element of 3D scans; otherwise it is the slice
    thickness times the slice count plus one gap between each two slices,
    or the third ``fov_mm`` element when these are missing.  Offsets,
    directions and oblique angles are ignored, and the patient position
    only selects the table: Head_Prone and Foot_Prone share one table,
    Head_Supine has its own.

    Parameters
    ----------
    scan : object, optional
        Scan or vendor handler with the attributes ``patient_position`` (or
        ``patient_pos``), ``read_orientation``, ``slice_orientation`` and
        ``fov_mm`` of shape (2,) or (3,), in read, phase and optional slice
        order, in mm. Numeric values are converted to float64.
        ``spatial_dimension`` is ``"2D"`` or ``"3D"``;
        ``slice_thickness_mm`` and ``slice_gap_mm`` are lengths in mm, and
        ``n_slices`` is a voxel count. These determine the slice extent.
        Supported positions are Head_Prone, Foot_Prone and Head_Supine;
        axial, sagittal or coronal slice orientation selects the valid
        L_R, A_P or H_F read orientation from the legacy table.

    Returns
    -------
    mm_z, mm_x, mm_y : float or None
        Field of view along z, x and y in mm. Read and phase lengths retain
        their pairing with the voxel counts of
        :func:`~hypermri.utils.coordinates.transformations.define_imagematrix_parameters`
        for all six Head_Prone/Foot_Prone orientation pairs, including
        coronal L_R. All three values are None without ``scan``.

    Raises
    ------
    ValueError
        If the patient position, an orientation or the read and phase field
        of view is missing, if the slice extent cannot be determined, or if
        the patient position or the orientation pair is not in the table.
        ParaVision writes the feet-first positions to ``ACQ_patient_pos`` as
        ``"Foot_Prone"`` and ``"Foot_Supine"``; ``"Feet_Prone"`` and
        ``"Feet_Supine"`` are rejected.

    Notes
    -----
    This deterministic metadata-only operation does not read signal data,
    reconstruct images, initialize plotting, cache results or mutate inputs.
    It returns axis lengths, not signed voxel positions or oblique geometry.
    The separate legacy Head_Supine table is retained; this function does not
    add supine support to the matrix or CSI plotting helpers.

    Examples
    --------
    Unequal read and phase lengths in a coronal L_R prone scan map to x and z:

    >>> from types import SimpleNamespace
    >>> scan = SimpleNamespace(
    ...     patient_position="Head_Prone", slice_orientation="coronal",
    ...     read_orientation="L_R", spatial_dimension="3D",
    ...     fov_mm=[8.0, 9.0, 6.0],
    ... )
    >>> np.asarray(define_imageFOV_parameters(scan)).tolist()
    [9.0, 8.0, 6.0]
    """
    mm_z = mm_x = mm_y = None

    if scan is None:
        return mm_z, mm_x, mm_y

    patient_pos = _get_geometry_attr(scan, "patient_position", "patient_pos")
    read_orient = _get_geometry_attr(scan, "read_orientation")
    slice_orient = _get_geometry_attr(scan, "slice_orientation")
    fov_mm = np.atleast_1d(_get_geometry_attr(scan, "fov_mm"))

    if patient_pos is None:
        raise ValueError("Missing patient_position on data object.")
    if read_orient is None:
        raise ValueError("Missing read_orientation on data object.")
    if slice_orient is None:
        raise ValueError("Missing slice_orientation on data object.")
    if fov_mm.size < 2 or fov_mm[0] is None:
        raise ValueError("Missing fov_mm with at least read and phase dimensions.")

    fov_mm = fov_mm.astype(float)
    slice_extent_mm = _slice_fov_mm(scan, fov_mm)

    logger.debug(
        "define_imageFOV_parameters: patient_pos=%s, read_orient=%s, slice_orient=%s",
        patient_pos,
        read_orient,
        slice_orient,
    )

    if patient_pos in ("Head_Prone", "Foot_Prone"):
        if slice_orient == "axial":
            if read_orient == "L_R":
                mm_x, mm_y, mm_z = fov_mm[0], fov_mm[1], slice_extent_mm
            elif read_orient == "A_P":
                mm_x, mm_y, mm_z = fov_mm[1], fov_mm[0], slice_extent_mm
            else:
                raise ValueError(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                mm_y, mm_z, mm_x = fov_mm[1], fov_mm[0], slice_extent_mm
            elif read_orient == "A_P":
                mm_y, mm_z, mm_x = fov_mm[0], fov_mm[1], slice_extent_mm
            else:
                raise ValueError(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "coronal":
            if read_orient == "H_F":
                mm_z, mm_x, mm_y = fov_mm[0], fov_mm[1], slice_extent_mm
            elif read_orient == "L_R":
                mm_z, mm_x, mm_y = fov_mm[1], fov_mm[0], slice_extent_mm
            else:
                raise ValueError(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "axial sagittal coronal":
            raise ValueError(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )
        else:
            raise ValueError("unknown slice orientation: " + slice_orient)

    elif patient_pos == "Head_Supine":
        if slice_orient == "axial":
            if read_orient == "L_R":
                mm_x, mm_y, mm_z = fov_mm[0], fov_mm[1], slice_extent_mm
            elif read_orient == "A_P":
                mm_x, mm_y, mm_z = fov_mm[1], fov_mm[0], slice_extent_mm
            else:
                raise ValueError(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "sagittal":
            if read_orient in ("H_F", "A_P"):
                mm_y, mm_z, mm_x = fov_mm[1], fov_mm[0], slice_extent_mm
            else:
                raise ValueError(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "coronal":
            if read_orient == "H_F":
                mm_x, mm_z, mm_y = fov_mm[1], fov_mm[0], slice_extent_mm
            elif read_orient == "L_R":
                mm_x, mm_z, mm_y = fov_mm[0], fov_mm[1], slice_extent_mm
            else:
                raise ValueError(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "axial sagittal coronal":
            raise ValueError(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )
        else:
            raise ValueError("unknown slice orientation: " + slice_orient)

    else:
        raise ValueError("unknown patient_position: " + patient_pos)

    return mm_z, mm_x, mm_y


def get_extent(scan=None):
    """
    Calculate extent (bounding box) of the data object.

    Returns
    -------
    ax_ext, sag_ext, cor_ext : list
        Extents for axial, sagittal, and coronal views.
    """
    if scan is None:
        return None, None, None

    try:
        read_orient = _get_geometry_attr(scan, "read_orientation")
        read_offset = _get_geometry_attr(scan, "read_offset_mm", default=0.0)
        phase_offset = _get_geometry_attr(
            scan, "phase_1_offset_mm", "phase_offset_mm", default=0.0
        )
        slice_offset = _get_geometry_attr(scan, "slice_offset_mm", default=0.0)
        mm_z, mm_x, mm_y = define_imageFOV_parameters(scan=scan)
        patient_pos = _get_geometry_attr(scan, "patient_position", "patient_pos")
        slice_orient = _get_geometry_attr(scan, "slice_orientation")

        if patient_pos == "Head_Supine":
            if slice_orient == "sagittal":
                if read_orient == "H_F":
                    phase_offset = -phase_offset
            elif slice_orient == "coronal":
                if read_orient == "H_F":
                    phase_offset = -phase_offset
                elif read_orient == "L_R":
                    read_offset = -read_offset
            if slice_orient == "axial":
                phase_offset = -phase_offset
                read_offset = -read_offset

        ax_ext = sag_ext = cor_ext = None

        if slice_orient == "coronal":
            if read_orient == "H_F":
                cor_ext = [
                    -mm_z / 2 - read_offset,
                    mm_z / 2 - read_offset,
                    -mm_x / 2 - phase_offset,
                    mm_x / 2 - phase_offset,
                ]
                ax_ext = [
                    -mm_x / 2 + phase_offset,
                    mm_x / 2 + phase_offset,
                    -mm_y / 2 + slice_offset,
                    mm_y / 2 + slice_offset,
                ]
                sag_ext = [
                    -mm_z / 2 + read_offset,
                    mm_z / 2 + read_offset,
                    -mm_y / 2 + slice_offset,
                    mm_y / 2 + slice_offset,
                ]
            elif read_orient == "L_R":
                cor_ext = [
                    -mm_z / 2 - phase_offset,
                    mm_z / 2 - phase_offset,
                    -mm_x / 2 - read_offset,
                    mm_x / 2 - read_offset,
                ]
                ax_ext = [
                    -mm_x / 2 + read_offset,
                    mm_x / 2 + read_offset,
                    -mm_y / 2 + slice_offset,
                    mm_y / 2 + slice_offset,
                ]
                sag_ext = [
                    -mm_z / 2 - phase_offset,
                    mm_z / 2 - phase_offset,
                    -mm_y / 2 + slice_offset,
                    mm_y / 2 + slice_offset,
                ]

        elif slice_orient == "axial":
            if read_orient == "L_R":
                cor_ext = [
                    -mm_x / 2 + read_offset,
                    mm_x / 2 + read_offset,
                    -mm_z / 2 + slice_offset,
                    mm_z / 2 + slice_offset,
                ]
                ax_ext = [
                    -mm_x / 2 - read_offset,
                    mm_x / 2 - read_offset,
                    -mm_y / 2 - phase_offset,
                    mm_y / 2 - phase_offset,
                ]
                sag_ext = [
                    -mm_z / 2 + slice_offset,
                    mm_z / 2 + slice_offset,
                    -mm_y / 2 + phase_offset,
                    mm_y / 2 + phase_offset,
                ]
            elif read_orient == "A_P":
                cor_ext = [
                    -mm_x / 2 + phase_offset,
                    mm_x / 2 + phase_offset,
                    -mm_z / 2 + slice_offset,
                    mm_z / 2 + slice_offset,
                ]
                ax_ext = [
                    -mm_x / 2 - phase_offset,
                    mm_x / 2 - phase_offset,
                    -mm_y / 2 - read_offset,
                    mm_y / 2 - read_offset,
                ]
                sag_ext = [
                    -mm_z / 2 + slice_offset,
                    mm_z / 2 + slice_offset,
                    -mm_y / 2 + read_offset,
                    mm_y / 2 + read_offset,
                ]

        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                cor_ext = [
                    -mm_x / 2 + slice_offset,
                    mm_x / 2 + slice_offset,
                    -mm_z / 2 + read_offset,
                    mm_z / 2 + read_offset,
                ]
                ax_ext = [
                    -mm_x / 2 + slice_offset,
                    mm_x / 2 + slice_offset,
                    -mm_y / 2 + phase_offset,
                    mm_y / 2 + phase_offset,
                ]
                sag_ext = [
                    -mm_z / 2 - read_offset,
                    mm_z / 2 - read_offset,
                    -mm_y / 2 - phase_offset,
                    mm_y / 2 - phase_offset,
                ]
            elif read_orient == "A_P":
                cor_ext = [
                    -mm_x / 2 + slice_offset,
                    mm_x / 2 + slice_offset,
                    -mm_z / 2 + phase_offset,
                    mm_z / 2 + phase_offset,
                ]
                ax_ext = [
                    -mm_x / 2 + slice_offset,
                    mm_x / 2 + slice_offset,
                    -mm_y / 2 + read_offset,
                    mm_y / 2 + read_offset,
                ]
                sag_ext = [
                    -mm_z / 2 - phase_offset,
                    mm_z / 2 - phase_offset,
                    -mm_y / 2 - read_offset,
                    mm_y / 2 - read_offset,
                ]

        scan.ax_ext = ax_ext
        scan.sag_ext = sag_ext
        scan.cor_ext = cor_ext
        return ax_ext, sag_ext, cor_ext

    except Exception as e:
        logger.error(f"Error in get_extent: {e}")
        return None, None, None


def get_plotting_extent(scan=None):
    """Return the extent for plotting 2D data."""
    if scan is None:
        return [0, 1, 0, 1]

    try:
        ax_ext, sag_ext, cor_ext = get_extent(scan=scan)
        patient_pos = _get_geometry_attr(scan, "patient_position", "patient_pos")
        read_orient = _get_geometry_attr(scan, "read_orientation")
        slice_orient = _get_geometry_attr(scan, "slice_orientation")

        if patient_pos == "Head_Prone":
            if slice_orient == "axial" and read_orient == "L_R":
                return ax_ext
            elif slice_orient == "sagittal" and read_orient == "H_F":
                return sag_ext
            elif slice_orient == "coronal":
                return cor_ext
        return ax_ext if ax_ext is not None else [0, 1, 0, 1]
    except BaseException:
        return [0, 1, 0, 1]
