"""
Coordinate system transformations for MRI scanner geometry.

This module handles coordinate transformations between different reference systems:
- Scanner coordinates (read, phase, slice - RPS)
- Physical coordinates (x, y, z - XYZ)
- Patient position and orientation handling

Extracted from utils_general.py as part of Phase 2 refactoring.
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)

#: Default dtype of scanner and RPS coordinates in mm.
#: ``float64`` keeps sub-micrometre resolution far from the isocentre; at
#: 100 mm ``float16`` resolves only 0.0625 mm.  This is the only definition;
#: other coordinate modules import it from here.
COORDINATE_PRECISION = np.float64

# The geometry position tolerance in mm, the tolerance of the geometry
# reference tests: DICOM positions carry six decimals in mm, and a wrong
# slice step of 1 mm exceeds it by three orders of magnitude.  This is the only
# definition; the geometry checks of the coordinate and reconstruction modules
# import it from here.
_POSITION_ATOL_MM = 1e-3


def define_imagematrix_parameters(
    scan=None,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Return the voxel counts along z, x and y from the orientation names.

    The read and phase matrix and the slice count, or the third matrix
    element of 3D scans, are assigned to the scanner axes by the slice and
    read orientation.  Offsets, directions and oblique angles are ignored,
    and the patient position only selects the table: Head_Prone and
    Foot_Prone share one table for six orientation pairs, while the
    Head_Supine table covers only axial scans.

    Parameters
    ----------
    scan : object, optional
        Scan or vendor handler with the attributes ``read_orientation``,
        ``slice_orientation``, ``patient_position`` (or ``patient_pos``) and
        ``matrix`` with at least the read and phase voxel counts.
        ``spatial_dimension``, ``"2D"`` or ``"3D"``, and
        ``n_slices`` determine the slice count.  The fallbacks
        ``patient_pos`` and ``spatial_dimension`` of a scan are read from its vendor
        handler ``scan.handler``.

    Returns
    -------
    dim_z, dim_x, dim_y : int or None
        Voxel counts along z, x and y.  All three are None without
        ``scan``, and for Head_Supine scans other than axial L_R or A_P.

    Raises
    ------
    ValueError
        If an orientation, the patient position or the read and phase
        matrix is missing.
    Exception
        If the patient position or the Head_Prone orientation pair is not in
        the table.  ParaVision writes the feet-first positions as
        ``"Foot_Prone"`` and ``"Foot_Supine"`` in ``ACQ_patient_pos``;
        ``"Feet_Prone"`` and ``"Feet_Supine"`` are rejected.

    Warns
    -----
    UserWarning
        For Head_Supine scans, whose table is incomplete.
    """
    # Initialize empty return values
    dim_z = dim_x = dim_y = None

    if scan is None:
        return dim_z, dim_x, dim_y

    read_orient = getattr(scan, "read_orientation", None)
    slice_orient = getattr(scan, "slice_orientation", None)
    patient_pos = getattr(scan, "patient_position", None)
    # The vendor handler of a scan holds the raw ParaVision fallbacks.
    vendor_source = getattr(scan, "handler", scan)
    if patient_pos is None:
        patient_pos = getattr(vendor_source, "patient_pos", None)

    matrix = np.atleast_1d(getattr(scan, "matrix", []))
    n_slices = getattr(scan, "n_slices", None)
    if n_slices is not None:
        n_slices = int(np.atleast_1d(n_slices)[0])
    spatial_dimension = getattr(scan, "spatial_dimension", None)
    if spatial_dimension is None:
        spatial_dimension = getattr(vendor_source, "spatial_dimension", None)
    if spatial_dimension is not None:
        spatial_dimension = str(spatial_dimension).strip("<>")

    if read_orient is None:
        raise ValueError("Missing read_orientation on data object.")
    if slice_orient is None:
        raise ValueError("Missing slice_orientation on data object.")
    if patient_pos is None:
        raise ValueError("Missing patient_position on data object.")
    if matrix.size < 2:
        raise ValueError("Missing matrix with at least read and phase dimensions.")

    if patient_pos in ("Head_Prone", "Foot_Prone"):
        # All axial slice options:
        if slice_orient == "axial":
            if read_orient == "L_R":
                # x - dimension
                dim_x = matrix[0]
                # y - dimension
                dim_y = matrix[1]
                # z - dimension
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_z = matrix[2]
                else:
                    dim_z = n_slices
            elif read_orient == "A_P":
                # x - dimension
                dim_x = matrix[1]
                # y - dimension
                dim_y = matrix[0]
                # z - dimension
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_z = matrix[2]
                else:
                    dim_z = n_slices
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        # All sagittal slice options:
        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                # y - dimension
                dim_y = matrix[1]
                # z - dimension
                dim_z = matrix[0]
                # x - dimension
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_x = matrix[2]
                else:
                    dim_x = n_slices
            elif read_orient == "A_P":
                # y - dimension
                dim_y = matrix[0]
                # z - dimension
                dim_z = matrix[1]
                # x - dimension
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_x = matrix[2]
                else:
                    dim_x = n_slices
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        # All coronal slice options:
        elif slice_orient == "coronal":
            if read_orient == "H_F":
                # z - dimension
                dim_z = matrix[0]
                # x - dimension
                dim_x = matrix[1]
                # y - dimension
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_y = matrix[2]
                else:
                    dim_y = n_slices
            elif read_orient == "L_R":
                # z - dimension
                dim_z = matrix[1]
                # x - dimension
                dim_x = matrix[0]
                # y - dimension
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_y = matrix[2]
                else:
                    dim_y = n_slices
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "axial sagittal coronal":
            raise Exception(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )
        else:
            raise Exception("unknown slice orientation: " + slice_orient)

    elif patient_pos == "Head_Supine":
        warnings.warn("Head_Supine is not fully implemented yet!", stacklevel=2)
        # Implementation for Head_Supine would go here
        # For now, using same logic as Head_Prone
        if slice_orient == "axial":
            if read_orient == "L_R":
                dim_x = matrix[0]
                dim_y = matrix[1]
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_z = matrix[2]
                else:
                    dim_z = n_slices
            elif read_orient == "A_P":
                dim_x = matrix[1]
                dim_y = matrix[0]
                if spatial_dimension == "3D" and matrix.size > 2:
                    dim_z = matrix[2]
                else:
                    dim_z = n_slices
        # Add other orientations as needed
    else:
        raise Exception(f"Unknown patient position: {patient_pos}")

    return dim_z, dim_x, dim_y


def translate_coordinates(original_coords, fov, matrix_size):
    """
    Translate coordinates based on field of view and matrix size.

    Parameters
    ----------
    original_coords : array_like
        Original coordinate values
    fov : float
        Field of view in mm
    matrix_size : int
        Matrix size in voxels

    Returns
    -------
    array_like
        Translated coordinates
    """
    voxel_size = fov / matrix_size
    center_offset = fov / 2.0

    # Convert from voxel indices to physical coordinates
    physical_coords = (original_coords * voxel_size) - center_offset

    return physical_coords


def find_indexing_scanner_dim(data=None, object_coord_name=None, object_idx=None):
    """
    Find the scanner dimension with the least variation along a specified dimension.
    This helps to find the matching scanner dimension for a given object dimension.

    Parameters
    ----------
    data: xr.DataArray
        The input data containing coordinates.
    object_coord_name: str
        name of the object coordinate (e.g., 'read_idx', 'phase_idx', 'slice_idx')
    object_idx: int
        index of the object coordinate to analyze

    Example
    -------
    >>> from hypermri.utils.coordinates.transformations import find_indexing_scanner_dim, find_original_object_dimension
    >>> scanner_dim_1 = find_indexing_scanner_dim(data=t2_cor.data, object_coord_name='slice_idx', object_idx=10)
    >>> print(f"scanner_dim_1={scanner_dim_1}")
    >>> print(f"Scanner dimension for read_idx: {scanner_dim_1[0]}, index: {scanner_dim_1[1]}, closest actual value: {scanner_dim_1[2]}")

    >>> result = find_original_object_dimension(
    >>>     data=t2_ax.data,
    >>>     scanner_coord_name=scanner_dim_1[0],  # 'z'
    >>>     closest_actual_val=scanner_dim_1[2]   # -14.5
    >>> )
    >>> print(f"Corrected dimension mapping: {result}")
    >>> fig, ax = plt.subplots(1,2)
    >>> ax[0].imshow(np.squeeze(np.abs(t2_cor.data.isel(slice_idx=10))), aspect='auto')
    >>> ax[1].imshow(np.squeeze(np.abs(t2_ax.data.isel(**result))), aspect='auto')

    Returns
    -------
    tuple
        [min_key, coordinate, str] the scanner dimension with the least variation
    """
    coordinate_diffs = {}
    for scanner_dim in ["x", "y", "z"]:
        if scanner_dim in data.coords:
            coordinate_diffs[scanner_dim] = np.min(
                data.coords[scanner_dim].isel(**{object_coord_name: object_idx}).values
            ) - np.max(
                data.coords[scanner_dim].isel(**{object_coord_name: object_idx}).values
            )

    # find minimum mean --> means that the cooridantes along this dimension vary the least
    min_val = None
    for key, value in coordinate_diffs.items():
        if min_val is None or np.abs(value) < min_val:
            min_val = np.abs(value)
            min_key = key

    mean_val = np.mean(
        data.coords[min_key].isel(**{object_coord_name: object_idx}).values
    )

    mean_coords_1d = data.coords[object_coord_name]
    for object_dim in ["read_idx", "phase_idx", "slice_idx"]:
        if object_dim in data.coords and object_dim != object_coord_name:
            mean_coords_1d = mean_coords_1d.mean(dim=object_dim)
    # print(f"Mean value along {object_coord_name} = {mean_coords_1d.values}")

    index = np.argmin(np.abs(np.array(mean_coords_1d) - mean_val))
    closest_actual_val = mean_coords_1d.values[int(index)]

    return [min_key, int(index), closest_actual_val]  # Convert to Python int


def find_original_object_dimension(data, scanner_coord_name, closest_actual_val):
    """
    Find which object dimension corresponds to the scanner coordinate by analyzing
    the relationship across the entire coordinate space, not just at one point.

    Parameters
    ----------
    data: xarray DataArray
        containing the coordinates
    scanner_coord_name: str
        name of the scanner coordinate (e.g., 'x', 'y', 'z')
    closest_actual_val: float
        the actual value of the scanner coordinate at the desired index

    Example
    -------
    >>> from hypermri.utils.coordinates.transformations import find_indexing_scanner_dim, find_original_object_dimension
    >>> scanner_dim_1 = find_indexing_scanner_dim(data=t2_cor.data, object_coord_name='slice_idx', object_idx=10)
    >>> print(f"scanner_dim_1={scanner_dim_1}")
    >>> print(f"Scanner dimension for read_idx: {scanner_dim_1[0]}, index: {scanner_dim_1[1]}, closest actual value: {scanner_dim_1[2]}")

    >>> result = find_original_object_dimension(
    >>>     data=t2_ax.data,
    >>>     scanner_coord_name=scanner_dim_1[0],  # 'z'
    >>>     closest_actual_val=scanner_dim_1[2]   # -14.5
    >>> )
    >>> print(f"Corrected dimension mapping: {result}")
    >>> fig, ax = plt.subplots(1,2)
    >>> ax[0].imshow(np.squeeze(np.abs(t2_cor.data.isel(slice_idx=10))), aspect='auto')
    >>> ax[1].imshow(np.squeeze(np.abs(t2_ax.data.isel(**result))), aspect='auto')

    Returns
    -------
    dict
        mapping of the object dimension to the index in that dimension closest to the target
    """
    coord_array = data.coords[scanner_coord_name]
    dims = coord_array.dims  # ('read_idx', 'phase_idx', 'slice_idx')

    # For each object dimension, calculate how much the scanner coordinate
    # varies when that dimension changes (averaged across other dimensions)
    dimension_variations = {}

    for dim in dims:
        # Calculate variation of scanner coordinate along this dimension
        # by averaging over the other dimensions
        other_dims = [d for d in dims if d != dim]

        # Average the scanner coordinate over the other dimensions
        averaged_coord = coord_array.mean(dim=other_dims)

        # Calculate the variation (range) along this dimension
        variation = float(averaged_coord.max().values - averaged_coord.min().values)
        dimension_variations[dim] = variation

    # The dimension with the MOST variation is the one that corresponds
    # to our scanner coordinate
    max_variation_dim = max(dimension_variations, key=dimension_variations.get)

    # Now find the index in that dimension closest to our target
    averaged_coord = coord_array.mean(dim=[d for d in dims if d != max_variation_dim])
    index = np.argmin(np.abs(averaged_coord.values - closest_actual_val))

    return {max_variation_dim: int(index)}
