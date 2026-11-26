"""
Coordinate system utilities for hypermri.

This package contains coordinate transformation and spatial grid functions
extracted from the monolithic utils_general.py as part of Phase 2 refactoring.

The coordinate system utilities are organized into:
- transformations: Scanner coordinate system transformations and matrix calculations
- grids: Spatial grid generation and management
"""

from .alignment import (
    ScanAlignmentResult,
    apply_alignment_to_data,
    compute_scan_alignment,
    get_2d_extent_from_alignment,
)
from .grids import (
    assign_xyz_coordinates,
    create_isotropic_grid,
    define_grid_lines,
    define_xyz_grid,
    get_rps_axes,
    get_rps_extent,
    get_rps_grid,
    get_xyz_axes,
    get_xyz_extent,
    get_xyz_extent_for_coordinate,
    get_xyz_grid,
    get_xyz_grid_from_visu,
    has_xyz_coordinates,
    resample_grid,
)
from .image_shifts import (
    align_image_data,
    apply_image_alignment,
    calc_mat_origin_diff,
    determine_FOV_shift,
    estimate_image_alignment,
    get_orientation_flip_multipliers,
    plot_alignment_overview,
    shift_image,
)
from .transformations import (
    COORDINATE_PRECISION,
    define_imagematrix_parameters,
    translate_coordinates,
)

__all__ = [
    # Cross-scan alignment
    "ScanAlignmentResult",
    "compute_scan_alignment",
    "apply_alignment_to_data",
    "get_2d_extent_from_alignment",
    # Transformations
    "define_imagematrix_parameters",
    "translate_coordinates",
    "COORDINATE_PRECISION",
    # Grids
    "assign_xyz_coordinates",
    "define_grid_lines",
    "define_xyz_grid",
    "get_rps_axes",
    "get_rps_grid",
    "get_xyz_grid",
    "get_xyz_grid_from_visu",
    "has_xyz_coordinates",
    "get_xyz_axes",
    "get_rps_extent",
    "get_xyz_extent",
    "get_xyz_extent_for_coordinate",
    "create_isotropic_grid",
    "resample_grid",
    # Image shifts
    "align_image_data",
    "apply_image_alignment",
    "calc_mat_origin_diff",
    "shift_image",
    "determine_FOV_shift",
    "get_orientation_flip_multipliers",
    "estimate_image_alignment",
    "plot_alignment_overview",
]
