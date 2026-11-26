"""Plot and manipulate single-voxel spectroscopy data with shared logging."""

# This file contains utility functions for single voxel spectroscopy methods such as PRESS, STEAM, semi-lASER, ISIS and their
# multi-voxel implementations mv-PRESS, mv-semi-LASER

import numpy as np

from ...utils.system.logging import LOG_MODES as LOG_MODES
from ...utils.system.logging import init_default_logger

logger = init_default_logger(__name__)


def _get_attr(obj, name, default=None):
    value = getattr(obj, name, default)
    return default if value is None else value


def define_extent_press(img_instance):
    """Calculate the 2D PRESS plotting extent from standardized geometry."""
    fov_mm = np.atleast_1d(_get_attr(img_instance, "fov_mm"))
    if fov_mm.size < 2:
        raise ValueError("Missing fov_mm with read and phase dimensions.")

    read_fov_mm = float(fov_mm[0])
    phase_fov_mm = float(fov_mm[1])
    read_offset_mm = float(_get_attr(img_instance, "read_offset_mm", 0.0))
    phase_offset_mm = float(
        _get_attr(
            img_instance,
            "phase_1_offset_mm",
            _get_attr(img_instance, "phase_offset_mm", 0.0),
        )
    )
    read_orientation = _get_attr(img_instance, "read_orientation")
    slice_orientation = _get_attr(img_instance, "slice_orientation")

    if read_orientation is None:
        raise ValueError("Missing read_orientation on image instance.")
    if slice_orientation is None:
        raise ValueError("Missing slice_orientation on image instance.")

    if slice_orientation == "coronal":
        if read_orientation == "H_F":
            return [
                phase_fov_mm / 2 + phase_offset_mm,
                -phase_fov_mm / 2 + phase_offset_mm,
                -read_fov_mm / 2 + read_offset_mm,
                read_fov_mm / 2 + read_offset_mm,
            ]
        if read_orientation == "L_R":
            return [
                read_fov_mm / 2 + read_offset_mm,
                -read_fov_mm / 2 + read_offset_mm,
                -phase_fov_mm / 2 + phase_offset_mm,
                phase_fov_mm / 2 + phase_offset_mm,
            ]

    elif slice_orientation == "axial":
        if read_orientation == "L_R":
            return [
                read_fov_mm / 2 + read_offset_mm,
                -read_fov_mm / 2 + read_offset_mm,
                -phase_fov_mm / 2 + phase_offset_mm,
                phase_fov_mm / 2 + phase_offset_mm,
            ]
        if read_orientation == "A_P":
            return [
                phase_fov_mm / 2 + phase_offset_mm,
                -phase_fov_mm / 2 + phase_offset_mm,
                -read_fov_mm / 2 + read_offset_mm,
                read_fov_mm / 2 + read_offset_mm,
            ]

    elif slice_orientation == "sagittal":
        if read_orientation == "H_F":
            return [
                phase_fov_mm / 2 + phase_offset_mm,
                -phase_fov_mm / 2 + phase_offset_mm,
                -read_fov_mm / 2 + read_offset_mm,
                read_fov_mm / 2 + read_offset_mm,
            ]
        return [
            read_fov_mm / 2 + read_offset_mm,
            -read_fov_mm / 2 + read_offset_mm,
            -phase_fov_mm / 2 + phase_offset_mm,
            phase_fov_mm / 2 + phase_offset_mm,
        ]

    raise KeyError(
        f"Unknown read orientation {read_orientation!r} for slice orientation {slice_orientation!r}."
    )


def show_voxel_overlap(centers, sizes):
    """
    Computes the overlap of voxels in a multi-voxel sequence like mv-press or mv-sLASER and shows it as a pandas dataframe.
    Written using ChatGPT 4o.

    Parameters
    ----------
    centers: np.array, voxel locations, i.e. mrs.method['PVM_VoxArrPosition']
    sizes: np.array,  method['PVM_VoxArrSize']

    Returns
    -------
    overlap_x_data
    overlap_y_data
    overlap_z_data
    total_overlap_data
    styled_df_x
    styled_df_y
    styled_df_z
    styled_df_total
    """

    def compute_bounds(center, size):
        """Calculate the minimum and maximum bounds of a voxel."""
        half_size = size / 2.0
        min_bound = center - half_size
        max_bound = center + half_size
        return min_bound, max_bound

    def overlap_amount(min1, max1, min2, max2):
        """Calculate the overlap between two intervals."""
        if max1 > min2 and min1 < max2:  # Overlap condition
            return min(max1, max2) - max(min1, min2)
        return 0  # No overlap

    def find_overlaps(centers, sizes):
        """Find overlaps between voxels given their centers and sizes."""
        n = len(centers)
        overlap_x_data = np.zeros((n, n))
        overlap_y_data = np.zeros((n, n))
        overlap_z_data = np.zeros((n, n))
        total_overlap_data = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Get bounds for voxel i
                min_i, max_i = compute_bounds(centers[i], sizes[i])
                # Get bounds for voxel j
                min_j, max_j = compute_bounds(centers[j], sizes[j])

                # Check overlaps in each dimension
                overlap_x = overlap_amount(min_i[0], max_i[0], min_j[0], max_j[0])
                overlap_y = overlap_amount(min_i[1], max_i[1], min_j[1], max_j[1])
                overlap_z = overlap_amount(min_i[2], max_i[2], min_j[2], max_j[2])

                # Fill the matrices symmetrically
                overlap_x_data[i, j] = round(overlap_x, 1)
                overlap_x_data[j, i] = round(overlap_x, 1)
                overlap_y_data[i, j] = round(overlap_y, 1)
                overlap_y_data[j, i] = round(overlap_y, 1)
                overlap_z_data[i, j] = round(overlap_z, 1)
                overlap_z_data[j, i] = round(overlap_z, 1)

                # Calculate the average percentage overlap across dimensions
                size_i = np.array(sizes[i])
                size_j = np.array(sizes[j])
                avg_size = (size_i + size_j) / 2.0

                overlap_x_percentage = (
                    (overlap_x / avg_size[0]) * 100 if avg_size[0] > 0 else 0
                )
                overlap_y_percentage = (
                    (overlap_y / avg_size[1]) * 100 if avg_size[1] > 0 else 0
                )
                overlap_z_percentage = (
                    (overlap_z / avg_size[2]) * 100 if avg_size[2] > 0 else 0
                )

                total_overlap_percentage = (
                    overlap_x_percentage + overlap_y_percentage + overlap_z_percentage
                ) / 3
                total_overlap_data[i, j] = round(total_overlap_percentage, 1)
                total_overlap_data[j, i] = round(total_overlap_percentage, 1)

        return (
            overlap_x_data,
            overlap_y_data,
            overlap_z_data,
            total_overlap_data,
            avg_size,
        )

    def visualize_overlaps(
        overlap_x_data, overlap_y_data, overlap_z_data, total_overlap_data, avg_size
    ):
        """Visualize the overlap matrices using color-coded DataFrames."""
        import pandas as pd
        from IPython.display import display

        n = overlap_x_data.shape[0]
        index_labels = [f"voxel {i}" for i in range(n)]

        # Create DataFrames for each axis and the total overlap
        df_x = pd.DataFrame(overlap_x_data, columns=index_labels, index=index_labels)
        df_y = pd.DataFrame(overlap_y_data, columns=index_labels, index=index_labels)
        df_z = pd.DataFrame(overlap_z_data, columns=index_labels, index=index_labels)
        df_total = pd.DataFrame(
            total_overlap_data, columns=index_labels, index=index_labels
        )

        # Define a color map for the overlap ranges
        def color_map(val):
            if val > 50:
                color = "red"
            elif val > 0:
                color = "orange"
            else:
                color = "green"
            return f"background-color: {color}"

        # Style the DataFrames

        styled_df_x = (
            df_x.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Overlap in X Axis")
        )
        styled_df_y = (
            df_y.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Overlap in Y Axis")
        )
        styled_df_z = (
            df_z.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Overlap in Z Axis")
        )
        styled_df_total = (
            df_total.style.applymap(color_map)
            .format("{:.1f}")
            .set_caption("Total Overlap Percentage")
        )

        # Display DataFrames (for Jupyter Notebooks)
        display(styled_df_x)
        display(styled_df_y)
        display(styled_df_z)
        display(styled_df_total)

        return styled_df_x, styled_df_y, styled_df_z, styled_df_total

    # Step 1: Calculate overlaps
    overlap_x_data, overlap_y_data, overlap_z_data, total_overlap_data, avg_size = (
        find_overlaps(centers, sizes)
    )

    # Step 2: Visualize overlaps
    styled_df_x, styled_df_y, styled_df_z, styled_df_total = visualize_overlaps(
        overlap_x_data, overlap_y_data, overlap_z_data, total_overlap_data, avg_size
    )
    return (
        overlap_x_data,
        overlap_y_data,
        overlap_z_data,
        total_overlap_data,
        styled_df_x,
        styled_df_y,
        styled_df_z,
        styled_df_total,
    )
