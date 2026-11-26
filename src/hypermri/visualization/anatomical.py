"""Static anatomical voxel grids and physical scale bars."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar

from hypermri.analysis.spectroscopy.schema import ensure_standard_spectroscopy_dataarray
from hypermri.utils.system.logging import init_default_logger

logger = init_default_logger(__name__)
ORIENTATION_TO_DIM = {
    "axial": "slice_idx",
    "sagittal": "read_idx",
    "coronal": "phase_idx",
}


def plot_voxel_grids(image1=None, image2=None, plot_params=None):
    """
    Plots two images side by side, each with an overlaid grid to highlight their respective voxel (pixel) sizes.
    The function supports rotation and custom color mapping for the visualization of each image. If only one image is provided,
    it will be displayed with its grid. The function allows customization of the slice index, line color, rotation,
    and colormap for each image through the `plot_params` dictionary.

    Parameters
    ----------
    image1 : ndarray, optional
        A 3D or higher-dimensional array representing the first image. The function expects the image to be in a specific
        format where the relevant slice is indexed as `image1[0, slice_ind_image1, :, :, 0, 0]`. If `None`, the function
        will not display this image.
    image2 : ndarray, optional
        A 3D or higher-dimensional array representing the second image. Similar to `image1`, it is expected to be in a
        format where the relevant slice is indexed as `image2[0, slice_ind_image2, :, :, 0, 0]`. If `None`, only `image1`
        will be displayed if it is provided.
    plot_params : dict, optional
        A dictionary containing parameters for plotting. Supported parameters include:
        - `slice_ind_image1` (int): The slice index for `image1`. Defaults to 0.
        - `slice_ind_image2` (int): The slice index for `image2`. Defaults to 0.
        - `line_color_image1` (str): Line color for the grid overlaid on `image1`. Defaults to "yellow".
        - `line_color_image2` (str): Line color for the grid overlaid on `image2`. Defaults to "white".
        - `rot_image` (int): Number of times the images should be rotated by 90 degrees. Defaults to 1.
        - `cmap` (str): The colormap used for both images. Defaults to "jet".

    Returns
    -------
    None
        The function does not return a value but displays a matplotlib plot with the images and their grids.

    Raises
    ------
    ValueError
        If `image1` is not provided, a critical log message is displayed indicating that `image1` must be passed.

    Notes
    -----
    The function is designed to visually compare two images of potentially different resolutions by overlaying a grid on each.
    This can be particularly useful for examining the alignment of voxel data in medical imaging or comparing model predictions
    against ground truth in machine learning tasks.
    Use :func:`hypermri.analysis.anatomical.masks.project_mask_to_data_grid`
    to project voxel coverage between grids in scanner coordinates.

    Examples
    --------
    >>> plot_voxel_grids(image1=my_image_array, plot_params={'slice_ind_image1': 10, 'linecolor_image1': 'blue'})
    This will display `my_image_array` with a slice index of 10 and a blue grid overlaid.
    """
    if plot_params is None:
        plot_params = {}
    figsize = plot_params.get("figsize", (12, 6))
    slice_ind_image1 = plot_params.get("slice_ind_image1", 0)
    slice_ind_image2 = plot_params.get("slice_ind_image2", 0)
    linecolor_image1 = plot_params.get("linecolor_image1", "yellow")
    linecolor_image2 = plot_params.get("linecolor_image2", "white")
    linewidth_image1 = plot_params.get("linewidth_image1", 0.5)
    linewidth_image2 = plot_params.get("linewidth_image2", 0.5)
    linestyle_image1 = plot_params.get("linestyle_image1", "-")
    linestyle_image2 = plot_params.get("linestyle_image2", "-")
    colorbar_image1 = plot_params.get("colorbar_image1", False)
    colorbar_image2 = plot_params.get("colorbar_image2", False)
    colorbar_label_image1 = plot_params.get("colorbar_label_image1", "signal [a.u]")
    colorbar_label_image2 = plot_params.get("colorbar_label_image2", "signal [a.u]")
    vmin_image1 = plot_params.get("vmin_image1", None)
    vmax_image1 = plot_params.get("vmax_image1", None)
    vmin_image2 = plot_params.get("vmin_image2", None)
    vmax_image2 = plot_params.get("vmax_image2", None)
    rot_image = plot_params.get("rot_image", 1)
    cmap_image1 = plot_params.get("cmap_image1", "jet")
    cmap_image2 = plot_params.get("cmap_image2", "jet")
    title_image1 = plot_params.get("title_image1", "")
    title_image2 = plot_params.get("title_image2", "")
    savepath = plot_params.get("savepath", None)

    orientation_image1 = plot_params.get("orientation_image1", "axial")
    orientation_image2 = plot_params.get("orientation_image2", "coronal")

    orientation_dim_image1 = ORIENTATION_TO_DIM.get(orientation_image1, "slice_idx")
    orientation_dim_image2 = ORIENTATION_TO_DIM.get(orientation_image2, "slice_idx")

    extent_image1 = plot_params.get("extent_image1", None)
    extent_image2 = plot_params.get("extent_image2", None)

    if extent_image1 is not None and extent_image2 is None:
        extent_image2 = extent_image1

    if extent_image2 is not None and extent_image1 is None:
        extent_image1 = extent_image2

    if extent_image1 is None and extent_image2 is None:
        extent_image1 = [0, 1, 0, 1]
        extent_image2 = [0, 1, 0, 1]

    startx_image1 = extent_image1[0]
    endx_image1 = extent_image1[1]
    starty_image1 = extent_image1[2]
    endy_image1 = extent_image1[3]

    startx_image2 = extent_image2[0]
    endx_image2 = extent_image2[1]
    starty_image2 = extent_image2[2]
    endy_image2 = extent_image2[3]

    provided_dims = plot_params.get("provided_dims", ["x", "y"])

    if image1 is not None and image2 is not None:
        image1_dims = provided_dims if np.ndim(image1) == len(provided_dims) else None
        image2_dims = provided_dims if np.ndim(image2) == len(provided_dims) else None
        image1 = ensure_standard_spectroscopy_dataarray(
            data=image1, provided_dims=image1_dims
        )
        image2 = ensure_standard_spectroscopy_dataarray(
            data=image2, provided_dims=image2_dims
        )
        print(f"image1.shape={image1.shape}")
        print(f"image2.shape={image2.shape}")
        image1 = np.squeeze(
            image1.isel({orientation_dim_image1: slice_ind_image1}).values
        )
        image2 = np.squeeze(
            image2.isel({orientation_dim_image2: slice_ind_image2}).values
        )
        print(f"image1.shape={image1.shape}")
        print(f"image2.shape={image2.shape}")
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Overlay the low-resolution image
        im2 = ax[0].imshow(
            np.rot90(image1, rot_image),
            extent=extent_image1,
            alpha=1,
            cmap=cmap_image1,
            vmin=vmin_image1,
            vmax=vmax_image1,
        )
        ax[0].set_title(title_image1)

        # Add grid for high-resolution image
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax[0].axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax[0].axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Add grid for low-resolution image
        for x in np.linspace(startx_image2, endx_image2, image2.shape[0] + 1):
            ax[0].axvline(
                x,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )
        for y in np.linspace(starty_image2, endy_image2, image2.shape[1] + 1):
            ax[0].axhline(
                y,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )

        # Set the limits and labels
        ax[0].set_xlim(startx_image1, endx_image1)
        ax[0].set_ylim(starty_image1, endy_image1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # # Display the high-resolution image
        im1 = ax[1].imshow(
            np.rot90(image2, rot_image),
            extent=extent_image2,
            alpha=1,
            cmap=cmap_image2,
            vmin=vmin_image2,
            vmax=vmax_image2,
        )

        # Add grid for high-resolution image
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax[1].axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax[1].axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Add grid for low-resolution image
        for x in np.linspace(startx_image2, endx_image2, image2.shape[0] + 1):
            ax[1].axvline(
                x,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )
        for y in np.linspace(starty_image2, endy_image2, image2.shape[1] + 1):
            ax[1].axhline(
                y,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )

        # Set the limits and labels
        ax[1].set_xlim(startx_image2, endx_image2)
        ax[1].set_ylim(starty_image2, endy_image2)
        ax[1].set_title(title_image2)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.tight_layout()
        plt.show()

        if colorbar_image2:
            fig.colorbar(im2, ax=ax[0], label=colorbar_label_image2, shrink=0.8)
        if colorbar_image1:
            fig.colorbar(im1, ax=ax[1], label=colorbar_label_image1, shrink=0.8)

        plt.tight_layout()

    elif image1 is not None and image2 is None:
        image1_dims = provided_dims if np.ndim(image1) == len(provided_dims) else None
        image1 = ensure_standard_spectroscopy_dataarray(
            data=image1, provided_dims=image1_dims
        )
        image1 = image1.isel({orientation_dim_image1: slice_ind_image1}).values
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Overlay the low-resolution image
        im1 = ax.imshow(
            np.rot90(image1, rot_image),
            extent=extent_image1,
            alpha=1,
            cmap=cmap_image1,
            vmin=vmin_image1,
            vmax=vmax_image1,
        )

        # Add grid
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax.axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax.axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Set the limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Set the limits and labels
        ax.set_xlim(startx_image1, endx_image1)
        ax.set_ylim(starty_image1, endy_image1)
        ax.set_title(title_image1)
        if colorbar_image1:
            fig.colorbar(im1, ax=ax, label=colorbar_label_image1, shrink=0.8)
        plt.show()
        plt.tight_layout()
    else:
        logger.critical("image1 has to be passed!")
    if savepath is not None:
        try:
            # plt.savefig(savepath + ".svg", format="svg", dpi=600, transparent=True)
            plt.savefig(savepath + ".png", format="png", dpi=600, transparent=True)
        except BaseException:
            raise Warning(f"image could not be save as {savepath}.png/.svg")


def add_scalebar(
    px,
    ax=None,
    unit="mm",
    fixed_value=None,
    color="k",
    box_alpha=0.0,
    box_color="w",
    location="lower right",
    frameon=False,
    pad=0.075,
    length_fraction=None,
    border_pad=0.075,
    fontsize=12,
):
    """
    Add scalebar to image, given the pixel size 'px'.
    """
    if not ax:
        ax = plt.gca()
    ax.add_artist(
        ScaleBar(
            px,
            unit=unit,
            fixed_value=fixed_value,
            color=color,
            box_alpha=box_alpha,
            box_color=box_color,
            location=location,
            frameon=frameon,
            pad=pad,
            length_fraction=length_fraction,
            border_pad=border_pad,
            font_properties={"size": fontsize},
        )
    )
