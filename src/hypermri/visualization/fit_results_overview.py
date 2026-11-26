"""Static anatomical overviews of labelled CSI fit results.

Existing plotting conventions and defaults are retained. Inputs are processed
only when the documented operations are called; importing reads no scans.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm.auto import tqdm

from hypermri import Study
from hypermri.compatibility import resolve_frequency_axis
from hypermri.utils.image.interpolation import img_interp
from hypermri.visualization.anatomical import add_scalebar

# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================


def plot_fit_results_overview(
    CSI_BrukerExp,
    fit_spectra=None,
    fit_chemical_shift_axis_ppm=None,
    fit_amps=None,
    rep=None,
    CSI_data=None,
    marked_voxel=None,
    anatomical_BrukerExp=None,
    image_stack=None,
    anatomical_image_numbers=None,
    plot_params=None,
    fit_params=None,
):
    """
    Plot an overview of the CSI fit results
    Parameters
    ----------
    CSI_BrukerExp: Bruker experiment object
        Contains CSI data
    fit_spectra: nd numpy array
        Array of fitted spectra
    fit_chemical_shift_axis_ppm: 1d numpy array
        Array of chemical shifts sampled in fitted_spectra
    fit_amps: nd numpy array
        Array of fitted amplitudes
    rep: int/range
        Repetition to be plotted
    CSI_data: nd numpy array
        The CSI data that should be plotted in overlay with fit_spectra
    marked_voxel: [i,j] tuple
        Indices of marked spectrum in overlay
    anatomical_BrukerExp: Bruker experiment object
        Contains anatomical reference images
    anatomical_image_numbers: list(int)
        Number of slices chosen for overlay (indexing starts at 0)
    plot_params: dict
        Plot parameters

    Returns
    -------
    A multisubplot figure

    Examples
    -------

    """

    if CSI_data is None and fit_spectra is None:
        raise ValueError("Pass either CSI_data or/and fit_spectra!")

    if plot_params is None:
        plot_params = {}
    if fit_params is None:
        fit_params = {}

    chemical_shift_axis_ppm = resolve_frequency_axis(CSI_BrukerExp.data, "ppm")

    if fit_spectra is None:
        fit_spectra = np.zeros_like(CSI_data)
        fit_chemical_shift_axis_ppm = np.zeros_like(chemical_shift_axis_ppm)

        linewidth_fit = plot_params.get("linewidth_fit", 0)
        plot_params["linewidth_fit"] = linewidth_fit
        alpha_fit = plot_params.get("alpha_fit", 0)
        plot_params["alpha_fit"] = alpha_fit
        linewidth_measurement = plot_params.get("linewidth_measurement", 0.5)
        plot_params["linewidth_measurement"] = linewidth_measurement
        alpha_measurement = plot_params.get("alpha_measurement", 1)
        plot_params["alpha_measurement"] = alpha_measurement
    if CSI_data.shape[1:3] != fit_spectra.shape[1:3]:
        raise ValueError(
            f"CSI_data and fit_spectra need to have same x-y dimensions but have"
            f"CSI_data: {CSI_data.shape[1:3]} and "
            f"fit_spectra: {fit_spectra.shape[1:3]}"
        )
    if anatomical_BrukerExp is None:
        raise ValueError("Please supply anatomical_BrukerExp")

    # check if data is complex, if yes apply ||
    if np.issubdtype(CSI_data.dtype, np.complexfloating):
        CSI_data = np.abs(CSI_data)

    if np.issubdtype(fit_spectra.dtype, np.complexfloating):
        fit_spectra = np.abs(fit_spectra)

    # decide which repetitions to plot:
    if rep is None:
        rep = range(0, 1)
    elif isinstance(rep, int):
        rep = range(rep, rep + 1)

    sample_ID = plot_params.get("sample_ID", "")
    figsize = plot_params.get("figsize", (8, 6.5))

    t_start_CSI_after_inj_s = plot_params.get("t_start_CSI_after_inj_s", 0)

    ylim = plot_params.get("ylim", (0, 1e7))
    if ylim == "max":
        ylim = (0, np.max(CSI_data))

    xlim = plot_params.get("xlim", (190, 165))

    linewidth_measurement = plot_params.get("linewidth_measurement", 0.5)
    linewidth_fit = plot_params.get("linewidth_fit", 0.25)

    linecolor_measurement = plot_params.get("linecolor_measurement", "y")
    linecolor_fit = plot_params.get("linecolor_fit", "r")

    alpha_measurement = plot_params.get("alpha_measurement", 1)
    alpha_fit = plot_params.get("alpha_fit", 1)

    alpha_CSI_amps = plot_params.get("alpha_lac_map", [0.5, 0.5])

    if marked_voxel is None:
        marked_voxel = plot_params.get("marked_voxel", None)
        if marked_voxel is None:
            marked_voxel = (0, 0)
            plot_params["marked_voxel"] = marked_voxel

        if not isinstance(marked_voxel, tuple):
            try:
                marked_voxel = tuple(marked_voxel)
            except:
                pass

    show_tqdm = plot_params.get("show_tqdm", True)

    vmin_anatomical_image = plot_params.get("vmin_anatomical_image", None)
    vmax_anatomical_image = plot_params.get("vmax_anatomical_image", None)

    CSI_extent = plot_params.get(
        "CSI_extent",
        [
            CSI_BrukerExp.rps_extent_mm[2],
            CSI_BrukerExp.rps_extent_mm[3],
            CSI_BrukerExp.rps_extent_mm[0],
            CSI_BrukerExp.rps_extent_mm[1],
        ],
    )
    anatomical_image_extent = plot_params.get("anatomical_image_extent", CSI_extent)

    thresholds = plot_params.get("thresholds", [5, 1])

    colormap_list = plot_params.get("colormap_list", ["jet", "jet", "jet"])
    interpolation_factor = plot_params.get("interpolation_factor", 2)
    scaling_factor = plot_params.get("scaling_factor", 1)
    vmin_percentiles = plot_params.get("vmin_percentiles", [0, 0])
    vmax_percentiles = plot_params.get("vmax_percentiles", [100, 100])
    metabolite_colors = plot_params.get(
        "metabolite_colors",
        ["dodgerblue", "magenta", "tab:green", "tab:red"],
    )

    save_fig = plot_params.get("save_fig", False)
    save_dpi = plot_params.get("save_dpi", 500)
    fig_transparent = plot_params.get("fig_transparent", True)
    save_path = plot_params.get("save_path", CSI_BrukerExp.path)
    fit_params_attrs = getattr(fit_params, "attrs", {})
    metabolite_labels = list(
        fit_params_attrs.get(
            "metabs",
            [f"metabolite {idx + 1}" for idx in range(max(fit_amps.shape[-1], 2))],
        )
    )

    if anatomical_image_numbers is None:
        anatomical_image_numbers = [0]

    x_axis_mm = np.linspace(
        CSI_extent[0] + 1 / 2 * (CSI_extent[1] - CSI_extent[0]) / CSI_data.shape[2],
        CSI_extent[1] - 1 / 2 * (CSI_extent[1] - CSI_extent[0]) / CSI_data.shape[2],
        CSI_data.shape[2],
    )
    y_axis_mm = np.linspace(
        CSI_extent[2] + 1 / 2 * (CSI_extent[3] - CSI_extent[2]) / CSI_data.shape[1],
        CSI_extent[3] - 1 / 2 * (CSI_extent[3] - CSI_extent[2]) / CSI_data.shape[1],
        CSI_data.shape[1],
    )

    x = marked_voxel[1]
    y = marked_voxel[0]

    if np.any(image_stack):
        image_stack = image_stack
    else:
        image_stack = anatomical_BrukerExp.scanner_data

    for r in tqdm(rep, disable=not show_tqdm):
        fig, ax = plt.subplots(figsize=figsize)
        ax.clear()

        for anatomical_image_number in anatomical_image_numbers:
            fig, ax = plt.subplots(
                3,
                3,
                figsize=figsize,
                gridspec_kw={"width_ratios": [1, 1, 0.1], "height_ratios": [1, 1, 1]},
            )

            for i in tqdm(range(3), disable=not show_tqdm):
                for j in tqdm(range(3), disable=not show_tqdm):
                    ax[i][j].clear()

                    if i == 0 and j == 0:
                        anatomical_image = np.squeeze(
                            np.abs(image_stack.isel(slice_idx=anatomical_image_number))
                        )

                        subplot = ax[i][j].imshow(
                            anatomical_image,
                            cmap="bone",
                            vmin=vmin_anatomical_image,
                            vmax=vmax_anatomical_image,
                            extent=anatomical_image_extent,
                            origin="lower",
                        )

                        ax[i][j].axvline(
                            x_axis_mm[x], c="magenta", linestyle="dashed", linewidth=1
                        )
                        ax[i][j].axhline(
                            y_axis_mm[y], c="magenta", linestyle="dashed", linewidth=1
                        )

                        ax[i][j].set_title(r"$^{1}$H $T_{2}$-w. image")
                        ax[i][j].axis("off")

                        ax[i][j].set_xlim(CSI_extent[0], CSI_extent[1])
                        ax[i][j].set_ylim(CSI_extent[2], CSI_extent[3])

                        divider = make_axes_locatable(ax[i][j])
                        cb_ax = divider.append_axes("bottom", size="5%", pad=0.05)
                        cb_ax.set_visible(False)

                        # iax     = inset_axes(cb_ax, width='75%', height='100%', loc='center', borderpad=0)
                        # cbar    = fig.colorbar(subplot, cax=iax, orientation='horizontal', ticks=[vmin, vmid, vmax], format='%.1e')

                        # image_start_time = '\n' + r'$t_{\mathrm{after~inj}}$ $\in$ = ' + '{0:.0f} s'.format(r * CSI_BrukerExp.frame_time_s[0] + t_start_CSI_after_inj_s)
                        # cbar.set_label(r'$^{13}$C FID amplitude' + '\n[a. u.]')
                        # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))

                        add_scalebar(
                            px=1,
                            ax=ax[i][j],
                            unit="mm",
                            fixed_value=5,
                            color="white",
                            box_alpha=0.0,
                            box_color="w",
                            location="lower right",
                            frameon=True,
                            pad=0.075,
                            length_fraction=None,
                            border_pad=0.075,
                            fontsize=8,
                        )

                        ###################################################################################################################################################
                        sp_ax = divider.append_axes("left", size="110%", pad=0.8)
                        sp_ax.set_visible(False)

                        spiax = inset_axes(
                            sp_ax,
                            width="100%",
                            height="100%",
                            loc="center",
                            borderpad=20,
                        )
                        spiax.set_visible(False)

                    elif j == 0:
                        image = img_interp(
                            np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, i - 1]))
                            * scaling_factor,
                            interp_factor=interpolation_factor,
                            interp_method="bilinear",
                            threshold=thresholds[i - 1],
                        )

                        ax[i][j].imshow(
                            anatomical_image,
                            cmap="bone",
                            extent=anatomical_image_extent,
                            vmin=vmin_anatomical_image,
                            vmax=vmax_anatomical_image,
                            origin="lower",
                        )
                        subplot2 = ax[i][j].imshow(
                            image,
                            cmap=colormap_list[i],
                            alpha=alpha_CSI_amps[i - 1],
                            extent=CSI_extent,
                            origin="lower",
                        )

                        ax[i][j].axvline(
                            x_axis_mm[x], c="magenta", linestyle="dashed", linewidth=1
                        )
                        ax[i][j].axhline(
                            y_axis_mm[y], c="magenta", linestyle="dashed", linewidth=1
                        )

                        ax[i][j].set_title(r"HP " + metabolite_labels[i - 1])
                        ax[i][j].axis("off")

                        ax[i][j].set_xlim(CSI_extent[0], CSI_extent[1])
                        ax[i][j].set_ylim(CSI_extent[2], CSI_extent[3])

                        vmin = np.nanpercentile(image, vmin_percentiles[i - 1])
                        vmax = np.nanpercentile(image, vmax_percentiles[i - 1])
                        vmid = (vmax + vmin) / 2

                        divider = make_axes_locatable(ax[i][j])
                        cb_ax = divider.append_axes("bottom", size="5%", pad=0.05)
                        cb_ax.set_visible(False)

                        iax = inset_axes(
                            cb_ax, width="75%", height="100%", loc="center", borderpad=0
                        )
                        cbar = fig.colorbar(
                            subplot2,
                            cax=iax,
                            orientation="horizontal",
                            ticks=[vmin, vmid, vmax],
                            format="%.1f",
                        )

                        cbar.set_label(r"$^{13}$C FID amplitude" + "\n[a. u.]")
                        # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))

                        add_scalebar(
                            px=1,
                            ax=ax[i][j],
                            unit="mm",
                            fixed_value=5,
                            color="white",
                            box_alpha=0.0,
                            box_color="w",
                            location="lower right",
                            frameon=True,
                            pad=0.075,
                            length_fraction=None,
                            border_pad=0.075,
                            fontsize=8,
                        )

                        ###################################################################################################################################################
                        sp_ax = divider.append_axes("left", size="110%", pad=0.8)
                        sp_ax.set_visible(False)

                        spiax = inset_axes(
                            sp_ax,
                            width="100%",
                            height="100%",
                            loc="center",
                            borderpad=20,
                        )
                        spiax.set_visible(False)

                    elif j == 1:
                        if i == 0:
                            image = img_interp(
                                np.squeeze(
                                    np.abs(np.mean(CSI_data[:, :, :, 0, r, :], axis=0))
                                )
                                * scaling_factor,
                                interp_factor=interpolation_factor,
                                interp_method="bilinear",
                                threshold=0,
                            )
                            ax[i][j].set_title(r"$^{13}$C density image")

                        else:
                            image = img_interp(
                                np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, i - 1]))
                                * scaling_factor,
                                interp_factor=interpolation_factor,
                                interp_method="bilinear",
                                threshold=0,
                            )

                            ax[i][j].set_title(r"HP " + metabolite_labels[i - 1])

                        ax[i][j].axvline(
                            x_axis_mm[x], c="magenta", linestyle="dashed", linewidth=1
                        )
                        ax[i][j].axhline(
                            y_axis_mm[y], c="magenta", linestyle="dashed", linewidth=1
                        )

                        subplot = ax[i][j].imshow(
                            image,
                            cmap=colormap_list[i],
                            extent=CSI_extent,
                            origin="lower",
                        )

                        ax[i][j].axis("off")

                        vmin = np.nanpercentile(image, vmin_percentiles[i - 1])
                        vmax = np.nanpercentile(image, vmax_percentiles[i - 1])
                        vmid = (vmax + vmin) / 2

                        divider = make_axes_locatable(ax[i][j])
                        cb_ax = divider.append_axes("bottom", size="5%", pad=0.05)
                        cb_ax.set_visible(False)

                        iax = inset_axes(
                            cb_ax, width="75%", height="100%", loc="center", borderpad=0
                        )
                        cbar = fig.colorbar(
                            subplot,
                            cax=iax,
                            orientation="horizontal",
                            ticks=[vmin, vmid, vmax],
                            format="%.1f",
                        )
                        # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))

                        if i == 0:
                            cbar.set_label(
                                r"$\langle$ $^{13}$C spectral signal $\rangle$"
                                + "\n[a. u.]"
                            )
                        else:
                            cbar.set_label(r"$^{13}$C FID amplitude" + "\n[a. u.]")

                        ###################################################################################################################################################
                        sp_ax = divider.append_axes("right", size="110%", pad=0.8)
                        sp_ax.set_visible(False)

                        spiax = inset_axes(
                            sp_ax,
                            width="100%",
                            height="100%",
                            loc="center",
                            borderpad=20,
                        )

                        if i == 0:
                            x1 = chemical_shift_axis_ppm
                            x2 = fit_chemical_shift_axis_ppm
                            xmin1 = xlim[0]
                            xmax1 = xlim[1]

                            y11 = (
                                np.squeeze(np.abs(CSI_data[:, y, x, 0, r, :]))
                                * scaling_factor
                            )
                            y12 = (
                                np.squeeze(np.abs(fit_spectra[:, y, x, 0, r, :]))
                                * scaling_factor
                            )
                            ymin1 = ylim[0]
                            ymax1 = ylim[1]

                            spiax.set_ylabel(r"$^{13}$C signal magnitude" + "\n[a. u.]")
                            spiax.set_xlabel(r"$^{13}$C chemical shift" + "\n[ppm]")

                            spiax.invert_xaxis()

                            spiax.plot(
                                x1,
                                y11,
                                color=linecolor_measurement,
                                lw=linewidth_measurement,
                                linestyle="-",
                                label=r"measurement",
                                alpha=alpha_measurement,
                            )
                            spiax.plot(
                                x2,
                                y12,
                                color=linecolor_fit,
                                lw=linewidth_fit,
                                linestyle="-",
                                label=r"fit",
                                alpha=alpha_fit,
                            )

                            spiax.set_xlim(xmin1, xmax1)
                            spiax.set_ylim(ymin1, ymax1)

                            spiax.tick_params(direction="in", right=True, top=True)

                            leg = spiax.legend(
                                framealpha=1,
                                title=r"$t_{\mathrm{after~injection}}$ = "
                                + "{0:.0f} s".format(
                                    r * CSI_BrukerExp.frame_time_s[0]
                                    + t_start_CSI_after_inj_s
                                ),
                                loc="upper right",
                                bbox_to_anchor=(0.95, 1.8),
                            )
                            leg.get_frame().set_edgecolor("black")

                        else:
                            x3 = (
                                np.arange(CSI_data.shape[-2])
                                * CSI_BrukerExp.frame_time_s[0]
                                + t_start_CSI_after_inj_s
                            )

                            y3 = np.abs(fit_amps[y, x, 0, :, 0, i - 1]) * scaling_factor

                            spiax.set_ylabel(r"$^{13}$C FID amplitude" + "\n[a. u.]")
                            spiax.set_xlabel(r"$t_{\mathrm{after~injection}}$ [s]")

                            spiax.plot(
                                x3,
                                y3,
                                color=metabolite_colors[i - 1],
                                lw=1.0,
                                linestyle="-",
                            )

                            spiax.axvline(
                                x3[r], color="k", linestyle="dashed", linewidth=1
                            )
                            # spiax.axvspan(x3[r], x3[r], alpha=0.2, color='gray', label='time span used\n for averaging')

                            # spiax.set_xlim(xmin3, xmax3)
                            # spiax.set_ylim(ymin3, ymax3)

                            spiax.tick_params(direction="in", right=True, top=True)

                            # leg = spiax.legend(framealpha=1, loc='upper right', bbox_to_anchor=(0.8, 1.4))
                            # leg.get_frame().set_edgecolor('black')
                        ###################################################################################################################################################

                    elif j == 2:
                        ax[i][j].set_visible(False)

                    # plt.tight_layout()

                    if save_fig:
                        save_name = plot_params.get(
                            "save_name",
                            "{0}_fit_results_rep_{1}_image_{2}".format(
                                sample_ID, r, anatomical_image_number
                            ),
                        )

                        plt.savefig(
                            os.path.join(save_path, save_name + ".png"),
                            transparent=fig_transparent,
                            dpi=save_dpi,
                        )
                        # plt.savefig(
                        #     os.path.join(save_path, save_name + '.svg'),
                        #     transparent=fig_transparent,
                        #     dpi=save_dpi,
                        # )
