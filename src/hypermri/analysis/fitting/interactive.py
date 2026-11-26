"""Interactive setup and visualization tools for fitting."""

from __future__ import annotations

import time

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import clear_output, display

from hypermri.analysis.spectroscopy.frequency import freq_hz_to_ppm, freq_ppm_to_hz
from hypermri.analysis.spectroscopy.schema import ensure_standard_spectroscopy_dataarray

from .models.spectroscopy.fid_model import def_fit_params


def fit_params_metabs_setup_tool(data=None, fit_params=None, plot_params=None):
    """
    Interactive viewer for 6D MRI spectroscopy data + helper to set up the
    parameters metabs_freqs_hz, metabs_freqs_ppm, metabs_freqs_hz by clicking and
    metabs_freq_ranges by dragging, using the right mouse button.
    """
    from ipywidgets import interact
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RectangleSelector

    if plot_params is None:
        plot_params = {}
    if fit_params is None:
        fit_params = {}
    try:
        from matplotlib import colormaps

        get_cmap = colormaps.get_cmap  # Matplotlib 3.10+
    except ImportError:
        from matplotlib.cm import get_cmap  # Matplotlib <3.10

    provided_dims = fit_params.get("provided_dims", None)
    provided_dims_ind = fit_params.get("provided_dims_ind", None)
    if (
        not isinstance(data, xr.DataArray)
        and provided_dims is None
        and provided_dims_ind is None
    ):
        provided_dims = np.array((np.ndim(data), 1))
        provided_dims[0] = 0  # spec
        if np.ndim(data) > 1:
            provided_dims[1] = 4  # "reps"
        if np.ndim(data) > 2:
            provided_dims[2] = 1  # "z"
        if np.ndim(data) > 3:
            provided_dims[3] = 2  # "x"
        if np.ndim(data) > 4:
            provided_dims[4] = 3  # "y"
        if np.ndim(data) > 5:
            provided_dims[5] = 5  # "chans"

    resolved_dims = provided_dims if provided_dims is not None else provided_dims_ind
    normalized_input = ensure_standard_spectroscopy_dataarray(
        data=data,
        provided_dims=resolved_dims,
    )
    fit_params["provided_dims"] = [0, 1, 2, 3, 4, 5]
    data = (
        normalized_input if isinstance(data, xr.DataArray) else normalized_input.values
    )
    signal_domain = fit_params.get("signal_domain", "spectral")
    fit_params["signal_domain"] = signal_domain
    if signal_domain == "spectral":
        pass
    else:
        data = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    data = np.rot90(
        data,
        k=1,
        axes=(2, 3),
    )

    current_z, current_rep, current_channel = (
        data.shape[1] // 2,
        data.shape[4] // 2,
        0,
    )
    current_x, current_y = data.shape[2] // 2, data.shape[3] // 2
    freq_axis_mode = "Hz"
    data_type = "Magnitude"
    figsize = plot_params.get("figsize", (15, 10))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    plt.subplots_adjust(bottom=0.25)

    # Prepare frequency axis
    freq_hz = fit_params["freq_axis_hz"]
    freq_ppm = fit_params["freq_axis_ppm"]
    metabs = fit_params.get("metabs", ["peak1"])
    metabs_freqs_hz = fit_params.get("metabs_freqs_hz", [0.0] * len(metabs))
    fit_params["metabs_freqs_hz"] = metabs_freqs_hz
    metabs_freq_ranges = fit_params.get("metabs_freq_ranges", [30.0] * len(metabs))
    fit_params["metabs_freq_ranges"] = metabs_freq_ranges
    fit_params["metabs"] = metabs
    # Generate a color map for metabolite lines
    color_map = get_cmap("tab10")
    metab_colors = [color_map(i) for i in np.linspace(0, 1, len(fit_params["metabs"]))]

    # Add vertical lines for metabolites with different colors and labels
    metab_lines = []
    metab_labels = []
    metab_ranges = []

    for i, (metab, freq, color, range_hz) in enumerate(
        zip(
            fit_params["metabs"],
            fit_params["metabs_freqs_hz"],
            metab_colors,
            fit_params["metabs_freq_ranges"],
            strict=False,
        )
    ):
        if freq is None:
            fit_params["metabs_freqs_hz"][i] = 0.0

        line = ax2.axvline(
            freq, color="gray", linestyle="--", linewidth=0.5, visible=False
        )
        label = ax2.text(
            freq,
            0,
            metab,
            color="gray",
            rotation=90,
            va="bottom",
            ha="right",
            visible=False,
        )
        range_rect = ax2.add_patch(
            Rectangle(
                (freq - range_hz / 2, 0),
                range_hz,
                1,
                facecolor=color,
                alpha=0.2,
                visible=False,
            )
        )
        metab_lines.append(line)
        metab_labels.append(label)
        metab_ranges.append(range_rect)

    def get_display_data(d):
        if data_type == "Real":
            return np.real(d)
        elif data_type == "Imaginary":
            return np.imag(d)
        elif data_type == "Phase":
            return np.angle(d)
        else:  # Magnitude
            return np.abs(d)

    def update_plots(z, rep, channel):
        nonlocal current_z, current_rep, current_channel
        current_z, current_rep, current_channel = z, rep, channel

        display_data = get_display_data(data[:, z, :, :, rep, channel])

        # Update image
        im.set_data(np.mean(display_data, axis=0))
        vmin, vmax = np.percentile(np.mean(display_data, axis=0), [2, 98])
        im.set_clim(vmin, vmax)

        # Update spectrum
        spectrum_data = display_data[:, current_x, current_y]
        spectrum_line.set_ydata(spectrum_data)

        # Update y-axis limits for spectrum
        if data_type == "Phase":
            ax2.set_ylim(-np.pi, np.pi)
        else:
            ax2.set_ylim(np.min(spectrum_data), 1.1 * np.max(spectrum_data))

        update_metab_lines()
        update_freq_table()

    def on_click_xy(event):
        nonlocal current_x, current_y
        if event.inaxes == ax1:
            current_x, current_y = int(event.ydata), int(event.xdata)
            spectrum_data = get_display_data(
                data[:, current_z, current_x, current_y, current_rep, current_channel]
            )
            spectrum_line.set_ydata(spectrum_data)
            ax2.set_title(f"Spectrum at X:{current_y}, Y:{current_x}")
            point_highlight.set_data([current_y], [current_x])

            if data_type == "Phase":
                ax2.set_ylim(-np.pi, np.pi)
            else:
                ax2.set_ylim(np.min(spectrum_data), 1.1 * np.max(spectrum_data))

            fig.canvas.draw_idle()

    def on_click_spectrum(event):
        if event.inaxes == ax2 and event.button == 1:  # Left click
            freq = event.xdata
            metab_str = metab_selector.value
            metab_idx = fit_params["metabs"].index(metab_str)
            if freq_axis_mode == "Hz":
                fit_params["metabs_freqs_hz"][metab_idx] = freq
                fit_params["metabs_freqs_ppm"][metab_idx] = freq_hz_to_ppm(
                    freq_hz=[freq], hz_axis=freq_hz, ppm_axis=freq_ppm
                )[0]
            else:
                fit_params["metabs_freqs_ppm"][metab_idx] = freq
                fit_params["metabs_freqs_hz"][metab_idx] = freq_ppm_to_hz(
                    freq_ppm=[freq], ppm_axis=freq_ppm, hz_axis=freq_hz
                )[0]

            update_metab_lines()
            update_freq_table()

    def on_select_range(eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        metab_str = metab_selector.value
        metab_idx = fit_params["metabs"].index(metab_str)
        new_range = abs(x2 - x1)
        if freq_axis_mode == "ppm":
            new_range = (
                new_range
                * (fit_params["freq_axis_hz"][-1] - fit_params["freq_axis_hz"][0])
                / (fit_params["freq_axis_ppm"][-1] - fit_params["freq_axis_ppm"][0])
            )
        fit_params["metabs_freq_ranges"][metab_idx] = new_range
        update_metab_lines()
        update_freq_table()
        rs.set_active(False)
        rs.set_visible(False)
        fig.canvas.draw_idle()
        rs.set_active(True)

    def update_metab_lines():
        y_positions = np.linspace(
            0.1, np.max(spectrum_line.get_data()[1]), len(metab_lines)
        )

        sorted_indices = sorted(
            range(len(metab_lines)),
            key=lambda k: (
                fit_params["metabs_freqs_hz"][k]
                if freq_axis_mode == "Hz"
                else fit_params["metabs_freqs_ppm"][k]
            ),
            reverse=(freq_axis_mode == "ppm"),
        )

        selected_metab = metab_selector.value
        for i, idx in enumerate(sorted_indices):
            freq_hz = fit_params["metabs_freqs_hz"][idx]
            freq_ppm = fit_params["metabs_freqs_ppm"][idx]
            range_hz = fit_params["metabs_freq_ranges"][idx]

            if freq_axis_mode == "Hz":
                freq = freq_hz
                range_start = freq - range_hz / 2
                range_end = freq + range_hz / 2
            else:
                freq = freq_ppm
                range_ppm = (
                    range_hz
                    * (fit_params["freq_axis_ppm"][-1] - fit_params["freq_axis_ppm"][0])
                    / (fit_params["freq_axis_hz"][-1] - fit_params["freq_axis_hz"][0])
                )
                range_start = freq - range_ppm / 2
                range_end = freq + range_ppm / 2

            metab_lines[idx].set_xdata([freq])
            metab_lines[idx].set_visible(True)

            if fit_params["metabs"][idx] == selected_metab:
                metab_lines[idx].set_color(metab_colors[idx])
                metab_labels[idx].set_color(metab_colors[idx])
            else:
                metab_lines[idx].set_color("gray")
                metab_labels[idx].set_color("gray")

            label = metab_labels[idx]
            label.set_position((freq, y_positions[i]))
            label.set_visible(True)

            metab_ranges[idx].set_xy((range_start, 0))
            metab_ranges[idx].set_width(range_end - range_start)
            metab_ranges[idx].set_height(ax2.get_ylim()[1])
            metab_ranges[idx].set_visible(True)

        fig.canvas.draw_idle()

    def update_freq_table():
        ax3.clear()
        ax3.axis("off")
        table_data = []
        for i, metab in enumerate(fit_params["metabs"]):
            freq_hz = fit_params["metabs_freqs_hz"][i]
            freq_ppm = fit_params["metabs_freqs_ppm"][i]
            range_hz = fit_params["metabs_freq_ranges"][i]
            table_data.append(
                [metab, f"{freq_hz:.2f}", f"{freq_ppm:.2f}", f"±{range_hz / 2:.2f}"]
            )

        table = ax3.table(
            cellText=table_data,
            colLabels=["Metabolite", "Freq (Hz)", "Freq (ppm)", "Range (±Hz)"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        selected_metab = metab_selector.value
        selected_idx = fit_params["metabs"].index(selected_metab)
        for i in range(len(table_data[0])):
            table[(selected_idx + 1, i)].set_facecolor(metab_colors[selected_idx])
            table[(selected_idx + 1, i)].set_text_props(weight="bold")

        ax3.set_title("Metabolite Frequencies and Ranges")
        fig.canvas.draw_idle()

    def switch_freq_axis(change):
        nonlocal freq_axis_mode
        freq_axis_mode = change.new
        if freq_axis_mode == "Hz":
            spectrum_line.set_xdata(freq_hz)
            ax2.set_xlim(freq_hz[0], freq_hz[-1])
            ax2.set_xlabel("Frequency (Hz)")
        else:
            spectrum_line.set_xdata(freq_ppm)
            ax2.set_xlim(freq_ppm[-1], freq_ppm[0])
            ax2.set_xlabel("Frequency (ppm)")
        update_metab_lines()
        update_freq_table()

    def switch_data_type(change):
        nonlocal data_type
        data_type = change.new
        update_plots(current_z, current_rep, current_channel)

    def on_metab_select(change):
        update_metab_lines()
        update_freq_table()

    display_data = get_display_data(
        data[:, current_z, :, :, current_rep, current_channel]
    )
    im = ax1.imshow(np.mean(display_data, axis=0), cmap="viridis")
    ax1.set_title(f"data.shape={data.shape}")
    (point_highlight,) = ax1.plot(
        [current_x], [current_y], "ro", markersize=10, fillstyle="none"
    )

    spectrum_data = display_data[:, current_x, current_y]
    (spectrum_line,) = ax2.plot(freq_hz, spectrum_data)
    ax2.set_title("Spectrum")
    ax2.set_xlabel("Frequency (Hz)")

    vmin, vmax = np.percentile(display_data, [2, 98])
    im.set_clim(vmin, vmax)
    ymin, ymax = np.percentile(spectrum_data, [1, 99])
    y_range = ymax - ymin
    ax2.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    fig.canvas.mpl_connect("button_press_event", on_click_xy)
    fig.canvas.mpl_connect("button_press_event", on_click_spectrum)

    rs = RectangleSelector(
        ax2,
        on_select_range,
        useblit=True,
        button=[3],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )

    ax2.set_navigate(True)
    ax2.set_navigate_mode(None)

    z_slider = widgets.IntSlider(
        min=0, max=data.shape[1] - 1, step=1, value=current_z, description="Z:"
    )
    rep_slider = widgets.IntSlider(
        min=0,
        max=data.shape[4] - 1,
        step=1,
        value=current_rep,
        description="Repetition:",
    )
    channel_slider = widgets.IntSlider(
        min=0,
        max=data.shape[5] - 1,
        step=1,
        value=current_channel,
        description="Channel:",
    )
    freq_axis_toggle = widgets.ToggleButtons(
        options=["Hz", "ppm"], value="Hz", description="Frequency Axis:"
    )
    metab_selector = widgets.Dropdown(
        options=fit_params["metabs"],
        value=fit_params["metabs"][0],
        description="Metabolite:",
    )
    data_type_selector = widgets.Dropdown(
        options=["Magnitude", "Real", "Imaginary", "Phase"],
        value="Magnitude",
        description="Data Type:",
    )

    freq_axis_toggle.observe(switch_freq_axis, names="value")
    data_type_selector.observe(switch_data_type, names="value")
    metab_selector.observe(on_metab_select, names="value")

    interact(update_plots, z=z_slider, rep=rep_slider, channel=channel_slider)

    display(widgets.HBox([freq_axis_toggle, metab_selector, data_type_selector]))

    update_metab_lines()
    update_freq_table()

    plt.gcf().rs = rs
    plt.tight_layout()


def interactive_setup_fit_params(data=None, fit_params=None):
    """Interactively set up fit parameters for data analysis."""
    if fit_params is None:
        fit_params = {}

    # Delegate to the actual implementation in def_fit_params
    fit_params = def_fit_params(data=data, fit_params=fit_params)

    def assign_metabolites():
        nonlocal fit_params
        frequency_unit = widgets.RadioButtons(
            options=["PPM", "Hz"], description="Frequency Unit:", disabled=False
        )

        metabolite_box = widgets.VBox()
        custom_metab_box = widgets.VBox()
        custom_metabs_display = widgets.Output()
        confirm_button = widgets.Button(description="Confirm Metabolites")

        custom_metabolites = {}

        def update_metabolites(change):
            nonlocal metabolite_box
            if change["new"] == "Hz":
                metabolites_cs = {"pyruvate": 0, "OMPD1": 1}
            else:
                metabolites_cs = {
                    "pyruvate": 171.076,
                    "lactate": 183.35,
                    "alanine": 176.5,
                }

            metabolites_cs.update(custom_metabolites)
            metabolite_checkboxes = [
                widgets.Checkbox(
                    description=f"{name}({value: .3f} {change['new']})",
                    value=name in custom_metabolites,
                )
                for name, value in metabolites_cs.items()
            ]
            metabolite_box.children = metabolite_checkboxes

        frequency_unit.observe(update_metabolites, names="value")

        new_metab_name = widgets.Text(description="Name:")
        new_metab_value = widgets.FloatText(description="Value:", value=-9999)
        add_button = widgets.Button(description="Add Metabolite")

        def add_custom_metabolite(b):
            name = new_metab_name.value
            value = new_metab_value.value
            if name and value != -9999:
                custom_metabolites[name] = value
                update_metabolites({"new": frequency_unit.value})
                new_metab_name.value = ""
                new_metab_value.value = -9999

        add_button.on_click(add_custom_metabolite)
        custom_metab_box.children = [new_metab_name, new_metab_value, add_button]

        def on_confirm_button_clicked(b):
            nonlocal fit_params
            selected_metabolites = {}
            for checkbox in metabolite_box.children:
                if checkbox.value:
                    name, value_str = checkbox.description.split(" (")
                    value = float(value_str.split()[0])
                    selected_metabolites[name] = value

            fit_params["metabs"] = list(selected_metabolites.keys())
            if frequency_unit.value == "PPM":
                fit_params["metabs_freqs_ppm"] = list(selected_metabolites.values())
                fit_params["metabs_freqs_hz"] = None
            else:
                fit_params["metabs_freqs_hz"] = list(selected_metabolites.values())
                fit_params["metabs_freqs_ppm"] = None

            new_params = def_fit_params(data=data, fit_params=fit_params)
            fit_params.update(
                new_params.attrs if isinstance(new_params, xr.Dataset) else {}
            )

            confirm_button.disabled = True
            fit_params_metabs_setup_tool(data=data, fit_params=fit_params)

        confirm_button.on_click(on_confirm_button_clicked)
        display(
            frequency_unit,
            widgets.HBox([metabolite_box, custom_metab_box]),
            custom_metabs_display,
            confirm_button,
        )
        update_metabolites({"new": "PPM"})

    assign_metabolites()
    return fit_params


def interactive_fit_params_setup(
    data: xr.DataArray,
    initial_params: dict = None,
) -> xr.Dataset:
    """Launch an interactive wizard to create a fit_params Dataset."""
    if initial_params is None:
        initial_params = {}
    output = widgets.Output()

    final_params_container = {}

    def on_confirm_clicked(b):
        selected_metabs = [cb.description for cb in metab_checkboxes if cb.value]
        if not selected_metabs:
            with output:
                print("Error: Please select at least one metabolite.")
            return

        params_for_def = initial_params.copy()
        params_for_def["metabs"] = selected_metabs
        fit_params = def_fit_params(data, params_for_def)

        with output:
            clear_output(wait=True)
            fit_params_metabs_setup_tool(data, fit_params)
            final_params_container["fit_params"] = fit_params

    known_metabolites = ["pyruvate", "lactate", "alanine", "bicarbonate", "urea"]
    metab_checkboxes = [
        widgets.Checkbox(description=name, value=(name in ["pyruvate", "lactate"]))
        for name in known_metabolites
    ]
    confirm_button = widgets.Button(
        description="Confirm Metabolites", button_style="success"
    )
    confirm_button.on_click(on_confirm_clicked)

    display(
        widgets.VBox(
            [widgets.Label("Select metabolites:"), *metab_checkboxes, confirm_button]
        ),
        output,
    )

    while "fit_params" not in final_params_container:
        time.sleep(0.5)

    return final_params_container["fit_params"]
