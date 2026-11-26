"""Notebook views of legacy CSI phase evolution using shared signal calculations."""

import ipywidgets as widgets
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hypermri.analysis.spectroscopy.frequency import get_freq_axis
from hypermri.analysis.spectroscopy.processing import norm
from hypermri.analysis.spectroscopy.signals import (
    combine_isochromats,
    propagate_isochromats,
)


class CSImulator:
    """Explore phase evolution in legacy CSI objects.

    Parameters
    ----------
    csi_object : object
        CSI source supplying bandwidth_hz, n_acquisition, dwell_time_s and get for
        frequency metadata. Selected spectra use data in (frequency, row, column)
        order; full-volume simulation additionally requires seq2d and the legacy
        apply_to_every_spec iterator.
    voxel_selection : dict, optional
        Display label to (row, column) index mapping; retained by reference.

    Notes
    -----
    Construction derives frequency bins in Hz from the source without loading
    signal arrays or constructing widgets. Propagation uses the shared isochromat
    helper. Full-volume simulation rejects sources lacking the legacy iterator.
    Use SpeckSimulator for explicitly selected labelled spectra.
    """

    def __init__(self, csi_object, voxel_selection=None):
        """Retain the CSI source and derive its frequency bins in Hz.

        Parameters
        ----------
        csi_object : object
            Acquisition context accepted by get_freq_axis; no spectrum is read here.
        voxel_selection : dict, optional
            Display labels mapped to (row, column) indices, retained by reference.

        Returns
        -------
        None
            Initializes empty simulation results without constructing widgets.

        Raises
        ------
        AttributeError, ValueError
            If supplied source frequency metadata are invalid.
        """
        if voxel_selection is None:
            voxel_selection = dict()
        self.csi = csi_object
        self.fbins = get_freq_axis(scan=csi_object, unit="Hz", cut_off=0, npoints=None)

        self.voxel_selection = voxel_selection

        self.magnitudes = None
        self.phases = None
        self.timepoints = None

    @staticmethod
    def tuple2idx(tup):
        """Translate a row/column pair to a legacy spectral slice.

        Parameters
        ----------
        tup : tuple of int
            Row and column index in the legacy (frequency, row, column) array.

        Returns
        -------
        tuple
            All frequency bins followed by the supplied row and column indices.

        Raises
        ------
        IndexError
            If fewer than two indices are supplied.
        """
        return np.index_exp[:, tup[0], tup[1]]

    @property
    def labels(self):
        """Return display labels in voxel-selection insertion order.

        Returns
        -------
        list of str
            A fresh label list; stored selections and data are unchanged.
        """
        return list(self.voxel_selection.keys())

    @property
    def spectra(self):
        """Read spectra at the selected legacy CSI voxel indices.

        Returns
        -------
        list of numpy.ndarray
            Squeezed spectral component weights in source units and selection order.
            Data access is eager and may share input storage; no simulation is run.

        Raises
        ------
        AttributeError, IndexError
            If the legacy data representation or voxel indices are unavailable.
        """
        return [
            np.squeeze(self.csi.primary_data[self.tuple2idx(tup)])
            for tup in list(self.voxel_selection.values())
        ]

    def propagate(self, A_spec, t):
        """Evolve spectral weights using the shared isochromat calculation.

        Parameters
        ----------
        A_spec : array_like
            Component weights matching ``self.fbins`` in arbitrary signal units.
        t : float
            Evolution time in seconds.

        Returns
        -------
        vectors, phases_rad : numpy.ndarray
            Complex component signals and unwrapped phases in radians, with the
            broadcast input shape. Inputs are not mutated; evaluation is eager.

        Raises
        ------
        ValueError
            If weights and frequency shapes cannot broadcast.
        """
        return propagate_isochromats(A_spec, self.fbins, t)

    def spectrum2phase(self, A_spec, t, return_spin_vec_sum=False):
        """Return effective magnitude and principal phase after evolution.

        Parameters
        ----------
        A_spec : array_like
            Component weights matching ``self.fbins`` in arbitrary signal units.
        t : float
            Evolution time in seconds.
        return_spin_vec_sum : bool, optional
            Also return the complex component sum when True.

        Returns
        -------
        tuple
            Magnitude in signal units and phase in radians in ``[-pi, pi]``;
            optionally the complex sum. Deterministic, eager, without mutation.

        Raises
        ------
        ValueError
            If weights and frequency shapes cannot broadcast.
        """
        vectors, _ = self.propagate(A_spec, t)
        result = combine_isochromats(vectors)
        return result if return_spin_vec_sum else result[:2]

    def simulate(self, timepoints):
        """Simulate legacy CSI phase and magnitude images at requested times.

        Parameters
        ----------
        timepoints : sequence of float
            Evolution times in seconds, ordered as desired for the output frames.

        Returns
        -------
        None
            Stores squeezed phase (radians), magnitude (arbitrary signal units),
            complex_data and timepoints on this simulator. The acquired CSI data
            are not mutated.

        Raises
        ------
        NotImplementedError
            If the CSI object lacks the legacy apply_to_every_spec iterator.
        AttributeError, ValueError
            If legacy seq2d spatial dimensions or spectra are unavailable/inconsistent.

        Notes
        -----
        This eager simulation retains legacy spectral integration and dimension
        squeezing. Missing prerequisites fail before allocating simulation arrays.
        """
        if not hasattr(self.csi, "apply_to_every_spec"):
            raise NotImplementedError(
                "CSImulator.simulate requires the unavailable legacy CSI.apply_to_every_spec iterator; select labelled spectra explicitly for simulation."
            )
        spatial_dims = self.csi.seq2d.shape[1:]
        phases = np.zeros((len(timepoints), *spatial_dims))
        magnitudes = np.zeros((len(timepoints), *spatial_dims))
        complex_data = np.zeros((len(timepoints), *spatial_dims), dtype="complex")

        for i, t in enumerate(timepoints):

            def func(spec, t=t):
                return self.spectrum2phase(spec, t, True)

            for idx, ph in self.csi.apply_to_every_spec(func):
                magnitudes[(i, *idx)] = ph[0]
                phases[(i, *idx)] = ph[1]
                complex_data[(i, *idx)] = ph[2]

        self.phases = np.squeeze(phases)
        self.magnitudes = np.squeeze(magnitudes)
        self.complex_data = complex_data
        self.timepoints = np.array(timepoints)

    def clocks(self, t_min=0, t_max=3e-3, t_step=0.5e-6):
        """Show interactive polar vectors for the selected spectra.

        Parameters
        ----------
        t_min, t_max : float, optional
            Beginning and exclusive end of the evolution interval in seconds.
        t_step : float, optional
            Positive time increment in seconds.

        Returns
        -------
        None
            Displays a Matplotlib figure and notebook time controls. Signal evolution
            uses the shared deterministic isochromat helper; measured data are preserved.

        Raises
        ------
        AssertionError
            If no spectra are selected.
        ValueError, IndexError
            If the time interval is empty or spectra and frequency bins are inconsistent.
        """
        N = len(self.voxel_selection)

        assert N != 0, "No voxel selection provided!"

        fig, axes = plt.subplots(
            ncols=N, figsize=(9.5, 5.5), subplot_kw={"projection": "polar"}
        )

        r_total_t0 = [self.spectrum2phase(spec, t=0)[0] for spec in self.spectra]

        t_range = np.arange(t_min, t_max, t_step)

        self.__layout_adjusted = False  # ugly patch to fix fig.layout + widgets combo

        def plot_func(time_idx, show_total_vec):
            time = t_range[time_idx]
            for i in range(N):
                ax = axes[i]
                spec = self.spectra[i]
                label = self.labels[i]

                zs, true_phases = self.propagate(spec, time)

                rs = norm(np.abs(zs))
                phases = np.angle(zs)

                r_total = np.abs(np.sum(zs)) / r_total_t0[i] * 1.1
                phase_total = np.angle(np.sum(zs))

                ax.clear()
                ax.plot(phases, rs, color="blue")

                if show_total_vec:
                    ax.plot(
                        [r_total, phase_total],
                        [0, r_total],
                        linewidth=2.5,
                        color="orange",
                    )

                # some style improvements
                ax.set_rmax(1.5)
                ax.set_rticks([0.5, 1])  # Less radial ticks
                ax.set
                # Move radial labels away from plotted line
                ax.set_rlabel_position(90)
                ax.grid(True)

                title = r"$\bf{" + str(label) + "}$"
                title += "\n$\\phi_{tot} = $" + f"{phase_total:.3f} rad"
                ax.set_title(title)

            fig.suptitle(f"{time * 1e3:.2f} ms", fontweight=400, fontsize=20, y=0.99)

            if not self.__layout_adjusted:
                fig.tight_layout()
                self.__layout_adjusted = True

        time_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(t_range) - 1,
            step=1,
            readout=False,
            layout=widgets.Layout(width="500px"),
        )

        time_play = widgets.Play(
            value=0,
            min=0,
            max=len(t_range) - 1,
            step=1,
            interval=5,
            description="Press play",
            style={"description_width": "initial"},
        )

        play_speed_int = widgets.BoundedIntText(
            value=10,
            min=0,
            max=100,
            step=10,
            description="Speed:",
            layout=widgets.Layout(width="140px"),
        )

        widgets.jslink((time_play, "value"), (time_slider, "value"))
        widgets.jslink((time_play, "interval"), (play_speed_int, "value"))

        total_vec_button = widgets.ToggleButton(
            value=True,
            description="Total Vector",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        )
        # put both sliders inside a HBox for nice alignment  etc.
        row1 = widgets.HBox([time_play, time_slider, play_speed_int, total_vec_button])

        ui = widgets.VBox(
            [row1],
            layout=widgets.Layout(display="flex"),
        )

        sliders = widgets.interactive_output(
            plot_func,
            {
                "time_idx": time_slider,
                "show_total_vec": total_vec_button,
            },
        )

        display(ui, sliders)

    def plot(self, mag_cmap="gray", pha_cmap="bwr"):
        """Display stored simulated magnitude and phase frames.

        Parameters
        ----------
        mag_cmap, pha_cmap : str, optional
            Matplotlib colormaps for magnitude (signal units) and phase (radians).

        Returns
        -------
        None
            Displays a figure and notebook frame selector; stored samples are unchanged.

        Raises
        ------
        AssertionError
            If simulate has not been called.
        ValueError, IndexError
            If squeezed simulation arrays cannot be displayed as image frames.
        """
        assert self.timepoints is not None, "You have to call self.simulate() first."

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9.5, 5))

        # sort out arrangement of colorbar and plot
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)

        self.__layout_adjusted = False  # ugly patch for fig.tight_layout & widgets

        def plot_func(echo_nr):
            for ax, cax, cmap, data, title, label, ticks in zip(
                [ax1, ax2],
                [cax1, cax2],
                [mag_cmap, pha_cmap],
                [self.magnitudes, self.phases],
                ["magnitude", "phase"],
                ["a.u.", "rad"],
                [[], None],
                strict=False,
            ):
                ax.clear()
                ax.imshow(data[echo_nr], cmap=cmap)
                ax.set_title(title)
                # normalize internal data form 0 to 1
                norm = plt.Normalize(np.nanmin(data), np.nanmax(data))
                # plot the colorbar
                fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cax,
                    orientation="vertical",
                    label=label,
                    ticks=ticks,
                )

            fig.suptitle(
                f"{self.timepoints[echo_nr] * 1e3:.2f} ms", fontsize=15, fontweight=500
            )

            if not self.__layout_adjusted:
                fig.tight_layout()
                self.__layout_adjusted = True

        echo_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.timepoints) - 1,
            step=1,
            description="Echo Nr.:",
            style={"description_width": "initial"},
        )

        # put both sliders inside a HBox for nice alignment  etc.
        row1 = widgets.HBox([echo_slider])
        ui = widgets.VBox(
            [row1],
            layout=widgets.Layout(display="flex"),
        )
        sliders = widgets.interactive_output(
            plot_func,
            {
                "echo_nr": echo_slider,
            },
        )
        display(ui, sliders)

    def plot_selection_spectra(self, xmin=-1000, xmax=1000):
        """Plot the selected spectra against their frequency bins.

        Parameters
        ----------
        xmin, xmax : float, optional
            Lower and upper frequency display limits in Hz.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Displayed figure and axes. Spectra and their arbitrary signal units are
            preserved; no simulation is triggered.

        Raises
        ------
        ValueError
            If spectrum and frequency-axis lengths disagree.
        """
        fig, ax = plt.subplots(figsize=(9, 4))

        N = len(self.voxel_selection)

        for i in range(N):
            ax.plot(self.fbins, self.spectra[i], label=self.labels[i])

        ax.set_xlim(xmin, xmax)
        ax.legend()
        ax.set_xlabel("Frequency [Hz]")
        ax.set_title("Spectra of Voxel Selection")

        fig.tight_layout()
        plt.show()

    def plot_selection_phase(self):
        """Plot stored phase evolution for the selected spectra.

        Returns
        -------
        None
            Displays phase in radians against evolution time in milliseconds. Stored
            simulation arrays are not modified.

        Raises
        ------
        AssertionError
            If simulate has not been called.
        ValueError, IndexError
            If stored phase dimensions do not match the selected spectra.
        """
        assert self.timepoints is not None, "You have to call self.simulate() first."

        fig, ax = plt.subplots()

        N = len(self.voxel_selection)

        for i in range(N):
            idx_tuple = list(self.voxel_selection.values())[i]
            ax.plot(
                self.timepoints * 1e3,
                self.phases[self.tuple2idx(idx_tuple)],
                marker="s",
                label=self.labels[i],
            )

        ax.set_xlabel("t [ms]")
        ax.set_ylabel("phase [rad]")
        ax.set_title("Phase Evolution of Voxel Selection")
        ax.legend()
        fig.tight_layout()
