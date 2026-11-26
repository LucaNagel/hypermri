"""CPMG sequence implementation with T2 relaxation helpers."""

import traceback

import numpy as np
from scipy.optimize import curve_fit

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class CPMG(BaseSpectroscopy):
    """Carr-Purcell-Meiboom-Gill spectroscopy with relaxation analysis utilities."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialize CPMG sequence with proper BaseSpectroscopy inheritance."""
        super().__init__(handler, **kwargs)

    def calculate_T2_thermal(self):
        """
        Fit exponential to CPMG measurement to retrieve T2 value for a thermal T2 measurement with many repeitions.

        Returns
        -------
        T2 : float
            T2 decay constant from exponential fit
        T2_err : float
            Error of T2 decay constant from exponential fit
        """
        import matplotlib.pyplot as plt

        echotrain = []
        integral = []
        n_echos = self.n_echoes
        # apparently it always acquires 128 points
        # if the spec matrix is smaller than 128
        n_acquisition = self.n_acquisition
        if n_acquisition < 128:
            n_acquisition = 128
        else:
            pass
        echo_spacing = self.handler.method_parameter("EchoSpacing")
        n_repetitions = self.n_repetitions
        # average over repetitions
        fid_file = self.handler.fid
        if n_repetitions > 1:
            avg_fid = []
            for n in range(0, n_repetitions, 1):
                avg_fid.append(
                    fid_file[
                        n * n_acquisition * n_echos : n * n_acquisition * n_echos
                        + n_acquisition * n_echos
                    ]
                )
            avg_fid = np.array(avg_fid)
            # mean the fid
            fid_file = np.mean(avg_fid, axis=0)
        else:
            pass

        # extract data
        for n in range(0, n_echos, 1):
            echotrain.append(
                fid_file[n * n_acquisition : n * n_acquisition + n_acquisition]
            )
            integral.append(
                np.abs(
                    np.sum(
                        fid_file[n * n_acquisition : n * n_acquisition + n_acquisition]
                    )
                )
            )
        # form x axis for fit
        x_ax = np.arange(0, echo_spacing * n_echos, echo_spacing)
        # interpolate
        x_ax_itp = np.linspace(0, echo_spacing * n_echos, 10000)

        def exp(x, a, b, c):
            return a * np.exp(-x / b) + c

        # norm for easier fitting
        integral = integral / np.max(integral)
        coeff, err = curve_fit(exp, x_ax, integral, p0=(1, 20, 0.2))
        error = np.sqrt(np.diag(err))

        fig, ax = plt.subplots(1)
        ax.set_title(
            "CPMG - Echo spacing "
            + str(echo_spacing)
            + " , Echoes "
            + str(n_echos)
            + " , Repetitions "
            + str(n_repetitions)
        )
        ax.scatter(x_ax, integral, label="Data")
        ax.plot(
            x_ax_itp,
            exp(x_ax_itp, coeff[0], coeff[1], coeff[2]),
            label="Fit - T2 =" + str(np.round(coeff[1], 1)) + " ms",
        )
        ax.legend()
        ax.set_xlabel("[ms]")
        ax.set_ylabel("I [a.u.]")
        print(
            "T2=", np.round(coeff[1], 1), " plus minus ", np.round(error[1], 1), " ms"
        )
        T2 = coeff[1]
        T2_err = error[1]
        return T2, T2_err

    def calculate_T2_hyper(
        self,
        savepath: str,
        sample_name: str = "sample",
        starting_index: int = 0,
        initial_t2_ms: float = 5000.0,
        ax_to_plot_to: bool = False,
    ):
        """
        Loads CPMG measurement and fits exponential to it to retrieve T2 value.

        Parameters
        ----------
        savepath : str
            path to which figure will be saved
        Sample_ID : int or str
            Name of sample
        starting_index: int
            Where the fitting starts, maybe you want to leave out a few first data points
        guessedT2: int/float
            Estimate for the T2 in ms.

        Returns
        -------
        T2 : float
            T2 decay constant from exponential fit
        T2_err : float
            Error of T2 decay constant from exponential fit
        """
        import matplotlib.pyplot as plt

        echotrain = []
        integral = []
        n_echos = self.handler.method_parameter("NEchoes")
        # apparently it always acquires 128 points
        # if the spec matrix is smaller than 128
        n_acquisition = self.n_acquisition
        if n_acquisition < 128:
            n_acquisition = 128
        else:
            pass
        echo_spacing = self.handler.method_parameter("EchoSpacing")
        n_repetitions = self.n_repetitions
        # average over repetitions
        fid_file = self.handler.fid
        if n_repetitions > 1:
            # only use first rep
            fid_file = fid_file[
                n_acquisition * n_echos : n_acquisition * n_echos
                + n_acquisition * n_echos
            ]
        else:
            pass

        # extract data
        for n in range(0, n_echos, 1):
            echotrain.append(
                fid_file[n * n_acquisition : n * n_acquisition + n_acquisition]
            )
            integral.append(
                np.abs(
                    np.sum(
                        fid_file[n * n_acquisition : n * n_acquisition + n_acquisition]
                    )
                )
            )
        # form x axis for fit
        x_ax = np.arange(0, echo_spacing * n_echos, echo_spacing)
        # interpolate
        np.linspace(0, echo_spacing * n_echos, 10000)

        def exp(x, a, b, c):
            return a * np.exp(-x / b) + c

        # norm for easier fitting
        integral = integral / np.max(integral)
        # option to leave out first few points for the fit
        integral = integral[starting_index:]
        x_ax = x_ax[starting_index:]

        coeff, err = curve_fit(exp, x_ax, integral, p0=(1, initial_t2_ms, 0.2))
        error = np.sqrt(np.diag(err))
        if ax_to_plot_to:
            ax = ax_to_plot_to
            ax.set_title(
                "Echo spacing " + str(echo_spacing) + " , Echoes " + str(n_echos)
            )
        else:
            fig, ax = plt.subplots(1)
            ax.set_title(
                "CPMG - Echo spacing " + str(echo_spacing) + " , Echoes " + str(n_echos)
            )
        ax.scatter(x_ax / 1000, integral, label="Data", color="r")
        ax.plot(
            x_ax / 1000,
            exp(x_ax / 1000, coeff[0], coeff[1] / 1000, coeff[2]),
            label="Fit - T2 ="
            + str(np.round(coeff[1] / 1000, 1))
            + r"$\pm$"
            + str(np.round(error[1] / 1000, 1))
            + " s",
        )
        ax.legend()
        ax.set_xlabel(" Time since start of experiment [s]")
        ax.set_ylabel("I [a.u.]")
        if savepath:
            plt.savefig(
                savepath + "Sample_" + str(sample_name) + "_T2_measurement_7T.png"
            )
        print(
            "7T measurement -- T2= ",
            np.round(coeff[1] / 1000, 1),
            " plus minus ",
            np.round(error[1] / 1000, 1),
            " s",
        )
        T2 = coeff[1]
        T2_err = error[1]

        return T2, T2_err
