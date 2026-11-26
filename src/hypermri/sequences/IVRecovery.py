"""Inversion recovery spectroscopy sequence helper."""

import numpy as np
from scipy.optimize import curve_fit

from hypermri._optional import _notebook_alias as __getattr__
from hypermri.analysis.spectroscopy.inversion_recovery import (
    _fit_inversion_recovery_peak,
    _normalize_inversion_recovery_signal,
    _prepare_inversion_recovery_peaks,
)
from hypermri.loading.protocols import VendorHandlerProtocol

from .base_spectroscopy import BaseSpectroscopy


class IVRecovery(BaseSpectroscopy):
    """Inversion-recovery spectroscopy for estimating longitudinal relaxation."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan.

        Parameters
        ----------
        handler : VendorHandlerProtocol
            Vendor handler of the inversion-recovery scan. Its method file must
            contain ``NoIT`` (number of inversion times) and ``InversionTime``
            (inversion times in milliseconds).
        **kwargs
            Options passed unchanged to ``BaseSpectroscopy``.

        Raises
        ------
        KeyError
            If the method file lacks ``NoIT`` or ``InversionTime``; the message
            names the parameter and the scan.
        """
        super().__init__(handler, **kwargs)

        self.NI = self.handler.method_parameter("NoIT")
        self.IVTimes = self.handler.method_parameter("InversionTime")

    def calculate_T1(self, peak_ppms=None, lb=0):
        """Fit the legacy integrated inversion-recovery spectrum to an absolute exponential.

        The legacy ``get_spectra`` preparation is required and checked before
        plotting. No maintained analysis function implements this complete workflow.
        The fit uses ``abs(M0 + A * exp(-time / T1)) + offset`` and normalized
        spectral integrals; inversion times come from the handler in milliseconds.

        Parameters
        ----------
        peak_ppms : sequence of float, optional
            Two spectral integration endpoints in ppm. None integrates the full legacy interval.
        lb : float, optional
            Line broadening in Hz forwarded to the injected spectral preparation.

        Returns
        -------
        numpy.ndarray
            Four fitted parameters ``(M0, A, T1, offset)``. T1 is in milliseconds;
            the other coefficients follow the normalized, dimensionless signal.
            Calculation precision follows SciPy's nonlinear least-squares solver.

        Raises
        ------
        NotImplementedError
            If the legacy ``get_spectra`` preparation is unavailable.
        ValueError, RuntimeError
            If spectra or fit inputs are inconsistent or optimization fails.

        Notes
        -----
        This eager method closes existing Matplotlib figures, creates spectrum and
        fit plots, and prints T1. It does not change acquired samples or cache fit
        coefficients. The legacy fit bounds remain unchanged.
        """
        import matplotlib.pyplot as plt

        missing = [name for name in ["get_spectra"] if not hasattr(self, name)]
        if missing:
            raise NotImplementedError(
                "IVRecovery.calculate_T1 requires unavailable legacy inputs: "
                + ", ".join(missing)
                + ". No maintained implementation supplies the inversion-resolved spectra required by this method."
            )
        if peak_ppms:
            peak_start = np.min(peak_ppms)
            peak_stop = np.max(peak_ppms)
            print(
                [
                    np.argmin(np.abs(self.handler.freq_axis_ppm - peak_start)),
                    np.argmin(np.abs(self.handler.freq_axis_ppm - peak_stop)),
                ]
            )
            peak_indices = [
                np.argmin(np.abs(self.handler.freq_axis_ppm - peak_start)),
                np.argmin(np.abs(self.handler.freq_axis_ppm - peak_stop)),
            ]
            peak_indices.sort()
        else:
            peak_indices = [0, -1]
            # integrate the whole spectrum

        freq_axis_ppm = self.handler.freq_axis_ppm
        specs = self.get_spectra(lb)[0]
        ivs = self.handler.method_parameter("InversionTime")
        NI = len(ivs)
        plt.close("all")
        fig, (ax, int_ax) = plt.subplots(1, 2, figsize=(8, 5), tight_layout=True)
        colors = plt.cm.jet(np.linspace(0, 1, NI))
        for n in range(self.NI):
            ax.plot(
                freq_axis_ppm,
                specs[n] / np.max(specs),
                label="TI= " + str(ivs[n]) + " ms",
                color=colors[n],
            )

        ax.set_xlabel(r"$\sigma [freq_axis_ppm]$")
        ax.set_title("Spectral data")
        ax.set_ylabel("I [a.u.]")
        # ax.legend(loc='best')
        if peak_ppms:
            ax.axvline(peak_ppms[0], 0, 1, color="k", linestyle="dashed")
            ax.axvline(peak_ppms[1], 0, 1, color="k", linestyle="dashed")
        else:
            pass
        ints = [
            np.abs(np.sum(specs[n, peak_indices[0] : peak_indices[1]]))
            for n in range(self.NI)
        ]

        int_ax.set_xlabel("InversionTime [ms]")
        int_ax.set_ylabel("Integrated spectrum [a.U.]")
        int_ax.set_title(
            "Integrated from "
            + str(peak_indices[0])
            + " - "
            + str(peak_indices[0])
            + " freq_axis_ppm"
        )

        # fitting all pixels
        def expo(x, M0, A, T1, offset):
            return np.abs(M0 + A * np.exp(-x / T1)) + offset

        ydata = ints / np.max(ints)
        int_ax.scatter(ivs, ydata, marker="x", color="r", label="data")
        coeff, err = curve_fit(
            expo, ivs, ydata, bounds=([0.3, -80, 10, 0], [1.1, 40, 1000, 1])
        )

        itpl_xax = np.linspace(0, np.max(ivs), 1000)
        int_ax.plot(
            itpl_xax,
            expo(itpl_xax, coeff[0], coeff[1], coeff[2], coeff[3]),
            label="Fit",
        )
        int_ax.set_title("$T_1$ = " + str(np.round(coeff[2], 1)) + " ms")
        print("T1 = ", coeff[2])
        int_ax.legend()

        return coeff

    def GUI_calculate_T1(self, peak_ppms=None, lb=0):
        """Fit the legacy integrated inversion-recovery spectrum to an absolute exponential.

        The legacy ``get_spectra`` preparation is required and checked before
        plotting. No maintained analysis function implements this complete workflow.
        The fit uses ``abs(M0 + A * exp(-time / T1)) + offset`` and normalized
        spectral integrals; inversion times come from the handler in milliseconds.

        Parameters
        ----------
        peak_ppms : sequence of float, optional
            Two spectral integration endpoints in ppm. The legacy display requires both endpoints.
        lb : float, optional
            Line broadening in Hz forwarded to the injected spectral preparation.

        Returns
        -------
        numpy.ndarray
            Four fitted parameters ``(M0, A, T1, offset)``. T1 is in milliseconds;
            the other coefficients follow the normalized, dimensionless signal.
            Calculation precision follows SciPy's nonlinear least-squares solver.

        Raises
        ------
        NotImplementedError
            If the legacy ``get_spectra`` preparation is unavailable.
        ValueError, RuntimeError
            If spectra or fit inputs are inconsistent or optimization fails.
        TypeError
            If endpoints are omitted after spectral preparation is supplied.

        Notes
        -----
        This eager method closes existing Matplotlib figures, creates spectrum and
        fit plots, and prints T1. It does not change acquired samples or cache fit
        coefficients. The legacy fit bounds remain unchanged.
        """
        import matplotlib.pyplot as plt

        missing = [name for name in ["get_spectra"] if not hasattr(self, name)]
        if missing:
            raise NotImplementedError(
                "IVRecovery.GUI_calculate_T1 requires unavailable legacy inputs: "
                + ", ".join(missing)
                + ". No maintained implementation supplies the inversion-resolved spectra required by this method."
            )
        if peak_ppms:
            peak_start = np.min(peak_ppms)
            peak_stop = np.max(peak_ppms)
            print(
                [
                    np.argmin(np.abs(self.handler.freq_axis_ppm - peak_start)),
                    np.argmin(np.abs(self.handler.freq_axis_ppm - peak_stop)),
                ]
            )
            peak_indices = [
                np.argmin(np.abs(self.handler.freq_axis_ppm - peak_start)),
                np.argmin(np.abs(self.handler.freq_axis_ppm - peak_stop)),
            ]
            peak_indices.sort()
        else:
            peak_indices = [0, -1]
            # integrate the whole spectrum

        freq_axis_ppm = self.handler.freq_axis_ppm
        specs = self.get_spectra(lb)[0]
        ivs = self.handler.method_parameter("InversionTime")
        NI = len(ivs)
        plt.close("all")
        fig, (ax, int_ax) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        colors = plt.cm.jet(np.linspace(0, 1, NI))
        for n in range(self.NI):
            ax.plot(
                freq_axis_ppm,
                specs[n] / np.max(specs),
                label="TI= " + str(ivs[n]) + " ms",
                color=colors[n],
            )

        ax.set_xlabel(r"$\sigma [freq_axis_ppm]$")
        ax.set_title("Spectral data")
        ax.set_ylabel("I [a.u.]")
        ax.legend(loc="best")
        ax.axvline(peak_ppms[0], 0, 1, color="k", linestyle="dashed")
        ax.axvline(peak_ppms[1], 0, 1, color="k", linestyle="dashed")

        ints = [
            np.abs(np.sum(specs[n, peak_indices[0] : peak_indices[1]]))
            for n in range(self.NI)
        ]

        int_ax.set_xlabel("InversionTime [ms]")
        int_ax.set_ylabel("Integrated spectrum [a.U.]")
        int_ax.set_title(
            "Integrated from "
            + str(peak_indices[0])
            + " - "
            + str(peak_indices[0])
            + " freq_axis_ppm"
        )

        # fitting all pixels
        def expo(x, M0, A, T1, offset):
            return np.abs(M0 + A * np.exp(-x / T1)) + offset

        ydata = ints / np.max(ints)
        int_ax.scatter(ivs, ydata, marker="x", color="r", label="data")
        coeff, err = curve_fit(
            expo, ivs, ydata, bounds=([0.5, -80, 100, 0], [1.1, 40, 60000, 1])
        )

        itpl_xax = np.linspace(0, np.max(ivs), 1000)
        int_ax.plot(
            itpl_xax,
            expo(itpl_xax, coeff[0], coeff[1], coeff[2], coeff[3]),
            label="Fit",
        )
        int_ax.set_title("$T_1$ = " + str(np.round(coeff[2], 1)) + " ms")
        print("T1 = ", coeff[2])
        int_ax.legend()

        return coeff

    def InversionRecovery_Calculate_T1_multi_peak(
        self, intervalls, bounds, axlist=None
    ):
        """Fit multiple legacy spectral integrals to inversion recovery.

        The injected ``get_ppm`` preparation is required and checked first. Raw job
        samples are Fourier transformed at each inversion, absolute spectral
        integrals are normalized per peak, and each peak is fit to
        ``abs(M0 + A * exp(-time / T1)) + offset``. Inversion times use milliseconds.

        Parameters
        ----------
        intervalls : sequence of int
            Flat, paired start/stop spectral indices, with two entries per peak.
        bounds : tuple of array_like
            Lower and upper bounds for ``(M0, A, T1, offset)``; each has four entries
            or is a scalar accepted by SciPy. T1 bounds are in milliseconds.
        axlist : sequence of matplotlib.axes.Axes, optional
            Spectrum axis followed by one fit axis per peak. None creates the axes.

        Returns
        -------
        coeffs : list of numpy.ndarray
            One four-parameter fit per peak in ``(M0, A, T1, offset)`` order.
            T1 is in milliseconds; other coefficients follow normalized signal.
        coeff_errs : list of numpy.ndarray
            One SciPy covariance matrix of shape (4, 4) per fit. These are covariance
            estimates, despite the legacy return name, rather than standard errors.

        Raises
        ------
        NotImplementedError
            If the legacy ``get_ppm`` preparation is unavailable.
        ValueError, RuntimeError
            If intervals, raw sample counts or bounds are invalid or a fit fails.

        Notes
        -----
        This eager route reads ``handler.rawdatajob0``, plots spectra and fits,
        prints T1 and may update spectral metadata through the injected preparation.
        The FFT retains NumPy's precision policy and acquired samples are preserved.
        Numerical stages share the analysis operation
        ``fit_inversion_recovery_peaks``; this shell retains preparation/plot/print
        ordering, including partially drawn axes when a later fit fails.
        """
        import matplotlib.pyplot as plt

        missing = [name for name in ["get_ppm"] if not hasattr(self, name)]
        if missing:
            raise NotImplementedError(
                "IVRecovery.InversionRecovery_Calculate_T1_multi_peak requires unavailable legacy inputs: "
                + ", ".join(missing)
                + ". No maintained implementation supplies the legacy get_ppm preparation required by this method."
            )
        NI = self.handler.method_parameter("NoIT")
        n_acquisition = self.n_acquisition
        self.get_ppm(0)
        n_peaks = int(len(intervalls) / 2)
        specs, integrals = _prepare_inversion_recovery_peaks(
            self.handler.rawdatajob0, NI, n_acquisition, intervalls
        )
        ivs = self.handler.method_parameter("InversionTime")
        coeffs = []
        coeff_errs = []

        if axlist is None:
            fig, ax = plt.subplots(1, (n_peaks + 1), figsize=(8, 5))
        else:
            ax = axlist

        normalized_specs = _normalize_inversion_recovery_signal(specs)
        for n in range(NI):
            ax[0].plot(
                np.arange(n_acquisition),
                normalized_specs[n],
                label="TI= " + str(ivs[n]) + " ms",
            )

        ax[0].set_xlabel(r"$\sigma[freq_axis_ppm]$")
        ax[0].set_title("Spectral data")
        ax[0].legend()

        for idx, integral in enumerate(integrals):
            ax[idx + 1].set_xlabel("InversionTime [ms]")
            ax[idx + 1].set_ylabel("Integrated spectrum [a.U.]")
            ydata = _normalize_inversion_recovery_signal(integral)
            ax[idx + 1].scatter(ivs, ydata, marker="x", color="r", label="data")
            coeff, err, itpl_xax, fit_signal = _fit_inversion_recovery_peak(
                ydata, ivs, bounds
            )
            ax[idx + 1].plot(
                itpl_xax,
                fit_signal,
                label="Fit",
            )
            ax[idx + 1].set_title(
                "$T_1$ Peak "
                + str(idx + 1)
                + " = "
                + str(np.round(coeff[2], 1))
                + " ms"
            )
            print("T1 = ", coeff[2])
            ax[idx + 1].legend()

            coeffs.append(coeff)
            coeff_errs.append(err)

        return coeffs, coeff_errs

    def Interactive_Fitting(self, intervalls, axlist):
        """Open the legacy multi-peak inversion-recovery fit controls.

        Parameters
        ----------
        intervalls : sequence of int
            Paired start/stop spectral integration indices.
        axlist : sequence of matplotlib.axes.Axes
            Axes modified by the legacy fit display.

        Returns
        -------
        None
            Displays widgets after validating the legacy prerequisites.

        Raises
        ------
        NotImplementedError
            If get_ppm preparation is unavailable; checked before constructing widgets.
        ImportError
            If the optional ``hypermri[notebook]`` runtime is unavailable.

        Notes
        -----
        The controls target the four-parameter absolute exponential model; T1 uses
        milliseconds. Calculation delegates through the retained multi-peak
        display shell to analysis stages shared with ``fit_inversion_recovery_peaks``.
        The injected legacy get_ppm prerequisite is still required; opening the
        controls does not supply it or migrate to quantitative MRI's T1 grid model.
        """
        # create sliders to adjust bounds
        if not hasattr(self, "get_ppm"):
            raise NotImplementedError(
                "IVRecovery.Interactive_Fitting requires unavailable legacy get_ppm preparation; no maintained replacement exists."
            )
        from hypermri._optional import _import_optional

        widgets = _import_optional("ipywidgets", "notebook")
        display = _import_optional("IPython.display", "notebook").display

        M0_slider = widgets.FloatRangeSlider(
            value=[-10.0, 10.0],
            min=-10.0,
            max=10.0,
            step=0.1,
            description="M0:",
            disabled=False,
            readout=True,
            readout_format=".1f",
        )

        A_slider = widgets.FloatRangeSlider(
            value=[-10.0, 10.0],
            min=-10.0,
            max=10.0,
            step=0.1,
            description="A:",
            disabled=False,
            readout=True,
            readout_format=".1f",
        )

        T1_slider = widgets.FloatRangeSlider(
            value=[0.0, 10000.0],
            min=0.0,
            max=10000.0,
            step=0.1,
            description="T1:",
            disabled=False,
            readout=True,
            readout_format=".1f",
        )

        offset_slider = widgets.FloatRangeSlider(
            value=[-5.0, 5.0],
            min=-5.0,
            max=5.0,
            step=0.1,
            description="offset:",
            disabled=False,
            readout=True,
            readout_format=".1f",
        )

        ui_main = widgets.VBox(
            [M0_slider, A_slider, T1_slider, offset_slider],
            layout=widgets.Layout(display="flex"),
        )

        def plot(M0_input, A_input, T1_input, offset_input):
            # build bounds dict
            bounds = np.array(
                [
                    [M0_input[0], A_input[0], T1_input[0], offset_input[0]],
                    [M0_input[1], A_input[1], T1_input[1], offset_input[1]],
                ]
            )

            for ax in axlist:
                ax.clear()

            output = self.InversionRecovery_Calculate_T1_multi_peak(
                intervalls, bounds, axlist
            )

            print("M0: " + str(output[0][0][0]) + " and " + str(output[0][1][0]))
            print("A: " + str(output[0][0][1]) + " and " + str(output[0][1][1]))
            print("T1: " + str(output[0][0][2]) + " and " + str(output[0][1][2]))
            print("Offset: " + str(output[0][0][3]) + " and " + str(output[0][1][3]))

        out = widgets.interactive_output(
            plot,
            {
                "M0_input": M0_slider,
                "A_input": A_slider,
                "T1_input": T1_slider,
                "offset_input": offset_slider,
            },
        )

        display(ui_main, out)
