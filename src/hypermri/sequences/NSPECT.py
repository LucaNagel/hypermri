"""Provide NSPECT sequence utilities with package-controlled logging."""

import numpy as np

from hypermri._optional import _notebook_alias as __getattr__
from hypermri.loading.protocols import VendorHandlerProtocol
from hypermri.reconstruction.axes import get_freq_axis
from hypermri.utils.mri.fourier import spectral_fft, spectral_ifft
from hypermri.utils.mri.signal import rotate_phase

from ..utils.system.logging import LOG_MODES as LOG_MODES
from ..utils.system.logging import init_default_logger
from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__)


class NSPECT(BaseSpectroscopy):
    """Wrapper which decides wether to load the single or dualchannel NSPECT class.

    Built from the vendor handler of its scan.
    """

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)

    def get_spec(self, LB=0, cut_off=0):
        """
        Calculate spectra and ppm axis for non localized spectroscopy measurements with repetitions.

        Parameters
        ----------
        LB : float, optional
            linebroadening applied in Hz, default is 0
        cut_off : float, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        spec : array
            magnitude spectra in a n-dimensional array
        fids: array
            linebroadened fids in a n-dimensional array
        complex_spec: array
            complex spectra in n-dimensional array

        Raises
        ------
        KeyError
            If the method file lacks ``PVM_SpecAcquisitionTime`` (acquisition
            time in milliseconds); the message names the parameter and the scan.
        """
        # if we dont have rawdatajob0 file we load fid
        if len(self.handler.rawdatajob0) == 0:
            fid = self.handler.fid
        else:
            fid = self.handler.rawdatajob0

        n_repetitions = self.n_repetitions
        n_acquisition = self.n_acquisition
        ac_time = self.handler.method_parameter("PVM_SpecAcquisitionTime")

        time_ax = np.linspace(0, ac_time, n_acquisition)[cut_off:] / 1000
        sigma = 2.0 * np.pi * LB

        fids = np.zeros((n_repetitions, n_acquisition - cut_off), dtype=complex)
        spec = np.zeros((n_repetitions, n_acquisition - cut_off))
        complex_spec = np.zeros((n_repetitions, n_acquisition - cut_off), dtype=complex)

        def ffts(x):
            return spectral_fft(x, axis=-1)

        for rep in range(n_repetitions):
            tmp_spec = ffts(
                fid[cut_off + rep * n_acquisition : n_acquisition + rep * n_acquisition]
                * np.exp(-sigma * time_ax)
            )

            complex_spec[rep, :] = tmp_spec

            fids[rep, :] = fid[
                cut_off + rep * n_acquisition : n_acquisition + rep * n_acquisition
            ] * np.exp(-sigma * time_ax)

        spec = np.abs(complex_spec)

        s = np.squeeze
        return s(spec), s(fids), s(complex_spec)

    def phase_correct_fid(self, number=0):
        """Display a phase slider for one acquired FID repetition.

        Parameters
        ----------
        number : int, optional
            Zero-based repetition index in ``handler.fid``. The selected complex
            FID is transformed with the existing FFT/centering convention.

        Returns
        -------
        None
            Displays Matplotlib curves and notebook controls eagerly. Phase uses
            degrees and plotted intensities use arbitrary signal units. Acquired
            FID values are not mutated; slider changes update the display only.

        Raises
        ------
        ImportError
            If ipywidgets is missing from the environment.
        IndexError
            If the requested repetition is outside the acquired FID array.
        """
        import matplotlib.pyplot as plt

        from hypermri._optional import _import_optional

        widgets = _import_optional("ipywidgets")
        fid = self.handler.fid[number, :]
        spec_complex = spectral_fft(fid, axis=-1)
        # perform a baseline correction
        spec_base_line_corr = spec_complex - np.mean(spec_complex)
        # phase the real spectrum
        Integrals_th = []
        phases = np.linspace(0, 360, 50)
        for phase in phases:
            itgl = np.sum(
                np.real(rotate_phase(spec_base_line_corr, (phase * np.pi) / 180.0))
            )
            Integrals_th.append(itgl)
        initial_guess_phase = phases[
            np.argmin(np.abs(Integrals_th - np.max(Integrals_th)))
        ]

        fig, ax = plt.subplots(1, figsize=(12, 4), tight_layout=True)

        (line_real,) = ax.plot(
            np.real(
                rotate_phase(spec_base_line_corr, (initial_guess_phase * np.pi) / 180.0)
            ),
            label="Real-phased",
            color="k",
        )
        (line_abs,) = ax.plot(
            np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr)),
            label="Magnitude",
            color="r",
        )
        ax.hlines(
            0,
            0,
            len(spec_base_line_corr),
            linestyles="dashed",
            alpha=0.3,
            color="b",
            label="Baseline",
        )
        # ax.fill_between([8000, 9000],np.min(np.real(spec_base_line_corr)),np.max(np.real(spec_base_line_corr)), alpha=0.3, color='C2', label='Background')
        ax.set_xlabel("Points")
        ax.set_ylabel("I [a.u.]")
        ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")
        ax.legend()

        @widgets.interact(phase=(0, 360, 0.1))
        def update(phase):
            line_real.set_ydata(
                np.real(rotate_phase(spec_base_line_corr, (phase * np.pi) / 180.0))
            )
            line_abs.set_ydata(
                np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr))
            )
            ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")
