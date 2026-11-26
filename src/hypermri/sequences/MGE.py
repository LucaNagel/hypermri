"""Multi-echo gradient echo (MGE) sequence utilities."""

import numpy as np

from hypermri.loading.protocols import VendorHandlerProtocol

from .base_spectroscopy import BaseSpectroscopy


class MGE(BaseSpectroscopy):
    """Represent a multi-echo gradient echo scan and its echo-dependent data.

    Parameters
    ----------
    handler : VendorHandlerProtocol
        Vendor handler of the scan, with EffectiveTE in milliseconds.
    **kwargs
        Loading and construction options; see ``BaseSequence``.

    Notes
    -----
    Echo timing belongs to the sequence. Acquired readout timing remains on the
    handler. Lazy construction reads metadata without reconstructing samples.
    """

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("seq2d",),
        "full": ("seq2d", "raw"),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Construct an MGE sequence with local echo timing metadata.

        Parameters
        ----------
        handler : VendorHandlerProtocol
            Vendor handler of the scan. Acquired readout sample count, dwell
            time, bandwidth and sampling times remain on that handler.
        **kwargs
            BaseSequence construction options including load_data, log_mode,
            reco_options, path, force_vendor and handler_options. Loading follows the
            sequence policy; a lazy construction reads metadata only.

        Raises
        ------
        TypeError
            If ``handler`` is not a vendor handler or a keyword is unknown.
        ValueError
            If the requested loading policy or reconstruction metadata are invalid.
        KeyError
            If an MGE or EPSI scan lacks its echo timing parameter; the
            message names the parameter and the scan.

        Notes
        -----
        EffectiveTE is interpreted in milliseconds and local sampling_time_axis_s
        uses seconds. Echo bandwidth uses uniform echo spacing (a single echo uses
        its echo time). ``bandwidth_ppm`` divides the echo bandwidth in Hz by the
        handler's ``reference_frequency_mhz`` and is None when the scan has no
        reference frequency. No acquired samples or handler timing metadata are
        mutated.
        """
        super().__init__(handler, **kwargs)

        # Initialize MGE specific timing parameters
        self.TEs = self.get_TEs()
        if self.TEs is not None:
            self.TEs = np.atleast_1d(self.TEs)
            # ParaVision EffectiveTE is expressed in milliseconds.
            self.sampling_time_axis_s = self.TEs / 1000.0

            # Calculate dwell time from echoes
            if len(self.sampling_time_axis_s) > 1:
                # Assume uniform spacing for dwell time calculation
                self.dwell_time_s = (
                    self.sampling_time_axis_s[1] - self.sampling_time_axis_s[0]
                )
            elif len(self.sampling_time_axis_s) == 1:
                self.dwell_time_s = self.sampling_time_axis_s[0]
            else:
                self.dwell_time_s = 0.001  # Default to 1ms if unknown

            # Set bandwidth_hz for compatibility with spectroscopy tools
            if self.dwell_time_s > 0:
                self.bandwidth_hz = 1.0 / self.dwell_time_s
            else:
                self.bandwidth_hz = 0.0

            # The handler owns the 0-ppm reference frequency (PVM_FrqRef of
            # the observed nucleus); without it the ppm bandwidth is unknown.
            reference_frequency_mhz = getattr(
                self.handler, "reference_frequency_mhz", None
            )
            if reference_frequency_mhz is not None and reference_frequency_mhz > 0:
                self.bandwidth_ppm = self.bandwidth_hz / reference_frequency_mhz
            else:
                self.bandwidth_ppm = None

            # Echo metadata belong to the sequence, acquired readout metadata
            # remain on the vendor handler.
            self.n_acquisition = len(self.TEs)

    def get_TEs(self):
        """Return protocol echo timing without loading or mutating samples.

        Returns
        -------
        scalar, array-like or None
            EffectiveTE in milliseconds in protocol echo order, or EPSI echo
            spacing in milliseconds. The protocol value itself is returned;
            callers needing an array should use ``numpy.atleast_1d``. None
            indicates an unrecognised method without EffectiveTE.

        Raises
        ------
        KeyError
            If a recognised MGE or EPSI protocol omits its required echo
            timing; the message names the parameter and the scan.
        """
        if self.type == "<Bruker:MGE>":
            return self.handler.method_parameter("EffectiveTE")
        elif self.type == "<Bruker:EPSI>":
            return self.handler.method_parameter("PVM_EpiEchoSpacing")
        elif "EffectiveTE" in self.handler.method:
            return self.handler.method_parameter("EffectiveTE")
        else:
            pass
