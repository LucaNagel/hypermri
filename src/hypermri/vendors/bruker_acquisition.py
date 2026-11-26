"""Fill the vendor-neutral CSI acquisition description from a Bruker scan.

This module is the only place that turns the ParaVision parameters of a CSI
scan into :class:`hypermri.reconstruction.csi_acquisition.CSIAcquisition`.
The encoding positions come from
:meth:`hypermri.reconstruction.strategies.bruker.csi.BrukerCsiRecoStrategy.get_phase_encoding_order`,
which remains their single parser.
"""

from __future__ import annotations

from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy


def bruker_to_csi_acquisition(handler) -> CSIAcquisition:
    """Describe a Bruker CSI scan as a :class:`CSIAcquisition`.

    Parameters
    ----------
    handler : hypermri.vendors.bruker.BrukerHandler
        Parsed Bruker scan of a supported CSI method. Uses the canonical
        handler attributes ``matrix`` and ``fov_mm`` (read, phase first;
        ``fov_mm`` may be ``None``),
        ``read_offset_mm``, ``phase_offset_mm``, ``slice_offset_mm``,
        ``n_acquisition``, ``dwell_time_s``, ``reference_frequency_mhz`` and
        ``center_ppm``, and the encoding parameters in ``handler.method``.

    Returns
    -------
    CSIAcquisition
        Encoding order and one read/phase grid position per acquired FID,
        matrix and field of view of the read and phase axes, field-of-view
        offset ``(read, phase, slice)`` in mm, samples per FID, dwell time in
        s, reference frequency in MHz and centre chemical shift in ppm.

    Raises
    ------
    KeyError
        If an encoding parameter is missing from ``handler.method``.
    ValueError
        If the method or encoding order is not supported, or if a value does
        not pass the :class:`CSIAcquisition` validation. The message names the
        parameter or field.

    Notes
    -----
    Deterministic and eager; reads no acquired samples and changes neither
    the handler nor its parameters. A custom method without stored pairs logs
    the warning of the encoding-order parser.

    Examples
    --------
    >>> acquisition = bruker_to_csi_acquisition(scan.handler)
    >>> acquisition.matrix
    (16, 16)
    """
    encoding_order, read_positions, phase_positions = (
        BrukerCsiRecoStrategy().get_phase_encoding_order(handler)
    )
    return CSIAcquisition(
        encoding_order=encoding_order,
        matrix=tuple(handler.matrix[:2]),
        fov_mm=None if handler.fov_mm is None else tuple(handler.fov_mm[:2]),
        fov_offset_mm=(
            handler.read_offset_mm,
            handler.phase_offset_mm,
            handler.slice_offset_mm,
        ),
        read_positions=read_positions,
        phase_positions=phase_positions,
        n_points=handler.n_acquisition,
        dwell_time_s=handler.dwell_time_s,
        reference_frequency_mhz=handler.reference_frequency_mhz,
        center_ppm=handler.center_ppm,
    )


__all__ = ["bruker_to_csi_acquisition"]
