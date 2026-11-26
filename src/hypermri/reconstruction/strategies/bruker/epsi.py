"""Raw reconstruction for Bruker echo-planar spectroscopic imaging (EPSI).

ParaVision's ``<Bruker:EPSI>`` method samples one spatial read direction and
the spectral dimension with an echo train of bipolar read gradients. Each phase
encoding step is repeated for ``NSegments`` (``PVM_EpiNShots``) excitations.
Segment ``s`` delays its echo train by ``s`` times the spectral dwell time, so
the echo ``e`` of segment ``s`` is the spectral sample ``e * NSegments + s``.
The interleaved spectral dwell time is ``PVM_EpiEchoSpacing / NSegments``.

The result has the layout of the MGE reconstruction: one complex FID per voxel
along ``acquisition_idx``, with time and frequency coordinates. The spectral
Fourier transform is left to spectroscopy analysis, as for CSI.
"""

from typing import ClassVar

import numpy as np

from hypermri.reconstruction.centred_dft import centered_fft, centered_ifft

from .epi import BrukerEpiRawRecoStrategy
from .raw import BrukerMGERawRecoStrategy, _fill_encoded_kspace

#: Relative tolerance between the interleaved dwell time and the segment
#: delays ``PVM_EpiEchoShiftA``, and between the dwell time and
#: ``1 / PVM_EpiEffBandwidth``. ParaVision stores both to at least 1 µs on a
#: dwell of about 0.1 ms, so 2 % accepts rounding and rejects another layout.
_EPSI_TIMING_RTOL = 0.02

#: Fraction of the strongest read position below which a voxel does not enter
#: the odd/even phase estimate; noise-only positions have random phase.
_ODD_EVEN_WEIGHT_THRESHOLD = 0.05


def epsi_dwell_time_ms(method) -> float:
    """Return the spectral dwell time of a Bruker EPSI acquisition.

    Parameters
    ----------
    method : Mapping
        ParaVision method parameters with ``PVM_EpiEchoSpacing`` in ms and
        ``NSegments`` or ``PVM_EpiNShots``. ``PVM_EpiEchoShiftA`` (ms) and
        ``PVM_EpiEffBandwidth`` (Hz) are checked when present.

    Returns
    -------
    float
        Time between consecutive interleaved spectral samples in ms, equal to
        the echo spacing divided by the number of segments.

    Raises
    ------
    ValueError
        If the echo spacing or segment count is missing or not positive, or if
        the segment delays or the effective bandwidth contradict the
        interleaved dwell time.

    Examples
    --------
    >>> epsi_dwell_time_ms({"PVM_EpiEchoSpacing": 0.48, "NSegments": 4})
    0.12
    """
    n_segments = epsi_segment_count(method)
    try:
        spacing_ms = float(method["PVM_EpiEchoSpacing"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Bruker EPSI reconstruction requires a numeric PVM_EpiEchoSpacing in ms."
        ) from error
    if not spacing_ms > 0:
        raise ValueError(
            f"Bruker EPSI PVM_EpiEchoSpacing must be positive, got {spacing_ms}."
        )
    dwell_ms = spacing_ms / n_segments

    shifts = method.get("PVM_EpiEchoShiftA")
    if shifts is not None:
        shifts_ms = np.asarray(shifts, dtype=float).reshape(-1)
        if shifts_ms.size != n_segments or not np.allclose(
            np.diff(shifts_ms), dwell_ms, rtol=_EPSI_TIMING_RTOL
        ):
            raise ValueError(
                "Bruker EPSI PVM_EpiEchoShiftA does not delay each of the "
                f"{n_segments} segments by one dwell time of {dwell_ms:g} ms: "
                f"{shifts_ms.tolist()}. Use scan.reconstruct('seq2d') for the "
                "scanner reconstruction."
            )
    bandwidth = method.get("PVM_EpiEffBandwidth")
    if bandwidth is not None and not np.isclose(
        1000.0 / float(bandwidth), dwell_ms, rtol=_EPSI_TIMING_RTOL
    ):
        raise ValueError(
            f"Bruker EPSI PVM_EpiEffBandwidth={float(bandwidth):g} Hz does not "
            f"match the interleaved dwell time of {dwell_ms:g} ms."
        )
    return dwell_ms


def epsi_segment_count(method) -> int:
    """Return the number of interleaved echo-train segments of Bruker EPSI.

    Parameters
    ----------
    method : Mapping
        ParaVision method parameters with ``NSegments`` and/or
        ``PVM_EpiNShots``.

    Returns
    -------
    int
        Positive number of segments; 1 when neither parameter is present.

    Raises
    ------
    ValueError
        If a value is not a positive integer or the two parameters disagree.
    """
    values = {
        name: method.get(name)
        for name in ("NSegments", "PVM_EpiNShots")
        if method.get(name) is not None
    }
    counts = set()
    for name, value in values.items():
        try:
            count = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Bruker EPSI {name} must be a positive integer, got {value!r}."
            ) from error
        if count < 1 or count != float(value):
            raise ValueError(
                f"Bruker EPSI {name} must be a positive integer, got {value!r}."
            )
        counts.add(count)
    if len(counts) > 1:
        raise ValueError(
            f"Bruker EPSI segment counts disagree: {values}. Use "
            "scan.reconstruct('seq2d') for the scanner reconstruction."
        )
    return counts.pop() if counts else 1


def correct_odd_even_echo_phase(kspace, *, echo_axis, read_axis):
    """Remove the linear read phase between odd and even EPSI echoes.

    Echoes acquired under the negative read gradient (odd index) are shifted
    against even echoes by gradient delays and eddy currents, which appears as
    a phase linear in the read position after the read Fourier transform. The
    phase is estimated from the data: each odd echo is compared with the mean
    of its two even neighbours, which cancels the spectral phase evolution to
    first order. A straight line is fitted to the phase difference over read
    positions, weighted by signal magnitude, and removed from the odd echoes.

    Parameters
    ----------
    kspace : numpy.ndarray
        Complex k-space with odd echoes already reversed along ``read_axis``.
        All axes other than ``echo_axis`` and ``read_axis`` are pooled into
        one estimate.
    echo_axis : int
        Axis of the echoes of one echo train, before segment interleaving.
    read_axis : int
        Axis of the read samples.

    Returns
    -------
    numpy.ndarray
        Corrected k-space; a new array with the dtype of ``kspace``.
    dict
        ``epsi_odd_even_phase_rad`` with the constant and the linear phase
        (rad and rad per read pixel, relative to the read centre index
        ``n // 2``) that was removed from the odd echoes. Both are zero when
        fewer than three echoes or two read positions carry signal.
    """
    n_read = kspace.shape[read_axis]
    n_echoes = kspace.shape[echo_axis]
    coefficients = np.zeros(2)
    hybrid = centered_ifft(kspace, (read_axis,))
    if n_echoes >= 3:
        echoes = np.moveaxis(hybrid, (echo_axis, read_axis), (-2, -1))
        odd = echoes[..., 1:-1:2, :]
        even = 0.5 * (echoes[..., 0:-2:2, :] + echoes[..., 2::2, :])
        cross = (odd * np.conj(even)).reshape(-1, n_read).sum(axis=0)
        weight = np.abs(cross)
        position = np.arange(n_read, dtype=float) - n_read // 2
        used = weight > _ODD_EVEN_WEIGHT_THRESHOLD * weight.max()
        if np.count_nonzero(used) >= 2:
            phase = np.unwrap(np.angle(cross[used]))
            root_weight = np.sqrt(weight[used])
            design = np.stack([np.ones(used.sum()), position[used]], axis=1)
            coefficients = np.linalg.lstsq(
                design * root_weight[:, None], phase * root_weight, rcond=None
            )[0]
        correction = np.exp(-1j * (coefficients[0] + coefficients[1] * position))
        shape = [1] * kspace.ndim
        shape[read_axis] = n_read
        odd_index = [slice(None)] * kspace.ndim
        odd_index[echo_axis] = slice(1, None, 2)
        hybrid[tuple(odd_index)] *= correction.reshape(shape)
    corrected = centered_fft(hybrid, (read_axis,)).astype(kspace.dtype, copy=False)
    return corrected, {"epsi_odd_even_phase_rad": coefficients.tolist()}


class BrukerEpsiRawRecoStrategy(BrukerMGERawRecoStrategy):
    """Reconstruct Bruker EPSI raw data into one FID per voxel.

    The k-space preparation follows the ParaVision online reconstruction:
    reverse the odd echoes of each echo train, regrid ramp-sampled read data
    with ``PVM_EpiTrajAdjkx``, remove the odd/even echo phase and interleave the
    ``NSegments`` echo trains into one spectral time axis. The read and phase
    Fourier transforms and the labelling are those of
    :class:`~hypermri.reconstruction.strategies.bruker.raw.BrukerMGERawRecoStrategy`.

    The odd/even phase is estimated from the data by
    :func:`correct_odd_even_echo_phase`. ParaVision's stored
    ``PVM_EpiPhaseCorrection`` is not used, because its coordinate convention
    for EPSI is not documented and could not be reproduced.
    """

    accepted_options: ClassVar[frozenset[str]] = (
        BrukerMGERawRecoStrategy.accepted_options
    )

    def _echo_times_ms(self, handler):
        """Return the spectral sample times of the interleaved FID in ms.

        Parameters
        ----------
        handler : VendorHandler
            Handler with the ParaVision ``method`` parameters and
            ``n_echoes`` (``SpecSize``, the interleaved FID length).

        Returns
        -------
        numpy.ndarray
            ``n * dwell`` for ``n = 0 .. n_echoes - 1``, with the dwell time of
            :func:`epsi_dwell_time_ms`. Time zero is the centre of the first
            echo of the first segment.
        """
        dwell_ms = epsi_dwell_time_ms(handler.method)
        return np.arange(int(handler.n_echoes)) * dwell_ms

    def reshape_fid_to_kspace(self, values=None, handler=None):
        """Sort an EPSI FID into k-space with an interleaved spectral axis.

        ParaVision stores, from the outermost loop inwards, repetitions, phase
        lines, segments, echoes and read samples. Echo ``e`` of segment ``s``
        becomes spectral sample ``e * NSegments + s``.

        Parameters
        ----------
        values : numpy.ndarray, optional
            Complex FID samples. Defaults to the acquired ``handler.fid``.
            Not modified.
        handler : VendorHandler
            Handler of a single-slice, single-receiver 2D EPSI scan with
            ``enc_matrix``, ``matrix``, ``n_repetitions``, ``n_echoes``
            (``SpecSize``) and the ParaVision ``method`` parameters
            ``NSegments``/``PVM_EpiNShots``, ``PVM_EpiNEchoes`` and
            ``PVM_EncSteps1``.

        Returns
        -------
        numpy.ndarray
            K-space with shape ``(1, repetitions, 1, phase, spectral samples,
            1, read)`` in the layout of
            :meth:`BrukerMGERawRecoStrategy.reshape_fid_to_kspace`. Odd echoes
            are reversed, the read axis is regridded when ParaVision measured
            the ramp trajectory, and the odd/even echo phase is removed.

        Raises
        ------
        ValueError
            If the scan has several slices or receivers, if the echo and
            segment counts do not give ``SpecSize``, if the FID length does not
            match, or if the timing parameters are inconsistent.
        """
        fid = handler.fid if values is None else values
        method = handler.method
        n_segments = epsi_segment_count(method)
        n_echoes = int(method.get("PVM_EpiNEchoes", handler.n_echoes))
        n_spectral = int(handler.n_echoes)
        n_read, n_phase, n_slices = (int(size) for size in handler.enc_matrix)
        epsi_dwell_time_ms(method)
        if n_slices != 1 or int(handler.n_receivers) != 1:
            raise ValueError(
                "Bruker EPSI raw reconstruction supports one slice and one "
                f"receiver, got {n_slices} slices and {handler.n_receivers} "
                "receivers. Use scan.reconstruct('seq2d') for the scanner images."
            )
        if n_echoes * n_segments != n_spectral:
            raise ValueError(
                f"Bruker EPSI PVM_EpiNEchoes={n_echoes} times NSegments="
                f"{n_segments} does not equal SpecSize={n_spectral}."
            )
        try:
            trains = np.reshape(
                fid,
                (handler.n_repetitions, n_phase, n_segments, n_echoes, n_read),
            )
        except ValueError as error:
            raise ValueError(
                f"Failed to reshape Bruker EPSI raw data with {n_segments} segments "
                f"of {n_echoes} echoes: {error}"
            ) from error

        trains = trains.copy()
        trains[:, :, :, 1::2] = trains[:, :, :, 1::2, ::-1]
        trains, _ = BrukerEpiRawRecoStrategy()._apply_trajectory_regridding(
            trains, method, {}, axis=4
        )
        trains, _ = correct_odd_even_echo_phase(trains, echo_axis=3, read_axis=4)
        spectral = np.swapaxes(trains, 2, 3).reshape(
            handler.n_repetitions, n_phase, n_spectral, n_read
        )
        kspace = spectral[None, :, None, :, :, None, :]
        return _fill_encoded_kspace(kspace, handler, read_axis=6)
