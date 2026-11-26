"""Lorentzian line shapes and complex free-induction decay generation."""

import numpy as np

__all__ = [
    "lorentzian",
    "sqrt_lorentzian",
    "generate_fid",
    "propagate_isochromats",
    "combine_isochromats",
]


def lorentzian(x, FWHM, center_position, peak_height) -> np.ndarray:
    """Evaluate a Lorentzian line shape.

    Parameters
    ----------
    x : array_like
        Frequency axis in the same units as ``center_position``.
    FWHM : float
        Full width at half maximum of the peak.
    center_position : float
        Peak centre location in the units of ``x``.
    peak_height : float
        Peak amplitude at the centre.

    Returns
    -------
    numpy.ndarray
        Lorentzian profile evaluated at ``x``.
    """
    return peak_height * FWHM**2 / (FWHM**2 + (2 * x - 2 * center_position) ** 2)


def sqrt_lorentzian(x, FWHM, center_position, peak_height):
    """Evaluate the retained square-root Lorentzian peak envelope.

    Parameters
    ----------
    x : array_like
        Frequency samples in the same units as the peak centre and width.
    FWHM : float
        Full width at half maximum in frequency-axis units.
    center_position : float
        Peak centre on the supplied frequency axis.
    peak_height : float
        Envelope height in arbitrary signal units.

    Returns
    -------
    numpy.ndarray
        Envelope with the shape of x and NumPy calculation precision. Inputs
        are not mutated; evaluation is eager, deterministic and uncached.
    """
    return peak_height / np.sqrt(1 + 3 * (x - center_position) ** 2 / (FWHM / 2) ** 2)


def generate_fid(
    time_axis_s: np.ndarray | None = None,
    amplitude: float = 1.0,
    t2_s: float = 0.02,
    freq_hz: float = 0.0,
    phase_rad: float = 0.0,
    dwell_time_s: float = 5e-4,
    npoints: int = 1000,
    noise_std: float = 0.0,
    sum_fids: bool = True,
) -> np.ndarray | None:
    """Generate complex Lorentzian free-induction decays on a sampling axis.

    Parameters
    ----------
    time_axis_s : numpy.ndarray or None, optional
        One-dimensional sample times in seconds. None constructs ``npoints``
        samples at ``dwell_time_s`` intervals starting at zero. An explicit
        axis determines the output sample count and is not mutated.
    amplitude : float or numpy.ndarray, optional
        Component amplitudes in arbitrary signal units. Amplitude, decay time,
        frequency and phase arrays must broadcast to one parameter shape.
    t2_s : float or numpy.ndarray, optional
        Positive transverse exponential decay times in seconds.
    freq_hz : float or numpy.ndarray, optional
        Component frequencies relative to the carrier in Hz, using the positive
        complex oscillation convention. No absolute ppm calibration is inferred.
    phase_rad : float or numpy.ndarray, optional
        Initial complex phases in radians, broadcast with component parameters.
    dwell_time_s : float, optional
        Positive sampling interval in seconds when ``time_axis_s`` is omitted.
    npoints : int, optional
        Positive sample count used only when ``time_axis_s`` is omitted.
    noise_std : float, optional
        Nonnegative Gaussian noise standard deviation per real/imaginary
        quadrature in signal units. Independent noise is added to each component.
    sum_fids : bool, optional
        Sum the last parameter axis when True, preserving the legacy convention.
        False retains all broadcast parameter axes after the sample axis.

    Returns
    -------
    numpy.ndarray
        Complex128 samples with shape ``(sample_count, *parameter_shape)`` when
        ``sum_fids=False``. With True the final parameter axis is summed; scalar
        component parameters produce a one-dimensional FID. No coordinates or
        acquisition metadata are added, and inputs are not mutated.

    Raises
    ------
    ValueError
        If component shapes cannot broadcast or the noise scale is invalid.

    Notes
    -----
    Generation is eager and uncached. NumPy's global random generator is consumed
    even at zero noise scale. Noiseless output is deterministic; noisy output
    depends on that generator's state. Summed-component noise follows the same
    legacy summation as the signals and is not renormalized.

    Examples
    --------
    >>> from hypermri.analysis.spectroscopy import apply_lb, generate_fid
    >>> from hypermri.utils.time.temporal import calc_sampling_time_axis

    >>> dt = 5e-4
    >>> npoints=1000
    >>> time_axis_s = calc_sampling_time_axis(npoints=npoints, dwell_time_s=dt)

    >>> f = np.fft.fftshift(np.fft.fftfreq(npoints, d=dt))
    >>> lb_hz=20.0
    >>> fid = generate_fid(time_axis_s=time_axis_s, t2_s=0.1, freq_hz=100.0, phase_rad=0.5, dwell_time_s=dt, npoints=npoints)
    >>> spec_lb = apply_lb(values=fid, lb_hz=lb_hz, dwell_time_s=dt, signal_domain="temporal")

    >>> plt.figure()
    >>> plt.subplot(1,2,1)
    >>> plt.plot(time_axis_s, np.real(fid))
    >>> plt.plot(time_axis_s, np.imag(fid))
    >>> plt.plot(time_axis_s, np.abs(fid))
    >>> plt.xlabel('t [s]')

    >>> plt.subplot(1,2,2)
    >>> plt.plot(f, np.abs(np.fft.fftshift(np.fft.fft(fid)))/np.max(np.abs(np.fft.fftshift(np.fft.fft(fid)))),label='original FID')
    >>> plt.plot(f, np.abs(spec_lb)/np.max(np.abs(spec_lb)),label=f'linebroadened FID {lb_hz}Hz')
    >>> plt.xlabel('f [Hz]')
    >>> plt.legend()
    """
    if time_axis_s is None:
        from ...utils.time.temporal import calc_sampling_time_axis

        # calculate the time_axis in seconds:
        time_axis_s = calc_sampling_time_axis(
            npoints=npoints, dwell_time_s=dwell_time_s
        )
    else:
        pass
    npoints = len(time_axis_s)

    # Convert all parameters to numpy arrays, handling list and tuple inputs
    amplitude = np.atleast_1d(np.array(amplitude))
    t2_s = np.atleast_1d(np.array(t2_s))
    freq_hz = np.atleast_1d(np.array(freq_hz))
    phase_rad = np.atleast_1d(np.array(phase_rad))
    # Determine the target shape based on the most complex parameter
    target_shape = np.broadcast(amplitude, t2_s, freq_hz, phase_rad).shape
    # Broadcast parameters to the target shape
    amplitude = np.broadcast_to(amplitude, target_shape)
    t2_s = np.broadcast_to(t2_s, target_shape)
    freq_hz = np.broadcast_to(freq_hz, target_shape)
    phase_rad = np.broadcast_to(phase_rad, target_shape)

    # Initialize the output FID array with the temporal dimension first
    fid = np.zeros((npoints, *target_shape), dtype="complex")

    # Generate the FID signal for each set of parameters
    for index in np.ndindex(target_shape):
        idx = (slice(None),) + index
        exp_decay = np.exp(-time_axis_s / t2_s[index])
        oscillation, _ = propagate_isochromats(
            1.0, freq_hz[index], time_axis_s, phase_rad[index]
        )
        noise = np.random.normal(
            size=npoints,
            scale=noise_std,
        ) + 1j * np.random.normal(
            size=npoints,
            scale=noise_std,
        )
        fid[idx] = amplitude[index] * exp_decay * oscillation + noise

    if sum_fids:
        fid = np.sum(fid, axis=-1)

    return fid


def propagate_isochromats(amplitudes, frequencies_hz, time_s, phase_offset_rad=0.0):
    """Evolve independent spectral components using positive complex phases.

    Parameters
    ----------
    amplitudes : array_like
        Real or complex component weights in arbitrary signal units.
    frequencies_hz : array_like
        Frequencies relative to the carrier in Hz, broadcast with amplitudes.
    time_s : float or array_like
        Evolution time in seconds, broadcast with component arrays. To evaluate
        multiple times explicitly add a time axis, e.g. ``times[:, None]``.
    phase_offset_rad : float or array_like, optional
        Initial phase in radians, broadcast with frequencies and times.

    Returns
    -------
    vectors : numpy.ndarray
        Complex component signals with the broadcast shape. NumPy determines
        calculation precision from the inputs; no storage conversion occurs.
    phases_rad : numpy.ndarray
        Unwrapped phases in radians with the frequency/time broadcast shape.

    Raises
    ------
    ValueError
        If input shapes cannot broadcast.

    Notes
    -----
    This eager, deterministic calculation neither mutates inputs nor adds
    relaxation, noise, absolute frequency calibration or metadata.

    Examples
    --------
    >>> vectors, phases = propagate_isochromats([2., 3.], [0., 250.], .001)
    >>> np.allclose(vectors, [2., 3j])
    True
    """
    phases = 2 * np.pi * np.asarray(frequencies_hz) * time_s + phase_offset_rad
    return np.asarray(amplitudes) * np.exp(1j * phases), phases


def combine_isochromats(vectors):
    """Sum component vectors and return effective magnitude, phase and signal.

    Parameters
    ----------
    vectors : array_like
        Complex component signals, in arbitrary units. All axes are collapsed.

    Returns
    -------
    magnitude : numpy scalar
        Magnitude of the sum in input signal units.
    phase_rad : numpy scalar
        Principal phase in radians, in NumPy's interval ``[-pi, pi]``.
    signal : numpy scalar
        Complex sum in NumPy's accumulation precision.

    Notes
    -----
    Evaluation is eager, deterministic and does not mutate inputs.
    """
    signal = np.sum(vectors)
    return np.abs(signal), np.angle(signal), signal
