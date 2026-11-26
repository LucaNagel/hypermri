"""Spectral normalization, apodization and temporal zero filling.

These operations preserve the established FFT convention and optional plotting.
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from hypermri.utils.mri.fourier import spectral_fft, spectral_ifft
from hypermri.utils.mri.signal import zero_fill_temporal

__all__ = [
    "norm",
    "conv_func",
    "multi_dim_linebroadening",
    "multi_dim_zeropadding",
    "apply_lb",
    "zeropad_data",
]


def norm(x, a=0, b=1):
    """Rescale array values linearly to the interval [a, b].

    Parameters
    ----------
    x : numpy.ndarray
        Input values in arbitrary units and shape.
    a, b : float, optional
        Lower and upper output bounds; a must be smaller than b.

    Returns
    -------
    numpy.ndarray
        Rescaled values with input shape and NumPy arithmetic precision.
        Constant inputs retain the legacy division-by-zero/NaN behavior.
        Inputs are not mutated; evaluation is eager and deterministic.

    Raises
    ------
    AssertionError
        If a is not smaller than b.
    """
    assert a < b

    min_x = np.min(x)
    max_x = np.max(x)

    normed_x_01 = (x - min_x) / (max_x - min_x)  # normed array from 0 to 1
    normed_x_custom = (b - a) * normed_x_01 + a  # normed array within [a,b]

    return normed_x_custom


def conv_func(
    name: str = None, lb_hz: float = 0.0, time_s: np.ndarray = None
) -> np.ndarray:
    """Evaluate a legacy spectroscopy apodization window.

    Parameters
    ----------
    name : str or None, optional
        "Lorentzian" uses exp(-lb_hz * time_s). "Gaussian", None and
        unrecognized names use exp(-(pi * lb_hz * time_s)^2 / (2 * log(2))).
        These retained width conventions differ from vendor reconstruction windows.
    lb_hz : float, optional
        Legacy broadening parameter in Hz; zero produces a unit window.
    time_s : numpy.ndarray
        Sample times in seconds, in the desired output order.

    Returns
    -------
    numpy.ndarray
        Real window with the time-axis shape and NumPy calculation precision.
        Evaluation is eager, deterministic and does not mutate input times.
    """
    if name == "Gaussian":
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))

    elif name == "Lorentzian":
        filter = np.exp(-(lb_hz * time_s))

    elif name is None:
        # default Gaussian:
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))
    else:
        # default Gaussian:
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))
    return filter


def multi_dim_linebroadening(
    data: Any = None,
    lb_hz: float = 0.0,
    fid_start: int = 0,
    fid_end: int = 0,
    dwell_time_s: float = 0,
    conv_func_name: str = "Gaussian",
    signal_domain: str = "spectral",
    axis: int = 0,
    scan: Any = None,
) -> np.ndarray | None:
    """Apply legacy line broadening independently along a chosen sample axis.

    Parameters
    ----------
    data : numpy.ndarray or None, optional
        Signal array in arbitrary units and arbitrary dimension order. None
        resolves scan.data once. No labelled coordinates are propagated.
    lb_hz : float, optional
        Broadening parameter in Hz under conv_func's retained width convention.
    fid_start, fid_end : int, optional
        Start and exclusive end indices of the temporal segment; zero end uses
        the full length. Use the full segment for a window of matching length.
    dwell_time_s : float, optional
        Sampling interval in seconds. Zero resolves the data object's dwell time.
    conv_func_name : str, optional
        Window name passed to conv_func, default "Gaussian".
    signal_domain : str, optional
        "spectral" applies an inverse FFT first and returns a spectrum;
        "temporal" processes and returns the FID.
    axis : int, optional
        Sampling axis. All other axes are processed independently.
    scan : object or None, optional
        Source of missing samples or dwell time; acquired samples are not mutated.

    Returns
    -------
    numpy.ndarray or None
        Processed signals with NumPy FFT/arithmetic precision; None if no dwell
        time or data object is supplied. Evaluation is eager and uncached.

    Raises
    ------
    Exception
        If input samples cannot be resolved or a temporal segment is invalid.
    """
    if data is None:
        try:
            data = scan.data
        except BaseException:
            raise Exception("provide either data or scan")

    else:
        pass

    if dwell_time_s == 0 and scan is not None:  # define sampling duration to do: check
        from ...utils.time.temporal import get_dwell_time_s

        dwell_time_s = get_dwell_time_s(scan=scan)
    elif dwell_time_s == 0 and scan is None:
        return None
    else:
        pass

    def wrapper_apply_lb(arr):
        return apply_lb(
            values=arr,
            lb_hz=lb_hz,
            fid_start=fid_start,
            fid_end=fid_end,
            dwell_time_s=dwell_time_s,
            conv_func_name=conv_func_name,
            signal_domain=signal_domain,
            scan=scan,
        )

    # Apply along the first axis (axis=0)
    linebroadend_data = np.apply_along_axis(wrapper_apply_lb, axis=axis, arr=data)

    return linebroadend_data


def multi_dim_zeropadding(
    values: np.ndarray = None,
    num_zp: int = 0,
    fid_start: int = 0,
    fid_end: int = 0,
    signal_domain: str = "spectral",
    scan: Any = None,
):
    """Zero-fill independent FIDs along the first array axis.

    Parameters
    ----------
    values : numpy.ndarray or None, optional
        Samples ordered sampling axis first, then any independent dimensions.
        None resolves scan.seq2d_reordered, then scan.spec.
    num_zp : int, optional
        Nonnegative number of zero samples appended to each temporal segment.
    fid_start, fid_end : int, optional
        Start and exclusive end indices; zero end uses the complete FID.
    signal_domain : str, optional
        "spectral" or "temporal"; output remains in the chosen domain.
    scan : object or None, optional
        Legacy sample source when values is omitted.

    Returns
    -------
    numpy.ndarray
        Array with a resized first axis and unchanged independent axes. Temporal
        data preserve dtype; spectral data use NumPy FFT precision. No coordinates
        are inferred. Evaluation is eager, deterministic and does not mutate inputs.

    Raises
    ------
    Exception
        If samples cannot be resolved or a temporal segment is invalid.
    ValueError
        If the zero-padding count is invalid.
    """
    if values is None:
        try:
            values = scan.seq2d_reordered
        except BaseException:
            try:
                values = scan.spec
            except BaseException:
                raise Exception("provide either values or spec_data")
    else:
        pass

    def wrapper_zeropad_data(arr):
        return zeropad_data(
            values=arr,
            num_zp=num_zp,
            fid_start=fid_start,
            fid_end=fid_end,
            signal_domain=signal_domain,
        )

    # Apply along the first axis (axis=0)
    zeropadded_data = np.apply_along_axis(wrapper_zeropad_data, axis=0, arr=values)
    return zeropadded_data


def apply_lb(
    values: np.array = None,
    lb_hz: float = 0.0,
    fid_start: int = 0,
    fid_end: int = 0,
    dwell_time_s: float = 0.0,
    conv_func_name: str = "Gaussian",
    signal_domain: str = None,
    scan: Any = None,
    plot: bool = False,
) -> np.ndarray | None:
    """Apply a legacy apodization window to one temporal signal.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional complex FID or centred spectrum in arbitrary signal units.
        Singleton axes are squeezed before processing.
    lb_hz : float, optional
        Broadening parameter in Hz under conv_func's retained width convention.
    fid_start, fid_end : int, optional
        Start and exclusive end of the FID segment. Zero end means full length;
        the full segment is required for the full-length window to broadcast.
    dwell_time_s : float, optional
        Sampling interval in seconds. Zero reads it from scan when provided.
    conv_func_name : str, optional
        "Gaussian" or "Lorentzian", under conv_func's documented conventions.
    signal_domain : str or None, optional
        "temporal" returns an apodized FID; "spectral" returns a centred
        spectrum. Other values retain the legacy spectral output and warning.
    scan : object or None, optional
        Source of missing dwell time; does not supply missing values.
    plot : bool, optional
        Plot the window and processed samples when True.

    Returns
    -------
    numpy.ndarray or None
        Apodized signal with NumPy arithmetic/FFT precision. None if dwell time
        cannot be obtained. No coordinates are added, and input is not mutated.

    Raises
    ------
    Exception
        If values is missing, not one-dimensional, or the segment is invalid.
    ValueError
        If the selected signal and window cannot broadcast.

    Notes
    -----
    Evaluation is eager, deterministic and uncached. Spectral processing uses
    ifft(ifftshift(input)) followed by fftshift(fft(windowed_signal)).

    Examples
    --------
    >>> fid = np.ones(4, dtype=complex)
    >>> apply_lb(fid, lb_hz=0, dwell_time_s=.001, signal_domain="temporal").shape
    (4,)
    """
    # check if data was input:
    if values is None:
        raise Exception("No values")
    else:
        pass

    values = np.atleast_1d(np.squeeze(values))

    # apply 1D Fourier-Transform
    if signal_domain == "temporal":
        pass
    elif signal_domain == "spectral":
        values = spectral_ifft(values, axis=0)

    # check dimensions of input data:
    if values.ndim == 1:
        if fid_end == 0:
            fid_end = values.shape[0]
        elif fid_end > values.shape[0]:
            fid_end = values.shape[0]
        elif fid_end < fid_start:
            raise Exception("fid_end has to be higher than fid_start")
        else:
            pass
    else:
        raise Exception("values has to have dim=1")

    if dwell_time_s == 0 and scan is not None:  # define sampling duration to do: check
        from ...utils.time.temporal import get_dwell_time_s

        dwell_time_s = get_dwell_time_s(
            scan=scan
        )  # scan.method["PVM_SpecDwellTime"] * 2.0 / 1000.0 / 1000.0
    elif dwell_time_s == 0 and scan is None:
        return None
    else:
        pass

    # get time axis in seconds
    if dwell_time_s == 0:
        raise Exception("sampling duration has to be > 0")
    else:
        from ...utils.time.temporal import calc_sampling_time_axis

        time_ax = calc_sampling_time_axis(
            npoints=len(values), dwell_time_s=dwell_time_s
        )

    # define filter:
    filter = conv_func(name=conv_func_name, lb_hz=lb_hz, time_s=time_ax)

    fid_times_dec_exp = values[fid_start:fid_end] * filter
    if signal_domain == "temporal":
        return_lb = fid_times_dec_exp
    elif signal_domain == "spectral":
        return_lb = spectral_fft(fid_times_dec_exp, axis=0)
    else:
        return_lb = spectral_fft(fid_times_dec_exp, axis=0)
        warnings.warn(
            "signal_domain not recognized - returning spectrum as output - use signal_domain='spectral' or 'temporal'",
            stacklevel=2,
        )
    if plot:
        _, ax = plt.subplots(1, 1)
        ax.plot(time_ax, np.real(values), label="original FID")
        ax.plot(time_ax, np.real(fid_times_dec_exp), label="linebroadened FID")
        ax.set_xlabel("t [s]")
        ax.legend()

    return return_lb


def zeropad_data(
    values: np.ndarray = None,
    num_zp: int = 0,
    fid_start: int = 0,
    fid_end: int = 0,
    signal_domain: str = "spectral",
):
    """Append temporal zeros to one FID or centred spectrum.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional FID or spectrum in arbitrary signal units.
    num_zp : int, optional
        Nonnegative number of zeros appended after selecting the temporal segment.
    fid_start, fid_end : int, optional
        Start and exclusive end indices. Zero end uses the full FID; an end
        beyond the signal is clipped to its length.
    signal_domain : str, optional
        "temporal" or "spectral". Spectral samples are transformed to time
        before padding and back to the centred spectrum afterwards.

    Returns
    -------
    numpy.ndarray
        Processed temporal segment plus num_zp samples. Temporal dtype is
        preserved; spectra follow NumPy FFT precision. Units remain unchanged.
        No frequency/time coordinates are inferred. Inputs are not mutated.

    Raises
    ------
    Exception
        If samples are missing, not one-dimensional, or the segment is invalid.
    ValueError
        If num_zp is negative or not an integer.

    Notes
    -----
    Evaluation is eager, deterministic and uncached. Zero filling interpolates
    the spectrum; it does not increase acquired spectral resolution.
    """
    # check if data was input:
    if values is None:
        raise Exception("No values")
    else:
        pass

    # apply 1D Fourier-Transform
    if signal_domain == "temporal":
        pass
    elif signal_domain == "spectral":
        values = spectral_ifft(values, axis=0)

    # check dimensions of input data:
    if values.ndim == 1:
        if fid_end == 0:
            fid_end = values.shape[0]
        elif fid_end > values.shape[0]:
            fid_end = values.shape[0]
        elif fid_end < fid_start:
            raise Exception("fid_end has to be higher than fid_start")
        else:
            pass
        fid = values[fid_start:fid_end]
    else:
        raise Exception("values has to have dim=1")

    # append zeros to the FID:
    fid_zp = zero_fill_temporal(fid, num_zp, axis=0)
    if signal_domain == "temporal":
        return fid_zp
    elif signal_domain == "spectral":
        # apply 1D Fourier-Transform
        spec_zp = spectral_fft(fid_zp, axis=0)
        return spec_zp
