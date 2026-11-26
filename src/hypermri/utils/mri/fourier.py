"""Spectral Fourier transforms with one explicit shift convention."""

import numpy as np

__all__ = ["fid_is_recoverable", "spectral_fft", "spectral_ifft"]


def spectral_fft(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Transform temporal samples into a centred, unnormalised spectrum.

    Parameters
    ----------
    values : numpy.ndarray
        Temporal samples in acquired signal units. Other axes are independent.
    axis : int, optional
        Sampling axis. Input samples start at time zero and are not shifted.

    Returns
    -------
    numpy.ndarray
        ``fftshift(fft(values))`` along the selected axis, with unchanged shape.
        Frequencies run from negative to positive; amplitude is the unnormalised
        NumPy FFT amplitude. NumPy owns calculation precision; inputs are not
        mutated and no coordinates, units or storage conversion are added.

    Raises
    ------
    IndexError
        If the axis is outside the input dimensions.
    ValueError
        If the sampling axis has zero length.

    Notes
    -----
    Evaluation is eager, deterministic and uncached. Spatial transforms with a
    centred input have a different convention and are deliberately separate.
    """
    return np.fft.fftshift(np.fft.fft(values, axis=axis), axes=axis)


def spectral_ifft(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Invert a centred spectrum into temporal samples starting at zero.

    Parameters
    ----------
    values : numpy.ndarray
        Spectrum in shifted FFT order. Other axes are independent.
    axis : int, optional
        Frequency axis to transform.

    Returns
    -------
    numpy.ndarray
        ``ifft(ifftshift(values))`` with the input shape. NumPy's inverse FFT
        divides amplitudes by the transform length and determines precision.
        Evaluation is eager and deterministic, without mutation or metadata.

    Raises
    ------
    IndexError
        If the axis is outside the input dimensions.
    ValueError
        If the frequency axis has zero length.
    """
    return np.fft.ifft(np.fft.ifftshift(values, axes=axis), axis=axis)


def fid_is_recoverable(data) -> bool:
    """Tell whether a time-domain signal can be computed from ``data``.

    Temporal data are FIDs already.  Spectral data give the FID through
    :func:`spectral_ifft` only when they are complex: a real spectrum, such
    as a ParaVision ``2dseq`` stored as ``MAGNITUDE_IMAGE``, has lost its
    phase, and its inverse transform is Hermitian, ``|s(t)| = |s(T - t)|``,
    which shows a false echo at the end of the acquisition window.

    Parameters
    ----------
    data : xarray.DataArray
        Signal whose attribute ``signal_domain`` is ``"temporal"`` (the
        default when absent) or ``"spectral"``.

    Returns
    -------
    bool
        False only for real-valued spectral data.  No data are copied or
        transformed.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> spectra = xr.DataArray(np.ones(4), attrs={"signal_domain": "spectral"})
    >>> fid_is_recoverable(spectra)
    False
    """
    domain = str(data.attrs.get("signal_domain", "temporal")).lower()
    return domain != "spectral" or np.issubdtype(data.dtype, np.complexfloating)
