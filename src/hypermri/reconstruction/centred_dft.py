"""Centred discrete Fourier transforms between k-space and the image grid.

ParaVision places voxel ``j`` of an axis with ``n`` voxels at the position
``(j - n / 2) * spacing`` from the field-of-view centre: ``VisuCorePosition``
of a scan without offsets is the first voxel at ``-n / 2 * spacing``, and the
FOV centre lies at index ``n / 2``.  For even ``n`` this is a voxel centre;
for odd ``n`` it is the border between voxels ``(n - 1) / 2`` and
``(n + 1) / 2``.  :func:`centered_ifft` puts the FOV centre at index
``n / 2`` for both, so a reconstructed image lies on the grid of the
``VisuCore`` coordinates and of ``2dseq``.  For odd ``n`` this needs a
half-voxel shift, applied as a linear phase in k-space before the transform.

Evidence (22 November 2026, local study with three RARE scans and one CSI
scan): with the plain ``fftshift(ifft(ifftshift(k)))`` the raw CSI image
(35 read voxels) lies -0.5 voxel from ``2dseq``; the former raw transform of
``BrukerRawRecoStrategy`` put the RARE images (213 read and 133 phase voxels)
+0.5 voxel from ``2dseq``.  With the half-voxel shift each correlates with
``2dseq`` at zero shift at 1.0000 for RARE and 0.997 for CSI.

The slice-encoding axis of a 3D scan follows a second ParaVision rule: it
takes the forward FFT, and its FOV centre lies at index
``(n - 1) / 2``, the centre of the symmetric slice grid that
``VisuCorePosition`` describes (``RECO_rotate`` 0.475 for 20 slices).
:func:`slice_encoding_to_image` and :func:`image_to_slice_encoding` own this
axis.  Evidence (23 November 2026, two local 3D bSSFP scans with 20 slices):
the plain forward FFT puts the raw image half a slice from ``2dseq``
(correlation 0.62 and 0.53 at zero shift over 40 repetitions); with the
half-slice phase both correlate at 1.0000.  Odd slice counts take no
half-slice phase, since the plain transform already centres them at
``(n - 1) / 2``; no local scan has one, so this case is unverified.

These rules are the project standard for every spatial transform
("Spatial Fourier convention" in the simplification standards); new code
calls these functions instead of writing its own ``fftshift``/``fft`` steps.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = [
    "centered_fft",
    "centered_ifft",
    "image_to_slice_encoding",
    "slice_encoding_to_image",
]


def _half_voxel_ramp(n: int, *, inverse: bool) -> np.ndarray:
    """Return the k-space phase that moves an odd-length image by half a voxel."""
    k = np.arange(n) - n // 2
    sign = -1.0 if inverse else 1.0
    return np.exp(sign * 1j * np.pi * k / n)


def _along(values: np.ndarray, ramp: np.ndarray, axis: int) -> np.ndarray:
    shape = [1] * values.ndim
    shape[axis] = ramp.size
    return values * ramp.reshape(shape)


def centered_ifft(
    values: np.ndarray, axes: Iterable[int], *, norm: str = "backward"
) -> np.ndarray:
    """Transform centred k-space to the image grid of ``2dseq``.

    For each axis in ``axes``, one after another, computes
    ``fftshift(ifft(ifftshift(k)))``.  The k-space centre (zero frequency)
    is expected at index ``n // 2``.  The FOV centre of the image is at
    index ``n / 2``: a voxel centre for even ``n``, and for odd ``n`` the
    border between the two central voxels, reached by multiplying k-space by
    ``exp(-i pi k / n)`` before the transform (``k`` is the centred integer
    k-space index).  Even lengths take exactly the floating-point operations
    of the plain transform.

    Parameters
    ----------
    values : numpy.ndarray
        Complex k-space samples.  Other axes are left untouched.
    axes : iterable of int
        Spatial axes to transform.
    norm : {"backward", "forward", "ortho"}, optional
        NumPy normalization of the inverse FFT.  ``"backward"`` scales by
        ``1 / n`` per axis; ``"forward"`` keeps the scale of the forward FFT,
        as the Bruker raw reconstruction does.

    Returns
    -------
    numpy.ndarray
        New complex array with the shape of ``values``.  The input is not
        modified.

    Examples
    --------
    An object at the FOV centre has constant k-space.  For an odd length it
    lies between the two central voxels, with equal magnitude in both:

    >>> import numpy as np
    >>> image = np.abs(centered_ifft(np.ones(5, dtype=complex), (0,)))
    >>> bool(np.isclose(image[2], image[3]))
    True
    """
    for axis in axes:
        n = values.shape[axis]
        if n % 2:
            values = _along(values, _half_voxel_ramp(n, inverse=True), axis)
        values = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(values, axes=axis), axis=axis, norm=norm),
            axes=axis,
        )
    return values


def centered_fft(
    values: np.ndarray, axes: Iterable[int], *, norm: str = "backward"
) -> np.ndarray:
    """Transform the image grid of ``2dseq`` back to centred k-space.

    The exact inverse of :func:`centered_ifft` with the same ``norm``: for
    each axis computes ``fftshift(fft(ifftshift(values)))`` and, for an odd
    length, removes the half-voxel phase.  A sample acquired at ``+k``
    returns at ``+k``, with zero frequency at index ``n // 2``.

    Parameters
    ----------
    values : numpy.ndarray
        Complex image samples.  Other axes are left untouched.
    axes : iterable of int
        Spatial axes to transform.
    norm : {"backward", "forward", "ortho"}, optional
        NumPy normalization of the forward FFT; use the value given to
        :func:`centered_ifft`.

    Returns
    -------
    numpy.ndarray
        New complex k-space array with the shape of ``values``.  The input is
        not modified.

    Examples
    --------
    >>> import numpy as np
    >>> kspace = np.arange(5, dtype=complex)
    >>> np.allclose(centered_fft(centered_ifft(kspace, (0,)), (0,)), kspace)
    True
    """
    for axis in axes:
        n = values.shape[axis]
        values = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(values, axes=axis), axis=axis, norm=norm),
            axes=axis,
        )
        if n % 2:
            values = _along(values, _half_voxel_ramp(n, inverse=False), axis)
    return values


def slice_encoding_to_image(
    values: np.ndarray, axes: Iterable[int], *, norm: str = "backward"
) -> np.ndarray:
    """Transform the slice-encoding axis of a 3D scan to the ``2dseq`` slices.

    For each axis in ``axes`` computes ``fftshift(fft(ifftshift(k)))``, the
    forward FFT that ParaVision applies along the 3D slice-encoding axis, so
    that slice ``j`` of the raw image is slice ``j`` of ``2dseq`` without a
    reversal.  The k-space centre is expected at index ``n // 2``.  The FOV
    centre of the image is at index ``(n - 1) / 2``: for even ``n`` k-space
    is first multiplied by ``exp(-i pi k / n)``, which moves the image by
    half a slice towards lower indices; odd ``n`` needs no phase.

    Parameters
    ----------
    values : numpy.ndarray
        Complex k-space samples.  Other axes are left untouched.
    axes : iterable of int
        Slice-encoding axes to transform; in practice the one ``slice_idx``
        axis of a 3D acquisition.
    norm : {"backward", "forward", "ortho"}, optional
        NumPy normalization of the forward FFT.  ``"backward"``, the
        default, does not scale, as the Bruker raw reconstruction expects.

    Returns
    -------
    numpy.ndarray
        New complex array with the shape of ``values``.  The input is not
        modified.

    Examples
    --------
    An object at the FOV centre has constant k-space.  For an even number of
    slices it lies between the two central slices:

    >>> import numpy as np
    >>> image = np.abs(slice_encoding_to_image(np.ones(4, dtype=complex), (0,)))
    >>> bool(np.isclose(image[1], image[2]))
    True
    """
    for axis in axes:
        n = values.shape[axis]
        if n % 2 == 0:
            values = _along(values, _half_voxel_ramp(n, inverse=True), axis)
        values = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(values, axes=axis), axis=axis, norm=norm),
            axes=axis,
        )
    return values


def image_to_slice_encoding(
    values: np.ndarray, axes: Iterable[int], *, norm: str = "backward"
) -> np.ndarray:
    """Transform the ``2dseq`` slices of a 3D scan back to slice encoding.

    The exact inverse of :func:`slice_encoding_to_image` with the same
    ``norm``: for each axis computes ``fftshift(ifft(ifftshift(values)))``
    and, for an even length, removes the half-slice phase.

    Parameters
    ----------
    values : numpy.ndarray
        Complex image samples.  Other axes are left untouched.
    axes : iterable of int
        Slice axes to transform.
    norm : {"backward", "forward", "ortho"}, optional
        NumPy normalization of the inverse FFT; use the value given to
        :func:`slice_encoding_to_image`.

    Returns
    -------
    numpy.ndarray
        New complex k-space array with the shape of ``values``, zero
        frequency at index ``n // 2``.  The input is not modified.

    Examples
    --------
    >>> import numpy as np
    >>> kspace = np.arange(4, dtype=complex)
    >>> image = slice_encoding_to_image(kspace, (0,))
    >>> np.allclose(image_to_slice_encoding(image, (0,)), kspace)
    True
    """
    for axis in axes:
        n = values.shape[axis]
        values = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(values, axes=axis), axis=axis, norm=norm),
            axes=axis,
        )
        if n % 2 == 0:
            values = _along(values, _half_voxel_ramp(n, inverse=False), axis)
    return values
