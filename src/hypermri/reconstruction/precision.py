"""Floating-point precision of reconstruction inputs and stored results.

A reconstruction option ``data_precision`` names one of three real floating
types: ``float16``, ``float32`` or ``float64``.  This module is the one place
that reads that option and converts arrays to it.  It distinguishes two uses:

* **Calculation precision** (:func:`convert_for_calculation`) is applied to
  the acquired FID before the reconstruction computes with it.  A complex
  FID is only reduced for ``float16``, to ``complex64``; ``float32`` and
  ``float64`` keep its dtype, so the Fourier transform runs at the precision
  in which the FID was read.
* **Storage precision** (:func:`convert_for_storage`) is applied to the
  arrays a reconstruction returns, such as k-space and images.  Complex
  results become ``complex64`` for ``float16`` and ``float32`` and
  ``complex128`` for ``float64``, because NumPy has no complex type built
  from ``float16``.

Real arrays are converted to the requested type in both cases.  The type of
an array (real or complex) is decided by its dtype, never by its values, so a
complex array whose imaginary parts are all zero stays complex.

Examples
--------
>>> import numpy as np
>>> from hypermri.reconstruction.precision import convert_for_storage, parse_data_precision
>>> parse_data_precision("Float32")
<class 'numpy.float32'>
>>> convert_for_storage(np.array([1 + 0j]), "float32").dtype
dtype('complex64')
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hypermri.utils.system.logging import init_default_logger

logger = init_default_logger(__name__)

PRECISIONS = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
"""Accepted values of the ``data_precision`` option and their NumPy types."""

_COMPLEX_STORAGE_TYPE = {
    np.float16: np.complex64,
    np.float32: np.complex64,
    np.float64: np.complex128,
}


def parse_data_precision(value: Any) -> type:
    """Return the NumPy type named by a ``data_precision`` option.

    Parameters
    ----------
    value : str, numpy type or numpy.dtype
        ``"float16"``, ``"float32"`` or ``"float64"`` in any letter case, or
        the NumPy type or dtype ``float16``, ``float32`` or ``float64``.

    Returns
    -------
    builtins.type
        ``numpy.float16``, ``numpy.float32`` or ``numpy.float64``.  The same
        object is returned for every spelling of one precision, so results
        can be compared with ``is``.

    Raises
    ------
    ValueError
        If ``value`` names any other type, including ``None``, Python
        ``float`` and complex types.  The message lists the accepted values.

    Examples
    --------
    >>> import numpy as np
    >>> parse_data_precision(np.dtype("float64")) is np.float64
    True
    """
    if isinstance(value, str):
        precision = PRECISIONS.get(value.strip().lower())
    elif isinstance(value, (type, np.dtype)) and value in PRECISIONS.values():
        precision = np.dtype(value).type
    else:
        precision = None
    if precision is None:
        raise ValueError(
            f"Unsupported data_precision {value!r}; use one of "
            f"{list(PRECISIONS)} or the NumPy type of the same name."
        )
    return precision


def convert_for_calculation(
    array: Optional[np.ndarray], precision: Any
) -> Optional[np.ndarray]:
    """Convert acquired data to the precision a reconstruction computes in.

    Parameters
    ----------
    array : numpy.ndarray or None
        Acquired data of any shape, usually the complex FID.  It is not
        modified.
    precision : str, numpy type or numpy.dtype
        Requested precision; see :func:`parse_data_precision`.

    Returns
    -------
    numpy.ndarray or None
        ``None`` for ``None`` and the input itself for an empty array.  A
        complex array becomes a ``complex64`` copy for ``float16`` and is
        returned unchanged, without a copy, for ``float32`` and
        ``float64``.  A real array becomes a copy of the requested type.

    Raises
    ------
    ValueError
        If ``precision`` is not supported.

    Notes
    -----
    NaN or infinite values are logged as a warning and kept.  A complex
    array whose magnitude exceeds the ``float32`` range is logged as a
    warning before it is reduced to ``complex64``.
    """
    precision = parse_data_precision(precision)
    if array is None or array.size == 0:
        return array

    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        logger.warning("The array to reconstruct contains NaN or infinite values.")

    if not np.iscomplexobj(array):
        return array.astype(precision)
    if precision is not np.float16:
        return array
    largest = np.max(np.abs(array))
    if largest > np.finfo(np.float32).max:
        logger.warning(
            f"Values up to {largest} exceed the range of complex64 and overflow."
        )
    return array.astype(np.complex64)


def convert_for_storage(
    array: Optional[np.ndarray], precision: Any
) -> Optional[np.ndarray]:
    """Convert a reconstruction result to the precision it is stored in.

    Parameters
    ----------
    array : numpy.ndarray or None
        Result of any shape, such as k-space, an image or a ``2dseq``
        array.  It is not modified.
    precision : str, numpy type or numpy.dtype
        Requested precision; see :func:`parse_data_precision`.

    Returns
    -------
    numpy.ndarray or None
        ``None`` for ``None`` and the input itself for an empty array.
        Otherwise a copy: complex data as ``complex64`` for ``float16`` and
        ``float32`` and as ``complex128`` for ``float64``, real data as the
        requested type.

    Raises
    ------
    ValueError
        If ``precision`` is not supported.
    """
    precision = parse_data_precision(precision)
    if array is None or array.size == 0:
        return array
    if np.iscomplexobj(array):
        return array.astype(_COMPLEX_STORAGE_TYPE[precision])
    return array.astype(precision)


def convert_complex_for_storage(
    array: np.ndarray, precision: Any = "float32"
) -> np.ndarray:
    """Store explicitly complex samples using the shared precision mapping.

    Parameters
    ----------
    array : numpy.ndarray
        Real or complex samples of any shape, interpreted as complex signal.
        Real inputs acquire a zero imaginary component. Input is not mutated.
    precision : str, numpy type or numpy.dtype, optional
        Requested real precision; see :func:`parse_data_precision`. Defaults
        to float32, retaining the existing precision of vendor FID readers.

    Returns
    -------
    numpy.ndarray
        Same shape as input, complex64 for float16/float32 or complex128 for
        float64. Returns the same array when its dtype already matches;
        otherwise allocates a copy. Conversion is eager and deterministic.

    Raises
    ------
    ValueError
        If the precision is unsupported.
    """
    return array.astype(
        _COMPLEX_STORAGE_TYPE[parse_data_precision(precision)], copy=False
    )


__all__ = [
    "PRECISIONS",
    "convert_for_calculation",
    "convert_complex_for_storage",
    "convert_for_storage",
    "parse_data_precision",
]
