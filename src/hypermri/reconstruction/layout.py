"""Common dimension layout of the imaging reconstructions of one scan.

The raw reconstruction and the reconstruction of the scanner images (for
example the Bruker ``2dseq`` file) of one imaging scan share the dimensions
and the order of :data:`IMAGING_DIM_ORDER`, so that code and users can address
both alike.  :func:`to_imaging_layout` is the one owner of this layout, and
:meth:`hypermri.loading.experiment.MRIExperiment.reconstruct` applies it to the
raw and scanner reconstructions of a non-spectroscopic scan before storing
them. Direct Bruker scanner loading applies the same idempotent operation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["IMAGING_DIM_ORDER", "to_imaging_layout"]

IMAGING_DIM_ORDER = (
    "read_idx",
    "phase_idx",
    "slice_idx",
    "waterfat_idx",
    "inversion_idx",
    "echo_idx",
    "bvalue_idx",
    "channel_idx",
    "acquisition_idx",
    "repetition_idx",
)


def _assign_time_coordinate(data, name, dimension, values):
    """Label a complete time axis in seconds without replacing declared labels."""
    if name in data.coords or dimension not in data.dims or values is None:
        return data
    times = np.asarray(values)
    size = data.sizes[dimension]
    if times.ndim != 1 or times.size < size:
        return data
    return data.assign_coords(
        {
            name: (
                dimension,
                times[:size],
                {
                    "unit": "s",
                    "long_name": (
                        "Frame acquisition time"
                        if name == "time"
                        else "Acquisition time"
                    ),
                    "description": (
                        "Time in seconds for each repetition/frame"
                        if name == "time"
                        else "Time in seconds for each acquisition point"
                    ),
                },
            )
        }
    )


def to_imaging_layout(
    data: xr.DataArray,
    *,
    sampling_time_s: np.ndarray | None = None,
    frame_time_s: np.ndarray | None = None,
) -> xr.DataArray:
    """Return ``data`` with the dimensions of :data:`IMAGING_DIM_ORDER`.

    Each dimension of :data:`IMAGING_DIM_ORDER` that ``data`` lacks is added
    with length 1 and the index coordinate ``[0]``.  The dimensions are then
    transposed to that order; dimensions outside the order keep their
    relative order and follow it. Existing nonspatial dimensions without
    coordinates receive integer index labels. Optional complete time axes label
    acquired points and repetitions in seconds. Existing coordinates take
    precedence; values and attributes are unchanged. The input is not modified.

    Parameters
    ----------
    data : xarray.DataArray
        Image-space reconstruction of an imaging scan with the spatial
        dimensions ``read_idx``, ``phase_idx`` and ``slice_idx``, in any
        order and with any further dimensions.
    sampling_time_s : numpy.ndarray, optional
        One-dimensional acquisition times in seconds. Used only when its length
        equals ``acquisition_idx``; readout times are not assigned to differently
        sized processed or echo axes. An existing ``acquisition_time_s`` wins.
    frame_time_s : numpy.ndarray, optional
        One-dimensional repetition times in seconds. A complete array labels
        ``time`` along ``repetition_idx``; surplus entries are ignored. Missing
        or short arrays do not create time coordinates. Existing ``time`` wins.

    Returns
    -------
    xarray.DataArray
        Array of the same values whose leading dimensions are
        ``read_idx``, ``phase_idx``, ``slice_idx``, ``waterfat_idx``,
        ``inversion_idx``, ``echo_idx``, ``bvalue_idx``, ``channel_idx``,
        ``acquisition_idx`` and ``repetition_idx``.  The values are a view of
        the input where xarray can provide one.

    Raises
    ------
    ValueError
        If ``data`` lacks one of the spatial dimensions ``read_idx``,
        ``phase_idx`` or ``slice_idx``.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> image = xr.DataArray(
    ...     np.zeros((2, 3, 4)), dims=("slice_idx", "read_idx", "phase_idx")
    ... )
    >>> to_imaging_layout(image).dims[:4]
    ('read_idx', 'phase_idx', 'slice_idx', 'waterfat_idx')
    """
    missing = [dim for dim in IMAGING_DIM_ORDER[:3] if dim not in data.dims]
    if missing:
        raise ValueError(
            "to_imaging_layout needs the dimensions read_idx, phase_idx and "
            f"slice_idx; {missing} missing from {data.dims}."
        )
    padded = data
    for dim in IMAGING_DIM_ORDER[3:]:
        if dim not in padded.dims:
            padded = padded.expand_dims({dim: np.arange(1)})
        elif dim not in padded.coords:
            padded = padded.assign_coords({dim: (dim, np.arange(padded.sizes[dim]))})
    others = [dim for dim in data.dims if dim not in IMAGING_DIM_ORDER]
    sampling = None if sampling_time_s is None else np.asarray(sampling_time_s)
    if sampling is not None and sampling.ndim == 1:
        if sampling.size == padded.sizes["acquisition_idx"]:
            padded = _assign_time_coordinate(
                padded, "acquisition_time_s", "acquisition_idx", sampling
            )
    padded = _assign_time_coordinate(padded, "time", "repetition_idx", frame_time_s)
    return padded.transpose(*IMAGING_DIM_ORDER, *others)
