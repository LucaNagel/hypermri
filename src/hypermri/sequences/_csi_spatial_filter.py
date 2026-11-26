"""Translate acquired CSI layouts to the shared physical spatial filter.

This adapter reads the method name, slice and receiver counts from the handler
and the encoding grid, positions and field of view from
``handler.to_csi_acquisition()``. It never loads/reconstructs data, chooses
another method or changes acquired samples or handler caches.
"""

import numpy as np
import xarray as xr

from hypermri.analysis.spatial_filtering import (
    _spatial_filter_provenance,
    apply_spatial_filter,
    kspace_filter_weights,
)
from hypermri.reconstruction.csi_raw import (
    FOURIER_ENCODED_METHODS,
    SCANNER_SPATIAL_METHODS,
)


def _raw_spacing_mm(acquisition, explicit):
    if explicit is not None:
        values = np.asarray(explicit, dtype=float)
    else:
        if acquisition.fov_mm is None:
            raise ValueError(
                "Raw CSI spatial filtering requires fov_mm or explicit spatial_spacing_mm; use a calibrated labelled array."
            )
        values = np.asarray(acquisition.fov_mm) / np.asarray(acquisition.matrix)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(
            "Raw CSI spatial_spacing_mm must contain positive finite Read/Phase distances."
        )
    return tuple(values)


def _filter_raw_csi_spatial(data, sequence, options, spatial_spacing_mm):
    handler = sequence.handler
    method = (getattr(handler, "method", {}) or {}).get("Method")
    if method not in FOURIER_ENCODED_METHODS | SCANNER_SPATIAL_METHODS:
        raise ValueError(
            "Raw CSI spatial filtering requires a verified Fourier/scanner-grid method; supply calibrated labelled data instead."
        )
    acquisition = handler.to_csi_acquisition()
    shape = acquisition.matrix
    dx, dy = _raw_spacing_mm(acquisition, spatial_spacing_mm)
    read, phase = acquisition.read_positions, acquisition.phase_positions
    values = np.asarray(data)
    n_acquisition = int(sequence.n_acquisition)
    n_rep = int(getattr(sequence, "n_repetitions", 1))
    n_slice = int(getattr(handler, "n_slices", 1))
    n_channel = int(getattr(handler, "n_receivers", 1))
    expected = n_rep * n_slice * read.size * n_channel * n_acquisition
    if (
        values.ndim not in (1, 2)
        or values.size != expected
        or min(n_acquisition, n_rep, n_slice, n_channel) < 1
        or (values.ndim == 2 and values.shape[0] != n_acquisition)
    ):
        raise ValueError(
            "Raw CSI spatial filter input must match repetition/slice/point/channel/FID metadata; matrices use (FID sample, acquired point)."
        )
    raw = values if values.ndim == 1 else values.T.reshape(-1)
    blocks = raw.reshape(n_rep, n_slice, read.size, n_channel, n_acquisition)
    if method in FOURIER_ENCODED_METHODS:
        weights = kspace_filter_weights(
            (read - shape[0] // 2) / (shape[0] * dx),
            (phase - shape[1] // 2) / (shape[1] * dy),
            options,
        )
        filtered = blocks * weights[None, None, :, None, None]
    else:
        flat_indices = np.ravel_multi_index((read, phase), shape)
        if read.size != np.prod(shape) or np.unique(flat_indices).size != read.size:
            raise ValueError(
                "Scanner-spatial CSI filtering requires a complete unique grid; use a reconstructed labelled array."
            )
        grid = np.empty(
            (n_rep, n_slice, *shape, n_channel, n_acquisition), dtype=values.dtype
        )
        grid[:, :, read, phase, :, :] = blocks
        labelled = xr.DataArray(
            grid,
            dims=(
                "repetition_idx",
                "slice_idx",
                "read_idx",
                "phase_idx",
                "channel_idx",
                "acquisition_idx",
            ),
            attrs={"domain": "spatial", "csi_spatial_reconstruction": "scanner_grid"},
        )
        filtered_grid = apply_spatial_filter(
            labelled, options, spatial_spacing_mm=(dx, dy)
        )
        filtered = filtered_grid.values[:, :, read, phase, :, :]
    result = filtered.reshape(-1)
    return (
        result if values.ndim == 1 else result.reshape(-1, n_acquisition).T,
        _spatial_filter_provenance(
            options,
            "kspace" if method in FOURIER_ENCODED_METHODS else "spatial",
            (dx, dy),
            method in SCANNER_SPATIAL_METHODS,
        ),
    )
