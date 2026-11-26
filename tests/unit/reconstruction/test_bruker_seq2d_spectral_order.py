"""Frequency order of spectroscopic Bruker ``2dseq`` data."""

import numpy as np
import xarray as xr

from hypermri.reconstruction.strategies.bruker.seq2d import (
    _spectra_in_ascending_frequency,
)


def _centred_spectrum(fid):
    return np.fft.fftshift(np.fft.fft(fid))


def test_descending_2dseq_spectrum_is_put_in_ascending_frequency_order():
    """Regression test: scanner CSI spectra appeared mirrored about the centre.

    A local Bruker CSI ``2dseq`` held the raw spectrum at ``-f``: pyruvate
    and lactate lay at 139 and 127 ppm instead of 171 and 183 ppm.  The
    synthetic FID has one line at +5 frequency samples; ParaVision's
    descending order puts it at -5.
    """
    size = 16
    fid = np.exp(2j * np.pi * 5 * np.arange(size) / size)
    ascending = np.abs(_centred_spectrum(fid))
    descending = np.abs(_centred_spectrum(np.conj(fid)))
    stored = xr.DataArray(
        descending[:, None],
        dims=["acquisition_idx", "read_idx"],
        coords={"acquisition_idx": np.arange(size)},
        attrs={"signal_domain": "spectral"},
    )
    original = stored.copy(deep=True)

    result = _spectra_in_ascending_frequency(stored)

    np.testing.assert_allclose(result.values[:, 0], ascending, rtol=0, atol=1e-12)
    assert int(np.argmax(result.values[:, 0])) == size // 2 + 5
    xr.testing.assert_identical(stored, original)
    assert result.attrs == stored.attrs
    np.testing.assert_array_equal(result.acquisition_idx, stored.acquisition_idx)


def test_reordering_twice_restores_the_stored_order():
    stored = xr.DataArray(np.arange(7.0), dims=["acquisition_idx"])
    twice = _spectra_in_ascending_frequency(_spectra_in_ascending_frequency(stored))
    np.testing.assert_array_equal(twice.values, stored.values)
