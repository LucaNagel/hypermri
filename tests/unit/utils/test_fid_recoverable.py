"""The FID view needs complex spectra.

ParaVision stores spectroscopic ``2dseq`` data as ``MAGNITUDE_IMAGE``. Their
inverse transform is Hermitian and shows a false echo, so the FID view is
refused instead of drawn. Invented fixtures only.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.utils.mri.fourier import fid_is_recoverable
from hypermri.utils.plot.data_preprocessor import DataPreprocessor


def _signal(values, domain):
    return xr.DataArray(
        values, dims=("acquisition_idx",), attrs={"signal_domain": domain}
    )


def test_only_real_spectra_lose_the_fid():
    fid = np.exp(2j * np.pi * 0.1 * np.arange(16))
    assert fid_is_recoverable(_signal(fid, "temporal"))
    assert fid_is_recoverable(_signal(fid.real, "temporal"))
    assert fid_is_recoverable(_signal(np.fft.fft(fid), "spectral"))
    assert not fid_is_recoverable(_signal(np.abs(np.fft.fft(fid)), "spectral"))


def test_fid_view_of_magnitude_spectra_is_refused():
    spectra = _signal(np.ones(16, dtype=np.float32), "spectral")
    with pytest.raises(ValueError, match="FID view needs complex spectra"):
        DataPreprocessor().apply_domain_transform(
            spectra, is_fid=False, show_temporal=True, npoints=16
        )
    spectrum = DataPreprocessor().apply_domain_transform(
        spectra, is_fid=False, show_temporal=False, npoints=16
    )
    assert spectrum.sizes["acquisition_idx"] == 16
