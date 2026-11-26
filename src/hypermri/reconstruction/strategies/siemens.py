"""Siemens reconstruction strategies."""

from typing import ClassVar

import numpy as np
import xarray as xr

from .base import BaseRecoStrategy


class SiemensRawRecoStrategy(BaseRecoStrategy):
    """Reconstruction strategy for raw Siemens data."""

    accepted_options: ClassVar[frozenset[str]] = frozenset()

    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        """
        Perform the reconstruction for Siemens raw data.
        This is currently a placeholder and needs to be implemented.
        """
        print("SiemensRawRecoStrategy.reconstruct called")
        # Placeholder for Siemens reconstruction logic.
        # This will involve processing the data from handler.twix
        # and converting it into an xarray.DataArray.
        raise NotImplementedError(
            "Siemens raw data reconstruction is not yet implemented."
        )


class SiemensCsiRecoStrategy(BaseRecoStrategy):
    """Reconstruction strategy for Siemens CSI data."""

    def reconstruct(self, handler, reco_options: dict) -> xr.DataArray:
        # 1. For Siemens, the phase-encoding is standard. No reordering needed.
        k_space = handler

        # 2. Perform Fourier Transform
        image_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(k_space)))

        # 3. Create and return the xarray.DataArray
        # ... (xarray creation logic) ...

        # For demonstration, returning a placeholder
        return xr.DataArray(np.abs(image_data), dims=("z", "y", "x"), name="reco_image")


__all__ = ["SiemensRawRecoStrategy", "SiemensCsiRecoStrategy"]
