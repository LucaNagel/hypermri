"""B1 calibration sequence wrapper for acquiring power-response curves."""

from hypermri.loading.protocols import VendorHandlerProtocol

from .base_spectroscopy import BaseSpectroscopy


class B1Calibration(BaseSpectroscopy):
    """Single-voxel spectroscopy sequence used to calibrate B1 flip angles."""

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("seq2d",),
        "full": ("seq2d", "raw"),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Accepts directory path or BrukerExp object as input."""
        super().__init__(handler, **kwargs)
