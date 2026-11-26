from hypermri.loading.protocols import VendorHandlerProtocol

from .base_spectroscopy import BaseSpectroscopy


class gNSPECT_MS(BaseSpectroscopy):
    """Generalized multi-slice NMR spectroscopy sequence wrapper."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
