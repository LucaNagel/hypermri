"""Single-slice spectroscopy sequence specialised for Geoff studies."""

from hypermri.loading.protocols import VendorHandlerProtocol

from .base_spectroscopy import BaseSpectroscopy


class GeoffSlcSpec(BaseSpectroscopy):
    """Single-slice spectroscopy variant used in the Geoff protocol."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
