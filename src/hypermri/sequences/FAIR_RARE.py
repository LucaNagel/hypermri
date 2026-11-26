"""FAIR-RARE perfusion imaging sequence implementation."""

import traceback

from hypermri.loading.protocols import VendorHandlerProtocol

from .base_anatomy import BaseAnatomy


class FAIR_RARE(BaseAnatomy):
    """Flow-sensitive alternating inversion recovery combined with RARE readout."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the FAIR-RARE wrapper and run the default reconstructions."""
        super().__init__(handler, **kwargs)
