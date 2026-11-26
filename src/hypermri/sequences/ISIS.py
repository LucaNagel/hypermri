"""ISIS spectroscopy sequence implementation."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class ISIS(BaseSpectroscopy):
    """Image-selected in vivo spectroscopy sequence."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the ISIS sequence wrapper and reconstruct base datasets."""
        super().__init__(handler, **kwargs)
