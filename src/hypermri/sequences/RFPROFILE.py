"""Provide RF profile sequence utilities with package-controlled logging."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import LOG_MODES as LOG_MODES
from ..utils.system.logging import init_default_logger
from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__)


class RFPROFILE(BaseSpectroscopy):
    """RF Profile and review helpers."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
