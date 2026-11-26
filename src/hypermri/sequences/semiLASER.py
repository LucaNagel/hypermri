"""Provide semi-LASER spectroscopy with package-controlled logging."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import LOG_MODES as LOG_MODES
from ..utils.system.logging import init_default_logger
from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__)


class semiLASER(BaseSpectroscopy):
    """Semi-LASER localized spectroscopy sequence with coil combination support."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
