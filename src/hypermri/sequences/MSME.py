"""Multi-Slice Multi-Echo (MSME) sequence implementation."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class MSME(BaseAnatomy):
    """Multi-slice multi-echo imaging with standard anatomy utilities."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the MSME sequence wrapper and reconstruct base datasets."""
        super().__init__(handler, **kwargs)
