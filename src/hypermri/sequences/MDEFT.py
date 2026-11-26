"""Modified Driven Equilibrium Fourier Transform (MDEFT)"""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class MDEFT(BaseAnatomy):
    """Modified Driven Equilibrium Fourier Transform (MDEFT)."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the MDEFT sequence wrapper and reconstruct base datasets."""
        super().__init__(handler, **kwargs)
