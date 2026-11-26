"""Diffusion tensor imaging sequence wrapper for HyperMRI."""

import traceback

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class DTI(BaseAnatomy):
    """Diffusion tensor imaging"""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the Diffusion tensor imaging sequence and reconstruct data products."""
        super().__init__(handler, **kwargs)
