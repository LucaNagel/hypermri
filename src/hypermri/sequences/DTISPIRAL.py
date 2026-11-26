"""SPIRAL (readout) with Diffusion Tensor imaging (DTI) module sequence implementation."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class DTISPIRAL(BaseAnatomy):
    """DTI Spiral readout anatomy utilities."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the DTI SPIRAL sequence wrapper and reconstruct base datasets."""
        super().__init__(handler, **kwargs)
