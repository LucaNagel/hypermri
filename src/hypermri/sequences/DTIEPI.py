"""Diffusion tensor EPI sequence wrapper for HyperMRI."""

import traceback

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class DTIEPI(BaseAnatomy):
    """Diffusion tensor imaging based on echo-planar readouts."""

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("seq2d",),
        "full": ("seq2d", "raw"),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the diffusion EPI sequence and reconstruct data products."""
        super().__init__(handler, **kwargs)
