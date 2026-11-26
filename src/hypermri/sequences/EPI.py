"""Echo-planar imaging (EPI) sequence implementation."""

import traceback

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class EPI(BaseAnatomy):
    """Gradient-echo echo-planar imaging with standard anatomy utilities."""

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("seq2d",),
        "full": ("seq2d", "raw"),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Initialise the EPI sequence wrapper and reconstruct base datasets."""
        super().__init__(handler, **kwargs)
