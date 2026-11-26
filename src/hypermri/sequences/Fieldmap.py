"""Field mapping sequence wrapper providing reconvenience helpers."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")

DB_MSG = False  # Change this to False to hide detailed error messages


class Fieldmap(BaseAnatomy):
    """Gradient-echo acquisition tailored for B0 field mapping."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
