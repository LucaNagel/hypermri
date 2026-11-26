"""FLOWMAP (flow-compensated gradient echo method) wrapper for anatomical imaging."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class FLOWMAP(BaseAnatomy):
    """flow-compensated gradient echo method sequence for anatomical imaging."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
