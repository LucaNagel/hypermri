from hypermri.loading.protocols import VendorHandlerProtocol
from hypermri.utils.system.logging import init_default_logger

from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class Singlepulse(BaseSpectroscopy):
    """Wrapper which decides wether to load the single or dualchannel Singlepulse class.

    Built from the vendor handler of its scan.
    """

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)

    # ---------------------------------------------
    # This needs to brought to reconstruction strategy module
    # ---------------------------------------------
