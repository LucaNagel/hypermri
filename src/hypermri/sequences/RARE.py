from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import init_default_logger
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__)


class RARE(BaseAnatomy):
    """Rapid acquisition with relaxation enhancement for fast spin-echo anatomy."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)

        if hasattr(self, "handler") and self.handler:
            logger.debug(
                f"RARE: handler available, has fid: {hasattr(self.handler, 'fid')}"
            )
        else:
            logger.warning(
                "RARE: No handler available after BaseAnatomy initialization"
            )
