"""Provide stimulated-echo spectroscopy with package-controlled logging."""

from hypermri.loading.protocols import VendorHandlerProtocol

from ..utils.system.logging import LOG_MODES as LOG_MODES
from ..utils.system.logging import init_default_logger
from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__)


class STEAM(BaseSpectroscopy):
    """Stimulated echo acquisition mode sequence for localized spectroscopy."""

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("raw",),
        "full": ("seq2d", "raw"),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
