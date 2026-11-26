from hypermri.loading.protocols import VendorHandlerProtocol

from .base_anatomy import BaseAnatomy


class RAREVTR(BaseAnatomy):
    """Variable repetition-time RARE acquisition for T1 mapping."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Build the sequence from the vendor handler of its scan."""
        super().__init__(handler, **kwargs)
