"""Conventional spin-echo imaging sequence utilities."""

from hypermri.loading.protocols import VendorHandlerProtocol

from .base_anatomy import BaseAnatomy


class SpinEcho(BaseAnatomy):
    """Conventional single- or multi-slice spin-echo imaging."""

    RECONSTRUCTION_POLICY = {
        "lazy": (),
        "minimal": ("raw",),
        "full": ("raw",),
    }

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Wrap a loaded spin-echo experiment with anatomy helpers."""
        super().__init__(handler, **kwargs)
