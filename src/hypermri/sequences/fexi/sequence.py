"""FEXI sequence wrapper."""

from __future__ import annotations

from hypermri.loading.protocols import VendorHandlerProtocol

from ..base_anatomy import BaseAnatomy
from .compute import describe_fexi_layout
from .parameters import collect_fexi_parameters


class FEXI(BaseAnatomy):
    """Basic Filter Exchange Imaging sequence scaffold."""

    def __init__(self, handler: VendorHandlerProtocol, **kwargs):
        """Accept a pre-loaded MRIExperiment object as input."""
        super().__init__(handler, **kwargs)
        self.fexi_parameters = collect_fexi_parameters(self)

    def describe_layout(self) -> dict:
        """Return a lightweight summary of loaded FEXI data and parameters."""
        return describe_fexi_layout(self)
