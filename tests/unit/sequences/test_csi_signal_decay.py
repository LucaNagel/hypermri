import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.sequences.CSI import CSI


def test_csi_reset_is_removed():
    """``CSI.reset`` reloaded data without undoing the transposition swap."""
    assert not hasattr(CSI, "reset")
