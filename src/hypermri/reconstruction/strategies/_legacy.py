"""Compatibility imports for the former monolithic strategy module."""

from .agilent import AgilentSpectroRecoStrategy
from .base import BaseRecoStrategy
from .bruker.csi import BrukerCsiRecoStrategy
from .bruker.rare import BrukerRareRawRecoStrategy
from .bruker.raw import BrukerMGERawRecoStrategy, BrukerRawRecoStrategy
from .bruker.seq2d import Brukerseq2dRecoStrategy
from .bruker.spectroscopy import BrukerSpectroRecoStrategy
from .magritek import MagritekSpectroRecoStrategy
from .siemens import SiemensCsiRecoStrategy, SiemensRawRecoStrategy

__all__ = [
    "BaseRecoStrategy",
    "BrukerRawRecoStrategy",
    "BrukerMGERawRecoStrategy",
    "Brukerseq2dRecoStrategy",
    "BrukerRareRawRecoStrategy",
    "BrukerSpectroRecoStrategy",
    "BrukerCsiRecoStrategy",
    "SiemensRawRecoStrategy",
    "SiemensCsiRecoStrategy",
    "AgilentSpectroRecoStrategy",
    "MagritekSpectroRecoStrategy",
]
