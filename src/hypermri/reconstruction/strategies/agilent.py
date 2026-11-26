"""Agilent reconstruction strategies."""

from ._spectroscopy import FidSpectroRecoStrategy


class AgilentSpectroRecoStrategy(FidSpectroRecoStrategy):
    """Reconstruction strategy for Agilent/Varian spectroscopy data."""

    spectrum_name = "agilent_spectrum"
    missing_fid_message = "Agilent handler has no FID loaded"


__all__ = ["AgilentSpectroRecoStrategy"]
