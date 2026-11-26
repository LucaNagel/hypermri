"""Magritek reconstruction strategies."""

import numpy as np

from ._spectroscopy import FidSpectroRecoStrategy


class MagritekSpectroRecoStrategy(FidSpectroRecoStrategy):
    """Reconstruct Magritek FID stacks into spectral-domain data."""

    spectrum_name = "magritek_spectrum"
    missing_fid_message = "Magritek handler has no FID data loaded"

    @staticmethod
    def _build_frequency_axes(n_points, dwell_time, handler):
        """Use the acquired calibration with the processed point count."""
        frequency_hz, _ = FidSpectroRecoStrategy._build_frequency_axes(
            n_points, dwell_time, handler
        )
        bandwidth_hz = getattr(handler, "bandwidth_hz", None)
        bandwidth_ppm = getattr(handler, "bandwidth_ppm", None)
        frequency_ppm = None
        if bandwidth_hz not in (None, 0) and bandwidth_ppm not in (None, 0):
            frequency_ppm = (
                frequency_hz * (bandwidth_ppm / bandwidth_hz) + handler.center_ppm
            )
        return frequency_hz, frequency_ppm

    def _ensure_handler_fid(self, handler):
        data = getattr(handler, "scanner_data", None)
        fid = getattr(handler, "fid", None)
        if data is None and fid is None:
            data = handler.load_data(getattr(handler, "path", None))
            fid = getattr(handler, "fid", None)
        if data is not None and hasattr(data, "dims"):
            if fid is not None:
                if np.shape(fid) != data.shape:
                    raise ValueError(
                        "Magritek fid shape differs from its labelled export; reload the export before reconstruction."
                    )
                return data.copy(data=fid)
            return data
        return fid


__all__ = ["MagritekSpectroRecoStrategy"]
