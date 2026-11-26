"""Base reconstruction strategy class."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import xarray as xr

from hypermri.compatibility import assign_frequency_axis, resolve_frequency_axis


def _has_spectral_axis(handler):
    """Use the mapped sequence category before the vendor's spectroscopy flag."""
    from hypermri.loading.contracts import SequenceCategory
    from hypermri.sequences.registry import get_sequence_registry

    vendor = getattr(handler, "vendor", None)
    vendor = getattr(vendor, "value", vendor)
    entry = get_sequence_registry().resolve_sequence(
        getattr(handler, "type", None) or getattr(handler, "sequence_name", ""),
        vendor,
    )
    if entry is not None:
        return entry["category"] == SequenceCategory.SPECTROSCOPY
    return bool(getattr(handler, "is_spectroscopic", False))


class BaseRecoStrategy(ABC):
    """Abstract base class for a reconstruction strategy."""

    accepted_options: ClassVar[frozenset[str]] = frozenset()
    """Names of the keyword options that :meth:`reconstruct` reads.

    The loading code rejects every other option before reading data, and each
    default option of :data:`hypermri.reconstruction.selection.RECONSTRUCTIONS`
    must be one of these names. Each concrete strategy defines its own set.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the reconstruction strategy."""
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def reconstruct(self, handler, **kwargs) -> xr.DataArray:
        """
        Perform the reconstruction.

        Parameters
        ----------
        handler : object
            Vendor handler that supplies the raw samples and parameters.
        **kwargs
            Strategy-specific reconstruction options.

        Returns
        -------
        A reconstructed xarray.DataArray.
        """

    def _append_parameters(self, array, handler, **kwargs):
        """
        Append parameters to the xarray DataArray.

        Parameters
        ----------
        array: xr.DataArray
            The xarray DataArray to which parameters will be appended.
        handler:
            The vendor handler containing the parameters.
        kwargs:
            Additional keyword arguments to append.

        """
        sampling = array.coords.get("acquisition_time_s")
        array.attrs["sampling_time_axis_s"] = (
            sampling.values
            if sampling is not None
            else handler.get("sampling_time_axis_s", None)
        )
        if _has_spectral_axis(handler):
            # Reconstructed coordinates describe the processed sample grid;
            # handler frequencies still describe the original acquisition.
            for unit, name in (("ppm", "freq_axis_ppm"), ("Hz", "freq_axis_hz")):
                values = resolve_frequency_axis(array, unit)
                if values is None:
                    values = getattr(handler, name, None)
                length = array.sizes.get("acquisition_idx")
                if values is not None and (length is None or np.size(values) == length):
                    array = assign_frequency_axis(array, values, unit)
        array.attrs["is_spectroscopic"] = handler.is_spectroscopic
        return array


__all__ = ["BaseRecoStrategy"]
