"""Shared vendor-handler contract and retained legacy source dispatch.

Concrete handlers own path validation, metadata parsing and acquired-data reads.
New loading workflows use :func:`hypermri.load_scan`; VendorFactory remains
available for compatibility with its historical two-vendor dispatch only.
Importing this module constructs no handler and reads no scanner data.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

import numpy as np
import xarray as xr


class VendorHandler(ABC):
    """Abstract interface for vendor-specific experiment handlers.

    Vendor handlers are responsible for three things:

    1. Validating and normalising the incoming source path.
    2. Mapping vendor metadata onto HyperMRI's canonical parameter schema.
    3. Loading raw vendor data into standardized array objects.

    Handlers that also inherit from :class:`hypermri.mixins.StandardizedMRIMixin`
    can store a validated parameter snapshot in ``self.params``. The default
    :meth:`summary` implementation exposes that snapshot uniformly across
    vendors.
    """

    # Each subclass must define these properties and methods.
    @property
    @abstractmethod
    def id(self) -> str:
        """Return the vendor's experiment identifier.

        Returns
        -------
        str
            Identifier supplied by the concrete handler, without loading data.
        """
        pass

    def _derive_b0_tesla(self) -> float | None:
        """Return the static field from the handler's reference frequency.

        Every handler sets ``b0_tesla`` through this method after it has set
        ``reference_frequency_mhz`` and ``nucleus``, so all vendors share the
        definition of :func:`hypermri.utils.physics.b0_tesla_from_reference_frequency`.

        Returns
        -------
        float or None
            Static field in T, or None if the scan reports no reference
            frequency or no nucleus.

        Raises
        ------
        ValueError
            If the nucleus has no known gyromagnetic ratio or the reference
            frequency is not positive; the message names the scan path.
        """
        from hypermri.utils.physics import b0_tesla_from_reference_frequency

        try:
            return b0_tesla_from_reference_frequency(
                getattr(self, "reference_frequency_mhz", None),
                getattr(self, "nucleus", None),
            )
        except ValueError as error:
            raise ValueError(
                f"{type(self).__name__} for {getattr(self, 'path', None)}: "
                f"cannot derive b0_tesla: {error}"
            ) from error

    @property
    def dwell_times_s(self) -> float:
        """Return the acquired sampling interval in seconds.

        Returns
        -------
        float
            Existing dwell_time_s when available, otherwise the value derived
            by the shared temporal helper from the handler's metadata. Nothing
            is cached or changed, and no signal samples are read.

            The shared helper retains its historical zero fallback for absent
            metadata; this property does not introduce additional validation.
        """
        dwell_time_s = getattr(self, "dwell_time_s", None)
        if dwell_time_s is not None:
            return dwell_time_s

        from hypermri.utils.time.temporal import get_dwell_time_s

        return get_dwell_time_s(scan=self)

    def _parse_parameters(self, source) -> Mapping[str, Any]:
        """Return vendor-specific parameters mapped to canonical keys."""
        raise NotImplementedError

    def load_parameters(self, source) -> Dict[str, Any]:
        """Parse and store the vendor's standardized parameter snapshot.

        Parameters
        ----------
        source : object
            Vendor-specific source understood by the concrete parser, usually
            a scan directory or measurement file path.

        Returns
        -------
        dict of str to object
            Parsed parameters, with units and shapes defined by the concrete
            handler. Stores the values through set_parameters when available,
            otherwise assigns params. This is a shallow mapping conversion;
            contained arrays are not copied.

        Raises
        ------
        NotImplementedError
            If a subclass has not implemented its parameter parser.
        OSError
            If the concrete parser cannot read its metadata source.
        ValueError
            If the concrete parser rejects the metadata.
        """
        parsed = dict(self._parse_parameters(source))
        if hasattr(self, "set_parameters"):
            self.set_parameters(parsed)
        else:
            self.params = parsed  # type: ignore[attr-defined]
        return parsed

    def get_parameters(self) -> Dict[str, Any]:
        """Return a shallow copy of the stored standardized parameters.

        Returns
        -------
        dict of str to object
            Stored parameter mapping, or an empty mapping before parsing.
            Values retain their units, shapes and identities. No files are read
            and editing mapping keys does not change the stored snapshot.
        """
        if hasattr(self, "params"):
            return dict(self.params)
        return {}

    def summary(self) -> Dict[str, Any]:
        """Return the stored vendor-independent parameter summary.

        Returns
        -------
        dict of str to object
            Result of get_parameters with the same shallow-copy, units and
            shape contract. Does not load data or modify the handler.
        """
        return self.get_parameters()

    @abstractmethod
    def load_data(self, source) -> xr.DataArray:
        """Read acquired vendor data into a labelled array.

        Parameters
        ----------
        source : object
            Vendor-specific source, interpreted by the concrete handler.

        Returns
        -------
        xarray.DataArray
            Eager acquired signal; dimensions, coordinates, units, dtype,
            provenance and storage side effects are specified by the concrete
            implementation. The acquired source itself must remain unchanged.

        Raises
        ------
        OSError
            If the concrete reader cannot access the acquired source.
        ValueError
            If the concrete reader rejects the signal or metadata layout.
        """


class VendorFactory:
    """Retain the legacy two-vendor dispatcher for existing callers.

    New workflows use :func:`hypermri.load_scan`, which owns validated vendor
    and sequence detection. This class preserves its historical directory/fid
    and .dat checks and does not implement the supported detection policy.
    """

    @staticmethod
    def get_handler(source):
        """Construct a handler using the historical source checks.

        Parameters
        ----------
        source : str
            Directory containing fid for Bruker, or a .dat file for Siemens.
            This compatibility API retains its original string-only behavior.

        Returns
        -------
        VendorHandler
            Constructed vendor handler with its original constructor defaults
            and metadata/data-loading side effects.

        Raises
        ------
        ValueError
            If neither historical source check matches or parsing fails.
        OSError
            If the source cannot be listed or the handler cannot read it.

        Examples
        --------
        Use the supported loading API for new code:

        >>> from hypermri import load_scan
        >>> scan = load_scan("path/to/scan", mode="lazy")  # doctest: +SKIP
        """
        if os.path.isdir(source) and "fid" in os.listdir(source):
            from hypermri.vendors.bruker import BrukerHandler

            return BrukerHandler(source)
        elif source.endswith(".dat"):
            from hypermri.vendors.siemens import SiemensHandler

            return SiemensHandler(source)
        else:
            raise ValueError(f"Unsupported data source: {source}")
