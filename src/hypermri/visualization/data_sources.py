"""Lazy discovery and resolution of plot data sources."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import xarray as xr

DATA_SOURCE_LABELS = OrderedDict(
    (
        ("data", "raw-reco"),
        ("scanner_data", "scanner-reco"),
        ("dicom", "dicom"),
    )
)


@dataclass(frozen=True)
class DataSourceEntry:
    key: str
    label: str


class DataSourceCatalog:
    """Describe the plot data sources of a scan without loading DICOM pixels.

    The catalog lists the scan's raw reconstruction ``data``, its
    scanner reconstruction ``scanner_data`` and, when the vendor handler
    reports a DICOM reconstruction, the DICOM series.  Arrays that are
    already available are cached by reference; DICOM pixels are read only on
    the first :meth:`load` of ``"dicom"``.

    Parameters
    ----------
    source_obj : object or None
        Scan whose ``data`` and ``scanner_data`` attributes are read.
        ``has_dicom_reco``, ``load_dicom_data`` and ``dicom_data`` are read
        from its vendor handler ``source_obj.handler``, or from
        ``source_obj`` itself when it has no handler.  None gives an empty
        catalog apart from ``current_data``.
    current_data : xarray.DataArray or xarray.Dataset, optional
        Data shown at present.  It is identified among the sources by object
        identity, or as ``"dicom"`` by its ``source`` attribute; otherwise it
        is listed as ``"provided"``.

    Attributes
    ----------
    source_obj : object or None
        The scan given to the constructor.
    current_key : str or None
        Key of ``current_data``, or None without it.

    Notes
    -----
    Reading ``data`` may start a lazy reconstruction of the scan; an
    exception during that read leaves the source out of the catalog.
    """

    def __init__(self, source_obj: Any, current_data: Any = None):
        self.source_obj = source_obj
        # DICOM series belong to the vendor handler of a scan.
        self._vendor_source = getattr(source_obj, "handler", source_obj)
        self._cache: dict[str, xr.DataArray | xr.Dataset] = {}
        self._entries: OrderedDict[str, DataSourceEntry] = OrderedDict()
        self._discover_array("data")
        self._discover_array("scanner_data")
        self._discover_dicom()

        self.current_key = self.identify(current_data)
        if current_data is not None and self.current_key is None:
            if isinstance(current_data, xr.DataArray | xr.Dataset):
                self._cache["provided"] = current_data
                self._entries["provided"] = DataSourceEntry("provided", "provided data")
                self.current_key = "provided"

    def _discover_array(self, key: str) -> None:
        if self.source_obj is None:
            return
        try:
            value = getattr(self.source_obj, key, None)
        except Exception:
            return
        if isinstance(value, xr.DataArray | xr.Dataset):
            self._cache[key] = value
            self._entries[key] = DataSourceEntry(key, DATA_SOURCE_LABELS[key])

    def _discover_dicom(self) -> None:
        if self.source_obj is None:
            return
        try:
            available = bool(getattr(self._vendor_source, "has_dicom_reco", False))
        except Exception:
            available = False
        if not available:
            return
        self._entries["dicom"] = DataSourceEntry("dicom", DATA_SOURCE_LABELS["dicom"])

    @property
    def has_canonical_sources(self) -> bool:
        return any(key in DATA_SOURCE_LABELS for key in self._entries)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def options(self) -> list[tuple[str, str]]:
        return [(entry.label, entry.key) for entry in self._entries.values()]

    def identify(self, data: Any) -> str | None:
        if data is None:
            return None
        for key, value in self._cache.items():
            if data is value:
                return key
        if isinstance(data, xr.DataArray) and data.attrs.get("source") == "dicom":
            if "dicom" in self._entries:
                self._cache["dicom"] = data
                return "dicom"
        return None

    def load(self, key: str) -> xr.DataArray | xr.Dataset:
        """Return the data of one source, reading DICOM pixels on first use.

        Parameters
        ----------
        key : str
            Source key from :attr:`keys`.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The cached array, or the DICOM series of the vendor handler,
            which is cached for later calls.

        Raises
        ------
        KeyError
            If ``key`` is not a source of this catalog.
        ValueError
            If the source is not available or the DICOM loader does not
            return a DataArray.
        """
        if key not in self._entries:
            raise KeyError(f"Unknown data source: {key}")
        if key in self._cache:
            return self._cache[key]
        if key != "dicom":
            raise ValueError(f"Data source is unavailable: {key}")

        loader = getattr(self._vendor_source, "load_dicom_data", None)
        if callable(loader):
            value = loader()
        else:
            value = getattr(self._vendor_source, "dicom_data", None)
        if not isinstance(value, xr.DataArray):
            raise ValueError("DICOM source did not return an xarray DataArray")
        self._cache[key] = value
        return value


__all__ = ["DATA_SOURCE_LABELS", "DataSourceCatalog", "DataSourceEntry"]
