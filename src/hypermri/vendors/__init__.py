"""
Vendor module for hypermri.

Registry functions import without loading vendor definitions. Handler exports
resolve lazily and retain their original identities; no acquired data is read.
"""

from importlib import import_module as _import_module

from .registry import (
    VendorRegistry,
    detect_vendor,
    get_vendor_handler,
    get_vendor_registry,
    register_vendor,
)

__all__ = [
    "AgilentHandler",
    "BrukerHandler",
    "BrukerTopSpinHandler",
    "ExternalDataHandler",
    "MagritekHandler",
    "SiemensHandler",
    "VendorRegistry",
    "VendorHandler",
    "get_vendor_registry",
    "register_vendor",
    "get_vendor_handler",
    "detect_vendor",
]


_HANDLER_MODULES = {
    "AgilentHandler": ".agilent",
    "BrukerHandler": ".bruker",
    "BrukerTopSpinHandler": ".topspin",
    "ExternalDataHandler": ".external",
    "MagritekHandler": ".magritek",
    "SiemensHandler": ".siemens",
    "VendorHandler": ".base",
}


def __getattr__(name: str):
    """Import a vendor definition without constructing a handler.

    Parameters
    ----------
    name : str
        Exported vendor handler name.

    Returns
    -------
    type
        Original vendor class, cached for later access.

    Raises
    ------
    AttributeError
        If the requested name is not a handler export.
    """
    module = _HANDLER_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module, __name__), name)
    globals()[name] = value
    return value
