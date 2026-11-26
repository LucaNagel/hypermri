"""Resolve utility re-exports without importing plotting during package loading.

Domain modules own the implementations. Attribute access imports the original
object once and caches it. No scan or widget is constructed by this package.
"""

from importlib import import_module as _import_module

_IMPORTS = {
    "combine_channels": ".mri",
    "disable_usetex_if_unavailable": ".plot",
    "install_matplotlib_usetex_fallback": ".plot",
    "functools": ".system.decorators",
    "init_default_logger": ".system.decorators",
    "logger": ".system.decorators",
    "onlycallonce": ".system.decorators",
    "optional_arg_decorator": ".system.decorators",
}
__all__ = sorted(_IMPORTS)


def __getattr__(name: str):
    """Resolve one existing utility export on first access.

    Parameters
    ----------
    name : str
        Name exported by this package.

    Returns
    -------
    object
        Original domain object, cached without wrapping it.

    Raises
    ------
    AttributeError
        If the name is not an exported utility.
    """
    module = _IMPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module, __name__), name)
    globals()[name] = value
    return value
