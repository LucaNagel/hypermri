"""Load, reconstruct and analyze labelled MRI data.

The supported root API is ``Study``, ``MRIExperiment``, ``load_scan``,
``create_experiment``, ``load_external_experiment`` and ``__version__``.
Importing the package loads no vendor, plotting backend or notebook runtime.
Public objects are resolved on first access and retain their implementation's
identity. Import vendor handlers from ``hypermri.vendors`` and mixins from
``hypermri.mixins``; the former root aliases were removed on 15 November 2026.

Examples
--------
>>> from hypermri import Study, MRIExperiment, load_scan
>>> from hypermri.visualization import PlotOptions, plot
"""

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("hypermri")
except _PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "MRIExperiment",
    "load_scan",
    "create_experiment",
    "load_external_experiment",
    "Study",
]

_ROOT_IMPORTS = {
    "MRIExperiment": ".loading.experiment",
    "load_scan": ".loading.scans",
    "create_experiment": ".loading.scans",
    "load_external_experiment": ".loading.scans",
    "Study": ".study",
}


def __getattr__(name: str):
    """Resolve a supported root export lazily.

    Parameters
    ----------
    name : str
        Name listed in ``__all__`` other than ``__version__``.

    Returns
    -------
    object
        Original implementation, cached after the first lookup, without
        constructing an instance.

    Raises
    ------
    AttributeError
        If the name is not a supported root export.
    ImportError
        If a dependency of the requested implementation is unavailable.
    """
    module = _ROOT_IMPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    """Return the discoverable root names without importing them.

    Returns
    -------
    list of str
        Sorted existing attributes and lazily available root exports.
    """
    return sorted(set(globals()) | set(_ROOT_IMPORTS))
