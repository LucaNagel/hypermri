"""Interactive browser viewer of labelled MRI data.

Use :func:`plot` (or ``scan.plot``) with :class:`PlotOptions` to show a scan;
it returns a :class:`~hypermri.visualization.scan_view.ScanView`.
:func:`~hypermri.visualization.scan_view.plot_side_by_side` places several
views next to each other with linked slices and masks,
:func:`~hypermri.visualization.study_view.plot_study` lists the scans of a
study, and :func:`~hypermri.visualization.signal_decay.plot_signal_decay`
chooses the FID filter of a CSI scan. Importing this package
constructs no widget and reads no scan; the viewer and its optional
dependencies are imported on first use.
"""

from importlib import import_module as _import_module

from .api import PlotOptions, plot

__all__ = [
    "ScanView",
    "PlotOptions",
    "plot",
    "plot_scan",
    "plot_side_by_side",
    "plot_signal_decay",
    "plot_study",
]

_LAZY = {
    "ScanView": "hypermri.visualization.scan_view",
    "plot_scan": "hypermri.visualization.scan_view",
    "plot_side_by_side": "hypermri.visualization.scan_view",
    "plot_signal_decay": "hypermri.visualization.signal_decay",
    "plot_study": "hypermri.visualization.study_view",
}


def __getattr__(name: str):
    """Import the viewer objects on first access."""
    if name in _LAZY:
        return getattr(_import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
