"""Plot labelled MRI data in the interactive browser viewer.

:func:`plot` is the one entry for scans and labelled arrays; it hands its
options once to :func:`~hypermri.visualization.scan_view.plot_scan`. Importing
this module constructs no widgets and reads no scans.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import xarray as xr

from hypermri.visualization.state import PlotState

# Options of the earlier Python-side plotter, with what replaces them.
_REMOVED_OPTIONS = {
    "anatomy_align": "use coordinate_mode='xyz' to align anatomy and map in scanner space",
    "return_controller": "plot returns the ScanView itself",
    "enable_mask": "set view.tool = 'masks' on the returned view",
    "enable_peak_selector": "set view.tool = 'fit' on the returned view",
    "quick_fit": "set view.tool = 'fit' on the returned view",
    "skip_quick_fit": "the fit tool starts only when view.tool = 'fit' is set",
    "fit_params": "choose the lines in the fit tool (view.tool = 'fit')",
    "peak_selector_params": "choose the lines in the fit tool (view.tool = 'fit')",
    "backend": "the view always draws with Plotly in the browser",
    "profile_logging": "the browser viewer has no profiling log",
    "profile_log_path": "the browser viewer has no profiling log",
    "auto_hide": "each layer always shows its plane nearest to the shown one",
    "percent_difference": "difference overlays are no longer offered",
    "subtract_overlay": "difference overlays are no longer offered",
    "state_overrides": "pass a PlotState as PlotOptions.state or use the view's set_* methods",
}


@dataclass(frozen=True)
class PlotOptions:
    """Settings of :func:`plot` and ``scan.plot``; None keeps the default.

    Parameters
    ----------
    anatomy : object or None, optional
        Anatomical scan, or a DataArray with scanner coordinates ``x``,
        ``y``, ``z``, drawn under the map. For single-voxel spectroscopy the
        anatomy is shown with the outline of the excited voxel instead.
    overlay : object or None, optional
        Parameter maps drawn above the map: a DataArray, a Dataset such as
        fit results, or a mapping of layer names to either.
    variable : str or None, optional
        Dataset variable shown first when ``data`` is a Dataset such as
        ``scan.fit_results``.
    coordinate_mode : {"rps", "xyz"} or None, optional
        Spatial axes: read/phase positions of the scan (``"rps"``) or
        scanner coordinates in mm (``"xyz"``). None chooses ``"xyz"`` when
        an anatomy or overlay is given, otherwise the saved or default mode.
    curve_dim : str or None, optional
        Dimension of the curve panel, for example ``"repetition_idx"`` for a
        time course. None shows the spectrum, or for images the first of
        inversion, b-value, echo, repetition and channel dimensions present.
    split_metabolites : bool or None, optional
        Split repetitions that alternate between two metabolites into
        ``metabolite_idx``. None splits only the interleaved FISP sequence.
    settings_context : str or None, optional
        Key under which ``view.save_settings()`` stores the display on the
        scan. None uses ``"plot_scan"``.
    state : PlotState or None, optional
        Display state to start from, for example with a colour map or line
        broadening (Hz) already chosen. The object is completed in place and
        kept by the view. None uses the saved settings or the defaults.
    slice_indices : mapping of str to int or None, optional
        Start position as source indices by dimension name, for example
        ``{"slice_idx": 3}``. Positions, not coordinate values. None starts
        at the saved position or the voxel with the strongest signal.
    plot_height : int or None, optional
        Height of the plots in pixels, at least 160. None keeps 400 px. It
        can be changed later with ``view.plot_height``.

    Raises
    ------
    ValueError
        If an option of the earlier plotter is given (the message names its
        replacement), or ``state`` is not a PlotState.
    TypeError
        If an unknown option is given.

    Notes
    -----
    Construction reads no scan and builds no widget. Mappings are copied
    into read-only mappings; the arrays and scans they hold are not copied.

    Examples
    --------
    >>> options = PlotOptions(plot_height=500, slice_indices={"slice_idx": 2})
    >>> view = plot(scan, options)  # doctest: +SKIP
    """

    anatomy: Any | None = None
    overlay: Any | None = None
    variable: str | None = None
    coordinate_mode: str | None = None
    curve_dim: str | None = None
    split_metabolites: bool | None = None
    settings_context: str | None = None
    state: PlotState | None = None
    slice_indices: Mapping[str, int] | None = None
    plot_height: int | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> PlotOptions:
        _reject_unknown_options(kwargs, "PlotOptions")
        return super().__new__(cls)

    def __post_init__(self) -> None:
        if self.state is not None and not isinstance(self.state, PlotState):
            raise ValueError(
                "PlotOptions.state must be a PlotState; got "
                f"{type(self.state).__name__}."
            )
        if self.slice_indices is not None:
            object.__setattr__(
                self, "slice_indices", MappingProxyType(dict(self.slice_indices))
            )
        if isinstance(self.overlay, Mapping) and not isinstance(
            self.overlay, xr.Dataset
        ):
            object.__setattr__(self, "overlay", MappingProxyType(dict(self.overlay)))


def _reject_unknown_options(names: Mapping[str, Any], operation: str) -> None:
    """Raise for options of the earlier plotter and for unknown names."""
    removed = [name for name in names if name in _REMOVED_OPTIONS]
    if removed:
        hints = "; ".join(f"{name}: {_REMOVED_OPTIONS[name]}" for name in removed)
        raise ValueError(f"{operation}: option(s) no longer available. {hints}.")
    known = {field.name for field in fields(PlotOptions)}
    unknown = sorted(set(names) - known)
    if unknown:
        raise TypeError(
            f"{operation}: unknown option(s) {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(known))}."
        )


def plot(
    data: xr.DataArray | xr.Dataset | Any,
    options: PlotOptions | None = None,
    *,
    scan: Any | None = None,
) -> Any:
    """Show a scan or labelled array in the interactive browser viewer.

    The map shows the selected plane, the curve panel the spectrum (or
    curve) of the selected voxel. Clicking and the arrow keys change the
    selection in the browser; masks are drawn with ``view.tool = "masks"``
    and spectra are fitted with ``view.tool = "fit"``. See
    :func:`~hypermri.visualization.scan_view.plot_scan` for every detail of
    the data, anatomy and overlay handling.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset or object
        Labelled samples with at least two of ``read_idx``, ``phase_idx`` and
        ``slice_idx`` (or a single-voxel spectrum), a Dataset of maps such as
        ``scan.fit_results``, or a scan whose ``primary_data`` supplies them.
        The samples are not modified.
    options : PlotOptions or None, optional
        Display settings. None keeps saved settings and defaults.
    scan : object or None, optional
        Scan supplying frequency axis, geometry and mask storage for an
        explicit DataArray or Dataset. Do not combine with a scan as ``data``.

    Returns
    -------
    ScanView
        The widget; display it as the last expression of a notebook cell.
        Construction prepares the shown plane eagerly.

    Raises
    ------
    TypeError
        If ``options`` is not a PlotOptions or the data are not labelled.
    ValueError
        For data or options the viewer cannot show; the message names the
        option and how to correct it.
    ImportError
        If the optional visualization dependencies are unavailable.

    Examples
    --------
    >>> view = plot(scan)  # doctest: +SKIP
    >>> view = plot(scan, PlotOptions(anatomy=t2_scan, plot_height=500))  # doctest: +SKIP
    >>> view.tool = "masks"  # doctest: +SKIP
    """
    if options is None:
        options = PlotOptions()
    if not isinstance(options, PlotOptions):
        raise TypeError(
            "plot options must be PlotOptions or None; for example "
            "plot(scan, PlotOptions(plot_height=500))."
        )
    from hypermri.visualization.scan_view import plot_scan

    view = plot_scan(
        data,
        variable=options.variable,
        anatomy=options.anatomy,
        overlay=dict(options.overlay)
        if isinstance(options.overlay, MappingProxyType)
        else options.overlay,
        coordinate_mode=options.coordinate_mode,
        curve_dim=options.curve_dim,
        split_metabolites=options.split_metabolites,
        scan=scan,
        state=options.state,
        settings_context=options.settings_context,
        slice_indices=options.slice_indices,
    )
    if options.plot_height is not None:
        view.plot_height = options.plot_height
    return view


def _scan_plot_options(
    options: PlotOptions | None,
    option_values: Mapping[str, Any],
    operation: str,
) -> PlotOptions:
    """Return the PlotOptions of a scan method given options or keywords."""
    if options is not None and not isinstance(options, PlotOptions):
        raise TypeError(f"{operation}: options must be PlotOptions or None.")
    if options is not None and option_values:
        raise TypeError(
            f"{operation} takes PlotOptions or option keywords, not both; put "
            f"{', '.join(sorted(option_values))} into the PlotOptions."
        )
    if options is None:
        _reject_unknown_options(option_values, operation)
        options = PlotOptions(**option_values)
    return options


__all__ = ["PlotOptions", "plot"]
