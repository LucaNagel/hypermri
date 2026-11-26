"""Arrange the browser viewer: which panels and control groups appear, and where.

The browser viewer (:class:`~hypermri.visualization.scan_view.ScanView`) is put
together from named pieces: panels (plots) and control groups (named sets of
inputs, :data:`CONTROL_GROUPS`). From top to bottom:

* The **top bar** (:data:`TOP_BAR`) holds the control groups needed all the
  time, for example the slice sliders, the anatomy slice and the opacity
  of map and overlay, and the "Plot options" button.
* **Plot options** (:data:`OPTION_SECTIONS`) are titled sections of control
  groups in a panel below the top bar that the "Plot options" button opens
  and closes. It starts closed. A section whose groups have nothing to offer
  for the shown data (no anatomy, no overlay) is hidden.
* **Panels** are the plots side by side: ``"image"`` (the map of one plane),
  ``"curve"`` (the spectrum of the selected voxel) and ``"time"`` (its time
  course over repetitions, shown only when there are several).
* **Below the plots** (:data:`BELOW_PLOTS`) sit the control groups that work
  on the plots, by default the ``"tool"`` menu and the "show parameters"
  checkbox (:data:`CONTROL_GROUPS` ``"parameters"``), then a status line.

To move a control group, move its name between these tuples. Nothing else
needs to change: the browser code builds exactly the groups listed here.

* **Tools** (plugins) are control groups that are shown one at a time. The
  ``"tool"`` control group is a menu listing :data:`TOOLS`; the inputs of the
  chosen tool appear right after the menu and the others are hidden.
  :attr:`ScanView.tool <hypermri.visualization.scan_view.ScanView.tool>` is the
  chosen tool. To add a tool, write its control group and list its name in
  :data:`TOOLS`.

The browser code itself lives as small JavaScript files in the ``browser``
folder next to this module, one file per panel or control group
(:data:`BROWSER_PARTS`). :func:`browser_script` joins them, in the listed
order, into the single script that the notebook loads. No build tool is
needed: the parts are plain JavaScript that share one module scope.

Examples
--------
>>> browser_layout()["panels"]
['image', 'curve', 'time']
>>> browser_layout()["tools"]
['masks', 'fit', 'curve_fit', 'ideal', 'signal_decay']
>>> browser_layout(
...     top_bar=["slices"], option_sections=[("Map", ["display"])],
...     below_plots=[], tools=[],
... )["option_sections"]
[['Map', ['display']]]
"""

from collections.abc import Sequence
from importlib.resources import files as resource_files
from typing import Any

__all__ = [
    "BROWSER_PARTS",
    "CONTROL_GROUPS",
    "BELOW_PLOTS",
    "OPTION_SECTIONS",
    "PANELS",
    "PANEL_DESCRIPTIONS",
    "TOOLS",
    "TOP_BAR",
    "browser_layout",
    "browser_script",
]

#: Panels the browser can draw, with what each one shows.
PANEL_DESCRIPTIONS = {
    "image": (
        "map of the displayed plane with anatomy, overlay and masks; for a "
        "single voxel or slab the anatomy with its outline"
    ),
    "curve": "spectrum of the selected voxel or mean spectrum inside a mask",
    "time": "time course of a spectral point or range, for the voxel and a mask",
}

#: Control groups the browser can build, with what each one sets.
CONTROL_GROUPS = {
    "slices": (
        "variable of a fit result and index sliders for dimensions that are "
        "neither image nor curve"
    ),
    "spectrum": "complex part, frequency unit and line broadening",
    "colour_range": "colour range mode and limits of the map",
    "curve_scale": (
        "y range of the curve (automatic, whole plane or map colours) and "
        "frequency window of the spectrum"
    ),
    "display": (
        "smoothing, min -> NaN, scalebar, grid lines, tooltips and crosshair of the map"
    ),
    "kspace": (
        "image or spatial k-space of the map and curves, with a Gaussian or "
        "Fermi spatial filter"
    ),
    "save": "button that stores the view settings on the scan",
    "layers": (
        "anatomy slice, map opacity above the anatomy and overlay opacity, or "
        "the anatomy slice under a voxel outline (only with anatomy or overlays)"
    ),
    "anatomy": "visibility and colour scale of the anatomy (only with anatomy)",
    "overlay": (
        "overlay visibility, layer, source, slice, colours and colour range "
        "(only with overlays)"
    ),
    "tool": "menu that chooses which tool (plugin) shows its inputs",
    "masks": "drawing, styling and ROI mean of masks (a tool)",
    "fit": "peak table, lineshape and fit run of all voxels (a tool)",
    "curve_fit": (
        "ADC, T2 or T1 fit of all voxels along the curve axis, chosen by the "
        "axis (a tool)"
    ),
    "ideal": (
        "IDEAL separation of chemical components in complex echoes, with "
        "component frequencies and echo spacing (a tool)"
    ),
    "signal_decay": (
        "decay of the CSI raw data over the acquired k-space points with "
        "time-domain and spatial k-space filters and the signal end (a tool, "
        "only in plot_signal_decay)"
    ),
    "time_course": "dimension, spectral range and mask mean of the time panel",
    "parameters": (
        "checkbox that shows the measurement parameters of the scan "
        "(get_parameters) as a filterable table"
    ),
}

#: Panels shown side by side, from left to right.
PANELS = ("image", "curve", "time")

#: Control groups in the top bar, always visible, from left to right.
TOP_BAR = ("slices", "layers")

#: Sections of the collapsible "Plot options" panel: (title, control groups).
OPTION_SECTIONS = (
    ("Spectrum", ("spectrum", "curve_scale")),
    ("Map", ("colour_range", "display")),
    ("k-space", ("kspace",)),
    ("Anatomy", ("anatomy",)),
    ("Overlay", ("overlay",)),
    ("Time course", ("time_course",)),
    ("View settings", ("save",)),
)

#: Control groups below the plots, from left to right.
BELOW_PLOTS = ("tool", "parameters")

#: Tools (plugins) offered in the ``"tool"`` menu, in menu order. Each is a
#: control group that is shown only while it is chosen, right after the menu.
TOOLS = ("masks", "fit", "curve_fit", "ideal", "signal_decay")

#: JavaScript files in ``browser/``, in the order they are joined. Helpers
#: come first; ``viewer.js`` builds the page and exports ``render``.
BROWSER_PARTS = (
    "helpers.js",
    "block.js",
    "image_panel.js",
    "curve_panel.js",
    "controls_processing.js",
    "controls_curve_axes.js",
    "controls_colour_range.js",
    "controls_display.js",
    "controls_anatomy.js",
    "controls_overlay.js",
    "controls_layers.js",
    "controls_masks.js",
    "controls_fit.js",
    "controls_curve_fit.js",
    "controls_ideal.js",
    "controls_signal_decay.js",
    "controls_kspace.js",
    "controls_tool.js",
    "controls_time.js",
    "controls_parameters.js",
    "time_panel.js",
    "viewer.js",
)


def browser_layout(
    panels: Sequence[str] = PANELS,
    top_bar: Sequence[str] = TOP_BAR,
    option_sections: Sequence[tuple[str, Sequence[str]]] = OPTION_SECTIONS,
    below_plots: Sequence[str] = BELOW_PLOTS,
    tools: Sequence[str] = TOOLS,
) -> dict[str, Any]:
    """Return the checked layout sent to the browser.

    Parameters
    ----------
    panels : sequence of str, optional
        Panel names from :data:`PANEL_DESCRIPTIONS`, left to right. At least
        one; each at most once.
    top_bar : sequence of str, optional
        Control group names from :data:`CONTROL_GROUPS` shown above the plots
        at all times, left to right.
    option_sections : sequence of (str, sequence of str), optional
        Titled sections of the collapsible "Plot options" panel, each with
        its control group names. Sections without groups are dropped.
    below_plots : sequence of str, optional
        Control group names shown below the plots, left to right.
    tools : sequence of str, optional
        Control group names offered in the ``"tool"`` menu, in menu order.
        They must not be placed anywhere else, and ``"tool"`` must be placed
        when any tool is given.

    Returns
    -------
    dict
        ``{"panels": [...], "top_bar": [...], "option_sections": [[title,
        [...]], ...], "below_plots": [...], "tools": [...]}`` with plain
        lists, ready for the synchronised ``_layout`` trait.

    Raises
    ------
    ValueError
        If a panel or control group name is unknown or appears twice over all
        places and tools, if no panel is given, or if tools are given without
        the ``"tool"`` menu.

    Examples
    --------
    >>> browser_layout(panels=["image"])["panels"]
    ['image']
    """
    panel_list = [str(name) for name in panels]
    _check_names("panel", panel_list, PANEL_DESCRIPTIONS)
    if not panel_list:
        raise ValueError("browser_layout needs at least one panel, e.g. 'image'.")
    top = [str(name) for name in top_bar]
    sections = [
        [str(title), [str(name) for name in names]] for title, names in option_sections
    ]
    below = [str(name) for name in below_plots]
    tool_list = [str(name) for name in tools]
    placed = top + [name for _, names in sections for name in names] + below
    _check_names("control group", placed + tool_list, CONTROL_GROUPS)
    if tool_list and "tool" not in placed:
        raise ValueError(
            f"browser_layout: tools {tool_list} need the 'tool' control group "
            "in top_bar, option_sections or below_plots; add it or pass tools=[]."
        )
    return {
        "panels": panel_list,
        "top_bar": top,
        "option_sections": [section for section in sections if section[1]],
        "below_plots": below,
        "tools": tool_list,
    }


def _check_names(kind: str, names: list[str], known: dict[str, str]) -> None:
    """Raise ValueError for unknown or repeated layout names."""
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"browser_layout: unknown {kind} {unknown}; choose from {sorted(known)}."
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"browser_layout: {kind} {repeated} listed more than once; "
            "list each at most once."
        )


def browser_script() -> str:
    """Return the browser code of the viewer as one JavaScript module.

    Returns
    -------
    str
        The files of :data:`BROWSER_PARTS` from the ``browser`` folder joined
        in that order, each preceded by a comment with its file name. The
        files are read on every call, so edits show after the widget is
        created again.

    Raises
    ------
    FileNotFoundError
        If a listed part is missing from the installed package.
    """
    folder = resource_files("hypermri.visualization").joinpath("browser")
    parts = []
    for name in BROWSER_PARTS:
        parts.append(f"// ---- browser/{name} ----\n")
        parts.append(folder.joinpath(name).read_text(encoding="utf-8"))
    return "\n".join(parts)
