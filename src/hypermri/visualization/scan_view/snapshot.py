"""Standalone HTML snapshot of a :class:`ScanView`.

The snapshot holds the browser code of the viewer, its style, Plotly.js and
the values the widget has sent to the browser, so it opens in any browser
without Python. Everything the browser does on its own keeps working: voxel
and curve-position selection, hover values, colour range, curve axes and
the display switches. Controls that need Python to compute a new block
(slices, processing, layers, masks and tools) are shown switched off.
"""

from __future__ import annotations

import base64
import html
import json
from importlib.resources import files as resource_files
from typing import TYPE_CHECKING, Any, Literal

from hypermri.visualization.layout import browser_script

if TYPE_CHECKING:
    from hypermri.visualization.scan_view.widget import ScanView

__all__ = ["PLOTLY_JS_SOURCES", "snapshot_html"]

#: Where a snapshot takes Plotly.js from: ``"inline"`` embeds the copy of the
#: installed plotly package (about 4.6 MB, works offline), ``"cdn"`` loads
#: the same version from the jsDelivr CDN (small file, needs the web).
PLOTLY_JS_SOURCES = ("inline", "cdn")

# Traits of every ipywidget; they describe the widget, not the scan.
_WIDGET_TRAITS = frozenset(
    {
        "_dom_classes",
        "_model_module",
        "_model_module_version",
        "_model_name",
        "_view_count",
        "_view_module",
        "_view_module_version",
        "_view_name",
        "_esm",
        "_css",
        "_anywidget_id",
        "layout",
        "tabbable",
        "tooltip",
    }
)


def _plotly_js_text() -> str:
    """Return the Plotly.js bundle of the installed plotly package."""
    path = resource_files("plotly").joinpath("package_data", "plotly.min.js")
    return path.read_text(encoding="utf-8")


def _json_value(value: Any) -> Any:
    """Return a trait value as JSON, with binary buffers as base64 text."""
    if isinstance(value, bytes | bytearray | memoryview):
        return {"base64": base64.b64encode(bytes(value)).decode("ascii")}
    return value


def _snapshot_state(view: ScanView) -> dict[str, Any]:
    """Return the synchronised traits of *view* as the snapshot shows them."""
    state = {
        name: _json_value(getattr(view, name))
        for name in view.trait_names(sync=True)
        if name not in _WIDGET_TRAITS
    }
    # Tools need Python for every action; the snapshot leaves them out.
    state["_layout"] = {**state["_layout"], "tools": [], "snapshot": True}
    state["tool"] = ""
    return state


def _script_text(source: str) -> str:
    """Return JavaScript that cannot end the surrounding ``<script>`` early."""
    return source.replace("</script", "<\\/script")


def snapshot_html(
    view: ScanView, *, plotly_js: Literal["inline", "cdn"] = "inline"
) -> str:
    """Return a standalone HTML page that shows *view* without Python.

    Parameters
    ----------
    view : ScanView
        Viewer whose shown plane, selection, colour range, masks, layers and
        name are saved. The view is read only, not changed.
    plotly_js : {"inline", "cdn"}, optional
        Source of Plotly.js, see :data:`PLOTLY_JS_SOURCES`. Default
        ``"inline"``.

    Returns
    -------
    str
        Complete HTML document (UTF-8). Its size is dominated by Plotly.js
        when ``plotly_js="inline"`` and otherwise by the float32 block of
        the shown plane, stored as base64 text (4/3 of its size in bytes).

    Raises
    ------
    ValueError
        If *plotly_js* is not one of :data:`PLOTLY_JS_SOURCES`.
    """
    if plotly_js not in PLOTLY_JS_SOURCES:
        raise ValueError(
            f"ScanView.to_html: plotly_js must be one of {PLOTLY_JS_SOURCES}, "
            f"not {plotly_js!r}."
        )
    if plotly_js == "inline":
        plotly_tag = f"<script>{_script_text(_plotly_js_text())}</script>"
    else:
        version = view._plotly_version
        plotly_tag = (
            '<script src="https://cdn.jsdelivr.net/npm/'
            f'plotly.js-dist-min@{version}/plotly.min.js"></script>'
        )
    folder = resource_files("hypermri.visualization").joinpath("browser")
    css = folder.joinpath("viewer.css").read_text(encoding="utf-8")
    start = folder.joinpath("snapshot.js").read_text(encoding="utf-8")
    state = json.dumps(_snapshot_state(view), allow_nan=False).replace("</", "<\\/")
    title = html.escape(view.name or "hypermri scan view")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="hypermri ScanView.to_html">
<title>{title}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 16px; background: #ffffff; color: #1f2328; }}
{css}
</style>
{plotly_tag}
</head>
<body>
<div id="hypermri-view"></div>
<script id="hypermri-state" type="application/json">{state}</script>
<script type="module">
{_script_text(browser_script())}
{_script_text(start)}
</script>
</body>
</html>
"""
