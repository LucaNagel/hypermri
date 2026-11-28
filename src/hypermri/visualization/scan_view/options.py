"""Option choices, overlay names and small argument checks of the browser viewer."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from hypermri._optional import _import_optional

_import_optional("anywidget")
_import_optional("cmap")


SPECTRAL_DIM = "acquisition_idx"
#: Overlay name under which the fit tool registers its result; its layers
#: are ``fit_<variable>``, for example ``fit_fit_amps``.
FIT_OVERLAY_NAME = "fit"
# Name of the overlay holding the result maps of the curve-fit tool.
CURVE_FIT_OVERLAY_NAME = "curve_fit"
# Name of the overlay holding the component and field-offset maps of IDEAL.
IDEAL_OVERLAY_NAME = "ideal"
# Overlays of the tools; no saved fit may take their names.
_TOOL_OVERLAY_NAMES = (FIT_OVERLAY_NAME, CURVE_FIT_OVERLAY_NAME, IDEAL_OVERLAY_NAME)
# Tools whose fit ScanView.save_fit saves.
_SAVE_FIT_TOOLS = ("spectral", "curve", "ideal")
# Colours of the views of plot_side_by_side (bar above each view) and of
# their slice lines in the other views (show_slices_of). Muted ochre, teal,
# mauve, sage and terracotta: calm on light and dark pages, still told
# apart over the colour maps.
SLICE_LINE_COLOURS = ("#c9a227", "#4a9fb5", "#b5679e", "#6fa35c", "#c97a3d")
SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
SPEC_PART_OPTIONS = ("Magnitude", "Real", "Imaginary", "Phase")
OVERLAY_RANGE_MODES = ("Percentage", "Min/Max")
# As in plot: "Min/Max" returns to the full range when the shown values
# change (complex part, line broadening); "Absolute" keeps its limits.
MAP_RANGE_MODES = ("Percentage", "Min/Max", "Absolute")
# Smoothing of the CSI map: none, or Plotly's "fast" or "best" heatmap
# smoothing (``state.spatial.interpolation`` and ``zsmooth_mode`` in plot).
MAP_SMOOTHING_OPTIONS = ("off", "fast", "best")
# Display switches of the map, as the checkboxes of plot: name -> (state
# section, field). Each only redraws, except the scalebar (Python computes it).
_DISPLAY_SWITCHES = {
    "min_to_nan": ("spatial", "min_to_nan"),
    "show_scalebar": ("spatial", "show_scalebar"),
    "show_colorbar": ("spatial", "show_colorbar"),
    "show_grid_lines": ("global_settings", "show_grid_lines"),
    "show_tooltips": ("global_settings", "show_tooltips"),
    "show_crosshair": ("global_settings", "show_crosshair"),
    "show_overlay_tooltips": ("overlay", "show_tooltips"),
}
SCAN_SETTINGS_CONTEXT = "plot_scan"
# y axis of the curve panel: fitted to the shown curves, the value range of
# the whole plane, or the colour limits of the map (couple_ylims in plot).
CURVE_SCALE_MODES = ("auto", "plane", "map")
SPEC_UNIT_OPTIONS = ("ppm", "Hz")
# Processing of the shown signal, as the processing controls of plot.
# Repetitions: unchanged, mean over a range, or running sum over a range.
REPETITION_MODES = ("off", "average", "cumulative sum")
# Channel combination methods offered by plot's "Method" menu.
CHANNEL_COMBINE_METHODS = ("rss", "sum", "mean", "adaptive")
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


#: Widths of SpatialFilterOptions in cycles/mm, in the order the browser shows them.
_SPATIAL_FILTER_WIDTHS = ("sigma_per_mm", "radius_per_mm", "transition_per_mm")


def _json_values(values: Any) -> list[float | None]:
    """Return values as a JSON list; NaN becomes null, a gap in the Plotly line."""
    return [float(v) if np.isfinite(v) else None for v in np.ravel(values)]


def _as_float(value: Any) -> float:
    """Return a value as float, or NaN when it is no number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _checked_curve_axes(
    settings: dict[str, Any], operation: str
) -> tuple[float, float] | None:
    """Check curve-axis settings and return the frequency window as floats."""
    if settings["scale"] not in CURVE_SCALE_MODES:
        raise ValueError(
            f"{operation}: the curve scale must be one of "
            f"{', '.join(CURVE_SCALE_MODES)}; got {settings['scale']!r}."
        )
    if settings["window_unit"] not in SPEC_UNIT_OPTIONS:
        raise ValueError(
            f"{operation}: the frequency window unit must be one of "
            f"{', '.join(SPEC_UNIT_OPTIONS)}; got {settings['window_unit']!r}."
        )
    window = settings["window"]
    if window is not None:
        limits = [_as_float(value) for value in window]
        if len(limits) != 2 or not np.all(np.isfinite(limits)):
            raise ValueError(
                f"{operation}: the frequency window needs two finite limits; "
                f"got {window!r}."
            )
        if limits[0] == limits[1]:
            raise ValueError(
                f"{operation}: the frequency window needs two different "
                f"limits; got {limits[0]} twice."
            )
        window = tuple(limits)
    return window


def _check_limits(operation: str, vmin: Any, vmax: Any) -> None:
    """Fail on colour limits that are not finite or not increasing."""
    for name, value in (("vmin", vmin), ("vmax", vmax)):
        try:
            finite = bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(
                f"{operation} {name} must be a finite number; got {value!r}."
            )
    if not float(vmin) < float(vmax):
        raise ValueError(f"{operation}: vmin ({vmin!r}) must be below vmax ({vmax!r}).")


def _check_mask_style(color: str, alpha: float) -> None:
    """Raise a ValueError for a colour other than #rrggbb or alpha outside [0, 1]."""
    if not isinstance(color, str) or not _HEX_COLOR.match(color):
        raise ValueError(f"Mask colour must look like '#ff0000'; got {color!r}.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"Mask opacity must lie in [0, 1]; got {alpha!r}.")
