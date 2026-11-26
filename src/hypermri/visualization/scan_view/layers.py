"""Image layers of the browser viewer: map, anatomy, outline and overlay as browser payloads.

Pure functions of prepared payloads; they draw nothing and make no scientific decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from hypermri._optional import _import_optional
from hypermri.visualization.geometry import (
    display_voxel_centres,
)
from hypermri.visualization.spatial_overlay import (
    SpatialOverlayLayer,
    overlay_variable_names,
    split_overlay_variable,
)
from hypermri.visualization.volume_outline import (
    AnatomyOutline,
)

_import_optional("anywidget", "visualization")
_import_optional("cmap", "visualization")

from cmap import Color, Colormap  # noqa: E402

from hypermri.visualization.scan_view.options import (  # noqa: E402
    SPECTRAL_DIM,
)


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    """Return the finite (minimum, maximum) of displayed values.

    Matches ``finiteRange`` in ``browser/helpers.js``: no finite value gives (0, 1),
    a constant plane is widened by 0.5 on each side.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        return low - 0.5, high + 0.5
    return low, high


def _cell_edges(centres: np.ndarray, first: float, last: float) -> list[float]:
    """Return the outer edges of the first and last displayed cell.

    Parameters
    ----------
    centres : numpy.ndarray
        Cell centres along one displayed axis, in display order.
    first, last : float
        Fallback edges from the payload extent, used for a single cell.

    Returns
    -------
    list of float
        ``[edge before centres[0], edge after centres[-1]]`` in axis units.
    """
    if centres.size < 2:
        return [float(first), float(last)]
    step = float(centres[1] - centres[0])
    return [float(centres[0] - step / 2), float(centres[-1] + step / 2)]


def _widened_range(edges: list[float], first: float, last: float) -> list[float]:
    """Widen an axis range to include ``first`` and ``last``, keeping its direction.

    Parameters
    ----------
    edges : list of float
        Current ``[start, end]`` range of a displayed axis; ``start > end``
        means a reversed axis.
    first, last : float
        Outer edges of another layer along the same axis, in any order.

    Returns
    -------
    list of float
        Range covering both, ordered like ``edges``.
    """
    low = min(edges[0], edges[-1], first, last)
    high = max(edges[0], edges[-1], first, last)
    return [low, high] if edges[0] <= edges[-1] else [high, low]


def _image_layer(
    values: np.ndarray,
    extent: Sequence[float],
    cmap: str,
    clim: Sequence[float] | None,
    *,
    interpolate: bool,
    warning: str = "",
) -> tuple[bytes, dict[str, Any]]:
    """Return an image drawn under the map as float32 bytes and its description.

    The description holds shape, voxel centres, extent, Plotly colour scale,
    colour limits (the finite value range when ``clim`` is None) and
    smoothing, as the browser image panel draws it. ``value_range`` is the
    finite ``[minimum, maximum]`` of ``values`` (None without finite values),
    from which the browser computes percentage limits.
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    n_rows, n_columns = values.shape[:2]
    x, y = display_voxel_centres(list(extent), (n_rows, n_columns))
    limits = np.asarray(
        clim if clim is not None else (np.nanmin(values), np.nanmax(values)),
        dtype=float,
    ).reshape(-1)
    finite = bool(limits.size >= 2 and np.all(np.isfinite(limits[:2])))
    has_values = bool(np.isfinite(values).any())
    return values.tobytes(), {
        "shape": [int(n_rows), int(n_columns)],
        "x": np.asarray(x, dtype=float).tolist(),
        "y": np.asarray(y, dtype=float).tolist(),
        "extent": [float(value) for value in extent],
        "colorscale": Colormap(cmap).to_plotly(),
        "zmin": float(limits[0]) if finite else None,
        "zmax": float(limits[1]) if finite else None,
        "value_range": (
            [float(np.nanmin(values)), float(np.nanmax(values))] if has_values else None
        ),
        "zsmooth": "best" if interpolate else False,
        "warning": warning,
    }


def _scalebar_view(scalebar: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the scalebar of ``geometry._build_spatial_scalebar`` for the browser.

    None (no scalebar) when it is switched off or the axes have no physical
    voxel spacing. Matplotlib colour names such as ``"w"`` become hex colours.
    """
    if not scalebar:
        return None
    return {
        "length": float(scalebar["length"]),
        "label": str(scalebar["label"]),
        "color": Color(scalebar.get("color") or "black").hex,
        "location": str(scalebar.get("location", "lower right")).lower(),
        "fontsize": int(scalebar.get("fontsize", 12)),
    }


def _anatomy_layer(payload: Any) -> tuple[bytes, dict[str, Any]]:
    """Return the anatomy image under the CSI map as bytes and description.

    The image, its extent, colour limits, colormap and smoothing are taken
    unchanged from the payload, as in ``plot``. The opacity of the CSI map
    above it is kept in ``ScanView._anatomy_opacity``. Without a background
    both results are empty.
    """
    background = payload.spatial_background
    if background is None or not payload.background_extent:
        return b"", {}
    return _image_layer(
        background,
        payload.background_extent,
        payload.background_cmap,
        payload.background_clim,
        interpolate=bool(payload.metadata.get("background_interpolation")),
        warning=str(payload.metadata.get("anatomy_slice_warning") or ""),
    )


def _outline_layer(outline: AnatomyOutline, kind: str) -> tuple[bytes, dict, dict]:
    """Return the anatomy plane under a volume outline and both descriptions.

    Returns the anatomy bytes, its layer description (as
    :func:`_image_layer`) and the outline description: closed polygon ``x``
    and ``y`` in axis units, the anatomy slice and the axis ranges and
    labels of the plane.
    """
    warning = (
        "" if outline.outline_x else f"The {kind} does not cross this anatomy slice."
    )
    anatomy, layer = _image_layer(
        outline.image,
        outline.extent,
        outline.cmap,
        outline.clim,
        interpolate=outline.interpolate,
        warning=warning,
    )
    x0, x1, y0, y1 = outline.extent
    y_range = [y1, y0] if outline.origin == "upper" else [y0, y1]
    return (
        anatomy,
        layer,
        {
            "kind": kind,
            "x": outline.outline_x,
            "y": outline.outline_y,
            "slice_dim": outline.slice_dim,
            "slice_index": outline.slice_index,
            "n_slices": outline.n_slices,
            "x_range": [x0, x1],
            "y_range": y_range,
            "x_label": outline.x_label,
            "y_label": outline.y_label,
        },
    )


def _overlay_source(overlays: dict[str, Any], variable: str) -> xr.DataArray:
    """Return the registered samples of one overlay layer before any reduction."""
    overlay_name, data_variable = split_overlay_variable(overlays, variable)
    data = overlays.get(overlay_name)
    if isinstance(data, xr.Dataset) and data_variable in data.data_vars:
        return data[data_variable]
    if isinstance(data, xr.DataArray) and data_variable is None:
        return data
    raise ValueError(
        f"ScanView overlay: unknown layer {variable!r}; choose one of "
        f"{overlay_variable_names(overlays)}."
    )


def _drawable_overlay_names(overlays: dict[str, Any]) -> list[str]:
    """Return the overlay layers the browser view can draw: maps without spectra."""
    return [
        name
        for name in overlay_variable_names(overlays)
        if SPECTRAL_DIM not in _overlay_source(overlays, name).dims
    ]


def _overlay_layer_view(
    payload: Any, layer: SpatialOverlayLayer | None
) -> tuple[bytes, dict[str, Any]]:
    """Return the overlay plane as float32 bytes and its drawing description.

    The plane, its extent (the map extent when the overlay has no own
    scanner coordinates), colour limits, colormap and colour-bar label are
    taken unchanged from the payload, as in ``plot``. The opacity is kept in
    ``ScanView._overlay_opacity``. Without a plane, for example when no layer
    is selected or the overlay does not cross the shown plane and is hidden,
    both results are empty.
    """
    if layer is None or layer.name not in payload.spatial_overlays:
        return b"", {}
    values = np.ascontiguousarray(
        payload.spatial_overlays[layer.name], dtype=np.float32
    )
    n_rows, n_columns = values.shape[:2]
    extent = payload.spatial_overlay_extents.get(layer.name, payload.spatial_extent)
    x, y = display_voxel_centres(extent, (n_rows, n_columns))
    limits = payload.overlay_clim
    finite = limits is not None and all(np.isfinite(limits))
    return values.tobytes(), {
        "name": layer.name,
        "shape": [int(n_rows), int(n_columns)],
        "x": np.asarray(x, dtype=float).tolist(),
        "y": np.asarray(y, dtype=float).tolist(),
        "colorscale": Colormap(payload.overlay_cmap).to_plotly(),
        "zmin": float(limits[0]) if finite else None,
        "zmax": float(limits[1]) if finite else None,
        "label": str(payload.labels.get("overlay_colorbar") or layer.name),
        "warning": str(payload.metadata.get("overlay_slice_warning") or ""),
    }
