"""Spatial overlay inputs, names and value ranges shared by the viewers.

A spatial overlay is a parameter map or second scan drawn semi-transparently
over the shown plane, for example fitted metabolite amplitudes over a CSI
scan. ``plot`` and the browser view accept the same ``overlay`` argument,
name its layers the same way and compute colour limits the same way; this
module owns those three steps. Cutting the overlay to the shown plane belongs
to :meth:`hypermri.visualization.render_data.PlotDataModel.get_render_payload`,
which receives a :class:`SpatialOverlayLayer`.

Layer naming: a Dataset registered under ``name`` offers one layer per data
variable, named ``f"{name}_{variable}"``; a DataArray registered under
``name`` offers one layer called ``name``.

Data sources: an overlay given as a scan object can be shown from another
reconstruction of that scan (raw reconstruction, scanner reconstruction or
DICOM series, :class:`~hypermri.visualization.data_sources.DataSourceCatalog`).
This module chooses the catalog of the shown layer and prepares the chosen
source for both viewers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from hypermri.utils.coordinates.grids import assign_xyz_coordinates, has_xyz_coordinates
from hypermri.visualization.data_sources import DataSourceCatalog
from hypermri.visualization.state import OverlayState

__all__ = [
    "SpatialOverlayLayer",
    "active_overlay_source_catalog",
    "apply_overlay_start_state",
    "default_overlay_variable",
    "load_overlay_source",
    "normalize_overlays",
    "overlay_source_catalogs",
    "overlay_value_range",
    "overlay_variable_names",
    "preferred_fit_variable",
    "prepare_overlay_value",
    "split_overlay_variable",
]

_FIT_AMPLITUDE_VARIABLE = "fit_amps"
_FIT_FREQUENCY_VARIABLE = "fit_freqs"
_T1_MODEL_NAME = "t1_inversion_recovery"
_T1_OVERLAY_CMAP = "lipari"


@dataclass(frozen=True)
class SpatialOverlayLayer:
    """Overlay samples ready to be cut to the shown plane.

    Parameters
    ----------
    name : str
        Layer name used as key of ``RenderPayload.spatial_overlays``, for
        example ``"fit_fit_amps"``; see the module docstring for the naming
        rule.
    data : xarray.DataArray
        Overlay samples with labelled spatial dimensions (read_idx, phase_idx,
        slice_idx) and optionally acquisition_idx. Metabolite and repetition
        reductions and the spectral display preprocessing are already applied
        (:meth:`~hypermri.visualization.render_data.PlotDataModel.reduce_spatial_overlay`
        and :meth:`~hypermri.visualization.render_data.PlotDataModel.preprocess_spatial_overlay`).
        Scanner x/y/z coordinates in mm place the layer in scanner
        coordinate mode. Units are those of the overlay; not mutated.
    value_range : tuple of float or None
        Finite (minimum, maximum) over the whole of ``data`` in overlay units,
        magnitudes for complex samples (:func:`overlay_value_range`). Percentage
        colour limits are fractions of this range; None leaves the colour
        limits unset.
    """

    name: str
    data: xr.DataArray
    value_range: tuple[float, float] | None


def _source_xarray(source: Any) -> xr.DataArray | xr.Dataset | None:
    """Return the labelled array held by a scan object, or None."""
    if isinstance(source, xr.DataArray | xr.Dataset):
        return source
    for attr_name in ("primary_data", "data", "scanner_data"):
        try:
            candidate = getattr(source, attr_name, None)
        except Exception:
            continue
        if isinstance(candidate, xr.DataArray):
            return candidate
    return None


def _matching_xyz_coordinates(
    data: xr.DataArray | xr.Dataset,
    reference: xr.DataArray | xr.Dataset | None,
) -> dict[str, xr.DataArray]:
    """Return the reference x/y/z coordinates when data shares its index grid."""
    if reference is None or not has_xyz_coordinates(reference):
        return {}

    coord_updates: dict[str, xr.DataArray] = {}
    spatial_dims: set[str] = set()
    for coord_name in ("x", "y", "z"):
        coord = reference.coords[coord_name]
        if not all(
            dim in data.dims and int(data.sizes[dim]) == int(coord.sizes[dim])
            for dim in coord.dims
        ):
            return {}
        spatial_dims.update(coord.dims)
        coord_updates[coord_name] = coord.copy(deep=False)

    # Matching sizes alone do not establish shared geometry. Derived maps,
    # including fit results, retain the source index coordinates, so those
    # coordinates must agree before the reference x/y/z are inherited.
    for dim in spatial_dims:
        if dim not in data.coords or dim not in reference.coords:
            return {}
        target_coord = data.coords[dim]
        reference_coord = reference.coords[dim]
        if target_coord.dims != reference_coord.dims:
            return {}
        try:
            if not np.array_equal(
                np.asarray(target_coord.values),
                np.asarray(reference_coord.values),
                equal_nan=True,
            ):
                return {}
        except (TypeError, ValueError):
            if not target_coord.equals(reference_coord):
                return {}

    return coord_updates


def prepare_overlay_value(
    source: Any,
    coordinate_mode: str,
    *,
    geometry_reference: xr.DataArray | xr.Dataset | None = None,
    scan: Any | None = None,
) -> Any:
    """Return the labelled overlay data, with scanner coordinates when needed.

    Parameters
    ----------
    source : Any
        DataArray, Dataset or scan object holding one (``primary_data``,
        ``data``, ``scanner_data`` or ``data``).
    coordinate_mode : str
        ``"xyz"`` adds scanner x/y/z coordinates in mm when the data has none;
        other modes leave the data as it is.
    geometry_reference : xarray.DataArray or xarray.Dataset, optional
        Shown scan. Its x/y/z coordinates are reused when the overlay has the
        same index grid and index coordinate values, as for fit results of
        that scan.
    scan : object, optional
        Scan whose geometry gives scanner coordinates to an explicit
        DataArray ``source``, for example one data source of that scan. A
        scan object passed as ``source`` supplies its own geometry.

    Returns
    -------
    Any
        The labelled overlay data, or source unchanged when it holds no
        labelled array. Values, units and attributes are not changed; source
        is not mutated.
    """
    data = _source_xarray(source)
    if data is None:
        return source
    if str(coordinate_mode).strip().lower() == "xyz" and not has_xyz_coordinates(data):
        geometry_obj = (
            source if not isinstance(source, xr.DataArray | xr.Dataset) else scan
        )
        if geometry_obj is not None and isinstance(data, xr.DataArray):
            data = assign_xyz_coordinates(data, scan=geometry_obj)
        if not has_xyz_coordinates(data):
            xyz_coords = _matching_xyz_coordinates(data, geometry_reference)
            if xyz_coords:
                data = data.assign_coords(xyz_coords)
    return data


def normalize_overlays(
    overlay: Any,
    coordinate_mode: str,
    *,
    geometry_reference: xr.DataArray | xr.Dataset | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Register overlay input under names, as ``plot`` and ``plot_scan`` do.

    Parameters
    ----------
    overlay : Any
        None; one DataArray, Dataset or scan object, registered as
        ``"overlay"``; or a dict mapping names to such values.
    coordinate_mode : str
        Coordinate mode of the viewer, see :func:`prepare_overlay_value`.
    geometry_reference : xarray.DataArray or xarray.Dataset, optional
        Shown scan, see :func:`prepare_overlay_value`.

    Returns
    -------
    overlays : dict of str to Any
        Labelled overlay data by registered name.
    source_objects : dict of str to Any
        The scan objects among the inputs by registered name, for views that
        reconstruct them again with other options.

    Examples
    --------
    >>> overlays, _ = normalize_overlays(fit_results, "xyz", geometry_reference=csi)  # doctest: +SKIP
    >>> list(overlays)  # doctest: +SKIP
    ['overlay']
    """
    if overlay is None:
        return {}, {}
    entries = overlay if isinstance(overlay, dict) else {"overlay": overlay}
    overlays: dict[str, Any] = {}
    source_objects: dict[str, Any] = {}
    for name, value in entries.items():
        if not isinstance(value, xr.DataArray | xr.Dataset):
            source_objects[name] = value
        overlays[name] = prepare_overlay_value(
            value, coordinate_mode, geometry_reference=geometry_reference
        )
    return overlays, source_objects


def overlay_variable_names(overlays: dict[str, Any]) -> list[str]:
    """Return the layer names offered by registered overlays.

    Parameters
    ----------
    overlays : dict of str to Any
        Registered overlays (:func:`normalize_overlays`). Entries that are
        neither DataArray nor Dataset offer no layer.

    Returns
    -------
    list of str
        Layer names in registration and data-variable order.
    """
    names: list[str] = []
    for name, data in overlays.items():
        if isinstance(data, xr.Dataset):
            names.extend(f"{name}_{variable}" for variable in data.data_vars)
        elif isinstance(data, xr.DataArray):
            names.append(name)
    return names


def split_overlay_variable(
    overlays: dict[str, Any], layer_name: str | None
) -> tuple[str | None, str | None]:
    """Split a layer name into registered overlay name and Dataset variable.

    Parameters
    ----------
    overlays : dict of str to Any
        Registered overlays.
    layer_name : str or None
        Layer name, see the module docstring.

    Returns
    -------
    overlay_name : str or None
        Registered name the layer belongs to; the longest matching name wins
        so that names containing underscores resolve correctly. None when no
        registered name matches.
    variable : str or None
        Dataset variable, or None for a DataArray overlay or no match.
    """
    if not layer_name:
        return None, None
    for name in sorted(overlays, key=len, reverse=True):
        if layer_name == name:
            return name, None
        if layer_name.startswith(f"{name}_"):
            return name, layer_name[len(name) + 1 :]
    return None, None


def preferred_fit_variable(variables: Any, *, first: bool = True) -> str | None:
    """Return the fit-result variable shown first.

    Fitted amplitudes (``fit_amps``) are preferred, then fitted frequencies
    (``fit_freqs``). Both the overlay layers and :func:`plot_scan` with a fit
    result as data use this order.

    Parameters
    ----------
    variables : iterable of str
        Variable names in their order, for example ``dataset.data_vars``.
    first : bool, optional
        Whether the first name is returned when neither preferred variable is
        present. Default True.

    Returns
    -------
    str or None
        Chosen name; None when ``variables`` is empty, or when no preferred
        variable is present and ``first`` is False.
    """
    names = [str(name) for name in variables]
    for preferred in (_FIT_AMPLITUDE_VARIABLE, _FIT_FREQUENCY_VARIABLE):
        if preferred in names:
            return preferred
    return names[0] if first and names else None


def default_overlay_variable(overlays: dict[str, Any]) -> str | None:
    """Return the layer shown first.

    Fitted amplitudes are preferred, then fitted frequencies, then the first
    layer offered (:func:`overlay_variable_names`).

    Parameters
    ----------
    overlays : dict of str to Any
        Registered overlays.

    Returns
    -------
    str or None
        Layer name, or None when no overlay offers a layer.
    """
    for name, data in overlays.items():
        if isinstance(data, xr.Dataset):
            preferred = preferred_fit_variable(data.data_vars, first=False)
            if preferred is not None:
                return f"{name}_{preferred}"
    names = overlay_variable_names(overlays)
    return names[0] if names else None


def apply_overlay_start_state(
    overlay_state: OverlayState, overlays: dict[str, Any]
) -> None:
    """Choose the first overlay layer when none is selected yet.

    When ``overlay_state.active_variable`` is unset, the layer of
    :func:`default_overlay_variable` is selected; when it belongs to a T1
    inversion-recovery fit result (Dataset attribute ``model_name``), the
    overlay colormap becomes ``"lipari"``. A selected layer, for example
    from saved settings, is kept unchanged.

    Parameters
    ----------
    overlay_state : OverlayState
        Overlay display state; ``active_variable`` and ``cmap`` may be set in
        place.
    overlays : dict of str to Any
        Registered overlays (:func:`normalize_overlays`).

    Returns
    -------
    None
        The state is updated in place.
    """
    if overlay_state.active_variable or not overlays:
        return
    overlay_state.active_variable = default_overlay_variable(overlays)
    overlay_name, _ = split_overlay_variable(overlays, overlay_state.active_variable)
    shown = overlays.get(overlay_name)
    if (
        isinstance(shown, xr.Dataset)
        and shown.attrs.get("model_name") == _T1_MODEL_NAME
    ):
        overlay_state.cmap = _T1_OVERLAY_CMAP


def overlay_value_range(data: xr.DataArray) -> tuple[float, float] | None:
    """Return the finite value range of an overlay.

    Parameters
    ----------
    data : xarray.DataArray
        Overlay samples after reduction and display preprocessing. Complex
        samples contribute their magnitude. Not mutated.

    Returns
    -------
    tuple of float or None
        (minimum, maximum) in overlay units, or None when no sample is finite.
        The whole array is read eagerly.
    """
    values = np.asarray(data.values)
    if np.iscomplexobj(values):
        values = np.abs(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def overlay_source_catalogs(
    overlays: dict[str, Any],
    source_objects: dict[str, Any],
    *,
    requested: str | None,
    coordinate_mode: str,
    geometry_reference: xr.DataArray | xr.Dataset | None = None,
) -> tuple[dict[str, Any], dict[str, DataSourceCatalog]]:
    """Describe the data sources of overlays given as scans and apply a choice.

    Parameters
    ----------
    overlays : dict of str to Any
        Registered overlays (:func:`normalize_overlays`). Not mutated.
    source_objects : dict of str to Any
        Scan objects among the overlay inputs by registered name, as returned
        by :func:`normalize_overlays`.
    requested : str or None
        Source key to show, for example ``state.overlay.data_source`` from
        saved settings (``"data"``, ``"scanner_data"`` or ``"dicom"``).
        Overlays whose scan does not offer it keep their current data.
    coordinate_mode : str
        Coordinate mode of the viewer, see :func:`prepare_overlay_value`.
    geometry_reference : xarray.DataArray or xarray.Dataset, optional
        Shown scan, see :func:`prepare_overlay_value`.

    Returns
    -------
    overlays : dict of str to Any
        A new dict of registered overlays with the requested source loaded
        where available.
    catalogs : dict of str to DataSourceCatalog
        Catalog per registered name of a scan overlay, with ``current_key``
        set to the shown source. A DICOM series is read only when it is
        requested.
    """
    shown = dict(overlays)
    catalogs: dict[str, DataSourceCatalog] = {}
    for name, source_obj in source_objects.items():
        catalog = DataSourceCatalog(source_obj, shown.get(name))
        if requested in catalog.keys and requested != catalog.current_key:
            shown[name] = load_overlay_source(
                catalog,
                requested,
                coordinate_mode,
                geometry_reference=geometry_reference,
            )
        catalogs[name] = catalog
    return shown, catalogs


def active_overlay_source_catalog(
    overlays: dict[str, Any],
    catalogs: dict[str, DataSourceCatalog],
    layer_name: str | None,
) -> tuple[str | None, DataSourceCatalog | None]:
    """Return the scan overlay whose data source the source control changes.

    Parameters
    ----------
    overlays : dict of str to Any
        Registered overlays.
    catalogs : dict of str to DataSourceCatalog
        Catalogs of the scan overlays (:func:`overlay_source_catalogs`).
    layer_name : str or None
        Shown layer, see the module docstring.

    Returns
    -------
    name : str or None
        Registered name of the scan overlay: the one the shown layer belongs
        to, otherwise the only scan overlay. None when neither applies.
    catalog : DataSourceCatalog or None
        Its catalog, or None.
    """
    overlay_name, _ = split_overlay_variable(overlays, layer_name)
    if overlay_name in catalogs:
        return overlay_name, catalogs[overlay_name]
    if len(catalogs) == 1:
        return next(iter(catalogs.items()))
    return None, None


def load_overlay_source(
    catalog: DataSourceCatalog,
    key: str,
    coordinate_mode: str,
    *,
    geometry_reference: xr.DataArray | xr.Dataset | None = None,
) -> xr.DataArray | xr.Dataset:
    """Load one data source of a scan overlay, ready to be registered.

    Parameters
    ----------
    catalog : DataSourceCatalog
        Catalog of the scan overlay; ``current_key`` is set to ``key`` on
        success.
    key : str
        Source key from ``catalog.keys``.
    coordinate_mode : str
        Coordinate mode of the viewer. In ``"xyz"`` the source receives
        scanner x/y/z coordinates in mm from the scan's geometry, see
        :func:`prepare_overlay_value`.
    geometry_reference : xarray.DataArray or xarray.Dataset, optional
        Shown scan, see :func:`prepare_overlay_value`.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The source data. Values are not changed; the cached source is not
        mutated. A DICOM series is read on first use.

    Raises
    ------
    KeyError
        If the catalog has no source ``key``.
    ValueError
        If the source cannot be loaded.
    """
    data = catalog.load(key)
    prepared = prepare_overlay_value(
        data,
        coordinate_mode,
        geometry_reference=geometry_reference,
        scan=catalog.source_obj,
    )
    catalog.current_key = key
    return prepared
