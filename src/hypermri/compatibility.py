"""
Backward compatibility utilities for HyperMRI.

This module provides utilities to help migrate from the old variable naming
conventions to the new standardized ones while maintaining backward compatibility.
"""

from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np
import xarray as xr

XarrayObject = TypeVar("XarrayObject", xr.DataArray, xr.Dataset)


# Common legacy mappings
COMMON_DATA_ARRAY_MAPPINGS = {
    "seq2d": "scanner_data",
    "reco_data": "scanner_data",
    "kspace_xarray": "kspace_array",
}

COMMON_PATH_MAPPINGS = {
    # Most path attributes should just use 'path'
}


def _frequency_axis_key(unit: str) -> str:
    """Return the canonical attribute and coordinate name for ``unit``."""
    unit_norm = str(unit).strip().lower()
    if unit_norm not in {"hz", "ppm"}:
        raise ValueError(
            f"Unsupported frequency unit '{unit}'. Expected one of ['Hz', 'ppm']."
        )
    return f"freq_axis_{unit_norm}"


def _frequency_unit_label(unit: str) -> str:
    """Return the conventional spelling ``'Hz'`` or ``'ppm'`` of ``unit``."""
    return "Hz" if str(unit).strip().lower() == "hz" else "ppm"


def resolve_frequency_axis(data: Mapping | Any | None, unit: str) -> Any:
    """Return the frequency axis stored on a result.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset or Mapping or None
        Result whose ``coords`` and ``attrs`` are inspected, or a plain
        attribute mapping.
    unit : {'Hz', 'ppm'}
        Frequency unit to resolve, case-insensitive. The axis is read from
        ``freq_axis_hz`` or ``freq_axis_ppm``.

    Returns
    -------
    Any
        The frequency axis in ``unit``: the values of the coordinate of that
        name (shape ``(n_acquisition,)`` along ``acquisition_idx``), else the
        attribute of that name, else None.

    Raises
    ------
    ValueError
        If ``unit`` is not ``'Hz'`` or ``'ppm'``.

    Notes
    -----
    The input is not modified. ``assign_frequency_axis`` stores the axis as a
    coordinate when the result has a matching ``acquisition_idx`` dimension
    and as an attribute otherwise, so only one of the two exists.
    """
    if data is None:
        return None
    key = _frequency_axis_key(unit)
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        if key in data.coords:
            return data.coords[key].values
        return data.attrs.get(key, None)
    if isinstance(data, Mapping):
        return data.get(key, None)
    return None


def assign_frequency_axis(
    data: XarrayObject, freq_axis: Any, unit: str
) -> XarrayObject:
    """Store a frequency axis once on a result.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Result that receives the axis.
    freq_axis : array_like or None
        Frequency axis in ``unit``. None returns ``data`` unchanged.
    unit : {'Hz', 'ppm'}
        Frequency unit, case-insensitive; selects the name ``freq_axis_hz``
        or ``freq_axis_ppm``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        When ``data`` has an ``acquisition_idx`` dimension and ``freq_axis``
        is one-dimensional with that length, a new object with the
        coordinate ``freq_axis_hz`` or ``freq_axis_ppm`` along
        ``acquisition_idx`` (coordinate attribute ``unit``) and without an
        attribute of the same name. Otherwise ``data`` itself, with the axis
        written to ``data.attrs`` in place; this keeps calibration axes on
        results without that dimension (for example fit-result datasets) and
        acquisition axes whose length differs from the processed samples.

    Raises
    ------
    ValueError
        If ``unit`` is not ``'Hz'`` or ``'ppm'``.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> fid = xr.DataArray(np.zeros(4), dims=["acquisition_idx"])
    >>> fid = assign_frequency_axis(fid, np.arange(4.0), "Hz")
    >>> "freq_axis_hz" in fid.coords, "freq_axis_hz" in fid.attrs
    (True, False)
    """
    if freq_axis is None:
        return data
    key = _frequency_axis_key(unit)
    values = np.asarray(freq_axis)
    if (
        "acquisition_idx" in data.dims
        and values.ndim == 1
        and values.size == data.sizes["acquisition_idx"]
    ):
        data = data.assign_coords(
            {key: ("acquisition_idx", values, {"unit": _frequency_unit_label(unit)})}
        )
        data.attrs.pop(key, None)
        return data
    data.attrs[key] = freq_axis
    return data


def migrate_object_attributes(obj, mappings: dict = None):
    """
    Migrate an object's attributes from legacy names to standardized names.

    Parameters
    ----------
    obj : object
        Object to migrate
    mappings : dict, optional
        Custom attribute mappings. If None, uses common mappings.
    """
    if mappings is None:
        mappings = COMMON_DATA_ARRAY_MAPPINGS

    for old_attr, new_attr in mappings.items():
        if hasattr(obj, old_attr) and not hasattr(obj, new_attr):
            # Copy legacy attribute to new name
            legacy_value = getattr(obj, old_attr)
            if legacy_value is not None:
                setattr(obj, new_attr, legacy_value)
                # Optionally remove the old attribute or leave it for compatibility
                # delattr(obj, old_attr)
