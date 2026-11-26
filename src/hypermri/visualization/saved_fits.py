"""Fits of the browser viewer saved on the plotted object.

A fit run in the browser viewer (spectral fit tool, curve-fit tool or IDEAL
tool) is kept by the view until the next run. Saving it stores a copy under a
chosen name in ``saved_fits`` of the plotted object: the attribute
``scan.saved_fits`` of a scan, or ``data.attrs["saved_fits"]`` of a plotted
DataArray. Nothing is written to disk. A view opened later on the same
object shows the saved fits as overlays.

Examples
--------
>>> import xarray as xr
>>> data = xr.DataArray([[1.0, 2.0]], dims=("read_idx", "phase_idx"))
>>> fit = SavedFit("t2", xr.Dataset({"t2": data}))
>>> save_fit(data, "kidney_t2", fit)
>>> sorted(saved_fits(data))
['kidney_t2']
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import xarray as xr

__all__ = ["FIT_KINDS", "SavedFit", "check_fit_name", "save_fit", "saved_fits"]

#: Kinds of saved fits: the spectral fit tool, the curve-fit models and IDEAL.
FIT_KINDS = ("spectral", "adc", "t2", "t1", "ideal")

# Name of the attribute (scan) or attrs key (DataArray) holding the fits.
_SAVED_FITS = "saved_fits"

# Fit names are overlay names and prefixes of overlay map names.
_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class SavedFit:
    """One fit saved from the browser viewer.

    Attributes
    ----------
    kind : str
        One of :data:`FIT_KINDS`: ``"spectral"`` (fit tool), ``"adc"``,
        ``"t2"``, ``"t1"`` (curve-fit tool) or ``"ideal"`` (IDEAL tool).
    results : xarray.Dataset
        Fit result with the voxel dimensions of the fitted data. Spectral
        fits hold ``fit_freqs`` (Hz), ``fit_t2s`` (s) and ``fit_amps`` over
        ``metabolite``; curve fits hold ``adc`` (mm²/s), ``t2`` or ``t1``
        (ms) and their companions; IDEAL fits hold complex ``fit_amps`` over
        ``metabolite`` and, when fitted, ``field_offset`` (Hz).
    params : xarray.Dataset or None
        Fit options of a spectral fit (the ``fit_params`` of
        :func:`~hypermri.visualization.fit_tool.run_fit`); None for the
        other kinds.
    """

    kind: str
    results: xr.Dataset
    params: xr.Dataset | None = None

    def __post_init__(self) -> None:
        """Check the kind and the Dataset types."""
        if self.kind not in FIT_KINDS:
            raise ValueError(
                f"SavedFit kind must be one of {FIT_KINDS}; got {self.kind!r}."
            )
        if not isinstance(self.results, xr.Dataset):
            raise TypeError("SavedFit results must be an xarray.Dataset.")
        if self.params is not None and not isinstance(self.params, xr.Dataset):
            raise TypeError("SavedFit params must be an xarray.Dataset or None.")


def check_fit_name(name: str) -> str:
    """Return a fit name without surrounding spaces, or raise.

    Parameters
    ----------
    name : str
        Name typed by the user. It must start with a letter and contain only
        letters, digits, ``_`` and ``-``, because it also names the overlay
        and its maps.

    Returns
    -------
    str
        The name without leading and trailing spaces.

    Raises
    ------
    ValueError
        If the name is empty or contains other characters.
    """
    stripped = str(name).strip()
    if not _NAME_PATTERN.fullmatch(stripped):
        raise ValueError(
            f"Fit name {stripped!r} is not valid: start with a letter and use "
            "only letters, digits, '_' and '-'."
        )
    return stripped


def saved_fits(obj: Any) -> dict[str, SavedFit]:
    """Return the fits saved on a plotted object.

    Parameters
    ----------
    obj : object
        Scan or DataArray (or Dataset) shown by the viewer. Not modified.

    Returns
    -------
    dict of str to SavedFit
        The stored dictionary (``obj.saved_fits`` or
        ``obj.attrs["saved_fits"]``), so deleting an entry removes the fit;
        an empty new dictionary when nothing is saved yet or ``obj`` is
        None.
    """
    if obj is None:
        return {}
    if isinstance(obj, xr.DataArray | xr.Dataset):
        stored = obj.attrs.get(_SAVED_FITS)
    else:
        stored = getattr(obj, _SAVED_FITS, None)
    return stored if isinstance(stored, dict) else {}


def save_fit(obj: Any, name: str, fit: SavedFit) -> None:
    """Store a deep copy of a fit on a plotted object under a name.

    A fit already saved under ``name`` is replaced.

    Parameters
    ----------
    obj : object
        Scan or DataArray (or Dataset) shown by the viewer. A scan gets or
        extends the attribute ``saved_fits``; a DataArray or Dataset the
        ``attrs`` entry ``"saved_fits"``. Such an entry is not written by
        ``to_netcdf``; remove it before writing.
    name : str
        Fit name, checked with :func:`check_fit_name`.
    fit : SavedFit
        Fit to store. Its Datasets are copied, so later runs in the viewer
        do not change the saved fit.

    Raises
    ------
    ValueError
        If the name is not valid or ``obj`` is None.
    """
    name = check_fit_name(name)
    if obj is None:
        raise ValueError("save_fit needs the plotted scan or DataArray.")
    copy = SavedFit(
        fit.kind,
        fit.results.copy(deep=True),
        None if fit.params is None else fit.params.copy(deep=True),
    )
    if isinstance(obj, xr.DataArray | xr.Dataset):
        stored = obj.attrs.get(_SAVED_FITS)
        if not isinstance(stored, dict):
            stored = obj.attrs[_SAVED_FITS] = {}
    else:
        stored = getattr(obj, _SAVED_FITS, None)
        if not isinstance(stored, dict):
            stored = {}
            setattr(obj, _SAVED_FITS, stored)
    stored[name] = copy
