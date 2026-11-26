"""Temporal sample selection shared by DCE fit and quality workflows.

Model callers own option parsing, AIF thresholds and any historical index
convention. This module intersects explicit selections without modifying inputs.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .sampling import _coerce_numeric_1d


def _select_fit_samples(
    time_s: np.ndarray,
    *,
    start_index,
    end_index,
    start_time_s,
    end_time_s,
    fit_indices=None,
    fit_times_s=None,
) -> np.ndarray:
    """Intersect absolute index bounds, inclusive times and explicit samples.

    ``time_s`` is a caller-validated 1D array in seconds. Index bounds use
    comparisons against zero-based indices; the end is exclusive. Explicit
    indices retain NumPy indexing, including negative indices. Requested times
    select the nearest sample, with the first index winning a tie. Returns a
    fresh boolean array in input order; no values or settings are mutated.
    """
    mask = np.ones(len(time_s), dtype=bool)
    indices = np.arange(len(time_s))
    mask &= indices >= start_index
    if end_index is not None:
        mask &= indices < end_index
    mask &= time_s >= start_time_s
    mask &= time_s <= end_time_s

    if fit_indices is not None:
        selected = np.zeros(len(time_s), dtype=bool)
        selected[np.asarray(fit_indices, dtype=int)] = True
        mask &= selected

    if fit_times_s is not None:
        selected = np.zeros(len(time_s), dtype=bool)
        for requested_time in np.atleast_1d(fit_times_s):
            selected[int(np.argmin(np.abs(time_s - requested_time)))] = True
        mask &= selected
    return mask


def select_fit_window(
    time_points_s: np.ndarray | xr.DataArray,
    fit_params: dict | None = None,
    *,
    fit_type: str = "patlak",
    aif_signal: np.ndarray | xr.DataArray | None = None,
) -> xr.DataArray:
    """Build a generic labelled DCE window without mutating inputs.

    Parameters
    ----------
    time_points_s : numpy.ndarray or xarray.DataArray
        Acquisition times in seconds, 1D after squeezing. Coordinate order is
        preserved; selections use the numeric sample values.
    fit_params : dict or None, optional
        Intersect absolute ``fit_start_index`` (default 0), exclusive
        ``fit_end_index`` (alias ``fit_stop_index``), inclusive
        ``fit_start_time_s`` and ``fit_end_time_s`` (alias ``fit_stop_time_s``),
        ``fit_indices`` and ``fit_times_s``. Time bounds default to infinity.
        Negative end indices select no samples. Explicit indices follow NumPy
        indexing, including negative indices. Requested times select the nearest
        sample, with the first index winning a tie.
    fit_type : str, default "patlak"
        ``patlak`` and ``sourbron`` apply their AIF threshold when an AIF is
        supplied. Other model keys do not apply an AIF condition. This shared
        window honors all selectors; model-local wrappers retain their explicit
        differences (Tofts negative-end slicing, Patlak/Sourbron selector support).
    aif_signal : numpy.ndarray or xarray.DataArray, optional
        Plasma concentration in mM on the same samples. For Patlak/Sourbron,
        select values strictly above ``aif_threshold`` (default 1e-6/0 mM).

    Returns
    -------
    xarray.DataArray
        Eager boolean ``fit_window_mask`` with dimension ``repetition_idx`` and
        consecutive zero-based index coordinates. The caller may reattach the
        original acquisition coordinates. Selection is deterministic.

    Raises
    ------
    ValueError
        If a squeezed input is not 1D or the applicable AIF length differs.
    IndexError
        If an explicit sample index is outside the time array.

    Examples
    --------
    >>> select_fit_window(np.array([0., 2., 5.]), {"fit_start_index": 1}).values
    array([False,  True,  True])
    """
    fit_params = dict(fit_params or {})
    time_points = _coerce_numeric_1d("time_points_s", time_points_s)
    fit_start_index = int(fit_params.get("fit_start_index", 0) or 0)
    fit_end_index = fit_params.get(
        "fit_end_index", fit_params.get("fit_stop_index", None)
    )
    fit_start_time_s = fit_params.get("fit_start_time_s", -np.inf)
    fit_end_time_s = fit_params.get(
        "fit_end_time_s", fit_params.get("fit_stop_time_s", np.inf)
    )

    mask = _select_fit_samples(
        time_points,
        start_index=fit_start_index,
        end_index=None if fit_end_index is None else int(fit_end_index),
        start_time_s=fit_start_time_s,
        end_time_s=fit_end_time_s,
        fit_indices=fit_params.get("fit_indices"),
        fit_times_s=fit_params.get("fit_times_s"),
    )

    if fit_type in {"patlak", "sourbron"} and aif_signal is not None:
        aif_array = _coerce_numeric_1d("aif_signal", aif_signal)
        if len(aif_array) != len(mask):
            raise ValueError(
                f"{fit_type.capitalize()} fit window requires AIF and time axes with matching length."
            )
        default_threshold = 1e-6 if fit_type == "patlak" else 0.0
        mask &= aif_array > float(fit_params.get("aif_threshold", default_threshold))

    return xr.DataArray(
        mask,
        dims=["repetition_idx"],
        coords={"repetition_idx": np.arange(len(time_points), dtype=int)},
        name="fit_window_mask",
    )
