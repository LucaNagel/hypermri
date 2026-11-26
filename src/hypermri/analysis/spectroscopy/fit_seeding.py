"""Start values of a spectral fit taken from an earlier fit.

After a first fit, the fitted frequency and T2* of each voxel are often better
start values than one typed value for all voxels. :func:`seed_start_values`
copies them, separately for frequency and T2*, into prepared fit parameters
of :func:`~hypermri.analysis.fitting.spectroscopy.def_fit_params`.
:func:`apply_t2_windows` then optionally narrows the T2* bounds of a line to a
window around each voxel's start value.

Frequencies are in Hz and T2* values in s, as in the fit parameters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _describes_voigt_fit,
    _update_frequency_bounds,
)

__all__ = ["apply_t2_windows", "fit_results_can_seed", "seed_start_values"]

# Smallest T2* bound in s of a T2* window, so the lower bound stays positive.
_MIN_T2_BOUND_S = 1e-6


def fit_results_can_seed(fit_results: xr.Dataset, target: object) -> bool:
    """Return whether earlier fit results may seed a fit of ``target``.

    Frequencies and T2* seed a fit of the same lineshape, and Lorentz results
    also seed a Voigt fit: the Lorentz T2* (which then includes all line
    broadening) is a short but usable start for the Voigt Lorentz T2*. Voigt
    results do not seed a Lorentz fit, because their Lorentz T2* leaves out
    the Gaussian broadening.

    Parameters
    ----------
    fit_results : xarray.Dataset
        Earlier fit results; their attributes or a ``fit_gauss_fwhm_hz`` map
        tell the lineshape.
    target : xarray.Dataset or mapping
        Fit parameters or fit options of the new fit.

    Returns
    -------
    bool
        True when the results may seed the new fit.
    """
    return _describes_voigt_fit(target) or not _describes_voigt_fit(fit_results)


def seed_start_values(
    fit_params: xr.Dataset,
    fit_results: xr.Dataset | None,
    *,
    metabs: Sequence[str],
    seed_freq: Sequence[bool],
    seed_t2: Sequence[bool],
    source_metabs: Sequence[str] | None = None,
    t2_windows_s: Sequence[float | None] | None = None,
) -> tuple[xr.Dataset, str]:
    """Replace start frequency and T2* of chosen lines by earlier fit values.

    For line ``i`` of ``metabs`` with ``seed_freq[i]``, every voxel's
    ``initial_freqs_hz`` becomes the earlier ``fit_freqs`` of that line; with
    ``seed_t2[i]``, ``initial_t2s_s`` becomes the earlier ``fit_t2s``. The
    earlier line is the one of the same name, else ``source_metabs[i]``.
    Non-finite earlier values, and T2* values that are not positive, keep the
    typed start of that voxel. For a line without a T2* window, a seeded T2*
    outside the line's ``[min_t2_s, max_t2_s]`` is moved to the nearer bound
    (so the bounds can be narrowed after a first fit); a line with a window
    gets its bounds around the unclipped start from :func:`apply_t2_windows`.
    Frequency bounds are rebuilt from the new starts.

    Parameters
    ----------
    fit_params : xarray.Dataset
        Prepared fit parameters with ``initial_freqs_hz`` (Hz) and
        ``initial_t2s_s`` (s) over the voxel dimensions and ``metabolite``.
        Not modified.
    fit_results : xarray.Dataset or None
        Earlier fit results with ``fit_freqs`` (Hz) and ``fit_t2s`` (s) over
        ``metabolite`` and voxel dimensions that equal those of
        ``fit_params`` or have size 1. None means no earlier fit.
    metabs : sequence of str
        Line names of the new fit, in row order.
    seed_freq, seed_t2 : sequence of bool
        Per line: take the start frequency or the start T2* from the earlier
        fit. Missing entries count as False.
    source_metabs : sequence of str, optional
        Per line: name of the earlier line to use when no earlier line has
        the new name.
    t2_windows_s : sequence of float or None, optional
        Per line: full width in s of the T2* window, or None. A line with a
        window keeps its seeded T2* unclipped.

    Returns
    -------
    fit_params : xarray.Dataset
        New parameters, or ``fit_params`` itself when nothing was seeded.
    note : str
        Sentence(s) naming the seeded lines, clipped T2* starts, lines
        without an earlier value, or why nothing was seeded; empty when no
        line asked for seeding.
    """
    flags = [
        (bool(freq), bool(t2))
        for freq, t2 in zip(
            _padded(seed_freq, metabs), _padded(seed_t2, metabs), strict=True
        )
    ]
    if not any(freq or t2 for freq, t2 in flags):
        return fit_params, ""
    if fit_results is None:
        return fit_params, "No earlier fit to take start values from. "
    if not fit_results_can_seed(fit_results, fit_params):
        return (
            fit_params,
            "The earlier fit is a Voigt fit, so its T2* values cannot seed this "
            "Lorentz fit; start values from the fit are skipped. ",
        )
    if (
        "fit_freqs" not in fit_results
        or "fit_t2s" not in fit_results
        or "metabolite" not in fit_results.coords
        or "metabolite" not in fit_params.coords
    ):
        return (
            fit_params,
            "The earlier fit has no fit_freqs/fit_t2s per metabolite; start "
            "values from the fit are skipped. ",
        )

    sources = list(source_metabs or [])
    windows = list(t2_windows_s or [])
    earlier = {str(name) for name in fit_results.coords["metabolite"].values}
    current = {str(name) for name in fit_params.coords["metabolite"].values}
    updated = fit_params.copy(deep=True)
    seeded: list[str] = []
    missing: list[str] = []
    clipped: list[str] = []
    for index, (metab, (freq, t2)) in enumerate(zip(metabs, flags, strict=True)):
        metab = str(metab)
        if not (freq or t2) or metab not in current:
            continue
        candidates = [metab] + ([str(sources[index])] if index < len(sources) else [])
        source = next((name for name in candidates if name in earlier), None)
        if source is None:
            missing.append(metab)
            continue
        line = {"metabolite": metab}
        pairs = []
        if freq:
            pairs.append(("fit_freqs", "initial_freqs_hz"))
        if t2:
            pairs.append(("fit_t2s", "initial_t2s_s"))
        aligned = {
            start: _aligned_to(
                fit_results[result].sel(metabolite=source, drop=True),
                updated[start].sel(line, drop=True),
            )
            for result, start in pairs
        }
        if any(values is None for values in aligned.values()):
            missing.append(metab)
            continue
        if freq:
            typed = updated["initial_freqs_hz"].sel(line, drop=True)
            values = aligned["initial_freqs_hz"]
            updated["initial_freqs_hz"].loc[line] = xr.where(
                np.isfinite(values), values, typed
            )
        if t2:
            typed = updated["initial_t2s_s"].sel(line, drop=True)
            values = aligned["initial_t2s_s"]
            t2_seeded = xr.where(np.isfinite(values) & (values > 0), values, typed)
            has_window = index < len(windows) and windows[index] is not None
            if not has_window and "min_t2_s" in updated and "max_t2_s" in updated:
                low = updated["min_t2_s"].sel(line, drop=True)
                high = updated["max_t2_s"].sel(line, drop=True)
                n_outside = int(((t2_seeded < low) | (t2_seeded > high)).sum())
                if n_outside:
                    t2_seeded = t2_seeded.clip(low, high)
                    clipped.append(f"{metab} ({n_outside} voxels)")
            updated["initial_t2s_s"].loc[line] = t2_seeded
        parts = [part for part, on in (("freq", freq), ("T2*", t2)) if on]
        seeded.append(f"{metab} ({' and '.join(parts)})")

    note = ""
    if seeded:
        note += "Start values from the earlier fit for: " + ", ".join(seeded) + ". "
    if clipped:
        note += (
            "Start T2* outside min/max moved to the nearer bound for: "
            + ", ".join(clipped)
            + ". "
        )
    if missing:
        note += (
            "No matching earlier fit values for: "
            + ", ".join(missing)
            + "; their typed start values are used. "
        )
    if not seeded:
        return fit_params, note
    return _update_frequency_bounds(updated), note


def apply_t2_windows(
    fit_params: xr.Dataset,
    *,
    metabs: Sequence[str],
    windows_s: Sequence[float | None],
) -> xr.Dataset:
    """Bound T2* of chosen lines to a window around each voxel's start value.

    For every line with a window ``w`` (full width in s), the per-voxel
    bounds become ``initial_t2s_s - w/2`` (at least 1 µs) and
    ``initial_t2s_s + w/2`` and ``metabs_t2_ranges`` becomes ``w``; the start
    values themselves stay unchanged. Use it after :func:`seed_start_values`,
    so the window lies around the seeded starts.

    Parameters
    ----------
    fit_params : xarray.Dataset
        Prepared fit parameters with ``initial_t2s_s`` (s), ``metabs_t2_ranges``
        and the T2* bound maps ``min_t2_s``/``max_t2_s`` and/or
        ``min_t2s_s``/``max_t2s_s``. Not modified.
    metabs : sequence of str
        Line names, in the order of ``windows_s``.
    windows_s : sequence of float or None
        Full window width in s per line; None keeps that line's bounds.

    Returns
    -------
    xarray.Dataset
        New parameters, or ``fit_params`` itself when no line has a window.
    """
    if not any(window is not None for window in windows_s):
        return fit_params
    updated = fit_params.copy(deep=True)
    bound_keys = [
        (low, high)
        for low, high in (("min_t2_s", "max_t2_s"), ("min_t2s_s", "max_t2s_s"))
        if low in updated and high in updated
    ]
    for metab, window_s in zip(metabs, windows_s, strict=True):
        if window_s is None:
            continue
        line = {"metabolite": metab}
        start = updated["initial_t2s_s"].loc[line]
        for low, high in bound_keys:
            updated[low].loc[line] = np.maximum(start - window_s / 2.0, _MIN_T2_BOUND_S)
            updated[high].loc[line] = start + window_s / 2.0
        updated["metabs_t2_ranges"].loc[line] = window_s
    return updated


def _padded(flags: Sequence[bool], metabs: Sequence[str]) -> list[bool]:
    """Return one flag per line; missing flags are False."""
    values = [bool(flag) for flag in flags][: len(metabs)]
    return values + [False] * (len(metabs) - len(values))


def _aligned_to(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray | None:
    """Return ``source`` on the dimensions of ``target``, or None.

    Dimensions of ``source`` that ``target`` lacks are dropped when they
    have size 1; otherwise the values do not belong to the target voxels and
    None is returned. The rest is broadcast to ``target``.
    """
    for dim in list(source.dims):
        if dim in target.dims:
            continue
        if source.sizes[dim] != 1:
            return None
        source = source.isel({dim: 0}, drop=True)
    try:
        return source.broadcast_like(target).transpose(*target.dims)
    except (ValueError, KeyError):
        return None
