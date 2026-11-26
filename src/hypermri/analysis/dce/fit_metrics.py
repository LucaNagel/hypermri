"""Residual statistics shared by DCE curves and labelled quality maps.

Callers select samples and the scientific residual space. Curve calculations
preserve NumPy float64 reductions; map calculations preserve labelled xarray
reductions, missing-sample handling and the explicit quality-map RSS floor.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _sum_squared_residuals(residual, *, sample_dim=None, skipna=False):
    """Sum squared residuals in the caller's units without changing precision."""
    squared = residual**2
    if isinstance(squared, xr.DataArray):
        return squared.sum(dim=sample_dim, skipna=skipna)
    return np.sum(squared)


def _residual_statistics(observed, predicted, *, sample_dim=None, rss_floor=None):
    """Return RSS, RMSE and R² for selected real samples in one residual space.

    NumPy inputs are finite 1D concentration curves, already filtered by the
    caller. Labelled map inputs use ``sample_dim`` and retain the existing
    skip-NaN rule, counting finite observations independently of predictions.
    For labelled maps, ``rss_floor`` affects RMSE and information criteria only,
    never stored RSS or R². Curve callers leave it unset. Constant observations
    yield NaN R². Inputs are not mutated.
    """
    residual = predicted - observed
    labelled = isinstance(observed, xr.DataArray)
    rss = _sum_squared_residuals(residual, sample_dim=sample_dim, skipna=labelled)
    if labelled:
        n_obs = np.isfinite(observed).sum(dim=sample_dim)
        rss_safe = (
            rss if rss_floor is None else xr.where(rss > rss_floor, rss, rss_floor)
        )
        rmse = np.sqrt(rss_safe / xr.where(n_obs > 0, n_obs, np.nan))
        mean_obs = observed.mean(dim=sample_dim, skipna=True)
    else:
        rss = float(rss)
        n_obs = observed.size
        rss_safe = rss if rss_floor is None else max(rss, rss_floor)
        rmse = float(np.sqrt(np.mean(residual**2)))
        mean_obs = float(np.mean(observed))
    tss = _sum_squared_residuals(
        observed - mean_obs, sample_dim=sample_dim, skipna=labelled
    )
    if labelled:
        r_squared = xr.where(tss > 0, 1.0 - rss / tss, np.nan)
    else:
        tss = float(tss)
        r_squared = np.nan if tss <= 0 else float(1.0 - rss / tss)
    return {
        "rss": rss,
        "rss_safe": rss_safe,
        "rmse": rmse,
        "r_squared": r_squared,
        "n_observations": n_obs,
    }
