"""The "parameter at bound" check uses the bounds that the DCE fitters apply.

Fixtures are synthetic: a piecewise-linear AIF in mM on an 8 s grid with 1 s
sampling and noise-free Tofts tissue curves.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import interp1d

from hypermri.analysis.dce import fitting_support
from hypermri.analysis.dce.fitting_support import compute_fit_quality_metrics
from hypermri.analysis.dce.models import sourbron
from hypermri.analysis.dce.models.tofts import (
    TOFTS_1COMP_BOUNDS,
    TOFTS_2COMP_BOUNDS,
    fit_single_voxel_tofts_helper,
    tofts_model_1comp,
)

TIMES_S = np.arange(8, dtype=np.float64)
AIF_MM = np.array([0.0, 0.0, 0.4, 0.9, 0.6, 0.3, 0.15, 0.05], dtype=np.float64)
SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


def _spatial(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1, 1)
    return xr.DataArray(
        values,
        dims=SPATIAL_DIMS,
        coords={
            "read_idx": np.arange(values.shape[0]),
            "phase_idx": [0],
            "slice_idx": [0],
        },
    )


def _concentration_map(curves):
    curves = np.asarray(curves, dtype=np.float64).reshape(len(curves), 1, 1, -1)
    return xr.DataArray(
        curves,
        dims=(*SPATIAL_DIMS, "repetition_idx"),
        coords={
            "read_idx": np.arange(curves.shape[0]),
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(curves.shape[-1]),
        },
    )


def _tofts_1comp_quality(ktrans, kep, curves):
    fit_results = xr.Dataset(
        {
            "Ktrans_1c": _spatial(ktrans),
            "kep_1c": _spatial(kep),
            "rss_1c": _spatial(np.zeros(len(ktrans))),
        },
        attrs={"model_mode": "1comp"},
    )
    return compute_fit_quality_metrics(
        fit_results,
        _concentration_map(curves),
        fit_type="tofts",
        aif_concentration=AIF_MM,
        aif_time_s=TIMES_S,
        fit_params={"model_mode": "1comp"},
    )


def test_tofts_fit_and_bound_check_share_one_bounds_owner():
    assert TOFTS_1COMP_BOUNDS == ((0.0, 0.0), (5.0, 5.0))
    assert TOFTS_2COMP_BOUNDS == ((0.0, 0.0, 0.0), (5.0, 5.0, 1.0))
    assert fitting_support.TOFTS_1COMP_BOUNDS is TOFTS_1COMP_BOUNDS
    assert fitting_support.TOFTS_2COMP_BOUNDS is TOFTS_2COMP_BOUNDS


def test_sourbron_bound_check_uses_the_fitter_bounds():
    assert fitting_support._default_sourbron_bounds is sourbron._default_sourbron_bounds


def test_tofts_fit_saturating_at_upper_bound_is_flagged():
    """Ktrans of 8 1/s exceeds the upper bound; the fit stops at the bound."""
    cp_func = interp1d(TIMES_S, AIF_MM, kind="linear", fill_value="extrapolate")
    saturating = tofts_model_1comp(TIMES_S, 8.0, 0.5, cp_func)
    interior = tofts_model_1comp(TIMES_S, 0.18, 0.55, cp_func)

    fitted = [
        fit_single_voxel_tofts_helper(TIMES_S, curve, cp_func, "1comp")[0]
        for curve in (saturating, interior)
    ]
    ktrans = [params[0] for params in fitted]
    kep = [params[1] for params in fitted]

    # least_squares keeps the solution inside the bounds; 1e-6 1/s is the
    # absolute tolerance of the bound check.
    assert ktrans[0] == pytest.approx(TOFTS_1COMP_BOUNDS[1][0], abs=1e-6)
    quality = _tofts_1comp_quality(ktrans, kep, [saturating, interior])
    assert quality["hit_bound"].values.ravel().tolist() == [True, False]
