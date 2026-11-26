"""The contrast-agent relaxivity reaches the AIF conversion from one owner.

Fixtures are synthetic: a 10-frame raw AIF signal with a baseline of 100 a.u.
and an enhancement of at most 40 %, an explicit blood T1 of 1500 ms, TR 5 ms
and a 15 degree flip angle, which stays far below SPGR saturation.  The SPGR
inversion returns the relaxation-rate change divided by ``r1``, so doubling
``r1`` halves the AIF concentration at every time point.
"""

import inspect

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.dce import aif, fitting_support, pipeline
from hypermri.analysis.dce.aif import prepare_aif
from hypermri.analysis.dce.conversions import (
    DEFAULT_R1_PER_MM_S,
    resolve_relaxivity,
)
from hypermri.analysis.dce.models import tofts

TIMES_S = np.arange(10, dtype=np.float64)
RAW_AIF = np.array(
    [100.0, 100.0, 100.0, 120.0, 140.0, 130.0, 120.0, 112.0, 106.0, 103.0]
)
# The concentrations are float64 results of the same SPGR inversion scaled by
# 1/r1, so they agree to rounding (about 1e-16 relative); 1e-12 separates
# rounding from a wrong relaxivity (relative error >= 0.1 in these cases).
RTOL = 1e-12


def _aif_concentration(**kwargs):
    dce = xr.DataArray(
        np.broadcast_to(RAW_AIF, (1, 1, 1, RAW_AIF.size)).copy(),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
    )
    t1_map = xr.DataArray(
        np.full((1, 1, 1), 1500.0), dims=("read_idx", "phase_idx", "slice_idx")
    )
    info = prepare_aif(
        dce,
        t1_map,
        aif_signal=RAW_AIF,
        aif_times=TIMES_S,
        n_precontrast=3,
        tr_s=0.005,
        fa_deg=15.0,
        aif_t1_ms=1500.0,
        **kwargs,
    )
    return np.asarray(info["aif_concentration"], dtype=np.float64)


def test_prepare_aif_uses_its_r1_argument():
    reference = _aif_concentration(r1=3.5)
    doubled = _aif_concentration(r1=7.0)

    assert np.nanmax(reference) > 0.0
    np.testing.assert_allclose(doubled, 0.5 * reference, rtol=RTOL)


def test_prepare_aif_reads_r1_from_fit_params_without_argument():
    np.testing.assert_allclose(
        _aif_concentration(fit_params={"r1": 7.0}),
        _aif_concentration(r1=7.0),
        rtol=RTOL,
    )


def test_prepare_aif_defaults_to_the_shared_relaxivity():
    np.testing.assert_allclose(
        _aif_concentration(),
        _aif_concentration(r1=DEFAULT_R1_PER_MM_S),
        rtol=RTOL,
    )


def test_prepare_aif_rejects_conflicting_r1():
    with pytest.raises(ValueError, match=r"prepare_aif received r1=7\.0"):
        _aif_concentration(r1=7.0, fit_params={"r1": 3.5})


@pytest.mark.parametrize("r1", [0.0, -1.0, np.nan])
def test_resolve_relaxivity_rejects_non_positive_values(r1):
    with pytest.raises(ValueError, match="positive finite relaxivity"):
        resolve_relaxivity(r1, {}, operation="prepare_aif")


def test_resolve_relaxivity_does_not_modify_fit_params():
    fit_params = {"r1": 4.5}

    assert resolve_relaxivity(None, fit_params, operation="prepare_aif") == 4.5
    assert fit_params == {"r1": 4.5}


@pytest.mark.parametrize(
    "function",
    [
        aif.prepare_aif,
        fitting_support.convert_signal_to_concentration_map,
        tofts.tofts_model_reconstruction,
    ],
)
def test_signature_defaults_use_the_shared_relaxivity(function):
    default = inspect.signature(function).parameters["r1"].default

    assert default is None or default == DEFAULT_R1_PER_MM_S


def test_pipeline_has_no_own_relaxivity_default():
    assert not hasattr(pipeline, "_DEFAULT_R1_PER_MM_S")
