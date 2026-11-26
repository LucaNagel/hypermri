"""Signal-space Tofts reconstruction of parameter maps.

The fixture is synthetic: a piecewise-linear AIF in mM on an 8 s grid with
1 s sampling and a two-voxel line with known standard Tofts parameters, a
baseline signal of 100 and 120 a.u. and a pre-contrast T1 of 1500 and 1400 ms,
acquired with TR 5 ms and a 15 degree flip angle.  The reference applies the
SPGR forward model of each voxel to the concentration reconstruction.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.dce.conversions import concentration_to_signal
from hypermri.analysis.dce.models.tofts import (
    tofts_model_reconstruction,
)

TIMES_S = np.arange(8, dtype=np.float64)
AIF_MM = np.array([0.0, 0.0, 0.4, 0.9, 0.6, 0.3, 0.15, 0.05], dtype=np.float64)
TR_S = 0.005
FA_DEG = 15.0
R1_PER_MM_S = 3.5
S0 = np.array([100.0, 120.0])
T10_MS = np.array([1500.0, 1400.0])

# Reconstruction and reference evaluate the same float64 expressions, so they
# agree to rounding (about 1e-16 relative).  1e-12 is far below the signal
# enhancement of these voxels (several per cent), which a voxel using the
# wrong s0 or T1 would change.
RTOL = 1e-12


def _map(values):
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims=("read_idx",))


def _reconstruct(**kwargs):
    return tofts_model_reconstruction(
        TIMES_S,
        AIF_MM,
        model_mode="1comp",
        Ktrans_1c=_map([0.18, 0.30]),
        kep_1c=_map([0.55, 0.80]),
        tr_s=TR_S,
        fa_deg=FA_DEG,
        r1=R1_PER_MM_S,
        **kwargs,
    )


def _expected_signal(s0, t10_ms):
    concentration = _reconstruct().transpose("read_idx", "repetition_idx").values
    return np.stack(
        [
            concentration_to_signal(
                concentration[voxel],
                s0[voxel],
                t10_ms[voxel] / 1000.0,
                TR_S,
                FA_DEG,
                R1_PER_MM_S,
            )
            for voxel in range(concentration.shape[0])
        ]
    )


def test_signal_reconstruction_uses_s0_and_t10_maps_per_voxel():
    signal = _reconstruct(s0=_map(S0), t10_ms=_map(T10_MS), reco_space="signal")

    assert signal.dims == ("read_idx", "repetition_idx")
    np.testing.assert_allclose(signal.values, _expected_signal(S0, T10_MS), rtol=RTOL)
    assert np.all(signal.values[:, 3] > S0)


def test_signal_reconstruction_broadcasts_scalar_s0_and_t10():
    signal = _reconstruct(s0=110.0, t10_ms=1450.0, reco_space="signal")

    np.testing.assert_allclose(
        signal.transpose("read_idx", "repetition_idx").values,
        _expected_signal(np.full(2, 110.0), np.full(2, 1450.0)),
        rtol=RTOL,
    )


def test_signal_reconstruction_without_s0_is_nan_for_every_voxel():
    signal = _reconstruct(t10_ms=_map(T10_MS), reco_space="signal")

    assert signal.shape == (2, TIMES_S.size)
    assert np.isnan(signal.values).all()


def test_signal_reconstruction_rejects_unlabelled_s0_arrays():
    with pytest.raises(ValueError, match="tofts_model_reconstruction.*s0"):
        _reconstruct(s0=S0, t10_ms=_map(T10_MS), reco_space="signal")


def test_dataset_reconstruct_reads_s0_and_t1_maps_in_signal_space():
    fit_result = xr.Dataset(
        {
            "Ktrans_1c": _map([0.18, 0.30]),
            "kep_1c": _map([0.55, 0.80]),
            "s0_map": _map(S0),
            "t1_map": _map(T10_MS),
            "time_points_s": ("repetition_idx", TIMES_S),
            "aif_signal": ("aif_idx", AIF_MM),
            "aif_time_s": ("aif_idx", TIMES_S),
        },
        attrs={
            "model_name": "tofts_model",
            "model_mode": "1comp",
            "tr_s": TR_S,
            "fa_deg": FA_DEG,
            "r1": R1_PER_MM_S,
        },
    )

    signal = fit_result.hp.reconstruct(reco_space="signal")

    np.testing.assert_allclose(
        signal.transpose("read_idx", "repetition_idx").values,
        _expected_signal(S0, T10_MS),
        rtol=RTOL,
    )
