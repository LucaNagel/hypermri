"""Tests for coverage-weighted parameter statistics inside masks.

Synthetic fit maps on a 4 x 3 x 2 voxel grid with two lines. Expected
values are computed by hand from the weighted mean and population standard
deviation; tolerances are float64 rounding (rtol 1e-12).
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.anatomical.roi import mask_statistics

DIMS = ("read_idx", "phase_idx", "slice_idx")
SHAPE = (4, 3, 2)


def _maps():
    rng = np.random.default_rng(3)
    freqs = rng.normal(0.0, 5.0, (*SHAPE, 2))
    t2s = rng.uniform(0.01, 0.05, (*SHAPE, 2))
    amps = rng.normal(size=(*SHAPE, 2)) + 1j * rng.normal(size=(*SHAPE, 2))
    coords = {"metabolite": ["pyruvate", "lactate"]}
    dims = (*DIMS, "metabolite")
    return xr.Dataset(
        {
            "fit_freqs": (dims, freqs, {"units": "Hz"}),
            "fit_t2s": (dims, t2s, {"units": "s"}),
            "fit_amps": (dims, amps),
            "fit_status": (dims, np.full((*SHAPE, 2), "ok")),
            "fit_gauss_fwhm_hz": (DIMS, np.ones(SHAPE)),
        },
        coords=coords,
    )


def _mask(values):
    return xr.DataArray(np.asarray(values, dtype=np.float32).reshape(SHAPE), dims=DIMS)


def _expected(x, w):
    keep = np.isfinite(x) & (w > 0)
    x, w = x[keep], w[keep]
    mean = np.sum(w * x) / np.sum(w)
    return mean, np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w)), keep.sum(), w.sum()


def test_weighted_mean_and_std_per_mask_and_line():
    maps = _maps()
    coverage = np.zeros(SHAPE)
    coverage[1:3, :2, 0] = 1.0
    coverage[0, 0, 0] = 0.25
    masks = xr.Dataset({"tumour": _mask(coverage), "all": _mask(np.ones(SHAPE))})

    table = mask_statistics(maps, masks)

    row = table[
        (table["mask"] == "tumour")
        & (table.variable == "fit_freqs")
        & (table.metabolite == "lactate")
    ].iloc[0]
    mean, std, n, wsum = _expected(
        maps.fit_freqs.values[..., 1].ravel(), coverage.ravel()
    )
    np.testing.assert_allclose([row["mean"], row["std"]], [mean, std], rtol=1e-12)
    assert row.n_voxels == n == 5
    assert row.weight_sum == pytest.approx(wsum)
    assert row.units == "Hz"
    assert list(table.columns) == [
        "mask",
        "variable",
        "metabolite",
        "mean",
        "std",
        "n_voxels",
        "weight_sum",
        "units",
    ]
    assert set(table.variable) == {
        "fit_freqs",
        "fit_t2s",
        "fit_amps",
        "fit_gauss_fwhm_hz",
    }
    assert list(table["mask"].unique()) == ["tumour", "all"]


def test_failed_fits_are_left_out_and_complex_values_use_magnitude():
    maps = _maps()
    maps["fit_t2s"][0, 0, 0, 0] = np.nan
    coverage = np.ones(SHAPE)
    table = mask_statistics(
        maps, {"all": _mask(coverage)}, variables=["fit_t2s", "fit_amps"]
    )

    t2 = table[(table.variable == "fit_t2s") & (table.metabolite == "pyruvate")].iloc[0]
    mean, std, n, _ = _expected(maps.fit_t2s.values[..., 0].ravel(), coverage.ravel())
    assert t2.n_voxels == n == 23
    np.testing.assert_allclose([t2["mean"], t2["std"]], [mean, std], rtol=1e-12)

    amp = table[(table.variable == "fit_amps") & (table.metabolite == "lactate")].iloc[
        0
    ]
    expected = np.abs(maps.fit_amps.values[..., 1]).mean()
    assert amp["mean"] == pytest.approx(expected, rel=1e-12)


def test_empty_mask_gives_nan():
    table = mask_statistics(
        _maps(), {"empty": _mask(np.zeros(SHAPE))}, variables=["fit_freqs"]
    )
    assert table["mean"].isna().all()
    assert (table.n_voxels == 0).all()


def test_inputs_are_not_modified():
    maps = _maps()
    before = maps.copy(deep=True)
    mask_statistics(maps, {"m": _mask(np.full(SHAPE, 2.0))})
    xr.testing.assert_identical(maps, before)


def test_mask_on_another_grid_is_rejected():
    wrong = xr.DataArray(np.ones((5, 3, 2)), dims=DIMS)
    with pytest.raises(ValueError, match="mask 'm' has 'read_idx' of size 5"):
        mask_statistics(_maps(), {"m": wrong}, variables=["fit_freqs"])


def test_unknown_or_non_numeric_variable_is_rejected():
    masks = {"m": _mask(np.ones(SHAPE))}
    with pytest.raises(ValueError, match="no variable 'fit_width'"):
        mask_statistics(_maps(), masks, variables=["fit_width"])
    with pytest.raises(ValueError, match="'fit_status' is not numeric"):
        mask_statistics(_maps(), masks, variables=["fit_status"])
    with pytest.raises(ValueError, match="at least one mask"):
        mask_statistics(_maps(), {})
