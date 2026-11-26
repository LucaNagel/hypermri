"""Tests of start values taken from an earlier spectral fit (invented maps)."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.fit_seeding import (
    apply_t2_windows,
    fit_results_can_seed,
    seed_start_values,
)

METABS = ["pyr", "lac"]


def _params():
    """Prepared fit parameters of 2 x 3 voxels and two lines (Hz and s)."""
    shape = (2, 3, 2)
    dims = ("x", "y", "metabolite")
    coords = {"metabolite": METABS}

    def full(values):
        return xr.DataArray(
            np.broadcast_to(values, shape).astype(float), dims=dims, coords=coords
        )

    return xr.Dataset(
        {
            "initial_freqs_hz": full([0.0, 400.0]),
            "initial_t2s_s": full([0.02, 0.02]),
            "metabs_freq_ranges": full([100.0, 100.0]),
            "metabs_t2_ranges": full([0.04, 0.04]),
            "min_t2_s": full([0.005, 0.005]),
            "max_t2_s": full([0.05, 0.05]),
        }
    )


def _results(freqs, t2s, metabs=METABS):
    """Earlier fit results with per-voxel frequencies (Hz) and T2* (s)."""
    dims = ("x", "y", "metabolite")
    coords = {"metabolite": metabs}
    return xr.Dataset(
        {
            "fit_freqs": xr.DataArray(
                np.asarray(freqs, float), dims=dims, coords=coords
            ),
            "fit_t2s": xr.DataArray(np.asarray(t2s, float), dims=dims, coords=coords),
        },
        attrs={"lineshape": "lorentz"},
    )


def _maps(value_pyr, value_lac):
    return np.stack(
        [np.full((2, 3), value_pyr, float), np.full((2, 3), value_lac, float)], axis=-1
    )


def test_frequency_and_t2_are_seeded_separately_per_line():
    params = _params()
    freqs = _maps(10.0, 390.0)
    freqs[0, 0, 0] = np.nan
    results = _results(freqs, _maps(0.03, 0.01))

    seeded, note = seed_start_values(
        params, results, metabs=METABS, seed_freq=[True, False], seed_t2=[False, True]
    )

    pyr = seeded.sel(metabolite="pyr")
    lac = seeded.sel(metabolite="lac")
    # A non-finite earlier frequency keeps the typed start of that voxel.
    assert pyr.initial_freqs_hz.values[0, 0] == 0.0
    np.testing.assert_array_equal(pyr.initial_freqs_hz.values.ravel()[1:], 10.0)
    np.testing.assert_array_equal(pyr.initial_t2s_s, 0.02)
    np.testing.assert_array_equal(lac.initial_freqs_hz, 400.0)
    np.testing.assert_array_equal(lac.initial_t2s_s, 0.01)
    # Frequency bounds follow the new starts: start +/- half of 100 Hz.
    np.testing.assert_array_equal(
        pyr.freq_range_lower_bound_hz.values.ravel()[1:], -40.0
    )
    assert "pyr (freq)" in note and "lac (T2*)" in note
    # The input is not modified.
    np.testing.assert_array_equal(params.initial_freqs_hz.sel(metabolite="pyr"), 0.0)


def test_seeded_t2_outside_the_bounds_is_clipped_without_a_window():
    results = _results(_maps(0.0, 400.0), _maps(0.2, 0.001))
    seeded, note = seed_start_values(
        _params(),
        results,
        metabs=METABS,
        seed_freq=[False, False],
        seed_t2=[True, True],
    )
    np.testing.assert_array_equal(seeded.initial_t2s_s.sel(metabolite="pyr"), 0.05)
    np.testing.assert_array_equal(seeded.initial_t2s_s.sel(metabolite="lac"), 0.005)
    assert "moved to the nearer bound" in note


def test_window_keeps_the_seeded_t2_and_sets_bounds_around_it():
    results = _results(_maps(0.0, 400.0), _maps(0.2, 0.02))
    seeded, _ = seed_start_values(
        _params(),
        results,
        metabs=METABS,
        seed_freq=[False, False],
        seed_t2=[True, False],
        t2_windows_s=[0.01, None],
    )
    np.testing.assert_array_equal(seeded.initial_t2s_s.sel(metabolite="pyr"), 0.2)
    windowed = apply_t2_windows(seeded, metabs=METABS, windows_s=[0.01, None])
    pyr = windowed.sel(metabolite="pyr")
    np.testing.assert_allclose(pyr.min_t2_s, 0.195)
    np.testing.assert_allclose(pyr.max_t2_s, 0.205)
    np.testing.assert_allclose(pyr.metabs_t2_ranges, 0.01)
    # The line without a window keeps its bounds.
    np.testing.assert_array_equal(windowed.min_t2_s.sel(metabolite="lac"), 0.005)


def test_window_lower_bound_stays_positive():
    windowed = apply_t2_windows(_params(), metabs=METABS, windows_s=[1.0, None])
    assert float(windowed.min_t2_s.sel(metabolite="pyr").min()) == pytest.approx(1e-6)


def test_lines_without_earlier_values_and_without_flags():
    params = _params()
    results = _results(_maps(5.0, 405.0)[..., :1], _maps(0.03, 0.03)[..., :1], ["pyr"])
    unchanged, note = seed_start_values(
        params, results, metabs=METABS, seed_freq=[False], seed_t2=[]
    )
    assert unchanged is params and note == ""

    seeded, note = seed_start_values(
        params, results, metabs=METABS, seed_freq=[False, True], seed_t2=[False, False]
    )
    assert seeded is params
    assert "No matching earlier fit values for: lac" in note

    renamed, note = seed_start_values(
        params,
        results,
        metabs=METABS,
        seed_freq=[False, True],
        seed_t2=[False, False],
        source_metabs=["pyr", "pyr"],
    )
    np.testing.assert_array_equal(renamed.initial_freqs_hz.sel(metabolite="lac"), 5.0)


def test_voxel_grid_that_does_not_match_is_not_used():
    results = _results(_maps(5.0, 405.0), _maps(0.03, 0.03)).isel(y=slice(0, 2))
    seeded, note = seed_start_values(
        _params(),
        results,
        metabs=METABS,
        seed_freq=[True, True],
        seed_t2=[False, False],
    )
    np.testing.assert_array_equal(seeded.initial_freqs_hz.sel(metabolite="pyr"), 0.0)
    assert "No matching earlier fit values" in note


def test_no_earlier_fit_and_voigt_results_do_not_seed_a_lorentz_fit():
    params = _params()
    flags = {"metabs": METABS, "seed_freq": [True, True], "seed_t2": [True, True]}
    assert seed_start_values(params, None, **flags)[0] is params
    voigt = _results(_maps(5.0, 405.0), _maps(0.03, 0.03))
    voigt.attrs["lineshape"] = "voigt"
    assert not fit_results_can_seed(voigt, params)
    seeded, note = seed_start_values(params, voigt, **flags)
    assert seeded is params and "Voigt" in note
    lorentz = _results(_maps(5.0, 405.0), _maps(0.03, 0.03))
    assert fit_results_can_seed(lorentz, {"lineshape": "voigt"})
