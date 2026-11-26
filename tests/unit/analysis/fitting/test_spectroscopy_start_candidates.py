"""Characterize start candidates and keep-best selection of the FID fit.

Fixtures are synthetic FIDs of 256 samples at 4 kHz with complex white noise
of sigma 0.05 per quadrature; every row of the ``x`` dimension has its own
noise. The Voigt fixture has lines at -300, 0 and 12 Hz (areas 10, 20 and 12,
T2* 50 ms) under one Gaussian decay of 15 Hz FWHM. Started from a Gaussian
FWHM of 1 Hz the search ends in a local minimum with about twice the RSS of
the start at 15 Hz (about 3.4 against 1.4 a.u.^2, expected 2 * 256 * 0.05**2
= 1.28), so the two starts are a known local and a near-global minimum.
The nested fixture has only the lines at -300 and 0 Hz, so a third line
explains noise only and the nested fit must not end above its parent's RSS
(monotone search: only strict objective improvements are accepted). The
relative tolerance 1e-12 covers rounding of the final RSS evaluation.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    add_metab_to_fit_params,
    def_fit_params,
    fit_best_of_starts,
    fit_data_pseudo_inv,
)

TIME_S = np.arange(256) / 4000.0
NOISE_SIGMA = 0.05


def synthetic_fid(freqs_hz, areas, *, gauss_fwhm_hz=0.0, n_spectra=4, seed=0):
    """Return labelled noisy FIDs of Lorentz lines with T2* 50 ms."""
    rng = np.random.default_rng(seed)
    lines = np.asarray(areas)[:, None] * np.exp(
        2j * np.pi * np.asarray(freqs_hz, dtype=float)[:, None] * TIME_S - TIME_S / 0.05
    )
    gauss = np.exp(-((np.pi * gauss_fwhm_hz * TIME_S) ** 2) / (4 * np.log(2)))
    noise = NOISE_SIGMA * (
        rng.standard_normal((n_spectra, TIME_S.size))
        + 1j * rng.standard_normal((n_spectra, TIME_S.size))
    )
    data = xr.DataArray(
        (lines.sum(0) * gauss)[None] + noise,
        dims=["x", "acquisition_idx"],
        coords={"x": np.arange(n_spectra)},
    )
    data.attrs["sampling_time_axis_s"] = TIME_S
    return data


def voigt_data():
    """Return the Voigt fixture with a known local minimum."""
    return synthetic_fid(
        [-300.0, 0.0, 12.0],
        [10.0, 20.0, 12.0 * np.exp(2.5j)],
        gauss_fwhm_hz=15.0,
    )


def voigt_params(data, gauss_start_hz):
    """Return three-line Voigt parameters with one Gaussian FWHM start in Hz."""
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b", "c"],
            "metabs_freqs_hz": [-299.0, -5.0, 20.0],
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": [8.0, 40.0, 40.0],
            "min_t2_s": 0.005,
            "max_t2_s": 0.2,
            "lineshape": "voigt",
            "gauss_initial_fwhm_hz": gauss_start_hz,
        },
    )


def test_local_minimum_is_left_for_the_better_start():
    """The bad Gaussian start ends worse; keep-best returns the good fit."""
    data = voigt_data()
    local = voigt_params(data, 1.0)
    good = voigt_params(data, 15.0)
    local_fit = fit_data_pseudo_inv(data, local)
    good_fit = fit_data_pseudo_inv(data, good)
    assert np.all(local_fit.fit_rss > 1.5 * good_fit.fit_rss)

    best = fit_best_of_starts(data, start_params=[local, good])

    np.testing.assert_array_equal(best.fit_start_candidate, 1)
    assert best.fit_start_candidate.dtype == np.int64
    assert best.fit_start_candidate.dims == ("x",)
    for name in good_fit.data_vars:
        xr.testing.assert_identical(best[name], good_fit[name])
    np.testing.assert_array_equal(
        best.fit_start_objective.sel(start_candidate=0), local_fit.fit_objective
    )
    assert best.fit_start_objective.dims == ("start_candidate", "x")
    assert "lowest finite fit_objective" in best.attrs["start_selection"]


def test_choice_is_made_per_spectrum():
    """Per-spectrum starts give a per-spectrum choice with that row's values."""
    data = voigt_data()
    first_rows = xr.DataArray([True, True, False, False], dims="x")
    good_first = voigt_params(data, 15.0)
    good_first["initial_gauss_fwhm_hz"] = good_first.initial_gauss_fwhm_hz.where(
        first_rows, 1.0
    )
    good_last = voigt_params(data, 1.0)
    good_last["initial_gauss_fwhm_hz"] = good_last.initial_gauss_fwhm_hz.where(
        first_rows, 15.0
    )
    fits = [fit_data_pseudo_inv(data, good_first), fit_data_pseudo_inv(data, good_last)]

    best = fit_best_of_starts(data, start_params=[good_first, good_last])

    np.testing.assert_array_equal(best.fit_start_candidate, [0, 0, 1, 1])
    for row, candidate in enumerate([0, 0, 1, 1]):
        for name in ("fit_amps", "fit_freqs", "fit_t2s", "fit_gauss_fwhm_hz"):
            xr.testing.assert_identical(
                best[name].isel(x=row), fits[candidate][name].isel(x=row)
            )


def test_ties_choose_the_first_candidate():
    """Identical candidates tie in every spectrum and keep index 0."""
    data = voigt_data()
    params = voigt_params(data, 15.0)
    best = fit_best_of_starts(
        data, start_params=[params, params.copy(deep=True), params]
    )
    np.testing.assert_array_equal(best.fit_start_candidate, 0)
    assert best.sizes["start_candidate"] == 3


def test_failed_candidate_is_not_chosen():
    """A NaN objective loses against any finite one; all-NaN keeps index 0."""
    data = voigt_data()
    data[1] = np.nan
    params = voigt_params(data, 15.0)
    best = fit_best_of_starts(data, start_params=[params, voigt_params(data, 1.0)])
    assert best.fit_status.isel(x=1).item() == "invalid_input"
    assert best.fit_start_candidate.isel(x=1).item() == 0
    assert np.isnan(best.fit_start_objective.isel(x=1)).all()
    np.testing.assert_array_equal(best.fit_start_candidate.isel(x=[0, 2, 3]), 0)


def test_serial_and_parallel_choices_are_equal():
    """Joblib execution chooses the same candidates and values as serial."""
    data = voigt_data()
    candidates = [voigt_params(data, 1.0), voigt_params(data, 15.0)]
    serial = fit_best_of_starts(data, start_params=candidates)
    parallel = fit_best_of_starts(data, start_params=candidates, n_jobs=2, batch_size=1)
    np.testing.assert_array_equal(
        parallel.fit_start_candidate, serial.fit_start_candidate
    )
    xr.testing.assert_allclose(parallel, serial, rtol=1e-10, atol=1e-12)


def nested_case(options):
    """Return data, the two-line parent fit and the nested three-line start."""
    data = synthetic_fid(
        [-300.0, 0.0],
        [10.0, 20.0],
        gauss_fwhm_hz=8.0 if options.get("lineshape") == "voigt" else 0.0,
        n_spectra=6,
    )
    parent = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b"],
            "metabs_freqs_hz": [-299.0, 3.0],
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": [8.0, 40.0],
            "min_t2_s": 0.005,
            "max_t2_s": 0.2,
            **options,
        },
    )
    parent_fit = fit_data_pseudo_inv(data, parent)
    nested = add_metab_to_fit_params(
        40.0,
        new_metab_name="c",
        fit_params=parent,
        fit_results=parent_fit,
        freeze_other_freqs=False,
        freeze_other_t2s=False,
    )
    return data, parent_fit, nested


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"lineshape": "voigt", "gauss_initial_fwhm_hz": 5.0},
        {"phase_band_tolerance_deg": {"a": 10.0, "b": 10.0}, "phase_band_tau_s": 0.0},
    ],
    ids=["lorentz", "voigt", "phase_band"],
)
def test_nested_start_is_never_worse_than_its_parent(options):
    """Adding a noise-only line from the parent solution cannot raise the RSS."""
    data, parent_fit, nested = nested_case(options)
    nested_fit = fit_data_pseudo_inv(data, nested)
    assert np.all(nested_fit.fit_rss <= parent_fit.fit_rss * (1 + 1e-12))


def test_candidates_must_describe_the_same_model():
    """Differing options, lines or line order are rejected before fitting."""
    data = voigt_data()
    params = voigt_params(data, 15.0)
    other_option = params.copy(deep=True)
    other_option.attrs["niter"] = params.attrs["niter"] + 1
    with pytest.raises(ValueError, match=r"attrs\['niter'\]"):
        fit_best_of_starts(data, start_params=[params, other_option])
    with pytest.raises(ValueError, match="same lines"):
        fit_best_of_starts(data, start_params=[params, params.isel(metabolite=[0, 1])])
    swapped = params.copy(deep=True)
    swapped["initial_freqs_hz"].loc[{"metabolite": "b"}] = 30.0
    swapped["initial_freqs_hz"].loc[{"metabolite": "c"}] = -10.0
    with pytest.raises(ValueError, match="orders the lines"):
        fit_best_of_starts(data, start_params=[params, swapped])
    with pytest.raises(ValueError, match="at least two"):
        fit_best_of_starts(data, start_params=[params])
    with pytest.raises(TypeError, match="single Dataset"):
        fit_best_of_starts(data, start_params=params)


def test_single_spectrum_has_scalar_choice():
    """A spectrum without loop dimensions gets a 0-d choice."""
    data = voigt_data().isel(x=1)
    best = fit_best_of_starts(
        data, start_params=[voigt_params(data, 1.0), voigt_params(data, 15.0)]
    )
    assert best.fit_start_candidate.dims == ()
    assert best.fit_start_candidate.item() == 1
    assert best.fit_start_objective.dims == ("start_candidate",)
