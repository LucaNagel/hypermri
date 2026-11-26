"""Fit results report whether a frequency or T2* sits at its search bound.

Fixtures are synthetic noise-free FIDs, 1024 points at 1 kHz (spectral point
spacing 0.98 Hz), with one line of T2* = 30 ms per spectrum. Each case places
the true value outside, inside or on the edge of the search bounds, so the
expected status follows from the bounds alone.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv

TIME_S = np.arange(1024) / 1000.0
TRUE_T2_S = 0.03


def _data(freqs_hz, amplitudes=None):
    amplitudes = np.ones(len(freqs_hz)) if amplitudes is None else amplitudes
    fids = [
        amplitude * np.exp(-TIME_S / TRUE_T2_S + 2j * np.pi * freq * TIME_S)
        for freq, amplitude in zip(freqs_hz, amplitudes, strict=True)
    ]
    return xr.DataArray(
        np.array(fids),
        dims=("x", "acquisition_idx"),
        coords={"x": np.arange(len(freqs_hz))},
        attrs={"signal_domain": "temporal", "sampling_time_axis_s": TIME_S},
    )


def _fit(data, **options):
    params = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a"],
            "metabs_freqs_hz": [0.0],
            "rep_fitting": 2,
            "npoints": 11,
            **options,
        },
    )
    return fit_data_pseudo_inv(data, params)


def test_frequency_outside_the_window_is_at_the_nearer_bound():
    # Lines at +6 and -6 Hz with a window of +/- 2 Hz around 0 Hz; the line at
    # 0.5 Hz lies inside it.
    results = _fit(_data([6.0, -6.0, 0.5]), metabs_freq_ranges=[4.0])

    status = results.fit_freq_status.sel(metabolite="a").values
    assert status.tolist() == ["at_upper_bound", "at_lower_bound", "estimated"]
    np.testing.assert_allclose(
        results.fit_freqs.sel(metabolite="a").values[:2], [2.0, -2.0], atol=1e-3
    )
    assert "frequency window" in results.fit_freq_status.attrs["bounds"]


@pytest.mark.parametrize(
    ("bounds_s", "expected"),
    [
        (
            {"min_t2_s": 0.005, "max_t2_s": 0.02, "initial_t2s_s": 0.015},
            "at_upper_bound",
        ),
        ({"min_t2_s": 0.04, "max_t2_s": 0.07, "initial_t2s_s": 0.05}, "at_lower_bound"),
        ({"min_t2_s": 0.005, "max_t2_s": 0.07, "initial_t2s_s": 0.025}, "estimated"),
    ],
)
def test_t2_status_follows_the_t2_bounds(bounds_s, expected):
    results = _fit(_data([0.0]), metabs_freq_ranges=[10.0], **bounds_s)

    assert results.fit_t2_status.sel(metabolite="a").item() == expected


def test_frozen_and_unfitted_lines_are_labelled():
    # The second spectrum is all zeros, so its fit does not succeed.
    results = _fit(
        _data([0.0, 0.0], amplitudes=[1.0, 0.0]),
        metabs_freq_ranges=[0.0],
        metabs_t2_ranges=[0.0],
    )

    for name in ("fit_freq_status", "fit_t2_status"):
        assert results[name].sel(metabolite="a").values.tolist() == [
            "fixed",
            "not_fitted",
        ]
        assert results[name].dims == results.fit_freqs.dims


def test_a_fixed_ratio_dependent_follows_its_leaders_search():
    data = _data([0.0])
    params = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b"],
            "metabs_freqs_hz": [0.0, 100.0],
            "metabs_freq_ranges": [10.0, 10.0],
            "metabs_t2_ranges": [0.04, 0.0],
            "t2_coupling": {"b": "a"},
            "rep_fitting": 2,
            "npoints": 11,
        },
    )

    results = fit_data_pseudo_inv(data, params)

    assert results.fit_t2_status.sel(metabolite="b").item() != "fixed"


def test_a_window_narrower_than_one_point_warns_with_line_and_spacing():
    data = _data([0.0])
    with pytest.warns(UserWarning, match=r"of a \(0\.5 Hz\) is narrower.*0\.977 Hz"):
        _fit(data, metabs_freq_ranges=[0.5])


def test_overlapping_windows_of_uncoupled_lines_warn_and_coupled_do_not():
    data = _data([0.0])
    options = {
        "metabs": ["a", "b"],
        "metabs_freqs_hz": [0.0, 20.0],
        "metabs_freq_ranges": [30.0, 30.0],
        "rep_fitting": 1,
        "npoints": 5,
    }
    params = def_fit_params(data=data, fit_params=options)
    with pytest.warns(UserWarning, match=r"a \[-15, 15\] Hz and b \[5, 35\] Hz"):
        fit_data_pseudo_inv(data, params)

    coupled = def_fit_params(
        data=data, fit_params={**options, "t2_coupling": {"b": "a"}}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit_data_pseudo_inv(data, coupled)
