"""Characterize the forward search for a missing line in the fit residual.

Fixtures are synthetic FIDs of 256 samples at 4 kHz (one spectral point is
15.6 Hz) with Lorentz lines of T2* 50 ms and complex white noise of sigma
0.05 per quadrature; every row of ``x`` has its own noise. The parent model
fits 'a' at -300 Hz and 'b' at 0 Hz (areas 10 and 20). A hidden line of area
1 at 150 Hz changes the RSS by about 35,000 sigma**2, a doublet partner of
area 5 at 12 Hz by about 270,000 sigma**2, far above the threshold
``4 ln(512) = 24.95``. On pure noise the largest ``delta_chi2`` over 200
spectra and the whole spectral width was 22.9 (median 11.8, 99th percentile
21.8) and no line was accepted, so the stated false-line rate is at most
2.5 % (one of 40 spectra). The fitted frequency of a found line agrees with
the true one within 0.5 Hz, a thirtieth of a spectral point; the largest
measured deviation was 0.18 Hz for the separate and 0.07 Hz for the doublet
line, and the fitted area agreed within 5 %.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
)
from hypermri.analysis.spectroscopy import add_line_from_residual

TIME_S = np.arange(256) / 4000.0
NOISE_SIGMA = 0.05
PARENT = {
    "metabs": ["a", "b"],
    "metabs_freqs_hz": [-299.0, 3.0],
    "initial_t2s_s": 0.03,
    "metabs_freq_ranges": [8.0, 40.0],
}
THRESHOLD = 4 * np.log(2 * TIME_S.size)


def synthetic_fid(extra_lines=(), *, n_spectra=4, seed=0):
    """Return labelled FIDs of the parent lines plus ``(freq_hz, area)`` lines."""
    lines = [(-300.0, 10.0), (0.0, 20.0), *extra_lines]
    freqs = np.array([freq for freq, _ in lines])[:, None]
    areas = np.array([area for _, area in lines])[:, None]
    signal = (areas * np.exp(2j * np.pi * freqs * TIME_S - TIME_S / 0.05)).sum(0)
    rng = np.random.default_rng(seed)
    noise = NOISE_SIGMA * (
        rng.standard_normal((n_spectra, TIME_S.size))
        + 1j * rng.standard_normal((n_spectra, TIME_S.size))
    )
    data = xr.DataArray(
        signal[None] + noise,
        dims=["x", "acquisition_idx"],
        coords={"x": np.arange(n_spectra)},
    )
    data.attrs["sampling_time_axis_s"] = TIME_S
    return data


def parent_params(data):
    """Return the two-line parent parameters for ``data``."""
    return def_fit_params(data=data, fit_params=dict(PARENT))


@pytest.mark.parametrize(
    "line, search_range_hz",
    [((150.0, 1.0), (50.0, 400.0)), ((12.0, 5.0), (-100.0, 400.0))],
    ids=["separate", "doublet"],
)
def test_hidden_line_is_found_at_its_frequency(line, search_range_hz):
    """A hidden line is proposed near and fitted at its true frequency."""
    data = synthetic_fid([line])
    result = add_line_from_residual(
        data,
        parent_params(data),
        search_range_hz=search_range_hz,
        noise=NOISE_SIGMA,
        new_metab_name="c",
    )
    assert result.metabolite.values.tolist() == ["a", "b", "c"]
    assert result.line_search_accepted.all()
    assert result.line_search_tested.all()
    assert not result.fit_skipped.sel(metabolite="c").any()
    spectral_point_hz = 1 / (TIME_S.size * (TIME_S[1] - TIME_S[0]))
    np.testing.assert_allclose(
        result.line_search_start_freq_hz, line[0], atol=spectral_point_hz
    )
    np.testing.assert_allclose(result.fit_freqs.sel(metabolite="c"), line[0], atol=0.5)
    np.testing.assert_allclose(
        np.abs(result.fit_amps.sel(metabolite="c")), line[1], rtol=0.05
    )


def test_pure_noise_yields_no_more_than_the_stated_false_rate():
    """Without a missing line at most one of 40 spectra gains a line."""
    data = synthetic_fid(n_spectra=40, seed=11)
    result = add_line_from_residual(
        data, parent_params(data), search_range_hz=(-1500.0, 1500.0), noise=0.05
    )
    assert result.line_search_tested.all()
    assert int(result.line_search_accepted.sum()) <= 1
    assert float(result.line_search_delta_chi2.min()) >= 0.0


def test_rejected_spectra_keep_the_parent_fit():
    """Where the line is not kept, the parent fit is returned unchanged."""
    data = synthetic_fid(seed=3)
    params = parent_params(data)
    parent = fit_data_pseudo_inv(data, params)
    result = add_line_from_residual(
        data, params, search_range_hz=(50.0, 400.0), noise=NOISE_SIGMA
    )
    assert not result.line_search_accepted.any()
    for name in ("fit_amps", "fit_freqs", "fit_t2s", "fit_fwhm_hz"):
        xr.testing.assert_identical(
            result[name].sel(metabolite=["a", "b"]), parent[name]
        )
    for name in ("fit_rss", "fit_status", "fit_n_free_parameters"):
        xr.testing.assert_identical(result[name], parent[name])
    assert result.fit_skipped.sel(metabolite="residual_line").all()
    np.testing.assert_array_equal(result.fit_amps.sel(metabolite="residual_line"), 0)


def test_criterion_and_its_value_are_recorded():
    """delta_chi2 and threshold follow the stated BIC from the two fits."""
    data = synthetic_fid([(150.0, 1.0)])
    params = parent_params(data)
    parent = fit_data_pseudo_inv(data, params)
    result = add_line_from_residual(
        data, params, search_range_hz=(50.0, 400.0), noise=NOISE_SIGMA
    )
    k_m1 = 3 * 4
    s2 = np.maximum(NOISE_SIGMA**2, result.fit_rss / (2 * 256 - k_m1))
    np.testing.assert_allclose(
        result.line_search_delta_chi2, (parent.fit_rss - result.fit_rss) / s2
    )
    np.testing.assert_allclose(result.line_search_threshold, THRESHOLD)
    np.testing.assert_allclose(result.line_search_noise_sigma, NOISE_SIGMA)
    assert result.attrs["line_search_new_line"] == "residual_line"
    assert result.attrs["line_search_range_hz"] == [50.0, 400.0]
    assert result.attrs["line_search_n_samples"] == 256
    assert result.attrs["line_search_window_hz"] == pytest.approx(2 * 4000 / 256)
    assert "(k_M1 - k_M0) * ln(2 N)" in result.attrs["line_search_criterion"]
    assert result.line_search_delta_chi2.attrs["unit"] == "1"


def test_carried_acquisition_noise_is_used_by_default():
    """noise=None takes the acquisition-tail sigma the input carries."""
    data = synthetic_fid([(150.0, 1.0)], n_spectra=2)
    sigma = xr.DataArray(
        [NOISE_SIGMA, 2 * NOISE_SIGMA],
        dims="x",
        attrs={"noise_source": "acquisition_tail"},
    )
    carried = data.assign_coords(noise_sigma_quadrature=sigma)
    result = add_line_from_residual(
        carried, parent_params(data), search_range_hz=(50.0, 400.0)
    )
    np.testing.assert_allclose(result.line_search_noise_sigma, sigma)
    with pytest.raises(ValueError, match="noise_sigma_quadrature"):
        add_line_from_residual(data, parent_params(data), search_range_hz=(50.0, 400.0))


def test_unselected_spectra_are_not_fitted():
    """Spectra outside the selection keep status 'skipped' and no proposal."""
    data = synthetic_fid([(150.0, 1.0)])
    selection = xr.DataArray([True, False, True, False], dims="x")
    result = add_line_from_residual(
        data,
        parent_params(data),
        search_range_hz=(50.0, 400.0),
        noise=NOISE_SIGMA,
        selection=selection,
    )
    np.testing.assert_array_equal(result.line_search_accepted, [True, False] * 2)
    assert (result.fit_status.isel(x=[1, 3]) == "skipped").all()
    assert np.isnan(result.line_search_start_freq_hz.isel(x=[1, 3])).all()


def test_proposals_keep_the_minimum_separation():
    """A residual peak closer than min_peak_separation_hz is not proposed."""
    data = synthetic_fid([(150.0, 1.0)], n_spectra=1)
    params = def_fit_params(
        data=data, fit_params={**PARENT, "min_peak_separation_hz": 5.0}
    )
    result = add_line_from_residual(
        data, params, search_range_hz=(-10.0, 10.0), noise=NOISE_SIGMA
    )
    start = result.line_search_start_freq_hz.isel(x=0).item()
    fitted_b = result.fit_freqs.sel(metabolite="b").isel(x=0).item()
    assert abs(start - fitted_b) >= 5.0 or np.isnan(start)


def test_invalid_requests_fail_before_fitting():
    """Name, range, window and unsupported options are checked first."""
    data = synthetic_fid(n_spectra=1)
    params = parent_params(data)
    with pytest.raises(ValueError, match="already a line"):
        add_line_from_residual(
            data, params, search_range_hz=(50.0, 400.0), noise=0.05, new_metab_name="b"
        )
    with pytest.raises(ValueError, match="low < high"):
        add_line_from_residual(data, params, search_range_hz=(400.0, 50.0), noise=0.05)
    with pytest.raises(ValueError, match="leaves the spectrum"):
        add_line_from_residual(data, params, search_range_hz=(50.0, 1999.0), noise=0.05)
    with pytest.raises(ValueError, match="line_window_hz"):
        add_line_from_residual(
            data, params, search_range_hz=(50.0, 400.0), noise=0.05, line_window_hz=0
        )
    with pytest.raises(TypeError, match="two frequencies"):
        add_line_from_residual(data, params, search_range_hz=50.0, noise=0.05)
    with pytest.raises(ValueError, match="finite and positive"):
        add_line_from_residual(data, params, search_range_hz=(50.0, 400.0), noise=0.0)
    magnitude = def_fit_params(data=data, fit_params={**PARENT, "use_sse_mag": True})
    with pytest.raises(ValueError, match="use_sse_mag"):
        add_line_from_residual(
            data, magnitude, search_range_hz=(50.0, 400.0), noise=0.05
        )
