"""Verify final numerical errors independently of optimizer diagnostics."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv


@pytest.mark.parametrize("domain", ["temporal", "spectral"])
@pytest.mark.parametrize("use_sse_mag", [False, True])
def test_final_rss_uses_returned_state_and_fitted_sample_prefix(domain, use_sse_mag):
    """RSS uses the fitted FID prefix, while magnitude objective uses its FFT."""
    time = np.arange(24) * 0.001
    rng = np.random.default_rng(709)
    fid = (1 + 0.4j) * np.exp(2j * np.pi * 15 * time - time / 0.027)
    fid += 0.03 * (rng.normal(size=time.size) + 1j * rng.normal(size=time.size))
    data = xr.DataArray(
        fid if domain == "temporal" else np.fft.fftshift(np.fft.fft(fid)),
        dims="acquisition_idx",
        coords={"acquisition_idx": time},
        attrs={"signal_domain": domain},
    )
    params = def_fit_params(
        data,
        {
            "metabs": ["peak"],
            "metabs_freqs_hz": [12.0],
            "nsamplepoints": 16,
            "npoints": 4,
            "rep_fitting": 2,
            "metabs_freq_ranges": [20.0],
            "metabs_t2_ranges": [0.03],
            "use_sse_mag": use_sse_mag,
        },
    )
    result = fit_data_pseudo_inv(data, params)
    reconstructed = complex(result.fit_amps.item()) * np.exp(
        2j * np.pi * result.fit_freqs.item() * time[:16]
        - time[:16] / result.fit_t2s.item()
    )
    rss = np.sum(np.abs(fid[:16] - reconstructed) ** 2)
    objective = (
        np.sum((np.abs(np.fft.fft(fid[:16])) - np.abs(np.fft.fft(reconstructed))) ** 2)
        if use_sse_mag
        else rss
    )
    assert result.fit_rss.item() == pytest.approx(rss, abs=1e-12)
    assert result.fit_objective.item() == pytest.approx(objective, abs=1e-12)
    assert result.attrs["fit_objective_name"] == (
        "spectral_magnitude_rss" if use_sse_mag else "temporal_complex_rss"
    )
    for suffix in ("freq", "t2"):
        legacy = result[f"fit_r2_{suffix}"]
        canonical = result[f"fit_search_diagnostics_{suffix}"]
        np.testing.assert_equal(legacy.values, canonical.values)
        assert canonical.search_metric.values.tolist() == [
            "candidate",
            "temporal_rss",
            "spectral_magnitude_rss",
        ]
        assert legacy.attrs.get("unit") != "R2"
        assert legacy.attrs["deprecated_alias_for"] == canonical.name


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_empty_invalid_result_errors(value):
    """An invalid spectrum must not appear to have zero residual error."""
    data = xr.DataArray(
        np.full(8, value, dtype=complex),
        dims="acquisition_idx",
        coords={"acquisition_idx": np.arange(8) * 0.001},
        attrs={"signal_domain": "temporal"},
    )
    params = def_fit_params(data, {"metabs": ["peak"], "metabs_freqs_hz": [0.0]})
    result = fit_data_pseudo_inv(data, params)
    if value == 0:
        assert result.fit_rss.item() == result.fit_objective.item() == 0
    else:
        assert np.isnan(result.fit_rss.item())
        assert np.isnan(result.fit_objective.item())
