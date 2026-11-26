"""Check search diagnostics against an independent one-peak least-squares result."""

from copy import deepcopy

import numpy as np
import pytest

from hypermri.analysis.fitting.models.spectroscopy import fid_model as fid_pseudo_inv


@pytest.mark.parametrize("search_name", ["fit_freq_pseudo_inv", "fit_t2_pseudo_inv"])
@pytest.mark.parametrize("use_sse_mag", [False, True])
def test_grid_search_scores_and_input_domains_agree(search_name, use_sse_mag):
    """Verify both SSE columns, criterion selection and domain equivalence.

    A 64-point, phased noisy FID is enough to distinguish the two T2* scoring
    criteria. Use the closed-form amplitude of a single basis vector as an
    independent reference, checking pinv and ridge without a large simulation.
    """
    time_s = np.arange(64) * 0.0004
    rng = np.random.default_rng(20260904)
    fid = (1 + 0.3j) * np.exp(2j * np.pi * -70 * time_s - time_s / 0.025)
    fid += 0.1 * (rng.normal(size=time_s.size) + 1j * rng.normal(size=time_s.size))
    ridge_lambda = 0.25 if use_sse_mag else 0.0
    params = {
        "metabs": ["pyruvate"],
        "signal_domain": "temporal",
        "sampling_time_axis_s": time_s,
        "npoints": 7,
        "niter": 1,
        "metabs_freqs_hz": np.array([-65.0]),
        "initial_t2s_s": np.array([0.03]),
        "metabs_freq_ranges": np.array([30.0]),
        "freq_range_lower_bound_hz": np.array([-80.0]),
        "freq_range_upper_bound_hz": np.array([-50.0]),
        "metabs_t2_ranges": np.array([0.08]),
        "min_t2_s": np.array([0.005]),
        "max_t2_s": np.array([0.08]),
        "amp_solver": "ridge" if use_sse_mag else "pinv",
        "amp_ridge_lambda": ridge_lambda,
        "use_sse_mag": use_sse_mag,
    }
    search = getattr(fid_pseudo_inv, search_name)
    original_fid, original_params = fid.copy(), deepcopy(params)
    temporal_result = search(fid, params)
    np.testing.assert_array_equal(fid, original_fid)
    np.testing.assert_equal(params, original_params)

    spectral_input = np.fft.fftshift(np.fft.fft(fid))
    original_spectrum = spectral_input.copy()
    spectral_result = search(spectral_input, {**params, "signal_domain": "spectral"})
    np.testing.assert_array_equal(spectral_input, original_spectrum)
    for temporal_values, spectral_values in zip(
        temporal_result, spectral_result, strict=True
    ):
        np.testing.assert_allclose(
            temporal_values, spectral_values, rtol=1e-12, atol=1e-12
        )

    diagnostics = temporal_result[3][0, 0]
    for candidate, temporal_sse, spectral_sse in diagnostics:
        frequency = candidate if search_name == "fit_freq_pseudo_inv" else -65.0
        t2_s = candidate if search_name == "fit_t2_pseudo_inv" else 0.03
        if t2_s < 0.005 or t2_s > 0.08:
            assert np.isinf(temporal_sse) and np.isinf(spectral_sse)
            continue
        basis = np.exp(2j * np.pi * frequency * time_s - time_s / t2_s)
        amplitude = np.vdot(basis, fid) / (np.vdot(basis, basis) + ridge_lambda)
        fitted = basis * amplitude
        expected_errors = [
            np.sum(np.abs(fid - fitted) ** 2),
            np.sum((np.abs(np.fft.fft(fid)) - np.abs(np.fft.fft(fitted))) ** 2),
        ]
        np.testing.assert_allclose(
            [temporal_sse, spectral_sse], expected_errors, rtol=1e-12, atol=1e-12
        )

    best_index = np.argmin(diagnostics[:, 2 if use_sse_mag else 1])
    assert temporal_result[1][0] == pytest.approx(diagnostics[best_index, 0])
    if search_name == "fit_t2_pseudo_inv":
        # A swap of the scoring criteria must be detectable in this fixture.
        assert np.argmin(diagnostics[:, 1]) != np.argmin(diagnostics[:, 2])


@pytest.mark.parametrize("parameter", ["frequency", "t2"])
@pytest.mark.parametrize("npoints", [1, 4, 5])
def test_grid_only_evaluates_bounds_and_retains_current(
    parameter, npoints, monkeypatch
):
    """An illegal unconstrained optimum must never be scored and then clipped.

    The score's unique minimum is the initial state. Even grids must keep this
    state as a candidate; no later iteration may worsen it.
    """
    params = {
        "metabs": ["peak"],
        "signal_domain": "temporal",
        "sampling_time_axis_s": np.arange(8) * 0.001,
        "metabs_freqs_hz": np.array([0.0]),
        "metabs_t2s": np.array([0.025]),
        "metabs_freq_ranges": np.array([80.0]),
        "metabs_t2_ranges": np.array([0.1]),
        "freq_range_lower_bound_hz": np.array([-10.0]),
        "freq_range_upper_bound_hz": np.array([10.0]),
        "min_t2_s": np.array([0.02]),
        "max_t2_s": np.array([0.04]),
        "npoints": npoints,
        "niter": 2,
    }
    seen = []

    def evaluate(data, spectrum, freqs, t2s, options):
        assert -10 <= freqs[0] <= 10
        assert 0.02 <= t2s[0] <= 0.04
        seen.append((freqs[0], t2s[0]))
        score = freqs[0] ** 2 + (t2s[0] - 0.025) ** 2
        return score, score

    monkeypatch.setattr(fid_pseudo_inv, "_evaluate_grid_candidate", evaluate)
    search = (
        fid_pseudo_inv.fit_freq_pseudo_inv
        if parameter == "frequency"
        else fid_pseudo_inv.fit_t2_pseudo_inv
    )
    _, fitted, _, diagnostics = search(np.ones(8, dtype=complex), params)
    assert seen
    np.testing.assert_array_equal(fitted, [0 if parameter == "frequency" else 0.025])
    lower, upper = (-10, 10) if parameter == "frequency" else (0.02, 0.04)
    candidates = diagnostics[..., 0]
    outside = (candidates < lower) | (candidates > upper)
    assert np.all(np.isinf(diagnostics[..., 1:][outside]))
