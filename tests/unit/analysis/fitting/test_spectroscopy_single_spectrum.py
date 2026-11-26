"""Focused checks for the outer fit's state, sample prefix and empty-input paths."""

from copy import deepcopy

import numpy as np
import pytest

from hypermri.analysis.fitting.models.spectroscopy import fid_model as model


def _patch_grid_searches(monkeypatch, frequency, decay):
    """Route the alternating search's grid steps to fake 4-output searches.

    ``frequency`` and ``decay`` take ``(data, options)`` like the public
    ``fit_freq_pseudo_inv``/``fit_t2_pseudo_inv``; the adapter appends the
    identity line permutation that ``_fit_parameter_grid`` also returns.
    """

    def grid(data, options, *, parameter):
        search = frequency if parameter == "frequency" else decay
        return (*search(data, options), np.arange(len(options["metabs"])))

    monkeypatch.setattr(model, "_fit_parameter_grid", grid)


@pytest.fixture
def fit_arguments() -> dict:
    """Return small, distinct per-metabolite settings for two refinement passes."""
    return {
        "data_slice": np.ones(8, dtype=complex),
        "initial_freq_hz": np.array([10.0, 50.0]),
        "initial_t2_s": np.array([0.02, 0.03]),
        "freq_search_range_hz": np.array([20.0, 40.0]),
        "t2_search_range_s": np.array([0.01, 0.02]),
        "min_t2_s": np.array([0.005, 0.01]),
        "max_t2_s": np.array([0.07, 0.08]),
        "freq_search_range_lower_bound_hz": np.array([0.0, 30.0]),
        "freq_search_range_upper_bound_hz": np.array([20.0, 70.0]),
        "metabs": ["a", "b"],
        "signal_domain": "temporal",
        "sampling_time_axis_s": np.arange(8) * 0.001,
        "rep_fitting": 2,
        "niter": 1,
        "npoints": 3,
        "zoomfactor": 2.0,
    }


@pytest.mark.parametrize("empty_value", [0.0, np.nan, np.inf])
def test_empty_input_skips_numerical_search(fit_arguments, empty_value, monkeypatch):
    """Skip all numerical searches for all-zero or all-NaN input.

    All-zero input keeps the start values; non-finite input reports NaN
    frequencies and T2* values, so it cannot pass for a converged fit.
    """

    def unexpected_score(*args, **kwargs):
        """Fail if the empty-input shortcut reaches amplitude scoring."""
        pytest.fail("Empty input must not be scored or searched.")

    monkeypatch.setattr(model, "_score_fit_parameters", unexpected_score)
    fit_arguments["data_slice"][:] = empty_value
    original = deepcopy(fit_arguments)
    amps, freqs, t2s, freq_diagnostics, t2_diagnostics, excluded = (
        model._fit_single_spectrum_pseudo_inv(**fit_arguments)
    )
    np.testing.assert_array_equal(
        amps,
        np.zeros(2, dtype=complex)
        if empty_value == 0
        else np.full(2, complex(np.nan, np.nan)),
    )
    if empty_value == 0:
        np.testing.assert_array_equal(freqs, fit_arguments["initial_freq_hz"])
        np.testing.assert_array_equal(t2s, fit_arguments["initial_t2_s"])
    else:
        assert np.all(np.isnan(freqs)) and np.all(np.isnan(t2s))
    for diagnostics in (freq_diagnostics, t2_diagnostics):
        assert diagnostics.shape == (2, 2, 1, 3, 3)
        assert np.all(np.isnan(diagnostics))
    assert not np.any(excluded)
    np.testing.assert_equal(fit_arguments, original)


def test_frequency_improvement_keeps_matching_amplitudes(fit_arguments, monkeypatch):
    """Reject a T2* proposal better than the old T2* score but worse than frequency.

    Former separate best values mixed frequency score 6 with T2* score 6.5.
    Keep the complete state at 6 and verify it once more before returning.
    The scripted scores cover the grid passes, so the joint refinement is off.
    """
    scores = iter([10.0, 8.0, 7.0, 6.0, 6.5, 6.0])
    evaluated = []

    def score(data, freqs, t2s, options):
        value = next(scores)
        evaluated.append((freqs.copy(), t2s.copy()))
        return value, np.full(2, value + 0.4j)

    def frequency(data, options):
        return None, options["metabs_freqs_hz"] + 1, None, np.zeros((2, 1, 3, 3))

    def decay(data, options):
        return None, options["metabs_t2s"] + 0.001, None, np.zeros((2, 1, 3, 3))

    monkeypatch.setattr(model, "_score_fit_parameters", score)
    _patch_grid_searches(monkeypatch, frequency, decay)
    amps, freqs, t2s, *_ = model._fit_single_spectrum_pseudo_inv(
        **fit_arguments, joint_refinement=False
    )
    np.testing.assert_array_equal(freqs, fit_arguments["initial_freq_hz"] + 2)
    np.testing.assert_array_equal(t2s, fit_arguments["initial_t2_s"] + 0.001)
    np.testing.assert_array_equal(amps, [6 + 0.4j] * 2)
    np.testing.assert_array_equal(evaluated[-1][0], freqs)
    np.testing.assert_array_equal(evaluated[-1][1], t2s)


@pytest.mark.parametrize("bad_sample", [np.nan, complex(1, np.nan), np.inf])
def test_partial_nonfinite_data_does_not_become_zero_signal(fit_arguments, bad_sample):
    """Report every active parameter of a partly non-finite spectrum as NaN.

    A skipped metabolite keeps its zero amplitude and start values.
    """
    fit_arguments["data_slice"][2] = bad_sample
    amps, freqs, t2s, freq_diag, t2_diag, _ = model._fit_single_spectrum_pseudo_inv(
        **fit_arguments, skip_fitting_mask=[False, True]
    )
    assert np.isnan(amps[0]) and np.isnan(freqs[0]) and np.isnan(t2s[0])
    assert amps[1] == 0
    assert freqs[1] == fit_arguments["initial_freq_hz"][1]
    assert t2s[1] == fit_arguments["initial_t2_s"][1]
    assert np.all(np.isnan(freq_diag)) and np.all(np.isnan(t2_diag))


def test_final_amplitudes_match_independent_complex_lstsq(fit_arguments):
    """The returned parameters must reconstruct the least-squares fitted FID."""
    time = fit_arguments["sampling_time_axis_s"]
    rng = np.random.default_rng(841)
    fit_arguments["data_slice"] = rng.normal(size=8) + 1j * rng.normal(size=8)
    amps, freqs, t2s, *_ = model._fit_single_spectrum_pseudo_inv(**fit_arguments)
    basis = np.exp(2j * np.pi * time[:, None] * freqs - time[:, None] / t2s)
    reference = np.linalg.lstsq(basis, fit_arguments["data_slice"], rcond=None)[0]
    np.testing.assert_allclose(amps, reference, rtol=1e-12, atol=1e-12)


def test_invalid_edited_bounds_are_rejected(fit_arguments):
    """Do not evaluate or silently clip a starting point outside manual bounds."""
    fit_arguments["initial_freq_hz"][0] = -1
    with pytest.raises(ValueError, match="within finite.*bounds"):
        model._fit_single_spectrum_pseudo_inv(**fit_arguments)


def test_spectral_input_is_transformed_before_truncation(fit_arguments):
    """Fit the same FID prefix from a longer temporal or centered spectral input."""
    time_s = fit_arguments["sampling_time_axis_s"]
    basis = np.exp(
        2j * np.pi * time_s[:, None] * fit_arguments["initial_freq_hz"]
        - time_s[:, None] / fit_arguments["initial_t2_s"]
    )
    expected_amplitudes = np.array([1.0 + 0.2j, 0.05 - 0.01j])
    # A large tail makes cutting frequency bins instead of FID samples detectable.
    fid = np.concatenate([basis @ expected_amplitudes, np.full(8, 5j)])
    fit_arguments.update(
        data_slice=fid,
        freq_search_range_hz=np.zeros(2),
        t2_search_range_s=np.zeros(2),
    )
    original = deepcopy(fit_arguments)
    temporal_result = model._fit_single_spectrum_pseudo_inv(**fit_arguments)
    spectral_arguments = {
        **fit_arguments,
        "data_slice": np.fft.fftshift(np.fft.fft(fid)),
        "signal_domain": "spectral",
    }
    original_spectral = deepcopy(spectral_arguments)
    spectral_result = model._fit_single_spectrum_pseudo_inv(**spectral_arguments)
    np.testing.assert_allclose(temporal_result[0], expected_amplitudes, atol=1e-12)
    for temporal, spectral in zip(temporal_result, spectral_result, strict=True):
        np.testing.assert_allclose(temporal, spectral, rtol=1e-11, atol=1e-12)
    np.testing.assert_equal(fit_arguments, original)
    np.testing.assert_equal(spectral_arguments, original_spectral)


def test_outer_search_retains_best_values_and_shrinks_widths(
    fit_arguments, monkeypatch
):
    """Retain accepted parameters on ties/worse scores and keep original widths.

    The second frequency proposal improves its historical frequency score but
    is worse than the jointly accepted T2* state. Neither worse proposal may
    replace the first pass's accepted result. The scripted scores cover the
    grid passes, so the joint refinement is off.
    """
    original = deepcopy(fit_arguments)
    scores = iter([10.0, 8.0, 7.0, 7.5, 9.0, 7.0])
    frequency_calls, t2_calls = [], []

    def score_candidate(values, freq_hz, t2_star_s, fit_params):
        """Provide controlled SSE values and distinguishable complex amplitudes."""
        score = next(scores)
        return score, np.full(2, score + 0.25j)

    def search_frequency(values, fit_params):
        """Propose a new frequency each pass and retain its received state."""
        frequency_calls.append(deepcopy(fit_params))
        proposal = original["initial_freq_hz"] + len(frequency_calls)
        diagnostics = np.full((2, 1, 3, 3), len(frequency_calls), dtype=float)
        return None, proposal, None, diagnostics

    def search_t2(values, fit_params):
        """Propose a new T2* each pass and retain its received state."""
        t2_calls.append(deepcopy(fit_params))
        proposal = original["initial_t2_s"] + len(t2_calls) * 0.001
        diagnostics = np.full((2, 1, 3, 3), 10 + len(t2_calls), dtype=float)
        return None, proposal, None, diagnostics

    monkeypatch.setattr(model, "_score_fit_parameters", score_candidate)
    _patch_grid_searches(monkeypatch, search_frequency, search_t2)
    amps, freqs, t2s, freq_diagnostics, t2_diagnostics, excluded = (
        model._fit_single_spectrum_pseudo_inv(**fit_arguments, joint_refinement=False)
    )
    np.testing.assert_array_equal(amps, [7.0 + 0.25j] * 2)
    np.testing.assert_array_equal(freqs, original["initial_freq_hz"] + 1)
    np.testing.assert_array_equal(t2s, original["initial_t2_s"] + 0.001)
    np.testing.assert_array_equal(frequency_calls[1]["metabs_t2s"], t2s)
    for call in t2_calls:
        np.testing.assert_array_equal(call["metabs_freqs_hz"], freqs)
    for index in range(2):
        np.testing.assert_array_equal(
            frequency_calls[index]["metabs_freq_ranges"],
            original["freq_search_range_hz"] / 2**index,
        )
        np.testing.assert_array_equal(
            t2_calls[index]["metabs_t2_ranges"],
            original["t2_search_range_s"] / 2**index,
        )
        assert np.all(freq_diagnostics[:, index] == index + 1)
        assert np.all(t2_diagnostics[:, index] == index + 11)
    assert not np.any(excluded)
    np.testing.assert_equal(fit_arguments, original)
