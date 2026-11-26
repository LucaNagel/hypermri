"""Check the claims printed by the t2_coupling and phase-band example script.

The script's synthetic fixtures are seeded, noiseless lines plus complex white
noise (see its module docstring). Three noise realizations keep this test
short; the checked properties are constraints of the fit settings and hold for
every realization, not only on average.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import example_t2_coupling_and_phase_band as example

N_REPETITIONS = 3


def _t2s(result):
    return result.fit_t2s.transpose("repetition", "metabolite").values


def test_t2_coupling_examples_respect_fixed_ratio_and_band(capsys):
    results = example.t2_coupling_examples(N_REPETITIONS)

    for result in (
        results["equal"]["fixed 1:1  {'c': 'b'}"],
        results["half"]["fixed 1:1 (wrong)"],
    ):
        t2s = _t2s(result)
        np.testing.assert_allclose(t2s[:, 1], t2s[:, 0], rtol=1e-12)

    half = results["half"]
    t2s = _t2s(half["fixed 0.5  {'c': ('b', 0.5)}"])
    np.testing.assert_allclose(t2s[:, 1], 0.5 * t2s[:, 0], rtol=1e-12)
    # A band that excludes the true ratio 0.5 holds the ratio at or near its
    # lower edge; noise can lift single realizations slightly above it.
    t2s = _t2s(half["band (0.75, 1.25) (excludes 0.5)"])
    ratio = t2s[:, 1] / t2s[:, 0]
    assert np.all((ratio >= 0.75 * (1 - 1e-12)) & (ratio < 0.85))
    t2s = _t2s(half["band (0.4, 0.6), start 30/15 ms"])
    ratio = t2s[:, 1] / t2s[:, 0]
    assert np.all((ratio >= 0.4 * (1 - 1e-12)) & (ratio <= 0.6 * (1 + 1e-12)))
    assert "leaves the ratio band (0.4, 0.6)" in capsys.readouterr().out


def test_phase_band_examples_keep_lines_on_the_phase_law(tmp_path):
    signals, results = example.phase_band_examples(N_REPETITIONS)
    assert signals.sizes == {"repetition": N_REPETITIONS, "acquisition_idx": 256}

    # Relative to 'ref', both lines may deviate by at most twice the tolerance.
    deviation = example.phase_deviation_deg(results["band 5 deg"], "ref")
    assert np.abs(deviation).max() <= 10.0 + 1e-6
    deviation = example.phase_deviation_deg(
        results["band 0 deg (exactly on law)"], "ref"
    )
    np.testing.assert_allclose(deviation, 0.0, atol=1e-6)

    # tau = 0 cannot give 'ref' at -250 Hz its 135 deg law phase.
    wrong_tau = np.abs(
        results["band 5 deg, wrong tau = 0"].fit_amps.sel(metabolite="ref")
    )
    assert wrong_tau.max() < 1e-6
    correct_tau = np.abs(results["band 5 deg"].fit_amps.sel(metabolite="ref"))
    assert correct_tau.min() == pytest.approx(8.0, rel=0.1)

    figure = tmp_path / "phase_band.png"
    example.plot_phase_band(signals, results, figure)
    assert figure.stat().st_size > 0
