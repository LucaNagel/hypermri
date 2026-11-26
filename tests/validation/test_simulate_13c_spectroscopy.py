"""Small synthetic checks of the 13C simulation and its summary denominators."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hypermri.analysis.fitting.spectroscopy import def_fit_params
from hypermri.analysis.spectroscopy import estimate_fit_covariance, fit_spectrum
from hypermri.analysis.spectroscopy.metabolites import get_metab_cs_ppm
from hypermri.utils.physics import get_gmr
from scripts import simulate_13c_spectroscopy as simulation


def _config(arguments=()):
    return simulation._configuration(simulation._parser().parse_args(arguments))


def test_simulation_has_256_samples_correct_snr_and_independent_quadratures():
    config = _config(["--repeats", "256"])
    ratios = config["ratio_profiles"]["weak_products"]
    t2_ms = config["t2_profiles_ms"]["mixed"]
    data, clean, frequencies, sigma = simulation._simulate(
        config, ratios, t2_ms, 5.0, np.random.default_rng(42)
    )
    assert data.dims == ("realization", "acquisition_idx")
    assert data.shape == (256, 256)
    assert data.dtype == np.complex128
    np.testing.assert_allclose(data.sampling_time_axis_s, np.arange(256) * 0.0004)
    assert abs(ratios[0]) / sigma == pytest.approx(5.0)
    assert abs(ratios[-1]) / sigma == pytest.approx(0.15)
    assert clean[0] == pytest.approx(sum(ratios) * np.exp(0.35j))
    np.testing.assert_allclose(
        frequencies,
        config["nominal_frequencies_hz"],
    )
    noise = data.values - clean
    # 65,536 independent samples: a 2% tolerance is >5 SD for each SD estimate.
    assert noise.real.std(ddof=1) == pytest.approx(sigma, rel=0.02)
    assert noise.imag.std(ddof=1) == pytest.approx(sigma, rel=0.02)
    assert np.mean(abs(noise) ** 2) == pytest.approx(2 * sigma**2, rel=0.02)
    assert abs(np.corrcoef(noise.real.ravel(), noise.imag.ravel())[0, 1]) < 0.02
    repeat = simulation._simulate(config, ratios, t2_ms, 5.0, np.random.default_rng(42))
    np.testing.assert_array_equal(data, repeat[0])


def test_default_frequencies_follow_requested_7t_ppm_conversion_without_offsets():
    config = _config(["--repeats", "2"])
    expected = (
        7
        * get_gmr(nucleus="13c")
        * np.array(
            [
                get_metab_cs_ppm(metab=name.lower())
                - get_metab_cs_ppm(metab="pyruvate")
                for name in config["metabolites"]
            ]
        )
    )
    assert config["larmor_mhz"] == pytest.approx(74.9588)
    assert config["field_strength_t"] == pytest.approx(7.0)
    assert config["gmr_mhz_per_t"] == pytest.approx(10.7084)
    np.testing.assert_allclose(config["nominal_frequencies_hz"], expected, atol=1e-10)
    np.testing.assert_allclose(
        config["fit_options"]["metabs_freqs_hz"], expected, atol=1e-10
    )
    # Numerical reference: the user's ppm offsets and the repository 13C GMR.
    np.testing.assert_allclose(
        expected,
        [0, 920.0443112, 406.5765312, 631.4529312, -755.2848688],
        rtol=0,
        atol=1e-9,
    )
    ratios = config["ratio_profiles"]["balanced"]
    t2_ms = config["t2_profiles_ms"]["mixed"]
    data, clean, frequencies, _ = simulation._simulate(
        config, ratios, t2_ms, 10, np.random.default_rng(1)
    )
    time_s = data.sampling_time_axis_s.values
    independent_clean = sum(
        amplitude
        * np.exp(0.35j + 2j * np.pi * frequency * time_s - time_s / (t2 * 1e-3))
        for amplitude, frequency, t2 in zip(ratios, expected, t2_ms, strict=True)
    )
    np.testing.assert_allclose(frequencies, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(clean, independent_clean, rtol=1e-12, atol=1e-12)


def test_explicit_larmor_override_remains_supported():
    config = _config(["--larmor-mhz", "80"])
    ppm = np.asarray(config["chemical_shifts_ppm"])
    assert config["larmor_mhz"] == 80
    np.testing.assert_allclose(config["nominal_frequencies_hz"], 80 * (ppm - ppm[0]))
    assert config["field_strength_t"] == pytest.approx(80 / get_gmr(nucleus="13c"))


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--snr", "0"], "snr"),
        (["--snr", "nan"], "snr"),
        (["--snr", "1", "1"], "unique"),
        (["--repeats", "1"], "repeats"),
        (["--n-jobs", "0"], "n-jobs"),
        (["--seed", "-1"], "seed"),
        (["--ratios", "1", "1", "0", "1", "1"], "ratios"),
        (["--t2-ms", "5", "20", "20", "20", "20"], "t2-ms"),
        (["--t2-ms", "20", "80", "20", "20", "20"], "t2-ms"),
        (["--dwell-ms", "1"], "Nyquist"),
        (["--larmor-mhz", "1"], "disjoint"),
    ],
)
def test_configuration_rejects_invalid_simulations(arguments, message):
    with pytest.raises(ValueError, match=message):
        _config(arguments)


def test_custom_profiles_normalize_pyruvate_and_keep_metabolite_order():
    config = _config(["--ratios", "2", "1", "0.4", "0.2", "0.1"])
    assert config["ratio_profiles"] == {"custom": [1, 0.5, 0.2, 0.1, 0.05]}
    assert config["metabolites"][-1] == "Bicarbonate"


def _summary_fixture():
    table = pd.DataFrame(
        {
            "ratio_profile": "test",
            "t2_profile": "test",
            "snr_pyruvate": 10.0,
            "metabolite": "Pyruvate",
            "snr_metabolite": 10.0,
            "fit_status": "success",
            "covariance_status": ["success"] * 3 + ["parameter_bound"],
            "amplitude_abs_status": ["success"] * 3 + ["parameter_bound"],
        }
    )
    for parameter, truth in (
        ("amplitude", 1.0),
        ("frequency", 40.0),
        ("t2", 25.0),
        ("fwhm", 11.0),
    ):
        table[f"{parameter}_true"] = truth
        table[f"{parameter}_fit"] = truth + np.array([-0.1, 0, 0.1, 9])
        table[f"{parameter}_std"] = [0.2, 0.2, 0.2, np.nan]
    return table


def test_summary_preserves_bad_uncertainty_fits_and_compares_identical_subsets():
    summary = simulation._summarize(_summary_fixture()).set_index("parameter")
    amplitude = summary.loc["amplitude"]
    assert amplitude.n_total == amplitude.n_fit == 4
    assert amplitude.n_uncertainty == 3
    assert amplitude.uncertainty_available_fraction == 0.75
    assert amplitude.bias == pytest.approx(2.25)
    assert amplitude.rmse == pytest.approx(np.sqrt(81.02 / 4))
    assert amplitude.empirical_std_eligible == pytest.approx(0.1)
    assert amplitude.predicted_std_rms_eligible == pytest.approx(0.2)
    assert amplitude.predicted_to_empirical_std == pytest.approx(2)
    assert amplitude.coverage_68_eligible == amplitude.coverage_95_eligible == 1
    assert amplitude.empirical_std_all_fits > 40 * amplitude.empirical_std_eligible
    assert summary.loc["t2", "unit"] == "ms"


def test_near_zero_and_failed_fits_have_explicit_parameter_specific_counts():
    table = _summary_fixture()
    table.loc[0, "amplitude_abs_status"] = "near_zero"
    table.loc[1, "fit_status"] = "failed"
    result = simulation._summarize(table).set_index("parameter")
    assert result.loc["amplitude", "n_uncertainty"] == 1
    assert result.loc["frequency", "n_uncertainty"] == 2
    assert result.loc["amplitude", "n_fit"] == 3
    assert np.isnan(result.loc["amplitude", "empirical_std_eligible"])
    table["fit_status"] = "failed"
    result = simulation._summarize(table)
    assert (result.n_fit == 0).all()
    assert (result.uncertainty_available_fraction == 0).all()
    assert result.coverage_95_eligible.isna().all()
    assert result.fit_mean.isna().all()


def test_canonical_fitter_recovers_five_peak_noise_free_signal():
    config = _config(["--repeats", "2"])
    ratios, t2 = config["ratio_profiles"]["balanced"], config["t2_profiles_ms"]["mixed"]
    data, clean, frequencies, sigma = simulation._simulate(
        config, ratios, t2, 100, np.random.default_rng(42)
    )
    data[:] = clean
    params = def_fit_params(data, config["fit_options"])
    fit = fit_spectrum(data, params, noise=sigma, show_tqdm=False)
    errors = estimate_fit_covariance(fit, params)
    table = simulation._trial_table(fit, errors, ratios, t2, frequencies, sigma)
    # Independent clean truth; tolerances allow the production grid resolution.
    np.testing.assert_allclose(table.amplitude_fit, table.amplitude_true, rtol=0.003)
    np.testing.assert_allclose(table.frequency_fit, table.frequency_true, atol=0.05)
    np.testing.assert_allclose(table.t2_fit, table.t2_true, atol=0.1)
    assert table.covariance_status.eq("success").all()
    assert (table.amplitude_std > 0).all()
    np.testing.assert_allclose(
        table.fwhm_std, 1000 * table.t2_std / (np.pi * table.t2_fit**2)
    )


def test_script_exports_csv_figures_and_reproducible_configuration(tmp_path):
    simulation._main(
        [
            "--output",
            str(tmp_path / "output"),
            "--snr",
            "100",
            "--repeats",
            "2",
            "--ratios",
            "1",
            "0.5",
            "0.3",
            "0.25",
            "0.15",
            "--t2-ms",
            "35",
            "25",
            "20",
            "18",
            "12",
        ]
    )
    output = tmp_path / "output"
    trials = pd.read_csv(output / "trials.csv")
    summary = pd.read_csv(output / "summary.csv")
    config = json.loads((output / "configuration.json").read_text())
    assert len(trials) == 10
    assert len(summary) == 20
    assert config["n_spectra_fitted"] == 2
    assert config["effective_fit_options"]["npoints"] == 31
    expected = dict(
        zip(config["metabolites"], config["nominal_frequencies_hz"], strict=True)
    )
    np.testing.assert_allclose(
        trials.frequency_true, trials.metabolite.map(expected), atol=1e-10
    )
    assert config["field_strength_t"] == pytest.approx(7)
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 4
    spectra = np.load(output / "example_spectra.npz")
    assert spectra["noisy"].shape == spectra["fitted"].shape == (1, 256)
    with pytest.raises(SystemExit):
        simulation._main(["--output", str(output)])
