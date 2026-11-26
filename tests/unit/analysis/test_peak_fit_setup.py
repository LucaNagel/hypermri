"""Tests of the peak table to fit configuration step (invented values only)."""

import pytest

from hypermri.analysis.spectroscopy.peak_fit_setup import (
    FIT_RESULT_SEEDING_KEYS,
    PeakSetting,
    manual_peak_setting,
    peak_fit_config,
    peak_settings_from_detection,
    unique_peak_names,
)


def _config(peaks, **options):
    settings = {
        "freq_step_hz": 10.0,
        "snr_threshold": 0.0,
        "min_peak_separation_hz": 5.0,
        "nsamplepoints": 32,
        **options,
    }
    return peak_fit_config(peaks, **settings)


def test_unique_peak_names_fills_empty_and_repeated_names():
    assert unique_peak_names(["pyr", " ", "pyr", " lac "]) == (
        ["pyr", "metab_2", "pyr_2", "lac"],
        True,
    )
    assert unique_peak_names(["pyr", "lac"]) == (["pyr", "lac"], False)


def test_t2_values_are_ordered_clamped_and_converted_to_seconds():
    config, note = _config([PeakSetting("pyr", 0.0, -20.0, 500.0, 80.0, 4.0)])

    assert config["metabs_freq_ranges"] == [20.0]
    assert config["min_t2_s"] == [0.004]
    assert config["max_t2_s"] == [0.08]
    assert config["initial_t2s_s"] == [0.08]
    assert config["metabs_t2_ranges"] == [pytest.approx(0.076)]
    assert note == ""


def test_fixed_lines_window_and_disabled_lines():
    peaks = [
        PeakSetting("pyr", 0.0, 20.0, 30.0, 5.0, 100.0, fixed_freq=True),
        PeakSetting("lac", 390.0, 20.0, 30.0, 5.0, 100.0, fixed_t2=True, fit=False),
        PeakSetting(
            "ala", 200.0, 20.0, 30.0, 5.0, 100.0, seed_from_fit=True, t2_window_ms=8.0
        ),
        PeakSetting("bic", -300.0, 20.0, 30.0, 5.0, 100.0, t2_window_ms=8.0),
    ]
    config, _ = _config(peaks)

    assert config["metabs_freq_ranges"] == [0.0, 20.0, 20.0, 20.0]
    assert config["metabs_t2_ranges"] == [
        pytest.approx(0.095),
        0.0,
        0.008,
        pytest.approx(0.095),
    ]
    assert config["fit_init_t2_windows_s"] == [None, None, 0.008, None]
    assert config["use_fit_results_init"] == [False, False, True, False]
    assert config["skip_fitting"] == [False, True, False, False]
    assert set(FIT_RESULT_SEEDING_KEYS) - set(config) == {"source_metab_names"}


def test_resolution_is_automatic_unless_given():
    narrow = [PeakSetting("pyr", 0.0, 20.0, 30.0, 5.0, 100.0)]
    wide = [PeakSetting("pyr", 0.0, 400.0, 30.0, 5.0, 100.0)]

    assert _config(narrow)[0]["npoints"] == 31
    assert _config(wide)[0]["npoints"] == 51
    config, _ = _config(narrow, rep_fitting=0, npoints=1)
    assert (config["rep_fitting"], config["npoints"]) == (1, 3)


def test_close_start_frequencies_and_renamed_lines_give_a_note():
    peaks = [
        PeakSetting("pyr", 0.0, 20.0, 30.0, 5.0, 100.0),
        PeakSetting("pyr", 2.0, 20.0, 30.0, 5.0, 100.0),
    ]
    config, note = _config(peaks)

    assert config["metabs"] == ["pyr", "pyr_2"]
    assert note.startswith("Metabolite names were normalized")
    assert "closer than the requested 5 Hz" in note


def test_an_empty_table_is_rejected():
    with pytest.raises(ValueError, match="at least one spectral line"):
        _config([])


def test_rows_from_detection_and_by_hand():
    detected = {
        "cluster_summary": [
            {"name": "peak_1", "center_hz": 10.0},
            {"name": "peak_2", "center_hz": -50.0},
        ],
        "freq_ranges_hz": [12.0, 14.0],
        "initial_t2s_s": [0.02, 0.03],
    }

    rows = peak_settings_from_detection(detected, min_t2_ms=5.0, max_t2_ms=100.0)

    assert rows == [
        PeakSetting("peak_1", 10.0, 12.0, 20.0, 5.0, 100.0),
        PeakSetting("peak_2", -50.0, 14.0, 30.0, 5.0, 100.0),
    ]
    assert manual_peak_setting(
        7.0, name="manual_1", freq_range_hz=20.0, min_t2_ms=100.0, max_t2_ms=10.0
    ) == PeakSetting("manual_1", 7.0, 20.0, 55.0, 10.0, 100.0)
