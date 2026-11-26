"""Tests of the browser fit tool state and fit run (invented spectra only)."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.peak_fit_setup import (
    frequency_axis_hz,
    frequency_step_hz,
)
from hypermri.visualization.fit_tool import (
    FitToolState,
    LineCoupling,
    LineSeeding,
    add_peak,
    add_residual_line,
    align_spectra,
    detect_peaks,
    edit_peak,
    fit_tool_settings,
    lines_from_fit,
    mask_fit_table,
    next_problem_voxel,
    refit_voxel,
    remove_peak,
    run_fit,
    set_fit_option,
    store_fit_on_scan,
    stored_fit,
    voxel_fit_table,
)

DIMS = (
    "read_idx",
    "phase_idx",
    "slice_idx",
    "repetition_idx",
    "channel_idx",
    "acquisition_idx",
)
DWELL_S = 2e-4
# Invented lines (frequency in Hz, amplitude) with T2* = 30 ms.
LINES = ((-800.0, 1.0), (300.0, 0.4))


def _fids(n_spec=128, noise=0.01):
    """Invented FIDs of 3 x 2 voxels whose amplitude grows with read_idx."""
    time_s = np.arange(n_spec) * DWELL_S
    fid = sum(
        amplitude * np.exp(2j * np.pi * freq_hz * time_s - time_s / 0.03)
        for freq_hz, amplitude in LINES
    )
    scale = np.arange(1, 4)[:, None, None, None, None, None]
    rng = np.random.default_rng(0)
    shape = (3, 2, 1, 1, 1, n_spec)
    values = scale * fid + noise * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
    return xr.DataArray(
        values.astype(np.complex64),
        dims=DIMS,
        coords={"sampling_time_axis_s": ("acquisition_idx", time_s)},
        attrs={"dwell_time_s": DWELL_S, "signal_domain": "temporal"},
    )


def test_frequency_axis_from_dwell_time_and_its_step():
    data = _fids(n_spec=64)
    axis = frequency_axis_hz(data)
    assert axis.shape == (64,)
    assert frequency_step_hz(data) == pytest.approx(1 / (64 * DWELL_S))
    with pytest.raises(ValueError, match="acquisition_idx"):
        frequency_axis_hz(data.isel(acquisition_idx=slice(0, 1)))


def test_detection_fills_the_table_and_replaces_lines_added_by_hand():
    tool = FitToolState(n_jobs=1)
    add_peak(tool, 100.0)
    detect_peaks(tool, _fids())

    step_hz = frequency_step_hz(_fids())
    found = sorted(peak.freq_hz for peak in tool.peaks)
    assert len(found) == 2
    for freq_hz, (true_hz, _) in zip(found, LINES, strict=True):
        assert freq_hz == pytest.approx(true_hz, abs=step_hz)
    assert tool.message == "Found 2 line(s)."


def test_detection_without_lines_raises():
    tool = FitToolState(peak_height_snr=1e6)
    with pytest.raises(ValueError):
        detect_peaks(tool, _fids())


def test_lines_added_edited_and_removed_by_hand():
    tool = FitToolState(freq_search_range_hz=30.0, min_t2_ms=4.0, max_t2_ms=90.0)
    add_peak(tool, -800.0)
    add_peak(tool, 300.0)
    assert [peak.name for peak in tool.peaks] == ["manual_1", "manual_2"]
    assert tool.peaks[0].freq_range_hz == 30.0
    assert (tool.peaks[0].min_t2_ms, tool.peaks[0].max_t2_ms) == (4.0, 90.0)

    edit_peak(tool, 0, "name", "pyr")
    edit_peak(tool, 1, "name", "pyr")
    edit_peak(tool, 1, "freq_hz", "310.5")
    edit_peak(tool, 1, "fixed_freq", True)
    assert [peak.name for peak in tool.peaks] == ["pyr", "pyr_2"]
    assert tool.peaks[1].freq_hz == 310.5
    assert tool.peaks[1].fixed_freq is True

    remove_peak(tool, 0)
    assert [peak.name for peak in tool.peaks] == ["pyr_2"]
    add_peak(tool, 0.0)
    assert tool.peaks[-1].name == "manual_3"


@pytest.mark.parametrize(
    ("index", "column", "value"),
    [
        (5, "freq_hz", 1.0),
        (0, "colour", 1.0),
        (0, "freq_hz", "nan"),
        (0, "freq_hz", "x"),
    ],
)
def test_invalid_edits_raise_and_keep_the_table(index, column, value):
    tool = FitToolState()
    add_peak(tool, 0.0)
    before = list(tool.peaks)
    with pytest.raises(ValueError):
        edit_peak(tool, index, column, value)
    assert tool.peaks == before
    with pytest.raises(ValueError):
        remove_peak(tool, 3)


def test_options_are_converted_and_checked():
    tool = FitToolState()
    set_fit_option(tool, "lineshape", "voigt")
    set_fit_option(tool, "max_peaks", 2.0)
    set_fit_option(tool, "n_jobs", -1)
    assert (tool.lineshape, tool.max_peaks) == ("voigt", 2)
    for name, value in [
        ("lineshape", "gauss"),
        ("max_peaks", -1),
        ("n_jobs", 0),
        ("unknown", 1),
        ("min_t2_ms", float("inf")),
    ]:
        with pytest.raises(ValueError):
            set_fit_option(tool, name, value)


def test_settings_for_the_browser_are_plain_values():
    tool = FitToolState()
    add_peak(tool, 12.0)
    settings = fit_tool_settings(tool)
    assert "n_manual" not in settings
    assert settings["peaks"][0]["name"] == "manual_1"
    assert settings["peaks"][0]["freq_hz"] == 12.0
    assert set(settings["peaks"][0]) == {
        "name",
        "freq_hz",
        "freq_range_hz",
        "fixed_freq",
        "initial_t2_ms",
        "min_t2_ms",
        "max_t2_ms",
        "fixed_t2",
        "fit",
        "t2_leader",
        "t2_ratio_low",
        "t2_ratio_high",
        "phase_band",
        "phase_tol_deg",
        "freq_from_fit",
        "t2_from_fit",
        "t2_window_ms",
    }
    assert "couplings" not in settings


@pytest.mark.parametrize("lineshape", ["lorentz", "voigt"])
def test_fit_recovers_invented_lines_in_every_voxel(lineshape):
    data = _fids()
    tool = FitToolState(n_jobs=1, lineshape=lineshape)
    detect_peaks(tool, data)
    fit_params, fit_results = run_fit(tool, data)

    order = np.argsort(fit_results.fit_freqs.isel(read_idx=0, phase_idx=0).squeeze())
    freqs = fit_results.fit_freqs.values.squeeze()[..., order]
    amps = np.abs(fit_results.fit_amps.values.squeeze()[..., order])
    scale = np.arange(1, 4)[:, None, None]
    # Noise of 1 % of the weakest line: 1 Hz and 2 % amplitude agreement.
    np.testing.assert_allclose(
        freqs, np.broadcast_to([-800.0, 300.0], freqs.shape), atol=1.0
    )
    np.testing.assert_allclose(
        amps, scale * np.array([1.0, 0.4]) * np.ones(freqs.shape), rtol=0.02
    )
    assert fit_results.attrs["lineshape"] == lineshape
    assert "Fit done." in tool.message
    assert fit_params.attrs.get("dwell_time_s") == pytest.approx(DWELL_S)


def test_fit_without_lines_raises():
    with pytest.raises(ValueError):
        run_fit(FitToolState(n_jobs=1), _fids())


def test_fit_is_stored_on_the_scan_for_review():
    data = _fids()
    tool = FitToolState(n_jobs=1)
    detect_peaks(tool, data)
    fit_params, fit_results = run_fit(tool, data)

    class Scan:
        pass

    scan = Scan()
    store_fit_on_scan(scan, fit_params, fit_results, data)
    assert scan.fit_results_review is fit_results
    assert scan.interactive_fit_results is fit_results
    assert scan.fit_params_review is fit_params
    assert scan.interactive_fit_params is fit_params
    xr.testing.assert_identical(scan.data_review, data)
    assert scan.data_review is not data


def _two_lines():
    tool = FitToolState(n_jobs=1)
    add_peak(tool, -800.0)
    add_peak(tool, 300.0)
    return tool


def test_coupling_columns_need_another_line_and_positive_numbers():
    tool = _two_lines()
    edit_peak(tool, 1, "t2_leader", "manual_1")
    edit_peak(tool, 1, "t2_ratio_high", 2.0)
    edit_peak(tool, 0, "phase_band", True)
    assert tool.couplings == [
        LineCoupling(phase_band=True),
        LineCoupling(t2_leader="manual_1", t2_ratio_high=2.0),
    ]
    with pytest.raises(ValueError, match="another line"):
        edit_peak(tool, 0, "t2_leader", "manual_1")
    with pytest.raises(ValueError, match="another line"):
        edit_peak(tool, 0, "t2_leader", "lactate")
    with pytest.raises(ValueError, match="must be positive"):
        edit_peak(tool, 0, "phase_tol_deg", 0)


def test_follower_keeps_a_renamed_leader_and_is_freed_on_removal():
    tool = _two_lines()
    edit_peak(tool, 1, "t2_leader", "manual_1")
    edit_peak(tool, 0, "name", "pyruvate")
    assert tool.couplings[1].t2_leader == "pyruvate"
    remove_peak(tool, 0)
    assert tool.couplings == [LineCoupling()]


def test_detection_resets_the_couplings():
    tool = _two_lines()
    edit_peak(tool, 1, "t2_leader", "manual_1")
    detect_peaks(tool, _fids())
    assert tool.couplings == [LineCoupling()] * len(tool.peaks)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("prominence_snr", -1, "must not be negative"),
        ("nsamplepoints", -5, "must not be negative"),
        ("max_peaks_per_spectrum", 0, "at least 1"),
        ("phase_band_tau_ms", float("nan"), "finite"),
    ],
)
def test_advanced_settings_are_checked(name, value, message):
    with pytest.raises(ValueError, match=message):
        set_fit_option(FitToolState(), name, value)


def test_one_peak_per_spectrum_keeps_only_the_strongest_line():
    tool = FitToolState()
    set_fit_option(tool, "max_peaks_per_spectrum", 1)
    detect_peaks(tool, _fids())
    assert len(tool.peaks) == 1
    assert tool.peaks[0].freq_hz == pytest.approx(-800.0, abs=40.0)


def test_fit_resolution_and_fitted_samples_reach_the_fit_options():
    data = _fids()
    tool = FitToolState(n_jobs=1, rep_fitting=1, npoints=5, nsamplepoints=64)
    detect_peaks(tool, data)
    fit_params, _ = run_fit(tool, data)
    assert fit_params.attrs["rep_fitting"] == 1
    assert fit_params.attrs["npoints"] == 5
    assert fit_params.attrs["nsamplepoints"] == 64


def test_coupled_and_phase_banded_fit_recovers_the_invented_lines():
    """Both invented lines share T2* = 30 ms and phase 0 at the first sample."""
    data = _fids()
    tool = FitToolState(n_jobs=1, phase_band_tau_ms=1e-6)
    detect_peaks(tool, data)
    edit_peak(tool, 1, "t2_leader", tool.peaks[0].name)
    edit_peak(tool, 0, "phase_band", True)
    edit_peak(tool, 1, "phase_band", True)
    fit_params, fit_results = run_fit(tool, data)

    first, second = (peak.name for peak in tool.peaks)
    assert fit_params.attrs["t2_coupling"][second][0] == first
    assert set(fit_params.attrs["phase_band_tolerance_deg"]) == {first, second}
    assert fit_params.attrs["phase_band_tau_s"] == pytest.approx(1e-9)
    order = np.argsort(fit_results.fit_freqs.isel(read_idx=0, phase_idx=0).squeeze())
    freqs = fit_results.fit_freqs.values.squeeze()[..., order]
    # Noise of 1 % of the weakest line: 1 Hz agreement, as for the free fit.
    np.testing.assert_allclose(
        freqs, np.broadcast_to([-800.0, 300.0], freqs.shape), atol=1.0
    )


def test_phase_band_without_delay_or_scan_asks_for_tau():
    data = _fids()
    tool = FitToolState(n_jobs=1)
    detect_peaks(tool, data)
    edit_peak(tool, 0, "phase_band", True)
    with pytest.raises(ValueError, match="Enter tau in ms"):
        run_fit(tool, data)


def _shifted_fids(shifts_hz=(-20.0, 0.0, 40.0)):
    """Invented FIDs of 3 x 1 voxels whose lines move by a B0 shift per voxel."""
    time_s = np.arange(128) * DWELL_S
    fids = [
        sum(
            amplitude * np.exp(2j * np.pi * (freq_hz + shift) * time_s - time_s / 0.03)
            for freq_hz, amplitude in LINES
        )
        for shift in shifts_hz
    ]
    values = np.asarray(fids)[:, None, None, None, None, :]
    return xr.DataArray(
        values.astype(np.complex64),
        dims=DIMS,
        coords={"sampling_time_axis_s": ("acquisition_idx", time_s)},
        attrs={"dwell_time_s": DWELL_S, "signal_domain": "temporal"},
    )


def _lines_by_hand():
    """Tool with the two invented lines, each searched within 150 Hz."""
    tool = FitToolState(n_jobs=1, freq_search_range_hz=150.0)
    for freq_hz, _ in LINES:
        add_peak(tool, freq_hz)
    return tool


def test_aligned_spectra_put_the_lines_of_every_voxel_at_one_frequency():
    data = _shifted_fids()
    original = data.copy(deep=True)
    tool = _lines_by_hand()
    fit_params, fit_results = run_fit(tool, data)

    aligned, aligned_params, aligned_results = align_spectra(
        tool, data, fit_params, fit_results
    )
    xr.testing.assert_identical(data, original)
    assert "Spectra aligned to" in tool.message
    # Equal amplitudes in all voxels: the weighted mean shift is 20/3 Hz.
    offsets = aligned.coords["frequency_alignment_offset_hz"].values.squeeze()
    np.testing.assert_allclose(offsets, [20 / 3 + 20, 20 / 3, 20 / 3 - 40], atol=1.0)
    np.testing.assert_allclose(
        aligned_results.fit_freqs.values.squeeze().std(axis=0), 0.0, atol=1.0
    )
    # Start frequencies move with each voxel.
    shift = aligned_params.initial_freqs_hz - fit_params.initial_freqs_hz
    np.testing.assert_allclose(
        shift.transpose("read_idx", ...).values.reshape(3, -1).mean(axis=1),
        offsets,
        atol=1e-6,
    )

    # A new fit of the aligned spectra finds the lines at one frequency.
    _, refit = run_fit(tool, aligned)
    freqs = np.sort(refit.fit_freqs.values.squeeze(), axis=-1)
    expected = np.array([-800.0, 300.0]) + 20 / 3
    np.testing.assert_allclose(freqs, np.broadcast_to(expected, freqs.shape), atol=1.0)


def test_alignment_takes_the_timing_from_the_scan_when_the_data_lack_it():
    data = _shifted_fids()
    tool = _lines_by_hand()
    fit_params, fit_results = run_fit(tool, data)
    bare = data.copy()
    bare.attrs.pop("dwell_time_s")

    class Scan:
        dwell_time_s = DWELL_S

    with_scan, _, _ = align_spectra(tool, bare, fit_params, fit_results, scan=Scan())
    with_attrs, _, _ = align_spectra(tool, data, fit_params, fit_results)
    np.testing.assert_allclose(with_scan.values, with_attrs.values, atol=1e-6)


def test_seeding_columns_are_edited_and_checked():
    tool = _two_lines()
    edit_peak(tool, 0, "freq_from_fit", True)
    edit_peak(tool, 1, "t2_window_ms", "4")
    assert tool.seedings[0] == LineSeeding(freq_from_fit=True)
    assert tool.seedings[1].t2_window_ms == 4.0
    with pytest.raises(ValueError, match="t2_window_ms"):
        edit_peak(tool, 1, "t2_window_ms", -1)
    rows = fit_tool_settings(tool)["peaks"]
    assert rows[0]["freq_from_fit"] is True and rows[1]["t2_window_ms"] == 4.0
    remove_peak(tool, 0)
    assert tool.seedings == [LineSeeding(t2_window_ms=4.0)]


def test_fit_takes_start_values_from_an_earlier_fit_and_skips_rows_switched_off():
    data = _fids()
    tool = _two_lines()
    for row in (0, 1):
        edit_peak(tool, row, "freq_range_hz", 100.0)
    _, first = run_fit(tool, data)
    lac = tool.peaks[1].name
    # Typed start 30 Hz off; the earlier fit gives the start per voxel.
    edit_peak(tool, 1, "freq_hz", 330.0)
    edit_peak(tool, 1, "freq_from_fit", True)
    edit_peak(tool, 1, "t2_from_fit", True)
    edit_peak(tool, 0, "fit", False)
    fit_params, fit_results = run_fit(tool, data, earlier_fit=first)

    np.testing.assert_allclose(
        fit_params.initial_freqs_hz.sel(metabolite=lac).squeeze(),
        first.fit_freqs.sel(metabolite=lac).squeeze(),
    )
    np.testing.assert_allclose(
        fit_params.initial_t2s_s.sel(metabolite=lac).squeeze(),
        first.fit_t2s.sel(metabolite=lac).squeeze(),
    )
    # The -800 Hz line is no longer modelled; its tail moves the 300 Hz line
    # by about 1 Hz in the weakest voxel, so 2 Hz agreement.
    np.testing.assert_allclose(
        fit_results.fit_freqs.sel(metabolite=lac).squeeze(), 300.0, atol=2.0
    )
    assert f"{lac} (freq and T2*)" in tool.message
    assert tool.peaks[0].name not in tool.message.split("for:")[1]


def test_stored_fit_is_read_in_order_and_skips_other_fits():
    data = _fids()
    tool = FitToolState(n_jobs=1)
    detect_peaks(tool, data)
    fit_params, fit_results = run_fit(tool, data)

    class Scan:
        fit_results = {"t1": "map fits of another kind"}

    scan = Scan()
    assert stored_fit(None) is None
    assert stored_fit(scan) is None
    scan.fit_results_review = fit_results
    assert stored_fit(scan) == ("fit_results_review", fit_results, None)
    store_fit_on_scan(scan, fit_params, fit_results, data)
    name, results, params = stored_fit(scan)
    assert name == "interactive_fit_results"
    assert results is fit_results and params is fit_params


def test_lines_from_fit_fill_the_table_with_seeded_lines():
    data = _fids()
    tool = _two_lines()
    for row in (0, 1):
        edit_peak(tool, row, "freq_range_hz", 100.0)
    edit_peak(tool, 1, "t2_leader", tool.peaks[0].name)
    _, fit_results = run_fit(tool, data)

    fresh = FitToolState(n_jobs=1, snr_threshold=5.0)
    lines_from_fit(fresh, fit_results)
    assert [peak.name for peak in fresh.peaks] == list(fit_results.metabolite.values)
    np.testing.assert_allclose(
        sorted(peak.freq_hz for peak in fresh.peaks), [-800.0, 300.0], atol=1.0
    )
    assert all(
        peak.min_t2_ms <= peak.initial_t2_ms <= peak.max_t2_ms for peak in fresh.peaks
    )
    assert fresh.seedings == [LineSeeding(freq_from_fit=True, t2_from_fit=True)] * 2
    assert fresh.couplings == [LineCoupling()] * 2
    assert fresh.snr_threshold == 0.0

    with pytest.raises(ValueError, match="fit_freqs"):
        lines_from_fit(fresh, fit_results.drop_vars("fit_t2s"))
    assert len(fresh.peaks) == 2


def _broad_fit():
    data = _fids()
    tool = _two_lines()
    for row in (0, 1):
        edit_peak(tool, row, "freq_range_hz", 100.0)
    fit_params, fit_results = run_fit(tool, data)
    return data, tool, fit_params, fit_results


def test_refit_changes_only_the_selected_voxel():
    data, tool, fit_params, fit_results = _broad_fit()
    spoiled = fit_results.copy(deep=True)
    spoiled["fit_freqs"][dict(read_idx=2, phase_idx=1)] = 0.0
    indices = {"read_idx": 2, "phase_idx": 1}

    updated = refit_voxel(tool, data, fit_params, spoiled, indices)

    # Same invented spectrum as the full fit: 1 Hz agreement as above.
    np.testing.assert_allclose(
        updated.fit_freqs.isel(read_idx=2, phase_idx=1).squeeze(),
        fit_results.fit_freqs.isel(read_idx=2, phase_idx=1).squeeze(),
        atol=1.0,
    )
    others = dict(read_idx=slice(0, 2))
    xr.testing.assert_identical(
        updated.fit_freqs.isel(others), spoiled.fit_freqs.isel(others)
    )
    assert (spoiled.fit_freqs.isel(read_idx=2, phase_idx=1) == 0.0).all()
    assert tool.message.startswith("Refitted voxel")


def test_refit_asks_for_a_full_fit_after_renaming_a_line():
    data, tool, fit_params, fit_results = _broad_fit()
    edit_peak(tool, 0, "name", "pyr")
    with pytest.raises(ValueError, match="fit all voxels"):
        refit_voxel(tool, data, fit_params, fit_results, {"read_idx": 0})


def test_problem_voxels_are_visited_in_turn_and_wrap_around():
    _, tool, _, fit_results = _broad_fit()
    indices = {"read_idx": 0, "phase_idx": 0}
    visited = []
    for _ in range(7):
        indices = next_problem_voxel(tool, fit_results, indices)
        visited.append((indices["read_idx"], indices["phase_idx"]))
    assert sorted(set(visited)) == [(r, p) for r in range(3) for p in range(2)]
    assert visited[6] == visited[0]
    assert tool.message.startswith("Problem voxel")

    unfitted = fit_results.copy(deep=True)
    unfitted["fit_rss"][...] = np.nan
    unfitted["fit_freqs"][...] = np.nan
    with pytest.raises(ValueError):
        next_problem_voxel(tool, unfitted, indices)


def test_line_from_residual_adds_the_missing_line_only_when_it_is_there():
    # The reconstruction carries the acquisition-tail noise; here it is the
    # known noise level of the invented FIDs.
    sigma = xr.DataArray(
        np.full((3, 2), 0.01), dims=DIMS[:2], attrs={"noise_source": "acquisition_tail"}
    )
    data = _fids().assign_coords(noise_sigma_quadrature=sigma)
    tool = FitToolState(n_jobs=1)
    add_peak(tool, -800.0)
    edit_peak(tool, 0, "freq_range_hz", 100.0)
    fit_params, _ = run_fit(tool, data)
    indices = {"read_idx": 2, "phase_idx": 0}

    assert add_residual_line(tool, data, fit_params, indices, (200.0, 400.0))
    assert [peak.name for peak in tool.peaks] == ["manual_1", "residual_2"]
    added = tool.peaks[1]
    assert added.freq_hz == pytest.approx(300.0, abs=frequency_step_hz(data))
    assert added.min_t2_ms <= added.initial_t2_ms <= added.max_t2_ms
    assert len(tool.couplings) == len(tool.seedings) == 2

    tool.peaks.pop()
    tool.couplings.pop()
    tool.seedings.pop()
    n_manual = tool.n_manual
    assert not add_residual_line(tool, data, fit_params, indices, (1500.0, 2000.0))
    assert len(tool.peaks) == 1 and tool.n_manual == n_manual
    assert tool.message.startswith("No line")

    with pytest.raises(ValueError, match="search range"):
        add_residual_line(tool, data, fit_params, indices, (5.0, 5.0))


def _invented_results(voxel_dims=("read_idx", "phase_idx")):
    """Invented fit results of 2 x 3 voxels with two lines and bound statuses."""
    lines = ["pyruvate", "lactate"]
    shape = (2, 3, 2)
    dims = (*voxel_dims, "metabolite")
    amps = np.zeros(shape, dtype=complex)
    amps[1, 2] = [2.0 * np.exp(1j * np.pi / 2), np.nan]
    freqs = np.zeros(shape)
    freqs[1, 2] = [-800.0, 300.0]
    t2s = np.full(shape, 0.03)
    freq_status = np.full(shape, "estimated", dtype="<U14")
    t2_status = np.full(shape, "estimated", dtype="<U14")
    t2_status[1, 2] = ["at_lower_bound", "at_upper_bound"]
    return xr.Dataset(
        {
            "fit_amps": (dims, amps),
            "fit_freqs": (dims, freqs),
            "fit_freqs_ppm": (dims, freqs / 30.0),
            "fit_t2s": (dims, t2s),
            "fit_freq_status": (dims, freq_status),
            "fit_t2_status": (dims, t2_status),
            "fit_gauss_fwhm_hz": (voxel_dims, np.full(shape[:2], 12.5)),
            "fit_gauss_fwhm_status": (
                voxel_dims,
                np.full(shape[:2], "at_upper_bound", dtype="<U14"),
            ),
        },
        coords={"metabolite": lines},
    )


def test_voxel_fit_table_lists_the_lines_of_the_selected_voxel():
    table = voxel_fit_table(_invented_results(), {"read_idx": 1, "phase_idx": 2})

    first, second = table["rows"]
    assert first["line"] == "pyruvate"
    assert first["amplitude"] == pytest.approx(2.0)
    assert first["phase_deg"] == pytest.approx(90.0)
    assert first["freq_hz"] == pytest.approx(-800.0)
    assert first["freq_ppm"] == pytest.approx(-800.0 / 30.0)
    assert first["t2_ms"] == pytest.approx(30.0)
    assert (first["freq_status"], first["t2_status"]) == ("estimated", "at_lower_bound")
    # A value that is not finite is None, so the table can be sent as JSON.
    assert second["amplitude"] is None and second["phase_deg"] is None
    assert second["t2_status"] == "at_upper_bound"
    assert table["gauss_fwhm_hz"] == pytest.approx(12.5)
    assert table["gauss_status"] == "at_upper_bound"


def test_voxel_fit_table_clamps_indices_and_skips_missing_variables():
    results = _invented_results().drop_vars(
        ["fit_freqs_ppm", "fit_t2_status", "fit_gauss_fwhm_hz", "fit_gauss_fwhm_status"]
    )
    table = voxel_fit_table(results, {"read_idx": 9, "phase_idx": 7, "slice_idx": 4})

    assert table["rows"][0]["freq_hz"] == pytest.approx(-800.0)
    assert table["rows"][0]["freq_ppm"] is None
    assert table["rows"][0]["t2_status"] == ""
    assert table["gauss_fwhm_hz"] is None and table["gauss_status"] == ""


def test_mask_fit_table_gives_weighted_mean_and_spread_per_mask_and_line():
    results = _invented_results()
    results["fit_freqs"][0, :, 0] = [-790.0, -800.0, -810.0]
    dims = ("read_idx", "phase_idx")
    first_row = xr.DataArray(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]), dims=dims)
    corner = xr.DataArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), dims=dims)

    table = mask_fit_table(results, {"top": first_row, "corner": corner}, {})

    assert table["message"] == ""
    assert [(row["mask"], row["line"]) for row in table["rows"]] == [
        ("top", "pyruvate"),
        ("top", "lactate"),
        ("corner", "pyruvate"),
        ("corner", "lactate"),
    ]
    top = table["rows"][0]
    assert top["n_voxels"] == 3
    np.testing.assert_allclose(top["freq_hz"], [-800.0, np.sqrt(200.0 / 3.0)])
    np.testing.assert_allclose(top["t2_ms"], [30.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(top["gauss_fwhm_hz"], [12.5, 0.0])
    corner_lactate = table["rows"][3]
    # The lactate amplitude of the corner voxel is a failed fit (NaN).
    assert corner_lactate["amplitude"] == [None, None]
    assert corner_lactate["freq_hz"] == [300.0, 0.0]


def test_mask_fit_table_explains_a_mask_on_another_grid():
    wrong = xr.DataArray(np.ones((4, 3)), dims=("read_idx", "phase_idx"))
    table = mask_fit_table(_invented_results(), {"roi": wrong}, {})
    assert table["rows"] == []
    assert "mask 'roi'" in table["message"]
    assert mask_fit_table(_invented_results(), {}, {}) == {"rows": [], "message": ""}
