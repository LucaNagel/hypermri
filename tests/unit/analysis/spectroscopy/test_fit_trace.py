"""Trace of one FID fit: timeline, grids, model states and stored comparison."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy
from hypermri.analysis.spectroscopy import fit_spectrum_trace


def voxels(*, domain="temporal"):
    """Two noisy voxels over ``x_idx`` with three lines, the middle one skipped."""
    time = np.arange(32) * 0.001
    frequencies = np.array([-35.0, 90.0, 210.0])
    rng = np.random.default_rng(4711)
    fids = []
    for scale in (1.0, 0.7):
        amplitudes = scale * np.array([1.0 + 0.4j, 0.0, 0.3 - 0.1j])
        basis = np.exp(2j * np.pi * time[:, None] * frequencies - time[:, None] / 0.025)
        noise = 0.001 * (rng.normal(size=32) + 1j * rng.normal(size=32))
        fids.append(basis @ amplitudes + noise)
    fids = np.stack(fids, axis=1)
    values = (
        fids
        if domain == "temporal"
        else np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)
    )
    data = xr.DataArray(
        values,
        dims=("acquisition_idx", "x_idx"),
        coords={"acquisition_idx": time},
        attrs={"signal_domain": domain},
    )
    params = spectroscopy.def_fit_params(
        data,
        {
            "metabs": ["a", "masked", "b"],
            "metabs_freqs_hz": frequencies + 2,
            "initial_t2s_s": [0.025] * 3,
            "skip_fitting": [False, True, False],
            "metabs_freq_ranges": [30, 0, 40],
            "metabs_t2_ranges": [0.06, 0, 0.07],
            "min_t2_s": [0.015] * 3,
            "max_t2_s": [0.035] * 3,
            "npoints": 5,
            "niter": 2,
            "rep_fitting": 2,
            "phase_band_tolerance_deg": {"a": 20.0, "b": 20.0},
            "phase_band_tau_s": 0.0005,
        },
    )
    return data, params, fids


@pytest.mark.parametrize("domain", ["temporal", "spectral"])
def test_trace_ends_in_the_full_fit_result(domain):
    data, params, fids = voxels(domain=domain)
    before = (data.copy(deep=True), params.copy(deep=True))
    full = spectroscopy.fit_data_pseudo_inv(data, params)
    trace = fit_spectrum_trace(
        data, fit_params=params, selection={"x_idx": 1}, fit_results=full
    )

    xr.testing.assert_identical(data, before[0])
    xr.testing.assert_identical(params, before[1])
    assert (
        "store_optimization_history" not in params.attrs
        or not params.attrs["store_optimization_history"]
    )
    assert trace["selection"] == {"x_idx": 1}
    np.testing.assert_allclose(
        trace["measured"]["fid"], fids[:, 1], rtol=1e-12, atol=1e-12
    )

    last = trace["entries"][-1]
    assert trace["entries"][0]["kind"] == "initial"
    assert last["kind"] == "returned"
    voxel = full.isel(x_idx=1)
    np.testing.assert_array_equal(trace["result"]["frequency_hz"], voxel.fit_freqs)
    np.testing.assert_array_equal(trace["result"]["amplitude"], voxel.fit_amps)
    assert trace["stored_comparison"]["matches"]

    # The model of the returned state reproduces the reported temporal RSS.
    residual = trace["measured"]["fid"] - trace["model_fid"][-1]
    np.testing.assert_allclose(
        np.sum(np.abs(residual) ** 2), trace["result"]["rss"], rtol=1e-9
    )
    np.testing.assert_allclose(
        trace["model_spectrum"][-1], np.fft.fftshift(np.fft.fft(trace["model_fid"][-1]))
    )


def test_grids_stay_inside_bounds_and_mark_their_choice():
    data, params, _ = voxels()
    trace = fit_spectrum_trace(data, fit_params=params, selection={"x_idx": 0})
    lines = trace["lines"]
    assert lines[1]["skipped"] and not lines[0]["skipped"]
    assert lines[0]["frequency_lower_hz"] == pytest.approx(-33.0 - 15.0)
    assert lines[2]["frequency_upper_hz"] == pytest.approx(212.0 + 20.0)
    assert {grid["parameter"] for grid in trace["grids"]} == {"frequency", "t2"}
    for grid in trace["grids"]:
        line = lines[grid["line"]]
        assert grid["center_index"] >= 0
        assert grid["accepted_index"] < len(grid["values"])
        evaluated = grid["values"][grid["evaluated"]]
        if grid["parameter"] == "frequency":
            assert np.all(evaluated >= line["frequency_lower_hz"] - 1e-9)
            assert np.all(evaluated <= line["frequency_upper_hz"] + 1e-9)
        else:
            assert np.all(evaluated >= line["t2_min_s"] - 1e-12)
            assert np.all(evaluated <= line["t2_max_s"] + 1e-12)
        rejected = ~grid["evaluated"]
        assert all(
            grid["rejection_reason"][i] != "none" for i in np.flatnonzero(rejected)
        )
    grid_entries = [entry for entry in trace["entries"] if entry["kind"] == "grid"]
    assert grid_entries and all(entry["line"] in (0, 2) for entry in grid_entries)


def test_phase_law_and_reference_follow_the_band():
    data, params, _ = voxels()
    trace = fit_spectrum_trace(data, fit_params=params, selection={})
    assert trace["selection"] == {"x_idx": 0}
    assert trace["lines"][0]["phase_tolerance_deg"] == 20.0
    assert trace["lines"][1]["phase_tolerance_deg"] is None
    entry = trace["entries"][-1]
    tau = trace["options"]["phase_band_tau_s"]
    expected = np.angle(
        entry["amplitude"] * np.exp(-2j * np.pi * entry["frequency_hz"] * tau)
    )
    np.testing.assert_allclose(entry["phase_law_rad"][[0, 2]], expected[[0, 2]])
    assert np.isfinite(entry["phase_reference_rad"])
    # The solver keeps every band line within its tolerance of one phi0, so the
    # display reference finds such a phase for each evaluated state.
    for state in trace["entries"]:
        lines = [i for i in (0, 2) if state["active"][i]]
        deviation = np.angle(
            np.exp(1j * (state["phase_law_rad"][lines] - state["phase_reference_rad"]))
        )
        assert np.all(np.abs(np.rad2deg(deviation)) <= 20.0 + 1e-6)


def test_phase_reference_centres_lines_at_twice_the_tolerance():
    # Two band lines exactly 40 degrees apart (plus a rounding-sized excess)
    # with 20 degree tolerances, unequal weights pulling the circular mean
    # off-centre: the reference must sit in the middle, as the solver
    # placed phi0 at the shared boundary.
    from hypermri.analysis.spectroscopy.fit_trace import _phase_law

    phases = np.deg2rad([8.247, -31.753 - 1e-9])
    amplitudes = np.array([[3.0 * np.exp(1j * phases[0]), np.exp(1j * phases[1])]])
    law, reference = _phase_law(
        amplitudes, np.zeros((1, 2)), 0.0, np.array([20.0, 20.0])
    )
    deviation = np.rad2deg(np.angle(np.exp(1j * (law[0] - reference[0]))))
    np.testing.assert_allclose(np.abs(deviation), 20.0, atol=1e-6)


def test_stored_comparison_flags_a_changed_voxel():
    data, params, _ = voxels()
    full = spectroscopy.fit_data_pseudo_inv(data, params)
    edited = full.copy(deep=True)
    edited["fit_freqs"][dict(x_idx=0, metabolite=0)] += 1.5
    comparison = fit_spectrum_trace(
        data, fit_params=params, selection={"x_idx": 0}, fit_results=edited
    )["stored_comparison"]
    assert not comparison["matches"]
    assert comparison["frequency_difference_hz"] == pytest.approx(1.5)


def test_selection_and_lineshape_errors():
    data, params, _ = voxels()
    with pytest.raises(ValueError, match="outside"):
        fit_spectrum_trace(data, fit_params=params, selection={"x_idx": 2})
    with pytest.raises(ValueError, match="spectral dimension"):
        fit_spectrum_trace(
            data.rename(acquisition_idx="t"), fit_params=params, selection={}
        )
    stacked = params.expand_dims(z_idx=2)
    with pytest.raises(ValueError, match="z_idx"):
        fit_spectrum_trace(data, fit_params=stacked, selection={"x_idx": 0})
    voigt = params.copy()
    voigt.attrs["lineshape"] = "voigt"
    with pytest.raises(ValueError, match="voigt"):
        fit_spectrum_trace(data, fit_params=voigt, selection={"x_idx": 0})
