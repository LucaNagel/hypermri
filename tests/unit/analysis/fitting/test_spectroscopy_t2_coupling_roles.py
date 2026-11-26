"""Regression tests for T2* coupling roles, band leaders and line windows.

``t2_coupling`` names a leader and a dependent by metabolite name. Names, T2*
values and coupling roles belong to frequency positions: the k-th lowest line
of a coupling group carries the name with the k-th lowest mean start
frequency. Each line keeps its own frequency search width and bounds,
which travel with it when it passes another line of its group. A band
leader is also searched on its own after its joint move, so a weak leader no
longer has to follow a strong banded dependent.

Fixture provenance: two noise-free voxels of 256 samples at 4 kHz. Voxel 0
holds a strong line S (area 20, phase -0.2 rad, 0 Hz, T2* 30 ms) and a weak
line W (area 2, phase 0.5 rad, +8 Hz, T2* 25.5 ms). Voxel 1 is voxel 0
shifted by +8 Hz, so S sits at 8 Hz and W at 16 Hz. The fit starts at 0 and
8 Hz with 40 Hz ranges and 30 ms T2*. In voxel 1 the dependent name starts on
S, and the leader name, searched first, moves to W; previously the roles
stayed exchanged, the band stayed at its start ratio and T2*(S) was off by
-0.52 ms and T2*(W) by +3.5 ms. With the fixed ratio 0.85 (``FIXED_COUPLING``)
the fit of voxel 1 ended at 2.99 and 8.0 Hz with W unfitted, because every
frequency candidate that moved the leader name past S was scored with the
strong line's T2* scaled by 0.85.

With W as band leader (``WEAK_LEADER_COUPLING``) and correct frequencies,
T2*(W) stayed at its 30 ms start previously, because every leader candidate
also moved S.

Nested windows (``NESTED_RANGES_HZ``, 40 and 10 Hz): in voxel 1 the line
started at 0 Hz must pass the line started at 8 Hz, whose window is 3-13 Hz.
In an intermediate version every such configuration raised ``ValueError`` before
the first frequency step, because the windows stayed with the names.

Tolerances: without noise, correctly assigned lines recover T2* to 0.1 ms
(grid resolution after the zoomed passes) and frequencies to 0.1 Hz. The
Voigt fit of this fixture is not accurate (shared Gaussian FWHM about 5.2 Hz
against 6 Hz), so the Voigt test only checks that both voxels agree to
0.01 Hz and 0.01 ms; previously they differed by up to 7.7 Hz and 8.8 ms.
Fits whose two voxels search different windows relative to their lines
(nested windows with a crossing in voxel 1 only) agree to 0.1 Hz, 0.2 ms and
0.2 degrees; the reported add-a-line workflow uses the coarse default grid
and recovers W only to 1 Hz.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy import fid_model
from hypermri.analysis.fitting.models.spectroscopy.diagnostics import HISTORY_EVENTS
from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _coupled_frequency_positions,
    _coupled_line_order,
    _t2_move_constraints,
    _t2_search_moves,
    _update_frequency_bounds,
)
from hypermri.analysis.fitting.spectroscopy import (
    add_metab_to_fit_params,
    def_fit_params,
    fit_data_pseudo_inv,
)

STRONG_T2_S = 0.030
WEAK_T2_S = 0.0255
SHIFT_HZ = 8.0
BAND_COUPLING = {"OMPD1_2": ("OMPD1_1", (0.8, 1.25))}
FIXED_COUPLING = {"OMPD1_2": ("OMPD1_1", WEAK_T2_S / STRONG_T2_S)}
WEAK_LEADER_COUPLING = {"OMPD1_1": ("OMPD1_2", (0.8, 1.25))}
NESTED_RANGES_HZ = [40.0, 10.0]
EXPECTED_FREQS_HZ = [[0.0, SHIFT_HZ], [SHIFT_HZ, 2 * SHIFT_HZ]]
EXPECTED_T2S_S = [[STRONG_T2_S, WEAK_T2_S], [STRONG_T2_S, WEAK_T2_S]]
VOIGT_OPTIONS = {
    "lineshape": "voigt",
    "gauss_initial_fwhm_hz": 2.0,
    "gauss_max_fwhm_hz": 12.0,
}


def two_voxel_doublet(gauss_fwhm_hz=0.0):
    """Return the noise-free strong/weak doublet and its +8 Hz shifted copy.

    Parameters
    ----------
    gauss_fwhm_hz : float
        FWHM in Hz of a Gaussian decay shared by both lines; 0 gives
        Lorentz lines.

    Returns
    -------
    xarray.DataArray
        Complex FIDs with dimensions ``("x", "acquisition_idx")``, shape
        ``(2, 256)``, and ``attrs["sampling_time_axis_s"]`` in seconds.
    """
    t = np.arange(256) / 4000.0
    strong = 20.0 * np.exp(-0.2j) * np.exp(-t / STRONG_T2_S)
    weak = 2.0 * np.exp(0.5j) * np.exp(2j * np.pi * SHIFT_HZ * t - t / WEAK_T2_S)
    gauss = np.exp(-((np.pi * gauss_fwhm_hz * t) ** 2) / (4 * np.log(2)))
    fid = (strong + weak) * gauss
    data = xr.DataArray(
        np.stack([fid, fid * np.exp(2j * np.pi * SHIFT_HZ * t)]),
        dims=["x", "acquisition_idx"],
        coords={"x": [0, 1]},
    )
    data.attrs["sampling_time_axis_s"] = t
    return data


def doublet_params(data, **options):
    """Return fit parameters with the fixture's starts, ranges and T2* bounds."""
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": ["OMPD1_1", "OMPD1_2"],
            "metabs_freqs_hz": [0.0, SHIFT_HZ],
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 40.0,
            "min_t2_s": 0.005,
            "max_t2_s": 0.2,
            **options,
        },
    )


@pytest.fixture(scope="module")
def band_fit():
    """Fit both voxels with the band coupling, clipped starts and history."""
    data = two_voxel_doublet()
    params = doublet_params(
        data,
        t2_coupling=BAND_COUPLING,
        t2_coupling_clip_start=True,
        store_optimization_history=True,
    )
    return fit_data_pseudo_inv(data, params)


def test_coupled_lines_keep_start_frequency_order(band_fit):
    """The leader name stays on the lower, strong line in both voxels."""
    freqs = band_fit.fit_freqs
    amps = np.abs(band_fit.fit_amps)

    np.testing.assert_allclose(freqs.sel(x=0).values, [0.0, SHIFT_HZ], atol=0.1)
    np.testing.assert_allclose(
        freqs.sel(x=1).values, [SHIFT_HZ, 2 * SHIFT_HZ], atol=0.1
    )
    for x in (0, 1):
        assert amps.sel(x=x, metabolite="OMPD1_1") > 5 * amps.sel(
            x=x, metabolite="OMPD1_2"
        )


def test_band_recovers_t2_in_shifted_voxel(band_fit):
    """Both voxels recover T2*(S) and T2*(W) within the band."""
    np.testing.assert_allclose(
        band_fit.fit_t2s.values,
        [[STRONG_T2_S, WEAK_T2_S], [STRONG_T2_S, WEAK_T2_S]],
        atol=1e-4,
    )


def test_frequency_candidates_keep_coupled_lines_in_start_order(band_fit):
    """Every scored frequency candidate has the leader name below W.

    In voxel 1 the leader name is searched past S; its candidates are scored
    with the two frequencies given back in start order. Candidates rejected
    for coinciding lines are not scored and are left out.
    """
    info = band_fit.fit_history_info
    freqs = band_fit.fit_history_freqs
    for x in (0, 1):
        scored = (
            (
                info.sel(x=x, history_field="event")
                == HISTORY_EVENTS["frequency_candidate"]
            )
            & (info.sel(x=x, history_field="evaluated") == 1)
        ).values
        rows = freqs.sel(x=x).values[scored]

        assert len(rows) > 0
        assert np.all(rows[:, 0] < rows[:, 1])


def test_fixed_ratio_recovers_shifted_voxel():
    """A fixed T2* ratio no longer stays in the exchanged-role minimum."""
    data = two_voxel_doublet()
    params = doublet_params(data, t2_coupling=FIXED_COUPLING)

    result = fit_data_pseudo_inv(data, params)

    np.testing.assert_allclose(
        result.fit_freqs.values,
        [[0.0, SHIFT_HZ], [SHIFT_HZ, 2 * SHIFT_HZ]],
        atol=0.1,
    )
    np.testing.assert_allclose(
        result.fit_t2s.values,
        [[STRONG_T2_S, WEAK_T2_S], [STRONG_T2_S, WEAK_T2_S]],
        atol=1e-4,
    )


def test_coupled_line_order_gives_positions_by_frequency():
    """The k-th lowest line of a group moves to its k-th position slot.

    Lines outside every group stay in place, ties keep slot order and the
    input array is not changed.
    """
    positions = [np.array([1, 2, 3, 0])]
    frequencies_hz = np.array([1.0, 12.0, 6.0, 6.0, 99.0])

    order = _coupled_line_order(frequencies_hz, positions)

    np.testing.assert_array_equal(order, [1, 0, 2, 3, 4])
    np.testing.assert_array_equal(frequencies_hz[order], [12.0, 1.0, 6.0, 6.0, 99.0])
    np.testing.assert_array_equal(frequencies_hz, [1.0, 12.0, 6.0, 6.0, 99.0])


def test_coupled_frequency_positions_follow_start_rank_and_names():
    """Positions follow the supplied rank, else this call's starts and names."""
    params = {
        "metabs": ["pyr", "lac", "ala", "bic"],
        "metabs_freqs_hz": np.array([5.0, 0.0, 5.0, 3.0]),
        "t2_coupling": {"lac": ("pyr", 0.7), "ala": ("pyr", (0.6, 1.2))},
    }

    (fallback,) = _coupled_frequency_positions(params)
    (supplied,) = _coupled_frequency_positions(
        {**params, "_frequency_position_rank": np.array([3, 2, 0, 1])}
    )

    np.testing.assert_array_equal(fallback, [1, 2, 0])
    np.testing.assert_array_equal(supplied, [2, 1, 0])


@pytest.mark.parametrize(
    "coupling",
    [BAND_COUPLING, FIXED_COUPLING, {"OMPD1_2": ("OMPD1_1", (0.7, 0.95))}],
    ids=["band", "fixed", "asymmetric-band"],
)
def test_nested_windows_travel_with_their_lines(coupling):
    """A wide window may pass a narrow one; both voxels recover S and W.

    The test pins ``search_order="metabolite"``, so the line started at
    0 Hz moves first and passes the narrow window. With the ``"amplitude"``
    default the strong line at 8 Hz moves first and voxel 1 ends 0.1-1.2 Hz
    and up to 2.6 ms off W after 11 repetitions (residual sum of squares
    0.002-0.02 instead of 0); doubling ``rep_fitting`` roughly halves it.
    """
    data = two_voxel_doublet()
    params = doublet_params(
        data,
        metabs_freq_ranges=NESTED_RANGES_HZ,
        t2_coupling=coupling,
        t2_coupling_clip_start=True,
        search_order="metabolite",
    )

    result = fit_data_pseudo_inv(data, params)

    assert list(result.fit_status.values) == ["success", "success"]
    np.testing.assert_allclose(result.fit_freqs.values, EXPECTED_FREQS_HZ, atol=0.1)
    np.testing.assert_allclose(result.fit_t2s.values, EXPECTED_T2S_S, atol=1e-4)


@pytest.mark.parametrize("lineshape", [{}, VOIGT_OPTIONS], ids=["lorentz", "voigt"])
def test_each_line_is_searched_in_its_own_window(monkeypatch, lineshape):
    """Every frequency step keeps each line inside the window it started with.

    The voxel-1 fit exchanges the lines once; afterwards the 3-13 Hz window
    belongs to the leader name, which now holds the line started at 8 Hz.
    """
    steps = []
    grid_search = fid_model._fit_parameter_grid

    def recording_grid_search(data, options, *, parameter):
        result = grid_search(data, options, parameter=parameter)
        if parameter == "frequency":
            steps.append(
                (
                    np.real(options["freq_range_lower_bound_hz"]).copy(),
                    np.real(options["freq_range_upper_bound_hz"]).copy(),
                    result[1].copy(),
                    result[4].copy(),
                )
            )
        return result

    monkeypatch.setattr(fid_model, "_fit_parameter_grid", recording_grid_search)
    data = two_voxel_doublet(gauss_fwhm_hz=6.0 if lineshape else 0.0).isel(x=[1])
    params = doublet_params(
        data,
        metabs_freq_ranges=NESTED_RANGES_HZ,
        t2_coupling=BAND_COUPLING,
        t2_coupling_clip_start=True,
        **lineshape,
    )

    fit_data_pseudo_inv(data, params)

    np.testing.assert_array_equal(steps[0][0], [-20.0, 3.0])
    np.testing.assert_array_equal(steps[0][3], [1, 0])
    np.testing.assert_array_equal(steps[1][0], [3.0, -20.0])
    for lower, upper, frequencies, order in steps:
        assert np.all(lower[order] <= frequencies)
        assert np.all(frequencies <= upper[order])


def test_reported_workflow_adds_a_narrow_line_inside_a_wide_one():
    """Fit S alone in an 80 Hz window, then add W with a 50 Hz window.

    In an intermediate version this raised ``ValueError``: with these windows the
    starts would have had to lie 15 Hz apart.
    """
    data = two_voxel_doublet()
    first = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["OMPD1_1"],
            "metabs_freqs_hz": [0.0],
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": [80.0],
            "min_t2_s": 0.005,
            "max_t2_s": 0.2,
        },
    )
    strong_hz = fit_data_pseudo_inv(data, first).fit_freqs.sel(
        metabolite="OMPD1_1", drop=True
    )
    first = first.copy(deep=True)
    first["initial_freqs_hz"].loc[{"metabolite": "OMPD1_1"}] = strong_hz
    params = add_metab_to_fit_params(
        new_freq=strong_hz + SHIFT_HZ,
        new_metab_name="OMPD1_2",
        fit_params=first,
        new_t2_s=0.03,
        new_min_t2_s=0.005,
        new_max_t2_s=0.2,
        new_freq_range=50.0,
        new_freq_unit="hz",
        freeze_other_freqs=False,
        freeze_other_t2s=False,
    ).assign_attrs(t2_coupling=BAND_COUPLING, t2_coupling_clip_start=True)

    result = fit_data_pseudo_inv(data, params)

    assert list(result.fit_status.values) == ["success", "success"]
    freqs = result.fit_freqs.values
    np.testing.assert_allclose(freqs[:, 0], [0.0, SHIFT_HZ], atol=0.1)
    np.testing.assert_allclose(freqs[:, 1], [SHIFT_HZ, 2 * SHIFT_HZ], atol=1.0)
    np.testing.assert_allclose(freqs[1], freqs[0] + SHIFT_HZ, atol=0.01)


def test_frozen_line_keeps_its_frequency_when_passed():
    """A frozen line stays put; the passing line takes the upper name.

    W is added 6 Hz below the frozen S line (window width 0) and must pass it
    to reach its peak. The frozen frequency then belongs to the lower name,
    OMPD1_2, in both voxels, and both voxels reach W. The starts and ranges
    are changed in place, so the frequency bounds are recomputed; with the
    bounds of ``doublet_params`` voxel 1 stopped at 14 Hz
    (``test_spectroscopy_doublet_search.py``).
    """
    data = two_voxel_doublet()
    params = doublet_params(data, metabs_freqs_hz=[0.0, -6.0]).copy(deep=True)
    params["metabs_freq_ranges"].loc[{"metabolite": "OMPD1_1"}] = 0.0
    params["metabs_freq_ranges"].loc[{"metabolite": "OMPD1_2"}] = 50.0
    params["initial_freqs_hz"].loc[{"metabolite": "OMPD1_1"}] = [0.0, SHIFT_HZ]
    params["initial_freqs_hz"].loc[{"metabolite": "OMPD1_2"}] = [-6.0, SHIFT_HZ - 6]
    params = _update_frequency_bounds(
        params.assign_attrs(t2_coupling=BAND_COUPLING, t2_coupling_clip_start=True)
    )

    result = fit_data_pseudo_inv(data, params)

    frozen = result.fit_freqs.sel(metabolite="OMPD1_2").values
    passing = result.fit_freqs.sel(metabolite="OMPD1_1").values
    np.testing.assert_array_equal(frozen, [0.0, SHIFT_HZ])
    np.testing.assert_allclose(passing, [SHIFT_HZ, 2 * SHIFT_HZ], atol=0.1)
    amps = np.abs(result.fit_amps)
    assert np.all(amps.sel(metabolite="OMPD1_2") > 5 * amps.sel(metabolite="OMPD1_1"))


@pytest.mark.parametrize("execution", ["serial", "threads", "dask"])
def test_positions_follow_mean_start_order_in_every_execution_path(execution):
    """A spectrum whose starts contradict the mean order gets the mean roles.

    Both voxels hold the unshifted doublet. In voxel 1 OMPD1_1 starts at
    14 Hz above OMPD1_2 at 8 Hz, but its mean start (7 Hz) is lower, so it
    takes the lower line S in both voxels. Batches and chunks must use the
    order of all spectra, not of their own subset.
    """
    data = two_voxel_doublet().isel(x=[0, 0]).assign_coords(x=[0, 1])
    params = doublet_params(
        data,
        metabs_freq_ranges=[40.0, 20.0],
        t2_coupling=BAND_COUPLING,
        t2_coupling_clip_start=True,
    ).copy(deep=True)
    params["initial_freqs_hz"].loc[{"x": 1, "metabolite": "OMPD1_1"}] = 14.0
    params = _update_frequency_bounds(params)
    options = {}
    if execution == "threads":
        options = {"n_jobs": 2, "batch_size": 1, "parallel_backend": "threading"}
    elif execution == "dask":
        pytest.importorskip("dask")
        data = data.chunk({"x": 1})

    result = fit_data_pseudo_inv(data, params, **options).compute()

    np.testing.assert_allclose(
        result.fit_freqs.values, [[0.0, SHIFT_HZ], [0.0, SHIFT_HZ]], atol=0.1
    )
    np.testing.assert_allclose(
        result.fit_t2s.values, [[STRONG_T2_S, WEAK_T2_S]] * 2, atol=1e-4
    )


def test_amplitude_refit_reuses_the_window_of_the_kept_line(monkeypatch):
    """After the exchange, the refit searches S in the window it started with.

    The SNR threshold removes W, which ends under OMPD1_2 with the 40 Hz
    window; the refit of S under OMPD1_1 must keep the 3-13 Hz window.
    """
    refit_windows = []
    grid_search = fid_model._fit_parameter_grid

    def recording_grid_search(data, options, *, parameter):
        if parameter == "frequency" and len(options["metabs"]) == 1:
            refit_windows.append(
                (
                    list(options["metabs"]),
                    float(np.real(options["freq_range_lower_bound_hz"][0])),
                    float(np.real(options["freq_range_upper_bound_hz"][0])),
                )
            )
        return grid_search(data, options, parameter=parameter)

    monkeypatch.setattr(fid_model, "_fit_parameter_grid", recording_grid_search)
    data = two_voxel_doublet().isel(x=[1])
    params = doublet_params(
        data,
        metabs_freq_ranges=NESTED_RANGES_HZ,
        t2_coupling=BAND_COUPLING,
        t2_coupling_clip_start=True,
        snr_threshold=3.0,
    )

    result = fit_data_pseudo_inv(data, params)

    assert refit_windows
    assert all(window == (["OMPD1_1"], 3.0, 13.0) for window in refit_windows)
    assert bool(result.fit_low_snr_refitted.sel(metabolite="OMPD1_2").item())
    assert 3.0 <= float(result.fit_freqs.sel(metabolite="OMPD1_1").item()) <= 13.0


def test_skipped_leader_leaves_the_dependent_in_its_own_window():
    """Without its leader, OMPD1_2 is fitted alone inside its 3-13 Hz window."""
    data = two_voxel_doublet()
    params = doublet_params(
        data,
        metabs_freq_ranges=NESTED_RANGES_HZ,
        t2_coupling=BAND_COUPLING,
        t2_coupling_clip_start=True,
    ).copy(deep=True)
    params["skip_fitting"].loc[{"metabolite": "OMPD1_1", "x": 1}] = True

    result = fit_data_pseudo_inv(data, params)

    assert list(result.fit_status.values) == ["success", "success"]
    np.testing.assert_allclose(
        result.fit_freqs.sel(x=0).values, [0.0, SHIFT_HZ], atol=0.1
    )
    assert 3.0 <= float(result.fit_freqs.sel(x=1, metabolite="OMPD1_2")) <= 13.0
    assert bool(result.fit_skipped.sel(x=1, metabolite="OMPD1_1"))


def test_coupled_lines_accept_different_phase_tolerances():
    """Group members may carry different phase-band tolerances.

    The true phase difference of S and W is 40 degrees; with tolerances 0
    and 30 degrees the phase law allows at most 30 degrees in both voxels.
    """
    data = two_voxel_doublet()
    params = doublet_params(
        data,
        metabs_freq_ranges=NESTED_RANGES_HZ,
        t2_coupling=BAND_COUPLING,
        t2_coupling_clip_start=True,
        phase_band_tolerance_deg={"OMPD1_1": 0.0, "OMPD1_2": 30.0},
        phase_band_tau_s=0.0,
    )

    result = fit_data_pseudo_inv(data, params)

    assert list(result.fit_status.values) == ["success", "success"]
    phases_deg = np.degrees(np.angle(result.fit_amps.values))
    assert np.all(np.abs(phases_deg[:, 1] - phases_deg[:, 0]) <= 30.0 + 1e-6)
    np.testing.assert_allclose(phases_deg[1], phases_deg[0], atol=0.2)
    np.testing.assert_allclose(
        result.fit_freqs.values[1], result.fit_freqs.values[0] + SHIFT_HZ, atol=0.1
    )
    np.testing.assert_allclose(
        result.fit_t2s.values[1], result.fit_t2s.values[0], atol=2e-4
    )


def test_weak_band_leader_recovers_its_own_t2():
    """A weak band leader leaves the strong dependent's T2* in both voxels."""
    data = two_voxel_doublet()
    params = doublet_params(
        data, t2_coupling=WEAK_LEADER_COUPLING, t2_coupling_clip_start=True
    )

    result = fit_data_pseudo_inv(data, params)

    np.testing.assert_allclose(
        result.fit_t2s.values,
        [[STRONG_T2_S, WEAK_T2_S], [STRONG_T2_S, WEAK_T2_S]],
        atol=1e-4,
    )


def test_band_leader_step_is_recorded_in_history(band_fit):
    """The leader's own move has as many candidates as its joint move."""
    info = band_fit.fit_history_info.sel(x=0)
    event = info.sel(history_field="event")
    metabolite = info.sel(history_field="metabolite_index")
    own = event == HISTORY_EVENTS["t2_band_leader_candidate"]
    joint = event == HISTORY_EVENTS["t2_candidate"]

    assert int(own.sum()) > 0
    assert set(metabolite.values[own.values]) == {0}
    assert int(own.sum()) == int((joint & (metabolite == 0)).sum())


def test_voigt_coupled_fit_is_shift_invariant():
    """A weak Voigt band leader gives the same fit in both voxels."""
    data = two_voxel_doublet(gauss_fwhm_hz=6.0)
    params = doublet_params(
        data,
        t2_coupling=WEAK_LEADER_COUPLING,
        t2_coupling_clip_start=True,
        **VOIGT_OPTIONS,
    )

    result = fit_data_pseudo_inv(data, params)

    np.testing.assert_allclose(
        result.fit_freqs.sel(x=1).values,
        result.fit_freqs.sel(x=0).values + SHIFT_HZ,
        atol=0.01,
    )
    np.testing.assert_allclose(
        result.fit_t2s.sel(x=1).values, result.fit_t2s.sel(x=0).values, atol=1e-5
    )


def test_t2_search_moves_add_an_own_move_for_band_leaders():
    """Fixed dependents are not searched; band leaders move twice."""
    coupling = [(1, 0, 0.5)]
    bands = [(2, 0, 0.8, 1.25)]

    assert _t2_search_moves(4, coupling, bands) == [
        (0, True),
        (0, False),
        (2, True),
        (3, True),
    ]


def test_t2_move_constraints_hold_banded_dependents_in_the_own_move():
    """Joint moves carry banded dependents; own moves check their bands."""
    coupling = [(1, 0, 0.5)]
    bands = [(2, 0, 0.8, 1.25)]
    t2s = np.array([0.04, 0.02, 0.036, 0.05])

    followers, checked = _t2_move_constraints(0, True, t2s, coupling, bands)
    assert followers == [(1, 0.5), (2, pytest.approx(0.9))]
    assert checked == []

    followers, checked = _t2_move_constraints(0, False, t2s, coupling, bands)
    assert followers == [(1, 0.5)]
    assert checked == bands

    followers, checked = _t2_move_constraints(2, True, t2s, coupling, bands)
    assert followers == []
    assert checked == bands
