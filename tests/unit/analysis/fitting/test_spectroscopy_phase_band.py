"""Characterize the phase-band amplitude solver of the FID fit.

Fixtures are synthetic FIDs: 256 samples at 4 kHz, all T2* 0.03 s, lines at
-300, 0, 20 and 45 Hz and a phase-law delay tau of 1.5 ms. Line phases follow
``phi0 + 2 pi f tau`` plus chosen deviations, and complex white noise has
sigma 0.02 per quadrature. With a non-zero tolerance a range of phi0 values
reaches the zero residual of a noiseless law signal, so recovery is checked
to rtol 1e-9. With tolerance 0 only the true phi0 does, and the bounded
search stops within 1e-4 rad of it, so recovery is checked to rtol 2e-4.
Fitted phases with tolerance 0 follow the law to 1e-9 rad for any phi0. Where the free solution already satisfies the
band, the solver returns it without the phase search; both paths agree to
1e-9 relative, or to the search's phase tolerance with a zero tolerance.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.optimize import lsq_linear

from hypermri.analysis.fitting.models.spectroscopy import fid_model
from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    fid_basefunc,
    solve_for_complex_amplitudes,
)
from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv
from hypermri.analysis.spectroscopy import (
    estimate_fit_covariance,
    refit_spectroscopy_voxel,
)

TAU_S = 1.5e-3
TIME_S = np.arange(256) / 4000.0
FREQS_HZ = np.array([-300.0, 0.0, 20.0, 45.0])
T2S_S = np.full(4, 0.03)
NOISE_SIGMA = 0.02


def law_amplitudes(areas, phi0_rad, deviations_rad=0.0, freqs_hz=FREQS_HZ):
    """Return complex amplitudes that follow the phase law plus deviations."""
    return np.asarray(areas) * np.exp(
        1j * (phi0_rad + 2 * np.pi * freqs_hz * TAU_S + np.asarray(deviations_rad))
    )


def law_phase(amplitudes, freqs_hz):
    """Return each line's phase after removing ``2 pi f tau``, in radians."""
    return np.angle(np.asarray(amplitudes) * np.exp(-2j * np.pi * freqs_hz * TAU_S))


def wrapped(angle_rad):
    """Map angles to [-pi, pi)."""
    return (np.asarray(angle_rad) + np.pi) % (2 * np.pi) - np.pi


def noisy_fid(amplitudes, freqs_hz=FREQS_HZ, seed=0):
    """Return one noisy FID of Lorentz lines with T2* 0.03 s."""
    rng = np.random.default_rng(seed)
    clean = fid_basefunc(freqs_hz, np.full(len(freqs_hz), 0.03), TIME_S).T @ amplitudes
    noise = rng.standard_normal(TIME_S.size) + 1j * rng.standard_normal(TIME_S.size)
    return clean + NOISE_SIGMA * noise


def solve(fid, tolerance_deg):
    """Solve amplitudes at the true frequencies and T2* with a phase band."""
    return solve_for_complex_amplitudes(
        fid,
        FREQS_HZ,
        T2S_S,
        TIME_S,
        signal_domain="temporal",
        amp_solver="pinv",
        phase_band_tolerance_deg=np.asarray(tolerance_deg, dtype=float),
        phase_band_tau_s=TAU_S,
    )[1]


def test_free_lines_give_the_pinv_solution():
    """Infinite tolerances leave every amplitude to the ordinary pinv solve."""
    fid = noisy_fid(law_amplitudes([10, 20, 12, 5], 0.4, [0.3, -0.5, 0.2, 1.0]))
    free = solve_for_complex_amplitudes(
        fid, FREQS_HZ, T2S_S, TIME_S, signal_domain="temporal", amp_solver="pinv"
    )[1]
    np.testing.assert_array_equal(solve(fid, np.full(4, np.inf)), free)


@pytest.mark.parametrize(
    ("tolerance_deg", "rtol"), [(0.0, 2e-4), (3.0, 1e-9), (90.0, 1e-9)]
)
def test_noiseless_law_signal_is_recovered(tolerance_deg, rtol):
    """A noiseless feasible signal is the unique zero-residual solution."""
    truth = law_amplitudes([10, 20, 12, 5], 0.4)
    fid = fid_basefunc(FREQS_HZ, T2S_S, TIME_S).T @ truth
    np.testing.assert_allclose(solve(fid, np.full(4, tolerance_deg)), truth, rtol=rtol)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_banded_lines_stay_inside_their_band(seed):
    """Banded phases differ by at most the sum of their tolerances.

    The true phases deviate from the law by up to 30 degrees, so the free
    pinv solution leaves the band and the constraint is active.
    """
    tolerance_deg = np.array([20.0, 3.0, 3.0, np.inf])
    deviations = np.radians([25.0, -30.0, 30.0, 80.0])
    fid = noisy_fid(law_amplitudes([10, 20, 12, 5], 0.4, deviations), seed=seed)
    phases = law_phase(solve(fid, tolerance_deg), FREQS_HZ)
    free_phases = law_phase(solve(fid, np.full(4, np.inf)), FREQS_HZ)
    banded = np.flatnonzero(np.isfinite(tolerance_deg))
    limit = np.radians(tolerance_deg[banded, None] + tolerance_deg[None, banded])
    spread = np.abs(wrapped(phases[banded, None] - phases[None, banded]))
    free_spread = np.abs(wrapped(free_phases[banded, None] - free_phases[None, banded]))
    assert np.all(spread <= limit + 1e-9)
    assert np.any(free_spread > limit)


def brute_force_rss(fid, tolerance_deg, n_phases=3600):
    """Return the smallest RSS over a dense phi0 grid with edge rays.

    For tolerances below 90 degrees each banded amplitude is a non-negative
    combination of its two edge rays; free lines get unbounded real and
    imaginary weights. ``lsq_linear`` solves each grid point independently of
    the projection and QR path of the library solver.
    """
    basis = fid_basefunc(FREQS_HZ, T2S_S, TIME_S).T
    target = np.concatenate([fid.real, fid.imag])
    best = np.inf
    for phi0 in np.linspace(-np.pi, np.pi, n_phases, endpoint=False):
        columns, lower = [], []
        for line, tolerance in enumerate(np.radians(tolerance_deg)):
            if np.isfinite(tolerance):
                centre = phi0 + 2 * np.pi * FREQS_HZ[line] * TAU_S
                for edge in (-tolerance, tolerance):
                    columns.append(basis[:, line] * np.exp(1j * (centre + edge)))
                    lower.append(0.0)
            else:
                columns += [basis[:, line], 1j * basis[:, line]]
                lower += [-np.inf, -np.inf]
        matrix = np.array(columns).T
        matrix = np.vstack([matrix.real, matrix.imag])
        result = lsq_linear(matrix, target, bounds=(lower, np.inf))
        best = min(best, float(np.sum(result.fun**2)))
    return best


@pytest.mark.parametrize("seed", [0, 1])
def test_phase_search_reaches_the_dense_grid_optimum(seed):
    """The 24-point grid plus bounded refinement is not worse than 3600 points."""
    tolerance_deg = np.array([20.0, 3.0, 3.0, np.inf])
    deviations = np.radians([25.0, -30.0, 30.0, 80.0])
    fid = noisy_fid(law_amplitudes([10, 20, 12, 5], 0.4, deviations), seed=seed)
    fitted = fid_basefunc(FREQS_HZ, T2S_S, TIME_S).T @ solve(fid, tolerance_deg)
    rss = float(np.sum(np.abs(fid - fitted) ** 2))
    assert rss <= brute_force_rss(fid, tolerance_deg) * (1 + 1e-9)


@pytest.mark.parametrize(
    ("amplitudes", "law_phase_rad", "tolerance_deg", "expected"),
    [
        ([1, 1j], [0.0, np.pi / 2], [5.0, 5.0], True),
        ([1, np.exp(0.15j)], [0.0, 0.0], [5.0, 6.0], True),
        ([1, np.exp(0.15j)], [0.0, 0.0], [3.0, 3.0], False),
        ([np.exp(3.1j), np.exp(-3.1j)], [0.0, 0.0], [5.0, 5.0], True),
        ([np.exp(3.1j), np.exp(-3.1j)], [0.0, 0.0], [2.0, 2.0], False),
        ([1, 0, -1], [0.0, 0.0, np.pi], [0.0, 0.0, 0.0], True),
        ([1, 0.0, 1j], [0.0, 1.0, 0.0], [10.0, 0.0, 10.0], False),
        ([0, 0], [0.0, 1.0], [0.0, 0.0], True),
    ],
    ids=[
        "law",
        "overlap",
        "apart",
        "across_pi",
        "apart_across_pi",
        "zero_tolerance",
        "zero_amplitude",
        "all_zero",
    ],
)
def test_phase_band_holds_tests_for_one_common_phase(
    amplitudes, law_phase_rad, tolerance_deg, expected
):
    """Arcs of allowed common phases intersect, also across +/- pi."""
    assert (
        fid_model._phase_band_holds(
            np.asarray(amplitudes, dtype=complex),
            np.asarray(law_phase_rad),
            np.radians(tolerance_deg),
        )
        is expected
    )


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "tolerance_deg",
    [[10.0, 10.0, 10.0, 10.0], [20.0, 3.0, 3.0, np.inf], [0.0, 30.0, 30.0, np.inf]],
    ids=["all_banded", "one_free", "zero_tolerance"],
)
def test_early_exit_equals_the_phase_search(monkeypatch, seed, tolerance_deg):
    """Skipping the search where the free solution holds changes nothing.

    Deviations from the law of up to 12 degrees put the free solution inside
    some bands and outside others. With positive tolerances both paths give
    the same amplitudes to 1e-9 relative (measured: below 1e-14 where the
    exit applies). A zero tolerance pins the common phase to one line, so the
    search result carries its 1e-4 rad phase tolerance while the exit is the
    exact optimum: amplitudes then agree to 1e-4 relative and the exit's RSS
    is never higher.
    """
    rng = np.random.default_rng(seed)
    deviations = np.radians(rng.uniform(-12, 12, 4))
    fid = noisy_fid(law_amplitudes([10, 20, 12, 5], 0.4, deviations), seed=seed)
    tolerance_deg = np.asarray(tolerance_deg)
    with_exit = solve(fid, tolerance_deg)
    monkeypatch.setattr(fid_model, "_phase_band_holds", lambda *args: False)
    searched = solve(fid, tolerance_deg)

    rtol = 1e-4 if np.any(tolerance_deg == 0) else 1e-9
    np.testing.assert_allclose(
        with_exit, searched, rtol=0, atol=rtol * np.max(np.abs(searched))
    )
    basis = fid_basefunc(FREQS_HZ, T2S_S, TIME_S).T
    rss_exit, rss_searched = (
        np.sum(np.abs(fid - basis @ amplitudes) ** 2)
        for amplitudes in (with_exit, searched)
    )
    assert rss_exit <= rss_searched * (1 + 1e-12)


def test_early_exit_skips_the_phase_search_only_inside_the_band(monkeypatch):
    calls = []
    original = fid_model.nnls

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fid_model, "nnls", counted)
    inside = noisy_fid(law_amplitudes([10, 20, 12, 5], 0.4, np.radians([2, -2, 1, 0])))
    outside = noisy_fid(
        law_amplitudes([10, 20, 12, 5], 0.4, np.radians([25, -30, 0, 0]))
    )

    solve(inside, np.full(4, 10.0))
    assert not calls
    solve(outside, np.full(4, 10.0))
    assert calls


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"tolerance_deg": None}, "given together"),
        ({"tau_s": None}, "given together"),
        ({"tau_s": np.inf}, "must be finite"),
        ({"tolerance_deg": np.zeros(3)}, "one value per metabolite"),
        ({"tolerance_deg": np.array([0, 0, 0, 91.0])}, "must lie in"),
        ({"tolerance_deg": np.array([0, 0, 0, np.nan])}, "must lie in"),
        ({"amp_solver": "normal_eq"}, "requires amp_solver='pinv'"),
    ],
)
def test_direct_solver_rejects_invalid_band(options, message):
    """Direct calls fail before solving on incomplete or invalid settings."""
    settings = {"tolerance_deg": np.zeros(4), "tau_s": TAU_S, "amp_solver": "pinv"}
    settings.update(options)
    with pytest.raises(ValueError, match=message):
        solve_for_complex_amplitudes(
            noisy_fid(law_amplitudes([10, 20, 12, 5], 0.4)),
            FREQS_HZ,
            T2S_S,
            TIME_S,
            signal_domain="temporal",
            amp_solver=settings["amp_solver"],
            phase_band_tolerance_deg=settings["tolerance_deg"],
            phase_band_tau_s=settings["tau_s"],
        )


def labelled_fid(amplitudes, freqs_hz, seed=0):
    """Return one labelled noisy FID as a single-voxel DataArray."""
    data = xr.DataArray(
        noisy_fid(amplitudes, freqs_hz, seed)[None],
        dims=["x", "acquisition_idx"],
        coords={"x": [0]},
    )
    data.attrs["sampling_time_axis_s"] = TIME_S
    return data


def fit_params(data, names, freqs_hz, **options):
    """Prepare fit parameters with starts 1 Hz away from the true lines."""
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": names,
            "metabs_freqs_hz": list(np.asarray(freqs_hz) + 1.0),
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 8.0,
            "min_t2_s": 0.005,
            "max_t2_s": 0.1,
            **options,
        },
    )


THREE_FREQS_HZ = FREQS_HZ[:3]
THREE_NAMES = ["a", "b", "c"]
OFF_LAW = law_amplitudes(
    [10, 20, 12], 0.4, np.radians([20.0, -25.0, 25.0]), THREE_FREQS_HZ
)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"phase_band_tolerance_deg": ["a"]}, "must be a dict"),
        ({"phase_band_tolerance_deg": {"x": 3.0}}, "not a fitted metabolite"),
        ({"phase_band_tolerance_deg": {"a": 91.0}}, "must lie in"),
        ({"phase_band_tolerance_deg": {"a": -1.0}}, "must lie in"),
        (
            {"phase_band_tolerance_deg": {"a": 3.0}, "phase_band_tau_s": None},
            "requires phase_band_tau_s",
        ),
        (
            {"phase_band_tolerance_deg": {"a": 3.0}, "phase_band_tau_s": np.nan},
            "finite scalar",
        ),
        ({"phase_band_tolerance_deg": None}, "only used with"),
        (
            {"phase_band_tolerance_deg": {"a": 3.0}, "amp_solver": "ridge"},
            "amp_solver='pinv'",
        ),
        (
            {"phase_band_tolerance_deg": {"a": 3.0}, "detection_target": "c"},
            "detection_target",
        ),
    ],
)
def test_invalid_band_options_are_rejected(options, message):
    """Parameter preparation names the wrong option and how to correct it."""
    data = labelled_fid(OFF_LAW, THREE_FREQS_HZ)
    with pytest.raises(ValueError, match=message):
        fit_params(
            data, THREE_NAMES, THREE_FREQS_HZ, **{"phase_band_tau_s": TAU_S, **options}
        )


def test_fit_follows_the_law_and_records_the_options():
    """With tolerance 0 the fitted lines share one law phase; pinv does not."""
    data = labelled_fid(OFF_LAW, THREE_FREQS_HZ)
    band = {"a": 0.0, "b": 0, "c": 0.0}
    banded = fit_data_pseudo_inv(
        data,
        fit_params(
            data,
            THREE_NAMES,
            THREE_FREQS_HZ,
            phase_band_tolerance_deg=band,
            phase_band_tau_s=TAU_S,
        ),
    )
    free = fit_data_pseudo_inv(data, fit_params(data, THREE_NAMES, THREE_FREQS_HZ))
    assert banded.attrs["phase_band_tolerance_deg"] == {"a": 0.0, "b": 0.0, "c": 0.0}
    assert banded.attrs["phase_band_tau_s"] == TAU_S
    assert banded.fit_status.values[0] == "success"
    phases = law_phase(banded.fit_amps.values[0], banded.fit_freqs.values[0])
    free_phases = law_phase(free.fit_amps.values[0], free.fit_freqs.values[0])
    np.testing.assert_allclose(wrapped(phases - phases[0]), 0.0, atol=1e-9)
    assert np.max(np.abs(wrapped(free_phases - free_phases[0]))) > np.radians(10)


@pytest.mark.parametrize(
    "exclusion",
    [{"skip_fitting": [True, False, False, False]}, {"snr_threshold": 5.0}],
    ids=["skipped", "amplitude_filtered"],
)
def test_band_follows_names_when_a_line_is_excluded(exclusion):
    """Tolerances stay with their line when an earlier line is not fitted.

    'z' at 300 Hz has no signal and is excluded. With a positional mix-up,
    the tolerance 0 of 'a' would move to the free line 'b'. A skipped 'z'
    gives exactly the fit without 'z'; an amplitude-filtered 'z' is refitted
    from the first pass, so only the law of 'a' and 'c' is checked there.
    """
    freqs_hz = np.array([300.0, *THREE_FREQS_HZ])
    data = labelled_fid(np.array([0.0, *OFF_LAW]), freqs_hz)
    band = {"z": 0.0, "a": 0.0, "c": 0.0}
    result = fit_data_pseudo_inv(
        data,
        fit_params(
            data,
            ["z", *THREE_NAMES],
            freqs_hz,
            phase_band_tolerance_deg=band,
            phase_band_tau_s=TAU_S,
            **exclusion,
        ),
    )
    amplitudes, freqs = result.fit_amps.values[0], result.fit_freqs.values[0]
    phases = law_phase(amplitudes, freqs)
    np.testing.assert_allclose(wrapped(phases[3] - phases[1]), 0.0, atol=1e-9)
    assert abs(wrapped(phases[2] - phases[1])) > np.radians(10)
    if "skip_fitting" in exclusion:
        reference = fit_data_pseudo_inv(
            data,
            fit_params(
                data,
                THREE_NAMES,
                THREE_FREQS_HZ,
                phase_band_tolerance_deg={"a": 0.0, "c": 0.0},
                phase_band_tau_s=TAU_S,
            ),
        )
        np.testing.assert_allclose(
            amplitudes[1:], reference.fit_amps.values[0], rtol=1e-12
        )
    else:
        np.testing.assert_array_equal(
            result.fit_low_snr_refitted.values[0], [True, False, False, False]
        )


def test_voxel_refit_keeps_the_band():
    """The review refit and its final amplitude solve use the stored band."""
    data = labelled_fid(OFF_LAW, THREE_FREQS_HZ)
    params = fit_params(
        data,
        THREE_NAMES,
        THREE_FREQS_HZ,
        phase_band_tolerance_deg={"a": 0.0, "b": 0.0, "c": 0.0},
        phase_band_tau_s=TAU_S,
    )
    refitted = refit_spectroscopy_voxel(
        data, params, fit_data_pseudo_inv(data, params), voxel_indices={"x": 0}
    )["fit_results"]
    phases = law_phase(
        np.ravel(refitted.fit_amps.values), np.ravel(refitted.fit_freqs.values)
    )
    np.testing.assert_allclose(wrapped(phases - phases[0]), 0.0, atol=1e-9)


def test_local_covariance_rejects_banded_fits():
    """The local Jacobian treats amplitudes as free, so banded fits are refused."""
    data = labelled_fid(OFF_LAW, THREE_FREQS_HZ)
    params = fit_params(
        data,
        THREE_NAMES,
        THREE_FREQS_HZ,
        phase_band_tolerance_deg={"a": 3.0},
        phase_band_tau_s=TAU_S,
    )
    result = fit_data_pseudo_inv(data, params)
    with pytest.raises(ValueError, match="phase_band_tolerance_deg"):
        estimate_fit_covariance(result, params, noise=NOISE_SIGMA)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_parallel_banded_fit_equals_serial_fit(backend):
    """Joblib workers give the serial phase-band fit of every voxel.

    The interactive fit workflow starts with ``n_jobs=-1``, so the phase-band
    solver must not depend on the execution path. Three voxels with
    different noise draws are fitted one per task; each voxel's arithmetic
    is the same in both paths, so values agree to rtol 1e-12.
    """
    data = xr.concat(
        [labelled_fid(OFF_LAW, THREE_FREQS_HZ, seed) for seed in range(3)], dim="x"
    ).assign_coords(x=[0, 1, 2])
    data.attrs["sampling_time_axis_s"] = TIME_S
    params = fit_params(
        data,
        THREE_NAMES,
        THREE_FREQS_HZ,
        phase_band_tolerance_deg={"a": 10.0, "b": 10.0, "c": 10.0},
        phase_band_tau_s=TAU_S,
    )

    serial = fit_data_pseudo_inv(data, params, n_jobs=1)
    parallel = fit_data_pseudo_inv(
        data, params, n_jobs=2, batch_size=1, parallel_backend=backend
    )

    assert list(serial.fit_status.values) == ["success"] * 3
    for name in ("fit_amps", "fit_freqs", "fit_t2s"):
        np.testing.assert_allclose(
            parallel[name].values, serial[name].values, rtol=1e-12, atol=0
        )
