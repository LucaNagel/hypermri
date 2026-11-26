"""Joint refinement of all frequencies and T2* after the FID grid passes.

Fixture provenance: synthetic two-line FIDs, 512 samples at 4 kHz. The
strong line S has area 20, phase -0.2 rad and T2* 30 ms at 0 Hz; the weak
line W has area 2, phase 0.5 rad and T2* 25 ms at 8 Hz. Both starts lie
2 Hz too high, with 40 Hz ranges; with 31 grid points both true frequencies
lie between two points of the first grid, and the grid alone ends several Hz
off (``test_spectroscopy_doublet_search.py``). Noisy voxels add complex
Gaussian noise with standard deviation 0.05 per channel from a fixed seed.

Tolerances: bounds and coupling ratios hold up to floating-point rounding
(relative 1e-12); recovered noise-free lines lie within 0.1 Hz and 0.1 ms
of the truth, the grid resolution after the zoomed passes. RSS comparisons
allow 1e-9 relative for summation order; the magnitude objective is
recomputed with the default minimum peak separation, which both fits keep.

Analytic derivatives: ``J.T @ r`` matches the central-difference gradient of
``|r|**2 / 2`` (step 1e-6 relative) to 1e-7 relative, and to 1e-4 when every
line is phase-banded, where the objective carries the common-phase search
tolerance of 1e-4 rad. Fits with analytic and finite-difference derivatives
agree to 1e-6 Hz, 1e-8 s and 1e-12 relative RSS without a phase band. With
one, both stop on the same noisy objective within 0.01 Hz and 0.05 ms; the
analytic fit's RSS is at most 1e-6 relative higher (it is often lower, by
up to 1e-4 on this fixture).
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy import fid_model
from hypermri.analysis.fitting.models.spectroscopy.diagnostics import HISTORY_EVENTS
from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv

TIME_S = np.arange(512) / 4000.0
FREQUENCY_TOLERANCE_HZ = 0.1
T2_TOLERANCE_S = 1e-4


def doublet(n_voxels=1, noise_std=0.0, *, gauss_fwhm_hz=0.0):
    """Return the S/W doublet, optionally with Gaussian broadening and noise."""
    gauss = np.exp(-((np.pi * gauss_fwhm_hz * TIME_S) ** 2) / (4 * np.log(2)))
    fid = gauss * (
        20.0 * np.exp(-0.2j) * np.exp(-TIME_S / 0.030)
        + 2.0 * np.exp(0.5j) * np.exp(2j * np.pi * 8.0 * TIME_S - TIME_S / 0.025)
    )
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((n_voxels, 512)) + 1j * rng.standard_normal(
        (n_voxels, 512)
    )
    data = xr.DataArray(
        fid[None] + noise_std * noise,
        dims=["x", "acquisition_idx"],
        coords={"x": np.arange(n_voxels)},
    )
    data.attrs["sampling_time_axis_s"] = TIME_S
    return data


def params(data, **options):
    """Return the fixture's fit parameters with extra ``options``."""
    return def_fit_params(
        data,
        {
            "metabs": ["strong", "weak"],
            "metabs_freqs_hz": [2.0, 10.0],
            "metabs_freq_ranges": 40.0,
            "min_t2_s": 0.005,
            "max_t2_s": 0.2,
            "keep_fit_order": True,
            **options,
        },
    )


def test_refinement_is_on_by_default_and_must_be_boolean():
    data = doublet()

    assert fid_model.JOINT_REFINEMENT_DEFAULT is True
    with pytest.raises(ValueError, match="joint_refinement must be boolean"):
        fit_data_pseudo_inv(data, params(data, joint_refinement="yes"))


def test_refinement_recovers_the_doublet_the_grid_misses():
    data = doublet()

    result = fit_data_pseudo_inv(data, params(data))

    np.testing.assert_allclose(
        result.fit_freqs.values[0], [0.0, 8.0], atol=FREQUENCY_TOLERANCE_HZ
    )
    np.testing.assert_allclose(
        result.fit_t2s.values[0], [0.030, 0.025], atol=T2_TOLERANCE_S
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"use_sse_mag": True},
        {"lineshape": "voigt"},
        {"t2_coupling": {"weak": ("strong", 0.85)}},
        {
            "t2_coupling": {"weak": ("strong", (0.8, 0.9))},
            "t2_coupling_clip_start": True,
        },
    ],
    ids=["lorentz", "sse_mag", "voigt", "fixed_ratio", "ratio_band"],
)
def test_refinement_never_raises_the_objective(options):
    data = doublet(n_voxels=3, noise_std=0.05)

    grid = fit_data_pseudo_inv(data, params(data, joint_refinement=False, **options))
    refined = fit_data_pseudo_inv(data, params(data, **options))

    assert np.all(
        objective(data, refined, options) <= objective(data, grid, options) * (1 + 1e-9)
    )


def objective(data, result, options):
    """Return the grid objective of every spectrum's fitted parameters.

    It is ``fit_rss`` except with ``use_sse_mag``, where it is recomputed
    from the fitted frequencies and T2* values.
    """
    if not options.get("use_sse_mag"):
        return result.fit_rss.values
    return np.array(
        [
            fid_model._score_fit_parameters(
                data.values[voxel],
                result.fit_freqs.values[voxel],
                result.fit_t2s.values[voxel],
                {"sampling_time_axis_s": TIME_S, "use_sse_mag": True},
            )[0]
            for voxel in range(data.sizes["x"])
        ]
    )


def test_refinement_keeps_the_coupling_ratios():
    data = doublet(n_voxels=3, noise_std=0.05)

    fixed = fit_data_pseudo_inv(
        data, params(data, t2_coupling={"weak": ("strong", 0.85)})
    )
    band = fit_data_pseudo_inv(
        data,
        params(
            data,
            t2_coupling={"weak": ("strong", (0.8, 0.9))},
            t2_coupling_clip_start=True,
        ),
    )

    np.testing.assert_allclose(
        fixed.fit_t2s.sel(metabolite="weak"),
        0.85 * fixed.fit_t2s.sel(metabolite="strong"),
        rtol=1e-12,
    )
    ratio = band.fit_t2s.sel(metabolite="weak") / band.fit_t2s.sel(metabolite="strong")
    assert np.all((ratio >= 0.8 * (1 - 1e-12)) & (ratio <= 0.9 * (1 + 1e-12)))


def test_refinement_stays_inside_the_absolute_bounds():
    # The weak line's truth (8 Hz, 25 ms) lies outside its bounds.
    data = doublet()
    prepared = params(
        data,
        metabs_freq_ranges=[40.0, 8.0],
        initial_t2s_s=[0.03, 0.015],
        min_t2_s=[0.005, 0.005],
        max_t2_s=[0.2, 0.02],
    )

    result = fit_data_pseudo_inv(data, prepared)

    weak = result.sel(metabolite="weak")
    assert 6.0 <= weak.fit_freqs.item() <= 14.0
    assert weak.fit_t2s.item() <= 0.02


def test_refinement_leaves_frozen_parameters_at_their_start():
    data = doublet(n_voxels=2, noise_std=0.05)
    prepared = params(
        data,
        metabs_freqs_hz=[0.3, 10.0],
        metabs_freq_ranges=[0.0, 40.0],
        initial_t2s_s=[0.028, 0.03],
        metabs_t2_ranges=[0.0, 0.03],
    )

    result = fit_data_pseudo_inv(data, prepared)

    strong = result.sel(metabolite="strong")
    np.testing.assert_array_equal(strong.fit_freqs.values, [0.3, 0.3])
    np.testing.assert_array_equal(strong.fit_t2s.values, [0.028, 0.028])


def test_refinement_keeps_the_minimum_peak_separation():
    # The truth is 8 Hz apart; 10 Hz are required.
    data = doublet(n_voxels=3, noise_std=0.05)
    prepared = params(data, metabs_freqs_hz=[0.0, 12.0], min_peak_separation_hz=10.0)

    result = fit_data_pseudo_inv(data, prepared)

    separation = result.fit_freqs.sel(metabolite="weak") - result.fit_freqs.sel(
        metabolite="strong"
    )
    assert np.all(np.abs(separation) >= 10.0)


@pytest.mark.parametrize("joint_refinement", [False, True])
def test_history_records_one_refinement_per_search(joint_refinement):
    data = doublet(n_voxels=2, noise_std=0.05)
    prepared = params(
        data, joint_refinement=joint_refinement, store_optimization_history=True
    )

    result = fit_data_pseudo_inv(data, prepared)

    events = result.fit_history_info.sel(history_field="event")
    count = (events == HISTORY_EVENTS["joint_refinement"]).sum("history_step")
    np.testing.assert_array_equal(count, [int(joint_refinement)] * 2)


def test_refinement_keeps_the_gaussian_width_of_a_voigt_fit():
    data = doublet(gauss_fwhm_hz=4.0)
    prepared = params(data, lineshape="voigt", gauss_search_width_hz=0.0)

    result = fit_data_pseudo_inv(data, prepared)

    assert result.fit_gauss_fwhm_hz.item() == prepared.attrs["gauss_initial_fwhm_hz"]


def test_spectral_input_gives_the_temporal_result():
    data = doublet(n_voxels=2, noise_std=0.05)
    spectrum = data.copy(
        data=np.fft.fftshift(np.fft.fft(data.values, axis=-1), axes=-1)
    )

    temporal = fit_data_pseudo_inv(data, params(data))
    spectral = fit_data_pseudo_inv(spectrum, params(spectrum, signal_domain="spectral"))

    np.testing.assert_allclose(
        spectral.fit_freqs, temporal.fit_freqs, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(spectral.fit_t2s, temporal.fit_t2s, atol=1e-9, rtol=0)


def test_parallel_refinement_equals_serial():
    data = doublet(n_voxels=4, noise_std=0.05)
    prepared = params(data)

    serial = fit_data_pseudo_inv(data, prepared)
    parallel = fit_data_pseudo_inv(
        data, prepared, n_jobs=2, batch_size=1, parallel_backend="threading"
    )

    np.testing.assert_array_equal(parallel.fit_freqs, serial.fit_freqs)
    np.testing.assert_array_equal(parallel.fit_t2s, serial.fit_t2s)
    np.testing.assert_array_equal(parallel.fit_amps, serial.fit_amps)


PHASE_BAND = {"phase_band_tau_s": 1e-3}
DERIVATIVE_CASES = {
    "lorentz": ({}, 1e-7),
    "voigt": ({"lineshape": "voigt"}, 1e-7),
    "fixed_ratio": ({"t2_coupling": {"weak": ("strong", 0.85)}}, 1e-7),
    "ratio_band": (
        {
            "t2_coupling": {"weak": ("strong", (0.8, 0.9))},
            "t2_coupling_clip_start": True,
        },
        1e-7,
    ),
    "one_banded": ({"phase_band_tolerance_deg": {"weak": 15.0}, **PHASE_BAND}, 1e-7),
    "all_banded": (
        {"phase_band_tolerance_deg": {"strong": 10.0, "weak": 10.0}, **PHASE_BAND},
        1e-4,
    ),
}


def refinement_calls(monkeypatch, data, options, *, forced_jac=None):
    """Fit and return the ``least_squares`` calls of the joint refinement.

    With ``forced_jac`` the refinement uses it instead of its own choice.
    """
    calls = []
    original = fid_model.least_squares

    def recorded(fun, x0, *, jac, bounds, **kwargs):
        jac = jac if forced_jac is None else forced_jac
        calls.append({"fun": fun, "jac": jac, "bounds": bounds})
        return original(fun, x0, jac=jac, bounds=bounds, **kwargs)

    monkeypatch.setattr(fid_model, "least_squares", recorded)
    result = fit_data_pseudo_inv(data, params(data, **options))
    return calls, result


@pytest.mark.parametrize(
    ("options", "rtol"), DERIVATIVE_CASES.values(), ids=DERIVATIVE_CASES.keys()
)
def test_analytic_gradient_matches_finite_differences(monkeypatch, options, rtol):
    data = doublet(noise_std=0.05)
    call = refinement_calls(monkeypatch, data, options)[0][0]
    lower, upper = call["bounds"]
    rng = np.random.default_rng(3)

    def half_rss(x):
        return 0.5 * np.sum(call["fun"](x) ** 2)

    assert callable(call["jac"])
    for _ in range(3):
        x = lower + (upper - lower) * rng.uniform(0.2, 0.8, size=lower.size)
        gradient = call["jac"](x).T @ call["fun"](x)
        steps = 1e-6 * np.maximum(1.0, np.abs(x))
        numeric = [
            (half_rss(x + step * unit) - half_rss(x - step * unit)) / (2 * step)
            for step, unit in zip(steps, np.eye(x.size), strict=True)
        ]
        np.testing.assert_allclose(
            gradient, numeric, rtol=0, atol=rtol * np.max(np.abs(numeric))
        )


def test_magnitude_objective_uses_finite_differences(monkeypatch):
    calls, _ = refinement_calls(monkeypatch, doublet(), {"use_sse_mag": True})

    assert {call["jac"] for call in calls} == {"2-point"}


@pytest.mark.parametrize(
    ("options", "tolerances"),
    [
        ({}, (1e-6, 1e-8, 1e-12)),
        (DERIVATIVE_CASES["voigt"][0], (1e-6, 1e-8, 1e-12)),
        (DERIVATIVE_CASES["all_banded"][0], (0.01, 5e-5, 1e-6)),
    ],
    ids=["lorentz", "voigt", "all_banded"],
)
def test_analytic_and_finite_difference_fits_agree(monkeypatch, options, tolerances):
    freq_atol_hz, t2_atol_s, rss_rtol = tolerances
    data = doublet(n_voxels=3, noise_std=0.05)

    _, analytic = refinement_calls(monkeypatch, data, options)
    _, numeric = refinement_calls(monkeypatch, data, options, forced_jac="2-point")

    np.testing.assert_allclose(
        analytic.fit_freqs.values, numeric.fit_freqs.values, atol=freq_atol_hz
    )
    np.testing.assert_allclose(
        analytic.fit_t2s.values, numeric.fit_t2s.values, atol=t2_atol_s
    )
    assert np.all(analytic.fit_rss.values <= numeric.fit_rss.values * (1 + rss_rtol))
    if "phase_band_tolerance_deg" not in options:
        np.testing.assert_allclose(
            analytic.fit_rss.values, numeric.fit_rss.values, rtol=rss_rtol
        )
