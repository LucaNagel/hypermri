"""Characterize the backward elimination of unnecessary overlapping lines.

Fixtures are synthetic FIDs: 256 samples at 4 kHz, all T2* 0.03 s, lines 'a'
at -300 Hz, 'b' at 0 Hz, 'c' at 20 Hz and 'd' at 40 Hz with areas 10, 20, 3
and 3, plus complex white noise of sigma 0.02 per quadrature. Line phases
follow ``0.3 + 2 pi f tau`` with tau 1.5 ms, so a phase band can be applied
to the same data. Absent lines fitted to noise lower the RSS by a few sigma**2
only, far below the threshold ``delta_k * ln(512)`` of at least 18.7, while
the weak line 'c' changes it by several thousand sigma**2. Fits of one
spectrum do not depend on other spectra, so results of the elimination are
compared with direct fits to rtol 1e-12.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import def_fit_params
from hypermri.analysis.spectroscopy import (
    eliminate_unnecessary_lines,
    fit_spectrum,
)
from hypermri.analysis.spectroscopy.relevance import _n_free_parameters

TIME_S = np.arange(256) / 4000.0
TAU_S = 1.5e-3
NAMES = ["a", "b", "c", "d"]
FREQS_HZ = np.array([-300.0, 0.0, 20.0, 40.0])
AREAS = np.array([10.0, 20.0, 3.0, 3.0])
NOISE_SIGMA = 0.02
DETECTION_OPTIONS = {
    "detection_sigma": NOISE_SIGMA,
    "detection_null_samples": 10,
    "detection_seed": 1,
}


def synthetic_fids(present, seed=0):
    """Return labelled noisy FIDs; row ``i`` of ``present`` switches lines on."""
    rng = np.random.default_rng(seed)
    present = np.asarray(present, dtype=float)
    amplitudes = present * AREAS * np.exp(1j * (0.3 + 2 * np.pi * FREQS_HZ * TAU_S))
    lines = np.exp(2j * np.pi * FREQS_HZ[:, None] * TIME_S - TIME_S / 0.03)
    noise = rng.standard_normal((len(present), TIME_S.size))
    noise = noise + 1j * rng.standard_normal(noise.shape)
    data = xr.DataArray(
        amplitudes @ lines + NOISE_SIGMA * noise,
        dims=["x", "acquisition_idx"],
        coords={"x": np.arange(len(present))},
    )
    data.attrs["sampling_time_axis_s"] = TIME_S
    return data


def line_params(data, names=NAMES[:3], **options):
    """Prepare parameters with starts 1 Hz away from the true lines."""
    freqs_hz = FREQS_HZ[[NAMES.index(name) for name in names]] + 1.0
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": list(names),
            "metabs_freqs_hz": list(freqs_hz),
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 8.0,
            "min_t2_s": 0.005,
            "max_t2_s": 0.1,
            **options,
        },
    )


def with_skipped(params, skipped):
    """Return a parameter copy with the named lines skipped everywhere."""
    params = params.copy(deep=True)
    skip = params["skip_fitting"]
    named = xr.DataArray(
        np.isin(params.metabolite.values, skipped),
        dims=["metabolite"],
        coords={"metabolite": params.metabolite},
    )
    params["skip_fitting"] = skip.copy(
        data=named.broadcast_like(skip).transpose(*skip.dims).values
    )
    return params


@pytest.fixture(scope="module")
def two_voxels():
    """Voxel 0 holds the weak line 'c'; voxel 1 lacks it."""
    return synthetic_fids([[1, 1, 1, 0], [1, 1, 0, 0]])


@pytest.mark.parametrize("coupling", [None, {"c": "b"}], ids=["independent", "coupled"])
def test_absent_line_is_removed_and_weak_line_kept(two_voxels, coupling):
    """Only the line missing from the data is removed, at the first step.

    Each final fit equals a direct fit with the same lines skipped. With
    'c' coupled to 'b', removing 'c' saves frequency, amplitude and phase
    only, so the threshold is 3 ln(512) instead of 4 ln(512).
    """
    params = line_params(two_voxels, t2_coupling=coupling)
    data_before, params_before = two_voxels.copy(deep=True), params.copy(deep=True)
    result = eliminate_unnecessary_lines(
        two_voxels, params, candidates=["b", "c"], noise=NOISE_SIGMA
    )
    xr.testing.assert_identical(two_voxels, data_before)
    xr.testing.assert_identical(params, params_before)
    np.testing.assert_array_equal(
        result.necessity_removed.values, [[False] * 3, [False, False, True]]
    )
    np.testing.assert_array_equal(
        result.necessity_removal_step.values, [[0, 0, 0], [0, 0, 1]]
    )
    np.testing.assert_array_equal(result.necessity_tested.values, [True, True])
    threshold = (3 if coupling else 4) * np.log(512)
    np.testing.assert_allclose(
        result.necessity_threshold.values[:, 1:], threshold, rtol=1e-12
    )
    assert np.isnan(result.necessity_threshold.values[:, 0]).all()
    assert result.necessity_delta_chi2.values[0, 2] > 1000
    assert result.necessity_delta_chi2.values[1, 2] < threshold
    assert result.fit_skipped.values[1, 2]
    assert result.fit_amps.values[1, 2] == 0

    full = fit_spectrum(two_voxels, params, show_tqdm=False)
    reduced = fit_spectrum(two_voxels, with_skipped(params, ["c"]), show_tqdm=False)
    for name in ["fit_amps", "fit_freqs", "fit_t2s", "fit_rss"]:
        np.testing.assert_allclose(
            result[name].values[0], full[name].values[0], rtol=1e-12
        )
        np.testing.assert_allclose(
            result[name].values[1], reduced[name].values[1], rtol=1e-12
        )


def test_lines_are_removed_one_per_step_and_one_stays():
    """Absent candidates go one per step; the last candidate always stays.

    Voxel 0 holds only 'a' and 'b', so 'c' and 'd' are removed in steps 1
    and 2. Voxel 1 holds only 'a', so two of the three absent candidates are
    removed and the third stays although the data do not need it either.
    """
    data = synthetic_fids([[1, 1, 0, 0], [1, 0, 0, 0]], seed=1)
    params = line_params(data, NAMES)
    result = eliminate_unnecessary_lines(
        data, params, candidates=["b", "c", "d"], noise=NOISE_SIGMA
    )
    steps = result.necessity_removal_step.values
    np.testing.assert_array_equal(steps[0, :2], [0, 0])
    np.testing.assert_array_equal(np.sort(steps[0, 2:]), [1, 2])
    np.testing.assert_array_equal(np.sort(steps[1, 1:]), [0, 1, 2])
    assert steps[1, 0] == 0
    np.testing.assert_array_equal(result.necessity_removed.values, steps > 0)
    np.testing.assert_array_equal(result.fit_skipped.values, steps > 0)


def test_unselected_and_untestable_spectra_are_not_tested(two_voxels):
    """Unselected spectra are not fitted; one fitted candidate is not tested."""
    params = line_params(two_voxels)
    selection = xr.DataArray([False, True], dims=["x"], coords={"x": [0, 1]})
    result = eliminate_unnecessary_lines(
        two_voxels,
        params,
        candidates=["b", "c"],
        noise=NOISE_SIGMA,
        selection=selection,
    )
    assert result.fit_status.values.tolist() == ["skipped", "success"]
    np.testing.assert_array_equal(result.necessity_tested.values, [False, True])
    assert not result.necessity_removed.values[0].any()
    assert result.necessity_removed.values[1, 2]

    skipped_c = with_skipped(params, ["c"])
    untested = eliminate_unnecessary_lines(
        two_voxels, skipped_c, candidates=["b", "c"], noise=NOISE_SIGMA
    )
    reference = fit_spectrum(two_voxels, skipped_c, show_tqdm=False)
    assert not untested.necessity_tested.values.any()
    assert not untested.necessity_removed.values.any()
    assert np.isnan(untested.necessity_delta_chi2.values).all()
    np.testing.assert_allclose(
        untested.fit_amps.values, reference.fit_amps.values, rtol=1e-12
    )


def test_settings_are_recorded(two_voxels):
    """Candidates, sample count and the noise map are kept with the result."""
    params = line_params(two_voxels)
    noise = xr.DataArray([0.02, 0.03], dims=["x"], coords={"x": [0, 1]})
    result = eliminate_unnecessary_lines(
        two_voxels, params, candidates=("c", "b"), noise=noise
    )
    assert result.attrs["necessity_candidates"] == ["c", "b"]
    assert result.attrs["necessity_n_samples"] == 256
    assert "ln(2 N)" in result.attrs["necessity_criterion"]
    np.testing.assert_array_equal(result.necessity_noise_sigma.values, [0.02, 0.03])
    assert result.necessity_delta_chi2.attrs["unit"] == "1"


def test_phase_band_fits_can_be_tested(two_voxels):
    """With a phase band on 'b' and 'c' the same line is removed."""
    params = line_params(
        two_voxels,
        phase_band_tolerance_deg={"b": 3.0, "c": 3.0},
        phase_band_tau_s=TAU_S,
    )
    result = eliminate_unnecessary_lines(
        two_voxels, params, candidates=["b", "c"], noise=NOISE_SIGMA
    )
    assert result.attrs["phase_band_tolerance_deg"] == {"b": 3.0, "c": 3.0}
    np.testing.assert_array_equal(
        result.necessity_removed.values, [[False] * 3, [False, False, True]]
    )


@pytest.mark.parametrize(
    ("candidates", "error", "message"),
    [
        ("bc", TypeError, "not the single string"),
        (["b"], ValueError, "at least two unique"),
        (["b", "b"], ValueError, "at least two unique"),
        (["b", "x"], ValueError, "unknown candidates \\['x'\\]"),
    ],
)
def test_invalid_candidates_are_rejected(two_voxels, candidates, error, message):
    """Candidates must name at least two different fitted lines."""
    with pytest.raises(error, match=message):
        eliminate_unnecessary_lines(
            two_voxels,
            line_params(two_voxels),
            candidates=candidates,
            noise=NOISE_SIGMA,
        )


@pytest.mark.parametrize("noise", [0.0, np.nan, -0.02])
def test_invalid_noise_is_rejected(two_voxels, noise):
    """The noise must be finite and positive before any fit runs."""
    with pytest.raises(ValueError, match="finite and positive"):
        eliminate_unnecessary_lines(
            two_voxels,
            line_params(two_voxels),
            candidates=["b", "c"],
            noise=noise,
        )


@pytest.mark.parametrize(
    ("option", "value", "extra"),
    [
        ("detection_target", "c", DETECTION_OPTIONS),
        ("use_sse_mag", True, {}),
        ("store_optimization_history", True, {}),
    ],
)
def test_unsupported_fit_options_are_rejected(two_voxels, option, value, extra):
    """Options that change what the RSS or the fit means are refused."""
    params = line_params(two_voxels, **{option: value}, **extra)
    with pytest.raises(ValueError, match=f"does not support {option}"):
        eliminate_unnecessary_lines(
            two_voxels, params, candidates=["b", "c"], noise=NOISE_SIGMA
        )


@pytest.mark.parametrize(
    ("active", "coupling", "voigt", "expected"),
    [
        ((True, True, True), None, False, 12),
        ((True, True, True), {"c": ("b", 1.0)}, False, 11),
        ((True, True, True), {"c": ("b", 1.0)}, True, 12),
        ((True, False, True), {"c": ("b", 1.0)}, False, 8),
        ((False, True, True), {"b": ("a", 0.75), "c": ("a", 0.375)}, False, 7),
        ((False, False, False), None, True, 0),
    ],
    ids=["independent", "coupled", "voigt", "lone_dependent", "new_leader", "empty"],
)
def test_parameter_count_follows_coupling(active, coupling, voigt, expected):
    """Three per line, one per own T2* after promotion, one for Voigt."""
    assert _n_free_parameters(["a", "b", "c"], active, coupling, voigt) == expected
