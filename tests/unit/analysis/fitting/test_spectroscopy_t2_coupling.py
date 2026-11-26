"""Characterize T2* coupling between lines of the FID pseudo-inverse fit.

Fixtures are synthetic single-voxel FIDs: 256 samples at 4 kHz with a doublet
at 0 and 20 Hz (areas 20 and 12) and a separated line at -300 Hz (area 10),
plus complex white noise of sigma 0.02 per quadrature. A coupled dependent is
set from its leader in every accepted step, so the ratio is checked to
floating-point precision (rtol 1e-12); a banded dependent is checked
against its band with the same relative edge tolerance. Recovery tolerances (0.003 s for T2*)
leave margin above the T2* grid resolution after the zoomed passes.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    _clip_t2_coupling_start_values,
    def_fit_params,
    fit_data_pseudo_inv,
)
from hypermri.analysis.spectroscopy import estimate_fit_covariance

FREQS_HZ = np.array([-300.0, 0.0, 20.0])
AMPLITUDES = np.array([10.0, 20.0, 12.0]) * np.exp(1j * np.array([0.3, -0.2, 0.5]))
NOISE_SIGMA = 0.02


def synthetic_fid(t2s_s, *, gauss_fwhm_hz=0.0, seed=0):
    """Return one labelled noisy FID of three lines with T2* values in seconds."""
    rng = np.random.default_rng(seed)
    t = np.arange(256) / 4000.0
    lines = AMPLITUDES[:, None] * np.exp(
        2j * np.pi * FREQS_HZ[:, None] * t - t / np.asarray(t2s_s)[:, None]
    )
    signal = lines.sum(0) * np.exp(
        -((np.pi * gauss_fwhm_hz * t) ** 2) / (4 * np.log(2))
    )
    noise = NOISE_SIGMA * (
        rng.standard_normal((1, t.size)) + 1j * rng.standard_normal((1, t.size))
    )
    data = xr.DataArray(
        signal[None] + noise, dims=["x", "acquisition_idx"], coords={"x": [0]}
    )
    data.attrs["sampling_time_axis_s"] = t
    return data


def coupled_params(data, **options):
    """Prepare parameters for the separated line 'a' and the doublet 'b', 'c'."""
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b", "c"],
            "metabs_freqs_hz": [-299.0, 1.0, 19.0],
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 8.0,
            "min_t2_s": 0.005,
            "max_t2_s": 0.1,
            **options,
        },
    )


@pytest.mark.parametrize(
    ("coupling", "message"),
    [
        (["c", "b"], "must be a dict"),
        ({"c": "x"}, "not fitted metabolites"),
        ({"c": "c"}, "cannot follow itself"),
        ({"c": ("b", 0.0)}, "must be positive"),
        ({"c": ("b", "fast")}, "t2_ratio"),
        ({"c": "b", "b": "a"}, "both leader and dependent"),
        ({"c": ("b", (0.0, 1.0))}, "0 < low <= high"),
        ({"c": ("b", (1.25, 0.75))}, "0 < low <= high"),
        ({"c": ("b", (0.75,))}, "\\(low, high\\) pair"),
    ],
)
def test_invalid_coupling_is_rejected(coupling, message):
    """Invalid shapes, names, ratios and chains fail when parameters are made."""
    data = synthetic_fid([0.03, 0.03, 0.03])
    with pytest.raises(ValueError, match=message):
        coupled_params(data, t2_coupling=coupling)


def test_coupling_is_normalized_and_kept_as_provenance():
    """A plain leader name means ratio 1.0 and the fit records the option."""
    data = synthetic_fid([0.03, 0.03, 0.03])
    params = coupled_params(data, t2_coupling={"c": "b"})
    assert params.attrs["t2_coupling"] == {"c": ("b", 1.0)}
    result = fit_data_pseudo_inv(data, params)
    assert result.attrs["t2_coupling"] == {"c": ("b", 1.0)}


def test_coupling_with_detection_target_is_rejected():
    """Detection fits keep independent T2* until coupling is supported there."""
    data = synthetic_fid([0.03, 0.03, 0.03])
    with pytest.raises(ValueError, match="detection_target"):
        coupled_params(data, t2_coupling={"c": "b"}, detection_target="c")


@pytest.mark.parametrize("ratio", [1.0, 0.5])
def test_coupled_doublet_keeps_the_ratio_and_recovers_t2(ratio):
    """The dependent equals ratio times the leader and both match the truth."""
    true_t2s_s = [0.04, 0.03, 0.03 * ratio]
    data = synthetic_fid(true_t2s_s)
    params = coupled_params(data, t2_coupling={"c": ("b", ratio)})
    fit_t2s_s = fit_data_pseudo_inv(data, params).fit_t2s.values[0]
    np.testing.assert_allclose(fit_t2s_s[2], ratio * fit_t2s_s[1], rtol=1e-12)
    np.testing.assert_allclose(fit_t2s_s, true_t2s_s, atol=0.003)


def test_coupling_holds_in_voigt_mode():
    """The Gaussian step keeps the Lorentz T2* ratio exact."""
    data = synthetic_fid([0.04, 0.03, 0.015], gauss_fwhm_hz=6.0)
    params = coupled_params(
        data,
        lineshape="voigt",
        gauss_initial_fwhm_hz=2.0,
        gauss_max_fwhm_hz=12.0,
        t2_coupling={"c": ("b", 0.5)},
    )
    fit_t2s_s = fit_data_pseudo_inv(data, params).fit_t2s.values[0]
    np.testing.assert_allclose(fit_t2s_s[2], 0.5 * fit_t2s_s[1], rtol=1e-12)


def test_skipped_leader_frees_a_lone_dependent():
    """Without its leader a single dependent is fitted as an independent line.

    The skipped leader 'a' (start 0.04 s, width 0.04 s) reports its start
    T2*. 'c' must give exactly the uncoupled fit that starts at 0.5 * 0.04 s
    with half of the leader's search width, instead of staying frozen at its
    coupled start. The unmodelled line of 'a' biases both fits equally.
    """
    data = synthetic_fid([0.04, 0.03, 0.015])
    common = {"skip_fitting": [True, False, False], "metabs_t2_ranges": 0.04}
    coupled = coupled_params(
        data, initial_t2s_s=[0.04, 0.03, 0.03], t2_coupling={"c": ("a", 0.5)}, **common
    )
    independent = coupled_params(
        data,
        initial_t2s_s=[0.04, 0.03, 0.02],
        **{**common, "metabs_t2_ranges": [0.04, 0.04, 0.02]},
    )
    fit_t2s_s = fit_data_pseudo_inv(data, coupled).fit_t2s.values[0]
    reference_t2s_s = fit_data_pseudo_inv(data, independent).fit_t2s.values[0]
    assert fit_t2s_s[0] == pytest.approx(0.04)
    assert abs(fit_t2s_s[2] - 0.02) > 0.005
    np.testing.assert_allclose(fit_t2s_s, reference_t2s_s, rtol=1e-12)


@pytest.mark.parametrize(
    "exclusion",
    [{"skip_fitting": [True, False, False]}, {"snr_threshold": 10.5}],
    ids=["skipped", "amplitude_filtered"],
)
def test_first_dependent_takes_over_a_missing_leader(exclusion):
    """'b' leads 'c' at ratio 0.375 / 0.75 when leader 'a' is not fitted.

    With snr_threshold 10.5 only 'a' (area 10) is filtered after the first
    pass. Both remaining lines are fitted with the original ratio between
    them and recover the true T2* values 0.03 s and 0.015 s.
    """
    data = synthetic_fid([0.04, 0.03, 0.015])
    params = coupled_params(
        data,
        initial_t2s_s=[0.03, 0.03, 0.03],
        t2_coupling={"b": ("a", 0.75), "c": ("a", 0.375)},
        **exclusion,
    )
    result = fit_data_pseudo_inv(data, params)
    fit_t2s_s = result.fit_t2s.values[0]
    np.testing.assert_allclose(fit_t2s_s[2], 0.5 * fit_t2s_s[1], rtol=1e-12)
    np.testing.assert_allclose(fit_t2s_s[1:], [0.03, 0.015], atol=0.003)
    if "snr_threshold" in exclusion:
        np.testing.assert_array_equal(
            result.fit_low_snr_refitted.values[0], [True, False, False]
        )


@pytest.mark.parametrize(
    ("band", "normalized"),
    [((0.75, 1.25), ("b", (0.75, 1.25))), ([0.5, 0.5], ("b", 0.5))],
)
def test_ratio_band_is_normalized(band, normalized):
    """A (low, high) band is kept as floats; low == high is a fixed ratio."""
    data = synthetic_fid([0.03, 0.03, 0.03])
    params = coupled_params(data, t2_coupling={"c": ("b", band)})
    assert params.attrs["t2_coupling"] == {"c": normalized}


def test_degenerate_band_fits_like_the_fixed_ratio():
    """(0.5, 0.5) gives exactly the fit of the fixed ratio 0.5."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    banded = coupled_params(data, t2_coupling={"c": ("b", (0.5, 0.5))})
    fixed = coupled_params(data, t2_coupling={"c": ("b", 0.5)})
    np.testing.assert_array_equal(
        fit_data_pseudo_inv(data, banded).fit_t2s.values,
        fit_data_pseudo_inv(data, fixed).fit_t2s.values,
    )


def test_band_containing_the_truth_recovers_both_lines():
    """With the true ratio 0.9 inside (0.75, 1.25) both T2* are recovered."""
    true_t2s_s = [0.04, 0.03, 0.027]
    data = synthetic_fid(true_t2s_s)
    params = coupled_params(data, t2_coupling={"c": ("b", (0.75, 1.25))})
    fit_t2s_s = fit_data_pseudo_inv(data, params).fit_t2s.values[0]
    assert 0.75 <= fit_t2s_s[2] / fit_t2s_s[1] <= 1.25
    np.testing.assert_allclose(fit_t2s_s, true_t2s_s, atol=0.003)


@pytest.mark.parametrize("lineshape", ["lorentz", "voigt"])
def test_band_holds_when_the_truth_lies_outside(lineshape):
    """The true ratio 0.5 is below the band, so the fit stays inside it.

    Without the band, 'c' would reach 0.015 s. The banded fit must keep the
    ratio in [0.75, 1.25] (to the 1e-12 edge tolerance) and end near the lower
    edge, where the data pull it. In Voigt mode the Gaussian step must not
    leave the band either.
    """
    voigt = {
        "lineshape": "voigt",
        "gauss_initial_fwhm_hz": 2.0,
        "gauss_max_fwhm_hz": 12.0,
    }
    data = synthetic_fid(
        [0.04, 0.03, 0.015], gauss_fwhm_hz=6.0 if lineshape == "voigt" else 0.0
    )
    params = coupled_params(
        data,
        t2_coupling={"c": ("b", (0.75, 1.25))},
        **(voigt if lineshape == "voigt" else {}),
    )
    fit_t2s_s = fit_data_pseudo_inv(data, params).fit_t2s.values[0]
    ratio = fit_t2s_s[2] / fit_t2s_s[1]
    assert 0.75 * (1 - 1e-12) <= ratio <= 1.25 * (1 + 1e-12)
    assert ratio < 0.85


def test_band_start_outside_the_band_is_rejected():
    """A start ratio 0.01 / 0.03 outside (0.75, 1.25) fails before fitting."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    params = coupled_params(
        data,
        initial_t2s_s=[0.03, 0.03, 0.01],
        t2_coupling={"c": ("b", (0.75, 1.25))},
    )
    with pytest.raises(ValueError, match="leaves the ratio band"):
        fit_data_pseudo_inv(data, params)


def test_banded_line_follows_a_fixed_new_leader():
    """Without 'a', 'c' keeps its band (0.3, 0.45) / 0.75 relative to 'b'."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    params = coupled_params(
        data,
        initial_t2s_s=[0.04, 0.03, 0.015],
        skip_fitting=[True, False, False],
        t2_coupling={"b": ("a", 0.75), "c": ("a", (0.3, 0.45))},
    )
    fit_t2s_s = fit_data_pseudo_inv(data, params).fit_t2s.values[0]
    assert 0.4 * (1 - 1e-12) <= fit_t2s_s[2] / fit_t2s_s[1] <= 0.6 * (1 + 1e-12)
    np.testing.assert_allclose(fit_t2s_s[1:], [0.03, 0.015], atol=0.003)


def test_banded_line_without_fixed_partner_becomes_independent():
    """A skipped leader with only a banded dependent leaves 'c' uncoupled."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    common = {"skip_fitting": [True, False, False], "initial_t2s_s": 0.03}
    banded = coupled_params(data, t2_coupling={"c": ("a", (0.5, 1.0))}, **common)
    independent = coupled_params(data, **common)
    np.testing.assert_array_equal(
        fit_data_pseudo_inv(data, banded).fit_t2s.values,
        fit_data_pseudo_inv(data, independent).fit_t2s.values,
    )


def test_coupled_start_outside_dependent_bounds_is_rejected():
    """Fitting stops before work when ratio times the leader start is invalid."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    params = coupled_params(data, t2_coupling={"c": ("b", 10.0)})
    with pytest.raises(ValueError, match="leaves \\[min_t2_s, max_t2_s\\] of 'c'"):
        fit_data_pseudo_inv(data, params)


def test_local_covariance_rejects_coupled_fits():
    """The Jacobian treats every T2* as free, so coupled fits are refused."""
    data = synthetic_fid([0.04, 0.03, 0.03])
    params = coupled_params(data, t2_coupling={"c": "b"})
    result = fit_data_pseudo_inv(data, params)
    with pytest.raises(ValueError, match="t2_coupling"):
        estimate_fit_covariance(result, params, noise=NOISE_SIGMA)


def two_voxel_fid():
    """Return the three-line FID twice, as voxels x = 0 and x = 1."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    return xr.concat([data, data.assign_coords(x=[1])], "x")


def test_clipping_moves_only_inadmissible_band_starts_to_the_band():
    """Voxel 0 starts c below the band, voxel 1 inside: only voxel 0 changes."""
    data = two_voxel_fid()
    params = coupled_params(
        data,
        t2_coupling={"c": ("b", (0.75, 1.25))},
        t2_coupling_clip_start=True,
    )
    starts = params["initial_t2s_s"].copy(deep=True)
    starts.loc[{"metabolite": "c"}] = [0.01, 0.033]
    params = params.assign(initial_t2s_s=starts)
    clipped = _clip_t2_coupling_start_values(params)["initial_t2s_s"]
    np.testing.assert_allclose(clipped.sel(metabolite="c"), [0.0225, 0.033])
    np.testing.assert_array_equal(
        clipped.sel(metabolite=["a", "b"]),
        params.initial_t2s_s.sel(metabolite=["a", "b"]),
    )
    assert params["initial_t2s_s"].sel(metabolite="c", x=0) == 0.01


def test_clipped_band_start_fits_like_the_explicit_start():
    """The clipped start 0.75 * 0.03 s gives the same fit as setting it directly."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    coupling = {"c": ("b", (0.75, 1.25))}
    clipped = coupled_params(
        data,
        initial_t2s_s=[0.03, 0.03, 0.01],
        t2_coupling=coupling,
        t2_coupling_clip_start=True,
    )
    explicit = coupled_params(
        data, initial_t2s_s=[0.03, 0.03, 0.0225], t2_coupling=coupling
    )
    result = fit_data_pseudo_inv(data, clipped)
    np.testing.assert_allclose(
        result.fit_t2s.values, fit_data_pseudo_inv(data, explicit).fit_t2s.values
    )
    assert result.attrs["t2_coupling_clip_start"] is True


def test_clipping_moves_the_leader_start_for_fixed_ratios():
    """Ratio 10 and max_t2_s 0.1 s of 'c' allow a leader start of at most 0.01 s."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    params = coupled_params(
        data, t2_coupling={"c": ("b", 10.0)}, t2_coupling_clip_start=True
    )
    clipped = _clip_t2_coupling_start_values(params)["initial_t2s_s"]
    leader_start = float(clipped.sel(metabolite="b", x=0))
    assert leader_start == pytest.approx(0.01, rel=1e-12)
    assert 10.0 * leader_start <= 0.1
    fit_data_pseudo_inv(data, params)


def test_clipping_cannot_repair_starts_without_an_admissible_value():
    """Ratio 100 needs a leader start <= 0.001 s, below its min_t2_s 0.005 s."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    params = coupled_params(
        data, t2_coupling={"c": ("b", 100.0)}, t2_coupling_clip_start=True
    )
    with pytest.raises(ValueError, match="leaves \\[min_t2_s, max_t2_s\\] of 'c'"):
        fit_data_pseudo_inv(data, params)


def test_clip_start_option_must_be_boolean():
    """Strings such as 'yes' are rejected instead of being read as true."""
    data = synthetic_fid([0.04, 0.03, 0.015])
    with pytest.raises(ValueError, match="t2_coupling_clip_start must be boolean"):
        coupled_params(data, t2_coupling={"c": "b"}, t2_coupling_clip_start="yes")
