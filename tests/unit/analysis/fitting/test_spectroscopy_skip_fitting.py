"""Check voxel-specific exclusions against explicit fits of retained metabolites."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy import fid_model as model
from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
    reconstruct_fitted_signal,
)


@pytest.fixture(scope="module")
def masked_problem():
    """Build three 64-sample voxels and two independent unmasked subset fits.

    The skipped peak lies only 2 Hz from an active peak, below the 5 Hz minimum
    separation. Distinct parameter maps and non-contiguous active indices expose
    misalignment. An active weak peak also exercises amplitude-threshold refits.
    """
    metabolites = ["a", "skipped", "b", "c"]
    voxels = ["partial", "smaller", "all"]
    time_s = np.arange(64) * 0.0004
    frequencies = np.array([-70.0, -68.0, 340.0, 857.0])
    t2_s = np.array([0.025, 0.028, 0.018, 0.035])
    masks = np.array(
        [[False, True, False, False], [True, True, False, False], [True] * 4]
    )
    basis = np.exp(2j * np.pi * time_s[:, None] * frequencies - time_s[:, None] / t2_s)
    amplitudes = np.array([1.0 + 0.3j, 0.0, 0.02 - 0.01j, 0.4 + 0.2j])
    signals = basis @ (amplitudes * ~masks).T
    signals[:, 2] = signals[:, 0]  # The all-masked voxel still has measured signal.
    rng = np.random.default_rng(20260904)
    signals += 0.001 * (
        rng.normal(size=signals.shape) + 1j * rng.normal(size=signals.shape)
    )
    data = xr.DataArray(
        signals,
        dims=("acquisition_idx", "voxel"),
        coords={"acquisition_idx": time_s, "voxel": voxels},
        attrs={"signal_domain": "temporal", "simulation": "masked_components"},
    )
    params = def_fit_params(
        data,
        {
            "metabs": metabolites,
            "metabs_freqs_hz": frequencies,
            "initial_t2s_s": t2_s,
            "metabs_freq_ranges": [12.0, 14.0, 20.0, 18.0],
            "metabs_t2_ranges": [0.01, 0.02, 0.008, 0.016],
            "min_t2_s": [0.005, 0.006, 0.007, 0.008],
            "max_t2_s": [0.06, 0.07, 0.08, 0.09],
            "snr_threshold": 0.1,
            "npoints": 5,
            "niter": 1,
            "rep_fitting": 2,
        },
    )
    # Deliberately use a different dimension order from the other parameter maps.
    params["skip_fitting"] = xr.DataArray(
        masks,
        dims=("voxel", "metabolite"),
        coords={"voxel": voxels, "metabolite": metabolites},
    )
    references = {}
    for voxel, mask in zip(voxels[:2], masks[:2], strict=True):
        retained = np.asarray(metabolites)[~mask].tolist()
        subset_params = params.sel(voxel=voxel, metabolite=retained).copy(deep=True)
        subset_params.attrs["metabs"] = retained
        references[voxel] = fit_data_pseudo_inv(data.sel(voxel=voxel), subset_params)
    return data, params, references


@pytest.mark.parametrize(
    ("n_jobs", "show_tqdm"),
    [(1, False), (1, True), (2, False)],
    ids=["serial", "serial-progress", "parallel-loky"],
)
def test_masked_voxels_match_explicit_subset_fits(masked_problem, n_jobs, show_tqdm):
    """Keep active outputs, exclusion flags and labels consistent in each backend."""
    data, params, references = masked_problem
    original_data, original_params = data.copy(deep=True), params.copy(deep=True)
    result = fit_data_pseudo_inv(
        data, params, n_jobs=n_jobs, show_tqdm=show_tqdm, batch_size=1
    )
    assert result.fit_amps.dims == ("voxel", "metabolite")
    xr.testing.assert_identical(result.metabolite, params.metabolite)
    assert result.attrs["simulation"] == "masked_components"
    for voxel in data.voxel.values:
        mask = params.skip_fitting.sel(voxel=voxel)
        skipped_names = params.metabolite.values[mask.values]
        skipped = result.sel(voxel=voxel, metabolite=skipped_names)
        initial = params.sel(voxel=voxel, metabolite=skipped_names)
        assert np.all(skipped.fit_amps.values == 0)
        np.testing.assert_array_equal(skipped.fit_freqs, initial.initial_freqs_hz)
        np.testing.assert_array_equal(skipped.fit_t2s, initial.initial_t2s_s)
        assert np.all(np.isnan(skipped.fit_r2_freq.values))
        assert np.all(np.isnan(skipped.fit_r2_t2.values))
        assert not np.any(skipped.fit_low_snr_refitted.values)
        if voxel in references:
            reference = references[voxel]
            retained = result.sel(voxel=voxel, metabolite=reference.metabolite)
            # Component indices count the metabolites of each fit, which differ
            # between the masked and the subset fit.
            # The joint refinement stops at a continuous minimum, where the
            # parameters are determined only to about the square root of the
            # objective's rounding error; worker processes may round the
            # linear algebra differently (measured: 4e-10 relative in T2*).
            for name in set(reference.data_vars) - {"fit_component_index"}:
                xr.testing.assert_allclose(
                    retained[name], reference[name], rtol=1e-8, atol=1e-12
                )

    assert bool(result.fit_low_snr_refitted.sel(voxel="partial", metabolite="b"))
    assert bool(result.fit_low_snr_refitted.sel(voxel="smaller", metabolite="b"))
    fitted = reconstruct_fitted_signal(result, data)
    assert np.all(fitted.sel(voxel="all").values == 0)
    xr.testing.assert_identical(data, original_data)
    xr.testing.assert_identical(params, original_params)


def test_scalar_skip_all_bypasses_numerical_work(masked_problem, monkeypatch):
    """A scalar True excludes every peak, even when the measured signal is nonzero."""

    def unexpected_score(*args, **kwargs):
        """Fail if a fully masked spectrum reaches amplitude scoring."""
        pytest.fail("Fully masked spectra must not be scored or searched.")

    monkeypatch.setattr(model, "_score_fit_parameters", unexpected_score)
    data = masked_problem[0].sel(voxel="all")
    params = def_fit_params(
        data, {"metabs": ["a"], "metabs_freqs_hz": [-70.0], "skip_fitting": True}
    )
    result = fit_data_pseudo_inv(data, params)
    assert np.all(result.fit_amps.values == 0)
    np.testing.assert_array_equal(result.fit_freqs, params.initial_freqs_hz)
    np.testing.assert_array_equal(result.fit_t2s, params.initial_t2s_s)
    assert np.all(np.isnan(result.fit_r2_freq.values))
    assert np.all(np.isnan(result.fit_r2_t2.values))
    assert not np.any(result.fit_low_snr_refitted.values)


def test_edited_mask_rejects_nan_before_excluding_components(masked_problem):
    """Reject missing values introduced after parameter preparation."""
    data, params, _ = masked_problem
    params = params.copy(deep=True)
    params["skip_fitting"] = xr.full_like(params.skip_fitting, np.nan, dtype=float)
    with pytest.raises(
        ValueError, match="skip_fitting must contain finite real values"
    ):
        fit_data_pseudo_inv(data, params)
