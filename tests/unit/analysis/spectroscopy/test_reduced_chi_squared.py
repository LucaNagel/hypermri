"""Reduced chi-squared and free-parameter count of the FID fit.

Fixtures are synthetic: two Lorentzian lines with known frequencies and T2*,
plus white complex Gaussian noise of known per-quadrature sigma. For the true
model, ``fit_rss / sigma**2`` follows a chi-squared distribution with
``2 * n_samples - k`` degrees of freedom, so the reduced value has mean 1 and
standard deviation ``sqrt(2 / dof)``; tolerances are stated as multiples of the
standard error of the mean over the fitted spectra.
"""

import copy
import pickle

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    count_free_parameters,
    fitted_sample_count,
)
from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv
from hypermri.analysis.spectroscopy import fit_spectrum
from hypermri.analysis.spectroscopy.fit_diagnostics import (
    calculate_reduced_chi_squared,
)
from hypermri.analysis.spectroscopy.noise_estimation import (
    attach_acquisition_tail_noise,
)
from hypermri.analysis.spectroscopy.refitting import (
    SpectroscopyRefitEdit,
    refit_spectroscopy_voxel,
)

DWELL_S = 1 / 3000.0
N_SAMPLES = 96
SIGMA = 2.0
N_SPECTRA = 24


def _noisy_fids(n_spectra=N_SPECTRA, seed=0, dims=("x", "acquisition_idx")):
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) * DWELL_S
    truth = 100 * np.exp(2j * np.pi * 200 * t - t / 0.03) + 60 * np.exp(
        -2j * np.pi * 300 * t - t / 0.02
    )
    noise = SIGMA * (
        rng.normal(size=(n_spectra, N_SAMPLES))
        + 1j * rng.normal(size=(n_spectra, N_SAMPLES))
    )
    data = xr.DataArray(
        truth[None, :] + noise,
        dims=("x", "acquisition_idx"),
        coords={
            "x": np.arange(n_spectra),
            "acquisition_idx": np.arange(N_SAMPLES),
            "sampling_time_axis_s": ("acquisition_idx", t),
        },
        attrs={"signal_domain": "temporal", "unit": "a.u."},
    )
    return data.transpose(*dims)


def _with_tail_noise(data, source="acquisition_tail"):
    sigma = xr.DataArray(
        np.full(data.sizes["x"], SIGMA),
        dims="x",
        attrs={"noise_source": source, "noise_onset_point": 3},
    )
    return data.assign_coords(noise_sigma_quadrature=sigma)


def _params(data, **options):
    return def_fit_params(
        data,
        {
            "metabs": ["a", "b"],
            "metabs_freqs_hz": [200.0, -300.0],
            "initial_t2s_s": [0.025, 0.025],
            "metabs_freq_ranges": 20.0,
            "metabs_t2_ranges": 0.03,
            "min_t2_s": 0.005,
            "max_t2_s": 0.07,
            "npoints": 5,
            "niter": 1,
            "rep_fitting": 2,
            **options,
        },
    )


@pytest.fixture(scope="module")
def carried_fit():
    data = _with_tail_noise(_noisy_fids())
    return fit_data_pseudo_inv(data, _params(data))


def test_true_model_on_white_noise_gives_reduced_chi_squared_one(carried_fit):
    # 2 lines x (2 amplitude + 1 frequency + 1 T2*) free real parameters.
    assert (carried_fit.fit_n_free_parameters == 8).all()
    dof = 2 * N_SAMPLES - 8
    assert (carried_fit.fit_residual_degrees_of_freedom == dof).all()
    chi2 = carried_fit.fit_reduced_chi_squared
    assert chi2.dims == ("x",)
    standard_error = np.sqrt(2 / dof) / np.sqrt(N_SPECTRA)
    assert float(chi2.mean()) == pytest.approx(1.0, abs=4 * standard_error)
    assert chi2.attrs["noise_source"] == "acquisition_tail"
    assert chi2.attrs["noise_onset_point"] == 3
    assert chi2.attrs["unit"] == "dimensionless"
    assert (
        carried_fit.fit_residual_degrees_of_freedom.attrs["n_fitted_complex_samples"]
        == N_SAMPLES
    )


def test_reduced_chi_squared_is_nan_without_carried_noise():
    data = _noisy_fids(n_spectra=2)
    result = fit_data_pseudo_inv(data, _params(data))

    assert result.fit_reduced_chi_squared.isnull().all()
    assert result.fit_reduced_chi_squared.attrs["noise_source"] == "none"
    assert (result.fit_n_free_parameters == 8).all()


def test_noise_coordinate_of_unknown_origin_is_rejected():
    data = _with_tail_noise(_noisy_fids(n_spectra=2), source="supplied")
    with pytest.raises(ValueError, match="noise_sigma_quadrature"):
        fit_data_pseudo_inv(data, _params(data))


def test_reduced_chi_squared_does_not_depend_on_dimension_order(carried_fit):
    data = _with_tail_noise(_noisy_fids(dims=("acquisition_idx", "x")))
    swapped = fit_data_pseudo_inv(data, _params(data))

    np.testing.assert_allclose(
        swapped.fit_reduced_chi_squared.transpose(*carried_fit.fit_rss.dims),
        carried_fit.fit_reduced_chi_squared,
        rtol=1e-9,
    )


def test_quality_outputs_and_options_survive_snapshot_and_netcdf(carried_fit, tmp_path):
    for restored in (
        pickle.loads(pickle.dumps(carried_fit)),
        copy.deepcopy(carried_fit),
    ):
        # assert_identical also compares the attrs: the effective fit options
        # and the noise provenance of the quality variables.
        xr.testing.assert_identical(restored, carried_fit)
        assert restored.attrs["metabs_freq_ranges"] == 20.0
        assert restored.fit_reduced_chi_squared.attrs["noise_source"] == (
            "acquisition_tail"
        )

    path = tmp_path / "quality.nc"
    xr.Dataset(
        {
            name: carried_fit[name]
            for name in (
                "fit_reduced_chi_squared",
                "fit_n_free_parameters",
                "fit_residual_degrees_of_freedom",
            )
        }
    ).to_netcdf(path)
    with xr.open_dataset(path) as loaded:
        np.testing.assert_allclose(
            loaded.fit_reduced_chi_squared, carried_fit.fit_reduced_chi_squared
        )
        assert loaded.fit_reduced_chi_squared.attrs["noise_source"] == (
            "acquisition_tail"
        )


def test_fit_spectrum_reports_reduced_chi_squared_for_supplied_noise():
    data = _noisy_fids(n_spectra=4)
    result = fit_spectrum(data, _params(data), noise=SIGMA)

    expected = result.fit_rss / SIGMA**2 / result.fit_residual_degrees_of_freedom
    np.testing.assert_allclose(result.fit_reduced_chi_squared, expected, rtol=1e-12)
    assert result.fit_reduced_chi_squared.attrs["noise_source"] == "supplied"


def test_fit_spectrum_rejects_two_noise_sources():
    data = _with_tail_noise(_noisy_fids(n_spectra=2))
    with pytest.raises(ValueError, match="noise_sigma_quadrature"):
        fit_spectrum(data, _params(data), noise=SIGMA)


def test_calculate_reduced_chi_squared_requires_the_fit_quality_variables():
    with pytest.raises(KeyError, match="fit_residual_degrees_of_freedom"):
        calculate_reduced_chi_squared(
            xr.Dataset({"fit_rss": xr.DataArray([1.0], dims="x")}), 1.0
        )


def _count(active, fit_options, *, freq_range=20.0, t2_range=0.03, n=3):
    return count_free_parameters(
        np.asarray(active, dtype=bool),
        np.full(n, freq_range),
        np.full(n, t2_range),
        np.full(n, 0.005),
        np.full(n, 0.07),
        np.full(n, -500.0),
        np.full(n, 500.0),
        metabolites=["lead", "follow", "other"][:n],
        fit_options=fit_options,
    )


@pytest.mark.parametrize(
    ("active", "fit_options", "kwargs", "expected"),
    [
        # Three free lines: 3 x (2 + 1 + 1).
        ([1, 1, 1], {}, {}, 12),
        # Skipped or low-amplitude lines are inactive.
        ([1, 0, 1], {}, {}, 8),
        # Fixed frequencies and T2*: only the complex amplitudes remain.
        ([1, 1, 1], {}, {"freq_range": 0.0, "t2_range": 0.0}, 6),
        # A fixed-ratio coupling shares one T2* between leader and follower.
        ([1, 1, 1], {"t2_coupling": {"follow": ("lead", 1.0)}}, {}, 11),
        # The group keeps its T2* when only the follower is active.
        ([0, 1, 1], {"t2_coupling": {"follow": ("lead", 1.0)}}, {}, 8),
        # A banded dependent keeps its own T2*.
        ([1, 1, 1], {"t2_coupling": {"follow": ("lead", (0.5, 2.0))}}, {}, 12),
        # Two zero-tolerance phase-band lines: 2 magnitudes + one shared phase.
        (
            [1, 1, 1],
            {"phase_band_tolerance_deg": {"lead": 0.0, "follow": 0.0}},
            {},
            11,
        ),
        # A positive tolerance keeps magnitude and phase deviation.
        ([1, 1, 1], {"phase_band_tolerance_deg": {"lead": 10.0}}, {}, 12),
        # Voigt adds one shared Gaussian width.
        ([1, 1, 1], {"lineshape": "voigt", "gauss_search_width_hz": 30.0}, {}, 13),
        ([1, 1, 1], {"lineshape": "voigt", "gauss_search_width_hz": 0.0}, {}, 12),
        ([0, 0, 0], {"lineshape": "voigt", "gauss_search_width_hz": 30.0}, {}, 0),
    ],
)
def test_free_parameter_count_follows_the_documented_convention(
    active, fit_options, kwargs, expected
):
    assert _count(active, fit_options, **kwargs) == expected


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, 100),
        ({"nsamplepoints": 40}, 40),
        ({"sampling_time_axis_s": np.zeros(60), "nsamplepoints": 40}, 60),
        ({"nsamplepoints": 400}, 100),
        ({"nsamplepoints": 0}, 1),
    ],
)
def test_fitted_sample_count_uses_the_fit_axis_within_the_input(options, expected):
    assert fitted_sample_count(100, options) == expected


def _tail_sigma(values, source="acquisition_tail", **coords):
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims="x",
        coords=coords,
        attrs={"noise_source": source, "noise_onset_point": 3},
    )


def test_attached_noise_gives_the_carried_reduced_chi_squared(carried_fit):
    # Same data and sigma as the carried fixture, attached through the helper:
    # identical inputs give identical fits, so equality is exact.
    data = _noisy_fids()
    attached = attach_acquisition_tail_noise(
        data, noise=_tail_sigma(np.full(N_SPECTRA, SIGMA))
    )
    xr.testing.assert_identical(attached.drop_vars("noise_sigma_quadrature"), data)
    assert attached.noise_sigma_quadrature.attrs["noise_source"] == "acquisition_tail"
    fitted = fit_data_pseudo_inv(attached, _params(attached))
    xr.testing.assert_identical(
        fitted.fit_reduced_chi_squared, carried_fit.fit_reduced_chi_squared
    )


def test_attached_noise_follows_the_data_dimension_order():
    data = (
        _noisy_fids(n_spectra=3)
        .expand_dims(rep=2)
        .transpose("x", "acquisition_idx", "rep")
    )
    noise = xr.DataArray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dims=("rep", "x"),
        attrs={"noise_source": "acquisition_tail"},
    )
    attached = attach_acquisition_tail_noise(data, noise=noise)
    assert attached.noise_sigma_quadrature.dims == ("x", "rep")
    np.testing.assert_array_equal(attached.noise_sigma_quadrature, noise.T)


@pytest.mark.parametrize(
    ("noise", "match"),
    [
        (_tail_sigma([SIGMA] * 2, source="rf_off"), "noise_source='rf_off'"),
        (_tail_sigma([SIGMA, 0.0]), "positive"),
        (_tail_sigma([SIGMA, np.nan]), "positive"),
        (_tail_sigma([SIGMA] * 3), "3 entries along 'x'"),
        (_tail_sigma([SIGMA] * 2, x=[5, 6]), "coordinate 'x'"),
        (
            xr.DataArray(
                np.ones(N_SAMPLES),
                dims="acquisition_idx",
                attrs={"noise_source": "acquisition_tail"},
            ),
            "subset of the voxel dimensions",
        ),
        (
            xr.DataArray(
                [1.0], dims="coil", attrs={"noise_source": "acquisition_tail"}
            ),
            "subset of the voxel dimensions",
        ),
    ],
)
def test_attaching_invalid_noise_fails_with_the_reason(noise, match):
    with pytest.raises(ValueError, match=match):
        attach_acquisition_tail_noise(_noisy_fids(n_spectra=2), noise=noise)


def test_attaching_twice_or_to_wrong_data_fails():
    data = _with_tail_noise(_noisy_fids(n_spectra=2))
    with pytest.raises(ValueError, match="already carry"):
        attach_acquisition_tail_noise(data, noise=_tail_sigma([SIGMA] * 2))
    with pytest.raises(ValueError, match="spectral dimension 'time'"):
        attach_acquisition_tail_noise(
            _noisy_fids(n_spectra=2),
            noise=_tail_sigma([SIGMA] * 2),
            spectral_dim="time",
        )
    with pytest.raises(TypeError, match="DataArray"):
        attach_acquisition_tail_noise(_noisy_fids(n_spectra=2), noise=SIGMA)


def _stale(results, voxel):
    # Wrong stored quality values show that the refit recomputes them.
    stale = results.copy(deep=True)
    for name in ("fit_rss", "fit_objective", "fit_reduced_chi_squared"):
        stale[name].loc[{"x": voxel}] = 1e9
    stale["fit_n_free_parameters"].loc[{"x": voxel}] = 999
    return stale


def test_voxel_refit_recomputes_rss_and_reduced_chi_squared(carried_fit):
    data = _with_tail_noise(_noisy_fids())
    params = _params(data)
    refitted = refit_spectroscopy_voxel(
        data, params, _stale(carried_fit, 3), voxel_indices={"x": 3}
    )["fit_results"]
    full = carried_fit.isel(x=3)
    # The refit starts from the converged full fit of the same spectrum and
    # repeats its search, so it ends at the same minimum up to the search
    # tolerance; 1e-3 relative is far below the chi-squared scatter
    # sqrt(2 / dof) ~ 0.1.
    assert refitted.fit_rss.item() == pytest.approx(full.fit_rss.item(), rel=1e-3)
    assert refitted.fit_objective.item() == pytest.approx(refitted.fit_rss.item())
    assert refitted.fit_n_free_parameters.item() == 8
    assert refitted.fit_residual_degrees_of_freedom.item() == 2 * N_SAMPLES - 8
    expected = refitted.fit_rss.item() / SIGMA**2 / (2 * N_SAMPLES - 8)
    assert refitted.fit_reduced_chi_squared.item() == pytest.approx(expected, rel=1e-12)
    assert refitted.fit_reduced_chi_squared.attrs["noise_source"] == "acquisition_tail"


def test_disabled_line_leaves_the_refit_model_and_the_parameter_count(carried_fit):
    data = _with_tail_noise(_noisy_fids())
    refitted = refit_spectroscopy_voxel(
        data,
        _params(data),
        carried_fit,
        voxel_indices={"x": 3},
        edits={"b": SpectroscopyRefitEdit(enabled=False)},
    )["fit_results"]
    # One Lorentzian line with 2 amplitude + 1 frequency + 1 T2* parameters.
    assert refitted.fit_n_free_parameters.item() == 4
    assert refitted.fit_residual_degrees_of_freedom.item() == 2 * N_SAMPLES - 4
    # Line b (amplitude 60) is then left in the residual: chi2/N >> 1.
    assert refitted.fit_reduced_chi_squared.item() > 10


def test_voxel_refit_without_carried_noise_reports_nan_chi_squared(carried_fit):
    data = _noisy_fids()
    refitted = refit_spectroscopy_voxel(
        data, _params(data), carried_fit, voxel_indices={"x": 3}
    )["fit_results"]
    assert np.isfinite(refitted.fit_rss.item())
    assert np.isnan(refitted.fit_reduced_chi_squared.item())
    assert refitted.fit_reduced_chi_squared.attrs["noise_source"] == "none"


def test_voxel_refit_rejects_noise_of_another_source(carried_fit):
    data = _with_tail_noise(_noisy_fids(), source="rf_off")
    with pytest.raises(ValueError, match="noise_source='rf_off'"):
        refit_spectroscopy_voxel(
            data, _params(data), carried_fit, voxel_indices={"x": 3}
        )
