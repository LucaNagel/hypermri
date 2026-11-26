"""Characterize the optional Voigt lineshape of the FID pseudo-inverse fit.

Fixtures are synthetic single-voxel FIDs: 256 samples at 4 kHz with three
separated lines (-300, 0 and 250 Hz; areas 10, 30 and 3) and complex white
noise of sigma 0.02 per quadrature. Tolerances were set with margin from four
noise seeds, whose largest deviations were 0.1 Hz for the Gaussian FWHM
(grid resolution after 11 zoomed passes), 0.06 Hz for frequencies, 0.12 Hz
for Lorentz FWHM and 0.3 % for amplitude magnitudes.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _require_same_lineshape,
    final_fit_errors,
)
from hypermri.analysis.fitting.spectroscopy import (
    add_metab_to_fit_params,
    def_fit_params,
    fit_data_pseudo_inv,
    initialise_fit_params_from_results,
    mask_fit_dataset_where_input_nonfinite,
    reconstruct_fitted_signal,
)

FREQS_HZ = np.array([-300.0, 0.0, 250.0])
AMPLITUDES = np.array([10.0, 30.0, 3.0]) * np.exp(1j * np.array([0.3, -0.2, 0.5]))
START_OFFSETS_HZ = np.array([2.0, -1.5, 1.0])
VOIGT_VARIABLES = {
    "fit_gauss_fwhm_hz",
    "fit_gauss_fwhm_status",
    "fit_lorentz_fwhm_hz",
    "initial_gauss_fwhm_hz",
}


def synthetic_fid(gauss_fwhm_hz, t2s_s, *, seed=0, n_voxels=1):
    """Return labelled noisy FIDs of three lines with one Gaussian FWHM in Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(256) / 4000.0
    lines = AMPLITUDES[:, None] * np.exp(
        2j * np.pi * FREQS_HZ[:, None] * t - t / np.asarray(t2s_s)[:, None]
    )
    signal = lines.sum(0) * np.exp(
        -((np.pi * gauss_fwhm_hz * t) ** 2) / (4 * np.log(2))
    )
    noise = 0.02 * (
        rng.standard_normal((n_voxels, t.size))
        + 1j * rng.standard_normal((n_voxels, t.size))
    )
    data = xr.DataArray(
        signal[None] + noise,
        dims=["x", "acquisition_idx"],
        coords={"x": np.arange(n_voxels)},
    )
    data.attrs["sampling_time_axis_s"] = t
    return data


def voigt_params(data, **options):
    """Prepare fit parameters for the three synthetic lines."""
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b", "c"],
            "metabs_freqs_hz": list(FREQS_HZ + START_OFFSETS_HZ),
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 10.0,
            **options,
        },
    )


@pytest.fixture(scope="module")
def voigt_fit():
    """Return data, parameters and results of one Voigt fit with G = 14 Hz."""
    data = synthetic_fid(14.0, [0.05, 0.04, 0.06])
    params = voigt_params(data, lineshape="voigt")
    return data, params, fit_data_pseudo_inv(data, params)


def test_lorentz_default_is_unchanged_by_explicit_lorentz():
    """Omitting lineshape and passing 'lorentz' give identical Lorentz fits."""
    data = synthetic_fid(0.0, [0.02, 0.015, 0.03])
    default = fit_data_pseudo_inv(data, voigt_params(data))
    explicit = fit_data_pseudo_inv(data, voigt_params(data, lineshape="lorentz"))
    assert not VOIGT_VARIABLES & set(default.variables)
    assert default.attrs["lineshape"] == "lorentz"
    assert explicit.attrs["lineshape"] == "lorentz"
    xr.testing.assert_identical(
        default.drop_attrs(deep=False), explicit.drop_attrs(deep=False)
    )


@pytest.mark.parametrize(
    ("gauss_fwhm_hz", "t2s_s"),
    [(14.0, [0.05, 0.04, 0.06]), (25.0, [0.5, 0.5, 0.5])],
    ids=["mixed", "gauss-dominated"],
)
def test_voigt_recovers_shared_gaussian_width(gauss_fwhm_hz, t2s_s):
    """Recover G, frequencies, areas and Lorentz widths of Voigt lines."""
    data = synthetic_fid(gauss_fwhm_hz, t2s_s)
    result = fit_data_pseudo_inv(data, voigt_params(data, lineshape="voigt"))
    assert result.fit_gauss_fwhm_status.item(0) == "estimated"
    assert result.fit_gauss_fwhm_hz.attrs["unit"] == "Hz"
    np.testing.assert_allclose(result.fit_gauss_fwhm_hz.values, gauss_fwhm_hz, atol=0.5)
    np.testing.assert_allclose(result.fit_freqs.values[0], FREQS_HZ, atol=0.2)
    np.testing.assert_allclose(
        np.abs(result.fit_amps.values[0]), np.abs(AMPLITUDES), rtol=0.01
    )
    lorentz_fwhm_hz = result.fit_lorentz_fwhm_hz.values[0]
    np.testing.assert_allclose(
        lorentz_fwhm_hz, 1 / (np.pi * np.asarray(t2s_s)), atol=0.3
    )
    np.testing.assert_allclose(
        lorentz_fwhm_hz, 1 / (np.pi * result.fit_t2s.values[0]), rtol=1e-12
    )
    expected_total = 0.5346 * lorentz_fwhm_hz + np.sqrt(
        0.2166 * lorentz_fwhm_hz**2 + result.fit_gauss_fwhm_hz.values[0] ** 2
    )
    np.testing.assert_allclose(result.fit_fwhm_hz.values[0], expected_total, rtol=1e-12)


def test_voigt_on_lorentz_lines_keeps_small_gaussian_width():
    """Pure Lorentz data give a small G and a residual at the noise level.

    The reference is the RSS of the true Lorentz lines. With the default 11
    passes neither lineshape converges exactly; over four seeds both fits
    stayed within 1 % of that reference.
    """
    t2s_s = np.array([0.02, 0.015, 0.03])
    data = synthetic_fid(0.0, t2s_s)
    voigt = fit_data_pseudo_inv(data, voigt_params(data, lineshape="voigt"))
    true_rss, _ = final_fit_errors(
        data.values[0],
        AMPLITUDES,
        FREQS_HZ,
        t2s_s,
        fit_options=dict(voigt.attrs),
        gauss_fwhm_hz=0.0,
    )
    assert voigt.fit_gauss_fwhm_hz.item(0) < 2.0
    assert voigt.fit_rss.item(0) <= 1.02 * true_rss


def test_fixed_gaussian_width_is_kept():
    """A zero Gaussian search width keeps G exactly at its start value."""
    data = synthetic_fid(14.0, [0.05, 0.04, 0.06])
    params = voigt_params(
        data, lineshape="voigt", gauss_initial_fwhm_hz=12.5, gauss_search_width_hz=0.0
    )
    result = fit_data_pseudo_inv(data, params)
    assert result.fit_gauss_fwhm_hz.item(0) == 12.5
    assert result.fit_gauss_fwhm_status.item(0) == "fixed"


def test_gaussian_width_reports_upper_bound():
    """A true width above gauss_max_fwhm_hz stops at the bound and says so."""
    data = synthetic_fid(14.0, [0.05, 0.04, 0.06])
    result = fit_data_pseudo_inv(
        data, voigt_params(data, lineshape="voigt", gauss_max_fwhm_hz=8.0)
    )
    np.testing.assert_allclose(result.fit_gauss_fwhm_hz.values, 8.0, atol=1e-9)
    assert result.fit_gauss_fwhm_status.item(0) == "at_upper_bound"


def test_voigt_parameters_have_start_map_and_long_t2_bound(voigt_fit):
    """Voigt preparation adds the G start map and a 1 s Lorentz T2 bound."""
    _, params, _ = voigt_fit
    assert params.attrs["lineshape"] == "voigt"
    np.testing.assert_array_equal(params.initial_gauss_fwhm_hz.values, 0.0)
    np.testing.assert_array_equal(params.max_t2_s.values, 1.0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"lineshape": "gauss"}, "lineshape must be one of"),
        ({"gauss_max_fwhm_hz": 30.0}, "require lineshape='voigt'"),
        ({"lineshape": "voigt", "gauss_width_hz": 1.0}, "Unknown Gaussian"),
        (
            {"lineshape": "voigt", "gauss_initial_fwhm_hz": 50.0},
            "gauss_initial_fwhm_hz",
        ),
        (
            {"lineshape": "voigt", "gauss_search_width_hz": -1.0},
            "gauss_search_width_hz",
        ),
        (
            {"lineshape": "voigt", "initial_gauss_fwhm_hz": 5.0},
            "per-spectrum start map",
        ),
        (
            {"lineshape": "voigt", "detection_target": "c"},
            "does not support detection_target",
        ),
        (
            {"lineshape": "voigt", "store_optimization_history": True},
            "store_optimization_history",
        ),
    ],
)
def test_invalid_lineshape_options_fail_during_preparation(options, message):
    """Reject invalid or unsupported lineshape settings before fitting."""
    data = synthetic_fid(0.0, [0.02, 0.015, 0.03])
    with pytest.raises(ValueError, match=message):
        voigt_params(data, **options)


def test_voigt_reconstruction_matches_fit_residual(voigt_fit):
    """Both reconstruction paths include G and reproduce the fitted RSS."""
    data, _, result = voigt_fit
    total = reconstruct_fitted_signal(result, data)
    rss = (np.abs(data - total) ** 2).sum("acquisition_idx")
    np.testing.assert_allclose(rss.values, result.fit_rss.values, rtol=1e-9)

    components = result.hp.reconstruct().sum("metabolite")
    np.testing.assert_allclose(
        components.transpose(*total.dims).values, total.values, rtol=1e-12, atol=1e-12
    )


def test_voigt_reconstruction_requires_gaussian_width(voigt_fit):
    """Dropping the G map of a Voigt fit fails instead of reconstructing Lorentz."""
    data, _, result = voigt_fit
    stripped = result.drop_vars(["fit_gauss_fwhm_hz", "fit_gauss_fwhm_status"])
    with pytest.raises(ValueError, match="fit_gauss_fwhm_hz"):
        stripped.hp.reconstruct()
    with pytest.raises(ValueError, match="fit_gauss_fwhm_hz"):
        reconstruct_fitted_signal(stripped, data)


def test_serial_and_parallel_voigt_fits_agree():
    """Joblib workers return the same Voigt results as the serial fit."""
    data = synthetic_fid(14.0, [0.05, 0.04, 0.06], n_voxels=3)
    params = voigt_params(data, lineshape="voigt")
    serial = fit_data_pseudo_inv(data, params)
    parallel = fit_data_pseudo_inv(data, params, n_jobs=2, batch_size=1)
    xr.testing.assert_allclose(parallel, serial, rtol=1e-10, atol=1e-12)


def test_seeding_from_voigt_results_carries_gaussian_width(voigt_fit):
    """Seeded and extended Voigt parameters start from the fitted G."""
    _, params, result = voigt_fit
    seeded = initialise_fit_params_from_results(params, result)
    xr.testing.assert_equal(
        seeded.initial_gauss_fwhm_hz,
        result.fit_gauss_fwhm_hz.rename(seeded.initial_gauss_fwhm_hz.name),
    )
    extended = add_metab_to_fit_params(
        new_freq=120.0, new_metab_name="d", fit_params=params, fit_results=result
    )
    assert "d" in extended.metabolite.values
    np.testing.assert_allclose(
        extended.initial_gauss_fwhm_hz.values, result.fit_gauss_fwhm_hz.values
    )


def test_mixed_lineshapes_cannot_seed_each_other(voigt_fit):
    """Lorentz parameters cannot be seeded from Voigt results."""
    data, _, result = voigt_fit
    lorentz_params = voigt_params(data)
    with pytest.raises(ValueError, match="same lineshape, got 'lorentz' and 'voigt'"):
        initialise_fit_params_from_results(lorentz_params, result)
    with pytest.raises(ValueError, match="same lineshape"):
        _require_same_lineshape("check", lorentz_params, result)
    assert _require_same_lineshape("check", *voigt_fit[1:]) is True


@pytest.mark.parametrize("lineshape", ["lorentz", "voigt"])
def test_nan_voxels_get_nan_parameters_and_seed_a_valid_refit(lineshape):
    """Voxels with NaN samples report NaN results; seeding keeps their starts.

    Voxel 1 has one complex sample with a NaN imaginary part, voxel 2 is all
    NaN. Their fitted values must not echo the start values, and seeding the
    next fit from these results must keep the previous starts there, so the
    refit runs (a NaN Gaussian start would otherwise abort a Voigt refit).
    """
    data = synthetic_fid(14.0, [0.05, 0.04, 0.06], n_voxels=3)
    data[1, 5] = complex(1.0, np.nan)
    data[2, :] = np.nan
    params = voigt_params(data, lineshape=lineshape)
    result = fit_data_pseudo_inv(data, params, n_jobs=2, batch_size=1)

    bad = {"x": [1, 2]}
    assert result.fit_status.values.tolist() == ["success"] + 2 * ["invalid_input"]
    for name in (
        "fit_amps",
        "fit_freqs",
        "fit_t2s",
        "fit_fwhm_hz",
        "fit_magnitude_fwhm_hz",
    ):
        assert np.isnan(result[name].sel(bad)).all(), name
        assert np.isfinite(result[name].sel(x=0)).all(), name
    if lineshape == "voigt":
        assert np.isnan(result.fit_gauss_fwhm_hz.sel(bad)).all()

    seeded = initialise_fit_params_from_results(params, result)
    for name in ("initial_freqs_hz", "initial_t2s_s"):
        xr.testing.assert_equal(seeded[name].sel(bad), params[name].sel(bad))
    np.testing.assert_allclose(
        seeded.initial_freqs_hz.sel(x=0), result.fit_freqs.sel(x=0)
    )
    refit = fit_data_pseudo_inv(data, seeded)
    assert np.isnan(refit.fit_freqs.sel(bad)).all()


@pytest.mark.parametrize("lineshape", ["lorentz", "voigt"])
def test_masking_sets_float_values_of_nan_voxels_to_nan(lineshape):
    """Parameters and results become NaN exactly where the input has NaN.

    Voxel 1 has one sample with a NaN imaginary part. All float and complex
    variables (start values, ranges, bounds, amplitudes, diagnostics) become
    NaN there; voxels 0 and 2, non-float variables, dims, coords and attrs are
    unchanged. A fit with the masked parameters still runs (a NaN Gaussian
    start is accepted for a voxel that is not fitted).
    """
    data = synthetic_fid(14.0, [0.05, 0.04, 0.06], n_voxels=3)
    data[1, 5] = complex(1.0, np.nan)
    params = voigt_params(data, lineshape=lineshape)
    result = fit_data_pseudo_inv(data, params)

    for original in (params, result):
        masked = mask_fit_dataset_where_input_nonfinite(original, data)
        assert masked.attrs.keys() == original.attrs.keys()
        for name, variable in original.data_vars.items():
            assert masked[name].dims == variable.dims, name
            assert masked[name].dtype == variable.dtype, name
            xr.testing.assert_identical(
                masked[name].sel(x=[0, 2]), variable.sel(x=[0, 2])
            )
            if variable.dtype.kind in "fc":
                values = masked[name].sel(x=1).values
                assert np.isnan(values.real).all(), name
                if variable.dtype.kind == "c":
                    assert np.isnan(values.imag).all(), name
            else:
                xr.testing.assert_identical(masked[name], variable)

    masked_params = mask_fit_dataset_where_input_nonfinite(params, data)
    # def_fit_params masks by default, so the masking is idempotent here.
    assert np.isnan(params.initial_freqs_hz.sel(x=1)).all()
    assert np.isfinite(params.initial_freqs_hz.sel(x=[0, 2])).all()
    assert "initial_freqs_hz" in masked_params and "skip_fitting" in masked_params
    if lineshape == "voigt":
        assert np.isnan(masked_params.initial_gauss_fwhm_hz.sel(x=1))
    refit = fit_data_pseudo_inv(data, masked_params)
    assert refit.fit_status.values.tolist() == ["success", "invalid_input", "success"]
    np.testing.assert_allclose(
        refit.fit_freqs.sel(x=[0, 2]), result.fit_freqs.sel(x=[0, 2])
    )


def test_masking_matches_labels_and_shared_dimensions():
    """Masking follows coordinate labels and masks shared values only if all fail.

    The dataset holds a subset of the voxels in another order and a variable
    without the repetition dimension, which is NaN only when every repetition
    of that voxel is invalid.
    """
    values = np.ones((3, 2, 4), dtype=complex)
    values[1, 0, 2] = np.nan
    values[2, :, 0] = complex(np.inf, 0.0)
    data = xr.DataArray(
        values,
        dims=("x", "repetition", "acquisition_idx"),
        coords={"x": [10, 11, 12], "repetition": [0, 1]},
    )
    dataset = xr.Dataset(
        {
            "per_repetition": (("repetition", "x"), np.ones((2, 2))),
            "shared": (("x",), np.ones(2)),
            "global": ((), 1.0),
            "status": (("x",), np.array(["ok", "ok"])),
        },
        coords={"x": [12, 11], "repetition": [0, 1]},
    )

    masked = mask_fit_dataset_where_input_nonfinite(dataset, data)

    np.testing.assert_array_equal(
        np.isnan(masked.per_repetition), [[True, True], [True, False]]
    )
    np.testing.assert_array_equal(np.isnan(masked.shared), [True, False])
    assert masked["global"].item() == 1.0
    xr.testing.assert_identical(masked.status, dataset.status)
    assert not np.isnan(dataset.per_repetition).any()


def test_masking_requires_the_spectral_dimension():
    """An input without the dataset's spectral dimension is rejected."""
    data = xr.DataArray(np.ones((2, 4)), dims=("x", "time"))
    with pytest.raises(ValueError, match="spectral dimension 'acquisition_idx'"):
        mask_fit_dataset_where_input_nonfinite(xr.Dataset(), data)
