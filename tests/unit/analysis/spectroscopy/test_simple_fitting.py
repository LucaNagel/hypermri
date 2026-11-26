"""Contracts for the simple spectroscopy fit, noise and uncertainty path."""

import numpy as np
import pytest
import xarray as xr

import hypermri.analysis.spectroscopy.noise_estimation as noise_estimation_module
from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
)
from hypermri.analysis.spectroscopy import (
    compare_metabolite_relevance,
    estimate_fit_uncertainty,
    estimate_noise,
    fit_spectrum,
)


def _single_fid() -> xr.DataArray:
    time = np.arange(12, dtype=float) * 0.001
    signal = (1.2 + 0.3j) * np.exp(2j * np.pi * 40.0 * time) * np.exp(-time / 0.025)
    return xr.DataArray(
        signal,
        dims="acquisition_idx",
        coords={
            "acquisition_idx": np.arange(time.size),
            "sampling_time_axis_s": ("acquisition_idx", time),
        },
        attrs={"signal_domain": "temporal", "unit": "a.u."},
    )


def _fit_options(**options) -> dict:
    return {
        "metabs": ["target"],
        "metabs_freqs_hz": [40.0],
        "initial_t2s_s": [0.025],
        "metabs_freq_ranges": 0.0,
        "metabs_t2_ranges": 0.0,
        "min_t2_s": 0.025,
        "max_t2_s": 0.025,
        "npoints": 3,
        "niter": 1,
        "rep_fitting": 1,
        **options,
    }


def _params(data: xr.DataArray, **options) -> xr.Dataset:
    return def_fit_params(data, _fit_options(**options))


def test_def_fit_params_rejects_a_misspelled_option_and_names_it():
    """Regression: a misspelled key used to keep its default silently."""
    with pytest.raises(ValueError, match="metabs_freq_range") as error:
        def_fit_params(_single_fid(), _fit_options(metabs_freq_range=20.0))

    message = str(error.value)
    assert "did you mean 'metabs_freq_ranges'" in message
    assert "'initial_t2s_s'" in message
    assert "'uncertainty_seed'" in message


@pytest.mark.parametrize(
    "entry",
    [
        lambda data, options: fit_spectrum(data, options),
        lambda data, options: compare_metabolite_relevance(
            data, options, targets="target"
        ),
        lambda data, options: estimate_fit_uncertainty(
            data, options, fit_spectrum(data, _params(data)), noise=0.1
        ),
    ],
    ids=["fit_spectrum", "compare_metabolite_relevance", "estimate_fit_uncertainty"],
)
def test_public_entries_reject_unknown_fit_options(entry):
    with pytest.raises(ValueError, match="not_an_option"):
        entry(_single_fid(), _fit_options(not_an_option=1))


@pytest.mark.parametrize("lineshape", ["lorentz", "voigt"])
def test_prepared_fit_options_can_be_prepared_again(lineshape):
    """The attributes written by ``def_fit_params`` are accepted as input."""
    data = _single_fid()
    prepared = _params(data, lineshape=lineshape, metabs_t2_ranges=0.01)

    repeated = def_fit_params(data, dict(prepared.attrs))

    assert repeated.attrs.keys() == prepared.attrs.keys()
    for name in prepared.data_vars:
        xr.testing.assert_identical(repeated[name], prepared[name])


def test_fit_spectrum_without_noise_is_identical_to_numerical_entry():
    data = _single_fid()
    params = _params(data)

    expected = fit_data_pseudo_inv(data, params)
    actual = fit_spectrum(data, params)

    xr.testing.assert_identical(actual, expected)
    assert "fit_freqs_ppm" not in actual


def test_fit_spectrum_adds_fit_freqs_ppm_for_calibrated_axes():
    data = _single_fid()
    params = _params(data)
    hz_axis = np.array([-200.0, -100.0, 0.0, 100.0])
    params.attrs["freq_axis_hz"] = hz_axis
    params.attrs["freq_axis_ppm"] = 170.0 + hz_axis / 100.0

    actual = fit_spectrum(data, params)

    assert actual.fit_freqs_ppm.dims == actual.fit_freqs.dims
    assert actual.fit_freqs_ppm.attrs["unit"] == "ppm"
    np.testing.assert_allclose(
        actual.fit_freqs_ppm.values, 170.0 + actual.fit_freqs.values / 100.0
    )


def test_scalar_noise_adds_diagnostics_without_changing_point_estimates():
    data = _single_fid()
    params = _params(data)

    baseline = fit_spectrum(data, params)
    result = fit_spectrum(data, params, noise=0.02)

    for name in ("fit_amps", "fit_freqs", "fit_t2s", "fit_rss", "fit_objective"):
        xr.testing.assert_identical(result[name], baseline[name])
    assert result.noise_sigma_quadrature.item() == pytest.approx(0.02)
    assert np.isfinite(result.amplitude_std_quadrature_conditional).all()
    assert np.isfinite(result.amplitude_snr_conditional).all()
    assert result.attrs["supplied_noise_diagnostics"] is True


def test_labelled_noise_map_aligns_by_spectrum_coordinates():
    data = xr.concat([_single_fid(), 2 * _single_fid()], dim="voxel").assign_coords(
        voxel=["left", "right"]
    )
    params = _params(data)
    sigma = xr.DataArray(
        [0.01, 0.03], dims="voxel", coords={"voxel": ["left", "right"]}
    )

    result = fit_spectrum(data, params, noise=sigma)

    xr.testing.assert_equal(result.noise_sigma_quadrature, sigma)
    ratio = result.amplitude_std_quadrature_conditional.sel(
        voxel="right"
    ) / result.amplitude_std_quadrature_conditional.sel(voxel="left")
    assert ratio.item() == pytest.approx(3.0)


@pytest.mark.parametrize("noise", [0.0, -1.0, np.nan])
def test_invalid_noise_fails_before_numerical_fit(monkeypatch, noise):
    data = _single_fid()
    params = _params(data)

    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("Invalid noise must be rejected before fitting.")

    monkeypatch.setattr(
        "hypermri.analysis.spectroscopy.fitting.fit_data_pseudo_inv",
        unexpected_fit,
    )
    with pytest.raises(ValueError, match="finite positive"):
        fit_spectrum(data, params, noise=noise)


def test_noise_map_coordinate_mismatch_is_rejected():
    data = xr.concat([_single_fid(), _single_fid()], dim="voxel").assign_coords(
        voxel=["left", "right"]
    )
    params = _params(data)
    sigma = xr.DataArray(
        [0.01, 0.02], dims="voxel", coords={"voxel": ["left", "other"]}
    )

    with pytest.raises(ValueError, match="coordinates must exactly match"):
        fit_spectrum(data, params, noise=sigma)


def test_estimate_noise_returns_scalar_or_labelled_map():
    samples = xr.DataArray(
        [[1 + 1j, -1 - 1j], [2 + 2j, -2 - 2j]],
        dims=("voxel", "acquisition_idx"),
        coords={"voxel": ["a", "b"], "acquisition_idx": [0, 1]},
        attrs={"signal_domain": "temporal", "unit": "a.u."},
    )

    scalar = estimate_noise(samples, method="rf_off")
    noise_map = estimate_noise(
        samples,
        method="rf_off",
        reduce_dims="acquisition_idx",
    )
    pipeline_map = estimate_noise(
        samples,
        method="rf_off",
        reduce_dims="acquisition_idx",
        return_pipeline=True,
    )

    assert scalar.item() == pytest.approx(np.sqrt(5 / 2))
    np.testing.assert_allclose(noise_map.values, [1.0, 2.0])
    assert noise_map.dims == ("voxel",)
    np.testing.assert_array_equal(noise_map.noise_sample_count, [2, 2])
    assert noise_map.attrs["noise_source"] == "rf_off"
    assert pipeline_map.fit_noise_sigma.dims == ("fit_method", "voxel")
    np.testing.assert_array_equal(pipeline_map.voxel, ["a", "b"])
    xr.testing.assert_equal(
        pipeline_map.fit_noise_sigma.sel(
            fit_method="complex_second_moment", drop=True
        ).rename("noise_sigma_quadrature"),
        pipeline_map.noise_sigma_quadrature,
    )


def test_estimate_noise_can_return_an_inspectable_processing_pipeline():
    samples = xr.DataArray(
        [1 + 1j, 3 + 1j],
        dims="acquisition_idx",
        attrs={"signal_domain": "temporal", "unit": "a.u."},
    )

    noise = estimate_noise(samples, method="complex_samples")
    pipeline = estimate_noise(
        samples,
        method="complex_samples",
        return_pipeline=True,
    )

    assert isinstance(noise, xr.DataArray)
    assert isinstance(pipeline, xr.Dataset)
    xr.testing.assert_identical(pipeline.noise_sigma_quadrature, noise)
    assert pipeline.noise_sigma_quadrature.item() == pytest.approx(np.sqrt(3.0))
    assert pipeline.sample_mean_real.item() == pytest.approx(2.0)
    assert pipeline.sample_mean_imag.item() == pytest.approx(1.0)
    assert pipeline.sample_std_real.item() == pytest.approx(1.0)
    assert pipeline.sample_std_imag.item() == pytest.approx(0.0)
    assert pipeline.noise_sigma_std_rms.item() == pytest.approx(1 / np.sqrt(2))
    assert pipeline.noise_sigma_std_mean_legacy.item() == pytest.approx(0.5)
    assert pipeline.noise_sigma_rayleigh_mle.item() == pytest.approx(noise.item())
    np.testing.assert_array_equal(
        pipeline.fit_method,
        [
            "gaussian_real",
            "gaussian_imaginary",
            "rayleigh_magnitude",
            "complex_second_moment",
        ],
    )
    assert pipeline.fit_location.sel(
        fit_method="gaussian_real"
    ).item() == pytest.approx(2.0)
    assert pipeline.fit_scale.sel(fit_method="gaussian_real").item() == pytest.approx(
        1.0
    )
    assert pipeline.fit_location.sel(
        fit_method="gaussian_imaginary"
    ).item() == pytest.approx(1.0)
    assert pipeline.fit_scale.sel(
        fit_method="gaussian_imaginary"
    ).item() == pytest.approx(0.0)
    assert pipeline.fit_location.sel(
        fit_method="rayleigh_magnitude"
    ).item() == pytest.approx(0.0)
    assert pipeline.fit_scale.sel(
        fit_method="rayleigh_magnitude"
    ).item() == pytest.approx(noise.item())
    assert np.isnan(pipeline.fit_scale.sel(fit_method="complex_second_moment").item())
    assert pipeline.fit_noise_sigma.sel(
        fit_method="complex_second_moment"
    ).item() == pytest.approx(noise.item())
    assert pipeline.fit_is_primary.sel(fit_method="complex_second_moment").item()
    assert not pipeline.fit_is_primary.sel(fit_method="rayleigh_magnitude").item()
    np.testing.assert_array_equal(pipeline.selected_noise_samples, samples)
    np.testing.assert_array_equal(pipeline.processed_noise_samples, samples)
    assert pipeline.processing_description.size == 7
    assert pipeline.attrs["noise_processing_pipeline_version"] == "2"
    assert pipeline.attrs["fit_parameter_contract_version"] == "1"
    assert pipeline.attrs["noise_method"] == "complex_samples"
    assert pipeline.attrs["centered"] is False


def test_centered_noise_pipeline_records_samples_entering_estimator():
    samples = xr.DataArray(
        [1 + 1j, 3 + 1j],
        dims="acquisition_idx",
        attrs={"signal_domain": "temporal"},
    )

    pipeline = estimate_noise(
        samples,
        method="complex_samples",
        center=True,
        return_pipeline=True,
    )

    np.testing.assert_allclose(pipeline.processed_noise_samples, [-1 + 0j, 1 + 0j])
    assert pipeline.noise_sigma_quadrature.item() == pytest.approx(1 / np.sqrt(2))
    assert pipeline.noise_sigma_std_rms.item() == pytest.approx(
        pipeline.noise_sigma_quadrature.item()
    )
    assert pipeline.noise_sigma_rayleigh_mle.item() == pytest.approx(
        pipeline.noise_sigma_quadrature.item()
    )
    assert pipeline.fit_location.sel(
        fit_method="gaussian_real"
    ).item() == pytest.approx(0.0)
    assert pipeline.fit_location.sel(
        fit_method="gaussian_imaginary"
    ).item() == pytest.approx(0.0)
    assert pipeline.attrs["centered"] is True


def test_noise_pipeline_reuses_the_primary_temporal_transformation(monkeypatch):
    samples = xr.DataArray(
        [1 + 1j, -1 - 1j],
        dims="acquisition_idx",
        attrs={"signal_domain": "temporal"},
    )
    original = noise_estimation_module._temporal_noise_data
    calls = []

    def tracked_transform(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        noise_estimation_module,
        "_temporal_noise_data",
        tracked_transform,
    )

    estimate_noise(samples, method="complex_samples", return_pipeline=True)

    assert len(calls) == 1


def test_estimate_noise_dispatches_frames_and_outside_mask():
    samples = xr.DataArray(
        np.asarray(
            [
                [[1 + 1j, 1 - 1j], [20 + 20j, 20 - 20j]],
                [[2 + 2j, 2 - 2j], [30 + 30j, 30 - 30j]],
            ]
        ),
        dims=("frame", "voxel", "acquisition_idx"),
        coords={"frame": [10, 20], "voxel": ["outside", "inside"]},
    )
    frames = estimate_noise(
        samples,
        method="noise_frames",
        selection=[0],
        selection_dim="frame",
    )
    mask = xr.DataArray(
        [0.0, 1.0], dims="voxel", coords={"voxel": ["outside", "inside"]}
    )
    outside = estimate_noise(
        samples,
        method="outside_object",
        mask=mask,
    )
    frames_pipeline = estimate_noise(
        samples,
        method="noise_frames",
        selection=[0],
        selection_dim="frame",
        return_pipeline=True,
    )
    outside_pipeline = estimate_noise(
        samples,
        method="outside_object",
        mask=mask,
        return_pipeline=True,
    )

    expected_frame = np.sqrt(np.mean(np.abs(samples.isel(frame=0).values) ** 2) / 2)
    expected_outside = np.sqrt(
        np.mean(np.abs(samples.sel(voxel="outside").values) ** 2) / 2
    )
    assert frames.item() == pytest.approx(expected_frame)
    assert outside.item() == pytest.approx(expected_outside)
    assert frames_pipeline.noise_sigma_quadrature.item() == pytest.approx(
        expected_frame
    )
    assert frames_pipeline.selected_noise_samples.sizes["frame"] == 1
    assert outside_pipeline.noise_sigma_quadrature.item() == pytest.approx(
        expected_outside
    )
    assert outside_pipeline.selected_noise_samples.sel(voxel="inside").isnull().all()


@pytest.mark.parametrize("method", ["outer_kspace", "automatic_outer_kspace"])
def test_estimate_noise_has_no_outer_kspace_method(method):
    samples = xr.DataArray(np.ones(8, dtype=complex), dims="acquisition_idx")

    with pytest.raises(ValueError, match="separate function estimate_csi_noise"):
        estimate_noise(samples, method=method)


def test_estimate_noise_plot_is_opt_in_and_method_specific(monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    samples = xr.DataArray(
        np.asarray(
            [
                [[1 + 1j, 1 - 1j], [20 + 20j, 20 - 20j]],
                [[2 + 2j, 2 - 2j], [30 + 30j, 30 - 30j]],
            ]
        ),
        dims=("frame", "voxel", "acquisition_idx"),
        coords={"frame": [10, 20], "voxel": ["outside", "inside"]},
    )
    mask = xr.DataArray(
        [0.0, 1.0], dims="voxel", coords={"voxel": ["outside", "inside"]}
    )
    captured_titles = []
    captured_annotations = []
    captured_legend_labels = []

    def capture_show():
        figure = plt.gcf()
        captured_titles.append([axis.get_title() for axis in figure.axes])
        captured_annotations.append(
            [text.get_text() for axis in figure.axes for text in axis.texts]
        )
        captured_legend_labels.append(
            [
                text.get_text()
                for axis in figure.axes
                if axis.get_legend() is not None
                for text in axis.get_legend().get_texts()
            ]
        )
        plt.close(figure)

    monkeypatch.setattr(plt, "show", capture_show)
    baseline = estimate_noise(samples, method="rf_off", reduce_dims="acquisition_idx")
    assert captured_titles == []

    plotted = estimate_noise(
        samples,
        method="rf_off",
        reduce_dims="acquisition_idx",
        plot=True,
    )
    xr.testing.assert_identical(plotted, baseline)
    estimate_noise(samples, method="complex_samples", plot=True)
    estimate_noise(
        samples,
        method="noise_frames",
        selection=[0],
        selection_dim="frame",
        plot=True,
    )
    estimate_noise(samples, method="outside_object", mask=mask, plot=True)

    assert len(captured_titles) == 4
    assert "RF-off sample stability" in captured_titles[0]
    assert "Magnitude distribution + Rayleigh fit" in captured_titles[1]
    assert "Noise-frame selection" in captured_titles[2]
    assert "Mask and selected outside region" in captured_titles[3]
    assert any("real: mean=" in text for text in captured_annotations[0])
    assert any("fit scale/noise=" in text for text in captured_annotations[0])
    assert "noise sigma radius" in captured_legend_labels[0]


def test_estimate_noise_plot_uses_the_requested_figure_size(monkeypatch):
    """Regression: ``figsize`` was nested in a dict and the default was used."""
    plt = pytest.importorskip("matplotlib.pyplot")
    samples = xr.DataArray(
        np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]), dims="acquisition_idx"
    )
    figure_sizes = []

    def capture_show():
        figure = plt.gcf()
        figure_sizes.append(tuple(figure.get_size_inches()))
        plt.close(figure)

    monkeypatch.setattr(plt, "show", capture_show)
    estimate_noise(samples, method="rf_off", plot=True)
    estimate_noise(samples, method="rf_off", plot=True, figsize=(5.0, 3.0))

    # Matplotlib stores the requested inches as floats without conversion.
    assert figure_sizes == [(12.0, 5.0), (5.0, 3.0)]


def test_magnitude_noise_returns_rayleigh_scale_of_complex_noise():
    """Magnitude samples give the Rayleigh scale, which equals the complex sigma.

    Synthetic complex Gaussian noise with per-quadrature sigma 2.5 (seed 7,
    20000 samples). Taking ``abs`` and estimating with
    ``sample_type="magnitude"`` must reproduce the complex estimate exactly,
    because ``abs(z)**2`` is the quantity both estimators sum. The recovery
    tolerance of 2 % is about four standard errors of the Rayleigh scale MLE
    (relative SE ``1/(2*sqrt(N))`` = 0.35 %).
    """
    rng = np.random.default_rng(7)
    sigma = 2.5
    complex_noise = xr.DataArray(
        sigma * (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)),
        dims="acquisition_idx",
        attrs={"unit": "a.u."},
    )
    magnitude = np.abs(complex_noise)
    magnitude.attrs = {"unit": "a.u."}
    original = magnitude.copy(deep=True)

    complex_sigma = estimate_noise(complex_noise, method="complex_samples")
    magnitude_sigma = estimate_noise(
        magnitude, method="complex_samples", sample_type="magnitude"
    )
    expected = np.sqrt(np.sum(magnitude.values**2) / (2 * magnitude.size))

    assert magnitude_sigma.item() == pytest.approx(expected, rel=1e-12)
    assert magnitude_sigma.item() == pytest.approx(complex_sigma.item(), rel=1e-12)
    assert magnitude_sigma.item() == pytest.approx(sigma, rel=0.02)
    assert magnitude_sigma.attrs["sample_type"] == "magnitude"
    assert complex_sigma.attrs["sample_type"] == "complex"
    xr.testing.assert_identical(magnitude, original)


def test_magnitude_noise_works_with_mask_and_pipeline():
    image = xr.DataArray(
        [[3.0, 4.0], [100.0, 120.0]],
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    body = xr.DataArray(
        [[0.0, 0.0], [1.0, 1.0]], dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}
    )

    pipeline = estimate_noise(
        image,
        method="outside_object",
        mask=body,
        sample_type="magnitude",
        return_pipeline=True,
    )

    expected = np.sqrt((3.0**2 + 4.0**2) / (2 * 2))
    assert pipeline.noise_sigma_quadrature.item() == pytest.approx(expected)
    assert pipeline.noise_sigma_rayleigh_mle.item() == pytest.approx(expected)
    assert pipeline.attrs["sample_type"] == "magnitude"
    assert "magnitude" in str(pipeline.processing_description.values[0])


@pytest.mark.parametrize(
    ("data", "options", "error", "message"),
    [
        (np.asarray([1 + 1j, 2 + 0j]), {}, TypeError, "real-valued magnitude"),
        (np.asarray([1.0, -2.0]), {}, ValueError, "non-negative"),
        (np.asarray([1.0, 2.0]), {"center": True}, ValueError, "center=True"),
        (
            np.asarray([1.0, 2.0]),
            {"signal_domain": "spectral"},
            ValueError,
            "signal_domain='temporal'",
        ),
    ],
)
def test_magnitude_noise_rejects_invalid_input(data, options, error, message):
    samples = xr.DataArray(data, dims="acquisition_idx")

    with pytest.raises(error, match=message):
        estimate_noise(
            samples, method="complex_samples", sample_type="magnitude", **options
        )


def test_estimate_noise_rejects_unknown_sample_type():
    samples = xr.DataArray(np.asarray([1 + 1j, 1 - 1j]), dims="acquisition_idx")

    with pytest.raises(ValueError, match="sample_type must be one of"):
        estimate_noise(samples, method="rf_off", sample_type="phase")


def test_real_samples_still_require_explicit_magnitude_type():
    samples = xr.DataArray(np.asarray([1.0, 2.0]), dims="acquisition_idx")

    with pytest.raises(TypeError, match="complex-valued"):
        estimate_noise(samples, method="rf_off")


def test_magnitude_noise_plot_marks_quadrature_panels(monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    samples = xr.DataArray(np.asarray([1.0, 2.0, 0.5, 1.5]), dims="acquisition_idx")
    captured = []

    def capture_show():
        figure = plt.gcf()
        captured.append(
            [text.get_text() for axis in figure.axes for text in axis.texts]
        )
        plt.close(figure)

    monkeypatch.setattr(plt, "show", capture_show)
    estimate_noise(samples, method="rf_off", sample_type="magnitude", plot=True)

    assert captured[0].count("Not defined for magnitude samples") == 2
    assert any("fit scale/noise=" in text for text in captured[0])


def test_estimate_noise_rejects_unknown_keyword_arguments():
    """Unknown options fail instead of being silently ignored."""
    samples = xr.DataArray(
        np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]), dims="acquisition_idx"
    )

    with pytest.raises(TypeError, match="colour"):
        estimate_noise(samples, method="rf_off", colour="red")


@pytest.mark.parametrize(
    "options",
    [
        {"uncertainty_bootstrap_samples": 1},
        {"uncertainty_bootstrap_samples": 2.5},
        {"uncertainty_seed": -1},
        {"uncertainty_confidence_level": 1.0},
    ],
)
def test_invalid_uncertainty_fit_settings_fail_early(options):
    with pytest.raises(ValueError, match="uncertainty_"):
        _params(_single_fid(), **options)


def test_parameter_bootstrap_reads_b_from_fit_parameters_and_is_deterministic():
    data = _single_fid()
    params = _params(
        data,
        uncertainty_bootstrap_samples=7,
        uncertainty_confidence_level=0.8,
        uncertainty_seed=19,
    )
    fit = fit_spectrum(data, params, noise=0.02)

    first = estimate_fit_uncertainty(data, params, fit)
    second = estimate_fit_uncertainty(data, params, fit)

    xr.testing.assert_identical(first, second)
    assert first.bootstrap_requested.item() == 7
    assert first.bootstrap_successful.item() == 7
    assert first.bootstrap_status.item() == "success"
    assert first.attrs["uncertainty_confidence_level"] == pytest.approx(0.8)
    assert np.isfinite(first.amplitude_abs_ci_low).all()
    assert np.isfinite(first.frequency_ci_low).all()
    assert np.isfinite(first.t2_ci_low).all()


def test_parameter_bootstrap_accepts_b_in_fit_parameter_mapping():
    data = _single_fid()
    options = _fit_options(
        uncertainty_bootstrap_samples=3,
        uncertainty_seed=4,
    )
    fit = fit_spectrum(data, options, noise=0.02)

    result = estimate_fit_uncertainty(data, options, fit)

    assert result.bootstrap_requested.item() == 3
    assert result.attrs["uncertainty_seed"] == 4


def test_quick_relevance_refits_m0_once_and_reports_observed_rss(monkeypatch):
    data = xr.concat([_single_fid(), 2 * _single_fid()], dim="voxel").assign_coords(
        voxel=["left", "right"]
    )
    params = _params(data)
    fit = fit_spectrum(data, params)
    selection = xr.DataArray(
        [True, False], dims="voxel", coords={"voxel": ["left", "right"]}
    )
    from hypermri.analysis.spectroscopy import relevance

    numerical_fit = relevance.fit_data_pseudo_inv
    calls = []

    def counted_fit(*args, **kwargs):
        calls.append(1)
        return numerical_fit(*args, **kwargs)

    monkeypatch.setattr(relevance, "fit_data_pseudo_inv", counted_fit)
    result = compare_metabolite_relevance(
        data,
        params,
        fit,
        targets="target",
        selection=selection,
    )

    assert len(calls) == 1
    assert result.rss_improvement.sel(target="target", voxel="left").item() > 0
    assert np.isnan(result.rss_improvement.sel(target="target", voxel="right").item())
    assert result.m0_fit_status.sel(target="target", voxel="right").item() == (
        "not_selected"
    )
    assert "not a p-value" in result.attrs["relevance_scope"]


def test_multi_spectrum_bootstrap_requires_and_respects_selection():
    data = xr.concat([_single_fid(), 2 * _single_fid()], dim="voxel").assign_coords(
        voxel=["left", "right"]
    )
    params = _params(data, uncertainty_bootstrap_samples=3, uncertainty_seed=7)
    fit = fit_spectrum(data, params, noise=0.02)

    with pytest.raises(ValueError, match="requires a labelled selection"):
        estimate_fit_uncertainty(data, params, fit)

    selection = xr.DataArray(
        [True, False], dims="voxel", coords={"voxel": ["left", "right"]}
    )
    result = estimate_fit_uncertainty(data, params, fit, selection=selection)

    assert result.bootstrap_status.sel(voxel="left").item() == "success"
    assert result.bootstrap_status.sel(voxel="right").item() == "not_selected"
    assert result.bootstrap_requested.sel(voxel="left").item() == 3
    assert result.bootstrap_requested.sel(voxel="right").item() == 0
    assert np.isnan(result.amplitude_abs_ci_low.sel(voxel="right")).all()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "fid_tail"}, "separate function estimate_csi_noise"),
        (
            {"method": "noise_frames", "selection": [0, 1]},
            "requires selection and selection_dim",
        ),
        ({"method": "outside_object"}, "requires a labelled mask"),
    ],
    ids=["unknown-method", "frames-without-dimension", "outside-without-mask"],
)
def test_estimate_noise_rejects_incomplete_source_selection(options, message):
    with pytest.raises(ValueError, match=message):
        estimate_noise(_single_fid(), **options)


def test_fit_spectrum_rejects_noise_diagnostics_with_a_detection_test(monkeypatch):
    def fail_fit(*args, **kwargs):
        raise AssertionError("the numerical fit must not start")

    monkeypatch.setattr(
        "hypermri.analysis.spectroscopy.fitting.fit_data_pseudo_inv", fail_fit
    )
    options = _fit_options(
        detection_target="target",
        detection_sigma=0.1,
        detection_null_samples=3,
        detection_seed=7,
    )
    with pytest.raises(ValueError, match="cannot be combined with detection_target"):
        fit_spectrum(_single_fid(), options, noise=0.1)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ("absent", "Unknown relevance targets"),
        (["target", "target"], "unique metabolite names"),
        ([], "unique metabolite names"),
    ],
    ids=["unknown", "duplicate", "empty"],
)
def test_quick_relevance_rejects_invalid_targets(targets, message):
    with pytest.raises(ValueError, match=message):
        compare_metabolite_relevance(_single_fid(), _fit_options(), targets=targets)


def test_parameter_bootstrap_requires_noise_and_known_metabolites():
    data = _single_fid()
    params = _params(data, uncertainty_bootstrap_samples=3, uncertainty_seed=7)
    fit = fit_spectrum(data, params)

    with pytest.raises(ValueError, match="requires noise or a fit containing"):
        estimate_fit_uncertainty(data, params, fit)
    with pytest.raises(ValueError, match="Unknown metabolites requested"):
        estimate_fit_uncertainty(data, params, fit, noise=0.02, metabolites=["absent"])


def test_per_source_noise_functions_are_removed():
    import hypermri.analysis.spectroscopy as public_spectroscopy

    for name in (
        "estimate_noise_from_complex_samples",
        "estimate_noise_from_frames",
        "estimate_noise_from_outside_mask",
        "estimate_noise_from_rf_off",
    ):
        assert name not in noise_estimation_module.__all__
        assert not hasattr(noise_estimation_module, name)
        with pytest.raises(AttributeError):
            getattr(public_spectroscopy, name)
