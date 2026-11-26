"""``pca_denoise`` input handling fails early or records its noise estimator.

The fixtures are small seeded complex Gaussian arrays in arbitrary signal
units. Values are compared with ``rtol=1e-12`` because each comparison
repeats the same eigendecomposition on identical input.
"""

import importlib

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.denoising import pca_denoise


def _noise(shape, dims, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return xr.DataArray(values, dims=dims, attrs={"unit": "a.u."})


def _image_series(n_repetitions):
    return _noise((4, 5, n_repetitions), ("read_idx", "phase_idx", "repetition_idx"))


def _spectral_series(n_repetitions):
    return _noise((16, n_repetitions), ("acquisition_idx", "repetition_idx"))


@pytest.mark.parametrize("components", [np.int64(2), np.int32(2), np.uint8(2)])
def test_numpy_integer_component_count_matches_python_int(components):
    data = _image_series(6)

    expected = pca_denoise(data, components=2)
    result = pca_denoise(data, components=components)

    xr.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("components", [True, 2.0, "2"])
def test_component_count_of_other_types_names_the_valid_types(components):
    with pytest.raises(ValueError, match="components must be an int"):
        pca_denoise(_image_series(6), components=components)


@pytest.mark.parametrize("method", ["last_reps", "auto"])
def test_last_reps_rejects_fewer_than_ten_redundant_samples(method):
    data = _spectral_series(9)

    with pytest.raises(ValueError, match="at least 10 redundant samples; got 9"):
        pca_denoise(data, noise_estimation_method=method)


def test_last_reps_uses_the_last_tenth_of_redundant_samples():
    data = _spectral_series(20)
    flattened = data.transpose("repetition_idx", "acquisition_idx").values.T
    variance = float(np.var(flattened[:, -2:]))

    result = pca_denoise(data, noise_estimation_method="last_reps")
    expected = pca_denoise(
        data, noise_estimation_method="manual", noise_threshold=variance
    )

    xr.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    assert result.attrs["pca_noise_estimation_method"] == "last_reps"
    assert "pca_noise_estimation_fallback_reason" not in result.attrs


def test_auto_records_the_corner_estimate():
    data = _image_series(6)

    result = pca_denoise(data)
    expected = pca_denoise(data, noise_estimation_method="corner")

    xr.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    assert result.attrs["pca_noise_estimation_method"] == "corner"
    assert "pca_noise_estimation_fallback_reason" not in result.attrs


def test_auto_records_the_last_reps_fallback_and_its_reason():
    data = _spectral_series(20)

    result = pca_denoise(data)
    expected = pca_denoise(data, noise_estimation_method="last_reps")

    xr.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    assert result.attrs["pca_noise_estimation_method"] == "last_reps"
    assert (
        "read_idx and phase_idx"
        in (result.attrs["pca_noise_estimation_fallback_reason"])
    )


def test_corner_without_image_axes_names_the_missing_axes():
    with pytest.raises(ValueError, match="requires read_idx and phase_idx"):
        pca_denoise(_spectral_series(20), noise_estimation_method="corner")


def test_noise_threshold_is_recorded_as_manual_and_skips_estimation():
    data = _spectral_series(9)

    result = pca_denoise(data, noise_estimation_method="auto", noise_threshold=1.0)
    expected = pca_denoise(data, noise_estimation_method="manual", noise_threshold=1.0)

    xr.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    assert result.attrs["pca_noise_estimation_method"] == "manual"


def test_unknown_noise_estimator_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="'auto', 'corner', 'last_reps', 'manual'"):
        pca_denoise(_image_series(6), noise_estimation_method="median")


def test_explicit_components_add_no_estimation_attributes_and_keep_input_attrs():
    data = _image_series(6)

    result = pca_denoise(data, components=2)
    pca_denoise(data)

    assert result.attrs == {"unit": "a.u."}
    assert data.attrs == {"unit": "a.u."}


def test_module_exports_only_its_own_function_and_imports_pyplot_lazily():
    module = importlib.import_module("hypermri.analysis.denoising.algorithms")

    assert module.__all__ == ["pca_denoise"]
    assert not hasattr(module, "plt")


def test_scree_plot_is_drawn_on_request(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))

    pca_denoise(_image_series(6), plot=True)

    assert len(shown) == 1
    assert len(shown[0].axes) == 2
    plt.close("all")
