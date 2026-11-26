"""Patch-wise MP-PCA, threshold PCA and general PCA denoising.

References: Veraart et al., NeuroImage 142 (2016) for MP-PCA and Henriques
et al., Imaging Neuroscience 1 (2023), Eqs. 8 and 9, for GPCA and TPCA.
Fixtures are seeded Gaussian low-rank matrices with unit noise variance. For
the chosen seeds the asymptotic Marchenko-Pastur edge separates signal from
noise eigenvalues, so ranks are exact; finite-size fluctuations make other
seeds differ by one component. The 5 % noise-sigma tolerance covers the
spread of 20 seeds for a 250 x 20 matrix (largest deviation 2.4 %). Results
of one decomposition compared in two layouts use ``rtol=1e-10``.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.denoising import (
    PCANoiseClassification,
    compute_marchenko_pastur_upper_edge,
    denoise_gpca,
    denoise_mppca,
    denoise_tpca,
    estimate_gpca_rank,
    estimate_mppca_rank,
    estimate_tpca_rank,
)

_FEATURES = ("repetition_idx",)
_SAMPLES = ("voxel_idx",)


def _low_rank(n_voxels, n_repetitions, rank, amplitude, seed):
    """Return a noise-free rank-``rank`` matrix and its noisy copy."""
    rng = np.random.default_rng(seed)
    clean = amplitude * (
        rng.standard_normal((n_voxels, rank))
        @ rng.standard_normal((rank, n_repetitions))
    )
    return clean, clean + rng.standard_normal((n_voxels, n_repetitions))


def _matrix(values):
    return xr.DataArray(values, dims=(*_SAMPLES, *_FEATURES))


def _oracle_truncation(noisy, rank):
    """Project the centred noisy matrix onto its leading ``rank`` components."""
    mean = noisy.mean(axis=0)
    centred = noisy - mean
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    basis = right[:rank].conj().T
    return centred @ basis @ basis.conj().T + mean


def _rmse(estimate, reference):
    return np.sqrt(np.mean(np.abs(estimate - reference) ** 2))


def _centred_eigenvalues(noisy):
    centred = noisy - noisy.mean(axis=0)
    n_rows = centred.shape[0] - 1
    return np.linalg.eigvalsh(centred.T @ centred / n_rows), n_rows


def test_upper_edge_formula():
    assert compute_marchenko_pastur_upper_edge(2.0, aspect_ratio=0.25) == 4.5
    with pytest.raises(ValueError, match="aspect_ratio"):
        compute_marchenko_pastur_upper_edge(1.0, aspect_ratio=0.0)
    with pytest.raises(ValueError, match="noise_variance"):
        compute_marchenko_pastur_upper_edge(-1.0, aspect_ratio=0.5)


def test_mppca_kernel_estimates_noise_of_a_250_by_20_matrix():
    for seed in range(5):
        _, noisy = _low_rank(250, 20, 0, 0.0, seed)
        eigenvalues, n_rows = _centred_eigenvalues(noisy)

        result = estimate_mppca_rank(eigenvalues, n_rows=n_rows, n_cols=20)

        assert isinstance(result, PCANoiseClassification)
        assert result.n_signal_components == 0
        assert np.sqrt(result.noise_variance) == pytest.approx(1.0, rel=0.05)


def test_tpca_kernel_counts_eigenvalues_above_the_edge():
    eigenvalues = np.array([0.5, 1.0, 2.0, 2.26, 9.0])
    result = estimate_tpca_rank(eigenvalues, n_rows=20, n_cols=5, noise_variance=1.0)
    # gamma = 5 / 20 gives the edge 2.25.
    assert result == PCANoiseClassification(2, 1.0)


def test_gpca_kernel_keeps_the_largest_noise_group_with_mean_at_most_prior():
    eigenvalues = np.array([3.0, 1.4, 1.0, 0.6])
    # Tail means: 1.5, 1.0, 0.8, 0.6; the first not above 1.0 starts at index 1.
    assert estimate_gpca_rank(eigenvalues, noise_variance=1.0) == (
        PCANoiseClassification(1, 1.0)
    )
    assert estimate_gpca_rank(eigenvalues, noise_variance=0.1).n_signal_components == 4


@pytest.mark.parametrize(
    "call",
    [
        lambda: estimate_mppca_rank(np.ones(3), n_rows=10, n_cols=4),
        lambda: estimate_mppca_rank(np.array([1.0, np.nan]), n_rows=2, n_cols=5),
        lambda: estimate_tpca_rank(np.ones(2), n_rows=2, n_cols=5, noise_variance=0.0),
        lambda: estimate_gpca_rank(np.ones((2, 2)), noise_variance=1.0),
    ],
)
def test_kernels_reject_invalid_input(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    ("shape", "rank", "amplitude", "seed"),
    [
        ((1024, 20), 3, 1.0, 0),
        ((1024, 20), 3, 0.5, 0),
        ((256, 16), 4, 1.0, 3),
        ((64, 40), 2, 1.0, 2),
    ],
)
def test_global_mppca_and_tpca_recover_rank_and_match_oracle(
    shape, rank, amplitude, seed
):
    clean, noisy = _low_rank(*shape, rank, amplitude, seed)
    oracle_error = _rmse(_oracle_truncation(noisy, rank), clean)

    for result in (
        denoise_mppca(_matrix(noisy), feature_dims=_FEATURES, sample_dims=_SAMPLES),
        denoise_tpca(
            _matrix(noisy),
            feature_dims=_FEATURES,
            sample_dims=_SAMPLES,
            noise_variance=1.0,
        ),
    ):
        assert np.all(result["n_components"].values == rank)
        assert _rmse(result["denoised"].values, clean) <= 1.05 * oracle_error


def test_global_gpca_recovers_rank_and_noise_mean_stays_below_prior():
    clean, noisy = _low_rank(1024, 20, 3, 1.0, 0)
    result = denoise_gpca(
        _matrix(noisy), feature_dims=_FEATURES, sample_dims=_SAMPLES, noise_variance=1.0
    )

    assert np.all(result["n_components"].values == 3)
    eigenvalues, _ = _centred_eigenvalues(noisy)
    assert np.sort(eigenvalues)[:-3].mean() <= 1.0
    assert _rmse(result["denoised"].values, clean) <= 1.05 * _rmse(
        _oracle_truncation(noisy, 3), clean
    )


def test_pure_noise_keeps_no_components_for_mppca_and_tpca():
    _, noisy = _low_rank(250, 20, 0, 0.0, 0)
    data = _matrix(noisy)

    mppca = denoise_mppca(data, feature_dims=_FEATURES, sample_dims=_SAMPLES)
    tpca = denoise_tpca(
        data, feature_dims=_FEATURES, sample_dims=_SAMPLES, noise_variance=1.0
    )

    for result in (mppca, tpca):
        assert np.all(result["n_components"].values == 0)
        # Zero components leave the voxel mean of each repetition.
        np.testing.assert_allclose(
            result["denoised"].values,
            np.broadcast_to(noisy.mean(axis=0), noisy.shape),
            rtol=1e-12,
        )


def test_pure_noise_gpca_keeps_at_most_one_component():
    # The mean of all noise eigenvalues equals the prior in expectation, so
    # GPCA removes one component for about half of the seeds.
    for seed in range(10):
        _, noisy = _low_rank(250, 20, 0, 0.0, seed)
        result = denoise_gpca(
            _matrix(noisy),
            feature_dims=_FEATURES,
            sample_dims=_SAMPLES,
            noise_variance=1.0,
        )
        assert result["n_components"].item(0) <= 1


def _image_series(seed, *, complex_valued=False):
    """Return a 12 x 10 grid with 20 repetitions of a rank-2 signal."""
    clean, noisy = _low_rank(120, 20, 2, 1.0, seed)
    if complex_valued:
        rng = np.random.default_rng(seed + 100)
        clean = clean * (1 + 0.5j)
        noise = (
            rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
        ) / np.sqrt(2)
        noisy = clean + noise
    coords = {
        "read_idx": np.arange(12) * 0.5,
        "phase_idx": np.arange(10) * 0.25,
        "repetition_idx": np.arange(20),
    }
    data = xr.DataArray(
        noisy.reshape(12, 10, 20),
        dims=("read_idx", "phase_idx", "repetition_idx"),
        coords=coords,
        name="signal",
        attrs={"units": "a.u."},
    )
    return clean.reshape(12, 10, 20), data


_GRID = ("read_idx", "phase_idx")


@pytest.mark.parametrize(
    "patch_shape", [{"read_idx": 5, "phase_idx": 5}, {"read_idx": 3}]
)
def test_patches_reduce_error_and_preserve_labels(patch_shape):
    clean, data = _image_series(4)
    original = data.copy(deep=True)

    result = denoise_mppca(
        data, feature_dims=_FEATURES, sample_dims=_GRID, patch_shape=patch_shape
    )

    denoised = result["denoised"]
    assert denoised.dims == data.dims
    assert denoised.name == "denoised"
    assert denoised.attrs == {"units": "a.u."}
    xr.testing.assert_identical(denoised.coords.to_dataset(), data.coords.to_dataset())
    assert result["noise_sigma"].dims == _GRID
    assert result.attrs["patch_shape"] == [
        patch_shape.get("read_idx", 12),
        patch_shape.get("phase_idx", 10),
    ]
    assert _rmse(denoised.values, clean) < 0.6 * _rmse(data.values, clean)
    xr.testing.assert_identical(data, original)


def test_patch_with_fewer_voxels_than_measurements_uses_the_voxel_gram():
    clean, data = _image_series(5)
    result = denoise_tpca(
        data,
        feature_dims=_FEATURES,
        sample_dims=_GRID,
        noise_variance=1.0,
        patch_shape={"read_idx": 3, "phase_idx": 3},
    )
    assert _rmse(result["denoised"].values, clean) < _rmse(data.values, clean)


def test_full_extent_patch_equals_global_denoising():
    _, data = _image_series(6)
    global_result = denoise_mppca(data, feature_dims=_FEATURES, sample_dims=_GRID)
    patch_result = denoise_mppca(
        data,
        feature_dims=_FEATURES,
        sample_dims=_GRID,
        patch_shape={"read_idx": 12, "phase_idx": 10},
    )
    xr.testing.assert_allclose(global_result, patch_result, rtol=1e-10)


def test_result_does_not_depend_on_dimension_order():
    _, data = _image_series(7)
    patch_shape = {"read_idx": 4, "phase_idx": 5}
    reference = denoise_gpca(
        data,
        feature_dims=_FEATURES,
        sample_dims=_GRID,
        noise_variance=1.0,
        patch_shape=patch_shape,
    )
    permuted = denoise_gpca(
        data.transpose("repetition_idx", "phase_idx", "read_idx"),
        feature_dims=_FEATURES,
        sample_dims=("phase_idx", "read_idx"),
        noise_variance=1.0,
        patch_shape=patch_shape,
    )
    np.testing.assert_allclose(
        permuted["denoised"].transpose(*data.dims).values,
        reference["denoised"].values,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        permuted["n_components"].transpose(*_GRID).values,
        reference["n_components"].values,
    )


def test_complex_data_recover_rank_with_total_noise_variance():
    clean, data = _image_series(8, complex_valued=True)
    result = denoise_tpca(
        data, feature_dims=_FEATURES, sample_dims=_GRID, noise_variance=1.0
    )
    assert np.iscomplexobj(result["denoised"].values)
    assert np.all(result["n_components"].values == 2)
    assert _rmse(result["denoised"].values, clean) < 0.6 * _rmse(data.values, clean)


def test_float32_input_returns_float32_and_noise_map_matches_scalar():
    _, data = _image_series(9)
    single = data.astype(np.float32)
    noise_map = xr.full_like(data.isel(repetition_idx=0, drop=True), 1.0)

    scalar = denoise_tpca(
        single, feature_dims=_FEATURES, sample_dims=_GRID, noise_variance=1.0
    )
    mapped = denoise_tpca(
        single, feature_dims=_FEATURES, sample_dims=_GRID, noise_variance=noise_map
    )

    assert scalar["denoised"].dtype == np.float32
    xr.testing.assert_identical(scalar, mapped)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"feature_dims": (), "sample_dims": _GRID}, "feature_dims"),
        ({"feature_dims": _FEATURES, "sample_dims": ("read_idx",)}, "phase_idx"),
        (
            {"feature_dims": _FEATURES, "sample_dims": (*_GRID, "repetition_idx")},
            "both",
        ),
        (
            {
                "feature_dims": _FEATURES,
                "sample_dims": _GRID,
                "patch_shape": {"repetition_idx": 3},
            },
            "not sample",
        ),
        (
            {
                "feature_dims": _FEATURES,
                "sample_dims": _GRID,
                "patch_shape": {"read_idx": 13},
            },
            "between 1",
        ),
        (
            {
                "feature_dims": _FEATURES,
                "sample_dims": _GRID,
                "patch_shape": {"read_idx": 1, "phase_idx": 1},
            },
            "two voxels",
        ),
    ],
)
def test_invalid_dimensions_and_patches_fail_early(kwargs, message):
    _, data = _image_series(10)
    with pytest.raises(ValueError, match=message):
        denoise_mppca(data, **kwargs)


def test_invalid_data_and_noise_variance_fail_early():
    _, data = _image_series(11)
    with pytest.raises(TypeError, match="DataArray"):
        denoise_mppca(data.values, feature_dims=_FEATURES, sample_dims=_GRID)
    with pytest.raises(ValueError, match="NaN"):
        denoise_mppca(
            data.where(data.read_idx > 0), feature_dims=_FEATURES, sample_dims=_GRID
        )
    with pytest.raises(ValueError, match="positive finite"):
        denoise_gpca(
            data, feature_dims=_FEATURES, sample_dims=_GRID, noise_variance=0.0
        )
    with pytest.raises(ValueError, match="not sample"):
        denoise_tpca(
            data,
            feature_dims=_FEATURES,
            sample_dims=_GRID,
            noise_variance=xr.DataArray(np.ones(20), dims="repetition_idx"),
        )
