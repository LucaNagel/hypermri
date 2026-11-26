"""Marchenko-Pastur component selection of the global ``pca_denoise``.

The reference edge is Eq. 9 of Henriques et al., Imaging Neuroscience 1
(2023): for an ``M x N`` matrix of ``M`` pixel samples and ``N`` redundant
measurements, pure noise of variance ``sigma^2`` gives covariance eigenvalues
below ``sigma^2 * (1 + sqrt(N / M))^2``. The fixtures are seeded Gaussian
low-rank matrices; for each seed the largest noise eigenvalue lies below that
asymptotic edge and every signal eigenvalue above it, so the selected rank is
exact. Projections are compared with ``rtol=1e-10`` because both sides run the
same floating-point decomposition.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.denoising import pca_denoise


def _low_rank_with_noise(n_read, n_phase, n_repetitions, rank, amplitude, seed):
    """Return a labelled noisy rank-``rank`` matrix and its noise-free signal."""
    rng = np.random.default_rng(seed)
    n_pixels = n_read * n_phase
    clean = amplitude * (
        rng.standard_normal((n_pixels, rank))
        @ rng.standard_normal((rank, n_repetitions))
    )
    noisy = clean + rng.standard_normal((n_pixels, n_repetitions))
    data = xr.DataArray(
        noisy.reshape(n_read, n_phase, n_repetitions),
        dims=("read_idx", "phase_idx", "repetition_idx"),
    )
    return data, clean.reshape(n_read, n_phase, n_repetitions)


def _covariance_eigenvalues(data):
    """Eigenvalues of ``X^H X / M`` for pixel rows and centred repetition columns."""
    matrix = data.values.reshape(-1, data.sizes["repetition_idx"])
    centred = matrix - matrix.mean(axis=0)
    covariance = centred.conj().T @ centred / centred.shape[0]
    return np.sort(np.linalg.eigvalsh(covariance))[::-1]


def _upper_edge(noise_variance, n_pixels, n_repetitions):
    """Marchenko-Pastur upper edge with ``gamma = n_repetitions / n_pixels``."""
    return noise_variance * (1 + np.sqrt(n_repetitions / n_pixels)) ** 2


@pytest.mark.parametrize(
    ("shape", "rank", "amplitude", "seed"),
    [
        ((32, 32, 20), 3, 1.0, 0),
        ((32, 32, 20), 3, 0.5, 0),
        ((16, 16, 16), 4, 1.0, 3),
        ((8, 8, 40), 2, 1.0, 2),
    ],
)
def test_threshold_keeps_exactly_the_components_above_the_edge(
    shape, rank, amplitude, seed
):
    data, _ = _low_rank_with_noise(*shape, rank, amplitude, seed)
    n_pixels = shape[0] * shape[1]
    edge = _upper_edge(1.0, n_pixels, shape[2])
    eigenvalues = _covariance_eigenvalues(data)
    assert np.sum(eigenvalues > edge) == rank

    result = pca_denoise(data, noise_estimation_method="manual", noise_threshold=1.0)
    expected = pca_denoise(data, components=rank)

    np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)


def test_stated_edge_for_the_reported_reproduction():
    # 32 x 32 pixels and 20 repetitions: gamma = 20 / 1024.
    assert _upper_edge(1.0, 1024, 20) == pytest.approx(1.2990, abs=5e-5)


def test_threshold_denoising_reduces_error_against_the_noise_free_signal():
    data, clean = _low_rank_with_noise(32, 32, 20, 3, 1.0, 0)

    result = pca_denoise(data, noise_estimation_method="manual", noise_threshold=1.0)

    raw_rmse = np.sqrt(np.mean((data.values - clean) ** 2))
    denoised_rmse = np.sqrt(np.mean((result.values - clean) ** 2))
    # Rank-3 truncation of 20 repetitions keeps about 3/20 of the noise energy
    # plus the pixel means: RMSE 0.389 against 0.994 for this seed.
    assert denoised_rmse < 0.45 * raw_rmse


def test_pure_noise_keeps_the_documented_minimum_of_one_component():
    data, _ = _low_rank_with_noise(32, 32, 20, 0, 0.0, 1)
    assert np.all(_covariance_eigenvalues(data) < _upper_edge(1.0, 1024, 20))

    result = pca_denoise(data, noise_estimation_method="manual", noise_threshold=1.0)
    expected = pca_denoise(data, components=1)

    np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)


def test_explicit_components_match_an_independent_projection():
    data, _ = _low_rank_with_noise(16, 16, 16, 4, 1.0, 3)
    matrix = data.values.reshape(-1, 16)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    basis = right[:4].conj().T
    reference = (centred @ basis @ basis.conj().T + mean).reshape(16, 16, 16)

    result = pca_denoise(data, components=4)

    np.testing.assert_allclose(result.values, reference, rtol=1e-10, atol=1e-12)
