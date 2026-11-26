"""Centring of the spatial transforms and Bruker raw zero filling.

The fixtures are synthetic.  An object at the FOV centre has constant
k-space.  ParaVision puts the FOV centre at index ``n / 2`` of an axis with
``n`` voxels (``VisuCorePosition`` of a scan without offsets is
``-n / 2`` voxels): a voxel centre for even ``n``, the border between the
two central voxels for odd ``n``.  The 3D slice-encoding axis takes
the forward transform and is centred at ``(n - 1) / 2``.  The Bruker handlers are minimal
namespaces with symmetric ``PVM_EncSteps1``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.reconstruction.centred_dft import (
    centered_fft,
    centered_ifft,
    image_to_slice_encoding,
    slice_encoding_to_image,
)
from hypermri.reconstruction.strategies.bruker.raw import (
    BrukerMGERawRecoStrategy,
    BrukerRawRecoStrategy,
)

# Sums of at most 16 unit-magnitude float64 terms round at about 1e-15;
# 1e-12 separates rounding from a voxel offset or a linear phase (>= 0.1).
ATOL = 1e-12


def _impulse(n, index):
    image = np.zeros(n, dtype=np.complex128)
    image[index] = 1.0
    return image


def _centred_fft(image):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image)))


def _centred_reconstruction(kspace, axes):
    for axis in axes:
        kspace = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(kspace, axes=axis), axis=axis), axes=axis
        )
    return kspace


def _band_limited_point(n, position):
    """Return the band-limited image of a point ``position`` voxels from index 0."""
    frequency = np.arange(n) - n // 2
    voxel = np.arange(n)
    return np.exp(2j * np.pi * np.outer(voxel - position, frequency) / n).sum(1) / n


@pytest.mark.parametrize("n", [4, 6, 8])
def test_even_lengths_return_fov_centre_at_central_voxel(n):
    image = centered_ifft(np.ones(n, dtype=np.complex128), (0,))

    np.testing.assert_allclose(image, _impulse(n, n // 2), atol=ATOL)


@pytest.mark.parametrize("n", [5, 7, 9])
def test_odd_lengths_return_fov_centre_between_central_voxels(n):
    """Regression test: the centre lies at n / 2, as in ``2dseq``."""
    image = centered_ifft(np.ones(n, dtype=np.complex128), (0,))

    np.testing.assert_allclose(image, _band_limited_point(n, n / 2), atol=ATOL)
    magnitude = np.abs(image)
    assert np.isclose(magnitude[n // 2], magnitude[n // 2 + 1])
    assert magnitude[n // 2] == pytest.approx(magnitude.max())


@pytest.mark.parametrize("n", [5, 7, 8])
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_forward_transform_inverts_the_inverse(n, norm):
    rng = np.random.default_rng(20261122)
    kspace = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))

    image = centered_ifft(kspace, (0,), norm=norm)

    np.testing.assert_allclose(centered_fft(image, (0,), norm=norm), kspace, atol=ATOL)


def test_even_lengths_are_unchanged_bit_for_bit():
    rng = np.random.default_rng(20261002)
    kspace = rng.normal(size=(6, 8)) + 1j * rng.normal(size=(6, 8))
    previous = kspace
    for axis in (0, 1):
        previous = np.fft.ifftshift(
            np.fft.ifft(np.fft.ifftshift(previous, axes=axis), axis=axis), axes=axis
        )

    # fftshift and ifftshift are the same permutation for even lengths, and
    # no half-voxel phase is applied, so the operations are identical.
    np.testing.assert_array_equal(centered_ifft(kspace, (0, 1)), previous)


@pytest.mark.parametrize("n", [4, 6, 20])
def test_even_slice_counts_return_fov_centre_between_central_slices(n):
    """Regression test: 3D slices are centred at (n - 1) / 2, as in ``2dseq``."""
    image = slice_encoding_to_image(np.ones(n, dtype=np.complex128), (0,))

    magnitude = np.abs(image)
    assert np.isclose(magnitude[n // 2 - 1], magnitude[n // 2])
    assert magnitude[n // 2] == pytest.approx(magnitude.max())


@pytest.mark.parametrize("n", [5, 7])
def test_odd_slice_counts_take_the_plain_forward_transform(n):
    rng = np.random.default_rng(20261123)
    kspace = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))

    np.testing.assert_array_equal(
        slice_encoding_to_image(kspace, (0,)), _centred_reconstruction(kspace, (0,))
    )


@pytest.mark.parametrize("n", [5, 8, 20])
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_slice_encoding_round_trip(n, norm):
    rng = np.random.default_rng(20261123)
    kspace = rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))

    image = slice_encoding_to_image(kspace, (1,), norm=norm)

    np.testing.assert_allclose(
        image_to_slice_encoding(image, (1,), norm=norm), kspace, atol=ATOL
    )


def test_input_is_not_modified():
    kspace = np.ones((5, 4), dtype=np.complex128)

    centered_ifft(kspace, (0, 1))
    centered_fft(kspace, (0, 1))
    slice_encoding_to_image(kspace, (0, 1))
    image_to_slice_encoding(kspace, (0, 1))

    np.testing.assert_array_equal(kspace, 1.0)


def _raw_handler(enc_read, enc_phase, matrix_read, matrix_phase):
    steps = np.arange(enc_phase) - enc_phase // 2
    return SimpleNamespace(
        n_repetitions=1,
        enc_matrix=np.array([enc_read, enc_phase, 1]),
        matrix=np.array([matrix_read, matrix_phase, 1]),
        n_receivers=1,
        n_echoes=2,
        spatial_dimension="2D",
        method={"PVM_EncSteps1": steps},
    )


@pytest.mark.parametrize(
    "enc_read,enc_phase,matrix_read,matrix_phase",
    [(4, 4, 8, 6), (3, 5, 6, 8), (5, 3, 9, 7)],
)
def test_raw_zero_filling_is_centred(enc_read, enc_phase, matrix_read, matrix_phase):
    handler = _raw_handler(enc_read, enc_phase, matrix_read, matrix_phase)
    acquired = (np.arange(enc_phase * enc_read) + 1).reshape(enc_phase, enc_read)

    kspace = BrukerRawRecoStrategy().reshape_fid_to_kspace(
        acquired.astype(np.complex64), handler
    )

    read_start = matrix_read // 2 - enc_read // 2
    phase_start = matrix_phase // 2 - enc_phase // 2
    expected = np.zeros((matrix_phase, matrix_read), dtype=np.complex64)
    expected[
        phase_start : phase_start + enc_phase, read_start : read_start + enc_read
    ] = acquired
    assert kspace.shape == (1, 1, 1, matrix_phase, 1, matrix_read)
    assert kspace.dtype == np.complex64
    np.testing.assert_array_equal(kspace[0, 0, 0, :, 0, :], expected)


def test_zero_filled_fov_centre_object_has_no_linear_phase():
    """Constant, symmetric acquired k-space gives a real image peaked at n // 2.

    Odd encoding lengths cover k = -m..m, so the k-space is Hermitian and the
    centred reconstruction is real unless the zero filling moves its centre.
    """
    handler = _raw_handler(5, 3, 8, 6)

    kspace = BrukerRawRecoStrategy().reshape_fid_to_kspace(
        np.ones((3, 5), dtype=np.complex128), handler
    )
    image = _centred_reconstruction(kspace[0, 0, 0, :, 0, :], axes=(0, 1))

    np.testing.assert_allclose(image.imag, 0.0, atol=ATOL)
    assert np.unravel_index(np.argmax(image.real), image.shape) == (3, 4)


def test_mge_zero_filling_is_centred():
    handler = _raw_handler(4, 4, 8, 6)
    acquired = np.ones((4, 2, 4), dtype=np.complex64)  # phase, echo, read

    kspace = BrukerMGERawRecoStrategy().reshape_fid_to_kspace(acquired, handler)

    assert kspace.shape == (1, 1, 1, 6, 2, 1, 8)
    filled = np.abs(kspace[0, 0, 0, :, 0, 0, :]) > 0
    expected = np.zeros((6, 8), dtype=bool)
    expected[1:5, 2:6] = True
    np.testing.assert_array_equal(filled, expected)


def test_raw_zero_filling_rejects_matrix_smaller_than_encoding():
    handler = _raw_handler(4, 4, 2, 4)

    with pytest.raises(ValueError, match="PVM_Matrix and PVM_EncMatrix"):
        BrukerRawRecoStrategy().reshape_fid_to_kspace(
            np.ones((4, 4), dtype=np.complex64), handler
        )
