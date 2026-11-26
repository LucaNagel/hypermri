"""Voxel shifts applied by ``shift_image``.

The fixtures are synthetic unit impulses on small grids.  The k-space input is
built with the transforms of the Bruker raw reconstruction: ``centered_ifft``
for read and phase and ``slice_encoding_to_image`` for the slice axis of 3D
data, with the k-space centre at index ``n // 2``.  The shift is defined in
this reconstructed image.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.centred_dft import (
    centered_fft,
    centered_ifft,
    image_to_slice_encoding,
    slice_encoding_to_image,
)
from hypermri.utils.coordinates.image_shifts import shift_image

# An integer shift of an impulse is exact in exact arithmetic.  The float64
# FFTs and spline evaluations used here round at about 1e-15 per sample, so
# 1e-10 separates rounding from a misplaced or rephased impulse (error >= 1).
ATOL = 1e-10


def _impulse(shape, index):
    image = np.zeros(shape, dtype=np.complex128)
    image[index] = 1.0
    return image


def _image_to_kspace(image, axes):
    return centered_fft(image, axes)


def _kspace_to_image(kspace, axes):
    return centered_ifft(kspace, axes)


def _kspace_xarray(data, **attrs):
    return xr.DataArray(
        data,
        dims=("read_idx", "phase_idx"),
        attrs={"domain": "kspace", "spatial_dimension": "2D", **attrs},
    )


@pytest.mark.parametrize("n", [7, 8])
def test_scipy_image_domain_shift_moves_impulse_once(n):
    shifted = shift_image(
        _impulse((n,), 3), [1.0], shift_method="scipy", domain="image"
    )

    np.testing.assert_allclose(shifted, _impulse((n,), 4), atol=ATOL)


@pytest.mark.parametrize("n", [7, 8])
def test_phase_image_domain_shift_moves_impulse_without_phase(n):
    shifted = shift_image(
        _impulse((n,), 3), [1.0], shift_method="phase", domain="image"
    )

    np.testing.assert_allclose(shifted, _impulse((n,), 4), atol=ATOL)


@pytest.mark.parametrize("shift_vox", [(1.0, 0.0), (1.0, -2.0), (3.0, 1.0)])
def test_phase_kspace_shift_moves_impulse_without_phase(shift_vox):
    shape = (8, 6)
    kspace = _image_to_kspace(_impulse(shape, (3, 2)), axes=(0, 1))

    shifted = shift_image(kspace, list(shift_vox), shift_method="phase")

    expected_index = tuple(
        (i + int(s)) % n for i, s, n in zip((3, 2), shift_vox, shape, strict=True)
    )
    np.testing.assert_allclose(
        _kspace_to_image(shifted, axes=(0, 1)),
        _impulse(shape, expected_index),
        atol=ATOL,
    )


@pytest.mark.parametrize("shape", [(7, 5), (8, 6)])
@pytest.mark.parametrize("shift_method", ["phase", "scipy"])
def test_kspace_and_image_input_shift_the_reconstructed_image_alike(
    shape, shift_method
):
    image = _impulse(shape, (3, 2))

    from_image = shift_image(
        image, [2.0, -1.0], shift_method=shift_method, domain="image"
    )
    from_kspace = shift_image(
        _image_to_kspace(image, (0, 1)), [2.0, -1.0], shift_method=shift_method
    )

    np.testing.assert_allclose(from_image, _impulse(shape, (5, 1)), atol=ATOL)
    np.testing.assert_allclose(
        _kspace_to_image(from_kspace, (0, 1)), _impulse(shape, (5, 1)), atol=ATOL
    )


@pytest.mark.parametrize("n_slices", [4, 5])
def test_phase_shift_of_3d_kspace_moves_the_slice_image_to_higher_indices(n_slices):
    shape = (8, 6, n_slices)
    image = _impulse(shape, (3, 2, 1))
    kspace = image_to_slice_encoding(centered_fft(image, (0, 1)), (2,))
    kspace = xr.DataArray(
        kspace,
        dims=("read_idx", "phase_idx", "slice_idx"),
        attrs={"domain": "kspace", "spatial_dimension": "3D"},
    )

    shifted = shift_image(kspace, [1.0, 2.0, 1.0], shift_method="phase")

    reconstructed = slice_encoding_to_image(centered_ifft(shifted.values, (0, 1)), (2,))
    np.testing.assert_allclose(reconstructed, _impulse(shape, (4, 4, 2)), atol=ATOL)


def test_phase_and_scipy_methods_agree_for_kspace_input():
    shape = (8, 6)
    kspace = _image_to_kspace(_impulse(shape, (3, 2)), axes=(0, 1))

    by_phase = shift_image(kspace, [2.0, 1.0], shift_method="phase")
    by_scipy = shift_image(kspace, [2.0, 1.0], shift_method="scipy")

    np.testing.assert_allclose(by_phase, by_scipy, atol=ATOL)


def test_shift_counter_starts_at_one():
    kspace = _kspace_xarray(_image_to_kspace(_impulse((8, 6), (3, 2)), (0, 1)))

    shifted = shift_image(kspace, [1.0, 0.0])

    assert shifted.attrs["num_image_fov_shift"] == 1


def test_forced_shift_increments_counter_and_unforced_shift_is_skipped():
    kspace = _kspace_xarray(
        _image_to_kspace(_impulse((8, 6), (3, 2)), (0, 1)), num_image_fov_shift=1
    )
    before = kspace.values.copy()

    skipped = shift_image(kspace, [1.0, 0.0])
    np.testing.assert_array_equal(skipped.values, before)
    assert skipped.attrs["num_image_fov_shift"] == 1

    forced = shift_image(kspace, [1.0, 0.0], force_shift=True)
    assert forced.attrs["num_image_fov_shift"] == 2
    np.testing.assert_allclose(
        _kspace_to_image(forced.values, (0, 1)), _impulse((8, 6), (4, 2)), atol=ATOL
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_input_keeps_its_dtype_when_it_contains_zeros(dtype):
    kspace = np.zeros((8, 6), dtype=dtype)
    kspace[4, 3] = 1.0

    shifted = shift_image(kspace, [1.0, 0.0], shift_method="phase")

    assert shifted.dtype == dtype


# The slice package scans below are 2D Head_Prone FLASH scans with a phase
# voxel size of 0.25 mm.  Head_Prone axial L_R and sagittal H_F are in the
# orientation table; Head_Prone axial H_F is not (see
# ``get_orientation_flip_multipliers``).  A positive phase offset moves the
# reconstructed image towards higher phase indices.  All offsets are
# multiples of the voxel size, so every shift is a whole number of voxels.
def _slice_package_scan(**overrides):
    values = {
        "type": "FLASH",
        "spatial_dimension": "2D",
        "patient_pos": "Head_Prone",
        "slice_orientation": np.array(["axial", "sagittal"]),
        "read_orientation": np.array(["L_R", "H_F"]),
        "resolution_mm": np.array([0.25, 0.25]),
        "slices_per_pack": (2, 1),
        "slice_pack_offsets_mm": np.array([[0.0, 0.5, 0.0], [0.0, -0.25, 1.0]]),
        "read_offset_mm": 0.0,
        "phase_offset_mm": 0.5,
        "slice_offset_mm": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _slice_package_kspace(n_slices=3, spatial_dimension="2D"):
    """Return k-space whose image has an impulse at read 3, phase 2 per slice."""
    image = np.zeros((8, 6, n_slices), dtype=np.complex128)
    image[3, 2, :] = 1.0
    return xr.DataArray(
        _image_to_kspace(image, axes=(0, 1)),
        dims=("read_idx", "phase_idx", "slice_idx"),
        attrs={"domain": "kspace", "spatial_dimension": spatial_dimension},
    )


def _impulse_phase_per_slice(kspace):
    image = _kspace_to_image(kspace.values, axes=(0, 1))
    np.testing.assert_allclose(np.abs(image).sum(axis=(0, 1)), 1.0, atol=ATOL)
    read, phase = np.unravel_index(
        np.abs(image).reshape(-1, image.shape[2]).argmax(axis=0), image.shape[:2]
    )
    np.testing.assert_array_equal(read, 3)
    return phase.tolist()


def test_fov_shift_moves_each_slice_package_by_its_own_offset():
    """Regression test: every slice package has its own offset.

    Previously every slice was shifted by the offset of the first package,
    +2 phase voxels, so the impulse of the third slice moved to phase 4.
    """
    shifted = shift_image(
        _slice_package_kspace(), scan=_slice_package_scan(), domain="kspace"
    )

    assert _impulse_phase_per_slice(shifted) == [4, 4, 1]
    assert shifted.attrs["num_image_fov_shift"] == 1


def test_fov_shift_rejects_a_slice_package_outside_the_flip_table():
    """A package without a table entry no longer gets the multipliers +1.

    Previously the second package, Head_Prone axial H_F, was shifted with
    the multipliers +1 without a reviewed table entry.
    """
    scan = _slice_package_scan(
        slice_orientation=np.array(["axial", "axial"]),
        read_orientation=np.array(["L_R", "H_F"]),
        slice_pack_offsets_mm=np.array([[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]),
    )

    with pytest.raises(ValueError, match="read_orientation 'H_F'"):
        shift_image(_slice_package_kspace(), scan=scan)


def test_fov_shift_sign_applies_to_every_slice_package():
    shifted = shift_image(
        _slice_package_kspace(), scan=_slice_package_scan(), fov_shift_sign=-1.0
    )

    assert _impulse_phase_per_slice(shifted) == [0, 0, 3]


@pytest.mark.parametrize("slices_per_pack", [(3,), None])
def test_fov_shift_of_one_slice_package_moves_every_slice(slices_per_pack):
    """A scan with one package keeps the shift of its first package offsets."""
    scan = _slice_package_scan(slices_per_pack=slices_per_pack)
    if slices_per_pack is None:
        del scan.slices_per_pack
        del scan.slice_pack_offsets_mm

    shifted = shift_image(_slice_package_kspace(), scan=scan)

    assert _impulse_phase_per_slice(shifted) == [4, 4, 4]


def test_fov_shift_rejects_slices_that_differ_from_the_slice_packages():
    scan = _slice_package_scan(slices_per_pack=(2, 2))

    with pytest.raises(ValueError, match="slice_idx has 3 slices"):
        shift_image(_slice_package_kspace(), scan=scan)


def test_fov_shift_rejects_3d_data_with_several_slice_packages():
    kspace = _slice_package_kspace(spatial_dimension="3D")

    with pytest.raises(ValueError, match="2D data only"):
        shift_image(kspace, scan=_slice_package_scan(spatial_dimension="3D"))


# The spline shift of an xarray input acts on the named read, phase and
# slice axes.  The fixtures are impulses on an 8 x 6 grid; the expected
# positions are exact for whole-voxel shifts up to the rounding of the
# spline (ATOL).
@pytest.mark.parametrize("shift_method", ["phase", "scipy"])
def test_xarray_shift_follows_the_named_axes_for_phase_before_read(shift_method):
    image = xr.DataArray(
        _impulse((6, 8), (2, 3)),
        dims=("phase_idx", "read_idx"),
        attrs={"domain": "image", "spatial_dimension": "2D"},
    )

    shifted = shift_image(image, [1.0, 0.0], shift_method=shift_method)

    np.testing.assert_allclose(shifted.values, _impulse((6, 8), (2, 4)), atol=ATOL)


@pytest.mark.parametrize("shift_method", ["phase", "scipy"])
def test_xarray_shift_leaves_further_dimensions_untouched(shift_method):
    data = np.stack([_impulse((8, 6), (3, 2)), 2 * _impulse((8, 6), (5, 1))], axis=-1)
    image = xr.DataArray(
        data,
        dims=("read_idx", "phase_idx", "slice_idx"),
        attrs={"domain": "image", "spatial_dimension": "2D"},
    )

    shifted = shift_image(image, [1.0, 1.0], shift_method=shift_method)

    expected = np.stack([_impulse((8, 6), (4, 3)), 2 * _impulse((8, 6), (6, 2))], -1)
    np.testing.assert_allclose(shifted.values, expected, atol=ATOL)


def test_scipy_shift_wraps_content_across_the_border_periodically():
    shifted = shift_image(
        _impulse((6,), 1), [-1.0], shift_method="scipy", domain="image"
    )

    np.testing.assert_allclose(shifted, _impulse((6,), 0), atol=ATOL)


def test_scipy_shift_wraps_content_across_the_upper_border_periodically():
    shifted = shift_image(
        _impulse((8,), 7), [1.0], shift_method="scipy", domain="image"
    )

    np.testing.assert_allclose(shifted, _impulse((8,), 0), atol=ATOL)


def test_scipy_shift_wraps_a_kspace_input_with_the_same_period():
    kspace = _image_to_kspace(_impulse((8, 6), (7, 1)), (0, 1))

    shifted = shift_image(kspace, [1.0, -1.0], shift_method="scipy", domain="kspace")

    expected = _image_to_kspace(_impulse((8, 6), (0, 0)), (0, 1))
    np.testing.assert_allclose(shifted, expected, atol=ATOL)


def test_scipy_shift_of_a_smooth_periodic_image_matches_the_phase_method():
    # A band-limited periodic image, so the exact shift is the phase method.
    # Cubic spline interpolation of this 32 x 32 image deviates from it by
    # 3e-5 of the peak (measured); the tolerance leaves a factor of 30.
    n = 32
    x = np.arange(n)
    image = np.exp(
        np.cos(2 * np.pi * x[:, None] / n) + np.sin(2 * np.pi * x[None, :] / n)
    ).astype(np.complex128)

    exact = shift_image(image, [3.4, -2.6], shift_method="phase", domain="image")
    spline = shift_image(image, [3.4, -2.6], shift_method="scipy", domain="image")

    assert np.abs(spline - exact).max() < 1e-3 * np.abs(exact).max()


def test_xarray_scipy_shift_of_a_3d_scan_uses_all_three_named_axes():
    image = xr.DataArray(
        _impulse((4, 8, 6), (1, 3, 2)),
        dims=("slice_idx", "read_idx", "phase_idx"),
        attrs={"domain": "image", "spatial_dimension": "3D"},
    )

    shifted = shift_image(image, [1.0, 2.0, 1.0], shift_method="scipy")

    np.testing.assert_allclose(
        shifted.values, _impulse((4, 8, 6), (2, 4, 4)), atol=ATOL
    )
