"""Spatial Fourier transform of the Bruker raw reconstruction.

The k-space is seeded complex Gaussian noise, not scanner data, so that every
voxel order is distinguishable.  The expected read and phase images are the
explicit DFT onto the ``VisuCore`` voxel grid: voxel ``j`` of ``n`` lies at
``j - n / 2`` voxels from the FOV centre, and k-space index ``m`` holds the
spatial frequency ``m - n // 2``.  For even ``n`` this equals the pipeline
previously, the centred forward FFT ``fftshift(fft(fftshift(k)))``,
``np.roll(-1)`` and a reversal of the axis, which reproduced the ``2dseq``
voxel order of the 109 local FLASH, RARE and MGE scans with a raw
reconstruction (``tests/unit/reconstruction/test_bruker_raw_versus_2dseq.py``
checks it on synthetic scans).  For odd ``n`` that pipeline lies one half
voxel beside ``2dseq``; the DFT onto the voxel grid matches ``2dseq`` of three
local RARE scans with 147, 213 and 133 voxels at correlation 1.0000 at zero
shift (22 November 2026).

The transforms run in complex128.  For matrices of at most 17 points their
rounding error is of order 1e-15 relative to the image maximum; ``ATOL_RELATIVE``
leaves three orders of magnitude of margin, while a half-voxel shift or a
reversed axis changes the noise images by order one.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.strategies.bruker.raw import BrukerRawRecoStrategy

ATOL_RELATIVE = 1e-12
SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


def _kspace(shape, dims=SPATIAL_DIMS, seed=7):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return xr.DataArray(values, dims=dims)


def _scan(patient_pos, slice_orientation, read_orientation, spatial_dimension="2D"):
    return SimpleNamespace(
        patient_pos=patient_pos,
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
        spatial_dimension=spatial_dimension,
        git_commit_hash=None,
    )


def _forward(values, axis):
    shifted = np.fft.fftshift(values, axes=axis)
    return np.fft.fftshift(np.fft.fft(shifted, axis=axis), axes=axis)


def _pre_d7_read_and_phase(values, read_axis=0, phase_axis=1):
    """Return the earlier read and phase transform (``2dseq`` for even sizes)."""
    for axis in (read_axis, phase_axis):
        values = np.flip(np.roll(_forward(values, axis), -1, axis=axis), axis=axis)
    return values


def _dft_onto_voxel_grid(values, axes=(0, 1)):
    """Return the explicit DFT onto the ``VisuCore`` grid, unnormalized."""
    for axis in axes:
        n = values.shape[axis]
        position = np.arange(n) - n / 2
        frequency = np.arange(n) - n // 2
        matrix = np.exp(2j * np.pi * np.outer(position, frequency) / n)
        values = np.moveaxis(
            np.tensordot(matrix, np.moveaxis(values, axis, 0), axes=(1, 0)), 0, axis
        )
    return values


def _reconstruct(kspace, scan, axes_names):
    return BrukerRawRecoStrategy().set_data(
        kspace_xarray=kspace, axes_names=axes_names, handler=scan
    )


def _assert_close(actual, expected):
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=ATOL_RELATIVE * np.abs(expected).max()
    )


@pytest.mark.parametrize(
    "shape",
    [(16, 16, 3), (16, 15, 3), (15, 17, 3)],
    ids=["even", "odd-phase", "odd-read-and-phase"],
)
@pytest.mark.parametrize(
    "orientation",
    [
        ("Head_Prone", "axial", "L_R"),
        ("Head_Supine", "coronal", "H_F"),
        ("Foot_Prone", "sagittal", "A_P"),
    ],
    ids="-".join,
)
def test_read_and_phase_land_on_the_2dseq_voxel_grid(shape, orientation):
    """Read and phase take the DFT onto the ``VisuCore`` grid for even and odd sizes.

    Regression test: odd axes were half a voxel beside ``2dseq``.
    """
    kspace = _kspace(shape)

    image = _reconstruct(kspace, _scan(*orientation), ["read_idx", "phase_idx"])

    assert image.dims == SPATIAL_DIMS
    _assert_close(image.to_numpy(), _dft_onto_voxel_grid(kspace.to_numpy()))


def test_even_sizes_keep_the_pre_d7_voxel_order():
    """For even sizes the voxel grid DFT is the pipeline that matched ``2dseq``."""
    kspace = _kspace((16, 16, 3)).to_numpy()

    _assert_close(_dft_onto_voxel_grid(kspace), _pre_d7_read_and_phase(kspace))


@pytest.mark.parametrize(
    "orientation",
    [
        pytest.param(
            ("Head_Prone", "axial", "H_F"), id="read-orientation-not-in-table"
        ),
        pytest.param(("Head_Left", "axial", "L_R"), id="patient-position-not-in-table"),
        pytest.param(
            ("Head_Prone", np.array(["axial", "axial"]), np.array(["L_R", "H_F"])),
            id="slice-packages-disagree",
        ),
    ],
)
def test_orientation_outside_the_flip_table_is_rejected(orientation):
    """Regression test: the raw reconstruction does not guess an orientation.

    The voxel order of read and phase does not depend on the flip
    table, so previously these combinations reconstructed with the
    multipliers (1, 1, 1), whose slice entry is the only one the
    reconstruction reads.  Now the slice entry is unknown, and the
    reconstruction raises.
    """
    kspace = _kspace((16, 16, 3))

    with pytest.raises(ValueError, match="get_orientation_flip_multipliers"):
        _reconstruct(kspace, _scan(*orientation), ["read_idx", "phase_idx"])


def _dft_onto_slice_grid(values, axis):
    """Return the forward DFT onto the 3D slice grid, centred at ``(n - 1) / 2``."""
    n = values.shape[axis]
    position = np.arange(n) - (n - 1) / 2
    frequency = np.arange(n) - n // 2
    matrix = np.exp(-2j * np.pi * np.outer(position, frequency) / n)
    return np.moveaxis(
        np.tensordot(matrix, np.moveaxis(values, axis, 0), axes=(1, 0)), 0, axis
    )


@pytest.mark.parametrize("n_slices", [6, 5])
@pytest.mark.parametrize(
    ("orientation", "reverse_slices"),
    [
        pytest.param(("Head_Supine", "axial", "L_R"), False, id="head-supine-axial"),
        pytest.param(("Head_Prone", "coronal", "L_R"), False, id="head-prone-coronal"),
    ],
)
def test_3d_slice_axis_lands_on_the_2dseq_slice_grid(
    orientation, reverse_slices, n_slices
):
    """The slice axis takes the forward DFT onto the slice grid of ``2dseq``.

    The forward direction follows the Fourier convention: with the inverse transform
    two local 3D FLASH scans correlated with ``2dseq`` at only 0.53 and 0.58.
    The slice grid is centred at ``(n - 1) / 2``; regression test:
    the plain forward FFT put two local 3D bSSFP scans with 20 slices half a
    slice beside ``2dseq`` (correlation 0.62 and 0.53 at zero shift, 1.0000
    on the slice grid, 23 November 2026).  The odd slice count has no local
    scanner reference.  Regression test: the raw reconstruction
    reversed the slices of Head_Prone coronal 3D scans.  A local Head_Prone
    coronal H_F bSSFP scan correlates with its ``2dseq`` at 0.895 at zero
    shift without this reversal and at 0.044 with it.
    """
    kspace = _kspace((16, 15, n_slices))

    image = _reconstruct(
        kspace, _scan(*orientation, spatial_dimension="3D"), SPATIAL_DIMS
    )

    expected = _dft_onto_slice_grid(_dft_onto_voxel_grid(kspace.to_numpy()), axis=2)
    if reverse_slices:
        expected = np.flip(expected, axis=2)
    _assert_close(image.to_numpy(), expected)


def test_untransformed_read_and_phase_keep_their_order():
    """Regression test: only transformed axes change their voxel order.

    A spectroscopic reconstruction transforms no spatial axis, and its
    ``phase_idx`` may hold echoes.  Previously a Head_Prone axial L_R scan
    reversed this axis although it was never transformed.
    """
    kspace = _kspace((8, 1, 4, 1), dims=("acquisition_idx", *SPATIAL_DIMS))
    scan = _scan("Head_Prone", "axial", "L_R", spatial_dimension=None)

    image = _reconstruct(kspace, scan, [])

    np.testing.assert_array_equal(image.to_numpy(), kspace.to_numpy())


def test_coordinates_stay_with_their_voxels(caplog):
    """Index coordinates and their attributes pass through unchanged.

    The scan has no ``VisuCore`` geometry, so no ``x``, ``y`` and ``z`` are
    attached and the reason is logged.
    """
    read_mm = np.linspace(-7.5, 7.5, 16)
    phase_mm = np.linspace(-3.5, 3.5, 15)
    kspace = _kspace((16, 15, 1)).assign_coords(
        read_idx=("read_idx", read_mm, {"unit": "mm"}),
        phase_idx=("phase_idx", phase_mm, {"unit": "mm"}),
    )

    with caplog.at_level(logging.WARNING, logger="hypermri.utils.coordinates.grids"):
        image = _reconstruct(
            kspace, _scan("Head_Supine", "axial", "L_R"), ["read_idx", "phase_idx"]
        )

    np.testing.assert_array_equal(image["read_idx"], read_mm)
    np.testing.assert_array_equal(image["phase_idx"], phase_mm)
    assert image["read_idx"].attrs == {"unit": "mm"}
    assert not {"x", "y", "z"} & set(image.coords)
    assert caplog.records


def test_non_spatial_axes_are_rejected():
    """``set_data`` transforms only read, phase and slice and names other axes."""
    kspace = _kspace((8, 8, 1))

    with pytest.raises(ValueError, match=r"set_data.*\['acquisition_idx'\]"):
        BrukerRawRecoStrategy().set_data(
            kspace_xarray=kspace.rename(slice_idx="acquisition_idx"),
            axes_names=["read_idx", "acquisition_idx"],
            handler=_scan("Head_Prone", "axial", "L_R"),
        )


def test_input_kspace_is_not_modified():
    """The image is a new array; the k-space values and attributes stay."""
    kspace = _kspace((8, 8, 1))
    kspace.attrs["domain"] = "kspace"
    before = kspace.copy(deep=True)

    _reconstruct(kspace, _scan("Head_Prone", "axial", "L_R"), ["read_idx", "phase_idx"])

    xr.testing.assert_identical(kspace, before)
