"""Physical radial filters on invented grids, independent of viewer/scanner data.

Double-precision Fourier and window references use 1e-12 absolute tolerance;
float32/complex64 storage uses 1e-6. Coordinates are in cycles/mm, not radians/mm.
"""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spatial_filtering import (
    SpatialFilterOptions,
    apply_spatial_filter,
    kspace_filter_weights,
    spatial_to_kspace,
)


def _image(shape=(5, 6), dtype=np.complex128):
    values = np.zeros((*shape, 2, 3), dtype=dtype)
    values[shape[0] // 2, shape[1] // 2] = 12
    return xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "echo_idx", "slice_idx"),
        coords={
            "read_idx": ("read_idx", 4 - np.arange(shape[0]) * 2, {"units": "mm"}),
            "phase_idx": ("phase_idx", np.arange(shape[1]) * 3, {"units": "mm"}),
            "echo_idx": ("echo_idx", [0.01, 0.02], {"units": "s"}),
        },
        attrs={"domain": "spatial", "units": "a.u.", "invented": True},
    )


def test_gaussian_uses_physical_radius_and_equal_distances_have_equal_weights():
    options = SpatialFilterOptions(kind="gaussian", sigma_per_mm=0.1)
    actual = kspace_filter_weights([0, 0.1, 0.06], [0, 0, 0.08], options)
    np.testing.assert_allclose(
        actual, [1, np.exp(-0.5), np.exp(-0.5)], rtol=0, atol=1e-12
    )
    with pytest.raises(FrozenInstanceError):
        options.sigma_per_mm = 0.2


def test_fermi_has_explicit_radius_and_transition_without_implicit_normalization():
    options = SpatialFilterOptions(
        kind="fermi", radius_per_mm=0.1, transition_per_mm=0.02
    )
    actual = kspace_filter_weights([0, 0.08, 0.1, 0.12, 1e6], 0, options)
    expected = [1 / (1 + np.exp(-5)), 1 / (1 + np.exp(-1)), 0.5, 1 / (1 + np.e), 0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape", [(5, 6), (6, 5), (5, 5), (6, 6)])
@pytest.mark.parametrize("permuted", [False, True])
def test_spatial_filter_matches_independent_fourier_reference_and_preserves_labels(
    shape, permuted
):
    data = _image(shape)
    if permuted:
        data = data.transpose("slice_idx", "phase_idx", "echo_idx", "read_idx")
    before = data.copy(deep=True)
    axes = tuple(data.get_axis_num(d) for d in ("read_idx", "phase_idx"))
    read = np.fft.fftshift(np.fft.fftfreq(shape[0], 2))
    phase = np.fft.fftshift(np.fft.fftfreq(shape[1], 3))
    window = np.exp(-0.5 * (read[:, None] ** 2 + phase[None, :] ** 2) / 0.1**2)
    weights = xr.DataArray(window, dims=("read_idx", "phase_idx")).broadcast_like(data)
    spectrum = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(data.values, axes=axes), axes=axes), axes=axes
    )
    expected = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(spectrum * weights.values, axes=axes), axes=axes),
        axes=axes,
    )
    result = apply_spatial_filter(data, SpatialFilterOptions(sigma_per_mm=0.1))
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-12)
    assert result.dims == data.dims and result.dtype == data.dtype
    for name in data.coords:
        xr.testing.assert_identical(result[name], data[name])
    assert result.attrs["units"] == "a.u."
    provenance = json.loads(result.attrs["spatial_filter"])
    assert provenance["options"]["sigma_per_mm"] == 0.1
    assert provenance["axes"] == ["read_idx", "phase_idx"]
    xr.testing.assert_identical(data, before)
    np.testing.assert_allclose(
        result.sum(("read_idx", "phase_idx")), 12, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_storage_precision_and_independent_planes(dtype):
    data = _image(dtype=dtype)
    data.values[..., 1, 2] *= 3
    result = apply_spatial_filter(data, SpatialFilterOptions(sigma_per_mm=0.1))
    assert result.dtype == dtype
    np.testing.assert_allclose(
        result.values[..., 1, 2], result.values[..., 0, 0] * 3, rtol=0, atol=1e-6
    )


def test_kspace_coordinates_can_be_nonuniform_descending_and_permuted():
    data = xr.DataArray(
        np.full((3, 2, 4), 2 + 3j),
        dims=("phase_idx", "channel_idx", "read_idx"),
        coords={
            "read_idx": ("read_idx", [0.12, 0.06, 0, -0.09], {"units": "1/mm"}),
            "phase_idx": ("phase_idx", [-0.08, 0, 0.2], {"units": "1/mm"}),
        },
        attrs={"domain": "kspace"},
    )
    result = apply_spatial_filter(data, SpatialFilterOptions(sigma_per_mm=0.1))
    radius_sq = (
        np.array([-0.08, 0, 0.2])[:, None, None] ** 2
        + np.array([0.12, 0.06, 0, -0.09])[None, None, :] ** 2
    )
    np.testing.assert_allclose(
        result.values,
        np.broadcast_to((2 + 3j) * np.exp(-radius_sq / (2 * 0.1**2)), data.shape),
        atol=1e-12,
        rtol=0,
    )
    xr.testing.assert_identical(result.read_idx, data.read_idx)


def test_explicit_spacing_and_units_give_the_same_physical_filter():
    data = _image()
    options = SpatialFilterOptions(sigma_per_mm=0.1)
    reference = apply_spatial_filter(data, options)
    index_data = data.drop_vars(["read_idx", "phase_idx"])
    result = apply_spatial_filter(index_data, options, spatial_spacing_mm=(2, 3))
    np.testing.assert_allclose(result, reference, atol=1e-12, rtol=0)
    converted = data.assign_coords(
        read_idx=data.read_idx / 10, phase_idx=data.phase_idx / 10
    )
    converted.read_idx.attrs["units"] = converted.phase_idx.attrs["units"] = "cm"
    np.testing.assert_allclose(
        apply_spatial_filter(converted, options), reference, atol=1e-12, rtol=0
    )


def test_scanner_grid_modulation_keeps_its_existing_zero_frequency_convention():
    data = _image((5, 6))
    axes = (0, 1)
    # This convention already places k=0 at the central index without fftshift.
    kspace = np.zeros_like(data.values)
    kspace[2, 3] = 30
    data.values = np.fft.fftshift(np.fft.ifftn(kspace, axes=axes), axes=axes)
    data.attrs["csi_spatial_reconstruction"] = "scanner_grid"
    result = apply_spatial_filter(data, SpatialFilterOptions(sigma_per_mm=0.1))
    np.testing.assert_allclose(result, data, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"sigma_per_mm": 0}, "positive"),
        ({"sigma_per_mm": np.inf}, "finite"),
        ({"kind": "unknown", "sigma_per_mm": 0.1}, "kind"),
        ({"kind": "fermi", "radius_per_mm": 0.1}, "transition"),
        ({"sigma_per_mm": 0.1, "radius_per_mm": 0.1}, "radius"),
    ],
)
def test_invalid_settings_fail_at_construction(change, match):
    with pytest.raises(ValueError, match=match):
        SpatialFilterOptions(**change)


def test_unknown_spacing_and_nonuniform_image_grids_fail_explicitly():
    options = SpatialFilterOptions(sigma_per_mm=0.1)
    data = _image()
    with pytest.raises(ValueError, match="spacing"):
        apply_spatial_filter(data.drop_vars(["read_idx", "phase_idx"]), options)
    nonuniform = data.assign_coords(
        read_idx=("read_idx", [0, 1, 2, 4, 5], {"units": "mm"})
    )
    with pytest.raises(ValueError, match="uniform"):
        apply_spatial_filter(nonuniform, options)
    with pytest.raises(ValueError, match="spacing"):
        apply_spatial_filter(data, options, spatial_spacing_mm=(1, 3))


def test_fermi_image_with_singleton_read_axis_and_integer_storage():
    data = xr.DataArray(
        np.array([[0, 0, 10, 0, 0]]),
        dims=("read_idx", "phase_idx"),
        coords={
            "read_idx": ("read_idx", [0], {"resolution_mm": 0}),
            "phase_idx": ("phase_idx", np.arange(5) * 2, {"units": "mm"}),
        },
    )
    options = SpatialFilterOptions(
        kind="fermi", radius_per_mm=0.1, transition_per_mm=0.02
    )
    result = apply_spatial_filter(data, options)
    # A centred delta has constant Fourier amplitude; bins are [-.2,-.1,0,.1,.2].
    window = 1 / (
        1 + np.exp((np.abs(np.array([-0.2, -0.1, 0, 0.1, 0.2])) - 0.1) / 0.02)
    )
    expected = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(10 * window))).real
    np.testing.assert_allclose(result.values[0], expected, atol=1e-12, rtol=0)
    assert result.dtype == np.float64


def test_legacy_kspace_requires_explicit_image_spacing_and_matches_physical_coords():
    data = _image((5, 6)).assign_attrs(domain="kspace")
    options = SpatialFilterOptions(sigma_per_mm=0.1)
    with pytest.raises(ValueError, match="1/mm"):
        apply_spatial_filter(data, options)
    explicit = apply_spatial_filter(data, options, spatial_spacing_mm=(2, 3))
    physical = data.assign_coords(
        read_idx=("read_idx", np.array([-2, -1, 0, 1, 2]) / 10.0, {"units": "1/mm"}),
        phase_idx=("phase_idx", np.arange(-3, 3) / 18.0, {"units": "1/mm"}),
    )
    np.testing.assert_allclose(
        explicit, apply_spatial_filter(physical, options), atol=1e-12, rtol=0
    )


def test_separable_multidimensional_scanner_coordinates_retain_geometry():
    data = _image()
    read, phase = np.meshgrid(
        data.read_idx.values, data.phase_idx.values, indexing="ij"
    )
    data = data.assign_coords(
        read_idx=(("read_idx", "phase_idx"), read, {"units": "mm"}),
        phase_idx=(("read_idx", "phase_idx"), phase, {"units": "mm"}),
    )
    options = SpatialFilterOptions(sigma_per_mm=0.1)
    result = apply_spatial_filter(data, options)
    xr.testing.assert_identical(result.read_idx, data.read_idx)
    xr.testing.assert_identical(result.phase_idx, data.phase_idx)
    data.read_idx.values[0, 1] += 1
    with pytest.raises(ValueError, match="separable"):
        apply_spatial_filter(data, options)


def test_spatial_to_kspace_matches_independent_centred_fourier_reference():
    data = _image((5, 6))
    data = data + 1j * data.copy(data=np.random.default_rng(3).normal(size=data.shape))
    original = data.copy(deep=True)
    actual = spatial_to_kspace(data)
    expected = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(data.values, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    # The odd read axis (5 voxels) has its FOV centre between voxels 2 and 3,
    # as in the reconstruction, so k-space carries a half-voxel phase.
    k_read = np.arange(5) - 2
    expected = expected * np.exp(1j * np.pi * k_read / 5)[:, None, None, None]
    np.testing.assert_allclose(actual.values, expected, rtol=0, atol=1e-12)
    assert actual.dims == data.dims
    assert actual.dtype == np.complex128
    assert actual.attrs == {**data.attrs, "domain": "kspace"}
    np.testing.assert_allclose(
        actual.read_idx, np.fft.fftshift(np.fft.fftfreq(5, d=2.0)), atol=1e-12
    )
    np.testing.assert_allclose(
        actual.phase_idx, np.fft.fftshift(np.fft.fftfreq(6, d=3.0)), atol=1e-12
    )
    assert actual.read_idx.attrs["units"] == "1/mm"
    np.testing.assert_array_equal(actual.echo_idx, data.echo_idx)
    xr.testing.assert_identical(data, original)


def test_spatial_to_kspace_window_lies_where_the_filter_applies_it():
    data = _image((6, 6))
    options = SpatialFilterOptions(kind="gaussian", sigma_per_mm=0.05)
    kspace = spatial_to_kspace(data)
    filtered = spatial_to_kspace(apply_spatial_filter(data, options))
    weights = kspace_filter_weights(
        kspace.read_idx.values[:, None], kspace.phase_idx.values[None, :], options
    )
    np.testing.assert_allclose(
        filtered.values,
        kspace.values * weights[:, :, None, None],
        rtol=0,
        atol=1e-12,
    )


def test_spatial_to_kspace_keeps_the_scanner_grid_convention_and_drops_xyz():
    data = _image((4, 4)).assign_attrs(csi_spatial_reconstruction="scanner_grid")
    data = data.assign_coords(
        x=(("read_idx", "phase_idx"), np.ones((4, 4)), {"units": "mm"})
    )
    actual = spatial_to_kspace(data)
    expected = np.fft.fftn(np.fft.ifftshift(data.values, axes=(0, 1)), axes=(0, 1))
    np.testing.assert_allclose(actual.values, expected, rtol=0, atol=1e-12)
    assert "x" not in actual.coords


def test_spatial_to_kspace_fails_for_kspace_input_missing_axes_and_spacing():
    data = _image()
    with pytest.raises(ValueError, match="image-space data only"):
        spatial_to_kspace(data.assign_attrs(domain="kspace"))
    with pytest.raises(ValueError, match="read_idx and phase_idx"):
        spatial_to_kspace(data.isel(read_idx=0))
    with pytest.raises(ValueError, match="spacing"):
        spatial_to_kspace(data.drop_vars(["read_idx", "phase_idx"]))
    with pytest.raises(TypeError, match="numeric"):
        spatial_to_kspace(data.values)
    explicit = spatial_to_kspace(
        data.drop_vars(["read_idx", "phase_idx"]), spatial_spacing_mm=(2.0, 3.0)
    )
    np.testing.assert_allclose(explicit.values, spatial_to_kspace(data).values)
