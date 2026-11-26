"""Invented scanner geometry: independent voxel positions, no measured data."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.strategies.bruker.geometry import _attach_image_geometry


def _handler(shape=(2, 3, 1), *, packages=(1,), missing=False):
    visu = {
        "VisuCoreSize": [shape[0], shape[1]],
        "VisuCoreExtent": [4.0, 9.0],
        "VisuCorePosition": [[10.0, 20.0, 30.0]],
        "VisuCoreOrientation": [np.eye(3).ravel()],
        "VisuCoreFrameThickness": 5.0,
    }
    if missing:
        visu.pop("VisuCorePosition")
    handler = SimpleNamespace(
        slices_per_pack=packages,
        acquisition_matrix=np.eye(3),
        spatial_dimension="2D",
        n_acquisition=1,
        method={},
        resolution_mm=[99.0, 99.0, 99.0],
        reco_transposition=0,
        visucore_position=[[0.0, 0.0, 0.0]],
        visucore_orientation=[np.eye(3).ravel()],
        visucore_size=[2, 3],
        visucore_extent=[2.0, 3.0],
        visucore_thickness=1.0,
    )
    handler.load_reconstruction_parameters = lambda _: ({"RECO_transposition": 0}, visu)
    return handler


@pytest.mark.parametrize("shape", [(2, 3, 1), (1, 3, 1), (2, 1, 1)])
def test_selected_mm_axes_and_extents_use_selected_geometry(shape):
    """Voxel centres and half-voxel extents agree to 1e-12 mm."""
    values = np.arange(np.prod(shape)).reshape(shape)
    data = xr.DataArray(values, dims=("read_idx", "phase_idx", "slice_idx"))
    handler = _handler(shape)
    result = _attach_image_geometry(data, handler, recon_num=2)
    np.testing.assert_array_equal(result.values, values)
    for dim, size, origin, extent in zip(
        data.dims, shape, [10.0, 20.0, 30.0], [4.0, 9.0, 5.0], strict=True
    ):
        spacing = extent / size
        expected = origin + np.arange(size) * spacing
        selector = {other: 0 for other in data.dims if other != dim}
        np.testing.assert_allclose(
            result[dim].isel(selector), expected, atol=1e-12, rtol=0
        )
        assert result[dim].attrs["unit"] == "mm"
        np.testing.assert_allclose(
            result[dim].attrs["extent_mm"],
            [origin - spacing / 2, expected[-1] + spacing / 2],
            atol=1e-12,
            rtol=0,
        )
    assert handler.resolution_mm == [99.0, 99.0, 99.0]
    assert result.x.dims == data.dims


def test_selected_missing_geometry_does_not_reuse_acquisition_coordinates():
    data = xr.DataArray(np.ones((2, 3, 1)), dims=("read_idx", "phase_idx", "slice_idx"))
    data = data.assign_coords(
        read_idx=(data.dims, np.zeros(data.shape), {"unit": "mm"})
    )
    result = _attach_image_geometry(data, _handler(missing=True), recon_num=2)
    assert "x" not in result.coords
    assert result.read_idx.attrs["unit"] == "index"
    np.testing.assert_array_equal(result.read_idx, [0, 1])


def test_multiple_packages_keep_indices_and_physical_xyz():
    data = xr.DataArray(np.ones((2, 3, 1)), dims=("read_idx", "phase_idx", "slice_idx"))
    result = _attach_image_geometry(data, _handler(packages=(1, 1)), recon_num=2)
    assert all(result[dim].attrs["unit"] == "index" for dim in data.dims)
    assert all(result[name].attrs["unit"] == "mm" for name in ("x", "y", "z"))


def test_selected_orientation_controls_physical_positions_and_directions():
    handler = _handler()
    reco, visu = handler.load_reconstruction_parameters(2)
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    visu["VisuCoreOrientation"] = [rotation.ravel()]
    handler.load_reconstruction_parameters = lambda _: (reco, visu)
    data = xr.DataArray(np.ones((2, 3, 1)), dims=("read_idx", "phase_idx", "slice_idx"))
    data.attrs.update(
        slice_orientation="axial",
        read_orientation="L_R",
        slice_spacing_mm=99.0,
        slice_thickness_mm=99.0,
        n_slices=99,
    )
    result = _attach_image_geometry(data, handler, recon_num=2)
    np.testing.assert_allclose(
        result.x[:, :, 0], [[10.0, 7.0, 4.0], [10.0, 7.0, 4.0]], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        result.y[:, :, 0], [[20.0, 20.0, 20.0], [22.0, 22.0, 22.0]], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        result.attrs["read_direction"], [0, 1, 0], atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(handler.acquisition_matrix, np.eye(3))
    assert "read_orientation" not in result.attrs
    assert result.attrs["acquisition_read_orientation"] == "L_R"
    assert result.attrs["n_slices"] == 1
    assert result.attrs["slice_spacing_mm"] == 5.0
    assert result.attrs["slice_thickness_mm"] == 5.0
    assert data.attrs["slice_thickness_mm"] == 99.0


def test_fallback_spatial_sizes_follow_selected_reconstruction():
    from hypermri.reconstruction.strategies.bruker.seq2d import Brukerseq2dRecoStrategy

    handler = _handler()
    handler.matrix = [16, 16, 5]
    handler.n_slices = 5
    reco, visu = handler.load_reconstruction_parameters(2)
    dims, sizes = Brukerseq2dRecoStrategy._infer_fallback_dims(
        handler, reco, visu, np.zeros((2, 3, 5)), "flash", "FLASH"
    )
    assert dims == ["read_idx", "phase_idx", "slice_idx"]
    assert sizes["read_idx"] == 2
    assert sizes["phase_idx"] == 3
