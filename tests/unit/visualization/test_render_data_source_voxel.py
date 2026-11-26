"""Check that the ``plot`` spectrum comes from the voxel under the crosshair.

Display preprocessing reverses ``read_idx`` when its coordinates descend and,
without ``read_idx``, reverses ``slice_idx`` when its coordinates ascend. The
selected voxel indices in ``PlotState`` are source indices, so the spectrum and
its data overlays must be taken from the same source voxel. All arrays are
invented: each voxel's lines are scaled by a factor that encodes its position,
so a mirrored voxel gives a different spectrum. No scanner data or metadata are
used.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.visualization.render_data import PlotDataModel
from hypermri.visualization.state import PlotState


def _lines(n_spec=64):
    time_s = np.arange(n_spec) * 2e-4
    return sum(
        amplitude * np.exp(2j * np.pi * freq_hz * time_s - time_s / 0.03)
        for freq_hz, amplitude in ((-800, 1.0), (300, 0.4), (1200, 0.2))
    )


def _csi(spatial_sizes):
    """Return invented CSI data whose voxel scale encodes every spatial index."""
    dims = (*spatial_sizes, "repetition_idx", "channel_idx", "acquisition_idx")
    shape = (*spatial_sizes.values(), 2, 1, 64)
    scale = np.ones(shape[:-1])
    for axis, size in enumerate(shape[:-1]):
        index_shape = [1] * len(shape[:-1])
        index_shape[axis] = size
        scale = scale + 10**axis * np.arange(size).reshape(index_shape)
    values = (scale[..., None] * _lines()).astype(np.complex64)
    return xr.DataArray(
        values,
        dims=dims,
        attrs={"units": "a.u.", "signal_domain": "temporal", "invented_fixture": True},
    )


def _payload(data, indices, rotation, overlays=None):
    state = PlotState()
    state.spatial.rotation = rotation
    state.spectral.spec_part = "Real"
    state.spatial.slice_indices = {
        **indices,
        "repetition_idx": 1,
        "channel_idx": 0,
        "acquisition_idx": 10,
    }
    model = PlotDataModel(SimpleNamespace(primary_data=data, is_spectroscopic=True))
    return model.get_render_payload(data, state, overlays=overlays)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("read_index", [0, 1, 4])
def test_spectrum_follows_source_voxel_with_descending_read_coordinates(
    rotation, read_index
):
    plain = _csi({"read_idx": 5, "phase_idx": 7, "slice_idx": 1})
    descending = plain.assign_coords(
        read_idx=10.0 - 2.5 * np.arange(plain.sizes["read_idx"])
    )
    indices = {"read_idx": read_index, "phase_idx": 2, "slice_idx": 0}
    reference = _payload(plain, indices, rotation)
    payload = _payload(descending, indices, rotation)
    # Same invented float32 values on the same path: only the voxel can differ.
    np.testing.assert_allclose(payload.spectral_y, reference.spectral_y, rtol=1e-6)


@pytest.mark.parametrize("slice_index", [0, 2])
def test_spectrum_follows_source_voxel_with_ascending_slice_coordinates(
    slice_index,
):
    plain = _csi({"phase_idx": 4, "slice_idx": 3})
    ascending = plain.assign_coords(slice_idx=np.array([-5.0, 0.0, 5.0]))
    indices = {"phase_idx": 1, "slice_idx": slice_index}
    reference = _payload(plain, indices, rotation=0)
    payload = _payload(ascending, indices, rotation=0)
    np.testing.assert_allclose(payload.spectral_y, reference.spectral_y, rtol=1e-6)


def test_data_overlay_follows_source_voxel_with_descending_read_coordinates():
    plain = _csi({"read_idx": 5, "phase_idx": 7, "slice_idx": 1})
    descending = plain.assign_coords(
        read_idx=10.0 - 2.5 * np.arange(plain.sizes["read_idx"])
    )
    indices = {"read_idx": 1, "phase_idx": 2, "slice_idx": 0}
    reference = _payload(plain, indices, rotation=0, overlays={"copy": plain})
    payload = _payload(descending, indices, rotation=0, overlays={"copy": descending})
    np.testing.assert_allclose(
        payload.overlays["copy"], reference.overlays["copy"], rtol=1e-6
    )
    np.testing.assert_allclose(payload.overlays["copy"], payload.spectral_y, rtol=1e-6)


def test_grid_coordinates_on_a_selected_plane_keep_source_positions():
    # Scanner data may carry read_idx/phase_idx/slice_idx as voxel-centre
    # positions in mm on the whole (read, phase, slice) grid. Selecting one
    # slice drops a dimension of these coordinates; that must not count as a
    # changed voxel order.
    plain = _csi({"read_idx": 5, "phase_idx": 7, "slice_idx": 2})
    read_mm, phase_mm, slice_mm = np.meshgrid(
        -6.0 + 3.0 * np.arange(5),
        -9.0 + 3.0 * np.arange(7),
        1.5 * np.arange(2),
        indexing="ij",
    )
    grid_dims = ("read_idx", "phase_idx", "slice_idx")
    gridded = plain.assign_coords(
        read_idx=(grid_dims, read_mm),
        phase_idx=(grid_dims, phase_mm),
        slice_idx=(grid_dims, slice_mm),
    )
    indices = {"read_idx": 1, "phase_idx": 2, "slice_idx": 1}
    state = PlotState()
    state.spectral.spec_part = "Real"
    state.spatial.slice_indices = {
        **indices,
        "repetition_idx": 1,
        "channel_idx": 0,
        "acquisition_idx": 10,
    }
    model = PlotDataModel(SimpleNamespace(primary_data=gridded, is_spectroscopic=True))
    payload = model.get_render_payload(gridded, state, include_curve_block=True)
    assert payload.curve_block is not None
    row_dim, column_dim = payload.spatial_dims
    np.testing.assert_array_equal(
        np.sort(payload.spatial_source_indices[row_dim]),
        np.arange(gridded.sizes[row_dim]),
    )
    reference = _payload(plain, indices, rotation=0)
    np.testing.assert_allclose(payload.spectral_y, reference.spectral_y, rtol=1e-6)
