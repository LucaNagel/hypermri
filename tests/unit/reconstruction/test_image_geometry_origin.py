"""Image-origin regression controls with invented coronal voxel centres and samples.

The two-slice grid has read/phase/slice spacings 1/2/2 mm. Read and phase
positions agree with the method-derived RPS axes; the slice origin has the
opposite sign. No measured metadata or scanner samples are copied here.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.consistency import check_reconstruction_geometry
from hypermri.reconstruction.strategies.bruker.geometry import (
    _attach_image_geometry,
    _physical_slice_axis,
    _same_transposition,
)
from hypermri.utils.coordinates.grids import get_rps_axes

SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
POSITION_ATOL_MM = 1e-3


def _coronal_handler(offset_mm, transposition):
    read = [0.0, 0.0, 1.0]
    phase = [1.0, 0.0, 0.0]
    slice_ = [0.0, 1.0, 0.0]
    orientation = [read, phase, slice_] if transposition == 0 else [phase, read, slice_]
    size = [4, 3, 2] if transposition == 0 else [3, 4, 2]
    extent = [4.0, 6.0, 4.0] if transposition == 0 else [6.0, 4.0, 4.0]
    visu = {
        "VisuCoreSize": size,
        "VisuCoreExtent": extent,
        "VisuCorePosition": [[-3.0, offset_mm - 1.0, -2.0]],
        "VisuCoreOrientation": [np.asarray(orientation).ravel()],
        "VisuCoreFrameThickness": 2.0,
    }
    handler = SimpleNamespace(
        matrix=np.array([4, 3, 2]),
        fov_mm=np.array([4.0, 6.0, 4.0]),
        resolution_mm=np.array([1.0, 2.0, 2.0]),
        slices_per_pack=[2],
        n_slices=2,
        spatial_dimension="3D",
        n_acquisition=1,
        acquisition_matrix=np.asarray([read, phase, slice_]),
        method={},
        read_offset_mm=0.0,
        phase_offset_mm=0.0,
        slice_offset_mm=-offset_mm,
        reco_transposition=np.array([transposition]),
        visucore_size=visu["VisuCoreSize"],
        visucore_extent=visu["VisuCoreExtent"],
        visucore_position=visu["VisuCorePosition"],
        visucore_orientation=visu["VisuCoreOrientation"],
        visucore_thickness=visu["VisuCoreFrameThickness"],
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    handler.load_reconstruction_parameters = lambda _: (
        {"RECO_transposition": transposition},
        visu,
    )
    return handler, visu


def _image():
    return xr.DataArray(
        np.arange(24, dtype=np.float32).reshape(4, 3, 2), dims=SPATIAL_DIMS
    )


@pytest.mark.parametrize("offset_mm", [3.0, 5.0])
@pytest.mark.parametrize("transposition", [0, 1])
def test_equivalent_transposition_representations_share_physical_slice_origins(
    offset_mm, transposition
):
    """Scalar/list transposition forms share XYZ-derived labels within 0.001 mm."""
    handler, _ = _coronal_handler(offset_mm, transposition)
    source = _image()
    original = source.copy(deep=True)
    raw = _attach_image_geometry(source, handler)
    scanner = _attach_image_geometry(source, handler, recon_num=1)

    for name in ("x", "y", "z", "read_idx", "phase_idx"):
        np.testing.assert_allclose(
            raw[name], scanner[name], rtol=0, atol=POSITION_ATOL_MM
        )
    np.testing.assert_allclose(
        raw.slice_idx, scanner.slice_idx, rtol=0, atol=POSITION_ATOL_MM
    )
    # Increasing slice index is scanner +Y, so this projection is simply Y.
    np.testing.assert_allclose(
        scanner.slice_idx, scanner.y, rtol=0, atol=POSITION_ATOL_MM
    )
    np.testing.assert_allclose(
        scanner.slice_idx[0, 0, :],
        [offset_mm - 1, offset_mm + 1],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    check_reconstruction_geometry(
        raw, scanner, candidate_name="raw", reference_name="seq2d"
    )
    for result in (raw, scanner):
        np.testing.assert_allclose(
            result.slice_idx, result.y, rtol=0, atol=POSITION_ATOL_MM
        )
        np.testing.assert_allclose(
            result.slice_idx.attrs["axis_mm"],
            [offset_mm - 1, offset_mm + 1],
            rtol=0,
            atol=POSITION_ATOL_MM,
        )
        np.testing.assert_allclose(
            result.slice_idx.attrs["extent_mm"],
            [offset_mm - 2, offset_mm + 2],
            rtol=0,
            atol=POSITION_ATOL_MM,
        )
        np.testing.assert_array_equal(result.values, original.values)
        assert result.dtype == np.float32
    xr.testing.assert_identical(source, original)
    np.testing.assert_array_equal(handler.reco_transposition, [transposition])


@pytest.mark.parametrize("transposition", [0, 1])
def test_normalized_transposition_still_uses_physical_slice_origins(
    transposition,
):
    """Branch normalization cannot make both paths retain method-centred labels."""
    handler, visu = _coronal_handler(3.0, transposition)
    handler.load_reconstruction_parameters = lambda _: (
        {"RECO_transposition": np.array([transposition])},
        visu,
    )
    raw = _attach_image_geometry(_image(), handler)
    scanner = _attach_image_geometry(_image(), handler, recon_num=1)

    check_reconstruction_geometry(
        raw, scanner, candidate_name="raw", reference_name="seq2d"
    )
    for result in (raw, scanner):
        np.testing.assert_allclose(
            result.slice_idx, result.y, rtol=0, atol=POSITION_ATOL_MM
        )


def test_correcting_mm_slice_labels_preserves_read_phase_and_is_repeatable():
    """Existing in-plane labels and attributes survive slice-only correction."""
    handler, _ = _coronal_handler(3.0, 1)
    source = _image()
    axes = get_rps_axes(handler)
    source = source.assign_coords(
        {
            dim: (SPATIAL_DIMS, grid, {"unit": "mm", "description": "retained"})
            for dim, grid in zip(
                SPATIAL_DIMS, np.meshgrid(*axes, indexing="ij"), strict=True
            )
        }
    )
    original = source.copy(deep=True)
    result = _attach_image_geometry(source, handler)
    for dim in ("read_idx", "phase_idx"):
        xr.testing.assert_identical(
            result[dim].drop_vars(["x", "y", "z", "slice_idx"]),
            original[dim].drop_vars("slice_idx"),
        )
    repeated = _attach_image_geometry(result, handler)
    xr.testing.assert_identical(repeated, result)
    xr.testing.assert_identical(source, original)


@pytest.mark.parametrize("reverse", [False, True])
def test_oblique_slice_projection_follows_increasing_index(reverse):
    """An invented orthonormal oblique stack has an analytic projected origin."""
    handler, visu = _coronal_handler(3.0, 0)
    root_half = np.sqrt(0.5)
    read = np.array([root_half, -root_half, 0.0])
    phase = np.array([0.0, 0.0, 1.0])
    direction = np.array([root_half, root_half, 0.0]) * (-1 if reverse else 1)
    orientation = [np.array([read, phase, direction]).ravel()]
    origin = [2.0 * direction - 2.0 * read - 3.0 * phase]
    handler.visucore_orientation = visu["VisuCoreOrientation"] = orientation
    handler.visucore_position = visu["VisuCorePosition"] = origin
    result = _attach_image_geometry(_image(), handler)
    np.testing.assert_allclose(
        result.slice_idx[0, 0, :], [2.0, 4.0], rtol=0, atol=POSITION_ATOL_MM
    )
    np.testing.assert_allclose(
        result.attrs["slice_direction"], direction, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "first,second,equal",
    [
        (0, [0], True),
        ([1], np.array([[1]]), True),
        (None, None, True),
        (0, [1], False),
        ([0, 1], [1, 0], False),
        (None, [0], False),
    ],
)
def test_transposition_comparison_normalizes_shape_without_changing_frames(
    first, second, equal
):
    assert _same_transposition(first, second) is equal


@pytest.mark.parametrize(
    "kind", ["missing", "singleton", "degenerate", "sheared", "nonuniform"]
)
def test_slice_projection_does_not_invent_an_axis_for_unusable_geometry(kind):
    shape = (2, 3, 1 if kind == "singleton" else 3)
    data = xr.DataArray(np.zeros(shape), dims=SPATIAL_DIMS)
    if kind != "missing":
        xyz = list(
            np.meshgrid(
                *[np.arange(size, dtype=float) for size in shape], indexing="ij"
            )
        )
        if kind == "degenerate":
            xyz[2][:] = 0
        elif kind == "sheared":
            xyz[0] += xyz[2]
        elif kind == "nonuniform":
            xyz[2][..., -1] += 1
        data = data.assign_coords(
            {
                name: (SPATIAL_DIMS, grid)
                for name, grid in zip(("x", "y", "z"), xyz, strict=True)
            }
        )
    assert _physical_slice_axis(data) is None
