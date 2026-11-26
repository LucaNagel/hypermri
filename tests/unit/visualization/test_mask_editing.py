"""Check the widget-free mask functions shared by ``plot`` and ``plot_scan``.

All arrays are invented; no scanner data are used. Expected coverages are
exact areas of simple squares on a unit voxel grid.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.visualization.mask_editing import (
    DEFAULT_MASK_ALPHA,
    DEFAULT_MASK_COLOR,
    combine_plane_coverage,
    empty_mask,
    mask_dims,
    masked_mean_curve,
    masks_from_dataset,
    masks_to_dataset,
    outline_to_shown,
    outline_to_source,
    plane_masked_mean,
    polygon_coverage,
    shown_coverage_to_source,
    source_coverage_to_shown,
    transfer_coverage,
)
from hypermri.visualization.slice_lines import VoxelGrid, nearest_voxel, voxel_grid

# Extent [xmin, xmax, ymin, ymax] of a 4 x 4 plane with unit voxels.
EXTENT = [0.0, 4.0, 0.0, 4.0]


def test_polygon_coverage_is_exact_overlap_area():
    # Square from (0.5, 0.5) to (1.5, 1.5) covers a quarter of four voxels.
    square = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    coverage = polygon_coverage(square, EXTENT, (4, 4))
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0:2, 0:2] = 0.25
    assert coverage.dtype == np.float32
    np.testing.assert_allclose(coverage, expected, atol=1e-6)


def test_polygon_coverage_full_voxel_and_row_direction():
    # Row index follows y: y in [2, 3] is row 2, x in [1, 2] is column 1.
    square = [(1, 2), (2, 2), (2, 3), (1, 3)]
    coverage = polygon_coverage(square, EXTENT, (4, 4))
    assert coverage[2, 1] == pytest.approx(1.0)
    assert coverage.sum() == pytest.approx(1.0)


def test_polygon_coverage_two_points_draw_a_line_and_one_point_nothing():
    line = polygon_coverage([(0.5, 0.5), (3.5, 0.5)], EXTENT, (4, 4))
    assert np.all(line[0] > 0) and np.all(line[1:] == 0)
    assert not polygon_coverage([(0.5, 0.5)], EXTENT, (4, 4)).any()


def test_combine_adds_by_maximum_and_erases_by_subtraction():
    previous = np.array([[0.2, 0.8], [1.0, 0.0]], dtype=np.float32)
    drawn = np.array([[0.5, 0.5], [0.0, 0.3]], dtype=np.float32)
    before = previous.copy()
    np.testing.assert_allclose(
        combine_plane_coverage(previous, drawn, erase=False), [[0.5, 0.8], [1.0, 0.3]]
    )
    np.testing.assert_allclose(
        combine_plane_coverage(previous, drawn, erase=True),
        [[0.0, 0.3], [1.0, 0.0]],
        atol=1e-6,
    )
    np.testing.assert_array_equal(previous, before)


def test_empty_mask_follows_spatial_dims_and_coords():
    data = xr.DataArray(
        np.zeros((3, 4, 2, 8)),
        dims=("read_idx", "phase_idx", "slice_idx", "acquisition_idx"),
        coords={"read_idx": [0.0, 2.5, 5.0]},
    )
    mask = empty_mask(data)
    assert mask_dims(data) == ["read_idx", "phase_idx", "slice_idx"]
    assert mask.dims == ("read_idx", "phase_idx", "slice_idx")
    assert mask.dtype == np.float32 and not mask.values.any()
    np.testing.assert_array_equal(mask["read_idx"], [0.0, 2.5, 5.0])


def test_dataset_round_trip_keeps_style():
    data = xr.DataArray(np.zeros((2, 2, 4)), dims=("read_idx", "phase_idx", "x"))
    mask = empty_mask(data)
    mask.values[0, 1] = 0.5
    dataset = masks_to_dataset(
        {"tumour": mask, "muscle": empty_mask(data)},
        {"tumour": "#00ff00"},
        {"tumour": 0.7},
        data=data,
        scan=None,
    )
    masks, colors, alphas, outlines = masks_from_dataset(dataset)
    np.testing.assert_array_equal(masks["tumour"].values, mask.values)
    assert colors == {"tumour": "#00ff00", "muscle": DEFAULT_MASK_COLOR}
    assert alphas == {"tumour": 0.7, "muscle": DEFAULT_MASK_ALPHA}
    assert outlines == {"tumour": [], "muscle": []}


def test_masked_mean_curve_sorts_axis_and_labels_ppm():
    # Invented signal: two voxels, three spectral points on a descending ppm
    # axis. The curve is drawn ascending with an inverted axis.
    data = xr.DataArray(
        np.array([[1 + 0j, 2j, -3], [3 + 0j, -2j, 5]]),
        dims=("read_idx", "acquisition_idx"),
        coords={
            "acquisition_idx": ("acquisition_idx", [3.0, 2.0, 1.0], {"units": "ppm"})
        },
        attrs={"units": "a.u."},
    )
    mask = xr.DataArray(np.array([1.0, 0.5], dtype=np.float32), dims="read_idx")
    curve = masked_mean_curve(data, mask, secondary_dim="acquisition_idx", selectors={})
    weighted = (np.array([1, 2, 3]) + 0.5 * np.array([3, 2, 5])) / 1.5
    np.testing.assert_allclose(curve.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(curve.y, weighted[::-1], rtol=1e-12)
    np.testing.assert_array_equal(curve.sort_order, [2, 1, 0])
    assert curve.x_label == "acquisition_idx [ppm]"
    assert curve.y_label == "Mean magnitude [a.u.]"
    assert curve.invert_x


@pytest.mark.parametrize(
    "shown_dims, rows, columns",
    [
        (["read_idx", "phase_idx"], [0, 1, 2], [0, 1]),
        (["read_idx", "phase_idx"], [2, 1, 0], [1, 0]),
        (["phase_idx", "read_idx"], [1, 0], [0, 1, 2]),
        (["phase_idx", "read_idx"], [0, 1], [2, 1, 0]),
    ],
)
def test_shown_coverage_reaches_the_indexed_source_voxel(shown_dims, rows, columns):
    # Every shown plane of a 3 x 2 grid (transposed and/or reversed) maps
    # each shown cell to the source voxel named by its row and column index.
    sizes = {"read_idx": 3, "phase_idx": 2}
    indices = {shown_dims[0]: np.array(rows), shown_dims[1]: np.array(columns)}
    coverage = np.zeros((len(rows), len(columns)))
    coverage[0, 1] = 1.0
    coverage[1, 0] = 0.25
    source = shown_coverage_to_source(
        coverage, shown_dims, indices, sizes, ["read_idx", "phase_idx"]
    )
    assert source.dtype == np.float32
    assert source.shape == (3, 2)
    expected = np.zeros((3, 2))
    for row, column in [(0, 1), (1, 0)]:
        voxel = {shown_dims[0]: rows[row], shown_dims[1]: columns[column]}
        expected[voxel["read_idx"], voxel["phase_idx"]] = coverage[row, column]
    np.testing.assert_array_equal(source, expected)
    plane = xr.DataArray(source, dims=("read_idx", "phase_idx"))
    np.testing.assert_array_equal(
        source_coverage_to_shown(plane, shown_dims, indices), coverage
    )


def test_shown_coverage_mapping_rejects_other_dims_and_shapes():
    indices = {"read_idx": np.arange(3), "phase_idx": np.arange(2)}
    sizes = {"read_idx": 3, "phase_idx": 2}
    shown_dims = ["read_idx", "phase_idx"]
    with pytest.raises(ValueError, match="same two dimensions"):
        shown_coverage_to_source(
            np.zeros((3, 2)), shown_dims, indices, sizes, ["read_idx", "slice_idx"]
        )
    with pytest.raises(ValueError, match="no source indices"):
        shown_coverage_to_source(
            np.zeros((3, 2)), shown_dims, {"read_idx": np.arange(3)}, sizes, shown_dims
        )
    with pytest.raises(ValueError, match="coverage shape"):
        shown_coverage_to_source(
            np.zeros((2, 3)), shown_dims, indices, sizes, shown_dims
        )


def test_plane_masked_mean_weights_by_coverage_and_skips_gaps():
    curves = np.zeros((2, 2, 3))
    curves[0, 0] = [1.0, 2.0, np.nan]
    curves[1, 1] = [4.0, 8.0, 6.0]
    coverage = np.array([[1.0, 0.0], [0.0, 0.5]])
    mean = plane_masked_mean(curves, coverage)
    np.testing.assert_allclose(mean, [2.0, 4.0, 6.0])
    assert np.isnan(plane_masked_mean(curves, np.zeros((2, 2)))).all()
    with pytest.raises(ValueError):
        plane_masked_mean(curves, np.ones((3, 2)))


def _single_slice_grid_data():
    """Invented single axial slice: 2 mm pixels at x = 0..6, y = 0..4, z = 1 mm."""
    dims = ("read_idx", "phase_idx")
    x, y = np.meshgrid(
        np.arange(0.0, 8.0, 2.0), np.arange(0.0, 6.0, 2.0), indexing="ij"
    )
    return xr.DataArray(
        np.zeros(x.shape),
        dims=dims,
        coords={
            "x": (dims, x),
            "y": (dims, y),
            "z": (dims, np.full(x.shape, 1.0)),
        },
    )


def test_voxel_grid_of_a_single_slice_uses_the_slice_thickness():
    data = _single_slice_grid_data()
    grid = voxel_grid(data, ["read_idx", "phase_idx"], 3.0)

    assert grid.shape == (4, 3, 1)
    np.testing.assert_allclose(np.abs(grid.steps_mm[2]), [0.0, 0.0, 3.0])
    points = np.array([[2.9, 3.1, 2.4], [2.9, 3.1, 2.6], [-1.1, 0.0, 1.0]])
    np.testing.assert_array_equal(
        nearest_voxel(grid, points), [[1, 2, 0], [-1, -1, -1], [-1, -1, -1]]
    )
    with pytest.raises(ValueError, match="slice thickness"):
        voxel_grid(data, ["read_idx", "phase_idx"])


def test_nearest_voxel_takes_the_nearest_slice_along_clamped_axes():
    data = _single_slice_grid_data()
    grid = voxel_grid(data, ["read_idx", "phase_idx"], 3.0)
    points = np.array([[2.9, 3.1, 2.6], [2.9, 3.1, -40.0], [-1.1, 0.0, 1.0]])

    np.testing.assert_array_equal(
        nearest_voxel(grid, points, clamp_axes=(2,)),
        [[1, 2, 0], [1, 2, 0], [-1, -1, -1]],
    )


def test_transfer_coverage_samples_the_source_voxel_at_each_target_centre():
    source_grid = voxel_grid(_single_slice_grid_data(), ["read_idx", "phase_idx"], 2.0)
    # Invented target: 1 mm pixels over the same slice, one plane at z = 1 mm.
    target_grid = VoxelGrid(
        np.array([0.0, 0.0, 1.0]), np.diag([1.0, 1.0, 1.0]), (8, 6, 1)
    )
    source = np.zeros((4, 3), dtype=np.float32)
    source[1, 2] = 0.5
    target = np.full((8, 6), 0.25, dtype=np.float32)

    moved = transfer_coverage(source, source_grid, target, target_grid)
    # Source pixel (1, 2) at (2, 4) mm contains target centres x = 1, 2 and
    # y = 3, 4 mm; target centres x = 7 or y = 5 mm lie half a pixel beyond
    # the source and keep their coverage.
    expected = np.zeros((8, 6), dtype=np.float32)
    expected[1:3, 3:5] = 0.5
    expected[7, :] = expected[:, 5] = 0.25
    np.testing.assert_array_equal(moved, expected)
    np.testing.assert_array_equal(target, 0.25)

    changed = np.zeros((4, 3), dtype=bool)
    changed[1, 2] = True
    only = transfer_coverage(np.ones((4, 3)), source_grid, target, target_grid, changed)
    assert np.count_nonzero(only == 1.0) == 4


def _plane(row_indices, column_indices, *, slice_index=None):
    """Shown 4 x 4 plane with unit voxels, as ScanView._mask_plane."""
    return {
        "extent": EXTENT,
        "shape": (4, 4),
        "row_dim": "phase_idx",
        "column_dim": "read_idx",
        "source_indices": {
            "phase_idx": np.asarray(row_indices),
            "read_idx": np.asarray(column_indices),
        },
        "slicing_dim": None if slice_index is None else "slice_idx",
        "slice_index": slice_index,
    }


def test_outline_keeps_the_drawn_points_on_the_data_grid():
    # Rows shown reversed (descending scanner axis), columns 2..5 of the data.
    plane = _plane([3, 2, 1, 0], [2, 3, 4, 5], slice_index=1)
    drawn = [(0.25, 0.5), (1.75, 0.5), (1.0, 2.25)]
    outline = outline_to_source(drawn, plane, erase=True)
    assert outline["dims"] == ["phase_idx", "read_idx"]
    # Shown row edge f is data edge 4 - f; shown column edge f is data edge 2 + f.
    assert outline["points"] == [[3.5, 2.25], [3.5, 3.75], [1.75, 3.0]]
    assert outline["erase"] is True and outline["slice_index"] == 1

    shown = outline_to_shown(outline, plane)
    np.testing.assert_allclose(shown["x"], [p[0] for p in drawn])
    np.testing.assert_allclose(shown["y"], [p[1] for p in drawn])
    assert shown["erase"] is True


def test_outline_follows_a_rotated_plane_and_stays_in_its_slice():
    plane = _plane([0, 1, 2, 3], [0, 1, 2, 3], slice_index=0)
    outline = outline_to_source([(0.5, 0.5), (3.5, 0.5), (2.0, 3.5)], plane)
    rotated = {
        **plane,
        "row_dim": "read_idx",
        "column_dim": "phase_idx",
        "source_indices": {
            "read_idx": np.array([3, 2, 1, 0]),
            "phase_idx": np.array([0, 1, 2, 3]),
        },
    }
    shown = outline_to_shown(outline, rotated)
    # Data point (phase 0.5, read 0.5) is shown at column 0.5, row 4 - 0.5.
    np.testing.assert_allclose(shown["x"], [0.5, 0.5, 3.5])
    np.testing.assert_allclose(shown["y"], [3.5, 0.5, 2.0])
    assert outline_to_shown(outline, {**plane, "slice_index": 1}) is None


def test_outline_is_not_kept_for_unevenly_stepped_axes():
    assert outline_to_source([(0, 0), (1, 1)], _plane([0, 2, 4, 6], range(4))) is None


def test_dataset_round_trip_keeps_outlines():
    data = xr.DataArray(np.zeros((4, 4)), dims=("read_idx", "phase_idx"))
    outline = outline_to_source(
        [(0.5, 0.5), (2, 0.5), (1, 2)], _plane(range(4), range(4))
    )
    dataset = masks_to_dataset(
        {"roi": empty_mask(data)},
        {},
        {},
        data=data,
        scan=None,
        outlines={"roi": [outline]},
    )
    assert isinstance(dataset["roi"].attrs["outlines"], str)
    masks, _, _, outlines = masks_from_dataset(dataset)
    assert outlines == {"roi": [outline]}
    assert "outlines" not in masks["roi"].attrs

    dataset["roi"].attrs["outlines"] = "not json"
    with pytest.raises(ValueError, match="'outlines' attr of mask 'roi'"):
        masks_from_dataset(dataset)
