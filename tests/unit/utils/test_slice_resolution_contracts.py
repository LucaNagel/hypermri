"""Verify the slice thickness and centre spacing contracts.

All inputs are synthetic geometry, without acquired samples or identifying
metadata. The five 1 mm slices with 4 mm gaps reproduce the geometry of the
ParaVision reference tests. Other cases are illustrative controls.
Generated arrays only exercise geometry; no acquired pixels are used.
Legacy arrays retain their fallbacks, while reconstructed single-package
2D arrays distinguish nominal thickness from centre spacing.
"""

from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import xarray as xr

from hypermri.analysis.anatomical.masks import (
    _default_plane_tolerance_mm,
    _estimate_xyz_affine,
    project_mask_to_data_grid,
)
from hypermri.reconstruction.strategies.bruker.raw import BrukerRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.seq2d import Brukerseq2dRecoStrategy
from hypermri.utils.coordinates.image_shifts import determine_FOV_shift
from hypermri.utils.io.dicom_export import (
    _geometry_from_xyz,
    export_to_dicom,
)
from hypermri.utils.plot.data_preprocessor import DataPreprocessor
from hypermri.vendors.bruker import BrukerHandler
from hypermri.visualization.render_data import PlotDataModel

# The geometry position tolerance separates the 0.8 mm spacing defect from
# coordinate rounding. Fourier checks recover spacing in mm before comparing.
POSITION_ATOL_MM = 1e-3
SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


@pytest.mark.parametrize(
    ("count", "gap", "dimension", "expected"),
    [
        (5, 4.0, "2D", 5.0),
        (5, 0.0, "2D", 1.0),
        (1, 4.0, "2D", 1.0),
        (1, 0.0, "3D", 2.0),
    ],
)
def test_explicit_slice_spacing_is_the_sampling_distance(
    tmp_path, write_paravision_parameters, count, gap, dimension, expected
):
    """Spacing follows the slice positions, including single-slice and 3D controls."""
    handler = _handler(
        tmp_path,
        write_paravision_parameters,
        count=count,
        gap=gap,
        dimension=dimension,
    )
    original_resolution = handler.resolution_mm.copy()
    np.testing.assert_allclose(
        handler.slice_spacing_mm,
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_array_equal(handler.resolution_mm, original_resolution)


def test_explicit_slice_spacing_is_unknown_without_a_single_spatial_stack(
    tmp_path, write_paravision_parameters
):
    """Several packages and missing geometry have no shared sampling interval."""
    handler = _handler(tmp_path, write_paravision_parameters)
    handler.n_slice_packs = 3
    assert handler.slice_spacing_mm is None
    handler.n_slice_packs = 1
    handler.spatial_dimension = None
    assert handler.slice_spacing_mm is None


@pytest.mark.parametrize("raw", [False, True])
def test_reconstruction_preserves_explicit_slice_spacing(
    tmp_path, write_paravision_parameters, raw
):
    """Both metadata paths retain the sampling distance on the slice axis."""
    handler = _handler(tmp_path, write_paravision_parameters)
    dims = ("acquisition_idx", *SPATIAL_DIMS, "repetition_idx", "channel_idx")
    data = np.zeros((1, 4, 6, 5, 1, 1))
    if raw:
        array = BrukerRawRecoStrategy().set_kspace_xarray(handler, data=data, dims=dims)
    else:
        array = Brukerseq2dRecoStrategy().set_scanner_data(data, dims, handler=handler)
    np.testing.assert_allclose(
        array.slice_idx.attrs["slice_spacing_mm"],
        5.0,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_fourier_axis_uses_explicit_spacing_without_changing_the_transform():
    """Changing the physical scale does not change Fourier samples."""
    data = _geometry()
    data.values[:] = np.arange(data.size).reshape(data.shape)
    before = DataPreprocessor().apply_kspace_transform(data)
    data.slice_idx.attrs["slice_spacing_mm"] = 5.0
    original = data.copy(deep=True)
    after = DataPreprocessor().apply_kspace_transform(data)
    np.testing.assert_allclose(after.values, before.values, rtol=1e-12, atol=1e-12)
    xr.testing.assert_identical(data, original)
    np.testing.assert_allclose(
        after.slice_idx,
        [-0.08, -0.04, 0.0, 0.04, 0.08],
        rtol=0,
        atol=1e-12,
    )
    assert after.slice_idx.attrs["unit"] == "1/mm"


def test_explicit_slice_spacing_has_mm_units_even_with_index_coordinates():
    """A physical sampling attribute retains its units on index-only arrays."""
    data = _geometry()
    data.slice_idx.attrs.update(unit="index", slice_spacing_mm=5.0)
    axis, unit = DataPreprocessor._spatial_frequency_axis(data, "slice_idx")
    assert unit == "1/mm"
    np.testing.assert_allclose(
        1.0 / (len(axis) * (axis[1] - axis[0])), 5.0, rtol=0, atol=POSITION_ATOL_MM
    )


@pytest.mark.parametrize("selection", ["stack", "singleton", "dropped"])
def test_dicom_separates_physical_thickness_from_spacing(tmp_path, selection):
    """Synthetic DICOM geometry retains 1 mm thickness and 5 mm spacing."""
    data = _geometry()
    data.attrs.update(slice_thickness_mm=1.0, slice_spacing_mm=5.0)
    if selection == "singleton":
        data = data.isel(slice_idx=slice(2, 3))
    elif selection == "dropped":
        data = data.isel(slice_idx=2, drop=True)
    original = data.copy(deep=True)
    assert export_to_dicom(data, tmp_path)
    datasets = [
        pydicom.dcmread(p, stop_before_pixels=True)
        for p in sorted(tmp_path.glob("*.dcm"))
    ]
    for ds in datasets:
        np.testing.assert_allclose(
            [float(ds.SliceThickness), float(ds.SpacingBetweenSlices)],
            [1.0, 5.0],
            rtol=0,
            atol=POSITION_ATOL_MM,
        )
    expected_z = np.arange(5) * 5.0 if selection == "stack" else [10.0]
    np.testing.assert_allclose(
        [ds.ImagePositionPatient[2] for ds in datasets],
        expected_z,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    xr.testing.assert_identical(data, original)


def test_dicom_reference_singleton_preserves_independent_spacing(tmp_path):
    """Synthetic reference tags keep their distinct thickness and spacing."""
    reference = pydicom.Dataset()
    reference.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    reference.ImagePositionPatient = [0, 0, 0]
    reference.PixelSpacing = [0.5, 0.5]
    reference.SliceThickness = 1.0
    reference.SpacingBetweenSlices = 5.0
    reference.InstanceNumber = 1
    data = xr.DataArray(np.zeros((4, 6)), dims=("read_idx", "phase_idx"))
    assert export_to_dicom(data, tmp_path, reference_dicoms=[reference])
    ds = pydicom.dcmread(next(tmp_path.glob("*.dcm")), stop_before_pixels=True)
    np.testing.assert_allclose(
        [float(ds.SliceThickness), float(ds.SpacingBetweenSlices)],
        [1.0, 5.0],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("dimension, expected", [("2D", 1.0), ("3D", 2.0)])
def test_reconstruction_carries_voxel_thickness_after_slice_selection(
    tmp_path, write_paravision_parameters, raw, dimension, expected
):
    """Array metadata retain 2D thickness or 3D partition size on selection."""
    count = 5 if dimension == "2D" else 1
    handler = _handler(
        tmp_path, write_paravision_parameters, count=count, dimension=dimension
    )
    dims = ("acquisition_idx", *SPATIAL_DIMS, "repetition_idx", "channel_idx")
    data = np.zeros((1, 4, 6, int(handler.matrix[2]), 1, 1))
    if raw:
        array = BrukerRawRecoStrategy().set_kspace_xarray(handler, data=data, dims=dims)
    else:
        array = Brukerseq2dRecoStrategy().set_scanner_data(data, dims, handler=handler)
    np.testing.assert_allclose(
        array.isel(slice_idx=0, drop=True).attrs["slice_thickness_mm"],
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_allclose(
        array.slice_idx.attrs["slice_thickness_mm"],
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_dicom_handler_geometry_retains_slice_thickness(
    tmp_path, write_paravision_parameters
):
    """A lazy synthetic handler supplies both physical slice sizes without data."""
    handler = _handler(tmp_path, write_paravision_parameters)
    handler.visucore_position = np.column_stack(
        (np.zeros(5), np.zeros(5), np.arange(5) * 5.0)
    )
    handler.visucore_orientation = np.tile(np.eye(3).reshape(1, 9), (5, 1))
    handler.visucore_size = np.array([4, 6])
    handler.visucore_extent = np.array([2.0, 3.0])
    data = xr.DataArray(np.zeros((4, 6, 5)), dims=SPATIAL_DIMS)
    output = tmp_path / "dicom"
    assert export_to_dicom(data, output, geometry=handler)
    ds = pydicom.dcmread(next(output.glob("*.dcm")), stop_before_pixels=True)
    np.testing.assert_allclose(
        [float(ds.SliceThickness), float(ds.SpacingBetweenSlices)],
        [1.0, 5.0],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize("key", ["slice_thickness_mm", "slice_spacing_mm"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, [1.0, 2.0]])
def test_dicom_rejects_invalid_explicit_slice_metadata(tmp_path, key, value):
    """Invalid supplied physical sizes fail before any synthetic file is written."""
    data = _geometry().isel(slice_idx=slice(0, 1))
    data.attrs[key] = value
    with pytest.raises(ValueError, match=key):
        export_to_dicom(data, tmp_path)
    assert not list(tmp_path.glob("*.dcm"))


@pytest.mark.parametrize(
    "selection, expected", [("stack", 5.0), ("singleton", 4.2), ("dropped", 1.0)]
)
def test_legacy_3d_slab_metadata_is_not_exported_as_partition_thickness(
    selection, expected
):
    """Legacy raw arrays keep their previous partition fallback, not the slab."""
    data = _geometry()
    data.attrs.update(spatial_dimension="3D", slice_thickness_mm=16.0)
    if selection == "singleton":
        data = data.isel(slice_idx=slice(0, 1))
    elif selection == "dropped":
        data = data.isel(slice_idx=0, drop=True)
    geometry = _geometry_from_xyz(data)
    np.testing.assert_allclose(
        geometry.slice_thickness, expected, rtol=0, atol=POSITION_ATOL_MM
    )


def _handler(tmp_path, write_parameters, *, count=5, gap=4.0, dimension="2D", packs=1):
    """Load synthetic geometry-only parameter files using the real parser."""
    tmp_path = tmp_path / "1"
    matrix = [4, 6] if dimension == "2D" else [4, 6, 8]
    fov = [2.0, 3.0] if dimension == "2D" else [2.0, 3.0, 16.0]
    method = {
        "Method": "<Bruker:FLASH>",
        "PVM_SpatDimEnum": f"<{dimension}>",
        "PVM_Matrix": matrix,
        "PVM_Fov": fov,
        "PVM_NSPacks": packs,
        "PVM_SPackArrNSlices": count if packs == 1 else [count] * packs,
        "PVM_SliceThick": 1.0 if dimension == "2D" else 16.0,
        "PVM_SPackArrSliceGap": gap,
        "PVM_ObjOrderList": list(range(count * packs)),
        "PVM_SPackArrReadOrient": "<L_R>",
        "PVM_SPackArrSliceOrient": "<axial>",
    }
    write_parameters(tmp_path / "method", method)
    write_parameters(
        tmp_path / "acqp",
        {"ACQ_grad_matrix": [np.eye(3).tolist()], "ACQ_patient_pos": "Head_Prone"},
    )
    write_parameters(tmp_path / "pdata" / "1" / "reco", {})
    return BrukerHandler(tmp_path, log_mode="critical")


def _geometry():
    """Return an axial synthetic stack with the legacy 4.2 mm attribute."""
    axes = (np.arange(4) * 0.5, np.arange(6) * 0.5, np.arange(5) * 5.0)
    xyz = np.meshgrid(*axes, indexing="ij")
    data = xr.DataArray(
        np.zeros((4, 6, 5)),
        dims=SPATIAL_DIMS,
        coords={
            **dict(zip(SPATIAL_DIMS, axes, strict=True)),
            **{
                name: (SPATIAL_DIMS, values)
                for name, values in zip("xyz", xyz, strict=True)
            },
        },
    )
    for dim, resolution in zip(SPATIAL_DIMS, (0.5, 0.5, 4.2), strict=True):
        data[dim].attrs.update(resolution_mm=resolution, unit="mm")
    return data


@pytest.mark.parametrize(
    ("count", "gap", "dimension", "expected"),
    [
        (5, 4.0, "2D", 1.0),
        (5, 0.0, "2D", 1.0),
        (1, 4.0, "2D", 1.0),
        (1, 0.0, "3D", 2.0),
    ],
)
def test_handler_resolution_is_voxel_thickness(
    tmp_path, write_paravision_parameters, count, gap, dimension, expected
):
    """2D gaps do not inflate thickness; no-gap, single and 3D sizes stay."""
    handler = _handler(
        tmp_path,
        write_paravision_parameters,
        count=count,
        gap=gap,
        dimension=dimension,
    )
    np.testing.assert_allclose(
        handler.resolution_mm,
        [0.5, 0.5, expected],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_gapped_resolution_denotes_nominal_thickness(
    tmp_path, write_paravision_parameters
):
    """Use the 1 mm thickness independently of the 5 mm sampling step."""
    handler = _handler(tmp_path, write_paravision_parameters)
    np.testing.assert_allclose(
        [handler.resolution_mm[2], handler.slice_spacing_mm],
        [1.0, 5.0],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize("raw", [False, True])
def test_reconstruction_preserves_the_handler_resolution(
    tmp_path, write_paravision_parameters, raw
):
    """Both metadata setters propagate 1 mm thickness without recomputing it."""
    handler = _handler(tmp_path, write_paravision_parameters)
    dims = ("acquisition_idx", *SPATIAL_DIMS, "repetition_idx", "channel_idx")
    data = np.zeros((1, 4, 6, 5, 1, 1))
    if raw:
        array = BrukerRawRecoStrategy().set_kspace_xarray(handler, data=data, dims=dims)
    else:
        array = Brukerseq2dRecoStrategy().set_scanner_data(data, dims, handler=handler)
    np.testing.assert_allclose(
        array.slice_idx.attrs["resolution_mm"],
        1.0,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_multiple_packages_have_no_global_slice_fov_or_resolution(
    tmp_path, write_paravision_parameters
):
    """Several slice packages keep their in-plane geometry without inventing a combined stack."""
    handler = _handler(tmp_path, write_paravision_parameters, packs=3)
    np.testing.assert_allclose(
        handler.resolution_mm[:2],
        [0.5, 0.5],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_allclose(handler.fov_mm[:2], [2.0, 3.0])
    assert np.isnan(handler.fov_mm[2])
    assert np.isnan(handler.resolution_mm[2])
    assert handler.slice_thickness_mm == 1.0
    assert handler.slice_spacing_mm is None
    assert handler._slice_geometry_attributes() == {"slice_thickness_mm": 1.0}
    # Parsed acquisition metadata remain separate from derived geometry.
    np.testing.assert_array_equal(handler.method["PVM_Fov"], [2.0, 3.0])


@pytest.mark.parametrize("raw", [False, True])
def test_multiple_packages_propagate_unavailable_resolution_without_changing_signal(
    tmp_path, write_paravision_parameters, raw
):
    handler = _handler(tmp_path, write_paravision_parameters, packs=3)
    dims = ("acquisition_idx", *SPATIAL_DIMS, "repetition_idx", "channel_idx")
    data = np.arange(4 * 6 * 15).reshape((1, 4, 6, 15, 1, 1)).astype(float)
    before = data.copy()
    if raw:
        result = BrukerRawRecoStrategy().set_kspace_xarray(
            handler, data=data, dims=dims
        )
    else:
        result = Brukerseq2dRecoStrategy().set_scanner_data(data, dims, handler=handler)
    assert np.isnan(result.slice_idx.attrs["resolution_mm"])
    assert result.attrs["slice_thickness_mm"] == 1.0
    assert result.slice_idx.attrs["slice_thickness_mm"] == 1.0
    assert "slice_spacing_mm" not in result.attrs
    np.testing.assert_array_equal(result.slice_idx, np.arange(15))
    np.testing.assert_array_equal(result.values, before)
    np.testing.assert_array_equal(data, before)


def test_multiple_packages_require_package_selection_for_uniform_rps_geometry(
    tmp_path, write_paravision_parameters
):
    from hypermri.utils.coordinates.grids import get_rps_axes, get_rps_extent

    handler = _handler(tmp_path, write_paravision_parameters, packs=3)
    for operation in (get_rps_axes, get_rps_extent):
        with pytest.raises(ValueError, match="Select an individual package"):
            operation(handler)


@pytest.mark.parametrize("raw", [False, True])
def test_reconstructed_plane_uses_thickness_for_mask_and_plot(
    tmp_path, write_paravision_parameters, raw
):
    """A 1 mm plane covers points within 0.5 mm and has a 1 mm plot normal."""
    handler = _handler(tmp_path, write_paravision_parameters)
    dims = ("acquisition_idx", *SPATIAL_DIMS, "repetition_idx", "channel_idx")
    data = np.zeros((1, 4, 6, 5, 1, 1))
    if raw:
        array = BrukerRawRecoStrategy().set_kspace_xarray(handler, data=data, dims=dims)
    else:
        array = Brukerseq2dRecoStrategy().set_scanner_data(data, dims, handler=handler)
    array = array.isel(acquisition_idx=0, repetition_idx=0, channel_idx=0, drop=True)
    # Analytic XYZ centres isolate the geometry consumers from scanner files.
    array = array.assign_coords(
        {name: (SPATIAL_DIMS, _geometry()[name].values) for name in "xyz"}
    )
    plane = array.isel(slice_idx=slice(0, 1))
    geometry = _estimate_xyz_affine(plane)
    model = object.__new__(PlotDataModel)
    np.testing.assert_allclose(
        [
            _default_plane_tolerance_mm(geometry),
            np.linalg.norm(model._xyz_vector_for_dim(plane, "slice_idx")),
        ],
        [0.5, 1.0],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    target = _geometry().isel(slice_idx=slice(0, 2))
    target = target.assign_coords(
        z=(SPATIAL_DIMS, np.broadcast_to([0.49, 0.51], target.shape))
    )
    projected = project_mask_to_data_grid(
        xr.ones_like(plane),
        target,
        coverage_samples=1,
    )
    # The probes are 0.01 mm away from the boundary, ten position tolerances.
    np.testing.assert_array_equal(
        projected.values, np.broadcast_to([1.0, 0.0], target.shape)
    )
    np.testing.assert_allclose(
        np.linalg.norm(model._xyz_vector_for_dim(array, "slice_idx")),
        5.0,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize(("plane", "expected"), [(False, 5.0), (True, 4.2)])
def test_mask_spacing_uses_xyz_except_for_a_selected_plane(plane, expected):
    """A singleton retains the stack's attribute and sets the plane tolerance."""
    data = _geometry()
    if plane:
        data = data.isel(slice_idx=slice(0, 1))
    geometry = _estimate_xyz_affine(data)
    np.testing.assert_allclose(
        geometry["dim_resolutions"][2],
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_allclose(
        _default_plane_tolerance_mm(geometry),
        2.1 if plane else 1e-3,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize(("plane", "expected"), [(False, 5.0), (True, 4.2)])
def test_dicom_thickness_currently_depends_on_selected_slice_count(plane, expected):
    """Whole stacks use centre distance; a selected plane uses resolution."""
    data = _geometry()
    if plane:
        data = data.isel(slice_idx=slice(0, 1))
    geometry = _geometry_from_xyz(data)
    np.testing.assert_allclose(
        geometry.slice_thickness,
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_fourier_axis_currently_prefers_resolution_over_mm_centres():
    """The 4.2 mm attribute overrides the explicit 5 mm coordinate step."""
    axis, unit = DataPreprocessor._spatial_frequency_axis(_geometry(), "slice_idx")
    assert unit == "1/mm"
    inferred_spacing = 1.0 / (len(axis) * (axis[1] - axis[0]))
    np.testing.assert_allclose(inferred_spacing, 4.2, rtol=0, atol=POSITION_ATOL_MM)


def test_dicom_without_slice_dimension_retains_global_thickness():
    """Dropping the slice dimension retains explicit physical thickness."""
    data = _geometry().isel(slice_idx=0, drop=True)
    data.attrs["slice_thickness_mm"] = 3.0
    geometry = _geometry_from_xyz(data)
    np.testing.assert_allclose(
        geometry.slice_thickness, 3.0, rtol=0, atol=POSITION_ATOL_MM
    )


def test_mask_without_slice_dimension_uses_half_the_physical_thickness():
    """Xarray selection retains thickness for the physical plane normal."""
    data = _geometry().isel(slice_idx=0, drop=True)
    data.attrs["slice_thickness_mm"] = 1.0
    geometry = _estimate_xyz_affine(data)
    np.testing.assert_allclose(
        _default_plane_tolerance_mm(geometry), 0.5, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(("plane", "expected"), [(False, 5.0), (True, 4.2)])
def test_plot_voxel_vector_uses_xyz_except_for_a_selected_plane(plane, expected):
    """The UI-free geometry path uses resolution for a singleton normal."""
    model = object.__new__(PlotDataModel)
    data = _geometry()
    if plane:
        data = data.isel(slice_idx=slice(0, 1))
    vector = model._xyz_vector_for_dim(data, "slice_idx")
    np.testing.assert_allclose(
        np.linalg.norm(vector),
        expected,
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


@pytest.mark.parametrize("slice_resolution", [1.0, 4.2, 5.0])
def test_2d_fov_shift_discards_the_slice_component(slice_resolution):
    """Neither proposed meaning changes a public 2D field-of-view shift."""
    scan = SimpleNamespace(
        resolution_mm=np.array([0.5, 0.5, slice_resolution]),
        read_offset_mm=0.0,
        phase_offset_mm=2.0,
        slice_offset_mm=3.0,
        type="FLASH",
        spatial_dimension="2D",
        patient_pos="Head_Prone",
        slice_orientation="axial",
        read_orientation="L_R",
    )
    voxels, millimetres = determine_FOV_shift(scan)
    # 0.001 mm corresponds to 0.002 voxels at the synthetic 0.5 mm spacing.
    np.testing.assert_allclose(voxels, [0.0, 4.0], rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        millimetres,
        [0.0, 2.0],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
