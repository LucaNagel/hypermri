"""Coordinate and frequency properties return None only for absent axes.

The properties of ``UniformDataPropertiesMixin`` used to return None for any
exception, so a geometry or frequency-axis defect looked like an absent axis.
Now only ``GeometryUnavailableError`` (absent metadata or several slice
packages) gives None and every other error propagates. The scans are small
synthetic parameter sets in mm; no pixel data is involved.
"""

import numpy as np
import pytest

import hypermri.reconstruction.axes as frequency
import hypermri.utils.coordinates.grids as grids
from hypermri.mixins import UniformDataPropertiesMixin
from hypermri.utils.coordinates.grids import GeometryUnavailableError

RPS_PROPERTIES = {
    "rps_extent_mm": "get_rps_extent",
    "rps_axes_mm": "get_rps_axes",
    "rps_grid_mm": "get_rps_grid",
}
XYZ_PROPERTIES = {
    "xyz_extent_mm": "get_xyz_extent",
    "xyz_axes_mm": "get_xyz_axes",
    "xyz_grid_mm": "get_xyz_grid",
}
GEOMETRY_PROPERTIES = {**RPS_PROPERTIES, **XYZ_PROPERTIES}
AXIAL = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


class _Scan(UniformDataPropertiesMixin):
    """Synthetic scan whose metadata are plain attributes."""

    def __init__(self, **values):
        super().__init__()
        self.__dict__.update(values)

    def get(self, name, default=None):
        return getattr(self, name, default)


def _rps_scan(**overrides):
    """Return a 2D scan of 4 x 4 voxels of 8 mm and 5 slices 5 mm apart."""
    values = {
        "spatial_dimension": "2D",
        "fov_mm": np.array([32.0, 32.0, 21.0]),
        "matrix": np.array([4, 4, 5]),
        "n_slices": 5,
        "slice_thickness_mm": 1.0,
        "slice_gap_mm": 4.0,
        "resolution_mm": np.array([8.0, 8.0]),
    }
    values.update(overrides)
    return _Scan(**values)


def _xyz_scan(**overrides):
    """Return one axial slice of 4 x 2 voxels of 1 mm, 1 mm thick, at z = 0 mm."""
    values = {
        "matrix": np.array([4, 2, 1]),
        "n_slices": 1,
        "visucore_position": np.array([[-2.0, -1.0, 0.0]]),
        "visucore_orientation": np.array([AXIAL]),
        "visucore_size": np.array([4, 2]),
        "visucore_extent": np.array([4.0, 2.0]),
        "visucore_thickness": 1.0,
    }
    values.update(overrides)
    return _Scan(**values)


@pytest.mark.parametrize("name", RPS_PROPERTIES)
def test_rps_properties_of_a_complete_scan_match_the_grid_functions(name):
    scan = _rps_scan()

    expected = getattr(grids, RPS_PROPERTIES[name])(scan=_rps_scan())

    for actual, reference in zip(getattr(scan, name), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("name", RPS_PROPERTIES)
@pytest.mark.parametrize(
    "overrides",
    [
        {"fov_mm": None},
        {"matrix": None},
        {"fov_mm": np.array([32.0, 32.0]), "slice_thickness_mm": None},
        {"slices_per_pack": np.array([2, 3])},
    ],
    ids=["no-fov", "no-matrix", "no-slice-thickness", "two-slice-packages"],
)
def test_rps_properties_are_none_without_rps_geometry(name, overrides):
    assert getattr(_rps_scan(**overrides), name) is None


def test_rps_extent_is_none_without_resolution_for_the_half_voxel_correction():
    assert _rps_scan(resolution_mm=None).rps_extent_mm is None


@pytest.mark.parametrize("name", RPS_PROPERTIES)
@pytest.mark.parametrize(
    "overrides",
    [
        {"fov_mm": np.array([32.0, -32.0, 21.0])},
        {"matrix": np.array([4, 4]), "n_slices": 2.5},
    ],
    ids=["negative-fov", "fractional-slice-count"],
)
def test_rps_properties_raise_for_invalid_declared_geometry(name, overrides):
    scan = _rps_scan(**overrides)

    with pytest.raises(ValueError, match="missing or incomplete") as error:
        getattr(scan, name)
    assert not isinstance(error.value, GeometryUnavailableError)


@pytest.mark.parametrize("name", XYZ_PROPERTIES)
def test_xyz_properties_are_none_without_visucore_geometry(name):
    scan = _xyz_scan(
        visucore_position=None,
        visucore_orientation=None,
        visucore_size=None,
        visucore_extent=None,
    )

    assert getattr(scan, name) is None


@pytest.mark.parametrize("name", XYZ_PROPERTIES)
def test_xyz_properties_raise_for_incomplete_visucore_geometry(name):
    scan = _xyz_scan(visucore_extent=None)

    with pytest.raises(ValueError, match="visucore_extent not available"):
        getattr(scan, name)


@pytest.mark.parametrize("name", XYZ_PROPERTIES)
def test_xyz_properties_of_a_complete_scan_are_available(name):
    assert getattr(_xyz_scan(), name) is not None


@pytest.mark.parametrize("name", GEOMETRY_PROPERTIES)
def test_geometry_property_propagates_other_helper_errors(name, monkeypatch):
    def failing(**_):
        raise RuntimeError("geometry helper defect")

    monkeypatch.setattr(grids, GEOMETRY_PROPERTIES[name], failing)

    with pytest.raises(RuntimeError, match="geometry helper defect"):
        getattr(_rps_scan(), name)


@pytest.mark.parametrize("name", GEOMETRY_PROPERTIES)
def test_geometry_property_caches_results_but_not_unavailability(name, monkeypatch):
    calls = []
    results = [GeometryUnavailableError("absent"), ("grid",)]

    def helper(**_):
        calls.append(1)
        result = results[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(grids, GEOMETRY_PROPERTIES[name], helper)
    scan = _rps_scan()

    assert getattr(scan, name) is None
    assert getattr(scan, name) == ("grid",)
    assert getattr(scan, name) == ("grid",)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "name, unit",
    [("ppm", "ppm"), ("hz", "Hz"), ("freq_axis_ppm", "ppm"), ("freq_axis_hz", "Hz")],
)
def test_frequency_property_propagates_axis_errors(name, unit, monkeypatch):
    def failing(scan, unit):
        raise ValueError(f"inconsistent spectral metadata for {unit}")

    monkeypatch.setattr(frequency, "get_freq_axis", failing)

    with pytest.raises(ValueError, match=f"inconsistent spectral metadata for {unit}"):
        getattr(_rps_scan(), name)


@pytest.mark.parametrize("name", ["ppm", "hz", "freq_axis_ppm", "freq_axis_hz"])
def test_frequency_property_is_empty_without_bandwidth(name):
    assert getattr(_Scan(bandwidth_hz=None), name) == []


@pytest.mark.parametrize("name", ["ppm", "hz", "freq_axis_ppm", "freq_axis_hz"])
def test_frequency_property_is_none_without_bandwidth_attribute(name):
    assert getattr(_Scan(), name) is None


def test_geometry_unavailable_error_is_a_value_error():
    assert issubclass(GeometryUnavailableError, ValueError)
