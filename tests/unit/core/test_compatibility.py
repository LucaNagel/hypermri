import numpy as np
import pytest
import xarray as xr

from hypermri.compatibility import assign_frequency_axis, resolve_frequency_axis

LEGACY_FREQUENCY_NAMES = {"freq_range_hz", "freq_range_Hz", "freq_range_ppm"}


@pytest.mark.parametrize(
    ("unit", "key"), [("ppm", "freq_axis_ppm"), ("Hz", "freq_axis_hz")]
)
def test_assign_frequency_axis_stores_a_matching_axis_only_as_coordinate(unit, key):
    axis = np.linspace(100.0, 200.0, 5)
    data = xr.DataArray(
        np.zeros((2, 5)), dims=("repetition_idx", "acquisition_idx"), attrs={key: 0}
    )

    result = assign_frequency_axis(data, axis, unit)

    assert result[key].dims == ("acquisition_idx",)
    assert result[key].attrs == {"unit": unit}
    np.testing.assert_array_equal(result[key], axis)
    assert key not in result.attrs
    assert data.attrs == {key: 0}
    assert key not in data.coords


@pytest.mark.parametrize(
    ("data", "axis"),
    [
        (xr.Dataset({"fit_amps": ("metabolite", [1.0, 2.0])}), np.arange(5.0)),
        (xr.DataArray(np.zeros(3), dims="acquisition_idx"), np.arange(5.0)),
    ],
    ids=["no-acquisition-dimension", "different-length"],
)
def test_assign_frequency_axis_keeps_other_axes_as_attribute(data, axis):
    result = assign_frequency_axis(data, axis, "Hz")

    assert result is data
    assert "freq_axis_hz" not in result.coords
    np.testing.assert_array_equal(result.attrs["freq_axis_hz"], axis)
    assert not LEGACY_FREQUENCY_NAMES & set(result.attrs)


def test_assign_frequency_axis_ignores_a_missing_axis():
    data = xr.DataArray(np.zeros(3), dims="acquisition_idx")

    assert assign_frequency_axis(data, None, "ppm") is data
    assert data.attrs == {}


def test_resolve_frequency_axis_ignores_removed_legacy_names():
    axis = np.linspace(1.0, 2.0, 4)
    attrs = dict.fromkeys(LEGACY_FREQUENCY_NAMES, axis)

    assert resolve_frequency_axis(attrs, "ppm") is None
    assert resolve_frequency_axis(attrs, "Hz") is None
    assert set(attrs) == LEGACY_FREQUENCY_NAMES


def test_resolve_frequency_axis_prefers_the_coordinate_without_mutation():
    axis = np.linspace(160.0, 180.0, 4)
    data = xr.DataArray(
        np.zeros(4),
        dims="acquisition_idx",
        coords={"freq_axis_ppm": ("acquisition_idx", axis)},
        attrs={"freq_axis_ppm": np.zeros(7)},
    )

    np.testing.assert_array_equal(resolve_frequency_axis(data, "ppm"), axis)
    assert data.attrs["freq_axis_ppm"].shape == (7,)


def test_resolve_frequency_axis_reads_dataset_attributes():
    """A Dataset is a mapping of variables; its axis is in coords or attrs."""
    axis = np.arange(3.0)
    results = xr.Dataset({"fit_freqs": ("x", [1.0])}, attrs={"freq_axis_hz": axis})

    np.testing.assert_array_equal(resolve_frequency_axis(results, "Hz"), axis)
    assert resolve_frequency_axis(results, "ppm") is None


def test_resolve_frequency_axis_rejects_unknown_units():
    with pytest.raises(ValueError, match="Unsupported frequency unit"):
        resolve_frequency_axis({}, "T")
