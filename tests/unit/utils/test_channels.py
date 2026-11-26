import numpy as np
import xarray as xr

from hypermri.utils.mri.channels import PhaseAlignedConfig, combine_channels


def test_combine_channels_preserves_metadata_and_coordinates():
    values = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    data_array = xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "channel_idx"),
        coords={
            "read_idx": [0, 1],
            "phase_idx": [0, 1, 2],
            "channel_idx": [10, 11],
            "read_mm": ("read_idx", [1.0, 2.0]),
            "phase_mm": ("phase_idx", [5.0, 6.0, 7.0]),
            "coil_gain": (("channel_idx", "read_idx"), [[1.0, 2.0], [3.0, 4.0]]),
            "scan_label": "scan-001",
        },
        name="reco_data",
        attrs={"domain": "spatial", "custom_attr": "kept"},
    )
    data_array.coords["read_mm"].attrs["unit"] = "mm"
    data_array.coords["coil_gain"].attrs["long_name"] = "coil gain map"

    combined = combine_channels(data_array, method="sum")

    assert combined.name == data_array.name
    assert combined.dims == data_array.dims
    assert combined.sizes["channel_idx"] == 1

    assert combined.attrs["domain"] == "spatial"
    assert combined.attrs["custom_attr"] == "kept"
    assert combined.attrs["channels_combined"] == "sum"
    assert combined.attrs["n_channels_combined"] == 2

    assert set(combined.coords) == set(data_array.coords)

    for coord_name in data_array.coords:
        source_coord = data_array.coords[coord_name]
        if "channel_idx" in source_coord.dims:
            expected_coord = source_coord.isel(channel_idx=slice(0, 1))
        else:
            expected_coord = source_coord
        xr.testing.assert_identical(combined.coords[coord_name], expected_coord)


def test_combine_channels_mean_method():
    data_array = xr.DataArray(
        np.array([[1.0, 3.0], [5.0, 7.0]]),
        dims=("acquisition_idx", "channel_idx"),
        coords={"acquisition_idx": [0, 1], "channel_idx": [0, 1]},
    )

    combined = combine_channels(data_array, method="mean")

    np.testing.assert_array_equal(combined.values[:, 0], np.array([2.0, 6.0]))
    assert combined.attrs["channels_combined"] == "mean"


def test_combine_channels_phase_aligned_method():
    data_array = xr.DataArray(
        np.array([[1.0 + 0j, 1.0j], [1.0 + 0j, 1.0j]]),
        dims=("acquisition_idx", "channel_idx"),
        coords={"acquisition_idx": [0, 1], "channel_idx": [0, 1]},
    )
    config = PhaseAlignedConfig(
        raw_fid_data=np.array([1.0 + 0j, 0.0, 1.0j, 0.0]),
        combination_method="mean",
    )

    combined = combine_channels(data_array, method="phase_aligned", config=config)

    np.testing.assert_allclose(combined.values[:, 0], np.array([1.0, 1.0]))
    assert combined.attrs["channels_combined"] == "phase_aligned"
    assert combined.attrs["phase_correction_deg"] == 90.0
