import numpy as np
import xarray as xr

from hypermri.analysis.spectroscopy.schema import (
    STANDARD_SPECTROSCOPY_DIMS,
    ensure_standard_spectroscopy_dataarray,
)


def test_ensure_standard_spectroscopy_dataarray_maps_numpy_dims():
    input_data = np.arange(6).reshape(2, 3)

    normalized = ensure_standard_spectroscopy_dataarray(
        data=input_data,
        provided_dims=["x", "y"],
    )

    assert normalized.dims == STANDARD_SPECTROSCOPY_DIMS
    assert normalized.shape == (1, 1, 2, 3, 1, 1)
    np.testing.assert_array_equal(
        normalized.isel(
            acquisition_idx=0,
            slice_idx=0,
            repetition_idx=0,
            channel_idx=0,
        ).values,
        input_data,
    )


def test_ensure_standard_spectroscopy_dataarray_renames_xarray_dims():
    input_data = xr.DataArray(
        np.arange(8).reshape(2, 4),
        dims=("reps", "spec"),
    )

    normalized = ensure_standard_spectroscopy_dataarray(data=input_data)

    assert normalized.dims == STANDARD_SPECTROSCOPY_DIMS
    assert normalized.shape == (4, 1, 1, 1, 2, 1)
    np.testing.assert_array_equal(
        normalized.isel(
            slice_idx=0,
            read_idx=0,
            phase_idx=0,
            channel_idx=0,
        ).values.T,
        input_data.values,
    )
