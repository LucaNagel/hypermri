"""Tests of the common imaging dimension layout.

All arrays are small synthetic arrays; no recorded data is involved.
"""

import doctest

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction import layout
from hypermri.reconstruction.layout import IMAGING_DIM_ORDER, to_imaging_layout


def _image(dims, shape):
    return xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape),
        dims=dims,
        attrs={"patient_pos": "Head_Prone"},
    )


def test_missing_dimensions_are_added_with_length_one_and_an_index_coordinate():
    image = _image(("read_idx", "phase_idx", "slice_idx"), (4, 3, 2))

    result = to_imaging_layout(image)

    assert result.dims == IMAGING_DIM_ORDER
    assert result.shape == (4, 3, 2, 1, 1, 1, 1, 1, 1, 1)
    assert result["echo_idx"].values.tolist() == [0]
    np.testing.assert_array_equal(result.values.squeeze(), image.values)


def test_dimensions_are_transposed_to_the_canonical_order_with_their_values():
    image = _image(
        ("acquisition_idx", "read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        (2, 4, 3, 2, 5),
    )

    result = to_imaging_layout(image)

    assert result.dims.index("acquisition_idx") < result.dims.index("repetition_idx")
    assert result.dims[:3] == ("read_idx", "phase_idx", "slice_idx")
    xr.testing.assert_equal(
        result.squeeze(drop=True).drop_vars(["acquisition_idx", "repetition_idx"]),
        image.transpose(*result.squeeze().dims),
    )


def test_dimensions_outside_the_order_follow_it_in_their_relative_order():
    image = _image(
        ("time_idx", "read_idx", "phase_idx", "slice_idx", "other"), (2, 3, 3, 1, 2)
    )

    assert to_imaging_layout(image).dims[-2:] == ("time_idx", "other")


def test_attributes_are_kept_and_the_input_is_not_modified():
    image = _image(("slice_idx", "read_idx", "phase_idx"), (2, 4, 3))
    before = image.copy(deep=True)

    result = to_imaging_layout(image)

    assert result.attrs == image.attrs
    xr.testing.assert_identical(image, before)


def test_the_layout_is_idempotent():
    image = _image(("slice_idx", "read_idx", "phase_idx"), (2, 4, 3))
    once = to_imaging_layout(image)

    xr.testing.assert_identical(to_imaging_layout(once), once)


def test_complete_time_axes_label_padded_dimensions_without_mutation():
    image = _image(("read_idx", "phase_idx", "slice_idx"), (2, 3, 1))
    before = image.copy(deep=True)

    result = to_imaging_layout(
        image, sampling_time_s=np.array([0.0]), frame_time_s=np.array([0.0, 2.0])
    )

    assert result["acquisition_time_s"].dims == ("acquisition_idx",)
    assert result["time"].dims == ("repetition_idx",)
    np.testing.assert_array_equal(result["time"], [0.0])
    assert result["time"].attrs["unit"] == "s"
    assert result["acquisition_time_s"].attrs["unit"] == "s"
    xr.testing.assert_identical(image, before)
    xr.testing.assert_identical(
        to_imaging_layout(result, sampling_time_s=np.array([1.0])), result
    )


def test_readout_times_do_not_replace_a_different_sized_processed_axis():
    image = _image(
        ("read_idx", "phase_idx", "slice_idx", "acquisition_idx", "repetition_idx"),
        (2, 3, 1, 2, 3),
    )

    result = to_imaging_layout(
        image, sampling_time_s=np.arange(8.0), frame_time_s=np.arange(2.0)
    )

    assert "time" not in result.coords
    assert "acquisition_time_s" not in result.coords
    np.testing.assert_array_equal(result["acquisition_idx"], [0, 1])


def test_declared_echo_timing_takes_precedence_over_readout_metadata():
    image = _image(
        ("read_idx", "phase_idx", "slice_idx", "acquisition_idx"), (2, 3, 1, 2)
    ).assign_coords(acquisition_time_s=("acquisition_idx", [0.01, 0.02], {"unit": "s"}))

    result = to_imaging_layout(image, sampling_time_s=np.array([0.0, 0.001]))

    np.testing.assert_array_equal(result["acquisition_time_s"], [0.01, 0.02])


@pytest.mark.parametrize("missing", ["read_idx", "phase_idx", "slice_idx"])
def test_a_missing_spatial_dimension_raises(missing):
    dims = [dim for dim in ("read_idx", "phase_idx", "slice_idx") if dim != missing]

    with pytest.raises(ValueError, match=missing):
        to_imaging_layout(_image(dims, (2, 2)))


def test_docstring_examples_run():
    result = doctest.testmod(layout)

    assert result.attempted > 0
    assert result.failed == 0
