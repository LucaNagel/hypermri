import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import shift

from hypermri.analysis.dce.motion_correction import (
    dce_motion_correction as canonical_dce_motion_correction,
)
from hypermri.analysis.motion_correction import (
    apply_motion_corrector,
    dce_motion_correction,
)


def _make_shifted_dce_series() -> xr.DataArray:
    base = np.zeros((24, 24), dtype=np.float64)
    base[5:10, 4:8] = 1.0
    base[14:19, 13:20] = 0.6
    base[8:12, 15:17] = 0.3

    shifted_frame = shift(
        base,
        shift=(2.0, -1.0),
        order=0,
        mode="constant",
        cval=0.0,
    )

    frame0 = np.stack([base, base * 1.5], axis=2)
    frame1 = np.stack([shifted_frame, shifted_frame * 1.5], axis=2)
    values = np.stack([frame0, frame1], axis=3)
    values = values[:, :, :, None, :]

    return xr.DataArray(
        values,
        dims=("read_idx", "phase_idx", "channel_idx", "echo_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(values.shape[0]),
            "phase_idx": np.arange(values.shape[1]),
            "channel_idx": np.arange(values.shape[2]),
            "echo_idx": [0],
            "repetition_idx": np.arange(values.shape[4]),
        },
        name="dce_signal",
        attrs={"repetition_time_ms": 50.0, "flipangle_deg": 15.0},
    )


def _assert_layout_preserved(input_data: xr.DataArray, corrected: xr.DataArray) -> None:
    assert corrected.dims == input_data.dims
    assert corrected.shape == input_data.shape
    for coord_name in input_data.coords:
        np.testing.assert_array_equal(
            np.asarray(corrected.coords[coord_name]),
            np.asarray(input_data.coords[coord_name]),
        )


def _frame_mse(data: xr.DataArray) -> float:
    return float(
        np.mean(
            (
                data.isel(channel_idx=0, echo_idx=0, repetition_idx=1).values
                - data.isel(channel_idx=0, echo_idx=0, repetition_idx=0).values
            )
            ** 2
        )
    )


def test_dce_motion_correction_default_skimage_preserves_layout_and_alignment():
    input_data = _make_shifted_dce_series()

    corrected, metadata = dce_motion_correction(
        input_data,
        upsample_factor=20,
        return_metadata=True,
    )

    _assert_layout_preserved(input_data, corrected)

    pre_mse = _frame_mse(input_data)
    post_mse = _frame_mse(corrected)

    assert post_mse < pre_mse
    assert metadata["method"] == "skimage"
    assert metadata["spatial_dims"] == ("read_idx", "phase_idx")
    assert metadata["estimated_shifts"].shape == (2, 2)
    assert corrected.attrs["motion_correction_parameters"]["method"] == "skimage"


def test_dce_motion_correction_skimage_preserves_xarray_layout_and_improves_alignment():
    input_data = _make_shifted_dce_series()

    corrected, metadata = dce_motion_correction(
        input_data,
        correction_method="skimage",
        upsample_factor=20,
        return_metadata=True,
    )

    _assert_layout_preserved(input_data, corrected)

    pre_mse = _frame_mse(input_data)
    post_mse = _frame_mse(corrected)

    assert post_mse < pre_mse
    assert metadata["method"] == "skimage"
    assert metadata["spatial_dims"] == ("read_idx", "phase_idx")
    assert metadata["estimated_shifts"].shape == (2, 2)
    np.testing.assert_allclose(metadata["estimated_shifts"][0], (0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(metadata["estimated_shifts"][1], (-2.0, 1.0), atol=0.25)
    assert corrected.attrs["motion_correction_parameters"]["method"] == "skimage"


def test_apply_motion_corrector_accepts_mapping_configuration():
    input_data = _make_shifted_dce_series()

    corrected = apply_motion_corrector(
        input_data,
        {
            "correction_method": "skimage",
            "upsample_factor": 20,
        },
    )

    _assert_layout_preserved(input_data, corrected)
    assert _frame_mse(corrected) < _frame_mse(input_data)


def test_apply_motion_corrector_mapping_defaults_to_skimage():
    input_data = _make_shifted_dce_series()

    corrected = apply_motion_corrector(
        input_data,
        {
            "upsample_factor": 20,
        },
    )

    _assert_layout_preserved(input_data, corrected)
    assert _frame_mse(corrected) < _frame_mse(input_data)


def test_dce_motion_correction_is_exposed_from_canonical_dce_module():
    input_data = _make_shifted_dce_series()

    corrected = canonical_dce_motion_correction(
        input_data,
        correction_method="skimage",
        upsample_factor=20,
    )

    _assert_layout_preserved(input_data, corrected)
    assert _frame_mse(corrected) < _frame_mse(input_data)


def test_dce_motion_correction_rejects_unknown_backend():
    input_data = _make_shifted_dce_series()

    with pytest.raises(ValueError, match="Unsupported correction_method"):
        dce_motion_correction(input_data, correction_method="unknown")
