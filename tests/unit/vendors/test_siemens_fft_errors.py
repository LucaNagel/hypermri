"""Preserve Siemens transforms while making transform failures visible."""

import numpy as np
import pytest
import xarray as xr

from hypermri.vendors.siemens import SiemensHandler


def _invented_data(dtype):
    values = np.arange(30).reshape(3, 5, 2) + 1j * np.arange(30, 60).reshape(3, 5, 2)
    return xr.DataArray(
        values.astype(dtype),
        dims=("phase_idx", "read_idx", "echo_idx"),
        coords={"phase_idx": [2, 1, 0], "read_idx": [0, 1, 2, 3, 4]},
        attrs={"units": "a.u.", "invented_fixture": True},
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("inverse", [False, True])
def test_siemens_transform_preserves_values_labels_and_input(dtype, inverse):
    data = _invented_data(dtype)
    before = data.copy(deep=True)
    axes = (1, 0)
    shift = np.fft.ifftshift if inverse else np.fft.fftshift
    transform = np.fft.ifftn if inverse else np.fft.fftn
    expected = shift(transform(shift(data.values, axes=axes), axes=axes), axes=axes)
    handler = SiemensHandler.__new__(SiemensHandler)
    actual = handler.ft(data, ift=inverse)
    tolerance = 1e-6 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(actual.values, expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == expected.dtype
    xr.testing.assert_identical(actual.coords.to_dataset(), data.coords.to_dataset())
    assert actual.dims == data.dims
    assert actual.attrs == data.attrs
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failed_second_axis_propagates_without_returning_partial_data(
    monkeypatch, inverse, error_type
):
    data = _invented_data(np.complex64)
    before = data.copy(deep=True)
    name = "ifft" if inverse else "fft"
    original = getattr(np.fft, name)
    error = error_type("invented second-axis failure")
    calls = []

    def transform(values, axis):
        calls.append(axis)
        if len(calls) == 2:
            raise error
        return original(values, axis=axis)

    monkeypatch.setattr(np.fft, name, transform)
    handler = SiemensHandler.__new__(SiemensHandler)
    with pytest.raises(error_type) as caught:
        handler.ft(data, ift=inverse)
    assert caught.value is error
    assert calls == [1, 0]
    xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("third_counter", ["Sli", "Par"])
def test_siemens_preview_distinguishes_slices_and_encoded_partitions(third_counter):
    """An invented Fourier pair validates 2D slices and rejects unverified 3D.

    The third counter is carried explicitly through the actual TWIX dimension
    mapper. Only the 2D Fourier reference is claimed validated, at 1e-12.
    """
    handler = SiemensHandler.__new__(SiemensHandler)
    handler.fov_mm = np.array([40, 40, 40])
    handler.raw_loader = "twixtools"
    image = np.zeros((4, 4, 4), dtype=complex)
    image[1, 0, 2] = 1
    axes = (0, 1) if third_counter == "Sli" else (0, 1, 2)
    kspace = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes
    )
    labelled = handler._to_standard_imaging_kspace_xarray(
        kspace, ["Col", "Lin", third_counter]
    )
    before = labelled.copy(deep=True)
    encoded = labelled.attrs["fourier_encoded_dims"]
    assert ("slice_idx" in encoded) == (third_counter == "Par")
    if third_counter == "Par":
        with pytest.raises(NotImplementedError, match="3D partition"):
            handler._image_preview(labelled)
    else:
        preview = handler._image_preview(labelled)
        np.testing.assert_allclose(
            preview.squeeze().transpose("read_idx", "phase_idx", "slice_idx"),
            image,
            rtol=1e-12,
            atol=1e-12,
        )
        assert preview.attrs["domain"] == "spatial"
    xr.testing.assert_identical(labelled, before)
