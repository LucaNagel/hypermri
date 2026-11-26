from types import SimpleNamespace

import numpy as np
import xarray as xr

from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.sequences.base_spectroscopy import BaseSpectroscopy


class _DummySequence:
    def __init__(self, n_acquisition=2, n_repetitions=1):
        self.n_acquisition = n_acquisition
        self.n_repetitions = n_repetitions
        self.handler = object()


def _handler_with_order(encoding_order, read_positions, phase_positions):
    """Return a handler whose invented CSI description has the given order."""
    read_positions = np.asarray(read_positions)
    phase_positions = np.asarray(phase_positions)
    matrix = (int(read_positions.max()) + 1, int(phase_positions.max()) + 1)
    acquisition = CSIAcquisition(
        encoding_order=encoding_order,
        matrix=matrix,
        fov_mm=(10.0 * matrix[0], 10.0 * matrix[1]),
        fov_offset_mm=(0.0, 0.0, 0.0),
        read_positions=read_positions,
        phase_positions=phase_positions,
        n_points=4,
        dwell_time_s=None,
        reference_frequency_mhz=None,
        center_ppm=0.0,
    )
    return SimpleNamespace(to_csi_acquisition=lambda: acquisition)


def _handler_failing_with(error):
    """Return a handler whose CSI description lookup raises ``error``."""

    def _raise():
        raise error

    return SimpleNamespace(to_csi_acquisition=_raise)


def test_filter_signal_keeps_legacy_1d_flattening_behavior():
    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    input_2d = np.array([[1 + 0j, 2 + 0j, 3 + 0j], [4 + 0j, 5 + 0j, 6 + 0j]])
    input_1d = input_2d.T.flatten()

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_1d,
        filter_type="None",
        signal_end=1,
        filter_axis=0,
    )

    expected_2d = np.array([[1 + 0j, 2 + 0j, 3 + 0j], [0 + 0j, 0 + 0j, 0 + 0j]])
    expected_1d = expected_2d.T.flatten()
    np.testing.assert_array_equal(result, expected_1d)


def test_filter_signal_defaults_to_kspace_axis_for_flattened_raw_data():
    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    input_2d = np.array([[1 + 0j, 2 + 0j, 3 + 0j], [4 + 0j, 5 + 0j, 6 + 0j]])
    input_1d = input_2d.T.flatten()

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_1d,
        filter_type="None",
        signal_end=1,
    )

    expected_2d = np.array([[1 + 0j, 0 + 0j, 0 + 0j], [4 + 0j, 0 + 0j, 0 + 0j]])
    np.testing.assert_array_equal(result, expected_2d.T.flatten())


def test_filter_signal_exponential_defaults_to_kspace_acquisition_order():
    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    input_2d = np.ones((2, 3), dtype=np.complex128)

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_2d,
        filter_type="Exponential",
        decay_constant=np.log(2),
    )

    expected_2d = np.array([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]], dtype=np.complex128)
    np.testing.assert_allclose(result, expected_2d.T.flatten())


def test_filter_signal_can_remove_lowest_signal_points_on_legacy_spatial_axis():
    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    input_2d = np.array(
        [
            [10 + 0j, 1 + 0j, 5 + 0j, 2 + 0j],
            [10 + 0j, 1 + 0j, 5 + 0j, 2 + 0j],
        ]
    )

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_2d,
        filter_type="None",
        signal_end=2,
        remove_lowest_signal=True,
        filter_axis=1,
    )

    expected_2d = np.array(
        [
            [10 + 0j, 0 + 0j, 5 + 0j, 0 + 0j],
            [10 + 0j, 0 + 0j, 5 + 0j, 0 + 0j],
        ]
    )
    expected_1d = expected_2d.T.flatten()
    np.testing.assert_array_equal(result, expected_1d)


def test_filter_signal_xarray_can_opt_in_to_acquisition_idx_filtering():
    seq = _DummySequence(n_acquisition=3, n_repetitions=1)
    input_data = xr.DataArray(
        np.array([[1 + 0j, 2 + 0j, 3 + 0j], [4 + 0j, 5 + 0j, 6 + 0j]]),
        dims=("read_idx", "acquisition_idx"),
        coords={"read_idx": [0, 1], "acquisition_idx": [0, 1, 2]},
    )

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        filter_type="None",
        signal_end=1,
        filter_axis=0,
    )

    expected = xr.DataArray(
        np.array([[1 + 0j, 0 + 0j, 0 + 0j], [4 + 0j, 0 + 0j, 0 + 0j]]),
        dims=("read_idx", "acquisition_idx"),
        coords={"read_idx": [0, 1], "acquisition_idx": [0, 1, 2]},
    )
    xr.testing.assert_identical(result, expected)


def test_filter_signal_full_csi_can_opt_in_to_acquisition_idx_filtering():
    seq = _DummySequence(n_acquisition=3, n_repetitions=1)
    input_data = xr.DataArray(
        np.ones((3, 2, 2), dtype=np.complex128),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
    )

    seq.handler = _handler_failing_with(
        AssertionError("FID-axis filtering must not inspect k-space order")
    )

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        filter_type="Exponential",
        decay_constant=np.log(2),
        filter_axis=0,
    )

    expected = input_data * xr.DataArray([1.0, 0.5, 0.25], dims=("acquisition_idx",))
    xr.testing.assert_allclose(result, expected)


def test_filter_signal_xarray_can_remove_lowest_signal_points():
    seq = _DummySequence(n_acquisition=3, n_repetitions=1)
    input_data = xr.DataArray(
        np.array([[10 + 0j, 1 + 0j, 5 + 0j], [10 + 0j, 1 + 0j, 5 + 0j]]),
        dims=("read_idx", "acquisition_idx"),
        coords={"read_idx": [0, 1], "acquisition_idx": [0, 1, 2]},
    )

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        filter_type="None",
        signal_end=2,
        remove_lowest_signal=True,
        filter_dim="acquisition_idx",
    )

    expected = xr.DataArray(
        np.array([[10 + 0j, 0 + 0j, 5 + 0j], [10 + 0j, 0 + 0j, 5 + 0j]]),
        dims=("read_idx", "acquisition_idx"),
        coords={"read_idx": [0, 1], "acquisition_idx": [0, 1, 2]},
    )
    xr.testing.assert_identical(result, expected)


def test_filter_signal_xarray_spatial_filter_uses_csi_acquisition_order():
    seq = _DummySequence(n_acquisition=1, n_repetitions=1)
    input_data = xr.DataArray(
        np.array([[[10 + 0j, 20 + 0j], [30 + 0j, 40 + 0j]]]),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
        coords={"acquisition_idx": [0], "phase_idx": [0, 1], "read_idx": [0, 1]},
    )

    seq.handler = _handler_with_order(
        "LINEAR_ENC LINEAR_ENC", np.array([1, 0, 1, 0]), np.array([0, 1, 1, 0])
    )

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        filter_type="Exponential",
        decay_constant=np.log(2),
        filter_axis=1,
    )

    expected = xr.DataArray(
        np.array([[[1.25 + 0j, 20 + 0j], [15 + 0j, 10 + 0j]]]),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
        coords={"acquisition_idx": [0], "phase_idx": [0, 1], "read_idx": [0, 1]},
    )
    xr.testing.assert_allclose(result, expected)


def test_filter_signal_xarray_signal_end_counts_fids_not_fid_length():
    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    # 2 acquisition points per FID, 4 FIDs total (2x2 k-space)
    # Values encode FID identity for easy checks.
    input_data = xr.DataArray(
        np.array(
            [
                [[11 + 0j, 21 + 0j], [31 + 0j, 41 + 0j]],
                [[12 + 0j, 22 + 0j], [32 + 0j, 42 + 0j]],
            ]
        ),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
    )

    seq.handler = _handler_with_order(
        "LINEAR_ENC LINEAR_ENC", np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    )

    # signal_end=3 should keep first 3 FIDs in acquisition order and zero only the 4th FID.
    # If signal_end were interpreted as FID-length (acquisition axis), this would not zero anything.
    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        filter_type="None",
        signal_end=3,
    )

    expected = input_data.copy(deep=True)
    expected.loc[dict(phase_idx=1, read_idx=1)] = 0 + 0j
    xr.testing.assert_allclose(result, expected)
    assert result.attrs.get("filter_fid_order_mode") == "encoding_order"


def test_filter_signal_xarray_encoding_order_aligns_to_measured_kspace_center():
    seq = _DummySequence(n_acquisition=1, n_repetitions=1)
    input_data = xr.DataArray(
        np.array(
            [
                [
                    [1 + 0j, 2 + 0j, 3 + 0j],
                    [4 + 0j, 5 + 0j, 100 + 0j],
                    [7 + 0j, 8 + 0j, 9 + 0j],
                ]
            ]
        ),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
        attrs={"domain": "kspace"},
    )

    seq.handler = _handler_with_order(
        "CENTRIC_ENC CENTRIC_ENC",
        np.array([1, 0, 2, 0, 1, 2, 0, 1, 2]),
        np.array([1, 0, 0, 1, 0, 1, 2, 2, 2]),
    )

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        filter_type="None",
        signal_end=1,
    )

    expected = xr.zeros_like(input_data)
    expected.loc[dict(phase_idx=1, read_idx=2)] = 100 + 0j
    xr.testing.assert_allclose(result, expected)
    assert result.attrs.get("filter_fid_order_mode") == "encoding_order"


def test_filter_signal_xarray_fallback_preserves_native_kspace_order():
    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    # FID magnitudes (sum over acquisition): [200, 20, 120, 40]
    input_data = xr.DataArray(
        np.array(
            [
                [[100 + 0j, 10 + 0j], [60 + 0j, 20 + 0j]],
                [[100 + 0j, 10 + 0j], [60 + 0j, 20 + 0j]],
            ]
        ),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
    )

    seq.handler = _handler_failing_with(RuntimeError("encoding order unavailable"))

    # Keep the first two FIDs in native flattened k-space order. Magnitude
    # sorting would change the meaning of signal_end and is not used.
    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        filter_type="None",
        signal_end=2,
    )

    expected = input_data.copy(deep=True)
    expected.loc[dict(phase_idx=1)] = 0 + 0j
    xr.testing.assert_allclose(result, expected)
    assert result.attrs.get("filter_kspace_order_mode") == "native_order_fallback"


def test_filter_signal_native_mode_flattens_all_kspace_points():
    seq = _DummySequence(n_acquisition=1, n_repetitions=1)
    input_data = xr.DataArray(
        np.arange(1, 5, dtype=np.complex128).reshape(1, 2, 2),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
    )

    seq.handler = _handler_failing_with(
        AssertionError("native mode must not inspect encoding metadata")
    )

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_data,
        signal_end=2,
        spatial_order_mode="native",
    )

    expected = input_data.copy(deep=True)
    expected.loc[dict(phase_idx=1)] = 0 + 0j
    xr.testing.assert_allclose(result, expected)
    assert result.attrs.get("filter_kspace_order_mode") == "native_order"


def test_filter_signal_can_require_encoding_metadata():
    seq = _DummySequence(n_acquisition=1, n_repetitions=1)
    input_data = xr.DataArray(
        np.ones((1, 2, 2), dtype=np.complex128),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
    )

    seq.handler = _handler_failing_with(RuntimeError("encoding order unavailable"))

    with np.testing.assert_raises_regex(ValueError, "could not be resolved"):
        BaseSpectroscopy.filter_signal(
            seq,
            data=input_data,
            signal_end=2,
            spatial_order_mode="csi_acquisition_order",
        )


def test_filter_signal_flipangle_correction_divides_by_remaining_mz():
    # Regression: the pulse count was passed as an index array and crashed numpy.
    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    flipangle_deg = 30.0
    remaining_mz = np.cos(np.deg2rad(flipangle_deg)) ** np.arange(3)
    input_2d = np.ones((2, 3), dtype=np.complex128) * remaining_mz

    result = BaseSpectroscopy.filter_signal(
        seq,
        data=input_2d,
        apply_flipangle_correction=True,
        flipangle_deg=flipangle_deg,
    )

    np.testing.assert_allclose(result, np.ones(6, dtype=np.complex128), rtol=1e-12)


def test_filter_signal_spatial_input_zeroes_the_acquired_kspace_points():
    # Regression: spatial input was returned to a point-mirrored k-space (k -> -k),
    # so an asymmetric acquisition order removed the opposite samples.
    from hypermri.reconstruction.centred_dft import centered_ifft

    seq = _DummySequence(n_acquisition=2, n_repetitions=1)
    n_phase, n_read = 4, 5
    rng = np.random.default_rng(7)
    kspace = rng.normal(size=(2, n_phase, n_read)) + 1j * rng.normal(
        size=(2, n_phase, n_read)
    )
    # Dominant zero-frequency sample keeps the measured centre at n // 2.
    kspace[:, n_phase // 2, n_read // 2] = 50.0
    phase_order, read_order = np.divmod(np.arange(n_phase * n_read), n_read)
    seq.handler = _handler_with_order("linear", read_order, phase_order)
    spatial = xr.DataArray(
        centered_ifft(kspace, (1, 2)),
        dims=("acquisition_idx", "phase_idx", "read_idx"),
        attrs={"domain": "spatial"},
    )
    n_keep = 7

    result = BaseSpectroscopy.filter_signal(
        seq, data=spatial, filter_type="None", signal_end=n_keep
    )

    expected_kspace = kspace.reshape(2, -1).copy()
    expected_kspace[:, n_keep:] = 0
    expected = centered_ifft(expected_kspace.reshape(kspace.shape), (1, 2))
    np.testing.assert_allclose(result.values, expected, atol=1e-12)
