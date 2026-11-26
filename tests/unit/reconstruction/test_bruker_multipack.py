"""Regression tests for Bruker raw multi-slice-pack acquisitions."""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.reconstruction.strategies.bruker.rare import BrukerRareRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.raw import BrukerRawRecoStrategy
from hypermri.utils.coordinates.image_shifts import get_orientation_flip_multipliers


def test_raw_spatial_coordinates_fall_back_to_indices_without_regular_rps_grid():
    dims = (
        "acquisition_idx",
        "repetition_idx",
        "slice_idx",
        "phase_idx",
        "channel_idx",
        "read_idx",
    )
    shape = (1, 1, 13, 256, 1, 256)
    handler = SimpleNamespace(rps_grid_mm=None)

    coords = BrukerRawRecoStrategy._spatial_coordinates(handler, dims, shape)

    assert tuple(coords) == ("read_idx", "phase_idx", "slice_idx")
    assert coords["read_idx"][2]["unit"] == "index"
    np.testing.assert_array_equal(coords["read_idx"][1], np.arange(256))
    np.testing.assert_array_equal(coords["phase_idx"][1], np.arange(256))
    np.testing.assert_array_equal(coords["slice_idx"][1], np.arange(13))


def test_segmented_flash_reshape_honours_acquisition_phase_factor():
    read_size, phase_size, slice_size, phase_factor = 2, 4, 3, 2
    fid_rows = []
    for phase_segment in range(phase_size // phase_factor):
        for slice_idx in range(slice_size):
            for phase_in_segment in range(phase_factor):
                phase_idx = phase_segment * phase_factor + phase_in_segment
                fid_rows.append(100 * phase_idx + 10 * slice_idx + np.arange(read_size))
    fid = np.asarray(fid_rows, dtype=np.complex64)
    handler = SimpleNamespace(
        n_repetitions=1,
        enc_matrix=np.array([read_size, phase_size, slice_size]),
        matrix=np.array([read_size, phase_size, slice_size]),
        n_receivers=1,
        spatial_dimension="2D",
        acqp={"ACQ_phase_factor": phase_factor},
        method={"PVM_EncSteps1": np.arange(phase_size) - phase_size // 2},
    )

    kspace = BrukerRawRecoStrategy().reshape_fid_to_kspace(fid, handler)

    assert kspace.shape == (1, 1, slice_size, phase_size, 1, read_size)
    for slice_idx in range(slice_size):
        for phase_idx in range(phase_size):
            expected = 100 * phase_idx + 10 * slice_idx + np.arange(read_size)
            np.testing.assert_array_equal(
                kspace[0, 0, slice_idx, phase_idx, 0, :],
                expected,
            )


@pytest.mark.parametrize("phase_factor", [0, 1.5, 3, "invalid", [1, 2]])
def test_segmented_reshape_rejects_invalid_acquisition_phase_factor(phase_factor):
    handler = SimpleNamespace(
        n_repetitions=1,
        enc_matrix=np.array([2, 4, 1]),
        matrix=np.array([2, 4, 1]),
        n_receivers=1,
        spatial_dimension="2D",
        acqp={"ACQ_phase_factor": phase_factor},
        method={"PVM_EncSteps1": np.arange(4) - 2},
    )

    with pytest.raises(ValueError, match="ACQ_phase_factor"):
        BrukerRawRecoStrategy().reshape_fid_to_kspace(
            np.zeros((4, 2), dtype=np.complex64),
            handler,
        )


def test_common_multi_pack_orientation_flip_is_applied_globally():
    handler = SimpleNamespace(
        patient_pos="Head_Prone",
        slice_orientation=np.array(["axial", "sagittal", "coronal"]),
        read_orientation=np.array(["L_R", "H_F", "H_F"]),
        spatial_dimension="2D",
    )

    np.testing.assert_array_equal(
        get_orientation_flip_multipliers(handler),
        np.array([1, 1, 1]),
    )


def test_unlisted_multi_pack_orientation_is_rejected():
    """The unlisted package gave the multipliers +1, +1 and +1 for all packages."""
    handler = SimpleNamespace(
        patient_pos="Head_Prone",
        slice_orientation=np.array(["axial", "sagittal"]),
        read_orientation=np.array(["L_R", "unexpected"]),
        spatial_dimension="2D",
    )

    with pytest.raises(ValueError, match="read_orientation 'unexpected'"):
        get_orientation_flip_multipliers(handler)


def test_multi_pack_axial_and_coronal_3d_share_their_flips():
    """Regression test: 3D Head_Prone coronal no longer reverses slices."""
    handler = SimpleNamespace(
        patient_pos="Head_Prone",
        slice_orientation=np.array(["axial", "coronal"]),
        read_orientation=np.array(["L_R", "L_R"]),
        spatial_dimension="3D",
    )

    np.testing.assert_array_equal(get_orientation_flip_multipliers(handler), [1, 1, 1])


def test_orientation_lists_of_different_length_are_rejected():
    handler = SimpleNamespace(
        patient_pos="Head_Prone",
        slice_orientation=np.array(["axial", "sagittal", "coronal"]),
        read_orientation=np.array(["L_R", "H_F"]),
        spatial_dimension="2D",
    )

    with pytest.raises(ValueError, match="3 entries .* 2"):
        get_orientation_flip_multipliers(handler)


def _rare_handler(
    *, nr, inversion_count=1, phase_size=4, rare_factor=2, fair_mode=None
):
    return SimpleNamespace(
        expnum=1,
        rare_factor=rare_factor,
        n_receivers=1,
        matrix=np.array([3, phase_size, 1]),
        enc_matrix=np.array([3, phase_size, 1]),
        n_repetitions=1,
        n_inversion_time=inversion_count,
        inversion_time_ms=np.arange(inversion_count),
        fat_water_separation="Off",
        spatial_dimension="2D",
        acqp={"NR": nr},
        method={} if fair_mode is None else {"PVM_FairMode": fair_mode},
    )


def _run_synthetic_rare_postprocess(monkeypatch, raw, handler):
    strategy = BrukerRareRawRecoStrategy()
    monkeypatch.setattr(
        strategy, "postprocess_fid_file", lambda values, handler: values
    )
    monkeypatch.setattr(
        strategy,
        "_get_phase_encode_order",
        lambda handler: (np.arange(handler.enc_matrix[1]), None),
    )
    handler.fid = raw.ravel()
    return strategy.postprocess_fid_rare(handler)


def test_rarevtr_preserves_acq_nr_as_acquisition_axis(monkeypatch):
    handler = _rare_handler(nr=3)
    raw = np.zeros((3, 2, 2, 1, 3), dtype=np.complex64)
    for acquisition_idx in range(3):
        raw[acquisition_idx, ...] = acquisition_idx + 1

    processed = _run_synthetic_rare_postprocess(monkeypatch, raw, handler)

    assert processed.shape == (3, 1, 1, 4, 1, 3)
    for acquisition_idx in range(3):
        assert np.all(processed[acquisition_idx] == acquisition_idx + 1)


def test_fair_rare_preserves_mode_and_inversion_axes_from_acq_nr(monkeypatch):
    handler = _rare_handler(nr=6, inversion_count=3, fair_mode="INTERLEAVED")
    raw = np.zeros((2, 3, 2, 2, 1, 3), dtype=np.complex64)
    for mode_idx in range(2):
        for inversion_idx in range(3):
            raw[mode_idx, inversion_idx, ...] = 10 * mode_idx + inversion_idx

    processed = _run_synthetic_rare_postprocess(monkeypatch, raw, handler)

    assert processed.shape == (2, 1, 1, 4, 3, 1, 3)
    for mode_idx in range(2):
        for inversion_idx in range(3):
            assert np.all(
                processed[mode_idx, 0, 0, :, inversion_idx, 0, :]
                == 10 * mode_idx + inversion_idx
            )


def test_fair_rare_interleaved2_preserves_alternating_mode_order(monkeypatch):
    handler = _rare_handler(nr=6, inversion_count=3, fair_mode="INTERLEAVED2")
    raw = np.zeros((3, 2, 2, 2, 1, 3), dtype=np.complex64)
    for inversion_idx in range(3):
        for mode_idx in range(2):
            raw[inversion_idx, mode_idx, ...] = 10 * mode_idx + inversion_idx

    processed = _run_synthetic_rare_postprocess(monkeypatch, raw, handler)

    assert processed.shape == (2, 1, 1, 4, 3, 1, 3)
    for mode_idx in range(2):
        for inversion_idx in range(3):
            assert np.all(
                processed[mode_idx, 0, 0, :, inversion_idx, 0, :]
                == 10 * mode_idx + inversion_idx
            )


@pytest.mark.parametrize("nr", [0, 2.5, "invalid", [3, 6]])
def test_rare_rejects_invalid_acquisition_repetition_count(nr):
    handler = _rare_handler(nr=nr)

    with pytest.raises(ValueError, match="ACQ NR"):
        BrukerRareRawRecoStrategy._acquisition_loop_count(handler, 1, 1)


@pytest.mark.parametrize("nr", [2, 7])
def test_fair_rare_rejects_acq_nr_that_cannot_be_factorised(nr):
    handler = _rare_handler(nr=nr, inversion_count=3, fair_mode="INTERLEAVED")

    with pytest.raises(ValueError, match="Cannot map Bruker ACQ NR"):
        BrukerRareRawRecoStrategy._acquisition_loop_count(handler, 1, 3)


def test_fair_rare_rejects_unknown_loop_order():
    handler = _rare_handler(nr=6, inversion_count=3, fair_mode="UNKNOWN")
    data = np.zeros((6, 1), dtype=np.complex64)

    with pytest.raises(ValueError, match="PVM_FairMode"):
        BrukerRareRawRecoStrategy._reshape_fair_iterations(data, handler, 2, 1, 3)


def test_rare_partial_fourier_uses_declared_phase_encoding_positions():
    handler = SimpleNamespace(method={"PVM_EncSteps1": np.array([-2, -1, 0, 1])})

    indices = BrukerRareRawRecoStrategy._phase_target_indices(
        handler,
        final_phase_size=8,
        encoded_phase_size=4,
    )

    np.testing.assert_array_equal(indices, np.array([2, 3, 4, 5]))


def test_rare_asymmetric_partial_fourier_is_not_centered():
    handler = SimpleNamespace(method={"PVM_EncSteps1": np.array([0, 1, 2, 3])})

    indices = BrukerRareRawRecoStrategy._phase_target_indices(
        handler,
        final_phase_size=8,
        encoded_phase_size=4,
    )

    np.testing.assert_array_equal(indices, np.array([4, 5, 6, 7]))


def test_rare_partial_fourier_centres_only_when_encoding_steps_are_missing():
    handler = SimpleNamespace(method={})

    indices = BrukerRareRawRecoStrategy._phase_target_indices(
        handler,
        final_phase_size=8,
        encoded_phase_size=4,
    )

    np.testing.assert_array_equal(indices, np.array([2, 3, 4, 5]))


@pytest.mark.parametrize(
    "encoding_steps,match",
    [
        ([-2, -1, 0], "length"),
        ([-2, -1, 0, 0], "repeat"),
        ([-5, -1, 0, 1], "outside"),
        ([-2, -1, 0.5, 1], "integer"),
    ],
)
def test_rare_partial_fourier_rejects_invalid_encoding_steps(
    encoding_steps,
    match,
):
    handler = SimpleNamespace(method={"PVM_EncSteps1": encoding_steps})

    with pytest.raises(ValueError, match=match):
        BrukerRareRawRecoStrategy._phase_target_indices(
            handler,
            final_phase_size=8,
            encoded_phase_size=4,
        )


def _rare_structural_result(monkeypatch, spatial, fair, waterfat):
    """Trace every synthetic 2D/3D, FAIR and fat-water loop without scanner data."""
    handler = _rare_handler(
        nr=4 if fair else 2,
        inversion_count=2 if fair else 1,
        fair_mode="INTERLEAVED" if fair else None,
    )
    handler.spatial_dimension = spatial
    handler.n_receivers = 2
    handler.n_repetitions = 2
    handler.fat_water_separation = "On" if waterfat else "Off"
    handler.method["PVM_FwsLoop"] = 2
    handler.matrix = np.array([3, 6, 3])
    handler.enc_matrix = np.array([3, 4, 2])
    handler.method["PVM_EncSteps1"] = np.array([-2, -1, 0, 1])
    count = 2 * (2 if fair else 1) * 3 * 4 * 2 * 2 * (2 if waterfat else 1)
    raw = np.arange(count).astype(np.complex128) * (1 + 0.125j)
    return _run_synthetic_rare_postprocess(monkeypatch, raw, handler)


# Recorded from the working checkout before the reconstruction rewrite on 6 October 2026 (NumPy 2.2.6).
# Synthetic samples encode loop positions; permutations and zero fill are exact.
_RARE_STRUCTURAL_REFERENCES = {
    (
        "2D",
        False,
        False,
    ): "e3dcf178d36e8810c8486fc1a7dadc134ac0cff050d01f91c80bd098facd2ea2",
    (
        "2D",
        False,
        True,
    ): "1cbe2d609581505ea81a4cdcc1d7cce1bc852317715f5bf919a32fd0753216f2",
    (
        "2D",
        True,
        False,
    ): "9ba9cfa1ab2bae9890e1da5c6dae521cec2e545cbc87e9011d4e01659d364671",
    (
        "2D",
        True,
        True,
    ): "129b34f22b1cee1421f133004b649d8186ad2a1cb9c846f147b06c8df8d14d6a",
    (
        "3D",
        False,
        False,
    ): "8b54b63319932e64bb7569ecb0d5376c5ba9b464fb1071af34ab6463f44ae46c",
    (
        "3D",
        False,
        True,
    ): "64cebd640b925407ab509878280738987ca309f4bf0cfbf19320cdcdecacbd4c",
    (
        "3D",
        True,
        False,
    ): "adddd8b1f75f83d98da6526e9e093d9e3c11e70d9fc5a9962d02f9622274f336",
    (
        "3D",
        True,
        True,
    ): "c4c7c6a71a5110b7678103a757c449098a4807c70df214449da3d83b060801da",
}


@pytest.mark.parametrize("spatial", ["2D", "3D"])
@pytest.mark.parametrize("fair", [False, True])
@pytest.mark.parametrize("waterfat", [False, True])
def test_rare_structural_reference(monkeypatch, spatial, fair, waterfat):
    """Exact baseline: sorting/zero filling are permutations and zero insertion."""
    import hashlib

    result = _rare_structural_result(monkeypatch, spatial, fair, waterfat)
    assert (
        hashlib.sha256(result.tobytes()).hexdigest()
        == _RARE_STRUCTURAL_REFERENCES[(spatial, fair, waterfat)]
    )
    assert result.dtype == np.complex128


def test_3d_rare_sorts_both_phase_encoding_directions(monkeypatch):
    """3D RARE places every echo at its PVM_EncSteps1/PVM_EncSteps2 position.

    Each synthetic sample encodes 100 x second phase step + 10 x first phase
    step + read index + 1000 x receiver, in Bruker acquisition order (second
    phase direction outermost, then echoes, receivers, read samples). The
    encoding and final matrices are equal, so no zero filling happens.
    """
    steps1 = np.array([1, -2, 0, -1])
    steps2 = np.array([0, -1, 1])
    n_read, n_channels = 2, 2
    raw = np.array(
        [
            [[100 * a + 10 * b + r + 1000 * c for r in range(n_read)]]
            for a in steps2
            for b in steps1
            for c in range(n_channels)
        ],
        dtype=np.complex128,
    )
    handler = _rare_handler(nr=1)
    handler.spatial_dimension = "3D"
    handler.n_receivers = n_channels
    handler.matrix = np.array([n_read, 4, 3])
    handler.enc_matrix = np.array([n_read, 4, 3])
    handler.method = {"PVM_EncSteps1": steps1, "PVM_EncSteps2": steps2}
    handler.method_parameter = handler.method.__getitem__
    strategy = BrukerRareRawRecoStrategy()
    monkeypatch.setattr(
        strategy, "postprocess_fid_file", lambda values, handler: values
    )
    handler.fid = raw.ravel()

    result = strategy.postprocess_fid_rare(handler)

    assert result.shape == (1, 1, 3, 4, n_channels, n_read)
    expected = (
        100 * np.sort(steps2)[:, None, None, None]
        + 10 * np.sort(steps1)[None, :, None, None]
        + 1000 * np.arange(n_channels)[None, None, :, None]
        + np.arange(n_read)[None, None, None, :]
    )
    np.testing.assert_array_equal(result[0, 0].real, expected)
