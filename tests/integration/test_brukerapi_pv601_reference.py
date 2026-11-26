"""Numerical integration tests against the public BrukerAPI PV6.0.1 corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypermri.loading.scans import create_experiment

DATASET_ID = "brukerapi_pv601_lego_phantom"
STABLE_CASES = {
    "csi": 30,
    "fair_rare": 9,
    "flash": 3,
    "isis": 37,
    "nspect": 33,
    "press": 35,
    "rare": 8,
    "rarevtr": 10,
    "singlepulse": 32,
    "steam": 36,
}
EXPECTED_METHODS = {
    "csi": "<Bruker:CSI>",
    "fair_rare": "<Bruker:FAIR_RARE>",
    "flash": "<Bruker:FLASH>",
    "isis": "<Bruker:ISIS>",
    "nspect": "<Bruker:NSPECT>",
    "press": "<Bruker:PRESS>",
    "rare": "<Bruker:RARE>",
    "rarevtr": "<Bruker:RAREVTR>",
    "singlepulse": "<Bruker:SINGLEPULSE>",
    "steam": "<Bruker:STEAM>",
}
SPECTROSCOPY_CASES = ("isis", "nspect", "press", "singlepulse", "steam")
IMAGING_SHAPES = {
    "flash": (256, 256, 3),
    "rare": (256, 256, 13),
    # ParaVision stores three inversion times and two FAIR modes.
    "fair_rare": (256, 256, 3, 2),
    # The last axis contains all three variable-TR frames.
    "rarevtr": (256, 128, 3),
}
# Raw image reconstructions use one dimension order; axes absent from a scan
# have length one.
RAW_IMAGING_DIMS = (
    "read_idx",
    "phase_idx",
    "slice_idx",
    "waterfat_idx",
    "inversion_idx",
    "echo_idx",
    "bvalue_idx",
    "channel_idx",
    "acquisition_idx",
    "repetition_idx",
)
# Lengths of the axes longer than one, read from each scan's method and acqp.
NEW_RAW_RECONSTRUCTION_CASES = {
    5: {"read_idx": 256, "phase_idx": 256, "slice_idx": 4, "acquisition_idx": 8},
    13: {"read_idx": 256, "phase_idx": 256, "slice_idx": 6, "acquisition_idx": 2},
    15: {"read_idx": 128, "phase_idx": 128, "slice_idx": 2, "bvalue_idx": 6},
    16: {"read_idx": 96, "phase_idx": 96},
    17: {"read_idx": 96, "phase_idx": 64, "inversion_idx": 3, "acquisition_idx": 2},
    18: {"read_idx": 96, "phase_idx": 64, "slice_idx": 5, "acquisition_idx": 4},
    19: {"read_idx": 96, "phase_idx": 64, "bvalue_idx": 29},
    # T1-EPI: the eight movie frames are inversion times.
    20: {"read_idx": 96, "phase_idx": 96, "inversion_idx": 8},
    21: {"read_idx": 64, "phase_idx": 64, "echo_idx": 5},
    22: {"read_idx": 64, "phase_idx": 64, "echo_idx": 4},
    45: {"read_idx": 128, "phase_idx": 96, "slice_idx": 5},
    48: {"read_idx": 128, "phase_idx": 96, "slice_idx": 16, "bvalue_idx": 45},
}
EPI_TRAJECTORY_CASES = {16, 17, 18, 19, 20, 21, 22, 45, 48}


@pytest.fixture(scope="module")
def reference_study(real_dataset_path) -> Path:
    """Return the downloaded, checksum-verified BrukerAPI phantom study."""
    return real_dataset_path(DATASET_ID)


def _reference_fid(study: Path, scan_id: int) -> np.ndarray:
    path = study / str(scan_id) / "fid.npz"
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["data"])


def _reference_2dseq(study: Path, scan_id: int) -> np.ndarray:
    path = study / str(scan_id) / "pdata" / "1" / "2dseq.npz"
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["data"]).squeeze()


def _align_multi_pack_planes(actual: np.ndarray, scan) -> np.ndarray:
    """Express multi-pack planes in ParaVision's reconstructed frame axes."""
    aligned = actual.copy()
    start = 0
    pack_counts = scan.handler.slices_per_pack
    orientations = np.atleast_1d(scan.handler.read_orientation)
    assert len(pack_counts) == len(orientations)

    for count, orientation in zip(pack_counts, orientations, strict=True):
        end = start + count
        if orientation == "H_F":
            transposed = np.swapaxes(aligned[:, :, start:end], 0, 1)
            assert transposed.shape == aligned[:, :, start:end].shape
            aligned[:, :, start:end] = transposed
        start = end

    assert start == aligned.shape[2]
    return aligned


def _aligned_imaging_frames(scan, sequence_id: str) -> np.ndarray:
    """Return reconstructed image frames in the pinned 2dseq axis order."""
    raw = scan.reconstructions["raw"]
    actual = np.asarray(raw).squeeze()

    if sequence_id in {"flash", "rare"}:
        return _align_multi_pack_planes(actual, scan)
    if sequence_id == "fair_rare":
        assert raw.sizes["acquisition_idx"] == 2
        assert raw.sizes["inversion_idx"] == 3
        # raw: read, phase, inversion, FAIR mode (as in 2dseq)
        return actual
    if sequence_id == "rarevtr":
        assert raw.sizes["acquisition_idx"] == 3
        # raw: read, phase, variable-TR frame (as in 2dseq)
        return actual
    raise AssertionError(f"Unsupported imaging reference case: {sequence_id}")


def _normalised_magnitude_frames(data: np.ndarray) -> np.ndarray:
    """L2-normalise every 2D magnitude frame independently."""
    magnitude = np.abs(data).astype(np.float64, copy=False)
    frames = magnitude.reshape(magnitude.shape[0] * magnitude.shape[1], -1)
    norms = np.linalg.norm(frames, axis=0)
    assert np.all(norms > 0)
    return frames / norms


@pytest.mark.integration
def test_manifest_maps_every_stable_reference_case(real_data_manifest):
    """Keep the executable cases synchronized with the reviewed manifest."""
    manifest_cases = real_data_manifest["datasets"][DATASET_ID]["stable_raw_cases"]
    assert manifest_cases == STABLE_CASES


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("sequence_id,scan_id", STABLE_CASES.items())
def test_stable_raw_case_reconstructs(
    reference_study: Path,
    sequence_id: str,
    scan_id: int,
):
    """Every stable allowlist entry must produce finite complex raw output."""
    scan = create_experiment(reference_study / str(scan_id), load_data=True)

    assert scan.type == EXPECTED_METHODS[sequence_id]
    assert "raw" in scan.reconstructions
    raw = scan.reconstructions["raw"]
    assert raw.size > 0
    assert np.iscomplexobj(raw.values)
    assert np.isfinite(raw.values.real).all()
    assert np.isfinite(raw.values.imag).all()
    if scan_id in EPI_TRAJECTORY_CASES:
        assert raw.attrs["epi_trajectory_regridding"] is True
        assert raw.attrs["epi_trajectory_source"] == "PVM_EpiTrajAdjkx"
        assert raw.attrs["epi_trajectory_points"] == raw.sizes["read_idx"]
        assert raw.attrs["epi_b0_trajectory_available"] is True
        assert raw.attrs["epi_b0_trajectory_correction"] is False


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("scan_id,expected", NEW_RAW_RECONSTRUCTION_CASES.items())
def test_new_raw_case_preserves_complete_axes(
    reference_study: Path,
    scan_id: int,
    expected: dict[str, int],
):
    """Previously failing raw cases must reconstruct all metadata-derived axes."""
    scan = create_experiment(reference_study / str(scan_id), load_data=True)

    assert "raw" in scan.reconstructions
    raw = scan.reconstructions["raw"]
    assert raw.dims == RAW_IMAGING_DIMS
    assert dict(raw.sizes) == {dim: expected.get(dim, 1) for dim in RAW_IMAGING_DIMS}
    assert np.iscomplexobj(raw.values)
    assert np.isfinite(raw.values.real).all()
    assert np.isfinite(raw.values.imag).all()


@pytest.mark.integration
@pytest.mark.slow
def test_dti_standard_raw_magnitude_matches_paravision(reference_study: Path):
    """DtiStandard raw frames must retain scanner image order and geometry."""
    scan_id = 15
    scan = create_experiment(reference_study / str(scan_id), load_data=True)
    actual = np.asarray(scan.reconstructions["raw"]).squeeze()
    expected = _reference_2dseq(reference_study, scan_id)

    assert actual.shape == expected.shape == (128, 128, 2, 6)
    actual_frames = _normalised_magnitude_frames(actual)
    expected_frames = _normalised_magnitude_frames(expected)
    frame_distances = np.linalg.norm(actual_frames - expected_frames, axis=0)

    actual_centred = actual_frames - actual_frames.mean(axis=0)
    expected_centred = expected_frames - expected_frames.mean(axis=0)
    correlations = np.sum(actual_centred * expected_centred, axis=0) / (
        np.linalg.norm(actual_centred, axis=0)
        * np.linalg.norm(expected_centred, axis=0)
    )

    assert float(frame_distances.max()) < 1e-3
    assert float(correlations.min()) > 0.99999


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("scan_id,min_correlation", [(16, 0.98), (45, 0.94)])
def test_basic_epi_raw_reconstruction_matches_paravision_geometry(
    reference_study: Path,
    scan_id: int,
    min_correlation: float,
):
    """Trajectory-aware EPI output must retain ParaVision's image geometry."""
    scan = create_experiment(reference_study / str(scan_id), load_data=True)
    actual = np.asarray(scan.reconstructions["raw"]).squeeze()
    expected = _reference_2dseq(reference_study, scan_id)

    assert actual.shape == expected.shape
    actual_frames = _normalised_magnitude_frames(actual)
    expected_frames = _normalised_magnitude_frames(expected)
    actual_centred = actual_frames - actual_frames.mean(axis=0)
    expected_centred = expected_frames - expected_frames.mean(axis=0)
    correlations = np.sum(actual_centred * expected_centred, axis=0) / (
        np.linalg.norm(actual_centred, axis=0)
        * np.linalg.norm(expected_centred, axis=0)
    )

    assert float(correlations.min()) > min_correlation


@pytest.mark.integration
@pytest.mark.slow
def test_csi_raw_matches_published_fid_reference(reference_study: Path):
    """Stock CSI must discard its leading padding and preserve the reference grid."""
    scan_id = STABLE_CASES["csi"]
    scan = create_experiment(reference_study / str(scan_id), load_data=True)
    raw = scan.reconstructions["raw"]
    reference = _reference_fid(reference_study, scan_id).squeeze()
    cutoff = int(scan.cut_off)
    expected = reference[cutoff:]
    actual = np.asarray(raw).squeeze()

    assert cutoff == 70
    assert raw.dims == (
        "acquisition_idx",
        "read_idx",
        "phase_idx",
        "slice_idx",
        "repetition_idx",
        "channel_idx",
    )
    assert raw.shape == (1978, 16, 16, 1, 1, 1)
    assert actual.shape == expected.shape == (1978, 16, 16)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    expected_time_s = np.arange(expected.shape[0], dtype=float) * scan.dwell_time_s
    for coord_name in ("acquisition_idx", "acquisition_time_s"):
        assert coord_name in raw.coords
        actual_time_s = np.asarray(raw.coords[coord_name])
        assert actual_time_s.shape == expected_time_s.shape
        np.testing.assert_allclose(actual_time_s, expected_time_s, rtol=0.0, atol=1e-15)
    assert raw.coords["acquisition_time_s"].attrs["unit"] == "s"

    np.testing.assert_allclose(
        np.asarray(raw.attrs["sampling_time_axis_s"]),
        expected_time_s,
        rtol=0.0,
        atol=1e-15,
    )
    assert len(np.asarray(scan.hz)) == expected.shape[0]
    assert len(np.asarray(scan.ppm)) == expected.shape[0]
    for name in ("freq_axis_hz", "freq_axis_ppm"):
        assert raw.coords[name].dims == ("acquisition_idx",)
        assert raw.coords[name].size == expected.shape[0]
        assert name not in raw.attrs


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("sequence_id", IMAGING_SHAPES)
def test_imaging_matches_paravision_reference(
    reference_study: Path,
    sequence_id: str,
):
    """Match every normalised image frame to ParaVision's 2dseq output."""
    scan_id = STABLE_CASES[sequence_id]
    scan = create_experiment(reference_study / str(scan_id), load_data=True)
    actual = _aligned_imaging_frames(scan, sequence_id)
    expected = _reference_2dseq(reference_study, scan_id)

    assert actual.shape == IMAGING_SHAPES[sequence_id]
    assert expected.shape == IMAGING_SHAPES[sequence_id]

    actual_frames = _normalised_magnitude_frames(actual)
    expected_frames = _normalised_magnitude_frames(expected)
    frame_distances = np.linalg.norm(actual_frames - expected_frames, axis=0)

    actual_centred = actual_frames - actual_frames.mean(axis=0)
    expected_centred = expected_frames - expected_frames.mean(axis=0)
    correlation_denominator = np.linalg.norm(actual_centred, axis=0) * np.linalg.norm(
        expected_centred,
        axis=0,
    )
    assert np.all(correlation_denominator > 0)
    correlations = np.sum(actual_centred * expected_centred, axis=0) / (
        correlation_denominator
    )

    assert float(frame_distances.max()) < 1e-3
    assert float(correlations.min()) > 0.99999


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("sequence_id", SPECTROSCOPY_CASES)
def test_spectroscopy_matches_published_fid_reference(
    reference_study: Path,
    sequence_id: str,
):
    """Match HyperMRI's documented cutoff and FFT to each published FID."""
    scan_id = STABLE_CASES[sequence_id]
    scan = create_experiment(reference_study / str(scan_id), load_data=True)
    raw = scan.reconstructions["raw"]
    actual = np.asarray(raw).squeeze()
    reference = _reference_fid(reference_study, scan_id).squeeze()
    cutoff = int(getattr(scan, "cut_off", 0))
    expected = np.fft.fftshift(
        np.fft.fft(reference[cutoff:], axis=0),
        axes=0,
    )

    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=0.5)

    npoints = raw.sizes["acquisition_idx"]
    expected_time_s = np.arange(npoints, dtype=float) * scan.dwell_time_s
    for coord_name in ("acquisition_idx", "acquisition_time_s"):
        assert coord_name in raw.coords
        np.testing.assert_allclose(
            np.asarray(raw.coords[coord_name]),
            expected_time_s,
            rtol=0.0,
            atol=1e-15,
        )
    np.testing.assert_allclose(
        np.asarray(raw.attrs["sampling_time_axis_s"]),
        expected_time_s,
        rtol=0.0,
        atol=1e-15,
    )
    assert raw.attrs["cut_off"] == cutoff
    assert "applied_cut_off" not in raw.attrs
    assert raw.attrs["raw_n_acquisition"] == reference.shape[0]
    for coord_name in ("freq_axis_hz", "freq_axis_ppm"):
        assert raw.coords[coord_name].size == npoints
        assert coord_name not in raw.attrs
    assert len(np.asarray(scan.sampling_time_axis)) == npoints
    assert len(np.asarray(scan.time_axis)) == npoints


@pytest.mark.integration
def test_reference_manifest_has_reviewed_attribution(real_data_manifest):
    """The executable data dependency must retain its CC BY attribution."""
    dataset = real_data_manifest["datasets"][DATASET_ID]
    citation_fields = {
        key: dataset[key]
        for key in (
            "creator",
            "dataset_version",
            "doi",
            "license",
            "license_url",
            "publisher",
            "source_url",
            "title",
        )
    }

    assert all(isinstance(value, str) and value for value in citation_fields.values())
    assert dataset["doi"] == "10.5281/zenodo.4048253"
    assert dataset["license"] == "CC-BY-4.0"
