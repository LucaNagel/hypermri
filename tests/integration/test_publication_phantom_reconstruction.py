"""Read-only geometry and scanner-FOV checks against the owner's external water-phantom scans.

Set ``HYPERMRI_PUBLICATION_QUAD_CRYO_STUDY`` to the original quad-cryo study
directory. The acquired files stay outside Git. Numeric shape, orientation and
offset controls reject unrelated studies; no identifying metadata are retained.
The independent acquisition-derived default intentionally differs from scanner
FOV processing. Its strict expected parity failures remain as controls; the
explicit scanner option is validated separately against the original criterion.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hypermri.loading.scans import load_scan
from hypermri.reconstruction.layout import to_imaging_layout
from hypermri.reconstruction.selection import find_reconstruction
from hypermri.utils.coordinates.image_shifts import (
    determine_FOV_shift,
    get_orientation_flip_multipliers,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.vendor_bruker]

_STUDY_ENVIRONMENT = "HYPERMRI_PUBLICATION_QUAD_CRYO_STUDY"
_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
_POSITION_ATOL_MM = 1e-3
_CORRELATION_THRESHOLD = 0.99
_SCAN_LAYOUTS = {10: (128, 32, 32, 1), 14: (128, 64, 64, 4), 58: (128, 64, 64, 1)}


@pytest.fixture(scope="module", params=_SCAN_LAYOUTS)
def phantom_scan(request):
    """Load only the original acquisition's metadata before strategy checks."""
    root = os.environ.get(_STUDY_ENVIRONMENT)
    if not root:
        pytest.skip(f"Set {_STUDY_ENVIRONMENT} to the original quad-cryo phantom study")
    scan_id = request.param
    scan = load_scan(
        Path(root).expanduser() / str(scan_id), mode="lazy", log_mode="critical"
    )
    handler = scan.handler
    assert handler.spatial_dimension == "3D"
    assert handler.slice_orientation == "coronal"
    assert handler.read_orientation == "H_F"
    np.testing.assert_allclose(
        handler.slice_offset_mm, -3.0, atol=_POSITION_ATOL_MM, rtol=0
    )
    return scan_id, scan


@pytest.fixture(scope="module")
def phantom_reconstructions(phantom_scan):
    """Load raw and scanner strategies independently without bypassing a guard."""
    scan_id, scan = phantom_scan
    handler = scan.handler
    results = []
    for kind in ("raw", "seq2d"):
        strategy_class, options = find_reconstruction("bruker", scan.type, kind)
        result = to_imaging_layout(strategy_class().reconstruct(handler, **options))
        layout = tuple(result.sizes[dim] for dim in (*_SPATIAL_DIMS, "repetition_idx"))
        assert layout == _SCAN_LAYOUTS[scan_id], "Unexpected phantom scan layout"
        results.append(result)
    return results


def _projected_slice_grid(data):
    """Project VisuCore voxel centres onto the direction of increasing index."""
    xyz = np.stack(
        [data[name].transpose(*_SPATIAL_DIMS).values for name in ("x", "y", "z")],
        axis=-1,
    )
    direction = xyz[0, 0, -1] - xyz[0, 0, 0]
    direction /= np.linalg.norm(direction)
    return xyz @ direction


def test_scanner_slice_labels_agree_with_physical_voxel_positions(
    phantom_reconstructions,
):
    _, scanner = phantom_reconstructions
    projected = _projected_slice_grid(scanner)
    np.testing.assert_allclose(
        scanner.slice_idx, projected, atol=_POSITION_ATOL_MM, rtol=0
    )
    np.testing.assert_allclose(projected.mean(), 3.0, atol=_POSITION_ATOL_MM, rtol=0)


def test_raw_and_scanner_physical_voxel_positions_agree(phantom_reconstructions):
    raw, scanner = phantom_reconstructions
    for name in ("x", "y", "z"):
        np.testing.assert_allclose(
            raw[name], scanner[name], atol=_POSITION_ATOL_MM, rtol=0
        )


def test_public_loading_retains_raw_and_scanner_reconstructions(phantom_scan):
    """The ordinary loader must pass its unchanged geometry guard."""
    _, lazy_scan = phantom_scan
    loaded = load_scan(lazy_scan.path, mode="full", log_mode="critical")
    assert {"raw", "seq2d"} <= loaded.reconstructions.keys()


def test_raw_slice_labels_agree_with_physical_voxel_positions(phantom_reconstructions):
    raw, _ = phantom_reconstructions
    np.testing.assert_allclose(
        raw.slice_idx, _projected_slice_grid(raw), atol=_POSITION_ATOL_MM, rtol=0
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The independent acquisition FOV ramp intentionally differs from scanner mode",
)
def test_raw_magnitude_matches_scanner_images(
    phantom_reconstructions, magnitude_cross_correlation
):
    """Use the >0.99 correlation and zero-shift criterion per frame."""
    raw, scanner = phantom_reconstructions
    for repetition in range(raw.sizes["repetition_idx"]):
        first = raw.isel(repetition_idx=repetition)
        second = scanner.isel(repetition_idx=repetition)
        correlation, shift = magnitude_cross_correlation(first, second)
        assert shift == (0, 0, 0), (repetition, correlation, shift)
        assert correlation > _CORRELATION_THRESHOLD, (repetition, correlation)


def _translate_complex_slice(frame, displacement_vox, *, positive_nyquist=True):
    """Translate one frame by the image-domain Fourier shift theorem.

    This independent diagnostic uses the DFT of the complex image, not the
    package's k-space phase ramp or FOV metadata. Positive displacement moves
    content towards higher slice indices. The raw slice reconstruction uses
    a forward DFT of centred k-space: its -0.5-cycle/sample endpoint becomes
    +0.5 in the image DFT. Keep that positive Nyquist branch for even lengths;
    the two branches agree only at integer displacements. Magnitude is taken
    only after the inverse transform. ``positive_nyquist=False`` provides the
    ordinary negative-branch control, which is not equivalent at half a voxel.
    """
    values = frame.squeeze(drop=True).transpose(*_SPATIAL_DIMS).values
    frequencies = np.fft.fftfreq(values.shape[2])
    if positive_nyquist and values.shape[2] % 2 == 0:
        frequencies[values.shape[2] // 2] = 0.5
    multiplier = np.exp(-2j * np.pi * frequencies * displacement_vox)
    return np.fft.ifft(
        np.fft.fft(values.astype(np.complex128), axis=2) * multiplier,
        axis=2,
    )


def test_independent_half_voxel_translation_explains_scanner_magnitudes(
    phantom_scan, phantom_reconstructions, magnitude_cross_correlation
):
    """Test the validated scanner image translation and its opposite-sign control.

    A -0.5-voxel slice translation must pass the original criterion in
    every stored repetition. The opposite sign must leave a +1-voxel peak,
    rejecting that sign even when its peak correlation is high. Neither
    operation changes acquired samples, scanner geometry or raw defaults.
    """
    scan_id, _ = phantom_scan
    raw, scanner = phantom_reconstructions
    original_values = raw.values.copy()
    for repetition in range(raw.sizes["repetition_idx"]):
        frame = raw.isel(repetition_idx=repetition)
        reference = scanner.isel(repetition_idx=repetition)
        translated = _translate_complex_slice(frame, -0.5)
        correlation, shift = magnitude_cross_correlation(translated, reference)
        assert shift == (0, 0, 0), (repetition, correlation, shift)
        assert correlation > _CORRELATION_THRESHOLD, (repetition, correlation)

        opposite = _translate_complex_slice(frame, 0.5)
        _, opposite_shift = magnitude_cross_correlation(opposite, reference)
        assert opposite_shift == (0, 0, 1), (repetition, opposite_shift)

        if scan_id == 58:
            ordinary = _translate_complex_slice(frame, -0.5, positive_nyquist=False)
            ordinary_correlation, _ = magnitude_cross_correlation(ordinary, reference)
            assert ordinary_correlation < _CORRELATION_THRESHOLD, (
                repetition,
                ordinary_correlation,
            )
    np.testing.assert_array_equal(raw.values, original_values)


def test_declared_scanner_rotation_explains_magnitude_disagreement(
    phantom_scan, phantom_reconstructions, magnitude_cross_correlation
):
    """Validate the explicit scanner FOV option; default raw stays independent.

    These three coronal volumes have zero in-plane offsets. RECO_rotate declares
    2.5 or 5.5 slice voxels relative to its 0.5 baseline. The optional scanner
    mode adds one voxel and reverses the sign for the scoped transform
    convention. Its -3.5/-6.5 ramps differ from the independent -3/-6
    acquisition-offset ramps by half a voxel. Other scanner settings retain
    their independent contracts and are not inferred from this comparison.
    """
    _, scan = phantom_scan
    handler = scan.handler
    raw, scanner = phantom_reconstructions
    acquisition_shift, _ = determine_FOV_shift(handler)
    rotation = np.asarray(handler.reco["RECO_rotate"], dtype=float)
    assert rotation.shape == (3, 1)
    np.testing.assert_allclose(rotation[:2, 0], 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(acquisition_shift[:2], 0.0, rtol=0, atol=1e-12)
    assert handler.patient_pos == "Head_Supine"
    assert get_orientation_flip_multipliers(handler)[2] == 1
    declared_shift = (rotation[2, 0] - 0.5) * handler.matrix[2]
    # This observed centering adjustment is limited to the declared scan layout.
    candidate_shift = -(declared_shift + 1.0)
    np.testing.assert_allclose(
        candidate_shift - acquisition_shift[2], -0.5, rtol=0, atol=1e-12
    )
    original_values = raw.values.copy()
    original_fid = handler.fid.copy()
    # Store scanner geometry first so the public raw operation also exercises
    # the ordinary cross-reconstruction consistency guard.
    scan.reconstruct("seq2d")
    candidate = scan.reconstruct("raw", fov_shift_source="scanner")
    assert candidate.attrs["fov_shift_source"] == "scanner"
    assert candidate.attrs["fov_shift_vox"] == (0.0, 0.0, candidate_shift)
    assert candidate.attrs["bruker_reco_parameter_source"] == "pdata/1/reco"
    np.testing.assert_array_equal(candidate.attrs["bruker_reco_rotate"], rotation[:, 0])
    assert candidate.dtype == raw.dtype == np.complex64
    assert candidate.dims == raw.dims
    for coordinate in (*_SPATIAL_DIMS, "x", "y", "z"):
        np.testing.assert_array_equal(candidate[coordinate], raw[coordinate])
    for repetition in range(candidate.sizes["repetition_idx"]):
        independent = _translate_complex_slice(
            raw.isel(repetition_idx=repetition), -0.5
        )
        reconstructed = (
            candidate.isel(repetition_idx=repetition)
            .squeeze(drop=True)
            .transpose(*_SPATIAL_DIMS)
            .values
        )
        # One part per million of peak complex amplitude covers float32
        # storage rounding without dividing by near-zero background samples.
        np.testing.assert_allclose(
            reconstructed, independent, rtol=0, atol=1e-6 * np.max(np.abs(independent))
        )
        correlation, shift = magnitude_cross_correlation(
            candidate.isel(repetition_idx=repetition),
            scanner.isel(repetition_idx=repetition),
        )
        assert shift == (0, 0, 0), (repetition, correlation, shift)
        assert correlation > _CORRELATION_THRESHOLD, (repetition, correlation)
    np.testing.assert_array_equal(raw.values, original_values)
    np.testing.assert_array_equal(handler.fid, original_fid)
