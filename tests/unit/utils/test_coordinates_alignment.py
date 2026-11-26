"""Slice planes and in-plane rotations in ``compute_scan_alignment``.

The scans are synthetic: an axial reference with the identity direction matrix
and sources tilted about their read direction or rotated about their slice
direction.  A tilt by ``theta`` moves the corners of a source field of view with
phase extent ``P`` by ``P * sin(theta)`` along the reference slice axis.  An
in-plane rotation by ``phi`` from the nearest multiple of 90 degrees moves the
two ends of each edge of length ``L`` apart by ``L * sin(phi)`` along the
reference read or phase axis that the edge crosses; the longest edge is the
phase extent.  ``tests/unit/utils/test_coordinates_paravision_reference.py``
checks parallel and crossing planes against the DICOM exports of the ParaVision reference series.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.utils.coordinates.alignment import compute_scan_alignment

# The position tolerance of the ParaVision reference tests, which is also the
# tilt and the in-plane edge spread that compute_scan_alignment accepts.
# Parallel pairs of local scans tilt by at most 1e-12 mm, crossing pairs by at
# least 10 mm.
POSITION_ATOL_MM = 1e-3

PHASE_EXTENT_MM = 40.0
SOURCE_EXTENT_MM = (-16.0, 16.0, -20.0, 20.0, -1.0, 1.0)

# In-plane rotation that moves the ends of the 40 mm phase edge apart by the
# tolerance along the reference axis it crosses, about 0.0014 degrees.
IN_PLANE_ATOL_DEG = float(np.degrees(np.arcsin(POSITION_ATOL_MM / PHASE_EXTENT_MM)))


def _scan(direction_matrix, extent_mm=SOURCE_EXTENT_MM):
    """Return a scan with three voxels per axis spanning ``extent_mm``."""
    axes = [
        np.linspace(low + (high - low) / 6, high - (high - low) / 6, 3)
        for low, high in zip(extent_mm[::2], extent_mm[1::2], strict=True)
    ]
    return SimpleNamespace(
        acquisition_matrix=np.asarray(direction_matrix, dtype=float),
        rps_grid_mm=np.meshgrid(*axes, indexing="ij"),
        rps_extent_mm=list(extent_mm),
    )


def _tilted_about_read(tilt_mm):
    """Return direction rows tilting the phase extent by ``tilt_mm`` in z."""
    sin_theta = tilt_mm / PHASE_EXTENT_MM
    cos_theta = np.sqrt(1.0 - sin_theta**2)
    return [
        [1.0, 0.0, 0.0],
        [0.0, cos_theta, sin_theta],
        [0.0, -sin_theta, cos_theta],
    ]


def _rotated_in_plane(angle_deg, reflected=False):
    """Return axial direction rows with read rotated by ``angle_deg`` about z.

    ``reflected`` reverses the phase direction, so the in-plane block is a
    reflection instead of a rotation.
    """
    phi = np.radians(angle_deg)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    phase = [sin_phi, -cos_phi, 0.0] if reflected else [-sin_phi, cos_phi, 0.0]
    return [[cos_phi, sin_phi, 0.0], phase, [0.0, 0.0, 1.0]]


AXIAL = np.eye(3)
CORONAL = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]


def test_tilt_beyond_the_tolerance_is_rejected():
    """Twice the tolerance is no rounding of the direction matrices."""
    source = _scan(_tilted_about_read(2 * POSITION_ATOL_MM))

    with pytest.raises(ValueError, match="not parallel") as excinfo:
        compute_scan_alignment(source_obj=source, reference_obj=_scan(AXIAL))

    assert "0.002 mm along the reference slice axis" in str(excinfo.value)
    assert "anatomy_align=True" in str(excinfo.value)


def test_tilt_within_the_tolerance_is_aligned_as_parallel():
    """A tilt of a hundredth of the tolerance is accepted as parallel planes."""
    source = _scan(_tilted_about_read(POSITION_ATOL_MM / 100))

    result = compute_scan_alignment(source_obj=source, reference_obj=_scan(AXIAL))

    assert result.transpose_order == (0, 1, 2)
    np.testing.assert_allclose(
        result.extent_mm,
        [-16.0, 16.0, -20.0, 20.0, -1.0, 1.0],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )


def test_crossing_planes_report_the_angle_between_their_normals():
    """An axial source on a coronal reference crosses it at 90 degrees."""
    with pytest.raises(ValueError, match="normals differ by 90.00 degrees"):
        compute_scan_alignment(source_obj=_scan(AXIAL), reference_obj=_scan(CORONAL))


@pytest.mark.parametrize(
    ("role", "attribute"),
    [
        ("source", "acquisition_matrix"),
        ("source", "rps_grid_mm"),
        ("reference", "rps_extent_mm"),
    ],
)
def test_unknown_scan_geometry_is_named(role, attribute):
    """A geometry property that evaluates to None names the scan and property."""
    scans = {"source": _scan(AXIAL), "reference": _scan(AXIAL)}
    setattr(scans[role], attribute, None)

    with pytest.raises(ValueError, match=f"{attribute} of the {role} scan is None"):
        compute_scan_alignment(
            source_obj=scans["source"], reference_obj=scans["reference"]
        )


# Read and phase rows of the eight axial sources whose read and phase are
# swapped or reversed with respect to the reference, with the aligned read and
# phase extent: the reference read and phase positions of the source read and
# phase limits.  The slice extent stays [-1, 1].
SIGNED_PERMUTATIONS = [
    ([1, 0, 0], [0, 1, 0], (0, 1, 2), [-16.0, 16.0, -20.0, 20.0]),
    ([0, 1, 0], [-1, 0, 0], (1, 0, 2), [20.0, -20.0, -16.0, 16.0]),
    ([-1, 0, 0], [0, -1, 0], (0, 1, 2), [16.0, -16.0, 20.0, -20.0]),
    ([0, -1, 0], [1, 0, 0], (1, 0, 2), [-20.0, 20.0, 16.0, -16.0]),
    ([1, 0, 0], [0, -1, 0], (0, 1, 2), [-16.0, 16.0, 20.0, -20.0]),
    ([0, 1, 0], [1, 0, 0], (1, 0, 2), [-20.0, 20.0, -16.0, 16.0]),
    ([-1, 0, 0], [0, 1, 0], (0, 1, 2), [16.0, -16.0, -20.0, 20.0]),
    ([0, -1, 0], [-1, 0, 0], (1, 0, 2), [20.0, -20.0, 16.0, -16.0]),
]


@pytest.mark.parametrize(
    ("read", "phase", "order", "in_plane_extent_mm"),
    SIGNED_PERMUTATIONS,
    ids=[
        "rotation-0",
        "rotation-90",
        "rotation-180",
        "rotation-270",
        "reflection-0",
        "reflection-90",
        "reflection-180",
        "reflection-270",
    ],
)
def test_swapped_and_reversed_read_and_phase_are_aligned(
    read, phase, order, in_plane_extent_mm
):
    """Rotations by multiples of 90 degrees and reflections keep their extent."""
    source = _scan([read, phase, [0, 0, 1]])

    result = compute_scan_alignment(source_obj=source, reference_obj=_scan(AXIAL))

    assert result.transpose_order == order
    np.testing.assert_allclose(
        result.extent_mm,
        [*in_plane_extent_mm, -1.0, 1.0],
        rtol=0,
        atol=POSITION_ATOL_MM,
    )
    np.testing.assert_allclose(
        result.slice_positions_mm, [-2 / 3, 0.0, 2 / 3], rtol=0, atol=POSITION_ATOL_MM
    )


@pytest.mark.parametrize(
    ("angle_deg", "reflected", "order"),
    [
        (IN_PLANE_ATOL_DEG / 100, False, (0, 1, 2)),
        (-IN_PLANE_ATOL_DEG / 100, True, (0, 1, 2)),
        (90.0 - IN_PLANE_ATOL_DEG / 100, False, (1, 0, 2)),
        (180.0 + IN_PLANE_ATOL_DEG / 100, True, (0, 1, 2)),
    ],
)
def test_in_plane_rotation_within_the_tolerance_is_aligned(angle_deg, reflected, order):
    """A hundredth of the tolerance from a multiple of 90 degrees is rounding."""
    source = _scan(_rotated_in_plane(angle_deg, reflected))
    exact = _scan(_rotated_in_plane(90.0 * round(angle_deg / 90.0), reflected))

    result = compute_scan_alignment(source_obj=source, reference_obj=_scan(AXIAL))
    expected = compute_scan_alignment(source_obj=exact, reference_obj=_scan(AXIAL))

    assert result.transpose_order == order == expected.transpose_order
    np.testing.assert_allclose(
        result.extent_mm, expected.extent_mm, rtol=0, atol=POSITION_ATOL_MM
    )


@pytest.mark.parametrize("reflected", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize(
    ("angle_deg", "deviation_deg"),
    [
        (2 * IN_PLANE_ATOL_DEG, 2 * IN_PLANE_ATOL_DEG),
        (1.0, 1.0),
        (25.8, 25.8),
        (25.9, 25.9),
        (45.0, 45.0),
        (64.1, 25.9),
        (64.2, 25.8),
        (89.0, 1.0),
        (90.0 - 2 * IN_PLANE_ATOL_DEG, 2 * IN_PLANE_ATOL_DEG),
        (135.0, 45.0),
    ],
    ids=[
        "2x-tolerance",
        "1",
        "25.8",
        "25.9",
        "45",
        "64.1",
        "64.2",
        "89",
        "90-2x-tolerance",
        "135",
    ],
)
def test_in_plane_rotation_beyond_the_tolerance_is_rejected(
    angle_deg, deviation_deg, reflected
):
    """The 2D overlay cannot draw a source rotated within the slice plane."""
    source = _scan(_rotated_in_plane(angle_deg, reflected))

    with pytest.raises(
        ValueError, match=f"rotated by {deviation_deg:.4f} degrees"
    ) as excinfo:
        compute_scan_alignment(source_obj=source, reference_obj=_scan(AXIAL))

    message = str(excinfo.value)
    assert "within the parallel slice planes" in message
    assert "anatomy_align=True" in message
    spread = re.search(r"spans ([0-9.]+) mm along the reference read or phase", message)
    assert spread is not None, message
    # The message rounds the spread of the 40 mm phase edge to 1e-3 mm.
    assert float(spread.group(1)) == pytest.approx(
        PHASE_EXTENT_MM * np.sin(np.radians(deviation_deg)), abs=POSITION_ATOL_MM
    )
