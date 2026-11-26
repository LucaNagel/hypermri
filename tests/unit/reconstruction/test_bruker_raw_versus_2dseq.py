"""Raw against ``2dseq`` reconstruction for every ParaVision orientation.

Each test scan is a synthetic two-dimensional ParaVision FLASH scan:

* Geometry: the ParaVision parameters of one zero-offset, single-package
  series per patient position, slice orientation and read orientation from
  the ParaVision reference fixture ``tests/reference/paravision_water_phantom_geometry.json``,
  with the matrix reduced to 16 x 16 so that a swapped read and phase axis
  keeps the image shape.
* Pixel data: the asymmetric phantom of ``_asymmetric_phantom``, not scanner
  data.  Its intensity ramp, an off-centre block that moves with the slice and
  a bar that grows with the slice make every transposition, mirror and slice
  reversal visible.
* ``fid``: the centred two-dimensional DFT of the phantom over read and
  phase, stored as ``GO_32BIT_SGN_INT`` in the ``Standard_KBlock_Format``.
  Phase steps follow ``PVM_EncSteps1`` and slices follow
  ``PVM_ObjOrderList``.  The phase lines of each slice carry the inverse of
  the ramp with which the raw reconstruction applies the phase offset of its
  slice package (``_write_fid``); the sign is the one of the real localizers
  in ``tests/integration/test_paravision_raw_versus_2dseq.py``.
* ``2dseq``: the phantom as 16-bit integers with a slope of 1/32.  With
  ``RECO_transposition`` 0 the read axis is the fastest axis of a frame, with
  1 the phase axis, as in the DICOM exports of the ParaVision reference series.  Each frame
  follows its own ``RECO_transposition`` entry.

This forward model was checked against the 51 real series of the ParaVision reference study:
for all of them the raw and ``2dseq`` magnitudes correlate to 1.000000 at
zero shift, and for the two multi-package localizers also slice package by
slice package.  The synthetic scans of these localizers,
``MULTI_PACKAGE_SERIES``, test their coordinates and the ``2dseq`` and raw
magnitudes of each slice package.

Raw and ``2dseq`` magnitudes are compared with the normalised circular
cross-correlation of ``magnitude_cross_correlation``.  The threshold of
0.99 has a wide margin on both sides: integer storage keeps the correlation
within 1e-10 of 1, while a transposition, a mirrored axis or a one-voxel
shift of the phantom gives at most 0.76 and moves the correlation peak away
from zero shift (``test_phantom_detects_swapped_mirrored_and_shifted_axes``).

Cases that fail because of a known open problem are marked
``xfail(strict=True, raises=AssertionError)`` with a reason describing it.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from hypermri.exceptions import ReconstructionError
from hypermri.loading.scans import load_scan
from hypermri.reconstruction.strategies.bruker.seq2d import Brukerseq2dRecoStrategy
from hypermri.utils.coordinates.image_shifts import get_orientation_flip_multipliers
from hypermri.vendors.bruker import BrukerHandler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "reference" / "paravision_water_phantom_geometry.json"
)
FIXTURE_SERIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["series"]

# Square so that a swapped read and phase axis keeps the shape; even so that
# the centred DFT has its centre on a voxel.
MATRIX_SIZE = 16
# A power of two, so that the stored 16-bit values are exact.
INTENSITY_SLOPE = 0.03125
# Largest fid sample; well inside the signed 32-bit range.
FID_FULL_SCALE = 2**21
KBLOCK_BYTES = 1024

CORRELATION_THRESHOLD = 0.99
ALIGNED = (0, 0, 0)
SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
SPATIAL_COORDS = (*SPATIAL_DIMS, "x", "y", "z")

# One zero-offset series per patient position, slice orientation and read
# orientation of the ParaVision reference fixture.
ORIENTATION_SERIES = {
    ("Head_Prone", "axial", "L_R"): "head_prone/10",
    ("Head_Prone", "axial", "A_P"): "head_prone/14",
    ("Head_Prone", "sagittal", "A_P"): "head_prone/18",
    ("Head_Prone", "sagittal", "H_F"): "head_prone/22",
    ("Head_Prone", "coronal", "H_F"): "head_prone/26",
    ("Head_Prone", "coronal", "L_R"): "head_prone/30",
    ("Head_Supine", "axial", "L_R"): "head_supine/2",
    ("Head_Supine", "axial", "A_P"): "head_supine/6",
    ("Head_Supine", "sagittal", "A_P"): "head_supine/10",
    ("Head_Supine", "sagittal", "H_F"): "head_supine/15",
    ("Head_Supine", "coronal", "H_F"): "head_supine/19",
    ("Head_Supine", "coronal", "L_R"): "head_supine/23",
}

# The three-plane localizers of the ParaVision reference fixture: three packages of seven
# slices with their own orientation and offsets.
MULTI_PACKAGE_SERIES = ("head_prone/4", "head_supine/1")

# No feet-first scanner data exists: the Head_Supine series above with a
# feet-first ACQ_patient_pos.  ACQ_grad_matrix and visu_pars keep the
# head-first geometry, which neither reconstruction reads for the axis order.
FEET_FIRST_SERIES = {
    (feet_position, slice_orientation, read_orientation): name
    for feet_position in ("Foot_Prone", "Foot_Supine")
    for (patient_pos, slice_orientation, read_orientation), name in (
        ORIENTATION_SERIES.items()
    )
    if patient_pos == "Head_Supine"
}

COORDINATE_SET_REASON = (
    "Raw has an index coordinate for acquisition_idx that the synthetic "
    "2dseq lacks, although the 2dseq of a real scan has one"
)


def _orientation_key(series: dict[str, Any]) -> tuple[str, str, str]:
    paravision = series["paravision"]
    return (
        paravision["acqp"]["ACQ_patient_pos"],
        paravision["method"]["PVM_SPackArrSliceOrient"],
        paravision["method"]["PVM_SPackArrReadOrient"],
    )


def _series_params(xfail_reasons: dict[str, str]):
    params = []
    for name in ORIENTATION_SERIES.values():
        reason = xfail_reasons.get(name)
        marks = (
            [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)]
            if reason
            else []
        )
        params.append(pytest.param(name, marks=marks, id=name.replace("/", "-")))
    return params


# --------------------------------------------------------------------------
# Synthetic phantom and ParaVision scan
# --------------------------------------------------------------------------


def _asymmetric_phantom(n_slices: int) -> np.ndarray:
    """Return a magnitude phantom with shape ``(read, phase, slice)``.

    A ramp from 100 to 160 along read, a 3 x 5 block that moves by one read
    voxel per slice and a bar along phase that grows by one voxel per slice.
    No transposition, mirror or slice reversal maps the phantom onto itself.
    """
    size = MATRIX_SIZE
    read_ramp = 100.0 + 60.0 * np.arange(size) / (size - 1)
    phantom = np.broadcast_to(read_ramp[:, None, None], (size, size, n_slices)).copy()
    for slice_index in range(n_slices):
        phantom[2 + slice_index : 5 + slice_index, 9:14, slice_index] += 400.0
        phantom[11:14, 2 : 3 + slice_index, slice_index] += 250.0
    return phantom


def _phase_shift_vox(method: dict[str, Any]) -> np.ndarray:
    """Return the raw field of view shift along phase of every slice in voxels.

    Each slice takes ``PVM_SPackArrPhase1Offset / (PVM_Fov[1] / MATRIX_SIZE)``
    of its package: the phase offset with the multiplier +1 of the
    Head_Prone and Head_Supine orientations of the fixture.
    """
    n_slices = np.atleast_1d(method["PVM_SPackArrNSlices"])
    phase_offsets_mm = np.atleast_1d(method["PVM_SPackArrPhase1Offset"])
    phase_voxel_mm = method["PVM_Fov"][1] / MATRIX_SIZE
    return np.repeat(phase_offsets_mm / phase_voxel_mm, n_slices)


def _write_fid(path: Path, phantom: np.ndarray, method: dict[str, Any]) -> None:
    """Write the centred k-space of the phantom in ParaVision acquisition order.

    The phase lines of every slice are multiplied by
    ``exp(2j * pi * k * s / MATRIX_SIZE)`` with the centred index ``k`` and
    the shift ``s`` of ``_phase_shift_vox``: the inverse of the ramp with
    which the raw reconstruction applies its field of view shift.
    """
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(phantom, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    k = np.arange(MATRIX_SIZE) - MATRIX_SIZE // 2
    kspace = kspace * np.exp(
        2j * np.pi * k[None, :, None] * _phase_shift_vox(method) / MATRIX_SIZE
    )
    phase_steps = np.asarray(method["PVM_EncSteps1"]) + MATRIX_SIZE // 2
    acquired = kspace[:, phase_steps][:, :, method["PVM_ObjOrderList"]]
    # One read line per phase step (outer loop) and slice (inner loop).
    lines = np.transpose(acquired, (1, 2, 0)).reshape(-1, MATRIX_SIZE)
    samples = np.stack([lines.real, lines.imag], axis=-1).reshape(len(lines), -1)
    samples *= FID_FULL_SCALE / np.abs(samples).max()
    # Standard_KBlock_Format pads every read line to whole 1024-byte blocks.
    padded = np.zeros((len(lines), KBLOCK_BYTES // 4), dtype="<i4")
    padded[:, : samples.shape[1]] = np.rint(samples)
    path.write_bytes(padded.tobytes())


def _write_2dseq(
    path: Path, phantom: np.ndarray, *, transposed: bool | list[bool]
) -> None:
    """Write the phantom as 16-bit frames in ParaVision storage order.

    ``transposed`` holds one entry per frame or one for all frames.
    """
    stored = np.rint(phantom / INTENSITY_SLOPE).astype("<i2")
    n_frames = stored.shape[2]
    frame_transposed = np.broadcast_to(np.asarray(transposed, dtype=bool), n_frames)
    # A frame is stored row by row; its fastest axis is phase when transposed
    # and read otherwise.
    frames = [
        stored[:, :, k] if frame_transposed[k] else stored[:, :, k].T
        for k in range(n_frames)
    ]
    path.write_bytes(b"".join(np.ascontiguousarray(f).tobytes() for f in frames))


def _write_synthetic_scan(
    scan_dir: Path,
    series: dict[str, Any],
    write_parameters,
    *,
    transpose_2dseq: bool | None = None,
) -> np.ndarray:
    """Write a synthetic ParaVision FLASH scan and return its phantom.

    ``transpose_2dseq`` overrides the storage order of all ``2dseq`` frames;
    by default each frame follows its ``RECO_transposition`` entry.  A list
    of slice counts in ``PVM_SPackArrNSlices`` gives a scan with several
    slice packages, whose slices are numbered across the packages.
    """
    paravision = series["paravision"]
    size = MATRIX_SIZE
    n_slices = int(np.sum(paravision["method"]["PVM_SPackArrNSlices"]))
    method = {
        **paravision["method"],
        "Method": "<Bruker:FLASH>",
        "PVM_Matrix": [size, size],
        "PVM_EncMatrix": [size, size],
        "PVM_EncSteps1": list(range(-size // 2, size // 2)),
        "PVM_EncNReceivers": 1,
        "PVM_EncSpectroscopy": "No",
    }
    acqp = {
        **paravision["acqp"],
        "ACQ_experiment_mode": "SingleExperiment",
        "ACQ_size": [2 * size, size],
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "BYTORDA": "little",
        "AQ_mod": "qdig",
        "GO_block_size": "Standard_KBlock_Format",
        "NI": n_slices,
        "NR": 1,
    }
    # brukerapi reads the frame layout and scaling of 2dseq from these fields.
    visu_pars = {
        **paravision["visu_pars"],
        "VisuCoreSize": [size, size],
        "VisuCreatorVersion": "7.0.0",
        "VisuCoreWordType": "_16BIT_SGN_INT",
        "VisuCoreByteOrder": "littleEndian",
        "VisuCoreFrameCount": n_slices,
        "VisuCoreDimDesc": ["spatial", "spatial"],
        "VisuFGOrderDescDim": 1,
        "VisuFGOrderDesc": [f"({n_slices}, <FG_SLICE>, <>, 0, 0)"],
        "VisuCoreDataSlope": [INTENSITY_SLOPE] * n_slices,
        "VisuCoreDataOffs": [0] * n_slices,
        "VisuCoreFrameType": "MAGNITUDE_IMAGE",
    }
    write_parameters(scan_dir / "method", method)
    write_parameters(scan_dir / "acqp", acqp)
    write_parameters(scan_dir / "visu_pars", visu_pars)
    write_parameters(scan_dir / "pdata" / "1" / "visu_pars", visu_pars)
    write_parameters(scan_dir / "pdata" / "1" / "reco", paravision["reco"])

    phantom = _asymmetric_phantom(n_slices)
    _write_fid(scan_dir / "fid", phantom, method)
    if transpose_2dseq is None:
        transpose_2dseq = [bool(t) for t in paravision["reco"]["RECO_transposition"]]
    _write_2dseq(
        scan_dir / "pdata" / "1" / "2dseq", phantom, transposed=transpose_2dseq
    )
    return phantom


def _load_reconstructions(scan_dir: Path):
    experiment = load_scan(scan_dir, mode="full", log_mode="critical")
    reconstructions = experiment.reconstructions
    return reconstructions["seq2d"], reconstructions["raw"]


@pytest.fixture(scope="module")
def reconstructions(tmp_path_factory, write_paravision_parameters):
    """Return ``(phantom, seq2d, raw)`` for every orientation series."""
    root = tmp_path_factory.mktemp("raw_versus_2dseq")
    loaded = {}
    for name in ORIENTATION_SERIES.values():
        scan_dir = root / name
        phantom = _write_synthetic_scan(
            scan_dir, FIXTURE_SERIES[name], write_paravision_parameters
        )
        loaded[name] = (phantom, *_load_reconstructions(scan_dir))
    return loaded


@pytest.fixture(scope="module")
def multi_package_reconstructions(tmp_path_factory, write_paravision_parameters):
    """Return ``(phantom, seq2d, raw)`` for every multi-package series."""
    root = tmp_path_factory.mktemp("multi_package_raw_versus_2dseq")
    loaded = {}
    for name in MULTI_PACKAGE_SERIES:
        scan_dir = root / name
        phantom = _write_synthetic_scan(
            scan_dir, FIXTURE_SERIES[name], write_paravision_parameters
        )
        loaded[name] = (phantom, *_load_reconstructions(scan_dir))
    return loaded


@pytest.fixture(scope="module")
def feet_first_reconstructions(tmp_path_factory, write_paravision_parameters):
    """Return ``(seq2d, raw)`` for every series of ``FEET_FIRST_SERIES``."""
    root = tmp_path_factory.mktemp("feet_first_raw_versus_2dseq")
    loaded = {}
    for key, name in FEET_FIRST_SERIES.items():
        series = copy.deepcopy(FIXTURE_SERIES[name])
        series["paravision"]["acqp"]["ACQ_patient_pos"] = key[0]
        scan_dir = root / key[0] / name
        _write_synthetic_scan(scan_dir, series, write_paravision_parameters)
        loaded[key] = _load_reconstructions(scan_dir)
    return loaded


# --------------------------------------------------------------------------
# Test design
# --------------------------------------------------------------------------


def test_orientation_series_cover_every_fixture_orientation():
    """One zero-offset, single-package series per orientation of the fixture."""
    single_package = {
        _orientation_key(series)
        for series in FIXTURE_SERIES.values()
        if not isinstance(series["paravision"]["method"]["PVM_SPackArrNSlices"], list)
    }
    assert set(ORIENTATION_SERIES) == single_package
    for key, name in ORIENTATION_SERIES.items():
        method = FIXTURE_SERIES[name]["paravision"]["method"]
        assert _orientation_key(FIXTURE_SERIES[name]) == key
        assert method["PVM_SpatDimEnum"] == "<2D>", name
        offsets = (
            method["PVM_SPackArrReadOffset"],
            method["PVM_SPackArrPhase1Offset"],
            method["PVM_SPackArrSliceOffset"],
        )
        assert offsets == (0, 0, 0), name


def test_multi_package_series_are_the_fixture_localizers():
    """Each multi-package series has three packages in three orientations."""
    multi_package = {
        name
        for name, series in FIXTURE_SERIES.items()
        if isinstance(series["paravision"]["method"]["PVM_SPackArrNSlices"], list)
    }
    assert multi_package == set(MULTI_PACKAGE_SERIES)
    for name in MULTI_PACKAGE_SERIES:
        method = FIXTURE_SERIES[name]["paravision"]["method"]
        assert method["PVM_SpatDimEnum"] == "<2D>", name
        assert method["PVM_SPackArrNSlices"] == [7, 7, 7], name
        assert method["PVM_SPackArrSliceOrient"] == ["axial", "sagittal", "coronal"]


def test_phantom_detects_swapped_mirrored_and_shifted_axes(
    magnitude_cross_correlation,
):
    """Every axis error of the phantom fails both parts of the comparison criterion.

    Measured zero-shift correlations: 0.16 for swapped read and phase, at
    most 0.34 for mirrors and the slice reversal, 0.63, 0.76 and 0.54 for
    one-voxel shifts along read, phase and slice.
    """
    phantom = _asymmetric_phantom(5)
    zero_shift, peak_shift = magnitude_cross_correlation(phantom, phantom)
    assert zero_shift == pytest.approx(1.0)
    assert peak_shift == ALIGNED
    axis_errors = {
        "read and phase swapped": phantom.transpose(1, 0, 2),
        "read mirrored": phantom[::-1],
        "phase mirrored": phantom[:, ::-1],
        "slices reversed": phantom[:, :, ::-1],
        "read and phase mirrored": phantom[::-1, ::-1],
        "read shifted by one voxel": np.roll(phantom, 1, axis=0),
        "phase shifted by one voxel": np.roll(phantom, 1, axis=1),
        "slices shifted by one": np.roll(phantom, 1, axis=2),
    }
    for error, image in axis_errors.items():
        zero_shift, peak_shift = magnitude_cross_correlation(image, phantom)
        assert zero_shift < 0.76, (error, zero_shift)
        assert peak_shift != ALIGNED, error


# --------------------------------------------------------------------------
# Raw against 2dseq
# --------------------------------------------------------------------------


@pytest.mark.parametrize("empty_file", [False, True], ids=["missing", "empty"])
def test_full_load_without_seq2d_preserves_raw_reconstruction(
    tmp_path, write_paravision_parameters, empty_file
):
    """Absent scanner images do not change synthetic FLASH raw results.

    Both scans contain the same integer k-space of the synthetic asymmetric
    phantom. Float32 output agreement uses a 1e-7 relative tolerance; acquired
    bytes and labelled dimensions/coordinates must remain unchanged.
    """
    series = FIXTURE_SERIES["head_prone/26"]
    complete = tmp_path / "1"
    raw_only = tmp_path / "2"
    for scan_dir in (complete, raw_only):
        _write_synthetic_scan(scan_dir, series, write_paravision_parameters)
    pdata = raw_only / "pdata" / "1"
    (pdata / "visu_pars").unlink()
    if empty_file:
        (pdata / "2dseq").write_bytes(b"")
    else:
        (pdata / "2dseq").unlink()
    acquired_bytes = (raw_only / "fid").read_bytes()
    reference = load_scan(complete, mode="full", log_mode="critical")

    result = load_scan(raw_only, mode="full", log_mode="critical")

    assert "seq2d" not in result.reconstructions
    raw = result.reconstructions["raw"]
    expected = reference.reconstructions["raw"]
    assert raw.dims == expected.dims
    assert set(raw.coords) == set(expected.coords)
    for name in raw.coords:
        xr.testing.assert_identical(raw.coords[name], expected.coords[name])
    assert raw.dtype == expected.dtype
    np.testing.assert_allclose(raw.data, expected.data, rtol=1e-7, atol=0)
    assert (raw_only / "fid").read_bytes() == acquired_bytes


@pytest.mark.parametrize("name", _series_params({}))
def test_2dseq_reproduces_the_phantom(
    reconstructions, magnitude_cross_correlation, name
):
    """The ``2dseq`` reconstruction keeps the stored read, phase and slice axes."""
    phantom, seq2d, _ = reconstructions[name]
    zero_shift, peak_shift = magnitude_cross_correlation(seq2d, phantom)
    assert peak_shift == ALIGNED, name
    assert zero_shift > CORRELATION_THRESHOLD, name


@pytest.mark.parametrize("name", _series_params({}))
def test_2dseq_has_the_raw_spatial_order(reconstructions, name):
    """Regression test: ``2dseq`` is ordered read, phase, slice.

    Previously the strict path ordered the coronal series
    ``(read_idx, slice_idx, phase_idx)``, so ``seq2d[:, :, k]`` selected a
    phase row instead of slice ``k``.  The values lie at the positions of the
    phantom, which is ordered (read, phase, slice), within half the 1/32
    storage step of the 16-bit ``2dseq``.
    """
    phantom, seq2d, raw = reconstructions[name]
    assert seq2d.dims[:3] == SPATIAL_DIMS, name
    assert tuple(dim for dim in raw.dims if dim in SPATIAL_DIMS) == SPATIAL_DIMS
    np.testing.assert_allclose(
        np.squeeze(seq2d.to_numpy()), phantom, rtol=0, atol=INTENSITY_SLOPE / 2
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("head_prone/26", id="transposed-coronal"),
        pytest.param("head_prone/10", id="axial"),
    ],
)
def test_2dseq_fallback_has_the_raw_spatial_order(
    tmp_path, write_paravision_parameters, magnitude_cross_correlation, name
):
    """Regression test: the ``2dseq`` fallback is ordered read, phase, slice.

    The fallback labels the axes by size.  ``head_prone/26`` has
    ``RECO_transposition`` 1; previously the fallback labelled its stored axes,
    which start with ``phase_idx``, and kept that order.  The test passes the
    array as the strategy does, with the swap undone.  The 16 x 16 matrix
    makes the read and phase labels depend on the search order of the
    fallback, not on the sizes, so the correlation with the phantom checks
    the labels, and the comparison within half the storage step checks the
    positions.
    """
    phantom = _write_synthetic_scan(
        tmp_path / name, FIXTURE_SERIES[name], write_paravision_parameters
    )
    handler = BrukerHandler(tmp_path / name)

    seq2d = handler.undo_reco_transposition(handler.load_2dseq_file()).squeeze()
    fallback = Brukerseq2dRecoStrategy()._fallback_seq2d_scanner_data(
        handler, seq2d, {}, append_parameters=False
    )

    assert fallback.dims == SPATIAL_DIMS
    zero_shift, peak_shift = magnitude_cross_correlation(fallback, phantom)
    assert peak_shift == ALIGNED, (zero_shift, peak_shift)
    assert zero_shift > CORRELATION_THRESHOLD, zero_shift
    np.testing.assert_allclose(
        fallback.to_numpy(), phantom, rtol=0, atol=INTENSITY_SLOPE / 2
    )


@pytest.mark.parametrize("name", _series_params({}))
def test_raw_magnitude_matches_2dseq(
    reconstructions, magnitude_cross_correlation, name
):
    """Raw and ``2dseq`` magnitudes correlate above 0.99 at zero shift."""
    _, seq2d, raw = reconstructions[name]
    zero_shift, peak_shift = magnitude_cross_correlation(raw, seq2d)
    assert peak_shift == ALIGNED, (name, zero_shift, peak_shift)
    assert zero_shift > CORRELATION_THRESHOLD, (name, zero_shift)


@pytest.mark.parametrize("spatial_dimension", ["2D", "3D"])
def test_head_supine_axial_a_p_keeps_the_slice_order(spatial_dimension):
    """Regression test: Head_Supine axial A_P keeps its slice order.

    An earlier table reversed the slice axis of this orientation for 2D and
    3D encoding, and the raw and ``2dseq`` magnitudes of the synthetic
    ``head_supine/6`` scan correlated at 0.335 with the peak two slices away.
    The 3D entry has no scanner reference; the owner decided to drop its
    slice reversal as well.
    """
    scan = SimpleNamespace(
        patient_pos="Head_Supine",
        slice_orientation="axial",
        read_orientation="A_P",
        spatial_dimension=spatial_dimension,
    )
    np.testing.assert_array_equal(get_orientation_flip_multipliers(scan), [1, 1, 1])


@pytest.mark.parametrize("key", FEET_FIRST_SERIES, ids="-".join)
def test_feet_first_raw_magnitude_matches_2dseq(
    feet_first_reconstructions, magnitude_cross_correlation, key
):
    """Regression test: feet-first scans keep the ``2dseq`` voxel order.

     Previously the feet-first scans kept the FFT axis order: the raw and
     ``2dseq`` magnitudes correlated at 0.031 with the peak away from zero
     shift.  The read and phase order no longer depends on the
     flip table.  ParaVision writes the feet-first positions as
     ``Foot_Prone`` and ``Foot_Supine``.  The synthetic forward model
     does not depend on the patient position, so this test checks that
     feet-first scans follow the verified head-first scans, not the scanner
    .
    """
    seq2d, raw = feet_first_reconstructions[key]
    zero_shift, peak_shift = magnitude_cross_correlation(raw, seq2d)
    assert peak_shift == ALIGNED, (key, zero_shift, peak_shift)
    assert zero_shift > CORRELATION_THRESHOLD, (key, zero_shift)


def test_lazy_raw_reconstruction_equals_the_full_load(
    tmp_path, write_paravision_parameters
):
    """Regression test: ``reconstruct("raw")`` of a lazily loaded scan.

    A lazy load reads no ``fid``.  Previously the raw strategy found no fid
    data and failed; it now reads the ``fid`` file on demand, before the
    precision conversion, so the result equals the ``raw`` reconstruction
    of a full load exactly, values, dtype, coordinates and attributes.
    """
    name = "head_prone/10"
    scan_dir = tmp_path / name
    _write_synthetic_scan(scan_dir, FIXTURE_SERIES[name], write_paravision_parameters)
    full_raw = load_scan(scan_dir, mode="full", log_mode="critical").reconstructions[
        "raw"
    ]

    experiment = load_scan(scan_dir, mode="lazy", log_mode="critical")
    assert getattr(experiment.handler, "fid", None) is None
    lazy_raw = experiment.reconstruct("raw")

    xr.testing.assert_identical(lazy_raw, full_raw)
    assert lazy_raw.dtype == full_raw.dtype


def test_lazy_raw_reconstruction_without_fid_names_the_missing_file(
    tmp_path, write_paravision_parameters
):
    """A lazily loaded scan without ``fid`` or ``ser`` file fails clearly."""
    name = "head_prone/10"
    scan_dir = tmp_path / name
    _write_synthetic_scan(scan_dir, FIXTURE_SERIES[name], write_paravision_parameters)
    (scan_dir / "fid").unlink()

    experiment = load_scan(scan_dir, mode="lazy", log_mode="critical")

    with pytest.raises(ReconstructionError, match="'fid' or 'ser' file"):
        experiment.reconstruct("raw")


def _count_2dseq_reads(monkeypatch) -> list[Path]:
    """Record the path of every ``2dseq`` that brukerapi opens."""
    from hypermri.vendors import bruker

    reads: list[Path] = []
    original = bruker.Dataset

    def counting_dataset(path, *args, **kwargs):
        reads.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(bruker, "Dataset", counting_dataset)
    return reads


# head_prone/10 stores every frame unswapped, head_prone/14 every frame
# swapped (a view of the read array), and the localizer head_prone/4 mixes
# both (a new array).
E3_SERIES = ("head_prone/10", "head_prone/14", "head_prone/4")


@pytest.mark.parametrize("name", E3_SERIES, ids=lambda n: n.replace("/", "-"))
def test_full_load_reads_2dseq_once(
    tmp_path, write_paravision_parameters, monkeypatch, name
):
    """Regression test: the handler and the seq2d strategy share a read."""
    scan_dir = tmp_path / name
    _write_synthetic_scan(scan_dir, FIXTURE_SERIES[name], write_paravision_parameters)
    reads = _count_2dseq_reads(monkeypatch)

    experiment = load_scan(scan_dir, mode="full", log_mode="critical")

    assert "seq2d" in experiment.reconstructions
    assert reads == [scan_dir / "pdata" / "1" / "2dseq"]


@pytest.mark.parametrize("name", E3_SERIES, ids=lambda n: n.replace("/", "-"))
def test_full_load_2dseq_equals_a_reconstruction_that_reads_the_file(
    tmp_path, write_paravision_parameters, name
):
    """The shared read does not change the values, dtype or labels.

    A lazy load reads ``2dseq`` in the seq2d strategy itself.  The full load
    result is a separate array: it shares no memory with the handler's
    ``seq2d`` or ``scanner_data``.
    """
    scan_dir = tmp_path / name
    phantom = _write_synthetic_scan(
        scan_dir, FIXTURE_SERIES[name], write_paravision_parameters
    )
    full = load_scan(scan_dir, mode="full", log_mode="critical")
    lazy_seq2d = load_scan(scan_dir, mode="lazy", log_mode="critical").reconstruct(
        "seq2d"
    )

    full_seq2d = full.reconstructions["seq2d"]
    xr.testing.assert_identical(full_seq2d, lazy_seq2d)
    assert full_seq2d.dtype == np.float32
    np.testing.assert_array_equal(
        full_seq2d.transpose(*SPATIAL_DIMS, ...).values.reshape(phantom.shape),
        np.rint(phantom / INTENSITY_SLOPE) * INTENSITY_SLOPE,
    )
    handler = full.handler
    assert not np.shares_memory(full_seq2d.values, handler.seq2d)
    assert not np.shares_memory(full_seq2d.values, handler.scanner_data.values)


def test_later_2dseq_reconstruction_reads_the_file_again(
    tmp_path, write_paravision_parameters, monkeypatch
):
    """Only the first reader takes the array of the full load.

    A later explicit reconstruction reads the file, so a change to the
    handler's arrays after loading does not reach it.
    """
    name = "head_prone/14"
    scan_dir = tmp_path / name
    _write_synthetic_scan(scan_dir, FIXTURE_SERIES[name], write_paravision_parameters)
    experiment = load_scan(scan_dir, mode="full", log_mode="critical")
    first = experiment.reconstructions["seq2d"].copy(deep=True)
    experiment.handler.seq2d[...] = 0
    reads = _count_2dseq_reads(monkeypatch)

    again = experiment.handler.load_reconstruction(recon_num=1, set_as_primary=False)

    assert reads == [scan_dir / "pdata" / "1" / "2dseq"]
    xr.testing.assert_equal(again, first)


def test_patient_position_outside_the_flip_table_fails_the_raw_reconstruction(
    tmp_path, write_paravision_parameters
):
    """Regression test: the flip table rejects ``Head_Left``.

    ``Head_Left``, a lateral position, is not in the flip table.  The
    the raw voxel order does not use the table, so previously the raw and
    ``2dseq`` magnitudes of this synthetic scan agreed, but the field of
    view shift took the sign of its read and phase offsets from the
    multipliers (1, 1, 1).  A full load, which requests the raw
    reconstruction, now fails and names the patient position, while a
    minimal load still provides ``seq2d``.
    """
    name = "head_supine/2"
    series = copy.deepcopy(FIXTURE_SERIES[name])
    series["paravision"]["acqp"]["ACQ_patient_pos"] = "Head_Left"
    _write_synthetic_scan(tmp_path / name, series, write_paravision_parameters)

    with pytest.raises(ReconstructionError, match="unknown patient_pos 'Head_Left'"):
        load_scan(tmp_path / name, mode="full", log_mode="critical")
    experiment = load_scan(tmp_path / name, mode="minimal", log_mode="critical")

    assert experiment.reconstructions["seq2d"] is not None
    assert "raw" not in experiment.reconstructions


@pytest.mark.parametrize("spatial_dimension", ["2D", "3D"])
@pytest.mark.parametrize("key", FEET_FIRST_SERIES, ids="-".join)
def test_feet_first_multipliers_match_the_head_first_entries(key, spatial_dimension):
    """Feet-first scans share the head-first read, phase and slice entries.

    Every entry is +1: the field of view offsets keep their sign in
    ``determine_FOV_shift`` and the slice order is kept.  The 3D entries
    have no scanner reference.
    """
    patient_pos, slice_orientation, read_orientation = key
    scan = SimpleNamespace(
        patient_pos=patient_pos,
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
        spatial_dimension=spatial_dimension,
    )
    np.testing.assert_array_equal(get_orientation_flip_multipliers(scan), [1, 1, 1])


@pytest.mark.parametrize(
    ("patient_pos", "slice_orientation", "read_orientation"),
    [
        ("Head_Prone", "axial", "H_F"),
        ("Head_Prone", "sagittal", "L_R"),
        ("Head_Supine", "axial", "H_F"),
        ("Head_Supine", "sagittal", "L_R"),
        ("Head_Supine", "coronal", "A_P"),
    ],
)
def test_read_orientation_outside_the_table_is_rejected(
    patient_pos, slice_orientation, read_orientation
):
    """Regression test: an unknown read orientation is reported.

    The table created ``Warning`` objects for these combinations without
    issuing them; after the warning was issued it was hidden at the
    default log level and the multipliers were (1, 1, 1).
    """
    scan = SimpleNamespace(
        patient_pos=patient_pos,
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
        spatial_dimension="2D",
    )
    with pytest.raises(ValueError, match=f"read_orientation '{read_orientation}'"):
        get_orientation_flip_multipliers(scan)


@pytest.mark.parametrize("name", _series_params({}))
def test_raw_and_2dseq_share_spatial_coordinates(reconstructions, name):
    """Index and ``x``, ``y``, ``z`` coordinates agree.

    The raw transform reorders voxels, not coordinates, so each
    coordinate keeps its position in mm.
    """
    _, seq2d, raw = reconstructions[name]
    for coord in SPATIAL_COORDS:
        np.testing.assert_array_equal(
            raw[coord].transpose(*seq2d[coord].dims).to_numpy(),
            seq2d[coord].to_numpy(),
            err_msg=f"{name}: coordinate {coord!r}",
        )


@pytest.mark.parametrize("name", MULTI_PACKAGE_SERIES)
def test_multi_package_scans_have_index_spatial_coordinates(
    multi_package_reconstructions, name
):
    """Regression test: several slice packages have no RPS positions.

    Previously both reconstructions labelled ``read_idx``, ``phase_idx`` and
    ``slice_idx`` with the mm grid of the first package, which is axial,
    for the sagittal and coronal packages as well.  Now the three
    dimensions have index coordinates; the matrix and the read and phase
    voxel size, ``PVM_Fov / PVM_Matrix``, remain attributes, while
    ``extent_mm`` and ``axis_mm`` are left out.
    """
    _, seq2d, raw = multi_package_reconstructions[name]
    fov_mm = FIXTURE_SERIES[name]["paravision"]["method"]["PVM_Fov"]
    sizes = (MATRIX_SIZE, MATRIX_SIZE, 21)
    for reconstruction in (seq2d, raw):
        for dim, size in zip(SPATIAL_DIMS, sizes, strict=True):
            coordinate = reconstruction[dim]
            assert coordinate.dims == (dim,), (name, dim)
            assert coordinate.attrs["unit"] == "index", (name, dim)
            np.testing.assert_array_equal(coordinate, np.arange(size))
            assert coordinate.attrs["matrix"] == size, (name, dim)
            assert "extent_mm" not in coordinate.attrs, (name, dim)
            assert "axis_mm" not in coordinate.attrs, (name, dim)
        for dim, fov in zip(SPATIAL_DIMS[:2], fov_mm, strict=True):
            assert reconstruction[dim].attrs["resolution_mm"] == pytest.approx(
                fov / MATRIX_SIZE
            ), (name, dim)


@pytest.mark.parametrize("name", MULTI_PACKAGE_SERIES)
def test_multi_package_2dseq_reproduces_the_phantom(
    multi_package_reconstructions, magnitude_cross_correlation, name
):
    """Regression test: every slice package keeps read before phase.

    The localizers store the seven frames of the axial package with
    ``RECO_transposition`` 0 and the 14 frames of the other two packages
    with 1.  Previously the ``2dseq`` reconstruction swapped read and phase
    in all 21 frames, so the axial package came out transposed.  Each
    package is compared with the phantom on its own, by correlation and
    within half the 1/32 storage step of the 16-bit ``2dseq``.
    """
    phantom, seq2d, _ = multi_package_reconstructions[name]
    assert seq2d.dims[:3] == SPATIAL_DIMS, name
    values = np.squeeze(seq2d.to_numpy())
    n_slices = FIXTURE_SERIES[name]["paravision"]["method"]["PVM_SPackArrNSlices"]
    stops = np.cumsum(n_slices)
    for package, (start, stop) in enumerate(zip(stops - n_slices, stops, strict=True)):
        zero_shift, peak_shift = magnitude_cross_correlation(
            values[:, :, start:stop], phantom[:, :, start:stop]
        )
        assert peak_shift == ALIGNED, (name, package, zero_shift, peak_shift)
        assert zero_shift > CORRELATION_THRESHOLD, (name, package, zero_shift)
    np.testing.assert_allclose(values, phantom, rtol=0, atol=INTENSITY_SLOPE / 2)


@pytest.mark.parametrize("name", MULTI_PACKAGE_SERIES)
def test_multi_package_raw_magnitude_matches_2dseq(
    multi_package_reconstructions, magnitude_cross_correlation, name
):
    """Regression test: the raw field of view shift follows each package.

    The coronal package of each localizer has a phase offset of -6.34 mm,
    1.585 voxels of the 16 x 16 matrix; the other two packages have none.
    Previously every package was shifted by the offsets of the first, axial
    package, so the raw and ``2dseq`` magnitudes of the coronal package
    correlated at 0.877 with the peak two phase voxels away.
    """
    _, seq2d, raw = multi_package_reconstructions[name]
    n_slices = FIXTURE_SERIES[name]["paravision"]["method"]["PVM_SPackArrNSlices"]
    stops = np.cumsum(n_slices)
    for package, (start, stop) in enumerate(zip(stops - n_slices, stops, strict=True)):
        package_slices = {"slice_idx": slice(start, stop)}
        zero_shift, peak_shift = magnitude_cross_correlation(
            raw.isel(package_slices), seq2d.isel(package_slices)
        )
        assert peak_shift == ALIGNED, (name, package, zero_shift, peak_shift)
        assert zero_shift > CORRELATION_THRESHOLD, (name, package, zero_shift)


@pytest.mark.parametrize("name", list(ORIENTATION_SERIES.values()))
def test_raw_and_2dseq_share_the_dimension_order(reconstructions, name):
    """The raw reconstruction pads and orders its dimensions like ``2dseq``.

    Previously the raw reconstruction started with ``acquisition_idx`` and
    lacked the size-1 dimensions of the ``2dseq`` reconstruction.  The
    synthetic ``2dseq`` has no ``acquisition_idx``; scans with ``2dseq``
    frame groups have one (still open).
    """
    _, seq2d, raw = reconstructions[name]
    assert raw.dims == seq2d.dims


@pytest.mark.parametrize(
    "name",
    _series_params({}),
)
def test_raw_and_2dseq_share_coordinate_names(reconstructions, name):
    """The coordinate names of the raw and the ``2dseq`` reconstruction are identical."""
    _, seq2d, raw = reconstructions[name]
    assert set(raw.coords) == set(seq2d.coords), name


@pytest.mark.parametrize("name", sorted(ORIENTATION_SERIES.values()))
def test_2dseq_keeps_the_index_coordinate_of_inversion_idx(reconstructions, name):
    """The one-dimensional ``inversion_idx`` coordinate of ``2dseq`` is kept.

    The coordinate was dropped by a check that iterated the characters of the
    dimension name.  A scan without inversion times has the single index 0.
    """
    _, seq2d, raw = reconstructions[name]
    assert "inversion_idx" in seq2d.coords
    assert seq2d["inversion_idx"].dims == ("inversion_idx",)
    np.testing.assert_array_equal(
        seq2d["inversion_idx"].values, raw["inversion_idx"].values
    )


def test_correlation_detects_a_2dseq_with_swapped_axes(
    tmp_path, write_paravision_parameters, magnitude_cross_correlation
):
    """A ``2dseq`` written with the wrong frame layout fails the threshold."""
    name = "head_prone/10"
    assert FIXTURE_SERIES[name]["paravision"]["reco"]["RECO_transposition"][0] == 0
    _write_synthetic_scan(
        tmp_path / name,
        FIXTURE_SERIES[name],
        write_paravision_parameters,
        transpose_2dseq=True,
    )
    seq2d, raw = _load_reconstructions(tmp_path / name)
    zero_shift, _ = magnitude_cross_correlation(raw, seq2d)
    assert zero_shift < CORRELATION_THRESHOLD
