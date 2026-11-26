"""Raw against ``2dseq`` reconstruction of the local ParaVision phantom.

Opt-in check of the raw-versus-2dseq criterion on the 51 water-phantom series whose geometry
the ParaVision reference fixture ``tests/reference/paravision_water_phantom_geometry.json``
records.  The studies are local and not in Git.  Set
``HYPERMRI_PARAVISION_HEAD_PRONE_STUDY`` and
``HYPERMRI_PARAVISION_HEAD_SUPINE_STUDY`` to the study directories that
``scripts/extract_geometry_fixture.py`` read with ``--study head_prone=...``
and ``--study head_supine=...``; without them the tests are skipped.  Each
scan must reproduce the fixture parameters, so a wrong study fails instead of
being compared.

The comparison and its threshold are those of
``tests/unit/reconstruction/test_bruker_raw_versus_2dseq.py``.  Unlike the
synthetic scans, these series include read, phase and slice offsets and the
multi-package localizers.

Cases that fail because of a known open problem are marked
``xfail(strict=True, raises=AssertionError)`` with a reason describing it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hypermri.loading.scans import load_scan

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.vendor_bruker]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "reference" / "paravision_water_phantom_geometry.json"
)
FIXTURE_SERIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["series"]

STUDY_ENVIRONMENT = {
    "head_prone": "HYPERMRI_PARAVISION_HEAD_PRONE_STUDY",
    "head_supine": "HYPERMRI_PARAVISION_HEAD_SUPINE_STUDY",
}
HANDLER_PARAMETER_ATTRIBUTES = {
    "method": "method",
    "acqp": "acqp",
    "visu_pars": "visu",
    "reco": "reco",
}

CORRELATION_THRESHOLD = 0.99
ALIGNED = (0, 0, 0)
SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
SPATIAL_COORDS = (*SPATIAL_DIMS, "x", "y", "z")


def _is_multi_package(series: dict[str, Any]) -> bool:
    return isinstance(series["paravision"]["method"]["PVM_SPackArrNSlices"], list)


def _series_params(reason_for):
    params = []
    for name, series in FIXTURE_SERIES.items():
        reason = reason_for(series)
        marks = (
            [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)]
            if reason
            else []
        )
        params.append(pytest.param(name, marks=marks, id=name.replace("/", "-")))
    return params


def _package_params():
    """Return one parameter per slice package of every multi-package series."""
    params = []
    for name, series in FIXTURE_SERIES.items():
        if not _is_multi_package(series):
            continue
        n_packages = len(series["paravision"]["method"]["PVM_SPackArrNSlices"])
        params.extend(
            pytest.param(name, package, id=f"{name.replace('/', '-')}-package{package}")
            for package in range(n_packages)
        )
    return params


def _scan_dir(name: str) -> Path:
    study, scan = name.split("/")
    variable = STUDY_ENVIRONMENT[study]
    root = os.environ.get(variable)
    if not root:
        pytest.skip(
            f"Set {variable} to the local ParaVision {study} study of the ParaVision reference "
            "fixture to compare its raw and 2dseq reconstructions"
        )
    scan_dir = Path(root).expanduser() / scan
    if not scan_dir.is_dir():
        pytest.fail(
            f"{variable} has no scan directory {scan!r}; point it at the study "
            "from which tests/reference/paravision_water_phantom_geometry.json "
            "was extracted"
        )
    return scan_dir


def _assert_fixture_parameters(handler, name: str) -> None:
    """Fail when the scan is not the one the fixture was extracted from."""
    for section, attribute in HANDLER_PARAMETER_ATTRIBUTES.items():
        parsed = getattr(handler, attribute)
        for parameter, expected in FIXTURE_SERIES[name]["paravision"][section].items():
            value = parsed.get(parameter)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            assert value == expected, (
                f"{name}: {section} parameter {parameter} differs from the ParaVision reference "
                "fixture; check the study environment variables"
            )


@pytest.fixture(scope="module")
def reconstructions():
    """Return a loader of ``(seq2d, raw)`` that loads every scan once."""
    loaded = {}

    def load(name: str):
        if name not in loaded:
            experiment = load_scan(_scan_dir(name), mode="full", log_mode="critical")
            _assert_fixture_parameters(experiment.handler, name)
            recos = experiment.reconstructions
            loaded[name] = recos["seq2d"], recos["raw"]
        return loaded[name]

    return load


@pytest.mark.parametrize("name", _series_params(lambda series: None))
def test_raw_magnitude_matches_2dseq(
    reconstructions, magnitude_cross_correlation, name
):
    """Raw and ``2dseq`` magnitudes correlate above 0.99 at zero shift."""
    seq2d, raw = reconstructions(name)
    zero_shift, peak_shift = magnitude_cross_correlation(raw, seq2d)
    assert peak_shift == ALIGNED, (name, zero_shift, peak_shift)
    assert zero_shift > CORRELATION_THRESHOLD, (name, zero_shift)


@pytest.mark.parametrize(("name", "package"), _package_params())
def test_raw_magnitude_matches_2dseq_per_slice_package(
    reconstructions, magnitude_cross_correlation, name, package
):
    """Each slice package of a localizer correlates above 0.99 on its own.

    The axial package stores its frames with ``RECO_transposition`` 0, the
    other two with 1.  Previously the ``2dseq`` reconstruction swapped read
    and phase in the axial frames as well, which then correlated with the
    raw magnitude at no more than 0.22 per frame.  The raw reconstruction
    also shifted every package by the offsets of the first package, so the
    coronal package, whose phase offset is -6.34 mm, correlated at no more
    than 0.25 with the peak 13 phase voxels away.
    """
    seq2d, raw = reconstructions(name)
    n_slices = FIXTURE_SERIES[name]["paravision"]["method"]["PVM_SPackArrNSlices"]
    start = sum(n_slices[:package])
    frames = {"slice_idx": slice(start, start + n_slices[package])}
    zero_shift, peak_shift = magnitude_cross_correlation(
        raw.isel(frames), seq2d.isel(frames)
    )
    assert peak_shift == ALIGNED, (name, package, zero_shift, peak_shift)
    assert zero_shift > CORRELATION_THRESHOLD, (name, package, zero_shift)


@pytest.mark.parametrize("name", _series_params(lambda series: None))
def test_raw_and_2dseq_share_spatial_coordinates(reconstructions, name):
    """Index and ``x``, ``y``, ``z`` coordinates agree."""
    seq2d, raw = reconstructions(name)
    for coord in SPATIAL_COORDS:
        np.testing.assert_array_equal(
            raw[coord].transpose(*seq2d[coord].dims).to_numpy(),
            seq2d[coord].to_numpy(),
            err_msg=f"{name}: coordinate {coord!r}",
        )


@pytest.mark.parametrize("name", _series_params(lambda series: None))
def test_raw_and_2dseq_share_the_spatial_order(reconstructions, name):
    """``read_idx``, ``phase_idx`` and ``slice_idx`` follow in this order.

    Previously the 16 coronal series and the two localizers had another order
    in ``2dseq``.  The sizes of the spatial dimensions agree as well.
    """
    seq2d, raw = reconstructions(name)
    for reconstruction in (seq2d, raw):
        spatial = tuple(dim for dim in reconstruction.dims if dim in SPATIAL_DIMS)
        assert spatial == SPATIAL_DIMS, name
    for dim in SPATIAL_DIMS:
        assert seq2d.sizes[dim] == raw.sizes[dim], (name, dim)


def _dimension_order_params():
    """Return all series, including the spatial-only scanner fallback scans."""
    return _series_params(lambda series: None)


@pytest.mark.parametrize("name", _dimension_order_params())
def test_raw_and_2dseq_share_the_dimension_order(reconstructions, name):
    """The raw reconstruction pads and orders its dimensions like ``2dseq``."""
    seq2d, raw = reconstructions(name)
    assert raw.dims == seq2d.dims, name


@pytest.mark.parametrize("name", _dimension_order_params())
def test_raw_and_2dseq_share_coordinate_names(reconstructions, name):
    """The coordinate names of the raw and the ``2dseq`` reconstruction are identical.

    Spatial-only fallback scans receive the same canonical imaging layout.
    """
    seq2d, raw = reconstructions(name)
    assert set(raw.coords) == set(seq2d.coords), name


@pytest.mark.parametrize("name", _dimension_order_params())
def test_raw_and_2dseq_share_nonspatial_coordinate_values(reconstructions, name):
    """Timing/index labels agree to 1 ps for these identical parsed time axes."""
    seq2d, raw = reconstructions(name)
    for coordinate in set(seq2d.coords) - set(SPATIAL_COORDS):
        assert raw[coordinate].dims == seq2d[coordinate].dims, (name, coordinate)
        actual, expected = raw[coordinate].values, seq2d[coordinate].values
        if np.issubdtype(expected.dtype, np.number):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=0,
                atol=1e-12,
                err_msg=f"{name}: nonspatial coordinate {coordinate!r}",
            )
        else:
            np.testing.assert_array_equal(
                actual, expected, err_msg=f"{name}: coordinate {coordinate!r}"
            )
    for coordinate in ("time", "acquisition_time_s"):
        if coordinate in raw.coords:
            assert raw[coordinate].attrs["unit"] == "s"
            assert seq2d[coordinate].attrs["unit"] == "s"


@pytest.mark.parametrize("name", _dimension_order_params())
def test_raw_and_2dseq_share_orientation_attributes(reconstructions, name):
    """The patient position and three physical voxel directions agree."""
    seq2d, raw = reconstructions(name)
    for attribute in ("patient_pos", "slice_orientation", "read_orientation"):
        assert raw.attrs[attribute] == seq2d.attrs[attribute], (name, attribute)
    for attribute in ("read_direction", "phase_direction", "slice_direction"):
        np.testing.assert_allclose(
            raw.attrs[attribute],
            seq2d.attrs[attribute],
            rtol=0,
            atol=1e-12,
            err_msg=f"{name}: orientation {attribute!r}",
        )
