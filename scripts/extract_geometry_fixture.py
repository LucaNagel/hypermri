#!/usr/bin/env python3
"""Extract the geometry-only ParaVision reference fixture.

The geometry reference tests compare hypermri coordinates with the
DICOM export that ParaVision writes for each reconstructed image series.  The
source studies are local, Git-ignored water-phantom acquisitions
that cannot be published, so this script turns them into a small, versioned JSON fixture.

Only whitelisted geometry fields are read:

* ``DICOM_KEYWORDS`` from every exported DICOM frame, read without pixel data;
* ``PARAVISION_FIELDS`` from the ``method``, ``acqp``, ``visu_pars`` and
  ``pdata/1/reco`` parameter files, exactly as ``BrukerHandler`` parses them.

Pixel data, names, identifiers, UIDs, dates, times, device, coil and
institution details, and sequence or protocol names never enter the fixture.
Series are keyed by a caller-chosen study label and the scan number, never by
the source folder name.

Usage::

    python scripts/extract_geometry_fixture.py \
        --study head_prone=<path to head-prone study> \
        --study head_supine=<path to head-supine study>

The output is deterministic for the same input studies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.multival import MultiValue
from pydicom.valuerep import IS, DSfloat

from hypermri.loading.scans import load_scan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "tests" / "reference" / "paravision_water_phantom_geometry.json"
)
DICOM_EXPORT_DIRECTORY = Path("pdata") / "1" / "dicom"
STUDY_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
INLINE_LIST_WIDTH = 88

PER_FRAME_DICOM_FIELDS = {
    "InstanceNumber": "instance_numbers",
    "ImagePositionPatient": "image_position_patient_mm",
    "ImageOrientationPatient": "image_orientation_patient",
}
PER_SERIES_DICOM_FIELDS = {
    "Rows": "rows",
    "Columns": "columns",
    "PixelSpacing": "pixel_spacing_mm",
    "SliceThickness": "slice_thickness_mm",
    "PatientPosition": "patient_position",
    "AnatomicalOrientationType": "anatomical_orientation_type",
}
DICOM_KEYWORDS = (*PER_FRAME_DICOM_FIELDS, *PER_SERIES_DICOM_FIELDS)

PARAVISION_FIELDS = {
    "method": (
        "PVM_SpatDimEnum",
        "PVM_SPackArrSliceOrient",
        "PVM_SPackArrReadOrient",
        "PVM_SPackArrReadOffset",
        "PVM_SPackArrPhase1Offset",
        "PVM_SPackArrSliceOffset",
        "PVM_SliceOffset",
        "PVM_Matrix",
        "PVM_Fov",
        "PVM_SliceThick",
        "PVM_SPackArrSliceGap",
        "PVM_SPackArrNSlices",
        "PVM_ObjOrderList",
    ),
    "acqp": ("ACQ_grad_matrix", "ACQ_patient_pos"),
    "reco": ("RECO_transposition",),
    "visu_pars": (
        "VisuCorePosition",
        "VisuCoreOrientation",
        "VisuCoreSize",
        "VisuCoreExtent",
        "VisuCoreFrameThickness",
        "VisuCoreDiskSliceOrder",
    ),
}
HANDLER_PARAMETER_ATTRIBUTES = {
    "method": "method",
    "acqp": "acqp",
    "reco": "reco",
    "visu_pars": "visu",
}

PROVENANCE = {
    "source": (
        "ParaVision DICOM exports of a water phantom in head-prone and "
        "head-supine position; local and not in Git"
    ),
    "coordinate_system": (
        "DICOM patient coordinates (LPS) in mm, anatomical orientation type "
        "QUADRUPED; ImagePositionPatient is the centre of the first voxel"
    ),
    "dicom_fields": list(DICOM_KEYWORDS),
    "paravision_fields": {
        section: list(keys) for section, keys in PARAVISION_FIELDS.items()
    },
    "paravision_parsing": (
        "values as parsed by hypermri's BrukerHandler from method, acqp, "
        "scan-level visu_pars and pdata/1/reco; null marks an absent parameter"
    ),
    "excluded": (
        "pixel data, names, identifiers, UIDs, dates, times, device, coil and "
        "institution details, sequence and protocol names, source folder names"
    ),
    "extraction_script": "scripts/extract_geometry_fixture.py",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments without the program name.  ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options with ``study`` (list of ``LABEL=PATH`` strings) and
        ``output`` (fixture path).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--study",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Study label and ParaVision study directory, for example "
            "head_prone=/data/study. Repeat for each study."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Fixture path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)}).",
    )
    return parser.parse_args(argv)


def parse_study_argument(argument: str) -> tuple[str, Path]:
    """Split one ``LABEL=PATH`` study argument.

    Parameters
    ----------
    argument : str
        Study label and directory joined by ``=``.  The label becomes the
        series-key prefix in the fixture and must match ``[a-z][a-z0-9_]*`` so
        that no folder name, date or initials can enter the fixture through it.

    Returns
    -------
    tuple of (str, pathlib.Path)
        Study label and study directory.

    Raises
    ------
    ValueError
        If the argument has no ``=``, the label is invalid, or the path is not
        a directory.
    """
    label, separator, path_text = argument.partition("=")
    if not separator:
        raise ValueError(f"--study {argument!r}: expected LABEL=PATH")
    if STUDY_LABEL_RE.fullmatch(label) is None:
        raise ValueError(
            f"--study {argument!r}: label {label!r} must match "
            f"{STUDY_LABEL_RE.pattern}, for example head_prone"
        )
    path = Path(path_text).expanduser()
    if not path.is_dir():
        raise ValueError(f"--study {argument!r}: {path} is not a directory")
    return label, path


def to_json_value(value: Any) -> Any:
    """Convert a parsed parameter value to a JSON-compatible value.

    Parameters
    ----------
    value : Any
        Value parsed from a DICOM element or a ParaVision parameter file:
        ``None``, a string, a Python or NumPy scalar, a NumPy array, or a
        nested list of these.

    Returns
    -------
    Any
        ``None``, ``str``, ``int``, ``float`` or nested lists thereof.  Arrays
        keep their shape as nested lists; strings are kept verbatim, including
        ParaVision enum brackets such as ``"<2D>"``.

    Raises
    ------
    TypeError
        If the value has a type that cannot be represented faithfully.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.ndarray):
        return to_json_value(value.tolist())
    if isinstance(value, np.generic):
        return to_json_value(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    raise TypeError(f"Cannot convert {type(value).__name__} to a fixture value")


def read_dicom_frame_geometry(path: Path) -> dict[str, Any]:
    """Read the whitelisted geometry elements of one DICOM frame.

    Parameters
    ----------
    path : pathlib.Path
        DICOM file of one exported image frame.

    Returns
    -------
    dict
        Mapping from each keyword in ``DICOM_KEYWORDS`` to its value.  Decimal
        strings become ``float`` in mm, integer strings ``int``, and code
        strings ``str``; multi-valued elements become lists.

    Raises
    ------
    ValueError
        If a whitelisted element is missing from the file.

    Notes
    -----
    The file is read with ``stop_before_pixels=True`` and ``specific_tags``
    limited to ``DICOM_KEYWORDS``, so pixel data and all other elements are
    never parsed.
    """
    dataset = pydicom.dcmread(
        path, stop_before_pixels=True, specific_tags=list(DICOM_KEYWORDS)
    )
    geometry = {}
    for keyword in DICOM_KEYWORDS:
        if keyword not in dataset:
            raise ValueError(f"{path}: DICOM element {keyword} is missing")
        value = dataset[keyword].value
        if isinstance(value, MultiValue):
            geometry[keyword] = [_dicom_scalar(item) for item in value]
        else:
            geometry[keyword] = _dicom_scalar(value)
    return geometry


def _dicom_scalar(value: Any) -> Any:
    if isinstance(value, DSfloat):
        return float(value)
    if isinstance(value, IS):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def extract_dicom_series(dicom_dir: Path) -> dict[str, Any]:
    """Collect the DICOM reference geometry of one exported series.

    Parameters
    ----------
    dicom_dir : pathlib.Path
        Directory with the ``*.dcm`` frames of one series.

    Returns
    -------
    dict
        Per-series values (``rows``, ``columns``, ``pixel_spacing_mm`` as
        ``[row spacing, column spacing]``, ``slice_thickness_mm``,
        ``patient_position``, ``anatomical_orientation_type``) and per-frame
        lists ordered by ascending ``InstanceNumber``
        (``instance_numbers``, ``image_position_patient_mm`` with shape
        ``(n_frames, 3)``, ``image_orientation_patient`` with shape
        ``(n_frames, 6)``).

    Raises
    ------
    ValueError
        If the directory has no frames, instance numbers are not unique, or a
        per-series value differs between frames.
    """
    frames = [
        read_dicom_frame_geometry(path) for path in sorted(dicom_dir.glob("*.dcm"))
    ]
    if not frames:
        raise ValueError(f"{dicom_dir}: no DICOM frames found")
    frames.sort(key=lambda frame: frame["InstanceNumber"])
    instance_numbers = [frame["InstanceNumber"] for frame in frames]
    if len(set(instance_numbers)) != len(instance_numbers):
        raise ValueError(f"{dicom_dir}: InstanceNumber values are not unique")

    series: dict[str, Any] = {}
    for keyword, key in PER_SERIES_DICOM_FIELDS.items():
        values = {json.dumps(frame[keyword]) for frame in frames}
        if len(values) != 1:
            raise ValueError(f"{dicom_dir}: {keyword} differs between frames")
        series[key] = frames[0][keyword]
    for keyword, key in PER_FRAME_DICOM_FIELDS.items():
        series[key] = [frame[keyword] for frame in frames]
    return series


def extract_paravision_parameters(scan_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the whitelisted ParaVision geometry parameters of one scan.

    Parameters
    ----------
    scan_dir : pathlib.Path
        ParaVision scan directory with ``method``, ``acqp``, ``visu_pars`` and
        ``pdata/1/reco``.

    Returns
    -------
    dict
        One mapping per section in ``PARAVISION_FIELDS`` from each whitelisted
        parameter to its JSON-compatible value, or ``None`` when absent.

    Raises
    ------
    ValueError
        If the handler could not parse one of the parameter files.

    Notes
    -----
    The scan is opened with ``load_scan(mode="lazy")`` without resolving a
    sequence, so no image data is read or reconstructed.
    """
    handler = load_scan(
        scan_dir, mode="lazy", resolve_sequence=False, log_mode="critical"
    ).handler
    parameters = {}
    for section, keys in PARAVISION_FIELDS.items():
        source = getattr(handler, HANDLER_PARAMETER_ATTRIBUTES[section], None)
        if not isinstance(source, dict):
            raise ValueError(f"{scan_dir}: could not parse the {section} parameters")
        parameters[section] = {key: to_json_value(source.get(key)) for key in keys}
    return parameters


def extract_study(label: str, study_dir: Path) -> dict[str, dict[str, Any]]:
    """Extract every scan of one study that has a DICOM export.

    Parameters
    ----------
    label : str
        Study label used as series-key prefix, for example ``"head_prone"``.
    study_dir : pathlib.Path
        ParaVision study directory whose numbered subdirectories are scans.

    Returns
    -------
    dict
        Mapping from ``"<label>/<scan number>"`` to ``{"dicom": ...,
        "paravision": ...}``, ordered by ascending scan number.  Scans without
        ``pdata/1/dicom`` are skipped.
    """
    scan_dirs = sorted(
        (path for path in study_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    series = {}
    for scan_dir in scan_dirs:
        dicom_dir = scan_dir / DICOM_EXPORT_DIRECTORY
        if not dicom_dir.is_dir():
            continue
        series[f"{label}/{int(scan_dir.name)}"] = {
            "dicom": extract_dicom_series(dicom_dir),
            "paravision": extract_paravision_parameters(scan_dir),
        }
    return series


def build_fixture(studies: list[tuple[str, Path]]) -> dict[str, Any]:
    """Build the complete fixture from labelled studies.

    Parameters
    ----------
    studies : list of tuple of (str, pathlib.Path)
        Study labels and directories in output order.

    Returns
    -------
    dict
        ``{"provenance": PROVENANCE, "series": {...}}``.

    Raises
    ------
    ValueError
        If a label is repeated or no study contains a DICOM export.
    """
    labels = [label for label, _ in studies]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Study labels must be unique, got {labels}")
    series = {}
    for label, study_dir in studies:
        series.update(extract_study(label, study_dir))
    if not series:
        raise ValueError("None of the studies contains a pdata/1/dicom export")
    return {"provenance": PROVENANCE, "series": series}


def format_fixture(value: Any, level: int = 0) -> str:
    """Format fixture JSON with short lists on one line.

    Parameters
    ----------
    value : Any
        JSON-compatible fixture value.
    level : int, optional
        Current indentation level (two spaces per level).

    Returns
    -------
    str
        Deterministic JSON text.  Objects are indented.  A list stays on one
        line when it contains only scalars or when its compact form fits in
        ``INLINE_LIST_WIDTH`` characters, so that coordinate triplets and
        3 x 3 gradient matrices remain readable in review diffs.
    """
    inner = "  " * (level + 1)
    outer = "  " * level
    if isinstance(value, dict):
        items = [
            f"{inner}{json.dumps(key)}: {format_fixture(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{outer}}}" if items else "{}"
    if isinstance(value, list):
        compact = json.dumps(value)
        nested = any(isinstance(item, (dict, list)) for item in value)
        if not nested or ("{" not in compact and len(compact) <= INLINE_LIST_WIDTH):
            return compact
        items = [f"{inner}{format_fixture(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{outer}]"
    return json.dumps(value)


def main(argv: list[str] | None = None) -> int:
    """Run the fixture extraction.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments without the program name.  ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status: ``0`` on success, ``1`` on invalid input.
    """
    args = parse_args(argv)
    try:
        studies = [parse_study_argument(argument) for argument in args.study]
        fixture = build_fixture(studies)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_fixture(fixture) + "\n", encoding="utf-8")
    print(f"Wrote {len(fixture['series'])} series to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
