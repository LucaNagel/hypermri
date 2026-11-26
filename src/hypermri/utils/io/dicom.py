"""DICOM series discovery and conversion to canonical xarray images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


@dataclass(frozen=True)
class DicomSeriesInfo:
    """Header-level description of one DICOM series on disk."""

    series_instance_uid: str
    paths: tuple[Path, ...]
    series_description: str = ""
    modality: str = ""
    rows: int | None = None
    columns: int | None = None

    @property
    def number_of_instances(self) -> int:
        return len(self.paths)


def _candidate_paths(source: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_file():
            return [path]
        if not path.is_dir():
            raise FileNotFoundError(f"DICOM source does not exist: {path}")
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and not candidate.name.startswith(".")
        )

    return sorted(Path(path) for path in source if Path(path).is_file())


def discover_dicom_series(
    source: str | Path | Iterable[str | Path],
) -> list[DicomSeriesInfo]:
    """Discover readable DICOM files and group them by Series Instance UID."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required for DICOM discovery") from exc

    grouped: dict[str, list[tuple[Path, object]]] = {}
    for path in _candidate_paths(source):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not series_uid:
            logger.debug("Ignoring DICOM without SeriesInstanceUID: %s", path)
            continue
        grouped.setdefault(series_uid, []).append((path, dataset))

    series = []
    for series_uid, entries in grouped.items():
        entries.sort(
            key=lambda item: (
                int(getattr(item[1], "InstanceNumber", 0) or 0),
                str(item[0]),
            )
        )
        first = entries[0][1]
        series.append(
            DicomSeriesInfo(
                series_instance_uid=series_uid,
                paths=tuple(path for path, _ in entries),
                series_description=str(getattr(first, "SeriesDescription", "") or ""),
                modality=str(getattr(first, "Modality", "") or ""),
                rows=int(first.Rows) if hasattr(first, "Rows") else None,
                columns=int(first.Columns) if hasattr(first, "Columns") else None,
            )
        )
    return sorted(series, key=lambda item: item.series_instance_uid)


def _select_series(
    source: DicomSeriesInfo | str | Path | Iterable[str | Path],
    series_uid: str | None,
) -> DicomSeriesInfo:
    if isinstance(source, DicomSeriesInfo):
        if series_uid is not None and source.series_instance_uid != series_uid:
            raise ValueError(
                f"Requested series {series_uid}, got {source.series_instance_uid}"
            )
        return source

    discovered = discover_dicom_series(source)
    if series_uid is not None:
        matches = [
            item for item in discovered if item.series_instance_uid == series_uid
        ]
        if not matches:
            raise ValueError(f"DICOM series not found: {series_uid}")
        return matches[0]
    if not discovered:
        raise ValueError("No readable DICOM series found")
    if len(discovered) > 1:
        raise ValueError(
            "Multiple DICOM series found; pass series_uid explicitly: "
            f"{[item.series_instance_uid for item in discovered]}"
        )
    return discovered[0]


def _frame_value(dataset) -> int | float | str:
    for name in (
        "TemporalPositionIdentifier",
        "AcquisitionNumber",
        "EchoNumbers",
    ):
        value = getattr(dataset, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)
    return 1


def _normalise_direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("DICOM direction vector has zero length")
    return vector / norm


def load_dicom_series(
    source: DicomSeriesInfo | str | Path | Iterable[str | Path],
    *,
    series_uid: str | None = None,
    dtype: np.dtype | str = np.float32,
) -> xr.DataArray:
    """Load one classic single-frame DICOM series as a spatial DataArray.

    Repeated slice stacks are exposed as ``repetition_idx``. Enhanced
    multi-frame DICOM is intentionally rejected until its functional-group
    geometry can be handled without guessing.
    """
    try:
        import pydicom
        from pydicom.pixels import apply_rescale
    except ImportError as exc:
        raise ImportError("pydicom is required for DICOM loading") from exc

    info = _select_series(source, series_uid)
    datasets = [pydicom.dcmread(path) for path in info.paths]
    if not datasets:
        raise ValueError("DICOM series contains no instances")
    if any(
        int(getattr(dataset, "NumberOfFrames", 1) or 1) != 1 for dataset in datasets
    ):
        raise NotImplementedError("Enhanced multi-frame DICOM is not supported yet")

    first = datasets[0]
    required = ("Rows", "Columns", "PixelSpacing", "ImageOrientationPatient")
    missing = [name for name in required if not hasattr(first, name)]
    if missing:
        raise ValueError(f"DICOM series is missing geometry tags: {missing}")
    if any(not hasattr(dataset, "ImagePositionPatient") for dataset in datasets):
        raise ValueError("Every DICOM instance must define ImagePositionPatient")

    rows, columns = int(first.Rows), int(first.Columns)
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)
    if pixel_spacing.size != 2:
        raise ValueError("PixelSpacing must contain row and column spacing")
    row_spacing, column_spacing = (float(value) for value in pixel_spacing)

    orientation = np.asarray(first.ImageOrientationPatient, dtype=float).reshape(2, 3)
    column_direction = _normalise_direction(orientation[0])
    row_direction = _normalise_direction(orientation[1])
    slice_normal = _normalise_direction(np.cross(column_direction, row_direction))

    phase_encoding = str(
        getattr(first, "InPlanePhaseEncodingDirection", "COL") or "COL"
    ).upper()
    if phase_encoding not in {"ROW", "COL"}:
        phase_encoding = "COL"

    grouped: dict[int | float | str, list[object]] = {}
    for dataset in datasets:
        if int(dataset.Rows) != rows or int(dataset.Columns) != columns:
            raise ValueError("DICOM series contains inconsistent image matrices")
        grouped.setdefault(_frame_value(dataset), []).append(dataset)

    frame_values = sorted(grouped, key=lambda value: (str(type(value)), value))
    positions_by_frame: list[list[np.ndarray]] = []
    sorted_frames: list[list[object]] = []
    expected_projections: np.ndarray | None = None
    for frame_value in frame_values:
        frame = sorted(
            grouped[frame_value],
            key=lambda dataset: float(
                np.dot(
                    np.asarray(dataset.ImagePositionPatient, dtype=float),
                    slice_normal,
                )
            ),
        )
        projections = np.asarray(
            [
                np.dot(
                    np.asarray(dataset.ImagePositionPatient, dtype=float),
                    slice_normal,
                )
                for dataset in frame
            ],
            dtype=float,
        )
        if len(np.unique(np.round(projections, decimals=6))) != len(frame):
            raise ValueError(
                "DICOM frame contains duplicate slice positions; "
                "additional dimensions are not supported yet"
            )
        if expected_projections is None:
            expected_projections = projections
        elif projections.shape != expected_projections.shape or not np.allclose(
            projections, expected_projections, atol=1e-4
        ):
            raise ValueError("DICOM frames do not share the same slice positions")
        sorted_frames.append(frame)
        positions_by_frame.append(
            [np.asarray(dataset.ImagePositionPatient, dtype=float) for dataset in frame]
        )

    slice_count = len(sorted_frames[0])
    if phase_encoding == "ROW":
        read_size, phase_size = rows, columns
        read_direction, phase_direction = row_direction, column_direction
        read_spacing, phase_spacing = row_spacing, column_spacing
    else:
        read_size, phase_size = columns, rows
        read_direction, phase_direction = column_direction, row_direction
        read_spacing, phase_spacing = column_spacing, row_spacing

    values = np.empty(
        (read_size, phase_size, slice_count, len(frame_values)), dtype=dtype
    )
    for frame_index, frame in enumerate(sorted_frames):
        for slice_index, dataset in enumerate(frame):
            pixels = np.asarray(
                apply_rescale(dataset.pixel_array, dataset), dtype=dtype
            )
            values[:, :, slice_index, frame_index] = (
                pixels if phase_encoding == "ROW" else pixels.T
            )

    origins = np.asarray(positions_by_frame[0], dtype=float)[None, None, :, :]
    read_offsets = (
        np.arange(read_size, dtype=float)[:, None, None, None]
        * read_spacing
        * read_direction
    )
    phase_offsets = (
        np.arange(phase_size, dtype=float)[None, :, None, None]
        * phase_spacing
        * phase_direction
    )
    xyz = origins + read_offsets + phase_offsets

    coords: dict[str, object] = {
        "read_idx": (
            "read_idx",
            np.arange(read_size, dtype=float) * read_spacing,
            {"unit": "mm"},
        ),
        "phase_idx": (
            "phase_idx",
            np.arange(phase_size, dtype=float) * phase_spacing,
            {"unit": "mm"},
        ),
        "slice_idx": (
            "slice_idx",
            expected_projections,
            {"unit": "mm"},
        ),
    }
    xyz_attrs = {"unit": "mm", "coordinate_system": "dicom_lps"}
    for name, component in zip(("x", "y", "z"), range(3), strict=True):
        coords[name] = (
            ("read_idx", "phase_idx", "slice_idx"),
            xyz[..., component],
            xyz_attrs,
        )

    dims = ["read_idx", "phase_idx", "slice_idx", "repetition_idx"]
    coords["repetition_idx"] = (
        "repetition_idx",
        frame_values,
        {"unit": "index"},
    )
    data_array = xr.DataArray(
        values,
        dims=dims,
        coords=coords,
        name=info.series_description or "dicom_image",
    )
    if len(frame_values) == 1:
        data_array = data_array.isel(repetition_idx=0, drop=True)

    data_array.attrs.update(
        {
            "source": "dicom",
            "domain": "spatial",
            "unit": str(getattr(first, "RescaleType", "a.u.") or "a.u."),
            "series_instance_uid": info.series_instance_uid,
            "study_instance_uid": str(getattr(first, "StudyInstanceUID", "") or ""),
            "frame_of_reference_uid": str(
                getattr(first, "FrameOfReferenceUID", "") or ""
            ),
            "modality": str(getattr(first, "Modality", "") or ""),
            "series_description": info.series_description,
            "inplane_phase_encoding_direction": phase_encoding,
            "read_direction": read_direction,
            "phase_direction": phase_direction,
            "slice_direction": slice_normal,
        }
    )
    return data_array


__all__ = ["DicomSeriesInfo", "discover_dicom_series", "load_dicom_series"]
