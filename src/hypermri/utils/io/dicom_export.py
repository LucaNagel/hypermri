"""Geometry-aware DICOM export for xarray image data.

The exporter writes a derived MR series whose ``ImagePositionPatient``,
``ImageOrientationPatient`` and ``PixelSpacing`` come from the data's DICOM
LPS ``x``, ``y`` and ``z`` coordinates, from a geometry source or from
reference DICOMs.  Data without any of these are rejected unless the caller
explicitly accepts a labelled identity geometry.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import xarray as xr

from ..system.logging import init_default_logger
from .dicom import DicomSeriesInfo, discover_dicom_series

logger = init_default_logger(__name__)

# DICOM limits SeriesDescription (VR LO) to 64 characters.
_SERIES_DESCRIPTION_LENGTH = 64
_IDENTITY_GEOMETRY_LABEL = " [identity geometry]"
_IDENTITY_GEOMETRY_COMMENT = (
    "Identity geometry: HyperMRI wrote a centred 1 mm grid because the data "
    "had no spatial geometry. Positions and orientation are not the patient "
    "position."
)


@dataclass(frozen=True)
class _ExportGeometry:
    row_dim: str
    column_dim: str
    image_orientation_patient: np.ndarray
    image_positions_patient: np.ndarray
    pixel_spacing: tuple[float, float]
    slice_thickness: float
    slice_spacing: float
    source: str


def _spatial_data_array(source: Any) -> xr.DataArray | None:
    if isinstance(source, xr.DataArray):
        return source
    if isinstance(source, xr.Dataset):
        candidates = [
            value
            for value in source.data_vars.values()
            if {"read_idx", "phase_idx"}.issubset(value.dims)
        ]
        return candidates[0] if len(candidates) == 1 else None
    for name in ("primary_data", "data", "scanner_data"):
        try:
            value = getattr(source, name, None)
        except Exception:
            continue
        if isinstance(value, xr.DataArray):
            return value
    return None


def _xyz_values(data: xr.DataArray) -> np.ndarray | None:
    spatial_dims = ["read_idx", "phase_idx"]
    has_slice = "slice_idx" in data.dims
    if has_slice:
        spatial_dims.append("slice_idx")
    if any(name not in data.coords for name in ("x", "y", "z")):
        return None

    components = []
    for name in ("x", "y", "z"):
        coord = data.coords[name]
        if not all(dim in coord.dims for dim in spatial_dims):
            return None
        values = np.asarray(coord.transpose(*spatial_dims).values, dtype=float)
        if not has_slice:
            values = values[..., None]
        components.append(values)
    return np.stack(components, axis=-1)


def _unit_vector(vector: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0:
        raise ValueError(f"Cannot derive {name} from XYZ coordinates")
    return np.asarray(vector, dtype=float) / length, length


def _positive_slice_scalar(value: Any, name: str) -> float | None:
    """Validate a supplied scalar length in mm without inventing missing data."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    values = np.asarray(value)
    try:
        result = float(values.item())
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"DICOM export requires a positive scalar {name} in mm"
        ) from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"DICOM export requires a positive scalar {name} in mm")
    return result


def _slice_attribute(data: xr.DataArray, name: str) -> float | None:
    """Read slice-coordinate metadata first, then array metadata after selection."""
    coord = data.coords.get("slice_idx")
    value = getattr(coord, "attrs", {}).get(name)
    if value is None:
        value = data.attrs.get(name)
    return _positive_slice_scalar(value, name)


def _nonregular_stack_message(source: Any) -> str:
    """Name known scanner packages and explain the one-stack export limit."""
    handler = getattr(source, "handler", source)
    counts = np.atleast_1d(getattr(handler, "slices_per_pack", ()))
    packages = ""
    if counts.size > 1:
        names = ", ".join(
            f"package {index} ({int(count)} slices)"
            for index, count in enumerate(counts, start=1)
        )
        packages = f" The geometry source contains {names}."
    return (
        "XYZ slice positions do not form a regular orthogonal stack."
        f"{packages} DICOM export writes one regular stack per series; "
        "select and export each slice package separately with its own XYZ "
        "coordinates. For a single package, verify its slice positions."
    )


def _geometry_from_xyz(
    data: xr.DataArray, *, geometry_source: Any = None
) -> _ExportGeometry | None:
    xyz = _xyz_values(data)
    if xyz is None:
        return None
    read_size, phase_size, slice_size = xyz.shape[:3]
    if read_size < 2 or phase_size < 2:
        raise ValueError("XYZ-based DICOM export requires at least 2x2 in-plane pixels")

    read_direction, row_spacing = _unit_vector(
        xyz[1, 0, 0] - xyz[0, 0, 0], "read direction"
    )
    phase_direction, column_spacing = _unit_vector(
        xyz[0, 1, 0] - xyz[0, 0, 0], "phase direction"
    )
    if abs(float(np.dot(read_direction, phase_direction))) > 1e-4:
        raise ValueError("XYZ in-plane directions are not orthogonal")

    positions = np.asarray(xyz[0, 0, :], dtype=float)
    slice_normal = np.cross(phase_direction, read_direction)
    # Legacy raw 3D arrays stored excitation slab thickness globally. Without
    # the new sampling metadata, use the partition geometry.
    dimension = data.attrs.get("spatial_dimension")
    if dimension == "3D" and "slice_spacing_mm" not in data.attrs:
        coord = data.coords.get("slice_idx")
        slice_thickness = _positive_slice_scalar(
            getattr(coord, "attrs", {}).get("slice_thickness_mm"), "slice_thickness_mm"
        )
    else:
        slice_thickness = _slice_attribute(data, "slice_thickness_mm")
    if slice_size > 1:
        slice_steps = np.diff(positions, axis=0)
        slice_spacing = float(np.median(np.linalg.norm(slice_steps, axis=1)))
        if not np.allclose(
            np.abs(slice_steps @ slice_normal),
            slice_spacing,
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError(_nonregular_stack_message(geometry_source))
    else:
        slice_spacing = _slice_attribute(data, "slice_spacing_mm")
        if slice_spacing is None:
            slice_coord = data.coords.get("slice_idx")
            slice_spacing = _positive_slice_scalar(
                getattr(slice_coord, "attrs", {}).get("resolution_mm"), "resolution_mm"
            )
        if slice_spacing is None:
            slice_spacing = slice_thickness or 1.0
    if slice_thickness is None:
        slice_thickness = slice_spacing

    orientation = np.concatenate([phase_direction, read_direction])
    return _ExportGeometry(
        row_dim="read_idx",
        column_dim="phase_idx",
        image_orientation_patient=orientation,
        image_positions_patient=positions,
        pixel_spacing=(row_spacing, column_spacing),
        slice_thickness=slice_thickness,
        slice_spacing=slice_spacing,
        source="xarray_xyz",
    )


def _normalise_reference_datasets(reference: Any) -> list[Any]:
    if reference is None:
        return []
    try:
        import pydicom
        from pydicom.dataset import Dataset
    except ImportError as exc:
        raise ImportError("pydicom is required for DICOM export") from exc

    if isinstance(reference, Dataset):
        return [reference]
    if isinstance(reference, DicomSeriesInfo):
        return [pydicom.dcmread(path) for path in reference.paths]
    if isinstance(reference, str | Path):
        path = Path(reference)
        discovered = discover_dicom_series(path)
        if not discovered:
            raise ValueError(f"No readable DICOM series found at {path}")
        selected = max(
            discovered,
            key=lambda item: (item.number_of_instances, item.series_instance_uid),
        )
        return [pydicom.dcmread(item) for item in selected.paths]
    # A scan holds its loaded DICOM datasets on its vendor handler.
    vendor_source = getattr(reference, "handler", reference)
    if hasattr(vendor_source, "dicom"):
        datasets = getattr(vendor_source, "dicom", None)
        if datasets:
            return list(datasets)
    if isinstance(reference, Sequence):
        datasets = list(reference)
        if datasets and all(isinstance(item, Dataset) for item in datasets):
            return datasets
    raise TypeError(
        "reference_dicoms must be a path, DicomSeriesInfo, pydicom Dataset, "
        "sequence of Datasets, or scan object with loaded DICOMs"
    )


def _unique_reference_slices(datasets: Sequence[Any]) -> list[Any]:
    ordered = sorted(
        datasets,
        key=lambda dataset: int(getattr(dataset, "InstanceNumber", 0) or 0),
    )
    unique = []
    seen = set()
    for dataset in ordered:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is None:
            continue
        key = tuple(np.round(np.asarray(position, dtype=float), decimals=6))
        if key not in seen:
            seen.add(key)
            unique.append(dataset)
    return unique


def _geometry_from_references(
    datasets: Sequence[Any], data: xr.DataArray
) -> _ExportGeometry | None:
    if not datasets:
        return None
    slices = _unique_reference_slices(datasets)
    if not slices:
        return None
    first = slices[0]
    required = ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing")
    if any(not hasattr(first, name) for name in required):
        return None

    slice_count = int(data.sizes.get("slice_idx", 1))
    if len(slices) != slice_count:
        raise ValueError(
            f"Reference DICOM has {len(slices)} unique slices, data has {slice_count}"
        )
    phase_encoding = str(
        getattr(first, "InPlanePhaseEncodingDirection", "COL") or "COL"
    ).upper()
    if phase_encoding == "ROW":
        row_dim, column_dim = "read_idx", "phase_idx"
    else:
        row_dim, column_dim = "phase_idx", "read_idx"

    positions = np.asarray(
        [dataset.ImagePositionPatient for dataset in slices], dtype=float
    )
    spacing = tuple(float(value) for value in first.PixelSpacing)
    slice_thickness = (
        _positive_slice_scalar(getattr(first, "SliceThickness", None), "SliceThickness")
        or 1.0
    )
    if slice_count > 1:
        slice_spacing = float(
            np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        )
    else:
        slice_spacing = (
            _positive_slice_scalar(
                getattr(first, "SpacingBetweenSlices", None), "SpacingBetweenSlices"
            )
            or slice_thickness
        )
    return _ExportGeometry(
        row_dim=row_dim,
        column_dim=column_dim,
        image_orientation_patient=np.asarray(
            first.ImageOrientationPatient, dtype=float
        ),
        image_positions_patient=positions,
        pixel_spacing=spacing,
        slice_thickness=slice_thickness,
        slice_spacing=slice_spacing,
        source="reference_dicom",
    )


def _geometry_from_source(source: Any, data: xr.DataArray) -> _ExportGeometry | None:
    source_data = _spatial_data_array(source)
    if source_data is not None:
        expected = {
            dim: int(data.sizes.get(dim, 1))
            for dim in ("read_idx", "phase_idx", "slice_idx")
        }
        actual = {
            dim: int(source_data.sizes.get(dim, 1))
            for dim in ("read_idx", "phase_idx", "slice_idx")
        }
        if expected != actual:
            raise ValueError(
                f"Geometry source spatial shape {actual} does not match data {expected}"
            )
        geometry = _geometry_from_xyz(source_data, geometry_source=source)
        if geometry is not None:
            return geometry

    if source is not None:
        from hypermri.utils.coordinates.grids import assign_xyz_coordinates

        template = xr.DataArray(
            np.empty(
                (
                    int(data.sizes["read_idx"]),
                    int(data.sizes["phase_idx"]),
                    int(data.sizes.get("slice_idx", 1)),
                )
            ),
            dims=("read_idx", "phase_idx", "slice_idx"),
        )
        template = assign_xyz_coordinates(template, scan=source, dtype=np.float64)
        slice_geometry = getattr(
            getattr(source, "handler", source), "_slice_geometry_attributes", None
        )
        if slice_geometry is not None:
            template.attrs.update(slice_geometry())
        return _geometry_from_xyz(template, geometry_source=source)
    return None


def _fallback_geometry(data: xr.DataArray) -> _ExportGeometry:
    read_size = int(data.sizes["read_idx"])
    phase_size = int(data.sizes["phase_idx"])
    slice_size = int(data.sizes.get("slice_idx", 1))
    positions = np.column_stack(
        [
            np.full(slice_size, -(read_size - 1) / 2.0),
            np.full(slice_size, -(phase_size - 1) / 2.0),
            np.arange(slice_size, dtype=float) - (slice_size - 1) / 2.0,
        ]
    )
    return _ExportGeometry(
        row_dim="phase_idx",
        column_dim="read_idx",
        image_orientation_patient=np.asarray([1, 0, 0, 0, 1, 0], dtype=float),
        image_positions_patient=positions,
        pixel_spacing=(1.0, 1.0),
        slice_thickness=1.0,
        slice_spacing=1.0,
        source="fallback_identity",
    )


def _missing_geometry_message(
    geometry: Any, references: Sequence[Any], data: xr.DataArray
) -> str:
    if geometry is None:
        source = "no geometry source was given"
    else:
        shape = " x ".join(
            f"{dim} {int(data.sizes.get(dim, 1))}"
            for dim in ("read_idx", "phase_idx", "slice_idx")
        )
        source = (
            f"the geometry source {type(geometry).__name__} yields no x, y and "
            f"z coordinates for {shape} (no Bruker VisuCore geometry, or a "
            "different matrix)"
        )
    if references:
        reference = (
            "the reference DICOMs lack ImageOrientationPatient, "
            "ImagePositionPatient or PixelSpacing"
        )
    else:
        reference = "no reference DICOMs were given"
    return (
        "DICOM export found no spatial geometry: the data have no x, y and z "
        f"coordinates, {source}, and {reference}.  Pass geometry= (a scan or "
        "a DataArray with x, y and z), reference_dicoms= (a DICOM series with "
        "the same slices), or allow_identity_geometry=True to write a centred "
        "1 mm identity grid that is labelled as such."
    )


def _label_identity_geometry(series_description: str) -> str:
    kept = _SERIES_DESCRIPTION_LENGTH - len(_IDENTITY_GEOMETRY_LABEL)
    return f"{series_description[:kept]}{_IDENTITY_GEOMETRY_LABEL}"


def _prepare_frames(
    data: xr.DataArray, geometry: _ExportGeometry
) -> tuple[np.ndarray, list[str], tuple[int, ...]]:
    spatial_dims = {"read_idx", "phase_idx", "slice_idx"}
    frame_priority = (
        "acquisition_idx",
        "repetition_idx",
        "time_idx",
        "time",
        "inversion_idx",
        "echo_idx",
        "dynamic_idx",
    )
    frame_dims = [dim for dim in frame_priority if dim in data.dims]
    frame_dims.extend(
        dim for dim in data.dims if dim not in spatial_dims and dim not in frame_dims
    )
    slice_dim = ["slice_idx"] if "slice_idx" in data.dims else []
    order = frame_dims + slice_dim + [geometry.row_dim, geometry.column_dim]
    ordered = data.transpose(*order)
    frame_shape = tuple(int(ordered.sizes[dim]) for dim in frame_dims)
    frame_count = int(np.prod(frame_shape)) if frame_shape else 1
    slice_count = int(ordered.sizes.get("slice_idx", 1))
    rows = int(ordered.sizes[geometry.row_dim])
    columns = int(ordered.sizes[geometry.column_dim])
    values = np.asarray(ordered.values).reshape(frame_count, slice_count, rows, columns)
    if np.iscomplexobj(values):
        logger.warning("Complex-valued data detected; exporting magnitude")
        values = np.abs(values)
    return values, frame_dims, frame_shape


def _scale_pixel_data(values: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    values = np.asarray(values)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Cannot export an image series containing only NaN/Inf")
    has_padding = not finite.all()
    finite_values = values[finite]

    if (
        np.issubdtype(values.dtype, np.integer)
        and finite_values.min() >= -32767
        and finite_values.max() <= 32767
    ):
        pixels = values.astype(np.int16, copy=True)
        if has_padding:
            pixels[~finite] = -32768
        return pixels, 1.0, 0.0, has_padding

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if minimum == maximum:
        slope = 1.0
        intercept = minimum
        pixels = np.zeros(values.shape, dtype=np.int16)
    else:
        slope = (maximum - minimum) / 65534.0
        intercept = minimum + 32767.0 * slope
        safe_values = np.where(finite, values, intercept)
        pixels = np.rint((safe_values - intercept) / slope).astype(np.int16)
    if has_padding:
        pixels[~finite] = -32768
    return pixels, slope, intercept, has_padding


_REFERENCE_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "StudyDescription",
    "ReferringPhysicianName",
    "PatientPosition",
    "PositionReferenceIndicator",
)


def export_to_dicom(
    data: xr.DataArray | np.ndarray,
    output_dir: str | Path,
    reference_dicoms: Any = None,
    filename_prefix: str = "MRIm",
    series_description: str = "Exported Data",
    series_number: int = 999,
    unit: str | None = None,
    *,
    geometry: Any = None,
    allow_identity_geometry: bool = False,
) -> bool:
    """
    Export image data as a geometry-complete derived MR DICOM series.

    The data are written as one MR Image Storage instance per frame and
    slice.  Their spatial placement is taken from the first available of
    these sources:

    1. ``x``, ``y`` and ``z`` coordinates of ``data`` in mm (DICOM LPS) over
       ``read_idx``, ``phase_idx`` and, if present, ``slice_idx``;
    2. ``geometry``: its ``x``, ``y`` and ``z`` coordinates, or the Bruker
       ``VisuCore`` geometry of a scan object;
    3. ``reference_dicoms``: ``ImageOrientationPatient``,
       ``ImagePositionPatient``, ``PixelSpacing`` and ``SliceThickness`` of a
       series with one position per slice of ``data``.

    Without any of them the export fails, unless ``allow_identity_geometry``
    accepts a centred 1 mm identity grid that is labelled in the series.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Image values.  A DataArray needs the dimensions ``read_idx`` and
        ``phase_idx`` and may have ``slice_idx``.  Every other dimension of
        size greater than one becomes a frame dimension: ``acquisition_idx``,
        ``repetition_idx``, ``time_idx``, ``time``, ``inversion_idx``,
        ``echo_idx`` and ``dynamic_idx`` in this order, then the rest in the
        order of ``data``.  Other size-one dimensions are dropped.  A NumPy
        array must be 2D or 3D after squeezing; its axes are
        ``(phase_idx, read_idx)`` or ``(phase_idx, read_idx, slice_idx)`` and
        it has no geometry of its own.  Complex values are exported as
        magnitude. Slice-coordinate attributes ``slice_thickness_mm`` and
        ``slice_spacing_mm`` describe physical lengths; array attributes
        supply them when the coordinate attribute is missing, including
        after dropping ``slice_idx``. Supplied lengths must be positive,
        finite scalars in mm. The input is not modified.
    output_dir : str or pathlib.Path
        Directory for the DICOM files.  It and its parents are created when
        missing; files with the same names are overwritten.
    reference_dicoms : object, optional
        A DICOM directory (the series with the most instances is used), a
        :class:`~hypermri.utils.io.dicom.DicomSeriesInfo`, a
        ``pydicom.Dataset``, a sequence of Datasets, or a scan object with
        loaded DICOMs in ``dicom``.  Besides the geometry fallback, its
        ``StudyInstanceUID``, ``FrameOfReferenceUID`` and patient and study
        tags (``PatientName``, ``PatientID``, ``PatientBirthDate``,
        ``PatientSex``, ``StudyDate``, ``StudyTime``, ``AccessionNumber``,
        ``StudyDescription``, ``ReferringPhysicianName``,
        ``PatientPosition``, ``PositionReferenceIndicator``) are copied into
        every instance.  Without it, new study and frame-of-reference UIDs
        are generated.
    filename_prefix : str, optional
        File-name prefix; files are named ``<prefix><instance:04d>.dcm``.
    series_description : str, optional
        ``SeriesDescription`` of the new series.
    series_number : int, optional
        ``SeriesNumber`` of the new series.
    unit : str, optional
        Unit of the rescaled values, written to ``RescaleType``, for example
        ``"s"`` or ``"a.u."``.  Omitted when None.
    geometry : object, optional
        Geometry source when ``data`` has no ``x``, ``y`` and ``z``: a
        DataArray or Dataset with these coordinates, or a scan object whose
        primary data carry them or whose ``VisuCore`` parameters define them.
        Its ``read_idx``, ``phase_idx`` and ``slice_idx`` sizes must equal
        those of ``data``. When a stack is rejected, known scanner slice
        packages are named in the error even if ``data`` supplies its own
        XYZ coordinates.
    allow_identity_geometry : bool, optional
        If True and no geometry source applies, write a 1 mm grid centred at
        the isocentre with ``ImageOrientationPatient`` ``[1, 0, 0, 0, 1, 0]``,
        rows along ``phase_idx`` and columns along ``read_idx``.  The series
        is then labelled: ``SeriesDescription`` ends in
        ``" [identity geometry]"`` (the description is shortened to keep the
        DICOM limit of 64 characters), ``ImageComments`` states that the
        positions are not the patient position, and a ``UserWarning`` is
        issued.  Ignored when a geometry source applies.  Defaults to False,
        which rejects data without geometry.

    Returns
    -------
    bool
        True once all instances are written.

    Raises
    ------
    ImportError
        If ``pydicom`` is not installed.
    TypeError
        If ``reference_dicoms`` has an unsupported type.
    ValueError
        If the data lack ``read_idx`` or ``phase_idx``, a NumPy array is not
        2D or 3D, no geometry is found and ``allow_identity_geometry`` is
        False, the geometry source or reference series does not match the
        spatial shape of ``data``, the ``x``, ``y`` and ``z`` coordinates do
        not form a regular orthogonal grid, a reference directory contains
        no readable DICOM series, a used slice-size attribute is not a
        positive finite scalar, or all values are NaN or infinite. Nothing
        is written in these cases.

    Warns
    -----
    UserWarning
        If the identity geometry is written.

    Notes
    -----
    ``SliceThickness`` is the nominal physical slice thickness, independently
    of ``SpacingBetweenSlices``. With XYZ geometry, explicit thickness is
    used when available. Spacing comes from neighbouring XYZ centres for a
    stack, or explicit ``slice_spacing_mm`` for a single plane. A singleton
    without explicit spacing uses its coordinate's legacy ``resolution_mm``,
    then known thickness, then 1 mm. Missing thickness falls back to spacing.
    Legacy 3D array-level thickness without ``slice_spacing_mm`` is ignored
    because it may describe the excitation slab rather than a partition.
    A reference series supplies ``SliceThickness`` (default 1 mm) and either
    centre distances or, for one plane, ``SpacingBetweenSlices`` (defaulting
    to thickness). Export does not change positions to match these sizes.

    Each series requires one regular orthogonal slice stack. XYZ slice
    positions that do not form such a stack are rejected before writing;
    the error names packages when
    ``geometry`` provides ``slices_per_pack`` (directly or on its handler).
    Select each package along ``slice_idx`` and export it to a separate
    directory with its own XYZ coordinates. ``allow_identity_geometry``
    does not bypass rejection of an invalid stack.

    Instance numbers run over slices fastest, then over frames; each frame
    is one ``AcquisitionNumber`` and, for several frames, one
    ``TemporalPositionIdentifier``.  Values are stored as signed 16-bit
    integers with ``RescaleSlope`` and ``RescaleIntercept`` shared by the
    whole series; integer data within +-32767 are stored unscaled.
    Non-finite values become the ``PixelPaddingValue`` -32768.  Every
    instance is marked ``DERIVED``/``SECONDARY`` and names its geometry
    source (``xarray_xyz``, ``reference_dicom`` or ``fallback_identity``) in
    ``DerivationDescription``.  UIDs and ``ContentDate``/``ContentTime`` are
    new on every call, so repeated exports are not byte-identical.

    Examples
    --------
    Export a fitted map with the geometry of its scan:

    >>> export_to_dicom(t1_map, "t1_dicom", geometry=scan, unit="s")  # doctest: +SKIP
    True

    Export data without geometry, accepting the labelled identity grid:

    >>> export_to_dicom(
    ...     values, "values_dicom", allow_identity_geometry=True
    ... )  # doctest: +SKIP
    True
    """
    try:
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    except ImportError as exc:
        raise ImportError("pydicom is required for DICOM export") from exc

    if isinstance(data, xr.DataArray):
        squeeze_dims = [
            dim
            for dim, size in data.sizes.items()
            if size == 1 and dim not in {"read_idx", "phase_idx", "slice_idx"}
        ]
        image = data.squeeze(squeeze_dims, drop=True)
    else:
        values = np.squeeze(np.asarray(data))
        if values.ndim == 2:
            values = values[..., None]
        if values.ndim != 3:
            raise ValueError(
                f"NumPy DICOM export requires 2D or 3D data, got {values.shape}"
            )
        image = xr.DataArray(
            values,
            dims=("phase_idx", "read_idx", "slice_idx"),
        )
    if not {"read_idx", "phase_idx"}.issubset(image.dims):
        raise ValueError(
            "DICOM export requires read_idx and phase_idx dimensions for xarray data"
        )

    references = _normalise_reference_datasets(reference_dicoms)
    export_geometry = _geometry_from_xyz(image, geometry_source=geometry)
    if export_geometry is None and geometry is not None:
        export_geometry = _geometry_from_source(geometry, image)
    if export_geometry is None:
        export_geometry = _geometry_from_references(references, image)
    image_comments = None
    if export_geometry is None:
        if not allow_identity_geometry:
            raise ValueError(_missing_geometry_message(geometry, references, image))
        warnings.warn(
            "DICOM export writes a centred 1 mm identity geometry; viewers "
            "place these images at the isocentre, not at the patient position",
            UserWarning,
            stacklevel=2,
        )
        export_geometry = _fallback_geometry(image)
        series_description = _label_identity_geometry(series_description)
        image_comments = _IDENTITY_GEOMETRY_COMMENT

    frames, _, _ = _prepare_frames(image, export_geometry)
    frame_count, slice_count, rows, columns = frames.shape
    if len(export_geometry.image_positions_patient) != slice_count:
        raise ValueError("DICOM geometry slice count does not match image data")
    pixels, slope, intercept, has_padding = _scale_pixel_data(frames)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_source = references[0] if references else None
    study_uid = (
        str(getattr(metadata_source, "StudyInstanceUID", "") or "")
        if metadata_source is not None
        else ""
    ) or generate_uid()
    frame_uid = (
        str(getattr(metadata_source, "FrameOfReferenceUID", "") or "")
        if metadata_source is not None
        else ""
    ) or generate_uid()
    series_uid = generate_uid()
    now = datetime.now()
    spacing_between_slices = export_geometry.slice_spacing

    instance_number = 0
    for frame_index in range(frame_count):
        for slice_index in range(slice_count):
            instance_number += 1
            sop_uid = generate_uid()
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = sop_uid
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
            dataset.SOPClassUID = MRImageStorage
            dataset.SOPInstanceUID = sop_uid
            dataset.StudyInstanceUID = study_uid
            dataset.SeriesInstanceUID = series_uid
            dataset.FrameOfReferenceUID = frame_uid
            dataset.Modality = "MR"
            dataset.ImageType = ["DERIVED", "SECONDARY"]
            dataset.DerivationDescription = (
                f"HyperMRI derived image; geometry={export_geometry.source}"
            )
            dataset.SeriesDescription = series_description
            if image_comments is not None:
                dataset.ImageComments = image_comments
            dataset.SeriesNumber = int(series_number)
            dataset.InstanceNumber = instance_number
            dataset.AcquisitionNumber = frame_index + 1
            dataset.ImagesInAcquisition = slice_count
            if frame_count > 1:
                dataset.NumberOfTemporalPositions = frame_count
                dataset.TemporalPositionIdentifier = frame_index + 1
            dataset.ContentDate = now.strftime("%Y%m%d")
            dataset.ContentTime = now.strftime("%H%M%S.%f")

            if metadata_source is not None:
                for tag in _REFERENCE_TAGS:
                    if hasattr(metadata_source, tag):
                        setattr(dataset, tag, getattr(metadata_source, tag))

            dataset.Rows = rows
            dataset.Columns = columns
            dataset.PixelSpacing = list(export_geometry.pixel_spacing)
            dataset.SliceThickness = float(export_geometry.slice_thickness)
            dataset.SpacingBetweenSlices = spacing_between_slices
            dataset.InPlanePhaseEncodingDirection = (
                "ROW" if export_geometry.row_dim == "read_idx" else "COL"
            )
            dataset.ImageOrientationPatient = list(
                np.asarray(
                    export_geometry.image_orientation_patient, dtype=float
                ).reshape(6)
            )
            dataset.ImagePositionPatient = list(
                np.asarray(
                    export_geometry.image_positions_patient[slice_index], dtype=float
                ).reshape(3)
            )
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 1
            dataset.RescaleSlope = float(slope)
            dataset.RescaleIntercept = float(intercept)
            if unit is not None:
                dataset.RescaleType = unit
            if has_padding:
                dataset.PixelPaddingValue = -32768
            dataset.PixelData = pixels[frame_index, slice_index].tobytes()

            filename = f"{filename_prefix}{instance_number:04d}.dcm"
            dataset.save_as(output_path / filename, enforce_file_format=True)

    logger.info(
        "Exported %d DICOM instances to %s using %s geometry",
        instance_number,
        output_path,
        export_geometry.source,
    )
    return True


__all__ = ["export_to_dicom"]
