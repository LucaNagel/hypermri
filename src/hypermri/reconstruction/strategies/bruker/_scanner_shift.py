"""Bounded scanner FOV-phase convention for the validated FLASH reference geometry.

This private adapter validates metadata without loading samples. It returns the
explicit read/phase/slice phase-ramp vector for the existing shift operator;
it does not reproduce filtering or other ParaVision processing.
"""

from collections.abc import Mapping

import numpy as np

from hypermri.utils.coordinates.image_shifts import determine_FOV_shift

_CENTERING_ATOL_VOX = 1e-9


def _invalid(field):
    return ValueError(
        f"Bruker scanner FOV shift: unsupported or missing {field}. "
        "Use fov_shift_source='acquisition' for independent raw reconstruction; "
        "scanner mode requires the validated FLASH metadata (stock FLASH, 3D, Head_Supine coronal)."
    )


def _finite_values(value, size, field):
    """Read a fixed number of finite real metadata values without fallbacks."""
    values = np.asarray(value)
    if values.dtype.kind not in "iuf" or values.size != size:
        raise _invalid(field)
    values = values.astype(np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise _invalid(field)
    return values


def _scanner_fov_shift(handler):
    """Resolve the empirically validated 3D FLASH vector and its provenance once."""
    # This is the observed scanner Method value, not another alias mapping.
    # Custom FLASH methods need independent evidence for their centering.
    if getattr(handler, "type", None) != "<Bruker:FLASH>":
        raise _invalid("stock <Bruker:FLASH> sequence")
    for name, expected in (
        ("spatial_dimension", "3D"),
        ("patient_pos", "Head_Supine"),
        ("slice_orientation", "coronal"),
        ("read_orientation", "H_F"),
    ):
        values = np.asarray(getattr(handler, name, None)).reshape(-1)
        if values.size != 1 or values[0] != expected:
            raise _invalid(f"{name}={expected}")

    method = getattr(handler, "method", {})
    if not isinstance(method, Mapping):
        raise _invalid("method parameter mapping")
    for key, attribute in (
        ("PVM_NSPacks", "n_slice_packs"),
        ("PVM_SPackArrNSlices", "slices_per_pack"),
    ):
        # In this 3D acquisition NSlices counts excited slabs, not partitions.
        declared = _finite_values(method.get(key), 1, key)
        effective = _finite_values(getattr(handler, attribute, None), 1, key)
        if declared[0] != 1 or effective[0] != 1:
            raise _invalid(f"single slab/package {key}")

    matrix = _finite_values(getattr(handler, "matrix", None), 3, "matrix")
    if np.any(matrix <= 0) or np.any(matrix != np.floor(matrix)) or matrix[2] % 2:
        raise _invalid("positive integer matrix with even slice partitions")
    encoding = _finite_values(getattr(handler, "enc_matrix", None), 3, "PVM_EncMatrix")
    if not np.array_equal(encoding, matrix):
        raise _invalid("PVM_EncMatrix matching the full matrix")
    resolution = _finite_values(
        getattr(handler, "resolution_mm", None), 3, "resolution_mm"
    )
    if np.any(resolution <= 0):
        raise _invalid("positive resolution_mm")

    for key, attribute, in_plane in (
        ("PVM_SPackArrReadOffset", "read_offset_mm", True),
        ("PVM_SPackArrPhase1Offset", "phase_offset_mm", True),
        ("PVM_SPackArrSliceOffset", "slice_offset_mm", False),
    ):
        declared = _finite_values(method.get(key), 1, key)[0]
        effective = _finite_values(getattr(handler, attribute, None), 1, key)[0]
        if effective != declared or (in_plane and declared != 0):
            raise _invalid(
                f"consistent zero in-plane offsets ({key})" if in_plane else key
            )

    source = getattr(handler, "reco_parameter_source", None)
    if source != "pdata/1/reco":
        raise _invalid("reco_parameter_source='pdata/1/reco'")
    reco = getattr(handler, "reco", {})
    if not isinstance(reco, Mapping):
        raise _invalid("reco parameter mapping with RECO_rotate")
    rotation = np.asarray(reco.get("RECO_rotate"))
    if rotation.shape != (3, 1):
        raise _invalid("RECO_rotate shape (3, 1)")
    rotation = _finite_values(rotation, 3, "RECO_rotate")
    if not np.array_equal(rotation[:2], [0.5, 0.5]):
        raise _invalid("RECO_rotate in-plane baseline 0.5")

    acquisition, _ = determine_FOV_shift(handler)
    declared_slice = (rotation[2] - 0.5) * matrix[2]
    if not np.isclose(
        acquisition[2], -(declared_slice + 0.5), rtol=0, atol=_CENTERING_ATOL_VOX
    ):
        raise _invalid("RECO_rotate/acquisition slice centering relation")
    # A comparison with the scanner image found the slices half a slice below the acquisition ramp of
    # the plain forward slice transform.  The slice-encoding transform now
    # centres the slices at (n - 1) / 2 itself, which is that half-slice
    # translation exactly, so the scanner vector equals the acquisition one.
    shift = (0.0, 0.0, float(-(declared_slice + 0.5)))
    return shift, {
        "fov_shift_source": "scanner",
        "bruker_reco_parameter_source": source,
        "bruker_reco_rotate": tuple(float(value) for value in rotation),
    }
