"""CSI acquisition bookkeeping, independent of fitted signal magnitudes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

CSI_RAW_CONTRACT = "1"
SCANNER_SPATIAL_METHODS = {"<Bruker:CSI>", "<User:csi_fix>"}
FOURIER_ENCODED_METHODS = {
    "<User:lucaCSI>",
    "<User:lucaCSI2>",
    "<User:lucaCSI3>",
    "<User:lucaCSI4>",
    "<User:geoffCSI>",
    "<User:ssfpCSI>",
}
# Methods that acquire each k-space position as often as the method parameter
# AverageList says (one count per position, in acquisition order). ParaVision
# averages these repeats in 2dseq; adding them instead keeps the counts as a
# k-space density weighting (reconstruction option repeated_positions="sum").
DENSITY_WEIGHTED_METHODS = {"<User:ssfpCSI>"}
# Methods whose phase-encoding plan comes from PVM_EncOrder / PVM_EncSteps0/1.
PVM_ENCODING_METHODS = (
    SCANNER_SPATIAL_METHODS | {"<User:geoffCSI>"} | DENSITY_WEIGHTED_METHODS
)
# How the sorter combines FIDs that share one read/phase position.
REPEATED_POSITION_RULES = {
    "mean": "mean_with_acquisition_count",
    "sum": "sum_with_acquisition_count",
}
# Methods that can store one read/phase position per FID in their method file.
PAIRED_ORDER_METHODS = FOURIER_ENCODED_METHODS - PVM_ENCODING_METHODS
# PVM_EncOrder values (read axis, phase axis) with a verified reconstruction.
PVM_ENCODING_ORDERS = ("LINEAR_ENC LINEAR_ENC", "CENTRIC_ENC CENTRIC_ENC")
SUPPORTED_ENCODINGS = {*PVM_ENCODING_ORDERS, "luca_centric"}


def audit_csi_raw_input(data):
    """Audit origin separately from candidate signal content."""
    if (
        data.attrs.get("domain") == "spatial"
        or data.attrs.get("csi_grid_domain") == "spatial"
    ):
        return False, "spatial_input", {}
    if data.attrs.get("csi_raw_contract") != CSI_RAW_CONTRACT:
        return False, "missing_raw_provenance", {}
    try:
        trace = json.loads(data.attrs["csi_raw_trace"])
    except (KeyError, TypeError, ValueError):
        return False, "invalid_raw_provenance", {}
    if not isinstance(trace, dict):
        return False, "invalid_raw_provenance", {}
    if (
        data.attrs.get("domain") != "kspace"
        or data.attrs.get("csi_grid_domain") != "kspace"
        or trace.get("grid_domain") != "kspace"
        or trace.get("source") != "fid"
        or trace.get("method") not in FOURIER_ENCODED_METHODS
        or trace.get("encoding") not in SUPPORTED_ENCODINGS
    ):
        return False, "unverified_kspace_domain", trace
    # Noise propagation divides the variance of a bin by its count, which is
    # only correct for averaged repeats.
    if trace.get("repeated_positions") != REPEATED_POSITION_RULES["mean"]:
        return False, "summed_repeated_positions", trace
    if data.attrs.get("signal_domain") != "temporal" or not np.iscomplexobj(data.data):
        return False, "requires_complex_temporal_data", trace
    # The current vendor FID loader decodes little-endian int32 pairs. Merely
    # matching the expected number of samples cannot validate another codec.
    if (
        trace.get("declared_storage_format") != "GO_32BIT_SGN_INT"
        or trace.get("declared_byte_order") != "little"
    ):
        return False, "unverified_storage_codec", trace
    if any(
        trace.get(name) != "none"
        for name in (
            "spatial_fft",
            "temporal_fft",
            "window",
            "channel_combination",
            "interpolation",
            "zero_filling",
        )
    ):
        return False, "processed_input", trace
    if "csi_acquisition_count" not in data.coords or "csi_acquired" not in data.coords:
        return False, "missing_acquisition_mask", trace
    counts, acquired = data.csi_acquisition_count, data.csi_acquired
    if (
        set(counts.dims) != {"read_idx", "phase_idx"}
        or set(acquired.dims) != set(counts.dims)
        or not np.issubdtype(counts.dtype, np.integer)
        or acquired.dtype != bool
        or np.any(counts.values < 0)
        or not np.array_equal(
            acquired.transpose(*counts.dims).values, counts.values > 0
        )
    ):
        return False, "invalid_acquisition_mask", trace
    if csi_raw_fingerprint(data) != data.attrs.get("csi_raw_fingerprint"):
        return False, "raw_content_changed", trace
    time = data.coords.get("acquisition_time_s")
    if (
        time is None
        or time.dims != ("acquisition_idx",)
        or time.attrs.get("unit") != "s"
    ):
        return False, "unverified_sampling_times", trace
    dt = np.diff(time.values)
    if (
        not np.all(np.isfinite(dt))
        or not np.all(dt > 0)
        or not np.allclose(dt, dt[0], rtol=1e-7, atol=1e-15)
    ):
        return False, "nonuniform_sampling_times", trace
    return True, "calibration_required", trace


def strip_csi_storage_padding(
    values,
    *,
    samples: int,
    channels: int,
    records: int,
    record_samples: int | None,
):
    """Remove the padding a declared storage layout appends to each record.

    A record is the block of ``channels * samples`` complex samples stored for
    one repetition, slice and encoding position. The vendor adapter declares
    ``record_samples``, the number of complex samples one record occupies in
    the file including padding; this function never infers padding from the
    sample values. Values that already have the unpadded size, an undeclared
    layout (``None``) and a payload whose size does not match the declared
    layout are returned unchanged, so the caller's exact size check decides.

    Parameters
    ----------
    values : array_like
        Complex samples in file order, record after record. Not modified.
    samples : int
        Complex samples per FID and receiver.
    channels : int
        Receivers stored in each record.
    records : int
        Stored records (repetitions x slices x encoding positions).
    record_samples : int or None
        Declared complex samples per stored record including padding, or
        ``None`` when the vendor declares no padded layout.

    Returns
    -------
    numpy.ndarray
        One-dimensional samples without padding, or ``values`` unchanged.
    int
        Removed complex samples per record; zero when nothing was removed.

    Raises
    ------
    ValueError
        If ``record_samples`` is smaller than ``channels * samples``.
    """
    values = np.asarray(values)
    record_size = samples * channels
    if values.size == record_size * records or record_samples is None:
        return values, 0
    if record_samples < record_size:
        raise ValueError(
            f"strip_csi_storage_padding: record_samples={record_samples} is "
            f"smaller than one record of {record_size} complex samples "
            f"({channels} channel(s) x {samples} samples); check the vendor "
            "storage layout."
        )
    if values.size != records * record_samples:
        return values, 0
    unpadded = values.reshape(records, record_samples)[:, :record_size]
    return unpadded.reshape(-1), record_samples - record_size


def sort_csi_encoding_grid(
    payload, read_positions, phase_positions, grid_shape, repeated_positions="mean"
):
    """Scatter explicit integer acquisition positions, retaining occupancy counts.

    ``payload`` has axes (sample, slice, position, repetition, channel).
    Both position arrays must describe every acquired FID; no Cartesian-product
    or orientation guess is made. Repeated positions are averaged
    (``repeated_positions="mean"``) or added (``"sum"``, for density-weighted
    acquisitions); missing bins are NaN, and every channel/repetition/slice
    follows exactly the same map.
    """
    if repeated_positions not in REPEATED_POSITION_RULES:
        raise ValueError(
            f"sort_csi_encoding_grid: repeated_positions={repeated_positions!r} "
            f"is not one of {sorted(REPEATED_POSITION_RULES)}."
        )
    payload = np.asarray(payload)
    if payload.ndim != 5:
        raise ValueError(
            "CSI payload must have sample/slice/position/repetition/channel axes."
        )
    positions = []
    for name, values, size in zip(
        ("read", "phase"), (read_positions, phase_positions), grid_shape, strict=True
    ):
        values = np.asarray(values)
        if (
            values.ndim != 1
            or values.size != payload.shape[2]
            or np.iscomplexobj(values)
            or not np.all(np.isfinite(values))
            or np.any(values != np.floor(values))
            or np.any(values < 0)
            or np.any(values >= size)
        ):
            raise ValueError(
                f"CSI {name} encoding must specify one in-grid integer position per acquired FID."
            )
        positions.append(values.astype(np.int64))
    read, phase = positions
    counts = np.zeros(grid_shape, dtype=np.int64)
    np.add.at(counts, (read, phase), 1)
    # Sum at full precision, then retain the payload's floating/complex dtype.
    sums = np.zeros(
        (*grid_shape, *payload.shape[:2], *payload.shape[3:]),
        dtype=np.result_type(payload.dtype, np.float64),
    )
    divisor = counts[..., None, None, None, None]
    if repeated_positions == "sum":
        divisor = np.minimum(divisor, 1)
    averaged = np.full_like(sums, np.nan)
    # Preserve nonfinite observations for explicit downstream rejection rather
    # than converting them to zeros or emitting arithmetic warnings per point.
    with np.errstate(over="ignore", invalid="ignore"):
        np.add.at(sums, (read, phase), payload.transpose(2, 0, 1, 3, 4))
        np.divide(sums, divisor, out=averaged, where=divisor > 0)
    dtype = payload.dtype if np.issubdtype(payload.dtype, np.inexact) else np.float64
    # Historical canonical intermediate: sample, slice, read, phase, rep, channel.
    return averaged.astype(dtype).transpose(2, 3, 0, 1, 4, 5), counts


def csi_raw_fingerprint(data: xr.DataArray) -> str:
    """Detect stale acquisition metadata after value/coordinate processing.

    This content identifier is not a security signature or physical proof of a
    scanner domain. Transposing axes alone preserves it; slicing, padding,
    filtering, channel combination and altered acquisition masks invalidate it.
    """
    digest = hashlib.sha256()
    for name, array in [("data", data), *sorted(data.coords.items())]:
        dims = sorted(array.dims)
        values = np.asarray(array.transpose(*dims).values)
        if values.dtype.hasobject:
            values = values.astype(str)
        values = np.ascontiguousarray(values)
        digest.update(json.dumps([name, dims, values.shape, values.dtype.str]).encode())
        digest.update(values.tobytes())
    digest.update(str(data.attrs.get("csi_raw_trace", "")).encode())
    return digest.hexdigest()


def attach_csi_raw_provenance(
    data,
    *,
    counts,
    method,
    encoding,
    source,
    requested_source,
    storage_padding_complex=0,
    declared_storage_format="unknown",
    declared_byte_order="unknown",
    repeated_positions="mean",
):
    """Describe a freshly sorted payload; never certify arbitrary named arrays.

    ``declared_storage_format`` and ``declared_byte_order`` are the sample
    codec and byte order the vendor adapter reads from the raw-file
    description; ``"unknown"`` when it declares none. ``repeated_positions``
    names how the sorter combined repeated FIDs (``"mean"`` or ``"sum"``).
    """
    if method in SCANNER_SPATIAL_METHODS:
        domain, evidence = "spatial", "scanner_spatial_grid_method_policy"
    elif (
        method in FOURIER_ENCODED_METHODS
        and source in {"fid", "job"}
        and encoding in SUPPORTED_ENCODINGS
    ):
        domain = "kspace"
        evidence = "raw_payload_and_explicit_phase_encoding_method_policy"
    else:
        domain, evidence = "unknown", "unverified_source_or_method"
    trace = {
        "version": CSI_RAW_CONTRACT,
        "method": method,
        "encoding": encoding,
        "source": source,
        "requested_source": requested_source,
        "grid_domain": domain,
        "domain_evidence": evidence,
        "source_dtype": str(data.dtype),
        "declared_storage_format": declared_storage_format,
        "declared_byte_order": declared_byte_order,
        "spatial_fft": "none",
        "temporal_fft": "none",
        "window": "none",
        "channel_combination": "none",
        "channel_weights": "identity",
        "interpolation": "none",
        "zero_filling": "none",
        "repeated_positions": REPEATED_POSITION_RULES[repeated_positions],
        "applied_cut_off": int(data.attrs["cut_off"]),
        "raw_n_acquisition": int(data.attrs["raw_n_acquisition"]),
        "storage_padding_complex_samples_per_record": int(storage_padding_complex),
    }
    data = data.assign_coords(
        csi_acquisition_count=(("read_idx", "phase_idx"), counts),
        csi_acquired=(("read_idx", "phase_idx"), counts > 0),
    )
    data.attrs.update(
        domain=domain,
        signal_domain="temporal",
        csi_grid_domain=domain,
        csi_domain_evidence=evidence,
        csi_raw_contract=CSI_RAW_CONTRACT,
        csi_raw_trace=json.dumps(trace, sort_keys=True),
    )
    data.coords["csi_acquired"].attrs["description"] = (
        "Acquisition-plan occupancy; independent of measured amplitude, not a noise-selection mask."
    )
    data.coords["csi_acquisition_count"].attrs["description"] = (
        "Number of payload FIDs combined into each read/phase bin (see "
        "csi_raw_trace repeated_positions), before any spatial transform."
    )
    paths = [Path(__file__), Path(__file__).parent / "strategies/bruker/csi.py"]
    data.attrs["csi_sort_source_hash"] = hashlib.sha256(
        b"".join(p.read_bytes() for p in paths)
    ).hexdigest()
    data.attrs["csi_raw_fingerprint"] = csi_raw_fingerprint(data)
    return data


__all__ = ["csi_raw_fingerprint"]
