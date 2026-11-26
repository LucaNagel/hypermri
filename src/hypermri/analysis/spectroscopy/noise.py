"""Inspect CSI noise-reference candidates without granting statistical acceptance.

This first stage audits the raw-data contract and reports descriptive diagnostics
on an explicit selection block. A calibrated contamination model and independent
selection/estimation validation are required before any sigma is released.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from hypermri.reconstruction.csi_raw import audit_csi_raw_input

CANDIDATE_REASONS = {
    "not_outer": 1,
    "not_acquired": 2,
    "averaged_acquisitions": 4,
    "nonfinite_samples": 8,
    "zero_energy": 16,
    "unverified_input": 32,
    "calibration_required": 64,
    "nonfinite_diagnostics": 128,
}
DIAGNOSTIC_NAMES = (
    "mean_energy",
    "spectral_concentration",
    "lag1_correlation_magnitude",
    "late_to_early_energy_ratio",
    "real_imag_correlation",
    "mean_real",
    "mean_imag",
)


def _sample_block(block, count, name):
    if not isinstance(block, slice) or block.step not in (None, 1):
        raise ValueError(f"{name} must be a contiguous slice with step 1.")
    start = 0 if block.start is None else block.start
    stop = count if block.stop is None else block.stop
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in (start, stop)
    ):
        raise ValueError(f"{name} requires integer slice endpoints.")
    if not 0 <= start < stop <= count or stop - start < 4:
        raise ValueError(f"{name} must select at least four samples within the FID.")
    return np.arange(start, stop, dtype=np.int64)


def _point_diagnostics(signal):
    signal = np.asarray(signal, dtype=np.complex128)
    if not np.all(np.isfinite(signal)):
        return np.full(len(DIAGNOSTIC_NAMES), np.nan)
    energy = np.abs(signal) ** 2
    mean_energy = np.mean(energy)
    if mean_energy == 0:
        return np.array([0, np.nan, np.nan, np.nan, np.nan, 0, 0])
    spectrum_energy = np.abs(np.fft.fft(signal)) ** 2
    lag_denominator = np.sqrt(np.sum(energy[:-1]) * np.sum(energy[1:]))
    lag = (
        abs(np.vdot(signal[:-1], signal[1:])) / lag_denominator
        if lag_denominator > 0
        else np.nan
    )
    half = signal.size // 2
    early = np.mean(energy[:half])
    ratio = np.mean(energy[-half:]) / early if early > 0 else np.nan
    real = signal.real - np.mean(signal.real)
    imag = signal.imag - np.mean(signal.imag)
    denominator = np.linalg.norm(real) * np.linalg.norm(imag)
    correlation = np.dot(real, imag) / denominator if denominator > 0 else np.nan
    return np.array(
        [
            mean_energy,
            np.max(spectrum_energy) / np.sum(spectrum_energy),
            lag,
            ratio,
            correlation,
            np.mean(signal.real),
            np.mean(signal.imag),
        ]
    )


def _channel_correlation(signal):
    """Centered correlation magnitude, including any shared signal content."""
    signal = np.asarray(signal, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = signal - signal.mean(axis=-1, keepdims=True)
        cross = centered @ centered.conj().swapaxes(-1, -2)
        norms = np.linalg.norm(centered, axis=-1)
        denominator = norms[..., :, None] * norms[..., None, :]
        result = np.full(denominator.shape, np.nan)
        np.divide(np.abs(cross), denominator, out=result, where=denominator > 0)
    return result


def inspect_csi_noise_candidates(
    raw_data: xr.DataArray,
    *,
    diagnostic_samples: slice,
    estimation_samples: slice,
    outer_fraction: float = 0.25,
) -> xr.Dataset:
    """Return auditable outer-grid candidates and descriptive block diagnostics.

    Use the freshly sorted Bruker ``fid`` output before reconstruction/filtering.
    The explicit disjoint contiguous blocks plan selection and later estimation;
    separation alone does not establish independence under correlated noise or
    stationary contamination under signal decay. Only diagnostic-block values
    enter the metrics. This function is eager, including for Dask-backed input.

    ``outer_fraction`` is the geometric border fraction of each non-singleton
    read/phase axis (0 < value < 0.5). It does not rank points by measured energy.
    Method/encoding metadata is checked against the repository's reconstruction
    policy; it is not independent scanner-domain or signal-absence validation.

    All results currently have ``noise_status='insufficient'`` and an empty
    selected mask. Probability, contamination fraction/uncertainty and sigma
    remain NaN until a calibrated model and reconstruction propagation exist.
    No diagnostics are treated as noise-dominance probabilities.
    """
    if not isinstance(raw_data, xr.DataArray):
        raise TypeError("raw_data must be an xarray.DataArray.")
    required = {"acquisition_idx", "read_idx", "phase_idx", "channel_idx"}
    if not required.issubset(raw_data.dims):
        raise ValueError(f"Raw CSI data must retain dimensions {sorted(required)}.")
    if (
        isinstance(outer_fraction, (bool, np.bool_))
        or not np.isscalar(outer_fraction)
        or not np.isrealobj(outer_fraction)
        or not 0 < outer_fraction < 0.5
    ):
        raise ValueError(
            "outer_fraction must be a real scalar strictly between 0 and 0.5."
        )
    count = raw_data.sizes["acquisition_idx"]
    diagnostic = _sample_block(diagnostic_samples, count, "diagnostic_samples")
    estimation = _sample_block(estimation_samples, count, "estimation_samples")
    if np.intersect1d(diagnostic, estimation).size:
        raise ValueError("Diagnostic and estimation samples must be disjoint.")
    # Work on a private view; computing is explicit here, never in the fitter.
    raw = raw_data.compute() if raw_data.chunks is not None else raw_data
    verified, reason, trace = audit_csi_raw_input(raw)
    loops = [dim for dim in raw.dims if dim != "acquisition_idx"]
    template = raw.isel(acquisition_idx=0, drop=True).transpose(*loops)
    template = template.drop_vars(
        [
            name
            for name in ("csi_acquired", "csi_acquisition_count")
            if name in template.coords
        ]
    )
    template.attrs = {}
    flags = xr.zeros_like(template, dtype=np.uint16)
    outer = xr.zeros_like(template, dtype=bool)
    for dim in ("read_idx", "phase_idx"):
        size = raw.sizes[dim]
        if size > 1:
            position = np.arange(size)
            border = (
                np.minimum(position, size - 1 - position) + 0.5
            ) / size <= outer_fraction
            # Label by position; the legacy spatial coordinates can be 3-D.
            outer = outer | xr.DataArray(border, dims=[dim])
    flags = flags | xr.where(outer, 0, CANDIDATE_REASONS["not_outer"]).astype(np.uint16)
    acquired = xr.zeros_like(template, dtype=bool)
    counts = xr.zeros_like(template, dtype=np.int64)
    if verified:
        counts = raw.csi_acquisition_count.broadcast_like(template).transpose(*loops)
        acquired = counts > 0
    else:
        flags = flags | CANDIDATE_REASONS["unverified_input"]
    flags = flags | xr.where(acquired, 0, CANDIDATE_REASONS["not_acquired"]).astype(
        np.uint16
    )
    flags = flags | xr.where(
        counts > 1, CANDIDATE_REASONS["averaged_acquisitions"], 0
    ).astype(np.uint16)
    block = raw.isel(acquisition_idx=diagnostic).transpose(*loops, "acquisition_idx")
    finite = np.isfinite(block.values).all(axis=-1)
    metrics = np.full((*template.shape, len(DIAGNOSTIC_NAMES)), np.nan)
    # Unknown/unacquired samples never enter even descriptive noise diagnostics.
    for index in np.ndindex(template.shape):
        if bool(acquired.values[index]):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                metrics[index] = _point_diagnostics(block.values[index])
    flags = flags | xr.DataArray(
        np.where(finite, 0, CANDIDATE_REASONS["nonfinite_samples"]), dims=loops
    ).astype(np.uint16)
    zero = metrics[..., 0] == 0
    failed_metric = finite & acquired.values & ~np.isfinite(metrics[..., 0])
    flags = flags | xr.DataArray(
        np.where(failed_metric, CANDIDATE_REASONS["nonfinite_diagnostics"], 0),
        dims=loops,
    ).astype(np.uint16)
    flags = flags | xr.DataArray(
        np.where(zero, CANDIDATE_REASONS["zero_energy"], 0), dims=loops
    ).astype(np.uint16)
    eligible = (flags == 0) & np.isfinite(metrics[..., 0])
    flags = flags | xr.where(
        eligible, CANDIDATE_REASONS["calibration_required"], 0
    ).astype(np.uint16)
    if verified and not bool(eligible.any()):
        reason = "no_eligible_candidates"
    result = xr.Dataset(
        {
            "outer_candidate": outer,
            "acquired": acquired,
            "acquisition_count": counts,
            "candidate_eligible": eligible,
            "selected_mask": xr.zeros_like(template, dtype=bool),
            "candidate_rejection_reason": flags,
            **{
                name: template.copy(data=metrics[..., i])
                for i, name in enumerate(DIAGNOSTIC_NAMES)
            },
        }
    )
    channel_block = block.where(acquired)
    result["channel_correlation_magnitude"] = xr.apply_ufunc(
        _channel_correlation,
        channel_block,
        input_core_dims=[["channel_idx", "acquisition_idx"]],
        output_core_dims=[["channel_idx", "channel_partner"]],
    ).assign_coords(channel_partner=raw.channel_idx.values)
    result.channel_correlation_magnitude.attrs = {
        "definition": "abs(centered complex cross-product)/product of channel norms on diagnostic samples; includes shared signal, not a noise covariance estimate"
    }
    for name in (
        "p_noise_dominated",
        "signal_fraction",
        "signal_fraction_low",
        "signal_fraction_high",
    ):
        result[name] = xr.full_like(template, np.nan, dtype=float)
    group = template.isel(read_idx=0, phase_idx=0, drop=True)
    result["noise_sigma_quadrature"] = xr.full_like(group, np.nan, dtype=float)
    result["diagnostic_sample_index"] = xr.DataArray(
        diagnostic, dims="diagnostic_sample"
    )
    result["estimation_sample_index"] = xr.DataArray(
        estimation, dims="estimation_sample"
    )
    result.candidate_rejection_reason.attrs["bit_flags"] = json.dumps(
        CANDIDATE_REASONS, sort_keys=True
    )
    result.mean_energy.attrs["definition"] = (
        "mean(abs(z)**2), includes signal and noise; not 2*sigma**2 without noise-dominance validation"
    )
    result.spectral_concentration.attrs["definition"] = (
        "max(abs(fft(z))**2)/sum(abs(fft(z))**2), unwindowed diagnostic block"
    )
    result.lag1_correlation_magnitude.attrs["definition"] = (
        "abs(sum(conj(z[:-1])*z[1:]))/sqrt(sum(abs(z[:-1])**2)*sum(abs(z[1:])**2)); uncentered, not Welch coherence or a probability"
    )
    result.late_to_early_energy_ratio.attrs["definition"] = (
        "mean energy of final floor(n/2) diagnostic samples / first floor(n/2); no exponential-decay assumption"
    )
    result.real_imag_correlation.attrs["definition"] = (
        "Centered Pearson correlation of real and imaginary diagnostic samples; descriptive only"
    )
    result.attrs.update(
        noise_status="insufficient",
        noise_reason=reason,
        candidate_diagnostics_version="1",
        raw_input_verified=bool(verified),
        domain_verification="method_policy_and_acquisition_contract_only",
        split_independence="unverified",
        calibration_status="not_available",
        outer_fraction=float(outer_fraction),
        noise_sigma_convention="SD per real/imaginary quadrature; no estimate released",
        raw_fingerprint=raw.attrs.get("csi_raw_fingerprint", "unavailable"),
        raw_trace=json.dumps(trace, sort_keys=True),
        sort_source_hash=raw.attrs.get("csi_sort_source_hash", "unavailable"),
        diagnostics_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    return result


__all__ = ["inspect_csi_noise_candidates"]
