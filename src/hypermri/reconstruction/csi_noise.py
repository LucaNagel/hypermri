"""Explicit linear CSI reconstruction with factored complex noise covariance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import xarray as xr

from .centred_dft import centered_ifft
from .csi_raw import audit_csi_raw_input

ACQUISITION_NOISE_MODEL = (
    "proper_complex_white_samples_independent_points_spatially_homogeneous"
)


def noise_runtime():
    return json.dumps(
        {"numpy": np.__version__, "scipy": scipy.__version__, "xarray": xr.__version__},
        sort_keys=True,
    )


def noise_source_hash():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / name
        for name in (
            "analysis/spectroscopy/noise.py",
            "analysis/spectroscopy/noise_calibration.py",
            "analysis/spectroscopy/noise_estimation.py",
            "reconstruction/csi_raw.py",
            "reconstruction/centred_dft.py",
            "reconstruction/csi_noise.py",
            "reconstruction/strategies/bruker/csi.py",
            "utils/coordinates/image_shifts.py",
        )
    ]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def dataset_digest(dataset):
    """Content hash for numerical artifacts; the id itself is excluded."""
    digest = hashlib.sha256()
    for name in sorted(dataset.variables):
        value = np.asarray(dataset[name].values)
        if value.dtype.hasobject:
            value = value.astype(str)
        digest.update(
            json.dumps(
                [name, dataset[name].dims, value.shape, value.dtype.str]
            ).encode()
        )
        digest.update(np.ascontiguousarray(value).tobytes())
    attrs = {
        k: v
        for k, v in dataset.attrs.items()
        if k not in {"calibration_id", "noise_id", "reconstruction_id"}
    }
    digest.update(json.dumps(attrs, sort_keys=True).encode())
    return digest.hexdigest()


def _spatial_operator(shape, output_shape, weights):
    operators = []
    for size, output in zip(shape, output_shape, strict=True):
        if (
            isinstance(output, (bool, np.bool_))
            or not isinstance(output, (int, np.integer))
            or output < size
        ):
            raise ValueError(
                "output_shape must contain two integers at least as large as the acquired grid."
            )
        # Align the zero-frequency bins for both even and odd grid lengths.
        insertion = output // 2 - size // 2
        basis = np.zeros((output, size), dtype=complex)
        basis[insertion : insertion + size] = np.eye(size)
        operators.append(centered_ifft(basis, (0,)))
    return np.kron(*operators) * weights.reshape(1, -1)


def reconstruct_csi_with_noise(
    raw_data: xr.DataArray,
    noise: xr.Dataset,
    *,
    spatial_weights=None,
    temporal_operator=None,
    channel_weights=None,
    output_shape=None,
) -> xr.Dataset:
    """Apply explicit linear operators to data and the same estimated covariance.

    ``spatial_weights`` is a complex read/phase grid (window and/or FOV phases),
    ``temporal_operator`` maps input FID samples to output samples, and
    ``channel_weights`` maps input to output channels. Defaults are identity.
    The function does not infer scanner shifts: supply them or use the Bruker
    strategy's ``reconstruct_with_noise``, which obtains its actual FOV phases.
    Missing acquired-grid positions are explicitly zero-filled. Repeated points
    were averaged by the sorter, so their variance is divided by acquisition
    count under the declared independent-acquisition model.

    Cov[(v,t,c),(u,s,d)] = (F F^H)[v,u] (B B^H)[t,s] C[c,d].
    F and B are returned instead of materializing the potentially huge full
    covariance. C includes complex channel weights and all channel correlations.
    Per-quadrature sigma and bootstrap bounds describe marginal variances only;
    they must not be used as a white-noise likelihood after correlated transforms.
    This API retains complex128 precision and does not support nonlinear RSS.
    """
    raw = raw_data.compute() if raw_data.chunks is not None else raw_data
    verified, reason, _ = audit_csi_raw_input(raw)
    if not verified:
        raise ValueError(f"Cannot propagate noise from unverified raw data: {reason}.")
    if (
        noise.attrs.get("source_hash") != noise_source_hash()
        or noise.attrs.get("runtime") != noise_runtime()
        or noise.attrs.get("noise_id") != dataset_digest(noise)
    ):
        raise ValueError("Noise estimate source/content provenance does not match.")
    if noise.attrs.get("raw_fingerprint") != raw.attrs.get("csi_raw_fingerprint"):
        raise ValueError("Noise estimate belongs to different raw observations.")
    if noise.attrs.get("acquisition_noise_model") != ACQUISITION_NOISE_MODEL:
        raise ValueError("Unsupported raw noise covariance structure.")
    if noise.attrs.get("noise_status") != "estimated" or not bool(
        (noise.group_noise_status == "estimated").all()
    ):
        raise ValueError(
            "Noise estimate is insufficient for one or more acquisition groups."
        )
    groups = sorted(
        set(raw.dims) - {"acquisition_idx", "read_idx", "phase_idx", "channel_idx"}
    )
    shape = (raw.sizes["read_idx"], raw.sizes["phase_idx"])
    group_shape = tuple(raw.sizes[d] for d in groups)
    times, channels = raw.sizes["acquisition_idx"], raw.sizes["channel_idx"]
    output_shape = shape if output_shape is None else tuple(output_shape)
    if len(output_shape) != 2:
        raise ValueError("output_shape must have read/phase lengths.")
    spatial = (
        np.ones(shape, dtype=complex)
        if spatial_weights is None
        else np.asarray(spatial_weights, dtype=complex)
    )
    temporal = (
        np.eye(times, dtype=complex)
        if temporal_operator is None
        else np.asarray(temporal_operator, dtype=complex)
    )
    channel = (
        np.eye(channels, dtype=complex)
        if channel_weights is None
        else np.asarray(channel_weights, dtype=complex)
    )
    if spatial.shape != shape or not np.all(np.isfinite(spatial)):
        raise ValueError("spatial_weights must be a finite complex read/phase grid.")
    if (
        temporal.ndim != 2
        or temporal.shape[1] != times
        or temporal.shape[0] < 1
        or not np.all(np.isfinite(temporal))
    ):
        raise ValueError(
            "temporal_operator must be finite with shape (output_sample, input_sample)."
        )
    if (
        channel.ndim != 2
        or channel.shape[1] != channels
        or channel.shape[0] < 1
        or not np.all(np.isfinite(channel))
    ):
        raise ValueError(
            "channel_weights must be finite with shape (output_channel, input_channel)."
        )
    operator = _spatial_operator(shape, output_shape, spatial)
    counts = raw.csi_acquisition_count.transpose(
        "read_idx", "phase_idx"
    ).values.reshape(-1)
    acquired = counts > 0
    values = raw.transpose(
        *groups, "read_idx", "phase_idx", "acquisition_idx", "channel_idx"
    ).values.reshape(*group_shape, -1, times, channels)
    if not np.all(np.isfinite(values[..., acquired, :, :])):
        raise ValueError("Acquired input samples must be finite for reconstruction.")
    values = np.where(acquired[:, None, None], values, 0)
    data = np.einsum("vk,...ktc->...vtc", operator, values, optimize=True)
    data = np.einsum("st,...vtc,oc->...vso", temporal, data, channel, optimize=True)
    data = data.reshape(
        *group_shape, *output_shape, temporal.shape[0], channel.shape[0]
    )
    factor = (
        operator
        * np.sqrt(
            np.divide(
                1.0, counts, out=np.zeros_like(counts, dtype=float), where=acquired
            )
        )[None, :]
    )
    covariance = noise.channel_covariance.transpose(
        *groups, "channel_idx", "channel_partner"
    ).values
    draws = noise.channel_covariance_bootstrap.transpose(
        *groups, "bootstrap", "channel_idx", "channel_partner"
    ).values
    combined = np.einsum(
        "oc,...cd,pd->...op", channel, covariance, channel.conj(), optimize=True
    )
    combined_draws = np.einsum(
        "oc,...bcd,pd->...bop", channel, draws, channel.conj(), optimize=True
    )
    spatial_variance = np.sum(np.abs(factor) ** 2, axis=1).reshape(output_shape)
    temporal_variance = np.sum(np.abs(temporal) ** 2, axis=1)
    variance_scale = spatial_variance[..., None] * temporal_variance
    channel_variance = np.diagonal(combined, axis1=-2, axis2=-1).real
    sigma = np.sqrt(
        np.maximum(
            channel_variance[..., None, None, None, :] * variance_scale[..., None] / 2,
            0,
        )
    )
    # Quantiles after complex channel combination preserve covariance uncertainty.
    draw_sigma = np.sqrt(
        np.maximum(np.diagonal(combined_draws, axis1=-2, axis2=-1).real / 2, 0)
    )
    if any(
        not np.all(np.isfinite(value))
        for value in (data, factor, combined, combined_draws, sigma, draw_sigma)
    ):
        raise ValueError(
            "Numerical overflow or invalid covariance during noise propagation."
        )
    data_dims = groups + ["read_idx", "phase_idx", "acquisition_idx", "channel_idx"]
    coords = {
        d: raw.coords[d].values
        if d in raw.coords and raw.coords[d].dims == (d,)
        else np.arange(raw.sizes[d])
        for d in groups
    }
    coords.update(
        read_idx=np.arange(output_shape[0]),
        phase_idx=np.arange(output_shape[1]),
        acquisition_idx=raw.acquisition_time_s.values
        if temporal_operator is None
        else np.arange(temporal.shape[0]),
        channel_idx=np.arange(channel.shape[0]),
        channel_partner=np.arange(channel.shape[0]),
    )
    result = xr.Dataset(
        {
            "signal": (data_dims, data),
            "noise_sigma_quadrature": (data_dims, sigma),
            "channel_covariance": (
                groups + ["channel_idx", "channel_partner"],
                combined,
            ),
            "channel_covariance_bootstrap": (
                groups + ["bootstrap", "channel_idx", "channel_partner"],
                combined_draws,
            ),
            "spatial_noise_factor": (("voxel", "input_point"), factor),
            "spatial_operator": (("voxel", "input_point"), operator),
            "temporal_operator": (("acquisition_idx", "input_sample"), temporal),
            "channel_weights": (("channel_idx", "input_channel"), channel),
            "spatial_weights": (("input_read", "input_phase"), spatial),
            "input_acquisition_count": ("input_point", counts),
            "input_acquisition_time_s": ("input_sample", raw.acquisition_time_s.values),
        },
        coords=coords,
    )
    for name in ("read_idx", "phase_idx", "slice_idx"):
        if name in raw.coords:
            original = raw.coords[name]
            renamed_dims = [
                {"read_idx": "input_read", "phase_idx": "input_phase"}.get(d, d)
                for d in original.dims
            ]
            result[f"input_coordinate_{name}"] = (
                renamed_dims,
                original.values,
                original.attrs,
            )
    for name, q in (("noise_sigma_low", 0.025), ("noise_sigma_high", 0.975)):
        result[name] = (
            data_dims,
            np.quantile(draw_sigma, q, axis=-2)[..., None, None, None, :]
            * np.sqrt(variance_scale[..., None]),
        )
    result.attrs.update(
        noise_id=noise.attrs["noise_id"],
        raw_fingerprint=raw.attrs["csi_raw_fingerprint"],
        source_hash=noise_source_hash(),
        runtime=noise_runtime(),
        domain="spatial",
        signal_domain="temporal"
        if temporal_operator is None
        else "explicit_linear_transform",
        noise_status="propagated",
        interval_scope=noise.attrs["interval_scope"],
        noise_sigma_convention=noise.attrs["noise_sigma_convention"],
        covariance_definition="E[n(v,t,c) conj(n(u,s,d))] = (F F^H)[v,u] (B B^H)[t,s] C[c,d]; F=spatial_noise_factor, B=temporal_operator, C=channel_covariance; proper complex model",
        point_order="read-major, phase-minor; voxel uses output_shape, input_point uses input_shape",
        input_shape=json.dumps(shape),
        output_shape=json.dumps([int(s) for s in output_shape]),
        fft_normalization="numpy backward 1/output_axis_length; ifftshift before and fftshift after each spatial ifft; odd lengths multiply k-space by exp(-i pi k / n) first (FOV centre at index n / 2)",
        zero_filling="missing points and explicit center-aligned padding contribute zero signal and zero acquired noise",
        channel_combination="explicit complex linear weights; no RSS",
        likelihood_scope="marginal sigma is not a white-noise likelihood; retain the covariance factors for correlated outputs",
    )
    result.signal.attrs.update(
        domain="spatial", signal_domain=result.attrs["signal_domain"]
    )
    result.attrs["reconstruction_id"] = dataset_digest(result)
    return result


__all__ = ["reconstruct_csi_with_noise"]
