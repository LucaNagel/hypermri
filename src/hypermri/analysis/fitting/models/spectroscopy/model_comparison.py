"""Single-target, supplied-noise M0/M1 comparison with full-refit calibration.

The plug-in parametric bootstrap is approximate. Numerical success, provisional
weak-component diagnostics, detection and amplitude intervals are distinct.
This module never constrains complex amplitudes or adapts the simulation budget.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import xarray as xr
from scipy.stats import binomtest

from .diagnostics import FIT_STATUSES

DETECTION_VERSION = "1"
DETECTION_STATUSES = (
    "detected",
    "not_detected",
    "inconclusive",
    "fit_failed",
    "not_tested",
)
DETECTION_REASONS = (
    "none",
    "target_skipped",
    "m0_failed",
    "m1_failed",
    "m1_worse_than_m0",
    "null_fit_failed",
    "mc_resolution",
    "nonfinite_statistic",
)
WEAK_REASONS = {
    "low_conditional_snr": 1,
    "bound_hit": 2,
    "ill_conditioned": 4,
    "uncertainty_unavailable": 8,
}


def prepare_detection_options(options):
    """Validate opt-in settings, including edited prepared parameter attrs."""
    from .fid_model import _coerce_finite_scalar

    options = dict(options)
    allowed = {
        "detection_target",
        "detection_sigma",
        "detection_alpha",
        "detection_weak_snr",
        "detection_null_samples",
        "detection_seed",
        "detection_store_null",
        "detection_version",
        "detection_source_hash",
        "detection_source_manifest",
        "detection_runtime",
    }
    unknown = {key for key in options if key.startswith("detection_")} - allowed
    if unknown:
        raise ValueError(f"Unsupported detection settings: {sorted(unknown)}")
    target = options.get("detection_target")
    if target is None:
        if any(
            key.startswith("detection_") for key in options if key != "detection_target"
        ):
            raise ValueError("detection_target is required with detection settings.")
        return options
    if not isinstance(target, str) or list(options["metabs"]).count(target) != 1:
        raise ValueError(
            "detection_target must name exactly one configured metabolite."
        )
    store_null = options.get("detection_store_null", False)
    if not isinstance(store_null, (bool, np.bool_)):
        raise ValueError("detection_store_null must be boolean.")
    options["detection_store_null"] = bool(store_null)
    if options.get("snr_threshold") is not None:
        raise ValueError(
            "Statistical detection requires snr_threshold=None (no legacy filtering)."
        )
    if options.get("use_sse_mag", False):
        raise ValueError(
            "Statistical detection requires the complex RSS objective (use_sse_mag=False)."
        )
    if options.get("amp_solver") == "ridge" or options.get("amp_ridge_lambda", 0) != 0:
        raise ValueError(
            "Statistical detection requires unregularized complex amplitudes."
        )
    if str(options.get("signal_domain", "temporal")).lower() not in {
        "temporal",
        "spectral",
    }:
        raise ValueError("Statistical detection requires temporal or spectral input.")
    for key, default in (
        ("detection_sigma", None),
        ("detection_alpha", 0.05),
        ("detection_weak_snr", 3.0),
    ):
        value = _coerce_finite_scalar(options.get(key, default), parameter_name=key)
        if value <= 0 or (key == "detection_alpha" and value >= 1):
            raise ValueError(
                f"{key} must be positive"
                + (" and less than one." if key == "detection_alpha" else ".")
            )
        options[key] = value
    with np.errstate(over="ignore", under="ignore"):
        variance = np.square(options["detection_sigma"])
    if not np.isfinite(variance) or variance == 0:
        raise ValueError("detection_sigma squared must be finite and positive.")
    for key, minimum in (("detection_null_samples", 1), ("detection_seed", 0)):
        value = options.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= value <= np.iinfo(np.int64).max
        ):
            raise ValueError(
                f"{key} must be an explicit integer >= {minimum} within int64 range."
            )
        options[key] = int(value)
    return options


def detection_schema(options=None):
    """Fixed-size compact outputs; no Monte Carlo trajectories by default."""
    schema = {}
    for name in (
        "rss_m0",
        "rss_m1",
        "rss_m1_initial",
        "test_statistic_initial",
        "test_statistic_raw",
        "detection_p_value",
        "detection_mc_tail_low",
        "detection_mc_tail_high",
        "basis_condition_number",
    ):
        schema[name] = ([], np.float64)
    for name in (
        "detection_status",
        "detection_reason",
        "m0_fit_status",
        "m0_fit_failure_reason",
    ):
        schema[name] = ([], np.int16)
    for name in (
        "detection_n_null",
        "detection_n_valid_null",
        "detection_n_extreme",
        "detection_n_failed_null",
        "detection_n_recovered_null",
        "detection_stream",
    ):
        schema[name] = ([], np.int64)
    schema["m1_recovery_attempted"] = ([], bool)
    schema["m1_recovery_used"] = ([], bool)
    for name, dtype in (
        ("m0_amps", np.complex128),
        ("m0_freqs", float),
        ("m0_t2s", float),
        ("amplitude_std_quadrature_conditional", float),
        ("amplitude_snr_conditional", float),
        ("weak_candidate", bool),
        ("weak_reason", np.int16),
        ("fit_bound_hit", bool),
    ):
        schema[name] = (["metabolite"], dtype)
    schema["calibration_digest"] = (["sha256_byte"], np.uint8)
    if (options or {}).get("detection_store_null", False):
        schema["detection_null_statistic"] = (["null_replicate"], np.float64)
        schema["detection_null_reason"] = (["null_replicate"], np.int16)
        schema["detection_null_recovery_used"] = (["null_replicate"], bool)
    return schema


def source_provenance():
    """Hash actual package sources, including uncommitted edits, once per call."""
    root = Path(__file__).resolve().parents[4]
    manifest = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return {
        "detection_version": DETECTION_VERSION,
        "detection_source_hash": hashlib.sha256(encoded.encode()).hexdigest(),
        "detection_source_manifest": encoded,
        "detection_runtime": json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "xarray": xr.__version__,
            },
            sort_keys=True,
        ),
    }


def _valid(result):
    return FIT_STATUSES[int(result[6])] in {
        "success",
        "zero_signal",
        "skipped",
    } and np.isfinite(result[8])


def _rss_tolerance(data, rss0, rss1):
    return (
        128
        * np.finfo(float).eps
        * max(float(np.vdot(data, data).real), rss0, rss1, np.finfo(float).tiny)
    )


def fit_model_pair(data, maps, options, numerical_fit):
    """Refit M0 and M1; recover a worse/failed M1 using M0 nuisance parameters.

    Keep the target start and every original bound/search width. A recovery
    changes starting guesses only and is applied identically to null replicates.
    Return original M1 RSS too; never clip negative differences.
    """
    target = list(options["metabs"]).index(options["detection_target"])
    m0_maps = [np.array(value, copy=True) for value in maps]
    m0_maps[8][target] = True
    m0 = numerical_fit(
        data, *m0_maps, **{**options, "store_optimization_history": False}
    )
    m1 = numerical_fit(data, *maps, **options)
    initial_rss = m1[8]
    attempted = used = False
    reason = "none"
    if not _valid(m0):
        return m0, m1, initial_rss, attempted, used, "m0_failed"
    if not _valid(m1) or m1[8] > m0[8] + _rss_tolerance(data, m0[8], m1[8]):
        attempted = True
        restart = [np.array(value, copy=True) for value in maps]
        shared = ~np.asarray(m0_maps[8], dtype=bool)
        restart[0][shared] = m0[1][shared]
        restart[1][shared] = m0[2][shared]
        recovered = numerical_fit(data, *restart, **options)
        if _valid(recovered) and (not _valid(m1) or recovered[8] < m1[8]):
            m1, used = recovered, True
    if not _valid(m1):
        reason = "m1_failed"
    elif m1[8] > m0[8] + _rss_tolerance(data, m0[8], m1[8]):
        reason = "m1_worse_than_m0"
    elif not np.isfinite(_statistic(m0[8], m1[8], options["detection_sigma"] ** 2)):
        reason = "nonfinite_statistic"
    return m0, m1, initial_rss, attempted, used, reason


def _statistic(rss0, rss1, variance):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(np.divide(rss0 - rss1, variance))


def conditional_fit_diagnostics(
    amplitudes,
    frequencies,
    t2s,
    skipped,
    frequency_widths,
    t2_widths,
    min_t2s,
    max_t2s,
    frequency_lower_bounds,
    frequency_upper_bounds,
    sigma,
    sampling_time_axis_s,
):
    """Return the shared conditional linear diagnostics for one M1 fit.

    The amplitude standard deviation is conditional on the fitted frequencies
    and T2* values.  It is not a confidence interval.  Frozen parameters are
    not boundary hits.  This function performs no fitting and never changes
    any input array.
    """
    from .fid_model import fid_basefunc

    amplitudes = np.asarray(amplitudes)
    frequencies = np.asarray(frequencies, dtype=float)
    t2s = np.asarray(t2s, dtype=float)
    skipped = np.asarray(skipped, dtype=bool)
    active = ~skipped
    std = np.full(active.shape, np.nan, dtype=float)
    snr = np.full(active.shape, np.nan, dtype=float)
    frequency_bound = np.zeros(active.shape, dtype=bool)
    t2_bound = np.zeros(active.shape, dtype=bool)
    condition = np.nan
    rank = 0

    if np.any(active) and np.all(
        np.isfinite(
            np.concatenate(
                [
                    amplitudes[active].real,
                    amplitudes[active].imag,
                    frequencies[active],
                    t2s[active],
                ]
            )
        )
    ):
        basis = fid_basefunc(
            frequencies[active], t2s[active], np.asarray(sampling_time_axis_s)
        ).T
        try:
            condition = float(np.linalg.cond(basis))
            rank = int(np.linalg.matrix_rank(basis))
            if rank == np.count_nonzero(active) and np.isfinite(condition):
                inverse = np.linalg.pinv(basis)
                std[active] = float(sigma) * np.sqrt(
                    np.sum(np.abs(inverse) ** 2, axis=1)
                )
                snr[active] = np.abs(amplitudes[active]) / std[active]
        except np.linalg.LinAlgError:
            condition = np.nan
            rank = 0

        frequency_widths = np.asarray(frequency_widths, dtype=float)
        frequency_lower_bounds = np.asarray(frequency_lower_bounds, dtype=float)
        frequency_upper_bounds = np.asarray(frequency_upper_bounds, dtype=float)
        frequency_bound = (
            active
            & (frequency_widths > 0)
            & (frequency_upper_bounds > frequency_lower_bounds)
            & (
                np.isclose(frequencies, frequency_lower_bounds, rtol=1e-7, atol=1e-12)
                | np.isclose(frequencies, frequency_upper_bounds, rtol=1e-7, atol=1e-12)
            )
        )
        t2_widths = np.asarray(t2_widths, dtype=float)
        min_t2s = np.asarray(min_t2s, dtype=float)
        max_t2s = np.asarray(max_t2s, dtype=float)
        t2_bound = (
            active
            & (t2_widths > 0)
            & (max_t2s > min_t2s)
            & (
                np.isclose(t2s, min_t2s, rtol=1e-7, atol=1e-12)
                | np.isclose(t2s, max_t2s, rtol=1e-7, atol=1e-12)
            )
        )
    return condition, rank, std, snr, frequency_bound, t2_bound


def _weak_diagnostics(result, maps, options):
    """Conditional linear uncertainty is a diagnostic, not an amplitude CI."""
    active = ~np.asarray(maps[8], dtype=bool)
    std = np.full(active.shape, np.nan)
    snr = std.copy()
    reasons = np.zeros(active.shape, dtype=np.int16)
    bound = np.zeros(active.shape, dtype=bool)
    condition = np.nan
    if _valid(result) and np.any(active):
        condition, rank, std, snr, frequency_bound, t2_bound = (
            conditional_fit_diagnostics(
                result[0],
                result[1],
                result[2],
                maps[8],
                maps[2],
                maps[3],
                maps[4],
                maps[5],
                maps[6],
                maps[7],
                sigma=options["detection_sigma"],
                sampling_time_axis_s=options["sampling_time_axis_s"],
            )
        )
        if (
            rank < np.count_nonzero(active)
            or not np.isfinite(condition)
            or condition > 1e8
        ):
            reasons[active] |= WEAK_REASONS["ill_conditioned"]
        bound = frequency_bound | t2_bound
    reasons[active & ~np.isfinite(std)] |= WEAK_REASONS["uncertainty_unavailable"]
    reasons[active & (snr < options["detection_weak_snr"])] |= WEAK_REASONS[
        "low_conditional_snr"
    ]
    reasons[bound] |= WEAK_REASONS["bound_hit"]
    return condition, std, snr, reasons != 0, reasons, bound


def _calibration_digest(data, maps, m0, options, stream):
    # Arrays capture voxel-specific bounds/freezes/masks; options include noise,
    # source/runtime hashes, seed, algorithm and requested fixed sample budget.
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported calibration setting: {type(value).__name__}")

    keys = (
        "metabs",
        "sampling_time_axis_s",
        "npoints",
        "niter",
        "rep_fitting",
        "zoomfactor",
        "min_peak_separation_hz",
        "amp_solver",
        "amp_ridge_lambda",
        "detection_target",
        "detection_sigma",
        "detection_seed",
        "detection_null_samples",
        "detection_version",
        "detection_source_hash",
        "detection_runtime",
    )
    settings = {key: options.get(key) for key in keys}
    settings["stream"] = stream
    digest = hashlib.sha256(
        json.dumps(settings, default=encode, sort_keys=True).encode()
    )
    for value in (data, *maps, *m0[:3]):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def compare_single_spectrum(data, maps, stream, options, numerical_fit):
    """Fixed-budget full-refit plug-in calibration, local to one spectrum."""
    from .fid_model import _prepare_single_spectrum_data, fid_basefunc

    # Normalize once so observed data, reconstruction and simulated noise all
    # share exactly the temporal samples and the per-quadrature sigma units.
    data, prepared = _prepare_single_spectrum_data(data, options)
    if str(prepared.get("signal_domain", "temporal")).lower() == "spectral":
        data = np.fft.ifft(np.fft.ifftshift(data))
    options = {**prepared, "signal_domain": "temporal"}
    n_metabolites = len(maps[0])
    values = {
        name: np.full(
            tuple(
                {
                    "metabolite": n_metabolites,
                    "sha256_byte": 32,
                    "null_replicate": options["detection_null_samples"],
                }[dim]
                for dim in dims
            ),
            np.nan if np.issubdtype(dtype, np.inexact) else 0,
            dtype=dtype,
        )
        for name, (dims, dtype) in detection_schema(options).items()
    }
    if options.get("detection_store_null", False):
        # -1 distinguishes an unattempted draw from a successful one (code 0).
        values["detection_null_reason"][:] = -1
    values["detection_stream"] = stream
    values["detection_n_extreme"] = -1
    target = list(options["metabs"]).index(options["detection_target"])
    status, reason = "not_tested", "target_skipped"
    if bool(maps[8][target]):
        m1 = numerical_fit(data, *maps, **options)
        m0 = m1
        values.update(rss_m0=m0[8], rss_m1=m1[8], rss_m1_initial=m1[8])
    else:
        m0, m1, initial_rss, attempted, used, reason = fit_model_pair(
            data, maps, options, numerical_fit
        )
        variance = options["detection_sigma"] ** 2
        statistic = _statistic(m0[8], m1[8], variance)
        values.update(
            rss_m0=m0[8],
            rss_m1=m1[8],
            rss_m1_initial=initial_rss,
            test_statistic_initial=_statistic(m0[8], initial_rss, variance),
            test_statistic_raw=statistic,
            m1_recovery_attempted=attempted,
            m1_recovery_used=used,
        )
        status = "inconclusive" if reason == "m1_worse_than_m0" else "fit_failed"
        if reason == "none":
            null_mean = (
                fid_basefunc(m0[1], m0[2], options["sampling_time_axis_s"]).T @ m0[0]
            )
            # A separate deterministic stream per spectrum; no shared RNG and
            # no inner parallel pool. Fix the entire budget before any draws.
            rng = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([options["detection_seed"], stream])
                )
            )
            count = options["detection_null_samples"]
            failed = extreme = recovered_count = 0
            null_options = {**options, "store_optimization_history": False}
            for replicate in range(count):
                noise = options["detection_sigma"] * (
                    rng.standard_normal(data.size) + 1j * rng.standard_normal(data.size)
                )
                b0, b1, _, _, recovery_used, failure = fit_model_pair(
                    null_mean + noise, maps, null_options, numerical_fit
                )
                recovered_count += int(recovery_used)
                if options.get("detection_store_null", False):
                    values["detection_null_reason"][replicate] = (
                        DETECTION_REASONS.index(failure)
                    )
                    values["detection_null_recovery_used"][replicate] = recovery_used
                    values["detection_null_statistic"][replicate] = _statistic(
                        b0[8], b1[8], variance
                    )
                if failure != "none":
                    failed += 1
                    continue
                null_statistic = _statistic(b0[8], b1[8], variance)
                tolerance = 100 * np.finfo(float).eps * abs(statistic)
                extreme += int(null_statistic >= statistic - tolerance)
            values.update(
                detection_n_null=count,
                detection_n_valid_null=count - failed,
                detection_n_failed_null=failed,
                detection_n_extreme=extreme,
                detection_n_recovered_null=recovered_count,
            )
            status, reason = "inconclusive", "null_fit_failed"
            # Never discard failed replicates and use a smaller denominator.
            if failed == 0:
                p_value = (extreme + 1) / (count + 1)
                interval = binomtest(extreme, count).proportion_ci(
                    confidence_level=0.95, method="exact"
                )
                values.update(
                    detection_p_value=p_value,
                    detection_mc_tail_low=interval.low,
                    detection_mc_tail_high=interval.high,
                )
                if 1 / (count + 1) > options["detection_alpha"]:
                    status, reason = "inconclusive", "mc_resolution"
                else:
                    status = (
                        "detected"
                        if p_value <= options["detection_alpha"]
                        else "not_detected"
                    )
                    reason = "none"
    values.update(
        m0_amps=m0[0],
        m0_freqs=m0[1],
        m0_t2s=m0[2],
        m0_fit_status=m0[6],
        m0_fit_failure_reason=m0[7],
        detection_status=DETECTION_STATUSES.index(status),
        detection_reason=DETECTION_REASONS.index(reason),
    )
    weak = _weak_diagnostics(m1, maps, options)
    for name, value in zip(
        (
            "basis_condition_number",
            "amplitude_std_quadrature_conditional",
            "amplitude_snr_conditional",
            "weak_candidate",
            "weak_reason",
            "fit_bound_hit",
        ),
        weak,
        strict=True,
    ):
        values[name] = value
    values["calibration_digest"] = _calibration_digest(data, maps, m0, options, stream)
    # Dask also needs array scalars when there are no loop dimensions.
    return m1 + tuple(
        np.asarray(values[name], dtype=dtype)
        for name, (_, dtype) in detection_schema(options).items()
    )


def digest_to_hex(values):
    values = np.asarray(values, dtype=np.uint8)
    return np.asarray(
        [row.tobytes().hex() for row in values.reshape(-1, 32)], dtype="U64"
    ).reshape(values.shape[:-1])


def annotate_detection_result(result, params):
    """Keep target acceptance and masked amplitudes separate from raw M1."""
    options = params.attrs
    target = result.metabolite == options["detection_target"]
    result["accepted"] = (
        target & (result.detection_status == "detected") & ~result.fit_skipped
    )
    result["accepted"] = result.accepted.transpose(*result.fit_amps.dims)
    result["fit_amps_accepted"] = result.fit_amps.where(result.accepted)
    result.accepted.attrs["description"] = (
        "Single-target M0/M1 acceptance only; other metabolites are untested (False), not declared absent."
    )
    result["m0_skipped"] = result.fit_skipped | target
    result["detection_alpha"] = options["detection_alpha"]
    result["noise_sigma_quadrature"] = (
        params["detection_sigma_map"].broadcast_like(result.rss_m0)
        if "detection_sigma_map" in params
        else options["detection_sigma"]
    )
    result["calibration_id"] = xr.apply_ufunc(
        digest_to_hex,
        result.calibration_digest,
        input_core_dims=[["sha256_byte"]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[np.dtype("U64")],
    )
    result = result.drop_vars("calibration_digest").drop_dims(
        "sha256_byte", errors="ignore"
    )
    for name in params.data_vars:
        result[f"detection_setting_{name}"] = params[name]
    result.weak_reason.attrs["bit_flags"] = json.dumps(WEAK_REASONS, sort_keys=True)
    result.weak_candidate.attrs["description"] = (
        "Provisional diagnostic only; does not select tests or change any model degrees of freedom."
    )
    result.amplitude_std_quadrature_conditional.attrs["definition"] = (
        "sigma * sqrt(diag(pinv(X) @ pinv(X).conj().T)); per quadrature, conditional on fitted frequencies/T2*. Not an amplitude confidence interval."
    )
    result.amplitude_snr_conditional.attrs["definition"] = (
        "abs(complex amplitude) / amplitude_std_quadrature_conditional"
    )
    result.test_statistic_raw.attrs["definition"] = (
        "(rss_m0 - rss_m1) / noise_sigma_quadrature**2; no clipping, not Student t"
    )
    if "detection_null_statistic" in result:
        result.detection_null_statistic.attrs["description"] = (
            "Raw full-refit null statistics, including invalid draws. Use only reason code 0 for successful draws; failed draws still invalidate the test."
        )
        result.detection_null_reason.attrs["codes"] = json.dumps(
            {-1: "not_attempted", **dict(enumerate(DETECTION_REASONS))}, sort_keys=True
        )
    result.detection_p_value.attrs["definition"] = (
        "(n_extreme + 1)/(n_null + 1), upper-tail full-refit M0 plug-in parametric bootstrap; NaN if any null fit failed"
    )
    for suffix in ("low", "high"):
        result[f"detection_mc_tail_{suffix}"].attrs["description"] = (
            "95% Clopper-Pearson interval for the fitted-M0 exceedance probability; Monte Carlo error only, not signal probability or amplitude uncertainty."
        )
    noise_description = json.dumps(
        {
            "model": "white_circular_complex_gaussian",
            "sigma_per_quadrature": result.noise_sigma_quadrature.values.tolist(),
            "sigma_dimensions": list(result.noise_sigma_quadrature.dims),
            "domain": "fitted_temporal_samples",
        },
        sort_keys=True,
    )
    result.attrs.update(
        noise_model=noise_description,
        noise_id=hashlib.sha256(noise_description.encode()).hexdigest(),
        detection_calibration="approximate_plugin_full_refit_fixed_budget",
        detection_multiplicity="uncorrected_single_target_per_spectrum",
        detection_seed_scheme="PCG64(SeedSequence([detection_seed, detection_stream])); optional detection_stream_id map, otherwise C-order spectrum index in input loop-dimension order",
        detection_validation="Implementation/regression checks only; independent FPR/power/coverage validation pending",
    )
    return result
