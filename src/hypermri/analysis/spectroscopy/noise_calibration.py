"""Reference-population calibration for CSI noise-dominance selection.

Probabilities refer to an explicitly supplied population and a specified maximum
signal-energy fraction. They are not universal probabilities of signal absence.
Training, histogram calibration and validation use separate realizations. The
selection rule is frozen before validation; failures are returned, not tuned away.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta, binomtest

from hypermri.reconstruction.csi_noise import (
    dataset_digest,
    noise_runtime,
    noise_source_hash,
)

from .noise import _sample_block

FEATURES = (
    "spectral_concentration",
    "lag1",
    "energy_change",
    "quadrature_correlation",
    "dc_fraction",
)
CALIBRATION_VERSION = "1"


def _positive_integer(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def make_csi_noise_reference(
    signals, time_s, channel_covariance, *, seed, population_id
):
    """Generate independent proper, temporally white noise around supplied signals.

    ``signals`` is (case, sample, channel); its exact population/mixture is chosen
    by the caller, not inferred from a local metabolite model. Channel covariance
    is E[n n^H], so a single-channel per-quadrature sigma gives C=2*sigma**2.
    No signal parameters are generated or constrained by this function.
    """
    seed = _positive_integer(seed, "seed", 0)
    signals = np.asarray(signals, dtype=np.complex128)
    time = np.asarray(time_s, dtype=float)
    covariance = np.asarray(channel_covariance, dtype=np.complex128)
    if signals.ndim != 3 or signals.shape[1] != time.size or time.size < 8:
        raise ValueError(
            "signals must have case/sample/channel axes matching at least eight times."
        )
    if not isinstance(population_id, str) or not population_id.strip():
        raise ValueError("An explicit reference population_id is required.")
    if (
        not np.all(np.isfinite(signals))
        or not np.all(np.isfinite(time))
        or not np.all(np.diff(time) > 0)
    ):
        raise ValueError("Reference signals and increasing times must be finite.")
    if (
        covariance.shape != (signals.shape[2], signals.shape[2])
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.conj().T, rtol=1e-12, atol=0)
    ):
        raise ValueError(
            "channel_covariance must be finite and Hermitian with one row per channel."
        )
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("channel_covariance must be positive definite.") from exc
    rng = np.random.Generator(np.random.PCG64(seed))
    white = (
        rng.standard_normal(signals.shape) + 1j * rng.standard_normal(signals.shape)
    ) / np.sqrt(2)
    noise = white @ factor.T
    result = xr.Dataset(
        {
            "observed": (("case", "acquisition_idx", "channel_idx"), signals + noise),
            "signal": (("case", "acquisition_idx", "channel_idx"), signals),
            "channel_covariance": (("channel_idx", "channel_partner"), covariance),
        },
        coords={
            "case": np.arange(signals.shape[0]),
            "acquisition_idx": time,
            "channel_idx": np.arange(signals.shape[2]),
            "channel_partner": np.arange(signals.shape[2]),
        },
        attrs={
            "population_id": population_id,
            "seed": seed,
            "noise_structure": "proper_complex_temporally_independent",
            "reference_kind": "known_signal_plus_simulated_noise",
        },
    )
    return result


def noise_features(samples):
    """Scale-invariant features of (..., sample, channel), using every channel."""
    x = np.asarray(samples, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        energy = np.abs(x) ** 2
        total = energy.sum(axis=-2)
        spectral = np.abs(np.fft.fft(x, axis=-2)) ** 2
        concentration = spectral.max(axis=-2) / spectral.sum(axis=-2)
        lag = np.abs((x[..., :-1, :].conj() * x[..., 1:, :]).sum(axis=-2)) / np.sqrt(
            energy[..., :-1, :].sum(axis=-2) * energy[..., 1:, :].sum(axis=-2)
        )
        half = x.shape[-2] // 2
        change = np.abs(
            np.log(
                energy[..., -half:, :].mean(axis=-2)
                / energy[..., :half, :].mean(axis=-2)
            )
        )
        centered = x - x.mean(axis=-2, keepdims=True)
        denominator = np.sqrt(
            (centered.real**2).sum(axis=-2) * (centered.imag**2).sum(axis=-2)
        )
        quadrature = np.abs((centered.real * centered.imag).sum(axis=-2)) / denominator
        dc = np.abs(x.sum(axis=-2)) ** 2 / (x.shape[-2] * total)
        # Max across channels predicts the worst-channel contamination event.
        return np.stack(
            [v.max(axis=-1) for v in (concentration, lag, change, quadrature, dc)],
            axis=-1,
        )


def _reference_arrays(reference, diagnostic, estimation):
    if (
        reference.attrs.get("noise_structure")
        != "proper_complex_temporally_independent"
    ):
        raise ValueError(
            "Reference must explicitly specify proper, temporally independent noise."
        )
    observed = reference.observed.transpose(
        "case", "acquisition_idx", "channel_idx"
    ).values
    signal = reference.signal.transpose("case", "acquisition_idx", "channel_idx").values
    covariance = reference.channel_covariance.transpose(
        "channel_idx", "channel_partner"
    ).values
    if (
        observed.shape != signal.shape
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(signal))
    ):
        raise ValueError(
            "Reference observations and known signals must be finite with equal shapes."
        )
    if (
        covariance.shape != (observed.shape[-1],) * 2
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.conj().T)
        or np.linalg.eigvalsh(covariance).min() <= 0
    ):
        raise ValueError(
            "Known reference channel covariance must be positive definite Hermitian."
        )
    power = np.diag(covariance).real
    fractions = []
    for indices in (diagnostic, estimation):
        signal_power = (np.abs(signal[:, indices]) ** 2).mean(axis=1)
        fractions.append((signal_power / (signal_power + power)).max(axis=-1))
    truth = np.maximum(*fractions)
    features = noise_features(observed[:, diagnostic])
    if not np.all(np.isfinite(features)):
        raise ValueError("Reference diagnostic features must be finite.")
    hashes = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in observed
    ]
    if len(set(hashes)) != len(hashes):
        raise ValueError(
            "Reference cases must contain independent, distinct realizations."
        )
    return features, truth, observed[:, estimation], covariance, set(hashes)


def _bins(features, model):
    standardized = (features - model.feature_center.values) / model.feature_scale.values
    score = expit(
        standardized @ model.classifier_weight.values
        + float(model.classifier_intercept)
    )
    # This score only assigns a calibration bin; it is never reported as p_noise.
    return np.minimum(
        (score * model.sizes["probability_bin"]).astype(int),
        model.sizes["probability_bin"] - 1,
    )


def _in_envelope(features, model):
    return (
        np.isfinite(features).all(axis=-1)
        & (features >= model.feature_min.values).all(axis=-1)
        & (features <= model.feature_max.values).all(axis=-1)
    )


def predict_noise_dominance(samples, model):
    """Report calibrated bin probabilities and predictive contamination bounds."""
    features = noise_features(samples)
    finite = np.isfinite(features).all(axis=-1)
    safe = np.where(np.isfinite(features), features, model.feature_center.values)
    bins = _bins(safe, model)
    supported = finite & _in_envelope(safe, model)
    supported &= model.calibration_count.values[bins] >= model.attrs["min_bin_cases"]
    names = (
        "probability",
        "probability_low",
        "probability_high",
        "fraction_median",
        "fraction_low",
        "fraction_high",
    )
    predictions = {
        name: np.where(supported, model[name].values[bins], np.nan) for name in names
    }
    predictions["supported"] = supported
    predictions["selected"] = supported & (
        predictions["probability_low"] >= model.attrs["minimum_probability"]
    )
    return predictions


def calibrate_csi_noise_selection(
    training,
    calibration,
    validation,
    *,
    diagnostic_samples,
    estimation_samples,
    contamination_tolerance,
    minimum_probability=0.95,
    maximum_selection_error=0.05,
    max_relative_variance_bias=0.1,
    min_validation_selected=60,
    min_bin_cases=30,
    probability_bins=5,
):
    """Train a score, calibrate probabilities and validate a fixed selection rule.

    The label is: signal-energy fraction <= tolerance in BOTH specified blocks
    and EVERY channel. Probabilities are Beta(1,1)-smoothed class frequencies in
    fixed score bins. Bounds describe finite reference calibration uncertainty;
    fraction bounds are predictive bin quantiles, not parameter confidence bounds.
    Separate holdout checks assess probability agreement, selected contamination
    rate and variance bias on the untouched estimation block. No validation-set
    tuning, weak-peak model assumptions or changes to the metabolite fitter occur.
    """
    for name, value in (
        ("contamination_tolerance", contamination_tolerance),
        ("minimum_probability", minimum_probability),
        ("maximum_selection_error", maximum_selection_error),
        ("max_relative_variance_bias", max_relative_variance_bias),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isscalar(value)
            or not np.isrealobj(value)
            or not 0 < value < 1
        ):
            raise ValueError(f"{name} must lie strictly between zero and one.")
    for name, value in (
        ("min_validation_selected", min_validation_selected),
        ("min_bin_cases", min_bin_cases),
        ("probability_bins", probability_bins),
    ):
        _positive_integer(value, name, 2)
    references = (training, calibration, validation)
    population = training.attrs.get("population_id")
    if not population or any(
        r.attrs.get("population_id") != population for r in references
    ):
        raise ValueError("All references must declare the same explicit population_id.")
    seeds = [r.attrs.get("seed") for r in references]
    if any(s is None for s in seeds) or len(set(seeds)) != 3:
        raise ValueError(
            "Training, calibration and validation require distinct explicit seeds."
        )
    time = training.acquisition_idx.values
    if any(
        not np.array_equal(r.acquisition_idx.values, time)
        or not np.array_equal(r.channel_idx.values, training.channel_idx.values)
        for r in references
    ):
        raise ValueError("Reference times and channel coordinates must match exactly.")
    dt = np.diff(time)
    if (
        not np.all(np.isfinite(time))
        or not np.all(dt > 0)
        or not np.allclose(dt, dt[0], rtol=1e-7, atol=1e-15)
    ):
        raise ValueError("Reference times must be finite, increasing and uniform.")
    diagnostic = _sample_block(diagnostic_samples, len(time), "diagnostic_samples")
    estimation = _sample_block(estimation_samples, len(time), "estimation_samples")
    if np.intersect1d(diagnostic, estimation).size:
        raise ValueError("Diagnostic and estimation blocks must be disjoint.")
    arrays = [_reference_arrays(r, diagnostic, estimation) for r in references]
    if any(not np.array_equal(a[3], arrays[0][3]) for a in arrays[1:]):
        raise ValueError(
            "Reference partitions must use the same known channel covariance."
        )
    if any(arrays[i][4] & arrays[j][4] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Reference realizations overlap between partitions.")
    x, truth, *_ = arrays[0]
    labels = (truth <= contamination_tolerance).astype(float)
    if np.unique(labels).size != 2:
        raise ValueError("Training needs noise-dominated and contaminated examples.")
    center = np.median(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-6)
    design = np.column_stack([np.ones(len(x)), (x - center) / scale])

    def loss(weights):
        logits = design @ weights
        penalty = 1e-3
        value = (
            np.mean(np.logaddexp(0, logits) - labels * logits)
            + penalty * np.dot(weights[1:], weights[1:]) / 2
        )
        grad = design.T @ (expit(logits) - labels) / len(x)
        grad[1:] += penalty * weights[1:]
        return value, grad

    optimum = minimize(
        loss,
        np.zeros(design.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 1000},
    )
    if not optimum.success:
        raise ValueError(f"Noise classifier training failed: {optimum.message}")
    # Support envelope is fixed from training, with one training SD margin.
    model = xr.Dataset(
        {
            "feature_center": ("feature", center),
            "feature_scale": ("feature", scale),
            "feature_min": ("feature", x.min(axis=0) - scale),
            "feature_max": ("feature", x.max(axis=0) + scale),
            "classifier_weight": ("feature", optimum.x[1:]),
            "classifier_intercept": optimum.x[0],
            "diagnostic_sample_index": ("diagnostic_sample", diagnostic),
            "estimation_sample_index": ("estimation_sample", estimation),
            "reference_time_s": ("reference_sample", time),
        },
        coords={
            "feature": list(FEATURES),
            "probability_bin": np.arange(probability_bins),
            "channel_idx": training.channel_idx.values,
        },
        attrs={
            "version": CALIBRATION_VERSION,
            "population_id": population,
            "noise_structure": "proper_complex_temporally_independent",
            "contamination_tolerance": float(contamination_tolerance),
            "minimum_probability": float(minimum_probability),
            "maximum_selection_error": float(maximum_selection_error),
            "max_relative_variance_bias": float(max_relative_variance_bias),
            "min_validation_selected": int(min_validation_selected),
            "min_bin_cases": int(min_bin_cases),
            "seeds": json.dumps([int(s) for s in seeds]),
            "source_hash": noise_source_hash(),
            "runtime": noise_runtime(),
            "reference_ids": json.dumps([dataset_digest(r) for r in references]),
            "selection_rule": "supported feature envelope and calibrated probability lower 95% bound >= minimum_probability",
            "probability_model": "Beta(1,1) binomial class probability within fixed logistic-score bins; population-relative",
        },
    )
    xcal, fcal, *_ = arrays[1]
    in_envelope = _in_envelope(xcal, model)
    xcal, fcal = xcal[in_envelope], fcal[in_envelope]
    bins = _bins(xcal, model)
    count = np.bincount(bins, minlength=probability_bins)
    good = np.bincount(
        bins, weights=(fcal <= contamination_tolerance), minlength=probability_bins
    )
    model["calibration_count"] = ("probability_bin", count)
    model["calibration_good"] = ("probability_bin", good.astype(int))
    model["probability"] = ("probability_bin", (good + 1) / (count + 2))
    for name, quantile in (("probability_low", 0.025), ("probability_high", 0.975)):
        model[name] = (
            "probability_bin",
            beta.ppf(quantile, good + 1, count - good + 1),
        )
    for name, quantile in (
        ("fraction_low", 0.025),
        ("fraction_median", 0.5),
        ("fraction_high", 0.975),
    ):
        model[name] = (
            "probability_bin",
            [
                np.quantile(fcal[bins == b], quantile) if count[b] else np.nan
                for b in range(probability_bins)
            ],
        )
    xv, fv, yv, cv, _ = arrays[2]
    predictions = predict_noise_dominance(
        validation.observed.transpose("case", "acquisition_idx", "channel_idx").values[
            :, diagnostic
        ],
        model,
    )
    calibration_observed = calibration.observed.transpose(
        "case", "acquisition_idx", "channel_idx"
    ).values
    calibration_selected = predict_noise_dominance(
        calibration_observed[:, diagnostic], model
    )["selected"]
    # Retain the population's selected block-energy distribution, including the
    # effect of diagnosis-based selection, for a cross-block stationarity check.
    model["selected_reference_block_energy"] = (
        ("selected_reference_case", "block", "channel_idx"),
        np.stack(
            [
                (
                    np.abs(calibration_observed[calibration_selected][:, indices]) ** 2
                ).mean(axis=1)
                for indices in (diagnostic, estimation)
            ],
            axis=1,
        ),
    )
    selected = predictions["selected"]
    selected_count = int(selected.sum())
    bad_count = int(np.count_nonzero(selected & (fv > contamination_tolerance)))
    error_upper = (
        binomtest(bad_count, selected_count).proportion_ci(0.95, method="exact").high
        if selected_count
        else 1.0
    )
    vbins = _bins(xv, model)
    supported = predictions["supported"]
    bin_count = np.bincount(vbins[supported], minlength=probability_bins)
    bin_good = np.bincount(
        vbins[supported],
        weights=(fv[supported] <= contamination_tolerance),
        minlength=probability_bins,
    )
    agreement = bool(np.all(bin_count[count >= min_bin_cases] >= min_bin_cases))
    # Compare reference-calibration and holdout bin intervals, Bonferroni sized.
    for b in range(probability_bins):
        if count[b] >= min_bin_cases and bin_count[b] >= min_bin_cases:
            a = binomtest(int(good[b]), int(count[b])).proportion_ci(
                1 - 0.05 / probability_bins, method="exact"
            )
            v = binomtest(int(bin_good[b]), int(bin_count[b])).proportion_ci(
                1 - 0.05 / probability_bins, method="exact"
            )
            agreement &= max(a.low, v.low) <= min(a.high, v.high)
    relative_bias = np.full(cv.shape[0], np.nan)
    fraction_covered = (fv >= predictions["fraction_low"]) & (
        fv <= predictions["fraction_high"]
    )
    fraction_coverage_low = (
        binomtest(
            int(np.count_nonzero(supported & fraction_covered)), int(supported.sum())
        )
        .proportion_ci(0.95, method="exact")
        .low
        if supported.any()
        else 0.0
    )
    if selected_count:
        relative_bias = (np.abs(yv[selected]) ** 2).mean(axis=(0, 1)) / np.diag(
            cv
        ).real - 1
    ready = (
        selected_count >= min_validation_selected
        and error_upper <= maximum_selection_error
        and agreement
        and fraction_coverage_low >= 0.9
        and np.all(np.abs(relative_bias) <= max_relative_variance_bias)
    )
    model["validation_count"] = ("probability_bin", bin_count)
    model["validation_good"] = ("probability_bin", bin_good.astype(int))
    model["validation_relative_variance_bias"] = ("channel_idx", relative_bias)
    model.attrs.update(
        calibration_status="validated" if ready else "insufficient",
        validation_selected=selected_count,
        validation_selected_contaminated=bad_count,
        validation_selection_error_upper=float(error_upper),
        validation_probability_agreement=bool(agreement),
        validation_fraction_coverage_lower=float(fraction_coverage_low),
        validation_scope="supplied population; held-out selection purity and estimation-block variance bias; no universal domain/FPR guarantee",
    )
    model.attrs["calibration_id"] = dataset_digest(model)
    return model


def validate_calibration(model):
    if model.attrs.get("version") != CALIBRATION_VERSION:
        return "calibration_version_mismatch"
    if model.attrs.get("runtime") != noise_runtime():
        return "calibration_runtime_mismatch"
    if model.attrs.get("source_hash") != noise_source_hash():
        return "calibration_source_mismatch"
    if model.attrs.get("calibration_id") != dataset_digest(model):
        return "calibration_content_changed"
    if model.attrs.get("calibration_status") != "validated":
        return "calibration_validation_failed"
    return None


__all__ = ["make_csi_noise_reference", "calibrate_csi_noise_selection"]
