"""Estimate spectroscopy noise from validated or explicitly selected samples.

The module contains the calibrated split-block CSI estimator and the lightweight
per-quadrature estimator used by the ordinary spectroscopy fitting workflow.
All estimators operate on labelled complex :class:`xarray.DataArray` inputs and
report noise in temporal-sample amplitude units; :func:`estimate_noise` also
accepts real-valued magnitude samples and then returns the Rayleigh scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from hypermri.reconstruction.csi_noise import (
    ACQUISITION_NOISE_MODEL,
    dataset_digest,
    noise_source_hash,
)

from .noise import inspect_csi_noise_candidates
from .noise_calibration import (
    _positive_integer,
    predict_noise_dominance,
    validate_calibration,
)
from .signal_decay import (
    _band_limits,
    _band_selection,
    _repetition_block,
    spectral_band_power,
)


def _structure_statistic(samples):
    """Omnibus diagnostic for departures from the declared white/proper model.

    Scale by channel RMS, without decorrelation: the reference simulation keeps
    channel covariance. Include DC, all available lags up to eight, quadrature
    impropriety, block energy change and adjacent selected-point dependence.
    This diagnostic can reject a model; passing it cannot establish independence.
    """
    x = samples / np.sqrt(np.mean(np.abs(samples) ** 2, axis=(0, 1)))
    points, times, _ = x.shape
    flat = x.reshape(-1, x.shape[-1])
    stats = [np.abs(flat.mean(axis=0)).max() * np.sqrt(points * times)]
    stats.append(np.abs(flat.T @ flat / len(flat)).max() * np.sqrt(len(flat)))
    for lag in range(1, min(8, times - 1) + 1):
        a, b = x[:, :-lag].reshape(-1, x.shape[-1]), x[:, lag:].reshape(-1, x.shape[-1])
        stats.append(np.abs(a.T @ b.conj() / len(a)).max() * np.sqrt(len(a)))
    half = times // 2
    energy = np.abs(x) ** 2
    stats.append(
        np.abs(
            energy[:, :half].mean(axis=(0, 1)) - energy[:, -half:].mean(axis=(0, 1))
        ).max()
        * np.sqrt(points * half / 2)
    )
    if points > 1:
        a, b = x[:-1].reshape(-1, x.shape[-1]), x[1:].reshape(-1, x.shape[-1])
        stats.append(np.abs(a.T @ b.conj() / len(a)).max() * np.sqrt(len(a)))
    return float(np.max(stats))


def _estimate_group(samples, *, rng, bootstrap_samples, structure_samples):
    # Never center/subtract a fitted mean: that could remove real contamination.
    point_covariance = (
        np.einsum("ptc,ptd->pcd", samples, samples.conj()) / samples.shape[1]
    )
    covariance = point_covariance.mean(axis=0)
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return None
    statistic = _structure_statistic(samples)
    exceed = 0
    for _ in range(structure_samples):
        white = (
            rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
        ) / np.sqrt(2)
        exceed += _structure_statistic(white @ factor.T) >= statistic
    indices = rng.integers(samples.shape[0], size=(bootstrap_samples, samples.shape[0]))
    bootstrap = point_covariance[indices].mean(axis=1)
    return covariance, bootstrap, statistic, (1 + exceed) / (1 + structure_samples)


def _split_stationarity_pvalue(
    diagnostic, estimation, reference_energy, *, rng, simulations
):
    """Compare block-energy ratio to the selected reference population."""
    center_energy = reference_energy.mean(axis=0)
    center = np.log(center_energy[1] / center_energy[0])
    observed = np.log(
        np.mean(np.abs(estimation) ** 2, axis=(0, 1))
        / np.mean(np.abs(diagnostic) ** 2, axis=(0, 1))
    )
    indices = rng.integers(len(reference_energy), size=(simulations, len(estimation)))
    draws = reference_energy[indices].mean(axis=1)
    null = np.log(draws[:, 1] / draws[:, 0])
    exceed = np.count_nonzero(
        np.max(np.abs(null - center), axis=-1) >= np.max(np.abs(observed - center))
    )
    return (1 + exceed) / (simulations + 1)


def estimate_csi_noise(
    raw_data: xr.DataArray,
    calibration: xr.Dataset,
    *,
    population_id: str,
    acquisition_noise_model: str,
    seed: int,
    outer_fraction: float = 0.25,
    minimum_points: int = 12,
    bootstrap_samples: int = 399,
    structure_samples: int = 199,
    structure_alpha: float = 0.01,
) -> xr.Dataset:
    """Estimate CSI receiver noise from calibrated outer-k-space samples.

    This is the advanced, explicitly named noise source
    ``"automatic_outer_kspace"``. It is never selected by :func:`estimate_noise`
    or by the quality workflow on their own; use :func:`estimate_noise` with an
    RF-off acquisition, explicit noise frames or an outside-object mask whenever
    such a reference exists.

    Encoding points in the geometric outer border of the read/phase grid are
    candidates. A population calibration from
    ``calibrate_csi_noise_selection``
    predicts from the diagnostic sample block whether each candidate is noise
    dominated. The channel covariance is then estimated only from the reserved
    estimation sample block of the selected points, separately for each loop
    group (for example slice or repetition).

    The caller must justify the population match and the declared acquisition
    model (white samples, independent points and homogeneous spatial
    covariance). Separate blocks or nonsignificant diagnostics do not establish
    that model. Correlated channels are supported; correlated raw sample or
    point noise needs a different calibration and estimator.

    Parameters
    ----------
    raw_data : xarray.DataArray
        Complex Bruker CSI k-space samples before reconstruction or filtering,
        as returned by ``BrukerCsiRecoStrategy().sort_kspace(handler,
        data_type="fid", cut_off=0)``. Required dimensions are
        ``acquisition_idx``, ``read_idx``, ``phase_idx`` and ``channel_idx`` in
        any order; every further dimension defines one independent group. The
        ``acquisition_time_s`` coordinate (seconds) and the ``channel_idx``
        coordinate must equal those of the calibration, and the raw-input
        contract (acquisition counts and fingerprint) must pass
        ``hypermri.reconstruction.csi_raw.audit_csi_raw_input``. Dask
        input is computed eagerly. The input is not mutated.
    calibration : xarray.Dataset
        Model from ``calibrate_csi_noise_selection`` with
        ``calibration_status="validated"``, created by the current noise source
        code. It defines the diagnostic and estimation sample blocks
        (``diagnostic_sample_index`` and ``estimation_sample_index``).
    population_id : str
        Identifier of the population the caller asserts ``raw_data`` belongs
        to. It must equal ``calibration.attrs["population_id"]``.
    acquisition_noise_model : str
        Declared acquisition noise model. Only
        ``ACQUISITION_NOISE_MODEL`` is supported.
    seed : int
        Non-negative seed. Each group draws from an independent
        :class:`numpy.random.SeedSequence` child, so results are deterministic
        for a given seed and independent of dimension order and chunking.
    outer_fraction : float, default=0.25
        Geometric border fraction of each non-singleton read/phase axis that
        forms the candidate region, strictly between 0 and 0.5.
    minimum_points : int, default=12
        Minimum number of selected encoding points per group, at least 2.
        Groups with fewer points report ``insufficient``.
    bootstrap_samples : int, default=399
        Number of point-bootstrap replicates for the sigma intervals, at least
        99.
    structure_samples : int, default=199
        Number of simulations for each of the two structure checks, at least
        99.
    structure_alpha : float, default=0.01
        Family-wise rejection level of the two Bonferroni-corrected structure
        checks, strictly between 0 and 1 and at least
        ``1 / (structure_samples + 1)``.

    Returns
    -------
    xarray.Dataset
        The candidate report of
        ``inspect_csi_noise_candidates``
        extended with the per-point selection (``selected_mask``,
        ``selection_reason``, ``p_noise_dominated*``, ``signal_fraction*``)
        and per-group results (``selected_point_count``,
        ``group_noise_status``, ``group_noise_reason``, ``structure_p_value``,
        ``split_stationarity_p_value``). ``channel_covariance`` is
        ``E[n n^H]`` with dimensions groups, ``channel_idx`` and
        ``channel_partner``. ``noise_sigma_quadrature`` is
        ``sqrt(diag(C)/2)``, the SD per real/imaginary quadrature, with
        dimensions groups and ``channel_idx``, in raw k-space sample amplitude
        units; it is not on the scale of reconstructed spectra and must be
        propagated with
        ``hypermri.reconstruction.csi_noise.reconstruct_csi_with_noise``.
        ``noise_sigma_low`` and ``noise_sigma_high`` are 95 % point-bootstrap
        percentile limits conditional on the selection and declared model.
        Attributes include ``noise_source="automatic_outer_kspace"``,
        ``noise_status`` (``"estimated"`` only when every group is estimated),
        ``noise_reason``, the calibration and source provenance and the
        content digest ``noise_id``.

    Raises
    ------
    ValueError
        If ``seed``, ``minimum_points``, ``bootstrap_samples``,
        ``structure_samples``, ``structure_alpha`` or ``outer_fraction`` is
        out of range, or if ``raw_data`` lacks a required dimension.
    TypeError
        If ``raw_data`` is not an :class:`xarray.DataArray`.

    Notes
    -----
    Calibration, population, model or sampling mismatches do not raise; they
    return ``noise_status="insufficient"`` with the reason in
    ``noise_reason`` and NaN sigma. The point bootstrap preserves each selected
    point's complete channel/time block. Its intervals quantify sampling
    variability conditional on the selection and declared model, not
    signal-model or population uncertainty. No fitted-spectrum detection
    decisions are made by this function.
    """
    seed = _positive_integer(seed, "seed", 0)
    minimum_points = _positive_integer(minimum_points, "minimum_points", 2)
    bootstrap_samples = _positive_integer(bootstrap_samples, "bootstrap_samples", 99)
    structure_samples = _positive_integer(structure_samples, "structure_samples", 99)
    if (
        not np.isscalar(structure_alpha)
        or not np.isrealobj(structure_alpha)
        or not 0 < structure_alpha < 1
    ):
        raise ValueError("structure_alpha must lie strictly between zero and one.")
    if structure_alpha < 1 / (structure_samples + 1):
        raise ValueError(
            "structure_samples cannot resolve the requested structure_alpha."
        )
    diagnostic = calibration.diagnostic_sample_index.values
    estimation = calibration.estimation_sample_index.values
    raw_data = raw_data.transpose(*sorted(raw_data.dims))
    report = inspect_csi_noise_candidates(
        raw_data,
        diagnostic_samples=slice(int(diagnostic[0]), int(diagnostic[-1]) + 1),
        estimation_samples=slice(int(estimation[0]), int(estimation[-1]) + 1),
        outer_fraction=outer_fraction,
    )
    reason = validate_calibration(calibration)
    if not report.attrs["raw_input_verified"]:
        reason = report.attrs["noise_reason"]
    elif population_id != calibration.attrs.get("population_id"):
        reason = "reference_population_mismatch"
    elif acquisition_noise_model != ACQUISITION_NOISE_MODEL:
        reason = "unsupported_acquisition_noise_model"
    elif not np.array_equal(
        raw_data.acquisition_time_s.values, calibration.reference_time_s.values
    ) or not np.array_equal(
        raw_data.channel_idx.values, calibration.channel_idx.values
    ):
        reason = "reference_sampling_or_channels_mismatch"
    report.attrs.update(
        noise_source="automatic_outer_kspace",
        calibration_id=calibration.attrs.get("calibration_id", "unavailable"),
        calibration_status=calibration.attrs.get("calibration_status", "unavailable"),
        source_hash=noise_source_hash(),
        runtime=calibration.attrs.get("runtime", "unavailable"),
        population_id=population_id,
        acquisition_noise_model=acquisition_noise_model,
        split_independence="conditional_on_declared_acquisition_model",
        seed=seed,
        rng="PCG64; independent SeedSequence child per canonical group",
        minimum_points=minimum_points,
        bootstrap_samples=bootstrap_samples,
        structure_samples=structure_samples,
        structure_alpha=float(structure_alpha),
        contamination_tolerance=calibration.attrs["contamination_tolerance"],
        noise_sigma_convention="SD per real/imaginary quadrature; E[abs(n)**2]=2*sigma**2",
        interval_scope="95% point-bootstrap percentile intervals, conditional on selection and acquisition model; excludes population mismatch and residual signal bias",
        structure_diagnostic="fixed-budget plus-one parametric plug-in omnibus p-value and selected-reference block-energy bootstrap p-value; Bonferroni across two checks; rejection checks, not probabilities of white noise or proof of independence",
        spatial_scope="one covariance per slice/repetition/other loop group, homogeneous across acquired encoding points by declared model",
    )
    raw = raw_data.compute() if raw_data.chunks is not None else raw_data
    groups = sorted(
        set(raw.dims) - {"acquisition_idx", "read_idx", "phase_idx", "channel_idx"}
    )
    dims = groups + ["read_idx", "phase_idx", "acquisition_idx", "channel_idx"]
    values = raw.transpose(*dims).values
    shape = tuple(raw.sizes[d] for d in groups)
    channels = raw.sizes["channel_idx"]
    point_shape = shape + (raw.sizes["read_idx"], raw.sizes["phase_idx"])
    prediction_fields = {
        name: np.full(point_shape, np.nan)
        for name in (
            "p_noise_dominated",
            "p_noise_dominated_low",
            "p_noise_dominated_high",
            "signal_fraction",
            "signal_fraction_low",
            "signal_fraction_high",
        )
    }
    selection = np.zeros(point_shape, dtype=bool)
    rejection = np.full(point_shape, "ineligible", dtype="U64")
    covariance = np.full(shape + (channels, channels), np.nan + 1j * np.nan)
    bootstrap = np.full(
        shape + (bootstrap_samples, channels, channels), np.nan + 1j * np.nan
    )
    status = np.full(shape, "insufficient", dtype="U16")
    reasons = np.full(shape, reason or "too_few_selected_points", dtype="U64")
    count = np.zeros(shape, dtype=int)
    statistic, pvalue = np.full(shape, np.nan), np.full(shape, np.nan)
    split_pvalue = np.full(shape, np.nan)
    eligible = (
        report.candidate_eligible.all("channel_idx")
        .transpose(*groups, "read_idx", "phase_idx")
        .values
    )
    streams = np.random.SeedSequence(seed).spawn(max(1, int(np.prod(shape))))
    for flat_index, group in enumerate(np.ndindex(shape)):
        if reason:
            rejection[group] = reason
            continue
        predictions = predict_noise_dominance(
            values[group][..., diagnostic, :], calibration
        )
        for name, source in zip(
            prediction_fields,
            (
                "probability",
                "probability_low",
                "probability_high",
                "fraction_median",
                "fraction_low",
                "fraction_high",
            ),
            strict=True,
        ):
            prediction_fields[name][group] = np.where(
                eligible[group], predictions[source], np.nan
            )
        selection[group] = eligible[group] & predictions["selected"]
        rejection[group] = np.where(
            ~eligible[group],
            "ineligible",
            np.where(
                ~predictions["supported"],
                "outside_calibration_support",
                np.where(
                    predictions["selected"],
                    "selected",
                    "contamination_probability_too_low",
                ),
            ),
        )
        count[group] = selection[group].sum()
        if count[group] < minimum_points:
            continue
        samples = values[group][selection[group]][:, estimation, :]
        if not np.all(np.isfinite(samples)):
            reasons[group] = "nonfinite_estimation_samples"
            continue
        rng = np.random.default_rng(streams[flat_index])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = _estimate_group(
                samples,
                rng=rng,
                bootstrap_samples=bootstrap_samples,
                structure_samples=structure_samples,
            )
        if (
            result is None
            or not np.all(np.isfinite(result[0]))
            or not np.isfinite(result[2])
        ):
            reasons[group] = "invalid_covariance_estimate"
            continue
        estimate, draws, statistic[group], pvalue[group] = result
        split_pvalue[group] = _split_stationarity_pvalue(
            values[group][selection[group]][:, diagnostic, :],
            samples,
            calibration.selected_reference_block_energy.values,
            rng=rng,
            simulations=structure_samples,
        )
        if min(pvalue[group], split_pvalue[group]) * 2 <= structure_alpha:
            reasons[group] = "acquisition_noise_model_rejected"
            continue
        covariance[group], bootstrap[group] = estimate, draws
        status[group], reasons[group] = (
            "estimated",
            "validated_reference_and_declared_model",
        )
    point_dims = groups + ["read_idx", "phase_idx"]
    for name, data in prediction_fields.items():
        report[name] = (point_dims, data)
    report["selected_mask"] = (point_dims, selection)
    report["selection_reason"] = (point_dims, rejection)
    report["selected_point_count"] = (groups, count)
    report["group_noise_status"] = (groups, status)
    report["group_noise_reason"] = (groups, reasons)
    report["structure_statistic"] = (groups, statistic)
    report["structure_p_value"] = (groups, pvalue)
    report["split_stationarity_p_value"] = (groups, split_pvalue)
    report["channel_covariance"] = (
        groups + ["channel_idx", "channel_partner"],
        covariance,
    )
    report["channel_covariance_bootstrap"] = (
        groups + ["bootstrap", "channel_idx", "channel_partner"],
        bootstrap,
    )
    report["noise_sigma_quadrature"] = (
        groups + ["channel_idx"],
        np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1).real / 2),
    )
    sigma_draws = np.sqrt(np.diagonal(bootstrap, axis1=-2, axis2=-1).real / 2)
    for name, quantile in (("noise_sigma_low", 0.025), ("noise_sigma_high", 0.975)):
        report[name] = (
            groups + ["channel_idx"],
            np.quantile(sigma_draws, quantile, axis=-2),
        )
    report.attrs.update(
        noise_status="estimated" if np.all(status == "estimated") else "insufficient",
        noise_reason="all_groups_estimated"
        if np.all(status == "estimated")
        else "see_group_noise_reason",
    )
    report.attrs["noise_id"] = dataset_digest(report)
    return report


ACQUISITION_TAIL_NOISE_SOURCE = "acquisition_tail"


@dataclass(frozen=True)
class AcquisitionTailNoise:
    """Acquisition noise measured from the signal-free end of a CSI acquisition.

    Produced by :func:`estimate_acquisition_tail_noise`; all sigma values are
    standard deviations per real or imaginary quadrature in acquired signal
    units, so that ``E|n|^2 = 2 * sigma**2`` per complex sample.

    Attributes
    ----------
    sigma_quadrature : float
        Noise estimate: the Rayleigh maximum-likelihood scale
        ``sqrt(mean(|s|**2) / 2)`` of the noise-only samples.
    sigma_standard_error : float
        Sampling standard error of ``sigma_quadrature`` for white Gaussian
        noise, ``sigma_quadrature / sqrt(2 * n_samples)``.
    sigma_real, sigma_imag : float
        Gaussian maximum-likelihood cross-checks: standard deviations of the
        real and imaginary parts of the same samples.
    onset_point : int
        First acquired point (zero-based, acquisition order) from which every
        later point is consistent with noise in both detectors.
    n_points : int
        Acquired points used, ``onset_point`` to the last point.
    n_fid_samples : int
        FID samples used per point; the last FID sample is excluded.
    n_samples : int
        Complex samples used, ``n_points * n_fid_samples``.
    z_band : float
        Band-power z-score of the noise-only remainder, below ``z_limit`` in
        magnitude. The energy z-score is zero by construction, because sigma
        is estimated from the same samples.
    bands_hz : tuple of (float, float) or None
        Frequency bands of the band-power detector in Hz; None means the
        whole spectral width.
    dwell_time_s : float
        FID sampling interval in seconds.
    """

    sigma_quadrature: float
    sigma_standard_error: float
    sigma_real: float
    sigma_imag: float
    onset_point: int
    n_points: int
    n_fid_samples: int
    n_samples: int
    z_band: float
    bands_hz: tuple[tuple[float, float], ...] | None
    dwell_time_s: float

    def provenance(self) -> dict[str, object]:
        """Return the estimate's provenance as netCDF-compatible attributes.

        Returns
        -------
        dict of str to object
            ``noise_source`` (``"acquisition_tail"``), ``noise_estimator``,
            ``noise_onset_point``, ``noise_n_points``,
            ``noise_n_fid_samples``, ``noise_n_samples``,
            ``noise_sigma_acquired`` (acquired signal units),
            ``noise_sigma_acquired_standard_error``, ``noise_bands_hz`` (a
            flat ``[low, high, ...]`` list in Hz, empty for the whole
            spectral width) and ``noise_dwell_time_s``.
        """
        bands = [] if self.bands_hz is None else [v for b in self.bands_hz for v in b]
        return {
            "noise_source": ACQUISITION_TAIL_NOISE_SOURCE,
            "noise_estimator": "rayleigh_mle_without_last_fid_sample",
            "noise_onset_point": int(self.onset_point),
            "noise_n_points": int(self.n_points),
            "noise_n_fid_samples": int(self.n_fid_samples),
            "noise_n_samples": int(self.n_samples),
            "noise_sigma_acquired": float(self.sigma_quadrature),
            "noise_sigma_acquired_standard_error": float(self.sigma_standard_error),
            "noise_bands_hz": bands,
            "noise_dwell_time_s": float(self.dwell_time_s),
        }


def _tail_z_scores(per_point: np.ndarray, n_per_point: int) -> np.ndarray:
    """Return the z-score of every remainder ``per_point[p0:]`` against 1."""
    remainder_sum = np.cumsum(per_point[::-1])[::-1]
    remainder_count = np.arange(per_point.size, 0, -1)
    mean = remainder_sum / remainder_count
    return (mean - 1.0) * np.sqrt(n_per_point * remainder_count)


def suggest_signal_end(
    matrix: np.ndarray,
    *,
    dwell_time_s: float,
    bands_hz: Sequence[tuple[float, float]] | None = None,
    margin_points: int = 0,
    n_repetitions: int = 1,
    repetition_index: int = 0,
) -> int:
    """Suggest the CSI ``signal_end`` as the noise onset of the acquisition.

    ``signal_end`` is the number of acquired k-space points kept by the
    reconstruction (``CSI.create_filtered_data(end_index=...)``,
    ``filter_type="None"``). Keeping fewer points than carry signal loses
    signal; keeping noise-only points adds noise. The suggestion is the noise
    onset of :func:`estimate_acquisition_tail_noise`, the first acquired point
    from which every later point is consistent with noise, plus
    ``margin_points``. With the default margin 0 exactly the points before
    the onset are kept.

    Parameters
    ----------
    matrix : numpy.ndarray
        Complex (FID sample, acquired k-space point) matrix in acquisition
        order, before any filter, as shown by :meth:`CSI.plot_signal_decay`.
        Repetitions are contiguous equal blocks along the second axis.
    dwell_time_s : float
        Positive FID sampling interval in seconds.
    bands_hz : sequence of (float, float) or None, optional
        Inclusive frequency ranges in Hz of the metabolite lines on the axis
        of :func:`spectral_band_power`. Narrow bands find weak late signal;
        None uses the whole spectral width.
    margin_points : int, optional
        Non-negative number of acquired points added after the onset;
        default 0 (decided on 19 November 2026).
    n_repetitions : int, optional
        Positive number of repetition blocks in ``matrix``; default 1. A
        point count not divisible by it uses the whole matrix.
    repetition_index : int, optional
        Zero-based repetition whose onset is suggested; default 0.

    Returns
    -------
    int
        Suggested number of kept acquired points, between 1 and the points
        per repetition.

    Raises
    ------
    ValueError
        For a negative margin, an invalid repetition selection, or any
        failure of :func:`estimate_acquisition_tail_noise`, for example when
        the acquisition has no signal-free remainder of 50 points.

    Notes
    -----
    Only a suggestion: no default ``signal_end`` is changed. Eager,
    deterministic and free of input mutation and caching.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.normal(size=(64, 400)) + 1j * rng.normal(size=(64, 400))
    >>> signal = 20.0 * (np.arange(400) < 150)
    >>> suggest_signal_end(
    ...     signal + noise, dwell_time_s=1e-3, bands_hz=[(-100, 100)]
    ... )
    150
    """
    margin = int(margin_points)
    if margin != margin_points or margin < 0:
        raise ValueError(
            f"margin_points must be a non-negative integer, got {margin_points!r}."
        )
    block = _repetition_block(matrix, int(n_repetitions), int(repetition_index))
    tail = estimate_acquisition_tail_noise(
        block, dwell_time_s=dwell_time_s, bands_hz=bands_hz
    )
    return int(np.clip(tail.onset_point + margin, 1, block.shape[1]))


def estimate_acquisition_tail_noise(
    matrix: np.ndarray,
    *,
    dwell_time_s: float,
    bands_hz: Sequence[tuple[float, float]] | None = None,
    z_limit: float = 2.0,
    min_noise_points: int = 50,
    initial_tail_fraction: float = 0.25,
    max_iterations: int = 3,
) -> AcquisitionTailNoise:
    """Estimate acquisition noise from the signal-free end of a CSI acquisition.

    In a centric CSI acquisition the late acquired k-space points carry no
    signal. Two detectors are evaluated per acquired point, both normalised by
    the current noise power ``2 sigma**2``: the energy of the whole FID and the
    metabolite-band power of :func:`spectral_band_power`. For white noise both
    have mean 1. The noise onset is the earliest point ``p0`` from which the
    mean of every later point of both detectors lies within ``z_limit``
    standard errors of 1. Sigma is re-estimated from the points after the onset
    and the search is repeated until the onset no longer changes or
    ``max_iterations`` is reached. The first guess uses the last
    ``initial_tail_fraction`` of the points.

    Parameters
    ----------
    matrix : numpy.ndarray
        Complex (FID sample, acquired k-space point) matrix of one repetition
        in acquisition order, in acquired signal units, before any filter or
        scaling. :meth:`CSI.plot_signal_decay` uses the same layout. FID
        samples start at time zero.
    dwell_time_s : float
        Positive FID sampling interval in seconds.
    bands_hz : sequence of (float, float) or None, optional
        Inclusive frequency ranges in Hz where metabolite lines sit, on the
        axis of :func:`spectral_band_power`. None uses the whole spectral
        width, so the band detector equals the energy detector and weak late
        signal is found later.
    z_limit : float, optional
        Positive detector tolerance in standard errors; default 2.
    min_noise_points : int, optional
        Minimum number of acquired points after the onset; default 50.
    initial_tail_fraction : float, optional
        Fraction of the points, in (0, 1], at the end of the acquisition used
        for the first noise guess; default 0.25.
    max_iterations : int, optional
        Positive number of onset searches; default 3.

    Returns
    -------
    AcquisitionTailNoise
        Per-quadrature sigma in acquired signal units, its standard error,
        Gaussian cross-checks, the onset point and the samples used.

    Raises
    ------
    ValueError
        For a matrix that is not complex and two-dimensional, has fewer than
        two FID samples or contains non-finite values; an invalid dwell time,
        band or option; or when no remainder of at least ``min_noise_points``
        points is consistent with noise (the acquisition has too few
        signal-free points or the noise is not white).

    Notes
    -----
    The last FID sample is always excluded because the scanner's digital
    filter attenuates it. The detectors assume independent white complex
    Gaussian noise of equal variance in every acquired point. Eager,
    deterministic and free of input mutation and caching.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.normal(size=(64, 400)) + 1j * rng.normal(size=(64, 400))
    >>> t = np.arange(64)[:, None] * 1e-3
    >>> signal = 50 * np.exp(-t / 0.02) * np.exp(-np.arange(400) / 20.0)
    >>> result = estimate_acquisition_tail_noise(
    ...     signal + noise, dwell_time_s=1e-3, bands_hz=[(-100, 100)]
    ... )
    >>> bool(abs(result.sigma_quadrature - 1.0) < 0.02)
    True
    """
    values = np.asarray(matrix)
    if values.ndim != 2 or not np.iscomplexobj(values):
        raise ValueError(
            "estimate_acquisition_tail_noise needs a complex (FID sample, "
            f"acquired point) matrix; got shape {values.shape} and dtype "
            f"{values.dtype}."
        )
    n_fid, n_points = values.shape
    if n_fid < 2:
        raise ValueError(
            "estimate_acquisition_tail_noise needs at least two FID samples, "
            "because the last sample is excluded."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "estimate_acquisition_tail_noise needs finite samples; pass the "
            "acquired data before any masking."
        )
    dwell = float(dwell_time_s) if dwell_time_s is not None else np.nan
    if not (np.isfinite(dwell) and dwell > 0):
        raise ValueError(
            "estimate_acquisition_tail_noise needs a positive dwell_time_s in "
            f"seconds, got {dwell_time_s!r}; read it from "
            "handler.to_csi_acquisition().dwell_time_s."
        )
    if not (np.isfinite(z_limit) and z_limit > 0):
        raise ValueError(f"z_limit must be positive, got {z_limit!r}.")
    if not 0 < initial_tail_fraction <= 1:
        raise ValueError(
            f"initial_tail_fraction must be in (0, 1], got {initial_tail_fraction!r}."
        )
    min_points = _positive_integer(min_noise_points, "min_noise_points")
    iterations = _positive_integer(max_iterations, "max_iterations")
    if n_points < min_points:
        raise ValueError(
            f"estimate_acquisition_tail_noise needs at least min_noise_points="
            f"{min_points} acquired points, got {n_points}."
        )

    samples = values[:-1].astype(np.complex128, copy=False)
    n_used = samples.shape[0]
    band_count = int(_band_selection(n_used, dwell, bands_hz).sum())
    energy = np.mean(np.abs(samples) ** 2, axis=0)
    band = spectral_band_power(samples, dwell_time_s=dwell, bands_hz=bands_hz)

    first = n_points - max(min_points, int(np.ceil(initial_tail_fraction * n_points)))
    power = np.mean(energy[max(first, 0) :])
    onset = None
    for _ in range(iterations):
        if not power > 0:
            raise ValueError(
                "estimate_acquisition_tail_noise found zero power in the "
                "acquisition tail; the data contain no noise."
            )
        z_e = _tail_z_scores(energy / power, n_used)
        z_b = _tail_z_scores(band / power, band_count)
        candidates = np.flatnonzero((np.abs(z_e) < z_limit) & (np.abs(z_b) < z_limit))
        candidates = candidates[candidates <= n_points - min_points]
        if candidates.size == 0:
            raise ValueError(
                "estimate_acquisition_tail_noise found no remainder of at "
                f"least {min_points} acquired points consistent with noise "
                f"(|z| < {z_limit:g}). The acquisition may have too few "
                "signal-free points; lower min_noise_points or check the bands."
            )
        new_onset = int(candidates[0])
        power = float(np.mean(energy[new_onset:]))
        if new_onset == onset:
            break
        onset = new_onset

    tail = samples[:, onset:]
    sigma = float(np.sqrt(power / 2.0))
    n_tail = int(tail.size)
    return AcquisitionTailNoise(
        sigma_quadrature=sigma,
        sigma_standard_error=float(sigma / np.sqrt(2.0 * n_tail)),
        sigma_real=float(np.std(tail.real)),
        sigma_imag=float(np.std(tail.imag)),
        onset_point=onset,
        n_points=n_points - onset,
        n_fid_samples=n_used,
        n_samples=n_tail,
        z_band=float(z_b[onset]),
        bands_hz=_band_limits(bands_hz),
        dwell_time_s=dwell,
    )


def propagate_acquisition_tail_noise(
    noise: AcquisitionTailNoise,
    *,
    point_weights: np.ndarray,
    n_spatial_points: int,
    amplitude_scale: float = 1.0,
    mask_weight: float | xr.DataArray = 1.0,
) -> xr.DataArray:
    """Propagate acquisition noise to the per-voxel noise of a CSI reconstruction.

    The CSI reconstruction is linear: each acquired point ``k`` is multiplied by
    its filter weight ``w_k`` (window, ``signal_end`` truncation, spatial
    window), the grid is transformed with
    :func:`~hypermri.reconstruction.centred_dft.centered_ifft` (backward
    normalisation ``1 / n_spatial_points``), and the voxel is scaled by
    ``amplitude_scale`` and the mask weight ``W``. For white acquisition noise
    every voxel therefore has the per-quadrature noise::

        sigma_voxel = |W| * |amplitude_scale| * sigma * sqrt(sum |w_k|**2) / n_spatial_points

    which is ``E|n_voxel|^2 = W^2 * 2 sigma^2 * n_kept / N^2 * scale^2`` for
    uniform weights.

    Parameters
    ----------
    noise : AcquisitionTailNoise
        Result of :func:`estimate_acquisition_tail_noise` on the unfiltered
        acquisition.
    point_weights : numpy.ndarray
        Real or complex weight of every acquired k-space point that the
        reconstruction applies, shape (n_acquired_points,). For the windows of
        ``CSI.filter_signal`` filter a complex matrix of ones with the same
        settings;
        with ``remove_lowest_signal`` use 1 for kept and 0 for zeroed points.
    n_spatial_points : int
        Positive number of read times phase grid points of the spatial FFT.
    amplitude_scale : float, optional
        Linear factor applied to the acquired data before or after the
        reconstruction, for example a user rescaling; default 1.
    mask_weight : float or xarray.DataArray, optional
        Mask weight ``W`` that ``MRIExperiment.apply_mask_to`` multiplies into
        each voxel, as a scalar or a labelled per-voxel map; default 1.

    Returns
    -------
    xarray.DataArray
        ``noise_sigma_quadrature``: per-quadrature voxel noise in the units of
        the reconstructed FIDs, scalar or with the dimensions of
        ``mask_weight``. It can be passed as ``noise`` to
        :func:`~hypermri.analysis.spectroscopy.fitting.fit_spectrum`. The
        attributes hold :meth:`AcquisitionTailNoise.provenance` plus
        ``noise_point_weight_sum_sq``, ``noise_n_kept_points``,
        ``noise_n_spatial_points`` and ``noise_amplitude_scale``.

    Raises
    ------
    TypeError
        If ``noise`` is not an :class:`AcquisitionTailNoise`.
    ValueError
        For empty or non-finite point weights, a non-positive grid size, or a
        non-finite amplitude scale or mask weight.

    Notes
    -----
    Assumes one acquisition per grid point (no averaging of repeated points)
    and weights that do not vary along the FID. Neighbouring voxels are
    uncorrelated only for uniform weights. Eager and deterministic.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> tail = rng.normal(size=(32, 300)) + 1j * rng.normal(size=(32, 300))
    >>> acquired = estimate_acquisition_tail_noise(tail, dwell_time_s=1e-3)
    >>> weights = np.r_[np.ones(200), np.zeros(100)]
    >>> voxel = propagate_acquisition_tail_noise(
    ...     acquired, point_weights=weights, n_spatial_points=300
    ... )
    >>> voxel.attrs["noise_n_kept_points"]
    200
    """
    if not isinstance(noise, AcquisitionTailNoise):
        raise TypeError(
            "propagate_acquisition_tail_noise needs the AcquisitionTailNoise "
            "returned by estimate_acquisition_tail_noise."
        )
    weights = np.asarray(point_weights).reshape(-1)
    if weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError(
            "point_weights must be a nonempty finite weight per acquired point."
        )
    n_grid = _positive_integer(n_spatial_points, "n_spatial_points")
    scale = float(amplitude_scale)
    if not np.isfinite(scale):
        raise ValueError(f"amplitude_scale must be finite, got {amplitude_scale!r}.")
    weight_map = (
        mask_weight
        if isinstance(mask_weight, xr.DataArray)
        else xr.DataArray(float(mask_weight))
    )
    if np.iscomplexobj(weight_map) or not np.all(np.isfinite(weight_map.values)):
        raise ValueError("mask_weight must be real and finite.")
    sum_sq = float(np.sum(np.abs(weights) ** 2))
    factor = abs(scale) * np.sqrt(sum_sq) / n_grid
    sigma = np.abs(weight_map).astype(float) * (noise.sigma_quadrature * factor)
    sigma.name = "noise_sigma_quadrature"
    sigma.attrs = {
        "definition": (
            "SD per real/imaginary quadrature of the reconstructed temporal "
            "FID samples; E[abs(noise)**2]=2*sigma**2"
        ),
        **noise.provenance(),
        "noise_point_weight_sum_sq": sum_sq,
        "noise_n_kept_points": int(np.count_nonzero(weights)),
        "noise_n_spatial_points": n_grid,
        "noise_amplitude_scale": scale,
    }
    return sigma


def attach_acquisition_tail_noise(
    data: xr.DataArray,
    *,
    noise: xr.DataArray,
    spectral_dim: str = "acquisition_idx",
) -> xr.DataArray:
    """Attach propagated acquisition-tail noise to reconstructed FIDs.

    ``fit_data_pseudo_inv`` reads the coordinate ``noise_sigma_quadrature`` of
    its input and then reports a finite ``fit_reduced_chi_squared`` with
    ``noise_source="acquisition_tail"``. This function checks the noise once
    and attaches it, so every fit of the returned data (including the fits and
    single-voxel refits of ``MRIExperiment.interactive_fit``) uses the same
    per-voxel sigma.

    Parameters
    ----------
    data : xarray.DataArray
        Reconstructed FIDs with the spectral dimension ``spectral_dim`` and
        any number of voxel or repetition dimensions.
    noise : xarray.DataArray
        Per-quadrature voxel noise from :func:`propagate_acquisition_tail_noise`
        in the signal units of ``data``: a scalar, or labelled with a subset of
        the non-spectral dimensions of ``data``. Shared dimension coordinates
        must equal those of ``data``. Its attribute ``noise_source`` must be
        ``"acquisition_tail"``; all attributes are kept.
    spectral_dim : str, optional
        Name of the FID sample dimension of ``data``; default
        ``"acquisition_idx"``, as in ``fit_data_pseudo_inv``.

    Returns
    -------
    xarray.DataArray
        A shallow copy of ``data`` with the float64 coordinate
        ``noise_sigma_quadrature`` over the dimensions of ``noise`` (in the
        order of ``data``). Values, other coordinates and attributes of
        ``data`` are unchanged; ``data`` itself is not modified.

    Raises
    ------
    TypeError
        If ``data`` or ``noise`` is not a DataArray.
    ValueError
        If ``noise_source`` is not ``"acquisition_tail"``; if sigma is complex,
        non-finite or not positive; if ``noise`` has the spectral dimension or
        a dimension that ``data`` lacks; if shared coordinates differ; or if
        ``data`` already carries ``noise_sigma_quadrature``.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> tail = rng.normal(size=(32, 300)) + 1j * rng.normal(size=(32, 300))
    >>> acquired = estimate_acquisition_tail_noise(tail, dwell_time_s=1e-3)
    >>> sigma = propagate_acquisition_tail_noise(
    ...     acquired, point_weights=np.ones(300), n_spatial_points=300
    ... )
    >>> fids = xr.DataArray(
    ...     np.zeros((64, 4), dtype=complex), dims=("acquisition_idx", "x")
    ... )
    >>> attached = attach_acquisition_tail_noise(fids, noise=sigma)
    >>> attached.coords["noise_sigma_quadrature"].attrs["noise_source"]
    'acquisition_tail'
    """
    operation = "attach_acquisition_tail_noise"
    if not isinstance(data, xr.DataArray) or not isinstance(noise, xr.DataArray):
        raise TypeError(
            f"{operation}: data and noise must be xarray.DataArray; pass the "
            "result of propagate_acquisition_tail_noise as noise."
        )
    source = noise.attrs.get("noise_source")
    if source != ACQUISITION_TAIL_NOISE_SOURCE:
        raise ValueError(
            f"{operation}: noise has noise_source={source!r}; only "
            f"'{ACQUISITION_TAIL_NOISE_SOURCE}' noise from "
            "propagate_acquisition_tail_noise can be attached. Pass other noise "
            "explicitly to fit_spectrum(noise=...)."
        )
    if "noise_sigma_quadrature" in data.coords:
        raise ValueError(
            f"{operation}: data already carry 'noise_sigma_quadrature'; drop it "
            "with data.drop_vars('noise_sigma_quadrature') before attaching new noise."
        )
    if spectral_dim not in data.dims:
        raise ValueError(
            f"{operation}: data lack the spectral dimension {spectral_dim!r}; "
            "set spectral_dim to the FID sample dimension."
        )
    unknown = [dim for dim in noise.dims if dim not in data.dims]
    if spectral_dim in noise.dims or unknown:
        raise ValueError(
            f"{operation}: noise dimensions {noise.dims} must be a subset of the "
            f"voxel dimensions of data {tuple(d for d in data.dims if d != spectral_dim)}; "
            "sigma is one value per spectrum."
        )
    values = np.asarray(noise.values)
    if (
        np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError(f"{operation}: noise sigma must be real, finite and positive.")
    for dim in noise.dims:
        if noise.sizes[dim] != data.sizes[dim]:
            raise ValueError(
                f"{operation}: noise has {noise.sizes[dim]} entries along {dim!r}, "
                f"data have {data.sizes[dim]}."
            )
        if dim in noise.coords and dim in data.coords:
            if not np.array_equal(noise[dim].values, data[dim].values):
                raise ValueError(
                    f"{operation}: coordinate {dim!r} of noise differs from data."
                )
    order = [dim for dim in data.dims if dim in noise.dims]
    sigma = xr.DataArray(
        np.asarray(noise.transpose(*order).values, dtype=float),
        dims=order,
        attrs=dict(noise.attrs),
    )
    return data.assign_coords(noise_sigma_quadrature=sigma)


def _temporal_noise_data(
    data: xr.DataArray,
    *,
    signal_domain: str | None,
    spectral_dim: str,
) -> xr.DataArray:
    """Return complex noise samples in the temporal fit domain."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("Noise data must be an xarray.DataArray.")
    if not np.iscomplexobj(data):
        raise TypeError("Noise data must be complex-valued.")
    domain = str(
        signal_domain
        if signal_domain is not None
        else data.attrs.get("signal_domain", "temporal")
    ).lower()
    if domain == "temporal":
        return data
    if domain != "spectral":
        raise ValueError("signal_domain must be 'temporal' or 'spectral'.")
    if spectral_dim not in data.dims:
        raise ValueError(f"Spectral noise data is missing dimension '{spectral_dim}'.")
    axis = data.get_axis_num(spectral_dim)
    values = np.fft.ifft(
        np.fft.ifftshift(np.asarray(data.values), axes=axis),
        axis=axis,
    )
    result = data.copy(data=values)
    result.attrs = dict(data.attrs)
    result.attrs["signal_domain"] = "temporal"
    return result


_SAMPLE_TYPES = ("complex", "magnitude")


def _noise_samples_as_complex(
    data: xr.DataArray,
    *,
    sample_type: str,
    center: bool,
    signal_domain: str | None,
) -> xr.DataArray:
    """Return complex noise samples, embedding magnitude samples as real parts.

    Magnitude samples ``m`` become ``m + 0j`` so that the shared estimator
    ``sqrt(sum(abs(z)**2)/(2*N))`` yields the zero-location Rayleigh scale
    maximum-likelihood estimate. The input is not mutated.
    """
    if sample_type not in _SAMPLE_TYPES:
        raise ValueError(
            f"estimate_noise: sample_type must be one of {_SAMPLE_TYPES}, "
            f"got {sample_type!r}."
        )
    if sample_type == "complex":
        return data
    if not isinstance(data, xr.DataArray):
        raise TypeError("Noise data must be an xarray.DataArray.")
    if np.iscomplexobj(data):
        raise TypeError(
            "estimate_noise(sample_type='magnitude') requires real-valued "
            "magnitude samples; pass complex samples with sample_type='complex'."
        )
    if center:
        raise ValueError(
            "estimate_noise(sample_type='magnitude') does not support "
            "center=True: subtracting the mean would destroy the Rayleigh "
            "distribution of the magnitude samples."
        )
    domain = str(
        signal_domain
        if signal_domain is not None
        else data.attrs.get("signal_domain", "temporal")
    ).lower()
    if domain != "temporal":
        raise ValueError(
            "estimate_noise(sample_type='magnitude') requires "
            "signal_domain='temporal'; magnitude spectra cannot be inverse "
            "Fourier transformed into temporal noise samples."
        )
    values = np.asarray(data.values, dtype=np.float64)
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError(
            "estimate_noise(sample_type='magnitude') received negative values; "
            "magnitude samples must be non-negative."
        )
    result = data.copy(data=values.astype(np.complex128))
    result.attrs = dict(data.attrs)
    return result


def _noise_reduction_dims(
    data: xr.DataArray,
    reduce_dims: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate dimensions whose samples form each noise estimate."""
    if reduce_dims is None:
        return tuple(data.dims)
    dims = (reduce_dims,) if isinstance(reduce_dims, str) else tuple(reduce_dims)
    if not dims:
        raise ValueError("reduce_dims must contain at least one dimension.")
    if len(set(dims)) != len(dims):
        raise ValueError("reduce_dims must not contain duplicates.")
    unknown = set(dims) - set(data.dims)
    if unknown:
        raise ValueError(f"Unknown noise reduction dimensions: {sorted(unknown)}.")
    return dims


@dataclass(frozen=True)
class _NoiseComputation:
    """Intermediate labelled values produced by one noise-estimation pass."""

    noise: xr.DataArray
    temporal: xr.DataArray
    selected: xr.DataArray
    processed: xr.DataArray
    reduction_dims: tuple[str, ...]
    sample_count: xr.DataArray


def _estimate_noise_computation(
    data: xr.DataArray,
    *,
    reduce_dims: str | Sequence[str] | None,
    center: bool,
    source: str,
    signal_domain: str | None,
    spectral_dim: str,
) -> _NoiseComputation:
    """Calculate one estimate and retain its already-computed processing stages."""
    temporal = _temporal_noise_data(
        data,
        signal_domain=signal_domain,
        spectral_dim=spectral_dim,
    )
    dims = _noise_reduction_dims(temporal, reduce_dims)
    finite = np.isfinite(temporal)
    count = finite.sum(dim=dims).astype(np.int64)
    selected = temporal.where(finite)
    processed = selected
    if center:
        processed = processed - processed.mean(dim=dims, skipna=True)
    energy = (np.abs(processed) ** 2).sum(dim=dims, skipna=True)
    sigma = np.sqrt(energy / (2.0 * count))
    count_values = np.asarray(count.values)
    sigma_values = np.asarray(sigma.values)
    if (
        np.any(count_values < 2)
        or not np.all(np.isfinite(sigma_values))
        or np.any(sigma_values <= 0)
    ):
        raise ValueError(
            "Every noise estimate requires at least two finite samples and a "
            "finite positive quadrature sigma."
        )
    sigma = sigma.astype(float).rename("noise_sigma_quadrature")
    sigma = sigma.assign_coords(noise_sample_count=count)
    sigma.attrs.update(
        noise_source=source,
        estimator=(
            "sqrt(sum(abs(z-mean(z))**2)/(2*N))"
            if center
            else "sqrt(sum(abs(z)**2)/(2*N))"
        ),
        centered=bool(center),
        reduction_dims=dims,
        signal_domain="temporal",
        quadrature_convention=(
            "SD per temporal real/imaginary quadrature; E[abs(noise)**2]=2*sigma**2"
        ),
        unit=data.attrs.get("unit", "a.u."),
    )
    return _NoiseComputation(
        noise=sigma,
        temporal=temporal,
        selected=selected,
        processed=processed,
        reduction_dims=dims,
        sample_count=count,
    )


def _select_noise_frames(
    reference_data: xr.DataArray,
    *,
    selection: xr.DataArray | Sequence[int],
    selection_dim: str,
) -> xr.DataArray:
    """Validate and apply an explicit frame selection without transforming data."""
    if not isinstance(reference_data, xr.DataArray):
        raise TypeError("Noise frame data must be an xarray.DataArray.")
    if selection_dim not in reference_data.dims:
        raise ValueError(
            f"Noise frame dimension '{selection_dim}' is absent from the data."
        )
    if isinstance(selection, xr.DataArray) and selection.dtype.kind == "b":
        if selection.dims != (selection_dim,):
            raise ValueError("A boolean frame selection must use only selection_dim.")
        try:
            selection, _ = xr.align(
                selection,
                reference_data[selection_dim],
                join="exact",
                copy=False,
            )
        except ValueError as exc:
            raise ValueError(
                "Noise frame selection coordinates must exactly match the data."
            ) from exc
        indices = np.flatnonzero(np.asarray(selection.values))
    else:
        indices = np.asarray(selection)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError(
                "Frame selection must be a one-dimensional integer sequence or "
                "a labelled boolean DataArray."
            )
    selected = reference_data.isel({selection_dim: indices.astype(int)})
    if selected.sizes[selection_dim] == 0:
        raise ValueError("Noise frame selection is empty.")
    return selected


def _select_noise_mask(
    data: xr.DataArray,
    *,
    mask: xr.DataArray,
    mask_threshold: float,
    in_mask: bool,
) -> xr.DataArray:
    """Validate a labelled mask and apply its selected region to the data."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("Masked noise data must be an xarray.DataArray.")
    if not isinstance(mask, xr.DataArray):
        raise TypeError("mask must be an xarray.DataArray.")
    if not np.isrealobj(mask):
        raise TypeError("mask must be real-valued.")
    threshold = float(mask_threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("mask_threshold must be finite and lie in [0, 1].")
    unknown = set(mask.dims) - set(data.dims)
    if unknown:
        raise ValueError(f"Mask dimensions are absent from data: {sorted(unknown)}.")
    try:
        mask, data = xr.align(mask, data, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("Mask coordinates must exactly match the noise data.") from exc
    mask_values = np.asarray(mask.values, dtype=float)
    if not np.all(np.isfinite(mask_values)):
        raise ValueError("Mask values must be finite.")
    inside = mask >= threshold
    region = inside if in_mask else ~inside
    return data.where(region.broadcast_like(data))


def _estimate_noise_method(
    data: xr.DataArray,
    *,
    method: str,
    reduce_dims: str | Sequence[str] | None,
    center: bool,
    selection: xr.DataArray | Sequence[int] | None,
    selection_dim: str | None,
    mask: xr.DataArray | None,
    mask_threshold: float,
    in_mask: bool,
    signal_domain: str | None,
    spectral_dim: str,
) -> _NoiseComputation:
    """Select one explicit source and execute the shared numerical estimator."""
    if method == "rf_off":
        selected, source = data, "rf_off"
    elif method == "complex_samples":
        selected, source = data, "complex_samples"
    elif method == "noise_frames":
        if selection is None or selection_dim is None:
            raise ValueError(
                "method='noise_frames' requires selection and selection_dim."
            )
        selected = _select_noise_frames(
            data,
            selection=selection,
            selection_dim=selection_dim,
        )
        source = "noise_frames"
    elif method == "outside_object":
        if mask is None:
            raise ValueError("method='outside_object' requires a labelled mask.")
        selected = _select_noise_mask(
            data,
            mask=mask,
            mask_threshold=mask_threshold,
            in_mask=in_mask,
        )
        source = "inside_mask" if in_mask else "outside_object"
    else:
        raise ValueError(
            "method must be 'rf_off', 'complex_samples', 'noise_frames', or "
            "'outside_object'. Calibrated outer-k-space selection is the "
            "separate function estimate_csi_noise."
        )
    return _estimate_noise_computation(
        selected,
        reduce_dims=reduce_dims,
        center=center,
        source=source,
        signal_domain=signal_domain,
        spectral_dim=spectral_dim,
    )


def _representative_sigma(noise: xr.DataArray) -> float:
    """Return one robust value for plot overlays when the estimate is a map."""
    values = np.asarray(noise.values, dtype=float)
    return float(np.median(values[np.isfinite(values)]))


def _plot_quadrature_histogram(ax, samples: np.ndarray, sigma: float) -> None:
    """Plot fitted Gaussian real/imaginary distributions and their parameters."""
    from scipy import stats

    bins = min(60, max(10, int(np.sqrt(samples.size))))
    real = np.real(samples)
    imag = np.imag(samples)
    mean_real, std_real = stats.norm.fit(real)
    mean_imag, std_imag = stats.norm.fit(imag)
    lower = min(float(np.min(real)), float(np.min(imag)), -3.5 * sigma)
    upper = max(float(np.max(real)), float(np.max(imag)), 3.5 * sigma)
    x = np.linspace(lower, upper, 300)
    ax.hist(real, bins=bins, density=True, alpha=0.35, color="tab:blue")
    ax.hist(imag, bins=bins, density=True, alpha=0.35, color="tab:orange")
    if std_real > 0:
        ax.plot(
            x,
            stats.norm.pdf(x, loc=mean_real, scale=std_real),
            color="tab:blue",
            label="Gaussian fit: real",
        )
        ax.axvline(std_real, color="tab:blue", linestyle="--")
        ax.axvline(-std_real, color="tab:blue", linestyle="--")
    if std_imag > 0:
        ax.plot(
            x,
            stats.norm.pdf(x, loc=mean_imag, scale=std_imag),
            color="tab:orange",
            label="Gaussian fit: imaginary",
        )
        ax.axvline(std_imag, color="tab:orange", linestyle="--")
        ax.axvline(-std_imag, color="tab:orange", linestyle="--")
    ax.set_xlabel("Quadrature amplitude")
    ax.set_ylabel("Density")
    ax.set_title("I/Q distributions + Gaussian fits")
    ax.legend(loc="lower left", fontsize="small")
    ax.text(
        0.98,
        0.98,
        (
            f"real: mean={mean_real:.4g}, std={std_real:.4g}\n"
            f"imag: mean={mean_imag:.4g}, std={std_imag:.4g}\n"
            f"noise sigma={sigma:.4g}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )


def _plot_complex_samples(ax, samples: np.ndarray, sigma: float) -> None:
    """Plot complex samples with the returned quadrature sigma as radius."""
    from matplotlib.patches import Circle

    step = max(1, int(np.ceil(samples.size / 4000)))
    shown = samples[::step]
    ax.scatter(np.real(shown), np.imag(shown), s=9, alpha=0.35)
    one_sigma = Circle(
        (0.0, 0.0),
        sigma,
        fill=False,
        color="tab:green",
        linestyle="-",
        label="noise sigma radius",
    )
    ax.add_patch(one_sigma)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Complex sample cloud")
    ax.legend(fontsize="small")


def _plot_not_applicable_to_magnitude(ax, title: str) -> None:
    """Mark a quadrature panel as undefined for magnitude-only samples."""
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        "Not defined for magnitude samples",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    ax.set_axis_off()


def _plot_sample_trace(ax, samples: np.ndarray, sigma: float, title: str) -> None:
    """Plot the first samples in acquisition order with a noise envelope."""
    shown = samples[:2000]
    index = np.arange(shown.size)
    ax.plot(index, np.real(shown), linewidth=0.8, label="real")
    ax.plot(index, np.imag(shown), linewidth=0.8, alpha=0.75, label="imaginary")
    ax.axhspan(-1.0 * sigma, 1.0 * sigma, color="0.7", alpha=0.4, label="+/- sigma")
    ax.set_xlabel("Selected sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(fontsize="small")


def _plot_magnitude_histogram(ax, samples: np.ndarray, sigma: float) -> None:
    """Plot magnitude samples and an unbinned, zero-location Rayleigh fit."""
    from scipy import stats

    magnitude = np.abs(samples)
    bins = min(60, max(10, int(np.sqrt(samples.size))))
    fit_location, fit_scale = stats.rayleigh.fit(magnitude, floc=0.0)
    fitted_mean = stats.rayleigh.mean(loc=fit_location, scale=fit_scale)
    fitted_std = stats.rayleigh.std(loc=fit_location, scale=fit_scale)
    sample_mean = float(np.mean(magnitude))
    sample_std = float(np.std(magnitude))
    ax.hist(magnitude, bins=bins, density=True, alpha=0.5, label="magnitude")
    upper = max(float(np.max(magnitude)), 4.0 * fit_scale, 4.0 * sigma)
    x = np.linspace(0.0, upper, 300)
    ax.plot(
        x,
        stats.rayleigh.pdf(x, loc=fit_location, scale=fit_scale),
        color="tab:red",
        label="Rayleigh fit (location=0)",
    )
    if not np.isclose(fit_scale, sigma):
        ax.plot(
            x,
            stats.rayleigh.pdf(x, loc=0.0, scale=sigma),
            color="black",
            linestyle="--",
            label="primary noise model",
        )
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Density")
    ax.set_title("Magnitude distribution + Rayleigh fit")
    ax.legend(fontsize="small")
    ax.text(
        0.98,
        0.98,
        (
            f"samples: mean={sample_mean:.4g}, std={sample_std:.4g}\n"
            f"fit: mean={fitted_mean:.4g}, std={fitted_std:.4g}\n"
            f"fit scale/noise={fit_scale:.4g}\n"
            f"primary noise sigma={sigma:.4g}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )


def _frame_selection_indices(
    data: xr.DataArray,
    selection: xr.DataArray | Sequence[int],
    selection_dim: str,
) -> np.ndarray:
    """Return normalized frame positions after estimator validation succeeded."""
    if isinstance(selection, xr.DataArray) and selection.dtype.kind == "b":
        indices = np.flatnonzero(np.asarray(selection.values))
    else:
        indices = np.asarray(selection, dtype=int)
    size = data.sizes[selection_dim]
    return np.where(indices < 0, indices + size, indices)


def _noise_fit_parameters(
    computation: _NoiseComputation,
    *,
    unit: str,
) -> xr.Dataset:
    """Return comparable parameters for every reported noise estimator."""
    processed = computation.processed
    dims = computation.reduction_dims
    count = computation.sample_count
    primary_sigma = computation.noise.reset_coords("noise_sample_count", drop=True)

    gaussian_mean_real = np.real(processed).mean(dim=dims, skipna=True)
    gaussian_mean_imag = np.imag(processed).mean(dim=dims, skipna=True)
    gaussian_sigma_real = np.real(processed).std(dim=dims, skipna=True)
    gaussian_sigma_imag = np.imag(processed).std(dim=dims, skipna=True)
    rayleigh_scale = np.sqrt(
        (np.abs(processed) ** 2).sum(dim=dims, skipna=True) / (2.0 * count)
    )
    rayleigh_mean = rayleigh_scale * np.sqrt(np.pi / 2.0)
    rayleigh_std = rayleigh_scale * np.sqrt((4.0 - np.pi) / 2.0)
    undefined = xr.full_like(rayleigh_scale, np.nan, dtype=float)
    methods = xr.IndexVariable(
        "fit_method",
        [
            "gaussian_real",
            "gaussian_imaginary",
            "rayleigh_magnitude",
            "complex_second_moment",
        ],
    )

    parameters = xr.Dataset(
        {
            "fit_location": xr.concat(
                [
                    gaussian_mean_real,
                    gaussian_mean_imag,
                    xr.zeros_like(rayleigh_scale),
                    undefined,
                ],
                dim=methods,
            ),
            "fit_scale": xr.concat(
                [
                    gaussian_sigma_real,
                    gaussian_sigma_imag,
                    rayleigh_scale,
                    undefined,
                ],
                dim=methods,
            ),
            "fit_distribution_mean": xr.concat(
                [
                    gaussian_mean_real,
                    gaussian_mean_imag,
                    rayleigh_mean,
                    undefined,
                ],
                dim=methods,
            ),
            "fit_distribution_std": xr.concat(
                [
                    gaussian_sigma_real,
                    gaussian_sigma_imag,
                    rayleigh_std,
                    undefined,
                ],
                dim=methods,
            ),
            "fit_noise_sigma": xr.concat(
                [
                    gaussian_sigma_real,
                    gaussian_sigma_imag,
                    rayleigh_scale,
                    primary_sigma,
                ],
                dim=methods,
            ),
            "fit_sample_count": xr.concat([count] * 4, dim=methods),
            "fit_is_primary": xr.DataArray(
                [False, False, False, True],
                dims="fit_method",
                coords={"fit_method": methods},
            ),
            "fit_formula": xr.DataArray(
                np.asarray(
                    [
                        "Normal(mean(real(z)), std(real(z), ddof=0))",
                        "Normal(mean(imag(z)), std(imag(z), ddof=0))",
                        "Rayleigh(location=0, scale=sqrt(sum(abs(z)**2)/(2*N)))",
                        "sqrt(sum(abs(z)**2)/(2*N))",
                    ],
                    dtype="U96",
                ),
                dims="fit_method",
                coords={"fit_method": methods},
            ),
        },
        attrs={
            "fit_parameter_contract_version": "1",
            "fit_parameter_samples": "processed_noise_samples",
            "rayleigh_location_policy": "fixed_at_zero",
            "gaussian_scale_policy": "population_standard_deviation_ddof_0",
        },
    )
    for name in (
        "fit_location",
        "fit_scale",
        "fit_distribution_mean",
        "fit_distribution_std",
        "fit_noise_sigma",
    ):
        parameters[name].attrs["unit"] = unit
    parameters["fit_location"].attrs["description"] = (
        "Gaussian mean or fixed Rayleigh location; undefined for second moment"
    )
    parameters["fit_scale"].attrs["description"] = (
        "Gaussian standard deviation or Rayleigh scale; undefined for second moment"
    )
    parameters["fit_noise_sigma"].attrs["description"] = (
        "Noise sigma implied by each fit or estimator"
    )
    parameters["fit_distribution_mean"].attrs["description"] = (
        "Mean implied by the fitted distribution"
    )
    parameters["fit_distribution_std"].attrs["description"] = (
        "Standard deviation implied by the fitted distribution"
    )
    parameters["fit_sample_count"].attrs["description"] = (
        "Finite processed samples used by the fit or estimator"
    )
    parameters["fit_is_primary"].attrs["description"] = (
        "True only for the estimator returned as noise_sigma_quadrature"
    )
    parameters["fit_formula"].attrs["description"] = (
        "Distribution model or estimator applied to the processed samples"
    )
    return parameters


def _noise_processing_pipeline(
    data: xr.DataArray,
    computation: _NoiseComputation,
    *,
    method: str,
    center: bool,
    selection_dim: str | None,
    mask_threshold: float,
    in_mask: bool,
    signal_domain: str | None,
    spectral_dim: str,
) -> xr.Dataset:
    """Build a labelled, inspectable record of the noise calculation."""
    noise = computation.noise
    selected = computation.selected
    processed = computation.processed
    dims = computation.reduction_dims
    count = computation.sample_count
    mean_real = np.real(selected).mean(dim=dims, skipna=True)
    mean_imag = np.imag(selected).mean(dim=dims, skipna=True)
    std_real = np.real(selected).std(dim=dims, skipna=True)
    std_imag = np.imag(selected).std(dim=dims, skipna=True)
    std_rms = np.sqrt((std_real**2 + std_imag**2) / 2.0)
    legacy_std_mean = (std_real + std_imag) / 2.0
    rayleigh_sigma = np.sqrt(
        (np.abs(processed) ** 2).sum(dim=dims, skipna=True) / (2.0 * count)
    )
    magnitude_mean = np.abs(processed).mean(dim=dims, skipna=True)

    domain = str(
        signal_domain
        if signal_domain is not None
        else data.attrs.get("signal_domain", "temporal")
    ).lower()
    if method == "noise_frames":
        selection_step = f"select explicit positions along {selection_dim}"
    elif method == "outside_object":
        region = "inside" if in_mask else "outside"
        selection_step = f"select {region} mask region at coverage threshold {float(mask_threshold):g}"
    elif method == "rf_off":
        selection_step = "use all finite samples from the RF-off reference"
    else:
        selection_step = "use all explicitly supplied complex samples"
    transform_step = (
        f"inverse FFT along {spectral_dim} into the temporal domain"
        if domain == "spectral"
        else "keep samples in the temporal domain"
    )
    center_step = (
        "subtract the complex mean along the reduction dimensions"
        if center
        else "retain the declared zero reference without mean subtraction"
    )
    sample_type = noise.attrs.get("sample_type", "complex")
    validation_step = (
        "validate labelled, non-negative magnitude input and embed it as m + 0j"
        if sample_type == "magnitude"
        else "validate labelled, complex-valued input"
    )
    steps = np.asarray(
        [
            validation_step,
            transform_step,
            selection_step,
            "exclude non-finite samples",
            center_step,
            "calculate sigma = sqrt(sum(abs(z)**2) / (2*N))",
            "calculate Gaussian, Rayleigh and complex-second-moment parameters",
        ],
        dtype="U160",
    )

    sigma = noise.reset_coords("noise_sample_count", drop=True)
    unit = data.attrs.get("unit", "a.u.")
    pipeline = xr.Dataset(
        {
            "noise_sigma_quadrature": sigma,
            "noise_sample_count": count,
            "sample_mean_real": mean_real,
            "sample_mean_imag": mean_imag,
            "sample_std_real": std_real,
            "sample_std_imag": std_imag,
            "noise_sigma_std_rms": std_rms,
            "noise_sigma_std_mean_legacy": legacy_std_mean,
            "noise_sigma_rayleigh_mle": rayleigh_sigma,
            "sample_magnitude_mean": magnitude_mean,
            "selected_noise_samples": selected,
            "processed_noise_samples": processed,
            "processing_description": xr.DataArray(
                steps,
                dims="processing_step",
                coords={"processing_step": np.arange(1, steps.size + 1)},
            ),
        },
        attrs={
            "noise_processing_pipeline_version": "2",
            "noise_method": method,
            "sample_type": sample_type,
            "input_signal_domain": domain,
            "estimation_signal_domain": "temporal",
            "spectral_transform_applied": domain == "spectral",
            "centered": bool(center),
            "reduction_dims": dims,
            "primary_estimator": noise.attrs["estimator"],
            "quadrature_convention": noise.attrs["quadrature_convention"],
            "unit": unit,
            "legacy_comparison": (
                "noise_sigma_std_mean_legacy is (std(real)+std(imag))/2 with "
                "population standard deviations; it is diagnostic only"
            ),
            "rayleigh_comparison": (
                "noise_sigma_rayleigh_mle is the closed-form Rayleigh scale MLE "
                "with location fixed at zero; no histogram curve fit is used"
            ),
        },
    )
    pipeline = pipeline.set_coords("noise_sample_count")
    for name in (
        "noise_sigma_quadrature",
        "sample_mean_real",
        "sample_mean_imag",
        "sample_std_real",
        "sample_std_imag",
        "noise_sigma_std_rms",
        "noise_sigma_std_mean_legacy",
        "noise_sigma_rayleigh_mle",
        "sample_magnitude_mean",
    ):
        pipeline[name].attrs["unit"] = unit
    pipeline["noise_sigma_quadrature"].attrs.update(noise.attrs)
    pipeline["noise_sigma_std_mean_legacy"].attrs["role"] = "diagnostic_only"
    pipeline["noise_sigma_rayleigh_mle"].attrs.update(
        role="diagnostic_only",
        rayleigh_location=0.0,
    )
    pipeline["selected_noise_samples"].attrs.update(
        stage="method_selection_and_finite_filter",
        signal_domain="temporal",
    )
    pipeline["processed_noise_samples"].attrs.update(
        stage="samples_entering_primary_estimator",
        signal_domain="temporal",
        centered=bool(center),
    )
    fit_parameters = _noise_fit_parameters(computation, unit=unit)
    pipeline = xr.merge([pipeline, fit_parameters], compat="identical")
    pipeline.attrs.update(fit_parameters.attrs)
    return pipeline


def _plot_frame_selection(
    ax,
    temporal: xr.DataArray,
    selection: xr.DataArray | Sequence[int],
    selection_dim: str,
) -> None:
    """Plot per-frame quadrature RMS and highlight the selected frames."""
    other_dims = tuple(dim for dim in temporal.dims if dim != selection_dim)
    energy = np.abs(temporal) ** 2
    if other_dims:
        energy = energy.mean(dim=other_dims, skipna=True)
    frame_sigma = np.sqrt(np.asarray(energy.values, dtype=float) / 2.0)
    positions = np.arange(temporal.sizes[selection_dim])
    selected = np.unique(_frame_selection_indices(temporal, selection, selection_dim))
    ax.plot(positions, frame_sigma, marker="o", markersize=3, label="all frames")
    ax.scatter(
        selected,
        frame_sigma[selected],
        color="tab:red",
        marker="o",
        label="selected",
        zorder=3,
    )
    coordinate = np.asarray(temporal.coords[selection_dim].values)
    if coordinate.size <= 12:
        ax.set_xticks(positions, [str(value) for value in coordinate])
    ax.set_xlabel(selection_dim)
    ax.set_ylabel("Per-quadrature RMS")
    ax.set_title("Noise-frame selection")
    ax.legend(fontsize="small")


def _representative_mask_slice(
    mask: xr.DataArray,
    data: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Reduce mask and a matching spatial noise map to at most two dimensions."""
    non_mask_dims = tuple(dim for dim in data.dims if dim not in mask.dims)
    energy = np.abs(data) ** 2
    if non_mask_dims:
        energy = energy.mean(dim=non_mask_dims, skipna=True)
    spatial_sigma = np.sqrt(energy / 2.0).transpose(*mask.dims)
    display_mask = mask
    for dim in mask.dims[:-2]:
        index = mask.sizes[dim] // 2
        display_mask = display_mask.isel({dim: index})
        spatial_sigma = spatial_sigma.isel({dim: index})
    return display_mask, spatial_sigma


def _plot_mask_selection(
    mask_ax,
    noise_ax,
    temporal: xr.DataArray,
    mask: xr.DataArray,
    *,
    threshold: float,
    in_mask: bool,
) -> None:
    """Plot the selected mask region and its representative spatial noise RMS."""
    display_mask, spatial_sigma = _representative_mask_slice(mask, temporal)
    mask_values = np.asarray(display_mask.values, dtype=float)
    selected = mask_values >= threshold
    if not in_mask:
        selected = ~selected
    region = "inside" if in_mask else "outside"

    if display_mask.ndim == 2:
        mask_image = mask_ax.imshow(mask_values, cmap="gray", origin="lower")
        mask_ax.figure.colorbar(mask_image, ax=mask_ax, shrink=0.75, label="Mask")
        overlay = np.ma.masked_where(~selected, np.ones_like(mask_values))
        mask_ax.imshow(
            overlay, cmap="autumn", origin="lower", alpha=0.35, vmin=0, vmax=1
        )
        noise_image = noise_ax.imshow(
            np.asarray(spatial_sigma.values, dtype=float),
            cmap="viridis",
            origin="lower",
        )
        noise_ax.figure.colorbar(
            noise_image,
            ax=noise_ax,
            shrink=0.75,
            label="Per-quadrature RMS",
        )
        if np.any(selected) and np.any(~selected):
            noise_ax.contour(selected.astype(float), levels=[0.5], colors="white")
    elif display_mask.ndim == 1:
        positions = np.arange(display_mask.size)
        mask_ax.plot(positions, mask_values, marker="o", markersize=3)
        mask_ax.axhline(threshold, color="black", linestyle="--", label="threshold")
        mask_ax.scatter(
            positions[selected],
            mask_values[selected],
            color="tab:red",
            label="selected",
            zorder=3,
        )
        mask_ax.legend(fontsize="small")
        noise_ax.plot(
            positions,
            np.asarray(spatial_sigma.values, dtype=float),
            marker="o",
            markersize=3,
        )
        noise_ax.scatter(
            positions[selected],
            np.asarray(spatial_sigma.values, dtype=float)[selected],
            color="tab:red",
            zorder=3,
        )
        mask_ax.set_xlabel(display_mask.dims[0])
        noise_ax.set_xlabel(display_mask.dims[0])
    else:
        mask_ax.text(0.5, 0.5, f"Mask value: {float(mask_values):.3g}", ha="center")
        noise_ax.text(
            0.5,
            0.5,
            f"RMS: {float(np.asarray(spatial_sigma.values)):.3g}",
            ha="center",
        )
    mask_ax.set_title(f"Mask and selected {region} region")
    noise_ax.set_title("Representative spatial noise RMS")


def _plot_noise_diagnostics(
    data: xr.DataArray,
    computation: _NoiseComputation,
    *,
    method: str,
    sample_type: str,
    selection: xr.DataArray | Sequence[int] | None,
    selection_dim: str | None,
    mask: xr.DataArray | None,
    mask_threshold: float,
    in_mask: bool,
    signal_domain: str | None,
    spectral_dim: str,
    figsize: tuple[float, float],
) -> None:
    """Display diagnostics tailored to the selected public noise method.

    Magnitude samples have no quadrature components, so the I/Q histogram and
    complex-cloud panels are replaced by a note for ``sample_type="magnitude"``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "estimate_noise(plot=True) requires matplotlib; install the "
            "project's plotting dependencies or call with plot=False."
        ) from exc

    noise = computation.noise
    temporal = computation.temporal
    values = np.asarray(computation.processed.values).ravel()
    samples = values[np.isfinite(values)]
    sigma = _representative_sigma(noise)
    figure, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    magnitude = sample_type == "magnitude"

    def quadrature_panel(ax) -> None:
        if magnitude:
            _plot_not_applicable_to_magnitude(ax, "I/Q distributions")
        else:
            _plot_quadrature_histogram(ax, samples, sigma)

    def cloud_panel(ax) -> None:
        if magnitude:
            _plot_not_applicable_to_magnitude(ax, "Complex sample cloud")
        else:
            _plot_complex_samples(ax, samples, sigma)

    if method == "rf_off":
        _plot_sample_trace(axes[0], samples, sigma, "RF-off sample stability")
        quadrature_panel(axes[1])
        _plot_magnitude_histogram(axes[2], samples, sigma)
        cloud_panel(axes[3])
    elif method == "complex_samples":
        cloud_panel(axes[0])
        _plot_sample_trace(axes[1], samples, sigma, "Complex sample stability")
        quadrature_panel(axes[2])
        _plot_magnitude_histogram(axes[3], samples, sigma)
    elif method == "noise_frames":
        assert selection is not None and selection_dim is not None
        full_temporal = _temporal_noise_data(
            data,
            signal_domain=signal_domain,
            spectral_dim=spectral_dim,
        )
        _plot_frame_selection(axes[0], full_temporal, selection, selection_dim)
        _plot_sample_trace(axes[1], samples, sigma, "Selected-frame samples")
        quadrature_panel(axes[2])
        _plot_magnitude_histogram(axes[3], samples, sigma)
    else:
        assert mask is not None
        _plot_mask_selection(
            axes[0],
            axes[1],
            temporal,
            mask,
            threshold=mask_threshold,
            in_mask=in_mask,
        )
        quadrature_panel(axes[2])
        _plot_magnitude_histogram(axes[3], samples, sigma)

    estimate_kind = "scalar" if noise.ndim == 0 else "labelled map"
    figure.suptitle(
        f"Noise diagnostics: {method} ({estimate_kind}, median sigma={sigma:.4g})"
    )
    plt.show()


def estimate_noise(
    data: xr.DataArray,
    *,
    method: str = "rf_off",
    reduce_dims: str | Sequence[str] | None = None,
    center: bool = False,
    selection: xr.DataArray | Sequence[int] | None = None,
    selection_dim: str | None = None,
    mask: xr.DataArray | None = None,
    mask_threshold: float = 0.5,
    in_mask: bool = False,
    signal_domain: str | None = None,
    spectral_dim: str = "acquisition_idx",
    plot: bool = False,
    return_pipeline: bool = False,
    figsize: tuple[float, float] = (12.0, 5.0),
    sample_type: str = "complex",
) -> xr.DataArray | xr.Dataset:
    """Estimate scalar or labelled per-quadrature spectroscopy noise.

    The method selects an explicit noise source and applies one shared estimator
    in the temporal domain. No FID-tail, residual or outer-k-space fallback is
    selected automatically; the calibrated outer-k-space source is the separate
    advanced function :func:`estimate_csi_noise`. With the default ``center=False``, coherent offsets remain
    visible in the second moment. Plotting and the processing Dataset reuse the
    same selected and transformed samples as the returned estimate.

    Parameters
    ----------
    data : xarray.DataArray
        Complex noise-reference samples with arbitrary labelled dimensions.
        Dimension order is unrestricted because reductions are name based.
        Values must have the same scaling and amplitude unit as the temporal FID
        samples used by the fit. The optional ``unit`` and ``signal_domain``
        attributes are read but the input is never mutated.
    method : {"rf_off", "complex_samples", "noise_frames", "outside_object"}, default="rf_off"
        Explicit source-selection method. ``rf_off`` uses an independent RF-off
        acquisition, ``complex_samples`` uses all supplied finite values,
        ``noise_frames`` selects positions along one dimension, and
        ``outside_object`` applies a labelled spatial mask.
    reduce_dims : str or sequence of str, optional
        Dimensions whose finite samples form each estimate. The default reduces
        every input dimension to one scalar. Dimensions omitted from this list
        are preserved, with their coordinates and direction, in a noise map.
    center : bool, default=False
        Subtract the complex mean along ``reduce_dims`` before estimation. Enable
        this only for a physically justified coherent receiver or DC offset;
        centering can otherwise hide signal contamination.
    selection : xarray.DataArray or sequence of int, optional
        Frame selection required by ``method="noise_frames"``. A boolean
        DataArray must have exactly ``(selection_dim,)`` and matching coordinates.
        Integer values are positional indices accepted by
        :meth:`xarray.DataArray.isel`.
    selection_dim : str, optional
        Labelled dimension along which ``selection`` is applied. Required for
        ``method="noise_frames"``.
    mask : xarray.DataArray, optional
        Finite real-valued mask required by ``method="outside_object"``. Its
        dimensions must be a subset of ``data.dims`` and shared coordinates
        must match exactly; missing dimensions are broadcast.
    mask_threshold : float, default=0.5
        Inclusive inside-mask coverage threshold in the interval ``[0, 1]``.
    in_mask : bool, default=False
        For ``outside_object``, select values at or above ``mask_threshold`` when
        true and values below the threshold otherwise.
    signal_domain : {"temporal", "spectral"}, optional
        Explicit domain of ``data``. If omitted, read the
        ``signal_domain`` attribute and fall back to ``"temporal"``. Spectral
        input is inverse Fourier transformed before estimation.
    spectral_dim : str, default="acquisition_idx"
        Dimension inverse Fourier transformed for spectral input.
    plot : bool, default=False
        Display method-specific Matplotlib diagnostics and call
        :func:`matplotlib.pyplot.show`. The Gaussian and Rayleigh panels pool the
        selected samples; for a noise map, overlays use its median sigma. No
        plotting dependency is imported when false.
    return_pipeline : bool, default=False
        Return a labelled processing Dataset instead of the bare noise
        DataArray. It contains the exact estimate, selected and centered samples,
        finite counts, processing descriptions, component statistics and
        diagnostic legacy/Rayleigh comparisons. Along ``fit_method``, the
        variables ``fit_location``, ``fit_scale``, ``fit_distribution_mean``,
        ``fit_distribution_std`` and ``fit_noise_sigma`` report the parameters
        for the real and imaginary Gaussian fits, the zero-location Rayleigh fit
        and the complex-second-moment estimator. ``fit_is_primary`` identifies
        the latter as the estimator used for the ordinary return value. This
        output can be as large as ``data``.
    figsize : tuple of float, default=(12.0, 5.0)
        Width and height in inches of the diagnostic figure. Used only when
        ``plot=True``; it has no effect on the estimate.
    sample_type : {"complex", "magnitude"}, default="complex"
        Kind of samples in ``data``. ``"complex"`` requires complex
        quadrature samples. ``"magnitude"`` accepts real-valued, non-negative
        magnitudes ``m = abs(z)`` of pure noise, for example the background of
        a magnitude image. The returned value is then the zero-location
        Rayleigh scale ``sqrt(sum(m**2)/(2*N))``, which estimates the same
        per-quadrature sigma as complex input. Magnitude input requires
        ``center=False`` and temporal (non-spectral) samples. Regions that
        contain signal follow a Rician, not a Rayleigh, distribution and bias
        the estimate upwards.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        By default, a float64 ``noise_sigma_quadrature`` DataArray in the input
        amplitude unit, with unreduced dimensions preserved and
        ``noise_sample_count`` attached as a coordinate. With
        ``return_pipeline=True``, a Dataset containing that identical DataArray
        and the documented intermediate values is returned instead.

    Raises
    ------
    TypeError
        If the input, frame selection or mask type is unsupported, the samples
        are not complex-valued for ``sample_type="complex"`` or not real-valued
        for ``sample_type="magnitude"``, or an unknown keyword argument is
        passed.
    ValueError
        If the method, domain, dimensions, mask, coordinates, selection or
        ``sample_type`` are invalid, if magnitude samples are negative or are
        combined with ``center=True`` or spectral input, or if any estimate
        lacks two finite samples and a finite positive value.
    ImportError
        If ``plot=True`` and Matplotlib cannot be imported.

    Notes
    -----
    For selected temporal samples ``z`` and ``center=False``, the estimator is
    ``sqrt(sum(abs(z)**2)/(2*N))``. With ``center=True``, ``z`` is replaced by
    ``z - mean(z)`` along the reduction dimensions. For
    ``sample_type="magnitude"`` the magnitudes ``m`` take the place of
    ``abs(z)``; because ``E[m**2] = 2*sigma**2`` for complex Gaussian noise,
    the formula is the maximum-likelihood Rayleigh scale. The result is
    deterministic.
    Validation eagerly reads the reduced estimate; spectral transformation and
    optional pipeline or plotting data may also materialize array values.

    Examples
    --------
    Estimate one RF-off noise value and inspect the optional processing record::

        noise = estimate_noise(rf_off_fids, method="rf_off")
        pipeline = estimate_noise(
            rf_off_fids,
            method="rf_off",
            plot=True,
            return_pipeline=True,
        )
        noise = pipeline["noise_sigma_quadrature"]
        fit_parameters = pipeline[
            [
                "fit_location",
                "fit_scale",
                "fit_distribution_mean",
                "fit_distribution_std",
                "fit_noise_sigma",
                "fit_formula",
                "fit_is_primary",
            ]
        ]

    Estimate the noise sigma from the background of a magnitude image::

        sigma = estimate_noise(
            magnitude_image,
            method="outside_object",
            mask=body_mask,
            sample_type="magnitude",
        )
    """
    normalized = str(method).strip().lower()
    normalized_sample_type = str(sample_type).strip().lower()
    data = _noise_samples_as_complex(
        data,
        sample_type=normalized_sample_type,
        center=center,
        signal_domain=signal_domain,
    )
    computation = _estimate_noise_method(
        data,
        method=normalized,
        reduce_dims=reduce_dims,
        center=center,
        selection=selection,
        selection_dim=selection_dim,
        mask=mask,
        mask_threshold=mask_threshold,
        in_mask=in_mask,
        signal_domain=signal_domain,
        spectral_dim=spectral_dim,
    )
    computation.noise.attrs["sample_type"] = normalized_sample_type
    if plot:
        _plot_noise_diagnostics(
            data,
            computation,
            method=normalized,
            sample_type=normalized_sample_type,
            selection=selection,
            selection_dim=selection_dim,
            mask=mask,
            mask_threshold=mask_threshold,
            in_mask=in_mask,
            signal_domain=signal_domain,
            spectral_dim=spectral_dim,
            figsize=figsize,
        )
    if return_pipeline:
        return _noise_processing_pipeline(
            data,
            computation,
            method=normalized,
            center=center,
            selection_dim=selection_dim,
            mask_threshold=mask_threshold,
            in_mask=in_mask,
            signal_domain=signal_domain,
            spectral_dim=spectral_dim,
        )
    return computation.noise


__all__ = [
    "ACQUISITION_NOISE_MODEL",
    "ACQUISITION_TAIL_NOISE_SOURCE",
    "AcquisitionTailNoise",
    "attach_acquisition_tail_noise",
    "estimate_acquisition_tail_noise",
    "estimate_csi_noise",
    "estimate_noise",
    "propagate_acquisition_tail_noise",
    "suggest_signal_end",
]
