"""Understand whether fitted spectroscopy peaks are trustworthy.

Most users need only two operations from this module:

1. describe measured noise with one of the ``make_*_noise_reference`` functions;
2. call :func:`evaluate_spectroscopy_fit_quality` and read
   ``peak_quality_class`` plus ``peak_quality_message``.

The smaller calculation, test-selection and decision functions remain public
for reproducible scientific inspection.  They all return labelled xarray data.
No quality decision masks, deletes or replaces a fitted peak.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _require_lorentz_lineshape,
)

from .fit_diagnostics import (
    calculate_spectroscopy_fit_diagnostics,
)
from .fit_diagnostics import (
    noise_sigma_for_fit as _noise_sigma_for_fit,
)

PEAK_QUALITY_VERSION = "1"

QUALITY_CLASSES = (
    "Q0_NOT_RELIABLY_DETECTED",
    "Q1_DETECTED_LIMITED",
    "Q2_LIKELY_QUANTIFIABLE",
)
QUALITY_STATUSES = (
    "noise_reference_missing",
    "noise_reference_rejected",
    "noise_model_unsupported",
    "fit_invalid",
    "incomplete_detection",
    "detection_not_significant",
    "detection_undecidable",
    "detected_full_m0_m1",
    "detected_fit_snr_fast_path",
)
QUALITY_STATUS_MESSAGES = {
    "noise_reference_missing": (
        "No trustworthy noise reference was supplied, so the peak was not "
        "assessed automatically."
    ),
    "noise_reference_rejected": (
        "The supplied noise reference did not pass its safety checks."
    ),
    "noise_model_unsupported": (
        "The noise reference is not compatible with the statistical test."
    ),
    "fit_invalid": "The numerical peak fit is not valid enough for assessment.",
    "incomplete_detection": (
        "This peak needs a full M0/M1 detection test, but no complete result is "
        "available yet."
    ),
    "detection_not_significant": (
        "The full M0/M1 test did not detect the peak at the configured alpha."
    ),
    "detection_undecidable": (
        "The full M0/M1 test could not produce a reliable decision."
    ),
    "detected_full_m0_m1": "The peak was detected by the full M0/M1 test.",
    "detected_fit_snr_fast_path": (
        "The peak passed the validated fit-SNR detection fast path."
    ),
}
DETECTION_BASES = ("none", "full_m0_m1", "fit_snr_fast_path")

QUALITY_FLAGS = {
    name: 1 << index
    for index, name in enumerate(
        (
            "FIT_INVALID",
            "FREQUENCY_BOUND_HIT",
            "T2_BOUND_HIT",
            "ILL_CONDITIONED",
            "CONDITION_OUTSIDE_VALIDATED_RANGE",
            "NOISE_REFERENCE_MISSING",
            "NOISE_REFERENCE_REJECTED",
            "NOISE_MODEL_UNSUPPORTED",
            "NOISE_FALLBACK_USED",
            "FULL_DETECTION_REQUIRED",
            "DETECTION_NOT_SIGNIFICANT",
            "DETECTION_UNDECIDABLE",
            "UNCERTAINTY_NOT_ASSESSED",
            "UNCERTAINTY_UNAVAILABLE",
            "UNCERTAINTY_INVALID",
            "UNCERTAINTY_LOW_RESOLUTION",
            "STABILITY_NOT_ASSESSED",
            "START_VALUE_INSTABILITY",
            "STRUCTURED_RESIDUAL_WARNING",
            "MODEL_ORDER_AMBIGUITY",
            "ATTRIBUTION_MODEL_DEPENDENT",
            "OUTSIDE_VALIDATED_DOMAIN",
        )
    )
}

DEFAULT_FORCE_FULL_DETECTION_FLAGS = (
    QUALITY_FLAGS["FREQUENCY_BOUND_HIT"] | QUALITY_FLAGS["T2_BOUND_HIT"]
)
DEFAULT_Q2_BLOCKING_FLAGS = sum(
    QUALITY_FLAGS[name]
    for name in (
        "FREQUENCY_BOUND_HIT",
        "T2_BOUND_HIT",
        "CONDITION_OUTSIDE_VALIDATED_RANGE",
        "UNCERTAINTY_NOT_ASSESSED",
        "UNCERTAINTY_UNAVAILABLE",
        "UNCERTAINTY_INVALID",
        "START_VALUE_INSTABILITY",
        "STRUCTURED_RESIDUAL_WARNING",
        "MODEL_ORDER_AMBIGUITY",
        "ATTRIBUTION_MODEL_DEPENDENT",
        "OUTSIDE_VALIDATED_DOMAIN",
    )
)

NOISE_SOURCE_PRIORITY = {
    "matched_rf_off": 0,
    "explicit_noise_frames": 1,
    "registered_outside_object": 2,
    "automatic_outer_kspace": 3,
    "known_exact": 0,
    "external_scalar": 4,
    "external_map": 4,
    "external_covariance": 4,
    "fid_tail": 100,
    "missing": 101,
}


_QUALITY_OPTION_DEFAULTS: dict[str, float | int] = {
    "detection_fit_snr": 10.0,
    "quantification_fit_snr": 20.0,
    "detection_alpha": 0.05,
    "hard_condition_max": 1e8,
    "validated_condition_warning": 20.0,
    "detection_null_samples": 39,
    "detection_seed": 202609100,
    "force_full_detection_flags": DEFAULT_FORCE_FULL_DETECTION_FLAGS,
    "q2_blocking_flags": DEFAULT_Q2_BLOCKING_FLAGS,
}
_QUALITY_REAL_OPTIONS = (
    "detection_fit_snr",
    "quantification_fit_snr",
    "detection_alpha",
    "hard_condition_max",
    "validated_condition_warning",
)
_ALL_QUALITY_FLAG_BITS = sum(QUALITY_FLAGS.values())


def _real_quality_option(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"Spectroscopy quality option '{name}' must be a real number, "
            f"got {value!r}."
        )
    return float(value)


def _integer_quality_option(name: str, value: object, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        requirement = "a positive" if minimum == 1 else "a non-negative"
        raise ValueError(f"{name} must be {requirement} integer, got {value!r}.")
    return int(value)


def _quality_flag_option(name: str, value: object) -> int:
    flags = _integer_quality_option(name, value, minimum=0)
    unknown_bits = flags & ~_ALL_QUALITY_FLAG_BITS
    if unknown_bits:
        raise ValueError(
            f"{name} contains bits {unknown_bits:#x} that are not in "
            "QUALITY_FLAGS; combine only QUALITY_FLAGS values."
        )
    return flags


def normalize_spectroscopy_quality_options(
    options: Mapping[str, float | int] | None = None,
) -> dict[str, float | int]:
    """Validate peak-quality decision settings and fill in the defaults.

    This function owns the defaults of the Q0/Q1/Q2 decision rule.  Every
    quality function passes its ``config`` argument through it, so a partial
    mapping, the complete normalized mapping and ``None`` are all valid inputs.

    Parameters
    ----------
    options : mapping of str to float or int, optional
        Settings that differ from the defaults.  Valid keys and defaults:

        ``detection_fit_snr`` (10.0)
            Conditional amplitude SNR below which a complete M0/M1 detection
            test is required.  At or above it, the provisional detection fast
            path may be used.  Must be positive.
        ``quantification_fit_snr`` (20.0)
            Conditional amplitude SNR at or above which Q2 is possible after
            every other Q2 gate.  Must be at least ``detection_fit_snr``.
        ``detection_alpha`` (0.05)
            Significance level of the M0/M1 detection test, strictly between
            zero and one.
        ``hard_condition_max`` (1e8)
            Basis condition number above which a fit is ill-conditioned (Q0).
        ``validated_condition_warning`` (20.0)
            Basis condition number above which the fit lies outside the
            validated range; Q2 is then blocked.  Must be positive and at most
            ``hard_condition_max``.
        ``detection_null_samples`` (39)
            Number of null simulations per M0/M1 test, a positive integer.
            The default reproduces the v4 regression grid, whose smallest
            p-value is 1/(39+1) = 0.025; it is not finer Monte Carlo precision.
        ``detection_seed`` (202609100)
            Non-negative integer base seed of the null simulations.
        ``force_full_detection_flags`` (``FREQUENCY_BOUND_HIT | T2_BOUND_HIT``)
            ``QUALITY_FLAGS`` bits that require a complete M0/M1 test
            regardless of SNR.
        ``q2_blocking_flags``
            ``QUALITY_FLAGS`` bits that prevent Q2.  The default blocks bound
            hits, conditioning outside the validated range, missing or invalid
            uncertainty, start-value instability, structured residuals, model
            order ambiguity, model-dependent attribution and inputs outside the
            validated domain.

        ``None`` or an empty mapping selects every default.

    Returns
    -------
    dict of str to float or int
        A new mapping with every key above.  Thresholds are ``float``; sample
        counts, seeds and flag masks are ``int``.  The input mapping is not
        modified.

    Raises
    ------
    TypeError
        If ``options`` is not a mapping or a threshold is not a real number.
    ValueError
        If a key is unknown, a threshold is not finite or violates the order
        constraints above, a count or seed is not an integer in its range, or
        a flag mask contains bits outside ``QUALITY_FLAGS``.

    Examples
    --------
    >>> from hypermri.analysis.spectroscopy import (
    ...     normalize_spectroscopy_quality_options,
    ... )
    >>> options = normalize_spectroscopy_quality_options({"detection_fit_snr": 8})
    >>> options["detection_fit_snr"], options["quantification_fit_snr"]
    (8.0, 20.0)

    The same mapping, or only its changed entries, is then passed as
    ``config`` to :func:`evaluate_spectroscopy_fit_quality`.
    """
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError(
            "Spectroscopy quality options must be a mapping of option names to "
            f"values, got {type(options).__name__}."
        )
    unknown = sorted(set(options) - set(_QUALITY_OPTION_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown spectroscopy quality option(s) {unknown}. Valid options "
            f"are {sorted(_QUALITY_OPTION_DEFAULTS)}."
        )
    merged = {**_QUALITY_OPTION_DEFAULTS, **options}
    normalized: dict[str, float | int] = {
        name: _real_quality_option(name, merged[name]) for name in _QUALITY_REAL_OPTIONS
    }
    if not np.all(np.isfinite(list(normalized.values()))):
        raise ValueError("Quality thresholds must be finite.")
    if normalized["detection_fit_snr"] <= 0:
        raise ValueError("detection_fit_snr must be positive.")
    if normalized["quantification_fit_snr"] < normalized["detection_fit_snr"]:
        raise ValueError("quantification_fit_snr must be at least detection_fit_snr.")
    if not 0 < normalized["detection_alpha"] < 1:
        raise ValueError("detection_alpha must lie strictly between zero and one.")
    if (
        not 0
        < normalized["validated_condition_warning"]
        <= normalized["hard_condition_max"]
    ):
        raise ValueError("Condition thresholds must be positive and ordered.")
    normalized["detection_null_samples"] = _integer_quality_option(
        "detection_null_samples", merged["detection_null_samples"], minimum=1
    )
    normalized["detection_seed"] = _integer_quality_option(
        "detection_seed", merged["detection_seed"], minimum=0
    )
    for name in ("force_full_detection_flags", "q2_blocking_flags"):
        normalized[name] = _quality_flag_option(name, merged[name])
    return normalized


def _quality_provenance_record(options: Mapping[str, float | int]) -> dict:
    """Return the normalized settings together with the rule version."""
    return {**options, "version": PEAK_QUALITY_VERSION}


def _quality_options_id(options: Mapping[str, float | int]) -> str:
    """Return the SHA-256 content hash of the settings and rule version."""
    encoded = json.dumps(
        _quality_provenance_record(options), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _resolve_quality_options(
    config: Mapping[str, float | int] | None,
) -> dict[str, float | int]:
    """Translate the ``config`` argument of a quality function to options."""
    return normalize_spectroscopy_quality_options(config)


def _string_scalar(value: str, width: int = 128) -> xr.DataArray:
    return xr.DataArray(np.asarray(str(value), dtype=f"U{width}"))


def _noise_reference_dataset(
    *,
    status: str,
    reason: str,
    source: str,
    sigma: float | xr.DataArray = np.nan,
    sigma_low: float | xr.DataArray = np.nan,
    sigma_high: float | xr.DataArray = np.nan,
    sample_count: int = 0,
    signal_fingerprint: str = "unavailable",
    reference_fingerprint: str = "unavailable",
    acquisition_compatible: bool = False,
    likelihood_compatible: bool = False,
    covariance: xr.DataArray | np.ndarray | None = None,
    unit: str = "a.u.",
    domain: str = "fitted_temporal_samples",
    provenance: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Build the stable labelled noise-reference result contract."""
    sigma = sigma if isinstance(sigma, xr.DataArray) else xr.DataArray(float(sigma))
    sigma_low = (
        sigma_low
        if isinstance(sigma_low, xr.DataArray)
        else xr.DataArray(float(sigma_low))
    )
    sigma_high = (
        sigma_high
        if isinstance(sigma_high, xr.DataArray)
        else xr.DataArray(float(sigma_high))
    )
    result = xr.Dataset(
        {
            "noise_reference_status": _string_scalar(status, 16),
            "noise_reference_reason": _string_scalar(reason, 128),
            "noise_source": _string_scalar(source, 32),
            "noise_source_priority": xr.DataArray(
                np.int16(NOISE_SOURCE_PRIORITY.get(source, 99))
            ),
            "noise_sigma_quadrature": sigma,
            "noise_sigma_low": sigma_low,
            "noise_sigma_high": sigma_high,
            "noise_sample_count": xr.DataArray(np.int64(sample_count)),
            "noise_signal_fingerprint": _string_scalar(signal_fingerprint, 128),
            "noise_reference_fingerprint": _string_scalar(reference_fingerprint, 128),
            "noise_acquisition_compatible": xr.DataArray(bool(acquisition_compatible)),
            "noise_likelihood_compatible": xr.DataArray(bool(likelihood_compatible)),
            "noise_fallback_used": xr.DataArray(False),
        },
        attrs={
            "noise_reference_contract_version": "1",
            "noise_sigma_unit": unit,
            "noise_domain": domain,
            "noise_sigma_convention": (
                "standard deviation per real or imaginary quadrature; "
                "E[abs(n)**2]=2*sigma**2"
            ),
            "noise_provenance": json.dumps(
                dict(provenance or {}), sort_keys=True, default=str
            ),
        },
    )
    if covariance is not None:
        if isinstance(covariance, xr.DataArray):
            result["noise_covariance"] = covariance
        else:
            covariance = np.asarray(covariance, dtype=np.complex128)
            result["noise_covariance"] = xr.DataArray(
                covariance,
                dims=("channel_idx", "channel_partner"),
            )
    result.noise_sigma_quadrature.attrs.update(
        unit=unit,
        definition="SD per real/imaginary quadrature",
    )
    result.noise_sigma_low.attrs["interval_scope"] = (
        "noise-reference sampling uncertainty only, when available"
    )
    result.noise_sigma_high.attrs.update(result.noise_sigma_low.attrs)
    descriptions = {
        "noise_reference_status": (
            "Whether this reference is estimated, missing, insufficient, or rejected."
        ),
        "noise_reference_reason": "Why the noise reference has its current status.",
        "noise_source": "Scientific source of the noise observations.",
        "noise_source_priority": (
            "Smaller values are preferred when several references are supplied."
        ),
        "noise_sample_count": "Number of explicit samples used for the estimate.",
        "noise_signal_fingerprint": "Identity of the signal data being assessed.",
        "noise_reference_fingerprint": "Identity of the separate noise data.",
        "noise_acquisition_compatible": (
            "Whether signal and reference acquisition metadata match."
        ),
        "noise_likelihood_compatible": (
            "Whether this noise model can be used by the current statistical test."
        ),
        "noise_fallback_used": (
            "True when a lower-priority usable source was selected."
        ),
    }
    for name, description in descriptions.items():
        result[name].attrs["description"] = description
    return result


def missing_spectroscopy_noise_reference(
    reason: str = "no_noise_reference_supplied",
) -> xr.Dataset:
    """Return an explicit missing result; never invent a sigma."""
    return _noise_reference_dataset(status="missing", reason=reason, source="missing")


def _metadata_compatible(
    signal: Mapping[str, object] | None,
    reference: Mapping[str, object] | None,
) -> bool:
    if not signal or not reference:
        return False
    if "compatibility_id" in signal or "compatibility_id" in reference:
        return bool(
            signal.get("compatibility_id")
            and signal.get("compatibility_id") == reference.get("compatibility_id")
        )
    keys = (
        "sampling_time_axis_s",
        "dwell_time_s",
        "channel_labels",
        "n_receivers",
        "reconstruction_model",
        "acquisition_model",
    )
    compared = 0
    for key in keys:
        if key not in signal or key not in reference:
            continue
        compared += 1
        if not np.array_equal(np.asarray(signal[key]), np.asarray(reference[key])):
            return False
    return compared > 0


def _select_explicit_samples(
    data: xr.DataArray,
    *,
    source: str,
    selection: xr.DataArray | np.ndarray | Sequence[int] | None,
    selection_dim: str | None,
) -> xr.DataArray:
    if source in {
        "matched_rf_off",
        "known_exact",
        "external_scalar",
        "external_map",
        "external_covariance",
    }:
        return data
    if selection is None:
        raise ValueError(f"{source} requires an explicit selection.")
    if isinstance(selection, xr.DataArray):
        if selection.dtype == bool:
            if selection.ndim == 0:
                raise ValueError(
                    "A boolean selection must name at least one dimension."
                )
            unknown_dims = set(selection.dims) - set(data.dims)
            if unknown_dims:
                raise ValueError(
                    "The selection uses dimensions that are absent from the noise "
                    f"data: {sorted(unknown_dims)}."
                )
            selection_axis = "__noise_selection"
            if selection_axis in data.dims:
                raise ValueError(
                    f"Reserved selection dimension '{selection_axis}' is in use."
                )
            selected_dims = list(selection.dims)
            for dim in selected_dims:
                if selection.sizes[dim] != data.sizes[dim]:
                    raise ValueError(
                        "The boolean selection size does not match the noise data "
                        f"along '{dim}'."
                    )
                if dim in selection.indexes and dim in data.indexes:
                    if not selection.indexes[dim].equals(data.indexes[dim]):
                        raise ValueError(
                            "The boolean selection coordinates do not match the "
                            f"noise data along '{dim}'."
                        )
            # CSI spatial coordinates can be curvilinear: a coordinate named
            # ``read_idx`` may itself depend on read, phase and slice.  A
            # PandasMultiIndex accepts only one-dimensional level variables,
            # so stack positions here while preserving the labelled auxiliary
            # coordinates on the selected data.
            stacked_data = data.stack(
                {selection_axis: selected_dims}, create_index=False
            )
            stacked_mask = selection.stack(
                {selection_axis: selected_dims}, create_index=False
            )
            stacked_mask, stacked_data = xr.align(
                stacked_mask, stacked_data, join="exact"
            )
            indices = np.flatnonzero(np.asarray(stacked_mask.values, dtype=bool))
            return stacked_data.isel({selection_axis: indices})
        if selection.ndim != 1:
            raise ValueError("Integer selections must be one-dimensional.")
        dim = selection_dim or selection.dims[0]
        return data.isel({dim: np.asarray(selection.values, dtype=int)})
    values = np.asarray(selection)
    if values.dtype == bool:
        if selection_dim is None:
            matching = [dim for dim in data.dims if data.sizes[dim] == values.size]
            if len(matching) != 1:
                raise ValueError(
                    "selection_dim is required when a boolean selection is ambiguous."
                )
            selection_dim = matching[0]
        return data.isel({selection_dim: values})
    if selection_dim is None:
        raise ValueError("selection_dim is required for integer selections.")
    return data.isel({selection_dim: values.astype(int)})


def estimate_explicit_noise_reference(
    reference_data: xr.DataArray | None = None,
    *,
    source: str,
    signal_fingerprint: str,
    reference_fingerprint: str,
    signal_acquisition: Mapping[str, object] | None = None,
    reference_acquisition: Mapping[str, object] | None = None,
    acquisition_compatible: bool | None = None,
    independent: bool = False,
    signal_free_attested: bool = False,
    selection: xr.DataArray | np.ndarray | Sequence[int] | None = None,
    selection_dim: str | None = None,
    registered: bool = False,
    sigma_quadrature: float | xr.DataArray | None = None,
    sigma_low: float | xr.DataArray = np.nan,
    sigma_high: float | xr.DataArray = np.nan,
    covariance: xr.DataArray | np.ndarray | None = None,
    trusted: bool = False,
    likelihood_compatible: bool | None = None,
    channel_dim: str = "channel_idx",
    unit: str = "a.u.",
    domain: str = "fitted_temporal_samples",
    provenance: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Estimate or register one explicitly identified noise source.

    The result is the audited noise-reference record consumed by
    :func:`resolve_spectroscopy_noise_reference` and
    :func:`evaluate_spectroscopy_fit_quality`. Each source has its own
    acceptance rule; a reference that fails it is returned with
    ``noise_reference_status="rejected"`` and the reason, never silently
    replaced by another source.

    Supported sources are ``matched_rf_off``, ``explicit_noise_frames``,
    ``registered_outside_object``, ``known_exact``, ``external_scalar``,
    ``external_map`` and ``external_covariance``.  Noise frames require explicit
    IDs/masks; outside-object samples require a registered mask and an expert
    signal-free attestation.  ``fid_tail`` is always rejected.
    ``automatic_outer_kspace`` is also rejected here because it requires the
    calibration contract of
    :func:`~hypermri.analysis.spectroscopy.noise_estimation.estimate_csi_noise`;
    pass that function's result to
    :func:`resolve_spectroscopy_noise_reference` instead.  Independent RF-off
    signal/reference fingerprints are intentionally kept distinct.

    Parameters
    ----------
    reference_data : xarray.DataArray, optional
        Complex noise samples on the scale and in the domain of the fitted
        samples. When given, sigma is estimated as ``sqrt(mean(abs(z)**2)/2)``
        without centering, per channel if ``channel_dim`` is present. Omit it
        to register a known or external ``sigma_quadrature``.
    source : str
        Noise source name. The aliases ``rf_off``, ``noise_frames`` and
        ``outside_object`` map to ``matched_rf_off``,
        ``explicit_noise_frames`` and ``registered_outside_object``.
        Matching is case-insensitive and ignores surrounding whitespace.
    signal_fingerprint : str
        Identity of the signal data the reference is used for.
    reference_fingerprint : str
        Identity of the noise data. For ``matched_rf_off`` it must differ from
        ``signal_fingerprint``.
    signal_acquisition, reference_acquisition : mapping, optional
        Acquisition descriptions compared when ``acquisition_compatible`` is
        not given, either through a shared ``compatibility_id`` or through the
        keys ``sampling_time_axis_s``, ``dwell_time_s``, ``channel_labels``,
        ``n_receivers``, ``reconstruction_model`` and ``acquisition_model``.
    acquisition_compatible : bool, optional
        Explicit compatibility decision that overrides the comparison above.
    independent : bool, default=False
        Attest that an RF-off reference is a separate acquisition.
        Required for ``matched_rf_off``.
    signal_free_attested : bool, default=False
        Expert attestation that selected frames or outside-object samples
        contain no signal. Required for ``explicit_noise_frames`` and
        ``registered_outside_object``.
    selection : xarray.DataArray, numpy.ndarray or sequence of int, optional
        Samples used for ``explicit_noise_frames`` and
        ``registered_outside_object``: a labelled boolean mask whose
        dimensions and coordinates match ``reference_data``, or a boolean or
        integer position selection along ``selection_dim``.
    selection_dim : str, optional
        Dimension of an unlabelled selection. It is required for integer
        selections and for ambiguous boolean selections.
    registered : bool, default=False
        Attest that an outside-object mask is registered to the reference grid.
        Required for ``registered_outside_object``.
    sigma_quadrature : float or xarray.DataArray, optional
        Known per-quadrature standard deviation in ``unit``. It is replaced by
        the estimate when ``reference_data`` is given.
    sigma_low, sigma_high : float or xarray.DataArray, default=nan
        Optional lower and upper sigma limits from the reference's own sampling
        uncertainty, in ``unit``.
    covariance : xarray.DataArray or numpy.ndarray, optional
        Known complex channel covariance ``E[n n^H]`` with dimensions
        ``(channel_idx, channel_partner)``. It is replaced by the estimate when
        ``reference_data`` is given.
    trusted : bool, default=False
        Attest that an ``external_*`` or ``known_exact`` value comes from a
        trusted origin. Required for those sources.
    likelihood_compatible : bool, optional
        Whether sigma describes the noise of the fitted samples. The default is
        true only for ``matched_rf_off``, ``explicit_noise_frames``,
        ``registered_outside_object`` and ``known_exact`` in the
        ``fitted_temporal_samples`` domain. External sources are rejected
        unless this is true.
    channel_dim : str, default="channel_idx"
        Receiver-channel dimension of ``reference_data``.
    unit : str, default="a.u."
        Amplitude unit of sigma, equal to the unit of the fitted samples.
    domain : str, default="fitted_temporal_samples"
        Signal domain that sigma describes.
    provenance : mapping, optional
        Origin record stored as JSON. External sources must provide
        ``origin``, ``unit``, ``quadrature_convention`` and ``domain``, with
        ``unit`` and ``domain`` equal to the arguments above and
        ``quadrature_convention`` either ``"per_real_imag"`` or
        ``"per_real_or_imag_quadrature"``.

    Returns
    -------
    xarray.Dataset
        Noise-reference record with the scalar variables
        ``noise_reference_status`` (``"estimated"`` or ``"rejected"``),
        ``noise_reference_reason``, ``noise_source``, ``noise_source_priority``
        (smaller is preferred), ``noise_sample_count``, fingerprints,
        compatibility flags and ``noise_fallback_used``; the variables
        ``noise_sigma_quadrature``, ``noise_sigma_low`` and
        ``noise_sigma_high`` (float64, scalar or labelled by ``channel_dim``,
        in ``unit``); and optionally ``noise_covariance``. Attributes record
        the contract version, unit, domain, quadrature convention and
        provenance.

    Raises
    ------
    TypeError
        If ``reference_data`` is given but is not a complex
        :class:`xarray.DataArray`.

    Notes
    -----
    Invalid selections, too few or non-finite samples and invalid sigma values
    produce a rejected record instead of an exception. The function is
    deterministic and does not mutate its inputs.
    """
    source = str(source).strip().lower()
    aliases = {
        "rf_off": "matched_rf_off",
        "noise_frames": "explicit_noise_frames",
        "outside_object": "registered_outside_object",
    }
    source = aliases.get(source, source)
    if source == "fid_tail":
        return _noise_reference_dataset(
            status="rejected",
            reason="fid_tail_forbidden",
            source=source,
            signal_fingerprint=signal_fingerprint,
            reference_fingerprint=reference_fingerprint,
            provenance=provenance,
        )
    if source == "automatic_outer_kspace":
        return _noise_reference_dataset(
            status="rejected",
            reason="automatic_outer_kspace_requires_estimate_csi_noise",
            source=source,
            signal_fingerprint=signal_fingerprint,
            reference_fingerprint=reference_fingerprint,
            provenance=provenance,
        )
    if source not in NOISE_SOURCE_PRIORITY:
        return _noise_reference_dataset(
            status="rejected",
            reason="unsupported_noise_source",
            source=source,
            signal_fingerprint=signal_fingerprint,
            reference_fingerprint=reference_fingerprint,
            provenance=provenance,
        )

    compatible = (
        _metadata_compatible(signal_acquisition, reference_acquisition)
        if acquisition_compatible is None
        else bool(acquisition_compatible)
    )
    reason = "validated_explicit_reference"
    accepted = True
    external = source.startswith("external_") or source == "known_exact"
    if source == "matched_rf_off":
        if not independent:
            accepted, reason = False, "rf_off_not_independent"
        elif signal_fingerprint == reference_fingerprint:
            accepted, reason = False, "rf_off_matches_signal_observations"
        elif not compatible:
            accepted, reason = False, "rf_off_acquisition_mismatch"
    elif source == "explicit_noise_frames":
        if not compatible:
            accepted, reason = False, "noise_frames_acquisition_mismatch"
        elif selection is None:
            accepted, reason = False, "explicit_noise_frame_ids_required"
        elif not signal_free_attested:
            accepted, reason = False, "noise_frames_not_attested_signal_free"
    elif source == "registered_outside_object":
        if not compatible:
            accepted, reason = False, "outside_object_acquisition_mismatch"
        elif selection is None or not registered:
            accepted, reason = False, "registered_outside_mask_required"
        elif not signal_free_attested:
            accepted, reason = False, "outside_object_not_attested_signal_free"
    elif external:
        required = {"origin", "unit", "quadrature_convention", "domain"}
        if not trusted or not required.issubset(provenance or {}):
            accepted, reason = False, "external_reference_provenance_incomplete"
        elif provenance["unit"] != unit or provenance["domain"] != domain:
            accepted, reason = False, "external_reference_unit_or_domain_mismatch"
        elif provenance["quadrature_convention"] not in {
            "per_real_imag",
            "per_real_or_imag_quadrature",
        }:
            accepted, reason = False, "external_quadrature_convention_unsupported"

    selected = reference_data
    sample_count = 0
    estimated_covariance = covariance
    if selected is not None:
        if not isinstance(selected, xr.DataArray) or not np.iscomplexobj(selected):
            raise TypeError("reference_data must be a complex xarray.DataArray.")
        try:
            selected = _select_explicit_samples(
                selected,
                source=source,
                selection=selection,
                selection_dim=selection_dim,
            )
        except ValueError as exc:
            accepted, reason = False, str(exc)
        if selected is not None:
            values = np.asarray(selected.values)
            if values.size < 2 or not np.all(np.isfinite(values)):
                accepted, reason = False, "insufficient_or_nonfinite_noise_samples"
            else:
                if channel_dim in selected.dims:
                    matrix = np.moveaxis(
                        values, selected.get_axis_num(channel_dim), -1
                    ).reshape(-1, selected.sizes[channel_dim])
                    estimated_covariance = np.einsum(
                        "sc,sd->cd", matrix, matrix.conj()
                    ) / len(matrix)
                    sample_count = int(matrix.shape[0])
                    sigma_quadrature = xr.DataArray(
                        np.sqrt(np.diag(estimated_covariance).real / 2),
                        dims=[channel_dim],
                        coords={channel_dim: selected.coords[channel_dim]},
                    )
                else:
                    flat = values.reshape(-1)
                    sample_count = int(flat.size)
                    sigma_quadrature = float(np.sqrt(np.mean(np.abs(flat) ** 2) / 2))
                    estimated_covariance = np.asarray(
                        [[2 * sigma_quadrature**2]], dtype=np.complex128
                    )

    if sigma_quadrature is None:
        accepted, reason = False, "noise_sigma_unavailable"
        sigma_quadrature = np.nan
    sigma_values = np.asarray(
        sigma_quadrature.values
        if isinstance(sigma_quadrature, xr.DataArray)
        else sigma_quadrature
    )
    if not np.all(np.isfinite(sigma_values)) or np.any(sigma_values <= 0):
        accepted, reason = False, "noise_sigma_invalid"

    if likelihood_compatible is None:
        likelihood_compatible = bool(
            domain == "fitted_temporal_samples"
            and source
            in {
                "matched_rf_off",
                "explicit_noise_frames",
                "registered_outside_object",
                "known_exact",
            }
        )
    if external and not likelihood_compatible:
        accepted, reason = False, "external_likelihood_compatibility_unverified"

    return _noise_reference_dataset(
        status="estimated" if accepted else "rejected",
        reason=reason,
        source=source,
        sigma=sigma_quadrature,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        sample_count=sample_count,
        signal_fingerprint=signal_fingerprint,
        reference_fingerprint=reference_fingerprint,
        acquisition_compatible=compatible,
        likelihood_compatible=bool(likelihood_compatible),
        covariance=estimated_covariance,
        unit=unit,
        domain=domain,
        provenance=provenance,
    )


def make_rf_off_noise_reference(
    reference_data: xr.DataArray,
    *,
    signal_fingerprint: str,
    reference_fingerprint: str,
    signal_acquisition: Mapping[str, object],
    reference_acquisition: Mapping[str, object],
    unit: str = "a.u.",
) -> xr.Dataset:
    """Describe a separate matching RF-off acquisition as the noise reference.

    This is the preferred entry point for scientists with an independently
    acquired RF-off scan.  The two fingerprints must identify different data.
    Sampling, channels, reconstruction and acquisition metadata are compared
    before the reference can be accepted.
    """
    return estimate_explicit_noise_reference(
        reference_data,
        source="matched_rf_off",
        signal_fingerprint=signal_fingerprint,
        reference_fingerprint=reference_fingerprint,
        signal_acquisition=signal_acquisition,
        reference_acquisition=reference_acquisition,
        independent=True,
        unit=unit,
        provenance={"origin": "independent_rf_off_acquisition"},
    )


def make_noise_frame_reference(
    reference_data: xr.DataArray,
    *,
    frame_ids: xr.DataArray | np.ndarray | Sequence[int],
    frame_dimension: str,
    signal_free_confirmed: bool,
    signal_fingerprint: str,
    reference_fingerprint: str,
    signal_acquisition: Mapping[str, object],
    reference_acquisition: Mapping[str, object],
    unit: str = "a.u.",
) -> xr.Dataset:
    """Use explicitly identified acquisition frames that contain only noise.

    Frames are never guessed from their position.  The caller must name the
    frame IDs and confirm that the selected frames are scientifically known to
    be signal-free.
    """
    return estimate_explicit_noise_reference(
        reference_data,
        source="explicit_noise_frames",
        signal_fingerprint=signal_fingerprint,
        reference_fingerprint=reference_fingerprint,
        signal_acquisition=signal_acquisition,
        reference_acquisition=reference_acquisition,
        selection=frame_ids,
        selection_dim=frame_dimension,
        signal_free_attested=signal_free_confirmed,
        unit=unit,
        provenance={
            "origin": "explicit_acquisition_noise_frames",
            "frame_dimension": frame_dimension,
        },
    )


def make_outside_object_noise_reference(
    reference_data: xr.DataArray,
    *,
    outside_object_mask: xr.DataArray | np.ndarray,
    mask_dimension: str | None,
    registration_confirmed: bool,
    signal_free_confirmed: bool,
    signal_fingerprint: str,
    reference_fingerprint: str,
    signal_acquisition: Mapping[str, object],
    reference_acquisition: Mapping[str, object],
    unit: str = "a.u.",
) -> xr.Dataset:
    """Use a registered and expertly confirmed signal-free outside region.

    A geometric mask alone is not enough.  Acceptance requires explicit
    confirmation of both registration and absence of relevant signal.
    """
    return estimate_explicit_noise_reference(
        reference_data,
        source="registered_outside_object",
        signal_fingerprint=signal_fingerprint,
        reference_fingerprint=reference_fingerprint,
        signal_acquisition=signal_acquisition,
        reference_acquisition=reference_acquisition,
        selection=outside_object_mask,
        selection_dim=mask_dimension,
        registered=registration_confirmed,
        signal_free_attested=signal_free_confirmed,
        unit=unit,
        provenance={"origin": "registered_outside_object_region"},
    )


def _normalize_noise_reference(reference: xr.Dataset) -> xr.Dataset:
    if "noise_reference_status" in reference:
        return reference.copy(deep=False)
    # A reference must name its source; an unlabelled Dataset is never read as
    # a calibrated outer-k-space estimate.
    source = str(reference.attrs.get("noise_source", "unidentified"))
    status = str(reference.attrs.get("noise_status", "insufficient"))
    reason = str(reference.attrs.get("noise_reason", "unavailable"))
    if "noise_source" not in reference.attrs:
        status, reason = "rejected", "noise_source_missing"
    sigma = reference.get("noise_sigma_quadrature", xr.DataArray(np.nan))
    count = reference.get("selected_point_count", xr.DataArray(0))
    count_value = int(np.nanmin(np.asarray(count.values))) if count.size else 0
    if source == "automatic_outer_kspace":
        # estimate_csi_noise owns the point minimum and records it; a fixed
        # floor here would contradict an explicitly requested minimum.
        minimum_points = reference.attrs.get("minimum_points")
        if reference.attrs.get("calibration_status") != "validated":
            status, reason = "rejected", "automatic_calibration_not_validated"
        elif not reference.attrs.get("raw_input_verified", False):
            status, reason = "rejected", "automatic_kspace_domain_not_verified"
        elif isinstance(minimum_points, bool) or not isinstance(
            minimum_points, (int, np.integer)
        ):
            status, reason = "rejected", "automatic_minimum_points_missing"
        elif count_value < minimum_points:
            status, reason = "insufficient", "automatic_too_few_selected_points"
        elif "group_noise_status" not in reference:
            status, reason = "rejected", "automatic_structure_checks_missing"
        elif not bool(
            np.all(np.asarray(reference.group_noise_status.values) == "estimated")
        ):
            status, reason = "rejected", "automatic_structure_or_stationarity_failed"
    normalized = _noise_reference_dataset(
        status="estimated" if status in {"estimated", "propagated"} else status,
        reason=reason,
        source=source,
        sigma=sigma,
        sigma_low=reference.get("noise_sigma_low", xr.DataArray(np.nan)),
        sigma_high=reference.get("noise_sigma_high", xr.DataArray(np.nan)),
        sample_count=count_value,
        signal_fingerprint=str(
            reference.attrs.get("signal_fingerprint", "unavailable")
        ),
        reference_fingerprint=str(
            reference.attrs.get("raw_fingerprint", "unavailable")
        ),
        acquisition_compatible=bool(
            reference.attrs.get("acquisition_compatible", status == "estimated")
        ),
        likelihood_compatible=bool(reference.attrs.get("likelihood_compatible", False)),
        covariance=reference.get("channel_covariance"),
        provenance={
            "noise_id": reference.attrs.get("noise_id", "unavailable"),
            "calibration_id": reference.attrs.get("calibration_id", "unavailable"),
        },
    )
    for name, value in reference.data_vars.items():
        if name not in normalized:
            normalized[name] = value
    return normalized


def resolve_spectroscopy_noise_reference(
    *references: xr.Dataset | Sequence[xr.Dataset] | None,
) -> xr.Dataset:
    """Resolve trusted references in deterministic scientific priority order.

    Each Dataset is first normalized to the noise-reference contract, then the
    references are sorted by ``NOISE_SOURCE_PRIORITY`` (smaller is preferred;
    the sort is stable, so equal priorities keep their input order).  The first
    usable reference is returned.  A source is usable only when it reports
    ``estimated``, a positive finite sigma and a likelihood compatible with the
    fitted temporal samples; ``fid_tail`` is never usable.  Rejected
    higher-priority sources are retained in ``noise_resolution_trace``.

    A reference with ``noise_source="automatic_outer_kspace"`` additionally
    needs a validated calibration, verified raw input, a ``minimum_points``
    attribute, at least ``minimum_points`` selected points in every group and
    the per-group status ``estimated``.  The point minimum is the one recorded
    by :func:`estimate_csi_noise`; there is no second default here.

    Parameters
    ----------
    *references : xarray.Dataset, sequence of xarray.Dataset or None
        Candidate references, either as separate arguments or as one sequence.
        Datasets already in the contract (they contain
        ``noise_reference_status``) are used as they are; others are read from
        the attributes and variables written by the estimators, such as
        ``noise_source``, ``noise_sigma_quadrature`` and
        ``selected_point_count``.  Values that are not Datasets, including
        ``None``, are ignored.  The inputs are not mutated.

    Returns
    -------
    xarray.Dataset
        The chosen reference in the noise-reference contract, including
        ``noise_reference_status``, ``noise_reference_reason``,
        ``noise_source``, ``noise_source_priority``, ``noise_sigma_quadrature``
        (standard deviation per real or imaginary quadrature, in the unit
        given by the ``noise_sigma_unit`` attribute) and
        ``noise_likelihood_compatible``.  ``noise_fallback_used`` is true only
        for a usable reference that is not of priority 0 or that follows an
        unusable higher-priority candidate.  The attribute ``noise_resolution_trace`` is a
        JSON list with the source, status, reason and usability of each
        examined candidate.  If no candidate is usable, the highest-priority
        candidate is returned with its non-``estimated`` status; without any
        Dataset the result of :func:`missing_spectroscopy_noise_reference` is
        returned.  The result is deterministic.

    Examples
    --------
    Prefer a matched RF-off reference and fall back to an outer-k-space
    estimate only when the RF-off reference is not usable::

        resolved = resolve_spectroscopy_noise_reference(rf_off_reference, csi_noise)
        resolved.noise_reference_status.item()
    """
    if len(references) == 1 and isinstance(references[0], Sequence):
        references = tuple(references[0])
    normalized = [
        _normalize_noise_reference(reference)
        for reference in references
        if isinstance(reference, xr.Dataset)
    ]
    if not normalized:
        return missing_spectroscopy_noise_reference()
    normalized.sort(key=lambda item: int(item.noise_source_priority.item()))
    trace = []
    chosen = None
    chosen_was_usable = False
    for candidate in normalized:
        source = str(candidate.noise_source.item())
        status = str(candidate.noise_reference_status.item())
        reason = str(candidate.noise_reference_reason.item())
        sigma = np.asarray(candidate.noise_sigma_quadrature.values)
        likelihood = bool(candidate.noise_likelihood_compatible.item())
        usable = bool(
            status == "estimated"
            and likelihood
            and np.all(np.isfinite(sigma))
            and np.all(sigma > 0)
            and source != "fid_tail"
        )
        trace.append(
            {"source": source, "status": status, "reason": reason, "usable": usable}
        )
        if usable:
            chosen = candidate
            chosen_was_usable = True
            break
    if chosen is None:
        chosen = normalized[0]
    chosen = chosen.copy(deep=False)
    chosen["noise_fallback_used"] = xr.DataArray(
        chosen_was_usable
        and (
            any(not item["usable"] for item in trace[:-1])
            or int(chosen.noise_source_priority.item()) > 0
        )
    )
    chosen.attrs = dict(chosen.attrs)
    chosen.attrs["noise_resolution_trace"] = json.dumps(trace, sort_keys=True)
    return chosen


def _broadcast_like(
    dataset: xr.Dataset,
    name: str,
    template: xr.DataArray,
    default,
    dtype=None,
) -> xr.DataArray:
    if name not in dataset:
        if isinstance(default, xr.DataArray):
            return default.broadcast_like(template).transpose(*template.dims)
        return xr.full_like(template, default, dtype=dtype)
    value = dataset[name]
    value = value.broadcast_like(template).transpose(*template.dims)
    return value.astype(dtype) if dtype is not None else value


def _detection_on_template(
    detection: xr.Dataset,
    name: str,
    template: xr.DataArray,
    default,
    dtype=None,
) -> xr.DataArray:
    if name not in detection:
        return xr.full_like(template, default, dtype=dtype)
    value = detection[name]
    if "detection_target" in value.dims:
        value = value.rename({"detection_target": "metabolite"}).reindex(
            metabolite=template.metabolite
        )
    elif "metabolite" not in value.dims:
        target = detection.attrs.get("detection_target")
        expanded = value.broadcast_like(template)
        if target is not None:
            value = xr.where(template.metabolite == target, expanded, default)
        else:
            value = expanded
    value = value.broadcast_like(template).transpose(*template.dims).fillna(default)
    return value.astype(dtype) if dtype is not None else value


def _decode_codes(values, *, labels):
    return np.asarray(np.asarray(labels)[np.asarray(values, dtype=int)])


def _flags_to_reasons(value):
    value = int(value)
    names = [name for name, bit in QUALITY_FLAGS.items() if value & bit]
    return np.asarray("|".join(names) if names else "NONE", dtype="U512")


def _status_to_message(value):
    return np.asarray(
        QUALITY_STATUS_MESSAGES.get(str(value), "Unknown assessment status."),
        dtype="U192",
    )


def _add_flag(flags: xr.DataArray, mask: xr.DataArray, name: str) -> xr.DataArray:
    return flags | xr.where(mask, QUALITY_FLAGS[name], 0).astype(np.uint32)


def classify_spectroscopy_fit_quality(
    fit_results: xr.Dataset,
    noise_reference: xr.Dataset,
    diagnostics: xr.Dataset,
    detection_results: xr.Dataset | None = None,
    optional_diagnostics: xr.Dataset | None = None,
    config: Mapping[str, float | int] | None = None,
) -> xr.Dataset:
    """Purely classify every labelled spectrum/metabolite pair as Q0/Q1/Q2.

    Existing complete M0/M1 outcomes override the SNR fast path.  Missing,
    rejected or likelihood-incompatible noise and technically invalid M1 fits
    block automatic assessment.  All returned classes are post-fit annotations;
    raw fit variables and the caller's datasets remain unchanged.  The function
    is deterministic and runs no fit or simulation.

    Parameters
    ----------
    fit_results : xr.Dataset
        Unfiltered M1 fit results with ``fit_amps`` labelled by the spectrum
        dimensions and ``metabolite``, plus ``fit_status``,
        ``fit_failure_reason``, ``fit_skipped`` and ``fit_quality``.
    noise_reference : xr.Dataset
        Resolved noise reference, for example from
        :func:`resolve_spectroscopy_noise_reference`.
    diagnostics : xr.Dataset
        Post-fit diagnostics from :func:`calculate_spectroscopy_fit_diagnostics`
        (conditional amplitude SNR, basis condition number and rank, bound
        hits).
    detection_results : xr.Dataset, optional
        Full M0/M1 test records with ``detection_status`` and
        ``detection_p_value`` over the spectrum dimensions and
        ``detection_target``.  Defaults to ``fit_results``, which then supplies
        no completed tests unless it contains these variables.
    optional_diagnostics : xr.Dataset, optional
        Boolean uncertainty, stability, residual and model-order diagnostics on
        the ``fit_amps`` grid.  Missing variables count as not assessed.
    config : mapping of str to float or int, optional
        Quality options that differ from the defaults, with the keys and ranges
        of :func:`normalize_spectroscopy_quality_options`.  ``None`` selects
        every default.

    Returns
    -------
    xr.Dataset
        On the ``fit_amps`` grid: ``peak_quality_class`` (one of
        ``QUALITY_CLASSES``), ``peak_quality_status`` (one of
        ``QUALITY_STATUSES``), ``peak_quality_message``,
        ``peak_quality_flags`` (``uint32`` bits of ``QUALITY_FLAGS``),
        ``peak_quality_reasons``, ``peak_detection_basis`` and
        ``full_detection_test_required``.  The attributes record
        ``peak_quality_version``, the normalized options as JSON in
        ``peak_quality_config`` and their SHA-256 hash in
        ``peak_quality_config_id``.

    Raises
    ------
    TypeError, ValueError
        If ``config`` is not a valid set of quality options.
    """
    options = _resolve_quality_options(config)
    optional = (
        optional_diagnostics if optional_diagnostics is not None else xr.Dataset()
    )
    detection = detection_results if detection_results is not None else fit_results
    template = fit_results.fit_amps
    flags = xr.zeros_like(template.real, dtype=np.uint32)

    noise_status = str(noise_reference.noise_reference_status.item())
    noise_likelihood = bool(noise_reference.noise_likelihood_compatible.item())
    sigma = _noise_sigma_for_fit(noise_reference, template).broadcast_like(template)
    sigma_valid = np.isfinite(sigma) & (sigma > 0)
    noise_missing = noise_status == "missing"
    noise_rejected = noise_status in {"rejected", "insufficient"}
    noise_unsupported = noise_status == "estimated" and not noise_likelihood
    flags = _add_flag(
        flags,
        xr.full_like(template.real, noise_missing, dtype=bool),
        "NOISE_REFERENCE_MISSING",
    )
    flags = _add_flag(
        flags,
        xr.full_like(template.real, noise_rejected, dtype=bool),
        "NOISE_REFERENCE_REJECTED",
    )
    flags = _add_flag(
        flags,
        xr.full_like(template.real, noise_unsupported, dtype=bool),
        "NOISE_MODEL_UNSUPPORTED",
    )
    flags = _add_flag(
        flags,
        xr.full_like(
            template.real,
            bool(noise_reference.noise_fallback_used.item()),
            dtype=bool,
        ),
        "NOISE_FALLBACK_USED",
    )
    noise_ok = (noise_status == "estimated") & noise_likelihood & sigma_valid

    fit_status = _broadcast_like(
        fit_results, "fit_status", template, "fit_failed", np.dtype("U32")
    )
    skipped = _broadcast_like(fit_results, "fit_skipped", template, True, bool)
    condition = diagnostics.basis_condition_number.broadcast_like(template)
    rank = diagnostics.basis_rank.broadcast_like(template)
    active_count = (~skipped).sum("metabolite").broadcast_like(template)
    finite_fit = (
        np.isfinite(fit_results.fit_amps.real)
        & np.isfinite(fit_results.fit_amps.imag)
        & np.isfinite(fit_results.fit_freqs)
        & np.isfinite(fit_results.fit_t2s)
    )
    ill_conditioned = (
        ~np.isfinite(condition)
        | (condition > options["hard_condition_max"])
        | (rank < active_count)
    )
    fit_invalid = (
        ~fit_status.isin(["success", "zero_signal"])
        | skipped
        | ~finite_fit
        | ill_conditioned
        | (noise_ok & ~np.isfinite(diagnostics.amplitude_snr_conditional))
    )
    flags = _add_flag(flags, fit_invalid, "FIT_INVALID")
    flags = _add_flag(flags, ill_conditioned, "ILL_CONDITIONED")
    frequency_bound = diagnostics.frequency_bound_hit.astype(bool)
    t2_bound = diagnostics.t2_bound_hit.astype(bool)
    flags = _add_flag(flags, frequency_bound, "FREQUENCY_BOUND_HIT")
    flags = _add_flag(flags, t2_bound, "T2_BOUND_HIT")
    flags = _add_flag(
        flags,
        (condition > options["validated_condition_warning"])
        & (condition <= options["hard_condition_max"]),
        "CONDITION_OUTSIDE_VALIDATED_RANGE",
    )

    uncertainty_assessed = _broadcast_like(
        optional, "uncertainty_assessed", template, False, bool
    )
    uncertainty_available = _broadcast_like(
        optional, "uncertainty_available", template, uncertainty_assessed, bool
    )
    uncertainty_valid = _broadcast_like(
        optional, "uncertainty_valid", template, False, bool
    )
    flags = _add_flag(flags, ~uncertainty_assessed, "UNCERTAINTY_NOT_ASSESSED")
    flags = _add_flag(
        flags,
        uncertainty_assessed & ~uncertainty_available,
        "UNCERTAINTY_UNAVAILABLE",
    )
    flags = _add_flag(
        flags,
        uncertainty_assessed & uncertainty_available & ~uncertainty_valid,
        "UNCERTAINTY_INVALID",
    )
    low_resolution = _broadcast_like(
        optional, "uncertainty_low_resolution", template, False, bool
    )
    if "uncertainty_effective_rank_mass" in optional:
        low_resolution = low_resolution | (
            _broadcast_like(
                optional,
                "uncertainty_effective_rank_mass",
                template,
                np.nan,
                float,
            )
            < 0.948
        )
    flags = _add_flag(flags, low_resolution, "UNCERTAINTY_LOW_RESOLUTION")

    stability_assessed = _broadcast_like(
        optional, "stability_assessed", template, False, bool
    )
    flags = _add_flag(flags, ~stability_assessed, "STABILITY_NOT_ASSESSED")
    for variable, flag_name in (
        ("start_value_instability", "START_VALUE_INSTABILITY"),
        ("structured_residual_warning", "STRUCTURED_RESIDUAL_WARNING"),
        ("model_order_ambiguity", "MODEL_ORDER_AMBIGUITY"),
        ("attribution_model_dependent", "ATTRIBUTION_MODEL_DEPENDENT"),
        ("outside_validated_domain", "OUTSIDE_VALIDATED_DOMAIN"),
    ):
        flags = _add_flag(
            flags,
            _broadcast_like(optional, variable, template, False, bool),
            flag_name,
        )

    snr = diagnostics.amplitude_snr_conditional
    force_by_flag = (flags & np.uint32(options["force_full_detection_flags"])) != 0
    full_required = (snr < options["detection_fit_snr"]) | force_by_flag
    flags = _add_flag(flags, full_required, "FULL_DETECTION_REQUIRED")

    test_status = _detection_on_template(
        detection,
        "detection_status",
        template,
        "not_tested",
        np.dtype("U32"),
    )
    test_p = _detection_on_template(
        detection, "detection_p_value", template, np.nan, float
    )
    test_present = test_status != "not_tested"
    test_positive = (
        (test_status == "detected")
        & np.isfinite(test_p)
        & (test_p <= options["detection_alpha"])
    )
    test_negative = test_status.isin(["detected", "not_detected"]) & (
        np.isfinite(test_p) & (test_p > options["detection_alpha"])
    )
    test_undecidable = test_present & ~test_positive & ~test_negative
    flags = _add_flag(flags, test_negative, "DETECTION_NOT_SIGNIFICANT")
    flags = _add_flag(flags, test_undecidable, "DETECTION_UNDECIDABLE")

    core_valid = noise_ok & ~fit_invalid
    incomplete = core_valid & full_required & ~test_present
    fast_detected = (
        core_valid
        & ~full_required
        & ~test_present
        & (snr >= options["detection_fit_snr"])
    )
    detected = core_valid & (test_positive | fast_detected)
    q2_blocked = (flags & np.uint32(options["q2_blocking_flags"])) != 0
    q2 = (
        detected
        & (snr >= options["quantification_fit_snr"])
        & ~q2_blocked
        & uncertainty_assessed
        & uncertainty_available
        & uncertainty_valid
    )
    class_code = xr.where(q2, 2, xr.where(detected, 1, 0)).astype(np.uint8)

    status_code = xr.full_like(class_code, 4, dtype=np.uint8)
    status_code = xr.where(noise_missing, 0, status_code)
    status_code = xr.where(noise_rejected, 1, status_code)
    status_code = xr.where(noise_unsupported, 2, status_code)
    status_code = xr.where(noise_ok & fit_invalid, 3, status_code)
    status_code = xr.where(core_valid & test_negative, 5, status_code)
    status_code = xr.where(core_valid & test_undecidable, 6, status_code)
    status_code = xr.where(incomplete, 4, status_code)
    status_code = xr.where(core_valid & test_positive, 7, status_code)
    status_code = xr.where(fast_detected, 8, status_code).astype(np.uint8)

    basis_code = xr.full_like(class_code, 0, dtype=np.uint8)
    basis_code = xr.where(test_present, 1, basis_code)
    basis_code = xr.where(fast_detected, 2, basis_code).astype(np.uint8)
    quality = xr.Dataset(
        {
            "peak_quality_class": xr.apply_ufunc(
                _decode_codes,
                class_code,
                kwargs={"labels": np.asarray(QUALITY_CLASSES, dtype="U32")},
                dask="parallelized",
                output_dtypes=[np.dtype("U32")],
            ),
            "peak_quality_status": xr.apply_ufunc(
                _decode_codes,
                status_code,
                kwargs={"labels": np.asarray(QUALITY_STATUSES, dtype="U32")},
                dask="parallelized",
                output_dtypes=[np.dtype("U32")],
            ),
            "peak_quality_flags": flags.astype(np.uint32),
            "peak_quality_reasons": xr.apply_ufunc(
                _flags_to_reasons,
                flags,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.dtype("U512")],
            ),
            "peak_detection_basis": xr.apply_ufunc(
                _decode_codes,
                basis_code,
                kwargs={"labels": np.asarray(DETECTION_BASES, dtype="U32")},
                dask="parallelized",
                output_dtypes=[np.dtype("U32")],
            ),
            "full_detection_test_required": full_required.astype(bool),
        }
    )
    quality["peak_quality_message"] = xr.apply_ufunc(
        _status_to_message,
        quality.peak_quality_status,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.dtype("U192")],
    )
    quality.peak_quality_class.attrs["codes"] = json.dumps(
        dict(enumerate(QUALITY_CLASSES)), sort_keys=True
    )
    quality.peak_quality_class.attrs["description"] = (
        "Plain three-level conclusion for this spectrum and metabolite."
    )
    quality.peak_quality_status.attrs["codes"] = json.dumps(
        dict(enumerate(QUALITY_STATUSES)), sort_keys=True
    )
    quality.peak_quality_status.attrs["description"] = (
        "Reason the assessment reached its class, kept separate from Q0/Q1/Q2."
    )
    quality.peak_quality_message.attrs["description"] = (
        "Human-readable explanation of peak_quality_status."
    )
    quality.peak_detection_basis.attrs["codes"] = json.dumps(
        dict(enumerate(DETECTION_BASES)), sort_keys=True
    )
    quality.peak_detection_basis.attrs["description"] = (
        "Evidence used for detection: no decision, full M0/M1 test, or fit-SNR."
    )
    quality.peak_quality_reasons.attrs["description"] = (
        "All active named warning flags, separated by '|'; NONE means no flag."
    )
    quality.full_detection_test_required.attrs["description"] = (
        "Whether the scientific rule requires a full M0/M1 test for this pair."
    )
    quality.peak_quality_flags.attrs.update(
        bit_flags=json.dumps(QUALITY_FLAGS, sort_keys=True),
        q0_forcing=(
            "FIT_INVALID, ILL_CONDITIONED, noise-blocking flags and negative/"
            "undecidable/incomplete detection according to peak_quality_status"
        ),
        q2_blocking=json.dumps(
            [
                name
                for name, bit in QUALITY_FLAGS.items()
                if options["q2_blocking_flags"] & bit
            ]
        ),
        informational=(
            "NOISE_FALLBACK_USED, UNCERTAINTY_LOW_RESOLUTION and "
            "STABILITY_NOT_ASSESSED do not by themselves lower a detected peak"
        ),
    )
    quality.attrs.update(
        peak_quality_version=PEAK_QUALITY_VERSION,
        peak_quality_config_id=_quality_options_id(options),
        peak_quality_config=json.dumps(
            _quality_provenance_record(options), sort_keys=True
        ),
        detection_fit_snr_description=(
            "Below this value a complete M0/M1 test is required; at or above it "
            "the provisional detection fast path may be used."
        ),
        quantification_fit_snr_description=(
            "At or above this separate value Q2 is possible only after every Q2 gate."
        ),
        uncertainty_scope=(
            "conditional_on_fitted_peak; conditional amplitude SD is not a "
            "validated confidence interval"
        ),
        how_to_read_peak_quality=(
            "Read peak_quality_class for the conclusion, peak_quality_message "
            "for a plain explanation, and peak_quality_reasons for every active "
            "technical warning. Raw fit values remain in fit_amps, fit_freqs and "
            "fit_t2s."
        ),
    )
    parts = [fit_results.copy(deep=False), diagnostics.copy(deep=False), quality]
    if detection_results is not None:
        parts.append(detection_results.copy(deep=False))
    parts.append(noise_reference.copy(deep=False))
    for name in optional.data_vars:
        if name not in {item for part in parts for item in part.data_vars}:
            parts.append(optional[[name]].copy(deep=False))
    result = xr.merge(parts, compat="override", join="outer")
    result.attrs.update(fit_results.attrs)
    result.attrs.update(quality.attrs)
    return result


def select_full_detection_test_mask(
    fit_results: xr.Dataset,
    noise_reference: xr.Dataset,
    diagnostics: xr.Dataset,
    detection_results: xr.Dataset | None = None,
    optional_diagnostics: xr.Dataset | None = None,
    config: Mapping[str, float | int] | None = None,
) -> xr.DataArray:
    """Return pairs which still require an executable full M0/M1 test.

    The mask is derived by the same pure classifier used for the final result.
    It excludes invalid fits, unusable noise references and already terminal
    detection outcomes, even when a threshold or boundary flag would otherwise
    require a full test.

    Parameters
    ----------
    fit_results, noise_reference, diagnostics, detection_results, optional_diagnostics
        Inputs of :func:`classify_spectroscopy_fit_quality` with the same
        meaning.
    config : mapping of str to float or int, optional
        Quality options that differ from the defaults, with the keys and ranges
        of :func:`normalize_spectroscopy_quality_options`.  ``None`` selects
        every default.

    Returns
    -------
    xr.DataArray
        Boolean ``full_detection_test_selected`` on the ``fit_amps`` grid.  Its
        ``selection_rule_version`` attribute is ``PEAK_QUALITY_VERSION``.

    Raises
    ------
    TypeError, ValueError
        If ``config`` is not a valid set of quality options.
    """
    classified = classify_spectroscopy_fit_quality(
        fit_results,
        noise_reference,
        diagnostics,
        detection_results=detection_results,
        optional_diagnostics=optional_diagnostics,
        config=config,
    )
    selected = classified.full_detection_test_required & (
        classified.peak_quality_status == "incomplete_detection"
    )
    selected.name = "full_detection_test_selected"
    selected.attrs.update(
        description=(
            "True only for valid spectrum/metabolite pairs which require and do "
            "not yet have a terminal full M0/M1 detection result."
        ),
        selection_rule_version=PEAK_QUALITY_VERSION,
    )
    return selected


def _strip_detection_settings(fit_params: xr.Dataset) -> xr.Dataset:
    """Return an unfiltered M1 parameter copy without detection settings."""
    result = fit_params.copy(deep=False)
    detection_vars = [
        name
        for name in result.data_vars
        if name.startswith("detection_") or name.startswith("detection_setting_")
    ]
    if detection_vars:
        result = result.drop_vars(detection_vars)
    attrs = {
        key: value
        for key, value in result.attrs.items()
        if not key.startswith("detection_")
    }
    attrs["snr_threshold"] = None
    result.attrs = attrs
    return result


_DETECTION_SCALAR_RENAMES = {
    "rss_m0": "detection_rss_m0",
    "rss_m1": "detection_rss_m1",
    "rss_m1_initial": "detection_rss_m1_initial",
    "test_statistic_initial": "detection_test_statistic_initial",
    "test_statistic_raw": "detection_test_statistic_raw",
    "detection_p_value": "detection_p_value",
    "detection_mc_tail_low": "detection_mc_tail_low",
    "detection_mc_tail_high": "detection_mc_tail_high",
    "detection_status": "detection_status",
    "detection_reason": "detection_reason",
    "detection_n_null": "detection_n_null",
    "detection_n_valid_null": "detection_n_valid_null",
    "detection_n_extreme": "detection_n_extreme",
    "detection_n_failed_null": "detection_n_failed_null",
    "detection_n_recovered_null": "detection_n_recovered_null",
    "detection_stream": "detection_stream",
    "detection_alpha": "detection_alpha",
    "noise_sigma_quadrature": "detection_noise_sigma_quadrature",
    "calibration_id": "detection_calibration_id",
    "m0_fit_status": "detection_m0_fit_status",
    "m0_fit_failure_reason": "detection_m0_fit_failure_reason",
    "m1_recovery_attempted": "detection_m1_recovery_attempted",
    "m1_recovery_used": "detection_m1_recovery_used",
    "fit_status": "detection_m1_fit_status",
    "fit_failure_reason": "detection_m1_fit_failure_reason",
    "fit_rss": "detection_m1_fit_rss",
}

_DETECTION_MODEL_RENAMES = {
    "m0_amps": "detection_m0_amps",
    "m0_freqs": "detection_m0_freqs",
    "m0_t2s": "detection_m0_t2s",
    "fit_amps": "detection_m1_amps",
    "fit_freqs": "detection_m1_freqs",
    "fit_t2s": "detection_m1_t2s",
    "fit_skipped": "detection_m1_skipped",
    "amplitude_std_quadrature_conditional": (
        "detection_m1_amplitude_std_quadrature_conditional"
    ),
    "amplitude_snr_conditional": "detection_m1_amplitude_snr_conditional",
    "weak_candidate": "detection_m1_weak_candidate",
    "weak_reason": "detection_m1_weak_reason",
    "fit_bound_hit": "detection_m1_bound_hit",
}


def _normalize_single_target_detection(
    result: xr.Dataset,
    *,
    target: object,
    stacked_dim: str,
    full_stacked_coord: xr.DataArray,
) -> xr.Dataset:
    """Keep the paired-fit record while separating it from the raw M1 fit."""
    variables: dict[str, xr.DataArray] = {}
    consumed: set[str] = set()
    for source, destination in _DETECTION_SCALAR_RENAMES.items():
        if source in result:
            variables[destination] = result[source]
            consumed.add(source)
    for source, destination in _DETECTION_MODEL_RENAMES.items():
        if source not in result:
            continue
        value = result[source]
        if "metabolite" in value.dims:
            value = value.rename({"metabolite": "model_metabolite"})
        variables[destination] = value
        consumed.add(source)
    for source in (
        "detection_null_statistic",
        "detection_null_reason",
        "detection_null_recovery_used",
    ):
        if source in result:
            variables[source] = result[source]
            consumed.add(source)
    for source in result.data_vars:
        if source in consumed:
            continue
        value = result[source]
        if "metabolite" in value.dims:
            value = value.rename({"metabolite": "model_metabolite"})
        variables[f"detection_record_{source}"] = value

    reindexed = {}
    for name, value in variables.items():
        if stacked_dim not in value.dims:
            value = value.broadcast_like(result.detection_status)
        kind = value.dtype.kind
        if kind in "iu":
            fill_value = -1
        elif kind == "b":
            fill_value = False
        elif kind in "USO":
            fill_value = (
                "not_tested"
                if name.endswith("status")
                else "none"
                if name.endswith("reason")
                else ""
            )
        else:
            fill_value = np.nan
        reindexed[name] = value.reindex(
            {stacked_dim: full_stacked_coord}, fill_value=fill_value
        )
    normalized = xr.Dataset(reindexed, attrs=dict(result.attrs))
    normalized = normalized.expand_dims(detection_target=[target])
    normalized.attrs.update(
        detection_result_schema_version="1",
        detection_record_scope=(
            "selective full-refit M0/M1 records; raw production M1 is stored in "
            "fit_amps/fit_freqs/fit_t2s and is never replaced"
        ),
    )
    return normalized


def _empty_detection_results(fit_results: xr.Dataset) -> xr.Dataset:
    template = fit_results.fit_amps
    status = xr.full_like(template.real, "not_tested", dtype=np.dtype("U32"))
    status = status.rename({"metabolite": "detection_target"})
    result = xr.Dataset(
        {
            "detection_status": status,
            "detection_p_value": xr.full_like(status, np.nan, dtype=float),
        },
        attrs={"detection_result_schema_version": "1"},
    )
    return result


def _stable_detection_stream_matrix(
    *,
    loop_dims: Sequence[str],
    loop_index,
    spectrum_count: int,
    metabolites: Sequence[object],
) -> np.ndarray:
    """Hash labelled spectrum identity and target into unique int64 streams."""
    if loop_dims:
        entries = list(loop_index)
        if len(loop_dims) == 1:
            entries = [(entry,) for entry in entries]
    else:
        entries = [()]
    if len(entries) != spectrum_count:
        raise ValueError("Could not derive stable streams from the spectrum index.")
    streams = np.empty((spectrum_count, len(metabolites)), dtype=np.int64)
    seen: set[int] = set()
    for spectrum_index, labels in enumerate(entries):
        identity = [
            {
                "dimension": dim,
                "dtype": np.asarray(label).dtype.str,
                "value": label.item() if isinstance(label, np.generic) else label,
            }
            for dim, label in sorted(zip(loop_dims, labels, strict=True))
        ]
        for target_index, target in enumerate(metabolites):
            payload = json.dumps(
                {
                    "spectrum": identity,
                    "target_dtype": np.asarray(target).dtype.str,
                    "target": target.item()
                    if isinstance(target, np.generic)
                    else target,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
            stream = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
            stream &= np.iinfo(np.int64).max
            if stream in seen:
                raise RuntimeError(
                    "Stable detection stream hash collision; rename a coordinate "
                    "or provide a distinct labelled spectrum identity."
                )
            seen.add(stream)
            streams[spectrum_index, target_index] = stream
    return streams


def run_selective_spectroscopy_detection_tests(
    data: xr.DataArray,
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    noise_reference: xr.Dataset,
    diagnostics: xr.Dataset,
    detection_results: xr.Dataset | None = None,
    optional_diagnostics: xr.Dataset | None = None,
    config: Mapping[str, float | int] | None = None,
    *,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
    show_tqdm: bool = False,
    store_null: bool = False,
) -> xr.Dataset:
    """Run full M0/M1 tests only for selected spectrum/metabolite pairs.

    Each target metabolite is dispatched through the existing, unchanged
    single-target fitter.  Stream IDs are hashes of dimension names, labelled
    spectrum coordinates and the target label, making random calibration
    invariant to input order, selection, target grouping, Joblib batch size and
    Dask chunks.
    Determining the sparse selection is necessarily eager; selected Dask FIDs
    and all refits remain on the existing lazy fitting path.

    Parameters
    ----------
    data : xr.DataArray
        Complex temporal FIDs with the spectral dimension named by
        ``fit_params.attrs["spectral_dim"]`` (default ``"acquisition_idx"``).
    fit_params : xr.Dataset
        Prepared Lorentz fit parameters.  Existing detection settings are
        removed before the M0/M1 refits.
    fit_results, noise_reference, diagnostics, detection_results, optional_diagnostics
        Inputs of :func:`classify_spectroscopy_fit_quality` used to select the
        pairs to test.
    config : mapping of str to float or int, optional
        Quality options that differ from the defaults, with the keys and ranges
        of :func:`normalize_spectroscopy_quality_options`.  ``None`` selects
        every default.
        ``detection_alpha``, ``detection_fit_snr``,
        ``detection_null_samples`` and ``detection_seed`` configure the tests.
    n_jobs, batch_size, parallel_backend, show_tqdm
        Execution options forwarded unchanged to ``fit_data_pseudo_inv``.
    store_null : bool
        Whether to keep the simulated null statistics in the result.

    Returns
    -------
    xr.Dataset
        Detection records such as ``detection_status``, ``detection_p_value``
        and ``detection_stream`` over the spectrum dimensions and
        ``detection_target``.  Pairs that were not selected have status
        ``"not_tested"``.  The ``peak_quality_config_id`` attribute holds the
        hash of the normalized options.

    Raises
    ------
    TypeError, ValueError
        If ``config`` is invalid, the spectral dimension is missing from
        ``data`` or the reserved orchestration dimension is in use.
    """
    from hypermri.analysis.fitting.spectroscopy import (
        _FIT_PARAM_MAP_NAMES,
        _align_fit_params_to_input_data,
        fit_data_pseudo_inv,
    )

    options = _resolve_quality_options(config)
    selection = select_full_detection_test_mask(
        fit_results,
        noise_reference,
        diagnostics,
        detection_results=detection_results,
        optional_diagnostics=optional_diagnostics,
        config=options,
    )
    spectral_dim = fit_params.attrs.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in data.dims:
        raise ValueError(f"Input data is missing spectral dimension '{spectral_dim}'.")
    loop_dims = [dim for dim in data.dims if dim != spectral_dim]
    stacked_dim = "__quality_spectrum"
    if stacked_dim in data.dims or stacked_dim in fit_params.dims:
        raise ValueError(f"Reserved orchestration dimension '{stacked_dim}' is in use.")

    clean_params = _strip_detection_settings(fit_params)
    aligned_params = _align_fit_params_to_input_data(
        data=data,
        fit_params=clean_params,
    )
    if loop_dims:
        stacked_input = data.transpose(*loop_dims, spectral_dim).stack(
            {stacked_dim: loop_dims}
        )
        loop_index = stacked_input.indexes[stacked_dim]
        stacked_selection = selection.transpose(*loop_dims, "metabolite").stack(
            {stacked_dim: loop_dims}
        )
        stacked_sigma = (
            _noise_sigma_for_fit(noise_reference, fit_results.fit_amps)
            .transpose(*loop_dims)
            .stack({stacked_dim: loop_dims})
        )
        stacked_maps = {
            name: aligned_params[name]
            .transpose(*loop_dims, "metabolite")
            .stack({stacked_dim: loop_dims})
            for name in _FIT_PARAM_MAP_NAMES
        }
        stacked_input = stacked_input.reset_index(stacked_dim, drop=True)
        stacked_selection = stacked_selection.reset_index(stacked_dim, drop=True)
        stacked_sigma = stacked_sigma.reset_index(stacked_dim, drop=True)
        stacked_maps = {
            name: value.reset_index(stacked_dim, drop=True)
            for name, value in stacked_maps.items()
        }
    else:
        loop_index = None
        stacked_input = data.expand_dims({stacked_dim: [0]})
        stacked_selection = selection.expand_dims({stacked_dim: [0]})
        stacked_sigma = _noise_sigma_for_fit(
            noise_reference, fit_results.fit_amps
        ).expand_dims({stacked_dim: [0]})
        stacked_maps = {
            name: aligned_params[name].expand_dims({stacked_dim: [0]})
            for name in _FIT_PARAM_MAP_NAMES
        }
    stacked_input = stacked_input.transpose(stacked_dim, spectral_dim)
    stacked_selection = stacked_selection.transpose(stacked_dim, "metabolite")
    stacked_sigma = stacked_sigma.transpose(stacked_dim)
    stacked_maps = {
        name: value.transpose(stacked_dim, "metabolite")
        for name, value in stacked_maps.items()
    }
    integer_coord = xr.IndexVariable(
        stacked_dim, np.arange(stacked_input.sizes[stacked_dim], dtype=np.int64)
    )
    stacked_input = stacked_input.assign_coords({stacked_dim: integer_coord})
    stacked_selection = stacked_selection.assign_coords({stacked_dim: integer_coord})
    stacked_sigma = stacked_sigma.assign_coords({stacked_dim: integer_coord})
    stacked_maps = {
        name: value.assign_coords({stacked_dim: integer_coord})
        for name, value in stacked_maps.items()
    }
    full_coord = integer_coord
    selection_values = np.asarray(stacked_selection.values, dtype=bool)
    target_results = []
    stream_matrix = _stable_detection_stream_matrix(
        loop_dims=loop_dims,
        loop_index=loop_index,
        spectrum_count=stacked_input.sizes[stacked_dim],
        metabolites=fit_results.metabolite.values,
    )
    for target_index, target in enumerate(fit_results.metabolite.values):
        selected_indices = np.flatnonzero(selection_values[:, target_index])
        if selected_indices.size == 0:
            continue
        selected_input = stacked_input.isel({stacked_dim: selected_indices})
        target_params = xr.Dataset(
            {
                name: value.isel({stacked_dim: selected_indices})
                for name, value in stacked_maps.items()
            },
            attrs=dict(clean_params.attrs),
        )
        selected_sigma = stacked_sigma.isel({stacked_dim: selected_indices})
        streams = stream_matrix[selected_indices, target_index]
        target_params["detection_sigma_map"] = selected_sigma
        target_params["detection_stream_id"] = xr.DataArray(
            streams,
            dims=[stacked_dim],
            coords={stacked_dim: selected_input[stacked_dim]},
        )
        target_params.attrs.update(
            detection_target=str(target),
            detection_sigma=float(np.asarray(selected_sigma.values).reshape(-1)[0]),
            detection_alpha=options["detection_alpha"],
            detection_weak_snr=options["detection_fit_snr"],
            detection_null_samples=options["detection_null_samples"],
            detection_seed=options["detection_seed"],
            detection_store_null=bool(store_null),
            snr_threshold=None,
        )
        paired = fit_data_pseudo_inv(
            selected_input,
            target_params,
            show_tqdm=show_tqdm,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
        )
        target_results.append(
            _normalize_single_target_detection(
                paired,
                target=target,
                stacked_dim=stacked_dim,
                full_stacked_coord=full_coord,
            )
        )

    if not target_results:
        return _empty_detection_results(fit_results)
    result = xr.concat(target_results, dim="detection_target", join="outer")
    reindexed_targets = {}
    for name, value in result.data_vars.items():
        kind = value.dtype.kind
        if kind in "iu":
            fill_value = -1
        elif kind == "b":
            fill_value = False
        elif kind in "USO":
            fill_value = (
                "not_tested"
                if name.endswith("status")
                else "none"
                if name.endswith("reason")
                else ""
            )
        else:
            fill_value = np.nan
        reindexed_targets[name] = value.reindex(
            detection_target=fit_results.metabolite.values,
            fill_value=fill_value,
        )
    result = xr.Dataset(reindexed_targets, attrs=dict(result.attrs))
    if loop_dims:
        result = result.assign_coords(
            xr.Coordinates.from_pandas_multiindex(loop_index, stacked_dim)
        )
        result = result.unstack(stacked_dim)
        ordered = [
            dim
            for dim in (*loop_dims, "detection_target", "model_metabolite")
            if dim in result.dims
        ]
        trailing = [dim for dim in result.dims if dim not in ordered]
        result = result.transpose(*ordered, *trailing, missing_dims="ignore")
    else:
        result = result.isel({stacked_dim: 0}, drop=True)
    result.attrs.update(
        peak_quality_config_id=_quality_options_id(options),
        detection_selection_eager=True,
        detection_stream_definition=(
            "first 63 bits of SHA-256 over sorted loop-dimension names/labels and "
            "target label"
        ),
    )
    return result


def _combine_detection_results(
    existing: xr.Dataset | None,
    new: xr.Dataset,
) -> xr.Dataset:
    """Prefer terminal existing records and fill their gaps from new tests."""
    if existing is None or "detection_status" not in existing:
        return new
    existing, new = xr.align(existing, new, join="outer", copy=False)
    terminal = existing.detection_status.fillna("not_tested") != "not_tested"
    variables = {}
    for name in set(existing.data_vars) | set(new.data_vars):
        if name in existing and name in new:
            mask = terminal
            for dim in existing[name].dims:
                if dim not in mask.dims:
                    mask = mask.expand_dims({dim: existing[name].coords[dim]})
            variables[name] = xr.where(mask, existing[name], new[name])
        elif name in existing:
            variables[name] = existing[name]
        else:
            variables[name] = new[name]
    result = xr.Dataset(variables, attrs=dict(new.attrs))
    result.attrs["existing_terminal_detection_preserved"] = True
    return result


def evaluate_spectroscopy_fit_quality(
    data: xr.DataArray,
    fit_params: xr.Dataset,
    noise_references: xr.Dataset | Sequence[xr.Dataset] | None = None,
    *,
    fit_results: xr.Dataset | None = None,
    detection_results: xr.Dataset | None = None,
    optional_diagnostics: xr.Dataset | None = None,
    config: Mapping[str, float | int] | None = None,
    run_detection: bool = True,
    n_jobs: int | None = 1,
    batch_size: int | str = "auto",
    parallel_backend: str = "loky",
    show_tqdm: bool = False,
) -> xr.Dataset:
    """Assess fitted spectroscopy peaks without deleting or changing a peak.

    This is the main function for the complete quality workflow.  Give it the
    spectroscopy data, its prepared fit settings and a checked noise reference.
    It then performs five visible steps: ordinary M1 fitting when necessary,
    noise validation, post-fit diagnostics, only the required M0/M1 tests, and
    the final Q0/Q1/Q2 classification.

    Existing fits can be supplied with ``fit_results`` to avoid fitting them
    again.  They must come from an unfiltered fit (``snr_threshold=None``).
    Missing or unsuitable noise does not raise a false scientific conclusion:
    it returns Q0 with a clear blocking status and runs no null simulations.
    Only ``lineshape='lorentz'`` fits are supported, because the diagnostics
    and M0/M1 tests use a Lorentz basis; Voigt inputs are rejected before
    any fitting.

    Parameters
    ----------
    data : xr.DataArray
        Complex temporal FIDs.
    fit_params : xr.Dataset
        Prepared Lorentz fit parameters; detection settings are ignored.
    noise_references : xr.Dataset or sequence of xr.Dataset, optional
        Checked noise references; ``None`` yields a blocking Q0 status.
    fit_results : xr.Dataset, optional
        Existing unfiltered M1 results of ``fit_params``.
    detection_results, optional_diagnostics : xr.Dataset, optional
        Earlier full-test records and extra diagnostics to reuse.
    config : mapping of str to float or int, optional
        Quality options that differ from the defaults, with the keys and ranges
        of :func:`normalize_spectroscopy_quality_options`.  ``None`` selects
        every default.
    run_detection : bool
        Whether to run the required selective M0/M1 tests.
    n_jobs, batch_size, parallel_backend, show_tqdm
        Execution options forwarded unchanged to ``fit_data_pseudo_inv``.

    Returns
    -------
    xr.Dataset
        The unchanged raw fit fields together with human-readable quality
        messages, continuous diagnostics, noise provenance and any selective
        full-test records.  ``peak_quality_class`` is the short conclusion;
        ``peak_quality_message`` explains why it was reached.

    Raises
    ------
    ValueError
        If ``fit_params`` or ``fit_results`` describes a ``lineshape='voigt'``
        fit, ``fit_results`` comes from a filtered fit, or ``config`` contains
        an unknown or invalid option.
    TypeError
        If ``config`` is not a mapping or a threshold is not a real number.

    Examples
    --------
    Change only the settings that differ from the defaults:

    >>> result = evaluate_spectroscopy_fit_quality(  # doctest: +SKIP
    ...     data,
    ...     fit_params,
    ...     noise_reference,
    ...     config={"detection_fit_snr": 8.0, "detection_null_samples": 99},
    ... )
    """
    from hypermri.analysis.fitting.spectroscopy import fit_data_pseudo_inv

    _require_lorentz_lineshape(
        "evaluate_spectroscopy_fit_quality", fit_params, fit_results
    )
    options = _resolve_quality_options(config)
    clean_params = _strip_detection_settings(fit_params)
    if fit_results is None:
        fit_results = fit_data_pseudo_inv(
            data,
            clean_params,
            show_tqdm=show_tqdm,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_backend=parallel_backend,
        )
    elif fit_results.attrs.get("snr_threshold") not in {None, 0}:
        raise ValueError(
            "Existing fit_results used for peak quality must come from an "
            "unfiltered M1 fit with snr_threshold=None."
        )

    if noise_references is None:
        noise = missing_spectroscopy_noise_reference()
    elif isinstance(noise_references, xr.Dataset):
        noise = resolve_spectroscopy_noise_reference(noise_references)
    else:
        noise = resolve_spectroscopy_noise_reference(*noise_references)
    diagnostics = calculate_spectroscopy_fit_diagnostics(
        fit_results, clean_params, noise
    )
    combined_detection = detection_results
    if run_detection:
        selected = select_full_detection_test_mask(
            fit_results,
            noise,
            diagnostics,
            detection_results=detection_results,
            optional_diagnostics=optional_diagnostics,
            config=options,
        )
        if bool(np.asarray(selected.any().values)):
            new_detection = run_selective_spectroscopy_detection_tests(
                data,
                clean_params,
                fit_results,
                noise,
                diagnostics,
                detection_results=detection_results,
                optional_diagnostics=optional_diagnostics,
                config=options,
                n_jobs=n_jobs,
                batch_size=batch_size,
                parallel_backend=parallel_backend,
                show_tqdm=show_tqdm,
            )
            combined_detection = _combine_detection_results(
                detection_results, new_detection
            )
    result = classify_spectroscopy_fit_quality(
        fit_results,
        noise,
        diagnostics,
        detection_results=combined_detection,
        optional_diagnostics=optional_diagnostics,
        config=options,
    )
    result.attrs.update(
        peak_quality_entry_point="evaluate_spectroscopy_fit_quality",
        legacy_amplitude_filter_disabled=True,
        raw_m1_preserved=True,
    )
    return result


__all__ = [
    # Everyday public workflow.
    "evaluate_spectroscopy_fit_quality",
    "make_noise_frame_reference",
    "make_outside_object_noise_reference",
    "make_rf_off_noise_reference",
    "normalize_spectroscopy_quality_options",
    # Explicit scientific steps and result vocabulary.
    "DETECTION_BASES",
    "PEAK_QUALITY_VERSION",
    "QUALITY_CLASSES",
    "QUALITY_FLAGS",
    "QUALITY_STATUS_MESSAGES",
    "QUALITY_STATUSES",
    "calculate_spectroscopy_fit_diagnostics",
    "classify_spectroscopy_fit_quality",
    "estimate_explicit_noise_reference",
    "missing_spectroscopy_noise_reference",
    "resolve_spectroscopy_noise_reference",
    "run_selective_spectroscopy_detection_tests",
    "select_full_detection_test_mask",
    # Deprecated compatibility adapter.
]
