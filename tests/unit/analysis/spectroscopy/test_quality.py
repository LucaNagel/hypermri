"""Decision-table tests for the spectroscopy peak-quality postprocessor."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy
from hypermri.analysis.spectroscopy.quality import (
    QUALITY_FLAGS,
    calculate_spectroscopy_fit_diagnostics,
    classify_spectroscopy_fit_quality,
    estimate_explicit_noise_reference,
    evaluate_spectroscopy_fit_quality,
    make_noise_frame_reference,
    make_outside_object_noise_reference,
    make_rf_off_noise_reference,
    missing_spectroscopy_noise_reference,
    normalize_spectroscopy_quality_options,
    resolve_spectroscopy_noise_reference,
    run_selective_spectroscopy_detection_tests,
    select_full_detection_test_mask,
)


def _noise(*, source="known_exact", likelihood=True):
    return estimate_explicit_noise_reference(
        source=source,
        signal_fingerprint="signal",
        reference_fingerprint="reference",
        sigma_quadrature=0.1,
        trusted=True,
        likelihood_compatible=likelihood,
        provenance={
            "origin": "unit_test",
            "unit": "a.u.",
            "quadrature_convention": "per_real_imag",
            "domain": "fitted_temporal_samples",
        },
    )


def _unsupported_noise_model():
    reference = _noise()
    reference["noise_likelihood_compatible"] = xr.DataArray(False)
    reference["noise_reference_reason"] = xr.DataArray(
        np.asarray("likelihood_model_not_supported", dtype="U128")
    )
    return reference


def _inputs(
    snr,
    *,
    condition=2.0,
    rank=2,
    frequency_bound=False,
    t2_bound=False,
    fit_status="success",
):
    snr = np.atleast_1d(np.asarray(snr, dtype=float))
    spectra = np.arange(snr.size)
    coords = {"spectrum": spectra, "metabolite": ["A", "B"]}
    fit = xr.Dataset(
        {
            "fit_amps": xr.DataArray(
                np.ones((snr.size, 2), dtype=complex),
                dims=("spectrum", "metabolite"),
                coords=coords,
            ),
            "fit_freqs": xr.DataArray(
                np.broadcast_to([0.0, 25.0], (snr.size, 2)).copy(),
                dims=("spectrum", "metabolite"),
                coords=coords,
            ),
            "fit_t2s": xr.DataArray(
                np.full((snr.size, 2), 0.025),
                dims=("spectrum", "metabolite"),
                coords=coords,
            ),
            "fit_status": xr.DataArray(
                np.full(snr.size, fit_status, dtype="U32"),
                dims="spectrum",
                coords={"spectrum": spectra},
            ),
            "fit_failure_reason": xr.DataArray(
                np.full(snr.size, "none", dtype="U32"),
                dims="spectrum",
                coords={"spectrum": spectra},
            ),
            "fit_skipped": xr.DataArray(
                np.zeros((snr.size, 2), dtype=bool),
                dims=("spectrum", "metabolite"),
                coords=coords,
            ),
            "fit_quality": xr.DataArray(
                np.ones(snr.size, dtype=bool),
                dims="spectrum",
                coords={"spectrum": spectra},
            ),
        }
    )
    condition_values = np.broadcast_to(condition, (snr.size,)).astype(float)
    rank_values = np.broadcast_to(rank, (snr.size,)).astype(np.int16)
    snr_values = np.broadcast_to(snr[:, None], (snr.size, 2)).copy()
    diagnostics = xr.Dataset(
        {
            "amplitude_snr_conditional": xr.DataArray(
                snr_values, dims=("spectrum", "metabolite"), coords=coords
            ),
            "amplitude_std_quadrature_conditional": xr.DataArray(
                1 / snr_values,
                dims=("spectrum", "metabolite"),
                coords=coords,
            ),
            "basis_condition_number": xr.DataArray(
                condition_values, dims="spectrum", coords={"spectrum": spectra}
            ),
            "basis_rank": xr.DataArray(
                rank_values, dims="spectrum", coords={"spectrum": spectra}
            ),
            "frequency_bound_hit": xr.DataArray(
                np.full((snr.size, 2), frequency_bound, dtype=bool),
                dims=("spectrum", "metabolite"),
                coords=coords,
            ),
            "t2_bound_hit": xr.DataArray(
                np.full((snr.size, 2), t2_bound, dtype=bool),
                dims=("spectrum", "metabolite"),
                coords=coords,
            ),
        }
    )
    optional = xr.Dataset(
        {
            "uncertainty_assessed": xr.full_like(fit.fit_amps.real, True, dtype=bool),
            "uncertainty_available": xr.full_like(fit.fit_amps.real, True, dtype=bool),
            "uncertainty_valid": xr.full_like(fit.fit_amps.real, True, dtype=bool),
            "stability_assessed": xr.full_like(fit.fit_amps.real, True, dtype=bool),
        }
    )
    return fit, diagnostics, optional


def _detection(fit, statuses, p_values):
    targets = fit.metabolite.values
    shape = (fit.sizes["spectrum"], len(targets))
    return xr.Dataset(
        {
            "detection_status": xr.DataArray(
                np.broadcast_to(np.asarray(statuses, dtype="U32"), shape),
                dims=("spectrum", "detection_target"),
                coords={"spectrum": fit.spectrum, "detection_target": targets},
            ),
            "detection_p_value": xr.DataArray(
                np.broadcast_to(np.asarray(p_values, dtype=float), shape),
                dims=("spectrum", "detection_target"),
                coords={"spectrum": fit.spectrum, "detection_target": targets},
            ),
        }
    )


@pytest.mark.parametrize(
    ("snr", "expected_class", "expected_status", "required"),
    [
        (9.999, "Q0_NOT_RELIABLY_DETECTED", "incomplete_detection", True),
        (10.0, "Q1_DETECTED_LIMITED", "detected_fit_snr_fast_path", False),
        (19.999, "Q1_DETECTED_LIMITED", "detected_fit_snr_fast_path", False),
        (20.0, "Q2_LIKELY_QUANTIFIABLE", "detected_fit_snr_fast_path", False),
    ],
)
def test_exact_snr_boundaries(snr, expected_class, expected_status, required):
    fit, diagnostics, optional = _inputs(snr)
    result = classify_spectroscopy_fit_quality(
        fit, _noise(), diagnostics, optional_diagnostics=optional
    )
    assert set(result.peak_quality_class.values.ravel()) == {expected_class}
    assert set(result.peak_quality_status.values.ravel()) == {expected_status}
    assert all(
        isinstance(message, str) and message.endswith(".")
        for message in result.peak_quality_message.values.ravel()
    )
    assert bool(result.full_detection_test_required.all()) is required


def test_positive_full_test_below_ten_is_q1_and_negative_overrides_fast_path():
    fit, diagnostics, optional = _inputs([7.0, 30.0])
    detection = _detection(
        fit,
        np.asarray([["detected", "detected"], ["not_detected", "not_detected"]]),
        np.asarray([[0.05, 0.025], [0.075, 0.5]]),
    )
    result = classify_spectroscopy_fit_quality(
        fit,
        _noise(),
        diagnostics,
        detection_results=detection,
        optional_diagnostics=optional,
    )
    assert set(result.peak_quality_class.sel(spectrum=0).values) == {
        "Q1_DETECTED_LIMITED"
    }
    assert set(result.peak_detection_basis.sel(spectrum=0).values) == {"full_m0_m1"}
    assert set(result.peak_quality_class.sel(spectrum=1).values) == {
        "Q0_NOT_RELIABLY_DETECTED"
    }
    assert set(result.peak_quality_status.sel(spectrum=1).values) == {
        "detection_not_significant"
    }


@pytest.mark.parametrize("status", ["inconclusive", "fit_failed"])
def test_undecidable_full_test_overrides_fast_path(status):
    fit, diagnostics, optional = _inputs(30.0)
    detection = _detection(fit, [status, status], [np.nan, np.nan])
    result = classify_spectroscopy_fit_quality(
        fit,
        _noise(),
        diagnostics,
        detection_results=detection,
        optional_diagnostics=optional,
    )
    assert set(result.peak_quality_class.values.ravel()) == {"Q0_NOT_RELIABLY_DETECTED"}
    assert set(result.peak_quality_status.values.ravel()) == {"detection_undecidable"}


@pytest.mark.parametrize(
    ("noise", "status", "flag"),
    [
        (
            missing_spectroscopy_noise_reference(),
            "noise_reference_missing",
            "NOISE_REFERENCE_MISSING",
        ),
        (
            estimate_explicit_noise_reference(
                source="fid_tail",
                signal_fingerprint="signal",
                reference_fingerprint="signal",
            ),
            "noise_reference_rejected",
            "NOISE_REFERENCE_REJECTED",
        ),
        (
            _unsupported_noise_model(),
            "noise_model_unsupported",
            "NOISE_MODEL_UNSUPPORTED",
        ),
    ],
)
def test_noise_contract_blocks_assessment(noise, status, flag):
    fit, diagnostics, optional = _inputs(30.0)
    result = classify_spectroscopy_fit_quality(
        fit, noise, diagnostics, optional_diagnostics=optional
    )
    assert set(result.peak_quality_class.values.ravel()) == {"Q0_NOT_RELIABLY_DETECTED"}
    assert set(result.peak_quality_status.values.ravel()) == {status}
    assert np.all(result.peak_quality_flags.values & QUALITY_FLAGS[flag])
    np.testing.assert_array_equal(result.fit_amps, fit.fit_amps)


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"condition": 1e8 + 1}, "ILL_CONDITIONED"),
        ({"rank": 1}, "ILL_CONDITIONED"),
        ({"fit_status": "fit_failed"}, "FIT_INVALID"),
    ],
)
def test_hard_fit_failures_are_q0(kwargs, flag):
    fit, diagnostics, optional = _inputs(30.0, **kwargs)
    result = classify_spectroscopy_fit_quality(
        fit, _noise(), diagnostics, optional_diagnostics=optional
    )
    assert set(result.peak_quality_class.values.ravel()) == {"Q0_NOT_RELIABLY_DETECTED"}
    assert np.all(result.peak_quality_flags.values & QUALITY_FLAGS[flag])


@pytest.mark.parametrize(
    ("bound_name", "flag"),
    [
        ("frequency_bound", "FREQUENCY_BOUND_HIT"),
        ("t2_bound", "T2_BOUND_HIT"),
    ],
)
def test_separate_bounds_force_full_test_and_prevent_q2(bound_name, flag):
    fit, diagnostics, optional = _inputs(30.0, **{bound_name: True})
    incomplete = classify_spectroscopy_fit_quality(
        fit, _noise(), diagnostics, optional_diagnostics=optional
    )
    assert set(incomplete.peak_quality_status.values.ravel()) == {
        "incomplete_detection"
    }
    detection = _detection(fit, ["detected", "detected"], [0.025, 0.025])
    positive = classify_spectroscopy_fit_quality(
        fit,
        _noise(),
        diagnostics,
        detection_results=detection,
        optional_diagnostics=optional,
    )
    assert set(positive.peak_quality_class.values.ravel()) == {"Q1_DETECTED_LIMITED"}
    assert np.all(positive.peak_quality_flags.values & QUALITY_FLAGS[flag])


def test_condition_warning_limits_q2_without_deleting_peak():
    fit, diagnostics, optional = _inputs(30.0, condition=20.0001)
    original = fit.copy(deep=True)
    result = classify_spectroscopy_fit_quality(
        fit, _noise(), diagnostics, optional_diagnostics=optional
    )
    assert set(result.peak_quality_class.values.ravel()) == {"Q1_DETECTED_LIMITED"}
    assert np.all(
        result.peak_quality_flags.values
        & QUALITY_FLAGS["CONDITION_OUTSIDE_VALIDATED_RANGE"]
    )
    xr.testing.assert_identical(fit, original)
    np.testing.assert_array_equal(result.fit_amps, original.fit_amps)


@pytest.mark.parametrize(
    ("variable", "flag"),
    [
        ("start_value_instability", "START_VALUE_INSTABILITY"),
        ("structured_residual_warning", "STRUCTURED_RESIDUAL_WARNING"),
        ("model_order_ambiguity", "MODEL_ORDER_AMBIGUITY"),
        ("attribution_model_dependent", "ATTRIBUTION_MODEL_DEPENDENT"),
        ("outside_validated_domain", "OUTSIDE_VALIDATED_DOMAIN"),
    ],
)
def test_optional_warnings_limit_q2(variable, flag):
    fit, diagnostics, optional = _inputs(30.0)
    optional[variable] = xr.full_like(fit.fit_amps.real, True, dtype=bool)
    result = classify_spectroscopy_fit_quality(
        fit, _noise(), diagnostics, optional_diagnostics=optional
    )
    assert set(result.peak_quality_class.values.ravel()) == {"Q1_DETECTED_LIMITED"}
    assert np.all(result.peak_quality_flags.values & QUALITY_FLAGS[flag])


def test_uncertainty_states_and_low_resolution_semantics():
    fit, diagnostics, optional = _inputs(30.0)
    absent = classify_spectroscopy_fit_quality(fit, _noise(), diagnostics)
    assert set(absent.peak_quality_class.values.ravel()) == {"Q1_DETECTED_LIMITED"}
    assert np.all(
        absent.peak_quality_flags.values & QUALITY_FLAGS["UNCERTAINTY_NOT_ASSESSED"]
    )

    optional["uncertainty_effective_rank_mass"] = xr.full_like(
        fit.fit_amps.real, 0.912, dtype=float
    )
    low_resolution = classify_spectroscopy_fit_quality(
        fit, _noise(), diagnostics, optional_diagnostics=optional
    )
    assert set(low_resolution.peak_quality_class.values.ravel()) == {
        "Q2_LIKELY_QUANTIFIABLE"
    }
    assert np.all(
        low_resolution.peak_quality_flags.values
        & QUALITY_FLAGS["UNCERTAINTY_LOW_RESOLUTION"]
    )
    assert "validated 95" not in low_resolution.attrs["uncertainty_scope"]


def test_config_ids_separate_detection_and_quantification_thresholds():
    fit, diagnostics, optional = _inputs([15.0])

    def config_id(options):
        return classify_spectroscopy_fit_quality(
            fit, _noise(), diagnostics, optional_diagnostics=optional, config=options
        ).attrs["peak_quality_config_id"]

    ids = {
        config_id(None),
        config_id({"detection_fit_snr": 11.0}),
        config_id({"quantification_fit_snr": 21.0}),
    }
    assert len(ids) == 3
    defaults = normalize_spectroscopy_quality_options()
    assert defaults["detection_null_samples"] == 39
    assert 1 / (defaults["detection_null_samples"] + 1) == 0.025


def test_rf_off_priority_fallback_and_fid_tail_rejection():
    rejected_rf = estimate_explicit_noise_reference(
        xr.DataArray(np.ones(16, dtype=complex), dims="acquisition_idx"),
        source="matched_rf_off",
        signal_fingerprint="same",
        reference_fingerprint="same",
        independent=False,
        acquisition_compatible=True,
    )
    frames = estimate_explicit_noise_reference(
        xr.DataArray(
            np.arange(24, dtype=float).reshape(3, 8) + 1j,
            dims=("frame", "acquisition_idx"),
        ),
        source="explicit_noise_frames",
        signal_fingerprint="signal",
        reference_fingerprint="frames",
        acquisition_compatible=True,
        signal_free_attested=True,
        selection=[0, 2],
        selection_dim="frame",
    )
    resolved = resolve_spectroscopy_noise_reference(frames, rejected_rf)
    assert resolved.noise_source.item() == "explicit_noise_frames"
    assert resolved.noise_fallback_used.item()
    trace = json.loads(resolved.attrs["noise_resolution_trace"])
    assert trace[0]["source"] == "matched_rf_off"
    tail = estimate_explicit_noise_reference(
        source="fid_tail",
        signal_fingerprint="signal",
        reference_fingerprint="signal",
    )
    assert tail.noise_reference_status.item() == "rejected"
    assert tail.noise_reference_reason.item() == "fid_tail_forbidden"


def test_valid_rf_off_precedes_frames_and_registered_outside_requires_attestation():
    samples = xr.DataArray(
        np.arange(24, dtype=float).reshape(3, 8) + 1j,
        dims=("frame", "acquisition_idx"),
    )
    frames = estimate_explicit_noise_reference(
        samples,
        source="explicit_noise_frames",
        signal_fingerprint="signal",
        reference_fingerprint="frames",
        acquisition_compatible=True,
        signal_free_attested=True,
        selection=[0, 2],
        selection_dim="frame",
    )
    rf_off = estimate_explicit_noise_reference(
        samples.isel(frame=0),
        source="matched_rf_off",
        signal_fingerprint="signal",
        reference_fingerprint="rf-off",
        acquisition_compatible=True,
        independent=True,
    )
    resolved = resolve_spectroscopy_noise_reference(frames, rf_off)
    assert resolved.noise_source.item() == "matched_rf_off"
    assert not resolved.noise_fallback_used.item()

    outside = estimate_explicit_noise_reference(
        samples,
        source="registered_outside_object",
        signal_fingerprint="signal",
        reference_fingerprint="outside",
        acquisition_compatible=True,
        registered=True,
        signal_free_attested=False,
        selection=[0],
        selection_dim="frame",
    )
    assert outside.noise_reference_status.item() == "rejected"
    assert outside.noise_reference_reason.item() == (
        "outside_object_not_attested_signal_free"
    )


def test_plain_language_noise_helpers_cover_the_three_preferred_sources():
    metadata = {"compatibility_id": "same-acquisition"}
    rf_off = make_rf_off_noise_reference(
        xr.DataArray(np.ones(12, dtype=complex), dims="acquisition_idx"),
        signal_fingerprint="signal",
        reference_fingerprint="rf-off",
        signal_acquisition=metadata,
        reference_acquisition=metadata,
    )
    assert rf_off.noise_reference_status.item() == "estimated"

    frames_data = xr.DataArray(
        np.arange(24, dtype=float).reshape(3, 8) + 1j,
        dims=("frame", "acquisition_idx"),
    )
    frames = make_noise_frame_reference(
        frames_data,
        frame_ids=[0, 2],
        frame_dimension="frame",
        signal_free_confirmed=True,
        signal_fingerprint="signal",
        reference_fingerprint="noise-frames",
        signal_acquisition=metadata,
        reference_acquisition=metadata,
    )
    assert frames.noise_reference_status.item() == "estimated"
    assert frames.noise_sample_count.item() == 16

    outside_data = xr.DataArray(
        np.arange(32, dtype=float).reshape(2, 2, 8) + 1j,
        dims=("read_idx", "phase_idx", "acquisition_idx"),
    )
    outside_mask = xr.DataArray(
        [[True, False], [False, True]], dims=("read_idx", "phase_idx")
    )
    outside = make_outside_object_noise_reference(
        outside_data,
        outside_object_mask=outside_mask,
        mask_dimension=None,
        registration_confirmed=True,
        signal_free_confirmed=True,
        signal_fingerprint="signal",
        reference_fingerprint="outside-object",
        signal_acquisition=metadata,
        reference_acquisition=metadata,
    )
    assert outside.noise_reference_status.item() == "estimated"
    assert outside.noise_sample_count.item() == 16


def test_outside_object_mask_accepts_curvilinear_csi_coordinates():
    samples = xr.DataArray(
        np.ones((2, 3, 8), dtype=complex),
        dims=("read_idx", "phase_idx", "acquisition_idx"),
        coords={
            "read_idx": (
                ("read_idx", "phase_idx"),
                np.broadcast_to(np.arange(2)[:, None], (2, 3)),
            ),
            "phase_idx": (
                ("read_idx", "phase_idx"),
                np.broadcast_to(np.arange(3)[None, :], (2, 3)),
            ),
        },
    )
    outside_mask = xr.DataArray(
        [[True, False, False], [False, False, True]],
        dims=("read_idx", "phase_idx"),
    )

    outside = make_outside_object_noise_reference(
        samples,
        outside_object_mask=outside_mask,
        mask_dimension=None,
        registration_confirmed=True,
        signal_free_confirmed=True,
        signal_fingerprint="signal",
        reference_fingerprint="outside-object",
        signal_acquisition={"compatibility_id": "same-acquisition"},
        reference_acquisition={"compatibility_id": "same-acquisition"},
    )

    assert outside.noise_reference_status.item() == "estimated"
    assert outside.noise_sample_count.item() == 16


def test_plain_language_workflow_is_available_at_the_domain_boundary():
    import hypermri.analysis.spectroscopy as public_spectroscopy

    for name in (
        "evaluate_spectroscopy_fit_quality",
        "make_rf_off_noise_reference",
        "make_noise_frame_reference",
        "make_outside_object_noise_reference",
        "normalize_spectroscopy_quality_options",
    ):
        assert callable(getattr(public_spectroscopy, name))


def test_automatic_outer_kspace_requires_full_conditional_contract():
    auto = xr.Dataset(
        {
            "noise_sigma_quadrature": xr.DataArray(0.1),
            "selected_point_count": xr.DataArray(12),
            "group_noise_status": xr.DataArray("estimated"),
        },
        attrs={
            "noise_source": "automatic_outer_kspace",
            "noise_status": "estimated",
            "noise_reason": "all_groups_estimated",
            "calibration_status": "validated",
            "raw_input_verified": True,
            "likelihood_compatible": True,
            "minimum_points": 12,
        },
    )
    resolved = resolve_spectroscopy_noise_reference(auto)
    assert resolved.noise_reference_status.item() == "estimated"
    assert resolved.noise_fallback_used.item()

    # The point minimum is read from the estimate and has no second default.
    without_minimum = auto.copy(deep=True)
    del without_minimum.attrs["minimum_points"]
    rejected = resolve_spectroscopy_noise_reference(without_minimum)
    assert rejected.noise_reference_status.item() == "rejected"
    assert rejected.noise_reference_reason.item() == (
        "automatic_minimum_points_missing"
    )

    unverified = auto.copy(deep=True)
    unverified.attrs["raw_input_verified"] = False
    rejected = resolve_spectroscopy_noise_reference(unverified)
    assert rejected.noise_reference_status.item() == "rejected"
    assert rejected.noise_reference_reason.item() == (
        "automatic_kspace_domain_not_verified"
    )


def test_reference_without_noise_source_is_not_assumed_outer_kspace():
    # Regression: a Dataset without ``noise_source`` used to be read as an
    # automatic outer-k-space estimate and could be accepted at priority 3.
    unlabelled = xr.Dataset(
        {
            "noise_sigma_quadrature": xr.DataArray(0.1),
            "selected_point_count": xr.DataArray(12),
            "group_noise_status": xr.DataArray("estimated"),
        },
        attrs={
            "noise_status": "estimated",
            "noise_reason": "all_groups_estimated",
            "calibration_status": "validated",
            "raw_input_verified": True,
            "likelihood_compatible": True,
        },
    )

    resolved = resolve_spectroscopy_noise_reference(unlabelled)

    assert resolved.noise_reference_status.item() == "rejected"
    assert resolved.noise_reference_reason.item() == "noise_source_missing"
    assert resolved.noise_source.item() == "unidentified"


def test_explicit_reference_rejects_automatic_outer_kspace_source():
    # Regression: the calibrated outer-k-space source used to be accepted
    # here as "validated_explicit_reference" without its calibration contract.
    reference = _noise(source="automatic_outer_kspace")

    assert reference.noise_reference_status.item() == "rejected"
    assert reference.noise_reference_reason.item() == (
        "automatic_outer_kspace_requires_estimate_csi_noise"
    )
    resolved = resolve_spectroscopy_noise_reference(reference)
    assert resolved.noise_reference_status.item() == "rejected"


def test_shared_postfit_diagnostics_match_independent_linear_algebra():
    time = np.arange(32) * 0.001
    frequencies = np.array([0.0, 25.0])
    basis = np.exp(2j * np.pi * time[:, None] * frequencies - time[:, None] / 0.025)
    amplitudes = np.array([1 + 0.2j, 0.5 - 0.1j])
    data = xr.DataArray(
        basis @ amplitudes,
        dims="acquisition_idx",
        coords={"acquisition_idx": time},
        attrs={"signal_domain": "temporal"},
    )
    from hypermri.analysis.fitting.spectroscopy import (
        def_fit_params,
        fit_data_pseudo_inv,
    )

    params = def_fit_params(
        data,
        {
            "metabs": ["A", "B"],
            "metabs_freqs_hz": frequencies,
            "initial_t2s_s": 0.025,
            "metabs_freq_ranges": 0.0,
            "metabs_t2_ranges": 0.0,
            "npoints": 1,
            "rep_fitting": 1,
            "snr_threshold": None,
        },
    )
    fit = fit_data_pseudo_inv(data, params)
    diagnostics = calculate_spectroscopy_fit_diagnostics(fit, params, _noise())
    inverse = np.linalg.pinv(basis)
    expected_std = 0.1 * np.sqrt(np.sum(np.abs(inverse) ** 2, axis=1))
    np.testing.assert_allclose(
        diagnostics.amplitude_std_quadrature_conditional, expected_std
    )
    np.testing.assert_allclose(
        diagnostics.amplitude_snr_conditional,
        np.abs(fit.fit_amps) / expected_std,
    )
    assert diagnostics.basis_rank.item() == 2
    assert not diagnostics.frequency_bound_hit.any()
    assert not diagnostics.t2_bound_hit.any()


def _orchestration_problem():
    time = np.arange(24) * 0.001
    frequencies = np.asarray([-25.0, 110.0])
    basis = np.exp(2j * np.pi * time[:, None] * frequencies - time[:, None] / 0.025)
    fids = np.stack(
        [
            basis @ np.asarray([1.0 + 0.1j, 0.12 - 0.04j]),
            basis @ np.asarray([0.08 + 0.02j, 1.1 - 0.2j]),
        ]
    )
    data = xr.DataArray(
        fids,
        dims=("spectrum", "acquisition_idx"),
        coords={"spectrum": ["left", "right"], "acquisition_idx": time},
        attrs={"signal_domain": "temporal"},
    )
    params = spectroscopy.def_fit_params(
        data,
        {
            "metabs": ["A", "B"],
            "metabs_freqs_hz": frequencies,
            "initial_t2s_s": [0.025, 0.025],
            "metabs_freq_ranges": 0,
            "metabs_t2_ranges": 0,
            "npoints": 3,
            "rep_fitting": 1,
            "snr_threshold": None,
        },
    )
    fit = spectroscopy.fit_data_pseudo_inv(data, params)
    noise = _noise()
    diagnostics = calculate_spectroscopy_fit_diagnostics(fit, params, noise)
    diagnostics["amplitude_snr_conditional"] = xr.DataArray(
        [[5.0, 20.0], [20.0, 5.0]],
        dims=("spectrum", "metabolite"),
        coords={"spectrum": data.spectrum, "metabolite": fit.metabolite},
    )
    optional = xr.Dataset(
        {
            "uncertainty_assessed": xr.full_like(fit.fit_amps.real, True, dtype=bool),
            "uncertainty_available": xr.full_like(fit.fit_amps.real, True, dtype=bool),
            "uncertainty_valid": xr.full_like(fit.fit_amps.real, True, dtype=bool),
            "stability_assessed": xr.full_like(fit.fit_amps.real, True, dtype=bool),
        }
    )
    return data, params, fit, noise, diagnostics, optional


@pytest.mark.parametrize("backend", ["serial", "joblib", "dask"])
def test_selective_orchestration_preserves_grid_and_stable_streams(backend):
    data, params, fit, noise, diagnostics, optional = _orchestration_problem()
    config = {"detection_null_samples": 1}
    kwargs = {}
    if backend == "joblib":
        kwargs["n_jobs"] = 2
    elif backend == "dask":
        pytest.importorskip("dask.array")
        data = data.chunk({"spectrum": 1, "acquisition_idx": -1})
    selected = select_full_detection_test_mask(
        fit,
        noise,
        diagnostics,
        optional_diagnostics=optional,
        config=config,
    )
    np.testing.assert_array_equal(selected, [[True, False], [False, True]])
    result = run_selective_spectroscopy_detection_tests(
        data,
        params,
        fit,
        noise,
        diagnostics,
        optional_diagnostics=optional,
        config=config,
        **kwargs,
    ).compute()
    assert result.attrs["peak_quality_config_id"] == ONE_NULL_SAMPLE_CONFIG_ID
    assert result.detection_status.dims == ("spectrum", "detection_target")
    assert (
        result.detection_status.sel(spectrum="left", detection_target="B").item()
        == "not_tested"
    )
    assert (
        result.detection_status.sel(spectrum="right", detection_target="A").item()
        == "not_tested"
    )
    left_stream = result.detection_stream.sel(
        spectrum="left", detection_target="A"
    ).item()
    right_stream = result.detection_stream.sel(
        spectrum="right", detection_target="B"
    ).item()
    assert 0 <= left_stream <= np.iinfo(np.int64).max
    assert 0 <= right_stream <= np.iinfo(np.int64).max
    assert left_stream != right_stream
    assert "detection_m0_amps" in result
    assert "detection_m1_amps" in result
    assert "detection_record_fit_search_diagnostics_freq" in result
    assert result.detection_stream.dtype == np.dtype(np.int64)
    assert "fit_amps" not in result
    if backend == "serial":
        reversed_result = run_selective_spectroscopy_detection_tests(
            data.sel(spectrum=["right", "left"]),
            params,
            fit.sel(spectrum=["right", "left"]),
            noise,
            diagnostics.sel(spectrum=["right", "left"]),
            optional_diagnostics=optional.sel(spectrum=["right", "left"]),
            config=config,
        )
        assert (
            reversed_result.detection_stream.sel(
                spectrum="left", detection_target="A"
            ).item()
            == left_stream
        )
        assert (
            reversed_result.detection_stream.sel(
                spectrum="right", detection_target="B"
            ).item()
            == right_stream
        )


def test_quality_entry_point_disables_legacy_filter_and_preserves_manual_quality():
    data, params, fit, noise, _, optional = _orchestration_problem()
    fit["fit_quality"] = xr.DataArray(
        [True, False], dims="spectrum", coords={"spectrum": data.spectrum}
    )
    params = params.assign_attrs(snr_threshold=999.0, detection_target="A")
    result = evaluate_spectroscopy_fit_quality(
        data,
        params,
        noise,
        fit_results=fit,
        optional_diagnostics=optional,
        run_detection=False,
    )
    assert result.attrs["legacy_amplitude_filter_disabled"] is True
    assert result.attrs["snr_threshold"] is None
    assert "fit_quality" in result
    assert "peak_quality_class" in result
    assert not result.fit_low_snr_refitted.any()
    assert result.attrs["raw_m1_preserved"] is True


def test_multiple_loop_dimensions_align_and_inputs_are_not_mutated():
    fit, diagnostics, optional = _inputs(20.0)
    fit = fit.isel(spectrum=0, drop=True).expand_dims(
        read_idx=[0, 1], phase_idx=["a", "b"]
    )
    diagnostics = diagnostics.isel(spectrum=0, drop=True).expand_dims(
        read_idx=[0, 1], phase_idx=["a", "b"]
    )
    optional = optional.isel(spectrum=0, drop=True).expand_dims(
        read_idx=[0, 1], phase_idx=["a", "b"]
    )
    noise = _noise()
    originals = [item.copy(deep=True) for item in (fit, diagnostics, optional, noise)]
    result = classify_spectroscopy_fit_quality(
        fit, noise, diagnostics, optional_diagnostics=optional
    )
    assert set(result.peak_quality_class.dims) == {
        "read_idx",
        "phase_idx",
        "metabolite",
    }
    assert set(result.peak_quality_class.values.ravel()) == {"Q2_LIKELY_QUANTIFIABLE"}
    for actual, expected in zip(
        (fit, diagnostics, optional, noise), originals, strict=True
    ):
        xr.testing.assert_identical(actual, expected)


def test_terminal_detection_is_not_selected_again():
    fit, diagnostics, optional = _inputs(5.0)
    detection = _detection(fit, ["not_detected", "inconclusive"], [0.5, np.nan])
    selected = select_full_detection_test_mask(
        fit,
        _noise(),
        diagnostics,
        detection_results=detection,
        optional_diagnostics=optional,
    )
    assert not selected.any()


def test_high_level_selective_detection_keeps_raw_and_paired_m1_separate():
    data, params, fit, noise, _, optional = _orchestration_problem()
    raw = fit[["fit_amps", "fit_freqs", "fit_t2s"]].copy(deep=True)
    result = evaluate_spectroscopy_fit_quality(
        data,
        params,
        noise,
        fit_results=fit,
        optional_diagnostics=optional,
        config={"detection_null_samples": 1},
    )
    for name in raw:
        xr.testing.assert_identical(result[name], raw[name])
    assert result.attrs["peak_quality_config_id"] == ONE_NULL_SAMPLE_CONFIG_ID
    assert "detection_m0_amps" in result
    assert "detection_m1_amps" in result
    assert "detection_target" in result.dims


def test_postfit_diagnostics_regress_existing_single_target_values():
    data, params, _, noise, _, _ = _orchestration_problem()
    detection_params = params.copy(deep=True)
    detection_params.attrs.update(
        detection_target="B",
        detection_sigma=0.1,
        detection_null_samples=1,
        detection_seed=71,
    )
    detected = spectroscopy.fit_data_pseudo_inv(data, detection_params)
    postfit = calculate_spectroscopy_fit_diagnostics(detected, params, noise)
    xr.testing.assert_allclose(
        postfit.basis_condition_number, detected.basis_condition_number
    )
    xr.testing.assert_allclose(
        postfit.amplitude_std_quadrature_conditional,
        detected.amplitude_std_quadrature_conditional,
    )
    xr.testing.assert_allclose(
        postfit.amplitude_snr_conditional,
        detected.amplitude_snr_conditional,
    )
    xr.testing.assert_equal(
        postfit.frequency_bound_hit | postfit.t2_bound_hit,
        detected.fit_bound_hit,
    )


# Fingerprints of the default and one changed quality setting, computed from
# the option defaults at commit 6be3ea06.  They pin the provenance contract
# across the S5 option migration; the hash is exact by construction.
DEFAULT_QUALITY_CONFIG_ID = (
    "42325aa80df72f6a1766a6442a3419544c325aeaa2679b3d35a12d07beb16e1a"
)
ONE_NULL_SAMPLE_CONFIG_ID = (
    "c587d1febe92442c2e43764922002e70f5442dbe495141a6f8db247a2c5f0b8a"
)
DEFAULT_QUALITY_CONFIG_JSON = (
    '{"detection_alpha": 0.05, "detection_fit_snr": 10.0, '
    '"detection_null_samples": 39, "detection_seed": 202609100, '
    '"force_full_detection_flags": 6, "hard_condition_max": 100000000.0, '
    '"q2_blocking_flags": 4091926, "quantification_fit_snr": 20.0, '
    '"validated_condition_warning": 20.0, "version": "1"}'
)


def test_default_quality_provenance_attributes_are_stable():
    fit, diagnostics, optional = _inputs([5.0, 15.0, 25.0])
    result = classify_spectroscopy_fit_quality(
        fit, _noise(), diagnostics, optional_diagnostics=optional
    )
    assert result.attrs["peak_quality_version"] == "1"
    assert result.attrs["peak_quality_config_id"] == DEFAULT_QUALITY_CONFIG_ID
    assert result.attrs["peak_quality_config"] == DEFAULT_QUALITY_CONFIG_JSON


@pytest.mark.parametrize(
    ("options", "classes", "flags"),
    [
        (
            {},
            [
                "Q0_NOT_RELIABLY_DETECTED",
                "Q1_DETECTED_LIMITED",
                "Q2_LIKELY_QUANTIFIABLE",
            ],
            [512, 0, 0],
        ),
        (
            {"detection_fit_snr": 16.0, "quantification_fit_snr": 26.0},
            [
                "Q0_NOT_RELIABLY_DETECTED",
                "Q0_NOT_RELIABLY_DETECTED",
                "Q1_DETECTED_LIMITED",
            ],
            [512, 512, 0],
        ),
        (
            {"validated_condition_warning": 1.5},
            ["Q0_NOT_RELIABLY_DETECTED", "Q1_DETECTED_LIMITED", "Q1_DETECTED_LIMITED"],
            [528, 16, 16],
        ),
    ],
)
def test_quality_options_select_the_classification_thresholds(options, classes, flags):
    fit, diagnostics, optional = _inputs([5.0, 15.0, 25.0])
    result = classify_spectroscopy_fit_quality(
        fit,
        _noise(),
        diagnostics,
        optional_diagnostics=optional,
        config=options,
    )
    assert result.peak_quality_class.sel(metabolite="A").values.tolist() == classes
    assert result.peak_quality_flags.sel(metabolite="A").values.tolist() == flags


def test_quality_option_defaults_are_complete_and_idempotent():
    defaults = normalize_spectroscopy_quality_options()
    assert defaults == {
        "detection_fit_snr": 10.0,
        "quantification_fit_snr": 20.0,
        "detection_alpha": 0.05,
        "hard_condition_max": 1e8,
        "validated_condition_warning": 20.0,
        "detection_null_samples": 39,
        "detection_seed": 202609100,
        "force_full_detection_flags": (
            QUALITY_FLAGS["FREQUENCY_BOUND_HIT"] | QUALITY_FLAGS["T2_BOUND_HIT"]
        ),
        "q2_blocking_flags": 4091926,
    }
    assert normalize_spectroscopy_quality_options(defaults) == defaults
    assert normalize_spectroscopy_quality_options({}) == defaults
    changed = {"detection_fit_snr": 8}
    normalized = normalize_spectroscopy_quality_options(changed)
    assert changed == {"detection_fit_snr": 8}
    assert normalized["detection_fit_snr"] == 8.0
    assert isinstance(normalized["detection_fit_snr"], float)


def test_integer_valued_quality_options_have_the_default_fingerprint():
    # Regression: the dataclass hashed the given Python type, so 10 and 10.0
    # produced different peak_quality_config_id values for the same rule.
    fit, diagnostics, optional = _inputs([15.0])
    result = classify_spectroscopy_fit_quality(
        fit,
        _noise(),
        diagnostics,
        optional_diagnostics=optional,
        config={"detection_fit_snr": 10, "detection_null_samples": 39.0},
    )
    assert result.attrs["peak_quality_config_id"] == DEFAULT_QUALITY_CONFIG_ID
    assert result.attrs["peak_quality_config"] == DEFAULT_QUALITY_CONFIG_JSON


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"detection_snr": 8.0}, ValueError, "Unknown spectroscopy quality option"),
        ({"version": "2"}, ValueError, "Unknown spectroscopy quality option"),
        ({"detection_fit_snr": True}, TypeError, "must be a real number"),
        ({"detection_fit_snr": "10"}, TypeError, "must be a real number"),
        ({"hard_condition_max": np.inf}, ValueError, "must be finite"),
        ({"detection_fit_snr": 0.0}, ValueError, "detection_fit_snr must be positive"),
        ({"quantification_fit_snr": 5.0}, ValueError, "at least detection_fit_snr"),
        ({"detection_alpha": 1.0}, ValueError, "strictly between zero and one"),
        ({"validated_condition_warning": 2e8}, ValueError, "positive and ordered"),
        ({"detection_null_samples": 0}, ValueError, "positive integer"),
        ({"detection_null_samples": 2.5}, ValueError, "positive integer"),
        ({"detection_seed": -1}, ValueError, "non-negative integer"),
        ({"q2_blocking_flags": -1}, ValueError, "non-negative integer"),
        ({"q2_blocking_flags": 1 << 31}, ValueError, "not in QUALITY_FLAGS"),
    ],
)
def test_invalid_quality_options_fail_with_the_option_name(options, error, match):
    with pytest.raises(error, match=match):
        normalize_spectroscopy_quality_options(options)


def test_quality_options_must_be_a_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        normalize_spectroscopy_quality_options([("detection_fit_snr", 8.0)])


def test_former_quality_config_class_is_removed():
    import hypermri.analysis.spectroscopy as public_spectroscopy
    import hypermri.analysis.spectroscopy.quality as quality

    assert not hasattr(quality, "SpectroscopyQualityConfig")
    assert "SpectroscopyQualityConfig" not in quality.__all__
    with pytest.raises(AttributeError):
        public_spectroscopy.SpectroscopyQualityConfig
