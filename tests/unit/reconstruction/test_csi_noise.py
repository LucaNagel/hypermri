"""Population calibration, split estimation and actual linear reconstruction."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from test_csi_raw import handler_fixture

from hypermri.analysis.spectroscopy.noise_calibration import (
    calibrate_csi_noise_selection,
    dataset_digest,
    make_csi_noise_reference,
    predict_noise_dominance,
)
from hypermri.analysis.spectroscopy.noise_estimation import (
    ACQUISITION_NOISE_MODEL,
    estimate_csi_noise,
)
from hypermri.reconstruction.csi_noise import reconstruct_csi_with_noise
from hypermri.reconstruction.csi_raw import csi_raw_fingerprint
from hypermri.reconstruction.strategies.bruker.csi import BrukerCsiRecoStrategy

COVARIANCE = np.array([[2, 0.6 + 0.3j], [0.6 - 0.3j, 3]])
POPULATION = "unit-test equal mixture: noise and amplitude-6 coherent signal"


def reference(seed, *, hidden_signal=False):
    times = np.arange(64) * 0.001
    signals = np.zeros((800, 64, 2), complex)
    signals[400:] = 6 * np.exp(2j * np.pi * 100 * times)[None, :, None]
    if hidden_signal:
        # Diagnosis looks like pure noise, but the reserved block is contaminated.
        signals[:400, 32:] = 6
    return make_csi_noise_reference(
        signals, times, COVARIANCE, seed=seed, population_id=POPULATION
    )


def calibrate(*refs, **kwargs):
    return calibrate_csi_noise_selection(
        *refs,
        diagnostic_samples=slice(0, 32),
        estimation_samples=slice(32, 64),
        contamination_tolerance=0.05,
        **kwargs,
    )


@pytest.fixture(scope="module")
def calibration():
    return calibrate(*(reference(s) for s in (1, 2, 3)))


def raw_fixture(*, shape=(12, 12), repetitions=1, slices=1, transform=None):
    handler, payload, _, _ = handler_fixture(
        shape=shape, repetitions=repetitions, slices=slices
    )
    payload = np.einsum(
        "...ct,dc->...dt", payload, np.linalg.cholesky(COVARIANCE) / np.sqrt(2)
    )
    if transform is not None:
        payload = transform(payload)
    handler.fid = payload.reshape(-1)
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid", cut_off=0)
    return handler, raw


def estimate(raw, calibration, **kwargs):
    options = dict(
        population_id=POPULATION,
        acquisition_noise_model=ACQUISITION_NOISE_MODEL,
        seed=98,
        bootstrap_samples=99,
        structure_samples=199,
    )
    options.update(kwargs)
    return estimate_csi_noise(raw, calibration, **options)


@pytest.fixture(scope="module")
def estimated(calibration):
    handler, raw = raw_fixture()
    return handler, raw, estimate(raw, calibration)


def test_calibration_uses_independent_counts_and_known_contamination(calibration):
    assert calibration.attrs["calibration_status"] == "validated"
    assert calibration.attrs["validation_selection_error_upper"] < 0.02
    assert calibration.attrs["validation_selected"] == 400
    assert np.all(np.abs(calibration.validation_relative_variance_bias) < 0.02)
    assert calibration.attrs["validation_fraction_coverage_lower"] > 0.95
    good, count = calibration.calibration_good, calibration.calibration_count
    xr.testing.assert_allclose(calibration.probability, (good + 1) / (count + 2))
    assert np.all((calibration.probability > 0) & (calibration.probability < 1))
    assert calibration.attrs["calibration_id"] == dataset_digest(calibration)


def test_holdout_hidden_contamination_invalidates_rule_without_retuning():
    model = calibrate(reference(1), reference(2), reference(3, hidden_signal=True))
    assert model.attrs["calibration_status"] == "insufficient"
    assert model.attrs["validation_selected_contaminated"] == 400
    assert np.all(model.validation_relative_variance_bias > 10)


def test_calibration_disallows_reused_randomness_and_data():
    a, b, c = (reference(s) for s in (1, 2, 3))
    with pytest.raises(ValueError, match="distinct explicit seeds"):
        calibrate(a, a, c)
    b.observed.values[0] = a.observed.values[0]
    with pytest.raises(ValueError, match="overlap"):
        calibrate(a, b, c)


def test_calibration_is_axis_order_invariant(calibration):
    actual = calibrate(
        *(
            reference(s).transpose(
                "channel_idx",
                "acquisition_idx",
                "case",
                "channel_partner",
                missing_dims="ignore",
            )
            for s in (1, 2, 3)
        )
    )
    for name in calibration.data_vars:
        xr.testing.assert_equal(actual[name], calibration[name])


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("source_hash", "old", "calibration_source_mismatch"),
        ("runtime", "old", "calibration_runtime_mismatch"),
        ("calibration_status", "insufficient", "calibration_content_changed"),
    ],
)
def test_stale_calibration_never_releases_sigma(calibration, field, value, reason):
    _, raw = raw_fixture()
    invalid = calibration.copy(deep=True)
    invalid.attrs[field] = value
    result = estimate(raw, invalid)
    assert result.attrs["noise_status"] == "insufficient"
    assert result.group_noise_reason.item() == reason
    assert np.isnan(result.noise_sigma_quadrature).all()
    assert not result.selected_mask.any()


@pytest.mark.parametrize(
    "option,value,reason",
    [
        ("population_id", "unknown", "reference_population_mismatch"),
        (
            "acquisition_noise_model",
            "temporally_correlated",
            "unsupported_acquisition_noise_model",
        ),
        ("minimum_points", 1000, "too_few_selected_points"),
    ],
)
def test_insufficient_reference_or_points(calibration, option, value, reason):
    _, raw = raw_fixture()
    result = estimate(raw, calibration, **{option: value})
    assert result.attrs["noise_status"] == "insufficient"
    assert result.group_noise_reason.item() == reason
    assert np.isnan(result.channel_covariance).all()


def test_split_estimate_retains_complex_channel_covariance(estimated):
    _, _, result = estimated
    assert result.attrs["noise_status"] == "estimated"
    assert result.selected_point_count.item() >= 95
    np.testing.assert_allclose(
        result.channel_covariance.values.squeeze(), COVARIANCE, atol=0.13
    )
    np.testing.assert_allclose(
        result.noise_sigma_quadrature.values.squeeze(),
        np.sqrt(np.diag(COVARIANCE).real / 2),
        rtol=0.03,
    )
    assert np.all(result.noise_sigma_low < result.noise_sigma_quadrature)
    assert np.all(result.noise_sigma_high > result.noise_sigma_quadrature)
    assert result.structure_p_value.item() > 0
    assert result.split_stationarity_p_value.item() > 0


def test_selection_never_reads_reserved_values_and_detects_block_drift(
    calibration, estimated
):
    _, raw, baseline = estimated
    changed = raw.copy(deep=True)
    changed.loc[{"acquisition_idx": changed.acquisition_idx.values[32:]}] *= 2
    changed.attrs["csi_raw_fingerprint"] = csi_raw_fingerprint(changed)
    actual = estimate(changed, calibration)
    xr.testing.assert_equal(actual.selected_mask, baseline.selected_mask)
    xr.testing.assert_equal(actual.p_noise_dominated, baseline.p_noise_dominated)
    assert actual.group_noise_reason.item() == "acquisition_noise_model_rejected"
    assert actual.split_stationarity_p_value.item() == 0.005
    assert np.isnan(actual.noise_sigma_quadrature).all()


def test_estimation_does_not_use_unselected_points(calibration, estimated):
    _, raw, baseline = estimated
    changed = raw.copy(deep=True)
    mask = (~baseline.selected_mask).broadcast_like(changed).values
    block = (
        xr.DataArray(np.arange(64) >= 32, dims="acquisition_idx")
        .broadcast_like(changed)
        .values
    )
    changed.values[mask & block] *= 100
    changed.attrs["csi_raw_fingerprint"] = csi_raw_fingerprint(changed)
    actual = estimate(changed, calibration)
    xr.testing.assert_equal(
        actual.channel_covariance_bootstrap, baseline.channel_covariance_bootstrap
    )


@pytest.mark.parametrize("corruption", ["correlated", "improper", "nonfinite"])
def test_reserved_noise_model_failures_are_explicit(calibration, corruption):
    def transform(payload):
        if corruption == "correlated":
            for i in range(33, 64):
                payload[..., i] = (
                    0.95 * payload[..., i - 1] + np.sqrt(1 - 0.95**2) * payload[..., i]
                )
        elif corruption == "improper":
            payload[..., 32:] = np.sqrt(2) * payload[..., 32:].real
        else:
            payload[..., 40] = np.nan
        return payload

    _, raw = raw_fixture(transform=transform)
    result = estimate(raw, calibration)
    assert result.attrs["noise_status"] == "insufficient"
    assert result.group_noise_reason.item() == (
        "nonfinite_estimation_samples"
        if corruption == "nonfinite"
        else "acquisition_noise_model_rejected"
    )


def test_global_scaling_and_group_estimates(calibration, estimated):
    _, raw, baseline = estimated
    scaled = raw.copy(data=raw.values * 3)
    scaled.attrs["csi_raw_fingerprint"] = csi_raw_fingerprint(scaled)
    actual = estimate(scaled, calibration)
    np.testing.assert_allclose(
        actual.noise_sigma_quadrature, baseline.noise_sigma_quadrature * 3
    )

    def scale_groups(payload):
        payload[1] *= 2
        return payload

    _, grouped = raw_fixture(repetitions=2, slices=2, transform=scale_groups)
    result = estimate(grouped, calibration)
    assert result.attrs["noise_status"] == "estimated"
    assert result.channel_covariance.dims == (
        "repetition_idx",
        "slice_idx",
        "channel_idx",
        "channel_partner",
    )
    np.testing.assert_allclose(
        result.noise_sigma_quadrature.isel(repetition_idx=1)
        / result.noise_sigma_quadrature.isel(repetition_idx=0),
        2,
        rtol=0.05,
    )


def test_dask_and_axis_order_keep_selection_and_rng(calibration, estimated):
    _, raw, baseline = estimated
    actual = estimate(
        raw.transpose(*reversed(raw.dims)).chunk({"read_idx": 3}), calibration
    )
    xr.testing.assert_equal(
        actual.channel_covariance_bootstrap, baseline.channel_covariance_bootstrap
    )
    xr.testing.assert_equal(actual.selected_mask, baseline.selected_mask)


def test_strong_outer_signal_is_rejected(calibration):
    def add_signal(payload):
        payload[:, :, :12] += 6 * np.exp(2j * np.pi * 100 * np.arange(64) * 0.001)
        return payload

    _, raw = raw_fixture(transform=add_signal)
    result = estimate(raw, calibration)
    assert result.attrs["noise_status"] == "estimated"
    assert not result.selected_mask.isel(phase_idx=0).any()
    assert result.selected_point_count.item() < 100


def test_actual_bruker_reconstruction_preserves_phase_and_precision(
    estimated, monkeypatch
):
    handler, raw, noise = estimated
    handler.read_offset_mm = 0.7
    handler.phase_offset_mm = -0.3
    handler.type = "CSI"
    strategy = BrukerCsiRecoStrategy()
    legacy = strategy._sorted_csi_to_data(raw, handler, data_precision=np.float64)
    from hypermri.reconstruction.strategies.bruker import csi

    original_shift = csi.shift_image
    calls = []

    def shift_once(**kwargs):
        calls.append(kwargs["data"])
        return original_shift(**kwargs)

    def redundant_fft(*args, **kwargs):
        pytest.fail("The covariance path must not also run the legacy signal FFT.")

    monkeypatch.setattr(csi, "shift_image", shift_once)
    monkeypatch.setattr(csi, "centered_ifft", redundant_fft)
    propagated, bundle = strategy._reconstruct_sorted_csi(
        raw, handler, noise_estimate=noise
    )
    assert len(calls) == 1
    np.testing.assert_allclose(propagated.values, legacy.values, atol=1e-13)
    assert propagated.dims == legacy.dims
    xr.testing.assert_identical(
        propagated.coords.to_dataset(), legacy.coords.to_dataset()
    )
    for name, value in legacy.attrs.items():
        np.testing.assert_equal(propagated.attrs[name], value)
    assert propagated.dtype == np.complex128
    assert propagated.attrs["csi_noise_id"] == noise.attrs["noise_id"]
    assert not hasattr(handler, "csi_noise_reconstruction")
    np.testing.assert_allclose(
        bundle.noise_sigma_quadrature.isel(read_idx=0, phase_idx=0, acquisition_idx=0),
        noise.noise_sigma_quadrature / 12,
    )
    public, public_noise = strategy.reconstruct_with_noise(
        handler,
        data_type="fid",
        cut_off=0,
        append_parameters=False,
        noise_estimate=noise,
    )
    np.testing.assert_allclose(public.values, legacy.values, atol=1e-13)
    np.testing.assert_allclose(public_noise.signal, bundle.signal, atol=1e-13)
    assert not hasattr(handler, "csi_noise_reconstruction")
    assert len(calls) == 2  # One phase operation for each explicit reconstruction.


def test_csi_reconstruction_dependencies_point_toward_data():
    """Reconstruction must consume data without importing downstream analysis."""
    import hypermri.reconstruction.csi_noise as reconstruction
    import hypermri.reconstruction.csi_raw as raw

    for module in (reconstruction, raw):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "analysis" not in (node.module or "").split(".")
                assert all(name.name != "*" for name in node.names)
            elif isinstance(node, ast.Import):
                assert all(
                    "analysis" not in name.name.split(".") for name in node.names
                )


def test_reconstruct_rejects_the_removed_noise_option(estimated):
    """Propagated noise is only returned by ``reconstruct_with_noise``."""
    handler, _, estimate = estimated
    handler.type = "CSI"
    strategy = BrukerCsiRecoStrategy()

    assert "noise_estimate" not in BrukerCsiRecoStrategy.accepted_options
    with pytest.raises(ValueError, match="reconstruct_with_noise"):
        strategy.reconstruct(
            handler,
            data_type="fid",
            cut_off=0,
            noise_estimate=estimate,
            append_parameters=False,
        )
    assert not hasattr(handler, "csi_noise_reconstruction")


def test_noise_public_functions_keep_one_implementation():
    from hypermri.analysis import spectroscopy
    from hypermri.analysis.spectroscopy import (
        noise,
        noise_calibration,
        noise_estimation,
    )
    from hypermri.reconstruction import csi_noise

    assert (
        spectroscopy.inspect_csi_noise_candidates is noise.inspect_csi_noise_candidates
    )
    assert (
        spectroscopy.make_csi_noise_reference
        is noise_calibration.make_csi_noise_reference
    )
    assert (
        spectroscopy.calibrate_csi_noise_selection
        is noise_calibration.calibrate_csi_noise_selection
    )
    assert spectroscopy.estimate_csi_noise is noise_estimation.estimate_csi_noise
    assert spectroscopy.ACQUISITION_NOISE_MODEL == csi_noise.ACQUISITION_NOISE_MODEL


@pytest.mark.parametrize("shape,output", [((5, 7), (5, 7)), ((4, 5), (7, 8))])
def test_fft_impulses_padding_and_missing_acquisitions(calibration, shape, output):
    _, raw = raw_fixture(shape=shape)
    # A missing point is zero-filled for signal, but never counted as noise.
    raw = raw.copy(deep=True)
    raw[{"read_idx": 0, "phase_idx": 0}] = np.nan
    raw.csi_acquisition_count.values[0, 0] = 0
    raw.csi_acquired.values[0, 0] = False
    raw.attrs["csi_raw_fingerprint"] = csi_raw_fingerprint(raw)
    noise = estimate(raw, calibration, minimum_points=8)
    result = reconstruct_csi_with_noise(raw, noise, output_shape=output)
    operator = result.spatial_operator.values
    basis = np.eye(np.prod(shape)).reshape(*shape, -1)
    padded = np.zeros((*output, np.prod(shape)), complex)
    start = [out // 2 - size // 2 for out, size in zip(output, shape, strict=True)]
    padded[start[0] : start[0] + shape[0], start[1] : start[1] + shape[1]] = basis
    # The FOV centre lies at index n / 2 of each output axis.
    for axis, n in enumerate(output):
        if n % 2:
            ramp = np.exp(-1j * np.pi * (np.arange(n) - n // 2) / n)
            padded = padded * np.expand_dims(ramp, [i for i in range(3) if i != axis])
    expected = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(padded, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
    ).reshape(-1, np.prod(shape))
    np.testing.assert_allclose(operator, expected, atol=1e-15)
    assert np.all(result.spatial_noise_factor.values[:, 0] == 0)
    expected_scale = np.sqrt(np.prod(shape) - 1) / np.prod(output)
    np.testing.assert_allclose(
        result.noise_sigma_quadrature.isel(read_idx=0, phase_idx=0, acquisition_idx=0),
        noise.noise_sigma_quadrature * expected_scale,
    )


def test_complex_windows_channels_time_and_full_covariance_monte_carlo(estimated):
    _, raw, noise = estimated
    spatial = np.outer(np.hanning(12), np.hanning(12)) * np.exp(0.4j * np.arange(12))
    temporal = np.zeros((2, 64), complex)
    temporal[0, :2], temporal[1, :2] = [1, 0.5j], [0.3, 0.8]
    channels = np.array([[1, 0.4j], [0.2j, -0.7]])
    result = reconstruct_csi_with_noise(
        raw,
        noise,
        spatial_weights=spatial,
        temporal_operator=temporal,
        channel_weights=channels,
    )
    f = result.spatial_noise_factor.values[[0, 1]]
    s = f @ f.conj().T
    t = temporal @ temporal.conj().T
    c = result.channel_covariance.values.squeeze()
    expected = np.kron(np.kron(s, t), c)
    rng = np.random.default_rng(77)
    # 4096 independent noise draws, no metabolite refits or large validation run.
    white = (
        rng.standard_normal((4096, 144, 2, 2))
        + 1j * rng.standard_normal((4096, 144, 2, 2))
    ) / np.sqrt(2)
    samples = white @ np.linalg.cholesky(noise.channel_covariance.values.squeeze()).T
    observed = np.einsum(
        "vk,nktc,st,oc->nvso", f, samples, temporal[:, :2], channels, optimize=True
    ).reshape(4096, -1)
    actual = observed.T @ observed.conj() / len(observed)
    np.testing.assert_allclose(
        actual, expected, atol=np.max(np.diag(expected).real) * 0.065
    )
    marginal = result.noise_sigma_quadrature.values.squeeze()[:1, :2].reshape(-1)
    np.testing.assert_allclose(
        marginal, np.sqrt(np.diag(expected).real / 2), atol=1e-15
    )
    assert np.max(np.abs(t - np.diag(np.diag(t)))) > 0
    assert result.attrs["reconstruction_id"] == dataset_digest(result)


@pytest.mark.parametrize("kind", ["raw", "noise", "source", "insufficient"])
def test_propagation_rejects_changed_or_insufficient_input(estimated, kind):
    _, raw, noise = estimated
    raw, noise = raw.copy(deep=True), noise.copy(deep=True)
    if kind == "raw":
        raw.values.flat[0] *= 2
    elif kind == "noise":
        noise.channel_covariance.values.flat[0] *= 2
    elif kind == "source":
        noise.attrs["source_hash"] = "old"
    else:
        noise.attrs["noise_status"] = "insufficient"
        noise.attrs["noise_id"] = dataset_digest(noise)
    with pytest.raises(ValueError):
        reconstruct_csi_with_noise(raw, noise)


@pytest.mark.parametrize(
    "options",
    [
        dict(output_shape=(2, 2)),
        dict(channel_weights="RSS"),
        dict(temporal_operator=np.ones((2, 8))),
        dict(spatial_weights=np.nan),
    ],
)
def test_unsupported_reconstruction_operators(estimated, options):
    _, raw, noise = estimated
    with pytest.raises(ValueError):
        reconstruct_csi_with_noise(raw, noise, **options)


def test_no_implicit_detector_acceptance(estimated):
    _, raw, noise = estimated
    result = reconstruct_csi_with_noise(raw, noise)
    assert "detection_p_value" not in result
    assert "accepted" not in result
    assert "white-noise likelihood" in result.attrs["likelihood_scope"]
    assert json.loads(result.attrs["input_shape"]) == [12, 12]


def test_outer_kspace_estimate_names_its_source(estimated):
    from hypermri.analysis.spectroscopy.quality import (
        resolve_spectroscopy_noise_reference,
    )

    _, _, noise = estimated
    assert noise.attrs["noise_source"] == "automatic_outer_kspace"

    resolved = resolve_spectroscopy_noise_reference(noise)
    assert resolved.noise_source.item() == "automatic_outer_kspace"
    assert resolved.noise_reference_status.item() == "estimated"
    # Raw k-space sigma is not on the fitted temporal sample scale.
    assert not resolved.noise_likelihood_compatible.item()


@pytest.mark.parametrize(
    "shape,minimum_points,count,status,reason",
    [
        # 10 selected points satisfy minimum_points=4; the old fixed floor of
        # 12 in the quality check rejected them as too few.
        ((3, 4), 4, 10, "estimated", "all_groups_estimated"),
        # 108 points fail minimum_points=200; the old floor passed them and
        # reported a structure failure instead.
        ((12, 12), 200, 108, "insufficient", "automatic_too_few_selected_points"),
    ],
)
def test_quality_check_uses_the_minimum_points_of_the_estimate(
    calibration, shape, minimum_points, count, status, reason
):
    """Regression: ``estimate_csi_noise`` owns the point minimum."""
    from hypermri.analysis.spectroscopy.quality import (
        resolve_spectroscopy_noise_reference,
    )

    _, raw = raw_fixture(shape=shape)
    noise = estimate(raw, calibration, minimum_points=minimum_points)
    assert noise.attrs["minimum_points"] == minimum_points
    assert int(noise.selected_point_count.min()) == count

    resolved = resolve_spectroscopy_noise_reference(noise)

    assert resolved.noise_reference_status.item() == status
    assert resolved.noise_reference_reason.item() == reason
